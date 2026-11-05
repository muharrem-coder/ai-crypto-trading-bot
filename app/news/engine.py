"""Async news ingestion engine: poll sources, run the pipeline, publish ``NewsEvent`` on ``Topic.NEWS``.

Pipeline per item (``NewsEngine.ingest``): age filter -> dedup -> relevance -> sentiment/importance ->
cache -> persist -> publish. Only newly published events are returned; duplicates just enrich the original's
``duplicate_sources`` in the cache. All state is bounded (see Deduplicator / NewsCache).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.config.settings import NewsSettings
from app.core.events import EventBus, Topic
from app.core.types import utcnow
from app.news import sentiment
from app.news.cache import NewsCache
from app.news.contracts import NewsEvent
from app.news.dedup import Deduplicator
from app.news.persistence import NewsPersistence
from app.news.relevance import RelevanceFilter
from app.news.sources import (
    CryptoPanicSource,
    NewsSource,
    RawNewsItem,
    RSSSource,
    XSource,
    redact_text,
)

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 300.0
SUMMARY_CHARS = 500
_FUTURE_TOLERANCE = timedelta(minutes=5)
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD", "PERP")


@dataclass(slots=True)
class NewsStats:
    fetched: int = 0
    published: int = 0
    duplicates: int = 0
    irrelevant: int = 0
    too_old: int = 0
    invalid: int = 0
    persist_errors: int = 0
    source_errors: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetched": self.fetched, "published": self.published, "duplicates": self.duplicates,
            "irrelevant": self.irrelevant, "too_old": self.too_old, "invalid": self.invalid,
            "persist_errors": self.persist_errors, "source_errors": dict(self.source_errors),
        }


def base_asset(symbol: str) -> str:
    """"BTC/USDT:USDT" -> "BTC"; "ETHUSDT" -> "ETH"; "sol" -> "SOL"."""
    base = symbol.strip().upper().split(":")[0].split("/")[0]
    for suffix in _QUOTE_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            return base[: -len(suffix)]
    return base


def build_default_sources(settings: Any) -> list[NewsSource]:
    """RSS feeds (+ CryptoPanic / X when their tokens are configured). Accepts NewsSettings or Settings."""
    news: NewsSettings = getattr(settings, "news", settings)
    sources: list[NewsSource] = []
    for key, url in news.rss_feeds.items():
        name = key if key in ("sec", "fed") else f"rss:{key}"
        sources.append(RSSSource(name, url))
    cp_token = news.cryptopanic_token.get_secret_value()
    if cp_token:
        sources.append(CryptoPanicSource(cp_token, list(news.symbol_keywords.keys())))
    x_token = news.x_bearer_token.get_secret_value()
    if x_token:
        sources.append(XSource(x_token, list(news.x_accounts)))
    return sources


def _default_client_factory(settings: NewsSettings) -> Callable[[], httpx.AsyncClient]:
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers={"User-Agent": "quant-core-news/1.0"},
            follow_redirects=True,
        )

    return factory


class NewsEngine:
    def __init__(
        self,
        sources: Iterable[NewsSource],
        bus: EventBus,
        settings: Any,
        *,
        persistence: NewsPersistence | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings: NewsSettings = getattr(settings, "news", settings)
        self.sources: list[NewsSource] = list(sources)
        self.bus = bus
        self.persistence = persistence
        self._clock = clock
        self._sleep = sleep
        self._client_factory = client_factory or _default_client_factory(self.settings)
        self.dedup = Deduplicator(self.settings, clock)
        self.relevance = RelevanceFilter(self.settings.symbol_keywords, self.settings.macro_keywords)
        self.cache = NewsCache(self.settings.cache_max_items, self.settings.cache_ttl_hours, clock)
        self.stats = NewsStats()
        self._tasks: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        self._running = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._warm_start()
        self._client = self._client_factory()
        for source in self.sources:
            self.stats.source_errors.setdefault(source.name, 0)
            self._tasks.append(asyncio.create_task(self._poll_loop(source), name=f"news-poll-{source.name}"))
        logger.info("NewsEngine started with %d sources", len(self.sources))

    async def stop(self) -> None:
        self._running = False
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("NewsEngine: error closing HTTP client", exc_info=False)

    async def _warm_start(self) -> None:
        if self.persistence is None:
            return
        try:
            events = await self.persistence.load_recent(self.settings.cache_ttl_hours, limit=self.settings.cache_max_items)
        except Exception as exc:  # noqa: BLE001 - warm start is best effort
            self.stats.persist_errors += 1
            logger.warning("NewsEngine: warm start failed: %s", redact_text(f"{type(exc).__name__}: {exc}"))
            return
        for event in events:
            self.cache.add(event)
            self.dedup.seed(event.news_id, event.url, event.title, event.published_at, event.ingested_at, event.source)
        logger.info("NewsEngine: warmed %d events from persistence", len(events))

    # ------------------------------------------------------------------ polling
    def _interval(self, source: NewsSource) -> float:
        return max(float(self.settings.poll_interval_seconds), float(getattr(source, "min_interval", 0.0)))

    async def _poll_loop(self, source: NewsSource) -> None:
        interval = self._interval(source)
        failures = 0
        while self._running:
            try:
                assert self._client is not None
                timeout = max(30.0, float(self.settings.request_timeout_seconds) * 3)
                items = await asyncio.wait_for(source.fetch(self._client), timeout=timeout)
                await self.ingest(items)
                failures = 0
                delay = interval
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad source must never kill the others
                failures += 1
                self.stats.source_errors[source.name] = self.stats.source_errors.get(source.name, 0) + 1
                delay = min(MAX_BACKOFF_SECONDS, interval * (2 ** (failures - 1)))
                logger.warning(
                    "news source %s failed (%d consecutive): %s; retry in %.0fs",
                    source.name, failures, redact_text(f"{type(exc).__name__}: {exc}"), delay,
                )
            await self._sleep(delay)

    # ------------------------------------------------------------------ pipeline
    async def ingest(self, raw_items: Iterable[RawNewsItem]) -> list[NewsEvent]:
        """Run the full pipeline; returns the events newly published on ``Topic.NEWS``."""
        published: list[NewsEvent] = []
        for raw in raw_items:
            self.stats.fetched += 1
            try:
                event = await self._process(raw)
            except Exception as exc:  # noqa: BLE001 - a malformed item must not drop the batch
                self.stats.invalid += 1
                logger.warning("news item from %s rejected: %s", getattr(raw, "source", "?"),
                               redact_text(f"{type(exc).__name__}: {exc}"))
                continue
            if event is not None:
                published.append(event)
        return published

    async def _process(self, raw: RawNewsItem) -> NewsEvent | None:
        if not isinstance(raw.title, str) or not raw.title.strip():
            raise ValueError("empty title")
        now = self._clock()
        published_at = raw.published_at
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        published_at = published_at.astimezone(timezone.utc)
        if now - published_at > timedelta(hours=float(self.settings.max_age_hours)):
            self.stats.too_old += 1
            return None
        if published_at > now + _FUTURE_TOLERANCE:
            published_at = now
        if published_at is not raw.published_at:
            raw = RawNewsItem(raw.source, raw.external_id, raw.title, raw.url, published_at, raw.body, raw.extra)

        result = self.dedup.check(raw)
        if result.is_duplicate:
            self.stats.duplicates += 1
            if result.original_id:
                self.cache.mark_duplicate(result.original_id, raw.source)
                if self.persistence is not None and hasattr(self.persistence, "add_duplicate_source"):
                    try:
                        await self.persistence.add_duplicate_source(result.original_id, raw.source)
                    except Exception:  # noqa: BLE001 - persistence must never stop ingestion
                        self.stats.persist_errors += 1
                        logger.exception("failed to record duplicate source %s", raw.source)
            return None

        match = self.relevance.match(raw.title, raw.body)
        if not match.relevant:
            self.stats.irrelevant += 1
            return None

        score, label = sentiment.score(raw.title, raw.body)
        summary = (raw.body or "").strip()[:SUMMARY_CHARS] or None
        event = NewsEvent(
            news_id=result.news_id,
            source=raw.source,
            external_id=raw.external_id,
            title=raw.title.strip(),
            url=raw.url,
            published_at=published_at,
            ingested_at=now,
            symbols=match.symbols,
            macro_tags=match.macro_tags,
            sentiment_score=score,
            sentiment_label=label,
            importance=sentiment.importance(raw.source, match.macro_tags, match.symbols, score),
            summary=summary,
        )
        self.cache.add(event)
        await self._persist(event)
        self.bus.publish(Topic.NEWS, event)
        self.stats.published += 1
        return event

    async def _persist(self, event: NewsEvent) -> None:
        if self.persistence is None:
            return
        try:
            await self.persistence.save(event)
        except Exception as exc:  # noqa: BLE001 - persistence must never stop ingestion
            self.stats.persist_errors += 1
            logger.warning("news persistence failed: %s", redact_text(f"{type(exc).__name__}: {exc}"))

    # ------------------------------------------------------------------ queries
    def recent_for(self, symbol: str, since: datetime, limit: int = 10) -> list[NewsEvent]:
        """Newest-first news for a unified symbol ("BTC/USDT:USDT" -> BTC) plus macro-tagged news."""
        return self.cache.recent([base_asset(symbol)], since, limit, include_macro=True)
