"""NewsEngine pipeline tests: dedup, relevance, age filter, memory caps, persistence, backoff."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio

from app.config.settings import NewsSettings
from app.core.events import EventBus, Topic
from app.db.session import async_session_factory, create_engine_from_url, init_db
from app.news.contracts import NewsEvent
from app.news.engine import NewsEngine, base_asset, build_default_sources
from app.news.persistence import NewsPersistence
from app.news.sources import CryptoPanicSource, RawNewsItem, RSSSource, XSource

pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def clock() -> datetime:
    return NOW


def raw(source: str, title: str, url: str | None, minutes_ago: float = 10, ext: str | None = None, body: str | None = None):
    return RawNewsItem(
        source=source, external_id=ext or f"{source}-{abs(hash(title)) % 10**8}", title=title, url=url,
        published_at=NOW - timedelta(minutes=minutes_ago), body=body,
    )


def make_engine(settings: NewsSettings | None = None, **kw) -> tuple[NewsEngine, EventBus]:
    bus = EventBus()
    engine = NewsEngine(kw.pop("sources", []), bus, settings or NewsSettings(), clock=clock, **kw)
    return engine, bus


STORY = "Bitcoin ETF sees record $1B inflow"


def story_items() -> list[RawNewsItem]:
    return [
        raw("rss:coindesk", f"{STORY} - CoinDesk",
            "https://www.coindesk.com/markets/2026/11/05/bitcoin-etf-record-inflow/?utm_source=rss&utm_medium=feed#c", 60, "cd-1"),
        raw("cryptopanic", STORY, "https://cryptopanic.com/news/9999/click/", 50, "9999"),
        raw("x:WatcherGuru", f"{STORY} https://t.co/abc", "https://x.com/WatcherGuru/status/1", 45, "1"),
    ]


async def test_same_story_from_three_sources_published_once():
    engine, bus = make_engine()
    sub = bus.subscribe(Topic.NEWS)
    published = await engine.ingest(story_items())
    assert len(published) == 1
    event = published[0]
    assert event.source == "rss:coindesk" and "BTC" in event.symbols and "ETF" in event.macro_tags
    assert event.sentiment_label == "BULLISH" and 0 <= event.importance <= 10
    assert len(sub.drain()) == 1 and bus.published[Topic.NEWS] == 1
    cached = engine.cache.get(event.news_id)
    assert cached.duplicate_sources == ("cryptopanic", "x:WatcherGuru")
    assert engine.stats.published == 1 and engine.stats.duplicates == 2 and engine.stats.fetched == 3
    # arriving later in separate batches / repeated polls does not republish either
    assert await engine.ingest(story_items()) == []
    assert engine.cache.get(event.news_id).duplicate_sources == ("cryptopanic", "x:WatcherGuru")
    assert len(sub.drain()) == 0


async def test_same_canonical_url_and_amp_variant_deduped():
    engine, _ = make_engine()
    out = await engine.ingest([
        raw("rss:a", "Ethereum upgrade goes live", "https://site.com/eth-upgrade/"),
        raw("rss:b", "Totally different wording about ETH", "https://www.site.com/eth-upgrade/amp/?utm_source=x"),
        raw("rss:c", "Another wording for ether news", "https://site.com/eth-upgrade?outputType=amp"),
    ])
    assert len(out) == 1 and engine.stats.duplicates == 2


async def test_irrelevant_dropped_and_false_positives():
    engine, bus = make_engine()
    sub = bus.subscribe(Topic.NEWS)
    out = await engine.ingest([
        raw("rss:a", "Local bakery wins award for best croissant", "https://a.com/1"),
        raw("rss:a", "Our solution is a better method, wait a second", "https://a.com/2"),
        raw("rss:a", "Hackathon winners in ethics category", "https://a.com/3"),
    ])
    assert out == [] and sub.drain() == []
    assert engine.stats.irrelevant == 3 and engine.stats.published == 0


async def test_macro_only_news_is_kept_with_fed_tag():
    engine, _ = make_engine()
    out = await engine.ingest([raw("fed", "FOMC minutes signal slower rate cuts", "https://federalreserve.gov/m1")])
    assert len(out) == 1
    ev = out[0]
    assert "FED" in ev.macro_tags and ev.symbols == () and ev.importance >= 7


async def test_too_old_dropped_and_future_clamped():
    engine, _ = make_engine()
    out = await engine.ingest([
        raw("rss:a", "Bitcoin hits new high", "https://a.com/old", minutes_ago=13 * 60),
        raw("rss:a", "Bitcoin hits another high", "https://a.com/future", minutes_ago=-120),
    ])
    assert engine.stats.too_old == 1 and len(out) == 1
    assert out[0].published_at == NOW


async def test_malformed_item_counted_not_fatal():
    engine, _ = make_engine()
    bad = RawNewsItem("rss:a", "1", None, None, NOW)  # type: ignore[arg-type]
    out = await engine.ingest([bad, raw("rss:a", "Bitcoin rallies", "https://a.com/ok")])
    assert len(out) == 1 and engine.stats.invalid == 1


async def test_memory_stays_capped_after_50k_inserts():
    settings = NewsSettings(cache_max_items=100, dedup_window=1000)
    engine, bus = make_engine(settings)
    bus_sub = bus.subscribe(Topic.NEWS, maxsize=10)
    for start in range(0, 50_000, 500):
        batch = [raw("rss:a", f"Bitcoin update {i} headline", f"https://a.com/s/{i}") for i in range(start, start + 500)]
        await engine.ingest(batch)
    assert engine.stats.published == 50_000
    assert len(engine.cache) == 100
    assert engine.dedup.seen_size <= 1000 and engine.dedup.title_window_size <= 1000
    assert bus_sub.queue.qsize() <= 10


def test_base_asset_and_recent_for():
    assert base_asset("BTC/USDT:USDT") == "BTC" and base_asset("eth/usdt") == "ETH"
    assert base_asset("SOLUSDT") == "SOL" and base_asset("AAPL") == "AAPL"


async def test_recent_for_unified_symbol_includes_macro():
    engine, _ = make_engine()
    await engine.ingest([
        raw("rss:a", "Solana network congestion returns", "https://a.com/sol", 30),
        raw("rss:a", "Bitcoin whales accumulate", "https://a.com/btc", 20),
        raw("fed", "FOMC holds rates steady", "https://fed.gov/1", 10),
    ])
    got = engine.recent_for("BTC/USDT:USDT", NOW - timedelta(hours=1), 10)
    assert [e.url for e in got] == ["https://fed.gov/1", "https://a.com/btc"]
    assert engine.recent_for("BTC/USDT:USDT", NOW - timedelta(minutes=5), 10) == []
    assert len(engine.recent_for("SOL/USDT:USDT", NOW - timedelta(hours=1), 1)) == 1


# ------------------------------------------------------------------------------ persistence
@pytest_asyncio.fixture
async def session_factory():
    engine = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    yield async_session_factory(engine)
    await engine.dispose()


def _event(**over) -> NewsEvent:
    from decimal import Decimal

    base = dict(
        news_id="n1", source="rss:coindesk", external_id="ext-1", title="Bitcoin ETF approved", url="https://a.com/x",
        published_at=NOW - timedelta(hours=1), ingested_at=NOW, symbols=("BTC",), macro_tags=("ETF",),
        sentiment_score=Decimal("0.43"), sentiment_label="BULLISH", importance=7, summary="sum",
    )
    base.update(over)
    return NewsEvent(**base)


async def test_persistence_save_duplicate_key_safe_and_roundtrip(session_factory):
    p = NewsPersistence(session_factory, clock)
    ev = _event()
    assert await p.save(ev) is True
    assert await p.save(ev) is False  # same (source, external_id): rolled back, no exception
    assert await p.save(_event(news_id="n2", title="other", url="https://a.com/y", external_id="ext-2")) is True
    assert await p.save(_event(external_id=None, news_id="n3", url="https://a.com/z")) is True
    loaded = await p.load_recent(24)
    assert len(loaded) == 3
    first = next(e for e in loaded if e.external_id == "ext-1")
    assert first.symbols == ("BTC",) and first.macro_tags == ("ETF",)
    assert first.sentiment_label == "BULLISH" and str(first.sentiment_score) == "0.43" and first.importance == 7
    assert first.published_at == ev.published_at and first.summary == "sum"
    assert await p.load_recent(0.5) == []


async def test_warm_restart_prevents_republishing(session_factory):
    p = NewsPersistence(session_factory, clock)
    e1, bus1 = make_engine(persistence=p)
    out = await e1.ingest(story_items()[:1])
    assert len(out) == 1

    e2, bus2 = make_engine(persistence=p, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    sub = bus2.subscribe(Topic.NEWS)
    await e2.start()
    try:
        assert len(e2.cache) == 1
        assert await e2.ingest(story_items()) == []  # same story via other sources and same source
        assert sub.drain() == [] and e2.stats.duplicates == 3
        assert e2.cache.get(out[0].news_id).duplicate_sources == ("cryptopanic", "x:WatcherGuru")
        assert len(await e2.ingest([raw("rss:a", "Ethereum devs ship upgrade", "https://a.com/new")])) == 1
    finally:
        await e2.stop()


async def test_persistence_failure_never_stops_ingestion():
    class Broken:
        async def save(self, event):
            raise RuntimeError("db down")

        async def load_recent(self, hours, limit=0):
            raise RuntimeError("db down")

    engine, bus = make_engine(persistence=Broken())  # type: ignore[arg-type]
    sub = bus.subscribe(Topic.NEWS)
    out = await engine.ingest([raw("rss:a", "Bitcoin rallies", "https://a.com/1"), raw("rss:a", "Ethereum rallies", "https://a.com/2")])
    assert len(out) == 2 and len(sub.drain()) == 2 and engine.stats.persist_errors == 2
    await engine.start()  # warm start failure is logged + counted, not raised
    await engine.stop()
    assert engine.stats.persist_errors == 3


# ------------------------------------------------------------------------------ polling / backoff
async def test_source_failure_backoff_does_not_kill_other_sources():
    class Bad:
        name = "bad"

        async def fetch(self, client):
            raise httpx.ConnectError("auth_token=SECRETXYZ unreachable")

    class Good:
        name = "good"
        calls = 0

        async def fetch(self, client):
            Good.calls += 1
            return [raw("good", "Bitcoin rallies on ETF news", "https://g.com/1")]

    delays: list[float] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)
        await asyncio.sleep(0.001)

    settings = NewsSettings(poll_interval_seconds=30)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    engine, bus = make_engine(settings, sources=[Bad(), Good()], client_factory=lambda: client, sleep=fake_sleep)
    sub = bus.subscribe(Topic.NEWS)
    await engine.start()
    try:
        for _ in range(500):
            if engine.stats.source_errors["bad"] >= 7 and Good.calls >= 3:
                break
            await asyncio.sleep(0.005)
    finally:
        await engine.stop()
    assert client.is_closed
    assert engine.stats.source_errors["bad"] >= 7 and engine.stats.source_errors["good"] == 0
    assert Good.calls >= 3 and engine.stats.published == 1 and len(sub.drain()) == 1
    assert max(delays) == 300.0  # exponential backoff capped at 5 minutes
    assert {30.0, 60.0, 120.0, 240.0, 300.0} <= set(delays)


async def test_engine_polls_real_sources_through_mock_transport():
    rss = (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title><link>https://e.com</link><description>d</description>'
        "<item><title>Bitcoin ETF sees record $1B inflow - CoinDesk</title><link>https://e.com/a?utm_source=rss</link>"
        "<pubDate>Wed, 05 Nov 2026 11:00:00 GMT</pubDate></item></channel></rss>"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cryptopanic.com":
            return httpx.Response(200, json={"results": [{
                "id": 5, "title": STORY, "published_at": "2026-11-05T11:05:00Z",
                "original_url": "https://other.com/z", "currencies": [{"code": "BTC"}],
            }]})
        return httpx.Response(200, content=rss.encode())

    sources = [RSSSource("rss:coindesk", "https://e.com/feed", clock=clock), CryptoPanicSource("tok", ["BTC"], clock=clock),
               XSource("", ["a"])]

    async def quick_sleep(d: float) -> None:
        await asyncio.sleep(0.01)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    engine, _ = make_engine(sources=sources, client_factory=lambda: client, sleep=quick_sleep)
    await engine.start()
    try:
        for _ in range(200):
            if engine.stats.fetched >= 2:
                break
            await asyncio.sleep(0.005)
    finally:
        await engine.stop()
    assert engine.stats.published == 1 and engine.stats.duplicates >= 1


def test_build_default_sources():
    plain = build_default_sources(NewsSettings(cryptopanic_token="", x_bearer_token=""))
    names = [s.name for s in plain]
    assert "sec" in names and "fed" in names and "rss:coindesk" in names
    assert "cryptopanic" not in names and "x" not in names
    full = build_default_sources(NewsSettings(cryptopanic_token="a", x_bearer_token="b"))
    assert {"cryptopanic", "x"} <= {s.name for s in full}
