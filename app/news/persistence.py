"""Persistence of published news into ``news_items`` and warm-start loading.

Schema mapping:
* ``news_hash`` = NewsEvent.news_id, ``macro_tags`` / ``duplicate_sources`` are JSON lists, ``summary`` its own column;
* ``external_id`` falls back to ``news_id`` when absent;
* rows written by older builds stored macro tags in ``symbols`` with a ``#`` prefix; these are still read;
* BULLISH/BEARISH/NEUTRAL <-> POSITIVE/NEGATIVE/NEUTRAL.
``duplicate_sources`` is not persisted (cache-only).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SAWarning
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.types import utcnow
from app.db.models import NewsItem, SentimentLabel
from app.news.contracts import NewsEvent
from app.news.normalize import make_news_id

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*does not support Decimal objects natively.*", category=SAWarning)

_TO_DB = {"BULLISH": SentimentLabel.POSITIVE, "BEARISH": SentimentLabel.NEGATIVE, "NEUTRAL": SentimentLabel.NEUTRAL}
_FROM_DB = {SentimentLabel.POSITIVE: "BULLISH", SentimentLabel.NEGATIVE: "BEARISH", SentimentLabel.NEUTRAL: "NEUTRAL"}
_TAG_PREFIX = "#"


class NewsPersistence:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], clock: Callable[[], datetime] = utcnow) -> None:
        self._sf = session_factory
        self._clock = clock

    async def save(self, event: NewsEvent) -> bool:
        """Insert ``event``; False when (source, external_id) already exists. Other DB errors propagate
        (after rollback) so the engine can log + count them without stopping ingestion."""
        row = NewsItem(
            source=event.source[:64],
            external_id=(event.external_id or event.news_id)[:256],
            url=event.url,
            title=event.title,
            body=event.summary,
            summary=event.summary,
            news_hash=event.news_id,
            macro_tags=list(event.macro_tags),
            duplicate_sources=list(event.duplicate_sources),
            published_at=event.published_at,
            ingested_at=event.ingested_at,
            symbols=list(event.symbols),
            sentiment_score=Decimal(event.sentiment_score),
            sentiment_label=_TO_DB.get(event.sentiment_label, SentimentLabel.NEUTRAL),
            importance=event.importance,
        )
        async with self._sf() as session:
            try:
                session.add(row)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return False
            except BaseException:
                await session.rollback()
                raise
        return True

    async def add_duplicate_source(self, news_id: str, source: str) -> bool:
        """Record that ``source`` also carried the story ``news_id``. False if the story is not stored."""
        async with self._sf() as session:
            row = (await session.execute(select(NewsItem).where(NewsItem.news_hash == news_id).limit(1))).scalar_one_or_none()
            if row is None:
                return False
            current = list(row.duplicate_sources or [])
            if source in current or source == row.source:
                return True
            row.duplicate_sources = [*current, source]
            await session.commit()
        return True

    async def load_recent(self, hours: float, limit: int = 5000) -> list[NewsEvent]:
        """Events published within ``hours`` (oldest first, at most the newest ``limit``)."""
        cutoff = self._clock() - timedelta(hours=hours)
        stmt = (
            select(NewsItem)
            .where(NewsItem.published_at >= cutoff)
            .order_by(NewsItem.published_at.desc())
            .limit(limit)
        )
        async with self._sf() as session:
            rows = (await session.execute(stmt)).scalars().all()
        events = [self._to_event(r) for r in rows]
        events.reverse()
        return events

    @staticmethod
    def _to_event(row: NewsItem) -> NewsEvent:
        stored = list(row.symbols or [])
        return NewsEvent(
            news_id=row.news_hash or make_news_id(row.url, row.title, row.source),
            source=row.source,
            external_id=row.external_id,
            title=row.title,
            url=row.url,
            published_at=row.published_at,
            ingested_at=row.ingested_at,
            symbols=tuple(s for s in stored if not s.startswith(_TAG_PREFIX)),
            macro_tags=tuple(row.macro_tags or ()) or tuple(s[len(_TAG_PREFIX):] for s in stored if s.startswith(_TAG_PREFIX)),
            duplicate_sources=tuple(row.duplicate_sources or ()),
            sentiment_score=Decimal(row.sentiment_score if row.sentiment_score is not None else 0).quantize(Decimal("0.01")),
            sentiment_label=_FROM_DB.get(row.sentiment_label, "NEUTRAL") if row.sentiment_label else "NEUTRAL",
            importance=int(row.importance or 0),
            summary=row.summary if row.summary is not None else row.body,
        )
