"""News domain object emitted on Topic.NEWS after dedup + relevance filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True)
class NewsEvent:
    news_id: str  # stable sha256 of the canonical URL (fallback: normalised title + source)
    source: str  # "cryptopanic", "rss:coindesk", "sec", "fed", "x:<handle>"
    external_id: str | None
    title: str
    url: str | None
    published_at: datetime  # UTC
    ingested_at: datetime  # UTC
    symbols: tuple[str, ...]  # base assets matched, e.g. ("BTC", "ETH"); empty for pure macro
    macro_tags: tuple[str, ...]  # e.g. ("FED", "CPI", "ETF", "HACK", "SEC", "REGULATION")
    sentiment_score: Decimal  # -1..1 deterministic lexicon score (Claude does the real reading)
    sentiment_label: str  # BULLISH | BEARISH | NEUTRAL
    importance: int  # 0..10
    summary: str | None = None
    duplicate_sources: tuple[str, ...] = field(default_factory=tuple)  # other sources that carried it

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        d["ingested_at"] = self.ingested_at.isoformat()
        d["sentiment_score"] = str(self.sentiment_score)
        d["symbols"] = list(self.symbols)
        d["macro_tags"] = list(self.macro_tags)
        d["duplicate_sources"] = list(self.duplicate_sources)
        return d
