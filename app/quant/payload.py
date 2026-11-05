"""Builds the JSON-serialisable ``SignalPayload`` for the Claude decision core."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from app.core.types import utcnow
from app.news.contracts import NewsEvent
from app.quant.contracts import IndicatorSnapshot, PreFilterOutcome, PreFilterResult, SignalPayload, StrategySignal
from app.quant.prefilter import combined_score


def _sentiment_avg(news: Sequence[NewsEvent]) -> float | None:
    if not news:
        return None
    total_w = sum(n.importance for n in news)
    if total_w <= 0:
        return float(sum(n.sentiment_score for n in news) / len(news))
    return float(sum(n.sentiment_score * n.importance for n in news) / total_w)


def build_signal_payload(
    result: PreFilterResult,
    supporting: Sequence[StrategySignal],
    snap: IndicatorSnapshot,
    news: Sequence[NewsEvent],
    market_health: dict[str, Any],
    *,
    max_news: int,
    clock: Callable[[], datetime] = utcnow,
) -> SignalPayload:
    if result.outcome is not PreFilterOutcome.TRIGGER:
        raise ValueError(f"payload requires a TRIGGER result, got {result.outcome.value}")
    best = result.best_signal
    if best is None or result.fee_check is None:
        raise ValueError("TRIGGER result must carry best_signal and fee_check")
    recent = tuple(sorted(news, key=lambda n: n.published_at, reverse=True)[: max(0, max_news)])
    factors: list[str] = []
    seen: set[str] = set()
    for sig in (best, *supporting):
        for f in sig.factors:
            if f not in seen:
                seen.add(f)
                factors.append(f)
    return SignalPayload(
        signal_id=SignalPayload.new_id(),
        created_at=clock(),
        symbol=best.symbol,
        timeframe=snap.timeframe,
        direction=best.direction,
        primary_strategy=best.strategy,
        confluence_score=combined_score(best, supporting),
        confluence_factors=tuple(factors),
        supporting_strategies=tuple(s.strategy for s in supporting),
        proposed_entry=best.entry,
        proposed_stop_loss=best.stop_loss,
        proposed_take_profit=best.take_profit,
        indicators=snap,
        fee_precheck=result.fee_check,
        market_health=dict(market_health),
        recent_news=recent,
        news_sentiment_avg=_sentiment_avg(recent),
    )


def payload_to_json(payload: SignalPayload) -> str:
    return json.dumps(payload.to_json_dict(), ensure_ascii=False, sort_keys=True, default=str)
