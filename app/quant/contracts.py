"""Quant-engine outputs and the SignalPayload handed to the Claude decision core (Phase 3).

Indicator VALUES are floats (fast, streaming analytics). Anything that becomes an order
price (entry / stop / take-profit) is Decimal, rounded later by the broker spec.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from app.core.types import PositionSide
from app.news.contracts import NewsEvent


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    symbol: str
    timeframe: str
    timestamp: datetime  # open_time of the closed candle these values belong to
    close: float
    rsi: float | None
    ema_fast: float | None  # EMA(20)
    ema_slow: float | None  # EMA(50)
    vwap: float | None  # session VWAP (UTC-day reset)
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    bb_width_pct: float | None  # (upper-lower)/middle
    atr: float | None
    atr_pct: float | None  # atr / close
    obi: float | None  # order-book imbalance in [-1, 1]: (bidVol-askVol)/(bidVol+askVol), top-N levels
    spread_bps: float | None
    volume_zscore: float | None
    warm: bool  # True once every indicator has enough history

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "timeframe": self.timeframe, "timestamp": self.timestamp.isoformat(),
            "close": self.close, "rsi": self.rsi, "ema20": self.ema_fast, "ema50": self.ema_slow,
            "vwap": self.vwap, "bb_upper": self.bb_upper, "bb_middle": self.bb_middle, "bb_lower": self.bb_lower,
            "bb_width_pct": self.bb_width_pct, "atr": self.atr, "atr_pct": self.atr_pct, "obi": self.obi,
            "spread_bps": self.spread_bps, "volume_zscore": self.volume_zscore, "warm": self.warm,
        }


class StrategyName(str, Enum):
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM = "MOMENTUM"
    SCALP_OBI = "SCALP_OBI"


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """Raw output of one deterministic strategy for one closed candle."""

    strategy: StrategyName
    symbol: str
    direction: PositionSide
    score: float  # 0..1 confluence strength for this strategy
    factors: tuple[str, ...]  # human-readable reasons, e.g. "RSI 27.4 < 30"
    entry: Decimal
    stop_loss: Decimal  # ATR-based
    take_profit: Decimal  # ATR-based
    timestamp: datetime


class PreFilterOutcome(str, Enum):
    TRIGGER = "TRIGGER"
    NO_SETUP = "NO_SETUP"
    LOW_CONFLUENCE = "LOW_CONFLUENCE"
    FEE_PRECHECK_FAILED = "FEE_PRECHECK_FAILED"
    DATA_PAUSED = "DATA_PAUSED"  # stale/disconnected feed -> never trigger
    NOT_WARM = "NOT_WARM"
    COOLDOWN = "COOLDOWN"
    RATE_LIMITED = "RATE_LIMITED"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"


@dataclass(frozen=True, slots=True)
class FeePreCheck:
    expected_gross_profit: Decimal
    round_trip_cost: Decimal
    cover_multiple: Decimal  # expected_gross_profit / round_trip_cost
    required_multiple: Decimal
    reference_quantity: Decimal  # notional-normalised qty used for the check
    passes: bool

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "expected_gross_profit": str(self.expected_gross_profit), "round_trip_cost": str(self.round_trip_cost),
            "cover_multiple": str(self.cover_multiple), "required_multiple": str(self.required_multiple),
            "reference_quantity": str(self.reference_quantity), "passes": self.passes,
        }


@dataclass(frozen=True, slots=True)
class PreFilterResult:
    symbol: str
    outcome: PreFilterOutcome
    timestamp: datetime
    detail: str
    best_signal: StrategySignal | None = None
    fee_check: FeePreCheck | None = None


@dataclass(frozen=True, slots=True)
class SignalPayload:
    """Everything Claude needs for a single ONAY / RED / RİSK AZALTMA decision. JSON-serialisable."""

    signal_id: str
    created_at: datetime
    symbol: str
    timeframe: str
    direction: PositionSide
    primary_strategy: StrategyName
    confluence_score: float  # 0..1 combined
    confluence_factors: tuple[str, ...]
    supporting_strategies: tuple[StrategyName, ...]
    proposed_entry: Decimal
    proposed_stop_loss: Decimal
    proposed_take_profit: Decimal
    indicators: IndicatorSnapshot
    fee_precheck: FeePreCheck
    market_health: dict[str, Any]  # feed ages, spread, gate state at trigger time
    recent_news: tuple[NewsEvent, ...]  # relevant, deduplicated, newest first, capped
    news_sentiment_avg: float | None
    portfolio_context: dict[str, Any] = field(default_factory=dict)  # filled by Phase 3 before calling Claude

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "created_at": self.created_at.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "primary_strategy": self.primary_strategy.value,
            "confluence_score": round(self.confluence_score, 4),
            "confluence_factors": list(self.confluence_factors),
            "supporting_strategies": [s.value for s in self.supporting_strategies],
            "proposed_entry": str(self.proposed_entry),
            "proposed_stop_loss": str(self.proposed_stop_loss),
            "proposed_take_profit": str(self.proposed_take_profit),
            "indicators": self.indicators.as_dict(),
            "fee_precheck": self.fee_precheck.as_dict(),
            "market_health": self.market_health,
            "recent_news": [n.to_dict() for n in self.recent_news],
            "news_sentiment_avg": self.news_sentiment_avg,
            "portfolio_context": self.portfolio_context,
        }
