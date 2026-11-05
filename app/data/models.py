"""Market-data domain objects produced by the ingestion engine. Prices/quantities are Decimal."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from app.core.types import Side


class FeedHealth(str, Enum):
    CONNECTING = "CONNECTING"
    LIVE = "LIVE"
    STALE = "STALE"  # connected but last update older than DataSettings.stale_after_seconds
    DISCONNECTED = "DISCONNECTED"  # socket dropped, reconnect with exponential backoff in progress


class GateState(str, Enum):
    ACTIVE = "ACTIVE"  # quant engine may emit signals
    PAUSED = "PAUSED"  # stale / disconnected data: NO signals may be produced


class DataChannel(str, Enum):
    TRADES_OR_TICKER = "TICKER"
    ORDER_BOOK = "ORDER_BOOK"
    CANDLES = "CANDLES"


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timeframe: str  # "1m", "5m", ...
    open_time: datetime  # UTC, start of bar
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal  # base volume
    closed: bool  # True once the bar is final


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Top-N L2 book. Levels sorted best-first. Bounded depth (DataSettings.orderbook_depth)."""

    symbol: str
    bids: tuple[tuple[Decimal, Decimal], ...]  # (price, qty), descending price
    asks: tuple[tuple[Decimal, Decimal], ...]  # (price, qty), ascending price
    exchange_ts: datetime | None
    received_at: datetime  # local receipt time (UTC) - used for staleness

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2


@dataclass(frozen=True, slots=True)
class LiquidationEvent:
    symbol: str
    side: Side  # side of the liquidated order (SELL = long liquidated)
    price: Decimal
    quantity: Decimal
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class FeedHealthEvent:
    symbol: str
    channel: DataChannel
    health: FeedHealth
    age_seconds: float | None  # seconds since last message on this channel (None = never received)
    reason: str
    timestamp: datetime
    reconnect_attempt: int = 0
    next_retry_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class TradingGateEvent:
    state: GateState
    reasons: tuple[str, ...]  # e.g. ("BTC/USDT:USDT ORDER_BOOK STALE 2.4s",)
    timestamp: datetime
    paused_symbols: tuple[str, ...] = field(default_factory=tuple)
