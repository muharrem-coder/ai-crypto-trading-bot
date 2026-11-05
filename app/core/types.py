"""Domain types shared by every layer (brokers, risk, strategy, Claude core, API).

CONTRACT: All monetary / price / quantity values are ``decimal.Decimal``. Never float.
All timestamps are timezone-aware UTC ``datetime``.
These types are broker-agnostic: crypto (CCXT) and US equities (Alpaca/IBKR, Phase 2)
map into the same structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AssetClass(str, Enum):
    CRYPTO = "CRYPTO"
    US_EQUITY = "US_EQUITY"  # TODO: Phase 2 - US Equities (Alpaca/IBKR)


class MarketType(str, Enum):
    SPOT = "SPOT"
    PERPETUAL = "PERPETUAL"  # linear USDT-margined swap
    EQUITY = "EQUITY"  # TODO: Phase 2 - US Equities (Alpaca/IBKR)


class TradingMode(str, Enum):
    PAPER = "PAPER"  # live MAINNET market data + news, orders filled by the in-process PaperBroker
    TESTNET = "TESTNET"  # real CCXT orders against the exchange's testnet (market data from the testnet too)
    LIVE = "LIVE"  # real money - only after every lock in Settings + the runtime live guard has passed


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def entry_side(self) -> Side:
        return Side.BUY if self is PositionSide.LONG else Side.SELL

    @property
    def exit_side(self) -> Side:
        return self.entry_side.opposite


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"  # hard stop-loss resting on the exchange
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "POST_ONLY"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}


class MarginMode(str, Enum):
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"  # never allowed by the risk engine; kept for mapping exchange state


class LiquidityRole(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Fee rates as fractions (0.0004 == 0.04%)."""

    maker_rate: Decimal
    taker_rate: Decimal

    def rate(self, role: LiquidityRole) -> Decimal:
        return self.maker_rate if role is LiquidityRole.MAKER else self.taker_rate


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Static trading rules for one symbol, normalised across brokers."""

    symbol: str  # unified symbol, e.g. "BTC/USDT:USDT" (perp) or "BTC/USDT" (spot) or "AAPL"
    asset_class: AssetClass
    market_type: MarketType
    base: str
    quote: str
    settle: str | None
    tick_size: Decimal  # minimum price increment
    step_size: Decimal  # minimum quantity increment
    min_qty: Decimal
    min_notional: Decimal
    contract_size: Decimal = Decimal("1")
    max_leverage: int = 1
    fees: FeeSchedule = field(default_factory=lambda: FeeSchedule(Decimal("0.001"), Decimal("0.001")))


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None  # LIMIT price
    stop_price: Decimal | None = None  # trigger for STOP / TAKE_PROFIT
    trailing_callback_rate: Decimal | None = None  # fraction, e.g. 0.005 == 0.5%
    trailing_activation_price: Decimal | None = None
    reduce_only: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str | None = None
    leverage: int | None = None
    margin_mode: MarginMode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Order:
    order_id: str
    client_order_id: str | None
    symbol: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: Decimal
    filled_quantity: Decimal
    price: Decimal | None
    average_fill_price: Decimal | None
    stop_price: Decimal | None
    reduce_only: bool
    fee_paid: Decimal
    fee_currency: str | None
    created_at: datetime
    updated_at: datetime
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Fill:
    trade_id: str
    order_id: str
    symbol: str
    side: Side
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: str
    liquidity: LiquidityRole
    timestamp: datetime


@dataclass(slots=True)
class Position:
    symbol: str
    side: PositionSide
    quantity: Decimal  # always positive, in base units (or shares)
    entry_price: Decimal
    mark_price: Decimal
    leverage: int
    margin_mode: MarginMode
    unrealized_pnl: Decimal
    liquidation_price: Decimal | None = None
    initial_margin: Decimal | None = None
    opened_at: datetime | None = None

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.mark_price


@dataclass(frozen=True, slots=True)
class Balance:
    currency: str
    total: Decimal  # wallet balance incl. margin in use
    free: Decimal
    used: Decimal


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Point-in-time account state used by the risk engine and watchdog."""

    equity: Decimal  # wallet balance + unrealized PnL, in quote/settle currency
    wallet_balance: Decimal
    available_balance: Decimal
    unrealized_pnl: Decimal
    margin_ratio: Decimal | None  # maintenance margin / margin balance (exchange definition); None for spot
    positions: tuple[Position, ...]
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class Ticker:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    mark: Decimal | None
    timestamp: datetime

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class FundingInfo:
    symbol: str
    rate: Decimal  # per funding interval, fraction (0.0001 == 0.01%)
    interval_hours: int
    next_funding_time: datetime | None
