"""SQLAlchemy 2.0 typed declarative models for the persistence layer.

Conventions
-----------
* Money / price / quantity columns are ``NUMERIC(38, 18)`` and map to ``Decimal``.
* Every timestamp is ``TIMESTAMPTZ`` and always comes back timezone-aware UTC
  (see :class:`UTCDateTime`, which also fixes SQLite's naive round-trip).
* Enums are stored as ``VARCHAR(32)`` with a CHECK constraint (``native_enum=False``).
* JSONB on PostgreSQL, plain JSON on SQLite so the same models run in unit tests.
* ``equity_snapshots``, ``position_snapshots`` and ``market_candles`` are TimescaleDB
  hypertables on PostgreSQL (see ``migrations/001_initial_schema.sql``); their primary
  keys include the time column as Timescale requires.

The hand-written migration in ``migrations/001_initial_schema.sql`` must stay in sync
with this file (``tests/test_db_models.py`` checks table coverage).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.types import (
    AssetClass,
    LiquidityRole,
    MarginMode,
    MarketType,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    TradingMode,
    utcnow,
)

# --------------------------------------------------------------------------------------
# Persistence-only enums (the domain enums live in app.core.types)
# --------------------------------------------------------------------------------------


class SignalStatus(str, PyEnum):
    NEW = "NEW"
    SENT_TO_CLAUDE = "SENT_TO_CLAUDE"
    ACTED = "ACTED"
    DISCARDED = "DISCARDED"
    EXPIRED = "EXPIRED"


class DecisionAction(str, PyEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class RiskVerdict(str, PyEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REDUCED = "REDUCED"


class OrderPurpose(str, PyEnum):
    ENTRY = "ENTRY"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    EXIT = "EXIT"
    KILL_SWITCH = "KILL_SWITCH"


class ExitReason(str, PyEnum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    LIQUIDATION = "LIQUIDATION"


class TradeStatus(str, PyEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class RiskLevel(str, PyEnum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class SentimentLabel(str, PyEnum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


# --------------------------------------------------------------------------------------
# JSON helpers (Decimal / datetime / Enum are not JSON serialisable by default)
# --------------------------------------------------------------------------------------


def json_default(obj: Any) -> Any:
    """``json.dumps(default=...)`` hook. Decimals become strings to preserve precision."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return (obj if obj.tzinfo else obj.replace(tzinfo=timezone.utc)).isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, PyEnum):
        return obj.value
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into plain JSON types (used before writing JSON columns)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, PyEnum):  # str-Enum members are also str instances
            return value.value
        return value
    if isinstance(value, dict):
        return {str(k.value if isinstance(k, PyEnum) else k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return json_default(value)


# --------------------------------------------------------------------------------------
# Column types
# --------------------------------------------------------------------------------------


class UTCDateTime(TypeDecorator[datetime]):
    """TIMESTAMPTZ that always yields timezone-aware UTC datetimes (also on SQLite)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


MONEY = Numeric(38, 18)
JSONType = JSONB().with_variant(JSON(), "sqlite")
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")  # SQLite only autoincrements INTEGER PKs
ENUM_LENGTH = 32


def _enum(enum_cls: type[PyEnum], name: str) -> SAEnum:
    """VARCHAR(32) + CHECK(col IN (...)) enum, validated on the Python side as well."""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        validate_strings=True,
        create_constraint=True,
        length=ENUM_LENGTH,
        values_callable=lambda e: [m.value for m in e],
    )


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(UTCDateTime, nullable=False, default=utcnow, server_default=func.now())


# --------------------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------------------


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (
        UniqueConstraint("broker", "asset_class", "market_type", "trading_mode", name="uq_accounts_broker_class_market_mode"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    broker: Mapped[str] = mapped_column(String(32), nullable=False)
    # TODO: Phase 2 - US Equities (Alpaca/IBKR)
    asset_class: Mapped[AssetClass] = mapped_column(_enum(AssetClass, "asset_class"), nullable=False)
    market_type: Mapped[MarketType] = mapped_column(_enum(MarketType, "market_type"), nullable=False)
    trading_mode: Mapped[TradingMode] = mapped_column(_enum(TradingMode, "trading_mode"), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = _created_at()


class Instrument(Base):
    __tablename__ = "instruments"
    __table_args__ = (
        UniqueConstraint("account_id", "symbol", name="uq_instruments_account_symbol"),
        CheckConstraint("max_leverage >= 1", name="max_leverage_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_class: Mapped[AssetClass] = mapped_column(_enum(AssetClass, "asset_class"), nullable=False)
    market_type: Mapped[MarketType] = mapped_column(_enum(MarketType, "market_type"), nullable=False)
    base: Mapped[str] = mapped_column(String(32), nullable=False)
    quote: Mapped[str] = mapped_column(String(32), nullable=False)
    tick_size: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    step_size: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    min_qty: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    min_notional: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    maker_fee: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    taker_fee: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    max_leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utcnow, onupdate=utcnow, server_default=func.now()
    )


# --------------------------------------------------------------------------------------
# Signals and Claude decisions
# --------------------------------------------------------------------------------------


class Signal(Base):
    """Output of the deterministic quant engine (input to Claude)."""

    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = _uuid_pk()
    created_at: Mapped[datetime] = _created_at()
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    direction: Mapped[PositionSide] = mapped_column(_enum(PositionSide, "direction"), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    status: Mapped[SignalStatus] = mapped_column(
        _enum(SignalStatus, "status"), nullable=False, default=SignalStatus.NEW, server_default=SignalStatus.NEW.value
    )


class ClaudeDecision(Base):
    __tablename__ = "claude_decisions"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint("leverage IS NULL OR (leverage >= 1 AND leverage <= 20)", name="leverage_range"),
        CheckConstraint("final_leverage IS NULL OR (final_leverage >= 1 AND final_leverage <= 20)", name="final_leverage_range"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    created_at: Mapped[datetime] = _created_at()
    signal_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("signals.id", ondelete="SET NULL"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[DecisionAction] = mapped_column(_enum(DecisionAction, "action"), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    leverage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stop_loss: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    take_profit: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    # Turkish explanation shown in the web terminal ("Claude neden bu işlemi açtı?")
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    # technical signal + news + portfolio snapshot sent to Claude
    input_context: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_verdict: Mapped[RiskVerdict | None] = mapped_column(_enum(RiskVerdict, "risk_verdict"), nullable=True)
    risk_reasons: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    final_leverage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_quantity: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)  # TIMEOUT / RATE_LIMITED / ...
    execution_status: Mapped[str | None] = mapped_column(String(32), nullable=True)  # app.execution ExecutionStatus
    execution_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


Index("ix_claude_decisions_created_at_desc", ClaudeDecision.created_at.desc())
Index("ix_claude_decisions_symbol_created_at", ClaudeDecision.symbol, ClaudeDecision.created_at)
Index("ix_signals_symbol_created_at", Signal.symbol, Signal.created_at)


# --------------------------------------------------------------------------------------
# Orders, fills, trades
# --------------------------------------------------------------------------------------


class OrderRecord(Base):
    """Persisted order (named ``OrderRecord`` to avoid clashing with ``app.core.types.Order``)."""

    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("client_order_id", name="uq_orders_client_order_id"),
        UniqueConstraint("account_id", "exchange_order_id", name="uq_orders_account_exchange_order_id"),
        CheckConstraint("quantity >= 0", name="quantity_non_negative"),
        CheckConstraint("filled_quantity >= 0", name="filled_quantity_non_negative"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False)
    decision_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("claude_decisions.id", ondelete="SET NULL"), nullable=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    client_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[Side] = mapped_column(_enum(Side, "side"), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(_enum(OrderType, "order_type"), nullable=False)
    status: Mapped[OrderStatus] = mapped_column(_enum(OrderStatus, "status"), nullable=False)
    purpose: Mapped[OrderPurpose] = mapped_column(_enum(OrderPurpose, "purpose"), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    filled_quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=Decimal("0"), server_default="0")
    average_fill_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    fee_paid: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=Decimal("0"), server_default="0")
    fee_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utcnow, onupdate=utcnow, server_default=func.now()
    )
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)


Index("ix_orders_account_status", OrderRecord.account_id, OrderRecord.status)
Index("ix_orders_decision_id", OrderRecord.decision_id)
Index("ix_orders_symbol_created_at", OrderRecord.symbol, OrderRecord.created_at)


class FillRecord(Base):
    __tablename__ = "fills"
    __table_args__ = (
        UniqueConstraint("order_id", "exchange_trade_id", name="uq_fills_order_exchange_trade"),
        CheckConstraint("quantity > 0", name="quantity_positive"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    order_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)
    exchange_trade_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[Side] = mapped_column(_enum(Side, "side"), nullable=False)
    price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    fee: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=Decimal("0"), server_default="0")
    fee_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    liquidity: Mapped[LiquidityRole] = mapped_column(_enum(LiquidityRole, "liquidity"), nullable=False)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


Index("ix_fills_order_id", FillRecord.order_id)
Index("ix_fills_symbol_ts", FillRecord.symbol, FillRecord.ts)


class Trade(Base):
    """Round-trip ledger row (entry to exit) backing the terminal's history table."""

    __tablename__ = "trades"
    __table_args__ = (
        CheckConstraint("leverage >= 1", name="leverage_positive"),
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("status = 'OPEN' OR closed_at IS NOT NULL", name="closed_has_closed_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False)
    decision_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("claude_decisions.id", ondelete="SET NULL"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[PositionSide] = mapped_column(_enum(PositionSide, "side"), nullable=False)
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    margin_mode: Mapped[MarginMode] = mapped_column(_enum(MarginMode, "margin_mode"), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    entry_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    gross_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    fees_paid: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=Decimal("0"), server_default="0")
    funding_paid: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=Decimal("0"), server_default="0")
    net_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    r_multiple: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    initial_stop: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    exit_reason: Mapped[ExitReason | None] = mapped_column(_enum(ExitReason, "exit_reason"), nullable=True)
    status: Mapped[TradeStatus] = mapped_column(
        _enum(TradeStatus, "status"), nullable=False, default=TradeStatus.OPEN, server_default=TradeStatus.OPEN.value
    )


Index("ix_trades_account_status", Trade.account_id, Trade.status)
Index("ix_trades_symbol_opened_at", Trade.symbol, Trade.opened_at)
Index("ix_trades_opened_at_desc", Trade.opened_at.desc())
Index("ix_trades_decision_id", Trade.decision_id)


# --------------------------------------------------------------------------------------
# News, risk events, kill switch
# --------------------------------------------------------------------------------------


class NewsItem(Base):
    __tablename__ = "news_items"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_news_items_source_external_id"),
        CheckConstraint("sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)", name="sentiment_range"),
        CheckConstraint("importance IS NULL OR (importance >= 0 AND importance <= 10)", name="importance_range"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow, server_default=func.now())
    symbols: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)  # JSON list for portability
    sentiment_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    sentiment_label: Mapped[SentimentLabel | None] = mapped_column(_enum(SentimentLabel, "sentiment_label"), nullable=True)
    importance: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)  # 0 (noise) .. 10 (market moving)
    news_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # NewsEvent.news_id (canonical-URL sha256)
    macro_tags: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    duplicate_sources: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)


Index("ix_news_items_published_at_desc", NewsItem.published_at.desc())
Index("ix_news_items_news_hash", NewsItem.news_hash)


class RiskEvent(Base):
    """Every risk rejection / reduction and every kill-switch event."""

    __tablename__ = "risk_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    ts: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow, server_default=func.now())
    level: Mapped[RiskLevel] = mapped_column(_enum(RiskLevel, "level"), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)


Index("ix_risk_events_ts_desc", RiskEvent.ts.desc())
Index("ix_risk_events_code_ts", RiskEvent.code, RiskEvent.ts)


class KillSwitchState(Base):
    """Singleton row (id = 1) holding the global kill switch and daily-drawdown baseline."""

    __tablename__ = "kill_switch_state"
    __table_args__ = (CheckConstraint("id = 1", name="singleton"),)

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=1, autoincrement=False)
    engaged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    engaged_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    trading_day: Mapped[date | None] = mapped_column(Date, nullable=True)
    day_start_equity: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    engaged_day: Mapped[date | None] = mapped_column(Date, nullable=True)
    peak_equity: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    consecutive_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utcnow, onupdate=utcnow, server_default=func.now()
    )


# --------------------------------------------------------------------------------------
# Time series (TimescaleDB hypertables; composite PK includes the time column)
# --------------------------------------------------------------------------------------


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True)
    equity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    wallet_balance: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    margin_ratio: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    daily_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    daily_drawdown_pct: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)


class MarketCandle(Base):
    __tablename__ = "market_candles"

    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(16), primary_key=True)
    o: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    h: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    l: Mapped[Decimal] = mapped_column(MONEY, nullable=False)  # noqa: E741
    c: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    v: Mapped[Decimal] = mapped_column(MONEY, nullable=False)


Index("ix_market_candles_symbol_timeframe_ts", MarketCandle.symbol, MarketCandle.timeframe, MarketCandle.ts.desc())


class PositionSnapshot(Base):
    __tablename__ = "position_snapshots"

    ts: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), primary_key=True)
    side: Mapped[PositionSide] = mapped_column(_enum(PositionSide, "side"), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    mark_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    liquidation_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)


Index("ix_position_snapshots_account_symbol_ts", PositionSnapshot.account_id, PositionSnapshot.symbol, PositionSnapshot.ts.desc())
