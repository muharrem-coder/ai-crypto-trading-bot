"""Execution engine contracts shared by the executor, the position manager and the recorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from app.core.types import FeeSchedule, Order, PositionSide


class ExecutionStatus(str, Enum):
    HOLD = "HOLD"  # Claude (or fallback) said HOLD
    REJECTED_BY_RISK = "REJECTED_BY_RISK"
    KILL_SWITCH = "KILL_SWITCH"
    ABORTED_PRICE_DRIFT = "ABORTED_PRICE_DRIFT"
    ABORTED_STALE_DATA = "ABORTED_STALE_DATA"
    ENTRY_NOT_FILLED = "ENTRY_NOT_FILLED"  # LIMIT not filled in time, cancelled, nothing open
    OPENED_PROTECTED = "OPENED_PROTECTED"  # entry filled, hard SL resting (TP may have warnings)
    NAKED_POSITION_CLOSED = "NAKED_POSITION_CLOSED"  # SL could not be placed -> emergency market close
    NAKED_CLOSE_FAILED = "NAKED_CLOSE_FAILED"  # emergency close failed too -> kill switch engaged
    CLOSED_BY_DECISION = "CLOSED_BY_DECISION"  # Claude action CLOSE
    NOTHING_TO_CLOSE = "NOTHING_TO_CLOSE"
    ERROR = "ERROR"


@dataclass(slots=True)
class ExecutionReport:
    status: ExecutionStatus
    symbol: str
    decision_id: str | None
    signal_id: str | None
    detail: str  # human readable (Turkish ok)
    side: PositionSide | None = None
    requested_leverage: int | None = None
    approved_leverage: int | None = None
    approved_quantity: Decimal | None = None
    filled_quantity: Decimal = Decimal(0)
    entry_price: Decimal | None = None  # average fill
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    drift_atr: Decimal | None = None  # adverse drift in ATR units at execution time
    risk_reasons: list[str] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trade_id: str | None = None
    timestamp: datetime | None = None


@dataclass(slots=True)
class ManagedPosition:
    """A position opened by the executor and handed to the PositionManager."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal  # average fill price
    initial_stop: Decimal
    take_profit: Decimal | None
    atr: Decimal  # ATR at signal time (price units) - drives trailing distance
    fees: FeeSchedule
    tick_size: Decimal
    stop_order_id: str
    take_profit_order_id: str | None
    decision_id: str | None
    trade_id: str | None
    opened_at: datetime
    current_stop: Decimal = Decimal(0)  # set to initial_stop on register
    best_price: Decimal = Decimal(0)  # most favourable price seen since entry
    breakeven_locked: bool = False
    stop_updates: int = 0
    last_stop_update_at: datetime | None = None


class TradeRecorder(Protocol):
    """Persistence hooks used by the executor and position manager (DB impl in app.execution.recorder).

    Implementations must never raise into the trading path: they log and swallow DB errors.
    """

    async def record_order(self, order: Order, purpose: str, decision_id: str | None) -> None: ...

    async def open_trade(self, position: ManagedPosition, leverage: int) -> str | None: ...

    async def close_trade(self, trade_id: str | None, exit_price: Decimal, exit_reason: str,
                          fees_paid: Decimal | None) -> Decimal | None: ...

    async def record_execution(self, report: ExecutionReport) -> None: ...

    async def risk_event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None: ...
