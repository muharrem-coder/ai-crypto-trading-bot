"""Test doubles for the execution layer: recorders, a failure-injecting broker wrapper and a stub PositionManager."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.brokers.base import BaseBrokerAdapter
from app.core.errors import BrokerError
from app.core.types import (
    AccountSnapshot,
    FundingInfo,
    InstrumentSpec,
    MarginMode,
    Order,
    OrderRequest,
    OrderType,
    Position,
    Ticker,
)
from app.execution.contracts import ExecutionReport, ManagedPosition


class NullTradeRecorder:
    """Discards everything."""

    async def record_order(self, order: Order, purpose: str, decision_id: str | None) -> None:
        return None

    async def open_trade(self, position: ManagedPosition, leverage: int) -> str | None:
        return None

    async def close_trade(self, trade_id: str | None, exit_price: Decimal, exit_reason: str,
                          fees_paid: Decimal | None) -> Decimal | None:
        return None

    async def record_execution(self, report: ExecutionReport) -> None:
        return None

    async def risk_event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        return None


@dataclass(slots=True)
class RecordedOrder:
    order: Order
    purpose: str
    decision_id: str | None


@dataclass(slots=True)
class RecordedRiskEvent:
    level: str
    code: str
    message: str
    details: dict[str, Any]


@dataclass(slots=True)
class RecordedClose:
    trade_id: str | None
    exit_price: Decimal
    exit_reason: str
    fees_paid: Decimal | None


class InMemoryTradeRecorder:
    """Keeps everything in lists so tests can assert on it."""

    def __init__(self) -> None:
        self.orders: list[RecordedOrder] = []
        self.opened: list[tuple[ManagedPosition, int]] = []
        self.closed: list[RecordedClose] = []
        self.executions: list[ExecutionReport] = []
        self.risk_events: list[RecordedRiskEvent] = []
        self._trade_seq = 0

    async def record_order(self, order: Order, purpose: str, decision_id: str | None) -> None:
        self.orders.append(RecordedOrder(order, purpose, decision_id))

    async def open_trade(self, position: ManagedPosition, leverage: int) -> str | None:
        self._trade_seq += 1
        self.opened.append((position, leverage))
        return f"trade-{self._trade_seq}"

    async def close_trade(self, trade_id: str | None, exit_price: Decimal, exit_reason: str,
                          fees_paid: Decimal | None) -> Decimal | None:
        self.closed.append(RecordedClose(trade_id, exit_price, exit_reason, fees_paid))
        return None

    async def record_execution(self, report: ExecutionReport) -> None:
        self.executions.append(report)

    async def risk_event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        self.risk_events.append(RecordedRiskEvent(level, code, message, dict(details)))

    # ---- helpers
    def purposes(self) -> list[str]:
        return [r.purpose for r in self.orders]

    def events_by_code(self, code: str) -> list[RecordedRiskEvent]:
        return [e for e in self.risk_events if e.code == code]

    def events_by_level(self, level: str) -> list[RecordedRiskEvent]:
        return [e for e in self.risk_events if e.level == level]


class StubPositionManager:
    """Records registrations. ``is_managed`` is True for every registered symbol until ``release`` is called."""

    def __init__(self, raise_on_register: Exception | None = None) -> None:
        self.registered: list[ManagedPosition] = []
        self.managed: set[str] = set()
        self.raise_on_register = raise_on_register

    async def register(self, position: ManagedPosition) -> None:
        if self.raise_on_register is not None:
            raise self.raise_on_register
        self.registered.append(position)
        self.managed.add(position.symbol)

    def is_managed(self, symbol: str) -> bool:
        return symbol in self.managed

    def release(self, symbol: str) -> None:
        self.managed.discard(symbol)


def _default_exc() -> Exception:
    return BrokerError("injected broker failure")


class FlakyBroker(BaseBrokerAdapter):
    """Wraps a broker (normally PaperBrokerAdapter), delegating everything but failing chosen calls.

    ``fail_order_types[OrderType.STOP_MARKET] = 3`` makes the first three STOP_MARKET ``create_order`` calls raise
    ``exc_factory()``; ``-1`` fails forever. ``fail_close_position`` does the same for ``close_position``.
    Every call is appended to ``calls`` (``"create_order:STOP_MARKET"``, ``"close_position:BTC/USDT:USDT"`` ...) and
    every submitted OrderRequest to ``requests``; attempts per order type are counted in ``attempts``.
    Unknown attributes (``update_price``, ``orders``, ``fills`` ...) fall through to the inner broker.
    """

    def __init__(
        self,
        inner: BaseBrokerAdapter,
        fail_order_types: dict[OrderType, int] | None = None,
        exc_factory: Callable[[], Exception] = _default_exc,
        fail_close_position: int = 0,
    ) -> None:
        super().__init__()
        self._inner = inner
        self.remaining: dict[OrderType, int] = dict(fail_order_types or {})
        self.close_failures_remaining = fail_close_position
        self._exc_factory = exc_factory
        self.calls: list[str] = []
        self.requests: list[OrderRequest] = []
        self.attempts: dict[OrderType, int] = defaultdict(int)
        self.close_attempts = 0

    def __getattr__(self, name: str) -> Any:  # only reached for attributes not defined here
        if name in ("_inner", "__setstate__"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    @property
    def inner(self) -> BaseBrokerAdapter:
        return self._inner

    # ---- lifecycle / metadata
    async def connect(self) -> None:
        self.calls.append("connect")
        await self._inner.connect()

    async def close(self) -> None:
        self.calls.append("close")
        await self._inner.close()

    async def load_instruments(self) -> dict[str, InstrumentSpec]:
        self.calls.append("load_instruments")
        return await self._inner.load_instruments()

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        self.calls.append("get_instrument")
        return await self._inner.get_instrument(symbol)

    async def fetch_ticker(self, symbol: str) -> Ticker:
        self.calls.append("fetch_ticker")
        return await self._inner.fetch_ticker(symbol)

    async def fetch_funding(self, symbol: str) -> FundingInfo | None:
        self.calls.append("fetch_funding")
        return await self._inner.fetch_funding(symbol)

    # ---- account
    async def fetch_account(self) -> AccountSnapshot:
        self.calls.append("fetch_account")
        return await self._inner.fetch_account()

    async def fetch_positions(self) -> list[Position]:
        self.calls.append("fetch_positions")
        return await self._inner.fetch_positions()

    async def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        self.calls.append("fetch_open_orders")
        return await self._inner.fetch_open_orders(symbol)

    # ---- trading
    async def create_order(self, req: OrderRequest) -> Order:
        self.calls.append(f"create_order:{req.order_type.value}")
        self.requests.append(req)
        self.attempts[req.order_type] += 1
        left = self.remaining.get(req.order_type, 0)
        if left != 0:
            if left > 0:
                self.remaining[req.order_type] = left - 1
            raise self._exc_factory()
        return await self._inner.create_order(req)

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        self.calls.append("cancel_order")
        return await self._inner.cancel_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        self.calls.append("cancel_all_orders")
        return await self._inner.cancel_all_orders(symbol)

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        self.calls.append(f"set_leverage:{leverage}")
        await self._inner.set_leverage(symbol, leverage)

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        self.calls.append(f"set_margin_mode:{mode.value}")
        await self._inner.set_margin_mode(symbol, mode)

    async def close_position(self, symbol: str) -> Order | None:
        self.calls.append(f"close_position:{symbol}")
        self.close_attempts += 1
        left = self.close_failures_remaining
        if left != 0:
            if left > 0:
                self.close_failures_remaining = left - 1
            raise self._exc_factory()
        return await self._inner.close_position(symbol)

    async def close_all_positions(self) -> list[Order]:
        self.calls.append("close_all_positions")
        return await self._inner.close_all_positions()
