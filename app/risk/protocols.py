"""Structural interfaces that decouple the hard-risk engine from brokers and the database."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from app.core.types import AccountSnapshot, InstrumentSpec, Order, Position


@runtime_checkable
class BrokerLike(Protocol):
    async def fetch_account(self) -> AccountSnapshot: ...

    async def fetch_positions(self) -> list[Position]: ...

    async def fetch_open_orders(self, symbol: str | None = None) -> list[Order]: ...

    async def close_all_positions(self) -> list[Order]: ...

    async def close_position(self, symbol: str) -> Order | None: ...

    async def cancel_all_orders(self, symbol: str | None = None) -> int: ...

    async def get_instrument(self, symbol: str) -> InstrumentSpec: ...

    def round_qty_down(self, spec: InstrumentSpec, qty: Decimal) -> Decimal: ...

    def round_price_down(self, spec: InstrumentSpec, price: Decimal) -> Decimal: ...

    def round_price_up(self, spec: InstrumentSpec, price: Decimal) -> Decimal: ...


@runtime_checkable
class RiskEventSink(Protocol):
    async def record(self, level: str, code: str, message: str, details: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class KillSwitchState:
    """Durable kill-switch state. Survives restarts through a KillSwitchStore."""

    engaged: bool = False
    reason: str | None = None
    engaged_at: datetime | None = None
    engaged_day: date | None = None  # UTC trading day on which it was engaged
    trading_day: date | None = None  # UTC trading day the baselines below belong to
    day_start_equity: Decimal | None = None
    peak_equity: Decimal | None = None
    consecutive_losses: int = 0


@runtime_checkable
class KillSwitchStore(Protocol):
    async def load(self) -> KillSwitchState | None: ...

    async def save(self, state: KillSwitchState) -> None: ...
