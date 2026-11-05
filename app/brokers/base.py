"""Broker adapter contract.

Adapter pattern
---------------
Every venue (CCXT crypto exchanges, the in-memory paper exchange, and later Alpaca / IBKR)
speaks a different dialect: different symbol formats, precision rules, order types, margin
models and error classes. The rest of the system (risk engine, execution, strategies, the
Claude layer) must not care. ``BaseBrokerAdapter`` is the single, narrow interface they code
against; each concrete adapter *adapts* a venue's native API to this interface and to the
broker-agnostic domain types in ``app.core.types``. Swapping paper for live, or crypto for
equities, is then a matter of constructing a different adapter in ``app.brokers.factory``.

Conventions
-----------
* All money / price / quantity values are ``Decimal``. Floats only exist at the very edge of a
  third-party library call and are converted immediately.
* Timestamps are timezone-aware UTC.
* Adapters raise ``BrokerError`` subclasses only; venue-specific exceptions never leak.

# TODO: Phase 2 - US Equities (Alpaca/IBKR)
# Equities need market-hours gating (``market_hours_status``), pattern-day-trader checks and
# T+1 settlement awareness. Those concerns are exposed as non-abstract hooks below
# (``market_hours_status`` / ``pre_trade_regulatory_check``) that are permissive for crypto
# and overridden by the equity adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

from app.core.errors import BrokerError, OrderRejectedError
from app.core.types import (
    AccountSnapshot,
    AssetClass,
    FundingInfo,
    InstrumentSpec,
    MarginMode,
    Order,
    OrderRequest,
    Position,
    Ticker,
)


@dataclass(frozen=True, slots=True)
class MarketHoursStatus:
    """Trading-session state. Crypto is permanently ``is_open=True`` / session ``"24x7"``."""

    is_open: bool
    session: str  # "24x7" | "REGULAR" | "PRE" | "POST" | "CLOSED"
    next_change: datetime | None = None  # next open/close instant (UTC) when known


class BaseBrokerAdapter(ABC):
    """Abstract broker. See module docstring for the Adapter pattern rationale."""

    asset_class: AssetClass = AssetClass.CRYPTO
    supports_shorting: bool = True
    supports_leverage: bool = True

    def __init__(self) -> None:
        self._instruments: dict[str, InstrumentSpec] = {}

    # ------------------------------------------------------------------ lifecycle
    @abstractmethod
    async def connect(self) -> None:
        """Open connections and load instrument metadata."""

    @abstractmethod
    async def close(self) -> None:
        """Release all network resources. Must be safe to call twice."""

    # ------------------------------------------------------------------ market data / metadata
    @abstractmethod
    async def load_instruments(self) -> dict[str, InstrumentSpec]:
        """Fetch (and cache in ``self._instruments``) all tradable instruments keyed by symbol."""

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        """Return the cached spec for ``symbol``; loads the cache lazily on first use."""
        if not self._instruments:
            await self.load_instruments()
        try:
            return self._instruments[symbol]
        except KeyError:
            raise BrokerError(f"Unknown instrument: {symbol}") from None

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Latest bid/ask/last (and mark where available)."""

    @abstractmethod
    async def fetch_funding(self, symbol: str) -> FundingInfo | None:
        """Current funding rate for perpetuals; ``None`` where the concept does not apply."""

    # ------------------------------------------------------------------ account
    @abstractmethod
    async def fetch_account(self) -> AccountSnapshot:
        """Equity, balances, margin ratio and open positions in one snapshot."""

    @abstractmethod
    async def fetch_positions(self) -> list[Position]:
        """All open positions (empty list when flat)."""

    @abstractmethod
    async def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Resting orders, optionally restricted to one symbol."""

    # ------------------------------------------------------------------ trading
    @abstractmethod
    async def create_order(self, req: OrderRequest) -> Order:
        """Submit an order; raises OrderRejectedError / InsufficientFundsError on refusal."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """Cancel one open order and return its final state."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel every open order (optionally for one symbol). Returns how many were cancelled."""

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set per-symbol leverage (no-op error-swallowing when unchanged)."""

    @abstractmethod
    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        """Set per-symbol margin mode (the risk engine only ever requests ISOLATED)."""

    @abstractmethod
    async def close_position(self, symbol: str) -> Order | None:
        """Close the position in ``symbol`` with a reduce-only MARKET order. ``None`` if flat."""

    @abstractmethod
    async def close_all_positions(self) -> list[Order]:
        """Flatten every open position with reduce-only market orders."""

    # ------------------------------------------------------------------ market hours / regulatory hooks
    # TODO: Phase 2 - US Equities (Alpaca/IBKR): override with NYSE calendar, PDT and T+1 logic.
    def market_hours_status(self) -> MarketHoursStatus:
        """Crypto trades 24x7."""
        return MarketHoursStatus(is_open=True, session="24x7", next_change=None)

    def is_market_open(self) -> bool:
        return self.market_hours_status().is_open

    def pre_trade_regulatory_check(self, req: OrderRequest) -> None:
        """Raise ``RiskViolation`` if venue/regulatory rules forbid ``req``. Permissive for crypto."""
        return None

    # ------------------------------------------------------------------ rounding helpers
    @staticmethod
    def _quantize_to(value: Decimal, increment: Decimal, rounding: str) -> Decimal:
        if increment <= 0:
            raise ValueError(f"increment must be positive, got {increment}")
        return (value / increment).to_integral_value(rounding=rounding) * increment

    @classmethod
    def round_price(cls, spec: InstrumentSpec, price: Decimal) -> Decimal:
        """Round to the nearest tick (ties away from zero)."""
        return cls._quantize_to(price, spec.tick_size, ROUND_HALF_UP)

    @classmethod
    def round_price_down(cls, spec: InstrumentSpec, price: Decimal) -> Decimal:
        """Floor to tick. Use for long stops / sell limits where rounding up would be unsafe."""
        return cls._quantize_to(price, spec.tick_size, ROUND_FLOOR)

    @classmethod
    def round_price_up(cls, spec: InstrumentSpec, price: Decimal) -> Decimal:
        """Ceil to tick. Use for short stops / buy limits where rounding down would be unsafe."""
        return cls._quantize_to(price, spec.tick_size, ROUND_CEILING)

    @classmethod
    def round_qty_down(cls, spec: InstrumentSpec, qty: Decimal) -> Decimal:
        """Floor quantity to the step size (never rounds up, so risk is never exceeded)."""
        return cls._quantize_to(qty, spec.step_size, ROUND_FLOOR)

    @staticmethod
    def validate_order_against_spec(
        spec: InstrumentSpec, req: OrderRequest, ref_price: Decimal | None = None
    ) -> None:
        """Reject orders the venue would refuse anyway.

        Checks positive quantity, ``min_qty`` and step alignment. ``min_notional`` is checked when a
        reference price is known (``ref_price``, else the order's limit / stop price) and the order is
        not reduce-only (exchanges exempt reduce-only orders from the minimum notional).
        """
        qty = req.quantity
        if qty <= 0:
            raise OrderRejectedError(f"{req.symbol}: quantity must be positive, got {qty}")
        if qty < spec.min_qty:
            raise OrderRejectedError(f"{req.symbol}: quantity {qty} below min_qty {spec.min_qty}")
        if (qty / spec.step_size) % 1 != 0:
            raise OrderRejectedError(f"{req.symbol}: quantity {qty} not aligned to step_size {spec.step_size}")
        price = ref_price if ref_price is not None else (req.price or req.stop_price)
        if price is not None and not req.reduce_only:
            notional = qty * price  # quantities are in base units
            if notional < spec.min_notional:
                raise OrderRejectedError(
                    f"{req.symbol}: notional {notional} below min_notional {spec.min_notional}"
                )
