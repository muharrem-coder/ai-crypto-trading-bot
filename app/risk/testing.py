"""Test doubles and builders for the risk layer (FakeBroker implements BrokerLike)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from itertools import count

from app.core.errors import BrokerError
from app.core.types import (
    AccountSnapshot,
    AssetClass,
    FeeSchedule,
    InstrumentSpec,
    MarginMode,
    MarketType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Side,
)
from app.risk.rounding import ceil_to_step, floor_to_step

D = Decimal
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_spec(
    symbol: str = "BTC/USDT:USDT",
    market_type: MarketType = MarketType.PERPETUAL,
    max_leverage: int = 20,
    maker: str = "0.0002",
    taker: str = "0.0005",
) -> InstrumentSpec:
    return InstrumentSpec(
        symbol=symbol, asset_class=AssetClass.CRYPTO, market_type=market_type, base="BTC", quote="USDT",
        settle="USDT" if market_type is MarketType.PERPETUAL else None,
        tick_size=D("0.1"), step_size=D("0.001"), min_qty=D("0.001"), min_notional=D("5"),
        max_leverage=max_leverage, fees=FeeSchedule(D(maker), D(taker)),
    )


def make_position(
    symbol: str = "BTC/USDT:USDT", side: PositionSide = PositionSide.LONG, qty: str = "0.05",
    entry: str = "50000", mark: str | None = None, leverage: int = 3,
) -> Position:
    return Position(symbol=symbol, side=side, quantity=D(qty), entry_price=D(entry), mark_price=D(mark or entry),
                    leverage=leverage, margin_mode=MarginMode.ISOLATED, unrealized_pnl=D(0))


def make_snapshot(
    equity: str = "10000", available: str | None = None, margin_ratio: str | None = "0.1",
    positions: tuple[Position, ...] = (),
) -> AccountSnapshot:
    return AccountSnapshot(
        equity=D(equity), wallet_balance=D(equity), available_balance=D(available or equity), unrealized_pnl=D(0),
        margin_ratio=D(margin_ratio) if margin_ratio is not None else None, positions=positions, timestamp=T0,
    )


_ids = count(1)


def make_order(
    symbol: str = "BTC/USDT:USDT", side: Side = Side.SELL, order_type: OrderType = OrderType.STOP_MARKET,
    reduce_only: bool = True, status: OrderStatus = OrderStatus.NEW, qty: str = "0.05", stop: str | None = "49000",
) -> Order:
    return Order(order_id=str(next(_ids)), client_order_id=None, symbol=symbol, side=side, order_type=order_type,
                 status=status, quantity=D(qty), filled_quantity=D(0), price=None, average_fill_price=None,
                 stop_price=D(stop) if stop else None, reduce_only=reduce_only, fee_paid=D(0), fee_currency=None,
                 created_at=T0, updated_at=T0)


class FakeBroker:
    """In-memory BrokerLike. Failure injection via attributes; every call is appended to ``calls``."""

    def __init__(self, equity: str = "10000", positions: list[Position] | None = None,
                 orders: list[Order] | None = None, margin_ratio: str | None = "0.1") -> None:
        self.equity = D(equity)
        self.margin_ratio = margin_ratio
        self.positions: list[Position] = list(positions or [])
        self.orders: list[Order] = list(orders or [])
        self.calls: list[str] = []
        self.fail_account = False  # fetch_account/fetch_open_orders raise
        self.close_all_noop_attempts = 0  # close_all_positions silently does nothing this many times
        self.close_all_raises = False
        self.close_position_fails = False

    async def fetch_account(self) -> AccountSnapshot:
        self.calls.append("fetch_account")
        if self.fail_account:
            raise BrokerError("unreachable")
        return make_snapshot(str(self.equity), margin_ratio=self.margin_ratio, positions=tuple(self.positions))

    async def fetch_positions(self) -> list[Position]:
        self.calls.append("fetch_positions")
        return list(self.positions)

    async def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        self.calls.append("fetch_open_orders")
        if self.fail_account:
            raise BrokerError("unreachable")
        return [o for o in self.orders if symbol is None or o.symbol == symbol]

    async def close_all_positions(self) -> list[Order]:
        self.calls.append("close_all_positions")
        if self.close_all_raises:
            raise BrokerError("close failed")
        if self.close_all_noop_attempts > 0:
            self.close_all_noop_attempts -= 1
            return []
        self.positions.clear()
        return []

    async def close_position(self, symbol: str) -> Order | None:
        self.calls.append(f"close_position:{symbol}")
        if self.close_position_fails:
            raise BrokerError("close failed")
        self.positions = [p for p in self.positions if p.symbol != symbol]
        return None

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        self.calls.append("cancel_all_orders")
        n = len([o for o in self.orders if symbol is None or o.symbol == symbol])
        self.orders = [o for o in self.orders if symbol is not None and o.symbol != symbol]
        return n

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        return make_spec(symbol)

    def round_qty_down(self, spec: InstrumentSpec, qty: Decimal) -> Decimal:
        return floor_to_step(qty, spec.step_size)

    def round_price_down(self, spec: InstrumentSpec, price: Decimal) -> Decimal:
        return floor_to_step(price, spec.tick_size)

    def round_price_up(self, spec: InstrumentSpec, price: Decimal) -> Decimal:
        return ceil_to_step(price, spec.tick_size)
