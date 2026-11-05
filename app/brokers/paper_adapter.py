"""In-memory simulated exchange for dry-run trading and tests.

Model
-----
* One wallet balance in the quote currency. Fees and realized PnL settle straight into it.
* Perpetuals: one *netted* position per symbol (LONG or SHORT), isolated margin
  ``notional / leverage`` reserved from the free balance. ``free = wallet - sum(margin)``.
  Unrealized PnL counts toward equity but is not spendable.
* Spot: same engine with leverage fixed at 1 and no shorting. Holdings are exposed as LONG positions
  (``fetch_positions``), selling more than is held raises ``BrokerNotSupportedError``.
* Market orders fill at ask (buy) / bid (sell) moved against the trader by ``slippage_bps`` (rounded
  against the trader to the tick) and pay the taker fee. Resting LIMIT orders fill at their limit price
  and pay the maker fee. Triggered stop / take-profit / trailing orders become taker market orders at
  the then-current bid/ask (+ slippage), so gaps fill at the gapped price.
* Triggers use the mark price when supplied, else the mid price.
* Liquidation approximation (isolated): long ``entry * (1 - 1/lev + mmr)``, short
  ``entry * (1 + 1/lev - mmr)`` with ``mmr = 0.005``; when mark crosses it the position is closed at the
  liquidation price.
* Margin for resting orders is not reserved; it is re-checked at fill time (rejected if insufficient).
* ``realized_pnl_today`` is net of all fees paid today (UTC day of the injected clock).

Everything is deterministic: time comes from the injectable ``now`` callable and ids from counters.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from app.brokers.base import BaseBrokerAdapter
from app.core.errors import (
    BrokerError,
    BrokerNotSupportedError,
    InsufficientFundsError,
    OrderRejectedError,
)
from app.core.types import (
    AccountSnapshot,
    AssetClass,
    Fill,
    FundingInfo,
    InstrumentSpec,
    LiquidityRole,
    MarginMode,
    MarketType,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Side,
    Ticker,
    TimeInForce,
    utcnow,
)

log = logging.getLogger(__name__)

MAINTENANCE_MARGIN_RATE = Decimal("0.005")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_BPS = Decimal("10000")
_STOP_TYPES = (OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET, OrderType.TRAILING_STOP_MARKET)


@dataclass(slots=True)
class _PosState:
    side: PositionSide
    qty: Decimal
    entry: Decimal
    leverage: int
    margin: Decimal
    opened_at: datetime


@dataclass(slots=True)
class _Quote:
    bid: Decimal
    ask: Decimal
    mark: Decimal | None
    last: Decimal
    ts: datetime

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def trigger_price(self) -> Decimal:
        return self.mark if self.mark is not None else self.mid


@dataclass(slots=True)
class _Resting:
    order: Order
    req: OrderRequest
    active: bool = True  # trailing: becomes True once the activation price is reached
    extreme: Decimal | None = None  # trailing high-water (sell) / low-water (buy) mark


class PaperBrokerAdapter(BaseBrokerAdapter):
    asset_class = AssetClass.CRYPTO

    def __init__(
        self,
        instruments: dict[str, InstrumentSpec],
        starting_balance: Decimal,
        quote: str = "USDT",
        slippage_bps: Decimal = Decimal("2"),
        price_source: Callable[[str], Awaitable[Ticker]] | None = None,
        now: Callable[[], datetime] = utcnow,
        market_type: MarketType | None = None,
        instrument_loader: Callable[[], Awaitable[dict[str, InstrumentSpec]]] | None = None,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._instruments = dict(instruments)
        self.quote = quote
        self.slippage_bps = Decimal(slippage_bps)
        self.price_source = price_source
        self.now = now
        self._instrument_loader = instrument_loader
        self._on_close = on_close

        inferred = market_type
        if inferred is None:
            kinds = {s.market_type for s in self._instruments.values()}
            inferred = MarketType.SPOT if kinds == {MarketType.SPOT} else MarketType.PERPETUAL
        self.market_type = inferred
        self.supports_shorting = inferred is MarketType.PERPETUAL
        self.supports_leverage = inferred is MarketType.PERPETUAL

        self.wallet = Decimal(starting_balance)
        self.starting_balance = Decimal(starting_balance)
        self.fills: list[Fill] = []
        self.orders: dict[str, Order] = {}
        self._resting: dict[str, _Resting] = {}
        self._positions: dict[str, _PosState] = {}
        self._quotes: dict[str, _Quote] = {}
        self._leverage: dict[str, int] = {}
        self._margin_modes: dict[str, MarginMode] = {}
        self._funding: dict[str, FundingInfo] = {}
        self._realized: list[tuple[datetime, Decimal, Decimal]] = []  # (ts, gross pnl, fee)
        self._order_seq = 0
        self._trade_seq = 0

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        if self._instrument_loader is not None:
            self._instruments = dict(await self._instrument_loader())
            kinds = {s.market_type for s in self._instruments.values()}
            if kinds:
                self.market_type = MarketType.SPOT if kinds == {MarketType.SPOT} else MarketType.PERPETUAL
                self.supports_shorting = self.supports_leverage = self.market_type is MarketType.PERPETUAL

    async def close(self) -> None:
        if self._on_close is not None:
            callback, self._on_close = self._on_close, None
            await callback()

    async def load_instruments(self) -> dict[str, InstrumentSpec]:
        if self._instrument_loader is not None and not self._instruments:
            await self.connect()
        return dict(self._instruments)

    # ------------------------------------------------------------------ price feed
    def update_price(
        self,
        symbol: str,
        bid: Decimal,
        ask: Decimal,
        mark: Decimal | None = None,
        last: Decimal | None = None,
    ) -> None:
        """Feed a quote and run triggers (stops, limits, trailing, liquidation) for ``symbol``."""
        bid, ask = Decimal(bid), Decimal(ask)
        if bid <= 0 or ask <= 0 or bid > ask:
            raise ValueError(f"invalid quote bid={bid} ask={ask}")
        self._quotes[symbol] = _Quote(
            bid=bid,
            ask=ask,
            mark=Decimal(mark) if mark is not None else None,
            last=Decimal(last) if last is not None else (bid + ask) / 2,
            ts=self.now(),
        )
        self._process_symbol(symbol)

    def set_funding(self, info: FundingInfo) -> None:
        self._funding[info.symbol] = info

    async def _refresh_price(self, symbol: str) -> _Quote:
        if self.price_source is not None:
            t = await self.price_source(symbol)
            self.update_price(symbol, t.bid, t.ask, t.mark, t.last)
        quote = self._quotes.get(symbol)
        if quote is None:
            raise BrokerError(f"{symbol}: no price available (feed update_price or configure price_source)")
        return quote

    async def fetch_ticker(self, symbol: str) -> Ticker:
        await self.get_instrument(symbol)
        q = await self._refresh_price(symbol)
        return Ticker(symbol=symbol, bid=q.bid, ask=q.ask, last=q.last, mark=q.mark, timestamp=q.ts)

    async def fetch_funding(self, symbol: str) -> FundingInfo | None:
        if self.market_type is not MarketType.PERPETUAL:
            return None
        return self._funding.get(symbol)

    # ------------------------------------------------------------------ account
    @staticmethod
    def _direction(side: PositionSide) -> Decimal:
        return _ONE if side is PositionSide.LONG else -_ONE

    def _mark(self, symbol: str, pos: _PosState) -> Decimal:
        q = self._quotes.get(symbol)
        return q.trigger_price if q is not None else pos.entry

    def _unrealized(self, symbol: str, pos: _PosState) -> Decimal:
        return (self._mark(symbol, pos) - pos.entry) * pos.qty * self._direction(pos.side)

    def _margin_used(self) -> Decimal:
        return sum((p.margin for p in self._positions.values()), _ZERO)

    @property
    def free_balance(self) -> Decimal:
        return self.wallet - self._margin_used()

    @property
    def realized_pnl_today(self) -> Decimal:
        """Net realized PnL (gross closing PnL minus every fee) for the current UTC day."""
        today = self.now().date()
        return sum((g - f for ts, g, f in self._realized if ts.date() == today), _ZERO)

    @property
    def gross_realized_pnl_today(self) -> Decimal:
        today = self.now().date()
        return sum((g for ts, g, _ in self._realized if ts.date() == today), _ZERO)

    def _liquidation_price(self, pos: _PosState) -> Decimal | None:
        if self.market_type is not MarketType.PERPETUAL:
            return None
        lev = Decimal(pos.leverage)
        if pos.side is PositionSide.LONG:
            return pos.entry * (_ONE - _ONE / lev + MAINTENANCE_MARGIN_RATE)
        return pos.entry * (_ONE + _ONE / lev - MAINTENANCE_MARGIN_RATE)

    def _to_position(self, symbol: str, pos: _PosState) -> Position:
        return Position(
            symbol=symbol,
            side=pos.side,
            quantity=pos.qty,
            entry_price=pos.entry,
            mark_price=self._mark(symbol, pos),
            leverage=pos.leverage,
            margin_mode=self._margin_modes.get(symbol, MarginMode.ISOLATED),
            unrealized_pnl=self._unrealized(symbol, pos),
            liquidation_price=self._liquidation_price(pos),
            initial_margin=pos.margin,
            opened_at=pos.opened_at,
        )

    async def fetch_positions(self) -> list[Position]:
        return [self._to_position(s, p) for s, p in self._positions.items()]

    async def fetch_account(self) -> AccountSnapshot:
        positions = [self._to_position(s, p) for s, p in self._positions.items()]
        unrealized = sum((p.unrealized_pnl for p in positions), _ZERO)
        equity = self.wallet + unrealized
        margin_ratio: Decimal | None = None
        if self.market_type is MarketType.PERPETUAL:
            maintenance = sum((p.quantity * p.mark_price * MAINTENANCE_MARGIN_RATE for p in positions), _ZERO)
            margin_ratio = maintenance / equity if equity > 0 else (Decimal("1") if positions else _ZERO)
        return AccountSnapshot(
            equity=equity,
            wallet_balance=self.wallet,
            available_balance=self.free_balance,
            unrealized_pnl=unrealized,
            margin_ratio=margin_ratio,
            positions=tuple(positions),
            timestamp=self.now(),
        )

    async def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            replace(r.order)
            for r in self._resting.values()
            if symbol is None or r.order.symbol == symbol
        ]

    # ------------------------------------------------------------------ settings
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        spec = await self.get_instrument(symbol)
        if self.market_type is not MarketType.PERPETUAL:
            if leverage != 1:
                raise BrokerNotSupportedError("Spot trading does not support leverage")
            return
        if leverage < 1 or leverage > spec.max_leverage:
            raise OrderRejectedError(f"{symbol}: leverage {leverage} outside 1..{spec.max_leverage}")
        pos = self._positions.get(symbol)
        if pos is not None and leverage != pos.leverage:
            new_margin = pos.entry * pos.qty / leverage
            others = self._margin_used() - pos.margin
            if self.wallet - others - new_margin < 0:
                raise InsufficientFundsError(f"{symbol}: not enough balance for leverage {leverage}")
            pos.margin, pos.leverage = new_margin, leverage
        self._leverage[symbol] = leverage

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        await self.get_instrument(symbol)
        if mode is not MarginMode.ISOLATED:
            raise BrokerNotSupportedError("Paper broker only simulates isolated margin")
        self._margin_modes[symbol] = mode

    # ------------------------------------------------------------------ order entry
    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"paper-{self._order_seq}"

    def _slipped(self, spec: InstrumentSpec, side: Side, quote: _Quote) -> Decimal:
        slip = self.slippage_bps / _BPS
        if side is Side.BUY:
            return self.round_price_up(spec, quote.ask * (_ONE + slip))
        return self.round_price_down(spec, quote.bid * (_ONE - slip))

    def _new_order(self, req: OrderRequest, status: OrderStatus = OrderStatus.NEW) -> Order:
        now = self.now()
        order = Order(
            order_id=self._next_order_id(),
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            status=status,
            quantity=req.quantity,
            filled_quantity=_ZERO,
            price=req.price,
            average_fill_price=None,
            stop_price=req.stop_price,
            reduce_only=req.reduce_only,
            fee_paid=_ZERO,
            fee_currency=self.quote,
            created_at=now,
            updated_at=now,
        )
        self.orders[order.order_id] = order
        return order

    def _check_reduce_only(self, req: OrderRequest) -> None:
        pos = self._positions.get(req.symbol)
        if pos is None:
            raise OrderRejectedError(f"{req.symbol}: reduce-only order rejected, no open position")
        if req.side is pos.side.entry_side:
            raise OrderRejectedError(f"{req.symbol}: reduce-only order would increase the position")
        if req.quantity > pos.qty:
            raise OrderRejectedError(
                f"{req.symbol}: reduce-only quantity {req.quantity} exceeds position {pos.qty} (would flip)"
            )

    async def create_order(self, req: OrderRequest) -> Order:
        spec = await self.get_instrument(req.symbol)
        self.pre_trade_regulatory_check(req)
        quote = await self._refresh_price(req.symbol)

        if self.market_type is MarketType.SPOT and (req.leverage or 1) > 1:
            raise BrokerNotSupportedError("Spot trading does not support leverage")
        if req.leverage is not None:
            await self.set_leverage(req.symbol, req.leverage)
        if req.margin_mode is not None:
            await self.set_margin_mode(req.symbol, req.margin_mode)

        ref = req.price or req.stop_price or (quote.ask if req.side is Side.BUY else quote.bid)
        self.validate_order_against_spec(spec, req, ref_price=ref)
        if req.price is not None and (req.price / spec.tick_size) % 1 != 0:
            raise OrderRejectedError(f"{req.symbol}: price {req.price} not aligned to tick {spec.tick_size}")
        if req.stop_price is not None and (req.stop_price / spec.tick_size) % 1 != 0:
            raise OrderRejectedError(f"{req.symbol}: stop price {req.stop_price} not aligned to tick {spec.tick_size}")

        if self.market_type is MarketType.SPOT and req.side is Side.SELL:
            held = self._positions.get(req.symbol)
            if held is None or req.quantity > held.qty:
                raise BrokerNotSupportedError(f"{req.symbol}: spot SELL exceeds holdings (shorting unsupported)")
        if req.reduce_only:
            self._check_reduce_only(req)

        ot = req.order_type
        if ot is OrderType.MARKET:
            order = self._new_order(req)
            self._fill_order(order, req, spec, self._slipped(spec, req.side, quote), LiquidityRole.TAKER)
            return replace(order)
        if ot is OrderType.LIMIT:
            return await self._create_limit(req, spec, quote)
        if ot in _STOP_TYPES:
            return self._create_conditional(req, quote)
        raise OrderRejectedError(f"Unsupported order type {ot}")

    async def _create_limit(self, req: OrderRequest, spec: InstrumentSpec, quote: _Quote) -> Order:
        if req.price is None:
            raise OrderRejectedError(f"{req.symbol}: LIMIT order requires price")
        marketable = req.price >= quote.ask if req.side is Side.BUY else req.price <= quote.bid
        if marketable:
            if req.time_in_force is TimeInForce.POST_ONLY:
                raise OrderRejectedError(f"{req.symbol}: post-only order would cross the book")
            order = self._new_order(req)
            fill_price = quote.ask if req.side is Side.BUY else quote.bid
            self._fill_order(order, req, spec, fill_price, LiquidityRole.TAKER)
            return replace(order)
        order = self._new_order(req)
        if req.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
            order.status = OrderStatus.EXPIRED
            return replace(order)
        self._resting[order.order_id] = _Resting(order=order, req=req)
        return replace(order)

    def _create_conditional(self, req: OrderRequest, quote: _Quote) -> Order:
        price = quote.trigger_price
        if req.order_type is OrderType.TRAILING_STOP_MARKET:
            rate = req.trailing_callback_rate
            if rate is None or not (0 < rate < 1):
                raise OrderRejectedError(f"{req.symbol}: trailing_callback_rate must be a fraction in (0, 1)")
            act = req.trailing_activation_price
            active = act is None or (price >= act if req.side is Side.SELL else price <= act)
            state = _Resting(order=self._new_order(req), req=req, active=active, extreme=price if active else None)
            self._refresh_trailing_stop_price(state)
        else:
            stop = req.stop_price
            if stop is None:
                raise OrderRejectedError(f"{req.symbol}: {req.order_type.value} requires stop_price")
            if self._stop_triggered(req, price):
                raise OrderRejectedError(
                    f"{req.symbol}: {req.order_type.value} at {stop} would trigger immediately (price {price})"
                )
            state = _Resting(order=self._new_order(req), req=req)
        self._resting[state.order.order_id] = state
        return replace(state.order)

    @staticmethod
    def _stop_triggered(req: OrderRequest, price: Decimal) -> bool:
        stop = req.stop_price
        assert stop is not None
        is_stop = req.order_type is OrderType.STOP_MARKET
        if (req.side is Side.SELL) == is_stop:  # SELL stop / BUY take-profit: price falling through
            return price <= stop
        return price >= stop  # BUY stop / SELL take-profit: price rising through

    @staticmethod
    def _refresh_trailing_stop_price(state: _Resting) -> None:
        rate = state.req.trailing_callback_rate
        if state.extreme is None or rate is None:
            return
        if state.req.side is Side.SELL:
            state.order.stop_price = state.extreme * (_ONE - rate)
        else:
            state.order.stop_price = state.extreme * (_ONE + rate)

    # ------------------------------------------------------------------ cancel
    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        state = self._resting.get(order_id)
        if state is None or state.order.symbol != symbol:
            raise BrokerError(f"{symbol}: order {order_id} not found among open orders")
        return replace(self._cancel(state, OrderStatus.CANCELED))

    def _cancel(self, state: _Resting, status: OrderStatus) -> Order:
        self._resting.pop(state.order.order_id, None)
        state.order.status = status
        state.order.updated_at = self.now()
        return state.order

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        targets = [s for s in self._resting.values() if symbol is None or s.order.symbol == symbol]
        for state in targets:
            self._cancel(state, OrderStatus.CANCELED)
        return len(targets)

    # ------------------------------------------------------------------ fills
    def _fill_order(
        self,
        order: Order,
        req: OrderRequest,
        spec: InstrumentSpec,
        price: Decimal,
        role: LiquidityRole,
        qty: Decimal | None = None,
    ) -> None:
        """Execute ``order`` fully; on failure marks it REJECTED and re-raises."""
        qty = qty if qty is not None else order.quantity
        fee = price * qty * spec.fees.rate(role)
        try:
            realized = self._apply_fill(spec, req.side, qty, price, fee, req.reduce_only)
        except BrokerError as exc:
            order.status = OrderStatus.REJECTED
            order.updated_at = self.now()
            order.raw["reject_reason"] = str(exc)
            raise
        now = self.now()
        self._trade_seq += 1
        self.fills.append(
            Fill(
                trade_id=f"paper-t-{self._trade_seq}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                price=price,
                quantity=qty,
                fee=fee,
                fee_currency=self.quote,
                liquidity=role,
                timestamp=now,
            )
        )
        self._realized.append((now, realized, fee))
        order.status = OrderStatus.FILLED
        order.filled_quantity = qty
        order.quantity = qty
        order.average_fill_price = price
        order.fee_paid = fee
        order.updated_at = now
        order.raw["realized_pnl"] = realized

    def _apply_fill(
        self,
        spec: InstrumentSpec,
        side: Side,
        qty: Decimal,
        price: Decimal,
        fee: Decimal,
        reduce_only: bool,
    ) -> Decimal:
        """Net the fill into the position and wallet. Atomic: raises before mutating anything."""
        symbol = spec.symbol
        direction = PositionSide.LONG if side is Side.BUY else PositionSide.SHORT
        pos = self._positions.get(symbol)

        close_qty = _ZERO
        if pos is not None and pos.side is not direction:
            close_qty = min(qty, pos.qty)
        open_qty = qty - close_qty

        if reduce_only and (pos is None or close_qty == 0 or open_qty > 0):
            raise OrderRejectedError(f"{symbol}: reduce-only fill would open or flip the position")
        if self.market_type is MarketType.SPOT and open_qty > 0 and side is Side.SELL:
            raise BrokerNotSupportedError(f"{symbol}: spot cannot sell more than held")

        realized = _ZERO
        released = _ZERO
        if close_qty > 0:
            assert pos is not None
            realized = (price - pos.entry) * close_qty * self._direction(pos.side)
            released = pos.margin * close_qty / pos.qty

        leverage = pos.leverage if (pos is not None and pos.side is direction) else self._leverage.get(symbol, 1)
        open_margin = (open_qty * price / leverage) if open_qty > 0 else _ZERO
        new_wallet = self.wallet + realized - fee
        new_margin_total = self._margin_used() - released + open_margin
        if open_qty > 0 and new_wallet - new_margin_total < 0:
            raise InsufficientFundsError(
                f"{symbol}: need margin {open_margin} + fee {fee}, "
                f"free {self.wallet - self._margin_used()}"
            )

        # ---- commit
        self.wallet = new_wallet
        now = self.now()
        if close_qty > 0:
            assert pos is not None
            pos.qty -= close_qty
            pos.margin -= released
            if pos.qty == 0:
                del self._positions[symbol]
                self._cancel_reduce_only(symbol)
                pos = None
        if open_qty > 0:
            if pos is not None:  # same direction: add and re-average
                total = pos.qty + open_qty
                pos.entry = (pos.entry * pos.qty + price * open_qty) / total
                pos.qty = total
                pos.margin += open_margin
            else:
                self._positions[symbol] = _PosState(
                    side=direction, qty=open_qty, entry=price, leverage=leverage, margin=open_margin, opened_at=now
                )
        return realized

    def _cancel_reduce_only(self, symbol: str) -> None:
        for state in [s for s in self._resting.values() if s.order.symbol == symbol and s.order.reduce_only]:
            self._cancel(state, OrderStatus.CANCELED)

    # ------------------------------------------------------------------ trigger engine
    def _process_symbol(self, symbol: str) -> None:
        quote = self._quotes[symbol]
        spec = self._instruments.get(symbol)
        if spec is None:
            return
        for state in [s for s in self._resting.values() if s.order.symbol == symbol]:
            if state.order.order_id not in self._resting:  # cancelled by an earlier fill this tick
                continue
            req = state.req
            if req.order_type is OrderType.LIMIT:
                self._try_limit(state, spec, quote)
            elif req.order_type is OrderType.TRAILING_STOP_MARKET:
                self._try_trailing(state, spec, quote)
            elif self._stop_triggered(req, quote.trigger_price):
                self._trigger(state, spec, quote)
        self._check_liquidation(symbol, spec, quote)

    def _try_limit(self, state: _Resting, spec: InstrumentSpec, quote: _Quote) -> None:
        req = state.req
        assert req.price is not None
        crossed = quote.ask <= req.price if req.side is Side.BUY else quote.bid >= req.price
        if not crossed:
            return
        self._execute_resting(state, spec, req.price, LiquidityRole.MAKER)

    def _try_trailing(self, state: _Resting, spec: InstrumentSpec, quote: _Quote) -> None:
        req = state.req
        rate = req.trailing_callback_rate
        assert rate is not None
        price = quote.trigger_price
        selling = req.side is Side.SELL
        if not state.active:
            act = req.trailing_activation_price
            if act is not None and (price >= act if selling else price <= act):
                state.active, state.extreme = True, price
                self._refresh_trailing_stop_price(state)
            return
        assert state.extreme is not None
        if selling:
            state.extreme = max(state.extreme, price)
            hit = price <= state.extreme * (_ONE - rate)
        else:
            state.extreme = min(state.extreme, price)
            hit = price >= state.extreme * (_ONE + rate)
        self._refresh_trailing_stop_price(state)
        if hit:
            self._trigger(state, spec, quote)

    def _trigger(self, state: _Resting, spec: InstrumentSpec, quote: _Quote) -> None:
        self._execute_resting(state, spec, self._slipped(spec, state.req.side, quote), LiquidityRole.TAKER)

    def _execute_resting(self, state: _Resting, spec: InstrumentSpec, price: Decimal, role: LiquidityRole) -> None:
        order, req = state.order, state.req
        qty = order.quantity
        if req.reduce_only:  # exchanges clamp reduce-only orders to the live position
            pos = self._positions.get(order.symbol)
            if pos is None or req.side is pos.side.entry_side:
                self._cancel(state, OrderStatus.CANCELED)
                return
            qty = min(qty, pos.qty)
        self._resting.pop(order.order_id, None)
        try:
            self._fill_order(order, req, spec, price, role, qty=qty)
        except BrokerError as exc:
            log.warning("resting order %s rejected at fill: %s", order.order_id, exc)

    def _check_liquidation(self, symbol: str, spec: InstrumentSpec, quote: _Quote) -> None:
        pos = self._positions.get(symbol)
        if pos is None or self.market_type is not MarketType.PERPETUAL:
            return
        liq = self._liquidation_price(pos)
        assert liq is not None
        mark = quote.trigger_price
        breached = mark <= liq if pos.side is PositionSide.LONG else mark >= liq
        if not breached:
            return
        req = OrderRequest(
            symbol=symbol,
            side=pos.side.exit_side,
            order_type=OrderType.MARKET,
            quantity=pos.qty,
            reduce_only=True,
            client_order_id="liquidation",
        )
        order = self._new_order(req)
        order.raw["liquidation"] = True
        log.warning("paper liquidation %s %s qty=%s at %s", symbol, pos.side.value, pos.qty, liq)
        self._fill_order(order, req, spec, liq, LiquidityRole.TAKER)

    # ------------------------------------------------------------------ closing
    async def close_position(self, symbol: str) -> Order | None:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        return await self.create_order(
            OrderRequest(
                symbol=symbol,
                side=pos.side.exit_side,
                order_type=OrderType.MARKET,
                quantity=pos.qty,
                reduce_only=True,
                metadata={"reason": "close_position"},
            )
        )

    async def close_all_positions(self) -> list[Order]:
        orders: list[Order] = []
        for symbol in list(self._positions):
            order = await self.close_position(symbol)
            if order is not None:
                orders.append(order)
        return orders
