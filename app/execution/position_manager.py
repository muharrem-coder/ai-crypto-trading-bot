"""PositionManager: dynamic break-even + trailing stop, OCO cleanup and trade-close accounting.

Rule R5 ("1 cent bile olsa kari kilitla, kardaki islemi zarara dondurme")
------------------------------------------------------------------------
For a position with entry ``E``, quantity ``q`` and the instrument's taker rate ``f`` the *net unrealized
PnL* at the closing-side price ``P`` (bid for LONG, ask for SHORT) is::

    gross = (P - E) * q                       (LONG)      gross = (E - P) * q   (SHORT)
    cost  = f*E*q + f*P*q + s*(E + P)*q       (entry fee already paid + exit fee at P + slippage, s = bps/10000)
    net   = gross - cost

As soon as ``net >= breakeven_trigger_fee_multiple * cost`` (default 2x) the hard stop is moved to the
break-even-plus-fee price ``FeeCalculator.breakeven_price`` (the exit price at which ``net == 0`` after both
fees and slippage), rounded toward profit to the tick. ``breakeven_locked`` is set. Afterwards the stop trails::

    LONG : candidate = max(floor_tick(best - trail_atr_multiple * ATR), break_even_stop)
    SHORT: candidate = min(ceil_tick (best + trail_atr_multiple * ATR), break_even_stop)

and is only applied when it improves the current stop by at least ``trail_min_step_atr * ATR``, at least
``trail_min_interval_seconds`` passed since the previous move, and the new stop would not trigger immediately
(LONG: new stop < bid, SHORT: new stop > ask). The stop never loosens.

Stop replacement safety: the NEW reduce-only STOP_MARKET is placed first; the OLD one is cancelled only after
the new one was accepted. If placing fails the old stop stays (retried next tick). If cancelling the old one
fails both stay resting (reduce-only orders clamp to the position) and the cancel is retried on later ticks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from app.brokers.base import BaseBrokerAdapter
from app.config.settings import ExecutionSettings, RiskSettings
from app.core.types import (
    LiquidityRole,
    MarketType,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    utcnow,
)
from app.execution.contracts import ManagedPosition, TradeRecorder
from app.risk.fee_calculator import FeeCalculator
from app.risk.rounding import ceil_to_step, floor_to_step

log = logging.getLogger(__name__)

ZERO = Decimal(0)
_FAILED_STATUSES = frozenset({OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED})


class KillSwitchLike(Protocol):
    async def record_trade_result(self, net_pnl: Decimal) -> None: ...


def _fmt(value: Decimal) -> str:
    return format(value, "f")


def _direction(side: PositionSide) -> Decimal:
    return Decimal(1) if side is PositionSide.LONG else Decimal(-1)


class PositionManager:
    def __init__(
        self,
        broker: BaseBrokerAdapter,
        recorder: TradeRecorder,
        settings: ExecutionSettings,
        risk_settings: RiskSettings,
        *,
        kill_switch: KillSwitchLike | None = None,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._broker = broker
        self._recorder = recorder
        self._settings = settings
        self._risk = risk_settings
        self._kill_switch = kill_switch
        self._clock = clock
        self._sleep = sleep
        self._positions: dict[str, ManagedPosition] = {}
        self._stale_stops: dict[str, list[str]] = {}  # old stops whose cancel failed (bounded by managed symbols)
        self._fail_streak: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ registry
    async def register(self, position: ManagedPosition) -> None:
        async with self._lock:
            position.current_stop = position.initial_stop
            position.best_price = position.entry_price
            self._positions[position.symbol] = position
            self._stale_stops.pop(position.symbol, None)
            self._fail_streak.pop(position.symbol, None)
        log.info("managing %s %s qty=%s entry=%s stop=%s", position.symbol, position.side.value,
                 position.quantity, position.entry_price, position.initial_stop)

    def is_managed(self, symbol: str) -> bool:
        return symbol in self._positions

    def positions(self) -> tuple[ManagedPosition, ...]:
        return tuple(self._positions.values())

    def _unregister(self, symbol: str) -> None:
        self._positions.pop(symbol, None)
        self._stale_stops.pop(symbol, None)
        self._fail_streak.pop(symbol, None)

    # ------------------------------------------------------------------ maths
    def net_unrealized(self, pos: ManagedPosition, price: Decimal) -> tuple[Decimal, Decimal]:
        """Return ``(net_pnl, round_trip_cost)`` if the position were closed at ``price`` (taker fees + slippage)."""
        gross = (price - pos.entry_price) * pos.quantity * _direction(pos.side)
        cost = FeeCalculator.round_trip_cost(
            pos.entry_price, price, pos.quantity, pos.fees,
            LiquidityRole.TAKER, LiquidityRole.TAKER,
            self._risk.default_slippage_bps, ZERO, ZERO, 8, pos.side, MarketType.PERPETUAL,
        ).total
        return gross - cost, cost

    def breakeven_stop(self, pos: ManagedPosition) -> Decimal:
        """Break-Even + Fee stop, rounded toward profit (LONG up, SHORT down)."""
        price = FeeCalculator.breakeven_price(
            pos.side, pos.entry_price, pos.fees, LiquidityRole.TAKER, LiquidityRole.TAKER,
            self._risk.default_slippage_bps,
        )
        # Stops trigger on the mark price but fill on the bid (LONG) / ask (SHORT): add a buffer so the
        # spread at trigger time cannot turn the locked break-even into a small loss.
        buffer = price * self._settings.breakeven_buffer_bps / Decimal(10_000)
        price = price + buffer if pos.side is PositionSide.LONG else price - buffer
        if pos.tick_size > 0:
            price = ceil_to_step(price, pos.tick_size) if pos.side is PositionSide.LONG else floor_to_step(price, pos.tick_size)
        return price

    # ------------------------------------------------------------------ recorder helpers (never raise)
    async def _record_order(self, order: Order, purpose: str, pos: ManagedPosition) -> None:
        try:
            await self._recorder.record_order(order, purpose, pos.decision_id)
        except Exception:  # noqa: BLE001 - persistence must never break trading
            log.exception("recorder.record_order failed")

    async def _event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        try:
            await self._recorder.risk_event(level, code, message, details)
        except Exception:  # noqa: BLE001
            log.exception("recorder.risk_event failed")

    # ------------------------------------------------------------------ price handling
    async def on_price(self, symbol: str, bid: Decimal, ask: Decimal) -> list[str]:
        async with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                return []
            if bid <= 0 or ask <= 0 or bid > ask:
                log.warning("%s: ignoring invalid quote bid=%s ask=%s", symbol, bid, ask)
                return []
            actions: list[str] = []
            await self._retry_stale_cancels(pos, None)

            is_long = pos.side is PositionSide.LONG
            price = bid if is_long else ask
            if (is_long and price > pos.best_price) or (not is_long and price < pos.best_price):
                pos.best_price = price

            if not pos.breakeven_locked:
                action = await self._maybe_lock_breakeven(pos, bid, ask, price)
                if action:
                    actions.append(action)
            if pos.breakeven_locked:
                action = await self._maybe_trail(pos, bid, ask)
                if action:
                    actions.append(action)
            return actions

    @staticmethod
    def _would_trigger(pos: ManagedPosition, stop: Decimal, bid: Decimal, ask: Decimal) -> bool:
        return stop >= bid if pos.side is PositionSide.LONG else stop <= ask

    async def _maybe_lock_breakeven(self, pos: ManagedPosition, bid: Decimal, ask: Decimal, price: Decimal) -> str | None:
        net, cost = self.net_unrealized(pos, price)
        if net < self._settings.breakeven_trigger_fee_multiple * cost:
            return None
        be = self.breakeven_stop(pos)
        is_long = pos.side is PositionSide.LONG
        improves = be > pos.current_stop if is_long else be < pos.current_stop
        if not improves:  # stop is already at/after break-even (e.g. a tighter initial stop)
            pos.breakeven_locked = True
            return "BREAKEVEN_LOCKED (stop already protects profit)"
        if self._would_trigger(pos, be, bid, ask):
            log.info("%s: break-even stop %s would trigger immediately (bid=%s ask=%s), skipped", pos.symbol, be, bid, ask)
            return None
        old = pos.current_stop
        if not await self._replace_stop(pos, be, "STOP_LOSS"):
            return None
        pos.breakeven_locked = True
        return f"BREAKEVEN_LOCKED {_fmt(old)}→{_fmt(be)}"

    async def _maybe_trail(self, pos: ManagedPosition, bid: Decimal, ask: Decimal) -> str | None:
        atr = pos.atr
        if atr <= 0:
            return None
        s = self._settings
        is_long = pos.side is PositionSide.LONG
        be = self.breakeven_stop(pos)
        dist = s.trail_atr_multiple * atr
        if is_long:
            candidate = max(floor_to_step(pos.best_price - dist, pos.tick_size), be)
            improvement = candidate - pos.current_stop
        else:
            candidate = min(ceil_to_step(pos.best_price + dist, pos.tick_size), be)
            improvement = pos.current_stop - candidate
        if improvement < s.trail_min_step_atr * atr:
            return None
        now = self._clock()
        if pos.last_stop_update_at is not None and (now - pos.last_stop_update_at).total_seconds() < s.trail_min_interval_seconds:
            return None
        if self._would_trigger(pos, candidate, bid, ask):
            log.info("%s: trail stop %s would trigger immediately (bid=%s ask=%s), skipped", pos.symbol, candidate, bid, ask)
            return None
        old = pos.current_stop
        if not await self._replace_stop(pos, candidate, "TRAILING_STOP"):
            return None
        return f"TRAIL_MOVED {_fmt(old)}→{_fmt(candidate)}"

    # ------------------------------------------------------------------ stop replacement (never naked)
    async def _replace_stop(self, pos: ManagedPosition, new_stop: Decimal, purpose: str) -> bool:
        req = OrderRequest(
            symbol=pos.symbol,
            side=pos.side.exit_side,
            order_type=OrderType.STOP_MARKET,
            quantity=pos.quantity,
            stop_price=new_stop,
            reduce_only=True,
        )
        try:
            new_order = await self._broker.create_order(req)
            if new_order.status in _FAILED_STATUSES:
                raise RuntimeError(f"stop order came back {new_order.status.value}")
        except Exception as exc:  # noqa: BLE001 - keep the old stop, retry next tick
            streak = self._fail_streak.get(pos.symbol, 0) + 1
            self._fail_streak[pos.symbol] = streak
            log.warning("%s: new stop %s not placed (%s); keeping old stop %s (attempt %d)",
                        pos.symbol, new_stop, exc, pos.current_stop, streak)
            if streak == 1:
                await self._event("WARN", "STOP_REPLACE_FAILED",
                                  f"{pos.symbol}: yeni stop {new_stop} konamadi, eski stop {pos.current_stop} korunuyor: {exc}",
                                  {"symbol": pos.symbol, "new_stop": new_stop, "old_stop": pos.current_stop})
            return False

        self._fail_streak.pop(pos.symbol, None)
        old_id = pos.stop_order_id
        old_stop = pos.current_stop
        pos.stop_order_id = new_order.order_id
        pos.current_stop = new_stop
        pos.stop_updates += 1
        pos.last_stop_update_at = self._clock()
        await self._record_order(new_order, purpose, pos)

        if old_id and old_id != new_order.order_id:
            try:
                cancelled = await self._broker.cancel_order(old_id, pos.symbol)
                await self._record_order(cancelled, "STOP_LOSS", pos)
            except Exception as exc:  # noqa: BLE001 - both stops rest; reduce-only clamps; retry later
                self._stale_stops.setdefault(pos.symbol, []).append(old_id)
                log.warning("%s: could not cancel old stop %s (%s); both stops rest, will retry", pos.symbol, old_id, exc)
                await self._event("WARN", "OLD_STOP_CANCEL_FAILED",
                                  f"{pos.symbol}: eski stop {old_id} iptal edilemedi, tekrar denenecek: {exc}",
                                  {"symbol": pos.symbol, "old_stop_order": old_id, "old_stop": old_stop})
        return True

    async def _retry_stale_cancels(self, pos: ManagedPosition, open_ids: set[str] | None) -> None:
        stale = self._stale_stops.get(pos.symbol)
        if not stale:
            return
        remaining: list[str] = []
        for oid in stale:
            if open_ids is not None and oid not in open_ids:
                continue  # already gone at the broker
            try:
                await self._broker.cancel_order(oid, pos.symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: retry cancel of old stop %s failed: %s", pos.symbol, oid, exc)
                remaining.append(oid)
        if remaining:
            self._stale_stops[pos.symbol] = remaining
        else:
            self._stale_stops.pop(pos.symbol, None)

    # ------------------------------------------------------------------ reconcile
    async def reconcile(self) -> list[str]:
        async with self._lock:
            if not self._positions:
                return []
            try:
                broker_positions = await self._broker.fetch_positions()
                open_orders = await self._broker.fetch_open_orders()
            except Exception as exc:  # noqa: BLE001 - never treat a failed fetch as "closed"
                log.warning("reconcile skipped, broker state unavailable: %s", exc)
                return []
            live: dict[str, Position] = {p.symbol: p for p in broker_positions}
            open_ids = {o.order_id for o in open_orders}
            actions: list[str] = []
            for pos in list(self._positions.values()):
                bp = live.get(pos.symbol)
                if bp is None or bp.side is not pos.side:
                    actions.append(await self._finalize_closed(pos, open_ids))
                    continue
                await self._retry_stale_cancels(pos, open_ids)
                if pos.stop_order_id not in open_ids:
                    actions.append(await self._restore_stop(pos))
                if pos.take_profit_order_id is not None and pos.take_profit_order_id not in open_ids:
                    log.warning("%s: take-profit order %s no longer resting", pos.symbol, pos.take_profit_order_id)
                    pos.take_profit_order_id = None
            return [a for a in actions if a]

    async def _restore_stop(self, pos: ManagedPosition) -> str:
        """Stop vanished while the position is open: re-place it now, else market close."""
        log.error("%s: protective stop %s missing, re-placing at %s", pos.symbol, pos.stop_order_id, pos.current_stop)
        req = OrderRequest(
            symbol=pos.symbol, side=pos.side.exit_side, order_type=OrderType.STOP_MARKET,
            quantity=pos.quantity, stop_price=pos.current_stop, reduce_only=True,
        )
        try:
            order = await self._broker.create_order(req)
            if order.status in _FAILED_STATUSES:
                raise RuntimeError(f"stop order came back {order.status.value}")
            pos.stop_order_id = order.order_id
            await self._record_order(order, "TRAILING_STOP" if pos.breakeven_locked else "STOP_LOSS", pos)
            await self._event("WARN", "STOP_RESTORED", f"{pos.symbol}: eksik stop yeniden konuldu @ {pos.current_stop}",
                              {"symbol": pos.symbol, "stop": pos.current_stop})
            return f"STOP_RESTORED {pos.symbol} @ {_fmt(pos.current_stop)}"
        except Exception as exc:  # noqa: BLE001
            await self._event("CRITICAL", "STOP_MISSING", f"{pos.symbol}: stop yeniden konamadi ({exc}), pozisyon kapatiliyor",
                              {"symbol": pos.symbol, "stop": pos.current_stop})
        try:
            closing = await self._broker.close_position(pos.symbol)
        except Exception as exc:  # noqa: BLE001
            await self._event("CRITICAL", "STOP_MISSING_CLOSE_FAILED",
                              f"{pos.symbol}: stopsuz pozisyon kapatilamadi: {exc}", {"symbol": pos.symbol})
            engage = getattr(self._kill_switch, "engage", None)
            if engage is not None:
                with contextlib.suppress(Exception):
                    await engage(f"NAKED_POSITION {pos.symbol}: stop missing and close failed")
            return f"NAKED_POSITION {pos.symbol}"
        return await self._finalize_closed(pos, set(), closing_order=closing, forced_reason="MANUAL")

    # ------------------------------------------------------------------ closed-position accounting
    def _stop_ids(self, pos: ManagedPosition) -> set[str]:
        return {pos.stop_order_id, *self._stale_stops.get(pos.symbol, ())}

    def _exit_from_fills(self, pos: ManagedPosition) -> tuple[Decimal, Decimal, str] | None:
        """Latest exit-side fill of the symbol (paper/any adapter exposing ``fills``) -> (price, fee, order_id)."""
        fills = getattr(self._broker, "fills", None)
        if not isinstance(fills, list):
            return None
        exit_side = pos.side.exit_side
        last = next((f for f in reversed(fills) if f.symbol == pos.symbol and f.side is exit_side), None)
        if last is None:
            return None
        same = [f for f in fills if f.order_id == last.order_id and f.symbol == pos.symbol]
        qty = sum((f.quantity for f in same), ZERO)
        if qty <= 0:
            return None
        price = sum((f.price * f.quantity for f in same), ZERO) / qty
        return price, sum((f.fee for f in same), ZERO), last.order_id

    def _reason_for_order(self, pos: ManagedPosition, order_id: str) -> str:
        orders = getattr(self._broker, "orders", None)
        if isinstance(orders, dict):
            o = orders.get(order_id)
            if o is not None and o.raw.get("liquidation"):
                return "LIQUIDATION"
        if order_id in self._stop_ids(pos):
            return self._stop_reason(pos)
        if pos.take_profit_order_id is not None and order_id == pos.take_profit_order_id:
            return "TAKE_PROFIT"
        return "MANUAL"

    @staticmethod
    def _stop_reason(pos: ManagedPosition) -> str:
        return "TRAILING_STOP" if pos.breakeven_locked else "STOP_LOSS"

    def _infer_exit_from_orders(self, pos: ManagedPosition, open_ids: set[str]) -> tuple[Decimal, str] | None:
        """Generic fallback: the exit order that is no longer resting tells us price and reason."""
        stop_gone = pos.stop_order_id not in open_ids
        tp_gone = pos.take_profit_order_id is not None and pos.take_profit_order_id not in open_ids
        if tp_gone and pos.take_profit is not None and (not stop_gone or self._tp_reached(pos)):
            return pos.take_profit, "TAKE_PROFIT"
        if stop_gone:
            return pos.current_stop, self._stop_reason(pos)
        return None

    @staticmethod
    def _tp_reached(pos: ManagedPosition) -> bool:
        if pos.take_profit is None:
            return False
        return pos.best_price >= pos.take_profit if pos.side is PositionSide.LONG else pos.best_price <= pos.take_profit

    async def _cancel_leftovers(self, pos: ManagedPosition) -> None:
        """OCO emulation: once flat, no exit order of this symbol may keep resting."""
        try:
            resting = await self._broker.fetch_open_orders(pos.symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: could not list leftovers: %s", pos.symbol, exc)
            return
        known = self._stop_ids(pos) | ({pos.take_profit_order_id} if pos.take_profit_order_id else set())
        for o in resting:
            if not (o.reduce_only or o.order_id in known):
                continue
            try:
                cancelled = await self._broker.cancel_order(o.order_id, pos.symbol)
                purpose = "TAKE_PROFIT" if o.order_type is OrderType.TAKE_PROFIT_MARKET else "STOP_LOSS"
                await self._record_order(cancelled, purpose, pos)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: could not cancel leftover %s: %s", pos.symbol, o.order_id, exc)

    async def _finalize_closed(
        self,
        pos: ManagedPosition,
        open_ids: set[str],
        *,
        closing_order: Order | None = None,
        forced_reason: str | None = None,
    ) -> str:
        exit_price: Decimal | None = None
        exit_fee: Decimal | None = None
        reason = forced_reason
        exit_order_id: str | None = None

        if closing_order is not None and closing_order.average_fill_price is not None:
            exit_price, exit_fee, exit_order_id = closing_order.average_fill_price, closing_order.fee_paid, closing_order.order_id
        else:
            found = self._exit_from_fills(pos)
            if found is not None:
                exit_price, exit_fee, exit_order_id = found
                reason = reason or self._reason_for_order(pos, exit_order_id)
            else:
                inferred = self._infer_exit_from_orders(pos, open_ids)
                if inferred is not None:
                    exit_price, reason = inferred[0], reason or inferred[1]
        if exit_price is None:
            try:
                t = await self._broker.fetch_ticker(pos.symbol)
                exit_price = t.bid if pos.side is PositionSide.LONG else t.ask
            except Exception:  # noqa: BLE001
                exit_price = pos.current_stop
            reason = reason or "MANUAL"
        reason = reason or "MANUAL"

        if exit_fee is None:
            exit_fee = exit_price * pos.quantity * pos.fees.taker_rate
        entry_fee = pos.entry_price * pos.quantity * pos.fees.taker_rate
        fees = entry_fee + exit_fee

        await self._cancel_leftovers(pos)

        net: Decimal | None = None
        try:
            net = await self._recorder.close_trade(pos.trade_id, exit_price, reason, fees)
        except Exception:  # noqa: BLE001
            log.exception("recorder.close_trade failed for %s", pos.symbol)
        if net is None:
            net = (exit_price - pos.entry_price) * pos.quantity * _direction(pos.side) - fees

        if self._kill_switch is not None:
            try:
                await self._kill_switch.record_trade_result(net)
            except Exception:  # noqa: BLE001
                log.exception("kill_switch.record_trade_result failed")

        await self._event("INFO", "TRADE_CLOSED", f"{pos.symbol} {pos.side.value} kapandi: {reason} net={net}",
                          {"symbol": pos.symbol, "reason": reason, "exit_price": exit_price, "net_pnl": net,
                           "fees": fees, "trade_id": pos.trade_id})
        self._unregister(pos.symbol)
        log.info("%s closed: %s exit=%s net=%s", pos.symbol, reason, exit_price, net)
        return f"CLOSED {pos.symbol} {reason} exit={_fmt(exit_price)} net={_fmt(net)}"

    # ------------------------------------------------------------------ loop
    async def run_once(self) -> list[str]:
        actions = await self.reconcile()
        for symbol in list(self._positions):
            try:
                t = await self._broker.fetch_ticker(symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: ticker unavailable: %s", symbol, exc)
                continue
            actions.extend(await self.on_price(symbol, t.bid, t.ask))
        return actions

    async def _run(self) -> None:
        while True:
            try:
                for action in await self.run_once():
                    log.info("position-manager: %s", action)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("position manager tick failed")
            await self._sleep(self._settings.manager_poll_seconds)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._run(), name="position-manager")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
