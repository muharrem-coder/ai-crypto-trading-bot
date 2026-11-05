"""PositionManager on the REAL PaperBrokerAdapter: rule R5 (never turn a winner into a loser), trailing,
replacement safety, OCO cleanup and reconcile. No network."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from typing import Any

import pytest

from app.brokers.paper_adapter import PaperBrokerAdapter
from app.config.settings import ExecutionSettings, RiskSettings
from app.core.errors import BrokerError, OrderRejectedError
from app.core.types import (
    Order,
    OrderRequest,
    OrderType,
    PositionSide,
    Side,
)
from app.execution.contracts import ExecutionReport, ManagedPosition
from app.execution.position_manager import PositionManager
from app.risk.rounding import floor_to_step
from app.risk.testing import make_spec

BTC = "BTC/USDT:USDT"
T0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
QTY = D("0.1")
ATR = D("100")
START = D("100000")


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


class FlakyBroker(PaperBrokerAdapter):
    """Real paper exchange with failure injection and an optional "no OCO" mode."""

    def __init__(self, *a: Any, oco: bool = True, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.oco = oco
        self.fail_create_stop = False
        self.fail_cancel = False
        self.fail_fetch = False
        self.stop_create_calls = 0

    async def create_order(self, req: OrderRequest) -> Order:
        if req.order_type is OrderType.STOP_MARKET:
            self.stop_create_calls += 1
            if self.fail_create_stop:
                raise OrderRejectedError("injected stop failure")
        return await super().create_order(req)

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        if self.fail_cancel:
            raise BrokerError("injected cancel failure")
        return await super().cancel_order(order_id, symbol)

    async def fetch_positions(self):  # type: ignore[no-untyped-def]
        if self.fail_fetch:
            raise BrokerError("injected fetch failure")
        return await super().fetch_positions()

    def _cancel_reduce_only(self, symbol: str) -> None:
        if self.oco:
            super()._cancel_reduce_only(symbol)


class FakeRecorder:
    def __init__(self) -> None:
        self.orders: list[tuple[str, str]] = []
        self.closes: list[dict[str, Any]] = []
        self.events: list[tuple[str, str]] = []
        self.trades: dict[str, tuple[PositionSide, D, D]] = {}

    async def record_order(self, order: Order, purpose: str, decision_id: str | None) -> None:
        self.orders.append((order.order_id, purpose))

    async def open_trade(self, position: ManagedPosition, leverage: int) -> str | None:
        return None

    async def close_trade(self, trade_id, exit_price, exit_reason, fees_paid):  # type: ignore[no-untyped-def]
        side, entry, qty = self.trades[trade_id]
        gross = (exit_price - entry) * qty * (1 if side is PositionSide.LONG else -1)
        net = gross - (fees_paid or D(0))
        self.closes.append({"trade_id": trade_id, "exit": exit_price, "reason": exit_reason, "fees": fees_paid, "net": net})
        return net

    async def record_execution(self, report: ExecutionReport) -> None:
        return None

    async def risk_event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        self.events.append((level, code))


class FakeKillSwitch:
    def __init__(self) -> None:
        self.results: list[D] = []

    async def record_trade_result(self, net_pnl: D) -> None:
        self.results.append(net_pnl)


class Harness:
    def __init__(self, side: PositionSide, *, oco: bool = True, slippage: str = "2", tp_delta: str | None = None,
                 exec_kw: dict[str, Any] | None = None) -> None:
        self.side = side
        self.u = D(1) if side is PositionSide.LONG else D(-1)
        self.clock = Clock()
        self.spec = make_spec(BTC)
        self.broker = FlakyBroker({BTC: self.spec}, START, "USDT", D(slippage), now=self.clock, oco=oco)
        self.rec = FakeRecorder()
        self.ks = FakeKillSwitch()
        self.tp_delta = tp_delta
        kw = {"trail_min_interval_seconds": 1.0, "trail_atr_multiple": D("1"), "trail_min_step_atr": D("0.1")}
        kw.update(exec_kw or {})
        self.mgr = PositionManager(
            self.broker, self.rec, ExecutionSettings(**kw), RiskSettings(default_slippage_bps=D("2")),
            kill_switch=self.ks, clock=self.clock,
        )
        self.pos: ManagedPosition
        self.entry = D(0)

    def feed(self, price: D) -> tuple[D, D]:
        """Quote such that the *valuation* price (bid for LONG, ask for SHORT) equals ``price``."""
        bid, ask = (price, price + D("0.1")) if self.side is PositionSide.LONG else (price - D("0.1"), price)
        self.broker.update_price(BTC, bid, ask)
        return bid, ask

    async def setup(self) -> "Harness":
        self.feed(D("50000"))
        entry_order = await self.broker.create_order(
            OrderRequest(BTC, self.side.entry_side, OrderType.MARKET, QTY)
        )
        assert entry_order.average_fill_price is not None
        self.entry = entry_order.average_fill_price
        initial_stop = self.entry - self.u * D("500")
        stop = await self.broker.create_order(
            OrderRequest(BTC, self.side.exit_side, OrderType.STOP_MARKET, QTY, stop_price=initial_stop, reduce_only=True)
        )
        tp_id: str | None = None
        tp_price: D | None = None
        if self.tp_delta is not None:
            tp_price = self.entry + self.u * D(self.tp_delta)
            tp = await self.broker.create_order(
                OrderRequest(BTC, self.side.exit_side, OrderType.TAKE_PROFIT_MARKET, QTY, stop_price=tp_price,
                             reduce_only=True)
            )
            tp_id = tp.order_id
        self.pos = ManagedPosition(
            symbol=BTC, side=self.side, quantity=QTY, entry_price=self.entry, initial_stop=initial_stop,
            take_profit=tp_price, atr=ATR, fees=self.spec.fees, tick_size=self.spec.tick_size,
            stop_order_id=stop.order_id, take_profit_order_id=tp_id, decision_id=None, trade_id="trade-1",
            opened_at=T0,
        )
        self.rec.trades["trade-1"] = (self.side, self.entry, QTY)
        await self.mgr.register(self.pos)
        return self

    async def tick(self, price: D) -> list[str]:
        bid, ask = self.feed(price)
        return await self.mgr.on_price(BTC, bid, ask)

    def at(self, delta: str | D) -> D:
        """Price ``delta`` in the profitable direction from entry."""
        return self.entry + self.u * D(delta)

    async def stops(self) -> list[Order]:
        return [o for o in await self.broker.fetch_open_orders(BTC)
                if o.order_type is OrderType.STOP_MARKET and o.reduce_only]

    async def assert_protected(self) -> None:
        """The position must always have a resting reduce-only stop at the manager's current stop."""
        stops = await self.stops()
        assert stops, "position is NAKED"
        mine = [o for o in stops if o.order_id == self.pos.stop_order_id]
        assert mine and mine[0].stop_price == self.pos.current_stop

    def better(self, a: D, b: D) -> bool:
        """a is a strictly more protective stop than b."""
        return a > b if self.side is PositionSide.LONG else a < b


SIDES = pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT])


# ---------------------------------------------------------------------------------------------- registry
@SIDES
async def test_register_sets_state_and_snapshot(side):
    h = await Harness(side).setup()
    assert h.mgr.is_managed(BTC) and not h.mgr.is_managed("ETH/USDT:USDT")
    assert h.pos.current_stop == h.pos.initial_stop
    assert h.pos.best_price == h.entry
    snap = h.mgr.positions()
    assert isinstance(snap, tuple) and snap == (h.pos,)
    assert await h.mgr.on_price("ETH/USDT:USDT", D(1), D(1)) == []


# ---------------------------------------------------------------------------------------------- R5
@SIDES
async def test_small_profit_below_two_times_fees_keeps_stop(side):
    h = await Harness(side).setup()
    start_stop = h.pos.current_stop
    net, cost = h.mgr.net_unrealized(h.pos, h.at(100))
    assert 0 < net < 2 * cost  # profitable, but not yet 2x the round-trip cost
    assert await h.tick(h.at(100)) == []
    assert h.pos.current_stop == start_stop and not h.pos.breakeven_locked
    await h.assert_protected()
    assert len(await h.stops()) == 1


@SIDES
async def test_net_pnl_just_below_threshold_does_not_lock_and_just_above_does(side):
    h = await Harness(side).setup()
    below = h.at(200)  # ~ +20 gross vs 2x cost ~ 14.1 + cost 7.05 -> still under 3x cost
    net, cost = h.mgr.net_unrealized(h.pos, below)
    assert net < 2 * cost
    assert await h.tick(below) == []
    above = h.at(230)
    net, cost = h.mgr.net_unrealized(h.pos, above)
    assert net >= 2 * cost
    assert await h.tick(above)
    assert h.pos.breakeven_locked


@SIDES
@pytest.mark.parametrize("broker_slippage", ["0", "2"])
async def test_breakeven_lock_then_reversal_never_loses_money(side, broker_slippage):
    """CORE PROOF: once locked, falling back to the stop realises net PnL >= 0 after both fees (+slippage)."""
    h = await Harness(side, slippage=broker_slippage).setup()
    assert h.broker.wallet < START  # entry taker fee already paid

    actions = await h.tick(h.at(250))
    assert len(actions) == 1 and actions[0].startswith("BREAKEVEN_LOCKED")
    assert h.pos.breakeven_locked and h.pos.stop_updates == 1
    be = h.mgr.breakeven_stop(h.pos)
    assert h.pos.current_stop == be
    assert h.better(be, h.entry)  # beyond entry, in profit direction
    await h.assert_protected()
    assert len(await h.stops()) == 1  # old stop cancelled after the new one was accepted

    # price reverses and tags the break-even stop
    stop_price = h.pos.current_stop
    h.broker.update_price(BTC, stop_price, stop_price)
    assert await h.broker.fetch_positions() == []  # stop executed
    realized = h.broker.wallet - START
    assert realized >= 0, f"winner turned into a loser: {realized}"
    equity = (await h.broker.fetch_account()).equity
    assert equity - START >= 0

    out = await h.mgr.reconcile()
    assert len(out) == 1 and "TRAILING_STOP" in out[0]
    close = h.rec.closes[0]
    assert close["reason"] == "TRAILING_STOP"
    assert close["net"] >= 0
    assert h.ks.results == [close["net"]] and h.ks.results[0] >= 0
    assert not h.mgr.is_managed(BTC)


async def test_breakeven_stop_formula_numbers():
    h = await Harness(PositionSide.LONG, slippage="0").setup()
    # E*(1+fe+s)/(1-fx-s) with fe=fx=0.0005, s=0.0002, rounded UP to tick 0.1
    exact = h.entry * D("1.0007") / D("0.9993") * D("1.0002")  # + breakeven_buffer_bps (2)
    be = h.mgr.breakeven_stop(h.pos)
    assert exact <= be < exact + D("0.1")
    s = await Harness(PositionSide.SHORT).setup()
    exact_s = s.entry * D("0.9993") / D("1.0007") * D("0.9998")  # - breakeven_buffer_bps (2)
    be_s = s.mgr.breakeven_stop(s.pos)
    assert exact_s - D("0.1") < be_s <= exact_s  # rounded DOWN (toward profit for a short)


# ---------------------------------------------------------------------------------------------- trailing
@SIDES
async def test_trailing_ratchets_never_loosens_respects_step_and_interval(side):
    h = await Harness(side).setup()
    tick = h.spec.tick_size
    assert (await h.tick(h.at(250)))[0].startswith("BREAKEVEN_LOCKED")
    stop_history = [h.pos.current_stop]
    be = h.pos.current_stop

    # interval gate: same clock instant -> no trail even though the price ran far
    assert await h.tick(h.at(400)) == []
    assert h.pos.current_stop == be
    await h.assert_protected()

    h.clock.advance(2)
    acts = await h.tick(h.at(400))
    assert len(acts) == 1 and acts[0].startswith("TRAIL_MOVED")
    expected = (h.at(400) - h.u * ATR)
    expected = floor_to_step(expected, tick) if side is PositionSide.LONG else -floor_to_step(-expected, tick)
    assert h.pos.current_stop == expected
    assert h.better(h.pos.current_stop, be)
    stop_history.append(h.pos.current_stop)
    await h.assert_protected()
    assert len(await h.stops()) == 1

    # pullback (still above the stop): never loosens
    h.clock.advance(5)
    assert await h.tick(h.at(350)) == []
    assert h.pos.current_stop == stop_history[-1]

    # new extreme but improvement < 0.1 ATR (10): ignored
    h.clock.advance(5)
    assert await h.tick(h.at(405)) == []
    assert h.pos.current_stop == stop_history[-1]

    # a real move is allowed (interval elapsed) ...
    acts = await h.tick(h.at(600))
    assert len(acts) == 1 and acts[0].startswith("TRAIL_MOVED")
    stop_history.append(h.pos.current_stop)
    assert h.better(stop_history[2], stop_history[1]) and h.better(stop_history[1], stop_history[0])
    # ... but the next one right away is blocked by the minimum interval
    assert await h.tick(h.at(700)) == []
    assert h.pos.current_stop == stop_history[-1]
    h.clock.advance(1)
    assert (await h.tick(h.at(700)))[0].startswith("TRAIL_MOVED")
    assert h.better(h.pos.current_stop, stop_history[-1])
    assert h.pos.best_price == h.at(700)
    await h.assert_protected()

    # trailed stop fill locks real profit
    sp = h.pos.current_stop
    h.broker.update_price(BTC, sp, sp)
    assert h.broker.wallet - START > 0
    assert (await h.mgr.reconcile())[0].startswith("CLOSED")
    assert h.rec.closes[0]["reason"] == "TRAILING_STOP" and h.rec.closes[0]["net"] > 0


@SIDES
async def test_trail_never_below_breakeven_stop(side):
    # huge trail distance: candidate would be far behind break-even -> clamped, no move
    h = await Harness(side, exec_kw={"trail_atr_multiple": D("10")}).setup()
    await h.tick(h.at(250))
    be = h.pos.current_stop
    h.clock.advance(5)
    assert await h.tick(h.at(400)) == []
    assert h.pos.current_stop == be


# ---------------------------------------------------------------------------------------------- safety
@SIDES
async def test_new_stop_failure_keeps_old_stop_then_retries(side):
    h = await Harness(side).setup()
    initial = h.pos.current_stop
    h.broker.fail_create_stop = True
    assert await h.tick(h.at(250)) == []
    assert not h.pos.breakeven_locked and h.pos.current_stop == initial
    await h.assert_protected()
    assert await h.tick(h.at(260)) == []  # still failing: still protected
    await h.assert_protected()
    assert [e for e in h.rec.events if e[1] == "STOP_REPLACE_FAILED"] == [("WARN", "STOP_REPLACE_FAILED")]  # no spam

    h.broker.fail_create_stop = False
    acts = await h.tick(h.at(260))
    assert acts and acts[0].startswith("BREAKEVEN_LOCKED")
    await h.assert_protected()
    assert len(await h.stops()) == 1


@SIDES
async def test_trail_stop_failure_keeps_previous_stop(side):
    h = await Harness(side).setup()
    await h.tick(h.at(250))
    be = h.pos.current_stop
    h.clock.advance(5)
    h.broker.fail_create_stop = True
    assert await h.tick(h.at(500)) == []
    assert h.pos.current_stop == be
    await h.assert_protected()
    h.broker.fail_create_stop = False
    assert (await h.tick(h.at(500)))[0].startswith("TRAIL_MOVED")
    await h.assert_protected()


@SIDES
async def test_old_stop_cancel_failure_keeps_both_then_cleans_up(side):
    h = await Harness(side).setup()
    old_id = h.pos.stop_order_id
    h.broker.fail_cancel = True
    assert (await h.tick(h.at(250)))[0].startswith("BREAKEVEN_LOCKED")
    stops = await h.stops()
    assert {o.order_id for o in stops} == {old_id, h.pos.stop_order_id}  # both resting, never naked
    await h.assert_protected()
    h.broker.fail_cancel = False
    await h.tick(h.at(255))  # next tick retries the cancel
    stops = await h.stops()
    assert [o.order_id for o in stops] == [h.pos.stop_order_id]


@SIDES
async def test_stop_that_would_trigger_immediately_is_skipped(side):
    h = await Harness(side, exec_kw={"trail_atr_multiple": D("0.1")}).setup()  # trail distance only 10
    await h.tick(h.at(250))
    be = h.pos.current_stop
    assert await h.tick(h.at(500)) == []  # interval gate; best_price now at +500
    h.clock.advance(5)
    calls = h.broker.stop_create_calls
    # price pulls back below best-10: candidate (+490) is at/through the market -> skip the update
    assert await h.tick(h.at(485)) == []
    assert h.broker.stop_create_calls == calls  # nothing was even sent
    assert h.pos.current_stop == be
    await h.assert_protected()
    # market recovers beyond the candidate -> allowed again
    assert (await h.tick(h.at(600)))[0].startswith("TRAIL_MOVED")


# ---------------------------------------------------------------------------------------------- reconcile
@SIDES
async def test_reconcile_after_take_profit_cancels_leftover_stop(side):
    h = await Harness(side, oco=False, tp_delta="1000").setup()
    tp = h.pos.take_profit
    assert tp is not None
    h.broker.update_price(BTC, tp, tp)  # TP fills on the exchange (no OCO emulation here)
    assert await h.broker.fetch_positions() == []
    leftovers = await h.broker.fetch_open_orders(BTC)
    assert any(o.order_type is OrderType.STOP_MARKET for o in leftovers)  # test is meaningful

    out = await h.mgr.reconcile()
    assert len(out) == 1 and "TAKE_PROFIT" in out[0]
    assert await h.broker.fetch_open_orders(BTC) == []
    close = h.rec.closes[0]
    assert close["reason"] == "TAKE_PROFIT" and close["net"] > 0
    fill = h.broker.fills[-1]
    assert close["exit"] == fill.price
    assert close["fees"] == h.entry * QTY * h.spec.fees.taker_rate + fill.fee
    assert h.ks.results == [close["net"]] and h.ks.results[0] > 0
    assert not h.mgr.is_managed(BTC)
    assert await h.mgr.reconcile() == []


@SIDES
async def test_reconcile_after_stop_fill_cancels_take_profit(side):
    h = await Harness(side, oco=False, tp_delta="1000").setup()
    sp = h.pos.current_stop
    h.broker.update_price(BTC, sp, sp)
    assert any(o.order_type is OrderType.TAKE_PROFIT_MARKET for o in await h.broker.fetch_open_orders(BTC))
    out = await h.mgr.reconcile()
    assert "STOP_LOSS" in out[0]
    assert await h.broker.fetch_open_orders(BTC) == []
    close = h.rec.closes[0]
    assert close["reason"] == "STOP_LOSS" and close["net"] < 0
    assert h.ks.results[0] < 0


async def test_reconcile_generic_fallback_without_fills_uses_resting_orders():
    h = await Harness(PositionSide.LONG, oco=False, tp_delta="1000").setup()
    tp = h.pos.take_profit
    h.broker.update_price(BTC, tp, tp)
    h.broker.fills = []  # adapter that exposes no usable fills
    out = await h.mgr.reconcile()
    assert "TAKE_PROFIT" in out[0]
    assert h.rec.closes[0]["exit"] == tp and h.rec.closes[0]["reason"] == "TAKE_PROFIT"
    assert await h.broker.fetch_open_orders(BTC) == []


@SIDES
async def test_reconcile_replaces_missing_stop(side):
    h = await Harness(side).setup()
    await h.broker.cancel_order(h.pos.stop_order_id, BTC)  # stop vanished
    assert await h.stops() == []
    out = await h.mgr.reconcile()
    assert out and out[0].startswith("STOP_RESTORED")
    await h.assert_protected()
    assert h.mgr.is_managed(BTC)
    assert await h.mgr.reconcile() == []


@SIDES
async def test_missing_stop_that_cannot_be_replaced_closes_position(side):
    h = await Harness(side).setup()
    await h.broker.cancel_order(h.pos.stop_order_id, BTC)
    h.broker.fail_create_stop = True
    out = await h.mgr.reconcile()
    assert out and out[0].startswith("CLOSED")
    assert await h.broker.fetch_positions() == []
    assert ("CRITICAL", "STOP_MISSING") in h.rec.events
    assert h.rec.closes[0]["reason"] == "MANUAL"
    assert not h.mgr.is_managed(BTC)
    assert len(h.ks.results) == 1


async def test_reconcile_with_broker_outage_keeps_position_managed():
    h = await Harness(PositionSide.LONG).setup()
    h.broker.fail_fetch = True
    assert await h.mgr.reconcile() == []
    assert h.mgr.is_managed(BTC) and h.rec.closes == []


# ---------------------------------------------------------------------------------------------- loop
async def test_run_once_locks_from_ticker():
    h = await Harness(PositionSide.LONG).setup()
    h.feed(h.at(250))
    actions = await h.mgr.run_once()
    assert any(a.startswith("BREAKEVEN_LOCKED") for a in actions)
    await h.assert_protected()


async def test_background_loop_runs_and_stops_cleanly():
    h = await Harness(PositionSide.LONG).setup()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)

    mgr = PositionManager(h.broker, h.rec, ExecutionSettings(manager_poll_seconds=0.5), RiskSettings(),
                          kill_switch=h.ks, clock=h.clock, sleep=fake_sleep)
    await mgr.register(h.pos)
    h.feed(h.at(250))
    mgr.start()
    mgr.start()  # idempotent
    for _ in range(20):
        await asyncio.sleep(0)
    await mgr.stop()
    await mgr.stop()  # safe twice
    assert sleeps and set(sleeps) == {0.5}
    assert h.pos.breakeven_locked
