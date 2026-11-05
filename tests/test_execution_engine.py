"""Execution engine against the REAL paper broker, risk engine and kill switch."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from app.brokers.paper_adapter import PaperBrokerAdapter
from app.claude.contracts import ClaudeDecision, DecisionAction, EntryType
from app.config.settings import ExecutionSettings, RiskSettings
from app.core.types import MarginMode, OrderStatus, OrderType, PositionSide as S, Side
from app.data.models import OrderBookSnapshot
from app.execution.contracts import ExecutionStatus as X
from app.execution.executor import ExecutionEngine
from app.execution.testing import FlakyBroker, InMemoryTradeRecorder, NullTradeRecorder, StubPositionManager
from app.quant.contracts import FeePreCheck, IndicatorSnapshot, SignalPayload, StrategyName
from app.risk.kill_switch import KillSwitch
from app.risk.memory import InMemoryKillSwitchStore, InMemoryRiskEventSink
from app.risk.risk_engine import HardRiskEngine
from app.risk.testing import make_spec

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
SYM = "BTC/USDT:USDT"
SPEC = make_spec()


# ------------------------------------------------------------------ builders
def payload(direction=S.LONG, entry="50000", atr=100.0) -> SignalPayload:
    ind = IndicatorSnapshot(SYM, "1m", NOW, 50000.0, 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, atr, 0.002, 0.0, 1.0,
                            0.0, True)
    fee = FeePreCheck(D(1), D(1), D(2), D(2), D(1), True)
    stop, tp = (D("49000"), D("52000")) if direction is S.LONG else (D("51000"), D("48000"))
    return SignalPayload("sig-1", NOW, SYM, "1m", direction, StrategyName.MOMENTUM, 0.8, ("x",), (), D(entry), stop, tp,
                         ind, fee, {}, (), None)


def decision(action=DecisionAction.LONG, leverage=10, confidence="0.7", entry_type=EntryType.MARKET, entry_price=None,
             stop=None, tp=None) -> ClaudeDecision:
    long = action is not DecisionAction.SHORT
    if action in (DecisionAction.HOLD, DecisionAction.CLOSE):
        return ClaudeDecision(action=action, confidence=D(confidence), leverage=1,
                              reasoning_tr="Test kararı için yeterince uzun gerekçe metni.")
    return ClaudeDecision(
        action=action, confidence=D(confidence), leverage=leverage, entry_type=entry_type, entry_price=entry_price,
        stop_loss=D(stop) if stop else D("49000") if long else D("51000"),
        take_profit=D(tp) if tp else D("52000") if long else D("48000"),
        reasoning_tr="Test kararı için yeterince uzun gerekçe metni.")


class Env:
    def __init__(self, fail_orders=None, fail_close=0, *, settings=None, ks_on_flaky=False):
        self.paper = PaperBrokerAdapter({SYM: SPEC}, D(10000), now=lambda: NOW)
        self.broker = FlakyBroker(self.paper, fail_orders or {}, fail_close_position=fail_close)
        self.sink = InMemoryRiskEventSink()
        rs = RiskSettings()
        self.slept: list[float] = []
        self.ks = KillSwitch(self.broker if ks_on_flaky else self.paper, InMemoryKillSwitchStore(), self.sink, rs,
                             clock=lambda: NOW, sleep=self._sleep)
        self.risk = HardRiskEngine(rs, self.ks, self.sink)
        self.pm = StubPositionManager()
        self.rec = InMemoryTradeRecorder()
        self.cfg = settings or ExecutionSettings()
        self.book_age = 0.0
        self.book_enabled = True
        self.set_market("49999.9", "50000")
        self.engine = ExecutionEngine(self.broker, self.risk, self.ks, self.pm, self.rec, self.cfg,
                                      book_provider=self._book, clock=lambda: NOW, sleep=self._sleep)

    async def _sleep(self, s):
        self.slept.append(s)

    def set_market(self, bid, ask):
        self.bid, self.ask = D(bid), D(ask)
        self.paper.update_price(SYM, self.bid, self.ask)

    def _book(self, symbol):
        if not self.book_enabled:
            return None
        return OrderBookSnapshot(symbol, ((self.bid, D(5)),), ((self.ask, D(5)),), None,
                                 NOW - timedelta(seconds=self.book_age))

    async def run(self, p=None, d=None, did="dec-1"):
        return await self.engine.execute(p or payload(), d or decision(), did)

    def resting(self):
        return self.paper._resting  # noqa: SLF001

    def orders_of(self, t):
        return [r.order for r in self.paper._resting.values() if r.order.order_type is t]  # noqa: SLF001

    def created(self, t):
        return [r for r in self.broker.requests if r.order_type is t]

    @property
    def no_orders(self):
        return not self.paper.orders and not self.broker.requests


# ------------------------------------------------------------------ happy paths
async def test_long_market_opens_protected():
    e = Env()
    rep = await e.run()
    assert rep.status is X.OPENED_PROTECTED, rep.detail
    assert rep.side is S.LONG and rep.requested_leverage == 10 and rep.approved_leverage == 5
    assert rep.filled_quantity == rep.approved_quantity > 0
    assert rep.entry_price == D("50010")  # ask 50000 + 2bps paper slippage
    assert rep.drift_atr == 0 and rep.trade_id == "trade-1"

    pos = (await e.paper.fetch_positions())[0]
    assert pos.side is S.LONG and pos.quantity == rep.filled_quantity and pos.leverage == 5
    assert e.paper._leverage[SYM] == 5 and e.paper._margin_modes[SYM] is MarginMode.ISOLATED  # noqa: SLF001
    assert "set_leverage:5" in e.broker.calls and "set_leverage:10" not in e.broker.calls

    stops, tps = e.orders_of(OrderType.STOP_MARKET), e.orders_of(OrderType.TAKE_PROFIT_MARKET)
    assert len(stops) == 1 and len(tps) == 1 and not e.orders_of(OrderType.TRAILING_STOP_MARKET)
    for o, price in ((stops[0], D("49000")), (tps[0], D("52000"))):
        assert o.reduce_only and o.side is Side.SELL and o.quantity == pos.quantity and o.stop_price == price
    assert rep.stop_loss == D("49000") and rep.take_profit == D("52000")

    # strict ordering of the broker calls
    calls = [c for c in e.broker.calls if c.startswith(("set_", "create_order"))]
    assert calls == ["set_margin_mode:ISOLATED", "set_leverage:5", "create_order:MARKET",
                     "create_order:STOP_MARKET", "create_order:TAKE_PROFIT_MARKET"]

    # hand-over
    assert len(e.pm.registered) == 1
    mp = e.pm.registered[0]
    assert mp.symbol == SYM and mp.side is S.LONG and mp.quantity == pos.quantity and mp.entry_price == D("50010")
    assert mp.initial_stop == mp.current_stop == D("49000") and mp.take_profit == D("52000")
    assert mp.atr == D("100.0") and mp.fees == SPEC.fees and mp.tick_size == SPEC.tick_size
    assert mp.stop_order_id == stops[0].order_id and mp.take_profit_order_id == tps[0].order_id
    assert mp.decision_id == "dec-1" and mp.trade_id == "trade-1" and mp.opened_at == NOW

    # persistence
    assert e.rec.purposes() == ["ENTRY", "STOP_LOSS", "TAKE_PROFIT"]
    assert all(r.decision_id == "dec-1" for r in e.rec.orders)
    assert e.rec.executions == [rep] and rep.timestamp == NOW
    assert e.rec.opened[0][1] == 5


async def test_short_market_opens_protected():
    e = Env()
    e.set_market("50000", "50000.1")
    rep = await e.run(payload(S.SHORT), decision(DecisionAction.SHORT))
    assert rep.status is X.OPENED_PROTECTED, rep.detail
    pos = (await e.paper.fetch_positions())[0]
    assert pos.side is S.SHORT and pos.quantity == rep.filled_quantity
    stop, = e.orders_of(OrderType.STOP_MARKET)
    tp, = e.orders_of(OrderType.TAKE_PROFIT_MARKET)
    assert stop.side is Side.BUY and stop.reduce_only and stop.stop_price == D("51000")
    assert tp.side is Side.BUY and tp.reduce_only and tp.stop_price == D("48000")
    assert e.pm.registered[0].side is S.SHORT


async def test_hold_makes_zero_broker_calls():
    e = Env()
    rep = await e.run(d=decision(DecisionAction.HOLD))
    assert rep.status is X.HOLD
    assert e.broker.calls == [] and e.rec.executions == [rep]


async def test_direction_mismatch_refused():
    e = Env()
    rep = await e.run(payload(S.LONG), decision(DecisionAction.SHORT))
    assert rep.status is X.REJECTED_BY_RISK and e.no_orders


# ------------------------------------------------------------------ risk
async def test_low_confidence_rejected_nothing_sent():
    e = Env()
    rep = await e.run(d=decision(confidence="0.3"))
    assert rep.status is X.REJECTED_BY_RISK
    assert any(r.startswith("LOW_CONFIDENCE") for r in rep.risk_reasons)
    assert e.no_orders
    assert not any(c.startswith(("set_", "create_order")) for c in e.broker.calls)
    assert not e.pm.registered and e.rec.executions == [rep]


async def test_fee_gate_rejected_nothing_sent():
    e = Env()
    rep = await e.run(d=decision(stop="49750", tp="50055"))
    assert rep.status is X.REJECTED_BY_RISK and any(r.startswith("FEE_GATE") for r in rep.risk_reasons)
    assert e.no_orders and not (await e.paper.fetch_positions())


async def test_leverage_comes_only_from_risk_decision():
    e = Env()
    rep = await e.run(d=decision(leverage=20))
    assert rep.status is X.OPENED_PROTECTED and rep.approved_leverage == 5
    assert (await e.paper.fetch_positions())[0].leverage == 5
    assert not any(c in ("set_leverage:20",) for c in e.broker.calls)


# ------------------------------------------------------------------ drift / staleness (rule 3)
async def test_adverse_drift_035_atr_aborts_without_orders():
    e = Env()
    e.set_market("50034.9", "50035")
    rep = await e.run()
    assert rep.status is X.ABORTED_PRICE_DRIFT and rep.drift_atr == D("0.35")
    assert e.no_orders and e.broker.calls == []  # not even an account fetch
    assert e.rec.executions == [rep]


async def test_short_adverse_drift_aborts():
    e = Env()
    e.set_market("49965", "49965.1")
    rep = await e.run(payload(S.SHORT), decision(DecisionAction.SHORT))
    assert rep.status is X.ABORTED_PRICE_DRIFT and e.no_orders


async def test_small_favourable_drift_executes():
    e = Env()
    e.set_market("49979.9", "49980")  # -0.2 ATR in our favour
    rep = await e.run()
    assert rep.status is X.OPENED_PROTECTED and rep.drift_atr == D("-0.2")


async def test_small_adverse_drift_within_limit_executes():
    e = Env()
    e.set_market("50019.9", "50020")
    rep = await e.run()
    assert rep.status is X.OPENED_PROTECTED and rep.drift_atr == D("0.2")


async def test_stale_book_aborts():
    e = Env()
    e.book_age = 3.0
    rep = await e.run()
    assert rep.status is X.ABORTED_STALE_DATA and e.no_orders and e.broker.calls == []


async def test_missing_book_aborts():
    e = Env()
    e.book_enabled = False
    rep = await e.run()
    assert rep.status is X.ABORTED_STALE_DATA and e.no_orders


async def test_book_provider_may_be_async_and_exceptions_are_stale():
    e = Env()

    async def boom(symbol):
        raise RuntimeError("feed down")

    eng = ExecutionEngine(e.broker, e.risk, e.ks, e.pm, e.rec, e.cfg, book_provider=boom, clock=lambda: NOW,
                          sleep=e._sleep)
    rep = await eng.execute(payload(), decision(), "d")
    assert rep.status is X.ABORTED_STALE_DATA and e.no_orders


# ------------------------------------------------------------------ naked position protection (rule 4)
async def test_stop_fails_three_times_closes_position_naked_closed():
    e = Env({OrderType.STOP_MARKET: 3})
    rep = await e.run()
    assert rep.status is X.NAKED_POSITION_CLOSED, rep.detail
    assert e.broker.attempts[OrderType.STOP_MARKET] == 3
    assert e.slept == [0.1, 0.2]  # backoff x attempt, no sleep after the last attempt
    assert not (await e.paper.fetch_positions()) and not e.resting()
    assert not e.broker.requests or not e.created(OrderType.TAKE_PROFIT_MARKET)  # TP never placed
    ev = e.rec.events_by_code("NAKED_POSITION_CLOSED")
    assert len(ev) == 1 and ev[0].level == "CRITICAL"
    assert not e.pm.registered and not e.ks.is_engaged
    assert "EXIT" in e.rec.purposes() and "STOP_LOSS" not in e.rec.purposes()
    assert any(o.reduce_only and o.order_type is OrderType.MARKET for o in rep.orders)


async def test_stop_fails_twice_then_succeeds():
    e = Env({OrderType.STOP_MARKET: 2})
    rep = await e.run()
    assert rep.status is X.OPENED_PROTECTED
    assert e.broker.attempts[OrderType.STOP_MARKET] == 3
    assert len(e.orders_of(OrderType.STOP_MARKET)) == 1 and e.slept == [0.1, 0.2]
    assert not e.rec.events_by_code("NAKED_POSITION_CLOSED")


async def test_stop_and_close_both_fail_engages_kill_switch():
    e = Env({OrderType.STOP_MARKET: -1}, fail_close=-1)
    rep = await e.run()
    assert rep.status is X.NAKED_CLOSE_FAILED, rep.detail
    assert e.broker.close_attempts == 3 and e.ks.is_engaged
    assert e.ks.reason.startswith("NAKED_POSITION_UNPROTECTED")
    assert e.rec.events_by_code("NAKED_CLOSE_FAILED")[0].level == "CRITICAL"
    assert e.sink.by_code("KILL_SWITCH_ENGAGED")
    assert not e.pm.registered


async def test_emergency_close_that_does_not_flatten_is_not_trusted():
    # close_position "succeeds" once failures are exhausted -> flat is verified; with 2 failures the 3rd attempt works
    e = Env({OrderType.STOP_MARKET: -1}, fail_close=2)
    rep = await e.run()
    assert rep.status is X.NAKED_POSITION_CLOSED and e.broker.close_attempts == 3
    assert not (await e.paper.fetch_positions()) and not e.ks.is_engaged


async def test_take_profit_failure_is_not_fatal():
    e = Env({OrderType.TAKE_PROFIT_MARKET: -1})
    rep = await e.run()
    assert rep.status is X.OPENED_PROTECTED
    assert any(w.startswith("TAKE_PROFIT_FAILED") for w in rep.warnings)
    assert [ev.level for ev in e.rec.events_by_code("TAKE_PROFIT_FAILED")] == ["WARN"]
    assert len(e.orders_of(OrderType.STOP_MARKET)) == 1 and not e.orders_of(OrderType.TAKE_PROFIT_MARKET)
    mp = e.pm.registered[0]
    assert mp.take_profit is None and mp.take_profit_order_id is None and rep.take_profit is None
    assert e.rec.purposes() == ["ENTRY", "STOP_LOSS"]


async def test_recorder_failure_does_not_break_protection():
    e = Env()

    async def bad_open_trade(position, leverage):
        raise RuntimeError("db down")

    e.rec.open_trade = bad_open_trade
    rep = await e.run()
    assert rep.status is X.OPENED_PROTECTED and any("OPEN_TRADE_RECORD_FAILED" in w for w in rep.warnings)
    assert e.pm.registered and len(e.orders_of(OrderType.STOP_MARKET)) == 1


async def test_unexpected_exception_with_open_unprotected_position_emergency_closes():
    e = Env()

    async def boom(req):
        raise RuntimeError("bug between fill and stop")

    e.engine._place_stop = boom  # noqa: SLF001 - simulate an unforeseen crash right after the entry filled
    rep = await e.run()
    assert rep.status is X.ERROR and any("EMERGENCY_CLOSED" in w for w in rep.warnings)
    assert not (await e.paper.fetch_positions()) and not e.resting()
    assert e.rec.events_by_code("NAKED_POSITION_CLOSED")[0].level == "CRITICAL"


async def test_unexpected_exception_when_position_unverifiable_engages_kill_switch():
    e = Env()

    async def boom(req):
        raise RuntimeError("bug")

    e.engine._place_stop = boom  # noqa: SLF001
    orig = e.broker.fetch_positions
    calls = {"n": 0}

    async def flaky_positions():
        calls["n"] += 1
        if calls["n"] > 1:  # first call is the entry settlement
            raise RuntimeError("exchange unreachable")
        return await orig()

    e.broker.fetch_positions = flaky_positions
    rep = await e.run()
    assert rep.status is X.ERROR and e.ks.is_engaged and e.ks.reason.startswith("POSITION_UNVERIFIABLE")


async def test_exception_after_entry_triggers_emergency_close():
    e = Env()
    orig = e.broker.create_order

    async def explode_on_stop(req):
        if req.order_type is OrderType.STOP_MARKET:
            raise asyncio.TimeoutError()  # not a BrokerError; still handled by the retry loop
        return await orig(req)

    e.broker.create_order = explode_on_stop
    rep = await e.run()
    assert rep.status is X.NAKED_POSITION_CLOSED
    assert not (await e.paper.fetch_positions())


async def test_unexpected_exception_status_error_with_position_check():
    e = Env()

    async def boom(*a, **k):
        raise RuntimeError("db down")

    e.rec.record_order = boom  # swallowed by the engine -> normal success
    rep = await e.run()
    assert rep.status is X.OPENED_PROTECTED

    e2 = Env()
    e2.pm.register = boom  # register failure: warning + CRITICAL event, position stays protected
    rep2 = await e2.run()
    assert rep2.status is X.OPENED_PROTECTED
    assert e2.rec.events_by_code("POSITION_MANAGER_REGISTER_FAILED")[0].level == "CRITICAL"


async def test_error_before_entry_returns_error_without_raising():
    e = Env()

    async def boom(symbol, lev):
        raise RuntimeError("exchange says no")

    e.broker.set_leverage = boom
    rep = await e.run()
    assert rep.status is X.ERROR and e.no_orders
    assert any("UNEXPECTED_ERROR" in w for w in rep.warnings) and e.rec.executions == [rep]


# ------------------------------------------------------------------ LIMIT entries
async def test_limit_entry_not_filled_is_cancelled():
    e = Env()
    rep = await e.run(d=decision(entry_type=EntryType.LIMIT, entry_price=D("49950")))
    assert rep.status is X.ENTRY_NOT_FILLED, rep.detail
    assert not (await e.paper.fetch_positions()) and not e.resting()
    assert e.created(OrderType.LIMIT)[0].price == D("49950")
    assert sum(e.slept) >= e.cfg.limit_entry_timeout_seconds - 1e-9
    assert not e.created(OrderType.STOP_MARKET) and not e.pm.registered
    entry = e.paper.orders["paper-1"]
    assert entry.status is OrderStatus.CANCELED and rep.filled_quantity == 0


async def test_limit_entry_fills_while_polling():
    e = Env()
    state = {"n": 0}
    inner_sleep = e._sleep

    async def sleep(s):
        await inner_sleep(s)
        state["n"] += 1
        if state["n"] == 3:
            e.set_market("49939.9", "49940")  # ask crosses the 49950 limit -> maker fill at 49950

    e.engine._sleep = sleep
    rep = await e.run(d=decision(entry_type=EntryType.LIMIT, entry_price=D("49950")))
    assert rep.status is X.OPENED_PROTECTED, rep.detail
    assert rep.entry_price == D("49950") and rep.filled_quantity == rep.approved_quantity
    assert len(e.orders_of(OrderType.STOP_MARKET)) == 1 and e.pm.registered
    assert state["n"] == 3  # stopped polling as soon as the fill was seen


async def test_limit_partial_fill_protects_only_filled_quantity():
    e = Env()
    state = {"n": 0}
    inner_sleep = e._sleep
    half = D("0.020")

    async def sleep(s):
        await inner_sleep(s)
        state["n"] += 1
        if state["n"] == 2:  # exchange fills part of the resting limit
            e.paper._apply_fill(SPEC, Side.BUY, half, D("49950"), D(0), False)  # noqa: SLF001

    e.engine._sleep = sleep
    rep = await e.run(d=decision(entry_type=EntryType.LIMIT, entry_price=D("49950")))
    assert rep.status is X.OPENED_PROTECTED, rep.detail
    assert rep.filled_quantity == half < rep.approved_quantity
    assert any(w.startswith("PARTIAL_FILL") for w in rep.warnings)
    stop, = e.orders_of(OrderType.STOP_MARKET)
    assert stop.quantity == half and e.pm.registered[0].quantity == half
    assert not e.orders_of(OrderType.LIMIT)  # remainder cancelled
    assert (await e.paper.fetch_positions())[0].quantity == half


# ------------------------------------------------------------------ CLOSE / kill switch / concurrency
async def test_close_decision_closes_and_cancels():
    e = Env()
    assert (await e.run()).status is X.OPENED_PROTECTED
    assert e.resting()
    rep = await e.run(d=decision(DecisionAction.CLOSE), did="dec-2")
    assert rep.status is X.CLOSED_BY_DECISION, rep.detail
    assert not (await e.paper.fetch_positions()) and not e.resting()
    assert "EXIT" in e.rec.purposes() and rep.decision_id == "dec-2"
    assert any(c.startswith("close_position") for c in e.broker.calls) and "cancel_all_orders" in e.broker.calls


async def test_close_with_nothing_open():
    e = Env()
    rep = await e.run(d=decision(DecisionAction.CLOSE))
    assert rep.status is X.NOTHING_TO_CLOSE and e.no_orders


async def test_close_failure_keeps_stop_and_reports_error():
    e = Env()
    assert (await e.run()).status is X.OPENED_PROTECTED
    e.broker.close_failures_remaining = -1
    rep = await e.run(d=decision(DecisionAction.CLOSE), did="dec-2")
    assert rep.status is X.ERROR
    assert len(e.orders_of(OrderType.STOP_MARKET)) == 1  # stop NOT cancelled while the close is failing
    assert e.rec.events_by_code("CLOSE_FAILED")


async def test_kill_switch_blocks_entries():
    e = Env()
    await e.ks.engage("test halt")
    rep = await e.run()
    assert rep.status is X.KILL_SWITCH and "test halt" in rep.detail
    assert e.no_orders and not any(c.startswith(("set_", "create_order")) for c in e.broker.calls)


async def test_close_allowed_while_kill_switch_engaged():
    e = Env()
    assert (await e.run()).status is X.OPENED_PROTECTED
    # engaging flattens by itself; a later CLOSE finds nothing, which must not raise
    await e.ks.engage("halt")
    rep = await e.run(d=decision(DecisionAction.CLOSE))
    assert rep.status is X.NOTHING_TO_CLOSE


async def test_already_managed_refused():
    e = Env()
    assert (await e.run()).status is X.OPENED_PROTECTED
    n_orders = len(e.broker.requests)
    rep = await e.run()
    assert rep.status is X.REJECTED_BY_RISK and any(r.startswith("ALREADY_MANAGED") for r in rep.risk_reasons)
    assert len(e.broker.requests) == n_orders


async def test_concurrent_executions_on_same_symbol_second_refused():
    e = Env()
    inner = e._book

    async def slow_book(symbol):
        await asyncio.sleep(0.01)  # a real suspension point so both tasks overlap
        return inner(symbol)

    eng = ExecutionEngine(e.broker, e.risk, e.ks, e.pm, e.rec, e.cfg, book_provider=slow_book, clock=lambda: NOW,
                          sleep=e._sleep)
    r1, r2 = await asyncio.gather(eng.execute(payload(), decision(), "a"), eng.execute(payload(), decision(), "b"))
    assert r1.status is X.OPENED_PROTECTED
    assert r2.status is X.REJECTED_BY_RISK and any(r.startswith("EXECUTION_IN_PROGRESS") for r in r2.risk_reasons)
    assert len(e.created(OrderType.MARKET)) == 1 and len(await e.paper.fetch_positions()) == 1


async def test_null_recorder_works():
    e = Env()
    eng = ExecutionEngine(e.broker, e.risk, e.ks, e.pm, NullTradeRecorder(), e.cfg, book_provider=e._book,
                          clock=lambda: NOW, sleep=e._sleep)
    rep = await eng.execute(payload(), decision(), None)
    assert rep.status is X.OPENED_PROTECTED and rep.trade_id is None
