"""Exchange reconciliation against the REAL paper broker, PositionManager and ExecutionEngine (no network)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal as D
from typing import Any

import pytest
from pydantic import SecretStr

from app.brokers.paper_adapter import PaperBrokerAdapter
from app.claude.transport import ScriptedTransport
from app.config.settings import (
    ApiSettings,
    DataSettings,
    ExecutionSettings,
    OrphanAction,
    ReconciliationSettings,
    RiskSettings,
    Settings,
    LIVE_CONFIRMATION_PHRASE,
)
from app.core.errors import BrokerError
from app.core.events import EventBus, Topic
from app.core.types import MarginMode, OrderRequest, OrderType, Position, PositionSide, Side, TradingMode
from app.data.mock_source import ScriptedMarketSource
from app.db import repository
from app.db.session import get_session
from app.execution.contracts import ManagedPosition
from app.execution.executor import ExecutionEngine
from app.execution.live_guard import LiveGuardError
from app.execution.position_manager import PositionManager
from app.execution.reconciliation import ReconciliationService
from app.execution.testing import FlakyBroker, InMemoryTradeRecorder
from app.risk.kill_switch import KillSwitch
from app.risk.memory import InMemoryKillSwitchStore, InMemoryRiskEventSink
from app.risk.risk_engine import HardRiskEngine
from app.risk.testing import make_spec
from app.runtime import build_runtime

pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

SYM = "BTC/USDT:USDT"
SPEC = make_spec(SYM)
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MEM_DB = "sqlite+aiosqlite:///:memory:"
BID, ASK, MARK = D("49999.9"), D("50000.1"), D("50000")


class CrossPaper(PaperBrokerAdapter):
    """Paper exchange that reports every position as CROSS margin."""

    async def fetch_positions(self) -> list[Position]:
        out = await super().fetch_positions()
        for p in out:
            p.margin_mode = MarginMode.CROSS
        return out


class BrokenReads(PaperBrokerAdapter):
    fail_positions = False
    fail_orders = False

    async def fetch_positions(self) -> list[Position]:
        if self.fail_positions:
            raise BrokerError("network blip (positions)")
        return await super().fetch_positions()

    async def fetch_open_orders(self, symbol: str | None = None):
        if self.fail_orders:
            raise BrokerError("network blip (orders)")
        return await super().fetch_open_orders(symbol)


def make_paper(cls: type[PaperBrokerAdapter] = PaperBrokerAdapter) -> PaperBrokerAdapter:
    paper = cls({SYM: SPEC}, D(10000), "USDT", D(0), now=lambda: NOW)  # zero slippage: entry == ask / bid
    paper.update_price(SYM, BID, ASK, mark=MARK)
    return paper


class Env:
    def __init__(self, *, paper: PaperBrokerAdapter | None = None, fail_orders: dict[OrderType, int] | None = None,
                 fail_close: int = 0, **recon: Any) -> None:
        self.paper = paper or make_paper()
        self.broker = FlakyBroker(self.paper, fail_orders or {}, fail_close_position=fail_close)
        self.slept: list[float] = []
        self.exec_cfg = ExecutionSettings(stop_retry_backoff_seconds=0)
        self.settings = Settings(_env_file=None, symbols=[SYM], execution=self.exec_cfg,
                                 recon=ReconciliationSettings(**recon))
        self.sink = InMemoryRiskEventSink()
        rs = RiskSettings()
        self.ks = KillSwitch(self.paper, InMemoryKillSwitchStore(), self.sink, rs, clock=lambda: NOW,
                             sleep=self._sleep)
        self.rec = InMemoryTradeRecorder()
        self.bus = EventBus()
        self.pm = PositionManager(self.broker, self.rec, self.exec_cfg, rs, kill_switch=self.ks, clock=lambda: NOW,
                                  sleep=self._sleep)
        self.executor = ExecutionEngine(self.broker, HardRiskEngine(rs, self.ks, self.sink), self.ks, self.pm,
                                        self.rec, self.exec_cfg, book_provider=lambda s: None, clock=lambda: NOW,
                                        sleep=self._sleep)
        self.service = ReconciliationService(self.broker, self.pm, self.executor, self.rec, self.ks, self.bus,
                                             self.settings, clock=lambda: NOW, sleep=self._sleep)
        self.broker.calls.clear()

    async def _sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        await asyncio.sleep(0)

    # ---- exchange-side setup (goes straight to the paper exchange: not counted by the FlakyBroker)
    async def open_position(self, side: PositionSide = PositionSide.LONG, qty: str = "0.1", stop: str | None = None,
                            tp: str | None = None) -> tuple[Any, Any]:
        req_side = Side.BUY if side is PositionSide.LONG else Side.SELL
        await self.paper.create_order(OrderRequest(SYM, req_side, OrderType.MARKET, D(qty)))
        stop_o = tp_o = None
        if stop is not None:
            stop_o = await self.paper.create_order(OrderRequest(SYM, side.exit_side, OrderType.STOP_MARKET, D(qty),
                                                                stop_price=D(stop), reduce_only=True))
        if tp is not None:
            tp_o = await self.paper.create_order(OrderRequest(SYM, side.exit_side, OrderType.TAKE_PROFIT_MARKET,
                                                              D(qty), stop_price=D(tp), reduce_only=True))
        return stop_o, tp_o

    async def open_managed(self, side: PositionSide = PositionSide.LONG, qty: str = "0.1") -> ManagedPosition:
        stop, tp = ("49000", "52000") if side is PositionSide.LONG else ("51000", "48000")
        stop_o, tp_o = await self.open_position(side, qty, stop, tp)
        pos = (await self.paper.fetch_positions())[0]
        mp = ManagedPosition(symbol=SYM, side=side, quantity=pos.quantity, entry_price=pos.entry_price,
                             initial_stop=D(stop), take_profit=D(tp), atr=D(100), fees=SPEC.fees,
                             tick_size=SPEC.tick_size, stop_order_id=stop_o.order_id,
                             take_profit_order_id=tp_o.order_id, decision_id=None, trade_id=None, opened_at=NOW)
        mp.trade_id = await self.rec.open_trade(mp, 3)
        await self.pm.register(mp)
        return mp

    def resting(self, order_type: OrderType | None = None) -> list[Any]:
        return [r.order for r in self.paper._resting.values()  # noqa: SLF001
                if order_type is None or r.order.order_type is order_type]

    def stops(self) -> list[Any]:
        return self.resting(OrderType.STOP_MARKET)

    async def positions(self) -> list[Position]:
        return await self.paper.fetch_positions()


# ====================================================================== restart scenario
async def test_restart_adopts_position_and_reuses_existing_stop():
    env = Env()
    stop_o, tp_o = await env.open_position(PositionSide.LONG, "0.1", stop="49000", tp="52000")
    assert not env.pm.is_managed(SYM)  # "fresh process": nothing tracked

    report = await env.service.run_startup()

    assert report.ok and report.adopted == [SYM] and report.stops_placed == [] and report.closed == []
    managed = env.pm.positions()[0]
    assert env.pm.is_managed(SYM)
    assert managed.stop_order_id == stop_o.order_id and managed.take_profit_order_id == tp_o.order_id
    assert managed.current_stop == D("49000") and managed.take_profit == D("52000")
    assert managed.quantity == D("0.1") and managed.side is PositionSide.LONG
    assert managed.atr == D("1000.1") / D("1.5")  # (entry 50000.1 - stop 49000) / atr_stop_multiple
    assert env.broker.attempts[OrderType.STOP_MARKET] == 0  # nothing placed ...
    assert [o.order_id for o in env.stops()] == [stop_o.order_id]  # ... exactly one stop rests, no duplicate
    assert len(env.rec.opened) == 1  # no DB trade known -> a new one was opened
    assert env.service.last_report is report


async def test_second_pass_after_adoption_is_clean_and_idempotent():
    env = Env()
    await env.open_position(PositionSide.LONG, "0.1", stop="49000", tp="52000")
    await env.service.run_once("startup")
    again = await env.service.run_once()
    assert again.clean and again.ok and again.level == "INFO"
    assert len(env.stops()) == 1 and len(env.rec.opened) == 1


# ====================================================================== orphan without a stop
async def test_orphan_long_gets_stop_two_percent_below_entry_rounded_down():
    env = Env()
    await env.open_position(PositionSide.LONG, "0.1")  # entry == ask 50000.1, no stop
    assert env.stops() == []

    report = await env.service.run_once("startup")

    assert report.adopted == [SYM] and report.stops_placed == [SYM] and report.ok
    (stop,) = env.stops()
    # 50000.1 * 0.98 = 49000.098 -> floored to the 0.1 tick (away from entry)
    assert stop.stop_price == D("49000.0") and stop.reduce_only and stop.side is Side.SELL
    assert stop.quantity == D("0.1")
    managed = env.pm.positions()[0]
    assert managed.stop_order_id == stop.order_id and managed.current_stop == D("49000.0")
    assert managed.initial_stop == D("49000.0")
    assert managed.atr == (D("50000.1") - D("49000.0")) / D("1.5")
    assert report.level == "WARN"
    assert env.rec.events_by_code("RECON_ORPHAN_ADOPTED")


async def test_orphan_short_gets_stop_two_percent_above_entry_rounded_up():
    env = Env()
    await env.open_position(PositionSide.SHORT, "0.1")  # entry == bid 49999.9

    report = await env.service.run_once("startup")

    assert report.adopted == [SYM] and report.stops_placed == [SYM]
    (stop,) = env.stops()
    # 49999.9 * 1.02 = 50999.898 -> ceiled to the 0.1 tick (away from entry)
    assert stop.stop_price == D("50999.9") and stop.side is Side.BUY and stop.reduce_only
    assert env.pm.positions()[0].side is PositionSide.SHORT


async def test_orphan_is_protected_for_real_the_stop_fires():
    env = Env()
    await env.open_position(PositionSide.LONG, "0.1")
    await env.service.run_once("startup")
    env.paper.update_price(SYM, D("48899.9"), D("48900.1"), mark=D("48900"))  # through the 49000 stop
    assert await env.positions() == []


async def test_orphan_with_stop_that_would_trigger_immediately_is_closed():
    env = Env()
    await env.open_position(PositionSide.LONG, "0.1")
    env.paper.update_price(SYM, D("47999.9"), D("48000.1"), mark=D("48000"))  # market already below entry - 2%

    report = await env.service.run_once("startup")

    assert report.closed == [SYM] and report.adopted == [] and report.stops_placed == []
    assert await env.positions() == []
    assert env.broker.attempts[OrderType.STOP_MARKET] == 0
    assert env.rec.events_by_level("CRITICAL") and report.level == "CRITICAL"


async def test_oversized_orphan_is_closed_with_critical_event():
    env = Env(max_orphan_notional_pct=D("0.3"))
    await env.open_position(PositionSide.LONG, "0.1")  # ~5000 notional vs ~10000 equity = 0.5 > 0.3

    report = await env.service.run_once("startup")

    assert report.closed == [SYM] and report.adopted == []
    assert await env.positions() == []
    (evt,) = env.rec.events_by_code("RECON_SECURITY_CLOSE")
    assert evt.level == "CRITICAL" and "notional" in evt.message
    assert report.level == "CRITICAL" and not env.ks.is_engaged


async def test_cross_margin_orphan_is_closed():
    env = Env(paper=make_paper(CrossPaper))
    await env.open_position(PositionSide.LONG, "0.1", stop="49000")

    report = await env.service.run_once("startup")

    assert report.closed == [SYM] and report.adopted == []
    assert "CROSS" in env.rec.events_by_code("RECON_SECURITY_CLOSE")[0].message


async def test_cross_margin_allowed_when_setting_disabled():
    env = Env(paper=make_paper(CrossPaper), close_if_cross_margin=False)
    await env.open_position(PositionSide.LONG, "0.1", stop="49000")
    report = await env.service.run_once("startup")
    assert report.adopted == [SYM] and report.closed == []


async def test_orphan_action_close_flattens_everything_unmanaged():
    env = Env(orphan_action=OrphanAction.CLOSE)
    await env.open_position(PositionSide.LONG, "0.1", stop="49000")
    report = await env.service.run_once("startup")
    assert report.closed == [SYM] and await env.positions() == [] and env.stops() == []


# ====================================================================== naked-position escalation
async def test_stop_placement_failing_three_times_triggers_emergency_close():
    env = Env(fail_orders={OrderType.STOP_MARKET: 3})
    await env.open_position(PositionSide.LONG, "0.1")

    report = await env.service.run_once("startup")

    assert env.broker.attempts[OrderType.STOP_MARKET] == 3
    assert report.closed == [SYM] and report.adopted == [] and not env.pm.is_managed(SYM)
    assert await env.positions() == []
    assert env.rec.events_by_code("RECON_NAKED_POSITION")[0].level == "CRITICAL"
    assert not env.ks.is_engaged


async def test_stop_and_close_both_failing_engages_kill_switch():
    env = Env(fail_orders={OrderType.STOP_MARKET: -1}, fail_close=-1)
    await env.open_position(PositionSide.LONG, "0.1")

    report = await env.service.run_once("startup")

    assert env.ks.is_engaged and env.ks.reason.startswith("RECON_NAKED_POSITION")
    assert not report.ok and report.closed == [] and report.level == "CRITICAL"
    assert env.rec.events_by_code("RECON_NAKED_POSITION_FAILED")
    assert env.broker.close_attempts >= 1


# ====================================================================== managed positions
async def test_managed_position_whose_stop_vanished_gets_a_new_stop_at_current_stop():
    env = Env()
    mp = await env.open_managed()
    old_id = mp.stop_order_id
    await env.paper.cancel_order(old_id, SYM)  # stop vanished on the exchange
    assert env.stops() == []

    report = await env.service.run_once()

    assert report.stops_placed == [SYM] and report.ok and report.adopted == []
    (stop,) = env.stops()
    assert stop.stop_price == mp.current_stop == D("49000") and stop.reduce_only and stop.quantity == D("0.1")
    assert mp.stop_order_id == stop.order_id != old_id


async def test_quantity_change_replaces_stop_new_before_old_cancelled():
    env = Env()
    mp = await env.open_managed(qty="0.1")
    old_id = mp.stop_order_id
    await env.paper.create_order(OrderRequest(SYM, Side.SELL, OrderType.MARKET, D("0.04"), reduce_only=True))
    env.broker.calls.clear()

    report = await env.service.run_once()

    assert report.stops_placed == [SYM] and mp.quantity == D("0.06")
    (stop,) = env.stops()
    assert stop.quantity == D("0.06") and stop.order_id == mp.stop_order_id != old_id
    calls = env.broker.calls
    assert calls.index("create_order:STOP_MARKET") < calls.index("cancel_order")
    assert env.rec.events_by_code("RECON_STOP_REPLACED")


async def test_healthy_managed_position_is_left_alone():
    env = Env()
    await env.open_managed()
    env.broker.calls.clear()
    report = await env.service.run_once()
    assert report.clean
    assert not any(c.startswith("create_order") or c == "cancel_order" for c in env.broker.calls)


async def test_managed_stop_failure_closes_position():
    env = Env(fail_orders={OrderType.STOP_MARKET: -1})
    mp = await env.open_managed()
    await env.paper.cancel_order(mp.stop_order_id, SYM)

    report = await env.service.run_once()

    assert report.closed == [SYM] and await env.positions() == []


# ====================================================================== local state vs flat exchange
async def test_exchange_flat_but_manager_tracks_closes_local_trade():
    env = Env()
    mp = await env.open_managed()
    await env.paper.close_position(SYM)  # closed manually on the exchange (paper cancels reduce-only leftovers)
    assert env.pm.is_managed(SYM)

    report = await env.service.run_once()

    assert report.local_closed == [SYM] and not env.pm.is_managed(SYM)
    assert len(env.rec.closed) == 1 and env.rec.closed[0].trade_id == mp.trade_id
    assert report.level == "WARN" and report.ok


async def test_stale_reduce_only_orders_on_flat_symbol_are_cancelled():
    env = Env()
    await env.open_position(PositionSide.LONG, "0.1", stop="49000", tp="52000")
    env.paper._positions.pop(SYM)  # noqa: SLF001 - manual close that left SL/TP behind
    assert len(env.resting()) == 2

    report = await env.service.run_once()

    assert len(report.orphan_orders_cancelled) == 2 and env.resting() == []
    assert env.rec.events_by_code("RECON_STALE_ORDER_CANCELLED")


async def test_non_reduce_only_orders_on_flat_symbol_are_not_touched():
    env = Env()
    await env.paper.create_order(OrderRequest(SYM, Side.BUY, OrderType.LIMIT, D("0.01"), price=D("40000")))
    report = await env.service.run_once()
    assert report.orphan_orders_cancelled == [] and len(env.resting()) == 1


# ====================================================================== broker failures
@pytest.mark.parametrize("which", ["positions", "orders"])
async def test_broker_fetch_failure_marks_report_not_ok_and_closes_nothing(which):
    paper = make_paper(BrokenReads)
    env = Env(paper=paper)
    mp = await env.open_managed()
    setattr(paper, f"fail_{which}", True)
    env.broker.calls.clear()

    report = await env.service.run_once()

    assert report.ok is False and report.errors
    assert env.pm.is_managed(SYM) and env.rec.closed == []
    assert report.closed == [] and report.local_closed == [] and report.adopted == []
    assert not any(c.startswith(("create_order", "cancel", "close_position")) for c in env.broker.calls)
    assert mp.stop_order_id in {o.order_id for o in env.stops()}
    assert env.rec.events_by_code("RECON_BROKER_ERROR")


async def test_startup_retries_three_times_with_backoff_when_errors_remain():
    paper = make_paper(BrokenReads)
    env = Env(paper=paper)
    paper.fail_positions = True
    report = await env.service.run_startup()
    assert not report.ok
    assert env.service.runs == 4 and env.slept == [1.0, 2.0, 4.0]


async def test_startup_recovers_after_transient_failure():
    paper = make_paper(BrokenReads)
    env = Env(paper=paper)
    await env.open_position(PositionSide.LONG, "0.1", stop="49000")
    paper.fail_positions = True
    original = env._sleep

    async def heal(seconds: float) -> None:
        paper.fail_positions = False
        await original(seconds)

    env.service._sleep = heal  # noqa: SLF001
    report = await env.service.run_startup()
    assert report.ok and report.adopted == [SYM] and env.service.runs == 2


# ====================================================================== summary event + loop
async def test_summary_is_published_info_when_clean_and_warn_otherwise():
    env = Env()
    sub = env.bus.subscribe(Topic.RISK_EVENT)
    await env.service.run_once()
    clean = sub.get_nowait()
    assert clean["level"] == "INFO" and clean["code"] == "RECON_SUMMARY" and clean["details"]["ok"] is True

    await env.open_position(PositionSide.LONG, "0.1", stop="49000")
    await env.service.run_once()
    warn = sub.get_nowait()
    assert warn["level"] == "WARN" and warn["details"]["adopted"] == [SYM]


async def test_periodic_loop_survives_a_crashing_pass():
    env = Env()
    calls = {"n": 0}
    original = env.service.run_once

    async def flaky_run(reason: str = "periodic"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return await original(reason)

    env.service.run_once = flaky_run  # type: ignore[method-assign]
    env.service.start()
    env.service.start()  # idempotent
    for _ in range(200):
        if calls["n"] >= 3:
            break
        await asyncio.sleep(0)
    assert calls["n"] >= 3 and env.service.running and env.service.failed_runs == 1
    await env.service.stop()
    assert not env.service.running
    await env.service.stop()  # idempotent


# ====================================================================== real runtime (DB, start order, LIVE guard)
def runtime_settings(**kw: Any) -> Settings:
    kw.setdefault("risk", RiskSettings(watchdog_interval_seconds=30))
    return Settings(_env_file=None, symbols=[SYM],
                    data=DataSettings(stale_after_seconds=60, resume_after_live_seconds=0,
                                      health_check_interval_seconds=0.05),
                    execution=ExecutionSettings(manager_poll_seconds=30), **kw)


async def make_rt(settings: Settings, paper: PaperBrokerAdapter):
    return await build_runtime(settings, market_source=ScriptedMarketSource(), broker=paper,
                               transport=ScriptedTransport(), news_sources=[], database_url=MEM_DB)


async def settle(rt) -> None:
    """Let the first watchdog tick finish its DB writes (cancelling mid-DB-call on the shared in-memory SQLite
    connection is a test artefact, see tests/test_api_runtime.py)."""
    for _ in range(300):
        if rt.watchdog.ticks >= 1:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.15)


async def test_restart_with_fresh_runtime_adopts_existing_position_and_recovers_db_trade():
    paper = make_paper()
    await paper.create_order(OrderRequest(SYM, Side.BUY, OrderType.MARKET, D("0.1")))
    stop_o = await paper.create_order(OrderRequest(SYM, Side.SELL, OrderType.STOP_MARKET, D("0.1"),
                                                   stop_price=D("49000"), reduce_only=True))
    rt = await make_rt(runtime_settings(), paper)
    # the previous process left an OPEN trade row behind
    previous = ManagedPosition(symbol=SYM, side=PositionSide.LONG, quantity=D("0.1"), entry_price=D("50000.1"),
                               initial_stop=D("48800"), take_profit=None, atr=D(100), fees=SPEC.fees,
                               tick_size=SPEC.tick_size, stop_order_id="x", take_profit_order_id=None,
                               decision_id=None, trade_id=None, opened_at=NOW)
    trade_id = await rt.recorder.open_trade(previous, 3)
    assert trade_id
    try:
        await rt.start()
        await settle(rt)
        report = rt.reconciliation.last_report
        assert report is not None and report.ok and report.adopted == [SYM] and report.stops_placed == []
        managed = rt.position_manager.positions()[0]
        assert managed.trade_id == trade_id and managed.initial_stop == D("48800")  # recovered from the DB row
        assert managed.current_stop == D("49000") and managed.stop_order_id == stop_o.order_id
        assert len([o for o in await paper.fetch_open_orders() if o.order_type is OrderType.STOP_MARKET]) == 1
        async with get_session(rt.session_factory) as session:
            assert len(await repository.list_open_trades(session)) == 1  # no duplicate trade row
        assert rt.reconciliation.running
    finally:
        await rt.stop()
    assert not rt.reconciliation.running


async def test_runtime_start_reconciles_before_decision_service_and_in_paper_mode():
    paper = make_paper()
    await paper.create_order(OrderRequest(SYM, Side.BUY, OrderType.MARKET, D("0.1")))  # unprotected orphan
    rt = await make_rt(runtime_settings(), paper)
    order: list[str] = []
    original_start = rt.decision_service.start
    rt.decision_service.start = lambda: (order.append("decision_service"), original_start())[1]  # type: ignore[method-assign]
    original_run = rt.reconciliation.run_startup

    async def spy() -> Any:
        order.append("recon")
        return await original_run()

    rt.reconciliation.run_startup = spy  # type: ignore[method-assign]
    try:
        await rt.start()
        await settle(rt)
        assert order == ["recon", "decision_service"]
        assert [o.order_type for o in await paper.fetch_open_orders()] == [OrderType.STOP_MARKET]  # protected
        assert "reconciliation" not in rt.started_components  # existing component list is unchanged
    finally:
        await rt.stop()


async def test_db_trade_without_exchange_position_is_closed_on_startup():
    paper = make_paper()
    rt = await make_rt(runtime_settings(), paper)
    ghost = ManagedPosition(symbol=SYM, side=PositionSide.LONG, quantity=D("0.1"), entry_price=D("50000"),
                            initial_stop=D("49000"), take_profit=None, atr=D(100), fees=SPEC.fees,
                            tick_size=SPEC.tick_size, stop_order_id="x", take_profit_order_id=None,
                            decision_id=None, trade_id=None, opened_at=NOW)
    await rt.recorder.open_trade(ghost, 3)
    try:
        await rt.start()
        await settle(rt)
        assert rt.reconciliation.last_report.local_closed == [SYM]
        async with get_session(rt.session_factory) as session:
            assert await repository.list_open_trades(session) == []
    finally:
        await rt.stop()


def live_settings(**kw: Any) -> Settings:
    kw.setdefault("risk", RiskSettings(max_leverage=5, max_daily_drawdown_pct=D("0.03"), watchdog_interval_seconds=30))
    return runtime_settings(trading_mode=TradingMode.LIVE, live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
                            exchange_api_key=SecretStr("key"), exchange_api_secret=SecretStr("secret"),
                            api=ApiSettings(control_token=SecretStr("t" * 16)), **kw)


async def test_runtime_start_aborts_in_live_when_guard_fails():
    paper = PaperBrokerAdapter({SYM: SPEC}, D(10), "USDT", D(0), now=lambda: NOW)  # 10 USDT < live_min_balance 50
    paper.update_price(SYM, BID, ASK, mark=MARK)
    rt = await make_rt(live_settings(), paper)
    with pytest.raises(LiveGuardError, match="min_balance"):
        await rt.start()
    assert not rt.is_running and rt.started_components == []
    assert rt.reconciliation.last_report is None  # nothing reconciled / traded after a refused start
    assert rt.position_manager._task is None and not rt.watchdog.running  # noqa: SLF001


async def test_runtime_starts_in_live_with_healthy_broker():
    paper = make_paper()
    rt = await make_rt(live_settings(), paper)
    try:
        await rt.start()
        await settle(rt)
        assert rt.is_running and rt.reconciliation.last_report.ok
    finally:
        await rt.stop()
