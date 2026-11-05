import asyncio
from decimal import Decimal as D

from app.config.settings import RiskSettings
from app.core.types import OrderStatus
from app.risk.kill_switch import KillSwitch
from app.risk.memory import InMemoryKillSwitchStore, InMemoryRiskEventSink
from app.risk.testing import FakeBroker, make_order, make_position
from app.risk.watchdog import RiskWatchdog


def build(broker, repair=None, **kw):
    sink = InMemoryRiskEventSink()
    settings = RiskSettings(watchdog_interval_seconds=0.01)

    async def nosleep(_):
        return None

    ks = KillSwitch(broker, InMemoryKillSwitchStore(), sink, settings, sleep=nosleep)
    return RiskWatchdog(broker, ks, sink, settings, repair, **kw), ks, sink


async def test_protected_position_untouched():
    b = FakeBroker(positions=[make_position()], orders=[make_order()])
    wd, ks, sink = build(b)
    r = await wd.run_once()
    assert r.ok and not r.repaired and not r.closed and len(b.positions) == 1
    assert not sink.by_level("CRITICAL")


async def test_short_needs_buy_side_stop():
    from app.core.types import PositionSide, Side
    pos = make_position(side=PositionSide.SHORT)
    b = FakeBroker(positions=[pos], orders=[make_order(side=Side.SELL)])  # wrong side for a short
    wd, *_ = build(b)
    r = await wd.run_once()
    assert r.closed == [pos.symbol]


async def test_non_reduce_only_or_terminal_stop_does_not_count():
    b = FakeBroker(positions=[make_position()],
                   orders=[make_order(reduce_only=False), make_order(status=OrderStatus.CANCELED)])
    wd, *_ = build(b)
    assert (await wd.run_once()).closed == ["BTC/USDT:USDT"]


async def test_missing_stop_is_repaired():
    b = FakeBroker(positions=[make_position()])
    calls = []

    async def repair(pos):
        calls.append(pos.symbol)
        b.orders.append(make_order(symbol=pos.symbol))

    wd, ks, sink = build(b, repair)
    r = await wd.run_once()
    assert calls == ["BTC/USDT:USDT"] and r.repaired == ["BTC/USDT:USDT"] and not r.closed
    assert len(b.positions) == 1
    assert sink.by_code("UNPROTECTED_POSITION") and sink.by_code("STOP_REPAIRED")


async def test_repair_failure_flattens_position():
    b = FakeBroker(positions=[make_position()])

    async def repair(pos):
        raise RuntimeError("exchange rejected stop")

    wd, ks, sink = build(b, repair)
    r = await wd.run_once()
    assert r.closed == ["BTC/USDT:USDT"] and b.positions == []
    assert sink.by_code("STOP_REPAIR_FAILED") and sink.by_code("UNPROTECTED_POSITION_CLOSED")
    assert all(e.level == "CRITICAL" for e in sink.by_code("UNPROTECTED_POSITION"))


async def test_repair_that_places_nothing_also_flattens():
    b = FakeBroker(positions=[make_position()])

    async def repair(pos):
        return None

    wd, *_ = build(b, repair)
    assert (await wd.run_once()).closed == ["BTC/USDT:USDT"]


async def test_no_repair_callback_flattens():
    b = FakeBroker(positions=[make_position()])
    wd, *_ = build(b)
    await wd.run_once()
    assert b.positions == []


async def test_close_failure_engages_kill_switch():
    b = FakeBroker(positions=[make_position()])
    b.close_position_fails = True
    wd, ks, sink = build(b)
    await wd.run_once()
    assert ks.is_engaged and "UNPROTECTED" in ks.reason
    assert sink.by_code("UNPROTECTED_CLOSE_FAILED")


async def test_drawdown_engages_via_watchdog():
    b = FakeBroker(equity="10000", positions=[make_position()], orders=[make_order()])
    wd, ks, _ = build(b)
    await wd.run_once()
    b.equity = D(9600)
    r = await wd.run_once()
    assert r.engaged and ks.is_engaged and b.positions == []


async def test_broker_failures_counted_then_engage():
    b = FakeBroker()
    b.fail_account = True
    wd, ks, sink = build(b, max_consecutive_broker_failures=3)
    for _ in range(2):
        r = await wd.run_once()
        assert not r.ok
    assert not ks.is_engaged and wd.consecutive_broker_failures == 2
    await wd.run_once()
    assert ks.is_engaged and ks.reason == "BROKER_UNREACHABLE"
    assert wd.error_count == 3


async def test_success_resets_failure_counter():
    b = FakeBroker()
    wd, ks, _ = build(b, max_consecutive_broker_failures=3)
    b.fail_account = True
    await wd.run_once()
    await wd.run_once()
    b.fail_account = False
    await wd.run_once()
    assert wd.consecutive_broker_failures == 0
    b.fail_account = True
    await wd.run_once()
    assert not ks.is_engaged


async def test_tick_exception_is_contained():
    b = FakeBroker(positions=[make_position()], orders=[make_order()])
    wd, ks, sink = build(b)

    async def boom(_):
        raise ValueError("bug")

    ks.evaluate = boom  # type: ignore[assignment]
    r = await wd.run_once()
    assert not r.ok and wd.error_count == 1 and sink.by_code("WATCHDOG_TICK_FAILED")


async def test_start_stop_loop():
    b = FakeBroker(positions=[make_position()], orders=[make_order()])
    wd, *_ = build(b)
    wd.start()
    assert wd.running
    await asyncio.sleep(0.08)
    await wd.stop()
    assert not wd.running and wd.ticks >= 2
