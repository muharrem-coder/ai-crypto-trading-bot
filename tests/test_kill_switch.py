from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from app.config.settings import RiskSettings
from app.core.errors import KillSwitchEngaged, RiskViolation
from app.risk.kill_switch import KillSwitch
from app.risk.memory import InMemoryKillSwitchStore, InMemoryRiskEventSink
from app.risk.testing import FakeBroker, make_position, make_snapshot


class Clock:
    def __init__(self):
        self.now = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def build(broker=None, store=None, clock=None, **kw):
    broker = broker or FakeBroker(positions=[make_position()])
    store = store or InMemoryKillSwitchStore()
    sink = InMemoryRiskEventSink()
    clock = clock or Clock()
    sleeps = []

    async def sleep(s):
        sleeps.append(s)

    ks = KillSwitch(broker, store, sink, RiskSettings(), clock=clock, sleep=sleep, **kw)
    return ks, broker, store, sink, clock, sleeps


async def test_below_threshold_does_not_engage():
    ks, broker, *_ = build()
    await ks.evaluate(make_snapshot("10000"))
    st = await ks.evaluate(make_snapshot("9701"))
    assert not st.engaged and st.daily_pnl == D(-299)
    ks.assert_trading_allowed()
    assert "close_all_positions" not in broker.calls


async def test_engages_at_exactly_three_percent_and_flattens():
    ks, broker, store, sink, *_ = build()
    await ks.evaluate(make_snapshot("10000"))
    st = await ks.evaluate(make_snapshot("9700"))
    assert st.engaged and st.daily_drawdown_pct == D("0.03") and st.limit == D("0.03")
    assert broker.positions == []
    assert broker.calls.index("cancel_all_orders") < broker.calls.index("close_all_positions")
    assert ks.flat_verified is True
    assert sink.by_code("KILL_SWITCH_ENGAGED") and sink.by_level("CRITICAL")
    with pytest.raises(KillSwitchEngaged):
        ks.assert_trading_allowed()
    assert (await store.load()).engaged


async def test_engage_is_idempotent():
    ks, broker, store, sink, *_ = build()
    assert await ks.engage("A") is True
    assert await ks.engage("B") is False
    assert ks.reason == "A" and broker.calls.count("close_all_positions") == 1
    assert len(sink.by_code("KILL_SWITCH_ENGAGED")) == 1


async def test_margin_ratio_triggers():
    ks, *_ = build()
    st = await ks.evaluate(make_snapshot("10000", margin_ratio="0.8"))
    assert st.engaged and "MARGIN_RATIO" in ks.reason


async def test_consecutive_losses_trigger_and_win_resets():
    ks, broker, *_ = build()
    for _ in range(4):
        await ks.record_trade_result(D(-1))
    await ks.record_trade_result(D(5))
    assert ks.state.consecutive_losses == 0
    for _ in range(5):
        await ks.record_trade_result(D(-1))
    assert ks.is_engaged and "CONSECUTIVE_LOSSES" in ks.reason
    assert broker.positions == []


async def test_close_retries_with_backoff_then_succeeds():
    broker = FakeBroker(positions=[make_position()])
    broker.close_all_noop_attempts = 2
    ks, _, _, sink, _, sleeps = build(broker)
    await ks.engage("T")
    assert broker.calls.count("close_all_positions") == 3
    assert sleeps == [1.0, 2.0]
    assert ks.flat_verified is True and not sink.by_code("KILL_SWITCH_NOT_FLAT")


async def test_not_flat_logs_critical_and_next_evaluate_retries():
    broker = FakeBroker(positions=[make_position()])
    broker.close_all_noop_attempts = 3
    ks, _, _, sink, *_ = build(broker)
    await ks.engage("T")
    assert broker.calls.count("close_all_positions") == 3
    assert ks.flat_verified is False and sink.by_code("KILL_SWITCH_NOT_FLAT")
    await ks.evaluate(make_snapshot(positions=tuple(broker.positions)))  # still engaged, tries again
    assert broker.positions == [] and ks.flat_verified is True


async def test_close_exception_is_survived():
    broker = FakeBroker(positions=[make_position()])
    broker.close_all_raises = True
    ks, _, _, sink, *_ = build(broker)
    await ks.engage("T")
    assert ks.is_engaged and ks.flat_verified is False
    assert sink.by_code("KILL_SWITCH_CLOSE_FAILED")


async def test_state_persists_across_restart():
    ks, _, store, _, clock, _ = build()
    await ks.evaluate(make_snapshot("10000"))
    await ks.evaluate(make_snapshot("9600"))
    ks2, *_ = build(FakeBroker(), store, clock)  # new process, same store
    assert not ks2.is_engaged
    await ks2.start()
    assert ks2.is_engaged
    with pytest.raises(KillSwitchEngaged):
        ks2.assert_trading_allowed()
    st = await ks2.evaluate(make_snapshot("10000"))  # even with recovered equity it stays engaged
    assert st.engaged


async def test_manual_reset_requires_confirm():
    ks, *_ = build()
    await ks.engage("T")
    for bad in ("reset", "", "YES"):
        with pytest.raises(RiskViolation):
            await ks.manual_reset("muharrem", bad)
        assert ks.is_engaged
    with pytest.raises(RiskViolation):
        await ks.manual_reset("  ", "RESET")
    await ks.manual_reset("muharrem", "RESET")
    assert not ks.is_engaged
    ks.assert_trading_allowed()
    # baseline rebased: old drawdown must not instantly re-trigger
    st = await ks.evaluate(make_snapshot("9000"))
    assert not st.engaged


async def test_no_auto_reset_next_day_by_default():
    ks, _, _, _, clock, _ = build()
    await ks.engage("T")
    clock.now += timedelta(days=1)
    st = await ks.evaluate(make_snapshot())
    assert st.engaged


async def test_auto_reset_next_utc_day_only():
    ks, _, _, _, clock, _ = build(auto_reset_next_day=True)
    clock.now = datetime(2026, 3, 1, 23, 59, tzinfo=timezone.utc)
    await ks.engage("T")
    clock.now = datetime(2026, 3, 1, 23, 59, 59, tzinfo=timezone.utc)
    assert (await ks.evaluate(make_snapshot())).engaged
    clock.now = datetime(2026, 3, 2, 0, 0, 1, tzinfo=timezone.utc)
    st = await ks.evaluate(make_snapshot("9500"))
    assert not st.engaged and st.day_start_equity == D(9500)


async def test_utc_day_rollover_rebases_daily_baseline():
    ks, _, _, _, clock, _ = build()
    await ks.evaluate(make_snapshot("10000"))
    clock.now += timedelta(days=1)
    st = await ks.evaluate(make_snapshot("9800"))
    assert st.daily_pnl == 0 and st.day_start_equity == D(9800)


async def test_profit_gives_negative_drawdown_and_peak_tracked():
    ks, *_ = build()
    await ks.evaluate(make_snapshot("10000"))
    await ks.evaluate(make_snapshot("10500"))
    st = await ks.evaluate(make_snapshot("10200"))
    assert st.daily_drawdown_pct < 0 and st.peak_equity == D(10500)
    assert st.peak_drawdown_pct == D(300) / D(10500)
