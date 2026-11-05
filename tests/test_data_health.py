"""Feed health + TradingGate (rule R1) with an injected fake clock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.config.settings import DataSettings
from app.core.events import EventBus, Topic
from app.data.health import FeedHealthMonitor, candle_stale_after, timeframe_seconds
from app.data.models import DataChannel, FeedHealth, GateState

SYM = "BTC/USDT:USDT"
OB, TK, CD = DataChannel.ORDER_BOOK, DataChannel.TRADES_OR_TICKER, DataChannel.CANDLES
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return BASE + timedelta(seconds=self.t)


def make(resume: float = 1.0, timeframe: str = "1m"):
    clock = FakeClock()
    bus = EventBus()
    settings = DataSettings(stale_after_seconds=2.0, resume_after_live_seconds=resume, timeframe=timeframe)
    mon = FeedHealthMonitor(settings, bus, clock, clock.wall)
    mon.register(SYM, OB, required=True)
    mon.register(SYM, TK, required=True)
    mon.register(SYM, CD, required=False)
    return clock, bus, mon


def go_live(clock, mon, resume: float = 1.0) -> None:
    mon.mark_message(SYM, OB)
    mon.mark_message(SYM, TK)
    mon.check()
    clock.t += resume
    mon.mark_message(SYM, OB)
    mon.mark_message(SYM, TK)
    mon.check()


def test_timeframe_parsing():
    assert [timeframe_seconds(t) for t in ("30s", "1m", "5m", "15m", "1h", "4h", "1d")] == [
        30, 60, 300, 900, 3600, 14400, 86400,
    ]
    assert candle_stale_after("1m") == 125.0
    assert candle_stale_after("15m") == 1805.0
    with pytest.raises(ValueError):
        timeframe_seconds("abc")
    with pytest.raises(ValueError):
        timeframe_seconds("0m")


def test_gate_starts_paused_fail_closed():
    clock, bus, mon = make()
    assert not mon.gate.is_active() and not mon.gate.is_active(SYM)
    assert mon.gate.state is GateState.PAUSED
    assert not mon.gate.is_active("UNKNOWN/SYMBOL")
    mon.check()
    assert not mon.gate.is_active(SYM)  # nothing received yet


def test_exactly_stale_threshold_is_live_and_just_over_is_stale():
    clock, bus, mon = make(resume=0.0)
    mon.mark_message(SYM, OB)
    mon.mark_message(SYM, TK)
    mon.check()
    assert mon.gate.is_active(SYM)
    clock.t += 2.0  # age == 2.0 -> still LIVE
    events = mon.check()
    assert events == [] and mon.health(SYM, OB) is FeedHealth.LIVE and mon.gate.is_active(SYM)
    clock.t += 0.01  # age 2.01 -> STALE
    events = mon.check()
    assert {(e.channel, e.health) for e in events} == {(OB, FeedHealth.STALE), (TK, FeedHealth.STALE)}
    assert not mon.gate.is_active(SYM) and not mon.gate.is_active()
    assert any("ORDER_BOOK STALE" in r for r in mon.gate.paused_reasons(SYM))


def test_one_stale_required_feed_pauses_symbol_and_global_only_that_symbol_blocked():
    clock = FakeClock()
    bus = EventBus()
    mon = FeedHealthMonitor(DataSettings(resume_after_live_seconds=0.0), bus, clock, clock.wall)
    for s in ("A", "B"):
        mon.register(s, OB, True)
        mon.register(s, TK, True)
        for ch in (OB, TK):
            mon.mark_message(s, ch)
    mon.check()
    assert mon.gate.is_active("A") and mon.gate.is_active("B") and mon.gate.is_active()
    clock.t += 1.5
    for ch in (OB, TK):
        mon.mark_message("A", ch)
    mon.mark_message("B", TK)  # B's book is not refreshed
    clock.t += 1.0  # B book age 2.5, A age 1.0
    mon.check()
    assert mon.gate.is_active("A") and not mon.gate.is_active("B")
    assert not mon.gate.is_active()  # global paused if any symbol paused
    assert mon.gate.paused_symbols() == ("B",)
    assert all(r.startswith("B ") for r in mon.gate.paused_reasons())


def test_feed_health_events_only_on_transitions():
    clock, bus, mon = make(resume=0.0)
    sub = bus.subscribe(Topic.FEED_HEALTH)
    mon.mark_message(SYM, OB)
    mon.mark_message(SYM, TK)
    assert len(mon.check()) == 2  # CONNECTING -> LIVE x2
    assert mon.check() == [] and mon.check() == []
    clock.t += 3
    assert len(mon.check()) == 2  # -> STALE
    assert mon.check() == []
    mon.mark_message(SYM, OB)
    ev = mon.check()
    assert len(ev) == 1 and ev[0].channel is OB and ev[0].health is FeedHealth.LIVE
    topics = sub.drain()
    assert len(topics) == 5


def test_disconnect_pauses_and_event_carries_retry_info():
    clock, bus, mon = make(resume=0.0)
    go_live(clock, mon, resume=0.0)
    assert mon.gate.is_active(SYM)
    mon.mark_disconnected(SYM, OB, "NetworkError: boom", attempt=2, next_retry=1.5)
    events = mon.check()
    assert len(events) == 1
    e = events[0]
    assert e.health is FeedHealth.DISCONNECTED and e.reconnect_attempt == 2 and e.next_retry_seconds == 1.5
    assert "boom" in e.reason
    assert not mon.gate.is_active(SYM)
    mon.mark_connecting(SYM, OB, attempt=2)
    assert mon.check()[0].health is FeedHealth.CONNECTING
    assert not mon.gate.is_active(SYM)


def test_resume_only_after_continuously_live_for_resume_window():
    clock, bus, mon = make(resume=1.0)
    gate_sub = bus.subscribe(Topic.TRADING_GATE)
    # first data -> LIVE but gate waits for resume window
    mon.mark_message(SYM, OB)
    mon.mark_message(SYM, TK)
    mon.check()
    assert not mon.gate.is_active(SYM)
    assert any("resuming" in r for r in mon.gate.paused_reasons(SYM))
    clock.t += 0.9
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert not mon.gate.is_active(SYM)
    clock.t += 0.1  # exactly 1.0s of continuous LIVE
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert mon.gate.is_active(SYM) and mon.gate.is_active()

    # book goes stale -> PAUSED
    clock.t += 2.5
    mon.mark_message(SYM, TK)
    mon.check()
    assert not mon.gate.is_active(SYM)

    # book recovers: LIVE again but must stay LIVE 1.0s before ACTIVE
    mon.mark_message(SYM, OB)
    mon.check()
    assert mon.health(SYM, OB) is FeedHealth.LIVE and not mon.gate.is_active(SYM)
    clock.t += 0.5
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert not mon.gate.is_active(SYM)
    clock.t += 0.5
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert mon.gate.is_active(SYM)

    states = [e.state for e in gate_sub.drain()]
    assert states == [GateState.ACTIVE, GateState.PAUSED, GateState.ACTIVE]


def test_flapping_feed_restarts_live_streak():
    clock, bus, mon = make(resume=1.0)
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    clock.t += 0.8
    mon.mark_message(SYM, TK)
    mon.mark_disconnected(SYM, OB, "drop", 1, 0.5)
    mon.check()
    mon.mark_connecting(SYM, OB, 1)
    clock.t += 0.5
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)  # book streak restarts here
    mon.check()
    clock.t += 0.9
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert not mon.gate.is_active(SYM)  # only 0.9s since reconnect
    clock.t += 0.1
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert mon.gate.is_active(SYM)


def test_gate_events_on_every_global_or_symbol_transition():
    clock = FakeClock()
    bus = EventBus()
    mon = FeedHealthMonitor(DataSettings(resume_after_live_seconds=0.0), bus, clock, clock.wall)
    sub = bus.subscribe(Topic.TRADING_GATE)
    for s in ("A", "B"):
        mon.register(s, OB, True)
        mon.register(s, TK, True)
    for ch in (OB, TK):
        mon.mark_message("A", ch)
    mon.check()  # A active, B still paused, global paused
    assert mon.gate.is_active("A") and not mon.gate.is_active("B") and not mon.gate.is_active()
    assert mon.gate.paused_symbols() == ("B",)
    sub.drain()
    for ch in (OB, TK):
        mon.mark_message("B", ch)
    mon.check()  # B active -> global active
    events = sub.drain()
    assert events[-1].state is GateState.ACTIVE and events[-1].paused_symbols == ()
    clock.t += 3
    for ch in (OB, TK):
        mon.mark_message("A", ch)
    mon.check()  # B stale -> global paused
    last = sub.drain()[-1]
    assert last.state is GateState.PAUSED and last.paused_symbols == ("B",)
    assert any(r.startswith("B ORDER_BOOK STALE") for r in last.reasons)


def test_candle_channel_uses_longer_threshold_and_is_not_gating():
    clock, bus, mon = make(resume=0.0, timeframe="1m")
    go_live(clock, mon, resume=0.0)
    mon.mark_message(SYM, CD)
    mon.check()
    assert mon.health(SYM, CD) is FeedHealth.LIVE
    clock.t += 100  # > 2s but < 125s
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert mon.health(SYM, CD) is FeedHealth.LIVE and mon.gate.is_active(SYM)
    clock.t += 26  # 126s > 125s
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert mon.health(SYM, CD) is FeedHealth.STALE
    assert mon.gate.is_active(SYM)  # candles are not required for gating


def test_required_candles_gate():
    clock = FakeClock()
    mon = FeedHealthMonitor(DataSettings(resume_after_live_seconds=0.0), EventBus(), clock, clock.wall)
    mon.register(SYM, OB, True); mon.register(SYM, TK, True); mon.register(SYM, CD, True)
    mon.mark_message(SYM, OB); mon.mark_message(SYM, TK)
    mon.check()
    assert not mon.gate.is_active(SYM)  # candles required, none yet
    mon.mark_message(SYM, CD)
    mon.check()
    assert mon.gate.is_active(SYM)


def test_snapshot_is_json_safe_and_complete():
    import json

    clock, bus, mon = make(resume=0.0)
    mon.mark_message(SYM, OB)
    clock.t += 0.5
    mon.mark_disconnected(SYM, TK, "x", 3, 2.0)
    mon.check()
    snap = mon.snapshot()
    json.dumps(snap)
    ch = snap["symbols"][SYM]["channels"]
    assert ch["ORDER_BOOK"]["health"] == "LIVE" and ch["ORDER_BOOK"]["age_seconds"] == 0.5
    assert ch["TICKER"]["health"] == "DISCONNECTED" and ch["TICKER"]["reconnect_attempts"] == 3
    assert ch["TICKER"]["next_retry_seconds"] == 2.0 and ch["TICKER"]["age_seconds"] is None
    assert ch["CANDLES"]["required"] is False and ch["CANDLES"]["stale_after_seconds"] == 125.0
    assert snap["gate"]["state"] == "PAUSED" and snap["symbols"][SYM]["gate"] == "PAUSED"
    assert snap["gate"]["paused_symbols"] == [SYM]


def test_unregistered_feed_raises():
    clock, bus, mon = make()
    with pytest.raises(KeyError):
        mon.mark_message("NOPE", OB)
