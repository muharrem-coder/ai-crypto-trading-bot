"""MarketDataEngine end-to-end with ScriptedMarketSource, fake clocks and a fake sleep."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import ccxt
import pytest

from app.config.settings import DataSettings
from app.core.events import EventBus, Topic
from app.core.types import Side
from app.data.engine import MarketDataEngine, dt_to_ms
from app.data.mock_source import ScriptedMarketSource
from app.data.models import DataChannel, FeedHealth, GateState

SYM = "BTC/USDT:USDT"
OB, TK, CD = DataChannel.ORDER_BOOK, DataChannel.TRADES_OR_TICKER, DataChannel.CANDLES
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
BASE_MS = dt_to_ms(BASE)
MIN_MS = 60_000


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return BASE + timedelta(seconds=self.t - 1000.0)


class ControlledSleep:
    """Records every requested delay; in blocking mode each sleep waits for one ``allow()`` permit."""

    def __init__(self, blocking: bool = False) -> None:
        self.delays: list[float] = []
        self.blocking = blocking
        self._permits: asyncio.Queue[None] = asyncio.Queue()

    def allow(self, n: int = 1) -> None:
        for _ in range(n):
            self._permits.put_nowait(None)

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if self.blocking:
            await self._permits.get()  # one permit releases exactly one backoff sleep
        else:
            await asyncio.sleep(0)


def row(i: int, close: str = "100", vol: str = "1") -> list:
    return [BASE_MS + i * MIN_MS, close, close, close, close, vol]


async def settle(predicate=None, timeout: float = 1.0) -> None:
    """Yield to the loop until ``predicate()`` holds (or a few turns pass when no predicate)."""
    if predicate is None:
        for _ in range(30):
            await asyncio.sleep(0)
        return
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.001)


def make_settings(**kw) -> DataSettings:
    base = dict(
        stale_after_seconds=2.0,
        resume_after_live_seconds=1.0,
        reconnect_base_delay_seconds=0.5,
        reconnect_max_delay_seconds=4.0,
        reconnect_jitter=0.0,
        health_check_interval_seconds=5.0,  # background loop effectively idle; tests call checks
        bootstrap_candles=0,
        candle_buffer_size=100,
        orderbook_depth=5,
        liquidation_buffer_size=10,
    )
    base.update(kw)
    return DataSettings(**base)


class Harness:
    def __init__(self, sleep: ControlledSleep | None = None, settings: DataSettings | None = None, **kw) -> None:
        self.clock = FakeClock()
        self.bus = EventBus()
        self.source = ScriptedMarketSource()
        self.sleep = sleep or ControlledSleep()
        self.settings = settings or make_settings()
        self.engine = MarketDataEngine(
            self.source, self.bus, [SYM], self.settings,
            clock=self.clock, wall_clock=self.clock.wall, sleep=self.sleep,
            rng=random.Random(1), **kw,
        )

    def feed(self, book: bool = True, ticker: bool = True) -> None:
        if book:
            self.source.push_order_book(SYM, [[99.5, 2], [99.0, 3]], [[100.5, 1], [101.0, 4]], BASE_MS)
        if ticker:
            self.source.push_ticker(SYM, 99.5, 100.5, 100.0, BASE_MS)


@pytest.fixture
async def h():
    harness = Harness()
    yield harness
    await harness.engine.stop()


async def test_messages_flow_to_accessors_and_bus(h: Harness):
    sub_ob = h.bus.subscribe(Topic.ORDER_BOOK)
    sub_tk = h.bus.subscribe(Topic.TICKER)
    await h.engine.start()
    h.feed()
    await settle(lambda: h.engine.order_book(SYM) and h.engine.last_ticker(SYM))
    book = h.engine.order_book(SYM)
    assert book.best_bid == Decimal("99.5") and book.best_ask == Decimal("100.5")
    assert book.received_at == h.clock.wall()
    t = h.engine.last_ticker(SYM)
    assert (t.bid, t.ask, t.last) == (Decimal("99.5"), Decimal("100.5"), Decimal("100.0"))
    assert t.timestamp == BASE and t.mark is None
    assert len(sub_ob.drain()) == 1 and len(sub_tk.drain()) == 1
    snap = h.engine.health_snapshot()
    assert snap["symbols"][SYM]["channels"]["ORDER_BOOK"]["health"] in ("CONNECTING", "LIVE")
    assert snap["engine"][SYM]["candles_buffered"] == 0


async def test_order_book_truncated_to_depth_and_crossed_rejected(h: Harness):
    await h.engine.start()
    bids = [[100 - i * 0.1, 1] for i in range(1, 12)]
    asks = [[101 + i * 0.1, 1] for i in range(1, 12)]
    h.source.push_order_book(SYM, bids, asks)
    await settle(lambda: h.engine.order_book(SYM) is not None)
    good = h.engine.order_book(SYM)
    assert len(good.bids) == 5 and len(good.asks) == 5 and good.exchange_ts is None
    sub = h.bus.subscribe(Topic.ORDER_BOOK)
    h.source.push_order_book(SYM, [[105, 1]], [[104, 1]])  # crossed
    await settle(lambda: h.source.pending(SYM, OB) == 0)
    await settle()
    assert h.engine.order_book(SYM) is good
    assert h.engine.health_snapshot()["engine"][SYM]["orderbook_crossed_rejects"] == 1
    assert sub.drain() == []


async def test_crossed_book_does_not_keep_feed_live(h: Harness):
    await h.engine.start()
    h.feed()
    await settle(lambda: h.engine.order_book(SYM) is not None)
    await h.engine.run_health_check_once()
    h.clock.t += 2.5
    h.source.push_order_book(SYM, [[105, 1]], [[104, 1]])  # crossed => not a fresh update
    h.source.push_ticker(SYM, 99.5, 100.5, 100.0)
    await settle()
    await h.engine.run_health_check_once()
    assert h.engine.monitor.health(SYM, OB) is FeedHealth.STALE
    assert not h.engine.gate.is_active(SYM)


async def test_candle_closed_published_once_per_bar_with_duplicates_and_out_of_order(h: Harness):
    closed_sub = h.bus.subscribe(Topic.CANDLE_CLOSED)
    update_sub = h.bus.subscribe(Topic.CANDLE_UPDATE)
    await h.engine.start()
    h.source.push_ohlcv(SYM, [row(0, "100")])
    h.source.push_ohlcv(SYM, [row(0, "101")])  # update of forming bar
    h.source.push_ohlcv(SYM, [row(0, "101")])  # exact duplicate -> no new update
    await settle(lambda: h.source.pending(SYM, CD) == 0)
    await settle()
    assert closed_sub.drain() == []
    updates = update_sub.drain()
    assert [u.close for u in updates] == [Decimal("100"), Decimal("101")]
    assert all(not u.closed for u in updates)

    # ccxt resends the whole window, unordered, with a duplicate row and a new bar
    h.source.push_ohlcv(SYM, [row(1, "103"), row(0, "102"), row(1, "104"), row(2, "105")])
    await settle(lambda: len(h.engine.candles(SYM)) == 3)
    await settle()
    closed = closed_sub.drain()
    assert [c.open_time for c in closed] == [BASE, BASE + timedelta(minutes=1)]
    assert all(c.closed for c in closed)
    assert closed[0].close == Decimal("102")  # latest state of bar 0 before bar 1 appeared
    assert closed[1].close == Decimal("104")  # last duplicate wins
    assert h.engine.candles(SYM).current.open_time == BASE + timedelta(minutes=2)

    # old window re-sent: nothing closes twice
    h.source.push_ohlcv(SYM, [row(0), row(1), row(2, "105")])
    await settle()
    assert closed_sub.drain() == []
    assert len(h.engine.candles(SYM).closed()) == 2


async def test_bootstrap_drops_forming_bar_and_stays_silent():
    settings = make_settings(bootstrap_candles=5)
    harness = Harness(settings=settings)
    closed_sub = harness.bus.subscribe(Topic.CANDLE_CLOSED)
    # wall clock = BASE + 0s; bar 0 started at BASE so it is still forming
    harness.source.set_history(SYM, [row(i - 5, str(90 + i)) for i in range(0, 6)])
    await harness.engine.start()
    try:
        buf = harness.engine.candles(SYM)
        assert len(buf.closed()) == 5 and buf.current is not None
        assert buf.current.open_time == BASE and not buf.current.closed
        assert buf.last_closed.open_time == BASE - timedelta(minutes=1)
        assert closed_sub.drain() == []
        assert harness.source.calls["fetch_ohlcv"] == 1
        # the live stream continues the forming bar and closes it exactly once
        harness.source.push_ohlcv(SYM, [row(0, "200"), row(1, "201")])
        await settle(lambda: len(buf.closed()) == 6)
        closed = closed_sub.drain()
        assert len(closed) == 1 and closed[0].close == Decimal("200")
    finally:
        await harness.engine.stop()


async def test_bootstrap_with_all_closed_history_and_failure_tolerated():
    harness = Harness(settings=make_settings(bootstrap_candles=3))
    harness.source.set_history(SYM, [row(i - 10) for i in range(4)])  # all long past -> all closed
    await harness.engine.start()
    try:
        assert len(harness.engine.candles(SYM).closed()) == 4 and harness.engine.candles(SYM).current is None
    finally:
        await harness.engine.stop()

    class Boom(ScriptedMarketSource):
        async def fetch_ohlcv(self, symbol, timeframe, limit):
            raise ccxt.NetworkError("rest down")

    bad = Harness(settings=make_settings(bootstrap_candles=3))
    bad.engine._source = Boom()
    await bad.engine.start()  # must not raise
    await bad.engine.stop()
    assert len(bad.engine.candles(SYM)) == 0


async def test_candle_buffer_in_engine_is_capped():
    harness = Harness(settings=make_settings(candle_buffer_size=100))
    await harness.engine.start()
    try:
        for start in range(0, 2000, 100):
            harness.source.push_ohlcv(SYM, [row(i) for i in range(start, start + 100)])
        await settle(lambda: harness.engine.candles(SYM).current is not None and
                     harness.engine.candles(SYM).current.open_time == BASE + timedelta(minutes=1999))
        assert len(harness.engine.candles(SYM)) == 100
    finally:
        await harness.engine.stop()


async def test_ticker_edge_cases(h: Harness):
    await h.engine.start()
    h.source.push_ticker(SYM, None, None, 100.0)  # only last -> bid/ask fall back to last
    await settle(lambda: h.engine.last_ticker(SYM) is not None)
    t = h.engine.last_ticker(SYM)
    assert t.bid == t.ask == t.last == Decimal("100.0")
    h.source.push_ticker(SYM, 101, 100, 100.5)  # bid > ask -> rejected, previous kept
    await settle()
    assert h.engine.last_ticker(SYM) is t


async def test_liquidations_buffered_published_and_bounded(h: Harness):
    sub = h.bus.subscribe(Topic.LIQUIDATION)
    await h.engine.start()
    for i in range(25):
        h.source.push_liquidation(SYM, Side.SELL if i % 2 else "buy", 100 + i, "0.5", BASE_MS + i)
    await settle(lambda: h.source.pending(SYM, ScriptedMarketSource.LIQUIDATIONS) == 0)
    await settle()
    liqs = h.engine.liquidations(SYM)
    assert len(liqs) == 10  # liquidation_buffer_size
    assert liqs[-1].price == Decimal("124") and liqs[-1].quantity == Decimal("0.5")
    assert liqs[-1].side is Side.BUY and liqs[-2].side is Side.SELL
    assert len(sub.drain()) == 25


async def test_liquidations_not_supported_disables_channel_quietly():
    harness = Harness()
    harness.source.liquidations_supported = False
    await harness.engine.start()
    try:
        await settle(lambda: harness.source.calls["watch_liquidations"] == 1)
        await settle()
        assert harness.source.calls["watch_liquidations"] == 1  # not retried
        assert harness.sleep.delays == []
        assert harness.engine.health_snapshot()["engine"][SYM]["liquidations_disabled"] is True
        harness.feed()
        await settle(lambda: harness.engine.order_book(SYM) and harness.engine.last_ticker(SYM))
        await harness.engine.run_health_check_once()
        # gating is unaffected by the missing channel
        harness.clock.t += 1.0
        harness.feed()
        await settle(lambda: harness.source.pending(SYM, OB) == 0 and harness.source.pending(SYM, TK) == 0)
        await settle()
        await harness.engine.run_health_check_once()
        assert harness.engine.gate.is_active(SYM)
    finally:
        await harness.engine.stop()


async def test_r1_disconnect_pauses_gate_backoff_reconnects_and_resumes():
    sleep = ControlledSleep(blocking=True)
    harness = Harness(sleep=sleep)
    eng, src, clock = harness.engine, harness.source, harness.clock
    gate_sub = harness.bus.subscribe(Topic.TRADING_GATE)
    health_sub = harness.bus.subscribe(Topic.FEED_HEALTH)
    await eng.start()

    # --- warm up: both required feeds live, gate resumes after 1.0s of continuous LIVE ----
    harness.feed()
    await settle(lambda: eng.order_book(SYM) and eng.last_ticker(SYM))
    await eng.run_health_check_once()
    assert not eng.gate.is_active(SYM)  # LIVE but not yet for resume_after_live_seconds
    clock.t += 1.0
    harness.feed()
    await settle(lambda: src.pending(SYM, OB) == 0 and src.pending(SYM, TK) == 0)
    await settle()
    await eng.run_health_check_once()
    assert eng.gate.is_active(SYM) and eng.gate.is_active()
    assert [e.state for e in gate_sub.drain()] == [GateState.ACTIVE]
    health_sub.drain()

    # --- socket drops on the order book (twice in a row => backoff grows) ----------------
    src.disconnect(SYM, OB)
    src.disconnect(SYM, OB, ccxt.RequestTimeout("timeout"))
    harness.feed(book=False)  # ticker keeps flowing
    await settle(lambda: sleep.delays == [0.5])
    assert eng.monitor.health(SYM, OB) is FeedHealth.DISCONNECTED  # engine checks right away
    assert not eng.gate.is_active(SYM) and not eng.gate.is_active()
    events = gate_sub.drain()
    assert events and events[-1].state is GateState.PAUSED and events[-1].paused_symbols == (SYM,)
    assert any("ORDER_BOOK DISCONNECTED" in r for r in events[-1].reasons)
    hevents = health_sub.drain()
    disc = [e for e in hevents if e.health is FeedHealth.DISCONNECTED]
    assert disc and disc[0].channel is OB and disc[0].reconnect_attempt == 1 and disc[0].next_retry_seconds == 0.5
    assert eng.health_snapshot()["symbols"][SYM]["channels"]["ORDER_BOOK"]["reconnect_attempts"] == 1

    # --- reconnect attempt #1 fails again -> delay doubles --------------------------------
    calls_before = src.calls["watch_order_book"]
    sleep.allow()
    await settle(lambda: sleep.delays == [0.5, 1.0])
    assert src.calls["watch_order_book"] == calls_before + 1
    assert eng.monitor.health(SYM, OB) is FeedHealth.DISCONNECTED
    assert not eng.gate.is_active(SYM)

    # --- third attempt succeeds: new messages arrive -> backoff resets --------------------
    sleep.allow()
    clock.t += 0.5
    harness.feed()
    await settle(lambda: src.pending(SYM, OB) == 0 and src.pending(SYM, TK) == 0)
    await settle()
    await eng.run_health_check_once()
    assert eng.monitor.health(SYM, OB) is FeedHealth.LIVE
    assert not eng.gate.is_active(SYM)  # LIVE again but must stay LIVE for 1.0s
    assert any("resuming" in r for r in eng.gate.paused_reasons(SYM))
    clock.t += 0.5
    harness.feed()
    await settle(lambda: src.pending(SYM, OB) == 0 and src.pending(SYM, TK) == 0)
    await settle()
    await eng.run_health_check_once()
    assert not eng.gate.is_active(SYM)  # 0.5s < 1.0s
    clock.t += 0.5
    harness.feed()
    await settle(lambda: src.pending(SYM, OB) == 0 and src.pending(SYM, TK) == 0)
    await settle()
    await eng.run_health_check_once()
    assert eng.gate.is_active(SYM) and eng.gate.is_active()
    assert gate_sub.drain()[-1].state is GateState.ACTIVE

    # --- backoff was reset by the successful message: next failure starts from base ------
    src.disconnect(SYM, OB)
    await settle(lambda: len(sleep.delays) == 3)
    assert sleep.delays == [0.5, 1.0, 0.5]
    await eng.stop()
    assert src.closed


async def test_backoff_delays_cap_at_maximum_and_use_jitter_rng():
    sleep = ControlledSleep()
    harness = Harness(sleep=sleep, settings=make_settings(reconnect_jitter=0.2))
    for _ in range(8):
        harness.source.disconnect(SYM, OB)
    await harness.engine.start()
    try:
        await settle(lambda: len(sleep.delays) >= 8)
        d = sleep.delays[:8]
        raws = [0.5, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        for delay, raw in zip(d, raws):
            assert raw * 0.8 - 1e-9 <= delay <= min(4.0, raw * 1.2) + 1e-9
        assert d[0] != 0.5 or d[1] != 1.0  # jitter applied
        assert max(d) <= 4.0
    finally:
        await harness.engine.stop()


async def test_every_channel_reconnects_independently():
    sleep = ControlledSleep()
    harness = Harness(sleep=sleep)
    await harness.engine.start()
    try:
        harness.source.disconnect(SYM, TK)
        harness.source.disconnect(SYM, CD)
        await settle(lambda: len(sleep.delays) == 2)
        await settle()
        harness.source.push_ticker(SYM, 99, 101, 100)
        harness.source.push_ohlcv(SYM, [row(0)])
        await settle(lambda: harness.engine.last_ticker(SYM) is not None and
                     harness.engine.candles(SYM).current is not None)
        assert harness.source.calls["watch_ticker"] >= 2 and harness.source.calls["watch_ohlcv"] >= 2
    finally:
        await harness.engine.stop()


async def test_non_ccxt_error_in_feed_also_reconnects():
    sleep = ControlledSleep()
    harness = Harness(sleep=sleep)
    harness.source.disconnect(SYM, OB, RuntimeError("socket reset"))
    await harness.engine.start()
    try:
        await settle(lambda: sleep.delays == [0.5])
        snap = harness.engine.health_snapshot()["symbols"][SYM]["channels"]["ORDER_BOOK"]
        assert "socket reset" in snap["reason"] or snap["health"] in ("CONNECTING", "DISCONNECTED")
    finally:
        await harness.engine.stop()


async def test_stop_cancels_tasks_closes_source_and_is_idempotent():
    harness = Harness()
    await harness.engine.start()
    await settle()
    tasks = list(harness.engine._tasks)
    assert len(tasks) == 5  # 4 channels + health loop
    await harness.engine.stop()
    assert all(t.done() for t in tasks)
    assert harness.source.calls["close"] == 1
    await harness.engine.stop()
    assert harness.source.calls["close"] == 1


async def test_stop_during_backoff_sleep_is_clean():
    sleep = ControlledSleep(blocking=True)
    harness = Harness(sleep=sleep)
    harness.source.disconnect(SYM, OB)
    await harness.engine.start()
    await settle(lambda: sleep.delays == [0.5])
    await harness.engine.stop()
    assert all(t.done() for t in asyncio.all_tasks() if t.get_name().startswith("md-"))


async def test_background_health_loop_pauses_gate_without_manual_checks():
    settings = make_settings(health_check_interval_seconds=0.01, resume_after_live_seconds=0.0)
    harness = Harness(settings=settings)
    await harness.engine.start()
    try:
        harness.feed()
        await settle(lambda: harness.engine.gate.is_active(SYM))
        harness.clock.t += 2.5  # fake time passes; no messages
        await settle(lambda: not harness.engine.gate.is_active(SYM))
        assert any("STALE" in r for r in harness.engine.gate.paused_reasons(SYM))
    finally:
        await harness.engine.stop()


async def test_scripted_source_semantics():
    src = ScriptedMarketSource(queue_size=4)
    for i in range(5):
        src.push_ticker(SYM, 1, 2, i)
    assert src.pending(SYM, TK) == 4  # bounded: oldest dropped
    assert (await src.watch_ticker(SYM))["last"] == 1
    src.disconnect(SYM, TK)
    src.push_ticker(SYM, 1, 2, 99)
    assert (await src.watch_ticker(SYM))["last"] == 3  # queued items delivered before the marker
    assert (await src.watch_ticker(SYM))["last"] == 4
    with pytest.raises(ccxt.NetworkError):
        await src.watch_ticker(SYM)
    assert (await src.watch_ticker(SYM))["last"] == 99
    assert src.calls["watch_ticker"] == 5

    # a pending watch is woken by disconnect
    pending = asyncio.create_task(src.watch_order_book(SYM, 5))
    await asyncio.sleep(0)
    src.disconnect(SYM, OB, ccxt.ExchangeNotAvailable("down"))
    with pytest.raises(ccxt.ExchangeNotAvailable):
        await pending
    src.set_history(SYM, [row(i) for i in range(10)])
    assert len(await src.fetch_ohlcv(SYM, "1m", 4)) == 4
    assert (await src.fetch_ohlcv(SYM, "1m", 100))[0][0] == BASE_MS
    await src.close()
    assert src.closed
