"""Buffers, order book state and backoff (no network, no real sleeping)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.data.backoff import ExponentialBackoff
from app.data.buffers import CandleBuffer, OrderBookState, RingBuffer, to_decimal
from app.data.models import Candle

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)


def bar(i: int, close: str = "100", closed: bool = False) -> Candle:
    c = Decimal(close)
    return Candle("BTC/USDT:USDT", "1m", T0 + timedelta(minutes=i), c, c, c, c, Decimal("1"), closed)


# ---- RingBuffer ------------------------------------------------------------------------
def test_ring_buffer_basics_and_eviction():
    rb: RingBuffer[int] = RingBuffer(3)
    assert rb.maxlen == 3 and rb.latest is None and len(rb) == 0
    rb.extend([1, 2, 3, 4, 5])
    assert list(rb) == [3, 4, 5]
    assert rb.latest == 5
    assert rb.last(2) == (4, 5)
    assert rb.last(10) == (3, 4, 5)
    assert rb.last(0) == ()
    rb.clear()
    assert len(rb) == 0


def test_ring_buffer_rejects_unbounded():
    with pytest.raises(ValueError):
        RingBuffer(0)


def test_to_decimal():
    assert to_decimal(0.1) == Decimal("0.1")
    assert to_decimal("1.5") == Decimal("1.5")
    assert to_decimal(None) is None and to_decimal("abc") is None and to_decimal(float("nan")) is None


# ---- CandleBuffer ----------------------------------------------------------------------
def test_upsert_replaces_in_progress_bar():
    buf = CandleBuffer(10)
    assert buf.upsert(bar(0, "100")) is None
    assert buf.upsert(bar(0, "101")) is None
    assert len(buf) == 1
    assert buf.current is not None and buf.current.close == Decimal("101")
    assert buf.closed() == () and buf.last_closed is None


def test_upsert_newer_closes_previous_exactly_once():
    buf = CandleBuffer(10)
    buf.upsert(bar(0, "100"))
    buf.upsert(bar(0, "102"))
    closed = buf.upsert(bar(1, "103"))
    assert closed is not None
    assert closed.open_time == T0 and closed.closed is True and closed.close == Decimal("102")
    assert buf.last_closed == closed
    assert buf.current is not None and buf.current.open_time == T0 + timedelta(minutes=1)
    # a repeat of the new bar must not close anything
    assert buf.upsert(bar(1, "104")) is None
    assert len(buf.closed()) == 1


def test_upsert_ignores_older_and_duplicate_closed_bars():
    buf = CandleBuffer(10)
    buf.upsert(bar(0))
    buf.upsert(bar(1))
    buf.upsert(bar(2))
    assert len(buf) == 3
    assert buf.upsert(bar(0, "999")) is None  # out of order
    assert buf.upsert(bar(1, "999")) is None  # already closed duplicate
    assert [c.close for c in buf] == [Decimal("100")] * 3
    assert [c.closed for c in buf] == [True, True, False]


def test_upsert_gap_still_closes_previous_bar():
    buf = CandleBuffer(10)
    buf.upsert(bar(0))
    closed = buf.upsert(bar(5))
    assert closed is not None and closed.open_time == T0
    assert len(buf) == 2


def test_upsert_with_closed_flag():
    buf = CandleBuffer(10)
    assert buf.upsert(bar(0, closed=True)) is not None  # first bar already final
    assert buf.upsert(bar(0, closed=True)) is None  # duplicate
    buf.upsert(bar(1))
    assert buf.upsert(bar(1, "105", closed=True)) is not None  # closed in place
    assert buf.current is None and buf.last_closed.close == Decimal("105")


def test_seed_is_silent_sorted_and_deduped():
    buf = CandleBuffer(10)
    stored = buf.seed([bar(2), bar(0), bar(1), bar(1)])
    assert stored == 3
    assert [c.open_time for c in buf] == [T0 + timedelta(minutes=i) for i in range(3)]
    assert all(c.closed for c in buf)
    assert buf.current is None


# ---- Order book ------------------------------------------------------------------------
def test_order_book_sorts_truncates_and_uses_decimal():
    ob = OrderBookState("BTC/USDT:USDT", depth=3)
    bids = [[99.5, 1], [100.0, 2], [99.0, 3], [98.0, 4], [97.0, 5]]
    asks = [[102, 1], [101.0, 2], [103, 3], [104, 4], [105, 5]]
    snap = ob.apply_snapshot(bids, asks, None, NOW)
    assert snap is not None and ob.last_accepted
    assert [p for p, _ in snap.bids] == [Decimal("100.0"), Decimal("99.5"), Decimal("99.0")]
    assert [p for p, _ in snap.asks] == [Decimal("101.0"), Decimal("102"), Decimal("103")]
    assert snap.best_bid == Decimal("100.0") and snap.best_ask == Decimal("101.0")
    assert snap.mid == Decimal("100.5")
    assert snap.received_at == NOW


def test_crossed_book_rejected_keeps_previous():
    ob = OrderBookState("X", depth=5)
    good = ob.apply_snapshot([[100, 1]], [[101, 1]], None, NOW)
    kept = ob.apply_snapshot([[102, 1]], [[101, 1]], None, NOW)
    assert kept is good and not ob.last_accepted
    assert ob.crossed_rejects == 1
    ob.apply_snapshot([[101, 1]], [[101, 1]], None, NOW)  # locked book also rejected
    assert ob.crossed_rejects == 2
    assert ob.snapshot is good


def test_crossed_first_book_returns_none_and_invalid_sides_counted():
    ob = OrderBookState("X", depth=5)
    assert ob.apply_snapshot([[101, 1]], [[100, 1]], None, NOW) is None
    assert ob.apply_snapshot([], [[100, 1]], None, NOW) is None
    assert ob.apply_snapshot([[0, 1], ["x", 1]], [[100, 0]], None, NOW) is None
    assert ob.crossed_rejects == 1 and ob.invalid_rejects == 2


# ---- R2: memory caps -------------------------------------------------------------------
def test_r2_candle_buffer_capped_after_50k_candles():
    buf = CandleBuffer(500)
    closes = 0
    for i in range(50_000):
        if buf.upsert(bar(i, str(100 + i % 7))) is not None:
            closes += 1
    assert len(buf) == 500
    assert buf.maxlen == 500
    assert closes == 49_999
    assert buf.current is not None and buf.current.open_time == T0 + timedelta(minutes=49_999)
    assert len(buf.closed()) == 499


def test_r2_order_book_stays_top_n_after_50k_updates():
    depth = 10
    ob = OrderBookState("X", depth=depth)
    for i in range(50_000):
        base = 100 + (i % 5)
        bids = [[base - 0.01 * k, 1 + k] for k in range(1, 21)]  # 20 levels in, only top 10 kept
        asks = [[base + 0.01 * k, 1 + k] for k in range(1, 21)]
        snap = ob.apply_snapshot(bids, asks, None, NOW)
        assert snap is not None
        assert len(snap.bids) == depth and len(snap.asks) == depth
    assert len(ob.snapshot.bids) == depth and len(ob.snapshot.asks) == depth


def test_r2_ring_buffer_capped_for_liquidations_like_streams():
    rb: RingBuffer[int] = RingBuffer(500)
    for i in range(50_000):
        rb.append(i)
    assert len(rb) == 500 and rb.latest == 49_999 and rb.last(1) == (49_999,)


# ---- Backoff ---------------------------------------------------------------------------
def test_backoff_grows_exponentially_and_caps():
    b = ExponentialBackoff(0.5, 8.0, jitter=0.0)
    delays = [b.next_delay() for _ in range(8)]
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0]
    assert b.attempt == 8


def test_backoff_reset():
    b = ExponentialBackoff(1.0, 30.0, jitter=0.0)
    b.next_delay(); b.next_delay(); b.next_delay()
    assert b.attempt == 3
    b.reset()
    assert b.attempt == 0 and b.next_delay() == 1.0


def test_backoff_jitter_within_bounds_and_varies():
    b = ExponentialBackoff(1.0, 16.0, jitter=0.2, rng=random.Random(42))
    seen = set()
    for attempt in range(200):
        raw = min(16.0, 1.0 * 2 ** min(attempt % 6, 62))
        b.reset()
        for _ in range(attempt % 6):
            b.next_delay()
        d = b.next_delay()
        assert raw * 0.8 - 1e-9 <= d <= min(16.0, raw * 1.2) + 1e-9
        seen.add(round(d, 6))
    assert len(seen) > 6  # jitter actually randomises


def test_backoff_is_deterministic_with_seeded_rng():
    a = ExponentialBackoff(0.5, 30, 0.3, random.Random(7))
    b = ExponentialBackoff(0.5, 30, 0.3, random.Random(7))
    assert [a.next_delay() for _ in range(10)] == [b.next_delay() for _ in range(10)]


def test_backoff_huge_attempt_does_not_overflow():
    b = ExponentialBackoff(0.5, 30.0, 0.0)
    for _ in range(5000):
        b.next_delay()
    assert b.next_delay() == 30.0


@pytest.mark.parametrize("args", [(0, 1, 0.1), (1, 0.5, 0.1), (1, 2, 1.5)])
def test_backoff_validation(args):
    with pytest.raises(ValueError):
        ExponentialBackoff(*args)
