from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.config.settings import QuantSettings
from app.quant.indicator_engine import IndicatorEngine
from app.quant.indicators import (
    EMA,
    Bollinger,
    RollingMeanStd,
    SessionVWAP,
    VolumeZScore,
    WilderATR,
    WilderRSI,
    order_book_imbalance,
    spread_bps,
)
from tests.test_quant_support import SYM, T0, D, book, mk_candle, synthetic_candles

N = 2000
REL = 1e-9


@pytest.fixture(scope="module")
def candles():
    return synthetic_candles(N)


@pytest.fixture(scope="module")
def arr(candles):
    return {k: np.array([float(getattr(c, k)) for c in candles]) for k in ("open", "high", "low", "close", "volume")}


def close_to(a: float, b: float) -> bool:
    return abs(a - b) <= REL * max(1.0, abs(b))


def test_ema_matches_sma_seeded_reference(arr):
    x = arr["close"]
    p = 20
    ref = np.full(N, np.nan)
    ref[p - 1] = x[:p].mean()
    a = 2 / (p + 1)
    for i in range(p, N):
        ref[i] = ref[i - 1] + a * (x[i] - ref[i - 1])
    e = EMA(p)
    for i, v in enumerate(x):
        out = e.update(float(v))
        if i < p - 1:
            assert out is None and not e.ready
        else:
            assert close_to(out, ref[i])
    assert e.ready and e.value == out


def test_wilder_rsi_matches_reference(arr):
    x = arr["close"]
    p = 14
    delta = np.diff(x)
    gain, loss = np.maximum(delta, 0), np.maximum(-delta, 0)
    ref = np.full(N, np.nan)
    ag, al = gain[:p].mean(), loss[:p].mean()
    ref[p] = 100 - 100 / (1 + ag / al)
    for i in range(p, N - 1):
        ag = (ag * (p - 1) + gain[i]) / p
        al = (al * (p - 1) + loss[i]) / p
        ref[i + 1] = 100 - 100 / (1 + ag / al)
    r = WilderRSI(p)
    for i, v in enumerate(x):
        out = r.update(float(v))
        if i < p:
            assert out is None
        else:
            assert close_to(out, ref[i])
    # pandas cross-check of the same Wilder smoothing (ewm alpha=1/p seeded by SMA is equivalent after warm-up)
    s = pd.Series(x).diff()
    g = s.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    l = (-s.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    pd_rsi = 100 - 100 / (1 + g / l)
    assert abs(pd_rsi.iloc[-1] - ref[-1]) < 1e-3  # seeds differ slightly, converge geometrically


def test_rsi_edge_cases():
    r = WilderRSI(3)
    for v in (1, 2, 3, 4, 5):
        out = r.update(float(v))
    assert out == 100.0
    r = WilderRSI(3)
    for _ in range(6):
        out = r.update(5.0)
    assert out == 50.0


def test_rolling_mean_std_matches_numpy_and_recomputes(arr):
    x = arr["close"] * 1000 + 1e6  # large offset stresses numerical stability
    p = 20
    rms = RollingMeanStd(p)
    for i, v in enumerate(x):
        m = rms.update(float(v))
        if i < p - 1:
            assert m is None
        else:
            w = x[i - p + 1 : i + 1]
            assert close_to(m, w.mean())
            assert abs(rms.std - w.std()) <= 1e-7 * max(1.0, w.std())
    assert len(rms) == p


def test_bollinger_matches_pandas(arr):
    x = arr["close"]
    s = pd.Series(x)
    mid = s.rolling(20).mean()
    sd = s.rolling(20).std(ddof=0)
    bb = Bollinger(20, 2.0)
    for i, v in enumerate(x):
        out = bb.update(float(v))
        if i < 19:
            assert out is None
        else:
            lo, m, up = out
            assert close_to(m, mid[i]) and close_to(lo, mid[i] - 2 * sd[i]) and close_to(up, mid[i] + 2 * sd[i])


def test_wilder_atr_matches_reference(arr):
    h, l, c = arr["high"], arr["low"], arr["close"]
    p = 14
    tr = np.empty(N)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    ref = np.full(N, np.nan)
    ref[p - 1] = tr[:p].mean()
    for i in range(p, N):
        ref[i] = (ref[i - 1] * (p - 1) + tr[i]) / p
    a = WilderATR(p)
    for i in range(N):
        out = a.update(float(h[i]), float(l[i]), float(c[i]))
        if i < p - 1:
            assert out is None
        else:
            assert close_to(out, ref[i])


def test_volume_zscore_matches_reference(arr):
    v = pd.Series(arr["volume"])
    z = (v - v.rolling(20).mean()) / v.rolling(20).std(ddof=0)
    vz = VolumeZScore(20)
    for i, x in enumerate(v):
        out = vz.update(float(x))
        if i < 19:
            assert out is None
        else:
            assert abs(out - z[i]) <= 1e-8 * max(1.0, abs(z[i]))
    flat = VolumeZScore(5)
    for _ in range(8):
        out = flat.update(3.0)
    assert out == 0.0


def test_session_vwap_matches_reference_and_resets_at_utc_midnight():
    start = datetime(2026, 3, 1, 23, 40, tzinfo=timezone.utc)
    cs = synthetic_candles(60, seed=3, start=start)
    vw = SessionVWAP()
    cum_pv = cum_v = 0.0
    day = None
    saw_reset = False
    for c in cs:
        h, l, cl, v = float(c.high), float(c.low), float(c.close), float(c.volume)
        if c.open_time.date() != day:
            if day is not None:
                saw_reset = True
            day, cum_pv, cum_v = c.open_time.date(), 0.0, 0.0
        cum_pv += (h + l + cl) / 3 * v
        cum_v += v
        out = vw.update(c.open_time, h, l, cl, v)
        assert close_to(out, cum_pv / cum_v)
    assert saw_reset
    # first bar after midnight equals its own typical price
    c = next(c for c in cs if c.open_time.hour == 0 and c.open_time.minute == 0)
    vw2 = SessionVWAP()
    vw2.update(c.open_time - timedelta(minutes=1), 10, 10, 10, 5)
    out = vw2.update(c.open_time, 30.0, 20.0, 25.0, 1.0)
    assert close_to(out, 25.0)


def test_obi_and_spread_hand_built():
    b = book(3.0, 1.0, mid=100.0, half_spread=0.005, levels=4)
    assert order_book_imbalance(b, 4) == pytest.approx((12 - 4) / 16)
    assert order_book_imbalance(b, 2) == pytest.approx((6 - 2) / 8)
    assert spread_bps(b) == pytest.approx(0.01 / 100.0 * 1e4)
    # levels beyond depth are fine
    assert order_book_imbalance(b, 50) == pytest.approx(0.5)
    sym = b.symbol
    from app.data.models import OrderBookSnapshot

    empty = OrderBookSnapshot(sym, (), (), None, T0)
    assert order_book_imbalance(empty, 5) is None and spread_bps(empty) is None
    one_side = OrderBookSnapshot(sym, b.bids, (), None, T0)
    assert spread_bps(one_side) is None
    assert order_book_imbalance(one_side, 3) == 1.0
    sell = book(1.0, 3.0, levels=3)
    assert order_book_imbalance(sell, 3) == pytest.approx(-0.5)


def test_indicator_engine_snapshot_matches_pandas_and_idempotent(candles, arr):
    q = QuantSettings()
    eng = IndicatorEngine(q)
    snaps = [eng.on_candle_closed(c, None) for c in candles]
    assert snaps[48].warm is False and snaps[49].warm is True  # warm once EMA50 ready
    assert snaps[48].ema_slow is None and snaps[49].ema_slow is not None
    s = snaps[-1]
    close = pd.Series(arr["close"])
    assert s.warm and s.obi is None and s.spread_bps is None
    assert close_to(s.bb_middle, close.rolling(20).mean().iloc[-1])
    assert close_to(s.bb_width_pct, (s.bb_upper - s.bb_lower) / s.bb_middle)
    assert close_to(s.atr_pct, s.atr / s.close)
    assert eng.latest(SYM) is s
    # idempotence
    again = eng.on_candle_closed(candles[-1], book(2, 1))
    assert again is s
    older = eng.on_candle_closed(candles[10], None)
    assert older is s
    open_c = mk_candle(N + 5, 100, 101, 99, 100, 1, closed=False)
    assert eng.on_candle_closed(open_c, None) is s
    assert eng.latest(SYM) is s
    # book attaches OBI / spread
    nxt = mk_candle(N, 100, 101, 99, 100.5, 5)
    s2 = eng.on_candle_closed(nxt, book(3, 1))
    assert s2.obi == pytest.approx(0.5) and s2.spread_bps is not None


def test_warmup_equals_streaming(candles):
    q = QuantSettings()
    a, b = IndicatorEngine(q), IndicatorEngine(q)
    a.warmup(SYM, candles[:300])
    for c in candles[:300]:
        b.on_candle_closed(c, None)
    assert a.latest(SYM) == b.latest(SYM)
    a.warmup(SYM, candles[:300])  # re-warming with old data is a no-op
    assert a.latest(SYM) == b.latest(SYM)


def test_r2_memory_bounded_and_fast_over_50k_candles():
    q = QuantSettings()
    eng = IndicatorEngine(q)
    cs = synthetic_candles(50_000, seed=11)
    # prime so lazy state exists, then time the steady-state loop
    t0 = time.perf_counter()
    for c in cs:
        eng.on_candle_closed(c, None)
    per_update = (time.perf_counter() - t0) / len(cs)
    fp = eng.memory_footprint(SYM)
    assert fp["bb"] == q.bb_period and fp["volume_z"] == q.volume_z_period
    assert per_update < 1e-3, f"{per_update * 1e6:.1f} us per update"
    print(f"indicator per-update: {per_update * 1e6:.1f} us")
    # numerical drift check after 50k updates against a fresh window computation
    last = cs[-20:]
    xs = np.array([float(c.close) for c in last])
    s = eng.latest(SYM)
    assert close_to(s.bb_middle, xs.mean())
    assert abs((s.bb_upper - s.bb_lower) / 4 - xs.std()) < 1e-7 * max(1, xs.std())
