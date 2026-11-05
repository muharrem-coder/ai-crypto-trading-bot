from __future__ import annotations

from decimal import Decimal

import pytest

from app.config.settings import QuantSettings
from app.core.types import PositionSide
from app.quant.contracts import StrategyName
from app.quant.strategies import MeanReversionStrategy, MomentumStrategy, ScalpOBIStrategy
from tests.test_quant_support import D, mk_candle, snap

Q = QuantSettings()
L, S = PositionSide.LONG, PositionSide.SHORT


def candle_for(sn):
    c = sn.close
    return mk_candle(0, c, c, c, c, 1.0)


def test_mean_reversion_long_and_levels():
    sn = snap(close=97.5, rsi=22.0, obi=0.3, bb_lower=98.0, bb_upper=102.0, atr=0.8)
    sig = MeanReversionStrategy(Q).evaluate(sn, candle_for(sn))
    assert sig and sig.direction is L and sig.strategy is StrategyName.MEAN_REVERSION
    assert 0.5 <= sig.score <= 1.0
    assert sig.entry == D("97.5")
    assert abs(sig.stop_loss - (D("97.5") - D("1.5") * D("0.8"))) < D("1e-7")
    assert abs(sig.take_profit - (D("97.5") + D("3.0") * D("0.8"))) < D("1e-7")
    assert any("RSI 22.0 < 30" in f and "aşırı satım" in f for f in sig.factors)
    assert all(isinstance(x, Decimal) for x in (sig.entry, sig.stop_loss, sig.take_profit))


def test_mean_reversion_requires_all_conditions():
    ev = MeanReversionStrategy(Q).evaluate
    base = dict(close=97.5, rsi=22.0, obi=0.3, bb_lower=98.0, bb_upper=102.0)
    for patch in ({"close": 98.5}, {"rsi": 35.0}, {"obi": -0.1}, {"obi": None}, {"warm": False}):
        sn = snap(**{**base, **patch})
        assert ev(sn, candle_for(sn)) is None


def test_mean_reversion_short_mirrored_and_stronger_scores_more():
    sn = snap(close=102.5, rsi=78.0, obi=-0.3, atr=0.8)
    sig = MeanReversionStrategy(Q).evaluate(sn, candle_for(sn))
    assert sig and sig.direction is S
    assert sig.stop_loss > sig.entry > sig.take_profit
    assert abs(sig.stop_loss - D("102.5") - D("1.2")) < D("1e-7")
    weak = snap(close=97.99, rsi=29.5, obi=0.0)
    strong = snap(close=96.0, rsi=15.0, obi=0.5)
    sw = MeanReversionStrategy(Q).evaluate(weak, candle_for(weak))
    ss = MeanReversionStrategy(Q).evaluate(strong, candle_for(strong))
    assert 0.5 <= sw.score < ss.score <= 1.0
    assert ss.score == pytest.approx(1.0)


def test_refuses_without_atr():
    for atr in (None, 0.0):
        sn = snap(close=97.5, rsi=22.0, obi=0.3, atr=atr)
        assert MeanReversionStrategy(Q).evaluate(sn, candle_for(sn)) is None
        sn = snap(ema_fast=101, ema_slow=100, close=102, vwap=101, rsi=65, volume_zscore=2, atr=atr)
        assert MomentumStrategy(Q).evaluate(sn, candle_for(sn)) is None
        sn = snap(obi=0.5, close=101, vwap=100, rsi=55, atr=atr)
        assert ScalpOBIStrategy(Q).evaluate(sn, candle_for(sn)) is None


def test_momentum_long_short_and_rejections():
    ev = MomentumStrategy(Q).evaluate
    sn = snap(ema_fast=101.0, ema_slow=100.0, close=102.0, vwap=101.0, rsi=65.0, volume_zscore=2.0, atr=1.0)
    sig = ev(sn, candle_for(sn))
    assert sig and sig.direction is L and sig.strategy is StrategyName.MOMENTUM
    assert sig.score > 0.8 and len(sig.factors) == 5
    assert sig.stop_loss == D("100.5") and sig.take_profit == D("105.0")
    sh = snap(ema_fast=99.0, ema_slow=100.0, close=98.0, vwap=99.0, rsi=35.0, volume_zscore=2.0, atr=1.0)
    s2 = ev(sh, candle_for(sh))
    assert s2 and s2.direction is S and s2.stop_loss == D("99.5")
    good = dict(ema_fast=101.0, ema_slow=100.0, close=102.0, vwap=101.0, rsi=65.0, volume_zscore=2.0, atr=1.0)
    for patch in ({"rsi": 80.0}, {"rsi": 50.0}, {"volume_zscore": 0.5}, {"close": 100.9}, {"vwap": 103.0},
                  {"ema_fast": 99.0}):
        x = snap(**{**good, **patch})
        assert ev(x, candle_for(x)) is None


def test_scalp_obi_long_short_and_rejections():
    ev = ScalpOBIStrategy(Q).evaluate
    sn = snap(obi=0.5, spread_bps=1.0, close=100.8, vwap=100.0, rsi=57.0)
    sig = ev(sn, candle_for(sn))
    assert sig and sig.direction is L and sig.strategy is StrategyName.SCALP_OBI and sig.score > 0.8
    sh = snap(obi=-0.5, spread_bps=1.0, close=99.2, vwap=100.0, rsi=43.0)
    s2 = ev(sh, candle_for(sh))
    assert s2 and s2.direction is S
    good = dict(obi=0.5, spread_bps=1.0, close=100.8, vwap=100.0, rsi=57.0)
    for patch in ({"obi": 0.2}, {"spread_bps": 6.0}, {"close": 99.0}, {"rsi": 75.0}, {"obi": None},
                  {"spread_bps": None}):
        x = snap(**{**good, **patch})
        assert ev(x, candle_for(x)) is None


def test_score_always_in_range_over_grid():
    mr, mo, sc = MeanReversionStrategy(Q), MomentumStrategy(Q), ScalpOBIStrategy(Q)
    for rsi in range(0, 101, 5):
        for obi in (-1.0, -0.4, 0.0, 0.4, 1.0):
            for close in (90.0, 97.0, 100.0, 103.0, 110.0):
                x = snap(close=close, rsi=float(rsi), obi=obi, ema_fast=101, ema_slow=100, vwap=99.5,
                         volume_zscore=3.0)
                for st in (mr, mo, sc):
                    sig = st.evaluate(x, candle_for(x))
                    if sig:
                        assert 0.5 <= sig.score <= 1.0
