"""Deterministic strategies. Each returns a ``StrategySignal`` only on clear confluence.

Score formula (shared): every hard condition must hold; then
    score = 0.5 + 0.5 * sum(w_i * s_i),   sum(w_i) == 1,   s_i in [0, 1]
i.e. a setup that just barely satisfies all conditions scores 0.5 and a textbook one approaches 1.0.
Strength terms s_i are documented per strategy. Stops/targets are ATR based; no ATR -> no signal.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from app.config.settings import QuantSettings
from app.core.types import PositionSide
from app.data.models import Candle
from app.quant.contracts import IndicatorSnapshot, StrategyName, StrategySignal

_Q = Decimal("0.00000001")


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class Strategy(Protocol):
    name: StrategyName

    def evaluate(self, snap: IndicatorSnapshot, candle: Candle) -> StrategySignal | None: ...


def _levels(
    q: QuantSettings, snap: IndicatorSnapshot, candle: Candle, direction: PositionSide
) -> tuple[Decimal, Decimal, Decimal] | None:
    """(entry, stop, target) as Decimals from ATR multiples, or None if ATR unusable / stop invalid."""
    if snap.atr is None or snap.atr <= 0.0:
        return None
    entry = candle.close
    atr = Decimal(repr(snap.atr))
    sl_d = q.atr_stop_multiple * atr
    tp_d = q.atr_target_multiple * atr
    if direction is PositionSide.LONG:
        sl, tp = entry - sl_d, entry + tp_d
    else:
        sl, tp = entry + sl_d, entry - tp_d
    sl, tp = sl.quantize(_Q), tp.quantize(_Q)
    if sl <= 0 or tp <= 0:
        return None
    return entry, sl, tp


def _make(
    name: StrategyName, q: QuantSettings, snap: IndicatorSnapshot, candle: Candle, direction: PositionSide,
    score: float, factors: list[str],
) -> StrategySignal | None:
    lv = _levels(q, snap, candle, direction)
    if lv is None:
        return None
    entry, sl, tp = lv
    return StrategySignal(
        strategy=name, symbol=snap.symbol, direction=direction, score=round(min(1.0, max(0.0, score)), 6),
        factors=tuple(factors), entry=entry, stop_loss=sl, take_profit=tp, timestamp=snap.timestamp,
    )


class MeanReversionStrategy:
    """Long: close <= BB lower AND RSI < oversold AND OBI >= 0. Short mirrored (close >= BB upper,
    RSI > overbought, OBI <= 0). A missing order book (OBI None) means no signal.

    Strengths: s_bb = clip(distance beyond band / (0.25 * band width)), weight 0.4;
    s_rsi = clip((oversold - RSI) / 10) (short: (RSI - overbought)/10), weight 0.4;
    s_obi = clip(|OBI| / obi_threshold) in the favourable direction, weight 0.2.
    """

    name = StrategyName.MEAN_REVERSION

    def __init__(self, settings: QuantSettings) -> None:
        self.q = settings

    def evaluate(self, snap: IndicatorSnapshot, candle: Candle) -> StrategySignal | None:
        if not snap.warm or snap.rsi is None or snap.obi is None or snap.bb_lower is None or snap.bb_upper is None:
            return None
        q = self.q
        c = snap.close
        band = snap.bb_upper - snap.bb_lower
        if band <= 0.0:
            return None
        if c <= snap.bb_lower and snap.rsi < q.rsi_oversold and snap.obi >= 0.0:
            s_bb = _clip01((snap.bb_lower - c) / (0.25 * band))
            s_rsi = _clip01((q.rsi_oversold - snap.rsi) / 10.0)
            s_obi = _clip01(snap.obi / q.obi_threshold)
            score = 0.5 + 0.5 * (0.4 * s_bb + 0.4 * s_rsi + 0.2 * s_obi)
            factors = [
                f"Fiyat {c:.4f} <= Bollinger alt bant {snap.bb_lower:.4f} (bant dışı)",
                f"RSI {snap.rsi:.1f} < {q.rsi_oversold:g} (aşırı satım)",
                f"OBI {snap.obi:+.2f} >= 0 (alıcılar çökmüyor)",
            ]
            return _make(self.name, q, snap, candle, PositionSide.LONG, score, factors)
        if c >= snap.bb_upper and snap.rsi > q.rsi_overbought and snap.obi <= 0.0:
            s_bb = _clip01((c - snap.bb_upper) / (0.25 * band))
            s_rsi = _clip01((snap.rsi - q.rsi_overbought) / 10.0)
            s_obi = _clip01(-snap.obi / q.obi_threshold)
            score = 0.5 + 0.5 * (0.4 * s_bb + 0.4 * s_rsi + 0.2 * s_obi)
            factors = [
                f"Fiyat {c:.4f} >= Bollinger üst bant {snap.bb_upper:.4f} (bant dışı)",
                f"RSI {snap.rsi:.1f} > {q.rsi_overbought:g} (aşırı alım)",
                f"OBI {snap.obi:+.2f} <= 0 (satıcılar baskın/alıcılar zayıf)",
            ]
            return _make(self.name, q, snap, candle, PositionSide.SHORT, score, factors)
        return None


class MomentumStrategy:
    """Long: EMA20 > EMA50, close > VWAP, close > EMA20, RSI in 55..75, volume z >= 1.
    Short: EMA20 < EMA50, close < VWAP, close < EMA20, RSI in 25..45, volume z >= 1.

    Strengths (weight 0.2 each): trend = clip(|EMA20-EMA50|/EMA50 / 0.002); vwap = clip(|close-VWAP|/ATR);
    ema = clip(|close-EMA20|/ATR); rsi = 1 - |RSI - 65|/10 (short: centre 35); volume = clip((z-1)/2).
    """

    name = StrategyName.MOMENTUM
    RSI_LONG = (55.0, 75.0)
    RSI_SHORT = (25.0, 45.0)
    MIN_VOL_Z = 1.0

    def __init__(self, settings: QuantSettings) -> None:
        self.q = settings

    def evaluate(self, snap: IndicatorSnapshot, candle: Candle) -> StrategySignal | None:
        if (not snap.warm or snap.rsi is None or snap.ema_fast is None or snap.ema_slow is None
                or snap.vwap is None or snap.volume_zscore is None or snap.atr is None or snap.atr <= 0.0):
            return None
        c, ef, es, vw, rsi, vz, atr = snap.close, snap.ema_fast, snap.ema_slow, snap.vwap, snap.rsi, snap.volume_zscore, snap.atr
        if vz < self.MIN_VOL_Z or es <= 0.0:
            return None
        lo, hi = self.RSI_LONG
        if ef > es and c > vw and c > ef and lo <= rsi <= hi:
            s = [
                _clip01((ef - es) / es / 0.002), _clip01((c - vw) / atr), _clip01((c - ef) / atr),
                _clip01(1.0 - abs(rsi - 65.0) / 10.0), _clip01((vz - 1.0) / 2.0),
            ]
            factors = [
                f"EMA20 {ef:.4f} > EMA50 {es:.4f} (yükseliş trendi)",
                f"Fiyat {c:.4f} > VWAP {vw:.4f}",
                f"Fiyat {c:.4f} > EMA20 {ef:.4f}",
                f"RSI {rsi:.1f} momentum bölgesinde ({lo:g}-{hi:g})",
                f"Hacim z-skoru {vz:.2f} >= {self.MIN_VOL_Z:g}",
            ]
            return _make(self.name, self.q, snap, candle, PositionSide.LONG, 0.5 + 0.5 * sum(s) / 5.0, factors)
        lo, hi = self.RSI_SHORT
        if ef < es and c < vw and c < ef and lo <= rsi <= hi:
            s = [
                _clip01((es - ef) / es / 0.002), _clip01((vw - c) / atr), _clip01((ef - c) / atr),
                _clip01(1.0 - abs(rsi - 35.0) / 10.0), _clip01((vz - 1.0) / 2.0),
            ]
            factors = [
                f"EMA20 {ef:.4f} < EMA50 {es:.4f} (düşüş trendi)",
                f"Fiyat {c:.4f} < VWAP {vw:.4f}",
                f"Fiyat {c:.4f} < EMA20 {ef:.4f}",
                f"RSI {rsi:.1f} düşüş momentum bölgesinde ({lo:g}-{hi:g})",
                f"Hacim z-skoru {vz:.2f} >= {self.MIN_VOL_Z:g}",
            ]
            return _make(self.name, self.q, snap, candle, PositionSide.SHORT, 0.5 + 0.5 * sum(s) / 5.0, factors)
        return None


class ScalpOBIStrategy:
    """Long: OBI >= obi_threshold, spread <= max_spread_bps, close > VWAP, RSI in 45..70.
    Short: OBI <= -obi_threshold, spread <= max, close < VWAP, RSI in 30..55.

    Strengths: obi (w 0.4) = clip((|OBI|-thr)/thr); spread (w 0.2) = 1 - spread/max;
    vwap (w 0.2) = clip(|close-VWAP|/ATR); rsi (w 0.2) = 1 - |RSI - centre|/12.5 (centre 57.5 long / 42.5 short).
    """

    name = StrategyName.SCALP_OBI
    RSI_LONG = (45.0, 70.0)
    RSI_SHORT = (30.0, 55.0)

    def __init__(self, settings: QuantSettings) -> None:
        self.q = settings

    def evaluate(self, snap: IndicatorSnapshot, candle: Candle) -> StrategySignal | None:
        if (not snap.warm or snap.obi is None or snap.spread_bps is None or snap.vwap is None
                or snap.rsi is None or snap.atr is None or snap.atr <= 0.0):
            return None
        q = self.q
        obi, sp, c, vw, rsi, atr = snap.obi, snap.spread_bps, snap.close, snap.vwap, snap.rsi, snap.atr
        if sp > q.max_spread_bps:
            return None
        lo, hi = self.RSI_LONG
        if obi >= q.obi_threshold and c > vw and lo <= rsi <= hi:
            s_obi = _clip01((obi - q.obi_threshold) / q.obi_threshold)
            score = 0.5 + 0.5 * (0.4 * s_obi + 0.2 * _clip01(1.0 - sp / q.max_spread_bps)
                                 + 0.2 * _clip01((c - vw) / atr) + 0.2 * _clip01(1.0 - abs(rsi - 57.5) / 12.5))
            factors = [
                f"OBI {obi:+.2f} >= {q.obi_threshold:g} (alış baskısı)",
                f"Spread {sp:.2f} bps <= {q.max_spread_bps:g} bps",
                f"Fiyat {c:.4f} > VWAP {vw:.4f}",
                f"RSI {rsi:.1f} nötr-güçlü bölgede ({lo:g}-{hi:g})",
            ]
            return _make(self.name, q, snap, candle, PositionSide.LONG, score, factors)
        lo, hi = self.RSI_SHORT
        if obi <= -q.obi_threshold and c < vw and lo <= rsi <= hi:
            s_obi = _clip01((-obi - q.obi_threshold) / q.obi_threshold)
            score = 0.5 + 0.5 * (0.4 * s_obi + 0.2 * _clip01(1.0 - sp / q.max_spread_bps)
                                 + 0.2 * _clip01((vw - c) / atr) + 0.2 * _clip01(1.0 - abs(rsi - 42.5) / 12.5))
            factors = [
                f"OBI {obi:+.2f} <= -{q.obi_threshold:g} (satış baskısı)",
                f"Spread {sp:.2f} bps <= {q.max_spread_bps:g} bps",
                f"Fiyat {c:.4f} < VWAP {vw:.4f}",
                f"RSI {rsi:.1f} nötr-zayıf bölgede ({lo:g}-{hi:g})",
            ]
            return _make(self.name, q, snap, candle, PositionSide.SHORT, score, factors)
        return None


def default_strategies(settings: QuantSettings) -> list[Strategy]:
    return [MeanReversionStrategy(settings), MomentumStrategy(settings), ScalpOBIStrategy(settings)]
