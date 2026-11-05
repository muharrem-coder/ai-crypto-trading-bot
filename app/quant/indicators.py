"""Streaming O(1) technical indicators (rule R2: bounded memory, constant work per update).

All indicators hold floats internally. Callers convert Decimal inputs once (see
``IndicatorEngine``). ``update`` returns the current value, or ``None`` until warm.
Every window is a ``collections.deque(maxlen=period)``.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import date, datetime, timezone

from app.data.models import OrderBookSnapshot

RECOMPUTE_EVERY = 256  # recompute running sums from the deque to cancel float drift


class EMA:
    """Exponential moving average, alpha = 2/(period+1), seeded with the SMA of the first ``period`` values."""

    __slots__ = ("period", "_alpha", "_seed", "_count", "_sum", "_value")

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._alpha = 2.0 / (period + 1)
        self._count = 0
        self._sum = 0.0
        self._value: float | None = None

    def update(self, x: float) -> float | None:
        if self._value is None:
            self._count += 1
            self._sum += x
            if self._count >= self.period:
                self._value = self._sum / self.period
        else:
            self._value += self._alpha * (x - self._value)
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None


class WilderRSI:
    """Wilder RSI. Needs ``period + 1`` closes: the first average gain/loss is the simple mean of the first
    ``period`` changes, afterwards avg = (avg*(period-1) + x)/period. avg_loss == 0 -> 100 (both 0 -> 50)."""

    __slots__ = ("period", "_prev", "_n", "_sum_gain", "_sum_loss", "_avg_gain", "_avg_loss", "_value")

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._prev: float | None = None
        self._n = 0
        self._sum_gain = 0.0
        self._sum_loss = 0.0
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._value: float | None = None

    def update(self, close: float) -> float | None:
        prev = self._prev
        self._prev = close
        if prev is None:
            return None
        change = close - prev
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        p = self.period
        if self._value is None and self._n < p:
            self._n += 1
            self._sum_gain += gain
            self._sum_loss += loss
            if self._n < p:
                return None
            self._avg_gain = self._sum_gain / p
            self._avg_loss = self._sum_loss / p
        else:
            self._avg_gain = (self._avg_gain * (p - 1) + gain) / p
            self._avg_loss = (self._avg_loss * (p - 1) + loss) / p
        if self._avg_loss == 0.0:
            self._value = 50.0 if self._avg_gain == 0.0 else 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._value = 100.0 - 100.0 / (1.0 + rs)
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None


class RollingMeanStd:
    """Rolling mean and population std over a ``deque(maxlen=period)`` with running sums.

    Sums are kept relative to a shift (the window mean at the last recompute) which keeps
    ``E[x^2]-E[x]^2`` well conditioned; every ``RECOMPUTE_EVERY`` updates everything is rebuilt
    from the deque to eliminate accumulated rounding drift.
    """

    __slots__ = ("period", "_win", "_shift", "_s1", "_s2", "_since", "_mean", "_std")

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._win: deque[float] = deque(maxlen=period)
        self._shift = 0.0
        self._s1 = 0.0
        self._s2 = 0.0
        self._since = 0
        self._mean: float | None = None
        self._std: float | None = None

    def _rebuild(self) -> None:
        win = self._win
        n = len(win)
        self._shift = math.fsum(win) / n if n else 0.0
        sh = self._shift
        self._s1 = math.fsum(x - sh for x in win)
        self._s2 = math.fsum((x - sh) * (x - sh) for x in win)
        self._since = 0

    def update(self, x: float) -> float | None:
        win = self._win
        if len(win) == self.period:
            old = win[0]
            d = old - self._shift
            self._s1 -= d
            self._s2 -= d * d
        win.append(x)
        d = x - self._shift
        self._s1 += d
        self._s2 += d * d
        self._since += 1
        if self._since >= RECOMPUTE_EVERY or len(win) == 1:
            self._rebuild()
        n = len(win)
        if n < self.period:
            self._mean = self._std = None
            return None
        m = self._s1 / n
        var = self._s2 / n - m * m
        self._mean = self._shift + m
        self._std = math.sqrt(var) if var > 0.0 else 0.0
        return self._mean

    @property
    def value(self) -> float | None:
        return self._mean

    @property
    def mean(self) -> float | None:
        return self._mean

    @property
    def std(self) -> float | None:
        return self._std

    @property
    def ready(self) -> bool:
        return self._mean is not None

    def __len__(self) -> int:
        return len(self._win)


class Bollinger:
    """Bollinger bands: SMA(period) +/- k * population std. ``value`` is ``(lower, middle, upper)``."""

    __slots__ = ("k", "_rms", "_value")

    def __init__(self, period: int = 20, k: float = 2.0) -> None:
        self.k = k
        self._rms = RollingMeanStd(period)
        self._value: tuple[float, float, float] | None = None

    def update(self, close: float) -> tuple[float, float, float] | None:
        m = self._rms.update(close)
        if m is None:
            self._value = None
            return None
        s = self._rms.std or 0.0
        self._value = (m - self.k * s, m, m + self.k * s)
        return self._value

    @property
    def value(self) -> tuple[float, float, float] | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None


class WilderATR:
    """Wilder ATR. TR = max(h-l, |h-prev_close|, |l-prev_close|) (first bar: h-l). Seed = mean of the first
    ``period`` TRs, then atr = (atr*(period-1) + tr)/period."""

    __slots__ = ("period", "_prev_close", "_n", "_sum", "_value")

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._prev_close: float | None = None
        self._n = 0
        self._sum = 0.0
        self._value: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        pc = self._prev_close
        tr = high - low if pc is None else max(high - low, abs(high - pc), abs(low - pc))
        self._prev_close = close
        if self._value is None:
            self._n += 1
            self._sum += tr
            if self._n >= self.period:
                self._value = self._sum / self.period
        else:
            self._value = (self._value * (self.period - 1) + tr) / self.period
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None


class SessionVWAP:
    """Session VWAP with typical price (h+l+c)/3 weighted by volume; resets when the UTC date of the
    candle's open_time changes. Before any volume accumulates the value is None."""

    __slots__ = ("_day", "_pv", "_v", "_value")

    def __init__(self) -> None:
        self._day: date | None = None
        self._pv = 0.0
        self._v = 0.0
        self._value: float | None = None

    def update(self, open_time: datetime, high: float, low: float, close: float, volume: float) -> float | None:
        ot = open_time if open_time.tzinfo is None else open_time.astimezone(timezone.utc)
        day = ot.date()
        if day != self._day:
            self._day = day
            self._pv = 0.0
            self._v = 0.0
            self._value = None
        tp = (high + low + close) / 3.0
        self._pv += tp * volume
        self._v += volume
        if self._v > 0.0:
            self._value = self._pv / self._v
        elif self._value is None:
            self._value = None
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None


class VolumeZScore:
    """z = (v - mean) / population std over the last ``period`` volumes (current included). std == 0 -> 0."""

    __slots__ = ("_rms", "_value")

    def __init__(self, period: int = 20) -> None:
        self._rms = RollingMeanStd(period)
        self._value: float | None = None

    def update(self, volume: float) -> float | None:
        m = self._rms.update(volume)
        if m is None:
            self._value = None
            return None
        s = self._rms.std or 0.0
        self._value = (volume - m) / s if s > 1e-12 * max(1.0, abs(m)) else 0.0
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None


def order_book_imbalance(snapshot: OrderBookSnapshot, levels: int) -> float | None:
    """(sum bid qty - sum ask qty) / (sum bid qty + sum ask qty) over the top ``levels`` levels."""
    bids = snapshot.bids[:levels]
    asks = snapshot.asks[:levels]
    if not bids and not asks:
        return None
    b = float(sum(q for _, q in bids))
    a = float(sum(q for _, q in asks))
    tot = a + b
    if tot <= 0.0:
        return None
    return (b - a) / tot


def spread_bps(snapshot: OrderBookSnapshot) -> float | None:
    """(best_ask - best_bid) / mid * 10_000, None if a side is empty or prices are invalid."""
    bb, ba = snapshot.best_bid, snapshot.best_ask
    if bb is None or ba is None:
        return None
    mid = (float(bb) + float(ba)) / 2.0
    if mid <= 0.0:
        return None
    return (float(ba) - float(bb)) / mid * 10_000.0
