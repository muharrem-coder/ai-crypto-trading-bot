"""Per-symbol streaming indicator state producing ``IndicatorSnapshot`` on every closed candle."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime

from app.config.settings import QuantSettings
from app.data.models import Candle, OrderBookSnapshot
from app.quant.contracts import IndicatorSnapshot
from app.quant.indicators import (
    EMA,
    Bollinger,
    SessionVWAP,
    VolumeZScore,
    WilderATR,
    WilderRSI,
    order_book_imbalance,
    spread_bps,
)


class _SymbolState:
    __slots__ = ("rsi", "ema_fast", "ema_slow", "vwap", "bb", "atr", "vz", "last_open", "latest")

    def __init__(self, q: QuantSettings) -> None:
        self.rsi = WilderRSI(q.rsi_period)
        self.ema_fast = EMA(q.ema_fast)
        self.ema_slow = EMA(q.ema_slow)
        self.vwap = SessionVWAP()
        self.bb = Bollinger(q.bb_period, q.bb_std)
        self.atr = WilderATR(q.atr_period)
        self.vz = VolumeZScore(q.volume_z_period)
        self.last_open: datetime | None = None
        self.latest: IndicatorSnapshot | None = None


class IndicatorEngine:
    def __init__(self, settings: QuantSettings) -> None:
        self._q = settings
        self._state: dict[str, _SymbolState] = {}

    # -- helpers -----------------------------------------------------------------------------
    def accepts(self, candle: Candle) -> bool:
        """True if the candle is closed and newer than the last processed one for its symbol."""
        if not candle.closed:
            return False
        st = self._state.get(candle.symbol)
        return st is None or st.last_open is None or candle.open_time > st.last_open

    def latest(self, symbol: str) -> IndicatorSnapshot | None:
        st = self._state.get(symbol)
        return st.latest if st else None

    def symbols(self) -> tuple[str, ...]:
        return tuple(self._state)

    def memory_footprint(self, symbol: str) -> dict[str, int]:
        """Lengths of every internal window (R2 verification)."""
        st = self._state[symbol]
        return {"bb": len(st.bb._rms), "volume_z": len(st.vz._rms)}

    # -- core --------------------------------------------------------------------------------
    def _ingest(self, st: _SymbolState, c: Candle) -> IndicatorSnapshot:
        o_t = c.open_time
        h, lo, cl, v = float(c.high), float(c.low), float(c.close), float(c.volume)
        rsi = st.rsi.update(cl)
        ef = st.ema_fast.update(cl)
        es = st.ema_slow.update(cl)
        vw = st.vwap.update(o_t, h, lo, cl, v)
        bb = st.bb.update(cl)
        atr = st.atr.update(h, lo, cl)
        vz = st.vz.update(v)
        st.last_open = o_t
        lower = middle = upper = width = None
        if bb is not None:
            lower, middle, upper = bb
            width = (upper - lower) / middle if middle else None
        atr_pct = atr / cl if (atr is not None and cl > 0) else None
        warm = rsi is not None and ef is not None and es is not None and bb is not None and atr is not None
        snap = IndicatorSnapshot(
            symbol=c.symbol, timeframe=c.timeframe, timestamp=o_t, close=cl, rsi=rsi, ema_fast=ef, ema_slow=es,
            vwap=vw, bb_upper=upper, bb_middle=middle, bb_lower=lower, bb_width_pct=width, atr=atr,
            atr_pct=atr_pct, obi=None, spread_bps=None, volume_zscore=vz, warm=warm,
        )
        st.latest = snap
        return snap

    def warmup(self, symbol: str, candles: Iterable[Candle]) -> None:
        st = self._state.get(symbol)
        if st is None:
            st = self._state[symbol] = _SymbolState(self._q)
        for c in candles:
            if c.symbol != symbol or not c.closed:
                continue
            if st.last_open is not None and c.open_time <= st.last_open:
                continue
            self._ingest(st, c)

    def on_candle_closed(self, candle: Candle, book: OrderBookSnapshot | None) -> IndicatorSnapshot:
        st = self._state.get(candle.symbol)
        if st is None:
            st = self._state[candle.symbol] = _SymbolState(self._q)
        if not self.accepts(candle):
            if st.latest is not None:
                return st.latest
            return IndicatorSnapshot(
                symbol=candle.symbol, timeframe=candle.timeframe, timestamp=candle.open_time,
                close=float(candle.close), rsi=None, ema_fast=None, ema_slow=None, vwap=None, bb_upper=None,
                bb_middle=None, bb_lower=None, bb_width_pct=None, atr=None, atr_pct=None, obi=None,
                spread_bps=None, volume_zscore=None, warm=False,
            )
        snap = self._ingest(st, candle)
        if book is not None:
            obi = order_book_imbalance(book, self._q.obi_levels)
            sp = spread_bps(book)
            snap = replace(snap, obi=obi, spread_bps=sp)
            st.latest = snap
        return snap
