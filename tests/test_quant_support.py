"""Shared builders for the quant tests (no tests in here)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.models import Candle, OrderBookSnapshot
from app.news.contracts import NewsEvent
from app.quant.contracts import IndicatorSnapshot, StrategyName, StrategySignal
from app.core.types import PositionSide

D = Decimal
T0 = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
SYM = "BTC/USDT:USDT"


def dec(x: float) -> Decimal:
    return Decimal(repr(round(x, 8)))


def mk_candle(i: int, o: float, h: float, lo: float, c: float, v: float, *, symbol: str = SYM,
              start: datetime = T0, closed: bool = True) -> Candle:
    return Candle(symbol, "1m", start + timedelta(minutes=i), dec(o), dec(h), dec(lo), dec(c), dec(v), closed)


def synthetic_candles(n: int, seed: int = 7, start: datetime = T0, symbol: str = SYM, base: float = 100.0) -> list[Candle]:
    rng = random.Random(seed)
    out: list[Candle] = []
    price = base
    for i in range(n):
        o = price
        c = max(1.0, o * (1 + rng.gauss(0, 0.004)))
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.002)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, 0.002)))
        v = max(0.1, rng.lognormvariate(3, 0.5))
        out.append(mk_candle(i, o, h, lo, c, v, symbol=symbol, start=start))
        price = c
    return out


def book(bid_qty: float, ask_qty: float, *, mid: float = 100.0, half_spread: float = 0.002, levels: int = 10,
         symbol: str = SYM) -> OrderBookSnapshot:
    bids = tuple((dec(mid - half_spread - i * 0.01), dec(bid_qty)) for i in range(levels))
    asks = tuple((dec(mid + half_spread + i * 0.01), dec(ask_qty)) for i in range(levels))
    return OrderBookSnapshot(symbol, bids, asks, None, T0)


def snap(**kw) -> IndicatorSnapshot:
    base = dict(symbol=SYM, timeframe="1m", timestamp=T0, close=100.0, rsi=50.0, ema_fast=100.0, ema_slow=100.0,
                vwap=100.0, bb_upper=102.0, bb_middle=100.0, bb_lower=98.0, bb_width_pct=0.04, atr=1.0,
                atr_pct=0.01, obi=0.0, spread_bps=1.0, volume_zscore=0.0, warm=True)
    base.update(kw)
    return IndicatorSnapshot(**base)


def signal(direction: PositionSide = PositionSide.LONG, strategy: StrategyName = StrategyName.MEAN_REVERSION,
           score: float = 0.8, entry: str = "100", atr_pct: float = 0.01, symbol: str = SYM) -> StrategySignal:
    e = D(entry)
    a = e * D(str(atr_pct))
    sgn = 1 if direction is PositionSide.LONG else -1
    return StrategySignal(strategy, symbol, direction, score, ("faktör",), e, e - sgn * D("1.5") * a,
                          e + sgn * D("3") * a, T0)


def news_event(i: int, minutes_ago: int, sentiment: str, importance: int) -> NewsEvent:
    pub = T0 - timedelta(minutes=minutes_ago)
    return NewsEvent(f"id{i}", "rss:coindesk", None, f"Haber {i}", None, pub, pub, ("BTC",), (), D(sentiment),
                     "BULLISH" if D(sentiment) > 0 else "BEARISH", importance)
