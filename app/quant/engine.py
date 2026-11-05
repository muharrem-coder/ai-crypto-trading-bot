"""Quant engine: indicators -> strategies -> pre-filter -> payload. Pure, synchronous, deterministic."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.config.settings import QuantSettings, RiskSettings
from app.core.types import InstrumentSpec, utcnow
from app.data.models import Candle, OrderBookSnapshot
from app.news.contracts import NewsEvent
from app.quant.contracts import (
    IndicatorSnapshot,
    PreFilterOutcome,
    PreFilterResult,
    SignalPayload,
    StrategySignal,
)
from app.quant.indicator_engine import IndicatorEngine
from app.quant.payload import build_signal_payload
from app.quant.prefilter import SignalPreFilter
from app.quant.strategies import Strategy, default_strategies


class QuantEngine:
    def __init__(
        self,
        settings: QuantSettings,
        risk: RiskSettings,
        spec_provider: Callable[[str], InstrumentSpec],
        gate_is_active: Callable[[str], bool],
        *,
        strategies: Sequence[Strategy] | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.settings = settings
        self.risk = risk
        self._clock = clock
        self.indicators = IndicatorEngine(settings)
        self.prefilter = SignalPreFilter(settings, risk, spec_provider, gate_is_active, clock=clock)
        self.strategies: list[Strategy] = list(strategies) if strategies is not None else default_strategies(settings)
        # bounded: one counter per PreFilterOutcome plus two bookkeeping keys
        self.stats: dict[str, int] = {o.value: 0 for o in PreFilterOutcome}
        self.stats["CANDLES"] = 0
        self.stats["IGNORED"] = 0

    def warmup(self, symbol: str, candles: Sequence[Candle]) -> None:
        self.indicators.warmup(symbol, candles)

    def process_closed_candle(
        self,
        candle: Candle,
        book: OrderBookSnapshot | None,
        news: Sequence[NewsEvent],
        market_health: dict[str, Any],
        funding_rate: Decimal | None = None,
    ) -> tuple[IndicatorSnapshot, PreFilterResult, SignalPayload | None]:
        if not self.indicators.accepts(candle):
            snap = self.indicators.on_candle_closed(candle, book)  # returns latest / blank, no state change
            self.stats["IGNORED"] += 1
            result = PreFilterResult(candle.symbol, PreFilterOutcome.NO_SETUP, self._clock(),
                                     "Mum yok sayıldı (kapanmamış/yinelenen/eski)")
            return snap, result, None
        snap = self.indicators.on_candle_closed(candle, book)
        self.stats["CANDLES"] += 1
        signals: list[StrategySignal] = []
        if snap.warm:
            for strat in self.strategies:
                sig = strat.evaluate(snap, candle)
                if sig is not None:
                    signals.append(sig)
        result, support = self.prefilter.evaluate_with_support(snap, signals, funding_rate)
        self.stats[result.outcome.value] += 1
        payload: SignalPayload | None = None
        if result.outcome is PreFilterOutcome.TRIGGER:
            payload = build_signal_payload(
                result, support, snap, news, market_health,
                max_news=self.settings.news_in_payload, clock=self._clock,
            )
        return snap, result, payload
