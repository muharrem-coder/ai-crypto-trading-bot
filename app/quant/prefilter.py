"""High-probability pre-filter (rule R3): decides whether Claude is called for a closed candle.

Evaluation order (first match wins):
DATA_PAUSED -> NOT_WARM -> NO_SETUP -> CONFLICTING_SIGNALS -> LOW_CONFLUENCE -> FEE_PRECHECK_FAILED
-> COOLDOWN -> RATE_LIMITED -> TRIGGER. Cooldown / rate state is only written on TRIGGER.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from app.config.settings import QuantSettings, RiskSettings
from app.core.types import InstrumentSpec, LiquidityRole, PositionSide, utcnow
from app.quant.contracts import (
    FeePreCheck,
    IndicatorSnapshot,
    PreFilterOutcome,
    PreFilterResult,
    StrategySignal,
)
from app.risk.fee_calculator import FeeCalculator

AGREE_BONUS = 0.1
CONFLICT_MIN_SCORE = 0.5
_EPS = 1e-9
_HOUR = timedelta(seconds=3600)


def combined_score(best: StrategySignal, supporting: Sequence[StrategySignal]) -> float:
    """best score + 0.1 per agreeing extra strategy, capped at 1.0."""
    return min(1.0, best.score + AGREE_BONUS * len(supporting))


class SignalPreFilter:
    def __init__(
        self,
        quant: QuantSettings,
        risk: RiskSettings,
        spec_provider: Callable[[str], InstrumentSpec],
        gate_is_active: Callable[[str], bool],
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.quant = quant
        self.risk = risk
        self._spec = spec_provider
        self._gate = gate_is_active
        self._clock = clock
        self._cooldown: dict[tuple[str, PositionSide], deque[datetime]] = {}
        self._triggers: deque[datetime] = deque(maxlen=quant.max_triggers_per_hour)

    # -- state ---------------------------------------------------------------------------
    def _rate_count(self, now: datetime) -> int:
        trig = self._triggers
        while trig and now - trig[0] >= _HOUR:
            trig.popleft()
        return len(trig)

    def _cooldown_remaining(self, key: tuple[str, PositionSide], now: datetime) -> float:
        dq = self._cooldown.get(key)
        if not dq:
            return 0.0
        left = self.quant.trigger_cooldown_seconds - (now - dq[-1]).total_seconds()
        return left if left > 0 else 0.0

    # -- fee pre-check ---------------------------------------------------------------------
    def fee_precheck(self, sig: StrategySignal, funding_rate: Decimal | None, funding_interval_hours: int) -> FeePreCheck:
        spec = self._spec(sig.symbol)
        q, r = self.quant, self.risk
        qty = q.reference_notional_usd / sig.entry
        econ = FeeCalculator.evaluate_trade_economics(
            side=sig.direction, entry=sig.entry, take_profit=sig.take_profit, stop_loss=sig.stop_loss,
            quantity=qty, fees=spec.fees, entry_role=LiquidityRole.TAKER, exit_role=LiquidityRole.TAKER,
            slippage_bps=r.default_slippage_bps, funding_rate_per_interval=funding_rate or Decimal(0),
            expected_holding_hours=q.expected_holding_hours, funding_interval_hours=funding_interval_hours,
            min_fee_cover_multiple=r.min_fee_cover_multiple, min_net_profit_usd=Decimal(0),
            market_type=spec.market_type,
        )
        cost = econ.total_round_trip_cost
        gross = econ.expected_gross_profit
        cover = gross / cost if cost > 0 else (Decimal("Infinity") if gross > 0 else Decimal(0))
        return FeePreCheck(
            expected_gross_profit=gross, round_trip_cost=cost, cover_multiple=cover,
            required_multiple=r.min_fee_cover_multiple, reference_quantity=qty, passes=econ.passes_fee_gate,
        )

    # -- main ------------------------------------------------------------------------------
    def evaluate_with_support(
        self,
        snap: IndicatorSnapshot,
        signals: Sequence[StrategySignal],
        funding_rate: Decimal | None = None,
        funding_interval_hours: int = 8,
    ) -> tuple[PreFilterResult, tuple[StrategySignal, ...]]:
        now = self._clock()
        sym = snap.symbol

        def res(outcome: PreFilterOutcome, detail: str, best: StrategySignal | None = None,
                fee: FeePreCheck | None = None, support: tuple[StrategySignal, ...] = ()
                ) -> tuple[PreFilterResult, tuple[StrategySignal, ...]]:
            return PreFilterResult(sym, outcome, now, detail, best, fee), support

        # 1. data gate - always first
        if not self._gate(sym):
            return res(PreFilterOutcome.DATA_PAUSED, "Veri kapısı PAUSED (bayat/kopuk akış): sinyal üretilmez")
        # 2. warm
        if not snap.warm:
            return res(PreFilterOutcome.NOT_WARM, "Göstergeler henüz ısınmadı")
        # 3. no setup / wide spread
        if not signals:
            return res(PreFilterOutcome.NO_SETUP, "Hiçbir strateji sinyal üretmedi")
        if snap.spread_bps is not None and snap.spread_bps > self.quant.max_spread_bps:
            return res(PreFilterOutcome.NO_SETUP,
                       f"Spread {snap.spread_bps:.2f} bps > {self.quant.max_spread_bps:g} bps (likidite zayıf)")
        # 4. conflicts
        longs = [s for s in signals if s.direction is PositionSide.LONG]
        shorts = [s for s in signals if s.direction is PositionSide.SHORT]
        if any(s.score >= CONFLICT_MIN_SCORE for s in longs) and any(s.score >= CONFLICT_MIN_SCORE for s in shorts):
            return res(PreFilterOutcome.CONFLICTING_SIGNALS,
                       "Zıt yönlü güçlü sinyaller: LONG " + ",".join(s.strategy.value for s in longs)
                       + " / SHORT " + ",".join(s.strategy.value for s in shorts))
        # 5. combine
        best = max(signals, key=lambda s: s.score)
        seen = {best.strategy}
        support: list[StrategySignal] = []
        for s in signals:
            if s is not best and s.direction is best.direction and s.strategy not in seen:
                seen.add(s.strategy)
                support.append(s)
        support_t = tuple(support)
        score = combined_score(best, support_t)
        if score + _EPS < self.quant.min_confluence_score:
            return res(PreFilterOutcome.LOW_CONFLUENCE,
                       f"Birleşik skor {score:.2f} < {self.quant.min_confluence_score:.2f}", best, None, support_t)
        # 6. fee-aware pre-check
        fee = self.fee_precheck(best, funding_rate, funding_interval_hours)
        if not fee.passes:
            return res(PreFilterOutcome.FEE_PRECHECK_FAILED,
                       f"Beklenen brüt kâr {fee.expected_gross_profit:.4f} USDT, toplam maliyet "
                       f"{fee.round_trip_cost:.4f} USDT x {fee.required_multiple} gerekli; karşılama {fee.cover_multiple:.2f}x",
                       best, fee, support_t)
        # 7. cooldown
        key = (sym, best.direction)
        left = self._cooldown_remaining(key, now)
        if left > 0:
            return res(PreFilterOutcome.COOLDOWN,
                       f"{sym} {best.direction.value} için bekleme süresi: {left:.0f} sn kaldı", best, fee, support_t)
        # 8. global hourly cap
        n = self._rate_count(now)
        if n >= self.quant.max_triggers_per_hour:
            return res(PreFilterOutcome.RATE_LIMITED,
                       f"Saatlik tetik sınırı doldu ({n}/{self.quant.max_triggers_per_hour})", best, fee, support_t)
        # 9. trigger: record state only now
        self._cooldown.setdefault(key, deque(maxlen=1)).append(now)
        self._triggers.append(now)
        names = "+".join([best.strategy.value] + [s.strategy.value for s in support_t])
        return res(PreFilterOutcome.TRIGGER,
                   f"{best.direction.value} {names}: skor {score:.2f}, ücret karşılama {fee.cover_multiple:.2f}x",
                   best, fee, support_t)

    def evaluate(
        self,
        snap: IndicatorSnapshot,
        signals: Sequence[StrategySignal],
        funding_rate: Decimal | None = None,
        funding_interval_hours: int = 8,
    ) -> PreFilterResult:
        return self.evaluate_with_support(snap, signals, funding_rate, funding_interval_hours)[0]
