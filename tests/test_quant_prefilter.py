from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from app.config.settings import QuantSettings, RiskSettings
from app.core.types import MarketType, PositionSide
from app.quant.contracts import PreFilterOutcome as O
from app.quant.contracts import StrategyName as N
from app.quant.prefilter import SignalPreFilter
from app.risk.testing import make_spec
from tests.test_quant_support import SYM, T0, D, signal, snap

L, S = PositionSide.LONG, PositionSide.SHORT
GOOD_ATR = 0.00334  # 3 x ATR ~ 1.0% target
MICRO_ATR = 0.000267  # 3 x ATR ~ 0.08% target


class Env:
    def __init__(self, **q):
        self.now = T0
        self.active = True
        self.quant = QuantSettings(**q)
        self.risk = RiskSettings()
        self.pf = SignalPreFilter(self.quant, self.risk, lambda s: make_spec(s), lambda s: self.active,
                                  clock=lambda: self.now)


def test_paused_gate_blocks_even_perfect_setup_and_is_checked_first():
    e = Env()
    e.active = False
    sigs = [signal(score=1.0, atr_pct=GOOD_ATR), signal(strategy=N.MOMENTUM, score=1.0, atr_pct=GOOD_ATR)]
    r = e.pf.evaluate(snap(), sigs)
    assert r.outcome is O.DATA_PAUSED
    # paused wins even over NOT_WARM and over no signals
    assert e.pf.evaluate(snap(warm=False), []).outcome is O.DATA_PAUSED
    e.active = True
    assert e.pf.evaluate(snap(), sigs).outcome is O.TRIGGER


def test_gate_not_consulted_state_untouched_when_paused():
    e = Env()
    e.active = False
    e.pf.evaluate(snap(), [signal(score=1.0, atr_pct=GOOD_ATR)])
    e.active = True
    assert e.pf.evaluate(snap(), [signal(score=1.0, atr_pct=GOOD_ATR)]).outcome is O.TRIGGER


def test_not_warm_and_no_setup():
    e = Env()
    assert e.pf.evaluate(snap(warm=False), [signal(atr_pct=GOOD_ATR)]).outcome is O.NOT_WARM
    assert e.pf.evaluate(snap(), []).outcome is O.NO_SETUP
    wide = e.pf.evaluate(snap(spread_bps=9.0), [signal(score=1.0, atr_pct=GOOD_ATR)])
    assert wide.outcome is O.NO_SETUP and "Spread" in wide.detail


def test_conflicting_signals():
    e = Env()
    r = e.pf.evaluate(snap(), [signal(L, score=0.9, atr_pct=GOOD_ATR),
                               signal(S, N.MOMENTUM, score=0.6, atr_pct=GOOD_ATR)])
    assert r.outcome is O.CONFLICTING_SIGNALS
    # a weak opposite signal (<0.5) is not a conflict
    r = e.pf.evaluate(snap(), [signal(L, score=0.9, atr_pct=GOOD_ATR), signal(S, N.MOMENTUM, score=0.4, atr_pct=GOOD_ATR)])
    assert r.outcome is O.TRIGGER


def test_low_confluence_and_agreement_bonus():
    e = Env()
    r = e.pf.evaluate(snap(), [signal(score=0.65, atr_pct=GOOD_ATR)])
    assert r.outcome is O.LOW_CONFLUENCE and r.best_signal is not None
    # 0.65 + 0.1 (one agreeing strategy) = 0.75 >= 0.7
    sigs = [signal(score=0.65, atr_pct=GOOD_ATR), signal(strategy=N.MOMENTUM, score=0.6, atr_pct=GOOD_ATR)]
    res, support = e.pf.evaluate_with_support(snap(), sigs)
    assert res.outcome is O.TRIGGER and [s.strategy for s in support] == [N.MOMENTUM]
    assert res.best_signal.strategy is N.MEAN_REVERSION
    # exact threshold is inclusive (0.6 + 0.1)
    e2 = Env()
    r2 = e2.pf.evaluate(snap(), [signal(score=0.6, atr_pct=GOOD_ATR), signal(strategy=N.SCALP_OBI, score=0.55, atr_pct=GOOD_ATR)])
    assert r2.outcome is O.TRIGGER


def test_bonus_capped_at_one():
    e = Env()
    sigs = [signal(score=0.98, atr_pct=GOOD_ATR), signal(strategy=N.MOMENTUM, score=0.9, atr_pct=GOOD_ATR),
            signal(strategy=N.SCALP_OBI, score=0.9, atr_pct=GOOD_ATR)]
    res, support = e.pf.evaluate_with_support(snap(), sigs)
    from app.quant.prefilter import combined_score
    assert res.outcome is O.TRIGGER and combined_score(res.best_signal, support) == 1.0


def test_micro_move_fails_fee_precheck_but_one_percent_passes():
    e = Env()
    r = e.pf.evaluate(snap(), [signal(score=0.95, atr_pct=MICRO_ATR)])
    assert r.outcome is O.FEE_PRECHECK_FAILED
    fc = r.fee_check
    assert fc is not None and not fc.passes
    # gross ~ 0.08% of 1000 = 0.8 USD; cost = 0.5 + 0.5 + 0.4 slippage (2bps x 2 legs) + 0 funding = 1.4
    assert abs(fc.expected_gross_profit - D("0.8")) < D("0.01")
    assert abs(fc.round_trip_cost - D("1.4")) < D("0.01")
    assert fc.required_multiple == D("2") and fc.cover_multiple < 1
    assert fc.reference_quantity == D("1000") / D("100")
    ok = e.pf.evaluate(snap(), [signal(score=0.95, atr_pct=GOOD_ATR)])
    assert ok.outcome is O.TRIGGER and ok.fee_check.passes and ok.fee_check.cover_multiple > 2


def test_fee_precheck_uses_funding_and_direction():
    e = Env()
    # expensive funding for longs flips a marginal trade to a failure; shorts receive it
    sigs_l = [signal(L, score=0.95, atr_pct=0.0012)]
    sigs_s = [signal(S, score=0.95, atr_pct=0.0012)]
    assert e.pf.evaluate(snap(), sigs_l, funding_rate=D("0")).outcome is O.TRIGGER
    e2 = Env()
    assert e2.pf.evaluate(snap(), sigs_l, funding_rate=D("0.002")).outcome is O.FEE_PRECHECK_FAILED
    e3 = Env()
    assert e3.pf.evaluate(snap(), sigs_s, funding_rate=D("0.002")).outcome is O.TRIGGER


def test_spot_has_no_funding():
    e = Env()
    e.pf._spec = lambda s: make_spec(s, market_type=MarketType.SPOT)
    r = e.pf.evaluate(snap(), [signal(L, score=0.95, atr_pct=0.0012)], funding_rate=D("0.002"))
    assert r.outcome is O.TRIGGER


def test_cooldown_per_symbol_direction():
    e = Env(trigger_cooldown_seconds=300)
    sigs = [signal(score=0.95, atr_pct=GOOD_ATR)]
    assert e.pf.evaluate(snap(), sigs).outcome is O.TRIGGER
    e.now += timedelta(seconds=299)
    r = e.pf.evaluate(snap(), sigs)
    assert r.outcome is O.COOLDOWN and "kaldı" in r.detail
    # other direction and other symbol are independent
    assert e.pf.evaluate(snap(), [signal(S, score=0.95, atr_pct=GOOD_ATR)]).outcome is O.TRIGGER
    other = signal(score=0.95, atr_pct=GOOD_ATR, symbol="ETH/USDT:USDT")
    assert e.pf.evaluate(snap(symbol="ETH/USDT:USDT"), [other]).outcome is O.TRIGGER
    e.now += timedelta(seconds=2)
    assert e.pf.evaluate(snap(), sigs).outcome is O.TRIGGER


def test_failed_evaluations_do_not_consume_cooldown_or_rate():
    e = Env(max_triggers_per_hour=1)
    assert e.pf.evaluate(snap(), [signal(score=0.95, atr_pct=MICRO_ATR)]).outcome is O.FEE_PRECHECK_FAILED
    assert e.pf.evaluate(snap(), [signal(score=0.3, atr_pct=GOOD_ATR)]).outcome is O.LOW_CONFLUENCE
    assert e.pf.evaluate(snap(), [signal(score=0.95, atr_pct=GOOD_ATR)]).outcome is O.TRIGGER


def test_hourly_rate_limit_rolling_window():
    e = Env(max_triggers_per_hour=3, trigger_cooldown_seconds=0)
    sigs = [signal(score=0.95, atr_pct=GOOD_ATR)]
    for _ in range(3):
        assert e.pf.evaluate(snap(), sigs).outcome is O.TRIGGER
        e.now += timedelta(seconds=10)
    r = e.pf.evaluate(snap(), sigs)
    assert r.outcome is O.RATE_LIMITED and "3/3" in r.detail
    # global: another symbol is limited too
    other = signal(score=0.95, atr_pct=GOOD_ATR, symbol="ETH/USDT:USDT")
    assert e.pf.evaluate(snap(symbol="ETH/USDT:USDT"), [other]).outcome is O.RATE_LIMITED
    # first trigger leaves the 3600s window -> one slot frees up
    e.now = T0 + timedelta(seconds=3600)
    assert e.pf.evaluate(snap(), sigs).outcome is O.TRIGGER
    assert e.pf.evaluate(snap(), sigs).outcome is O.RATE_LIMITED
    assert len(e.pf._triggers) <= 3


def test_state_is_bounded():
    e = Env(max_triggers_per_hour=5, trigger_cooldown_seconds=0)
    for i in range(500):
        e.now += timedelta(seconds=1000)
        e.pf.evaluate(snap(), [signal(score=0.95, atr_pct=GOOD_ATR)])
    assert len(e.pf._triggers) <= 5
    assert all(d.maxlen == 1 for d in e.pf._cooldown.values())
