from __future__ import annotations

import json
import time
from decimal import Decimal

import pytest

from app.config.settings import QuantSettings, RiskSettings
from app.core.types import PositionSide
from app.quant.contracts import PreFilterOutcome as O
from app.quant.contracts import PreFilterResult, StrategyName
from app.quant.engine import QuantEngine
from app.quant.payload import build_signal_payload, payload_to_json
from app.risk.testing import make_spec
from tests.test_quant_support import SYM, T0, D, book, mk_candle, news_event, signal, snap, synthetic_candles


def crafted_selloff():
    """120 calm zig-zag bars, one violent sell-off bar, then a few quiet bars."""
    cs, p = [], 100.0
    for i in range(120):
        o, c = p, 100.0 + (0.1 if i % 2 else -0.1)
        cs.append(mk_candle(i, o, max(o, c) + 0.15, min(o, c) - 0.15, c, 10.0))
        p = c
    for j, drop in enumerate([2.5, 0.2, -0.1, -0.1, -0.1]):
        o, c = p, p - drop
        cs.append(mk_candle(120 + j, o, max(o, c) + 0.05, min(o, c) - 0.05, c, 10.0))
        p = c
    return cs


def make_engine(active=True, **q):
    state = {"active": active}
    eng = QuantEngine(QuantSettings(**q), RiskSettings(), lambda s: make_spec(s), lambda s: state["active"],
                      clock=lambda: T0)
    return eng, state


def run(eng, cs, news=(), health=None):
    out = []
    for c in cs:
        out.append(eng.process_closed_candle(c, book(11, 9, mid=float(c.close)), news, health or {}))
    return out


def test_selloff_produces_exactly_one_mean_reversion_long_trigger():
    eng, _ = make_engine()
    news = [news_event(1, 30, "-0.6", 8), news_event(2, 5, "0.2", 2), news_event(3, 90, "-0.1", 0)]
    health = {"gate": "ACTIVE", "book_age_s": 0.3}
    res = run(eng, crafted_selloff(), news, health)
    triggers = [(s, r, p) for s, r, p in res if r.outcome is O.TRIGGER]
    assert len(triggers) == 1 and eng.stats["TRIGGER"] == 1
    assert sum(1 for _, r, p in res if p is not None) == 1
    snap_, r, p = triggers[0]
    assert snap_.warm and snap_.rsi < 30 and snap_.close <= snap_.bb_lower
    assert p.direction is PositionSide.LONG and p.primary_strategy is StrategyName.MEAN_REVERSION
    assert p.symbol == SYM and p.timeframe == "1m" and p.indicators is snap_
    assert p.proposed_entry == crafted_selloff()[120].close
    atr = D(repr(snap_.atr))
    assert abs(p.proposed_stop_loss - (p.proposed_entry - D("1.5") * atr)) < D("1e-7")
    assert abs(p.proposed_take_profit - (p.proposed_entry + D("3.0") * atr)) < D("1e-7")
    assert p.proposed_stop_loss < p.proposed_entry < p.proposed_take_profit
    assert p.fee_precheck.passes and p.fee_precheck.cover_multiple > 2
    assert p.confluence_score >= 0.7 and p.confluence_factors
    assert any("aşırı satım" in f for f in p.confluence_factors)
    assert [n.news_id for n in p.recent_news] == ["id2", "id1", "id3"]  # newest first
    # importance weighted: (0.2*2 + -0.6*8 + -0.1*0) / 10 = -0.44
    assert p.news_sentiment_avg == pytest.approx(-0.44)
    assert p.market_health == health and p.market_health is not health
    assert eng.stats["CANDLES"] == len(res)
    assert sum(eng.stats[o.value] for o in O) == eng.stats["CANDLES"]


def test_trigger_payload_json_round_trip():
    eng, _ = make_engine()
    res = run(eng, crafted_selloff(), [news_event(1, 10, "-0.5", 5)], {"gate": "ACTIVE", "ages": {"book": 0.2}})
    p = next(p for _, _, p in res if p)
    text = payload_to_json(p)
    back = json.loads(text)
    assert back["signal_id"] == p.signal_id
    assert back["direction"] == "LONG" and back["primary_strategy"] == "MEAN_REVERSION"
    assert Decimal(back["proposed_entry"]) == p.proposed_entry
    assert Decimal(back["proposed_stop_loss"]) == p.proposed_stop_loss
    assert Decimal(back["proposed_take_profit"]) == p.proposed_take_profit
    assert back["indicators"]["rsi"] == p.indicators.rsi
    assert back["fee_precheck"]["passes"] is True
    assert back["market_health"]["ages"]["book"] == 0.2
    assert back["recent_news"][0]["title"] == "Haber 1"
    assert any("aşırı satım" in f for f in back["confluence_factors"])  # ensure_ascii=False keeps Turkish
    assert "aşırı" in text
    assert text == json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True)


def test_paused_gate_never_triggers_and_no_payload():
    eng, state = make_engine(active=False)
    res = run(eng, crafted_selloff())
    assert all(r.outcome is O.DATA_PAUSED and p is None for _, r, p in res)
    assert eng.stats["DATA_PAUSED"] == len(res) and eng.stats["TRIGGER"] == 0
    # indicators kept updating while paused, so the engine is warm when the gate reopens
    assert eng.indicators.latest(SYM).warm


def test_not_warm_before_enough_history():
    eng, _ = make_engine()
    res = run(eng, crafted_selloff()[:30])
    assert all(r.outcome is O.NOT_WARM for _, r, _p in res)


def test_duplicate_old_and_open_candles_are_ignored():
    eng, _ = make_engine()
    cs = crafted_selloff()
    res = run(eng, cs)
    n_trig = eng.stats["TRIGGER"]
    last = res[-1][0]
    snap_d, r, p = eng.process_closed_candle(cs[120], book(11, 9), [], {})  # replay the trigger candle
    assert r.outcome is O.NO_SETUP and p is None and snap_d is last
    open_c = mk_candle(500, 100, 101, 99, 100, 1, closed=False)
    _, r2, p2 = eng.process_closed_candle(open_c, None, [], {})
    assert p2 is None and eng.stats["TRIGGER"] == n_trig and eng.stats["IGNORED"] == 2
    assert eng.indicators.latest(SYM) is last


def test_deterministic_replay():
    outs = []
    for _ in range(2):
        eng, _ = make_engine()
        res = run(eng, crafted_selloff())
        outs.append([(r.outcome, s.rsi) for s, r, _p in res])
    assert outs[0] == outs[1]


def test_payload_requires_trigger_and_caps_news():
    pf_result = PreFilterResult(SYM, O.NO_SETUP, T0, "x")
    with pytest.raises(ValueError):
        build_signal_payload(pf_result, [], snap(), [], {}, max_news=3)
    eng, _ = make_engine()
    sig = signal(score=0.9, atr_pct=0.0034)
    sup = signal(strategy=StrategyName.MOMENTUM, score=0.6, atr_pct=0.0034)
    sup = type(sup)(sup.strategy, sup.symbol, sup.direction, sup.score, ("faktör", "ikinci faktör"), sup.entry,
                    sup.stop_loss, sup.take_profit, sup.timestamp)
    r, support = eng.prefilter.evaluate_with_support(snap(), [sig, sup])
    assert r.outcome is O.TRIGGER
    news = [news_event(i, i * 3, "0.5", 3) for i in range(1, 8)]
    p = build_signal_payload(r, support, snap(), news, {"a": 1}, max_news=3, clock=lambda: T0)
    assert [n.news_id for n in p.recent_news] == ["id1", "id2", "id3"]
    assert p.confluence_factors == ("faktör", "ikinci faktör")  # de-duplicated, order preserved
    assert p.supporting_strategies == (StrategyName.MOMENTUM,)
    assert p.confluence_score == pytest.approx(1.0)  # 0.9 + 0.1
    assert p.created_at == T0
    none = build_signal_payload(r, support, snap(), [], {}, max_news=3)
    assert none.recent_news == () and none.news_sentiment_avg is None
    zero_imp = build_signal_payload(r, support, snap(), [news_event(1, 1, "0.4", 0), news_event(2, 2, "0.2", 0)],
                                    {}, max_news=5)
    assert zero_imp.news_sentiment_avg == pytest.approx(0.3)
    assert build_signal_payload(r, support, snap(), news, {}, max_news=0).recent_news == ()


def test_r2_pipeline_memory_and_latency_50k_candles():
    eng, _ = make_engine()
    cs = synthetic_candles(50_000, seed=5)
    bk = book(11, 9)
    t0 = time.perf_counter()
    for c in cs:
        eng.process_closed_candle(c, bk, (), {})
    per = (time.perf_counter() - t0) / len(cs)
    print(f"full pipeline per candle: {per * 1e6:.1f} us")
    assert per < 1e-3
    q = eng.settings
    assert eng.indicators.memory_footprint(SYM) == {"bb": q.bb_period, "volume_z": q.volume_z_period}
    assert len(eng.prefilter._triggers) <= q.max_triggers_per_hour
    assert len(eng.prefilter._cooldown) <= 2
    assert len(eng.stats) == len(O) + 2
    assert eng.stats["CANDLES"] == 50_000
    assert eng.indicators.symbols() == (SYM,)
