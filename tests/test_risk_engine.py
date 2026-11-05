from datetime import datetime, timezone
from decimal import Decimal as D

import pytest

from app.config.settings import RiskSettings
from app.core.errors import RiskViolation
from app.core.types import FundingInfo, MarginMode, MarketType, OrderType, PositionSide as S, Side, Ticker
from app.risk.kill_switch import KillSwitch
from app.risk.memory import InMemoryKillSwitchStore, InMemoryRiskEventSink
from app.risk.risk_engine import HardRiskEngine, TradeProposal, Verdict, breakeven_stop_price
from app.risk.testing import FakeBroker, make_position, make_snapshot, make_spec

SPEC = make_spec()


def build(**settings):
    sink = InMemoryRiskEventSink()
    s = RiskSettings(**settings)
    ks = KillSwitch(FakeBroker(), InMemoryKillSwitchStore(), sink, s)
    return HardRiskEngine(s, ks, sink), ks, sink, s


def prop(**kw):
    base = dict(symbol="BTC/USDT:USDT", side=S.LONG, entry_price=D(50000), stop_loss=D(49000),
                take_profit=D(52000), leverage=3, confidence=D("0.7"), source="claude", decision_id="d1")
    base.update(kw)
    return TradeProposal(**base)


async def run(engine, p, snap=None, spec=SPEC, **kw):
    return await engine.validate(p, snap or make_snapshot(), spec, kw.get("funding"), kw.get("ticker"))


async def test_clean_long_approved():
    e, _, sink, _ = build()
    d = await run(e, prop())
    assert d.verdict is Verdict.APPROVED and d.approved_quantity == D("0.046")
    assert d.approved_leverage == 3 and d.economics.passes_fee_gate
    assert all(d.checks.values()) and not sink.events


async def test_clean_short_approved():
    e, *_ = build()
    d = await run(e, prop(side=S.SHORT, stop_loss=D(51000), take_profit=D(48000)))
    assert d.verdict is Verdict.APPROVED and d.approved_quantity > 0


async def test_fee_gate_rejects_micro_scalp_and_accepts_one_percent():
    e, _, sink, _ = build()
    scalp = await run(e, prop(take_profit=D(50055), stop_loss=D(49750)))
    assert scalp.verdict is Verdict.REJECTED and "FEE_GATE" in scalp.reason_codes
    msg = next(r for r in scalp.reasons if r.startswith("FEE_GATE"))
    assert "Beklenen brüt kâr" in msg and "eşiğini karşılamıyor" in msg
    assert sink.events[-1].level == "WARN"
    ok = await run(e, prop(take_profit=D(50500), stop_loss=D(49750)))
    assert ok.verdict is Verdict.APPROVED


async def test_leverage_clamped_to_five_reduced():
    e, _, _, _ = build()
    d = await run(e, prop(leverage=10))
    assert d.verdict is Verdict.REDUCED and d.approved_leverage == 5
    assert "LEVERAGE_CLAMPED" in d.reason_codes and not d.checks["leverage_within_limit"]


async def test_leverage_respects_lower_settings_and_spec_caps():
    e, *_ = build(max_leverage=2)
    assert (await run(e, prop(leverage=4))).approved_leverage == 2
    e2, *_ = build()
    d = await run(e2, prop(leverage=5), spec=make_spec(max_leverage=3))
    assert d.approved_leverage == 3


async def test_cross_margin_rejected():
    e, *_ = build()
    d = await run(e, prop(margin_mode=MarginMode.CROSS))
    assert d.verdict is Verdict.REJECTED and "MARGIN_MODE" in d.reason_codes


async def test_long_with_stop_above_entry_rejected():
    e, *_ = build()
    d = await run(e, prop(stop_loss=D(50500)))
    assert d.verdict is Verdict.REJECTED and "STOP_WRONG_SIDE" in d.reason_codes
    assert d.approved_quantity == 0


async def test_short_with_stop_below_entry_and_bad_tp_rejected():
    e, *_ = build()
    d = await run(e, prop(side=S.SHORT, stop_loss=D(49000), take_profit=D(52000)))
    assert {"STOP_WRONG_SIDE", "TAKE_PROFIT_WRONG_SIDE"} <= set(d.reason_codes)


async def test_missing_stop_rejected():
    e, *_ = build()
    d = await run(e, prop(stop_loss=None))
    assert "STOP_LOSS_MISSING" in d.reason_codes and d.verdict is Verdict.REJECTED


async def test_stop_distance_bounds():
    e, *_ = build()
    assert "STOP_TOO_TIGHT" in (await run(e, prop(stop_loss=D("49990")))).reason_codes
    assert "STOP_TOO_WIDE" in (await run(e, prop(stop_loss=D(45000)))).reason_codes


async def test_liquidation_before_stop_reduces_leverage():
    e, *_ = build(max_stop_distance_pct="0.3")
    d = await run(e, prop(stop_loss=D(40000), take_profit=D(60000), leverage=5))
    assert d.verdict is Verdict.REDUCED and d.approved_leverage == 4
    assert "LEVERAGE_LIQ_REDUCED" in d.reason_codes


async def test_liquidation_before_stop_rejects_if_unfixable():
    e, *_ = build(max_stop_distance_pct="0.5")
    d = await run(e, prop(stop_loss=D(200), take_profit=D(90000), leverage=5))
    # defensive branch: stop so deep that even 1x liquidation (~0.5% of entry) is not behind it
    assert d.verdict is Verdict.REJECTED and "LIQ_BEFORE_STOP" in d.reason_codes


async def test_kill_switch_blocks():
    e, ks, *_ = build()
    await ks.engage("TEST")
    d = await run(e, prop())
    assert d.verdict is Verdict.REJECTED and "KILL_SWITCH" in d.reason_codes


async def test_blocked_symbol():
    e, *_ = build(blocked_symbols=["BTC/USDT:USDT"])
    assert "SYMBOL_BLOCKED" in (await run(e, prop())).reason_codes


async def test_spot_cannot_short_and_leverage_forced_to_one():
    e, *_ = build()
    spot = make_spec("BTC/USDT", MarketType.SPOT)
    d = await run(e, prop(symbol="BTC/USDT", side=S.SHORT, stop_loss=D(51000), take_profit=D(48000)), spec=spot)
    assert "SHORT_NOT_SUPPORTED" in d.reason_codes
    d2 = await run(e, prop(symbol="BTC/USDT", leverage=5), spec=spot)
    assert d2.verdict is Verdict.REDUCED and d2.approved_leverage == 1
    assert d2.economics.costs_tp.funding_cost == 0


async def test_max_open_positions():
    e, *_ = build(max_open_positions=2)
    snap = make_snapshot(positions=(make_position("ETH/USDT:USDT"), make_position("SOL/USDT:USDT")))
    assert "MAX_POSITIONS" in (await run(e, prop(), snap)).reason_codes


async def test_opposite_and_pyramiding_rejected():
    e, *_ = build()
    snap_long = make_snapshot(positions=(make_position(side=S.LONG),))
    assert "PYRAMIDING" in (await run(e, prop(), snap_long)).reason_codes
    assert "OPPOSITE_POSITION" in (await run(e, prop(side=S.SHORT, stop_loss=D(51000), take_profit=D(48000)), snap_long)).reason_codes


async def test_requested_quantity_is_min_of_requested_and_sized():
    e, *_ = build()
    small = await run(e, prop(quantity=D("0.01")))
    assert small.approved_quantity == D("0.010") and small.verdict is Verdict.APPROVED
    big = await run(e, prop(quantity=D(5)))
    assert big.approved_quantity == D("0.046") and big.verdict is Verdict.REDUCED
    assert "SIZE_REDUCED" in big.reason_codes


async def test_quantity_below_minimum_rejected():
    e, *_ = build()
    d = await run(e, prop(quantity=D("0.0001")))
    assert d.verdict is Verdict.REJECTED and "BELOW_MINIMUMS" in d.reason_codes


async def test_total_exposure_cap_reduces_then_rejects():
    e, *_ = build(max_total_exposure_pct="0.5")
    existing = make_position("ETH/USDT:USDT", qty="1.7", entry="2500")  # 4250 notional of 5000 cap
    d = await run(e, prop(), make_snapshot(positions=(existing,)))
    assert d.verdict is Verdict.REDUCED and "EXPOSURE_REDUCED" in d.reason_codes
    assert d.approved_quantity * D(50000) <= D(750)
    full = make_position("ETH/USDT:USDT", qty="2", entry="2500")
    d2 = await run(e, prop(), make_snapshot(positions=(full,)))
    assert "EXPOSURE_CAP" in d2.reason_codes


async def test_insufficient_margin():
    e, *_ = build()
    d = await run(e, prop(), make_snapshot(available="1"))
    assert d.verdict is Verdict.REJECTED and "INSUFFICIENT_MARGIN" in d.reason_codes
    d2 = await run(e, prop(leverage=1), make_snapshot(available="1500"))
    assert d2.verdict is Verdict.REDUCED and "MARGIN_REDUCED" in d2.reason_codes
    assert d2.approved_quantity * D(50000) <= D(1500)


async def test_low_confidence_threshold_is_constructor_arg():
    e, *_ = build()
    assert "LOW_CONFIDENCE" in (await run(e, prop(confidence=D("0.54")))).reason_codes
    sink = InMemoryRiskEventSink()
    s = RiskSettings()
    e2 = HardRiskEngine(s, KillSwitch(FakeBroker(), InMemoryKillSwitchStore(), sink, s), sink, min_confidence=D("0.9"))
    assert "LOW_CONFIDENCE" in (await e2.validate(prop(confidence=D("0.8")), make_snapshot(), SPEC, None, None)).reason_codes


async def test_collects_all_failures_and_emits_warn_per_rejection():
    e, _, sink, _ = build()
    d = await run(e, prop(margin_mode=MarginMode.CROSS, confidence=D("0.1"), stop_loss=D(50500)))
    assert {"MARGIN_MODE", "STOP_WRONG_SIDE", "LOW_CONFIDENCE"} <= set(d.reason_codes)
    assert len(sink.by_code("RISK_REJECTED")) == 1 and sink.events[0].level == "WARN"


async def test_funding_raises_cost_for_longs():
    e, *_ = build()
    base = await run(e, prop())
    f = FundingInfo("BTC/USDT:USDT", D("0.001"), 8, None)
    paid = await run(e, prop(), funding=f)
    assert paid.economics.cost_at_tp > base.economics.cost_at_tp


async def test_stale_entry_vs_ticker_rejected():
    e, *_ = build()
    t = Ticker("BTC/USDT:USDT", D(55000), D(55010), D(55005), None, datetime.now(timezone.utc))
    assert "ENTRY_STALE" in (await run(e, prop(), ticker=t)).reason_codes


async def test_limit_entry_uses_maker_fee():
    e, *_ = build()
    m = await run(e, prop())
    l = await run(e, prop(entry_order_type=OrderType.LIMIT))
    assert l.economics.costs_tp.entry_fee / (l.approved_quantity * 50000) == D("0.0002")
    assert m.economics.costs_tp.entry_fee / (m.approved_quantity * 50000) == D("0.0005")


async def test_llm_cannot_raise_limits():
    # nothing on the proposal (confidence 1, source claude, huge quantity/leverage) can exceed settings
    e, _, _, s = build()
    d = await run(e, prop(confidence=D(1), leverage=125, quantity=D(1000)))
    assert d.approved_leverage <= 5
    assert d.approved_quantity * D(50000) <= s.max_position_notional_pct * D(10000)


# ---------------------------------------------------------------- protective orders
async def test_protective_orders_long():
    e, _, _, s = build()
    p = prop(stop_loss=D("49000.07"), take_profit=D("52000.07"))
    d = await run(e, p)
    stop, tp, trail = e.compute_protective_orders(d, p, SPEC)
    assert stop.order_type is OrderType.STOP_MARKET and stop.reduce_only and stop.side is Side.SELL
    assert stop.stop_price == D("49000.0")  # long stop rounded DOWN (away)
    assert tp.order_type is OrderType.TAKE_PROFIT_MARKET and tp.reduce_only and tp.stop_price == D("52000.0")
    assert trail.order_type is OrderType.TRAILING_STOP_MARKET and trail.reduce_only
    assert trail.trailing_activation_price == D("50999.9") or trail.trailing_activation_price == D("51000.0")
    assert trail.trailing_activation_price >= D("50999.93")  # entry + 1R, never earlier
    assert trail.trailing_callback_rate == D("0.010")  # 2% stop distance / 2
    assert stop.quantity == tp.quantity == trail.quantity == d.approved_quantity
    assert stop.metadata["decision_id"] == "d1"


async def test_protective_orders_short_round_up_and_callback_clamp():
    e, *_ = build(min_stop_distance_pct="0")
    p = prop(side=S.SHORT, stop_loss=D("50100.03"), take_profit=D(48000))
    d = await run(e, p)
    stop, tp, trail = e.compute_protective_orders(d, p, SPEC)
    assert stop.side is Side.BUY and stop.stop_price == D("50100.1")  # short stop rounded UP
    assert trail.trailing_callback_rate == D("0.001")  # 0.2%/2 = 0.1% (floor of clamp)
    wide = prop(stop_loss=D(49000))
    assert e.compute_protective_orders(await run(e, wide), wide, SPEC)[2].trailing_callback_rate <= D("0.05")


async def test_protective_orders_refuse_rejected_decision():
    e, *_ = build()
    p = prop(stop_loss=D(50500))
    d = await run(e, p)
    with pytest.raises(RiskViolation):
        e.compute_protective_orders(d, p, SPEC)


def test_breakeven_stop_price_locks_net_zero():
    pos = make_position(entry="50000")
    px = breakeven_stop_price(pos, SPEC.fees)
    assert px == D(50000) * (1 + D("0.0005")) / (1 - D("0.0005"))
    rounded = breakeven_stop_price(pos, SPEC.fees, D(2), D("0.1"))
    assert rounded > px and rounded % D("0.1") == 0
    short = make_position(side=S.SHORT, entry="50000")
    assert breakeven_stop_price(short, SPEC.fees) < D(50000)
