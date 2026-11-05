from decimal import Decimal as D

from app.core.types import FeeSchedule, LiquidityRole as R, MarketType, PositionSide as S
from app.risk.fee_calculator import FeeCalculator, funding_intervals

FEES = FeeSchedule(D("0.0002"), D("0.0005"))
KW = dict(fees=FEES, entry_role=R.TAKER, exit_role=R.TAKER, slippage_bps=D(2),
          funding_rate_per_interval=D(0), expected_holding_hours=D(4), funding_interval_hours=8)


def test_round_trip_basic():
    c = FeeCalculator.round_trip_cost(D(100), D(101), D(10), side=S.LONG, **KW)
    assert c.entry_fee == D("0.5") and c.exit_fee == D("0.505")
    assert c.slippage_cost == D("0.4020")
    assert c.funding_cost == 0
    assert c.total == c.entry_fee + c.exit_fee + c.slippage_cost


def test_maker_cheaper_than_taker():
    kw = {**KW, "entry_role": R.MAKER}
    assert FeeCalculator.round_trip_cost(D(100), D(101), D(1), side=S.LONG, **kw).entry_fee == D("0.02")


def test_funding_direction_and_intervals():
    kw = {**KW, "funding_rate_per_interval": D("0.0001"), "expected_holding_hours": D(9)}
    assert funding_intervals(D(9), 8) == 2
    assert funding_intervals(D(8), 8) == 1
    assert funding_intervals(D(0), 8) == 0
    long = FeeCalculator.round_trip_cost(D(100), D(101), D(10), side=S.LONG, **kw)
    short = FeeCalculator.round_trip_cost(D(100), D(99), D(10), side=S.SHORT, **kw)
    assert long.funding_cost == D("0.2")  # 1000 * 0.0001 * 2
    assert short.funding_cost == D("-0.2")  # shorts receive


def test_negative_rate_longs_receive():
    kw = {**KW, "funding_rate_per_interval": D("-0.0001")}
    assert FeeCalculator.round_trip_cost(D(100), D(101), D(10), side=S.LONG, **kw).funding_cost < 0


def test_spot_has_no_funding():
    kw = {**KW, "funding_rate_per_interval": D("0.01"), "market_type": MarketType.SPOT}
    assert FeeCalculator.round_trip_cost(D(100), D(101), D(10), side=S.LONG, **kw).funding_cost == 0


def econ(entry, tp, sl, side=S.LONG, qty=D(1), **over):
    kw = {**KW, **over}
    return FeeCalculator.evaluate_trade_economics(side, D(entry), D(tp), D(sl), qty, **kw)


def test_micro_scalp_fails_fee_gate_and_one_percent_passes():
    # 0.11% move vs 0.05% taker x2 legs (+ slippage) x2 multiple
    scalp = econ("50000", "50055", "49750", qty=D("0.1"))
    assert scalp.expected_gross_profit == D("5.5")
    assert not scalp.passes_fee_gate
    big = econ("50000", "50500", "49750", qty=D("0.1"))
    assert big.passes_fee_gate
    assert big.net_profit_at_tp < big.expected_gross_profit


def test_economics_long_and_short_symmetry():
    long = econ("100", "102", "99", qty=D(10))
    short = econ("100", "98", "101", side=S.SHORT, qty=D(10))
    assert long.expected_gross_profit == 20 and long.expected_gross_loss == 10
    assert short.expected_gross_profit == 20 and short.expected_gross_loss == 10
    assert long.net_loss_at_sl > long.expected_gross_loss
    assert long.reward_risk_net < D(2)


def test_breakeven_gives_zero_net_pnl():
    for side in (S.LONG, S.SHORT):
        be = FeeCalculator.breakeven_price(side, D(100), FEES, R.TAKER, R.TAKER, D(2), D("0.0001"), 2)
        c = FeeCalculator.round_trip_cost(D(100), be, D(1), FEES, R.TAKER, R.TAKER, D(2), D("0.0001"), D(16), 8, side)
        gross = (be - 100) if side is S.LONG else (100 - be)
        assert abs(gross - c.total) < D("1e-20")
    assert FeeCalculator.breakeven_price(S.LONG, D(100), FEES) > 100
    assert FeeCalculator.breakeven_price(S.SHORT, D(100), FEES) < 100


def test_min_net_profit_enforced():
    e = econ("100", "101", "99", qty=D("0.0001"), min_net_profit_usd=D(1))
    assert not e.passes_fee_gate


def test_min_profitable_move_pct():
    e = econ("100", "103", "99")
    assert e.min_profitable_move_pct > D("0.002")  # (0.05%*2 + 0.04%) * 2
    assert e.min_profitable_move_pct < D("0.003")
