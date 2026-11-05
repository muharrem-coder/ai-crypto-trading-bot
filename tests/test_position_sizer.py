from decimal import Decimal as D

from app.config.settings import RiskSettings
from app.risk.position_sizer import size_position
from app.risk.testing import make_spec

SPEC = make_spec()
RS = RiskSettings()


def test_risk_based_quantity():
    r = size_position(D(10000), D(50000), D(49000), SPEC, RS, 3, SPEC.fees)
    # risk 50 / (1000 + 99000*0.0005 + 99000*0.0002) = 50/1069.3
    assert r.risk_amount == D(50)
    assert r.quantity == D("0.046")
    assert r.capped_by is None and r.reason is None
    assert r.notional == D("2300.000")
    assert r.required_margin == r.notional / 3
    # actual loss incl. costs never exceeds the risk budget
    loss = r.quantity * (1000 + 99000 * D("0.0007"))
    assert loss <= 50


def test_short_symmetric():
    long = size_position(D(10000), D(50000), D(49000), SPEC, RS, 1, SPEC.fees)
    short = size_position(D(10000), D(50000), D(51000), SPEC, RS, 1, SPEC.fees)
    assert abs(long.quantity - short.quantity) <= D("0.001")


def test_notional_cap():
    r = size_position(D(10000), D(50000), D("49950"), SPEC, RS, 5, SPEC.fees)
    assert r.capped_by == "MAX_POSITION_NOTIONAL"
    assert r.notional <= D(10000) * RS.max_position_notional_pct


def test_floors_to_step_not_rounds():
    r = size_position(D(10000), D(50000), D(49000), SPEC, RS, 1, SPEC.fees)
    assert r.quantity % SPEC.step_size == 0


def test_too_small_returns_zero_with_reason():
    r = size_position(D(10), D(50000), D(49000), SPEC, RS, 1, SPEC.fees)
    assert r.quantity == 0 and r.reason == "BELOW_MIN_QTY"


def test_below_min_notional():
    spec = make_spec()
    object.__setattr__(spec, "min_notional", D(100000))
    r = size_position(D(10000), D(50000), D(49000), spec, RS, 1, spec.fees)
    assert r.quantity == 0 and r.reason == "BELOW_MIN_NOTIONAL"


def test_zero_distance_and_bad_inputs():
    assert size_position(D(10000), D(50000), D(50000), SPEC, RS, 1, SPEC.fees).reason == "ZERO_STOP_DISTANCE"
    assert size_position(D(0), D(50000), D(49000), SPEC, RS, 1, SPEC.fees).reason == "INVALID_INPUT"


def test_decimal_only():
    r = size_position(D(10000), D(50000), D(49000), SPEC, RS, 2, SPEC.fees)
    for v in (r.quantity, r.notional, r.required_margin, r.risk_amount):
        assert isinstance(v, D)
