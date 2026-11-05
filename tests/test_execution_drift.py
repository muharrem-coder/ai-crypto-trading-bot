"""Rule 3: price-drift guard (pure function)."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from app.claude.contracts import ClaudeDecision, DecisionAction, EntryType
from app.config.settings import ExecutionSettings
from app.core.types import PositionSide as S
from app.data.models import OrderBookSnapshot
from app.execution.contracts import ExecutionStatus
from app.execution.drift_guard import check_price_drift
from app.quant.contracts import FeePreCheck, IndicatorSnapshot, SignalPayload, StrategyName

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
SYM = "BTC/USDT:USDT"
SETTINGS = ExecutionSettings()  # max_drift 0.3, max_abs 1.0, book age 2s


def payload(direction=S.LONG, entry="50000", atr=100.0) -> SignalPayload:
    ind = IndicatorSnapshot(SYM, "1m", NOW, 50000.0, 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, atr, 0.002, 0.0, 1.0,
                            0.0, True)
    fee = FeePreCheck(D(1), D(1), D(2), D(2), D(1), True)
    stop, tp = (D("49000"), D("52000")) if direction is S.LONG else (D("51000"), D("48000"))
    return SignalPayload("sig-1", NOW, SYM, "1m", direction, StrategyName.MOMENTUM, 0.8, ("x",), (), D(entry), stop, tp,
                         ind, fee, {}, (), None)


def decision(action=DecisionAction.LONG, entry_type=EntryType.MARKET, entry_price=None) -> ClaudeDecision:
    long = action is DecisionAction.LONG
    return ClaudeDecision(action=action, confidence=D("0.7"), leverage=5, entry_type=entry_type,
                          entry_price=entry_price, stop_loss=D("49000") if long else D("51000"),
                          take_profit=D("52000") if long else D("48000"),
                          reasoning_tr="Test kararı için yeterince uzun gerekçe metni.")


def book(bid="49999.9", ask="50000", age=0.0) -> OrderBookSnapshot:
    return OrderBookSnapshot(SYM, ((D(bid), D(1)),), ((D(ask), D(1)),), None, NOW - timedelta(seconds=age))


def test_no_drift_ok_and_reference_is_best_ask_for_long():
    c = check_price_drift(payload(), decision(), book(), NOW, SETTINGS)
    assert c.ok and c.status is None and c.reference_price == D("50000") and c.drift_atr == 0


def test_short_reference_is_best_bid():
    c = check_price_drift(payload(S.SHORT), decision(DecisionAction.SHORT), book(bid="50000", ask="50000.1"), NOW,
                          SETTINGS)
    assert c.ok and c.reference_price == D("50000")


def test_long_adverse_035_atr_aborts():
    c = check_price_drift(payload(), decision(), book(ask="50035"), NOW, SETTINGS)
    assert not c.ok and c.status is ExecutionStatus.ABORTED_PRICE_DRIFT
    assert c.drift_atr == D("0.35") and c.reference_price == D("50035")


def test_exactly_at_threshold_is_allowed():
    c = check_price_drift(payload(), decision(), book(ask="50030"), NOW, SETTINGS)
    assert c.ok and c.drift_atr == D("0.3")


def test_short_adverse_when_bid_falls():
    c = check_price_drift(payload(S.SHORT), decision(DecisionAction.SHORT), book(bid="49965", ask="49966"), NOW, SETTINGS)
    assert not c.ok and c.status is ExecutionStatus.ABORTED_PRICE_DRIFT and c.drift_atr == D("0.35")


def test_favourable_small_drift_ok_and_negative():
    c = check_price_drift(payload(), decision(), book(bid="49979", ask="49980"), NOW, SETTINGS)
    assert c.ok and c.drift_atr == D("-0.2") and c.abs_drift_atr == D("0.2")


def test_huge_favourable_jump_is_anomaly():
    c = check_price_drift(payload(), decision(), book(bid="49889", ask="49890"), NOW, SETTINGS)  # -1.1 ATR
    assert not c.ok and c.status is ExecutionStatus.ABORTED_PRICE_DRIFT and "ABS_DRIFT" in c.detail


def test_stale_book_aborts():
    c = check_price_drift(payload(), decision(), book(age=3), NOW, SETTINGS)
    assert not c.ok and c.status is ExecutionStatus.ABORTED_STALE_DATA and c.book_age_s == 3.0


def test_book_exactly_at_max_age_ok():
    assert check_price_drift(payload(), decision(), book(age=2.0), NOW, SETTINGS).ok


def test_missing_and_empty_book_abort_stale():
    assert check_price_drift(payload(), decision(), None, NOW, SETTINGS).status is ExecutionStatus.ABORTED_STALE_DATA
    empty = OrderBookSnapshot(SYM, (), (), None, NOW)
    assert check_price_drift(payload(), decision(), empty, NOW, SETTINGS).status is ExecutionStatus.ABORTED_STALE_DATA


def test_limit_uses_decision_entry_price_not_book():
    d = decision(entry_type=EntryType.LIMIT, entry_price=D("50040"))
    c = check_price_drift(payload(), d, book(ask="50000"), NOW, SETTINGS)
    assert not c.ok and c.reference_price == D("50040") and c.drift_atr == D("0.4")
    d2 = decision(entry_type=EntryType.LIMIT, entry_price=D("49950"))
    assert check_price_drift(payload(), d2, book(), NOW, SETTINGS).ok


@pytest.mark.parametrize("atr", [None, 0.0, -1.0, float("nan")])
def test_unusable_atr_fails_closed(atr):
    c = check_price_drift(payload(atr=atr), decision(), book(), NOW, SETTINGS)
    assert not c.ok and c.status is ExecutionStatus.ABORTED_PRICE_DRIFT and "ATR_UNAVAILABLE" in c.detail
