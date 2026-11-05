"""Risk-based position sizing. Decimal only, no broker dependency."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.config.settings import RiskSettings
from app.core.types import FeeSchedule, InstrumentSpec
from app.risk.rounding import floor_to_step

ZERO = Decimal(0)
BPS = Decimal(10000)


@dataclass(frozen=True, slots=True)
class SizingResult:
    quantity: Decimal
    notional: Decimal
    required_margin: Decimal
    risk_amount: Decimal
    capped_by: str | None = None  # "MAX_POSITION_NOTIONAL" when the notional cap bound
    reason: str | None = None  # set when quantity == 0


def _zero(risk_amount: Decimal, reason: str) -> SizingResult:
    return SizingResult(ZERO, ZERO, ZERO, risk_amount, None, reason)


def size_position(
    equity: Decimal,
    entry: Decimal,
    stop: Decimal,
    spec: InstrumentSpec,
    risk_settings: RiskSettings,
    requested_leverage: int,
    fees: FeeSchedule,
) -> SizingResult:
    """qty = equity*risk_pct / (|entry-stop| + taker fees on both legs + slippage on both legs), per unit.

    Capped by max_position_notional_pct * equity / entry, floored to ``step_size``.
    """
    risk_amount = equity * risk_settings.max_risk_per_trade_pct
    if equity <= 0 or entry <= 0 or stop <= 0:
        return _zero(risk_amount, "INVALID_INPUT")
    distance = abs(entry - stop)
    if distance == 0:
        return _zero(risk_amount, "ZERO_STOP_DISTANCE")
    slip = risk_settings.default_slippage_bps / BPS
    per_unit = distance + (entry + stop) * fees.taker_rate + (entry + stop) * slip
    qty = risk_amount / per_unit
    capped_by: str | None = None
    cap_qty = risk_settings.max_position_notional_pct * equity / entry
    if qty > cap_qty:
        qty = cap_qty
        capped_by = "MAX_POSITION_NOTIONAL"
    qty = floor_to_step(qty, spec.step_size)
    if qty <= 0 or qty < spec.min_qty:
        return _zero(risk_amount, "BELOW_MIN_QTY")
    notional = qty * entry
    if notional < spec.min_notional:
        return _zero(risk_amount, "BELOW_MIN_NOTIONAL")
    leverage = Decimal(max(1, requested_leverage))
    return SizingResult(qty, notional, notional / leverage, risk_amount, capped_by, None)
