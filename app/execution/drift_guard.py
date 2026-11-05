"""Rule 3: price-drift guard. Pure function, no I/O.

Between the moment a signal fired and the moment we actually send the entry, the market may have run away from
us (Claude's latency alone is seconds). The guard compares the CURRENT executable price with the price the signal
(and therefore the stop / take-profit geometry) was built on, measured in ATR units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.claude.contracts import ClaudeDecision, DecisionAction, EntryType
from app.config.settings import ExecutionSettings
from app.core.types import PositionSide
from app.data.models import OrderBookSnapshot
from app.execution.contracts import ExecutionStatus
from app.quant.contracts import SignalPayload


@dataclass(frozen=True, slots=True)
class DriftCheck:
    ok: bool
    status: ExecutionStatus | None  # None when ok, else ABORTED_STALE_DATA / ABORTED_PRICE_DRIFT
    reference_price: Decimal | None  # best ask (LONG) / best bid (SHORT); decision.entry_price for LIMIT
    drift_atr: Decimal | None  # ADVERSE drift in ATR units: positive = worse for us, negative = favourable
    abs_drift_atr: Decimal | None
    book_age_s: float | None
    detail: str


def _fail(status: ExecutionStatus, detail: str, *, ref: Decimal | None = None, drift: Decimal | None = None,
          age: float | None = None) -> DriftCheck:
    return DriftCheck(False, status, ref, drift, abs(drift) if drift is not None else None, age, detail)


def check_price_drift(
    payload: SignalPayload,
    decision: ClaudeDecision,
    book: OrderBookSnapshot | None,
    now: datetime,
    settings: ExecutionSettings,
) -> DriftCheck:
    """Decide whether the market is still close enough to ``payload.proposed_entry`` to execute.

    * missing / stale / empty book           -> ABORTED_STALE_DATA
    * ATR unusable (cannot size the drift)   -> ABORTED_PRICE_DRIFT (fail closed)
    * adverse drift > max_drift_atr, or |drift| > max_abs_drift_atr -> ABORTED_PRICE_DRIFT
    """
    if book is None:
        return _fail(ExecutionStatus.ABORTED_STALE_DATA, "NO_BOOK: order book unavailable at execution time")
    age = max((now - book.received_at).total_seconds(), 0.0)
    if age > settings.max_book_age_seconds:
        return _fail(ExecutionStatus.ABORTED_STALE_DATA,
                     f"STALE_BOOK: order book is {age:.2f}s old (max {settings.max_book_age_seconds}s)", age=age)

    side = (PositionSide.SHORT if decision.action is DecisionAction.SHORT
            else PositionSide.LONG if decision.action is DecisionAction.LONG else payload.direction)

    if decision.entry_type is EntryType.LIMIT:
        ref = decision.entry_price
        if ref is None:
            return _fail(ExecutionStatus.ABORTED_PRICE_DRIFT, "LIMIT_PRICE_MISSING: LIMIT decision has no entry_price",
                         age=age)
    else:
        ref = book.best_ask if side is PositionSide.LONG else book.best_bid
        if ref is None or ref <= 0:
            return _fail(ExecutionStatus.ABORTED_STALE_DATA,
                         f"EMPTY_BOOK: no {'ask' if side is PositionSide.LONG else 'bid'} side in the order book",
                         age=age)

    atr_raw = payload.indicators.atr
    if atr_raw is None or not math.isfinite(atr_raw) or atr_raw <= 0:
        return _fail(ExecutionStatus.ABORTED_PRICE_DRIFT, f"ATR_UNAVAILABLE: cannot measure drift (atr={atr_raw})",
                     ref=ref, age=age)
    atr = Decimal(str(atr_raw))

    proposed = payload.proposed_entry
    adverse = (ref - proposed) if side is PositionSide.LONG else (proposed - ref)
    drift = adverse / atr
    absd = abs(drift)

    if drift > settings.max_drift_atr:
        return DriftCheck(False, ExecutionStatus.ABORTED_PRICE_DRIFT, ref, drift, absd, age,
                          f"ADVERSE_DRIFT: {drift:.3f} ATR > {settings.max_drift_atr} ATR "
                          f"(ref {ref} vs signal {proposed})")
    if absd > settings.max_abs_drift_atr:
        return DriftCheck(False, ExecutionStatus.ABORTED_PRICE_DRIFT, ref, drift, absd, age,
                          f"ABS_DRIFT: |{drift:.3f}| ATR > {settings.max_abs_drift_atr} ATR (anomalous jump, "
                          f"ref {ref} vs signal {proposed})")
    return DriftCheck(True, None, ref, drift, absd, age, f"OK: drift {drift:.3f} ATR, book age {age:.2f}s")
