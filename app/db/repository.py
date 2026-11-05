"""Async repository helpers used by the risk and execution layers.

Every function takes an ``AsyncSession`` as its first argument, flushes (never commits) and
returns the ORM row. Transaction boundaries belong to the caller (see ``get_session``).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import QuantCoreError
from app.core.types import (
    AccountSnapshot,
    AssetClass,
    Fill,
    MarginMode,
    MarketType,
    Order,
    PositionSide,
    TradingMode,
    utcnow,
)
from app.db.models import (
    Account,
    ClaudeDecision,
    DecisionAction,
    EquitySnapshot,
    ExitReason,
    FillRecord,
    KillSwitchState,
    OrderPurpose,
    OrderRecord,
    RiskEvent,
    RiskLevel,
    RiskVerdict,
    Trade,
    TradeStatus,
    jsonable,
)

ZERO = Decimal("0")
_R_PRECISION = Decimal("0.00000001")


class RecordNotFoundError(QuantCoreError):
    """A referenced row (decision, order, trade) does not exist."""


class TradeStateError(QuantCoreError):
    """Illegal trade lifecycle transition (e.g. closing an already closed trade)."""


# --------------------------------------------------------------------------------------
# Accounts
# --------------------------------------------------------------------------------------


async def get_or_create_account(
    session: AsyncSession,
    *,
    broker: str,
    asset_class: AssetClass,
    market_type: MarketType,
    trading_mode: TradingMode,
    quote_currency: str,
) -> Account:
    stmt = select(Account).where(
        Account.broker == broker,
        Account.asset_class == asset_class,
        Account.market_type == market_type,
        Account.trading_mode == trading_mode,
    )
    account = (await session.execute(stmt)).scalar_one_or_none()
    if account is None:
        account = Account(
            broker=broker,
            asset_class=asset_class,
            market_type=market_type,
            trading_mode=trading_mode,
            quote_currency=quote_currency,
        )
        session.add(account)
        await session.flush()
    return account


# --------------------------------------------------------------------------------------
# Risk events
# --------------------------------------------------------------------------------------


async def record_risk_event(
    session: AsyncSession,
    *,
    level: RiskLevel,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    ts: datetime | None = None,
) -> RiskEvent:
    """Persist a risk rejection / reduction / kill-switch event."""
    event = RiskEvent(
        level=level,
        code=code,
        message=message,
        details=jsonable(details) if details is not None else None,
        ts=ts or utcnow(),
    )
    session.add(event)
    await session.flush()
    return event


# --------------------------------------------------------------------------------------
# Claude decisions
# --------------------------------------------------------------------------------------


async def save_decision(
    session: AsyncSession,
    *,
    symbol: str,
    model: str,
    action: DecisionAction,
    confidence: Decimal,
    reasoning: str,
    signal_id: uuid.UUID | None = None,
    leverage: int | None = None,
    stop_loss: Decimal | None = None,
    take_profit: Decimal | None = None,
    input_context: dict[str, Any] | None = None,
    raw_response: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    risk_verdict: RiskVerdict | None = None,
    risk_reasons: list[Any] | dict[str, Any] | None = None,
    final_leverage: int | None = None,
    final_quantity: Decimal | None = None,
    decision_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> ClaudeDecision:
    """Store one Claude decision with its Turkish ``reasoning`` and the exact context sent."""
    decision = ClaudeDecision(
        id=decision_id or uuid.uuid4(),
        created_at=created_at or utcnow(),
        signal_id=signal_id,
        symbol=symbol,
        model=model,
        action=action,
        confidence=confidence,
        leverage=leverage,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning=reasoning,
        input_context=jsonable(input_context) if input_context is not None else None,
        raw_response=jsonable(raw_response) if raw_response is not None else None,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        risk_verdict=risk_verdict,
        risk_reasons=jsonable(risk_reasons) if risk_reasons is not None else None,
        final_leverage=final_leverage,
        final_quantity=final_quantity,
    )
    session.add(decision)
    await session.flush()
    return decision


async def update_decision_risk_verdict(
    session: AsyncSession,
    decision_id: uuid.UUID,
    verdict: RiskVerdict,
    reasons: list[Any] | dict[str, Any] | None = None,
    final_leverage: int | None = None,
    final_quantity: Decimal | None = None,
) -> ClaudeDecision:
    """Attach the hard-risk engine's verdict (and any reduced leverage / size) to a decision."""
    decision = await session.get(ClaudeDecision, decision_id)
    if decision is None:
        raise RecordNotFoundError(f"claude_decisions row {decision_id} not found")
    decision.risk_verdict = verdict
    decision.risk_reasons = jsonable(reasons) if reasons is not None else None
    decision.final_leverage = final_leverage
    decision.final_quantity = final_quantity
    await session.flush()
    return decision


async def list_recent_decisions(session: AsyncSession, limit: int = 50, symbol: str | None = None) -> list[ClaudeDecision]:
    """Newest first; backs the terminal's "Claude neden bu işlemi açtı?" panel."""
    stmt = select(ClaudeDecision).order_by(ClaudeDecision.created_at.desc()).limit(max(1, limit))
    if symbol is not None:
        stmt = stmt.where(ClaudeDecision.symbol == symbol)
    return list((await session.execute(stmt)).scalars())


# --------------------------------------------------------------------------------------
# Orders and fills
# --------------------------------------------------------------------------------------

_TERMINAL = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}


async def _find_order(session: AsyncSession, account_id: int, client_order_id: str | None, exchange_order_id: str | None) -> OrderRecord | None:
    if client_order_id:
        row = (await session.execute(select(OrderRecord).where(OrderRecord.client_order_id == client_order_id))).scalar_one_or_none()
        if row is not None:
            return row
    if exchange_order_id:
        stmt = select(OrderRecord).where(
            OrderRecord.account_id == account_id, OrderRecord.exchange_order_id == exchange_order_id
        )
        return (await session.execute(stmt)).scalar_one_or_none()
    return None


async def upsert_order_from_domain(
    session: AsyncSession,
    order: Order,
    purpose: OrderPurpose,
    decision_id: uuid.UUID | None,
    account_id: int,
) -> OrderRecord:
    """Insert or update the persisted order matching ``client_order_id`` / exchange order id.

    A stale update (non-terminal status arriving after a terminal one) is ignored, and
    ``filled_quantity`` never moves backwards. ``purpose`` / ``decision_id`` of an existing
    row are kept unless the row has none.
    """
    row = await _find_order(session, account_id, order.client_order_id, order.order_id)

    if row is None:
        row = OrderRecord(
            account_id=account_id,
            decision_id=decision_id,
            exchange_order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            status=order.status,
            purpose=purpose,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            reduce_only=order.reduce_only,
            fee_paid=order.fee_paid,
            fee_currency=order.fee_currency,
            created_at=order.created_at,
            updated_at=order.updated_at,
            raw=jsonable(order.raw),
        )
        session.add(row)
        await session.flush()
        return row

    row_terminal = row.status.value in _TERMINAL
    if row_terminal and order.status.value not in _TERMINAL:
        return row  # stale update

    row.exchange_order_id = order.order_id or row.exchange_order_id
    row.client_order_id = order.client_order_id or row.client_order_id
    row.status = order.status
    row.quantity = order.quantity
    row.price = order.price
    row.stop_price = order.stop_price
    row.filled_quantity = max(order.filled_quantity, row.filled_quantity)
    row.average_fill_price = order.average_fill_price if order.average_fill_price is not None else row.average_fill_price
    row.reduce_only = order.reduce_only
    row.fee_paid = order.fee_paid
    row.fee_currency = order.fee_currency or row.fee_currency
    row.updated_at = order.updated_at
    row.raw = jsonable(order.raw)
    if row.decision_id is None:
        row.decision_id = decision_id
    await session.flush()
    return row


async def record_fill(session: AsyncSession, fill: Fill, account_id: int) -> FillRecord:
    """Persist a fill idempotently (unique on order + exchange trade id).

    ``fill.order_id`` is the exchange order id; the order must already be stored via
    :func:`upsert_order_from_domain`.
    """
    order = (
        await session.execute(
            select(OrderRecord).where(OrderRecord.account_id == account_id, OrderRecord.exchange_order_id == fill.order_id)
        )
    ).scalar_one_or_none()
    if order is None:
        raise RecordNotFoundError(f"order with exchange id {fill.order_id!r} not found for account {account_id}")

    existing = (
        await session.execute(
            select(FillRecord).where(FillRecord.order_id == order.id, FillRecord.exchange_trade_id == fill.trade_id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    record = FillRecord(
        order_id=order.id,
        exchange_trade_id=fill.trade_id,
        symbol=fill.symbol,
        side=fill.side,
        price=fill.price,
        quantity=fill.quantity,
        fee=fill.fee,
        fee_currency=fill.fee_currency,
        liquidity=fill.liquidity,
        ts=fill.timestamp,
    )
    session.add(record)
    await session.flush()
    return record


# --------------------------------------------------------------------------------------
# Trades (round-trip ledger)
# --------------------------------------------------------------------------------------


async def open_trade(
    session: AsyncSession,
    *,
    account_id: int,
    symbol: str,
    side: PositionSide,
    leverage: int,
    margin_mode: MarginMode,
    entry_price: Decimal,
    quantity: Decimal,
    initial_stop: Decimal | None = None,
    decision_id: uuid.UUID | None = None,
    fees_paid: Decimal = ZERO,
    opened_at: datetime | None = None,
    trade_id: uuid.UUID | None = None,
) -> Trade:
    trade = Trade(
        id=trade_id or uuid.uuid4(),
        account_id=account_id,
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        leverage=leverage,
        margin_mode=margin_mode,
        opened_at=opened_at or utcnow(),
        entry_price=entry_price,
        quantity=quantity,
        initial_stop=initial_stop,
        fees_paid=fees_paid,
        funding_paid=ZERO,
        status=TradeStatus.OPEN,
    )
    session.add(trade)
    await session.flush()
    return trade


async def close_trade(
    session: AsyncSession,
    trade_id: uuid.UUID,
    *,
    exit_price: Decimal,
    exit_reason: ExitReason,
    fees_paid: Decimal | None = None,
    funding_paid: Decimal | None = None,
    closed_at: datetime | None = None,
) -> Trade:
    """Close a trade and compute its PnL.

    * ``gross_pnl`` = (exit - entry) * qty for LONG, (entry - exit) * qty for SHORT.
    * ``net_pnl``   = gross_pnl - fees_paid - funding_paid.
    * ``r_multiple`` = net_pnl / initial risk, where initial risk = |entry - initial_stop| * qty
      (``None`` when no usable initial stop was recorded).

    ``fees_paid`` / ``funding_paid`` are round-trip totals (entry + exit fees, net funding
    paid; negative funding means funding received). ``None`` keeps the value already stored.
    """
    trade = (await session.execute(select(Trade).where(Trade.id == trade_id).with_for_update())).scalar_one_or_none()
    if trade is None:
        raise RecordNotFoundError(f"trades row {trade_id} not found")
    if trade.status is TradeStatus.CLOSED:
        raise TradeStateError(f"trade {trade_id} is already closed")

    if fees_paid is not None:
        trade.fees_paid = fees_paid
    if funding_paid is not None:
        trade.funding_paid = funding_paid

    if trade.side is PositionSide.LONG:
        gross = (exit_price - trade.entry_price) * trade.quantity
    else:
        gross = (trade.entry_price - exit_price) * trade.quantity
    net = gross - trade.fees_paid - trade.funding_paid

    r_multiple: Decimal | None = None
    if trade.initial_stop is not None:
        risk = abs(trade.entry_price - trade.initial_stop) * trade.quantity
        if risk > ZERO:
            r_multiple = (net / risk).quantize(_R_PRECISION)

    trade.exit_price = exit_price
    trade.exit_reason = exit_reason
    trade.closed_at = closed_at or utcnow()
    trade.gross_pnl = gross
    trade.net_pnl = net
    trade.r_multiple = r_multiple
    trade.status = TradeStatus.CLOSED
    await session.flush()
    return trade


async def list_open_trades(session: AsyncSession, account_id: int | None = None) -> list[Trade]:
    stmt = select(Trade).where(Trade.status == TradeStatus.OPEN).order_by(Trade.opened_at.desc())
    if account_id is not None:
        stmt = stmt.where(Trade.account_id == account_id)
    return list((await session.execute(stmt)).scalars())


# --------------------------------------------------------------------------------------
# Equity snapshots
# --------------------------------------------------------------------------------------


async def save_equity_snapshot(
    session: AsyncSession,
    account_id: int,
    snapshot: AccountSnapshot,
    *,
    daily_pnl: Decimal | None = None,
    daily_drawdown_pct: Decimal | None = None,
) -> EquitySnapshot:
    """Insert (or overwrite the same ``(ts, account_id)``) equity time-series point."""
    row = EquitySnapshot(
        ts=snapshot.timestamp,
        account_id=account_id,
        equity=snapshot.equity,
        wallet_balance=snapshot.wallet_balance,
        unrealized_pnl=snapshot.unrealized_pnl,
        margin_ratio=snapshot.margin_ratio,
        daily_pnl=daily_pnl,
        daily_drawdown_pct=daily_drawdown_pct,
    )
    merged = await session.merge(row)
    await session.flush()
    return merged


# --------------------------------------------------------------------------------------
# Kill switch (singleton row id = 1)
# --------------------------------------------------------------------------------------


async def get_kill_switch_state(session: AsyncSession) -> KillSwitchState:
    """Return the singleton row, creating the default (disengaged) one if missing."""
    row = (await session.execute(select(KillSwitchState).where(KillSwitchState.id == 1))).scalar_one_or_none()
    if row is None:
        row = KillSwitchState(id=1, engaged=False)
        session.add(row)
        await session.flush()
    return row


async def set_kill_switch_state(
    session: AsyncSession,
    *,
    engaged: bool,
    reason: str | None = None,
    trading_day: date | None = None,
    day_start_equity: Decimal | None = None,
) -> KillSwitchState:
    """Engage / release the kill switch.

    ``engaged_at`` is stamped on the False to True transition and cleared on release.
    ``trading_day`` / ``day_start_equity`` are only overwritten when provided.
    """
    row = (
        await session.execute(select(KillSwitchState).where(KillSwitchState.id == 1).with_for_update())
    ).scalar_one_or_none()
    if row is None:
        row = KillSwitchState(id=1, engaged=False)
        session.add(row)

    if engaged and not row.engaged:
        row.engaged_at = utcnow()
    elif not engaged:
        row.engaged_at = None
    row.engaged = engaged
    row.reason = reason
    if trading_day is not None:
        row.trading_day = trading_day
    if day_start_equity is not None:
        row.day_start_equity = day_start_equity
    row.updated_at = utcnow()
    await session.flush()
    return row
