"""DB-backed TradeRecorder. Every method swallows (and counts) database errors: persistence must never
break the trading path."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from decimal import Decimal
from typing import Any, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.types import AssetClass, MarginMode, MarketType, Order, TradingMode
from app.db import repository
from app.db.models import ClaudeDecision, ExitReason, OrderPurpose, RiskLevel
from app.db.session import get_session
from app.execution.contracts import ExecutionReport, ManagedPosition

log = logging.getLogger(__name__)

T = TypeVar("T")

_PURPOSE_ALIASES: dict[str, OrderPurpose] = {
    "ENTRY": OrderPurpose.ENTRY,
    "STOP_LOSS": OrderPurpose.STOP_LOSS,
    "STOP": OrderPurpose.STOP_LOSS,
    "SL": OrderPurpose.STOP_LOSS,
    "TAKE_PROFIT": OrderPurpose.TAKE_PROFIT,
    "TP": OrderPurpose.TAKE_PROFIT,
    "TRAILING_STOP": OrderPurpose.TRAILING_STOP,
    "TRAILING": OrderPurpose.TRAILING_STOP,
    "EXIT": OrderPurpose.EXIT,
    "CLOSE": OrderPurpose.EXIT,
    "EMERGENCY_CLOSE": OrderPurpose.EXIT,
    "KILL_SWITCH": OrderPurpose.KILL_SWITCH,
}

_EXIT_ALIASES: dict[str, ExitReason] = {
    "TAKE_PROFIT": ExitReason.TAKE_PROFIT,
    "TP": ExitReason.TAKE_PROFIT,
    "STOP_LOSS": ExitReason.STOP_LOSS,
    "SL": ExitReason.STOP_LOSS,
    "TRAILING_STOP": ExitReason.TRAILING_STOP,
    "MANUAL": ExitReason.MANUAL,
    "KILL_SWITCH": ExitReason.KILL_SWITCH,
    "SIGNAL_EXIT": ExitReason.SIGNAL_EXIT,
    "CLOSED_BY_DECISION": ExitReason.SIGNAL_EXIT,
    "LIQUIDATION": ExitReason.LIQUIDATION,
}


def map_purpose(purpose: str) -> OrderPurpose:
    key = str(getattr(purpose, "value", purpose)).strip().upper()
    mapped = _PURPOSE_ALIASES.get(key)
    if mapped is None:
        log.warning("unknown order purpose %r, stored as EXIT", purpose)
        return OrderPurpose.EXIT
    return mapped


def map_exit_reason(reason: str) -> ExitReason:
    key = str(getattr(reason, "value", reason)).strip().upper()
    mapped = _EXIT_ALIASES.get(key)
    if mapped is None:
        log.warning("unknown exit reason %r, stored as MANUAL", reason)
        return ExitReason.MANUAL
    return mapped


def _parse_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


class DbTradeRecorder:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        account: Mapping[str, Any],
    ) -> None:
        self._factory = session_factory
        self._account_spec = {
            "broker": str(account["broker"]),
            "asset_class": AssetClass(account.get("asset_class", AssetClass.CRYPTO)),
            "market_type": MarketType(account.get("market_type", MarketType.PERPETUAL)),
            "trading_mode": TradingMode(account.get("trading_mode", TradingMode.PAPER)),
            "quote_currency": str(account.get("quote_currency", "USDT")),
        }
        self._account_id: int | None = None
        self.errors = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ plumbing
    async def _run(self, name: str, work: Callable[[AsyncSession], Awaitable[T]], default: T) -> T:
        try:
            async with get_session(self._factory) as session:
                return await work(session)
        except Exception as exc:  # noqa: BLE001 - swallow by contract
            self.errors += 1
            self.last_error = f"{name}: {type(exc).__name__}: {exc}"
            log.error("DbTradeRecorder.%s failed: %s", name, exc, exc_info=True)
            return default

    async def _get_account_id(self, session: AsyncSession) -> int:
        if self._account_id is None:
            account = await repository.get_or_create_account(session, **self._account_spec)
            self._account_id = account.id
        return self._account_id

    @staticmethod
    async def _existing_decision_id(session: AsyncSession, decision_id: str | None) -> uuid.UUID | None:
        parsed = _parse_uuid(decision_id)
        if parsed is None:
            return None
        return parsed if await session.get(ClaudeDecision, parsed) is not None else None

    # ------------------------------------------------------------------ TradeRecorder
    async def record_order(self, order: Order, purpose: str, decision_id: str | None) -> None:
        async def work(session: AsyncSession) -> None:
            account_id = await self._get_account_id(session)
            did = await self._existing_decision_id(session, decision_id)
            await repository.upsert_order_from_domain(session, order, map_purpose(purpose), did, account_id)

        await self._run("record_order", work, None)

    async def open_trade(self, position: ManagedPosition, leverage: int) -> str | None:
        async def work(session: AsyncSession) -> str:
            account_id = await self._get_account_id(session)
            did = await self._existing_decision_id(session, position.decision_id)
            entry_fee = position.entry_price * position.quantity * position.fees.taker_rate
            trade = await repository.open_trade(
                session,
                account_id=account_id,
                symbol=position.symbol,
                side=position.side,
                leverage=leverage,
                margin_mode=MarginMode.ISOLATED,
                entry_price=position.entry_price,
                quantity=position.quantity,
                initial_stop=position.initial_stop,
                decision_id=did,
                fees_paid=entry_fee,
                opened_at=position.opened_at,
            )
            return str(trade.id)

        return await self._run("open_trade", work, None)

    async def close_trade(self, trade_id: str | None, exit_price: Decimal, exit_reason: str,
                          fees_paid: Decimal | None) -> Decimal | None:
        tid = _parse_uuid(trade_id)
        if tid is None:
            return None

        async def work(session: AsyncSession) -> Decimal | None:
            trade = await repository.close_trade(
                session, tid, exit_price=exit_price, exit_reason=map_exit_reason(exit_reason), fees_paid=fees_paid
            )
            return trade.net_pnl

        return await self._run("close_trade", work, None)

    async def record_execution(self, report: ExecutionReport) -> None:
        did = _parse_uuid(report.decision_id)
        if did is None:
            return

        async def work(session: AsyncSession) -> None:
            decision = await session.get(ClaudeDecision, did)
            if decision is None:
                return
            decision.execution_status = report.status.value[:32]
            decision.execution_detail = report.detail
            if report.approved_leverage is not None and 1 <= report.approved_leverage <= 20:
                decision.final_leverage = report.approved_leverage
            if report.approved_quantity is not None:
                decision.final_quantity = report.approved_quantity
            await session.flush()

        await self._run("record_execution", work, None)

    async def risk_event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        try:
            lvl = RiskLevel(str(level).upper())
        except ValueError:
            lvl = RiskLevel.WARN

        async def work(session: AsyncSession) -> None:
            await repository.record_risk_event(session, level=lvl, code=code[:64], message=message, details=details)

        await self._run("risk_event", work, None)
