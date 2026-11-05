"""Persistence of Claude decisions (claude_decisions table) for audit and the Phase-4 terminal."""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.claude.contracts import DecisionResult
from app.claude.prompt import sanitized_news_items, signal_json_without_news
from app.db import repository as repo
from app.db.models import ClaudeDecision as ClaudeDecisionRow
from app.db.models import DecisionAction as OrmDecisionAction
from app.db.models import RiskVerdict, jsonable
from app.db.session import get_session
from app.quant.contracts import SignalPayload

logger = logging.getLogger(__name__)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class DecisionLogger:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], *, max_news_chars: int = 600) -> None:
        self._factory = session_factory
        self._max_news_chars = max_news_chars

    def build_input_context(self, result: DecisionResult, payload: SignalPayload, portfolio_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "signal_id": payload.signal_id,  # signals row may not exist -> kept here, not as FK
            "payload": signal_json_without_news(payload),
            "portfolio_context": portfolio_context or {},
            "news": sanitized_news_items(payload.recent_news, self._max_news_chars),
        }

    async def log(self, result: DecisionResult, payload: SignalPayload, portfolio_context: dict[str, Any]) -> bool:
        try:
            d = result.decision
            ctx = self.build_input_context(result, payload, portfolio_context)
            async with get_session(self._factory) as session:
                row = await repo.save_decision(
                    session,
                    decision_id=_as_uuid(result.decision_id),
                    signal_id=None,
                    symbol=result.symbol,
                    model=result.model,
                    action=OrmDecisionAction(d.action.value),
                    confidence=d.confidence,
                    leverage=d.leverage,
                    stop_loss=d.stop_loss,
                    take_profit=d.take_profit,
                    reasoning=d.reasoning_tr,
                    input_context=ctx,
                    raw_response=jsonable({
                        **result.raw_response,
                        "risk_flags": d.risk_flags,
                        "fallback": result.fallback.value if result.fallback else None,
                        "error": result.error,
                    }),
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    created_at=result.created_at,
                )
                row.fallback_reason = result.fallback.value if result.fallback else None
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to log claude decision %s: %s", getattr(result, "decision_id", "?"), type(exc).__name__)
            return False

    async def record_risk_verdict(
        self,
        decision_id: str | uuid.UUID,
        verdict: str,
        reasons: list[str],
        final_leverage: int | None,
        final_quantity: Decimal | None,
    ) -> bool:
        try:
            parsed = RiskVerdict(str(verdict).upper())
            async with get_session(self._factory) as session:
                await repo.update_decision_risk_verdict(
                    session, _as_uuid(decision_id), parsed, list(reasons), final_leverage, final_quantity
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to record risk verdict for %s: %s", decision_id, type(exc).__name__)
            return False

    async def record_execution(self, decision_id: str | uuid.UUID, status: str, detail: str) -> bool:
        try:
            async with get_session(self._factory) as session:
                row = await session.get(ClaudeDecisionRow, _as_uuid(decision_id))
                if row is None:
                    logger.error("record_execution: decision %s not found", decision_id)
                    return False
                row.execution_status = str(status)[:32]
                row.execution_detail = detail
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to record execution for %s: %s", decision_id, type(exc).__name__)
            return False

    async def recent(self, limit: int = 50, symbol: str | None = None) -> list[dict[str, Any]]:
        try:
            async with get_session(self._factory) as session:
                rows = await repo.list_recent_decisions(session, limit, symbol)
                out: list[dict[str, Any]] = []
                for r in rows:
                    ctx = r.input_context or {}
                    out.append({
                        "id": str(r.id),
                        "created_at": r.created_at.isoformat(),
                        "signal_id": ctx.get("signal_id") if isinstance(ctx, dict) else None,
                        "symbol": r.symbol,
                        "model": r.model,
                        "action": r.action.value,
                        "confidence": str(r.confidence),
                        "leverage": r.leverage,
                        "stop_loss": str(r.stop_loss) if r.stop_loss is not None else None,
                        "take_profit": str(r.take_profit) if r.take_profit is not None else None,
                        "reasoning": r.reasoning,
                        "risk_verdict": r.risk_verdict.value if r.risk_verdict else None,
                        "risk_reasons": r.risk_reasons,
                        "final_leverage": r.final_leverage,
                        "final_quantity": str(r.final_quantity) if r.final_quantity is not None else None,
                        "fallback_reason": r.fallback_reason,
                        "execution_status": r.execution_status,
                        "execution_detail": r.execution_detail,
                        "latency_ms": r.latency_ms,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "input_context": r.input_context,
                    })
                return out
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to read recent decisions: %s", type(exc).__name__)
            return []
