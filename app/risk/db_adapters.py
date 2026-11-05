"""Database-backed implementations of the risk-layer protocols.

The risk engine depends only on the protocols in ``app.risk.protocols``; these classes bind
them to PostgreSQL/TimescaleDB (or SQLite in tests) through the repository layer.
A failing database must never block a kill-switch flatten, so the sink logs and swallows
write errors, while the store re-raises (a kill switch that cannot persist is reported loudly).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import repository
from app.db.models import RiskLevel
from app.risk.protocols import KillSwitchState

logger = logging.getLogger(__name__)


class DbRiskEventSink:
    """Writes every risk rejection / reduction / kill-switch event to ``risk_events``."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def record(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        log_level = {"INFO": logging.INFO, "WARN": logging.WARNING, "CRITICAL": logging.CRITICAL}.get(
            level, logging.WARNING
        )
        logger.log(log_level, "risk_event code=%s %s", code, message)
        try:
            async with self._factory() as session:
                await repository.record_risk_event(
                    session, level=RiskLevel(level), code=code, message=message, details=details
                )
                await session.commit()
        except Exception:  # noqa: BLE001 - persistence must not break risk enforcement
            logger.exception("Failed to persist risk event %s", code)


class DbKillSwitchStore:
    """Persists the kill-switch singleton row so an engaged switch survives restarts."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def load(self) -> KillSwitchState | None:
        async with self._factory() as session:
            row = await repository.get_kill_switch_state(session)
            await session.commit()
            return KillSwitchState(
                engaged=row.engaged,
                reason=row.reason,
                engaged_at=row.engaged_at,
                engaged_day=row.engaged_day,
                trading_day=row.trading_day,
                day_start_equity=row.day_start_equity,
                peak_equity=row.peak_equity,
                consecutive_losses=row.consecutive_losses or 0,
            )

    async def save(self, state: KillSwitchState) -> None:
        async with self._factory() as session:
            row = await repository.get_kill_switch_state(session)
            row.engaged = state.engaged
            row.reason = state.reason
            row.engaged_at = state.engaged_at
            row.engaged_day = state.engaged_day
            row.trading_day = state.trading_day
            row.day_start_equity = state.day_start_equity
            row.peak_equity = state.peak_equity
            row.consecutive_losses = state.consecutive_losses
            await session.commit()
