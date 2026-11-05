"""In-memory sink and store for tests and paper mode."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from app.core.types import utcnow
from app.risk.protocols import KillSwitchState


@dataclass(slots=True)
class RiskEvent:
    level: str
    code: str
    message: str
    details: dict[str, Any]
    timestamp: datetime = field(default_factory=utcnow)


class InMemoryRiskEventSink:
    def __init__(self) -> None:
        self.events: list[RiskEvent] = []

    async def record(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        self.events.append(RiskEvent(level, code, message, dict(details)))

    def by_level(self, level: str) -> list[RiskEvent]:
        return [e for e in self.events if e.level == level]

    def by_code(self, code: str) -> list[RiskEvent]:
        return [e for e in self.events if e.code == code]


class InMemoryKillSwitchStore:
    def __init__(self, state: KillSwitchState | None = None) -> None:
        self._state = replace(state) if state else None
        self.save_count = 0

    async def load(self) -> KillSwitchState | None:
        return replace(self._state) if self._state else None

    async def save(self, state: KillSwitchState) -> None:
        self._state = replace(state)
        self.save_count += 1
