"""Structural interfaces the execution engine depends on (implemented elsewhere)."""

from __future__ import annotations

from typing import Protocol

from app.execution.contracts import ManagedPosition


class PositionManagerLike(Protocol):
    """The slice of the PositionManager the executor needs: hand over a protected position, ask if one is managed."""

    async def register(self, position: ManagedPosition) -> None: ...

    def is_managed(self, symbol: str) -> bool: ...
