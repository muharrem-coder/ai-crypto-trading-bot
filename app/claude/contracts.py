"""Claude decision contract: the ONLY shape the LLM may return (enforced via forced tool_use).

The LLM proposes; the Hard-Risk Engine disposes. Nothing here can raise a risk limit:
`leverage` is only a request that the risk engine clamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DecisionAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class EntryType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ClaudeDecision(BaseModel):
    """Strict schema. extra='forbid' -> unknown keys are a validation error -> HOLD fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: DecisionAction
    confidence: Decimal = Field(ge=0, le=1)
    leverage: int = Field(ge=1, le=20)
    entry_type: EntryType = EntryType.MARKET
    entry_price: Decimal | None = Field(default=None, gt=0)  # required for LIMIT
    stop_loss: Decimal | None = Field(default=None, gt=0)
    take_profit: Decimal | None = Field(default=None, gt=0)
    reasoning_tr: str = Field(min_length=20, max_length=4000)
    risk_flags: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _structural(self) -> "ClaudeDecision":
        if self.action in (DecisionAction.LONG, DecisionAction.SHORT):
            if self.stop_loss is None or self.take_profit is None:
                raise ValueError("LONG/SHORT requires stop_loss and take_profit")
            if self.action is DecisionAction.LONG and not self.stop_loss < self.take_profit:
                raise ValueError("LONG requires stop_loss < take_profit")
            if self.action is DecisionAction.SHORT and not self.take_profit < self.stop_loss:
                raise ValueError("SHORT requires take_profit < stop_loss")
            if self.entry_type is EntryType.LIMIT and self.entry_price is None:
                raise ValueError("LIMIT entry requires entry_price")
        return self

    @classmethod
    def hold(cls, reason_tr: str, flags: list[str] | None = None) -> "ClaudeDecision":
        text = reason_tr if len(reason_tr) >= 20 else f"{reason_tr} - sistem güvenli moda geçti (HOLD)."
        return cls(action=DecisionAction.HOLD, confidence=Decimal(0), leverage=1, entry_type=EntryType.MARKET,
                   reasoning_tr=text[:4000], risk_flags=list(flags or []))


class FallbackReason(str, Enum):
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"  # no tool call / schema violation / semantic violation
    CIRCUIT_OPEN = "CIRCUIT_OPEN"  # too many consecutive failures, calls suspended
    NO_API_KEY = "NO_API_KEY"


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """What the decision core returns. Never raises to the caller."""

    decision_id: str  # UUID, also the claude_decisions primary key
    signal_id: str
    symbol: str
    decision: ClaudeDecision
    model: str
    created_at: datetime
    latency_ms: int
    fallback: FallbackReason | None = None  # set => decision is the automatic HOLD
    error: str | None = None  # redacted
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return self.fallback is not None
