"""The single forced tool through which Claude must answer, and its strict parser."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.claude.contracts import ClaudeDecision

TOOL_NAME = "submit_trading_decision"

_NUMBER_FIELDS = ("confidence", "entry_price", "stop_loss", "take_profit")
_ALLOWED_KEYS = {
    "action", "confidence", "leverage", "entry_type", "entry_price", "stop_loss", "take_profit",
    "reasoning_tr", "risk_flags",
}


def decision_tool_definition() -> dict[str, Any]:
    nullable_number = {"type": ["number", "null"]}
    return {
        "name": TOOL_NAME,
        "description": (
            "Submit the final trading decision for the proposed signal. Must be called exactly once. "
            "You may only confirm the proposed direction, HOLD, or CLOSE. The hard-risk engine clamps "
            "leverage (max 5x) and may reject the trade."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["LONG", "SHORT", "HOLD", "CLOSE"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "leverage": {"type": "integer", "minimum": 1, "maximum": 20,
                             "description": "Requested leverage; the risk engine clamps it to at most 5."},
                "entry_type": {"type": "string", "enum": ["MARKET", "LIMIT"]},
                "entry_price": {**nullable_number, "description": "Required for LIMIT, null otherwise."},
                "stop_loss": {**nullable_number, "description": "Required for LONG/SHORT, null for HOLD/CLOSE."},
                "take_profit": {**nullable_number, "description": "Required for LONG/SHORT, null for HOLD/CLOSE."},
                "reasoning_tr": {"type": "string", "minLength": 20, "maxLength": 4000,
                                 "description": "Detailed Turkish explanation."},
                "risk_flags": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            },
            "required": ["action", "confidence", "leverage", "entry_type", "reasoning_tr", "risk_flags"],
        },
    }


def parse_tool_input(data: dict[str, Any]) -> ClaudeDecision:
    """Convert the tool input into a validated ClaudeDecision.

    Numbers go through ``Decimal(str(x))`` (never float arithmetic). Raises ValueError (pydantic's
    ValidationError is a ValueError subclass) on any violation, including unknown keys.
    """
    if not isinstance(data, dict):
        raise ValueError("tool input is not an object")
    unknown = set(data) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown keys in tool input: {sorted(unknown)}")
    converted: dict[str, Any] = dict(data)
    for key in _NUMBER_FIELDS:
        value = converted.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
            raise ValueError(f"{key} must be a number")
        try:
            dec = Decimal(str(value))
        except Exception as exc:  # InvalidOperation
            raise ValueError(f"{key} is not a valid number") from exc
        if not dec.is_finite():
            raise ValueError(f"{key} must be finite")
        converted[key] = dec
    lev = converted.get("leverage")
    if isinstance(lev, bool) or (lev is not None and not isinstance(lev, int)):
        if isinstance(lev, float) and lev.is_integer():
            converted["leverage"] = int(lev)
        else:
            raise ValueError("leverage must be an integer")
    return ClaudeDecision.model_validate(converted)
