"""LLM transport abstraction: real Anthropic client + a scripted fake for tests (no network)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMTransport(Protocol):
    async def create_message(self, **kwargs: Any) -> Any:
        """Mirror of ``AsyncAnthropic().messages.create`` returning a Message-like object."""
        ...


class AnthropicTransport:
    """Real transport. SDK timeout set and retries disabled: retry policy belongs to the core."""

    def __init__(self, api_key: str, timeout: float) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)

    async def create_message(self, **kwargs: Any) -> Any:
        return await self._client.messages.create(**kwargs)


@dataclass(frozen=True)
class Delay:
    """Scripted entry: sleep ``seconds`` (async) and then return ``response`` (or an empty text reply)."""

    seconds: float
    response: Any = None


def tool_response(
    input_dict: dict[str, Any],
    input_tokens: int = 1200,
    output_tokens: int = 300,
    *,
    name: str = "submit_trading_decision",
    stop_reason: str = "tool_use",
    message_id: str = "msg_test_001",
) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", id="toolu_test_001", name=name, input=input_dict)
    return SimpleNamespace(
        id=message_id, type="message", role="assistant", content=[block], stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def text_response(text: str, input_tokens: int = 800, output_tokens: int = 40) -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        id="msg_test_text", type="message", role="assistant", content=[block], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class ScriptedTransport:
    """Pops one scripted entry per call: a response object, an Exception (raised) or a ``Delay``."""

    def __init__(self, script: Sequence[Any] = (), *, repeat_last: bool = False) -> None:
        self._queue: list[Any] = list(script)
        self._repeat_last = repeat_last
        self._last: Any = None
        self.calls: list[dict[str, Any]] = []

    def push(self, *entries: Any) -> None:
        self._queue.extend(entries)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def create_message(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._queue:
            entry = self._queue.pop(0)
            self._last = entry
        elif self._repeat_last and self._last is not None:
            entry = self._last
        else:
            raise RuntimeError("ScriptedTransport: script exhausted")
        if isinstance(entry, Delay):
            await asyncio.sleep(entry.seconds)
            return entry.response if entry.response is not None else text_response("late")
        if isinstance(entry, BaseException):
            raise entry
        return entry
