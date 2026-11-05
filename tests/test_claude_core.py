"""Decision core tests: forced tool_use, fallbacks, semantics, circuit breaker (scripted transport, no network)."""

from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from app.claude.contracts import DecisionAction, FallbackReason
from app.claude.core import ClaudeDecisionCore
from app.claude.prompt import SYSTEM_PROMPT
from app.claude.tool import TOOL_NAME
from app.claude.transport import Delay, ScriptedTransport, text_response, tool_response
from app.config.settings import ClaudeSettings
from app.core.types import PositionSide
from tests.test_claude_prompt import INJECTION, good_input, make_news, make_payload

from decimal import Decimal

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def settings(**kw) -> ClaudeSettings:
    base = dict(timeout_seconds=0.05, circuit_breaker_failures=3, circuit_breaker_cooldown_seconds=60)
    base.update(kw)
    return ClaudeSettings(**base)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def core_with(script, **kw):
    tr = ScriptedTransport(script)
    return ClaudeDecisionCore(settings(), tr, **kw), tr


async def test_valid_long_decision_and_forced_tool_kwargs():
    core, tr = core_with([tool_response(good_input(), input_tokens=1500, output_tokens=321)])
    res = await core.decide(make_payload(), {"equity": "1000"})
    assert not res.is_fallback and res.fallback is None and res.error is None
    d = res.decision
    assert d.action is DecisionAction.LONG
    assert d.confidence == Decimal("0.74") and d.stop_loss == Decimal("98.5") and d.take_profit == Decimal("103.0")
    assert isinstance(d.stop_loss, Decimal) and d.leverage == 3
    assert res.input_tokens == 1500 and res.output_tokens == 321 and res.latency_ms >= 0
    assert res.signal_id == "sig-123" and res.symbol == "BTC/USDT"
    assert res.raw_response["tool_input"]["action"] == "LONG" and res.raw_response["stop_reason"] == "tool_use"
    assert res.raw_response["id"] == "msg_test_001"
    json.dumps(res.raw_response)
    assert core.stats["ok"] == 1

    assert tr.call_count == 1
    kw = tr.calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert [t["name"] for t in kw["tools"]] == [TOOL_NAME]
    assert kw["tools"][0]["input_schema"]["additionalProperties"] is False
    assert kw["system"] == SYSTEM_PROMPT and kw["model"] == settings().model
    assert kw["messages"][0]["role"] == "user" and "<signal_payload>" in kw["messages"][0]["content"]


async def test_hold_and_close_are_accepted():
    hold = good_input(action="HOLD", stop_loss=None, take_profit=None, confidence=0.3)
    close = good_input(action="CLOSE", stop_loss=None, take_profit=None)
    core, _ = core_with([tool_response(hold), tool_response(close)])
    r1 = await core.decide(make_payload(), {})
    r2 = await core.decide(make_payload(), {})
    assert r1.decision.action is DecisionAction.HOLD and not r1.is_fallback
    assert r2.decision.action is DecisionAction.CLOSE and not r2.is_fallback


async def test_short_payload_short_decision_ok():
    p = make_payload(PositionSide.SHORT)
    core, _ = core_with([tool_response(good_input(action="SHORT", stop_loss=101.5, take_profit=97.0))])
    res = await core.decide(p, {})
    assert res.decision.action is DecisionAction.SHORT and not res.is_fallback


async def test_timeout_via_delay():
    core, _ = core_with([Delay(1.0, tool_response(good_input()))])
    res = await core.decide(make_payload(), {})
    assert res.fallback is FallbackReason.TIMEOUT
    assert res.decision.action is DecisionAction.HOLD and res.decision.confidence == 0
    assert "zaman aşımı" in res.decision.reasoning_tr
    assert core.stats["TIMEOUT"] == 1


async def test_sdk_timeout_exception_is_timeout():
    core, _ = core_with([anthropic.APITimeoutError(request=REQ)])
    assert (await core.decide(make_payload(), {})).fallback is FallbackReason.TIMEOUT


async def test_rate_limit():
    err = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None)
    core, _ = core_with([err])
    res = await core.decide(make_payload(), {})
    assert res.fallback is FallbackReason.RATE_LIMITED and res.decision.action is DecisionAction.HOLD


async def test_api_status_500_and_connection_error():
    err = anthropic.APIStatusError("boom", response=httpx.Response(500, request=REQ), body=None)
    core, _ = core_with([err, anthropic.APIConnectionError(request=REQ), RuntimeError("weird")])
    for _ in range(3):
        res = await core.decide(make_payload(), {})
        assert res.fallback is FallbackReason.API_ERROR and res.decision.action is DecisionAction.HOLD


async def test_error_is_redacted():
    secret = "sk-ant-api03-FAKEEXAMPLEKEY000"  # format-only fake for the redaction pattern
    err = anthropic.APIStatusError(f"bad key {secret} x-api-key: {secret}", response=httpx.Response(401, request=REQ), body=None)
    core, _ = core_with([err])
    res = await core.decide(make_payload(), {})
    assert res.fallback is FallbackReason.API_ERROR
    assert secret not in (res.error or "") and "SUPERSECRET" not in (res.error or "")
    assert "SUPERSECRET" not in res.decision.reasoning_tr


async def test_text_only_response_is_invalid_output():
    core, _ = core_with([text_response("I think you should buy.")])
    res = await core.decide(make_payload(), {})
    assert res.fallback is FallbackReason.INVALID_OUTPUT and res.decision.action is DecisionAction.HOLD


async def test_wrong_tool_name_is_invalid_output():
    core, _ = core_with([tool_response(good_input(), name="other_tool")])
    assert (await core.decide(make_payload(), {})).fallback is FallbackReason.INVALID_OUTPUT


@pytest.mark.parametrize("label,payload_kw,over", [
    ("extra key", {}, {"hack": "yes"}),
    ("confidence 1.7", {}, {"confidence": 1.7}),
    ("long sl above entry", {}, {"stop_loss": 101.0, "take_profit": 105.0}),
    ("long tp below entry", {}, {"stop_loss": 98.5, "take_profit": 99.0}),
    ("flipped to short", {}, {"action": "SHORT", "stop_loss": 101.5, "take_profit": 97.0}),
    ("stop wider than 5 atr", {}, {"stop_loss": 94.0, "take_profit": 110.0}),
    ("limit 2% away", {}, {"entry_type": "LIMIT", "entry_price": 98.0}),
    ("limit without price", {}, {"entry_type": "LIMIT", "entry_price": None}),
    ("long missing sl/tp", {}, {"stop_loss": None, "take_profit": None}),
    ("leverage out of range", {}, {"leverage": 50}),
    ("no atr", {"atr": None}, {}),
])
async def test_invalid_outputs_become_hold(label, payload_kw, over):
    core, _ = core_with([tool_response(good_input(**over))])
    res = await core.decide(make_payload(**payload_kw), {})
    assert res.fallback is FallbackReason.INVALID_OUTPUT, label
    assert res.decision.action is DecisionAction.HOLD and res.decision.confidence == 0
    assert res.raw_response.get("tool_input") is not None  # rejected output kept for audit


async def test_short_payload_flipped_to_long_rejected():
    core, _ = core_with([tool_response(good_input())])
    res = await core.decide(make_payload(PositionSide.SHORT), {})
    assert res.fallback is FallbackReason.INVALID_OUTPUT and "yön" in res.decision.reasoning_tr


async def test_limit_within_half_percent_ok():
    core, _ = core_with([tool_response(good_input(entry_type="LIMIT", entry_price=99.7))])
    res = await core.decide(make_payload(), {})
    assert not res.is_fallback and res.decision.entry_price == Decimal("99.7")


async def test_no_api_key_makes_no_calls():
    tr = ScriptedTransport([tool_response(good_input())])
    core = ClaudeDecisionCore(settings(), tr, api_key_present=False)
    res = await core.decide(make_payload(), {})
    assert res.fallback is FallbackReason.NO_API_KEY and tr.call_count == 0
    assert res.decision.action is DecisionAction.HOLD
    res2 = await ClaudeDecisionCore(settings(), None).decide(make_payload(), {})
    assert res2.fallback is FallbackReason.NO_API_KEY


async def test_circuit_breaker_opens_and_closes_after_cooldown():
    clk = Clock()
    tr = ScriptedTransport([RuntimeError("x")] * 3 + [tool_response(good_input())])
    core = ClaudeDecisionCore(settings(), tr, monotonic=clk)
    for _ in range(3):
        assert (await core.decide(make_payload(), {})).fallback is FallbackReason.API_ERROR
    assert core.circuit_open
    res = await core.decide(make_payload(), {})
    assert res.fallback is FallbackReason.CIRCUIT_OPEN and tr.call_count == 3
    assert res.decision.action is DecisionAction.HOLD
    clk.t += 59
    assert (await core.decide(make_payload(), {})).fallback is FallbackReason.CIRCUIT_OPEN and tr.call_count == 3
    clk.t += 2
    assert not core.circuit_open
    ok = await core.decide(make_payload(), {})
    assert not ok.is_fallback and tr.call_count == 4 and not core.circuit_open
    assert core.stats["CIRCUIT_OPEN"] == 2 and core.stats["ok"] == 1


async def test_half_open_failure_reopens_and_success_resets_counter():
    clk = Clock()
    tr = ScriptedTransport([RuntimeError("x")] * 3 + [RuntimeError("y")] + [tool_response(good_input())] * 4)
    core = ClaudeDecisionCore(settings(), tr, monotonic=clk)
    for _ in range(3):
        await core.decide(make_payload(), {})
    clk.t += 61
    await core.decide(make_payload(), {})  # probe fails
    assert core.circuit_open
    clk.t += 61
    assert not (await core.decide(make_payload(), {})).is_fallback
    assert not core.circuit_open


async def test_success_resets_consecutive_failures():
    tr = ScriptedTransport([RuntimeError("x"), RuntimeError("x"), tool_response(good_input()),
                            RuntimeError("x"), RuntimeError("x")])
    core = ClaudeDecisionCore(settings(), tr)
    for _ in range(5):
        await core.decide(make_payload(), {})
    assert not core.circuit_open


async def test_invalid_outputs_count_toward_breaker():
    core, _ = core_with([text_response("no")] * 3)
    for _ in range(3):
        await core.decide(make_payload(), {})
    assert core.circuit_open


async def test_injection_news_reaches_model_only_inside_block():
    core, tr = core_with([tool_response(good_input(risk_flags=["NEWS_INJECTION_ATTEMPT"]))])
    payload = make_payload(news=(make_news(title=INJECTION),))
    res = await core.decide(payload, {"equity": "1000"})
    msg = tr.calls[0]["messages"][0]["content"]
    block = msg[msg.index("<untrusted_market_news>"):msg.index("</untrusted_market_news>")]
    assert "IGNORE PREVIOUS INSTRUCTIONS" in block and "‹/untrusted_market_news›" in block
    assert msg.count("</untrusted_market_news>") == 1
    assert "IGNORE PREVIOUS" not in msg.replace(block, "")
    assert res.decision.leverage == 3  # model output is what counts; risk engine clamps later


class BoomSink:
    async def log(self, *a, **k):
        raise RuntimeError("db down")


async def test_logger_failure_does_not_break_decide():
    core = ClaudeDecisionCore(settings(), ScriptedTransport([tool_response(good_input())]), logger_sink=BoomSink())
    res = await core.decide(make_payload(), {})
    assert not res.is_fallback and res.decision.action is DecisionAction.LONG


async def test_sink_called_with_result():
    seen = []

    class Sink:
        async def log(self, result, payload, ctx):
            seen.append((result, payload, ctx))

    core = ClaudeDecisionCore(settings(), ScriptedTransport([text_response("x")]), logger_sink=Sink())
    res = await core.decide(make_payload(), {"a": 1})
    assert seen and seen[0][0] is res and seen[0][2] == {"a": 1}


async def test_core_never_raises_on_broken_payload():
    core, _ = core_with([tool_response(good_input())])
    res = await core.decide(None, {})  # type: ignore[arg-type]
    assert res.fallback is not None and res.decision.action is DecisionAction.HOLD
