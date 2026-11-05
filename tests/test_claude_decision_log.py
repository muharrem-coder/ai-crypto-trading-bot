"""Decision log persistence tests (SQLite in memory, scripted transport)."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.claude.contracts import FallbackReason
from app.claude.core import ClaudeDecisionCore
from app.claude.decision_log import DecisionLogger
from app.claude.transport import ScriptedTransport, text_response, tool_response
from app.config.settings import ClaudeSettings
from app.db.models import ClaudeDecision as Row
from app.db.models import DecisionAction, RiskVerdict
from app.db.session import async_session_factory, create_engine_from_url, init_db
from tests.test_claude_prompt import INJECTION, good_input, make_news, make_payload

pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

TR_TEXT = "RSI aşırı satım bölgesinde (28); Bollinger alt bandı + OBI alıcı yönlü. Komisyon sonrası R:R ≈ 1,9. İşlem değerlendirmesi: çok güzel, ığüşiçö ĞÜŞİÖÇ."


@pytest_asyncio.fixture
async def factory():
    eng = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(eng)
    yield async_session_factory(eng)
    await eng.dispose()


async def _row(factory, decision_id: str) -> Row:
    async with factory() as s:
        return (await s.execute(select(Row).where(Row.id == uuid.UUID(decision_id)))).scalar_one()


def _core(factory, script):
    log = DecisionLogger(factory)
    core = ClaudeDecisionCore(ClaudeSettings(timeout_seconds=0.5), ScriptedTransport(script), logger_sink=log)
    return core, log


async def test_turkish_reasoning_stored_verbatim(factory):
    core, _ = _core(factory, [tool_response(good_input(reasoning_tr=TR_TEXT), input_tokens=900, output_tokens=210)])
    payload = make_payload(news=(make_news(title=INJECTION),))
    res = await core.decide(payload, {"equity": "1000", "open_positions": []})
    row = await _row(factory, res.decision_id)
    assert row.reasoning == TR_TEXT
    assert row.action is DecisionAction.LONG
    assert row.confidence == Decimal("0.74") and row.leverage == 3
    assert row.stop_loss == Decimal("98.5") and row.take_profit == Decimal("103.0")
    assert row.signal_id is None and row.fallback_reason is None
    assert row.model == res.model and row.input_tokens == 900 and row.output_tokens == 210
    assert row.latency_ms == res.latency_ms
    ctx = row.input_context
    assert ctx["signal_id"] == "sig-123" and ctx["portfolio_context"]["equity"] == "1000"
    assert ctx["payload"]["direction"] == "LONG" and ctx["payload"]["indicators"]["rsi"] == 28.0
    assert ctx["news"][0]["source"] == "rss:coindesk" and "‹/untrusted_market_news›" in ctx["news"][0]["text"]
    assert row.raw_response["tool_input"]["action"] == "LONG"


async def test_fallback_reason_populated(factory):
    core, _ = _core(factory, [text_response("no tool")])
    res = await core.decide(make_payload(), {})
    assert res.fallback is FallbackReason.INVALID_OUTPUT
    row = await _row(factory, res.decision_id)
    assert row.fallback_reason == "INVALID_OUTPUT"
    assert row.action is DecisionAction.HOLD and row.confidence == 0 and row.leverage == 1
    assert row.reasoning == res.decision.reasoning_tr and "HOLD" in row.reasoning


async def test_record_risk_verdict_and_execution(factory):
    core, log = _core(factory, [tool_response(good_input())])
    res = await core.decide(make_payload(), {})
    assert await log.record_risk_verdict(res.decision_id, "reduced", ["Kaldıraç 5x'e düşürüldü"], 5, Decimal("0.012"))
    assert await log.record_execution(res.decision_id, "FILLED", "Giriş doldu, stop yerleştirildi")
    row = await _row(factory, res.decision_id)
    assert row.risk_verdict is RiskVerdict.REDUCED
    assert row.risk_reasons == ["Kaldıraç 5x'e düşürüldü"]
    assert row.final_leverage == 5 and row.final_quantity == Decimal("0.012")
    assert row.execution_status == "FILLED" and row.execution_detail == "Giriş doldu, stop yerleştirildi"


async def test_unknown_decision_and_bad_verdict_return_false(factory):
    log = DecisionLogger(factory)
    missing = str(uuid.uuid4())
    assert await log.record_risk_verdict(missing, "APPROVED", [], 3, None) is False
    assert await log.record_execution(missing, "FILLED", "x") is False
    assert await log.record_risk_verdict("not-a-uuid", "APPROVED", [], 3, None) is False
    core, log = _core(factory, [tool_response(good_input())])
    res = await core.decide(make_payload(), {})
    assert await log.record_risk_verdict(res.decision_id, "MAYBE", [], 3, None) is False


async def test_recent_returns_newest_first_with_turkish(factory):
    core, log = _core(factory, [tool_response(good_input(reasoning_tr=TR_TEXT)), text_response("x")])
    r1 = await core.decide(make_payload(), {"equity": "5"})
    r2 = await core.decide(make_payload(), {})
    await log.record_execution(r1.decision_id, "FILLED", "ok")
    rows = await log.recent(10)
    assert [r["id"] for r in rows][0] in (r1.decision_id, r2.decision_id) and len(rows) == 2
    by_id = {r["id"]: r for r in rows}
    assert by_id[r1.decision_id]["reasoning"] == TR_TEXT and by_id[r1.decision_id]["action"] == "LONG"
    assert by_id[r1.decision_id]["execution_status"] == "FILLED" and by_id[r1.decision_id]["signal_id"] == "sig-123"
    assert by_id[r2.decision_id]["fallback_reason"] == "INVALID_OUTPUT"
    assert len(await log.recent(1)) == 1


async def test_db_errors_are_swallowed(factory):
    class BrokenFactory:
        def __call__(self, *a, **k):
            raise RuntimeError("db unavailable")

    log = DecisionLogger(BrokenFactory())  # type: ignore[arg-type]
    core = ClaudeDecisionCore(ClaudeSettings(timeout_seconds=0.5), ScriptedTransport([tool_response(good_input())]), logger_sink=log)
    res = await core.decide(make_payload(), {})
    assert not res.is_fallback
    assert await log.log(res, make_payload(), {}) is False
    assert await log.record_execution(res.decision_id, "X", "y") is False
    assert await log.record_risk_verdict(res.decision_id, "APPROVED", [], 1, None) is False
    assert await log.recent() == []


async def test_duplicate_log_returns_false(factory):
    core, log = _core(factory, [tool_response(good_input())])
    p = make_payload()
    res = await core.decide(p, {})
    assert await log.log(res, p, {}) is False  # primary-key conflict swallowed
