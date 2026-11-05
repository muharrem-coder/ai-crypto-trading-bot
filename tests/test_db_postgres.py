"""ORM + migration tests against a real PostgreSQL (TimescaleDB) server.

Skipped unless QUANT_TEST_DATABASE_URL is set, e.g.
    QUANT_TEST_DATABASE_URL=postgresql+asyncpg://quant:pw@127.0.0.1:55432/quant pytest -m postgres
The database is shared with whatever lives there, so every test uses unique symbols / ids.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.core.types import AccountSnapshot, AssetClass, MarginMode, MarketType, PositionSide, TradingMode
from app.db import repository as repo
from app.db.models import DecisionAction, ExitReason, NewsItem, RiskLevel, RiskVerdict, SentimentLabel, TradeStatus
from app.db.session import apply_sql_migrations, async_session_factory, create_engine_from_url, get_session

DATABASE_URL = os.environ.get("QUANT_TEST_DATABASE_URL", "")

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DATABASE_URL, reason="QUANT_TEST_DATABASE_URL is not set"),
]

DOMAIN_TABLES = {
    "accounts", "instruments", "signals", "claude_decisions", "orders", "fills", "trades",
    "news_items", "risk_events", "kill_switch_state", "equity_snapshots", "market_candles", "position_snapshots",
}


@pytest_asyncio.fixture
async def engine():
    eng = create_engine_from_url(DATABASE_URL)
    await apply_sql_migrations(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def factory(engine):
    return async_session_factory(engine)


@pytest_asyncio.fixture
async def account(factory):
    async with get_session(factory) as s:
        return await repo.get_or_create_account(
            s, broker="binance", asset_class=AssetClass.CRYPTO, market_type=MarketType.PERPETUAL,
            trading_mode=TradingMode.PAPER, quote_currency="USDT",
        )


async def test_migrations_idempotent_and_schema_complete(engine):
    assert await apply_sql_migrations(engine) == []  # second run: nothing to do
    async with engine.connect() as conn:
        tables = {r[0] for r in await conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))}
        assert DOMAIN_TABLES | {"schema_migrations"} <= tables
        has_ts = (await conn.execute(text("SELECT count(*) FROM pg_extension WHERE extname = 'timescaledb'"))).scalar_one()
        if has_ts:
            hyper = {r[0] for r in await conn.execute(text(
                "SELECT hypertable_name FROM timescaledb_information.hypertables"))}
            assert {"equity_snapshots", "market_candles", "position_snapshots"} <= hyper
        singleton = (await conn.execute(text("SELECT engaged FROM kill_switch_state WHERE id = 1"))).scalar_one()
        assert singleton is False


async def test_save_decision_turkish_reasoning_roundtrip(factory):
    reasoning = "Fiyat EMA20 üzerinde, hacim artışı var; işlem açmak mantıklı. Şüpheli haber yok: çğıöşü ÇĞİÖŞÜ."
    symbol = f"T{uuid.uuid4().hex[:8]}/USDT:USDT"
    async with get_session(factory) as s:
        d = await repo.save_decision(
            s, symbol=symbol, model="claude-test", action=DecisionAction.LONG, confidence=Decimal("0.7300"),
            reasoning=reasoning, leverage=3, stop_loss=Decimal("99.5"), take_profit=Decimal("103.25"),
            input_context={"haber": "Merkez Bankası faiz kararı", "n": [1, 2, 3]},
            raw_response={"ok": True}, latency_ms=812,
        )
        decision_id = d.id
        await repo.update_decision_risk_verdict(
            s, decision_id, RiskVerdict.REDUCED, reasons=["kaldıraç düşürüldü"], final_leverage=2,
            final_quantity=Decimal("0.015"),
        )
    async with get_session(factory) as s:
        rows = await repo.list_recent_decisions(s, limit=5, symbol=symbol)
        assert len(rows) == 1
        row = rows[0]
        assert row.reasoning == reasoning
        assert row.input_context == {"haber": "Merkez Bankası faiz kararı", "n": [1, 2, 3]}
        assert row.risk_verdict is RiskVerdict.REDUCED
        assert row.risk_reasons == ["kaldıraç düşürüldü"]
        assert row.confidence == Decimal("0.7300")
        jtype = (await s.execute(text("SELECT jsonb_typeof(input_context) FROM claude_decisions WHERE id = :i"),
                                 {"i": decision_id})).scalar_one()
        assert jtype == "object"


async def test_confidence_check_constraint(factory):
    with pytest.raises(IntegrityError):
        async with get_session(factory) as s:
            await repo.save_decision(
                s, symbol="X/USDT:USDT", model="m", action=DecisionAction.HOLD, confidence=Decimal("1.5"), reasoning="x")


async def test_open_and_close_trade(factory, account):
    symbol = f"T{uuid.uuid4().hex[:8]}/USDT:USDT"
    async with get_session(factory) as s:
        t = await repo.open_trade(
            s, account_id=account.id, symbol=symbol, side=PositionSide.LONG, leverage=3, margin_mode=MarginMode.ISOLATED,
            entry_price=Decimal("100"), quantity=Decimal("2"), initial_stop=Decimal("95"), fees_paid=Decimal("0.08"))
        trade_id = t.id
        assert t.status is TradeStatus.OPEN
    async with get_session(factory) as s:
        closed = await repo.close_trade(
            s, trade_id, exit_price=Decimal("110"), exit_reason=ExitReason.TAKE_PROFIT, fees_paid=Decimal("0.16"))
        assert closed.gross_pnl == Decimal("20")
        assert closed.net_pnl == Decimal("19.84")
        assert closed.status is TradeStatus.CLOSED
        assert closed.r_multiple is not None and closed.r_multiple > 1
    async with get_session(factory) as s:
        with pytest.raises(repo.TradeStateError):
            await repo.close_trade(s, trade_id, exit_price=Decimal("1"), exit_reason=ExitReason.MANUAL)
    async with get_session(factory) as s:
        assert all(t.id != trade_id for t in await repo.list_open_trades(s, account.id))


async def test_record_risk_event(factory):
    code = f"TEST_{uuid.uuid4().hex[:8]}"
    async with get_session(factory) as s:
        await repo.record_risk_event(
            s, level=RiskLevel.WARN, code=code, message="Günlük zarar limiti yaklaştı",
            details={"oran": Decimal("0.025"), "semboller": ["BTC", "ETH"]})
    async with get_session(factory) as s:
        row = (await s.execute(text("SELECT message, details FROM risk_events WHERE code = :c"), {"c": code})).one()
        assert row.message == "Günlük zarar limiti yaklaştı"
        assert row.details["semboller"] == ["BTC", "ETH"]


async def test_kill_switch_state(factory):
    async with get_session(factory) as s:
        row = await repo.set_kill_switch_state(s, engaged=True, reason="test: elle durdurma",
                                               trading_day=datetime.now(timezone.utc).date(),
                                               day_start_equity=Decimal("10000"))
        assert row.engaged and row.engaged_at is not None
    async with get_session(factory) as s:
        row = await repo.get_kill_switch_state(s)
        assert row.engaged and row.reason == "test: elle durdurma"
        assert row.day_start_equity == Decimal("10000")
    async with get_session(factory) as s:
        row = await repo.set_kill_switch_state(s, engaged=False)
        assert not row.engaged and row.engaged_at is None
    async with get_session(factory) as s:
        count = (await s.execute(text("SELECT count(*) FROM kill_switch_state"))).scalar_one()
        assert count == 1


async def test_news_item_jsonb_fields(factory):
    ext = uuid.uuid4().hex
    async with get_session(factory) as s:
        s.add(NewsItem(
            source="test", external_id=ext, url="https://example.com/x", title="Fed faizi sabit tuttu",
            published_at=datetime.now(timezone.utc), symbols=["BTC", "ETH"], macro_tags=["FED"],
            duplicate_sources=["coindesk"], sentiment_score=Decimal("-0.25"), sentiment_label=SentimentLabel.NEGATIVE,
            importance=7, news_hash=ext[:32], summary="Özet metin"))
    async with get_session(factory) as s:
        row = (await s.execute(text(
            "SELECT jsonb_typeof(symbols) jtype, symbols, macro_tags, duplicate_sources, sentiment_label "
            "FROM news_items WHERE source = 'test' AND external_id = :e"), {"e": ext})).one()
        assert row.jtype == "array"
        assert row.symbols == ["BTC", "ETH"] and row.macro_tags == ["FED"] and row.duplicate_sources == ["coindesk"]
        assert row.sentiment_label == "NEGATIVE"
    with pytest.raises(IntegrityError):  # unique (source, external_id)
        async with get_session(factory) as s:
            s.add(NewsItem(source="test", external_id=ext, title="dup", published_at=datetime.now(timezone.utc)))


async def test_equity_snapshot_insert_and_overwrite(factory, account):
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    snap = AccountSnapshot(
        equity=Decimal("10050.123456789"), wallet_balance=Decimal("10000"), available_balance=Decimal("9000"),
        unrealized_pnl=Decimal("50.123456789"), margin_ratio=Decimal("0.1"), positions=(), timestamp=ts)
    async with get_session(factory) as s:
        await repo.save_equity_snapshot(s, account.id, snap, daily_pnl=Decimal("50"), daily_drawdown_pct=Decimal("0"))
    async with get_session(factory) as s:
        await repo.save_equity_snapshot(s, account.id, snap, daily_pnl=Decimal("60"))  # same PK -> overwrite
    async with get_session(factory) as s:
        rows = (await s.execute(text(
            "SELECT equity, daily_pnl FROM equity_snapshots WHERE account_id = :a AND ts = :t"),
            {"a": account.id, "t": ts})).all()
        assert len(rows) == 1
        assert rows[0].equity == Decimal("10050.123456789")
        assert rows[0].daily_pnl == Decimal("60")


async def test_notify_trigger_fires_on_risk_event(engine, factory):
    import asyncio
    import json

    code = f"NOTIFY_{uuid.uuid4().hex[:8]}"
    got: asyncio.Queue[str] = asyncio.Queue()
    async with engine.connect() as conn:
        raw = await conn.get_raw_connection()
        driver = raw.driver_connection
        await driver.add_listener("terminal_events", lambda _c, _pid, _ch, payload: got.put_nowait(payload))
        async with get_session(factory) as s:
            await repo.record_risk_event(s, level=RiskLevel.INFO, code=code, message="bildirim testi")
        payload = json.loads(await asyncio.wait_for(got.get(), timeout=5))
    assert payload["table"] == "risk_events" and payload["data"]["code"] == code
