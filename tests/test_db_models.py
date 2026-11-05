"""Persistence layer tests (SQLite + aiosqlite in memory)."""

from __future__ import annotations

import dataclasses
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SAWarning

from app.core.types import (
    AccountSnapshot,
    AssetClass,
    Fill,
    LiquidityRole,
    MarginMode,
    MarketType,
    Order,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    TradingMode,
)
from app.db import repository as repo
from app.db.models import (
    Base,
    ClaudeDecision,
    DecisionAction,
    ExitReason,
    KillSwitchState,
    OrderPurpose,
    RiskEvent,
    RiskLevel,
    RiskVerdict,
    TradeStatus,
)
from app.db.session import (
    apply_sql_migrations,
    async_session_factory,
    create_engine_from_url,
    get_session,
    init_db,
)

warnings.filterwarnings("ignore", message=".*does not support Decimal objects natively.*", category=SAWarning)

MIGRATION = Path(__file__).resolve().parents[1] / "migrations" / "001_initial_schema.sql"
NOW = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def engine():
    eng = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def factory(engine):
    return async_session_factory(engine)


@pytest_asyncio.fixture
async def session(factory):
    async with get_session(factory) as s:
        yield s


@pytest_asyncio.fixture
async def account(session):
    return await repo.get_or_create_account(
        session,
        broker="binance",
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.PERPETUAL,
        trading_mode=TradingMode.PAPER,
        quote_currency="USDT",
    )


async def test_create_all_creates_every_table(engine):
    async with engine.connect() as conn:
        names = await conn.run_sync(lambda c: set(__import__("sqlalchemy").inspect(c).get_table_names()))
    assert set(Base.metadata.tables) <= names
    assert len(Base.metadata.tables) == 13


async def test_apply_sql_migrations_skips_non_postgres(engine):
    assert await apply_sql_migrations(engine, MIGRATION.parent) == []


async def test_decision_with_turkish_reasoning_roundtrip(factory, account):
    reasoning = "BTC 4 saatlik grafikte direnci kırdı; hacim artışı ve pozitif haber akışı nedeniyle LONG açtım. Stop: 61.200 ₺ değil, 61.200 USDT."
    async with get_session(factory) as s:
        d = await repo.save_decision(
            s,
            symbol="BTC/USDT:USDT",
            model="claude-test",
            action=DecisionAction.LONG,
            confidence=Decimal("0.7250"),
            reasoning=reasoning,
            leverage=3,
            stop_loss=Decimal("61200.5"),
            take_profit=Decimal("64000"),
            input_context={"signal": {"score": Decimal("0.81")}, "news": ["ETF onayı"], "portfolio": {"equity": Decimal("10000")}},
            raw_response={"action": "LONG"},
        )
        decision_id = d.id
    async with get_session(factory) as s:
        row = await s.get(ClaudeDecision, decision_id)
        assert row.reasoning == reasoning
        assert row.action is DecisionAction.LONG
        assert row.confidence == Decimal("0.725")
        assert row.input_context["signal"]["score"] == "0.81"  # Decimals stored as strings
        assert row.input_context["news"] == ["ETF onayı"]
        assert row.created_at.tzinfo is not None and row.created_at.utcoffset() == timedelta(0)
        recent = await repo.list_recent_decisions(s, limit=5)
        assert [r.id for r in recent] == [decision_id]


async def test_update_decision_risk_verdict(factory):
    async with get_session(factory) as s:
        d = await repo.save_decision(
            s, symbol="ETH/USDT:USDT", model="m", action=DecisionAction.SHORT, confidence=Decimal("0.6"),
            reasoning="Aşırı alım bölgesi.", leverage=5,
        )
        await repo.update_decision_risk_verdict(
            s, d.id, RiskVerdict.REDUCED, ["Kaldıraç düşürüldü"], final_leverage=2, final_quantity=Decimal("0.5")
        )
        assert d.risk_verdict is RiskVerdict.REDUCED
        assert d.final_leverage == 2
        assert d.risk_reasons == ["Kaldıraç düşürüldü"]
        with pytest.raises(repo.RecordNotFoundError):
            await repo.update_decision_risk_verdict(s, __import__("uuid").uuid4(), RiskVerdict.APPROVED)


@pytest.mark.parametrize("confidence,leverage", [(Decimal("1.5"), 2), (Decimal("-0.1"), 2), (Decimal("0.5"), 21), (Decimal("0.5"), 0)])
async def test_decision_check_constraints(factory, confidence, leverage):
    with pytest.raises(IntegrityError):
        async with get_session(factory) as s:
            await repo.save_decision(
                s, symbol="BTC/USDT:USDT", model="m", action=DecisionAction.LONG, confidence=confidence,
                reasoning="x", leverage=leverage,
            )


async def test_decision_requires_reasoning(factory):
    with pytest.raises(IntegrityError):
        async with get_session(factory) as s:
            s.add(ClaudeDecision(symbol="BTC/USDT:USDT", model="m", action=DecisionAction.HOLD, confidence=Decimal("0.5"), reasoning=None))
            await s.flush()


async def test_kill_switch_singleton_enforced(factory):
    async with get_session(factory) as s:
        state = await repo.get_kill_switch_state(s)
        assert state.id == 1 and state.engaged is False

    with pytest.raises(IntegrityError):  # id must be 1
        async with get_session(factory) as s:
            s.add(KillSwitchState(id=2, engaged=False))
            await s.flush()

    with pytest.raises(IntegrityError):  # duplicate id 1
        async with get_session(factory) as s:
            s.add(KillSwitchState(id=1, engaged=True))
            await s.flush()


async def test_kill_switch_engage_and_release(factory):
    async with get_session(factory) as s:
        st = await repo.set_kill_switch_state(
            s, engaged=True, reason="Günlük zarar limiti aşıldı", trading_day=date(2026, 1, 5), day_start_equity=Decimal("10000")
        )
        assert st.engaged and st.engaged_at is not None
    async with get_session(factory) as s:
        st = await repo.set_kill_switch_state(s, engaged=False)
        assert not st.engaged and st.engaged_at is None
        assert st.trading_day == date(2026, 1, 5)  # kept
        assert st.day_start_equity == Decimal("10000")
        assert len((await s.execute(select(KillSwitchState))).scalars().all()) == 1


async def test_open_close_trade_long_pnl_and_r_multiple(factory, account):
    async with get_session(factory) as s:
        t = await repo.open_trade(
            s, account_id=account.id, symbol="BTC/USDT:USDT", side=PositionSide.LONG, leverage=3,
            margin_mode=MarginMode.ISOLATED, entry_price=Decimal("100"), quantity=Decimal("2"),
            initial_stop=Decimal("95"), opened_at=NOW,
        )
        assert t.status is TradeStatus.OPEN
        assert [x.id for x in await repo.list_open_trades(s)] == [t.id]
        closed = await repo.close_trade(
            s, t.id, exit_price=Decimal("110"), exit_reason=ExitReason.TAKE_PROFIT,
            fees_paid=Decimal("1.5"), funding_paid=Decimal("0.5"), closed_at=NOW + timedelta(hours=3),
        )
        # gross = (110-100)*2 = 20 ; net = 20 - 1.5 - 0.5 = 18 ; risk = |100-95|*2 = 10 ; R = 1.8
        assert closed.gross_pnl == Decimal("20")
        assert closed.net_pnl == Decimal("18")
        assert closed.r_multiple == Decimal("1.8")
        assert closed.status is TradeStatus.CLOSED and closed.exit_reason is ExitReason.TAKE_PROFIT
        assert await repo.list_open_trades(s) == []
        with pytest.raises(repo.TradeStateError):
            await repo.close_trade(s, t.id, exit_price=Decimal("1"), exit_reason=ExitReason.MANUAL)


async def test_close_trade_short_loss_and_no_stop(factory, account):
    async with get_session(factory) as s:
        short = await repo.open_trade(
            s, account_id=account.id, symbol="ETH/USDT:USDT", side=PositionSide.SHORT, leverage=2,
            margin_mode=MarginMode.ISOLATED, entry_price=Decimal("50"), quantity=Decimal("4"), initial_stop=Decimal("52"),
        )
        c = await repo.close_trade(s, short.id, exit_price=Decimal("52"), exit_reason=ExitReason.STOP_LOSS, fees_paid=Decimal("0.4"))
        # gross = (50-52)*4 = -8 ; net = -8.4 ; risk = 2*4 = 8 ; R = -1.05
        assert c.gross_pnl == Decimal("-8")
        assert c.net_pnl == Decimal("-8.4")
        assert c.r_multiple == Decimal("-1.05")

        nostop = await repo.open_trade(
            s, account_id=account.id, symbol="SOL/USDT:USDT", side=PositionSide.LONG, leverage=1,
            margin_mode=MarginMode.ISOLATED, entry_price=Decimal("10"), quantity=Decimal("1"),
        )
        c2 = await repo.close_trade(s, nostop.id, exit_price=Decimal("11"), exit_reason=ExitReason.MANUAL)
        assert c2.net_pnl == Decimal("1") and c2.r_multiple is None


async def test_close_unknown_trade(session):
    with pytest.raises(repo.RecordNotFoundError):
        await repo.close_trade(session, __import__("uuid").uuid4(), exit_price=Decimal("1"), exit_reason=ExitReason.MANUAL)


async def test_record_risk_event(factory):
    async with get_session(factory) as s:
        ev = await repo.record_risk_event(
            s, level=RiskLevel.CRITICAL, code="KILL_SWITCH", message="Kill switch devrede",
            details={"drawdown": Decimal("0.031"), "limit": Decimal("0.03")},
        )
        assert ev.id is not None
    async with get_session(factory) as s:
        rows = (await s.execute(select(RiskEvent))).scalars().all()
        assert len(rows) == 1
        assert rows[0].level is RiskLevel.CRITICAL
        assert rows[0].details == {"drawdown": "0.031", "limit": "0.03"}
        assert rows[0].ts.tzinfo is not None


def _order(**kw) -> Order:
    base = dict(
        order_id="EX-1", client_order_id="CL-1", symbol="BTC/USDT:USDT", side=Side.BUY, order_type=OrderType.LIMIT,
        status=OrderStatus.NEW, quantity=Decimal("1"), filled_quantity=Decimal("0"), price=Decimal("100"),
        average_fill_price=None, stop_price=None, reduce_only=False, fee_paid=Decimal("0"), fee_currency="USDT",
        created_at=NOW, updated_at=NOW, raw={"info": {"x": Decimal("1.5")}},
    )
    base.update(kw)
    return Order(**base)


async def test_order_upsert_and_fill_idempotent(factory, account):
    async with get_session(factory) as s:
        o1 = await repo.upsert_order_from_domain(s, _order(), OrderPurpose.ENTRY, None, account.id)
        o2 = await repo.upsert_order_from_domain(
            s,
            _order(status=OrderStatus.FILLED, filled_quantity=Decimal("1"), average_fill_price=Decimal("100"), fee_paid=Decimal("0.04")),
            OrderPurpose.ENTRY, None, account.id,
        )
        assert o1.id == o2.id and o2.status is OrderStatus.FILLED
        # stale non-terminal update is ignored
        o3 = await repo.upsert_order_from_domain(s, _order(status=OrderStatus.NEW), OrderPurpose.ENTRY, None, account.id)
        assert o3.status is OrderStatus.FILLED

        fill = Fill(
            trade_id="T1", order_id="EX-1", symbol="BTC/USDT:USDT", side=Side.BUY, price=Decimal("100"),
            quantity=Decimal("1"), fee=Decimal("0.04"), fee_currency="USDT", liquidity=LiquidityRole.TAKER, timestamp=NOW,
        )
        f1 = await repo.record_fill(s, fill, account.id)
        f2 = await repo.record_fill(s, fill, account.id)
        assert f1.id == f2.id
        with pytest.raises(repo.RecordNotFoundError):
            await repo.record_fill(s, dataclasses.replace(fill, order_id="NOPE"), account.id)


async def test_save_equity_snapshot(factory, account):
    snap = AccountSnapshot(
        equity=Decimal("10050"), wallet_balance=Decimal("10000"), available_balance=Decimal("9000"),
        unrealized_pnl=Decimal("50"), margin_ratio=Decimal("0.1"), positions=(), timestamp=NOW,
    )
    async with get_session(factory) as s:
        await repo.save_equity_snapshot(s, account.id, snap, daily_pnl=Decimal("50"), daily_drawdown_pct=Decimal("0"))
        await repo.save_equity_snapshot(s, account.id, snap, daily_pnl=Decimal("60"))  # same PK -> overwrite
    async with get_session(factory) as s:
        from app.db.models import EquitySnapshot

        rows = (await s.execute(select(EquitySnapshot))).scalars().all()
        assert len(rows) == 1 and rows[0].daily_pnl == Decimal("60") and rows[0].ts == NOW


def test_migration_covers_every_model_table():
    sql = MIGRATION.read_text(encoding="utf-8")
    created = {m.group(1).lower() for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:public\.)?(\w+)", sql, re.I)}
    missing = set(Base.metadata.tables) - created
    assert not missing, f"tables missing from migration: {missing}"
    for hypertable in ("equity_snapshots", "market_candles", "position_snapshots"):
        assert re.search(rf"create_hypertable\(\s*'{hypertable}'\s*,\s*'ts'", sql), hypertable
    assert "terminal_events" in sql
    assert re.search(r"INSERT INTO kill_switch_state.*ON CONFLICT", sql, re.I | re.S)


def test_migration_defines_every_model_constraint_name():
    sql = MIGRATION.read_text(encoding="utf-8")
    names = {c.name for t in Base.metadata.tables.values() for c in t.constraints if c.name}
    names |= {i.name for t in Base.metadata.tables.values() for i in t.indexes}
    missing = {n for n in names if n not in sql}
    assert not missing, f"constraint/index names missing from migration: {sorted(missing)}"
