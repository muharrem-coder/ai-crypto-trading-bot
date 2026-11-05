"""DbTradeRecorder against SQLite (aiosqlite). No network."""

from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone
from decimal import Decimal as D

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import SAWarning

from app.core.types import (
    AssetClass,
    FeeSchedule,
    MarketType,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    TradingMode,
)
from app.db import repository as repo
from app.db.models import (
    ClaudeDecision,
    DecisionAction,
    ExitReason,
    OrderPurpose,
    OrderRecord,
    RiskEvent,
    RiskLevel,
    Trade,
    TradeStatus,
)
from app.db.session import async_session_factory, create_engine_from_url, get_session, init_db
from app.execution.contracts import ExecutionReport, ExecutionStatus, ManagedPosition
from app.execution.recorder import DbTradeRecorder, map_exit_reason, map_purpose
from app.risk.testing import make_order

warnings.filterwarnings("ignore", message=".*does not support Decimal objects natively.*", category=SAWarning)
pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

BTC = "BTC/USDT:USDT"
NOW = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
ACCOUNT = dict(broker="binance", asset_class=AssetClass.CRYPTO, market_type=MarketType.PERPETUAL,
               trading_mode=TradingMode.PAPER, quote_currency="USDT")


@pytest_asyncio.fixture
async def engine():
    eng = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def factory(engine):
    return async_session_factory(engine)


@pytest.fixture
def recorder(factory):
    return DbTradeRecorder(factory, account=ACCOUNT)


def managed(decision_id: str | None = None, side: PositionSide = PositionSide.LONG) -> ManagedPosition:
    return ManagedPosition(
        symbol=BTC, side=side, quantity=D("0.1"), entry_price=D("50000"),
        initial_stop=D("49500") if side is PositionSide.LONG else D("50500"), take_profit=D("51000"),
        atr=D("100"), fees=FeeSchedule(D("0.0002"), D("0.0005")), tick_size=D("0.1"),
        stop_order_id="s1", take_profit_order_id="t1", decision_id=decision_id, trade_id=None, opened_at=NOW,
    )


async def make_decision(factory) -> uuid.UUID:
    async with get_session(factory) as s:
        d = await repo.save_decision(
            s, symbol=BTC, model="claude-test", action=DecisionAction.LONG, confidence=D("0.8"),
            reasoning="BTC direnci kırdı; hacim ve haber akışı destekliyor, LONG açıldı.", leverage=3,
        )
        return d.id


def test_mappings():
    assert map_purpose("trailing_stop") is OrderPurpose.TRAILING_STOP
    assert map_purpose("TP") is OrderPurpose.TAKE_PROFIT
    assert map_purpose("EMERGENCY_CLOSE") is OrderPurpose.EXIT
    assert map_purpose("???") is OrderPurpose.EXIT
    assert map_exit_reason("TRAILING_STOP") is ExitReason.TRAILING_STOP
    assert map_exit_reason("CLOSED_BY_DECISION") is ExitReason.SIGNAL_EXIT
    assert map_exit_reason("???") is ExitReason.MANUAL


async def test_record_order_upserts_by_exchange_id(recorder, factory):
    order = make_order(order_type=OrderType.STOP_MARKET, side=Side.SELL, stop="49500")
    await recorder.record_order(order, "STOP_LOSS", None)
    order.status = OrderStatus.CANCELED
    await recorder.record_order(order, "STOP_LOSS", None)
    async with get_session(factory) as s:
        rows = (await s.execute(select(OrderRecord))).scalars().all()
    assert len(rows) == 1
    assert rows[0].purpose is OrderPurpose.STOP_LOSS and rows[0].status is OrderStatus.CANCELED
    assert rows[0].stop_price == D("49500") and rows[0].reduce_only is True
    assert recorder.errors == 0


async def test_record_order_links_existing_decision_and_ignores_unknown(recorder, factory):
    did = await make_decision(factory)
    await recorder.record_order(make_order(), "ENTRY", str(did))
    await recorder.record_order(make_order(), "ENTRY", str(uuid.uuid4()))  # unknown decision: no FK failure
    await recorder.record_order(make_order(), "ENTRY", "not-a-uuid")
    async with get_session(factory) as s:
        rows = (await s.execute(select(OrderRecord).order_by(OrderRecord.created_at, OrderRecord.id))).scalars().all()
    assert len(rows) == 3 and recorder.errors == 0
    assert sorted(str(r.decision_id) for r in rows if r.decision_id) == [str(did)]


async def test_open_and_close_trade_net_pnl(recorder, factory):
    did = await make_decision(factory)
    tid = await recorder.open_trade(managed(str(did)), leverage=3)
    assert tid is not None and uuid.UUID(tid)
    async with get_session(factory) as s:
        trade = await s.get(Trade, uuid.UUID(tid))
        assert trade.status is TradeStatus.OPEN and trade.decision_id == did and trade.leverage == 3
        assert abs(trade.fees_paid - D("2.5")) < D("1e-9")  # taker entry fee estimate
    # exit 50500: gross 50, fees round trip 5.0 -> net 45
    net = await recorder.close_trade(tid, D("50500"), "TAKE_PROFIT", D("5.0"))
    assert net == pytest.approx(D("45.0"), abs=D("1e-9"))  # SQLite stores NUMERIC as float
    async with get_session(factory) as s:
        trade = await s.get(Trade, uuid.UUID(tid))
        assert trade.status is TradeStatus.CLOSED and trade.exit_reason is ExitReason.TAKE_PROFIT
        assert abs(trade.net_pnl - D("45")) < D("1e-9") and abs(trade.gross_pnl - D("50")) < D("1e-9")
    assert recorder.errors == 0
    # closing twice is a swallowed error
    assert await recorder.close_trade(tid, D("50500"), "TAKE_PROFIT", D("5.0")) is None
    assert recorder.errors == 1


async def test_short_trade_loss(recorder):
    tid = await recorder.open_trade(managed(side=PositionSide.SHORT), leverage=2)
    net = await recorder.close_trade(tid, D("50500"), "STOP_LOSS", D("10"))
    assert net == pytest.approx(D("-60"), abs=D("1e-9"))  # gross -50 - fees 10


async def test_close_trade_without_id_is_noop(recorder):
    assert await recorder.close_trade(None, D("1"), "MANUAL", None) is None
    assert await recorder.close_trade("garbage", D("1"), "MANUAL", None) is None
    assert recorder.errors == 0


async def test_account_is_cached(recorder, factory):
    await recorder.open_trade(managed(), 1)
    await recorder.open_trade(managed(), 1)
    async with get_session(factory) as s:
        from app.db.models import Account
        assert len((await s.execute(select(Account))).scalars().all()) == 1


async def test_record_execution_updates_decision(recorder, factory):
    did = await make_decision(factory)
    report = ExecutionReport(
        status=ExecutionStatus.OPENED_PROTECTED, symbol=BTC, decision_id=str(did), signal_id=None,
        detail="Pozisyon açıldı, SL aktif", approved_leverage=3, approved_quantity=D("0.1"),
    )
    await recorder.record_execution(report)
    async with get_session(factory) as s:
        d = await s.get(ClaudeDecision, did)
    assert d.execution_status == "OPENED_PROTECTED"
    assert d.execution_detail == "Pozisyon açıldı, SL aktif"
    assert d.final_leverage == 3 and abs(d.final_quantity - D("0.1")) < D("1e-9")
    assert recorder.errors == 0


async def test_record_execution_ignores_missing_or_invalid_decision(recorder):
    for did in (None, "nope", str(uuid.uuid4())):
        await recorder.record_execution(
            ExecutionReport(status=ExecutionStatus.HOLD, symbol=BTC, decision_id=did, signal_id=None, detail="x")
        )
    assert recorder.errors == 0


async def test_risk_event(recorder, factory):
    await recorder.risk_event("CRITICAL", "STOP_MISSING", "stop yok", {"price": D("1.5"), "symbol": BTC})
    await recorder.risk_event("weird", "X", "unknown level", {})
    async with get_session(factory) as s:
        rows = (await s.execute(select(RiskEvent).order_by(RiskEvent.code))).scalars().all()
    assert [(r.level, r.code) for r in rows] == [(RiskLevel.CRITICAL, "STOP_MISSING"), (RiskLevel.WARN, "X")]
    assert rows[0].details["price"] == "1.5"


async def test_db_errors_are_swallowed_and_counted():
    eng = create_engine_from_url("sqlite+aiosqlite:///:memory:")  # no tables created
    rec = DbTradeRecorder(async_session_factory(eng), account=ACCOUNT)
    await rec.record_order(make_order(), "ENTRY", None)
    assert await rec.open_trade(managed(), 2) is None
    await rec.risk_event("WARN", "C", "m", {})
    assert rec.errors == 3 and rec.last_error
    await eng.dispose()


async def test_errors_swallowed_after_engine_disposed(engine, factory):
    rec = DbTradeRecorder(factory, account=ACCOUNT)
    tid = await rec.open_trade(managed(), 2)
    assert tid is not None
    await engine.dispose()  # in-memory DB is gone with the connection
    assert await rec.close_trade(tid, D("50500"), "TAKE_PROFIT", D("1")) is None
    await rec.record_order(make_order(), "STOP_LOSS", None)
    await rec.record_execution(
        ExecutionReport(status=ExecutionStatus.ERROR, symbol=BTC, decision_id=str(uuid.uuid4()), signal_id=None, detail="x")
    )
    assert rec.errors >= 2
