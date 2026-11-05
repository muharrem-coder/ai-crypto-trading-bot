"""End-to-end Phase 1 checks: real PaperBrokerAdapter + HardRiskEngine + KillSwitch + Watchdog + DB stores.

No network. These tests prove the layers written separately actually fit together.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import select

from app.brokers.paper_adapter import PaperBrokerAdapter
from app.config.settings import RiskSettings
from app.core.types import MarginMode, OrderRequest, OrderType, PositionSide, Side
from app.db.models import RiskEvent
from app.db.session import async_session_factory, create_engine_from_url, init_db
from app.risk.db_adapters import DbKillSwitchStore, DbRiskEventSink
from app.risk.kill_switch import KillSwitch
from app.risk.protocols import BrokerLike
from app.risk.risk_engine import HardRiskEngine, TradeProposal, Verdict
from app.risk.testing import make_spec
from app.risk.watchdog import RiskWatchdog

SYMBOL = "BTC/USDT:USDT"
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
async def db():
    engine = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    factory = async_session_factory(engine)
    yield factory
    await engine.dispose()


@pytest.fixture
def broker() -> PaperBrokerAdapter:
    b = PaperBrokerAdapter(
        instruments={SYMBOL: make_spec(SYMBOL)},
        starting_balance=D("10000"),
        quote="USDT",
        slippage_bps=D("2"),
        now=lambda: NOW,
    )
    b.update_price(SYMBOL, D("99999.9"), D("100000.0"), mark=D("100000"))
    return b


async def _stack(broker, db, settings: RiskSettings | None = None):
    settings = settings or RiskSettings()
    sink = DbRiskEventSink(db)
    store = DbKillSwitchStore(db)
    ks = KillSwitch(broker, store, sink, settings, clock=lambda: NOW, sleep=_no_sleep)
    await ks.start()
    engine = HardRiskEngine(settings, ks, sink)
    return settings, ks, engine, sink, store


async def _open_via_engine(broker, engine, proposal):
    spec = await broker.get_instrument(SYMBOL)
    snap = await broker.fetch_account()
    ticker = await broker.fetch_ticker(SYMBOL)
    decision = await engine.validate(proposal, snap, spec, None, ticker)
    assert decision.approved, decision.reasons
    await broker.set_margin_mode(SYMBOL, MarginMode.ISOLATED)
    await broker.set_leverage(SYMBOL, decision.approved_leverage)
    await broker.create_order(
        OrderRequest(SYMBOL, proposal.side.entry_side, OrderType.MARKET, decision.approved_quantity,
                     leverage=decision.approved_leverage, margin_mode=MarginMode.ISOLATED)
    )
    for req in engine.compute_protective_orders(decision, proposal, spec):
        await broker.create_order(req)
    return decision


def test_paper_broker_satisfies_risk_protocol(broker):
    assert isinstance(broker, BrokerLike)


async def test_llm_leverage_is_clamped_and_hard_stop_limits_loss(broker, db):
    _, ks, engine, _, _ = await _stack(broker, db)
    proposal = TradeProposal(SYMBOL, PositionSide.LONG, D("100000"), D("99000"), D("103000"),
                             leverage=10, confidence=D("0.8"), source="claude")
    decision = await _open_via_engine(broker, engine, proposal)
    assert decision.verdict is Verdict.REDUCED and decision.approved_leverage == 5

    orders = await broker.fetch_open_orders(SYMBOL)
    assert any(o.order_type is OrderType.STOP_MARKET and o.reduce_only for o in orders)
    watchdog = RiskWatchdog(broker, ks, DbRiskEventSink(db), RiskSettings())
    report = await watchdog.run_once()
    assert report.ok and not report.repaired and not report.closed

    broker.update_price(SYMBOL, D("98800"), D("98800.1"), mark=D("98800"))
    assert await broker.fetch_positions() == []
    equity = (await broker.fetch_account()).equity
    loss = D("10000") - equity
    # max_risk_per_trade_pct = 0.5% of 10k = $50, sizing already includes fees + slippage;
    # gap through the stop (99000 -> 98800) adds a little on top.
    assert D("40") < loss < D("65"), loss


async def test_micro_scalp_rejected_by_fee_gate_and_logged(broker, db):
    _, _, engine, _, _ = await _stack(broker, db)
    spec = await broker.get_instrument(SYMBOL)
    proposal = TradeProposal(SYMBOL, PositionSide.LONG, D("100000"), D("99850"), D("100110"),
                             leverage=3, confidence=D("0.9"), source="claude")
    decision = await engine.validate(proposal, await broker.fetch_account(), spec, None,
                                     await broker.fetch_ticker(SYMBOL))
    assert decision.verdict is Verdict.REJECTED and "FEE_GATE" in decision.reason_codes
    async with db() as s:
        events = (await s.execute(select(RiskEvent))).scalars().all()
    assert any(e.code == "RISK_REJECTED" for e in events)


async def test_kill_switch_flattens_persists_and_blocks_after_restart(broker, db):
    settings = RiskSettings(max_risk_per_trade_pct=D("0.05"), max_position_notional_pct=D("5"),
                            max_stop_distance_pct=D("0.2"))
    _, ks, engine, _, _ = await _stack(broker, db, settings)
    proposal = TradeProposal(SYMBOL, PositionSide.LONG, D("100000"), D("90000"), D("120000"),
                             leverage=3, confidence=D("0.8"))
    await _open_via_engine(broker, engine, proposal)
    await ks.evaluate(await broker.fetch_account())  # sets day_start_equity baseline
    assert not ks.is_engaged

    broker.update_price(SYMBOL, D("92000"), D("92000.1"), mark=D("92000"))  # > 3% equity drawdown
    status = await ks.evaluate(await broker.fetch_account())
    assert status.engaged and ks.is_engaged
    assert await broker.fetch_positions() == []
    assert await broker.fetch_open_orders(None) == []

    # "Restart": brand-new objects, same database.
    _, ks2, engine2, _, _ = await _stack(broker, db, settings)
    assert ks2.is_engaged
    broker.update_price(SYMBOL, D("99999.9"), D("100000.0"), mark=D("100000"))
    d = await engine2.validate(proposal, await broker.fetch_account(), await broker.get_instrument(SYMBOL),
                               None, await broker.fetch_ticker(SYMBOL))
    assert d.verdict is Verdict.REJECTED and "KILL_SWITCH" in d.reason_codes


async def test_watchdog_repairs_missing_stop(broker, db):
    _, ks, engine, _, _ = await _stack(broker, db)
    await broker.create_order(OrderRequest(SYMBOL, Side.BUY, OrderType.MARKET, D("0.01"), leverage=2,
                                           margin_mode=MarginMode.ISOLATED))
    repaired: list[str] = []

    async def repair(position):
        repaired.append(position.symbol)
        await broker.create_order(OrderRequest(SYMBOL, Side.SELL, OrderType.STOP_MARKET, position.quantity,
                                               stop_price=D("98000"), reduce_only=True))

    wd = RiskWatchdog(broker, ks, DbRiskEventSink(db), RiskSettings(), repair_stop=repair)
    report = await wd.run_once()
    assert repaired == [SYMBOL] and SYMBOL in report.repaired
    assert len(await broker.fetch_positions()) == 1
