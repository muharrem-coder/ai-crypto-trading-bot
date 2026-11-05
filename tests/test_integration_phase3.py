"""Full-chain Phase 3 integration (integrator-owned).

Adım 2  mock WebSocket + breaking news -> SignalPayload on Topic.CLAUDE_TRIGGER
Adım 3  ClaudeDecisionCore (scripted Messages API, forced tool_use) -> decision + reasoning in DB
Adım 1  HardRiskEngine: leverage 10 requested -> clamped to 5
Adım 3  ExecutionEngine on the PaperBroker: entry -> hard SL (reduce-only) -> TP
        PositionManager: break-even+fee lock -> stop fill -> trade closed with net >= 0
Plus: naked-position emergency close, price-drift abort, Claude timeout -> HOLD.
Everything runs against the real components; only the exchange (paper) and the Claude API
transport (scripted) are simulated.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.brokers.paper_adapter import PaperBrokerAdapter
from app.claude.contracts import DecisionAction, FallbackReason
from app.claude.core import ClaudeDecisionCore
from app.claude.decision_log import DecisionLogger
from app.claude.transport import Delay, ScriptedTransport, tool_response
from app.config.settings import ClaudeSettings, ExecutionSettings, RiskSettings
from app.core.events import Topic
from app.core.types import OrderType, PositionSide
from app.db.models import ClaudeDecision as DecisionRow
from app.db.models import RiskEvent, Trade
from app.db.session import async_session_factory, create_engine_from_url, init_db
from app.decision_service import DecisionService, broker_portfolio_context
from app.execution.contracts import ExecutionStatus
from app.execution.executor import ExecutionEngine
from app.execution.position_manager import PositionManager
from app.execution.recorder import DbTradeRecorder
from app.execution.testing import FlakyBroker
from app.news.sources import RawNewsItem
from app.risk.db_adapters import DbKillSwitchStore, DbRiskEventSink
from app.risk.kill_switch import KillSwitch
from app.risk.risk_engine import HardRiskEngine
from app.risk.testing import make_spec
from tests.test_integration_phase2 import SYM, Harness, book_levels, breaking_news

pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

REASONING = ("RSI aşırı satım bölgesinde ve fiyat Bollinger alt bandının altına sarktı; emir defteri alıcı "
             "ağırlıklı (OBI pozitif). ETF girişi haberi destekleyici. Komisyon sonrası R:R yaklaşık 2. "
             "Risk: ani satış devam edebilir, bu yüzden stop ATR'nin 1.5 katında.")


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


class Stack:
    """Adım 1 + Adım 3 components wired on top of the Adım 2 harness."""

    def __init__(self, h: Harness, factory, transport: ScriptedTransport, *, broker=None,
                 claude: ClaudeSettings | None = None) -> None:
        self.h = h
        self.paper = PaperBrokerAdapter({SYM: make_spec(SYM)}, Decimal("10000"), "USDT", Decimal("2"),
                                        now=lambda: h.wall[0])
        self.broker = broker(self.paper) if broker else self.paper
        self.factory = factory
        risk = RiskSettings()
        sink = DbRiskEventSink(factory)
        self.kill_switch = KillSwitch(self.broker, DbKillSwitchStore(factory), sink, risk,
                                      clock=lambda: h.wall[0], sleep=_no_sleep)
        self.risk_engine = HardRiskEngine(risk, self.kill_switch, sink)
        self.recorder = DbTradeRecorder(factory, account={"broker": "paper"})
        ex = ExecutionSettings()
        self.manager = PositionManager(self.broker, self.recorder, ex, risk, kill_switch=self.kill_switch,
                                       clock=lambda: h.wall[0], sleep=_no_sleep)
        self.executor = ExecutionEngine(self.broker, self.risk_engine, self.kill_switch, self.manager,
                                        self.recorder, ex, book_provider=h.market.order_book,
                                        clock=lambda: h.wall[0], sleep=_no_sleep)
        self.transport = transport
        self.log = DecisionLogger(factory)
        self.core = ClaudeDecisionCore(claude or ClaudeSettings(), transport, logger_sink=self.log,
                                       clock=lambda: h.wall[0])
        self.service = DecisionService(self.core, self.executor, h.bus, decision_log=self.log,
                                       portfolio_provider=lambda s: broker_portfolio_context(self.broker,
                                                                                             self.kill_switch, s))

    async def start(self) -> None:
        await self.kill_switch.start()
        await self.paper.connect()

    async def quotes(self, mid: float) -> None:
        """Same quote into the market-data feed (book/ticker) and into the paper exchange."""
        bids, asks = book_levels(mid)
        self.paper.update_price(SYM, Decimal(str(bids[0][0])), Decimal(str(asks[0][0])), mark=Decimal(str(mid)))
        await self.h.feed_quotes(mid)

    async def decision_row(self) -> DecisionRow:
        async with self.factory() as s:
            return (await s.execute(select(DecisionRow))).scalar_one()

    async def events(self) -> list[RiskEvent]:
        async with self.factory() as s:
            return list((await s.execute(select(RiskEvent))).scalars().all())


def claude_long(payload, leverage: int = 10) -> dict:
    return {
        "action": "LONG", "confidence": 0.78, "leverage": leverage, "entry_type": "MARKET", "entry_price": None,
        "stop_loss": float(payload.proposed_stop_loss), "take_profit": float(payload.proposed_take_profit),
        "reasoning_tr": REASONING, "risk_flags": ["HIGH_VOLATILITY"],
    }


@pytest.fixture
async def env():
    engine = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    factory = async_session_factory(engine)
    h = Harness()
    await h.start()
    yield h, factory
    await h.stop()
    await engine.dispose()


async def _trigger(h: Harness, stack: Stack):
    sell_close = h.rows[120][4]
    await stack.quotes(sell_close)
    h.mono[0] += 1.2
    await stack.quotes(sell_close)
    await h.market.run_health_check_once()
    assert h.market.gate.is_active(SYM)
    await h.close_bar(120)
    triggers = h.triggers.drain()
    assert len(triggers) == 1
    return triggers[0]


async def test_full_chain_signal_to_claude_to_risk_to_paper_with_trailing_and_db(env):
    h, factory = env
    assert len(await h.news.ingest(breaking_news())) == 1
    stack = Stack(h, factory, ScriptedTransport())
    await stack.start()
    payload = await _trigger(h, stack)
    stack.transport.push(tool_response(claude_long(payload, leverage=10)))

    result, report = await stack.service.handle(payload)

    # Claude: forced tool_use, news fenced as untrusted data
    call = stack.transport.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "submit_trading_decision"}
    user_text = str(call["messages"])
    assert "<untrusted_market_news>" in user_text and "ETF" in user_text
    assert result.decision.action is DecisionAction.LONG and not result.is_fallback

    # Hard-Risk: leverage 10 -> 5; protected entry on the exchange
    assert report.status is ExecutionStatus.OPENED_PROTECTED, report.detail
    assert report.requested_leverage == 10 and report.approved_leverage == 5
    positions = await stack.paper.fetch_positions()
    assert len(positions) == 1 and positions[0].leverage == 5 and positions[0].side is PositionSide.LONG
    orders = await stack.paper.fetch_open_orders(SYM)
    stops = [o for o in orders if o.order_type is OrderType.STOP_MARKET and o.reduce_only]
    tps = [o for o in orders if o.order_type is OrderType.TAKE_PROFIT_MARKET and o.reduce_only]
    assert len(stops) == 1 and len(tps) == 1
    assert not any(o.order_type is OrderType.TRAILING_STOP_MARKET for o in orders)
    assert stack.manager.is_managed(SYM)

    # DB: reasoning verbatim + verdict + execution status
    row = await stack.decision_row()
    assert row.reasoning == REASONING
    assert row.risk_verdict.value == "REDUCED" and row.final_leverage == 5
    assert row.execution_status == "OPENED_PROTECTED" and row.fallback_reason is None
    assert row.input_context["signal_id"] == payload.signal_id

    # PositionManager: price runs up 2 ATR -> stop to break-even + fee
    entry = positions[0].entry_price
    atr = Decimal(str(payload.indicators.atr))
    up = float(entry + 2 * atr)
    await stack.quotes(up)
    bids, asks = book_levels(up)
    actions = await stack.manager.on_price(SYM, Decimal(str(bids[0][0])), Decimal(str(asks[0][0])))
    assert any("BREAKEVEN" in a for a in actions), actions
    managed = stack.manager.positions()[0]
    assert managed.breakeven_locked and managed.current_stop > entry
    resting = [o for o in await stack.paper.fetch_open_orders(SYM) if o.order_type is OrderType.STOP_MARKET]
    assert len(resting) == 1 and resting[0].stop_price == managed.current_stop

    # Price falls back to the stop -> break-even stop fills -> the winner did not become a loser.
    # (mark touches the stop, fill at bid - slippage; BE+fee+buffer covers both.
    #  A true gap THROUGH the stop can still lose - no stop order can prevent that.)
    await stack.quotes(float(managed.current_stop))  # mark touches the stop; fill at bid (half-spread lower)
    await stack.manager.reconcile()
    assert await stack.paper.fetch_positions() == []
    assert await stack.paper.fetch_open_orders(SYM) == []  # OCO: TP cancelled
    assert not stack.manager.is_managed(SYM)
    async with factory() as s:
        trade = (await s.execute(select(Trade))).scalar_one()
    assert trade.status.value == "CLOSED" and trade.exit_reason.value == "TRAILING_STOP"
    assert trade.net_pnl >= 0
    equity = (await stack.paper.fetch_account()).equity
    assert equity >= Decimal("10000") - Decimal("0.01")
    assert stack.kill_switch.state.consecutive_losses == 0


async def test_full_chain_stop_rejected_closes_naked_position(env):
    h, factory = env
    stack = Stack(h, factory, ScriptedTransport(),
                  broker=lambda p: FlakyBroker(p, {OrderType.STOP_MARKET: -1}))
    await stack.start()
    payload = await _trigger(h, stack)
    stack.transport.push(tool_response(claude_long(payload, leverage=3)))

    _, report = await stack.service.handle(payload)

    assert report.status is ExecutionStatus.NAKED_POSITION_CLOSED, report.detail
    assert report.filled_quantity > 0  # the entry DID fill...
    assert await stack.paper.fetch_positions() == []  # ...and was closed at market immediately
    assert await stack.paper.fetch_open_orders(SYM) == []
    assert stack.broker.attempts[OrderType.STOP_MARKET] == 3
    codes = {(e.level.value, e.code) for e in await stack.events()}
    assert ("CRITICAL", "NAKED_POSITION_CLOSED") in codes
    row = await stack.decision_row()
    assert row.execution_status == "NAKED_POSITION_CLOSED"
    assert not stack.kill_switch.is_engaged  # flat was verified, no need to halt


async def test_full_chain_price_drift_during_claude_latency_aborts(env):
    h, factory = env
    stack = Stack(h, factory, ScriptedTransport())
    await stack.start()
    payload = await _trigger(h, stack)
    atr = payload.indicators.atr
    stack.transport.push(Delay(0.02, tool_response(claude_long(payload))))

    # while Claude "thinks", the market runs away by 0.5 ATR
    await stack.quotes(float(payload.proposed_entry) + 0.5 * atr)
    _, report = await stack.service.handle(payload)

    assert report.status is ExecutionStatus.ABORTED_PRICE_DRIFT, report.detail
    assert report.drift_atr is not None and report.drift_atr > Decimal("0.3")
    assert await stack.paper.fetch_positions() == []
    assert await stack.paper.fetch_open_orders(SYM) == []
    row = await stack.decision_row()
    assert row.execution_status == "ABORTED_PRICE_DRIFT" and row.risk_verdict.value == "REJECTED"


async def test_full_chain_claude_timeout_falls_back_to_hold(env):
    h, factory = env
    stack = Stack(h, factory, ScriptedTransport([Delay(1.0)]), claude=ClaudeSettings(timeout_seconds=0.05))
    await stack.start()
    payload = await _trigger(h, stack)
    reasoning_stream = h.bus.subscribe("claude.decision")

    result, report = await stack.service.handle(payload)

    assert result.fallback is FallbackReason.TIMEOUT and result.decision.action is DecisionAction.HOLD
    assert report.status is ExecutionStatus.HOLD
    assert await stack.paper.fetch_positions() == [] and await stack.paper.fetch_open_orders(SYM) == []
    row = await stack.decision_row()
    assert row.fallback_reason == "TIMEOUT" and row.execution_status == "HOLD"
    streamed = reasoning_stream.drain()
    assert streamed and streamed[0]["fallback"] == "TIMEOUT" and streamed[0]["action"] == "HOLD"


async def test_service_loop_consumes_trigger_topic(env):
    h, factory = env
    stack = Stack(h, factory, ScriptedTransport())
    await stack.start()
    reports = h.bus.subscribe("execution.report")
    payload = await _trigger(h, stack)
    stack.transport.push(tool_response(claude_long(payload, leverage=2)))
    stack.service.start()  # subscribes to Topic.CLAUDE_TRIGGER
    h.bus.publish(Topic.CLAUDE_TRIGGER, payload)
    h.bus.publish(Topic.CLAUDE_TRIGGER, payload)  # duplicate while in flight -> dropped, not double-executed
    for _ in range(200):
        await asyncio.sleep(0)
        if reports.queue.qsize():
            break
    await stack.service.stop()
    got = reports.drain()
    assert len(got) == 1 and got[0].status is ExecutionStatus.OPENED_PROTECTED
    assert stack.service.stats["dropped_inflight"] == 1 and len(stack.transport.calls) == 1
