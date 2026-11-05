"""TradingRuntime orchestration + RuntimeTerminalBackend shapes / control (no network, in-memory SQLite)."""

from __future__ import annotations

import asyncio
import itertools
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from app.api.backend import RuntimeTerminalBackend, metrics_summary_from_context
from app.api.protocol import SystemStatus
from app.brokers.paper_adapter import PaperBrokerAdapter
from app.claude.transport import ScriptedTransport
from app.config.settings import DataSettings, ExecutionSettings, RiskSettings, Settings
from app.core.events import Topic
from app.core.types import OrderRequest, OrderType, PositionSide, Side, utcnow
from app.data.mock_source import ScriptedMarketSource
from app.db import repository
from app.db.models import DecisionAction, RiskVerdict
from app.execution.contracts import ManagedPosition
from app.news.sources import RawNewsItem
from app.risk.testing import make_spec
from app.runtime import TradingRuntime, build_runtime

pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

SYM = "BTC/USDT:USDT"
MEM_DB = "sqlite+aiosqlite:///:memory:"

KPI_KEYS = {
    "equity", "equity_change_pct", "unrealized_pnl", "realized_pnl_24h", "trades_24h", "fees_paid_24h", "win_rate",
    "profit_factor", "daily_drawdown_pct", "daily_drawdown_limit_pct", "system_status", "kill_switch_engaged",
    "kill_switch_reason", "trading_mode", "exchange", "active_symbol", "funding_rate", "claude_model",
    "min_net_profit_usd",
}
POSITION_KEYS = {
    "symbol", "side", "leverage", "quantity", "entry_price", "mark_price", "liquidation_price", "hard_stop",
    "take_profit", "trailing_active", "est_fee", "unrealized_pnl", "net_pnl", "opened_at", "decision_id",
}
TRADE_KEYS = {
    "trade_id", "symbol", "side", "opened_at", "closed_at", "entry_price", "exit_price", "quantity", "leverage",
    "gross_pnl", "fees_paid", "funding_paid", "net_pnl", "r_multiple", "exit_reason", "trigger",
}
CLAUDE_KEYS = {
    "decision_id", "timestamp", "symbol", "action", "confidence", "leverage", "metrics_summary", "reasoning_tr",
    "risk_flags", "risk_decision", "execution_status", "fallback", "latency_ms",
}
NEWS_KEYS = {
    "news_id", "time", "source", "title", "url", "symbols", "macro_tags", "sentiment_score", "sentiment_label",
    "importance", "duplicate_sources",
}


def make_settings(**overrides: Any) -> Settings:
    data = DataSettings(bootstrap_candles=200, stale_after_seconds=60, resume_after_live_seconds=0,
                        health_check_interval_seconds=0.05)
    # long poll intervals: the first watchdog / manager tick runs immediately, later ticks never overlap a test
    # (cancelling a task in the middle of a DB call on the single shared in-memory connection is a test artefact)
    risk = RiskSettings(watchdog_interval_seconds=30)
    execution = ExecutionSettings(manager_poll_seconds=30)
    return Settings(_env_file=None, symbols=[SYM], data=data, risk=risk, execution=execution, **overrides)


def make_paper() -> PaperBrokerAdapter:
    return PaperBrokerAdapter({SYM: make_spec(SYM)}, Decimal("10000"), "USDT", Decimal("2"))


async def make_runtime(settings: Settings | None = None, *, source: ScriptedMarketSource | None = None,
                       broker: PaperBrokerAdapter | None = None, transport: ScriptedTransport | None = None,
                       **kwargs: Any) -> TradingRuntime:
    return await build_runtime(
        settings or make_settings(),
        market_source=source or ScriptedMarketSource(),
        broker=broker or make_paper(),
        transport=transport or ScriptedTransport(),
        news_sources=[],
        database_url=MEM_DB,
        **kwargs,
    )


def history_rows(n: int = 120) -> list[list[float]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = int((start + timedelta(minutes=i)).timestamp() * 1000)
        base = 50000.0 + i
        rows.append([ts, base, base + 5, base - 5, base + 1, 10.0])
    return rows


def book(mid: float = 50000.0) -> tuple[list[list[float]], list[list[float]]]:
    bids = [[round(mid - 0.1 - i * 0.1, 1), 5.0] for i in range(10)]
    asks = [[round(mid + 0.1 + i * 0.1, 1), 5.0] for i in range(10)]
    return bids, asks


async def eventually(pred: Callable[[], Any | Awaitable[Any]], timeout: float = 3.0) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        value = pred()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return value
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


async def feed_market(source: ScriptedMarketSource, mid: float = 50000.0) -> None:
    bids, asks = book(mid)
    source.push_order_book(SYM, bids, asks)
    source.push_ticker(SYM, bids[0][0], asks[0][0], mid)


async def open_long(rt: TradingRuntime, qty: str = "0.1", stop: str = "49000", tp: str = "52000",
                    leverage: int = 3) -> ManagedPosition:
    """Open a protected LONG on the paper broker and hand it to the position manager (what the executor does)."""
    paper = rt.broker
    assert isinstance(paper, PaperBrokerAdapter)
    paper.update_price(SYM, Decimal("49999.9"), Decimal("50000.1"), mark=Decimal("50000"))
    await paper.set_leverage(SYM, leverage)
    await paper.create_order(OrderRequest(SYM, Side.BUY, OrderType.MARKET, Decimal(qty), leverage=leverage))
    stop_order = await paper.create_order(OrderRequest(SYM, Side.SELL, OrderType.STOP_MARKET, Decimal(qty),
                                                       stop_price=Decimal(stop), reduce_only=True))
    tp_order = await paper.create_order(OrderRequest(SYM, Side.SELL, OrderType.TAKE_PROFIT_MARKET, Decimal(qty),
                                                     stop_price=Decimal(tp), reduce_only=True))
    pos = (await paper.fetch_positions())[0]
    spec = make_spec(SYM)
    mp = ManagedPosition(
        symbol=SYM, side=PositionSide.LONG, quantity=pos.quantity, entry_price=pos.entry_price,
        initial_stop=Decimal(stop), take_profit=Decimal(tp), atr=Decimal("100"), fees=spec.fees,
        tick_size=spec.tick_size, stop_order_id=stop_order.order_id, take_profit_order_id=tp_order.order_id,
        decision_id=None, trade_id=None, opened_at=utcnow(),
    )
    mp.trade_id = await rt.recorder.open_trade(mp, leverage)
    await rt.position_manager.register(mp)
    return mp


def fresh_backend(rt: TradingRuntime) -> RuntimeTerminalBackend:
    """Backend whose monotonic clock jumps 10 s per call, so the short-lived caches never serve stale data."""
    ticks = itertools.count()
    return RuntimeTerminalBackend(rt, monotonic=lambda: next(ticks) * 10.0)


@pytest.fixture
async def source() -> ScriptedMarketSource:
    src = ScriptedMarketSource()
    src.set_history(SYM, history_rows())
    return src


@pytest.fixture
async def rt(source):
    runtime = await make_runtime(source=source)
    await runtime.start()
    await eventually(lambda: runtime.watchdog.ticks >= 1)
    await asyncio.sleep(0.15)  # let the first tick finish its DB writes
    yield runtime
    await runtime.stop()


@pytest.fixture
def backend(rt) -> RuntimeTerminalBackend:
    return fresh_backend(rt)


# ====================================================================== runtime lifecycle
async def test_start_order_and_idempotent_stop():
    src = ScriptedMarketSource()
    rt = await make_runtime(source=src)
    order: list[str] = []

    def record(obj: Any, attr: str, label: str) -> None:
        original = getattr(obj, attr)

        if asyncio.iscoroutinefunction(original):
            async def wrapper(*a: Any, **k: Any) -> Any:
                order.append(label)
                return await original(*a, **k)
        else:
            def wrapper(*a: Any, **k: Any) -> Any:
                order.append(label)
                return original(*a, **k)

        setattr(obj, attr, wrapper)

    record(rt.kill_switch, "start", "start:kill_switch")
    record(rt.market, "start", "start:market")
    record(rt.pipeline, "start", "start:pipeline")
    record(rt.news, "start", "start:news")
    record(rt.position_manager, "start", "start:position_manager")
    record(rt.watchdog, "start", "start:watchdog")
    record(rt.decision_service, "start", "start:decision_service")
    for name, component in [("decision_service", rt.decision_service), ("watchdog", rt.watchdog),
                            ("position_manager", rt.position_manager), ("news", rt.news),
                            ("pipeline", rt.pipeline), ("market", rt.market)]:
        record(component, "stop", f"stop:{name}")
    record(rt.broker, "close", "stop:broker")

    await rt.start()
    assert rt.is_running
    assert rt.started_components == ["broker", "kill_switch", "market", "pipeline", "news", "position_manager",
                                     "watchdog", "decision_service", "paper_bridge"]
    assert [o for o in order if o.startswith("start:")] == [
        "start:kill_switch", "start:market", "start:pipeline", "start:news", "start:position_manager",
        "start:watchdog", "start:decision_service"]

    await rt.start()  # second start is a no-op
    await rt.stop()
    assert not rt.is_running
    assert [o for o in order if o.startswith("stop:")] == [
        "stop:decision_service", "stop:watchdog", "stop:position_manager", "stop:news", "stop:pipeline",
        "stop:market", "stop:broker"]
    assert src.closed

    n = len(order)
    await rt.stop()  # idempotent
    assert len(order) == n
    with pytest.raises(RuntimeError):
        await rt.start()


async def test_failure_during_start_stops_started_components_and_reraises():
    src = ScriptedMarketSource()
    rt = await make_runtime(source=src)
    closed: list[str] = []
    original_close = rt.broker.close

    async def close() -> None:
        closed.append("broker")
        await original_close()

    rt.broker.close = close  # type: ignore[method-assign]

    async def boom() -> None:
        raise RuntimeError("news exploded")

    rt.news.start = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="news exploded"):
        await rt.start()

    assert not rt.is_running
    assert rt.started_components == []
    assert src.closed  # market engine (started before news) was stopped
    assert closed == ["broker"]
    assert rt.pipeline._task is None and rt.decision_service._task is None
    assert rt.position_manager._task is None and not rt.watchdog.running
    await rt.stop()  # still idempotent afterwards
    assert closed == ["broker"]


async def test_stop_survives_component_errors_and_timeouts():
    rt = await make_runtime()
    rt.stop_timeout = 0.05
    await rt.start()

    async def hang() -> None:
        await asyncio.sleep(30)

    async def fail() -> None:
        raise RuntimeError("stop failed")

    rt._started = [(n, hang if n == "watchdog" else fail if n == "news" else f) for n, f in rt._started]
    await asyncio.wait_for(rt.stop(), timeout=3)  # never raises, never hangs
    assert not rt.is_running
    # real tasks that were replaced by the stubs must still be cleaned up by hand
    await rt.position_manager.stop()
    await rt.decision_service.stop()
    await rt.pipeline.stop()
    await rt.market.stop()
    await rt.watchdog.stop()
    await rt._stop_paper_bridge()


async def test_no_api_key_means_claude_core_holds_everything():
    rt = await build_runtime(make_settings(), market_source=ScriptedMarketSource(), broker=make_paper(),
                             transport=None, news_sources=[], database_url=MEM_DB)
    try:
        assert rt.api_key_present is False
        assert rt.claude_core._transport is None
    finally:
        await rt.stop()


async def test_news_can_be_disabled():
    rt = await make_runtime(enable_news=False)
    try:
        assert rt.news is None
        await rt.start()
        assert "news" not in rt.started_components
    finally:
        await rt.stop()


async def test_paper_price_bridge_moves_paper_prices(rt, source):
    bids, asks = book(51000.0)
    source.push_order_book(SYM, bids, asks)
    await eventually(lambda: _paper_bid(rt) == Decimal("50999.9"))
    t = await rt.broker.fetch_ticker(SYM)
    assert t.bid == Decimal("50999.9") and t.ask == Decimal("51000.1") and t.mark == Decimal("51000.0")

    bids, asks = book(49000.0)
    source.push_order_book(SYM, bids, asks)
    await eventually(lambda: _paper_bid(rt) == Decimal("48999.9"))


def _paper_bid(rt: TradingRuntime) -> Decimal | None:
    quote = rt.broker._quotes.get(SYM)  # type: ignore[attr-defined]
    return None if quote is None else quote.bid


async def test_paper_bridge_triggers_resting_stop(rt, source):
    await open_long(rt, stop="49500", tp="52000")
    bids, asks = book(49400.0)  # below the stop
    source.push_order_book(SYM, bids, asks)
    await eventually(lambda: _no_positions(rt))


async def _no_positions(rt: TradingRuntime) -> bool:
    return not await rt.broker.fetch_positions()


# ====================================================================== backend: shapes
async def test_backend_static_properties_and_idle_status(rt, backend):
    assert backend.symbols == [SYM]
    assert backend.active_symbol == SYM
    assert backend.timeframe == "1m"
    assert backend.system_status() is SystemStatus.DATA_PAUSED  # no market data yet (gate fails closed)
    health = backend.market_health()
    assert set(health) == {"system_status", "gate", "paused_symbols", "reasons", "channels"}
    assert health["gate"] == "PAUSED" and SYM in health["paused_symbols"]
    assert SYM in health["channels"]


async def test_backend_status_follows_market_gate_and_kill_switch(rt, backend, source):
    await feed_market(source)
    await eventually(lambda: backend.system_status() is SystemStatus.AUTONOMOUS)
    assert backend.market_health()["gate"] == "ACTIVE"
    await rt.kill_switch.engage("test")
    assert backend.system_status() is SystemStatus.HALTED
    assert backend.market_health()["system_status"] == SystemStatus.HALTED.value


async def test_backend_candles_current_candle_and_market_tick(rt, backend, source):
    candles = await backend.candles(SYM, 120)
    assert 0 < len(candles) <= 120
    assert set(candles[0]) == {"time", "open", "high", "low", "close"}
    assert all(isinstance(c["time"], int) for c in candles)
    assert [c["time"] for c in candles] == sorted(c["time"] for c in candles)
    assert len(await backend.candles(SYM, 5)) == 5
    assert await backend.candles("NOPE/USDT:USDT") == []  # unknown symbol never raises
    assert await backend.current_candle(SYM) is None

    next_ts = history_rows()[-1][0] + 60_000
    source.push_ohlcv(SYM, [[next_ts, 50120.0, 50130.0, 50110.0, 50125.0, 3.0]])
    cur = await eventually(lambda: backend.current_candle(SYM))
    assert cur == {"time": next_ts // 1000, "open": 50120.0, "high": 50130.0, "low": 50110.0, "close": 50125.0}
    assert (await backend.candles(SYM, 120))[-1]["time"] == next_ts // 1000  # closed + current

    assert await backend.market_tick(SYM) is None
    await feed_market(source, 50000.0)
    tick = await eventually(lambda: backend.market_tick(SYM))
    assert set(tick) == {"symbol", "bid", "ask", "last", "mark", "spread", "funding_rate"}
    assert tick["bid"] == 49999.9 and tick["ask"] == 50000.1 and tick["last"] == 50000.0
    assert tick["spread"] == pytest.approx(0.2)
    assert tick["funding_rate"] is None  # paper broker has no funding configured


async def test_backend_equity_kpis_shape_and_values(rt, backend):
    kpis = await backend.equity_kpis()
    assert set(kpis) == KPI_KEYS
    assert kpis["equity"] == 10000.0 and kpis["trades_24h"] == 0
    assert kpis["win_rate"] is None and kpis["profit_factor"] is None
    assert kpis["trading_mode"] == "PAPER" and kpis["active_symbol"] == SYM and kpis["exchange"] == "binance"
    assert kpis["kill_switch_engaged"] is False and kpis["kill_switch_reason"] is None
    assert kpis["system_status"] == SystemStatus.DATA_PAUSED.value
    assert kpis["daily_drawdown_limit_pct"] == pytest.approx(0.03)
    assert kpis["claude_model"] == rt.settings.claude.model
    assert kpis["daily_drawdown_pct"] == 0.0 and kpis["equity_change_pct"] == 0.0


async def test_backend_kpis_use_day_start_equity_and_cache(rt):
    mono = [0.0]
    be = RuntimeTerminalBackend(rt, monotonic=lambda: mono[0])
    rt.kill_switch.state.day_start_equity = Decimal("10200")  # equity 10000 -> -1.96 %
    kpis = await be.equity_kpis()
    assert kpis["equity_change_pct"] == pytest.approx(-200 / 10200)
    assert kpis["daily_drawdown_pct"] == pytest.approx(200 / 10200)

    calls = {"n": 0}
    original = rt.broker.fetch_account

    async def counting() -> Any:
        calls["n"] += 1
        return await original()

    rt.broker.fetch_account = counting  # type: ignore[method-assign]
    mono[0] = 100.0
    await be.equity_kpis()
    await be.equity_kpis()
    assert calls["n"] == 1  # second call inside the TTL is served from cache
    mono[0] = 105.0
    await be.equity_kpis()
    assert calls["n"] == 2


async def test_backend_open_position_item(rt, backend):
    mp = await open_long(rt)
    items = await backend.open_positions()
    assert len(items) == 1
    item = items[0]
    assert set(item) == POSITION_KEYS
    assert item["symbol"] == SYM and item["side"] == "LONG" and item["leverage"] == 3
    assert item["quantity"] == pytest.approx(0.1)
    assert item["hard_stop"] == 49000.0 and item["take_profit"] == 52000.0
    assert item["trailing_active"] is False
    assert item["est_fee"] == pytest.approx(0.0005 * (item["entry_price"] + item["mark_price"]) * 0.1)
    assert item["net_pnl"] == pytest.approx(item["unrealized_pnl"] - item["est_fee"])
    assert item["liquidation_price"] is not None
    assert item["opened_at"] == mp.opened_at.isoformat()

    mp.breakeven_locked = True
    mp.current_stop = Decimal("50100")
    item = (await backend.open_positions())[0]
    assert item["trailing_active"] is True and item["hard_stop"] == 50100.0

    kpis = await backend.equity_kpis()
    assert kpis["unrealized_pnl"] == pytest.approx(item["unrealized_pnl"])


async def test_backend_close_position_trade_history_markers_and_kpis(rt, backend):
    await open_long(rt)
    result = await backend.close_position(SYM)
    assert result["ok"] is True and result["status"] == "CLOSED_BY_DECISION" and result["detail"]
    assert await backend.open_positions() == []
    assert rt.position_manager.positions() == ()  # reconcile finalised the trade

    history = await backend.trade_history()
    assert len(history) == 1
    trade = history[0]
    assert set(trade) == TRADE_KEYS
    assert trade["symbol"] == SYM and trade["side"] == "LONG" and trade["exit_reason"] == "MANUAL"
    assert trade["trigger"].startswith("Manuel")
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["fees_paid"] - trade["funding_paid"])
    assert trade["net_pnl"] < 0  # paid spread + slippage + fees

    since = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    markers = await backend.markers(SYM, since)
    assert [m["time"] for m in markers] == sorted(m["time"] for m in markers)
    entry, exit_ = markers[0], markers[-1]
    assert entry["position"] == "belowBar" and entry["shape"] == "arrowUp" and entry["color"] == "#00C805"
    assert exit_["position"] == "aboveBar" and exit_["shape"] == "arrowDown" and exit_["color"] == "#FF3B30"
    assert exit_["text"].startswith("EXIT -$")
    assert await backend.markers(SYM, int(datetime.now(timezone.utc).timestamp()) + 3600) == []
    assert await backend.markers("OTHER/USDT:USDT", since) == []

    kpis = await backend.equity_kpis()
    assert kpis["trades_24h"] == 1 and kpis["fees_paid_24h"] > 0
    assert kpis["realized_pnl_24h"] == pytest.approx(trade["net_pnl"])
    assert kpis["win_rate"] == 0.0 and kpis["profit_factor"] == 0.0


async def test_backend_close_position_without_position_is_ok_but_nothing_to_close(rt, backend):
    result = await backend.close_position(SYM)
    assert result["status"] == "NOTHING_TO_CLOSE" and result["ok"] is True


async def test_backend_claude_logs_mapping(rt, backend):
    ctx = {"payload": {"indicators": {"rsi": 28.44, "obi": 0.31, "atr_pct": 0.0052},
                       "fee_precheck": {"cover_multiple": "4.2"}, "confluence_score": 0.83}}
    async with rt.session_factory() as session:
        await repository.save_decision(
            session, symbol=SYM, model="claude-x", action=DecisionAction.LONG, confidence=Decimal("0.78"),
            reasoning="RSI aşırı satımda. İkinci cümle.", leverage=10, input_context=ctx, latency_ms=812,
            risk_verdict=RiskVerdict.REDUCED, risk_reasons=["LEVERAGE: 10 -> 5"], final_leverage=5)
        await session.commit()
    logs = await backend.claude_logs()
    assert len(logs) == 1
    log = logs[0]
    assert set(log) == CLAUDE_KEYS
    assert log["action"] == "LONG" and log["confidence"] == pytest.approx(0.78) and log["leverage"] == 5
    assert log["metrics_summary"] == "RSI(14): 28.4 | OBI: +0.31 | ATR%: 0.52 | Fee x4.2 | Skor: 0.83"
    assert log["reasoning_tr"].startswith("RSI aşırı") and log["risk_decision"] == "REDUCED"
    assert log["risk_flags"] == ["LEVERAGE: 10 -> 5"] and log["latency_ms"] == 812
    uuid.UUID(log["decision_id"])


def test_metrics_summary_handles_missing_context():
    assert metrics_summary_from_context(None) == ""
    assert metrics_summary_from_context({"payload": "x"}) == ""
    assert metrics_summary_from_context({"payload": {}}) == ""


async def test_backend_news_feed_newest_first(rt, backend):
    now = utcnow()
    await rt.news.ingest([
        RawNewsItem("rss:coindesk", "n1", "Bitcoin ETF sees record inflow", "https://x.test/a", now - timedelta(minutes=10)),
        RawNewsItem("sec", "n2", "SEC approves new ETF rules for bitcoin", "https://x.test/b", now - timedelta(minutes=1)),
        RawNewsItem("rss:coindesk", "n3", "Celebrity chef opens restaurant", "https://x.test/c", now),
    ])
    feed = await backend.news_feed(25)
    assert len(feed) == 2  # irrelevant item filtered by the news engine
    assert set(feed[0]) == NEWS_KEYS
    assert feed[0]["time"] > feed[1]["time"]
    assert feed[0]["sentiment_label"] in {"BULLISH", "BEARISH", "NEUTRAL"} and isinstance(feed[0]["symbols"], list)
    assert len(await backend.news_feed(1)) == 1


async def test_backend_news_feed_empty_when_news_disabled():
    rt = await make_runtime(enable_news=False)
    try:
        assert await fresh_backend(rt).news_feed() == []
    finally:
        await rt.stop()


async def test_backend_read_methods_never_raise_when_db_is_gone(rt, backend):
    await rt.db_engine.dispose()
    rt.session_factory = None  # type: ignore[assignment]  # any DB access now raises TypeError
    assert await backend.trade_history() == []
    assert await backend.markers(SYM, 0) == []
    assert await backend.claude_logs() == []
    kpis = await backend.equity_kpis()
    assert kpis["trades_24h"] == 0  # DB part falls back to an empty summary


async def test_backend_health_components(rt, backend):
    health = await backend.health()
    assert set(health) == {"ok", "components", "uptime_s"}
    comps = health["components"]
    assert set(comps) == {"db", "broker", "market_data", "claude", "kill_switch", "news"}
    assert comps["db"]["ok"] is True and comps["db"]["latency_ms"] >= 0
    assert comps["broker"]["ok"] is True and comps["broker"]["latency_ms"] >= 0
    assert comps["market_data"]["gate"] == "PAUSED" and comps["market_data"]["ok"] is False
    assert comps["claude"]["api_key_present"] is True and comps["claude"]["circuit_open"] is False
    assert isinstance(comps["claude"]["stats"], dict)
    assert comps["kill_switch"] == {"ok": True, "engaged": False, "reason": None}
    assert comps["news"]["enabled"] is True and "fetched" in comps["news"]["stats"]
    assert health["ok"] is False  # market data not yet live
    assert health["uptime_s"] >= 0


# ====================================================================== backend: control
async def test_kill_switch_flattens_and_halts(rt, backend):
    await open_long(rt)
    events = rt.bus.subscribe(Topic.KILL_SWITCH)
    result = await backend.engage_kill_switch("UI PANIC")
    assert result["engaged"] is True and result["system_status"] == SystemStatus.HALTED.value
    assert result["closed"] == [SYM] and result["still_open"] == []
    assert result["reason"] == "UI PANIC"
    assert await rt.broker.fetch_positions() == [] and await rt.broker.fetch_open_orders() == []
    assert rt.position_manager.positions() == ()
    assert backend.system_status() is SystemStatus.HALTED
    assert any(e["engaged"] for e in events.drain())

    kpis = await backend.equity_kpis()
    assert kpis["kill_switch_engaged"] is True and kpis["kill_switch_reason"] == "UI PANIC"
    assert len(await backend.trade_history()) == 1  # the trade was closed in the DB

    again = await backend.engage_kill_switch("second press")  # idempotent
    assert again["engaged"] is True and again["newly_engaged"] is False and again["reason"] == "UI PANIC"


async def test_kill_switch_empty_reason_defaults(rt, backend):
    result = await backend.engage_kill_switch("   ")
    assert result["engaged"] is True and result["reason"] == "MANUAL_UI_TRIGGER"


async def test_resume_requires_exact_reset_confirmation(rt, backend):
    await backend.engage_kill_switch("test")
    bad = await backend.resume("alice", "reset")
    assert bad["ok"] is False and bad["error"] == "RESET_DENIED" and bad["engaged"] is True
    assert rt.kill_switch.is_engaged
    blank = await backend.resume("   ", "RESET")
    assert blank["ok"] is False and rt.kill_switch.is_engaged

    good = await backend.resume("alice", "RESET")
    assert good["ok"] is True and good["engaged"] is False and not rt.kill_switch.is_engaged
    assert backend.system_status() is not SystemStatus.HALTED


async def test_broker_connect_retries_then_succeeds(monkeypatch):
    import app.runtime as runtime_mod

    attempts: list[int] = []
    slept: list[float] = []
    paper = make_paper()

    async def flaky(settings):  # noqa: ANN001, ANN202
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("exchange unreachable")
        return paper

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    monkeypatch.setattr(runtime_mod, "create_broker_async", flaky)
    assert await runtime_mod._connect_broker_with_retry(make_settings(), fake_sleep) is paper  # noqa: SLF001
    assert len(attempts) == 3 and slept == [2.0, 4.0]


async def test_broker_connect_gives_up_and_config_errors_are_not_retried(monkeypatch):
    import app.runtime as runtime_mod
    from app.core.errors import ConfigError

    async def down(settings):  # noqa: ANN001, ANN202
        raise OSError("451 restricted location")

    async def no_sleep(d: float) -> None:
        return None

    monkeypatch.setattr(runtime_mod, "create_broker_async", down)
    with pytest.raises(RuntimeError, match="after 7 attempts"):
        await runtime_mod._connect_broker_with_retry(make_settings(), no_sleep)  # noqa: SLF001

    calls: list[int] = []

    async def bad_config(settings):  # noqa: ANN001, ANN202
        calls.append(1)
        raise ConfigError("bad market")

    monkeypatch.setattr(runtime_mod, "create_broker_async", bad_config)
    with pytest.raises(ConfigError):
        await runtime_mod._connect_broker_with_retry(make_settings(), no_sleep)  # noqa: SLF001
    assert calls == [1]
