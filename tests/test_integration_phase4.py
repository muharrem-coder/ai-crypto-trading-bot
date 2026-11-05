"""End-to-end Phase 4 integration (integrator-owned).

Real FastAPI app (create_app) + real lifespan + real TradingRuntime + real TerminalHub + real index.html route.
Only the outside world is simulated: market WebSocket (ScriptedMarketSource), exchange (PaperBroker),
Claude API (a transport that confirms the signal it is shown), database (in-memory SQLite).

1. /ws/terminal -> first message is a complete INITIAL_SNAPSHOT (fast)
2. claude.decision (Turkish reasoning) + position.opened published on the EventBus reach the browser socket
3. Full chain over the socket: mock market data -> trigger -> Claude -> risk -> paper fill -> POSITION_OPENED,
   then POST /api/v1/kill-switch -> position flattened -> KILL_SWITCH with HALTED on the socket
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.brokers.paper_adapter import PaperBrokerAdapter
from app.claude.transport import tool_response
from app.config.settings import ApiSettings, DataSettings, ExecutionSettings, RiskSettings, Settings
from app.core.events import Topic
from app.data.mock_source import ScriptedMarketSource
from app.risk.testing import make_spec
from app.runtime import build_runtime

pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

SYM = "BTC/USDT:USDT"
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(minutes=130)
STATIC = Path(__file__).resolve().parents[1] / "static"
SNAPSHOT_KEYS = {"equity_kpis", "active_symbol", "symbols", "timeframe", "candles", "markers", "open_positions",
                 "trade_history", "claude_logs", "news_feed", "market_health", "server_time", "partial"}
REASONING = ("RSI aşırı satımda, fiyat Bollinger alt bandının altında ve emir defteri alıcı ağırlıklı. "
             "Komisyon sonrası R:R yeterli; ani satış riski için stop ATR'nin 1.5 katında. Şğüöçı İĞÜÖÇ.")


def _ms(i: int) -> int:
    return int((START + timedelta(minutes=i)).timestamp() * 1000)


def crafted_rows() -> list[list[float]]:
    """120 calm bars, then a sell-off bar (index 120) that produces a mean-reversion LONG trigger."""
    rows, p = [], 100.0
    for i in range(120):
        o, c = p, 100.0 + (0.1 if i % 2 else -0.1)
        rows.append([_ms(i), o, max(o, c) + 0.15, min(o, c) - 0.15, c, 10.0])
        p = c
    for j, drop in enumerate([2.5, 0.2, -0.1, -0.1, -0.1]):
        o, c = p, p - drop
        rows.append([_ms(120 + j), o, max(o, c) + 0.05, min(o, c) - 0.05, c, 10.0])
        p = c
    return rows


def book(mid: float) -> tuple[list[list[float]], list[list[float]]]:
    bids = [[round(mid - 0.002 - i * 0.01, 6), 11.0] for i in range(10)]
    asks = [[round(mid + 0.002 + i * 0.01, 6), 9.0] for i in range(10)]
    return bids, asks


class ConfirmingTransport:
    """Stands in for the Anthropic Messages API: confirms the signal it receives via the forced tool."""

    def __init__(self, leverage: int = 10) -> None:
        self.leverage = leverage
        self.calls: list[dict[str, Any]] = []

    async def create_message(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = kwargs["messages"][0]["content"]
        m = re.search(r"<signal_payload>\s*(\{.*?\})\s*</signal_payload>", text, re.S)
        assert m, "signal payload missing from prompt"
        sig = json.loads(m.group(1))
        return tool_response({
            "action": sig["direction"], "confidence": 0.8, "leverage": self.leverage, "entry_type": "MARKET",
            "entry_price": None, "stop_loss": float(sig["proposed_stop_loss"]),
            "take_profit": float(sig["proposed_take_profit"]), "reasoning_tr": REASONING,
            "risk_flags": ["HIGH_VOLATILITY"],
        })


class Env:
    def __init__(self) -> None:
        self.rows = crafted_rows()
        self.source = ScriptedMarketSource()
        self.source.set_history(SYM, self.rows[:120])
        self.paper = PaperBrokerAdapter({SYM: make_spec(SYM)}, Decimal("10000"), "USDT", Decimal("2"),
                                        now=lambda: NOW)
        self.transport = ConfirmingTransport()
        self.runtime = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.settings = Settings(
            _env_file=None, symbols=[SYM],
            data=DataSettings(_env_file=None, bootstrap_candles=200, stale_after_seconds=60,
                              resume_after_live_seconds=0, health_check_interval_seconds=0.05),
            risk=RiskSettings(_env_file=None, watchdog_interval_seconds=30),
            execution=ExecutionSettings(_env_file=None, manager_poll_seconds=30),
            api=ApiSettings(_env_file=None, control_token="t" * 32),
        )

        async def factory(settings: Settings):  # noqa: ANN202
            self.loop = asyncio.get_running_loop()  # the app's loop (used by the real-browser test)
            self.runtime = await build_runtime(
                settings, market_source=self.source, broker=self.paper, transport=self.transport,
                news_sources=[], database_url="sqlite+aiosqlite:///:memory:", clock=lambda: NOW,
            )
            return self.runtime

        self.app = create_app(settings=self.settings, runtime_factory=factory, static_dir=STATIC)
        self.client = TestClient(self.app)

    def call(self, fn: Any, *args: Any) -> Any:
        return self.client.portal.call(fn, *args)

    def publish(self, topic: str, payload: Any) -> None:
        self.call(self.runtime.bus.publish, topic, payload)

    def quotes(self, mid: float) -> None:
        bids, asks = book(mid)

        def push() -> None:
            self.source.push_order_book(SYM, bids, asks)
            self.source.push_ticker(SYM, bids[0][0], asks[0][0], mid)
        self.call(push)

    def push_bar(self, i: int) -> None:
        self.call(self.source.push_ohlcv, SYM, [self.rows[i]])


def collect_until(ws: Any, want: str, *, predicate=lambda m: True, timeout: float = 8.0) -> tuple[dict, list]:
    """Read messages using PING/PONG barriers (never blocks forever) until a `want` message matches."""
    seen: list[dict] = []
    deadline = time.monotonic() + timeout
    n = 0
    while time.monotonic() < deadline:
        n += 1
        ws.send_json({"type": "PING", "t": n})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "PONG" and msg["data"]["t"] == n:
                break
            seen.append(msg)
            if msg["type"] == want and predicate(msg):
                return msg, seen
        time.sleep(0.05)
    raise AssertionError(f"{want} not received; got {[m['type'] for m in seen]}")


@pytest.fixture
def env():
    e = Env()
    with e.client:
        yield e


def test_index_page_is_the_users_terminal(env: Env):
    r = env.client.get("/")
    assert r.status_code == 200 and "QUANT<span class=\"text-textMuted\">//</span>CLAUDE" in r.text
    assert "/ws/terminal" in r.text and "lightweight-charts@4.1.3" in r.text
    assert "US EQUITIES (NASDAQ/NYSE)" in r.text and "setInterval(() => {\n      if (!systemActive)" not in r.text


def test_ws_initial_snapshot_complete_and_fast(env: Env):
    t0 = time.perf_counter()
    with env.client.websocket_connect("/ws/terminal") as ws:
        first = ws.receive_json()
        elapsed = time.perf_counter() - t0
    assert first["type"] == "INITIAL_SNAPSHOT" and first["seq"] == 1
    data = first["data"]
    assert SNAPSHOT_KEYS <= set(data), SNAPSHOT_KEYS - set(data)
    assert data["partial"] == [], data["partial"]
    assert data["active_symbol"] == SYM and data["symbols"] == [SYM]
    assert 100 <= len(data["candles"]) <= 120
    c = data["candles"][-1]
    assert set(c) >= {"time", "open", "high", "low", "close"} and isinstance(c["time"], int)
    kpis = data["equity_kpis"]
    assert kpis["equity"] == pytest.approx(10000) and kpis["system_status"] in {"AUTONOMOUS 24/7", "DATA_PAUSED"}
    assert data["open_positions"] == [] and data["trade_history"] == []
    assert elapsed < 0.5  # wall time incl. TestClient thread hop; the hub itself has an 80 ms build deadline


def test_bus_events_reach_the_socket_with_turkish_text(env: Env):
    with env.client.websocket_connect("/ws/terminal") as ws:
        assert ws.receive_json()["type"] == "INITIAL_SNAPSHOT"
        env.publish(Topic.CLAUDE_DECISION, {
            "decision_id": "d-1", "symbol": SYM, "action": "LONG", "confidence": "0.81", "leverage": 3,
            "reasoning_tr": REASONING, "risk_flags": ["NEWS_CONFLICT"], "fallback": None, "latency_ms": 1200,
            "signal_id": "s-1", "created_at": NOW.isoformat(), "timestamp": NOW.isoformat(),
            "metrics_summary": "RSI(14): 27.9 | OBI: +0.10", "risk_decision": None, "execution_status": None,
        })
        env.publish(Topic.POSITION_OPENED, {
            "trade_id": "t-1", "symbol": SYM, "side": "LONG", "leverage": 3, "quantity": 0.5, "entry_price": 97.5,
            "hard_stop": 96.8, "take_profit": 99.0, "decision_id": "d-1", "opened_at": NOW.isoformat(),
        })
        decision, seen = collect_until(ws, "CLAUDE_DECISION")
        assert decision["data"]["reasoning_tr"] == REASONING and decision["data"]["confidence"] == pytest.approx(0.81)
        opened, more = collect_until(ws, "POSITION_OPENED")
        assert opened["data"]["symbol"] == SYM and opened["data"]["hard_stop"] == pytest.approx(96.8)
        seqs = [m["seq"] for m in seen + more]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_full_chain_over_socket_then_kill_switch_halts_everything(env: Env):
    rt = env.runtime
    with env.client.websocket_connect("/ws/terminal") as ws:
        assert ws.receive_json()["type"] == "INITIAL_SNAPSHOT"
        sell_close = env.rows[120][4]
        env.quotes(sell_close)
        collect_until(ws, "MARKET_HEALTH", predicate=lambda m: m["data"].get("system_status") == "AUTONOMOUS 24/7")

        env.push_bar(120)  # the sell-off bar forms ...
        env.push_bar(121)  # ... and closes -> indicators -> pre-filter -> CLAUDE_TRIGGER -> decision service
        decision, seen = collect_until(ws, "CLAUDE_DECISION",
                                       predicate=lambda m: m["data"].get("execution_status") == "OPENED_PROTECTED")
        d = decision["data"]
        assert d["action"] == "LONG" and d["reasoning_tr"] == REASONING and d["risk_decision"] == "REDUCED"
        assert "RSI(14)" in d["metrics_summary"]
        # position.opened is published while the order is executing, i.e. before the decision's update message
        opened = next((m for m in seen if m["type"] == "POSITION_OPENED"), None) or collect_until(ws, "POSITION_OPENED")[0]
        assert [m["type"] for m in seen].count("CLAUDE_DECISION") >= 1  # first decision message (pre-execution)
        assert opened["data"]["leverage"] == 5  # Claude asked for 10x; the hard-risk engine clamped it
        positions = env.call(env.paper.fetch_positions)
        assert len(positions) == 1 and positions[0].leverage == 5
        pnl, _ = collect_until(ws, "POSITIONS_PNL", predicate=lambda m: len(m["data"]["positions"]) == 1)
        row = pnl["data"]["positions"][0]
        assert row["hard_stop"] is not None and row["take_profit"] is not None

        # emergency stop from the UI button
        r = env.client.post("/api/v1/kill-switch", json={"reason": "MANUAL_UI_TRIGGER"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["system_status"] == "HALTED" and body["engaged"] is True and body["still_open"] == []
        assert env.call(env.paper.fetch_positions) == []
        assert env.call(env.paper.fetch_open_orders, SYM) == []
        halted, _ = collect_until(ws, "KILL_SWITCH", predicate=lambda m: m["data"].get("system_status") == "HALTED")
        assert halted["data"]["engaged"] is True
        kpis, _ = collect_until(ws, "EQUITY_KPIS", predicate=lambda m: m["data"]["system_status"] == "HALTED")
        assert kpis["data"]["kill_switch_engaged"] is True

        # nothing new can open while halted: the next trigger is refused by the risk layer
        assert rt.kill_switch.is_engaged

        # resume needs the control token and the RESET confirmation
        assert env.client.post("/api/v1/kill-switch/resume",
                               json={"operator": "muharrem", "confirm": "RESET"}).status_code == 401
        r = env.client.post("/api/v1/kill-switch/resume", json={"operator": "muharrem", "confirm": "RESET"},
                            headers={"X-Control-Token": "t" * 32})
        assert r.status_code == 200 and not rt.kill_switch.is_engaged


def test_health_endpoint_reports_components(env: Env):
    r = env.client.get("/api/v1/health")
    assert r.status_code == 200
    comps = r.json()["components"]
    assert {"db", "broker", "market_data", "claude", "kill_switch"} <= set(comps)
    assert comps["db"]["ok"] is True and comps["broker"]["ok"] is True
