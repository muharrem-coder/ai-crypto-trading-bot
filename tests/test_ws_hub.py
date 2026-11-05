"""Tests for the /ws/terminal hub (app.api.ws_hub) using FastAPI + TestClient and an in-loop fake websocket."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.protocol import MessageType, SystemStatus
from app.api.testing import BASE_TIME, FakeTerminalBackend
from app.api.ws_hub import (TerminalHub, claude_log_item, envelope, execution_report_dict, news_item,
                            origin_allowed, to_jsonable)
from app.config.settings import ApiSettings
from app.core.events import EventBus, Topic
from app.core.types import Order, OrderStatus, OrderType, PositionSide, Side, Ticker
from app.data.models import DataChannel, FeedHealth, FeedHealthEvent, GateState, TradingGateEvent
from app.execution.contracts import ExecutionReport, ExecutionStatus
from app.news.contracts import NewsEvent

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
TOPICS = [Topic.CLAUDE_DECISION, Topic.EXECUTION_REPORT, Topic.POSITION_OPENED, Topic.POSITION_UPDATED,
          Topic.POSITION_CLOSED, Topic.KILL_SWITCH, Topic.RISK_EVENT, Topic.NEWS, Topic.FEED_HEALTH,
          Topic.TRADING_GATE, Topic.TICKER, Topic.ORDER_BOOK, Topic.CANDLE_UPDATE, Topic.CANDLE_CLOSED]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_news_event(i: int = 1) -> NewsEvent:
    return NewsEvent(
        news_id=f"n{i}", source="rss:coindesk", external_id=None, title=f"Fed faizi sabit tuttu {i}",
        url="https://example.test/n", published_at=NOW, ingested_at=NOW, symbols=("BTC",), macro_tags=("FED",),
        sentiment_score=Decimal("0.35"), sentiment_label="BULLISH", importance=7, duplicate_sources=("sec",))


def make_ticker(symbol: str = BTC, last: float = 100.0) -> Ticker:
    return Ticker(symbol=symbol, bid=Decimal(str(last - 0.1)), ask=Decimal(str(last + 0.1)), last=Decimal(str(last)),
                  mark=Decimal(str(last)), timestamp=NOW)


def decision_payload(update: bool = False) -> dict[str, Any]:
    payload = {
        "decision_id": "dec-77", "symbol": BTC, "action": "OPEN_LONG", "confidence": "0.82", "leverage": 3,
        "reasoning_tr": "Momentum güçlü; stop ATR altında. İşlem açılıyor: şğüöçı İĞÜÖÇİ",
        "risk_flags": ["HIGH_FUNDING"], "fallback": None, "latency_ms": 910, "signal_id": "sig-1",
        "created_at": NOW.isoformat(), "timestamp": NOW.isoformat(), "metrics_summary": "RSI(14): 28.4",
        "risk_decision": None, "execution_status": None,
    }
    if update:
        payload.update({"risk_decision": "APPROVED", "execution_status": "OPENED_PROTECTED", "update": True})
    return payload


# ------------------------------------------------------------------------------------- TestClient harness
class Harness:
    def __init__(self, backend: FakeTerminalBackend | None = None, *, tick_interval: float = 3600.0,
                 **settings_kw: Any) -> None:
        self.bus = EventBus()
        self.backend = backend or FakeTerminalBackend()
        self.clock = FakeClock()
        self.settings = ApiSettings(**settings_kw)
        self.hub = TerminalHub(self.bus, self.backend, self.settings, clock=self.clock, tick_interval=tick_interval)
        hub = self.hub

        async def lifespan(_app: FastAPI):  # noqa: ANN202
            await hub.start()
            yield
            await hub.stop()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.websocket("/ws/terminal")
        async def terminal(websocket: WebSocket) -> None:
            await hub.serve(websocket)

        self.client = TestClient(self.app)

    def __enter__(self) -> Harness:
        self.client.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.__exit__(*exc)

    def connect(self, **kw: Any):  # noqa: ANN201
        return self.client.websocket_connect("/ws/terminal", **kw)

    def call(self, fn: Any, *args: Any) -> Any:
        return self.client.portal.call(fn, *args)

    def publish(self, *events: tuple[str, Any]) -> None:
        def _publish_all() -> None:
            for topic, payload in events:
                self.bus.publish(topic, payload)
        self.call(_publish_all)

    def settle(self) -> None:
        async def _settle() -> None:
            for _ in range(10):
                await asyncio.sleep(0)
        self.call(_settle)

    def tick(self, advance: float = 0.0) -> None:
        self.clock.advance(advance)
        self.settle()
        self.call(self.hub.tick_once)


def barrier(ws: Any, n: int = 1) -> list[dict[str, Any]]:
    """Send PING and return every message that arrived before the PONG (ordering barrier)."""
    ws.send_json({"type": "PING", "t": n})
    out = []
    while True:
        msg = ws.receive_json()
        if msg["type"] == "PONG" and msg["data"]["t"] == n:
            return out
        out.append(msg)


def of_type(msgs: list[dict[str, Any]], t: str) -> list[dict[str, Any]]:
    return [m for m in msgs if m["type"] == t]


# ================================================================================================ snapshot
def test_first_message_is_full_initial_snapshot() -> None:
    with Harness() as h, h.connect() as ws:
        snap = ws.receive_json()
        assert snap["type"] == "INITIAL_SNAPSHOT"
        assert snap["seq"] == 1
        datetime.fromisoformat(snap["ts"])
        d = snap["data"]
        assert set(d) == {"equity_kpis", "active_symbol", "symbols", "timeframe", "candles", "markers",
                          "open_positions", "trade_history", "claude_logs", "news_feed", "market_health",
                          "server_time", "partial"}
        assert d["partial"] == []
        assert d["active_symbol"] == BTC and d["symbols"][0] == BTC and d["timeframe"] == "1m"
        assert len(d["candles"]) == 120
        assert len(d["trade_history"]) == 50
        assert len(d["claude_logs"]) == 25
        assert len(d["news_feed"]) == 25
        assert d["equity_kpis"]["equity"] == 10_000.0
        assert d["open_positions"][0]["symbol"] == BTC
        assert d["market_health"]["gate"] == "ACTIVE"
        datetime.fromisoformat(d["server_time"])
        # markers are requested from the first candle time of the snapshot
        first_time = d["candles"][0]["time"]
        assert first_time == BASE_TIME + 80 * 60
        assert d["markers"] and all(m["time"] >= first_time for m in d["markers"])
        args = dict((name, a) for name, a in h.backend.call_args)
        assert args["candles"] == (BTC, 120)
        assert args["markers"] == (BTC, first_time)
        assert args["trade_history"] == (50,) and args["claude_logs"] == (25,) and args["news_feed"] == (25,)


def test_snapshot_not_delayed_by_slow_backend_part() -> None:
    backend = FakeTerminalBackend()
    backend.delays["claude_logs"] = 1.0
    with Harness(backend, ws_snapshot_deadline_seconds=0.05) as h, h.connect() as ws:
        started = time.monotonic()  # the websocket is accepted at this point
        snap = ws.receive_json()
        elapsed = time.monotonic() - started
        assert elapsed < 0.1, elapsed
        assert snap["type"] == "INITIAL_SNAPSHOT"
        assert snap["data"]["partial"] == ["claude_logs"]
        assert snap["data"]["claude_logs"] == []
        assert len(snap["data"]["candles"]) == 120 and len(snap["data"]["news_feed"]) == 25


def test_snapshot_failed_parts_use_defaults_and_markers_depend_on_candles() -> None:
    backend = FakeTerminalBackend()
    backend.errors["candles"] = RuntimeError("db down")
    backend.errors["news_feed"] = RuntimeError("boom")
    backend.errors["market_health"] = RuntimeError("boom")
    backend.errors["equity_kpis"] = RuntimeError("boom")
    with Harness(backend) as h, h.connect() as ws:
        d = ws.receive_json()["data"]
        assert d["partial"] == ["equity_kpis", "candles", "markers", "news_feed", "market_health"]
        assert d["equity_kpis"] == {} and d["candles"] == [] and d["markers"] == [] and d["news_feed"] == []
        assert d["market_health"] is None
        assert len(d["trade_history"]) == 50
        assert "markers" not in {n for n, _ in backend.call_args}
        assert barrier(ws) == []  # connection stays healthy


def test_event_published_during_snapshot_is_delivered_after_it() -> None:
    backend = FakeTerminalBackend()
    backend.delays["equity_kpis"] = 0.02
    with Harness(backend) as h:
        backend.on_call["equity_kpis"] = lambda: h.bus.publish(Topic.NEWS, make_news_event(5))
        with h.connect() as ws:
            first = ws.receive_json()
            second = ws.receive_json()
            assert first["type"] == "INITIAL_SNAPSHOT" and first["seq"] == 1
            assert second["type"] == "NEWS" and second["seq"] == 2
            assert second["data"]["news_id"] == "n5"
            assert first["data"]["partial"] == []


# ================================================================================================ live events
def test_live_events_arrive_in_order_with_increasing_seq() -> None:
    order = Order(order_id="o1", client_order_id=None, symbol=BTC, side=Side.BUY, order_type=OrderType.MARKET,
                  status=OrderStatus.FILLED, quantity=Decimal("0.01"), filled_quantity=Decimal("0.01"), price=None,
                  average_fill_price=Decimal("60000.5"), stop_price=None, reduce_only=False, fee_paid=Decimal("0.3"),
                  fee_currency="USDT", created_at=NOW, updated_at=NOW)
    report = ExecutionReport(status=ExecutionStatus.OPENED_PROTECTED, symbol=BTC, decision_id="dec-77",
                             signal_id="sig-1", detail="Giriş doldu", side=PositionSide.LONG,
                             approved_quantity=Decimal("0.01"), filled_quantity=Decimal("0.01"),
                             entry_price=Decimal("60000.5"), orders=[order], timestamp=NOW)
    position = {"symbol": BTC, "side": "LONG", "quantity": 0.01, "entry_price": 60000.5, "hard_stop": 59400.0}
    closed = {"trade_id": "t1", "symbol": BTC, "exit_price": 60300.0, "exit_reason": "TAKE_PROFIT", "net_pnl": 2.5}
    with Harness() as h, h.connect() as ws:
        assert ws.receive_json()["type"] == "INITIAL_SNAPSHOT"
        h.backend.status = SystemStatus.HALTED
        h.publish(
            (Topic.CLAUDE_DECISION, decision_payload()),
            (Topic.EXECUTION_REPORT, report),
            (Topic.POSITION_OPENED, position),
            (Topic.POSITION_UPDATED, {**position, "hard_stop": 60000.5, "trailing_active": True}),
            (Topic.POSITION_CLOSED, closed),
            (Topic.KILL_SWITCH, {"engaged": True, "code": "KILL_SWITCH_ENGAGED", "message": "m", "reason": "manual"}),
            (Topic.NEWS, make_news_event()),
            (Topic.RISK_EVENT, {"level": "WARNING", "code": "DD", "message": "drawdown", "details": {"x": 1}}),
        )
        msgs = [ws.receive_json() for _ in range(8)]
        assert [m["type"] for m in msgs] == ["CLAUDE_DECISION", "EXECUTION_REPORT", "POSITION_OPENED",
                                             "POSITION_UPDATED", "POSITION_CLOSED", "KILL_SWITCH", "NEWS",
                                             "RISK_EVENT"]
        assert [m["seq"] for m in msgs] == list(range(2, 10))

        dec = msgs[0]["data"]
        assert dec["confidence"] == 0.82 and isinstance(dec["confidence"], float)
        assert dec["reasoning_tr"].startswith("Momentum güçlü") and "şğüöçı İĞÜÖÇİ" in dec["reasoning_tr"]
        assert dec["timestamp"] == NOW.isoformat() and dec["update"] is False and dec["decision_id"] == "dec-77"

        ex = msgs[1]["data"]
        assert ex["status"] == "OPENED_PROTECTED" and ex["side"] == "LONG"
        assert ex["approved_quantity"] == 0.01 and ex["entry_price"] == 60000.5
        assert ex["timestamp"] == NOW.isoformat()
        assert ex["orders"][0]["order_type"] == "MARKET" and ex["orders"][0]["status"] == "FILLED"
        assert ex["orders"][0]["average_fill_price"] == 60000.5

        assert msgs[2]["data"] == position
        assert msgs[3]["data"]["trailing_active"] is True
        assert msgs[4]["data"] == {"symbol": BTC, "trade": closed}
        ks = msgs[5]["data"]
        assert ks["engaged"] is True and ks["system_status"] == "HALTED" and ks["reason"] == "manual"
        news = msgs[6]["data"]
        assert news == {"news_id": "n1", "time": NOW.isoformat(), "source": "rss:coindesk",
                        "title": "Fed faizi sabit tuttu 1", "url": "https://example.test/n", "symbols": ["BTC"],
                        "macro_tags": ["FED"], "sentiment_score": 0.35, "sentiment_label": "BULLISH",
                        "importance": 7, "duplicate_sources": ["sec"]}
        assert msgs[7]["data"] == {"level": "WARNING", "code": "DD", "message": "drawdown"}


def test_decision_update_is_flagged_for_upsert() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.publish((Topic.CLAUDE_DECISION, decision_payload()), (Topic.CLAUDE_DECISION, decision_payload(update=True)))
        first, second = ws.receive_json(), ws.receive_json()
        assert first["data"]["update"] is False and first["data"]["risk_decision"] is None
        assert second["data"]["update"] is True and second["data"]["risk_decision"] == "APPROVED"
        assert second["data"]["execution_status"] == "OPENED_PROTECTED"
        assert first["data"]["decision_id"] == second["data"]["decision_id"]


def test_market_health_only_sent_when_changed() -> None:
    feed = FeedHealthEvent(symbol=BTC, channel=DataChannel.ORDER_BOOK, health=FeedHealth.STALE, age_seconds=3.0,
                           reason="stale", timestamp=NOW)
    gate = TradingGateEvent(state=GateState.PAUSED, reasons=("BTC stale",), timestamp=NOW, paused_symbols=(BTC,))
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.publish((Topic.FEED_HEALTH, feed), (Topic.TRADING_GATE, gate))  # backend health unchanged
        assert barrier(ws, 1) == []
        h.backend.health.update({"gate": "PAUSED", "paused_symbols": [BTC], "reasons": ["BTC stale"]})
        h.publish((Topic.FEED_HEALTH, feed), (Topic.TRADING_GATE, gate))
        msgs = barrier(ws, 2)
        assert [m["type"] for m in msgs] == ["MARKET_HEALTH"]
        assert msgs[0]["data"]["gate"] == "PAUSED" and msgs[0]["data"]["paused_symbols"] == [BTC]
        h.publish((Topic.TRADING_GATE, gate))  # duplicate -> not resent
        assert barrier(ws, 3) == []


def test_broadcast_api() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.call(h.hub.broadcast, MessageType.RISK_EVENT, {"level": "INFO", "code": "X", "message": "Merhaba ğ"})
        msg = ws.receive_json()
        assert msg["type"] == "RISK_EVENT" and msg["data"]["message"] == "Merhaba ğ"


# ================================================================================================ throttling
def test_thousand_tickers_in_one_second_yield_at_most_five_ticks() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.tick()  # first cycle: initial tick/candle/positions for the new client
        first = barrier(ws, 1)
        assert len(of_type(first, "MARKET_TICK")) == 1
        assert len(of_type(first, "CANDLE_UPDATE")) == 1
        assert len(of_type(first, "POSITIONS_PNL")) == 1
        assert of_type(first, "EQUITY_KPIS") == []  # identical to the snapshot value
        before = h.backend.calls["market_tick"]

        async def storm() -> None:
            for i in range(1000):
                h.backend.ticks[BTC]["last"] = float(i)  # type: ignore[index]
                h.bus.publish(Topic.TICKER, make_ticker(BTC, 100.0 + i))
                h.clock.advance(0.001)
                if (i + 1) % 200 == 0:  # fake 5 Hz ticker
                    for _ in range(3):
                        await asyncio.sleep(0)
                    await h.hub.tick_once()

        h.call(storm)
        msgs = barrier(ws, 2)
        ticks = of_type(msgs, "MARKET_TICK")
        assert 1 <= len(ticks) <= 5
        assert ticks[-1]["data"]["last"] == 999.0  # latest value wins
        # computed once per tick, not once per event
        assert h.backend.calls["market_tick"] - before == 5
        seqs = [m["seq"] for m in msgs]
        assert seqs == sorted(set(seqs))


def test_ticks_coalesce_and_unchanged_values_are_not_resent() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.tick()
        barrier(ws, 1)
        for last in (1.0, 2.0, 3.0):
            h.backend.ticks[BTC]["last"] = last  # type: ignore[index]
            h.publish((Topic.TICKER, make_ticker(BTC, last)))
        h.tick(0.2)
        msgs = barrier(ws, 2)
        assert [m["data"]["last"] for m in of_type(msgs, "MARKET_TICK")] == [3.0]

        h.publish((Topic.TICKER, make_ticker(BTC, 3.0)), (Topic.ORDER_BOOK, type("B", (), {"symbol": BTC})()))
        h.tick(0.2)
        assert barrier(ws, 3) == []  # dirty but same value -> nothing sent

        h.backend.current[BTC]["close"] = 123.0  # type: ignore[index]
        h.publish((Topic.CANDLE_UPDATE, type("C", (), {"symbol": BTC})()))
        h.tick(0.2)
        msgs = barrier(ws, 4)
        assert [m["type"] for m in msgs] == ["CANDLE_UPDATE"]
        assert msgs[0]["data"] == {"symbol": BTC, "candle": {**h.backend.current[BTC]}}  # type: ignore[dict-item]


def test_equity_and_positions_refresh_every_second_only_when_changed() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.tick()
        barrier(ws, 1)
        calls = h.backend.calls["equity_kpis"]
        h.backend.kpis["equity"] = 10_100.0
        h.tick(0.4)  # not due, nothing dirty
        assert barrier(ws, 2) == [] and h.backend.calls["equity_kpis"] == calls
        h.tick(0.7)  # >= 1 s since the last slow cycle
        msgs = barrier(ws, 3)
        assert [m["type"] for m in msgs] == ["EQUITY_KPIS"] and msgs[0]["data"]["equity"] == 10_100.0
        h.backend.positions[0]["unrealized_pnl"] = 50.0
        h.publish((Topic.POSITION_UPDATED, {"symbol": BTC, "hard_stop": 1.0}))  # dirty -> immediate refresh
        h.tick(0.1)
        msgs = barrier(ws, 4)
        pnl = of_type(msgs, "POSITIONS_PNL")
        assert len(pnl) == 1 and pnl[0]["data"]["unrealized_total"] == 50.0
        assert pnl[0]["data"]["positions"][0]["symbol"] == BTC


def test_backend_error_in_tick_does_not_break_hub() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.backend.errors["market_tick"] = RuntimeError("boom")
        h.tick()
        msgs = barrier(ws, 1)
        assert of_type(msgs, "MARKET_TICK") == [] and len(of_type(msgs, "POSITIONS_PNL")) == 1
        del h.backend.errors["market_tick"]
        h.publish((Topic.TICKER, make_ticker(BTC)))
        h.tick(0.2)
        assert len(of_type(barrier(ws, 2), "MARKET_TICK")) == 1


# ================================================================================================ client messages
def test_ping_pong() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        ws.send_json({"type": "PING", "t": 1234567})
        msg = ws.receive_json()
        assert msg["type"] == "PONG" and msg["seq"] == 2
        assert msg["data"]["t"] == 1234567
        datetime.fromisoformat(msg["data"]["server_ts"])


def test_subscribe_symbol_sends_candles_and_switches_tick_symbol() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        h.tick()
        barrier(ws, 1)
        ws.send_json({"type": "SUBSCRIBE_SYMBOL", "symbol": ETH})
        msg = ws.receive_json()
        assert msg["type"] == "CANDLES"
        assert msg["data"]["symbol"] == ETH and len(msg["data"]["candles"]) == 120
        assert msg["data"]["markers"] == h.backend.marker_data[ETH]
        assert ("candles", (ETH, 120)) in h.backend.call_args
        # BTC activity no longer reaches this client; ETH does
        h.publish((Topic.TICKER, make_ticker(BTC)), (Topic.TICKER, make_ticker(ETH)))
        h.backend.ticks[ETH]["last"] = 77.0  # type: ignore[index]
        h.tick(0.2)
        ticks = of_type(barrier(ws, 2), "MARKET_TICK")
        assert [t["data"]["symbol"] for t in ticks] == [ETH]


def test_subscribe_unknown_symbol_and_bad_messages_keep_connection_open() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        ws.send_json({"type": "SUBSCRIBE_SYMBOL", "symbol": "DOGE/USDT:USDT"})
        assert ws.receive_json()["type"] == "ERROR"
        ws.send_text("{this is not json")
        err = ws.receive_json()
        assert err["type"] == "ERROR" and err["data"]["code"] == "INVALID_JSON"
        ws.send_text("[1, 2]")
        assert ws.receive_json()["data"]["code"] == "BAD_MESSAGE"
        ws.send_json({"type": "DANCE"})
        assert ws.receive_json()["data"]["code"] == "UNKNOWN_TYPE"
        ws.send_json({"type": "SUBSCRIBE_SYMBOL"})
        assert ws.receive_json()["data"]["code"] == "UNKNOWN_SYMBOL"
        assert barrier(ws, 9) == []  # still alive
        assert h.hub.client_count == 1


def test_oversized_client_message_closes_1009() -> None:
    with Harness() as h, h.connect() as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "PING", "t": "x" * 5000}))
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1009


# ================================================================================================ clients / lifecycle
def test_two_clients_get_their_own_snapshot_and_seq() -> None:
    with Harness() as h:
        with h.connect() as a:
            assert a.receive_json()["seq"] == 1
            h.publish((Topic.NEWS, make_news_event(1)))
            assert a.receive_json()["seq"] == 2
            with h.connect() as b:
                snap_b = b.receive_json()
                assert snap_b["type"] == "INITIAL_SNAPSHOT" and snap_b["seq"] == 1
                assert h.hub.client_count == 2
                h.publish((Topic.NEWS, make_news_event(2)))
                ma, mb = a.receive_json(), b.receive_json()
                assert (ma["seq"], mb["seq"]) == (3, 2)
                a.send_json({"type": "SUBSCRIBE_SYMBOL", "symbol": ETH})
                assert a.receive_json()["type"] == "CANDLES"
                assert barrier(b, 1) == []  # b is unaffected
        h.settle()
        assert h.hub.client_count == 0


def test_max_clients_enforced_with_1013() -> None:
    with Harness(ws_max_clients=1) as h, h.connect() as first:
        assert first.receive_json()["type"] == "INITIAL_SNAPSHOT"
        with pytest.raises(WebSocketDisconnect) as exc:
            with h.connect() as second:
                second.receive_json()
        assert exc.value.code == 1013
        assert h.hub.client_count == 1
        assert barrier(first, 1) == []


def test_bad_origin_rejected_and_good_origins_accepted() -> None:
    with Harness() as h:
        with pytest.raises(WebSocketDisconnect) as exc:
            with h.connect(headers={"origin": "http://evil.example"}) as ws:
                ws.receive_json()
        assert exc.value.code == 1008
        with h.connect(headers={"origin": "http://localhost:8000"}) as ws:  # allowed_origins
            assert ws.receive_json()["type"] == "INITIAL_SNAPSHOT"
        with h.connect(headers={"origin": "http://testserver"}) as ws:  # same host as the Host header
            assert ws.receive_json()["type"] == "INITIAL_SNAPSHOT"
        with h.connect() as ws:  # no Origin header (non-browser client)
            assert ws.receive_json()["type"] == "INITIAL_SNAPSHOT"


def test_hub_stop_closes_clients_with_1001_and_unsubscribes() -> None:
    with Harness() as h:
        assert all(h.bus.subscriber_count(t) == 1 for t in TOPICS)  # subscribed once, not per client
        with h.connect() as a, h.connect() as b:
            a.receive_json()
            b.receive_json()
            assert h.bus.subscriber_count(Topic.TICKER) == 1
            h.call(h.hub.stop)
            for ws in (a, b):
                with pytest.raises(WebSocketDisconnect) as exc:
                    ws.receive_json()
                assert exc.value.code == 1001
        assert h.hub.client_count == 0
        assert all(h.bus.subscriber_count(t) == 0 for t in TOPICS)
        assert not h.hub.running
        h.call(h.hub.stop)  # idempotent


# ================================================================================ in-loop fake websocket tests
class FakeWebSocket:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = {"host": "testserver", **(headers or {})}
        self.client = ("127.0.0.1", 9)
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.accepted = False
        self.close_code: int | None = None
        self.block_sends = False
        self._unblock = asyncio.Event()

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, Any]:
        return await self.incoming.get()

    async def send_text(self, text: str) -> None:
        if self.block_sends:
            await self._unblock.wait()
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code

    def feed(self, text: str) -> None:
        self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    def disconnect(self) -> None:
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})


async def wait_until(pred: Any, timeout: float = 3.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


async def spin(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def started_hub(**settings_kw: Any) -> tuple[TerminalHub, EventBus, FakeTerminalBackend]:
    bus = EventBus()
    backend = FakeTerminalBackend()
    hub = TerminalHub(bus, backend, ApiSettings(**settings_kw), clock=FakeClock(), tick_interval=3600)
    await hub.start()
    return hub, bus, backend


async def test_slow_consumer_is_dropped_without_affecting_other_client() -> None:
    hub, bus, _backend = await started_hub(ws_send_queue=50)
    slow, fast = FakeWebSocket(), FakeWebSocket()
    t_slow = asyncio.create_task(hub.serve(slow))
    t_fast = asyncio.create_task(hub.serve(fast))
    try:
        await wait_until(lambda: slow.sent and fast.sent)
        slow.block_sends = True  # browser stops reading
        for i in range(200):
            bus.publish(Topic.POSITION_UPDATED, {"symbol": BTC, "hard_stop": float(i)})
            await spin(4)
        await wait_until(lambda: slow.close_code is not None)
        assert slow.close_code == 1013
        await asyncio.wait_for(t_slow, 2)
        assert hub.client_count == 1 and hub.stats["slow_closed"] >= 1
        stops = [m["data"]["hard_stop"] for m in fast.sent if m["type"] == "POSITION_UPDATED"]
        assert stops == [float(i) for i in range(200)]  # fast client got everything, in order
        assert fast.close_code is None
        assert [m["seq"] for m in fast.sent] == list(range(1, len(fast.sent) + 1))
    finally:
        fast.disconnect()
        await asyncio.wait_for(t_fast, 2)
        await hub.stop()


async def test_full_queue_drops_oldest_throttled_before_closing_client() -> None:
    hub, _bus, _backend = await started_hub(ws_send_queue=50)
    ws = FakeWebSocket()
    task = asyncio.create_task(hub.serve(ws))
    try:
        await wait_until(lambda: ws.sent)
        ws.block_sends = True
        hub.broadcast(MessageType.RISK_EVENT, {"level": "INFO", "code": "first", "message": "m"})
        await wait_until(lambda: hub._clients and next(iter(hub._clients)).queue.empty())  # writer is now stuck
        for t in (MessageType.MARKET_TICK, MessageType.CANDLE_UPDATE, MessageType.POSITIONS_PNL,
                  MessageType.EQUITY_KPIS):
            hub.broadcast(t, {"v": 1})
        for i in range(46):
            hub.broadcast(MessageType.RISK_EVENT, {"level": "INFO", "code": str(i), "message": "m"})
        client = next(iter(hub._clients))
        assert client.queue.full()
        for k in range(4):  # each new critical message evicts one throttled message
            hub.broadcast(MessageType.RISK_EVENT, {"level": "INFO", "code": f"x{k}", "message": "m"})
            assert hub.client_count == 1 and hub.stats["dropped_throttled"] == k + 1
        assert not any(o.throttled for o in client.queue._queue)
        hub.broadcast(MessageType.RISK_EVENT, {"level": "INFO", "code": "overflow", "message": "m"})
        assert hub.client_count == 0  # only critical messages left -> slow consumer
        await wait_until(lambda: ws.close_code is not None)
        assert ws.close_code == 1013
        await asyncio.wait_for(task, 2)
    finally:
        await hub.stop()


async def test_throttled_messages_are_coalesced_in_client_queue() -> None:
    hub, _bus, _backend = await started_hub()
    ws = FakeWebSocket()
    task = asyncio.create_task(hub.serve(ws))
    try:
        await wait_until(lambda: ws.sent)
        ws.block_sends = True
        hub.broadcast(MessageType.RISK_EVENT, {"level": "INFO", "code": "stuck", "message": "m"})
        await wait_until(lambda: next(iter(hub._clients)).queue.empty())
        for v in range(10):
            hub.broadcast(MessageType.MARKET_TICK, {"last": v})
        assert next(iter(hub._clients)).queue.qsize() == 1
        ws._unblock.set()
        await wait_until(lambda: any(m["type"] == "MARKET_TICK" for m in ws.sent))
        ticks = [m for m in ws.sent if m["type"] == "MARKET_TICK"]
        assert len(ticks) == 1 and ticks[0]["data"] == {"last": 9}
    finally:
        ws.disconnect()
        await asyncio.wait_for(task, 2)
        await hub.stop()


async def test_serve_before_start_and_after_stop_is_refused() -> None:
    bus = EventBus()
    hub = TerminalHub(bus, FakeTerminalBackend(), ApiSettings(), tick_interval=3600)
    ws = FakeWebSocket()
    await hub.serve(ws)
    assert ws.close_code == 1013 and ws.sent == []
    await hub.start()
    await hub.stop()
    ws2 = FakeWebSocket()
    await hub.serve(ws2)
    assert ws2.close_code == 1013


async def test_disconnect_removes_client_and_frees_resources() -> None:
    hub, _bus, _backend = await started_hub()
    ws = FakeWebSocket()
    task = asyncio.create_task(hub.serve(ws))
    await wait_until(lambda: ws.sent)
    assert hub.client_count == 1
    ws.disconnect()
    await asyncio.wait_for(task, 2)
    assert hub.client_count == 0 and ws.close_code is None
    await hub.stop()


# ================================================================================================ helpers
class Color(Enum):
    RED = "red"


@dataclass
class Box:
    a: Decimal
    b: Color


def test_to_jsonable_and_envelope() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    obj = {"d": Decimal("1.5"), "t": naive, "e": Color.RED, "s": SystemStatus.HALTED, "n": float("nan"),
           "inf": Decimal("Infinity"), "set": {3, 1, 2}, "tuple": (1, "a"), "box": Box(Decimal("2"), Color.RED),
           "none": None, "k": {Color.RED: 1}}
    out = to_jsonable(obj)
    assert out == {"d": 1.5, "t": "2026-01-01T12:00:00+00:00", "e": "red", "s": "HALTED", "n": None, "inf": None,
                   "set": [1, 2, 3], "tuple": [1, "a"], "box": {"a": 2.0, "b": "red"}, "none": None,
                   "k": {"red": 1}}
    text = envelope(MessageType.PONG, 7, {"msg": "Türkçe ğüşiöç", "x": Decimal("0.1"), "dt": NOW}, NOW)
    assert "Türkçe ğüşiöç" in text  # ensure_ascii=False
    parsed = json.loads(text)
    assert parsed == {"type": "PONG", "seq": 7, "ts": NOW.isoformat(),
                      "data": {"msg": "Türkçe ğüşiöç", "x": 0.1, "dt": NOW.isoformat()}}
    assert list(parsed) == ["type", "seq", "ts", "data"]


def test_claude_log_item_normalisation() -> None:
    item = claude_log_item({"decision_id": "d", "symbol": BTC, "action": "HOLD", "confidence": Decimal("0.7"),
                            "created_at": NOW, "leverage": "3", "risk_flags": ("A",), "reasoning_tr": "Bekle"})
    assert item["confidence"] == 0.7 and item["timestamp"] == NOW.isoformat() and item["leverage"] == 3
    assert item["risk_flags"] == ["A"] and item["update"] is False and item["risk_decision"] is None
    assert item["metrics_summary"] == "" and item["latency_ms"] is None and item["fallback"] is None
    bad = claude_log_item({"decision_id": "d", "confidence": "n/a", "update": True})
    assert bad["confidence"] == 0.0 and bad["update"] is True and bad["timestamp"]
    assert set(item) == {"decision_id", "timestamp", "symbol", "action", "confidence", "leverage",
                         "metrics_summary", "reasoning_tr", "risk_flags", "risk_decision", "execution_status",
                         "fallback", "latency_ms", "update"}


def test_news_item_and_execution_report_dict() -> None:
    item = news_item(make_news_event(3))
    assert item["time"] == NOW.isoformat() and item["sentiment_score"] == 0.35 and item["symbols"] == ["BTC"]
    json.dumps(item)
    report = ExecutionReport(status=ExecutionStatus.HOLD, symbol=BTC, decision_id=None, signal_id=None, detail="d",
                             drift_atr=Decimal("0.25"), risk_reasons=["r"], warnings=["w"], timestamp=None)
    d = execution_report_dict(report)
    assert d["status"] == "HOLD" and d["side"] is None and d["drift_atr"] == 0.25 and d["filled_quantity"] == 0.0
    assert d["orders"] == [] and d["timestamp"] is None and d["risk_reasons"] == ["r"]
    json.dumps(d)


@pytest.mark.parametrize(("origin", "host", "ok"), [
    (None, "x", True),
    ("http://localhost:8000", "other:1", True),
    ("http://localhost:8000/", "other:1", True),
    ("HTTP://LOCALHOST:8000", "other:1", True),
    ("http://evil.example", "localhost:8000", False),
    ("http://evil.example", "evil.example", True),
    ("http://evil.example:81", "evil.example", False),
    ("null", "localhost:8000", False),
    ("", "localhost:8000", False),
])
def test_origin_allowed(origin: str | None, host: str, ok: bool) -> None:
    assert origin_allowed(origin, host, ["http://127.0.0.1:8000", "http://localhost:8000"]) is ok
