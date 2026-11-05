"""FastAPI app: lifespan wiring, static page, REST control API, token protection, WS delegation (no network)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.main import create_app
from app.api.protocol import MessageType
from app.api.server import check_bind_safety, is_loopback_host, running_in_docker
from app.core.types import TradingMode
from app.config.settings import ApiSettings
from app.data.mock_source import ScriptedMarketSource
from app.runtime import TradingRuntime
from tests.test_api_runtime import SYM, history_rows, make_runtime, make_settings, open_long

pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")

TOKEN = "s3cret-control-token-0123456789"
HTML = "<!doctype html><html><body>terminal</body></html>"


class StubHub:
    """Stands in for app.api.ws_hub.TerminalHub."""

    def __init__(self, bus: Any, backend: Any, settings: Any) -> None:
        self.bus, self.backend, self.settings = bus, backend, settings
        self.started = False
        self.stopped = False
        self.broadcasts: list[tuple[MessageType, dict[str, Any]]] = []
        self.served = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def broadcast(self, msg_type: MessageType, data: dict[str, Any]) -> None:
        self.broadcasts.append((msg_type, data))

    async def serve(self, websocket: Any) -> None:
        self.served += 1
        await websocket.accept()
        await websocket.send_json({"type": "HELLO", "symbols": self.backend.symbols})
        await websocket.close()

    def types(self) -> list[MessageType]:
        return [t for t, _ in self.broadcasts]


class Env:
    def __init__(self, client: TestClient, hub: StubHub, runtime: TradingRuntime) -> None:
        self.client, self.hub, self.runtime = client, hub, runtime

    def run(self, fn: Any, *args: Any) -> Any:
        """Run a coroutine function on the app's event loop (the loop that owns the DB connection)."""
        return self.client.portal.call(fn, *args)  # type: ignore[union-attr]


@contextlib.contextmanager
def running_app(tmp_path: Path, *, token: str = "") -> Iterator[Env]:
    static = tmp_path / "static"
    static.mkdir(exist_ok=True)
    (static / "index.html").write_text(HTML, encoding="utf-8")
    (static / "app.js").write_text("console.log('x')", encoding="utf-8")
    settings = make_settings(api=ApiSettings(control_token=SecretStr(token)))
    holder: dict[str, Any] = {}

    async def runtime_factory(s: Any) -> TradingRuntime:
        src = ScriptedMarketSource()
        src.set_history(SYM, history_rows())
        rt = await make_runtime(s, source=src)
        holder["runtime"] = rt
        return rt

    def hub_factory(bus: Any, backend: Any, api_settings: Any) -> StubHub:
        hub = StubHub(bus, backend, api_settings)
        holder["hub"] = hub
        return hub

    app = create_app(settings=settings, runtime_factory=runtime_factory, hub_factory=hub_factory, static_dir=static)
    with TestClient(app) as client:
        env = Env(client, holder["hub"], holder["runtime"])

        async def settle() -> None:
            deadline = time.monotonic() + 3
            while env.runtime.watchdog.ticks < 1 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.15)

        env.run(settle)
        yield env


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    with running_app(tmp_path) as e:
        yield e


@pytest.fixture
def secure(tmp_path: Path) -> Iterator[Env]:
    with running_app(tmp_path, token=TOKEN) as e:
        yield e


def _expire_rate_limit(env: Env) -> None:
    env.client.app.state.kill_switch_last_call = time.monotonic() - 5  # type: ignore[attr-defined]


# ====================================================================== static + lifespan
def test_index_returns_html_no_store(env: Env) -> None:
    r = env.client.get("/")
    assert r.status_code == 200 and r.text == HTML
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"


def test_static_mount_and_missing_index(env: Env, tmp_path: Path) -> None:
    assert env.client.get("/static/app.js").text == "console.log('x')"
    assert env.client.get("/static/nope.js").status_code == 404
    (tmp_path / "static" / "index.html").unlink()
    assert env.client.get("/").status_code == 404


def test_openapi_and_docs_are_disabled(env: Env) -> None:
    assert env.client.get("/docs").status_code == 404
    assert env.client.get("/openapi.json").status_code == 404


def test_lifespan_starts_runtime_and_hub_and_stops_them(tmp_path: Path) -> None:
    with running_app(tmp_path) as env:
        assert env.runtime.is_running and env.hub.started and not env.hub.stopped
        assert env.client.app.state.hub is env.hub  # type: ignore[attr-defined]
        assert env.hub.backend.symbols == [SYM] and env.hub.settings is env.runtime.settings.api
        assert env.hub.bus is env.runtime.bus
    assert env.hub.stopped
    assert not env.runtime.is_running and env.runtime.started_components == []


def test_startup_failure_stops_runtime_and_propagates(tmp_path: Path) -> None:
    holder: dict[str, TradingRuntime] = {}

    async def factory(s: Any) -> TradingRuntime:
        rt = await make_runtime(s)

        async def boom() -> None:
            raise RuntimeError("market failed")

        rt.market.start = boom  # type: ignore[method-assign]
        holder["rt"] = rt
        return rt

    app = create_app(settings=make_settings(), runtime_factory=factory, hub_factory=StubHub, static_dir=tmp_path)
    with pytest.raises(RuntimeError, match="market failed"):
        with TestClient(app):
            pass
    assert not holder["rt"].is_running and holder["rt"].started_components == []


def test_api_unavailable_before_startup(tmp_path: Path) -> None:
    app = create_app(settings=make_settings(), runtime_factory=None, hub_factory=StubHub, static_dir=tmp_path)
    client = TestClient(app)  # lifespan NOT run
    assert client.get("/api/v1/health").status_code == 503


def test_websocket_endpoint_delegates_to_hub(env: Env) -> None:
    with env.client.websocket_connect("/ws/terminal") as ws:
        assert ws.receive_json() == {"type": "HELLO", "symbols": [SYM]}
    assert env.hub.served == 1


# ====================================================================== health / snapshot
def test_health_ok_with_components(env: Env) -> None:
    r = env.client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ok", "components", "uptime_s"}
    assert set(body["components"]) == {"db", "broker", "market_data", "claude", "kill_switch", "news"}
    assert body["components"]["db"]["ok"] is True and body["components"]["broker"]["ok"] is True
    assert r.headers["cache-control"] == "no-store"


def test_snapshot_has_initial_snapshot_keys(env: Env) -> None:
    r = env.client.get("/api/v1/snapshot")
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"equity_kpis", "active_symbol", "symbols", "timeframe", "candles", "markers",
                         "open_positions", "trade_history", "claude_logs", "news_feed", "market_health",
                         "server_time"}
    assert data["active_symbol"] == SYM and data["symbols"] == [SYM] and data["timeframe"] == "1m"
    assert 0 < len(data["candles"]) <= 120
    assert data["equity_kpis"]["equity"] == 10000.0


# ====================================================================== kill switch
def test_kill_switch_needs_no_token_and_flattens(secure: Env) -> None:
    secure.run(open_long, secure.runtime)
    r = secure.client.post("/api/v1/kill-switch")  # no token, no body, token IS configured
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["engaged"] is True and body["system_status"] == "HALTED"
    assert body["closed"] == [SYM] and body["still_open"] == []
    assert body["reason"] == "MANUAL_UI_TRIGGER"
    assert secure.runtime.kill_switch.is_engaged
    types = secure.hub.types()
    assert MessageType.KILL_SWITCH in types and MessageType.POSITIONS_PNL in types and MessageType.EQUITY_KPIS in types
    ks = next(d for t, d in secure.hub.broadcasts if t is MessageType.KILL_SWITCH)
    assert ks == {"engaged": True, "reason": "MANUAL_UI_TRIGGER", "system_status": "HALTED"}
    pnl = next(d for t, d in secure.hub.broadcasts if t is MessageType.POSITIONS_PNL)
    assert pnl == {"positions": [], "unrealized_total": 0.0}


def test_kill_switch_custom_reason_and_rate_limit(env: Env) -> None:
    r = env.client.post("/api/v1/kill-switch", json={"reason": "PANIC BUTTON"})
    assert r.status_code == 200 and r.json()["reason"] == "PANIC BUTTON"
    again = env.client.post("/api/v1/kill-switch")
    assert again.status_code == 429 and again.json()["ok"] is False
    assert again.headers["retry-after"] == "1"
    assert env.runtime.kill_switch.reason == "PANIC BUTTON"
    _expire_rate_limit(env)
    third = env.client.post("/api/v1/kill-switch", json={"reason": "later"})
    assert third.status_code == 200 and third.json()["reason"] == "PANIC BUTTON"  # already engaged: first reason kept


def test_kill_switch_rejects_overlong_reason(env: Env) -> None:
    assert env.client.post("/api/v1/kill-switch", json={"reason": "x" * 500}).status_code == 422
    assert not env.runtime.kill_switch.is_engaged


# ====================================================================== resume
def test_resume_requires_token_when_configured(secure: Env) -> None:
    secure.client.post("/api/v1/kill-switch")
    payload = {"operator": "alice", "confirm": "RESET"}
    assert secure.client.post("/api/v1/kill-switch/resume", json=payload).status_code == 401
    wrong = secure.client.post("/api/v1/kill-switch/resume", json=payload, headers={"X-Control-Token": "nope"})
    assert wrong.status_code == 401
    assert secure.runtime.kill_switch.is_engaged

    ok = secure.client.post("/api/v1/kill-switch/resume", json=payload, headers={"X-Control-Token": TOKEN})
    assert ok.status_code == 200
    assert ok.json()["ok"] is True and ok.json()["engaged"] is False
    assert not secure.runtime.kill_switch.is_engaged
    assert MessageType.KILL_SWITCH in secure.hub.types()
    last = [d for t, d in secure.hub.broadcasts if t is MessageType.KILL_SWITCH][-1]
    assert last["engaged"] is False


def test_resume_wrong_confirm_is_409_and_validation_is_422(secure: Env) -> None:
    secure.client.post("/api/v1/kill-switch")
    headers = {"X-Control-Token": TOKEN}
    r = secure.client.post("/api/v1/kill-switch/resume", json={"operator": "alice", "confirm": "reset"}, headers=headers)
    assert r.status_code == 409 and r.json()["ok"] is False and r.json()["error"] == "RESET_DENIED"
    assert secure.runtime.kill_switch.is_engaged
    assert secure.client.post("/api/v1/kill-switch/resume", json={"operator": "", "confirm": "RESET"},
                              headers=headers).status_code == 422
    assert secure.client.post("/api/v1/kill-switch/resume", json={"confirm": "RESET"}, headers=headers).status_code == 422
    assert secure.runtime.kill_switch.is_engaged


def test_resume_without_configured_token_is_open_but_still_needs_reset(env: Env) -> None:
    env.client.post("/api/v1/kill-switch")
    assert env.client.post("/api/v1/kill-switch/resume",
                           json={"operator": "bob", "confirm": "NOPE"}).status_code == 409
    assert env.client.post("/api/v1/kill-switch/resume",
                           json={"operator": "bob", "confirm": "RESET"}).status_code == 200


def test_non_ascii_token_header_is_rejected_not_crashing(secure: Env) -> None:
    r = secure.client.post("/api/v1/kill-switch/resume", json={"operator": "a", "confirm": "RESET"},
                           headers={"X-Control-Token": "tökén".encode("latin-1")})
    assert r.status_code == 401


# ====================================================================== close position
def test_close_requires_token_and_handles_symbol_with_slash_and_colon(secure: Env) -> None:
    secure.run(open_long, secure.runtime)
    url = f"/api/v1/positions/{SYM}/close"  # contains "/" and ":"
    assert secure.client.post(url).status_code == 401
    assert secure.client.post(url, headers={"X-Control-Token": "wrong"}).status_code == 401
    assert secure.run(secure.runtime.broker.fetch_positions)  # still open

    r = secure.client.post(url, headers={"X-Control-Token": TOKEN})
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["status"] == "CLOSED_BY_DECISION"
    assert secure.run(secure.runtime.broker.fetch_positions) == []
    assert secure.hub.types().count(MessageType.POSITIONS_PNL) >= 1
    assert MessageType.EQUITY_KPIS in secure.hub.types()


def test_close_accepts_url_encoded_symbol(env: Env) -> None:
    env.run(open_long, env.runtime)
    r = env.client.post("/api/v1/positions/BTC%2FUSDT%3AUSDT/close")
    assert r.status_code == 200 and r.json()["status"] == "CLOSED_BY_DECISION"
    assert env.run(env.runtime.broker.fetch_positions) == []


def test_close_without_position_reports_nothing_to_close(env: Env) -> None:
    r = env.client.post(f"/api/v1/positions/{SYM}/close")
    assert r.status_code == 200 and r.json() == {
        "ok": True, "status": "NOTHING_TO_CLOSE", "detail": r.json()["detail"]}


def test_close_rejects_garbage_symbol(env: Env) -> None:
    assert env.client.post("/api/v1/positions/bad%20sym%21/close").status_code == 422
    assert env.client.post("/api/v1/positions/" + "A" * 200 + "/close").status_code == 422


def test_unknown_route_and_method(env: Env) -> None:
    assert env.client.get("/api/v1/kill-switch").status_code == 405
    assert env.client.get("/api/v1/nothing").status_code == 404


# ====================================================================== server helpers
@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True),
    ("0.0.0.0", False), ("192.168.1.10", False), ("example.com", False), ("::", False),
])
def test_is_loopback_host(host: str, expected: bool) -> None:
    assert is_loopback_host(host) is expected


def test_bind_safety_requires_token_for_non_loopback() -> None:
    check_bind_safety("127.0.0.1", "")
    check_bind_safety("0.0.0.0", TOKEN)
    with pytest.raises(SystemExit) as exc:
        check_bind_safety("0.0.0.0", "")
    assert "API_CONTROL_TOKEN" in str(exc.value)


def test_bind_safety_docker_allows_non_live_without_token(caplog: pytest.LogCaptureFixture) -> None:
    for mode in (TradingMode.PAPER, TradingMode.TESTNET):
        caplog.clear()
        with caplog.at_level("WARNING", logger="app.api.server"):
            check_bind_safety("0.0.0.0", "", trading_mode=mode, in_docker=True)
        assert "127.0.0.1" in caplog.text and "Docker" in caplog.text


def test_bind_safety_docker_never_allows_live_without_token() -> None:
    with pytest.raises(SystemExit):
        check_bind_safety("0.0.0.0", "", trading_mode=TradingMode.LIVE, in_docker=True)
    check_bind_safety("0.0.0.0", TOKEN, trading_mode=TradingMode.LIVE, in_docker=True)


def test_bind_safety_outside_docker_still_strict() -> None:
    with pytest.raises(SystemExit):
        check_bind_safety("0.0.0.0", "", trading_mode=TradingMode.PAPER, in_docker=False)
    with pytest.raises(SystemExit):  # unknown mode counts as LIVE
        check_bind_safety("0.0.0.0", "", in_docker=True)


def test_bind_safety_reads_quant_in_docker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUANT_IN_DOCKER", raising=False)
    assert running_in_docker() is False
    with pytest.raises(SystemExit):
        check_bind_safety("0.0.0.0", "", trading_mode=TradingMode.PAPER)
    monkeypatch.setenv("QUANT_IN_DOCKER", "1")
    assert running_in_docker() is True
    check_bind_safety("0.0.0.0", "", trading_mode=TradingMode.PAPER)
    with pytest.raises(SystemExit):
        check_bind_safety("0.0.0.0", "", trading_mode=TradingMode.LIVE)
