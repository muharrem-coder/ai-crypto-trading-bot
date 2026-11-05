"""Real browser against the REAL stack (integrator-owned).

uvicorn + create_app + TradingRuntime + TerminalHub, with only market data / exchange / Claude simulated.
Opens the user's terminal in headless Chromium and checks: snapshot rendered from the real backend, live
latency from PING/PONG, a position opened through the full chain shows up in the table and in the reasoning
log, and the red EMERGENCY KILL-SWITCH button really flattens the paper account and turns the badge HALTED.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from tests.test_integration_phase4 import SYM, Env, book

sync_api = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")
SHOT = Path(__file__).resolve().parent / "ui" / "terminal_real_stack.png"


class LiveServer:
    def __init__(self) -> None:
        self.env = Env()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.env.settings.api.allowed_origins.append(f"http://127.0.0.1:{self.port}")
        self.server = uvicorn.Server(uvicorn.Config(self.env.app, host="127.0.0.1", port=self.port,
                                                    log_level="warning", ws="websockets-sansio"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        for _ in range(200):
            if self.server.started and self.env.loop is not None:
                return
            time.sleep(0.05)
        raise AssertionError("server did not start")

    def run(self, fn: Any, *args: Any) -> None:
        """Run a sync callable on the server's event loop (asyncio queues are not thread-safe)."""
        done = threading.Event()

        def wrapper() -> None:
            try:
                fn(*args)
            finally:
                done.set()
        self.env.loop.call_soon_threadsafe(wrapper)
        assert done.wait(5)

    def call_async(self, coro_fn: Any, *args: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro_fn(*args), self.env.loop).result(10)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def live():
    srv = LiveServer()
    srv.start()
    yield srv
    srv.stop()


def test_terminal_in_browser_against_real_backend(live: LiveServer):
    env = live.env
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1).new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("dialog", lambda d: d.accept())  # confirm() of the kill switch
        page.goto(f"http://127.0.0.1:{live.port}/", wait_until="domcontentloaded")
        page.wait_for_function("() => typeof LightweightCharts !== 'undefined'", timeout=15000)

        # INITIAL_SNAPSHOT from the real backend + real PING/PONG latency
        page.wait_for_function("document.getElementById('kpiEquity').textContent.includes('10,000')", timeout=15000)
        page.wait_for_function("!document.getElementById('latencyValue').textContent.includes('—')", timeout=15000)

        # full chain: market data -> trigger -> Claude (confirming transport) -> risk -> paper fill
        mid = env.rows[120][4]
        bids, asks = book(mid)

        def push_quotes() -> None:
            env.source.push_order_book(SYM, bids, asks)
            env.source.push_ticker(SYM, bids[0][0], asks[0][0], mid)
        live.run(push_quotes)
        page.wait_for_function("document.getElementById('systemModeBadge').textContent.includes('AUTONOMOUS')",
                               timeout=15000)
        live.run(env.source.push_ohlcv, SYM, [env.rows[120]])
        live.run(env.source.push_ohlcv, SYM, [env.rows[121]])
        page.wait_for_function("document.querySelectorAll('#positionsTableBody tr').length === 1", timeout=15000)
        assert "LONG" in page.inner_text("#positionsTableBody")
        page.wait_for_function(
            "document.getElementById('claudeLogContainer').textContent.includes('aşırı satımda')", timeout=15000)
        page.wait_for_timeout(600)  # let throttled KPI/PnL frames land before the screenshot
        page.screenshot(path=str(SHOT))

        # the red EMERGENCY KILL-SWITCH button
        page.click("#killSwitchBtn")
        page.wait_for_function("document.getElementById('systemModeBadge').textContent.includes('HALTED')",
                               timeout=15000)
        page.wait_for_function("document.querySelectorAll('#positionsTableBody tr').length === 0", timeout=15000)
        assert live.call_async(env.paper.fetch_positions) == []
        assert env.runtime.kill_switch.is_engaged
        browser.close()
    assert not [e for e in errors if "fonts.g" not in e], errors
