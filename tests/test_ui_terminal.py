"""Headless-Chromium checks for static/index.html against tests/ui/mock_terminal_server.py."""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "ui"))
from mock_terminal_server import create_app  # noqa: E402

playwright_api = pytest.importorskip("playwright.async_api")
sync_api = pytest.importorskip("playwright.sync_api")

UI_DIR = Path(__file__).resolve().parent / "ui"


APP = create_app()


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(APP, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "mock server did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        yield b
        b.close()


def _cdn_available(page) -> bool:
    return bool(page.evaluate("() => typeof tailwind !== 'undefined' && typeof LightweightCharts !== 'undefined'"))


def test_terminal_ui(base_url, browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    page = ctx.new_page()
    errors: list[str] = []
    dialogs: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    def on_dialog(d):
        dialogs.append(d.message)
        d.dismiss()

    page.on("dialog", on_dialog)
    page.goto(base_url + "/", wait_until="domcontentloaded")

    if not _wait_cdn(page):
        pytest.skip("CDN scripts (tailwind / lightweight-charts) unreachable; UI cannot run")

    # snapshot rendered
    page.wait_for_function("document.querySelectorAll('#positionsTableBody tr').length === 2", timeout=8000)
    assert page.locator("#positionsTableBody tr").count() == 2
    assert page.inner_text("#posCount") == "2"
    assert page.inner_text("#histCount") == "5"
    assert page.locator("#historyTableBody tr").count() == 5

    # live messages: KPI equity, decision upsert (no duplicate), kill switch
    page.wait_for_function("document.getElementById('kpiEquity').textContent === '$12,345.67'", timeout=8000)
    page.wait_for_function("document.getElementById('systemModeBadge').textContent.startsWith('HALTED')", timeout=8000)
    assert "HALTED" in page.inner_text("#systemModeBadge")
    assert "RESUME" in page.inner_text("#killSwitchBtn")

    logs = page.locator("#claudeLogContainer > div")
    assert logs.count() == 5  # 3 snapshot + 1 live decision (first + update merged) + kill-switch warn card
    log_text = page.inner_text("#claudeLogContainer")
    assert "şğüöçıİ" in log_text
    assert "güçlü alım" in log_text
    assert "APPROVED" in log_text and "OPENED_PROTECTED" in log_text
    assert "REJECTED" in log_text

    # injection is rendered as text, never as an element
    assert page.locator("#newsContainer img, #claudeLogContainer img").count() == 0
    assert "<img src=x onerror=alert(1)>" in page.inner_text("#newsContainer")
    assert page.locator("#newsContainer > div").count() == 5
    assert "BULLISH" in page.inner_text("#newsAggregate") or "NEUTRAL" in page.inner_text("#newsAggregate")

    # header ticks / latency (PONG)
    assert page.inner_text("#liveHeaderPrice") == "$89,420.50"
    page.wait_for_function("/^\\d+ms$/.test(document.getElementById('latencyValue').textContent)", timeout=8000)
    assert page.inner_text("#claudeEngine") == "CLAUDE-SONNET-MOCK"
    assert page.inner_text("#kpiDdCurrent") == "Current: 0.42%"
    assert page.inner_text("#kpiTradeCount") == "38 Trades"

    # tabs + US equities overlay still work
    page.click("#tabHistory")
    assert page.is_visible("#viewHistory") and not page.is_visible("#viewPositions")
    page.click("#tabPositions")
    page.click("#btnUsStocks")
    assert page.is_visible("#usEquitiesOverlay")
    page.click("#btnCrypto")
    assert not page.is_visible("#usEquitiesOverlay")

    page.wait_for_timeout(300)
    page.screenshot(path=str(UI_DIR / "terminal.png"))

    assert dialogs == [], f"unexpected dialogs (XSS?): {dialogs}"
    assert errors == [], f"console errors: {errors}"
    ctx.close()


def _wait_cdn(page, timeout_ms: int = 15000) -> bool:
    try:
        page.wait_for_function("typeof tailwind !== 'undefined' && typeof LightweightCharts !== 'undefined'",
                               timeout=timeout_ms)
        return True
    except Exception:
        return False


def test_rest_token_flow_and_close(base_url, browser):
    APP.state.rest_calls.clear()
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(base_url + "/", wait_until="domcontentloaded")
    if not _wait_cdn(page):
        pytest.skip("CDN scripts unreachable")
    page.wait_for_function("document.querySelectorAll('#positionsTableBody tr').length === 2", timeout=8000)
    # let the scripted live messages (incl. KILL_SWITCH card) finish first
    page.wait_for_function("document.getElementById('systemModeBadge').textContent.startsWith('HALTED')", timeout=8000)

    prompts = []

    def on_dialog(d):
        if d.type == "prompt":
            prompts.append(d.message)
            d.accept("secret-token")
        else:
            d.accept()

    page.on("dialog", on_dialog)
    before = page.locator("#claudeLogContainer > div").count()
    page.locator("#positionsTableBody button").first.click()
    deadline = time.time() + 5
    while time.time() < deadline and len(APP.state.rest_calls) < 2:
        page.wait_for_timeout(50)  # sync Playwright only dispatches dialog events while it waits
    calls = APP.state.rest_calls
    assert [c["token"] for c in calls] == [None, "secret-token"], calls
    assert calls[0]["symbol"] == "BTC/USDT:USDT"  # encodeURIComponent round-trips through the path
    assert len(prompts) == 1
    assert page.locator("#claudeLogContainer > div").count() == before  # success: no error card
    assert page.evaluate("() => localStorage.length") == 0

    # second close re-uses the in-memory token: no new prompt
    page.locator("#positionsTableBody button").nth(1).click()
    deadline = time.time() + 5
    while time.time() < deadline and len(APP.state.rest_calls) < 3:
        page.wait_for_timeout(50)  # sync Playwright only dispatches dialog events while it waits
    assert APP.state.rest_calls[2]["token"] == "secret-token"
    assert len(prompts) == 1
    ctx.close()
