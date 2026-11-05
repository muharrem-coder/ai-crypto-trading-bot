"""Pre-flight CLI tests. No real network: fake ccxt / anthropic objects, sqlite file DB, local RESP server."""

from __future__ import annotations

import os
import asyncio
import json
import time
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import SecretStr

from app.cli import preflight as pf
from app.cli.preflight import Status, mask, render_text, run_preflight
from app.config.settings import LIVE_CONFIRMATION_PHRASE, ApiSettings, ClaudeSettings, RiskSettings, Settings
from app.core.errors import ConfigError
from app.core.types import TradingMode
from app.db.session import create_engine_from_url, init_db

API_KEY = "testkey-FAKE-EXCHANGE-0001"
API_SECRET = "SECRETFAKE0987654321ZYXW"
ANTHROPIC_KEY = "test-anthropic-FAKE-KEY-0002"
TOKEN = "t" * 40
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
_ENV_PREFIXES = ("TRADING_MODE", "EXCHANGE_", "RISK_", "DATA_", "CLAUDE_", "EXEC_", "API_", "DATABASE_URL",
                 "REDIS_URL", "ANTHROPIC_API_KEY", "LIVE_TRADING_CONFIRMATION", "SYMBOLS", "QUOTE_CURRENCY",
                 "NO_COLOR", "QUANT_", "NEWS_", "PAPER_")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    import os

    for k in list(os.environ):
        if k.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env


# ----------------------------------------------------------------------------- fakes
class FakeExchange:
    def __init__(self) -> None:
        now_markets = {s: {"symbol": s, "active": True, "swap": True, "spot": False} for s in SYMBOLS}
        self.markets = now_markets
        self.time_offset_ms = 0
        self.hang_load_markets = False
        self.balance: Any = {"USDT": {"total": 5000, "free": 4800}}
        self.positions: list[dict[str, Any]] = []
        self.binance_restrictions: Any = {"enableWithdrawals": False, "enableFutures": True,
                                          "enableReading": True, "ipRestrict": True}
        self.bybit_info: Any = {"retCode": 0, "result": {"permissions": {"Wallet": ["AccountTransfer"]},
                                                         "readOnly": 0, "ips": ["1.2.3.4"]}}
        self.balance_error: BaseException | None = None
        self.calls: list[tuple[str, dict, bool]] = []
        self.closed = 0

    def factory(self, exchange_id: str, config: dict, testnet: bool = False) -> "FakeExchange":
        self.calls.append((exchange_id, config, testnet))
        return self

    async def load_markets(self):
        if self.hang_load_markets:
            await asyncio.sleep(3600)
        return self.markets

    async def fetch_ticker(self, symbol):
        return {"symbol": symbol, "last": 65000.5}

    async def fetch_time(self):
        return int(time.time() * 1000) + self.time_offset_ms

    async def fetch_balance(self):
        if self.balance_error:
            raise self.balance_error
        return self.balance

    async def fetch_positions(self):
        return self.positions

    async def sapi_get_account_apirestrictions(self):
        if isinstance(self.binance_restrictions, BaseException):
            raise self.binance_restrictions
        return self.binance_restrictions

    async def private_get_v5_user_query_api(self):
        return self.bybit_info

    async def close(self):
        self.closed += 1


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://api.anthropic.test/v1/models/x"))


class FakeAnthropic:
    def __init__(self, retrieve_exc: BaseException | None = None, create_result: Any = None, delay: float = 0) -> None:
        self.retrieve_exc = retrieve_exc
        self.create_result = create_result
        self.delay = delay
        self.retrieved: list[str] = []
        self.models = SimpleNamespace(retrieve=self._retrieve)
        self.messages = SimpleNamespace(create=self._create)
        self.create_kwargs: dict[str, Any] = {}

    async def _retrieve(self, model):
        self.retrieved.append(model)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.retrieve_exc:
            raise self.retrieve_exc
        return SimpleNamespace(id=model)

    async def _create(self, **kw):
        self.create_kwargs = kw
        return self.create_result

    async def close(self):
        pass


def tool_use(inp: dict[str, Any]) -> Any:
    block = SimpleNamespace(type="tool_use", name="submit_trading_decision", input=inp)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


GOOD_INPUT = {"action": "HOLD", "confidence": 0.1, "leverage": 1, "entry_type": "MARKET",
              "reasoning_tr": "Sentetik test, islem yapilmaz.", "risk_flags": []}


# ----------------------------------------------------------------------------- helpers
def paper_settings(**kw: Any) -> Settings:
    kw.setdefault("symbols", list(SYMBOLS))
    kw.setdefault("anthropic_api_key", SecretStr(ANTHROPIC_KEY))
    kw.setdefault("redis_url", "redis://127.0.0.1:1/0")  # refused immediately -> WARN
    return Settings(trading_mode=TradingMode.PAPER, **kw)


def live_settings(**kw: Any) -> Settings:
    kw.setdefault("symbols", list(SYMBOLS))
    kw.setdefault("anthropic_api_key", SecretStr(ANTHROPIC_KEY))
    kw.setdefault("redis_url", "redis://127.0.0.1:1/0")
    kw.setdefault("exchange_api_key", SecretStr(API_KEY))
    kw.setdefault("exchange_api_secret", SecretStr(API_SECRET))
    kw.setdefault("api", ApiSettings(control_token=SecretStr(TOKEN)))
    return Settings(trading_mode=TradingMode.LIVE, live_trading_confirmation=LIVE_CONFIRMATION_PHRASE, **kw)


def tn_settings(**kw: Any) -> Settings:
    kw.setdefault("symbols", list(SYMBOLS))
    kw.setdefault("anthropic_api_key", SecretStr(ANTHROPIC_KEY))
    kw.setdefault("redis_url", "redis://127.0.0.1:1/0")
    kw.setdefault("exchange_api_key", SecretStr(API_KEY))
    kw.setdefault("exchange_api_secret", SecretStr(API_SECRET))
    return Settings(trading_mode=TradingMode.TESTNET, **kw)


@pytest.fixture
async def db_url(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'pf.db'}"
    engine = create_engine_from_url(url)
    await init_db(engine)
    await engine.dispose()
    return url


async def go(settings: Settings, fx: FakeExchange | None, db_url: str, *, anth: Any = "default", **kw: Any):
    fx = fx or FakeExchange()
    if anth == "default":
        anth = FakeAnthropic()
    return await run_preflight(
        lambda: settings, None, exchange_factory=fx.factory,
        anthropic_factory=lambda key, timeout: anth, db_url_override=db_url, **kw,
    )


def find(report, name: str, contains: str = "") -> pf.CheckResult:
    hits = [r for r in report.results if r.name == name and contains in r.detail]
    assert hits, f"no result {name!r} containing {contains!r}; have: {[(r.name, r.detail[:50]) for r in report.results]}"
    return hits[0]


# ----------------------------------------------------------------------------- RESP server
async def start_resp_server(password: str | None = None):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        authed = password is None
        try:
            while True:
                head = await reader.readline()
                if not head:
                    break
                n = int(head[1:])
                parts = []
                for _ in range(n):
                    ln = int((await reader.readline())[1:])
                    parts.append((await reader.readexactly(ln + 2))[:-2].decode())
                cmd = parts[0].upper()
                if cmd == "AUTH":
                    if parts[-1] == password:
                        authed = True
                        writer.write(b"+OK\r\n")
                    else:
                        writer.write(b"-WRONGPASS invalid username-password pair\r\n")
                elif not authed:
                    writer.write(b"-NOAUTH Authentication required.\r\n")
                elif cmd == "PING":
                    writer.write(b"+PONG\r\n")
                elif cmd == "SELECT":
                    writer.write(b"+OK\r\n")
                else:
                    writer.write(b"-ERR unknown\r\n")
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


# ----------------------------------------------------------------------------- tests
async def test_all_green_paper_exits_zero(db_url):
    server, port = await start_resp_server(password="redispw123")
    try:
        s = paper_settings(redis_url=f"redis://:redispw123@127.0.0.1:{port}/2")
        report = await go(s, None, db_url)
    finally:
        server.close()
        await server.wait_closed()
    assert report.exit_code == 0 and report.ready
    assert report.count(Status.FAIL) == 0
    assert find(report, "Redis").status is Status.OK
    assert "PONG" in find(report, "Redis").detail
    assert find(report, "TimescaleDB").status is Status.WARN
    assert find(report, "DB migrations").status is Status.WARN
    assert find(report, "DB tables").status is Status.OK
    assert find(report, "Kill switch").status in (Status.OK, Status.WARN)
    assert find(report, "Exchange (private)").status is Status.SKIP
    assert find(report, "Claude API").status is Status.OK
    assert report.verdict == "HAZIR / READY for PAPER"
    assert "READY for PAPER" in render_text(report)


async def test_live_withdrawals_enabled_fails(db_url):
    fx = FakeExchange()
    fx.binance_restrictions = {"enableWithdrawals": True, "enableFutures": True, "enableReading": True, "ipRestrict": True}
    report = await go(live_settings(), fx, db_url)
    r = find(report, "Binance API key", "withdrawals")
    assert r.status is Status.FAIL and "withdrawals ENABLED" in r.detail
    assert report.exit_code == 1 and report.verdict == "HAZIR DEĞİL / NOT READY"
    # the binance permission call used mainnet (no sandbox)
    assert all(not tn for (_id, _cfg, tn) in fx.calls)


async def test_binance_futures_not_enabled_fails_and_good_key_ok(db_url):
    fx = FakeExchange()
    fx.binance_restrictions = {"enableWithdrawals": False, "enableFutures": False, "enableReading": True, "ipRestrict": True}
    report = await go(paper_settings(exchange_api_key=SecretStr(API_KEY), exchange_api_secret=SecretStr(API_SECRET)), fx, db_url)
    assert find(report, "Binance API key", "futures").status is Status.FAIL
    fx.binance_restrictions["enableFutures"] = True
    report = await go(paper_settings(exchange_api_key=SecretStr(API_KEY), exchange_api_secret=SecretStr(API_SECRET)), fx, db_url)
    assert find(report, "Binance API key", "withdrawals disabled").status is Status.OK
    assert find(report, "Binance API key", "futures trading enabled").status is Status.OK


async def test_bybit_withdraw_permission_fails_and_readonly_fails(db_url):
    fx = FakeExchange()
    fx.bybit_info = {"result": {"permissions": {"Wallet": ["AccountTransfer", "Withdraw"]}, "readOnly": 1, "ips": ["*"]}}
    s = tn_settings(exchange_id="bybit")
    report = await go(s, fx, db_url)
    assert find(report, "Bybit API key", "Withdraw").status is Status.FAIL
    assert find(report, "Bybit API key", "READ-ONLY").status is Status.FAIL
    assert find(report, "Bybit API key", "IP restriction").status is Status.WARN
    assert report.exit_code == 1


async def test_ip_restrict_false_is_warn_then_fail_in_live(db_url):
    fx = FakeExchange()
    fx.binance_restrictions["ipRestrict"] = False
    keys = dict(exchange_api_key=SecretStr(API_KEY), exchange_api_secret=SecretStr(API_SECRET))
    report = await go(paper_settings(**keys), fx, db_url)
    r = find(report, "Binance API key", "IP")
    assert r.status is Status.WARN and "IP whitelist" in r.detail and not r.escalated
    assert find(report, "Mode locks (PAPER)").status is Status.WARN  # keys present but unused
    assert report.exit_code == 0

    report = await go(live_settings(), fx, db_url)
    r = find(report, "Binance API key", "IP")
    assert r.status is Status.FAIL and r.escalated
    assert report.exit_code == 1


async def test_live_non_escalating_warn_stays_warn(db_url):
    # Redis unreachable is WARN in LIVE and is NOT escalated
    report = await go(live_settings(), None, db_url)
    assert find(report, "Redis").status is Status.WARN
    assert not find(report, "Redis").escalated


async def test_binance_testnet_permissions_skipped(db_url):
    report = await go(tn_settings(), None, db_url)
    assert find(report, "Binance API key").status is Status.SKIP
    assert "testnet" in find(report, "Binance API key").detail
    assert find(report, "Exchange balance").status is Status.OK


async def test_unknown_exchange_permission_check_warns(db_url):
    s = tn_settings(exchange_id="okx")
    report = await go(s, None, db_url)
    r = find(report, "Okx API key")
    assert r.status is Status.WARN and "not implemented for okx" in r.detail


async def test_missing_symbol_fails(db_url):
    fx = FakeExchange()
    del fx.markets["ETH/USDT:USDT"]
    fx.markets["BTC/USDT:USDT"]["active"] = False
    report = await go(paper_settings(), fx, db_url)
    r = find(report, "Symbols")
    assert r.status is Status.FAIL and "unknown: ETH/USDT:USDT" in r.detail and "inactive: BTC/USDT:USDT" in r.detail
    assert report.exit_code == 1


@pytest.mark.parametrize("offset,expected", [(300, Status.OK), (2500, Status.WARN), (-9000, Status.FAIL)])
async def test_clock_drift(db_url, offset, expected):
    fx = FakeExchange()
    fx.time_offset_ms = offset
    report = await go(paper_settings(), fx, db_url)
    assert find(report, "Clock drift").status is expected


async def test_clock_drift_warn_escalates_in_live(db_url):
    fx = FakeExchange()
    fx.time_offset_ms = 2500
    report = await go(live_settings(), fx, db_url)
    r = find(report, "Clock drift")
    assert r.status is Status.FAIL and r.escalated


@pytest.mark.parametrize("exc,needle", [
    (anthropic.AuthenticationError("bad", response=_resp(401), body=None), "invalid key"),
    (anthropic.NotFoundError("nf", response=_resp(404), body=None), "model id not found: set CLAUDE_MODEL"),
])
async def test_claude_errors_fail(db_url, exc, needle):
    report = await go(paper_settings(), None, db_url, anth=FakeAnthropic(retrieve_exc=exc))
    r = find(report, "Claude API")
    assert r.status is Status.FAIL and needle in r.detail
    assert report.exit_code == 1


async def test_claude_rate_limit_warns_and_connection_error_depends_on_mode(db_url):
    rl = anthropic.RateLimitError("slow down", response=_resp(429), body=None)
    report = await go(paper_settings(), None, db_url, anth=FakeAnthropic(retrieve_exc=rl))
    assert find(report, "Claude API").status is Status.WARN
    conn = anthropic.APIConnectionError(request=httpx.Request("GET", "https://x.test"))
    report = await go(paper_settings(), None, db_url, anth=FakeAnthropic(retrieve_exc=conn))
    assert find(report, "Claude API").status is Status.WARN
    report = await go(tn_settings(), None, db_url, anth=FakeAnthropic(retrieve_exc=conn))
    assert find(report, "Claude API").status is Status.FAIL


async def test_claude_missing_key_by_mode(db_url):
    report = await go(paper_settings(anthropic_api_key=SecretStr("")), None, db_url)
    assert find(report, "Claude API").status is Status.WARN
    report = await go(tn_settings(anthropic_api_key=SecretStr("")), None, db_url)
    r = find(report, "Claude API")
    assert r.status is Status.FAIL and "HOLD" in r.detail


async def test_claude_model_and_timeout_are_used(db_url):
    anth = FakeAnthropic()
    s = paper_settings(claude=ClaudeSettings(model="claude-test-model", timeout_seconds=4))
    seen: dict[str, Any] = {}

    def anth_factory(key, timeout):
        seen["timeout"] = timeout
        return anth

    fx = FakeExchange()
    await run_preflight(lambda: s, None, exchange_factory=fx.factory, anthropic_factory=anth_factory, db_url_override=db_url)
    assert anth.retrieved == ["claude-test-model"] and seen["timeout"] == 4


async def test_claude_roundtrip(db_url):
    anth = FakeAnthropic(create_result=tool_use(GOOD_INPUT))
    report = await go(paper_settings(), None, db_url, anth=anth, claude_roundtrip=True)
    r = find(report, "Claude round-trip")
    assert r.status is Status.OK and "valid ClaudeDecision" in r.detail
    assert anth.create_kwargs["max_tokens"] == 300
    assert anth.create_kwargs["tool_choice"] == {"type": "tool", "name": "submit_trading_decision"}

    bad = FakeAnthropic(create_result=tool_use({"action": "LONG", "confidence": 0.5, "leverage": 2,
                                                "entry_type": "MARKET", "reasoning_tr": "x" * 30, "risk_flags": []}))
    report = await go(paper_settings(), None, db_url, anth=bad, claude_roundtrip=True)
    assert find(report, "Claude round-trip").status is Status.FAIL
    none = FakeAnthropic(create_result=SimpleNamespace(content=[], stop_reason="end_turn"))
    report = await go(paper_settings(), None, db_url, anth=none, claude_roundtrip=True)
    assert "no submit_trading_decision" in find(report, "Claude round-trip").detail
    # off by default
    report = await go(paper_settings(), None, db_url, anth=FakeAnthropic(create_result=tool_use(GOOD_INPUT)))
    assert not [r for r in report.results if r.name == "Claude round-trip"]


async def test_risk_limit_leverage_10_fails(db_url):
    s = paper_settings(risk=RiskSettings(max_leverage=10, max_risk_per_trade_pct=Decimal("0.02")))
    report = await go(s, None, db_url)
    r = find(report, "Risk limits")
    assert r.status is Status.FAIL
    assert "RISK_MAX_LEVERAGE=10" in r.detail and "RISK_MAX_RISK_PER_TRADE_PCT=0.02" in r.detail
    assert report.exit_code == 1


async def test_risk_defaults_pass(db_url):
    report = await go(paper_settings(), None, db_url)
    assert find(report, "Risk limits").status is Status.OK


async def test_open_positions_cross_or_high_leverage(db_url):
    fx = FakeExchange()
    fx.positions = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long", "leverage": 3, "marginMode": "isolated"},
        {"symbol": "ETH/USDT:USDT", "contracts": 0.5, "side": "short", "leverage": 10, "marginMode": "cross"},
        {"symbol": "SOL/USDT:USDT", "contracts": 0, "side": None, "leverage": 20, "marginMode": "cross"},
    ]
    report = await go(tn_settings(), fx, db_url)
    r = find(report, "Open positions")
    assert r.status is Status.WARN and "ETH/USDT:USDT" in r.detail and "SOL" not in r.detail.split("in:")[-1]
    report = await go(live_settings(), fx, db_url)
    assert find(report, "Open positions").status is Status.FAIL
    fx.positions = fx.positions[:1]
    report = await go(live_settings(), fx, db_url)
    assert find(report, "Open positions").status is Status.OK


async def test_balance_failure_and_zero_balance(db_url):
    fx = FakeExchange()
    fx.balance_error = RuntimeError(f"401 for key {API_KEY} signature invalid")
    report = await go(tn_settings(), fx, db_url)
    r = find(report, "Exchange balance")
    assert r.status is Status.FAIL and API_KEY not in r.detail
    fx.balance_error = None
    fx.balance = {"total": {"USDT": 0}, "free": {"USDT": 0}}
    report = await go(tn_settings(), fx, db_url)
    assert find(report, "Exchange balance").status is Status.WARN


async def test_live_balance_floor_uses_recon_setting_when_present(db_url):
    class WithRecon(Settings):
        recon: Any = None

    s = WithRecon(trading_mode=TradingMode.LIVE, live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
                  symbols=list(SYMBOLS), exchange_api_key=SecretStr(API_KEY), exchange_api_secret=SecretStr(API_SECRET),
                  api=ApiSettings(control_token=SecretStr(TOKEN)), anthropic_api_key=SecretStr(ANTHROPIC_KEY),
                  redis_url="redis://127.0.0.1:1/0", recon=SimpleNamespace(live_min_balance=Decimal(10000)))
    report = await go(s, None, db_url)
    assert find(report, "Live balance floor").status is Status.WARN
    # real Settings: SKIP while it has no recon section (another agent adds one), otherwise judged by its value
    report = await go(live_settings(), None, db_url)
    expected = {Status.OK, Status.WARN} if "recon" in Settings.model_fields else {Status.SKIP}
    assert find(report, "Live balance floor").status in expected
    # not LIVE: no such line
    report = await go(paper_settings(), None, db_url)
    assert not [r for r in report.results if r.name == "Live balance floor"]


async def test_kill_switch_engaged_fails(tmp_path):
    from sqlalchemy import text

    url = f"sqlite+aiosqlite:///{tmp_path / 'ks.db'}"
    engine = create_engine_from_url(url)
    await init_db(engine)
    async with engine.begin() as conn:
        await conn.execute(text("INSERT INTO kill_switch_state (id, engaged, reason) VALUES (1, 1, 'daily drawdown 3.2%')"))
    await engine.dispose()
    report = await go(paper_settings(), None, url)
    r = find(report, "Kill switch")
    assert r.status is Status.FAIL and "ENGAGED" in r.detail and "daily drawdown 3.2%" in r.detail
    assert report.exit_code == 1


async def test_database_missing_tables_and_unreachable(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}"
    report = await go(paper_settings(), None, url)
    r = find(report, "DB tables")
    assert r.status is Status.WARN and "13 of 13 missing" in r.detail
    assert find(report, "DB migrations").status is Status.WARN
    bad = await go(paper_settings(), None, f"sqlite+aiosqlite:///{tmp_path / 'nodir' / 'x.db'}")
    assert find(bad, "Database").status is Status.FAIL


async def test_live_timescale_missing_is_fail(db_url):
    report = await go(live_settings(), None, db_url)
    assert find(report, "TimescaleDB").status is Status.FAIL


async def test_redis_variants():
    s = paper_settings()
    # wrong password -> WARN with server message
    server, port = await start_resp_server(password="right-password")
    try:
        ctx = pf.Ctx(paper_settings(redis_url=f"redis://:wrong@127.0.0.1:{port}/0"), TradingMode.PAPER, skip_network=False,
                     claude_roundtrip=False, exchange_factory=None, anthropic_factory=None, db_url_override=None,
                     check_timeout=5)
        r = await pf.check_redis(ctx)
        assert r.status is Status.WARN and "WRONGPASS" in r.detail
        latency, where = await pf.redis_probe(f"redis://:right-password@127.0.0.1:{port}/3")
        assert where.endswith("/3") and latency >= 0
    finally:
        server.close()
        await server.wait_closed()
    ctx.settings = s
    r = await pf.check_redis(ctx)
    assert r.status is Status.WARN and "unreachable" in r.detail


async def test_api_exposure(db_url, monkeypatch):
    report = await go(paper_settings(), None, db_url)
    assert find(report, "API exposure").status is Status.OK
    s = paper_settings(api=ApiSettings(host="0.0.0.0", control_token=SecretStr("")))
    report = await go(s, None, db_url)
    assert find(report, "API exposure").status is Status.FAIL
    monkeypatch.setattr(pf, "_in_docker", lambda: True)
    monkeypatch.setenv("API_PUBLISH_HOST", "127.0.0.1")
    s = paper_settings(api=ApiSettings(host="0.0.0.0", control_token=SecretStr(TOKEN)))
    report = await go(s, None, db_url)
    assert find(report, "API exposure").status is Status.OK
    monkeypatch.delenv("API_PUBLISH_HOST")
    report = await go(s, None, db_url)
    assert find(report, "API exposure").status is Status.WARN


async def test_mode_locks_live_token_length(db_url):
    s = live_settings(api=ApiSettings(control_token=SecretStr("y" * 20)))
    report = await go(s, None, db_url)
    r = find(report, "Mode locks (LIVE)")
    assert r.status is Status.FAIL and r.escalated and "32" in r.detail


async def test_settings_invalid_reports_message_and_skips_rest(db_url):
    def factory() -> Settings:
        return Settings(trading_mode=TradingMode.LIVE, live_trading_confirmation="nope")

    report = await run_preflight(factory, None, db_url_override=db_url)
    r = find(report, "Settings")
    assert r.status is Status.FAIL and "confirmation phrase" in r.detail
    assert find(report, "Database").status is Status.SKIP
    assert find(report, "Python").status is Status.OK
    assert report.exit_code == 1 and report.mode == "LIVE" or report.mode == "PAPER"


async def test_python_and_packages_ok():
    ctx = pf.Ctx(None, TradingMode.PAPER, skip_network=False, claude_roundtrip=False, exchange_factory=None,
                 anthropic_factory=None, db_url_override=None, check_timeout=5)
    assert pf.check_python(ctx).status is Status.OK
    r = pf.check_packages(ctx)
    assert r.status is Status.OK and "ccxt" in r.detail and "pydantic_settings" in r.detail


async def test_skip_network(db_url):
    fx = FakeExchange()
    report = await go(paper_settings(), fx, db_url, skip_network=True)
    assert find(report, "Exchange (public)").status is Status.SKIP
    assert find(report, "Claude API").status is Status.SKIP
    assert fx.calls == []
    live = await go(live_settings(), fx, db_url, skip_network=True)
    assert find(live, "Exchange (private)").status is Status.FAIL  # cannot claim READY for LIVE unverified


async def test_secrets_never_printed(db_url, capsys):
    fx = FakeExchange()
    fx.balance_error = RuntimeError(f"bad key {API_KEY} secret {API_SECRET} token {TOKEN}")
    anth = FakeAnthropic(retrieve_exc=RuntimeError(f"oops {ANTHROPIC_KEY}"))
    report = await go(live_settings(), fx, db_url, anth=anth)
    text = render_text(report, colour=True)
    blob = json.dumps(report.to_dict(), ensure_ascii=False)
    for secret in (API_KEY, API_SECRET, ANTHROPIC_KEY, TOKEN):
        assert secret not in text and secret not in blob
    assert mask(API_KEY) == "test…01" and mask(API_KEY) in text
    assert mask("short") == "****"


async def test_json_shape(db_url):
    report = await go(paper_settings(), None, db_url)
    data = json.loads(json.dumps(report.to_dict()))
    assert set(data) == {"mode", "ready", "verdict", "exit_code", "summary", "duration_ms", "checks"}
    assert data["mode"] == "PAPER" and data["ready"] is True and data["exit_code"] == 0
    assert set(data["summary"]) == {"ok", "fail", "warn", "skip"}
    assert sum(data["summary"].values()) == len(data["checks"])
    for c in data["checks"]:
        assert set(c) == {"name", "status", "detail", "duration_ms", "escalated"}
        assert c["status"] in {"OK", "FAIL", "WARN", "SKIP"} and c["duration_ms"] >= 0


async def test_colour_only_when_enabled(db_url, monkeypatch):
    report = await go(paper_settings(), None, db_url)
    assert "\x1b[" not in render_text(report, colour=False)
    coloured = render_text(report, colour=True)
    assert "\x1b[32m[ OK ]" in coloured and "\x1b[33m[WARN]" in coloured and "\x1b[90m[SKIP]" in coloured

    class Tty:
        def isatty(self):
            return True

    class NoTty:
        def isatty(self):
            return False

    assert pf.colour_enabled(Tty()) is True
    assert pf.colour_enabled(NoTty()) is False
    monkeypatch.setenv("NO_COLOR", "1")
    assert pf.colour_enabled(Tty()) is False
    fail_report = pf.PreflightReport("PAPER", [pf.fail("x", "boom")])
    assert "\x1b[31m[FAIL]" in render_text(fail_report, colour=True)
    assert "HAZIR DEĞİL / NOT READY" in render_text(fail_report)


async def test_one_check_timeout_does_not_stop_others(db_url):
    fx = FakeExchange()
    fx.hang_load_markets = True
    started = time.perf_counter()
    report = await go(paper_settings(), fx, db_url, check_timeout=0.3)
    assert time.perf_counter() - started < 5
    hung = find(report, "Exchange (public)")
    assert hung.status is Status.FAIL and "timed out" in hung.detail
    assert find(report, "Risk limits").status is Status.OK
    assert find(report, "Database").status is Status.OK
    assert find(report, "Claude API").status is Status.OK


async def test_crashing_check_is_contained(db_url, monkeypatch):
    def boom(ctx):
        raise ValueError("kaboom")

    monkeypatch.setattr(pf, "PLAN", tuple(pf._Plan("Risk limits", boom) if p.label == "Risk limits" else p for p in pf.PLAN))
    report = await go(paper_settings(), None, db_url)
    r = find(report, "Risk limits")
    assert r.status is Status.FAIL and "kaboom" in r.detail
    assert find(report, "API exposure").status is Status.OK


async def test_exchange_closed_after_run(db_url):
    fx = FakeExchange()
    await go(paper_settings(), fx, db_url)
    assert fx.closed >= 1


# ----------------------------------------------------------------------------- main()
def test_main_json_and_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'm.db'}")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    code = pf.main(["--skip-network", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["mode"] == "PAPER" and code == data["exit_code"] == 0
    assert "\x1b[" not in out

    code = pf.main(["--skip-network"])
    out = capsys.readouterr().out
    assert code == 0 and "\x1b[" not in out and "READY for PAPER" in out and "[SKIP] Claude API" in out


def test_main_mode_override_and_env_file(tmp_path, monkeypatch, capsys):
    env = tmp_path / "custom.env"
    env.write_text(
        f"DATABASE_URL=sqlite+aiosqlite:///{tmp_path / 'e.db'}\nREDIS_URL=redis://127.0.0.1:1/0\n"
        "# comment\nRISK_MAX_LEVERAGE=\"10\"\n"
    )
    monkeypatch.delenv("RISK_MAX_LEVERAGE", raising=False)
    code = pf.main(["--mode", "paper", "--skip-network", "--env-file", str(env)])
    out = capsys.readouterr().out
    # --env-file writes straight into os.environ; monkeypatch.delenv here would RESTORE the leaked values at
    # teardown (it records the current value), so pop them directly.
    for key in ("RISK_MAX_LEVERAGE", "DATABASE_URL", "REDIS_URL"):
        os.environ.pop(key, None)
    assert code == 1 and "RISK_MAX_LEVERAGE=10" in out
    assert pf.main(["--env-file", str(tmp_path / "missing.env")]) == 2


def test_main_live_without_confirmation_fails(monkeypatch, capsys):
    monkeypatch.setenv("EXCHANGE_API_KEY", API_KEY)
    monkeypatch.setenv("EXCHANGE_API_SECRET", API_SECRET)
    code = pf.main(["--mode", "LIVE", "--skip-network", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 1 and data["mode"] == "LIVE"
    settings_check = next(c for c in data["checks"] if c["name"] == "Settings")
    assert settings_check["status"] == "FAIL" and "confirmation" in settings_check["detail"]
    assert API_KEY not in json.dumps(data)


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / "x.env"
    f.write_text("export FOO_PF=1\nBAR_PF='two words'\nBAZ_PF=3 # trailing\n")
    monkeypatch.setenv("BAR_PF", "real")
    for k in ("FOO_PF", "BAZ_PF"):
        monkeypatch.delenv(k, raising=False)
    import os

    assert pf.load_env_file(f) == 2
    assert os.environ["FOO_PF"] == "1" and os.environ["BAR_PF"] == "real" and os.environ["BAZ_PF"] == "3"
    for k in ("FOO_PF", "BAZ_PF"):
        monkeypatch.delenv(k, raising=False)


def test_config_error_type_is_handled():
    assert issubclass(ConfigError, Exception)
