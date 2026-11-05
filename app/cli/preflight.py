"""Pre-flight safety check for the trading system.

    python -m app.cli.preflight [--mode PAPER|TESTNET|LIVE] [--json] [--skip-network]
                                [--env-file .env] [--claude-roundtrip] [--timeout 10]

One line per check (``[ OK ]``/``[FAIL]``/``[WARN]``/``[SKIP]``), a summary and a verdict. Exit code 0 when
no check FAILed (in LIVE, WARNs flagged ``live_fail`` are escalated to FAIL), 1 otherwise, 2 on bad usage.

``--skip-network`` skips only the checks that talk to the internet (exchange, Claude API). The database and
Redis checks still run because they are part of your own infrastructure.

Secrets are never printed: keys are shown as first 4 chars + "…" + last 2, and every output string is
additionally passed through a redactor that knows all configured secrets.

All network clients are injectable (``exchange_factory``, ``anthropic_factory``, ``db_url_override``) so the
runner is fully testable offline.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib
import importlib.metadata
import ipaddress
import json
import os
import re
import ssl
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from app.config.settings import LIVE_CONFIRMATION_PHRASE, Settings
from app.core.errors import ConfigError
from app.core.types import TradingMode

DEFAULT_CHECK_TIMEOUT = 10.0
CLAUDE_SLOW_MS = 3000
CLOCK_WARN_MS = 1000
CLOCK_FAIL_MS = 5000
MAX_POSITION_LEVERAGE = Decimal(5)

DOMAIN_TABLES = (
    "accounts", "instruments", "signals", "claude_decisions", "orders", "fills", "trades",
    "news_items", "risk_events", "kill_switch_state", "equity_snapshots", "market_candles",
    "position_snapshots",
)
HYPERTABLES = ("equity_snapshots", "market_candles", "position_snapshots")

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (  # (import name, distribution name)
    ("ccxt", "ccxt"), ("anthropic", "anthropic"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
    ("sqlalchemy", "SQLAlchemy"), ("asyncpg", "asyncpg"), ("httpx", "httpx"),
    ("feedparser", "feedparser"), ("pydantic_settings", "pydantic-settings"),
)


# --------------------------------------------------------------------------------------- result types
class Status(str, Enum):
    OK = "OK"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    duration_ms: float = 0.0
    live_fail: bool = False  # a WARN that becomes FAIL when the mode is LIVE
    escalated: bool = False  # set by the runner when that happened


def ok(name: str, detail: str, **kw: Any) -> CheckResult:
    return CheckResult(name, Status.OK, detail, **kw)


def fail(name: str, detail: str, **kw: Any) -> CheckResult:
    return CheckResult(name, Status.FAIL, detail, **kw)


def warn(name: str, detail: str, *, live_fail: bool = False, **kw: Any) -> CheckResult:
    return CheckResult(name, Status.WARN, detail, live_fail=live_fail, **kw)


def skip(name: str, detail: str, **kw: Any) -> CheckResult:
    return CheckResult(name, Status.SKIP, detail, **kw)


@dataclass
class PreflightReport:
    mode: str
    results: list[CheckResult] = field(default_factory=list)
    duration_ms: float = 0.0

    def count(self, status: Status) -> int:
        return sum(1 for r in self.results if r.status is status)

    @property
    def ready(self) -> bool:
        return self.count(Status.FAIL) == 0

    @property
    def exit_code(self) -> int:
        return 0 if self.ready else 1

    @property
    def verdict(self) -> str:
        return f"HAZIR / READY for {self.mode}" if self.ready else "HAZIR DEĞİL / NOT READY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ready": self.ready,
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "summary": {s.value.lower(): self.count(s) for s in Status},
            "duration_ms": round(self.duration_ms, 1),
            "checks": [
                {"name": r.name, "status": r.status.value, "detail": r.detail,
                 "duration_ms": round(r.duration_ms, 1), "escalated": r.escalated}
                for r in self.results
            ],
        }


# --------------------------------------------------------------------------------------- rendering
_LABEL = {Status.OK: "[ OK ]", Status.FAIL: "[FAIL]", Status.WARN: "[WARN]", Status.SKIP: "[SKIP]"}
_COLOUR = {Status.OK: "32", Status.FAIL: "31", Status.WARN: "33", Status.SKIP: "90"}


def colour_enabled(stream: Any) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _enable_windows_vt() -> None:
    if os.name == "nt":
        try:
            os.system("")  # flips the console into VT mode; no colorama needed
        except Exception:  # noqa: BLE001
            pass


def render_text(report: PreflightReport, *, colour: bool = False) -> str:
    def paint(text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if colour else text

    lines = [f"Pre-flight check - mode {report.mode}", ""]
    for r in report.results:
        detail = " ".join(r.detail.split())
        lines.append(f"{paint(_LABEL[r.status], _COLOUR[r.status])} {r.name}: {detail}")
    lines.append("")
    lines.append(
        f"Summary: {report.count(Status.OK)} OK, {report.count(Status.WARN)} WARN, "
        f"{report.count(Status.FAIL)} FAIL, {report.count(Status.SKIP)} SKIP "
        f"({report.duration_ms / 1000:.1f} s)"
    )
    lines.append(paint(report.verdict, "1;32" if report.ready else "1;31"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- helpers
def mask(secret: str) -> str:
    """First 4 chars + '…' + last 2; very short values are fully hidden."""
    if len(secret) < 8:
        return "****" if secret else "(empty)"
    return f"{secret[:4]}…{secret[-2:]}"


def _secret(value: Any) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value or "")


def _dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _exc_text(exc: BaseException, limit: int = 300) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}"[:limit] if text else type(exc).__name__


def _worst(findings: Sequence[tuple[Status, str]]) -> Status:
    for st in (Status.FAIL, Status.WARN, Status.OK):
        if any(f[0] is st for f in findings):
            return st
    return Status.OK


def _combine(name: str, findings: Sequence[tuple[Status, str]], *, live_fail: bool = False) -> CheckResult:
    status = _worst(findings)
    return CheckResult(name, status, "; ".join(f[1] for f in findings), live_fail=live_fail and status is Status.WARN)


def load_env_file(path: Path) -> int:
    """Minimal dotenv loader: real environment variables win (same precedence as pydantic-settings)."""
    loaded = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


# --------------------------------------------------------------------------------------- context
ExchangeFactory = Callable[..., Any]  # (exchange_id, config, testnet=bool) -> ccxt-like async exchange
AnthropicFactory = Callable[[str, float], Any]  # (api_key, timeout_seconds) -> AsyncAnthropic-like


def default_exchange_factory(exchange_id: str, config: dict[str, Any], testnet: bool = False) -> Any:
    import ccxt.async_support as ccxt_async

    cls = getattr(ccxt_async, exchange_id, None) if not exchange_id.startswith("_") else None
    if cls is None:
        raise ConfigError(f"Unknown or unsupported exchange id: {exchange_id!r}")
    exchange = cls(config)
    if testnet:
        try:
            exchange.set_sandbox_mode(True)
        except Exception as exc:  # noqa: BLE001 - ccxt.NotSupported etc.
            raise ConfigError(f"{exchange_id} has no sandbox/testnet via ccxt ({_exc_text(exc)})") from exc
    return exchange


def default_anthropic_factory(api_key: str, timeout: float) -> Any:
    import anthropic

    return anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)


class Ctx:
    def __init__(
        self,
        settings: Settings | None,
        mode: TradingMode,
        *,
        skip_network: bool,
        claude_roundtrip: bool,
        exchange_factory: ExchangeFactory | None,
        anthropic_factory: AnthropicFactory | None,
        db_url_override: str | None,
        check_timeout: float,
    ) -> None:
        self.settings = settings
        self.mode = mode
        self.skip_network = skip_network
        self.claude_roundtrip = claude_roundtrip
        self.exchange_factory = exchange_factory or default_exchange_factory
        self.anthropic_factory = anthropic_factory or default_anthropic_factory
        self.db_url_override = db_url_override
        self.check_timeout = check_timeout
        self.state: dict[str, Any] = {}
        self._closeables: list[Any] = []
        self._secrets: list[str] = []
        if settings is not None:
            self._collect_secrets(settings)

    @property
    def live(self) -> bool:
        return self.mode is TradingMode.LIVE

    def _collect_secrets(self, s: Settings) -> None:
        candidates: list[Any] = [
            s.exchange_api_key, s.exchange_api_secret, s.exchange_api_password, s.anthropic_api_key,
            s.api.control_token, s.alpaca_api_key, s.alpaca_api_secret, s.news.cryptopanic_token,
            s.news.x_bearer_token,
        ]
        self._secrets = sorted({v for v in (_secret(c) for c in candidates) if len(v) >= 6}, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, mask(secret))
        return text

    def make_exchange(self, exchange_id: str, *, keys: bool, testnet: bool, market: str | None = None) -> Any:
        s = self.settings
        assert s is not None
        market = (market or s.exchange_market).upper()
        if market not in ("SPOT", "PERPETUAL"):
            raise ConfigError(f"EXCHANGE_MARKET must be SPOT or PERPETUAL, got {s.exchange_market!r}")
        options: dict[str, Any] = {"defaultType": "swap" if market == "PERPETUAL" else "spot"}
        if market == "PERPETUAL":
            options["defaultSubType"] = "linear"
        config: dict[str, Any] = {"enableRateLimit": True, "timeout": int(self.check_timeout * 800), "options": options}
        if keys:
            config["apiKey"] = _secret(s.exchange_api_key)
            config["secret"] = _secret(s.exchange_api_secret)
            if _secret(s.exchange_api_password):
                config["password"] = _secret(s.exchange_api_password)
        ex = self.exchange_factory(exchange_id, config, testnet=testnet)
        self._closeables.append(ex)
        return ex

    def track(self, obj: Any) -> Any:
        self._closeables.append(obj)
        return obj

    async def aclose(self) -> None:
        for obj in self._closeables:
            closer = getattr(obj, "close", None)
            if closer is None:
                continue
            try:
                res = closer()
                if asyncio.iscoroutine(res):
                    await asyncio.wait_for(res, 2.0)
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
        self._closeables.clear()

    def has_keys(self) -> bool:
        s = self.settings
        return bool(s and _secret(s.exchange_api_key) and _secret(s.exchange_api_secret))


# --------------------------------------------------------------------------------------- 1. python / packages
def check_python(ctx: Ctx) -> CheckResult:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro} (>= 3.11)"
    return ok("Python", text) if v >= (3, 11) else fail("Python", f"{text} - Python 3.11+ is required")


def check_packages(ctx: Ctx) -> CheckResult:
    found: list[str] = []
    missing: list[str] = []
    for module, dist in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{module} ({type(exc).__name__})")
            continue
        try:
            version = importlib.metadata.version(dist)
        except Exception:  # noqa: BLE001
            version = str(getattr(mod, "__version__", "?"))
        found.append(f"{module} {version}")
    if missing:
        return fail("Packages", f"not importable: {', '.join(missing)} - run `pip install -e .`")
    return ok("Packages", ", ".join(found))


# --------------------------------------------------------------------------------------- 2. settings
def _validation_text(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(p) for p in e['loc']) or '?'}: {e['msg']}" for e in exc.errors()]
        return "; ".join(parts)
    return _exc_text(exc, 600)


def check_mode_locks(ctx: Ctx) -> CheckResult:
    s = ctx.settings
    assert s is not None
    key, secret = _secret(s.exchange_api_key), _secret(s.exchange_api_secret)
    f: list[tuple[Status, str]] = []
    live_fail = False
    name = f"Mode locks ({ctx.mode.value})"
    if ctx.mode is TradingMode.LIVE:
        if s.live_trading_confirmation == LIVE_CONFIRMATION_PHRASE:
            f.append((Status.OK, "confirmation phrase matches"))
        else:
            f.append((Status.FAIL, "LIVE_TRADING_CONFIRMATION is not the exact confirmation phrase"))
        if key and secret:
            f.append((Status.OK, f"exchange keys present ({mask(key)})"))
        else:
            f.append((Status.FAIL, "EXCHANGE_API_KEY / EXCHANGE_API_SECRET missing"))
        n = len(_secret(s.api.control_token))
        if n < 16:
            f.append((Status.FAIL, f"API_CONTROL_TOKEN too short ({n} chars, need >= 16)"))
        elif n < 32:
            f.append((Status.WARN, f"API_CONTROL_TOKEN is {n} chars; >= 32 recommended"))
            live_fail = True
        else:
            f.append((Status.OK, f"control token {n} chars"))
        if s.exchange_testnet:
            f.append((Status.FAIL, "EXCHANGE_TESTNET is true; LIVE needs false"))
        else:
            f.append((Status.OK, "EXCHANGE_TESTNET=false"))
    elif ctx.mode is TradingMode.TESTNET:
        if key and secret:
            f.append((Status.OK, f"testnet keys present ({mask(key)}); orders go to the exchange testnet"))
        else:
            f.append((Status.FAIL, "EXCHANGE_API_KEY / EXCHANGE_API_SECRET (testnet keys) missing"))
    else:
        f.append((Status.OK, "orders are simulated in-process; exchange keys are not needed"))
        if key or secret:
            f.append((Status.WARN, f"exchange keys are set ({mask(key)}) but not used in PAPER"))
    return _combine(name, f, live_fail=live_fail)


# --------------------------------------------------------------------------------------- 3. database
async def check_database(ctx: Ctx) -> list[CheckResult]:
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy import text
    from sqlalchemy.engine import make_url

    from app.db.session import create_engine_from_url

    s = ctx.settings
    assert s is not None
    url = ctx.db_url_override or s.database_url
    shown = make_url(url).render_as_string(hide_password=True)
    engine = create_engine_from_url(url)
    out: list[CheckResult] = []
    try:
        async with engine.connect() as conn:
            t0 = time.perf_counter()
            await conn.execute(text("SELECT 1"))
            latency = (time.perf_counter() - t0) * 1000
            dialect = engine.dialect.name
            if dialect == "postgresql":
                full = (await conn.execute(text("SELECT version()"))).scalar_one()
                m = re.match(r"PostgreSQL (\S+)", str(full))
                version = f"PostgreSQL {m.group(1)}" if m else str(full)[:60]
            else:
                ver = (await conn.execute(text("SELECT sqlite_version()"))).scalar_one() if dialect == "sqlite" else "?"
                version = f"{dialect} {ver}"
            out.append(ok("Database", f"{shown} - {version}, SELECT 1 in {latency:.0f} ms"))

            # TimescaleDB
            has_ts = False
            if dialect == "postgresql":
                ext = (await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname='timescaledb'"))).scalar()
                has_ts = ext is not None
                if has_ts:
                    out.append(ok("TimescaleDB", f"extension {ext} installed"))
                elif ctx.live:
                    out.append(fail("TimescaleDB", "extension missing; LIVE needs hypertables (use a timescaledb image)"))
                else:
                    out.append(warn("TimescaleDB", "extension missing; plain tables will work but are not optimised"))
            else:
                msg = f"not available on {dialect} (dev/test database only)"
                out.append(fail("TimescaleDB", msg + "; LIVE needs PostgreSQL+TimescaleDB") if ctx.live else warn("TimescaleDB", msg))

            # tables + migrations
            present = set(await conn.run_sync(lambda c: sa_inspect(c).get_table_names()))
            migrations: list[str] = []
            if "schema_migrations" in present:
                migrations = [str(r[0]) for r in (await conn.execute(text("SELECT version FROM schema_migrations ORDER BY version"))).all()]
            missing = [t for t in DOMAIN_TABLES if t not in present]
            total = len(DOMAIN_TABLES)
            if not missing:
                out.append(ok("DB tables", f"all {total} domain tables present"))
            elif migrations:
                out.append(fail("DB tables", f"{len(missing)} of {total} missing: {', '.join(missing)}"))
            else:
                out.append(warn("DB tables", f"{len(missing)} of {total} missing: {', '.join(missing)} (not created yet)"))
            if migrations:
                out.append(ok("DB migrations", f"{len(migrations)} applied (latest {migrations[-1]})"))
            else:
                out.append(warn("DB migrations", "migrations not applied yet; the app applies them on first start"))

            # hypertables
            existing_ht = [t for t in HYPERTABLES if t in present]
            if dialect != "postgresql" or not has_ts:
                out.append(skip("Hypertables", "requires PostgreSQL with the timescaledb extension"))
            elif not existing_ht:
                out.append(skip("Hypertables", "time-series tables do not exist yet"))
            else:
                hts = {str(r[0]) for r in (await conn.execute(text(
                    "SELECT hypertable_name FROM timescaledb_information.hypertables"))).all()}
                not_ht = [t for t in existing_ht if t not in hts]
                if not not_ht:
                    out.append(ok("Hypertables", ", ".join(existing_ht)))
                else:
                    out.append(warn("Hypertables", f"plain tables, not hypertables: {', '.join(not_ht)}", live_fail=True))

            # kill switch
            if "kill_switch_state" not in present:
                out.append(skip("Kill switch", "table does not exist yet"))
            else:
                row = (await conn.execute(text("SELECT engaged, reason FROM kill_switch_state WHERE id = 1"))).first()
                if row is None:
                    out.append(warn("Kill switch", "state row missing; the app creates it on first start"))
                elif _flag(row[0]):
                    out.append(fail("Kill switch", f"ENGAGED - trading would be blocked. Reason: {row[1] or '(none recorded)'}"))
                else:
                    out.append(ok("Kill switch", "not engaged"))
    except Exception as exc:  # noqa: BLE001
        out.append(fail("Database", f"cannot use {shown}: {_exc_text(exc)}"))
    finally:
        await engine.dispose()
    return out


# --------------------------------------------------------------------------------------- 4. redis (RESP)
def _resp(*parts: str) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        b = p.encode()
        out.append(b"$%d\r\n%s\r\n" % (len(b), b))
    return b"".join(out)


async def redis_probe(url: str) -> tuple[float, str]:
    """Return (PING latency ms, 'host:port/db'). Raises on any failure. Minimal RESP client."""
    u = urlparse(url)
    if u.scheme not in ("redis", "rediss"):
        raise ValueError(f"unsupported REDIS_URL scheme {u.scheme!r}")
    host, port = u.hostname or "localhost", u.port or 6379
    db = int(u.path.strip("/") or 0)
    sslctx = ssl.create_default_context() if u.scheme == "rediss" else None
    reader, writer = await asyncio.open_connection(host, port, ssl=sslctx)
    try:
        async def command(*parts: str) -> bytes:
            writer.write(_resp(*parts))
            await writer.drain()
            line = await reader.readline()
            if not line:
                raise ConnectionError("connection closed by server")
            if line.startswith(b"-"):
                raise RuntimeError(line[1:].decode(errors="replace").strip())
            return line.strip()

        if u.password:
            pw = unquote(u.password)
            await command(*(("AUTH", unquote(u.username), pw) if u.username else ("AUTH", pw)))
        if db:
            await command("SELECT", str(db))
        t0 = time.perf_counter()
        reply = await command("PING")
        latency = (time.perf_counter() - t0) * 1000
        if reply != b"+PONG":
            raise RuntimeError(f"unexpected PING reply {reply[:40]!r}")
        return latency, f"{host}:{port}/{db}"
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


async def check_redis(ctx: Ctx) -> CheckResult:
    s = ctx.settings
    assert s is not None
    try:
        latency, where = await redis_probe(s.redis_url)
    except Exception as exc:  # noqa: BLE001
        return warn("Redis", f"unreachable ({_exc_text(exc)}); optional in this version")
    return ok("Redis", f"{where} PONG in {latency:.1f} ms")


# --------------------------------------------------------------------------------------- 5. exchange public
async def check_exchange_public(ctx: Ctx) -> list[CheckResult]:
    s = ctx.settings
    assert s is not None
    ex_id = s.exchange_id
    testnet = s.uses_exchange_testnet
    market_kind = s.exchange_market.upper()
    ex = ctx.make_exchange(ex_id, keys=False, testnet=testnet)
    where = f"{ex_id}{' testnet' if testnet else ''} {market_kind}"
    out: list[CheckResult] = []

    markets: dict[str, Any] | None = None
    try:
        t0 = time.perf_counter()
        markets = await ex.load_markets()
        out.append(ok("Exchange markets", f"{where}: {len(markets)} markets loaded in {(time.perf_counter() - t0) * 1000:.0f} ms"))
    except Exception as exc:  # noqa: BLE001
        out.append(fail("Exchange markets", f"{where}: load_markets failed - {_exc_text(exc)}"))

    if markets is None:
        out.append(skip("Symbols", "markets not loaded"))
    else:
        unknown, inactive, wrong_type = [], [], []
        for sym in s.symbols:
            m = markets.get(sym)
            if m is None:
                unknown.append(sym)
                continue
            if m.get("active") is False:
                inactive.append(sym)
            if (market_kind == "PERPETUAL" and m.get("swap") is False) or (market_kind == "SPOT" and m.get("spot") is False):
                wrong_type.append(sym)
        if unknown or inactive or wrong_type:
            parts = []
            if unknown:
                parts.append(f"unknown: {', '.join(unknown)}")
            if inactive:
                parts.append(f"inactive: {', '.join(inactive)}")
            if wrong_type:
                parts.append(f"not a {market_kind} market: {', '.join(wrong_type)}")
            hint = " (perpetuals look like BTC/USDT:USDT, spot like BTC/USDT)"
            out.append(fail("Symbols", "; ".join(parts) + hint))
        else:
            out.append(ok("Symbols", f"{len(s.symbols)} configured symbols exist and are active"))

    if markets is None or not s.symbols:
        out.append(skip("Ticker", "needs loaded markets and at least one symbol"))
    else:
        try:
            t0 = time.perf_counter()
            tk = await ex.fetch_ticker(s.symbols[0])
            last = tk.get("last") if isinstance(tk, dict) else None
            out.append(ok("Ticker", f"{s.symbols[0]} last={last} in {(time.perf_counter() - t0) * 1000:.0f} ms"))
        except Exception as exc:  # noqa: BLE001
            out.append(fail("Ticker", f"fetch_ticker({s.symbols[0]}) failed - {_exc_text(exc)}"))

    try:
        before = time.time() * 1000
        server = await ex.fetch_time()
        after = time.time() * 1000
        drift = float(server) - (before + after) / 2
        rtt = after - before
        text = f"server clock differs by {drift:+.0f} ms (round trip {rtt:.0f} ms)"
        if abs(drift) > CLOCK_FAIL_MS:
            out.append(fail("Clock drift", f"{text}; signed requests would be rejected - sync the system clock (NTP)"))
        elif abs(drift) > CLOCK_WARN_MS:
            out.append(warn("Clock drift", f"{text}; > {CLOCK_WARN_MS} ms, sync the system clock (NTP)", live_fail=True))
        else:
            out.append(ok("Clock drift", text))
    except Exception as exc:  # noqa: BLE001
        out.append(fail("Clock drift", f"fetch_time failed - {_exc_text(exc)}"))
    return out


# --------------------------------------------------------------------------------------- 6. exchange private
def _quote_balance(bal: Any, quote: str) -> tuple[Decimal | None, Decimal | None]:
    total = free = None
    if isinstance(bal, dict):
        entry = bal.get(quote)
        if isinstance(entry, dict):
            total, free = _dec(entry.get("total")), _dec(entry.get("free"))
        if total is None:
            total = _dec((bal.get("total") or {}).get(quote)) if isinstance(bal.get("total"), dict) else None
            free = _dec((bal.get("free") or {}).get(quote)) if isinstance(bal.get("free"), dict) else None
    return total, free


def _key_label(ex_id: str) -> str:
    return "Binance API key" if ex_id.startswith("binance") else f"{ex_id.capitalize()} API key"


def _perm_failure(ctx: Ctx, label: str, exc: BaseException) -> CheckResult:
    auth = type(exc).__name__ in ("AuthenticationError", "PermissionDenied", "AccountSuspended")
    base = "key rejected by the exchange" if auth else "could not verify permissions (withdrawals must be disabled)"
    text = f"{base} - {_exc_text(exc)}"
    return fail(label, text) if (auth or ctx.live) else warn(label, text)


async def _binance_permissions(ctx: Ctx) -> list[CheckResult]:
    s = ctx.settings
    assert s is not None
    label = "Binance API key"
    if s.uses_exchange_testnet:
        return [skip(label, "permission endpoint /sapi/v1/account/apiRestrictions does not exist on the testnet")]
    ex = ctx.make_exchange("binance", keys=True, testnet=False, market="SPOT")  # sapi lives on the spot host
    try:
        r = await ex.sapi_get_account_apirestrictions()
    except Exception as exc:  # noqa: BLE001
        return [_perm_failure(ctx, label, exc)]
    if not isinstance(r, dict):
        return [warn(label, "unexpected apiRestrictions response", live_fail=True)]
    out: list[CheckResult] = []
    w = _flag(r.get("enableWithdrawals"))
    if w is True:
        out.append(fail(label, 'withdrawals ENABLED — disable "Enable Withdrawals" on the key'))
    elif w is False:
        out.append(ok(label, "withdrawals disabled"))
    else:
        out.append(warn(label, "withdrawal flag missing from response; verify manually", live_fail=True))
    if s.exchange_market.upper() == "PERPETUAL":
        out.append(ok(label, "futures trading enabled") if _flag(r.get("enableFutures")) is True
                   else fail(label, 'futures NOT enabled — tick "Enable Futures" on the key'))
    out.append(ok(label, "reading enabled") if _flag(r.get("enableReading")) is True
               else fail(label, "reading NOT enabled — tick \"Enable Reading\" on the key"))
    ip = _flag(r.get("ipRestrict"))
    if ip is True:
        out.append(ok(label, "IP whitelist active"))
    else:
        out.append(warn(label, "no IP restriction (ipRestrict=false) — IP whitelist önerilir", live_fail=True))
    return out


async def _bybit_permissions(ctx: Ctx, ex: Any) -> list[CheckResult]:
    label = "Bybit API key"
    try:
        r = await ex.private_get_v5_user_query_api()
    except Exception as exc:  # noqa: BLE001
        return [_perm_failure(ctx, label, exc)]
    res = r.get("result") if isinstance(r, dict) else None
    if not isinstance(res, dict):
        return [warn(label, "unexpected query-api response; verify manually", live_fail=True)]
    out: list[CheckResult] = []
    perms = res.get("permissions") or {}
    wallet = [str(x) for x in (perms.get("Wallet") or [])]
    if any(x.lower() == "withdraw" for x in wallet):
        out.append(fail(label, "Withdraw permission ENABLED — remove 'Withdraw' from the key's Wallet permissions"))
    else:
        out.append(ok(label, "no withdraw permission"))
    ro = res.get("readOnly")
    if ro is not None and int(_dec(ro) or 0) != 0:
        out.append(fail(label, "key is READ-ONLY (readOnly=1); trading needs write access"))
    else:
        out.append(ok(label, "key has trade (write) access"))
    ips = [str(x) for x in (res.get("ips") or [])]
    if not ips or ips == ["*"]:
        out.append(warn(label, "no IP restriction (ips empty or '*') — IP whitelist önerilir", live_fail=True))
    else:
        out.append(ok(label, f"IP whitelist active ({len(ips)} IPs)"))
    return out


def _fmt_position(p: dict[str, Any]) -> str:
    size = p.get("contracts") if p.get("contracts") is not None else p.get("contractSize")
    lev = p.get("leverage")
    return f"{p.get('symbol')} {p.get('side') or ''} {size} x{lev if lev is not None else '?'} {p.get('marginMode') or '?'}".replace("  ", " ")


async def _open_positions(ctx: Ctx, ex: Any) -> CheckResult:
    s = ctx.settings
    assert s is not None
    if s.exchange_market.upper() != "PERPETUAL":
        return skip("Open positions", "spot market has no positions")
    try:
        raw = await ex.fetch_positions()
    except Exception as exc:  # noqa: BLE001
        text = f"fetch_positions failed - {_exc_text(exc)}"
        return fail("Open positions", text) if ctx.mode is not TradingMode.PAPER else warn("Open positions", text)
    open_pos = [p for p in (raw or []) if isinstance(p, dict) and abs(_dec(p.get("contracts")) or 0) > 0]
    if not open_pos:
        return ok("Open positions", "none")
    bad = [p for p in open_pos if str(p.get("marginMode") or "").lower() == "cross"
           or (_dec(p.get("leverage")) or 0) > MAX_POSITION_LEVERAGE]
    shown = "; ".join(_fmt_position(p) for p in open_pos[:5]) + (f"; +{len(open_pos) - 5} more" if len(open_pos) > 5 else "")
    if bad:
        text = (f"{len(open_pos)} open: {shown} — CROSS margin or leverage > {MAX_POSITION_LEVERAGE} "
                f"in: {', '.join(str(p.get('symbol')) for p in bad)}; the bot only runs isolated <= 5x")
        return fail("Open positions", text) if ctx.live else warn("Open positions", text)
    return ok("Open positions", f"{len(open_pos)} open (isolated, leverage <= {MAX_POSITION_LEVERAGE}): {shown}")


async def check_exchange_private(ctx: Ctx) -> list[CheckResult]:
    s = ctx.settings
    assert s is not None
    if ctx.mode is TradingMode.PAPER and not ctx.has_keys():
        return [skip("Exchange (private)", "PAPER mode without keys - nothing to verify")]
    if not ctx.has_keys():
        return [fail("Exchange (private)", "API key/secret missing")]
    ex_id = s.exchange_id
    ex = ctx.make_exchange(ex_id, keys=True, testnet=s.uses_exchange_testnet)
    out: list[CheckResult] = []
    key = _secret(s.exchange_api_key)

    try:
        t0 = time.perf_counter()
        bal = await ex.fetch_balance()
        ms = (time.perf_counter() - t0) * 1000
        total, free = _quote_balance(bal, s.quote_currency)
        if total is None:
            out.append(warn("Exchange balance", f"authenticated ({mask(key)}) but no {s.quote_currency} balance reported"))
        else:
            ctx.state["balance_total"] = total
            line = f"{s.quote_currency} total {total} / free {free if free is not None else '?'} (key {mask(key)}, {ms:.0f} ms)"
            out.append(warn("Exchange balance", line + " - zero balance") if total == 0 else ok("Exchange balance", line))
    except Exception as exc:  # noqa: BLE001
        text = f"fetch_balance failed (key {mask(key)}) - {_exc_text(exc)}"
        out.append(warn("Exchange balance", text) if ctx.mode is TradingMode.PAPER else fail("Exchange balance", text))

    if ex_id in ("binance", "binanceusdm"):
        out.extend(await _binance_permissions(ctx))
    elif ex_id == "bybit":
        out.extend(await _bybit_permissions(ctx, ex))
    else:
        out.append(warn(f"{ex_id.capitalize()} API key",
                        f"permission check not implemented for {ex_id}; verify manually that withdrawals are disabled",
                        live_fail=True))
    out.append(await _open_positions(ctx, ex))
    return out


# --------------------------------------------------------------------------------------- 7. claude
def _claude_error(ctx: Ctx, exc: BaseException, what: str) -> CheckResult:
    import anthropic

    model = ctx.settings.claude.model if ctx.settings else "?"
    strict = ctx.mode is not TradingMode.PAPER
    if isinstance(exc, anthropic.AuthenticationError):
        return fail(what, "invalid key (401) - check ANTHROPIC_API_KEY")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return fail(what, f"key has no permission for this request (403): {_exc_text(exc, 160)}")
    if isinstance(exc, anthropic.NotFoundError):
        return fail(what, f"model id not found: set CLAUDE_MODEL (currently {model!r})")
    if isinstance(exc, anthropic.RateLimitError):
        return warn(what, "rate limited (429): key is valid but decisions may be throttled")
    if isinstance(exc, anthropic.APITimeoutError):
        text = f"timed out after {ctx.settings.claude.timeout_seconds}s (decision budget)" if ctx.settings else "timed out"
        return fail(what, text) if strict else warn(what, text)
    if isinstance(exc, anthropic.APIConnectionError):
        text = f"cannot reach the Claude API: {_exc_text(exc, 160)}"
        return fail(what, text) if strict else warn(what, text)
    text = f"API error: {_exc_text(exc, 200)}"
    return fail(what, text) if strict else warn(what, text)


async def check_claude(ctx: Ctx) -> list[CheckResult]:
    s = ctx.settings
    assert s is not None
    key = _secret(s.anthropic_api_key)
    if not key:
        if ctx.mode is TradingMode.PAPER:
            return [warn("Claude API", "ANTHROPIC_API_KEY missing - every decision would be HOLD")]
        return [fail("Claude API", "ANTHROPIC_API_KEY missing - every decision would be HOLD")]
    client = ctx.track(ctx.anthropic_factory(key, s.claude.timeout_seconds))
    out: list[CheckResult] = []
    try:
        t0 = time.perf_counter()
        await client.models.retrieve(s.claude.model)
        ms = (time.perf_counter() - t0) * 1000
        text = f"key {mask(key)} valid, model {s.claude.model} found in {ms:.0f} ms"
        if ms > CLAUDE_SLOW_MS:
            out.append(warn("Claude API", f"{text}; slow vs the {s.claude.timeout_seconds}s decision budget", live_fail=True))
        else:
            out.append(ok("Claude API", text))
    except Exception as exc:  # noqa: BLE001
        return [_claude_error(ctx, exc, "Claude API")]
    if ctx.claude_roundtrip:
        out.append(await _claude_roundtrip(ctx, client))
    return out


async def _claude_roundtrip(ctx: Ctx, client: Any) -> CheckResult:
    from app.claude.tool import TOOL_NAME, decision_tool_definition, parse_tool_input

    s = ctx.settings
    assert s is not None
    prompt = (
        "PRE-FLIGHT SELF TEST (synthetic, no real market). Proposed signal: LONG TESTUSDT at 100.0, ATR 1.0. "
        "Submit your decision with the tool. Answer HOLD, leverage 1, entry_type MARKET, no prices, and a short "
        "Turkish explanation of at least 20 characters in reasoning_tr."
    )
    try:
        t0 = time.perf_counter()
        resp = await client.messages.create(
            model=s.claude.model, max_tokens=300, tools=[decision_tool_definition()],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": prompt}],
        )
        ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        return _claude_error(ctx, exc, "Claude round-trip")
    block = next((b for b in (getattr(resp, "content", None) or [])
                  if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == TOOL_NAME), None)
    if block is None:
        return fail("Claude round-trip", f"no {TOOL_NAME} tool_use block returned (stop_reason={getattr(resp, 'stop_reason', '?')})")
    try:
        decision = parse_tool_input(dict(block.input))
    except Exception as exc:  # noqa: BLE001
        return fail("Claude round-trip", f"invalid ClaudeDecision: {_exc_text(exc, 200)}")
    text = f"valid ClaudeDecision ({decision.action.value}) in {ms:.0f} ms"
    if ms > s.claude.timeout_seconds * 1000 * 0.75:
        return warn("Claude round-trip", f"{text}; close to the {s.claude.timeout_seconds}s budget", live_fail=True)
    return ok("Claude round-trip", text)


# --------------------------------------------------------------------------------------- 8. risk limits
def check_risk_limits(ctx: Ctx) -> CheckResult:
    s = ctx.settings
    assert s is not None
    rules: list[tuple[str, Any, str, Any]] = [
        ("RISK_MAX_LEVERAGE", s.risk.max_leverage, "<=", 5),
        ("RISK_MAX_DAILY_DRAWDOWN_PCT", s.risk.max_daily_drawdown_pct, "<=", "0.03"),
        ("RISK_MAX_RISK_PER_TRADE_PCT", s.risk.max_risk_per_trade_pct, "<=", "0.01"),
        ("RISK_MAX_OPEN_POSITIONS", s.risk.max_open_positions, "<=", 5),
        ("RISK_MIN_FEE_COVER_MULTIPLE", s.risk.min_fee_cover_multiple, ">=", 2),
        ("EXEC_STOP_PLACE_RETRIES", s.execution.stop_place_retries, ">=", 1),
        ("CLAUDE_TIMEOUT_SECONDS", s.claude.timeout_seconds, "<=", 10),
        ("DATA_STALE_AFTER_SECONDS", s.data.stale_after_seconds, "<=", 5),
    ]
    bad: list[str] = []
    for name, value, op, limit in rules:
        v, lim = Decimal(str(value)), Decimal(str(limit))
        good = v <= lim if op == "<=" else v >= lim
        if not good:
            bad.append(f"{name}={value} (must be {op} {limit})")
    if bad:
        return fail("Risk limits", "; ".join(bad))
    return ok("Risk limits", f"all {len(rules)} limits within safe bounds")


# --------------------------------------------------------------------------------------- 9. api exposure
def _in_docker() -> bool:
    return Path("/.dockerenv").exists() or bool(os.environ.get("RUNNING_IN_DOCKER"))


def check_api_exposure(ctx: Ctx) -> CheckResult:
    s = ctx.settings
    assert s is not None
    host = s.api.host.strip()
    token_len = len(_secret(s.api.control_token))
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        loopback = False
    if loopback:
        return ok("API exposure", f"{host}:{s.api.port} is loopback only")
    publish = os.environ.get("API_PUBLISH_HOST", "")
    if _in_docker() and host in ("0.0.0.0", "::"):
        if publish and ipaddress.ip_address(publish).is_loopback:
            return ok("API exposure", f"binds {host} inside Docker, published on {publish}")
        if token_len >= 16:
            return warn("API exposure", f"binds {host} inside Docker; make sure the port is published as 127.0.0.1:{s.api.port}:{s.api.port} "
                                        "(cannot be verified from inside the container)", live_fail=True)
        return fail("API exposure", f"binds {host} in Docker and API_CONTROL_TOKEN is not set (>= 16 chars)")
    if token_len < 16:
        return fail("API exposure", f"API_HOST={host} is reachable from the network and API_CONTROL_TOKEN is missing/short")
    return warn("API exposure", f"API_HOST={host} is not loopback; put a TLS reverse proxy in front", live_fail=True)


# --------------------------------------------------------------------------------------- 10. live balance floor
def check_live_balance(ctx: Ctx) -> list[CheckResult]:
    if not ctx.live:
        return []
    s = ctx.settings
    assert s is not None
    minimum = _dec(getattr(getattr(s, "recon", None), "live_min_balance", None))
    if minimum is None:
        return [skip("Live balance floor", "RECON_LIVE_MIN_BALANCE not configured in Settings")]
    total = ctx.state.get("balance_total")
    if total is None:
        return [skip("Live balance floor", "exchange balance unavailable")]
    if total < minimum:
        return [warn("Live balance floor", f"balance {total} {s.quote_currency} is below RECON_LIVE_MIN_BALANCE {minimum}")]
    return [ok("Live balance floor", f"balance {total} >= {minimum} {s.quote_currency}")]


# --------------------------------------------------------------------------------------- runner
@dataclass(frozen=True)
class _Plan:
    label: str
    fn: Callable[[Ctx], Any]
    needs_settings: bool = True
    network: bool = False
    extra_timeout: float = 0.0


PLAN: tuple[_Plan, ...] = (
    _Plan("Python", check_python, needs_settings=False),
    _Plan("Packages", check_packages, needs_settings=False),
    _Plan("Mode locks", check_mode_locks),
    _Plan("Database", check_database),
    _Plan("Redis", check_redis),
    _Plan("Exchange (public)", check_exchange_public, network=True),
    _Plan("Exchange (private)", check_exchange_private, network=True),
    _Plan("Claude API", check_claude, network=True, extra_timeout=0.0),
    _Plan("Risk limits", check_risk_limits),
    _Plan("API exposure", check_api_exposure),
    _Plan("Live balance floor", check_live_balance),
)


async def _run_one(ctx: Ctx, plan: _Plan) -> list[CheckResult]:
    t0 = time.perf_counter()
    timeout = ctx.check_timeout
    if plan.label == "Claude API" and ctx.settings is not None:
        timeout += 2 * ctx.settings.claude.timeout_seconds  # retrieve + optional round-trip, each bounded by the SDK timeout
    try:
        res = plan.fn(ctx)
        if asyncio.iscoroutine(res) or isinstance(res, Awaitable):
            res = await asyncio.wait_for(res, timeout)
        results = list(res) if isinstance(res, (list, tuple)) else [res]
    except asyncio.TimeoutError:
        results = [fail(plan.label, f"timed out after {timeout:g} s")]
    except Exception as exc:  # noqa: BLE001 - a check must never crash the run
        results = [fail(plan.label, f"check crashed: {_exc_text(exc)}")]
    ms = (time.perf_counter() - t0) * 1000
    return [dataclasses.replace(r, duration_ms=ms) for r in results]


def _finalise(ctx: Ctx, r: CheckResult) -> CheckResult:
    r = dataclasses.replace(r, name=ctx.redact(r.name), detail=ctx.redact(r.detail))
    if ctx.live and r.status is Status.WARN and r.live_fail:
        r = dataclasses.replace(r, status=Status.FAIL, escalated=True, detail=r.detail + " [WARN is FAIL in LIVE]")
    return r


async def run_preflight(
    settings_factory: Callable[[], Settings],
    mode: TradingMode | None = None,
    *,
    skip_network: bool = False,
    claude_roundtrip: bool = False,
    exchange_factory: ExchangeFactory | None = None,
    anthropic_factory: AnthropicFactory | None = None,
    db_url_override: str | None = None,
    check_timeout: float = DEFAULT_CHECK_TIMEOUT,
) -> PreflightReport:
    t_start = time.perf_counter()
    settings: Settings | None = None
    load_error: str | None = None
    try:
        settings = settings_factory()
    except (ConfigError, ValidationError) as exc:
        load_error = _validation_text(exc)
    except Exception as exc:  # noqa: BLE001
        load_error = _exc_text(exc, 600)

    eff_mode = mode or (settings.trading_mode if settings else TradingMode.PAPER)
    ctx = Ctx(settings, eff_mode, skip_network=skip_network, claude_roundtrip=claude_roundtrip,
              exchange_factory=exchange_factory, anthropic_factory=anthropic_factory,
              db_url_override=db_url_override, check_timeout=check_timeout)
    report = PreflightReport(mode=eff_mode.value)
    raw: list[CheckResult] = []
    if settings is None:
        settings_result = fail("Settings", f"cannot load/validate: {load_error}")
    elif settings.trading_mode is not eff_mode:
        settings_result = fail("Settings", f"loaded mode {settings.trading_mode.value} differs from requested {eff_mode.value}")
        ctx.settings = None
    else:
        settings_result = ok("Settings", f"loaded and validated (exchange {settings.exchange_id}/{settings.exchange_market}, "
                                         f"{len(settings.symbols)} symbols)")
    try:
        for plan in PLAN:
            if plan.needs_settings and ctx.settings is None:
                raw.append(skip(plan.label, "settings invalid"))
            elif plan.network and skip_network:
                msg = "skipped (--skip-network)"
                raw.append(warn(plan.label, msg + "; required for LIVE", live_fail=True) if ctx.live else skip(plan.label, msg))
            else:
                raw.extend(await _run_one(ctx, plan))
            if plan.label == "Packages":
                raw.append(settings_result)
    finally:
        await ctx.aclose()
    report.results = [_finalise(ctx, r) for r in raw]
    report.duration_ms = (time.perf_counter() - t_start) * 1000
    return report


# --------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.cli.preflight", description="Pre-flight safety check.")
    p.add_argument("--mode", choices=[m.value for m in TradingMode], type=str.upper, help="default: TRADING_MODE from settings")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.add_argument("--skip-network", action="store_true", help="skip exchange and Claude API checks (DB/Redis still run)")
    p.add_argument("--env-file", default=None, help="dotenv file to load first (real environment variables win)")
    p.add_argument("--claude-roundtrip", action="store_true", help="also run one forced tool_use call (costs a few tokens)")
    p.add_argument("--timeout", type=float, default=DEFAULT_CHECK_TIMEOUT, help="per-check timeout in seconds")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    if args.env_file:
        path = Path(args.env_file)
        if not path.is_file():
            print(f"env file not found: {path}", file=sys.stderr)
            return 2
        load_env_file(path)
    mode = TradingMode(args.mode) if args.mode else None

    def factory() -> Settings:
        return Settings(trading_mode=mode) if mode else Settings()

    if sys.platform == "win32":  # aiodns/aiohttp need the selector loop on Windows
        policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy is not None:
            asyncio.set_event_loop_policy(policy())

    report = asyncio.run(run_preflight(
        factory, mode, skip_network=args.skip_network, claude_roundtrip=args.claude_roundtrip,
        check_timeout=args.timeout,
    ))
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        colour = colour_enabled(sys.stdout)
        if colour:
            _enable_windows_vt()
        print(render_text(report, colour=colour))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
