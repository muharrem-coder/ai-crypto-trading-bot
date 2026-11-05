"""LIVE start-up guard: everything that must be true BEFORE real money is allowed to trade.

``verify_live_start`` runs every check (never short-circuits, never raises) and returns a ``LiveGuardReport``;
``enforce_live_guard`` raises ``LiveGuardError`` listing every failed check. The runtime calls the latter when
``TRADING_MODE=LIVE`` (and it can be used as a dry run in TESTNET).

Checks
------
* the Settings locks (confirmation phrase, API keys, control token, not testnet) are re-asserted
* ``fetch_account`` works and equity >= ``recon.live_min_balance`` (quote currency)
* every configured symbol: the instrument exists, ISOLATED margin could be set ("already isolated" answers count
  as success) and leverage ``min(risk.max_leverage, 5)`` could be set
* every position already on the exchange is ISOLATED with leverage <= ``recon.live_max_existing_leverage``
* risk limits are sane: ``max_leverage <= 5`` and ``max_daily_drawdown_pct <= 0.03``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from app.brokers.base import BaseBrokerAdapter
from app.config.settings import LIVE_CONFIRMATION_PHRASE, Settings
from app.core.types import MarginMode, Position, TradingMode

log = logging.getLogger(__name__)

MAX_LIVE_LEVERAGE = 5
MAX_LIVE_DAILY_DRAWDOWN = Decimal("0.03")
_NOOP_MARGIN_MARKERS = ("no need", "-4046", "not modified", "not changed", "already")


class LiveGuardError(RuntimeError):
    """LIVE start refused; the message lists every failed check."""

    def __init__(self, message: str, report: "LiveGuardReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(slots=True)
class LiveGuardReport:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append((name, ok, detail))
        return ok

    def summary(self) -> str:
        return "; ".join(f"{name}: {detail}" for name, ok, detail in self.checks if not ok) or "all checks passed"


def _is_noop_margin_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _NOOP_MARGIN_MARKERS)


def _check_locks(settings: Settings, report: LiveGuardReport) -> None:
    mode = settings.trading_mode
    report.add("trading_mode", mode in (TradingMode.LIVE, TradingMode.TESTNET), f"mode={mode.value}")
    has_keys = bool(settings.exchange_api_key.get_secret_value() and settings.exchange_api_secret.get_secret_value())
    report.add("api_keys", has_keys, "exchange API key/secret present" if has_keys else "EXCHANGE_API_KEY/SECRET missing")
    if mode is TradingMode.LIVE:
        phrase_ok = settings.live_trading_confirmation == LIVE_CONFIRMATION_PHRASE
        report.add("confirmation_phrase", phrase_ok, "ok" if phrase_ok else "LIVE_TRADING_CONFIRMATION is wrong")
        token_ok = len(settings.api.control_token.get_secret_value()) >= 16
        report.add("control_token", token_ok, "ok" if token_ok else "API_CONTROL_TOKEN shorter than 16 chars")
        report.add("not_testnet", not settings.exchange_testnet,
                   "ok" if not settings.exchange_testnet else "EXCHANGE_TESTNET=true in LIVE mode")


def _check_risk_limits(settings: Settings, report: LiveGuardReport) -> None:
    lev = settings.risk.max_leverage
    report.add("risk_max_leverage", lev <= MAX_LIVE_LEVERAGE, f"RISK_MAX_LEVERAGE={lev} (limit {MAX_LIVE_LEVERAGE})")
    dd = settings.risk.max_daily_drawdown_pct
    report.add("risk_daily_drawdown", dd <= MAX_LIVE_DAILY_DRAWDOWN,
               f"RISK_MAX_DAILY_DRAWDOWN_PCT={dd} (limit {MAX_LIVE_DAILY_DRAWDOWN})")


async def _check_account(broker: BaseBrokerAdapter, settings: Settings, report: LiveGuardReport) -> None:
    minimum = settings.recon.live_min_balance
    try:
        account = await broker.fetch_account()
    except Exception as exc:  # noqa: BLE001
        report.add("account", False, f"fetch_account failed: {exc!r}")
        return
    report.add("account", True, "fetch_account ok")
    report.add("min_balance", account.equity >= minimum,
               f"equity {account.equity} {settings.quote_currency} (minimum {minimum})")


async def _check_positions(broker: BaseBrokerAdapter, settings: Settings, report: LiveGuardReport) -> list[Position]:
    try:
        positions = list(await broker.fetch_positions())
    except Exception as exc:  # noqa: BLE001
        report.add("existing_positions", False, f"fetch_positions failed: {exc!r}")
        return []
    max_lev = settings.recon.live_max_existing_leverage
    bad: list[str] = []
    for p in positions:
        if p.margin_mode is not MarginMode.ISOLATED:
            bad.append(f"{p.symbol} margin={p.margin_mode.value}")
        if p.leverage > max_lev:
            bad.append(f"{p.symbol} leverage={p.leverage}>{max_lev}")
    report.add("existing_positions", not bad,
               f"{len(positions)} open position(s) ok" if not bad else "; ".join(bad))
    return positions


async def _check_symbols(broker: BaseBrokerAdapter, settings: Settings, positions: list[Position],
                         report: LiveGuardReport) -> None:
    leverage = min(settings.risk.max_leverage, MAX_LIVE_LEVERAGE)
    isolated_open = {p.symbol for p in positions if p.margin_mode is MarginMode.ISOLATED}
    for symbol in settings.symbols:
        try:
            await broker.get_instrument(symbol)
        except Exception as exc:  # noqa: BLE001
            report.add(f"instrument:{symbol}", False, f"unknown instrument: {exc!r}")
            continue
        report.add(f"instrument:{symbol}", True, "exists")
        if symbol in isolated_open:  # exchanges refuse margin-type changes with an open position; already isolated
            report.add(f"margin_mode:{symbol}", True, "open position already ISOLATED")
        else:
            try:
                await broker.set_margin_mode(symbol, MarginMode.ISOLATED)
                report.add(f"margin_mode:{symbol}", True, "ISOLATED set")
            except Exception as exc:  # noqa: BLE001
                if _is_noop_margin_error(exc):
                    report.add(f"margin_mode:{symbol}", True, "already ISOLATED")
                else:
                    report.add(f"margin_mode:{symbol}", False, f"set_margin_mode failed: {exc!r}")
        try:
            await broker.set_leverage(symbol, leverage)
            report.add(f"leverage:{symbol}", True, f"{leverage}x set")
        except Exception as exc:  # noqa: BLE001
            report.add(f"leverage:{symbol}", False, f"set_leverage({leverage}) failed: {exc!r}")


async def verify_live_start(broker: BaseBrokerAdapter, settings: Settings) -> LiveGuardReport:
    """Run every check and report. Never raises (a crashing check becomes a failed check)."""
    report = LiveGuardReport()
    for name, sync_check in (("locks", _check_locks), ("risk_limits", _check_risk_limits)):
        try:
            sync_check(settings, report)
        except Exception as exc:  # noqa: BLE001
            report.add(name, False, f"check crashed: {exc!r}")
    try:
        await _check_account(broker, settings, report)
    except Exception as exc:  # noqa: BLE001
        report.add("account", False, f"check crashed: {exc!r}")
    try:
        positions = await _check_positions(broker, settings, report)
        await _check_symbols(broker, settings, positions, report)
    except Exception as exc:  # noqa: BLE001
        report.add("symbols", False, f"check crashed: {exc!r}")
    for name, ok, detail in report.checks:
        (log.info if ok else log.error)("live guard %s %s: %s", "PASS" if ok else "FAIL", name, detail)
    return report


async def enforce_live_guard(broker: BaseBrokerAdapter, settings: Settings) -> LiveGuardReport:
    """``verify_live_start`` and raise ``LiveGuardError`` (listing all failed checks) when not ok."""
    report = await verify_live_start(broker, settings)
    if not report.ok:
        lines = "\n".join(f"  - {name}: {detail}" for name, ok, detail in report.checks if not ok)
        raise LiveGuardError(f"LIVE start refused, {len(report.failed)} check(s) failed:\n{lines}", report)
    return report
