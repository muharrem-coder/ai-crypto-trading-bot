"""LIVE start-up guard against the real paper broker (no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal as D
from typing import Any

import pytest
from pydantic import SecretStr

from app.brokers.paper_adapter import PaperBrokerAdapter
from app.config.settings import (
    LIVE_CONFIRMATION_PHRASE,
    ApiSettings,
    ReconciliationSettings,
    RiskSettings,
    Settings,
)
from app.core.errors import BrokerError
from app.core.types import MarginMode, OrderRequest, OrderType, Position, Side, TradingMode
from app.execution.live_guard import LiveGuardError, enforce_live_guard, verify_live_start
from app.risk.testing import make_spec

SYM = "BTC/USDT:USDT"
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def live_settings(**kw: Any) -> Settings:
    kw.setdefault("symbols", [SYM])
    # explicit values: immune to RISK_* variables leaked into os.environ by other test modules
    kw.setdefault("risk", RiskSettings(max_leverage=5, max_daily_drawdown_pct=D("0.03")))
    return Settings(_env_file=None, trading_mode=TradingMode.LIVE,
                    live_trading_confirmation=LIVE_CONFIRMATION_PHRASE, exchange_api_key=SecretStr("key"),
                    exchange_api_secret=SecretStr("secret"), api=ApiSettings(control_token=SecretStr("t" * 16)),
                    **kw)


class GuardPaper(PaperBrokerAdapter):
    margin_error: Exception | None = None
    leverage_error: Exception | None = None
    cross = False
    calls: list[tuple[str, Any]]

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        self.calls = getattr(self, "calls", [])
        self.calls.append(("margin", (symbol, mode)))
        if self.margin_error is not None:
            raise self.margin_error
        await super().set_margin_mode(symbol, mode)

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        self.calls = getattr(self, "calls", [])
        self.calls.append(("leverage", (symbol, leverage)))
        if self.leverage_error is not None:
            raise self.leverage_error
        await super().set_leverage(symbol, leverage)

    async def fetch_positions(self) -> list[Position]:
        out = await super().fetch_positions()
        if self.cross:
            for p in out:
                p.margin_mode = MarginMode.CROSS
        return out


def make_paper(balance: str = "1000") -> GuardPaper:
    paper = GuardPaper({SYM: make_spec(SYM)}, D(balance), "USDT", D(0), now=lambda: NOW)
    paper.update_price(SYM, D("49999.9"), D("50000.1"), mark=D("50000"))
    return paper


def failed_names(report) -> set[str]:
    return {name for name, ok, _ in report.checks if not ok}


async def test_healthy_paper_broker_with_live_shaped_settings_passes():
    paper = make_paper()
    report = await verify_live_start(paper, live_settings())
    assert report.ok, report.summary()
    names = {name for name, _, _ in report.checks}
    assert {"confirmation_phrase", "api_keys", "control_token", "account", "min_balance", f"instrument:{SYM}",
            f"margin_mode:{SYM}", f"leverage:{SYM}", "risk_max_leverage", "risk_daily_drawdown",
            "existing_positions"} <= names
    assert ("margin", (SYM, MarginMode.ISOLATED)) in paper.calls
    assert ("leverage", (SYM, 5)) in paper.calls  # min(risk.max_leverage=5, 5)
    assert (await enforce_live_guard(paper, live_settings())).ok


async def test_testnet_dry_run_is_supported():
    settings = Settings(_env_file=None, symbols=[SYM], trading_mode=TradingMode.TESTNET,
                        exchange_api_key=SecretStr("k"), exchange_api_secret=SecretStr("s"))
    report = await verify_live_start(make_paper(), settings)
    assert report.ok, report.summary()
    assert "confirmation_phrase" not in {n for n, _, _ in report.checks}  # LIVE-only locks


async def test_low_balance_fails():
    report = await verify_live_start(make_paper("10"), live_settings())
    assert failed_names(report) == {"min_balance"}
    assert "10" in dict((n, d) for n, _, d in report.checks)["min_balance"]


async def test_balance_threshold_comes_from_recon_settings():
    settings = live_settings(recon=ReconciliationSettings(live_min_balance=D("5000")))
    assert failed_names(await verify_live_start(make_paper("1000"), settings)) == {"min_balance"}


async def test_margin_mode_failure_fails():
    paper = make_paper()
    paper.margin_error = BrokerError("exchange said no")
    report = await verify_live_start(paper, live_settings())
    assert failed_names(report) == {f"margin_mode:{SYM}"}


@pytest.mark.parametrize("message", ["No need to change margin type.", "binance {\"code\":-4046}", "already isolated"])
async def test_already_isolated_response_counts_as_success(message):
    paper = make_paper()
    paper.margin_error = BrokerError(message)
    report = await verify_live_start(paper, live_settings())
    assert report.ok, report.summary()


async def test_leverage_failure_fails():
    paper = make_paper()
    paper.leverage_error = BrokerError("leverage rejected")
    assert failed_names(await verify_live_start(paper, live_settings())) == {f"leverage:{SYM}"}


async def test_unknown_symbol_fails():
    settings = live_settings(symbols=[SYM, "DOGE/USDT:USDT"])
    report = await verify_live_start(make_paper(), settings)
    assert failed_names(report) == {"instrument:DOGE/USDT:USDT"}


async def test_existing_cross_position_fails():
    paper = make_paper()
    await paper.create_order(OrderRequest(SYM, Side.BUY, OrderType.MARKET, D("0.01")))
    paper.cross = True
    report = await verify_live_start(paper, live_settings())
    assert "existing_positions" in failed_names(report)
    assert "CROSS" in dict((n, d) for n, _, d in report.checks)["existing_positions"]


async def test_existing_position_with_too_high_leverage_fails():
    paper = make_paper()
    await paper.set_leverage(SYM, 8)
    await paper.create_order(OrderRequest(SYM, Side.BUY, OrderType.MARKET, D("0.01"), leverage=8))
    report = await verify_live_start(paper, live_settings())
    assert failed_names(report) == {"existing_positions"}


async def test_existing_isolated_low_leverage_position_passes_and_margin_is_not_touched():
    paper = make_paper()
    await paper.create_order(OrderRequest(SYM, Side.BUY, OrderType.MARKET, D("0.01"), leverage=3))
    paper.calls.clear()
    report = await verify_live_start(paper, live_settings())
    assert report.ok, report.summary()
    assert not any(kind == "margin" for kind, _ in paper.calls)  # exchanges refuse margin changes with a position


async def test_risk_leverage_above_five_fails_even_though_applied_leverage_is_capped():
    paper = make_paper()
    report = await verify_live_start(paper, live_settings(risk=RiskSettings(max_leverage=10)))
    assert failed_names(report) == {"risk_max_leverage"}
    assert ("leverage", (SYM, 5)) in paper.calls


async def test_daily_drawdown_above_three_percent_fails():
    report = await verify_live_start(make_paper(), live_settings(risk=RiskSettings(max_daily_drawdown_pct=D("0.05"))))
    assert failed_names(report) == {"risk_daily_drawdown"}


async def test_account_fetch_failure_fails_without_raising():
    class Down(GuardPaper):
        async def fetch_account(self):
            raise BrokerError("unreachable")

    paper = Down({SYM: make_spec(SYM)}, D(1000), "USDT", D(0), now=lambda: NOW)
    report = await verify_live_start(paper, live_settings())
    assert "account" in failed_names(report)


async def test_paper_mode_settings_fail_the_lock_checks():
    report = await verify_live_start(make_paper(), Settings(_env_file=None, symbols=[SYM]))
    assert {"trading_mode", "api_keys"} <= failed_names(report)


async def test_enforce_raises_listing_every_failed_check():
    paper = make_paper("10")
    paper.margin_error = BrokerError("nope")
    settings = live_settings(risk=RiskSettings(max_leverage=10))
    with pytest.raises(LiveGuardError) as err:
        await enforce_live_guard(paper, settings)
    text = str(err.value)
    assert "min_balance" in text and f"margin_mode:{SYM}" in text and "risk_max_leverage" in text
    assert "3 check(s) failed" in text
    assert err.value.report is not None and not err.value.report.ok
    assert isinstance(err.value, RuntimeError)
