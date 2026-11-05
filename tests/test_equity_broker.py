"""Alpaca/IBKR Phase 2 stubs: real NYSE hours + PDT logic, disabled I/O. Plus factory wiring (no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal as D
from zoneinfo import ZoneInfo

import pytest

from app.brokers.alpaca_adapter import AlpacaBrokerAdapter, IBKRBrokerAdapter, PDTTracker, _window_start
from app.brokers.factory import create_broker, create_broker_async
from app.brokers.paper_adapter import PaperBrokerAdapter
from app.brokers.ccxt_adapter import CCXTBrokerAdapter
from app.config.settings import ApiSettings, LIVE_CONFIRMATION_PHRASE, Settings
from app.core.errors import BrokerNotSupportedError, ConfigError, RiskViolation
from app.core.types import AssetClass, MarketType, OrderRequest, OrderType, Side, TradingMode
from datetime import date

NY = ZoneInfo("America/New_York")


def ny(y, m, d, h, mi) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=NY).astimezone(timezone.utc)


def adapter_at(dt: datetime, equity="10000", cls=AlpacaBrokerAdapter):
    return cls(now=lambda: dt, equity=D(equity))


def test_market_hours_regular_session():
    a = adapter_at(ny(2026, 1, 5, 10, 0))  # Monday
    s = a.market_hours_status()
    assert s.is_open and s.session == "REGULAR" and a.is_market_open()
    assert s.next_change == ny(2026, 1, 5, 16, 0)


@pytest.mark.parametrize(
    "dt",
    [ny(2026, 1, 5, 9, 29), ny(2026, 1, 5, 16, 0), ny(2026, 1, 3, 11, 0), ny(2026, 1, 4, 11, 0)],
)
def test_market_closed(dt):
    a = adapter_at(dt)
    assert not a.is_market_open() and a.market_hours_status().session == "CLOSED"


def test_next_open_skips_weekend_and_handles_dst():
    assert adapter_at(ny(2026, 1, 9, 17, 0)).market_hours_status().next_change == ny(2026, 1, 12, 9, 30)  # Fri -> Mon
    assert adapter_at(ny(2026, 1, 5, 8, 0)).market_hours_status().next_change == ny(2026, 1, 5, 9, 30)
    # summer (EDT): 09:30 NY is 13:30 UTC
    assert adapter_at(ny(2026, 7, 6, 13, 0)).is_market_open()
    assert adapter_at(datetime(2026, 7, 6, 13, 29, tzinfo=timezone.utc)).is_market_open() is False
    assert adapter_at(datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc)).is_market_open() is True


def test_window_start_is_five_business_days():
    assert _window_start(date(2026, 1, 9)) == date(2026, 1, 5)  # Fri back to Mon
    assert _window_start(date(2026, 1, 7)) == date(2026, 1, 1)  # Wed spans a weekend (holiday calendar is a Phase 2 TODO)


def test_pdt_tracker_rolling_window():
    t = PDTTracker()
    for d in (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)):
        t.record_day_trade(d)
    assert t.count(date(2026, 1, 6)) == 3
    assert t.count(date(2026, 1, 8)) == 3  # window Jan 2..8
    assert t.count(date(2026, 1, 9)) == 2  # Jan 2 rolled off
    assert t.would_exceed(date(2026, 1, 8)) and not t.would_exceed(date(2026, 1, 9))


def _close_req(symbol="AAPL") -> OrderRequest:
    return OrderRequest(symbol=symbol, side=Side.SELL, order_type=OrderType.MARKET, quantity=D("1"), reduce_only=True)


def test_pdt_blocks_fourth_day_trade_under_25k():
    a = adapter_at(ny(2026, 1, 8, 11, 0), equity="10000")
    for d in (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)):
        a.pdt.record_day_trade(d)
    a.pdt.record_open("AAPL", date(2026, 1, 8))
    with pytest.raises(RiskViolation) as ei:
        a.pre_trade_regulatory_check(_close_req())
    assert ei.value.code == "PDT"


def test_pdt_allows_when_not_day_trade_or_rich_or_under_limit():
    a = adapter_at(ny(2026, 1, 8, 11, 0), equity="10000")
    for d in (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)):
        a.pdt.record_day_trade(d)
    a.pre_trade_regulatory_check(_close_req())  # position was not opened today: swing exit, fine
    a.pdt.record_open("AAPL", date(2026, 1, 7))
    a.pre_trade_regulatory_check(_close_req())  # opened yesterday
    a.pdt.record_open("AAPL", date(2026, 1, 8))
    a.account_equity = D("25000")
    a.pre_trade_regulatory_check(_close_req())  # at threshold -> exempt
    b = adapter_at(ny(2026, 1, 8, 11, 0), equity="10000")
    b.pdt.record_day_trade(date(2026, 1, 7))
    b.pdt.record_open("AAPL", date(2026, 1, 8))
    b.pre_trade_regulatory_check(_close_req())  # only 1 used
    entry = OrderRequest(symbol="AAPL", side=Side.BUY, order_type=OrderType.MARKET, quantity=D("1"))
    a.account_equity = D("100")
    a.pre_trade_regulatory_check(entry)  # opening orders never trip PDT


@pytest.mark.parametrize("cls,name", [(AlpacaBrokerAdapter, "Alpaca"), (IBKRBrokerAdapter, "IBKR")])
async def test_io_is_not_enabled(cls, name):
    a = cls()
    assert a.asset_class is AssetClass.US_EQUITY
    for call in (
        a.connect(), a.load_instruments(), a.fetch_ticker("AAPL"), a.fetch_account(), a.fetch_positions(),
        a.fetch_open_orders(), a.create_order(_close_req()), a.cancel_order("1", "AAPL"),
        a.cancel_all_orders(), a.set_leverage("AAPL", 2), a.close_position("AAPL"), a.close_all_positions(),
    ):
        with pytest.raises(BrokerNotSupportedError, match=f"Phase 2: {name} adapter not enabled yet"):
            await call
    await a.close()


# ------------------------------------------------------------------ factory
def test_factory_paper_returns_unconnected_paper_broker():
    s = Settings(_env_file=None, trading_mode=TradingMode.PAPER, exchange_id="bybit", exchange_market="SPOT")
    b = create_broker(s)
    assert isinstance(b, PaperBrokerAdapter) and b.market_type is MarketType.SPOT
    assert b.wallet == s.paper_starting_balance and b.price_source is not None


def test_factory_live_returns_ccxt_adapter():
    s = Settings(
        _env_file=None, trading_mode=TradingMode.LIVE, live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
        exchange_id="bybit", exchange_api_key="k", exchange_api_secret="s", exchange_testnet=False,
        api=ApiSettings(_env_file=None, control_token="x" * 32),
    )
    b = create_broker(s)
    assert isinstance(b, CCXTBrokerAdapter) and b.market_type is MarketType.PERPETUAL


def test_factory_bad_market_and_equity_override():
    s = Settings(_env_file=None, exchange_market="FUTURES")
    with pytest.raises(ConfigError):
        create_broker(s)
    eq = create_broker(Settings(_env_file=None), AssetClass.US_EQUITY)
    assert isinstance(eq, AlpacaBrokerAdapter)


async def test_create_broker_async_equity_does_not_connect():
    eq = await create_broker_async(Settings(_env_file=None), AssetClass.US_EQUITY)
    assert isinstance(eq, AlpacaBrokerAdapter)


def test_live_requires_control_token():
    with pytest.raises(Exception, match="API_CONTROL_TOKEN"):
        Settings(
            _env_file=None, trading_mode=TradingMode.LIVE, live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
            exchange_api_key="k", exchange_api_secret="s", api=ApiSettings(_env_file=None),
        )
