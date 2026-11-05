"""Broker construction from ``Settings``.

* PAPER -> ``PaperBrokerAdapter`` whose instrument specs and live prices come from a *public, keyless*
  ccxt client for ``settings.exchange_id`` (mainnet market data; no credentials are ever sent).
* LIVE  -> ``CCXTBrokerAdapter`` with the configured keys.
* ``asset_class=US_EQUITY`` -> Phase 2 Alpaca stub.

``create_broker`` is synchronous and returns an adapter whose ``connect()`` must still be awaited (paper
instruments are loaded there). ``create_broker_async`` does both steps.
"""

from __future__ import annotations

import logging
from decimal import InvalidOperation

import ccxt
import ccxt.async_support as ccxt_async

from app.brokers.alpaca_adapter import AlpacaBrokerAdapter
from app.brokers.base import BaseBrokerAdapter
from app.brokers.ccxt_adapter import (
    CCXTBrokerAdapter,
    ccxt_ticker_to_ticker,
    map_ccxt_exception,
    market_is_wanted,
    market_to_spec,
)
from app.brokers.paper_adapter import PaperBrokerAdapter
from app.config.settings import Settings
from app.core.errors import BrokerError, ConfigError
from app.core.types import AssetClass, InstrumentSpec, MarketType, Ticker, TradingMode

log = logging.getLogger(__name__)


def _market_type(settings: Settings) -> MarketType:
    try:
        mt = MarketType(settings.exchange_market.upper())
    except ValueError:
        raise ConfigError(f"EXCHANGE_MARKET must be SPOT or PERPETUAL, got {settings.exchange_market!r}") from None
    if mt not in (MarketType.SPOT, MarketType.PERPETUAL):
        raise ConfigError(f"EXCHANGE_MARKET {mt.value} is not a crypto market type")
    return mt


class PublicMarketFeed:
    """Keyless ccxt client used by the paper broker for instrument metadata and quotes."""

    def __init__(self, exchange_id: str, market_type: MarketType) -> None:
        cls = getattr(ccxt_async, exchange_id, None)
        if cls is None:
            raise ConfigError(f"Unknown or unsupported exchange id: {exchange_id!r}")
        self.market_type = market_type
        self._exchange = cls(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap" if market_type is MarketType.PERPETUAL else "spot",
                    **({"defaultSubType": "linear"} if market_type is MarketType.PERPETUAL else {}),
                },
            }
        )

    async def load_instruments(self) -> dict[str, InstrumentSpec]:
        try:
            markets = await self._exchange.load_markets()
        except (ccxt.BaseError, OSError) as exc:
            raise map_ccxt_exception(exc) from exc
        specs: dict[str, InstrumentSpec] = {}
        for market in markets.values():
            if not market_is_wanted(market, self.market_type):
                continue
            try:
                spec = market_to_spec(market, self._exchange.precisionMode)
            except (BrokerError, KeyError, ValueError, InvalidOperation):
                continue
            specs[spec.symbol] = spec
        return specs

    async def fetch_ticker(self, symbol: str) -> Ticker:
        try:
            raw = await self._exchange.fetch_ticker(symbol)
        except (ccxt.BaseError, OSError) as exc:
            raise map_ccxt_exception(exc) from exc
        return ccxt_ticker_to_ticker(raw, symbol)

    async def close(self) -> None:
        await self._exchange.close()


def create_broker(settings: Settings, asset_class: AssetClass = AssetClass.CRYPTO) -> BaseBrokerAdapter:
    """Build (but do not connect) the broker for ``settings``. Await ``connect()`` before use."""
    if asset_class is AssetClass.US_EQUITY:
        # TODO: Phase 2 - US Equities (Alpaca/IBKR): choose Alpaca vs IBKR via a setting.
        return AlpacaBrokerAdapter(
            api_key=settings.alpaca_api_key.get_secret_value(),
            api_secret=settings.alpaca_api_secret.get_secret_value(),
            paper=settings.alpaca_paper,
        )

    market_type = _market_type(settings)

    if settings.trading_mode in (TradingMode.LIVE, TradingMode.TESTNET):
        key = settings.exchange_api_key.get_secret_value()
        secret = settings.exchange_api_secret.get_secret_value()
        if not key or not secret:
            raise ConfigError(f"{settings.trading_mode.value} mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET.")
        return CCXTBrokerAdapter(
            exchange_id=settings.exchange_id,
            market_type=market_type,
            api_key=key,
            api_secret=secret,
            password=settings.exchange_api_password.get_secret_value(),
            testnet=settings.uses_exchange_testnet,
            settle_currency=settings.quote_currency,
        )

    feed = PublicMarketFeed(settings.exchange_id, market_type)
    return PaperBrokerAdapter(
        instruments={},
        starting_balance=settings.paper_starting_balance,
        quote=settings.quote_currency,
        slippage_bps=settings.risk.default_slippage_bps,
        price_source=feed.fetch_ticker,
        market_type=market_type,
        instrument_loader=feed.load_instruments,
        on_close=feed.close,
    )


async def create_broker_async(
    settings: Settings, asset_class: AssetClass = AssetClass.CRYPTO
) -> BaseBrokerAdapter:
    """``create_broker`` + ``await connect()`` (loads markets). Phase 2 equity stubs are returned unconnected."""
    broker = create_broker(settings, asset_class)
    if asset_class is AssetClass.US_EQUITY:
        return broker  # connect() raises until Phase 2 is enabled
    try:
        await broker.connect()
    except BaseException:
        await broker.close()
        raise
    return broker
