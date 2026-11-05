"""Exchange data sources. The engine only talks to the ``MarketDataSource`` protocol."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import ccxt

from app.core.types import MarketType

logger = logging.getLogger(__name__)


@runtime_checkable
class MarketDataSource(Protocol):
    """Async market-data provider using ccxt shapes. Watch methods block until the next update."""

    async def watch_order_book(self, symbol: str, limit: int) -> dict[str, Any]:
        """ccxt order book: ``{'bids': [[p, q], ...], 'asks': [...], 'timestamp': ms | None}``."""

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> list[list[Any]]:
        """Candle rows ``[[ms, open, high, low, close, volume], ...]`` (may repeat / be unordered)."""

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        """ccxt ticker dict (``bid``, ``ask``, ``last``, ``timestamp``...)."""

    async def watch_liquidations(self, symbol: str) -> list[dict[str, Any]]:
        """Liquidation dicts. May raise ``ccxt.NotSupported`` when the venue has no such stream."""

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[Any]]:
        """REST candle history used to warm indicators at start-up."""

    async def close(self) -> None:
        """Release sockets / sessions."""


class CCXTProSource:
    """``MarketDataSource`` backed by ``ccxt.pro`` websockets (plus REST for history)."""

    def __init__(self, exchange_id: str, market_type: MarketType, testnet: bool = False) -> None:
        import ccxt.pro as ccxt_pro  # imported lazily: websockets are optional for non-live runs

        cls = getattr(ccxt_pro, exchange_id, None)
        if cls is None:
            raise ValueError(f"ccxt.pro has no exchange {exchange_id!r}")
        default_type = "swap" if market_type is MarketType.PERPETUAL else "spot"
        self.exchange_id = exchange_id
        self.market_type = market_type
        self.testnet = testnet
        self._exchange = cls(
            {
                "enableRateLimit": True,
                "newUpdates": False,  # always return the cached window, never an empty "delta"
                "options": {"defaultType": default_type},
            }
        )
        if testnet:
            self._exchange.set_sandbox_mode(True)

    async def watch_order_book(self, symbol: str, limit: int) -> dict[str, Any]:
        return await self._exchange.watch_order_book(symbol, limit)

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> list[list[Any]]:
        return await self._exchange.watch_ohlcv(symbol, timeframe)

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        return await self._exchange.watch_ticker(symbol)

    async def watch_liquidations(self, symbol: str) -> list[dict[str, Any]]:
        if not self._exchange.has.get("watchLiquidations"):
            raise ccxt.NotSupported(f"{self.exchange_id} does not support watchLiquidations")
        return await self._exchange.watch_liquidations(symbol)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[Any]]:
        return await self._exchange.fetch_ohlcv(symbol, timeframe, None, limit)

    async def close(self) -> None:
        try:
            await self._exchange.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.exception("error closing ccxt.pro exchange %s", self.exchange_id)
