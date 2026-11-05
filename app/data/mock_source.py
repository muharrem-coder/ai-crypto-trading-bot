"""Deterministic in-memory ``MarketDataSource`` for tests and end-to-end runs (no network)."""

from __future__ import annotations

import asyncio
import copy
from collections import Counter
from decimal import Decimal
from typing import Any

import ccxt

from app.core.types import Side
from app.data.models import DataChannel

_LIQUIDATIONS = "LIQUIDATIONS"


class _Disconnect:
    """Queue marker: the watch call that dequeues it raises ``exc``."""

    __slots__ = ("exc",)

    def __init__(self, exc: Exception) -> None:
        self.exc = exc


def _num(x: Any) -> Any:
    """ccxt-style payloads carry floats; keep ints / strs / floats and convert Decimal."""
    return float(x) if isinstance(x, Decimal) else x


def _levels(levels: Any) -> list[list[Any]]:
    return [[_num(p), _num(q)] for p, q in levels]


class ScriptedMarketSource:
    """Each ``watch_*`` awaits a bounded ``asyncio.Queue`` per (symbol, channel).

    Messages are delivered FIFO. ``disconnect`` enqueues a marker, so the first watch call that
    reaches it (pending or future) raises ``ccxt.NetworkError``; later pushes are delivered by
    the following call, which models a successful reconnect. A full queue drops its OLDEST item.
    """

    LIQUIDATIONS = _LIQUIDATIONS

    def __init__(self, queue_size: int = 1000, liquidations_supported: bool = True) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._queue_size = queue_size
        self.liquidations_supported = liquidations_supported
        self._queues: dict[tuple[str, str], asyncio.Queue[Any]] = {}
        self._history: dict[str, list[list[Any]]] = {}
        self.calls: Counter[str] = Counter()
        self.symbol_calls: Counter[tuple[str, str]] = Counter()  # (method, symbol)
        self.closed = False

    # ---- internals ---------------------------------------------------------------------
    @staticmethod
    def _key(symbol: str, channel: DataChannel | str) -> tuple[str, str]:
        return symbol, channel.value if isinstance(channel, DataChannel) else str(channel)

    def _queue(self, symbol: str, channel: DataChannel | str) -> asyncio.Queue[Any]:
        key = self._key(symbol, channel)
        queue = self._queues.get(key)
        if queue is None:
            queue = asyncio.Queue(maxsize=self._queue_size)
            self._queues[key] = queue
        return queue

    def _put(self, symbol: str, channel: DataChannel | str, item: Any) -> None:
        queue = self._queue(symbol, channel)
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(item)

    def _count(self, method: str, symbol: str) -> None:
        self.calls[method] += 1
        self.symbol_calls[(method, symbol)] += 1

    async def _next(self, symbol: str, channel: DataChannel | str) -> Any:
        item = await self._queue(symbol, channel).get()
        if isinstance(item, _Disconnect):
            raise item.exc
        return item

    def pending(self, symbol: str, channel: DataChannel | str) -> int:
        """Number of undelivered messages (including disconnect markers)."""
        return self._queue(symbol, channel).qsize()

    # ---- scripting API -----------------------------------------------------------------
    def set_history(self, symbol: str, rows: list[list[Any]]) -> None:
        self._history[symbol] = [list(r) for r in rows]

    def push_ohlcv(self, symbol: str, rows: list[list[Any]]) -> None:
        self._put(symbol, DataChannel.CANDLES, [[_num(v) for v in r] for r in rows])

    def push_order_book(self, symbol: str, bids: Any, asks: Any, ts_ms: int | None = None) -> None:
        self._put(
            symbol,
            DataChannel.ORDER_BOOK,
            {"symbol": symbol, "bids": _levels(bids), "asks": _levels(asks), "timestamp": ts_ms},
        )

    def push_ticker(self, symbol: str, bid: Any, ask: Any, last: Any, ts_ms: int | None = None) -> None:
        self._put(
            symbol,
            DataChannel.TRADES_OR_TICKER,
            {"symbol": symbol, "bid": _num(bid), "ask": _num(ask), "last": _num(last), "timestamp": ts_ms},
        )

    def push_liquidation(
        self, symbol: str, side: Side | str, price: Any, qty: Any, ts_ms: int | None = None
    ) -> None:
        side_str = side.value if isinstance(side, Side) else str(side)
        self._put(
            symbol,
            _LIQUIDATIONS,
            [
                {
                    "symbol": symbol,
                    "side": side_str.lower(),
                    "price": _num(price),
                    "amount": _num(qty),
                    "timestamp": ts_ms,
                }
            ],
        )

    def disconnect(self, symbol: str, channel: DataChannel | str, exc: Exception | None = None) -> None:
        """Make the next pending / future watch on ``channel`` raise (default ``ccxt.NetworkError``)."""
        error = exc if exc is not None else ccxt.NetworkError("scripted disconnect")
        self._put(symbol, channel, _Disconnect(error))

    # ---- MarketDataSource --------------------------------------------------------------
    async def watch_order_book(self, symbol: str, limit: int) -> dict[str, Any]:
        self._count("watch_order_book", symbol)
        return copy.deepcopy(await self._next(symbol, DataChannel.ORDER_BOOK))

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> list[list[Any]]:
        self._count("watch_ohlcv", symbol)
        return copy.deepcopy(await self._next(symbol, DataChannel.CANDLES))

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        self._count("watch_ticker", symbol)
        return copy.deepcopy(await self._next(symbol, DataChannel.TRADES_OR_TICKER))

    async def watch_liquidations(self, symbol: str) -> list[dict[str, Any]]:
        self._count("watch_liquidations", symbol)
        if not self.liquidations_supported:
            raise ccxt.NotSupported("scripted source: liquidations not supported")
        return copy.deepcopy(await self._next(symbol, _LIQUIDATIONS))

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[Any]]:
        self._count("fetch_ohlcv", symbol)
        rows = self._history.get(symbol, [])
        return [list(r) for r in rows[-limit:]] if limit > 0 else []

    async def close(self) -> None:
        self.calls["close"] += 1
        self.closed = True
