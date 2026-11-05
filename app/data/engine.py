"""Async market-data ingestion engine.

One task per (symbol, channel) pulls from a ``MarketDataSource``, normalises ccxt payloads into
Decimal domain objects, updates bounded buffers, reports to the ``FeedHealthMonitor`` and
publishes on the ``EventBus``. Any failure marks the feed DISCONNECTED (which PAUSES the
TradingGate) and reconnects forever with exponential backoff until ``stop()``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import ccxt

from app.config.settings import DataSettings
from app.core.events import EventBus, Topic
from app.core.types import Side, Ticker, utcnow
from app.data.backoff import ExponentialBackoff
from app.data.buffers import CandleBuffer, OrderBookState, to_decimal
from app.data.health import FeedHealthMonitor, TradingGate, timeframe_seconds
from app.data.models import Candle, DataChannel, FeedHealthEvent, LiquidationEvent, OrderBookSnapshot
from app.data.sources import MarketDataSource

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MS = timedelta(milliseconds=1)
_LIQ = "LIQUIDATIONS"


def ms_to_dt(ms: Any) -> datetime | None:
    """Epoch milliseconds -> tz-aware UTC datetime (``None`` for missing / invalid input)."""
    if ms is None:
        return None
    try:
        return _EPOCH + timedelta(milliseconds=int(ms))
    except (TypeError, ValueError, OverflowError):
        return None


def dt_to_ms(dt: datetime) -> int:
    return (dt - _EPOCH) // _MS


def row_to_candle(symbol: str, timeframe: str, row: Any, *, closed: bool = False) -> Candle | None:
    """``[ms, o, h, l, c, v]`` -> Candle; ``None`` if the row is malformed."""
    try:
        open_time = ms_to_dt(row[0])
        values = [to_decimal(row[i]) for i in range(1, 6)]
    except (IndexError, TypeError, KeyError):
        return None
    if open_time is None or any(v is None for v in values):
        return None
    o, h, low, c, v = values  # type: ignore[misc]
    return Candle(symbol, timeframe, open_time, o, h, low, c, v, closed)


class MarketDataEngine:
    def __init__(
        self,
        source: MarketDataSource,
        bus: EventBus,
        symbols: list[str],
        settings: DataSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        rng: random.Random | None = None,
        require_candles: bool = False,
    ) -> None:
        if not symbols:
            raise ValueError("MarketDataEngine needs at least one symbol")
        self._source = source
        self._bus = bus
        self._symbols = list(dict.fromkeys(symbols))
        self._settings = settings
        self._wall_clock = wall_clock
        self._sleep = sleep  # used for reconnect backoff only
        self._rng = rng
        self._require_candles = require_candles
        self._tf = settings.timeframe
        self._tf_delta = timedelta(seconds=timeframe_seconds(self._tf))

        self.monitor = FeedHealthMonitor(settings, bus, clock, wall_clock)
        self._candles = {s: CandleBuffer(settings.candle_buffer_size) for s in self._symbols}
        self._books = {s: OrderBookState(s, settings.orderbook_depth) for s in self._symbols}
        self._tickers: dict[str, Ticker] = {}
        self._liqs = {s: deque[LiquidationEvent](maxlen=settings.liquidation_buffer_size) for s in self._symbols}
        self._tasks: list[asyncio.Task[Any]] = []
        self._disabled: set[tuple[str, str]] = set()
        self._started = False
        self._stopping = False
        self._closed = False

    # ---- accessors ---------------------------------------------------------------------
    @property
    def gate(self) -> TradingGate:
        return self.monitor.gate

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._symbols)

    def candles(self, symbol: str) -> CandleBuffer:
        return self._candles[symbol]

    def order_book(self, symbol: str) -> OrderBookSnapshot | None:
        return self._books[symbol].snapshot

    def last_ticker(self, symbol: str) -> Ticker | None:
        return self._tickers.get(symbol)

    def liquidations(self, symbol: str) -> tuple[LiquidationEvent, ...]:
        return tuple(self._liqs[symbol])

    def health_snapshot(self) -> dict[str, Any]:
        snap = self.monitor.snapshot()
        snap["engine"] = {
            s: {
                "candles_buffered": len(self._candles[s]),
                "orderbook_crossed_rejects": self._books[s].crossed_rejects,
                "orderbook_invalid_rejects": self._books[s].invalid_rejects,
                "liquidations_buffered": len(self._liqs[s]),
                "liquidations_disabled": (s, _LIQ) in self._disabled,
            }
            for s in self._symbols
        }
        return snap

    async def run_health_check_once(self) -> list[FeedHealthEvent]:
        return self.monitor.check()

    # ---- lifecycle ---------------------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stopping = False
        for symbol in self._symbols:
            self.monitor.register(symbol, DataChannel.ORDER_BOOK, required=True)
            # The ORDER BOOK is the freshness signal for the gate (it carries best bid/ask and streams every
            # ~100 ms). Ticker streams are trade-driven on many venues (e.g. Kraken) and can be silent for
            # several seconds in a perfectly healthy market, so they are monitored but do not pause trading.
            self.monitor.register(symbol, DataChannel.TRADES_OR_TICKER, required=False,
                                  stale_after=max(30.0, self._settings.stale_after_seconds))
            self.monitor.register(symbol, DataChannel.CANDLES, required=self._require_candles)
        for symbol in self._symbols:
            await self._bootstrap(symbol)
        for symbol in self._symbols:
            self._spawn(symbol, DataChannel.ORDER_BOOK, DataChannel.ORDER_BOOK.value, self._watch_book(symbol), self._on_book)
            self._spawn(symbol, DataChannel.CANDLES, DataChannel.CANDLES.value, self._watch_candles(symbol), self._on_candles)
            self._spawn(symbol, DataChannel.TRADES_OR_TICKER, DataChannel.TRADES_OR_TICKER.value, self._watch_ticker(symbol), self._on_ticker)
            self._spawn(symbol, None, _LIQ, self._watch_liqs(symbol), self._on_liquidations)
        self._tasks.append(asyncio.create_task(self._health_loop(), name="md-health"))

    async def stop(self) -> None:
        self._stopping = True
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if not self._closed:
            self._closed = True
            try:
                await self._source.close()
            except Exception:
                logger.exception("error closing market data source")

    async def __aenter__(self) -> "MarketDataEngine":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    # ---- bootstrap ---------------------------------------------------------------------
    async def _bootstrap(self, symbol: str) -> None:
        n = self._settings.bootstrap_candles
        if n <= 0:
            return
        try:
            rows = await self._source.fetch_ohlcv(symbol, self._tf, n + 1)
        except Exception as exc:
            logger.warning("candle bootstrap failed for %s: %r (live stream will fill the buffer)", symbol, exc)
            return
        by_open: dict[datetime, Candle] = {}
        for row in rows or []:
            candle = row_to_candle(symbol, self._tf, row)
            if candle is not None:
                by_open[candle.open_time] = candle
        candles = [by_open[k] for k in sorted(by_open)]
        if not candles:
            return
        buf = self._candles[symbol]
        forming = candles[-1].open_time + self._tf_delta > self._wall_clock()
        buf.seed(candles[:-1] if forming else candles)
        if forming:
            buf.upsert(candles[-1])  # stays in progress (closed=False)
        logger.info("bootstrapped %d closed candles for %s", len(buf.closed()), symbol)

    # ---- channel loops -----------------------------------------------------------------
    def _spawn(
        self,
        symbol: str,
        channel: DataChannel | None,
        name: str,
        watch: Callable[[], Awaitable[Any]],
        handler: Callable[[str, Any], bool],
    ) -> None:
        self._tasks.append(
            asyncio.create_task(self._channel_loop(symbol, channel, name, watch, handler), name=f"md-{name}-{symbol}")
        )

    def _watch_book(self, symbol: str) -> Callable[[], Awaitable[Any]]:
        depth = self._settings.orderbook_depth
        return lambda: self._source.watch_order_book(symbol, depth)

    def _watch_candles(self, symbol: str) -> Callable[[], Awaitable[Any]]:
        return lambda: self._source.watch_ohlcv(symbol, self._tf)

    def _watch_ticker(self, symbol: str) -> Callable[[], Awaitable[Any]]:
        return lambda: self._source.watch_ticker(symbol)

    def _watch_liqs(self, symbol: str) -> Callable[[], Awaitable[Any]]:
        return lambda: self._source.watch_liquidations(symbol)

    async def _channel_loop(
        self,
        symbol: str,
        channel: DataChannel | None,
        name: str,
        watch: Callable[[], Awaitable[Any]],
        handler: Callable[[str, Any], bool],
    ) -> None:
        backoff = ExponentialBackoff(
            self._settings.reconnect_base_delay_seconds,
            self._settings.reconnect_max_delay_seconds,
            self._settings.reconnect_jitter,
            self._rng,
        )
        while not self._stopping:
            try:
                data = await watch()
                if handler(symbol, data) and channel is not None:
                    self.monitor.mark_message(symbol, channel)
                if backoff.attempt:
                    backoff.reset()  # first success after a failure
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if channel is None and isinstance(exc, ccxt.NotSupported):
                    logger.info("%s not supported for %s; channel disabled", name, symbol)
                    self._disabled.add((symbol, name))
                    return
                delay = backoff.next_delay()
                logger.warning(
                    "%s %s feed error (%s: %s); reconnect attempt %d in %.2fs",
                    symbol, name, type(exc).__name__, exc, backoff.attempt, delay,
                )
                if channel is not None:
                    self.monitor.mark_disconnected(
                        symbol, channel, f"{type(exc).__name__}: {exc}", backoff.attempt, delay
                    )
                    self.monitor.check()  # pause the gate immediately, not at the next health tick
                await self._sleep(delay)
                if channel is not None and not self._stopping:
                    self.monitor.mark_connecting(symbol, channel, backoff.attempt)
                    self.monitor.check()

    async def _health_loop(self) -> None:
        interval = self._settings.health_check_interval_seconds
        while not self._stopping:
            await asyncio.sleep(interval)  # real time: this is the watchdog cadence
            try:
                self.monitor.check()
            except Exception:
                logger.exception("feed health check failed")

    # ---- message handlers (return True when the message counts as a fresh update) -------
    def _on_book(self, symbol: str, data: dict[str, Any]) -> bool:
        state = self._books[symbol]
        snap = state.apply_snapshot(
            data.get("bids") or [],
            data.get("asks") or [],
            ms_to_dt(data.get("timestamp")),
            self._wall_clock(),
        )
        if not state.last_accepted or snap is None:
            logger.debug("%s order book update rejected (crossed=%d invalid=%d)",
                         symbol, state.crossed_rejects, state.invalid_rejects)
            return False  # a bad book is not a fresh book: it must not keep the feed LIVE
        self._bus.publish(Topic.ORDER_BOOK, snap)
        return True

    def _on_ticker(self, symbol: str, data: dict[str, Any]) -> bool:
        bid, ask, last = (to_decimal(data.get(k)) for k in ("bid", "ask", "last"))
        if bid is None or bid <= 0:
            bid = None
        if ask is None or ask <= 0:
            ask = None
        if last is None or last <= 0:
            last = None
        if last is None:
            if bid is None or ask is None:
                return False
            last = (bid + ask) / 2
        bid = bid if bid is not None else last
        ask = ask if ask is not None else last
        if bid > ask:
            return False
        info = data.get("info") or {}
        mark = to_decimal(data.get("markPrice") or (info.get("markPrice") if isinstance(info, dict) else None))
        ticker = Ticker(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            mark=mark if mark is not None and mark > 0 else None,
            timestamp=ms_to_dt(data.get("timestamp")) or self._wall_clock(),
        )
        self._tickers[symbol] = ticker
        self._bus.publish(Topic.TICKER, ticker)
        return True

    def _on_candles(self, symbol: str, rows: list[list[Any]]) -> bool:
        buf = self._candles[symbol]
        before = buf.current
        newest = buf.latest
        floor_ms = dt_to_ms(newest.open_time) if newest is not None else None
        by_ts: dict[int, Any] = {}
        for row in rows or []:  # ccxt may resend the whole window: skip old rows before parsing
            try:
                ts = int(row[0])
            except (IndexError, TypeError, ValueError):
                continue
            if floor_ms is None or ts >= floor_ms:
                by_ts[ts] = row  # duplicate timestamps: the last occurrence wins
        for ts in sorted(by_ts):
            candle = row_to_candle(symbol, self._tf, by_ts[ts])
            if candle is None:
                logger.debug("%s malformed candle row skipped: %r", symbol, by_ts[ts])
                continue
            closed = buf.upsert(candle)
            if closed is not None:
                self._bus.publish(Topic.CANDLE_CLOSED, closed)
        current = buf.current
        if current is not None and current != before:
            self._bus.publish(Topic.CANDLE_UPDATE, current)
        return True

    def _on_liquidations(self, symbol: str, items: list[dict[str, Any]]) -> bool:
        for item in items or []:
            event = self._liquidation_from_ccxt(symbol, item)
            if event is None:
                continue
            self._liqs[symbol].append(event)
            self._bus.publish(Topic.LIQUIDATION, event)
        return True

    def _liquidation_from_ccxt(self, symbol: str, item: dict[str, Any]) -> LiquidationEvent | None:
        info = item.get("info") if isinstance(item.get("info"), dict) else {}
        order = info.get("o") if isinstance(info.get("o"), dict) else {}
        raw_side = item.get("side") or order.get("S") or info.get("side")
        side = {"buy": Side.BUY, "sell": Side.SELL}.get(str(raw_side).lower()) if raw_side else None
        price = to_decimal(item.get("price"))
        qty = to_decimal(item.get("amount"))
        if qty is None:
            contracts = to_decimal(item.get("contracts"))
            size = to_decimal(item.get("contractSize")) or Decimal(1)
            qty = contracts * size if contracts is not None else to_decimal(item.get("baseValue"))
        if side is None or price is None or qty is None or price <= 0 or qty <= 0:
            logger.debug("%s unusable liquidation skipped: %r", symbol, item)
            return None
        return LiquidationEvent(
            symbol=symbol,
            side=side,
            price=price,
            quantity=qty,
            timestamp=ms_to_dt(item.get("timestamp")) or self._wall_clock(),
        )
