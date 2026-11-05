"""Signal pipeline: market data -> indicators -> strategies -> fee-aware pre-filter -> SignalPayload.

Wires the independently-built engines together on the EventBus:

    MarketDataEngine --CANDLE_CLOSED--> SignalPipeline --QuantEngine--> CLAUDE_TRIGGER (SignalPayload)
    NewsEngine (cache) --------------------^

Defence in depth for stale data (rule 1): a symbol may only trigger when BOTH the market-data
TradingGate says ACTIVE and the latest order book seen by this pipeline is younger than
``DataSettings.stale_after_seconds`` at the moment the closed candle is evaluated. The gate is
updated by the health loop on a ~250 ms cadence; the direct age check closes the gap between ticks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from app.config.settings import DataSettings, QuantSettings
from app.core.events import EventBus, Topic
from app.core.types import utcnow
from app.data.models import Candle, OrderBookSnapshot
from app.news.contracts import NewsEvent
from app.quant.contracts import IndicatorSnapshot, PreFilterOutcome, PreFilterResult, SignalPayload
from app.quant.engine import QuantEngine

logger = logging.getLogger(__name__)

_QUIET_OUTCOMES = {PreFilterOutcome.NO_SETUP, PreFilterOutcome.NOT_WARM}


class MarketView(Protocol):
    """The parts of MarketDataEngine the pipeline needs (keeps the pipeline testable)."""

    @property
    def symbols(self) -> list[str]: ...

    def order_book(self, symbol: str) -> OrderBookSnapshot | None: ...

    def candles(self, symbol: str) -> Any: ...

    def health_snapshot(self) -> dict[str, Any]: ...

    @property
    def gate(self) -> Any: ...


class NewsView(Protocol):
    def recent_for(self, symbol: str, since: datetime, limit: int = 10) -> list[NewsEvent]: ...


class SignalPipeline:
    def __init__(
        self,
        market: MarketView,
        quant: QuantEngine,
        bus: EventBus,
        data_settings: DataSettings,
        quant_settings: QuantSettings,
        *,
        news: NewsView | None = None,
        funding_provider: Callable[[str], Decimal | None] | None = None,
        wall_clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._market = market
        self._quant = quant
        self._bus = bus
        self._data = data_settings
        self._qs = quant_settings
        self._news = news
        self._funding = funding_provider
        self._clock = wall_clock
        self._task: asyncio.Task[None] | None = None
        self._sub = None
        self.stats: dict[str, int] = {"candles": 0, "triggers": 0, "rejections_published": 0, "errors": 0}

    # ------------------------------------------------------------------ gating
    def is_tradeable(self, symbol: str) -> bool:
        """Gate ACTIVE *and* the order book fresh right now. Passed to QuantEngine as gate_is_active."""
        if not self._market.gate.is_active(symbol):
            return False
        book = self._market.order_book(symbol)
        if book is None:
            return False
        age = (self._clock() - book.received_at).total_seconds()
        return age <= self._data.stale_after_seconds

    # ------------------------------------------------------------------ lifecycle
    def warmup(self) -> int:
        """Feed REST-bootstrapped closed candles into the indicator engine. Returns candles used."""
        n = 0
        for symbol in self._market.symbols:
            closed = tuple(self._market.candles(symbol).closed())
            if closed:
                self._quant.warmup(symbol, closed)
                n += len(closed)
        logger.info("signal pipeline warmed up with %d candles", n)
        return n

    def start(self) -> None:
        if self._task is not None:
            return
        self._sub = self._bus.subscribe(Topic.CANDLE_CLOSED, maxsize=1000)
        self._task = asyncio.create_task(self._loop(), name="signal-pipeline")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._sub is not None:
            self._bus.unsubscribe(self._sub)
            self._sub = None

    async def _loop(self) -> None:
        assert self._sub is not None
        while True:
            candle = await self._sub.get()
            try:
                self.handle_closed_candle(candle)
            except Exception:  # noqa: BLE001 - one bad candle must not stop a 24/7 pipeline
                self.stats["errors"] += 1
                logger.exception("signal pipeline failed on %s %s", candle.symbol, candle.open_time)

    # ------------------------------------------------------------------ core
    def handle_closed_candle(
        self, candle: Candle
    ) -> tuple[IndicatorSnapshot, PreFilterResult, SignalPayload | None]:
        self.stats["candles"] += 1
        symbol = candle.symbol
        book = self._market.order_book(symbol)
        news: list[NewsEvent] = []
        if self._news is not None:
            since = self._clock() - timedelta(minutes=self._qs.news_lookback_minutes)
            news = self._news.recent_for(symbol, since, limit=max(self._qs.news_in_payload, 1))
        health = self._market_health(symbol, book)
        funding = self._funding(symbol) if self._funding is not None else None

        snap, result, payload = self._quant.process_closed_candle(candle, book, news, health, funding)
        self._bus.publish(Topic.INDICATORS, snap)
        if payload is not None:
            self.stats["triggers"] += 1
            self._bus.publish(Topic.CLAUDE_TRIGGER, payload)
            logger.info("CLAUDE_TRIGGER %s %s %s score=%.2f", symbol, payload.direction.value,
                        payload.primary_strategy.value, payload.confluence_score)
        elif result.outcome not in _QUIET_OUTCOMES:
            self.stats["rejections_published"] += 1
            self._bus.publish(Topic.SIGNAL_REJECTED, result)
        return snap, result, payload

    def _market_health(self, symbol: str, book: OrderBookSnapshot | None) -> dict[str, Any]:
        full = self._market.health_snapshot()
        book_age = None if book is None else round((self._clock() - book.received_at).total_seconds(), 3)
        sym = full.get("symbols", {}).get(symbol, {}) if isinstance(full.get("symbols"), dict) else {}
        return {
            "symbol": symbol,
            "gate": self._market.gate.symbol_state(symbol).value,
            "tradeable": self.is_tradeable(symbol),
            "order_book_age_s": book_age,
            "stale_after_s": self._data.stale_after_seconds,
            "channels": sym.get("channels", {}),
        }
