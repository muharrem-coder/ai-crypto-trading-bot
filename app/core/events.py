"""In-process async pub/sub bus connecting the data, news, quant and (Phase 3) Claude layers.

Memory safety for 24/7 operation: every subscriber gets a BOUNDED queue. When a slow consumer
falls behind, the OLDEST event is dropped (latest market state matters more than history) and
a drop counter is incremented so the health monitor can report it. Nothing grows unbounded.
Phase 4 bridges selected topics to Redis Pub/Sub for the web terminal.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class Topic:
    CANDLE_CLOSED = "market.candle_closed"  # payload: app.data.models.Candle (closed=True)
    CANDLE_UPDATE = "market.candle_update"  # payload: Candle (closed=False, in-progress bar)
    ORDER_BOOK = "market.order_book"  # payload: app.data.models.OrderBookSnapshot
    TICKER = "market.ticker"  # payload: app.core.types.Ticker
    LIQUIDATION = "market.liquidation"  # payload: app.data.models.LiquidationEvent
    FEED_HEALTH = "market.feed_health"  # payload: app.data.models.FeedHealthEvent
    TRADING_GATE = "system.trading_gate"  # payload: app.data.models.TradingGateEvent (ACTIVE/PAUSED)
    NEWS = "news.item"  # payload: app.news.contracts.NewsEvent (already deduplicated + relevant)
    INDICATORS = "quant.indicators"  # payload: app.quant.contracts.IndicatorSnapshot
    SIGNAL_REJECTED = "quant.signal_rejected"  # payload: app.quant.contracts.PreFilterResult
    CLAUDE_TRIGGER = "quant.claude_trigger"  # payload: app.quant.contracts.SignalPayload
    # Phase 3/4: decision, execution and position lifecycle (consumed by the /ws/terminal bridge)
    CLAUDE_DECISION = "claude.decision"  # payload: dict (decision_id, symbol, action, reasoning_tr, ...)
    EXECUTION_REPORT = "execution.report"  # payload: app.execution.contracts.ExecutionReport
    POSITION_OPENED = "position.opened"  # payload: dict (see app.api.protocol.position_dict)
    POSITION_UPDATED = "position.updated"  # payload: dict (stop moved / break-even locked)
    POSITION_CLOSED = "position.closed"  # payload: dict (trade_id, symbol, exit_price, exit_reason, net_pnl)
    KILL_SWITCH = "risk.kill_switch"  # payload: dict (engaged, reason, code, message)
    RISK_EVENT = "risk.event"  # payload: dict (level, code, message, details)


@dataclass(slots=True)
class Subscription:
    topic: str
    queue: asyncio.Queue[Any]
    dropped: int = 0
    closed: bool = False

    async def get(self) -> Any:
        return await self.queue.get()

    def get_nowait(self) -> Any:
        return self.queue.get_nowait()

    def drain(self) -> list[Any]:
        items: list[Any] = []
        while True:
            try:
                items.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return items

    async def __aiter__(self) -> AsyncIterator[Any]:
        while not self.closed:
            yield await self.queue.get()


class EventBus:
    def __init__(self, default_maxsize: int = 1000) -> None:
        if default_maxsize <= 0:
            raise ValueError("maxsize must be positive (unbounded queues are forbidden)")
        self._default_maxsize = default_maxsize
        self._subs: dict[str, list[Subscription]] = defaultdict(list)
        self.published: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, maxsize: int | None = None) -> Subscription:
        size = maxsize or self._default_maxsize
        if size <= 0:
            raise ValueError("maxsize must be positive")
        sub = Subscription(topic=topic, queue=asyncio.Queue(maxsize=size))
        self._subs[topic].append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.closed = True
        subs = self._subs.get(sub.topic, [])
        if sub in subs:
            subs.remove(sub)

    def publish(self, topic: str, payload: Any) -> None:
        """Non-blocking publish. Never awaits, never raises because of slow consumers."""
        self.published[topic] += 1
        for sub in self._subs.get(topic, ()):
            if sub.queue.full():
                try:
                    sub.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                sub.dropped += 1
                if sub.dropped in (1, 100, 10_000) or sub.dropped % 100_000 == 0:
                    logger.warning("EventBus: slow subscriber on %s, dropped=%d", topic, sub.dropped)
            sub.queue.put_nowait(payload)

    def subscriber_count(self, topic: str) -> int:
        return len(self._subs.get(topic, ()))

    def stats(self) -> dict[str, Any]:
        return {
            "published": dict(self.published),
            "dropped": {t: sum(s.dropped for s in subs) for t, subs in self._subs.items()},
            "queued": {t: sum(s.queue.qsize() for s in subs) for t, subs in self._subs.items()},
        }
