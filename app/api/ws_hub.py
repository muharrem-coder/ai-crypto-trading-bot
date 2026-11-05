"""/ws/terminal hub: EventBus -> browser bridge with an instant snapshot and 5 Hz throttling.

Design summary
--------------
* The hub subscribes to the EventBus ONCE (bounded subscriptions, one consumer task per topic); every
  connected browser gets its own bounded send queue drained by its own writer task, so a slow browser can
  never block the bus, the ticker or other clients.
* Every connection starts with exactly one INITIAL_SNAPSHOT (seq 1). Backend parts are fetched concurrently
  under a shared deadline (``ws_snapshot_deadline_seconds``); a part that is late or fails is replaced by an
  empty default and listed in ``data["partial"]``. Live events that arrive while the snapshot is built sit in
  the client's queue and are only delivered after the snapshot.
* Event driven messages (decision, execution, positions, kill switch, news, risk, health) go out immediately.
  Throttled messages (MARKET_TICK, CANDLE_UPDATE, POSITIONS_PNL, EQUITY_KPIS) are computed once per hub tick
  (``tick_once``) and sent to a client only if the value differs from what that client last received.
* Backpressure: when a client's queue is full the oldest THROTTLED message is dropped; if only critical
  messages remain the client is closed with 1013 (slow consumer).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from app.api.protocol import THROTTLED_TYPES, MessageType, TerminalBackend
from app.config.settings import ApiSettings
from app.core.events import EventBus, Subscription, Topic
from app.core.types import utcnow

logger = logging.getLogger(__name__)

__all__ = ["TerminalHub", "envelope", "to_jsonable", "json_default", "news_item", "execution_report_dict",
           "claude_log_item", "origin_allowed"]

MAX_CLIENT_MESSAGE_BYTES = 4096
SEND_TIMEOUT_SECONDS = 5.0  # a single websocket send that takes longer means the browser is not reading
CLOSE_TIMEOUT_SECONDS = 1.0
BACKEND_CALL_TIMEOUT_SECONDS = 2.0
SLOW_CYCLE_SECONDS = 1.0  # EQUITY_KPIS / POSITIONS_PNL are refreshed at least this often
SNAPSHOT_CANDLES = 120
SNAPSHOT_TRADES = 50
SNAPSHOT_LOGS = 25
SNAPSHOT_NEWS = 25
EVENT_QUEUE_SIZE = 1000
MARKET_QUEUE_SIZE = 1024  # high rate topics only set dirty flags, so losing the oldest entries is harmless
_MAX_DEPTH = 25


# ------------------------------------------------------------------------------------------ JSON helpers
def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert to JSON-safe primitives: Enum -> value, Decimal -> float, datetime -> iso (UTC),
    dataclass -> dict, tuple/set -> list, NaN/Infinity -> None. Unknown objects become ``str(obj)``."""
    if _depth > _MAX_DEPTH:
        return str(obj)
    if isinstance(obj, Enum):
        return to_jsonable(obj.value, _depth + 1)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Decimal):
        return float(obj) if obj.is_finite() else None
    if isinstance(obj, datetime):
        return _iso_utc(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(to_jsonable(k, _depth + 1))): to_jsonable(v, _depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, deque)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [to_jsonable(v, _depth + 1) for v in obj]
        try:
            return sorted(items)
        except TypeError:
            return items
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name), _depth + 1) for f in dataclasses.fields(obj)}
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="replace")
    return str(obj)


def json_default(obj: Any) -> Any:
    """``default=`` hook for json.dumps (Decimal -> float, datetime -> iso, Enum -> value, ...)."""
    return to_jsonable(obj)


def _type_value(msg_type: MessageType | str) -> str:
    return msg_type.value if isinstance(msg_type, Enum) else str(msg_type)


def envelope(msg_type: MessageType | str, seq: int, data: Any, now: datetime | None = None) -> str:
    """Wire form of one server -> client message: {"type", "seq", "ts", "data"} as a JSON string."""
    ts = _iso_utc(now if now is not None else utcnow())
    return json.dumps({"type": _type_value(msg_type), "seq": seq, "ts": ts, "data": to_jsonable(data)},
                      ensure_ascii=False, default=json_default, separators=(",", ":"))


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------ payload normalisers
def claude_log_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Topic.CLAUDE_DECISION dict -> claude_logs item (confidence float, timestamp). ``update`` is True for the
    second publication of a decision (risk_decision / execution_status filled in): the UI upserts by decision_id."""
    ts = payload.get("timestamp") or payload.get("created_at") or utcnow()
    reasoning = payload.get("reasoning_tr")
    return to_jsonable({
        "decision_id": payload.get("decision_id"),
        "timestamp": ts,
        "symbol": payload.get("symbol"),
        "action": payload.get("action"),
        "confidence": _as_float(payload.get("confidence"), 0.0),
        "leverage": _as_int_or_none(payload.get("leverage")),
        "metrics_summary": payload.get("metrics_summary") or "",
        "reasoning_tr": reasoning if isinstance(reasoning, str) else ("" if reasoning is None else str(reasoning)),
        "risk_flags": list(payload.get("risk_flags") or []),
        "risk_decision": payload.get("risk_decision"),
        "execution_status": payload.get("execution_status"),
        "fallback": payload.get("fallback"),
        "latency_ms": _as_int_or_none(payload.get("latency_ms")),
        "update": bool(payload.get("update", False)),
    })


def news_item(event: Any) -> dict[str, Any]:
    """NewsEvent -> news_feed item. A dict that already has the item shape is passed through."""
    if isinstance(event, dict):
        return to_jsonable(event)
    return to_jsonable({
        "news_id": event.news_id,
        "time": event.published_at,
        "source": event.source,
        "title": event.title,
        "url": event.url,
        "symbols": list(event.symbols),
        "macro_tags": list(event.macro_tags),
        "sentiment_score": _as_float(event.sentiment_score, 0.0),
        "sentiment_label": event.sentiment_label,
        "importance": int(event.importance),
        "duplicate_sources": list(event.duplicate_sources),
    })


def _order_summary(order: Any) -> dict[str, Any]:
    return {
        "order_id": getattr(order, "order_id", None),
        "symbol": getattr(order, "symbol", None),
        "side": getattr(order, "side", None),
        "order_type": getattr(order, "order_type", None),
        "status": getattr(order, "status", None),
        "quantity": getattr(order, "quantity", None),
        "filled_quantity": getattr(order, "filled_quantity", None),
        "price": getattr(order, "price", None),
        "average_fill_price": getattr(order, "average_fill_price", None),
        "stop_price": getattr(order, "stop_price", None),
        "reduce_only": getattr(order, "reduce_only", None),
    }


def execution_report_dict(report: Any) -> dict[str, Any]:
    """ExecutionReport -> JSON dict (enums -> value, Decimals -> float, datetimes -> iso, orders summarised)."""
    return to_jsonable({
        "status": report.status,
        "symbol": report.symbol,
        "decision_id": report.decision_id,
        "signal_id": report.signal_id,
        "detail": report.detail,
        "side": report.side,
        "requested_leverage": report.requested_leverage,
        "approved_leverage": report.approved_leverage,
        "approved_quantity": report.approved_quantity,
        "filled_quantity": report.filled_quantity,
        "entry_price": report.entry_price,
        "stop_loss": report.stop_loss,
        "take_profit": report.take_profit,
        "drift_atr": report.drift_atr,
        "risk_reasons": list(report.risk_reasons),
        "orders": [_order_summary(o) for o in report.orders],
        "warnings": list(report.warnings),
        "trade_id": report.trade_id,
        "timestamp": report.timestamp,
    })


def origin_allowed(origin: str | None, host: str | None, allowed_origins: list[str]) -> bool:
    """No Origin header -> allowed (non-browser client). Otherwise the origin must be whitelisted or have the
    same host[:port] as the Host header the request was sent to (same-origin)."""
    if origin is None:
        return True
    norm = origin.strip().rstrip("/").lower()
    if norm in {o.strip().rstrip("/").lower() for o in allowed_origins}:
        return True
    netloc = urlsplit(norm).netloc
    return bool(netloc) and host is not None and netloc == host.strip().lower()


# ------------------------------------------------------------------------------------------ client state
class _Out:
    """One queued outbound message (the envelope is built by the writer so ``seq`` follows the wire order)."""

    __slots__ = ("type", "data", "throttled")

    def __init__(self, msg_type: MessageType, data: Any) -> None:
        self.type = msg_type
        self.data = data
        self.throttled = msg_type in THROTTLED_TYPES


class _SendQueue(asyncio.Queue):  # type: ignore[type-arg]
    """asyncio.Queue that can evict queued items (uses the documented ``_queue`` deque of Queue subclasses)."""

    def _adjust(self, removed: int) -> None:
        if removed:
            self._unfinished_tasks = max(0, self._unfinished_tasks - removed)  # type: ignore[attr-defined]
            if self._unfinished_tasks == 0:  # type: ignore[attr-defined]
                self._finished.set()  # type: ignore[attr-defined]

    def remove_first(self, pred: Callable[[_Out], bool]) -> bool:
        for item in self._queue:  # type: ignore[attr-defined]
            if pred(item):
                self._queue.remove(item)  # type: ignore[attr-defined]
                self._adjust(1)
                return True
        return False

    def remove_all(self, pred: Callable[[_Out], bool]) -> int:
        kept = [i for i in self._queue if not pred(i)]  # type: ignore[attr-defined]
        removed = len(self._queue) - len(kept)  # type: ignore[attr-defined]
        if removed:
            self._queue.clear()  # type: ignore[attr-defined]
            self._queue.extend(kept)  # type: ignore[attr-defined]
            self._adjust(removed)
        return removed


class _Client:
    __slots__ = ("ws", "queue", "seq", "symbol", "last", "closing", "done", "close_code", "close_reason",
                 "writer", "remote")

    def __init__(self, ws: Any, queue_size: int, symbol: str) -> None:
        self.ws = ws
        self.queue = _SendQueue(maxsize=queue_size)
        self.seq = 0
        self.symbol = symbol
        self.last: dict[MessageType, Any] = {}  # last value enqueued per throttled/deduplicated type
        self.closing = asyncio.Event()
        self.done = asyncio.Event()
        self.close_code: int | None = None
        self.close_reason = ""
        self.writer: asyncio.Task[None] | None = None
        self.remote = str(getattr(ws, "client", None))


# ---------------------------------------------------------------------------------------------- the hub
class TerminalHub:
    def __init__(self, bus: EventBus, backend: TerminalBackend, settings: ApiSettings, *,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], datetime] = utcnow,
                 tick_interval: float | None = None) -> None:
        self._bus = bus
        self._backend = backend
        self._settings = settings
        self._clock = clock
        self._wall = wall_clock
        self._interval = tick_interval if tick_interval is not None else settings.ws_throttle_seconds
        self._clients: set[_Client] = set()
        self._subs: list[Subscription] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._tick_lock = asyncio.Lock()
        self._dirty_tick: set[str] = set()
        self._dirty_candle: set[str] = set()
        self._dirty_positions = False
        self._dirty_equity = False
        self._last_slow: float | None = None
        self._position_symbols: set[str] = set()
        self.stats: dict[str, int] = {"connections": 0, "slow_closed": 0, "dropped_throttled": 0,
                                      "events": 0, "send_errors": 0}

    # ------------------------------------------------------------------ lifecycle
    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        plan: list[tuple[str, Callable[[Any], None], int]] = [
            (Topic.CLAUDE_DECISION, self._on_decision, EVENT_QUEUE_SIZE),
            (Topic.EXECUTION_REPORT, self._on_execution, EVENT_QUEUE_SIZE),
            (Topic.POSITION_OPENED, self._on_position_upsert(MessageType.POSITION_OPENED), EVENT_QUEUE_SIZE),
            (Topic.POSITION_UPDATED, self._on_position_upsert(MessageType.POSITION_UPDATED), EVENT_QUEUE_SIZE),
            (Topic.POSITION_CLOSED, self._on_position_closed, EVENT_QUEUE_SIZE),
            (Topic.KILL_SWITCH, self._on_kill_switch, EVENT_QUEUE_SIZE),
            (Topic.RISK_EVENT, self._on_risk, EVENT_QUEUE_SIZE),
            (Topic.NEWS, self._on_news, EVENT_QUEUE_SIZE),
            (Topic.FEED_HEALTH, self._on_health, 200),
            (Topic.TRADING_GATE, self._on_health, 200),
            (Topic.TICKER, self._on_ticker, MARKET_QUEUE_SIZE),
            (Topic.ORDER_BOOK, self._on_book, MARKET_QUEUE_SIZE),
            (Topic.CANDLE_UPDATE, self._on_candle, MARKET_QUEUE_SIZE),
            (Topic.CANDLE_CLOSED, self._on_candle, MARKET_QUEUE_SIZE),
        ]
        for topic, handler, size in plan:
            sub = self._bus.subscribe(topic, maxsize=size)
            self._subs.append(sub)
            self._tasks.append(asyncio.create_task(self._consume(sub, handler), name=f"ws-hub:{topic}"))
        self._tasks.append(asyncio.create_task(self._ticker_loop(), name="ws-hub:ticker"))

    async def stop(self) -> None:
        """Close every client with 1001, cancel the hub tasks and unsubscribe from the bus. Idempotent."""
        self._running = False
        clients = tuple(self._clients)
        for client in clients:
            self._kick(client, 1001, "server shutdown")
        if clients:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(3.0):
                    await asyncio.gather(*(c.done.wait() for c in clients))
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        subs, self._subs = self._subs, []
        for sub in subs:
            self._bus.unsubscribe(sub)

    # ------------------------------------------------------------------ bus consumers
    async def _consume(self, sub: Subscription, handler: Callable[[Any], None]) -> None:
        while True:
            first = await sub.get()
            for item in (first, *sub.drain()):
                try:
                    handler(item)
                except Exception:  # noqa: BLE001 - a bad event must never stop the consumer
                    logger.exception("ws hub: handler failed for topic %s", sub.topic)
            # yield after every batch so consumers of other topics interleave (keeps cross-topic order close
            # to publish order for lifecycle events, which are handled one at a time below).
            await asyncio.sleep(0)

    async def _ticker_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.tick_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("ws hub: tick failed")

    # ---- event handlers (synchronous: no awaits, so delivery order == bus order within a topic)
    def _on_decision(self, payload: Any) -> None:
        if isinstance(payload, dict):
            self._fanout(MessageType.CLAUDE_DECISION, claude_log_item(payload))

    def _on_execution(self, report: Any) -> None:
        self._fanout(MessageType.EXECUTION_REPORT, execution_report_dict(report))

    def _on_position_upsert(self, msg_type: MessageType) -> Callable[[Any], None]:
        def handler(payload: Any) -> None:
            self._dirty_positions = True
            self._dirty_equity = True
            self._fanout(msg_type, to_jsonable(payload if isinstance(payload, dict) else {}))
        return handler

    def _on_position_closed(self, payload: Any) -> None:
        self._dirty_positions = True
        self._dirty_equity = True
        body = payload if isinstance(payload, dict) else {}
        trade = body.get("trade") if isinstance(body.get("trade"), dict) else body
        self._fanout(MessageType.POSITION_CLOSED, {"symbol": body.get("symbol"), "trade": to_jsonable(trade)})

    def _on_kill_switch(self, payload: Any) -> None:
        data = to_jsonable(payload if isinstance(payload, dict) else {})
        status = self._backend.system_status()
        data["system_status"] = status.value if isinstance(status, Enum) else str(status)
        self._dirty_equity = True
        self._fanout(MessageType.KILL_SWITCH, data)

    def _on_risk(self, payload: Any) -> None:
        body = payload if isinstance(payload, dict) else {}
        self._fanout(MessageType.RISK_EVENT, to_jsonable({
            "level": body.get("level"), "code": body.get("code"), "message": body.get("message")}))

    def _on_news(self, event: Any) -> None:
        self._fanout(MessageType.NEWS, news_item(event))

    def _on_health(self, _event: Any) -> None:
        self._publish_health()

    @staticmethod
    def _symbol_of(item: Any) -> str | None:
        symbol = item.get("symbol") if isinstance(item, dict) else getattr(item, "symbol", None)
        return symbol if isinstance(symbol, str) else None

    def _on_ticker(self, ticker: Any) -> None:
        symbol = self._symbol_of(ticker)
        if symbol:
            self._dirty_tick.add(symbol)
            if symbol in self._position_symbols:
                self._dirty_positions = True  # live unrealized PnL

    def _on_book(self, book: Any) -> None:
        symbol = self._symbol_of(book)
        if symbol:
            self._dirty_tick.add(symbol)

    def _on_candle(self, candle: Any) -> None:
        symbol = self._symbol_of(candle)
        if symbol:
            self._dirty_candle.add(symbol)

    def _publish_health(self) -> None:
        try:
            health = to_jsonable(self._backend.market_health())
        except Exception:  # noqa: BLE001
            logger.exception("ws hub: market_health() failed")
            return
        for client in tuple(self._clients):
            self._send_if_changed(client, MessageType.MARKET_HEALTH, health)

    # ------------------------------------------------------------------ fan-out / backpressure
    def broadcast(self, msg_type: MessageType, data: Any) -> None:
        """Send one message to every connected client (throttled types are coalesced / droppable)."""
        self._fanout(msg_type, to_jsonable(data))

    def _fanout(self, msg_type: MessageType, data: Any) -> None:
        self.stats["events"] += 1
        for client in tuple(self._clients):
            self._enqueue(client, msg_type, data)

    def _send_if_changed(self, client: _Client, msg_type: MessageType, data: Any) -> bool:
        if client.last.get(msg_type) == data:
            return False
        client.last[msg_type] = data
        return self._enqueue(client, msg_type, data)

    def _enqueue(self, client: _Client, msg_type: MessageType, data: Any) -> bool:
        if client.closing.is_set():
            return False
        out = _Out(msg_type, data)
        queue = client.queue
        if out.throttled:
            queue.remove_all(lambda o: o.type is msg_type)  # coalesce: latest value wins
        if queue.full():
            if queue.remove_first(lambda o: o.throttled):
                self.stats["dropped_throttled"] += 1
            else:
                self.stats["slow_closed"] += 1
                logger.warning("ws hub: slow consumer %s (queue full of critical messages), closing", client.remote)
                self._kick(client, 1013, "slow consumer")
                return False
        queue.put_nowait(out)
        return True

    def _kick(self, client: _Client, code: int | None, reason: str) -> None:
        """Mark a client for closing (``code`` None: connection already dead, just clean up)."""
        if client.closing.is_set():
            return
        client.close_code = code
        client.close_reason = reason
        self._clients.discard(client)
        client.closing.set()

    # ------------------------------------------------------------------ throttled messages
    async def _guard(self, awaitable: Any, default: Any = None) -> Any:
        try:
            async with asyncio.timeout(BACKEND_CALL_TIMEOUT_SECONDS):
                return await awaitable
        except Exception:  # noqa: BLE001 - includes TimeoutError
            logger.exception("ws hub: backend call failed")
            return default

    async def tick_once(self) -> None:
        """One throttle cycle: compute each throttled value once and send it to every client whose last
        received value differs. Called every ``ws_throttle_seconds`` by the ticker task (tests call it directly)."""
        async with self._tick_lock:
            now = self._clock()
            slow_due = self._last_slow is None or now - self._last_slow >= SLOW_CYCLE_SECONDS
            if slow_due:
                self._last_slow = now
            tick_syms, self._dirty_tick = self._dirty_tick, set()
            candle_syms, self._dirty_candle = self._dirty_candle, set()
            want_positions = self._dirty_positions or slow_due
            want_equity = self._dirty_equity or slow_due
            self._dirty_positions = False
            self._dirty_equity = False

            clients = tuple(c for c in self._clients if not c.closing.is_set())
            if not clients:
                return
            wanted = {c.symbol for c in clients}
            tick_syms &= wanted
            candle_syms &= wanted

            jobs: dict[tuple[str, str | None], Any] = {}
            for sym in tick_syms:
                jobs[("tick", sym)] = self._guard(self._backend.market_tick(sym))
            for sym in candle_syms:
                jobs[("candle", sym)] = self._guard(self._backend.current_candle(sym))
            if want_positions:
                jobs[("positions", None)] = self._guard(self._backend.open_positions())
            if want_equity:
                jobs[("equity", None)] = self._guard(self._backend.equity_kpis())
            results = dict(zip(jobs.keys(), await asyncio.gather(*jobs.values()), strict=True))

            ticks = {k[1]: to_jsonable(v) for k, v in results.items() if k[0] == "tick" and v is not None}
            candles = {k[1]: to_jsonable(v) for k, v in results.items() if k[0] == "candle" and v is not None}
            positions_payload = None
            raw_positions = results.get(("positions", None))
            if isinstance(raw_positions, list):
                plist = to_jsonable(raw_positions)
                self._position_symbols = {p.get("symbol") for p in plist if isinstance(p, dict) and p.get("symbol")}
                positions_payload = {
                    "positions": plist,
                    "unrealized_total": sum(_as_float(p.get("unrealized_pnl"), 0.0) or 0.0
                                            for p in plist if isinstance(p, dict)),
                }
            raw_equity = results.get(("equity", None))
            equity_payload = to_jsonable(raw_equity) if isinstance(raw_equity, dict) else None

            for client in clients:
                if client.closing.is_set():
                    continue
                tick = ticks.get(client.symbol)
                if tick is not None:
                    self._send_if_changed(client, MessageType.MARKET_TICK, tick)
                candle = candles.get(client.symbol)
                if candle is not None:
                    self._send_if_changed(client, MessageType.CANDLE_UPDATE, {"symbol": client.symbol, "candle": candle})
                if positions_payload is not None:
                    self._send_if_changed(client, MessageType.POSITIONS_PNL, positions_payload)
                if equity_payload is not None:
                    self._send_if_changed(client, MessageType.EQUITY_KPIS, equity_payload)

    # ------------------------------------------------------------------ connection handling
    async def serve(self, websocket: Any) -> None:
        """Handle one /ws/terminal connection until it ends. Never raises because of client behaviour."""
        headers = websocket.headers
        if not origin_allowed(headers.get("origin"), headers.get("host"), self._settings.allowed_origins):
            await self._refuse(websocket, 1008, "origin not allowed")
            return
        if not self._running:
            await self._refuse(websocket, 1013, "hub not running")
            return
        if len(self._clients) >= self._settings.ws_max_clients:
            await self._refuse(websocket, 1013, "too many clients")
            return

        client = _Client(websocket, self._settings.ws_send_queue, self._safe(lambda: self._backend.active_symbol, ""))
        self._clients.add(client)  # registered before accept: events arriving while the snapshot is built queue up
        self.stats["connections"] += 1
        self._dirty_tick.add(client.symbol)
        self._dirty_candle.add(client.symbol)
        children: list[asyncio.Task[Any]] = []
        accepted = False
        try:
            await websocket.accept()
            accepted = True
            data = await self._build_snapshot(client)
            await self._send(client, envelope(MessageType.INITIAL_SNAPSHOT, client.seq + 1, data, self._wall()))
            client.seq += 1
            client.writer = asyncio.create_task(self._writer(client), name="ws-hub:writer")
            reader = asyncio.create_task(self._reader(client), name="ws-hub:reader")
            closer = asyncio.create_task(client.closing.wait(), name="ws-hub:closer")
            children = [client.writer, reader, closer]
            await asyncio.wait(children, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.info("ws hub: connection %s ended: %r", client.remote, exc)
        finally:
            self._kick(client, None, "")  # no-op when a close code was already set
            self._clients.discard(client)
            for task in children:
                task.cancel()
            if children:
                await asyncio.gather(*children, return_exceptions=True)
            if accepted and client.close_code is not None:
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(CLOSE_TIMEOUT_SECONDS):
                        await websocket.close(code=client.close_code, reason=client.close_reason)
            client.done.set()

    async def _refuse(self, websocket: Any, code: int, reason: str) -> None:
        with contextlib.suppress(Exception):
            await websocket.accept()
            async with asyncio.timeout(CLOSE_TIMEOUT_SECONDS):
                await websocket.close(code=code, reason=reason)

    @staticmethod
    def _safe(fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception:  # noqa: BLE001
            logger.exception("ws hub: backend accessor failed")
            return default

    async def _send(self, client: _Client, text: str) -> None:
        async with asyncio.timeout(SEND_TIMEOUT_SECONDS):
            await client.ws.send_text(text)

    async def _writer(self, client: _Client) -> None:
        try:
            while True:
                out = await client.queue.get()
                try:
                    text = envelope(out.type, client.seq + 1, out.data, self._wall())
                except Exception:  # noqa: BLE001 - unserialisable message: drop it, keep the connection
                    logger.exception("ws hub: cannot encode %s", out.type)
                    continue
                client.seq += 1
                await self._send(client, text)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self.stats["slow_closed"] += 1
            self._kick(client, 1013, "slow consumer")
        except Exception:  # noqa: BLE001 - socket gone
            self.stats["send_errors"] += 1
            self._kick(client, None, "")

    # ---- snapshot
    async def _build_snapshot(self, client: _Client) -> dict[str, Any]:
        backend = self._backend
        deadline = self._settings.ws_snapshot_deadline_seconds
        started = time.monotonic()
        symbol = client.symbol
        partial: list[str] = []
        failed: set[str] = set()

        def remaining() -> float:
            return max(0.001, deadline - (time.monotonic() - started))

        async def part(name: str, make: Callable[[], Any], default: Any, kind: type) -> Any:
            try:
                value = await asyncio.wait_for(make(), remaining())
                if isinstance(value, kind):
                    return value
                raise TypeError(f"{name}: expected {kind.__name__}, got {type(value).__name__}")
            except Exception as exc:  # noqa: BLE001 - includes TimeoutError
                failed.add(name)
                logger.warning("ws hub: snapshot part %s -> default (%r)", name, exc)
                return default

        async def health() -> Any:
            return backend.market_health()

        async def chart() -> tuple[list[Any], list[Any]]:
            candles = await part("candles", lambda: backend.candles(symbol, SNAPSHOT_CANDLES), [], list)
            if "candles" in failed:
                failed.add("markers")  # cannot know the first candle time
                return [], []
            try:
                since = int(candles[0]["time"]) if candles else 0
            except (KeyError, TypeError, ValueError, IndexError):
                since = 0
            markers = await part("markers", lambda: backend.markers(symbol, since), [], list)
            return candles, markers

        kpis, (candles, markers), positions, trades, logs, news, mh = await asyncio.gather(
            part("equity_kpis", lambda: backend.equity_kpis(), {}, dict),
            chart(),
            part("open_positions", lambda: backend.open_positions(), [], list),
            part("trade_history", lambda: backend.trade_history(SNAPSHOT_TRADES), [], list),
            part("claude_logs", lambda: backend.claude_logs(SNAPSHOT_LOGS), [], list),
            part("news_feed", lambda: backend.news_feed(SNAPSHOT_NEWS), [], list),
            part("market_health", health, None, dict),
        )
        order = ("equity_kpis", "candles", "markers", "open_positions", "trade_history", "claude_logs",
                 "news_feed", "market_health")
        partial = [name for name in order if name in failed]
        if "equity_kpis" not in failed:
            client.last[MessageType.EQUITY_KPIS] = to_jsonable(kpis)
        if "market_health" not in failed:
            client.last[MessageType.MARKET_HEALTH] = to_jsonable(mh)
        return {
            "equity_kpis": kpis,
            "active_symbol": symbol,
            "symbols": list(self._safe(lambda: backend.symbols, [])),
            "timeframe": self._safe(lambda: backend.timeframe, ""),
            "candles": candles,
            "markers": markers,
            "open_positions": positions,
            "trade_history": trades,
            "claude_logs": logs,
            "news_feed": news,
            "market_health": mh,
            "server_time": self._wall(),
            "partial": partial,
        }

    # ---- client -> server
    async def _reader(self, client: _Client) -> None:
        while True:
            try:
                message = await client.ws.receive()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - disconnected
                return
            if message.get("type") == "websocket.disconnect":
                return
            text = message.get("text")
            raw = message.get("bytes")
            if text is None and raw is None:
                continue
            size = len(text.encode("utf-8")) if text is not None else len(raw)
            if size > MAX_CLIENT_MESSAGE_BYTES:
                self._kick(client, 1009, "message too big")
                return
            try:
                if text is None:
                    text = raw.decode("utf-8")
                await self._handle_client_message(client, text)
            except UnicodeDecodeError:
                self._error(client, "BAD_ENCODING", "binary frames must be UTF-8 JSON")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("ws hub: client message failed")
                self._error(client, "INTERNAL", "internal error")

    def _error(self, client: _Client, code: str, message: str) -> None:
        self._enqueue(client, MessageType.ERROR, {"code": code, "message": message})

    async def _handle_client_message(self, client: _Client, text: str) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            self._error(client, "INVALID_JSON", "message is not valid JSON")
            return
        if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
            self._error(client, "BAD_MESSAGE", 'expected a JSON object with a string "type"')
            return
        kind = msg["type"]
        if kind == "PING":
            self._enqueue(client, MessageType.PONG, {"t": msg.get("t"), "server_ts": _iso_utc(self._wall())})
        elif kind == "SUBSCRIBE_SYMBOL":
            symbol = msg.get("symbol")
            symbols = self._safe(lambda: list(self._backend.symbols), [])
            if not isinstance(symbol, str) or symbol not in symbols:
                self._error(client, "UNKNOWN_SYMBOL", f"unknown symbol: {symbol!r}")
                return
            await self._subscribe_symbol(client, symbol)
        else:
            self._error(client, "UNKNOWN_TYPE", f"unknown message type: {kind[:40]!r}")

    async def _subscribe_symbol(self, client: _Client, symbol: str) -> None:
        candles = await self._guard(self._backend.candles(symbol, SNAPSHOT_CANDLES), None)
        if not isinstance(candles, list):
            self._error(client, "BACKEND_UNAVAILABLE", "could not load candles")
            return
        try:
            since = int(candles[0]["time"]) if candles else 0
        except (KeyError, TypeError, ValueError, IndexError):
            since = 0
        markers = await self._guard(self._backend.markers(symbol, since), [])
        if client.closing.is_set():
            return
        # switch + CANDLES are atomic (no await): everything after CANDLES refers to the new symbol
        client.symbol = symbol
        client.queue.remove_all(lambda o: o.type in (MessageType.MARKET_TICK, MessageType.CANDLE_UPDATE))
        client.last.pop(MessageType.MARKET_TICK, None)
        client.last.pop(MessageType.CANDLE_UPDATE, None)
        self._dirty_tick.add(symbol)
        self._dirty_candle.add(symbol)
        self._enqueue(client, MessageType.CANDLES, {
            "symbol": symbol, "candles": candles, "markers": markers if isinstance(markers, list) else []})
