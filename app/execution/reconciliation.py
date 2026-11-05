"""Exchange <-> local state reconciliation.

After a restart or a network blip the exchange's real state (positions, resting SL/TP) and our local tracking
(PositionManager, trades table) must converge, and NO position may ever stay without a resting reduce-only hard
stop. One pass (``run_once``) does, in this order:

1. read the exchange (positions + open orders + account). A failed read is an error (``ok=False``) and the pass
   stops there: a failed fetch is NEVER interpreted as "flat".
2. positions the manager tracks but the exchange no longer has (or holds on the other side) are handed to
   ``PositionManager.reconcile()`` which books the exit; they are listed in ``local_closed``.
3. managed positions: a resting reduce-only STOP_MARKET on the exit side with quantity >= position quantity must
   exist; a missing stop is re-placed at the manager's ``current_stop``; a changed quantity (partial fill, manual
   change) gets a replacement stop (new one placed first, old one cancelled afterwards).
4. orphans (exchange position the manager does not track): security checks first (CROSS margin, notional above
   ``max_orphan_notional_pct`` x equity, ``orphan_action=CLOSE``, unknown instrument) -> market close + CRITICAL;
   otherwise ADOPT_AND_PROTECT: reuse a resting reduce-only stop or place one at entry -/+ ``orphan_stop_pct``
   (rounded away from entry to the tick; a stop that would trigger immediately closes the position instead),
   recover the OPEN trades row (decision_id / trade id / initial stop) or open a new one, register in the manager.
5. a stop that cannot be placed after the configured retries (3) -> emergency market close; if that fails too ->
   kill switch ("RECON_NAKED_POSITION").
6. DB trades still OPEN for symbols that are flat on the exchange and unmanaged are closed (``local_closed``).
7. reduce-only orders resting for symbols with no position (stale SL/TP after a manual close) are cancelled.

Every pass publishes a summary dict on ``Topic.RISK_EVENT`` (INFO when clean, WARN / CRITICAL otherwise) and
keeps it in ``last_report``. Messages are bilingual (Turkish / English).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.brokers.base import BaseBrokerAdapter
from app.config.settings import OrphanAction, Settings
from app.core.errors import BrokerError
from app.core.events import EventBus, Topic
from app.core.types import (
    MarginMode,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    utcnow,
)
from app.db.models import Account, Trade, TradeStatus
from app.db.session import get_session
from app.execution.contracts import ExecutionStatus, ManagedPosition, TradeRecorder
from app.risk.rounding import ceil_to_step, floor_to_step

log = logging.getLogger(__name__)

ZERO = Decimal(0)
_DEAD = (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED)
_CLOSED_OK = (ExecutionStatus.CLOSED_BY_DECISION, ExecutionStatus.NOTHING_TO_CLOSE)
_LEVEL_RANK = {"INFO": 0, "WARN": 1, "CRITICAL": 2}
STARTUP_RETRIES = 3


@dataclass(slots=True)
class ReconciliationReport:
    started_at: datetime
    reason: str = "periodic"
    positions_seen: int = 0
    adopted: list[str] = field(default_factory=list)
    stops_placed: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    local_closed: list[str] = field(default_factory=list)
    orphan_orders_cancelled: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # symbols with an execution in flight (re-checked next pass)
    level: str = "INFO"
    ok: bool = True

    @property
    def clean(self) -> bool:
        return not (self.adopted or self.stops_placed or self.closed or self.local_closed
                    or self.orphan_orders_cancelled or self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(), "reason": self.reason, "positions_seen": self.positions_seen,
            "adopted": list(self.adopted), "stops_placed": list(self.stops_placed), "closed": list(self.closed),
            "local_closed": list(self.local_closed), "orphan_orders_cancelled": list(self.orphan_orders_cancelled),
            "errors": list(self.errors), "skipped": list(self.skipped), "level": self.level, "ok": self.ok,
        }


@dataclass(frozen=True, slots=True)
class _DbTrade:
    trade_id: str
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    initial_stop: Decimal | None
    decision_id: str | None
    leverage: int
    opened_at: datetime


class ReconciliationService:
    def __init__(
        self,
        broker: BaseBrokerAdapter,
        position_manager: Any,
        executor: Any,
        recorder: TradeRecorder,
        kill_switch: Any,
        bus: EventBus,
        settings: Settings,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._broker = broker
        self._pm = position_manager
        self._executor = executor
        self._recorder = recorder
        self._ks = kill_switch
        self._bus = bus
        self._settings = settings
        self._cfg = settings.recon
        self._factory = session_factory
        self._clock = clock
        self._sleep = sleep
        self._run_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self.last_report: ReconciliationReport | None = None
        self.runs = 0
        self.failed_runs = 0

    # ================================================================== lifecycle
    def start(self) -> None:
        """Start the periodic loop (idempotent). The first pass happens after ``interval_seconds``."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._loop(), name="reconciliation")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        while True:
            await self._sleep(self._cfg.interval_seconds)
            try:
                await self.run_once("periodic")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one failed run must never kill the loop
                self.failed_runs += 1
                log.exception("reconciliation pass crashed; will retry in %ss", self._cfg.interval_seconds)

    async def run_startup(self) -> ReconciliationReport:
        """Startup pass; when errors remain retry up to 3 times with exponential backoff. Never raises."""
        report = await self._safe_run("startup")
        for attempt in range(STARTUP_RETRIES):
            if not report.errors:
                break
            await self._sleep(min(30.0, 1.0 * (2 ** attempt)))
            report = await self._safe_run(f"startup-retry-{attempt + 1}")
        return report

    async def _safe_run(self, reason: str) -> ReconciliationReport:
        try:
            return await self.run_once(reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.failed_runs += 1
            log.exception("reconciliation pass crashed")
            report = ReconciliationReport(started_at=self._clock(), reason=reason, ok=False, level="CRITICAL",
                                          errors=[f"reconciliation crashed: {exc!r}"])
            self.last_report = report
            return report

    # ================================================================== one pass
    async def run_once(self, reason: str = "periodic") -> ReconciliationReport:
        async with self._run_lock:
            self.runs += 1
            report = ReconciliationReport(started_at=self._clock(), reason=reason)
            try:
                await self._run(report)
            finally:
                report.ok = not report.errors
                if report.errors:
                    self._escalate(report, "WARN")
                elif not report.clean:
                    self._escalate(report, "WARN")
                self.last_report = report
                self._publish_summary(report)
            return report

    async def _run(self, report: ReconciliationReport) -> None:
        broker = self._broker
        # ---- 1. read the exchange
        try:
            positions = list(await broker.fetch_positions())
            await broker.fetch_open_orders(None)  # first read doubles as a liveness probe before taking locks
        except Exception as exc:  # noqa: BLE001
            msg = f"borsa durumu okunamadi / exchange state unavailable: {exc!r}"
            report.errors.append(msg)
            await self._event("WARN", "RECON_BROKER_ERROR", msg, {"error": repr(exc)}, report)
            return
        report.positions_seen = len(positions)
        equity: Decimal | None = None
        try:
            equity = (await broker.fetch_account()).equity
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"fetch_account failed (notional check skipped): {exc!r}")

        # ---- 2. local positions the exchange no longer has
        exch = {p.symbol: p for p in positions}
        stale = [m.symbol for m in self._pm.positions()
                 if m.symbol not in exch or exch[m.symbol].side is not m.side]
        stale = [s for s in stale if not self._in_flight(s, report)]
        if stale:
            await self._close_local(stale, report)

        # ---- 3. managed positions (stop present, correct quantity)
        async with self._manager_guard():
            try:
                orders = list(await broker.fetch_open_orders(None))  # fresh read under the manager lock
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"fetch_open_orders failed: {exc!r}")
                return
            for pos in positions:
                managed = self._managed(pos.symbol)
                if managed is None or managed.side is not pos.side:
                    continue
                try:
                    await self._verify_managed(managed, pos, orders, report)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    report.errors.append(f"{pos.symbol}: managed verification failed: {exc!r}")
                    log.exception("managed verification failed for %s", pos.symbol)

        # ---- 4. orphans
        db_trades = await self._db_open_trades()
        for pos in positions:
            if self._managed(pos.symbol) is not None:
                continue
            if self._in_flight(pos.symbol, report):
                continue
            try:
                await self._handle_orphan(pos, orders, equity, db_trades, report)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not block the others
                report.errors.append(f"{pos.symbol}: orphan handling failed: {exc!r}")
                log.exception("orphan handling failed for %s", pos.symbol)
                await self._emergency(pos.symbol, f"orphan handling crashed: {exc!r}", report)

        # ---- 5/6. DB-only trades and stale reduce-only orders against the FINAL exchange state
        try:
            final_positions = list(await broker.fetch_positions())
            final_orders = list(await broker.fetch_open_orders(None))
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"final state read failed (cleanup skipped): {exc!r}")
            return
        held = {p.symbol for p in final_positions}
        await self._close_db_only(held, db_trades, report)
        await self._cancel_stale_orders(held, final_orders, report)

    # ================================================================== helpers: state access
    @contextlib.asynccontextmanager
    async def _manager_guard(self) -> AsyncIterator[None]:
        """Serialise with the PositionManager's own tick (it moves stops under this lock)."""
        lock = getattr(self._pm, "_lock", None)
        if isinstance(lock, asyncio.Lock):
            async with lock:
                yield
        else:
            yield

    def _managed(self, symbol: str) -> ManagedPosition | None:
        return next((m for m in self._pm.positions() if m.symbol == symbol), None)

    def _in_flight(self, symbol: str, report: ReconciliationReport) -> bool:
        locks = getattr(self._executor, "_locks", None)
        lock = locks.get(symbol) if isinstance(locks, dict) else None
        if lock is not None and lock.locked():
            if symbol not in report.skipped:
                report.skipped.append(symbol)
            log.info("reconciliation: %s has an execution in flight, skipped this pass", symbol)
            return True
        return False

    @staticmethod
    def _is_exit_stop(order: Order, symbol: str, side: PositionSide) -> bool:
        return (order.symbol == symbol and order.order_type is OrderType.STOP_MARKET and order.reduce_only
                and order.side is side.exit_side and not order.status.is_terminal)

    @staticmethod
    def _is_exit_tp(order: Order, symbol: str, side: PositionSide) -> bool:
        return (order.symbol == symbol and order.order_type is OrderType.TAKE_PROFIT_MARKET and order.reduce_only
                and order.side is side.exit_side and not order.status.is_terminal)

    @staticmethod
    def _covers(order: Order, quantity: Decimal) -> bool:
        raw = order.raw or {}
        info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
        close_all = str(raw.get("closePosition") or info.get("closePosition") or "").lower() == "true"
        return close_all or order.quantity >= quantity

    # ================================================================== events
    @staticmethod
    def _escalate(report: ReconciliationReport, level: str) -> None:
        if _LEVEL_RANK[level] > _LEVEL_RANK[report.level]:
            report.level = level

    async def _event(self, level: str, code: str, message: str, details: dict[str, Any],
                     report: ReconciliationReport) -> None:
        self._escalate(report, level)
        if level == "CRITICAL":
            log.critical("%s: %s", code, message)
        else:
            log.warning("%s: %s", code, message)
        try:
            await self._recorder.risk_event(level, code, message, details)
        except Exception:  # noqa: BLE001
            log.exception("recorder.risk_event failed")

    def _publish_summary(self, report: ReconciliationReport) -> None:
        if report.clean:
            msg = (f"Mutabakat temiz / reconciliation clean ({report.reason}): "
                   f"{report.positions_seen} pozisyon / position(s)")
        else:
            msg = (f"Mutabakat ({report.reason}): adopted={report.adopted} stops={report.stops_placed} "
                   f"closed={report.closed} local_closed={report.local_closed} "
                   f"stale_orders={report.orphan_orders_cancelled} errors={report.errors}")
        try:
            self._bus.publish(Topic.RISK_EVENT, {"level": report.level, "code": "RECON_SUMMARY", "message": msg,
                                                 "details": report.to_dict()})
        except Exception:  # noqa: BLE001
            log.exception("publishing reconciliation summary failed")

    async def _record_order(self, order: Order, purpose: str, decision_id: str | None) -> None:
        try:
            await self._recorder.record_order(order, purpose, decision_id)
        except Exception:  # noqa: BLE001
            log.exception("recorder.record_order failed")

    # ================================================================== step 2: local positions gone
    async def _close_local(self, stale: list[str], report: ReconciliationReport) -> None:
        try:
            await self._pm.reconcile()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"position_manager.reconcile failed: {exc!r}")
            return
        for symbol in stale:
            if self._managed(symbol) is None:
                report.local_closed.append(symbol)
                await self._event("WARN", "RECON_LOCAL_CLOSED",
                                  f"{symbol}: borsada pozisyon yok, yerel islem kapatildi / "
                                  f"no exchange position, local trade closed", {"symbol": symbol}, report)
            else:
                report.errors.append(f"{symbol}: tracked locally but not on exchange and reconcile did not close it")

    # ================================================================== step 3: managed positions
    async def _verify_managed(self, managed: ManagedPosition, pos: Position, orders: list[Order],
                              report: ReconciliationReport) -> None:
        symbol = pos.symbol
        stops = [o for o in orders if self._is_exit_stop(o, symbol, pos.side)]
        qty_changed = pos.quantity != managed.quantity
        covering = [o for o in stops if self._covers(o, pos.quantity)]
        if not qty_changed and covering:
            if managed.stop_order_id not in {o.order_id for o in covering}:
                best = max(covering, key=lambda o: o.stop_price or ZERO) if pos.side is PositionSide.LONG else \
                    min(covering, key=lambda o: o.stop_price or Decimal("Infinity"))
                log.warning("%s: manager stop id %s not resting, re-linked to %s", symbol, managed.stop_order_id,
                            best.order_id)
                managed.stop_order_id = best.order_id
            return

        why = (f"miktar degisti {managed.quantity}->{pos.quantity} / quantity changed" if qty_changed
               else "koruyucu stop yok / protective stop missing")
        new, exc = await self._place_stop_with_retries(symbol, pos.side, pos.quantity, managed.current_stop)
        if new is None:
            await self._event("CRITICAL", "RECON_STOP_FAILED",
                              f"{symbol}: stop konamadi ({why}): {exc!r} / could not place stop", {"symbol": symbol},
                              report)
            await self._emergency(symbol, f"managed stop could not be placed: {exc!r}", report)
            return
        old_qty = managed.quantity
        managed.quantity = pos.quantity
        managed.stop_order_id = new.order_id
        report.stops_placed.append(symbol)
        await self._record_order(new, "TRAILING_STOP" if managed.breakeven_locked else "STOP_LOSS", managed.decision_id)
        await self._event("WARN", "RECON_STOP_RESTORED" if not qty_changed else "RECON_STOP_REPLACED",
                          f"{symbol}: stop yeniden kondu @ {managed.current_stop} qty={pos.quantity} ({why}) / "
                          f"stop re-placed", {"symbol": symbol, "stop": managed.current_stop,
                                              "quantity": pos.quantity, "previous_quantity": old_qty}, report)
        for old in stops:  # new stop first, old ones afterwards (never naked in between)
            if old.order_id == new.order_id:
                continue
            try:
                cancelled = await self._broker.cancel_order(old.order_id, symbol)
                await self._record_order(cancelled, "STOP_LOSS", managed.decision_id)
            except Exception as cexc:  # noqa: BLE001 - reduce-only orders clamp to the position; harmless
                report.errors.append(f"{symbol}: old stop {old.order_id} not cancelled: {cexc!r}")

    # ================================================================== stop placement / emergency
    async def _place_stop_with_retries(self, symbol: str, side: PositionSide, quantity: Decimal,
                                       stop_price: Decimal) -> tuple[Order | None, Exception | None]:
        attempts = max(1, self._settings.execution.stop_place_retries)
        backoff = self._settings.execution.stop_retry_backoff_seconds
        last: Exception | None = None
        req = OrderRequest(symbol=symbol, side=side.exit_side, order_type=OrderType.STOP_MARKET, quantity=quantity,
                           stop_price=stop_price, reduce_only=True)
        for attempt in range(1, attempts + 1):
            try:
                order = await self._broker.create_order(req)
                if order.status in _DEAD:
                    raise BrokerError(f"stop order came back {order.status.value}")
                return order, None
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.error("recon stop placement %s/%s failed for %s: %r", attempt, attempts, symbol, exc)
                if attempt < attempts:
                    await self._sleep(backoff * attempt)
        return None, last

    async def _emergency(self, symbol: str, why: str, report: ReconciliationReport) -> bool:
        """Unprotectable position: market close; if that fails too, engage the kill switch."""
        return await self._close_position(symbol, "RECON_NAKED_POSITION", f"RECON_NAKED_POSITION {symbol}: {why}",
                                          f"{symbol}: korumasiz pozisyon piyasadan kapatiliyor / naked position "
                                          f"closed at market ({why})", report)

    async def _close_position(self, symbol: str, close_reason: str, kill_reason: str, message: str,
                              report: ReconciliationReport) -> bool:
        try:
            rep = await self._executor.close_symbol(symbol, close_reason)
            status = getattr(rep, "status", None)
            flat = status in _CLOSED_OK
        except Exception as exc:  # noqa: BLE001
            rep, flat = None, False
            log.exception("close_symbol raised for %s", symbol)
            report.errors.append(f"{symbol}: close raised {exc!r}")
        if flat:
            report.closed.append(symbol)
            await self._event("CRITICAL", close_reason, message, {"symbol": symbol}, report)
            if self._managed(symbol) is None:  # a managed trade is booked by the PositionManager itself
                await self._close_db_trades_for(symbol, getattr(rep, "entry_price", None))
            return True
        report.errors.append(f"{symbol}: close failed ({close_reason}): {getattr(rep, 'detail', 'no report')}")
        await self._event("CRITICAL", f"{close_reason}_FAILED",
                          f"{symbol}: KAPATMA BASARISIZ, pozisyon acik / close FAILED, position still open "
                          f"-> kill switch ({kill_reason})", {"symbol": symbol}, report)
        try:
            await self._ks.engage(kill_reason)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"kill switch engage failed: {exc!r}")
            log.critical("kill switch engage failed: %r", exc)
        return False

    # ================================================================== step 4: orphans
    async def _handle_orphan(self, pos: Position, orders: list[Order], equity: Decimal | None,
                             db_trades: list[_DbTrade], report: ReconciliationReport) -> None:
        symbol, cfg = pos.symbol, self._cfg
        violation: str | None = None
        if cfg.orphan_action is OrphanAction.CLOSE:
            violation = "orphan_action=CLOSE"
        elif cfg.close_if_cross_margin and pos.margin_mode is MarginMode.CROSS:
            violation = "CROSS margin"
        elif equity is not None and equity > 0 and pos.notional > cfg.max_orphan_notional_pct * equity:
            violation = (f"notional {pos.notional} > {cfg.max_orphan_notional_pct} x equity {equity}")
        if violation is not None:
            await self._close_position(
                symbol, "RECON_SECURITY_CLOSE", f"RECON_SECURITY_CLOSE_FAILED {symbol}: {violation}",
                f"{symbol}: guvenlik ihlali ({violation}), pozisyon kapatildi / security violation, closed", report)
            return

        try:
            spec = await self._broker.get_instrument(symbol)
        except Exception as exc:  # noqa: BLE001
            await self._close_position(
                symbol, "RECON_SECURITY_CLOSE", f"RECON_SECURITY_CLOSE_FAILED {symbol}: unknown instrument",
                f"{symbol}: enstruman bilinmiyor ({exc!r}), pozisyon kapatildi / unknown instrument, closed", report)
            return

        bid, ask, mark = await self._reference_prices(pos)
        side = pos.side
        existing = [o for o in orders if self._is_exit_stop(o, symbol, side) and o.stop_price is not None
                    and self._covers(o, pos.quantity)]
        stop_order: Order | None = None
        if existing:
            stop_order = (max if side is PositionSide.LONG else min)(existing, key=lambda o: o.stop_price or ZERO)
            stop_price = stop_order.stop_price
            assert stop_price is not None
            stop_id = stop_order.order_id
            placed = False
        else:
            raw = pos.entry_price * (1 - cfg.orphan_stop_pct if side is PositionSide.LONG else 1 + cfg.orphan_stop_pct)
            stop_price = (floor_to_step(raw, spec.tick_size) if side is PositionSide.LONG
                          else ceil_to_step(raw, spec.tick_size))
            triggers = (stop_price >= min(bid, mark)) if side is PositionSide.LONG else (stop_price <= max(ask, mark))
            if stop_price <= 0 or triggers:
                await self._close_position(
                    symbol, "RECON_NAKED_POSITION", f"RECON_NAKED_POSITION {symbol}: stop would trigger immediately",
                    f"{symbol}: hesaplanan stop {stop_price} hemen tetiklenir (bid={bid} ask={ask} mark={mark}), "
                    f"pozisyon kapatildi / stop would trigger immediately, closed", report)
                return
            new, exc = await self._place_stop_with_retries(symbol, side, pos.quantity, stop_price)
            if new is None:
                await self._event("CRITICAL", "RECON_STOP_FAILED",
                                  f"{symbol}: orphan stop konamadi: {exc!r} / could not protect orphan",
                                  {"symbol": symbol}, report)
                await self._emergency(symbol, f"orphan stop could not be placed: {exc!r}", report)
                return
            stop_order, stop_id, placed = new, new.order_id, True
            report.stops_placed.append(symbol)
            await self._record_order(new, "STOP_LOSS", None)

        tp_orders = [o for o in orders if self._is_exit_tp(o, symbol, side) and o.stop_price is not None]
        tp_order = tp_orders[0] if tp_orders else None

        db = next((t for t in db_trades if t.symbol == symbol and t.side is side), None)
        initial_stop = db.initial_stop if db is not None and db.initial_stop is not None else stop_price
        distance = abs(pos.entry_price - stop_price)
        on_loss_side = stop_price < pos.entry_price if side is PositionSide.LONG else stop_price > pos.entry_price
        atr_mult = self._settings.quant.atr_stop_multiple
        atr = distance / atr_mult if on_loss_side and distance > 0 else pos.entry_price * cfg.orphan_stop_pct / atr_mult

        managed = ManagedPosition(
            symbol=symbol, side=side, quantity=pos.quantity, entry_price=pos.entry_price, initial_stop=initial_stop,
            take_profit=tp_order.stop_price if tp_order is not None else None, atr=atr, fees=spec.fees,
            tick_size=spec.tick_size, stop_order_id=stop_id,
            take_profit_order_id=tp_order.order_id if tp_order is not None else None,
            decision_id=db.decision_id if db is not None else None, trade_id=db.trade_id if db is not None else None,
            opened_at=(pos.opened_at or (db.opened_at if db is not None else None) or self._clock()),
        )
        if db is None:
            try:
                managed.trade_id = await self._recorder.open_trade(managed, max(1, int(pos.leverage)))
            except Exception:  # noqa: BLE001
                log.exception("recorder.open_trade failed for adopted %s", symbol)
        else:
            remember = getattr(self._recorder, "remember_trade", None)  # terminal events for the adopted trade
            if callable(remember):
                remember(managed.trade_id, symbol)
        await self._pm.register(managed)
        # register() resets current_stop/best_price to initial_stop/entry: restore what is really resting
        managed.current_stop = stop_price
        favourable = mark > managed.best_price if side is PositionSide.LONG else mark < managed.best_price
        if favourable:
            managed.best_price = mark
        report.adopted.append(symbol)
        await self._event(
            "WARN", "RECON_ORPHAN_ADOPTED",
            f"{symbol}: sahipsiz {side.value} pozisyon devralindi, stop {'yeni konuldu' if placed else 'mevcut'} "
            f"@ {stop_price} (qty={pos.quantity}, entry={pos.entry_price}) / orphan position adopted, "
            f"stop {'placed' if placed else 'reused'}",
            {"symbol": symbol, "side": side.value, "quantity": pos.quantity, "entry_price": pos.entry_price,
             "stop": stop_price, "stop_order_id": stop_id, "stop_placed": placed, "trade_id": managed.trade_id,
             "from_db": db is not None}, report)

    async def _reference_prices(self, pos: Position) -> tuple[Decimal, Decimal, Decimal]:
        """(bid, ask, mark) for the immediate-trigger check; falls back to the position's mark price."""
        try:
            t = await self._broker.fetch_ticker(pos.symbol)
            return t.bid, t.ask, (t.mark if t.mark is not None else t.last)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: ticker unavailable (%r), using position mark %s", pos.symbol, exc, pos.mark_price)
            return pos.mark_price, pos.mark_price, pos.mark_price

    # ================================================================== DB access
    async def _db_open_trades(self) -> list[_DbTrade]:
        if self._factory is None:
            return []
        try:
            async with get_session(self._factory) as session:
                stmt = (select(Trade).join(Account, Trade.account_id == Account.id)
                        .where(Trade.status == TradeStatus.OPEN, Account.trading_mode == self._settings.trading_mode)
                        .order_by(Trade.opened_at.desc()))
                rows = list((await session.execute(stmt)).scalars())
                return [_DbTrade(str(t.id), t.symbol, t.side, t.quantity, t.entry_price, t.initial_stop,
                                 str(t.decision_id) if t.decision_id else None, t.leverage, t.opened_at)
                        for t in rows]
        except Exception:  # noqa: BLE001
            log.exception("reading open trades failed")
            return []

    async def _close_db_trades_for(self, symbol: str, exit_price: Decimal | None) -> None:
        """A position this service closed: its OPEN trades row (if any) must not stay open forever."""
        for t in await self._db_open_trades():
            if t.symbol == symbol:
                await self._close_trade_row(t, exit_price)

    async def _close_trade_row(self, trade: _DbTrade, exit_price: Decimal | None) -> None:
        price = exit_price
        if price is None:
            try:
                tk = await self._broker.fetch_ticker(trade.symbol)
                price = tk.mid
            except Exception:  # noqa: BLE001
                price = trade.entry_price
        try:
            await self._recorder.close_trade(trade.trade_id, price, "MANUAL", None)
        except Exception:  # noqa: BLE001
            log.exception("recorder.close_trade failed for %s", trade.symbol)

    async def _close_db_only(self, held: set[str], db_trades: list[_DbTrade], report: ReconciliationReport) -> None:
        for t in db_trades:
            if t.symbol in held or self._managed(t.symbol) is not None or self._in_flight(t.symbol, report):
                continue
            await self._close_trade_row(t, None)
            if t.symbol not in report.local_closed:
                report.local_closed.append(t.symbol)
            await self._event("WARN", "RECON_DB_TRADE_CLOSED",
                              f"{t.symbol}: DB'de acik islem var ama borsada pozisyon yok, kapatildi / "
                              f"open DB trade without exchange position closed", {"symbol": t.symbol,
                                                                                  "trade_id": t.trade_id}, report)

    # ================================================================== step 7: stale reduce-only orders
    async def _cancel_stale_orders(self, held: set[str], orders: list[Order], report: ReconciliationReport) -> None:
        for o in orders:
            if not o.reduce_only or o.symbol in held or o.status.is_terminal:
                continue
            if self._in_flight(o.symbol, report):
                continue
            try:
                cancelled = await self._broker.cancel_order(o.order_id, o.symbol)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{o.symbol}: stale order {o.order_id} not cancelled: {exc!r}")
                continue
            report.orphan_orders_cancelled.append(f"{o.symbol}:{o.order_id}")
            await self._record_order(cancelled, "TAKE_PROFIT" if o.order_type is OrderType.TAKE_PROFIT_MARKET
                                     else "STOP_LOSS", None)
            await self._event("WARN", "RECON_STALE_ORDER_CANCELLED",
                              f"{o.symbol}: pozisyon yok, artik reduce-only emir {o.order_id} iptal edildi / "
                              f"stale reduce-only order cancelled", {"symbol": o.symbol, "order_id": o.order_id},
                              report)
