"""Execution & order-lifecycle engine.

Strict order for LONG / SHORT (nothing is sent to the broker before step 6):

 1. refuse if the symbol is already managed / an execution is in flight (per-symbol lock)
 2. kill-switch check                                   -> KILL_SWITCH
 3. price-drift guard on a FRESH order book (rule 3)    -> ABORTED_STALE_DATA / ABORTED_PRICE_DRIFT
 4. TradeProposal -> HardRiskEngine.validate            -> REJECTED_BY_RISK (nothing sent)
 5. protective orders computed (before any order is sent)
 6. set_margin_mode(ISOLATED) + set_leverage(approved)
 7. entry (MARKET, or LIMIT polled until filled / timeout, remainder cancelled)
 8. hard STOP_MARKET (reduce-only, filled qty) with fast retries; failure -> emergency market close (rule 4)
 9. TAKE_PROFIT_MARKET (reduce-only); failure is a warning only
10. recorder.open_trade -> position_manager.register -> OPENED_PROTECTED

Only RiskDecision numbers (quantity, leverage, stop, take-profit) are ever used for orders. The exchange
TRAILING_STOP order is deliberately NOT placed: the PositionManager trails by moving the hard stop.
This class never raises (except asyncio.CancelledError, after protecting any open position).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from app.brokers.base import BaseBrokerAdapter
from app.claude.contracts import ClaudeDecision, DecisionAction, EntryType
from app.config.settings import ExecutionSettings
from app.core.errors import BrokerError
from app.core.types import (
    FundingInfo,
    MarginMode,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    PositionSide,
    Ticker,
    TimeInForce,
    utcnow,
)
from app.data.models import OrderBookSnapshot
from app.execution.contracts import ExecutionReport, ExecutionStatus, ManagedPosition, TradeRecorder
from app.execution.drift_guard import check_price_drift
from app.execution.protocols import PositionManagerLike
from app.quant.contracts import SignalPayload
from app.risk.kill_switch import KillSwitch
from app.risk.risk_engine import HardRiskEngine, TradeProposal, Verdict

log = logging.getLogger(__name__)

BookProvider = Callable[[str], "OrderBookSnapshot | None | Awaitable[OrderBookSnapshot | None]"]
FundingProvider = Callable[[str], "FundingInfo | None | Awaitable[FundingInfo | None]"]

_DEAD = (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED)


@dataclass(slots=True)
class _Ctx:
    report: ExecutionReport
    symbol: str
    side: PositionSide | None = None
    entry_sent: bool = False
    entry_order_id: str | None = None
    decision_id: str | None = None
    notes: list[str] = field(default_factory=list)


class ExecutionEngine:
    def __init__(
        self,
        broker: BaseBrokerAdapter,
        risk_engine: HardRiskEngine,
        kill_switch: KillSwitch,
        position_manager: PositionManagerLike,
        recorder: TradeRecorder,
        settings: ExecutionSettings,
        *,
        book_provider: BookProvider,
        funding_provider: FundingProvider | None = None,
        clock: Callable[[], Any] = utcnow,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._broker = broker
        self._risk = risk_engine
        self._ks = kill_switch
        self._pm = position_manager
        self._rec = recorder
        self._cfg = settings
        self._book_provider = book_provider
        self._funding_provider = funding_provider
        self._clock = clock
        self._sleep = sleep
        self._locks: dict[str, asyncio.Lock] = {}

    # ================================================================== public API
    async def execute(self, payload: SignalPayload, decision: ClaudeDecision, decision_id: str | None) -> ExecutionReport:
        """Run one decision to completion. Always returns a report (and records it); never raises."""
        symbol = payload.symbol
        report = ExecutionReport(status=ExecutionStatus.ERROR, symbol=symbol, decision_id=decision_id,
                                 signal_id=payload.signal_id, detail="not started")
        ctx = _Ctx(report=report, symbol=symbol, decision_id=decision_id)
        try:
            action = decision.action
            if action is DecisionAction.HOLD:
                return await self._finish(report, ExecutionStatus.HOLD, "Decision is HOLD; no broker calls made.")
            lock = self._locks.setdefault(symbol, asyncio.Lock())
            if action is DecisionAction.CLOSE:
                async with lock:  # a close waits for an in-flight execution instead of racing it
                    return await self._close(ctx)
            # LONG / SHORT: refuse instead of queueing behind another execution on the same symbol
            if lock.locked():
                report.risk_reasons.append("EXECUTION_IN_PROGRESS: another execution is running for this symbol")
                return await self._finish(report, ExecutionStatus.REJECTED_BY_RISK,
                                          f"{symbol}: execution already in progress, refused.")
            async with lock:
                return await self._open(ctx, payload, decision)
        except asyncio.CancelledError:
            if ctx.entry_sent:
                with contextlib.suppress(Exception):
                    await self._protect_after_error(ctx)
            raise
        except Exception as exc:  # noqa: BLE001 - the contract is: never raise to the caller
            log.exception("execution failed for %s", symbol)
            return await self._error(ctx, exc)

    async def close_symbol(self, symbol: str, reason: str = "MANUAL_UI_CLOSE") -> ExecutionReport:
        """Operator-initiated market close (terminal MARKET CLOSE / kill switch). Same path as a CLOSE decision:
        waits for any in-flight execution on the symbol, closes at market, cancels resting SL/TP, verifies flat."""
        report = ExecutionReport(status=ExecutionStatus.ERROR, symbol=symbol, decision_id=None, signal_id=None,
                                 detail=f"manual close ({reason}) not started")
        ctx = _Ctx(report=report, symbol=symbol, decision_id=None)
        try:
            lock = self._locks.setdefault(symbol, asyncio.Lock())
            async with lock:
                return await self._close(ctx)
        except Exception as exc:  # noqa: BLE001 - never raise to the caller
            log.exception("manual close failed for %s", symbol)
            return await self._error(ctx, exc)

    # ================================================================== CLOSE
    async def _close(self, ctx: _Ctx) -> ExecutionReport:
        rep, symbol = ctx.report, ctx.symbol
        closed: Order | None = None
        last_exc: Exception | None = None
        for attempt in range(1, self._cfg.emergency_close_retries + 1):
            try:
                closed = await self._broker.close_position(symbol)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.error("close_position attempt %s failed for %s: %r", attempt, symbol, exc)
                if attempt < self._cfg.emergency_close_retries:
                    await self._sleep(self._cfg.stop_retry_backoff_seconds * attempt)
        if last_exc is not None:
            # the protective stop is still resting (we have not cancelled anything yet), so the position stays covered
            await self._event("CRITICAL", "CLOSE_FAILED", f"{symbol}: CLOSE decision failed: {last_exc!r}",
                              {"symbol": symbol, "decision_id": ctx.decision_id})
            return await self._finish(rep, ExecutionStatus.ERROR, f"CLOSE failed: {last_exc!r}")
        if closed is not None:
            rep.orders.append(closed)
            await self._rec_order(closed, "EXIT", ctx.decision_id)
            rep.filled_quantity = closed.filled_quantity
            rep.entry_price = closed.average_fill_price
        # Close FIRST, cancel after: cancelling the stop before the close is confirmed would leave a naked position.
        cancelled = 0
        try:
            cancelled = await self._broker.cancel_all_orders(symbol)
        except Exception as exc:  # noqa: BLE001
            rep.warnings.append(f"CANCEL_ORDERS_FAILED: {exc!r}")
            await self._event("WARN", "CANCEL_ORDERS_FAILED", f"{symbol}: cancel after close failed: {exc!r}",
                              {"symbol": symbol})
        try:
            still_open = [p for p in await self._broker.fetch_positions() if p.symbol == symbol]
        except Exception as exc:  # noqa: BLE001
            still_open = []
            rep.warnings.append(f"FLAT_VERIFY_FAILED: {exc!r}")
        if still_open:
            await self._event("CRITICAL", "CLOSE_NOT_FLAT", f"{symbol}: still open after CLOSE", {"symbol": symbol})
            return await self._finish(rep, ExecutionStatus.ERROR, "CLOSE sent but position is still open")
        if closed is None:
            return await self._finish(rep, ExecutionStatus.NOTHING_TO_CLOSE,
                                      f"No open position in {symbol}; cancelled {cancelled} resting order(s).")
        return await self._finish(rep, ExecutionStatus.CLOSED_BY_DECISION,
                                  f"Position closed at market; cancelled {cancelled} resting order(s).")

    # ================================================================== LONG / SHORT
    async def _open(self, ctx: _Ctx, payload: SignalPayload, decision: ClaudeDecision) -> ExecutionReport:
        rep, symbol, did = ctx.report, ctx.symbol, ctx.decision_id
        side = PositionSide.LONG if decision.action is DecisionAction.LONG else PositionSide.SHORT
        ctx.side = rep.side = side
        rep.requested_leverage = decision.leverage

        if self._pm.is_managed(symbol):
            rep.risk_reasons.append("ALREADY_MANAGED: a position for this symbol is already managed")
            return await self._finish(rep, ExecutionStatus.REJECTED_BY_RISK, f"{symbol}: already managed, refused.")
        if payload.direction is not side:
            rep.risk_reasons.append(f"DIRECTION_MISMATCH: signal {payload.direction.value} vs decision {side.value}")
            return await self._finish(rep, ExecutionStatus.REJECTED_BY_RISK, "Decision direction contradicts signal.")

        # ---- 2. kill switch
        if self._ks.is_engaged:
            return await self._finish(rep, ExecutionStatus.KILL_SWITCH, f"Kill switch engaged: {self._ks.reason}")

        # ---- 3. drift guard on a fresh book
        book = await self._get_book(symbol)
        check = check_price_drift(payload, decision, book, self._clock(), self._cfg)
        rep.drift_atr = check.drift_atr
        if not check.ok or check.reference_price is None:
            status = check.status or ExecutionStatus.ABORTED_PRICE_DRIFT
            return await self._finish(rep, status, check.detail)

        spec = await self._broker.get_instrument(symbol)
        is_limit = decision.entry_type is EntryType.LIMIT
        ref = check.reference_price
        if is_limit:  # never round a buy limit up / a sell limit down
            ref = (self._broker.round_price_down(spec, ref) if side is PositionSide.LONG
                   else self._broker.round_price_up(spec, ref))

        # ---- 4. hard risk validation
        proposal = TradeProposal(
            symbol=symbol, side=side, entry_price=ref, stop_loss=decision.stop_loss, take_profit=decision.take_profit,
            leverage=decision.leverage, confidence=decision.confidence, source="claude", decision_id=did,
            margin_mode=MarginMode.ISOLATED,
            entry_order_type=OrderType.LIMIT if is_limit else OrderType.MARKET,
        )
        account = await self._broker.fetch_account()
        funding = await self._get_funding(symbol, rep)
        ticker = await self._get_ticker(symbol, rep)
        rd = await self._risk.validate(proposal, account, spec, funding, ticker)
        if rd.verdict is Verdict.REJECTED or rd.approved_quantity <= 0:
            rep.risk_reasons = list(rd.reasons)
            status = ExecutionStatus.KILL_SWITCH if "KILL_SWITCH" in rd.reason_codes else ExecutionStatus.REJECTED_BY_RISK
            return await self._finish(rep, status, "Rejected by hard risk engine: " + "; ".join(rd.reasons))
        rep.risk_reasons = list(rd.reasons)
        rep.approved_quantity = rd.approved_quantity
        rep.approved_leverage = rd.approved_leverage

        # ---- 5. protective orders (computed up-front so a failure here cannot happen with a position open)
        protective = self._risk.compute_protective_orders(rd, proposal, spec)
        stop_tpl = next(r for r in protective if r.order_type is OrderType.STOP_MARKET)
        tp_tpl = next(r for r in protective if r.order_type is OrderType.TAKE_PROFIT_MARKET)

        # ---- 6. margin mode + leverage (approved values only)
        await self._broker.set_margin_mode(symbol, MarginMode.ISOLATED)
        await self._broker.set_leverage(symbol, rd.approved_leverage)

        # ---- 7. entry
        entry_req = OrderRequest(
            symbol=symbol, side=side.entry_side, order_type=OrderType.LIMIT if is_limit else OrderType.MARKET,
            quantity=rd.approved_quantity, price=ref if is_limit else None, time_in_force=TimeInForce.GTC,
            client_order_id=f"{did[:20]}-en" if did else None,
            metadata={"role": "entry", "decision_id": did, "source": "claude"},
        )
        ctx.entry_sent = True
        entry_order = await self._broker.create_order(entry_req)
        ctx.entry_order_id = entry_order.order_id
        await self._rec_order(entry_order, "ENTRY", did)
        final_order, filled, avg = await self._settle_entry(entry_order, symbol, side)
        if final_order is not entry_order:
            await self._rec_order(final_order, "ENTRY", did)  # final state (upsert by order_id)
        rep.orders.append(final_order)
        if filled <= 0:
            return await self._finish(
                rep, ExecutionStatus.ENTRY_NOT_FILLED,
                f"Entry not filled within {self._cfg.limit_entry_timeout_seconds}s; remainder cancelled, no position.")
        rep.filled_quantity = filled
        avg_price = avg if avg is not None else ref
        rep.entry_price = avg_price
        if filled < rd.approved_quantity:
            rep.warnings.append(f"PARTIAL_FILL: filled {filled} of {rd.approved_quantity}; protecting filled quantity only")

        # ---- 8. hard stop (rule 4)
        stop_req = replace(stop_tpl, quantity=filled)
        stop_order, stop_err = await self._place_stop(stop_req)
        if stop_order is None:
            rep.stop_loss = stop_req.stop_price
            ok = await self._emergency_close(
                ctx, f"hard stop could not be placed after {self._cfg.stop_place_retries} attempts: {stop_err!r}")
            if ok:
                return await self._finish(rep, ExecutionStatus.NAKED_POSITION_CLOSED,
                                          "Hard stop could not be placed; position closed at market and verified flat.")
            return await self._finish(rep, ExecutionStatus.NAKED_CLOSE_FAILED,
                                      "Hard stop could not be placed AND emergency close failed; kill switch engaged.")
        rep.orders.append(stop_order)
        rep.stop_loss = stop_req.stop_price
        await self._rec_order(stop_order, "STOP_LOSS", did)

        # ---- 9. take profit (non fatal)
        tp_req = replace(tp_tpl, quantity=filled)
        tp_order: Order | None = None
        try:
            tp_order = await self._broker.create_order(tp_req)
            if tp_order.status in _DEAD:
                raise BrokerError(f"take-profit order came back {tp_order.status.value}")
            rep.orders.append(tp_order)
            rep.take_profit = tp_req.stop_price
            await self._rec_order(tp_order, "TAKE_PROFIT", did)
        except Exception as exc:  # noqa: BLE001
            tp_order = None
            msg = f"TAKE_PROFIT_FAILED: {exc!r}; position remains protected by the hard stop"
            rep.warnings.append(msg)
            log.warning("%s %s", symbol, msg)
            await self._event("WARN", "TAKE_PROFIT_FAILED", f"{symbol}: {msg}",
                              {"symbol": symbol, "decision_id": did, "stop_order_id": stop_order.order_id})

        # ---- 10. hand over
        mp = ManagedPosition(
            symbol=symbol, side=side, quantity=filled, entry_price=avg_price, initial_stop=stop_req.stop_price,
            take_profit=tp_req.stop_price if tp_order is not None else None,
            atr=Decimal(str(payload.indicators.atr)), fees=spec.fees, tick_size=spec.tick_size,
            stop_order_id=stop_order.order_id, take_profit_order_id=tp_order.order_id if tp_order else None,
            decision_id=did, trade_id=None, opened_at=self._clock(), current_stop=stop_req.stop_price,
            best_price=avg_price,
        )
        try:
            mp.trade_id = await self._rec.open_trade(mp, rd.approved_leverage)
        except Exception as exc:  # noqa: BLE001 - recorders should not raise, but the position is protected regardless
            rep.warnings.append(f"OPEN_TRADE_RECORD_FAILED: {exc!r}")
        rep.trade_id = mp.trade_id
        try:
            await self._pm.register(mp)
        except Exception as exc:  # noqa: BLE001
            rep.warnings.append(f"POSITION_MANAGER_REGISTER_FAILED: {exc!r}")
            await self._event("CRITICAL", "POSITION_MANAGER_REGISTER_FAILED",
                              f"{symbol}: protected position was not registered for management: {exc!r}",
                              {"symbol": symbol, "decision_id": did, "stop_order_id": stop_order.order_id})
        return await self._finish(
            rep, ExecutionStatus.OPENED_PROTECTED,
            f"{side.value} {filled} {symbol} @ {avg_price} opened; hard stop {stop_req.stop_price} resting"
            + (f", take-profit {tp_req.stop_price}" if tp_order is not None else ", NO take-profit"))

    # ================================================================== entry settlement
    async def _settle_entry(self, order: Order, symbol: str, side: PositionSide) -> tuple[Order, Decimal, Decimal | None]:
        """Wait for the entry to reach a terminal state, cancel any remainder, and report what is REALLY filled.

        The broker position is the source of truth for the filled quantity (an order that disappeared from the open
        list may have filled or been cancelled; ``fetch_open_orders`` cannot tell).
        """
        cfg = self._cfg
        cur = order
        gone = False
        waited = 0.0
        while not cur.status.is_terminal and waited < cfg.limit_entry_timeout_seconds:
            await self._sleep(cfg.entry_fill_poll_seconds)
            waited += cfg.entry_fill_poll_seconds
            try:
                refreshed = await self._refresh_order(cur)
            except Exception as exc:  # noqa: BLE001 - transient; keep polling until the timeout
                log.warning("entry poll failed for %s: %r", symbol, exc)
                continue
            if refreshed is None:
                gone = True
                break
            cur = refreshed
        if not cur.status.is_terminal and not gone:
            cur = await self._cancel_entry(cur, symbol)

        filled, avg = cur.filled_quantity, cur.average_fill_price
        try:
            positions = await self._broker.fetch_positions()
            pos = next((p for p in positions if p.symbol == symbol and p.side is side), None)
            filled = pos.quantity if pos is not None else Decimal(0)
            if pos is not None and avg is None:
                avg = pos.entry_price
        except Exception as exc:  # noqa: BLE001 - fall back to the order's own report
            log.error("fetch_positions failed while settling entry for %s: %r", symbol, exc)
        return cur, filled, avg

    async def _refresh_order(self, order: Order) -> Order | None:
        open_orders = await self._broker.fetch_open_orders(order.symbol)
        for o in open_orders:
            if o.order_id == order.order_id:
                return o
        fetch_order = getattr(self._broker, "fetch_order", None)
        if fetch_order is not None:
            got = fetch_order(order.order_id, order.symbol)
            return await got if inspect.isawaitable(got) else got
        return None  # no longer open and no way to ask: caller reconciles through the position

    async def _cancel_entry(self, order: Order, symbol: str) -> Order:
        """Cancel the unfilled remainder of the entry. If that cannot be confirmed, halt trading (kill switch)."""
        last_exc: Exception | None = None
        for attempt in range(1, self._cfg.emergency_close_retries + 1):
            try:
                return await self._broker.cancel_order(order.order_id, symbol)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                try:  # "not found" because it filled / was cancelled meanwhile is fine
                    if all(o.order_id != order.order_id for o in await self._broker.fetch_open_orders(symbol)):
                        return order
                except Exception:  # noqa: BLE001
                    pass
                if attempt < self._cfg.emergency_close_retries:
                    await self._sleep(self._cfg.stop_retry_backoff_seconds * attempt)
        await self._event("CRITICAL", "ENTRY_CANCEL_FAILED",
                          f"{symbol}: could not cancel resting entry {order.order_id}: {last_exc!r}",
                          {"symbol": symbol, "order_id": order.order_id})
        await self._engage_kill(f"ENTRY_ORDER_UNCANCELLABLE {symbol} {order.order_id}: {last_exc!r}")
        return order

    # ================================================================== stop + emergency
    async def _place_stop(self, req: OrderRequest) -> tuple[Order | None, Exception | None]:
        last: Exception | None = None
        n = self._cfg.stop_place_retries
        for attempt in range(1, n + 1):
            try:
                o = await self._broker.create_order(req)
                if o.status in _DEAD:
                    raise BrokerError(f"stop order came back {o.status.value}")
                return o, None
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.error("stop placement attempt %s/%s failed for %s: %r", attempt, n, req.symbol, exc)
                if attempt < n:
                    await self._sleep(self._cfg.stop_retry_backoff_seconds * attempt)
        return None, last

    async def _emergency_close(self, ctx: _Ctx, reason: str) -> bool:
        """Flatten ``ctx.symbol`` at market and verify. On failure engage the kill switch. True when verified flat."""
        symbol, did = ctx.symbol, ctx.decision_id
        flat = False
        last_exc: Exception | None = None
        for attempt in range(1, self._cfg.emergency_close_retries + 1):
            try:
                order = await self._broker.close_position(symbol)
                if order is not None:
                    ctx.report.orders.append(order)
                    await self._rec_order(order, "EXIT", did)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.error("emergency close attempt %s failed for %s: %r", attempt, symbol, exc)
            try:
                flat = not any(p.symbol == symbol for p in await self._broker.fetch_positions())
            except Exception as exc:  # noqa: BLE001
                flat = False
                last_exc = exc
            if flat:
                break
            if attempt < self._cfg.emergency_close_retries:
                await self._sleep(self._cfg.stop_retry_backoff_seconds * attempt)
        if flat:
            with contextlib.suppress(Exception):
                await self._broker.cancel_all_orders(symbol)  # stray entry remainder / TP
            await self._event("CRITICAL", "NAKED_POSITION_CLOSED",
                              f"{symbol}: position closed at market because {reason}",
                              {"symbol": symbol, "decision_id": did, "reason": reason})
            return True
        await self._event("CRITICAL", "NAKED_CLOSE_FAILED",
                          f"{symbol}: UNPROTECTED POSITION and emergency close failed ({reason}): {last_exc!r}",
                          {"symbol": symbol, "decision_id": did, "reason": reason})
        await self._engage_kill(f"NAKED_POSITION_UNPROTECTED {symbol}: {reason}")
        return False

    async def _engage_kill(self, reason: str) -> None:
        try:
            await self._ks.engage(reason)
        except Exception as exc:  # noqa: BLE001
            log.critical("kill switch engage failed: %r (%s)", exc, reason)
            await self._event("CRITICAL", "KILL_SWITCH_ENGAGE_FAILED", f"{reason}: {exc!r}", {"reason": reason})

    # ================================================================== error path
    async def _error(self, ctx: _Ctx, exc: Exception) -> ExecutionReport:
        rep = ctx.report
        rep.warnings.append(f"UNEXPECTED_ERROR: {exc!r}")
        if ctx.entry_sent:
            try:
                outcome = await self._protect_after_error(ctx)
                rep.warnings.append(f"POSITION_CHECK: {outcome}")
            except Exception as exc2:  # noqa: BLE001
                rep.warnings.append(f"POSITION_CHECK_FAILED: {exc2!r}")
        return await self._finish(rep, ExecutionStatus.ERROR, f"Unexpected error: {exc!r}")

    async def _protect_after_error(self, ctx: _Ctx) -> str:
        """After an unexpected failure with an entry in flight: make sure nothing is left open and unprotected."""
        symbol = ctx.symbol
        try:
            positions = await self._broker.fetch_positions()
        except Exception as exc:  # noqa: BLE001
            await self._event("CRITICAL", "POSITION_UNVERIFIABLE",
                              f"{symbol}: cannot verify position after error: {exc!r}", {"symbol": symbol})
            await self._engage_kill(f"POSITION_UNVERIFIABLE {symbol}: {exc!r}")
            return "UNVERIFIED_KILL_SWITCH"
        pos = next((p for p in positions if p.symbol == symbol), None)
        if pos is None:
            with contextlib.suppress(Exception):
                await self._broker.cancel_all_orders(symbol)  # a still-resting LIMIT entry must not fill later
            return "FLAT"
        try:
            orders = await self._broker.fetch_open_orders(symbol)
        except Exception:  # noqa: BLE001
            orders = []
        protected = any(
            o.reduce_only and o.order_type is OrderType.STOP_MARKET and o.side is pos.side.exit_side
            and not o.status.is_terminal for o in orders
        )
        if protected:
            if ctx.entry_order_id is not None:  # drop an unfilled entry remainder, keep the stop
                with contextlib.suppress(Exception):
                    await self._broker.cancel_order(ctx.entry_order_id, symbol)
            return "PROTECTED"
        ok = await self._emergency_close(ctx, "unexpected error left an open position without a resting stop")
        return "EMERGENCY_CLOSED" if ok else "EMERGENCY_CLOSE_FAILED_KILL_SWITCH"

    # ================================================================== helpers
    async def _get_book(self, symbol: str) -> OrderBookSnapshot | None:
        try:
            res = self._book_provider(symbol)
            if inspect.isawaitable(res):
                res = await res
            return res
        except Exception as exc:  # noqa: BLE001 - treated as "no book" -> ABORTED_STALE_DATA
            log.warning("book_provider failed for %s: %r", symbol, exc)
            return None

    async def _get_funding(self, symbol: str, rep: ExecutionReport) -> FundingInfo | None:
        if self._funding_provider is None:
            return None
        try:
            res = self._funding_provider(symbol)
            if inspect.isawaitable(res):
                res = await res
            return res
        except Exception as exc:  # noqa: BLE001
            rep.warnings.append(f"FUNDING_UNAVAILABLE: {exc!r}")
            return None

    async def _get_ticker(self, symbol: str, rep: ExecutionReport) -> Ticker | None:
        try:
            return await self._broker.fetch_ticker(symbol)
        except Exception as exc:  # noqa: BLE001
            rep.warnings.append(f"TICKER_UNAVAILABLE: {exc!r}")
            return None

    async def _rec_order(self, order: Order, purpose: str, decision_id: str | None) -> None:
        try:
            await self._rec.record_order(order, purpose, decision_id)
        except Exception:  # noqa: BLE001
            log.exception("recorder.record_order failed")

    async def _event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        try:
            await self._rec.risk_event(level, code, message, details)
        except Exception:  # noqa: BLE001
            log.exception("recorder.risk_event failed")

    async def _finish(self, report: ExecutionReport, status: ExecutionStatus, detail: str) -> ExecutionReport:
        report.status = status
        report.detail = detail
        report.timestamp = self._clock()
        try:
            await self._rec.record_execution(report)
        except Exception:  # noqa: BLE001
            log.exception("recorder.record_execution failed")
        return report
