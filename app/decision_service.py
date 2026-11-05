"""Decision service: Topic.CLAUDE_TRIGGER -> Claude decision -> Hard-Risk Engine -> execution.

Order of operations for every SignalPayload (integrator-owned glue):
  1. portfolio context snapshot (equity, open positions, kill-switch state)
  2. ClaudeDecisionCore.decide()        never raises; timeout / rate-limit / bad output -> HOLD
     (the decision + Turkish reasoning is persisted by the core's DecisionLogger BEFORE execution,
      so the terminal can always answer "Claude neden bu işlemi açtı?")
  3. ExecutionEngine.execute()          drift guard -> HardRiskEngine.validate -> entry -> hard SL -> TP
  4. risk verdict + execution status written back onto the same claude_decisions row
Only one decision per symbol is in flight; triggers arriving meanwhile for that symbol are dropped
(the pre-filter cooldown makes this rare) so a slow API call can never queue stale signals.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from app.claude.contracts import DecisionAction, DecisionResult
from app.claude.core import ClaudeDecisionCore
from app.claude.decision_log import DecisionLogger
from app.core.events import EventBus, Subscription, Topic
from app.core.types import utcnow
from app.execution.contracts import ExecutionReport, ExecutionStatus
from app.execution.executor import ExecutionEngine
from app.quant.contracts import SignalPayload

logger = logging.getLogger(__name__)

TERMINAL_DECISION_TOPIC = Topic.CLAUDE_DECISION  # payload: dict (terminal "Live Reasoning Stream")
TERMINAL_EXECUTION_TOPIC = Topic.EXECUTION_REPORT  # payload: ExecutionReport


def risk_verdict_for(report: ExecutionReport) -> tuple[str, list[str]]:
    """Map an execution report to the claude_decisions.risk_verdict column."""
    if report.status in (ExecutionStatus.HOLD, ExecutionStatus.CLOSED_BY_DECISION, ExecutionStatus.NOTHING_TO_CLOSE):
        return "APPROVED", list(report.risk_reasons)
    if report.status in (ExecutionStatus.REJECTED_BY_RISK, ExecutionStatus.KILL_SWITCH,
                         ExecutionStatus.ABORTED_PRICE_DRIFT, ExecutionStatus.ABORTED_STALE_DATA):
        return "REJECTED", list(report.risk_reasons) or [f"{report.status.value}: {report.detail}"]
    if (report.requested_leverage is not None and report.approved_leverage is not None
            and report.approved_leverage < report.requested_leverage) or any(
            r.startswith(("LEVERAGE", "QTY", "EXPOSURE", "MARGIN")) for r in report.risk_reasons):
        return "REDUCED", list(report.risk_reasons)
    return "APPROVED", list(report.risk_reasons)


def metrics_summary(payload: SignalPayload) -> str:
    """One-line technical summary for the terminal's reasoning log."""
    ind = payload.indicators
    parts = []
    if ind.rsi is not None:
        parts.append(f"RSI(14): {ind.rsi:.1f}")
    if ind.obi is not None:
        parts.append(f"OBI: {ind.obi:+.2f}")
    if ind.atr_pct is not None:
        parts.append(f"ATR%: {ind.atr_pct * 100:.2f}")
    if ind.vwap is not None:
        parts.append(f"VWAP Δ: {(ind.close / ind.vwap - 1) * 100:+.2f}%")
    parts.append(f"Fee x{payload.fee_precheck.cover_multiple:.1f}")
    parts.append(f"Skor: {payload.confluence_score:.2f}")
    return " | ".join(parts)


class DecisionService:
    def __init__(
        self,
        core: ClaudeDecisionCore,
        executor: ExecutionEngine,
        bus: EventBus,
        *,
        decision_log: DecisionLogger | None = None,
        portfolio_provider: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._core = core
        self._exec = executor
        self._bus = bus
        self._log = decision_log
        self._portfolio = portfolio_provider
        self._clock = clock
        self._inflight: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        self._sub: Subscription | None = None
        self._workers: set[asyncio.Task[Any]] = set()
        self.stats: dict[str, int] = {"triggers": 0, "dropped_inflight": 0, "errors": 0}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._task is None:
            self._sub = self._bus.subscribe(Topic.CLAUDE_TRIGGER, maxsize=100)
            self._task = asyncio.create_task(self._loop(), name="decision-service")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        if self._sub is not None:
            self._bus.unsubscribe(self._sub)
            self._sub = None

    async def _loop(self) -> None:
        assert self._sub is not None
        while True:
            payload: SignalPayload = await self._sub.get()
            if payload.symbol in self._inflight:
                self.stats["dropped_inflight"] += 1
                logger.info("decision for %s already in flight; trigger %s dropped", payload.symbol, payload.signal_id)
                continue
            self._inflight.add(payload.symbol)  # claim before the task starts: no double dispatch
            task = asyncio.create_task(self._guarded(payload), name=f"decide-{payload.symbol}")
            self._workers.add(task)
            task.add_done_callback(self._workers.discard)

    async def _guarded(self, payload: SignalPayload) -> None:
        try:
            await self._run(payload)
        except Exception:  # noqa: BLE001 - one failure must not stop the service
            self.stats["errors"] += 1
            logger.exception("decision pipeline failed for %s", payload.symbol)
        finally:
            self._inflight.discard(payload.symbol)

    # ------------------------------------------------------------------ core
    async def handle(self, payload: SignalPayload) -> tuple[DecisionResult, ExecutionReport]:
        """Process one trigger synchronously (used by the loop and by tests)."""
        if payload.symbol in self._inflight:
            raise RuntimeError(f"decision already in flight for {payload.symbol}")
        self._inflight.add(payload.symbol)
        try:
            return await self._run(payload)
        finally:
            self._inflight.discard(payload.symbol)

    async def _run(self, payload: SignalPayload) -> tuple[DecisionResult, ExecutionReport]:
        self.stats["triggers"] += 1
        context = await self._portfolio_context(payload.symbol)
        result = await self._core.decide(payload, context)
        self._bus.publish(TERMINAL_DECISION_TOPIC, {
            "decision_id": result.decision_id, "symbol": result.symbol, "action": result.decision.action.value,
            "confidence": str(result.decision.confidence), "leverage": result.decision.leverage,
            "reasoning_tr": result.decision.reasoning_tr, "risk_flags": list(result.decision.risk_flags),
            "fallback": result.fallback.value if result.fallback else None, "latency_ms": result.latency_ms,
            "signal_id": payload.signal_id, "created_at": result.created_at.isoformat(),
            "timestamp": result.created_at.isoformat(), "metrics_summary": metrics_summary(payload),
            "risk_decision": None, "execution_status": None,
        })
        report = await self._exec.execute(payload, result.decision, result.decision_id)
        if self._log is not None and result.decision.action is not DecisionAction.HOLD:
            verdict, reasons = risk_verdict_for(report)
            await self._log.record_risk_verdict(result.decision_id, verdict, reasons,
                                                report.approved_leverage, report.approved_quantity)
        verdict_for_ui = risk_verdict_for(report)[0] if result.decision.action is not DecisionAction.HOLD else None
        if self._log is not None:
            await self._log.record_execution(result.decision_id, report.status.value, report.detail)
        self._bus.publish(TERMINAL_DECISION_TOPIC, {
            "decision_id": result.decision_id, "symbol": result.symbol, "action": result.decision.action.value,
            "confidence": str(result.decision.confidence), "leverage": result.decision.leverage,
            "reasoning_tr": result.decision.reasoning_tr, "risk_flags": list(result.decision.risk_flags),
            "fallback": result.fallback.value if result.fallback else None, "latency_ms": result.latency_ms,
            "signal_id": payload.signal_id, "created_at": result.created_at.isoformat(),
            "timestamp": result.created_at.isoformat(), "metrics_summary": metrics_summary(payload),
            "risk_decision": verdict_for_ui, "execution_status": report.status.value, "update": True,
        })
        self._bus.publish(TERMINAL_EXECUTION_TOPIC, report)
        return result, report

    async def _portfolio_context(self, symbol: str) -> dict[str, Any]:
        if self._portfolio is None:
            return {}
        try:
            value = self._portfolio(symbol)
            if asyncio.iscoroutine(value):
                value = await value
            return dict(value or {})
        except Exception:  # noqa: BLE001 - Claude can decide without it; the risk engine re-checks the account
            logger.exception("portfolio context unavailable for %s", symbol)
            return {"error": "portfolio context unavailable"}


async def broker_portfolio_context(broker: Any, kill_switch: Any, symbol: str) -> dict[str, Any]:
    """Default portfolio provider: compact account view for the prompt (no secrets, JSON-safe)."""
    snap = await broker.fetch_account()
    return {
        "equity": str(snap.equity),
        "available_balance": str(snap.available_balance),
        "unrealized_pnl": str(snap.unrealized_pnl),
        "margin_ratio": None if snap.margin_ratio is None else str(snap.margin_ratio),
        "kill_switch_engaged": bool(getattr(kill_switch, "is_engaged", False)),
        "open_positions": [
            {"symbol": p.symbol, "side": p.side.value, "quantity": str(p.quantity), "entry": str(p.entry_price),
             "unrealized_pnl": str(p.unrealized_pnl), "leverage": p.leverage}
            for p in snap.positions
        ],
        "same_symbol_position_open": any(p.symbol == symbol for p in snap.positions),
    }
