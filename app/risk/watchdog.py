"""Risk watchdog: periodic account evaluation, stop-loss presence verification, broker liveness."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.config.settings import RiskSettings
from app.core.types import Order, OrderType, Position
from app.risk.kill_switch import KillSwitch
from app.risk.protocols import BrokerLike, RiskEventSink

log = logging.getLogger(__name__)

RepairStop = Callable[[Position], Awaitable[Any]]


@dataclass(slots=True)
class WatchdogReport:
    ok: bool = True
    engaged: bool = False
    repaired: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    error: str | None = None


def has_resting_stop(position: Position, orders: list[Order]) -> bool:
    return any(
        o.symbol == position.symbol
        and o.order_type is OrderType.STOP_MARKET
        and o.reduce_only
        and o.side is position.side.exit_side
        and not o.status.is_terminal
        for o in orders
    )


class RiskWatchdog:
    def __init__(
        self,
        broker: BrokerLike,
        kill_switch: KillSwitch,
        sink: RiskEventSink,
        settings: RiskSettings,
        repair_stop: RepairStop | None = None,
        *,
        max_consecutive_broker_failures: int = 10,
    ) -> None:
        self._broker = broker
        self._ks = kill_switch
        self._sink = sink
        self._settings = settings
        self._repair = repair_stop
        self._max_failures = max_consecutive_broker_failures
        self._task: asyncio.Task[None] | None = None
        self.ticks = 0
        self.error_count = 0
        self.consecutive_broker_failures = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="risk-watchdog")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self._settings.watchdog_interval_seconds)

    # ------------------------------------------------------------------ tick
    async def run_once(self) -> WatchdogReport:
        self.ticks += 1
        report = WatchdogReport()
        try:
            try:
                snapshot = await self._broker.fetch_account()
                orders = await self._broker.fetch_open_orders(None)
            except Exception as exc:  # noqa: BLE001 - any broker failure counts toward liveness
                self.consecutive_broker_failures += 1
                self.error_count += 1
                report.ok, report.error = False, repr(exc)
                log.error("watchdog broker failure #%d: %r", self.consecutive_broker_failures, exc)
                if self.consecutive_broker_failures >= self._max_failures and not self._ks.is_engaged:
                    await self._ks.engage("BROKER_UNREACHABLE")
                report.engaged = self._ks.is_engaged
                return report
            self.consecutive_broker_failures = 0

            status = await self._ks.evaluate(snapshot)
            report.engaged = status.engaged
            if status.engaged:
                return report  # kill switch owns flattening

            for position in snapshot.positions:
                if has_resting_stop(position, orders):
                    continue
                await self._handle_unprotected(position, report)
        except Exception as exc:  # noqa: BLE001
            self.error_count += 1
            report.ok, report.error = False, repr(exc)
            log.exception("watchdog tick failed")
            await self._emit("ERROR", "WATCHDOG_TICK_FAILED", f"watchdog tick failed: {exc!r}", {})
        report.engaged = self._ks.is_engaged
        return report

    async def _handle_unprotected(self, position: Position, report: WatchdogReport) -> None:
        sym = position.symbol
        await self._emit("CRITICAL", "UNPROTECTED_POSITION",
                         f"{sym} {position.side.value} has no resting reduce-only stop order", {"symbol": sym})
        repaired = False
        if self._repair is not None:
            try:
                await self._repair(position)
                repaired = has_resting_stop(position, await self._broker.fetch_open_orders(sym))
            except Exception as exc:  # noqa: BLE001
                self.error_count += 1
                await self._emit("CRITICAL", "STOP_REPAIR_FAILED", f"stop repair for {sym} raised {exc!r}", {"symbol": sym})
        if repaired:
            report.repaired.append(sym)
            await self._emit("WARN", "STOP_REPAIRED", f"stop restored for {sym}", {"symbol": sym})
            return
        try:
            await self._broker.close_position(sym)
            report.closed.append(sym)
            await self._emit("CRITICAL", "UNPROTECTED_POSITION_CLOSED",
                             f"{sym} closed because its stop could not be restored", {"symbol": sym})
        except Exception as exc:  # noqa: BLE001
            self.error_count += 1
            report.ok = False
            await self._emit("CRITICAL", "UNPROTECTED_CLOSE_FAILED", f"could not close {sym}: {exc!r}", {"symbol": sym})
            await self._ks.engage(f"UNPROTECTED_POSITION {sym}")

    async def _emit(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        if level in ("CRITICAL", "ERROR"):
            log.error("%s: %s", code, message)
        try:
            await self._sink.record(level, code, message, details)
        except Exception:  # noqa: BLE001
            log.exception("risk event sink failed")
