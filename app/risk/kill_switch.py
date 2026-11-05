"""Account-level kill switch: daily drawdown, margin ratio and consecutive-loss circuit breaker."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.config.settings import RiskSettings
from app.core.errors import KillSwitchEngaged, RiskViolation
from app.core.types import AccountSnapshot, utcnow
from app.risk.protocols import BrokerLike, KillSwitchState, KillSwitchStore, RiskEventSink

log = logging.getLogger(__name__)
ZERO = Decimal(0)
RESET_CONFIRMATION = "RESET"


@dataclass(frozen=True, slots=True)
class KillSwitchStatus:
    daily_pnl: Decimal
    daily_drawdown_pct: Decimal  # (day_start_equity - equity) / day_start_equity; negative when in profit
    limit: Decimal
    engaged: bool
    reason: str | None = None
    margin_ratio: Decimal | None = None
    consecutive_losses: int = 0
    day_start_equity: Decimal | None = None
    peak_equity: Decimal | None = None
    peak_drawdown_pct: Decimal = ZERO


class KillSwitch:
    def __init__(
        self,
        broker: BrokerLike,
        store: KillSwitchStore,
        sink: RiskEventSink,
        settings: RiskSettings,
        *,
        auto_reset_next_day: bool = False,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        close_attempts: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._broker = broker
        self._store = store
        self._sink = sink
        self._settings = settings
        self._auto_reset = auto_reset_next_day
        self._clock = clock
        self._sleep = sleep
        self._close_attempts = close_attempts
        self._backoff = backoff_seconds
        self._state = KillSwitchState()
        self._lock = asyncio.Lock()
        self.flat_verified: bool | None = None  # result of the last flatten attempt

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def is_engaged(self) -> bool:
        return self._state.engaged

    @property
    def reason(self) -> str | None:
        return self._state.reason

    def _today(self) -> date:
        return self._clock().date()

    async def start(self) -> None:
        """Load persisted state (call once on boot). An engaged switch stays engaged across restarts."""
        loaded = await self._store.load()
        if loaded is not None:
            self._state = loaded
        if self._state.engaged:
            await self._emit("CRITICAL", "KILL_SWITCH_RESTORED",
                             f"Kill switch restored as ENGAGED after restart: {self._state.reason}", {})
            await self._maybe_auto_reset()

    def assert_trading_allowed(self) -> None:
        if self._state.engaged:
            raise KillSwitchEngaged(f"Trading halted: {self._state.reason}")

    # ------------------------------------------------------------- evaluation
    async def evaluate(self, snapshot: AccountSnapshot) -> KillSwitchStatus:
        await self._maybe_auto_reset()
        st = self._state
        today = self._today()
        equity = snapshot.equity
        changed = False
        if st.trading_day != today or st.day_start_equity is None:
            if st.trading_day != today:
                st.consecutive_losses = 0  # the loss streak is a per-day counter
            st.trading_day = today
            st.day_start_equity = equity
            st.peak_equity = equity
            changed = True
        if st.peak_equity is None or equity > st.peak_equity:
            st.peak_equity = equity
            changed = True
        start = st.day_start_equity
        daily_pnl = equity - start
        drawdown = (start - equity) / start if start > 0 else ZERO
        peak = st.peak_equity or equity
        peak_dd = (peak - equity) / peak if peak > 0 else ZERO

        if not st.engaged:
            trigger: str | None = None
            if drawdown >= self._settings.max_daily_drawdown_pct:
                trigger = f"DAILY_DRAWDOWN {drawdown:.4%} >= {self._settings.max_daily_drawdown_pct:.4%}"
            elif snapshot.margin_ratio is not None and snapshot.margin_ratio >= self._settings.max_margin_ratio:
                trigger = f"MARGIN_RATIO {snapshot.margin_ratio} >= {self._settings.max_margin_ratio}"
            elif st.consecutive_losses >= self._settings.max_consecutive_losses:
                trigger = f"CONSECUTIVE_LOSSES {st.consecutive_losses} >= {self._settings.max_consecutive_losses}"
            if trigger:
                await self.engage(trigger)
                changed = False  # engage persisted
            elif changed:
                await self._store.save(st)
        elif snapshot.positions:
            # engaged but not flat (earlier flatten failed): keep trying
            await self._flatten()

        return KillSwitchStatus(
            daily_pnl=daily_pnl,
            daily_drawdown_pct=drawdown,
            limit=self._settings.max_daily_drawdown_pct,
            engaged=st.engaged,
            reason=st.reason,
            margin_ratio=snapshot.margin_ratio,
            consecutive_losses=st.consecutive_losses,
            day_start_equity=start,
            peak_equity=peak,
            peak_drawdown_pct=peak_dd,
        )

    async def record_trade_result(self, net_pnl: Decimal) -> None:
        st = self._state
        if net_pnl < 0:
            st.consecutive_losses += 1
        elif net_pnl > 0:
            st.consecutive_losses = 0
        await self._store.save(st)
        if not st.engaged and st.consecutive_losses >= self._settings.max_consecutive_losses:
            await self.engage(f"CONSECUTIVE_LOSSES {st.consecutive_losses} >= {self._settings.max_consecutive_losses}")

    # ---------------------------------------------------------------- engage
    async def engage(self, reason: str) -> bool:
        """Engage the switch and flatten the account. Idempotent: returns False if already engaged."""
        async with self._lock:
            if self._state.engaged:
                return False
            now = self._clock()
            st = self._state
            st.engaged = True
            st.reason = reason
            st.engaged_at = now
            st.engaged_day = now.date()
            await self._store.save(st)  # persist BEFORE touching the broker: a crash mid-flatten must stay halted
            await self._emit("CRITICAL", "KILL_SWITCH_ENGAGED", f"Kill switch engaged: {reason}", {"reason": reason})
        await self._flatten()
        return True

    async def _flatten(self) -> bool:
        try:
            cancelled = await self._broker.cancel_all_orders(None)
            log.warning("kill switch cancelled %s orders", cancelled)
        except Exception as exc:  # noqa: BLE001 - must continue to closing positions
            await self._emit("CRITICAL", "KILL_SWITCH_CANCEL_FAILED", f"cancel_all_orders failed: {exc!r}", {})
        flat = False
        for attempt in range(self._close_attempts):
            try:
                await self._broker.close_all_positions()
                flat = not await self._broker.fetch_positions()
            except Exception as exc:  # noqa: BLE001
                await self._emit("CRITICAL", "KILL_SWITCH_CLOSE_FAILED",
                                 f"close_all_positions attempt {attempt + 1} failed: {exc!r}", {"attempt": attempt + 1})
                flat = False
            if flat:
                break
            if attempt < self._close_attempts - 1:
                await self._sleep(self._backoff * (2 ** attempt))
        self.flat_verified = flat
        if flat:
            await self._emit("CRITICAL", "KILL_SWITCH_FLAT", "All positions closed and verified flat.", {})
        else:
            await self._emit("CRITICAL", "KILL_SWITCH_NOT_FLAT",
                             f"Positions STILL OPEN after {self._close_attempts} close attempts. MANUAL INTERVENTION REQUIRED.", {})
        return flat

    # ----------------------------------------------------------------- reset
    async def manual_reset(self, operator: str, confirm: str) -> None:
        if confirm != RESET_CONFIRMATION:
            raise RiskViolation("RESET_DENIED", 'manual reset requires confirm="RESET"')
        if not operator or not operator.strip():
            raise RiskViolation("RESET_DENIED", "manual reset requires an operator name")
        await self._reset(f"manual reset by {operator}")

    async def _maybe_auto_reset(self) -> None:
        st = self._state
        if self._auto_reset and st.engaged and st.engaged_day is not None and self._today() > st.engaged_day:
            await self._reset("auto reset at next UTC trading day")

    async def _reset(self, how: str) -> None:
        st = self._state
        previous = st.reason
        st.engaged = False
        st.reason = None
        st.engaged_at = None
        st.engaged_day = None
        st.consecutive_losses = 0
        st.day_start_equity = None  # rebased on the next evaluate() so the old drawdown cannot instantly re-trigger
        st.peak_equity = None
        st.trading_day = None
        self.flat_verified = None
        await self._store.save(st)
        await self._emit("WARN", "KILL_SWITCH_RESET", f"Kill switch reset ({how}); previous reason: {previous}", {})

    async def _emit(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        if level == "CRITICAL":
            log.critical("%s: %s", code, message)
        try:
            await self._sink.record(level, code, message, details)
        except Exception:  # noqa: BLE001 - the sink must never break risk handling
            log.exception("risk event sink failed")
