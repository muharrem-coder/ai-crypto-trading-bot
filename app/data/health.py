"""Feed health monitoring and the TradingGate (rule R1).

``FeedHealthMonitor`` tracks the last message time of every (symbol, channel). ``check()`` turns
that into LIVE / STALE / DISCONNECTED / CONNECTING, publishes ``Topic.FEED_HEALTH`` on each
transition, and drives the ``TradingGate``:

* a symbol is PAUSED while ANY of its required channels is not LIVE;
* it returns to ACTIVE only after all required channels have been continuously LIVE for
  ``DataSettings.resume_after_live_seconds``;
* the global gate is ACTIVE only if every registered symbol is ACTIVE.

The gate starts PAUSED (fail closed) until data has flowed.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.config.settings import DataSettings
from app.core.events import EventBus, Topic
from app.core.types import utcnow
from app.data.models import (
    DataChannel,
    FeedHealth,
    FeedHealthEvent,
    GateState,
    TradingGateEvent,
)

logger = logging.getLogger(__name__)

_TF_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800, "M": 2_592_000}


def timeframe_seconds(timeframe: str) -> int:
    """Parse ccxt timeframes such as ``1m``, ``5m``, ``15m``, ``1h``, ``1d``."""
    match = re.fullmatch(r"(\d+)([smhdwM])", timeframe.strip())
    if not match or int(match.group(1)) <= 0:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return int(match.group(1)) * _TF_UNITS[match.group(2)]


def candle_stale_after(timeframe: str) -> float:
    """CANDLES staleness threshold: 2 x timeframe + 5 seconds."""
    return 2.0 * timeframe_seconds(timeframe) + 5.0


class TradingGate:
    """Per-symbol and global ACTIVE / PAUSED state. Unknown symbols are never active."""

    def __init__(self, bus: EventBus, wall_clock: Callable[[], datetime] = utcnow) -> None:
        self._bus = bus
        self._wall_clock = wall_clock
        self._states: dict[str, GateState] = {}
        self._reasons: dict[str, tuple[str, ...]] = {}
        self._global = GateState.PAUSED
        self._global_reasons: tuple[str, ...] = ("no symbols registered",)

    # ---- queries -------------------------------------------------------------------------
    @property
    def state(self) -> GateState:
        return self._global

    def symbol_state(self, symbol: str) -> GateState:
        return self._states.get(symbol, GateState.PAUSED)

    def is_active(self, symbol: str | None = None) -> bool:
        """True if trading may proceed: for ``symbol``, or globally when ``symbol`` is None."""
        if symbol is None:
            return self._global is GateState.ACTIVE
        return self._states.get(symbol) is GateState.ACTIVE

    def paused_reasons(self, symbol: str | None = None) -> tuple[str, ...]:
        if symbol is None:
            return self._global_reasons if self._global is GateState.PAUSED else ()
        if symbol not in self._states:
            return (f"{symbol} not registered",)
        return self._reasons[symbol] if self._states[symbol] is GateState.PAUSED else ()

    def paused_symbols(self) -> tuple[str, ...]:
        return tuple(s for s, st in self._states.items() if st is GateState.PAUSED)

    def symbols(self) -> tuple[str, ...]:
        return tuple(self._states)

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self._global.value,
            "reasons": list(self.paused_reasons()),
            "paused_symbols": list(self.paused_symbols()),
            "symbols": {
                s: {"state": st.value, "reasons": list(self.paused_reasons(s))}
                for s, st in self._states.items()
            },
        }

    # ---- updates (used by FeedHealthMonitor) -------------------------------------------
    def add_symbol(self, symbol: str) -> None:
        if symbol in self._states:
            return
        self._states[symbol] = GateState.PAUSED
        self._reasons[symbol] = (f"{symbol} awaiting market data",)
        self._publish_if_changed(symbol, symbol_changed=False)

    def set_symbol(self, symbol: str, state: GateState, reasons: tuple[str, ...]) -> bool:
        """Set a symbol's state. Publishes a TradingGateEvent on a symbol or global transition."""
        if symbol not in self._states:
            self.add_symbol(symbol)
        changed = self._states[symbol] is not state
        self._states[symbol] = state
        self._reasons[symbol] = reasons if state is GateState.PAUSED else ()
        return self._publish_if_changed(symbol, symbol_changed=changed)

    def _publish_if_changed(self, symbol: str, *, symbol_changed: bool) -> bool:
        paused = self.paused_symbols()
        new_global = GateState.ACTIVE if self._states and not paused else GateState.PAUSED
        if new_global is GateState.ACTIVE:
            new_reasons: tuple[str, ...] = ("all required feeds LIVE",)
        elif self._states:
            new_reasons = tuple(r for s in paused for r in self._reasons[s])
        else:
            new_reasons = ("no symbols registered",)
        global_changed = new_global is not self._global
        self._global = new_global
        self._global_reasons = new_reasons if new_global is GateState.PAUSED else ()
        if not (symbol_changed or global_changed):
            return False
        event = TradingGateEvent(
            state=self._global,
            reasons=new_reasons,
            timestamp=self._wall_clock(),
            paused_symbols=paused,
        )
        logger.info(
            "TradingGate %s (global %s) paused=%s reasons=%s",
            symbol,
            self._global.value,
            paused,
            new_reasons,
        )
        self._bus.publish(Topic.TRADING_GATE, event)
        return True


@dataclass(slots=True)
class _Chan:
    symbol: str
    channel: DataChannel
    required: bool
    stale_after: float
    health: FeedHealth = FeedHealth.CONNECTING  # last PUBLISHED health
    last_msg: float | None = None  # monotonic time of the last message
    down: bool = False
    connecting: bool = True
    streak_start: float | None = None  # when the current continuous-LIVE streak began
    attempt: int = 0
    next_retry: float | None = None
    last_error: str = ""


class FeedHealthMonitor:
    def __init__(
        self,
        settings: DataSettings,
        bus: EventBus,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._clock = clock
        self._wall_clock = wall_clock
        self._chans: dict[tuple[str, DataChannel], _Chan] = {}
        self.gate = TradingGate(bus, wall_clock)
        self._candle_stale_after = candle_stale_after(settings.timeframe)

    # ---- registration & input ----------------------------------------------------------
    def register(
        self,
        symbol: str,
        channel: DataChannel,
        required: bool,
        stale_after: float | None = None,
    ) -> None:
        if stale_after is None:
            stale_after = (
                self._candle_stale_after if channel is DataChannel.CANDLES else self._settings.stale_after_seconds
            )
        self._chans[(symbol, channel)] = _Chan(symbol, channel, required, stale_after)
        self.gate.add_symbol(symbol)

    def _get(self, symbol: str, channel: DataChannel) -> _Chan:
        try:
            return self._chans[(symbol, channel)]
        except KeyError:
            raise KeyError(f"feed not registered: {symbol} {channel.value}") from None

    def mark_message(self, symbol: str, channel: DataChannel) -> None:
        st = self._get(symbol, channel)
        now = self._clock()
        gap_too_long = st.last_msg is not None and now - st.last_msg > st.stale_after
        if st.streak_start is None or st.down or st.connecting or gap_too_long:
            st.streak_start = now
        st.last_msg = now
        st.down = False
        st.connecting = False
        st.attempt = 0
        st.next_retry = None
        st.last_error = ""

    def mark_disconnected(
        self,
        symbol: str,
        channel: DataChannel,
        reason: str,
        attempt: int,
        next_retry: float | None,
    ) -> None:
        st = self._get(symbol, channel)
        st.down = True
        st.connecting = False
        st.streak_start = None
        st.attempt = attempt
        st.next_retry = next_retry
        st.last_error = reason

    def mark_connecting(self, symbol: str, channel: DataChannel, attempt: int = 0) -> None:
        st = self._get(symbol, channel)
        st.down = False
        st.connecting = True
        st.streak_start = None
        st.attempt = attempt
        st.next_retry = None

    # ---- evaluation --------------------------------------------------------------------
    def _evaluate(self, st: _Chan, now: float) -> tuple[FeedHealth, float | None, str]:
        age = None if st.last_msg is None else max(0.0, now - st.last_msg)
        if st.down:
            retry = f", retry in {st.next_retry:.1f}s" if st.next_retry is not None else ""
            err = f": {st.last_error}" if st.last_error else ""
            return FeedHealth.DISCONNECTED, age, f"disconnected{err} (attempt {st.attempt}{retry})"
        if st.connecting or st.last_msg is None:
            return FeedHealth.CONNECTING, age, "connecting, no message yet"
        assert age is not None
        if age > st.stale_after:
            return FeedHealth.STALE, age, f"no update for {age:.2f}s (limit {st.stale_after:.2f}s)"
        return FeedHealth.LIVE, age, "ok"

    @staticmethod
    def _gate_reason(st: _Chan, health: FeedHealth, age: float | None) -> str:
        label = f"{st.symbol} {st.channel.value} {health.value}"
        if health is FeedHealth.STALE and age is not None:
            return f"{label} {age:.1f}s"
        if health is FeedHealth.DISCONNECTED:
            return f"{label} (reconnect attempt {st.attempt})"
        return label

    def check(self) -> list[FeedHealthEvent]:
        """Re-evaluate every feed; publish FEED_HEALTH for transitions only; update the gate."""
        now = self._clock()
        events: list[FeedHealthEvent] = []
        computed: dict[tuple[str, DataChannel], tuple[FeedHealth, float | None]] = {}
        for key, st in self._chans.items():
            health, age, reason = self._evaluate(st, now)
            computed[key] = (health, age)
            if health is not FeedHealth.LIVE:
                st.streak_start = None
            if health is not st.health:
                st.health = health
                event = FeedHealthEvent(
                    symbol=st.symbol,
                    channel=st.channel,
                    health=health,
                    age_seconds=age,
                    reason=reason,
                    timestamp=self._wall_clock(),
                    reconnect_attempt=st.attempt,
                    next_retry_seconds=st.next_retry,
                )
                events.append(event)
                self._bus.publish(Topic.FEED_HEALTH, event)

        resume_after = self._settings.resume_after_live_seconds
        for symbol in self.gate.symbols():
            required = [st for st in self._chans.values() if st.symbol == symbol and st.required]
            blocking = [
                self._gate_reason(st, computed[(st.symbol, st.channel)][0], computed[(st.symbol, st.channel)][1])
                for st in required
                if computed[(st.symbol, st.channel)][0] is not FeedHealth.LIVE
            ]
            if not required:
                state, reasons = GateState.PAUSED, (f"{symbol} no required feeds registered",)
            elif blocking:
                state, reasons = GateState.PAUSED, tuple(blocking)
            elif self.gate.symbol_state(symbol) is GateState.ACTIVE:
                state, reasons = GateState.ACTIVE, ()
            else:
                streak = max(st.streak_start if st.streak_start is not None else now for st in required)
                live_for = now - streak
                if live_for >= resume_after:
                    state, reasons = GateState.ACTIVE, ()
                else:
                    state = GateState.PAUSED
                    reasons = (f"{symbol} resuming: LIVE {live_for:.1f}s of {resume_after:.1f}s",)
            self.gate.set_symbol(symbol, state, reasons)
        return events

    # ---- reporting ---------------------------------------------------------------------
    def health(self, symbol: str, channel: DataChannel) -> FeedHealth:
        """Last published health of a feed (as of the most recent ``check()``)."""
        return self._get(symbol, channel).health

    def is_active(self, symbol: str | None = None) -> bool:
        return self.gate.is_active(symbol)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe view of every feed and the gate (terminal + Claude payload ``market_health``)."""
        now = self._clock()
        symbols: dict[str, Any] = {}
        for sym in self.gate.symbols():
            symbols[sym] = {
                "gate": self.gate.symbol_state(sym).value,
                "paused_reasons": list(self.gate.paused_reasons(sym)),
                "channels": {},
            }
        for st in self._chans.values():
            health, age, reason = self._evaluate(st, now)
            symbols[st.symbol]["channels"][st.channel.value] = {
                "health": health.value,
                "age_seconds": None if age is None else round(age, 3),
                "stale_after_seconds": st.stale_after,
                "required": st.required,
                "reconnect_attempts": st.attempt,
                "next_retry_seconds": None if st.next_retry is None else round(st.next_retry, 3),
                "reason": reason,
            }
        return {
            "generated_at": self._wall_clock().isoformat(),
            "gate": self.gate.snapshot(),
            "symbols": symbols,
        }
