"""Decorators that turn persistence calls into EventBus lifecycle events for the terminal.

The executor / position manager / risk layer only know the TradeRecorder and RiskEventSink protocols.
Wrapping those (instead of editing the trading code) gives the UI position.opened / position.updated /
position.closed / risk.kill_switch events without touching the order path. Publishing never raises.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from app.core.events import EventBus, Topic
from app.core.types import Order, OrderStatus
from app.execution.contracts import ExecutionReport, ManagedPosition, TradeRecorder
from app.risk.protocols import RiskEventSink

logger = logging.getLogger(__name__)

_STOP_PURPOSES = {"STOP_LOSS", "TRAILING_STOP"}


def _f(x: Decimal | None) -> float | None:
    return None if x is None else float(x)


class PublishingTradeRecorder:
    """TradeRecorder decorator: persists via ``inner`` then publishes lifecycle events."""

    def __init__(self, inner: TradeRecorder, bus: EventBus) -> None:
        self._inner = inner
        self._bus = bus
        self._trade_symbol: dict[str, str] = {}  # trade_id -> symbol (bounded by open trades)
        self._open_symbols: set[str] = set()

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            self._bus.publish(topic, payload)
        except Exception:  # noqa: BLE001
            logger.exception("publishing %s failed", topic)

    def remember_trade(self, trade_id: str | None, symbol: str) -> None:
        """Track a trade opened before this process started (adopted by reconciliation after a restart)."""
        if trade_id:
            self._trade_symbol[trade_id] = symbol
        self._open_symbols.add(symbol)

    async def record_order(self, order: Order, purpose: str, decision_id: str | None) -> None:
        await self._inner.record_order(order, purpose, decision_id)
        if (purpose in _STOP_PURPOSES and order.symbol in self._open_symbols
                and order.status is OrderStatus.NEW and order.stop_price is not None):
            self._publish(Topic.POSITION_UPDATED, {
                "symbol": order.symbol, "hard_stop": _f(order.stop_price),
                "trailing_active": purpose == "TRAILING_STOP", "order_id": order.order_id,
            })

    async def open_trade(self, position: ManagedPosition, leverage: int) -> str | None:
        trade_id = await self._inner.open_trade(position, leverage)
        if trade_id:
            self._trade_symbol[trade_id] = position.symbol
        self._open_symbols.add(position.symbol)
        self._publish(Topic.POSITION_OPENED, {
            "trade_id": trade_id, "symbol": position.symbol, "side": position.side.value, "leverage": leverage,
            "quantity": _f(position.quantity), "entry_price": _f(position.entry_price),
            "hard_stop": _f(position.initial_stop), "take_profit": _f(position.take_profit),
            "decision_id": position.decision_id, "opened_at": position.opened_at.isoformat(),
        })
        return trade_id

    async def close_trade(self, trade_id: str | None, exit_price: Decimal, exit_reason: str,
                          fees_paid: Decimal | None) -> Decimal | None:
        net = await self._inner.close_trade(trade_id, exit_price, exit_reason, fees_paid)
        symbol = self._trade_symbol.pop(trade_id, None) if trade_id else None
        if symbol:
            self._open_symbols.discard(symbol)
        self._publish(Topic.POSITION_CLOSED, {
            "trade_id": trade_id, "symbol": symbol, "exit_price": _f(exit_price), "exit_reason": exit_reason,
            "fees_paid": _f(fees_paid), "net_pnl": _f(net),
        })
        return net

    async def record_execution(self, report: ExecutionReport) -> None:
        await self._inner.record_execution(report)

    async def risk_event(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        await self._inner.risk_event(level, code, message, details)
        self._publish(Topic.RISK_EVENT, {"level": level, "code": code, "message": message})


class PublishingRiskSink:
    """RiskEventSink decorator: kill-switch events also go out on Topic.KILL_SWITCH."""

    def __init__(self, inner: RiskEventSink, bus: EventBus) -> None:
        self._inner = inner
        self._bus = bus

    async def record(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        await self._inner.record(level, code, message, details)
        try:
            self._bus.publish(Topic.RISK_EVENT, {"level": level, "code": code, "message": message})
            if code.startswith("KILL_SWITCH"):
                engaged = code != "KILL_SWITCH_RESET"  # ENGAGED / RESTORED / FLAT / NOT_FLAT / *_FAILED
                self._bus.publish(Topic.KILL_SWITCH, {"engaged": engaged, "code": code, "message": message,
                                                      "reason": details.get("reason") if details else None})
        except Exception:  # noqa: BLE001
            logger.exception("publishing risk event %s failed", code)
