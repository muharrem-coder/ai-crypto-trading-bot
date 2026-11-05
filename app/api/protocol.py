"""/ws/terminal wire protocol + the backend interface the WebSocket hub and REST routes depend on.

Every server -> client message is an ENVELOPE:
    {"type": <MessageType>, "seq": <int, per-connection, starts at 1>, "ts": <ISO-8601 UTC>, "data": {...}}
Client -> server messages:
    {"type": "PING", "t": <client ms timestamp>}          -> server replies {"type": "PONG", ..., "data": {"t": t}}
    {"type": "SUBSCRIBE_SYMBOL", "symbol": "BTC/USDT:USDT"} -> server switches the active chart symbol for
                                                             that client and sends a fresh CANDLES message
All numbers that are money/prices are sent as JSON numbers (float) for the UI; the DB keeps Decimals.
Timestamps for chart candles/markers are UNIX seconds (TradingView Lightweight Charts format).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol


class MessageType(str, Enum):
    INITIAL_SNAPSHOT = "INITIAL_SNAPSHOT"
    # event-driven (sent immediately)
    CLAUDE_DECISION = "CLAUDE_DECISION"  # one claude_logs item
    EXECUTION_REPORT = "EXECUTION_REPORT"
    POSITION_OPENED = "POSITION_OPENED"  # one open_positions item
    POSITION_UPDATED = "POSITION_UPDATED"  # one open_positions item (stop moved, BE locked)
    POSITION_CLOSED = "POSITION_CLOSED"  # {"symbol", "trade": <trade_history item>}
    KILL_SWITCH = "KILL_SWITCH"  # {"engaged", "reason", "system_status"}
    NEWS = "NEWS"  # one news_feed item
    MARKET_HEALTH = "MARKET_HEALTH"  # {"system_status", "gate", "paused_symbols", "reasons"}
    RISK_EVENT = "RISK_EVENT"  # {"level", "code", "message"}
    CANDLES = "CANDLES"  # {"symbol", "candles": [...], "markers": [...]} (after SUBSCRIBE_SYMBOL)
    # throttled / coalesced (max 5 Hz per connection, latest value wins)
    MARKET_TICK = "MARKET_TICK"  # {"symbol", "bid", "ask", "last", "mark", "spread", "funding_rate"}
    CANDLE_UPDATE = "CANDLE_UPDATE"  # {"symbol", "candle": {time, open, high, low, close}}
    POSITIONS_PNL = "POSITIONS_PNL"  # {"positions": [open_positions items], "unrealized_total"}
    EQUITY_KPIS = "EQUITY_KPIS"  # equity_kpis object
    # control
    PONG = "PONG"
    ERROR = "ERROR"


THROTTLED_TYPES = frozenset({MessageType.MARKET_TICK, MessageType.CANDLE_UPDATE,
                             MessageType.POSITIONS_PNL, MessageType.EQUITY_KPIS})
THROTTLE_INTERVAL_SECONDS = 0.2  # 5 Hz


class SystemStatus(str, Enum):
    AUTONOMOUS = "AUTONOMOUS 24/7"
    HALTED = "HALTED"  # kill switch engaged
    DATA_PAUSED = "DATA_PAUSED"  # stale / disconnected market data


# ---------------------------------------------------------------- JSON shapes (documentation + helpers)
#
# equity_kpis = {
#   "equity": float, "equity_change_pct": float (vs day start), "unrealized_pnl": float,
#   "realized_pnl_24h": float, "trades_24h": int, "fees_paid_24h": float,
#   "win_rate": float | None (0..1, last 24h closed trades), "profit_factor": float | None,
#   "daily_drawdown_pct": float (0.0042 == 0.42%), "daily_drawdown_limit_pct": float,
#   "system_status": SystemStatus value, "kill_switch_engaged": bool, "kill_switch_reason": str | None,
#   "trading_mode": "PAPER" | "LIVE", "exchange": str, "active_symbol": str,
#   "funding_rate": float | None, "claude_model": str, "min_net_profit_usd": float,
# }
# candle   = {"time": int unix s, "open": float, "high": float, "low": float, "close": float}
# marker   = {"time": int, "position": "belowBar"|"aboveBar", "color": hex, "shape": "arrowUp"|"arrowDown",
#             "text": str}   (LONG entry: belowBar/arrowUp/#00C805, SHORT entry: aboveBar/arrowDown/#FF3B30,
#                               exits: TP green / SL red, text like "TP +$14.20")
# open_positions item = {
#   "symbol", "side": "LONG"|"SHORT", "leverage": int, "quantity": float, "entry_price": float,
#   "mark_price": float, "liquidation_price": float|None, "hard_stop": float|None, "take_profit": float|None,
#   "trailing_active": bool (break-even locked / trailing), "est_fee": float (round trip, taker),
#   "unrealized_pnl": float (gross), "net_pnl": float (after est_fee), "opened_at": iso|None,
#   "decision_id": str|None }
# trade_history item = {
#   "trade_id", "symbol", "side", "opened_at", "closed_at", "entry_price", "exit_price", "quantity",
#   "leverage", "gross_pnl", "fees_paid", "funding_paid", "net_pnl", "r_multiple"|None,
#   "exit_reason": "TAKE_PROFIT"|"STOP_LOSS"|"TRAILING_STOP"|"MANUAL"|"KILL_SWITCH"|..., "trigger": str }
# claude_logs item = {
#   "decision_id", "timestamp": iso, "symbol", "action", "confidence": float, "leverage": int|None,
#   "metrics_summary": str (e.g. "RSI(14): 28.4 | OBI: +0.31 | ATR%: 0.52 | Fee x4.2"),
#   "reasoning_tr": str, "risk_flags": [str], "risk_decision": "APPROVED"|"REDUCED"|"REJECTED"|None,
#   "execution_status": str|None, "fallback": str|None, "latency_ms": int|None }
# news_feed item = {
#   "news_id", "time": iso (published_at), "source", "title", "url"|None, "symbols": [str],
#   "macro_tags": [str], "sentiment_score": float (-1..1), "sentiment_label": "BULLISH"|"BEARISH"|"NEUTRAL",
#   "importance": int, "duplicate_sources": [str] }
# INITIAL_SNAPSHOT.data = {
#   "equity_kpis", "active_symbol", "symbols": [str], "timeframe": str, "candles": [candle x<=120],
#   "markers": [marker], "open_positions": [...], "trade_history": [x<=50], "claude_logs": [x<=25],
#   "news_feed": [x<=25], "market_health": {...}, "server_time": iso }


def num(x: Decimal | float | int | None) -> float | None:
    """Decimal -> float for the UI wire format (None-safe)."""
    if x is None:
        return None
    return float(x)


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


class TerminalBackend(Protocol):
    """Everything the WebSocket hub and REST routes need. Implemented by app.api.backend.RuntimeTerminalBackend
    over the live TradingRuntime; tests use a fake. All methods must be safe to call concurrently and must not
    raise for missing data (return empty lists / None instead)."""

    @property
    def active_symbol(self) -> str: ...

    @property
    def symbols(self) -> list[str]: ...

    @property
    def timeframe(self) -> str: ...

    def system_status(self) -> SystemStatus: ...

    async def equity_kpis(self) -> dict[str, Any]: ...

    async def candles(self, symbol: str, limit: int = 120) -> list[dict[str, Any]]: ...

    async def markers(self, symbol: str, since_unix: int) -> list[dict[str, Any]]: ...

    async def open_positions(self) -> list[dict[str, Any]]: ...

    async def trade_history(self, limit: int = 50) -> list[dict[str, Any]]: ...

    async def claude_logs(self, limit: int = 25) -> list[dict[str, Any]]: ...

    async def news_feed(self, limit: int = 25) -> list[dict[str, Any]]: ...

    def market_health(self) -> dict[str, Any]: ...

    async def market_tick(self, symbol: str) -> dict[str, Any] | None: ...

    async def current_candle(self, symbol: str) -> dict[str, Any] | None: ...

    # control (REST)
    async def engage_kill_switch(self, reason: str) -> dict[str, Any]: ...

    async def resume(self, operator: str, confirm: str) -> dict[str, Any]: ...

    async def close_position(self, symbol: str) -> dict[str, Any]: ...

    async def health(self) -> dict[str, Any]: ...
