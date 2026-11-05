"""In-memory TerminalBackend for tests (WebSocket hub, REST routes, UI work without a running trading runtime).

All data is plain mutable attributes in exactly the shapes documented in app.api.protocol. Every call is counted
(``calls``) and its arguments recorded (``call_args``). ``delays`` makes a method sleep (async methods only),
``errors`` makes a method raise, ``on_call`` runs a hook at the start of a call (e.g. to publish a bus event
while the hub is building a snapshot). Returned data is deep-copied so callers cannot corrupt the canned state.
"""

from __future__ import annotations

import asyncio
import copy
from collections import Counter
from collections.abc import Callable
from typing import Any

from app.api.protocol import SystemStatus

BASE_TIME = 1_700_000_000  # unix seconds of candle #0
CANDLE_STEP = 60

DEFAULT_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]


def make_candles(symbol: str, count: int = 200, base_price: float = 60_000.0) -> list[dict[str, Any]]:
    out = []
    price = base_price
    for i in range(count):
        o = price
        c = price + (1 if i % 2 == 0 else -0.5)
        out.append({"time": BASE_TIME + i * CANDLE_STEP, "open": o, "high": max(o, c) + 0.5,
                    "low": min(o, c) - 0.5, "close": c})
        price = c
    return out


def make_position(symbol: str = "BTC/USDT:USDT", side: str = "LONG", unrealized: float = 12.5) -> dict[str, Any]:
    return {
        "symbol": symbol, "side": side, "leverage": 3, "quantity": 0.01, "entry_price": 60_000.0,
        "mark_price": 61_250.0, "liquidation_price": 40_500.0, "hard_stop": 59_400.0, "take_profit": 61_800.0,
        "trailing_active": False, "est_fee": 0.6, "unrealized_pnl": unrealized, "net_pnl": unrealized - 0.6,
        "opened_at": "2026-01-01T00:00:00+00:00", "decision_id": "dec-1",
    }


def make_trade(i: int, symbol: str = "BTC/USDT:USDT") -> dict[str, Any]:
    return {
        "trade_id": f"trade-{i}", "symbol": symbol, "side": "LONG", "opened_at": "2026-01-01T00:00:00+00:00",
        "closed_at": "2026-01-01T01:00:00+00:00", "entry_price": 60_000.0, "exit_price": 60_300.0,
        "quantity": 0.01, "leverage": 3, "gross_pnl": 3.0, "fees_paid": 0.5, "funding_paid": 0.0,
        "net_pnl": 2.5, "r_multiple": 0.8, "exit_reason": "TAKE_PROFIT", "trigger": "RSI_OVERSOLD",
    }


def make_log(i: int, symbol: str = "BTC/USDT:USDT") -> dict[str, Any]:
    return {
        "decision_id": f"dec-{i}", "timestamp": "2026-01-01T00:00:00+00:00", "symbol": symbol, "action": "OPEN_LONG",
        "confidence": 0.82, "leverage": 3, "metrics_summary": "RSI(14): 28.4 | OBI: +0.31 | ATR%: 0.52 | Fee x4.2",
        "reasoning_tr": "RSI aşırı satımda, emir defteri alıcı yönlü; işlem açılıyor.", "risk_flags": [],
        "risk_decision": "APPROVED", "execution_status": "OPENED_PROTECTED", "fallback": None, "latency_ms": 840,
    }


def make_news(i: int) -> dict[str, Any]:
    return {
        "news_id": f"news-{i}", "time": "2026-01-01T00:00:00+00:00", "source": "rss:coindesk",
        "title": f"Haber {i}", "url": f"https://example.test/{i}", "symbols": ["BTC"], "macro_tags": ["ETF"],
        "sentiment_score": 0.4, "sentiment_label": "BULLISH", "importance": 5, "duplicate_sources": [],
    }


class FakeTerminalBackend:
    """Implements app.api.protocol.TerminalBackend with canned, mutable data."""

    def __init__(self, symbols: list[str] | None = None, *, active_symbol: str | None = None,
                 timeframe: str = "1m") -> None:
        self._symbols = list(symbols or DEFAULT_SYMBOLS)
        self._active_symbol = active_symbol or self._symbols[0]
        self._timeframe = timeframe
        self.status = SystemStatus.AUTONOMOUS
        self.kill_reason: str | None = None

        self.kpis: dict[str, Any] = {
            "equity": 10_000.0, "equity_change_pct": 0.0012, "unrealized_pnl": 12.5, "realized_pnl_24h": 40.0,
            "trades_24h": 3, "fees_paid_24h": 1.8, "win_rate": 0.66, "profit_factor": 1.9,
            "daily_drawdown_pct": 0.0042, "daily_drawdown_limit_pct": 0.03,
            "system_status": SystemStatus.AUTONOMOUS.value, "kill_switch_engaged": False,
            "kill_switch_reason": None, "trading_mode": "PAPER", "exchange": "binance",
            "active_symbol": self._active_symbol, "funding_rate": 0.0001, "claude_model": "claude-test",
            "min_net_profit_usd": 1.0,
        }
        self.candle_data: dict[str, list[dict[str, Any]]] = {
            s: make_candles(s, base_price=60_000.0 / (i + 1)) for i, s in enumerate(self._symbols)}
        self.marker_data: dict[str, list[dict[str, Any]]] = {
            s: [{"time": BASE_TIME + 150 * CANDLE_STEP, "position": "belowBar", "color": "#00C805",
                 "shape": "arrowUp", "text": "LONG"}] for s in self._symbols}
        self.positions: list[dict[str, Any]] = [make_position(self._symbols[0])]
        self.trades: list[dict[str, Any]] = [make_trade(i) for i in range(60)]  # newest first
        self.logs: list[dict[str, Any]] = [make_log(i) for i in range(30)]  # newest first
        self.news: list[dict[str, Any]] = [make_news(i) for i in range(30)]  # newest first
        self.health: dict[str, Any] = {"system_status": SystemStatus.AUTONOMOUS.value, "gate": "ACTIVE",
                                       "paused_symbols": [], "reasons": []}
        self.ticks: dict[str, dict[str, Any] | None] = {
            s: {"symbol": s, "bid": 100.0 + i, "ask": 100.1 + i, "last": 100.05 + i, "mark": 100.05 + i,
                "spread": 0.1, "funding_rate": 0.0001} for i, s in enumerate(self._symbols)}
        self.current: dict[str, dict[str, Any] | None] = {
            s: dict(self.candle_data[s][-1]) for s in self._symbols}

        self.calls: Counter[str] = Counter()
        self.call_args: list[tuple[str, tuple[Any, ...]]] = []
        self.delays: dict[str, float] = {}
        self.errors: dict[str, BaseException] = {}
        self.on_call: dict[str, Callable[[], None]] = {}

    # ------------------------------------------------------------------ plumbing
    def _enter(self, name: str, *args: Any) -> None:
        self.calls[name] += 1
        self.call_args.append((name, args))
        hook = self.on_call.get(name)
        if hook is not None:
            hook()
        err = self.errors.get(name)
        if err is not None:
            raise err

    async def _enter_async(self, name: str, *args: Any) -> None:
        self._enter(name, *args)
        delay = self.delays.get(name, 0.0)
        if delay:
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------ properties
    @property
    def active_symbol(self) -> str:
        return self._active_symbol

    @active_symbol.setter
    def active_symbol(self, value: str) -> None:
        self._active_symbol = value

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def timeframe(self) -> str:
        return self._timeframe

    def system_status(self) -> SystemStatus:
        self._enter("system_status")
        return self.status

    # ------------------------------------------------------------------ data
    async def equity_kpis(self) -> dict[str, Any]:
        await self._enter_async("equity_kpis")
        data = copy.deepcopy(self.kpis)
        data["system_status"] = self.status.value
        data["kill_switch_engaged"] = self.status is SystemStatus.HALTED
        return data

    async def candles(self, symbol: str, limit: int = 120) -> list[dict[str, Any]]:
        await self._enter_async("candles", symbol, limit)
        return copy.deepcopy(self.candle_data.get(symbol, [])[-limit:] if limit > 0 else [])

    async def markers(self, symbol: str, since_unix: int) -> list[dict[str, Any]]:
        await self._enter_async("markers", symbol, since_unix)
        return copy.deepcopy([m for m in self.marker_data.get(symbol, []) if m["time"] >= since_unix])

    async def open_positions(self) -> list[dict[str, Any]]:
        await self._enter_async("open_positions")
        return copy.deepcopy(self.positions)

    async def trade_history(self, limit: int = 50) -> list[dict[str, Any]]:
        await self._enter_async("trade_history", limit)
        return copy.deepcopy(self.trades[:limit])

    async def claude_logs(self, limit: int = 25) -> list[dict[str, Any]]:
        await self._enter_async("claude_logs", limit)
        return copy.deepcopy(self.logs[:limit])

    async def news_feed(self, limit: int = 25) -> list[dict[str, Any]]:
        await self._enter_async("news_feed", limit)
        return copy.deepcopy(self.news[:limit])

    def market_health(self) -> dict[str, Any]:
        self._enter("market_health")
        data = copy.deepcopy(self.health)
        data["system_status"] = self.status.value
        return data

    async def market_tick(self, symbol: str) -> dict[str, Any] | None:
        await self._enter_async("market_tick", symbol)
        return copy.deepcopy(self.ticks.get(symbol))

    async def current_candle(self, symbol: str) -> dict[str, Any] | None:
        await self._enter_async("current_candle", symbol)
        return copy.deepcopy(self.current.get(symbol))

    # ------------------------------------------------------------------ control
    async def engage_kill_switch(self, reason: str) -> dict[str, Any]:
        await self._enter_async("engage_kill_switch", reason)
        self.status = SystemStatus.HALTED
        self.kill_reason = reason
        return {"engaged": True, "reason": reason, "system_status": self.status.value}

    async def resume(self, operator: str, confirm: str) -> dict[str, Any]:
        await self._enter_async("resume", operator, confirm)
        self.status = SystemStatus.AUTONOMOUS
        self.kill_reason = None
        return {"engaged": False, "reason": None, "system_status": self.status.value}

    async def close_position(self, symbol: str) -> dict[str, Any]:
        await self._enter_async("close_position", symbol)
        before = len(self.positions)
        self.positions = [p for p in self.positions if p["symbol"] != symbol]
        return {"symbol": symbol, "closed": len(self.positions) < before}

    async def health(self) -> dict[str, Any]:
        await self._enter_async("health")
        return {"status": "ok", "system_status": self.status.value, "calls": dict(self.calls)}
