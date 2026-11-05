"""RuntimeTerminalBackend: implements ``app.api.protocol.TerminalBackend`` over the live ``TradingRuntime``.

Read methods never raise: failures are logged and an empty list / None / last good value is returned.
The 5 Hz terminal must stay cheap, so account and DB-derived KPI parts are cached for a short time and
funding rates for 30 s. Control methods (kill switch, resume, manual close) invalidate those caches.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text

from app.api.protocol import SystemStatus, iso, num
from app.core.errors import RiskViolation
from app.core.types import AccountSnapshot, Position, utcnow
from app.db.models import ClaudeDecision as ClaudeDecisionRow
from app.db.models import Trade, TradeStatus
from app.execution.contracts import ExecutionStatus, ManagedPosition

if TYPE_CHECKING:  # pragma: no cover
    from app.runtime import TradingRuntime

logger = logging.getLogger(__name__)

ZERO = Decimal(0)
GREEN = "#00C805"
RED = "#FF3B30"
_ACCOUNT_TTL = 0.2  # seconds; matches the 5 Hz UI cadence
_KPI_DB_TTL = 1.0
_FUNDING_TTL = 30.0
_FUNDING_FETCH_TIMEOUT = 2.0
_HEALTH_PROBE_TIMEOUT = 3.0

_EXIT_TEXT = {
    "TAKE_PROFIT": "Kâr al (TP) tetiklendi",
    "STOP_LOSS": "Zarar durdur (SL) tetiklendi",
    "TRAILING_STOP": "İz süren stop tetiklendi",
    "MANUAL": "Manuel kapatma",
    "KILL_SWITCH": "Kill switch ile kapatıldı",
    "SIGNAL_EXIT": "Claude CLOSE kararıyla kapatıldı",
    "LIQUIDATION": "Likidasyon",
}


def _money_text(value: Decimal | float | None) -> str:
    v = float(value or 0)
    return f"{'+' if v >= 0 else '-'}${abs(v):.2f}"


def _first_sentence(text_: str | None, limit: int = 160) -> str | None:
    if not text_:
        return None
    cleaned = " ".join(text_.split())
    for i, ch in enumerate(cleaned):
        if ch in ".!?" and (i + 1 == len(cleaned) or cleaned[i + 1] == " "):
            cleaned = cleaned[: i + 1]
            break
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _as_float(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def metrics_summary_from_context(ctx: Any) -> str:
    """Rebuild the one-line indicator summary from a stored ``input_context`` (same format as the live stream)."""
    if not isinstance(ctx, dict):
        return ""
    payload = ctx.get("payload")
    if not isinstance(payload, dict):
        return ""
    ind = payload.get("indicators") if isinstance(payload.get("indicators"), dict) else {}
    parts: list[str] = []
    rsi, obi, atr_pct = _as_float(ind.get("rsi")), _as_float(ind.get("obi")), _as_float(ind.get("atr_pct"))
    if rsi is not None:
        parts.append(f"RSI(14): {rsi:.1f}")
    if obi is not None:
        parts.append(f"OBI: {obi:+.2f}")
    if atr_pct is not None:
        parts.append(f"ATR%: {atr_pct * 100:.2f}")
    fee = payload.get("fee_precheck") if isinstance(payload.get("fee_precheck"), dict) else {}
    cover = _as_float(fee.get("cover_multiple"))
    if cover is not None:
        parts.append(f"Fee x{cover:.1f}")
    score = _as_float(payload.get("confluence_score"))
    if score is not None:
        parts.append(f"Skor: {score:.2f}")
    return " | ".join(parts)


class RuntimeTerminalBackend:
    def __init__(
        self,
        runtime: "TradingRuntime",
        *,
        clock: Callable[[], datetime] = utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rt = runtime
        self._clock = clock
        self._mono = monotonic
        self._account_cache: tuple[float, AccountSnapshot] | None = None
        self._kpi_db_cache: tuple[float, dict[str, Any]] | None = None
        self._funding_cache: dict[str, tuple[float, float | None]] = {}
        self._last_kpis: dict[str, Any] = {}

    # ================================================================== static info
    @property
    def symbols(self) -> list[str]:
        return list(self._rt.market.symbols)

    @property
    def active_symbol(self) -> str:
        symbols = self.symbols
        configured = self._rt.settings.api.active_symbol
        if configured and configured in symbols:
            return configured
        return symbols[0] if symbols else (configured or "")

    @property
    def timeframe(self) -> str:
        return self._rt.settings.data.timeframe

    def invalidate_caches(self) -> None:
        self._account_cache = None
        self._kpi_db_cache = None

    # ================================================================== status
    def system_status(self) -> SystemStatus:
        try:
            if self._rt.kill_switch.is_engaged:
                return SystemStatus.HALTED
            symbol = self.active_symbol
            if not self._rt.market.gate.is_active(symbol):
                return SystemStatus.DATA_PAUSED
            return SystemStatus.AUTONOMOUS
        except Exception:  # noqa: BLE001
            logger.exception("system_status failed; reporting DATA_PAUSED")
            return SystemStatus.DATA_PAUSED

    # ================================================================== account / KPIs
    async def _account(self) -> AccountSnapshot | None:
        now = self._mono()
        cached = self._account_cache
        if cached is not None and now - cached[0] <= _ACCOUNT_TTL:
            return cached[1]
        try:
            snap = await self._rt.broker.fetch_account()
        except Exception:  # noqa: BLE001
            logger.warning("fetch_account failed", exc_info=True)
            return cached[1] if cached is not None else None
        self._account_cache = (now, snap)
        return snap

    async def _db_kpis(self) -> dict[str, Any]:
        now = self._mono()
        cached = self._kpi_db_cache
        if cached is not None and now - cached[0] <= _KPI_DB_TTL:
            return cached[1]
        empty: dict[str, Any] = {"realized_pnl_24h": 0.0, "trades_24h": 0, "fees_paid_24h": 0.0,
                                 "win_rate": None, "profit_factor": None}
        since = self._clock() - timedelta(hours=24)
        try:
            async with self._rt.session_factory() as session:
                rows = (await session.execute(
                    select(Trade.net_pnl, Trade.fees_paid)
                    .where(Trade.status == TradeStatus.CLOSED, Trade.closed_at >= since)
                )).all()
        except Exception:  # noqa: BLE001
            logger.warning("KPI trade query failed", exc_info=True)
            return cached[1] if cached is not None else empty
        nets = [Decimal(r[0]) for r in rows if r[0] is not None]
        fees = sum((Decimal(r[1]) for r in rows if r[1] is not None), ZERO)
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n < 0]
        gross_win, gross_loss = sum(wins, ZERO), -sum(losses, ZERO)
        result: dict[str, Any] = {
            "realized_pnl_24h": float(sum(nets, ZERO)),
            "trades_24h": len(nets),
            "fees_paid_24h": float(fees),
            "win_rate": (len(wins) / len(nets)) if nets else None,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        }
        self._kpi_db_cache = (now, result)
        return result

    async def _funding_rate(self, symbol: str) -> float | None:
        now = self._mono()
        cached = self._funding_cache.get(symbol)
        if cached is not None and now - cached[0] <= _FUNDING_TTL:
            return cached[1]
        rate: float | None = cached[1] if cached is not None else None
        try:
            info = await asyncio.wait_for(self._rt.broker.fetch_funding(symbol), timeout=_FUNDING_FETCH_TIMEOUT)
            rate = None if info is None else float(info.rate)
        except Exception:  # noqa: BLE001 - funding is decoration; keep the last value
            logger.debug("funding fetch failed for %s", symbol, exc_info=True)
        self._funding_cache[symbol] = (now, rate)
        return rate

    async def equity_kpis(self) -> dict[str, Any]:
        try:
            rt = self._rt
            account = await self._account()
            if account is None:
                return dict(self._last_kpis)
            db = await self._db_kpis()
            state = rt.kill_switch.state
            equity = account.equity
            start = state.day_start_equity if state.day_start_equity and state.day_start_equity > 0 else equity
            change = (equity - start) / start if start > 0 else ZERO
            drawdown = max(ZERO, (start - equity) / start) if start > 0 else ZERO
            symbol = self.active_symbol
            kpis: dict[str, Any] = {
                "equity": float(equity),
                "equity_change_pct": float(change),
                "unrealized_pnl": float(account.unrealized_pnl),
                **db,
                "daily_drawdown_pct": float(drawdown),
                "daily_drawdown_limit_pct": float(rt.settings.risk.max_daily_drawdown_pct),
                "system_status": self.system_status().value,
                "kill_switch_engaged": bool(rt.kill_switch.is_engaged),
                "kill_switch_reason": rt.kill_switch.reason,
                "trading_mode": rt.settings.trading_mode.value,
                "exchange": rt.settings.exchange_id,
                "active_symbol": symbol,
                "funding_rate": await self._funding_rate(symbol) if symbol else None,
                "claude_model": rt.settings.claude.model,
                "min_net_profit_usd": float(rt.settings.risk.min_net_profit_usd),
            }
            self._last_kpis = kpis
            return dict(kpis)
        except Exception:  # noqa: BLE001
            logger.exception("equity_kpis failed")
            return dict(self._last_kpis)

    # ================================================================== chart
    @staticmethod
    def _candle_dict(c: Any) -> dict[str, Any]:
        return {"time": int(c.open_time.timestamp()), "open": float(c.open), "high": float(c.high),
                "low": float(c.low), "close": float(c.close)}

    async def candles(self, symbol: str, limit: int = 120) -> list[dict[str, Any]]:
        try:
            buf = self._rt.market.candles(symbol)
            return [self._candle_dict(c) for c in buf.last(max(0, limit))]
        except Exception:  # noqa: BLE001
            logger.warning("candles(%s) failed", symbol, exc_info=True)
            return []

    async def current_candle(self, symbol: str) -> dict[str, Any] | None:
        try:
            current = self._rt.market.candles(symbol).current
            return None if current is None else self._candle_dict(current)
        except Exception:  # noqa: BLE001
            logger.warning("current_candle(%s) failed", symbol, exc_info=True)
            return None

    async def markers(self, symbol: str, since_unix: int) -> list[dict[str, Any]]:
        try:
            since = datetime.fromtimestamp(since_unix, tz=timezone.utc)
            async with self._rt.session_factory() as session:
                trades = (await session.execute(
                    select(Trade)
                    .where(Trade.symbol == symbol)
                    .where((Trade.closed_at.is_(None)) | (Trade.closed_at >= since))
                    .order_by(Trade.opened_at.asc())
                    .limit(500)
                )).scalars().all()
            markers: list[dict[str, Any]] = []
            for t in trades:
                is_long = t.side.value == "LONG"
                opened = int(t.opened_at.timestamp())
                if opened >= since_unix:
                    markers.append({
                        "time": opened,
                        "position": "belowBar" if is_long else "aboveBar",
                        "color": GREEN if is_long else RED,
                        "shape": "arrowUp" if is_long else "arrowDown",
                        "text": f"{t.side.value} @ {float(t.entry_price):g}",
                    })
                if t.status is TradeStatus.CLOSED and t.closed_at is not None:
                    closed = int(t.closed_at.timestamp())
                    if closed >= since_unix:
                        markers.append(self._exit_marker(t, closed, is_long))
            markers.sort(key=lambda m: m["time"])
            return markers
        except Exception:  # noqa: BLE001
            logger.warning("markers(%s) failed", symbol, exc_info=True)
            return []

    @staticmethod
    def _exit_marker(t: Trade, closed_unix: int, is_long: bool) -> dict[str, Any]:
        net = t.net_pnl if t.net_pnl is not None else ZERO
        reason = t.exit_reason.value if t.exit_reason is not None else "MANUAL"
        label = {"TAKE_PROFIT": "TP", "STOP_LOSS": "SL", "TRAILING_STOP": "TRAIL", "KILL_SWITCH": "KILL",
                 "LIQUIDATION": "LIQ", "SIGNAL_EXIT": "EXIT", "MANUAL": "EXIT"}.get(reason, "EXIT")
        if reason == "TAKE_PROFIT":
            color = GREEN
        elif reason == "STOP_LOSS":
            color = RED
        else:
            color = GREEN if net >= 0 else RED
        return {
            "time": closed_unix,
            "position": "aboveBar" if is_long else "belowBar",
            "color": color,
            "shape": "arrowDown" if is_long else "arrowUp",
            "text": f"{label} {_money_text(net)}",
        }

    # ================================================================== positions / trades
    async def _taker_rate(self, symbol: str, managed: ManagedPosition | None) -> Decimal:
        if managed is not None:
            return managed.fees.taker_rate
        try:
            return (await self._rt.broker.get_instrument(symbol)).fees.taker_rate
        except Exception:  # noqa: BLE001
            return ZERO

    def _position_item(self, p: Position, managed: ManagedPosition | None, taker: Decimal) -> dict[str, Any]:
        est_fee = taker * (p.entry_price + p.mark_price) * p.quantity
        opened = managed.opened_at if managed is not None else p.opened_at
        return {
            "symbol": p.symbol,
            "side": p.side.value,
            "leverage": int(p.leverage),
            "quantity": float(p.quantity),
            "entry_price": float(p.entry_price),
            "mark_price": float(p.mark_price),
            "liquidation_price": num(p.liquidation_price),
            "hard_stop": num(managed.current_stop) if managed is not None else None,
            "take_profit": num(managed.take_profit) if managed is not None else None,
            "trailing_active": bool(managed.breakeven_locked) if managed is not None else False,
            "est_fee": float(est_fee),
            "unrealized_pnl": float(p.unrealized_pnl),
            "net_pnl": float(p.unrealized_pnl - est_fee),
            "opened_at": iso(opened),
            "decision_id": managed.decision_id if managed is not None else None,
        }

    async def open_positions(self) -> list[dict[str, Any]]:
        try:
            positions = await self._rt.broker.fetch_positions()
            managed = {m.symbol: m for m in self._rt.position_manager.positions()}
            items = []
            for p in positions:
                m = managed.get(p.symbol)
                if m is not None and m.side is not p.side:
                    m = None
                items.append(self._position_item(p, m, await self._taker_rate(p.symbol, m)))
            return items
        except Exception:  # noqa: BLE001
            logger.warning("open_positions failed", exc_info=True)
            return []

    def _trade_item(self, t: Trade, reasoning: str | None) -> dict[str, Any]:
        reason = t.exit_reason.value if t.exit_reason is not None else "MANUAL"
        trigger = _EXIT_TEXT.get(reason, reason)
        sentence = _first_sentence(reasoning)
        if sentence:
            trigger = f"{trigger} — {sentence}"
        return {
            "trade_id": str(t.id),
            "symbol": t.symbol,
            "side": t.side.value,
            "opened_at": iso(t.opened_at),
            "closed_at": iso(t.closed_at),
            "entry_price": float(t.entry_price),
            "exit_price": num(t.exit_price),
            "quantity": float(t.quantity),
            "leverage": int(t.leverage),
            "gross_pnl": num(t.gross_pnl),
            "fees_paid": num(t.fees_paid),
            "funding_paid": num(t.funding_paid),
            "net_pnl": num(t.net_pnl),
            "r_multiple": num(t.r_multiple),
            "exit_reason": reason,
            "trigger": trigger,
        }

    async def trade_history(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            async with self._rt.session_factory() as session:
                rows = (await session.execute(
                    select(Trade, ClaudeDecisionRow.reasoning)
                    .outerjoin(ClaudeDecisionRow, Trade.decision_id == ClaudeDecisionRow.id)
                    .where(Trade.status == TradeStatus.CLOSED)
                    .order_by(Trade.closed_at.desc())
                    .limit(max(1, limit))
                )).all()
            return [self._trade_item(t, reasoning) for t, reasoning in rows]
        except Exception:  # noqa: BLE001
            logger.warning("trade_history failed", exc_info=True)
            return []

    # ================================================================== Claude / news
    async def claude_logs(self, limit: int = 25) -> list[dict[str, Any]]:
        try:
            rows = await self._rt.decision_log.recent(max(1, limit))
            out: list[dict[str, Any]] = []
            for r in rows:
                flags: list[str] = []
                if r.get("fallback_reason"):
                    flags.append(f"FALLBACK_{r['fallback_reason']}")
                reasons = r.get("risk_reasons")
                if isinstance(reasons, list):
                    flags.extend(str(x) for x in reasons)
                leverage = r.get("final_leverage") or r.get("leverage")
                out.append({
                    "decision_id": r["id"],
                    "timestamp": r["created_at"],
                    "symbol": r["symbol"],
                    "action": r["action"],
                    "confidence": _as_float(r.get("confidence")) or 0.0,
                    "leverage": int(leverage) if leverage is not None else None,
                    "metrics_summary": metrics_summary_from_context(r.get("input_context")),
                    "reasoning_tr": r.get("reasoning") or "",
                    "risk_flags": flags,
                    "risk_decision": r.get("risk_verdict"),
                    "execution_status": r.get("execution_status"),
                    "fallback": r.get("fallback_reason"),
                    "latency_ms": r.get("latency_ms"),
                })
            return out
        except Exception:  # noqa: BLE001
            logger.warning("claude_logs failed", exc_info=True)
            return []

    async def news_feed(self, limit: int = 25) -> list[dict[str, Any]]:
        news = self._rt.news
        if news is None:
            return []
        try:
            since = self._clock() - timedelta(hours=float(news.settings.cache_ttl_hours))
            events = news.cache.recent(None, since, max(0, limit))
            return [{
                "news_id": e.news_id,
                "time": iso(e.published_at),
                "source": e.source,
                "title": e.title,
                "url": e.url,
                "symbols": list(e.symbols),
                "macro_tags": list(e.macro_tags),
                "sentiment_score": float(e.sentiment_score),
                "sentiment_label": e.sentiment_label,
                "importance": int(e.importance),
                "duplicate_sources": list(e.duplicate_sources),
            } for e in events]
        except Exception:  # noqa: BLE001
            logger.warning("news_feed failed", exc_info=True)
            return []

    # ================================================================== market data
    def market_health(self) -> dict[str, Any]:
        try:
            snap = self._rt.market.health_snapshot()
            gate = snap.get("gate", {}) if isinstance(snap.get("gate"), dict) else {}
            channels: dict[str, Any] = {}
            for symbol, info in (snap.get("symbols") or {}).items():
                channels[symbol] = {
                    name: {"health": ch.get("health"), "age_seconds": ch.get("age_seconds")}
                    for name, ch in (info.get("channels") or {}).items()
                }
            return {
                "system_status": self.system_status().value,
                "gate": gate.get("state"),
                "paused_symbols": list(gate.get("paused_symbols") or []),
                "reasons": list(gate.get("reasons") or []),
                "channels": channels,
            }
        except Exception:  # noqa: BLE001
            logger.warning("market_health failed", exc_info=True)
            return {}

    async def market_tick(self, symbol: str) -> dict[str, Any] | None:
        try:
            market = self._rt.market
            ticker = market.last_ticker(symbol)
            book = market.order_book(symbol)
            bid = book.best_bid if book is not None else None
            ask = book.best_ask if book is not None else None
            if bid is None and ticker is not None:
                bid = ticker.bid
            if ask is None and ticker is not None:
                ask = ticker.ask
            if bid is None or ask is None:
                return None
            mid = (bid + ask) / 2
            last = ticker.last if ticker is not None else mid
            mark = ticker.mark if ticker is not None and ticker.mark is not None else mid
            return {
                "symbol": symbol,
                "bid": float(bid),
                "ask": float(ask),
                "last": float(last),
                "mark": float(mark),
                "spread": float(ask - bid),
                "funding_rate": await self._funding_rate(symbol),
            }
        except Exception:  # noqa: BLE001
            logger.warning("market_tick(%s) failed", symbol, exc_info=True)
            return None

    # ================================================================== control
    async def engage_kill_switch(self, reason: str) -> dict[str, Any]:
        rt = self._rt
        reason = (reason or "MANUAL_UI_TRIGGER").strip()[:200] or "MANUAL_UI_TRIGGER"
        before: list[str] = []
        try:
            before = [p.symbol for p in await rt.broker.fetch_positions()]
        except Exception:  # noqa: BLE001
            logger.exception("kill switch: could not list positions before engaging")
        newly = False
        try:
            newly = await rt.kill_switch.engage(reason)  # persists first, then cancels + flattens via the broker
        except Exception:  # noqa: BLE001
            logger.critical("kill switch engage raised; continuing with manual flatten", exc_info=True)
        remaining: list[str] = []
        try:
            remaining = [p.symbol for p in await rt.broker.fetch_positions()]
        except Exception:  # noqa: BLE001
            logger.exception("kill switch: could not list positions after engage")
            remaining = list(before)
        for symbol in list(remaining):
            try:
                report = await rt.executor.close_symbol(symbol, "KILL_SWITCH")
                if report.status is ExecutionStatus.ERROR:
                    logger.critical("kill switch: close of %s failed: %s", symbol, report.detail)
            except Exception:  # noqa: BLE001
                logger.critical("kill switch: close_symbol(%s) raised", symbol, exc_info=True)
        try:
            await rt.position_manager.reconcile()
        except Exception:  # noqa: BLE001
            logger.exception("kill switch: reconcile failed")
        still_open: list[str] = []
        try:
            still_open = [p.symbol for p in await rt.broker.fetch_positions()]
        except Exception:  # noqa: BLE001
            still_open = list(remaining)
        self.invalidate_caches()
        return {
            "engaged": bool(rt.kill_switch.is_engaged),
            "newly_engaged": bool(newly),
            "reason": rt.kill_switch.reason,
            "system_status": self.system_status().value,
            "closed": [s for s in dict.fromkeys([*before, *remaining]) if s not in still_open],
            "still_open": still_open,
        }

    async def resume(self, operator: str, confirm: str) -> dict[str, Any]:
        rt = self._rt
        try:
            await rt.kill_switch.manual_reset(operator, confirm)
        except RiskViolation as exc:
            return {"ok": False, "error": exc.code, "detail": exc.message, "engaged": bool(rt.kill_switch.is_engaged),
                    "system_status": self.system_status().value}
        except Exception as exc:  # noqa: BLE001
            logger.exception("kill switch reset failed")
            return {"ok": False, "error": "RESET_FAILED", "detail": f"{type(exc).__name__}: {exc}",
                    "engaged": bool(rt.kill_switch.is_engaged), "system_status": self.system_status().value}
        self.invalidate_caches()
        return {"ok": True, "engaged": bool(rt.kill_switch.is_engaged), "operator": operator,
                "system_status": self.system_status().value}

    async def close_position(self, symbol: str) -> dict[str, Any]:
        rt = self._rt
        try:
            report = await rt.executor.close_symbol(symbol, "MANUAL_UI_CLOSE")
            await rt.position_manager.reconcile()
        except Exception as exc:  # noqa: BLE001
            logger.exception("manual close of %s failed", symbol)
            return {"ok": False, "status": ExecutionStatus.ERROR.value, "detail": f"{type(exc).__name__}: {exc}"}
        self.invalidate_caches()
        ok = report.status in (ExecutionStatus.CLOSED_BY_DECISION, ExecutionStatus.NOTHING_TO_CLOSE)
        return {"ok": ok, "status": report.status.value, "detail": report.detail}

    # ================================================================== health
    async def _timed(self, coro: Any) -> tuple[bool, float, str | None]:
        started = time.perf_counter()
        try:
            await asyncio.wait_for(coro, timeout=_HEALTH_PROBE_TIMEOUT)
            return True, round((time.perf_counter() - started) * 1000, 2), None
        except Exception as exc:  # noqa: BLE001
            return False, round((time.perf_counter() - started) * 1000, 2), f"{type(exc).__name__}: {exc}"

    async def _db_probe(self) -> None:
        async with self._rt.session_factory() as session:
            await session.execute(text("SELECT 1"))

    async def health(self) -> dict[str, Any]:
        rt = self._rt
        components: dict[str, Any] = {}
        db_ok, db_ms, db_err = await self._timed(self._db_probe())
        components["db"] = {"ok": db_ok, "latency_ms": db_ms, **({"error": db_err} if db_err else {})}
        br_ok, br_ms, br_err = await self._timed(rt.broker.fetch_account())
        components["broker"] = {"ok": br_ok, "latency_ms": br_ms, "type": type(rt.broker).__name__,
                                **({"error": br_err} if br_err else {})}
        try:
            snap = rt.market.health_snapshot()
            ages = [ch.get("age_seconds") for info in (snap.get("symbols") or {}).values()
                    for ch in (info.get("channels") or {}).values() if ch.get("age_seconds") is not None]
            gate_state = (snap.get("gate") or {}).get("state")
            components["market_data"] = {
                "ok": gate_state == "ACTIVE", "gate": gate_state,
                "max_channel_age_s": max(ages) if ages else None,
                "paused_symbols": list((snap.get("gate") or {}).get("paused_symbols") or []),
            }
        except Exception as exc:  # noqa: BLE001
            components["market_data"] = {"ok": False, "gate": None, "max_channel_age_s": None,
                                         "error": f"{type(exc).__name__}: {exc}"}
        core = rt.claude_core
        components["claude"] = {
            "ok": bool(rt.api_key_present) and not core.circuit_open,
            "api_key_present": bool(rt.api_key_present),
            "circuit_open": bool(core.circuit_open),
            "stats": core.stats,
        }
        components["kill_switch"] = {"ok": not rt.kill_switch.is_engaged, "engaged": bool(rt.kill_switch.is_engaged),
                                     "reason": rt.kill_switch.reason}
        components["news"] = ({"ok": True, "enabled": True, "stats": rt.news.stats.as_dict()}
                              if rt.news is not None else {"ok": True, "enabled": False, "stats": None})
        uptime = (self._clock() - rt.started_at).total_seconds()
        return {
            "ok": all(components[k]["ok"] for k in ("db", "broker", "market_data", "kill_switch")),
            "components": components,
            "uptime_s": round(max(0.0, uptime), 3),
        }
