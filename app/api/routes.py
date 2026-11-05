"""REST control + debug API under /api/v1.

Security model
--------------
* ``POST /kill-switch`` needs NO token: the emergency stop must work even when credentials are lost or the
  operator is panicking. It only moves the system toward safety. It is rate limited to 1 call / second.
* ``POST /kill-switch/resume`` and ``POST /positions/{symbol}/close`` need ``X-Control-Token`` equal to
  ``settings.api.control_token`` whenever that token is configured (compared in constant time). LIVE mode
  refuses to start without a token (see ``Settings``), and ``app.api.server`` refuses non-loopback binds
  without one.
* ``GET /health`` and ``GET /snapshot`` are read-only.
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.protocol import MessageType, TerminalBackend

logger = logging.getLogger(__name__)

KILL_SWITCH_MIN_INTERVAL_SECONDS = 1.0
DEFAULT_KILL_REASON = "MANUAL_UI_TRIGGER"
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,64}$")

router = APIRouter(prefix="/api/v1")


class KillSwitchBody(BaseModel):
    reason: str | None = Field(default=None, max_length=200)


class ResumeBody(BaseModel):
    operator: str = Field(min_length=1, max_length=64)
    confirm: str = Field(max_length=32)


# ---------------------------------------------------------------------- dependencies
def get_backend(request: Request) -> TerminalBackend:
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(status_code=503, detail="service starting")
    return backend


def require_control_token(request: Request, x_control_token: str | None = Header(default=None)) -> None:
    """401 unless the correct X-Control-Token is supplied (when a token is configured)."""
    settings = request.app.state.settings
    expected = settings.api.control_token.get_secret_value()
    if not expected:
        return
    supplied = x_control_token or ""
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="invalid or missing X-Control-Token",
                            headers={"WWW-Authenticate": "X-Control-Token"})


# ---------------------------------------------------------------------- broadcast helpers
async def _broadcast(request: Request, msg_type: MessageType, data: dict[str, Any]) -> None:
    hub = getattr(request.app.state, "hub", None)
    if hub is None:
        return
    try:
        result = hub.broadcast(msg_type, data)
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - UI fan-out must never fail a control action
        logger.warning("broadcast %s failed", msg_type, exc_info=True)


async def _broadcast_positions(request: Request, backend: TerminalBackend) -> None:
    positions = await backend.open_positions()
    total = sum(float(p.get("unrealized_pnl") or 0.0) for p in positions)
    await _broadcast(request, MessageType.POSITIONS_PNL, {"positions": positions, "unrealized_total": total})


async def _broadcast_kpis(request: Request, backend: TerminalBackend) -> None:
    kpis = await backend.equity_kpis()
    if kpis:
        await _broadcast(request, MessageType.EQUITY_KPIS, kpis)


async def _broadcast_kill_state(request: Request, backend: TerminalBackend, result: dict[str, Any]) -> None:
    await _broadcast(request, MessageType.KILL_SWITCH, {
        "engaged": bool(result.get("engaged")),
        "reason": result.get("reason"),
        "system_status": result.get("system_status") or backend.system_status().value,
    })


# ---------------------------------------------------------------------- routes
@router.post("/kill-switch")
async def kill_switch(request: Request, body: KillSwitchBody | None = None,
                      backend: TerminalBackend = Depends(get_backend)) -> JSONResponse:
    state = request.app.state
    lock: asyncio.Lock = state.kill_switch_lock
    async with lock:
        now = time.monotonic()
        last = state.kill_switch_last_call
        if last is not None and now - last < KILL_SWITCH_MIN_INTERVAL_SECONDS:
            retry = max(0.0, KILL_SWITCH_MIN_INTERVAL_SECONDS - (now - last))
            return JSONResponse(
                status_code=429,
                content={"ok": False, "detail": "kill switch rate limited (1 call/second)", "retry_after": round(retry, 3),
                         "system_status": backend.system_status().value},
                headers={"Retry-After": str(max(1, int(retry + 0.999)))},
            )
        state.kill_switch_last_call = now
        reason = (body.reason.strip() if body and body.reason else "") or DEFAULT_KILL_REASON
        result = await backend.engage_kill_switch(reason)
    await _broadcast_kill_state(request, backend, result)
    await _broadcast_positions(request, backend)
    await _broadcast_kpis(request, backend)
    return JSONResponse(content={"ok": bool(result.get("engaged")), **result})


@router.post("/kill-switch/resume", dependencies=[Depends(require_control_token)])
async def resume(request: Request, body: ResumeBody, backend: TerminalBackend = Depends(get_backend)) -> JSONResponse:
    result = await backend.resume(body.operator, body.confirm)
    if not result.get("ok"):
        code = 409 if result.get("error") == "RESET_DENIED" else 500
        return JSONResponse(status_code=code, content=result)
    await _broadcast_kill_state(request, backend, {**result, "reason": None})
    await _broadcast_kpis(request, backend)
    return JSONResponse(content=result)


@router.post("/positions/{symbol:path}/close", dependencies=[Depends(require_control_token)])
async def close_position(symbol: str, request: Request,
                         backend: TerminalBackend = Depends(get_backend)) -> JSONResponse:
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=422, detail="invalid symbol")
    result = await backend.close_position(symbol)
    await _broadcast_positions(request, backend)
    await _broadcast_kpis(request, backend)
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)
    return JSONResponse(content=result)


@router.get("/health")
async def health(backend: TerminalBackend = Depends(get_backend)) -> JSONResponse:
    return JSONResponse(content=await backend.health())


@router.get("/snapshot")
async def snapshot(backend: TerminalBackend = Depends(get_backend)) -> JSONResponse:
    """Same content as the INITIAL_SNAPSHOT WebSocket message (debugging aid)."""
    symbol = backend.active_symbol
    candles_task = backend.candles(symbol, 120)
    kpis, candles, positions, trades, logs, news = await asyncio.gather(
        backend.equity_kpis(), candles_task, backend.open_positions(), backend.trade_history(50),
        backend.claude_logs(25), backend.news_feed(25),
    )
    since = int(candles[0]["time"]) if candles else int(time.time()) - 6 * 3600
    markers = await backend.markers(symbol, since)
    return JSONResponse(content={
        "equity_kpis": kpis,
        "active_symbol": symbol,
        "symbols": backend.symbols,
        "timeframe": backend.timeframe,
        "candles": candles,
        "markers": markers,
        "open_positions": positions,
        "trade_history": trades,
        "claude_logs": logs,
        "news_feed": news,
        "market_health": backend.market_health(),
        "server_time": datetime.now(timezone.utc).isoformat(),
    })

