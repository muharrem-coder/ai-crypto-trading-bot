"""Standalone mock of the terminal backend for UI tests.

Serves static/index.html at / and a fake /ws/terminal speaking the protocol in app/api/protocol.py.
Deliberately independent of app/api/main.py and app/api/ws_hub.py.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Body, FastAPI, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
INJECTION = "<img src=x onerror=alert(1)>"
CONTROL_TOKEN = "secret-token"
SYMBOL = "BTC/USDT:USDT"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def build_snapshot() -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    t0 = int(now.timestamp()) // 60 * 60 - 119 * 60
    price = 89000.0
    candles = []
    for i in range(120):
        o = price
        c = round(o + ((i * 37) % 11 - 5) * 9.5, 2)
        candles.append({"time": t0 + 60 * i, "open": o, "high": round(max(o, c) + 20, 2),
                        "low": round(min(o, c) - 20, 2), "close": c})
        price = c
    markers = [
        {"time": candles[-28]["time"], "position": "belowBar", "color": "#00C805", "shape": "arrowUp",
         "text": f"CLAUDE LONG @ {candles[-28]['low']}"},
        {"time": candles[-12]["time"], "position": "aboveBar", "color": "#FF3B30", "shape": "arrowDown",
         "text": "TP +$14.20"},
    ]
    positions = [
        {"symbol": SYMBOL, "side": "LONG", "leverage": 3, "quantity": 0.15, "entry_price": 89120.0,
         "mark_price": price, "liquidation_price": 60000.0, "hard_stop": 88650.0, "take_profit": 89900.0,
         "trailing_active": False, "est_fee": 1.34, "unrealized_pnl": 43.7, "net_pnl": 42.36,
         "opened_at": _iso(now - timedelta(minutes=20)), "decision_id": "d-1"},
        {"symbol": "SOL/USDT:USDT", "side": "SHORT", "leverage": 2, "quantity": 25.0, "entry_price": 184.5,
         "mark_price": 183.9, "liquidation_price": None, "hard_stop": 186.8, "take_profit": None,
         "trailing_active": True, "est_fee": 0.46, "unrealized_pnl": 15.0, "net_pnl": -0.46,
         "opened_at": _iso(now - timedelta(minutes=5)), "decision_id": "d-2"},
    ]
    reasons = ["TAKE_PROFIT", "TRAILING_STOP", "STOP_LOSS", "TAKE_PROFIT", "MANUAL"]
    history = []
    for i, r in enumerate(reasons):
        net = [19.05, 7.75, -4.38, 16.46, 1.43][i]
        history.append({
            "trade_id": f"t-{i}", "symbol": ["BTC", "ETH", "BTC", "BNB", "SOL"][i] + "/USDT:USDT",
            "side": "LONG" if i % 2 == 0 else "SHORT", "opened_at": _iso(now - timedelta(minutes=60 * (i + 2))),
            "closed_at": _iso(now - timedelta(minutes=60 * (i + 1))), "entry_price": 89010.0 + i,
            "exit_price": 89145.0 + i, "quantity": 0.1, "leverage": 3, "gross_pnl": net + 1.2,
            "fees_paid": 1.2, "funding_paid": 0.0, "net_pnl": net, "r_multiple": 1.1, "exit_reason": r,
            "trigger": "Orderbook Bid Wall + RSI Bounce",
        })
    logs = [
        {"decision_id": "d-1", "timestamp": _iso(now - timedelta(minutes=1)), "symbol": SYMBOL,
         "action": "HOLD", "confidence": 0.7, "leverage": None,
         "metrics_summary": "RSI(14): 56.2 | OBI: +0.38 | Fee x4.2",
         "reasoning_tr": "Açık LONG pozisyon kârda. İz süren stop güncellendi: şğüöçıİ karakterleri korunmalı. " + INJECTION,
         "risk_flags": [], "risk_decision": None, "execution_status": None, "fallback": None, "latency_ms": 900},
        {"decision_id": "d-2", "timestamp": _iso(now - timedelta(minutes=2)), "symbol": "XRP/USDT:USDT",
         "action": "LONG", "confidence": 0.6, "leverage": 2,
         "metrics_summary": "RSI(14): 28.4 | OBI: +0.31", "reasoning_tr": "Beklenen net kâr komisyon şartını sağlamadı.",
         "risk_flags": [], "risk_decision": "REJECTED", "execution_status": "REJECTED_BY_RISK",
         "fallback": None, "latency_ms": 700},
        {"decision_id": "d-3", "timestamp": _iso(now - timedelta(minutes=3)), "symbol": "SOL/USDT:USDT",
         "action": "SHORT", "confidence": 0.8, "leverage": 2,
         "metrics_summary": "VWAP Δ: +2.10% | Skor: 0.81", "reasoning_tr": "184.50$ direncinde satıcı duvarı.",
         "risk_flags": [], "risk_decision": "APPROVED", "execution_status": "OPENED_PROTECTED",
         "fallback": None, "latency_ms": 650},
    ]
    news = []
    titles = [f"Kurumsal ETF girişleri arttı {INJECTION}", "Binance cüzdanları arasında 1,500 BTC transfer",
              "ABD tahvil getirilerinde geri çekilme", "Borsa API durumu: sağlıklı", "FED tutanakları bekleniyor"]
    scores = [0.84, 0.0, 0.62, 0.1, -0.2]
    labels = ["BULLISH", "NEUTRAL", "BULLISH", "NEUTRAL", "BEARISH"]
    for i, (ti, sc, lb) in enumerate(zip(titles, scores, labels)):
        news.append({"news_id": f"n-{i}", "time": _iso(now - timedelta(minutes=3 + 7 * i)), "source": "RSS",
                     "title": ti, "url": None, "symbols": ["BTC"], "macro_tags": [], "sentiment_score": sc,
                     "sentiment_label": lb, "importance": 2, "duplicate_sources": []})
    kpis = {
        "equity": 10482.65, "equity_change_pct": 0.0482, "unrealized_pnl": 42.34, "realized_pnl_24h": 128.9,
        "trades_24h": 38, "fees_paid_24h": 6.42, "win_rate": 0.714, "profit_factor": 2.41,
        "daily_drawdown_pct": 0.0042, "daily_drawdown_limit_pct": 0.03, "system_status": "AUTONOMOUS 24/7",
        "kill_switch_engaged": False, "kill_switch_reason": None, "trading_mode": "PAPER", "exchange": "mock",
        "active_symbol": SYMBOL, "funding_rate": 0.0001, "claude_model": "claude-sonnet-mock",
        "min_net_profit_usd": 0.05,
    }
    return {"equity_kpis": kpis, "active_symbol": SYMBOL, "symbols": [SYMBOL], "timeframe": "1m",
            "candles": candles, "markers": markers, "open_positions": positions, "trade_history": history,
            "claude_logs": logs, "news_feed": news,
            "market_health": {"system_status": "AUTONOMOUS 24/7", "gate": "OPEN", "paused_symbols": [], "reasons": []},
            "server_time": _iso(now)}


def create_app(live_delay: float = 0.4) -> FastAPI:
    app = FastAPI()
    rest_calls: list[dict] = []
    app.state.rest_calls = rest_calls

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/v1/kill-switch")
    async def kill(payload: dict = Body(...)) -> dict:
        rest_calls.append({"path": "kill-switch", "body": payload})
        return {"engaged": True}

    @app.post("/api/v1/kill-switch/resume")
    async def resume(payload: dict = Body(...), x_control_token: str | None = Header(default=None)):
        rest_calls.append({"path": "resume", "body": payload, "token": x_control_token})
        if x_control_token != CONTROL_TOKEN:
            return JSONResponse({"detail": "invalid token"}, status_code=401)
        return {"engaged": False}

    @app.post("/api/v1/positions/{symbol:path}/close")
    async def close(symbol: str, x_control_token: str | None = Header(default=None)):
        rest_calls.append({"path": "close", "symbol": symbol, "token": x_control_token})
        if x_control_token != CONTROL_TOKEN:
            return JSONResponse({"detail": "invalid token"}, status_code=401)
        return {"closed": symbol}

    @app.websocket("/ws/terminal")
    async def terminal(ws: WebSocket) -> None:
        await ws.accept()
        seq = 0

        async def send(type_: str, data: dict) -> None:
            nonlocal seq
            seq += 1
            await ws.send_text(json.dumps({"type": type_, "seq": seq, "ts": _iso(datetime.now(timezone.utc)),
                                           "data": data}, ensure_ascii=False))

        async def live() -> None:
            await asyncio.sleep(live_delay)
            snap = build_snapshot()
            last = snap["candles"][-1]
            await send("MARKET_TICK", {"symbol": SYMBOL, "bid": 89420.4, "ask": 89420.5, "last": 89420.5,
                                       "mark": 89420.5, "spread": 0.1, "funding_rate": 0.0001})
            await send("CANDLE_UPDATE", {"symbol": SYMBOL, "candle": {**last, "close": 89500.0, "high": 89510.0}})
            await send("EQUITY_KPIS", {**snap["equity_kpis"], "equity": 12345.67})
            decision = {"decision_id": "d-9", "timestamp": _iso(datetime.now(timezone.utc)), "symbol": SYMBOL,
                        "action": "LONG", "confidence": "0.7", "leverage": 3, "reasoning_tr": "Yeni sinyal: güçlü alım.",
                        "metrics_summary": "RSI(14): 31.0", "risk_decision": None, "execution_status": None,
                        "fallback": None}
            await send("CLAUDE_DECISION", decision)
            await send("CLAUDE_DECISION", {**decision, "risk_decision": "APPROVED",
                                           "execution_status": "OPENED_PROTECTED", "update": True})
            await asyncio.sleep(live_delay)
            await send("KILL_SWITCH", {"engaged": True, "reason": "MANUAL_UI_TRIGGER", "system_status": "HALTED"})

        task: asyncio.Task | None = None
        try:
            await send("INITIAL_SNAPSHOT", build_snapshot())
            task = asyncio.create_task(live())
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "PING":
                    await send("PONG", {"t": msg.get("t")})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if task:
                task.cancel()

    return app


if __name__ == "__main__":  # manual run: python tests/ui/mock_terminal_server.py
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8765)
