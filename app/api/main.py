"""FastAPI application: lifespan (runtime + WebSocket hub), static terminal page, WS endpoint, REST control API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.backend import RuntimeTerminalBackend
from app.api.routes import router
from app.config.settings import ApiSettings, Settings, get_settings
from app.core.events import EventBus
from app.runtime import TradingRuntime, build_runtime

logger = logging.getLogger(__name__)

DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

RuntimeFactory = Callable[[Settings], Awaitable[TradingRuntime]]
HubFactory = Callable[[EventBus, Any, ApiSettings], Any]


def _default_hub_factory(bus: EventBus, backend: Any, api_settings: ApiSettings) -> Any:
    from app.api.ws_hub import TerminalHub  # imported lazily: written separately, optional for REST-only tests

    return TerminalHub(bus, backend, api_settings)


async def _guard(what: str, coro: Awaitable[Any]) -> None:
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - shutdown continues
        logger.exception("%s failed", what)


def create_app(
    *,
    settings: Settings | None = None,
    runtime_factory: RuntimeFactory | None = None,
    hub_factory: HubFactory | None = None,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> FastAPI:
    settings = settings or get_settings()
    static_dir = Path(static_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        factory = runtime_factory or build_runtime
        runtime = await factory(settings)
        runtime_started = False
        hub: Any = None
        hub_started = False
        try:
            backend = RuntimeTerminalBackend(runtime)
            hub = (hub_factory or _default_hub_factory)(runtime.bus, backend, settings.api)
            await runtime.start()
            runtime_started = True
            await hub.start()
            hub_started = True
            app.state.runtime = runtime
            app.state.backend = backend
            app.state.hub = hub
            logger.info("terminal API ready (symbols=%s)", backend.symbols)
        except BaseException:
            logger.exception("application startup failed")
            if hub_started:
                await _guard("hub stop", hub.stop())
            await _guard("runtime stop", runtime.stop())
            raise
        try:
            yield
        finally:
            app.state.backend = None
            await _guard("hub stop", hub.stop())
            await _guard("runtime stop", runtime.stop())
            app.state.hub = None
            app.state.runtime = None
            logger.info("terminal API stopped")

    app = FastAPI(title="Claude Quant Terminal", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.runtime = None
    app.state.backend = None
    app.state.hub = None
    app.state.kill_switch_lock = asyncio.Lock()
    app.state.kill_switch_last_call = None

    @app.middleware("http")
    async def security_headers(request: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        page = static_dir / "index.html"
        if not page.is_file():
            return JSONResponse(status_code=404, content={"detail": "terminal page not found"})
        return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws/terminal")
    async def terminal_ws(websocket: WebSocket) -> None:
        hub = websocket.app.state.hub
        if hub is None:
            await websocket.close(code=1013)  # try again later: still starting / already stopping
            return
        await hub.serve(websocket)

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=False), name="static")
    return app
