"""``python -m app.api.server``: run the terminal API with uvicorn.

Refuses to bind a non-loopback interface unless ``API_CONTROL_TOKEN`` is set, so the control endpoints can never
be exposed on the network without authentication.

One exception: inside the official Docker image (``QUANT_IN_DOCKER=1``) the app must listen on 0.0.0.0 or the
published port would not work. That is allowed without a token in PAPER / TESTNET only, with a warning; the
compose file publishes the port on 127.0.0.1 of the host. LIVE always needs the token.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys

from app.config.settings import Settings, get_settings
from app.core.types import TradingMode

logger = logging.getLogger(__name__)

IN_DOCKER_ENV = "QUANT_IN_DOCKER"


def is_loopback_host(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip().strip("[]")).is_loopback
    except ValueError:
        return False


def running_in_docker() -> bool:
    return os.environ.get(IN_DOCKER_ENV, "").strip().lower() in {"1", "true", "yes"}


def check_bind_safety(
    host: str,
    control_token: str,
    *,
    trading_mode: TradingMode | None = None,
    in_docker: bool | None = None,
) -> None:
    """Raise SystemExit when ``host`` is reachable from the network and no control token is configured.

    Docker exception: with ``in_docker`` (default: ``QUANT_IN_DOCKER=1``) and a non-LIVE ``trading_mode`` a
    token-less non-loopback bind is allowed and a warning is logged. ``trading_mode=None`` counts as LIVE
    (the strict choice), so callers must pass the mode explicitly to get the exception.
    """
    if is_loopback_host(host) or control_token:
        return
    if in_docker is None:
        in_docker = running_in_docker()
    if in_docker and trading_mode is not None and trading_mode is not TradingMode.LIVE:
        logger.warning(
            "Binding %s without API_CONTROL_TOKEN inside Docker (mode %s). The container port MUST be published on "
            "127.0.0.1 only (ports: \"127.0.0.1:8000:8000\"); never expose it on 0.0.0.0 of the host. "
            "Set API_CONTROL_TOKEN to protect resume / manual close.",
            host, trading_mode.value,
        )
        return
    raise SystemExit(
        f"Refusing to bind {host!r}: it is not a loopback address and API_CONTROL_TOKEN is not set. "
        "Set API_CONTROL_TOKEN (min 16 chars) or bind API_HOST=127.0.0.1 and put a TLS reverse proxy in front."
    )


def main(settings: Settings | None = None) -> None:
    import uvicorn

    from app.api.main import create_app

    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    check_bind_safety(
        settings.api.host, settings.api.control_token.get_secret_value(), trading_mode=settings.trading_mode
    )
    uvicorn.run(
        create_app(settings=settings),
        host=settings.api.host,
        port=settings.api.port,
        log_level="info",
        server_header=False,
        proxy_headers=False,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(0)
