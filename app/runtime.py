"""Trading runtime: builds and orchestrates every component of the system from ``Settings``.

``build_runtime`` wires exactly what ``tests/test_integration_phase3.py`` wires by hand:

    DB -> EventBus -> broker -> market data -> news -> kill switch / hard risk -> recorder (publishing)
       -> position manager -> executor -> watchdog -> quant -> pipeline -> Claude core -> decision service

``TradingRuntime.start`` brings the parts up in a fixed order and tracks which ones are running; if any
step fails everything already started is stopped again and the error is re-raised. ``stop`` tears down in
reverse order, each step bounded by a timeout, never raises and is idempotent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.api.bus_adapters import PublishingRiskSink, PublishingTradeRecorder
from app.brokers.base import BaseBrokerAdapter
from app.brokers.factory import create_broker_async
from app.brokers.paper_adapter import PaperBrokerAdapter
from app.claude.core import ClaudeDecisionCore
from app.claude.decision_log import DecisionLogger
from app.claude.transport import AnthropicTransport, LLMTransport
from app.config.settings import Settings
from app.core.errors import BrokerError, ConfigError
from app.core.events import EventBus, Topic
from app.core.types import (
    InstrumentSpec,
    MarketType,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    TradingMode,
    utcnow,
)
from app.data.engine import MarketDataEngine
from app.data.sources import CCXTProSource, MarketDataSource
from app.db.session import (
    DEFAULT_MIGRATIONS_PATH,
    apply_sql_migrations,
    async_session_factory,
    create_engine_from_url,
    init_db,
)
from app.decision_service import DecisionService, broker_portfolio_context
from app.execution.executor import ExecutionEngine
from app.execution.live_guard import enforce_live_guard
from app.execution.position_manager import PositionManager
from app.execution.reconciliation import ReconciliationService
from app.execution.recorder import DbTradeRecorder
from app.news.engine import NewsEngine, build_default_sources
from app.news.persistence import NewsPersistence
from app.news.sources import NewsSource
from app.pipeline import SignalPipeline
from app.quant.engine import QuantEngine
from app.risk.db_adapters import DbKillSwitchStore, DbRiskEventSink
from app.risk.kill_switch import KillSwitch
from app.risk.risk_engine import HardRiskEngine
from app.risk.watchdog import RiskWatchdog

logger = logging.getLogger(__name__)

STOP_TIMEOUT_SECONDS = 5.0
_FALLBACK_MARKET = MarketType.PERPETUAL


def _market_type(settings: Settings) -> MarketType:
    try:
        mt = MarketType(settings.exchange_market.upper())
    except ValueError:
        raise ConfigError(f"EXCHANGE_MARKET must be SPOT or PERPETUAL, got {settings.exchange_market!r}") from None
    if mt not in (MarketType.SPOT, MarketType.PERPETUAL):
        raise ConfigError(f"EXCHANGE_MARKET {mt.value} is not a crypto market type")
    return mt


class TradingRuntime:
    """Holds every wired component as an attribute and owns start / stop ordering.

    Construct it with :func:`build_runtime`; the constructor only stores references.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        db_engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
        bus: EventBus,
        broker: BaseBrokerAdapter,
        market: MarketDataEngine,
        news: NewsEngine | None,
        kill_switch: KillSwitch,
        risk_engine: HardRiskEngine,
        watchdog: RiskWatchdog,
        recorder: PublishingTradeRecorder,
        position_manager: PositionManager,
        executor: ExecutionEngine,
        quant: QuantEngine,
        pipeline: SignalPipeline,
        decision_log: DecisionLogger,
        claude_core: ClaudeDecisionCore,
        decision_service: DecisionService,
        api_key_present: bool,
        reconciliation: ReconciliationService | None = None,
        broker_connected: bool = False,
        clock: Callable[[], datetime] = utcnow,
        stop_timeout: float = STOP_TIMEOUT_SECONDS,
    ) -> None:
        self.settings = settings
        self.db_engine = db_engine
        self.session_factory = session_factory
        self.bus = bus
        self.broker = broker
        self.market = market
        self.news = news
        self.kill_switch = kill_switch
        self.risk_engine = risk_engine
        self.watchdog = watchdog
        self.recorder = recorder
        self.position_manager = position_manager
        self.executor = executor
        self.quant = quant
        self.pipeline = pipeline
        self.decision_log = decision_log
        self.claude_core = claude_core
        self.decision_service = decision_service
        self.api_key_present = api_key_present
        self.reconciliation = reconciliation
        self.clock = clock
        self.started_at: datetime = clock()
        self.stop_timeout = stop_timeout

        self._broker_connected = broker_connected
        self._started: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        self._running = False
        self._stopped = False
        self._lifecycle_lock = asyncio.Lock()
        self._paper_task: asyncio.Task[None] | None = None
        self._paper_sub: Any = None
        self._recon_started = False

    # ------------------------------------------------------------------ introspection
    @property
    def is_running(self) -> bool:
        return self._running and not self._stopped

    @property
    def started_components(self) -> list[str]:
        """Names of the components currently started, in start order."""
        return [name for name, _ in self._started]

    # ------------------------------------------------------------------ start
    async def start(self) -> None:
        """Start every component in order. On failure stop what was started and re-raise."""
        async with self._lifecycle_lock:
            if self._stopped:
                raise RuntimeError("TradingRuntime was stopped and cannot be restarted; build a new one")
            if self._running:
                return
            try:
                await self._start_components()
            except BaseException:
                logger.exception("runtime start failed; stopping already started components")
                await self._stop_locked()
                raise
            self._running = True
            self.started_at = self.clock()
            logger.info("runtime started: %s", ", ".join(self.started_components))

    async def _start_components(self) -> None:
        if not self._broker_connected:
            await self.broker.connect()
            self._broker_connected = True
        self._mark("broker", self._noop)  # closed last, see _stop_locked

        await self.kill_switch.start()
        self._mark("kill_switch", self._noop)

        # LIVE guard first, then reconcile with the exchange: whatever is open there gets a hard stop (or is closed)
        # BEFORE market data, the decision service or any new trading starts.
        if self.settings.trading_mode is TradingMode.LIVE:
            await enforce_live_guard(self.broker, self.settings)
        if self.reconciliation is not None:
            recon_report = await self.reconciliation.run_startup()
            if self.settings.trading_mode is TradingMode.LIVE and not recon_report.ok:
                raise RuntimeError(f"startup reconciliation failed in LIVE mode: {recon_report.errors}")

        await self.market.start()
        self._mark("market", self.market.stop)

        warmed = self.pipeline.warmup()
        logger.info("pipeline warm-up fed %s candles", warmed)
        self.pipeline.start()
        self._mark("pipeline", self.pipeline.stop)

        if self.news is not None:
            await self.news.start()
            self._mark("news", self.news.stop)

        self.position_manager.start()
        self._mark("position_manager", self.position_manager.stop)

        self.watchdog.start()
        self._mark("watchdog", self.watchdog.stop)

        self.decision_service.start()
        self._mark("decision_service", self.decision_service.stop)

        if isinstance(self.broker, PaperBrokerAdapter):
            self._start_paper_bridge()
            self._mark("paper_bridge", self._stop_paper_bridge)

        if self.reconciliation is not None:  # periodic loop last; stopped first (see _stop_locked)
            self.reconciliation.start()
            self._recon_started = True

    def _mark(self, name: str, stopper: Callable[[], Awaitable[Any]]) -> None:
        self._started.append((name, stopper))

    @staticmethod
    async def _noop() -> None:
        return None

    # ------------------------------------------------------------------ paper price bridge
    def _start_paper_bridge(self) -> None:
        """Feed live order-book tops into the paper exchange so paper fills / stops follow real data."""
        if self._paper_task is not None:
            return
        self._paper_sub = self.bus.subscribe(Topic.ORDER_BOOK, maxsize=500)
        self._paper_task = asyncio.create_task(self._paper_bridge_loop(self._paper_sub), name="paper-price-bridge")

    async def _paper_bridge_loop(self, sub: Any) -> None:
        broker = self.broker
        assert isinstance(broker, PaperBrokerAdapter)
        while True:
            snap = await sub.get()
            try:
                bid, ask, mid = snap.best_bid, snap.best_ask, snap.mid
                if bid is None or ask is None or mid is None:
                    continue
                broker.update_price(snap.symbol, bid, ask, mark=mid)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bad quote must never stop the bridge
                logger.exception("paper price bridge failed for %s", getattr(snap, "symbol", "?"))

    async def _stop_paper_bridge(self) -> None:
        task, self._paper_task = self._paper_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        sub, self._paper_sub = self._paper_sub, None
        if sub is not None:
            self.bus.unsubscribe(sub)

    # ------------------------------------------------------------------ stop
    async def stop(self) -> None:
        """Stop in reverse order (5 s per step), then close the broker and dispose the DB. Idempotent."""
        async with self._lifecycle_lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        started, self._started = self._started, []
        if self.reconciliation is not None:
            await self._guarded("stop reconciliation", self.reconciliation.stop)
            self._recon_started = False
        for name, stopper in reversed(started):
            await self._guarded(f"stop {name}", stopper)
        # the paper bridge may exist even if start failed half-way
        await self._guarded("stop paper_bridge", self._stop_paper_bridge)
        await self._guarded("close broker", self.broker.close)
        await self._guarded("dispose db engine", self.db_engine.dispose)
        logger.info("runtime stopped")

    async def _guarded(self, what: str, fn: Callable[[], Awaitable[Any]]) -> None:
        try:
            await asyncio.wait_for(fn(), timeout=self.stop_timeout)
        except asyncio.TimeoutError:
            logger.error("%s timed out after %.1fs", what, self.stop_timeout)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - shutdown must continue
            logger.exception("%s failed", what)


# ====================================================================== builder


def _account_spec(settings: Settings, broker: BaseBrokerAdapter) -> dict[str, Any]:
    try:
        market_type = _market_type(settings)
    except ConfigError:
        market_type = _FALLBACK_MARKET
    return {
        "broker": "paper" if isinstance(broker, PaperBrokerAdapter) else settings.exchange_id,
        "trading_mode": settings.trading_mode,
        "market_type": market_type,
        "quote_currency": settings.quote_currency,
    }


def _make_spec_provider(broker: BaseBrokerAdapter) -> Callable[[str], InstrumentSpec]:
    def provider(symbol: str) -> InstrumentSpec:
        specs: dict[str, InstrumentSpec] = getattr(broker, "_instruments", None) or {}
        spec = specs.get(symbol)
        if spec is None:
            raise BrokerError(f"Unknown instrument: {symbol}")
        return spec

    return provider


def _make_stop_repair(broker: BaseBrokerAdapter, manager: PositionManager) -> Callable[[Position], Awaitable[Any]]:
    """Watchdog callback: re-place a reduce-only STOP_MARKET at the manager's current stop for the symbol."""

    async def repair(position: Position) -> Any:
        managed = next((m for m in manager.positions() if m.symbol == position.symbol), None)
        if managed is None:
            raise RuntimeError(f"{position.symbol}: no managed stop price to restore")
        req = OrderRequest(
            symbol=position.symbol,
            side=position.side.exit_side,
            order_type=OrderType.STOP_MARKET,
            quantity=position.quantity,
            stop_price=managed.current_stop,
            reduce_only=True,
        )
        order = await broker.create_order(req)
        if order.status in (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
            raise RuntimeError(f"restored stop came back {order.status.value}")
        managed.stop_order_id = order.order_id
        return order

    return repair


BROKER_CONNECT_DELAYS = (2.0, 4.0, 8.0, 16.0, 30.0, 30.0)  # ~90 s before giving up (then Docker restarts us)


async def _connect_broker_with_retry(settings: Settings, sleep: Callable[[float], Awaitable[None]]) -> BaseBrokerAdapter:
    """A temporary exchange outage at boot must not crash-loop the service: retry with backoff, then raise."""
    last: Exception | None = None
    for attempt, delay in enumerate((*BROKER_CONNECT_DELAYS, None), start=1):
        try:
            return await create_broker_async(settings)
        except Exception as exc:  # noqa: BLE001 - ConfigError is permanent, re-raise it at once
            if isinstance(exc, ConfigError):
                raise
            last = exc
            if delay is None:
                break
            logger.warning("exchange connect failed (attempt %d): %r; retrying in %.0fs", attempt, exc, delay)
            await sleep(delay)
    raise RuntimeError(f"could not connect to {settings.exchange_id} after {len(BROKER_CONNECT_DELAYS) + 1} attempts: "
                       f"{last!r}") from last


async def build_runtime(
    settings: Settings,
    *,
    market_source: MarketDataSource | None = None,
    broker: BaseBrokerAdapter | None = None,
    transport: LLMTransport | None = None,
    news_sources: Iterable[NewsSource] | None = None,
    database_url: str | None = None,
    clock: Callable[[], datetime] = utcnow,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    enable_news: bool = True,
) -> TradingRuntime:
    """Wire every component from ``settings``. Anything injected replaces the production default.

    Nothing is started; call ``await runtime.start()``. If building fails half-way, the resources created so
    far (DB engine, broker, market source) are released before the error propagates.
    """
    engine = create_engine_from_url(database_url or settings.database_url)
    created_broker: BaseBrokerAdapter | None = None
    source: MarketDataSource | None = market_source
    try:
        if engine.dialect.name == "postgresql":
            await apply_sql_migrations(engine, DEFAULT_MIGRATIONS_PATH)
        else:
            await init_db(engine)
        factory = async_session_factory(engine)
        bus = EventBus()

        broker_connected = False
        if broker is None:
            created_broker = broker = await _connect_broker_with_retry(settings, sleep)
            broker_connected = True

        if source is None:
            # PAPER: mainnet public data; TESTNET: the testnet's own book (orders fill there); LIVE: mainnet
            source = CCXTProSource(settings.exchange_id, _market_type(settings), settings.uses_exchange_testnet)
        market = MarketDataEngine(
            source, bus, list(settings.symbols), settings.data, clock=monotonic, wall_clock=clock, sleep=sleep,
        )

        news: NewsEngine | None = None
        if enable_news:
            sources = list(news_sources) if news_sources is not None else build_default_sources(settings.news)
            news = NewsEngine(sources, bus, settings.news, persistence=NewsPersistence(factory, clock),
                              clock=clock, sleep=sleep)

        risk_sink = PublishingRiskSink(DbRiskEventSink(factory), bus)
        kill_switch = KillSwitch(broker, DbKillSwitchStore(factory), risk_sink, settings.risk, clock=clock, sleep=sleep)
        risk_engine = HardRiskEngine(settings.risk, kill_switch, risk_sink)

        recorder = PublishingTradeRecorder(DbTradeRecorder(factory, account=_account_spec(settings, broker)), bus)
        manager = PositionManager(broker, recorder, settings.execution, settings.risk, kill_switch=kill_switch,
                                  clock=clock, sleep=sleep)
        executor = ExecutionEngine(
            broker, risk_engine, kill_switch, manager, recorder, settings.execution,
            book_provider=market.order_book, funding_provider=broker.fetch_funding, clock=clock, sleep=sleep,
        )
        watchdog = RiskWatchdog(broker, kill_switch, risk_sink, settings.risk,
                                repair_stop=_make_stop_repair(broker, manager))

        holder: dict[str, SignalPipeline] = {}

        def gate_is_active(symbol: str) -> bool:
            pipe = holder.get("pipeline")
            return pipe.is_tradeable(symbol) if pipe is not None else market.gate.is_active(symbol)

        quant = QuantEngine(settings.quant, settings.risk, _make_spec_provider(broker), gate_is_active, clock=clock)
        pipeline = SignalPipeline(market, quant, bus, settings.data, settings.quant, news=news,
                                  funding_provider=None, wall_clock=clock)
        holder["pipeline"] = pipeline

        api_key = settings.anthropic_api_key.get_secret_value()
        llm: LLMTransport | None = transport
        if llm is None and api_key:
            llm = AnthropicTransport(api_key, settings.claude.timeout_seconds)
        api_key_present = llm is not None
        decision_log = DecisionLogger(factory, max_news_chars=settings.claude.max_news_chars)
        claude_core = ClaudeDecisionCore(settings.claude, llm, api_key_present=api_key_present,
                                         logger_sink=decision_log, clock=clock, monotonic=monotonic)
        reconciliation = ReconciliationService(
            broker, manager, executor, recorder, kill_switch, bus, settings,
            session_factory=factory, clock=clock, sleep=sleep,
        )
        decision_service = DecisionService(
            claude_core, executor, bus, decision_log=decision_log,
            portfolio_provider=lambda symbol: broker_portfolio_context(broker, kill_switch, symbol), clock=clock,
        )

        return TradingRuntime(
            settings=settings, db_engine=engine, session_factory=factory, bus=bus, broker=broker, market=market,
            news=news, kill_switch=kill_switch, risk_engine=risk_engine, watchdog=watchdog, recorder=recorder,
            position_manager=manager, executor=executor, quant=quant, pipeline=pipeline, decision_log=decision_log,
            claude_core=claude_core, decision_service=decision_service, api_key_present=api_key_present,
            reconciliation=reconciliation, broker_connected=broker_connected, clock=clock,
        )
    except BaseException:
        logger.exception("build_runtime failed; releasing partially created resources")
        if created_broker is not None:
            with contextlib.suppress(Exception):
                await created_broker.close()
        if market_source is None and source is not None:
            with contextlib.suppress(Exception):
                await source.close()
        with contextlib.suppress(Exception):
            await engine.dispose()
        raise

