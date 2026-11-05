"""End-to-end Phase 2 integration (integrator-owned).

Real MarketDataEngine (fed by ScriptedMarketSource = mock live WebSocket) + real NewsEngine
(breaking news injected via ingest) + real QuantEngine + SignalPipeline on a real EventBus.

Flow under test:
    WebSocket data -> buffers -> CANDLE_CLOSED -> indicators -> strategies
    -> fee-aware pre-check -> SignalPayload on Topic.CLAUDE_TRIGGER (with deduplicated news)
and the stale-data guard: a dropped / silent feed must produce DATA_PAUSED, never a trigger.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.config.settings import DataSettings, NewsSettings, QuantSettings, RiskSettings
from app.core.events import EventBus, Topic
from app.core.types import PositionSide
from app.data.engine import MarketDataEngine
from app.data.mock_source import ScriptedMarketSource
from app.data.models import DataChannel, GateState
from app.news.engine import NewsEngine
from app.news.sources import RawNewsItem
from app.pipeline import SignalPipeline
from app.quant.contracts import PreFilterOutcome, StrategyName
from app.quant.engine import QuantEngine
from app.quant.payload import payload_to_json
from app.risk.testing import make_spec

SYM = "BTC/USDT:USDT"
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(minutes=130)


def _ms(i: int) -> int:
    return int((START + timedelta(minutes=i)).timestamp() * 1000)


def crafted_rows() -> list[list[float]]:
    """120 calm zig-zag bars, then one violent sell-off bar (index 120), then quiet bars."""
    rows, p = [], 100.0
    for i in range(120):
        o, c = p, 100.0 + (0.1 if i % 2 else -0.1)
        rows.append([_ms(i), o, max(o, c) + 0.15, min(o, c) - 0.15, c, 10.0])
        p = c
    for j, drop in enumerate([2.5, 0.2, -0.1, -0.1, -0.1]):
        o, c = p, p - drop
        rows.append([_ms(120 + j), o, max(o, c) + 0.05, min(o, c) - 0.05, c, 10.0])
        p = c
    return rows


def book_levels(mid: float, bid_qty: float = 11.0, ask_qty: float = 9.0, levels: int = 10):
    bids = [[round(mid - 0.002 - i * 0.01, 6), bid_qty] for i in range(levels)]
    asks = [[round(mid + 0.002 + i * 0.01, 6), ask_qty] for i in range(levels)]
    return bids, asks


class Harness:
    def __init__(self) -> None:
        self.wall = [NOW]
        self.mono = [1000.0]
        self.sleeps: list[float] = []
        self.rows = crafted_rows()
        self.bus = EventBus(default_maxsize=500)
        self.source = ScriptedMarketSource()
        self.source.set_history(SYM, self.rows[:120])
        self.data_settings = DataSettings(stale_after_seconds=2.0, resume_after_live_seconds=1.0,
                                          bootstrap_candles=200, health_check_interval_seconds=0.05)
        self.qs = QuantSettings()

        async def fake_sleep(delay: float) -> None:
            self.sleeps.append(delay)
            await asyncio.sleep(0)

        self.market = MarketDataEngine(self.source, self.bus, [SYM], self.data_settings,
                                       clock=lambda: self.mono[0], wall_clock=lambda: self.wall[0],
                                       sleep=fake_sleep)
        self.news = NewsEngine([], self.bus, NewsSettings(), clock=lambda: self.wall[0])
        self.pipeline: SignalPipeline | None = None
        self.quant: QuantEngine | None = None
        self.triggers = self.bus.subscribe(Topic.CLAUDE_TRIGGER)
        self.rejections = self.bus.subscribe(Topic.SIGNAL_REJECTED)
        self.gate_events = self.bus.subscribe(Topic.TRADING_GATE)

    async def start(self) -> None:
        await self.market.start()
        self.pipeline = None
        quant = QuantEngine(self.qs, RiskSettings(), lambda s: make_spec(s),
                            lambda s: self.pipeline.is_tradeable(s), clock=lambda: self.wall[0])
        self.quant = quant
        self.pipeline = SignalPipeline(self.market, quant, self.bus, self.data_settings, self.qs,
                                       news=self.news, wall_clock=lambda: self.wall[0])
        assert self.pipeline.warmup() == 120
        self.pipeline.start()

    async def stop(self) -> None:
        if self.pipeline:
            await self.pipeline.stop()
        await self.market.stop()

    async def pump(self, n: int = 30) -> None:
        for _ in range(n):
            await asyncio.sleep(0)

    async def feed_quotes(self, mid: float) -> None:
        bids, asks = book_levels(mid)
        self.source.push_order_book(SYM, bids, asks)
        self.source.push_ticker(SYM, bids[0][0], asks[0][0], mid)
        await self.pump()

    async def go_live(self, mid: float) -> None:
        await self.feed_quotes(mid)
        await self.market.run_health_check_once()
        self.mono[0] += 1.2  # stay LIVE longer than resume_after_live_seconds
        await self.feed_quotes(mid)
        await self.market.run_health_check_once()
        assert self.market.gate.is_active(SYM), self.market.gate.paused_reasons(SYM)

    async def close_bar(self, i: int) -> None:
        """Stream bar i as forming, then bar i+1 opens -> bar i closes."""
        self.source.push_ohlcv(SYM, [self.rows[i]])
        self.source.push_ohlcv(SYM, [self.rows[i + 1]])
        await self.pump(60)


def breaking_news() -> list[RawNewsItem]:
    pub = NOW - timedelta(minutes=4)
    return [
        RawNewsItem("rss:coindesk", "cd-1",
                    "Bitcoin ETF sees record $1B inflow as SEC approves new spot listings",
                    "https://www.coindesk.com/markets/2026/11/05/bitcoin-etf-record-inflow/?utm_source=x&utm_medium=rss",
                    pub, "Spot bitcoin ETFs recorded their largest daily inflow."),
        RawNewsItem("cryptopanic", "cp-99",
                    "Bitcoin ETF sees record $1B inflow as SEC approves new spot listings - CoinDesk",
                    "https://cryptopanic.com/news/123456/bitcoin-etf-record-inflow",
                    pub + timedelta(seconds=40), None),
        RawNewsItem("rss:coindesk", "cd-2", "Celebrity chef opens new restaurant downtown",
                    "https://www.coindesk.com/lifestyle/chef", pub, None),
    ]


@pytest.fixture
async def h():
    harness = Harness()
    await harness.start()
    yield harness
    await harness.stop()


async def test_end_to_end_ws_data_to_claude_signal_payload(h: Harness):
    published = await h.news.ingest(breaking_news())
    assert len(published) == 1  # duplicate merged, irrelevant dropped (rule 4)
    assert published[0].symbols == ("BTC",) and {"ETF", "SEC"} <= set(published[0].macro_tags)

    sell_close = h.rows[120][4]
    await h.go_live(sell_close)
    await h.close_bar(120)

    triggers = h.triggers.drain()
    assert len(triggers) == 1, h.quant.stats
    p = triggers[0]
    assert p.symbol == SYM and p.direction is PositionSide.LONG
    assert p.primary_strategy is StrategyName.MEAN_REVERSION
    assert p.indicators.warm and p.indicators.rsi < h.qs.rsi_oversold
    assert p.indicators.obi is not None and p.indicators.obi > 0
    assert p.proposed_stop_loss < p.proposed_entry < p.proposed_take_profit
    assert p.proposed_entry == Decimal(str(sell_close))
    # fee-aware pre-check (rule 3)
    assert p.fee_precheck.passes
    assert p.fee_precheck.expected_gross_profit > p.fee_precheck.round_trip_cost * 2
    # market health travelled with the payload
    assert p.market_health["tradeable"] is True and p.market_health["gate"] == "ACTIVE"
    assert p.market_health["order_book_age_s"] <= 2.0
    # breaking news attached once, with the duplicate source recorded
    assert len(p.recent_news) == 1
    assert p.recent_news[0].duplicate_sources == ("cryptopanic",)
    # JSON package for Claude round-trips with Turkish text intact
    doc = json.loads(payload_to_json(p))
    assert doc["direction"] == "LONG" and doc["fee_precheck"]["passes"] is True
    assert any("aşırı satım" in f for f in doc["confluence_factors"])
    assert h.pipeline.stats["triggers"] == 1


async def test_disconnect_pauses_and_blocks_perfect_setup_then_reconnects(h: Harness):
    sell_close = h.rows[120][4]
    await h.go_live(sell_close)

    h.source.disconnect(SYM, DataChannel.ORDER_BOOK)
    await h.pump(60)
    await h.market.run_health_check_once()
    assert not h.market.gate.is_active(SYM)
    assert h.market.gate.symbol_state(SYM) is GateState.PAUSED

    await h.close_bar(120)  # the exact setup that triggers in the happy path
    assert h.triggers.drain() == []
    rejected = h.rejections.drain()
    assert rejected and rejected[-1].outcome is PreFilterOutcome.DATA_PAUSED
    assert h.sleeps and h.sleeps[0] > 0  # exponential backoff was used before reconnecting

    # reconnect: fresh data flows again; gate re-opens only after resume_after_live_seconds
    await h.feed_quotes(sell_close)
    await h.market.run_health_check_once()
    assert not h.market.gate.is_active(SYM)
    h.mono[0] += 1.2
    await h.feed_quotes(sell_close)
    await h.market.run_health_check_once()
    assert h.market.gate.is_active(SYM)
    assert any(e.state is GateState.PAUSED for e in h.gate_events.drain())


async def test_silent_feed_older_than_2s_blocks_trigger_even_before_health_tick(h: Harness):
    sell_close = h.rows[120][4]
    await h.go_live(sell_close)
    assert h.market.gate.is_active(SYM)
    h.wall[0] = h.wall[0] + timedelta(seconds=2.5)  # no new book for 2.5s, health loop not yet run
    await h.close_bar(120)
    assert h.triggers.drain() == []
    rejected = h.rejections.drain()
    assert rejected and rejected[-1].outcome is PreFilterOutcome.DATA_PAUSED


async def test_buffers_stay_bounded_under_stream(h: Harness):
    await h.go_live(100.0)
    base = 200
    for k in range(1500):
        ts = int((START + timedelta(minutes=base + k)).timestamp() * 1000)
        h.source.push_ohlcv(SYM, [[ts, 100.0, 100.2, 99.8, 100.0, 5.0]])
        if k % 50 == 0:
            await h.pump(5)
    await h.pump(200)
    assert len(h.market.candles(SYM).closed()) <= h.data_settings.candle_buffer_size
    assert h.triggers.queue.maxsize == 500 and h.triggers.queue.qsize() <= 500


@pytest.mark.filterwarnings("ignore:.*does not support Decimal objects natively.*")
async def test_news_persisted_with_tags_and_duplicate_sources_survive_restart():
    from sqlalchemy import select

    from app.db.models import NewsItem
    from app.db.session import async_session_factory, create_engine_from_url, init_db
    from app.news.persistence import NewsPersistence

    engine = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    factory = async_session_factory(engine)
    clock = lambda: NOW  # noqa: E731
    news1 = NewsEngine([], EventBus(), NewsSettings(), persistence=NewsPersistence(factory, clock=clock), clock=clock)
    assert len(await news1.ingest(breaking_news())) == 1
    async with factory() as s:
        row = (await s.execute(select(NewsItem))).scalar_one()
    assert set(row.macro_tags) >= {"ETF", "SEC"} and row.symbols == ["BTC"]
    assert row.duplicate_sources == ["cryptopanic"] and row.news_hash

    # restart: a fresh engine warmed from the DB must not republish the same story
    bus = EventBus()
    sub = bus.subscribe(Topic.NEWS)
    news2 = NewsEngine([], bus, NewsSettings(), persistence=NewsPersistence(factory, clock=clock), clock=clock)
    await news2.start()
    assert await news2.ingest(breaking_news()) == []
    assert sub.drain() == []
    await news2.stop()
    await engine.dispose()


async def test_quiet_ticker_does_not_pause_while_order_book_is_live(h: Harness):
    """Found on live Kraken data: trade-driven ticker streams go quiet for seconds in a healthy market.
    Only the order book (best bid/ask, ~100 ms cadence) gates trading; a stale BOOK still pauses."""
    mid = h.rows[120][4]
    await h.go_live(mid)
    for _ in range(5):  # 5 s with fresh books but no ticker messages at all
        h.mono[0] += 1.0
        bids, asks = book_levels(mid)
        h.source.push_order_book(SYM, bids, asks)
        await h.pump()
        await h.market.run_health_check_once()
        assert h.market.gate.is_active(SYM), h.market.gate.paused_reasons(SYM)
    h.mono[0] += 2.5  # now the BOOK goes silent too
    await h.market.run_health_check_once()
    assert not h.market.gate.is_active(SYM)
