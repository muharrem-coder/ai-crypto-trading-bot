"""US equities adapters (Phase 2 templates): Alpaca and Interactive Brokers.

Only the *regulatory and calendar* logic is real today (NYSE regular hours, pattern-day-trader
tracking), so that the risk engine and tests can already depend on it. All broker I/O raises
``BrokerNotSupportedError`` until Phase 2 is enabled.

# TODO: Phase 2 - US Equities (Alpaca/IBKR): wire alpaca-py (REST + websocket) behind these methods.
# TODO: Phase 2 - US Equities (Alpaca/IBKR): NYSE holiday + early-close (13:00) calendar (``NYSE_HOLIDAYS``).
# TODO: Phase 2 - US Equities (Alpaca/IBKR): pre-market (04:00-09:30) / after-hours (16:00-20:00) sessions;
#   extended-hours orders must be LIMIT + DAY only.
# TODO: Phase 2 - US Equities (Alpaca/IBKR): T+1 settlement - cash accounts cannot reuse unsettled proceeds
#   (good-faith violations); margin accounts settle instantly for buying power only.
# TODO: Phase 2 - US Equities (Alpaca/IBKR): short selling needs a locate / easy-to-borrow check and must
#   respect the SEC Rule 201 short-sale restriction (SSR) after a -10% intraday drop.
# TODO: Phase 2 - US Equities (Alpaca/IBKR): fractional shares (market/DAY only on Alpaca), non-1 step sizes.
# TODO: Phase 2 - US Equities (Alpaca/IBKR): SEC 8-K / EDGAR news feed as a trading-halt signal.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from app.brokers.base import BaseBrokerAdapter, MarketHoursStatus
from app.core.errors import BrokerNotSupportedError, RiskViolation
from app.core.types import (
    AccountSnapshot,
    AssetClass,
    FundingInfo,
    InstrumentSpec,
    MarginMode,
    Order,
    OrderRequest,
    Position,
    Ticker,
    utcnow,
)

NY = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PDT_EQUITY_THRESHOLD = Decimal("25000")
PDT_MAX_DAY_TRADES = 3  # a 4th day trade inside the window flags the account as a pattern day trader
PDT_WINDOW_BUSINESS_DAYS = 5

# TODO: Phase 2 - US Equities (Alpaca/IBKR): populate with NYSE holidays (and 13:00 early closes).
NYSE_HOLIDAYS: frozenset[date] = frozenset()

_NOT_ENABLED = "Phase 2: {name} adapter not enabled yet"


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def _window_start(today: date, business_days: int = PDT_WINDOW_BUSINESS_DAYS) -> date:
    """First date of the rolling window of ``business_days`` business days ending on ``today``."""
    d, remaining = today, business_days - 1
    while remaining > 0:
        d -= timedelta(days=1)
        if _is_business_day(d):
            remaining -= 1
    return d


@dataclass(slots=True)
class PDTTracker:
    """Counts day trades (open and close of the same symbol on the same day) over a rolling
    5-business-day window, and remembers which symbols were opened today."""

    day_trades: deque[date] = field(default_factory=deque)
    opened_today: dict[str, date] = field(default_factory=dict)

    def record_open(self, symbol: str, day: date) -> None:
        self.opened_today[symbol] = day

    def record_day_trade(self, day: date) -> None:
        self.day_trades.append(day)

    def count(self, today: date) -> int:
        start = _window_start(today)
        return sum(1 for d in self.day_trades if start <= d <= today)

    def is_closing_a_day_trade(self, symbol: str, today: date) -> bool:
        return self.opened_today.get(symbol) == today

    def would_exceed(self, today: date) -> bool:
        """True if one more day trade would be the 4th (or later) inside the window."""
        return self.count(today) >= PDT_MAX_DAY_TRADES


class _EquityAdapterBase(BaseBrokerAdapter):
    """Shared NYSE-hours and PDT logic for equity brokers."""

    asset_class = AssetClass.US_EQUITY
    supports_shorting = True
    supports_leverage = True
    broker_name = "equity"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        paper: bool = True,
        now: Callable[[], datetime] = utcnow,
        equity: Decimal = Decimal("0"),
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._api_secret = api_secret
        self.paper = paper
        self.now = now
        self.account_equity = equity  # TODO: Phase 2 - US Equities (Alpaca/IBKR): refresh from fetch_account()
        self.pdt = PDTTracker()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(paper={self.paper})"

    # ------------------------------------------------------------------ calendar
    def market_hours_status(self) -> MarketHoursStatus:
        """Regular NYSE session: Mon-Fri 09:30-16:00 America/New_York, excluding ``NYSE_HOLIDAYS``."""
        now_ny = self.now().astimezone(NY)
        today = now_ny.date()
        open_dt = datetime.combine(today, MARKET_OPEN, tzinfo=NY)
        close_dt = datetime.combine(today, MARKET_CLOSE, tzinfo=NY)
        if _is_business_day(today) and open_dt <= now_ny < close_dt:
            return MarketHoursStatus(True, "REGULAR", close_dt.astimezone(timezone.utc))
        # next open: today if before the open on a business day, else the next business day
        candidate = today
        if not (_is_business_day(today) and now_ny < open_dt):
            candidate = today + timedelta(days=1)
            while not _is_business_day(candidate):
                candidate += timedelta(days=1)
        next_open = datetime.combine(candidate, MARKET_OPEN, tzinfo=NY).astimezone(timezone.utc)
        return MarketHoursStatus(False, "CLOSED", next_open)

    def is_market_open(self) -> bool:
        return self.market_hours_status().is_open

    # ------------------------------------------------------------------ regulatory
    def pre_trade_regulatory_check(self, req: OrderRequest) -> None:
        """Raise ``RiskViolation('PDT', ...)`` if this order would be a 4th day trade on a sub-$25k account.

        An order counts as a day trade when it closes (reduce-only or ``metadata['closes_position']``)
        a symbol that was opened earlier the same New York day.
        """
        today = self.now().astimezone(NY).date()
        closing = req.reduce_only or bool(req.metadata.get("closes_position"))
        if not closing or not self.pdt.is_closing_a_day_trade(req.symbol, today):
            return
        if self.account_equity < PDT_EQUITY_THRESHOLD and self.pdt.would_exceed(today):
            raise RiskViolation(
                "PDT",
                f"{req.symbol}: equity {self.account_equity} < {PDT_EQUITY_THRESHOLD} and "
                f"{self.pdt.count(today)} day trades already used in the last "
                f"{PDT_WINDOW_BUSINESS_DAYS} business days; a further day trade is blocked.",
            )

    # ------------------------------------------------------------------ I/O: not enabled
    def _unsupported(self) -> BrokerNotSupportedError:
        return BrokerNotSupportedError(_NOT_ENABLED.format(name=self.broker_name))

    async def connect(self) -> None:
        raise self._unsupported()

    async def close(self) -> None:
        return None  # nothing was opened

    async def load_instruments(self) -> dict[str, InstrumentSpec]:
        raise self._unsupported()

    async def fetch_ticker(self, symbol: str) -> Ticker:
        raise self._unsupported()

    async def fetch_funding(self, symbol: str) -> FundingInfo | None:
        return None  # equities have no funding rate

    async def fetch_account(self) -> AccountSnapshot:
        raise self._unsupported()

    async def fetch_positions(self) -> list[Position]:
        raise self._unsupported()

    async def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        raise self._unsupported()

    async def create_order(self, req: OrderRequest) -> Order:
        raise self._unsupported()

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        raise self._unsupported()

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        raise self._unsupported()

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        raise self._unsupported()

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        raise self._unsupported()

    async def close_position(self, symbol: str) -> Order | None:
        raise self._unsupported()

    async def close_all_positions(self) -> list[Order]:
        raise self._unsupported()


class AlpacaBrokerAdapter(_EquityAdapterBase):
    """Alpaca Markets (US equities). Phase 2 template; I/O disabled.

    # TODO: Phase 2 - US Equities (Alpaca/IBKR): alpaca-py TradingClient / StockHistoricalDataClient.
    # TODO: Phase 2 - US Equities (Alpaca/IBKR): map Alpaca bracket / OCO orders to STOP_MARKET + TAKE_PROFIT_MARKET.
    """

    broker_name = "Alpaca"


class IBKRBrokerAdapter(_EquityAdapterBase):
    """Interactive Brokers (US equities). Phase 2 stub; I/O disabled.

    # TODO: Phase 2 - US Equities (Alpaca/IBKR): ib_async gateway connection, contract qualification, pacing limits.
    """

    broker_name = "IBKR"
