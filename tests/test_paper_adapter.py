"""PaperBrokerAdapter behaviour: fills, stops, trailing, margin, reduce-only (deterministic, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from app.brokers.paper_adapter import PaperBrokerAdapter
from app.core.errors import (
    BrokerError,
    BrokerNotSupportedError,
    InsufficientFundsError,
    OrderRejectedError,
)
from app.core.types import (
    LiquidityRole,
    MarginMode,
    MarketType,
    OrderRequest,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    Ticker,
    TimeInForce,
)
from tests.test_base_helpers import make_spec

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
T0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> datetime:
        return self.t


def make_broker(balance: str = "10000", slippage: str = "0", clock: Clock | None = None, **kw) -> PaperBrokerAdapter:
    specs = {BTC: make_spec(BTC), ETH: make_spec(ETH, base="ETH")}
    return PaperBrokerAdapter(
        specs, D(balance), "USDT", D(slippage), now=clock or Clock(), **kw
    )


def market(side: Side, qty: str, symbol: str = BTC, **kw) -> OrderRequest:
    return OrderRequest(symbol=symbol, side=side, order_type=OrderType.MARKET, quantity=D(qty), **kw)


def stop(side: Side, qty: str, stop_price: str, otype=OrderType.STOP_MARKET, symbol: str = BTC) -> OrderRequest:
    return OrderRequest(
        symbol=symbol, side=side, order_type=otype, quantity=D(qty), stop_price=D(stop_price), reduce_only=True
    )


@pytest.fixture
def broker() -> PaperBrokerAdapter:
    b = make_broker()
    b.update_price(BTC, D("99.9"), D("100.0"))
    b.update_price(ETH, D("49.9"), D("50.0"))
    return b


# ---------------------------------------------------------------- market fills
async def test_open_long_market(broker):
    order = await broker.create_order(market(Side.BUY, "10", leverage=5))
    assert order.status is OrderStatus.FILLED
    assert order.average_fill_price == D("100.0")  # buys at ask
    assert order.fee_paid == D("0.5")  # 1000 * 0.0005 taker
    (pos,) = await broker.fetch_positions()
    assert pos.side is PositionSide.LONG and pos.quantity == D("10") and pos.leverage == 5
    assert pos.initial_margin == D("200")
    assert pos.margin_mode is MarginMode.ISOLATED
    assert pos.liquidation_price == D("100") * (1 - D("0.2") + D("0.005"))
    acct = await broker.fetch_account()
    assert acct.wallet_balance == D("9999.5")
    assert acct.available_balance == D("9799.5")
    assert broker.fills[0].liquidity is LiquidityRole.TAKER


async def test_slippage_moves_fill_against_trader():
    b = make_broker(slippage="10")  # 10 bps
    b.update_price(BTC, D("99.9"), D("100.0"))
    buy = await b.create_order(market(Side.BUY, "1"))
    assert buy.average_fill_price == D("100.1")
    sell = await b.create_order(market(Side.SELL, "1"))
    assert sell.average_fill_price == D("99.8")  # 99.9 * 0.999 = 99.8001 -> floor to tick


async def test_unknown_symbol_and_no_price():
    b = make_broker()
    with pytest.raises(BrokerError):
        await b.create_order(market(Side.BUY, "1", symbol="XRP/USDT:USDT"))
    with pytest.raises(BrokerError, match="no price"):
        await b.create_order(market(Side.BUY, "1"))


async def test_price_source_used_when_no_quote():
    async def source(symbol: str) -> Ticker:
        return Ticker(symbol, D("99.9"), D("100.0"), D("99.95"), None, T0)

    b = make_broker(price_source=source)
    order = await b.create_order(market(Side.BUY, "1"))
    assert order.average_fill_price == D("100.0")
    ticker = await b.fetch_ticker(BTC)
    assert ticker.bid == D("99.9")


# ---------------------------------------------------------------- stop loss / take profit
async def test_stop_loss_triggers_with_correct_net_pnl(broker):
    await broker.create_order(market(Side.BUY, "10", leverage=5))
    sl = await broker.create_order(stop(Side.SELL, "10", "95.0"))
    assert sl.status is OrderStatus.NEW
    assert [o.order_id for o in await broker.fetch_open_orders(BTC)] == [sl.order_id]

    broker.update_price(BTC, D("96.9"), D("97.0"))  # not yet
    assert await broker.fetch_positions()

    broker.update_price(BTC, D("94.9"), D("95.0"))  # mid 94.95 <= 95 -> triggers, fills at bid
    assert await broker.fetch_positions() == []
    assert await broker.fetch_open_orders() == []
    exit_fill = broker.fills[-1]
    assert exit_fill.price == D("94.9") and exit_fill.liquidity is LiquidityRole.TAKER
    # gross -51, fees 0.5 + 0.4745
    assert broker.gross_realized_pnl_today == D("-51.0")
    assert broker.realized_pnl_today == D("-51.9745")
    acct = await broker.fetch_account()
    assert acct.wallet_balance == D("10000") - D("51.9745")
    assert acct.available_balance == acct.wallet_balance  # margin released
    assert broker.orders[sl.order_id].status is OrderStatus.FILLED


async def test_take_profit_and_oco_cleanup(broker):
    await broker.create_order(market(Side.BUY, "10"))
    sl = await broker.create_order(stop(Side.SELL, "10", "90.0"))
    tp = await broker.create_order(stop(Side.SELL, "10", "110.0", OrderType.TAKE_PROFIT_MARKET))
    broker.update_price(BTC, D("110.1"), D("110.2"))
    assert await broker.fetch_positions() == []
    assert broker.orders[tp.order_id].status is OrderStatus.FILLED
    assert broker.orders[sl.order_id].status is OrderStatus.CANCELED  # reduce-only leftover cancelled
    assert await broker.fetch_open_orders() == []
    assert broker.gross_realized_pnl_today == D("101.0")  # (110.1-100)*10


async def test_stop_that_would_trigger_immediately_is_rejected(broker):
    await broker.create_order(market(Side.BUY, "1"))
    with pytest.raises(OrderRejectedError, match="immediately"):
        await broker.create_order(stop(Side.SELL, "1", "100.0"))  # mid 99.95 <= 100


async def test_short_side_pnl(broker):
    sell = await broker.create_order(market(Side.SELL, "10", leverage=2))
    assert sell.average_fill_price == D("99.9")  # sells at bid
    (pos,) = await broker.fetch_positions()
    assert pos.side is PositionSide.SHORT
    assert pos.liquidation_price == D("99.9") * (1 + D("0.5") - D("0.005"))
    sl = await broker.create_order(stop(Side.BUY, "10", "105.0"))
    broker.update_price(BTC, D("89.9"), D("90.0"))
    (pos,) = await broker.fetch_positions()
    assert pos.unrealized_pnl == D("99.9") * 10 - D("89.95") * 10 and pos.unrealized_pnl > 0
    order = await broker.close_position(BTC)
    assert order.average_fill_price == D("90.0")  # buys back at ask
    assert broker.gross_realized_pnl_today == (D("99.9") - D("90.0")) * 10
    fees = D("99.9") * 10 * D("0.0005") + D("90.0") * 10 * D("0.0005")
    assert broker.realized_pnl_today == D("99.0") - fees
    assert broker.orders[sl.order_id].status is OrderStatus.CANCELED


async def test_short_stop_loss_triggers_on_rise(broker):
    await broker.create_order(market(Side.SELL, "10"))
    await broker.create_order(stop(Side.BUY, "10", "105.0"))
    broker.update_price(BTC, D("105.4"), D("105.5"))
    assert await broker.fetch_positions() == []
    assert broker.gross_realized_pnl_today == (D("99.9") - D("105.5")) * 10


# ---------------------------------------------------------------- trailing
async def test_trailing_stop_follows_high_water_and_triggers(broker):
    await broker.create_order(market(Side.BUY, "10"))
    tr = await broker.create_order(
        OrderRequest(
            symbol=BTC, side=Side.SELL, order_type=OrderType.TRAILING_STOP_MARKET, quantity=D("10"),
            trailing_callback_rate=D("0.01"), reduce_only=True,
        )
    )
    broker.update_price(BTC, D("109.9"), D("110.0"), mark=D("110"))
    (open_order,) = await broker.fetch_open_orders()
    assert open_order.stop_price == D("108.9")  # 110 * 0.99
    broker.update_price(BTC, D("108.9"), D("109.0"), mark=D("109.0"))  # above 108.9: still alive
    assert await broker.fetch_positions()
    broker.update_price(BTC, D("112.9"), D("113.0"), mark=D("113"))  # new high-water
    (open_order,) = await broker.fetch_open_orders()
    assert open_order.stop_price == D("111.87")
    broker.update_price(BTC, D("111.7"), D("111.8"), mark=D("111.8"))  # <= 111.87
    assert await broker.fetch_positions() == []
    assert broker.fills[-1].price == D("111.7")
    assert broker.orders[tr.order_id].status is OrderStatus.FILLED


async def test_trailing_stop_waits_for_activation_price(broker):
    await broker.create_order(market(Side.BUY, "10"))
    await broker.create_order(
        OrderRequest(
            symbol=BTC, side=Side.SELL, order_type=OrderType.TRAILING_STOP_MARKET, quantity=D("10"),
            trailing_callback_rate=D("0.01"), trailing_activation_price=D("105"), reduce_only=True,
        )
    )
    broker.update_price(BTC, D("94.9"), D("95.0"))  # big drop but not active -> no trigger
    assert await broker.fetch_positions()
    broker.update_price(BTC, D("106.9"), D("107.0"), mark=D("107"))  # activates, hwm 107
    broker.update_price(BTC, D("105.8"), D("105.9"), mark=D("105.9"))  # 107*.99 = 105.93 -> hit
    assert await broker.fetch_positions() == []


async def test_trailing_buy_for_short(broker):
    await broker.create_order(market(Side.SELL, "10"))
    await broker.create_order(
        OrderRequest(
            symbol=BTC, side=Side.BUY, order_type=OrderType.TRAILING_STOP_MARKET, quantity=D("10"),
            trailing_callback_rate=D("0.02"), reduce_only=True,
        )
    )
    broker.update_price(BTC, D("89.9"), D("90.0"), mark=D("90"))  # low-water 90 -> trigger 91.8
    assert await broker.fetch_positions()
    broker.update_price(BTC, D("91.8"), D("91.9"), mark=D("91.8"))
    assert await broker.fetch_positions() == []


# ---------------------------------------------------------------- reduce-only
async def test_reduce_only_cannot_flip_or_increase(broker):
    await broker.create_order(market(Side.BUY, "10"))
    with pytest.raises(OrderRejectedError, match="exceeds position"):
        await broker.create_order(market(Side.SELL, "15", reduce_only=True))
    with pytest.raises(OrderRejectedError, match="increase"):
        await broker.create_order(market(Side.BUY, "1", reduce_only=True))
    (pos,) = await broker.fetch_positions()
    assert pos.quantity == D("10") and pos.side is PositionSide.LONG


async def test_reduce_only_without_position_rejected(broker):
    with pytest.raises(OrderRejectedError, match="no open position"):
        await broker.create_order(market(Side.SELL, "1", reduce_only=True))


async def test_reduce_only_partial_close(broker):
    await broker.create_order(market(Side.BUY, "10", leverage=5))
    await broker.create_order(market(Side.SELL, "4", reduce_only=True))
    (pos,) = await broker.fetch_positions()
    assert pos.quantity == D("6") and pos.initial_margin == D("120")


async def test_non_reduce_only_opposite_order_flips(broker):
    await broker.create_order(market(Side.BUY, "10"))
    await broker.create_order(market(Side.SELL, "15"))
    (pos,) = await broker.fetch_positions()
    assert pos.side is PositionSide.SHORT and pos.quantity == D("5")


# ---------------------------------------------------------------- funds / margin
async def test_insufficient_funds_leaves_state_untouched():
    b = make_broker(balance="100")
    b.update_price(BTC, D("99.9"), D("100.0"))
    with pytest.raises(InsufficientFundsError):
        await b.create_order(market(Side.BUY, "10"))  # margin 1000 at 1x
    assert await b.fetch_positions() == []
    assert b.wallet == D("100") and b.fills == []
    # 5x leverage -> 200 margin still too much; 10x -> 100 margin + fee > free
    with pytest.raises(InsufficientFundsError):
        await b.create_order(market(Side.BUY, "10", leverage=10))
    ok = await b.create_order(market(Side.BUY, "5", leverage=10))  # 50 margin
    assert ok.status is OrderStatus.FILLED


async def test_margin_plus_fee_must_fit_exactly():
    b = make_broker(balance="100.5")
    b.update_price(BTC, D("99.9"), D("100.0"))
    await b.create_order(market(Side.BUY, "1", leverage=1))  # margin 100 + fee 0.05 <= 100.5
    assert b.free_balance == D("0.45")


async def test_leverage_validation(broker):
    with pytest.raises(OrderRejectedError):
        await broker.set_leverage(BTC, 50)
    with pytest.raises(OrderRejectedError):
        await broker.set_leverage(BTC, 0)
    await broker.set_leverage(BTC, 3)
    order = await broker.create_order(market(Side.BUY, "3"))
    (pos,) = await broker.fetch_positions()
    assert pos.leverage == 3 and pos.initial_margin == D("100")


async def test_cross_margin_not_supported(broker):
    with pytest.raises(BrokerNotSupportedError):
        await broker.set_margin_mode(BTC, MarginMode.CROSS)
    await broker.set_margin_mode(BTC, MarginMode.ISOLATED)


async def test_spec_validation_applies(broker):
    with pytest.raises(OrderRejectedError, match="step_size"):
        await broker.create_order(market(Side.BUY, "0.0015"))
    with pytest.raises(OrderRejectedError, match="min_notional"):
        await broker.create_order(market(Side.BUY, "0.01"))
    with pytest.raises(OrderRejectedError, match="tick"):
        await broker.create_order(
            OrderRequest(symbol=BTC, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("1"), price=D("95.05"))
        )


# ---------------------------------------------------------------- limit orders
async def test_resting_limit_fills_as_maker(broker):
    lim = await broker.create_order(
        OrderRequest(symbol=BTC, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("10"), price=D("98.0"))
    )
    assert lim.status is OrderStatus.NEW and (await broker.fetch_open_orders())[0].order_id == lim.order_id
    broker.update_price(BTC, D("97.8"), D("97.9"))
    assert await broker.fetch_open_orders() == []
    fill = broker.fills[-1]
    assert fill.price == D("98.0") and fill.liquidity is LiquidityRole.MAKER
    assert fill.fee == D("98.0") * 10 * D("0.0002")


async def test_marketable_limit_fills_immediately_as_taker(broker):
    o = await broker.create_order(
        OrderRequest(symbol=BTC, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("1"), price=D("101.0"))
    )
    assert o.status is OrderStatus.FILLED and o.average_fill_price == D("100.0")
    assert broker.fills[-1].liquidity is LiquidityRole.TAKER


async def test_post_only_crossing_rejected_and_ioc_expires(broker):
    with pytest.raises(OrderRejectedError, match="post-only"):
        await broker.create_order(
            OrderRequest(symbol=BTC, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("1"), price=D("101.0"),
                         time_in_force=TimeInForce.POST_ONLY)
        )
    ioc = await broker.create_order(
        OrderRequest(symbol=BTC, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("1"), price=D("90.0"),
                     time_in_force=TimeInForce.IOC)
    )
    assert ioc.status is OrderStatus.EXPIRED and await broker.fetch_open_orders() == []


async def test_cancel_order_and_cancel_all(broker):
    a = await broker.create_order(
        OrderRequest(symbol=BTC, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("1"), price=D("90.0"))
    )
    await broker.create_order(
        OrderRequest(symbol=ETH, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("1"), price=D("40.0"))
    )
    await broker.create_order(
        OrderRequest(symbol=ETH, side=Side.BUY, order_type=OrderType.LIMIT, quantity=D("1"), price=D("41.0"))
    )
    cancelled = await broker.cancel_order(a.order_id, BTC)
    assert cancelled.status is OrderStatus.CANCELED
    with pytest.raises(BrokerError):
        await broker.cancel_order(a.order_id, BTC)
    assert await broker.cancel_all_orders(ETH) == 2
    assert await broker.cancel_all_orders() == 0


# ---------------------------------------------------------------- close all / liquidation / account
async def test_close_all_positions(broker):
    await broker.create_order(market(Side.BUY, "10", leverage=5))
    await broker.create_order(market(Side.SELL, "20", symbol=ETH, leverage=5))
    await broker.create_order(stop(Side.SELL, "10", "90.0"))
    orders = await broker.close_all_positions()
    assert len(orders) == 2 and all(o.status is OrderStatus.FILLED and o.reduce_only for o in orders)
    assert await broker.fetch_positions() == []
    assert await broker.fetch_open_orders() == []
    assert await broker.close_position(BTC) is None
    acct = await broker.fetch_account()
    assert acct.available_balance == acct.wallet_balance == acct.equity
    assert acct.margin_ratio == D("0")


async def test_liquidation_closes_position_at_liq_price():
    b = make_broker()
    b.update_price(BTC, D("99.9"), D("100.0"))
    await b.create_order(market(Side.BUY, "10", leverage=10))
    liq = D("100") * (1 - D("0.1") + D("0.005"))  # 90.5
    b.update_price(BTC, D("90.6"), D("90.7"))
    assert await b.fetch_positions()
    b.update_price(BTC, D("90.0"), D("90.1"))
    assert await b.fetch_positions() == []
    assert b.fills[-1].price == liq
    assert b.gross_realized_pnl_today == (liq - D("100")) * 10


async def test_account_equity_and_margin_ratio(broker):
    await broker.create_order(market(Side.BUY, "10", leverage=5))
    broker.update_price(BTC, D("109.9"), D("110.0"))
    acct = await broker.fetch_account()
    assert acct.unrealized_pnl == D("109.95") * 10 - D("1000")
    assert acct.equity == acct.wallet_balance + acct.unrealized_pnl
    assert acct.margin_ratio == D("10") * D("109.95") * D("0.005") / acct.equity


async def test_realized_pnl_today_resets_on_new_utc_day():
    clock = Clock()
    b = make_broker(clock=clock)
    b.update_price(BTC, D("99.9"), D("100.0"))
    await b.create_order(market(Side.BUY, "1"))
    await b.create_order(market(Side.SELL, "1", reduce_only=True))
    assert b.realized_pnl_today != 0
    clock.t = clock.t + timedelta(days=1)
    assert b.realized_pnl_today == 0


async def test_funding_none_by_default_and_settable(broker):
    from app.core.types import FundingInfo

    assert await broker.fetch_funding(BTC) is None
    info = FundingInfo(BTC, D("0.0001"), 8, None)
    broker.set_funding(info)
    assert await broker.fetch_funding(BTC) is info


def test_update_price_validation(broker):
    with pytest.raises(ValueError):
        broker.update_price(BTC, D("101"), D("100"))


# ---------------------------------------------------------------- spot
SPOT = "BTC/USDT"


def make_spot(balance: str = "10000") -> PaperBrokerAdapter:
    spec = make_spec(SPOT, market_type=MarketType.SPOT, settle=None, max_leverage=1)
    b = PaperBrokerAdapter({SPOT: spec}, D(balance), "USDT", D("0"), now=Clock())
    b.update_price(SPOT, D("99.9"), D("100.0"))
    return b


async def test_spot_buy_sell_roundtrip():
    b = make_spot()
    assert b.market_type is MarketType.SPOT and not b.supports_shorting and not b.supports_leverage
    await b.create_order(market(Side.BUY, "10", symbol=SPOT))
    acct = await b.fetch_account()
    assert acct.margin_ratio is None
    assert acct.available_balance == D("10000") - D("1000") - D("0.5")
    b.update_price(SPOT, D("109.9"), D("110.0"))
    await b.create_order(market(Side.SELL, "10", symbol=SPOT))
    assert await b.fetch_positions() == []
    acct = await b.fetch_account()
    assert acct.wallet_balance == D("10000") + D("99") - D("0.5") - D("109.9") * 10 * D("0.0005")
    assert await b.fetch_funding(SPOT) is None


async def test_spot_rejects_short_and_leverage():
    b = make_spot()
    with pytest.raises(BrokerNotSupportedError):
        await b.create_order(market(Side.SELL, "1", symbol=SPOT))
    await b.create_order(market(Side.BUY, "1", symbol=SPOT))
    with pytest.raises(BrokerNotSupportedError):
        await b.create_order(market(Side.SELL, "2", symbol=SPOT))
    with pytest.raises(BrokerNotSupportedError):
        await b.create_order(market(Side.BUY, "1", symbol=SPOT, leverage=2))
    with pytest.raises(BrokerNotSupportedError):
        await b.set_leverage(SPOT, 3)


async def test_spot_insufficient_funds():
    b = make_spot("500")
    with pytest.raises(InsufficientFundsError):
        await b.create_order(market(Side.BUY, "10", symbol=SPOT))
