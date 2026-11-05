"""Pure ccxt<->domain mapping functions (no network, no exchange instance traffic)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal as D

import ccxt
import pytest

import app.brokers.ccxt_adapter as mod
from app.brokers.ccxt_adapter import (
    CCXTBrokerAdapter,
    build_order_params,
    ccxt_order_to_order,
    ccxt_position_to_position,
    ccxt_ticker_to_ticker,
    compute_margin_ratio,
    funding_to_info,
    map_ccxt_exception,
    market_is_wanted,
    market_to_spec,
    _d,
)
from app.core.errors import (
    BrokerError,
    BrokerNotSupportedError,
    ConfigError,
    InsufficientFundsError,
    OrderRejectedError,
)
from app.core.types import (
    MarginMode,
    MarketType,
    OrderRequest,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    TimeInForce,
)

MARKET = {
    "symbol": "BTC/USDT:USDT", "base": "BTC", "quote": "USDT", "settle": "USDT",
    "spot": False, "swap": True, "future": False, "contract": True, "linear": True, "active": True,
    "contractSize": 1.0, "maker": 0.0002, "taker": 0.0005,
    "precision": {"price": 0.1, "amount": 0.001},
    "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}, "leverage": {"max": 125}},
}


def R(order_type: OrderType, **kw) -> OrderRequest:
    base = dict(symbol="BTC/USDT:USDT", side=Side.SELL, order_type=order_type, quantity=D("0.5"))
    base.update(kw)
    return OrderRequest(**base)


# ------------------------------------------------------------------ helpers
def test_d_conversions():
    assert _d(None) is None
    assert _d(0.1) == D("0.1")  # via str, no float noise
    assert _d("1.25") == D("1.25")
    assert _d(3) == D("3")
    assert _d("") is None


# ------------------------------------------------------------------ markets
def test_market_to_spec_tick_size_mode():
    spec = market_to_spec(MARKET, ccxt.TICK_SIZE)
    assert spec.tick_size == D("0.1") and spec.step_size == D("0.001")
    assert spec.min_qty == D("0.001") and spec.min_notional == D("5.0")
    assert spec.max_leverage == 125 and spec.market_type is MarketType.PERPETUAL
    assert spec.fees.maker_rate == D("0.0002") and spec.fees.taker_rate == D("0.0005")
    assert spec.settle == "USDT" and spec.base == "BTC"


def test_market_to_spec_decimal_places_mode():
    m = {**MARKET, "precision": {"price": 2, "amount": 3}}
    spec = market_to_spec(m, ccxt.DECIMAL_PLACES)
    assert spec.tick_size == D("0.01") and spec.step_size == D("0.001")


def test_market_to_spec_significant_digits_unsupported():
    with pytest.raises(BrokerNotSupportedError):
        market_to_spec(MARKET, ccxt.SIGNIFICANT_DIGITS)


def test_market_to_spec_defaults_and_spot():
    m = {
        "symbol": "ETH/USDT", "base": "ETH", "quote": "USDT", "settle": None, "spot": True, "swap": False,
        "precision": {"price": 0.01, "amount": 0.0001}, "limits": {"amount": {}, "cost": {}}, "maker": None,
    }
    spec = market_to_spec(m, ccxt.TICK_SIZE)
    assert spec.market_type is MarketType.SPOT
    assert spec.min_qty == spec.step_size == D("0.0001")
    assert spec.min_notional == D("0") and spec.max_leverage == 1
    assert spec.fees.maker_rate > 0  # default fee schedule


def test_market_to_spec_contract_size_converts_to_base_units():
    m = {**MARKET, "contractSize": 0.01, "precision": {"price": 0.1, "amount": 1}, "limits": {"amount": {"min": 1}}}
    spec = market_to_spec(m, ccxt.TICK_SIZE)
    assert spec.contract_size == D("0.01")
    assert spec.step_size == D("0.01") and spec.min_qty == D("0.01")


def test_market_is_wanted():
    assert market_is_wanted(MARKET, MarketType.PERPETUAL)
    assert not market_is_wanted(MARKET, MarketType.SPOT)
    assert not market_is_wanted({**MARKET, "linear": False}, MarketType.PERPETUAL)
    assert not market_is_wanted({**MARKET, "active": False}, MarketType.PERPETUAL)
    assert market_is_wanted({"spot": True, "swap": False}, MarketType.SPOT)


# ------------------------------------------------------------------ orders
def order_dict(**kw):
    d = {
        "id": 12345, "clientOrderId": "cid-1", "symbol": "BTC/USDT:USDT", "type": "limit", "side": "buy",
        "status": "open", "amount": 0.5, "filled": 0.0, "price": 100.5, "average": None, "stopPrice": None,
        "reduceOnly": False, "timestamp": 1_767_614_400_000, "fee": None,
    }
    d.update(kw)
    return d


@pytest.mark.parametrize(
    "status,filled,expected",
    [
        ("open", 0.0, OrderStatus.NEW),
        ("open", 0.2, OrderStatus.PARTIALLY_FILLED),
        ("closed", 0.5, OrderStatus.FILLED),
        ("canceled", 0.0, OrderStatus.CANCELED),
        ("cancelled", 0.0, OrderStatus.CANCELED),
        ("rejected", 0.0, OrderStatus.REJECTED),
        ("expired", 0.0, OrderStatus.EXPIRED),
    ],
)
def test_order_status_map(status, filled, expected):
    assert ccxt_order_to_order(order_dict(status=status, filled=filled)).status is expected


def test_order_basic_fields_are_decimal_utc():
    o = ccxt_order_to_order(order_dict(status="closed", filled=0.5, average=100.25,
                                       fee={"cost": 0.025, "currency": "USDT"}))
    assert o.order_id == "12345" and o.client_order_id == "cid-1"
    assert o.side is Side.BUY and o.order_type is OrderType.LIMIT
    assert o.quantity == D("0.5") and o.filled_quantity == D("0.5")
    assert o.price == D("100.5") and o.average_fill_price == D("100.25")
    assert o.fee_paid == D("0.025") and o.fee_currency == "USDT"
    assert o.created_at == datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    assert o.stop_price is None and o.reduce_only is False


def test_order_fees_list_summed():
    o = ccxt_order_to_order(order_dict(fees=[{"cost": 0.01, "currency": "USDT"}, {"cost": 0.02, "currency": "USDT"}]))
    assert o.fee_paid == D("0.03")


@pytest.mark.parametrize(
    "raw_type,extra,expected",
    [
        ("market", {}, OrderType.MARKET),
        ("limit", {}, OrderType.LIMIT),
        ("STOP_MARKET", {"stopPrice": 95.0}, OrderType.STOP_MARKET),
        ("stop_market", {}, OrderType.STOP_MARKET),
        ("take_profit_market", {"stopPrice": 110}, OrderType.TAKE_PROFIT_MARKET),
        ("TRAILING_STOP_MARKET", {"stopPrice": 0}, OrderType.TRAILING_STOP_MARKET),
        ("stop_limit", {"stopPrice": 95}, OrderType.LIMIT),
        ("market", {"stopLossPrice": 95.0}, OrderType.STOP_MARKET),
        ("market", {"takeProfitPrice": 110.0}, OrderType.TAKE_PROFIT_MARKET),
    ],
)
def test_order_type_map(raw_type, extra, expected):
    assert ccxt_order_to_order(order_dict(type=raw_type, **extra)).order_type is expected


def test_order_stop_price_zero_is_none_and_reduce_only():
    o = ccxt_order_to_order(order_dict(type="TRAILING_STOP_MARKET", stopPrice=0, reduceOnly=True, side="sell"))
    assert o.stop_price is None and o.reduce_only is True and o.side is Side.SELL
    o2 = ccxt_order_to_order(order_dict(type="STOP_MARKET", stopPrice="95.5"))
    assert o2.stop_price == D("95.5")


def test_order_contract_size_scaling_and_request_fallback():
    req = R(OrderType.MARKET, side=Side.BUY, client_order_id="mine", reduce_only=True)
    bare = {"id": "9", "status": "open"}  # bare create-ack
    o = ccxt_order_to_order(bare, request=req)
    assert o.symbol == req.symbol and o.side is Side.BUY and o.quantity == D("0.5")
    assert o.client_order_id == "mine" and o.reduce_only is True and o.order_type is OrderType.MARKET
    scaled = ccxt_order_to_order(order_dict(amount=50, filled=20), contract_size=D("0.01"))
    assert scaled.quantity == D("0.5") and scaled.filled_quantity == D("0.2")


def test_order_without_side_raises():
    with pytest.raises(BrokerError):
        ccxt_order_to_order({"id": "1"})


# ------------------------------------------------------------------ positions / ticker / funding
def test_position_mapping():
    raw = {"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "short", "entryPrice": 100, "markPrice": 95,
           "leverage": 5, "marginMode": "isolated", "unrealizedPnl": 2.5, "liquidationPrice": 119.5,
           "initialMargin": 9.5, "timestamp": 1_767_614_400_000}
    p = ccxt_position_to_position(raw)
    assert p.side is PositionSide.SHORT and p.quantity == D("0.5") and p.leverage == 5
    assert p.unrealized_pnl == D("2.5") and p.liquidation_price == D("119.5")
    assert p.margin_mode is MarginMode.ISOLATED and p.initial_margin == D("9.5")
    assert ccxt_position_to_position({**raw, "contracts": 0}) is None
    assert ccxt_position_to_position({**raw, "marginMode": "cross"}).margin_mode is MarginMode.CROSS


def test_position_computes_pnl_when_missing_and_scales_contracts():
    raw = {"symbol": "X", "contracts": 100, "side": "long", "entryPrice": 10, "markPrice": 11, "leverage": 3,
           "contractSize": 0.1}
    p = ccxt_position_to_position(raw)
    assert p.quantity == D("10.0") and p.unrealized_pnl == D("10.0")


def test_ticker_mapping_and_fallbacks():
    t = ccxt_ticker_to_ticker({"symbol": "BTC/USDT:USDT", "bid": 99.9, "ask": 100.0, "last": 99.95,
                               "timestamp": 1_767_614_400_000, "info": {"markPrice": "99.97"}})
    assert t.bid == D("99.9") and t.mark == D("99.97") and t.timestamp.tzinfo is not None
    only_last = ccxt_ticker_to_ticker({"last": 50.0}, "ETH/USDT")
    assert only_last.bid == only_last.ask == D("50.0") and only_last.symbol == "ETH/USDT"
    with pytest.raises(BrokerError):
        ccxt_ticker_to_ticker({"bid": None, "ask": None, "last": None}, "X")


def test_funding_mapping():
    f = funding_to_info({"fundingRate": 0.0001, "nextFundingTimestamp": 1_767_628_800_000, "interval": "4h"}, "BTC/USDT:USDT")
    assert f.rate == D("0.0001") and f.interval_hours == 4 and f.next_funding_time.tzinfo is not None
    assert funding_to_info({"fundingRate": None}, "X") is None
    assert funding_to_info({"fundingRate": 0.0002}, "X").interval_hours == 8


# ------------------------------------------------------------------ order params
def test_binance_stop_market():
    t, p = build_order_params("binance", R(OrderType.STOP_MARKET, stop_price=D("95.5"), reduce_only=True,
                                           client_order_id="c1"))
    assert t == "STOP_MARKET"
    assert p == {"reduceOnly": True, "newClientOrderId": "c1", "stopPrice": 95.5, "workingType": "MARK_PRICE"}


def test_binance_take_profit_and_binanceusdm_alias():
    t, p = build_order_params("binanceusdm", R(OrderType.TAKE_PROFIT_MARKET, stop_price=D("110")))
    assert t == "TAKE_PROFIT_MARKET" and p["stopPrice"] == 110.0 and p["workingType"] == "MARK_PRICE"


def test_binance_trailing_percent_and_activation():
    t, p = build_order_params("binance", R(OrderType.TRAILING_STOP_MARKET, trailing_callback_rate=D("0.005"),
                                           trailing_activation_price=D("101.5"), reduce_only=True))
    assert t == "TRAILING_STOP_MARKET"
    assert p["callbackRate"] == 0.5 and p["activationPrice"] == 101.5 and p["reduceOnly"] is True
    with pytest.raises(OrderRejectedError, match="callbackRate"):
        build_order_params("binance", R(OrderType.TRAILING_STOP_MARKET, trailing_callback_rate=D("0.06")))
    with pytest.raises(OrderRejectedError, match="callbackRate"):
        build_order_params("binance", R(OrderType.TRAILING_STOP_MARKET, trailing_callback_rate=D("0.0005")))


def test_binance_market_limit_and_post_only():
    assert build_order_params("binance", R(OrderType.MARKET, reduce_only=True)) == ("market", {"reduceOnly": True})
    t, p = build_order_params("binance", R(OrderType.LIMIT, price=D("100"), time_in_force=TimeInForce.POST_ONLY))
    assert t == "limit" and p == {"postOnly": True}
    t, p = build_order_params("binance", R(OrderType.LIMIT, price=D("100"), time_in_force=TimeInForce.IOC))
    assert p == {"timeInForce": "IOC"}


def test_binance_spot_has_no_stop_market():
    with pytest.raises(BrokerNotSupportedError):
        build_order_params("binance", R(OrderType.STOP_MARKET, stop_price=D("95")), MarketType.SPOT)


def test_bybit_unified_params():
    t, p = build_order_params("bybit", R(OrderType.STOP_MARKET, stop_price=D("95"), reduce_only=True,
                                         client_order_id="c2"))
    assert t == "market"
    assert p == {"reduceOnly": True, "clientOrderId": "c2", "positionIdx": 0, "stopLossPrice": 95.0}
    t, p = build_order_params("bybit", R(OrderType.TAKE_PROFIT_MARKET, stop_price=D("110")))
    assert t == "market" and p["takeProfitPrice"] == 110.0 and p["positionIdx"] == 0
    t, p = build_order_params("bybit", R(OrderType.MARKET), MarketType.SPOT)
    assert "positionIdx" not in p


def test_trailing_unsupported_exchanges():
    for ex in ("bybit", "okx", "kraken"):
        with pytest.raises(BrokerNotSupportedError):
            build_order_params(ex, R(OrderType.TRAILING_STOP_MARKET, trailing_callback_rate=D("0.01")))


def test_generic_fallback_params():
    t, p = build_order_params("okx", R(OrderType.STOP_MARKET, stop_price=D("95"), reduce_only=True, client_order_id="c"))
    assert t == "market" and p == {"reduceOnly": True, "clientOrderId": "c", "stopLossPrice": 95.0}
    t, p = build_order_params("okx", R(OrderType.TAKE_PROFIT_MARKET, stop_price=D("120")))
    assert p == {"takeProfitPrice": 120.0}
    t, p = build_order_params("okx", R(OrderType.LIMIT, price=D("100")))
    assert t == "limit" and p == {"timeInForce": "GTC"}


def test_malformed_requests_rejected():
    for ot in (OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET, OrderType.LIMIT, OrderType.TRAILING_STOP_MARKET):
        with pytest.raises(OrderRejectedError):
            build_order_params("binance", R(ot))


# ------------------------------------------------------------------ margin ratio
def test_margin_ratio_sources():
    assert compute_margin_ratio({"totalMaintMargin": "5", "totalMarginBalance": "100"}, [], D("100")) == D("0.05")
    bybit = {"result": {"list": [{"accountMMRate": "0.0123"}]}}
    assert compute_margin_ratio(bybit, [{"contracts": 1}], D("100")) == D("0.0123")
    positions = [{"contracts": 1, "maintenanceMargin": 2.0}, {"contracts": -2, "maintenanceMargin": 3.0}]
    assert compute_margin_ratio({}, positions, D("100")) == D("0.05")
    assert compute_margin_ratio(None, [{"contracts": 1}], D("100")) is None  # not reported
    assert compute_margin_ratio(None, [{"contracts": 0}], D("100")) == D("0")  # flat


# ------------------------------------------------------------------ exceptions
def test_exception_mapping():
    assert isinstance(map_ccxt_exception(ccxt.InsufficientFunds("x")), InsufficientFundsError)
    assert isinstance(map_ccxt_exception(ccxt.InvalidOrder("x")), OrderRejectedError)
    nf = map_ccxt_exception(ccxt.OrderNotFound("x"))
    assert type(nf) is BrokerError
    assert isinstance(map_ccxt_exception(ccxt.NotSupported("x")), BrokerNotSupportedError)
    assert type(map_ccxt_exception(ccxt.NetworkError("x"))) is BrokerError
    assert type(map_ccxt_exception(ccxt.ExchangeError("x"))) is BrokerError
    assert type(map_ccxt_exception(ccxt.AuthenticationError("x"))) is BrokerError


# ------------------------------------------------------------------ adapter construction (no network)
async def test_unknown_exchange_is_config_error():
    with pytest.raises(ConfigError):
        CCXTBrokerAdapter("nonexistent_ex", MarketType.SPOT, testnet=False)


async def test_equity_market_type_rejected():
    with pytest.raises(ConfigError):
        CCXTBrokerAdapter("bybit", MarketType.EQUITY, testnet=False)


async def test_repr_never_leaks_secrets_and_config_flags():
    a = CCXTBrokerAdapter("bybit", MarketType.PERPETUAL, "KEY123", "SECRET456", "PASS789", testnet=False)
    try:
        assert "KEY123" not in repr(a) and "SECRET456" not in repr(a) and "PASS789" not in repr(a)
        assert a.exchange.enableRateLimit is True
        assert a.exchange.options["defaultType"] == "swap"
        assert a.supports_shorting and a.supports_leverage
    finally:
        await a.close()
    spot = CCXTBrokerAdapter("bybit", MarketType.SPOT, testnet=False)
    try:
        assert spot.exchange.options["defaultType"] == "spot" and not spot.supports_shorting
    finally:
        await spot.close()


async def test_sandbox_not_supported_becomes_config_error(monkeypatch):
    def boom(self, enable):
        raise ccxt.NotSupported("no sandbox")

    monkeypatch.setattr(mod.ccxt_async.bybit, "set_sandbox_mode", boom)
    with pytest.raises(ConfigError, match="sandbox"):
        CCXTBrokerAdapter("bybit", MarketType.PERPETUAL, testnet=True)


async def test_testnet_enabled_when_supported():
    a = CCXTBrokerAdapter("bybit", MarketType.PERPETUAL, testnet=True)
    try:
        assert a.exchange.urls["api"] == a.exchange.urls["test"]
    finally:
        await a.close()


async def test_watch_requires_ccxt_pro(monkeypatch):
    monkeypatch.setattr(mod, "ccxt_pro", None)
    a = CCXTBrokerAdapter("bybit", MarketType.PERPETUAL, testnet=False)
    try:
        with pytest.raises(BrokerNotSupportedError):
            await a.watch_ticker("BTC/USDT:USDT")
        with pytest.raises(BrokerNotSupportedError):
            await a.watch_order_book("BTC/USDT:USDT", 5)
    finally:
        await a.close()
