"""Rounding / validation helpers and default hooks on BaseBrokerAdapter (no network)."""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from app.brokers.base import BaseBrokerAdapter
from app.brokers.paper_adapter import PaperBrokerAdapter
from app.core.errors import BrokerError, OrderRejectedError
from app.core.types import (
    AssetClass,
    FeeSchedule,
    InstrumentSpec,
    MarketType,
    OrderRequest,
    OrderType,
    Side,
)


def make_spec(symbol: str = "BTC/USDT:USDT", **kw) -> InstrumentSpec:
    base = dict(
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.PERPETUAL,
        base="BTC",
        quote="USDT",
        settle="USDT",
        tick_size=D("0.1"),
        step_size=D("0.001"),
        min_qty=D("0.001"),
        min_notional=D("5"),
        max_leverage=20,
        fees=FeeSchedule(maker_rate=D("0.0002"), taker_rate=D("0.0005")),
    )
    base.update(kw)
    return InstrumentSpec(**base)


def req(qty: str, price: str | None = None, **kw) -> OrderRequest:
    return OrderRequest(
        symbol="BTC/USDT:USDT",
        side=kw.pop("side", Side.BUY),
        order_type=OrderType.LIMIT if price else OrderType.MARKET,
        quantity=D(qty),
        price=D(price) if price else None,
        **kw,
    )


class TestRounding:
    spec = make_spec()

    def test_round_price_half_up(self):
        assert BaseBrokerAdapter.round_price(self.spec, D("100.04")) == D("100.0")
        assert BaseBrokerAdapter.round_price(self.spec, D("100.05")) == D("100.1")
        assert BaseBrokerAdapter.round_price(self.spec, D("100.06")) == D("100.1")

    def test_round_price_down_and_up(self):
        assert BaseBrokerAdapter.round_price_down(self.spec, D("100.09")) == D("100.0")
        assert BaseBrokerAdapter.round_price_up(self.spec, D("100.01")) == D("100.1")
        assert BaseBrokerAdapter.round_price_up(self.spec, D("100.1")) == D("100.1")

    def test_round_qty_down(self):
        assert BaseBrokerAdapter.round_qty_down(self.spec, D("0.12399")) == D("0.123")
        assert BaseBrokerAdapter.round_qty_down(self.spec, D("0.0009")) == D("0")

    def test_coarse_tick(self):
        spec = make_spec(tick_size=D("5"))
        assert BaseBrokerAdapter.round_price(spec, D("12.5")) == D("15")
        assert BaseBrokerAdapter.round_price_down(spec, D("14.99")) == D("10")


class TestValidation:
    spec = make_spec()

    def test_ok(self):
        BaseBrokerAdapter.validate_order_against_spec(self.spec, req("0.1", "100"))

    def test_below_min_qty(self):
        with pytest.raises(OrderRejectedError, match="min_qty"):
            BaseBrokerAdapter.validate_order_against_spec(make_spec(min_qty=D("0.01")), req("0.005", "100000"))

    def test_zero_or_negative_qty(self):
        with pytest.raises(OrderRejectedError):
            BaseBrokerAdapter.validate_order_against_spec(self.spec, req("0", "100"))

    def test_misaligned_step(self):
        with pytest.raises(OrderRejectedError, match="step_size"):
            BaseBrokerAdapter.validate_order_against_spec(self.spec, req("0.0015", "100"))

    def test_below_min_notional(self):
        with pytest.raises(OrderRejectedError, match="min_notional"):
            BaseBrokerAdapter.validate_order_against_spec(self.spec, req("0.01", "100"))  # 1 < 5

    def test_notional_uses_ref_price_for_market(self):
        with pytest.raises(OrderRejectedError, match="min_notional"):
            BaseBrokerAdapter.validate_order_against_spec(self.spec, req("0.01"), ref_price=D("100"))
        BaseBrokerAdapter.validate_order_against_spec(self.spec, req("0.01"))  # no price known: skipped

    def test_reduce_only_exempt_from_min_notional(self):
        BaseBrokerAdapter.validate_order_against_spec(
            self.spec, req("0.01", reduce_only=True), ref_price=D("100")
        )


class TestDefaultsAndCache:
    async def test_get_instrument_cache_and_unknown(self):
        spec = make_spec()
        broker = PaperBrokerAdapter({spec.symbol: spec}, D("1000"), "USDT", D("0"))
        assert await broker.get_instrument(spec.symbol) is spec
        with pytest.raises(BrokerError, match="Unknown instrument"):
            await broker.get_instrument("DOGE/USDT:USDT")

    def test_crypto_hooks_are_permissive(self):
        broker = PaperBrokerAdapter({}, D("1000"), "USDT", D("0"))
        assert broker.asset_class is AssetClass.CRYPTO
        assert broker.is_market_open() is True
        status = broker.market_hours_status()
        assert status.is_open and status.session == "24x7"
        assert broker.pre_trade_regulatory_check(req("1")) is None
