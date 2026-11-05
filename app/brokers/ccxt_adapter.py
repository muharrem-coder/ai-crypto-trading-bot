"""CCXT-backed broker adapter (crypto spot + linear perpetuals).

Layout
------
* Module-level *pure* mapping functions (``market_to_spec``, ``ccxt_order_to_order``,
  ``ccxt_position_to_position``, ``ccxt_ticker_to_ticker``, ``funding_to_info``,
  ``build_order_params``, ``compute_margin_ratio``, ``map_ccxt_exception``) hold all the
  exchange-dialect knowledge and are unit-tested without network access.
* ``CCXTBrokerAdapter`` is thin glue: it calls ccxt, maps results, and wraps exceptions.

Units: ``Order.quantity`` / ``Position.quantity`` / ``InstrumentSpec.step_size`` / ``min_qty`` are in
*base-asset units*. ccxt reports contract-denominated amounts on some venues (``contractSize != 1``);
the mappers multiply by ``contractSize`` on the way in and the adapter divides on the way out.

Floats appear only at the ccxt call boundary (``_f``); everything coming back is converted with
``_d`` (``Decimal(str(x))``). Credentials are never logged or included in ``repr``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import ccxt
import ccxt.async_support as ccxt_async

from app.brokers.base import BaseBrokerAdapter
from app.core.errors import (
    BrokerError,
    BrokerNotSupportedError,
    ConfigError,
    InsufficientFundsError,
    OrderRejectedError,
)
from app.core.types import (
    AccountSnapshot,
    AssetClass,
    FeeSchedule,
    FundingInfo,
    InstrumentSpec,
    MarginMode,
    MarketType,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Side,
    Ticker,
    TimeInForce,
    utcnow,
)

try:  # ccxt.pro ships inside ccxt >= 4.? ; websockets are optional at runtime
    import ccxt.pro as ccxt_pro  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - depends on the environment
    ccxt_pro = None

log = logging.getLogger(__name__)

_DEFAULT_FEES = FeeSchedule(maker_rate=Decimal("0.0002"), taker_rate=Decimal("0.0005"))
_BINANCE_IDS = frozenset({"binance", "binanceusdm", "binancecoinm"})
_TRAILING_CAPABLE = _BINANCE_IDS


# --------------------------------------------------------------------------- small helpers
def _d(x: Any) -> Decimal | None:
    """Convert a ccxt number (float/str/int) to Decimal via ``str``; ``None`` stays ``None``."""
    if x is None or x == "":
        return None
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(str(x))
    except InvalidOperation:
        return None


def _dz(x: Any) -> Decimal:
    v = _d(x)
    return v if v is not None else Decimal("0")


def _pos_d(x: Any) -> Decimal | None:
    """Like ``_d`` but treats zero / negative as missing (exchanges send 0 for 'no price')."""
    v = _d(x)
    return v if v is not None and v > 0 else None


def _f(x: Decimal) -> float:
    """The single Decimal->float conversion, used only at the ccxt call boundary."""
    return float(x)


def _ts(ms: Any, default: datetime | None = None) -> datetime:
    if ms is None:
        return default or utcnow()
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


# --------------------------------------------------------------------------- exception mapping
def map_ccxt_exception(exc: BaseException) -> BrokerError:
    """Translate a ccxt exception to our hierarchy (message only; never request headers/keys)."""
    msg = f"{type(exc).__name__}: {str(exc)[:500]}"
    if isinstance(exc, ccxt.InsufficientFunds):
        return InsufficientFundsError(msg)
    if isinstance(exc, ccxt.OrderNotFound):
        return BrokerError(msg)
    if isinstance(exc, ccxt.InvalidOrder):
        return OrderRejectedError(msg)
    if isinstance(exc, ccxt.NotSupported):
        return BrokerNotSupportedError(msg)
    return BrokerError(msg)


# --------------------------------------------------------------------------- markets
def market_to_spec(market: dict[str, Any], precision_mode: int) -> InstrumentSpec:
    """Map one ccxt market dict to ``InstrumentSpec`` (quantities in base units).

    ``precision_mode`` is ``exchange.precisionMode``: ``ccxt.TICK_SIZE`` (precision values are
    increments) or ``ccxt.DECIMAL_PLACES`` (precision values are digit counts).
    """
    precision = market.get("precision") or {}
    limits = market.get("limits") or {}

    def increment(value: Any, name: str) -> Decimal:
        v = _d(value)
        if v is None:
            raise BrokerError(f"{market.get('symbol')}: missing {name} precision")
        if precision_mode == ccxt.TICK_SIZE:
            if v <= 0:
                raise BrokerError(f"{market.get('symbol')}: non-positive {name} tick {v}")
            return v
        if precision_mode == ccxt.DECIMAL_PLACES:
            return Decimal(1).scaleb(-int(v))
        raise BrokerNotSupportedError(f"Unsupported ccxt precisionMode {precision_mode}")

    contract_size = _pos_d(market.get("contractSize")) or Decimal("1")
    tick = increment(precision.get("price"), "price")
    step = increment(precision.get("amount"), "amount") * contract_size
    min_amount = _pos_d((limits.get("amount") or {}).get("min"))
    min_qty = (min_amount * contract_size) if min_amount is not None else step
    min_cost = _d((limits.get("cost") or {}).get("min")) or Decimal("0")
    max_lev_raw = _d((limits.get("leverage") or {}).get("max"))
    max_leverage = int(max_lev_raw) if max_lev_raw is not None and max_lev_raw >= 1 else 1

    maker = _d(market.get("maker"))
    taker = _d(market.get("taker"))
    fees = FeeSchedule(
        maker_rate=maker if maker is not None else _DEFAULT_FEES.maker_rate,
        taker_rate=taker if taker is not None else _DEFAULT_FEES.taker_rate,
    )

    is_swap = bool(market.get("swap")) or bool(market.get("future"))
    return InstrumentSpec(
        symbol=market["symbol"],
        asset_class=AssetClass.CRYPTO,
        market_type=MarketType.PERPETUAL if is_swap else MarketType.SPOT,
        base=market.get("base") or "",
        quote=market.get("quote") or "",
        settle=market.get("settle"),
        tick_size=tick,
        step_size=step,
        min_qty=min_qty,
        min_notional=min_cost,
        contract_size=contract_size,
        max_leverage=max_leverage,
        fees=fees,
    )


def market_is_wanted(market: dict[str, Any], market_type: MarketType) -> bool:
    """Filter: linear swaps for PERPETUAL, spot for SPOT; skip explicitly inactive markets."""
    if market.get("active") is False:
        return False
    if market_type is MarketType.PERPETUAL:
        return bool(market.get("swap")) and market.get("linear") is not False and bool(market.get("contract", True))
    return bool(market.get("spot"))


# --------------------------------------------------------------------------- orders
_STATUS_MAP = {
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.EXPIRED,
}


def _map_status(raw_status: Any, filled: Decimal) -> OrderStatus:
    s = (raw_status or "open").lower()
    if s == "open":
        return OrderStatus.PARTIALLY_FILLED if filled > 0 else OrderStatus.NEW
    return _STATUS_MAP.get(s, OrderStatus.NEW)


def _map_order_type(d: dict[str, Any]) -> OrderType:
    t = str(d.get("type") or "").lower().replace("-", "_").replace(" ", "_")
    has_sl = _pos_d(d.get("stopLossPrice")) is not None
    has_tp = _pos_d(d.get("takeProfitPrice")) is not None
    has_trigger = _pos_d(d.get("stopPrice") or d.get("triggerPrice")) is not None
    if "trailing" in t:
        return OrderType.TRAILING_STOP_MARKET
    if "limit" in t and ("stop" in t or "take_profit" in t):
        return OrderType.LIMIT  # stop-limit family: trigger recorded in stop_price
    if t.startswith("take_profit"):
        return OrderType.TAKE_PROFIT_MARKET
    if t.startswith("stop"):
        return OrderType.STOP_MARKET
    if t == "limit":
        return OrderType.LIMIT
    if t == "market":
        if has_tp:
            return OrderType.TAKE_PROFIT_MARKET
        if has_sl or has_trigger:
            return OrderType.STOP_MARKET
        return OrderType.MARKET
    return OrderType.LIMIT if _pos_d(d.get("price")) is not None else OrderType.MARKET


def ccxt_order_to_order(
    d: dict[str, Any],
    contract_size: Decimal = Decimal("1"),
    request: OrderRequest | None = None,
) -> Order:
    """Map a ccxt unified order dict to ``Order``; ``request`` fills fields the exchange omitted."""
    amount = _d(d.get("amount"))
    filled = _dz(d.get("filled")) * contract_size
    quantity = amount * contract_size if amount is not None else (request.quantity if request else Decimal("0"))
    status = _map_status(d.get("status"), filled)

    side_raw = d.get("side")
    if side_raw:
        side = Side(str(side_raw).upper())
    elif request is not None:
        side = request.side
    else:
        raise BrokerError("ccxt order without side")

    order_type = _map_order_type(d) if d.get("type") else (request.order_type if request else OrderType.MARKET)

    fee_paid = Decimal("0")
    fee_currency: str | None = None
    fees = d.get("fees") or ([d["fee"]] if d.get("fee") else [])
    for fee in fees:
        if not fee:
            continue
        fee_paid += _dz(fee.get("cost"))
        fee_currency = fee_currency or fee.get("currency")

    created = _ts(d.get("timestamp"))
    updated = _ts(d.get("lastUpdateTimestamp") or d.get("lastTradeTimestamp"), created)
    stop_price = _pos_d(
        d.get("stopPrice") or d.get("triggerPrice") or d.get("stopLossPrice") or d.get("takeProfitPrice")
    )
    reduce_only = d.get("reduceOnly")
    average = _pos_d(d.get("average"))

    return Order(
        order_id=str(d["id"]),
        client_order_id=d.get("clientOrderId") or (request.client_order_id if request else None),
        symbol=d.get("symbol") or (request.symbol if request else ""),
        side=side,
        order_type=order_type,
        status=status,
        quantity=quantity,
        filled_quantity=filled,
        price=_pos_d(d.get("price")),
        average_fill_price=average,
        stop_price=stop_price,
        reduce_only=bool(reduce_only) if reduce_only is not None else (request.reduce_only if request else False),
        fee_paid=fee_paid,
        fee_currency=fee_currency,
        created_at=created,
        updated_at=updated,
        raw=dict(d),
    )


def ccxt_position_to_position(
    p: dict[str, Any], contract_size: Decimal | None = None
) -> Position | None:
    """Map a ccxt position; returns ``None`` for flat (zero-size) entries."""
    cs = contract_size or _pos_d(p.get("contractSize")) or Decimal("1")
    contracts = _dz(p.get("contracts"))
    if contracts == 0:
        return None
    side_raw = str(p.get("side") or "").lower()
    if side_raw not in ("long", "short"):
        side_raw = "long" if contracts > 0 else "short"
    qty = abs(contracts) * cs
    entry = _dz(p.get("entryPrice"))
    mark = _pos_d(p.get("markPrice")) or entry
    leverage_raw = _d(p.get("leverage"))
    margin_mode = MarginMode.CROSS if str(p.get("marginMode") or "").lower() == "cross" else MarginMode.ISOLATED
    side = PositionSide.LONG if side_raw == "long" else PositionSide.SHORT
    unrealized = _d(p.get("unrealizedPnl"))
    if unrealized is None:
        direction = Decimal(1) if side is PositionSide.LONG else Decimal(-1)
        unrealized = (mark - entry) * qty * direction
    ts = p.get("timestamp")
    return Position(
        symbol=p["symbol"],
        side=side,
        quantity=qty,
        entry_price=entry,
        mark_price=mark,
        leverage=int(leverage_raw) if leverage_raw is not None and leverage_raw >= 1 else 1,
        margin_mode=margin_mode,
        unrealized_pnl=unrealized,
        liquidation_price=_pos_d(p.get("liquidationPrice")),
        initial_margin=_d(p.get("initialMargin")),
        opened_at=_ts(ts) if ts else None,
    )


def ccxt_ticker_to_ticker(d: dict[str, Any], symbol: str | None = None) -> Ticker:
    last = _pos_d(d.get("last")) or _pos_d(d.get("close"))
    bid = _pos_d(d.get("bid")) or last
    ask = _pos_d(d.get("ask")) or last
    if bid is None or ask is None or last is None:
        raise BrokerError(f"{symbol or d.get('symbol')}: ticker without usable prices")
    info = d.get("info") or {}
    mark = _pos_d(d.get("markPrice")) or (_pos_d(info.get("markPrice")) if isinstance(info, dict) else None)
    return Ticker(
        symbol=d.get("symbol") or symbol or "",
        bid=bid,
        ask=ask,
        last=last,
        mark=mark,
        timestamp=_ts(d.get("timestamp")),
    )


def funding_to_info(d: dict[str, Any], symbol: str) -> FundingInfo | None:
    rate = _d(d.get("fundingRate"))
    if rate is None:
        return None
    interval_hours = 8
    interval = d.get("interval")
    if isinstance(interval, str) and interval.endswith("h") and interval[:-1].isdigit():
        interval_hours = int(interval[:-1])
    nxt = d.get("nextFundingTimestamp") or d.get("fundingTimestamp")
    return FundingInfo(
        symbol=symbol,
        rate=rate,
        interval_hours=interval_hours,
        next_funding_time=_ts(nxt) if nxt else None,
    )


# --------------------------------------------------------------------------- order params
def build_order_params(
    exchange_id: str,
    req: OrderRequest,
    market_type: MarketType = MarketType.PERPETUAL,
) -> tuple[str, dict[str, Any]]:
    """Return ``(ccxt_order_type, params)`` for ``req`` on ``exchange_id``.

    Exchange-specific branches: binance family (native STOP_MARKET / TAKE_PROFIT_MARKET /
    TRAILING_STOP_MARKET) and bybit (unified stopLossPrice / takeProfitPrice, positionIdx 0).
    Everything else uses ccxt unified params. Raises ``BrokerNotSupportedError`` for order types the
    venue cannot take, ``OrderRejectedError`` for malformed requests.
    """
    ot = req.order_type
    params: dict[str, Any] = {}

    if ot is OrderType.LIMIT and req.price is None:
        raise OrderRejectedError(f"{req.symbol}: LIMIT order requires price")
    if ot in (OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET) and req.stop_price is None:
        raise OrderRejectedError(f"{req.symbol}: {ot.value} requires stop_price")
    if ot is OrderType.TRAILING_STOP_MARKET and req.trailing_callback_rate is None:
        raise OrderRejectedError(f"{req.symbol}: TRAILING_STOP_MARKET requires trailing_callback_rate")

    eid = exchange_id.lower()
    is_binance = eid in _BINANCE_IDS
    is_bybit = eid == "bybit"
    is_perp = market_type is MarketType.PERPETUAL

    if ot is OrderType.TRAILING_STOP_MARKET and (eid not in _TRAILING_CAPABLE or not is_perp):
        raise BrokerNotSupportedError(f"Trailing stop orders are not supported via create_order on {exchange_id}")
    if is_binance and not is_perp and ot in (
        OrderType.STOP_MARKET,
        OrderType.TAKE_PROFIT_MARKET,
    ):
        raise BrokerNotSupportedError("Binance spot has no *_MARKET stop order types; use perpetuals or LIMIT")

    if req.reduce_only:
        params["reduceOnly"] = True
    if req.client_order_id:
        params["newClientOrderId" if is_binance else "clientOrderId"] = req.client_order_id

    # time in force / post-only (only meaningful for LIMIT)
    if ot is OrderType.LIMIT:
        if req.time_in_force is TimeInForce.POST_ONLY:
            params["postOnly"] = True
        else:
            params["timeInForce"] = req.time_in_force.value

    if is_binance:
        if ot is OrderType.STOP_MARKET:
            ccxt_type = "STOP_MARKET"
            params.update(stopPrice=_f(req.stop_price), workingType="MARK_PRICE")  # type: ignore[arg-type]
        elif ot is OrderType.TAKE_PROFIT_MARKET:
            ccxt_type = "TAKE_PROFIT_MARKET"
            params.update(stopPrice=_f(req.stop_price), workingType="MARK_PRICE")  # type: ignore[arg-type]
        elif ot is OrderType.TRAILING_STOP_MARKET:
            ccxt_type = "TRAILING_STOP_MARKET"
            callback_pct = req.trailing_callback_rate * 100  # type: ignore[operator]
            if not (Decimal("0.1") <= callback_pct <= Decimal("5")):
                raise OrderRejectedError(
                    f"{req.symbol}: binance callbackRate must be within 0.1%..5%, got {callback_pct}%"
                )
            params.update(callbackRate=_f(callback_pct), workingType="MARK_PRICE")
            if req.trailing_activation_price is not None:
                params["activationPrice"] = _f(req.trailing_activation_price)
        else:
            ccxt_type = "limit" if ot is OrderType.LIMIT else "market"
        return ccxt_type, params

    if is_bybit:
        if is_perp:
            params["positionIdx"] = 0  # one-way mode
        if ot is OrderType.STOP_MARKET:
            params["stopLossPrice"] = _f(req.stop_price)  # type: ignore[arg-type]
            return "market", params
        if ot is OrderType.TAKE_PROFIT_MARKET:
            params["takeProfitPrice"] = _f(req.stop_price)  # type: ignore[arg-type]
            return "market", params
        return ("limit" if ot is OrderType.LIMIT else "market"), params

    # generic fallback on ccxt unified params
    if ot is OrderType.STOP_MARKET:
        params["stopLossPrice"] = _f(req.stop_price)  # type: ignore[arg-type]
        return "market", params
    if ot is OrderType.TAKE_PROFIT_MARKET:
        params["takeProfitPrice"] = _f(req.stop_price)  # type: ignore[arg-type]
        return "market", params
    return ("limit" if ot is OrderType.LIMIT else "market"), params


# --------------------------------------------------------------------------- account helpers
def compute_margin_ratio(
    balance_info: Any,
    raw_positions: list[dict[str, Any]],
    equity: Decimal,
) -> Decimal | None:
    """Maintenance-margin ratio. Order of preference:

    1. exchange-reported figure in the balance ``info`` (binance totalMaintMargin/totalMarginBalance,
       bybit accountMMRate);
    2. sum(position.maintenanceMargin) / equity when positions report it;
    3. ``0`` when there are no open positions; otherwise ``None``.
    """
    if isinstance(balance_info, dict):
        mm = _d(balance_info.get("totalMaintMargin"))
        mb = _d(balance_info.get("totalMarginBalance"))
        if mm is not None and mb is not None and mb > 0:
            return mm / mb
        try:
            rate = _d(balance_info["result"]["list"][0].get("accountMMRate"))
        except (KeyError, IndexError, TypeError, AttributeError):
            rate = None
        if rate is not None:
            return rate
    open_positions = [p for p in raw_positions if _dz(p.get("contracts")) != 0]
    if not open_positions:
        return Decimal("0")
    maint = [_d(p.get("maintenanceMargin")) for p in open_positions]
    if all(m is not None for m in maint) and equity > 0:
        return sum(maint, Decimal("0")) / equity  # type: ignore[arg-type]
    return None


_NOOP_MARGIN_ERRORS = ("no need", "-4046", "not modified", "not changed")


def _is_noop_margin_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _NOOP_MARGIN_ERRORS)


# --------------------------------------------------------------------------- adapter
class CCXTBrokerAdapter(BaseBrokerAdapter):
    """Spot / linear-perpetual trading over ccxt's async REST client."""

    asset_class = AssetClass.CRYPTO

    def __init__(
        self,
        exchange_id: str,
        market_type: MarketType,
        api_key: str = "",
        api_secret: str = "",
        password: str = "",
        testnet: bool = True,
        settle_currency: str = "USDT",
    ) -> None:
        super().__init__()
        if market_type not in (MarketType.SPOT, MarketType.PERPETUAL):
            raise ConfigError(f"CCXT adapter supports SPOT and PERPETUAL only, got {market_type}")
        self.exchange_id = exchange_id
        self.market_type = market_type
        self.testnet = testnet
        self.settle_currency = settle_currency
        self.supports_shorting = market_type is MarketType.PERPETUAL
        self.supports_leverage = market_type is MarketType.PERPETUAL
        self._leverage: dict[str, int] = {}
        self._pro: Any = None
        self._config = self._build_config(api_key, api_secret, password)
        self._exchange = self._make_exchange(ccxt_async)

    # never expose credentials
    def __repr__(self) -> str:
        return (
            f"CCXTBrokerAdapter(exchange_id={self.exchange_id!r}, market_type={self.market_type.value}, "
            f"testnet={self.testnet})"
        )

    # ------------------------------------------------------------------ construction
    def _build_config(self, api_key: str, api_secret: str, password: str) -> dict[str, Any]:
        options: dict[str, Any] = {"defaultType": "swap" if self.market_type is MarketType.PERPETUAL else "spot"}
        if self.market_type is MarketType.PERPETUAL:
            options["defaultSubType"] = "linear"
        config: dict[str, Any] = {"enableRateLimit": True, "options": options}
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret
        if password:
            config["password"] = password
        return config

    def _make_exchange(self, module: Any) -> Any:
        cls = getattr(module, self.exchange_id, None)
        if cls is None:
            raise ConfigError(f"Unknown or unsupported exchange id: {self.exchange_id!r}")
        exchange = cls(dict(self._config, options=dict(self._config["options"])))
        if self.testnet:
            try:
                exchange.set_sandbox_mode(True)
            except ccxt.NotSupported as exc:
                raise ConfigError(
                    f"{self.exchange_id} does not provide a sandbox/testnet for {self.market_type.value} "
                    f"via ccxt ({exc}). Use paper trading (TRADING_MODE=PAPER) or set EXCHANGE_TESTNET=false."
                ) from exc
        return exchange

    def _get_pro(self) -> Any:
        if ccxt_pro is None:
            raise BrokerNotSupportedError("ccxt.pro (websocket streaming) is not available in this environment")
        if self._pro is None:
            self._pro = self._make_exchange(ccxt_pro)
        return self._pro

    @property
    def exchange(self) -> Any:
        return self._exchange

    # ------------------------------------------------------------------ call wrapper
    async def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except BrokerError:
            raise
        except (ccxt.BaseError, OSError) as exc:  # ccxt.NetworkError is a BaseError subclass
            raise map_ccxt_exception(exc) from exc

    def _contract_size(self, symbol: str) -> Decimal:
        spec = self._instruments.get(symbol)
        return spec.contract_size if spec else Decimal("1")

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        await self.load_instruments()
        log.info("connected exchange=%s market=%s testnet=%s instruments=%d",
                 self.exchange_id, self.market_type.value, self.testnet, len(self._instruments))

    async def close(self) -> None:
        for ex in (self._exchange, self._pro):
            if ex is not None:
                try:
                    await ex.close()
                except Exception:  # noqa: BLE001 - closing must never raise
                    log.debug("error closing exchange client", exc_info=True)

    # ------------------------------------------------------------------ metadata
    async def load_instruments(self) -> dict[str, InstrumentSpec]:
        markets = await self._call(self._exchange.load_markets)
        specs: dict[str, InstrumentSpec] = {}
        for market in markets.values():
            if not market_is_wanted(market, self.market_type):
                continue
            try:
                spec = market_to_spec(market, self._exchange.precisionMode)
            except (BrokerError, KeyError, ValueError, InvalidOperation):
                log.debug("skipping unmappable market %s", market.get("symbol"))
                continue
            specs[spec.symbol] = spec
        self._instruments = specs
        return dict(specs)

    async def fetch_ticker(self, symbol: str) -> Ticker:
        raw = await self._call(self._exchange.fetch_ticker, symbol)
        return ccxt_ticker_to_ticker(raw, symbol)

    async def fetch_funding(self, symbol: str) -> FundingInfo | None:
        if self.market_type is not MarketType.PERPETUAL or not self._exchange.has.get("fetchFundingRate"):
            return None
        raw = await self._call(self._exchange.fetch_funding_rate, symbol)
        return funding_to_info(raw, symbol)

    # ------------------------------------------------------------------ account
    async def _fetch_raw_positions(self) -> list[dict[str, Any]]:
        if self.market_type is not MarketType.PERPETUAL:
            return []
        return list(await self._call(self._exchange.fetch_positions))

    def _map_positions(self, raw_positions: list[dict[str, Any]]) -> list[Position]:
        result = []
        for raw in raw_positions:
            spec = self._instruments.get(raw.get("symbol", ""))
            pos = ccxt_position_to_position(raw, spec.contract_size if spec else None)
            if pos is not None:
                result.append(pos)
        return result

    async def fetch_positions(self) -> list[Position]:
        return self._map_positions(await self._fetch_raw_positions())

    async def fetch_account(self) -> AccountSnapshot:
        balance = await self._call(self._exchange.fetch_balance)
        currency = self.settle_currency
        entry = balance.get(currency) or {}
        wallet = _dz(entry.get("total"))
        free = _dz(entry.get("free"))
        now = utcnow()
        if self.market_type is MarketType.SPOT:
            return AccountSnapshot(
                equity=wallet,
                wallet_balance=wallet,
                available_balance=free,
                unrealized_pnl=Decimal("0"),
                margin_ratio=None,
                positions=(),
                timestamp=now,
            )
        raw_positions = await self._fetch_raw_positions()
        positions = self._map_positions(raw_positions)
        unrealized = sum((p.unrealized_pnl for p in positions), Decimal("0"))
        equity = wallet + unrealized
        return AccountSnapshot(
            equity=equity,
            wallet_balance=wallet,
            available_balance=free,
            unrealized_pnl=unrealized,
            margin_ratio=compute_margin_ratio(balance.get("info"), raw_positions, equity),
            positions=tuple(positions),
            timestamp=now,
        )

    async def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        raws = list(await self._call(self._exchange.fetch_open_orders, symbol))
        if self.exchange_id in _BINANCE_IDS and self.market_type is MarketType.PERPETUAL:
            try:  # conditional (algo) orders live behind a separate endpoint
                raws += list(await self._call(self._exchange.fetch_open_orders, symbol, None, None, {"trigger": True}))
            except BrokerError:
                log.debug("conditional open-order fetch failed", exc_info=True)
        seen: set[str] = set()
        orders = []
        for raw in raws:
            if str(raw.get("id")) in seen:
                continue
            seen.add(str(raw.get("id")))
            orders.append(ccxt_order_to_order(raw, self._contract_size(raw.get("symbol", ""))))
        return orders

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        raw = await self._call(self._exchange.fetch_order, order_id, symbol)
        return ccxt_order_to_order(raw, self._contract_size(symbol))

    # ------------------------------------------------------------------ trading
    async def create_order(self, req: OrderRequest) -> Order:
        spec = await self.get_instrument(req.symbol)
        self.pre_trade_regulatory_check(req)
        self.validate_order_against_spec(spec, req)
        build_order_params(self.exchange_id, req, self.market_type)  # fail fast before touching leverage

        if req.leverage is not None:
            await self.set_leverage(req.symbol, req.leverage)
        if req.margin_mode is not None and self.market_type is MarketType.PERPETUAL:
            await self.set_margin_mode(req.symbol, req.margin_mode)

        price = self.round_price(spec, req.price) if req.price is not None else None
        # build params from tick-rounded prices so exchange tick rules are respected
        rounded = replace(
            req,
            price=price,
            stop_price=self.round_price(spec, req.stop_price) if req.stop_price is not None else None,
            trailing_activation_price=(
                self.round_price(spec, req.trailing_activation_price)
                if req.trailing_activation_price is not None
                else None
            ),
        )
        ccxt_type, params = build_order_params(self.exchange_id, rounded, self.market_type)
        amount = _f(req.quantity / spec.contract_size)
        raw = await self._call(
            self._exchange.create_order,
            req.symbol,
            ccxt_type,
            req.side.value.lower(),
            amount,
            _f(price) if price is not None else None,
            params,
        )
        order = ccxt_order_to_order(raw, spec.contract_size, request=req)
        if req.order_type is OrderType.MARKET and not order.status.is_terminal:
            try:  # market orders usually fill instantly; the create response is often a bare ack
                order = await self.fetch_order(order.order_id, req.symbol)
            except BrokerError:
                log.debug("post-create fetch_order failed for %s", order.order_id, exc_info=True)
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        try:
            raw = await self._call(self._exchange.cancel_order, order_id, symbol)
        except BrokerError:
            if self.exchange_id not in _BINANCE_IDS:
                raise
            raw = await self._call(self._exchange.cancel_order, order_id, symbol, {"trigger": True})
        raw = dict(raw or {})
        raw.setdefault("id", order_id)
        raw.setdefault("symbol", symbol)
        order = ccxt_order_to_order(raw, self._contract_size(symbol), request=None) if raw.get("side") else None
        if order is None:
            now = utcnow()
            return Order(
                order_id=str(order_id), client_order_id=raw.get("clientOrderId"), symbol=symbol, side=Side.BUY,
                order_type=OrderType.LIMIT, status=OrderStatus.CANCELED, quantity=Decimal("0"),
                filled_quantity=Decimal("0"), price=None, average_fill_price=None, stop_price=None,
                reduce_only=False, fee_paid=Decimal("0"), fee_currency=None, created_at=now, updated_at=now,
                raw=raw,
            )
        if not order.status.is_terminal:
            order.status = OrderStatus.CANCELED
        return order

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        open_orders = await self.fetch_open_orders(symbol)
        for sym in sorted({o.symbol for o in open_orders}):
            await self._call(self._exchange.cancel_all_orders, sym)
            if self.exchange_id in _BINANCE_IDS and self.market_type is MarketType.PERPETUAL:
                try:
                    await self._call(self._exchange.cancel_all_orders, sym, {"trigger": True})
                except BrokerError:
                    log.debug("conditional cancel-all failed for %s", sym, exc_info=True)
        return len(open_orders)

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        if self.market_type is not MarketType.PERPETUAL:
            if leverage == 1:
                return
            raise BrokerNotSupportedError("Leverage is only available on perpetual markets")
        spec = await self.get_instrument(symbol)
        if leverage < 1 or leverage > spec.max_leverage:
            raise OrderRejectedError(f"{symbol}: leverage {leverage} outside 1..{spec.max_leverage}")
        try:
            await self._call(self._exchange.set_leverage, leverage, symbol)
        except BrokerError as exc:
            if not _is_noop_margin_error(exc):
                raise
        self._leverage[symbol] = leverage

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        if self.market_type is not MarketType.PERPETUAL:
            raise BrokerNotSupportedError("Margin mode is only available on perpetual markets")
        params: dict[str, Any] = {}
        if self.exchange_id == "bybit":  # bybit requires leverage alongside the margin-mode switch
            params["leverage"] = self._leverage.get(symbol, 1)
        try:
            await self._call(self._exchange.set_margin_mode, mode.value.lower(), symbol, params)
        except BrokerError as exc:
            if not _is_noop_margin_error(exc):
                raise

    async def close_position(self, symbol: str) -> Order | None:
        position = next((p for p in await self.fetch_positions() if p.symbol == symbol), None)
        if position is None:
            return None
        spec = await self.get_instrument(symbol)
        qty = self.round_qty_down(spec, position.quantity)
        if qty <= 0:
            raise BrokerError(f"{symbol}: position size {position.quantity} is below the step size")
        return await self.create_order(
            OrderRequest(
                symbol=symbol,
                side=position.side.exit_side,
                order_type=OrderType.MARKET,
                quantity=qty,
                reduce_only=True,
                metadata={"reason": "close_position"},
            )
        )

    async def close_all_positions(self) -> list[Order]:
        orders: list[Order] = []
        for position in await self.fetch_positions():
            order = await self.close_position(position.symbol)
            if order is not None:
                orders.append(order)
        return orders

    # ------------------------------------------------------------------ streaming (Phase 2 data engine)
    async def watch_ticker(self, symbol: str) -> Ticker:
        raw = await self._call(self._get_pro().watch_ticker, symbol)
        return ccxt_ticker_to_ticker(raw, symbol)

    async def watch_order_book(self, symbol: str, limit: int | None = None) -> dict[str, Any]:
        """Return the raw ccxt order book dict (``bids`` / ``asks`` as ``[price, size]`` lists)."""
        return await self._call(self._get_pro().watch_order_book, symbol, limit)
