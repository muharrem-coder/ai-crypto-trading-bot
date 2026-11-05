"""Hard-Risk Engine. The LLM proposes; this engine disposes.

Nothing in a TradeProposal (including confidence, source, or any free text) can raise a limit: every limit is read
from RiskSettings, and proposals can only be approved, reduced, or rejected. All money math is Decimal.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.config.settings import RiskSettings
from app.core.errors import RiskViolation
from app.core.types import (
    AccountSnapshot,
    FundingInfo,
    InstrumentSpec,
    LiquidityRole,
    MarginMode,
    MarketType,
    OrderRequest,
    OrderType,
    Position,
    PositionSide,
    FeeSchedule,
    Ticker,
)
from app.risk.fee_calculator import FeeCalculator, TradeEconomics
from app.risk.kill_switch import KillSwitch
from app.risk.position_sizer import SizingResult, size_position
from app.risk.protocols import RiskEventSink
from app.risk.rounding import ceil_to_step, floor_to_step

ZERO = Decimal(0)
ONE = Decimal(1)
ABSOLUTE_MAX_LEVERAGE = 5  # hard ceiling regardless of settings / exchange
DEFAULT_MMR = Decimal("0.005")
DEFAULT_LIQ_BUFFER_PCT = Decimal("0.005")  # liquidation must sit at least this fraction of entry beyond the stop
DEFAULT_FUNDING_INTERVAL_H = 8
MARGIN_USAGE_CAP = Decimal("0.95")  # never plan to use more than 95% of available balance


class Verdict(str, enum.Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REDUCED = "REDUCED"


@dataclass(slots=True)
class TradeProposal:
    symbol: str
    side: PositionSide
    entry_price: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None
    leverage: int
    confidence: Decimal
    quantity: Decimal | None = None
    source: str = "strategy"  # 'claude' | 'strategy' | 'manual'
    decision_id: str | None = None
    margin_mode: MarginMode = MarginMode.ISOLATED
    entry_order_type: OrderType = OrderType.MARKET
    expected_holding_hours: Decimal = Decimal(4)


@dataclass(slots=True)
class RiskDecision:
    verdict: Verdict
    approved_quantity: Decimal
    approved_leverage: int
    stop_loss: Decimal | None
    take_profit: Decimal | None
    economics: TradeEconomics | None = None
    sizing: SizingResult | None = None
    reasons: list[str] = field(default_factory=list)  # "CODE: message"
    checks: dict[str, bool] = field(default_factory=dict)
    decision_id: str | None = None

    @property
    def approved(self) -> bool:
        return self.verdict is not Verdict.REJECTED

    @property
    def reason_codes(self) -> list[str]:
        return [r.split(":", 1)[0] for r in self.reasons]


def _liq_price(side: PositionSide, entry: Decimal, leverage: int, mmr: Decimal) -> Decimal:
    """Approximate isolated-margin liquidation price."""
    inv = ONE / Decimal(leverage)
    if side is PositionSide.LONG:
        return entry * (ONE - inv + mmr)
    return entry * (ONE + inv - mmr)


def _liq_safe(side: PositionSide, entry: Decimal, stop: Decimal, leverage: int, mmr: Decimal, buffer_pct: Decimal) -> bool:
    liq = _liq_price(side, entry, leverage, mmr)
    buf = entry * buffer_pct
    if side is PositionSide.LONG:
        return liq <= stop - buf
    return liq >= stop + buf


def _money(d: Decimal) -> str:
    return format(d, ",.2f")


class HardRiskEngine:
    def __init__(
        self,
        settings: RiskSettings,
        kill_switch: KillSwitch,
        sink: RiskEventSink,
        *,
        min_confidence: Decimal = Decimal("0.55"),
        maintenance_margin_rate: Decimal = DEFAULT_MMR,
        liq_buffer_pct: Decimal = DEFAULT_LIQ_BUFFER_PCT,
        max_entry_deviation_pct: Decimal = Decimal("0.01"),
    ) -> None:
        self._s = settings
        self._ks = kill_switch
        self._sink = sink
        self._min_conf = min_confidence
        self._mmr = maintenance_margin_rate
        self._liq_buffer = liq_buffer_pct
        self._max_dev = max_entry_deviation_pct

    # ------------------------------------------------------------------ validate
    async def validate(
        self,
        proposal: TradeProposal,
        snapshot: AccountSnapshot,
        spec: InstrumentSpec,
        funding: FundingInfo | None = None,
        ticker: Ticker | None = None,
    ) -> RiskDecision:
        s = self._s
        reasons: list[str] = []
        checks: dict[str, bool] = {}
        reduced = False
        fail_codes: list[str] = []

        def fail(code: str, msg: str, check: str | None = None) -> None:
            fail_codes.append(code)
            reasons.append(f"{code}: {msg}")
            checks[check or code.lower()] = False

        def ok(check: str) -> None:
            checks[check] = True

        def note(code: str, msg: str) -> None:
            reasons.append(f"{code}: {msg}")

        p = proposal
        entry, stop, tp = p.entry_price, p.stop_loss, p.take_profit
        is_deriv = spec.market_type is MarketType.PERPETUAL
        equity = snapshot.equity

        # 1. kill switch
        if self._ks.is_engaged:
            fail("KILL_SWITCH", f"Kill switch devrede / kill switch engaged: {self._ks.reason}", "kill_switch_clear")
        else:
            ok("kill_switch_clear")

        # 2. symbol
        blocked = {b.upper() for b in s.blocked_symbols}
        if p.symbol != spec.symbol:
            fail("SYMBOL_MISMATCH", f"Teklif sembolü {p.symbol} != enstrüman {spec.symbol}", "symbol_allowed")
        elif p.symbol.upper() in blocked:
            fail("SYMBOL_BLOCKED", f"{p.symbol} engelli semboller listesinde / symbol is blocked", "symbol_allowed")
        else:
            ok("symbol_allowed")

        # 3. market supports side
        if p.side is PositionSide.SHORT and not is_deriv:
            fail("SHORT_NOT_SUPPORTED", f"{spec.market_type.value} piyasasında SHORT açılamaz / cannot short", "side_supported")
        else:
            ok("side_supported")

        # 4. margin mode
        if is_deriv and p.margin_mode is not MarginMode.ISOLATED:
            fail("MARGIN_MODE", "Türev işlemlerde yalnızca ISOLATED marj izinli / only ISOLATED margin allowed", "margin_mode_isolated")
        else:
            ok("margin_mode_isolated")

        # 5. prices / stop-loss presence and side
        prices_valid = entry is not None and entry > 0
        if not prices_valid:
            fail("ENTRY_INVALID", "Giriş fiyatı geçersiz / invalid entry price", "entry_valid")
        else:
            ok("entry_valid")
        stop_valid = False
        if stop is None or stop <= 0:
            fail("STOP_LOSS_MISSING", "Zorunlu stop-loss yok / mandatory stop-loss missing", "stop_loss_present")
        elif prices_valid:
            ok("stop_loss_present")
            right = stop < entry if p.side is PositionSide.LONG else stop > entry
            if right:
                stop_valid = True
                ok("stop_side_correct")
            else:
                fail("STOP_WRONG_SIDE", f"Stop ({stop}) giriş ({entry}) için yanlış tarafta / stop on wrong side of entry", "stop_side_correct")
        tp_valid = False
        if tp is None or tp <= 0:
            fail("TAKE_PROFIT_MISSING", "Take-profit yok / take-profit missing", "take_profit_valid")
        elif prices_valid:
            right = tp > entry if p.side is PositionSide.LONG else tp < entry
            if right:
                tp_valid = True
                ok("take_profit_valid")
            else:
                fail("TAKE_PROFIT_WRONG_SIDE", f"Take-profit ({tp}) yanlış tarafta / take-profit on wrong side of entry", "take_profit_valid")

        if prices_valid and ticker is not None and p.entry_order_type is OrderType.MARKET and ticker.last > 0:
            dev = abs(ticker.last - entry) / entry
            if dev > self._max_dev:
                fail("ENTRY_STALE", f"Giriş fiyatı piyasadan %{dev * 100:.2f} sapıyor / entry deviates from market", "entry_near_market")
            else:
                ok("entry_near_market")

        # 6. stop distance
        if stop_valid:
            dist_pct = abs(entry - stop) / entry
            if dist_pct < s.min_stop_distance_pct:
                fail("STOP_TOO_TIGHT", f"Stop mesafesi %{dist_pct * 100:.3f} < min %{s.min_stop_distance_pct * 100:.3f}", "stop_distance_ok")
            elif dist_pct > s.max_stop_distance_pct:
                fail("STOP_TOO_WIDE", f"Stop mesafesi %{dist_pct * 100:.3f} > maks %{s.max_stop_distance_pct * 100:.3f}", "stop_distance_ok")
            else:
                ok("stop_distance_ok")

        # 7. leverage clamp (never a rejection reason by itself)
        if is_deriv:
            lev_cap = min(s.max_leverage, spec.max_leverage, ABSOLUTE_MAX_LEVERAGE)
        else:
            lev_cap = 1
        lev_cap = max(1, lev_cap)
        lev = max(1, int(p.leverage))
        if lev > lev_cap:
            note("LEVERAGE_CLAMPED", f"İstenen kaldıraç {lev}x -> {lev_cap}x / requested leverage clamped")
            checks["leverage_within_limit"] = False
            lev = lev_cap
            reduced = True
        else:
            checks["leverage_within_limit"] = True

        # 8. liquidation beyond stop
        if is_deriv and stop_valid:
            start_lev = lev
            while lev > 1 and not _liq_safe(p.side, entry, stop, lev, self._mmr, self._liq_buffer):
                lev -= 1
            if _liq_safe(p.side, entry, stop, lev, self._mmr, self._liq_buffer):
                ok("liquidation_beyond_stop")
                if lev < start_lev:
                    note("LEVERAGE_LIQ_REDUCED", f"Likidasyon stop'tan önce olurdu; kaldıraç {start_lev}x -> {lev}x")
                    reduced = True
            else:
                fail("LIQ_BEFORE_STOP", "1x kaldıraçta bile likidasyon stop'tan önce / liquidation before stop even at 1x", "liquidation_beyond_stop")

        # 9/10. positions
        existing = [pos for pos in snapshot.positions if pos.symbol == p.symbol]
        if existing:
            if any(pos.side is not p.side for pos in existing):
                fail("OPPOSITE_POSITION", f"{p.symbol} için ters yönde açık pozisyon var / opposite position open", "no_conflicting_position")
            else:
                fail("PYRAMIDING", f"{p.symbol} için aynı yönde pozisyon var; ekleme yapılamaz / no pyramiding", "no_conflicting_position")
        else:
            ok("no_conflicting_position")
            if len(snapshot.positions) >= s.max_open_positions:
                fail("MAX_POSITIONS", f"Açık pozisyon sınırı doldu ({len(snapshot.positions)}/{s.max_open_positions})", "open_positions_ok")
            else:
                ok("open_positions_ok")

        def too_small(q: Decimal) -> bool:
            return q <= 0 or q < spec.min_qty or q * entry < spec.min_notional

        # 11. sizing
        qty = ZERO
        sizing: SizingResult | None = None
        if stop_valid and prices_valid and equity > 0:
            sizing = size_position(equity, entry, stop, spec, s, lev, spec.fees)
            if sizing.quantity <= 0:
                fail("SIZE_TOO_SMALL", f"Hesaplanan miktar sıfır ({sizing.reason}) / sized quantity is zero", "sizing_ok")
            else:
                qty = sizing.quantity
                if p.quantity is not None:
                    if p.quantity <= 0:
                        fail("QTY_INVALID", "İstenen miktar <= 0 / requested quantity must be positive", "sizing_ok")
                        qty = ZERO
                    else:
                        wanted = floor_to_step(p.quantity, spec.step_size)
                        if wanted > qty:
                            note("SIZE_REDUCED", f"Miktar {p.quantity} -> {qty} (risk limiti) / quantity capped by risk sizing")
                            reduced = True
                        else:
                            qty = wanted  # never larger than what the risk model allows
                        if too_small(qty):
                            fail("BELOW_MINIMUMS", f"İstenen miktar {p.quantity} borsa minimumlarının altında (min_qty={spec.min_qty}, min_notional={spec.min_notional})", "min_size_ok")
                            qty = ZERO
                        if qty > 0:
                            ok("sizing_ok")
                else:
                    ok("sizing_ok")
        elif equity <= 0:
            fail("NO_EQUITY", "Özkaynak <= 0 / no equity", "sizing_ok")

        # 12. total exposure
        if qty > 0:
            existing_notional = sum((pos.notional for pos in snapshot.positions), ZERO)
            room = s.max_total_exposure_pct * equity - existing_notional
            if qty * entry > room:
                new_qty = floor_to_step(max(room, ZERO) / entry, spec.step_size)
                if too_small(new_qty):
                    fail("EXPOSURE_CAP", "Toplam maruziyet sınırı aşılıyor / total exposure cap exceeded", "exposure_ok")
                    qty = ZERO
                else:
                    note("EXPOSURE_REDUCED", f"Toplam maruziyet sınırı: miktar {qty} -> {new_qty}")
                    qty = new_qty
                    reduced = True
                    ok("exposure_ok")
            else:
                ok("exposure_ok")

        # 13. available margin
        if qty > 0:
            fee_buf = spec.fees.taker_rate + s.default_slippage_bps / Decimal(10000)
            per_unit_margin = entry / Decimal(lev) + entry * fee_buf
            affordable = floor_to_step(snapshot.available_balance * MARGIN_USAGE_CAP / per_unit_margin, spec.step_size)
            if qty > affordable:
                if too_small(affordable):
                    fail("INSUFFICIENT_MARGIN", f"Yetersiz serbest marj (${_money(snapshot.available_balance)})", "margin_ok")
                    qty = ZERO
                else:
                    note("MARGIN_REDUCED", f"Serbest marja göre miktar {qty} -> {affordable}")
                    qty = affordable
                    reduced = True
                    ok("margin_ok")
            else:
                ok("margin_ok")

        # exchange minimums on the final quantity
        if qty > 0:
            if too_small(qty):
                fail("BELOW_MINIMUMS", f"Miktar/notional borsa minimumlarının altında (min_qty={spec.min_qty}, min_notional={spec.min_notional})", "min_size_ok")
                qty = ZERO
            else:
                ok("min_size_ok")

        # 14. fee gate
        economics: TradeEconomics | None = None
        if qty > 0 and stop_valid and tp_valid:
            rate = funding.rate if funding is not None else ZERO
            interval = funding.interval_hours if funding is not None else DEFAULT_FUNDING_INTERVAL_H
            entry_role = LiquidityRole.MAKER if p.entry_order_type is OrderType.LIMIT else LiquidityRole.TAKER
            economics = FeeCalculator.evaluate_trade_economics(
                side=p.side, entry=entry, take_profit=tp, stop_loss=stop, quantity=qty, fees=spec.fees,
                entry_role=entry_role, exit_role=LiquidityRole.TAKER,
                slippage_bps=s.default_slippage_bps, funding_rate_per_interval=rate,
                expected_holding_hours=p.expected_holding_hours, funding_interval_hours=interval,
                min_fee_cover_multiple=s.min_fee_cover_multiple, min_net_profit_usd=s.min_net_profit_usd,
                market_type=spec.market_type,
            )
            if economics.passes_fee_gate:
                ok("fee_gate")
            else:
                need = economics.cost_at_tp * s.min_fee_cover_multiple
                fail(
                    "FEE_GATE",
                    f"Beklenen brüt kâr (${_money(economics.expected_gross_profit)}) komisyon x{s.min_fee_cover_multiple.normalize():f} "
                    f"(${_money(need)}) eşiğini karşılamıyor (net kâr ${_money(economics.net_profit_at_tp)})",
                    "fee_gate",
                )

        # 15. confidence
        if p.confidence < self._min_conf:
            fail("LOW_CONFIDENCE", f"Güven {p.confidence} < eşik {self._min_conf} / confidence below threshold", "confidence_ok")
        else:
            ok("confidence_ok")

        # verdict
        if fail_codes:
            decision = RiskDecision(Verdict.REJECTED, ZERO, lev, stop, tp, economics, sizing, reasons, checks, p.decision_id)
            await self._safe_record(
                "WARN", "RISK_REJECTED",
                f"{p.symbol} {p.side.value} reddedildi: " + ", ".join(fail_codes),
                {"symbol": p.symbol, "side": p.side.value, "source": p.source, "decision_id": p.decision_id,
                 "codes": fail_codes, "reasons": reasons},
            )
            return decision
        verdict = Verdict.REDUCED if reduced else Verdict.APPROVED
        if sizing is not None and qty > 0:
            sizing = SizingResult(qty, qty * entry, qty * entry / Decimal(lev), sizing.risk_amount, sizing.capped_by, None)
        decision = RiskDecision(verdict, qty, lev, stop, tp, economics, sizing, reasons, checks, p.decision_id)
        if reduced:
            await self._safe_record("INFO", "RISK_REDUCED", f"{p.symbol} {p.side.value} küçültülerek onaylandı",
                                    {"symbol": p.symbol, "decision_id": p.decision_id, "reasons": reasons})
        return decision

    async def _safe_record(self, level: str, code: str, message: str, details: dict[str, Any]) -> None:
        try:
            await self._sink.record(level, code, message, details)
        except Exception:  # noqa: BLE001 - sink failure must not turn a rejection into an approval or crash
            pass

    # --------------------------------------------------------- protective orders
    def compute_protective_orders(
        self, decision: RiskDecision, proposal: TradeProposal, spec: InstrumentSpec
    ) -> list[OrderRequest]:
        """Hard exchange-resting protection: STOP_MARKET (mandatory), TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET.

        Stop prices round AWAY from entry (long: down, short: up) so rounding never tightens the stop.
        """
        if decision.verdict is Verdict.REJECTED or decision.approved_quantity <= 0:
            raise RiskViolation("NOT_APPROVED", "protective orders requested for a rejected decision")
        if decision.stop_loss is None or decision.take_profit is None:
            raise RiskViolation("STOP_LOSS_MISSING", "decision has no stop-loss / take-profit")
        long = proposal.side is PositionSide.LONG
        exit_side = proposal.side.exit_side
        qty = decision.approved_quantity
        entry = proposal.entry_price
        tick = spec.tick_size

        def away(price: Decimal) -> Decimal:  # stop: away from entry
            return floor_to_step(price, tick) if long else ceil_to_step(price, tick)

        def toward(price: Decimal) -> Decimal:  # take-profit: toward entry so it is reachable
            return floor_to_step(price, tick) if long else ceil_to_step(price, tick)

        def beyond(price: Decimal) -> Decimal:  # trailing activation: never earlier than +R
            return ceil_to_step(price, tick) if long else floor_to_step(price, tick)

        stop = away(decision.stop_loss)
        tp = toward(decision.take_profit)
        risk_per_unit = abs(entry - decision.stop_loss)
        r = self._s.trailing_activation_r
        activation = beyond(entry + risk_per_unit * r if long else entry - risk_per_unit * r)
        stop_dist_pct = risk_per_unit / entry
        # exchanges take callback rates in 0.1% steps
        callback = (stop_dist_pct / 2).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        callback = min(max(callback, Decimal("0.001")), Decimal("0.05"))
        did = (proposal.decision_id or "")[:20]

        def meta(role: str) -> dict[str, Any]:
            return {"role": role, "decision_id": proposal.decision_id, "source": proposal.source}

        common = dict(symbol=spec.symbol, side=exit_side, quantity=qty, reduce_only=True)
        return [
            OrderRequest(order_type=OrderType.STOP_MARKET, stop_price=stop,
                         client_order_id=f"{did}-sl" if did else None, metadata=meta("stop_loss"), **common),
            OrderRequest(order_type=OrderType.TAKE_PROFIT_MARKET, stop_price=tp,
                         client_order_id=f"{did}-tp" if did else None, metadata=meta("take_profit"), **common),
            OrderRequest(order_type=OrderType.TRAILING_STOP_MARKET, trailing_activation_price=activation,
                         trailing_callback_rate=callback,
                         client_order_id=f"{did}-tr" if did else None, metadata=meta("trailing_stop"), **common),
        ]


def breakeven_stop_price(
    position: Position,
    fees: FeeSchedule,
    slippage_bps: Decimal = ZERO,
    tick_size: Decimal | None = None,
) -> Decimal:
    """Stop price that locks net PnL >= 0 after taker fees on both legs (and slippage) once +1R is reached.

    Rounded toward profit (long: up, short: down) so the locked result is never negative.
    """
    price = FeeCalculator.breakeven_price(
        position.side, position.entry_price, fees, LiquidityRole.TAKER, LiquidityRole.TAKER, slippage_bps
    )
    if tick_size is not None and tick_size > 0:
        price = ceil_to_step(price, tick_size) if position.side is PositionSide.LONG else floor_to_step(price, tick_size)
    return price
