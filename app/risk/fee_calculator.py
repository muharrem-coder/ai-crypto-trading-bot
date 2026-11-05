"""Fee / slippage / funding arithmetic and the fee-cover gate. Decimal only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from app.core.types import FeeSchedule, LiquidityRole, MarketType, PositionSide

ZERO = Decimal(0)
BPS = Decimal(10000)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    entry_fee: Decimal
    exit_fee: Decimal
    slippage_cost: Decimal
    funding_cost: Decimal  # positive = we pay, negative = we receive
    total: Decimal


@dataclass(frozen=True, slots=True)
class TradeEconomics:
    expected_gross_profit: Decimal  # entry -> take profit, before costs
    expected_gross_loss: Decimal  # entry -> stop loss, before costs (positive number)
    cost_at_tp: Decimal
    cost_at_sl: Decimal
    costs_tp: CostBreakdown
    costs_sl: CostBreakdown
    net_profit_at_tp: Decimal
    net_loss_at_sl: Decimal  # positive number: gross loss + costs
    reward_risk_net: Decimal
    breakeven_price: Decimal
    min_profitable_move_pct: Decimal  # move needed for gross profit to clear cost * min_fee_cover_multiple
    min_fee_cover_multiple: Decimal
    passes_fee_gate: bool

    @property
    def total_round_trip_cost(self) -> Decimal:
        return self.cost_at_tp


def funding_intervals(expected_holding_hours: Decimal, funding_interval_hours: int | Decimal) -> int:
    if expected_holding_hours <= 0 or Decimal(funding_interval_hours) <= 0:
        return 0
    return int(math.ceil(expected_holding_hours / Decimal(funding_interval_hours)))


def _direction(side: PositionSide) -> Decimal:
    return Decimal(1) if side is PositionSide.LONG else Decimal(-1)


class FeeCalculator:
    @staticmethod
    def round_trip_cost(
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
        fees: FeeSchedule,
        entry_role: LiquidityRole,
        exit_role: LiquidityRole,
        slippage_bps: Decimal,
        funding_rate_per_interval: Decimal,
        expected_holding_hours: Decimal,
        funding_interval_hours: int | Decimal,
        side: PositionSide,
        market_type: MarketType = MarketType.PERPETUAL,
    ) -> CostBreakdown:
        """Cost of opening and closing ``quantity``.

        * fee = rate(role) * notional of that leg
        * slippage = slippage_bps/10000 * (entry notional + exit notional)
        * funding (perpetual only) = entry notional * rate * ceil(hours/interval); longs pay when rate>0,
          shorts receive.
        """
        entry_notional = entry_price * quantity
        exit_notional = exit_price * quantity
        entry_fee = entry_notional * fees.rate(entry_role)
        exit_fee = exit_notional * fees.rate(exit_role)
        slippage = (entry_notional + exit_notional) * slippage_bps / BPS
        if market_type is MarketType.PERPETUAL:
            n = funding_intervals(expected_holding_hours, funding_interval_hours)
            funding = entry_notional * funding_rate_per_interval * n * _direction(side)
        else:
            funding = ZERO
        return CostBreakdown(entry_fee, exit_fee, slippage, funding, entry_fee + exit_fee + slippage + funding)

    @staticmethod
    def breakeven_price(
        side: PositionSide,
        entry_price: Decimal,
        fees: FeeSchedule,
        entry_role: LiquidityRole = LiquidityRole.TAKER,
        exit_role: LiquidityRole = LiquidityRole.TAKER,
        slippage_bps: Decimal = ZERO,
        funding_rate_per_interval: Decimal = ZERO,
        funding_intervals_count: int = 0,
    ) -> Decimal:
        """Exit price at which net PnL is exactly zero after both fees, slippage and funding.

        Long : (P-E) - fe*E - fx*P - s*(E+P) - E*f = 0  ->  P = E(1+fe+s+f) / (1-fx-s)
        Short: (E-P) - fe*E - fx*P - s*(E+P) + E*f = 0  ->  P = E(1-fe-s+f) / (1+fx+s)
        where f = funding_rate * intervals (paid by longs when positive, received by shorts).
        """
        fe = fees.rate(entry_role)
        fx = fees.rate(exit_role)
        s = slippage_bps / BPS
        f = funding_rate_per_interval * funding_intervals_count
        if side is PositionSide.LONG:
            return entry_price * (1 + fe + s + f) / (1 - fx - s)
        return entry_price * (1 - fe - s + f) / (1 + fx + s)

    @classmethod
    def evaluate_trade_economics(
        cls,
        side: PositionSide,
        entry: Decimal,
        take_profit: Decimal,
        stop_loss: Decimal,
        quantity: Decimal,
        fees: FeeSchedule,
        entry_role: LiquidityRole,
        exit_role: LiquidityRole,
        slippage_bps: Decimal,
        funding_rate_per_interval: Decimal,
        expected_holding_hours: Decimal,
        funding_interval_hours: int | Decimal,
        min_fee_cover_multiple: Decimal = Decimal(2),
        min_net_profit_usd: Decimal = ZERO,
        market_type: MarketType = MarketType.PERPETUAL,
        stop_exit_role: LiquidityRole = LiquidityRole.TAKER,
    ) -> TradeEconomics:
        d = _direction(side)
        gross_profit = (take_profit - entry) * quantity * d
        gross_loss = (entry - stop_loss) * quantity * d
        common = dict(
            fees=fees,
            entry_role=entry_role,
            slippage_bps=slippage_bps,
            funding_rate_per_interval=funding_rate_per_interval,
            expected_holding_hours=expected_holding_hours,
            funding_interval_hours=funding_interval_hours,
            side=side,
            market_type=market_type,
        )
        costs_tp = cls.round_trip_cost(entry, take_profit, quantity, exit_role=exit_role, **common)
        costs_sl = cls.round_trip_cost(entry, stop_loss, quantity, exit_role=stop_exit_role, **common)
        net_profit = gross_profit - costs_tp.total
        net_loss = gross_loss + costs_sl.total
        rr = net_profit / net_loss if net_loss > 0 else ZERO
        n = funding_intervals(expected_holding_hours, funding_interval_hours) if market_type is MarketType.PERPETUAL else 0
        be = cls.breakeven_price(
            side, entry, fees, entry_role, exit_role, slippage_bps,
            funding_rate_per_interval if market_type is MarketType.PERPETUAL else ZERO, n,
        )
        entry_notional = entry * quantity
        cost_pct = costs_tp.total / entry_notional if entry_notional > 0 else ZERO
        passes = gross_profit > costs_tp.total * min_fee_cover_multiple and net_profit >= min_net_profit_usd
        return TradeEconomics(
            expected_gross_profit=gross_profit,
            expected_gross_loss=gross_loss,
            cost_at_tp=costs_tp.total,
            cost_at_sl=costs_sl.total,
            costs_tp=costs_tp,
            costs_sl=costs_sl,
            net_profit_at_tp=net_profit,
            net_loss_at_sl=net_loss,
            reward_risk_net=rr,
            breakeven_price=be,
            min_profitable_move_pct=cost_pct * min_fee_cover_multiple,
            min_fee_cover_multiple=min_fee_cover_multiple,
            passes_fee_gate=passes,
        )
