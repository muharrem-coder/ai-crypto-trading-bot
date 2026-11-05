"""Hard-risk layer: independent of the LLM; it can only approve, reduce or reject proposals."""

from app.risk.fee_calculator import CostBreakdown, FeeCalculator, TradeEconomics
from app.risk.kill_switch import KillSwitch, KillSwitchStatus
from app.risk.memory import InMemoryKillSwitchStore, InMemoryRiskEventSink
from app.risk.position_sizer import SizingResult, size_position
from app.risk.protocols import BrokerLike, KillSwitchState, KillSwitchStore, RiskEventSink
from app.risk.risk_engine import HardRiskEngine, RiskDecision, TradeProposal, Verdict, breakeven_stop_price
from app.risk.watchdog import RiskWatchdog

__all__ = [
    "BrokerLike", "CostBreakdown", "FeeCalculator", "HardRiskEngine", "InMemoryKillSwitchStore",
    "InMemoryRiskEventSink", "KillSwitch", "KillSwitchState", "KillSwitchStatus", "KillSwitchStore",
    "RiskDecision", "RiskEventSink", "RiskWatchdog", "SizingResult", "TradeEconomics", "TradeProposal",
    "Verdict", "breakeven_stop_price", "size_position",
]
