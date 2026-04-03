"""TradeIntent and related data contracts.

The central contract of the v2 system. See docs/v2/contracts.md for the
full specification.

TradeIntent is what:
- replay scores (simulates the trade, measures P&L)
- live executes (places IBKR orders)
- training learns to emit (model output -> TradeIntent)

v1 origin: training/live/contracts.py (DecisionIntent, ExecutionState)
"""
# TODO: TradeIntent dataclass (frozen=True) per docs/v2/contracts.md
# TODO: RiskAdjustment dataclass (frozen=True)
# TODO: Validation helpers (validate_intent)
# TODO: Serialization (to_dict, from_dict, deterministic JSON)
