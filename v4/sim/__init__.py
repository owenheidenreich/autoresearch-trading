"""v4.sim — simulator interface + order-state-machine skeleton.

Phase 0 contains interface only. Strategy logic is forbidden until Phase 2A.
Concrete fill-calibrated simulators arrive in Phase 4 / 4.5.
"""
from .order_state import (
    ALLOWED_TRANSITIONS,
    OrderEvent,
    OrderRecord,
    OrderState,
    terminal_states,
    transition_allowed,
)
from .simulator import (
    SIMULATOR_VERSION,
    ExecutionPath,
    FillModel,
    NullSimulator,
    OrderIntent,
    Simulator,
)
from .paper_replay import (
    DEFAULT_SLIPPAGE_SCENARIOS,
    PaperReplayConfig,
    SlippageScenario,
    apply_slippage,
    build_replay_frame,
    concentration_metrics,
    live_data_parity_checks,
    load_lifecycle_steps,
    load_selected_trades,
    order_state_summary,
    promotion_gate_status,
    summarize_by_split_seed,
)
from .shadow_paper import ShadowPaperConfig, ShadowTradeLedger, replay_shadow_paper

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ExecutionPath",
    "FillModel",
    "NullSimulator",
    "OrderEvent",
    "OrderIntent",
    "OrderRecord",
    "OrderState",
    "PaperReplayConfig",
    "SIMULATOR_VERSION",
    "ShadowPaperConfig",
    "ShadowTradeLedger",
    "Simulator",
    "SlippageScenario",
    "DEFAULT_SLIPPAGE_SCENARIOS",
    "apply_slippage",
    "build_replay_frame",
    "concentration_metrics",
    "live_data_parity_checks",
    "load_lifecycle_steps",
    "load_selected_trades",
    "order_state_summary",
    "promotion_gate_status",
    "replay_shadow_paper",
    "summarize_by_split_seed",
    "terminal_states",
    "transition_allowed",
]
