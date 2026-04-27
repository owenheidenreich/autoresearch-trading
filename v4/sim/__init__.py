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

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ExecutionPath",
    "FillModel",
    "NullSimulator",
    "OrderEvent",
    "OrderIntent",
    "OrderRecord",
    "OrderState",
    "SIMULATOR_VERSION",
    "Simulator",
    "terminal_states",
    "transition_allowed",
]
