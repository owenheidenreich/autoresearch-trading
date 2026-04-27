"""Order state machine.

Per protocol Section 4.1:

    candidate_seen → decision_made → order_submitted → broker_acknowledged
      → working → [partially_filled] → filled
      OR
      → working → cancel_requested → cancel_confirmed
      OR
      → working → expired

      THEN: → exit_submitted → exit_filled

Phase 0: skeleton only. Strategy logic that drives state transitions is
forbidden. This module defines:
- OrderState enum (every state in the lifecycle)
- OrderEvent dataclass (one transition record)
- OrderRecord dataclass (the per-order audit row from Section 4.1)
- transition_allowed(): pure function that validates a proposed transition
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderState(str, Enum):
    """States in the entry-and-exit order lifecycle."""

    # entry side
    CANDIDATE_SEEN = "candidate_seen"
    DECISION_MADE = "decision_made"
    ORDER_SUBMITTED = "order_submitted"
    BROKER_ACKNOWLEDGED = "broker_acknowledged"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCEL_CONFIRMED = "cancel_confirmed"
    EXPIRED = "expired"

    # exit side
    EXIT_SUBMITTED = "exit_submitted"
    EXIT_FILLED = "exit_filled"


# Allowed transition graph. Read as: from_state → set of valid next states.
ALLOWED_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.CANDIDATE_SEEN: {OrderState.DECISION_MADE},
    OrderState.DECISION_MADE: {OrderState.ORDER_SUBMITTED},
    OrderState.ORDER_SUBMITTED: {OrderState.BROKER_ACKNOWLEDGED},
    OrderState.BROKER_ACKNOWLEDGED: {OrderState.WORKING},
    OrderState.WORKING: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCEL_REQUESTED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.PARTIALLY_FILLED,  # multiple partial fills are normal
        OrderState.FILLED,
        OrderState.CANCEL_REQUESTED,
        OrderState.EXPIRED,
    },
    OrderState.FILLED: {OrderState.EXIT_SUBMITTED},
    OrderState.CANCEL_REQUESTED: {OrderState.CANCEL_CONFIRMED, OrderState.FILLED},
    OrderState.CANCEL_CONFIRMED: set(),
    OrderState.EXPIRED: set(),
    OrderState.EXIT_SUBMITTED: {OrderState.EXIT_FILLED},
    OrderState.EXIT_FILLED: set(),
}


def transition_allowed(from_state: OrderState, to_state: OrderState) -> bool:
    """Pure check: is this transition in the allowed graph?"""
    return to_state in ALLOWED_TRANSITIONS.get(from_state, set())


def terminal_states() -> set[OrderState]:
    """States with no outgoing transitions."""
    return {s for s, nexts in ALLOWED_TRANSITIONS.items() if not nexts}


@dataclass(frozen=True)
class OrderEvent:
    """One state transition. Immutable; the order's history is a list of these."""

    timestamp: datetime
    from_state: OrderState
    to_state: OrderState
    notes: str = ""

    def __post_init__(self) -> None:
        if not transition_allowed(self.from_state, self.to_state):
            raise ValueError(
                f"illegal transition: {self.from_state.value} → {self.to_state.value}. "
                f"See ALLOWED_TRANSITIONS in v4/sim/order_state.py."
            )


@dataclass
class OrderRecord:
    """The per-order audit record from protocol Section 4.1.

    Every simulated order produces one of these. The fields are the raw
    inputs to the fill-probability model trained in Phase 2 / Phase 4.5:
    they are the answer key the simulator must reproduce within tolerance.
    """

    order_id: str
    contract_id: str
    side: str  # 'BUY' | 'SELL' (the strategy's intent; both sides exist for entries)
    intended_size: int

    decision_time: datetime
    submit_time: Optional[datetime] = None
    ack_time: Optional[datetime] = None
    cancel_time: Optional[datetime] = None
    fill_times: list[datetime] = field(default_factory=list)

    limit_price: Optional[float] = None
    fill_prices: list[float] = field(default_factory=list)
    fill_sizes: list[int] = field(default_factory=list)
    nbbo_at_decision: Optional[tuple[float, float]] = None
    nbbo_at_submit: Optional[tuple[float, float]] = None
    nbbo_at_fill: Optional[tuple[float, float]] = None

    quote_age_ms_at_decision: Optional[int] = None
    quote_age_ms_at_fill: Optional[int] = None
    queue_proxy: Optional[float] = None
    spread_at_submit: Optional[float] = None
    option_price_at_submit: Optional[float] = None
    underlying_velocity_at_submit: Optional[float] = None
    sequencing_suspect_flag: bool = False

    history: list[OrderEvent] = field(default_factory=list)
    final_state: Optional[OrderState] = None

    @property
    def filled_size(self) -> int:
        return sum(self.fill_sizes)

    @property
    def fill_fraction(self) -> float:
        if self.intended_size <= 0:
            return 0.0
        return self.filled_size / self.intended_size

    @property
    def is_terminal(self) -> bool:
        return self.final_state in terminal_states()

    def transition(self, *, t: datetime, to_state: OrderState, notes: str = "") -> None:
        """Append a state transition to the history.

        Raises ValueError if the transition is not allowed.
        """
        from_state = (
            self.history[-1].to_state if self.history else OrderState.CANDIDATE_SEEN
        )
        event = OrderEvent(timestamp=t, from_state=from_state, to_state=to_state, notes=notes)
        self.history.append(event)
        self.final_state = to_state
