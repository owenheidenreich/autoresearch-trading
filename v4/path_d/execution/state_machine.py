"""Complete additive Path-D order lifecycle including unknown reconciliation."""
from __future__ import annotations

from enum import Enum


class OrderStateV1(str, Enum):
    CREATED = "CREATED"
    SUBMIT_AUTHORIZED = "SUBMIT_AUTHORIZED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCEL_CONFIRMED = "CANCEL_CONFIRMED"
    LATE_FILL_AFTER_CANCEL = "LATE_FILL_AFTER_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    DISCONNECTED = "DISCONNECTED"
    UNKNOWN_RECONCILE = "UNKNOWN_RECONCILE"
    RECONCILING = "RECONCILING"
    RECONCILED_FILLED = "RECONCILED_FILLED"
    RECONCILED_CANCELLED = "RECONCILED_CANCELLED"


ALLOWED_TRANSITIONS: dict[OrderStateV1, frozenset[OrderStateV1]] = {
    OrderStateV1.CREATED: frozenset({OrderStateV1.SUBMIT_AUTHORIZED, OrderStateV1.REJECTED}),
    OrderStateV1.SUBMIT_AUTHORIZED: frozenset({OrderStateV1.SUBMITTED}),
    OrderStateV1.SUBMITTED: frozenset({OrderStateV1.ACKNOWLEDGED, OrderStateV1.REJECTED, OrderStateV1.DISCONNECTED}),
    OrderStateV1.ACKNOWLEDGED: frozenset({OrderStateV1.WORKING, OrderStateV1.DISCONNECTED}),
    OrderStateV1.WORKING: frozenset({OrderStateV1.PARTIALLY_FILLED, OrderStateV1.FILLED, OrderStateV1.CANCEL_REQUESTED, OrderStateV1.EXPIRED, OrderStateV1.DISCONNECTED}),
    OrderStateV1.PARTIALLY_FILLED: frozenset({OrderStateV1.PARTIALLY_FILLED, OrderStateV1.FILLED, OrderStateV1.CANCEL_REQUESTED, OrderStateV1.DISCONNECTED}),
    OrderStateV1.CANCEL_REQUESTED: frozenset({OrderStateV1.CANCEL_CONFIRMED, OrderStateV1.LATE_FILL_AFTER_CANCEL, OrderStateV1.DISCONNECTED}),
    OrderStateV1.LATE_FILL_AFTER_CANCEL: frozenset({OrderStateV1.FILLED, OrderStateV1.UNKNOWN_RECONCILE}),
    OrderStateV1.DISCONNECTED: frozenset({OrderStateV1.UNKNOWN_RECONCILE}),
    OrderStateV1.UNKNOWN_RECONCILE: frozenset({OrderStateV1.RECONCILING}),
    OrderStateV1.RECONCILING: frozenset({OrderStateV1.RECONCILED_FILLED, OrderStateV1.RECONCILED_CANCELLED}),
    OrderStateV1.FILLED: frozenset(), OrderStateV1.CANCEL_CONFIRMED: frozenset(),
    OrderStateV1.REJECTED: frozenset(), OrderStateV1.EXPIRED: frozenset(),
    OrderStateV1.RECONCILED_FILLED: frozenset(), OrderStateV1.RECONCILED_CANCELLED: frozenset(),
}


def transition_allowed(source: OrderStateV1, target: OrderStateV1) -> bool:
    return target in ALLOWED_TRANSITIONS[source]

