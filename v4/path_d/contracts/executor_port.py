"""ExecutorPort dependency-inversion boundary."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .broker_state import BrokerStateSnapshotV1
from .execution_event import ExecutionEventV1
from .execution_intent import ExecutionIntentV1
from .governor_decision import GovernorDecisionV1


@runtime_checkable
class ExecutorPort(Protocol):
    def submit(
        self,
        intent: ExecutionIntentV1,
        authorization: GovernorDecisionV1,
    ) -> Sequence[ExecutionEventV1]: ...

    def reconcile(
        self,
        order_id: str,
        broker_state: BrokerStateSnapshotV1,
    ) -> Sequence[ExecutionEventV1]: ...

