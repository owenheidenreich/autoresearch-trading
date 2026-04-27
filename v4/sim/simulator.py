"""Simulator interface skeleton (no strategy logic).

Per protocol Section 4.1, the simulator is a first-class artifact and the
fill model is its own first-class supervised model. Phase 0 defines the
*interface* both will plug into; the actual fill model is calibrated in
Phase 4 and 4.5 against IBKR paper + live data.

This module:
- Defines the Path-A / Path-B execution paths (marketable vs passive).
- Declares the FillModel Protocol that future fill-probability models
  must implement.
- Declares the Simulator Protocol that future strategies will plug into.
- Provides a NullSimulator for tests / pipeline-integrity verification —
  it satisfies the interface but rejects any actual entry attempt.

Strategy logic is Phase 2A+ work and is forbidden in Phase 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from .order_state import OrderRecord, OrderState


SIMULATOR_VERSION = "sim-v4.0.0"
"""Version of the simulator interface. Bumped on breaking changes; recorded
in the audit log per ingest run."""


class ExecutionPath(str, Enum):
    """Execution paths from protocol Section 4.1."""

    MARKETABLE = "marketable"  # Path A: immediate fill at NBBO ± slip
    PASSIVE = "passive"        # Path B: limit order with fill probability


@dataclass(frozen=True)
class OrderIntent:
    """A strategy's request to enter or exit a position.

    The simulator turns OrderIntent into an OrderRecord by simulating the
    order lifecycle. Strategies are pure: they emit intents and consume
    OrderRecords. They never advance order state directly.
    """

    decision_time: datetime
    contract_id: str
    side: str  # 'BUY' | 'SELL'
    intended_size: int
    path: ExecutionPath
    limit_price: float | None = None  # required for PASSIVE; ignored for MARKETABLE
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path == ExecutionPath.PASSIVE and self.limit_price is None:
            raise ValueError("PASSIVE path requires limit_price")
        if self.intended_size <= 0:
            raise ValueError(f"intended_size must be positive; got {self.intended_size}")
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL'; got {self.side!r}")


@runtime_checkable
class FillModel(Protocol):
    """A fill-probability model.

    Phase 0 just declares the interface. The Path-B fill model is fit in
    Phase 4 against IBKR paper data; the Path-A slip model is fit similarly.
    Per protocol Section 4.1, both are validated against decomposed
    calibration gates (Brier on fill probability, MAD on fill price, etc.)
    before any model trained on top of them can be promoted.
    """

    def fill_probability(
        self,
        *,
        intent: OrderIntent,
        nbbo: tuple[float, float],
        spread: float,
        time_of_day_seconds: int,
        quote_age_ms: int,
    ) -> float: ...

    def expected_fill_price(
        self,
        *,
        intent: OrderIntent,
        nbbo: tuple[float, float],
        spread: float,
    ) -> float: ...


@runtime_checkable
class Simulator(Protocol):
    """The simulator interface a strategy plugs into.

    Phase 0 only declares the contract. Concrete simulators with full
    NBBO replay, fill-quality distributions, latency, partial fills, and
    sequencing-suspect diagnostics arrive in Phase 1+ once Databento
    historical NBBO data is available.
    """

    version: str

    def submit(self, intent: OrderIntent) -> OrderRecord: ...

    def step(self, current_time: datetime) -> list[OrderRecord]: ...
    """Advance the simulator clock; return any orders that changed state."""


@dataclass
class NullSimulator:
    """Phase-0 placeholder. Satisfies the Simulator interface but refuses
    to actually fill anything. Used in tests and the pipeline-integrity
    report to confirm the interface exists end-to-end without committing
    to fill-quality numbers we haven't calibrated.

    Concrete fill simulators arrive in Phase 1+ (Databento NBBO replay)
    and Phase 4+ (IBKR paper fill calibration).
    """

    version: str = SIMULATOR_VERSION
    rejected_intents: int = 0

    def submit(self, intent: OrderIntent) -> OrderRecord:
        rec = OrderRecord(
            order_id=f"null-{self.rejected_intents}",
            contract_id=intent.contract_id,
            side=intent.side,
            intended_size=intent.intended_size,
            decision_time=intent.decision_time,
            limit_price=intent.limit_price,
        )
        rec.transition(t=intent.decision_time, to_state=OrderState.DECISION_MADE,
                       notes="NullSimulator rejects all intents in Phase 0")
        rec.transition(t=intent.decision_time, to_state=OrderState.ORDER_SUBMITTED)
        rec.transition(t=intent.decision_time, to_state=OrderState.BROKER_ACKNOWLEDGED)
        rec.transition(t=intent.decision_time, to_state=OrderState.WORKING)
        rec.transition(t=intent.decision_time, to_state=OrderState.EXPIRED,
                       notes="Phase 0: no fill model calibrated yet")
        self.rejected_intents += 1
        return rec

    def step(self, current_time: datetime) -> list[OrderRecord]:
        return []
