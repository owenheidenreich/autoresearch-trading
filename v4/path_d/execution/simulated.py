"""Deterministic virtual-clock executor with auditable failure races."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Sequence

from v4.path_d.contracts import BrokerStateSnapshotV1, ExecutionEventV1, ExecutionIntentV1, GovernorDecisionV1

from .state_machine import OrderStateV1, transition_allowed


@dataclass
class VirtualMonotonicClock:
    wall_time_utc: datetime
    monotonic_ns: int = 0

    def advance_ms(self, milliseconds: int) -> None:
        if milliseconds < 0:
            raise ValueError("virtual clock cannot move backward")
        self.monotonic_ns += int(milliseconds) * 1_000_000
        self.wall_time_utc += timedelta(milliseconds=milliseconds)

    def iso(self) -> str:
        return self.wall_time_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SimulatedQuote:
    offset_ms: int
    bid_micros: int
    ask_micros: int


@dataclass(frozen=True)
class ExecutionScenario:
    outcome: str = "FULL"
    submit_latency_ms: int = 100
    acknowledge_latency_ms: int = 20
    cancel_after_ms: int = 500
    cancel_confirm_latency_ms: int = 20
    fill_price_mode: str = "ARRIVAL_QUOTE"

    def __post_init__(self) -> None:
        if self.outcome not in {"FULL", "NO_FILL", "PARTIAL", "LATE_FILL_AFTER_CANCEL", "DISCONNECT", "REJECT"}:
            raise ValueError(f"unsupported simulated outcome {self.outcome!r}")
        if self.fill_price_mode not in {"ARRIVAL_QUOTE", "SUBMITTED_HARD_LIMIT"}:
            raise ValueError(
                f"unsupported simulated fill-price mode {self.fill_price_mode!r}"
            )


class SimulatedExecutor:
    """Offline ExecutorPort implementation; never claims observed market fills."""

    def __init__(
        self,
        *,
        clock: VirtualMonotonicClock,
        quote_tape: Sequence[SimulatedQuote],
        scenario: ExecutionScenario | None = None,
    ) -> None:
        self.clock = clock
        self.quote_tape = tuple(sorted(quote_tape, key=lambda quote: quote.offset_ms))
        self.scenario = scenario or ExecutionScenario()
        self.events: list[ExecutionEventV1] = []
        self.states: dict[str, OrderStateV1] = {}
        self.intents: dict[str, ExecutionIntentV1] = {}

    def submit(self, intent: ExecutionIntentV1, authorization: GovernorDecisionV1) -> Sequence[ExecutionEventV1]:
        order_id = "sim-" + hashlib.sha256(intent.intent_id.encode("ascii")).hexdigest()[:16]
        self.intents[order_id] = intent
        self.states[order_id] = OrderStateV1.CREATED
        if authorization.intent_id != intent.intent_id or authorization.disposition != "ALLOW" or authorization.authorization_token is None:
            self._transition(order_id, OrderStateV1.REJECTED, "AUTHORIZATION_REQUIRED")
            return tuple(self.events)
        self._transition(order_id, OrderStateV1.SUBMIT_AUTHORIZED, "GOVERNOR_TOKEN_VERIFIED")
        self.clock.advance_ms(self.scenario.submit_latency_ms)
        self._transition(order_id, OrderStateV1.SUBMITTED, "VIRTUAL_SUBMIT")
        if self.scenario.outcome == "REJECT":
            self._transition(order_id, OrderStateV1.REJECTED, "SIMULATED_REJECT")
            return tuple(self.events)
        if self.scenario.outcome == "DISCONNECT":
            self._transition(order_id, OrderStateV1.DISCONNECTED, "SIMULATED_DISCONNECT")
            self._transition(order_id, OrderStateV1.UNKNOWN_RECONCILE, "BROKER_OUTCOME_UNKNOWN")
            return tuple(self.events)
        self.clock.advance_ms(self.scenario.acknowledge_latency_ms)
        self._transition(order_id, OrderStateV1.ACKNOWLEDGED, "SIMULATED_ACK")
        self._transition(order_id, OrderStateV1.WORKING, "SIMULATED_WORKING")
        quote = self._quote_at(self.scenario.submit_latency_ms + self.scenario.acknowledge_latency_ms)
        marketable = self._marketable(intent, quote)
        if self.scenario.outcome == "FULL" and marketable:
            self._transition(order_id, OrderStateV1.FILLED, "SIMULATED_MARKETABLE_FULL", filled_quantity=intent.decision.quantity, fill_price_micros=self._fill_price(intent, quote))
        elif self.scenario.outcome == "PARTIAL" and marketable:
            partial = max(1, intent.decision.quantity // 2)
            self._transition(order_id, OrderStateV1.PARTIALLY_FILLED, "SIMULATED_PARTIAL", filled_quantity=partial, fill_price_micros=self._fill_price(intent, quote))
            self.clock.advance_ms(self.scenario.cancel_after_ms)
            self._transition(order_id, OrderStateV1.CANCEL_REQUESTED, "PARTIAL_REMAINDER_CANCEL_REQUESTED")
            self.clock.advance_ms(self.scenario.cancel_confirm_latency_ms)
            self._transition(order_id, OrderStateV1.CANCEL_CONFIRMED, "PARTIAL_REMAINDER_CANCELLED", filled_quantity=partial)
        elif self.scenario.outcome == "LATE_FILL_AFTER_CANCEL":
            self.clock.advance_ms(self.scenario.cancel_after_ms)
            self._transition(order_id, OrderStateV1.CANCEL_REQUESTED, "SIMULATED_CANCEL_REQUESTED")
            self.clock.advance_ms(self.scenario.cancel_confirm_latency_ms)
            late_quote = self._quote_at(self.clock.monotonic_ns // 1_000_000)
            if late_quote is None or not self._marketable(intent, late_quote):
                self._transition(
                    order_id,
                    OrderStateV1.CANCEL_CONFIRMED,
                    "NO_PROVABLE_LATE_FILL_CANCEL_CONFIRMED",
                )
            else:
                self._transition(order_id, OrderStateV1.LATE_FILL_AFTER_CANCEL, "FILL_ARRIVED_DURING_CANCEL_RACE", filled_quantity=intent.decision.quantity, fill_price_micros=self._fill_price(intent, late_quote))
                self._transition(order_id, OrderStateV1.FILLED, "LATE_FILL_CONFIRMED", filled_quantity=intent.decision.quantity, fill_price_micros=self._fill_price(intent, late_quote))
        else:
            self.clock.advance_ms(self.scenario.cancel_after_ms)
            self._transition(order_id, OrderStateV1.CANCEL_REQUESTED, "NO_PROVABLE_FILL_CANCEL_REQUESTED")
            self.clock.advance_ms(self.scenario.cancel_confirm_latency_ms)
            self._transition(order_id, OrderStateV1.CANCEL_CONFIRMED, "SIMULATED_CANCEL_CONFIRMED")
        return tuple(self.events)

    def reconcile(self, order_id: str, broker_state: BrokerStateSnapshotV1) -> Sequence[ExecutionEventV1]:
        if self.states.get(order_id) != OrderStateV1.UNKNOWN_RECONCILE:
            raise ValueError("reconciliation is allowed only from UNKNOWN_RECONCILE")
        self._transition(order_id, OrderStateV1.RECONCILING, "FAKE_BROKER_STATE_READ")
        intent = self.intents[order_id]
        held = any(position.osi_symbol == intent.contract.osi_symbol for position in broker_state.open_positions)
        filled = held if intent.decision.position_effect == "OPEN" else not held
        target = OrderStateV1.RECONCILED_FILLED if filled else OrderStateV1.RECONCILED_CANCELLED
        self._transition(order_id, target, "RECONCILED_FROM_POSITION_STATE")
        return tuple(self.events)

    def write_transcript(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(event.to_json() + "\n" for event in self.events), encoding="utf-8")

    def _transition(self, order_id: str, target: OrderStateV1, reason: str, *, filled_quantity: int = 0, fill_price_micros: int | None = None) -> None:
        source = self.states[order_id]
        if not transition_allowed(source, target):
            raise ValueError(f"illegal Path-D transition {source.value}->{target.value}")
        intent = self.intents[order_id]
        event = ExecutionEventV1.create(
            order_id=order_id, intent_id=intent.intent_id, event_type=target.value,
            state_from=source.value, state_to=target.value, event_at_utc=self.clock.iso(),
            monotonic_ns=self.clock.monotonic_ns, filled_quantity=filled_quantity,
            fill_price_micros=fill_price_micros, reason_code=reason,
        )
        self.events.append(event)
        self.states[order_id] = target

    def _quote_at(self, offset_ms: int) -> SimulatedQuote | None:
        eligible = [quote for quote in self.quote_tape if quote.offset_ms <= offset_ms]
        if not eligible:
            return None
        return eligible[-1]

    @staticmethod
    def _marketable(intent: ExecutionIntentV1, quote: SimulatedQuote | None) -> bool:
        if quote is None:
            return False
        return quote.ask_micros <= intent.price_budget.hard_limit_micros if intent.decision.side == "BUY" else quote.bid_micros >= intent.price_budget.hard_limit_micros

    def _fill_price(self, intent: ExecutionIntentV1, quote: SimulatedQuote) -> int:
        if self.scenario.fill_price_mode == "SUBMITTED_HARD_LIMIT":
            return intent.price_budget.hard_limit_micros
        return quote.ask_micros if intent.decision.side == "BUY" else quote.bid_micros
