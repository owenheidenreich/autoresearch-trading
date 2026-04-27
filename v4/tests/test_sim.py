"""Tests for the simulator interface skeleton + order state machine."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from v4.sim import (
    ALLOWED_TRANSITIONS,
    ExecutionPath,
    FillModel,
    NullSimulator,
    OrderEvent,
    OrderIntent,
    OrderRecord,
    OrderState,
    Simulator,
    terminal_states,
    transition_allowed,
)


# ---------- order state machine ----------

def test_known_terminal_states() -> None:
    terminals = terminal_states()
    assert OrderState.CANCEL_CONFIRMED in terminals
    assert OrderState.EXPIRED in terminals
    assert OrderState.EXIT_FILLED in terminals


def test_filled_can_transition_only_to_exit_submitted() -> None:
    assert transition_allowed(OrderState.FILLED, OrderState.EXIT_SUBMITTED)
    # Cannot go back to working or working etc.
    assert not transition_allowed(OrderState.FILLED, OrderState.WORKING)
    assert not transition_allowed(OrderState.FILLED, OrderState.CANDIDATE_SEEN)


def test_working_branches() -> None:
    valid = ALLOWED_TRANSITIONS[OrderState.WORKING]
    assert OrderState.PARTIALLY_FILLED in valid
    assert OrderState.FILLED in valid
    assert OrderState.CANCEL_REQUESTED in valid
    assert OrderState.EXPIRED in valid


def test_event_rejects_illegal_transition() -> None:
    t = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="illegal transition"):
        OrderEvent(
            timestamp=t,
            from_state=OrderState.CANDIDATE_SEEN,
            to_state=OrderState.FILLED,  # cannot skip the lifecycle
        )


def test_partial_fill_can_loop() -> None:
    """Multiple partial fills are normal."""
    assert transition_allowed(OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED)
    assert transition_allowed(OrderState.PARTIALLY_FILLED, OrderState.FILLED)


def test_cancel_requested_can_still_fill() -> None:
    """Race condition: cancel arrives but the order fills first."""
    assert transition_allowed(OrderState.CANCEL_REQUESTED, OrderState.FILLED)
    assert transition_allowed(OrderState.CANCEL_REQUESTED, OrderState.CANCEL_CONFIRMED)


# ---------- OrderRecord ----------

def test_order_record_transition_logs_history() -> None:
    t = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    rec = OrderRecord(
        order_id="o1",
        contract_id="SPXW-20230102-04000.000-C",
        side="BUY",
        intended_size=1,
        decision_time=t,
    )
    rec.transition(t=t, to_state=OrderState.DECISION_MADE)
    rec.transition(t=t, to_state=OrderState.ORDER_SUBMITTED)
    assert len(rec.history) == 2
    assert rec.final_state == OrderState.ORDER_SUBMITTED


def test_order_record_rejects_illegal_skip() -> None:
    t = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    rec = OrderRecord(
        order_id="o1",
        contract_id="SPXW-20230102-04000.000-C",
        side="BUY",
        intended_size=1,
        decision_time=t,
    )
    with pytest.raises(ValueError, match="illegal transition"):
        rec.transition(t=t, to_state=OrderState.FILLED)


def test_fill_fraction_and_filled_size() -> None:
    t = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    rec = OrderRecord(
        order_id="o1",
        contract_id="SPXW-20230102-04000.000-C",
        side="BUY",
        intended_size=4,
        decision_time=t,
        fill_sizes=[1, 2],
    )
    assert rec.filled_size == 3
    assert rec.fill_fraction == 0.75


# ---------- OrderIntent ----------

def test_passive_intent_requires_limit_price() -> None:
    with pytest.raises(ValueError, match="PASSIVE path requires limit_price"):
        OrderIntent(
            decision_time=datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc),
            contract_id="x",
            side="BUY",
            intended_size=1,
            path=ExecutionPath.PASSIVE,
        )


def test_intent_rejects_zero_size() -> None:
    with pytest.raises(ValueError, match="intended_size must be positive"):
        OrderIntent(
            decision_time=datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc),
            contract_id="x",
            side="BUY",
            intended_size=0,
            path=ExecutionPath.MARKETABLE,
        )


def test_intent_rejects_bad_side() -> None:
    with pytest.raises(ValueError, match="side must be"):
        OrderIntent(
            decision_time=datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc),
            contract_id="x",
            side="long",  # wrong vocabulary
            intended_size=1,
            path=ExecutionPath.MARKETABLE,
        )


# ---------- NullSimulator ----------

def test_null_simulator_satisfies_simulator_protocol() -> None:
    sim = NullSimulator()
    assert isinstance(sim, Simulator)


def test_null_simulator_rejects_intent_into_expired_state() -> None:
    sim = NullSimulator()
    intent = OrderIntent(
        decision_time=datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc),
        contract_id="SPXW-20230102-04000.000-C",
        side="BUY",
        intended_size=1,
        path=ExecutionPath.MARKETABLE,
    )
    rec = sim.submit(intent)
    assert rec.final_state == OrderState.EXPIRED
    assert rec.is_terminal


def test_fillmodel_protocol_smoke() -> None:
    """Smoke-test: a class implementing the FillModel methods passes
    runtime isinstance() check."""

    class StubFill:
        def fill_probability(self, *, intent, nbbo, spread, time_of_day_seconds, quote_age_ms):
            return 0.5
        def expected_fill_price(self, *, intent, nbbo, spread):
            return sum(nbbo) / 2

    assert isinstance(StubFill(), FillModel)
