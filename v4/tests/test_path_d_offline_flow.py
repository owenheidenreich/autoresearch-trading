from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from v4.path_d.contracts import (
    CanonicalMarketEventV1,
    DecisionDirectiveV1,
    ExecutionIntentV1,
    GovernorDecisionV1,
    PositionV1,
)
from v4.path_d.decision import ExitFixtureState, OfflineDecisionService
from v4.path_d.execution.fake_gateway_stub import FakeGatewayStub
from v4.path_d.execution.latency import LATENCY_RUNGS_MS, OWNED_PAIRED_SESSIONS, run_latency_bounds
from v4.path_d.execution.simulated import ExecutionScenario, SimulatedExecutor, SimulatedQuote, VirtualMonotonicClock
from v4.path_d.risk import DeterministicGovernor, FeedHealthV1, GovernorConfigV1, LifecycleStateV1, fake_broker_state


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "v4/path_d/contracts/fixtures"


def _offline_result():
    events = [
        CanonicalMarketEventV1.create(
            source="THETADATA_SPX", event_type="SPX_INDEX", session_date="2026-06-30",
            source_timestamp_utc="2026-06-30T13:30:00Z", received_timestamp_utc="2026-06-30T13:30:00Z",
            index_symbol="SPX", index_price_micros=7_490_000_000, volume=10,
        ),
        CanonicalMarketEventV1.create(
            source="THETADATA_SPX", event_type="SPX_INDEX", session_date="2026-06-30",
            source_timestamp_utc="2026-06-30T13:30:01Z", received_timestamp_utc="2026-06-30T13:30:01Z",
            index_symbol="SPX", index_price_micros=7_491_000_000, volume=20,
        ),
        CanonicalMarketEventV1.create(
            source="DATABENTO_OPRA", event_type="OPTION_QUOTE", session_date="2026-06-30",
            source_timestamp_utc="2026-06-30T13:30:01Z", received_timestamp_utc="2026-06-30T13:30:02Z",
            osi_symbol="SPXW  260630C07490000", bid_price_micros=2_000_000, ask_price_micros=2_100_000,
            bid_size=10, ask_size=8,
        ),
    ]
    result = OfflineDecisionService().replay(
        reversed(events),
        exit_state=ExitFixtureState(
            held_osi_symbol="SPXW  260630C07490000", position_snapshot_version="fake-state-1",
            entry_bid_micros=2_500_000, running_max_bid_micros=3_000_000,
            opened_at_utc="2026-06-30T13:30:00Z",
        ),
    )
    return events, result


def _authorized(quantity: int = 1):
    _, result = _offline_result()
    assert result.intent is not None
    intent = result.intent
    if quantity != 1:
        intent = ExecutionIntentV1.create(
            trace_id=intent.trace_id, parent_intent_id=intent.parent_intent_id, origin=intent.origin,
            producer=intent.producer,
            decision=DecisionDirectiveV1(**{**intent.decision.to_dict(), "quantity": quantity}),
            contract=intent.contract, price_budget=intent.price_budget, clocks=intent.clocks,
            state_precondition=intent.state_precondition, execution_profile_version=intent.execution_profile_version,
        )
    broker = fake_broker_state(
        captured_at_utc="2026-06-30T13:30:02Z",
        positions=(PositionV1(intent.contract.osi_symbol, 1, 2_500_000, "2026-06-30T13:30:00Z"),),
    )
    feed = FeedHealthV1("2026-06-30T13:30:02Z", "2026-06-30T13:30:01Z")
    governor = DeterministicGovernor(GovernorConfigV1(max_quantity=max(1, quantity)))
    authorization = governor.evaluate(intent, broker_state=broker, feed_health=feed, now_utc="2026-06-30T13:30:02Z")
    return result, intent, broker, feed, authorization


def _executor(outcome: str, *, quantity: int = 1):
    result, intent, broker, _, authorization = _authorized(quantity)
    executor = SimulatedExecutor(
        clock=VirtualMonotonicClock(datetime(2026, 6, 30, 13, 30, 2, tzinfo=timezone.utc)),
        quote_tape=(SimulatedQuote(0, 2_000_000, 2_100_000), SimulatedQuote(1_000, 1_950_000, 2_050_000)),
        scenario=ExecutionScenario(outcome=outcome, submit_latency_ms=100),
    )
    return executor, intent, broker, authorization


def test_offline_decision_replays_received_clock_and_serializes_exit_intent() -> None:
    events, result = _offline_result()
    assert result.intent is not None
    assert result.intent.clocks.decision_clock == "received_timestamp_utc"
    assert result.intent.origin == "DETERMINISTIC_EXIT"
    assert result.intent.decision.reason_code == "DETERMINISTIC_UPWARD_FLOOR"
    assert result.intent.producer.feature_snapshot_sha256 == result.snapshot.snapshot_id
    assert json.loads(result.intent.to_json())["contract"]["osi_symbol"] == events[-1].osi_symbol


def test_governor_is_sole_authorizer_and_blocks_stale_feed_or_wrong_position() -> None:
    _, intent, broker, feed, authorization = _authorized()
    assert authorization.disposition == "ALLOW"
    assert authorization.authorization_token is not None
    stale = FeedHealthV1("2026-06-30T13:29:00Z", feed.spx_received_timestamp_utc)
    blocked = DeterministicGovernor().evaluate(intent, broker_state=broker, feed_health=stale, now_utc="2026-06-30T13:30:02Z")
    assert blocked.disposition == "BLOCK"
    assert "OPTION_FEED_STALE" in blocked.reason_codes
    executor, _, _, _ = _executor("FULL")
    fake_block = GovernorDecisionV1.create(
        intent_id=intent.intent_id, disposition="BLOCK", reason_codes=("TEST_BLOCK",),
        evaluated_at_utc="2026-06-30T13:30:02Z", broker_state_version=broker.snapshot_version,
    )
    events = executor.submit(intent, fake_block)
    assert events[-1].state_to == "REJECTED"
    assert events[-1].reason_code == "AUTHORIZATION_REQUIRED"


def test_holding_feed_loss_creates_governor_forced_flat_intent() -> None:
    result, intent, broker, _, _ = _authorized()
    assert result.intent is not None
    lifecycle = LifecycleStateV1(
        contract=intent.contract, position_snapshot_version=broker.snapshot_version,
        entry_bid_micros=2_500_000, running_max_bid_micros=3_000_000,
        current_bid_micros=2_900_000, current_ask_micros=3_000_000,
        opened_at_utc="2026-06-30T13:30:00Z",
        feature_contract_version=intent.producer.feature_contract_version,
        feature_snapshot_sha256=intent.producer.feature_snapshot_sha256,
    )
    forced = DeterministicGovernor().lifecycle_exit_intent(
        lifecycle,
        feed_health=FeedHealthV1("2026-06-30T13:29:00Z", "2026-06-30T13:30:01Z", option_feed_available=False),
        now_utc="2026-06-30T13:30:02Z",
    )
    assert forced is not None
    assert forced.origin == "RISK_GOVERNOR"
    assert forced.decision.urgency == "FORCED_FLAT"
    assert forced.decision.reason_code == "HOLDING_FEED_LOSS_FORCED_FLAT"
    failed_feed = FeedHealthV1(
        "2026-06-30T13:29:00Z", "2026-06-30T13:30:01Z", option_feed_available=False
    )
    authorized = DeterministicGovernor().evaluate(
        forced, broker_state=broker, feed_health=failed_feed, now_utc="2026-06-30T13:30:02Z"
    )
    assert authorized.disposition == "ALLOW"
    assert authorized.authorization_token is not None
    forged = ExecutionIntentV1.create(
        trace_id=forced.trace_id,
        parent_intent_id=forced.parent_intent_id,
        origin=forced.origin,
        producer=type(forced.producer)(
            **{**forced.producer.to_dict(), "component_version": "pathd.not-the-governor.v1"}
        ),
        decision=forced.decision,
        contract=forced.contract,
        price_budget=forced.price_budget,
        clocks=forced.clocks,
        state_precondition=forced.state_precondition,
        execution_profile_version=forced.execution_profile_version,
    )
    forged_decision = DeterministicGovernor().evaluate(
        forged, broker_state=broker, feed_health=failed_feed, now_utc="2026-06-30T13:30:02Z"
    )
    assert forged_decision.disposition == "BLOCK"
    assert "FORCED_FLAT_REQUIRES_GOVERNOR_PRODUCER" in forged_decision.reason_codes


@pytest.mark.parametrize(
    ("outcome", "quantity", "required_states", "terminal"),
    [
        ("FULL", 1, {"WORKING", "FILLED"}, "FILLED"),
        ("NO_FILL", 1, {"CANCEL_REQUESTED", "CANCEL_CONFIRMED"}, "CANCEL_CONFIRMED"),
        ("PARTIAL", 2, {"PARTIALLY_FILLED", "CANCEL_REQUESTED"}, "CANCEL_CONFIRMED"),
        ("LATE_FILL_AFTER_CANCEL", 1, {"CANCEL_REQUESTED", "LATE_FILL_AFTER_CANCEL"}, "FILLED"),
        ("REJECT", 1, {"REJECTED"}, "REJECTED"),
    ],
)
def test_simulated_executor_state_machine(outcome, quantity, required_states, terminal) -> None:
    executor, intent, _, authorization = _executor(outcome, quantity=quantity)
    events = executor.submit(intent, authorization)
    states = {event.state_to for event in events}
    assert required_states <= states
    assert events[-1].state_to == terminal
    monotonic = [event.monotonic_ns for event in events]
    assert monotonic == sorted(monotonic)


def test_disconnect_enters_unknown_and_reconciles_from_fake_broker_state() -> None:
    executor, intent, broker, authorization = _executor("DISCONNECT")
    events = executor.submit(intent, authorization)
    assert events[-1].state_to == "UNKNOWN_RECONCILE"
    order_id = events[-1].order_id
    reconciled_state = fake_broker_state(
        captured_at_utc="2026-06-30T13:30:03Z", snapshot_version="fake-state-2", positions=(),
    )
    final = executor.reconcile(order_id, reconciled_state)
    assert final[-2].state_to == "RECONCILING"
    assert final[-1].state_to == "RECONCILED_FILLED"


def test_fake_gateway_stub_has_no_external_side_effects() -> None:
    gateway = FakeGatewayStub()
    gateway.submit("order-1")
    gateway.cancel("order-1")
    assert gateway.submitted_order_ids == ["order-1"]
    assert gateway.cancelled_order_ids == ["order-1"]


def test_six_owned_day_latency_harness_reports_quote_bounds_not_actual_fills() -> None:
    rows = run_latency_bounds(ROOT)
    assert len(rows) == len(OWNED_PAIRED_SESSIONS) * len(LATENCY_RUNGS_MS) == 24
    assert {row.session_date for row in rows} == set(OWNED_PAIRED_SESSIONS)
    assert {row.latency_ms for row in rows} == set(LATENCY_RUNGS_MS)
    assert all(row.fill_quantity_lower_bound == 0 for row in rows)
    assert all(row.fill_quantity_upper_bound in {0, 1} for row in rows)
    assert all("FILL" in row.fill_bound_reason or "MARKETABLE" in row.fill_bound_reason for row in rows)
