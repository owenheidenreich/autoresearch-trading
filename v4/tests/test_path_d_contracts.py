from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from v4.path_d.contracts import (
    BROKER_STATE_SNAPSHOT_V1_JSON_SCHEMA,
    CANONICAL_MARKET_EVENT_V1_JSON_SCHEMA,
    EXECUTION_EVENT_V1_JSON_SCHEMA,
    EXECUTION_INTENT_V1_JSON_SCHEMA,
    FEATURE_SNAPSHOT_V1_JSON_SCHEMA,
    GOVERNOR_DECISION_V1_JSON_SCHEMA,
    BrokerStateSnapshotV1,
    CanonicalMarketEventV1,
    ExecutionEventV1,
    ExecutionIntentV1,
    FeatureSnapshotV1,
    GovernorDecisionV1,
    IntentClocksV1,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "v4/path_d/contracts/fixtures"


@pytest.mark.parametrize(
    ("filename", "contract"),
    [
        ("execution_intent_v1.json", ExecutionIntentV1),
        ("governor_decision_v1.json", GovernorDecisionV1),
        ("execution_event_v1.json", ExecutionEventV1),
        ("broker_state_snapshot_v1.json", BrokerStateSnapshotV1),
        ("canonical_market_event_v1.json", CanonicalMarketEventV1),
        ("feature_snapshot_v1.json", FeatureSnapshotV1),
    ],
)
def test_golden_contract_fixtures_round_trip_and_reject_extra_fields(filename, contract) -> None:
    payload = json.loads((FIXTURES / filename).read_text())
    parsed = contract.from_dict(payload)
    assert json.loads(parsed.to_json()) == payload
    with pytest.raises(ValueError, match="extras"):
        contract.from_dict({**payload, "unexpected": True})


def test_intent_semantic_hash_excludes_trace_and_measurement_timing_but_not_limit() -> None:
    original = ExecutionIntentV1.from_dict(json.loads((FIXTURES / "execution_intent_v1.json").read_text()))
    clocks = original.clocks
    changed_measurement = IntentClocksV1(
        **{
            **clocks.to_dict(),
            "model_started_at_utc": "2026-06-30T13:30:02.100000Z",
            "model_finished_at_utc": "2026-06-30T13:30:02.200000Z",
            "intent_emitted_at_utc": "2026-06-30T13:30:02.300000Z",
        }
    )
    same_semantics = ExecutionIntentV1.create(
        trace_id="different-trace",
        parent_intent_id=original.parent_intent_id,
        origin=original.origin,
        producer=original.producer,
        decision=original.decision,
        contract=original.contract,
        price_budget=original.price_budget,
        clocks=changed_measurement,
        state_precondition=original.state_precondition,
        execution_profile_version=original.execution_profile_version,
    )
    assert same_semantics.intent_id == original.intent_id
    changed_limit = type(original.price_budget)(
        **{**original.price_budget.to_dict(), "hard_limit_micros": original.price_budget.hard_limit_micros - 1}
    )
    different_semantics = ExecutionIntentV1.create(
        trace_id=original.trace_id,
        parent_intent_id=original.parent_intent_id,
        origin=original.origin,
        producer=original.producer,
        decision=original.decision,
        contract=original.contract,
        price_budget=changed_limit,
        clocks=original.clocks,
        state_precondition=original.state_precondition,
        execution_profile_version=original.execution_profile_version,
    )
    assert different_semantics.intent_id != original.intent_id


def test_all_contract_schema_objects_are_closed() -> None:
    schemas = (
        EXECUTION_INTENT_V1_JSON_SCHEMA,
        GOVERNOR_DECISION_V1_JSON_SCHEMA,
        EXECUTION_EVENT_V1_JSON_SCHEMA,
        BROKER_STATE_SNAPSHOT_V1_JSON_SCHEMA,
        CANONICAL_MARKET_EVENT_V1_JSON_SCHEMA,
        FEATURE_SNAPSHOT_V1_JSON_SCHEMA,
    )
    for schema in schemas:
        _assert_closed_objects(schema)


def _assert_closed_objects(value) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_closed_objects(child)
    elif isinstance(value, list):
        for child in value:
            _assert_closed_objects(child)


def test_contract_package_imports_only_standard_library_or_relative_modules() -> None:
    contract_root = ROOT / "v4/path_d/contracts"
    allowed = {
        "__future__", "dataclasses", "datetime", "hashlib", "json", "math", "typing",
    }
    for path in contract_root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names), path
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] in allowed, path


def test_path_d_import_boundaries_forbid_ibkr_from_decision_and_ib_libraries_elsewhere() -> None:
    path_d = ROOT / "v4/path_d"
    for path in path_d.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        if "decision" in path.parts:
            assert not any("ibkr" in item.lower() or "broker" in item.lower() or item.startswith(("ib_insync", "ibapi")) for item in imports), path
        if path.name != "ibkr_adapter.py":
            assert not any(item.startswith(("ib_insync", "ibapi")) for item in imports), path
