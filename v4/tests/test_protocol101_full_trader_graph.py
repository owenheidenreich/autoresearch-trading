from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from v4.scripts.run_protocol101_full_trader_graph import (
    DEFAULT_GRAPH_PATH,
    ENTRY_NODE,
    GRAPH_SCHEMA,
    PARKED_STATUS,
    READY_STATUS,
    RECEIPT_SCHEMA,
    REQUIRED_SCOPE_ASSERTIONS,
    START_AUTHORIZATION_SCHEMA,
    GraphContractError,
    advance_state,
    authorize_node_start,
    build_initial_state,
    dry_run_routes,
    initialize_state_dir,
    read_json,
    stable_hash,
    validate_graph,
    validate_state,
)


def _graph() -> dict:
    return read_json(DEFAULT_GRAPH_PATH)


def _receipt(graph: dict, state: dict, *, outcome: str) -> dict:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "graph_sha256": state["graph_sha256"],
        "product_contract_sha256": state["product_contract_sha256"],
        "node_id": state["current_node"],
        "outcome": outcome,
        "producer_id": "designer-agent",
        "independent_reviewer_id": None,
        "owner_approved": False,
        "scope_assertions": dict(REQUIRED_SCOPE_ASSERTIONS),
        "artifact_hashes": {"design.md": "a" * 64},
    }


def _start_authorization(state: dict) -> dict:
    return {
        "schema_version": START_AUTHORIZATION_SCHEMA,
        "graph_sha256": state["graph_sha256"],
        "product_contract_sha256": state["product_contract_sha256"],
        "node_id": state["current_node"],
        "authorized": True,
        "owner_identity": "test-owner",
        "scope_assertions": dict(REQUIRED_SCOPE_ASSERTIONS),
    }


def test_graph_is_valid_and_uses_full_trader_product_contract() -> None:
    graph = _graph()
    result = validate_graph(graph)
    assert graph["schema_version"] == GRAPH_SCHEMA
    assert result["valid"] is True
    assert result["node_count"] == result["reachable_node_count"]
    assert graph["product_contract"]["flat_state_learned_actions"] == [
        "WAIT",
        "BUY_SPECIFIC_ELIGIBLE_CALL_OR_PUT_CONTRACT",
    ]
    assert graph["product_contract"]["open_state_learned_actions"] == [
        "HOLD",
        "EXIT",
    ]


def test_initial_state_is_parked_before_full_ladder_design() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    assert validate_state(state, graph)["valid"] is True
    assert state["current_node"] == ENTRY_NODE
    assert state["execution_status"] == PARKED_STATUS
    assert state["owner_start_required"] is True
    assert state["training_executed"] is False
    assert state["commands_executed"] == []
    assert state["protected_resources_spent"] == []


def test_route_dry_run_checks_every_edge_without_execution() -> None:
    graph = _graph()
    result = dry_run_routes(graph)
    assert result["all_routes_valid"] is True
    assert result["routes_checked"] == len(graph["edges"])
    assert result["commands_executed"] == []
    assert result["training_executed"] is False
    assert result["protected_resource_accessed"] is False
    assert result["broker_contacted"] is False
    assert result["paper_order_submitted"] is False


def test_initialize_writes_parked_packet(tmp_path: Path) -> None:
    graph = _graph()
    state = initialize_state_dir(graph, tmp_path, force=False)
    assert state["current_node"] == ENTRY_NODE
    assert read_json(tmp_path / "state.json")["current_node"] == ENTRY_NODE
    assert read_json(tmp_path / "route_dry_run.json")["all_routes_valid"] is True
    receipt = read_json(tmp_path / "reset_receipt.json")
    assert receipt["training_executed"] is False
    assert receipt["next_node"] == ENTRY_NODE


def test_product_drift_receipt_is_rejected() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    state = authorize_node_start(state, _start_authorization(state), graph)
    receipt = _receipt(graph, state, outcome="design_ready")
    receipt["scope_assertions"]["strike_not_permanently_hard_coded"] = False
    with pytest.raises(GraphContractError, match="product_drift_detected"):
        advance_state(state, receipt, graph)


def test_valid_design_receipt_advances_only_to_review() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    state = authorize_node_start(state, _start_authorization(state), graph)
    receipt = _receipt(graph, state, outcome="design_ready")
    updated = advance_state(state, receipt, graph)
    assert updated["current_node"] == "FT-20-PARALLEL-DESIGN-REVIEW"
    assert updated["execution_status"] == READY_STATUS
    assert updated["commands_executed"] == []
    assert updated["transition_count"] == 1


def test_unknown_outcome_cannot_invent_a_route() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    state = authorize_node_start(state, _start_authorization(state), graph)
    receipt = _receipt(graph, state, outcome="silently_use_P5")
    with pytest.raises(GraphContractError, match="not_an_allowed_unique_route"):
        advance_state(state, receipt, graph)


def test_independent_reviewer_must_differ_from_producer() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    state = authorize_node_start(state, _start_authorization(state), graph)
    first = _receipt(graph, state, outcome="design_ready")
    review_state = advance_state(state, first, graph)
    receipt = _receipt(graph, review_state, outcome="pass")
    receipt["producer_id"] = "same-agent"
    receipt["reviewer_ids"] = {
        "trading_realism_reviewer": "same-agent",
        "ml_statistics_reviewer": "ml-reviewer",
        "live_parity_reviewer": "live-reviewer",
    }
    with pytest.raises(GraphContractError, match="may_not_independently_accept"):
        advance_state(review_state, receipt, graph)


def test_loop_budget_fails_closed() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    state["current_node"] = "FT-20-PARALLEL-DESIGN-REVIEW"
    state["execution_status"] = READY_STATUS
    state["owner_start_required"] = False
    state["loop_counts"]["design_repair"] = 2
    receipt = _receipt(graph, state, outcome="repairable_design_defect")
    receipt["producer_id"] = "review-packet-producer"
    receipt["reviewer_ids"] = {
        "trading_realism_reviewer": "trading-reviewer",
        "ml_statistics_reviewer": "ml-reviewer",
        "live_parity_reviewer": "live-reviewer",
    }
    with pytest.raises(GraphContractError, match="loop_budget_exhausted"):
        advance_state(state, receipt, graph)


def test_owner_gate_requires_owner_approval() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    state["current_node"] = "FT-21-OWNER-DESIGN-APPROVAL"
    state["execution_status"] = READY_STATUS
    state["owner_start_required"] = False
    receipt = _receipt(graph, state, outcome="approved")
    with pytest.raises(GraphContractError, match="owner_approval_required"):
        advance_state(state, receipt, graph)


def test_graph_hash_changes_if_product_is_narrowed() -> None:
    graph = _graph()
    original = stable_hash(graph)
    changed = deepcopy(graph)
    changed["product_contract"]["flat_state_learned_actions"] = ["WAIT"]
    assert stable_hash(changed) != original
    with pytest.raises(
        GraphContractError,
        match="full_ladder_specific_contract_action_missing",
    ):
        validate_graph(changed)


def test_parked_design_cannot_advance_without_owner_start() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    receipt = _receipt(graph, state, outcome="design_ready")
    with pytest.raises(
        GraphContractError,
        match="node_not_authorized_for_external_goal",
    ):
        advance_state(state, receipt, graph)


def test_owner_start_authorization_unlocks_only_current_node() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    authorization = _start_authorization(state)
    updated = authorize_node_start(state, authorization, graph)
    assert updated["current_node"] == ENTRY_NODE
    assert updated["execution_status"] == READY_STATUS
    assert updated["owner_start_required"] is False
    assert len(updated["start_authorizations"]) == 1


def test_parallel_review_requires_all_three_reviewers() -> None:
    graph = _graph()
    state = build_initial_state(graph)
    state = authorize_node_start(state, _start_authorization(state), graph)
    state = advance_state(state, _receipt(graph, state, outcome="design_ready"), graph)
    receipt = _receipt(graph, state, outcome="pass")
    receipt["independent_reviewer_id"] = "one-reviewer"
    with pytest.raises(
        GraphContractError,
        match="parallel_reviewer_ids_not_exact",
    ):
        advance_state(state, receipt, graph)


def test_closed_goals_are_archived_with_matching_hashes() -> None:
    training_root = DEFAULT_GRAPH_PATH.parents[1]
    active_goals = training_root / "goals"
    archived = (
        training_root
        / "history/closed_stage1_graph_and_goals_2026_07_28/goals"
    )
    assert sorted(path.name for path in active_goals.iterdir()) == ["README.md"]
    prompts = sorted(archived.glob("*.md"))
    checksums = sorted(archived.glob("*.sha256"))
    assert len(prompts) == 27
    assert len(checksums) == 27
    for prompt in prompts:
        expected = (archived / f"{prompt.name}.sha256").read_text().split()[0]
        actual = hashlib.sha256(prompt.read_bytes()).hexdigest()
        assert actual == expected


def test_closed_stage1_graph_bytes_are_preserved() -> None:
    archived_graph = (
        DEFAULT_GRAPH_PATH.parents[1]
        / "history/closed_stage1_graph_and_goals_2026_07_28/graph/"
        "PROTOCOL101_STAGE1_AUTORESEARCH_GRAPH_2026_07_25.md"
    )
    assert hashlib.sha256(archived_graph.read_bytes()).hexdigest() == (
        "c30511f28542b25525cff84cec51e7b28d8d14f7d8f31ef03cf52ba15c350448"
    )
