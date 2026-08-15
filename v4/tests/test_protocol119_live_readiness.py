from __future__ import annotations

from v4.scripts.run_protocol119_protocol101_live_readiness import (
    apply_surface_edge_portability,
    apply_protocol101_entry_router_smoke,
    audit_live_feature_dependencies,
    decide_live_readiness,
    next_requirements,
)


def test_protocol119_feature_audit_blocks_upstream_edge_features() -> None:
    audit = audit_live_feature_dependencies(["entry_bid", "edge", "hist_prev_max_edge"])

    assert audit["status"] == "blocked_missing_upstream_edge_generator"
    assert audit["upstream_edge_features"] == ["edge", "hist_prev_max_edge"]
    assert audit["unknown_features"] == []


def test_protocol119_decision_prioritizes_ibkr_connection_blocker() -> None:
    decision = decide_live_readiness(
        artifact={"loadable": True},
        protocol118={"decision": "pass_historical_no_order_shadow_rehearsal_live_capture_next"},
        ibkr={"decision": "blocked", "blocked_reason": "ibkr_connection_failed"},
        feature_audit={"status": "blocked_missing_upstream_edge_generator"},
    )

    assert decision == "blocked_ibkr_connection_failed"


def test_protocol119_next_requirements_explain_ibkr_and_edge_blockers() -> None:
    requirements = next_requirements(
        "blocked_ibkr_connection_failed",
        {"status": "blocked_missing_upstream_edge_generator"},
        {"decision": "blocked"},
    )

    assert any("IB Gateway" in item for item in requirements)
    assert any("edge" in item for item in requirements)


def test_protocol119_surface_portability_changes_edge_blocker_to_wiring() -> None:
    audit = apply_surface_edge_portability(
        audit_live_feature_dependencies(["entry_bid", "edge"]),
        {"decision": "pass_surface_edge_generator_loads_offline_live_router_next", "summary": {"finite_edge_fraction": 1.0}},
    )

    assert audit["status"] == "blocked_upstream_edge_generator_not_wired"
    assert audit["surface_edge_generator_loadable"] is True


def test_protocol119_protocol101_router_smoke_clears_edge_blocker() -> None:
    audit = apply_protocol101_entry_router_smoke(
        {"status": "blocked_upstream_edge_generator_not_wired"},
        {"decision": "pass_protocol101_entry_router_edge_wired", "summary": {"candidate_feature_rows": 10}},
    )

    assert audit["status"] == "pass"
    assert audit["protocol101_entry_router_edge_wired"] is True


def test_protocol119_ready_when_all_gates_pass() -> None:
    decision = decide_live_readiness(
        artifact={"loadable": True},
        protocol118={"decision": "pass_historical_no_order_shadow_rehearsal_live_capture_next"},
        ibkr={"decision": "pass"},
        feature_audit={"status": "pass"},
    )

    assert decision == "ready_for_protocol101_no_order_live_capture"
