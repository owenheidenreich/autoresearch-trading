from __future__ import annotations

from pathlib import Path

from v4.scripts.build_protocol101_training_scope_acceptance_registry import (
    build_training_scope_registry,
)
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import compute_registry_hash


def _record(session: str, status: str) -> dict:
    return {
        "session": session,
        "status": status,
        "verifier_version": 35,
        "early_close_session": False,
        "report_only_reason": "missing_index_context" if status == "report_only" else "",
        "checks": {"context_lag_exact_one_minute": status != "fail"},
        "processed": {
            "processed_exists": True,
            "neural_rows": 359,
            "processed_path": f"data/{session}.pkl",
            "processed_sha256": f"sha-{session}",
            "feature_contract_version": "protocol101-live-v2-microstructure-masked",
        },
        "label_spot_check": {"outcome_reasons": {"target_hit": 1, "stop_hit": 1}},
    }


def _source_registry() -> dict:
    payload = {
        "schema_version": "Protocol101OwnedRawAcceptanceRegistryV3_5",
        "verifier_version": 35,
        "minimum_fold_placement_verifier_version": 35,
        "status": "fail",
        "batch_id": "owned_raw_acceptance_fixture",
        "evidence_grade": "data_plane_only",
        "start_date": "2025-01-02",
        "end_date": "2025-01-06",
        "session_count": 3,
        "pass_count": 1,
        "report_only_count": 1,
        "fail_count": 1,
        "thresholds": {},
        "thresholds_are_defaults": True,
        "fee_model": {"fee_model": "gross_no_fees"},
        "max_quote_age_seconds": 60,
        "no_bid_convention": "bid_zero_blocks_exit",
        "label_policies": [],
        "forced_flat_before_et": "15:55",
        "cbbo_stamping_assumption": "fixture",
        "governance_checks": {"era_manifest_pass": True, "role_policy_pass": True},
        "batch_checks": {"governance_artifacts_pass": True, "monthly_label_outcome_coverage": True},
        "batch_label_outcome_reasons": {"target_hit": 3},
        "labels_used_for_strategy_selection": False,
        "pnl_used_for_strategy_selection": False,
        "strategy_metrics_used": False,
        "era_manifest_hash": "era-hash",
        "role_policy_hash": "role-hash",
        "sessions": [
            _record("2025-01-02", "pass"),
            _record("2025-01-03", "report_only"),
            _record("2025-01-06", "fail"),
        ],
        "placement_predicates": [],
    }
    payload["registry_hash"] = compute_registry_hash(payload)
    return payload


def test_build_training_scope_registry_filters_to_pass_sessions() -> None:
    era_manifest = {
        "status": "pass",
        "sessions": [
            {"session": "2025-01-02", "era": "pre_program_oct2024_jun2025"},
            {"session": "2025-01-03", "era": "pre_program_oct2024_jun2025"},
            {"session": "2025-01-06", "era": "pre_program_oct2024_jun2025"},
        ],
    }
    role_policy = {
        "status": "pass",
        "policy": {
            "pre_program_oct2024_jun2025": {
                "permitted_roles": ["train", "test", "diagnostics_only"]
            }
        },
    }

    scoped = build_training_scope_registry(
        source_registry=_source_registry(),
        source_registry_path=Path("source.json"),
        era_manifest=era_manifest,
        era_manifest_path=Path("era.json"),
        role_policy=role_policy,
        role_policy_path=Path("role.json"),
    )

    assert scoped["status"] == "pass"
    assert scoped["session_count"] == 1
    assert [record["session"] for record in scoped["sessions"]] == ["2025-01-02"]
    assert scoped["training_scope_filter"]["excluded_session_count"] == 2
    assert scoped["governance_checks"]["training_scope_excluded_fail_count"] == 1
    assert scoped["governance_checks"]["training_scope_excluded_report_only_count"] == 1
    assert compute_registry_hash(scoped) == scoped["registry_hash"]
    assert scoped["placement_predicates"][0]["placeable"] is True
