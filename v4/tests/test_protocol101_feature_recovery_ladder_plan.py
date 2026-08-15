from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.scripts.run_protocol101_feature_recovery_ladder_plan import (
    CONTRACT,
    TRANSFORM,
    build_payload,
    ladder_groups,
    render_report,
)
from v4.scripts.run_protocol101_stage1_bounded_hgb_search import (
    preregistration_payload,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _args(tmp_path: Path) -> argparse.Namespace:
    steering_doc = tmp_path / "steering.md"
    training_doc = tmp_path / "training.md"
    gates_doc = tmp_path / "gates.md"
    parity_doc = tmp_path / "parity.md"
    for path in (steering_doc, training_doc, gates_doc, parity_doc):
        path.write_text("ok\n")
    runner_plan = _write_json(
        tmp_path / "runner_plan.json",
        {
            "selected_feature_contract": CONTRACT,
            "model_scoring_feature_transform": TRANSFORM,
            "paper_submit_allowed": False,
        },
    )
    training_scope = _write_json(
        tmp_path / "training_scope.json",
        {
            "status": "pass",
            "session_count": 301,
            "pass_count": 301,
            "registry_hash": "abc123",
        },
    )
    return argparse.Namespace(
        out_dir=tmp_path / "out",
        steering_doc=steering_doc,
        training_doc=training_doc,
        stage1_gates_doc=gates_doc,
        runner_plan=runner_plan,
        training_scope=training_scope,
        parity_cert=parity_doc,
    )


def test_ladder_keeps_masked_baseline_as_control_not_ceiling() -> None:
    groups = ladder_groups()

    assert [group["order"] for group in groups] == list(range(7))
    assert groups[0]["feature_group"] == "masked_v2_baseline_control"
    assert groups[0]["status"] == "active_control"
    assert groups[0]["may_skip_parity_gate"] is True
    assert groups[-1]["feature_group"] == "raw_vendor_quote_microstructure"
    assert groups[-1]["current_role"] == "guards_fills_labels_pnl_audit_only"
    assert groups[-1]["status"] == "exceptional_addback_only"


def test_feature_addback_requires_parity_and_uplift(tmp_path: Path) -> None:
    payload = build_payload(_args(tmp_path))

    assert payload["status"] == "pass"
    assert payload["feature_contract"] == CONTRACT
    assert payload["model_facing_transform"] == TRANSFORM
    assert payload["masked_baseline_is_control_not_ceiling"] is True
    assert payload["raw_fields_preserved_for_market_mechanics"] is True
    assert payload["feature_addback_policy"] == "parity_plus_uplift_required"
    required = payload["gate_policy"]["feature_addback_requires"]
    assert "paired_ibkr_vs_historical_parity_report" in required
    assert "out_of_sample_cv_uplift_after_fees_and_stress" in required
    assert "add_feature_group_because_masked_baseline_pnl_is_weak" in payload["gate_policy"]["forbidden_shortcuts"]
    assert payload["side_effect_policy"]["broker_endpoint_called"] is False
    assert payload["side_effect_policy"]["paper_submit_allowed"] is False


def test_payload_blocks_wrong_runner_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.runner_plan.write_text(
        json.dumps(
            {
                "selected_feature_contract": "protocol101-live-v1",
                "model_scoring_feature_transform": TRANSFORM,
                "paper_submit_allowed": False,
            }
        )
        + "\n"
    )

    payload = build_payload(args)

    assert payload["status"] == "blocked"
    assert "runner_plan_contract_not_v2_microstructure_masked" in payload["blockers"]


def test_report_names_ladder_and_forbidden_shortcuts(tmp_path: Path) -> None:
    payload = build_payload(_args(tmp_path))
    report = render_report(payload)

    assert "Protocol101 Feature Recovery Ladder Plan" in report
    assert "`internally_computed_greeks_and_iv`" in report
    assert "`add_feature_group_because_masked_baseline_pnl_is_weak`" in report


def test_stage1_preregistration_embeds_feature_recovery_policy(tmp_path: Path) -> None:
    design = _write_json(
        tmp_path / "design.json",
        {
            "allowed_data": {
                "included_first_session": "2025-01-02",
                "included_last_session": "2026-03-31",
                "canonical_manifest": "manifest.json",
            },
            "split_policy": {
                "schema_version": "Protocol101ChronologicalExpandingWindowSplitPolicyV2",
                "required_expanding_window_cv": True,
                "fold_count": 5,
                "folds": [{"fold_id": "fold_01", "train_sessions": [], "validation_sessions": []}],
            },
        },
    )
    runner_plan = _write_json(
        tmp_path / "runner_plan.json",
        {"expanding_folds": {"fold_sessions": {"fold_01": {"train": [], "validation": []}}}},
    )
    training_scope = _write_json(
        tmp_path / "training_scope.json",
        {
            "registry_hash": "abc123",
            "session_count": 301,
            "pass_count": 301,
            "fail_count": 0,
            "report_only_count": 0,
        },
    )
    gates_doc = tmp_path / "gates.md"
    feature_recovery_plan = tmp_path / "feature_recovery.md"
    protected = tmp_path / "protected.json"
    full_acceptance = tmp_path / "full_acceptance.json"
    for path in (gates_doc, feature_recovery_plan, protected, full_acceptance):
        path.write_text("ok\n")
    args = argparse.Namespace(
        design=design,
        runner_plan=runner_plan,
        training_scope_registry=training_scope,
        full_acceptance=full_acceptance,
        protected_holdout=protected,
        gates_doc=gates_doc,
        feature_recovery_plan=feature_recovery_plan,
        out_dir=tmp_path / "out",
    )

    payload = preregistration_payload(args)

    assert payload["feature_recovery_policy"]["masked_v2_baseline_is_control_not_ceiling"] is True
    assert payload["feature_recovery_policy"]["feature_addback_policy"] == "parity_plus_uplift_required"
    assert payload["feature_recovery_policy"]["masked_feature_groups_may_not_be_restored_because_pnl_is_weak"] is True
    assert "feature_recovery_plan" in payload["artifacts"]
