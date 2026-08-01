"""Tests for the guarded Protocol101 fair-contract training runner."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.model.protocol101_governed_loader import GovernedLoaderArtifacts, file_sha256
from v4.scripts.build_protocol101_protected_holdout_artifact import build_artifact
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import compute_registry_hash
from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
from v4.model.supervised_pilot import (
    DecisionCandidates,
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    PilotConfig,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    POLICY_META,
    _attach_teacher_labels,
    _average_predictions,
    _balanced_classifier_sample_weight,
    _flatten_training_examples_with_rights,
    _split_fit_and_calibration_decisions,
    _targets_for_decision,
    build_runner_plan,
    choose_threshold_with_rule,
    parse_ensemble_seeds,
    paths_by_split,
)
from v4.scripts import run_protocol101_fair_contract_training_runner as runner_module


def _sessions() -> list[str]:
    return [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-03-03",
    ]


def _folds() -> list[dict]:
    sessions = _sessions()
    return [
        {
            "fold_index": idx,
            "fold_id": f"expanding_fold_{idx:02d}",
            "train_sessions": sessions[:idx],
            "validation_sessions": [sessions[idx + 1]],
            "embargoed_sessions": [
                {
                    "session": sessions[idx],
                    "reason": "one_trading_session_embargo",
                }
            ],
        }
        for idx in range(1, 6)
    ]


def _manifest() -> dict:
    return {
        "included_sessions": [
            {
                "session": session,
                "processed_file": f"/tmp/protocol101/{session}.pkl",
            }
            for session in _sessions()
        ]
    }


def _design() -> dict:
    return {
        "selected_feature_contract": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
        "paper_submit_allowed": False,
        "allowed_data": {
            "canonical_manifest": "/tmp/protocol101/manifest.json",
            "require_manifest_loading": True,
            "glob_loading_allowed": False,
        },
        "split_policy": {
            "required_expanding_window_cv": True,
            "required_fold_count": 5,
            "fold_count": 5,
            "folds": _folds(),
            "train_sessions": ["2026-01-02"],
            "validation_sessions": ["2026-01-05"],
            "diagnostic_test_sessions": ["2026-03-03"],
            "embargoed_sessions": [
                {
                    "session": "2026-03-02",
                    "reason": "one_trading_session_embargo",
                }
            ],
        },
    }


def _governed_manifest_and_artifacts(tmp_path: Path) -> tuple[dict, GovernedLoaderArtifacts]:
    sessions = _sessions()
    manifest_sessions = []
    records = []
    for session in sessions:
        path = tmp_path / f"{session}.pkl"
        path.write_bytes(f"{session}\n".encode())
        manifest_sessions.append({"session": session, "processed_file": str(path)})
        records.append(
            {
                "session": session,
                "status": "pass",
                "verifier_version": 35,
                "early_close_session": False,
                "processed": {
                    "processed_exists": True,
                    "neural_rows": 360,
                    "processed_path": str(path),
                    "processed_sha256": file_sha256(path),
                },
            }
        )
    acceptance = {
        "status": "pass",
        "verifier_version": 35,
        "thresholds_are_defaults": True,
        "fee_model": {"fee_model": "gross_no_fees", "fee_per_contract": 0.0},
        "sessions": records,
    }
    acceptance["registry_hash"] = compute_registry_hash(acceptance)
    era_manifest = {
        "status": "pass",
        "sessions": [
            {"session": session, "era": "q1_2026_development"}
            for session in sessions
        ],
    }
    role_policy = {
        "status": "pass",
        "policy": {
            "q1_2026_development": {"permitted_roles": ["train", "test", "diagnostics_only"]},
        },
    }
    return {"included_sessions": manifest_sessions}, GovernedLoaderArtifacts(
        acceptance_registry_path=tmp_path / "acceptance.json",
        acceptance_registry=acceptance,
        era_manifest_path=tmp_path / "era.json",
        era_manifest=era_manifest,
        role_policy_path=tmp_path / "policy.json",
        role_policy=role_policy,
        protected_holdout_path=tmp_path / "protected_holdout.json",
        protected_holdout=build_artifact(sessions=["2099-12-31"], owner_note="unit-test placeholder"),
    )


def test_paths_by_split_uses_manifest_sessions_only() -> None:
    paths, blockers = paths_by_split(_design(), _manifest())

    assert blockers == []
    assert [path.name for path in paths["train"]] == ["2026-01-02.pkl"]
    assert [path.name for path in paths["validation"]] == ["2026-01-05.pkl"]
    assert [path.name for path in paths["diagnostic_test"]] == ["2026-03-03.pkl"]


def test_training_mode_requires_explicit_owner_approval_flags(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=governance,
        mode="train",
        policy_index=1,
    )

    assert plan["status"] == "blocked"
    assert plan["model_training_executed"] is False
    assert plan["broker_endpoint_called"] is False
    assert plan["paper_submit_allowed"] is False
    assert "missing_owner_approved_model_training_flag" in plan["blockers"]
    assert "missing_owner_approved_threshold_selection_flag" in plan["blockers"]
    assert "missing_owner_approval_note" in plan["blockers"]
    assert "fold_aware_training_requires_hgb_tabular_model_family" in plan["blockers"]


def test_training_mode_rejects_old_single_split_even_with_owner_flags(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    old_design = {
        **_design(),
        "split_policy": {
            "train_sessions": ["2026-01-02"],
            "validation_sessions": ["2026-01-05"],
            "diagnostic_test_sessions": ["2026-03-03"],
        },
    }
    plan = build_runner_plan(
        design=old_design,
        manifest=manifest,
        governance_artifacts=governance,
        mode="train",
        policy_index=1,
        model_family="sklearn_hist_gradient_boosting",
        owner_approved_model_training=True,
        owner_approved_threshold_selection=True,
        owner_approval_note="unit-test owner approval",
    )

    assert plan["status"] == "blocked"
    assert "expanding_window_cv_required_for_training" in plan["blockers"]


def test_training_mode_ready_with_fold_aware_hgb_and_owner_flags(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=governance,
        mode="train",
        policy_index=1,
        model_family="sklearn_hist_gradient_boosting",
        owner_approved_model_training=True,
        owner_approved_threshold_selection=True,
        owner_approval_note="unit-test owner approval",
    )

    assert plan["status"] == "ready_to_train"
    assert plan["blockers"] == []


def test_dry_run_plan_is_ready_without_training_authorization(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=governance,
        mode="dry-run",
        policy_index=1,
    )

    assert plan["status"] == "dry_run_ready"
    assert plan["decision"] == "manifest_and_split_ready_no_training_executed"
    assert plan["model_training_executed"] is False
    assert plan["threshold_selection_executed"] is False
    assert plan["model_scoring_feature_transform"] == (
        FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE
    )
    assert plan["split_sessions"]["train"] == ["2026-01-02"]
    assert plan["governance"]["governance_hash"]
    assert sorted(plan["expanding_folds"]["fold_sessions"]) == [
        "expanding_fold_01",
        "expanding_fold_02",
        "expanding_fold_03",
        "expanding_fold_04",
        "expanding_fold_05",
    ]
    assert plan["expanding_folds"]["fold_sessions"]["expanding_fold_05"]["validation"] == [
        "2026-01-12"
    ]


def test_dry_run_blocks_protected_holdout_inside_any_fold(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    protected_governance = replace(
        governance,
        protected_holdout=build_artifact(
            sessions=["2026-01-06"],
            owner_note="unit-test protected fold session",
        ),
    )
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=protected_governance,
        mode="dry-run",
        policy_index=1,
        model_family="sklearn_hist_gradient_boosting",
    )

    assert plan["status"] == "blocked"
    assert any("fold_session_not_placeable:expanding_fold_01:validation:2026-01-06" in item for item in plan["blockers"])
    assert any("2026-01-06:protected_holdout_session_requested" in item for item in plan["blockers"])


def test_dry_run_blocks_report_only_session_inside_any_fold(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    acceptance = dict(governance.acceptance_registry)
    sessions = []
    for row in acceptance["sessions"]:
        updated = dict(row)
        if updated["session"] == "2026-01-06":
            updated["status"] = "report_only"
        sessions.append(updated)
    acceptance["sessions"] = sessions
    acceptance["registry_hash"] = compute_registry_hash(acceptance)
    blocked_governance = replace(governance, acceptance_registry=acceptance)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=blocked_governance,
        mode="dry-run",
        policy_index=1,
        model_family="sklearn_hist_gradient_boosting",
    )

    assert plan["status"] == "blocked"
    assert any("fold_session_not_placeable:expanding_fold_01:validation:2026-01-06" in item for item in plan["blockers"])
    assert any("expanding_fold_01:2026-01-06:session_acceptance_not_pass" in item for item in plan["blockers"])


def test_expanding_fold_plan_keeps_embargoed_sessions_out_of_fold_inputs(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=governance,
        mode="dry-run",
        policy_index=1,
        model_family="sklearn_hist_gradient_boosting",
    )

    fold_01 = plan["expanding_folds"]["fold_sessions"]["expanding_fold_01"]
    assert "2026-01-05" not in fold_01["train"]
    assert "2026-01-05" not in fold_01["validation"]
    assert fold_01["train"] == ["2026-01-02"]
    assert fold_01["validation"] == ["2026-01-06"]


def test_fold_aware_training_dispatches_all_five_folds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=governance,
        mode="train",
        policy_index=1,
        model_family="sklearn_hist_gradient_boosting",
        owner_approved_model_training=True,
        owner_approved_threshold_selection=True,
        owner_approval_note="unit-test owner approval",
    )
    calls: list[dict] = []

    def fake_single_split_training(fold_plan: dict, args) -> dict:
        calls.append(fold_plan["split_sessions"])
        return {
            "config": {"policy_index": int(args.policy_index), "feature_transform": str(args.feature_transform)},
            "experiment": {"model_family": str(args.model_family)},
            "model_out": str(args.model_out),
            "chosen_threshold": 1.0 + len(calls),
            "split_summary": {
                "train": {"sessions": len(fold_plan["split_sessions"]["train"]), "decisions": 1, "candidates": 1},
                "validation": {"sessions": len(fold_plan["split_sessions"]["validation"]), "decisions": 1, "candidates": 1},
                "diagnostic_test": {"sessions": 0, "decisions": 0, "candidates": 0},
            },
            "fit_calibration_summary": {},
            "training_preview": {},
            "training_history": [],
            "model_family_preview": {},
            "neural": {
                "validation": {
                    "metrics": {
                        "trades": 2,
                        "total_pnl": 100.0,
                        "profit_factor": 2.0,
                        "max_drawdown": -10.0,
                    }
                }
            },
            "baselines": {},
        }

    monkeypatch.setattr(runner_module, "run_single_split_training", fake_single_split_training)
    args = argparse.Namespace(
        policy_index=1,
        model_family="sklearn_hist_gradient_boosting",
        target_mode="return_on_premium_regression",
        target_clip=5.0,
        threshold_rule="max_validation_stressed_pnl",
        threshold_stress_per_trade=20.0,
        feature_transform=FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
        model_out=tmp_path / "fold_index.pt",
    )

    result = runner_module.run_training(plan, args)

    assert result["execution_mode"] == "fold_aware_expanding_cv"
    assert result["fold_count"] == 5
    assert len(calls) == 5
    assert calls[0]["train"] == ["2026-01-02"]
    assert calls[0]["validation"] == ["2026-01-06"]
    assert calls[-1]["train"] == [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]
    assert calls[-1]["validation"] == ["2026-01-12"]
    assert result["fold_summary"]["aggregate_validation_metrics"]["trades"] == 10


def test_runner_policy_meta_includes_all_trade_shape_menu_v2_labels(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=governance,
        mode="dry-run",
        policy_index=6,
    )

    assert sorted(POLICY_META) == list(range(7))
    assert POLICY_META[3] == ("ask_to_bid_stop50_target200_hold90m", 90)
    assert POLICY_META[6] == ("ask_to_bid_stop100_target9900_hold384m", 384)
    assert plan["status"] == "dry_run_ready"
    assert plan["policy_name"] == "ask_to_bid_stop100_target9900_hold384m"
    assert plan["cooldown_minutes"] == 384


def test_v2_runner_blocks_non_certified_model_scoring_transform(tmp_path: Path) -> None:
    manifest, governance = _governed_manifest_and_artifacts(tmp_path)
    plan = build_runner_plan(
        design=_design(),
        manifest=manifest,
        governance_artifacts=governance,
        mode="dry-run",
        policy_index=1,
        feature_transform="none",
    )

    assert plan["status"] == "blocked"
    assert "unexpected_model_scoring_feature_transform" in plan["blockers"]


def test_jan_fit_feb_calibration_split_uses_train_sessions_only() -> None:
    decisions = [
        DecisionCandidates(
            session=session,
            decision_time=datetime(2026, int(session[5:7]), 2, 14, 31, tzinfo=timezone.utc),
            features=np.ones((1, 2), dtype=np.float32),
            labels=np.asarray([10.0], dtype=np.float32),
            offsets=np.asarray([0.0], dtype=np.float32),
            rights=np.asarray(["C"], dtype=object),
            market_last=np.zeros(7, dtype=np.float32),
        )
        for session in ("2026-01-02", "2026-01-05", "2026-02-02")
    ]

    fit, calibration, summary = _split_fit_and_calibration_decisions(
        decisions,
        fit_mode="jan_fit_feb_calibration",
    )

    assert [decision.session for decision in fit] == ["2026-01-02", "2026-01-05"]
    assert [decision.session for decision in calibration] == ["2026-02-02"]
    assert summary["fallback_used"] is False


def test_train_tail20_calibration_uses_chronological_train_tail_only() -> None:
    decisions = [
        DecisionCandidates(
            session=session,
            decision_time=datetime.fromisoformat(f"{session}T14:31:00+00:00"),
            features=np.ones((1, 2), dtype=np.float32),
            labels=np.asarray([10.0], dtype=np.float32),
            offsets=np.asarray([0.0], dtype=np.float32),
            rights=np.asarray(["C"], dtype=object),
            market_last=np.zeros(7, dtype=np.float32),
        )
        for session in [
            "2025-07-01",
            "2025-07-02",
            "2025-07-03",
            "2025-07-07",
            "2025-07-08",
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-02-02",
        ]
    ]

    fit, calibration, summary = _split_fit_and_calibration_decisions(
        decisions,
        fit_mode="train_tail20_calibration",
    )

    assert [decision.session for decision in calibration] == [
        "2026-01-07",
        "2026-02-02",
    ]
    assert fit[-1].session == "2026-01-06"
    assert "2026-01-07" not in {decision.session for decision in fit}
    assert summary["fallback_used"] is False


def test_return_on_premium_target_uses_premium_at_risk() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
        features=np.ones((3, 2), dtype=np.float32),
        labels=np.asarray([50.0, -25.0, 300.0], dtype=np.float32),
        offsets=np.asarray([0.0, 5.0, 10.0], dtype=np.float32),
        rights=np.asarray(["C", "P", "C"], dtype=object),
        market_last=np.zeros(7, dtype=np.float32),
        entry_asks=np.asarray([2.0, 1.0, 0.5], dtype=np.float32),
    )
    config = replace(
        PilotConfig(),
        target_mode="return_on_premium_regression",
        target_clip=5.0,
    )

    targets = _targets_for_decision(decision, config=config)

    np.testing.assert_allclose(targets, np.asarray([0.25, -0.25, 5.0], dtype=np.float32))


def test_stressed_threshold_rule_can_prefer_fewer_higher_edge_trades() -> None:
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=base_time + timedelta(minutes=idx * 30),
            features=np.ones((1, 2), dtype=np.float32),
            labels=np.asarray([label], dtype=np.float32),
            offsets=np.asarray([0.0], dtype=np.float32),
            rights=np.asarray(["C"], dtype=object),
            market_last=np.zeros(7, dtype=np.float32),
        )
        for idx, label in enumerate([5.0, 200.0, 5.0])
    ]
    predictions = [
        np.asarray([1.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        np.asarray([3.0], dtype=np.float32),
    ]
    config = replace(PilotConfig(), min_validation_trades=1, cooldown_minutes=10)

    threshold, sweep = choose_threshold_with_rule(
        decisions,
        predictions,
        config=config,
        threshold_rule="max_validation_stressed_pnl",
        stress_per_trade=20.0,
    )

    assert threshold > 1.0
    chosen = [row for row in sweep if row["threshold"] == threshold][0]
    assert chosen["stressed_total_pnl"] > 0.0
    assert chosen["trades"] == 2


def test_frequency_sufficient_threshold_rule_prefers_more_coverage() -> None:
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=base_time + timedelta(minutes=idx * 12),
            features=np.ones((1, 2), dtype=np.float32),
            labels=np.asarray([25.0 if idx < 6 else -5.0], dtype=np.float32),
            offsets=np.asarray([0.0], dtype=np.float32),
            rights=np.asarray(["C"], dtype=object),
            market_last=np.zeros(7, dtype=np.float32),
        )
        for idx in range(8)
    ]
    predictions = [
        np.asarray([float(score)], dtype=np.float32)
        for score in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.1, 0.05]
    ]
    config = replace(PilotConfig(), min_validation_trades=3, cooldown_minutes=1)

    threshold, sweep = choose_threshold_with_rule(
        decisions,
        predictions,
        config=config,
        threshold_rule="frequency_sufficient_stressed",
        stress_per_trade=20.0,
    )

    chosen = [row for row in sweep if row["threshold"] == threshold][0]
    assert chosen["trades"] >= 6


def test_daily_stability_threshold_rule_uses_positive_day_fraction() -> None:
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    decisions = []
    predictions = []
    for day, labels in enumerate(([60.0, -80.0], [55.0, -75.0], [50.0, -70.0])):
        for idx, label in enumerate(labels):
            decisions.append(
                DecisionCandidates(
                    session=f"2026-01-0{day + 2}",
                    decision_time=base_time + timedelta(days=day, minutes=idx * 20),
                    features=np.ones((1, 2), dtype=np.float32),
                    labels=np.asarray([label], dtype=np.float32),
                    offsets=np.asarray([0.0], dtype=np.float32),
                    rights=np.asarray(["C"], dtype=object),
                    market_last=np.zeros(7, dtype=np.float32),
                )
            )
            predictions.append(np.asarray([0.9 if idx == 0 else 0.1], dtype=np.float32))
    config = replace(PilotConfig(), min_validation_trades=2, cooldown_minutes=1)

    threshold, sweep = choose_threshold_with_rule(
        decisions,
        predictions,
        config=config,
        threshold_rule="daily_stability_stressed",
        stress_per_trade=20.0,
    )

    chosen = [row for row in sweep if row["threshold"] == threshold][0]
    assert chosen["stressed_positive_day_fraction"] == 1.0
    assert chosen["stressed_total_pnl"] > 0.0


def test_drawdown_guarded_threshold_rule_prefers_lower_drawdown() -> None:
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    labels = [-5000.0, 7000.0, 300.0, 300.0, 300.0, 300.0]
    scores = [0.10, 0.11, 0.90, 0.89, 0.88, 0.87]
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=base_time + timedelta(minutes=idx * 12),
            features=np.ones((1, 2), dtype=np.float32),
            labels=np.asarray([label], dtype=np.float32),
            offsets=np.asarray([0.0], dtype=np.float32),
            rights=np.asarray(["C"], dtype=object),
            market_last=np.zeros(7, dtype=np.float32),
        )
        for idx, label in enumerate(labels)
    ]
    predictions = [np.asarray([score], dtype=np.float32) for score in scores]
    config = replace(PilotConfig(), min_validation_trades=3, cooldown_minutes=1)

    threshold, sweep = choose_threshold_with_rule(
        decisions,
        predictions,
        config=config,
        threshold_rule="drawdown_guarded_stressed",
        stress_per_trade=0.0,
    )

    chosen = [row for row in sweep if row["threshold"] == threshold][0]
    assert threshold > 0.10
    assert chosen["trades"] >= 4
    assert abs(chosen["stressed_max_drawdown"]) < 6000.0


def test_jitter_stability_threshold_rule_prefers_stable_actions() -> None:
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    labels = [100.0, 100.0, -10.0, -10.0]
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=base_time + timedelta(minutes=idx * 12),
            features=np.ones((1, 12), dtype=np.float32),
            labels=np.asarray([label], dtype=np.float32),
            offsets=np.asarray([0.0], dtype=np.float32),
            rights=np.asarray(["C"], dtype=object),
            market_last=np.zeros(7, dtype=np.float32),
        )
        for idx, label in enumerate(labels)
    ]
    predictions = [
        np.asarray([score], dtype=np.float32)
        for score in [0.9, 0.9, 0.4, 0.4]
    ]
    jitter_predictions = {
        "spread_widen_005": [
            np.asarray([score], dtype=np.float32)
            for score in [0.5, 0.5, 0.4, 0.4]
        ],
    }
    config = replace(PilotConfig(), min_validation_trades=2, cooldown_minutes=1)

    threshold, sweep = choose_threshold_with_rule(
        decisions,
        predictions,
        config=config,
        threshold_rule="jitter_stability_stressed",
        stress_per_trade=20.0,
        jitter_predictions_by_scenario=jitter_predictions,
    )

    chosen = [row for row in sweep if row["threshold"] == threshold][0]
    assert threshold < 0.9
    assert chosen["jitter_min_trades"] >= 2
    assert chosen["jitter_worst_total_pnl"] > 0.0
    assert chosen["jitter_min_action_presence_match_rate"] == 1.0


def test_balanced_classifier_sample_weight_offsets_rare_positive_labels() -> None:
    y = np.asarray([1, 0, 0, 0], dtype=np.int8)

    disabled = _balanced_classifier_sample_weight(y, mode="none")
    balanced = _balanced_classifier_sample_weight(y, mode="balanced_classifier")

    assert disabled is None
    assert balanced is not None
    assert float(balanced[y > 0][0]) == 2.0
    assert np.allclose(balanced[y <= 0], np.asarray([2.0 / 3.0] * 3, dtype=np.float32))
    assert np.isclose(float(balanced.sum()), 4.0)


def test_decision_balanced_classifier_sample_weight_equalizes_decision_mass() -> None:
    y = np.asarray([1, 0, 0, 0], dtype=np.int8)
    decision_ids = np.asarray([0, 0, 0, 1], dtype=np.int64)

    balanced = _balanced_classifier_sample_weight(
        y,
        mode="decision_balanced_classifier",
        decision_ids=decision_ids,
    )

    assert balanced is not None
    assert np.isclose(float(balanced.sum()), 4.0)
    assert float(balanced[0]) > float(balanced[1])
    assert np.isclose(float(balanced[y > 0].sum()), float(balanced[y <= 0].sum()))
    assert float(balanced[3]) > float(balanced[1])


def test_parse_ensemble_seeds_and_average_predictions() -> None:
    assert parse_ensemble_seeds("") == []
    assert parse_ensemble_seeds("17, 23,42") == [17, 23, 42]

    averaged = _average_predictions(
        [
            [np.asarray([1.0, 3.0], dtype=np.float32), np.asarray([5.0], dtype=np.float32)],
            [np.asarray([3.0, 5.0], dtype=np.float32), np.asarray([7.0], dtype=np.float32)],
        ]
    )

    assert np.allclose(averaged[0], np.asarray([2.0, 4.0], dtype=np.float32))
    assert np.allclose(averaged[1], np.asarray([6.0], dtype=np.float32))


def test_flatten_training_examples_with_rights_preserves_side_targets() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.ones((3, 4), dtype=np.float32),
        labels=np.asarray([25.0, 200.0, 75.0], dtype=np.float32),
        offsets=np.asarray([0.0, 10.0, -10.0], dtype=np.float32),
        rights=np.asarray(["C", "P", "C"], dtype=object),
        market_last=np.zeros(7, dtype=np.float32),
    )
    config = replace(
        PilotConfig(),
        target_mode="decision_top_profit_classifier",
        positive_label_threshold=20.0,
    )

    features, targets, rights = _flatten_training_examples_with_rights(
        [decision],
        config=config,
        max_examples=None,
        seed=42,
    )

    assert features.shape == (3, 4)
    assert np.array_equal(targets, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    assert list(rights) == ["C", "P", "C"]


def test_flatten_training_examples_with_rights_supports_decision_aggregate_targets() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.ones((3, 4), dtype=np.float32),
        labels=np.asarray([25.0, 200.0, 75.0], dtype=np.float32),
        offsets=np.asarray([0.0, 10.0, -10.0], dtype=np.float32),
        rights=np.asarray(["C", "P", "C"], dtype=object),
        market_last=np.zeros(7, dtype=np.float32),
    )
    config = replace(
        PilotConfig(),
        target_mode="decision_best_profit_regression",
        target_scale=100.0,
    )

    features, targets, rights = _flatten_training_examples_with_rights(
        [decision],
        config=config,
        max_examples=None,
        seed=42,
    )

    assert features.shape == (3, 4)
    assert np.array_equal(targets, np.asarray([2.0, 2.0, 2.0], dtype=np.float32))
    assert list(rights) == ["C", "P", "C"]


def test_attach_teacher_labels_marks_matching_contract_only(tmp_path) -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.ones((3, 4), dtype=np.float32),
        labels=np.asarray([25.0, 200.0, 75.0], dtype=np.float32),
        offsets=np.asarray([0.0, 10.0, -10.0], dtype=np.float32),
        rights=np.asarray(["C", "P", "C"], dtype=object),
        market_last=np.zeros(7, dtype=np.float32),
        contract_ids=np.asarray(["A", "B", "C"], dtype=object),
    )
    teacher_events = tmp_path / "teacher.parquet"
    pd.DataFrame(
        [
            {
                "split": "q1_2026",
                "session": "2026-01-02",
                "decision_time": "2026-01-02T14:31:00+00:00",
                "protocol101_action": "enter",
                "contract_id": "B",
                "seed": 1,
            },
            {
                "split": "q1_2026",
                "session": "2026-01-02",
                "decision_time": "2026-01-02T14:31:00+00:00",
                "protocol101_action": "wait",
                "contract_id": "C",
                "seed": 1,
            },
        ]
    ).to_parquet(teacher_events)

    summary = _attach_teacher_labels(
        {"train": {"decisions": [decision]}},
        teacher_events=teacher_events,
        min_seed_count=1,
    )

    assert summary["positives_by_split"]["train"]["positive_candidates"] == 1
    assert np.array_equal(
        decision.teacher_labels,
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
