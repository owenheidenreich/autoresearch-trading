"""Tests for fair-contract model search registry helpers."""
from __future__ import annotations

import json

from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
from v4.scripts.run_protocol101_fair_contract_model_search import (
    AttemptConfig,
    DEFAULT_ROUND_TRIP_FEE_DOLLARS,
    EXPERIMENT_REGISTRY_ENTRY_VERSION,
    attempt_score,
    current_implementation_versions,
    default_attempts,
    fee_adjusted_net_pnl,
    main,
    reusable_registry_entry,
    reason_for_result,
    run_attempt,
    select_attempts,
)
from v4.scripts import run_protocol101_fair_contract_model_search as model_search_module
from v4.model.supervised_pilot import (
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
)


def test_default_attempts_are_unique_and_safe() -> None:
    attempts = default_attempts()
    ids = [attempt.attempt_id for attempt in attempts]

    assert len(ids) == len(set(ids))
    assert len(attempts) == 7
    assert {attempt.policy_index for attempt in attempts} == set(range(7))
    assert {attempt.model_family for attempt in attempts} == {"sklearn_hist_gradient_boosting"}
    assert {attempt.target_mode for attempt in attempts} == {"return_on_premium_regression"}
    assert {attempt.threshold_rule for attempt in attempts} == {"max_validation_stressed_pnl"}
    assert {attempt.max_trades_per_session for attempt in attempts} == {3}
    assert all(attempt.run_feature_jitter_gate for attempt in attempts)
    assert {attempt.fit_mode for attempt in attempts} == {"train_tail20_calibration"}
    assert not any(attempt.threshold_rule == "risk_adjusted_stressed" for attempt in attempts)
    assert attempts[0].feature_transform == (
        FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE
    )


def test_select_attempts_filters_preregistered_ids() -> None:
    attempts = default_attempts()

    selected = select_attempts(
        attempts,
        max_attempts=1,
        attempt_ids=(
            "attempt_menuv2_policy0_hgb_rop_s42,"
            "attempt_menuv2_policy6_hgb_rop_s42"
        ),
    )

    assert [attempt.attempt_id for attempt in selected] == [
        "attempt_menuv2_policy0_hgb_rop_s42",
        "attempt_menuv2_policy6_hgb_rop_s42",
    ]


def test_attempt_score_selects_plain_fee_adjusted_net_pnl_after_hard_gates() -> None:
    lower_net = {
        "status": "pass",
        "metrics": {
            "validation": {"total_pnl": 1000, "trades": 25},
            "diagnostic_test": {"total_pnl": 500, "trades": 25},
        }
    }
    higher_net = {
        "status": "pass",
        "metrics": {
            "validation": {"total_pnl": 900, "trades": 10},
            "diagnostic_test": {"total_pnl": 800, "trades": 10},
        }
    }

    assert fee_adjusted_net_pnl(lower_net) == 1500 - 50 * DEFAULT_ROUND_TRIP_FEE_DOLLARS
    assert attempt_score(higher_net) > attempt_score(lower_net)


def test_attempt_score_places_failed_hard_gates_below_passed_candidates() -> None:
    failed_high_pnl = {
        "status": "fail",
        "blockers": ["diagnostic_profit_factor"],
        "metrics": {
            "validation": {"total_pnl": 50_000, "trades": 10},
            "diagnostic_test": {"total_pnl": 50_000, "trades": 10},
        },
    }
    passed_low_pnl = {
        "status": "pass",
        "blockers": [],
        "metrics": {
            "validation": {"total_pnl": 100, "trades": 1},
            "diagnostic_test": {"total_pnl": 100, "trades": 1},
        },
    }

    assert attempt_score(passed_low_pnl) > attempt_score(failed_high_pnl)


def test_attempt_score_penalizes_failed_feature_jitter_gate() -> None:
    replay = {
        "status": "pass",
        "blockers": [],
        "metrics": {
            "validation": {
                "total_pnl": 2000,
                "profit_factor": 1.6,
                "max_drawdown": -1000,
                "trades": 24,
            },
            "diagnostic_test": {
                "total_pnl": 1200,
                "profit_factor": 1.4,
                "max_drawdown": -900,
                "trades": 22,
            },
        },
    }

    passed = {"status": "pass", "blockers": []}
    failed = {
        "status": "fail",
        "blockers": ["spread_widen_005:diagnostic_test:selected_contract_match_rate"],
    }

    assert attempt_score(replay, passed) > attempt_score(replay, failed)


def test_reason_for_result_records_failed_blockers() -> None:
    reason = reason_for_result(
        validation_gate={"status": "fail", "blockers": ["validation_profit_factor"]},
        replay_summary={"status": "fail", "blockers": ["diagnostic_positive_pnl"]},
    )

    assert reason == "rejected:diagnostic_positive_pnl,validation_profit_factor"


def test_reason_for_result_records_feature_jitter_blockers() -> None:
    reason = reason_for_result(
        validation_gate={"status": "pass", "blockers": []},
        replay_summary={"status": "pass", "blockers": []},
        feature_jitter_summary={
            "status": "fail",
            "blockers": ["spread_widen_005:validation:selected_contract_match_rate"],
        },
    )

    assert reason == "rejected:spread_widen_005:validation:selected_contract_match_rate"


def test_reusable_registry_entry_requires_matching_config_and_artifacts(tmp_path) -> None:
    attempt = AttemptConfig(
        attempt_id="attempt_test",
        hypothesis="test reusable registry entry",
    )
    artifacts = {}
    for key in (
        "training_result",
        "model",
        "selected_candidates",
        "strict_replay_trades",
        "candidate_validation_report",
        "strict_replay_report",
    ):
        path = tmp_path / f"{key}.txt"
        path.write_text("ok\n")
        artifacts[key] = str(path)
    existing = {
        "schema_version": EXPERIMENT_REGISTRY_ENTRY_VERSION,
        "attempt_id": attempt.attempt_id,
        "implementation_versions": current_implementation_versions(),
        "config": {key: value for key, value in attempt.__dict__.items() if key != "entry_filter"},
        "artifacts": artifacts,
    }

    result = reusable_registry_entry(existing, attempt)
    assert result is not None
    assert result["schema_version"] == EXPERIMENT_REGISTRY_ENTRY_VERSION
    assert result["implementation_versions"] == current_implementation_versions()

    stale_schema = {**existing, "schema_version": "Protocol101FairContractExperimentRegistryEntryV1"}
    assert reusable_registry_entry(stale_schema, attempt) is None
    prior_schema = {
        **existing,
        "schema_version": "Protocol101FairContractExperimentRegistryEntryV2",
    }
    assert reusable_registry_entry(prior_schema, attempt) is not None
    stale_versions = {
        **existing,
        "implementation_versions": {
            **current_implementation_versions(),
            "selected_candidate_export": "old_policy_index_bug",
        },
    }
    assert reusable_registry_entry(stale_versions, attempt) is None
    mismatched = {**existing, "config": {**attempt.__dict__, "seed": 99}}
    assert reusable_registry_entry(mismatched, attempt) is None
    (tmp_path / "model.txt").unlink()
    assert reusable_registry_entry(existing, attempt) is None


def test_reusable_registry_entry_allows_fold_aware_attempt_without_selected_export(tmp_path) -> None:
    attempt = AttemptConfig(
        attempt_id="attempt_fold_aware",
        hypothesis="test fold-aware reusable registry entry",
        model_family="sklearn_hist_gradient_boosting",
        target_mode="return_on_premium_regression",
    )
    training_result = tmp_path / "training_result.json"
    model = tmp_path / "model.pt"
    training_result.write_text("{}\n")
    model.write_text("model-index\n")
    existing = {
        "schema_version": EXPERIMENT_REGISTRY_ENTRY_VERSION,
        "attempt_id": attempt.attempt_id,
        "implementation_versions": current_implementation_versions(),
        "config": attempt.__dict__,
        "fold_aware_training": {"execution_mode": "fold_aware_expanding_cv"},
        "artifacts": {
            "training_result": str(training_result),
            "model": str(model),
            "selected_candidates": "",
            "strict_replay_trades": "",
            "candidate_validation_report": "",
            "strict_replay_report": "",
        },
    }

    result = reusable_registry_entry(existing, attempt)

    assert result is not None
    assert result["schema_version"] == EXPERIMENT_REGISTRY_ENTRY_VERSION


def test_run_attempt_records_fold_aware_training_without_legacy_export(monkeypatch, tmp_path) -> None:
    attempt = AttemptConfig(
        attempt_id="attempt_fold_aware_run",
        hypothesis="test fold-aware run attempt",
        model_family="sklearn_hist_gradient_boosting",
        target_mode="return_on_premium_regression",
    )
    root = tmp_path / "model_search"
    train_dir = root / "attempts" / attempt.attempt_id / "training_runner"
    train_dir.mkdir(parents=True)
    training_result = train_dir / "training_result.json"
    training_result.write_text(
        json.dumps(
            {
                "execution_mode": "fold_aware_expanding_cv",
                "fold_count": 5,
                "chosen_threshold": 2.0,
                "fold_summary": {
                    "aggregate_validation_metrics": {
                        "trades": 10,
                        "total_pnl": 123.0,
                        "profit_factor": 1.4,
                        "max_drawdown": -25.0,
                    }
                },
                "neural": {
                    "validation": {
                        "metrics": {
                            "trades": 10,
                            "total_pnl": 123.0,
                        }
                    }
                },
            }
        )
        + "\n"
    )
    (train_dir / "model.pt").write_text("model-index\n")
    design = tmp_path / "design.json"
    design.write_text(
        json.dumps(
            {
                "selected_feature_contract": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
                "allowed_data": {
                    "included_session_count": 301,
                    "included_first_session": "2025-01-02",
                    "included_last_session": "2026-03-31",
                },
            }
        )
        + "\n"
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        model_search_module,
        "run_command",
        lambda args, *, cwd: calls.append(list(args)),
    )

    entry = run_attempt(
        attempt=attempt,
        root=root,
        design=design,
        cwd=tmp_path,
        force=False,
    )

    assert calls == []
    assert entry["model_training_executed"] is True
    assert entry["threshold_selection_executed"] is True
    assert entry["broker_endpoint_called"] is False
    assert entry["paper_submit_allowed"] is False
    assert entry["fold_aware_training"]["execution_mode"] == "fold_aware_expanding_cv"
    assert entry["fold_aware_training"]["fold_count"] == 5
    assert entry["data_scope"] == "301_sessions_2025-01-02_to_2026-03-31_5fold_expanding_cv"
    assert entry["candidate_validation_gate"]["blockers"] == [
        "fold_aware_selected_export_and_strict_replay_pending"
    ]
    assert entry["artifacts"]["selected_candidates"] == ""


def test_model_search_dry_run_does_not_execute_attempts(monkeypatch, tmp_path) -> None:
    design = tmp_path / "design.json"
    design.write_text(
        json.dumps({"selected_feature_contract": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED})
        + "\n"
    )
    out_dir = tmp_path / "model_search"
    monkeypatch.setattr(
        model_search_module,
        "run_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("run_attempt should not execute")),
    )

    exit_code = main(
        [
            "--dry-run",
            "--design",
            str(design),
            "--out-dir",
            str(out_dir),
            "--max-attempts",
            "2",
        ]
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert exit_code == 0
    assert summary["status"] == "dry_run_ready"
    assert summary["decision"] == "fold_aware_model_search_ready_no_training_executed"
    assert summary["attempts"] == 2
    assert summary["blockers"] == []
    assert summary["model_training_executed"] is False
    assert summary["threshold_selection_executed"] is False
    assert summary["broker_endpoint_called"] is False
    assert summary["paper_submit_allowed"] is False


def test_model_search_blocks_execution_without_owner_approval(monkeypatch, tmp_path) -> None:
    design = tmp_path / "design.json"
    design.write_text(
        json.dumps({"selected_feature_contract": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED})
        + "\n"
    )
    out_dir = tmp_path / "model_search"
    monkeypatch.setattr(
        model_search_module,
        "run_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("run_attempt should not execute")),
    )

    exit_code = main(
        [
            "--design",
            str(design),
            "--out-dir",
            str(out_dir),
            "--max-attempts",
            "1",
        ]
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert exit_code == 2
    assert summary["status"] == "blocked"
    assert summary["model_training_executed"] is False
    assert "missing_owner_approved_model_training_flag" in summary["blockers"]
    assert "missing_owner_approved_threshold_selection_flag" in summary["blockers"]
    assert "missing_owner_approval_note" in summary["blockers"]
