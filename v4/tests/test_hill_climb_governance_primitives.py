from __future__ import annotations

from pathlib import Path

from v4.foundation.experiment_registry import (
    ExperimentRegistryEntry,
    append_experiment_registry_entry,
    load_experiment_registry,
    validate_experiment_registry_entry,
)
from v4.foundation.purged_embargo_validation import (
    PurgedEmbargoValidationPlan,
    SplitWindow,
    validate_purged_embargo_plan,
)
from v4.model.balanced_profitability_score import score_balanced_profitability_v2


def test_experiment_registry_accepts_append_only_terminal_entry(tmp_path: Path) -> None:
    entry = ExperimentRegistryEntry(
        experiment_id="exp_unit_001",
        hypothesis="Runner exits improve expectancy without raising drawdown.",
        code_version="abc123",
        dataset_version="dataset_v1",
        feature_set="features_v1",
        label_definition="labels_v1",
        train_windows=("2024-10",),
        validation_windows=("2025-01",),
        protected_test_windows=("lockbox_2026",),
        primary_metric="BalancedProfitabilityScoreV2",
        baseline_id="PAPER_DEFAULT_PROTOCOL101",
        candidate_id="candidate_runner_exit",
        decision="rejected",
        reason="Unit-test negative result is stored.",
        negative_result_stored=True,
    )

    assert validate_experiment_registry_entry(entry)["status"] == "pass"
    registry = append_experiment_registry_entry(entry, registry_path=tmp_path / "registry.jsonl")

    rows = load_experiment_registry(registry)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp_unit_001"


def test_experiment_registry_rejects_protected_holdout_selection() -> None:
    entry = ExperimentRegistryEntry(
        experiment_id="exp_unit_bad",
        hypothesis="Bad holdout use.",
        code_version="abc123",
        dataset_version="dataset_v1",
        feature_set="features_v1",
        label_definition="labels_v1",
        train_windows=("2024-10",),
        validation_windows=("2025-01",),
        protected_test_windows=("lockbox_2026",),
        primary_metric="BalancedProfitabilityScoreV2",
        baseline_id="PAPER_DEFAULT_PROTOCOL101",
        candidate_id="candidate_bad",
        decision="accepted",
        reason="Should fail.",
        protected_holdout_used_for_selection=True,
    )

    result = validate_experiment_registry_entry(entry)

    assert result["status"] == "fail"
    assert "protected_holdout_used_for_selection" in result["errors"]


def test_purged_embargo_validation_requires_session_day_gap() -> None:
    plan = PurgedEmbargoValidationPlan(
        windows=(
            SplitWindow("train_a", "train", "2026-01-01", "2026-01-10"),
            SplitWindow("validation_a", "validation", "2026-01-11", "2026-01-20"),
        ),
        purge_days=1,
        embargo_days=1,
    )

    result = validate_purged_embargo_plan(plan)

    assert result["status"] == "fail"
    assert result["errors"] == ["insufficient_purge_embargo_gap:train_a:validation_a:required=2:actual=0"]


def test_purged_embargo_validation_accepts_protected_lockbox_gap() -> None:
    plan = PurgedEmbargoValidationPlan(
        windows=(
            SplitWindow("train_a", "train", "2026-01-01", "2026-01-10"),
            SplitWindow("validation_a", "validation", "2026-01-13", "2026-01-20"),
            SplitWindow("lockbox", "protected_test", "2026-01-23", "2026-01-31"),
        ),
        purge_days=1,
        embargo_days=1,
        protected_holdout_names=("lockbox",),
    )

    result = validate_purged_embargo_plan(plan)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_balanced_profitability_score_v2_blocks_missing_parity_and_overconcentration() -> None:
    result = score_balanced_profitability_v2(
        {
            **_good_metrics(),
            "decision_parity_passed": False,
            "top_1_day_pnl_fraction": 0.80,
        }
    )

    assert result["decision"] == "blocked_by_hard_gates"
    assert result["score"] == 0.0
    assert "decision_parity_passed" in result["hard_gate_failures"]
    assert "top_1_day_profit_concentration" in result["hard_gate_failures"]


def test_balanced_profitability_score_v2_caps_trade_count_sufficiency() -> None:
    base = score_balanced_profitability_v2(_good_metrics(trade_count=500))
    more = score_balanced_profitability_v2(_good_metrics(trade_count=2_000))

    assert base["decision"] == "eligible_for_hill_climb_ranking"
    assert more["decision"] == "eligible_for_hill_climb_ranking"
    assert base["components"]["trade_frequency_sufficiency_score"] == 1.0
    assert more["components"]["trade_frequency_sufficiency_score"] == 1.0


def _good_metrics(*, trade_count: int = 500) -> dict:
    return {
        "no_leakage": True,
        "decision_parity_passed": True,
        "protected_holdout_untouched": True,
        "all_flat_by_close": True,
        "net_pnl_under_required_slippage": 1_000.0,
        "trade_count": trade_count,
        "max_drawdown_fraction": 0.10,
        "unaffordable_trade_count": 0,
        "overlapping_headline_trade_count": 0,
        "top_1_day_pnl_fraction": 0.10,
        "top_5_day_pnl_fraction": 0.30,
        "top_10_trade_pnl_fraction": 0.20,
        "protected_holdout_tuning_count": 0,
        "robust_net_pnl_per_day": 300.0,
        "return_on_premium": 0.35,
        "drawdown_efficiency": 5.0,
        "win_rate": 0.75,
        "expectancy_per_trade": 80.0,
        "regime_stability_score": 0.80,
    }
