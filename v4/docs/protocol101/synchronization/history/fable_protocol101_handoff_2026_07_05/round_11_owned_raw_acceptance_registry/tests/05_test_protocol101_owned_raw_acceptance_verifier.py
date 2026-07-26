"""Tests for Protocol101 owned raw acceptance verifier."""
from __future__ import annotations

from pathlib import Path
import pickle

from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    AcceptanceThresholds,
    context_causality_quality,
    expected_decision_bounds,
    expected_decision_minutes,
    fold_placement_predicate,
    processed_quality,
    role_policy_allows,
)


def test_expected_decision_minutes_handles_full_and_early_close_days() -> None:
    assert expected_decision_minutes("2024-10-01") == 360
    assert expected_decision_minutes("2024-11-29") == 209
    assert expected_decision_minutes("2024-12-24") == 224


def test_expected_decision_bounds_match_calendar() -> None:
    first, last = expected_decision_bounds("2024-10-01")

    assert first.isoformat() == "2024-10-01T13:31:00+00:00"
    assert last.isoformat() == "2024-10-01T19:30:00+00:00"


def test_context_causality_quality_requires_one_minute_lag_and_no_open_backfill(tmp_path: Path) -> None:
    rows = []
    for minute in range(3):
        decision = f"2024-10-01T13:{31 + minute:02d}:00+00:00"
        source = f"2024-10-01T13:{30 + minute:02d}:00+00:00"
        rows.append(
            {
                "decision_time": decision,
                "source_context_time": source,
                "context_start_timestamp": "2024-10-01T13:30:00+00:00",
                "context_last_timestamp": source,
                "context_minute_rows": minute + 1,
                "context_ready": False,
            }
        )
    path = tmp_path / "2024-10-01.pkl"
    with path.open("wb") as handle:
        pickle.dump(rows, handle)

    quality = context_causality_quality("2024-10-01", tmp_path)

    assert quality["first_decision_matches_calendar"] is True
    assert quality["context_lag_exact_one_minute_share"] == 1.0
    assert quality["future_context_row_count"] == 0
    assert quality["opening_no_leading_backfill"] is True


def test_fold_placement_requires_role_processed_rows_and_acceptance() -> None:
    era_manifest = {
        "sessions": [
            {"session": "2024-10-01", "era": "pre_program_oct2024_jun2025"},
            {"session": "2026-07-01", "era": "confirmation_jun_jul2026"},
        ]
    }
    role_policy = {
        "policy": {
            "pre_program_oct2024_jun2025": {"permitted_roles": ["diagnostics_only", "train", "test"]},
            "confirmation_jun_jul2026": {"permitted_roles": ["confirmation_one_shot", "report_only"]},
        }
    }
    registry = {
        "sessions": [
            {
                "session": "2024-10-01",
                "status": "pass",
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
            {
                "session": "2024-10-02",
                "status": "pass",
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
            {
                "session": "2026-07-01",
                "status": "pass",
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
        ]
    }

    assert role_policy_allows(role_policy, "pre_program_oct2024_jun2025", "diagnostics_only")
    assert fold_placement_predicate(
        session="2024-10-01",
        role="diagnostics_only",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )["placeable"] is True
    missing_era = fold_placement_predicate(
        session="2024-10-02",
        role="diagnostics_only",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )
    assert missing_era["placeable"] is False
    assert missing_era["checks"]["era_permits_role"] is False
    confirmation_as_train = fold_placement_predicate(
        session="2026-07-01",
        role="train",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )
    assert confirmation_as_train["placeable"] is False
    assert confirmation_as_train["checks"]["era_permits_role"] is False


def test_processed_quality_handles_missing_file(tmp_path: Path) -> None:
    quality = processed_quality("2024-10-01", tmp_path)

    assert quality["processed_exists"] is False
    assert quality["neural_rows"] == 0
    assert quality["expected_decision_minutes"] == 360
    assert quality["full_ladder_share"] == 0.0
    assert quality["tradable_minute_share"] == 0.0


def test_threshold_defaults_are_data_plane_only() -> None:
    thresholds = AcceptanceThresholds()

    assert thresholds.min_full_ladder_share == 0.70
    assert thresholds.min_tradable_minute_share == 0.50
    assert thresholds.max_missing_processed_rows == 0
