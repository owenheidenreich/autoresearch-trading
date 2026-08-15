from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from v4.scripts import run_protocol249_entry_quality_calibrator as p249


def _event(decision: str, exit_time: str, pnl: float, uid: str) -> dict:
    decision_ts = pd.Timestamp(decision, tz="UTC")
    candidate = pd.DataFrame(
        [
            {
                "candidate_uid": uid,
                "trade_uid": uid,
                "split": "q3_2025",
                "session": "2025-07-01",
                "decision_dt": decision_ts,
                "candidate_exit_dt": pd.Timestamp(exit_time, tz="UTC"),
                "contract_id": f"SPXW-{uid}",
                "right": "P",
                "offset": -5.0,
                "entry_ask": 10.0,
                "candidate_pnl": pnl,
                "candidate_exit_reason": "target",
                "label_source": "synthetic",
            }
        ]
    )
    return {"split": "q3_2025", "session": "2025-07-01", "decision_dt": decision_ts, "candidates": candidate}


def _base(threshold: float = 0.0) -> p249.BaseArtifact:
    return p249.BaseArtifact(
        artifact_dir=Path("unused"),
        feature_columns=[],
        threshold=threshold,
        model=None,
        scaler=None,
    )


def test_calibrator_feature_leakage_guard_rejects_future_columns() -> None:
    with pytest.raises(ValueError):
        p249.assert_no_leakage_features(["base_score", "entry_premium", "candidate_exit_dt"])


def test_required_seed_count_blocks_subset_runs_from_promotion() -> None:
    aggregate = {
        split: {
            "seeds": 1,
            "median_total_pnl": 100.0,
            "positive_seed_fraction": 1.0,
            "median_profit_factor": 2.0,
            "median_stress_0_10_total_pnl": 50.0,
            "beats_frozen_protocol101": True,
            "median_delta_vs_frozen_protocol101": 25.0,
        }
        for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    }
    aggregate["promotion_checks"] = []
    aggregate["promotion_ready"] = True

    checked = p249.add_required_split_and_seed_checks(aggregate, required_seed_count=5)

    assert checked["promotion_ready"] is False
    assert any(check["name"] == "required_seed_count" and check["pass"] is False for check in checked["promotion_checks"])


def test_base_serial_replay_enforces_one_open_position() -> None:
    events = [
        _event("2025-07-01 14:00:00", "2025-07-01 14:10:00", 100.0, "first"),
        _event("2025-07-01 14:05:00", "2025-07-01 14:15:00", 500.0, "overlap"),
    ]
    predictions = [{"action": 1, "score": 1.0, "valid_action": True}, {"action": 1, "score": 1.0, "valid_action": True}]

    result = p249.simulate_base_from_predictions(
        events,
        predictions,
        _base(),
        slippage_per_side=0.0,
        starting_cash=10_000.0,
        strategy="synthetic",
    )

    assert result.summary["trades"] == 1
    assert result.summary["total_pnl"] == 100.0
    assert result.summary["skipped_overlap_candidates"] == 1


def test_gate_rejection_can_open_slot_for_later_trade() -> None:
    events = [
        _event("2025-07-01 14:00:00", "2025-07-01 14:10:00", -100.0, "first"),
        _event("2025-07-01 14:05:00", "2025-07-01 14:15:00", 500.0, "second"),
    ]
    predictions = [{"action": 1, "score": 1.0, "valid_action": True}, {"action": 1, "score": 1.0, "valid_action": True}]

    result = p249.simulate_calibrated(
        events,
        predictions,
        _base(),
        model=None,
        scaler=None,
        gate_threshold=0.0,
        feature_columns=[],
        gate_scores_by_index={0: -1.0, 1: 1.0},
        slippage_per_side=0.0,
        starting_cash=10_000.0,
        strategy="synthetic_gate",
    )

    assert result.summary["trades"] == 1
    assert result.trades[0]["trade_uid"] == "second"
    assert result.summary["total_pnl"] == 500.0
    assert result.summary["skipped_gate_candidates"] == 1
