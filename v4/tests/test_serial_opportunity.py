from __future__ import annotations

import pandas as pd
import pytest

from v4.live.shadow_lifecycle import strict_serial_shadow_rows
from v4.model.serial_opportunity import (
    ENTRY_FEATURE_COLUMNS,
    SerialOpportunityConfig,
    assert_feature_columns_are_causal,
    select_validation_threshold,
    serial_simulate_candidates,
    validate_serial_trades,
)
from v4.scripts.run_protocol101_event_history_policy import add_causal_history_features


def _candidate(
    uid: str,
    *,
    decision: str,
    exit_time: str,
    score: float,
    pnl: float,
    right: str = "C",
    offset: float = 0.0,
) -> dict:
    return {
        "candidate_uid": uid,
        "trade_uid": f"trade-{uid}",
        "split": "q3_2025",
        "seed": 1,
        "entry_seed": 11,
        "session": "2025-07-01",
        "decision_dt": pd.Timestamp(decision, tz="UTC"),
        "candidate_exit_dt": pd.Timestamp(exit_time, tz="UTC"),
        "contract_id": f"SPXW-{uid}",
        "right": right,
        "offset": offset,
        "score": score,
        "candidate_pnl": pnl,
        "candidate_exit_reason": "model_exit",
        "label_source": "protocol081",
    }


def test_protocol092_feature_columns_are_entry_only() -> None:
    assert_feature_columns_are_causal(ENTRY_FEATURE_COLUMNS)
    with pytest.raises(ValueError):
        assert_feature_columns_are_causal([*ENTRY_FEATURE_COLUMNS, "candidate_exit_pnl"])


def test_serial_simulator_enforces_one_open_position_and_terminal_final() -> None:
    frame = pd.DataFrame(
        [
            _candidate("a", decision="2025-07-01T14:00:00", exit_time="2025-07-01T14:10:00", score=5.0, pnl=100.0),
            _candidate("b", decision="2025-07-01T14:05:00", exit_time="2025-07-01T14:15:00", score=100.0, pnl=300.0),
            _candidate("c", decision="2025-07-01T14:11:00", exit_time="2025-07-01T14:20:00", score=10.0, pnl=50.0, right="P", offset=-5.0),
        ]
    )
    result = serial_simulate_candidates(
        frame,
        score_column="score",
        threshold=0.0,
        slippage_per_side=0.0,
        strategy="test",
    )

    assert [trade["candidate_uid"] for trade in result.trades] == ["a", "c"]
    assert result.summary["skipped_overlap_candidates"] == 1
    assert result.summary["max_concurrent_positions"] == 1
    assert result.summary["all_flat_by_session_end"] is True
    assert result.summary["terminal_final"] is True
    assert validate_serial_trades(result.trades)["status"] == "pass"


def test_same_decision_chooses_highest_scored_candidate() -> None:
    frame = pd.DataFrame(
        [
            _candidate("low", decision="2025-07-01T14:00:00", exit_time="2025-07-01T14:10:00", score=1.0, pnl=-100.0),
            _candidate("high", decision="2025-07-01T14:00:00", exit_time="2025-07-01T14:08:00", score=2.0, pnl=120.0, right="P"),
        ]
    )
    result = serial_simulate_candidates(
        frame,
        score_column="score",
        threshold=0.0,
        slippage_per_side=0.0,
        strategy="test",
    )
    assert [trade["candidate_uid"] for trade in result.trades] == ["high"]


def test_threshold_selection_records_validation_source_only() -> None:
    frame = pd.DataFrame(
        [
            _candidate("v1", decision="2025-07-01T14:00:00", exit_time="2025-07-01T14:04:00", score=1.0, pnl=100.0),
            _candidate("v2", decision="2025-07-01T14:10:00", exit_time="2025-07-01T14:14:00", score=2.0, pnl=150.0),
            _candidate("v3", decision="2025-07-01T14:20:00", exit_time="2025-07-01T14:24:00", score=3.0, pnl=-30.0),
        ]
    )
    selection = select_validation_threshold(
        frame,
        score_column="score",
        source_split="q2_2025",
        source_seed=1,
        config=SerialOpportunityConfig(min_validation_trades=1),
    )
    assert selection.source_split == "q2_2025"
    assert selection.source_seed == 1
    assert selection.source_rows == len(frame)
    assert selection.sweep


def test_serial_policy_output_survives_strict_shadow_lifecycle_transform() -> None:
    frame = pd.DataFrame(
        [
            _candidate("a", decision="2025-07-01T14:00:00", exit_time="2025-07-01T14:10:00", score=5.0, pnl=100.0),
            _candidate("b", decision="2025-07-01T14:05:00", exit_time="2025-07-01T14:15:00", score=100.0, pnl=300.0),
        ]
    )
    result = serial_simulate_candidates(
        frame,
        score_column="score",
        threshold=0.0,
        slippage_per_side=0.0,
        strategy="test",
    )
    rows = []
    for trade in result.trades:
        rows.extend(
            [
                {
                    "trade_uid": trade["candidate_uid"],
                    "entry_decision_time": trade["decision_time"],
                    "decision_time": trade["decision_time"],
                    "timestamp_ms": 1,
                    "sequence_step_index": 0,
                    "decision": {"action": "hold"},
                },
                {
                    "trade_uid": trade["candidate_uid"],
                    "entry_decision_time": trade["decision_time"],
                    "decision_time": trade["exit_time"],
                    "timestamp_ms": 2,
                    "sequence_step_index": 1,
                    "decision": {"action": "exit"},
                },
            ]
        )
    strict = strict_serial_shadow_rows(rows, require_terminal=True)
    assert strict.summary["selected_trades"] == len(result.trades)
    assert strict.summary["skipped_overlap_trades"] == 0
    assert strict.summary["post_terminal_rows_removed"] == 0


def test_event_history_features_are_past_only() -> None:
    first_candidates = pd.DataFrame(
        [
            {"right": "C", "edge": 10.0, "entry_gamma": 0.05, "entry_theta_burden": 12.0, "entry_spread_over_mid": 0.02},
            {"right": "P", "edge": 4.0, "entry_gamma": 0.03, "entry_theta_burden": 8.0, "entry_spread_over_mid": 0.03},
        ]
    )
    second_candidates = pd.DataFrame(
        [
            {"right": "C", "edge": 99.0, "entry_gamma": 0.50, "entry_theta_burden": 99.0, "entry_spread_over_mid": 0.50},
        ]
    )
    events = [
        {
            "split": "q4_2025",
            "seed": 1,
            "session": "2025-10-01",
            "decision_dt": pd.Timestamp("2025-10-01T14:00:00Z"),
            "candidates": first_candidates,
        },
        {
            "split": "q4_2025",
            "seed": 1,
            "session": "2025-10-01",
            "decision_dt": pd.Timestamp("2025-10-01T14:05:00Z"),
            "candidates": second_candidates,
        },
    ]
    add_causal_history_features(events)

    first = events[0]["candidates"].iloc[0]
    second = events[1]["candidates"].iloc[0]
    assert first["hist_events_seen"] == 0.0
    assert first["hist_prev_max_edge"] == 0.0
    assert second["hist_events_seen"] == 1.0
    assert second["hist_minutes_since_prev_event"] == 5.0
    assert second["hist_prev_max_edge"] == 10.0
    assert second["hist_prev_call_minus_put_edge"] == 6.0
