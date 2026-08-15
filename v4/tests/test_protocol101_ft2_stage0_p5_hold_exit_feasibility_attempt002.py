from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v4.scripts.run_protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002 import (
    EXPECTED_GOAL_SHA256,
    EXPECTED_TOTAL_OPPORTUNITIES,
    EXPECTED_VALIDATION_OPPORTUNITIES,
    GOAL_PATH,
    _row_outcome,
    strong_shuffle_targets,
)
from v4.scripts.run_protocol101_ft2_stage0_p5_hold_exit_feasibility import (
    sha256_path,
)


def test_attempt002_goal_and_smoke_anchors_are_frozen() -> None:
    assert sha256_path(GOAL_PATH) == EXPECTED_GOAL_SHA256
    assert EXPECTED_TOTAL_OPPORTUNITIES == 6_414
    assert EXPECTED_VALIDATION_OPPORTUNITIES == 1_354


def test_strong_shuffle_moves_complete_groups_across_sessions() -> None:
    records = []
    for session in ("a", "b", "c"):
        for episode in range(2):
            for state in range(4):
                records.append(
                    {
                        "episode_id": f"{session}-{episode}",
                        "session": session,
                        "right": "C",
                        "state_index": state,
                        "target_valid": True,
                        "target_a_hold_1m": float(
                            100 * ord(session) + 10 * episode + state
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    shuffled, receipt = strong_shuffle_targets(frame, seed=8700)
    assert receipt["same_session_maps"] == 0
    assert receipt["self_maps"] == 0
    assert receipt["unmatched_episode_count"] == 0
    assert receipt["finite_shuffled_rows"] == len(frame)
    assert len(shuffled) == len(frame)


def test_strong_shuffle_reports_unmatched_single_session_group() -> None:
    frame = pd.DataFrame(
        {
            "episode_id": ["only"] * 4,
            "session": ["a"] * 4,
            "right": ["P"] * 4,
            "state_index": range(4),
            "target_valid": [True] * 4,
            "target_a_hold_1m": [1.0, 2.0, 3.0, 4.0],
        }
    )
    shuffled, receipt = strong_shuffle_targets(frame, seed=8701)
    assert receipt["unmatched_episode_count"] == 1
    assert receipt["finite_shuffled_rows"] == 0
    assert pd.isna(shuffled).all()


def test_strong_shuffle_maps_all_feasible_imbalanced_session_groups() -> None:
    records = []
    for session, episode_count in (("a", 4), ("b", 2), ("c", 2)):
        for episode in range(episode_count):
            for state in range(3):
                records.append(
                    {
                        "episode_id": f"{session}-{episode}",
                        "session": session,
                        "right": "C",
                        "state_index": state,
                        "target_valid": True,
                        "target_a_hold_1m": float(
                            100 * ord(session) + 10 * episode + state
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    shuffled, receipt = strong_shuffle_targets(frame, seed=8700)
    assert receipt["same_session_maps"] == 0
    assert receipt["self_maps"] == 0
    assert receipt["unmatched_episode_count"] == 0
    assert receipt["finite_shuffled_rows"] == len(frame)
    assert np.isfinite(shuffled).all()


def test_attempt002_does_not_mutate_attempt001_goal_hash() -> None:
    from v4.scripts import (
        run_protocol101_ft2_stage0_p5_hold_exit_feasibility as attempt001,
    )

    assert sha256_path(attempt001.GOAL_PATH) == attempt001.EXPECTED_GOAL_SHA256


def test_model_configuration_remains_bounded() -> None:
    from v4.scripts.run_protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002 import (
        HGB_CONFIG,
        MODEL_SEED,
        SHUFFLE_SEEDS,
    )

    assert MODEL_SEED == 42
    assert SHUFFLE_SEEDS == (8700, 8701)
    assert HGB_CONFIG["max_depth"] == 3
    assert HGB_CONFIG["max_iter"] <= 200
    assert HGB_CONFIG["learning_rate"] == pytest.approx(0.05)


def test_deadline_fallback_preserves_two_exit_clocks() -> None:
    minute_ns = 60_000_000_000
    episode = {
        "episode_id": "episode",
        "role": "nested_validation",
        "fold": "fold-1",
        "session": "2026-01-01",
        "decision_time_ns": 100 * minute_ns,
        "contract_id": "contract",
        "right": "C",
        "canonical_slot": 21,
        "entry_ask": 1.0,
        "entry_quote_time_ns": 100 * minute_ns,
        "source_context_time_ns": 99 * minute_ns,
        "policy_deadline_ns": 115 * minute_ns,
    }
    row = pd.Series(
        {
            "state_time_ns": 114 * minute_ns,
            "state_quote_time_ns": 113 * minute_ns,
            "current_bid": 1.2,
            "current_mid": 1.25,
        }
    )
    outcome = _row_outcome(
        episode,
        row,
        comparator="hold_deadline",
        deadline_fallback=True,
    )
    assert outcome["exit_source_quote_time_ns"] == 113 * minute_ns
    assert outcome["realized_exit_time_ns"] == 115 * minute_ns
    assert outcome["holding_minutes"] == pytest.approx(15.0)
    assert outcome["exit_quote_age_ms"] == pytest.approx(120_000.0)


def test_early_learned_exit_uses_same_clock_terminal_reason() -> None:
    from v4.model.protocol101_regimen_repair import ExitReason

    minute_ns = 60_000_000_000
    episode = {
        "episode_id": "episode",
        "role": "nested_validation",
        "fold": "fold-1",
        "session": "2026-01-01",
        "decision_time_ns": 100 * minute_ns,
        "contract_id": "contract",
        "right": "P",
        "canonical_slot": 20,
        "entry_ask": 1.0,
        "entry_quote_time_ns": 100 * minute_ns,
        "source_context_time_ns": 99 * minute_ns,
        "policy_deadline_ns": 115 * minute_ns,
    }
    row = pd.Series(
        {
            "state_time_ns": 104 * minute_ns,
            "state_quote_time_ns": 104 * minute_ns,
            "current_bid": 1.2,
            "current_mid": 1.25,
        }
    )
    outcome = _row_outcome(episode, row, comparator="real_hgb")
    assert outcome["exit_source_quote_time_ns"] == 104 * minute_ns
    assert outcome["realized_exit_time_ns"] == 104 * minute_ns
    assert outcome["exit_reason_code"] == int(ExitReason.TAKE_PROFIT)
