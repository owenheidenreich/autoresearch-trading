from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v4.scripts.run_protocol101_ft2_stage0_p5_hold_exit_feasibility import (
    EXPECTED_GOAL_SHA256,
    GOAL_PATH,
    LIFECYCLE_FEATURES,
    MINIMUM_VALIDATION_EPISODES,
    TERMINAL_DECISIONS,
    assert_unique,
    balanced_training_weights,
    one_step_target,
    sha256_path,
)


def test_goal_hash_and_terminal_contract_are_frozen() -> None:
    assert sha256_path(GOAL_PATH) == EXPECTED_GOAL_SHA256
    assert MINIMUM_VALIDATION_EPISODES == 100
    assert "stop_insufficient_data" in TERMINAL_DECISIONS


def test_one_step_target_uses_same_entry_and_fee() -> None:
    target = one_step_target(entry_ask=5.0, current_bid=5.5, next_bid=5.8)
    assert target["target_q_exit"] == pytest.approx(47.0)
    assert target["target_q_hold_1m"] == pytest.approx(77.0)
    assert target["target_a_hold_1m"] == pytest.approx(30.0)


def test_balanced_weights_equalize_sessions_and_episodes() -> None:
    frame = pd.DataFrame(
        {
            "session": ["a"] * 6 + ["b"] * 2,
            "episode_id": ["a1"] * 4 + ["a2"] * 2 + ["b1"] * 2,
        }
    )
    frame["weight"] = balanced_training_weights(frame)
    session_sums = frame.groupby("session")["weight"].sum()
    assert session_sums["a"] == pytest.approx(session_sums["b"])
    episode_sums = frame.groupby(["session", "episode_id"])["weight"].sum()
    assert episode_sums[("a", "a1")] == pytest.approx(
        episode_sums[("a", "a2")]
    )


def test_identity_check_fails_closed_on_duplicate() -> None:
    frame = pd.DataFrame(
        {
            "episode_id": ["same", "same"],
            "state_time_ns": [1, 1],
        }
    )
    with pytest.raises(RuntimeError, match="duplicate lifecycle"):
        assert_unique(
            frame,
            ("episode_id", "state_time_ns"),
            label="lifecycle",
        )


def test_feature_contract_excludes_future_and_target_aliases() -> None:
    lowered = tuple(name.lower() for name in LIFECYCLE_FEATURES)
    assert not any("future" in name for name in lowered)
    assert not any("target" in name for name in lowered)
    assert not any("advantage" in name for name in lowered)
    assert "minutes_to_deadline" in LIFECYCLE_FEATURES
    assert "internal_delta" in LIFECYCLE_FEATURES
    assert "internal_gamma" in LIFECYCLE_FEATURES


def test_balanced_weights_are_finite_and_positive() -> None:
    frame = pd.DataFrame(
        {
            "session": ["a", "a", "b", "b"],
            "episode_id": ["a1", "a1", "b1", "b1"],
        }
    )
    weights = balanced_training_weights(frame)
    assert np.isfinite(weights).all()
    assert (weights > 0.0).all()
    assert weights.mean() == pytest.approx(1.0)
