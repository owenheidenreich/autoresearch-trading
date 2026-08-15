"""The occupancy walk and the threshold rule decide whether the curve is real.

If occupancy leaks, the equity curve holds overlapping contracts the account
could never have carried. If the threshold reads the test fold, the hit rate is
the answer written back as the question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import train_selective_policy as tsp
from v5.ops.build_decision_dataset import LABEL_HORIZON


def _candidates(minutes, scores, session="2024-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": session,
            "minute_index": list(minutes),
            "score": list(scores),
            "net_label": np.arange(len(list(minutes)), dtype=float),
        }
    )


def test_one_position_at_a_time_is_enforced() -> None:
    minutes = [570, 575, 580, 600, 605, 640]
    part = _candidates(minutes, [0.9] * len(minutes))
    got = tsp.walk_serially(part, np.ones(len(part), bool))

    taken = sorted(got["minute_index"].tolist())
    for a, b in zip(taken, taken[1:]):
        assert b - a >= LABEL_HORIZON, f"overlapping trades at {a} and {b}"


def test_occupancy_blocks_the_whole_horizon_then_releases() -> None:
    minutes = [570, 570 + LABEL_HORIZON - 5, 570 + LABEL_HORIZON]
    part = _candidates(minutes, [0.9, 0.99, 0.99])
    got = tsp.walk_serially(part, np.ones(len(part), bool))
    assert got["minute_index"].tolist() == [570, 570 + LABEL_HORIZON]


def test_the_best_scoring_candidate_at_a_minute_is_the_one_taken() -> None:
    part = pd.DataFrame(
        {
            "session": "2024-01-02",
            "minute_index": [570, 570, 570],
            "score": [0.2, 0.95, 0.5],
            "net_label": [1.0, 2.0, 3.0],
        }
    )
    got = tsp.walk_serially(part, np.ones(len(part), bool))
    assert len(got) == 1
    assert got.iloc[0]["net_label"] == 2.0


def test_occupancy_does_not_carry_across_sessions() -> None:
    part = pd.concat(
        [
            _candidates([570], [0.9], session="2024-01-02"),
            _candidates([570], [0.9], session="2024-01-03"),
        ],
        ignore_index=True,
    )
    got = tsp.walk_serially(part, np.ones(len(part), bool))
    assert len(got) == 2


def test_a_candidate_below_the_cut_does_not_occupy_the_bot() -> None:
    """Standing aside must leave the bot free for a later, better candidate."""

    part = _candidates([570, 575], [0.1, 0.9])
    got = tsp.walk_serially(part, np.array([False, True]))
    assert got["minute_index"].tolist() == [575]


def test_threshold_reaches_the_requested_precision_in_training() -> None:
    n = 20 * tsp.MIN_PRECISION_ROWS
    rng = np.random.default_rng(0)
    scores = rng.random(n)
    # Correctness rises with the score, so a precision target is reachable.
    correct = (rng.random(n) < scores * 0.8).astype(int)

    for target in (0.40, 0.55, 0.65):
        cut = tsp.threshold_for_precision(scores, correct, target)
        chosen = scores >= cut
        assert chosen.sum() >= tsp.MIN_PRECISION_ROWS
        assert correct[chosen].mean() >= target - 0.02


def test_threshold_is_infinite_when_the_target_is_unreachable() -> None:
    n = 10 * tsp.MIN_PRECISION_ROWS
    scores = np.linspace(0.0, 1.0, n)
    correct = np.zeros(n, dtype=int)
    assert np.isinf(tsp.threshold_for_precision(scores, correct, 0.60))


def test_unreachable_threshold_selects_nothing() -> None:
    part = _candidates([570, 600], [0.5, 0.9])
    cut = np.inf
    got = tsp.walk_serially(part, (part["score"] >= cut).to_numpy())
    assert got.empty


def test_threshold_prefers_the_lowest_cut_meeting_the_target() -> None:
    """A policy that trades more at the same precision is strictly better."""

    n = 10 * tsp.MIN_PRECISION_ROWS
    scores = np.linspace(1.0, 0.0, n)
    correct = np.ones(n, dtype=int)
    cut = tsp.threshold_for_precision(scores, correct, 0.90)
    assert cut == pytest.approx(scores.min())


def test_bootstrap_interval_brackets_the_sample_mean() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(10.0, 5.0, 800)
    sessions = np.repeat(np.arange(80), 10)
    lo, hi = tsp.bootstrap_ci(values, sessions)
    assert lo < values.mean() < hi


def test_every_declared_feature_is_knowable_at_the_decision_minute() -> None:
    """No outcome column may reach the feature list."""

    banned = ("gross_", "path_", "net_label", "profitable")
    for name in tsp.FEATURES:
        assert not name.startswith(banned), name
        assert "gross" not in name and "path" not in name


def test_a_precision_spike_on_a_handful_of_rows_cannot_set_the_threshold() -> None:
    """The defect the floor exists for.

    With a 50-row floor the shuffled-label null found a chance precision spike
    near the top of its own scores, latched a threshold there, and posted
    +$113/trade on 224 trades that meant nothing.
    """

    n = 10 * tsp.MIN_PRECISION_ROWS
    scores = np.linspace(1.0, 0.0, n)
    correct = np.zeros(n, dtype=int)
    correct[:100] = 1  # a perfect but tiny spike at the very top

    assert np.isinf(tsp.threshold_for_precision(scores, correct, 0.90))
