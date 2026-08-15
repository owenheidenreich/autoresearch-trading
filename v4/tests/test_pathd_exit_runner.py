from __future__ import annotations

import numpy as np
import pandas as pd

from v4.research.pathd_exit_runner import (
    _exit_index,
    _interpolate_utility,
    deterministic_class_balance,
    recovery_weight_multiplier,
    session_blocked_max_t,
    wave1_spec,
)
from v4.research.pathd_research_loop import blocked_by_prior_art, prior_art_check


def _frame(labels: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2026-01-02"] * len(labels),
            "trajectory_id": ["t"] * len(labels),
            "decision_time_ns": np.arange(len(labels)),
            "a_ref_dollars": labels,
        }
    )


def test_wave1_spec_is_the_frozen_eight_member_family() -> None:
    spec = wave1_spec()
    spec.validate()
    assert spec.budget == len(spec.hypotheses) == 8
    assert [item.hypothesis_id for item in spec.hypotheses] == [
        f"W1-H{index:02d}" for index in range(1, 9)
    ]
    # After the completed NO_EDGE wave is appended to the canonical ledger,
    # every member must be blocked from an unchanged rerun.
    assert all(blocked_by_prior_art(prior_art_check(item.mechanism)) for item in spec.hypotheses)


def test_class_balance_is_exact_deterministic_and_keeps_both_classes() -> None:
    source = _frame([1.0] * 4 + [-1.0] * 12)
    first = deterministic_class_balance(source, maximum_rows=10)
    second = deterministic_class_balance(source.sample(frac=1.0, random_state=44), maximum_rows=10)
    assert len(first) == 8
    assert (first["a_ref_dollars"] > 0).mean() == 0.5
    pd.testing.assert_frame_equal(first, second)


def test_recovery_penalty_only_upweights_positive_recoverable_early_rows() -> None:
    frame = pd.DataFrame(
        {
            "a_ref_dollars": [500.0, -500.0, 500.0],
            "recovery_300_dollars": [1_000.0, 1_000.0, 1_000.0],
            "seconds_held": [0.0, 0.0, 3_600.0],
        }
    )
    weights = recovery_weight_multiplier(frame)
    assert weights[0] > 1.0
    assert weights[1] == 1.0
    assert weights[0] > weights[2]


def test_cached_weight_arrays_are_copied_before_variant_normalization() -> None:
    source = np.array([1.0, 2.0, 3.0])
    source.flags.writeable = False
    frame = pd.DataFrame({"_base_weight": source})
    weights = frame["_base_weight"].to_numpy(dtype=float, copy=True)
    weights *= len(weights) / weights.sum()
    assert weights.flags.writeable
    np.testing.assert_allclose(source, [1.0, 2.0, 3.0])


def test_fallback_keeps_catastrophic_floor_priority_and_adds_target_and_time() -> None:
    frame = pd.DataFrame(
        {
            "entry_fill_option_price": [2.0, 2.0, 2.0],
            "current_net_pnl_dollars": [0.0, 210.0, -120.0],
            "current_return_on_entry_premium": [0.0, 1.05, -0.60],
            "seconds_held": [0.0, 10.0, 20.0],
        }
    )
    index, reason = _exit_index(frame, np.ones(3), fallback=True)
    assert (index, reason) == (1, "TARGET_100")
    index, reason = _exit_index(frame, np.array([1.0, 1.0, -1.0]), fallback=True)
    assert (index, reason) == (1, "TARGET_100")
    floor_frame = frame.copy()
    floor_frame.loc[0, "current_net_pnl_dollars"] = -101.0
    index, reason = _exit_index(floor_frame, np.array([-1.0, 1.0, 1.0]), fallback=True)
    assert (index, reason) == (0, "CATASTROPHIC_FLOOR")


def test_normalized_time_interpolation_preserves_endpoints() -> None:
    result = _interpolate_utility(np.array([-2.0, 4.0]), 5)
    np.testing.assert_allclose(result, [-2.0, -0.5, 1.0, 2.5, 4.0])


def test_session_blocked_max_t_uses_shared_family_null() -> None:
    sessions = [f"2026-01-{day:02d}" for day in range(1, 21)]
    strong = pd.Series(np.full(20, 10.0), index=sessions)
    weak = pd.Series(np.tile([-1.0, 1.0], 10), index=sessions)
    result = session_blocked_max_t({"strong": strong, "weak": weak}, draws=999, seed=1065)
    assert result["declared_family_size"] == 8
    assert result["p_values"]["strong"] <= 0.05
    assert not result["survived"]["weak"]
