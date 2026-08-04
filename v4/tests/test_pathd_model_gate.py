"""Tests for the Path-D acceptance gate and rejection tests.

Each test asserts the gate fires on the failure mode it exists to catch, and does
not fire on a healthy case. A gate that cannot fail is not a gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from v4.research.pathd_model_gate import (
    GateError,
    acceptance_tier,
    decile_monotonicity,
    exit_first_step_rate,
    label_balance,
    tail_preservation,
)


# --- exit_first_step_rate: the Protocol 029/030/031 signature ----------------


def test_exit_first_step_flags_immediate_exit_policy() -> None:
    result = exit_first_step_rate([0] * 951 + list(range(1, 50)))
    assert not result.passed
    assert result.metrics["first_step_rate"] > 0.9


def test_exit_first_step_accepts_a_distributed_policy() -> None:
    result = exit_first_step_rate([0] * 10 + list(range(1, 91)))
    assert result.passed


def test_exit_first_step_fails_closed_on_empty() -> None:
    with pytest.raises(GateError):
        exit_first_step_rate([])


# --- tail_preservation: amputating the convex tail --------------------------


def test_tail_preservation_flags_amputated_tail() -> None:
    rng = np.random.default_rng(0)
    baseline = np.concatenate([rng.normal(-50, 20, 900), rng.normal(3000, 500, 100)])
    policy = np.clip(baseline, None, 90.0)  # cap the upside, as an always-exit does
    result = tail_preservation(policy, baseline)
    assert not result.passed
    assert result.metrics["p99_ratio"] < 0.8


def test_tail_preservation_accepts_a_policy_that_keeps_the_tail() -> None:
    rng = np.random.default_rng(1)
    baseline = np.concatenate([rng.normal(-50, 20, 900), rng.normal(3000, 500, 100)])
    policy = baseline + 25.0  # strictly better, tail intact
    assert tail_preservation(policy, baseline).passed


def test_tail_preservation_requires_a_positive_baseline_tail() -> None:
    with pytest.raises(GateError):
        tail_preservation([-1.0] * 100, [-5.0] * 100)


# --- decile_monotonicity: April's closed coverage-threshold result ----------


def test_decile_monotonicity_flags_a_flat_ranker() -> None:
    rng = np.random.default_rng(2)
    scores = rng.normal(size=5000)
    outcomes = rng.normal(-40, 10, size=5000)  # score carries no information
    result = decile_monotonicity(scores, outcomes)
    assert not result.passed


def test_decile_monotonicity_accepts_a_real_ranker() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=5000)
    outcomes = 40.0 * scores + rng.normal(0, 5, size=5000)
    assert decile_monotonicity(scores, outcomes).passed


def test_decile_monotonicity_fails_closed_on_constant_score() -> None:
    with pytest.raises(GateError):
        decile_monotonicity([1.0] * 500, list(np.random.default_rng(4).normal(size=500)))


# --- label_balance: the 98%-exit-label disaster and its mirror --------------


def test_label_balance_flags_the_degenerate_low_rate() -> None:
    labels = [1.0] * 17 + [-1.0] * 83  # Path-D's 17% positive exit target
    assert not label_balance(labels).passed


def test_label_balance_flags_the_98_percent_disaster() -> None:
    labels = [1.0] * 98 + [-1.0] * 2
    assert not label_balance(labels).passed


def test_label_balance_accepts_a_usable_target() -> None:
    assert label_balance([1.0] * 45 + [-1.0] * 55).passed


# --- acceptance_tier --------------------------------------------------------


def _passing_tests():
    return [
        exit_first_step_rate([0] * 5 + list(range(1, 96))),
        label_balance([1.0] * 45 + [-1.0] * 55),
    ]


def _tier(**overrides):
    kwargs = dict(
        pooled_policy=100.0,
        pooled_comparator=50.0,
        fold_deltas={"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0, "4": -1.0},
        bootstrap_lcb=10.0,
        negative_controls_accepted={"sign_reversed": False, "shuffled": False},
        rejection_tests=_passing_tests(),
        maxt_survived=True,
        concentrated=False,
    )
    kwargs.update(overrides)
    return acceptance_tier(**kwargs)


def test_tier_a_requires_everything() -> None:
    assert _tier()["tier"] == "TIER_A"


def test_accepted_negative_control_forces_invalid_even_when_all_else_passes() -> None:
    verdict = _tier(negative_controls_accepted={"sign_reversed": True})
    assert verdict["tier"] == "INVALID"


def test_failed_rejection_test_blocks_tier_a() -> None:
    bad = _passing_tests() + [exit_first_step_rate([0] * 99 + [1])]
    assert _tier(rejection_tests=bad)["tier"] == "TIER_B"


def test_fold_instability_blocks_tier_a() -> None:
    deltas = {"0": 1.0, "1": -1.0, "2": -1.0, "3": -1.0, "4": -1.0}
    assert _tier(fold_deltas=deltas)["tier"] == "TIER_B"


def test_negative_lcb_blocks_tier_a() -> None:
    assert _tier(bootstrap_lcb=-1.0)["tier"] == "TIER_B"


def test_maxt_failure_blocks_tier_a() -> None:
    assert _tier(maxt_survived=False)["tier"] == "TIER_B"


def test_losing_to_comparator_is_no_edge() -> None:
    assert _tier(pooled_policy=10.0, pooled_comparator=50.0)["tier"] == "NO_EDGE"
