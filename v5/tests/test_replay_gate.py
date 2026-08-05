"""Regression tests for the four confirmed G5 defects, plus the gate's own rules.

Each defect test is written so that the *old* behaviour would pass and the new
gate refuses.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from v5.research.validation import replay_gate as rg


SESSIONS = 60


def _frame(policy_mean: float = 40.0, comparator_mean: float = 10.0, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    policy = rng.normal(policy_mean, 12.0, SESSIONS)
    comparator = rng.normal(comparator_mean, 12.0, SESSIONS)
    frame = pd.DataFrame(
        {
            "session": pd.date_range("2026-01-05", periods=SESSIONS, freq="B").astype(str),
            "fold": np.repeat(np.arange(5), SESSIONS // 5),
            "policy_fee15_lat0": policy,
            "policy_fee20_lat0": policy - 2.0,
            "comparator_fee15_lat0": comparator,
            "comparator_fee20_lat0": comparator - 2.0,
        }
    )
    for name in rg.CONTROL_NAMES:
        frame[f"control_{name}"] = rng.normal(-5.0, 12.0, SESSIONS)
    return frame


def _cells() -> list[rg.Cell]:
    return [
        rg.Cell("fee1.5_lat0", 1.5, 0, "policy_fee15_lat0", "comparator_fee15_lat0"),
        rg.Cell("fee2.0_lat0", 2.0, 0, "policy_fee20_lat0", "comparator_fee20_lat0"),
    ]


def _controls() -> dict[str, str]:
    return {name: f"control_{name}" for name in rg.CONTROL_NAMES}


def _power(index_sessions: int = SESSIONS) -> rg.PowerReceipt:
    return rg.make_power_receipt(
        declared_effect=30.0,
        session_sd=12.0,
        index_sessions=index_sessions,
        frozen_on="2026-08-01",
    )


def _lock() -> rg.ComparatorLock:
    return rg.make_comparator_lock(
        comparator_name="always_long",
        selected_on="2026-08-01",
        selection_basis="best inside training folds only",
    )


def _evaluate(frame=None, **overrides):
    kwargs = dict(
        cells=_cells(),
        controls=_controls(),
        power_receipt=_power(),
        comparator_lock=_lock(),
        evaluation_date="2026-08-05",
        skill_folds_positive=4,
        skill_null_passed_known_answer_gate=True,
    )
    kwargs.update(overrides)
    return rg.evaluate(_frame() if frame is None else frame, **kwargs)


# --------------------------------------------------------------------------
# The gate works at all
# --------------------------------------------------------------------------


def test_a_genuinely_strong_result_is_supported() -> None:
    result = _evaluate()
    assert result.verdict == "SUPPORTED", result.failures
    assert result.failures == []


def test_the_result_is_deterministic() -> None:
    assert _evaluate().to_dict()["receipt_sha256"] == _evaluate().to_dict()["receipt_sha256"]


# --------------------------------------------------------------------------
# Defect 1 — the fee cancelled
# --------------------------------------------------------------------------


def test_a_policy_that_loses_money_absolutely_cannot_pass_on_relative_uplift() -> None:
    """The old gate tested only policy minus comparator, so the fee dropped out.

    Here the policy beats its comparator in every cell while losing money. The
    old condition would be satisfied; the corrected gate must refuse.
    """

    frame = _frame(policy_mean=-20.0, comparator_mean=-60.0)
    result = _evaluate(frame)
    assert result.verdict == "NOT_SUPPORTED"
    assert any("absolute net lower bound" in failure for failure in result.failures)
    # The relative claim really is positive; only the absolute test catches it.
    assert result.cells["fee1.5_lat0"]["paired_lower_bound"] > 0.0


def test_the_fee_identity_is_stated_rather_than_counted_as_evidence() -> None:
    result = _evaluate()
    assert any("identical across fee levels" in note for note in result.notes)


# --------------------------------------------------------------------------
# Defect 2 — the time_shifted control imported the future
# --------------------------------------------------------------------------


def test_the_lag_control_uses_the_previous_row_not_the_next() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert list(rg.causal_lag_control(values)) == [0.0, 1.0, 2.0, 3.0]
    # The defective control was shift(-1), which is the next row.
    assert list(rg.causal_lag_control(values)) != [2.0, 3.0, 4.0, 0.0]


def test_a_control_that_passes_the_full_gate_makes_the_result_invalid() -> None:
    frame = _frame()
    frame["control_constant"] = frame["policy_fee15_lat0"]
    result = _evaluate(frame)
    assert result.verdict == "INVALID"
    assert result.controls["constant"]["passed_full_gate"] is True


def test_every_required_negative_control_must_be_supplied() -> None:
    result = _evaluate(controls={"constant": "control_constant"})
    assert any("missing required negative controls" in failure for failure in result.failures)


# --------------------------------------------------------------------------
# Defect 3 — one positive fold counted as skill
# --------------------------------------------------------------------------


def test_one_positive_skill_fold_is_not_enough() -> None:
    """The old code accepted `positive_target_skill_folds > 0`."""

    result = _evaluate(skill_folds_positive=1)
    assert result.verdict == "NOT_SUPPORTED"
    assert any("4 are required" in failure for failure in result.failures)


def test_four_positive_skill_folds_are_enough() -> None:
    assert _evaluate(skill_folds_positive=4).verdict == "SUPPORTED"


def test_a_skill_null_without_a_known_answer_gate_is_refused() -> None:
    result = _evaluate(skill_null_passed_known_answer_gate=False)
    assert any("known-answer gate" in failure for failure in result.failures)


def test_a_result_carried_by_one_fold_fails_the_fold_rule() -> None:
    frame = _frame()
    # One huge fold, four losing folds: the pooled mean is positive anyway.
    frame.loc[frame["fold"] == 0, "policy_fee15_lat0"] = 900.0
    frame.loc[frame["fold"] != 0, "policy_fee15_lat0"] = -5.0
    frame["policy_fee20_lat0"] = frame["policy_fee15_lat0"] - 2.0
    result = _evaluate(frame)
    assert result.verdict == "NOT_SUPPORTED"
    assert any("4-of-5 fold rule" in failure for failure in result.failures)


# --------------------------------------------------------------------------
# Defect 4 — "powered" counted rows
# --------------------------------------------------------------------------


def test_thirty_sessions_and_five_fold_labels_are_not_power() -> None:
    """The old condition was `sessions >= 30 and nunique(fold) == 5`.

    This index satisfies both and is still refused, because the receipt says the
    declared effect needs more sessions than the index has.
    """

    thin = _frame().head(30).copy()
    thin["fold"] = np.repeat(np.arange(5), 6)
    # A small edge against real session dispersion needs 36 sessions, not 30,
    # and the computed power on 30 is only ~0.74.
    small_edge = rg.make_power_receipt(
        declared_effect=5.0,
        session_sd=12.0,
        index_sessions=30,
        frozen_on="2026-08-01",
    )
    assert small_edge.required_sessions == 36
    assert small_edge.achieved_power < 0.80
    with pytest.raises(rg.ReplayGateError, match="power for the declared effect"):
        _evaluate(thin, power_receipt=small_edge)


def test_power_is_computed_not_asserted() -> None:
    """The old receipt took `achieved_power` on the caller's word."""

    with pytest.raises(TypeError):
        rg.make_power_receipt(
            declared_effect=30.0, session_sd=12.0, achieved_power=0.99,
            index_sessions=60, frozen_on="2026-08-01",
        )
    # An index exactly at the required count sits at ~80% analytic power.
    at_bound = rg.make_power_receipt(
        declared_effect=5.0, session_sd=12.0, index_sessions=36, frozen_on="2026-08-01"
    )
    assert 0.79 < at_bound.achieved_power < 0.85


def test_a_simulation_may_lower_power_but_never_raise_it() -> None:
    lowered = rg.make_power_receipt(
        declared_effect=30.0, session_sd=12.0, index_sessions=60,
        frozen_on="2026-08-01", simulated_power=0.55,
    )
    assert lowered.achieved_power == 0.55
    with pytest.raises(rg.ReplayGateError, match="power for the declared effect"):
        _evaluate(power_receipt=lowered)
    with pytest.raises(rg.ReplayGateError, match="may not claim more power"):
        rg.make_power_receipt(
            declared_effect=5.0, session_sd=12.0, index_sessions=30,
            frozen_on="2026-08-01", simulated_power=0.99,
        )


def test_a_frame_missing_declared_sessions_is_refused() -> None:
    """A dropped no-trade day inflates the mean; the gate must notice."""

    dropped = _frame().head(50)
    with pytest.raises(rg.ReplayGateError, match="never be dropped"):
        _evaluate(dropped)


def test_the_required_session_count_comes_from_the_effect_size() -> None:
    receipt = rg.make_power_receipt(
        declared_effect=1.0, session_sd=1.0,
        index_sessions=10_000, frozen_on="2026-08-01",
    )
    assert receipt.required_sessions == 7  # ceil(2.487 ** 2)


def test_a_tampered_power_receipt_is_refused() -> None:
    with pytest.raises(rg.ReplayGateError, match="self-hash mismatch"):
        _evaluate(power_receipt=replace(_power(), achieved_power=0.99))


# --------------------------------------------------------------------------
# Comparator freezing
# --------------------------------------------------------------------------


def test_a_comparator_reselected_on_the_evaluation_rows_is_refused() -> None:
    late = rg.make_comparator_lock(
        comparator_name="best_of_fifteen",
        selected_on="2026-08-05",
        selection_basis="highest pooled net on the evaluation rows",
    )
    with pytest.raises(rg.ReplayGateError, match="may not be reselected"):
        _evaluate(comparator_lock=late)


def test_a_tampered_comparator_lock_is_refused() -> None:
    with pytest.raises(rg.ReplayGateError, match="self-hash mismatch"):
        _evaluate(comparator_lock=replace(_lock(), comparator_name="something_else"))


# --------------------------------------------------------------------------
# Index hygiene and the bootstrap
# --------------------------------------------------------------------------


def test_duplicate_sessions_are_refused() -> None:
    frame = pd.concat([_frame(), _frame().head(1)], ignore_index=True)
    with pytest.raises(rg.ReplayGateError, match="one row per calendar session"):
        _evaluate(frame)


def test_the_block_bootstrap_is_seeded_and_reproducible() -> None:
    values = list(np.random.default_rng(1).normal(5.0, 2.0, 80))
    first = rg.block_bootstrap_lower_bound(values, seed=3)
    assert first == rg.block_bootstrap_lower_bound(values, seed=3)
    assert first < float(np.mean(values))


def test_blocking_is_more_conservative_than_independent_resampling() -> None:
    """Serial blocks must not produce a tighter bound than iid resampling."""

    rng = np.random.default_rng(5)
    drift = np.cumsum(rng.normal(0.0, 1.0, 200)) + 30.0
    iid = rg.block_bootstrap_lower_bound(list(drift), block_sessions=1, seed=11)
    blocked = rg.block_bootstrap_lower_bound(list(drift), block_sessions=20, seed=11)
    assert blocked <= iid


def test_an_empty_index_is_refused() -> None:
    with pytest.raises(rg.ReplayGateError, match="empty session index"):
        rg.block_bootstrap_lower_bound([])
