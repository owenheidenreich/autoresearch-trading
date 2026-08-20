"""Phase 4b must split the lift honestly and must not smuggle in a new selector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.research.feature_information_decomposition import (
    GAIN_MINUTE_COLUMN,
    LOSS_MINUTE_COLUMN,
    NULL_DRAWS,
    ORDERING_MATERIAL_BAR_PP,
    PRIMARY_LABEL,
    DecompositionError,
    apply_verdict,
    decompose,
    matched_precision,
    permute_label_side,
    resolved_mask,
    select_top,
)
from v5.research.feature_information_preflight import _top_per_session


def _frame(rows: int = 60, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_session = max(10, rows // 20)
    sessions = np.repeat(
        [f"2024-01-{i + 1:02d}" for i in range(max(1, rows // per_session))], per_session
    )[:rows]
    gain = np.where(rng.random(rows) < 0.4, rng.integers(1, 60, rows).astype(float), np.nan)
    loss = np.where(rng.random(rows) < 0.4, rng.integers(1, 60, rows).astype(float), np.nan)
    label = np.where(np.isfinite(gain) & ~np.isfinite(loss), 1.0, 0.0)
    return pd.DataFrame(
        {
            "session": sessions,
            "ask": rng.random(rows) * 10.0,
            "realised_vol_15m": rng.random(rows),
            GAIN_MINUTE_COLUMN: gain,
            LOSS_MINUTE_COLUMN: loss,
            PRIMARY_LABEL: label,
        }
    )


# ------------------------------------------------------- the selector contract


def test_the_default_selector_is_the_phase_4a_selector_itself() -> None:
    """D1's precondition needs bit-identical selection, not an equivalent one."""

    rng = np.random.default_rng(4)
    sessions = np.repeat(["a", "b", "c"], 8)
    scores = rng.normal(size=24)
    np.testing.assert_array_equal(
        select_top(sessions, scores), _top_per_session(sessions, scores, 2)
    )


def test_tied_scores_are_broken_by_row_order_without_a_seed() -> None:
    """The contamination Phase 4a suffered, pinned so it stays visible."""

    sessions = np.array(["a"] * 5)
    selected = select_top(sessions, np.zeros(5))
    assert list(selected) == [True, True, False, False, False]


def test_a_seeded_tie_break_moves_the_selection_off_row_order() -> None:
    sessions = np.array(["a"] * 40)
    first = select_top(sessions, np.zeros(40), tie_break_seed=1)
    second = select_top(sessions, np.zeros(40), tie_break_seed=2)
    assert first.sum() == second.sum() == 2
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, select_top(sessions, np.zeros(40)))


def test_a_seeded_tie_break_is_reproducible() -> None:
    sessions = np.array(["a"] * 40)
    np.testing.assert_array_equal(
        select_top(sessions, np.zeros(40), tie_break_seed=7),
        select_top(sessions, np.zeros(40), tie_break_seed=7),
    )


def test_a_tie_break_never_overrides_a_real_score_difference() -> None:
    sessions = np.array(["a"] * 6)
    scores = np.array([9.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    for seed in range(5):
        selected = select_top(sessions, scores, tie_break_seed=seed)
        assert list(selected[:2]) == [True, True]


# ---------------------------------------------------------- the decomposition


def test_resolution_and_ordering_sum_to_the_lift() -> None:
    """The split is an identity; if it stops summing, it is not a split."""

    frame = _frame(120, seed=3)
    selected = np.zeros(len(frame), dtype=bool)
    selected[::10] = True
    selected[1::10] = True
    result = decompose(frame, selected, label_column=PRIMARY_LABEL)
    assert result.resolution_component + result.ordering_component == pytest.approx(result.lift)


def test_selecting_only_on_resolution_shows_no_ordering() -> None:
    """A selection that raises resolution but expresses no view on which barrier
    arrives first must land its whole lift in the resolution component."""

    rng = np.random.default_rng(11)
    rows = 600
    sessions = np.repeat([f"2024-02-{i + 1:02d}" for i in range(rows // 10)], 10)
    resolves = rng.random(rows) < 0.5
    gain_first = rng.random(rows) < 0.4  # independent of `resolves`
    frame = pd.DataFrame(
        {
            "session": sessions,
            "ask": rng.random(rows),
            "realised_vol_15m": rng.random(rows),
            GAIN_MINUTE_COLUMN: np.where(resolves, 1.0, np.nan),
            LOSS_MINUTE_COLUMN: np.where(resolves & ~gain_first, 2.0, np.nan),
            PRIMARY_LABEL: (resolves & gain_first).astype(float),
        }
    )
    # Select resolved rows only: resolution rises, the gain-first share does not.
    selected = np.zeros(rows, dtype=bool)
    for session in np.unique(sessions):
        rows_in = np.flatnonzero((sessions == session) & resolves)[:2]
        selected[rows_in] = True
    result = decompose(frame, selected, label_column=PRIMARY_LABEL)
    assert result.resolution_component > 0.10
    assert abs(result.ordering_component) < 0.06


def test_resolved_means_either_barrier_was_touched() -> None:
    frame = _frame(40, seed=5)
    expected = (
        np.isfinite(frame[GAIN_MINUTE_COLUMN].to_numpy(float))
        | np.isfinite(frame[LOSS_MINUTE_COLUMN].to_numpy(float))
    )
    np.testing.assert_array_equal(resolved_mask(frame), expected)


def test_a_frame_without_the_touch_minutes_is_refused() -> None:
    frame = _frame(20).drop(columns=[GAIN_MINUTE_COLUMN])
    with pytest.raises(DecompositionError, match="missing the label-side column"):
        resolved_mask(frame)


def test_selecting_nothing_is_refused_rather_than_averaged() -> None:
    frame = _frame(20)
    with pytest.raises(DecompositionError, match="no action was selected"):
        decompose(frame, np.zeros(len(frame), dtype=bool), label_column=PRIMARY_LABEL)


# ----------------------------------------------------------------- the null


def test_the_null_permutes_the_label_side_as_one_block() -> None:
    """Permuting the label alone would make rows resolved but carry another
    action's outcome, and the decomposition reads both halves."""

    frame = _frame(200, seed=9)
    permuted = permute_label_side(frame, seed=2)
    columns = [PRIMARY_LABEL, GAIN_MINUTE_COLUMN, LOSS_MINUTE_COLUMN]
    # NaN never equals itself, so the touch minutes are compared through a
    # sentinel rather than as floats.
    original = frame[columns].fillna(-1.0).apply(tuple, axis=1)
    shuffled = permuted[columns].fillna(-1.0).apply(tuple, axis=1)
    assert sorted(original.tolist()) == sorted(shuffled.tolist())
    assert not original.equals(shuffled)


def test_the_null_keeps_every_record_inside_its_own_session() -> None:
    frame = _frame(200, seed=6)
    permuted = permute_label_side(frame, seed=3)
    for session, group in frame.groupby("session"):
        mine = permuted[permuted["session"] == session]
        assert sorted(group[PRIMARY_LABEL]) == sorted(mine[PRIMARY_LABEL])
    assert permuted["session"].equals(frame["session"])


# ---------------------------------------------------------------- the verdict


def _observed(ordering: float, resolution: float = 0.05):
    frame = _frame(40)
    selected = np.zeros(len(frame), dtype=bool)
    selected[:4] = True
    base = decompose(frame, selected, label_column=PRIMARY_LABEL)
    return type(base)(
        **{**base.__dict__, "ordering_component": ordering, "resolution_component": resolution}
    )


def test_an_ordering_component_inside_its_null_establishes_b() -> None:
    verdict = apply_verdict(_observed(0.004), [0.5] * NULL_DRAWS)
    assert verdict["verdict"] == "B_RESOLUTION_MECHANICS"


def test_a_material_ordering_component_outside_its_null_gives_a() -> None:
    verdict = apply_verdict(_observed(0.04), [0.5] * NULL_DRAWS)
    assert verdict["verdict"] == "A_ORDERING_INFORMATION"
    assert verdict["ordering_component_pp"] >= ORDERING_MATERIAL_BAR_PP


def test_real_but_small_routes_to_the_owner() -> None:
    verdict = apply_verdict(_observed(0.015), [0.5] * NULL_DRAWS)
    assert verdict["verdict"] == "OWNER_DECISION_REAL_BUT_SMALL"


def test_too_few_null_draws_is_refused_rather_than_quantiled() -> None:
    with pytest.raises(DecompositionError, match="at least"):
        apply_verdict(_observed(0.04), [0.5] * (NULL_DRAWS - 1))


# ------------------------------------------------------------------ matching


def test_matching_compares_inside_price_and_volatility_strata() -> None:
    # Fifty strata per session need a session with more than fifty actions; the
    # real corpus carries about 3,260. A ten-row session puts every action in its
    # own stratum and matches nothing.
    frame = _frame(6_000, seed=12)
    selected = np.zeros(len(frame), dtype=bool)
    selected[::10] = True
    result = matched_precision(frame, selected, label_column=PRIMARY_LABEL)
    assert result["strata_with_both"] > 0
    assert np.isfinite(result["difference_pp"])


def test_matching_without_a_comparable_action_is_refused() -> None:
    frame = _frame(20)
    with pytest.raises(DecompositionError, match="no selected action had a match"):
        matched_precision(frame, np.ones(len(frame), dtype=bool), label_column=PRIMARY_LABEL)
