"""The conditional-drift census is only worth anything if its cuts are causal.

Every other guard in this module is arithmetic. This one is the finding: if a
cell boundary reads a session that had not happened yet, the census reports
edges the bot could never have stood in, and the whole conclusion inverts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import measure_conditional_drift as mcd


def _sessions(n_sessions: int, per_session: int) -> tuple[np.ndarray, list[str]]:
    order = [f"2024-01-{i + 1:03d}" for i in range(n_sessions)]
    sessions = np.repeat(np.array(order), per_session)
    return sessions, order


def test_expanding_cuts_cannot_read_a_later_session() -> None:
    """Change only the last session; no earlier assignment may move."""

    sessions, order = _sessions(200, 10)
    rng = np.random.default_rng(0)
    values = rng.normal(size=len(sessions))

    before = mcd.expanding_cells(values.copy(), sessions, order)

    altered = values.copy()
    altered[sessions == order[-1]] = 1_000_000.0
    after = mcd.expanding_cells(altered, sessions, order)

    earlier = sessions != order[-1]
    assert np.array_equal(before[earlier], after[earlier])


def test_warmup_sessions_are_unassigned() -> None:
    sessions, order = _sessions(mcd.WARMUP_SESSIONS + 20, 40)
    values = np.random.default_rng(1).normal(size=len(sessions))
    cells = mcd.expanding_cells(values, sessions, order)

    inside = np.isin(sessions, order[: mcd.WARMUP_SESSIONS])
    assert (cells[inside] == -1).all()
    assert (cells[~inside] >= 0).any()


def test_cuts_fully_readapt_within_one_trailing_window() -> None:
    """The regression this was written for.

    A window that never forgets keeps cutting new-regime slots on old-regime
    quantiles, so every late slot lands in the top cells and the cell stops
    describing the observable and starts describing the calendar. Measured on
    the first draft: the last fifty sessions occupied cells {2, 3} only.
    """

    shift, n_sessions = 200, 200 + mcd.TRAILING_SESSIONS + 100
    sessions, order = _sessions(n_sessions, 20)
    rng = np.random.default_rng(2)
    values = np.where(
        np.isin(sessions, order[shift:]),
        rng.normal(100.0, 1.0, size=len(sessions)),
        rng.normal(0.0, 1.0, size=len(sessions)),
    )
    cells = mcd.expanding_cells(values, sessions, order)

    # These sessions see a trailing window containing only the new regime.
    late = np.isin(sessions, order[shift + mcd.TRAILING_SESSIONS :])
    assert sorted(set(cells[late].tolist())) == [0, 1, 2, 3]


def test_cells_stay_balanced_so_they_do_not_become_calendar_buckets() -> None:
    """Under a drifting scale, each cell must keep drawing from every era."""

    sessions, order = _sessions(800, 20)
    rng = np.random.default_rng(11)
    era = np.array([order.index(s) for s in sessions])
    # Scale grows sevenfold across the sample, as the real corpus does.
    values = rng.lognormal(0.0, 0.5, size=len(sessions)) * (1.0 + 6.0 * era / len(order))
    cells = mcd.expanding_cells(values, sessions, order)

    scored = cells >= 0
    late = era >= 600
    for cell in range(4):
        here = scored & (cells == cell)
        share_late = late[here].mean()
        # A fully expanding window drives this to 0.0 for low cells and 1.0 for
        # high ones. A trailing window keeps every cell near its 25% base rate.
        assert 0.15 < share_late < 0.40, f"cell {cell} share_late={share_late:.3f}"


def test_missing_values_are_refused_not_bucketed() -> None:
    sessions, order = _sessions(120, 10)
    values = np.random.default_rng(4).normal(size=len(sessions))
    values[5] = np.nan
    values[-1] = np.nan
    cells = mcd.expanding_cells(values, sessions, order)
    assert cells[-1] == -1


def test_fixed_cells_bucket_on_the_declared_edges() -> None:
    got = mcd.fixed_cells(np.array([0.0, 60.0, 61.0, 180.0, 400.0]), (60.0, 180.0, 300.0))
    assert got.tolist() == [0, 1, 1, 2, 3]


def test_permutation_preserves_each_session_multiset() -> None:
    sessions, _ = _sessions(30, 8)
    index = pd.factorize(sessions)[0]
    values = np.arange(len(sessions), dtype=float)
    keys = np.random.default_rng(5).random(len(sessions))

    shuffled = mcd.permuted_values(values, keys, index)

    for s in np.unique(index):
        assert sorted(values[index == s]) == sorted(shuffled[index == s])


def test_dispersion_is_zero_when_cells_agree_and_positive_when_they_do_not() -> None:
    cells = np.repeat([0, 1, 2, 3], 500)
    flat = np.tile([1.0, -1.0], 1000)
    assert mcd.dispersion(flat, cells, 4) == pytest.approx(0.0, abs=1e-12)

    planted = flat + np.repeat([0.0, 0.0, 0.0, 5.0], 500)
    assert mcd.dispersion(planted, cells, 4) > 1.0


def test_dispersion_ignores_cells_below_the_declared_minimum() -> None:
    cells = np.concatenate([np.zeros(500, int), np.ones(500, int), np.full(3, 2)])
    values = np.concatenate([np.zeros(1000), np.full(3, 99.0)])
    # The three-slot cell is far from the others and must not be allowed to
    # create dispersion out of a sample too small to mean anything.
    assert mcd.dispersion(values, cells, 3) == pytest.approx(0.0, abs=1e-12)


def test_planted_signal_beats_its_own_permutation_null() -> None:
    """A real within-session effect must survive the null the census uses."""

    sessions, _ = _sessions(200, 12)
    index = pd.factorize(sessions)[0]
    rng = np.random.default_rng(6)
    cells = np.tile(np.repeat([0, 1, 2, 3], 3), 200)
    values = rng.normal(size=len(sessions)) + np.where(cells == 3, 1.0, 0.0)

    observed = mcd.dispersion(values, cells, 4)
    null = np.array(
        [
            mcd.dispersion(
                mcd.permuted_values(values, rng.random(len(values)), index), cells, 4
            )
            for _ in range(200)
        ]
    )
    assert (null >= observed).mean() < 0.05


def test_pure_noise_does_not_beat_the_null() -> None:
    sessions, _ = _sessions(200, 12)
    index = pd.factorize(sessions)[0]
    rng = np.random.default_rng(7)
    cells = np.tile(np.repeat([0, 1, 2, 3], 3), 200)
    values = rng.normal(size=len(sessions))

    observed = mcd.dispersion(values, cells, 4)
    null = np.array(
        [
            mcd.dispersion(
                mcd.permuted_values(values, rng.random(len(values)), index), cells, 4
            )
            for _ in range(200)
        ]
    )
    assert (null >= observed).mean() > 0.05


def test_within_session_null_is_powerless_on_a_session_constant_state() -> None:
    """The defect this pair of nulls exists to fix.

    A state that never varies inside a session cannot be re-celled by a
    within-session shuffle, so the null lands exactly on the observed value and
    reports p=1.000 — "no information" — having measured nothing. The first run
    of this census produced exactly that for six of eleven states.
    """

    sessions, _ = _sessions(120, 10)
    index = pd.factorize(sessions)[0]
    rng = np.random.default_rng(8)
    cells = np.repeat(np.arange(120) % 4, 10)  # constant within each session
    values = rng.normal(size=len(sessions)) + np.where(cells == 3, 2.0, 0.0)

    observed = mcd.dispersion(values, cells, 4)
    null = np.array(
        [
            mcd.dispersion(
                mcd.permuted_values(values, rng.random(len(values)), index), cells, 4
            )
            for _ in range(200)
        ]
    )
    # A planted signal this large must not be reported as absent. The null sits
    # on top of the observed value (it differs only in float summation order),
    # so the p-value is pinned near 1.0 and carries no information either way.
    assert (null >= observed).mean() > 0.5

    rows, columns = mcd.session_layout(index, 120)
    swapped = np.array(
        [
            mcd.dispersion(
                mcd.swapped_sessions(values, rows, columns, 120, rng), cells, 4
            )
            for _ in range(200)
        ]
    )
    assert (swapped >= observed).mean() < 0.05


def test_dispersion_never_returns_nan_on_a_ragged_session_swap() -> None:
    """The second defect: a NaN statistic reads as evidence for the state.

    Sessions have different slot counts, so a swap leaves gaps. Unmasked, those
    gaps turned the statistic into NaN, `null >= observed` into False, and every
    ragged draw into a vote for the state — pinning six states at p=0.000.
    """

    lengths = np.random.default_rng(10).integers(4, 10, 60)
    index = np.concatenate([np.full(k, i) for i, k in enumerate(lengths)])
    rows, columns = mcd.session_layout(index, 60)
    rng = np.random.default_rng(12)
    values = rng.normal(size=len(index))
    cells = rng.integers(0, 2, size=len(index))

    for _ in range(50):
        swapped = mcd.swapped_sessions(values, rows, columns, 60, rng)
        assert not np.isfinite(swapped).all(), "test needs ragged sessions to bite"
        assert np.isfinite(mcd.dispersion(swapped, cells, 2))


def test_session_swap_preserves_every_payoff_exactly_once() -> None:
    sessions, _ = _sessions(40, 6)
    index = pd.factorize(sessions)[0]
    rows, columns = mcd.session_layout(index, 40)
    values = np.arange(len(sessions), dtype=float)

    got = mcd.swapped_sessions(values, rows, columns, 40, np.random.default_rng(9))

    assert sorted(got.tolist()) == sorted(values.tolist())


def test_session_layout_numbers_slots_inside_each_session() -> None:
    index = np.array([0, 0, 0, 1, 1, 2])
    rows, columns = mcd.session_layout(index, 3)
    assert rows.tolist() == [0, 0, 0, 1, 1, 2]
    assert columns.tolist() == [0, 1, 2, 0, 1, 0]


def test_within_session_share_separates_the_two_kinds_of_state() -> None:
    index = np.repeat(np.arange(50), 8)
    slot_position = np.tile(np.arange(8), 50).astype(float)
    session_constant = np.repeat(np.arange(50), 8).astype(float)

    assert mcd.within_session_share(slot_position, index) > 0.95
    assert mcd.within_session_share(session_constant, index) < 0.05


def test_bootstrap_arithmetic_reproduces_the_sample_mean() -> None:
    """Drawing every session exactly once must return the plain mean."""

    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    session_index = np.array([0, 0, 1, 1, 2])
    mask = np.ones(len(values), bool)
    sums, counts = mcd.session_totals(values, session_index, 3, mask)

    once = np.ones((1, 3))
    got = (once @ sums[:, None]) / (once @ counts[:, None])
    assert got.item() == pytest.approx(values.mean())


def test_cross_cells_refuse_when_either_parent_refuses() -> None:
    table = pd.DataFrame(
        {
            "session": ["2024-01-01"] * 4,
            "minutes_to_close": [30.0, 200.0, 30.0, 200.0],
            "implied_over_realised": [np.nan, 1.0, 2.0, 3.0],
        }
    )
    first = mcd.fixed_cells(table["minutes_to_close"].to_numpy(float), (60.0, 180.0, 300.0))
    second = np.array([-1, 0, 1, 2])
    width = int(second.max()) + 1
    crossed = np.where((first < 0) | (second < 0), -1, first * width + second)
    assert crossed[0] == -1
    assert (crossed[1:] >= 0).all()


def test_declared_grid_is_hashed_and_stable() -> None:
    assert mcd.grid_hash() == mcd.grid_hash()
    assert len(mcd.grid_hash()) == 64


def test_every_declared_state_says_why_it_is_there() -> None:
    for spec in mcd.DECLARED_STATES:
        assert spec["why"].strip()
        assert spec["kind"] in {"fixed", "expanding", "session", "cross"}
