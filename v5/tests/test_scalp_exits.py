"""The exit simulator, which is now the load-bearing piece.

Every prior conclusion assumed a clock exit. These tests pin the rules that make
a dynamic exit honest: a stop wins a tie inside a bar, a target fills at the
target rather than the extreme, and a gap through a stop does not fill at it.
"""
from __future__ import annotations

import numpy as np
import pytest

from v5.ops import measure_scalp_exits as se


def _bars(prices: list[tuple[float, float, float, float]]):
    """``(open, high, low, close)`` per minute, as four arrays."""

    arr = np.array(prices, dtype=float)
    return arr[:, 1], arr[:, 2], arr[:, 0], arr[:, 3]


def test_holding_to_the_horizon_takes_the_last_close() -> None:
    hi, lo, op, cl = _bars([(10, 12, 9, 11), (11, 14, 10, 13), (13, 13, 7, 8)])
    assert se.simulate_exit(hi, lo, op, cl, 10.0, (None, None, None)) == pytest.approx(8.0)


def test_a_target_fills_at_the_target_not_at_the_high() -> None:
    """Touching the level proves the limit filled; it does not grant the extreme."""

    hi, lo, op, cl = _bars([(10, 20, 9.5, 19)])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (0.20, None, None))
    assert got == pytest.approx(12.0)


def test_a_stop_fills_at_the_stop_when_the_bar_traded_through_it() -> None:
    hi, lo, op, cl = _bars([(10, 10.5, 7.0, 7.5)])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (None, -0.20, None))
    assert got == pytest.approx(8.0)


def test_a_gap_through_the_stop_fills_at_the_open_not_the_level() -> None:
    """Opening below the stop is a gap; the trade does not get the level."""

    hi, lo, op, cl = _bars([(6.0, 6.5, 5.0, 5.5)])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (None, -0.20, None))
    assert got == pytest.approx(6.0)


def test_the_stop_wins_a_tie_inside_one_bar() -> None:
    """A minute bar cannot order the two touches, so the loss is taken."""

    hi, lo, op, cl = _bars([(10, 13, 7, 12)])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (0.20, -0.20, None))
    assert got == pytest.approx(8.0)


def test_a_trailing_stop_follows_the_peak_up() -> None:
    # Rises to 20, then falls; a 30% trail exits at 20 * 0.7 = 14.
    hi, lo, op, cl = _bars([(10, 20, 10, 19), (19, 19, 12, 13)])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (None, None, 0.30))
    assert got == pytest.approx(14.0)


def test_a_trailing_stop_does_not_fire_on_the_bar_that_set_the_peak() -> None:
    """The peak is only known after the bar, so the trail cannot use it yet."""

    hi, lo, op, cl = _bars([(10, 20, 9.9, 19)])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (None, None, 0.30))
    # No exit triggered, so it rides to the final close.
    assert got == pytest.approx(19.0)


def test_missing_minutes_are_skipped_rather_than_treated_as_zero() -> None:
    hi = np.array([np.nan, 14.0])
    lo = np.array([np.nan, 10.0])
    op = np.array([np.nan, 11.0])
    cl = np.array([np.nan, 13.0])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (0.20, -0.50, None))
    assert got == pytest.approx(12.0)


def test_an_untouched_rule_rides_to_the_final_close() -> None:
    hi, lo, op, cl = _bars([(10, 10.5, 9.8, 10.2), (10.2, 10.6, 9.9, 10.4)])
    got = se.simulate_exit(hi, lo, op, cl, 10.0, (0.50, -0.50, None))
    assert got == pytest.approx(10.4)


def test_every_declared_rule_is_a_triple_of_declared_levels() -> None:
    for name, rule in se.EXIT_RULES.items():
        assert len(rule) == 3, name
        for level in rule:
            assert level is None or isinstance(level, float)


def test_the_baseline_rule_uses_no_levels_at_all() -> None:
    assert se.EXIT_RULES["hold_to_horizon"] == (None, None, None)


def test_the_cost_is_the_measured_round_trip() -> None:
    # Measured on 1,912,157 near-ATM contract-minutes, 2026-08-13.
    assert se.COST_PER_LEG_USD == 23.0
