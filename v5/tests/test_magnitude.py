"""The magnitude experiments: dataset, screen, variance premium, leg asymmetry.

The load-bearing tests here are the causality ones. The 2026-08-13 defect was a
filter that conditioned on an outcome, and the thresholds in the magnitude screen
are the obvious place for the same mistake to reappear: a whole-sample tercile
would let a session be judged against a distribution it helped create.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import measure_leg_asymmetry as la
from v5.ops import measure_variance_premium as vp
from v5.ops import screen_magnitude as sm


def _slots(n_sessions: int, *, straddle_gross: float, premium: float = 2_000.0,
           range_30: float = 5.0, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for slot in range(6):
            rows.append(
                {
                    "session": f"2025-{1 + s // 28:02d}-{1 + s % 28:02d}",
                    "entry_minute": f"1{slot}:00",
                    "minutes_to_close": 385 - 60 * slot,
                    "hold": 60,
                    "spot": 5_900.0,
                    "move": rng.normal(0, 8),
                    "abs_move": abs(rng.normal(0, 8)),
                    "range_15m": range_30 / 2,
                    "range_30m": range_30,
                    "range_60m": range_30 * 2,
                    "move_15m": 1.0,
                    "move_30m": 2.0,
                    "move_60m": 3.0,
                    "range_position": 0.5,
                    "session_range": 30.0,
                    "call_premium": premium / 2,
                    "put_premium": premium / 2,
                    "call_gross": straddle_gross / 2 + rng.normal(0, 5),
                    "put_gross": straddle_gross / 2 + rng.normal(0, 5),
                }
            )
    out = pd.DataFrame(rows)
    out["straddle_premium"] = out["call_premium"] + out["put_premium"]
    out["straddle_gross"] = out["call_gross"] + out["put_gross"]
    return out


# --------------------------------------------------------------------------
# the calibration must not see the session it judges
# --------------------------------------------------------------------------


def test_thresholds_never_include_the_session_they_judge() -> None:
    """The causality property the whole screen rests on."""

    early = _slots(60, straddle_gross=-20.0, range_30=2.0, seed=2)
    late = _slots(60, straddle_gross=-20.0, range_30=99.0, seed=3)
    late["session"] = late["session"].str.replace("2025-", "2026-", regex=False)
    table = sm.derived_features(pd.concat([early, late], ignore_index=True))

    got = sm.expanding_thresholds(table)
    first_late = sorted(late["session"].unique())[0]
    # By the first high-range session, the pool holds only low-range sessions,
    # so its threshold cannot yet have been lifted by its own values.
    assert got.loc[first_late, "range_30m_0.67"] < 50.0


def test_the_first_sessions_have_no_threshold_at_all() -> None:
    table = sm.derived_features(_slots(5, straddle_gross=-20.0))
    got = sm.expanding_thresholds(table)
    first = sorted(table["session"].unique())[0]
    assert not np.isfinite(got.loc[first, "range_30m_0.33"])


def test_a_rule_is_not_scored_on_sessions_without_thresholds() -> None:
    """An uncalibrated session must contribute no trades, not default ones."""

    table = sm.derived_features(_slots(6, straddle_gross=-20.0))
    thresholds = sm.expanding_thresholds(table)
    got = sm.score(table, thresholds, "high_recent_range", 2.7)
    assert got["trades"] == 0


# --------------------------------------------------------------------------
# scoring is the profit and loss of a position, never the size of the move
# --------------------------------------------------------------------------


def test_the_baseline_takes_every_calibrated_slot() -> None:
    table = sm.derived_features(_slots(400, straddle_gross=-20.0))
    thresholds = sm.expanding_thresholds(table)
    got = sm.score(table, thresholds, "always", 2.7)
    assert got["share_of_slots"] > 0.9


def test_the_score_is_half_the_straddle_less_one_round_trip() -> None:
    table = sm.derived_features(_slots(400, straddle_gross=-20.0))
    thresholds = sm.expanding_thresholds(table)
    got = sm.score(table, thresholds, "always", 2.7)
    assert got["mean_net_usd"] == pytest.approx(-10.0 - sm.COST_PER_LEG_USD, abs=1.0)


def test_a_window_that_pays_clears_zero_and_one_that_does_not_fails() -> None:
    z = 2.7
    good = _slots(400, straddle_gross=200.0)
    bad = _slots(400, straddle_gross=-20.0)
    for table, expected in ((good, True), (bad, False)):
        prepared = sm.derived_features(table)
        got = sm.score(prepared, sm.expanding_thresholds(prepared), "always", z)
        assert got["clears_zero"] is expected


def test_the_baseline_is_outside_the_declared_family() -> None:
    assert sm.FAMILY_SIZE == (len(sm.RULES) - 1) * 2
    assert sm.declaration()["baseline_excluded_from_family"] == "always"


def test_the_declaration_hash_moves_with_the_rule_set() -> None:
    before = sm.declaration_hash()
    original = sm.RULES
    try:
        sm.RULES = dict(original) | {"extra": sm.rule_always}
        assert sm.declaration_hash() != before
    finally:
        sm.RULES = original
    assert sm.declaration_hash() == before


# --------------------------------------------------------------------------
# the variance premium measurement
# --------------------------------------------------------------------------


def test_a_seller_keeps_what_a_buyer_pays_less_two_round_trips() -> None:
    got = vp.summarise(_slots(200, straddle_gross=-60.0), "test")
    assert got["mean_straddle_gross_usd"] == pytest.approx(-60.0, abs=1.0)
    for cost in vp.COSTS_PER_LEG_USD:
        cell = got["by_cost"][f"${cost}/leg"]
        assert cell["buyer_net_usd"] == pytest.approx(
            got["mean_straddle_gross_usd"] - 2 * cost, abs=1e-6
        )
        assert cell["seller_net_usd"] == pytest.approx(
            -got["mean_straddle_gross_usd"] - 2 * cost, abs=1e-6
        )


def test_both_sides_can_lose_at_once_when_the_spread_exceeds_the_premium() -> None:
    """The result that decides the whole question, as an invariant."""

    got = vp.summarise(_slots(200, straddle_gross=-10.0), "test")
    dear = got["by_cost"]["$25.0/leg"]
    assert not dear["buyer_profitable"]
    assert not dear["seller_profitable"]


def test_a_premium_larger_than_the_spread_lets_the_seller_win() -> None:
    got = vp.summarise(_slots(200, straddle_gross=-200.0), "test")
    assert got["by_cost"]["$25.0/leg"]["seller_profitable"]


def test_the_confidence_interval_brackets_the_mean() -> None:
    got = vp.summarise(_slots(300, straddle_gross=-60.0), "test")
    lo, hi = got["gross_ci95_usd"]
    assert lo < got["mean_straddle_gross_usd"] < hi


# --------------------------------------------------------------------------
# leg asymmetry
# --------------------------------------------------------------------------


def test_a_symmetric_chain_shows_no_skew() -> None:
    got = la.leg_summary(_slots(300, straddle_gross=-40.0), "test")
    assert not got["call_minus_put"]["distinguishable_from_zero"]


def test_a_dear_put_is_detected() -> None:
    table = _slots(300, straddle_gross=-40.0)
    table["put_gross"] = table["put_gross"] - 200.0
    table["straddle_gross"] = table["call_gross"] + table["put_gross"]
    got = la.leg_summary(table, "test")
    assert got["call_minus_put"]["distinguishable_from_zero"]
    assert got["call_minus_put"]["mean_usd"] > 100.0
