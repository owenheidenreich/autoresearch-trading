"""The declared intraday direction screen.

Synthetic data only. The point of these tests is that the screen cannot flatter
itself: mirror rules must be exact complements, a rule with no information must
not clear the bar, and a planted edge must be found.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import screen_intraday_direction as sc
from v5.research import statistics as st


# --------------------------------------------------------------------------
# the declaration is fixed before anything runs
# --------------------------------------------------------------------------


def test_the_family_size_is_the_grid_it_declares() -> None:
    assert sc.FAMILY_SIZE == len(sc.ENTRY_RULES) * len(sc.HOLDS_MINUTES)


def test_the_declaration_hash_changes_when_the_declaration_does() -> None:
    before = sc.declaration_hash()
    original = sc.HOLDS_MINUTES
    try:
        sc.HOLDS_MINUTES = (15, 60, 30)
        assert sc.declaration_hash() != before
    finally:
        sc.HOLDS_MINUTES = original
    assert sc.declaration_hash() == before


def test_the_declaration_names_every_rule_that_will_be_scored() -> None:
    assert set(sc.declaration()["entry_rules"]) == set(sc.ENTRY_RULES)


def test_the_bonferroni_level_is_stricter_than_a_single_hypothesis() -> None:
    assert st.bonferroni_quantile(sc.FAMILY_SIZE) > st.Z_95


def test_the_tail_level_inverts_the_quantile_it_is_paired_with() -> None:
    for z in (st.Z_95, st.bonferroni_quantile(14), 2.3263):
        assert st.normal_quantile(1.0 - sc._tail_level(z)) == pytest.approx(z, abs=1e-4)


# --------------------------------------------------------------------------
# the rules themselves
# --------------------------------------------------------------------------


def _features(**kwargs) -> dict:
    base = {
        "return_5m": np.zeros(4),
        "return_15m": np.zeros(4),
        "return_30m": np.zeros(4),
        "range_position": np.full(4, 0.5),
    }
    return base | {k: np.asarray(v, float) for k, v in kwargs.items()}


def test_mirror_rules_are_exact_opposites() -> None:
    f = _features(return_15m=[0.01, -0.01, 0.0, 0.02])
    assert np.array_equal(
        sc.momentum_with(f), -np.asarray(sc.momentum_against(f))
    )


def test_a_confirmation_rule_stands_down_when_the_windows_disagree() -> None:
    f = _features(return_5m=[0.01, 0.01, -0.01, 0.0], return_30m=[0.01, -0.01, -0.01, 0.01])
    got = sc.confirmed_momentum(f)
    assert list(got) == [1.0, 0.0, -1.0, 0.0]


def test_the_range_rules_only_act_at_the_edges() -> None:
    f = _features(range_position=[0.95, 0.5, 0.05, 0.7])
    assert list(sc.breakout_with(f)) == [1.0, 0.0, -1.0, 0.0]
    assert list(sc.contrarian_stretch(f)) == [-1.0, 0.0, 1.0, 0.0]


# --------------------------------------------------------------------------
# the underlying estimate
# --------------------------------------------------------------------------


def _pivot(spot: float, strikes=(5880.0, 5890.0, 5900.0, 5910.0, 5920.0)) -> pd.DataFrame:
    cols, vals = [], []
    for k in strikes:
        cols.append((k, "C"))
        vals.append(max(0.0, spot - k) + 12.0)
        cols.append((k, "P"))
        vals.append(max(0.0, k - spot) + 12.0)
    idx = pd.MultiIndex.from_tuples(cols, names=["strike", "right"])
    return pd.DataFrame([vals], index=["09:35"], columns=idx)


def test_parity_recovers_a_spot_between_the_strikes() -> None:
    """The whole reason this estimator replaced the old one."""

    got = sc.parity_spot(_pivot(5903.7))
    assert got["09:35"] == pytest.approx(5903.7, abs=1e-6)


def test_parity_is_continuous_not_snapped_to_the_strike_grid() -> None:
    a = sc.parity_spot(_pivot(5900.0))["09:35"]
    b = sc.parity_spot(_pivot(5901.5))["09:35"]
    assert b - a == pytest.approx(1.5, abs=1e-6)


def test_parity_refuses_a_chain_with_too_few_paired_strikes() -> None:
    assert sc.parity_spot(_pivot(5900.0, strikes=(5900.0, 5910.0))) is None


# --------------------------------------------------------------------------
# scoring cannot flatter itself
# --------------------------------------------------------------------------


def _table(n_sessions: int, edge: float, seed: int = 5) -> pd.DataFrame:
    """Sessions where ``momentum_with`` is right with probability ``edge``."""

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for slot in range(8):
            signal = rng.choice([-1.0, 1.0])
            right = rng.random() < edge
            move = abs(rng.normal(0, 12.0)) * (signal if right else -signal)
            # A near-ATM pair: the side that matches the move gains, the other loses.
            call = 40.0 * move - 60.0
            rows.append(
                {
                    "session": f"2025-{1 + s // 28:02d}-{1 + s % 28:02d}",
                    "entry_minute": f"1{slot}:00",
                    "hour": f"1{slot}",
                    "spot_move": move,
                    "return_5m": 0.001 * signal,
                    "return_15m": 0.001 * signal,
                    "return_30m": 0.001 * signal,
                    "range_position": 0.5,
                    "call_gross": call,
                    "put_gross": -call,
                    "call_premium": 1_000.0,
                    "put_premium": 1_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_a_rule_with_no_information_does_not_clear_the_bar() -> None:
    got = sc.score(_table(300, edge=0.5), "momentum_with", 25.0, st.bonferroni_quantile(14))
    assert not got["clears_zero"]
    assert got["accuracy"] == pytest.approx(0.5, abs=0.05)


def test_a_planted_edge_is_found() -> None:
    got = sc.score(_table(300, edge=0.62), "momentum_with", 25.0, st.bonferroni_quantile(14))
    assert got["clears_zero"]
    assert got["accuracy"] > 0.55
    assert got["bootstrap_lower_usd"] > 0


def test_the_mirror_of_a_winning_rule_loses() -> None:
    table = _table(300, edge=0.62)
    z = st.bonferroni_quantile(14)
    winner = sc.score(table, "momentum_with", 25.0, z)
    mirror = sc.score(table, "momentum_against", 25.0, z)
    assert winner["clears_zero"] and not mirror["clears_zero"]
    assert winner["accuracy"] + mirror["accuracy"] == pytest.approx(1.0, abs=1e-6)


def test_a_higher_cost_lowers_the_measured_edge() -> None:
    table = _table(300, edge=0.62)
    z = st.bonferroni_quantile(14)
    cheap = sc.score(table, "momentum_with", 3.08, z)
    dear = sc.score(table, "momentum_with", 25.0, z)
    assert cheap["mean_net_usd"] - dear["mean_net_usd"] == pytest.approx(21.92, abs=1e-6)
    assert cheap["accuracy"] == dear["accuracy"]


def test_a_rule_that_stands_down_reports_its_abstention() -> None:
    table = _table(200, edge=0.55)
    table["range_position"] = 0.5  # no edges touched, so the range rules abstain
    got = sc.score(table, "breakout_with", 25.0, st.bonferroni_quantile(14))
    assert got["trades"] == 0
    assert got["verdict"] == "too few trades"


def test_the_bound_is_stricter_for_a_larger_declared_family() -> None:
    table = _table(300, edge=0.58)
    single = sc.score(table, "momentum_with", 25.0, st.Z_95)
    family = sc.score(table, "momentum_with", 25.0, st.bonferroni_quantile(14))
    assert family["bootstrap_lower_usd"] <= single["bootstrap_lower_usd"]
