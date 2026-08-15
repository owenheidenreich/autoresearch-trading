"""Candidates are declared, enumerable, and countable before anything runs."""
from __future__ import annotations

import numpy as np
import pytest

from v5.research.autoresearch import budget as b, generator as g, outer


def _features(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "overnight_gap": rng.normal(0, 8, n),
        "first_five_minute_return": rng.normal(0, 5, n),
    }


def test_the_cross_product_is_enumerable_and_stable() -> None:
    cands = g.enumerate_candidates()
    assert len(cands) == len(g.ENTRY_RULES) * len(g.EXIT_RULES) == 24
    assert cands == g.enumerate_candidates(), "enumeration must be deterministic"
    assert len({c.name for c in cands}) == len(cands)


def test_the_multiplicity_is_knowable_before_the_loop_starts(tmp_path) -> None:
    """The whole point of enumerating rather than sampling."""

    led = b.AlphaLedger(
        tmp_path / "a.json", option_sessions=2520, universe="phase1_otm"
    )
    sweep = len(g.enumerate_candidates())
    bar_now = led.required_accuracy()
    bar_after = led.required_accuracy(experiments=sweep + 1)
    assert bar_after > bar_now
    # Measured: a full 24-candidate sweep costs 1.23 accuracy points on 2,520
    # sessions. Cheap enough to run, expensive enough that it must be counted.
    assert (bar_after - bar_now) == pytest.approx(0.0123, abs=0.002)


@pytest.mark.parametrize("bad", [{"entry": "nope"}, {"exit": "nope"}])
def test_an_undeclared_rule_is_refused(bad) -> None:
    base = {"entry": "gap_with", "exit": "stop_30"}
    with pytest.raises(g.GeneratorError, match="undeclared"):
        g.Candidate(**{**base, **bad})


def test_a_candidate_declares_only_the_features_its_rule_reads() -> None:
    assert g.Candidate("gap_with", "stop_30").features == ("overnight_gap",)
    assert set(g.Candidate("confirmed_gap", "stop_30").features) == {
        "overnight_gap",
        "first_five_minute_return",
    }


def test_calls_are_only_minus_one_zero_or_one() -> None:
    f = _features()
    for c in g.enumerate_candidates():
        calls = c.calls(f)
        assert set(np.unique(calls)) <= {-1, 0, 1}, c.name


def test_a_conjunction_rule_abstains_rather_than_guessing() -> None:
    """Standing down is a first-class outcome, not a failure."""

    f = {
        "overnight_gap": np.array([5.0, 5.0, -5.0]),
        "first_five_minute_return": np.array([2.0, -2.0, -2.0]),
    }
    confirmed = g.Candidate("confirmed_gap", "hold_to_horizon").calls(f)
    assert confirmed.tolist() == [1, 0, -1]
    contrarian = g.Candidate("contrarian_gap", "hold_to_horizon").calls(f)
    assert contrarian.tolist() == [0, -1, 0]


def test_opposite_entry_rules_are_exact_mirrors() -> None:
    f = _features()
    with_ = g.Candidate("gap_with", "stop_30").calls(f)
    against = g.Candidate("gap_against", "stop_30").calls(f)
    assert (with_ == -against).all()


def test_a_missing_feature_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(g.GeneratorError, match="missing features"):
        g.Candidate("confirmed_gap", "stop_30").calls(
            {"overnight_gap": np.array([1.0])}
        )


# --- the exit ----------------------------------------------------------------


def test_holding_to_the_horizon_returns_the_final_ratio() -> None:
    paths = np.array([[-0.1, -0.5, 0.2]])
    final = np.array([0.2])
    assert g.apply_exit(None, path_ratios=paths, final_ratio=final) == pytest.approx([0.2])


def test_a_stop_takes_the_fill_available_not_the_level_declared() -> None:
    """The slippage a declared stop cannot assume away."""

    paths = np.array([[-0.10, -0.45, 0.30]])  # gaps straight through -30%
    final = np.array([0.30])
    got = g.apply_exit(-0.30, path_ratios=paths, final_ratio=final)
    assert got == pytest.approx([-0.45]), "must fill at -45%, not the declared -30%"


def test_an_untouched_stop_leaves_the_trade_alone() -> None:
    paths = np.array([[-0.05, 0.10, 0.40]])
    final = np.array([0.40])
    assert g.apply_exit(-0.30, path_ratios=paths, final_ratio=final) == pytest.approx([0.40])


def test_a_stop_can_convert_a_winner_into_a_loser() -> None:
    """Measured at 19.9% of stopped paths; the cost must be visible, not netted."""

    paths = np.array([[-0.35, 0.50]])
    final = np.array([0.50])
    assert g.apply_exit(-0.30, path_ratios=paths, final_ratio=final) == pytest.approx([-0.35])


# --- composition with the outer loop -----------------------------------------


def test_the_generator_feeds_the_outer_loop_and_spends_alpha(tmp_path) -> None:
    led = b.AlphaLedger(
        tmp_path / "a.json", option_sessions=2520, universe="phase1_otm"
    )
    n = 200
    f = _features(n)
    rng = np.random.default_rng(1)
    realized = rng.normal(0, 8, n)
    cands = g.enumerate_candidates(entries=("gap_with", "confirmed_gap"), exits=("stop_30",))

    setting = outer.ConstraintSetting(
        setting_id="near_atm_60m",
        params={"moneyness_band": "near_atm"},
        rationale="the measured optimum of the strike ladder",
        declared_on="2026-08-13",
    )
    result = outer.run_setting(
        setting,
        ledger=led,
        inner=g.inner_loop(f, realized, candidates=cands, declared_on="2026-08-13"),
    )
    assert result.inner_experiments == 2
    assert led.experiments_run == 3  # one setting plus two candidates
