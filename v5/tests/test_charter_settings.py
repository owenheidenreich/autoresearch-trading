"""The ticket-size against breaker-level grid a charter amendment is read from."""
from __future__ import annotations

import numpy as np
import pytest

from v5.ops import check_occupancy_risk as risk
from v5.ops import search_charter_settings as cs


def _run(win: float, loss: float, *, accuracy: float, share: float, breaker: float,
         trades: int = 10, sessions: int = 30, paths: int = 400, seed: int = 3) -> dict:
    return cs.simulate_fractional(
        trades_per_session=trades,
        win_ratios=np.full(40, win),
        loss_ratios=np.full(40, loss),
        ticket_share=share,
        accuracy=accuracy,
        sessions=sessions,
        paths=paths,
        rng=np.random.default_rng(seed),
        daily_breaker=breaker,
    )


def test_the_breaker_ladder_starts_at_the_signed_value() -> None:
    assert cs.BREAKERS[0] == risk.DAILY_BREAKER
    assert list(cs.BREAKERS) == sorted(cs.BREAKERS)


def test_the_ladder_stops_below_the_survival_floor() -> None:
    """A daily breaker at half the account is not a daily control."""

    assert max(cs.BREAKERS) < risk.SURVIVAL_FLOOR


# --------------------------------------------------------------------------
# fractional sizing
# --------------------------------------------------------------------------


def test_a_wider_breaker_never_costs_occupancy() -> None:
    narrow = _run(0.5, -0.5, accuracy=0.5, share=0.08, breaker=0.05)
    wide = _run(0.5, -0.5, accuracy=0.5, share=0.08, breaker=0.25)
    assert wide["occupancy_kept"] >= narrow["occupancy_kept"]
    assert (
        wide["share_of_sessions_hitting_the_breaker"]
        <= narrow["share_of_sessions_hitting_the_breaker"]
    )


def test_the_breaker_stops_the_session_at_the_trade_that_crosses_it() -> None:
    # 3% of equity lost per trade against a 5% breaker: trips on the second.
    got = _run(0.3, -0.3, accuracy=0.0, share=0.10, breaker=0.05, sessions=3)
    assert got["realised_trades_per_session"] == pytest.approx(2.0)
    assert got["share_of_sessions_hitting_the_breaker"] == pytest.approx(1.0)


def test_a_smaller_ticket_survives_where_a_larger_one_does_not() -> None:
    big = _run(0.5, -0.5, accuracy=0.5, share=0.20, breaker=0.25, sessions=60)
    small = _run(0.5, -0.5, accuracy=0.5, share=0.01, breaker=0.25, sessions=60)
    assert (
        small["share_of_years_breaching_the_survival_floor"]
        < big["share_of_years_breaching_the_survival_floor"]
    )


def test_a_dead_account_stops_trading() -> None:
    got = _run(0.5, -0.5, accuracy=0.0, share=0.20, breaker=0.25, sessions=60)
    assert got["share_of_years_breaching_the_survival_floor"] == pytest.approx(1.0)
    assert got["occupancy_kept"] < 1.0


def test_equity_compounds_on_session_starting_capital() -> None:
    """Ten trades of +1% of equity each, thirty sessions, no losses."""

    got = _run(0.1, 0.1, accuracy=0.5, share=0.10, breaker=0.25, sessions=30, paths=50)
    assert got["median_year_end_equity_multiple"] == pytest.approx(1.1**30, rel=1e-6)
    assert got["occupancy_kept"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the verdict reports both readings
# --------------------------------------------------------------------------


def _cells(no_skill_passes: bool, skilled_breaker: float | None) -> list[dict]:
    out = []
    for breaker in cs.BREAKERS:
        for scenario in ("no_skill", "provable_skill"):
            passes = (
                no_skill_passes
                if scenario == "no_skill"
                else skilled_breaker is not None and breaker >= skilled_breaker
            )
            out.append(
                {
                    "scenario": scenario,
                    "daily_breaker": breaker,
                    "passes_breaker_tolerance": passes,
                    "passes_survival_floor": passes,
                    "passes_occupancy_retention": passes,
                    "share_of_years_breaching_the_survival_floor": 0.0 if passes else 0.4,
                }
            )
    return out


def test_the_verdict_names_the_narrowest_breaker_that_works_with_an_edge() -> None:
    got = cs.verdict(_cells(no_skill_passes=False, skilled_breaker=0.10))
    assert got["narrowest_breaker_passing_if_the_edge_is_real"] == pytest.approx(0.10)
    assert got["passes_with_no_skill"] is False


def test_the_verdict_carries_the_cost_of_being_wrong_at_that_setting() -> None:
    """A setting that only works with an edge must show what happens without one."""

    got = cs.verdict(_cells(no_skill_passes=False, skilled_breaker=0.10))
    assert got["no_skill_ruin_at_that_breaker"] == pytest.approx(0.4)
    assert got["no_skill_ruin_at_the_signed_breaker"] == pytest.approx(0.4)


def test_a_setting_that_survives_without_an_edge_is_reported_as_such() -> None:
    got = cs.verdict(_cells(no_skill_passes=True, skilled_breaker=0.05))
    assert got["passes_with_no_skill"] is True
    assert got["narrowest_breaker_passing_with_no_skill"] == pytest.approx(0.05)


def test_a_setting_that_never_works_reports_no_breaker() -> None:
    got = cs.verdict(_cells(no_skill_passes=False, skilled_breaker=None))
    assert got["narrowest_breaker_passing_if_the_edge_is_real"] is None
    assert got["no_skill_ruin_at_that_breaker"] is None


# --------------------------------------------------------------------------
# the resampling fix this work depends on
# --------------------------------------------------------------------------


def test_the_shared_resampler_draws_from_strata_not_quantiles() -> None:
    """Regression: uniform draws from a quantile grid trim the tail."""

    rng = np.random.default_rng(2)
    values = rng.lognormal(0.0, 2.0, size=20_000)
    from v5.ops.measure_hold_occupancy import strata_means

    strata = np.array(strata_means(values))
    drawn = risk.resample(np.random.default_rng(1), strata, (200_000,))
    assert drawn.mean() == pytest.approx(values.mean(), rel=0.02)
