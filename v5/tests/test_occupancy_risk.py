"""The charter risk check that Phase 2 of job 24 has to pass before it chooses a cell."""
from __future__ import annotations

import numpy as np
import pytest

from v5.ops import check_occupancy_risk as risk


ACCOUNT = 10_000.0


def _run(win: float, loss: float, *, accuracy: float, size: float, trades: int = 10,
         sessions: int = 20, paths: int = 200, seed: int = 7,
         premium: float = 100.0) -> dict:
    """``size`` is expressed as a share of a $10,000 account, in dollars."""

    return risk.simulate(
        trades_per_session=trades,
        win_quantiles_usd=np.full(50, win * size * ACCOUNT),
        loss_quantiles_usd=np.full(50, loss * size * ACCOUNT),
        mean_premium_usd=premium,
        accuracy=accuracy,
        account_usd=ACCOUNT,
        sessions=sessions,
        paths=paths,
        rng=np.random.default_rng(seed),
    )


# --------------------------------------------------------------------------
# the declared limits are the charter's, not this module's
# --------------------------------------------------------------------------


def test_the_limits_come_from_the_signed_amendment() -> None:
    assert risk.DAILY_BREAKER == 0.05
    assert risk.SURVIVAL_FLOOR == 0.50
    assert risk.CHARTER_PREMIUM_SHARE == 0.13


def test_the_breaker_tolerance_is_declared_in_advance() -> None:
    assert 0.0 < risk.BREAKER_TOLERANCE <= 0.10
    assert 0.0 < risk.MIN_OCCUPANCY_RETAINED <= 1.0


def test_an_account_that_only_passes_by_never_trading_is_not_a_pass() -> None:
    """The failure mode the retention criterion exists to catch.

    A 3% loss per trade against a 5% breaker stops the session after two of ten
    slots. The account never halves and the breaker rate is inside tolerance
    only because it barely trades — which is not a configuration that delivers
    the occupancy it was chosen for.
    """

    got = _run(win=0.03, loss=-0.03, accuracy=0.0, size=1.0, trades=10, sessions=5)
    assert got["occupancy_kept"] == pytest.approx(0.2)
    assert got["passes_breaker_tolerance"] is False
    assert got["passes_survival_floor"] is True
    assert got["passes_occupancy_retention"] is False


# --------------------------------------------------------------------------
# resampling reproduces the measured distribution and invents nothing
# --------------------------------------------------------------------------


def test_resampling_only_ever_returns_measured_values() -> None:
    measured = np.array([-1.0, -0.3, 0.2, 1.4])
    drawn = risk.resample(np.random.default_rng(3), measured, (500,))
    assert set(np.unique(drawn)).issubset(set(measured))


def test_resampling_reproduces_the_distribution_it_was_given() -> None:
    measured = np.linspace(-1.0, 1.0, 101)
    drawn = risk.resample(np.random.default_rng(5), measured, (200_000,))
    assert drawn.mean() == pytest.approx(measured.mean(), abs=0.01)


# --------------------------------------------------------------------------
# the breaker
# --------------------------------------------------------------------------


def test_a_session_that_only_wins_never_trips_and_keeps_every_slot() -> None:
    got = _run(win=0.4, loss=-0.4, accuracy=1.0, size=0.13)
    assert got["share_of_sessions_hitting_the_breaker"] == 0.0
    assert got["occupancy_kept"] == pytest.approx(1.0)
    assert got["realised_trades_per_session"] == pytest.approx(10.0)


def test_the_breaker_stops_the_session_at_the_trade_that_crosses_it() -> None:
    """A 3% loss per trade crosses a 5% breaker on the second one.

    Five sessions only, so equity stays above the survival floor and the
    breaker is the sole thing being measured.
    """

    got = _run(win=0.03, loss=-0.03, accuracy=0.0, size=1.0, trades=10, sessions=5)
    assert got["share_of_sessions_hitting_the_breaker"] == pytest.approx(1.0)
    assert got["realised_trades_per_session"] == pytest.approx(2.0)
    assert got["occupancy_kept"] == pytest.approx(0.2)


def test_a_smaller_position_keeps_more_of_the_occupancy() -> None:
    big = _run(win=0.03, loss=-0.03, accuracy=0.5, size=1.0)
    small = _run(win=0.03, loss=-0.03, accuracy=0.5, size=0.1)
    assert small["occupancy_kept"] > big["occupancy_kept"]
    assert (
        small["share_of_sessions_hitting_the_breaker"]
        < big["share_of_sessions_hitting_the_breaker"]
    )


# --------------------------------------------------------------------------
# the survival floor
# --------------------------------------------------------------------------


def test_an_account_that_halves_is_dead_and_stops_trading() -> None:
    got = _run(win=0.4, loss=-0.4, accuracy=0.0, size=1.0, trades=1, sessions=30)
    assert got["share_of_years_breaching_the_survival_floor"] == pytest.approx(1.0)
    assert got["median_year_end_equity_multiple"] <= risk.SURVIVAL_FLOOR
    # Dead paths take no further trades, so realised occupancy falls far below
    # the one trade a session the clock offered.
    assert got["realised_trades_per_session"] < 1.0


def test_a_winning_account_survives_and_compounds() -> None:
    got = _run(win=0.4, loss=-0.4, accuracy=1.0, size=0.02, trades=5, sessions=30)
    assert got["share_of_years_breaching_the_survival_floor"] == 0.0
    assert got["median_year_end_equity_multiple"] > 1.0
    assert got["passes_survival_floor"]


def test_an_account_too_small_for_one_contract_takes_no_trades() -> None:
    """One contract per trade: below the charter ceiling nothing can be opened."""

    got = _run(win=0.4, loss=-0.4, accuracy=1.0, size=0.02, premium=5_000.0)
    assert not got["affordable_under_the_charter_ceiling"]
    assert got["realised_trades_per_session"] == 0.0
    assert got["median_year_end_equity_multiple"] == pytest.approx(1.0)


def test_the_ticket_share_falls_as_the_account_grows() -> None:
    small = risk.simulate(
        trades_per_session=1, win_quantiles_usd=np.full(4, 10.0),
        loss_quantiles_usd=np.full(4, -10.0), mean_premium_usd=1_000.0,
        accuracy=0.5, account_usd=10_000.0, sessions=2, paths=10,
        rng=np.random.default_rng(1),
    )
    big = risk.simulate(
        trades_per_session=1, win_quantiles_usd=np.full(4, 10.0),
        loss_quantiles_usd=np.full(4, -10.0), mean_premium_usd=1_000.0,
        accuracy=0.5, account_usd=100_000.0, sessions=2, paths=10,
        rng=np.random.default_rng(1),
    )
    assert small["ticket_share_of_account"] == pytest.approx(0.10)
    assert big["ticket_share_of_account"] == pytest.approx(0.01)


def test_both_verdicts_have_to_hold_for_a_size_to_pass() -> None:
    # Wins every trade, but each ticket is large enough that nothing trips: passes.
    good = _run(win=0.4, loss=-0.4, accuracy=1.0, size=0.02, sessions=30)
    assert good["passes_breaker_tolerance"] and good["passes_survival_floor"]
    # Loses every trade: fails both.
    bad = _run(win=0.4, loss=-0.4, accuracy=0.0, size=0.02, sessions=30)
    assert not bad["passes_breaker_tolerance"] or not bad["passes_survival_floor"]


# --------------------------------------------------------------------------
# what the simulation reports back
# --------------------------------------------------------------------------


def test_the_worst_session_is_recorded_and_is_a_loss() -> None:
    got = _run(win=0.4, loss=-0.4, accuracy=0.5, size=0.05, sessions=30)
    assert got["worst_session_return_p01"] < 0.0


def test_zero_variance_at_a_positive_edge_never_trips_the_breaker() -> None:
    """A sanity floor: the breaker must not fire on an account that only gains."""

    got = _run(win=0.1, loss=0.1, accuracy=0.5, size=0.13, sessions=50)
    assert got["share_of_sessions_hitting_the_breaker"] == 0.0
    assert got["share_of_years_breaching_the_survival_floor"] == 0.0
