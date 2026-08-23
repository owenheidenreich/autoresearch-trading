"""What a contract requires, pinned — especially the inversion that cost this project money."""
from __future__ import annotations

import math

import pytest

from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD
from v5.research.contract_economics import (
    CONTRACT_MULTIPLIER,
    MAX_SEARCH_POINTS,
    contract_economics,
)

SPOT, SIGMA, SPREAD = 6800.0, 0.13, 0.20


def _ce(dist: float, minutes: float, hold: float = 20.0, is_call: bool = True, **kw):
    strike = SPOT + dist if is_call else SPOT - dist
    return contract_economics(
        spot=SPOT, strike=strike, minutes_to_expiry=minutes, hold_minutes=hold,
        sigma=kw.pop("sigma", SIGMA), spread=kw.pop("spread", SPREAD), is_call=is_call, **kw
    )


# ------------------------------------------------- the inversion that matters


def test_the_cheap_far_strike_needs_a_bigger_move_than_the_expensive_near_one() -> None:
    """The single most expensive misunderstanding in this project's ledger.

    `RIGHT IDEA, WRONG UNITS` closed a model that optimised percentage excursion,
    succeeded at it, and lost more than random. This is why: the cheap contract
    it preferred needs a far larger SPX move to break even. Measured here at 120
    minutes with a 20-minute hold, roughly 2.5 points for at-the-money against
    8.9 for fifty points out.
    """

    atm = _ce(0.0, 120.0)
    far = _ce(50.0, 120.0)
    assert far.entry_ask_usd < atm.entry_ask_usd, "the far strike is the cheap one"
    assert far.required_move_points > atm.required_move_points * 2.0, (
        f"cheap strike must require a bigger move: "
        f"{far.required_move_points:.1f} vs {atm.required_move_points:.1f}"
    )


def test_the_required_move_grows_as_expiry_approaches() -> None:
    """Same strike, less time: the move needed rises because decay outruns delta."""

    needs = [_ce(10.0, m).required_move_points for m in (120.0, 60.0, 30.0)]
    assert needs == sorted(needs), f"required move should rise as time runs out: {needs}"


def test_a_far_strike_close_to_expiry_is_effectively_unreachable() -> None:
    """Fifty points out with thirty minutes left needs a move that does not happen."""

    r = _ce(50.0, 30.0)
    assert r.required_move_points > 30.0
    assert r.entry_ask_usd < 50.0, "and it is cheap, which is exactly the trap"


def test_a_hopeless_contract_reports_a_huge_number_rather_than_pretending() -> None:
    """The honest answer for a far strike is a number, and the number is the verdict.

    A 200-point OTM call with ten minutes left is not *mathematically* impossible
    — it breaks even on a roughly 197-point move. It is impossible in the only
    sense that matters, and the module says so by reporting the requirement rather
    than by hiding it behind a flag. Whether 197 points is achievable is a question
    about the tape, not about the option, and is answered by the move-distribution
    work running alongside this.
    """

    r = _ce(200.0, 10.0, hold=9.0)
    assert r.required_move_points > 150.0
    assert r.entry_ask_usd < 30.0, "and it is nearly free, which is the whole trap"


def test_infinity_is_reserved_for_genuinely_unreachable_contracts() -> None:
    """Beyond the search bound there is no break-even at all, and that is not a number."""

    r = _ce(500.0, 10.0, hold=9.0)
    assert not r.reachable
    assert math.isinf(r.required_move_points) and math.isinf(r.required_move_pct)
    assert r.required_move_points > MAX_SEARCH_POINTS


# ----------------------------------------------------------------- mechanics


def test_you_start_in_a_hole_the_size_of_the_friction() -> None:
    """Decay-only P&L is negative before SPX has done anything."""

    r = _ce(0.0, 120.0)
    assert r.decay_only_pnl_usd < 0.0
    assert r.friction_usd == pytest.approx(SPREAD * CONTRACT_MULTIPLIER + FEES_PER_ROUND_TRIP_USD)


def test_a_wider_spread_requires_a_bigger_move() -> None:
    tight = _ce(10.0, 120.0, spread=0.05).required_move_points
    wide = _ce(10.0, 120.0, spread=0.50).required_move_points
    assert wide > tight


def test_higher_volatility_costs_more_and_demands_more() -> None:
    """A richer contract has more to lose to decay over the same hold."""

    calm = _ce(10.0, 120.0, sigma=0.08)
    wild = _ce(10.0, 120.0, sigma=0.25)
    assert wild.entry_ask_usd > calm.entry_ask_usd
    assert wild.required_move_points > calm.required_move_points


def test_a_longer_hold_bleeds_more_and_demands_more() -> None:
    short = _ce(10.0, 120.0, hold=5.0).required_move_points
    long_ = _ce(10.0, 120.0, hold=60.0).required_move_points
    assert long_ > short


def test_puts_mirror_calls_because_the_move_is_signed_for_them() -> None:
    """A put ten points below spot must need what a call ten points above needs."""

    call = _ce(10.0, 120.0, is_call=True)
    put = _ce(10.0, 120.0, is_call=False)
    assert put.required_move_points == pytest.approx(call.required_move_points, rel=0.05)
    assert put.entry_ask_usd == pytest.approx(call.entry_ask_usd, rel=0.10)


def test_a_hold_longer_than_the_life_of_the_contract_is_refused() -> None:
    r = _ce(10.0, 30.0, hold=45.0)
    assert not r.reachable and math.isnan(r.entry_ask_usd)


def test_the_required_move_is_also_reported_as_a_fraction_of_spot() -> None:
    """Points are era-dependent; a fraction of spot compares 2022 with 2026."""

    r = _ce(10.0, 120.0)
    assert r.required_move_pct == pytest.approx(r.required_move_points / SPOT)


def test_the_break_even_move_actually_breaks_even() -> None:
    """The solved root must reprice to roughly zero P&L, not merely be monotone."""

    from v5.research.contract_economics import _round_trip_pnl

    r = _ce(10.0, 120.0)
    pnl = _round_trip_pnl(
        r.required_move_points, spot=SPOT, strike=SPOT + 10.0, minutes_to_expiry=120.0,
        hold_minutes=20.0, sigma=SIGMA, spread=SPREAD, is_call=True,
        fees=FEES_PER_ROUND_TRIP_USD,
    )
    assert abs(pnl) < 1.0, f"solved move should break even, got {pnl:+.2f}"
