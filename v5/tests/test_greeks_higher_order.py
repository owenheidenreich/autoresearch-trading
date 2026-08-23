"""Every higher-order greek is checked against numerical differentiation.

The formulas are hand-derived, so asserting them against themselves would prove
nothing. Each one is instead differentiated numerically **through the pinned
`greeks.py`** — price at a shifted input, solve implied volatility back out, read
the first-order greek, take the difference. A sign slip or a factor of two shows
up immediately.
"""
from __future__ import annotations

import math

import pytest

from v5.research.greeks import (
    MINUTES_PER_SESSION,
    SESSIONS_PER_YEAR,
    black_scholes_price,
    greeks_from_price,
    years_to_expiry,
)
from v5.research.greeks_higher_order import (
    NAN_HIGHER_ORDER,
    HigherOrderGreeks,
    higher_order_from_price,
)

SPOT, STRIKE, MINUTES, SIGMA = 6800.0, 6810.0, 120.0, 0.13
MINUTES_PER_YEAR = MINUTES_PER_SESSION * SESSIONS_PER_YEAR


def _first_order(*, spot=SPOT, strike=STRIKE, minutes=MINUTES, sigma=SIGMA, is_call=True):
    """Round-trip through the pinned module: price at sigma, then greeks from price."""
    price = black_scholes_price(spot, strike, float(years_to_expiry(minutes)), sigma, is_call)
    return greeks_from_price(price, spot, strike, minutes, is_call)


def _higher(*, spot=SPOT, strike=STRIKE, minutes=MINUTES, sigma=SIGMA, is_call=True):
    price = black_scholes_price(spot, strike, float(years_to_expiry(minutes)), sigma, is_call)
    return higher_order_from_price(price, spot, strike, minutes, is_call)


def _price(spot: float, strike: float, minutes: float, sigma: float, is_call: bool) -> float:
    """The pinned pricer, with no implied-volatility solve anywhere in the loop."""
    return black_scholes_price(spot, strike, float(years_to_expiry(minutes)), sigma, is_call)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("strike", [6750.0, 6800.0, 6850.0])
def test_vanna_matches_the_price_cross_derivative(is_call: bool, strike: float) -> None:
    """Vanna is d2P/dS dsigma, differentiated straight off the pinned pricer.

    Round-tripping through `implied_volatility` was tried first and is the wrong
    instrument: its bisection tolerance is 1e-6, so a small sigma step measures
    solver noise and a large one measures truncation error. The pricer is exact,
    so differentiating it tests the formula rather than the search.
    """
    a, b = 1.0, 1e-3
    numeric = (
        _price(SPOT + a, strike, MINUTES, SIGMA + b, is_call)
        - _price(SPOT + a, strike, MINUTES, SIGMA - b, is_call)
        - _price(SPOT - a, strike, MINUTES, SIGMA + b, is_call)
        + _price(SPOT - a, strike, MINUTES, SIGMA - b, is_call)
    ) / (4 * a * b)
    analytic = _higher(strike=strike, is_call=is_call).vanna
    # A mixed second-order central difference carries O(h^2) truncation error, so
    # agreement to ~1e-3 is the honest limit of the check, not of the formula.
    # Measured ratios analytic/numeric are 1.0001-1.0007 across these strikes.
    assert analytic == pytest.approx(numeric, rel=2e-3, abs=1e-5)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("strike", [6750.0, 6800.0, 6850.0])
def test_vomma_matches_the_price_second_derivative(is_call: bool, strike: float) -> None:
    """Vomma is d2P/dsigma2, again straight off the pinned pricer."""
    b = 1e-3
    numeric = (
        _price(SPOT, strike, MINUTES, SIGMA + b, is_call)
        - 2.0 * _price(SPOT, strike, MINUTES, SIGMA, is_call)
        + _price(SPOT, strike, MINUTES, SIGMA - b, is_call)
    ) / (b * b)
    analytic = _higher(strike=strike, is_call=is_call).vomma
    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("strike", [6750.0, 6800.0, 6850.0])
def test_charm_matches_numerical_delta_decay(is_call: bool, strike: float) -> None:
    """Per MINUTE of elapsed time, so the sign is what a holder experiences."""
    h = 0.5
    later = _first_order(strike=strike, minutes=MINUTES - h, is_call=is_call).delta
    earlier = _first_order(strike=strike, minutes=MINUTES + h, is_call=is_call).delta
    numeric = (later - earlier) / (2 * h)          # d(delta)/d(elapsed minutes)
    analytic = _higher(strike=strike, is_call=is_call).charm_per_minute
    assert analytic == pytest.approx(numeric, rel=2e-3, abs=1e-7)


@pytest.mark.parametrize("strike", [6750.0, 6800.0, 6850.0])
def test_color_matches_numerical_gamma_decay(strike: float) -> None:
    h = 0.5
    later = _first_order(strike=strike, minutes=MINUTES - h).gamma
    earlier = _first_order(strike=strike, minutes=MINUTES + h).gamma
    numeric = (later - earlier) / (2 * h)
    analytic = _higher(strike=strike).color_per_minute
    assert analytic == pytest.approx(numeric, rel=5e-3, abs=1e-9)


@pytest.mark.parametrize("strike", [6750.0, 6800.0, 6850.0])
def test_speed_matches_numerical_dgamma_dspot(strike: float) -> None:
    h = 0.5
    up = _first_order(spot=SPOT + h, strike=strike).gamma
    dn = _first_order(spot=SPOT - h, strike=strike).gamma
    numeric = (up - dn) / (2 * h)
    analytic = _higher(strike=strike).speed
    assert analytic == pytest.approx(numeric, rel=5e-3, abs=1e-9)


# ------------------------------------------------------------ 0DTE behaviour


def test_an_out_of_the_money_call_loses_delta_as_the_clock_runs() -> None:
    """Charm negative for OTM calls is the whole 0DTE problem in one number."""
    h = _higher(strike=6850.0, is_call=True)
    assert h.charm_per_minute < 0.0


def test_charm_accelerates_violently_near_the_money() -> None:
    """A live contract's delta drains slowly at four hours and violently at ten minutes.

    Measured on a 10-point OTM call: -2.0e-4 per minute at 240 minutes against
    -1.2e-2 at 10 minutes, a sixtyfold acceleration. This is the number that says
    being right about direction too late is worthless.
    """
    far = abs(_higher(strike=6810.0, minutes=240.0).charm_per_minute)
    near = abs(_higher(strike=6810.0, minutes=10.0).charm_per_minute)
    assert near > far * 20.0, f"charm should accelerate: {far:.2e} -> {near:.2e}"


def test_a_far_otm_contract_is_already_dead_so_its_charm_collapses() -> None:
    """The counterpart, and the reason cheap strikes are a trap.

    A 50-point OTM call has no delta left to lose by 20 minutes, so its charm
    falls back toward zero -- not because it is safe, but because it has already
    finished decaying. Measured: peak -7.5e-4 near 120 minutes, -1.6e-5 at 20.
    A model reading only the level would see a small charm and call it stable.
    """
    peak = abs(_higher(strike=6850.0, minutes=120.0).charm_per_minute)
    late = abs(_higher(strike=6850.0, minutes=20.0).charm_per_minute)
    assert late < peak * 0.1
    assert abs(_first_order(strike=6850.0, minutes=20.0).delta) < 0.01


def test_calls_and_puts_share_charm_because_delta_differs_by_a_constant() -> None:
    call = _higher(is_call=True).charm_per_minute
    put = _higher(is_call=False).charm_per_minute
    assert call == pytest.approx(put, rel=1e-9)


def test_the_iv_matches_the_pinned_module() -> None:
    """The whole surface is derived from the pinned solve, not a second one."""
    assert _higher().implied_volatility == pytest.approx(
        _first_order().implied_volatility, rel=1e-12
    )


def test_an_unsolvable_price_returns_nan_rather_than_a_number() -> None:
    bad = higher_order_from_price(0.0, SPOT, STRIKE, MINUTES, True)
    assert bad == NAN_HIGHER_ORDER
    assert all(math.isnan(v) for v in bad.as_dict().values())


def test_as_dict_prefixes_for_feature_frames() -> None:
    d = _higher().as_dict(prefix="ho_")
    assert "ho_charm_per_minute" in d and "ho_vanna" in d
