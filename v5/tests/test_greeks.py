"""Black-Scholes greeks recomputed from price.

Two things are being defended here. The arithmetic — against known closed-form
values and against put/call parity, which no implementation error survives. And
the *parity* property that motivated computing them at all: the same function
must run on training rows and live rows, so it may depend on nothing but price,
spot, strike, clock and side.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from v5.research import greeks as g


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------


def test_the_normal_cdf_matches_known_values() -> None:
    assert g.norm_cdf(0.0) == pytest.approx(0.5)
    assert g.norm_cdf(1.96) == pytest.approx(0.9750021, abs=1e-6)
    assert g.norm_cdf(-1.96) == pytest.approx(0.0249979, abs=1e-6)


def test_the_normal_cdf_works_on_arrays() -> None:
    got = g.norm_cdf(np.array([-1.0, 0.0, 1.0]))
    assert got == pytest.approx([0.1586553, 0.5, 0.8413447], abs=1e-6)


def test_minutes_convert_to_the_year_fraction_the_formula_wants() -> None:
    # One full session is one trading day out of 252.
    assert g.years_to_expiry(390.0) == pytest.approx(1.0 / 252.0)


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------


def test_an_at_the_money_call_is_worth_roughly_the_textbook_value() -> None:
    # S=K=100, one year, 20% vol: the standard sanity case, about 8.9.
    got = g.black_scholes_price(100.0, 100.0, 1.0, 0.20, is_call=True)
    assert got == pytest.approx(9.925, abs=0.01)


def test_put_call_parity_holds() -> None:
    """C - P = S - K*exp(-rT). No pricing error survives this."""

    spot, strike, years, sigma = 5_900.0, 5_920.0, 0.01, 0.35
    call = g.black_scholes_price(spot, strike, years, sigma, True)
    put = g.black_scholes_price(spot, strike, years, sigma, False)
    expected = spot - strike * math.exp(-g.RISK_FREE_RATE * years)
    assert call - put == pytest.approx(expected, abs=1e-8)


def test_with_no_time_left_a_contract_is_worth_its_intrinsic() -> None:
    assert g.black_scholes_price(5_910.0, 5_900.0, 0.0, 0.3, True) == pytest.approx(10.0)
    assert g.black_scholes_price(5_890.0, 5_900.0, 0.0, 0.3, True) == pytest.approx(0.0)
    assert g.black_scholes_price(5_890.0, 5_900.0, 0.0, 0.3, False) == pytest.approx(10.0)


def test_a_more_volatile_contract_costs_more() -> None:
    cheap = g.black_scholes_price(5_900.0, 5_900.0, 0.01, 0.15, True)
    dear = g.black_scholes_price(5_900.0, 5_900.0, 0.01, 0.45, True)
    assert dear > cheap


# --------------------------------------------------------------------------
# implied volatility
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.08, 0.20, 0.55, 1.20])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_volatility_recovers_the_volatility_that_made_the_price(
    sigma: float, is_call: bool
) -> None:
    """The round trip that proves the solver and the pricer agree."""

    spot, strike, minutes = 5_900.0, 5_905.0, 120.0
    price = g.black_scholes_price(
        spot, strike, float(g.years_to_expiry(minutes)), sigma, is_call
    )
    assert g.implied_volatility(price, spot, strike, minutes, is_call) == pytest.approx(
        sigma, abs=1e-4
    )


def test_too_little_time_left_returns_no_volatility_rather_than_a_wrong_one() -> None:
    assert math.isnan(g.implied_volatility(5.0, 5_900.0, 5_900.0, 2.0, True))


def test_a_price_at_or_below_intrinsic_has_no_volatility_to_solve_for() -> None:
    # A call 20 points in the money quoted at 20 carries no time value.
    assert math.isnan(g.implied_volatility(20.0, 5_920.0, 5_900.0, 60.0, True))


def test_a_price_above_the_underlying_is_refused() -> None:
    assert math.isnan(g.implied_volatility(6_000.0, 5_900.0, 5_900.0, 60.0, True))


# --------------------------------------------------------------------------
# the greeks themselves
# --------------------------------------------------------------------------


def _greeks(sigma: float = 0.30, minutes: float = 120.0, strike: float = 5_900.0,
            is_call: bool = True, spot: float = 5_900.0) -> g.Greeks:
    price = g.black_scholes_price(
        spot, strike, float(g.years_to_expiry(minutes)), sigma, is_call
    )
    return g.greeks_from_price(price, spot, strike, minutes, is_call)


def test_an_at_the_money_call_has_delta_near_a_half() -> None:
    assert _greeks().delta == pytest.approx(0.5, abs=0.03)


def test_an_at_the_money_put_has_delta_near_minus_a_half() -> None:
    assert _greeks(is_call=False).delta == pytest.approx(-0.5, abs=0.03)


def test_a_deep_in_the_money_call_approaches_delta_one() -> None:
    assert _greeks(strike=5_600.0).delta > 0.9


def test_gamma_is_positive_and_largest_at_the_money() -> None:
    atm = _greeks(strike=5_900.0).gamma
    away = _greeks(strike=6_050.0).gamma
    assert atm > away > 0


def test_gamma_rises_as_expiry_approaches() -> None:
    """The property that makes 0DTE what it is."""

    assert _greeks(minutes=30.0).gamma > _greeks(minutes=300.0).gamma


def test_theta_is_negative_for_a_long_option_and_quoted_per_minute() -> None:
    got = _greeks()
    assert got.theta_per_minute < 0
    # A whole session of decay must be a sane fraction of an ATM premium.
    price = g.black_scholes_price(5_900.0, 5_900.0, float(g.years_to_expiry(120.0)), 0.30, True)
    assert abs(got.theta_per_minute) * 120.0 < price


def test_theta_bites_harder_close_to_expiry() -> None:
    assert _greeks(minutes=30.0).theta_per_minute < _greeks(minutes=300.0).theta_per_minute


def test_vega_is_positive_and_falls_towards_expiry() -> None:
    near, far = _greeks(minutes=30.0), _greeks(minutes=300.0)
    assert 0 < near.vega < far.vega


def test_an_unsolvable_contract_returns_nan_greeks_rather_than_zeros() -> None:
    got = g.greeks_from_price(0.0, 5_900.0, 5_900.0, 60.0, True)
    assert math.isnan(got.delta) and math.isnan(got.gamma)


def test_the_greeks_export_with_a_prefix_for_a_feature_table() -> None:
    keys = _greeks().as_dict("entry_").keys()
    assert set(keys) == {
        "entry_iv", "entry_delta", "entry_gamma", "entry_vega", "entry_theta_per_minute"
    }


# --------------------------------------------------------------------------
# the parity property that motivated computing these at all
# --------------------------------------------------------------------------


def test_greeks_depend_only_on_inputs_a_live_feed_also_has() -> None:
    """No vendor field, no history, no lookahead — only the five arguments."""

    signature = inspect.signature(g.greeks_from_price)
    assert list(signature.parameters) == ["price", "spot", "strike", "minutes", "is_call"]


def test_the_same_inputs_always_give_the_same_greeks() -> None:
    """Determinism is what makes offline and live agree at 1e-12."""

    a = g.greeks_from_price(45.0, 5_900.0, 5_905.0, 90.0, True)
    b = g.greeks_from_price(45.0, 5_900.0, 5_905.0, 90.0, True)
    assert a == b


def test_the_module_imports_no_vendor_or_market_data_code() -> None:
    source = inspect.getsource(g)
    for forbidden in ("databento", "pandas", "read_parquet", "requests"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# the batch forms exist only for speed and must agree with the definition
# --------------------------------------------------------------------------


def test_batch_implied_volatility_matches_the_scalar_solver() -> None:
    rng = np.random.default_rng(5)
    n = 400
    spot = rng.uniform(5_000, 6_000, n)
    strike = spot + rng.uniform(-40, 40, n)
    minutes = rng.uniform(10, 380, n)
    is_call = rng.random(n) < 0.5
    sigma = rng.uniform(0.05, 1.5, n)
    price = np.array([
        g.black_scholes_price(s, k, float(g.years_to_expiry(m)), v, bool(c))
        for s, k, m, v, c in zip(spot, strike, minutes, sigma, is_call)
    ])
    batch = g.implied_volatility_batch(price, spot, strike, minutes, is_call)
    scalar = np.array([
        g.implied_volatility(p, s, k, m, bool(c))
        for p, s, k, m, c in zip(price, spot, strike, minutes, is_call)
    ])
    both = np.isfinite(batch) & np.isfinite(scalar)
    assert both.sum() > n * 0.8
    assert batch[both] == pytest.approx(scalar[both], abs=1e-6)


def test_batch_greeks_match_the_scalar_greeks() -> None:
    rng = np.random.default_rng(6)
    n = 200
    spot = rng.uniform(5_500, 6_000, n)
    strike = spot + rng.uniform(-25, 25, n)
    minutes = rng.uniform(20, 300, n)
    is_call = rng.random(n) < 0.5
    price = np.array([
        g.black_scholes_price(s, k, float(g.years_to_expiry(m)), 0.35, bool(c))
        for s, k, m, c in zip(spot, strike, minutes, is_call)
    ])
    batch = g.greeks_batch(price, spot, strike, minutes, is_call)
    for i in range(n):
        one = g.greeks_from_price(price[i], spot[i], strike[i], minutes[i], bool(is_call[i]))
        if not math.isfinite(one.delta):
            continue
        assert batch["delta"][i] == pytest.approx(one.delta, abs=1e-6)
        assert batch["gamma"][i] == pytest.approx(one.gamma, abs=1e-9)
        assert batch["theta_per_minute"][i] == pytest.approx(one.theta_per_minute, abs=1e-6)


def test_batch_refuses_the_same_rows_the_scalar_form_refuses() -> None:
    price = np.array([20.0, 5.0, 6_000.0])
    spot = np.array([5_920.0, 5_900.0, 5_900.0])
    strike = np.array([5_900.0, 5_900.0, 5_900.0])
    minutes = np.array([60.0, 2.0, 60.0])
    is_call = np.array([True, True, True])
    assert np.all(np.isnan(g.implied_volatility_batch(price, spot, strike, minutes, is_call)))
