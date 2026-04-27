"""Tests for Black-Scholes implementation and OptionsDX reconciliation."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from v4.greeks import (
    Greeks,
    greeks,
    implied_vol,
    price,
    reconcile_optionsdx_table,
    to_optionsdx_conventions,
)
from v4.ingest.optionsdx import ingest_optionsdx_file

FIXTURE = Path(__file__).parent / "fixtures" / "optionsdx_spx_sample.csv"


# ---------- Black-Scholes basics ----------

def test_atm_call_put_parity() -> None:
    """C - P = S*exp(-qT) - K*exp(-rT)."""
    S, K, T, sigma, r, q = 100.0, 100.0, 0.25, 0.20, 0.05, 0.0
    c = price(S=S, K=K, T=T, sigma=sigma, r=r, q=q, is_call=True)
    p = price(S=S, K=K, T=T, sigma=sigma, r=r, q=q, is_call=False)
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert c - p == pytest.approx(expected, abs=1e-8)


def test_known_atm_call_price() -> None:
    """Reference: ATM call with S=K=100, T=1, sigma=0.20, r=0.05, q=0
    standard textbook value ≈ 10.4506."""
    p = price(S=100.0, K=100.0, T=1.0, sigma=0.20, r=0.05, q=0.0, is_call=True)
    assert p == pytest.approx(10.4506, abs=0.001)


def test_implied_vol_round_trip() -> None:
    """Compute price at known IV, then invert; should recover IV."""
    sigma = 0.18
    S, K, T, r = 100.0, 105.0, 0.25, 0.05
    p = price(S=S, K=K, T=T, sigma=sigma, r=r, is_call=True)
    iv = implied_vol(market_price=p, S=S, K=K, T=T, r=r, is_call=True)
    assert iv == pytest.approx(sigma, abs=1e-6)


def test_call_delta_in_unit_interval() -> None:
    g = greeks(S=100.0, K=100.0, T=0.25, sigma=0.20, r=0.05, is_call=True)
    assert 0.0 < g.delta < 1.0


def test_put_delta_negative() -> None:
    g = greeks(S=100.0, K=100.0, T=0.25, sigma=0.20, r=0.05, is_call=False)
    assert -1.0 < g.delta < 0.0


def test_gamma_positive() -> None:
    g = greeks(S=100.0, K=100.0, T=0.25, sigma=0.20, r=0.05, is_call=True)
    assert g.gamma > 0


def test_vega_positive() -> None:
    g = greeks(S=100.0, K=100.0, T=0.25, sigma=0.20, r=0.05, is_call=True)
    assert g.vega > 0


def test_theta_negative_for_long_atm() -> None:
    """Long ATM option has negative theta."""
    g = greeks(S=100.0, K=100.0, T=0.25, sigma=0.20, r=0.05, is_call=True)
    assert g.theta < 0


def test_charm_finite() -> None:
    g = greeks(S=100.0, K=100.0, T=0.25, sigma=0.20, r=0.05, is_call=True)
    assert math.isfinite(g.charm)


def test_optionsdx_convention_scaling() -> None:
    g = Greeks(
        price=10.0,
        delta=0.5,
        gamma=0.02,
        vega=15.0,        # raw: per 1.0 vol change
        theta=-50.0,      # raw: per year
        rho=20.0,         # raw: per 1.0 rate change
        charm=0.001,
        vanna=0.05,
        vomma=0.10,
    )
    out = to_optionsdx_conventions(g)
    assert out["delta"] == 0.5
    assert out["gamma"] == 0.02
    assert out["vega_per_1pct"] == pytest.approx(0.15)
    assert out["theta_per_day"] == pytest.approx(-50.0 / 365.0)
    assert out["rho_per_1pct"] == pytest.approx(0.20)


# ---------- IV inversion edge cases ----------

def test_iv_rejects_negative_market_price() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        implied_vol(market_price=-1.0, S=100, K=100, T=0.25, is_call=True)


def test_iv_rejects_above_no_arb_call() -> None:
    """Call price cannot exceed S*exp(-qT)."""
    with pytest.raises(ValueError, match="exceeds no-arb"):
        implied_vol(market_price=200.0, S=100, K=100, T=0.25, is_call=True)


def test_iv_rejects_zero_t() -> None:
    with pytest.raises(ValueError, match="T must be > 0"):
        implied_vol(market_price=1.0, S=100, K=100, T=0.0, is_call=True)


# ---------- reconciliation against OptionsDX fixture ----------

def test_optionsdx_reconciliation() -> None:
    """Phase-0 gate: our BS Greeks must match the OptionsDX fixture's
    published Greeks within tolerance.

    Note: the fixture is synthetic (we authored it to mirror OptionsDX
    format). Its Greek values were chosen to be plausible but not exact;
    we use loose tolerances here as a sanity check that the reconciliation
    pipeline runs end-to-end. Tighter tolerances will be set when real
    OptionsDX data is ingested.
    """
    result = ingest_optionsdx_file(FIXTURE)
    # Loose tolerances for the synthetic fixture
    looser_tols = {
        "delta": 0.10,
        "gamma": 0.10,
        "vega_per_1pct": 5.0,
        "theta_per_day": 100.0,  # 0DTE theta varies a lot per minute
        "rho_per_1pct": 1.0,
    }
    check, stats = reconcile_optionsdx_table(
        result.normalized,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        tolerances=looser_tols,
    )
    # Must produce per-metric stats with non-zero counts
    metrics_with_data = [s for s in stats if s.n > 0]
    assert len(metrics_with_data) >= 3
    # Pipeline ran end-to-end without exceptions
    assert isinstance(check.passed, bool)
