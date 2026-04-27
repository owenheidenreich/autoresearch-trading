"""Black-Scholes pricing and Greeks for European cash-settled index options.

Used by:
- The greeks module to compute IV/Greeks for vendors that don't supply them
  (Databento OPRA history is the primary case; per their docs, "we don't
  currently provide pre-calculated implied volatility (IV) or greeks").
- The reconciliation harness in `reconcile.py` to verify our BS implementation
  matches OptionsDX vendor-supplied Greeks within tolerance — a Phase-0 gate.

Conventions:
- T (time to expiry) is annualized (calendar year fraction).
- IV is annualized decimal (e.g., 0.20 for 20%).
- Greeks are returned in the "raw" form (delta dimensionless, gamma per $1
  underlying change, vega per 1.0 IV change, theta per year, rho per 1.0
  rate change). Helper functions convert to common vendor conventions
  (e.g., OptionsDX vega is per 1% IV change → divide by 100).
- SPX is cash-settled European; default dividend yield q=0. Adjust if the
  caller wants to model implied carry from futures basis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class Greeks:
    """All Greeks in raw (per-unit) form. See module docstring for conventions."""

    price: float
    delta: float
    gamma: float
    vega: float        # per 1.0 vol change
    theta: float       # per year
    rho: float         # per 1.0 rate change
    charm: float       # d delta / d t, per year
    vanna: float       # d delta / d vol
    vomma: float       # d vega / d vol


def _d1_d2(
    *, S: float, K: float, T: float, sigma: float, r: float, q: float
) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError(
            f"Black-Scholes inputs out of range: S={S}, K={K}, T={T}, sigma={sigma}"
        )
    sigma_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def price(
    *,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.05,
    q: float = 0.0,
    is_call: bool,
) -> float:
    """European option price under Black-Scholes."""
    if T <= 0:
        # At-or-past expiry: intrinsic value
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S=S, K=K, T=T, sigma=sigma, r=r, q=q)
    if is_call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def greeks(
    *,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.05,
    q: float = 0.0,
    is_call: bool,
) -> Greeks:
    """All Greeks for a European option under Black-Scholes."""
    d1, d2 = _d1_d2(S=S, K=K, T=T, sigma=sigma, r=r, q=q)
    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)
    e_qT = math.exp(-q * T)
    e_rT = math.exp(-r * T)

    p = price(S=S, K=K, T=T, sigma=sigma, r=r, q=q, is_call=is_call)

    if is_call:
        delta = e_qT * norm.cdf(d1)
        rho = K * T * e_rT * norm.cdf(d2)
        # theta has separate parts; standard call formula:
        theta = (
            -S * e_qT * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * e_rT * norm.cdf(d2)
            + q * S * e_qT * norm.cdf(d1)
        )
        charm = (
            -e_qT * pdf_d1 * (2 * (r - q) * T - d2 * sigma_sqrt_T) / (2 * T * sigma_sqrt_T)
            + q * e_qT * norm.cdf(d1)
        )
    else:
        delta = -e_qT * norm.cdf(-d1)
        rho = -K * T * e_rT * norm.cdf(-d2)
        theta = (
            -S * e_qT * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * e_rT * norm.cdf(-d2)
            - q * S * e_qT * norm.cdf(-d1)
        )
        charm = (
            -e_qT * pdf_d1 * (2 * (r - q) * T - d2 * sigma_sqrt_T) / (2 * T * sigma_sqrt_T)
            - q * e_qT * norm.cdf(-d1)
        )

    gamma = e_qT * pdf_d1 / (S * sigma_sqrt_T)
    vega = S * e_qT * pdf_d1 * sqrt_T
    vanna = -e_qT * pdf_d1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma

    return Greeks(
        price=p,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        charm=charm,
        vanna=vanna,
        vomma=vomma,
    )


def implied_vol(
    *,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    q: float = 0.0,
    is_call: bool,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
    tol: float = 1e-7,
) -> float:
    """Solve for IV using Brent's method.

    Raises ValueError if the market price is outside the no-arbitrage bounds.
    For 0DTE near-ATM options the price function is monotonic and the solver
    converges in a few iterations.
    """
    if T <= 0:
        raise ValueError("T must be > 0 to invert IV; at expiry IV is undefined")

    # No-arb bounds: option price is bounded by the underlying (call) or strike (put)
    if market_price < 0:
        raise ValueError(f"market_price must be >= 0; got {market_price}")
    if is_call:
        upper = S * math.exp(-q * T)
        if market_price > upper + tol:
            raise ValueError(f"call price {market_price} exceeds no-arb upper {upper}")
    else:
        upper = K * math.exp(-r * T)
        if market_price > upper + tol:
            raise ValueError(f"put price {market_price} exceeds no-arb upper {upper}")

    def diff(sigma: float) -> float:
        return price(S=S, K=K, T=T, sigma=sigma, r=r, q=q, is_call=is_call) - market_price

    f_lo = diff(sigma_lo)
    f_hi = diff(sigma_hi)
    if f_lo * f_hi > 0:
        # Both same sign — solver can't bracket. Likely market price < intrinsic.
        raise ValueError(
            f"cannot bracket IV: diff(sigma={sigma_lo})={f_lo}, "
            f"diff(sigma={sigma_hi})={f_hi}"
        )

    return brentq(diff, sigma_lo, sigma_hi, xtol=tol, maxiter=100)


# ---------- vendor convention adapters ----------

def to_optionsdx_conventions(g: Greeks) -> dict[str, float]:
    """Convert raw BS Greeks to OptionsDX-published conventions.

    OptionsDX scales: vega per 1% IV change (÷100), theta per 1 calendar day
    (÷365), rho per 1% rate change (÷100). Delta and gamma are unchanged.
    Charm/vanna/vomma are not published by OptionsDX so we leave raw form.
    """
    return {
        "delta": g.delta,
        "gamma": g.gamma,
        "vega_per_1pct": g.vega / 100.0,
        "theta_per_day": g.theta / 365.0,
        "rho_per_1pct": g.rho / 100.0,
        "charm": g.charm,
        "vanna": g.vanna,
        "vomma": g.vomma,
    }
