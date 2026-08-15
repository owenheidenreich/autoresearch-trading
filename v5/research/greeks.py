"""Black-Scholes greeks and implied volatility, computed from price.

The entry model has been reading the underlying's path and the contract's price
and nothing else. The design assumes greeks, so they have to come from somewhere,
and *where* matters more than it first appears.

**They are computed here rather than taken from a vendor, and that is a parity
decision, not a convenience.** The owned quote corpus carries a ``greek_source``
column precisely because greeks are a modelled quantity: two vendors disagree,
and a vendor's value at 09:35:00 is not necessarily the value the live system
would hold at 09:35:00. The train/live divergence register exists for exactly
this class of problem. A greek recomputed from inputs the live system also has —
the option's own price, a parity spot, the strike, and the clock — cannot
diverge, because the same function runs on both sides.

Every input is available on a live OPRA feed at decision time:

* the option price is the thing being quoted;
* the spot comes from put/call parity on the same chain, which the bot already
  computes;
* the strike and expiry are contract identity;
* the interest rate is a declared constant, and at intraday horizons on a 0DTE
  contract its effect is negligible — it is carried so the formula is complete,
  not because it matters.

SPX options are European and cash-settled, so Black-Scholes is the right model
rather than an approximation to an American one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Declared. At a horizon of hours on a 0DTE contract the discount factor moves
# the price by less than a cent; it is here for completeness.
RISK_FREE_RATE = 0.04
# Minutes in a trading day and trading days in a year, used to turn "minutes to
# expiry" into the year fraction Black-Scholes wants.
MINUTES_PER_SESSION = 390.0
SESSIONS_PER_YEAR = 252.0

# Below this many minutes to expiry an implied volatility is not a meaningful
# number: the price is almost all intrinsic and the solve is ill-conditioned.
MIN_MINUTES_TO_EXPIRY = 5.0
IV_BOUNDS = (0.01, 5.0)
IV_TOLERANCE = 1e-6
IV_MAX_ITERATIONS = 100


def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF, via the error function in the standard library."""

    if isinstance(x, np.ndarray):
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def years_to_expiry(minutes: float | np.ndarray) -> float | np.ndarray:
    """Minutes remaining, expressed as the year fraction the formula wants."""

    return np.asarray(minutes, dtype=float) / (MINUTES_PER_SESSION * SESSIONS_PER_YEAR)


def black_scholes_price(
    spot: float, strike: float, years: float, sigma: float, is_call: bool
) -> float:
    """Price of a European option. Intrinsic when there is no time or no vol."""

    if years <= 0.0 or sigma <= 0.0:
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(0.0, intrinsic)
    root = sigma * math.sqrt(years)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * sigma * sigma) * years) / root
    d2 = d1 - root
    discount = math.exp(-RISK_FREE_RATE * years)
    if is_call:
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)


def implied_volatility(
    price: float, spot: float, strike: float, minutes: float, is_call: bool
) -> float:
    """Volatility that reproduces ``price``, by bisection.

    Bisection rather than Newton: it cannot diverge, it needs no derivative, and
    on a 0DTE contract near expiry the vega that Newton would divide by goes to
    zero. Robustness matters more than speed for a value computed once a minute.
    """

    if minutes < MIN_MINUTES_TO_EXPIRY or price <= 0.0 or spot <= 0.0:
        return float("nan")
    years = float(years_to_expiry(minutes))
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    # A price at or below intrinsic carries no time value to solve for; one
    # above the underlying is not arbitrage-free.
    if price <= intrinsic + 1e-9 or price >= spot:
        return float("nan")
    low, high = IV_BOUNDS
    if black_scholes_price(spot, strike, years, high, is_call) < price:
        return float("nan")
    for _ in range(IV_MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        if black_scholes_price(spot, strike, years, mid, is_call) < price:
            low = mid
        else:
            high = mid
        if high - low < IV_TOLERANCE:
            break
    return 0.5 * (low + high)


@dataclass(frozen=True)
class Greeks:
    """Sensitivities, in the units a trader reads them in.

    ``theta`` is per **minute** rather than per day: a 0DTE scalp lives for
    minutes, and a per-day theta on a contract with an hour left is a number
    nobody can act on.
    """

    implied_volatility: float
    delta: float
    gamma: float
    vega: float
    theta_per_minute: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}iv": self.implied_volatility,
            f"{prefix}delta": self.delta,
            f"{prefix}gamma": self.gamma,
            f"{prefix}vega": self.vega,
            f"{prefix}theta_per_minute": self.theta_per_minute,
        }


NAN_GREEKS = Greeks(*(float("nan"),) * 5)


def greeks_from_price(
    price: float, spot: float, strike: float, minutes: float, is_call: bool
) -> Greeks:
    """Everything the entry model needs, from inputs a live feed also has."""

    sigma = implied_volatility(price, spot, strike, minutes, is_call)
    if not math.isfinite(sigma):
        return NAN_GREEKS
    years = float(years_to_expiry(minutes))
    root = sigma * math.sqrt(years)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * sigma * sigma) * years) / root
    d2 = d1 - root
    pdf = float(norm_pdf(d1))
    discount = math.exp(-RISK_FREE_RATE * years)
    delta = float(norm_cdf(d1)) if is_call else float(norm_cdf(d1)) - 1.0
    gamma = pdf / (spot * root)
    vega = spot * pdf * math.sqrt(years)
    theta_per_year = -spot * pdf * sigma / (2.0 * math.sqrt(years))
    if is_call:
        theta_per_year -= RISK_FREE_RATE * strike * discount * float(norm_cdf(d2))
    else:
        theta_per_year += RISK_FREE_RATE * strike * discount * float(norm_cdf(-d2))
    theta_per_minute = theta_per_year / (MINUTES_PER_SESSION * SESSIONS_PER_YEAR)
    return Greeks(sigma, delta, gamma, vega, theta_per_minute)


# --- batch forms -------------------------------------------------------------
# The scalar functions above are the definition and are what a live decision
# would call. Training needs the same numbers for millions of rows, so these
# vectorised forms exist purely for speed and are pinned by test to agree with
# the scalar path exactly. If the two ever disagree, the scalar one is right.


def black_scholes_price_batch(
    spot: np.ndarray, strike: np.ndarray, years: np.ndarray,
    sigma: np.ndarray, is_call: np.ndarray,
) -> np.ndarray:
    spot = np.asarray(spot, float)
    strike = np.asarray(strike, float)
    years = np.asarray(years, float)
    sigma = np.asarray(sigma, float)
    is_call = np.asarray(is_call, bool)
    out = np.where(
        is_call, np.maximum(0.0, spot - strike), np.maximum(0.0, strike - spot)
    )
    live = (years > 0.0) & (sigma > 0.0)
    if not live.any():
        return out
    root = np.where(live, sigma * np.sqrt(np.maximum(years, 1e-18)), 1.0)
    d1 = np.where(
        live,
        (np.log(np.maximum(spot, 1e-18) / np.maximum(strike, 1e-18))
         + (RISK_FREE_RATE + 0.5 * sigma * sigma) * years) / root,
        0.0,
    )
    d2 = d1 - root
    discount = np.exp(-RISK_FREE_RATE * years)
    call = spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    put = strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)
    return np.where(live, np.where(is_call, call, put), out)


def implied_volatility_batch(
    price: np.ndarray, spot: np.ndarray, strike: np.ndarray,
    minutes: np.ndarray, is_call: np.ndarray,
) -> np.ndarray:
    """Bisection on arrays, same bounds and iteration count as the scalar form."""

    price = np.asarray(price, float)
    spot = np.asarray(spot, float)
    strike = np.asarray(strike, float)
    minutes = np.asarray(minutes, float)
    is_call = np.asarray(is_call, bool)
    years = np.asarray(years_to_expiry(minutes), float)
    intrinsic = np.where(
        is_call, np.maximum(0.0, spot - strike), np.maximum(0.0, strike - spot)
    )
    ok = (
        (minutes >= MIN_MINUTES_TO_EXPIRY)
        & (price > 0.0)
        & (spot > 0.0)
        & (price > intrinsic + 1e-9)
        & (price < spot)
    )
    low = np.full(price.shape, IV_BOUNDS[0])
    high = np.full(price.shape, IV_BOUNDS[1])
    ok &= black_scholes_price_batch(spot, strike, years, high, is_call) >= price
    for _ in range(IV_MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        below = black_scholes_price_batch(spot, strike, years, mid, is_call) < price
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
        if np.all(high - low < IV_TOLERANCE):
            break
    return np.where(ok, 0.5 * (low + high), np.nan)


def greeks_batch(
    price: np.ndarray, spot: np.ndarray, strike: np.ndarray,
    minutes: np.ndarray, is_call: np.ndarray,
) -> dict[str, np.ndarray]:
    """The same five quantities as :func:`greeks_from_price`, for many rows."""

    spot = np.asarray(spot, float)
    strike = np.asarray(strike, float)
    minutes = np.asarray(minutes, float)
    is_call = np.asarray(is_call, bool)
    sigma = implied_volatility_batch(price, spot, strike, minutes, is_call)
    years = np.asarray(years_to_expiry(minutes), float)
    good = np.isfinite(sigma)
    safe_sigma = np.where(good, sigma, 0.2)
    safe_years = np.where(good & (years > 0), years, 1e-6)
    root = safe_sigma * np.sqrt(safe_years)
    d1 = (np.log(np.maximum(spot, 1e-18) / np.maximum(strike, 1e-18))
          + (RISK_FREE_RATE + 0.5 * safe_sigma**2) * safe_years) / root
    d2 = d1 - root
    pdf = np.asarray(norm_pdf(d1), float)
    discount = np.exp(-RISK_FREE_RATE * safe_years)
    delta = np.where(is_call, norm_cdf(d1), np.asarray(norm_cdf(d1), float) - 1.0)
    gamma = pdf / (spot * root)
    vega = spot * pdf * np.sqrt(safe_years)
    theta = -spot * pdf * safe_sigma / (2.0 * np.sqrt(safe_years))
    theta = np.where(
        is_call,
        theta - RISK_FREE_RATE * strike * discount * np.asarray(norm_cdf(d2), float),
        theta + RISK_FREE_RATE * strike * discount * np.asarray(norm_cdf(-d2), float),
    )
    per_minute = theta / (MINUTES_PER_SESSION * SESSIONS_PER_YEAR)
    blank = np.full(sigma.shape, np.nan)
    return {
        "iv": sigma,
        "delta": np.where(good, delta, blank),
        "gamma": np.where(good, gamma, blank),
        "vega": np.where(good, vega, blank),
        "theta_per_minute": np.where(good, per_minute, blank),
    }
