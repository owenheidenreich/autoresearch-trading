"""Second- and third-order greeks, for the shape of a 0DTE contract's decay.

Why this is a separate module rather than an edit
-------------------------------------------------
[`greeks.py`](greeks.py) is **hash-pinned by
`PREACQUISITION_SEMANTIC_FREEZE_V1.json`**, which fixes the semantics that
governed a completed vendor purchase. Editing it would break that proof for a
feature addition, so this module **imports** the pinned one and adds to it. The
same rule that kept the protocol-v2 header un-editable applies here: a pinned
file's hash is a historical record.

Why second and third order matter here specifically
---------------------------------------------------
First-order greeks say what a contract is worth *now*. On a 0DTE option the
question that actually decides a trade is what it will be worth in twenty
minutes, and that is a second-order question.

* **Charm** is how fast delta itself drains away. A 10-point-OTM call at 14:00
  may have a delta of 0.20 and a charm that takes it to 0.05 by 15:00 — so the
  same SPX move pays a quarter as much, and the contract dies while you are
  right about direction.
* **Color** is how fast gamma drains. Gamma is what makes a cheap contract
  explode when it comes into the money; color says how long that possibility
  survives.
* **Vanna** is how delta moves when volatility moves — the reason a position can
  be directionally right and still lose when the vol surface shifts under it.
* **Vomma** is the convexity of vega, and **speed** is the convexity of gamma.

The project's own ledger contains the cost of not having these. A model was
trained to pick the contract making the biggest *percentage* move, succeeded at
exactly that, and lost more money than random — closed as `RIGHT IDEA, WRONG
UNITS`, because "a 74% move on a $200 contract is $148 while a 39% move on a
$1,000 contract is $390". The measured consequence was an average traded ticket
of **$579** biased toward the cheapest, most leveraged strikes. That is what
buying without reading charm and color looks like.

Parity discipline, inherited unchanged
--------------------------------------
Every quantity here is computed from **the same four inputs the live system
holds at decision time** — the option's own price, a parity spot, the strike and
the clock — via the pinned module's implied volatility. Nothing is taken from a
vendor's greek field. The same function therefore runs on both sides of the
train/live boundary and cannot diverge.

Time derivatives are reported **per minute**, matching the pinned module's
`theta_per_minute`, because a per-day charm on a contract with an hour left is a
number nobody can act on.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from v5.research.greeks import (
    MINUTES_PER_SESSION,
    RISK_FREE_RATE,
    SESSIONS_PER_YEAR,
    implied_volatility,
    norm_pdf,
    years_to_expiry,
)

#: Time derivatives are per-minute; the analytic forms are per-year.
MINUTES_PER_YEAR = MINUTES_PER_SESSION * SESSIONS_PER_YEAR


@dataclass(frozen=True)
class HigherOrderGreeks:
    """Second- and third-order sensitivities, in units a trader can act on.

    ``charm`` and ``color`` are **per minute of elapsed time** and carry the
    sign of the change a holder experiences as the clock runs forward, so a
    decaying delta gives a negative charm.
    """

    implied_volatility: float
    vanna: float            # d(delta)/d(sigma)
    charm_per_minute: float  # d(delta)/d(time elapsed)
    vomma: float            # d(vega)/d(sigma)
    speed: float            # d(gamma)/d(spot)
    color_per_minute: float  # d(gamma)/d(time elapsed)

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


NAN_HIGHER_ORDER = HigherOrderGreeks(*(float("nan"),) * 6)


def higher_order_from_price(
    price: float, spot: float, strike: float, minutes: float, is_call: bool
) -> HigherOrderGreeks:
    """Second- and third-order greeks from the four live-available inputs.

    Mirrors `greeks.greeks_from_price`: it solves implied volatility from the
    quoted price first, so the whole surface is derived from the same number the
    live feed is quoting.
    """

    sigma = implied_volatility(price, spot, strike, minutes, is_call)
    if not math.isfinite(sigma) or spot <= 0.0 or strike <= 0.0:
        return NAN_HIGHER_ORDER
    years = float(years_to_expiry(minutes))
    if years <= 0.0:
        return NAN_HIGHER_ORDER

    root = sigma * math.sqrt(years)
    log_moneyness = math.log(spot / strike)
    d1 = (log_moneyness + (RISK_FREE_RATE + 0.5 * sigma * sigma) * years) / root
    d2 = d1 - root
    pdf = float(norm_pdf(d1))

    gamma = pdf / (spot * root)
    vega = spot * pdf * math.sqrt(years)

    # d(d1)/d(tau), with tau the time REMAINING. Derived rather than quoted:
    #   d1 = A/(sigma*sqrt(tau)) + (r + sigma^2/2)*sqrt(tau)/sigma
    # so the derivative separates cleanly into a decaying and a growing term.
    dd1_dtau = (
        -0.5 * log_moneyness / (sigma * years ** 1.5)
        + 0.5 * (RISK_FREE_RATE + 0.5 * sigma * sigma) / (sigma * math.sqrt(years))
    )

    # Delta = N(d1) for a call and N(d1) - 1 for a put, so both share d(delta)/d(tau).
    ddelta_dtau = pdf * dd1_dtau
    # Gamma is log-differentiated: d(ln gamma)/d(tau) = -d1 * dd1_dtau - 1/(2 tau).
    dgamma_dtau = gamma * (-d1 * dd1_dtau - 0.5 / years)

    # Elapsed time runs opposite to time remaining, and per minute rather than year.
    charm_per_minute = -ddelta_dtau / MINUTES_PER_YEAR
    color_per_minute = -dgamma_dtau / MINUTES_PER_YEAR

    vanna = -pdf * d2 / sigma
    vomma = vega * d1 * d2 / sigma
    speed = -gamma / spot * (d1 / root + 1.0)

    return HigherOrderGreeks(
        implied_volatility=sigma,
        vanna=vanna,
        charm_per_minute=charm_per_minute,
        vomma=vomma,
        speed=speed,
        color_per_minute=color_per_minute,
    )
