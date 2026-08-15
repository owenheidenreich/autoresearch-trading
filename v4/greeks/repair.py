"""Repair missing Databento/live SPXW Greeks from executable quote fields.

Databento OPRA does not provide IV/Greeks, and CBBO snapshots can produce
internally inconsistent mid prices when the underlying moves quickly inside the
minute. For long-entry research the normal source is the bid/ask mid. If that
cannot produce a valid implied volatility, the next best repair is the
executable entry ask. If no observed quote price can be inverted without
violating no-arbitrage bounds, the caller should treat the candidate as
Greek-unrepairable and skip it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from v4.greeks.black_scholes import greeks, implied_vol


@dataclass(frozen=True)
class GreekEstimate:
    iv: float
    delta: float
    gamma: float
    theta_per_day: float
    vega: float
    source: str
    price_used: float


def _finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _candidate_prices(
    *,
    mid: object = None,
    ask: object = None,
    bid: object = None,
) -> Iterable[tuple[str, float]]:
    seen: set[float] = set()
    for source, value in (("mid", mid), ("ask_repair", ask), ("bid_repair", bid)):
        price = _finite(value)
        if price is None or price <= 0.0:
            continue
        key = round(price, 8)
        if key in seen:
            continue
        seen.add(key)
        yield source, price


def compute_repaired_greeks(
    *,
    S: object,
    K: object,
    T: object,
    is_call: bool,
    mid: object = None,
    ask: object = None,
    bid: object = None,
    r: float = 0.05,
    q: float = 0.0,
) -> GreekEstimate | None:
    """Return finite IV/Greeks using mid, ask repair, then bid repair.

    The function does not fabricate an IV from a synthetic no-arbitrage floor.
    If all observable prices are impossible to invert, returning ``None`` is
    safer than pretending a broken/stale quote carries reliable contract
    economics.
    """

    s = _finite(S)
    k = _finite(K)
    t = _finite(T)
    if s is None or k is None or t is None or s <= 0.0 or k <= 0.0 or t <= 0.0:
        return None

    for source, market_price in _candidate_prices(mid=mid, ask=ask, bid=bid):
        for sigma_hi in (5.0, 10.0, 20.0):
            try:
                iv = implied_vol(
                    market_price=market_price,
                    S=s,
                    K=k,
                    T=t,
                    r=r,
                    q=q,
                    is_call=is_call,
                    sigma_hi=sigma_hi,
                )
                g = greeks(S=s, K=k, T=t, sigma=iv, r=r, q=q, is_call=is_call)
            except ValueError:
                continue
            values = (iv, g.delta, g.gamma, g.theta, g.vega)
            if all(np.isfinite(float(value)) for value in values):
                return GreekEstimate(
                    iv=float(iv),
                    delta=float(g.delta),
                    gamma=float(g.gamma),
                    theta_per_day=float(g.theta / 365.0),
                    vega=float(g.vega),
                    source=f"black_scholes_{source}",
                    price_used=float(market_price),
                )
    return None
