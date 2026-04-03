"""Dynamic candidate generation from option chain data.

Generates tradable option contracts around the current SPX spot.
Replaces v1's fixed 6-class direction head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STRIKE_GRID = 5.0
MAX_OFFSET = 30  # points from ATM
OFFSETS = list(range(-MAX_OFFSET, MAX_OFFSET + 1, int(STRIKE_GRID)))  # -30,-25,...,+25,+30


@dataclass(frozen=True)
class CandidateContract:
    """A single option contract that could be traded."""
    strike: float
    right: str           # "C" or "P"
    offset: int          # signed offset from ATM in points
    bid: float
    ask: float
    mid: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    iv: float = 0.0

    @property
    def spread_bps(self) -> float:
        if self.mid <= 0:
            return float('inf')
        return (self.ask - self.bid) / self.mid * 10000.0

    @property
    def is_otm(self) -> bool:
        return abs(self.offset) > 0


def atm_strike(spot: float) -> float:
    """Round spot price to nearest strike on 5-point grid."""
    return round(spot / STRIKE_GRID) * STRIKE_GRID


def generate_candidates(
    spot: float,
    chain: dict | None = None,
    expiry: str = "",
    min_bid: float = 0.0,
    max_spread_bps: float = 300.0,
    min_price: float = 0.05,
) -> list[CandidateContract]:
    """Generate tradable option candidates around current spot.

    Args:
        spot: current SPX price
        chain: dict mapping (strike, right) -> {bid, ask, iv, delta, gamma, theta}
               If None, generates synthetic candidates from spot price.
        expiry: expiry date string (YYYYMMDD)
        min_bid: minimum bid to include (filter dead options)
        max_spread_bps: maximum spread in basis points
        min_price: minimum mid price

    Returns:
        list of CandidateContract, filtered for tradability
    """
    atm = atm_strike(spot)
    candidates = []

    for offset in OFFSETS:
        strike = atm + offset
        if strike <= 0:
            continue

        for right in ("C", "P"):
            if chain is not None:
                key = (strike, right)
                if key not in chain:
                    continue
                data = chain[key]
                bid = float(data.get("bid", 0))
                ask = float(data.get("ask", 0))
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
                delta = float(data.get("delta", 0))
                gamma = float(data.get("gamma", 0))
                theta = float(data.get("theta", 0))
                iv = float(data.get("iv", 0))
            else:
                # Synthetic: estimate from intrinsic + time value
                # Used for historical replay when chain data is unavailable
                intrinsic = max(spot - strike, 0) if right == "C" else max(strike - spot, 0)
                time_value = max(0.5, 3.0 - abs(offset) * 0.08)
                mid = intrinsic + time_value
                # Realistic 0DTE spreads by premium tier
                spread_dollar = 0.15 if mid >= 5.0 else 0.10 if mid >= 2.0 else 0.05
                bid = mid - spread_dollar / 2.0
                ask = mid + spread_dollar / 2.0
                delta = gamma = theta = iv = 0.0

            c = CandidateContract(
                strike=strike, right=right, offset=offset,
                bid=bid, ask=ask, mid=mid,
                delta=delta, gamma=gamma, theta=theta, iv=iv,
            )

            # Filter
            if c.bid <= min_bid:
                continue
            if c.mid < min_price:
                continue
            if c.spread_bps > max_spread_bps:
                continue

            candidates.append(c)

    return candidates


def candidates_from_replay_data(
    spot: float,
    option_prices: dict,
    bar_idx: int,
    expiry: str = "",
) -> list[CandidateContract]:
    """Generate candidates from v1-format option_prices dict.

    v1 stores option prices as arrays keyed by action type.
    This converts them to CandidateContract objects.

    Args:
        spot: SPX price at this bar
        option_prices: dict with keys like 'atm_call_prices', 'otm5_call_prices', etc.
        bar_idx: global bar index into the price arrays
        expiry: date string
    """
    atm = atm_strike(spot)
    candidates = []

    # Map v1 keys to (offset, right)
    v1_keys = [
        ('atm_call_prices', 0, 'C'),
        ('otm5_call_prices', 5, 'C'),
        ('otm10_call_prices', 10, 'C'),
        ('otm15_call_prices', 15, 'C'),
        ('otm20_call_prices', 20, 'C'),
        ('otm25_call_prices', 25, 'C'),
        ('otm30_call_prices', 30, 'C'),
        ('atm_put_prices', 0, 'P'),
        ('otm5_put_prices', -5, 'P'),
        ('otm10_put_prices', -10, 'P'),
        ('otm15_put_prices', -15, 'P'),
        ('otm20_put_prices', -20, 'P'),
        ('otm25_put_prices', -25, 'P'),
        ('otm30_put_prices', -30, 'P'),
    ]

    for key, offset, right in v1_keys:
        if key not in option_prices:
            continue
        arr = option_prices[key]
        if bar_idx >= len(arr):
            continue
        px = float(arr[bar_idx])
        if np.isnan(px) or px <= 0:
            continue

        strike = atm + offset if right == "C" else atm + offset
        # Estimate bid/ask from mid
        spread_frac = 0.02 if abs(offset) == 0 else 0.04
        bid = px * (1.0 - spread_frac)
        ask = px * (1.0 + spread_frac)

        candidates.append(CandidateContract(
            strike=strike, right=right, offset=offset,
            bid=bid, ask=ask, mid=px,
        ))

    return candidates
