"""What SPX move a 0DTE contract needs before it pays for itself.

The question this answers
-------------------------
Every failure in this project's ledger shares a shape: a model selected a
contract without knowing what that contract *required* in order to make money.
`RIGHT IDEA, WRONG UNITS` is the clearest case — a model optimised percentage
excursion, succeeded, and lost more than random, because a 74% move on a $200
contract is $148 while a 39% move on a $1,000 one is $390. The measured
consequence was an average traded ticket of **$579** biased toward the cheapest,
most leveraged strikes.

So before any selection rule exists, this module answers the prior question:
**given a strike, a clock and a volatility, how far must SPX travel, and how
fast, merely to break even?** A contract whose required move almost never happens
does not belong in a bot's action space at all, however cleverly it is chosen.

How it is computed, and why not with greeks
--------------------------------------------
By **repricing with the pinned Black-Scholes pricer**, not by a delta/gamma/theta
Taylor expansion. On a 0DTE contract gamma is large and the expansion is wrong by
enough to matter over the moves being asked about. The greeks explain *why* the
answer is what it is; the pricer decides *what* it is.

The friction is the repository's measured friction, not an assumption:

* **fees `$3.08` per round trip**, from `build_quoted_dataset.FEES_PER_ROUND_TRIP_USD`
* **the spread is crossed once** — bought at the ask, sold at the bid — so a
  contract quoted `0.20` wide costs `$20` on a 100-multiplier before it has moved
  at all. Measured medians on the corpus ladder are `0.10-0.20`, roughly
  **1.6%-2.1% of mid**.

Parity, inherited unchanged
---------------------------
Everything here derives from inputs a live feed holds at decision time and runs
through [`greeks.py`](greeks.py), which is hash-pinned. Nothing is read from a
vendor greek field, so the same function runs on both sides of the train/live
boundary.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD
from v5.research.greeks import black_scholes_price, years_to_expiry

CONTRACT_MULTIPLIER = 100.0

#: No favourable move is worth searching past this; SPX does not travel 300
#: points in a session often enough for the answer to mean anything.
MAX_SEARCH_POINTS = 300.0
SOLVE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ContractEconomics:
    """What one contract costs, bleeds, and requires."""

    entry_ask_usd: float
    #: What the position is worth if SPX does not move at all over the hold.
    decay_only_pnl_usd: float
    #: Smallest favourable SPX move, in index points, that breaks even.
    required_move_points: float
    #: The same as a fraction of spot, which is the comparable number across eras.
    required_move_pct: float
    #: Round-trip friction in dollars: the spread crossed once, plus fees.
    friction_usd: float

    @property
    def reachable(self) -> bool:
        return math.isfinite(self.required_move_points)

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def _round_trip_pnl(
    move: float, *, spot: float, strike: float, minutes_to_expiry: float,
    hold_minutes: float, sigma: float, spread: float, is_call: bool, fees: float,
) -> float:
    """Dollars for one contract: bought at the ask now, sold at the bid later."""

    entry_mid = black_scholes_price(
        spot, strike, float(years_to_expiry(minutes_to_expiry)), sigma, is_call
    )
    entry_ask = entry_mid + 0.5 * spread
    signed = move if is_call else -move
    exit_mid = black_scholes_price(
        spot + signed, strike, float(years_to_expiry(minutes_to_expiry - hold_minutes)),
        sigma, is_call,
    )
    exit_bid = max(0.0, exit_mid - 0.5 * spread)
    return (exit_bid - entry_ask) * CONTRACT_MULTIPLIER - fees


def contract_economics(
    *,
    spot: float,
    strike: float,
    minutes_to_expiry: float,
    hold_minutes: float,
    sigma: float,
    spread: float,
    is_call: bool,
    fees: float = FEES_PER_ROUND_TRIP_USD,
) -> ContractEconomics:
    """Break-even economics for one contract over one holding period.

    `required_move_points` is **infinite when no move within `MAX_SEARCH_POINTS`
    breaks even** — which is a real and common answer for a cheap far-out strike
    with little time left, and is exactly the case a selection rule must never be
    allowed to choose.
    """

    if minutes_to_expiry <= hold_minutes or sigma <= 0.0 or spot <= 0.0:
        return ContractEconomics(
            float("nan"), float("nan"), float("inf"), float("inf"),
            spread * CONTRACT_MULTIPLIER + fees,
        )

    def pnl(move: float) -> float:
        return _round_trip_pnl(
            move, spot=spot, strike=strike, minutes_to_expiry=minutes_to_expiry,
            hold_minutes=hold_minutes, sigma=sigma, spread=spread,
            is_call=is_call, fees=fees,
        )

    entry_mid = black_scholes_price(
        spot, strike, float(years_to_expiry(minutes_to_expiry)), sigma, is_call
    )
    entry_ask = entry_mid + 0.5 * spread
    friction = spread * CONTRACT_MULTIPLIER + fees
    decay_only = pnl(0.0)

    if decay_only >= 0.0:
        required = 0.0
    elif pnl(MAX_SEARCH_POINTS) < 0.0:
        required = float("inf")
    else:
        low, high = 0.0, MAX_SEARCH_POINTS
        while high - low > SOLVE_TOLERANCE:
            mid = 0.5 * (low + high)
            if pnl(mid) < 0.0:
                low = mid
            else:
                high = mid
        required = high

    return ContractEconomics(
        entry_ask_usd=entry_ask * CONTRACT_MULTIPLIER,
        decay_only_pnl_usd=decay_only,
        required_move_points=required,
        required_move_pct=required / spot if math.isfinite(required) else float("inf"),
        friction_usd=friction,
    )
