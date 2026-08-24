"""The join: what a contract requires, against how often the tape delivers it.

Why this module exists
----------------------
The 2026-08-23 finding `MORNING_NEAR_MONEY_ASYMMETRY` reports a table of `R`
(break-even SPX move), `J` (the adverse move tripping a -40% stop) and a win
rate, but **the code that produced it was never committed** — both join commits
touched only `LOG.md`. The numbers were also revised three times (the 09:35 ATM
ask went $2,278 -> $2,266 -> $1,748 under changing IV assumptions). This module
reconstructs that join reproducibly so the table can be verified or corrected.

What is different here
----------------------
* **Real quotes, not a pricer, for entry.** The ladder stores the actual `ask`,
  `bid` and `spread` per session/minute/strike. Entry cost is measured, not
  modelled, so no spot or spread has to be assumed.
* **Per-session thresholds.** `R` and `J` are computed for each session at its
  own spot, its own IV and its own spread, then the race is evaluated against
  *that session's* `R`. The stored race grid could not do this: its
  `favourable_threshold_points` bottoms out at 2.0, while measured `R` is nearer
  1. Averaging an `R` and then racing against the average is a different and
  wrong question.
* **Exit is repriced** through the pinned pricer, because the exit contract at
  `hold_minutes` later at a shifted spot is not a quote that exists in the tape.

The three states are the finding's: reaching `R` (break even), tripping `J`
(the stop), or neither -- and *neither is a full loss to theta*, not a neutral
outcome.

This measures terrain and required moves. It does **not** produce an expected
value: reaching `R` means breaking even. Overshoot beyond `R` is the next join.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD
from v5.research.greeks import black_scholes_price, years_to_expiry

CONTRACT_MULTIPLIER = 100.0
MAX_SEARCH_POINTS = 300.0
SOLVE_TOLERANCE = 1e-4

#: The stop the charter declares. A contract is dead when it has lost this
#: fraction of the ask paid.
STOP_FRACTION = -0.40

#: Ladder nodes, in points out of the money. 0 is at the money.
OFFSETS_POINTS = (0.0, 10.0, 25.0)
START_TIMES = ("09:35", "11:30", "13:30", "15:00")
HOLD_MINUTES = 20

#: Sessions the upstream race receipt dispositions as defective and excludes.
#: Carried here verbatim so this join measures the same population; a session
#: with an interior whole-book freeze has no usable path.
EXCLUDED_SESSIONS = frozenset({
    "2022-11-25",  # vendor-padded early close; excluded upstream
    "2023-06-26",  # interior whole-book freeze (2 minutes)
    "2023-10-19",  # interior whole-book freeze (3 minutes)
    "2023-10-25",  # interior whole-book freezes (4 and 18 minutes)
})


@dataclass(frozen=True)
class SessionCell:
    """One session, one clock, one strike: what it cost and what it required."""

    session: str
    start_time_et: str
    offset_points: float
    spot: float
    strike: float
    ask_usd: float
    spread: float
    sigma: float
    minutes_to_expiry: float
    required_move_points: float
    stop_move_points: float


def _exit_pnl(
    move: float,
    *,
    spot: float,
    strike: float,
    minutes_to_expiry: float,
    hold_minutes: float,
    sigma: float,
    spread: float,
    entry_ask: float,
    fees: float,
) -> float:
    """Dollars for one contract bought at the real ask, sold at a repriced bid."""

    exit_mid = black_scholes_price(
        spot + move,
        strike,
        float(years_to_expiry(minutes_to_expiry - hold_minutes)),
        sigma,
        True,
    )
    exit_bid = max(0.0, exit_mid - 0.5 * spread)
    return (exit_bid - entry_ask) * CONTRACT_MULTIPLIER - fees


def _solve_move(target: float, pnl, *, low: float, high: float) -> float:
    """Smallest move in [low, high] with pnl(move) >= target, or inf/0.0."""

    if pnl(low) >= target:
        return low
    if pnl(high) < target:
        return float("inf")
    while high - low > SOLVE_TOLERANCE:
        mid = 0.5 * (low + high)
        if pnl(mid) < target:
            low = mid
        else:
            high = mid
    return high


def cell_economics(
    *,
    spot: float,
    strike: float,
    ask: float,
    spread: float,
    sigma: float,
    minutes_to_expiry: float,
    hold_minutes: float = HOLD_MINUTES,
    fees: float = FEES_PER_ROUND_TRIP_USD,
) -> tuple[float, float]:
    """`(R, J)` in SPX points for one real quote.

    `R` is the favourable move that breaks even. `J` is the *magnitude* of the
    adverse move that loses `STOP_FRACTION` of the ask. `J = 0.0` is a real and
    important answer: spread and decay alone have already breached the stop
    before SPX moves at all.
    """

    if minutes_to_expiry <= hold_minutes or sigma <= 0.0 or spot <= 0.0:
        return float("inf"), float("nan")

    def pnl(move: float) -> float:
        return _exit_pnl(
            move,
            spot=spot,
            strike=strike,
            minutes_to_expiry=minutes_to_expiry,
            hold_minutes=hold_minutes,
            sigma=sigma,
            spread=spread,
            entry_ask=ask,
            fees=fees,
        )

    required = _solve_move(0.0, pnl, low=0.0, high=MAX_SEARCH_POINTS)

    stop_dollars = STOP_FRACTION * ask * CONTRACT_MULTIPLIER
    # Adverse moves are negative; searching downward for where P&L crosses the
    # stop. pnl is monotone increasing in move, so negate to reuse the solver.
    if pnl(0.0) <= stop_dollars:
        stop_move = 0.0
    else:
        low, high = 0.0, MAX_SEARCH_POINTS
        while high - low > SOLVE_TOLERANCE:
            mid = 0.5 * (low + high)
            if pnl(-mid) > stop_dollars:
                low = mid
            else:
                high = mid
        stop_move = high if pnl(-high) <= stop_dollars else float("inf")
    return required, stop_move


def _session_path(frame: pd.DataFrame, start: str, hold: int) -> list[float] | None:
    """Underlying prices from `start` through `hold` minutes, or None if short."""

    minutes = frame[["minute", "underlying_price"]].drop_duplicates("minute")
    minutes = minutes.sort_values("minute")
    order = minutes["minute"].tolist()
    if start not in order:
        return None
    i = order.index(start)
    window = minutes.iloc[i : i + hold + 1]["underlying_price"].tolist()
    if len(window) < hold + 1:
        return None
    return window


def _race_outcome(path: list[float], required: float, stop: float) -> str:
    """Which came first from a real path: break-even, the stop, or neither.

    Ties inside a minute are unresolvable at this resolution and are charged to
    the adverse side, which is the conservative reading.
    """

    if not math.isfinite(required):
        return "flat"
    entry = path[0]
    for price in path[1:]:
        move = price - entry
        if math.isfinite(stop) and -move >= stop and stop >= 0.0:
            adverse_hit = True
        else:
            adverse_hit = False
        favourable_hit = move >= required
        if adverse_hit and favourable_hit:
            return "lose"
        if adverse_hit:
            return "lose"
        if favourable_hit:
            return "win"
    return "flat"


def _wilson(count: int, total: int) -> tuple[float, float]:
    """Wilson score interval, the convention used by the race artifacts."""

    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = count / total
    d = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class JoinResult:
    per_session: pd.DataFrame
    summary: pd.DataFrame
    skipped: pd.DataFrame


def run_analysis(ladder_root: Path, *, sessions: list[Path] | None = None) -> JoinResult:
    """Measure `R`, `J` and the three-state race per session, then pool."""

    ladder_root = Path(ladder_root)
    files = sessions if sessions is not None else sorted(ladder_root.glob("*.parquet"))
    rows: list[dict] = []
    skipped: list[dict] = []

    for path in files:
        session = path.stem
        if session in EXCLUDED_SESSIONS:
            skipped.append({"session": session, "reason": "excluded upstream as defective"})
            continue
        frame = pd.read_parquet(
            path,
            columns=[
                "minute", "strike", "is_call", "ask", "bid", "spread",
                "self_iv", "underlying_price", "minutes_to_expiry",
                "tensor_node_itm_points", "tensor_node_valid",
            ],
        )
        calls = frame[frame["is_call"]]
        for start in START_TIMES:
            path_prices = _session_path(frame, start, HOLD_MINUTES)
            if path_prices is None:
                skipped.append({"session": session, "start_time_et": start,
                                "reason": "path shorter than hold"})
                continue
            at_minute = calls[calls["minute"] == start]
            if at_minute.empty:
                skipped.append({"session": session, "start_time_et": start,
                                "reason": "no call quotes at start"})
                continue
            for offset in OFFSETS_POINTS:
                node = at_minute[at_minute["tensor_node_itm_points"] == -offset]
                node = node[node["tensor_node_valid"]] if "tensor_node_valid" in node else node
                if node.empty:
                    skipped.append({"session": session, "start_time_et": start,
                                    "offset_points": offset, "reason": "node absent"})
                    continue
                q = node.iloc[0]
                ask, spread, sigma = float(q["ask"]), float(q["spread"]), float(q["self_iv"])
                spot, strike = float(q["underlying_price"]), float(q["strike"])
                mte = float(q["minutes_to_expiry"])
                if not (ask > 0 and sigma > 0):
                    skipped.append({"session": session, "start_time_et": start,
                                    "offset_points": offset, "reason": "unusable quote"})
                    continue
                required, stop = cell_economics(
                    spot=spot, strike=strike, ask=ask, spread=spread,
                    sigma=sigma, minutes_to_expiry=mte,
                )
                rows.append({
                    "session": session, "start_time_et": start,
                    "offset_points": offset, "spot": spot, "strike": strike,
                    "ask_usd": ask * CONTRACT_MULTIPLIER, "spread": spread,
                    "sigma": sigma, "minutes_to_expiry": mte,
                    "required_move_points": required, "stop_move_points": stop,
                    "outcome": _race_outcome(path_prices, required, stop),
                })

    per_session = pd.DataFrame(rows)
    summary = _summarise(per_session)
    return JoinResult(per_session, summary, pd.DataFrame(skipped))


def _summarise(per_session: pd.DataFrame) -> pd.DataFrame:
    """Pool per-session cells into the finding's table shape."""

    if per_session.empty:
        return pd.DataFrame()
    out = []
    grouped = per_session.groupby(["start_time_et", "offset_points"], sort=True)
    for (start, offset), block in grouped:
        n = len(block)
        finite = block[block["required_move_points"].apply(math.isfinite)]
        counts = block["outcome"].value_counts()
        row = {
            "start_time_et": start,
            "offset_points": offset,
            "sessions": n,
            "median_sigma": block["sigma"].median(),
            "median_ask_usd": block["ask_usd"].median(),
            "median_required_move_points": finite["required_move_points"].median()
                if len(finite) else float("inf"),
            "median_stop_move_points": block["stop_move_points"].median(),
            "unreachable_sessions": n - len(finite),
        }
        for state in ("win", "lose", "flat"):
            c = int(counts.get(state, 0))
            lo, hi = _wilson(c, n)
            row[f"{state}_count"] = c
            row[f"{state}_probability"] = c / n if n else float("nan")
            row[f"{state}_ci_95_low"] = lo
            row[f"{state}_ci_95_high"] = hi
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Expectancy: what the position is actually worth at the exit
# ---------------------------------------------------------------------------
#
# Reaching `R` means breaking even. To learn what a trade *keeps*, the position
# has to be repriced at the exit at the spot the tape actually delivered, under
# a stated exit rule. This is the join the finding names as its next step.


def _exit_dollars(
    *,
    spot_at_exit: float,
    entry_spot: float,
    strike: float,
    sigma: float,
    minutes_to_expiry: float,
    hold_minutes: float,
    spread: float,
    entry_ask: float,
    fees: float = FEES_PER_ROUND_TRIP_USD,
) -> float:
    """Dollars kept, repricing the contract at the exit spot."""

    return _exit_pnl(
        spot_at_exit - entry_spot,
        spot=entry_spot,
        strike=strike,
        minutes_to_expiry=minutes_to_expiry,
        hold_minutes=hold_minutes,
        sigma=sigma,
        spread=spread,
        entry_ask=entry_ask,
        fees=fees,
    )


def simulate_trade(
    path: list[float],
    *,
    strike: float,
    sigma: float,
    minutes_to_expiry: float,
    spread: float,
    entry_ask: float,
    stop_move: float,
    target_move: float | None,
    fees: float = FEES_PER_ROUND_TRIP_USD,
) -> tuple[float, str, int]:
    """Run one trade to its exit and return `(dollars, reason, minutes_held)`.

    The exit rule, stated rather than fitted: leave on the declared stop, on the
    target if one is given, otherwise at the horizon. The stop is evaluated
    before the target within a minute, which charges ambiguous bars to the loss.
    """

    entry = path[0]
    for i, price in enumerate(path[1:], start=1):
        move = price - entry
        if math.isfinite(stop_move) and -move >= stop_move:
            return (
                _exit_dollars(
                    spot_at_exit=price, entry_spot=entry, strike=strike,
                    sigma=sigma, minutes_to_expiry=minutes_to_expiry,
                    hold_minutes=i, spread=spread, entry_ask=entry_ask, fees=fees,
                ),
                "stop",
                i,
            )
        if target_move is not None and move >= target_move:
            return (
                _exit_dollars(
                    spot_at_exit=price, entry_spot=entry, strike=strike,
                    sigma=sigma, minutes_to_expiry=minutes_to_expiry,
                    hold_minutes=i, spread=spread, entry_ask=entry_ask, fees=fees,
                ),
                "target",
                i,
            )
    held = len(path) - 1
    return (
        _exit_dollars(
            spot_at_exit=path[-1], entry_spot=entry, strike=strike, sigma=sigma,
            minutes_to_expiry=minutes_to_expiry, hold_minutes=held,
            spread=spread, entry_ask=entry_ask, fees=fees,
        ),
        "horizon",
        held,
    )
