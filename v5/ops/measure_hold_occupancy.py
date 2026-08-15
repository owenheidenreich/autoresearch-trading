"""Hold length against occupancy, on the exit-price convention Phase 0 settled.

The job-24 plan rests on one table: shorter serial holds fit more trades into a
session, and more trades lower the accuracy a screen can prove.  That table was
built with the **drop** convention for a contract that stops printing before its
exit minute, which Phase 0 measured to be wrong — it discards contracts that are
89-99% winners, so every break-even in it is two to five points too pessimistic.

This module rebuilds it, and changes three other things that were assumed:

**One position, one contract.** The bot holds one position at a time, so the
unit of a power calculation is the *slot*, not the contract-observation. Each
slot is represented by the contract closest to at-the-money, a rule that reads
only the entry minute.  The contract-level break-even is still reported next to
it, because that is what the earlier receipts measured and the two must be
comparable.

**Trades per session is measured, not assumed.** ``385 / hold`` is an upper
bound that assumes every slot can be priced.  A slot with too thin a chain to
locate the spot is skipped and counted.

**Trades inside a session are not independent.** A serial clock run through one
session shares that session's direction, so treating every trade as a fresh
observation overstates what a screen can see.  The intra-session correlation is
measured and charged as a design effect.

Fits nothing, searches nothing, proposes no policy.  Holds, band, cost and
entry time are declared constants.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.resolve_exit_price_convention import (
    CONTRACT_MULTIPLIER,
    NEAR_ATM_POINTS,
    ROUND_TRIP_USD,
    TRADE_CORPUS,
    breakeven,
)
from v5.research import statistics as st

FIRST_ENTRY_MINUTE = "09:35"
LAST_EXIT_MINUTE = "16:00"
HOLDS_MINUTES = (5, 10, 15, 30, 60)

# A parity spot read off two or three traded strikes is noise. A slot that
# cannot show this many strikes quoting both a call and a put is skipped and
# counted rather than priced badly.
MIN_PAIRED_STRIKES = 5

# Strikes within this distance of a first parity estimate are averaged for the
# refined one. A far in-the-money strike whose other side barely trades carries
# a stale parity and must not drag the estimate.
PARITY_WINDOW_POINTS = 30.0

# The exit the signed charter amendment names. It is declared, never searched,
# and it is not optional here: the amendment's case for a 13% ticket rests on
# the loss after this exit being 4.29% of the account rather than 7.55%, so a
# risk check run without it is checking a position size the charter does not
# permit. The fill is taken at the first minute at or below the level, so the
# gap between the declared stop and the realised one is measured rather than
# assumed away.
STOP_LEVEL = -0.30


def _index(minute: str) -> int:
    hours, minutes = (int(part) for part in minute.split(":"))
    return hours * 60 + minutes


def _label(index: int) -> str:
    return f"{index // 60:02d}:{index % 60:02d}"


FIRST_INDEX = _index(FIRST_ENTRY_MINUTE)
LAST_INDEX = _index(LAST_EXIT_MINUTE)
TRADEABLE_MINUTES = LAST_INDEX - FIRST_INDEX


def session_pivot(path: Path) -> tuple[pd.DataFrame, pd.Series] | None:
    """Close price per minute per contract, plus the parity spot per minute.

    Returns the raw pivot — NaN where a contract did not print — because the
    difference between a print and no print is the whole subject of Phase 0.
    """

    try:
        frame = pd.read_parquet(path, columns=["ts_event", "close", "strike", "right"])
    except Exception:
        return None
    if frame.empty:
        return None
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce").astype(float)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    frame = frame.dropna(subset=["strike", "close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return None
    frame["minute"] = (
        pd.to_datetime(frame["ts_event"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    pivot = frame.pivot_table(
        index="minute", columns=["strike", "right"], values="close", aggfunc="last"
    ).sort_index()
    if pivot.empty:
        return None

    calls = pivot.loc[:, pivot.columns.get_level_values("right") == "C"]
    puts = pivot.loc[:, pivot.columns.get_level_values("right") == "P"]
    calls.columns = calls.columns.get_level_values("strike")
    puts.columns = puts.columns.get_level_values("strike")
    shared = calls.columns.intersection(puts.columns)
    if len(shared) < MIN_PAIRED_STRIKES:
        return None
    # Put/call parity, ``S = K + C - P``, averaged over the strikes nearest the
    # money. The discount factor is negligible over hours.
    #
    # This replaced an estimator that took the strike where |C - P| was
    # smallest, which quantised the underlying to the five-point strike grid.
    # That was not a precision nuisance: slots where the quantised spot did not
    # move were skipped, which silently discarded 18.7% of sixty-minute slots
    # averaging -$186.90 each. Skipping them is a look-ahead filter, because
    # nothing at entry says how far the underlying will travel.
    implied = (calls[shared] - puts[shared]).add(pd.Series(shared, index=shared))
    coarse = implied.median(axis=1)
    offsets = np.abs(implied.columns.to_numpy(float) - coarse.to_numpy()[:, None])
    near = implied.where(offsets <= PARITY_WINDOW_POINTS)
    enough = near.notna().sum(axis=1) >= MIN_PAIRED_STRIKES
    if not enough.any():
        return None
    return pivot, near.mean(axis=1).where(enough)


def _stopped_exit(path: np.ndarray, entry: float, held: float) -> float:
    """Exit price under the declared stop, given the price path of one contract.

    The path starts at the entry minute, so the first element is skipped: a stop
    cannot fire on the price it was bought at. Where the level is never breached
    the trade rides to the horizon and returns the held exit price.
    """

    after = path[1:]
    if after.size == 0:
        return held
    breached = np.flatnonzero(np.isfinite(after) & (after / entry - 1.0 <= STOP_LEVEL))
    return float(after[breached[0]]) if breached.size else held


SKIP_REASONS = ("minute_absent", "spot_unplaceable", "exact_tie", "no_eligible_contract")


def session_trades(path: Path, hold: int) -> tuple[pd.DataFrame | None, int, dict]:
    """Every serial slot in one session at one hold length.

    Returns the trades, the number of slots offered by the clock, and a
    breakdown of why the rest were skipped. Only a chain too thin to price is a
    reason to skip: a slot is never dropped for a small move, because nothing at
    entry says how far the underlying will travel and dropping those slots
    conditions on the outcome.
    """

    skipped = dict.fromkeys(SKIP_REASONS, 0)
    got = session_pivot(path)
    if got is None:
        return None, 0, skipped
    pivot, spot = got
    filled = pivot.ffill()
    strikes = pivot.columns.get_level_values("strike").to_numpy(float)
    is_call = pivot.columns.get_level_values("right").to_numpy() == "C"

    rows, offered = [], 0
    entry_index = FIRST_INDEX
    while entry_index + hold <= LAST_INDEX:
        offered += 1
        entry_label, exit_label = _label(entry_index), _label(entry_index + hold)
        entry_index += hold
        if entry_label not in pivot.index or exit_label not in pivot.index:
            skipped["minute_absent"] += 1
            continue
        s0, s1 = spot.get(entry_label), spot.get(exit_label)
        if s0 is None or s1 is None or not np.isfinite(s0) or not np.isfinite(s1):
            skipped["spot_unplaceable"] += 1
            continue
        # A slot is NOT skipped for a small move. Doing so conditions on the
        # outcome: the discarded slots are the ones where the underlying went
        # nowhere, which is exactly where long premium loses to decay.
        if s0 == s1:
            skipped["exact_tie"] += 1
            continue

        entry_price = pivot.loc[entry_label].to_numpy(float)
        moneyness = np.where(is_call, s0 - strikes, strikes - s0)
        eligible = np.isfinite(entry_price) & (np.abs(moneyness) <= NEAR_ATM_POINTS)
        if not eligible.any():
            skipped["no_eligible_contract"] += 1
            continue

        exit_present = pivot.loc[exit_label].to_numpy(float)
        # The settled convention: the print at the exit minute where one exists,
        # otherwise the contract's last print inside the holding period.
        exit_price = np.where(np.isfinite(exit_present), exit_present,
                              filled.loc[exit_label].to_numpy(float))
        up = bool(s1 > s0)
        correct = is_call == up
        # One position at a time, so the unit is the slot. A policy picks a side
        # and then the nearest contract to the money on that side, both decided
        # on entry-minute information alone. Recording the nearest call *and*
        # the nearest put gives the conditional pair the break-even needs — what
        # the right side earned and what the wrong side lost — without a
        # tie-break between two contracts that sit equally close to the money.
        chosen = set()
        for side in (True, False):
            on_side = np.flatnonzero(eligible & (is_call == side))
            if on_side.size:
                chosen.add(int(on_side[np.argmin(np.abs(moneyness[on_side]))]))
        # The minute-by-minute path inside the holding period, which is what a
        # stop needs and what entry and exit prices alone cannot show.
        path_window = filled.loc[entry_label:exit_label]
        for i in np.flatnonzero(eligible):
            if not np.isfinite(exit_price[i]):
                continue
            rows.append(
                {
                    "session": path.name[:10],
                    "entry_minute": entry_label,
                    "entry_close": entry_price[i],
                    "exit_close": exit_price[i],
                    "stopped_close": (
                        _stopped_exit(
                            path_window.iloc[:, i].to_numpy(float),
                            entry_price[i],
                            exit_price[i],
                        )
                        if i in chosen
                        else np.nan
                    ),
                    "correct": bool(correct[i]),
                    "chosen": i in chosen,
                    "up": up,
                    "vanished": bool(not np.isfinite(exit_present[i])),
                }
            )
    return (pd.DataFrame(rows) if rows else None), offered, skipped


def intraclass_correlation(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """One-way ICC of ``values`` within ``groups``, and the effective group size.

    Applied to the *direction* of each slot, which is what makes a session's
    trades non-independent: if a session trends, every slot points the same way
    and a policy that reads it correctly is correct on all of them at once, so
    the session carries far less information than its trade count suggests. The
    two limits are the right ones — direction that persists perfectly across a
    session reduces the effective sample to one observation per session, and
    direction that is a fresh coin flip each slot leaves every trade counting in
    full.
    """

    frame = pd.DataFrame({"value": values.astype(float), "group": groups})
    grouped = frame.groupby("group")["value"]
    sizes = grouped.size().to_numpy(float)
    k, n = len(sizes), float(len(frame))
    if k < 2 or n <= k:
        return 0.0, float(sizes.mean()) if len(sizes) else 0.0
    group_mean = grouped.transform("mean").to_numpy(float)
    grand = float(frame["value"].mean())
    between = float(np.sum((group_mean - grand) ** 2)) / (k - 1)
    within = float(np.sum((frame["value"].to_numpy(float) - group_mean) ** 2)) / (n - k)
    # Correction for unequal group sizes; k0 is the effective group size.
    k0 = (n - float(np.sum(sizes**2)) / n) / (k - 1)
    if k0 <= 0:
        return 0.0, float(sizes.mean())
    if within <= 0:
        # Every group is internally constant. That is total clustering when the
        # groups differ from each other, and no information at all when they do
        # not. Falling through to 0.0 here would report the worst case as the
        # best one.
        return (1.0 if between > 0 else 0.0), k0
    icc = (between - within) / (between + (k0 - 1) * within)
    return max(0.0, min(1.0, icc)), k0


# Quantiles of the per-trade outcome, recorded so a downstream risk check can
# resample the measured distribution instead of assuming a two-point payoff.
# Every trade is one position, so this is also the distribution of the account's
# per-trade return once it is scaled by the position size.
OUTCOME_QUANTILES = tuple(round(0.005 + 0.0099 * i, 4) for i in range(100))

# How many equal-probability strata to summarise an outcome distribution with.
STRATA = 100


def strata_means(values: np.ndarray, strata: int = STRATA) -> list[float]:
    """Mean of each equal-probability slice of a distribution.

    Resampling uniformly from a quantile grid is not the same thing as sampling
    the distribution: a draw can never exceed the highest quantile stored, so
    the resampled mean is a *trimmed* mean. For an option payoff that matters a
    great deal, because a large share of the expected value lives in the right
    tail — measured here, a hundred-point quantile grid understates the mean
    winning trade by 6% to 19% while leaving the bounded losing side accurate to
    under 1%, which is a systematic bias against the strategy.

    Splitting the sorted sample into equal-count slices and storing each slice's
    mean fixes it exactly: uniform draws from these values reproduce the true
    mean, and the top slice carries the extreme tail rather than discarding it.
    """

    ordered = np.sort(np.asarray(values, float))
    if ordered.size == 0:
        return []
    groups = np.array_split(ordered, min(strata, ordered.size))
    return [round(float(g.mean()), 4) for g in groups if g.size]


def outcome_ratios(part: pd.DataFrame, exit_column: str = "exit_close") -> dict:
    """Per-trade outcome as measured quantiles, in both units.

    ``ratio`` is net return on the premium paid, so ``-1.0`` is a contract that
    expired worthless. ``usd`` is the same trade in dollars.

    The dollar column is the one a risk check must use. The charter buys **one
    contract**, so the money at stake is whatever that contract costs — not a
    chosen fraction of the account. Working in ratios instead would silently
    assume a fractional position and would over-weight cheap contracts, whose
    returns are far more extreme per dollar committed.
    """

    premium = part["entry_close"].to_numpy(float) * CONTRACT_MULTIPLIER
    net = (part[exit_column].to_numpy(float) - part["entry_close"].to_numpy(float))
    net = net * CONTRACT_MULTIPLIER - ROUND_TRIP_USD
    ratio = net / premium
    return {
        "n": int(len(ratio)),
        "mean": round(float(ratio.mean()), 6),
        "mean_usd": round(float(net.mean()), 4),
        "quantile_levels": list(OUTCOME_QUANTILES),
        "quantiles": [round(float(v), 6) for v in np.quantile(ratio, OUTCOME_QUANTILES)],
        "quantiles_usd": [round(float(v), 4) for v in np.quantile(net, OUTCOME_QUANTILES)],
        "premium_quantiles_usd": [
            round(float(v), 4) for v in np.quantile(premium, OUTCOME_QUANTILES)
        ],
        # What a risk check must resample from. See ``strata_means``.
        "strata_usd": strata_means(net),
        "strata_ratio": strata_means(ratio),
    }


def assess(trades: pd.DataFrame, offered: int, skipped: dict, sessions: int) -> dict:
    """Break-even and detectability for one hold length."""

    chosen = trades[trades["chosen"]]
    # One trade per slot: the slot is the decision, the two recorded contracts
    # are the two ways it could have gone.
    slots = chosen[["session", "entry_minute"]].drop_duplicates()
    contract_level = breakeven(
        trades["entry_close"].to_numpy(float),
        trades["exit_close"].to_numpy(float),
        trades["correct"].to_numpy(bool),
    )
    trade_level = breakeven(
        chosen["entry_close"].to_numpy(float),
        chosen["exit_close"].to_numpy(float),
        chosen["correct"].to_numpy(bool),
    )
    out = {
        "sessions": sessions,
        "slots_offered_per_session": round(offered / sessions, 2) if sessions else 0.0,
        "slots_skipped_per_session": (
            round(sum(skipped.values()) / sessions, 2) if sessions else 0.0
        ),
        "slots_skipped_by_reason_per_session": {
            reason: round(count / sessions, 2) if sessions else 0.0
            for reason, count in skipped.items()
        },
        "trades_per_session_measured": round(len(slots) / sessions, 2) if sessions else 0.0,
        "trades_total": int(len(slots)),
        "contract_observations": int(len(trades)),
        "vanished_share": round(float(trades["vanished"].mean()), 4),
        "mean_entry_premium_usd": round(
            float(chosen["entry_close"].mean() * CONTRACT_MULTIPLIER), 2
        ),
        "contract_level": contract_level,
        "trade_level": trade_level,
        "trade_level_with_declared_stop": breakeven(
            chosen["entry_close"].to_numpy(float),
            chosen["stopped_close"].to_numpy(float),
            chosen["correct"].to_numpy(bool),
        ),
        "declared_stop_level": STOP_LEVEL,
        "declared_stop_fired_share": round(
            float((chosen["stopped_close"] != chosen["exit_close"]).mean()), 4
        ),
        "outcome_when_correct": outcome_ratios(chosen[chosen["correct"]]),
        "outcome_when_wrong": outcome_ratios(chosen[~chosen["correct"]]),
        "outcome_when_correct_stopped": outcome_ratios(
            chosen[chosen["correct"]], "stopped_close"
        ),
        "outcome_when_wrong_stopped": outcome_ratios(
            chosen[~chosen["correct"]], "stopped_close"
        ),
    }
    win = trade_level.get("mean_net_when_correct_usd")
    loss = trade_level.get("mean_net_when_wrong_usd")
    if not win or not loss or win <= 0:
        out["detectable"] = None
        return out

    directions = chosen.drop_duplicates(subset=["session", "entry_minute"])
    icc, mean_group = intraclass_correlation(
        directions["up"].to_numpy(float), directions["session"].to_numpy()
    )
    design_effect = 1.0 + (mean_group - 1.0) * icc
    trades_total = len(slots)
    effective = trades_total / design_effect if design_effect > 0 else trades_total
    kwargs = {"win": win, "loss": -loss, "z_alpha": st.Z_95}
    out["detectable"] = {
        "breakeven_accuracy": round(st.breakeven_accuracy(win=win, loss=-loss), 6),
        "intraclass_correlation": round(icc, 6),
        "design_effect": round(design_effect, 4),
        "effective_trades": round(effective, 1),
        "accuracy_if_trades_were_independent": round(
            st.detectable_accuracy(trades_total, **kwargs), 6
        ),
        "accuracy_at_effective_n": round(
            st.detectable_accuracy(max(1, int(effective)), **kwargs), 6
        ),
        "accuracy_at_effective_n_with_conjunction_penalty": round(
            st.detectable_accuracy(max(1, int(effective)), penalty=2.0, **kwargs), 6
        ),
        "z_alpha": "one-sided 95% for a single pre-registered hypothesis",
        "penalty_note": (
            "2.0 is the measured G1 conjunction penalty at its lower end, charged "
            "for the fold rule, the bootstrap lower bound and the paired "
            "comparator. It is a screening bracket; a declared family must still "
            "compute its own power."
        ),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, action="append", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    roots = args.corpus or [TRADE_CORPUS]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.glob("*.parquet")))
    files = sorted(files, key=lambda f: f.name)
    if args.limit:
        files = files[: args.limit]

    results = {}
    for hold in HOLDS_MINUTES:
        parts, offered, sessions = [], 0, 0
        skipped = dict.fromkeys(SKIP_REASONS, 0)
        for i, path in enumerate(files, 1):
            got, slots, missed = session_trades(path, hold)
            offered += slots
            for reason, count in missed.items():
                skipped[reason] += count
            if got is not None:
                parts.append(got)
                sessions += 1
            if i % 200 == 0:
                print(f"  {hold}m: {i}/{len(files)}", flush=True)
        if not parts:
            continue
        trades = pd.concat(parts, ignore_index=True)
        results[f"{hold}m"] = assess(trades, offered, skipped, sessions)

    payload = {
        "schema_version": "v5.hold-occupancy.v1",
        "exit_price_convention": (
            "the print at the exit minute where one exists, else the contract's "
            "last print inside the holding period — settled by "
            "v5/ops/resolve_exit_price_convention.py on 2026-08-13"
        ),
        "first_entry_minute": FIRST_ENTRY_MINUTE,
        "last_exit_minute": LAST_EXIT_MINUTE,
        "tradeable_minutes": TRADEABLE_MINUTES,
        "moneyness_band_points": NEAR_ATM_POINTS,
        "round_trip_cost_usd": ROUND_TRIP_USD,
        "round_trip_cost_provenance": "CARRIED from the owned quote corpus, not measured here",
        "contract_multiplier": CONTRACT_MULTIPLIER,
        "corpora": [str(root) for root in roots],
        "files_scanned": len(files),
        "by_hold": results,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    head = (
        f"\n{'hold':>5} {'trades/sess':>12} {'trades':>8} {'break-even':>11} "
        f"{'ICC':>7} {'deff':>6} {'eff n':>8} {'provable':>10} {'+penalty':>10}"
    )
    print(head)
    print("-" * (len(head) - 1))
    for name, row in results.items():
        det = row["detectable"]
        if not det:
            print(f"{name:>5} {row['trades_per_session_measured']:>12.1f} — not measurable")
            continue
        print(
            f"{name:>5} {row['trades_per_session_measured']:>12.1f} "
            f"{row['trades_total']:>8,} {100 * det['breakeven_accuracy']:>10.2f}% "
            f"{det['intraclass_correlation']:>7.3f} {det['design_effect']:>6.2f} "
            f"{det['effective_trades']:>8,.0f} "
            f"{100 * det['accuracy_at_effective_n']:>9.2f}% "
            f"{100 * det['accuracy_at_effective_n_with_conjunction_penalty']:>9.2f}%"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
