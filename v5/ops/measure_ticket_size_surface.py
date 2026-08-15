"""What does a cheaper ticket cost in accuracy, and is any of it winnable?

The charter risk check found the binding constraint on this project is not
statistics, it is **the size of one ticket against the account**. A near-ATM
0DTE contract costs about $1,050, which is 10.5% of a $10,000 account, so a
single wrong call is a 5% day and the daily breaker permits roughly one trade a
session at any hold length.

There are only two ways out of that and both are charter changes: buy a cheaper
contract, or widen the breaker. This module measures the first, because it is
the one with a hard limit — the charter already bars deep out-of-the-money
contracts as *unwinnable*, on evidence that being right earns $4 while being
wrong costs $21.

The sweep is by **ticket size**, not by moneyness band, because the ticket is
what the account constraint is expressed in. At every slot and on each side the
contract whose entry premium sits closest to a declared target is selected,
using entry-minute information only. Where that lands on the moneyness ladder is
recorded as a result rather than assumed.

**Cost is charged by band, not flat.** A $35 contract does not pay the $25
round trip a $1,945 contract pays, and the ratio is what decides whether a cheap
contract can win at all. The per-band constants are carried from the owned quote
corpus, where the spread was measured per contract.

Fits nothing, searches nothing, proposes no policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.measure_hold_occupancy import (
    FIRST_INDEX,
    strata_means,
    HOLDS_MINUTES,
    LAST_INDEX,
    OUTCOME_QUANTILES,
    _label,
    intraclass_correlation,
    session_pivot,
)
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER, TRADE_CORPUS
from v5.research import statistics as st

# Declared ticket targets in dollars of premium. The ladder brackets the charter
# ceiling ($1,300 on a $10,000 account) and reaches down to the cheap end the
# charter currently bars, so the bar can be measured rather than inherited.
TICKET_TARGETS_USD = (50.0, 100.0, 200.0, 400.0, 800.0, 1_600.0, 3_200.0)

# Round trip by moneyness band, measured per contract on the owned quote corpus
# 2026-08-13 and carried here. Signed moneyness: positive is in the money.
BANDS = (
    (-1e9, -100.0, "deep OTM", 9.0),
    (-100.0, -25.0, "OTM", 12.0),
    (-25.0, 25.0, "near ATM", 25.0),
    (25.0, 100.0, "ITM", 84.0),
    (100.0, 300.0, "deep ITM", 369.0),
    (300.0, 1e9, "very deep ITM", 822.0),
)


def _band_index(moneyness: np.ndarray) -> np.ndarray:
    edges = np.array([b[1] for b in BANDS[:-1]])
    return np.searchsorted(edges, moneyness, side="left")


def session_tickets(path: Path, hold: int) -> pd.DataFrame | None:
    """One row per slot, side and ticket target."""

    got = session_pivot(path)
    if got is None:
        return None
    pivot, spot = got
    filled = pivot.ffill()
    strikes = pivot.columns.get_level_values("strike").to_numpy(float)
    is_call = pivot.columns.get_level_values("right").to_numpy() == "C"
    costs = np.array([b[3] for b in BANDS])
    names = [b[2] for b in BANDS]

    rows = []
    entry_index = FIRST_INDEX
    while entry_index + hold <= LAST_INDEX:
        entry_label, exit_label = _label(entry_index), _label(entry_index + hold)
        entry_index += hold
        if entry_label not in pivot.index or exit_label not in pivot.index:
            continue
        s0, s1 = spot.get(entry_label), spot.get(exit_label)
        if s0 is None or s1 is None or not np.isfinite(s0) or not np.isfinite(s1):
            continue
        # Never skip a small move: that conditions on the outcome.
        if s0 == s1:
            continue

        entry_price = pivot.loc[entry_label].to_numpy(float)
        exit_present = pivot.loc[exit_label].to_numpy(float)
        exit_price = np.where(
            np.isfinite(exit_present), exit_present, filled.loc[exit_label].to_numpy(float)
        )
        tradeable = np.isfinite(entry_price) & np.isfinite(exit_price)
        if not tradeable.any():
            continue
        moneyness = np.where(is_call, s0 - strikes, strikes - s0)
        premium = entry_price * CONTRACT_MULTIPLIER
        band = _band_index(moneyness)
        up = bool(s1 > s0)

        for side in (True, False):
            on_side = np.flatnonzero(tradeable & (is_call == side))
            if not on_side.size:
                continue
            for target in TICKET_TARGETS_USD:
                i = int(on_side[np.argmin(np.abs(premium[on_side] - target))])
                rows.append(
                    {
                        "session": path.name[:10],
                        "entry_minute": entry_label,
                        "target_usd": target,
                        "premium": premium[i],
                        "moneyness": moneyness[i],
                        "band": names[band[i]],
                        "round_trip_usd": costs[band[i]],
                        "net_usd": (exit_price[i] - entry_price[i]) * CONTRACT_MULTIPLIER
                        - costs[band[i]],
                        "correct": bool(is_call[i] == up),
                        "up": up,
                    }
                )
    return pd.DataFrame(rows) if rows else None


def assess_target(part: pd.DataFrame, sessions: int) -> dict:
    """Break-even and detectability for one ticket size at one hold."""

    net = part["net_usd"].to_numpy(float)
    ok = part["correct"].to_numpy(bool)
    if ok.sum() < 50 or (~ok).sum() < 50:
        return {}
    win, loss = float(net[ok].mean()), float(-net[~ok].mean())
    slots = part[["session", "entry_minute"]].drop_duplicates()
    out = {
        "trades_total": int(len(slots)),
        "trades_per_session": round(len(slots) / sessions, 2) if sessions else 0.0,
        "mean_premium_usd": round(float(part["premium"].mean()), 2),
        "median_premium_usd": round(float(part["premium"].median()), 2),
        "mean_round_trip_usd": round(float(part["round_trip_usd"].mean()), 2),
        "round_trip_share_of_premium": round(
            float(part["round_trip_usd"].mean() / part["premium"].mean()), 4
        ),
        "mean_moneyness": round(float(part["moneyness"].mean()), 2),
        "band_mix": {
            k: round(v, 4)
            for k, v in part["band"].value_counts(normalize=True).round(4).items()
        },
        "mean_net_when_correct_usd": round(win, 2),
        "mean_net_when_wrong_usd": round(-loss, 2),
    }
    if win <= 0:
        out["breakeven_accuracy"] = None
        out["verdict"] = "unwinnable: being right does not pay for the round trip"
        return out

    directions = part.drop_duplicates(subset=["session", "entry_minute"])
    icc, k0 = intraclass_correlation(
        directions["up"].to_numpy(float), directions["session"].to_numpy()
    )
    design_effect = 1.0 + (k0 - 1.0) * icc
    effective = max(1, int(len(slots) / design_effect))
    kwargs = {"win": win, "loss": loss, "z_alpha": st.Z_95}
    out |= {
        "breakeven_accuracy": round(st.breakeven_accuracy(win=win, loss=loss), 6),
        "design_effect": round(design_effect, 4),
        "effective_trades": effective,
        "provable_accuracy": round(st.detectable_accuracy(effective, **kwargs), 6),
        "outcome_when_correct": {
            "mean_usd": round(win, 4),
            "quantiles_usd": [
                round(float(v), 4) for v in np.quantile(net[ok], OUTCOME_QUANTILES)
            ],
            "strata_usd": strata_means(net[ok]),
            "strata_ratio": strata_means(
                net[ok] / part["premium"].to_numpy(float)[ok]
            ),
        },
        "outcome_when_wrong": {
            "mean_usd": round(-loss, 4),
            "quantiles_usd": [
                round(float(v), 4) for v in np.quantile(net[~ok], OUTCOME_QUANTILES)
            ],
            "strata_usd": strata_means(net[~ok]),
            "strata_ratio": strata_means(
                net[~ok] / part["premium"].to_numpy(float)[~ok]
            ),
        },
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=TRADE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.corpus.glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]

    results = {}
    for hold in HOLDS_MINUTES:
        parts, sessions = [], 0
        for i, path in enumerate(files, 1):
            got = session_tickets(path, hold)
            if got is not None:
                parts.append(got)
                sessions += 1
            if i % 200 == 0:
                print(f"  {hold}m: {i}/{len(files)}", flush=True)
        if not parts:
            continue
        table = pd.concat(parts, ignore_index=True)
        results[f"{hold}m"] = {
            "sessions": sessions,
            "by_ticket": {
                f"${int(target)}": row
                for target in TICKET_TARGETS_USD
                if (row := assess_target(table[table["target_usd"] == target], sessions))
            },
        }

    payload = {
        "schema_version": "v5.ticket-size-surface.v1",
        "question": (
            "The account constraint is expressed in ticket dollars. What does a "
            "cheaper ticket cost in the accuracy a screen must prove, and where "
            "does it stop being winnable at all?"
        ),
        "ticket_targets_usd": list(TICKET_TARGETS_USD),
        "selection_rule": (
            "at each slot and on each side, the contract whose entry premium is "
            "closest to the target; entry-minute information only"
        ),
        "exit_price_convention": (
            "the print at the exit minute where one exists, else the contract's "
            "last print inside the holding period"
        ),
        "round_trip_by_band_usd": {b[2]: b[3] for b in BANDS},
        "round_trip_provenance": (
            "measured per contract on the owned quote corpus 2026-08-13 and "
            "carried here; not measured on this corpus"
        ),
        "corpus": str(args.corpus),
        "files_scanned": len(files),
        "by_hold": results,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for hold, block in results.items():
        print(f"\n{hold} hold, {block['sessions']} sessions")
        head = (
            f"  {'target':>8} {'premium':>9} {'cost/prem':>10} {'moneyness':>10} "
            f"{'band':>13} {'break-even':>11} {'provable':>10}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for name, row in block["by_ticket"].items():
            be = row.get("breakeven_accuracy")
            top = max(row["band_mix"], key=row["band_mix"].get)
            be_txt = f"{100 * be:.2f}%" if be else "unwinnable"
            prov_txt = f"{100 * row['provable_accuracy']:.2f}%" if be else "-"
            print(
                f"  {name:>8} {row['mean_premium_usd']:>9,.0f} "
                f"{100 * row['round_trip_share_of_premium']:>9.1f}% "
                f"{row['mean_moneyness']:>10.1f} {top:>13} "
                f"{be_txt:>11} {prov_txt:>10}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
