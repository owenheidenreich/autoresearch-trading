"""What does the option charge for movement, and what does movement deliver?

The magnitude screen returned a result cleaner than a negative. Selecting the
busiest third of hours raises the realised move by roughly half — 4.67 points to
7.05 at fifteen minutes — and raises the straddle premium by almost exactly the
same proportion, $1,707 to $2,420. The profit and loss does not move: -$29.3
against -$29.6. **Volatility is predictable and it is priced.**

That makes the remaining question arithmetic rather than statistical, and this
module does the arithmetic on measured prices. Buying the at-the-money straddle
and holding it is a pure bet that the underlying travels further than the option
charged. Its gross profit and loss **is** the variance risk premium for this
instrument and horizon, with no model in between.

Three things are reported, because together they close the question:

* the **gross** premium, before any cost, which is what the market charges;
* the **net** for a buyer and for a seller at three execution qualities, which
  is what either side actually keeps;
* the same split by time of day and by volatility regime, to show whether the
  premium is uniform or concentrated somewhere a policy could stand.

No rule, model or threshold appears here. It is a measurement of the instrument.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fees only, half the measured spread, and the full aggressive round trip.
COSTS_PER_LEG_USD = (3.08, 14.0, 25.0)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260813


def bootstrap_ci(values: np.ndarray, sessions: np.ndarray) -> tuple[float, float]:
    """Session-block bootstrap, two-sided 95%."""

    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = values[np.concatenate([index[s] for s in chosen])].mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarise(part: pd.DataFrame, label: str) -> dict:
    gross = part["straddle_gross"].to_numpy(float)
    premium = part["straddle_premium"].to_numpy(float)
    lo, hi = bootstrap_ci(gross, part["session"].to_numpy())
    out = {
        "population": label,
        "slots": int(len(part)),
        "sessions": int(part["session"].nunique()),
        "mean_abs_move_points": round(float(part["abs_move"].mean()), 3),
        "mean_straddle_premium_usd": round(float(premium.mean()), 2),
        # Negative means the option charged more than the move delivered, which
        # is the variance risk premium accruing to the seller.
        "mean_straddle_gross_usd": round(float(gross.mean()), 2),
        "gross_share_of_premium": round(float(gross.mean() / premium.mean()), 6),
        "gross_ci95_usd": [round(lo, 2), round(hi, 2)],
        "gross_is_negative_with_confidence": bool(hi < 0.0),
        "by_cost": {},
    }
    for cost in COSTS_PER_LEG_USD:
        # A straddle is two legs, so two round trips, for buyer and seller alike.
        out["by_cost"][f"${cost}/leg"] = {
            "buyer_net_usd": round(float(gross.mean() - 2 * cost), 2),
            "seller_net_usd": round(float(-gross.mean() - 2 * cost), 2),
            "buyer_profitable": bool(gross.mean() - 2 * cost > 0),
            "seller_profitable": bool(-gross.mean() - 2 * cost > 0),
            "cost_share_of_premium": round(float(2 * cost / premium.mean()), 6),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw = pd.read_parquet(args.table)
    results = {}
    for hold, part in raw.groupby("hold"):
        part = part.copy()
        part["hour"] = part["entry_minute"].str[:2]
        blocks = [summarise(part, "all slots")]
        for hour, chunk in part.groupby("hour", sort=True):
            if len(chunk) >= 400:
                blocks.append(summarise(chunk, f"entered {hour}:00"))
        # Volatility regime by the prior thirty minutes, split at its own median
        # over the whole sample. This is a descriptive cut, not a tradeable
        # rule, and it is labelled as such.
        median = part["range_30m"].median()
        blocks.append(summarise(part[part["range_30m"] > median], "busy prior 30 min"))
        blocks.append(summarise(part[part["range_30m"] <= median], "quiet prior 30 min"))
        results[f"{hold}m"] = blocks

    payload = {
        "schema_version": "v5.variance-premium.v1",
        "question": (
            "Buying the at-the-money straddle is a pure bet that the underlying "
            "travels further than the option charged. What is that bet worth "
            "gross, and what does either side keep after execution?"
        ),
        "costs_per_leg_usd": list(COSTS_PER_LEG_USD),
        "note": (
            "A straddle is two legs, so both buyer and seller pay two round "
            "trips. The regime split is descriptive: it uses a whole-sample "
            "median and is not a rule anyone could have traded."
        ),
        "source_table": str(args.table),
        "by_hold": results,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for hold, blocks in results.items():
        print(f"\n{hold} hold — what the option charges against what it delivers")
        head = (
            f"  {'population':>20} {'slots':>7} {'|move|':>7} {'straddle':>9} "
            f"{'gross':>8} {'as %':>7} {'buy @$3':>9} {'sell @$3':>9} {'sell @$25':>10}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for b in blocks:
            cheap, dear = b["by_cost"]["$3.08/leg"], b["by_cost"]["$25.0/leg"]
            print(
                f"  {b['population']:>20} {b['slots']:>7,} {b['mean_abs_move_points']:>7.2f} "
                f"{b['mean_straddle_premium_usd']:>9,.0f} {b['mean_straddle_gross_usd']:>8,.1f} "
                f"{100 * b['gross_share_of_premium']:>6.2f}% {cheap['buyer_net_usd']:>9,.1f} "
                f"{cheap['seller_net_usd']:>9,.1f} {dear['seller_net_usd']:>10,.1f}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
