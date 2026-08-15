"""Is one side of the chain priced better than the other, and is any of it stable?

The variance-premium measurement showed that the straddle — both legs together —
is priced above what the underlying delivers. That is a statement about the
*pair*. It leaves two questions that decide whether anything here is usable.

**Asymmetry.** If the put is dearer than its own realised downside while the call
is fair, then a long call is not a long-volatility position at all: it is a
cheap way to be long the underlying, and the direction screen's null would not
apply to it. Equity index options are usually skewed this way, so it has to be
measured rather than assumed.

**Stability.** A premium averaged over four years is only tradeable if it is
present in most of them. A number that comes entirely from 2022 is a fact about
2022. This reports every year separately, and the sign in each.

Both are measurements of the instrument. No rule, model or threshold appears.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.measure_variance_premium import bootstrap_ci

COSTS_PER_LEG_USD = (3.08, 14.0, 25.0)


def leg_summary(part: pd.DataFrame, label: str) -> dict:
    out = {"population": label, "slots": int(len(part)), "legs": {}}
    sessions = part["session"].to_numpy()
    for side in ("call", "put"):
        gross = part[f"{side}_gross"].to_numpy(float)
        premium = part[f"{side}_premium"].to_numpy(float)
        lo, hi = bootstrap_ci(gross, sessions)
        out["legs"][side] = {
            "mean_premium_usd": round(float(premium.mean()), 2),
            "mean_gross_usd": round(float(gross.mean()), 2),
            "gross_share_of_premium": round(float(gross.mean() / premium.mean()), 6),
            "gross_ci95_usd": [round(lo, 2), round(hi, 2)],
            "distinguishable_from_zero": bool(hi < 0.0 or lo > 0.0),
            "net_at_25_usd": round(float(gross.mean() - 25.0), 2),
            "net_at_3_08_usd": round(float(gross.mean() - 3.08), 2),
        }
    # The difference is the interesting quantity: a long call financed by a
    # short put is a synthetic long underlying, so its cost is the skew.
    diff = (part["call_gross"] - part["put_gross"]).to_numpy(float)
    lo, hi = bootstrap_ci(diff, sessions)
    out["call_minus_put"] = {
        "mean_usd": round(float(diff.mean()), 2),
        "ci95_usd": [round(lo, 2), round(hi, 2)],
        "distinguishable_from_zero": bool(hi < 0.0 or lo > 0.0),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw = pd.read_parquet(args.table)
    raw["year"] = raw["session"].str[:4]

    results = {}
    for hold, part in raw.groupby("hold"):
        blocks = [leg_summary(part, "all slots")]
        years = []
        for year, chunk in part.groupby("year", sort=True):
            if len(chunk) < 300:
                continue
            gross = chunk["straddle_gross"].to_numpy(float)
            lo, hi = bootstrap_ci(gross, chunk["session"].to_numpy())
            years.append(
                {
                    "year": year,
                    "slots": int(len(chunk)),
                    "mean_abs_move_points": round(float(chunk["abs_move"].mean()), 3),
                    "mean_straddle_premium_usd": round(
                        float(chunk["straddle_premium"].mean()), 2
                    ),
                    "mean_straddle_gross_usd": round(float(gross.mean()), 2),
                    "gross_share_of_premium": round(
                        float(gross.mean() / chunk["straddle_premium"].mean()), 6
                    ),
                    "gross_ci95_usd": [round(lo, 2), round(hi, 2)],
                    "negative_with_confidence": bool(hi < 0.0),
                }
            )
        results[f"{hold}m"] = {"legs": blocks, "by_year": years}

    payload = {
        "schema_version": "v5.leg-asymmetry.v1",
        "questions": [
            "is one leg priced better than the other, so that a long call is a "
            "cheap synthetic long rather than a volatility position?",
            "is the variance premium present in every year, or is it one regime?",
        ],
        "costs_per_leg_usd": list(COSTS_PER_LEG_USD),
        "source_table": str(args.table),
        "by_hold": results,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for hold, block in results.items():
        print(f"\n{hold} hold — each leg on its own")
        head = (
            f"  {'leg':>6} {'premium':>9} {'gross':>8} {'as %':>7} "
            f"{'95% CI':>22} {'net @$25':>9} {'net @$3':>8}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        allslots = block["legs"][0]
        for side, row in allslots["legs"].items():
            lo, hi = row["gross_ci95_usd"]
            print(
                f"  {side:>6} {row['mean_premium_usd']:>9,.0f} {row['mean_gross_usd']:>8,.1f} "
                f"{100 * row['gross_share_of_premium']:>6.2f}% "
                f"{f'[{lo:,.1f}, {hi:,.1f}]':>22} {row['net_at_25_usd']:>9,.1f} "
                f"{row['net_at_3_08_usd']:>8,.1f}"
            )
        cmp = allslots["call_minus_put"]
        lo, hi = cmp["ci95_usd"]
        print(
            f"  call minus put: {cmp['mean_usd']:+,.1f} CI [{lo:,.1f}, {hi:,.1f}] "
            f"{'DISTINGUISHABLE' if cmp['distinguishable_from_zero'] else 'not distinguishable'}"
        )

        print(f"\n{hold} hold — is the premium present every year?")
        head = (
            f"  {'year':>6} {'slots':>7} {'|move|':>7} {'straddle':>9} {'gross':>8} "
            f"{'as %':>7} {'95% CI':>24} {'negative?':>10}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for row in block["by_year"]:
            lo, hi = row["gross_ci95_usd"]
            print(
                f"  {row['year']:>6} {row['slots']:>7,} {row['mean_abs_move_points']:>7.2f} "
                f"{row['mean_straddle_premium_usd']:>9,.0f} "
                f"{row['mean_straddle_gross_usd']:>8,.1f} "
                f"{100 * row['gross_share_of_premium']:>6.2f}% "
                f"{f'[{lo:,.1f}, {hi:,.1f}]':>24} "
                f"{('yes' if row['negative_with_confidence'] else 'no'):>10}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
