"""Does the hour of the day change what a 0DTE trade is worth?

Every measurement this project has made about the option layer starts at 09:35.
That is one slice of a session in an instrument whose defining property is that
its clock runs out today: at 09:35 a contract has six and a half hours of life
and at 15:00 it has one. Gamma, theta and the spread all move through the day,
and none of it has ever been looked at.

This module measures the conditional payoff by **entry hour**, and at three cost
assumptions rather than one. The cost assumption turned out to matter as much as
anything else the project has measured, so it is a reported axis here rather
than a frozen constant:

* **$25** — crossing the full spread both ways, the aggressive round trip
  measured on the owned quote corpus. Every prior number uses this.
* **$14** — crossing half the spread, roughly what a marketable limit at the
  midpoint gets when it fills.
* **$3.08** — fees only, the measured IBKR paper fill. The floor, achievable
  only by resting passive and accepting that some orders do not fill.

Fits nothing, searches nothing, proposes no policy. Entry hours, holds, band and
the three costs are all declared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.measure_hold_occupancy import (
    FIRST_INDEX,
    LAST_INDEX,
    _index,
    _label,
    session_pivot,
)
from v5.ops.resolve_exit_price_convention import (
    CONTRACT_MULTIPLIER,
    NEAR_ATM_POINTS,
    TRADE_CORPUS,
)

HOLDS_MINUTES = (15, 60)

# Fees only, half the spread, and the full aggressive round trip.
COSTS_USD = (3.08, 14.0, 25.0)


def session_rows(path: Path, hold: int) -> pd.DataFrame | None:
    """One row per slot and side, tagged with the hour it was entered."""

    got = session_pivot(path)
    if got is None:
        return None
    pivot, spot = got
    filled = pivot.ffill()
    strikes = pivot.columns.get_level_values("strike").to_numpy(float)
    is_call = pivot.columns.get_level_values("right").to_numpy() == "C"

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
        moneyness = np.where(is_call, s0 - strikes, strikes - s0)
        eligible = (
            np.isfinite(entry_price)
            & np.isfinite(exit_price)
            & (np.abs(moneyness) <= NEAR_ATM_POINTS)
        )
        if not eligible.any():
            continue
        up = bool(s1 > s0)
        for side in (True, False):
            on_side = np.flatnonzero(eligible & (is_call == side))
            if not on_side.size:
                continue
            i = int(on_side[np.argmin(np.abs(moneyness[on_side]))])
            rows.append(
                {
                    "session": path.name[:10],
                    "entry_minute": entry_label,
                    "hour": entry_label[:2],
                    "minutes_to_close": LAST_INDEX - _index(entry_label),
                    "premium": entry_price[i] * CONTRACT_MULTIPLIER,
                    "gross_usd": (exit_price[i] - entry_price[i]) * CONTRACT_MULTIPLIER,
                    "correct": bool(is_call[i] == up),
                }
            )
    return pd.DataFrame(rows) if rows else None


def assess(part: pd.DataFrame) -> dict:
    """Break-even at each declared cost, plus the straddle the pair implies."""

    gross = part["gross_usd"].to_numpy(float)
    ok = part["correct"].to_numpy(bool)
    if ok.sum() < 50 or (~ok).sum() < 50:
        return {}
    right, wrong = float(gross[ok].mean()), float(-gross[~ok].mean())
    out = {
        "trades": int(len(part)),
        "sessions": int(part["session"].nunique()),
        "mean_premium_usd": round(float(part["premium"].mean()), 2),
        "mean_gross_when_correct_usd": round(right, 2),
        "mean_gross_when_wrong_usd": round(-wrong, 2),
        # Buying both sides: positive means realised movement outran what the
        # options charged for it, over this corpus and this hour. It is a
        # statement about the volatility regime, not about any signal.
        "both_sides_gross_usd": round(right - wrong, 2),
        "breakeven_by_cost": {},
    }
    for cost in COSTS_USD:
        win, loss = right - cost, wrong + cost
        out["breakeven_by_cost"][f"${cost}"] = (
            round(loss / (win + loss), 6) if win > 0 else None
        )
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
        parts = []
        for i, path in enumerate(files, 1):
            got = session_rows(path, hold)
            if got is not None:
                parts.append(got)
            if i % 200 == 0:
                print(f"  {hold}m: {i}/{len(files)}", flush=True)
        if not parts:
            continue
        table = pd.concat(parts, ignore_index=True)
        results[f"{hold}m"] = {
            "pooled": assess(table),
            "by_hour": {
                hour: row
                for hour, part in table.groupby("hour", sort=True)
                if (row := assess(part))
            },
        }

    payload = {
        "schema_version": "v5.time-of-day.v1",
        "question": (
            "Every option-layer measurement so far enters at 09:35. Does the "
            "hour change the conditional payoff, and how much does the cost "
            "assumption change the break-even?"
        ),
        "costs_usd": list(COSTS_USD),
        "cost_meanings": {
            "3.08": "fees only, measured IBKR paper fill; requires resting passive",
            "14.0": "roughly half the measured near-ATM spread, plus fees",
            "25.0": "full aggressive round trip; what every prior number charges",
        },
        "moneyness_band_points": NEAR_ATM_POINTS,
        "selection_rule": "nearest contract to the money on each side, entry-minute information only",
        "corpus": str(args.corpus),
        "by_hold": results,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for hold, block in results.items():
        print(f"\n{hold} hold, near ATM, break-even accuracy by entry hour")
        head = (
            f"  {'hour':>5} {'trades':>8} {'premium':>9} {'both sides':>11} "
            f"{'@$25':>8} {'@$14':>8} {'@$3.08':>8}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for hour, row in block["by_hour"].items():
            be = row["breakeven_by_cost"]
            cells = " ".join(
                f"{(f'{100 * be[k]:.2f}%' if be[k] else 'n/a'):>8}"
                for k in ("$25.0", "$14.0", "$3.08")
            )
            print(
                f"  {hour + ':00':>5} {row['trades']:>8,} {row['mean_premium_usd']:>9,.0f} "
                f"{row['both_sides_gross_usd']:>+11,.0f} {cells}"
            )
        pooled = block["pooled"]["breakeven_by_cost"]
        print(
            f"  {'ALL':>5} {block['pooled']['trades']:>8,} "
            f"{block['pooled']['mean_premium_usd']:>9,.0f} "
            f"{block['pooled']['both_sides_gross_usd']:>+11,.0f} "
            + " ".join(
                f"{(f'{100 * pooled[k]:.2f}%' if pooled[k] else 'n/a'):>8}"
                for k in ("$25.0", "$14.0", "$3.08")
            )
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
