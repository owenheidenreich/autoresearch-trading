"""Does an early exit actually cap the loss on a near-ATM 0DTE contract?

The proposed charter amendment rests on one empirical claim: that a ~$1,300
ticket on a $10k account is survivable because a trained exit model cuts a
loser at about -30% of premium, so the account takes a 3.9% hit rather than a
13% one.

Two things could make that false, and only measurement separates them:

1. **The stop is outrun.** A 0DTE option can fall through a level between one
   minute and the next. A stop declared at -30% then fills far below it, and the
   protection is smaller than the arithmetic assumes.
2. **The stop converts winners into losers.** Cutting at -30% also exits paths
   that would have recovered. That cost has to be counted against the benefit.

This module measures the minute-by-minute path of every eligible contract and
reports both. It fits nothing and proposes no policy: the stop level is a fixed
declared number, not a searched one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER = 100.0
ENTRY_MINUTE = "09:35"
EXIT_MINUTE = "10:35"
# Declared, not searched. -30% is the level the proposed amendment names.
STOP_LEVELS = (-0.20, -0.30, -0.40, -0.50)


def _spot(frame: pd.DataFrame) -> float | None:
    piv = frame.pivot_table(
        index="strike", columns="right", values="mid", aggfunc="last"
    ).dropna()
    if piv.empty or not {"C", "P"}.issubset(piv.columns):
        return None
    return float((piv["C"] - piv["P"]).abs().idxmin())


def session_paths(path: Path, low: float, high: float) -> pd.DataFrame | None:
    """Minute paths for contracts whose entry premium lands in the band."""

    d = pd.read_parquet(
        path, columns=["event_time", "strike", "right", "mid", "bid", "ask"]
    )
    for col in ("strike", "mid", "bid", "ask"):
        d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    d = d.dropna(subset=["strike", "mid", "bid", "ask"])
    d = d[(d["mid"] > 0) & (d["ask"] > d["bid"]) & (d["bid"] > 0)]
    if d.empty:
        return None
    d["minute"] = (
        pd.to_datetime(d["event_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    window = d[(d["minute"] >= ENTRY_MINUTE) & (d["minute"] <= EXIT_MINUTE)]
    entry = window[window["minute"] == ENTRY_MINUTE]
    exit_ = window[window["minute"] == EXIT_MINUTE]
    if entry.empty or exit_.empty:
        return None
    s0, s1 = _spot(entry), _spot(exit_)
    if s0 is None or s1 is None or s0 == s1:
        return None

    entry = entry.copy()
    entry["premium"] = entry["mid"] * CONTRACT_MULTIPLIER
    picked = entry[(entry["premium"] >= low) & (entry["premium"] <= high)]
    if picked.empty:
        return None

    keys = set(zip(picked["strike"], picked["right"]))
    paths = window[
        [(s, r) in keys for s, r in zip(window["strike"], window["right"])]
    ].copy()
    ref = picked.set_index(["strike", "right"])["mid"].to_dict()
    idx = list(zip(paths["strike"], paths["right"]))
    paths["entry_mid"] = [ref[k] for k in idx]
    paths["ratio"] = paths["mid"] / paths["entry_mid"] - 1.0

    up = bool(s1 > s0)
    rows = []
    for (strike, right), grp in paths.groupby(["strike", "right"], sort=False):
        grp = grp.sort_values("minute")
        after = grp[grp["minute"] > ENTRY_MINUTE]
        if after.empty:
            continue
        entry_mid = float(grp["entry_mid"].iloc[0])
        final = float(grp[grp["minute"] == EXIT_MINUTE]["mid"].iloc[-1]) if (
            grp["minute"] == EXIT_MINUTE
        ).any() else float(grp["mid"].iloc[-1])
        ratios = after["ratio"].to_numpy(float)
        mids = after["mid"].to_numpy(float)
        rows.append(
            {
                "session": path.name.split("_")[-1].replace(".parquet", ""),
                "premium": entry_mid * CONTRACT_MULTIPLIER,
                "correct": (right == "C") == up,
                "final_ratio": final / entry_mid - 1.0,
                "min_ratio": float(ratios.min()),
                # Fill actually available at the first minute the level is
                # breached, which is what separates a declared stop from a
                # realised one.
                **{
                    f"fill_{int(-100 * lvl)}": _first_touch_fill(ratios, mids, lvl)
                    for lvl in STOP_LEVELS
                },
                "entry_mid": entry_mid,
            }
        )
    return pd.DataFrame(rows) if rows else None


def _first_touch_fill(ratios: np.ndarray, mids: np.ndarray, level: float) -> float:
    """Ratio actually obtained when exiting at the first minute at or below ``level``.

    Returns nan when the level was never touched. The gap between ``level`` and
    this value is the slippage a declared stop does not get to assume away.
    """

    hit = np.nonzero(ratios <= level)[0]
    if hit.size == 0:
        return float("nan")
    return float(ratios[hit[0]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path.home()
        / ".autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized",
    )
    p.add_argument("--low", type=float, default=1000.0)
    p.add_argument("--high", type=float, default=1600.0)
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = sorted(
        f for f in args.corpus.glob("databento_spxw_0dte_*.parquet")
        if "official_context" not in f.name
    )
    parts = []
    for i, path in enumerate(files, 1):
        got = session_paths(path, args.low, args.high)
        if got is not None:
            parts.append(got)
        if i % 25 == 0:
            print(f"  {i}/{len(files)} sessions", flush=True)
    t = pd.concat(parts, ignore_index=True)

    wrong = t[~t["correct"]]
    stops = []
    for lvl in STOP_LEVELS:
        col = f"fill_{int(-100 * lvl)}"
        touched = t[col].notna()
        fills = t.loc[touched, col]
        # What the stop returns overall: stopped paths take their fill, the rest
        # ride to the horizon.
        outcome = np.where(touched, t[col].fillna(0.0), t["final_ratio"])
        recovered = (touched & (t["final_ratio"] > 0)).sum()
        stops.append(
            {
                "declared_stop": lvl,
                "touched_share": round(float(touched.mean()), 4),
                "mean_fill_ratio": round(float(fills.mean()), 4) if len(fills) else None,
                "slippage_vs_declared": (
                    round(float(fills.mean() - lvl), 4) if len(fills) else None
                ),
                "worst_fill_ratio": round(float(fills.min()), 4) if len(fills) else None,
                "share_of_touched_that_would_have_recovered": (
                    round(float(recovered / touched.sum()), 4) if touched.sum() else None
                ),
                "mean_outcome_with_stop": round(float(outcome.mean()), 4),
                "mean_outcome_without_stop": round(float(t["final_ratio"].mean()), 4),
                "mean_account_hit_when_wrong_pct": round(
                    float(
                        -np.minimum(
                            np.where(
                                touched[~t["correct"]],
                                t.loc[~t["correct"], col].fillna(0.0),
                                t.loc[~t["correct"], "final_ratio"],
                            ),
                            0.0,
                        ).mean()
                        * t.loc[~t["correct"], "premium"].mean()
                        / args.account
                        * 100
                    ),
                    3,
                ),
            }
        )

    payload = {
        "schema_version": "v5.stop-effectiveness.v1",
        "premium_band_usd": [args.low, args.high],
        "account_usd": args.account,
        "entry_minute": ENTRY_MINUTE,
        "exit_minute": EXIT_MINUTE,
        "sessions": int(t["session"].nunique()),
        "contracts": int(len(t)),
        "mean_premium_usd": round(float(t["premium"].mean()), 2),
        "premium_share_of_account": round(
            float(t["premium"].mean() / args.account), 4
        ),
        "no_stop": {
            "mean_final_ratio": round(float(t["final_ratio"].mean()), 4),
            "mean_final_ratio_when_wrong": round(float(wrong["final_ratio"].mean()), 4),
            "mean_account_hit_when_wrong_pct": round(
                float(
                    -wrong["final_ratio"].clip(upper=0).mean()
                    * wrong["premium"].mean()
                    / args.account
                    * 100
                ),
                3,
            ),
            "share_of_wrong_worse_than_minus_30pct": round(
                float((wrong["final_ratio"] <= -0.30).mean()), 4
            ),
        },
        "stops": stops,
        "computes_no_policy": True,
        "note": (
            "The stop level is declared, never searched. Fills are taken at the "
            "first minute the level is breached, so the gap between the declared "
            "level and the realised fill is measured rather than assumed away."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\n{payload['sessions']} sessions, {payload['contracts']:,} contracts, "
        f"mean premium ${payload['mean_premium_usd']:,.0f} "
        f"({100 * payload['premium_share_of_account']:.1f}% of account)\n"
    )
    n = payload["no_stop"]
    print(
        f"no stop: mean outcome {100 * n['mean_final_ratio']:+.1f}%, "
        f"when wrong {100 * n['mean_final_ratio_when_wrong']:+.1f}% "
        f"= {n['mean_account_hit_when_wrong_pct']:.2f}% of account\n"
    )
    head = (
        f"{'stop':>6} {'touched':>8} {'mean fill':>10} {'slippage':>9} "
        f"{'worst':>8} {'recovered':>10} {'acct hit':>9}"
    )
    print(head)
    print("-" * len(head))
    for s in stops:
        print(
            f"{100 * s['declared_stop']:>5.0f}% {100 * s['touched_share']:>7.1f}% "
            f"{100 * s['mean_fill_ratio']:>9.1f}% {100 * s['slippage_vs_declared']:>8.1f}% "
            f"{100 * s['worst_fill_ratio']:>7.1f}% "
            f"{100 * s['share_of_touched_that_would_have_recovered']:>9.1f}% "
            f"{s['mean_account_hit_when_wrong_pct']:>8.2f}%"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
