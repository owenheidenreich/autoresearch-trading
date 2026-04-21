"""Time-of-day cap investigation for the NarrowRangeBreakout (NR10) teacher.

Simulates the NR10 trigger across every eligible bar in minutes [30, 240]
(10:00 AM - 13:30 ET) for the full 986-day cache. For each 30-minute bucket,
reports:

- Trigger frequency (how often the NR10 conditions fire)
- Forward 20-bar move distribution (signal quality: mean/median MFE in bps)
- Fake-out rate (% of triggers where price retraces inside the pre-break
  10-bar range within the next 10 bars)

Questions answered:

1. Is the NR10 trigger frequency and quality stable across the 10:00-13:30
   window, or does it degrade materially after lunch?

2. Does the fake-out rate jump during midday chop (12:00-13:00)?

3. Based on forward move × fake-out tradeoff, what's the right time cap
   for the teacher?

Minutes convention: minute-of-session since 09:30 ET. 30 = 10:00 AM.
"""
from __future__ import annotations

import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from v3.harness.v2_adapter import V2Dataset


TIME_BUCKETS = [
    ("10:00-10:30", 30, 60),
    ("10:30-11:00", 60, 90),
    ("11:00-11:30", 90, 120),
    ("11:30-12:00", 120, 150),
    ("12:00-12:30", 150, 180),
    ("12:30-13:00", 180, 210),
    ("13:00-13:30", 210, 240),
]


def _build_omar_levels(spx_1min_path: str) -> dict[str, dict[str, float]]:
    df = pickle.load(open(spx_1min_path, "rb"))
    out: dict[str, dict[str, float]] = {}
    for day, group in df.groupby("date"):
        g = group.reset_index(drop=True)
        if len(g) == 0:
            continue
        h = float(g.iloc[0]["spx_high"])
        l = float(g.iloc[0]["spx_low"])
        out[str(day)] = {
            "high": h,
            "low": l,
            "mid": (h + l) / 2.0,
            "range": max(h - l, 0.01),
        }
    return out


def _near_omar(close: float, omar: dict[str, float], threshold_units: float = 0.5) -> bool:
    r = omar["range"]
    levels = [omar["high"], omar["low"], omar["mid"]]
    return any(abs(close - lvl) / r <= threshold_units for lvl in levels)


def _simulate_nr10(
    ds: V2Dataset, omar_map: dict[str, dict[str, float]]
) -> list[dict]:
    results: list[dict] = []
    all_days = sorted(set(ds.dates))
    for day in all_days:
        omar = omar_map.get(day)
        if omar is None:
            continue
        try:
            day_start, day_end = ds.day_bar_range(day)
        except ValueError:
            continue
        day_closes = np.asarray(
            ds.spot_prices[day_start:day_end], dtype=float
        )
        f15_h, f15_l = ds.first15_by_day.get(day, (0.0, 0.0))
        if f15_h <= 0 or f15_l <= 0:
            continue
        omar_r = omar["range"]
        n_bars = len(day_closes)
        last_minute = min(240, n_bars - 20)  # need 20 forward bars
        for minute in range(30, last_minute):
            close_T = float(day_closes[minute])
            # 1. Inside first15 range
            if not (f15_l <= close_T <= f15_h):
                continue
            # 2. Pre-entry 10-bar range
            pre_window = day_closes[minute - 10 : minute]
            if len(pre_window) < 10:
                continue
            pre_max = float(pre_window.max())
            pre_min = float(pre_window.min())
            pre_range = pre_max - pre_min
            if pre_range > omar_r:
                continue
            # 3. Breakout trigger
            direction = None
            if close_T > pre_max:
                direction = "call"
            elif close_T < pre_min:
                direction = "put"
            else:
                continue
            # Forward 20-bar window
            fwd = day_closes[minute + 1 : minute + 21]
            if len(fwd) < 5:
                continue
            # Fake-out check (did price close back inside pre_range within 10 bars)
            fake_horizon = min(10, len(fwd))
            fake_out = bool(
                np.any((fwd[:fake_horizon] >= pre_min) & (fwd[:fake_horizon] <= pre_max))
            )
            # Forward moves (direction-adjusted, % of close)
            if close_T <= 0:
                continue
            if direction == "call":
                mfe_pct = float((fwd.max() - close_T) / close_T)
                mae_pct = float((fwd.min() - close_T) / close_T)
                final_pct = float((fwd[-1] - close_T) / close_T)
            else:
                mfe_pct = float((close_T - fwd.min()) / close_T)
                mae_pct = float((close_T - fwd.max()) / close_T)
                final_pct = float((close_T - fwd[-1]) / close_T)
            results.append(
                {
                    "day": day,
                    "minute": minute,
                    "direction": direction,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "final_pct": final_pct,
                    "fake_out": fake_out,
                    "near_omar": _near_omar(close_T, omar, 0.5),
                }
            )
    return results


def _summarize_bucket(
    results: list[dict], lo: int, hi: int, n_days: int
) -> dict:
    bucket = [r for r in results if lo <= r["minute"] < hi]
    if not bucket:
        return {"n": 0}
    n = len(bucket)
    mfe = np.asarray([r["mfe_pct"] for r in bucket])
    final = np.asarray([r["final_pct"] for r in bucket])
    fake_rate = float(np.mean([r["fake_out"] for r in bucket]))
    call_share = float(np.mean([r["direction"] == "call" for r in bucket]))
    triggers_per_day = n / n_days
    return {
        "n": n,
        "triggers_per_day": triggers_per_day,
        "mfe_mean_bps": float(mfe.mean()) * 10000,
        "mfe_median_bps": float(np.median(mfe)) * 10000,
        "mfe_p75_bps": float(np.percentile(mfe, 75)) * 10000,
        "final_mean_bps": float(final.mean()) * 10000,
        "final_median_bps": float(np.median(final)) * 10000,
        "fake_out_rate": fake_rate,
        "call_share": call_share,
    }


def main() -> int:
    print("Loading dataset and OMAR levels...")
    ds = V2Dataset.load()
    omar_map = _build_omar_levels(
        str(Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl")
    )
    n_days = len(set(ds.dates))
    print(f"  {n_days} days, {len(omar_map)} days with OMAR levels")

    print()
    print("Simulating NR10 breakout trigger across all eligible bars 10:00-13:30...")
    results = _simulate_nr10(ds, omar_map)
    print(f"  {len(results)} total triggers across full cache")

    print()
    print("=== NR10 trigger quality by 30-minute bucket ===")
    print()
    header = (
        f"{'bucket':<14}{'n':>8}{'trig/day':>10}"
        f"{'mfe_med_bps':>13}{'mfe_p75_bps':>13}"
        f"{'final_med_bps':>15}"
        f"{'fake_rate':>11}{'call_share':>12}"
    )
    print(header)
    print("-" * len(header))
    for label, lo, hi in TIME_BUCKETS:
        s = _summarize_bucket(results, lo, hi, n_days)
        if s.get("n", 0) == 0:
            print(f"{label:<14}{0:>8}  ---")
            continue
        print(
            f"{label:<14}{s['n']:>8}{s['triggers_per_day']:>10.2f}"
            f"{s['mfe_median_bps']:>13.1f}{s['mfe_p75_bps']:>13.1f}"
            f"{s['final_median_bps']:>15.1f}"
            f"{s['fake_out_rate']:>10.1%}"
            f"{s['call_share']:>11.1%}"
        )

    # Final-vs-MFE read: positive final means the move held through to bar+20
    # If final << mfe, it means the move peaked and retraced (catch-peak scenario)
    print()
    print("Reading key:")
    print("  mfe_med_bps: median max-favorable move in the break direction, in bps (%*100)")
    print("    e.g. 20 bps = 0.2% favorable excursion")
    print("  final_med_bps: median close-to-close move at bar+20 in the break direction")
    print("    if final << mfe, the move peaked and retraced before 20min")
    print("  fake_out_rate: % of triggers where price closed back inside pre-range within 10 bars")
    print("    high fake-out = trigger fires on noise; low fake-out = clean breakouts")

    # Second pass: same buckets, but filter to triggers that are ALSO near OMAR (<=0.5 OMAR)
    near_results = [r for r in results if r["near_omar"]]
    far_results = [r for r in results if not r["near_omar"]]
    print()
    print(f"=== NR10 + near OMAR filter (|dist| <= 0.5 OMAR from H/L/M) ===")
    print(f"    near-OMAR trigger count: {len(near_results):,}  ({100*len(near_results)/max(len(results),1):.1f}% of all NR10)")
    print()
    print(header)
    print("-" * len(header))
    for label, lo, hi in TIME_BUCKETS:
        s = _summarize_bucket(near_results, lo, hi, n_days)
        if s.get("n", 0) == 0:
            print(f"{label:<14}{0:>8}  ---")
            continue
        print(
            f"{label:<14}{s['n']:>8}{s['triggers_per_day']:>10.2f}"
            f"{s['mfe_median_bps']:>13.1f}{s['mfe_p75_bps']:>13.1f}"
            f"{s['final_median_bps']:>15.1f}"
            f"{s['fake_out_rate']:>10.1%}"
            f"{s['call_share']:>11.1%}"
        )

    print()
    print(f"=== NR10 + NOT near OMAR (|dist| > 0.5 OMAR) — as a control ===")
    print()
    print(header)
    print("-" * len(header))
    for label, lo, hi in TIME_BUCKETS:
        s = _summarize_bucket(far_results, lo, hi, n_days)
        if s.get("n", 0) == 0:
            print(f"{label:<14}{0:>8}  ---")
            continue
        print(
            f"{label:<14}{s['n']:>8}{s['triggers_per_day']:>10.2f}"
            f"{s['mfe_median_bps']:>13.1f}{s['mfe_p75_bps']:>13.1f}"
            f"{s['final_median_bps']:>15.1f}"
            f"{s['fake_out_rate']:>10.1%}"
            f"{s['call_share']:>11.1%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
