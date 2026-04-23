"""S/R confluence investigation — do abstention oracle bars cluster near
pre-defined support/resistance levels more than random bars do?

Pickles' framework uses a confluence of multiple level types. We test the
computable subset:

- Prior-day SPX high / low / close / mid
- Round numbers (multiples of 25 and 50 near current price)
- Weekly pivots: P = (H+L+C)/3, R1 = 2P−L, S1 = 2P−H, R2 = P+(H−L), S2 = P−(H−L)
- Fibonacci 38.2% / 50% / 61.8% of prior session H/L range
- Initial Balance (IB) high / low — first 30 min of the session

Not tested (data not available or complex):
- Overnight H/L (no pre-market data)
- Volume profile POC/VAH/VAL
- Pickles' own published R1/R2/R3 (proprietary)
- Multi-year macro static levels

For each cohort (entered_right / abstention / random control):
- Distance (in OMAR-range units) to the nearest level
- % within 0.5× OMAR of ANY level
- Confluence count (# of levels within 0.5× OMAR of close)

If abstention oracle bars have materially higher confluence than controls,
S/R confluence is a real filter. If not, skip this layer in the teacher.
"""
from __future__ import annotations

import glob
import json
import pickle
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from v3.harness.v2_adapter import V2Dataset


def _latest_attribution_file() -> str:
    files = sorted(glob.glob("v3/reference/attribution_full_*.txt"))
    if not files:
        raise FileNotFoundError
    return files[-1]


def _load_per_session(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _build_daily_levels(spx_1min_path: str) -> dict[str, dict[str, float]]:
    """Per day, return prior-day H/L/C/M, weekly pivots, Fibonacci, IB H/L, OMAR.

    Prior-day refers to the previous session in the data. Weekly pivots use
    the prior calendar week's H/L/C. Fibonacci references prior-session H/L
    range. IB uses first 30 min of the current session.
    """
    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)

    # Aggregate per-day H/L/C and first-30-min H/L
    daily: dict[str, dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        h = float(g["spx_high"].max())
        l = float(g["spx_low"].min())
        c = float(g.iloc[-1]["spx_close"])
        omar_h = float(g.iloc[0]["spx_high"])
        omar_l = float(g.iloc[0]["spx_low"])
        ib = g.head(30)
        ib_h = float(ib["spx_high"].max()) if len(ib) > 0 else h
        ib_l = float(ib["spx_low"].min()) if len(ib) > 0 else l
        daily[day] = {
            "high": h,
            "low": l,
            "close": c,
            "mid": (h + l) / 2.0,
            "ib_high": ib_h,
            "ib_low": ib_l,
            "omar_high": omar_h,
            "omar_low": omar_l,
            "omar_mid": (omar_h + omar_l) / 2.0,
            "omar_range": max(omar_h - omar_l, 0.01),
        }
    return daily


def _compute_levels_for_day(
    day: str, daily: dict[str, dict[str, float]], sorted_days: list[str]
) -> dict[str, float]:
    """Return a flat dict of level_name -> price for this day.

    Includes prior-day levels, weekly-pivot levels (using prior calendar week),
    Fibonacci retracements, and IB levels. OMAR is stored separately.
    """
    day_info = daily.get(day)
    if day_info is None:
        return {}
    levels: dict[str, float] = {}
    idx = sorted_days.index(day) if day in sorted_days else -1

    if idx >= 1:
        prev = daily.get(sorted_days[idx - 1])
        if prev:
            levels["PD_high"] = prev["high"]
            levels["PD_low"] = prev["low"]
            levels["PD_close"] = prev["close"]
            levels["PD_mid"] = prev["mid"]
            # Fibonacci of prior-day range
            rng = prev["high"] - prev["low"]
            if rng > 0:
                levels["fib_38"] = prev["low"] + 0.382 * rng
                levels["fib_50"] = prev["low"] + 0.500 * rng
                levels["fib_62"] = prev["low"] + 0.618 * rng

    # Weekly pivots (use last 5 trading days ending 1 day before `day`)
    if idx >= 5:
        prev_week = [daily[d] for d in sorted_days[max(0, idx - 5): idx] if d in daily]
        if prev_week:
            h = max(d["high"] for d in prev_week)
            l = min(d["low"] for d in prev_week)
            c = prev_week[-1]["close"]
            p = (h + l + c) / 3.0
            levels["WPP"] = p
            levels["WR1"] = 2 * p - l
            levels["WS1"] = 2 * p - h
            levels["WR2"] = p + (h - l)
            levels["WS2"] = p - (h - l)

    # IB (initial balance) H/L — first 30 min of CURRENT day
    levels["IB_high"] = day_info["ib_high"]
    levels["IB_low"] = day_info["ib_low"]

    # Round numbers near current close (use OMAR mid as anchor)
    anchor = day_info["omar_mid"]
    for n in (25, 50, 100):
        for delta in (-2, -1, 0, 1, 2):
            levels[f"round_{n}_{delta:+d}"] = round(anchor / n) * n + delta * n

    return levels


def _nearest_level_distance(
    spx_close: float, levels: dict[str, float], omar_range: float
) -> float | None:
    """Return min |dist to any level| in OMAR-range units. None if empty."""
    if not levels or omar_range <= 0:
        return None
    diffs = [abs(spx_close - lv) / omar_range for lv in levels.values()]
    return min(diffs) if diffs else None


def _confluence_count(
    spx_close: float,
    levels: dict[str, float],
    omar_range: float,
    threshold: float = 0.5,
) -> int:
    """Count how many levels are within `threshold` × OMAR-range of close."""
    if not levels or omar_range <= 0:
        return 0
    return sum(1 for lv in levels.values() if abs(spx_close - lv) / omar_range <= threshold)


def main() -> int:
    print("Loading dataset and daily levels...")
    ds = V2Dataset.load()
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"
    daily = _build_daily_levels(str(spx_path))
    sorted_days = sorted(daily.keys())
    print(f"  built daily levels for {len(daily)} days")

    sessions = _load_per_session(_latest_attribution_file())

    cohorts = {
        "entered_right": [s for s in sessions if s.get("outcome") == "entered_right"],
        "abstention": [s for s in sessions if s.get("outcome") == "abstention"],
        "side_error": [s for s in sessions if s.get("outcome") == "side_error"],
    }

    # Random control: one bar per day in the teacher window
    rng = random.Random(123)
    control: list[dict] = []
    for day in sorted_days:
        control.append({"day": day, "oracle_bar": rng.randint(40, 120)})
    cohorts["control"] = control

    # Compute metrics per cohort
    print()
    print("=== Distance-to-nearest-level + confluence count ===")
    print(
        "All levels: PD(H/L/C/M), weekly-pivot (WPP, WR1/2, WS1/2), "
        "Fib (38/50/62), IB(H/L), round-numbers ±2 × (25, 50, 100)"
    )
    print()
    print(
        f"{'cohort':<18}{'n':>6}{'p25_dist':>12}{'med_dist':>12}"
        f"{'p75_dist':>12}{'pct_within_0.5':>16}{'med_conf':>10}{'p75_conf':>10}{'p90_conf':>10}"
    )
    print("-" * 106)

    summary: dict[str, dict] = {}
    for cohort_name, rows in cohorts.items():
        distances: list[float] = []
        confluences: list[int] = []
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            day_info = daily.get(day)
            if day_info is None:
                continue
            try:
                day_start, day_end = ds.day_bar_range(day)
            except ValueError:
                continue
            abs_idx = day_start + minute
            if abs_idx >= day_end:
                continue
            close = float(ds.spot_prices[abs_idx])
            levels = _compute_levels_for_day(day, daily, sorted_days)
            if not levels:
                continue
            d = _nearest_level_distance(close, levels, day_info["omar_range"])
            if d is None:
                continue
            distances.append(d)
            c = _confluence_count(close, levels, day_info["omar_range"], threshold=0.5)
            confluences.append(c)

        if not distances:
            print(f"{cohort_name:<18}{0:>6}")
            continue
        d_arr = np.asarray(distances)
        c_arr = np.asarray(confluences)
        pct_within_half = 100 * np.mean(d_arr <= 0.5)
        summary[cohort_name] = {
            "n": len(d_arr),
            "med_dist": float(np.median(d_arr)),
            "pct_within_0.5": pct_within_half,
            "med_conf": float(np.median(c_arr)),
            "p75_conf": float(np.percentile(c_arr, 75)),
            "p90_conf": float(np.percentile(c_arr, 90)),
        }
        print(
            f"{cohort_name:<18}{len(d_arr):>6}"
            f"{np.percentile(d_arr, 25):>+12.3f}"
            f"{np.percentile(d_arr, 50):>+12.3f}"
            f"{np.percentile(d_arr, 75):>+12.3f}"
            f"{pct_within_half:>15.1f}%"
            f"{float(np.median(c_arr)):>10.1f}"
            f"{float(np.percentile(c_arr, 75)):>10.1f}"
            f"{float(np.percentile(c_arr, 90)):>10.1f}"
        )

    # Enrichment check
    print()
    print("=== Enrichment ratios (cohort / control) ===")
    ctrl = summary.get("control")
    if ctrl:
        for cohort_name in ("abstention", "entered_right", "side_error"):
            s = summary.get(cohort_name)
            if not s:
                continue
            pct_ratio = s["pct_within_0.5"] / max(ctrl["pct_within_0.5"], 1e-9)
            conf_ratio = s["p75_conf"] / max(ctrl["p75_conf"], 1e-9)
            print(
                f"  {cohort_name:<18}: pct_within_0.5 enrichment = {pct_ratio:.2f}×  "
                f"p75_confluence ratio = {conf_ratio:.2f}×"
            )

    # Per-level-type: which level types are oracle bars actually near?
    print()
    print("=== Which level TYPES do abstention oracle bars cluster near? ===")
    level_families = {
        "prior_day": ["PD_high", "PD_low", "PD_close", "PD_mid"],
        "fibonacci": ["fib_38", "fib_50", "fib_62"],
        "weekly_pivot": ["WPP", "WR1", "WR2", "WS1", "WS2"],
        "initial_balance": ["IB_high", "IB_low"],
        "round_number": None,  # handled specially
    }

    for cohort_name in ("abstention", "control"):
        rows = cohorts[cohort_name]
        fam_hits: dict[str, int] = defaultdict(int)
        fam_total = 0
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            day_info = daily.get(day)
            if day_info is None:
                continue
            try:
                day_start, day_end = ds.day_bar_range(day)
            except ValueError:
                continue
            abs_idx = day_start + minute
            if abs_idx >= day_end:
                continue
            close = float(ds.spot_prices[abs_idx])
            levels = _compute_levels_for_day(day, daily, sorted_days)
            if not levels:
                continue
            r = day_info["omar_range"]
            fam_total += 1
            for fam_name, keys in level_families.items():
                if keys is None:
                    near = any(
                        abs(close - lv) / r <= 0.5
                        for k, lv in levels.items()
                        if k.startswith("round_")
                    )
                else:
                    near = any(
                        abs(close - levels[k]) / r <= 0.5 for k in keys if k in levels
                    )
                if near:
                    fam_hits[fam_name] += 1
        print(f"  [{cohort_name}] n={fam_total}")
        for fam_name in level_families:
            pct = 100 * fam_hits.get(fam_name, 0) / max(fam_total, 1)
            print(f"    % near {fam_name:<16}: {pct:>5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
