"""Teacher-miss analysis.

Reads `v3/reference/attribution_full_YYYY-MM-DD.txt` (the per-session JSON
dump), finds the oracle bar for each session, pulls that bar's features from
`data.pt`, then compares feature distributions across three outcome groups:

- abstention: oracle bar present, no teacher triggered → what does the bar
  look like that neither ORC nor FailedBreak recognized?
- side_error: oracle bar had teacher trigger but wrong direction → what
  pushed the teacher's direction choice the wrong way?
- entered_right: teacher caught the oracle bar in the right direction → the
  baseline "working" distribution for comparison.

Output: structured findings table showing median values per group and
percentage of each group where key ORC/FailedBreak gate conditions were met.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from v3.harness.v2_adapter import V2Dataset


def _latest_attribution_file() -> str:
    files = sorted(glob.glob("v3/reference/attribution_full_*.txt"))
    if not files:
        raise FileNotFoundError("No v3/reference/attribution_full_*.txt found")
    return files[-1]


def _load_per_session(path: str) -> list[dict]:
    rows = []
    in_json = False
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _feature_at_bar(dataset: V2Dataset, day: str, minute: int, feature_name: str) -> float | None:
    if feature_name not in dataset.idx:
        return None
    day_start, day_end = dataset.day_bar_range(day)
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return None
    return float(dataset.X_sim[abs_idx, dataset.idx[feature_name]])


def _close_vs_first15(dataset: V2Dataset, day: str, minute: int) -> dict[str, float | bool]:
    day_start, day_end = dataset.day_bar_range(day)
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return {}
    close = float(dataset.spot_prices[abs_idx])
    h, l = dataset.first15_by_day.get(day, (0.0, 0.0))
    return {
        "close": close,
        "f15_high": h,
        "f15_low": l,
        "above_f15": close > h if h > 0 else False,
        "below_f15": close < l if l > 0 else False,
        "inside_f15": (l <= close <= h) if h > 0 else False,
    }


def main(attribution_path: str | None = None) -> int:
    if attribution_path is None:
        attribution_path = _latest_attribution_file()
    print(f"Reading {attribution_path}")
    sessions = _load_per_session(attribution_path)
    print(f"Loaded {len(sessions)} per-session rows")

    by_outcome: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        by_outcome[s.get("outcome", "unknown")].append(s)
    print(f"Outcomes: { {k: len(v) for k, v in by_outcome.items()} }")
    print()

    print("Loading dataset...")
    ds = V2Dataset.load()

    # Features to pull at each oracle bar
    feature_names = [
        "vwap_dist",
        "vwap_slope",
        "volume_ratio",
        "first15_range_pct",
        "first15_close_position",
        "first15_acceptance",
        "bars_since_break_above_first15",
        "bars_since_break_below_first15",
        "breakout_confirmation",
        "vix_regime",
        "atm_iv",
    ]

    # Collect per-outcome feature samples
    by_outcome_features: dict[str, dict[str, list[float]]] = {
        outcome: {fname: [] for fname in feature_names + ["above_f15", "below_f15", "inside_f15", "oracle_minute"]}
        for outcome in by_outcome.keys()
    }

    for outcome, rows in by_outcome.items():
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            for fname in feature_names:
                v = _feature_at_bar(ds, day, minute, fname)
                if v is not None:
                    by_outcome_features[outcome][fname].append(v)
            pos = _close_vs_first15(ds, day, minute)
            for k in ("above_f15", "below_f15", "inside_f15"):
                if k in pos:
                    by_outcome_features[outcome][k].append(1.0 if pos[k] else 0.0)
            by_outcome_features[outcome]["oracle_minute"].append(float(minute))

    # --- Print summary ---
    print("Outcome counts: " + ", ".join(
        f"{k}={len(v)}" for k, v in by_outcome.items()
    ))
    print()

    OUTCOMES_TO_SHOW = ["entered_right", "abstention", "side_error"]
    header_cols = OUTCOMES_TO_SHOW
    print(f"=== Median of oracle-bar features by outcome ===")
    print(f"{'feature':<36}" + "".join(f"{c:>18}" for c in header_cols))
    for fname in feature_names + ["above_f15", "below_f15", "inside_f15", "oracle_minute"]:
        row = f"{fname:<36}"
        for outcome in header_cols:
            vals = by_outcome_features.get(outcome, {}).get(fname, [])
            if not vals:
                row += f"{'---':>18}"
            else:
                median = float(np.median(vals))
                row += f"{median:>18.3f}"
        print(row)
    print()

    # --- Specific diagnostic: on abstention bars, did the ORC entry condition hold? ---
    print("=== ORC entry conditions at oracle bar, by outcome ===")
    print("ORC BUY_CALL requires: close > first15_high AND close > vwap AND vwap_slope > 0")
    print("ORC BUY_PUT  requires: close < first15_low  AND close < vwap AND vwap_slope < 0")
    print()
    print(f"{'outcome':<20}{'n':>6}{'%orc_call_gate_met':>22}{'%orc_put_gate_met':>22}{'%neither':>12}")
    for outcome in OUTCOMES_TO_SHOW:
        rows = by_outcome.get(outcome, [])
        if not rows:
            continue
        n_call_gate = 0
        n_put_gate = 0
        n_neither = 0
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            pos = _close_vs_first15(ds, day, minute)
            vwap_dist = _feature_at_bar(ds, day, minute, "vwap_dist") or 0.0
            vwap_slope = _feature_at_bar(ds, day, minute, "vwap_slope") or 0.0
            call_gate = pos.get("above_f15", False) and vwap_dist > 0 and vwap_slope > 0
            put_gate = pos.get("below_f15", False) and vwap_dist < 0 and vwap_slope < 0
            if call_gate:
                n_call_gate += 1
            if put_gate:
                n_put_gate += 1
            if not call_gate and not put_gate:
                n_neither += 1
        print(
            f"{outcome:<20}{len(rows):>6}"
            f"{100*n_call_gate/len(rows):>21.1f}%"
            f"{100*n_put_gate/len(rows):>21.1f}%"
            f"{100*n_neither/len(rows):>11.1f}%"
        )
    print()

    # --- Specific diagnostic: FailedBreak condition ---
    print("=== FailedBreak entry conditions at oracle bar, by outcome ===")
    print("FB triggers when: bar inside range AND had break in last 5 bars (contested=reject)")
    print()
    print(f"{'outcome':<20}{'n':>6}{'%fb_call_gate':>16}{'%fb_put_gate':>16}{'%contested':>12}{'%neither':>12}")
    for outcome in OUTCOMES_TO_SHOW:
        rows = by_outcome.get(outcome, [])
        if not rows:
            continue
        n_fb_call = 0
        n_fb_put = 0
        n_contested = 0
        n_neither = 0
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            pos = _close_vs_first15(ds, day, minute)
            bars_above = _feature_at_bar(ds, day, minute, "bars_since_break_above_first15") or -1
            bars_below = _feature_at_bar(ds, day, minute, "bars_since_break_below_first15") or -1
            inside = pos.get("inside_f15", False)
            had_above = 1 <= bars_above <= 5
            had_below = 1 <= bars_below <= 5
            if inside and had_above and had_below:
                n_contested += 1
            elif inside and had_above:
                n_fb_put += 1  # failed upside break → reversal = put
            elif inside and had_below:
                n_fb_call += 1  # failed downside break → reversal = call
            else:
                n_neither += 1
        print(
            f"{outcome:<20}{len(rows):>6}"
            f"{100*n_fb_call/len(rows):>15.1f}%"
            f"{100*n_fb_put/len(rows):>15.1f}%"
            f"{100*n_contested/len(rows):>11.1f}%"
            f"{100*n_neither/len(rows):>11.1f}%"
        )
    print()

    # --- Either-teacher-fires sanity check ---
    print("=== Share where ANY teacher's gate is met at oracle bar ===")
    for outcome in OUTCOMES_TO_SHOW:
        rows = by_outcome.get(outcome, [])
        if not rows:
            continue
        n_any = 0
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            pos = _close_vs_first15(ds, day, minute)
            vwap_dist = _feature_at_bar(ds, day, minute, "vwap_dist") or 0.0
            vwap_slope = _feature_at_bar(ds, day, minute, "vwap_slope") or 0.0
            bars_above = _feature_at_bar(ds, day, minute, "bars_since_break_above_first15") or -1
            bars_below = _feature_at_bar(ds, day, minute, "bars_since_break_below_first15") or -1
            inside = pos.get("inside_f15", False)
            orc_call = pos.get("above_f15", False) and vwap_dist > 0 and vwap_slope > 0
            orc_put = pos.get("below_f15", False) and vwap_dist < 0 and vwap_slope < 0
            fb_active = inside and (
                (1 <= bars_above <= 5 and not (1 <= bars_below <= 5))
                or (1 <= bars_below <= 5 and not (1 <= bars_above <= 5))
            )
            any_gate = orc_call or orc_put or fb_active
            if any_gate:
                n_any += 1
        print(f"  {outcome:<20}: {n_any}/{len(rows)}  ({100*n_any/len(rows):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
