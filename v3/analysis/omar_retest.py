"""Deep-dive: OMAR as a RETEST LEVEL (per Pickles' actual usage).

Pickles' claim: when SPX looks overextended in one direction, it often
retests OMAR (the first-minute high/low/mid). The big directional move of
the day frequently starts AFTER such a retest.

Questions this script answers empirically:

1. At the oracle bar, is SPX typically near an OMAR level? Measure distance
   from close to OMAR_high, OMAR_low, OMAR_mid in OMAR-range units. A
   "retest" means distance ≤ ~0.5× OMAR_range.

2. Do oracle bars cluster NEAR OMAR more than random non-oracle bars do?
   This is the discriminative test: if the answer is "no," then OMAR
   proximity isn't giving us useful signal; it's just a random bar feature.

3. Split oracle bars by cohort (abstention vs entered_right). Does being
   near OMAR differentiate which cohort a bar falls into?

4. Forward-looking test: on bars where SPX is near OMAR, is the next 20
   bars' move bigger than on bars far from OMAR? This tests Pickles'
   "retest predicts a big move" intuition directly.
"""
from __future__ import annotations

import glob
import json
import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np

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


def _build_omar_levels(spx_1min_path: str) -> dict[str, dict[str, float]]:
    """Per day: OMAR_high, OMAR_low, OMAR_mid, OMAR_range."""
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


def _distance_to_omar(
    ds: V2Dataset, day: str, minute: int, omar: dict[str, float]
) -> dict[str, float] | None:
    """Distance from bar close to OMAR levels, in OMAR-range units.

    Positive values = above the level; negative = below. `nearest_level`
    returns the absolute distance in OMAR units to the closest of H/L/M.
    """
    try:
        day_start, day_end = ds.day_bar_range(day)
    except ValueError:
        return None
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return None
    close = float(ds.spot_prices[abs_idx])
    r = omar["range"]
    return {
        "close": close,
        "dist_to_high_omar": (close - omar["high"]) / r,
        "dist_to_low_omar": (close - omar["low"]) / r,
        "dist_to_mid_omar": (close - omar["mid"]) / r,
        "nearest_level_abs_omar": min(
            abs(close - omar["high"]),
            abs(close - omar["low"]),
            abs(close - omar["mid"]),
        )
        / r,
    }


def _forward_move_size(
    ds: V2Dataset, day: str, minute: int, horizon: int = 20
) -> float | None:
    """Max-favorable-excursion of SPX over the next `horizon` bars, as %
    of current close. Used to test "near-OMAR predicts a big move" idea.
    """
    try:
        day_start, day_end = ds.day_bar_range(day)
    except ValueError:
        return None
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return None
    close = float(ds.spot_prices[abs_idx])
    end = min(abs_idx + horizon + 1, day_end)
    window = ds.spot_prices[abs_idx + 1 : end]
    if len(window) < 2 or close <= 0:
        return None
    mx = float(np.max(window))
    mn = float(np.min(window))
    max_excursion = max(abs(mx - close), abs(mn - close))
    return max_excursion / close


def _describe(vals: list[float], name: str) -> None:
    if not vals:
        print(f"  {name}: n=0")
        return
    arr = np.asarray(vals)
    print(
        f"  {name}: n={len(arr)} "
        f"p10={np.percentile(arr, 10):+.3f} "
        f"p25={np.percentile(arr, 25):+.3f} "
        f"median={np.percentile(arr, 50):+.3f} "
        f"p75={np.percentile(arr, 75):+.3f} "
        f"p90={np.percentile(arr, 90):+.3f}"
    )


def main() -> int:
    print("Loading dataset + OMAR levels...")
    ds = V2Dataset.load()
    omar_map = _build_omar_levels(
        str(Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl")
    )
    print(f"  OMAR levels for {len(omar_map)} days")

    sessions = _load_per_session(_latest_attribution_file())
    abstention = [s for s in sessions if s.get("outcome") == "abstention"]
    entered_right = [s for s in sessions if s.get("outcome") == "entered_right"]

    # --- Q1+Q3: Cohort distance-to-OMAR distributions ---
    print()
    print("=== Q1+Q3: Distance from oracle-bar close to OMAR levels (in OMAR-range units) ===")
    print("  |dist| <= 0.5 means 'within half an OMAR range of that level' -- a retest zone.")
    print()

    for cohort_name, cohort in (("abstention", abstention), ("entered_right", entered_right)):
        print(f"[{cohort_name}] n={len(cohort)}")
        nearest_vals = []
        dists_h, dists_l, dists_m = [], [], []
        n_near_any = 0
        for s in cohort:
            omar = omar_map.get(s["day"])
            if omar is None:
                continue
            d = _distance_to_omar(ds, s["day"], s["oracle_bar"], omar)
            if d is None:
                continue
            nearest_vals.append(d["nearest_level_abs_omar"])
            dists_h.append(d["dist_to_high_omar"])
            dists_l.append(d["dist_to_low_omar"])
            dists_m.append(d["dist_to_mid_omar"])
            if d["nearest_level_abs_omar"] <= 0.5:
                n_near_any += 1
        _describe(dists_h, "dist to OMAR high")
        _describe(dists_l, "dist to OMAR low ")
        _describe(dists_m, "dist to OMAR mid ")
        _describe(nearest_vals, "|nearest level|  ")
        if nearest_vals:
            pct = 100 * n_near_any / len(nearest_vals)
            print(f"  % within 0.5 OMAR of ANY level: {pct:.1f}%  ({n_near_any}/{len(nearest_vals)})")
        print()

    # --- Q2: Discriminative test. Compare to random non-oracle bars from the same days. ---
    print("=== Q2: Random non-oracle bars as CONTROL ===")
    print("  Pick 5 random minutes (in teacher union window [15, 120]) per day,")
    print("  excluding each day's oracle bar. If these behave like oracle bars,")
    print("  OMAR proximity is not discriminative.")
    print()
    rng = random.Random(0)
    control_nearest = []
    all_days = sorted(set(ds.dates))
    control_days = rng.sample(all_days, min(200, len(all_days)))
    for day in control_days:
        omar = omar_map.get(day)
        if omar is None:
            continue
        for _ in range(5):
            minute = rng.randint(15, 120)
            d = _distance_to_omar(ds, day, minute, omar)
            if d:
                control_nearest.append(d["nearest_level_abs_omar"])
    _describe(control_nearest, "control |nearest|")
    pct_control = 100 * sum(1 for v in control_nearest if v <= 0.5) / max(len(control_nearest), 1)
    print(f"  control % within 0.5 OMAR of ANY level: {pct_control:.1f}%")
    print()

    # --- Q4: Forward-move predictor test ---
    print("=== Q4: Does 'near OMAR' predict a bigger forward move? ===")
    print("  Sample 300 random eligible bars. Split into 'near OMAR' vs 'far from OMAR'.")
    print("  Compare the next-20-bar SPX % excursion.")
    print()
    near_moves, far_moves = [], []
    rng2 = random.Random(1)
    test_days = rng2.sample(all_days, min(400, len(all_days)))
    for day in test_days:
        omar = omar_map.get(day)
        if omar is None:
            continue
        minute = rng2.randint(15, 100)
        d = _distance_to_omar(ds, day, minute, omar)
        if d is None:
            continue
        move = _forward_move_size(ds, day, minute, horizon=20)
        if move is None:
            continue
        if d["nearest_level_abs_omar"] <= 0.5:
            near_moves.append(move)
        elif d["nearest_level_abs_omar"] >= 1.5:
            far_moves.append(move)
    if near_moves and far_moves:
        n_arr = np.asarray(near_moves)
        f_arr = np.asarray(far_moves)
        print(
            f"  near OMAR (|dist|<=0.5): n={len(n_arr)} "
            f"median_move={100*np.median(n_arr):.3f}%  "
            f"p75={100*np.percentile(n_arr,75):.3f}%  "
            f"p90={100*np.percentile(n_arr,90):.3f}%"
        )
        print(
            f"  far from OMAR (|dist|>=1.5): n={len(f_arr)} "
            f"median_move={100*np.median(f_arr):.3f}%  "
            f"p75={100*np.percentile(f_arr,75):.3f}%  "
            f"p90={100*np.percentile(f_arr,90):.3f}%"
        )
        ratio = np.median(n_arr) / max(np.median(f_arr), 1e-9)
        print(f"  ratio (near_median / far_median): {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
