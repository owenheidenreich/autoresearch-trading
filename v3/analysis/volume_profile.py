"""Volume Profile investigation — do oracle bars cluster near POC/VAH/VAL?

Pickles treats Volume Profile levels as his strongest S/R inputs:
- POC (Point of Control) = the single price that traded the most volume
- VAH (Value Area High) = upper boundary of the 70% volume zone around POC
- VAL (Value Area Low) = lower boundary of the 70% volume zone around POC

Two profiles tested:

1. **Prior-Day VP.** Yesterday's full-session POC/VAH/VAL. Pickles publishes
   these pre-market; traders expect current session to retest them.

2. **Intraday Developing VP.** Current session's POC/VAH/VAL computed from
   09:30 up to (minute_of_session − 1). This is what a trader sees in real
   time. Note: bars before minute ~20 have unstable values, so we only
   evaluate at bars ≥ 30.

Computed using SPY volume (SPX has no direct volume). SPY levels translated
to SPX-space via per-bar SPX/SPY ratio.

For each cohort:
- Distance to PD_POC, PD_VAH, PD_VAL (in OMAR-range units)
- % within 0.5× OMAR of any PD-VP level
- Same for intraday developing VP (from minute ≥30 only)
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


def _compute_vp(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    bucket_width: float,
) -> tuple[float | None, float | None, float | None]:
    """Return (POC, VAH, VAL) in price units. None if input is empty.

    Assigns each bar's volume to its typical price bucket ((H+L+C)/3).
    Value Area: expand from POC outward in the direction of higher adjacent
    volume until 70% of total session volume is contained.
    """
    if len(closes) == 0 or volumes.sum() == 0:
        return None, None, None
    typical = (highs + lows + closes) / 3.0
    buckets: dict[int, float] = defaultdict(float)
    for i in range(len(closes)):
        if volumes[i] <= 0 or not np.isfinite(typical[i]):
            continue
        bucket_idx = int(round(typical[i] / bucket_width))
        buckets[bucket_idx] += float(volumes[i])
    if not buckets:
        return None, None, None
    sorted_idx = sorted(buckets.keys())
    # POC = max-volume bucket
    poc_idx = max(buckets.items(), key=lambda kv: kv[1])[0]
    poc_pos = sorted_idx.index(poc_idx)
    total_vol = sum(buckets.values())
    target_vol = total_vol * 0.70
    included = buckets[poc_idx]
    lo_pos = poc_pos
    hi_pos = poc_pos
    while included < target_vol:
        lo_vol = buckets[sorted_idx[lo_pos - 1]] if lo_pos > 0 else -1
        hi_vol = buckets[sorted_idx[hi_pos + 1]] if hi_pos < len(sorted_idx) - 1 else -1
        if lo_vol < 0 and hi_vol < 0:
            break
        if lo_vol >= hi_vol and lo_pos > 0:
            lo_pos -= 1
            included += buckets[sorted_idx[lo_pos]]
        elif hi_pos < len(sorted_idx) - 1:
            hi_pos += 1
            included += buckets[sorted_idx[hi_pos]]
        else:
            break
    poc = poc_idx * bucket_width
    val = sorted_idx[lo_pos] * bucket_width
    vah = sorted_idx[hi_pos] * bucket_width
    return poc, vah, val


def _build_daily_vp(
    spy_path: str, bucket_pct: float = 0.0002
) -> dict[str, dict[str, float]]:
    """Per day: SPY POC/VAH/VAL (full session) + session close (for translation)."""
    import pandas as pd
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)

    out: dict[str, dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) < 10:
            continue
        mean_price = float(g["close"].mean())
        bucket_width = max(mean_price * bucket_pct, 0.01)
        poc, vah, val = _compute_vp(
            g["high"].to_numpy(dtype=float),
            g["low"].to_numpy(dtype=float),
            g["close"].to_numpy(dtype=float),
            g["volume"].to_numpy(dtype=float),
            bucket_width,
        )
        if poc is None:
            continue
        out[day] = {
            "poc": poc,
            "vah": vah,
            "val": val,
            "session_close": float(g.iloc[-1]["close"]),
            "session_open": float(g.iloc[0]["open"]),
        }
    return out


def _build_developing_vp_lookup(
    spy_path: str, bucket_pct: float = 0.0002, checkpoints: tuple[int, ...] = (30, 60, 90, 120)
) -> dict[tuple[str, int], dict[str, float]]:
    """Per (day, checkpoint_minute): POC/VAH/VAL from 09:30 through that minute.

    `checkpoint_minute` 30 means VP computed over the first 30 bars (09:30-10:00).
    """
    import pandas as pd
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        mean_price = float(g["close"].mean())
        bucket_width = max(mean_price * bucket_pct, 0.01)
        for cp in checkpoints:
            end = min(cp, len(g))
            if end < 10:
                continue
            sub = g.iloc[:end]
            poc, vah, val = _compute_vp(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float),
                sub["close"].to_numpy(dtype=float),
                sub["volume"].to_numpy(dtype=float),
                bucket_width,
            )
            if poc is None:
                continue
            out[(day, cp)] = {
                "poc": poc,
                "vah": vah,
                "val": val,
                "spy_close_at_cp": float(sub.iloc[-1]["close"]),
            }
    return out


def _omar_range_for_day(spx_1min_path: str) -> dict[str, dict[str, float]]:
    import pandas as pd
    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        h = float(g.iloc[0]["spx_high"])
        l = float(g.iloc[0]["spx_low"])
        out[day] = {"range": max(h - l, 0.01)}
    return out


def _translate_spy_to_spx(level_spy: float, spx_close: float, spy_close: float) -> float:
    if spy_close <= 0 or spx_close <= 0:
        return level_spy
    return level_spy * (spx_close / spy_close)


def _nearest_vp_level_dist_omar(
    spx_close: float,
    spy_levels: dict[str, float],
    spy_ref_close: float,
    omar_range: float,
) -> tuple[float | None, int]:
    """Return (min |dist to any VP level| in OMAR units, count within 0.5 OMAR)."""
    if omar_range <= 0 or spy_ref_close <= 0:
        return None, 0
    diffs = []
    count_close = 0
    for k in ("poc", "vah", "val"):
        lv_spy = spy_levels.get(k)
        if lv_spy is None or not np.isfinite(lv_spy):
            continue
        lv_spx = _translate_spy_to_spx(lv_spy, spx_close, spy_ref_close)
        d = abs(spx_close - lv_spx) / omar_range
        diffs.append(d)
        if d <= 0.5:
            count_close += 1
    return (min(diffs) if diffs else None, count_close)


def main() -> int:
    print("Loading data...")
    ds = V2Dataset.load()
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"

    print("Computing prior-day volume profiles (POC/VAH/VAL) on SPY...")
    pd_vp = _build_daily_vp(str(spy_path))
    print(f"  {len(pd_vp)} days with prior-day VP")

    print("Computing intraday developing VP at checkpoints 30/60/90/120...")
    dev_vp = _build_developing_vp_lookup(str(spy_path), checkpoints=(30, 60, 90, 120))
    print(f"  {len(dev_vp)} (day, checkpoint) records")

    omar_map = _omar_range_for_day(str(spx_path))
    sorted_days = sorted(pd_vp.keys())

    sessions = _load_per_session(_latest_attribution_file())
    cohorts = {
        "entered_right": [s for s in sessions if s.get("outcome") == "entered_right"],
        "abstention": [s for s in sessions if s.get("outcome") == "abstention"],
        "side_error": [s for s in sessions if s.get("outcome") == "side_error"],
    }
    # Random control
    rng = random.Random(7)
    control = [{"day": d, "oracle_bar": rng.randint(40, 120)} for d in sorted_days]
    cohorts["control"] = control

    print()
    print("=== PRIOR-DAY VP (POC, VAH, VAL) — oracle bar proximity ===")
    print(
        f"{'cohort':<18}{'n':>6}{'med_dist':>12}{'p25':>10}{'p75':>10}"
        f"{'pct_within_0.5':>16}{'avg_count_close':>18}"
    )
    print("-" * 90)

    summary_pd: dict[str, dict] = {}
    for cohort_name, rows in cohorts.items():
        distances: list[float] = []
        counts: list[int] = []
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            # Prior-day VP is the PREVIOUS day's (not today's)
            if day not in sorted_days:
                continue
            idx = sorted_days.index(day)
            if idx == 0:
                continue
            prev_day = sorted_days[idx - 1]
            prev_vp = pd_vp.get(prev_day)
            omar = omar_map.get(day)
            if prev_vp is None or omar is None:
                continue
            try:
                day_start, day_end = ds.day_bar_range(day)
            except ValueError:
                continue
            abs_idx = day_start + minute
            if abs_idx >= day_end:
                continue
            spx_close = float(ds.spot_prices[abs_idx])
            d, c = _nearest_vp_level_dist_omar(
                spx_close, prev_vp, prev_vp["session_close"], omar["range"]
            )
            if d is None:
                continue
            distances.append(d)
            counts.append(c)

        if not distances:
            print(f"{cohort_name:<18}{0:>6}")
            continue
        d_arr = np.asarray(distances)
        c_arr = np.asarray(counts)
        pct_close = 100 * np.mean(d_arr <= 0.5)
        summary_pd[cohort_name] = {
            "n": len(d_arr),
            "med_dist": float(np.median(d_arr)),
            "pct_within_0.5": pct_close,
            "avg_count_close": float(c_arr.mean()),
        }
        print(
            f"{cohort_name:<18}{len(d_arr):>6}"
            f"{np.percentile(d_arr, 50):>+12.3f}"
            f"{np.percentile(d_arr, 25):>+10.3f}"
            f"{np.percentile(d_arr, 75):>+10.3f}"
            f"{pct_close:>15.1f}%"
            f"{float(c_arr.mean()):>18.3f}"
        )

    print()
    print("=== PRIOR-DAY VP enrichment ratios (cohort / control) ===")
    ctrl = summary_pd.get("control")
    if ctrl:
        for cohort_name in ("abstention", "entered_right", "side_error"):
            s = summary_pd.get(cohort_name)
            if not s:
                continue
            ratio = s["pct_within_0.5"] / max(ctrl["pct_within_0.5"], 1e-9)
            cnt_ratio = s["avg_count_close"] / max(ctrl["avg_count_close"], 1e-9)
            print(
                f"  {cohort_name:<18}: pct_within_0.5 enrichment = {ratio:.2f}×  "
                f"avg_count_close ratio = {cnt_ratio:.2f}×"
            )

    print()
    print("=== INTRADAY DEVELOPING VP at checkpoint BEFORE oracle bar ===")
    print("  Uses the latest checkpoint that is <= oracle bar's minute.")
    print(
        f"{'cohort':<18}{'n':>6}{'med_dist':>12}{'p25':>10}{'p75':>10}"
        f"{'pct_within_0.5':>16}"
    )
    print("-" * 72)
    summary_dev: dict[str, dict] = {}
    for cohort_name, rows in cohorts.items():
        distances: list[float] = []
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            # Pick the most recent checkpoint <= minute
            cp = None
            for c_try in (120, 90, 60, 30):
                if c_try < minute and (day, c_try) in dev_vp:
                    cp = c_try
                    break
            if cp is None:
                continue
            vp = dev_vp[(day, cp)]
            omar = omar_map.get(day)
            if omar is None:
                continue
            try:
                day_start, day_end = ds.day_bar_range(day)
            except ValueError:
                continue
            abs_idx = day_start + minute
            if abs_idx >= day_end:
                continue
            spx_close = float(ds.spot_prices[abs_idx])
            d, _ = _nearest_vp_level_dist_omar(
                spx_close, vp, vp["spy_close_at_cp"], omar["range"]
            )
            if d is None:
                continue
            distances.append(d)
        if not distances:
            print(f"{cohort_name:<18}{0:>6}")
            continue
        d_arr = np.asarray(distances)
        pct_close = 100 * np.mean(d_arr <= 0.5)
        summary_dev[cohort_name] = {
            "n": len(d_arr),
            "pct_within_0.5": pct_close,
        }
        print(
            f"{cohort_name:<18}{len(d_arr):>6}"
            f"{np.percentile(d_arr, 50):>+12.3f}"
            f"{np.percentile(d_arr, 25):>+10.3f}"
            f"{np.percentile(d_arr, 75):>+10.3f}"
            f"{pct_close:>15.1f}%"
        )

    print()
    print("=== INTRADAY DEVELOPING VP enrichment (cohort / control) ===")
    ctrl = summary_dev.get("control")
    if ctrl:
        for cohort_name in ("abstention", "entered_right", "side_error"):
            s = summary_dev.get(cohort_name)
            if not s:
                continue
            ratio = s["pct_within_0.5"] / max(ctrl["pct_within_0.5"], 1e-9)
            print(
                f"  {cohort_name:<18}: pct_within_0.5 enrichment = {ratio:.2f}×"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
