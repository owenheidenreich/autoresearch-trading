"""Test 4: per-bar developing VP vs checkpoint-based VP.

The volume_profile research used 30/60/90/120-min checkpoints. Implementation
would recompute VP per bar. Question: does the finer resolution change the
1.32x abstention enrichment (67.6% near VP vs 51.3% control)?

Incremental histogram update keeps this O(n_bars) per day.
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


def _omar_ranges(spx_1min_path: str) -> dict[str, float]:
    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, float] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        h = float(g.iloc[0]["spx_high"])
        l = float(g.iloc[0]["spx_low"])
        out[day] = max(h - l, 0.01)
    return out


def _per_bar_vp_lookup(
    spy_path: str, bucket_pct: float = 0.0002
) -> dict[tuple[str, int], dict[str, float]]:
    """Per (day, minute): POC/VAH/VAL computed incrementally from 09:30 through minute."""
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) < 20:
            continue
        mean_price = float(g["close"].mean())
        bucket_width = max(mean_price * bucket_pct, 0.01)

        highs = g["high"].to_numpy(dtype=float)
        lows = g["low"].to_numpy(dtype=float)
        closes = g["close"].to_numpy(dtype=float)
        vols = g["volume"].to_numpy(dtype=float)

        buckets: dict[int, float] = defaultdict(float)
        for m in range(len(g)):
            if vols[m] <= 0:
                continue
            typical = (highs[m] + lows[m] + closes[m]) / 3.0
            if not np.isfinite(typical):
                continue
            bucket_idx = int(round(typical / bucket_width))
            buckets[bucket_idx] += float(vols[m])

            # Only snapshot at minutes we care about [40, 120], every 2 bars
            if m < 30 or m > 135 or m % 2 != 0:
                continue
            if not buckets:
                continue
            sorted_idx = sorted(buckets.keys())
            poc_idx = max(buckets.items(), key=lambda kv: kv[1])[0]
            poc_pos = sorted_idx.index(poc_idx)
            total = sum(buckets.values())
            if total <= 0:
                continue
            target = total * 0.70
            inc = buckets[poc_idx]
            lo_pos = poc_pos
            hi_pos = poc_pos
            while inc < target:
                lv = buckets[sorted_idx[lo_pos - 1]] if lo_pos > 0 else -1
                hv = buckets[sorted_idx[hi_pos + 1]] if hi_pos < len(sorted_idx) - 1 else -1
                if lv < 0 and hv < 0:
                    break
                if lv >= hv and lo_pos > 0:
                    lo_pos -= 1
                    inc += buckets[sorted_idx[lo_pos]]
                elif hi_pos < len(sorted_idx) - 1:
                    hi_pos += 1
                    inc += buckets[sorted_idx[hi_pos]]
                else:
                    break
            out[(day, m)] = {
                "poc": poc_idx * bucket_width,
                "vah": sorted_idx[hi_pos] * bucket_width,
                "val": sorted_idx[lo_pos] * bucket_width,
                "spy_close": float(closes[m]),
            }
    return out


def _nearest_vp_dist(spx_close, vp, omar_range):
    spy_ref = vp.get("spy_close", 0.0)
    if spy_ref <= 0 or omar_range <= 0:
        return None
    scale = spx_close / spy_ref
    diffs = []
    for k in ("poc", "vah", "val"):
        lv = vp.get(k)
        if lv is None:
            continue
        diffs.append(abs(spx_close - lv * scale) / omar_range)
    return min(diffs) if diffs else None


def main() -> int:
    print("Loading data...")
    ds = V2Dataset.load()
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"

    omar = _omar_ranges(str(spx_path))
    print("Computing per-bar developing VP (every 2 bars in [30, 135])...")
    per_bar = _per_bar_vp_lookup(str(spy_path))
    print(f"  {len(per_bar)} (day, bar) records")

    sessions = _load_per_session(_latest_attribution_file())
    cohorts = {
        "entered_right": [s for s in sessions if s.get("outcome") == "entered_right"],
        "abstention": [s for s in sessions if s.get("outcome") == "abstention"],
        "side_error": [s for s in sessions if s.get("outcome") == "side_error"],
    }
    rng = random.Random(7)
    control_days = sorted({s["day"] for s in sessions})
    cohorts["control"] = [{"day": d, "oracle_bar": rng.randint(40, 120)} for d in control_days]

    print()
    print(f"=== Per-bar developing VP enrichment ===")
    print(f"{'cohort':<18}{'n':>6}{'med_dist':>12}{'pct_within_0.5':>16}")
    print("-" * 52)

    summary = {}
    for name, rows in cohorts.items():
        dists = []
        for s in rows:
            day = s["day"]
            minute = s["oracle_bar"]
            # Find nearest snapshot (<= minute, even)
            snap_min = (minute // 2) * 2
            vp = per_bar.get((day, snap_min))
            if vp is None:
                continue
            r = omar.get(day)
            if r is None:
                continue
            spx_close = 0.0
            try:
                day_start, day_end = ds.day_bar_range(day)
                abs_idx = day_start + minute
                if abs_idx < day_end:
                    spx_close = float(ds.spot_prices[abs_idx])
            except ValueError:
                continue
            if spx_close <= 0:
                continue
            d = _nearest_vp_dist(spx_close, vp, r)
            if d is None:
                continue
            dists.append(d)
        if not dists:
            continue
        d_arr = np.asarray(dists)
        pct = 100 * np.mean(d_arr <= 0.5)
        summary[name] = {"n": len(d_arr), "pct": pct, "med": np.median(d_arr)}
        print(f"  {name:<16}{len(d_arr):>6}{np.median(d_arr):>+12.3f}{pct:>15.1f}%")

    print()
    ctrl = summary.get("control")
    if ctrl:
        for name in ("abstention", "entered_right", "side_error"):
            s = summary.get(name)
            if not s:
                continue
            ratio = s["pct"] / max(ctrl["pct"], 1e-9)
            print(f"  {name:<16}: enrichment = {ratio:.2f}x")

    print()
    print("=== Comparison to checkpoint-based VP from volume_profile doc ===")
    print("  checkpoint abstention: 67.6% near VP vs 51.3% control = 1.32x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
