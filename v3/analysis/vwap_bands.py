"""VWAP bands investigation — does Pickles' thesis hold for SPX 0DTE?

Pickles' rules (from his journal):
- "+1 to +2 VWAP = overextended. Wait for pullback to +1/+1.5 before longs."
- "-1 to -2 VWAP = overextended short. Wait for bounce to -0.5/-1 before puts."
- "Do NOT long at +2 VWAP."
- "Price between +1 and +2 sustained = strong uptrend."
- "First-touch VWAP rejection often fails; enter on 2nd test."

We test these empirically using SPY volume (since SPX has no volume). SPY
has Polygon-computed VWAP and raw volume from 2022-03 to 2026-04 in the
cache. Since SPY ≈ SPX/10, SPX vs SPY-derived VWAP bands should correspond
directly.

Two passes:

1. **Build SPY VWAP + bands.** Start from Polygon's pre-computed `vwap`
   column. Compute rolling session-to-date stddev of (close − vwap)
   weighted by volume. Produce ±1σ and ±2σ bands for each SPY minute.

2. **Translate to SPX-equivalent bands.** Use the per-bar SPX/SPY ratio to
   scale SPY-VWAP levels to SPX-space. Align by timestamp.

3. **Test each Pickles rule:**
   a) Where are entered_right BUY_CALL oracle bars relative to bands?
      (Are they near VWAP / ≤+1σ, or stretched to +2σ?)
   b) Where are entered_right BUY_PUT oracle bars?
   c) Where are abstention BUY_CALL / BUY_PUT oracle bars?
   d) Compare to random non-oracle bars as control.

A Pickles-aligned teacher would use: "only enter longs when SPX is near
VWAP or below +1σ"; our test is whether oracle bars actually satisfy that.
"""
from __future__ import annotations

import glob
import json
import pickle
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


def _build_spy_vwap_bands(
    spy_1min_path: str,
) -> dict[str, dict[str, np.ndarray]]:
    """For each day, compute per-minute VWAP + ±1σ and ±2σ deviation bands.

    Uses Polygon's pre-computed `vwap` column as the VWAP. Computes stddev
    of (close − vwap) weighted by volume, running cumulatively within each
    session. Returns per-day arrays keyed by date string.

    Array length matches that day's SPY minute count. Caller aligns to the
    SPX minute grid via timestamp/minute-of-session.
    """
    df = pickle.load(open(spy_1min_path, "rb"))
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, group in df.groupby("date"):
        g = group.reset_index(drop=True).copy()
        if len(g) == 0:
            continue
        vwap = g["vwap"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        volume = g["volume"].to_numpy(dtype=float)
        n = len(g)
        # Running cumulative weighted variance: var_t = Σ v_i*(close_i - vwap_i)^2 / Σ v_i
        sq_dev = (close - vwap) ** 2
        cum_v = np.cumsum(volume)
        cum_sq = np.cumsum(volume * sq_dev)
        # Avoid div-by-zero on first-bar volume==0
        cum_v_safe = np.where(cum_v > 0, cum_v, 1.0)
        var_run = cum_sq / cum_v_safe
        std_run = np.sqrt(np.maximum(var_run, 1e-12))
        out[str(day)] = {
            "minute_of_session": np.arange(n),
            "close": close,
            "vwap": vwap,
            "std": std_run,
            "upper_1": vwap + std_run,
            "upper_2": vwap + 2 * std_run,
            "lower_1": vwap - std_run,
            "lower_2": vwap - 2 * std_run,
        }
    return out


def _spy_level_at_minute(
    spy_day: dict[str, np.ndarray], minute: int, key: str
) -> float | None:
    arr = spy_day.get(key)
    if arr is None or minute >= len(arr) or minute < 0:
        return None
    val = float(arr[minute])
    return val if np.isfinite(val) and val > 0 else None


def _band_position(
    spx_close: float, spy_close: float, spy_vwap: float, spy_std: float
) -> float | None:
    """Position of SPX close in σ-units of SPY VWAP bands.

    Scales SPY VWAP/std to SPX-equivalent using the SPX/SPY ratio, then
    returns (SPX_close − SPX_VWAP_est) / SPX_std_est.

    Returns None on degenerate input.
    """
    if spy_close <= 0 or spy_vwap <= 0 or spy_std <= 0 or spx_close <= 0:
        return None
    ratio = spx_close / spy_close
    spx_vwap_est = spy_vwap * ratio
    spx_std_est = spy_std * ratio
    return (spx_close - spx_vwap_est) / spx_std_est


def _oracle_band_position(
    ds: V2Dataset,
    spy_vwap_map: dict[str, dict[str, np.ndarray]],
    day: str,
    minute: int,
) -> float | None:
    spy_day = spy_vwap_map.get(day)
    if spy_day is None:
        return None
    try:
        day_start, day_end = ds.day_bar_range(day)
    except ValueError:
        return None
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return None
    spx_close = float(ds.spot_prices[abs_idx])
    spy_vwap = _spy_level_at_minute(spy_day, minute, "vwap")
    spy_std = _spy_level_at_minute(spy_day, minute, "std")
    spy_close = _spy_level_at_minute(spy_day, minute, "close")
    if None in (spy_vwap, spy_std, spy_close):
        return None
    return _band_position(spx_close, spy_close, spy_vwap, spy_std)


def _bucketize(pos: float) -> str:
    if pos is None or not np.isfinite(pos):
        return "unknown"
    if pos >= 2.0:
        return "above +2σ"
    if pos >= 1.0:
        return "+1σ to +2σ"
    if pos >= 0.5:
        return "+0.5σ to +1σ"
    if pos >= -0.5:
        return "near VWAP (±0.5σ)"
    if pos >= -1.0:
        return "-1σ to -0.5σ"
    if pos >= -2.0:
        return "-2σ to -1σ"
    return "below -2σ"


BUCKET_ORDER = [
    "above +2σ",
    "+1σ to +2σ",
    "+0.5σ to +1σ",
    "near VWAP (±0.5σ)",
    "-1σ to -0.5σ",
    "-2σ to -1σ",
    "below -2σ",
]


def main() -> int:
    print("Loading SPX dataset...")
    ds = V2Dataset.load()

    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"
    print(f"Building SPY VWAP + bands from {spy_path}...")
    spy_map = _build_spy_vwap_bands(str(spy_path))
    print(f"  {len(spy_map)} SPY days with VWAP")

    sessions = _load_per_session(_latest_attribution_file())
    print(f"  {len(sessions)} per-session attribution rows")

    print()
    print("=== Pickles-rule test: where do oracle bars sit in VWAP σ-space? ===")
    print()

    cohorts = {
        "entered_right_call": [
            s for s in sessions
            if s.get("outcome") == "entered_right"
            and s.get("oracle_direction") == "call"
        ],
        "entered_right_put": [
            s for s in sessions
            if s.get("outcome") == "entered_right"
            and s.get("oracle_direction") == "put"
        ],
        "abstention_call": [
            s for s in sessions
            if s.get("outcome") == "abstention"
            and s.get("oracle_direction") == "call"
        ],
        "abstention_put": [
            s for s in sessions
            if s.get("outcome") == "abstention"
            and s.get("oracle_direction") == "put"
        ],
    }

    # Control: pick one random minute (post-MAGIC, within eligible window) per day
    import random
    rng = random.Random(42)
    control: list[dict] = []
    for day in sorted(set(ds.dates))[::2]:  # every other day to keep sample manageable
        control.append({"day": day, "oracle_bar": rng.randint(40, 120)})

    all_cohorts = list(cohorts.items()) + [("control_random", control)]

    # Build band-position distributions per cohort
    summaries: dict[str, dict[str, int]] = {}
    raw_positions: dict[str, list[float]] = {}
    for cohort_name, rows in all_cohorts:
        bucket_counts: dict[str, int] = defaultdict(int)
        positions: list[float] = []
        for s in rows:
            pos = _oracle_band_position(ds, spy_map, s["day"], s["oracle_bar"])
            if pos is None:
                continue
            positions.append(pos)
            bucket_counts[_bucketize(pos)] += 1
        summaries[cohort_name] = dict(bucket_counts)
        raw_positions[cohort_name] = positions

    header_cohorts = [
        ("entered_right_call", "right_call"),
        ("entered_right_put", "right_put"),
        ("abstention_call", "abs_call"),
        ("abstention_put", "abs_put"),
        ("control_random", "control"),
    ]

    print(f"{'bucket (σ from VWAP)':<22}" + "".join(f"{h[1]:>12}" for h in header_cohorts))
    print("-" * (22 + 12 * len(header_cohorts)))
    totals = {name: sum(summaries.get(name, {}).values()) for name, _ in header_cohorts}
    for bucket in BUCKET_ORDER:
        row = f"{bucket:<22}"
        for cohort_name, _ in header_cohorts:
            counts = summaries.get(cohort_name, {})
            n = totals[cohort_name]
            pct = 100 * counts.get(bucket, 0) / n if n else 0
            row += f"{pct:>11.1f}%"
        print(row)
    print("-" * (22 + 12 * len(header_cohorts)))
    row = f"{'n':<22}"
    for cohort_name, _ in header_cohorts:
        row += f"{totals[cohort_name]:>12}"
    print(row)
    print()

    # Percentile summary of raw band positions
    print("=== Raw band-position stats (σ from VWAP) ===")
    print(f"{'cohort':<22}{'n':>8}{'p10':>8}{'p25':>8}{'median':>10}{'p75':>8}{'p90':>8}")
    for cohort_name, _ in header_cohorts:
        vals = np.asarray(raw_positions.get(cohort_name, []))
        if vals.size == 0:
            print(f"{cohort_name:<22}{0:>8}  ---")
            continue
        print(
            f"{cohort_name:<22}{vals.size:>8}"
            f"{np.percentile(vals, 10):>+8.2f}"
            f"{np.percentile(vals, 25):>+8.2f}"
            f"{np.percentile(vals, 50):>+10.2f}"
            f"{np.percentile(vals, 75):>+8.2f}"
            f"{np.percentile(vals, 90):>+8.2f}"
        )

    # Pickles-rule verification
    print()
    print("=== Pickles-rule verification ===")
    print("Rule: 'do NOT long at +2σ. Longs work near VWAP or below +1σ.'")
    for cohort_name in ("entered_right_call", "abstention_call"):
        vals = np.asarray(raw_positions.get(cohort_name, []))
        if vals.size == 0:
            continue
        pct_above_plus2 = 100 * np.mean(vals >= 2.0)
        pct_above_plus1 = 100 * np.mean(vals >= 1.0)
        pct_near_or_below = 100 * np.mean(vals < 1.0)
        print(
            f"  [{cohort_name}] n={vals.size}  "
            f"%≥+2σ: {pct_above_plus2:.1f}%  "
            f"%≥+1σ: {pct_above_plus1:.1f}%  "
            f"%<+1σ: {pct_near_or_below:.1f}%"
        )
    print()
    print("Rule: 'do NOT put at -2σ. Puts work near VWAP or above -1σ.'")
    for cohort_name in ("entered_right_put", "abstention_put"):
        vals = np.asarray(raw_positions.get(cohort_name, []))
        if vals.size == 0:
            continue
        pct_below_minus2 = 100 * np.mean(vals <= -2.0)
        pct_below_minus1 = 100 * np.mean(vals <= -1.0)
        pct_near_or_above = 100 * np.mean(vals > -1.0)
        print(
            f"  [{cohort_name}] n={vals.size}  "
            f"%≤-2σ: {pct_below_minus2:.1f}%  "
            f"%≤-1σ: {pct_below_minus1:.1f}%  "
            f"%>-1σ: {pct_near_or_above:.1f}%"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
