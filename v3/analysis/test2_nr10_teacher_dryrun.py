"""Test 2: A2 dry-run — simulate the NR10 teacher across 986 days.

Walks every bar in [40, 120] of every session, applies the 5-gate spec:
  Window:         [40, 120]
  Eligibility:    SPX close inside first15 range
  Trigger:        close breaks last-10-bar high (CALL) or low (PUT)
  Squeeze gate:   last-10-bar SPX range <= 1.0 * OMAR range
  Direction gate: CALL requires sigma_pos <= +0.5; PUT requires sigma_pos >= -0.5
  Retest gate:    SPX close within 0.5 * OMAR of OMAR high, low, or mid

Output:
  - Total triggers (and triggers/day density)
  - Triggers that coincide with an abstention oracle bar
  - Triggers that coincide with a side_error oracle bar (potential new side_error)
  - Triggers that coincide with an entered_right oracle bar (overlap with ORC)
  - Forward-move MFE/MAE over +5/+10/+15/+20 minutes (fulfills Test 5 simultaneously)
  - Raw baseline: trigger + squeeze only (no direction/retest) for comparison

Question: does the 39.6% cohort-level abstention capture translate into
trigger-level capture? Answer is (triggers_at_abstention / 467_abstention_bars)
as a first-order approximation. Also measures whether filtered triggers have
a forward-move edge over raw (unfiltered-beyond-squeeze) triggers.
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


def _build_spx_bars(spx_1min_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Per day: high/low/close arrays indexed by minute_of_session."""
    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        out[day] = {
            "high": g["spx_high"].to_numpy(dtype=float),
            "low": g["spx_low"].to_numpy(dtype=float),
            "close": g["spx_close"].to_numpy(dtype=float),
        }
    return out


def _build_omar_map(spx_bars: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for day, arrs in spx_bars.items():
        if len(arrs["high"]) == 0:
            continue
        h = float(arrs["high"][0])
        l = float(arrs["low"][0])
        out[day] = {
            "high": h,
            "low": l,
            "mid": (h + l) / 2.0,
            "range": max(h - l, 0.01),
        }
    return out


def _build_spy_vwap(spy_path: str) -> dict[str, dict[str, np.ndarray]]:
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        vwap = g["vwap"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        volume = g["volume"].to_numpy(dtype=float)
        sq_dev = (close - vwap) ** 2
        cum_v = np.maximum(np.cumsum(volume), 1.0)
        cum_sq = np.cumsum(volume * sq_dev)
        std_run = np.sqrt(np.maximum(cum_sq / cum_v, 1e-12))
        out[day] = {"vwap": vwap, "close": close, "std": std_run}
    return out


def _sigma_pos(
    spy_day: dict[str, np.ndarray], minute: int, spx_close: float
) -> float | None:
    vwap = spy_day["vwap"]
    close = spy_day["close"]
    std = spy_day["std"]
    if minute >= len(vwap) or minute < 0:
        return None
    spy_vwap = float(vwap[minute])
    spy_close = float(close[minute])
    spy_std = float(std[minute])
    if not all(v > 0 for v in (spy_vwap, spy_close, spy_std, spx_close)):
        return None
    ratio = spx_close / spy_close
    spx_vwap = spy_vwap * ratio
    spx_std = spy_std * ratio
    return (spx_close - spx_vwap) / spx_std


def _retest_pass(
    spx_close: float, omar: dict[str, float], threshold: float = 0.5
) -> bool:
    r = omar["range"]
    if r <= 0:
        return False
    for k in ("high", "low", "mid"):
        if abs(spx_close - omar[k]) / r <= threshold:
            return True
    return False


def _forward_move(
    spx_close_array: np.ndarray, minute: int, direction: str, horizon: int
) -> tuple[float, float]:
    """Return (mfe_bps, mae_bps) over [minute+1, minute+horizon] relative to entry close."""
    n = len(spx_close_array)
    end = min(minute + horizon + 1, n)
    if end <= minute + 1:
        return 0.0, 0.0
    entry = spx_close_array[minute]
    if entry <= 0:
        return 0.0, 0.0
    window = spx_close_array[minute + 1: end]
    if direction == "call":
        mfe = float(np.max((window - entry) / entry) * 10000)
        mae = float(np.min((window - entry) / entry) * 10000)
    else:  # put
        mfe = float(np.max((entry - window) / entry) * 10000)
        mae = float(np.min((entry - window) / entry) * 10000)
    return mfe, mae


def main() -> int:
    print("Loading data...")
    ds = V2Dataset.load()
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"

    print("Loading SPX bars, OMAR, SPY VWAP...")
    spx_bars = _build_spx_bars(str(spx_path))
    omar_map = _build_omar_map(spx_bars)
    spy_vwap = _build_spy_vwap(str(spy_path))

    sessions = _load_per_session(_latest_attribution_file())
    # Index oracle bars by (day, minute) with outcome + direction
    oracle_index: dict[tuple[str, int], dict] = {}
    for s in sessions:
        oracle_index[(s["day"], s["oracle_bar"])] = s
    print(f"  {len(sessions)} sessions")

    # Stats
    stats = {
        "raw_triggers": 0,           # trigger + squeeze only
        "filtered_triggers": 0,      # all 5 gates
        "days_with_filtered": set(),
    }
    # Raw trigger records for forward-move comparison
    raw_records: list[dict] = []
    filtered_records: list[dict] = []
    # Oracle-coincidence counters
    coincidence_raw = defaultdict(int)
    coincidence_filtered = defaultdict(int)
    # Direction-match tracker (does the trigger direction align with oracle direction?)
    filtered_by_outcome: dict[str, dict[str, int]] = defaultdict(lambda: {"dir_match": 0, "dir_mismatch": 0})

    sorted_days = sorted(spx_bars.keys())

    for day in sorted_days:
        arrs = spx_bars[day]
        n_bars = len(arrs["high"])
        if n_bars <= 50:
            continue
        f15 = ds.first15_by_day.get(day)
        if not f15:
            continue
        f15_h, f15_l = f15
        if f15_h <= 0 or f15_l <= 0:
            continue
        omar = omar_map.get(day)
        if omar is None:
            continue
        spy_day = spy_vwap.get(day)
        if spy_day is None:
            continue

        # Walk bars [40, 120]
        for m in range(40, min(121, n_bars)):
            spx_close = float(arrs["close"][m])
            if spx_close <= 0:
                continue

            # Eligibility: close inside first15 range
            if not (f15_l <= spx_close <= f15_h):
                continue

            # Last-10-bar H/L and range
            lo = max(m - 10, 0)
            last10_high = float(arrs["high"][lo:m].max()) if m > lo else spx_close
            last10_low = float(arrs["low"][lo:m].min()) if m > lo else spx_close
            last10_range = last10_high - last10_low

            # Trigger
            breaks_high = spx_close > last10_high
            breaks_low = spx_close < last10_low
            if not (breaks_high or breaks_low):
                continue
            direction = "call" if breaks_high else "put"

            # Squeeze gate
            squeeze_ok = last10_range <= 1.0 * omar["range"]
            if not squeeze_ok:
                continue

            # At this point, raw trigger (trigger + squeeze). Record.
            stats["raw_triggers"] += 1
            mfe20, mae20 = _forward_move(arrs["close"], m, direction, horizon=20)
            raw_rec = {
                "day": day, "minute": m, "direction": direction,
                "mfe20": mfe20, "mae20": mae20,
            }
            raw_records.append(raw_rec)

            # Cross-reference with oracle
            oracle = oracle_index.get((day, m))
            if oracle:
                coincidence_raw[oracle["outcome"]] += 1

            # Direction gate
            sigma = _sigma_pos(spy_day, m, spx_close)
            if sigma is None:
                continue
            if direction == "call" and sigma > 0.5:
                continue
            if direction == "put" and sigma < -0.5:
                continue

            # Retest gate
            if not _retest_pass(spx_close, omar, threshold=0.5):
                continue

            # All gates passed. Filtered trigger.
            stats["filtered_triggers"] += 1
            stats["days_with_filtered"].add(day)
            mfe5, mae5 = _forward_move(arrs["close"], m, direction, horizon=5)
            mfe10, mae10 = _forward_move(arrs["close"], m, direction, horizon=10)
            mfe15, mae15 = _forward_move(arrs["close"], m, direction, horizon=15)
            mfe30, mae30 = _forward_move(arrs["close"], m, direction, horizon=30)
            filt_rec = {
                "day": day, "minute": m, "direction": direction,
                "mfe5": mfe5, "mae5": mae5, "mfe10": mfe10, "mae10": mae10,
                "mfe15": mfe15, "mae15": mae15, "mfe20": mfe20, "mae20": mae20,
                "mfe30": mfe30, "mae30": mae30,
            }
            filtered_records.append(filt_rec)

            if oracle:
                coincidence_filtered[oracle["outcome"]] += 1
                oracle_dir = oracle.get("oracle_direction")
                if oracle_dir in ("call", "put"):
                    if direction == oracle_dir:
                        filtered_by_outcome[oracle["outcome"]]["dir_match"] += 1
                    else:
                        filtered_by_outcome[oracle["outcome"]]["dir_mismatch"] += 1

    # Report
    n_days = len(sorted_days)
    print()
    print("=== Trigger density ===")
    print(f"  raw (trigger + squeeze) triggers total:  {stats['raw_triggers']:>5}"
          f"  ({stats['raw_triggers'] / max(n_days, 1):>6.2f} per day)")
    print(f"  filtered (all 5 gates) triggers total:   {stats['filtered_triggers']:>5}"
          f"  ({stats['filtered_triggers'] / max(n_days, 1):>6.2f} per day)")
    print(f"  days with at least one filtered trigger: {len(stats['days_with_filtered']):>5} / {n_days}")

    # Coincidence with oracle bars
    print()
    print("=== Oracle bar coincidence ===")
    counts = {"entered_right": 117, "abstention": 467, "side_error": 402}  # baseline reference
    print(f"{'outcome':<16}{'baseline':>10}{'raw_trig':>12}{'raw_capture':>14}{'filt_trig':>12}{'filt_capture':>14}")
    print("-" * 80)
    for outcome, baseline in counts.items():
        raw_c = coincidence_raw.get(outcome, 0)
        flt_c = coincidence_filtered.get(outcome, 0)
        raw_pct = 100 * raw_c / max(baseline, 1)
        flt_pct = 100 * flt_c / max(baseline, 1)
        print(f"  {outcome:<14}{baseline:>10}{raw_c:>12}{raw_pct:>13.1f}%"
              f"{flt_c:>12}{flt_pct:>13.1f}%")

    # Direction-match on filtered triggers at oracle bars
    print()
    print("=== Direction match on filtered triggers at oracle bars ===")
    for outcome in ("entered_right", "abstention", "side_error"):
        d = filtered_by_outcome.get(outcome, {"dir_match": 0, "dir_mismatch": 0})
        total = d["dir_match"] + d["dir_mismatch"]
        if total == 0:
            continue
        match_pct = 100 * d["dir_match"] / total
        print(f"  {outcome:<16}: match={d['dir_match']}/{total} ({match_pct:.1f}%)")

    # Forward-move quality
    print()
    print("=== Forward-move quality: filtered triggers ===")
    if filtered_records:
        f = np.array([r["mfe10"] for r in filtered_records])
        a = np.array([r["mae10"] for r in filtered_records])
        print(f"  n={len(filtered_records)}")
        for h in ("mfe5", "mfe10", "mfe15", "mfe20", "mfe30"):
            v = np.array([r[h] for r in filtered_records])
            print(f"    {h:>6} median={np.median(v):+6.1f} bps   p75={np.percentile(v, 75):+6.1f}   p25={np.percentile(v, 25):+6.1f}")
        for h in ("mae5", "mae10", "mae15", "mae20", "mae30"):
            v = np.array([r[h] for r in filtered_records])
            print(f"    {h:>6} median={np.median(v):+6.1f} bps   p25={np.percentile(v, 25):+6.1f}   p75={np.percentile(v, 75):+6.1f}")

    print()
    print("=== Forward-move quality: raw (trigger+squeeze only) triggers — for comparison ===")
    if raw_records:
        v = np.array([r["mfe20"] for r in raw_records])
        print(f"  n={len(raw_records)}")
        print(f"    mfe20 median={np.median(v):+6.1f} bps   p75={np.percentile(v, 75):+6.1f}   p25={np.percentile(v, 25):+6.1f}")
        v = np.array([r["mae20"] for r in raw_records])
        print(f"    mae20 median={np.median(v):+6.1f} bps   p25={np.percentile(v, 25):+6.1f}   p75={np.percentile(v, 75):+6.1f}")

    # Headline: edge gain from filter
    print()
    print("=== Edge comparison: filtered vs raw (both at mfe20) ===")
    if filtered_records and raw_records:
        f_mfe = np.array([r["mfe20"] for r in filtered_records])
        r_mfe = np.array([r["mfe20"] for r in raw_records])
        print(f"  raw      mfe20 median={np.median(r_mfe):+6.1f}  mean={np.mean(r_mfe):+6.1f}  n={len(r_mfe)}")
        print(f"  filtered mfe20 median={np.median(f_mfe):+6.1f}  mean={np.mean(f_mfe):+6.1f}  n={len(f_mfe)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
