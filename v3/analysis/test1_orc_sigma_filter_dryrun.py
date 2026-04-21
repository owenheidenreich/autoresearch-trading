"""Test 1: A1 dry-run — replay the proposed sigma_pos filter on existing ORC fires.

Question: if we apply the research-proposed filter
  - BUY_CALL requires sigma_pos <= 0.0
  - BUY_PUT  requires sigma_pos >= 0.0
to ORC fires that actually occurred at oracle bars across the 986-day cache,
how much side_error do we drop? How much entered_right do we keep?

Baseline (from side_error_deep_dive):
  - 104 ORC_correct  (52 call + 52 put)
  - 330 ORC_wrong    (171 call + 159 put)

Expected from medians:
  - ORC_call_correct median sigma = -0.26 -> most kept
  - ORC_call_wrong   median sigma = +0.14 -> most dropped
  - ORC_put_correct  median sigma = +0.09 -> most kept
  - ORC_put_wrong    median sigma = -0.31 -> most dropped

Success: drop majority of 330 wrong, keep most of 104 correct. Sweep
candidate thresholds (0.0, ±0.1, ±0.2) to see sensitivity.
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
from v3.teachers.base import BarContext, TeacherAction
from v3.teachers.orc import ORCTeacher


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


def _vwap_sigma_position(
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


def _orc_would_fire(bar_ctx: BarContext, orc: ORCTeacher) -> str:
    a = orc.evaluate(bar_ctx)
    if a == TeacherAction.BUY_CALL:
        return "call"
    if a == TeacherAction.BUY_PUT:
        return "put"
    return "none"


def main() -> int:
    print("Loading data...")
    ds = V2Dataset.load()
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"
    print("Computing SPY VWAP...")
    spy_vwap = _build_spy_vwap(str(spy_path))

    sessions = _load_per_session(_latest_attribution_file())
    print(f"  loaded {len(sessions)} sessions")

    orc = ORCTeacher()
    idx_vwap_dist = ds.idx.get("vwap_dist")
    idx_vwap_slope = ds.idx.get("vwap_slope")
    idx_volume_ratio = ds.idx.get("volume_ratio")
    idx_bars_above = ds.idx.get("bars_since_break_above_first15")
    idx_bars_below = ds.idx.get("bars_since_break_below_first15")
    idx_f15 = ds.idx.get("first15_range_pct")

    # Build per-bar records for sessions where ORC fires at oracle bar
    # Buckets: ORC_call_correct, ORC_call_wrong, ORC_put_correct, ORC_put_wrong
    # Also track original outcome for reclassification.
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    # Each entry: (original_outcome, sigma_pos)

    total_processed = 0
    orc_fires_at_oracle = 0
    for s in sessions:
        if s.get("outcome") not in ("entered_right", "side_error", "abstention"):
            continue
        day = s["day"]
        minute = s["oracle_bar"]
        oracle_dir = s.get("oracle_direction")
        if oracle_dir not in ("call", "put"):
            continue
        try:
            day_start, day_end = ds.day_bar_range(day)
        except ValueError:
            continue
        abs_idx = day_start + minute
        if abs_idx >= day_end:
            continue
        f15_h, f15_l = ds.first15_by_day.get(day, (0.0, 0.0))
        if f15_h <= 0 or f15_l <= 0:
            continue
        spx_close = float(ds.spot_prices[abs_idx])
        row = ds.X_sim[abs_idx]
        vwap_dist = float(row[idx_vwap_dist]) if idx_vwap_dist is not None else 0.0
        vwap_slope = float(row[idx_vwap_slope]) if idx_vwap_slope is not None else 0.0
        vwap = spx_close * (1.0 - vwap_dist) if spx_close > 0 else spx_close

        ctx = BarContext(
            minute_of_session=minute,
            close=spx_close,
            vwap=vwap,
            vwap_slope=vwap_slope,
            volume_ratio=float(row[idx_volume_ratio]) if idx_volume_ratio else 1.0,
            first15_high=f15_h,
            first15_low=f15_l,
            first15_range_pct=float(row[idx_f15]) if idx_f15 else 0.0,
            bars_since_break_above_first15=int(row[idx_bars_above]) if idx_bars_above else -1,
            bars_since_break_below_first15=int(row[idx_bars_below]) if idx_bars_below else -1,
        )
        total_processed += 1
        orc_fired = _orc_would_fire(ctx, orc)
        if orc_fired == "none":
            continue
        orc_fires_at_oracle += 1

        spy_day = spy_vwap.get(day)
        if spy_day is None:
            continue
        sigma = _vwap_sigma_position(spy_day, minute, spx_close)
        if sigma is None:
            continue

        outcome = s["outcome"]
        # Bucket: ORC direction + correct/wrong
        if orc_fired == oracle_dir:
            key = f"ORC_{orc_fired}_correct"
        else:
            key = f"ORC_{orc_fired}_wrong"
        buckets[key].append((outcome, sigma))

    print(f"\n  sessions processed: {total_processed}")
    print(f"  ORC fires at oracle bar: {orc_fires_at_oracle}")
    print()

    # Baseline bucket counts
    print("=== Baseline bucket counts (ORC fires at oracle bar) ===")
    for key in ("ORC_call_correct", "ORC_call_wrong", "ORC_put_correct", "ORC_put_wrong"):
        rows = buckets.get(key, [])
        sigs = [s for _, s in rows]
        med = np.median(sigs) if sigs else float("nan")
        print(f"  {key:<22} n={len(rows):>4}  median_sigma={med:+.3f}")

    # Threshold sweep
    print()
    print("=== Threshold sweep: CALL requires sigma <= threshold, PUT requires sigma >= -threshold ===")
    print(f"{'threshold':<12}{'kept_correct':>14}{'kept_wrong':>14}"
          f"{'dropped_correct':>18}{'dropped_wrong':>16}{'new_side_error':>18}"
          f"{'new_entered_right':>20}")
    print("-" * 112)

    for thresh in (0.0, -0.1, -0.2, 0.1, 0.2):
        # CALL filter: sigma <= thresh (more negative threshold = stricter)
        # PUT  filter: sigma >= -thresh (symmetric)
        kept_correct = 0
        kept_wrong = 0
        for outcome, sigma in buckets["ORC_call_correct"]:
            if sigma <= thresh:
                kept_correct += 1
        for outcome, sigma in buckets["ORC_put_correct"]:
            if sigma >= -thresh:
                kept_correct += 1
        for outcome, sigma in buckets["ORC_call_wrong"]:
            if sigma <= thresh:
                kept_wrong += 1
        for outcome, sigma in buckets["ORC_put_wrong"]:
            if sigma >= -thresh:
                kept_wrong += 1
        n_correct = len(buckets["ORC_call_correct"]) + len(buckets["ORC_put_correct"])
        n_wrong = len(buckets["ORC_call_wrong"]) + len(buckets["ORC_put_wrong"])
        dropped_correct = n_correct - kept_correct
        dropped_wrong = n_wrong - kept_wrong
        # Projected new attribution:
        # entered_right now = kept_correct (others become abstention after filter)
        # side_error now = kept_wrong
        # dropped correct -> abstention (ORC no longer fires -> if late-session teacher catches, becomes entered_right)
        # dropped wrong -> abstention (ORC no longer fires -> was side_error, now no ORC signal)
        print(f"{thresh:<+12.2f}{kept_correct:>14}{kept_wrong:>14}"
              f"{dropped_correct:>18}{dropped_wrong:>16}{kept_wrong:>18}"
              f"{kept_correct:>20}")

    # Summary at threshold 0.0 (the proposed filter)
    print()
    print("=== Summary at proposed threshold 0.0 ===")
    baseline_correct = len(buckets["ORC_call_correct"]) + len(buckets["ORC_put_correct"])
    baseline_wrong = len(buckets["ORC_call_wrong"]) + len(buckets["ORC_put_wrong"])
    kc = sum(1 for _, s in buckets["ORC_call_correct"] if s <= 0)
    kc += sum(1 for _, s in buckets["ORC_put_correct"] if s >= 0)
    kw = sum(1 for _, s in buckets["ORC_call_wrong"] if s <= 0)
    kw += sum(1 for _, s in buckets["ORC_put_wrong"] if s >= 0)
    print(f"  baseline:  correct={baseline_correct}, wrong={baseline_wrong}")
    print(f"  after filter: correct={kc} ({100*kc/max(baseline_correct,1):.1f}% retained)"
          f", wrong={kw} ({100*kw/max(baseline_wrong,1):.1f}% retained)")
    print(f"  side_error reduction at oracle bar: {baseline_wrong - kw} fewer "
          f"({100*(baseline_wrong-kw)/max(baseline_wrong,1):.1f}% dropped)")
    print(f"  entered_right loss at oracle bar:   {baseline_correct - kc} fewer "
          f"({100*(baseline_correct-kc)/max(baseline_correct,1):.1f}% dropped)")

    # Net impact interpretation
    print()
    print("=== Net impact on full 986-day attribution ===")
    print("  (ORC-fires-at-oracle subset; other 72 side_error sessions had ORC fire elsewhere)")
    baseline_side_err_total = 402
    baseline_entered_right_total = 117  # approx; read from attribution if needed
    projected_side_err = baseline_side_err_total - (baseline_wrong - kw)
    projected_entered_right = baseline_entered_right_total - (baseline_correct - kc)
    print(f"  projected side_error:    402 -> {projected_side_err} "
          f"({-((baseline_wrong-kw)):+d})")
    print(f"  projected entered_right: 117 -> {projected_entered_right} "
          f"({-((baseline_correct-kc)):+d})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
