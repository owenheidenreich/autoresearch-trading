"""Side-error deep-dive: what distinguishes true breakouts from false ones?

402 sessions where our teachers fired on the oracle bar but in the opposite
direction of what oracle says was best. 82% of those had ORC's gate met
(break + VWAP agree) in the wrong direction. These are "false breakouts"
— the classic problem in breakout trading.

Question: what feature configuration at the oracle bar distinguishes
"true breakouts that kept running" (entered_right) from "false breakouts
that reversed" (side_error)?

Candidate distinguishing features:

1. `breakout_confirmation` (feature 47 in data.pt): combines break of prev-session
   high/IB-high with volume_ratio > 1.05 and positive bar_delta. Positive = up
   break with volume+momentum support. Pre-packaged "is this a real break?"
2. `volume_ratio`: raw volume on the breakout bar
3. `vwap_dist` magnitude: how far from VWAP is the break happening?
4. `first15_range_pct`: wide opening range (strong day?) vs compressed (chop day?)
5. `bars_since_break_above/below`: how recent was the first-15 break?
6. VWAP σ-position (SPY-derived)
7. Intraday VP proximity: is the break happening near POC/VAH/VAL? Breaks at
   confluence (walls) may reverse; breaks away from them may continue
8. atm_iv: low-IV day vs high-IV day
9. Overnight gap: does gap direction predict continuation or reversal?

For each oracle bar, run ORC/FailedBreak teachers to determine the fired
direction, then bucket by (fired_direction, correct/wrong) and compare
features.
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

from v3.harness.v2_adapter import V2Dataset, load_day_sidecar, has_chain_snapshot
from v3.teachers.base import BarContext, TeacherAction
from v3.teachers.failed_break import FailedBreakTeacher
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


def _build_omar_map(spx_1min_path: str) -> dict[str, dict[str, float]]:
    df = pickle.load(open(spx_1min_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, float]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        h = float(g.iloc[0]["spx_high"])
        l = float(g.iloc[0]["spx_low"])
        out[day] = {
            "high": h,
            "low": l,
            "mid": (h + l) / 2.0,
            "range": max(h - l, 0.01),
        }
    return out


def _build_spy_vwap(spy_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Per day: arrays of per-minute vwap, close, std (for σ-position)."""
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
    """Return 'call', 'put', or 'none' based on ORC's evaluation."""
    a = orc.evaluate(bar_ctx)
    if a == TeacherAction.BUY_CALL:
        return "call"
    if a == TeacherAction.BUY_PUT:
        return "put"
    return "none"


def main() -> int:
    print("Loading data...")
    ds = V2Dataset.load()
    spx_path = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"

    omar_map = _build_omar_map(str(spx_path))
    print("Computing SPY VWAP bands...")
    spy_vwap = _build_spy_vwap(str(spy_path))

    sessions = _load_per_session(_latest_attribution_file())
    entered = [s for s in sessions if s.get("outcome") == "entered_right"]
    side_err = [s for s in sessions if s.get("outcome") == "side_error"]

    orc = ORCTeacher()
    # Feature indices
    idx_vwap_dist = ds.idx.get("vwap_dist")
    idx_vwap_slope = ds.idx.get("vwap_slope")
    idx_volume_ratio = ds.idx.get("volume_ratio")
    idx_f15 = ds.idx.get("first15_range_pct")
    idx_atm_iv = ds.idx.get("atm_iv")
    idx_bars_above = ds.idx.get("bars_since_break_above_first15")
    idx_bars_below = ds.idx.get("bars_since_break_below_first15")
    # breakout_confirmation name
    names = ds.feature_names
    idx_bo_conf = names.index("breakout_confirmation") if "breakout_confirmation" in names else None

    buckets: dict[str, list[dict]] = defaultdict(list)

    for cohort_name, rows in (("entered_right", entered), ("side_error", side_err)):
        for s in rows:
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
            omar = omar_map.get(day)
            if omar is None:
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
            orc_fired = _orc_would_fire(ctx, orc)

            # Classify
            if cohort_name == "entered_right":
                # ORC (or failed_break) got it right — want to only include ORC wins here
                if orc_fired == oracle_dir:
                    bucket_key = f"ORC_{oracle_dir}_correct"
                else:
                    continue  # not an ORC-correct case
            else:  # side_error
                if orc_fired not in ("call", "put"):
                    continue  # ORC didn't fire; skip (another teacher fired wrong)
                if orc_fired == oracle_dir:
                    continue  # shouldn't happen in side_error
                bucket_key = f"ORC_{orc_fired}_wrong"  # ORC fired this direction; oracle says opposite

            # Derived features
            spy_day = spy_vwap.get(day)
            sigma_pos = (
                _vwap_sigma_position(spy_day, minute, spx_close) if spy_day else None
            )
            abs_delta_from_vwap = abs(vwap_dist)
            # Overnight gap: yesterday's close to today's session open (bar 0 of day)
            overnight_gap_pct: float | None = None
            if day_start > 0:
                yest_close = float(ds.spot_prices[day_start - 1])
                today_open = float(ds.spot_prices[day_start])
                if yest_close > 0 and today_open > 0:
                    overnight_gap_pct = (today_open - yest_close) / yest_close

            breakout_confirm = float(row[idx_bo_conf]) if idx_bo_conf is not None else 0.0

            buckets[bucket_key].append({
                "breakout_confirm": breakout_confirm,
                "volume_ratio": ctx.volume_ratio,
                "vwap_dist_abs": abs_delta_from_vwap,
                "vwap_slope_abs": abs(vwap_slope),
                "first15_range_pct": ctx.first15_range_pct,
                "bars_since_above": ctx.bars_since_break_above_first15,
                "bars_since_below": ctx.bars_since_break_below_first15,
                "atm_iv": float(row[idx_atm_iv]) if idx_atm_iv else 0.0,
                "sigma_pos": sigma_pos if sigma_pos is not None else 0.0,
                "overnight_gap_pct": overnight_gap_pct if overnight_gap_pct is not None else 0.0,
                "oracle_direction": oracle_dir,
                "orc_fired": orc_fired,
            })

    # Print per-bucket size
    print()
    print("=== Bucket sizes ===")
    for k in ("ORC_call_correct", "ORC_put_correct", "ORC_call_wrong", "ORC_put_wrong"):
        print(f"  {k:<22}: {len(buckets.get(k, []))}")
    print()

    # --- Core comparison: ORC_correct vs ORC_wrong for each direction ---
    features_to_compare = [
        "breakout_confirm",
        "volume_ratio",
        "vwap_dist_abs",
        "vwap_slope_abs",
        "first15_range_pct",
        "atm_iv",
        "sigma_pos",
        "overnight_gap_pct",
    ]

    for direction in ("call", "put"):
        correct_key = f"ORC_{direction}_correct"
        wrong_key = f"ORC_{direction}_wrong"
        correct = buckets.get(correct_key, [])
        wrong = buckets.get(wrong_key, [])
        print(f"=== ORC {direction.upper()}: correct vs wrong ===")
        print(
            f"  n_correct = {len(correct)}, n_wrong = {len(wrong)}, "
            f"wrong_rate = {len(wrong)/(len(correct)+len(wrong)):.1%}"
        )
        print()
        header = f"{'feature':<24}{'correct_med':>14}{'wrong_med':>14}{'ratio':>10}"
        print(header)
        print("-" * len(header))
        for feat in features_to_compare:
            cv = np.asarray([r[feat] for r in correct])
            wv = np.asarray([r[feat] for r in wrong])
            if cv.size == 0 or wv.size == 0:
                continue
            c_med = float(np.median(cv))
            w_med = float(np.median(wv))
            ratio = c_med / w_med if w_med != 0 else float("nan")
            print(
                f"{feat:<24}{c_med:>+14.4f}{w_med:>+14.4f}{ratio:>+10.2f}"
            )
        # For signed features (direction-relevant), also show sign-aware
        # breakout_confirm interpretation
        print()
        bo_correct = np.asarray([r["breakout_confirm"] for r in correct])
        bo_wrong = np.asarray([r["breakout_confirm"] for r in wrong])
        print(
            f"  breakout_confirm stats (positive = up-break confirmed, negative = down-break confirmed):"
        )
        print(
            f"    correct: p10={np.percentile(bo_correct, 10):+.3f} "
            f"median={np.median(bo_correct):+.3f} "
            f"p90={np.percentile(bo_correct, 90):+.3f}"
        )
        print(
            f"    wrong:   p10={np.percentile(bo_wrong, 10):+.3f} "
            f"median={np.median(bo_wrong):+.3f} "
            f"p90={np.percentile(bo_wrong, 90):+.3f}"
        )
        # Sign alignment: for correct call, should be positive. For wrong call
        # (oracle says put was right), was breakout_confirm still positive?
        # If yes -> confirmation feature agrees with ORC's (wrong) direction
        # If negative -> confirmation DISAGREED with ORC — would've been a useful filter
        if direction == "call":
            correct_aligned = np.mean(bo_correct > 0)
            wrong_aligned = np.mean(bo_wrong > 0)
        else:
            correct_aligned = np.mean(bo_correct < 0)
            wrong_aligned = np.mean(bo_wrong < 0)
        print(
            f"  % where breakout_confirm SIGN AGREES with ORC's direction: "
            f"correct={correct_aligned:.1%} vs wrong={wrong_aligned:.1%}"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
