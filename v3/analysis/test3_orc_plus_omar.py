"""Test 3: ORC + σ-filter + OMAR retest stack.

Starting from Test 1's σ-filter, add Filter C (OMAR retest) as an additional
precondition on ORC fires. Measures whether stacking C gives incremental
side_error reduction beyond what σ-filter alone achieves.

Baseline (Test 1 @ σ=0):
  - correct kept: 71/104 (68.3%)
  - wrong kept: 104/330 (31.5%)
  - projected side_error: 402 -> 176

With σ + C added, both are strictly tighter subsets. The question is
whether the ratio improves — does C drop more wrong fires than correct?
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
            "high": h, "low": l, "mid": (h + l) / 2.0,
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


def _sigma_pos(spy_day, minute, spx_close):
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
    return (spx_close - spy_vwap * ratio) / (spy_std * ratio)


def _retest_pass(spx_close, omar, threshold=0.5):
    r = omar["range"]
    if r <= 0:
        return False
    return any(abs(spx_close - omar[k]) / r <= threshold for k in ("high", "low", "mid"))


def _orc_would_fire(bar_ctx, orc):
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
    spy_vwap = _build_spy_vwap(str(spy_path))
    sessions = _load_per_session(_latest_attribution_file())

    orc = ORCTeacher()
    idx_vwap_dist = ds.idx.get("vwap_dist")
    idx_vwap_slope = ds.idx.get("vwap_slope")
    idx_volume_ratio = ds.idx.get("volume_ratio")
    idx_bars_above = ds.idx.get("bars_since_break_above_first15")
    idx_bars_below = ds.idx.get("bars_since_break_below_first15")
    idx_f15 = ds.idx.get("first15_range_pct")

    # Bucket: (original_outcome, sigma, retest_pass)
    buckets: dict[str, list[tuple[str, float, bool]]] = defaultdict(list)

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
        omar = omar_map.get(day)
        spy_day = spy_vwap.get(day)
        if omar is None or spy_day is None:
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
        if orc_fired == "none":
            continue
        sigma = _sigma_pos(spy_day, minute, spx_close)
        if sigma is None:
            continue
        retest = _retest_pass(spx_close, omar, 0.5)

        if orc_fired == oracle_dir:
            key = f"ORC_{orc_fired}_correct"
        else:
            key = f"ORC_{orc_fired}_wrong"
        buckets[key].append((s["outcome"], sigma, retest))

    n_correct = len(buckets["ORC_call_correct"]) + len(buckets["ORC_put_correct"])
    n_wrong = len(buckets["ORC_call_wrong"]) + len(buckets["ORC_put_wrong"])

    def _apply_filter(sigma_thresh: float, require_retest: bool) -> tuple[int, int]:
        kc = 0
        kw = 0
        for bucket, direction in [
            ("ORC_call_correct", "call"), ("ORC_put_correct", "put"),
            ("ORC_call_wrong", "call"), ("ORC_put_wrong", "put"),
        ]:
            for _, sigma, retest in buckets[bucket]:
                if direction == "call" and sigma > sigma_thresh:
                    continue
                if direction == "put" and sigma < -sigma_thresh:
                    continue
                if require_retest and not retest:
                    continue
                if "correct" in bucket:
                    kc += 1
                else:
                    kw += 1
        return kc, kw

    print()
    print("=== Filter stack comparison ===")
    print(f"  baseline:  correct={n_correct}, wrong={n_wrong}")
    print()
    print(f"{'filter':<30}{'kept_correct':>14}{'kept_wrong':>14}"
          f"{'correct_retention':>20}{'wrong_retention':>18}")
    print("-" * 98)

    # Case 1: no filter
    print(f"{'(none)':<30}{n_correct:>14}{n_wrong:>14}"
          f"{'100.0%':>19} {'100.0%':>17} ")
    # Case 2: σ only @ 0.0
    kc, kw = _apply_filter(0.0, require_retest=False)
    print(f"{'sigma<=0 only':<30}{kc:>14}{kw:>14}"
          f"{100 * kc / n_correct:>19.1f}% {100 * kw / n_wrong:>17.1f}%")
    # Case 3: σ + C
    kc, kw = _apply_filter(0.0, require_retest=True)
    print(f"{'sigma<=0 + OMAR retest':<30}{kc:>14}{kw:>14}"
          f"{100 * kc / n_correct:>19.1f}% {100 * kw / n_wrong:>17.1f}%")
    # Case 4: C alone (no sigma filter; sigma threshold at +inf)
    kc, kw = _apply_filter(1e9, require_retest=True)
    print(f"{'OMAR retest only':<30}{kc:>14}{kw:>14}"
          f"{100 * kc / n_correct:>19.1f}% {100 * kw / n_wrong:>17.1f}%")
    # Case 5: σ tighter + C
    kc, kw = _apply_filter(-0.1, require_retest=True)
    print(f"{'sigma<=-0.1 + OMAR retest':<30}{kc:>14}{kw:>14}"
          f"{100 * kc / n_correct:>19.1f}% {100 * kw / n_wrong:>17.1f}%")

    print()
    print("=== Projected attribution change from baselines (402 side_err, 117 entered_right) ===")
    for name, thresh, require in [
        ("sigma only", 0.0, False),
        ("sigma + retest", 0.0, True),
        ("retest only", 1e9, True),
    ]:
        kc, kw = _apply_filter(thresh, require)
        side_err_drop = n_wrong - kw
        er_drop = n_correct - kc
        print(f"  {name:<22}: side_err 402 -> {402 - side_err_drop}  ({-side_err_drop:+d})"
              f"   entered_right 117 -> {117 - er_drop}  ({-er_drop:+d})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
