"""Walk-forward evaluation of the ORC sigma-position direction gate.

The original sigma-position finding is compelling, but the first dry-run was
still a full-sample read. This script keeps the same measurement target
("bars where ORC actually fires at the oracle bar") and reports threshold
behavior across chronological folds.

Question:
    Does the proposed ORC gate
        CALL requires sigma_pos <= threshold
        PUT  requires sigma_pos >= -threshold
    behave consistently out of sample?

This is still not a live PnL test. It is the right next confirmation test for
the specific claim made in `side_error_deep_dive_2026_04_20.md`.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from v3.harness.v2_adapter import V2Dataset
from v3.teachers.base import BarContext, TeacherAction
from v3.teachers.orc import ORCTeacher


THRESHOLDS = (-0.2, -0.1, 0.0, 0.1, 0.2)
N_FOLDS = 5


def _latest_attribution_file() -> str:
    files = sorted(glob.glob("v3/reference/attribution_full_*.txt"))
    if not files:
        raise FileNotFoundError("No attribution file found in v3/reference/")
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
    if minute < 0 or minute >= len(spy_day["vwap"]):
        return None
    spy_vwap = float(spy_day["vwap"][minute])
    spy_close = float(spy_day["close"][minute])
    spy_std = float(spy_day["std"][minute])
    if not all(v > 0 for v in (spy_vwap, spy_close, spy_std, spx_close)):
        return None
    ratio = spx_close / spy_close
    return (spx_close - spy_vwap * ratio) / (spy_std * ratio)


def _orc_would_fire(bar_ctx: BarContext, orc: ORCTeacher) -> str:
    action = orc.evaluate(bar_ctx)
    if action == TeacherAction.BUY_CALL:
        return "call"
    if action == TeacherAction.BUY_PUT:
        return "put"
    return "none"


def _threshold_pass(direction: str, sigma: float, threshold: float) -> bool:
    if direction == "call":
        return sigma <= threshold
    return sigma >= -threshold


def main() -> int:
    ds = V2Dataset.load()
    spy_path = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"
    spy_vwap = _build_spy_vwap(str(spy_path))
    sessions = _load_per_session(_latest_attribution_file())

    orc = ORCTeacher()
    idx_vwap_dist = ds.idx.get("vwap_dist")
    idx_vwap_slope = ds.idx.get("vwap_slope")
    idx_volume_ratio = ds.idx.get("volume_ratio")
    idx_bars_above = ds.idx.get("bars_since_break_above_first15")
    idx_bars_below = ds.idx.get("bars_since_break_below_first15")
    idx_f15 = ds.idx.get("first15_range_pct")

    records: list[dict] = []
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
        f15_high, f15_low = ds.first15_by_day.get(day, (0.0, 0.0))
        if f15_high <= 0 or f15_low <= 0:
            continue

        row = ds.X_sim[abs_idx]
        spx_close = float(ds.spot_prices[abs_idx])
        vwap_dist = float(row[idx_vwap_dist]) if idx_vwap_dist is not None else 0.0
        vwap = spx_close * (1.0 - vwap_dist) if spx_close > 0 else spx_close
        ctx = BarContext(
            minute_of_session=minute,
            close=spx_close,
            vwap=vwap,
            vwap_slope=float(row[idx_vwap_slope]) if idx_vwap_slope is not None else 0.0,
            volume_ratio=float(row[idx_volume_ratio]) if idx_volume_ratio is not None else 1.0,
            first15_high=f15_high,
            first15_low=f15_low,
            first15_range_pct=float(row[idx_f15]) if idx_f15 is not None else 0.0,
            bars_since_break_above_first15=int(row[idx_bars_above]) if idx_bars_above is not None else -1,
            bars_since_break_below_first15=int(row[idx_bars_below]) if idx_bars_below is not None else -1,
        )
        orc_dir = _orc_would_fire(ctx, orc)
        if orc_dir == "none":
            continue

        spy_day = spy_vwap.get(day)
        if spy_day is None:
            continue
        sigma = _vwap_sigma_position(spy_day, minute, spx_close)
        if sigma is None:
            continue

        records.append(
            {
                "day": day,
                "sigma": sigma,
                "orc_dir": orc_dir,
                "oracle_dir": oracle_dir,
                "correct": orc_dir == oracle_dir,
            }
        )

    records.sort(key=lambda r: r["day"])
    unique_days = sorted({r["day"] for r in records})
    day_folds = np.array_split(unique_days, N_FOLDS)

    print(f"ORC fires at oracle bar: {len(records)} records across {len(unique_days)} days")
    print()
    for i, fold_days in enumerate(day_folds, start=1):
        fold_set = set(fold_days.tolist())
        fold_rows = [r for r in records if r["day"] in fold_set]
        n_correct = sum(1 for r in fold_rows if r["correct"])
        n_wrong = len(fold_rows) - n_correct
        print(f"=== Fold {i} ({len(fold_set)} days) ===")
        print(f"baseline: correct={n_correct} wrong={n_wrong}")
        print(
            f"{'threshold':<10}{'keep_correct':>14}{'keep_wrong':>12}"
            f"{'correct_ret':>14}{'wrong_ret':>12}{'net_gap':>12}"
        )
        for threshold in THRESHOLDS:
            keep_correct = 0
            keep_wrong = 0
            for row in fold_rows:
                if not _threshold_pass(row["orc_dir"], row["sigma"], threshold):
                    continue
                if row["correct"]:
                    keep_correct += 1
                else:
                    keep_wrong += 1
            correct_ret = keep_correct / max(n_correct, 1)
            wrong_ret = keep_wrong / max(n_wrong, 1)
            net_gap = correct_ret - wrong_ret
            print(
                f"{threshold:<+10.2f}{keep_correct:>14}{keep_wrong:>12}"
                f"{100*correct_ret:>13.1f}%{100*wrong_ret:>11.1f}%{100*net_gap:>11.1f}pp"
            )
        print()

    print("=== Aggregate across all folds ===")
    print(
        f"{'threshold':<10}{'keep_correct':>14}{'keep_wrong':>12}"
        f"{'correct_ret':>14}{'wrong_ret':>12}{'net_gap':>12}"
    )
    total_correct = sum(1 for r in records if r["correct"])
    total_wrong = len(records) - total_correct
    for threshold in THRESHOLDS:
        keep_correct = 0
        keep_wrong = 0
        for row in records:
            if not _threshold_pass(row["orc_dir"], row["sigma"], threshold):
                continue
            if row["correct"]:
                keep_correct += 1
            else:
                keep_wrong += 1
        correct_ret = keep_correct / max(total_correct, 1)
        wrong_ret = keep_wrong / max(total_wrong, 1)
        net_gap = correct_ret - wrong_ret
        print(
            f"{threshold:<+10.2f}{keep_correct:>14}{keep_wrong:>12}"
            f"{100*correct_ret:>13.1f}%{100*wrong_ret:>11.1f}%{100*net_gap:>11.1f}pp"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
