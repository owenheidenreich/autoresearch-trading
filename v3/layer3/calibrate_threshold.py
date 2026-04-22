"""Honest prior-window threshold calibration for rolling Layer-3 reports.

Takes an existing rolling_layer3_report.json and its per-threshold per-trade
CSVs, and builds a "calibrated" trade set where the exit threshold for each
rolling window W is chosen using only windows 0..W-1's OOS replays. This
removes the same-OOS threshold-picking leak that the existing exploratory
best suffers from.

Algorithm:
  For each window W (in sorted order):
    if W == 0: fall back to the grid midpoint (no prior data)
    else:
      for each threshold t in the grid:
        PF_t = aggregate PF on windows 0..W-1 OOS rows at threshold t
      pick threshold with max PF_t (tiebreak: higher mean PnL, then lower t)
    take window W's rows from the chosen threshold's CSV
  concatenate all selected rows and compute aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.layer2.common import save_json
from v3.layer3.common import agg_exit_metrics


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _load_per_threshold_rows(run_dir: str, thresholds: list[float]) -> dict[float, pd.DataFrame]:
    frames: dict[float, pd.DataFrame] = {}
    for thr in thresholds:
        path = os.path.join(run_dir, f"layer3_trades_thr_{_threshold_tag(thr)}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing per-threshold CSV: {path}")
        df = pd.read_csv(path)
        df["_threshold"] = float(thr)
        frames[float(thr)] = df
    return frames


def _pf(rows: pd.DataFrame) -> tuple[float, float]:
    if rows.empty:
        return 0.0, 0.0
    pnl = rows["exit_pnl"].to_numpy(dtype=np.float64)
    wins = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl <= 0].sum())
    pf = wins / losses if losses > 0 else float("inf")
    return pf, float(pnl.mean())


def calibrate_per_window(
    frames: dict[float, pd.DataFrame],
    equity: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    thresholds = sorted(frames.keys())
    windows = sorted(
        {int(w) for df in frames.values() for w in df["window_idx"].unique()}
    )
    default_thr = thresholds[len(thresholds) // 2]

    chosen_rows: list[pd.DataFrame] = []
    per_window_log: list[dict[str, Any]] = []

    for window_idx in windows:
        prior_windows = [w for w in windows if w < window_idx]
        if not prior_windows:
            chosen_thr = default_thr
            calib_note = "fold0_default_mid_grid"
            calib_pf = None
            calib_trades = 0
        else:
            best_thr = None
            best_key: tuple[float, float, float] | None = None
            calib_pf_at_best = 0.0
            calib_trades_at_best = 0
            for thr in thresholds:
                df = frames[thr]
                prior_rows = df[df["window_idx"].isin(prior_windows)]
                pf, mean_pnl = _pf(prior_rows)
                # Prefer higher PF; then higher mean PnL; then lower threshold
                # (more selective on exits costs fewer bars of theta decay).
                key = (pf, mean_pnl, -thr)
                if best_key is None or key > best_key:
                    best_key = key
                    best_thr = thr
                    calib_pf_at_best = pf
                    calib_trades_at_best = int(len(prior_rows))
            chosen_thr = float(best_thr) if best_thr is not None else default_thr
            calib_note = "prior_window_max_pf"
            calib_pf = calib_pf_at_best
            calib_trades = calib_trades_at_best

        win_rows = frames[chosen_thr]
        win_rows = win_rows[win_rows["window_idx"] == window_idx].copy()
        chosen_rows.append(win_rows)
        per_window_log.append(
            {
                "window_idx": int(window_idx),
                "chosen_threshold": float(chosen_thr),
                "calibration_source": calib_note,
                "calibration_pf_on_prior": (
                    float(calib_pf) if calib_pf is not None and np.isfinite(calib_pf) else None
                ),
                "calibration_n_trades_on_prior": int(calib_trades),
                "oos_trades": int(len(win_rows)),
                "oos_pnl": float(win_rows["exit_pnl"].sum()) if not win_rows.empty else 0.0,
            }
        )

    if not chosen_rows:
        return pd.DataFrame(), per_window_log
    combined = pd.concat(chosen_rows, ignore_index=True)
    return combined, per_window_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Existing rolling L3 artifact directory")
    p.add_argument("--equity", type=float, default=25_000.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_path = os.path.join(args.run_dir, "rolling_layer3_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"No rolling L3 report at {report_path}")
    with open(report_path) as f:
        report = json.load(f)

    thresholds = [float(t) for t in report["meta"]["exit_thresholds"]]
    frames = _load_per_threshold_rows(args.run_dir, thresholds)

    calibrated_df, per_window_log = calibrate_per_window(frames, args.equity)
    calibrated_metrics = agg_exit_metrics(calibrated_df, args.equity)

    exploratory_best = report.get("best_threshold_by_aggregate_pf_exploratory", {})
    exp_thr = (
        float(exploratory_best.get("threshold"))
        if exploratory_best and "threshold" in exploratory_best
        else None
    )
    exp_pf = (
        float(exploratory_best.get("overall", {}).get("pf", 0.0))
        if exploratory_best
        else None
    )
    baseline = report.get("entry_time_stop_baseline", {})

    print(f"Run dir: {args.run_dir}", flush=True)
    print(
        f"Baseline entry+time-stop: PF={baseline.get('pf', 0):.3f} "
        f"DD={baseline.get('max_dd_pct', 0):.1f}% mean=${baseline.get('mean_pnl', 0):+.1f}",
        flush=True,
    )
    if exp_thr is not None:
        print(
            f"Exploratory best (same-OOS threshold picking): thr={exp_thr:.2f} PF={exp_pf:.3f}",
            flush=True,
        )
    n_trades = int(calibrated_metrics.get("trades", 0))
    print(
        f"Calibrated (per-window prior-OOS threshold):    "
        f"PF={calibrated_metrics['pf']:.3f} DD={calibrated_metrics['max_dd_pct']:.1f}% "
        f"mean=${calibrated_metrics['mean_pnl']:+.1f} trades={n_trades}",
        flush=True,
    )
    print("Per-window calibration log:", flush=True)
    for w in per_window_log:
        note = w["calibration_source"]
        calib_pf = w["calibration_pf_on_prior"]
        calib_pf_str = f"{calib_pf:.3f}" if calib_pf is not None else "N/A"
        print(
            f"  W{w['window_idx']:2d}: thr={w['chosen_threshold']:.2f} "
            f"({note}, prior PF={calib_pf_str} on {w['calibration_n_trades_on_prior']} trades)  "
            f"OOS trades={w['oos_trades']} PnL=${w['oos_pnl']:+.0f}",
            flush=True,
        )

    out_payload = {
        "meta": {
            "source_report": report_path,
            "thresholds_available": thresholds,
            "calibration_policy": "per_window_prior_oos_max_pf",
            "fold0_fallback_threshold": "grid_midpoint",
        },
        "calibrated_metrics": calibrated_metrics,
        "exploratory_best": exploratory_best,
        "entry_time_stop_baseline": baseline,
        "per_window_calibration": per_window_log,
    }
    out_path = os.path.join(args.run_dir, "rolling_layer3_calibrated.json")
    save_json(out_path, out_payload)
    print(f"Saved: {out_path}", flush=True)
    if not calibrated_df.empty:
        csv_path = os.path.join(args.run_dir, "layer3_trades_calibrated.csv")
        calibrated_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
