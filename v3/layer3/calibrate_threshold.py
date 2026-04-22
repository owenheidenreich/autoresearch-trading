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
    policy: str = "prior_window_max_pf",
    fixed_threshold: float | None = None,
    robust_slack: float = 0.90,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Pick each window's exit threshold under the given policy.

    Policies:
      - "prior_window_max_pf": per-window, pick threshold that maximized
        aggregated PF on windows 0..W-1 OOS rows (leakage-free).
      - "prior_window_robust": same as above, but among thresholds whose
        prior PF is within `robust_slack` of the best, pick the *lowest*.
        Motivated by the W07 diagnostic: lower thresholds are more robust
        to regime shift since they exit earlier and eat less theta.
      - "fixed": use `fixed_threshold` for every window, ignoring priors.
    """
    thresholds = sorted(frames.keys())
    windows = sorted(
        {int(w) for df in frames.values() for w in df["window_idx"].unique()}
    )
    mid_thr = thresholds[len(thresholds) // 2]

    if policy == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed policy requires fixed_threshold")
        # snap to nearest available threshold in the grid
        chosen_all = min(thresholds, key=lambda t: abs(t - float(fixed_threshold)))
    elif policy not in ("prior_window_max_pf", "prior_window_robust"):
        raise ValueError(f"unknown policy: {policy}")

    chosen_rows: list[pd.DataFrame] = []
    per_window_log: list[dict[str, Any]] = []

    for window_idx in windows:
        if policy == "fixed":
            chosen_thr = chosen_all
            calib_note = f"fixed_{fixed_threshold:.2f}"
            calib_pf = None
            calib_trades = 0
        else:
            prior_windows = [w for w in windows if w < window_idx]
            if not prior_windows:
                chosen_thr = mid_thr
                calib_note = "fold0_default_mid_grid"
                calib_pf = None
                calib_trades = 0
            else:
                per_thr_pf: list[tuple[float, float, float, int]] = []
                for thr in thresholds:
                    df = frames[thr]
                    prior_rows = df[df["window_idx"].isin(prior_windows)]
                    pf, mean_pnl = _pf(prior_rows)
                    per_thr_pf.append((thr, pf, mean_pnl, int(len(prior_rows))))

                if policy == "prior_window_max_pf":
                    best = max(per_thr_pf, key=lambda x: (x[1], x[2], -x[0]))
                    chosen_thr = best[0]
                    calib_pf = best[1]
                    calib_trades = best[3]
                    calib_note = "prior_window_max_pf"
                else:  # prior_window_robust
                    best_pf = max(x[1] for x in per_thr_pf)
                    threshold_floor = best_pf * float(robust_slack)
                    eligible = [x for x in per_thr_pf if x[1] >= threshold_floor and np.isfinite(x[1])]
                    if not eligible:
                        eligible = per_thr_pf
                    # within eligibility, pick lowest threshold
                    chosen = min(eligible, key=lambda x: x[0])
                    chosen_thr = chosen[0]
                    calib_pf = chosen[1]
                    calib_trades = chosen[3]
                    calib_note = f"prior_window_robust_slack_{robust_slack}"

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
    p.add_argument(
        "--policy",
        default="prior_window_max_pf",
        choices=("prior_window_max_pf", "prior_window_robust", "fixed"),
        help="Threshold-selection policy. `prior_window_max_pf` picks the argmax-PF on prior-window OOS rows. `prior_window_robust` picks the lowest threshold within --robust-slack of the best (more regime-shift resilient). `fixed` uses --threshold for every window.",
    )
    p.add_argument("--threshold", type=float, default=None, help="Used when --policy=fixed")
    p.add_argument("--robust-slack", type=float, default=0.90, help="PF slack fraction for prior_window_robust")
    p.add_argument(
        "--out-suffix",
        default="",
        help="Suffix appended to output filenames to avoid overwriting prior runs",
    )
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

    calibrated_df, per_window_log = calibrate_per_window(
        frames,
        args.equity,
        policy=args.policy,
        fixed_threshold=args.threshold,
        robust_slack=args.robust_slack,
    )
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
            "calibration_policy": args.policy,
            "fixed_threshold": float(args.threshold) if args.threshold is not None else None,
            "robust_slack": float(args.robust_slack) if args.policy == "prior_window_robust" else None,
            "fold0_fallback_threshold": "grid_midpoint" if args.policy != "fixed" else f"fixed_{args.threshold}",
        },
        "calibrated_metrics": calibrated_metrics,
        "exploratory_best": exploratory_best,
        "entry_time_stop_baseline": baseline,
        "per_window_calibration": per_window_log,
    }
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_path = os.path.join(args.run_dir, f"rolling_layer3_calibrated{suffix}.json")
    save_json(out_path, out_payload)
    print(f"Saved: {out_path}", flush=True)
    if not calibrated_df.empty:
        csv_path = os.path.join(args.run_dir, f"layer3_trades_calibrated{suffix}.csv")
        calibrated_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
