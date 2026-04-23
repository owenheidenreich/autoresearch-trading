from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import pandas as pd

from v3.layer2.common import replay_metrics_from_pnls, save_json, save_pickle
from v3.layer2.train_unified_policy import (
    OBJECTIVE_PNL_COL,
    TIME_STOP_PNL_COL,
    _select_daily_trades,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reselect unified-policy chosen trades from saved OOF predictions "
            "using a global decision-margin offset on top of each window's "
            "calibrated margin."
        )
    )
    p.add_argument("--seed-dir", required=True, help="Path like v3/artifacts/.../seed_42")
    p.add_argument("--output-dir", required=True, help="Where to save the reselected trades/report")
    p.add_argument(
        "--decision-margin-offset",
        type=float,
        default=0.0,
        help="Global additive offset applied to each window's calibrated decision margin.",
    )
    p.add_argument("--equity", type=float, default=25_000.0)
    return p.parse_args()


def _load_window_margins(seed_dir: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for name in sorted(os.listdir(seed_dir)):
        if not name.startswith("window_"):
            continue
        calibration_path = os.path.join(seed_dir, name, "calibration.json")
        if not os.path.exists(calibration_path):
            continue
        with open(calibration_path) as f:
            payload = json.load(f)
        out[int(name.split("_")[1])] = float(payload["decision_margin"])
    if not out:
        raise RuntimeError(f"No per-window calibration files found under {seed_dir}")
    return out


def _trade_metrics(trades: pd.DataFrame, pnl_col: str, equity: float) -> dict[str, Any]:
    pnls = trades[pnl_col].tolist() if not trades.empty else []
    metrics = replay_metrics_from_pnls(pnls, equity)
    return {
        "trades": int(len(trades)),
        "pf": float(metrics["pf"]) if pnls else 0.0,
        "max_dd_pct": float(metrics["max_dd_pct"]),
        "mean_pnl": float(metrics["mean_pnl"]),
    }


def main() -> int:
    args = parse_args()
    seed_dir = args.seed_dir
    os.makedirs(args.output_dir, exist_ok=True)

    pred_path = os.path.join(seed_dir, "oof_predictions.pkl")
    if not os.path.exists(pred_path):
        raise RuntimeError(f"Missing OOF predictions: {pred_path}")
    pred_df = pd.read_pickle(pred_path)
    if pred_df.empty:
        raise RuntimeError(f"OOF predictions are empty: {pred_path}")

    base_margins = _load_window_margins(seed_dir)
    selected_frames: list[pd.DataFrame] = []
    per_window: list[dict[str, Any]] = []
    for window_idx, window_df in pred_df.groupby("window_idx", sort=True):
        base_margin = float(base_margins[int(window_idx)])
        applied_margin = base_margin + float(args.decision_margin_offset)
        chosen = _select_daily_trades(window_df.copy(), applied_margin).copy()
        if not chosen.empty:
            chosen["window_idx"] = int(window_idx)
        selected_frames.append(chosen)
        per_window.append(
            {
                "window_idx": int(window_idx),
                "base_decision_margin": base_margin,
                "applied_decision_margin": applied_margin,
                "objective_metrics": _trade_metrics(chosen, OBJECTIVE_PNL_COL, args.equity),
                "time_stop_reference": _trade_metrics(chosen, TIME_STOP_PNL_COL, args.equity),
            }
        )

    if selected_frames and any(not df.empty for df in selected_frames):
        chosen_all = (
            pd.concat(selected_frames, ignore_index=True)
            .sort_values(["window_idx", "day", "bar_index"])
            .reset_index(drop=True)
        )
    else:
        chosen_all = pd.DataFrame(columns=pred_df.columns.tolist())

    save_pickle(os.path.join(args.output_dir, "chosen_trades.pkl"), chosen_all)
    report = {
        "source_seed_dir": seed_dir,
        "decision_margin_offset": float(args.decision_margin_offset),
        "aggregate_objective": _trade_metrics(chosen_all, OBJECTIVE_PNL_COL, args.equity),
        "aggregate_time_stop_reference": _trade_metrics(chosen_all, TIME_STOP_PNL_COL, args.equity),
        "per_window": per_window,
    }
    save_json(os.path.join(args.output_dir, "reselected_report.json"), report)
    print(
        f"Saved reselected trades: {os.path.join(args.output_dir, 'chosen_trades.pkl')} "
        f"(trades={len(chosen_all)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
