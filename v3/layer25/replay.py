"""Replay the Layer 2.5 patience-gated policy from saved surface predictions."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from v3.layer2.common import load_json, save_json
from v3.layer25.common import (
    DEFAULT_EQUITY,
    DEFAULT_OUT_DIR,
    policy_trades,
    trade_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold", type=float, default=None, help="Patience threshold; defaults to the report's recommended threshold.")
    p.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_path = os.path.join(args.surface_dir, "entry_patience_surface.json")
    csv_path = os.path.join(args.surface_dir, "surface_predictions.csv")

    report = load_json(report_path)
    surface_df = pd.read_csv(csv_path)
    threshold = (
        float(report["recommended_threshold_by_pf"]["threshold"])
        if args.threshold is None
        else float(args.threshold)
    )

    thresholds_by_window = {
        int(k): v for k, v in report["policy_meta"]["thresholds_by_window"].items()
    }
    manifest = {
        "direction_mode": report["policy_meta"]["direction_mode"],
        "score_mode": report["policy_meta"]["score_mode"],
        "side_score_weight": float(report["policy_meta"]["side_score_weight"]),
    }
    trades = policy_trades(
        surface_df,
        manifest=manifest,
        thresholds_by_window=thresholds_by_window,
        patience_threshold=threshold,
    )
    metrics = trade_metrics(trades, args.equity, int(surface_df["day"].nunique()))
    threshold_tag = f"{threshold:.2f}".replace(".", "p")
    trades_path = os.path.join(args.surface_dir, f"layer25_trades_thr_{threshold_tag}.csv")
    replay_path = os.path.join(args.surface_dir, f"layer25_replay_thr_{threshold_tag}.json")
    trades.to_csv(trades_path, index=False)
    save_json(
        replay_path,
        {
            "surface_dir": args.surface_dir,
            "threshold": float(threshold),
            "metrics": metrics,
            "trade_count": int(len(trades)),
        },
    )

    print("Layer 2.5 replay")
    print(
        f"threshold={threshold:.2f} trades={metrics['trades']} "
        f"share={metrics['trade_share']:.3f} pf={metrics['pf']:.3f} "
        f"mean=${metrics['mean_pnl']:.1f} dd={metrics['max_dd_pct']:.1f}%"
    )
    print(f"Saved: {trades_path}")
    print(f"Saved: {replay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
