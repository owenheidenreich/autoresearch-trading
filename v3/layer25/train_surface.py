"""Train and evaluate the Layer 2.5 full-surface entry patience gate."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from v3.layer2.common import save_json
from v3.layer25.common import (
    DEFAULT_EQUITY,
    DEFAULT_HORIZON_BARS,
    DEFAULT_MAE_FLOOR_PCT,
    DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_OUT_DIR,
    DEFAULT_ROLLING_DIR,
    DEFAULT_THRESHOLDS,
    build_report_payload,
    build_surface_frame,
    threshold_results,
    walkforward_probs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rolling-dir", default=DEFAULT_ROLLING_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    p.add_argument("--horizon-bars", type=int, default=DEFAULT_HORIZON_BARS, choices=(5, 10, 20))
    p.add_argument("--mae-floor-pct", type=float, default=DEFAULT_MAE_FLOOR_PCT)
    p.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    p.add_argument("--thresholds", type=float, nargs="*", default=list(DEFAULT_THRESHOLDS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    surface_df, meta = build_surface_frame(
        args.rolling_dir,
        equity=args.equity,
        horizon_bars=args.horizon_bars,
        mae_floor_pct=args.mae_floor_pct,
    )
    meta["rolling_dir"] = args.rolling_dir
    surface_df["clean_entry_prob"], walkforward_reports = walkforward_probs(
        surface_df,
        feature_cols=meta["model_feature_names"],
        min_train_rows=args.min_train_rows,
    )

    eval_rows = threshold_results(
        surface_df,
        manifest=meta["manifest"],
        thresholds_by_window=meta["thresholds_by_window"],
        thresholds=list(args.thresholds),
        equity=args.equity,
    )
    payload = build_report_payload(
        surface_df,
        meta=meta,
        walkforward_reports=walkforward_reports,
        threshold_eval=eval_rows,
        equity=args.equity,
    )
    payload["label_definition"] = {
        "clean_entry": (
            "selected_time_stop_value > 0 and selected-direction MAE over the first "
            f"{args.horizon_bars} minutes stays above {args.mae_floor_pct:.1f}% of entry premium"
        ),
        "horizon_bars": int(args.horizon_bars),
        "mae_floor_pct": float(args.mae_floor_pct),
    }

    json_path = os.path.join(args.out_dir, "entry_patience_surface.json")
    csv_path = os.path.join(args.out_dir, "surface_predictions.csv")
    save_json(json_path, payload)
    surface_df.to_csv(csv_path, index=False)

    auc = payload["walkforward_model"]["mean_auc"]
    base = payload["baseline_policy"]
    print("Layer 2.5 entry patience surface")
    print(
        f"Surface rows={payload['surface']['rows']} days={payload['surface']['days']} "
        f"clean_rate={payload['surface']['clean_entry_rate']:.3f}"
    )
    if auc is not None:
        print(f"Walk-forward mean AUC: {auc:.3f}")
    print(
        f"Baseline policy: trades={base['trades']} pf={base['pf']:.3f} "
        f"mean=${base['mean_pnl']:.1f}"
    )
    for result in eval_rows:
        print(
            f"thr={result['threshold']:.2f} trades={result['trades']} "
            f"share={result['trade_share']:.3f} pf={result['pf']:.3f} "
            f"mean=${result['mean_pnl']:.1f}"
        )
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

