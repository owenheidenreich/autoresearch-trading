"""Walk-forward Layer 2.5 entry patience gate.

This is a timing-aware post-selection layer that sits after Layer-2 picks
the bar and direction. Its job is trader-like caution: avoid entries that
either take too much early heat or need too much patience before the move
actually starts working.

Default label:
  clean_entry = selected trade ends profitable at time-stop AND
                selected-direction MAE over the first 10 minutes stays above
                -20% of entry premium.

The model is trained only on safe inputs:
  - rolling Layer-2 augmented features
  - Layer-2 entry_score
  - Layer-2 side_conf
  - chosen direction flag

No forward labels are used as features.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    build_labeled_day,
    effective_direction,
    load_json,
    load_pickle,
    per_day_choice,
    replay_metrics_from_pnls,
    save_json,
)


DEFAULT_ROLLING_DIR = os.path.join("v3", "artifacts", "rolling_l2")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "entry_patience_gate")
DEFAULT_HORIZON_BARS = 10
DEFAULT_MAE_FLOOR_PCT = -20.0
DEFAULT_THRESHOLDS = (0.40, 0.50, 0.60)
DEFAULT_MIN_TRAIN_TRADES = 60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rolling-dir", default=DEFAULT_ROLLING_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--horizon-bars", type=int, default=DEFAULT_HORIZON_BARS, choices=(5, 10, 20))
    p.add_argument("--mae-floor-pct", type=float, default=DEFAULT_MAE_FLOOR_PCT)
    p.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    p.add_argument("--thresholds", type=float, nargs="*", default=list(DEFAULT_THRESHOLDS))
    return p.parse_args()


def _feature_names(manifest: dict[str, Any]) -> list[str]:
    if "augmented_feature_names" in manifest:
        return list(manifest["augmented_feature_names"])
    if "feature_names" in manifest:
        return list(manifest["feature_names"])
    raise KeyError("rolling manifest missing augmented_feature_names")


def _selected_excursion(bar: Any, direction: str, horizon_bars: int, kind: str) -> float | None:
    attr = f"{kind}_{horizon_bars}min_call" if direction == "call" else f"{kind}_{horizon_bars}min_put"
    return getattr(bar.labels, attr)


def _timing_bucket(time_stop_pnl: float | None, mae_pct: float | None, mae_floor_pct: float) -> str:
    if time_stop_pnl is None or mae_pct is None:
        return "missing"
    if time_stop_pnl > 0 and mae_pct > mae_floor_pct:
        return "clean_winner"
    if time_stop_pnl > 0:
        return "shakeout_winner"
    if mae_pct <= mae_floor_pct:
        return "fast_loser"
    return "drift_loser"


def _build_chosen_trade_frame(
    rolling_dir: str,
    equity: float,
    horizon_bars: int,
    mae_floor_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_json(os.path.join(rolling_dir, "manifest.json"))
    safe_feature_names = _feature_names(manifest)
    model_feature_names = safe_feature_names + ["entry_score", "side_conf", "direction_is_call"]

    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    day_cache: dict[str, tuple[Any, Any]] = {}
    rows: list[dict[str, Any]] = []

    for window in manifest["windows"]:
        wi = int(window["window_idx"])
        thresholds = load_json(os.path.join(rolling_dir, f"window_{wi:02d}", "calibration.json"))
        oos_pred = load_pickle(os.path.join(rolling_dir, f"window_{wi:02d}", "oos_predictions.pkl"))
        oos_pred = oos_pred.copy()
        oos_pred["effective_direction"] = oos_pred.apply(
            effective_direction,
            axis=1,
            direction_mode=manifest["direction_mode"],
            policy_mode="scalar_side",
        )

        for day, day_rows in oos_pred.groupby("day"):
            chosen = per_day_choice(
                day_rows,
                thresholds["entry_threshold"],
                thresholds["side_threshold"],
                score_mode=manifest["score_mode"],
                side_score_weight=float(manifest["side_score_weight"]),
                policy_mode="scalar_side",
                direction_mode=manifest["direction_mode"],
            )
            if chosen is None:
                continue

            day = str(day)
            if day not in day_cache:
                day_cache[day] = build_labeled_day(ds, day, cfg, equity=equity)
            log, sidecar = day_cache[day]
            if log is None or sidecar is None:
                continue

            bar = next((b for b in log.bars if b.bar_index == int(chosen["bar_index"])), None)
            if bar is None:
                continue

            direction = str(chosen["effective_direction"])
            if direction not in {"call", "put"}:
                continue

            time_stop_pnl_selected = float(
                chosen["time_stop_pnl_call"] if direction == "call" else chosen["time_stop_pnl_put"]
            )
            mae_selected = _selected_excursion(bar, direction, horizon_bars, "mae")
            mfe_selected = _selected_excursion(bar, direction, horizon_bars, "mfe")

            row = {name: chosen[name] for name in safe_feature_names if name in chosen}
            row["entry_score"] = float(chosen["entry_score"])
            row["side_conf"] = float(chosen["side_conf"])
            row["direction_is_call"] = 1.0 if direction == "call" else 0.0
            row["window_idx"] = wi
            row["day"] = day
            row["bar_index"] = int(chosen["bar_index"])
            row["effective_direction"] = direction
            row["oracle_slice_outcome"] = str(chosen.get("oracle_slice_outcome", ""))
            row["time_stop_pnl_selected"] = time_stop_pnl_selected
            row["mae_selected"] = mae_selected
            row["mfe_selected"] = mfe_selected
            row["timing_bucket"] = _timing_bucket(time_stop_pnl_selected, mae_selected, mae_floor_pct)
            row["clean_entry_label"] = int(
                time_stop_pnl_selected > 0
                and mae_selected is not None
                and mae_selected > mae_floor_pct
            )
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["window_idx", "day", "bar_index"]).reset_index(drop=True)
    return df, {
        "manifest": manifest,
        "model_feature_names": model_feature_names,
    }


def _walkforward_probs(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_trades: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pred_prob = np.full(len(df), np.nan, dtype=np.float64)
    reports: list[dict[str, Any]] = []

    for wi in sorted(df["window_idx"].unique()):
        train_df = df[df["window_idx"] < wi]
        test_df = df[df["window_idx"] == wi]
        prior_mean = float(train_df["clean_entry_label"].mean()) if len(train_df) > 0 else 0.5

        if len(train_df) < min_train_trades or train_df["clean_entry_label"].nunique() < 2:
            probs = np.full(len(test_df), prior_mean, dtype=np.float64)
            mode = "prior_mean"
            auc = None
        else:
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_depth=3,
                min_samples_leaf=10,
                random_state=42,
            )
            X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            y_train = train_df["clean_entry_label"].to_numpy(dtype=np.int64)
            X_test = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            auc = None
            if test_df["clean_entry_label"].nunique() > 1:
                auc = float(roc_auc_score(test_df["clean_entry_label"].to_numpy(dtype=np.int64), probs))
            mode = "model"

        pred_prob[test_df.index.to_numpy()] = probs
        reports.append({
            "window_idx": int(wi),
            "train_trades": int(len(train_df)),
            "test_trades": int(len(test_df)),
            "train_clean_rate": prior_mean,
            "test_clean_rate": float(test_df["clean_entry_label"].mean()),
            "mode": mode,
            "auc": auc,
        })

    return pred_prob, reports


def _threshold_metrics(
    df: pd.DataFrame,
    thresholds: list[float],
    equity: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total_trades = max(len(df), 1)
    for threshold in thresholds:
        kept = df[df["clean_entry_prob"] >= threshold].copy()
        metrics = replay_metrics_from_pnls(kept["time_stop_pnl_selected"].tolist(), equity)
        results.append({
            "threshold": float(threshold),
            "trades": int(len(kept)),
            "trade_share": float(len(kept) / total_trades),
            "pf": float(metrics["pf"]),
            "max_dd_pct": float(metrics["max_dd_pct"]),
            "mean_pnl": float(metrics["mean_pnl"]),
            "clean_rate": float(kept["clean_entry_label"].mean()) if not kept.empty else 0.0,
        })
    return results


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chosen_df, meta = _build_chosen_trade_frame(
        args.rolling_dir,
        equity=args.equity,
        horizon_bars=args.horizon_bars,
        mae_floor_pct=args.mae_floor_pct,
    )
    if chosen_df.empty:
        raise RuntimeError("entry patience gate found zero chosen trades")

    chosen_df["clean_entry_prob"], walkforward_reports = _walkforward_probs(
        chosen_df,
        feature_cols=meta["model_feature_names"],
        min_train_trades=args.min_train_trades,
    )

    baseline_metrics = replay_metrics_from_pnls(chosen_df["time_stop_pnl_selected"].tolist(), args.equity)
    threshold_results = _threshold_metrics(chosen_df, list(args.thresholds), args.equity)
    aucs = [r["auc"] for r in walkforward_reports if r["auc"] is not None]
    recommended = max(
        threshold_results,
        key=lambda r: (r["pf"], r["trades"]),
    )

    payload = {
        "label_definition": {
            "clean_entry": (
                "selected trade ends profitable at time-stop and early adverse excursion "
                f"over the first {args.horizon_bars} minutes stays above {args.mae_floor_pct:.1f}%"
            ),
            "horizon_bars": int(args.horizon_bars),
            "mae_floor_pct": float(args.mae_floor_pct),
        },
        "baseline": {
            "trades": int(len(chosen_df)),
            "pf": float(baseline_metrics["pf"]),
            "max_dd_pct": float(baseline_metrics["max_dd_pct"]),
            "mean_pnl": float(baseline_metrics["mean_pnl"]),
            "clean_entry_rate": float(chosen_df["clean_entry_label"].mean()),
            "timing_bucket_counts": {
                k: int(v) for k, v in chosen_df["timing_bucket"].value_counts().sort_index().items()
            },
        },
        "walkforward_model": {
            "feature_count": int(len(meta["model_feature_names"])),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "per_window": walkforward_reports,
        },
        "threshold_results": threshold_results,
        "recommended_threshold_by_pf": recommended,
    }

    save_json(os.path.join(args.out_dir, "entry_patience_gate.json"), payload)
    chosen_df.to_csv(os.path.join(args.out_dir, "chosen_trade_predictions.csv"), index=False)

    print("Layer 2.5 entry patience gate")
    print(
        f"Baseline: trades={len(chosen_df)} pf={baseline_metrics['pf']:.3f} "
        f"mean=${baseline_metrics['mean_pnl']:.1f}"
    )
    if aucs:
        print(f"Walk-forward mean AUC: {np.mean(aucs):.3f}")
    for result in threshold_results:
        print(
            f"thr={result['threshold']:.2f} trades={result['trades']} "
            f"share={result['trade_share']:.3f} pf={result['pf']:.3f} "
            f"mean=${result['mean_pnl']:.1f}"
        )
    print(f"Saved: {os.path.join(args.out_dir, 'entry_patience_gate.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
