from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_ENTRY_QUANTILE,
    DEFAULT_RUN_DIR,
    DEFAULT_SIDE_QUANTILE,
    calibrate_thresholds,
    effective_direction,
    ensure_dir,
    load_export_bundle,
    per_day_choice,
    save_json,
    save_pickle,
    selected_value_for_direction_mode,
    side_accuracy_weighted,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v3 Layer-2 entry and side models.")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Layer-2 export bundle path.")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Output directory for models and reports.")
    p.add_argument(
        "--entry-target",
        default="time_stop_value_rank",
        choices=("entry_value_rank", "time_stop_value_rank"),
        help="Entry target column.",
    )
    p.add_argument(
        "--side-target",
        default="time_stop_margin_raw",
        choices=("side_margin_raw", "time_stop_margin_raw"),
        help="Side target column.",
    )
    p.add_argument(
        "--direction-mode",
        default="teacher_if_triggered_else_put",
        choices=("model", "teacher_if_triggered_else_model", "teacher_if_triggered_else_put", "always_put"),
        help="How replay and threshold calibration choose direction from the scored bar.",
    )
    p.add_argument(
        "--calibration-mode",
        default="search_mean_time_stop",
        choices=("search_mean_time_stop", "fixed_quantiles"),
        help="How fold-local thresholds are calibrated on the validation window.",
    )
    p.add_argument(
        "--score-mode",
        default="product",
        choices=("product", "entry_only", "entry_plus_side"),
        help="How eligible bars are ranked within a day after threshold gating.",
    )
    p.add_argument("--side-score-weight", type=float, default=0.15, help="Only used for score-mode=entry_plus_side.")
    p.add_argument("--entry-quantile", type=float, default=DEFAULT_ENTRY_QUANTILE, help="Used by fixed-quantile calibration.")
    p.add_argument("--side-quantile", type=float, default=DEFAULT_SIDE_QUANTILE, help="Used by fixed-quantile calibration.")
    return p.parse_args()


def _hist_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=3,
        min_samples_leaf=50,
        random_state=42,
    )


def _prepare_predictions(
    df: pd.DataFrame,
    entry_pred: np.ndarray,
    side_pred: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    out["entry_score"] = entry_pred
    out["side_score"] = side_pred
    out["side_conf"] = np.abs(side_pred)
    out["predicted_direction"] = np.where(side_pred > 0, "call", "put")
    both_available = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] > 0.5)
    only_call = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] <= 0.5)
    only_put = (out["has_passing_put"] > 0.5) & (out["has_passing_call"] <= 0.5)
    out.loc[~both_available, "side_score"] = 0.0
    out.loc[only_call, "predicted_direction"] = "call"
    out.loc[only_put, "predicted_direction"] = "put"
    out.loc[only_call | only_put, "side_conf"] = 1.0
    out.loc[(out["has_passing_call"] <= 0.5) & (out["has_passing_put"] <= 0.5), "predicted_direction"] = ""
    return out


def _selected_side_forward_value(row: pd.Series) -> float:
    if row["predicted_direction"] == "call":
        return float(row["best_forward_pnl_call"]) if pd.notna(row["best_forward_pnl_call"]) else float("nan")
    if row["predicted_direction"] == "put":
        return float(row["best_forward_pnl_put"]) if pd.notna(row["best_forward_pnl_put"]) else float("nan")
    return float("nan")


def main() -> int:
    args = parse_args()
    ensure_dir(args.run_dir)
    bundle = load_export_bundle(args.dataset)
    df: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    feature_names = list(meta["feature_names"])
    folds = list(meta["folds"])
    entry_target = args.entry_target
    side_target = args.side_target
    direction_mode = args.direction_mode

    oof_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    t0 = time.time()

    for fold in folds:
        fold_idx = int(fold["fold_idx"])
        print(f"Fold {fold_idx}: train={len(fold['train_days'])} val={len(fold['val_days'])} test={len(fold['test_days'])}")
        train_df = df[df["day"].isin(fold["train_days"])].copy()
        val_df = df[df["day"].isin(fold["val_days"])].copy()
        test_df = df[df["day"].isin(fold["test_days"])].copy()

        X_train = train_df[feature_names].to_numpy(dtype=np.float32)
        X_val = val_df[feature_names].to_numpy(dtype=np.float32)
        X_test = test_df[feature_names].to_numpy(dtype=np.float32)

        entry_train_mask = train_df[entry_target].notna().to_numpy()
        if not entry_train_mask.any():
            raise RuntimeError(f"Fold {fold_idx} has zero finite entry labels.")
        y_entry = train_df.loc[entry_train_mask, entry_target].to_numpy(dtype=np.float32)
        entry_weight = None
        if "time_stop_value_rank" in train_df:
            entry_weight = train_df.loc[entry_train_mask, "time_stop_value_rank"].fillna(0.5).to_numpy(dtype=np.float32)
            entry_weight = 0.5 + entry_weight
        entry_model = _hist_regressor()
        entry_model.fit(X_train[entry_train_mask], y_entry, sample_weight=entry_weight)

        side_train_mask = (
            train_df[side_target].notna().to_numpy()
            & (train_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
            & (train_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
        )
        if not side_train_mask.any():
            raise RuntimeError(f"Fold {fold_idx} has zero finite side labels.")
        side_model = _hist_regressor()
        side_weight = np.abs(train_df.loc[side_train_mask, side_target].to_numpy(dtype=np.float32))
        side_weight = np.clip(side_weight, 1.0, np.percentile(side_weight, 95) if len(side_weight) > 5 else np.max(side_weight))
        side_model.fit(
            X_train[side_train_mask],
            np.arcsinh(train_df.loc[side_train_mask, side_target].to_numpy(dtype=np.float32) / 100.0),
            sample_weight=side_weight,
        )

        val_pred = _prepare_predictions(
            val_df,
            entry_model.predict(X_val),
            side_model.predict(X_val),
        )
        thresholds = calibrate_thresholds(
            val_pred,
            direction_mode=direction_mode,
            calibration_mode=args.calibration_mode,
            score_mode=args.score_mode,
            side_score_weight=args.side_score_weight,
            entry_quantile=args.entry_quantile,
            side_quantile=args.side_quantile,
        )

        test_pred = _prepare_predictions(
            test_df,
            entry_model.predict(X_test),
            side_model.predict(X_test),
        )
        test_pred["effective_direction"] = test_pred.apply(
            effective_direction,
            axis=1,
            direction_mode=direction_mode,
        )
        test_pred["fold_idx"] = fold_idx
        test_pred["selected_forward_value"] = test_pred.apply(_selected_side_forward_value, axis=1)
        test_pred["selected_time_stop_value"] = test_pred.apply(
            selected_value_for_direction_mode,
            axis=1,
            direction_mode=direction_mode,
            call_col="time_stop_pnl_call",
            put_col="time_stop_pnl_put",
        )
        oof_frames.append(test_pred)

        fold_dir = os.path.join(args.run_dir, "folds", str(fold_idx))
        ensure_dir(fold_dir)
        with open(os.path.join(fold_dir, "entry_model.pkl"), "wb") as f:
            pickle.dump(entry_model, f)
        with open(os.path.join(fold_dir, "side_model.pkl"), "wb") as f:
            pickle.dump(side_model, f)
        save_json(os.path.join(fold_dir, "calibration.json"), thresholds)

        fold_reports.append({
            "fold_idx": fold_idx,
            "n_train_rows": int(len(train_df)),
            "n_val_rows": int(len(val_df)),
            "n_test_rows": int(len(test_df)),
            "entry_target": entry_target,
            "side_target": side_target,
            "direction_mode": direction_mode,
            **thresholds,
        })

    oof = pd.concat(oof_frames, ignore_index=True).sort_values(["day", "bar_index"]).reset_index(drop=True)
    save_pickle(os.path.join(args.run_dir, "oof_predictions.pkl"), oof)

    random_baseline_entry = float(np.nanmean(oof["entry_value_raw"]))
    top_decile_cut = float(np.quantile(oof["entry_score"], 0.90))
    top_decile = oof[oof["entry_score"] >= top_decile_cut]
    top_decile_entry_mean = float(np.nanmean(top_decile["entry_value_raw"]))

    teacher_rows = oof[(oof["teacher_any_triggered"] > 0.5) & oof["entry_value_raw"].notna()]
    teacher_baseline_entry = float(np.nanmean(teacher_rows["entry_value_raw"])) if not teacher_rows.empty else float("nan")

    chosen_rows = []
    thresholds_by_fold = {report["fold_idx"]: report for report in fold_reports}
    for fold_idx, fold_df in oof.groupby("fold_idx"):
        th = thresholds_by_fold[int(fold_idx)]
        for _, day_rows in fold_df.groupby("day"):
            row = per_day_choice(
                day_rows,
                th["entry_threshold"],
                th["side_threshold"],
                score_mode=args.score_mode,
                side_score_weight=args.side_score_weight,
            )
            if row is not None:
                chosen_rows.append(row)
    chosen = pd.DataFrame(chosen_rows) if chosen_rows else pd.DataFrame(columns=oof.columns)
    chosen_entry_mean = float(np.nanmean(chosen["entry_value_raw"])) if not chosen.empty else float("nan")
    chosen_side_margin_mean = float(np.nanmean(chosen["selected_forward_value"])) if not chosen.empty else float("nan")
    chosen_time_stop_mean = float(np.nanmean(chosen["selected_time_stop_value"])) if not chosen.empty else float("nan")
    always_call_baseline = float(np.nanmean(oof["best_forward_pnl_call"]))
    always_put_baseline = float(np.nanmean(oof["best_forward_pnl_put"]))
    always_call_time_stop = float(np.nanmean(oof["time_stop_pnl_call"]))
    always_put_time_stop = float(np.nanmean(oof["time_stop_pnl_put"]))

    side_error_slice = oof[
        (oof["opportunity_oracle_entry"])
        & (oof["oracle_slice_outcome"] == "side_error")
        & oof["side_margin_raw"].notna()
    ].copy()
    side_error_weighted_acc = side_accuracy_weighted(side_error_slice)

    audit = {
        "dataset": args.dataset,
        "run_dir": args.run_dir,
        "entry_target": entry_target,
        "side_target": side_target,
        "direction_mode": direction_mode,
        "calibration_mode": args.calibration_mode,
        "score_mode": args.score_mode,
        "side_score_weight": args.side_score_weight,
        "entry_quantile": args.entry_quantile,
        "side_quantile": args.side_quantile,
        "feature_names": feature_names,
        "elapsed_seconds": time.time() - t0,
        "fold_reports": fold_reports,
        "entry_learnability": {
            "random_eligible_baseline_mean_entry_value": random_baseline_entry,
            "oof_top_decile_mean_entry_value": top_decile_entry_mean,
            "teacher_triggered_baseline_mean_entry_value": teacher_baseline_entry,
            "oof_top1_per_day_mean_entry_value": chosen_entry_mean,
            "oof_top1_per_day_mean_time_stop_value": chosen_time_stop_mean,
        },
        "side_learnability": {
            "chosen_mean_selected_forward_value": chosen_side_margin_mean,
            "always_call_baseline_mean_forward_value": always_call_baseline,
            "always_put_baseline_mean_forward_value": always_put_baseline,
            "chosen_mean_selected_time_stop_value": chosen_time_stop_mean,
            "always_call_baseline_mean_time_stop_value": always_call_time_stop,
            "always_put_baseline_mean_time_stop_value": always_put_time_stop,
            "side_error_weighted_accuracy": side_error_weighted_acc,
            "side_error_count": int(len(side_error_slice)),
        },
    }
    save_json(os.path.join(args.run_dir, "audit.json"), audit)
    save_pickle(os.path.join(args.run_dir, "manifest.pkl"), {
        "dataset": args.dataset,
        "feature_names": feature_names,
        "fold_reports": fold_reports,
        "entry_target": entry_target,
        "side_target": side_target,
        "direction_mode": direction_mode,
        "calibration_mode": args.calibration_mode,
        "score_mode": args.score_mode,
        "side_score_weight": args.side_score_weight,
        "entry_quantile": args.entry_quantile,
        "side_quantile": args.side_quantile,
    })

    print("Saved models + OOF audit")
    print(f"Entry top-decile mean: {top_decile_entry_mean:.2f} vs random {random_baseline_entry:.2f}")
    print(f"Chosen top1/day mean:  {chosen_entry_mean:.2f} vs teacher {teacher_baseline_entry:.2f}")
    print(f"Chosen time-stop mean: {chosen_time_stop_mean:.2f} vs call {always_call_time_stop:.2f} / put {always_put_time_stop:.2f}")
    print(f"Chosen side mean:      {chosen_side_margin_mean:.2f} vs call {always_call_baseline:.2f} / put {always_put_baseline:.2f}")
    print(f"Side-error weighted accuracy: {side_error_weighted_acc:.3f} on n={len(side_error_slice)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
