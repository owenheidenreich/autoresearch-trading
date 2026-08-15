from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from v3.layer2.common import (
    DEFAULT_ENTRY_QUANTILE,
    DEFAULT_SIDE_QUANTILE,
    calibrate_thresholds,
    effective_direction,
    ensure_dir,
    load_export_bundle,
    per_day_choice,
    replay_metrics_from_pnls,
    save_json,
    save_pickle,
    selected_value_for_direction_mode,
    side_accuracy_weighted,
)
from v3.layer2.surface_dataset import DEFAULT_SURFACE_DATASET_PATH
from v3.layer2.surface_neural import train_surface_multitask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the W1 Layer-2 surface-aware shared encoder.")
    p.add_argument("--dataset", default=DEFAULT_SURFACE_DATASET_PATH)
    p.add_argument("--run-dir", default=os.path.join("v3", "artifacts", "layer2_surface_shared"))
    p.add_argument("--entry-target", default="time_stop_value_rank", choices=("entry_value_rank", "time_stop_value_rank"))
    p.add_argument("--side-target", default="time_stop_margin_raw", choices=("side_margin_raw", "time_stop_margin_raw"))
    p.add_argument("--direction-mode", default="model", choices=("model", "teacher_if_triggered_else_model", "teacher_if_triggered_else_put", "always_put"))
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seq-hidden-dim", type=int, default=64)
    p.add_argument("--contract-hidden-dim", type=int, default=48)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--w-entry", type=float, default=1.0)
    p.add_argument("--w-side", type=float, default=1.0)
    p.add_argument("--calibration-mode", default="fixed_quantiles", choices=("search_mean_time_stop", "fixed_quantiles"))
    p.add_argument("--score-mode", default="product", choices=("product", "entry_only", "entry_plus_side"))
    p.add_argument("--side-score-weight", type=float, default=0.15)
    p.add_argument("--entry-quantile", type=float, default=DEFAULT_ENTRY_QUANTILE)
    p.add_argument("--side-quantile", type=float, default=DEFAULT_SIDE_QUANTILE)
    p.add_argument("--latest-only", action="store_true")
    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return "cuda"
    if arg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_predictions(df: pd.DataFrame, entry_pred: np.ndarray, side_pred: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["entry_score"] = entry_pred
    out["side_score"] = side_pred
    out["side_conf"] = np.abs(side_pred)
    out["predicted_direction"] = np.where(side_pred > 0, "call", "put")
    both_available = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] > 0.5)
    only_call = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] <= 0.5)
    only_put = (out["has_passing_put"] > 0.5) & (out["has_passing_call"] <= 0.5)
    neither = (out["has_passing_call"] <= 0.5) & (out["has_passing_put"] <= 0.5)
    out.loc[~both_available, "side_score"] = 0.0
    out.loc[only_call, "predicted_direction"] = "call"
    out.loc[only_put, "predicted_direction"] = "put"
    out.loc[only_call | only_put, "side_conf"] = 1.0
    out.loc[neither, "predicted_direction"] = ""
    out.loc[neither, "side_conf"] = 0.0
    return out


def _selected_side_forward_value(row: pd.Series) -> float:
    direction = row["effective_direction"]
    if direction == "call":
        return float(row["best_forward_pnl_call"]) if pd.notna(row["best_forward_pnl_call"]) else float("nan")
    if direction == "put":
        return float(row["best_forward_pnl_put"]) if pd.notna(row["best_forward_pnl_put"]) else float("nan")
    return float("nan")


def _slice_arrays(mask: np.ndarray, bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "scalar": bundle["rows"].loc[mask, bundle["meta"]["scalar_feature_names"]].to_numpy(dtype=np.float32),
        "seq": bundle["sequence_features"][mask],
        "seq_mask": bundle["sequence_mask"][mask],
        "call": bundle["call_contract_features"][mask],
        "call_mask": bundle["call_contract_mask"][mask],
        "put": bundle["put_contract_features"][mask],
        "put_mask": bundle["put_contract_mask"][mask],
    }


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    ensure_dir(args.run_dir)

    bundle = load_export_bundle(args.dataset)
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    folds = list(meta["folds"])
    if args.latest_only:
        folds = [folds[-1]]

    oof_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    t0 = time.time()

    for fold in folds:
        fold_idx = int(fold["fold_idx"])
        print(f"Fold {fold_idx}: train={len(fold['train_days'])} val={len(fold['val_days'])} test={len(fold['test_days'])}", flush=True)
        train_mask = rows["day"].isin(fold["train_days"]).to_numpy()
        val_mask = rows["day"].isin(fold["val_days"]).to_numpy()
        test_mask = rows["day"].isin(fold["test_days"]).to_numpy()

        train_df = rows.loc[train_mask].copy()
        val_df = rows.loc[val_mask].copy()
        test_df = rows.loc[test_mask].copy()

        X_train = _slice_arrays(train_mask, bundle)
        X_val = _slice_arrays(val_mask, bundle)
        X_test = _slice_arrays(test_mask, bundle)

        entry_mask_train = train_df[args.entry_target].notna().to_numpy()
        entry_mask_val = val_df[args.entry_target].notna().to_numpy()
        y_entry_train = train_df[args.entry_target].fillna(0.0).to_numpy(dtype=np.float32)
        y_entry_val = val_df[args.entry_target].fillna(0.0).to_numpy(dtype=np.float32)
        entry_weight_train = 0.5 + train_df["time_stop_value_rank"].fillna(0.5).to_numpy(dtype=np.float32)
        entry_weight_val = 0.5 + val_df["time_stop_value_rank"].fillna(0.5).to_numpy(dtype=np.float32)

        side_mask_train = (
            train_df[args.side_target].notna().to_numpy()
            & (train_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
            & (train_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
        )
        side_mask_val = (
            val_df[args.side_target].notna().to_numpy()
            & (val_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
            & (val_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
        )
        side_raw_train = train_df[args.side_target].fillna(0.0).to_numpy(dtype=np.float32)
        side_raw_val = val_df[args.side_target].fillna(0.0).to_numpy(dtype=np.float32)
        y_side_train = np.arcsinh(side_raw_train / 100.0)
        y_side_val = np.arcsinh(side_raw_val / 100.0)
        side_weight_train = np.abs(side_raw_train)
        side_weight_val = np.abs(side_raw_val)
        side_weight_train = np.clip(
            side_weight_train,
            1.0,
            np.percentile(side_weight_train[side_mask_train], 95) if side_mask_train.sum() > 5 else max(float(np.max(side_weight_train)), 1.0),
        ).astype(np.float32)
        side_weight_val = np.clip(
            side_weight_val,
            1.0,
            np.percentile(side_weight_val[side_mask_val], 95) if side_mask_val.sum() > 5 else max(float(np.max(side_weight_val)), 1.0),
        ).astype(np.float32)

        entry_model, side_model, train_info = train_surface_multitask(
            scalar_train=X_train["scalar"],
            seq_train=X_train["seq"],
            seq_mask_train=X_train["seq_mask"],
            call_train=X_train["call"],
            call_mask_train=X_train["call_mask"],
            put_train=X_train["put"],
            put_mask_train=X_train["put_mask"],
            y_entry_train=y_entry_train,
            y_side_train=y_side_train,
            entry_mask_train=entry_mask_train,
            side_mask_train=side_mask_train,
            entry_weight_train=entry_weight_train,
            side_weight_train=side_weight_train,
            scalar_val=X_val["scalar"],
            seq_val=X_val["seq"],
            seq_mask_val=X_val["seq_mask"],
            call_val=X_val["call"],
            call_mask_val=X_val["call_mask"],
            put_val=X_val["put"],
            put_mask_val=X_val["put_mask"],
            y_entry_val=y_entry_val,
            y_side_val=y_side_val,
            entry_mask_val=entry_mask_val,
            side_mask_val=side_mask_val,
            entry_weight_val=entry_weight_val,
            side_weight_val=side_weight_val,
            w_entry_loss=args.w_entry,
            w_side_loss=args.w_side,
            device=device,
            seed=args.seed + fold_idx,
            hidden_dim=args.hidden_dim,
            seq_hidden_dim=args.seq_hidden_dim,
            contract_hidden_dim=args.contract_hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

        val_pred = _prepare_predictions(
            val_df,
            entry_model.predict(
                X_val["scalar"],
                X_val["seq"],
                X_val["seq_mask"],
                X_val["call"],
                X_val["call_mask"],
                X_val["put"],
                X_val["put_mask"],
            ),
            side_model.predict(
                X_val["scalar"],
                X_val["seq"],
                X_val["seq_mask"],
                X_val["call"],
                X_val["call_mask"],
                X_val["put"],
                X_val["put_mask"],
            ),
        )
        thresholds = calibrate_thresholds(
            val_pred,
            direction_mode=args.direction_mode,
            calibration_mode=args.calibration_mode,
            score_mode=args.score_mode,
            side_score_weight=args.side_score_weight,
            entry_quantile=args.entry_quantile,
            side_quantile=args.side_quantile,
        )

        test_pred = _prepare_predictions(
            test_df,
            entry_model.predict(
                X_test["scalar"],
                X_test["seq"],
                X_test["seq_mask"],
                X_test["call"],
                X_test["call_mask"],
                X_test["put"],
                X_test["put_mask"],
            ),
            side_model.predict(
                X_test["scalar"],
                X_test["seq"],
                X_test["seq_mask"],
                X_test["call"],
                X_test["call_mask"],
                X_test["put"],
                X_test["put_mask"],
            ),
        )
        test_pred["effective_direction"] = test_pred.apply(
            effective_direction,
            axis=1,
            direction_mode=args.direction_mode,
        )
        test_pred["fold_idx"] = fold_idx
        test_pred["selected_forward_value"] = test_pred.apply(_selected_side_forward_value, axis=1)
        test_pred["selected_time_stop_value"] = test_pred.apply(
            selected_value_for_direction_mode,
            axis=1,
            direction_mode=args.direction_mode,
            call_col="time_stop_pnl_call",
            put_col="time_stop_pnl_put",
        )
        oof_frames.append(test_pred)

        fold_dir = os.path.join(args.run_dir, "folds", str(fold_idx))
        ensure_dir(fold_dir)
        save_pickle(os.path.join(fold_dir, "entry_model.pkl"), entry_model)
        save_pickle(os.path.join(fold_dir, "side_model.pkl"), side_model)
        save_json(os.path.join(fold_dir, "calibration.json"), thresholds)
        save_json(os.path.join(fold_dir, "training_info.json"), train_info)

        fold_reports.append(
            {
                "fold_idx": fold_idx,
                "n_train_rows": int(len(train_df)),
                "n_val_rows": int(len(val_df)),
                "n_test_rows": int(len(test_df)),
                "entry_target": args.entry_target,
                "side_target": args.side_target,
                "direction_mode": args.direction_mode,
                "model": "surface_shared_encoder",
                "scalar_feature_count": int(len(meta["scalar_feature_names"])),
                "sequence_feature_count": int(len(meta["sequence_feature_names"])),
                "contract_feature_count": int(len(meta["contract_feature_names"])),
                "history_bars": int(meta["history_bars"]),
                "top_k_contracts": int(meta["top_k_contracts"]),
                **thresholds,
                **train_info,
            }
        )

    oof = pd.concat(oof_frames, ignore_index=True).sort_values(["day", "bar_index"]).reset_index(drop=True)
    save_pickle(os.path.join(args.run_dir, "oof_predictions.pkl"), oof)

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
                direction_mode=args.direction_mode,
            )
            if row is not None:
                chosen_rows.append(row)
    chosen = pd.DataFrame(chosen_rows) if chosen_rows else pd.DataFrame(columns=oof.columns)

    replay_metrics = replay_metrics_from_pnls(
        chosen["selected_time_stop_value"].astype(float).tolist() if not chosen.empty else [],
        25_000.0,
    )
    replay_metrics["trades_per_day"] = float(len(chosen) / max(oof["day"].nunique(), 1))
    replay_metrics["call_pct"] = float((chosen["effective_direction"] == "call").mean()) if not chosen.empty else 0.0
    replay_metrics["mean_entry_value_raw"] = float(chosen["entry_value_raw"].dropna().mean()) if not chosen.empty else 0.0
    replay_metrics["mean_time_stop_value"] = float(chosen["selected_time_stop_value"].dropna().mean()) if not chosen.empty else 0.0

    random_baseline_entry = float(np.nanmean(oof["entry_value_raw"]))
    top_decile_cut = float(np.quantile(oof["entry_score"], 0.90))
    top_decile = oof[oof["entry_score"] >= top_decile_cut]
    top_decile_entry_mean = float(np.nanmean(top_decile["entry_value_raw"]))
    chosen_entry_mean = float(np.nanmean(chosen["entry_value_raw"])) if not chosen.empty else float("nan")
    chosen_time_stop_mean = float(np.nanmean(chosen["selected_time_stop_value"])) if not chosen.empty else float("nan")
    chosen_side_margin_mean = float(np.nanmean(chosen["selected_forward_value"])) if not chosen.empty else float("nan")
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
        "entry_target": args.entry_target,
        "side_target": args.side_target,
        "direction_mode": args.direction_mode,
        "calibration_mode": args.calibration_mode,
        "score_mode": args.score_mode,
        "side_score_weight": args.side_score_weight,
        "entry_quantile": args.entry_quantile,
        "side_quantile": args.side_quantile,
        "elapsed_seconds": time.time() - t0,
        "surface_meta": {
            "scalar_feature_names": meta["scalar_feature_names"],
            "sequence_feature_names": meta["sequence_feature_names"],
            "contract_feature_names": meta["contract_feature_names"],
            "history_bars": int(meta["history_bars"]),
            "top_k_contracts": int(meta["top_k_contracts"]),
        },
        "fold_reports": fold_reports,
        "entry_learnability": {
            "random_eligible_baseline_mean_entry_value": random_baseline_entry,
            "oof_top_decile_mean_entry_value": top_decile_entry_mean,
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
        "replay_metrics": replay_metrics,
    }
    save_json(os.path.join(args.run_dir, "audit.json"), audit)
    save_json(
        os.path.join(args.run_dir, "replay_report.json"),
        {
            "dataset": args.dataset,
            "run_dir": args.run_dir,
            "model": "surface_shared_encoder",
            "direction_mode": args.direction_mode,
            "calibration_mode": args.calibration_mode,
            "score_mode": args.score_mode,
            "side_score_weight": args.side_score_weight,
            "layer2": replay_metrics,
        },
    )
    save_pickle(
        os.path.join(args.run_dir, "manifest.pkl"),
        {
            "dataset": args.dataset,
            "model": "surface_shared_encoder",
            "fold_reports": fold_reports,
            "entry_target": args.entry_target,
            "side_target": args.side_target,
            "direction_mode": args.direction_mode,
            "calibration_mode": args.calibration_mode,
            "score_mode": args.score_mode,
            "side_score_weight": args.side_score_weight,
            "entry_quantile": args.entry_quantile,
            "side_quantile": args.side_quantile,
            "surface_meta": audit["surface_meta"],
        },
    )

    print("Saved W1 surface models + OOF audit")
    print(f"Entry top-decile mean: {top_decile_entry_mean:.2f} vs random {random_baseline_entry:.2f}")
    print(f"Chosen top1/day mean:  {chosen_entry_mean:.2f}")
    print(f"Chosen time-stop mean: {chosen_time_stop_mean:.2f} vs call {always_call_time_stop:.2f} / put {always_put_time_stop:.2f}")
    print(f"Chosen side mean:      {chosen_side_margin_mean:.2f} vs call {always_call_baseline:.2f} / put {always_put_baseline:.2f}")
    print(f"Chosen PF/DD:          {replay_metrics['pf']:.3f} / {replay_metrics['max_dd_pct']:.1f}%")
    print(f"Side-error weighted accuracy: {side_error_weighted_acc:.3f} on n={len(side_error_slice)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
