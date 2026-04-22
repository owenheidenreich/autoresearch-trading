from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_ENTRY_QUANTILE,
    DEFAULT_POLICY_MODE,
    DEFAULT_RUN_DIR,
    DEFAULT_SIDE_QUANTILE,
    calibrate_thresholds,
    effective_direction,
    ensure_dir,
    load_export_bundle,
    per_day_choice,
    route_source_from_row,
    save_json,
    save_pickle,
    selected_value_for_direction_mode,
    side_accuracy_weighted,
)
from v3.layer2.neural import train_multitask, train_route_aware_multitask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train neural v3 Layer-2 entry and side models.")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Layer-2 export bundle path.")
    p.add_argument("--run-dir", default=os.path.join("v3", "artifacts", "layer2_neural"), help="Output directory.")
    p.add_argument("--entry-target", default="entry_value_rank", choices=("entry_value_rank", "time_stop_value_rank"))
    p.add_argument("--side-target", default="time_stop_margin_raw", choices=("side_margin_raw", "time_stop_margin_raw"))
    p.add_argument(
        "--direction-mode",
        default="teacher_if_triggered_else_put",
        choices=("model", "teacher_if_triggered_else_model", "teacher_if_triggered_else_put", "always_put"),
    )
    p.add_argument(
        "--policy-mode",
        default=DEFAULT_POLICY_MODE,
        choices=("scalar_side", "route_aware_fallback"),
        help="Scalar side-direction policy or detached route-aware fallback policy.",
    )
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=2, help="Trunk layers in the shared encoder (heads add 1 linear each).")
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--w-entry", type=float, default=1.0, help="Multitask loss weight on the entry head.")
    p.add_argument("--w-side", type=float, default=1.0, help="Multitask loss weight on the side head.")
    p.add_argument(
        "--side-dropout",
        type=float,
        default=None,
        help="If set, applies an ADDITIONAL dropout layer on the trunk output before the side head. "
             "Leaves trunk dropout = --dropout. Default None = no extra dropout.",
    )
    p.add_argument(
        "--detach-side",
        action="store_true",
        help="Detach the side head's gradient from the trunk. Side head becomes a linear probe "
             "on entry-optimized features. Default off.",
    )
    p.add_argument(
        "--wrong-side-alpha",
        type=float,
        default=1.0,
        help="Elementwise multiplier on side loss where prediction and target have OPPOSITE signs. "
             "1.0 = symmetric Huber (default). >1.0 penalises wrong-side samples harder.",
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
    p.add_argument("--latest-only", action="store_true", help="Train only the latest canonical fold.")
    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return "cuda"
    if arg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_predictions(
    df: pd.DataFrame,
    entry_pred: np.ndarray,
    side_pred: np.ndarray | None,
    direction_mode: str,
    policy_mode: str,
    fallback_call_pred: np.ndarray | None = None,
    fallback_put_pred: np.ndarray | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["entry_score"] = entry_pred

    if policy_mode == "route_aware_fallback":
        if fallback_call_pred is None or fallback_put_pred is None:
            raise ValueError("route_aware_fallback requires fallback_call_pred and fallback_put_pred")
        out["fallback_call_score"] = fallback_call_pred
        out["fallback_put_score"] = fallback_put_pred
        call_avail = out["has_passing_call"] > 0.5
        put_avail = out["has_passing_put"] > 0.5
        adj_call = np.where(call_avail.to_numpy(), fallback_call_pred, -np.inf)
        adj_put = np.where(put_avail.to_numpy(), fallback_put_pred, -np.inf)
        best_nonflat = np.maximum(adj_call, adj_put)
        out["side_conf"] = np.maximum(best_nonflat, 0.0)
        safe_call = np.where(np.isfinite(adj_call), adj_call, 0.0)
        safe_put = np.where(np.isfinite(adj_put), adj_put, 0.0)
        out["side_score"] = safe_put - safe_call
        out["predicted_direction"] = np.where(
            (adj_call >= adj_put) & (adj_call > 0.0),
            "call",
            np.where((adj_put > adj_call) & (adj_put > 0.0), "put", ""),
        )
    else:
        if side_pred is None:
            raise ValueError("scalar_side policy requires side_pred")
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

    out["effective_direction"] = out.apply(
        effective_direction,
        axis=1,
        direction_mode=direction_mode,
        policy_mode=policy_mode,
    )
    out["route_source"] = out.apply(
        route_source_from_row,
        axis=1,
        direction_mode=direction_mode,
        policy_mode=policy_mode,
    )
    return out


def _selected_side_forward_value(row: pd.Series) -> float:
    direction = row["effective_direction"]
    if direction == "call":
        return float(row["best_forward_pnl_call"]) if pd.notna(row["best_forward_pnl_call"]) else float("nan")
    if direction == "put":
        return float(row["best_forward_pnl_put"]) if pd.notna(row["best_forward_pnl_put"]) else float("nan")
    return float("nan")


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    detach_side_flag = True if args.policy_mode == "route_aware_fallback" else args.detach_side
    ensure_dir(args.run_dir)

    bundle = load_export_bundle(args.dataset)
    df: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    feature_names = list(meta["feature_names"])
    folds = list(meta["folds"])
    if args.latest_only:
        folds = [folds[-1]]

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

        entry_mask_train = train_df[args.entry_target].notna().to_numpy()
        y_entry_train = train_df[args.entry_target].fillna(0.0).to_numpy(dtype=np.float32)
        entry_weight_train = 0.5 + train_df["time_stop_value_rank"].fillna(0.5).to_numpy(dtype=np.float32)

        entry_mask_val = val_df[args.entry_target].notna().to_numpy()
        y_entry_val = val_df[args.entry_target].fillna(0.0).to_numpy(dtype=np.float32)
        entry_weight_val = 0.5 + val_df["time_stop_value_rank"].fillna(0.5).to_numpy(dtype=np.float32)

        if args.policy_mode == "route_aware_fallback":
            call_mask_train = (
                train_df["time_stop_pnl_call"].notna().to_numpy()
                & (train_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
            )
            call_raw_train = train_df["time_stop_pnl_call"].fillna(0.0).to_numpy(dtype=np.float32)
            y_call_train = np.arcsinh(call_raw_train / 100.0)
            call_weight_train = np.abs(call_raw_train)
            if call_mask_train.sum() > 5:
                _call_p95 = float(np.percentile(np.abs(call_raw_train[call_mask_train]), 95))
            else:
                _call_p95 = float(np.max(np.abs(call_raw_train))) if call_raw_train.size else 1.0
            call_weight_train = np.clip(call_weight_train, 1.0, max(_call_p95, 1.0)).astype(np.float32)

            put_mask_train = (
                train_df["time_stop_pnl_put"].notna().to_numpy()
                & (train_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
            )
            put_raw_train = train_df["time_stop_pnl_put"].fillna(0.0).to_numpy(dtype=np.float32)
            y_put_train = np.arcsinh(put_raw_train / 100.0)
            put_weight_train = np.abs(put_raw_train)
            if put_mask_train.sum() > 5:
                _put_p95 = float(np.percentile(np.abs(put_raw_train[put_mask_train]), 95))
            else:
                _put_p95 = float(np.max(np.abs(put_raw_train))) if put_raw_train.size else 1.0
            put_weight_train = np.clip(put_weight_train, 1.0, max(_put_p95, 1.0)).astype(np.float32)

            call_mask_val = (
                val_df["time_stop_pnl_call"].notna().to_numpy()
                & (val_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
            )
            call_raw_val = val_df["time_stop_pnl_call"].fillna(0.0).to_numpy(dtype=np.float32)
            y_call_val = np.arcsinh(call_raw_val / 100.0)
            call_weight_val = np.abs(call_raw_val)
            if call_mask_val.sum() > 5:
                _call_p95v = float(np.percentile(np.abs(call_raw_val[call_mask_val]), 95))
            else:
                _call_p95v = float(np.max(np.abs(call_raw_val))) if call_raw_val.size else 1.0
            call_weight_val = np.clip(call_weight_val, 1.0, max(_call_p95v, 1.0)).astype(np.float32)

            put_mask_val = (
                val_df["time_stop_pnl_put"].notna().to_numpy()
                & (val_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
            )
            put_raw_val = val_df["time_stop_pnl_put"].fillna(0.0).to_numpy(dtype=np.float32)
            y_put_val = np.arcsinh(put_raw_val / 100.0)
            put_weight_val = np.abs(put_raw_val)
            if put_mask_val.sum() > 5:
                _put_p95v = float(np.percentile(np.abs(put_raw_val[put_mask_val]), 95))
            else:
                _put_p95v = float(np.max(np.abs(put_raw_val))) if put_raw_val.size else 1.0
            put_weight_val = np.clip(put_weight_val, 1.0, max(_put_p95v, 1.0)).astype(np.float32)

            entry_model, fallback_call_model, fallback_put_model, mt_info = train_route_aware_multitask(
                X_train=X_train,
                y_entry_train=y_entry_train,
                y_call_train=y_call_train,
                y_put_train=y_put_train,
                entry_mask_train=entry_mask_train,
                call_mask_train=call_mask_train,
                put_mask_train=put_mask_train,
                entry_weight_train=entry_weight_train,
                call_weight_train=call_weight_train,
                put_weight_train=put_weight_train,
                X_val=X_val,
                y_entry_val=y_entry_val,
                y_call_val=y_call_val,
                y_put_val=y_put_val,
                entry_mask_val=entry_mask_val,
                call_mask_val=call_mask_val,
                put_mask_val=put_mask_val,
                entry_weight_val=entry_weight_val,
                call_weight_val=call_weight_val,
                put_weight_val=put_weight_val,
                w_entry_loss=args.w_entry,
                w_side_loss=args.w_side,
                device=device,
                seed=args.seed + fold_idx,
                hidden_dim=args.hidden_dim,
                depth=args.depth,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                side_dropout=args.side_dropout,
                detach_side=detach_side_flag,
            )
            side_model = None
            entry_info = {
                "best_val_loss": mt_info["best_entry_val_loss"],
                "best_epoch": mt_info["best_epoch"],
            }
            side_info = {
                "best_val_loss": 0.5 * (mt_info["best_call_val_loss"] + mt_info["best_put_val_loss"]),
                "best_epoch": mt_info["best_epoch"],
                "best_call_val_loss": mt_info["best_call_val_loss"],
                "best_put_val_loss": mt_info["best_put_val_loss"],
            }

            val_pred = _prepare_predictions(
                val_df,
                entry_model.predict(X_val),
                None,
                args.direction_mode,
                args.policy_mode,
                fallback_call_pred=fallback_call_model.predict(X_val),
                fallback_put_pred=fallback_put_model.predict(X_val),
            )
        else:
            side_mask_train = (
                train_df[args.side_target].notna().to_numpy()
                & (train_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
                & (train_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
            )
            side_raw_train = train_df[args.side_target].fillna(0.0).to_numpy(dtype=np.float32)
            y_side_train = np.arcsinh(side_raw_train / 100.0)
            side_weight_train = np.abs(side_raw_train)
            if side_mask_train.sum() > 5:
                _p95 = float(np.percentile(np.abs(side_raw_train[side_mask_train]), 95))
            else:
                _p95 = float(np.max(np.abs(side_raw_train))) if side_raw_train.size else 1.0
            side_weight_train = np.clip(side_weight_train, 1.0, max(_p95, 1.0)).astype(np.float32)

            side_mask_val = (
                val_df[args.side_target].notna().to_numpy()
                & (val_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
                & (val_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
            )
            side_raw_val = val_df[args.side_target].fillna(0.0).to_numpy(dtype=np.float32)
            y_side_val = np.arcsinh(side_raw_val / 100.0)
            side_weight_val = np.abs(side_raw_val)
            if side_mask_val.sum() > 5:
                _p95v = float(np.percentile(np.abs(side_raw_val[side_mask_val]), 95))
            else:
                _p95v = float(np.max(np.abs(side_raw_val))) if side_raw_val.size else 1.0
            side_weight_val = np.clip(side_weight_val, 1.0, max(_p95v, 1.0)).astype(np.float32)

            entry_model, side_model, mt_info = train_multitask(
                X_train=X_train,
                y_entry_train=y_entry_train,
                y_side_train=y_side_train,
                entry_mask_train=entry_mask_train,
                side_mask_train=side_mask_train,
                entry_weight_train=entry_weight_train,
                side_weight_train=side_weight_train,
                X_val=X_val,
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
                depth=args.depth,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                side_dropout=args.side_dropout,
                detach_side=detach_side_flag,
                wrong_side_alpha=args.wrong_side_alpha,
            )
            fallback_call_model = None
            fallback_put_model = None
            entry_info = {
                "best_val_loss": mt_info["best_entry_val_loss"],
                "best_epoch": mt_info["best_epoch"],
            }
            side_info = {
                "best_val_loss": mt_info["best_side_val_loss"],
                "best_epoch": mt_info["best_epoch"],
            }

            val_pred = _prepare_predictions(
                val_df,
                entry_model.predict(X_val),
                side_model.predict(X_val),
                args.direction_mode,
                args.policy_mode,
            )

        thresholds = calibrate_thresholds(
            val_pred,
            direction_mode=args.direction_mode,
            calibration_mode=args.calibration_mode,
            score_mode=args.score_mode,
            side_score_weight=args.side_score_weight,
            entry_quantile=args.entry_quantile,
            side_quantile=args.side_quantile,
            policy_mode=args.policy_mode,
        )

        if args.policy_mode == "route_aware_fallback":
            test_pred = _prepare_predictions(
                test_df,
                entry_model.predict(X_test),
                None,
                args.direction_mode,
                args.policy_mode,
                fallback_call_pred=fallback_call_model.predict(X_test),
                fallback_put_pred=fallback_put_model.predict(X_test),
            )
        else:
            test_pred = _prepare_predictions(
                test_df,
                entry_model.predict(X_test),
                side_model.predict(X_test),
                args.direction_mode,
                args.policy_mode,
            )
        test_pred["fold_idx"] = fold_idx
        test_pred["selected_forward_value"] = test_pred.apply(_selected_side_forward_value, axis=1)
        test_pred["selected_time_stop_value"] = test_pred.apply(
            selected_value_for_direction_mode,
            axis=1,
            direction_mode=args.direction_mode,
            call_col="time_stop_pnl_call",
            put_col="time_stop_pnl_put",
            policy_mode=args.policy_mode,
        )
        oof_frames.append(test_pred)

        fold_dir = os.path.join(args.run_dir, "folds", str(fold_idx))
        ensure_dir(fold_dir)
        with open(os.path.join(fold_dir, "entry_model.pkl"), "wb") as f:
            pickle.dump(entry_model, f)
        if args.policy_mode == "route_aware_fallback":
            with open(os.path.join(fold_dir, "fallback_call_model.pkl"), "wb") as f:
                pickle.dump(fallback_call_model, f)
            with open(os.path.join(fold_dir, "fallback_put_model.pkl"), "wb") as f:
                pickle.dump(fallback_put_model, f)
        else:
            with open(os.path.join(fold_dir, "side_model.pkl"), "wb") as f:
                pickle.dump(side_model, f)
        save_json(os.path.join(fold_dir, "calibration.json"), thresholds)
        save_json(os.path.join(fold_dir, "training_info.json"), {
            "device": device,
            "architecture": "route_aware_shared_encoder" if args.policy_mode == "route_aware_fallback" else "shared_encoder",
            "policy_mode": args.policy_mode,
            "entry": entry_info,
            "side": side_info,
            "multitask_best_val_loss": mt_info["best_val_loss"],
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "w_entry": args.w_entry,
            "w_side": args.w_side,
        })

        fold_reports.append({
            "fold_idx": fold_idx,
            "n_train_rows": int(len(train_df)),
            "n_val_rows": int(len(val_df)),
            "n_test_rows": int(len(test_df)),
            "entry_target": args.entry_target,
            "side_target": args.side_target,
            "direction_mode": args.direction_mode,
            "policy_mode": args.policy_mode,
            **thresholds,
            "entry_best_val_loss": entry_info["best_val_loss"],
            "entry_best_epoch": entry_info["best_epoch"],
            "side_best_val_loss": side_info["best_val_loss"],
            "side_best_epoch": side_info["best_epoch"],
            "best_call_val_loss": side_info.get("best_call_val_loss"),
            "best_put_val_loss": side_info.get("best_put_val_loss"),
            "multitask_best_val_loss": mt_info["best_val_loss"],
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
                policy_mode=args.policy_mode,
                direction_mode=args.direction_mode,
            )
            if row is not None:
                chosen_rows.append(row)
    chosen = pd.DataFrame(chosen_rows) if chosen_rows else pd.DataFrame(columns=oof.columns)

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
        & oof[args.side_target].notna()
    ].copy()
    side_error_weighted_acc = side_accuracy_weighted(side_error_slice)

    audit = {
        "dataset": args.dataset,
        "run_dir": args.run_dir,
        "device": device,
        "architecture": "route_aware_shared_encoder" if args.policy_mode == "route_aware_fallback" else "shared_encoder",
        "entry_target": args.entry_target,
        "side_target": args.side_target,
        "direction_mode": args.direction_mode,
        "policy_mode": args.policy_mode,
        "calibration_mode": args.calibration_mode,
        "score_mode": args.score_mode,
        "side_score_weight": args.side_score_weight,
        "entry_quantile": args.entry_quantile,
        "side_quantile": args.side_quantile,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "w_entry": args.w_entry,
        "w_side": args.w_side,
        "side_dropout": args.side_dropout,
        "detach_side": detach_side_flag,
        "wrong_side_alpha": args.wrong_side_alpha,
        "elapsed_seconds": time.time() - t0,
        "feature_names": feature_names,
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
        "entry_target": args.entry_target,
        "side_target": args.side_target,
        "direction_mode": args.direction_mode,
        "policy_mode": args.policy_mode,
        "calibration_mode": args.calibration_mode,
        "score_mode": args.score_mode,
        "side_score_weight": args.side_score_weight,
        "entry_quantile": args.entry_quantile,
        "side_quantile": args.side_quantile,
        "model_type": "torch_route_aware_shared_encoder" if args.policy_mode == "route_aware_fallback" else "torch_shared_encoder",
        "architecture": "route_aware_shared_encoder" if args.policy_mode == "route_aware_fallback" else "shared_encoder",
        "w_entry": args.w_entry,
        "w_side": args.w_side,
        "side_dropout": args.side_dropout,
        "detach_side": detach_side_flag,
        "wrong_side_alpha": args.wrong_side_alpha,
    })

    print("Saved neural models + OOF audit")
    print(f"Device: {device}")
    print(f"Entry top-decile mean: {top_decile_entry_mean:.2f} vs random {random_baseline_entry:.2f}")
    print(f"Chosen top1/day mean:  {chosen_entry_mean:.2f} vs teacher {teacher_baseline_entry:.2f}")
    print(f"Chosen time-stop mean: {chosen_time_stop_mean:.2f} vs call {always_call_time_stop:.2f} / put {always_put_time_stop:.2f}")
    print(f"Chosen side mean:      {chosen_side_margin_mean:.2f} vs call {always_call_baseline:.2f} / put {always_put_baseline:.2f}")
    print(f"Side-error weighted accuracy: {side_error_weighted_acc:.3f} on n={len(side_error_slice)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
