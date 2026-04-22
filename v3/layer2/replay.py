from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_POLICY_MODE,
    DEFAULT_RUN_DIR,
    build_labeled_day,
    compute_time_stop_pnl_for_direction,
    effective_direction,
    ensure_dir,
    load_export_bundle,
    load_json,
    load_pickle,
    per_day_choice,
    replay_metrics_from_pnls,
    route_source_from_row,
    save_json,
    teacher_baseline_choice,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay the v3 Layer-2 one-trade-per-day policy.")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Layer-2 export bundle path.")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Training artifact directory.")
    p.add_argument("--equity", type=float, default=25_000.0, help="Starting equity for DD reporting.")
    p.add_argument(
        "--direction-mode",
        default=None,
        choices=("model", "teacher_if_triggered_else_model", "teacher_if_triggered_else_put", "always_put"),
        help="How replay chooses direction once a bar is selected.",
    )
    p.add_argument(
        "--policy-mode",
        default=None,
        choices=("scalar_side", "route_aware_fallback"),
        help="Scalar side-direction policy or route-aware fallback policy.",
    )
    p.add_argument(
        "--score-mode",
        default=None,
        choices=("product", "entry_only", "entry_plus_side"),
        help="How eligible bars are ranked within a day after threshold gating.",
    )
    p.add_argument("--side-score-weight", type=float, default=None, help="Only used for score-mode=entry_plus_side.")
    p.add_argument("--latest-only", action="store_true", help="Replay only the latest canonical fold.")
    return p.parse_args()


def _load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _prepare_policy_predictions(
    test_df: pd.DataFrame,
    entry_pred: np.ndarray,
    *,
    direction_mode: str,
    policy_mode: str,
    side_pred: np.ndarray | None = None,
    fallback_call_pred: np.ndarray | None = None,
    fallback_put_pred: np.ndarray | None = None,
) -> pd.DataFrame:
    out = test_df.copy()
    out["entry_score"] = entry_pred

    if policy_mode == "route_aware_fallback":
        if fallback_call_pred is None or fallback_put_pred is None:
            raise ValueError("route_aware_fallback requires call/put fallback predictions")
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
            raise ValueError("scalar_side requires side_pred")
        out["side_score"] = side_pred
        out["side_conf"] = np.abs(out["side_score"])
        out["predicted_direction"] = np.where(out["side_score"] > 0, "call", "put")
        only_call = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] <= 0.5)
        only_put = (out["has_passing_put"] > 0.5) & (out["has_passing_call"] <= 0.5)
        neither = (out["has_passing_call"] <= 0.5) & (out["has_passing_put"] <= 0.5)
        out.loc[only_call, "predicted_direction"] = "call"
        out.loc[only_put, "predicted_direction"] = "put"
        out.loc[only_call | only_put, "side_conf"] = 1.0
        out.loc[neither, "predicted_direction"] = ""
        out.loc[neither, "side_conf"] = 0.0
        out.loc[~((out["has_passing_call"] > 0.5) & (out["has_passing_put"] > 0.5)), "side_score"] = 0.0

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


def main() -> int:
    args = parse_args()
    bundle = load_export_bundle(args.dataset)
    df: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    feature_names = list(meta["feature_names"])
    folds = list(meta["folds"])
    manifest_path = os.path.join(args.run_dir, "manifest.pkl")
    manifest = load_pickle(manifest_path) if os.path.exists(manifest_path) else {}
    direction_mode = args.direction_mode or manifest.get("direction_mode", "teacher_if_triggered_else_put")
    policy_mode = args.policy_mode or manifest.get("policy_mode", DEFAULT_POLICY_MODE)
    score_mode = args.score_mode or manifest.get("score_mode", "product")
    side_score_weight = (
        float(args.side_score_weight)
        if args.side_score_weight is not None
        else float(manifest.get("side_score_weight", 0.15))
    )
    if args.latest_only:
        folds = [folds[-1]]

    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    t0 = time.time()

    layer2_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    processed_folds: list[dict[str, Any]] = []

    for fold in folds:
        fold_idx = int(fold["fold_idx"])
        fold_dir = os.path.join(args.run_dir, "folds", str(fold_idx))
        if not (
            os.path.exists(os.path.join(fold_dir, "entry_model.pkl"))
            and os.path.exists(os.path.join(fold_dir, "calibration.json"))
        ):
            print(f"Skipping fold {fold_idx}: missing artifacts in {fold_dir}")
            continue
        if policy_mode == "route_aware_fallback":
            if not (
                os.path.exists(os.path.join(fold_dir, "fallback_call_model.pkl"))
                and os.path.exists(os.path.join(fold_dir, "fallback_put_model.pkl"))
            ):
                print(f"Skipping fold {fold_idx}: missing route-aware fallback artifacts in {fold_dir}")
                continue
        elif not os.path.exists(os.path.join(fold_dir, "side_model.pkl")):
            print(f"Skipping fold {fold_idx}: missing scalar side artifact in {fold_dir}")
            continue
        processed_folds.append(fold)
        entry_model = _load_model(os.path.join(fold_dir, "entry_model.pkl"))
        thresholds = load_json(os.path.join(fold_dir, "calibration.json"))

        test_df = df[df["day"].isin(fold["test_days"])].copy()
        X_test = test_df[feature_names].to_numpy(dtype=np.float32)
        if policy_mode == "route_aware_fallback":
            fallback_call_model = _load_model(os.path.join(fold_dir, "fallback_call_model.pkl"))
            fallback_put_model = _load_model(os.path.join(fold_dir, "fallback_put_model.pkl"))
            test_df = _prepare_policy_predictions(
                test_df,
                entry_model.predict(X_test),
                direction_mode=direction_mode,
                policy_mode=policy_mode,
                fallback_call_pred=fallback_call_model.predict(X_test),
                fallback_put_pred=fallback_put_model.predict(X_test),
            )
        else:
            side_model = _load_model(os.path.join(fold_dir, "side_model.pkl"))
            test_df = _prepare_policy_predictions(
                test_df,
                entry_model.predict(X_test),
                direction_mode=direction_mode,
                policy_mode=policy_mode,
                side_pred=side_model.predict(X_test),
            )

        for day, day_rows in test_df.groupby("day"):
            log, sidecar = build_labeled_day(ds, day, cfg, equity=args.equity)
            if log is None or sidecar is None:
                continue

            chosen = per_day_choice(
                day_rows,
                thresholds["entry_threshold"],
                thresholds["side_threshold"],
                score_mode=score_mode,
                side_score_weight=side_score_weight,
                policy_mode=policy_mode,
                direction_mode=direction_mode,
            )
            if chosen is not None:
                bar_index = int(chosen["bar_index"])
                bar = next((b for b in log.bars if b.bar_index == bar_index), None)
                if bar is not None:
                    pnl = compute_time_stop_pnl_for_direction(bar, sidecar, str(chosen["effective_direction"]))
                    if pnl is not None:
                        layer2_trades.append({
                            "fold_idx": fold_idx,
                            "day": day,
                            "bar_index": bar_index,
                            "direction": str(chosen["effective_direction"]),
                            "route_source": str(chosen.get("route_source", "")),
                            "pnl": float(pnl),
                            "entry_value_raw": float(chosen["entry_value_raw"]) if pd.notna(chosen["entry_value_raw"]) else None,
                            "oracle_slice_outcome": str(chosen["oracle_slice_outcome"]),
                        })

            baseline_bar, baseline_direction, teacher_name = teacher_baseline_choice(log)
            if baseline_bar is not None and baseline_direction is not None and teacher_name is not None:
                key = f"{teacher_name}_{baseline_direction}"
                eh = baseline_bar.labels.exit_headroom_by_selection.get(key) or {}
                pnl = eh.get("time_stop_pnl")
                if pnl is not None:
                    baseline_trades.append({
                        "fold_idx": fold_idx,
                        "day": day,
                        "bar_index": int(baseline_bar.bar_index),
                        "direction": baseline_direction,
                        "pnl": float(pnl),
                        "entry_value_raw": float(max(
                            x for x in (
                                baseline_bar.labels.best_forward_pnl_call,
                                baseline_bar.labels.best_forward_pnl_put,
                            ) if x is not None
                        )) if (
                            baseline_bar.labels.best_forward_pnl_call is not None
                            or baseline_bar.labels.best_forward_pnl_put is not None
                        ) else None,
                        "oracle_slice_outcome": str(next(
                            (row for _, row in day_rows.iterrows() if int(row["bar_index"]) == int(baseline_bar.bar_index)),
                            pd.Series({"oracle_slice_outcome": ""}),
                        )["oracle_slice_outcome"]),
                    })

    layer2_df = pd.DataFrame(layer2_trades)
    baseline_df = pd.DataFrame(baseline_trades)

    layer2_metrics = replay_metrics_from_pnls(layer2_df["pnl"].tolist() if not layer2_df.empty else [], args.equity)
    baseline_metrics = replay_metrics_from_pnls(baseline_df["pnl"].tolist() if not baseline_df.empty else [], args.equity)

    if not processed_folds:
        raise RuntimeError(f"No replayable folds found under {args.run_dir}")
    total_test_days = sum(len(fold["test_days"]) for fold in processed_folds)
    layer2_metrics["trades_per_day"] = float(len(layer2_df) / max(total_test_days, 1))
    baseline_metrics["trades_per_day"] = float(len(baseline_df) / max(total_test_days, 1))
    layer2_metrics["call_pct"] = float((layer2_df["direction"] == "call").mean()) if not layer2_df.empty else 0.0
    baseline_metrics["call_pct"] = float((baseline_df["direction"] == "call").mean()) if not baseline_df.empty else 0.0
    layer2_metrics["mean_entry_value_raw"] = float(layer2_df["entry_value_raw"].dropna().mean()) if not layer2_df.empty else 0.0
    baseline_metrics["mean_entry_value_raw"] = float(baseline_df["entry_value_raw"].dropna().mean()) if not baseline_df.empty else 0.0

    slice_report = {}
    for slice_name in ("abstention", "side_error"):
        l_slice = layer2_df[layer2_df["oracle_slice_outcome"] == slice_name] if not layer2_df.empty else pd.DataFrame()
        b_slice = baseline_df[baseline_df["oracle_slice_outcome"] == slice_name] if not baseline_df.empty else pd.DataFrame()
        slice_report[slice_name] = {
            "layer2_trades": int(len(l_slice)),
            "layer2_mean_pnl": float(l_slice["pnl"].mean()) if not l_slice.empty else 0.0,
            "baseline_trades": int(len(b_slice)),
            "baseline_mean_pnl": float(b_slice["pnl"].mean()) if not b_slice.empty else 0.0,
        }

    verdict = {
        "dataset": args.dataset,
        "run_dir": args.run_dir,
        "direction_mode": direction_mode,
        "policy_mode": policy_mode,
        "score_mode": score_mode,
        "side_score_weight": side_score_weight,
        "elapsed_seconds": time.time() - t0,
        "layer2": layer2_metrics,
        "post_a1_teacher_baseline": baseline_metrics,
        "slice_report": slice_report,
        "beats_baseline_pf": bool(layer2_metrics["pf"] > baseline_metrics["pf"]),
        "beats_baseline_dd": bool(layer2_metrics["max_dd_pct"] < baseline_metrics["max_dd_pct"]),
        "coverage_in_band": bool(0.60 <= layer2_metrics["trades_per_day"] <= 1.00),
    }

    ensure_dir(args.run_dir)
    if not layer2_df.empty:
        layer2_df.to_csv(os.path.join(args.run_dir, "layer2_trades.csv"), index=False)
    if not baseline_df.empty:
        baseline_df.to_csv(os.path.join(args.run_dir, "teacher_baseline_trades.csv"), index=False)
    save_json(os.path.join(args.run_dir, "replay_report.json"), verdict)

    print("Layer-2 replay complete")
    print(f"Layer2:  trades={layer2_metrics['trades']:.0f} pf={layer2_metrics['pf']:.3f} dd={layer2_metrics['max_dd_pct']:.1f}% tpd={layer2_metrics['trades_per_day']:.3f}")
    print(f"Teacher: trades={baseline_metrics['trades']:.0f} pf={baseline_metrics['pf']:.3f} dd={baseline_metrics['max_dd_pct']:.1f}% tpd={baseline_metrics['trades_per_day']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
