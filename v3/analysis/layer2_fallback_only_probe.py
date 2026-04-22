"""Cheap fallback-only probe on top of the detach-side Layer-2 baseline.

Hypothesis:
- the route-aware branch failed because teacher bars dominated training
- the real unresolved task is only the no-teacher subset
- a model trained only on non-teacher bars may improve fallback routing
  on the exact entry bars already chosen by the detach-side winner

This script keeps the detach-side chosen bars fixed and changes only the
action taken on bars where no teacher fired.

Controls reported on the same chosen bars:
- teacher+put
- teacher+call
- teacher+flat
- teacher+oracle_best
- teacher+put_or_flat_model
- teacher+fallback_model

The fallback model is deliberately cheap:
- train on pre-test non-teacher rows only
- two HistGradientBoostingRegressor heads
- targets are asinh(time_stop_pnl_call/100) and asinh(time_stop_pnl_put/100)
- flat is implicit with value 0.0 at policy time
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    load_export_bundle,
    replay_metrics_from_pnls,
    teacher_direction_hint_from_row,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer2_fallback_only_probe")


@dataclass
class FoldModels:
    call_model: HistGradientBoostingRegressor
    put_model: HistGradientBoostingRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fallback-only Layer-2 probe on detach-side chosen bars.")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_chosen_with_context(dataset_path: str, baseline_run_dir: str) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    bundle = load_export_bundle(dataset_path)
    rows: pd.DataFrame = bundle["rows"].copy()
    meta = bundle["meta"]
    feature_names = list(meta["feature_names"])
    folds = list(meta["folds"])

    trades = pd.read_csv(os.path.join(baseline_run_dir, "layer2_trades.csv"))
    join_cols = list(dict.fromkeys([
        "day",
        "bar_index",
        "fold_id",
        "teacher_any_triggered",
        "orc_buy_call",
        "orc_buy_put",
        "failed_break_buy_call",
        "failed_break_buy_put",
        "time_stop_pnl_call",
        "time_stop_pnl_put",
        "has_passing_call",
        "has_passing_put",
        *feature_names,
    ]))
    chosen = trades.merge(
        rows[join_cols],
        left_on=["day", "bar_index", "fold_idx"],
        right_on=["day", "bar_index", "fold_id"],
        how="left",
        validate="one_to_one",
    )
    chosen["teacher_direction"] = chosen.apply(teacher_direction_hint_from_row, axis=1)
    chosen["route_source"] = np.where(chosen["teacher_direction"] != "", "teacher", "fallback")
    return chosen, folds, feature_names


def _fit_head(
    train_df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    availability_col: str,
    seed: int,
) -> HistGradientBoostingRegressor:
    mask = train_df[target_col].notna() & (train_df[availability_col] > 0.5)
    if int(mask.sum()) < 100:
        raise RuntimeError(f"Not enough rows to train {target_col}: {int(mask.sum())}")
    y_raw = train_df.loc[mask, target_col].to_numpy(dtype=np.float32)
    y = np.arcsinh(y_raw / 100.0)
    w = np.abs(y_raw)
    p95 = float(np.percentile(w, 95)) if len(w) > 5 else float(np.max(w))
    sample_weight = np.clip(w, 1.0, max(p95, 1.0)).astype(np.float32)
    X = train_df.loc[mask, feature_names].to_numpy(dtype=np.float32)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=200,
        min_samples_leaf=50,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _train_models(
    rows: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_names: list[str],
    seed: int,
) -> dict[int, FoldModels]:
    nonteacher = rows[rows["teacher_any_triggered"] <= 0.5].copy()
    models: dict[int, FoldModels] = {}
    for fold in folds:
        fold_idx = int(fold["fold_idx"])
        pretest_days = set(fold["train_days"]) | set(fold["val_days"])
        train_df = nonteacher[nonteacher["day"].isin(pretest_days)].copy()
        models[fold_idx] = FoldModels(
            call_model=_fit_head(train_df, feature_names, "time_stop_pnl_call", "has_passing_call", seed + fold_idx * 10 + 1),
            put_model=_fit_head(train_df, feature_names, "time_stop_pnl_put", "has_passing_put", seed + fold_idx * 10 + 2),
        )
    return models


def _fallback_model_direction(row: pd.Series, models: dict[int, FoldModels], feature_names: list[str]) -> str:
    fold_idx = int(row["fold_idx"])
    fm = models[fold_idx]
    x = row[feature_names].to_numpy(dtype=np.float32).reshape(1, -1)
    call_score = float(fm.call_model.predict(x)[0]) if float(row["has_passing_call"]) > 0.5 else float("-inf")
    put_score = float(fm.put_model.predict(x)[0]) if float(row["has_passing_put"]) > 0.5 else float("-inf")
    row["fallback_call_score"] = call_score
    row["fallback_put_score"] = put_score
    best = max(call_score, put_score, 0.0)
    if best <= 0.0:
        return ""
    return "call" if call_score >= put_score else "put"


def _apply_control(chosen: pd.DataFrame, label: str, models: dict[int, FoldModels] | None, feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in chosen.iterrows():
        teacher_direction = str(row["teacher_direction"])
        if teacher_direction in {"call", "put"}:
            direction = teacher_direction
        elif label == "teacher+put":
            direction = "put"
        elif label == "teacher+call":
            direction = "call"
        elif label == "teacher+flat":
            direction = ""
        elif label == "teacher+oracle_best":
            call_val = float(row["time_stop_pnl_call"]) if pd.notna(row["time_stop_pnl_call"]) else float("-inf")
            put_val = float(row["time_stop_pnl_put"]) if pd.notna(row["time_stop_pnl_put"]) else float("-inf")
            best = max(call_val, put_val, 0.0)
            if best <= 0.0:
                direction = ""
            else:
                direction = "call" if call_val >= put_val else "put"
        elif label == "teacher+put_or_flat_model":
            if models is None:
                raise ValueError("teacher+put_or_flat_model requires models")
            fold_idx = int(row["fold_idx"])
            fm = models[fold_idx]
            if float(row["has_passing_put"]) <= 0.5:
                direction = ""
            else:
                x = row[feature_names].to_numpy(dtype=np.float32).reshape(1, -1)
                put_score = float(fm.put_model.predict(x)[0])
                direction = "put" if put_score > 0.0 else ""
        elif label == "teacher+fallback_model":
            if models is None:
                raise ValueError("teacher+fallback_model requires models")
            direction = _fallback_model_direction(row, models, feature_names)
        else:
            raise ValueError(f"Unknown control label {label!r}")

        if direction == "call":
            pnl = row["time_stop_pnl_call"]
        elif direction == "put":
            pnl = row["time_stop_pnl_put"]
        else:
            pnl = None
        if pnl is None or pd.isna(pnl):
            continue
        route_source = "teacher" if teacher_direction in {"call", "put"} else ("flat" if direction == "" else "fallback")
        rows.append({
            "fold_idx": int(row["fold_idx"]),
            "day": str(row["day"]),
            "bar_index": int(row["bar_index"]),
            "direction": direction,
            "route_source": route_source,
            "pnl": float(pnl),
            "entry_value_raw": float(row["entry_value_raw"]) if pd.notna(row["entry_value_raw"]) else None,
            "oracle_slice_outcome": str(row["oracle_slice_outcome"]),
        })
    return pd.DataFrame(rows)


def _metrics_for_trades(trades: pd.DataFrame, equity: float, n_days: int) -> dict[str, float]:
    if trades.empty:
        return {
            "pf": 0.0,
            "dd_pct": 0.0,
            "mean_pnl": 0.0,
            "trades": 0.0,
            "trades_per_day": 0.0,
            "call_pct": 0.0,
            "teacher_trades": 0.0,
            "fallback_trades": 0.0,
            "flat_days": float(n_days),
        }
    m = replay_metrics_from_pnls(trades["pnl"].astype(float).tolist(), equity)
    m["mean_pnl"] = float(trades["pnl"].mean())
    m["trades"] = float(len(trades))
    m["trades_per_day"] = float(len(trades) / max(n_days, 1))
    m["call_pct"] = float((trades["direction"] == "call").mean())
    m["teacher_trades"] = float((trades["route_source"] == "teacher").sum())
    m["fallback_trades"] = float((trades["route_source"] == "fallback").sum())
    traded_days = trades["day"].nunique()
    m["flat_days"] = float(max(n_days - traded_days, 0))
    m["dd_pct"] = float(m["max_dd_pct"])
    return m


def _fallback_subset_metrics(trades: pd.DataFrame) -> dict[str, float]:
    fallback = trades[trades["route_source"] == "fallback"]
    if fallback.empty:
        return {
            "n": 0.0,
            "mean_pnl": 0.0,
            "call_pct": 0.0,
            "put_pct": 0.0,
        }
    return {
        "n": float(len(fallback)),
        "mean_pnl": float(fallback["pnl"].mean()),
        "call_pct": float((fallback["direction"] == "call").mean()),
        "put_pct": float((fallback["direction"] == "put").mean()),
    }


def _print_control(label: str, overall: dict[str, float], fallback_subset: dict[str, float]) -> None:
    print(
        f"{label:24s} PF={overall['pf']:.3f} DD={overall['dd_pct']:.1f}% "
        f"TPD={overall['trades_per_day']:.3f} mean={overall['mean_pnl']:+.1f}$ "
        f"teacher={int(overall['teacher_trades'])} fallback={int(overall['fallback_trades'])} "
        f"fallback_mean={fallback_subset['mean_pnl']:+.1f}$"
    )


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_export_bundle(args.dataset)
    rows: pd.DataFrame = bundle["rows"].copy()
    chosen, folds, feature_names = _load_chosen_with_context(args.dataset, args.baseline_run_dir)
    models = _train_models(rows, folds, feature_names, args.seed)
    total_days = sum(len(fold["test_days"]) for fold in folds)
    fold_days = {int(f["fold_idx"]): len(f["test_days"]) for f in folds}

    results: dict[str, Any] = {}
    controls = [
        "teacher+put",
        "teacher+call",
        "teacher+flat",
        "teacher+oracle_best",
        "teacher+put_or_flat_model",
        "teacher+fallback_model",
    ]
    print("=" * 100)
    print("Fallback-only probe on detach-side chosen bars")
    print("=" * 100)
    print(f"Chosen bars: {len(chosen)} teacher={int((chosen['route_source'] == 'teacher').sum())} fallback={int((chosen['route_source'] == 'fallback').sum())}")

    for label in controls:
        needs_models = label in {"teacher+put_or_flat_model", "teacher+fallback_model"}
        sim = _apply_control(chosen, label, models if needs_models else None, feature_names)
        overall = _metrics_for_trades(sim, args.equity, total_days)
        per_fold = {
            fold_idx: _metrics_for_trades(sim[sim["fold_idx"] == fold_idx], args.equity, fold_days[fold_idx])
            for fold_idx in sorted(fold_days)
        }
        fallback_subset = _fallback_subset_metrics(sim)
        results[label] = {
            "overall": overall,
            "per_fold": per_fold,
            "fallback_subset": fallback_subset,
        }
        _print_control(label, overall, fallback_subset)

    results["meta"] = {
        "baseline_run_dir": args.baseline_run_dir,
        "dataset": args.dataset,
        "seed": args.seed,
        "chosen_bars": int(len(chosen)),
        "teacher_bars": int((chosen["route_source"] == "teacher").sum()),
        "fallback_bars": int((chosen["route_source"] == "fallback").sum()),
        "model": "HistGradientBoostingRegressor",
        "training_scope": "pretest non-teacher rows only (train + val days per fold)",
    }

    out_json = os.path.join(args.out_dir, "fallback_only_probe.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print()
    print(f"Saved results: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
