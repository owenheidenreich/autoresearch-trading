"""Side-prior audit: is the call bias a global prior or per-bar discrimination?

Loads a single window's UnifiedActionPredictor, runs it on every row of the
action-surface dataset (train + val + OOS), and tabulates the model's per-bar
best-call score vs best-put score, stratified by truth's best side.

If the model has a learned global side prior, mean(score[best_call]) will
exceed mean(score[best_put]) across the dataset and stay positive even on
bars where truth says put. If discrimination is intact, put scores rise
sharply on put-best bars.

Usage:

    .venv/bin/python -m v3.analysis.side_prior_audit \\
        --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \\
        --model v3/artifacts/layer2_unified_policy_spx_live_hybrid_001_seed42/seed_42 \\
        --window 5 \\
        --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42.npz \\
        --out v3/artifacts/side_prior_audit/seed42_window05.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from v3.harness.rolling_windows import generate_rolling_windows
from v3.layer2.common import load_export_bundle, load_pickle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="Path to seed_NN dir holding window_*/model.pkl")
    p.add_argument("--window", type=int, required=True, help="Window index (e.g. 5 for W5)")
    p.add_argument("--simulated-l3-oracle", default="", help="Optional oracle .npz; when set, label uses hybrid_live target")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _per_side_best(scores: np.ndarray, tradeable: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = scores[:, 1:].astype(np.float64).copy()
    tokens[~tradeable[:, 1:]] = -np.inf
    call = tokens[:, :top_k]
    put = tokens[:, top_k:]
    best_call = call.max(axis=1)
    best_put = put.max(axis=1)
    return best_call, best_put


def _stratified(label_best_side: np.ndarray, model_best_call: np.ndarray, model_best_put: np.ndarray, scope_mask: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if scope_mask.sum() == 0:
        return out
    mb_call = model_best_call[scope_mask]
    mb_put = model_best_put[scope_mask]
    finite = np.isfinite(mb_call) & np.isfinite(mb_put)
    if finite.sum() == 0:
        return out
    mb_call = mb_call[finite]
    mb_put = mb_put[finite]
    side_local = label_best_side[scope_mask][finite]
    out["n_bars"] = int(finite.sum())
    out["n_truth_call_best"] = int((side_local == "call").sum())
    out["n_truth_put_best"] = int((side_local == "put").sum())
    out["model_best_call_mean"] = float(mb_call.mean())
    out["model_best_call_median"] = float(np.median(mb_call))
    out["model_best_put_mean"] = float(mb_put.mean())
    out["model_best_put_median"] = float(np.median(mb_put))
    out["model_call_minus_put_mean"] = float((mb_call - mb_put).mean())
    out["model_call_minus_put_median"] = float(np.median(mb_call - mb_put))
    out["frac_model_call_above_put"] = float((mb_call > mb_put).mean())
    return out


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bundle = load_export_bundle(args.dataset)
    meta = bundle["meta"]
    rows: pd.DataFrame = bundle["rows"]
    top_k = int(meta["top_k_contracts_per_side"])
    scalar = rows.loc[:, meta["scalar_feature_names"]].to_numpy(dtype=np.float32)
    seq = bundle["sequence_features"]
    seq_mask = bundle["sequence_mask"]
    contracts = bundle["contract_features"]
    contract_mask = bundle["contract_mask"]
    tradeable_mask = np.nan_to_num(bundle["action_labels"]["tradeable_mask"], nan=0.0) > 0.5

    # Truth's best-side per bar — use simulated-L3 oracle if provided (matches training target),
    # else fall back to time-stop utility_raw.
    if args.simulated_l3_oracle:
        z = np.load(args.simulated_l3_oracle, allow_pickle=True)
        l3_pnl = z["l3_exit_pnl"]
        label_for_side = l3_pnl.astype(np.float64).copy()
        label_for_side[:, 0] = 0.0
    else:
        label_for_side = bundle["action_labels"]["utility_raw"].astype(np.float64).copy()
    label_best_call, label_best_put = _per_side_best(label_for_side, tradeable_mask, top_k)
    label_best_side = np.where(
        np.nan_to_num(label_best_put, nan=-np.inf) > np.nan_to_num(label_best_call, nan=-np.inf),
        "put",
        "call",
    )
    label_call_minus_put = label_best_call - label_best_put

    # Map each row to its rolling-window OOS phase relative to args.window
    unique_days = sorted(rows["day"].astype(str).unique().tolist())
    windows = generate_rolling_windows(unique_days)
    target = next((w for w in windows if int(w.window_idx) == int(args.window)), None)
    if target is None:
        raise RuntimeError(f"Window {args.window} not found in {len(windows)} windows")
    train_days = set(target.train_days)
    val_days = set(target.val_days)
    oos_days = set(target.oos_days)
    days = rows["day"].astype(str).to_numpy()
    is_train = np.isin(days, list(train_days))
    is_val = np.isin(days, list(val_days))
    is_oos = np.isin(days, list(oos_days))

    # Load model and predict
    model_path = os.path.join(args.model, f"window_{int(args.window):02d}", "model.pkl")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    predictor = load_pickle(model_path)
    pred = predictor.predict(scalar, seq, seq_mask, contracts, contract_mask, batch_size=2048)
    util = pred["utility"]  # shape (rows, 25), arcsinh-scale predicted utility
    model_best_call, model_best_put = _per_side_best(util, tradeable_mask, top_k)
    model_call_minus_put = model_best_call - model_best_put

    # Frac of bars where model picks call vs put as argmax of tokens
    tokens = util[:, 1:].astype(np.float64).copy()
    tokens[~tradeable_mask[:, 1:]] = -np.inf
    argmax_action = tokens.argmax(axis=1) + 1  # +1 because we excluded flat slot
    model_best_side = np.where(argmax_action <= top_k, "call", "put")

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "model_path": model_path,
        "window": int(args.window),
        "n_rows_total": int(len(rows)),
        "n_rows_train": int(is_train.sum()),
        "n_rows_val": int(is_val.sum()),
        "n_rows_oos": int(is_oos.sum()),
        "label_target": "hybrid_live (l3_exit_pnl)" if args.simulated_l3_oracle else "time_stop (utility_raw)",
        "label_best_side_counts": {
            "call": int((label_best_side == "call").sum()),
            "put": int((label_best_side == "put").sum()),
        },
        "model_best_side_counts": {
            "call": int((model_best_side == "call").sum()),
            "put": int((model_best_side == "put").sum()),
        },
    }

    # Cohort breakdowns
    for cohort_name, cohort_mask in [
        ("all", np.ones(len(rows), dtype=bool)),
        ("train", is_train),
        ("val", is_val),
        ("oos", is_oos),
    ]:
        # Model behavior on bars where truth says call vs put
        for truth in ("call", "put"):
            scope = cohort_mask & (label_best_side == truth)
            summary[f"{cohort_name}_truth_{truth}"] = _stratified(label_best_side, model_best_call, model_best_put, scope)
        # All bars in cohort
        summary[f"{cohort_name}_all"] = _stratified(label_best_side, model_best_call, model_best_put, cohort_mask)

    # Spot-check: 2024-04-01 alone
    spot_day = "2024-04-01"
    spot_mask = days == spot_day
    if spot_mask.any():
        summary["spot_2024_04_01"] = _stratified(label_best_side, model_best_call, model_best_put, spot_mask)
        summary["spot_2024_04_01"]["model_best_side_counts"] = {
            "call": int((model_best_side[spot_mask] == "call").sum()),
            "put": int((model_best_side[spot_mask] == "put").sum()),
        }
        summary["spot_2024_04_01"]["label_best_side_counts"] = {
            "call": int((label_best_side[spot_mask] == "call").sum()),
            "put": int((label_best_side[spot_mask] == "put").sum()),
        }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Wrote {args.out}")
    # Concise stdout summary
    print()
    print(f"{'cohort':<25} {'n_bars':>8} {'truth':<6} {'mb_call':>9} {'mb_put':>9} {'call-put':>10} {'frac_c>p':>9}")
    print("-" * 90)
    for cohort in ("all", "train", "val", "oos"):
        for truth in ("call", "put"):
            row = summary.get(f"{cohort}_truth_{truth}", {})
            if not row:
                continue
            print(f"{cohort+'/truth='+truth:<25} {row['n_bars']:>8} {truth:<6} {row['model_best_call_mean']:>9.4f} {row['model_best_put_mean']:>9.4f} {row['model_call_minus_put_mean']:>10.4f} {row['frac_model_call_above_put']:>9.3f}")
    print()
    if "spot_2024_04_01" in summary:
        s = summary["spot_2024_04_01"]
        print(f"2024-04-01 only: model picks call={s['model_best_side_counts']['call']} put={s['model_best_side_counts']['put']} | truth call={s['label_best_side_counts']['call']} put={s['label_best_side_counts']['put']}")
        print(f"   model best_call mean={s['model_best_call_mean']:.4f} best_put mean={s['model_best_put_mean']:.4f} call-put={s['model_call_minus_put_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
