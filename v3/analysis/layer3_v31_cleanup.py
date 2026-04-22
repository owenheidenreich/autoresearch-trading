"""Layer 3 v3.1 cleanup: teacher-augmented training + permutation importance.

Two sub-tasks per the strategic plan's Option 1:

A. **Teacher-augmented training.** Original Layer-3 trained on the 275
   chosen trades only. Augment with ~298 teacher-only trades from
   teacher_baseline_trades.csv. Train a fold-5 Layer-3 model on the
   combined 573 trades. Evaluate on:
   - In-sample 275 chosen trades (does augmentation hurt or help in-sample?)
   - OOS 20 chosen trades (does it improve generalization?)

B. **Permutation importance.** For the chosen-only fold-5 model AND
   the augmented model, shuffle each feature column and measure PF
   degradation. Identifies which features drive the exit decisions.
   Compare top features in-sample vs OOS — features whose importance
   collapses on OOS are signals that didn't generalize.

OOS context: layer3_oos_validation showed Layer-2 entry doesn't
generalize on the 20 cached days (PF 0.869 vs in-sample 1.472).
Permutation importance may explain WHAT the model is leaning on that
fails OOS.

Run:
    python -m v3.analysis.layer3_v31_cleanup
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v3.analysis.layer3_train_replay import (
    TRADE_STATE_NAMES,
    _build_per_trade_data,
    _flatten_to_rows,
    _replay_with_model,
    _agg,
)
from v3.analysis.layer3_oos_validation import (
    _apply_layer2_models,
    _build_oos_export_rows,
    _identify_oos_days,
    _select_oos_trades,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_labeled_day,
    load_export_bundle,
    load_json,
    load_pickle,
    replay_metrics_from_pnls,
)
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_v31_cleanup")
DEFAULT_THRESHOLD = 0.17
DEFAULT_SEED = 42
N_PERMUTATION_TRIALS = 3  # average over N shuffles per feature


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer 3 v3.1 cleanup.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-perm", type=int, default=N_PERMUTATION_TRIALS)
    return p.parse_args()


def _replay_pf(trade_data: list[dict], model: HistGradientBoostingClassifier,
               threshold: float, equity: float) -> float:
    sim = _replay_with_model(trade_data, model, threshold)
    if not sim:
        return 0.0
    df = pd.DataFrame(sim).sort_values(["day", "entry_bar"])
    pnls = df["exit_pnl"].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    return float(m["pf"])


def _replay_pf_with_shuffled_feature(
    trade_data: list[dict], model: HistGradientBoostingClassifier,
    threshold: float, equity: float, feature_idx: int, rng: np.random.Generator,
) -> float:
    """Replay PF after shuffling one feature column across all (trade, bar) rows."""
    # Collect all values for this feature column across all bars
    all_vals = []
    for td in trade_data:
        for b in td["per_bar"]:
            feats = np.concatenate([b["l2_feats"], b["trade_state"]])
            all_vals.append(feats[feature_idx])
    all_vals = np.array(all_vals, dtype=np.float32)
    rng.shuffle(all_vals)

    # Make a temporary copy of trade_data with shuffled feature_idx
    idx = 0
    shuffled = []
    for td in trade_data:
        td_copy = dict(td)
        per_bar_copy = []
        for b in td["per_bar"]:
            feats = np.concatenate([b["l2_feats"], b["trade_state"]]).copy()
            feats[feature_idx] = all_vals[idx]
            idx += 1
            # Re-split into l2_feats + trade_state
            n_l2 = len(b["l2_feats"])
            new_l2 = feats[:n_l2]
            new_ts = feats[n_l2:]
            per_bar_copy.append({**b, "l2_feats": new_l2, "trade_state": new_ts})
        td_copy["per_bar"] = per_bar_copy
        shuffled.append(td_copy)
    return _replay_pf(shuffled, model, threshold, equity)


def _permutation_importance(
    trade_data: list[dict], model: HistGradientBoostingClassifier,
    threshold: float, equity: float, n_features: int, n_trials: int,
    seed: int, label: str,
) -> list[tuple[int, float, float]]:
    """Returns list of (feature_idx, mean_pf_drop, std_pf_drop) sorted by drop."""
    base_pf = _replay_pf(trade_data, model, threshold, equity)
    print(f"  [{label}] baseline PF: {base_pf:.4f}")
    importance: list[tuple[int, float, float]] = []
    for fi in range(n_features):
        drops = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + fi * 100 + trial)
            shuf_pf = _replay_pf_with_shuffled_feature(
                trade_data, model, threshold, equity, fi, rng,
            )
            drops.append(base_pf - shuf_pf)
        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))
        importance.append((fi, mean_drop, std_drop))
        if (fi + 1) % 20 == 0:
            print(f"  [{label}] processed {fi+1}/{n_features} features...")
    importance.sort(key=lambda t: -abs(t[1]))
    return importance


def _build_in_sample_trade_data(
    in_sample_trades: pd.DataFrame, ds: V2Dataset, cfg: GuardrailConfig,
    equity: float, day_cache: dict, paths_cache: dict, minute_map_cache: dict,
) -> list[dict]:
    out = []
    for _, trade in in_sample_trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=equity)
            if log_d is None:
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        td = _build_per_trade_data(
            trade, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            out.append(td)
    return out


def _build_oos_trade_data(
    ds: V2Dataset, cfg: GuardrailConfig, args, day_cache: dict,
    paths_cache: dict, minute_map_cache: dict,
) -> tuple[list[dict], dict]:
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds_meta = list(bundle["meta"]["folds"])
    feature_names = list(bundle["meta"]["feature_names"])
    fold4 = folds_meta[-1]
    fold4_test = list(fold4["test_days"])
    fold_dir = os.path.join(args.baseline_run_dir, "folds", str(int(fold4["fold_idx"])))
    manifest = load_pickle(os.path.join(args.baseline_run_dir, "manifest.pkl"))
    direction_mode = manifest.get("direction_mode", "teacher_if_triggered_else_put")
    score_mode = manifest.get("score_mode", "product")
    side_score_weight = float(manifest.get("side_score_weight", 0.15))
    calib = load_json(os.path.join(fold_dir, "calibration.json"))

    oos_days = _identify_oos_days(ds, fold4_test)
    print(f"  Building OOS export rows for {len(oos_days)} days...")
    df_oos, oos_day_cache = _build_oos_export_rows(ds, cfg, oos_days, args.equity)
    day_cache.update(oos_day_cache)
    df_pred = _apply_layer2_models(
        df_oos, fold_dir, feature_names, direction_mode, score_mode, side_score_weight,
        calib["entry_threshold"], calib["side_threshold"],
    )
    oos_trades = _select_oos_trades(
        df_pred, oos_day_cache, calib["entry_threshold"], calib["side_threshold"],
        score_mode, side_score_weight, direction_mode,
    )
    print(f"  OOS chosen trades: {len(oos_trades)}")

    out = []
    for _, trade in oos_trades.iterrows():
        day = str(trade["day"])
        log, sidecar = day_cache[day]
        td = _build_per_trade_data(
            trade, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            out.append(td)
    return out, calib


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    # === Build in-sample chosen trade data ===
    print()
    print("Building in-sample chosen trade data (275 trades)...")
    in_sample_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    day_cache: dict = {}
    paths_cache: dict = {}
    minute_map_cache: dict = {}
    in_sample_data = _build_in_sample_trade_data(
        in_sample_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  In-sample chosen trades: {len(in_sample_data)}")

    # === Build teacher-only trade data ===
    print()
    print("Building teacher-only trade data...")
    teacher_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "teacher_baseline_trades.csv"))
    print(f"  teacher_baseline_trades.csv: {len(teacher_trades)} trades")
    teacher_data = _build_in_sample_trade_data(
        teacher_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  Built teacher trade data: {len(teacher_data)}")

    # === Build OOS trade data ===
    print()
    print("Building OOS trade data...")
    oos_data, oos_calib = _build_oos_trade_data(
        ds, cfg, args, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  Built OOS trade data: {len(oos_data)}")

    n_features = len(in_sample_data[0]["per_bar"][0]["l2_feats"]) + len(TRADE_STATE_NAMES)
    feature_names = [f"l2_{i}" for i in range(n_features - len(TRADE_STATE_NAMES))] + TRADE_STATE_NAMES

    # === Sub-task A: train chosen-only and augmented models ===
    print()
    print("=" * 100)
    print("Sub-task A: training chosen-only and teacher-augmented Layer-3 models")
    print("=" * 100)

    # Chosen-only
    X_co, y_co, _ = _flatten_to_rows(in_sample_data)
    print(f"  Chosen-only: {len(in_sample_data)} trades / {len(X_co)} bar-rows; pos_rate={y_co.mean():.3f}")
    model_co = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=args.seed + 999, early_stopping=False,
    )
    model_co.fit(X_co, y_co)

    # Augmented (chosen + teacher)
    augmented_data = in_sample_data + teacher_data
    X_aug, y_aug, _ = _flatten_to_rows(augmented_data)
    print(f"  Augmented:   {len(augmented_data)} trades / {len(X_aug)} bar-rows; pos_rate={y_aug.mean():.3f}")
    model_aug = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=args.seed + 1000, early_stopping=False,
    )
    model_aug.fit(X_aug, y_aug)

    # === Evaluate both models on in-sample chosen, in-sample teacher, and OOS ===
    print()
    print("Evaluating both models on three universes (PF at threshold 0.17)")
    print(f"{'model':<28}{'in_sample_chosen':>22}{'OOS':>10}")
    eval_results = {}
    for label, model in [("chosen_only", model_co), ("teacher_augmented", model_aug)]:
        pf_in = _replay_pf(in_sample_data, model, args.threshold, args.equity)
        pf_oos = _replay_pf(oos_data, model, args.threshold, args.equity)
        eval_results[label] = {"in_sample_pf": pf_in, "oos_pf": pf_oos}
        print(f"  {label:<26}{pf_in:>22.3f}{pf_oos:>10.3f}")

    # === Sub-task B: permutation importance ===
    print()
    print("=" * 100)
    print("Sub-task B: permutation importance on the chosen-only model (in-sample + OOS)")
    print("=" * 100)
    print("  (averaged over n_trials shuffles per feature)")

    print()
    print("In-sample permutation importance:")
    perm_in_sample = _permutation_importance(
        in_sample_data, model_co, args.threshold, args.equity,
        n_features, args.n_perm, args.seed + 7000, "in-sample",
    )
    print()
    print("OOS permutation importance:")
    perm_oos = _permutation_importance(
        oos_data, model_co, args.threshold, args.equity,
        n_features, args.n_perm, args.seed + 8000, "OOS",
    )

    # === Print top-15 features by absolute importance for each ===
    print()
    print("=" * 100)
    print("Top 15 features by IN-SAMPLE permutation importance")
    print("=" * 100)
    print(f"{'rank':<6}{'feat_idx':>10}{'feat_name':<30}{'mean_drop':>14}{'std':>10}")
    for rank, (fi, drop, std) in enumerate(perm_in_sample[:15], start=1):
        name = feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"
        print(f"{rank:<6}{fi:>10}{name:<30}{drop:>14.4f}{std:>10.4f}")

    print()
    print("=" * 100)
    print("Top 15 features by OOS permutation importance")
    print("=" * 100)
    print(f"{'rank':<6}{'feat_idx':>10}{'feat_name':<30}{'mean_drop':>14}{'std':>10}")
    for rank, (fi, drop, std) in enumerate(perm_oos[:15], start=1):
        name = feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"
        print(f"{rank:<6}{fi:>10}{name:<30}{drop:>14.4f}{std:>10.4f}")

    # === Identify features with biggest in-sample vs OOS divergence ===
    perm_in_dict = {fi: drop for fi, drop, _ in perm_in_sample}
    perm_oos_dict = {fi: drop for fi, drop, _ in perm_oos}
    div = []
    for fi in range(n_features):
        in_drop = perm_in_dict.get(fi, 0.0)
        oos_drop = perm_oos_dict.get(fi, 0.0)
        div.append((fi, in_drop, oos_drop, in_drop - oos_drop))
    div.sort(key=lambda t: -abs(t[3]))

    print()
    print("=" * 100)
    print("Top 15 features by IN-SAMPLE vs OOS divergence (in - oos importance)")
    print("=" * 100)
    print(f"{'rank':<6}{'feat_idx':>10}{'feat_name':<30}{'in_drop':>10}{'oos_drop':>10}{'gap':>10}")
    for rank, (fi, in_drop, oos_drop, gap) in enumerate(div[:15], start=1):
        name = feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"
        print(f"{rank:<6}{fi:>10}{name:<30}{in_drop:>10.4f}{oos_drop:>10.4f}{gap:>10.4f}")

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_features": int(n_features),
            "n_in_sample_trades": int(len(in_sample_data)),
            "n_teacher_trades": int(len(teacher_data)),
            "n_oos_trades": int(len(oos_data)),
            "threshold": float(args.threshold),
            "n_permutation_trials": int(args.n_perm),
        },
        "subtask_A_eval": eval_results,
        "subtask_B_in_sample_top15": [
            {"rank": r + 1, "feat_idx": fi,
             "feat_name": feature_names[fi] if fi < len(feature_names) else f"feat_{fi}",
             "mean_drop": drop, "std_drop": std}
            for r, (fi, drop, std) in enumerate(perm_in_sample[:15])
        ],
        "subtask_B_oos_top15": [
            {"rank": r + 1, "feat_idx": fi,
             "feat_name": feature_names[fi] if fi < len(feature_names) else f"feat_{fi}",
             "mean_drop": drop, "std_drop": std}
            for r, (fi, drop, std) in enumerate(perm_oos[:15])
        ],
        "in_sample_vs_oos_divergence_top15": [
            {"rank": r + 1, "feat_idx": fi,
             "feat_name": feature_names[fi] if fi < len(feature_names) else f"feat_{fi}",
             "in_drop": in_drop, "oos_drop": oos_drop, "gap": gap}
            for r, (fi, in_drop, oos_drop, gap) in enumerate(div[:15])
        ],
    }
    out = os.path.join(args.out_dir, "v31_cleanup.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
