"""Stage A of the post-OOS research workstream: permutation importance
on the AUGMENTED Layer-3 model (vs chosen-only, which v31_cleanup
already covered).

Question: did augmentation teach the model to lean on regime-stable
features (`direction_is_call`, `poc_dist`, `ema_cross`, ...) instead of
the brittle ones the chosen-only model relied on (`minutes_to_close`,
`mfe_norm`, `trend_5min`, ...)?

If YES, the OOS lift (1.028 -> 1.537 with augmentation) is mechanically
explainable as proper regularization.
If NO, the OOS lift may be lucky-noise; augmentation just smoothed the
loss without improving feature reliance.

Compares augmented model's permutation importance (in-sample + OOS)
against the chosen-only model's importance (loaded from prior
v31_cleanup.json).

Run:
    python -m v3.analysis.layer3_aug_permutation_importance
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
    _flatten_to_rows,
    _replay_with_model,
)
from v3.analysis.layer3_v31_cleanup import (
    _build_in_sample_trade_data,
    _build_oos_trade_data,
    _permutation_importance,
    _replay_pf,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import load_export_bundle, DEFAULT_DATASET_PATH


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_aug_permutation_importance")
DEFAULT_PRIOR_JSON = os.path.join("v3", "artifacts", "layer3_v31_cleanup", "v31_cleanup.json")
DEFAULT_THRESHOLD = 0.17
DEFAULT_SEED = 42
N_PERMUTATION_TRIALS = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmented Layer-3 permutation importance.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--prior-json", default=DEFAULT_PRIOR_JSON,
                   help="v31_cleanup.json with chosen-only permutation importance.")
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-perm", type=int, default=N_PERMUTATION_TRIALS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    print("Building in-sample chosen trades (275)...")
    in_sample_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    day_cache: dict = {}
    paths_cache: dict = {}
    minute_map_cache: dict = {}
    in_sample_data = _build_in_sample_trade_data(
        in_sample_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  -> {len(in_sample_data)} chosen trade datasets")

    print("Building teacher-only trades (~298)...")
    teacher_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "teacher_baseline_trades.csv"))
    teacher_data = _build_in_sample_trade_data(
        teacher_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  -> {len(teacher_data)} teacher trade datasets")

    print("Building OOS trade data...")
    oos_data, _ = _build_oos_trade_data(
        ds, cfg, args, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  -> {len(oos_data)} OOS trade datasets")

    n_features = len(in_sample_data[0]["per_bar"][0]["l2_feats"]) + len(TRADE_STATE_NAMES)
    feature_names = (
        [f"l2_{i}" for i in range(n_features - len(TRADE_STATE_NAMES))]
        + TRADE_STATE_NAMES
    )

    # Decode l2_X feature names from V2Dataset
    v2_feature_names = list(ds.feature_names)
    decoded_names = [
        (v2_feature_names[i] if i < len(v2_feature_names) else feature_names[i])
        for i in range(n_features - len(TRADE_STATE_NAMES))
    ] + TRADE_STATE_NAMES

    # === Train augmented model ===
    print()
    print("Training AUGMENTED Layer-3 (chosen + teacher = 573 trades)...")
    augmented_data = in_sample_data + teacher_data
    X_aug, y_aug, _ = _flatten_to_rows(augmented_data)
    print(f"  Augmented: {len(augmented_data)} trades / {len(X_aug)} bar-rows; pos_rate={y_aug.mean():.3f}")
    model_aug = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=args.seed + 1000, early_stopping=False,
    )
    model_aug.fit(X_aug, y_aug)

    # Sanity: PFs match v31_cleanup
    pf_in = _replay_pf(in_sample_data, model_aug, args.threshold, args.equity)
    pf_oos = _replay_pf(oos_data, model_aug, args.threshold, args.equity)
    print(f"  Augmented PF: in_sample={pf_in:.3f}, OOS={pf_oos:.3f}")
    print(f"  (v31_cleanup reported: in_sample 4.527, OOS 1.537)")

    # === Permutation importance on augmented model ===
    print()
    print("=" * 100)
    print("Permutation importance on AUGMENTED model -- in-sample (275 chosen trades)")
    print("=" * 100)
    perm_aug_in_sample = _permutation_importance(
        in_sample_data, model_aug, args.threshold, args.equity,
        n_features, args.n_perm, args.seed + 11000, "AUG-in-sample",
    )

    print()
    print("=" * 100)
    print("Permutation importance on AUGMENTED model -- OOS (20 cached days)")
    print("=" * 100)
    perm_aug_oos = _permutation_importance(
        oos_data, model_aug, args.threshold, args.equity,
        n_features, args.n_perm, args.seed + 12000, "AUG-OOS",
    )

    # === Load prior chosen-only importance for comparison ===
    print()
    print("Loading prior chosen-only importance from v31_cleanup.json...")
    with open(args.prior_json) as f:
        prior = json.load(f)
    co_in_top15 = prior.get("subtask_B_in_sample_top15", [])
    co_oos_top15 = prior.get("subtask_B_oos_top15", [])

    # Build dicts for quick lookup
    co_in_dict = {r["feat_idx"]: r["mean_drop"] for r in co_in_top15}
    co_oos_dict = {r["feat_idx"]: r["mean_drop"] for r in co_oos_top15}

    # === Print top-15 augmented in-sample ===
    print()
    print("=" * 100)
    print("Top 15 features by AUGMENTED IN-SAMPLE permutation importance")
    print("=" * 100)
    print(f"{'rank':<6}{'feat_idx':>10}{'feat_name':<32}{'aug_drop':>10}{'co_drop':>10}{'delta':>10}")
    for rank, (fi, drop, std) in enumerate(perm_aug_in_sample[:15], start=1):
        name = decoded_names[fi] if fi < len(decoded_names) else f"feat_{fi}"
        co_drop = co_in_dict.get(fi, float("nan"))
        delta = drop - co_drop if np.isfinite(co_drop) else float("nan")
        print(f"{rank:<6}{fi:>10}{name:<32}{drop:>10.4f}{co_drop:>10.4f}{delta:>10.4f}")

    print()
    print("=" * 100)
    print("Top 15 features by AUGMENTED OOS permutation importance")
    print("=" * 100)
    print(f"{'rank':<6}{'feat_idx':>10}{'feat_name':<32}{'aug_drop':>10}{'co_drop':>10}{'delta':>10}")
    for rank, (fi, drop, std) in enumerate(perm_aug_oos[:15], start=1):
        name = decoded_names[fi] if fi < len(decoded_names) else f"feat_{fi}"
        co_drop = co_oos_dict.get(fi, float("nan"))
        delta = drop - co_drop if np.isfinite(co_drop) else float("nan")
        print(f"{rank:<6}{fi:>10}{name:<32}{drop:>10.4f}{co_drop:>10.4f}{delta:>10.4f}")

    # === Identify features that GREW most in importance OOS under augmentation ===
    perm_aug_in_dict = {fi: drop for fi, drop, _ in perm_aug_in_sample}
    perm_aug_oos_dict = {fi: drop for fi, drop, _ in perm_aug_oos}
    feat_changes = []
    for fi in range(n_features):
        co_oos = co_oos_dict.get(fi, 0.0)
        aug_oos = perm_aug_oos_dict.get(fi, 0.0)
        diff_oos = aug_oos - co_oos
        feat_changes.append((fi, co_oos, aug_oos, diff_oos))
    feat_changes.sort(key=lambda t: -abs(t[3]))

    print()
    print("=" * 100)
    print("Top 15 features by OOS importance shift (augmented vs chosen-only)")
    print("(positive = augmented model leans MORE on this feature OOS)")
    print("=" * 100)
    print(f"{'rank':<6}{'feat_idx':>10}{'feat_name':<32}{'co_oos':>10}{'aug_oos':>10}{'delta':>10}")
    for rank, (fi, co_oos, aug_oos, diff) in enumerate(feat_changes[:15], start=1):
        name = decoded_names[fi] if fi < len(decoded_names) else f"feat_{fi}"
        print(f"{rank:<6}{fi:>10}{name:<32}{co_oos:>10.4f}{aug_oos:>10.4f}{diff:>+10.4f}")

    # === Verdict ===
    print()
    print("=" * 100)
    print("Stage A Verdict")
    print("=" * 100)
    # Define "regime-stable" anchor features (top OOS for chosen-only)
    regime_stable_features = {"direction_is_call", "poc_dist", "ema_cross",
                              "force_index_2", "bars_since_break_below_first15",
                              "session_range_position", "first15_range_pct"}
    # Define "brittle" anchor features (top in-sample-only divergence)
    brittle_features = {"trend_5min", "minutes_to_close", "mfe_norm",
                        "omar_range_pct", "gamma_pressure"}

    aug_in_top15_names = [decoded_names[fi] for fi, _, _ in perm_aug_in_sample[:15] if fi < len(decoded_names)]
    n_brittle_in_top15 = sum(1 for n in aug_in_top15_names if n in brittle_features)
    n_stable_in_top15 = sum(1 for n in aug_in_top15_names if n in regime_stable_features)

    aug_oos_top15_names = [decoded_names[fi] for fi, _, _ in perm_aug_oos[:15] if fi < len(decoded_names)]
    n_stable_oos_top15 = sum(1 for n in aug_oos_top15_names if n in regime_stable_features)

    print(f"  Augmented in-sample top-15: {n_brittle_in_top15} brittle features, {n_stable_in_top15} regime-stable features")
    print(f"  Augmented OOS top-15:       {n_stable_oos_top15} regime-stable features")

    if n_brittle_in_top15 <= 1 and n_stable_oos_top15 >= 4:
        verdict = "GO -- augmented model leans on regime-stable features; OOS lift is mechanistically supported"
    elif n_brittle_in_top15 >= 3:
        verdict = f"FALSIFIED -- augmented model still leans on brittle features ({n_brittle_in_top15} in top-15 in-sample)"
    else:
        verdict = (f"WEAK -- partial evidence: brittle in-sample={n_brittle_in_top15}, "
                   f"regime-stable OOS={n_stable_oos_top15}. Augmentation helped but not decisively")
    print(f"  VERDICT: {verdict}")

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_in_sample": int(len(in_sample_data)),
            "n_teacher": int(len(teacher_data)),
            "n_oos": int(len(oos_data)),
            "n_features": int(n_features),
            "threshold": float(args.threshold),
            "n_perm_trials": int(args.n_perm),
            "augmented_in_sample_pf": float(pf_in),
            "augmented_oos_pf": float(pf_oos),
        },
        "augmented_top15_in_sample": [
            {"rank": r + 1, "feat_idx": fi,
             "feat_name": decoded_names[fi] if fi < len(decoded_names) else f"feat_{fi}",
             "mean_drop": drop, "std_drop": std,
             "chosen_only_drop": co_in_dict.get(fi, None)}
            for r, (fi, drop, std) in enumerate(perm_aug_in_sample[:15])
        ],
        "augmented_top15_oos": [
            {"rank": r + 1, "feat_idx": fi,
             "feat_name": decoded_names[fi] if fi < len(decoded_names) else f"feat_{fi}",
             "mean_drop": drop, "std_drop": std,
             "chosen_only_drop": co_oos_dict.get(fi, None)}
            for r, (fi, drop, std) in enumerate(perm_aug_oos[:15])
        ],
        "oos_importance_shift_top15": [
            {"rank": r + 1, "feat_idx": fi,
             "feat_name": decoded_names[fi] if fi < len(decoded_names) else f"feat_{fi}",
             "chosen_only_oos": co_oos, "augmented_oos": aug_oos, "delta": diff}
            for r, (fi, co_oos, aug_oos, diff) in enumerate(feat_changes[:15])
        ],
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "aug_permutation_importance.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
