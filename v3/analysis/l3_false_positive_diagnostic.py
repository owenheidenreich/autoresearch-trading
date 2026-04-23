"""Phase 1C — L3 false-positive characterization on V1+L3 OOS trades.

For the 20 OOS V1+L3 composed trades, identify "false positives" —
where L3 cut a V1 trade that would have been profitable. Define:
  FP = (V1 PnL − V1+L3 PnL) > $500 AND V1+L3 exit was in first 10 bars

For each trade (FP and non-FP), retrain the augmented L3 model on
all in-sample data (chosen + teacher) and dump per-bar feature values
+ predict_proba at exit. Compare FP vs non-FP feature distributions
to identify systematic patterns.

Acceptance: if 2+ FPs share a clear feature signature → propose a
smarter exit rule for Phase 2/3 testing.
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
)
from v3.analysis.layer3_v31_cleanup import (
    _build_in_sample_trade_data,
    _build_oos_trade_data,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import DEFAULT_DATASET_PATH, load_export_bundle


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_V1_ALONE_OOS = os.path.join(
    "v3", "artifacts", "layer2_directional_variants", "oos_trades_V1.csv",
)
DEFAULT_V1_L3_OOS = os.path.join(
    "v3", "artifacts", "layer2_directional_composed_oos", "composed_oos_trades_V1.csv",
)
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "l3_false_positive_diagnostic")
DEFAULT_THRESHOLD = 0.19
DEFAULT_SEED = 42
FP_DELTA_USD = 500.0
FP_BARS_THRESHOLD = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--v1-alone-oos", default=DEFAULT_V1_ALONE_OOS)
    p.add_argument("--v1-l3-oos", default=DEFAULT_V1_L3_OOS)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--fp-delta-usd", type=float, default=FP_DELTA_USD)
    p.add_argument("--fp-bars-threshold", type=int, default=FP_BARS_THRESHOLD)
    return p.parse_args()


def _replay_with_features(
    trade_data: list[dict],
    model: HistGradientBoostingClassifier,
    threshold: float,
    feature_names: list[str],
) -> tuple[list[dict], list[dict]]:
    """For each trade, walk bars and exit at first P >= threshold.
    Return (per-trade summary, per-bar features at exit).
    """
    summary = []
    exit_features = []
    for td in trade_data:
        feats = np.stack([
            np.concatenate([b["l2_feats"], b["trade_state"]])
            for b in td["per_bar"]
        ])
        probs = model.predict_proba(feats)[:, 1]
        exit_idx = None
        trigger = "model"
        for i in range(len(probs)):
            if probs[i] >= threshold:
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = len(probs) - 1
            trigger = "time_stop_fallback"
        bar = td["per_bar"][exit_idx]
        summary.append({
            "fold_idx": td["fold_idx"],
            "day": td["day"],
            "entry_bar": td["entry_bar"],
            "exit_bar": bar["bar"],
            "direction": td["direction"],
            "exit_pnl": bar["current_pnl"],
            "trigger": trigger,
            "bars_held": bar["bar"] - td["entry_bar"],
            "exit_prob": float(probs[exit_idx]),
            "max_prob": float(probs.max()),
        })
        feat_dict = {
            "day": td["day"],
            "entry_bar": td["entry_bar"],
            "exit_bar": bar["bar"],
            "exit_prob": float(probs[exit_idx]),
        }
        all_feats = np.concatenate([bar["l2_feats"], bar["trade_state"]])
        for fi, fn in enumerate(feature_names):
            feat_dict[fn] = float(all_feats[fi])
        exit_features.append(feat_dict)
    return summary, exit_features


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset...", flush=True)
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)

    # === Build in-sample trade data (chosen + teacher) for training ===
    print(flush=True)
    print("Building chosen + teacher trade data for training augmented L3...", flush=True)
    in_sample_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    teacher_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "teacher_baseline_trades.csv"))
    day_cache: dict = {}
    paths_cache: dict = {}
    minute_map_cache: dict = {}
    in_sample_data = _build_in_sample_trade_data(
        in_sample_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    teacher_data = _build_in_sample_trade_data(
        teacher_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  chosen: {len(in_sample_data)}; teacher: {len(teacher_data)}", flush=True)

    # === Train augmented L3 ===
    augmented_data = in_sample_data + teacher_data
    X_aug, y_aug, _ = _flatten_to_rows(augmented_data)
    print(f"  Augmented training set: {len(augmented_data)} trades / {len(X_aug)} rows", flush=True)
    model_aug = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=args.seed + 1000, early_stopping=False,
    )
    model_aug.fit(X_aug, y_aug)
    print(f"  Augmented L3 trained.", flush=True)

    # === Build OOS V1 trade data ===
    print(flush=True)
    print("Building OOS V1 trade data...", flush=True)
    v1_alone = pd.read_csv(args.v1_alone_oos)
    v1_alone["day"] = v1_alone["day"].astype(str)
    print(f"  v1_alone OOS: {len(v1_alone)} trades", flush=True)

    # Build per-trade data for V1 OOS trades using V1's directions (puts forced)
    v1_oos_data = []
    for _, trade in v1_alone.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            from v3.layer2.common import build_labeled_day
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=args.equity)
            if log_d is None:
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        # Build a fake trade row matching what _build_per_trade_data needs
        fake_row = pd.Series({
            "day": day,
            "bar_index": int(trade["bar_index"]),
            "direction": str(trade["direction"]),
            "fold_idx": int(trade.get("fold_idx", -1)),
            "pnl": float(trade["pnl"]),
        })
        td = _build_per_trade_data(
            fake_row, log, sidecar, paths_cache, minute_map_cache, ds,
            session_end_bar=270, commission=2.0,
        )
        if td is not None:
            v1_oos_data.append(td)
    print(f"  Built {len(v1_oos_data)} V1 OOS trade datasets", flush=True)

    # === Replay V1 OOS through augmented L3 and dump features ===
    n_features = len(in_sample_data[0]["per_bar"][0]["l2_feats"]) + len(TRADE_STATE_NAMES)
    feature_names = (
        [f"l2_{i}" for i in range(n_features - len(TRADE_STATE_NAMES))]
        + TRADE_STATE_NAMES
    )
    summary, exit_features = _replay_with_features(
        v1_oos_data, model_aug, args.threshold, feature_names,
    )
    sim_df = pd.DataFrame(summary)
    feats_df = pd.DataFrame(exit_features)

    # === Compare to V1-alone PnL and identify FPs ===
    v1_alone_keyed = v1_alone.set_index(["day", "bar_index"])["pnl"].to_dict()
    sim_df["v1_alone_pnl"] = sim_df.apply(
        lambda r: v1_alone_keyed.get((str(r["day"]), int(r["entry_bar"])), float("nan")),
        axis=1,
    )
    sim_df["delta_vs_v1"] = sim_df["v1_alone_pnl"] - sim_df["exit_pnl"]
    sim_df["is_false_positive"] = (
        (sim_df["delta_vs_v1"] >= args.fp_delta_usd)
        & (sim_df["bars_held"] <= args.fp_bars_threshold)
    )
    n_fp = int(sim_df["is_false_positive"].sum())
    print(flush=True)
    print(f"False positives identified: {n_fp} / {len(sim_df)}", flush=True)
    print(f"  (criterion: V1_alone PnL − V1+L3 PnL >= ${args.fp_delta_usd:.0f}", flush=True)
    print(f"   AND bars_held <= {args.fp_bars_threshold})", flush=True)

    # === Print FP table ===
    print(flush=True)
    print("=" * 100)
    print("False positive trades")
    print("=" * 100)
    print(f"{'day':<12}{'entry':>7}{'exit':>6}{'bars':>6}{'V1+L3$':>10}{'V1$':>10}"
          f"{'delta':>10}{'p_exit':>10}", flush=True)
    fp_df = sim_df[sim_df["is_false_positive"]].sort_values("delta_vs_v1", ascending=False)
    for _, r in fp_df.iterrows():
        print(f"{r['day']:<12}{r['entry_bar']:>7}{r['exit_bar']:>6}{r['bars_held']:>6}"
              f"{r['exit_pnl']:>10.0f}{r['v1_alone_pnl']:>10.0f}"
              f"{r['delta_vs_v1']:>10.0f}{r['exit_prob']:>10.3f}", flush=True)

    # === Print non-FP trades for comparison ===
    print(flush=True)
    print("=" * 100)
    print("Non-false-positive trades (for comparison)")
    print("=" * 100)
    print(f"{'day':<12}{'entry':>7}{'exit':>6}{'bars':>6}{'V1+L3$':>10}{'V1$':>10}"
          f"{'delta':>10}{'p_exit':>10}", flush=True)
    nfp_df = sim_df[~sim_df["is_false_positive"]].sort_values("delta_vs_v1", ascending=False)
    for _, r in nfp_df.iterrows():
        print(f"{r['day']:<12}{r['entry_bar']:>7}{r['exit_bar']:>6}{r['bars_held']:>6}"
              f"{r['exit_pnl']:>10.0f}{r['v1_alone_pnl']:>10.0f}"
              f"{r['delta_vs_v1']:>10.0f}{r['exit_prob']:>10.3f}", flush=True)

    # === Per-feature comparison ===
    feats_df = feats_df.merge(
        sim_df[["day", "entry_bar", "is_false_positive", "delta_vs_v1", "bars_held"]],
        on=["day", "entry_bar"], how="left",
    )
    fp_feat = feats_df[feats_df["is_false_positive"]]
    nfp_feat = feats_df[~feats_df["is_false_positive"]]

    print(flush=True)
    print("=" * 100)
    print("Trade-state feature distributions at L3 exit bar (FP vs non-FP)")
    print("=" * 100)
    print(f"{'feature':<25}{'FP_mean':>12}{'FP_std':>10}{'nonFP_mean':>14}{'nonFP_std':>12}{'gap':>10}", flush=True)
    feat_comparison = []
    for fn in TRADE_STATE_NAMES:
        fp_vals = fp_feat[fn].dropna().values
        nfp_vals = nfp_feat[fn].dropna().values
        if len(fp_vals) == 0 or len(nfp_vals) == 0:
            continue
        fp_mean = float(np.mean(fp_vals))
        fp_std = float(np.std(fp_vals))
        nfp_mean = float(np.mean(nfp_vals))
        nfp_std = float(np.std(nfp_vals))
        gap = fp_mean - nfp_mean
        print(f"{fn:<25}{fp_mean:>12.3f}{fp_std:>10.3f}{nfp_mean:>14.3f}{nfp_std:>12.3f}{gap:>10.3f}", flush=True)
        feat_comparison.append({
            "feature": fn, "fp_mean": fp_mean, "fp_std": fp_std,
            "nonfp_mean": nfp_mean, "nonfp_std": nfp_std, "gap": gap,
        })

    # === Top L2 features by FP-vs-non-FP gap ===
    print(flush=True)
    print("=" * 100)
    print("Top 10 L2 features by |FP_mean − nonFP_mean| at exit bar")
    print("=" * 100)
    print(f"{'feature':<25}{'FP_mean':>12}{'nonFP_mean':>14}{'abs_gap':>12}", flush=True)
    l2_gaps = []
    for fn in feature_names:
        if fn in TRADE_STATE_NAMES:
            continue
        fp_vals = fp_feat[fn].dropna().values
        nfp_vals = nfp_feat[fn].dropna().values
        if len(fp_vals) == 0 or len(nfp_vals) == 0:
            continue
        fp_mean = float(np.mean(fp_vals))
        nfp_mean = float(np.mean(nfp_vals))
        gap = abs(fp_mean - nfp_mean)
        l2_gaps.append((fn, fp_mean, nfp_mean, gap))
    l2_gaps.sort(key=lambda t: -t[3])
    for fn, fpm, nfpm, g in l2_gaps[:10]:
        print(f"{fn:<25}{fpm:>12.3f}{nfpm:>14.3f}{g:>12.3f}", flush=True)

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_in_sample_chosen": int(len(in_sample_data)),
            "n_teacher": int(len(teacher_data)),
            "n_v1_oos": int(len(v1_oos_data)),
            "threshold": float(args.threshold),
            "fp_delta_usd": float(args.fp_delta_usd),
            "fp_bars_threshold": int(args.fp_bars_threshold),
        },
        "false_positives": fp_df.to_dict(orient="records"),
        "non_false_positives": nfp_df.to_dict(orient="records"),
        "trade_state_feature_comparison": feat_comparison,
        "top10_l2_gaps": [
            {"feature": fn, "fp_mean": fpm, "nonfp_mean": nfpm, "abs_gap": g}
            for fn, fpm, nfpm, g in l2_gaps[:10]
        ],
        "n_false_positives": n_fp,
    }
    out = os.path.join(args.out_dir, "l3_false_positive_diagnostic.json")
    with open(out, "w") as f:
        json.dump(
            payload, f, indent=2, sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )
    sim_df.to_csv(os.path.join(args.out_dir, "v1_l3_oos_with_v1_compare.csv"), index=False)
    feats_df.to_csv(os.path.join(args.out_dir, "v1_l3_oos_exit_features.csv"), index=False)
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    print(f"Saved: {os.path.join(args.out_dir, 'v1_l3_oos_with_v1_compare.csv')}", flush=True)
    print(f"Saved: {os.path.join(args.out_dir, 'v1_l3_oos_exit_features.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
