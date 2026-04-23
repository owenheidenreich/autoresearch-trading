"""Phase R5 — Conditional V0/V1 rule using entry-bar intraday features.

R4 showed V0 wins 10/13 windows, V1 wins window 7 only. Within each
window though, some days are V0-favorable and some V1-favorable
(because V0's call selections occasionally fail). If we can detect
V1-favorable days at the chosen entry bar (using intraday-developing
features), a conditional rule "use V0 by default, switch to V1 when
the classifier flags today's entry as V1-favorable" could outperform
blanket V0.

Methodology:
  1. For each window, get per-day V0 PnL and V1 PnL from R4 artifacts
  2. Label: 1 if V1_pnl > V0_pnl + $50 (V1-favorable day)
  3. Features: 51 augmented features at the chosen entry bar
     (extracted from R3's OOS prediction pickles)
  4. Walk-forward training: train classifier on windows 0..k-1, test
     on window k
  5. Decision: if pred_prob >= threshold, use V1 direction; else V0
  6. Compute conditional-rule PnL per window + cross-window aggregate

Outputs:
  - per-window AUC, precision, recall
  - conditional-rule PF per window and aggregate
  - comparison to blanket V0, blanket V1, oracle upper bound
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_score,
    recall_score, roc_auc_score,
)

from v3.analysis.intraday_feature_audit import (
    CATEGORY_A_ADDITIONS,
    CATEGORY_B_SPECS,
)
from v3.layer2.common import (
    load_json,
    load_pickle,
    replay_metrics_from_pnls,
    save_json,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROLLING_DIR = os.path.join("v3", "artifacts", "rolling_l2")
DEFAULT_EVAL_DIR = os.path.join("v3", "artifacts", "rolling_directional_eval")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "conditional_directional_rule")
DEFAULT_SEED = 42
DEFAULT_LABEL_MARGIN = 50.0

CAT_B_FEATURE_NAMES = tuple(s["name"] for s in CATEGORY_B_SPECS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rolling-dir", default=DEFAULT_ROLLING_DIR)
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--label-margin-usd", type=float, default=DEFAULT_LABEL_MARGIN)
    p.add_argument("--equity", type=float, default=25_000.0)
    return p.parse_args()


def profit_factor_from_pnls(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    arr = np.array(pnls, dtype=np.float64)
    pos = arr[arr > 0].sum()
    neg = -arr[arr < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # === Load trade pairs and OOS prediction features ===
    print("Loading R4 trade CSVs...", flush=True)
    v0 = pd.read_csv(os.path.join(args.eval_dir, "trades_V0.csv"))
    v1 = pd.read_csv(os.path.join(args.eval_dir, "trades_V1.csv"))
    print(f"  V0: {len(v0)}; V1: {len(v1)}", flush=True)

    # Join V0 and V1 by (day, bar_index) to get paired PnLs
    v0_keyed = v0.set_index(["day", "bar_index"])
    v1_keyed = v1.set_index(["day", "bar_index"])
    common_idx = v0_keyed.index.intersection(v1_keyed.index)
    pairs = pd.DataFrame({
        "day": [i[0] for i in common_idx],
        "bar_index": [i[1] for i in common_idx],
        "window_idx": v0_keyed.loc[common_idx, "window_idx"].values,
        "v0_direction": v0_keyed.loc[common_idx, "direction"].values,
        "v0_pnl": v0_keyed.loc[common_idx, "pnl"].values,
        "v1_pnl": v1_keyed.loc[common_idx, "pnl"].values,
    })
    pairs["pnl_delta"] = pairs["v1_pnl"] - pairs["v0_pnl"]
    pairs["label"] = (pairs["pnl_delta"] > args.label_margin_usd).astype(int)
    print(f"  Paired day-trades: {len(pairs)} ({pairs['label'].sum()} V1-favorable, "
          f"{pairs['label'].mean()*100:.1f}%)", flush=True)

    # === Load per-window manifest ===
    manifest = load_json(os.path.join(args.rolling_dir, "manifest.json"))
    augmented_feature_names = manifest["augmented_feature_names"]
    print(f"  Augmented features: {len(augmented_feature_names)}", flush=True)

    # === For each window, load OOS predictions and extract features at chosen entry bar ===
    print(flush=True)
    print("Extracting features at chosen entry bar per window...", flush=True)
    all_feat_rows = []
    for w in manifest["windows"]:
        wi = w["window_idx"]
        pred_path = os.path.join(args.rolling_dir, f"window_{wi:02d}", "oos_predictions.pkl")
        oos_pred = load_pickle(pred_path)
        # Filter to the (day, bar_index) pairs that R4 chose
        window_pairs = pairs[pairs["window_idx"] == wi]
        for _, p in window_pairs.iterrows():
            row_mask = (oos_pred["day"] == p["day"]) & (oos_pred["bar_index"] == p["bar_index"])
            rows = oos_pred[row_mask]
            if len(rows) == 0:
                continue
            r = rows.iloc[0]
            feat_vec = {name: float(r[name]) for name in augmented_feature_names}
            feat_vec.update({
                "day": p["day"], "bar_index": int(p["bar_index"]),
                "window_idx": int(wi),
                "v0_pnl": float(p["v0_pnl"]), "v1_pnl": float(p["v1_pnl"]),
                "pnl_delta": float(p["pnl_delta"]), "label": int(p["label"]),
                "v0_direction": str(p["v0_direction"]),
            })
            all_feat_rows.append(feat_vec)
    feat_df = pd.DataFrame(all_feat_rows)
    print(f"  Extracted {len(feat_df)} feature rows (one per chosen entry bar)", flush=True)
    print(f"  Per-window positive rate (V1-favorable): {feat_df.groupby('window_idx')['label'].mean().to_dict()}",
          flush=True)

    # === Oracle upper bound (pick max(V0, V1) per trade) ===
    oracle_pnls = np.maximum(feat_df["v0_pnl"].values, feat_df["v1_pnl"].values).tolist()
    oracle_pf = profit_factor_from_pnls(oracle_pnls)
    oracle_mean = float(np.mean(oracle_pnls))
    print(flush=True)
    print(f"Oracle upper bound (perfect foresight): PF={oracle_pf:.3f} mean=${oracle_mean:.0f}",
          flush=True)

    # === Walk-forward classifier training ===
    print(flush=True)
    print("=" * 100)
    print("Walk-forward conditional classifier")
    print("=" * 100, flush=True)
    print(f"{'window':<7}{'train_n':>9}{'train_pos':>11}{'test_n':>8}{'test_pos':>9}"
          f"{'AUC':>8}{'AP':>8}{'prec':>8}{'recall':>8}  {'cond_PF':>10}{'V0_PF':>10}", flush=True)
    windows = sorted(feat_df["window_idx"].unique())
    per_window_results = []
    conditional_trades_global: list[dict] = []

    X_cols = list(augmented_feature_names)

    for wi in windows:
        train_df = feat_df[feat_df["window_idx"] < wi]
        test_df = feat_df[feat_df["window_idx"] == wi]
        n_pos_tr = int(train_df["label"].sum())
        if len(train_df) == 0 or n_pos_tr == 0:
            # No training data - default to V0 (blanket)
            cond_pnls = test_df["v0_pnl"].tolist()
            cond_pf = profit_factor_from_pnls(cond_pnls)
            v0_pf = profit_factor_from_pnls(test_df["v0_pnl"].tolist())
            print(f"{wi:<7}{'-':>9}{'-':>11}{len(test_df):>8}{int(test_df['label'].sum()):>9}"
                  f"{'-':>8}{'-':>8}{'-':>8}{'-':>8}  {cond_pf:>10.3f}{v0_pf:>10.3f}  (fallback to V0)",
                  flush=True)
            per_window_results.append({
                "window_idx": int(wi),
                "n_train": 0, "n_train_pos": 0,
                "n_test": len(test_df), "n_test_pos": int(test_df["label"].sum()),
                "auc": None, "ap": None, "precision": None, "recall": None,
                "conditional_pf": cond_pf, "blanket_v0_pf": v0_pf,
                "threshold": None,
            })
            for _, r in test_df.iterrows():
                conditional_trades_global.append({
                    "day": r["day"], "bar_index": int(r["bar_index"]),
                    "window_idx": int(wi), "rule": "V0_fallback",
                    "pnl": float(r["v0_pnl"]),
                })
            continue

        # Train classifier
        X_tr = train_df[X_cols].to_numpy(dtype=np.float32)
        y_tr = train_df["label"].to_numpy(dtype=np.int8)
        X_te = test_df[X_cols].to_numpy(dtype=np.float32)
        y_te = test_df["label"].to_numpy(dtype=np.int8)

        # Class weights to handle imbalance
        sample_weight = np.where(y_tr == 1,
                                 (len(y_tr) - n_pos_tr) / max(n_pos_tr, 1),
                                 1.0)
        model = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_depth=4,
            max_iter=200, min_samples_leaf=20,
            random_state=args.seed + wi, early_stopping=False,
        )
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        y_score = model.predict_proba(X_te)[:, 1]
        # Fixed threshold 0.5; could calibrate later
        threshold = 0.5
        y_pred = (y_score >= threshold).astype(int)
        try:
            auc = float(roc_auc_score(y_te, y_score)) if len(np.unique(y_te)) >= 2 else float("nan")
            ap = float(average_precision_score(y_te, y_score)) if y_te.sum() > 0 else float("nan")
        except Exception:
            auc = float("nan"); ap = float("nan")
        prec = float(precision_score(y_te, y_pred, zero_division=0))
        rec = float(recall_score(y_te, y_pred, zero_division=0))

        # Conditional rule: if pred==1, use V1 PnL; else use V0 PnL
        cond_pnls = np.where(y_pred == 1, test_df["v1_pnl"].values, test_df["v0_pnl"].values).tolist()
        cond_pf = profit_factor_from_pnls(cond_pnls)
        v0_pf = profit_factor_from_pnls(test_df["v0_pnl"].tolist())

        for i, (_, r) in enumerate(test_df.iterrows()):
            conditional_trades_global.append({
                "day": r["day"], "bar_index": int(r["bar_index"]),
                "window_idx": int(wi),
                "rule": "V1" if y_pred[i] == 1 else "V0",
                "pred_score": float(y_score[i]),
                "label": int(y_te[i]),
                "v0_pnl": float(r["v0_pnl"]), "v1_pnl": float(r["v1_pnl"]),
                "pnl": float(cond_pnls[i]),
            })

        print(f"{wi:<7}{len(train_df):>9}{n_pos_tr:>11}{len(test_df):>8}{int(y_te.sum()):>9}"
              f"{auc:>8.3f}{ap:>8.3f}{prec:>8.3f}{rec:>8.3f}  {cond_pf:>10.3f}{v0_pf:>10.3f}",
              flush=True)
        per_window_results.append({
            "window_idx": int(wi),
            "n_train": len(train_df), "n_train_pos": n_pos_tr,
            "n_test": len(test_df), "n_test_pos": int(y_te.sum()),
            "auc": auc, "ap": ap, "precision": prec, "recall": rec,
            "conditional_pf": cond_pf, "blanket_v0_pf": v0_pf,
            "threshold": threshold,
        })

    # === Aggregate ===
    cond_df = pd.DataFrame(conditional_trades_global)
    cond_pnls_all = cond_df["pnl"].astype(float).tolist()
    cond_agg_pf = profit_factor_from_pnls(cond_pnls_all)
    v0_pnls_all = feat_df["v0_pnl"].astype(float).tolist()
    v0_agg_pf = profit_factor_from_pnls(v0_pnls_all)
    v1_pnls_all = feat_df["v1_pnl"].astype(float).tolist()
    v1_agg_pf = profit_factor_from_pnls(v1_pnls_all)

    print(flush=True)
    print("=" * 100)
    print("Cross-window aggregates")
    print("=" * 100, flush=True)
    print(f"  Oracle (perfect foresight):  PF {oracle_pf:.3f}  mean ${oracle_mean:.0f}", flush=True)
    print(f"  Conditional rule:            PF {cond_agg_pf:.3f}  mean ${np.mean(cond_pnls_all):.0f}", flush=True)
    print(f"  Blanket V0 (R4 baseline):    PF {v0_agg_pf:.3f}  mean ${np.mean(v0_pnls_all):.0f}", flush=True)
    print(f"  Blanket V1 (R4 comparison):  PF {v1_agg_pf:.3f}  mean ${np.mean(v1_pnls_all):.0f}", flush=True)

    cond_vs_v0 = cond_agg_pf - v0_agg_pf
    cond_vs_oracle = cond_agg_pf - oracle_pf
    print(flush=True)
    print(f"  Conditional vs V0:    {cond_vs_v0:+.3f} PF", flush=True)
    print(f"  Conditional vs Oracle: {cond_vs_oracle:+.3f} PF (gap to perfect)", flush=True)
    if cond_agg_pf > v0_agg_pf + 0.05:
        verdict = f"WIN — conditional beats blanket V0 by {cond_vs_v0:+.3f} PF"
    elif cond_agg_pf >= v0_agg_pf - 0.05:
        verdict = f"TIE — conditional within 0.05 of V0 ({cond_vs_v0:+.3f})"
    else:
        verdict = f"LOSE — conditional underperforms V0 by {-cond_vs_v0:+.3f} PF"
    print(flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    # === Save ===
    payload = {
        "meta": {
            "label_margin_usd": float(args.label_margin_usd),
            "seed": int(args.seed),
            "n_windows": int(len(windows)),
            "n_features": len(augmented_feature_names),
        },
        "per_window": per_window_results,
        "aggregate": {
            "conditional_pf": cond_agg_pf,
            "blanket_v0_pf": v0_agg_pf,
            "blanket_v1_pf": v1_agg_pf,
            "oracle_pf": oracle_pf,
            "oracle_mean_pnl": oracle_mean,
            "conditional_mean_pnl": float(np.mean(cond_pnls_all)),
            "conditional_vs_v0": cond_vs_v0,
            "conditional_vs_oracle": cond_vs_oracle,
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "conditional_eval.json")
    save_json(out, payload)
    cond_df.to_csv(os.path.join(args.out_dir, "conditional_trades.csv"), index=False)
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
