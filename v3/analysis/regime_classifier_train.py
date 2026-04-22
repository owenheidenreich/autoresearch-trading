"""Phase 3B — Regime classifier training + walk-forward.

Train a HistGradientBoostingClassifier per fold on the 13 per-day
features → binary label (V0-favorable). Walk-forward: fold k trains
on days from folds 0..k-1.

Reports:
- Per-fold AUC, accuracy, precision, recall
- Per-fold confusion matrix
- Aggregated AUC
- Permutation importance on the fold-4 model

Acceptance gates (per the plan):
- PASS: aggregated AUC >= 0.60 AND avg test-fold accuracy >= 0.60
- PARTIAL: AUC in [0.55, 0.60)
- ABANDON: AUC < 0.55 → skip Phase 3C/3D, proceed to Phase 4
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
    accuracy_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LABELS_CSV = os.path.join("v3", "artifacts", "regime_labels", "regime_labels.csv")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "regime_classifier_train")
DEFAULT_SEED = 42

PER_DAY_FEATURE_NAMES = [
    "opening_gap_pct", "session_open_dist", "vwap_dist",
    "first15_range_pct", "first15_close_position",
    "vix_roc", "atm_iv", "iv_percentile", "realized_vol", "vix_regime",
    "sigma_pos", "omar_range_pct", "omar_mid_pos_units",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_metric(metric_fn, y_true, y_pred, **kwargs) -> float:
    try:
        return float(metric_fn(y_true, y_pred, **kwargs))
    except Exception:
        return float("nan")


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    in_sample = df[df["origin"] == "in_sample"].copy()
    oos = df[df["origin"] == "oos"].copy()
    print(f"Loaded {len(df)} day-rows ({len(in_sample)} in-sample, {len(oos)} OOS)", flush=True)

    X_in = in_sample[PER_DAY_FEATURE_NAMES].astype(float).values
    y_in = in_sample["label"].astype(int).values
    fold_in = in_sample["fold_idx"].astype(int).values
    print(f"  Feature matrix: {X_in.shape}; positive rate: {y_in.mean():.3f}", flush=True)

    X_oos = oos[PER_DAY_FEATURE_NAMES].astype(float).values
    y_oos = oos["label"].astype(int).values

    # === Per-fold walk-forward ===
    print(flush=True)
    print("=" * 100)
    print("Per-fold walk-forward")
    print("=" * 100)
    print(f"{'fold':<6}{'train_n':>10}{'train_pos':>11}{'test_n':>10}{'test_pos':>10}"
          f"{'AUC':>10}{'AP':>10}{'acc':>8}{'prec':>8}{'recall':>8}", flush=True)
    fold_results = {}
    fold_oof_scores = {}
    for fi in sorted(np.unique(fold_in)):
        train_mask = fold_in < fi
        test_mask = fold_in == fi
        if train_mask.sum() == 0:
            print(f"{fi:<6}{0:>10}{0:>11}{int(test_mask.sum()):>10}"
                  f"{int(y_in[test_mask].sum()):>10}     fold0 fallback (no training)", flush=True)
            fold_results[int(fi)] = {
                "train_n": 0, "train_pos": 0,
                "test_n": int(test_mask.sum()), "test_pos": int(y_in[test_mask].sum()),
                "auc": float("nan"), "ap": float("nan"),
                "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "confusion_matrix": [[0,0],[0,0]],
            }
            continue
        X_tr, y_tr = X_in[train_mask], y_in[train_mask]
        X_te, y_te = X_in[test_mask], y_in[test_mask]
        n_pos_tr = int(y_tr.sum())
        n_pos_te = int(y_te.sum())

        # Class imbalance — use sample weights
        sample_weight = np.where(y_tr == 1,
                                 (len(y_tr) - n_pos_tr) / max(n_pos_tr, 1),
                                 1.0)
        model = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_depth=4,
            max_iter=200, min_samples_leaf=20,
            random_state=args.seed + int(fi), early_stopping=False,
            class_weight=None,
        )
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        y_score = model.predict_proba(X_te)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        auc = safe_auc(y_te, y_score)
        ap = float(average_precision_score(y_te, y_score)) if y_te.sum() > 0 else float("nan")
        acc = safe_metric(accuracy_score, y_te, y_pred)
        prec = safe_metric(precision_score, y_te, y_pred, zero_division=0)
        rec = safe_metric(recall_score, y_te, y_pred, zero_division=0)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist()
        fold_results[int(fi)] = {
            "train_n": int(train_mask.sum()), "train_pos": n_pos_tr,
            "test_n": int(test_mask.sum()), "test_pos": n_pos_te,
            "auc": auc, "ap": ap,
            "accuracy": acc, "precision": prec, "recall": rec,
            "confusion_matrix": cm,
        }
        fold_oof_scores[int(fi)] = (y_te, y_score)
        print(
            f"{fi:<6}{int(train_mask.sum()):>10}{n_pos_tr:>11}"
            f"{int(test_mask.sum()):>10}{n_pos_te:>10}"
            f"{auc:>10.3f}{ap:>10.3f}{acc:>8.3f}{prec:>8.3f}{rec:>8.3f}",
            flush=True,
        )

    # === Aggregate AUC across folds (concatenated OOF predictions) ===
    aggregated = []
    for fi, (yt, ys) in fold_oof_scores.items():
        for i in range(len(yt)):
            aggregated.append((int(yt[i]), float(ys[i])))
    if aggregated:
        y_all = np.array([a[0] for a in aggregated])
        s_all = np.array([a[1] for a in aggregated])
        agg_auc = safe_auc(y_all, s_all)
        agg_ap = float(average_precision_score(y_all, s_all)) if y_all.sum() > 0 else float("nan")
    else:
        agg_auc = float("nan")
        agg_ap = float("nan")

    # === OOS evaluation (use fold-4 model on the OOS rows) ===
    print(flush=True)
    print("=" * 100)
    print("OOS evaluation (fold-4 model = trained on all folds 0-3)")
    print("=" * 100)
    train_mask = fold_in < 4
    if train_mask.sum() > 0:
        X_tr, y_tr = X_in[train_mask], y_in[train_mask]
        n_pos_tr = int(y_tr.sum())
        sample_weight = np.where(y_tr == 1,
                                 (len(y_tr) - n_pos_tr) / max(n_pos_tr, 1),
                                 1.0)
        model_oos = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_depth=4,
            max_iter=200, min_samples_leaf=20,
            random_state=args.seed + 1000, early_stopping=False,
        )
        model_oos.fit(X_tr, y_tr, sample_weight=sample_weight)
        y_score_oos = model_oos.predict_proba(X_oos)[:, 1]
        y_pred_oos = (y_score_oos >= 0.5).astype(int)
        oos_auc = safe_auc(y_oos, y_score_oos)
        oos_ap = float(average_precision_score(y_oos, y_score_oos)) if y_oos.sum() > 0 else float("nan")
        oos_acc = safe_metric(accuracy_score, y_oos, y_pred_oos)
        oos_prec = safe_metric(precision_score, y_oos, y_pred_oos, zero_division=0)
        oos_rec = safe_metric(recall_score, y_oos, y_pred_oos, zero_division=0)
        oos_cm = confusion_matrix(y_oos, y_pred_oos, labels=[0, 1]).tolist()
        print(f"  OOS AUC={oos_auc:.3f}  AP={oos_ap:.3f}  acc={oos_acc:.3f} "
              f"prec={oos_prec:.3f} recall={oos_rec:.3f}", flush=True)
        print(f"  OOS confusion matrix [[TN, FP], [FN, TP]]: {oos_cm}", flush=True)
        print(f"  OOS class distribution: {y_oos.sum()}/{len(y_oos)} V0-favorable", flush=True)
        # Predicted scores per OOS day
        print(f"  Per-day OOS scores: ", flush=True)
        for i, (day_label, day_pred) in enumerate(zip(y_oos, y_score_oos)):
            tag = "✓" if (day_pred >= 0.5) == bool(day_label) else "✗"
            day_str = oos.iloc[i]["day"]
            print(f"    {day_str}: label={int(day_label)} pred_score={day_pred:.3f} {tag}", flush=True)
    else:
        oos_auc = oos_ap = oos_acc = oos_prec = oos_rec = float("nan")
        oos_cm = [[0, 0], [0, 0]]

    # === Permutation importance on fold-4 model ===
    print(flush=True)
    print("=" * 100)
    print("Permutation importance on fold-4 model (OOS evaluation)")
    print("=" * 100, flush=True)
    rng = np.random.default_rng(args.seed)
    base_auc = oos_auc
    importance = []
    if train_mask.sum() > 0 and not np.isnan(base_auc):
        for fi_idx, fname in enumerate(PER_DAY_FEATURE_NAMES):
            drops = []
            for trial in range(5):
                X_perm = X_oos.copy()
                rng.shuffle(X_perm[:, fi_idx])
                y_score_perm = model_oos.predict_proba(X_perm)[:, 1]
                shuf_auc = safe_auc(y_oos, y_score_perm)
                drops.append(base_auc - shuf_auc)
            mean_drop = float(np.mean(drops))
            std_drop = float(np.std(drops))
            importance.append((fname, mean_drop, std_drop))
        importance.sort(key=lambda t: -abs(t[1]))
        print(f"{'feature':<28}{'AUC drop':>12}{'std':>10}", flush=True)
        for fname, drop, std in importance:
            print(f"{fname:<28}{drop:>12.4f}{std:>10.4f}", flush=True)

    # === Verdict ===
    print(flush=True)
    print("=" * 100)
    print("Phase 3B verdict")
    print("=" * 100, flush=True)
    print(f"  Aggregated walk-forward AUC: {agg_auc:.3f}", flush=True)
    print(f"  OOS AUC (fold-4 model):      {oos_auc:.3f}", flush=True)
    valid_fold_aucs = [r["auc"] for r in fold_results.values() if not np.isnan(r["auc"])]
    avg_fold_auc = float(np.mean(valid_fold_aucs)) if valid_fold_aucs else float("nan")
    valid_fold_acc = [r["accuracy"] for r in fold_results.values() if not np.isnan(r["accuracy"])]
    avg_fold_acc = float(np.mean(valid_fold_acc)) if valid_fold_acc else float("nan")
    print(f"  Mean per-fold AUC:            {avg_fold_auc:.3f}", flush=True)
    print(f"  Mean per-fold accuracy:       {avg_fold_acc:.3f}", flush=True)

    if agg_auc >= 0.60 and avg_fold_acc >= 0.60:
        verdict = "PASS — proceed to Phase 3C"
    elif agg_auc >= 0.55:
        verdict = "PARTIAL — borderline; proceed to 3C with caution"
    elif np.isnan(agg_auc):
        verdict = "INCONCLUSIVE — too few positive examples per fold to evaluate AUC reliably"
    else:
        verdict = f"ABANDON — aggregated AUC {agg_auc:.3f} < 0.55; skip 3C/3D, proceed to Phase 4"
    print(flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    # === Save ===
    payload = {
        "meta": {
            "labels_csv": args.labels_csv,
            "n_in_sample": int(len(in_sample)),
            "n_oos": int(len(oos)),
            "seed": int(args.seed),
            "features": PER_DAY_FEATURE_NAMES,
        },
        "per_fold": fold_results,
        "aggregated_auc": agg_auc,
        "aggregated_ap": agg_ap,
        "mean_fold_auc": avg_fold_auc,
        "mean_fold_acc": avg_fold_acc,
        "oos": {
            "auc": oos_auc, "ap": oos_ap, "accuracy": oos_acc,
            "precision": oos_prec, "recall": oos_rec, "confusion_matrix": oos_cm,
        },
        "permutation_importance": [
            {"feature": fname, "auc_drop_mean": drop, "auc_drop_std": std}
            for fname, drop, std in importance
        ],
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "regime_classifier_train.json")
    with open(out, "w") as f:
        json.dump(
            payload, f, indent=2, sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
