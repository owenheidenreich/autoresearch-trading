"""Fork C Phase 1 evaluation — test fold scored once with val-selected thresholds.

Loads fitted models from ``artifacts/fitted_models.pkl``, applies each
model's val-fold top-30%-coverage threshold to the test fold, and reports:
- PR-AUC (bootstrap 95% CI)
- Precision @ top-30% (bootstrap 95% CI)
- Recall @ top-30% (bootstrap 95% CI)
- Brier score (bootstrap 95% CI), versus majority-baseline Brier
- Lift @ top-K% for K in {10%, 20%, 30%}
- Reliability deciles

Then applies the three-way Phase 2 gate:
- **advance** — PR-AUC LCB >= base_rate+0.05 AND Precision@30 LCB >=
  base_rate+0.05 AND Brier <= majority-baseline Brier.
- **signal_of_life** — point estimates clear the bars but at least one CI
  lower bound does not.
- **null** — point estimate doesn't clear, or Brier above majority.

Usage::
    python3 -m v2.fork_c.eval_tier1 \\
        --dataset v2/fork_c/tier1_dataset.csv \\
        --artifacts-dir v2/artifacts/fork_c_tier1
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from v2.fork_c.train_tier1 import (
    TOP_COVERAGE_FRAC,
    load_dataset,
    rows_to_xy,
    split_rows,
    threshold_at_top_coverage,
)


N_BOOTSTRAP = 1000
CI_LO, CI_HI = 2.5, 97.5
ADVANCE_MARGIN = 0.05  # plan §"Phase 2 gate"


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOTSTRAP,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) — point estimate plus 95% CI from 1000
    resamples with replacement. Resamples that yield NaN or undefined
    metric (e.g., zero positives) are dropped."""
    point = metric_fn(y_true, scores)
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    results = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = metric_fn(y_true[idx], scores[idx])
        if np.isfinite(v):
            results.append(v)
    if len(results) < 10:
        return float(point), float("nan"), float("nan")
    lo = float(np.percentile(results, CI_LO))
    hi = float(np.percentile(results, CI_HI))
    return float(point), lo, hi


def pr_auc_metric(y_true: np.ndarray, scores: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def brier_metric(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(brier_score_loss(y_true, scores))


def precision_at_coverage_metric_factory(threshold: float):
    def _f(y_true: np.ndarray, scores: np.ndarray) -> float:
        preds = (scores >= threshold).astype(np.int32)
        n_pred_pos = int(preds.sum())
        if n_pred_pos == 0:
            return float("nan")
        tp = int(((preds == 1) & (y_true == 1)).sum())
        return float(tp / n_pred_pos)
    return _f


def recall_at_coverage_metric_factory(threshold: float):
    def _f(y_true: np.ndarray, scores: np.ndarray) -> float:
        n_pos = int(y_true.sum())
        if n_pos == 0:
            return float("nan")
        preds = (scores >= threshold).astype(np.int32)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        return float(tp / n_pos)
    return _f


def lift_at_top_k(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    n = len(y_true)
    base = float(y_true.mean()) if n else 0.0
    if base == 0:
        return float("nan")
    k = int(np.ceil(n * frac))
    if k == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    top_idx = order[:k]
    precision_at_k = float(y_true[top_idx].mean())
    return precision_at_k / base


def reliability_deciles(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> list[dict]:
    order = np.argsort(scores)
    y_sorted = y_true[order]
    p_sorted = scores[order]
    n = len(y_true)
    out = []
    for b in range(n_bins):
        lo = n * b // n_bins
        hi = n * (b + 1) // n_bins
        if hi <= lo:
            continue
        out.append(
            {
                "bin": b,
                "n": int(hi - lo),
                "mean_pred": float(p_sorted[lo:hi].mean()),
                "mean_actual": float(y_sorted[lo:hi].mean()),
            }
        )
    return out


def majority_brier(y_test: np.ndarray, train_positive_rate: float) -> float:
    scores = np.full_like(y_test, fill_value=train_positive_rate, dtype=np.float64)
    return float(brier_score_loss(y_test, scores))


def score_model(fitted: dict, model_name: str, X_test: np.ndarray) -> np.ndarray:
    """Return test-fold probability scores for a named model."""
    if model_name == "majority":
        p = fitted.get("majority_p")
        if p is None:
            # Recover from state dict
            p = fitted["majority"]["majority_p"]
        return np.full(X_test.shape[0], fill_value=float(p), dtype=np.float64)
    model = fitted[model_name]
    return model.predict_proba(X_test)[:, 1]


def evaluate_one_model(
    name: str,
    val_scores: np.ndarray,
    y_val: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    majority_brier_test: float,
    base_rate_test: float,
) -> dict:
    # Val-selected threshold at TOP_COVERAGE_FRAC
    if name == "majority":
        threshold = float(val_scores[0]) if len(val_scores) else 0.0
    else:
        threshold = threshold_at_top_coverage(val_scores, TOP_COVERAGE_FRAC)

    pr_auc_pt, pr_auc_lo, pr_auc_hi = bootstrap_ci(y_test, test_scores, pr_auc_metric)
    brier_pt, brier_lo, brier_hi = bootstrap_ci(y_test, test_scores, brier_metric)
    p_fn = precision_at_coverage_metric_factory(threshold)
    r_fn = recall_at_coverage_metric_factory(threshold)
    p_pt, p_lo, p_hi = bootstrap_ci(y_test, test_scores, p_fn)
    r_pt, r_lo, r_hi = bootstrap_ci(y_test, test_scores, r_fn)

    lifts = {
        f"top_{int(k*100)}": lift_at_top_k(y_test, test_scores, k)
        for k in (0.10, 0.20, 0.30)
    }
    reliability = reliability_deciles(y_test, test_scores)

    # Three-way gate bars for this model.
    advance_bar = base_rate_test + ADVANCE_MARGIN
    pr_clear_point = pr_auc_pt >= advance_bar
    pr_clear_ci = (not np.isnan(pr_auc_lo)) and pr_auc_lo >= advance_bar
    p_clear_point = p_pt >= advance_bar
    p_clear_ci = (not np.isnan(p_lo)) and p_lo >= advance_bar
    brier_clear = brier_pt <= majority_brier_test

    if pr_clear_ci and p_clear_ci and brier_clear:
        gate = "advance"
    elif pr_clear_point and p_clear_point and brier_clear:
        gate = "signal_of_life"
    else:
        gate = "null"

    return {
        "model": name,
        "threshold_val_top30": float(threshold),
        "n_predicted_positive_test_at_threshold": int((test_scores >= threshold).sum()),
        "pr_auc": {"point": pr_auc_pt, "ci_lo": pr_auc_lo, "ci_hi": pr_auc_hi},
        "brier": {"point": brier_pt, "ci_lo": brier_lo, "ci_hi": brier_hi},
        "precision_at_top30": {"point": p_pt, "ci_lo": p_lo, "ci_hi": p_hi},
        "recall_at_top30": {"point": r_pt, "ci_lo": r_lo, "ci_hi": r_hi},
        "lifts": lifts,
        "reliability_deciles": reliability,
        "gate": {
            "advance_bar": advance_bar,
            "pr_auc_clear_point": pr_clear_point,
            "pr_auc_clear_ci": pr_clear_ci,
            "precision_clear_point": p_clear_point,
            "precision_clear_ci": p_clear_ci,
            "brier_clear": brier_clear,
            "majority_brier_test": majority_brier_test,
            "verdict": gate,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v2/fork_c/tier1_dataset.csv")
    ap.add_argument("--artifacts-dir", default="v2/artifacts/fork_c_tier1")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    artifacts_dir = Path(args.artifacts_dir)

    with open(artifacts_dir / "fitted_models.pkl", "rb") as f:
        artifacts = pickle.load(f)
    fitted = artifacts["fitted"]
    scaler_all = artifacts["scaler_all"]
    scaler_cur = artifacts["scaler_cur"]
    all_feature_cols = artifacts["all_feature_cols"]
    curated_cols = artifacts["curated_cols"]
    hgbt_fired = artifacts["hgbt_fired"]

    rows, _, _ = load_dataset(dataset_path)
    train_rows, val_rows, test_rows = split_rows(rows)

    X_train_all, y_train = rows_to_xy(train_rows, all_feature_cols)
    X_val_all, y_val = rows_to_xy(val_rows, all_feature_cols)
    X_test_all, y_test = rows_to_xy(test_rows, all_feature_cols)
    X_val_all_s = scaler_all.transform(X_val_all)
    X_test_all_s = scaler_all.transform(X_test_all)

    X_val_cur, _ = rows_to_xy(val_rows, curated_cols)
    X_test_cur, _ = rows_to_xy(test_rows, curated_cols)
    X_val_cur_s = scaler_cur.transform(X_val_cur)
    X_test_cur_s = scaler_cur.transform(X_test_cur)

    base_rate_train = float(y_train.mean())
    base_rate_val = float(y_val.mean())
    base_rate_test = float(y_test.mean())

    majority_b = majority_brier(y_test, base_rate_train)
    print(
        f"Test n={len(test_rows)}  pos={int(y_test.sum())}  "
        f"base_rate={base_rate_test:.3f}  majority-Brier={majority_b:.3f}  "
        f"advance_bar={base_rate_test + ADVANCE_MARGIN:.3f}"
    )
    print()

    evaluations: list[dict] = []

    # Majority.
    maj_p = fitted["majority"]["majority_p"]
    val_scores = np.full_like(y_val, fill_value=maj_p, dtype=np.float64)
    test_scores = np.full_like(y_test, fill_value=maj_p, dtype=np.float64)
    ev = evaluate_one_model(
        "majority", val_scores, y_val, test_scores, y_test, majority_b, base_rate_test
    )
    evaluations.append(ev)

    # LR all.
    lr_all = fitted["lr_all"]
    val_scores = lr_all.predict_proba(X_val_all_s)[:, 1]
    test_scores = lr_all.predict_proba(X_test_all_s)[:, 1]
    ev = evaluate_one_model(
        "lr_all", val_scores, y_val, test_scores, y_test, majority_b, base_rate_test
    )
    evaluations.append(ev)

    # LR curated.
    lr_cur = fitted["lr_curated"]
    val_scores = lr_cur.predict_proba(X_val_cur_s)[:, 1]
    test_scores = lr_cur.predict_proba(X_test_cur_s)[:, 1]
    ev = evaluate_one_model(
        "lr_curated", val_scores, y_val, test_scores, y_test, majority_b, base_rate_test
    )
    evaluations.append(ev)

    # HGBT if fired.
    if hgbt_fired and "hgbt_all" in fitted:
        hgbt = fitted["hgbt_all"]
        val_scores = hgbt.predict_proba(X_val_all)[:, 1]
        test_scores = hgbt.predict_proba(X_test_all)[:, 1]
        ev = evaluate_one_model(
            "hgbt_all", val_scores, y_val, test_scores, y_test, majority_b, base_rate_test
        )
        evaluations.append(ev)

    # Print summary
    for e in evaluations:
        print(f"[{e['model']}]  verdict={e['gate']['verdict']}")
        print(
            f"  PR-AUC  = {e['pr_auc']['point']:.3f}  "
            f"[CI {e['pr_auc']['ci_lo']:.3f}, {e['pr_auc']['ci_hi']:.3f}]"
        )
        print(
            f"  Brier   = {e['brier']['point']:.3f}  "
            f"[CI {e['brier']['ci_lo']:.3f}, {e['brier']['ci_hi']:.3f}]  "
            f"majority={majority_b:.3f}"
        )
        print(
            f"  P@30   = {e['precision_at_top30']['point']:.3f}  "
            f"[CI {e['precision_at_top30']['ci_lo']:.3f}, "
            f"{e['precision_at_top30']['ci_hi']:.3f}]  "
            f"(n_pred_pos={e['n_predicted_positive_test_at_threshold']}, "
            f"thr={e['threshold_val_top30']:.3f})"
        )
        print(
            f"  R@30   = {e['recall_at_top30']['point']:.3f}  "
            f"[CI {e['recall_at_top30']['ci_lo']:.3f}, "
            f"{e['recall_at_top30']['ci_hi']:.3f}]"
        )
        print(f"  lifts  = {e['lifts']}")
        print()

    # Overall Phase-2 verdict.
    # If any non-majority model reports 'advance' → advance.
    # Otherwise if any non-majority reports 'signal_of_life' → signal_of_life.
    # Otherwise → null.
    non_maj = [e for e in evaluations if e["model"] != "majority"]
    verdicts = {e["model"]: e["gate"]["verdict"] for e in non_maj}
    if any(v == "advance" for v in verdicts.values()):
        overall = "advance"
    elif any(v == "signal_of_life" for v in verdicts.values()):
        overall = "signal_of_life"
    else:
        overall = "null"
    print(f"OVERALL PHASE-2 VERDICT: {overall}")
    print(f"  per-model: {verdicts}")

    report = {
        "dataset_path": str(dataset_path),
        "test": {
            "n": len(test_rows),
            "n_positive": int(y_test.sum()),
            "base_rate": base_rate_test,
        },
        "val": {
            "n": len(val_rows),
            "n_positive": int(y_val.sum()),
            "base_rate": base_rate_val,
        },
        "train": {
            "n": len(train_rows),
            "n_positive": int(y_train.sum()),
            "base_rate": base_rate_train,
        },
        "advance_margin": ADVANCE_MARGIN,
        "advance_bar": base_rate_test + ADVANCE_MARGIN,
        "majority_brier_test": majority_b,
        "n_bootstrap": N_BOOTSTRAP,
        "evaluations": evaluations,
        "overall_verdict": overall,
        "per_model_verdicts": verdicts,
    }
    out_path = artifacts_dir / "test_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print(f"\nTest report → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
