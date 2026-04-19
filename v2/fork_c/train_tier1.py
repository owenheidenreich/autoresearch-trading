"""Fork C Phase 1 training — cascade of majority / LR-all / LR-curated / HGBT.

Pre-registered protocol:

1. Chronological 60/20/20 split (boundaries frozen in
   v2/artifacts/fork_c_tier1/preflight_acknowledged.json).
2. StandardScaler fit on train only.
3. Fit in order:
    a) Majority-class baseline.
    b) Logistic regression on all 84 features (79 base + 5 aggregate),
       L2 regularized, class-balanced weighting.
    c) Logistic regression on ~15 Pickles-vocabulary features.
    d) HistGradientBoostingClassifier — ONLY if LR-all val PR-AUC
       exceeds base_rate + 0.05. Library defaults; no tuning.
4. Threshold selection at val-fold fixed 30% coverage (NOT F1).
5. Emit trained models + val metrics + thresholds. Test fold is not
   touched by this script — eval_tier1.py applies val-selected
   thresholds to test exactly once.

Running this script is blocked unless
``v2/artifacts/fork_c_tier1/preflight_acknowledged.json`` exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler


CURATED_FEATURES = [
    # Overnight / early-session context (Pickles-core)
    "agg_ovn_proxy_direction",
    "agg_am_session_end_vs_open_bps",
    "agg_am_session_range_bps",
    "agg_prior_session_end_ret",
    "f_opening_gap_pct",
    # First 15m stats (Pickles' "first-fifteen" vocabulary)
    "f_first15_range_pct",
    "f_first15_close_position",
    # VWAP regime (core to 1000-MAGIC-TIME)
    "f_vwap_dist",
    "f_vwap_reclaim_state",
    "agg_vwap_sigma_session_frac",
    # Vol regime
    "f_vix_regime",
    "f_vix_roc",
    "f_iv_percentile",
    # Initial-balance + prior-high
    "f_ib_extension_pct",
    "f_prev_high_dist",
]

TOP_COVERAGE_FRAC = 0.30
HGBT_TRIGGER_MARGIN = 0.05


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(dataset_path: Path) -> tuple[list[dict], list[str], list[str]]:
    """Return (rows_sorted_by_date, feature_cols_all, curated_cols).

    feature_cols_all is every ``f_*`` + ``agg_*`` column in CSV order.
    """
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        all_cols = reader.fieldnames
    if all_cols is None:
        raise ValueError("empty dataset")
    feature_cols = [c for c in all_cols if c.startswith("f_") or c.startswith("agg_")]
    rows.sort(key=lambda r: r["date"])
    missing = [c for c in CURATED_FEATURES if c not in feature_cols]
    if missing:
        raise ValueError(f"curated features missing from dataset: {missing}")
    return rows, feature_cols, CURATED_FEATURES


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    train_end = n * 60 // 100
    val_end = n * 80 // 100
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def rows_to_xy(rows: list[dict], cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    return X, y


def threshold_at_top_coverage(scores: np.ndarray, coverage: float) -> float:
    """Return the score threshold that marks the top-coverage fraction as positive.

    Uses ceil so that coverage is never under-counted on small folds.
    """
    k = int(np.ceil(len(scores) * coverage))
    if k <= 0 or k > len(scores):
        raise ValueError(f"invalid k={k} for n={len(scores)} coverage={coverage}")
    sorted_desc = np.sort(scores)[::-1]
    # Threshold = k-th highest score. All scores >= this are "positive".
    return float(sorted_desc[k - 1])


def eval_val(y_true: np.ndarray, scores: np.ndarray) -> dict:
    n = len(y_true)
    n_pos = int(y_true.sum())
    pr_auc = float(average_precision_score(y_true, scores)) if n_pos > 0 else float("nan")
    brier = float(brier_score_loss(y_true, scores))
    thresh = threshold_at_top_coverage(scores, TOP_COVERAGE_FRAC)
    preds = (scores >= thresh).astype(np.int32)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "n": n,
        "n_positive": n_pos,
        "base_rate": float(n_pos / n) if n else 0.0,
        "pr_auc": pr_auc,
        "brier": brier,
        "threshold_top30": thresh,
        "precision_at_top30": float(precision),
        "recall_at_top30": float(recall),
        "n_predicted_positive_top30": int(preds.sum()),
    }


def eval_majority(y_train: np.ndarray, y_val: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    """Predict the train-majority class with constant probability equal to
    train base rate. Score = train positive rate (same for everyone).

    Majority baseline has no threshold — top-30% coverage is undefined for a
    constant score. We synthesize a "threshold = positive_rate" and report
    precision_at_top30 = base rate, which is the correct floor.
    """
    p = float(y_train.mean())
    scores_val = np.full_like(y_val, fill_value=p, dtype=np.float64)
    n_pos = int(y_val.sum())
    pr_auc = float(average_precision_score(y_val, scores_val)) if n_pos > 0 else float("nan")
    brier = float(brier_score_loss(y_val, scores_val))
    info = {
        "model": "majority",
        "train_positive_rate": p,
        "val": {
            "n": len(y_val),
            "n_positive": n_pos,
            "base_rate": float(n_pos / len(y_val)) if len(y_val) else 0.0,
            "pr_auc": pr_auc,
            "brier": brier,
            "threshold_top30": p,
            "precision_at_top30": float(n_pos / len(y_val)) if len(y_val) else 0.0,
            "recall_at_top30": 1.0,  # everyone predicted positive at constant score
            "n_predicted_positive_top30": len(y_val),
            "note": "constant-score baseline — threshold semantics N/A",
        },
    }
    return info, scores_val, {"majority_p": p}


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    name: str,
    feature_names: list[str],
) -> tuple[dict, np.ndarray, Any]:
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    model.fit(X_train, y_train)
    scores_val = model.predict_proba(X_val)[:, 1]
    val = eval_val(y_val, scores_val)
    info = {
        "model": name,
        "n_features": X_train.shape[1],
        "feature_names": list(feature_names),
        "val": val,
    }
    return info, scores_val, model


def train_hgbt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, np.ndarray, Any]:
    model = HistGradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    scores_val = model.predict_proba(X_val)[:, 1]
    val = eval_val(y_val, scores_val)
    info = {
        "model": "hgbt_all",
        "n_features": X_train.shape[1],
        "feature_names": list(feature_names),
        "val": val,
    }
    return info, scores_val, model


def assert_preflight_acknowledged(ack_path: Path) -> dict:
    if not ack_path.exists():
        print(
            f"ERROR: preflight acknowledgment missing at {ack_path}.\n"
            "Run v2.fork_c.preflight_tier1 and create this file before training.",
            file=sys.stderr,
        )
        sys.exit(2)
    with open(ack_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v2/fork_c/tier1_dataset.csv")
    ap.add_argument(
        "--acknowledged",
        default="v2/artifacts/fork_c_tier1/preflight_acknowledged.json",
    )
    ap.add_argument("--artifacts-dir", default="v2/artifacts/fork_c_tier1")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    ack_path = Path(args.acknowledged)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ack = assert_preflight_acknowledged(ack_path)
    print(f"Preflight acknowledged: decision={ack.get('decision')}")

    rows, all_feature_cols, curated_cols = load_dataset(dataset_path)
    train_rows, val_rows, test_rows = split_rows(rows)
    print(
        f"Split: train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}"
    )

    # Verify split boundaries match acknowledgment
    expected = ack["split_boundaries_frozen"]
    actual = {
        "train_start": train_rows[0]["date"],
        "train_end": train_rows[-1]["date"],
        "val_start": val_rows[0]["date"],
        "val_end": val_rows[-1]["date"],
        "test_start": test_rows[0]["date"],
        "test_end": test_rows[-1]["date"],
    }
    for k, v in expected.items():
        if k in actual and actual[k] != v:
            raise ValueError(
                f"split boundary drift: acknowledgment expects {k}={v}, got {actual[k]}"
            )

    X_train_all, y_train = rows_to_xy(train_rows, all_feature_cols)
    X_val_all, y_val = rows_to_xy(val_rows, all_feature_cols)

    X_train_cur, _ = rows_to_xy(train_rows, curated_cols)
    X_val_cur, _ = rows_to_xy(val_rows, curated_cols)

    # Scaler — fit on train only. Separate scaler per feature set so that
    # curated-LR's scaling is independent from all-feature scaling.
    scaler_all = StandardScaler().fit(X_train_all)
    scaler_cur = StandardScaler().fit(X_train_cur)
    X_train_all_s = scaler_all.transform(X_train_all)
    X_val_all_s = scaler_all.transform(X_val_all)
    X_train_cur_s = scaler_cur.transform(X_train_cur)
    X_val_cur_s = scaler_cur.transform(X_val_cur)

    # ---------- Train cascade ----------
    results: list[dict] = []
    fitted: dict[str, Any] = {}

    maj_info, _, maj_state = eval_majority(y_train, y_val)
    results.append(maj_info)
    fitted["majority"] = maj_state
    print(
        f"[majority]    val PR-AUC={maj_info['val']['pr_auc']:.3f}  "
        f"Brier={maj_info['val']['brier']:.3f}"
    )

    base_rate = float(y_train.mean())
    hgbt_trigger = base_rate + HGBT_TRIGGER_MARGIN
    print(f"HGBT trigger: val PR-AUC > {hgbt_trigger:.3f} (train base rate + {HGBT_TRIGGER_MARGIN})")

    lr_all_info, _, lr_all_model = train_logreg(
        X_train_all_s, y_train, X_val_all_s, y_val,
        name="lr_all",
        feature_names=all_feature_cols,
    )
    results.append(lr_all_info)
    fitted["lr_all"] = lr_all_model
    print(
        f"[lr_all]      val PR-AUC={lr_all_info['val']['pr_auc']:.3f}  "
        f"Brier={lr_all_info['val']['brier']:.3f}  "
        f"P@30={lr_all_info['val']['precision_at_top30']:.3f}  "
        f"R@30={lr_all_info['val']['recall_at_top30']:.3f}"
    )

    lr_cur_info, _, lr_cur_model = train_logreg(
        X_train_cur_s, y_train, X_val_cur_s, y_val,
        name="lr_curated",
        feature_names=curated_cols,
    )
    results.append(lr_cur_info)
    fitted["lr_curated"] = lr_cur_model
    print(
        f"[lr_curated]  val PR-AUC={lr_cur_info['val']['pr_auc']:.3f}  "
        f"Brier={lr_cur_info['val']['brier']:.3f}  "
        f"P@30={lr_cur_info['val']['precision_at_top30']:.3f}  "
        f"R@30={lr_cur_info['val']['recall_at_top30']:.3f}"
    )

    hgbt_fired = False
    if (
        not np.isnan(lr_all_info["val"]["pr_auc"])
        and lr_all_info["val"]["pr_auc"] > hgbt_trigger
    ):
        hgbt_fired = True
        hgbt_info, _, hgbt_model = train_hgbt(
            X_train_all, y_train, X_val_all, y_val,
            feature_names=all_feature_cols,
        )
        results.append(hgbt_info)
        fitted["hgbt_all"] = hgbt_model
        print(
            f"[hgbt_all]    val PR-AUC={hgbt_info['val']['pr_auc']:.3f}  "
            f"Brier={hgbt_info['val']['brier']:.3f}  "
            f"P@30={hgbt_info['val']['precision_at_top30']:.3f}  "
            f"R@30={hgbt_info['val']['recall_at_top30']:.3f}"
        )
    else:
        print(
            f"[hgbt_all]    SKIPPED (lr_all val PR-AUC "
            f"{lr_all_info['val']['pr_auc']:.3f} <= trigger {hgbt_trigger:.3f})"
        )

    # Sanity: scaler-fit discipline. Train-fit mean should differ from
    # full-dataset mean by a non-trivial amount for at least some features.
    X_full_all = np.concatenate([X_train_all, X_val_all], axis=0)
    full_mean = X_full_all.mean(axis=0)
    train_mean = X_train_all.mean(axis=0)
    max_abs_shift = float(np.max(np.abs(full_mean - train_mean)))
    print(
        f"Scaler discipline: max |train_mean - train+val_mean| across features "
        f"= {max_abs_shift:.6f}"
    )

    # ---------- Persist artifacts ----------
    # Save fitted objects for eval_tier1.py to load.
    with open(artifacts_dir / "fitted_models.pkl", "wb") as f:
        pickle.dump(
            {
                "fitted": fitted,
                "scaler_all": scaler_all,
                "scaler_cur": scaler_cur,
                "all_feature_cols": all_feature_cols,
                "curated_cols": curated_cols,
                "hgbt_fired": hgbt_fired,
            },
            f,
        )

    train_report = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_of_file(dataset_path),
        "preflight_acknowledged_path": str(ack_path),
        "split_boundaries": actual,
        "train": {"n": len(train_rows), "n_positive": int(y_train.sum())},
        "val": {"n": len(val_rows), "n_positive": int(y_val.sum())},
        "base_rate_train": base_rate,
        "hgbt_trigger": hgbt_trigger,
        "hgbt_fired": hgbt_fired,
        "scaler_max_abs_shift_train_vs_full": max_abs_shift,
        "models": results,
        "curated_features": curated_cols,
        "top_coverage_frac": TOP_COVERAGE_FRAC,
    }
    with open(artifacts_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(train_report, f, indent=2, sort_keys=True, default=str)

    print()
    print(f"Fitted models → {artifacts_dir / 'fitted_models.pkl'}")
    print(f"Train report → {artifacts_dir / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
