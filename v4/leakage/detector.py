"""Leak-detection harness.

Two designed-to-fail tests run on every promotion (and ideally on every CI):

1. **Shuffled-label test**: shuffle labels along the time axis, refit a
   probe, observe AUC. AUC should drop to ~0.5 (chance). If it doesn't,
   there's a hidden time-correlation pathology.

2. **Planted-leak test**: deliberately plant a column derived from the
   realized future (i.e., from the labels themselves). Refit. AUC should
   jump dramatically — proving the detector can catch a leak when one
   exists. If the planted leak does NOT produce a jump, the detector is
   broken.

These tests do not validate features in production. They validate that
the *leak detector* works. Real leak detection runs the shuffled-label
test on every model promotion (Section 4.6 of the protocol).

Anti-pattern this prevents: the v3 2026-04-25 incident where features
contained realized future PnL labels (`time_stop_margin_raw`,
`side_margin_raw`) and the gate reported +24% PF lift before the leak
was discovered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class LeakProbeResult:
    name: str
    auc: float
    n_samples: int
    notes: str = ""


def probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = 3,
    seed: int = 0,
) -> float:
    """Train a logistic-regression probe; return mean held-out AUC."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D; got shape {features.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"features and labels disagree on length: "
            f"{features.shape[0]} vs {labels.shape[0]}"
        )
    n = features.shape[0]
    if n < 2 * n_folds:
        # Too few samples for cross-validation; train+predict on full data.
        # AUC will be optimistic but tests still succeed at distinguishing
        # 'has signal' from 'doesn't'.
        clf = LogisticRegression(max_iter=200)
        clf.fit(features, labels)
        return roc_auc_score(labels, clf.predict_proba(features)[:, 1])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in kf.split(features):
        if len(set(labels[train_idx])) < 2 or len(set(labels[test_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=200)
        clf.fit(features[train_idx], labels[train_idx])
        proba = clf.predict_proba(features[test_idx])[:, 1]
        aucs.append(roc_auc_score(labels[test_idx], proba))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def shuffled_label_test(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
) -> LeakProbeResult:
    """Shuffle labels and refit. AUC should drop to ~0.5.

    Caller compares this AUC against a baseline. A shuffled-label AUC
    materially above 0.5 indicates time-correlation leakage somewhere
    in the feature set or split strategy.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(labels)
    auc = probe_auc(features, shuffled, seed=seed)
    return LeakProbeResult(
        name="shuffled_label_test",
        auc=auc,
        n_samples=features.shape[0],
        notes="AUC should be ≈ 0.5 if features contain no time-leak",
    )


def planted_leak_test(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
    leak_strength: float = 0.9,
) -> LeakProbeResult:
    """Plant a future-leak column, refit, and confirm AUC jumps.

    The planted leak is `labels + small_noise`. A working detector should
    show AUC jump from baseline → near 1.0. If it doesn't, the probe
    itself is broken (numerical issue, label encoding bug) and other
    leak-detection results cannot be trusted.

    `leak_strength` controls how cleanly the leak is correlated with
    the label (1.0 = perfect copy; lower = noisier copy).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1 - leak_strength, size=labels.shape[0])
    leak_col = labels.astype(float) + noise
    leaked_features = np.column_stack([features, leak_col])
    auc = probe_auc(leaked_features, labels, seed=seed)
    return LeakProbeResult(
        name="planted_leak_test",
        auc=auc,
        n_samples=features.shape[0],
        notes="AUC should jump toward 1.0 vs baseline",
    )


def baseline_probe(
    features: np.ndarray, labels: np.ndarray, *, seed: int = 0
) -> LeakProbeResult:
    """Train probe on real data; return baseline AUC for comparison."""
    auc = probe_auc(features, labels, seed=seed)
    return LeakProbeResult(
        name="baseline_probe",
        auc=auc,
        n_samples=features.shape[0],
        notes="model AUC on real (feature, label) pairs",
    )
