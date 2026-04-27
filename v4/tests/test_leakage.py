"""Tests for the leak-detection harness.

These tests verify the harness *works on synthetic data* — i.e., it can
distinguish leaky from non-leaky inputs. They are NOT tests of any real
features; that's a Phase 2A+ activity.
"""
from __future__ import annotations

import numpy as np
import pytest

from v4.leakage import (
    baseline_probe,
    planted_leak_test,
    probe_auc,
    shuffled_label_test,
)


def _signal_dataset(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic: features carry signal; labels follow a logistic on x[:, 0]."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logits = 1.5 * X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2]
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def _no_signal_dataset(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Features are pure noise; labels are independent."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = rng.integers(0, 2, size=n)
    return X, y


def test_baseline_recovers_signal_when_present() -> None:
    X, y = _signal_dataset()
    res = baseline_probe(X, y)
    # With moderate signal, AUC should be well above chance
    assert res.auc > 0.7


def test_baseline_chance_when_no_signal() -> None:
    X, y = _no_signal_dataset()
    res = baseline_probe(X, y)
    assert 0.4 < res.auc < 0.6


def test_shuffled_label_drops_to_chance_on_real_signal() -> None:
    """The whole point: with a real-signal dataset, shuffling labels
    should crush AUC to ~0.5."""
    X, y = _signal_dataset()
    baseline = baseline_probe(X, y)
    shuffled = shuffled_label_test(X, y)
    # Baseline should be much higher than shuffled
    assert baseline.auc - shuffled.auc > 0.15
    # Shuffled itself should be close to chance
    assert 0.4 < shuffled.auc < 0.6


def test_planted_leak_jumps_auc_to_near_one() -> None:
    """The detector must catch a planted leak. If this fails, the
    leak-detection itself is broken."""
    X, y = _signal_dataset()
    baseline = baseline_probe(X, y)
    leaked = planted_leak_test(X, y, leak_strength=0.95)
    # Leaked AUC should be close to 1.0 and well above baseline
    assert leaked.auc > 0.95
    assert leaked.auc - baseline.auc > 0.10


def test_planted_leak_jumps_even_with_no_baseline_signal() -> None:
    """Planted leak should produce strong AUC even when the underlying
    features carry no signal."""
    X, y = _no_signal_dataset()
    leaked = planted_leak_test(X, y, leak_strength=0.95)
    assert leaked.auc > 0.9


def test_probe_auc_rejects_shape_mismatch() -> None:
    X = np.zeros((5, 3))
    y = np.array([0, 1, 0])  # wrong length
    with pytest.raises(ValueError, match="disagree on length"):
        probe_auc(X, y)


def test_probe_auc_rejects_1d_features() -> None:
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="features must be 2-D"):
        probe_auc(X, y)
