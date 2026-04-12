"""Diagnostic: Direction Signal — can features predict call vs put?

Tests whether the 47 market features contain enough information to predict
the oracle's side choice (call vs put). Uses logistic regression as a
baseline — if even logistic regression can't beat 55%, the features lack
directional signal regardless of model architecture.

Also tests different lookback windows to see if more context helps.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_direction_dataset(data: dict, mask: np.ndarray, lookback: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and oracle side labels for bars where oracle trades."""
    features = data["X"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]

    from v2.core.chain_data import load_sidecar_cached
    from v2.core.policy import DEFAULT_POLICY

    mask_indices = np.where(mask)[0]
    all_dates = sorted(set(dates[i] for i in mask_indices))

    day_to_bars: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        day_to_bars.setdefault(d, []).append(i)

    X_list = []
    y_list = []

    for day in all_dates:
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))

        for bar_idx in day_to_bars.get(day, []):
            if bar_idx < lookback:
                continue
            bod = int(bar_of_day[bar_idx])
            if bod < DEFAULT_POLICY.no_trade_before_bar or bod >= DEFAULT_POLICY.no_trade_after_bar:
                continue
            if not bool(sidecar["bar_label_trade"][bod]):
                continue

            best_row_in_bar = int(sidecar["bar_best_contract_idx"][bod])
            if best_row_in_bar < 0:
                continue

            # Map relative index to absolute contract index
            ptr_start = int(sidecar["bar_ptrs"][bod])
            ptr_end = int(sidecar["bar_ptrs"][bod + 1])
            if ptr_start + best_row_in_bar >= ptr_end:
                continue
            best_contract_idx = int(sidecar["row_contract_idx"][ptr_start + best_row_in_bar])

            # Oracle's side: 0=call, 1=put
            side = int(sidecar["contract_right"][best_contract_idx])

            # Feature vector: use the LAST bar of the lookback window (current bar)
            feat_current = features[bar_idx]
            X_list.append(feat_current)
            y_list.append(side)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_direction_dataset_windowed(data: dict, mask: np.ndarray, lookback: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Same as above but uses MEAN of lookback window features (tests if more context helps)."""
    features = data["X"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]

    from v2.core.chain_data import load_sidecar_cached
    from v2.core.policy import DEFAULT_POLICY

    mask_indices = np.where(mask)[0]
    all_dates = sorted(set(dates[i] for i in mask_indices))

    day_to_bars: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        day_to_bars.setdefault(d, []).append(i)

    X_list = []
    y_list = []

    for day in all_dates:
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))

        for bar_idx in day_to_bars.get(day, []):
            if bar_idx < lookback:
                continue
            bod = int(bar_of_day[bar_idx])
            if bod < DEFAULT_POLICY.no_trade_before_bar or bod >= DEFAULT_POLICY.no_trade_after_bar:
                continue
            if not bool(sidecar["bar_label_trade"][bod]):
                continue

            best_row_in_bar = int(sidecar["bar_best_contract_idx"][bod])
            if best_row_in_bar < 0:
                continue

            ptr_start = int(sidecar["bar_ptrs"][bod])
            ptr_end = int(sidecar["bar_ptrs"][bod + 1])
            if ptr_start + best_row_in_bar >= ptr_end:
                continue
            best_contract_idx = int(sidecar["row_contract_idx"][ptr_start + best_row_in_bar])
            side = int(sidecar["contract_right"][best_contract_idx])

            # Use mean of lookback window as feature vector
            window = features[bar_idx - lookback : bar_idx]
            feat_mean = np.mean(window, axis=0)
            # Also include last bar and std for richer signal
            feat_std = np.std(window, axis=0)
            feat_last = features[bar_idx]
            feat = np.concatenate([feat_last, feat_mean, feat_std])
            X_list.append(feat)
            y_list.append(side)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def main():
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    print(f"Loaded data.pt: {len(data['dates'])} bars")

    train_mask = data["train_mask"].numpy()
    test_mask = data["promote_mask"].numpy()

    print(f"\n{'='*60}")
    print(f"  DIRECTION SIGNAL DIAGNOSTIC")
    print(f"{'='*60}")

    # --- Test 1: Single bar features (no lookback context) ---
    print("\n--- Test 1: Current-bar features only (47 features) ---")
    t0 = time.time()
    X_train, y_train = load_direction_dataset(data, train_mask, lookback=30)
    X_test, y_test = load_direction_dataset(data, test_mask, lookback=30)
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Train side balance: {np.mean(y_train):.3f} (0=call, 1=put)")
    print(f"  Test side balance:  {np.mean(y_test):.3f}")
    print(f"  Data loading: {time.time() - t0:.1f}s")

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=5.0, neginf=-5.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=5.0, neginf=-5.0)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Baseline (always majority): {max(np.mean(y_test), 1 - np.mean(y_test)):.4f}")

    # Top features by coefficient magnitude
    feat_importance = np.abs(clf.coef_[0])
    top_idx = np.argsort(feat_importance)[::-1][:10]
    print(f"\n  Top 10 features by |coefficient|:")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1}. Feature {idx}: coef={clf.coef_[0][idx]:.4f} (|{feat_importance[idx]:.4f}|)")

    # --- Test 2: Windowed features at different lookbacks ---
    for lb in [30, 60, 90]:
        print(f"\n--- Test 2: Windowed features (lookback={lb}, {47*3} features) ---")
        t0 = time.time()
        X_train_w, y_train_w = load_direction_dataset_windowed(data, train_mask, lookback=lb)
        X_test_w, y_test_w = load_direction_dataset_windowed(data, test_mask, lookback=lb)
        print(f"  Train: {len(X_train_w)}, Test: {len(X_test_w)}, Loading: {time.time() - t0:.1f}s")

        X_train_w = np.nan_to_num(X_train_w, nan=0.0, posinf=5.0, neginf=-5.0)
        X_test_w = np.nan_to_num(X_test_w, nan=0.0, posinf=5.0, neginf=-5.0)

        clf_w = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf_w.fit(X_train_w, y_train_w)
        train_acc_w = accuracy_score(y_train_w, clf_w.predict(X_train_w))
        test_acc_w = accuracy_score(y_test_w, clf_w.predict(X_test_w))
        print(f"  Train accuracy: {train_acc_w:.4f}")
        print(f"  Test accuracy:  {test_acc_w:.4f}")

    print(f"\n{'='*60}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print(f"  Test acc ~50%: Features lack directional signal")
    print(f"  Test acc 55-60%: Weak signal, architecture matters")
    print(f"  Test acc >60%: Strong signal, model should learn this")
    print(f"  Longer lookback improves: LOOKBACK=30 is too short")
    print(f"  Longer lookback same: Window length isn't the issue")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
