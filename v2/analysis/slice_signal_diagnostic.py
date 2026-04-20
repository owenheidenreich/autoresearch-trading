"""Dynamic-slice signal diagnostic.

Compares three signals on the dynamic near-ATM universe:
  1. raw context -> slice oracle PnL via ridge regression
  2. encoder context -> slice oracle PnL via ridge regression
  3. model's own best in-slice contract score vs slice oracle PnL

This is the cheap local audit for the new slice-first harness before spending
GPU on a fresh experiment.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from v2.analysis.signal_viability import spearman
from v2.core.chain_data import load_sidecar_cached, padded_snapshot_with_slice
from v2.core.policy import DEFAULT_POLICY
from v2.replay import adapt_windows_for_model
from v2.train import (
    LOOKBACK,
    TradingModel,
    _checkpoint_gate_arch,
    _checkpoint_input_feature_names,
    _checkpoint_uses_linear_heads,
)

BATCH_SIZE = 2048

EXP_171_FOLDS = [
    ("09802c942e02b9b6", "fold 0", "2024-12-19", "2025-03-19"),
    ("cf38c16e2f55ddd9", "fold 1", "2025-03-20", "2025-06-13"),
    ("e56a4d66770d7097", "fold 2", "2025-06-16", "2025-09-10"),
    ("9b92333bccfcbbb7", "fold 3", "2025-09-11", "2025-12-04"),
    ("3b2f7c5202c9ee35", "fold 4", "2025-12-05", "2026-03-04"),
]


def load_model_lenient(path: str, device: str = "cpu") -> TradingModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt.get("hyperparams", {}) or {}
    n_features = int(hp.get("num_features", ckpt["model_state_dict"]["input_proj.weight"].shape[1]))
    model = TradingModel(
        d_model=hp.get("d_model", 96),
        depth=hp.get("depth", 3),
        n_heads=hp.get("n_heads", 4),
        dropout=hp.get("dropout", 0.05),
        linear_score_heads=_checkpoint_uses_linear_heads(ckpt),
        num_features=n_features,
        gate_arch=_checkpoint_gate_arch(ckpt),
    )
    model.input_feature_names = _checkpoint_input_feature_names(ckpt)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"  [warn] missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  [info] dropped legacy keys: {result.unexpected_keys}")
    model.eval()
    return model


def _fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xb = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, Xb.T @ y_train)


def _predict_ridge(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ w


def _rho_train_test(features: np.ndarray, target: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> tuple[float, float]:
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    Xtr = np.nan_to_num(features[train_mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(features[test_mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Xtr = np.clip(Xtr, -1e6, 1e6)
    Xte = np.clip(Xte, -1e6, 1e6)
    ytr = np.nan_to_num(target[train_mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    yte = np.nan_to_num(target[test_mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = Xtr.mean(axis=0, keepdims=True)
    sigma = Xtr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Xtr = np.clip((Xtr - mu) / sigma, -10.0, 10.0)
    Xte = np.clip((Xte - mu) / sigma, -10.0, 10.0)
    w = _fit_ridge(Xtr, ytr)
    ptr = _predict_ridge(Xtr, w)
    pte = _predict_ridge(Xte, w)
    return spearman(ptr.tolist(), ytr.tolist()), spearman(pte.tolist(), yte.tolist())


def collect_slice_signals(model, data: dict, *, mask_key: str | None = None, date_range: tuple[str, str] | None = None, device: str = "cpu") -> dict | None:
    X = data["X"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    target_full = data.get("slice_best_contract_pnl", data["best_contract_pnl"]).numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    max_contracts = int(data["metadata"]["max_contracts_per_bar"])
    dataset_feature_names = list(data.get("metadata", {}).get("feature_names", []))

    if mask_key is not None:
        mask = data[mask_key].numpy().astype(bool)
    else:
        mask = np.ones(len(dates), dtype=bool)
    if date_range is not None:
        lo, hi = date_range
        mask &= np.array([lo <= d <= hi for d in dates], dtype=bool)

    eligible = []
    snapshots = []
    for bar_idx in np.where(mask)[0]:
        if bar_idx < LOOKBACK:
            continue
        bod = int(bar_of_day[bar_idx])
        if bod < DEFAULT_POLICY.no_trade_before_bar or bod >= DEFAULT_POLICY.no_trade_after_bar:
            continue
        day = dates[bar_idx]
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        contracts, labels, _, slice_mask = padded_snapshot_with_slice(sidecar, bod, max_contracts)
        if not slice_mask.any():
            continue
        eligible.append((day, int(bar_idx)))
        snapshots.append((contracts, labels, slice_mask))

    if not eligible:
        return None

    bar_indices = np.array([idx for _, idx in eligible], dtype=np.int64)
    offsets = np.arange(-LOOKBACK, 0).reshape(1, -1)
    windows = X[bar_indices.reshape(-1, 1) + offsets].astype(np.float32)
    windows = adapt_windows_for_model(windows, dataset_feature_names, model)
    contracts_arr = np.stack([s[0] for s in snapshots]).astype(np.float32)
    slice_masks = np.stack([s[2] for s in snapshots]).astype(bool)

    model = model.to(device)
    model.eval()

    contexts = []
    best_scores = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(eligible), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(eligible))
            bx = torch.from_numpy(windows[start:end]).to(device)
            bc = torch.from_numpy(contracts_arr[start:end]).to(device)
            out = model(bx, bc)
            contexts.append(out["context"].cpu().numpy())
            scores = out["contract_scores"].cpu().numpy()
            valid = out["valid_mask"].cpu().numpy().astype(bool) & slice_masks[start:end]
            masked = np.where(valid, scores, -1e9)
            best_scores.append(masked.max(axis=-1))
    print(f"  Inference: {time.time() - t0:.1f}s ({len(eligible)} bars)")

    return {
        "days": np.array([d for d, _ in eligible]),
        "bar_idx": bar_indices,
        "raw": X[bar_indices],
        "context": np.concatenate(contexts, axis=0),
        "best_score": np.concatenate(best_scores, axis=0),
        "target": target_full[bar_indices],
    }


def _day_split(days: np.ndarray, train_frac: float = 0.8, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    unique_days = sorted(set(days))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(unique_days))
    cut = int(len(unique_days) * train_frac)
    train_days = set(unique_days[i] for i in order[:cut])
    test_days = set(unique_days[i] for i in order[cut:])
    train_mask = np.array([d in train_days for d in days], dtype=bool)
    test_mask = np.array([d in test_days for d in days], dtype=bool)
    return train_mask, test_mask


def run_one(model_path: str, data_path: str, *, mask_key: str | None = None, date_range: tuple[str, str] | None = None, device: str = "cpu") -> None:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    model = load_model_lenient(model_path, device=device)
    sig = collect_slice_signals(model, data, mask_key=mask_key, date_range=date_range, device=device)
    if sig is None:
        print("No eligible slice bars found.")
        return

    train_mask, test_mask = _day_split(sig["days"])
    raw_tr, raw_te = _rho_train_test(sig["raw"], sig["target"], train_mask, test_mask)
    ctx_tr, ctx_te = _rho_train_test(sig["context"], sig["target"], train_mask, test_mask)
    score_te = spearman(sig["best_score"][test_mask].tolist(), sig["target"][test_mask].tolist())
    print(f"raw_lr    rho_train={raw_tr:+.4f} rho_test={raw_te:+.4f}")
    print(f"ctx_lr    rho_train={ctx_tr:+.4f} rho_test={ctx_te:+.4f}")
    print(f"model_top rho_test ={score_te:+.4f}")


def run_all_171(data_path: str, device: str = "cpu") -> None:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    rows = []
    for win_id, label, lo, hi in EXP_171_FOLDS:
        model_path = f"v2/artifacts/exp_171/folds/{win_id}/model.pt"
        if not os.path.exists(model_path):
            print(f"[skip] {label}: missing {model_path}")
            continue
        print(f"\n=== {label} ({lo} -> {hi}) ===")
        model = load_model_lenient(model_path, device=device)
        sig = collect_slice_signals(model, data, date_range=(lo, hi), device=device)
        if sig is None:
            print("  no eligible slice bars")
            continue
        train_mask, test_mask = _day_split(sig["days"])
        raw_tr, raw_te = _rho_train_test(sig["raw"], sig["target"], train_mask, test_mask)
        ctx_tr, ctx_te = _rho_train_test(sig["context"], sig["target"], train_mask, test_mask)
        score_te = spearman(sig["best_score"][test_mask].tolist(), sig["target"][test_mask].tolist())
        rows.append((label, raw_te, ctx_te, score_te))
        print(f"  raw_lr    rho_test={raw_te:+.4f}")
        print(f"  ctx_lr    rho_test={ctx_te:+.4f}")
        print(f"  model_top rho_test={score_te:+.4f}")

    if rows:
        print("\n=== Summary ===")
        for label, raw_te, ctx_te, score_te in rows:
            print(f"{label:>6}  raw={raw_te:+.4f}  ctx={ctx_te:+.4f}  model_top={score_te:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic-slice signal diagnostic")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--mask", type=str, default=None, choices=["train", "val", "promote", "shadow"])
    parser.add_argument("--date-range", type=str, default=None, help="YYYY-MM-DD:YYYY-MM-DD inclusive")
    parser.add_argument("--all-171", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.all_171:
        run_all_171(args.data, device=args.device)
        return

    if not args.model:
        raise SystemExit("--model is required unless --all-171 is used")

    date_range = None
    if args.date_range:
        lo, hi = args.date_range.split(":")
        date_range = (lo.strip(), hi.strip())
    mask_key = f"{args.mask}_mask" if args.mask else None
    run_one(args.model, args.data, mask_key=mask_key, date_range=date_range, device=args.device)


if __name__ == "__main__":
    main()
