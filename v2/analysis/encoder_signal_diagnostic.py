"""Encoder signal preservation diagnostic.

Answers: does the trained encoder preserve the oracle-pnl-max signal that is
extractable from raw 52-d features? If encoder embeddings have much lower
rank-predictive power than raw features, the encoder is destroying signal
(→ architectural pivot). If they're similar, heads are the bottleneck
(→ longer training).

For each eligible bar in a fold's test window, we extract:
  - raw context = data.X[bar_idx] (52-d)
  - encoder context = model.forward(lookback, contracts)["context"] (96-d)
  - oracle target = data.best_contract_pnl[bar_idx]
  - model's own best_contract_score (for reference)
  - opportunity_logit (for reference)
  - side_logit (for reference)

Then day-based 80/20 split, fit LinearRegression(features -> oracle) on
train, measure Spearman rho on test. Compared across feature sources.

Usage:
    # Single fold (fold 4 / promote_mask)
    python3 -m v2.analysis.encoder_signal_diagnostic \\
        --model v2/artifacts/exp_171/folds/3b2f7c5202c9ee35/model.pt \\
        --mask promote

    # All 5 fold 171 checkpoints (pooled read)
    python3 -m v2.analysis.encoder_signal_diagnostic --all-171
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from v2.analysis.signal_viability import spearman
from v2.core.chain_data import load_sidecar_cached, padded_snapshot
from v2.core.policy import DEFAULT_POLICY
from v2.train import LOOKBACK, TradingModel


def load_model_lenient(path: str, device: str = "cpu") -> TradingModel:
    """Load a model checkpoint tolerant of removed heads (e.g. legacy no_trade_head).

    The encoder + surviving heads (contract scores, opportunity, side,
    aggression) are exactly what this diagnostic needs. Missing or unexpected
    keys are reported for transparency.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt.get("hyperparams", {})
    model = TradingModel(
        d_model=hp.get("d_model", 96),
        depth=hp.get("depth", 3),
        n_heads=hp.get("n_heads", 4),
        dropout=hp.get("dropout", 0.05),
    )
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"  [warn] missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  [info] dropped legacy keys: {result.unexpected_keys}")
    model.eval()
    return model


BATCH_SIZE = 2048

EXP_171_FOLDS = [
    # (win_id, label, test_date_lo, test_date_hi, train_days)
    ("09802c942e02b9b6", "fold 0", "2024-12-19", "2025-03-19", 666),
    ("cf38c16e2f55ddd9", "fold 1", "2025-03-20", "2025-06-13", 726),
    ("e56a4d66770d7097", "fold 2", "2025-06-16", "2025-09-10", 786),
    ("9b92333bccfcbbb7", "fold 3", "2025-09-11", "2025-12-04", 846),
    ("3b2f7c5202c9ee35", "fold 4", "2025-12-05", "2026-03-04", 906),
]


def fold_train_date_range(data, test_lo: str, train_days: int) -> tuple[str, str]:
    """Return (lo, hi) date strings for the fold's train window.

    Walk-forward CV: train window is the `train_days` days ending the day
    before `test_lo`. We compute this from the sorted unique dates in data,
    excluding the 40 val days immediately before test_lo.
    """
    all_days = sorted(set(data["dates"]))
    test_start_idx = all_days.index(test_lo)
    val_start = test_start_idx - 40
    train_hi_idx = val_start - 1
    train_lo_idx = max(0, train_hi_idx - train_days + 1)
    return all_days[train_lo_idx], all_days[train_hi_idx]


def collect_signals(model, data, mask_key: str | None = None,
                    date_range: tuple[str, str] | None = None,
                    device: str = "cpu"):
    """Return dict of per-bar arrays for eligible bars.

    Filters bars by mask_key (e.g. "promote_mask") or date_range (inclusive).
    If both are given, both must match. If neither, all bars are considered.
    """
    X = data["X"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    best_pnl = data["best_contract_pnl"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    max_contracts = int(data["metadata"]["max_contracts_per_bar"])

    if mask_key is not None:
        mask = data[mask_key].numpy().astype(bool)
    else:
        mask = np.ones(len(dates), dtype=bool)
    if date_range is not None:
        lo, hi = date_range
        in_range = np.array([lo <= d <= hi for d in dates])
        mask = mask & in_range

    mask_indices = np.where(mask)[0]
    eligible = []
    snapshots = []
    for bar_idx in mask_indices:
        if bar_idx < LOOKBACK:
            continue
        bod = int(bar_of_day[bar_idx])
        if bod < DEFAULT_POLICY.no_trade_before_bar or bod >= DEFAULT_POLICY.no_trade_after_bar:
            continue
        day = dates[bar_idx]
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        contracts, labels, _ = padded_snapshot(sidecar, bod, max_contracts)
        eligible.append((day, int(bar_idx)))
        snapshots.append((contracts, labels))

    if not eligible:
        return None

    bar_indices = np.array([b for _, b in eligible], dtype=np.int64)
    offsets = np.arange(-LOOKBACK, 0).reshape(1, -1)
    gather_idx = bar_indices.reshape(-1, 1) + offsets
    windows = X[gather_idx].astype(np.float32)
    contracts_arr = np.stack([s[0] for s in snapshots]).astype(np.float32)

    model = model.to(device)
    model.eval()

    contexts = []
    best_scores = []
    opp_logits = []
    side_logits = []

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(eligible), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(eligible))
            bx = torch.from_numpy(windows[start:end]).to(device)
            bc = torch.from_numpy(contracts_arr[start:end]).to(device)
            out = model(bx, bc)
            contexts.append(out["context"].cpu().numpy())
            cs = out["contract_scores"].cpu().numpy()
            vm = out["valid_mask"].cpu().numpy().astype(bool)
            cs_masked = np.where(vm, cs, -1e9)
            best_scores.append(cs_masked.max(axis=-1))
            opp_logits.append(out["opportunity_logit"].cpu().numpy())
            side_logits.append(out["side_logit"].cpu().numpy())
    print(f"  Inference: {time.time() - t0:.1f}s ({len(eligible)} bars)")

    context_arr = np.concatenate(contexts, axis=0)
    best_score_arr = np.concatenate(best_scores, axis=0)
    opp_arr = np.concatenate(opp_logits, axis=0)
    side_arr = np.concatenate(side_logits, axis=0)
    raw_arr = X[bar_indices]
    target = best_pnl[bar_indices]
    days = np.array([d for d, _ in eligible])

    return {
        "days": days,
        "bar_idx": bar_indices,
        "raw": raw_arr,
        "context": context_arr,
        "best_score": best_score_arr,
        "opp_logit": opp_arr,
        "side_logit": side_arr,
        "target": target,
    }


def day_split(days: np.ndarray, train_frac: float = 0.8, seed: int = 42):
    """Return (train_mask, test_mask) arrays for bars, split by day."""
    unique_days = sorted(set(days))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(unique_days))
    cut = int(len(unique_days) * train_frac)
    train_days = set(unique_days[i] for i in order[:cut])
    test_days = set(unique_days[i] for i in order[cut:])
    train_mask = np.array([d in train_days for d in days])
    test_mask = np.array([d in test_days for d in days])
    return train_mask, test_mask


def fit_lr_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression via closed form. Returns weight vector including bias."""
    n, d = X_train.shape
    Xb = np.hstack([X_train, np.ones((n, 1))])
    A = Xb.T @ Xb + alpha * np.eye(d + 1)
    A[-1, -1] -= alpha  # don't regularize bias
    w = np.linalg.solve(A, Xb.T @ y_train)
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ w


def rho_from_feature(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float = 1.0,
) -> tuple[float, float]:
    """Fit ridge on train, return (rho_train, rho_test)."""
    Xtr, ytr = features[train_mask], target[train_mask]
    Xte, yte = features[test_mask], target[test_mask]
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
        Xte = Xte.reshape(-1, 1)
    w = fit_lr_ridge(Xtr, ytr, alpha=alpha)
    ptr = predict(Xtr, w)
    pte = predict(Xte, w)
    return spearman(ptr.tolist(), ytr.tolist()), spearman(pte.tolist(), yte.tolist())


def train_test_generalization(data):
    """For each fold: fit LR on raw features AND on encoder context over the
    fold's training window, evaluate Spearman ρ on the test window.

    Three-way comparison:
      - raw LR: pure features, no encoder.
      - context LR: uses fold's own trained encoder (frozen) as feature extractor.
      - model best_contract_score: the trained model's own ranking on test bars
        (read from the earlier per-fold table above, not recomputed here).

    Together these separate "signal in features", "signal in encoder
    representation", and "signal in the scoring head".
    """
    X = data["X"].numpy()
    dates_arr = np.array(data["dates"])
    bar_of_day = data["bar_of_day"].numpy()
    best_pnl = data["best_contract_pnl"].numpy()

    print(f"\n  {'fold':12s}  {'train_bars':>10s}  {'test_bars':>9s}  "
          f"{'raw_tr':>8s}  {'raw_te':>8s}  {'ctx_tr':>8s}  {'ctx_te':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for win_id, label, test_lo, test_hi, train_days in EXP_171_FOLDS:
        train_lo, train_hi = fold_train_date_range(data, test_lo, train_days)
        window_mask = (bar_of_day >= DEFAULT_POLICY.no_trade_before_bar) & (bar_of_day < DEFAULT_POLICY.no_trade_after_bar)
        tr_mask = (dates_arr >= train_lo) & (dates_arr <= train_hi) & window_mask
        te_mask = (dates_arr >= test_lo) & (dates_arr <= test_hi) & window_mask
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        # --- raw LR ---
        w_raw = fit_lr_ridge(X[tr_idx], best_pnl[tr_idx], alpha=1.0)
        raw_tr = spearman(predict(X[tr_idx], w_raw).tolist(), best_pnl[tr_idx].tolist())
        raw_te = spearman(predict(X[te_idx], w_raw).tolist(), best_pnl[te_idx].tolist())

        # --- context LR (encoder extracts embedding; heads bypassed) ---
        model_path = f"v2/artifacts/exp_171/folds/{win_id}/model.pt"
        if os.path.exists(model_path):
            model = load_model_lenient(model_path)
            # Subsample train bars 4x for tractable sidecar loading; evaluate on full test
            tr_sub = tr_idx[::4]
            ctx_tr_vec = _extract_contexts_for_bars(model, data, tr_sub)
            ctx_te_vec = _extract_contexts_for_bars(model, data, te_idx)
            w_ctx = fit_lr_ridge(ctx_tr_vec, best_pnl[tr_sub], alpha=1.0)
            ctx_tr = spearman(predict(ctx_tr_vec, w_ctx).tolist(), best_pnl[tr_sub].tolist())
            ctx_te = spearman(predict(ctx_te_vec, w_ctx).tolist(), best_pnl[te_idx].tolist())
        else:
            ctx_tr = ctx_te = float("nan")

        print(f"  {label:12s}  {len(tr_idx):>10d}  {len(te_idx):>9d}  "
              f"{raw_tr:>8.4f}  {raw_te:>8.4f}  {ctx_tr:>8.4f}  {ctx_te:>8.4f}")


def _extract_contexts_for_bars(model, data, bar_idx_array):
    """Run encoder on lookback windows for given bar indices; return context array."""
    X = data["X"].numpy().astype(np.float32)
    dates = data["dates"]
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    max_contracts = int(data["metadata"]["max_contracts_per_bar"])
    bar_of_day = data["bar_of_day"].numpy()

    valid = bar_idx_array[bar_idx_array >= LOOKBACK]
    offsets = np.arange(-LOOKBACK, 0).reshape(1, -1)
    windows = X[valid.reshape(-1, 1) + offsets]

    contracts_list = []
    for bi in valid:
        day = dates[bi]
        bod = int(bar_of_day[bi])
        sc = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        c, _, _ = padded_snapshot(sc, bod, max_contracts)
        contracts_list.append(c)
    contracts_arr = np.stack(contracts_list).astype(np.float32)

    model.eval()
    out_ctx = []
    with torch.no_grad():
        for start in range(0, len(valid), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(valid))
            bx = torch.from_numpy(windows[start:end])
            bc = torch.from_numpy(contracts_arr[start:end])
            out = model(bx, bc)
            out_ctx.append(out["context"].cpu().numpy())
    return np.concatenate(out_ctx, axis=0)


def diagnose(model_path: str, data, mask_key: str | None = None,
             date_range: tuple[str, str] | None = None, label: str = "") -> dict:
    tag = f"mask={mask_key}" if mask_key else f"dates={date_range[0]}:{date_range[1]}" if date_range else "all"
    print(f"\n=== {label or model_path}  ({tag}) ===")
    model = load_model_lenient(model_path)
    sig = collect_signals(model, data, mask_key=mask_key, date_range=date_range)
    if sig is None:
        print("  no eligible bars")
        return {}

    n = len(sig["target"])
    unique_days = len(set(sig["days"]))
    print(f"  eligible bars: {n}   days: {unique_days}")
    print(f"  target stats: mean={sig['target'].mean():.3f}  "
          f"std={sig['target'].std():.3f}  pct_pos={(sig['target']>0).mean():.2%}")

    train_mask, test_mask = day_split(sig["days"], train_frac=0.8, seed=42)
    ntr, nte = int(train_mask.sum()), int(test_mask.sum())
    print(f"  split (by day, seed=42): train {ntr} bars  test {nte} bars")

    # In-sample ρ (B3 audit methodology: no holdout)
    all_mask = np.ones(len(sig["target"]), dtype=bool)

    print(f"\n  {'feature':28s}  {'dim':>4s}  {'rho_pool':>10s}  {'rho_train':>10s}  {'rho_test':>10s}")
    print(f"  {'-'*28}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")
    results = {}
    for name, feat in [
        ("raw (52-d)",        sig["raw"]),
        ("encoder context (96-d)", sig["context"]),
        ("best_contract_score (1)", sig["best_score"]),
        ("opportunity_logit (1)",   sig["opp_logit"]),
        ("side_logit (1)",          sig["side_logit"]),
    ]:
        # Pooled (B3-style, in-sample fit and eval on same bars)
        rpool, _ = rho_from_feature(feat, sig["target"], all_mask, all_mask)
        rtr, rte = rho_from_feature(feat, sig["target"], train_mask, test_mask)
        results[name] = (rpool, rtr, rte)
        d = feat.shape[1] if feat.ndim > 1 else 1
        print(f"  {name:28s}  {d:4d}  {rpool:>10.4f}  {rtr:>10.4f}  {rte:>10.4f}")

    return {"label": label, "mask_key": mask_key, "n_bars": n, "n_days": unique_days,
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--mask", default=None, help="train/val/promote")
    ap.add_argument("--date-range", default=None, help="YYYY-MM-DD:YYYY-MM-DD")
    ap.add_argument("--all-171", action="store_true",
                    help="Run across all 5 exp_171 fold checkpoints on their own test windows")
    args = ap.parse_args()

    mask_key = None
    if args.mask:
        mask_key = args.mask if args.mask.endswith("_mask") else f"{args.mask}_mask"
    date_range = None
    if args.date_range:
        lo, hi = args.date_range.split(":")
        date_range = (lo, hi)

    print(f"Loading {args.data}...")
    data = torch.load(args.data, map_location="cpu", weights_only=False)

    if args.all_171:
        all_results = []
        for win_id, label, test_lo, test_hi, train_days in EXP_171_FOLDS:
            path = f"v2/artifacts/exp_171/folds/{win_id}/model.pt"
            if not os.path.exists(path):
                print(f"  [skip] {path} not found")
                continue
            r = diagnose(path, data, date_range=(test_lo, test_hi),
                         label=f"exp_171 {label} ({win_id[:8]})")
            all_results.append(r)

        print("\n\n" + "=" * 60)
        print("  TRAIN -> TEST generalization (mirrors model's own split)")
        print("=" * 60)
        print("LR fit on fold's training window, evaluated on fold's test window.")
        print("Pure feature-level read: does the signal that fits on train days")
        print("transfer to test days, 40-60 days later after the val gap?")
        train_test_generalization(data)
    else:
        path = args.model or "v2/artifacts/exp_171/folds/3b2f7c5202c9ee35/model.pt"
        diagnose(path, data, mask_key=mask_key, date_range=date_range,
                 label=os.path.basename(os.path.dirname(path)))


if __name__ == "__main__":
    main()
