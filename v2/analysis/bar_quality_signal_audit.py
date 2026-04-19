"""Bar-quality signal audit: is the binary bar-quality label learnable from features?

Motivation
----------
After the fold-0 partial-GPU readout for `exp_next_c1_screen_mini` showed
`val_replay` selecting the "abstain everything" checkpoint (epoch 1, PF=0.000)
as best across 15 epochs, the open question is whether the gate can ever learn
to commit. The Bayes-optimal action under a belief that `P(bar_quality=1 | x)`
is indistinguishable from its unconditional mean **is** always abstain — so if
a reference model given the same features cannot predict bar-quality above
chance, no gate-architecture tweak will rescue it.

This script fits two reference models per fold on the same train/val split that
`CKPT_SELECTION_MODE=val_replay` uses:
  * `sklearn.linear_model.LogisticRegression` on current-bar features
  * a 1-hidden-layer torch MLP on current-bar features

and reports, on the held-out val slice:
  * AUC
  * top-decile precision (matches `POLICY_GATE_TARGET_PASS_RATE=0.10`)
  * positive-label base rate

Decision rule
-------------
Proceed with the GPU rerun only if signal is learnable. The audit exits with
code 0 iff, on at least ceil(n_folds/2) folds, **both**:
  * LR or MLP AUC >= AUC_BAR (default 0.60)
  * LR or MLP top-decile precision >= PRECISION_BAR (default 0.25)

Use `--strict` to require both reference models to clear both bars on every
selected fold.

Example
-------
    python3 -m v2.analysis.bar_quality_signal_audit --screen-mode mini
    python3 -m v2.analysis.bar_quality_signal_audit --screen-mode latest --auc-bar 0.62
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from v2.core.chain_data import load_sidecar_cached
from v2.core.policy import DEFAULT_POLICY
from v2.core.walkforward import generate_folds, resolve_fold_indices
from v2.train import (
    ACTIVE_FEATURE_NAMES,
    BAR_QUALITY_PASS_THRESHOLD,
    LOOKBACK,
    _compute_bar_opportunity_quality,
)


@dataclass
class FoldResult:
    fold_idx: int
    train_n: int
    val_n: int
    train_base_rate: float
    val_base_rate: float
    lr_auc: float
    mlp_auc: float
    lr_top_decile_precision: float
    mlp_top_decile_precision: float


def _collect_fold_arrays(
    *,
    features: np.ndarray,
    dates: list[str],
    bar_of_day: np.ndarray,
    sidecar_dir: str,
    day_set: set[str],
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """For every eligible bar whose day is in `day_set`, return (X, y).

    Eligibility matches `gate_label_audit`: post-lookback, inside the trade
    window, and sidecar-labelable. Labels are the binary bar-quality target.

    representation:
      * "current" — last bar only (NUM_FEATURES dims)
      * "pooled"  — mean + std of lookback window (2 * NUM_FEATURES dims)
      * "flat"    — flattened lookback window (LOOKBACK * NUM_FEATURES dims)
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for global_bar in range(LOOKBACK, len(dates)):
        day = dates[global_bar]
        if day not in day_set:
            continue
        local_bar = int(bar_of_day[global_bar])
        if local_bar < DEFAULT_POLICY.no_trade_before_bar or local_bar >= DEFAULT_POLICY.no_trade_after_bar:
            continue
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        labelable = bool(sidecar.get("bar_slice_labelable", sidecar["bar_labelable"])[local_bar])
        if not labelable:
            continue
        quality = _compute_bar_opportunity_quality(sidecar, local_bar)
        window = features[global_bar - LOOKBACK:global_bar]  # [LOOKBACK, F]
        if representation == "current":
            row = features[global_bar]
        elif representation == "pooled":
            row = np.concatenate([window.mean(axis=0), window.std(axis=0)])
        elif representation == "flat":
            row = window.reshape(-1)
        else:
            raise ValueError(f"Unknown representation={representation!r}")
        X_rows.append(row)
        y_rows.append(1 if quality >= BAR_QUALITY_PASS_THRESHOLD else 0)
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.stack(X_rows).astype(np.float32, copy=False), np.asarray(y_rows, dtype=np.int32)


class _TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int = 64,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)
    model = _TinyMLP(in_dim=X_train.shape[1], hidden=hidden).to(device)

    pos = float(max(y_train.sum(), 1))
    neg = float(max(len(y_train) - pos, 1))
    pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    n = X_t.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _mlp_scores(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
    return logits.cpu().numpy()


def _top_decile_precision(y_true: np.ndarray, scores: np.ndarray, q: float = 0.10) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(math.floor(len(y_true) * q)))
    order = np.argsort(-scores)
    top = order[:k]
    return float(y_true[top].mean())


def _auc_safely(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _evaluate_fold(
    *,
    fold_idx: int,
    features: np.ndarray,
    dates: list[str],
    bar_of_day: np.ndarray,
    sidecar_dir: str,
    train_days: set[str],
    val_days: set[str],
    seed: int,
    representation: str,
) -> FoldResult:
    t0 = time.time()
    X_train, y_train = _collect_fold_arrays(
        features=features, dates=dates, bar_of_day=bar_of_day,
        sidecar_dir=sidecar_dir, day_set=train_days,
        representation=representation,
    )
    X_val, y_val = _collect_fold_arrays(
        features=features, dates=dates, bar_of_day=bar_of_day,
        sidecar_dir=sidecar_dir, day_set=val_days,
        representation=representation,
    )
    if X_train.size == 0 or X_val.size == 0:
        return FoldResult(
            fold_idx=fold_idx, train_n=len(y_train), val_n=len(y_val),
            train_base_rate=float("nan"), val_base_rate=float("nan"),
            lr_auc=float("nan"), mlp_auc=float("nan"),
            lr_top_decile_precision=float("nan"),
            mlp_top_decile_precision=float("nan"),
        )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        random_state=seed,
    ).fit(X_train_s, y_train)
    lr_scores = lr.decision_function(X_val_s)

    mlp = _train_mlp(X_train_s, y_train, seed=seed)
    mlp_scores = _mlp_scores(mlp, X_val_s)

    result = FoldResult(
        fold_idx=fold_idx,
        train_n=len(y_train),
        val_n=len(y_val),
        train_base_rate=float(y_train.mean()),
        val_base_rate=float(y_val.mean()),
        lr_auc=_auc_safely(y_val, lr_scores),
        mlp_auc=_auc_safely(y_val, mlp_scores),
        lr_top_decile_precision=_top_decile_precision(y_val, lr_scores),
        mlp_top_decile_precision=_top_decile_precision(y_val, mlp_scores),
    )
    print(
        f"fold {fold_idx}: "
        f"train_n={result.train_n:,} val_n={result.val_n:,} "
        f"base_rate(train)={result.train_base_rate:.3f} base_rate(val)={result.val_base_rate:.3f} | "
        f"LR_auc={result.lr_auc:.3f} MLP_auc={result.mlp_auc:.3f} | "
        f"LR_top10={result.lr_top_decile_precision:.3f} MLP_top10={result.mlp_top_decile_precision:.3f} "
        f"[{time.time() - t0:.1f}s]"
    )
    return result


def _gate_decision(
    results: list[FoldResult],
    *,
    auc_bar: float,
    precision_bar: float,
    strict: bool,
) -> tuple[bool, str]:
    if not results:
        return False, "no folds evaluated"

    required = len(results) if strict else math.ceil(len(results) / 2)
    passed = 0
    for r in results:
        auc_ok = (not math.isnan(r.lr_auc) and r.lr_auc >= auc_bar) or \
                 (not math.isnan(r.mlp_auc) and r.mlp_auc >= auc_bar)
        precision_ok = r.lr_top_decile_precision >= precision_bar or \
                       r.mlp_top_decile_precision >= precision_bar
        if auc_ok and precision_ok:
            passed += 1

    ok = passed >= required
    verdict = (
        f"{passed}/{len(results)} folds clear AUC>={auc_bar:.2f} "
        f"AND top10_precision>={precision_bar:.2f} "
        f"(required: {'all' if strict else f'>={required}'})"
    )
    return ok, verdict


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU diagnostic: does a LR/MLP baseline clear the bar-quality "
            "learnability bar before we spend GPU on another gate-tuning rerun?"
        )
    )
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--screen-mode", choices=("latest", "mini", "full"), default="mini")
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold indices.")
    parser.add_argument("--auc-bar", type=float, default=0.60)
    parser.add_argument("--precision-bar", type=float, default=0.25)
    parser.add_argument("--strict", action="store_true",
                        help="Require every selected fold to clear both bars.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--representation",
        choices=("current", "pooled", "flat"),
        default="current",
        help=(
            "Feature representation: 'current' = last-bar features (NUM_FEATURES dims), "
            "'pooled' = mean+std over lookback (2*NUM_FEATURES), "
            "'flat' = full lookback flattened (LOOKBACK*NUM_FEATURES)."
        ),
    )
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    features = data["X"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    unique_dates = sorted(set(dates))

    explicit = None
    if args.folds:
        explicit = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    all_folds = generate_folds(unique_dates)
    selected = resolve_fold_indices(args.screen_mode, len(all_folds), explicit)

    print(
        f"Bar-quality signal audit: screen_mode={args.screen_mode} folds={selected} "
        f"representation={args.representation} "
        f"auc_bar={args.auc_bar} precision_bar={args.precision_bar} strict={args.strict}"
    )
    if args.representation == "current":
        in_dims = features.shape[1]
    elif args.representation == "pooled":
        in_dims = 2 * features.shape[1]
    else:
        in_dims = LOOKBACK * features.shape[1]
    print(
        f"features: base_n_dims={features.shape[1]} effective_in_dims={in_dims} "
        f"lookback={LOOKBACK} active_feature_names[:3]={ACTIVE_FEATURE_NAMES[:3]} "
        f"bar_quality_threshold={BAR_QUALITY_PASS_THRESHOLD}"
    )
    print("-" * 88)

    results: list[FoldResult] = []
    for fold in all_folds:
        if fold.fold_idx not in selected:
            continue
        train_days = set(fold.train_days) - set(fold.val_days)
        val_days = set(fold.val_days)
        results.append(_evaluate_fold(
            fold_idx=fold.fold_idx,
            features=features,
            dates=dates,
            bar_of_day=bar_of_day,
            sidecar_dir=sidecar_dir,
            train_days=train_days,
            val_days=val_days,
            seed=args.seed,
            representation=args.representation,
        ))

    print("-" * 88)
    ok, verdict = _gate_decision(
        results,
        auc_bar=args.auc_bar,
        precision_bar=args.precision_bar,
        strict=args.strict,
    )
    print(f"decision: {'PASS' if ok else 'FAIL'} — {verdict}")
    if ok:
        print("  -> signal is learnable; proceed with the GPU rerun.")
    else:
        print("  -> signal is NOT learnable from current features. Do not spend GPU.")
        print("     Pivot options: (a) coarsen the bar-quality definition, "
              "(b) enrich features, (c) regime-condition the target.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
