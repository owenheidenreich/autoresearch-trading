"""Bar-level gate-target signal audit with governance metrics.

Tests whether a chosen binary bar-level label is learnable from the context
features **and** produces an operationally stable rank order over the realized
``slice_best_pnl``. Writes a provenance JSON (see ``v2/core/provenance.py``)
that proves comparability to future results.

Labels (``--label-mode``):
  * ``bar_quality``       — existing: ``_compute_bar_opportunity_quality >= BAR_QUALITY_PASS_THRESHOLD``
  * ``positive_ev``       — ``bar_slice_best_pnl[local_bar] > 0`` (research-tier)
  * ``oracle_side_call``  — best in-slice contract is a call (research-tier)

Metrics reported per fold:
  * per-fold base rate on train/val
  * AUC (logistic regression + 1-hidden-layer MLP) on the val slice
  * top-decile precision and precision *lift* (precision / val base rate)
  * threshold sweep at pass rates {0.05, 0.10, 0.20, 0.30, 0.50}: realized mean
    ``slice_best_pnl`` on the top-K val bars by score (plus monotonicity check)

Panel baselines (scored on the same sweep axis):
  * ``always_on``          — realized mean slice_best_pnl on every val bar
  * ``random_gate``        — uniform random score (matched pass rate in expectation)
  * ``prior_bar_quality``  — LR on the old ``bar_quality`` binary label
  * ``trivial5_lr``        — LR on 5 named features only (vix_regime,
                              intraday_sin, intraday_cos, ret_6, vwap_dist)

Gate A passes iff ≥ ceil(n_folds / 2) folds clear **all four** criteria:
  * LR or MLP AUC ≥ ``--auc-bar`` (default 0.60)
  * LR or MLP top-decile precision lift ≥ ``--lift-bar`` (default 1.5)
  * threshold sweep is monotone (one inversion allowed at the widest pass rate)
  * realized mean at 10% pass rate beats **both** prior ``bar_quality`` panel
    and ``trivial5_lr`` panel on the same fold

Exit code 0 iff Gate A passes. Otherwise exit 2.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from v2.core.chain_data import load_sidecar_cached
from v2.core.policy import DEFAULT_POLICY
from v2.core.provenance import build_provenance, write_provenance
from v2.core.walkforward import generate_folds, resolve_fold_indices
from v2.train import (
    ACTIVE_FEATURE_NAMES,
    BAR_QUALITY_PASS_THRESHOLD,
    LOOKBACK,
    _compute_bar_opportunity_quality,
)


SWEEP_PASS_RATES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)
TRIVIAL5_FEATURE_NAMES: tuple[str, ...] = (
    "vix_regime",
    "intraday_sin",
    "intraday_cos",
    "ret_6",
    "vwap_dist",
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
    lr_top_decile_lift: float
    mlp_top_decile_lift: float
    # threshold sweep: realized mean slice_best_pnl at each pass rate, LR scores
    lr_sweep_realized_mean: dict[float, float] = field(default_factory=dict)
    mlp_sweep_realized_mean: dict[float, float] = field(default_factory=dict)
    # panel baselines
    panel_always_on_mean: float = float("nan")
    panel_random_mean: float = float("nan")
    panel_prior_bq_sweep: dict[float, float] = field(default_factory=dict)
    panel_trivial5_sweep: dict[float, float] = field(default_factory=dict)


def _bar_label_value(sc: dict, local_bar: int, label_mode: str) -> int | None:
    """Binary label for the audit target, or ``None`` if the bar is not labelable."""
    if label_mode == "bar_quality":
        quality = _compute_bar_opportunity_quality(sc, local_bar)
        return 1 if quality >= BAR_QUALITY_PASS_THRESHOLD else 0
    if label_mode == "positive_ev":
        slice_best = sc.get("bar_slice_best_pnl")
        if slice_best is None:
            return None
        val = float(slice_best[local_bar])
        if not np.isfinite(val):
            return None
        return 1 if val > 0.0 else 0
    if label_mode == "oracle_side_call":
        best_idx_arr = sc.get("bar_slice_best_contract_idx")
        if best_idx_arr is None:
            return None
        bar_ptrs = sc["bar_ptrs"]
        start = int(bar_ptrs[local_bar])
        end = int(bar_ptrs[local_bar + 1])
        best_local = int(best_idx_arr[local_bar])
        if best_local < 0 or best_local >= (end - start):
            return None
        row = start + best_local
        contract_idx = int(sc["row_contract_idx"][row])
        # chain_data.md contract_right convention: 0=call, 1=put
        right = int(sc["contract_right"][contract_idx])
        return 1 if right == 0 else 0
    raise ValueError(f"Unknown label_mode={label_mode!r}")


def _collect_fold_arrays(
    *,
    features: np.ndarray,
    dates: list[str],
    bar_of_day: np.ndarray,
    sidecar_dir: str,
    day_set: set[str],
    representation: str,
    label_mode: str,
    also_bar_quality: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, slice_best_pnl, y_bar_quality) for all eligible bars in day_set.

    ``y_bar_quality`` is populated only when ``also_bar_quality`` is True
    (needed to score the prior-bar-quality panel baseline without re-walking
    the sidecars); otherwise it is a zero-size placeholder.
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    pnl_rows: list[float] = []
    y_bq_rows: list[int] = []
    for global_bar in range(LOOKBACK, len(dates)):
        day = dates[global_bar]
        if day not in day_set:
            continue
        local_bar = int(bar_of_day[global_bar])
        if (
            local_bar < DEFAULT_POLICY.no_trade_before_bar
            or local_bar >= DEFAULT_POLICY.no_trade_after_bar
        ):
            continue
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        labelable = bool(
            sidecar.get("bar_slice_labelable", sidecar["bar_labelable"])[local_bar]
        )
        if not labelable:
            continue
        y = _bar_label_value(sidecar, local_bar, label_mode)
        if y is None:
            continue
        slice_best = sidecar.get("bar_slice_best_pnl")
        pnl = float(slice_best[local_bar]) if slice_best is not None else float("nan")
        if not np.isfinite(pnl):
            # Without a realized pnl we cannot evaluate the threshold sweep;
            # drop the bar from training too so train/val stay aligned.
            continue
        window = features[global_bar - LOOKBACK:global_bar]
        if representation == "current":
            row = features[global_bar]
        elif representation == "pooled":
            row = np.concatenate([window.mean(axis=0), window.std(axis=0)])
        elif representation == "flat":
            row = window.reshape(-1)
        else:
            raise ValueError(f"Unknown representation={representation!r}")
        X_rows.append(row)
        y_rows.append(y)
        pnl_rows.append(pnl)
        if also_bar_quality:
            bq = _bar_label_value(sidecar, local_bar, "bar_quality")
            y_bq_rows.append(int(bq) if bq is not None else 0)
    if not X_rows:
        empty = np.zeros((0,), dtype=np.float32)
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            empty,
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.stack(X_rows).astype(np.float32, copy=False),
        np.asarray(y_rows, dtype=np.int32),
        np.asarray(pnl_rows, dtype=np.float32),
        np.asarray(y_bq_rows, dtype=np.int32) if also_bar_quality else np.zeros((0,), dtype=np.int32),
    )


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


def _threshold_sweep(
    scores: np.ndarray,
    realized_pnl: np.ndarray,
    pass_rates: tuple[float, ...] = SWEEP_PASS_RATES,
) -> dict[float, float]:
    """For each pass rate K, return mean ``realized_pnl`` on the top-K scored bars."""
    out: dict[float, float] = {}
    n = len(scores)
    if n == 0:
        return {k: float("nan") for k in pass_rates}
    order = np.argsort(-scores)
    for k in pass_rates:
        cut = max(1, int(math.floor(n * k)))
        top_idx = order[:cut]
        out[k] = float(realized_pnl[top_idx].mean())
    return out


def _sweep_is_monotone(sweep: dict[float, float], allow_inversions: int = 1) -> bool:
    """True iff the sweep is non-increasing as pass-rate rises. At most
    ``allow_inversions`` strict violations are tolerated (e.g. at the widest
    pass rate where the statistic stabilises at the population mean)."""
    rates = sorted(sweep.keys())
    vals = [sweep[r] for r in rates]
    inversions = 0
    for i in range(1, len(vals)):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i - 1]):
            continue
        if vals[i] > vals[i - 1] + 1e-9:
            inversions += 1
    return inversions <= allow_inversions


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
    label_mode: str,
    trivial5_idx: list[int] | None,
) -> FoldResult:
    t0 = time.time()

    X_train, y_train, pnl_train, y_bq_train = _collect_fold_arrays(
        features=features, dates=dates, bar_of_day=bar_of_day,
        sidecar_dir=sidecar_dir, day_set=train_days,
        representation=representation, label_mode=label_mode,
        also_bar_quality=(label_mode != "bar_quality"),
    )
    X_val, y_val, pnl_val, _y_bq_val = _collect_fold_arrays(
        features=features, dates=dates, bar_of_day=bar_of_day,
        sidecar_dir=sidecar_dir, day_set=val_days,
        representation=representation, label_mode=label_mode,
        also_bar_quality=False,
    )
    empty_sweep = {k: float("nan") for k in SWEEP_PASS_RATES}
    if X_train.size == 0 or X_val.size == 0:
        return FoldResult(
            fold_idx=fold_idx, train_n=len(y_train), val_n=len(y_val),
            train_base_rate=float("nan"), val_base_rate=float("nan"),
            lr_auc=float("nan"), mlp_auc=float("nan"),
            lr_top_decile_precision=float("nan"),
            mlp_top_decile_precision=float("nan"),
            lr_top_decile_lift=float("nan"),
            mlp_top_decile_lift=float("nan"),
            lr_sweep_realized_mean=dict(empty_sweep),
            mlp_sweep_realized_mean=dict(empty_sweep),
            panel_always_on_mean=float("nan"),
            panel_random_mean=float("nan"),
            panel_prior_bq_sweep=dict(empty_sweep),
            panel_trivial5_sweep=dict(empty_sweep),
        )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Primary model: LR + MLP on the audit label.
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs", random_state=seed,
    ).fit(X_train_s, y_train)
    lr_scores = lr.decision_function(X_val_s)

    mlp = _train_mlp(X_train_s, y_train, seed=seed)
    mlp_scores = _mlp_scores(mlp, X_val_s)

    val_base = float(y_val.mean()) if len(y_val) else float("nan")
    lr_p10 = _top_decile_precision(y_val, lr_scores)
    mlp_p10 = _top_decile_precision(y_val, mlp_scores)
    lr_lift = lr_p10 / val_base if val_base > 0 else float("nan")
    mlp_lift = mlp_p10 / val_base if val_base > 0 else float("nan")

    lr_sweep = _threshold_sweep(lr_scores, pnl_val)
    mlp_sweep = _threshold_sweep(mlp_scores, pnl_val)

    # Panels -------------------------------------------------------------
    always_on_mean = float(pnl_val.mean()) if len(pnl_val) else float("nan")
    rng = np.random.default_rng(seed)
    random_scores = rng.random(len(pnl_val), dtype=np.float32)
    random_mean = float(pnl_val[np.argsort(-random_scores)[:max(1, int(0.10 * len(pnl_val)))]].mean()) if len(pnl_val) else float("nan")

    # Prior bar-quality panel: LR on the OLD bar_quality label (only if the
    # audit label is not itself bar_quality).
    prior_bq_sweep: dict[float, float]
    if label_mode != "bar_quality" and len(y_bq_train) and y_bq_train.sum() > 0:
        lr_bq = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs", random_state=seed,
        ).fit(X_train_s, y_bq_train)
        bq_scores = lr_bq.decision_function(X_val_s)
        prior_bq_sweep = _threshold_sweep(bq_scores, pnl_val)
    else:
        prior_bq_sweep = dict(empty_sweep)

    # Trivial-5 panel: LR on five named features only.
    if trivial5_idx and representation == "current":
        X_train_5 = X_train[:, trivial5_idx]
        X_val_5 = X_val[:, trivial5_idx]
        scaler5 = StandardScaler().fit(X_train_5)
        lr5 = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs", random_state=seed,
        ).fit(scaler5.transform(X_train_5), y_train)
        s5 = lr5.decision_function(scaler5.transform(X_val_5))
        trivial5_sweep = _threshold_sweep(s5, pnl_val)
    else:
        trivial5_sweep = dict(empty_sweep)

    result = FoldResult(
        fold_idx=fold_idx,
        train_n=len(y_train),
        val_n=len(y_val),
        train_base_rate=float(y_train.mean()),
        val_base_rate=val_base,
        lr_auc=_auc_safely(y_val, lr_scores),
        mlp_auc=_auc_safely(y_val, mlp_scores),
        lr_top_decile_precision=lr_p10,
        mlp_top_decile_precision=mlp_p10,
        lr_top_decile_lift=lr_lift,
        mlp_top_decile_lift=mlp_lift,
        lr_sweep_realized_mean=lr_sweep,
        mlp_sweep_realized_mean=mlp_sweep,
        panel_always_on_mean=always_on_mean,
        panel_random_mean=random_mean,
        panel_prior_bq_sweep=prior_bq_sweep,
        panel_trivial5_sweep=trivial5_sweep,
    )

    sweep_repr = " | ".join(f"{k:.2f}={lr_sweep[k]:+.3f}" for k in sorted(lr_sweep.keys()))
    print(
        f"fold {fold_idx}: n_train={result.train_n:,} n_val={result.val_n:,} "
        f"base_train={result.train_base_rate:.3f} base_val={result.val_base_rate:.3f}\n"
        f"  LR_auc={result.lr_auc:.3f} MLP_auc={result.mlp_auc:.3f} "
        f"LR_top10={result.lr_top_decile_precision:.3f} ({result.lr_top_decile_lift:.2f}x) "
        f"MLP_top10={result.mlp_top_decile_precision:.3f} ({result.mlp_top_decile_lift:.2f}x)\n"
        f"  LR_sweep realized_mean slice_best_pnl: {sweep_repr}\n"
        f"  panel always_on_mean={result.panel_always_on_mean:+.3f} "
        f"random_top10_mean={result.panel_random_mean:+.3f} "
        f"prior_bq@10%={prior_bq_sweep.get(0.10, float('nan')):+.3f} "
        f"trivial5@10%={trivial5_sweep.get(0.10, float('nan')):+.3f} "
        f"[{time.time() - t0:.1f}s]"
    )
    return result


def _gate_decision(
    results: list[FoldResult],
    *,
    auc_bar: float,
    lift_bar: float,
    strict: bool,
) -> tuple[bool, str]:
    if not results:
        return False, "no folds evaluated"

    required = len(results) if strict else math.ceil(len(results) / 2)
    passed = 0
    reasons: list[str] = []
    for r in results:
        auc_ok = (
            (not math.isnan(r.lr_auc) and r.lr_auc >= auc_bar)
            or (not math.isnan(r.mlp_auc) and r.mlp_auc >= auc_bar)
        )
        lift_ok = (
            (not math.isnan(r.lr_top_decile_lift) and r.lr_top_decile_lift >= lift_bar)
            or (not math.isnan(r.mlp_top_decile_lift) and r.mlp_top_decile_lift >= lift_bar)
        )
        mono_ok = _sweep_is_monotone(r.lr_sweep_realized_mean) or _sweep_is_monotone(r.mlp_sweep_realized_mean)
        lr_realized = r.lr_sweep_realized_mean.get(0.10, float("nan"))
        prior_realized = r.panel_prior_bq_sweep.get(0.10, float("nan"))
        trivial_realized = r.panel_trivial5_sweep.get(0.10, float("nan"))
        panel_ok = (
            np.isfinite(lr_realized)
            and (not np.isfinite(prior_realized) or lr_realized >= prior_realized)
            and (not np.isfinite(trivial_realized) or lr_realized >= trivial_realized)
        )
        fold_pass = auc_ok and lift_ok and mono_ok and panel_ok
        if fold_pass:
            passed += 1
        reasons.append(
            f"fold {r.fold_idx}: auc={auc_ok} lift={lift_ok} mono={mono_ok} "
            f"panel={panel_ok} → {'PASS' if fold_pass else 'FAIL'}"
        )

    ok = passed >= required
    verdict = (
        f"{passed}/{len(results)} folds clear all four criteria "
        f"(AUC>={auc_bar:.2f}, lift>={lift_bar:.2f}x, monotone sweep, "
        f"realized@10% beats prior_bq and trivial5) "
        f"(required: {'all' if strict else f'>={required}'})"
    )
    for line in reasons:
        verdict += "\n  " + line
    return ok, verdict


def _serialise_sweep(sweep: dict[float, float]) -> dict[str, float]:
    return {f"{k:.2f}": float(v) for k, v in sweep.items()}


def _serialise_result(r: FoldResult) -> dict:
    return {
        "fold_idx": r.fold_idx,
        "train_n": r.train_n,
        "val_n": r.val_n,
        "train_base_rate": r.train_base_rate,
        "val_base_rate": r.val_base_rate,
        "lr_auc": r.lr_auc,
        "mlp_auc": r.mlp_auc,
        "lr_top_decile_precision": r.lr_top_decile_precision,
        "mlp_top_decile_precision": r.mlp_top_decile_precision,
        "lr_top_decile_lift": r.lr_top_decile_lift,
        "mlp_top_decile_lift": r.mlp_top_decile_lift,
        "lr_sweep_realized_mean": _serialise_sweep(r.lr_sweep_realized_mean),
        "mlp_sweep_realized_mean": _serialise_sweep(r.mlp_sweep_realized_mean),
        "panel_always_on_mean": r.panel_always_on_mean,
        "panel_random_top10_mean": r.panel_random_mean,
        "panel_prior_bq_sweep": _serialise_sweep(r.panel_prior_bq_sweep),
        "panel_trivial5_sweep": _serialise_sweep(r.panel_trivial5_sweep),
    }


def _resolve_trivial5_idx() -> list[int]:
    name_to_idx = {n: i for i, n in enumerate(ACTIVE_FEATURE_NAMES)}
    try:
        return [name_to_idx[n] for n in TRIVIAL5_FEATURE_NAMES]
    except KeyError as exc:
        raise RuntimeError(
            f"Trivial-5 feature {exc.args[0]!r} missing from ACTIVE_FEATURE_NAMES — "
            f"update TRIVIAL5_FEATURE_NAMES or the feature set."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU diagnostic: does a LR/MLP baseline clear the bar-level-label "
            "learnability bar before we spend GPU? Writes a provenance JSON."
        )
    )
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--screen-mode", choices=("latest", "mini", "full"), default="mini")
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold indices.")
    parser.add_argument("--auc-bar", type=float, default=0.60)
    parser.add_argument("--lift-bar", type=float, default=1.5)
    parser.add_argument("--precision-bar", type=float, default=0.25,
                        help="(legacy) absolute top-decile precision floor; no longer used for Gate A.")
    parser.add_argument("--strict", action="store_true",
                        help="Require every selected fold to clear all criteria.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--representation",
        choices=("current", "pooled", "flat"),
        default="current",
    )
    parser.add_argument(
        "--label-mode",
        choices=("bar_quality", "positive_ev", "oracle_side_call"),
        default="bar_quality",
    )
    parser.add_argument(
        "--provenance-out",
        default=None,
        help=(
            "Path to write provenance JSON. Defaults to "
            "v2/artifacts/cpu_audits/bar_quality_signal_<git_sha>_<label_mode>.json"
        ),
    )
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    features = data["X"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dataset_fingerprint = str(data["metadata"].get("dataset_fingerprint", "unknown"))
    sidecar_schema_version = str(data["metadata"].get("schema_version", "unknown"))
    unique_dates = sorted(set(dates))

    explicit = None
    if args.folds:
        explicit = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    all_folds = generate_folds(unique_dates)
    selected = resolve_fold_indices(args.screen_mode, len(all_folds), explicit)
    research_tier = args.label_mode != "bar_quality"
    trivial5_idx = _resolve_trivial5_idx() if args.representation == "current" else None

    print(
        f"Bar-level signal audit: label_mode={args.label_mode} "
        f"screen_mode={args.screen_mode} folds={selected} "
        f"representation={args.representation} "
        f"auc_bar={args.auc_bar} lift_bar={args.lift_bar} strict={args.strict} "
        f"research_tier={research_tier}"
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
            label_mode=args.label_mode,
            trivial5_idx=trivial5_idx,
        ))

    print("-" * 88)
    ok, verdict = _gate_decision(
        results,
        auc_bar=args.auc_bar,
        lift_bar=args.lift_bar,
        strict=args.strict,
    )
    print(f"decision: {'PASS' if ok else 'FAIL'} — {verdict}")
    if ok:
        print("  -> signal is learnable AND operationally stable; proceed to Gate B (GPU screen).")
    else:
        print("  -> Gate A failed. Do not spend GPU on this label.")
        print("     Pivot options: (a) redefine target, (b) enrich features, (c) regime-condition.")

    # Provenance ----------------------------------------------------------
    label_thresholds: dict[str, float | int] = {}
    if args.label_mode == "bar_quality":
        label_thresholds["BAR_QUALITY_PASS_THRESHOLD"] = BAR_QUALITY_PASS_THRESHOLD
    provenance = build_provenance(
        data_path=args.data,
        feature_names=list(ACTIVE_FEATURE_NAMES),
        label_mode=args.label_mode,
        label_thresholds=label_thresholds,
        screen_mode=args.screen_mode,
        n_folds=len(selected),
        research_tier=research_tier,
        dataset_fingerprint=dataset_fingerprint,
        sidecar_schema_version=sidecar_schema_version,
    )

    extra = {
        "representation": args.representation,
        "auc_bar": args.auc_bar,
        "lift_bar": args.lift_bar,
        "strict": args.strict,
        "selected_folds": list(selected),
        "decision": "PASS" if ok else "FAIL",
        "verdict": verdict,
        "fold_results": [_serialise_result(r) for r in results],
    }

    out_path = args.provenance_out
    if out_path is None:
        git_sha_short = provenance.git_commit[:8] if provenance.git_commit else "unknown"
        out_path = os.path.join(
            "v2", "artifacts", "cpu_audits",
            f"bar_quality_signal_{git_sha_short}_{args.label_mode}.json",
        )
    write_provenance(out_path, provenance, extra=extra)
    print(f"  provenance written: {out_path}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
