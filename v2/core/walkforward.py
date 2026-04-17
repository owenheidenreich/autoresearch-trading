"""Walk-forward cross-validation for ART² v2.

Standard quant evaluation: train on the past, test on the future, slide forward.
5 canonical folds, expanding training window, 60-day test windows.

This is the immutable evaluation harness — the AI researcher cannot modify this file.

Core invariants (see /Users/gduby/.claude/plans/delightful-yawning-tiger.md):

1. window_id: every fold has a SHA-1 hash of its six boundary dates. Same
   calendar window -> same window_id -> same training seed, regardless of
   whether the fold was run alone (screen_latest) or inside a 5-fold job.

2. No promotion bypass: this module NEVER writes to the promoted model path.
   Fold checkpoints live under v2/artifacts/<experiment_id>/folds/<window_id>/.
   Promotion of a deployable model is the exclusive job of v2.ops.run_final_train
   followed by v2.ops.model_manage.keep().

3. screening_mode is the only fold-subset selector: "full" (all 5), "latest"
   (fold_idx = 4 only), or "mini" (folds 0, 2, 4). Fold ordinals are stable
   across modes — fold_idx=4 is the latest window whether the run is "latest"
   or "full". The old fold-count flag is gone.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from v2.core.metrics import ReplayMetrics, score_config_fingerprint
from v2.core.policy import DecisionPolicy, DEFAULT_POLICY
from v2.core.cv_report import CVReport, FoldSlice, PooledSlice, StabilitySlice


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CANONICAL_N_FOLDS = 5
ARTIFACTS_ROOT = "v2/artifacts"


@dataclass
class FoldSpec:
    """One round of walk-forward: train on past, test on future.

    `window_id` is a deterministic SHA-1 digest of the six boundary dates.
    It is the identity the seed is pinned to; `fold_idx` is a human label only.
    """
    fold_idx: int
    window_id: str
    train_days: list[str]
    val_days: list[str]
    test_days: list[str]


@dataclass
class FoldRunResult:
    """Raw per-fold run output, before CVReport assembly."""
    fold_idx: int
    window_id: str
    seed: int
    score: float
    metrics: ReplayMetrics
    baseline_scores: dict[str, float]
    train_seconds: float
    trades: list = field(default_factory=list)
    checkpoint_path: str = ""


# ---------------------------------------------------------------------------
# Window identity
# ---------------------------------------------------------------------------

def compute_window_id(
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> str:
    """Deterministic identity for a fold window. Seed pin point.

    The id is stable across runs: same six dates -> same digest. Screening the
    latest fold and running full CV therefore produce the same seed for the
    same calendar window.
    """
    payload = "|".join([train_start, train_end, val_start, val_end, test_start, test_end])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def window_seed(base_seed: int, window_id: str) -> int:
    """Derive a training seed from window_id. Not from fold_idx."""
    window_int = int(window_id[:8], 16) & 0x7FFFFFFF
    return (base_seed + window_int) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def generate_folds(
    unique_dates: list[str],
    n_folds: int = CANONICAL_N_FOLDS,
    test_window: int = 60,
    val_window: int = 40,
    shadow_days: int = 20,
) -> list[FoldSpec]:
    """Generate walk-forward fold specifications.

    Working backward from the end:
    - Last `shadow_days` are reserved (never used)
    - Before that, `n_folds * test_window` days are carved into test windows
    - Each fold trains on everything before its test window
    - Val is the last `val_window` days of each fold's training data

    Fold ordering: fold_idx=0 is the earliest test window, fold_idx=n-1 the latest.
    This ordering is stable: fold_idx=4 always means the latest window when
    n_folds=5, whether you screen it alone or run all folds.
    """
    n_days = len(unique_dates)
    last_test_end = n_days - shadow_days  # day index (exclusive)

    folds = []
    for i in range(n_folds):
        fold_idx = n_folds - 1 - i
        test_end = last_test_end - i * test_window
        test_start = test_end - test_window

        if test_start < val_window + 100:
            raise ValueError(
                f"Fold {fold_idx}: test_start={test_start} too early, "
                f"not enough training data (need {val_window + 100} days minimum)"
            )

        train_end = test_start
        val_start = train_end - val_window

        train_days = unique_dates[:train_end]
        val_days = unique_dates[val_start:train_end]
        test_days = unique_dates[test_start:test_end]

        window_id = compute_window_id(
            train_start=train_days[0],
            train_end=train_days[-1],
            val_start=val_days[0],
            val_end=val_days[-1],
            test_start=test_days[0],
            test_end=test_days[-1],
        )

        folds.append(FoldSpec(
            fold_idx=fold_idx,
            window_id=window_id,
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
        ))

    folds.sort(key=lambda f: f.fold_idx)
    return folds


def dates_to_mask(all_dates: list[str], selected_dates: set[str]) -> torch.Tensor:
    """Convert a set of date strings into a boolean mask over all bars."""
    mask = torch.zeros(len(all_dates), dtype=torch.bool)
    for i, d in enumerate(all_dates):
        if d in selected_dates:
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Screening-mode resolution
# ---------------------------------------------------------------------------

SCREENING_MODES = {
    "latest": "latest fold only (matches fold n-1 of full CV)",
    "mini":   "folds 0, 2, 4 (early, mid, late regime triage)",
    "full":   "all canonical folds (official cross-validation)",
}


def resolve_fold_indices(
    mode: str,
    total_folds: int,
    explicit: list[int] | None = None,
) -> list[int]:
    """Which canonical fold indices to evaluate for a given screening mode."""
    if explicit is not None:
        for idx in explicit:
            if idx < 0 or idx >= total_folds:
                raise ValueError(
                    f"fold_indices contains {idx}, out of range [0,{total_folds})"
                )
        return sorted(set(explicit))

    if mode == "full":
        return list(range(total_folds))
    if mode == "latest":
        return [total_folds - 1]
    if mode == "mini":
        mid = total_folds // 2
        return sorted({0, mid, total_folds - 1})
    raise ValueError(
        f"Unknown screening_mode={mode!r}. "
        f"Expected one of: {sorted(SCREENING_MODES)}"
    )


# ---------------------------------------------------------------------------
# Walk-forward execution
# ---------------------------------------------------------------------------

def run_walkforward(
    data_path: str = "v2/data.pt",
    *,
    experiment_id: str,
    screening_mode: str = "full",
    fold_indices: list[int] | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    base_seed: int = 123,
    total_folds: int = CANONICAL_N_FOLDS,
    test_window: int = 60,
    val_window: int = 40,
    shadow_days: int = 20,
    artifacts_root: str = ARTIFACTS_ROOT,
) -> CVReport:
    """Execute walk-forward cross-validation and return a CVReport.

    This function NEVER writes to v2/models/. Fold checkpoints are saved under
    {artifacts_root}/{experiment_id}/folds/<window_id>/model.pt as debug artifacts
    only. Promotion of a deployable model requires v2.ops.run_final_train.

    `screening_mode` and `fold_indices` select the subset of canonical folds to
    evaluate. Canonical folds are always generated against the full test horizon,
    so fold_idx meaning is stable across modes.
    """
    from v2.train import train as train_model
    from v2.replay import (
        load_model_from_path, replay_validation, print_metrics,
        compute_baseline_random, compute_baseline_atm_always,
        compute_baseline_simple_rules, compute_baseline_atm_trailing,
    )

    t_total_start = time.time()

    # Snapshot the training config and env overrides at CV start. All folds in
    # this CV share these. run_final_train replays them verbatim so the
    # deployed model is trained under the same knobs as the CV-selected run.
    from v2.train import _capture_env_overrides, _get_config_fingerprint
    training_env_overrides = _capture_env_overrides()
    training_config_fingerprint = _get_config_fingerprint()
    print(f"Training config fingerprint: {training_config_fingerprint}")
    if training_env_overrides:
        print(f"Training env overrides: {training_env_overrides}")

    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    all_dates = data['dates']
    unique_dates = sorted(set(all_dates))

    print(f"Dataset: {len(all_dates):,} bars, {len(unique_dates)} unique days")

    # Canonical fold generation (always the full set)
    all_folds = generate_folds(
        unique_dates,
        n_folds=total_folds,
        test_window=test_window,
        val_window=val_window,
        shadow_days=shadow_days,
    )

    selected_idx = resolve_fold_indices(screening_mode, total_folds, fold_indices)
    folds_to_run = [f for f in all_folds if f.fold_idx in selected_idx]

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD: mode={screening_mode}  folds={selected_idx}")
    print(f"{'='*60}")
    for f in all_folds:
        marker = "*" if f.fold_idx in selected_idx else " "
        print(f"  {marker} Fold {f.fold_idx} [win_id={f.window_id}]: "
              f"train {len(f.train_days)}d, val {len(f.val_days)}d, "
              f"test {len(f.test_days)}d ({f.test_days[0]} -> {f.test_days[-1]})")

    # Per-fold execution
    fold_results: list[FoldRunResult] = []
    dataset_fp = data.get('metadata', {}).get('fingerprint', 'unknown')

    for fold in folds_to_run:
        print(f"\n{'='*60}")
        print(f"  FOLD {fold.fold_idx}  window_id={fold.window_id}")
        print(f"{'='*60}")

        train_set = set(fold.train_days)
        val_set = set(fold.val_days)
        test_set = set(fold.test_days)

        train_mask = dates_to_mask(all_dates, train_set - val_set)
        val_mask = dates_to_mask(all_dates, val_set)
        test_mask = dates_to_mask(all_dates, test_set)

        print(f"  Train bars: {train_mask.sum():,}, Val bars: {val_mask.sum():,}, "
              f"Test bars: {test_mask.sum():,}")

        seed = window_seed(base_seed, fold.window_id)
        os.environ["TRAIN_SEED"] = str(seed)

        fold_dir = os.path.join(artifacts_root, experiment_id, "folds", fold.window_id)
        os.makedirs(fold_dir, exist_ok=True)
        fold_model_path = os.path.join(fold_dir, "model.pt")

        print(f"\n--- TRAINING (seed={seed}, window_id={fold.window_id}) ---")
        t_train = time.time()
        try:
            train_model(
                data_path=data_path,
                model_path=fold_model_path,
                train_mask_override=train_mask,
                val_mask_override=val_mask,
            )
            train_seconds = time.time() - t_train
            print(f"Training completed in {train_seconds:.1f}s")
        except Exception as e:
            print(f"FOLD {fold.fold_idx} CRASHED: {e}")
            fold_results.append(FoldRunResult(
                fold_idx=fold.fold_idx,
                window_id=fold.window_id,
                seed=seed,
                score=-999.0,
                metrics=ReplayMetrics(),
                baseline_scores={},
                train_seconds=0,
                trades=[],
                checkpoint_path=fold_model_path,
            ))
            continue

        print(f"\n--- REPLAY (fold {fold.fold_idx} test) ---")
        model = load_model_from_path(fold_model_path)

        wf_key = f"_wf_test_{fold.fold_idx}"
        data[wf_key] = test_mask

        metrics, trades, _ = replay_validation(
            model, data, mask_key=wf_key, policy=policy,
        )
        print_metrics(f"Fold {fold.fold_idx}", metrics)

        b_random = compute_baseline_random(data, mask_key=wf_key, policy=policy)
        b_atm = compute_baseline_atm_always(data, mask_key=wf_key, policy=policy)
        b_rules = compute_baseline_simple_rules(data, mask_key=wf_key, policy=policy)
        b_trailing = compute_baseline_atm_trailing(data, mask_key=wf_key, policy=policy)
        baseline_scores = {
            "random": b_random.score,
            "atm": b_atm.score,
            "rules": b_rules.score,
            "trailing": b_trailing.score,
        }

        del data[wf_key]

        fold_results.append(FoldRunResult(
            fold_idx=fold.fold_idx,
            window_id=fold.window_id,
            seed=seed,
            score=metrics.score,
            metrics=metrics,
            baseline_scores=baseline_scores,
            train_seconds=train_seconds,
            trades=list(trades),
            checkpoint_path=fold_model_path,
        ))

        print(f"  Fold {fold.fold_idx} score: {metrics.score:.4f}")

    # --- Assemble CVReport with scope-separated slices ---
    cv_report = _build_cv_report(
        experiment_id=experiment_id,
        screening_mode=screening_mode,
        fold_results=fold_results,
        folds_to_run=folds_to_run,
        policy=policy,
        dataset_fp=dataset_fp,
        training_config_fingerprint=training_config_fingerprint,
        training_env_overrides=training_env_overrides,
        training_seconds=time.time() - t_total_start,
    )

    _print_summary(cv_report, fold_results)
    return cv_report


def _build_cv_report(
    *,
    experiment_id: str,
    screening_mode: str,
    fold_results: list[FoldRunResult],
    folds_to_run: list[FoldSpec],
    policy: DecisionPolicy,
    dataset_fp: str,
    training_config_fingerprint: str,
    training_env_overrides: dict[str, str],
    training_seconds: float,
) -> CVReport:
    from v2.core.metrics import compute_metrics

    by_window = {fs.window_id: fs for fs in folds_to_run}
    fold_slices: list[FoldSlice] = []
    all_trades = []
    total_eval_days = 0
    per_fold_scores: list[float] = []
    per_fold_gate_failures: list[bool] = []

    for fr in fold_results:
        spec = by_window[fr.window_id]
        total_eval_days += len(spec.test_days)
        all_trades.extend(fr.trades)
        per_fold_scores.append(fr.score)
        per_fold_gate_failures.append(bool(fr.metrics.gate_failure))

        fold_slices.append(FoldSlice(
            fold_idx=fr.fold_idx,
            window_id=fr.window_id,
            test_window_start=spec.test_days[0],
            test_window_end=spec.test_days[-1],
            train_window_start=spec.train_days[0],
            train_window_end=spec.train_days[-1],
            seed=fr.seed,
            score=fr.score,
            gate_failure=fr.metrics.gate_failure,
            metrics=fr.metrics.to_dict(),
            baseline_scores=fr.baseline_scores,
            n_trades=fr.metrics.total_trades,
            n_test_days=len(spec.test_days),
            n_traded_days=fr.metrics.traded_days,
            train_seconds=fr.train_seconds,
        ))

    # Pooled metrics: re-score across all trades from all evaluated folds
    pooled_metrics = compute_metrics(all_trades, num_days=max(total_eval_days, 1))
    pooled = PooledSlice(
        profit_factor=pooled_metrics.profit_factor,
        max_account_drawdown=pooled_metrics.max_account_drawdown,
        win_rate=pooled_metrics.win_rate,
        call_pct=pooled_metrics.call_pct,
        put_pct=pooled_metrics.put_pct,
        net_pnl_dollars=pooled_metrics.net_pnl_dollars,
        total_trades=pooled_metrics.total_trades,
        total_eval_days=total_eval_days,
        traded_days=pooled_metrics.traded_days,
        positive_day_rate=pooled_metrics.positive_day_rate,
        daily_sortino=pooled_metrics.daily_sortino,
    )

    if per_fold_scores:
        arr = np.array(per_fold_scores, dtype=float)
        stability = StabilitySlice(
            mean_fold_score=float(arr.mean()),
            min_fold_score=float(arr.min()),
            max_fold_score=float(arr.max()),
            std_fold_score=float(arr.std()),
            per_fold_scores=per_fold_scores,
            per_fold_gate_failures=per_fold_gate_failures,
            any_fold_gate_failure=any(per_fold_gate_failures),
        )
    else:
        stability = StabilitySlice(
            mean_fold_score=0.0, min_fold_score=0.0, max_fold_score=0.0,
            std_fold_score=0.0, per_fold_scores=[],
            per_fold_gate_failures=[], any_fold_gate_failure=True,
        )

    agg_baselines = {}
    for key in ("random", "atm", "rules", "trailing"):
        scores = [fr.baseline_scores.get(key, -999) for fr in fold_results if fr.baseline_scores]
        agg_baselines[key] = float(np.mean(scores)) if scores else -999.0
    mean_score = stability.mean_fold_score
    beats_all = all(mean_score > v for v in agg_baselines.values()) if agg_baselines else False

    return CVReport(
        experiment_id=experiment_id,
        screening_mode=screening_mode,
        folds=fold_slices,
        pooled=pooled,
        stability=stability,
        aggregate_baselines=agg_baselines,
        beats_all_baselines=beats_all,
        training_config_fingerprint=training_config_fingerprint,
        training_env_overrides=training_env_overrides,
        policy_fingerprint=policy.fingerprint(),
        dataset_fingerprint=dataset_fp,
        evaluator_fingerprint=score_config_fingerprint(),
        training_seconds=training_seconds,
    )


def _print_summary(cv: CVReport, fold_results: list[FoldRunResult]) -> None:
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD SUMMARY  (mode={cv.screening_mode})")
    print(f"{'='*60}")
    for fr in fold_results:
        print(f"  Fold {fr.fold_idx} [{fr.window_id}]: "
              f"score={fr.score:.4f}  trades={fr.metrics.total_trades}  "
              f"traded_days={fr.metrics.traded_days}")
    print(f"  ---")
    print(f"  STABILITY  mean={cv.stability.mean_fold_score:.4f}  "
          f"min={cv.stability.min_fold_score:.4f}  "
          f"std={cv.stability.std_fold_score:.4f}")
    print(f"  POOLED     PF={cv.pooled.profit_factor:.3f}  "
          f"DD={cv.pooled.max_account_drawdown:.1%}  "
          f"trades={cv.pooled.total_trades}  "
          f"traded_days={cv.pooled.traded_days}/{cv.pooled.total_eval_days}")
    print(f"  GATES      any_fail={cv.stability.any_fold_gate_failure}")
    print(f"  Training time: {cv.training_seconds:.0f}s")
