"""Walk-forward cross-validation for ART² v2.

Standard quant evaluation: train on the past, test on the future, slide forward.
5 rounds, expanding training window, 60-day test windows.

This is immutable evaluation harness -- the AI researcher cannot modify this file.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from v2.core.metrics import ReplayMetrics
from v2.core.policy import DecisionPolicy, DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoldSpec:
    """One round of walk-forward: train on past, test on future."""
    fold_idx: int
    train_days: list[str]   # dates for training
    val_days: list[str]     # dates for checkpoint selection (subset of train_days)
    test_days: list[str]    # dates for evaluation (never seen during training)


@dataclass
class FoldResult:
    """Results from one walk-forward round."""
    fold_idx: int
    score: float
    metrics: ReplayMetrics
    baseline_scores: dict[str, float]
    train_seconds: float
    n_trades: int
    n_test_days: int


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward rounds."""
    fold_results: list[FoldResult]
    aggregate_score: float
    min_fold_score: float
    max_fold_score: float
    std_fold_score: float
    per_fold_scores: list[float]
    total_trades: int
    total_test_days: int
    aggregate_baselines: dict[str, float]
    beats_all_baselines: bool
    training_seconds: float


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def generate_folds(
    unique_dates: list[str],
    n_folds: int = 5,
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
    """
    n_days = len(unique_dates)
    # Last test window ends before shadow
    last_test_end = n_days - shadow_days  # day index (exclusive)

    folds = []
    for i in range(n_folds):
        # Work backward: fold n_folds-1 is the latest, fold 0 is the earliest
        fold_idx = n_folds - 1 - i
        test_end = last_test_end - i * test_window
        test_start = test_end - test_window

        if test_start < val_window + 100:
            # Not enough training data (need at least 100 days + val)
            raise ValueError(
                f"Fold {fold_idx}: test_start={test_start} too early, "
                f"not enough training data (need {val_window + 100} days minimum)"
            )

        train_end = test_start  # train on everything before test
        val_start = train_end - val_window

        fold = FoldSpec(
            fold_idx=fold_idx,
            train_days=unique_dates[:train_end],
            val_days=unique_dates[val_start:train_end],
            test_days=unique_dates[test_start:test_end],
        )
        folds.append(fold)

    # Sort by fold_idx (earliest first)
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
# Walk-forward execution
# ---------------------------------------------------------------------------

def run_walkforward(
    data_path: str = "v2/data.pt",
    model_path: str = "v2/models/model.pt",
    n_folds: int = 5,
    test_window: int = 60,
    val_window: int = 40,
    shadow_days: int = 20,
    policy: DecisionPolicy = DEFAULT_POLICY,
    base_seed: int = 123,
) -> WalkForwardResult:
    """Execute full walk-forward cross-validation.

    For each fold:
    1. Generate train/val/test masks
    2. Train model from scratch (fresh random init with fold-specific seed)
    3. Replay on test mask
    4. Compute baselines on test mask
    5. Collect scores

    Returns aggregated results.
    """
    from v2.train import train as train_model, SEED as default_seed
    from v2.replay import (
        load_model_from_path, replay_validation, print_metrics,
        compute_baseline_random, compute_baseline_atm_always,
        compute_baseline_simple_rules, compute_baseline_atm_trailing,
    )

    t_total_start = time.time()

    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    all_dates = data['dates']
    unique_dates = sorted(set(all_dates))

    print(f"Dataset: {len(all_dates):,} bars, {len(unique_dates)} unique days")

    # Generate folds
    folds = generate_folds(
        unique_dates, n_folds=n_folds,
        test_window=test_window, val_window=val_window,
        shadow_days=shadow_days,
    )

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD: {n_folds} folds, {test_window}-day test windows")
    print(f"{'='*60}")
    for f in folds:
        print(f"  Fold {f.fold_idx}: train {len(f.train_days)}d, "
              f"val {len(f.val_days)}d, test {len(f.test_days)}d "
              f"({f.test_days[0]} to {f.test_days[-1]})")

    fold_results = []

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  FOLD {fold.fold_idx}")
        print(f"{'='*60}")

        # 1. Generate masks
        train_set = set(fold.train_days)
        val_set = set(fold.val_days)
        test_set = set(fold.test_days)

        train_mask = dates_to_mask(all_dates, train_set - val_set)  # exclude val from train
        val_mask = dates_to_mask(all_dates, val_set)
        test_mask = dates_to_mask(all_dates, test_set)

        print(f"  Train bars: {train_mask.sum():,}, Val bars: {val_mask.sum():,}, "
              f"Test bars: {test_mask.sum():,}")

        # 2. Train with fold-specific seed
        fold_seed = base_seed + fold.fold_idx
        os.environ["TRAIN_SEED"] = str(fold_seed)

        fold_model_path = f"v2/models/model_fold{fold.fold_idx}.pt"

        print(f"\n--- TRAINING (seed={fold_seed}) ---")
        t_train = time.time()
        try:
            model, train_metrics = train_model(
                data_path=data_path,
                model_path=fold_model_path,
                train_mask_override=train_mask,
                val_mask_override=val_mask,
            )
            train_seconds = time.time() - t_train
            print(f"Training completed in {train_seconds:.1f}s")
        except Exception as e:
            print(f"FOLD {fold.fold_idx} CRASHED: {e}")
            fold_results.append(FoldResult(
                fold_idx=fold.fold_idx, score=-999.0,
                metrics=ReplayMetrics(), baseline_scores={},
                train_seconds=0, n_trades=0, n_test_days=0,
            ))
            continue

        # 3. Replay on test mask
        print(f"\n--- REPLAY (fold {fold.fold_idx} test) ---")
        model = load_model_from_path(fold_model_path)

        # Inject test mask into data dict
        wf_key = f"_wf_test_{fold.fold_idx}"
        data[wf_key] = test_mask

        metrics, trades = replay_validation(
            model, data, mask_key=wf_key, policy=policy,
        )
        print_metrics(f"Fold {fold.fold_idx}", metrics)

        # 4. Baselines on same test mask
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

        # Clean up temp mask
        del data[wf_key]

        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            score=metrics.score,
            metrics=metrics,
            baseline_scores=baseline_scores,
            train_seconds=train_seconds,
            n_trades=metrics.total_trades,
            n_test_days=metrics.traded_days,
        ))

        print(f"  Fold {fold.fold_idx} score: {metrics.score:.4f}")

    # 5. Copy last fold's model to canonical path (most training data)
    last_fold_model = f"v2/models/model_fold{folds[-1].fold_idx}.pt"
    if os.path.exists(last_fold_model):
        import shutil
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.copy2(last_fold_model, model_path)
        print(f"\nProduction model: {last_fold_model} -> {model_path}")
    else:
        print(f"\nWARNING: Expected fold model {last_fold_model} not found — no production model saved")

    # Clean up fold models
    for fold in folds:
        fp = f"v2/models/model_fold{fold.fold_idx}.pt"
        if os.path.exists(fp) and fp != model_path:
            os.remove(fp)

    # 6. Aggregate
    per_fold_scores = [fr.score for fr in fold_results]
    total_training = time.time() - t_total_start

    # Aggregate baselines
    agg_baselines = {}
    for key in ["random", "atm", "rules", "trailing"]:
        scores = [fr.baseline_scores.get(key, -999) for fr in fold_results
                  if fr.baseline_scores]
        agg_baselines[key] = np.mean(scores) if scores else -999

    agg_score = np.mean(per_fold_scores)

    result = WalkForwardResult(
        fold_results=fold_results,
        aggregate_score=float(agg_score),
        min_fold_score=float(np.min(per_fold_scores)),
        max_fold_score=float(np.max(per_fold_scores)),
        std_fold_score=float(np.std(per_fold_scores)),
        per_fold_scores=per_fold_scores,
        total_trades=sum(fr.n_trades for fr in fold_results),
        total_test_days=sum(fr.n_test_days for fr in fold_results),
        aggregate_baselines=agg_baselines,
        beats_all_baselines=all(
            agg_score > agg_baselines[k] for k in agg_baselines
        ),
        training_seconds=total_training,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'='*60}")
    for fr in fold_results:
        print(f"  Fold {fr.fold_idx}: score={fr.score:.4f}  "
              f"trades={fr.n_trades}  days={fr.n_test_days}")
    print(f"  ---")
    print(f"  Aggregate score: {result.aggregate_score:.4f}")
    print(f"  Min fold score:  {result.min_fold_score:.4f}")
    print(f"  Std fold score:  {result.std_fold_score:.4f}")
    print(f"  Total trades:    {result.total_trades}")
    print(f"  Total test days: {result.total_test_days}")
    print(f"  Beats baselines: {result.beats_all_baselines}")
    print(f"  Total time:      {result.training_seconds:.0f}s")

    return result
