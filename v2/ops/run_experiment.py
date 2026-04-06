"""Run a single ART² experiment: train + replay + print score.

This is the v2 equivalent of `uv run train.py` in Karpathy's autoresearch.
One command, one score number. The AI researcher runs this after each mutation.

Usage:
    python v2/ops/run_experiment.py [--data v2/data.pt] [--id exp_001]

Output format (grep-friendly):
    ---
    score:                2.345678
    daily_sortino:        3.12
    positive_day_rate:    0.72
    max_account_drawdown: 0.06
    ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.train import train as train_model
from v2.replay import (
    load_model, replay_validation, print_metrics,
    compute_baseline_random, compute_baseline_atm_always,
    compute_baseline_simple_rules, compute_baseline_atm_trailing,
)
from v2.core.policy import DecisionPolicy, DEFAULT_POLICY
from v2.core.metrics import score_config_fingerprint
from v2.ops.artifact import save_artifact


def run_experiment(
    data_path: str = "v2/data.pt",
    model_path: str = "v2/model.pt",
    experiment_id: str | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    save_artifacts: bool = True,
) -> dict:
    """Run a complete experiment: train, replay, score, compare baselines.

    Returns a results dict with all metrics.
    """
    if experiment_id is None:
        experiment_id = f"exp_{int(time.time())}"

    results = {
        "experiment_id": experiment_id,
        "status": "running",
        "score": -999.0,
    }

    # --- Phase 1: Train ---
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {experiment_id}")
    print(f"{'='*60}")
    print(f"\n--- TRAINING ---")

    t_train_start = time.time()
    try:
        model, train_metrics = train_model(data_path=data_path, model_path=model_path)
        training_seconds = time.time() - t_train_start
        print(f"Training completed in {training_seconds:.1f}s")
    except Exception as e:
        results["status"] = "crash"
        results["error"] = str(e)
        _print_results(results)
        return results

    # --- Phase 2: Replay on promote_mask ---
    print(f"\n--- REPLAY (promote_mask) ---")

    data = torch.load(data_path, map_location="cpu", weights_only=False)

    # Check for promote_mask (new 4-way split)
    mask_key = "promote_mask"
    if mask_key not in data:
        print(f"WARNING: {mask_key} not in dataset, falling back to val_mask")
        mask_key = "val_mask"

    model = load_model(model_path)
    metrics, trades = replay_validation(
        model, data, mask_key=mask_key, policy=policy,
    )
    print_metrics("Model", metrics)

    # --- Phase 3: Baselines ---
    print(f"\n--- BASELINES (on {mask_key}) ---")
    b_random = compute_baseline_random(data, mask_key=mask_key, policy=policy)
    b_atm = compute_baseline_atm_always(data, mask_key=mask_key, policy=policy)
    b_rules = compute_baseline_simple_rules(data, mask_key=mask_key, policy=policy)
    b_trailing = compute_baseline_atm_trailing(data, mask_key=mask_key, policy=policy)

    beats_random = metrics.score > b_random.score
    beats_atm = metrics.score > b_atm.score
    beats_rules = metrics.score > b_rules.score
    beats_trailing = metrics.score > b_trailing.score

    # --- Phase 4: Save artifact ---
    dataset_fp = data.get('metadata', {}).get('fingerprint', 'unknown')
    artifact_dir = None
    if save_artifacts:
        try:
            artifact_dir = save_artifact(
                experiment_id=experiment_id,
                model_path=model_path,
                score=metrics.score,
                policy=policy,
                dataset_fingerprint=dataset_fp,
                extra_metadata={
                    "training_seconds": training_seconds,
                    "train_metrics": train_metrics,
                },
            )
            print(f"\nArtifact saved: {artifact_dir}")
        except Exception as e:
            print(f"\nWARNING: Failed to save artifact: {e}")

    # --- Build results ---
    results = {
        "experiment_id": experiment_id,
        "status": "complete",
        "score": metrics.score,
        "daily_sortino": metrics.daily_sortino,
        "positive_day_rate": metrics.positive_day_rate,
        "max_account_drawdown": metrics.max_account_drawdown,
        "net_pnl_dollars": metrics.net_pnl_dollars,
        "total_trades": metrics.total_trades,
        "traded_days": metrics.traded_days,
        "profit_factor": metrics.profit_factor,
        "win_rate": metrics.win_rate,
        "trades_per_day": metrics.trades_per_day,
        "sharpe": metrics.sharpe,
        "gate_failure": metrics.gate_failure,
        "call_count": metrics.call_count,
        "put_count": metrics.put_count,
        "beats_random": beats_random,
        "beats_atm": beats_atm,
        "beats_rules": beats_rules,
        "beats_trailing": beats_trailing,
        "baseline_random_score": b_random.score,
        "baseline_atm_score": b_atm.score,
        "baseline_rules_score": b_rules.score,
        "baseline_trailing_score": b_trailing.score,
        "training_seconds": training_seconds,
        "dataset_fingerprint": dataset_fp,
        "score_fingerprint": score_config_fingerprint(),
        "policy_fingerprint": policy.fingerprint(),
        "artifact_dir": artifact_dir,
    }

    _print_results(results)
    return results


def _print_results(results: dict) -> None:
    """Print results in the Karpathy-style structured format."""
    print(f"\n---")
    for key in [
        "score", "daily_sortino", "positive_day_rate", "max_account_drawdown",
        "net_pnl_dollars", "total_trades", "traded_days", "profit_factor",
        "win_rate", "trades_per_day", "beats_random", "beats_atm", "beats_rules", "beats_trailing",
        "training_seconds", "status",
    ]:
        if key in results:
            val = results[key]
            if isinstance(val, float):
                print(f"{key + ':':<26}{val:.6f}")
            elif isinstance(val, bool):
                print(f"{key + ':':<26}{'true' if val else 'false'}")
            else:
                print(f"{key + ':':<26}{val}")

    if results.get("gate_failure"):
        print(f"{'gate_failure:':<26}{results['gate_failure']}")
    if results.get("error"):
        print(f"{'error:':<26}{results['error']}")

    # JSON for machine parsing
    print(f"\nRESULTS_JSON:{json.dumps(results, default=str)}")


def main():
    parser = argparse.ArgumentParser(description="Run a single ART2 experiment")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--model", type=str, default="v2/model.pt")
    parser.add_argument("--id", type=str, default=None, help="Experiment ID")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="Skip saving artifact bundle")
    args = parser.parse_args()

    run_experiment(
        data_path=args.data,
        model_path=args.model,
        experiment_id=args.id,
        save_artifacts=not args.no_artifacts,
    )


if __name__ == "__main__":
    main()
