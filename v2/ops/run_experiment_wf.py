"""Run a single ART² experiment using walk-forward cross-validation.

Drop-in replacement for run_experiment.py. Same CLI, same output format.
Internally runs 5 walk-forward folds instead of a single train+replay.

Usage:
    python -m v2.ops.run_experiment_wf [--data v2/data.pt] [--id exp_001]

Output: same structured key:value lines + RESULTS_JSON as run_experiment.py,
plus per-fold breakdown.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.core.walkforward import run_walkforward
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
    """Run walk-forward CV experiment."""
    if experiment_id is None:
        experiment_id = f"exp_{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {experiment_id}")
    print(f"{'='*60}")

    try:
        wf = run_walkforward(
            data_path=data_path,
            model_path=model_path,
            policy=policy,
        )
    except Exception as e:
        results = {
            "experiment_id": experiment_id,
            "status": "crash",
            "score": -999.0,
            "error": str(e),
        }
        _print_results(results)
        return results

    # Use last fold's metrics for detailed reporting (most training data)
    last_fold = wf.fold_results[-1] if wf.fold_results else None
    m = last_fold.metrics if last_fold else None

    # Save artifact (last fold's model is already at model_path)
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    dataset_fp = data.get('metadata', {}).get('fingerprint', 'unknown')
    artifact_dir = None
    if save_artifacts and os.path.exists(model_path):
        try:
            artifact_dir = save_artifact(
                experiment_id=experiment_id,
                model_path=model_path,
                score=wf.aggregate_score,
                policy=policy,
                dataset_fingerprint=dataset_fp,
                extra_metadata={
                    "walk_forward": True,
                    "n_folds": len(wf.fold_results),
                    "per_fold_scores": wf.per_fold_scores,
                    "min_fold_score": wf.min_fold_score,
                    "std_fold_score": wf.std_fold_score,
                    "training_seconds": wf.training_seconds,
                },
            )
            print(f"\nArtifact saved: {artifact_dir}")
        except Exception as e:
            print(f"\nWARNING: Failed to save artifact: {e}")

    results = {
        "experiment_id": experiment_id,
        "status": "complete",
        "score": wf.aggregate_score,
        "min_fold_score": wf.min_fold_score,
        "max_fold_score": wf.max_fold_score,
        "std_fold_score": wf.std_fold_score,
        "per_fold_scores": wf.per_fold_scores,
        "daily_sortino": m.daily_sortino if m else 0,
        "positive_day_rate": m.positive_day_rate if m else 0,
        "max_account_drawdown": m.max_account_drawdown if m else 0,
        "net_pnl_dollars": m.net_pnl_dollars if m else 0,
        "total_trades": wf.total_trades,
        "traded_days": wf.total_test_days,
        "profit_factor": m.profit_factor if m else 0,
        "win_rate": m.win_rate if m else 0,
        "trades_per_day": wf.total_trades / max(wf.total_test_days, 1),
        "sharpe": m.sharpe if m else 0,
        "gate_failure": m.gate_failure if m else None,
        "call_count": m.call_count if m else 0,
        "put_count": m.put_count if m else 0,
        "beats_random": wf.aggregate_score > wf.aggregate_baselines.get("random", 999),
        "beats_atm": wf.aggregate_score > wf.aggregate_baselines.get("atm", 999),
        "beats_rules": wf.aggregate_score > wf.aggregate_baselines.get("rules", 999),
        "beats_trailing": wf.aggregate_score > wf.aggregate_baselines.get("trailing", 999),
        "baseline_random_score": wf.aggregate_baselines.get("random", -999),
        "baseline_atm_score": wf.aggregate_baselines.get("atm", -999),
        "baseline_rules_score": wf.aggregate_baselines.get("rules", -999),
        "baseline_trailing_score": wf.aggregate_baselines.get("trailing", -999),
        "training_seconds": wf.training_seconds,
        "dataset_fingerprint": dataset_fp,
        "score_fingerprint": score_config_fingerprint(),
        "policy_fingerprint": policy.fingerprint(),
        "artifact_dir": artifact_dir,
        "walk_forward": True,
        "n_folds": len(wf.fold_results),
    }

    _print_results(results)
    return results


def _print_results(results: dict) -> None:
    """Print results in structured format (same as run_experiment.py)."""
    print(f"\n---")
    for key in [
        "score", "min_fold_score", "max_fold_score", "std_fold_score",
        "daily_sortino", "positive_day_rate", "max_account_drawdown",
        "net_pnl_dollars", "total_trades", "traded_days", "profit_factor",
        "win_rate", "trades_per_day", "beats_random", "beats_atm",
        "beats_rules", "beats_trailing", "training_seconds", "status",
    ]:
        if key in results:
            val = results[key]
            if isinstance(val, float):
                print(f"{key + ':':<26}{val:.6f}")
            elif isinstance(val, bool):
                print(f"{key + ':':<26}{'true' if val else 'false'}")
            else:
                print(f"{key + ':':<26}{val}")

    if results.get("per_fold_scores"):
        scores_str = ", ".join(f"{s:.3f}" for s in results["per_fold_scores"])
        print(f"{'per_fold_scores:':<26}[{scores_str}]")

    if results.get("gate_failure"):
        print(f"{'gate_failure:':<26}{results['gate_failure']}")
    if results.get("error"):
        print(f"{'error:':<26}{results['error']}")

    print(f"\nRESULTS_JSON:{json.dumps(results, default=str)}")


def main():
    parser = argparse.ArgumentParser(description="Run ART2 walk-forward experiment")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--model", type=str, default="v2/model.pt")
    parser.add_argument("--id", type=str, default=None, help="Experiment ID")
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    run_experiment(
        data_path=args.data,
        model_path=args.model,
        experiment_id=args.id,
        save_artifacts=not args.no_artifacts,
    )


if __name__ == "__main__":
    main()
