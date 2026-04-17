"""Run a walk-forward CV experiment and emit a CVReport.

This module produces CV_EVAL artifacts only. It does NOT produce a deployable
model. A CV_EVAL artifact cannot be promoted to v2/models/model.pt — promotion
requires a separate v2.ops.run_final_train pass.

Usage:
    python -m v2.ops.run_experiment_wf --id exp_NNN --screen-mode {latest,mini,full}

See /Users/gduby/.claude/plans/delightful-yawning-tiger.md Phase 1 + Appendix E.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.core.artifact_kind import ArtifactKind
from v2.core.cv_report import CVReport
from v2.core.policy import DecisionPolicy
from v2.core.walkforward import run_walkforward, SCREENING_MODES
from v2.ops.artifact import save_cv_eval_artifact


def run_experiment(
    data_path: str = "v2/data.pt",
    experiment_id: str | None = None,
    screening_mode: str = "full",
    fold_indices: list[int] | None = None,
    policy: DecisionPolicy | None = None,
    save_artifacts: bool = True,
) -> dict:
    """Run walk-forward CV and emit a CVReport.

    Returns a flat result dict for deploy.sh to parse. The authoritative output
    is the saved CVReport JSON; the dict is a scannable summary only.
    """
    if policy is None:
        _side_mode = os.environ.get("SIDE_MODE", "off")
        _alpha_side = float(os.environ.get("ALPHA_SIDE", "0.0"))
        policy = DecisionPolicy(side_mode=_side_mode, alpha_side=_alpha_side)
    if experiment_id is None:
        experiment_id = f"exp_{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {experiment_id}  mode={screening_mode}  "
          f"fold_indices={fold_indices}")
    print(f"{'='*60}")

    try:
        cv: CVReport = run_walkforward(
            data_path=data_path,
            experiment_id=experiment_id,
            screening_mode=screening_mode,
            fold_indices=fold_indices,
            policy=policy,
        )
    except Exception as e:
        results = {
            "experiment_id": experiment_id,
            "screening_mode": screening_mode,
            "status": "crash",
            "stability_score": -999.0,
            "error": str(e),
        }
        _print_results(results)
        return results

    artifact_dir = None
    if save_artifacts:
        try:
            artifact_dir = save_cv_eval_artifact(
                experiment_id=experiment_id,
                cv_report=cv,
                policy=policy,
                kind=ArtifactKind.CV_EVAL,
            )
            print(f"\nCV_EVAL artifact saved: {artifact_dir}")
        except Exception as e:
            print(f"\nWARNING: Failed to save CV_EVAL artifact: {e}")

    results = _flatten_cv_for_deploy(cv, artifact_dir)
    _print_results(results)
    return results


def _flatten_cv_for_deploy(cv: CVReport, artifact_dir: str | None) -> dict:
    """Scope-named flat dict for deploy.sh. No blended fields.

    Every numeric field is namespaced by scope: `stability_*`, `pooled_*`, or
    `latest_fold_*`. A consumer cannot accidentally treat a fold metric as an
    aggregate or vice versa.
    """
    latest_fold = cv.folds[-1] if cv.folds else None

    return {
        "experiment_id": cv.experiment_id,
        "screening_mode": cv.screening_mode,
        "status": "complete",
        # Stability (distribution of per-fold scores)
        "stability_score": cv.stability.mean_fold_score,
        "stability_min": cv.stability.min_fold_score,
        "stability_max": cv.stability.max_fold_score,
        "stability_std": cv.stability.std_fold_score,
        "per_fold_scores": cv.stability.per_fold_scores,
        "per_fold_gate_failures": cv.stability.per_fold_gate_failures,
        "any_fold_gate_failure": cv.stability.any_fold_gate_failure,
        # Pooled economics across all evaluated folds
        "pooled_profit_factor": cv.pooled.profit_factor,
        "pooled_max_account_drawdown": cv.pooled.max_account_drawdown,
        "pooled_win_rate": cv.pooled.win_rate,
        "pooled_call_pct": cv.pooled.call_pct,
        "pooled_put_pct": cv.pooled.put_pct,
        "pooled_net_pnl_dollars": cv.pooled.net_pnl_dollars,
        "pooled_total_trades": cv.pooled.total_trades,
        "pooled_total_eval_days": cv.pooled.total_eval_days,
        "pooled_traded_days": cv.pooled.traded_days,
        "pooled_positive_day_rate": cv.pooled.positive_day_rate,
        "pooled_daily_sortino": cv.pooled.daily_sortino,
        # Latest evaluated fold (diagnostic only — NEVER treated as the summary)
        "latest_fold_idx": latest_fold.fold_idx if latest_fold else -1,
        "latest_fold_window_id": latest_fold.window_id if latest_fold else "",
        "latest_fold_score": latest_fold.score if latest_fold else 0.0,
        "latest_fold_gate_failure": (latest_fold.gate_failure if latest_fold else None),
        # Baseline comparison (uses stability_score)
        "beats_all_baselines": cv.beats_all_baselines,
        "baseline_scores": cv.aggregate_baselines,
        # Provenance
        "dataset_fingerprint": cv.dataset_fingerprint,
        "evaluator_fingerprint": cv.evaluator_fingerprint,
        "policy_fingerprint": cv.policy_fingerprint,
        "training_seconds": cv.training_seconds,
        "artifact_dir": artifact_dir,
        "artifact_kind": ArtifactKind.CV_EVAL.value,
        "schema_version": cv.schema_version,
    }


def _print_results(results: dict) -> None:
    """Human-readable summary plus RESULTS_JSON line consumed by deploy.sh."""
    print(f"\n---")
    scalar_keys = [
        "experiment_id", "screening_mode", "status", "artifact_kind",
        "stability_score", "stability_min", "stability_max", "stability_std",
        "pooled_profit_factor", "pooled_max_account_drawdown",
        "pooled_win_rate", "pooled_net_pnl_dollars",
        "pooled_total_trades", "pooled_traded_days", "pooled_total_eval_days",
        "pooled_positive_day_rate", "pooled_daily_sortino",
        "latest_fold_idx", "latest_fold_score", "latest_fold_gate_failure",
        "any_fold_gate_failure", "beats_all_baselines",
        "training_seconds",
    ]
    for key in scalar_keys:
        if key not in results:
            continue
        val = results[key]
        if isinstance(val, bool):
            print(f"{key + ':':<30}{'true' if val else 'false'}")
        elif isinstance(val, float):
            print(f"{key + ':':<30}{val:.6f}")
        else:
            print(f"{key + ':':<30}{val}")

    if "per_fold_scores" in results and results["per_fold_scores"]:
        scores_str = ", ".join(f"{s:.3f}" for s in results["per_fold_scores"])
        print(f"{'per_fold_scores:':<30}[{scores_str}]")
    if "per_fold_gate_failures" in results and results["per_fold_gate_failures"]:
        flags = ", ".join(str(b).lower() for b in results["per_fold_gate_failures"])
        print(f"{'per_fold_gate_failures:':<30}[{flags}]")

    if results.get("error"):
        print(f"{'error:':<30}{results['error']}")

    print(f"\nRESULTS_JSON:{json.dumps(results, default=str)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ART2 walk-forward CV experiment. Emits CV_EVAL artifact.",
    )
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--id", type=str, default=None, help="Experiment ID")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument(
        "--screen-mode",
        choices=sorted(SCREENING_MODES),
        default="full",
        help=(
            "Screening mode. 'full' is the only mode that produces a canonical "
            "CV_EVAL artifact suitable for run_final_train. "
            + "; ".join(f"{k}: {v}" for k, v in SCREENING_MODES.items())
        ),
    )
    parser.add_argument(
        "--fold-indices",
        type=str,
        default=None,
        help="Explicit comma-separated fold subset (e.g. '0,2,4'). Overrides --screen-mode.",
    )
    args = parser.parse_args()

    fold_indices = None
    if args.fold_indices:
        fold_indices = [int(x.strip()) for x in args.fold_indices.split(",") if x.strip()]

    run_experiment(
        data_path=args.data,
        experiment_id=args.id,
        screening_mode=args.screen_mode,
        fold_indices=fold_indices,
        save_artifacts=not args.no_artifacts,
    )


if __name__ == "__main__":
    main()
