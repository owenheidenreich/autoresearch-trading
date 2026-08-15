"""Protocol101 canonical-window random null/canary recalibration.

This computes no-skill serial replay bands for all seven menu-v2 policies from
the governed historical corpus. It intentionally performs no model training,
threshold tuning, promotion, broker calls, or sealed-data reads.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from v4.scripts.run_protocol101_stage1_experiment import (
    FEE_PER_TRADE,
    extract_table,
    replay,
    subset,
)
from v4.scripts.protocol101_training_scope import load_training_scope


SCHEMA_VERSION = "Protocol101CanonicalV1NullCanaryV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_canonical_v1_null_canary_training_scope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def one_random_per_minute(table: Any, rng: np.random.Generator) -> np.ndarray:
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (session, decision_time) in enumerate(zip(table.session_name, table.decision_time)):
        groups.setdefault((str(session), str(decision_time)), []).append(i)
    return np.asarray(sorted(rng.choice(indices) for indices in groups.values()), dtype=int)


def oracle_per_minute(table: Any, floor: float) -> np.ndarray:
    best: dict[tuple[str, str], int] = {}
    for i, (session, decision_time) in enumerate(zip(table.session_name, table.decision_time)):
        value = float(table.pnl[i])
        if value <= floor:
            continue
        key = (str(session), str(decision_time))
        if key not in best or value > float(table.pnl[best[key]]):
            best[key] = i
    return np.asarray(sorted(best.values()), dtype=int)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"count": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_scope = load_training_scope()
    sessions = training_scope.sessions
    table = extract_table(sessions)
    folds = training_scope.folds
    session_arr = np.asarray(table.session_name)
    rng = np.random.default_rng(args.seed)
    policy_results: dict[str, Any] = {}

    for policy_idx in range(7):
        policy_mask = table.policy_idx == policy_idx
        fold_rows = []
        random_pooled_pnls: list[float] = []
        random_pooled_drawdowns: list[float] = []
        for fold in folds:
            fold_mask = policy_mask & np.isin(session_arr, fold["test_sessions"])
            test = subset(table, fold_mask)
            run_rows = []
            for run_idx in range(args.random_runs):
                selected = one_random_per_minute(test, rng)
                sim = replay(test, selected, FEE_PER_TRADE, f"policy{policy_idx}_fold{fold['fold']}_random{run_idx}")
                run_rows.append(sim)
                random_pooled_pnls.append(float(sim["net_pnl"]))
                random_pooled_drawdowns.append(float(sim["max_drawdown"]))
            oracle = replay(test, oracle_per_minute(test, FEE_PER_TRADE), FEE_PER_TRADE, f"policy{policy_idx}_fold{fold['fold']}_oracle")
            fold_rows.append(
                {
                    "fold": fold["fold"],
                    "test_session_count": len(fold["test_sessions"]),
                    "random_net_pnl": summarize([float(row["net_pnl"]) for row in run_rows]),
                    "random_max_drawdown": summarize([float(row["max_drawdown"]) for row in run_rows]),
                    "random_trades_per_day": summarize([float(row["trades_per_day"]) for row in run_rows]),
                    "oracle": oracle,
                }
            )
        policy_results[str(policy_idx)] = {
            "folds": fold_rows,
            "pooled_random_net_pnl": summarize(random_pooled_pnls),
            "pooled_random_max_drawdown": summarize(random_pooled_drawdowns),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "scope": "offline_random_nulls_no_model_training_no_sealed_data",
        "sessions": len(sessions),
        "candidate_rows": int(len(table.pnl)),
        "design_path": str(training_scope.design_path),
        "manifest_path": str(training_scope.manifest_path),
        "acceptance_registry_path": str(training_scope.acceptance_registry_path),
        "acceptance_registry_hash": training_scope.acceptance_registry_hash,
        "fold_governance_hash": training_scope.fold_governance_hash,
        "random_runs_per_fold_policy": int(args.random_runs),
        "fee_overlay_dollars": float(FEE_PER_TRADE),
        "policy_results": policy_results,
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
            "sealed_market_data_read": False,
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    report = [
        "# Protocol101 Canonical V1 Null/Canary Recalibration",
        "",
        "Offline no-skill random bands for all seven menu-v2 policies. No model training or threshold tuning occurred.",
        "",
        f"- Sessions: `{summary['sessions']}`",
        f"- Candidate rows: `{summary['candidate_rows']}`",
        f"- Random runs per fold/policy: `{summary['random_runs_per_fold_policy']}`",
        "",
        "| Policy | Random pooled PnL p50 | Random pooled PnL p95 | Random DD p95 | Oracle pooled PnL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for policy_idx, result in policy_results.items():
        oracle_total = sum(float(row["oracle"]["net_pnl"]) for row in result["folds"])
        report.append(
            f"| {policy_idx} | ${result['pooled_random_net_pnl']['p50']:,.0f} | "
            f"${result['pooled_random_net_pnl']['p95']:,.0f} | "
            f"${result['pooled_random_max_drawdown']['p95']:,.0f} | "
            f"${oracle_total:,.0f} |"
        )
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
