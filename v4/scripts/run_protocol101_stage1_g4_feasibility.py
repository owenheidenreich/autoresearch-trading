"""Protocol101 Stage-1 G4 drawdown feasibility packet.

This is an offline measurement script. It does not train a model, select a
threshold, touch broker/runtime state, or inspect sealed recorder data. It
extends the historical print-only diagnostic into a governed artifact packet
for owner review before Stage-1 gate-graded training.
"""
from __future__ import annotations

import argparse
import json
import math
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


SCHEMA_VERSION = "Protocol101Stage1G4FeasibilityV2"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_stage1_g4_feasibility_training_scope_v2")
STARTING_CASH = 10_000.0
CURRENT_G4_DRAWDOWN_PCT = 0.25
FEE_OVERLAY_DOLLARS = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def per_minute_pick(tbl: Any, values: np.ndarray, floor_dollars: float) -> np.ndarray:
    selected: dict[tuple[str, str], int] = {}
    for i, (session, decision_time) in enumerate(zip(tbl.session_name, tbl.decision_time)):
        dollar_value = float(values[i])
        if not math.isfinite(dollar_value) or dollar_value <= floor_dollars:
            continue
        key = (str(session), str(decision_time))
        if key not in selected or dollar_value > float(values[selected[key]]):
            selected[key] = i
    return np.asarray(sorted(selected.values()), dtype=int)


def run_feasibility(seed: int) -> dict[str, Any]:
    training_scope = load_training_scope()
    sessions = training_scope.sessions
    table = extract_table(sessions)
    folds = training_scope.folds
    session_arr = np.asarray(table.session_name)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for fold in folds:
        test = subset(table, np.isin(session_arr, fold["test_sessions"]))
        minutes: dict[tuple[str, str], list[int]] = {}
        for i, (session, decision_time) in enumerate(zip(test.session_name, test.decision_time)):
            minutes.setdefault((str(session), str(decision_time)), []).append(i)
        random_selection = np.asarray(sorted(rng.choice(indices) for indices in minutes.values()))
        oracle_selection = per_minute_pick(test, test.pnl, FEE_PER_TRADE)
        cheap_values = np.where(test.entry_ask <= 3.0, test.pnl, -1e9)
        cheap_oracle_selection = per_minute_pick(test, cheap_values, FEE_PER_TRADE)
        fold_results = {
            "random": replay(test, random_selection, FEE_PER_TRADE, "random"),
            "oracle": replay(test, oracle_selection, FEE_PER_TRADE, "oracle"),
            "cheap_oracle": replay(test, cheap_oracle_selection, FEE_PER_TRADE, "cheap_oracle"),
        }
        rows.append(
            {
                "fold": fold["fold"],
                "test_session_count": len(fold["test_sessions"]),
                "test_first_session": min(fold["test_sessions"]) if fold["test_sessions"] else None,
                "test_last_session": max(fold["test_sessions"]) if fold["test_sessions"] else None,
                "results": {
                    name: {
                        "max_drawdown": float(result["max_drawdown"]),
                        "net_pnl": float(result["net_pnl"]),
                        "trade_count": int(result.get("trades", result.get("trade_count", 0))),
                        "drawdown_pct_of_starting_cash": float(result["max_drawdown"]) / STARTING_CASH,
                    }
                    for name, result in fold_results.items()
                },
            }
        )

    strategy_names = ["random", "oracle", "cheap_oracle"]
    aggregates = {}
    for name in strategy_names:
        drawdowns = [float(row["results"][name]["max_drawdown"]) for row in rows]
        pnls = [float(row["results"][name]["net_pnl"]) for row in rows]
        aggregates[name] = {
            "fold_count": len(drawdowns),
            "drawdown_mean": float(np.mean(drawdowns)) if drawdowns else None,
            "drawdown_max": float(np.max(drawdowns)) if drawdowns else None,
            "drawdown_p95": percentile(drawdowns, 95),
            "net_pnl_mean": float(np.mean(pnls)) if pnls else None,
            "net_pnl_min": float(np.min(pnls)) if pnls else None,
            "net_pnl_max": float(np.max(pnls)) if pnls else None,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "offline_governed_training_sessions_only_no_sealed_data",
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
        "inputs": {
            "governed_session_role": "training_scope_expanding_folds",
            "session_count": len(sessions),
            "candidate_rows": int(len(table.pnl)),
            "fee_overlay_dollars": FEE_OVERLAY_DOLLARS,
            "random_seed": int(seed),
            "simulator_source": "v4.scripts.run_protocol101_stage1_experiment.replay",
            "design_path": str(training_scope.design_path),
            "manifest_path": str(training_scope.manifest_path),
            "acceptance_registry_path": str(training_scope.acceptance_registry_path),
            "acceptance_registry_hash": training_scope.acceptance_registry_hash,
            "fold_governance_hash": training_scope.fold_governance_hash,
        },
        "current_g4": {
            "drawdown_cap_pct_of_peak_equity": CURRENT_G4_DRAWDOWN_PCT,
            "nominal_cap_on_10k_starting_cash": STARTING_CASH * CURRENT_G4_DRAWDOWN_PCT,
        },
        "folds": rows,
        "aggregates": aggregates,
        "interpretation": {
            "random_noise_floor_exceeds_current_relative_cap": (
                aggregates["random"]["drawdown_mean"] or 0.0
            )
            > STARTING_CASH * CURRENT_G4_DRAWDOWN_PCT,
            "oracle_zero_drawdown_observed": aggregates["oracle"]["drawdown_max"] == 0.0,
            "holdout_absolute_1500_cap_is_stale": True,
            "owner_review_required_before_gate_graded_training": True,
        },
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = summary["folds"]
    lines = [
        "# Protocol101 Stage-1 G4 Feasibility V2",
        "",
        "Offline measurement only. No model training, threshold selection, broker calls, paid downloads, paper-submit, launchd/runtime edits, promotion/default changes, real-money paths, or sealed market-data reads occurred.",
        "",
        "## Result",
        "",
        f"- Governed sessions: `{summary['inputs']['session_count']}`",
        f"- Candidate rows: `{summary['inputs']['candidate_rows']}`",
        f"- Current G4 nominal 10k cap: `${summary['current_g4']['nominal_cap_on_10k_starting_cash']:,.0f}`",
        f"- Random mean drawdown: `${summary['aggregates']['random']['drawdown_mean']:,.0f}`",
        f"- Random max drawdown: `${summary['aggregates']['random']['drawdown_max']:,.0f}`",
        f"- Oracle max drawdown: `${summary['aggregates']['oracle']['drawdown_max']:,.0f}`",
        f"- Cheap-oracle max drawdown: `${summary['aggregates']['cheap_oracle']['drawdown_max']:,.0f}`",
        "",
        "## Fold Table",
        "",
        "| Fold | Random DD | Random PnL | Oracle DD | Oracle PnL | Cheap Oracle DD | Cheap Oracle PnL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        random_result = row["results"]["random"]
        oracle_result = row["results"]["oracle"]
        cheap_result = row["results"]["cheap_oracle"]
        lines.append(
            f"| {row['fold']} | ${random_result['max_drawdown']:,.0f} | ${random_result['net_pnl']:,.0f} | "
            f"${oracle_result['max_drawdown']:,.0f} | ${oracle_result['net_pnl']:,.0f} | "
            f"${cheap_result['max_drawdown']:,.0f} | ${cheap_result['net_pnl']:,.0f} |"
        )
    lines += [
        "",
        "## Draft Owner Decision",
        "",
        "The stale holdout absolute `max DD <= $1,500` line should be replaced before any holdout burn. A consistent owner-signature draft is provided in `proposed_gates_revision.md`.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_proposed_revision(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Proposed Stage-1 G4 / Holdout Revision

Status: **DRAFT FOR OWNER SIGNATURE**.

This draft is generated from `{SCHEMA_VERSION}`. It does not change the signed
gate document by itself.

## Evidence

- Current G4 nominal cap on $10k starting cash: `${summary['current_g4']['nominal_cap_on_10k_starting_cash']:,.0f}`.
- Random-policy mean drawdown across folds: `${summary['aggregates']['random']['drawdown_mean']:,.0f}`.
- Random-policy max drawdown across folds: `${summary['aggregates']['random']['drawdown_max']:,.0f}`.
- Synthetic oracle max drawdown: `${summary['aggregates']['oracle']['drawdown_max']:,.0f}`.
- Cheap synthetic oracle max drawdown: `${summary['aggregates']['cheap_oracle']['drawdown_max']:,.0f}`.

## Proposed Text Replacement

Replace the holdout sentence:

> fee-adjusted PnL > 0, max DD <= $1,500, and result within the 90% bootstrap CI implied by CV

with:

> fee-adjusted PnL > 0; max strict-serial drawdown must satisfy the owner-signed
> G4 rule active for this candidate generation, with the same fee/stress and
> one-account semantics used in CV; and the result must fall within the 90%
> bootstrap CI implied by CV. A holdout result wildly above expectations remains
> an audit trigger, not a celebration.

## Rationale

The `$1,500` absolute drawdown cap is stale. It conflicts with the already
signed relative G4 direction and would make the protected holdout auto-fail
otherwise valid candidates for a retired rule.
"""
    (out_dir / "proposed_gates_revision.md").write_text(text)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_feasibility(args.seed)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir, summary)
    write_proposed_revision(out_dir, summary)


if __name__ == "__main__":
    main()
