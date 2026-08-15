"""Protocol 096: validation-selected blend of profit and slot scorers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.serial_opportunity import (
    ENTRY_FEATURE_COLUMNS,
    SerialOpportunityConfig,
    predict_scores,
    serial_simulate_candidates,
    strict_serial_baseline,
)
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS
from v4.scripts.run_protocol094_opportunity_threshold import (
    _aggregate_gate,
    _compare_to_protocol092,
    _json_dumps,
    _load_protocol092_artifact,
    select_opportunity_threshold,
)


DEFAULT_PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
DEFAULT_PROTOCOL095_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_095_slot_occupancy_target")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_096_profit_slot_blend")
MODEL_SEEDS = [1, 2, 3, 4, 5]
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--protocol095-dir", type=Path, default=DEFAULT_PROTOCOL095_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.protocol092_dir / "serial_opportunity_dataset.parquet")
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True)
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True)
    config = SerialOpportunityConfig(min_validation_trades=int(args.min_validation_trades))

    fold_results: list[dict[str, Any]] = []
    trade_ledgers: list[dict[str, Any]] = []
    for fold in FOLDS:
        validation_all = dataset[dataset["split"] == fold["validation_split"]].copy()
        for seed in args.seeds:
            profit_model, profit_scaler, _ = _load_protocol092_artifact(args.protocol092_dir, fold["name"], int(seed))
            slot_model, slot_scaler, _ = _load_protocol092_artifact(args.protocol095_dir, fold["name"], int(seed))

            validation_seed = validation_all[validation_all["seed"] == int(seed)].copy()
            validation_seed["profit_score"] = predict_scores(profit_model, profit_scaler, validation_seed, target_scale=100.0)
            validation_seed["slot_score"] = predict_scores(slot_model, slot_scaler, validation_seed, target_scale=100.0)
            selection = select_blend_and_threshold(validation_seed, seed=int(seed), config=config)

            seed_result = {
                "fold": fold["name"],
                "seed": int(seed),
                "train_splits": fold["train_splits"],
                "validation_split": fold["validation_split"],
                "test_split": fold["test_split"],
                "alpha": float(selection["alpha"]),
                "threshold": float(selection["threshold"]),
                "selection": selection,
                "splits": {},
            }
            for split_name, scored in _score_reported_frames(
                dataset,
                fold,
                seed,
                profit_model,
                profit_scaler,
                slot_model,
                slot_scaler,
                float(selection["alpha"]),
            ).items():
                baseline_source = dataset[
                    (dataset["split"] == ("q1_2026" if split_name == "march_2026" else split_name))
                    & (dataset["seed"] == int(seed))
                ].copy()
                if split_name == "march_2026":
                    baseline_source = baseline_source[baseline_source["session"] >= "2026-03-01"].copy()
                model_base = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(selection["threshold"]),
                    slippage_per_side=0.0,
                    strategy=f"protocol096_{fold['name']}",
                )
                model_stress10 = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(selection["threshold"]),
                    slippage_per_side=0.10,
                    strategy=f"protocol096_{fold['name']}_stress10",
                )
                model_stress25 = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(selection["threshold"]),
                    slippage_per_side=0.25,
                    strategy=f"protocol096_{fold['name']}_stress25",
                )
                baseline_base = strict_serial_baseline(baseline_source, seed=int(seed), slippage_per_side=0.0)
                baseline_stress10 = strict_serial_baseline(baseline_source, seed=int(seed), slippage_per_side=0.10)
                seed_result["splits"][split_name] = {
                    "model": model_base.summary,
                    "model_stress_0_10": model_stress10.summary,
                    "model_stress_0_25": model_stress25.summary,
                    "strict_serial_baseline": baseline_base.summary,
                    "strict_serial_baseline_stress_0_10": baseline_stress10.summary,
                    "validation_threshold_source": fold["validation_split"],
                }
                for trade in model_base.trades:
                    row = dict(trade)
                    row["fold"] = fold["name"]
                    row["reported_split"] = split_name
                    row["alpha"] = float(selection["alpha"])
                    trade_ledgers.append(row)
            fold_results.append(seed_result)

    aggregate = _aggregate_gate(fold_results)
    protocol092_summary = json.loads((args.protocol092_dir / "summary.json").read_text())
    payload = {
        "protocol": "096_profit_slot_blend",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol092_dir": str(args.protocol092_dir),
        "source_protocol095_dir": str(args.protocol095_dir),
        "pre_registration": _pre_registration(),
        "feature_columns": ENTRY_FEATURE_COLUMNS,
        "fold_results": fold_results,
        "aggregate_gate": aggregate,
        "protocol092_comparison": _compare_to_protocol092(aggregate, protocol092_summary["aggregate_gate"]),
        "decision": _decision(aggregate, protocol092_summary["aggregate_gate"]),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    (args.out_dir / "serial_policy_trades.json").write_text(_json_dumps(trade_ledgers))
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "comparison": payload["protocol092_comparison"]}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def select_blend_and_threshold(frame: pd.DataFrame, *, seed: int, config: SerialOpportunityConfig) -> dict[str, Any]:
    candidates = []
    for alpha in ALPHAS:
        item = frame.copy()
        item["score"] = (1.0 - float(alpha)) * item["profit_score"] + float(alpha) * item["slot_score"]
        threshold = select_opportunity_threshold(item, seed=int(seed), config=config)
        selected = threshold["selected"]
        candidates.append(
            {
                "alpha": float(alpha),
                "threshold": float(threshold["threshold"]),
                "threshold_selection": threshold,
                "delta_vs_baseline_stress_0_10": float(selected["delta_vs_baseline_stress_0_10"]),
                "delta_vs_baseline": float(selected["delta_vs_baseline"]),
                "model_stress_0_10_total_pnl": float(selected["model_stress_0_10"]["total_pnl"]),
                "model_trades": int(selected["model"]["trades"]),
            }
        )
    best = max(candidates, key=_selection_key)
    return {
        "alpha": float(best["alpha"]),
        "threshold": float(best["threshold"]),
        "objective": "validation grid over fixed alphas maximizing stress_0_10 delta vs strict serial baseline",
        "alpha_grid": ALPHAS,
        "selected": best,
        "candidates": candidates,
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["delta_vs_baseline_stress_0_10"]),
        float(row["delta_vs_baseline"]),
        float(row["model_stress_0_10_total_pnl"]),
        float(row["model_trades"]),
    )


def _score_reported_frames(
    dataset,
    fold,
    seed,
    profit_model,
    profit_scaler,
    slot_model,
    slot_scaler,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    out = {}
    test = dataset[(dataset["split"] == fold["test_split"]) & (dataset["seed"] == int(seed))].copy()
    profit_score = predict_scores(profit_model, profit_scaler, test, target_scale=100.0)
    slot_score = predict_scores(slot_model, slot_scaler, test, target_scale=100.0)
    test["score"] = (1.0 - alpha) * profit_score + alpha * slot_score
    out[fold["test_split"]] = test
    if fold["test_split"] == "q1_2026":
        out["march_2026"] = test[test["session"] >= "2026-03-01"].copy()
    return out


def _decision(aggregate: dict[str, Any], previous: dict[str, Any]) -> str:
    if aggregate["promotion_ready"]:
        return "keep_promote_candidate: Protocol 096 clears the strict serial gate"
    comparison = _compare_to_protocol092(aggregate, previous)
    q3_improved = comparison["q3_2025"]["median_total_pnl_delta"] > 0
    q4_improved = comparison["q4_2025"]["median_total_pnl_delta"] > 0
    if q3_improved and q4_improved:
        return "keep_for_research_only: Protocol 096 improves Q3/Q4 but still does not clear promotion"
    return "reject: profit/slot blend did not improve both Q3 and Q4"


def _pre_registration() -> dict[str, Any]:
    return {
        "hypothesis": "Protocol 095 improved Q3 but over-penalized later regimes. Slot occupancy may be useful as an auxiliary score rather than the primary target.",
        "single_change": "validation-selected blend of fixed Protocol 092 profit scorer and fixed Protocol 095 slot scorer",
        "alpha_grid": ALPHAS,
        "selection": "alpha and threshold selected only on validation, maximizing stress_0_10 delta vs strict serial baseline",
        "features": "unchanged Protocol 092 entry-only feature set",
        "exits": "frozen Protocol 081 candidate exits",
        "paid_data": "forbidden",
        "live_orders": "forbidden",
        "folds": FOLDS,
        "seeds": MODEL_SEEDS,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 096: Profit/Slot Score Blend",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Single change: `{payload['pre_registration']['single_change']}`",
        "",
        "## Gate",
        "",
        _table(
            [
                {
                    "split": split,
                    "median_pnl": payload["aggregate_gate"][split]["median_total_pnl"],
                    "pf": payload["aggregate_gate"][split]["median_profit_factor"],
                    "trades": payload["aggregate_gate"][split]["median_trades"],
                    "stress10": payload["aggregate_gate"][split]["median_stress_0_10_total_pnl"],
                    "stress25": payload["aggregate_gate"][split]["median_stress_0_25_total_pnl"],
                    "baseline": payload["aggregate_gate"][split]["strict_serial_baseline_median_total_pnl"],
                    "beats_baseline": payload["aggregate_gate"][split]["beats_strict_serial_baseline"],
                    "vs_protocol092": payload["protocol092_comparison"][split]["median_total_pnl_delta"],
                }
                for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
            ],
            ["split", "median_pnl", "pf", "trades", "stress10", "stress25", "baseline", "beats_baseline", "vs_protocol092"],
        ),
        "",
        "## Selected Alphas",
        "",
        _table(
            [
                {
                    "fold": result["fold"],
                    "seed": result["seed"],
                    "validation": result["validation_split"],
                    "alpha": result["alpha"],
                    "threshold": result["threshold"],
                }
                for result in payload["fold_results"]
            ],
            ["fold", "seed", "validation", "alpha", "threshold"],
        ),
        "",
        "## Promotion Checks",
        "",
        _table(payload["aggregate_gate"]["promotion_checks"], ["split", "name", "value", "pass"]),
    ]
    path.write_text("\n".join(lines) + "\n")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
