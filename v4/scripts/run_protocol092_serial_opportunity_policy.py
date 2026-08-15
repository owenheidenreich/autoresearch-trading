"""Protocol 092: serial opportunity-cost neural entry policy.

This runner uses already-collected artifacts only. It freezes Protocol 081
candidate exits, trains an entry-time arbitration scorer, and evaluates the
single-position opportunity cost through chronological folds.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.serial_opportunity import (
    ENTRY_FEATURE_COLUMNS,
    SerialOpportunityConfig,
    build_protocol092_dataset,
    model_artifact_manifest,
    predict_scores,
    select_validation_threshold,
    serial_simulate_candidates,
    strict_serial_baseline,
    train_serial_opportunity_model,
)


DEFAULT_SELECTED = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/"
    "selected_trades_sequence_exits.json"
)
DEFAULT_LIFECYCLE_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_077_q4start_lifecycle_sequence_dataset/"
    "protocol054_lifecycle_trades.parquet"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
MODEL_SEEDS = [1, 2, 3, 4, 5]
FOLDS = [
    {
        "name": "fold1_train_q1_validate_q2_test_q3",
        "train_splits": ["q1_2025"],
        "validation_split": "q2_2025",
        "test_split": "q3_2025",
        "reported_splits": ["q3_2025"],
    },
    {
        "name": "fold2_train_q1_q2_validate_q3_test_q4",
        "train_splits": ["q1_2025", "q2_2025"],
        "validation_split": "q3_2025",
        "test_split": "q4_2025",
        "reported_splits": ["q4_2025"],
    },
    {
        "name": "fold3_train_q1_q2_q3_validate_q4_test_q1_2026",
        "train_splits": ["q1_2025", "q2_2025", "q3_2025"],
        "validation_split": "q4_2025",
        "test_split": "q1_2026",
        "reported_splits": ["q1_2026", "march_2026"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-trades", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--lifecycle-trades", type=Path, default=DEFAULT_LIFECYCLE_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SerialOpportunityConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        min_validation_trades=int(args.min_validation_trades),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_protocol092_dataset(
        selected_trades_path=args.selected_trades,
        lifecycle_trades_path=args.lifecycle_trades,
    )
    dataset_path = args.out_dir / "serial_opportunity_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)

    fold_results: list[dict[str, Any]] = []
    trade_ledgers: list[dict[str, Any]] = []
    baseline_ledgers: list[dict[str, Any]] = []
    for fold in FOLDS:
        train_frame = dataset[dataset["split"].isin(fold["train_splits"])].copy()
        validation_all = dataset[dataset["split"] == fold["validation_split"]].copy()
        for seed in args.seeds:
            validation_seed = validation_all[validation_all["seed"] == int(seed)].copy()
            model, scaler, history = train_serial_opportunity_model(
                train_frame,
                validation_all,
                seed=int(seed),
                config=config,
            )
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pt"
            scaler_path = model_dir / "scaler.json"
            torch.save(model.state_dict(), model_path)
            scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")

            validation_scored = validation_seed.copy()
            validation_scored["score"] = predict_scores(
                model,
                scaler,
                validation_scored,
                target_scale=config.target_scale,
            )
            threshold = select_validation_threshold(
                validation_scored,
                score_column="score",
                source_split=fold["validation_split"],
                source_seed=int(seed),
                config=config,
                stress_slippage_per_side=0.10,
            )
            manifest = model_artifact_manifest(
                fold_name=fold["name"],
                seed=int(seed),
                config=config,
                threshold=threshold,
                history=history,
                model_path=model_path,
                scaler_path=scaler_path,
            )
            (model_dir / "manifest.json").write_text(_json_dumps(manifest))
            (model_dir / "threshold_sweep.json").write_text(_json_dumps(threshold.sweep))

            scored_frames = _score_reported_frames(dataset, fold, seed, model, scaler, config)
            seed_result = {
                "fold": fold["name"],
                "seed": int(seed),
                "train_splits": fold["train_splits"],
                "validation_split": fold["validation_split"],
                "test_split": fold["test_split"],
                "threshold": float(threshold.threshold),
                "threshold_source_split": threshold.source_split,
                "threshold_source_rows": threshold.source_rows,
                "history_last": history[-1] if history else {},
                "splits": {},
            }
            for split_name, scored in scored_frames.items():
                baseline_source = dataset[
                    (dataset["split"] == ("q1_2026" if split_name == "march_2026" else split_name))
                    & (dataset["seed"] == int(seed))
                ].copy()
                if split_name == "march_2026":
                    baseline_source = baseline_source[baseline_source["session"] >= "2026-03-01"].copy()

                model_base = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(threshold.threshold),
                    slippage_per_side=0.0,
                    strategy=f"protocol092_{fold['name']}",
                )
                model_stress10 = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(threshold.threshold),
                    slippage_per_side=0.10,
                    strategy=f"protocol092_{fold['name']}_stress10",
                )
                model_stress25 = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(threshold.threshold),
                    slippage_per_side=0.25,
                    strategy=f"protocol092_{fold['name']}_stress25",
                )
                baseline_base = strict_serial_baseline(baseline_source, seed=int(seed), slippage_per_side=0.0)
                baseline_stress10 = strict_serial_baseline(baseline_source, seed=int(seed), slippage_per_side=0.10)

                split_result = {
                    "model": model_base.summary,
                    "model_stress_0_10": model_stress10.summary,
                    "model_stress_0_25": model_stress25.summary,
                    "strict_serial_baseline": baseline_base.summary,
                    "strict_serial_baseline_stress_0_10": baseline_stress10.summary,
                    "validation_threshold_source": threshold.source_split,
                }
                seed_result["splits"][split_name] = split_result
                for trade in model_base.trades:
                    row = dict(trade)
                    row["fold"] = fold["name"]
                    row["reported_split"] = split_name
                    trade_ledgers.append(row)
                for trade in baseline_base.trades:
                    row = dict(trade)
                    row["fold"] = fold["name"]
                    row["reported_split"] = split_name
                    baseline_ledgers.append(row)
            fold_results.append(seed_result)

    aggregate = _aggregate_gate(fold_results)
    payload = {
        "protocol": "092_serial_opportunity_policy",
        "paid_data_downloaded": False,
        "live_orders": False,
        "selected_trades_path": str(args.selected_trades),
        "lifecycle_trades_path": str(args.lifecycle_trades),
        "dataset_path": str(dataset_path),
        "pre_registration": _pre_registration(),
        "feature_version": "protocol092_serial_opportunity_v1",
        "feature_columns": ENTRY_FEATURE_COLUMNS,
        "config": asdict(config),
        "dataset_summary": _dataset_summary(dataset),
        "fold_results": fold_results,
        "aggregate_gate": aggregate,
        "decision": _decision(aggregate),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    (args.out_dir / "serial_policy_trades.json").write_text(_json_dumps(trade_ledgers))
    (args.out_dir / "strict_serial_baseline_trades.json").write_text(_json_dumps(baseline_ledgers))
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "aggregate_gate": aggregate}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def _score_reported_frames(dataset, fold, seed, model, scaler, config) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    test = dataset[(dataset["split"] == fold["test_split"]) & (dataset["seed"] == int(seed))].copy()
    test["score"] = predict_scores(model, scaler, test, target_scale=config.target_scale)
    out[fold["test_split"]] = test
    if fold["test_split"] == "q1_2026":
        march = test[test["session"] >= "2026-03-01"].copy()
        out["march_2026"] = march
    return out


def _aggregate_gate(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    scored_splits = ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
    aggregate: dict[str, Any] = {}
    for split in scored_splits:
        rows = []
        for result in fold_results:
            if split in result["splits"]:
                row = {"seed": result["seed"], "fold": result["fold"], **result["splits"][split]}
                rows.append(row)
        aggregate[split] = _summarize_split(rows)
    aggregate["promotion_checks"] = _promotion_checks(aggregate)
    aggregate["promotion_ready"] = bool(all(item["pass"] for item in aggregate["promotion_checks"]))
    return aggregate


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"seeds": 0}
    base_pnl = _arr(row["model"]["total_pnl"] for row in rows)
    base_pf = _arr(_finite_pf(row["model"]["profit_factor"]) for row in rows)
    stress10 = _arr(row["model_stress_0_10"]["total_pnl"] for row in rows)
    stress25 = _arr(row["model_stress_0_25"]["total_pnl"] for row in rows)
    baseline = _arr(row["strict_serial_baseline"]["total_pnl"] for row in rows)
    trades = _arr(row["model"]["trades"] for row in rows)
    side_calls = int(sum(row["model"]["side_counts"].get("C", 0) for row in rows))
    side_puts = int(sum(row["model"]["side_counts"].get("P", 0) for row in rows))
    return {
        "seeds": int(len(rows)),
        "median_total_pnl": float(np.median(base_pnl)),
        "positive_seed_fraction": float((base_pnl > 0).mean()),
        "median_profit_factor": float(np.median(base_pf)),
        "median_trades": float(np.median(trades)),
        "median_stress_0_10_total_pnl": float(np.median(stress10)),
        "median_stress_0_25_total_pnl": float(np.median(stress25)),
        "strict_serial_baseline_median_total_pnl": float(np.median(baseline)),
        "beats_strict_serial_baseline": bool(float(np.median(base_pnl)) > float(np.median(baseline))),
        "total_side_counts": {"C": side_calls, "P": side_puts},
        "seed_rows": [
            {
                "seed": int(row["seed"]),
                "fold": row["fold"],
                "model_total_pnl": float(row["model"]["total_pnl"]),
                "model_profit_factor": float(row["model"]["profit_factor"]),
                "model_trades": int(row["model"]["trades"]),
                "stress_0_10_total_pnl": float(row["model_stress_0_10"]["total_pnl"]),
                "stress_0_25_total_pnl": float(row["model_stress_0_25"]["total_pnl"]),
                "strict_serial_baseline_total_pnl": float(row["strict_serial_baseline"]["total_pnl"]),
            }
            for row in rows
        ],
    }


def _promotion_checks(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        item = aggregate.get(split, {})
        checks.extend(
            [
                {
                    "split": split,
                    "name": "positive_median_pnl",
                    "value": item.get("median_total_pnl", 0.0),
                    "pass": item.get("median_total_pnl", 0.0) > 0.0,
                },
                {
                    "split": split,
                    "name": "positive_seed_fraction_ge_0_80",
                    "value": item.get("positive_seed_fraction", 0.0),
                    "pass": item.get("positive_seed_fraction", 0.0) >= 0.80,
                },
                {
                    "split": split,
                    "name": "median_profit_factor_ge_1_15",
                    "value": item.get("median_profit_factor", 0.0),
                    "pass": item.get("median_profit_factor", 0.0) >= 1.15,
                },
                {
                    "split": split,
                    "name": "positive_stress_0_10_median_pnl",
                    "value": item.get("median_stress_0_10_total_pnl", 0.0),
                    "pass": item.get("median_stress_0_10_total_pnl", 0.0) > 0.0,
                },
                {
                    "split": split,
                    "name": "beats_strict_serial_baseline",
                    "value": item.get("median_total_pnl", 0.0)
                    - item.get("strict_serial_baseline_median_total_pnl", 0.0),
                    "pass": bool(item.get("beats_strict_serial_baseline", False)),
                },
                {
                    "split": split,
                    "name": "reported_stress_0_25_positive",
                    "value": item.get("median_stress_0_25_total_pnl", 0.0),
                    "pass": item.get("median_stress_0_25_total_pnl", 0.0) > 0.0,
                },
            ]
        )
    return checks


def _decision(aggregate: dict[str, Any]) -> str:
    if aggregate.get("promotion_ready"):
        return "promotion_candidate: Protocol 092 beats strict serial baseline under all registered gates"
    hard_failures = [
        item
        for item in aggregate.get("promotion_checks", [])
        if not item["pass"] and item["name"] != "reported_stress_0_25_positive"
    ]
    if hard_failures:
        return "research_rejected_for_promotion: Protocol 092 did not clear the strict serial gate"
    return "research_continue_only: Protocol 092 clears primary gates but fails the 0.25 stress promotion blocker"


def _dataset_summary(dataset: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(dataset)),
        "splits": {str(k): int(v) for k, v in dataset["split"].value_counts().sort_index().items()},
        "label_sources": {str(k): int(v) for k, v in dataset["label_source"].value_counts().sort_index().items()},
        "seeds": sorted(int(x) for x in dataset["seed"].unique()),
        "decision_groups": int(dataset.groupby(["split", "seed", "session", "decision_time"]).ngroups),
    }


def _pre_registration() -> dict[str, Any]:
    return {
        "hypothesis": (
            "A serial entry scorer can improve the frozen Protocol 081 lifecycle stack by learning "
            "which candidate is worth occupying the single contract slot while preserving frozen exits."
        ),
        "paid_data": "forbidden",
        "live_orders": "forbidden",
        "exit_behavior": "frozen_protocol081_candidate_paths",
        "folds": FOLDS,
        "model_seeds": MODEL_SEEDS,
        "threshold_rule": "selected only on chronological validation split for the same reported seed",
        "stress": {
            "primary": "$0.10 extra each side must remain positive",
            "promotion_blocker": "$0.25 extra each side is reported and blocks promotion if negative",
        },
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    gate = payload["aggregate_gate"]
    lines = [
        "# Protocol 092: Serial Opportunity-Cost Neural Policy",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Dataset rows: `{payload['dataset_summary']['rows']}`",
        f"- Candidate dataset: `{payload['dataset_path']}`",
        f"- Feature count: `{len(payload['feature_columns'])}`",
        "",
        "## Registered Gate",
        "",
        _table(
            [
                {
                    "split": split,
                    "median_pnl": gate[split].get("median_total_pnl", 0.0),
                    "pf": gate[split].get("median_profit_factor", 0.0),
                    "positive_seeds": gate[split].get("positive_seed_fraction", 0.0),
                    "trades": gate[split].get("median_trades", 0.0),
                    "stress10": gate[split].get("median_stress_0_10_total_pnl", 0.0),
                    "stress25": gate[split].get("median_stress_0_25_total_pnl", 0.0),
                    "baseline": gate[split].get("strict_serial_baseline_median_total_pnl", 0.0),
                    "beats_baseline": gate[split].get("beats_strict_serial_baseline", False),
                    "sides": gate[split].get("total_side_counts", {}),
                }
                for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
            ],
            ["split", "median_pnl", "pf", "positive_seeds", "trades", "stress10", "stress25", "baseline", "beats_baseline", "sides"],
        ),
        "",
        "## Promotion Checks",
        "",
        _table(gate["promotion_checks"], ["split", "name", "value", "pass"]),
        "",
        "## Dataset Provenance",
        "",
        "```json",
        json.dumps(payload["dataset_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
        "Protocol 092 changes only entry arbitration. It does not alter Protocol 081 lifecycle exits, and it evaluates the strict one-contract opportunity cost directly.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _finite_pf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(out):
        return 999.0
    return out if math.isfinite(out) else 0.0


def _arr(values) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
