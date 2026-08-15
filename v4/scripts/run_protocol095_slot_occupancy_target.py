"""Protocol 095: slot-occupancy target for serial entry scoring.

This is one model-side change after Protocol 094 failed to move thresholds:
keep entry features and frozen Protocol 081 exits, but train the scorer on a
candidate utility target that subtracts the best near-future opportunity the
candidate would block while occupying the single contract slot.
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
    model_artifact_manifest,
    predict_scores,
    select_validation_threshold,
    serial_simulate_candidates,
    strict_serial_baseline,
    train_serial_opportunity_model,
)
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS


DEFAULT_PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_095_slot_occupancy_target")
MODEL_SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.protocol092_dir / "serial_opportunity_dataset.parquet")
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True)
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True)
    dataset = add_slot_occupancy_target(dataset)
    dataset.to_parquet(args.out_dir / "slot_occupancy_dataset.parquet", index=False)

    config = SerialOpportunityConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        min_validation_trades=int(args.min_validation_trades),
    )
    fold_results: list[dict[str, Any]] = []
    trade_ledgers: list[dict[str, Any]] = []
    for fold in FOLDS:
        train_frame = dataset[dataset["split"].isin(fold["train_splits"])].copy()
        validation_all = dataset[dataset["split"] == fold["validation_split"]].copy()
        for seed in args.seeds:
            model, scaler, history = train_serial_opportunity_model(
                train_frame,
                validation_all,
                seed=int(seed),
                config=config,
                target_column="slot_occupancy_target_pnl",
            )
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pt"
            scaler_path = model_dir / "scaler.json"
            torch.save(model.state_dict(), model_path)
            scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")

            validation_seed = validation_all[validation_all["seed"] == int(seed)].copy()
            validation_seed["score"] = predict_scores(model, scaler, validation_seed, target_scale=config.target_scale)
            threshold = select_validation_threshold(
                validation_seed,
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
            manifest["target_column"] = "slot_occupancy_target_pnl"
            manifest["target_definition"] = "candidate_pnl - max(0, best future candidate_pnl between entry and candidate exit)"
            (model_dir / "manifest.json").write_text(_json_dumps(manifest))
            (model_dir / "threshold_sweep.json").write_text(_json_dumps(threshold.sweep))

            seed_result = {
                "fold": fold["name"],
                "seed": int(seed),
                "train_splits": fold["train_splits"],
                "validation_split": fold["validation_split"],
                "test_split": fold["test_split"],
                "threshold": float(threshold.threshold),
                "threshold_selection": asdict(threshold),
                "history_last": history[-1] if history else {},
                "splits": {},
            }
            for split_name, scored in _score_reported_frames(dataset, fold, seed, model, scaler, config).items():
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
                    strategy=f"protocol095_{fold['name']}",
                )
                model_stress10 = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(threshold.threshold),
                    slippage_per_side=0.10,
                    strategy=f"protocol095_{fold['name']}_stress10",
                )
                model_stress25 = serial_simulate_candidates(
                    scored,
                    score_column="score",
                    threshold=float(threshold.threshold),
                    slippage_per_side=0.25,
                    strategy=f"protocol095_{fold['name']}_stress25",
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
                    trade_ledgers.append(row)
            fold_results.append(seed_result)

    aggregate = _aggregate_gate(fold_results)
    protocol092_summary = json.loads((args.protocol092_dir / "summary.json").read_text())
    payload = {
        "protocol": "095_slot_occupancy_target",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol092_dir": str(args.protocol092_dir),
        "pre_registration": _pre_registration(),
        "feature_columns": ENTRY_FEATURE_COLUMNS,
        "config": asdict(config),
        "target_summary": _target_summary(dataset),
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


def add_slot_occupancy_target(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    out["blocked_best_future_pnl"] = 0.0
    for _, group in out.groupby(["split", "seed", "session"], sort=False):
        ordered = group.sort_values("decision_dt")
        decisions = ordered["decision_dt"].astype("int64").to_numpy()
        exits = ordered["candidate_exit_dt"].astype("int64").to_numpy()
        pnls = ordered["candidate_pnl"].to_numpy(dtype=float)
        indices = ordered.index.to_numpy()
        for idx, decision_ns, exit_ns in zip(indices, decisions, exits):
            mask = (decisions > decision_ns) & (decisions < exit_ns)
            if mask.any():
                best = float(np.nanmax(pnls[mask]))
                out.at[idx, "blocked_best_future_pnl"] = max(0.0, best)
    out["slot_occupancy_target_pnl"] = out["candidate_pnl"] - out["blocked_best_future_pnl"]
    return out


def _score_reported_frames(dataset, fold, seed, model, scaler, config) -> dict[str, pd.DataFrame]:
    out = {}
    test = dataset[(dataset["split"] == fold["test_split"]) & (dataset["seed"] == int(seed))].copy()
    test["score"] = predict_scores(model, scaler, test, target_scale=config.target_scale)
    out[fold["test_split"]] = test
    if fold["test_split"] == "q1_2026":
        out["march_2026"] = test[test["session"] >= "2026-03-01"].copy()
    return out


def _aggregate_gate(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        rows = []
        for result in fold_results:
            if split in result["splits"]:
                rows.append({"seed": result["seed"], "fold": result["fold"], **result["splits"][split]})
        aggregate[split] = _summarize_split(rows)
    aggregate["promotion_checks"] = _promotion_checks(aggregate)
    aggregate["promotion_ready"] = bool(all(item["pass"] for item in aggregate["promotion_checks"]))
    return aggregate


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"seeds": 0}
    base_pnl = _arr(row["model"]["total_pnl"] for row in rows)
    stress10 = _arr(row["model_stress_0_10"]["total_pnl"] for row in rows)
    stress25 = _arr(row["model_stress_0_25"]["total_pnl"] for row in rows)
    pf = _arr(_finite_pf(row["model"]["profit_factor"]) for row in rows)
    trades = _arr(row["model"]["trades"] for row in rows)
    baseline = _arr(row["strict_serial_baseline"]["total_pnl"] for row in rows)
    return {
        "seeds": int(len(rows)),
        "median_total_pnl": float(np.median(base_pnl)),
        "positive_seed_fraction": float((base_pnl > 0).mean()),
        "median_profit_factor": float(np.median(pf)),
        "median_trades": float(np.median(trades)),
        "median_stress_0_10_total_pnl": float(np.median(stress10)),
        "median_stress_0_25_total_pnl": float(np.median(stress25)),
        "strict_serial_baseline_median_total_pnl": float(np.median(baseline)),
        "beats_strict_serial_baseline": bool(float(np.median(base_pnl)) > float(np.median(baseline))),
        "total_side_counts": {
            "C": int(sum(row["model"]["side_counts"].get("C", 0) for row in rows)),
            "P": int(sum(row["model"]["side_counts"].get("P", 0) for row in rows)),
        },
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
                {"split": split, "name": "positive_median_pnl", "value": item.get("median_total_pnl", 0.0), "pass": item.get("median_total_pnl", 0.0) > 0.0},
                {"split": split, "name": "positive_seed_fraction_ge_0_80", "value": item.get("positive_seed_fraction", 0.0), "pass": item.get("positive_seed_fraction", 0.0) >= 0.80},
                {"split": split, "name": "median_profit_factor_ge_1_15", "value": item.get("median_profit_factor", 0.0), "pass": item.get("median_profit_factor", 0.0) >= 1.15},
                {"split": split, "name": "positive_stress_0_10_median_pnl", "value": item.get("median_stress_0_10_total_pnl", 0.0), "pass": item.get("median_stress_0_10_total_pnl", 0.0) > 0.0},
                {"split": split, "name": "beats_strict_serial_baseline", "value": item.get("median_total_pnl", 0.0) - item.get("strict_serial_baseline_median_total_pnl", 0.0), "pass": bool(item.get("beats_strict_serial_baseline", False))},
                {"split": split, "name": "reported_stress_0_25_positive", "value": item.get("median_stress_0_25_total_pnl", 0.0), "pass": item.get("median_stress_0_25_total_pnl", 0.0) > 0.0},
            ]
        )
    return checks


def _compare_to_protocol092(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        now = current[split]
        old = previous[split]
        out[split] = {
            "median_total_pnl_delta": float(now["median_total_pnl"] - old["median_total_pnl"]),
            "median_profit_factor_delta": float(now["median_profit_factor"] - old["median_profit_factor"]),
            "median_trades_delta": float(now["median_trades"] - old["median_trades"]),
            "stress_0_10_delta": float(now["median_stress_0_10_total_pnl"] - old["median_stress_0_10_total_pnl"]),
            "baseline_beat_before": bool(old["beats_strict_serial_baseline"]),
            "baseline_beat_after": bool(now["beats_strict_serial_baseline"]),
        }
    return out


def _decision(aggregate: dict[str, Any], previous: dict[str, Any]) -> str:
    if aggregate["promotion_ready"]:
        return "keep_promote_candidate: Protocol 095 clears the strict serial gate"
    comparison = _compare_to_protocol092(aggregate, previous)
    q3_improved = comparison["q3_2025"]["median_total_pnl_delta"] > 0
    q4_improved = comparison["q4_2025"]["median_total_pnl_delta"] > 0
    if q3_improved and q4_improved:
        return "keep_for_research_only: Protocol 095 improves Q3/Q4 but still does not clear promotion"
    return "reject: slot-occupancy target did not improve both Q3 and Q4"


def _pre_registration() -> dict[str, Any]:
    return {
        "hypothesis": "Protocol 092 lost Q3 mainly through threshold/no-entry and slot occupancy. A target that subtracts the best positive candidate blocked before the selected exit should teach the entry scorer to avoid occupying the one-contract slot before better near-future opportunities.",
        "single_change": "training target only",
        "target": "candidate_pnl - max(0, best future candidate_pnl between candidate entry and candidate exit)",
        "features": "unchanged Protocol 092 entry-only feature set",
        "exits": "frozen Protocol 081 candidate exits",
        "paid_data": "forbidden",
        "live_orders": "forbidden",
        "folds": FOLDS,
        "seeds": MODEL_SEEDS,
    }


def _target_summary(dataset: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(dataset)),
        "blocked_positive_rows": int((dataset["blocked_best_future_pnl"] > 0.0).sum()),
        "candidate_pnl_mean": float(dataset["candidate_pnl"].mean()),
        "slot_target_mean": float(dataset["slot_occupancy_target_pnl"].mean()),
        "blocked_best_future_mean": float(dataset["blocked_best_future_pnl"].mean()),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 095: Slot-Occupancy Target",
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
        "## Target Summary",
        "",
        "```json",
        json.dumps(payload["target_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Promotion Checks",
        "",
        _table(payload["aggregate_gate"]["promotion_checks"], ["split", "name", "value", "pass"]),
        "",
        "## Interpretation",
        "",
        "Protocol 095 is a true model-target change. If it fails, the next hypothesis should move from independent candidate scoring to a sequential decision model that sees candidate sets over time.",
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
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
