"""Protocol168: threshold-objective replay for Protocol163 artifacts.

This is a no-retraining experiment. It reuses a frozen Protocol163-style model
run, changes only the validation threshold-selection objective, and evaluates
whether the same neural scores can beat the current strict-serial benchmarks
without adding model features or touching paid data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS
from v4.scripts.run_protocol097_sequential_event_policy import EventPolicyConfig, EventSetPolicy, build_events, event_margins
from v4.scripts.run_protocol101_event_history_policy import add_causal_history_features
from v4.scripts.run_protocol163_serial_one_account_training import (
    FEATURE_COLUMNS,
    MAY_DIAGNOSTIC_SPLIT,
    RECENT_FOLD,
    RECENT_SPLIT,
    STARTING_CASH,
    aggregate_results,
    reported_event_slices,
    simulate_one_account_event_policy,
    strict_one_account_baseline,
)


LOOP_ID = "v4_aplus_hypothesis_168_protocol163_threshold_replay"
DEFAULT_SOURCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_167_protocol163_wider_longer")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_PROTOCOL101_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json")
DEFAULT_RECENT_PROTOCOL101_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/summary.json")
OBJECTIVES = ("stress_delta", "stress_pf", "stress_avg_pnl", "drawdown_adjusted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-protocol101-summary", type=Path, default=DEFAULT_RECENT_PROTOCOL101_SUMMARY)
    parser.add_argument("--objectives", nargs="*", default=list(OBJECTIVES))
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source_dir / "summary.json").read_text())
    dataset = pd.read_parquet(source["dataset_path"])
    for column in ["decision_dt", "candidate_exit_dt", "entry_quote_dt"]:
        dataset[column] = pd.to_datetime(dataset[column], utc=True, errors="coerce")
    events = build_events(dataset)
    add_causal_history_features(events)
    config = EventPolicyConfig(min_validation_trades=int(args.min_validation_trades))
    protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_protocol101_summary)
    objective_results: dict[str, Any] = {}
    trade_rows: list[dict[str, Any]] = []
    for objective in args.objectives:
        fold_results: list[dict[str, Any]] = []
        for fold in [*FOLDS, RECENT_FOLD]:
            validation_all = [event for event in events if str(event["split"]) == fold["validation_split"]]
            for model_dir in sorted((args.source_dir / "model_artifacts" / fold["name"]).glob("seed_*")):
                seed = int(model_dir.name.split("_", 1)[1])
                model, scaler = load_model(model_dir)
                validation_seed = [event for event in validation_all if int(event["seed"]) == seed]
                threshold = select_threshold(
                    validation_seed,
                    model,
                    scaler,
                    objective=objective,
                    config=config,
                    starting_cash=float(args.starting_cash),
                )
                result = {
                    "fold": fold["name"],
                    "seed": seed,
                    "threshold": threshold["threshold"],
                    "threshold_selection": threshold,
                    "splits": {},
                }
                for split_name, event_slice in reported_event_slices(events, fold, seed).items():
                    base = simulate_one_account_event_policy(
                        event_slice,
                        model,
                        scaler,
                        threshold=float(threshold["threshold"]),
                        slippage_per_side=0.0,
                        strategy=f"protocol168_{objective}",
                        feature_columns=FEATURE_COLUMNS,
                        starting_cash=float(args.starting_cash),
                    )
                    stress10 = simulate_one_account_event_policy(
                        event_slice,
                        model,
                        scaler,
                        threshold=float(threshold["threshold"]),
                        slippage_per_side=0.10,
                        strategy=f"protocol168_{objective}_stress10",
                        feature_columns=FEATURE_COLUMNS,
                        starting_cash=float(args.starting_cash),
                    )
                    stress25 = simulate_one_account_event_policy(
                        event_slice,
                        model,
                        scaler,
                        threshold=float(threshold["threshold"]),
                        slippage_per_side=0.25,
                        strategy=f"protocol168_{objective}_stress25",
                        feature_columns=FEATURE_COLUMNS,
                        starting_cash=float(args.starting_cash),
                    )
                    baseline = strict_one_account_baseline(
                        event_slice,
                        seed=seed,
                        slippage_per_side=0.0,
                        starting_cash=float(args.starting_cash),
                    )
                    result["splits"][split_name] = {
                        "model": base.summary,
                        "model_stress_0_10": stress10.summary,
                        "model_stress_0_25": stress25.summary,
                        "strict_serial_baseline": baseline.summary,
                        "validation_threshold_source": fold["validation_split"],
                    }
                    trade_rows.extend({**trade, "objective": objective, "fold": fold["name"], "seed": seed, "reported_split": split_name} for trade in base.trades)
                fold_results.append(result)
        aggregate = aggregate_results(fold_results)
        objective_results[objective] = {
            "aggregate": aggregate,
            "comparison_to_protocol101": compare_to_protocol101(aggregate, protocol101),
            "decision": objective_decision(aggregate, protocol101),
        }
    best_objective = pick_best_objective(objective_results)
    payload = {
        "protocol": "168_protocol163_threshold_replay",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "source_dir": str(args.source_dir),
        "objectives": objective_results,
        "best_objective": best_objective,
        "decision": objective_results[best_objective]["decision"],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(trade_rows).to_csv(args.out_dir / "protocol168_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "best_objective": best_objective, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_model(model_dir: Path) -> tuple[EventSetPolicy, FeatureScaler]:
    manifest = json.loads((model_dir / "manifest.json").read_text())
    scaler_blob = json.loads((model_dir / "scaler.json").read_text())
    model = EventSetPolicy(input_dim=len(FEATURE_COLUMNS), hidden_dim=int(manifest.get("config", {}).get("hidden_dim", 96)))
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    scaler = FeatureScaler(
        fill=np.asarray(scaler_blob["fill"], dtype=np.float32),
        mean=np.asarray(scaler_blob["mean"], dtype=np.float32),
        std=np.asarray(scaler_blob["std"], dtype=np.float32),
    )
    return model, scaler


def select_threshold(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: FeatureScaler,
    *,
    objective: str,
    config: EventPolicyConfig,
    starting_cash: float,
) -> dict[str, Any]:
    margins = event_margins(events, model, scaler, feature_columns=FEATURE_COLUMNS)
    finite = margins[np.isfinite(margins)]
    thresholds = [float("inf")] if len(finite) == 0 else sorted(set(np.quantile(finite, [0, .1, .2, .35, .5, .65, .75, .85, .9, .95]).round(4).tolist() + [0.0, float(finite.min()) - 1e-3]))
    baseline = strict_one_account_baseline(events, seed=1, slippage_per_side=0.0, starting_cash=starting_cash)
    sweep = []
    for threshold in thresholds:
        base = simulate_one_account_event_policy(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, strategy="protocol168_validation", feature_columns=FEATURE_COLUMNS, starting_cash=starting_cash)
        stress = simulate_one_account_event_policy(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, strategy="protocol168_validation_stress10", feature_columns=FEATURE_COLUMNS, starting_cash=starting_cash)
        sweep.append({"threshold": float(threshold), "model": base.summary, "model_stress_0_10": stress.summary, "strict_serial_baseline": baseline.summary, "score": objective_score(objective, base.summary, stress.summary, baseline.summary)})
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: row["score"])
    return {"threshold": float(best["threshold"]), "objective": objective, "selected": best, "sweep": sweep}


def objective_score(objective: str, base: dict[str, Any], stress: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, ...]:
    stress_delta = float(stress["total_pnl"] - baseline["total_pnl"])
    if objective == "stress_delta":
        return (stress_delta, float(base["total_pnl"]), float(base["trades"]))
    if objective == "stress_pf":
        return (finite_pf(stress.get("profit_factor")), stress_delta, float(stress["total_pnl"]))
    if objective == "stress_avg_pnl":
        return (safe_div(float(stress["total_pnl"]), max(float(stress["trades"]), 1.0)), stress_delta, float(stress["trades"]))
    if objective == "drawdown_adjusted":
        return (float(stress["total_pnl"]) + float(stress.get("max_drawdown", 0.0)), stress_delta, float(stress["total_pnl"]))
    raise ValueError(f"unknown objective {objective!r}")


def load_protocol101_baselines(path: Path, recent_path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for split, item in payload.get("aggregate_gate", {}).items():
        if isinstance(item, dict) and "median_total_pnl" in item:
            out[split] = float(item["median_total_pnl"])
    if recent_path.exists():
        recent = json.loads(recent_path.read_text())
        serial = recent.get("serial_protocol081_summary", {})
        if "total_pnl" in serial:
            out[RECENT_SPLIT] = float(serial["total_pnl"])
    return out


def compare_to_protocol101(aggregate: dict[str, Any], protocol101: dict[str, float]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]:
        item = aggregate.get(split, {})
        baseline = protocol101.get(split)
        if not item or baseline is None:
            continue
        out[split] = {
            "median_total_pnl": item.get("median_total_pnl"),
            "protocol101_median_total_pnl": baseline,
            "delta": None if item.get("median_total_pnl") is None else float(item["median_total_pnl"] - baseline),
            "beats_protocol101": bool(item.get("median_total_pnl", -1e18) > baseline),
        }
    return out


def objective_decision(aggregate: dict[str, Any], protocol101: dict[str, float]) -> str:
    comparison = compare_to_protocol101(aggregate, protocol101)
    required = ["q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]
    if all(comparison.get(split, {}).get("beats_protocol101") for split in required):
        return "candidate_beats_protocol101_required_blocks"
    if aggregate.get(RECENT_SPLIT, {}).get("beats_strict_serial_baseline"):
        return "keep_for_research_recent_improved_but_not_protocol101_replacement"
    return "reject_threshold_objective"


def pick_best_objective(results: dict[str, Any]) -> str:
    def key(name: str) -> tuple[float, float, float]:
        agg = results[name]["aggregate"]
        cmp = results[name]["comparison_to_protocol101"]
        required = ["q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]
        beat_count = sum(1 for split in required if cmp.get(split, {}).get("beats_protocol101"))
        recent = float(agg.get(RECENT_SPLIT, {}).get("median_total_pnl", -1e18))
        q4_delta = float(cmp.get("q4_2025", {}).get("delta") or -1e18)
        return (float(beat_count), recent, q4_delta)
    return max(results, key=key)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Protocol168 Threshold Replay", "", f"- Decision: `{payload['decision']}`", f"- Best objective: `{payload['best_objective']}`", f"- Source: `{payload['source_dir']}`", "", "| objective | q4 delta vs P101 | q1 delta | march delta | recent delta | decision |", "|---|---:|---:|---:|---:|---|"]
    for name, result in payload["objectives"].items():
        cmp = result["comparison_to_protocol101"]
        lines.append(
            f"| {name} | {fmt(cmp.get('q4_2025', {}).get('delta'))} | {fmt(cmp.get('q1_2026', {}).get('delta'))} | "
            f"{fmt(cmp.get('march_2026', {}).get('delta'))} | {fmt(cmp.get(RECENT_SPLIT, {}).get('delta'))} | `{result['decision']}` |"
        )
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Trades: `{path.parent / 'protocol168_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def finite_pf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 999.0 if math.isinf(out) else (out if math.isfinite(out) else 0.0)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def fmt(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(out) else f"{out:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
