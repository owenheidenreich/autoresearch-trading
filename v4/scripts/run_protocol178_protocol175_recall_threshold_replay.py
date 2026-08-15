"""Protocol178: recall-constrained threshold replay for Protocol175.

Protocol177 showed Protocol175's Q4 loss is mostly flat-state skipping,
especially seed 3 post-open calls. This no-retraining protocol keeps the
Protocol175 model weights frozen and changes only validation threshold
selection: a threshold is eligible only if it preserves a configurable fraction
of the strict serial validation baseline's trade count.
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
from v4.scripts.run_protocol097_sequential_event_policy import EventPolicyConfig, EventSetPolicy, build_events, event_margins
from v4.scripts.run_protocol101_event_history_policy import add_causal_history_features
from v4.scripts.run_protocol163_serial_one_account_training import (
    FEATURE_COLUMNS,
    RECENT_SPLIT,
    STARTING_CASH,
    aggregate_results,
    reported_event_slices,
    simulate_one_account_event_policy,
    strict_one_account_baseline,
)
from v4.scripts.run_protocol175_protocol163_q4_prehistory import FOLDS_WITH_PREHISTORY, load_protocol101_baselines


LOOP_ID = "v4_aplus_hypothesis_178_protocol175_recall_threshold_replay"
DEFAULT_SOURCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_175_protocol163_q4_2024_prehistory")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_PROTOCOL101_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json")
DEFAULT_RECENT_PROTOCOL101_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/summary.json")
RECALL_FLOORS = (0.90, 0.95, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-protocol101-summary", type=Path, default=DEFAULT_RECENT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recall-floors", nargs="*", type=float, default=list(RECALL_FLOORS))
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
    all_trades: list[dict[str, Any]] = []
    for recall_floor in args.recall_floors:
        objective = f"recall_{int(round(float(recall_floor) * 100)):03d}"
        fold_results: list[dict[str, Any]] = []
        for fold in FOLDS_WITH_PREHISTORY:
            validation_all = [event for event in events if str(event["split"]) == fold["validation_split"]]
            artifact_root = args.source_dir / "model_artifacts" / fold["name"]
            for model_dir in sorted(artifact_root.glob("seed_*")):
                seed = int(model_dir.name.split("_", 1)[1])
                model, scaler = load_model(model_dir)
                validation_seed = [event for event in validation_all if int(event["seed"]) == seed]
                threshold = select_recall_threshold(
                    validation_seed,
                    model,
                    scaler,
                    seed=seed,
                    recall_floor=float(recall_floor),
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
                        strategy=f"protocol178_{objective}",
                        feature_columns=FEATURE_COLUMNS,
                        starting_cash=float(args.starting_cash),
                    )
                    stress10 = simulate_one_account_event_policy(
                        event_slice,
                        model,
                        scaler,
                        threshold=float(threshold["threshold"]),
                        slippage_per_side=0.10,
                        strategy=f"protocol178_{objective}_stress10",
                        feature_columns=FEATURE_COLUMNS,
                        starting_cash=float(args.starting_cash),
                    )
                    stress25 = simulate_one_account_event_policy(
                        event_slice,
                        model,
                        scaler,
                        threshold=float(threshold["threshold"]),
                        slippage_per_side=0.25,
                        strategy=f"protocol178_{objective}_stress25",
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
                    all_trades.extend(
                        {**trade, "objective": objective, "fold": fold["name"], "seed": seed, "reported_split": split_name}
                        for trade in base.trades
                    )
                fold_results.append(result)
        aggregate = aggregate_results(fold_results)
        objective_results[objective] = {
            "recall_floor": float(recall_floor),
            "aggregate": aggregate,
            "comparison_to_protocol101": compare_to_protocol101(aggregate, protocol101),
            "decision": objective_decision(aggregate, protocol101),
            "fold_results": fold_results,
        }
    best_objective = pick_best_objective(objective_results)
    payload = {
        "protocol": "178_protocol175_recall_threshold_replay",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "source_dir": str(args.source_dir),
        "objectives": objective_results,
        "best_objective": best_objective,
        "decision": objective_results[best_objective]["decision"],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(all_trades).to_csv(args.out_dir / "protocol178_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "best_objective": best_objective, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_model(model_dir: Path) -> tuple[EventSetPolicy, FeatureScaler]:
    manifest = json.loads((model_dir / "manifest.json").read_text())
    scaler_blob = json.loads((model_dir / "scaler.json").read_text())
    model = EventSetPolicy(input_dim=len(FEATURE_COLUMNS), hidden_dim=int(manifest.get("config", {}).get("hidden_dim", 96)))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    scaler = FeatureScaler(
        fill=np.asarray(scaler_blob["fill"], dtype=np.float32),
        mean=np.asarray(scaler_blob["mean"], dtype=np.float32),
        std=np.asarray(scaler_blob["std"], dtype=np.float32),
    )
    return model, scaler


def select_recall_threshold(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: FeatureScaler,
    *,
    seed: int,
    recall_floor: float,
    config: EventPolicyConfig,
    starting_cash: float,
) -> dict[str, Any]:
    margins = event_margins(events, model, scaler, feature_columns=FEATURE_COLUMNS)
    finite = margins[np.isfinite(margins)]
    thresholds = [float("inf")] if len(finite) == 0 else sorted(
        set(
            np.quantile(finite, [0, .05, .1, .2, .35, .5, .65, .75, .85, .9, .95]).round(4).tolist()
            + [0.0, float(finite.min()) - 1e-3]
        )
    )
    baseline = strict_one_account_baseline(events, seed=seed, slippage_per_side=0.0, starting_cash=starting_cash)
    baseline10 = strict_one_account_baseline(events, seed=seed, slippage_per_side=0.10, starting_cash=starting_cash)
    min_trades = max(int(config.min_validation_trades), int(math.ceil(float(recall_floor) * float(baseline.summary["trades"]))))
    sweep = []
    for threshold in thresholds:
        base = simulate_one_account_event_policy(
            events,
            model,
            scaler,
            threshold=float(threshold),
            slippage_per_side=0.0,
            strategy="protocol178_validation",
            feature_columns=FEATURE_COLUMNS,
            starting_cash=starting_cash,
        )
        stress = simulate_one_account_event_policy(
            events,
            model,
            scaler,
            threshold=float(threshold),
            slippage_per_side=0.10,
            strategy="protocol178_validation_stress10",
            feature_columns=FEATURE_COLUMNS,
            starting_cash=starting_cash,
        )
        sweep.append(
            {
                "threshold": float(threshold),
                "model": base.summary,
                "model_stress_0_10": stress.summary,
                "strict_serial_baseline": baseline.summary,
                "strict_serial_baseline_stress_0_10": baseline10.summary,
                "trade_recall_vs_strict_baseline": safe_div(float(base.summary["trades"]), float(baseline.summary["trades"])),
                "delta_vs_baseline": float(base.summary["total_pnl"] - baseline.summary["total_pnl"]),
                "delta_vs_baseline_stress_0_10": float(stress.summary["total_pnl"] - baseline10.summary["total_pnl"]),
            }
        )
    eligible = [
        row
        for row in sweep
        if int(row["model"]["trades"]) >= min_trades
        and row["model_stress_0_10"]["total_pnl"] > 0.0
    ]
    pool = eligible if eligible else sweep
    best = max(
        pool,
        key=lambda row: (
            row["delta_vs_baseline_stress_0_10"],
            row["delta_vs_baseline"],
            row["model_stress_0_10"]["total_pnl"],
            row["model"]["trades"],
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "objective": f"validation recall >= {recall_floor:.2f}, then stress_0_10 delta vs strict serial baseline",
        "source_seed": int(seed),
        "baseline_validation_trades": int(baseline.summary["trades"]),
        "min_validation_trades_after_recall": int(min_trades),
        "selected": best,
        "sweep": sweep,
    }


def compare_to_protocol101(aggregate: dict[str, Any], protocol101: dict[str, float]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]:
        item = aggregate.get(split, {})
        baseline = protocol101.get(split)
        if not item or baseline is None:
            continue
        median = float(item.get("median_total_pnl", 0.0))
        out[split] = {
            "median_total_pnl": median,
            "protocol101_median_total_pnl": float(baseline),
            "delta": float(median - baseline),
            "beats_protocol101": bool(median > baseline),
            "positive_seed_fraction": float(item.get("positive_seed_fraction", 0.0)),
            "median_stress_0_10_total_pnl": float(item.get("median_stress_0_10_total_pnl", 0.0)),
            "median_profit_factor": float(item.get("median_profit_factor", 0.0)),
        }
    return out


def objective_decision(aggregate: dict[str, Any], protocol101: dict[str, float]) -> str:
    comparison = compare_to_protocol101(aggregate, protocol101)
    required = ["q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]
    if all(
        comparison.get(split, {}).get("beats_protocol101")
        and comparison.get(split, {}).get("positive_seed_fraction", 0.0) >= 0.8
        and comparison.get(split, {}).get("median_stress_0_10_total_pnl", 0.0) > 0.0
        for split in required
    ):
        return "promote_research_candidate: recall-constrained Protocol175 threshold replay beats Protocol101 gate"
    if comparison.get("q4_2025", {}).get("delta", -1e18) > -1500.0:
        return "keep_for_research: recall constraint narrows Q4 gap but does not clear Protocol101 gate"
    return "reject_current_hypothesis: recall-constrained thresholds do not solve Q4 gap"


def pick_best_objective(results: dict[str, Any]) -> str:
    def key(name: str) -> tuple[float, float, float, float]:
        comparison = results[name]["comparison_to_protocol101"]
        return (
            float(comparison.get("q4_2025", {}).get("delta", -1e18)),
            float(comparison.get("q3_2025", {}).get("delta", -1e18)),
            float(comparison.get("q1_2026", {}).get("delta", -1e18)),
            float(comparison.get(RECENT_SPLIT, {}).get("delta", -1e18)),
        )

    return max(results, key=key)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol178 Recall-Constrained Threshold Replay",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source: `{payload['source_dir']}`",
        f"- Best objective: `{payload['best_objective']}`",
        f"- Paid data downloaded: `{payload['paid_data_downloaded_by_runner']}`",
        "",
        "## Objective Comparison",
        "",
        "| objective | split | median PnL | Protocol101 | delta | PF | stress $0.10 | trades |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for objective, result in payload["objectives"].items():
        aggregate = result["aggregate"]
        comparison = result["comparison_to_protocol101"]
        for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]:
            item = aggregate.get(split, {})
            if not item:
                continue
            comp = comparison.get(split, {})
            lines.append(
                f"| {objective} | {split} | {fmt(item.get('median_total_pnl'))} | "
                f"{fmt(comp.get('protocol101_median_total_pnl'))} | {fmt(comp.get('delta'))} | "
                f"{fmt(item.get('median_profit_factor'))} | {fmt(item.get('median_stress_0_10_total_pnl'))} | "
                f"{fmt(item.get('median_trades'))} |"
            )
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Trades: `{path.parent / 'protocol178_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return float(numerator) / float(denominator)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
