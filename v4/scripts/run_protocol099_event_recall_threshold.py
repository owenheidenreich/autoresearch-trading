"""Protocol 099: recall-preserving threshold for the sequential event policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.serial_opportunity import ENTRY_FEATURE_COLUMNS, strict_serial_baseline
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS
from v4.scripts.run_protocol097_sequential_event_policy import (
    DEFAULT_PROTOCOL092_DIR,
    EventPolicyConfig,
    EventSetPolicy,
    _aggregate_gate,
    _candidate_frame_from_events,
    _compare_to_protocol092,
    _json_dumps,
    _reported_event_slices,
    add_oracle_actions,
    build_events,
    event_margins,
    simulate_event_policy,
)


DEFAULT_PROTOCOL097_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_097_sequential_event_policy")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_099_event_recall_threshold")
MODEL_SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--protocol097-dir", type=Path, default=DEFAULT_PROTOCOL097_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--baseline-trade-ratio", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.protocol092_dir / "serial_opportunity_dataset.parquet")
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True)
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True)
    events = build_events(dataset)
    add_oracle_actions(events)
    config = EventPolicyConfig(min_validation_trades=int(args.min_validation_trades))

    fold_results: list[dict[str, Any]] = []
    trade_ledgers: list[dict[str, Any]] = []
    for fold in FOLDS:
        validation_all = [event for event in events if event["split"] == fold["validation_split"]]
        for seed in args.seeds:
            model, scaler = _load_protocol097_artifact(args.protocol097_dir, fold["name"], int(seed))
            validation_seed = [event for event in validation_all if int(event["seed"]) == int(seed)]
            threshold = select_recall_threshold(
                validation_seed,
                model,
                scaler,
                seed=int(seed),
                config=config,
                baseline_trade_ratio=float(args.baseline_trade_ratio),
            )
            seed_result = {
                "fold": fold["name"],
                "seed": int(seed),
                "train_splits": fold["train_splits"],
                "validation_split": fold["validation_split"],
                "test_split": fold["test_split"],
                "threshold": float(threshold["threshold"]),
                "threshold_selection": threshold,
                "splits": {},
            }
            for split_name, event_slice in _reported_event_slices(events, fold, seed).items():
                candidate_source = _candidate_frame_from_events(event_slice)
                base = simulate_event_policy(event_slice, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.0, strategy=f"protocol099_{fold['name']}")
                stress10 = simulate_event_policy(event_slice, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.10, strategy=f"protocol099_{fold['name']}_stress10")
                stress25 = simulate_event_policy(event_slice, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.25, strategy=f"protocol099_{fold['name']}_stress25")
                baseline = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.0)
                baseline10 = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.10)
                seed_result["splits"][split_name] = {
                    "model": base.summary,
                    "model_stress_0_10": stress10.summary,
                    "model_stress_0_25": stress25.summary,
                    "strict_serial_baseline": baseline.summary,
                    "strict_serial_baseline_stress_0_10": baseline10.summary,
                    "validation_threshold_source": fold["validation_split"],
                }
                for trade in base.trades:
                    row = dict(trade)
                    row["fold"] = fold["name"]
                    row["reported_split"] = split_name
                    trade_ledgers.append(row)
            fold_results.append(seed_result)

    aggregate = _aggregate_gate(fold_results)
    protocol092 = json.loads((args.protocol092_dir / "summary.json").read_text())
    protocol097 = json.loads((args.protocol097_dir / "summary.json").read_text())
    payload = {
        "protocol": "099_event_recall_threshold",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol097_dir": str(args.protocol097_dir),
        "pre_registration": _pre_registration(float(args.baseline_trade_ratio)),
        "feature_columns": ENTRY_FEATURE_COLUMNS,
        "fold_results": fold_results,
        "aggregate_gate": aggregate,
        "protocol092_comparison": _compare_to_protocol092(aggregate, protocol092["aggregate_gate"]),
        "protocol097_comparison": _compare_to_protocol092(aggregate, protocol097["aggregate_gate"]),
        "decision": _decision(aggregate, protocol092["aggregate_gate"]),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    (args.out_dir / "serial_policy_trades.json").write_text(_json_dumps(trade_ledgers))
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "vs_097": payload["protocol097_comparison"], "vs_092": payload["protocol092_comparison"]}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def select_recall_threshold(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: FeatureScaler,
    *,
    seed: int,
    config: EventPolicyConfig,
    baseline_trade_ratio: float,
) -> dict[str, Any]:
    margins = event_margins(events, model, scaler)
    finite = margins[np.isfinite(margins)]
    thresholds = [float("inf")] if len(finite) == 0 else sorted(set(np.quantile(finite, [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]).round(4).tolist() + [0.0, float(finite.min()) - 1e-3]))
    candidate_source = _candidate_frame_from_events(events)
    baseline = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.0)
    baseline10 = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.10)
    min_trades = max(int(config.min_validation_trades), int(np.floor(float(baseline.summary["trades"]) * baseline_trade_ratio)))
    sweep = []
    for threshold in thresholds:
        base = simulate_event_policy(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, strategy="protocol099_validation")
        stress10 = simulate_event_policy(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, strategy="protocol099_validation_stress10")
        sweep.append(
            {
                "threshold": float(threshold),
                "model": base.summary,
                "model_stress_0_10": stress10.summary,
                "strict_serial_baseline": baseline.summary,
                "strict_serial_baseline_stress_0_10": baseline10.summary,
                "delta_vs_baseline": float(base.summary["total_pnl"] - baseline.summary["total_pnl"]),
                "delta_vs_baseline_stress_0_10": float(stress10.summary["total_pnl"] - baseline10.summary["total_pnl"]),
                "min_validation_trades": int(min_trades),
            }
        )
    eligible = [
        row
        for row in sweep
        if row["model"]["trades"] >= min_trades
        and row["model_stress_0_10"]["total_pnl"] > 0.0
    ]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["delta_vs_baseline_stress_0_10"], row["delta_vs_baseline"], row["model"]["total_pnl"], row["model"]["trades"]))
    return {
        "threshold": float(best["threshold"]),
        "source_seed": int(seed),
        "source_rows": int(len(events)),
        "objective": "validation stress_0_10 delta vs strict serial baseline with baseline-trade-ratio floor",
        "baseline_trade_ratio": float(baseline_trade_ratio),
        "min_validation_trades": int(min_trades),
        "selected": best,
        "sweep": sweep,
    }


def _load_protocol097_artifact(protocol097_dir: Path, fold_name: str, seed: int) -> tuple[EventSetPolicy, FeatureScaler]:
    model_dir = protocol097_dir / "model_artifacts" / fold_name / f"seed_{seed}"
    manifest = json.loads((model_dir / "manifest.json").read_text())
    cfg = manifest.get("config", {})
    scaler_payload = json.loads((model_dir / "scaler.json").read_text())
    scaler = FeatureScaler(
        fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
        mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    model = EventSetPolicy(input_dim=len(ENTRY_FEATURE_COLUMNS), hidden_dim=int(cfg.get("hidden_dim", 96)))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, scaler


def _decision(aggregate: dict[str, Any], previous: dict[str, Any]) -> str:
    if aggregate["promotion_ready"]:
        return "keep_promote_candidate: Protocol 099 clears the strict serial gate"
    comparison = _compare_to_protocol092(aggregate, previous)
    if comparison["q3_2025"]["median_total_pnl_delta"] > 0 and comparison["q4_2025"]["median_total_pnl_delta"] > 0:
        return "keep_for_research_only: Protocol 099 improves Q3/Q4 but still does not clear promotion"
    return "reject: recall-preserving threshold did not improve both Q3 and Q4"


def _pre_registration(ratio: float) -> dict[str, Any]:
    return {
        "hypothesis": "Protocol 097's residual Q4 weakness is under-trading caused by an overactive wait action. A validation threshold floor requiring near-baseline trade recall should recover missed winners without retraining.",
        "single_change": f"validation threshold must keep at least {ratio:.2f} of strict-baseline validation trade count",
        "model_weights": "reuse Protocol 097 artifacts",
        "features": "unchanged Protocol 092 entry-only feature set",
        "exits": "frozen Protocol 081 candidate exits",
        "paid_data": "forbidden",
        "live_orders": "forbidden",
        "folds": FOLDS,
        "seeds": MODEL_SEEDS,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 099: Recall-Preserving Event Threshold",
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
                    "vs_protocol097": payload["protocol097_comparison"][split]["median_total_pnl_delta"],
                }
                for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
            ],
            ["split", "median_pnl", "pf", "trades", "stress10", "stress25", "baseline", "beats_baseline", "vs_protocol092", "vs_protocol097"],
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
