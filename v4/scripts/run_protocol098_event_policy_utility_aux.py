"""Protocol 098: sequential event policy with profit-quality auxiliary loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from v4.model.serial_opportunity import ENTRY_FEATURE_COLUMNS, strict_serial_baseline
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS
from v4.scripts.run_protocol097_sequential_event_policy import (
    DEFAULT_PROTOCOL092_DIR,
    EventPolicyConfig,
    _aggregate_gate,
    _candidate_frame_from_events,
    _compare_to_protocol092,
    _json_dumps,
    _reported_event_slices,
    add_oracle_actions,
    build_events,
    select_margin_threshold,
    simulate_event_policy,
    train_event_policy,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_098_event_policy_utility_aux")
MODEL_SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--utility-aux-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.protocol092_dir / "serial_opportunity_dataset.parquet")
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True)
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True)
    events = build_events(dataset)
    oracle_summary = add_oracle_actions(events)
    config = EventPolicyConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        min_validation_trades=int(args.min_validation_trades),
        utility_aux_weight=float(args.utility_aux_weight),
    )

    fold_results: list[dict[str, Any]] = []
    trade_ledgers: list[dict[str, Any]] = []
    for fold in FOLDS:
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_all = [event for event in events if event["split"] == fold["validation_split"]]
        for seed in args.seeds:
            model, scaler, history = train_event_policy(train_events, validation_all, seed=int(seed), config=config)
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (model_dir / "manifest.json").write_text(
                _json_dumps(
                    {
                        "protocol": "098_event_policy_utility_aux",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": ENTRY_FEATURE_COLUMNS,
                        "config": config.__dict__,
                        "history": history,
                    }
                )
            )

            validation_seed = [event for event in validation_all if int(event["seed"]) == int(seed)]
            threshold = select_margin_threshold(validation_seed, model, scaler, seed=int(seed), config=config)
            seed_result = {
                "fold": fold["name"],
                "seed": int(seed),
                "train_splits": fold["train_splits"],
                "validation_split": fold["validation_split"],
                "test_split": fold["test_split"],
                "threshold": float(threshold["threshold"]),
                "threshold_selection": threshold,
                "history_last": history[-1] if history else {},
                "splits": {},
            }
            for split_name, event_slice in _reported_event_slices(events, fold, seed).items():
                candidate_source = _candidate_frame_from_events(event_slice)
                base = simulate_event_policy(event_slice, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.0, strategy=f"protocol098_{fold['name']}")
                stress10 = simulate_event_policy(event_slice, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.10, strategy=f"protocol098_{fold['name']}_stress10")
                stress25 = simulate_event_policy(event_slice, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.25, strategy=f"protocol098_{fold['name']}_stress25")
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
    payload = {
        "protocol": "098_event_policy_utility_aux",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol092_dir": str(args.protocol092_dir),
        "pre_registration": _pre_registration(float(args.utility_aux_weight)),
        "feature_columns": ENTRY_FEATURE_COLUMNS,
        "config": config.__dict__,
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate_gate": aggregate,
        "protocol092_comparison": _compare_to_protocol092(aggregate, protocol092["aggregate_gate"]),
        "decision": _decision(aggregate, protocol092["aggregate_gate"]),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    (args.out_dir / "serial_policy_trades.json").write_text(_json_dumps(trade_ledgers))
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "comparison": payload["protocol092_comparison"]}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def _decision(aggregate: dict[str, Any], previous: dict[str, Any]) -> str:
    if aggregate["promotion_ready"]:
        return "keep_promote_candidate: Protocol 098 clears the strict serial gate"
    comparison = _compare_to_protocol092(aggregate, previous)
    if comparison["q3_2025"]["median_total_pnl_delta"] > 0 and comparison["q4_2025"]["median_total_pnl_delta"] > 0:
        return "keep_for_research_only: Protocol 098 improves Q3/Q4 but still does not clear promotion"
    return "reject: utility-aux event policy did not improve both Q3 and Q4"


def _pre_registration(weight: float) -> dict[str, Any]:
    return {
        "hypothesis": "Protocol 097 learned wait/take but reduced profit factor. Adding a fixed profit-quality auxiliary loss should preserve candidate value while keeping the sequential wait action.",
        "single_change": f"add fixed candidate utility auxiliary loss weight {weight}",
        "features": "unchanged Protocol 092 entry-only feature set",
        "exits": "frozen Protocol 081 candidate exits",
        "paid_data": "forbidden",
        "live_orders": "forbidden",
        "folds": FOLDS,
        "seeds": MODEL_SEEDS,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 098: Event Policy With Utility Auxiliary",
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
