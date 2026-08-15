"""Protocol 175: Protocol163 with Q4 2024 prehistory.

Protocol163 is the strongest near-term serial/account-aware model, but its
main weakness is Q4 2025 versus frozen Protocol101. Protocol175 tests one
specific hypothesis: adding already-collected Q4 2024 as chronological
prehistory improves regime transfer without buying new data or changing live
paper rules.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.scripts.run_protocol097_sequential_event_policy import (
    EventPolicyConfig,
    add_oracle_actions,
    build_events,
    train_event_policy,
)
from v4.scripts.run_protocol101_event_history_policy import add_causal_history_features
from v4.scripts.run_protocol163_serial_one_account_training import (
    DEFAULT_PROTOCOL092_DATASET,
    DEFAULT_RECENT_BASELINE_SUMMARY,
    DEFAULT_RECENT_CANDIDATES,
    DEFAULT_NORMALIZED_DIR,
    FEATURE_COLUMNS,
    MAY_DIAGNOSTIC_SPLIT,
    MODEL_SEEDS,
    RECENT_SPLIT,
    STARTING_CASH,
    _json_dumps,
    _load_recent_frozen_baseline,
    aggregate_results,
    build_protocol163_dataset,
    event_summary,
    reported_event_slices,
    select_account_threshold,
    simulate_one_account_event_policy,
    strict_one_account_baseline,
)


LOOP_ID = "v4_aplus_hypothesis_175_protocol163_q4_2024_prehistory"
DEFAULT_Q4_2024_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress/"
    "q4_2024_external_serial_opportunity_dataset.parquet"
)
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_PROTOCOL101_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json")
Q4_2024_SPLIT = "q4_2024_external"
FOLDS_WITH_PREHISTORY = [
    {
        "name": "fold1_q4_2024_q1_validate_q2_test_q3",
        "train_splits": [Q4_2024_SPLIT, "q1_2025"],
        "validation_split": "q2_2025",
        "test_split": "q3_2025",
        "reported_splits": ["q3_2025"],
    },
    {
        "name": "fold2_q4_2024_q1_q2_validate_q3_test_q4",
        "train_splits": [Q4_2024_SPLIT, "q1_2025", "q2_2025"],
        "validation_split": "q3_2025",
        "test_split": "q4_2025",
        "reported_splits": ["q4_2025"],
    },
    {
        "name": "fold3_q4_2024_q1_q2_q3_validate_q4_test_q1_2026",
        "train_splits": [Q4_2024_SPLIT, "q1_2025", "q2_2025", "q3_2025"],
        "validation_split": "q4_2025",
        "test_split": "q1_2026",
        "reported_splits": ["q1_2026", "march_2026"],
    },
    {
        "name": "fold4_q4_2024_2025_validate_q1_2026_test_recent",
        "train_splits": [Q4_2024_SPLIT, "q1_2025", "q2_2025", "q3_2025", "q4_2025"],
        "validation_split": "q1_2026",
        "test_split": RECENT_SPLIT,
        "reported_splits": [RECENT_SPLIT, MAY_DIAGNOSTIC_SPLIT],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dataset", type=Path, default=DEFAULT_PROTOCOL092_DATASET)
    parser.add_argument("--q4-2024-dataset", type=Path, default=DEFAULT_Q4_2024_DATASET)
    parser.add_argument("--recent-candidates", type=Path, default=DEFAULT_RECENT_CANDIDATES)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE_SUMMARY)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_source = args.out_dir / "serial_opportunity_with_q4_2024_prehistory.parquet"
    combined_audit = write_combined_source(args.protocol092_dataset, args.q4_2024_dataset, combined_source)
    config = EventPolicyConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        min_validation_trades=int(args.min_validation_trades),
    )
    dataset, dataset_audit = build_protocol163_dataset(
        protocol092_dataset=combined_source,
        recent_candidates=args.recent_candidates,
        normalized_dir=args.normalized_dir,
        seeds=args.seeds,
        starting_cash=float(args.starting_cash),
    )
    dataset_path = args.out_dir / "protocol175_serial_one_account_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    events = build_events(dataset)
    add_causal_history_features(events)
    oracle_summary = add_oracle_actions(events)

    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    for fold in FOLDS_WITH_PREHISTORY:
        train_events = [event for event in events if str(event["split"]) in set(fold["train_splits"])]
        validation_all = [event for event in events if str(event["split"]) == fold["validation_split"]]
        for seed in args.seeds:
            model, scaler, history = train_event_policy(
                train_events,
                validation_all,
                seed=int(seed),
                config=config,
                feature_columns=FEATURE_COLUMNS,
            )
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pt"
            scaler_path = model_dir / "scaler.json"
            torch.save(model.state_dict(), model_path)
            scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            validation_seed = [event for event in validation_all if int(event["seed"]) == int(seed)]
            threshold = select_account_threshold(
                validation_seed,
                model,
                scaler,
                seed=int(seed),
                config=config,
                feature_columns=FEATURE_COLUMNS,
                starting_cash=float(args.starting_cash),
            )
            manifest = {
                "protocol": "175_protocol163_q4_2024_prehistory",
                "fold_name": fold["name"],
                "seed": int(seed),
                "feature_columns": FEATURE_COLUMNS,
                "config": asdict(config),
                "starting_cash": float(args.starting_cash),
                "threshold_selection": threshold,
                "training_history": history,
                "files": {"model": str(model_path), "scaler": str(scaler_path)},
            }
            (model_dir / "manifest.json").write_text(_json_dumps(manifest))
            seed_result: dict[str, Any] = {
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
            for split_name, event_slice in reported_event_slices(events, fold, int(seed)).items():
                model_base = simulate_one_account_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    strategy=f"protocol175_{fold['name']}",
                    feature_columns=FEATURE_COLUMNS,
                    starting_cash=float(args.starting_cash),
                )
                model_stress10 = simulate_one_account_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    strategy=f"protocol175_{fold['name']}_stress10",
                    feature_columns=FEATURE_COLUMNS,
                    starting_cash=float(args.starting_cash),
                )
                model_stress25 = simulate_one_account_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    strategy=f"protocol175_{fold['name']}_stress25",
                    feature_columns=FEATURE_COLUMNS,
                    starting_cash=float(args.starting_cash),
                )
                baseline_base = strict_one_account_baseline(
                    event_slice,
                    seed=int(seed),
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                )
                baseline_stress10 = strict_one_account_baseline(
                    event_slice,
                    seed=int(seed),
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                )
                seed_result["splits"][split_name] = {
                    "model": model_base.summary,
                    "model_stress_0_10": model_stress10.summary,
                    "model_stress_0_25": model_stress25.summary,
                    "strict_serial_baseline": baseline_base.summary,
                    "strict_serial_baseline_stress_0_10": baseline_stress10.summary,
                    "validation_threshold_source": fold["validation_split"],
                }
                model_trades.extend({**trade, "fold": fold["name"], "reported_split": split_name} for trade in model_base.trades)
                baseline_trades.extend({**trade, "fold": fold["name"], "reported_split": split_name} for trade in baseline_base.trades)
            fold_results.append(seed_result)

    protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    payload = {
        "protocol": "175_protocol163_q4_2024_prehistory",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "starting_cash": float(args.starting_cash),
        "combined_source_audit": combined_audit,
        "source_protocol092_dataset": str(args.protocol092_dataset),
        "source_q4_2024_dataset": str(args.q4_2024_dataset),
        "source_recent_candidates": str(args.recent_candidates),
        "dataset_path": str(dataset_path),
        "feature_columns": FEATURE_COLUMNS,
        "dataset_audit": dataset_audit,
        "event_summary": event_summary(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate_results(fold_results),
        "frozen_protocol101_baselines": protocol101,
        "recent_frozen_protocol101_baseline": _load_recent_frozen_baseline(args.recent_baseline_summary),
    }
    payload["protocol101_comparison"] = compare_to_protocol101(payload["aggregate"], protocol101)
    payload["decision"] = decision(payload)
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    pd.DataFrame(model_trades).to_csv(args.out_dir / "protocol175_model_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "strict_one_account_baseline_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def write_combined_source(protocol092_path: Path, q4_2024_path: Path, out_path: Path) -> dict[str, Any]:
    base = pd.read_parquet(protocol092_path)
    q4 = pd.read_parquet(q4_2024_path)
    combined = pd.concat([q4, base], ignore_index=True, sort=False)
    combined = combined.sort_values(["split", "seed", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)
    combined.to_parquet(out_path, index=False)
    return {
        "base_rows": int(len(base)),
        "q4_2024_rows": int(len(q4)),
        "combined_rows": int(len(combined)),
        "rows_by_split": {str(k): int(v) for k, v in combined["split"].value_counts().sort_index().items()},
    }


def load_protocol101_baselines(protocol101_summary: Path, recent_summary: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if protocol101_summary.exists():
        payload = json.loads(protocol101_summary.read_text())
        for split, item in payload.get("aggregate_gate", {}).items():
            if isinstance(item, dict) and "median_total_pnl" in item:
                out[split] = float(item["median_total_pnl"])
    if recent_summary.exists():
        recent = json.loads(recent_summary.read_text()).get("serial_protocol081_summary", {})
        if "total_pnl" in recent:
            out[RECENT_SPLIT] = float(recent["total_pnl"])
    return out


def compare_to_protocol101(aggregate: dict[str, Any], baselines: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]:
        item = aggregate.get(split, {})
        frozen = baselines.get(split)
        if not isinstance(item, dict) or item.get("seeds", 0) == 0 or frozen is None:
            continue
        median = float(item["median_total_pnl"])
        stress = float(item["median_stress_0_10_total_pnl"])
        out[split] = {
            "median_total_pnl": median,
            "frozen_protocol101_total_pnl": float(frozen),
            "median_delta_vs_frozen_protocol101": float(median - frozen),
            "beats_frozen_protocol101": bool(median > frozen),
            "stress_0_10_positive": bool(stress > 0.0),
            "positive_seed_fraction": float(item.get("positive_seed_fraction", 0.0)),
            "median_profit_factor": float(item.get("median_profit_factor", 0.0)),
        }
    return out


def decision(payload: dict[str, Any]) -> str:
    comparison = payload.get("protocol101_comparison", {})
    required = ["q4_2025", "q1_2026", "march_2026", RECENT_SPLIT]
    if all(
        comparison.get(split, {}).get("beats_frozen_protocol101")
        and comparison.get(split, {}).get("stress_0_10_positive")
        and comparison.get(split, {}).get("positive_seed_fraction", 0.0) >= 0.8
        for split in required
    ):
        return "promote_research_candidate: Q4 2024 prehistory Protocol163 beats frozen Protocol101 gate"
    return "research_only: Q4 2024 prehistory did not clear the frozen Protocol101 gate"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol175 Q4 2024 Prehistory",
        "",
        "Tests whether adding already-collected Q4 2024 serial opportunity data to training improves the Protocol163 Q4 transfer gap.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Dataset: `{payload['dataset_path']}`",
        f"- Paid data downloaded by runner: `{payload['paid_data_downloaded_by_runner']}`",
        "",
        "## Aggregate",
        "",
        "| split | seeds | median PnL | strict baseline | delta baseline | frozen P101 | delta P101 | PF | stress $0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", RECENT_SPLIT, MAY_DIAGNOSTIC_SPLIT]:
        item = payload["aggregate"].get(split, {})
        if item.get("seeds", 0) == 0:
            continue
        comparison = payload["protocol101_comparison"].get(split, {})
        lines.append(
            f"| {split} | {item['seeds']} | {_fmt(item['median_total_pnl'])} | "
            f"{_fmt(item['strict_serial_baseline_median_total_pnl'])} | {_fmt(item['median_delta_vs_baseline'])} | "
            f"{_fmt(comparison.get('frozen_protocol101_total_pnl'))} | {_fmt(comparison.get('median_delta_vs_frozen_protocol101'))} | "
            f"{_fmt(item['median_profit_factor'])} | {_fmt(item['median_stress_0_10_total_pnl'])} |"
        )
    lines.extend(
        [
            "",
            "## Source Audit",
            "",
            "```json",
            json.dumps(payload["combined_source_audit"], indent=2, sort_keys=True),
            "```",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'protocol175_model_trades.csv'}`",
            f"- Baseline trades: `{path.parent / 'strict_one_account_baseline_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
