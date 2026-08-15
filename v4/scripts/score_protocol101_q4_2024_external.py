"""Score frozen Protocol 101 on Q4 2024 external candidate outcomes.

This is a temporal-regime stress audit, not a chronological promotion gate:
the frozen Protocol 101 fold-3 artifacts were trained on 2025 data and are
being applied backward to already-collected Q4 2024 data. The purpose is to
test whether the event-history policy collapses on a different market block
before asking for more paid historical data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.serial_opportunity import (
    _add_entry_features,
    _join_protocol081_selected,
    _sanitize_dataset,
    strict_serial_baseline,
)
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol097_sequential_event_policy import (
    EventSetPolicy,
    _candidate_frame_from_events,
    build_events,
    simulate_event_policy,
)
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS, add_causal_history_features


DEFAULT_PROTOCOL081_EXITS = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_106_q4_2024_external_protocol081_exits/"
    "selected_trades_sequence_exits.json"
)
DEFAULT_LIFECYCLE_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_105_q4_2024_external_lifecycle_sequence/"
    "protocol054_lifecycle_trades.parquet"
)
DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress")
DEFAULT_FOLD = "fold3_train_q1_q2_q3_validate_q4_test_q1_2026"
PROTOCOL_SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol081-exits", type=Path, default=DEFAULT_PROTOCOL081_EXITS)
    parser.add_argument("--lifecycle-trades", type=Path, default=DEFAULT_LIFECYCLE_TRADES)
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact-fold", default=DEFAULT_FOLD)
    parser.add_argument("--protocol-seeds", nargs="*", type=int, default=PROTOCOL_SEEDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_external_dataset(args.protocol081_exits, args.lifecycle_trades)
    dataset_path = args.out_dir / "q4_2024_external_serial_opportunity_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    events = build_events(dataset)
    add_causal_history_features(events)
    threshold_by_seed = _thresholds(args.protocol101_dir, args.artifact_fold)

    seed_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for seed in args.protocol_seeds:
        model, scaler = _load_protocol101_artifact(args.protocol101_dir, args.artifact_fold, int(seed))
        seed_events = [event for event in events if int(event["seed"]) == int(seed)]
        candidate_source = _candidate_frame_from_events(seed_events)
        threshold = float(threshold_by_seed[int(seed)])
        base = simulate_event_policy(
            seed_events,
            model,
            scaler,
            threshold=threshold,
            slippage_per_side=0.0,
            strategy="protocol101_q4_2024_external",
            feature_columns=FEATURE_COLUMNS,
        )
        stress10 = simulate_event_policy(
            seed_events,
            model,
            scaler,
            threshold=threshold,
            slippage_per_side=0.10,
            strategy="protocol101_q4_2024_external_stress10",
            feature_columns=FEATURE_COLUMNS,
        )
        stress25 = simulate_event_policy(
            seed_events,
            model,
            scaler,
            threshold=threshold,
            slippage_per_side=0.25,
            strategy="protocol101_q4_2024_external_stress25",
            feature_columns=FEATURE_COLUMNS,
        )
        baseline = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.0)
        baseline10 = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.10)
        seed_rows.append(
            {
                "seed": int(seed),
                "threshold": threshold,
                "model": base.summary,
                "model_stress_0_10": stress10.summary,
                "model_stress_0_25": stress25.summary,
                "strict_serial_baseline": baseline.summary,
                "strict_serial_baseline_stress_0_10": baseline10.summary,
                "margin_vs_baseline": float(base.summary["total_pnl"] - baseline.summary["total_pnl"]),
            }
        )
        for row in base.trades:
            trade_rows.append({"seed": int(seed), **row})
        for row in baseline.trades:
            baseline_rows.append({"seed": int(seed), **row})

    aggregate = _aggregate(seed_rows)
    payload = {
        "protocol": "107_protocol101_q4_2024_external_stress",
        "paid_data_downloaded": False,
        "live_orders": False,
        "model_training": False,
        "audit_type": "temporal_regime_stress_not_chronological_promotion_gate",
        "source_protocol081_exits": str(args.protocol081_exits),
        "source_lifecycle_trades": str(args.lifecycle_trades),
        "source_protocol101_dir": str(args.protocol101_dir),
        "artifact_fold": str(args.artifact_fold),
        "dataset_path": str(dataset_path),
        "dataset_rows": int(len(dataset)),
        "event_count": int(len(events)),
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "decision": _decision(aggregate),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out_dir / "serial_policy_trades.json").write_text(json.dumps(trade_rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out_dir / "strict_serial_baseline_trades.json").write_text(json.dumps(baseline_rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "aggregate": aggregate}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def _build_external_dataset(selected_path: Path, lifecycle_trades_path: Path) -> pd.DataFrame:
    selected = pd.read_json(selected_path)
    trades = pd.read_parquet(lifecycle_trades_path)
    dataset = _join_protocol081_selected(selected, trades)
    dataset = _add_entry_features(dataset)
    dataset = _sanitize_dataset(dataset)
    dataset["split"] = "q4_2024_external"
    return dataset


def _thresholds(protocol101_dir: Path, fold: str) -> dict[int, float]:
    payload = json.loads((protocol101_dir / "summary.json").read_text())
    out: dict[int, float] = {}
    for row in payload["fold_results"]:
        if row["fold"] == fold:
            out[int(row["seed"])] = float(row["threshold"])
    missing = [seed for seed in PROTOCOL_SEEDS if seed not in out]
    if missing:
        raise SystemExit(f"missing Protocol 101 thresholds for seeds: {missing}")
    return out


def _scaler_from_dict(payload: dict[str, Any]) -> FeatureScaler:
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def _load_protocol101_artifact(protocol101_dir: Path, fold: str, seed: int) -> tuple[EventSetPolicy, FeatureScaler]:
    artifact_dir = protocol101_dir / "model_artifacts" / fold / f"seed_{seed}"
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    hidden_dim = int(manifest["config"]["hidden_dim"])
    feature_columns = list(manifest["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        raise SystemExit(f"Protocol 101 feature mismatch for seed {seed}")
    model = EventSetPolicy(input_dim=len(FEATURE_COLUMNS), hidden_dim=hidden_dim)
    try:
        state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(artifact_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    scaler = _scaler_from_dict(json.loads((artifact_dir / "scaler.json").read_text()))
    return model, scaler


def _aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_pnl = [float(row["model"]["total_pnl"]) for row in seed_rows]
    model_pf = [float(row["model"]["profit_factor"]) for row in seed_rows]
    trades = [float(row["model"]["trades"]) for row in seed_rows]
    baseline = [float(row["strict_serial_baseline"]["total_pnl"]) for row in seed_rows]
    stress10 = [float(row["model_stress_0_10"]["total_pnl"]) for row in seed_rows]
    stress25 = [float(row["model_stress_0_25"]["total_pnl"]) for row in seed_rows]
    margins = [float(row["margin_vs_baseline"]) for row in seed_rows]
    return {
        "median_total_pnl": float(np.median(model_pnl)),
        "median_profit_factor": float(np.median(model_pf)),
        "median_trades": float(np.median(trades)),
        "strict_serial_baseline_median_total_pnl": float(np.median(baseline)),
        "median_margin_vs_baseline": float(np.median(margins)),
        "positive_seed_fraction": float(np.mean([value > 0.0 for value in model_pnl])),
        "positive_seed_margin_fraction": float(np.mean([value > 0.0 for value in margins])),
        "median_stress_0_10_total_pnl": float(np.median(stress10)),
        "median_stress_0_25_total_pnl": float(np.median(stress25)),
    }


def _decision(aggregate: dict[str, Any]) -> str:
    if (
        aggregate["median_total_pnl"] > 0.0
        and aggregate["median_margin_vs_baseline"] > 0.0
        and aggregate["positive_seed_fraction"] >= 0.8
        and aggregate["median_stress_0_25_total_pnl"] > 0.0
    ):
        return "external_temporal_stress_passed_research_only"
    return "external_temporal_stress_failed_or_fragile"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Protocol 107: Protocol 101 Q4 2024 External Stress",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used. No model was trained.",
        "",
        "**This is a temporal-regime stress audit, not a chronological promotion gate.**",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Candidate dataset rows: `{payload['dataset_rows']}`",
        f"- Decision events: `{payload['event_count']}`",
        "",
        "## Aggregate",
        "",
        f"- Median PnL: `{agg['median_total_pnl']:.0f}`",
        f"- Strict serial baseline median PnL: `{agg['strict_serial_baseline_median_total_pnl']:.0f}`",
        f"- Median margin vs baseline: `{agg['median_margin_vs_baseline']:.0f}`",
        f"- Median PF: `{agg['median_profit_factor']:.3f}`",
        f"- Positive seed fraction: `{agg['positive_seed_fraction']:.2f}`",
        f"- Positive seed-margin fraction: `{agg['positive_seed_margin_fraction']:.2f}`",
        f"- +0.25 each-side stress median PnL: `{agg['median_stress_0_25_total_pnl']:.0f}`",
        "",
        "## Seeds",
        "",
        "| seed | model_pnl | baseline_pnl | margin | pf | trades | +0.10 | +0.25 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_rows"]:
        lines.append(
            f"| {row['seed']} | {row['model']['total_pnl']:.0f} | "
            f"{row['strict_serial_baseline']['total_pnl']:.0f} | {row['margin_vs_baseline']:.0f} | "
            f"{row['model']['profit_factor']:.3f} | {row['model']['trades']:.0f} | "
            f"{row['model_stress_0_10']['total_pnl']:.0f} | {row['model_stress_0_25']['total_pnl']:.0f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Passing this audit means the frozen event-history policy did not collapse on the already-collected pre-2025 block. It does not approve paper trading and does not replace the need for chronological broader validation.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
