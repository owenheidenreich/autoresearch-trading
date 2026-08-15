"""Protocol 173: value-policy threshold discipline replay.

This is a no-retraining falsification of Protocol172. If the model score is an
oracle-advantage estimate, negative thresholds should be suspicious: they let
the bot trade candidates the model itself scores as below wait value. Protocol
173 reloads frozen Protocol172 artifacts and replays them with validation-only
threshold selection constrained to non-negative score thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol165_full_action_space_policy import (
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    FOLDS,
    aggregate,
    fmt,
    load_dataset,
    load_protocol101_baselines,
    reported_slices,
    simulate_first_affordable,
    simulate_matched_random,
    simulate_oracle,
)
from v4.scripts.run_protocol172_full_action_value_policy import (
    CandidateValueSetPolicy,
    FullActionValueConfig,
    add_oracle_advantages,
    build_events,
    event_scores,
    simulate_model,
)


LOOP_ID = "v4_aplus_hypothesis_173_value_threshold_discipline"
DEFAULT_SOURCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_172_full_action_value_policy_3sessions")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((args.source_dir / "summary.json").read_text())
    dataset_path = args.dataset or Path(source_summary["dataset"])
    dataset = load_dataset(dataset_path)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    add_oracle_advantages(events)
    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    for fold in FOLDS:
        validation_events = [event for event in events if event["split"] == fold["validation_split"]]
        if not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_validation_events", "splits": {}})
            continue
        for seed_dir in sorted((args.source_dir / "model_artifacts" / fold["name"]).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            model, scaler, manifest = load_artifact(seed_dir)
            threshold = select_nonnegative_threshold(
                validation_events,
                model,
                scaler,
                starting_cash=float(args.starting_cash),
            )
            result = {"fold": fold["name"], "seed": seed, "threshold": float(threshold["threshold"]), "splits": {}, "source_threshold": manifest["threshold_selection"]["threshold"], "threshold_selection": threshold}
            for split_name, split_events in reported_slices(events, fold).items():
                model_base = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.0, starting_cash=float(args.starting_cash), strategy="protocol173")
                model_10 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.10, starting_cash=float(args.starting_cash), strategy="protocol173_stress10")
                model_25 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.25, starting_cash=float(args.starting_cash), strategy="protocol173_stress25")
                first = simulate_first_affordable(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                random_base = simulate_matched_random(split_events, seed=seed, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                oracle = simulate_oracle(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                result["splits"][split_name] = {
                    "model": model_base.summary,
                    "model_stress_0_10": model_10.summary,
                    "model_stress_0_25": model_25.summary,
                    "first_affordable_baseline": first.summary,
                    "matched_random_baseline": random_base.summary,
                    "full_action_oracle": oracle.summary,
                    "edge_only_baseline": {"status": "not_available_no_surface_edge_feature"},
                }
                model_trades.extend({**trade, "fold": fold["name"], "seed": seed, "reported_split": split_name} for trade in model_base.trades)
                baseline_trades.extend({**trade, "fold": fold["name"], "seed": seed, "reported_split": split_name} for trade in first.trades)
            fold_results.append(result)
    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    payload = {
        "protocol": "173_value_threshold_discipline",
        "source_protocol": str(args.source_dir),
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "dataset": str(dataset_path),
        "fold_results": fold_results,
        "aggregate": aggregate(fold_results, frozen_protocol101),
        "frozen_protocol101_baselines": frozen_protocol101,
    }
    payload["decision"] = (
        "promote_research_candidate: Protocol173 threshold discipline beats frozen Protocol101"
        if payload["aggregate"].get("promotion_ready")
        else "research_only: nonnegative value thresholds did not clear the frozen Protocol101 promotion gate"
    )
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(model_trades).to_csv(args.out_dir / "protocol173_model_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "protocol173_first_affordable_baseline_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_artifact(seed_dir: Path) -> tuple[CandidateValueSetPolicy, FeatureScaler, dict[str, Any]]:
    manifest = json.loads((seed_dir / "manifest.json").read_text())
    scaler_payload = json.loads((seed_dir / "scaler.json").read_text())
    scaler = FeatureScaler(
        fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
        mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    model = CandidateValueSetPolicy(
        input_dim=len(manifest["feature_columns"]),
        hidden_dim=int(manifest["config"]["hidden_dim"]),
    )
    model.load_state_dict(torch.load(seed_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, scaler, manifest


def select_nonnegative_threshold(
    events: list[dict[str, Any]],
    model: CandidateValueSetPolicy,
    scaler: FeatureScaler,
    *,
    starting_cash: float,
) -> dict[str, Any]:
    scores = event_scores(events, model, scaler)
    finite_scores = scores[np.isfinite(scores)]
    positive = finite_scores[finite_scores >= 0.0]
    if len(positive):
        thresholds = sorted(set(np.quantile(positive, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist() + [0.0]))
    else:
        thresholds = [0.0, float("inf")]
    first = simulate_first_affordable(events, slippage_per_side=0.0, starting_cash=starting_cash)
    sweep = []
    for threshold in thresholds:
        base = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, starting_cash=starting_cash, strategy="protocol173_validation")
        stress = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, starting_cash=starting_cash, strategy="protocol173_validation_stress10")
        sweep.append({"threshold": float(threshold), "model": base.summary, "model_stress_0_10": stress.summary, "first_affordable_baseline": first.summary, "delta_vs_first": float(base.summary["total_pnl"] - first.summary["total_pnl"])})
    eligible = [row for row in sweep if row["model"]["trades"] >= 5 and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["model_stress_0_10"]["total_pnl"], row["delta_vs_first"], row["model"]["profit_factor"], row["model"]["trades"]))
    return {"threshold": float(best["threshold"]), "objective": "validation nonnegative score thresholds only; stress_0_10 pnl then delta vs first-affordable", "selected": best, "sweep": sweep}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol173 Value Threshold Discipline",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source protocol: `{payload['source_protocol']}`",
        f"- Dataset: `{payload['dataset']}`",
        "",
        "## Aggregate",
        "",
        "| split | seeds | median PnL | frozen Protocol101 | delta | PF | stress 0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["aggregate"].items():
        if not isinstance(item, dict) or item.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | {fmt(item.get('frozen_protocol101_total_pnl'))} | {fmt(item.get('median_delta_vs_frozen_protocol101'))} | {fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} |"
        )
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Model trades: `{path.parent / 'protocol173_model_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
