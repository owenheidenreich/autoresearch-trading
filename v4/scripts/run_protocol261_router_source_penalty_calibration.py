"""EXP_ROUTER_SOURCE_PENALTY_CALIBRATION_V1.

Historically Protocol261. This calibrates the Protocol260 router without
retraining it. Protocol260 still lost to the premium-blend stream on recent
2026 because some Protocol101 substitutions blocked better challenger trades.

This experiment selects a Protocol101 logit penalty on each fold's validation
split only, then applies that frozen penalty to the reported test splits. The
model may still choose Protocol101, but only when the validation-calibrated
source penalty says the substitution is strong enough.

No paid data is downloaded and no broker endpoint is called.
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

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol165_full_action_space_policy import (
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    FOLDS,
    STARTING_CASH,
    aggregate,
    load_protocol101_baselines,
    reported_slices,
    smoke_folds,
)
import v4.scripts.run_protocol249_entry_quality_calibrator as p249
import v4.scripts.run_protocol254_policy_router_union_slot_aware as p254
import v4.scripts.run_protocol255_policy_router_history_features as p255
import v4.scripts.run_protocol260_policy_router_reliability_priors as p260


ROLE_LABEL = "EXP_ROUTER_SOURCE_PENALTY_CALIBRATION_V1"
HISTORICAL_ID = "Protocol261"
CANDIDATE_LABEL = "CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1"
DEFAULT_INPUT = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_257_complete_router_proposal_stream/complete_router_proposal_stream.csv"
)
DEFAULT_MODEL_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_260_policy_router_reliability_priors/model_artifacts")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration")
MODEL_SEEDS = [1, 2, 3, 4, 5]
PENALTY_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-smoke-sessions", type=int, default=3)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_router_globals()
    proposals = load_protocol260_proposals(args.input)
    if args.smoke:
        proposals = p254.smoke_frame(proposals, max_sessions=int(args.max_smoke_sessions))
    events = p254.build_router_events(proposals)
    oracle_summary = p254.add_oracle_actions(events)
    folds = smoke_folds(events) if args.smoke else FOLDS

    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    oracle_trades: list[dict[str, Any]] = []
    protocol101_trades: list[dict[str, Any]] = []
    challenger_trades: list[dict[str, Any]] = []
    penalty_selections: list[dict[str, Any]] = []
    for fold in folds:
        print(json.dumps({"stage": "fold_start", "fold": fold["name"]}), flush=True)
        fold_events = p254.events_for_fold(events, fold["name"])
        validation_events = [event for event in fold_events if event["split"] == fold["validation_split"]]
        if not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": fold["name"], "seed": int(seed)}), flush=True)
            seed_validation = [event for event in validation_events if int(event["seed"]) == int(seed)]
            if not seed_validation:
                fold_results.append({"fold": fold["name"], "seed": int(seed), "skipped": True, "reason": "missing_seed_validation", "splits": {}})
                continue
            model, scaler, manifest, threshold = load_router_artifact(args.model_root, str(fold["name"]), int(seed))
            selection = select_penalty(
                seed_validation,
                model,
                scaler,
                threshold=float(threshold),
                starting_cash=float(args.starting_cash),
            )
            penalty = float(selection["penalty"])
            penalty_selections.append({"fold": fold["name"], "seed": int(seed), **selection})
            result = {"fold": fold["name"], "seed": int(seed), "threshold": float(threshold), "source_penalty": penalty, "splits": {}}
            for split_name, split_events in reported_slices(fold_events, fold).items():
                split_seed_events = [event for event in split_events if int(event["seed"]) == int(seed)]
                base = simulate_router_source_penalty(
                    split_seed_events,
                    model,
                    scaler,
                    threshold=float(threshold),
                    source_penalty=penalty,
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="router_source_penalty_calibrated",
                )
                stress10 = simulate_router_source_penalty(
                    split_seed_events,
                    model,
                    scaler,
                    threshold=float(threshold),
                    source_penalty=penalty,
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                    strategy="router_source_penalty_calibrated_stress10",
                )
                stress25 = simulate_router_source_penalty(
                    split_seed_events,
                    model,
                    scaler,
                    threshold=float(threshold),
                    source_penalty=penalty,
                    slippage_per_side=0.25,
                    starting_cash=float(args.starting_cash),
                    strategy="router_source_penalty_calibrated_stress25",
                )
                oracle = p254.simulate_oracle(split_seed_events, starting_cash=float(args.starting_cash), strategy="union_oracle")
                protocol101 = p254.simulate_policy_only(
                    split_seed_events,
                    "protocol101",
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="protocol101_from_union_stream",
                )
                challenger = p254.simulate_policy_only(
                    split_seed_events,
                    "challenger",
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="premium_blend_from_union_stream",
                )
                result["splits"][split_name] = {
                    "model": p254.add_router_metrics(base.summary, base.trades),
                    "model_stress_0_10": p254.add_router_metrics(stress10.summary, stress10.trades),
                    "model_stress_0_25": p254.add_router_metrics(stress25.summary, stress25.trades),
                    "union_oracle": p254.add_router_metrics(oracle.summary, oracle.trades),
                    "protocol101_stream": p254.add_router_metrics(protocol101.summary, protocol101.trades),
                    "challenger_stream": p254.add_router_metrics(challenger.summary, challenger.trades),
                    "delta_vs_protocol101_stream": float(base.summary["total_pnl"] - protocol101.summary["total_pnl"]),
                    "delta_vs_challenger_stream": float(base.summary["total_pnl"] - challenger.summary["total_pnl"]),
                    "oracle_gap": float(oracle.summary["total_pnl"] - base.summary["total_pnl"]),
                }
                model_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in base.trades)
                oracle_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in oracle.trades)
                protocol101_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in protocol101.trades)
                challenger_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in challenger.trades)
            fold_results.append(result)
            print(json.dumps({"stage": "seed_done", "fold": fold["name"], "seed": int(seed), "penalty": penalty}), flush=True)

    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    aggregate_payload = p249.add_required_split_and_seed_checks(aggregate(fold_results, frozen_protocol101), required_seed_count=len(args.seeds))
    model_frame = pd.DataFrame(model_trades)
    oracle_frame = pd.DataFrame(oracle_trades)
    p101_frame = pd.DataFrame(protocol101_trades)
    challenger_frame = pd.DataFrame(challenger_trades)
    p254.write_frame(model_frame, args.out_dir / "model_trades.csv")
    p254.write_frame(oracle_frame, args.out_dir / "union_oracle_trades.csv")
    p254.write_frame(p101_frame, args.out_dir / "protocol101_stream_trades.csv")
    p254.write_frame(challenger_frame, args.out_dir / "challenger_stream_trades.csv")
    (args.out_dir / "penalty_selections.json").write_text(json.dumps(penalty_selections, indent=2, sort_keys=True, default=str) + "\n")
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / validation-calibrated router source penalty",
        "changes_paper_default": False,
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_to_beat": "PAPER_DEFAULT_PROTOCOL101 strict serial replay and premium-blend stream",
        "data_used": str(args.input),
        "source_model_artifact": str(args.model_root),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "feature_columns": p254.FEATURE_COLUMNS,
        "penalty_grid": PENALTY_GRID,
        "penalty_selections": penalty_selections,
        "event_summary": p254.event_summary(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate_payload,
        "frozen_protocol101_baselines": frozen_protocol101,
        "router_summary": p254.router_summary(fold_results),
        "trade_profile": {
            "model": p254.trade_profile(model_frame),
            "union_oracle": p254.trade_profile(oracle_frame),
            "protocol101_stream": p254.trade_profile(p101_frame),
            "challenger_stream": p254.trade_profile(challenger_frame),
        },
        "invariants": p254.serial_invariants(model_frame),
        "decision": "",
        "next_experiment": "If source penalty still loses to premium-blend recent, investigate recurrent lifecycle/position-state modeling instead of more entry-stream routing.",
    }
    payload["decision"] = p254.decide(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    p254.write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        p254.append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def configure_router_globals() -> None:
    p254.ROLE_LABEL = ROLE_LABEL
    p254.HISTORICAL_ID = HISTORICAL_ID
    p254.CANDIDATE_LABEL = CANDIDATE_LABEL
    p254.DEFAULT_INPUT = DEFAULT_INPUT
    p254.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    p254.FEATURE_COLUMNS = list(dict.fromkeys([*p254.FEATURE_COLUMNS, *p255.HISTORY_FEATURES, *p260.PRIOR_FEATURES]))


def load_protocol260_proposals(path: Path) -> pd.DataFrame:
    active_columns = list(p254.FEATURE_COLUMNS)
    base_columns = [
        column
        for column in active_columns
        if column not in set(p255.HISTORY_FEATURES) and column not in set(p260.PRIOR_FEATURES)
    ]
    original_loader = p254.load_router_proposals
    p254.FEATURE_COLUMNS = base_columns
    try:
        base = original_loader(path)
    finally:
        p254.FEATURE_COLUMNS = active_columns
    return p260.add_causal_reliability_priors(p260.p258.add_fold_aware_history_features(base))


def load_router_artifact(root: Path, fold_name: str, seed: int) -> tuple[p254.RouterEventPolicy, FeatureScaler, dict[str, Any], float]:
    artifact_dir = root / fold_name / f"seed_{seed}"
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    model = p254.RouterEventPolicy(
        input_dim=len(manifest["feature_columns"]),
        hidden_dim=int(manifest.get("config", {}).get("hidden_dim", 96)),
    )
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
    model.eval()
    scaler = scaler_from_json(artifact_dir / "scaler.json")
    threshold = float(manifest["threshold_selection"]["threshold"])
    return model, scaler, manifest, threshold


def scaler_from_json(path: Path) -> FeatureScaler:
    payload = json.loads(path.read_text())
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def select_penalty(
    events: list[dict[str, Any]],
    model: p254.RouterEventPolicy,
    scaler: FeatureScaler,
    *,
    threshold: float,
    starting_cash: float,
) -> dict[str, Any]:
    protocol101 = p254.simulate_policy_only(events, "protocol101", slippage_per_side=0.0, starting_cash=starting_cash, strategy="protocol101_validation")
    challenger = p254.simulate_policy_only(events, "challenger", slippage_per_side=0.0, starting_cash=starting_cash, strategy="challenger_validation")
    best_stream = max(float(protocol101.summary["total_pnl"]), float(challenger.summary["total_pnl"]))
    sweep = []
    for penalty in PENALTY_GRID:
        sim = simulate_router_source_penalty(
            events,
            model,
            scaler,
            threshold=threshold,
            source_penalty=float(penalty),
            slippage_per_side=0.0,
            starting_cash=starting_cash,
            strategy="source_penalty_validation",
        )
        stress = simulate_router_source_penalty(
            events,
            model,
            scaler,
            threshold=threshold,
            source_penalty=float(penalty),
            slippage_per_side=0.10,
            starting_cash=starting_cash,
            strategy="source_penalty_validation_stress10",
        )
        sweep.append(
            {
                "penalty": float(penalty),
                "model": p254.add_router_metrics(sim.summary, sim.trades),
                "model_stress_0_10": p254.add_router_metrics(stress.summary, stress.trades),
                "protocol101_stream": p254.add_router_metrics(protocol101.summary, protocol101.trades),
                "challenger_stream": p254.add_router_metrics(challenger.summary, challenger.trades),
                "delta_vs_best_stream": float(sim.summary["total_pnl"] - best_stream),
            }
        )
    best = max(
        sweep,
        key=lambda row: (
            row["delta_vs_best_stream"],
            row["model_stress_0_10"]["total_pnl"],
            row["model"]["profit_factor"],
            -row["penalty"],
        ),
    )
    return {
        "penalty": float(best["penalty"]),
        "objective": "validation-only: delta vs best available stream, then stress/PF, prefer smaller source penalty on ties",
        "selected": best,
        "sweep": sweep,
    }


def simulate_router_source_penalty(
    events: list[dict[str, Any]],
    model: p254.RouterEventPolicy,
    scaler: FeatureScaler,
    *,
    threshold: float,
    source_penalty: float,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    trades: list[dict[str, Any]] = []
    round_trip_slippage = float(slippage_per_side) * 2.0 * p254.CONTRACT_MULTIPLIER
    for _, session_events in p254.group_events(events).items():
        equity = float(starting_cash)
        open_until: pd.Timestamp | None = None
        for event in sorted(session_events, key=lambda item: item["decision_dt"]):
            if open_until is not None and event["decision_dt"] < open_until:
                continue
            action, margin = predict_event_action_source_penalty(event, model, scaler, source_penalty=float(source_penalty))
            if action <= 0 or margin < threshold:
                continue
            row = event["candidates"].iloc[action - 1]
            if float(row["entry_premium"]) > equity:
                continue
            trade = p254.trade_from_proposal(row, score=margin, threshold=threshold, slippage_per_side=slippage_per_side, equity=equity, strategy=strategy)
            trade["source_penalty"] = float(source_penalty)
            trade["pnl"] = float(row["candidate_pnl"]) - round_trip_slippage
            equity += trade["pnl"]
            trade["account_equity_after"] = equity
            trades.append(trade)
            open_until = pd.Timestamp(row["candidate_exit_dt"])
    return p254.simulation_result(trades, events, skipped={}, starting_cash=starting_cash, strategy=strategy)


def predict_event_action_source_penalty(
    event: dict[str, Any],
    model: p254.RouterEventPolicy,
    scaler: FeatureScaler,
    *,
    source_penalty: float,
) -> tuple[int, float]:
    x, mask, _, _, _ = p254.event_tensors([event], scaler)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(mask)).cpu().numpy()[0]
    wait = float(logits[0])
    candidate_logits = logits[1:].copy()
    candidates = event["candidates"].reset_index(drop=True)
    for idx, row in candidates.iterrows():
        if idx < len(candidate_logits) and str(row["policy"]) == "protocol101":
            candidate_logits[idx] -= float(source_penalty)
    action = int(np.argmax(candidate_logits)) + 1
    margin = float(candidate_logits[action - 1] - wait)
    if action > len(candidates):
        return 0, margin
    return action, margin


if __name__ == "__main__":
    raise SystemExit(main())
