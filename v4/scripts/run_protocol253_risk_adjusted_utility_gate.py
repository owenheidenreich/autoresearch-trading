"""EXP_RISK_ADJUSTED_UTILITY_GATE_V1.

Historically Protocol253. This experiment keeps the premium-leaning blended
utility challenger's entry timing and selected contract fixed, then trains a
second-stage neural gate with a risk-adjusted utility target.

The purpose is to test whether the challenger can keep its broad directional
capture while rejecting trades that create the poor qualities seen in the
failure-surface audit: low win rate, hard-stop clustering, and drawdown. This
does not change PAPER_DEFAULT_PROTOCOL101, download data, or call a broker.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol165_full_action_space_policy import (
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    FOLDS,
    STARTING_CASH,
    aggregate,
    fmt,
    load_dataset,
    load_protocol101_baselines,
    reported_slices,
    smoke_folds,
)
from v4.scripts.run_protocol172_full_action_value_policy import build_events, summarize_events
import v4.scripts.run_protocol249_entry_quality_calibrator as p249


ROLE_LABEL = "EXP_RISK_ADJUSTED_UTILITY_GATE_V1"
HISTORICAL_ID = "Protocol253"
CANDIDATE_LABEL = "CHALLENGER_RISK_ADJUSTED_PREMIUM_BLEND_GATE_V1"
BASELINE_CHALLENGER_LABEL = "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1"
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_253_risk_adjusted_utility_gate")
MODEL_SEEDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class RiskAdjustedConfig:
    epochs: int = 10
    batch_size: int = 512
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_scale: float = 300.0
    target_clip: float = 1500.0
    min_validation_trades: int = 10
    hard_stop_penalty: float = 350.0
    large_loss_penalty: float = 0.50
    downside_penalty: float = 0.25
    premium_burden_penalty: float = 0.012
    premium_return_bonus: float = 150.0
    churn_loss_penalty: float = 150.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-smoke-sessions", type=int, default=3)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    if args.smoke:
        events = p249.smoke_events(events, max_sessions=int(args.max_smoke_sessions))
    folds = smoke_folds(events) if args.smoke else FOLDS
    config = RiskAdjustedConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
    )

    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    base_trades: list[dict[str, Any]] = []
    train_counts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for fold in folds:
        print(json.dumps({"stage": "fold_start", "fold": fold["name"]}), flush=True)
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_events = [event for event in events if event["split"] == fold["validation_split"]]
        if not train_events or not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": fold["name"], "seed": int(seed)}), flush=True)
            try:
                base = p249.load_base_artifact(fold["name"], int(seed), dataset)
            except FileNotFoundError as exc:
                failures.append({"fold": fold["name"], "seed": int(seed), "reason": str(exc)})
                continue
            train_pred = p249.predict_base_actions(train_events, base)
            validation_pred = p249.predict_base_actions(validation_events, base)
            train_opportunities = collect_risk_opportunities(train_events, train_pred, base)
            validation_opportunities = collect_risk_opportunities(validation_events, validation_pred, base)
            train_counts.append(
                {
                    "fold": fold["name"],
                    "seed": int(seed),
                    "train_opportunities": int(len(train_opportunities)),
                    "validation_opportunities": int(len(validation_opportunities)),
                }
            )
            if len(train_opportunities) < 20 or len(validation_opportunities) < 5:
                fold_results.append(
                    {
                        "fold": fold["name"],
                        "seed": int(seed),
                        "skipped": True,
                        "reason": "insufficient_opportunities",
                        "splits": {},
                    }
                )
                continue

            feature_columns = p249.calibrator_feature_columns(base.feature_columns)
            p249.assert_no_leakage_features(feature_columns)
            model, scaler, history = train_risk_gate(
                train_opportunities,
                validation_opportunities,
                seed=int(seed),
                feature_columns=feature_columns,
                config=config,
            )
            threshold = select_risk_threshold(
                validation_events,
                validation_pred,
                base,
                model,
                scaler,
                feature_columns=feature_columns,
                config=config,
                starting_cash=float(args.starting_cash),
            )

            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "role_label": ROLE_LABEL,
                        "historical_protocol": HISTORICAL_ID,
                        "fold": fold["name"],
                        "seed": int(seed),
                        "base_candidate": BASELINE_CHALLENGER_LABEL,
                        "base_artifact_dir": str(base.artifact_dir),
                        "base_threshold": float(base.threshold),
                        "feature_columns": feature_columns,
                        "config": asdict(config),
                        "threshold_selection": threshold,
                        "history": history,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

            result = {"fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"]), "splits": {}}
            for split_name, split_events in reported_slices(events, fold).items():
                split_pred = p249.predict_base_actions(split_events, base)
                scores = p249.precompute_gate_scores(split_events, split_pred, base, model, scaler, feature_columns)
                model_base = p249.simulate_calibrated(
                    split_events,
                    split_pred,
                    base,
                    model,
                    scaler,
                    gate_threshold=float(threshold["threshold"]),
                    feature_columns=feature_columns,
                    gate_scores_by_index=scores,
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="risk_adjusted_utility_gate",
                )
                model_10 = p249.simulate_calibrated(
                    split_events,
                    split_pred,
                    base,
                    model,
                    scaler,
                    gate_threshold=float(threshold["threshold"]),
                    feature_columns=feature_columns,
                    gate_scores_by_index=scores,
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                    strategy="risk_adjusted_utility_gate_stress10",
                )
                model_25 = p249.simulate_calibrated(
                    split_events,
                    split_pred,
                    base,
                    model,
                    scaler,
                    gate_threshold=float(threshold["threshold"]),
                    feature_columns=feature_columns,
                    gate_scores_by_index=scores,
                    slippage_per_side=0.25,
                    starting_cash=float(args.starting_cash),
                    strategy="risk_adjusted_utility_gate_stress25",
                )
                base_result = p249.simulate_base_from_predictions(
                    split_events,
                    split_pred,
                    base,
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="premium_blend_base",
                )
                result["splits"][split_name] = {
                    "model": p249.add_quality_metrics(model_base.summary, model_base.trades),
                    "model_stress_0_10": p249.add_quality_metrics(model_10.summary, model_10.trades),
                    "model_stress_0_25": p249.add_quality_metrics(model_25.summary, model_25.trades),
                    "base_challenger": p249.add_quality_metrics(base_result.summary, base_result.trades),
                    "quality_delta_vs_base": p249.quality_delta(model_base.summary, base_result.summary),
                    "risk_objective": risk_objective(model_base.summary, model_base.trades),
                    "base_risk_objective": risk_objective(base_result.summary, base_result.trades),
                }
                model_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in model_base.trades)
                base_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in base_result.trades)
            fold_results.append(result)
            print(json.dumps({"stage": "seed_done", "fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"])}), flush=True)

    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    aggregate_payload = aggregate(fold_results, frozen_protocol101)
    aggregate_payload = p249.add_required_split_and_seed_checks(aggregate_payload, required_seed_count=len(MODEL_SEEDS))
    model_frame = pd.DataFrame(model_trades)
    base_frame = pd.DataFrame(base_trades)
    if not model_frame.empty:
        model_frame.to_csv(args.out_dir / "model_trades.csv", index=False)
    if not base_frame.empty:
        base_frame.to_csv(args.out_dir / "base_challenger_trades.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / risk-adjusted model gate",
        "changes_paper_default": False,
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_challenger_label": BASELINE_CHALLENGER_LABEL,
        "baseline_to_beat": "PAPER_DEFAULT_PROTOCOL101 strict one-account serial replay",
        "data_used": str(args.dataset),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "event_summary": summarize_events(events),
        "fold_results": fold_results,
        "aggregate": aggregate_payload,
        "frozen_protocol101_baselines": frozen_protocol101,
        "train_counts": train_counts,
        "artifact_failures": failures,
        "trade_profile": {
            "model": p249.trade_profile(model_frame),
            "base_challenger": p249.trade_profile(base_frame),
        },
        "quality_summary": p249.quality_summary(fold_results),
        "risk_summary": risk_summary(fold_results),
        "decision": "",
        "next_experiment": "Stop and review: this is the fifth challenger-improvement hypothesis in the current loop.",
    }
    payload["decision"] = decide(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def collect_risk_opportunities(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    base: p249.BaseArtifact,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event, prediction in zip(events, predictions):
        if not prediction["valid_action"] or float(prediction["score"]) < base.threshold:
            continue
        action = int(prediction["action"])
        if action <= 0 or action > len(event["candidates"]):
            continue
        record = p249.opportunity_record(event, event["candidates"].iloc[action - 1], prediction, base)
        if record is None:
            continue
        row = event["candidates"].iloc[action - 1]
        record["exit_reason"] = str(row.get("exit_reason", ""))
        record["duration_minutes"] = (
            (pd.Timestamp(row["candidate_exit_dt"]) - pd.Timestamp(event["decision_dt"])).total_seconds() / 60.0
        )
        record["risk_utility"] = risk_adjusted_utility(record, config=None)
        rows.append(record)
    return pd.DataFrame(rows)


def risk_adjusted_utility(record: dict[str, Any], config: RiskAdjustedConfig | None) -> float:
    cfg = config or RiskAdjustedConfig()
    pnl = p249.finite(record.get("candidate_pnl"), 0.0)
    premium = max(p249.finite(record.get("entry_premium"), 0.0), 1.0)
    downside = max(-pnl, 0.0)
    large_loss = max(-pnl - 400.0, 0.0)
    premium_return = float(np.clip(pnl / premium, -2.0, 3.0))
    utility = pnl
    utility += cfg.premium_return_bonus * premium_return
    utility -= cfg.downside_penalty * downside
    utility -= cfg.large_loss_penalty * large_loss
    utility -= cfg.premium_burden_penalty * premium
    if str(record.get("exit_reason", "")).lower() == "hard_stop":
        utility -= cfg.hard_stop_penalty
    if pnl < 0.0 and float(record.get("duration_minutes", 0.0)) <= 5.0:
        utility -= cfg.churn_loss_penalty
    return float(utility)


def train_risk_gate(
    train_opportunities: pd.DataFrame,
    validation_opportunities: pd.DataFrame,
    *,
    seed: int,
    feature_columns: list[str],
    config: RiskAdjustedConfig,
) -> tuple[p249.EntryQualityCalibrator, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_train_raw, y_train, w_train = tensors(train_opportunities, feature_columns, config, scaler=None)
    scaler = FeatureScaler.fit(x_train_raw)
    x_train, y_train, w_train = tensors(train_opportunities, feature_columns, config, scaler=scaler)
    x_val, y_val, w_val = tensors(validation_opportunities, feature_columns, config, scaler=scaler)
    model = p249.EntryQualityCalibrator(input_dim=len(feature_columns), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(w_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, by, bw in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = nn.functional.huber_loss(pred, by, delta=1.0, reduction="none")
            loss = (loss * bw).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.from_numpy(x_val))
            val_loss = nn.functional.huber_loss(pred_val, torch.from_numpy(y_val), delta=1.0, reduction="none")
            val_loss = float((val_loss * torch.from_numpy(w_val)).mean().detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "is_best": val_loss <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def tensors(
    opportunities: pd.DataFrame,
    feature_columns: list[str],
    config: RiskAdjustedConfig,
    *,
    scaler: FeatureScaler | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = opportunities[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y_raw = np.clip(
        pd.to_numeric(opportunities["risk_utility"], errors="coerce").to_numpy(dtype=np.float32),
        -config.target_clip,
        config.target_clip,
    )
    y = (y_raw / float(config.target_scale)).astype(np.float32)
    weights = (1.0 + np.minimum(np.abs(y_raw) / 300.0, 5.0)).astype(np.float32)
    if scaler is not None:
        x = scaler.transform(x)
    return x.astype(np.float32), y, weights


def select_risk_threshold(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    base: p249.BaseArtifact,
    model: p249.EntryQualityCalibrator,
    scaler: FeatureScaler,
    *,
    feature_columns: list[str],
    config: RiskAdjustedConfig,
    starting_cash: float,
) -> dict[str, Any]:
    opportunities = collect_risk_opportunities(events, predictions, base)
    scores = p249.calibrator_scores(opportunities, model, scaler, feature_columns)
    finite_scores = scores[np.isfinite(scores)]
    thresholds = [float("-inf"), float("inf")]
    if len(finite_scores):
        thresholds.extend(np.quantile(finite_scores, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist())
        thresholds.append(float(finite_scores.min()) - 1e-3)
    thresholds = sorted(set(float(item) for item in thresholds))
    gate_scores_by_index = p249.precompute_gate_scores(events, predictions, base, model, scaler, feature_columns)
    sweep = []
    for threshold in thresholds:
        sim = p249.simulate_calibrated(
            events,
            predictions,
            base,
            model,
            scaler,
            gate_threshold=float(threshold),
            feature_columns=feature_columns,
            gate_scores_by_index=gate_scores_by_index,
            slippage_per_side=0.0,
            starting_cash=starting_cash,
            strategy="risk_adjusted_validation",
        )
        stress = p249.simulate_calibrated(
            events,
            predictions,
            base,
            model,
            scaler,
            gate_threshold=float(threshold),
            feature_columns=feature_columns,
            gate_scores_by_index=gate_scores_by_index,
            slippage_per_side=0.10,
            starting_cash=starting_cash,
            strategy="risk_adjusted_validation_stress10",
        )
        sweep.append(
            {
                "threshold": float(threshold),
                "model": p249.add_quality_metrics(sim.summary, sim.trades),
                "model_stress_0_10": p249.add_quality_metrics(stress.summary, stress.trades),
                "risk_objective": risk_objective(sim.summary, sim.trades),
            }
        )
    eligible = [
        row
        for row in sweep
        if row["model"]["trades"] >= config.min_validation_trades
        and row["model_stress_0_10"]["total_pnl"] > 0.0
    ]
    pool = eligible if eligible else sweep
    best = max(
        pool,
        key=lambda row: (
            row["risk_objective"],
            row["model_stress_0_10"]["total_pnl"],
            row["model"]["profit_factor"],
            row["model"]["win_rate"],
            row["model"]["total_pnl"],
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "objective": "validation-only risk objective: PnL - drawdown/worst-day penalties, then stress/PF/win",
        "selected": best,
        "sweep": sweep,
    }


def risk_objective(summary: dict[str, Any], trades: list[dict[str, Any]]) -> float:
    total_pnl = p249.finite(summary.get("total_pnl"), 0.0)
    max_drawdown = abs(p249.finite(summary.get("max_drawdown"), 0.0))
    worst_day = worst_day_pnl(trades)
    hard_stops = sum(1 for trade in trades if str(trade.get("exit_reason", "")).lower() == "hard_stop")
    trade_count = max(int(summary.get("trades", 0)), 1)
    return float(total_pnl - 0.65 * max_drawdown - 0.80 * max(-worst_day, 0.0) - 30.0 * hard_stops / trade_count)


def worst_day_pnl(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    frame = pd.DataFrame(trades)
    if "decision_time" not in frame.columns:
        return 0.0
    frame["day"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce").dt.date
    daily = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0).groupby(frame["day"]).sum()
    return float(daily.min()) if len(daily) else 0.0


def risk_summary(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        rows = [result["splits"][split] for result in fold_results if not result.get("skipped") and split in result.get("splits", {})]
        if not rows:
            continue
        out[split] = {
            "median_risk_objective": float(np.median([row["risk_objective"] for row in rows])),
            "median_base_risk_objective": float(np.median([row["base_risk_objective"] for row in rows])),
            "median_delta_risk_objective": float(np.median([row["risk_objective"] - row["base_risk_objective"] for row in rows])),
        }
    return out


def decide(payload: dict[str, Any]) -> str:
    aggregate_payload = payload.get("aggregate", {})
    quality = payload.get("quality_summary", {})
    risk = payload.get("risk_summary", {})
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    risk_improves = all(risk.get(split, {}).get("median_delta_risk_objective", -1.0) > 0.0 for split in required if split in risk)
    quality_improves = all(quality.get(split, {}).get("improved_dimension_count", 0) >= 2 for split in required if split in quality)
    if aggregate_payload.get("promotion_ready") and risk_improves and quality_improves:
        return "research_candidate_risk_adjusted_gate_clears_protocol101_and_improves_quality"
    if risk_improves or any(item.get("median_delta_pnl_vs_base", 0.0) > 0.0 for item in quality.values()):
        return "research_only_risk_adjusted_gate_partial_improvement_not_replacement"
    return "rejected_risk_adjusted_gate_no_clear_improvement"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Baseline challenger: {payload['baseline_challenger_label']}",
        f"Data used: `{payload['data_used']}`",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Aggregate Vs Protocol101",
        "",
        "| split | seeds | median PnL | Protocol101 | delta | PF | stress 0.10 | stress 0.25 | trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        item = payload["aggregate"].get(split, {})
        if not item or item.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | "
            f"{fmt(item.get('frozen_protocol101_total_pnl'))} | {fmt(item.get('median_delta_vs_frozen_protocol101'))} | "
            f"{fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} | "
            f"{fmt(item['median_stress_0_25_total_pnl'])} | {fmt(item['median_trades'])} |"
        )
    lines.extend(["", "## Quality Delta Vs Current Challenger", ""])
    for split, item in payload.get("quality_summary", {}).items():
        lines.append(f"- {split}: {item['improved_dimension_count']} improved dimensions, median PnL delta vs base {fmt(item['median_delta_pnl_vs_base'])}")
    lines.extend(["", "## Risk Objective Delta Vs Current Challenger", ""])
    for split, item in payload.get("risk_summary", {}).items():
        lines.append(
            f"- {split}: median risk objective delta {fmt(item['median_delta_risk_objective'])} "
            f"(model {fmt(item['median_risk_objective'])}, base {fmt(item['median_base_risk_objective'])})"
        )
    lines.extend(
        [
            "",
            "## Trade Profile",
            "",
            f"- Model: {payload['trade_profile']['model']}",
            f"- Base challenger: {payload['trade_profile']['base_challenger']}",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'model_trades.csv'}`",
            f"- Base challenger trades: `{path.parent / 'base_challenger_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text("# v4 Research Ledger\n\n")
    entry = (
        f"\n## {HISTORICAL_ID} - {ROLE_LABEL}\n\n"
        f"- What: risk-adjusted utility gate over `{BASELINE_CHALLENGER_LABEL}`.\n"
        f"- Paper default changed: no.\n"
        f"- Paid data downloaded: no.\n"
        f"- Broker endpoint called: no.\n"
        f"- Decision: `{payload['decision']}`.\n"
        f"- Report: `{out_dir / 'report.md'}`.\n"
    )
    with ledger.open("a") as handle:
        handle.write(entry)


if __name__ == "__main__":
    raise SystemExit(main())
