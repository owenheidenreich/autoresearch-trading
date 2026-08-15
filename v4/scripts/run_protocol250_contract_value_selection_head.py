"""EXP_CONTRACT_VALUE_SELECTION_HEAD_V1.

Historically Protocol250. This experiment keeps the frozen premium-blend
challenger's entry timing gate, but replaces its chosen contract with a
separate causal contract-value head trained on the full SPXW 0DTE action
surface.

The purpose is to test whether the challenger is often right about the market
moment, but wrong about which contract has the best executable value. It does
not change PAPER_DEFAULT_PROTOCOL101, download data, or call a broker endpoint.
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
    simulation_result,
    smoke_folds,
    trade_from_row,
)
from v4.scripts.run_protocol172_full_action_value_policy import build_events, summarize_events
import v4.scripts.run_protocol183_two_stage_full_action_policy as p183
import v4.scripts.run_protocol221_return_on_premium_policy as p221
import v4.scripts.run_protocol248_challenger_failure_surface as p248
import v4.scripts.run_protocol249_entry_quality_calibrator as p249


ROLE_LABEL = "EXP_CONTRACT_VALUE_SELECTION_HEAD_V1"
HISTORICAL_ID = "Protocol250"
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_250_contract_value_selection_head")
FEATURE_COLUMNS = p221.FEATURE_COLUMNS
MODEL_SEEDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class ContractValueConfig:
    epochs: int = 6
    batch_size: int = 8192
    hidden_dim: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_train_rows: int = 400_000
    dollar_weight: float = 0.45
    premium_weight: float = 0.55
    dollar_scale: float = 500.0
    target_clip: float = 3.0
    dollar_aux_weight: float = 0.05
    premium_aux_weight: float = 0.05
    target_aux_weight: float = 0.03
    stop_aux_weight: float = 0.03
    min_validation_trades: int = 10


class ContractValueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.dollar_head = nn.Linear(hidden_dim, 1)
        self.premium_return_head = nn.Linear(hidden_dim, 1)
        self.target_head = nn.Linear(hidden_dim, 1)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.encoder(features)
        return {
            "value": self.value_head(emb).squeeze(-1),
            "dollar": self.dollar_head(emb).squeeze(-1),
            "premium_return": self.premium_return_head(emb).squeeze(-1),
            "target": self.target_head(emb).squeeze(-1),
            "stop": self.stop_head(emb).squeeze(-1),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-train-rows", type=int, default=400_000)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    p183.set_active_feature_columns(FEATURE_COLUMNS, dataset)
    p249.assert_no_leakage_features(FEATURE_COLUMNS)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    if args.smoke:
        events = p249.smoke_events(events, max_sessions=3)
    folds = smoke_folds(events) if args.smoke else FOLDS
    config = ContractValueConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        max_train_rows=int(args.max_train_rows),
    )

    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    base_trades: list[dict[str, Any]] = []
    train_counts: list[dict[str, Any]] = []
    for fold in folds:
        print(json.dumps({"stage": "fold_start", "fold": fold["name"]}), flush=True)
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_events = [event for event in events if event["split"] == fold["validation_split"]]
        if not train_events or not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        train_frame = candidate_training_frame(train_events, config=config)
        if train_frame.empty:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "no_train_candidates", "splits": {}})
            continue
        for seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": fold["name"], "seed": int(seed), "train_rows": int(len(train_frame))}), flush=True)
            base = p249.load_base_artifact(fold["name"], int(seed), dataset)
            model, scaler, history = train_value_head(train_frame, seed=int(seed), feature_columns=FEATURE_COLUMNS, config=config)
            validation_pred = p249.predict_base_actions(validation_events, base)
            validation_scores = score_events(validation_events, model, scaler, FEATURE_COLUMNS)
            threshold = select_value_threshold(
                validation_events,
                validation_pred,
                validation_scores,
                base,
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
                        "base_candidate": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
                        "base_artifact_dir": str(base.artifact_dir),
                        "base_entry_threshold": float(base.threshold),
                        "feature_columns": FEATURE_COLUMNS,
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
            train_counts.append({"fold": fold["name"], "seed": int(seed), "train_rows": int(len(train_frame))})
            result = {"fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"]), "splits": {}}
            for split_name, split_events in reported_slices(events, fold).items():
                split_pred = p249.predict_base_actions(split_events, base)
                split_scores = score_events(split_events, model, scaler, FEATURE_COLUMNS)
                model_base = simulate_value_selection(
                    split_events,
                    split_pred,
                    split_scores,
                    base,
                    value_threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="contract_value_selection_head",
                )
                model_10 = simulate_value_selection(
                    split_events,
                    split_pred,
                    split_scores,
                    base,
                    value_threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                    strategy="contract_value_selection_head_stress10",
                )
                model_25 = simulate_value_selection(
                    split_events,
                    split_pred,
                    split_scores,
                    base,
                    value_threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    starting_cash=float(args.starting_cash),
                    strategy="contract_value_selection_head_stress25",
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
        "what_is_this": "experiment / model change",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_CONTRACT_VALUE_SELECTION_HEAD_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_challenger_label": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
        "baseline_to_beat": "PAPER_DEFAULT_PROTOCOL101 strict one-account serial replay",
        "data_used": str(args.dataset),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "utility": {
            "formula": "0.45 * pnl / 500 + 0.55 * pnl / entry_premium, with auxiliary dollar/return/target/stop heads",
            "dollar_weight": config.dollar_weight,
            "premium_weight": config.premium_weight,
            "dollar_scale": config.dollar_scale,
        },
        "feature_columns": FEATURE_COLUMNS,
        "event_summary": summarize_events(events),
        "train_counts": train_counts,
        "fold_results": fold_results,
        "aggregate": aggregate_payload,
        "quality_summary": p249.quality_summary(fold_results),
        "trade_profile": {
            "contract_value_head": p249.trade_profile(model_frame),
            "base_challenger": p249.trade_profile(base_frame),
        },
        "frozen_protocol101_baselines": frozen_protocol101,
        "decision": "",
        "next_experiment": (
            "If contract value selection fails by reducing large winners, move to lifecycle continuation. "
            "If it improves moneyness/premium efficiency but not drawdown, test risk-adjusted utility."
        ),
    }
    payload["decision"] = decide(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def candidate_training_frame(events: list[dict[str, Any]], *, config: ContractValueConfig) -> pd.DataFrame:
    frames = []
    for event in events:
        candidates = event["candidates"].copy()
        valid_exit = pd.to_datetime(candidates["candidate_exit_dt"], utc=True) > pd.Timestamp(event["decision_dt"])
        affordable = pd.to_numeric(candidates["entry_affordable_10k"], errors="coerce").fillna(0.0) >= 1.0
        frames.append(candidates[valid_exit & affordable])
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["candidate_pnl", "entry_premium"])
    if len(frame) > int(config.max_train_rows):
        frame = frame.sample(n=int(config.max_train_rows), random_state=17).sort_index()
    return frame.reset_index(drop=True)


def train_value_head(
    train_frame: pd.DataFrame,
    *,
    seed: int,
    feature_columns: list[str],
    config: ContractValueConfig,
) -> tuple[ContractValueHead, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    scaler = FeatureScaler.fit(raw)
    x, targets = value_tensors(train_frame, scaler, feature_columns, config)
    model = ContractValueHead(input_dim=len(feature_columns), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(targets["value"]),
            torch.from_numpy(targets["dollar"]),
            torch.from_numpy(targets["premium_return"]),
            torch.from_numpy(targets["target_flag"]),
            torch.from_numpy(targets["stop_flag"]),
            torch.from_numpy(targets["weights"]),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, bv, bd, br, bt, bs, bw in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss = nn.functional.huber_loss(out["value"], bv, delta=1.0, reduction="none")
            loss = loss + config.dollar_aux_weight * nn.functional.huber_loss(out["dollar"], bd, delta=1.0, reduction="none")
            loss = loss + config.premium_aux_weight * nn.functional.huber_loss(out["premium_return"], br, delta=1.0, reduction="none")
            loss = loss + config.target_aux_weight * nn.functional.binary_cross_entropy_with_logits(out["target"], bt, reduction="none")
            loss = loss + config.stop_aux_weight * nn.functional.binary_cross_entropy_with_logits(out["stop"], bs, reduction="none")
            loss = (loss * bw).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": epoch_loss, "is_best": epoch_loss <= best_loss})
    model.load_state_dict(best_state)
    return model, scaler, history


def value_tensors(
    frame: pd.DataFrame,
    scaler: FeatureScaler,
    feature_columns: list[str],
    config: ContractValueConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x = scaler.transform(raw)
    pnl = pd.to_numeric(frame["candidate_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    premium = pd.to_numeric(frame["entry_premium"], errors="coerce").replace(0.0, np.nan).to_numpy(dtype=np.float32)
    premium = np.where(np.isfinite(premium) & (premium > 0.0), premium, np.nan)
    dollar = np.clip(pnl / float(config.dollar_scale), -config.target_clip, config.target_clip).astype(np.float32)
    premium_return = np.clip(pnl / premium, -config.target_clip, config.target_clip)
    premium_return = np.nan_to_num(premium_return, nan=0.0, posinf=config.target_clip, neginf=-config.target_clip).astype(np.float32)
    total_weight = max(abs(config.dollar_weight) + abs(config.premium_weight), 1e-9)
    value = (config.dollar_weight / total_weight) * dollar + (config.premium_weight / total_weight) * premium_return
    value = np.clip(value, -config.target_clip, config.target_clip).astype(np.float32)
    reason = frame["candidate_exit_reason"].astype(str)
    target_flag = reason.eq("target").astype(np.float32).to_numpy()
    stop_flag = reason.eq("hard_stop").astype(np.float32).to_numpy()
    weights = (1.0 + np.minimum(np.abs(pnl) / 300.0, 5.0)).astype(np.float32)
    return x, {
        "value": value,
        "dollar": dollar,
        "premium_return": premium_return,
        "target_flag": target_flag,
        "stop_flag": stop_flag,
        "weights": weights,
    }


def score_events(
    events: list[dict[str, Any]],
    model: ContractValueHead,
    scaler: FeatureScaler,
    feature_columns: list[str],
) -> list[np.ndarray]:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for event in events:
            candidates = event["candidates"].head(p183.MAX_ACTION_CANDIDATES)
            raw = candidates[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            if len(raw) == 0:
                scores.append(np.asarray([], dtype=float))
                continue
            x = scaler.transform(raw)
            out = model(torch.from_numpy(x))["value"].cpu().numpy().astype(float)
            scores.append(out)
    return scores


def select_value_threshold(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scores: list[np.ndarray],
    base: p249.BaseArtifact,
    *,
    config: ContractValueConfig,
    starting_cash: float,
) -> dict[str, Any]:
    finite_scores = np.asarray([score for arr in scores for score in arr if np.isfinite(score)], dtype=float)
    thresholds = [float("-inf"), float("inf")]
    if len(finite_scores):
        thresholds.extend(np.quantile(finite_scores, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist())
        thresholds.append(float(finite_scores.min()) - 1e-3)
    thresholds = sorted(set(float(item) for item in thresholds))
    base_result = p249.simulate_base_from_predictions(
        events,
        predictions,
        base,
        slippage_per_side=0.0,
        starting_cash=starting_cash,
        strategy="premium_blend_base_validation",
    )
    sweep = []
    for threshold in thresholds:
        model_base = simulate_value_selection(
            events,
            predictions,
            scores,
            base,
            value_threshold=float(threshold),
            slippage_per_side=0.0,
            starting_cash=starting_cash,
            strategy="contract_value_validation",
        )
        stress = simulate_value_selection(
            events,
            predictions,
            scores,
            base,
            value_threshold=float(threshold),
            slippage_per_side=0.10,
            starting_cash=starting_cash,
            strategy="contract_value_validation_stress10",
        )
        sweep.append(
            {
                "threshold": float(threshold),
                "model": p249.add_quality_metrics(model_base.summary, model_base.trades),
                "model_stress_0_10": p249.add_quality_metrics(stress.summary, stress.trades),
                "base_challenger": p249.add_quality_metrics(base_result.summary, base_result.trades),
                "quality_delta_vs_base": p249.quality_delta(model_base.summary, base_result.summary),
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
            row["model_stress_0_10"]["total_pnl"],
            row["model"]["profit_factor"],
            row["model"]["avg_pnl"],
            -abs(float(row["model"].get("max_drawdown", 0.0))),
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "objective": "validation-only: stress_0_10 PnL, then PF, avg PnL, drawdown",
        "selected": best,
        "sweep": sweep,
    }


def simulate_value_selection(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scores: list[np.ndarray],
    base: p249.BaseArtifact,
    *,
    value_threshold: float,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    trades: list[dict[str, Any]] = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0, "value_gate": 0}
    ordered = sorted(enumerate(zip(events, predictions, scores)), key=lambda item: (item[1][0]["session"], item[1][0]["decision_dt"]))
    for _, (event, prediction, event_scores) in ordered:
        if open_until.get(event["session"]) is not None and event["decision_dt"] < open_until[event["session"]]:
            skipped["overlap"] += len(event["candidates"])
            continue
        if not prediction["valid_action"] or float(prediction["score"]) < base.threshold:
            skipped["threshold"] += len(event["candidates"])
            continue
        candidates = event["candidates"].head(len(event_scores)).copy()
        if candidates.empty or len(event_scores) == 0:
            skipped["invalid"] += 1
            continue
        valid_exit = pd.to_datetime(candidates["candidate_exit_dt"], utc=True) > pd.Timestamp(event["decision_dt"])
        affordable = pd.to_numeric(candidates["entry_affordable_10k"], errors="coerce").fillna(0.0) >= 1.0
        mask = (valid_exit & affordable).to_numpy(dtype=bool)
        masked_scores = np.where(mask, event_scores, -1e9)
        best_idx = int(np.argmax(masked_scores))
        best_score = float(masked_scores[best_idx])
        if best_score < value_threshold or best_score <= -1e8:
            skipped["value_gate"] += int(len(candidates))
            continue
        trade = trade_from_row(
            candidates.iloc[best_idx],
            score=best_score,
            threshold=value_threshold,
            slippage_per_side=slippage_per_side,
            equity=equity,
            strategy=strategy,
        )
        if trade is None:
            skipped["invalid"] += 1
            continue
        trade["base_score"] = float(prediction["score"])
        trade["base_threshold"] = float(base.threshold)
        trade["contract_value_score"] = best_score
        if trade["entry_premium_with_slippage"] > equity:
            skipped["unaffordable"] += 1
            continue
        trades.append(trade)
        equity += trade["pnl"]
        trades[-1]["account_equity_after"] = equity
        open_until[event["session"]] = pd.Timestamp(trade["exit_time"])
    return simulation_result(trades, events, skipped, starting_cash=starting_cash, strategy=strategy)


def decide(payload: dict[str, Any]) -> str:
    aggregate_payload = payload.get("aggregate", {})
    quality = payload.get("quality_summary", {})
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    quality_ok = all(quality.get(split, {}).get("improved_dimension_count", 0) >= 2 for split in required if split in quality)
    if aggregate_payload.get("promotion_ready") and quality_ok:
        return "research_candidate_contract_value_head_clears_protocol101_and_quality_gate"
    if aggregate_payload.get("promotion_ready"):
        return "research_only_contract_value_head_beats_protocol101_but_quality_gate_failed"
    return "rejected_contract_value_head_did_not_clear_protocol101_gate"


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
    lines.extend(["", "## Trade Profile", ""])
    for label, profile in payload["trade_profile"].items():
        lines.append(f"- {label}: {profile.get('rows', 0)} rows, PnL {fmt(profile.get('total_pnl'))}, win rate {p249.finite(profile.get('win_rate'), 0.0):.3f}, median premium {fmt(profile.get('median_entry_premium'))}")
        for row in profile.get("by_moneyness", []):
            lines.append(f"  - {row['moneyness']}: {row['trades']} trades, PnL {fmt(row['pnl'])}, win {float(row['win_rate']):.3f}, median premium {fmt(row['median_entry_premium'])}")
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Model trades: `{path.parent / 'model_trades.csv'}`", f"- Base challenger trades: `{path.parent / 'base_challenger_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    lines = [
        "",
        f"## {HISTORICAL_ID} - {ROLE_LABEL}",
        "",
        f"- What is this: {payload['what_is_this']}",
        "- Changes paper default: no",
        f"- Candidate: {payload['candidate_label']}",
        f"- Baseline: {payload['paper_default_label']}",
        f"- Data used: `{payload['data_used']}`",
        "- Paid data downloaded: false",
        "- Broker endpoint called: false",
        f"- Decision: `{payload['decision']}`",
        f"- Report: `{out_dir / 'report.md'}`",
    ]
    with ledger.open("a") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
