"""Protocol 172: full action-space value policy.

Protocol165 framed the full ladder as a sparse wait-vs-index classification
problem. Protocol172 keeps the same Protocol164 dataset and frozen
Protocol081 exits, but trains candidate scores against the dynamic-programming
oracle advantage:

    take this contract now + future value after exit - wait value

The model still trades one account, one contract, one open position max, with
ask entry, bid exit, and affordability checks. No paid data is downloaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol164_full_action_space_dataset import FULL_ACTION_FEATURE_COLUMNS
from v4.scripts.run_protocol165_full_action_space_policy import (
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    DEFAULT_PROTOCOL164_DIR,
    FOLDS,
    MAX_ACTION_CANDIDATES,
    STARTING_CASH,
    aggregate,
    fmt,
    finite,
    group_events,
    load_dataset,
    load_protocol101_baselines,
    reported_slices,
    simulate_first_affordable,
    simulate_matched_random,
    simulate_oracle,
    simulation_result,
    smoke_folds,
    trade_from_row,
)


LOOP_ID = "v4_aplus_hypothesis_172_full_action_value_policy"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
MODEL_SEEDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class FullActionValueConfig:
    epochs: int = 14
    batch_size: int = 512
    hidden_dim: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_scale: float = 100.0
    target_clip: float = 1200.0
    rank_aux_weight: float = 0.20
    min_validation_trades: int = 10


class CandidateValueSetPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.net(features).squeeze(-1)
        return scores.masked_fill(~mask, -1e9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol164-dir", type=Path, default=DEFAULT_PROTOCOL164_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset or args.protocol164_dir / "protocol164_full_action_space_dataset.parquet"
    dataset = load_dataset(dataset_path)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    oracle_summary = add_oracle_advantages(events)
    config = FullActionValueConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
    )
    folds = smoke_folds(events) if args.smoke else FOLDS
    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    for fold in folds:
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_events = [event for event in events if event["split"] == fold["validation_split"]]
        if not train_events or not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            model, scaler, history = train_policy(train_events, validation_events, seed=int(seed), config=config)
            threshold = select_threshold(
                validation_events,
                model,
                scaler,
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
                        "protocol": "172_full_action_value_policy",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": FULL_ACTION_FEATURE_COLUMNS,
                        "max_action_candidates": MAX_ACTION_CANDIDATES,
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
                model_base = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.0, starting_cash=float(args.starting_cash), strategy="protocol172")
                model_10 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.10, starting_cash=float(args.starting_cash), strategy="protocol172_stress10")
                model_25 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.25, starting_cash=float(args.starting_cash), strategy="protocol172_stress25")
                first = simulate_first_affordable(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                random_base = simulate_matched_random(split_events, seed=int(seed), slippage_per_side=0.0, starting_cash=float(args.starting_cash))
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
                model_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in model_base.trades)
                baseline_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in first.trades)
            fold_results.append(result)

    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    payload = {
        "protocol": "172_full_action_value_policy",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "dataset": str(dataset_path),
        "feature_columns": FULL_ACTION_FEATURE_COLUMNS,
        "max_action_candidates": MAX_ACTION_CANDIDATES,
        "event_summary": summarize_events(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate(fold_results, frozen_protocol101),
        "frozen_protocol101_baselines": frozen_protocol101,
    }
    payload["decision"] = decision(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(model_trades).to_csv(args.out_dir / "protocol172_model_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "protocol172_first_affordable_baseline_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def build_events(frame: pd.DataFrame, *, starting_cash: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ordered = frame.sort_values(["split", "session", "decision_dt", "contract_id"]).copy()
    for (split, session, decision_dt), group in ordered.groupby(["split", "session", "decision_dt"], sort=False):
        candidates = group.head(MAX_ACTION_CANDIDATES).copy().reset_index(drop=True)
        candidates["entry_affordable_dynamic"] = (pd.to_numeric(candidates["entry_premium"], errors="coerce") <= starting_cash).astype(float)
        events.append({"split": str(split), "session": str(session), "decision_dt": pd.Timestamp(decision_dt), "candidates": candidates})
    return events


def add_oracle_advantages(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    advantage_rows: list[dict[str, float | str]] = []
    for _, session_events in group_events(events).items():
        session_events.sort(key=lambda event: event["decision_dt"])
        times = np.asarray([event["decision_dt"].value for event in session_events], dtype=np.int64)
        values = np.zeros(len(session_events) + 1, dtype=float)
        actions = np.zeros(len(session_events), dtype=np.int64)
        advantages_by_event: list[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in session_events]
        for idx in range(len(session_events) - 1, -1, -1):
            wait_value = values[idx + 1]
            candidates = session_events[idx]["candidates"]
            advantages = np.full(len(candidates), -1e9, dtype=np.float32)
            best_value = wait_value
            best_action = 0
            for local_idx, row in candidates.iterrows():
                if float(row.get("entry_affordable_10k", 0.0)) < 1.0:
                    continue
                exit_ns = pd.Timestamp(row["candidate_exit_dt"]).value
                if exit_ns <= session_events[idx]["decision_dt"].value:
                    continue
                next_idx = int(np.searchsorted(times, exit_ns, side="left"))
                take_value = float(row["candidate_pnl"]) + values[next_idx]
                advantages[int(local_idx)] = float(take_value - wait_value)
                if take_value > best_value:
                    best_value = take_value
                    best_action = int(local_idx) + 1
            values[idx] = best_value
            actions[idx] = best_action
            advantages_by_event[idx] = advantages
        for idx, event in enumerate(session_events):
            event["oracle_action"] = int(actions[idx])
            event["oracle_advantages"] = advantages_by_event[idx]
            event["oracle_best_advantage"] = float(np.max(advantages_by_event[idx])) if len(advantages_by_event[idx]) else -1e9
            split = str(event["split"])
            item = by_split.setdefault(split, {"take": 0, "wait": 0})
            item["take" if int(actions[idx]) else "wait"] += 1
            finite_adv = advantages_by_event[idx][np.isfinite(advantages_by_event[idx]) & (advantages_by_event[idx] > -1e8)]
            if len(finite_adv):
                advantage_rows.append({"split": split, "max_advantage": float(np.max(finite_adv)), "positive_candidate_fraction": float((finite_adv > 0).mean())})
    return {
        "by_split": by_split,
        "advantage_summary": {
            split: {
                "median_max_advantage": float(group["max_advantage"].median()),
                "positive_event_fraction": float((group["max_advantage"] > 0).mean()),
                "median_positive_candidate_fraction": float(group["positive_candidate_fraction"].median()),
            }
            for split, group in pd.DataFrame(advantage_rows).groupby("split")
        }
        if advantage_rows
        else {},
    }


def train_policy(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    seed: int,
    config: FullActionValueConfig,
) -> tuple[CandidateValueSetPolicy, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(np.vstack([event["candidates"][FULL_ACTION_FEATURE_COLUMNS].to_numpy(dtype=np.float32) for event in train_events]))
    x_train, mask_train, target_train, label_train, weight_train = tensors(train_events, scaler, config)
    x_val, mask_val, target_val, label_val, weight_val = tensors(validation_events, scaler, config)
    model = CandidateValueSetPolicy(input_dim=len(FULL_ACTION_FEATURE_COLUMNS), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(mask_train),
            torch.from_numpy(target_train),
            torch.from_numpy(label_train),
            torch.from_numpy(weight_train),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, bm, bt, bl, bw in loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(bx, bm)
            loss = value_loss(scores, bm, bt, bl, bw, config)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(
                value_loss(
                    model(torch.from_numpy(x_val), torch.from_numpy(mask_val)),
                    torch.from_numpy(mask_val),
                    torch.from_numpy(target_val),
                    torch.from_numpy(label_val),
                    torch.from_numpy(weight_val),
                    config,
                ).detach().cpu()
            )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "is_best": val_loss <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def value_loss(
    scores: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    event_weights: torch.Tensor,
    config: FullActionValueConfig,
) -> torch.Tensor:
    if mask.any():
        per_candidate = nn.functional.huber_loss(scores[mask], target[mask], delta=1.0, reduction="none")
        expanded_weights = event_weights.unsqueeze(1).expand_as(scores)[mask]
        regression = (per_candidate * expanded_weights).mean()
    else:
        regression = scores.sum() * 0.0
    ranked = labels >= 0
    if config.rank_aux_weight > 0.0 and bool(ranked.any()):
        rank_loss = nn.functional.cross_entropy(scores[ranked], labels[ranked], reduction="none")
        rank_loss = (rank_loss * event_weights[ranked]).mean()
        return regression + config.rank_aux_weight * rank_loss
    return regression


def tensors(
    events: list[dict[str, Any]],
    scaler: FeatureScaler,
    config: FullActionValueConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(events), MAX_ACTION_CANDIDATES, len(FULL_ACTION_FEATURE_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=bool)
    target = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=np.float32)
    labels = np.full(len(events), -100, dtype=np.int64)
    weights = np.ones(len(events), dtype=np.float32)
    for idx, event in enumerate(events):
        candidates = event["candidates"].head(MAX_ACTION_CANDIDATES)
        raw = candidates[FULL_ACTION_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        n = len(raw)
        x[idx, :n, :] = scaler.transform(raw)
        affordable = candidates["entry_affordable_10k"].to_numpy(dtype=float) >= 1.0
        valid_exit = pd.to_datetime(candidates["candidate_exit_dt"], utc=True) > pd.Timestamp(event["decision_dt"])
        mask[idx, :n] = affordable & valid_exit.to_numpy(dtype=bool)
        advantages = np.asarray(event.get("oracle_advantages", np.zeros(n)), dtype=np.float32)[:n]
        advantages = np.where(np.isfinite(advantages) & (advantages > -1e8), advantages, -config.target_clip)
        target[idx, :n] = np.clip(advantages, -config.target_clip, config.target_clip) / config.target_scale
        if mask[idx, :n].any():
            masked_advantages = np.where(mask[idx, :n], advantages, -1e9)
            best = int(np.argmax(masked_advantages))
            if float(masked_advantages[best]) > 0.0:
                labels[idx] = best
                weights[idx] = 1.0 + min(float(masked_advantages[best]) / 300.0, 5.0)
    return x, mask, target, labels, weights


def select_threshold(
    events: list[dict[str, Any]],
    model: CandidateValueSetPolicy,
    scaler: FeatureScaler,
    *,
    config: FullActionValueConfig,
    starting_cash: float,
) -> dict[str, Any]:
    scores = event_scores(events, model, scaler)
    finite_scores = scores[np.isfinite(scores)]
    thresholds = [float("inf")] if len(finite_scores) == 0 else sorted(
        set(np.quantile(finite_scores, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist() + [0.0, float(finite_scores.min()) - 1e-3])
    )
    first = simulate_first_affordable(events, slippage_per_side=0.0, starting_cash=starting_cash)
    sweep = []
    for threshold in thresholds:
        base = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, starting_cash=starting_cash, strategy="protocol172_validation")
        stress = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, starting_cash=starting_cash, strategy="protocol172_validation_stress10")
        sweep.append({"threshold": float(threshold), "model": base.summary, "model_stress_0_10": stress.summary, "first_affordable_baseline": first.summary, "delta_vs_first": float(base.summary["total_pnl"] - first.summary["total_pnl"])})
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["model_stress_0_10"]["total_pnl"], row["delta_vs_first"], row["model"]["profit_factor"], row["model"]["trades"]))
    return {"threshold": float(best["threshold"]), "objective": "validation stress_0_10 pnl then delta vs first-affordable", "selected": best, "sweep": sweep}


def event_scores(events: list[dict[str, Any]], model: CandidateValueSetPolicy, scaler: FeatureScaler) -> np.ndarray:
    if not events:
        return np.asarray([], dtype=float)
    x, mask, _, _, _ = tensors(events, scaler, FullActionValueConfig())
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(events), 4096):
            scores = model(torch.from_numpy(x[start : start + 4096]), torch.from_numpy(mask[start : start + 4096])).cpu().numpy()
            out.append(np.max(scores, axis=1))
    return np.concatenate(out)


def predict_action(event: dict[str, Any], model: CandidateValueSetPolicy, scaler: FeatureScaler) -> tuple[int, float]:
    x, mask, _, _, _ = tensors([event], scaler, FullActionValueConfig())
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(x), torch.from_numpy(mask)).cpu().numpy()[0]
    action = int(np.argmax(scores)) + 1
    return action, float(scores[action - 1])


def simulate_model(
    events: list[dict[str, Any]],
    model: CandidateValueSetPolicy,
    scaler: FeatureScaler,
    *,
    threshold: float,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    trades = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0}
    for event in sorted(events, key=lambda row: (row["session"], row["decision_dt"])):
        if open_until.get(event["session"]) is not None and event["decision_dt"] < open_until[event["session"]]:
            skipped["overlap"] += len(event["candidates"])
            continue
        action, score = predict_action(event, model, scaler)
        if action <= 0 or score < threshold or action > len(event["candidates"]):
            skipped["threshold"] += len(event["candidates"])
            continue
        trade = trade_from_row(event["candidates"].iloc[action - 1], score=score, threshold=threshold, slippage_per_side=slippage_per_side, equity=equity, strategy=strategy)
        if trade is None:
            skipped["invalid"] += 1
            continue
        if trade["entry_premium_with_slippage"] > equity:
            skipped["unaffordable"] += 1
            continue
        trades.append(trade)
        equity += trade["pnl"]
        trades[-1]["account_equity_after"] = equity
        open_until[event["session"]] = pd.Timestamp(trade["exit_time"])
    return simulation_result(trades, events, skipped, starting_cash=starting_cash, strategy=strategy)


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for event in events:
        item = out.setdefault(event["split"], {"events": 0, "candidate_rows": 0, "sessions": set()})
        item["events"] += 1
        item["candidate_rows"] += len(event["candidates"])
        item["sessions"].add(event["session"])
    return {key: {"events": value["events"], "candidate_rows": value["candidate_rows"], "sessions": len(value["sessions"])} for key, value in sorted(out.items())}


def decision(payload: dict[str, Any]) -> str:
    if payload["aggregate"].get("promotion_ready"):
        return "promote_research_candidate: Protocol172 beats frozen Protocol101 under full-action serial replay"
    return "research_only: Protocol172 did not clear the frozen Protocol101 full-action promotion gate"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol172 Full Action-Space Value Policy",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Max candidates per decision: `{payload['max_action_candidates']}`",
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
    lines.extend([
        "",
        "## Oracle Advantage Summary",
        "",
        "```json",
        json.dumps(payload.get("oracle_summary", {}).get("advantage_summary", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Outputs",
        "",
        f"- Summary: `{path.parent / 'summary.json'}`",
        f"- Model trades: `{path.parent / 'protocol172_model_trades.csv'}`",
    ])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
