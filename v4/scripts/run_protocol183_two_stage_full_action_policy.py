"""Protocol183: two-stage full action-space policy.

The previous full-action models used one score to answer two different
questions: whether to trade at all, and which contract to choose. Protocol183
separates those questions:

* event head: trade this minute or wait
* candidate head: rank the contracts only when the oracle says taking a trade
  is better than waiting

The dataset, exits, account rules, and validation discipline remain unchanged.
No paid data is downloaded and no broker endpoint is called.
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
from v4.scripts.run_protocol164_full_action_space_dataset import FULL_ACTION_FEATURE_COLUMNS as BASE_FULL_ACTION_FEATURE_COLUMNS
from v4.scripts.run_protocol165_full_action_space_policy import (
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    DEFAULT_PROTOCOL164_DIR,
    FOLDS,
    MAX_ACTION_CANDIDATES,
    STARTING_CASH,
    aggregate,
    fmt,
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
from v4.scripts.run_protocol172_full_action_value_policy import (
    add_oracle_advantages,
    build_events,
    summarize_events,
)


LOOP_ID = "v4_aplus_hypothesis_183_two_stage_full_action_policy"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
MODEL_SEEDS = [1, 2, 3, 4, 5]
ACTIVE_FEATURE_COLUMNS = list(BASE_FULL_ACTION_FEATURE_COLUMNS)


@dataclass(frozen=True)
class TwoStageFullActionConfig:
    epochs: int = 16
    batch_size: int = 512
    hidden_dim: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    event_loss_weight: float = 1.0
    rank_loss_weight: float = 0.75
    value_loss_weight: float = 0.10
    target_scale: float = 100.0
    target_clip: float = 1200.0
    min_validation_trades: int = 10


class TwoStageFullActionPolicy(nn.Module):
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
        self.candidate_head = nn.Linear(hidden_dim, 1)
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(features)
        mask_f = mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (emb * mask_f).sum(dim=1) / count
        masked = emb.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_emb = masked.max(dim=1).values
        max_emb = torch.where(torch.isfinite(max_emb), max_emb, torch.zeros_like(max_emb))
        event_logit = self.event_head(torch.cat([mean, max_emb, count / float(MAX_ACTION_CANDIDATES)], dim=1)).squeeze(-1)
        candidate_scores = self.candidate_head(emb).squeeze(-1).masked_fill(~mask, -1e9)
        return event_logit, candidate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol164-dir", type=Path, default=DEFAULT_PROTOCOL164_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--extra-feature-columns", nargs="*", default=[], help="Additional dataset columns appended to the Protocol164 feature set.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset or args.protocol164_dir / "protocol164_full_action_space_dataset.parquet"
    dataset = load_dataset(dataset_path)
    set_active_feature_columns([*BASE_FULL_ACTION_FEATURE_COLUMNS, *list(args.extra_feature_columns or [])], dataset)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    oracle_summary = add_oracle_advantages(events)
    config = TwoStageFullActionConfig(
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
                        "protocol": "183_two_stage_full_action_policy",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": ACTIVE_FEATURE_COLUMNS,
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
                model_base = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.0, starting_cash=float(args.starting_cash), strategy="protocol183")
                model_10 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.10, starting_cash=float(args.starting_cash), strategy="protocol183_stress10")
                model_25 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.25, starting_cash=float(args.starting_cash), strategy="protocol183_stress25")
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
        "protocol": "183_two_stage_full_action_policy",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "dataset": str(dataset_path),
        "feature_columns": ACTIVE_FEATURE_COLUMNS,
        "max_action_candidates": MAX_ACTION_CANDIDATES,
        "event_summary": summarize_events(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate(fold_results, frozen_protocol101),
        "frozen_protocol101_baselines": frozen_protocol101,
    }
    payload["decision"] = decision(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(model_trades).to_csv(args.out_dir / "protocol183_model_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "protocol183_first_affordable_baseline_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def set_active_feature_columns(columns: list[str], dataset: pd.DataFrame | None = None) -> None:
    """Set the feature set for this run while keeping Protocol183 backward-compatible."""

    global ACTIVE_FEATURE_COLUMNS
    deduped = list(dict.fromkeys(str(column) for column in columns))
    if dataset is not None:
        missing = [column for column in deduped if column not in dataset.columns]
        if missing:
            raise ValueError(f"dataset is missing requested feature columns: {missing}")
    ACTIVE_FEATURE_COLUMNS = deduped


def train_policy(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    seed: int,
    config: TwoStageFullActionConfig,
) -> tuple[TwoStageFullActionPolicy, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(np.vstack([event["candidates"][ACTIVE_FEATURE_COLUMNS].to_numpy(dtype=np.float32) for event in train_events]))
    x_train, mask_train, y_event_train, y_rank_train, target_train, weight_train = tensors(train_events, scaler, config)
    x_val, mask_val, y_event_val, y_rank_val, target_val, weight_val = tensors(validation_events, scaler, config)
    positives = float(y_event_train.sum())
    negatives = float(len(y_event_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    model = TwoStageFullActionPolicy(input_dim=len(ACTIVE_FEATURE_COLUMNS), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(mask_train),
            torch.from_numpy(y_event_train),
            torch.from_numpy(y_rank_train),
            torch.from_numpy(target_train),
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
        for bx, bm, bye, byr, bt, bw in loader:
            optimizer.zero_grad(set_to_none=True)
            event_logit, candidate_scores = model(bx, bm)
            loss = two_stage_loss(event_logit, candidate_scores, bm, bye, byr, bt, bw, pos_weight, config)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            ve, vs = model(torch.from_numpy(x_val), torch.from_numpy(mask_val))
            val_loss = float(
                two_stage_loss(
                    ve,
                    vs,
                    torch.from_numpy(mask_val),
                    torch.from_numpy(y_event_val),
                    torch.from_numpy(y_rank_val),
                    torch.from_numpy(target_val),
                    torch.from_numpy(weight_val),
                    pos_weight,
                    config,
                ).detach().cpu()
            )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "is_best": val_loss <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def two_stage_loss(
    event_logit: torch.Tensor,
    candidate_scores: torch.Tensor,
    mask: torch.Tensor,
    y_event: torch.Tensor,
    y_rank: torch.Tensor,
    target: torch.Tensor,
    event_weights: torch.Tensor,
    pos_weight: torch.Tensor,
    config: TwoStageFullActionConfig,
) -> torch.Tensor:
    event_loss = nn.functional.binary_cross_entropy_with_logits(
        event_logit,
        y_event,
        weight=event_weights,
        pos_weight=pos_weight,
        reduction="mean",
    )
    ranked = y_rank >= 0
    if bool(ranked.any()):
        rank_loss = nn.functional.cross_entropy(candidate_scores[ranked], y_rank[ranked], reduction="none")
        rank_loss = (rank_loss * event_weights[ranked]).mean()
    else:
        rank_loss = candidate_scores.sum() * 0.0
    if mask.any():
        utility_loss = nn.functional.huber_loss(candidate_scores[mask], target[mask], delta=1.0, reduction="none").mean()
    else:
        utility_loss = candidate_scores.sum() * 0.0
    return (
        config.event_loss_weight * event_loss
        + config.rank_loss_weight * rank_loss
        + config.value_loss_weight * utility_loss
    )


def tensors(
    events: list[dict[str, Any]],
    scaler: FeatureScaler,
    config: TwoStageFullActionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(events), MAX_ACTION_CANDIDATES, len(ACTIVE_FEATURE_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=bool)
    y_event = np.zeros(len(events), dtype=np.float32)
    y_rank = np.full(len(events), -100, dtype=np.int64)
    target = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=np.float32)
    weights = np.ones(len(events), dtype=np.float32)
    for idx, event in enumerate(events):
        candidates = event["candidates"].head(MAX_ACTION_CANDIDATES)
        raw = candidates[ACTIVE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
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
                y_event[idx] = 1.0
                y_rank[idx] = best
                weights[idx] = 1.0 + min(float(masked_advantages[best]) / 300.0, 5.0)
    return x, mask, y_event, y_rank, target, weights


def select_threshold(
    events: list[dict[str, Any]],
    model: TwoStageFullActionPolicy,
    scaler: FeatureScaler,
    *,
    config: TwoStageFullActionConfig,
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
        base = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, starting_cash=starting_cash, strategy="protocol183_validation")
        stress = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, starting_cash=starting_cash, strategy="protocol183_validation_stress10")
        sweep.append(
            {
                "threshold": float(threshold),
                "model": base.summary,
                "model_stress_0_10": stress.summary,
                "first_affordable_baseline": first.summary,
                "delta_vs_first": float(base.summary["total_pnl"] - first.summary["total_pnl"]),
            }
        )
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["model_stress_0_10"]["total_pnl"], row["delta_vs_first"], row["model"]["profit_factor"], row["model"]["trades"]))
    return {"threshold": float(best["threshold"]), "objective": "validation event-logit stress_0_10 pnl then delta vs first-affordable", "selected": best, "sweep": sweep}


def event_scores(events: list[dict[str, Any]], model: TwoStageFullActionPolicy, scaler: FeatureScaler) -> np.ndarray:
    if not events:
        return np.asarray([], dtype=float)
    x, mask, _, _, _, _ = tensors(events, scaler, TwoStageFullActionConfig())
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(events), 4096):
            event_logit, _ = model(torch.from_numpy(x[start : start + 4096]), torch.from_numpy(mask[start : start + 4096]))
            out.append(event_logit.cpu().numpy())
    return np.concatenate(out)


def predict_action(event: dict[str, Any], model: TwoStageFullActionPolicy, scaler: FeatureScaler) -> tuple[int, float]:
    x, mask, _, _, _, _ = tensors([event], scaler, TwoStageFullActionConfig())
    model.eval()
    with torch.no_grad():
        event_logit, candidate_scores = model(torch.from_numpy(x), torch.from_numpy(mask))
    scores = candidate_scores.cpu().numpy()[0]
    action = int(np.argmax(scores)) + 1
    return action, float(event_logit.cpu().numpy()[0])


def simulate_model(
    events: list[dict[str, Any]],
    model: TwoStageFullActionPolicy,
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


def decision(payload: dict[str, Any]) -> str:
    if payload["aggregate"].get("promotion_ready"):
        return "promote_research_candidate: Protocol183 beats frozen Protocol101 under full-action serial replay"
    return "research_only: Protocol183 did not clear the frozen Protocol101 full-action promotion gate"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol183 Two-Stage Full Action-Space Policy",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Max candidates per decision: `{payload['max_action_candidates']}`",
        "",
        "## Aggregate",
        "",
        "| split | seeds | median PnL | frozen Protocol101 | delta | PF | stress 0.10 | trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["aggregate"].items():
        if not isinstance(item, dict) or item.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | "
            f"{fmt(item.get('frozen_protocol101_total_pnl'))} | {fmt(item.get('median_delta_vs_frozen_protocol101'))} | "
            f"{fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} | {fmt(item['median_trades'])} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Summary: `{path.parent / 'summary.json'}`",
        f"- Model trades: `{path.parent / 'protocol183_model_trades.csv'}`",
    ])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
