"""EXP_POLICY_ROUTER_UNION_SLOT_AWARE_V1.

Historically Protocol254. This is a router experiment, not a new entry
generator. It trains a neural event policy over the union of two already-known
proposal streams:

* PAPER_DEFAULT_PROTOCOL101 proposals
* CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 proposals

At each flat decision time the model may wait, take the Protocol101 proposal,
or take the premium-blend proposal. A session-level dynamic program creates the
wait/take target under one-account, one-open-position serial rules, so weak
early entries can be penalized when they block a better later proposal.

No paid data is downloaded. No broker endpoint is called. This does not change
the paper default.
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
    load_protocol101_baselines,
    reported_slices,
    simulation_result,
    smoke_folds,
)
import v4.scripts.run_protocol249_entry_quality_calibrator as p249


ROLE_LABEL = "EXP_POLICY_ROUTER_UNION_SLOT_AWARE_V1"
HISTORICAL_ID = "Protocol254"
CANDIDATE_LABEL = "CHALLENGER_POLICY_ROUTER_UNION_SLOT_AWARE_V1"
DEFAULT_INPUT = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_242_premium_blend_vs_protocol101_attribution/enriched_policy_trades.csv"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_254_policy_router_union_slot_aware")
MODEL_SEEDS = [1, 2, 3, 4, 5]
MAX_CANDIDATES = 4
CONTRACT_MULTIPLIER = 100.0
FEATURE_COLUMNS = [
    "policy_protocol101",
    "policy_challenger",
    "is_call",
    "is_put",
    "is_itm",
    "is_atm",
    "is_otm",
    "offset",
    "abs_offset",
    "score",
    "threshold",
    "score_margin",
    "entry_bid",
    "entry_ask",
    "entry_spread",
    "entry_spread_pct",
    "entry_premium",
    "entry_underlying",
    "minute_from_open",
    "minute_to_cutoff",
    "tod_sin",
    "tod_cos",
]
FORBIDDEN_FEATURE_TOKENS = (
    "pnl",
    "exit",
    "future",
    "mfe",
    "mae",
    "path",
    "duration",
    "label",
)


@dataclass(frozen=True)
class RouterConfig:
    epochs: int = 14
    batch_size: int = 512
    hidden_dim: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    utility_aux_weight: float = 0.15
    target_scale: float = 300.0
    target_clip: float = 1200.0
    min_validation_trades: int = 10


class RouterEventPolicy(nn.Module):
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
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(features)
        mask_f = mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (emb * mask_f).sum(dim=1) / count
        masked = emb.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_emb = masked.max(dim=1).values
        max_emb = torch.where(torch.isfinite(max_emb), max_emb, torch.zeros_like(max_emb))
        count_feature = count / float(MAX_CANDIDATES)
        wait_logit = self.wait_head(torch.cat([mean, max_emb, count_feature], dim=1))
        candidate_logits = self.candidate_head(emb).squeeze(-1).masked_fill(~mask, -1e9)
        return torch.cat([wait_logit, candidate_logits], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-smoke-sessions", type=int, default=3)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_leakage_features(FEATURE_COLUMNS)
    proposals = load_router_proposals(args.input)
    if args.smoke:
        proposals = smoke_frame(proposals, max_sessions=int(args.max_smoke_sessions))
    events = build_router_events(proposals)
    oracle_summary = add_oracle_actions(events)
    folds = smoke_folds(events) if args.smoke else FOLDS
    config = RouterConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
    )

    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    oracle_trades: list[dict[str, Any]] = []
    protocol101_trades: list[dict[str, Any]] = []
    challenger_trades: list[dict[str, Any]] = []
    for fold in folds:
        print(json.dumps({"stage": "fold_start", "fold": fold["name"]}), flush=True)
        fold_events = events_for_fold(events, fold["name"])
        train_events = [event for event in fold_events if event["split"] in set(fold["train_splits"])]
        validation_events = [event for event in fold_events if event["split"] == fold["validation_split"]]
        if not train_events or not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": fold["name"], "seed": int(seed)}), flush=True)
            seed_train = [event for event in train_events if int(event["seed"]) == int(seed)]
            seed_validation = [event for event in validation_events if int(event["seed"]) == int(seed)]
            if not seed_train or not seed_validation:
                fold_results.append({"fold": fold["name"], "seed": int(seed), "skipped": True, "reason": "missing_seed_train_or_validation", "splits": {}})
                continue
            model, scaler, history = train_router(seed_train, seed_validation, seed=int(seed), config=config)
            threshold = select_margin_threshold(
                seed_validation,
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
                        "role_label": ROLE_LABEL,
                        "historical_protocol": HISTORICAL_ID,
                        "fold": fold["name"],
                        "seed": int(seed),
                        "candidate_label": CANDIDATE_LABEL,
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
            result = {"fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"]), "splits": {}}
            for split_name, split_events in reported_slices(fold_events, fold).items():
                split_seed_events = [event for event in split_events if int(event["seed"]) == int(seed)]
                base = simulate_router(
                    split_seed_events,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="policy_router_union_slot_aware",
                )
                stress10 = simulate_router(
                    split_seed_events,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                    strategy="policy_router_union_slot_aware_stress10",
                )
                stress25 = simulate_router(
                    split_seed_events,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    starting_cash=float(args.starting_cash),
                    strategy="policy_router_union_slot_aware_stress25",
                )
                oracle = simulate_oracle(split_seed_events, starting_cash=float(args.starting_cash), strategy="union_oracle")
                protocol101 = simulate_policy_only(
                    split_seed_events,
                    "protocol101",
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="protocol101_from_union_stream",
                )
                challenger = simulate_policy_only(
                    split_seed_events,
                    "challenger",
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="premium_blend_from_union_stream",
                )
                result["splits"][split_name] = {
                    "model": add_router_metrics(base.summary, base.trades),
                    "model_stress_0_10": add_router_metrics(stress10.summary, stress10.trades),
                    "model_stress_0_25": add_router_metrics(stress25.summary, stress25.trades),
                    "union_oracle": add_router_metrics(oracle.summary, oracle.trades),
                    "protocol101_stream": add_router_metrics(protocol101.summary, protocol101.trades),
                    "challenger_stream": add_router_metrics(challenger.summary, challenger.trades),
                    "delta_vs_protocol101_stream": float(base.summary["total_pnl"] - protocol101.summary["total_pnl"]),
                    "delta_vs_challenger_stream": float(base.summary["total_pnl"] - challenger.summary["total_pnl"]),
                    "oracle_gap": float(oracle.summary["total_pnl"] - base.summary["total_pnl"]),
                }
                model_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in base.trades)
                oracle_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in oracle.trades)
                protocol101_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in protocol101.trades)
                challenger_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in challenger.trades)
            fold_results.append(result)
            print(json.dumps({"stage": "seed_done", "fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"])}), flush=True)

    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    aggregate_payload = aggregate(fold_results, frozen_protocol101)
    aggregate_payload = p249.add_required_split_and_seed_checks(aggregate_payload, required_seed_count=len(args.seeds))
    model_frame = pd.DataFrame(model_trades)
    oracle_frame = pd.DataFrame(oracle_trades)
    p101_frame = pd.DataFrame(protocol101_trades)
    challenger_frame = pd.DataFrame(challenger_trades)
    write_frame(model_frame, args.out_dir / "model_trades.csv")
    write_frame(oracle_frame, args.out_dir / "union_oracle_trades.csv")
    write_frame(p101_frame, args.out_dir / "protocol101_stream_trades.csv")
    write_frame(challenger_frame, args.out_dir / "challenger_stream_trades.csv")
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / policy-router model change",
        "changes_paper_default": False,
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_to_beat": "PAPER_DEFAULT_PROTOCOL101 strict one-account serial replay and premium-blend stream",
        "data_used": str(args.input),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "feature_columns": FEATURE_COLUMNS,
        "event_summary": event_summary(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate_payload,
        "frozen_protocol101_baselines": frozen_protocol101,
        "router_summary": router_summary(fold_results),
        "trade_profile": {
            "model": trade_profile(model_frame),
            "union_oracle": trade_profile(oracle_frame),
            "protocol101_stream": trade_profile(p101_frame),
            "challenger_stream": trade_profile(challenger_frame),
        },
        "invariants": serial_invariants(model_frame),
        "decision": "",
        "next_experiment": "If router underperforms the premium-blend stream, use attribution to decide between richer Protocol101 features or a recurrent router.",
    }
    payload["decision"] = decide(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_router_proposals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["policy"].isin(["protocol101", "challenger"])].copy()
    frame["decision_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["candidate_exit_dt"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["candidate_pnl"] = pd.to_numeric(frame["pnl"], errors="coerce")
    frame["entry_ask"] = coalesce_numeric(frame, "entry_ask", "entry_ask_live")
    frame["entry_bid"] = coalesce_numeric(frame, "entry_bid", "entry_bid_live")
    frame["entry_spread"] = (frame["entry_ask"] - frame["entry_bid"]).clip(lower=0.0)
    frame["entry_spread_pct"] = frame["entry_spread"] / frame["entry_ask"].replace(0.0, np.nan)
    frame["entry_premium"] = coalesce_numeric(frame, "entry_premium")
    missing_premium = ~np.isfinite(frame["entry_premium"]) | (frame["entry_premium"] <= 0.0)
    frame.loc[missing_premium, "entry_premium"] = frame.loc[missing_premium, "entry_ask"] * CONTRACT_MULTIPLIER
    frame["score"] = pd.to_numeric(frame.get("score"), errors="coerce").fillna(0.0)
    frame["threshold"] = pd.to_numeric(frame.get("threshold"), errors="coerce").fillna(0.0)
    frame["score_margin"] = frame["score"] - frame["threshold"]
    frame["entry_underlying"] = coalesce_numeric(frame, "entry_underlying", "entry_underlying_price")
    frame["offset"] = pd.to_numeric(frame["offset"], errors="coerce")
    frame["abs_offset"] = frame["offset"].abs()
    frame["right"] = frame["right"].astype(str)
    frame["is_call"] = (frame["right"] == "C").astype(float)
    frame["is_put"] = (frame["right"] == "P").astype(float)
    moneyness = [p249.p248.moneyness(right, offset) for right, offset in zip(frame["right"], frame["offset"])]
    frame["is_itm"] = [1.0 if item == "ITM" else 0.0 for item in moneyness]
    frame["is_atm"] = [1.0 if item == "ATM" else 0.0 for item in moneyness]
    frame["is_otm"] = [1.0 if item == "OTM" else 0.0 for item in moneyness]
    frame["policy_protocol101"] = (frame["policy"] == "protocol101").astype(float)
    frame["policy_challenger"] = (frame["policy"] == "challenger").astype(float)
    frame["minute_from_open"] = minute_from_open(frame["decision_dt"])
    frame["minute_to_cutoff"] = 360.0 - frame["minute_from_open"]
    minute_day = frame["decision_dt"].dt.hour * 60 + frame["decision_dt"].dt.minute
    frame["tod_sin"] = np.sin(2.0 * np.pi * minute_day / 1440.0)
    frame["tod_cos"] = np.cos(2.0 * np.pi * minute_day / 1440.0)
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid = (
        frame["decision_dt"].notna()
        & frame["candidate_exit_dt"].notna()
        & np.isfinite(frame["candidate_pnl"])
        & (frame["entry_ask"] > 0.0)
        & (frame["entry_premium"] > 0.0)
        & (frame["candidate_exit_dt"] > frame["decision_dt"])
    )
    return frame.loc[valid].sort_values(["reported_split", "seed", "session", "decision_dt", "policy", "contract_id"]).reset_index(drop=True)


def coalesce_numeric(frame: pd.DataFrame, *columns: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame.columns:
            out = out.combine_first(pd.to_numeric(frame[column], errors="coerce"))
    return out


def minute_from_open(times: pd.Series) -> pd.Series:
    eastern = times.dt.tz_convert("America/New_York")
    return ((eastern.dt.hour * 60 + eastern.dt.minute) - (9 * 60 + 30)).astype(float)


def smoke_frame(frame: pd.DataFrame, *, max_sessions: int) -> pd.DataFrame:
    sessions = sorted(frame["session"].unique())[:max_sessions]
    return frame[frame["session"].isin(sessions)].copy()


def build_router_events(proposals: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ordered = proposals.sort_values(["reported_split", "seed", "session", "decision_dt", "policy", "contract_id", "candidate_uid"])
    group_columns = ["reported_split", "seed", "session", "decision_time"]
    has_router_fold = "router_fold" in ordered.columns
    if has_router_fold:
        group_columns = ["router_fold", *group_columns]
    for key, group in ordered.groupby(group_columns, sort=False):
        group = group.sort_values(["policy_protocol101", "score"], ascending=[False, False]).head(MAX_CANDIDATES).copy()
        if has_router_fold:
            router_fold, split, seed, session, decision_time = key
        else:
            split, seed, session, decision_time = key
            router_fold = ""
        events.append(
            {
                "router_fold": str(router_fold),
                "split": str(split),
                "seed": int(seed),
                "session": str(session),
                "decision_time": str(decision_time),
                "decision_dt": pd.Timestamp(group["decision_dt"].iloc[0]),
                "candidates": group.reset_index(drop=True),
            }
        )
    return events


def events_for_fold(events: list[dict[str, Any]], fold_name: str) -> list[dict[str, Any]]:
    if not events or not any(event.get("router_fold") for event in events):
        return events
    return [event for event in events if str(event.get("router_fold", "")) == str(fold_name)]


def add_oracle_actions(events: list[dict[str, Any]]) -> dict[str, Any]:
    take_count = 0
    wait_count = 0
    by_split: dict[str, dict[str, int]] = {}
    by_policy: dict[str, int] = {}
    for _, session_events in group_events(events).items():
        session_events.sort(key=lambda event: event["decision_dt"])
        times = np.asarray([event["decision_dt"].value for event in session_events], dtype=np.int64)
        n = len(session_events)
        values = np.zeros(n + 1, dtype=float)
        actions = np.zeros(n, dtype=np.int64)
        weights = np.ones(n, dtype=float)
        for idx in range(n - 1, -1, -1):
            wait_value = values[idx + 1]
            best_value = wait_value
            best_action = 0
            best_take = -1e18
            for local_idx, row in session_events[idx]["candidates"].iterrows():
                exit_ns = pd.Timestamp(row["candidate_exit_dt"]).value
                next_idx = int(np.searchsorted(times, exit_ns, side="left"))
                take_value = float(row["candidate_pnl"]) + values[next_idx]
                best_take = max(best_take, take_value)
                if take_value > best_value:
                    best_value = take_value
                    best_action = int(local_idx) + 1
            values[idx] = best_value
            actions[idx] = best_action
            if best_action == 0:
                weights[idx] = 1.0 + min(max(wait_value - best_take, 0.0) / 300.0, 5.0)
                wait_count += 1
            else:
                weights[idx] = 1.0 + min(max(best_value - wait_value, 0.0) / 300.0, 5.0)
                take_count += 1
        for idx, event in enumerate(session_events):
            event["oracle_action"] = int(actions[idx])
            event["oracle_weight"] = float(weights[idx])
            split = str(event["split"])
            by_split.setdefault(split, {"take": 0, "wait": 0})
            if int(actions[idx]) == 0:
                by_split[split]["wait"] += 1
            else:
                by_split[split]["take"] += 1
                policy = str(event["candidates"].iloc[int(actions[idx]) - 1]["policy"])
                by_policy[policy] = by_policy.get(policy, 0) + 1
    return {"take_events": take_count, "wait_events": wait_count, "by_split": by_split, "by_policy": by_policy}


def train_router(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    seed: int,
    config: RouterConfig,
) -> tuple[RouterEventPolicy, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(np.vstack([event["candidates"][FEATURE_COLUMNS].to_numpy(dtype=np.float32) for event in train_events]))
    x_train, mask_train, y_train, w_train, utility_train = event_tensors(train_events, scaler, config=config)
    x_val, mask_val, y_val, w_val, utility_val = event_tensors(validation_events, scaler, config=config)
    model = RouterEventPolicy(input_dim=len(FEATURE_COLUMNS), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(mask_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
            torch.from_numpy(utility_train),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val)
    mask_val_t = torch.from_numpy(mask_val)
    y_val_t = torch.from_numpy(y_val)
    w_val_t = torch.from_numpy(w_val)
    utility_val_t = torch.from_numpy(utility_val)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_mask, batch_y, batch_w, batch_utility in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_mask)
            loss = nn.functional.cross_entropy(logits, batch_y, reduction="none")
            loss = (loss * batch_w).mean()
            if config.utility_aux_weight > 0.0:
                utility_loss = nn.functional.huber_loss(
                    logits[:, 1:][batch_mask],
                    batch_utility[batch_mask],
                    delta=1.0,
                    reduction="mean",
                )
                loss = loss + float(config.utility_aux_weight) * utility_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t, mask_val_t)
            val_loss = nn.functional.cross_entropy(val_logits, y_val_t, reduction="none")
            val_loss = float((val_loss * w_val_t).mean().detach().cpu())
            if config.utility_aux_weight > 0.0:
                utility_loss = nn.functional.huber_loss(
                    val_logits[:, 1:][mask_val_t],
                    utility_val_t[mask_val_t],
                    delta=1.0,
                    reduction="mean",
                )
                val_loss += float(config.utility_aux_weight) * float(utility_loss.detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "is_best": val_loss <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def event_tensors(
    events: list[dict[str, Any]],
    scaler: FeatureScaler,
    *,
    config: RouterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(events), MAX_CANDIDATES, len(FEATURE_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(events), MAX_CANDIDATES), dtype=bool)
    y = np.zeros(len(events), dtype=np.int64)
    weights = np.ones(len(events), dtype=np.float32)
    utility = np.zeros((len(events), MAX_CANDIDATES), dtype=np.float32)
    scale = float(config.target_scale) if config else 300.0
    clip = float(config.target_clip) if config else 1200.0
    for idx, event in enumerate(events):
        raw = event["candidates"][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        n = min(len(raw), MAX_CANDIDATES)
        x[idx, :n, :] = scaler.transform(raw[:n])
        mask[idx, :n] = True
        y[idx] = int(event.get("oracle_action", 0))
        weights[idx] = float(event.get("oracle_weight", 1.0))
        pnl = event["candidates"]["candidate_pnl"].to_numpy(dtype=np.float32)[:n]
        utility[idx, :n] = np.clip(pnl, -clip, clip) / scale
    return x, mask, y, weights, utility


def select_margin_threshold(
    events: list[dict[str, Any]],
    model: RouterEventPolicy,
    scaler: FeatureScaler,
    *,
    config: RouterConfig,
    starting_cash: float,
) -> dict[str, Any]:
    margins = event_margins(events, model, scaler)
    finite = margins[np.isfinite(margins)]
    if len(finite):
        thresholds = sorted(set(np.quantile(finite, [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]).round(4).tolist() + [0.0, float(finite.min()) - 1e-3]))
    else:
        thresholds = [float("inf")]
    protocol101 = simulate_policy_only(events, "protocol101", slippage_per_side=0.0, starting_cash=starting_cash, strategy="protocol101_validation")
    challenger = simulate_policy_only(events, "challenger", slippage_per_side=0.0, starting_cash=starting_cash, strategy="challenger_validation")
    sweep = []
    for threshold in thresholds:
        sim = simulate_router(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, starting_cash=starting_cash, strategy="router_validation")
        stress = simulate_router(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, starting_cash=starting_cash, strategy="router_validation_stress10")
        sweep.append(
            {
                "threshold": float(threshold),
                "model": add_router_metrics(sim.summary, sim.trades),
                "model_stress_0_10": add_router_metrics(stress.summary, stress.trades),
                "protocol101_stream": add_router_metrics(protocol101.summary, protocol101.trades),
                "challenger_stream": add_router_metrics(challenger.summary, challenger.trades),
                "delta_vs_best_stream": float(sim.summary["total_pnl"] - max(protocol101.summary["total_pnl"], challenger.summary["total_pnl"])),
            }
        )
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(
        pool,
        key=lambda row: (
            row["delta_vs_best_stream"],
            row["model_stress_0_10"]["total_pnl"],
            row["model"]["profit_factor"],
            row["model"]["total_pnl"],
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "objective": "validation-only: delta vs best available stream, then slippage stress/PF/PnL",
        "selected": best,
        "sweep": sweep,
    }


def event_margins(events: list[dict[str, Any]], model: RouterEventPolicy, scaler: FeatureScaler) -> np.ndarray:
    if not events:
        return np.asarray([], dtype=float)
    x, mask, _, _, _ = event_tensors(events, scaler)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(events), 4096):
            logits = model(torch.from_numpy(x[start : start + 4096]), torch.from_numpy(mask[start : start + 4096])).cpu().numpy()
            wait = logits[:, 0]
            take = np.max(logits[:, 1:], axis=1)
            out.append(take - wait)
    return np.concatenate(out)


def simulate_router(
    events: list[dict[str, Any]],
    model: RouterEventPolicy,
    scaler: FeatureScaler,
    *,
    threshold: float,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    trades: list[dict[str, Any]] = []
    round_trip_slippage = float(slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER
    for _, session_events in group_events(events).items():
        equity = float(starting_cash)
        open_until: pd.Timestamp | None = None
        for event in sorted(session_events, key=lambda item: item["decision_dt"]):
            if open_until is not None and event["decision_dt"] < open_until:
                continue
            action, margin = predict_event_action(event, model, scaler)
            if action <= 0 or margin < threshold:
                continue
            row = event["candidates"].iloc[action - 1]
            if float(row["entry_premium"]) > equity:
                continue
            trade = trade_from_proposal(row, score=margin, threshold=threshold, slippage_per_side=slippage_per_side, equity=equity, strategy=strategy)
            trade["pnl"] = float(row["candidate_pnl"]) - round_trip_slippage
            equity += trade["pnl"]
            trade["account_equity_after"] = equity
            trades.append(trade)
            open_until = pd.Timestamp(row["candidate_exit_dt"])
    return simulation_result(trades, events, skipped={}, starting_cash=starting_cash, strategy=strategy)


def predict_event_action(event: dict[str, Any], model: RouterEventPolicy, scaler: FeatureScaler) -> tuple[int, float]:
    x, mask, _, _, _ = event_tensors([event], scaler)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(mask)).cpu().numpy()[0]
    wait = float(logits[0])
    candidate_logits = logits[1:]
    action = int(np.argmax(candidate_logits)) + 1
    margin = float(candidate_logits[action - 1] - wait)
    if action > len(event["candidates"]):
        return 0, margin
    return action, margin


def simulate_policy_only(
    events: list[dict[str, Any]],
    policy: str,
    *,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    trades: list[dict[str, Any]] = []
    round_trip_slippage = float(slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER
    for _, session_events in group_events(events).items():
        equity = float(starting_cash)
        open_until: pd.Timestamp | None = None
        for event in sorted(session_events, key=lambda item: item["decision_dt"]):
            if open_until is not None and event["decision_dt"] < open_until:
                continue
            candidates = event["candidates"][event["candidates"]["policy"] == policy]
            if candidates.empty:
                continue
            row = candidates.sort_values("score", ascending=False).iloc[0]
            if float(row["entry_premium"]) > equity:
                continue
            trade = trade_from_proposal(row, score=float(row["score"]), threshold=float(row["threshold"]), slippage_per_side=slippage_per_side, equity=equity, strategy=strategy)
            trade["pnl"] = float(row["candidate_pnl"]) - round_trip_slippage
            equity += trade["pnl"]
            trade["account_equity_after"] = equity
            trades.append(trade)
            open_until = pd.Timestamp(row["candidate_exit_dt"])
    return simulation_result(trades, events, skipped={}, starting_cash=starting_cash, strategy=strategy)


def simulate_oracle(events: list[dict[str, Any]], *, starting_cash: float, strategy: str) -> Any:
    trades: list[dict[str, Any]] = []
    for _, session_events in group_events(events).items():
        equity = float(starting_cash)
        open_until: pd.Timestamp | None = None
        for event in sorted(session_events, key=lambda item: item["decision_dt"]):
            if open_until is not None and event["decision_dt"] < open_until:
                continue
            action = int(event.get("oracle_action", 0))
            if action <= 0 or action > len(event["candidates"]):
                continue
            row = event["candidates"].iloc[action - 1]
            if float(row["entry_premium"]) > equity:
                continue
            trade = trade_from_proposal(row, score=float(event.get("oracle_weight", 1.0)), threshold=0.0, slippage_per_side=0.0, equity=equity, strategy=strategy)
            equity += trade["pnl"]
            trade["account_equity_after"] = equity
            trades.append(trade)
            open_until = pd.Timestamp(row["candidate_exit_dt"])
    return simulation_result(trades, events, skipped={}, starting_cash=starting_cash, strategy=strategy)


def trade_from_proposal(row: pd.Series, *, score: float, threshold: float, slippage_per_side: float, equity: float, strategy: str) -> dict[str, Any]:
    return {
        "candidate_uid": str(row.get("candidate_uid", "")),
        "trade_uid": str(row.get("trade_uid", "")),
        "split": str(row.get("reported_split", row.get("split", ""))),
        "session": str(row["session"]),
        "decision_time": pd.Timestamp(row["decision_dt"]).isoformat(),
        "exit_time": pd.Timestamp(row["candidate_exit_dt"]).isoformat(),
        "contract_id": str(row["contract_id"]),
        "right": str(row["right"]),
        "offset": float(row["offset"]),
        "score": float(score),
        "threshold": float(threshold),
        "entry_ask": float(row["entry_ask"]),
        "entry_premium": float(row["entry_premium"]),
        "entry_premium_with_slippage": float(row["entry_premium"] + float(slippage_per_side) * CONTRACT_MULTIPLIER),
        "account_equity_before": float(equity),
        "account_equity_after": float(equity) + float(row["candidate_pnl"]),
        "pnl": float(row["candidate_pnl"]),
        "raw_candidate_pnl": float(row["candidate_pnl"]),
        "slippage_per_side": float(slippage_per_side),
        "strategy": strategy,
        "source_policy": str(row["policy"]),
        "exit_reason": str(row.get("exit_reason", "")),
        "label_source": str(row.get("label_source", "")),
    }


def add_router_metrics(summary: dict[str, Any], trades: list[dict[str, Any]]) -> dict[str, Any]:
    out = p249.add_quality_metrics(summary, trades)
    out["source_policy_counts"] = {}
    out["source_policy_pnl"] = {}
    if trades:
        frame = pd.DataFrame(trades)
        for policy, group in frame.groupby("source_policy", sort=True):
            out["source_policy_counts"][str(policy)] = int(len(group))
            out["source_policy_pnl"][str(policy)] = float(pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0).sum())
    return out


def group_events(events: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault((str(event["split"]), int(event["seed"]), str(event["session"])), []).append(event)
    return grouped


def router_summary(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        rows = [result["splits"][split] for result in fold_results if not result.get("skipped") and split in result.get("splits", {})]
        if not rows:
            continue
        out[split] = {
            "median_delta_vs_protocol101_stream": float(np.median([row["delta_vs_protocol101_stream"] for row in rows])),
            "median_delta_vs_challenger_stream": float(np.median([row["delta_vs_challenger_stream"] for row in rows])),
            "median_oracle_gap": float(np.median([row["oracle_gap"] for row in rows])),
        }
    return out


def trade_profile(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    out = {
        "rows": int(len(frame)),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0.0).mean()),
        "median_entry_premium": float(pd.to_numeric(frame["entry_premium"], errors="coerce").median()),
        "by_source_policy": [],
        "by_side": [],
    }
    for key in ["source_policy", "right"]:
        if key not in frame.columns:
            continue
        rows = []
        for value, group in frame.groupby(key, sort=True):
            gpnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
            rows.append({"value": str(value), "trades": int(len(group)), "pnl": float(gpnl.sum()), "win_rate": float((gpnl > 0).mean())})
        out["by_source_policy" if key == "source_policy" else "by_side"] = rows
    return out


def event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "events": int(len(events)),
        "candidates": int(sum(len(event["candidates"]) for event in events)),
        "by_split": {
            split: int(sum(1 for event in events if event["split"] == split))
            for split in sorted({event["split"] for event in events})
        },
    }
    if any(event.get("router_fold") for event in events):
        summary["by_router_fold"] = {
            fold: int(sum(1 for event in events if event.get("router_fold") == fold))
            for fold in sorted({str(event.get("router_fold", "")) for event in events if event.get("router_fold")})
        }
    return summary


def serial_invariants(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"overlap_violations": 0, "unaffordable_violations": 0, "nan_time_rows": 0}
    tmp = frame.copy()
    tmp["decision_dt"] = pd.to_datetime(tmp["decision_time"], utc=True, errors="coerce")
    tmp["exit_dt"] = pd.to_datetime(tmp["exit_time"], utc=True, errors="coerce")
    overlap = 0
    for _, group in tmp.groupby(["fold", "seed", "reported_split", "session"], sort=False):
        previous_exit = None
        for row in group.sort_values("decision_dt").itertuples(index=False):
            if previous_exit is not None and pd.notna(row.decision_dt) and row.decision_dt < previous_exit:
                overlap += 1
            if pd.notna(row.exit_dt):
                previous_exit = row.exit_dt
    premium = pd.to_numeric(tmp["entry_premium"], errors="coerce")
    equity = pd.to_numeric(tmp["account_equity_before"], errors="coerce")
    return {
        "overlap_violations": int(overlap),
        "unaffordable_violations": int(((premium > equity) | premium.isna() | equity.isna()).sum()),
        "nan_time_rows": int(tmp["decision_dt"].isna().sum() + tmp["exit_dt"].isna().sum()),
    }


def decide(payload: dict[str, Any]) -> str:
    invariants = payload.get("invariants", {})
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_policy_router_invariant_failure"
    aggregate_payload = payload.get("aggregate", {})
    router = payload.get("router_summary", {})
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    beats_challenger = all(router.get(split, {}).get("median_delta_vs_challenger_stream", -1.0) > 0.0 for split in required if split in router)
    if aggregate_payload.get("promotion_ready") and beats_challenger:
        return "research_candidate_policy_router_beats_protocol101_and_challenger_streams"
    if aggregate_payload.get("promotion_ready"):
        return "research_only_policy_router_beats_protocol101_but_not_best_challenger_stream"
    return "rejected_policy_router_does_not_clear_protocol101_gate"


def assert_no_leakage_features(columns: list[str]) -> None:
    offenders = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        and column not in {"entry_premium"}
    ]
    if offenders:
        raise ValueError(f"router feature leakage risk: {offenders}")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    if not frame.empty:
        frame.to_csv(path, index=False)
    else:
        path.write_text("")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        "Other baseline: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 proposal stream",
        f"Data used: `{payload['data_used']}`",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Aggregate Vs Frozen Protocol101",
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
    lines.extend(["", "## Router Vs Proposal Streams", ""])
    for split, item in payload.get("router_summary", {}).items():
        lines.append(
            f"- {split}: delta vs Protocol101-stream {fmt(item['median_delta_vs_protocol101_stream'])}, "
            f"delta vs premium-blend-stream {fmt(item['median_delta_vs_challenger_stream'])}, "
            f"oracle gap {fmt(item['median_oracle_gap'])}"
        )
    lines.extend(
        [
            "",
            "## Oracle And Invariants",
            "",
            f"- Oracle summary: `{payload['oracle_summary']}`",
            f"- Invariants: `{payload['invariants']}`",
            "",
            "## Trade Profile",
            "",
            f"- Model: {payload['trade_profile']['model']}",
            f"- Union oracle: {payload['trade_profile']['union_oracle']}",
            f"- Protocol101 stream: {payload['trade_profile']['protocol101_stream']}",
            f"- Premium-blend stream: {payload['trade_profile']['challenger_stream']}",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'model_trades.csv'}`",
            f"- Union oracle trades: `{path.parent / 'union_oracle_trades.csv'}`",
            f"- Protocol101 stream trades: `{path.parent / 'protocol101_stream_trades.csv'}`",
            f"- Premium-blend stream trades: `{path.parent / 'challenger_stream_trades.csv'}`",
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
        f"- What: policy-router event model over Protocol101 and premium-blend proposal streams.\n"
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
