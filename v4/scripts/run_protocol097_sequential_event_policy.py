"""Protocol 097: sequential candidate-set policy with explicit wait action."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v4.model.serial_opportunity import (
    CONTRACT_MULTIPLIER,
    ENTRY_FEATURE_COLUMNS,
    SerialOpportunityConfig,
    serial_metrics,
    strict_serial_baseline,
)
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS


DEFAULT_PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_097_sequential_event_policy")
MODEL_SEEDS = [1, 2, 3, 4, 5]
MAX_CANDIDATES = 10


@dataclass(frozen=True)
class EventPolicyConfig:
    epochs: int = 16
    batch_size: int = 1024
    hidden_dim: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_validation_trades: int = 10
    utility_aux_weight: float = 0.0
    target_scale: float = 100.0
    target_clip: float = 800.0


class EventSetPolicy(nn.Module):
    """Permutation-aware candidate-set policy with a wait logit."""

    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
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
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--min-validation-trades", type=int, default=10)
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
                        "protocol": "097_sequential_event_policy",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": ENTRY_FEATURE_COLUMNS,
                        "config": config.__dict__,
                        "history": history,
                        "action_space": "wait plus up to 10 same-minute candidates",
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
                base = simulate_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    strategy=f"protocol097_{fold['name']}",
                )
                stress10 = simulate_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    strategy=f"protocol097_{fold['name']}_stress10",
                )
                stress25 = simulate_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    strategy=f"protocol097_{fold['name']}_stress25",
                )
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
        "protocol": "097_sequential_event_policy",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol092_dir": str(args.protocol092_dir),
        "pre_registration": _pre_registration(),
        "feature_columns": ENTRY_FEATURE_COLUMNS,
        "config": config.__dict__,
        "event_summary": _event_summary(events),
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


def build_events(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    events = []
    ordered = dataset.sort_values(["split", "seed", "session", "decision_dt", "entry_seed", "contract_id", "candidate_uid"])
    for key, group in ordered.groupby(["split", "seed", "session", "decision_time"], sort=False):
        group = group.sort_values(["entry_seed", "contract_id", "candidate_uid"]).head(MAX_CANDIDATES).copy()
        events.append(
            {
                "split": str(key[0]),
                "seed": int(key[1]),
                "session": str(key[2]),
                "decision_time": str(key[3]),
                "decision_dt": pd.Timestamp(group["decision_dt"].iloc[0]),
                "candidates": group.reset_index(drop=True),
            }
        )
    return events


def add_oracle_actions(events: list[dict[str, Any]]) -> dict[str, Any]:
    take_count = 0
    wait_count = 0
    by_split: dict[str, dict[str, int]] = {}
    for _, session_events in _group_events(events).items():
        session_events.sort(key=lambda event: event["decision_dt"])
        times = np.asarray([event["decision_dt"].value for event in session_events], dtype=np.int64)
        n = len(session_events)
        values = np.zeros(n + 1, dtype=float)
        actions = np.zeros(n, dtype=np.int64)
        weights = np.ones(n, dtype=float)
        for i in range(n - 1, -1, -1):
            wait_value = values[i + 1]
            best_value = wait_value
            best_action = 0
            best_take = -1e18
            for local_idx, row in session_events[i]["candidates"].iterrows():
                exit_ns = pd.Timestamp(row["candidate_exit_dt"]).value
                next_idx = int(np.searchsorted(times, exit_ns, side="left"))
                take_value = float(row["candidate_pnl"]) + values[next_idx]
                if take_value > best_take:
                    best_take = take_value
                if take_value > best_value:
                    best_value = take_value
                    best_action = int(local_idx) + 1
            values[i] = best_value
            actions[i] = best_action
            if best_action == 0:
                weights[i] = 1.0 + min(max(wait_value - best_take, 0.0) / 300.0, 5.0)
                wait_count += 1
            else:
                weights[i] = 1.0 + min(max(best_value - wait_value, 0.0) / 300.0, 5.0)
                take_count += 1
        for i, event in enumerate(session_events):
            event["oracle_action"] = int(actions[i])
            event["oracle_weight"] = float(weights[i])
            split = str(event["split"])
            by_split.setdefault(split, {"take": 0, "wait": 0})
            by_split[split]["take" if int(actions[i]) else "wait"] += 1
    return {"take_events": take_count, "wait_events": wait_count, "by_split": by_split}


def train_event_policy(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    seed: int,
    config: EventPolicyConfig,
    feature_columns: list[str] | None = None,
) -> tuple[EventSetPolicy, FeatureScaler, list[dict[str, Any]]]:
    feature_columns = feature_columns or ENTRY_FEATURE_COLUMNS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(np.vstack([event["candidates"][feature_columns].to_numpy(dtype=np.float32) for event in train_events]))
    x_train, mask_train, y_train, w_train, utility_train = event_tensors(train_events, scaler, config=config, feature_columns=feature_columns)
    x_val, mask_val, y_val, w_val, utility_val = event_tensors(validation_events, scaler, config=config, feature_columns=feature_columns)
    model = EventSetPolicy(input_dim=len(feature_columns), hidden_dim=config.hidden_dim)
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
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_mask, batch_y, batch_w, batch_utility in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_mask)
            loss = nn.functional.cross_entropy(logits, batch_y, reduction="none")
            loss = (loss * batch_w).mean()
            if config.utility_aux_weight > 0.0:
                candidate_logits = logits[:, 1:]
                utility_loss = nn.functional.huber_loss(
                    candidate_logits[batch_mask],
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
                utility_val_t = torch.from_numpy(utility_val)
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
        history.append(
            {
                "epoch": epoch,
                "train_weighted_ce": float(np.mean(losses)) if losses else math.nan,
                "validation_weighted_ce": val_loss,
                "is_best": val_loss <= best_val,
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def event_tensors(
    events: list[dict[str, Any]],
    scaler: FeatureScaler,
    *,
    config: EventPolicyConfig | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_columns = feature_columns or ENTRY_FEATURE_COLUMNS
    x = np.zeros((len(events), MAX_CANDIDATES, len(feature_columns)), dtype=np.float32)
    mask = np.zeros((len(events), MAX_CANDIDATES), dtype=bool)
    y = np.zeros(len(events), dtype=np.int64)
    weights = np.ones(len(events), dtype=np.float32)
    utility = np.zeros((len(events), MAX_CANDIDATES), dtype=np.float32)
    scale = float(config.target_scale) if config is not None else 100.0
    clip = float(config.target_clip) if config is not None else 800.0
    for i, event in enumerate(events):
        raw = event["candidates"][feature_columns].to_numpy(dtype=np.float32)
        n = min(len(raw), MAX_CANDIDATES)
        x[i, :n, :] = scaler.transform(raw[:n])
        mask[i, :n] = True
        y[i] = int(event.get("oracle_action", 0))
        weights[i] = float(event.get("oracle_weight", 1.0))
        pnl = event["candidates"]["candidate_pnl"].to_numpy(dtype=np.float32)[:n]
        utility[i, :n] = np.clip(pnl, -clip, clip) / scale
    return x, mask, y, weights, utility


def select_margin_threshold(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: FeatureScaler,
    *,
    seed: int,
    config: EventPolicyConfig,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    feature_columns = feature_columns or ENTRY_FEATURE_COLUMNS
    margins = event_margins(events, model, scaler, feature_columns=feature_columns)
    finite = margins[np.isfinite(margins)]
    if len(finite) == 0:
        thresholds = [float("inf")]
    else:
        thresholds = sorted(set(np.quantile(finite, [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]).round(4).tolist() + [0.0, float(finite.min()) - 1e-3]))
    baseline = strict_serial_baseline(_candidate_frame_from_events(events), seed=int(seed), slippage_per_side=0.0)
    baseline10 = strict_serial_baseline(_candidate_frame_from_events(events), seed=int(seed), slippage_per_side=0.10)
    sweep = []
    for threshold in thresholds:
        base = simulate_event_policy(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, strategy="protocol097_validation", feature_columns=feature_columns)
        stress = simulate_event_policy(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, strategy="protocol097_validation_stress10", feature_columns=feature_columns)
        sweep.append(
            {
                "threshold": float(threshold),
                "model": base.summary,
                "model_stress_0_10": stress.summary,
                "strict_serial_baseline": baseline.summary,
                "strict_serial_baseline_stress_0_10": baseline10.summary,
                "delta_vs_baseline": float(base.summary["total_pnl"] - baseline.summary["total_pnl"]),
                "delta_vs_baseline_stress_0_10": float(stress.summary["total_pnl"] - baseline10.summary["total_pnl"]),
            }
        )
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["delta_vs_baseline_stress_0_10"], row["delta_vs_baseline"], row["model"]["total_pnl"], row["model"]["trades"]))
    return {
        "threshold": float(best["threshold"]),
        "source_seed": int(seed),
        "source_rows": int(len(events)),
        "objective": "validation stress_0_10 delta vs strict serial baseline",
        "selected": best,
        "sweep": sweep,
    }


def event_margins(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: FeatureScaler,
    *,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    if not events:
        return np.asarray([], dtype=float)
    x, mask, _, _, _ = event_tensors(events, scaler, feature_columns=feature_columns or ENTRY_FEATURE_COLUMNS)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(events), 4096):
            logits = model(torch.from_numpy(x[start : start + 4096]), torch.from_numpy(mask[start : start + 4096])).cpu().numpy()
            wait = logits[:, 0]
            take = np.max(logits[:, 1:], axis=1)
            out.append(take - wait)
    return np.concatenate(out)


def simulate_event_policy(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: FeatureScaler,
    *,
    threshold: float,
    slippage_per_side: float,
    strategy: str,
    feature_columns: list[str] | None = None,
) -> Any:
    trades: list[dict[str, Any]] = []
    round_trip_slippage = float(slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER
    grouped = _group_events(events)
    for _, session_events in grouped.items():
        session_events = sorted(session_events, key=lambda event: event["decision_dt"])
        open_until: pd.Timestamp | None = None
        for event in session_events:
            if open_until is not None and event["decision_dt"] < open_until:
                continue
            action, margin = predict_event_action(event, model, scaler, feature_columns=feature_columns or ENTRY_FEATURE_COLUMNS)
            if action <= 0 or margin < threshold:
                continue
            row = event["candidates"].iloc[action - 1]
            pnl = float(row["candidate_pnl"]) - round_trip_slippage
            trades.append(
                {
                    "candidate_uid": str(row["candidate_uid"]),
                    "trade_uid": str(row["trade_uid"]),
                    "split": str(row["split"]),
                    "seed": int(row["seed"]),
                    "entry_seed": int(row["entry_seed"]),
                    "session": str(row["session"]),
                    "decision_time": pd.Timestamp(row["decision_dt"]).isoformat(),
                    "exit_time": pd.Timestamp(row["candidate_exit_dt"]).isoformat(),
                    "contract_id": str(row["contract_id"]),
                    "right": str(row["right"]),
                    "offset": float(row["offset"]),
                    "score": float(margin),
                    "threshold": float(threshold),
                    "pnl": pnl,
                    "raw_candidate_pnl": float(row["candidate_pnl"]),
                    "slippage_per_side": float(slippage_per_side),
                    "strategy": strategy,
                    "exit_reason": str(row["candidate_exit_reason"]),
                    "label_source": str(row["label_source"]),
                }
            )
            open_until = pd.Timestamp(row["candidate_exit_dt"])
    summary = serial_metrics(trades)
    summary.update({"input_events": int(len(events)), "max_concurrent_positions": 1 if trades else 0, "serial_status": "pass"})
    return type("EventSimulationResult", (), {"trades": trades, "summary": summary})()


def predict_event_action(
    event: dict[str, Any],
    model: EventSetPolicy,
    scaler: FeatureScaler,
    *,
    feature_columns: list[str] | None = None,
) -> tuple[int, float]:
    x, mask, _, _, _ = event_tensors([event], scaler, feature_columns=feature_columns or ENTRY_FEATURE_COLUMNS)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(mask)).cpu().numpy()[0]
    wait = float(logits[0])
    candidate_logits = logits[1:]
    action = int(np.argmax(candidate_logits)) + 1
    margin = float(candidate_logits[action - 1] - wait)
    return action, margin


def _group_events(events: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault((str(event["split"]), int(event["seed"]), str(event["session"])), []).append(event)
    return grouped


def _reported_event_slices(events: list[dict[str, Any]], fold: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    test = [event for event in events if event["split"] == fold["test_split"] and int(event["seed"]) == int(seed)]
    out = {fold["test_split"]: test}
    if fold["test_split"] == "q1_2026":
        out["march_2026"] = [event for event in test if str(event["session"]) >= "2026-03-01"]
    return out


def _candidate_frame_from_events(events: list[dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    return pd.concat([event["candidates"] for event in events], ignore_index=True, sort=False)


def _aggregate_gate(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        rows = []
        for result in fold_results:
            if split in result["splits"]:
                rows.append({"seed": result["seed"], "fold": result["fold"], **result["splits"][split]})
        aggregate[split] = _summarize_split(rows)
    aggregate["promotion_checks"] = _promotion_checks(aggregate)
    aggregate["promotion_ready"] = bool(all(item["pass"] for item in aggregate["promotion_checks"]))
    return aggregate


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"seeds": 0}
    base_pnl = _arr(row["model"]["total_pnl"] for row in rows)
    stress10 = _arr(row["model_stress_0_10"]["total_pnl"] for row in rows)
    stress25 = _arr(row["model_stress_0_25"]["total_pnl"] for row in rows)
    pf = _arr(_finite_pf(row["model"]["profit_factor"]) for row in rows)
    trades = _arr(row["model"]["trades"] for row in rows)
    baseline = _arr(row["strict_serial_baseline"]["total_pnl"] for row in rows)
    return {
        "seeds": int(len(rows)),
        "median_total_pnl": float(np.median(base_pnl)),
        "positive_seed_fraction": float((base_pnl > 0).mean()),
        "median_profit_factor": float(np.median(pf)),
        "median_trades": float(np.median(trades)),
        "median_stress_0_10_total_pnl": float(np.median(stress10)),
        "median_stress_0_25_total_pnl": float(np.median(stress25)),
        "strict_serial_baseline_median_total_pnl": float(np.median(baseline)),
        "beats_strict_serial_baseline": bool(float(np.median(base_pnl)) > float(np.median(baseline))),
        "total_side_counts": {
            "C": int(sum(row["model"]["side_counts"].get("C", 0) for row in rows)),
            "P": int(sum(row["model"]["side_counts"].get("P", 0) for row in rows)),
        },
        "seed_rows": [
            {
                "seed": int(row["seed"]),
                "fold": row["fold"],
                "model_total_pnl": float(row["model"]["total_pnl"]),
                "model_profit_factor": float(row["model"]["profit_factor"]),
                "model_trades": int(row["model"]["trades"]),
                "stress_0_10_total_pnl": float(row["model_stress_0_10"]["total_pnl"]),
                "stress_0_25_total_pnl": float(row["model_stress_0_25"]["total_pnl"]),
                "strict_serial_baseline_total_pnl": float(row["strict_serial_baseline"]["total_pnl"]),
            }
            for row in rows
        ],
    }


def _promotion_checks(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        item = aggregate.get(split, {})
        checks.extend(
            [
                {"split": split, "name": "positive_median_pnl", "value": item.get("median_total_pnl", 0.0), "pass": item.get("median_total_pnl", 0.0) > 0.0},
                {"split": split, "name": "positive_seed_fraction_ge_0_80", "value": item.get("positive_seed_fraction", 0.0), "pass": item.get("positive_seed_fraction", 0.0) >= 0.80},
                {"split": split, "name": "median_profit_factor_ge_1_15", "value": item.get("median_profit_factor", 0.0), "pass": item.get("median_profit_factor", 0.0) >= 1.15},
                {"split": split, "name": "positive_stress_0_10_median_pnl", "value": item.get("median_stress_0_10_total_pnl", 0.0), "pass": item.get("median_stress_0_10_total_pnl", 0.0) > 0.0},
                {"split": split, "name": "beats_strict_serial_baseline", "value": item.get("median_total_pnl", 0.0) - item.get("strict_serial_baseline_median_total_pnl", 0.0), "pass": bool(item.get("beats_strict_serial_baseline", False))},
                {"split": split, "name": "reported_stress_0_25_positive", "value": item.get("median_stress_0_25_total_pnl", 0.0), "pass": item.get("median_stress_0_25_total_pnl", 0.0) > 0.0},
            ]
        )
    return checks


def _compare_to_protocol092(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        now = current[split]
        old = previous[split]
        out[split] = {
            "median_total_pnl_delta": float(now["median_total_pnl"] - old["median_total_pnl"]),
            "median_profit_factor_delta": float(now["median_profit_factor"] - old["median_profit_factor"]),
            "median_trades_delta": float(now["median_trades"] - old["median_trades"]),
            "stress_0_10_delta": float(now["median_stress_0_10_total_pnl"] - old["median_stress_0_10_total_pnl"]),
            "baseline_beat_before": bool(old["beats_strict_serial_baseline"]),
            "baseline_beat_after": bool(now["beats_strict_serial_baseline"]),
        }
    return out


def _decision(aggregate: dict[str, Any], previous: dict[str, Any]) -> str:
    if aggregate["promotion_ready"]:
        return "keep_promote_candidate: Protocol 097 clears the strict serial gate"
    comparison = _compare_to_protocol092(aggregate, previous)
    if comparison["q3_2025"]["median_total_pnl_delta"] > 0 and comparison["q4_2025"]["median_total_pnl_delta"] > 0:
        return "keep_for_research_only: Protocol 097 improves Q3/Q4 but still does not clear promotion"
    return "reject: sequential event policy did not improve both Q3 and Q4"


def _pre_registration() -> dict[str, Any]:
    return {
        "hypothesis": "Independent candidate scoring is hitting a ceiling. A sequential event policy with an explicit wait action can learn take-vs-wait from a train-period dynamic-programming oracle while keeping Protocol 081 exits frozen.",
        "model_class": "candidate-set neural policy with wait action",
        "oracle": "train-period session DP maximizing frozen candidate PnL with one open position",
        "features": "unchanged Protocol 092 entry-only feature set",
        "exits": "frozen Protocol 081 candidate exits",
        "paid_data": "forbidden",
        "live_orders": "forbidden",
        "folds": FOLDS,
        "seeds": MODEL_SEEDS,
    }


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = np.asarray([len(event["candidates"]) for event in events], dtype=float)
    return {
        "events": int(len(events)),
        "mean_candidates": float(sizes.mean()) if len(sizes) else 0.0,
        "max_candidates": int(sizes.max()) if len(sizes) else 0,
        "splits": {split: int(sum(1 for event in events if event["split"] == split)) for split in sorted(set(event["split"] for event in events))},
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 097: Sequential Event Policy",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Model class: `{payload['pre_registration']['model_class']}`",
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
        "## Event / Oracle Summary",
        "",
        "```json",
        json.dumps({"event_summary": payload["event_summary"], "oracle_summary": payload["oracle_summary"]}, indent=2, sort_keys=True),
        "```",
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


def _finite_pf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(out):
        return 999.0
    return out if math.isfinite(out) else 0.0


def _arr(values) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
