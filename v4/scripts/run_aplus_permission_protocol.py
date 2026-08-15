"""Run Protocol 003C: learned A+ entry permission.

Protocol 003 showed that A+ timing/value features help the neural policy, but
the fixed one-trade-per-day stress failed because the first qualifying trade was
often weak. This script keeps the same base A+ surface model and adds one
pre-registered permission layer:

* base surface model: January train, early-February calibration
* permission model: base proposals from January train, early-February validation
* permission threshold: selected on late-February only
* March and frozen Q4 2025: audit-only

No new data is downloaded, and no March/Q4 metric may choose a threshold.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.model.environment_diagnostics import time_bucket
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceVariant,
    bootstrap_trade_pnl,
    predict_surface_actions,
    stress_trades,
    summarize_random_baseline,
    token_feature_names,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import FeatureScaler, PilotConfig, Trade
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_permission_protocol_003c"
_NY = ZoneInfo("America/New_York")
PERMISSION_TRIAL = ProtocolTrial(
    name="fixed_permission_all_times_max4",
    allowed_buckets=("first_30", "post_open_morning", "midday", "late_afternoon"),
    min_edge_vs_no_trade=0.0,
    max_trades_per_day=4,
    daily_loss_stop=None,
)
THRESHOLD_GRID = tuple(round(x, 2) for x in np.arange(0.35, 0.91, 0.05))
SELECTED_TOKEN_FEATURES = (
    "pattern_count_norm",
    "abs_delta",
    "delta_atr_capture",
    "convexity_per_premium",
    "theta_burden_hold",
    "spread_tax",
    "breakeven_atr",
    "gamma_theta_ratio_scaled",
    "contract_value_score",
    "worth_spread_flag",
    "obvious_overpay_flag",
)


@dataclass
class PermissionExample:
    session: str
    decision_time: datetime
    features: np.ndarray
    target: float
    pnl: float
    token_idx: int
    edge: float


class PermissionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_permission_protocol_003c"))
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--permission-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--permission-batch-size", type=int, default=2048)
    return parser.parse_args()


def _variant() -> SurfaceVariant:
    return SurfaceVariant(
        name="surface_structure_aplus_huber",
        action_space="surface",
        market_mode="structure",
        token_mode="aplus",
        loss_mode="huber",
    )


def _safe(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def _local_time_features(decision_time: datetime) -> list[float]:
    local = decision_time.astimezone(_NY)
    minutes = local.hour * 60 + local.minute
    start = 9 * 60 + 30
    end = 15 * 60 + 30
    progress = min(max((minutes - start) / max(end - start, 1), 0.0), 1.0)
    bucket = time_bucket(decision_time)
    return [
        progress,
        1.0 - progress,
        math.sin(2.0 * math.pi * progress),
        math.cos(2.0 * math.pi * progress),
        float(bucket == "first_30"),
        float(bucket == "post_open_morning"),
        float(bucket == "midday"),
        float(bucket == "late_afternoon"),
    ]


def _top_action(
    decision,
    pred: np.ndarray,
    *,
    min_edge: float = 0.0,
) -> tuple[int, int, float, float, float, float] | None:
    action_mask = np.concatenate([[True], decision.token_mask])
    masked = np.asarray(pred, dtype=float).copy()
    masked[~action_mask] = -np.inf
    if not np.isfinite(masked).any():
        return None
    action = int(np.nanargmax(masked))
    if action == 0:
        return None
    edge = float(masked[action] - masked[0])
    if not np.isfinite(edge) or edge < min_edge:
        return None
    token_idx = action - 1
    pnl = float(decision.labels[token_idx])
    if not np.isfinite(pnl):
        return None
    finite_scores = np.sort(masked[np.isfinite(masked)])
    margin = float(finite_scores[-1] - finite_scores[-2]) if len(finite_scores) >= 2 else 0.0
    return action, token_idx, edge, pnl, float(masked[0]), margin


def _permission_feature(
    decision,
    *,
    token_idx: int,
    edge: float,
    flat_score: float,
    margin: float,
    proposal_count_before: int,
    feature_names: Sequence[str],
) -> np.ndarray:
    token = np.asarray(decision.token_features[token_idx], dtype=float)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    selected = [
        _safe(token[feature_index[name]]) if name in feature_index and feature_index[name] < len(token) else 0.0
        for name in SELECTED_TOKEN_FEATURES
    ]
    right = str(decision.rights[token_idx])
    offset = float(decision.offsets[token_idx])
    moneyness_steps = offset / 5.0 if right == "C" else -offset / 5.0
    side = [float(right == "C"), float(right == "P")]
    interactions = [
        edge * selected[0] / 100.0,
        edge * selected[7] / 100.0,
        edge * selected[8] / 100.0,
        float(selected[9] > 0.0 and selected[10] <= 0.0),
    ]
    out = [
        edge / 100.0,
        flat_score / 100.0,
        margin / 100.0,
        proposal_count_before / 20.0,
        offset / 50.0,
        abs(offset) / 50.0,
        moneyness_steps / 10.0,
        float(moneyness_steps < 0.0),
        float(moneyness_steps == 0.0),
        float(moneyness_steps > 0.0),
        float(decision.pattern_targets[token_idx]),
        float(decision.value_targets[token_idx]),
        *side,
        *_local_time_features(decision.decision_time),
        *selected,
        *interactions,
    ]
    return np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=8.0, neginf=-8.0)


def _proposal_examples(
    decisions: Sequence,
    predictions: np.ndarray,
    *,
    feature_names: Sequence[str],
    target_extra_cost: float = 25.0,
) -> list[PermissionExample]:
    examples: list[PermissionExample] = []
    proposal_count: dict[str, int] = {}
    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in set(PERMISSION_TRIAL.allowed_buckets):
            continue
        top = _top_action(decision, pred, min_edge=PERMISSION_TRIAL.min_edge_vs_no_trade)
        if top is None:
            continue
        _, token_idx, edge, pnl, flat_score, margin = top
        count_before = proposal_count.get(decision.session, 0)
        features = _permission_feature(
            decision,
            token_idx=token_idx,
            edge=edge,
            flat_score=flat_score,
            margin=margin,
            proposal_count_before=count_before,
            feature_names=feature_names,
        )
        proposal_count[decision.session] = count_before + 1
        examples.append(
            PermissionExample(
                session=decision.session,
                decision_time=decision.decision_time,
                features=features,
                target=float((pnl - target_extra_cost) > 0.0),
                pnl=pnl,
                token_idx=token_idx,
                edge=edge,
            )
        )
    return examples


def _stack(examples: Sequence[PermissionExample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([example.features for example in examples]).astype(np.float32)
    y = np.asarray([example.target for example in examples], dtype=np.float32)
    return x, y


def _train_permission_model(
    train_examples: Sequence[PermissionExample],
    validation_examples: Sequence[PermissionExample],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[PermissionMLP, FeatureScaler, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train_raw, y_train = _stack(train_examples)
    x_val_raw, y_val = _stack(validation_examples)
    scaler = FeatureScaler.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    model = PermissionMLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.binary_cross_entropy_with_logits(model(x_val_t), y_val_t, pos_weight=pos_weight).detach().cpu())
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_bce": float(np.mean(losses)),
                "validation_bce": val_loss,
                "is_best": is_best,
                "train_positive_fraction": float(y_train.mean()),
                "validation_positive_fraction": float(y_val.mean()),
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def _permission_probability(
    model: PermissionMLP,
    scaler: FeatureScaler,
    feature: np.ndarray,
) -> float:
    x = torch.from_numpy(scaler.transform(feature.reshape(1, -1)))
    model.eval()
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).cpu().numpy()[0])


def _simulate_permission_policy(
    decisions: Sequence,
    predictions: np.ndarray,
    *,
    permission_model: PermissionMLP,
    permission_scaler: FeatureScaler,
    threshold: float,
    cooldown_minutes: int,
    feature_names: Sequence[str],
    strategy: str,
) -> tuple[list[Trade], list[dict]]:
    trades: list[Trade] = []
    rows: list[dict] = []
    allowed = set(PERMISSION_TRIAL.allowed_buckets)
    proposal_count: dict[str, int] = {}
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in allowed:
            continue
        top = _top_action(decision, pred, min_edge=PERMISSION_TRIAL.min_edge_vs_no_trade)
        if top is None:
            continue
        _, token_idx, edge, pnl, flat_score, margin = top
        count_before = proposal_count.get(decision.session, 0)
        proposal_count[decision.session] = count_before + 1
        if trades_by_session.get(decision.session, 0) >= PERMISSION_TRIAL.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        feature = _permission_feature(
            decision,
            token_idx=token_idx,
            edge=edge,
            flat_score=flat_score,
            margin=margin,
            proposal_count_before=count_before,
            feature_names=feature_names,
        )
        probability = _permission_probability(permission_model, permission_scaler, feature)
        if probability < threshold:
            continue
        trade = Trade(
            session=decision.session,
            decision_time=decision.decision_time.isoformat(),
            pnl=pnl,
            score=probability,
            right=str(decision.rights[token_idx]),
            offset=float(decision.offsets[token_idx]),
            strategy=strategy,
        )
        token = np.asarray(decision.token_features[token_idx], dtype=float)
        index = {name: idx for idx, name in enumerate(feature_names)}
        selected_features = {
            f"feature_{name}": _safe(token[index[name]])
            for name in SELECTED_TOKEN_FEATURES
            if name in index and index[name] < len(token)
        }
        trades.append(trade)
        rows.append(
            {
                **trade.__dict__,
                "permission_probability": probability,
                "base_edge": edge,
                "contract_id": str(decision.contract_ids[token_idx]),
                "token_idx": int(token_idx),
                "pattern_present": float(decision.pattern_targets[token_idx]),
                "worth_spread_target": float(decision.value_targets[token_idx]),
                "proposal_count_before": int(count_before),
                **selected_features,
            }
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
    return trades, rows


def _selection_reward(metrics: dict) -> float:
    trades = float(metrics["trades"])
    if trades < 8:
        return -1_000_000.0 + trades
    pf = float(metrics["profit_factor"])
    if not np.isfinite(pf):
        pf = 5.0
    return (
        float(metrics["total_pnl"])
        + 1_000.0 * (min(pf, 5.0) - 1.0)
        + 1_500.0 * (float(metrics["positive_day_fraction"]) - 0.50)
        + 0.15 * float(metrics["max_drawdown"])
        - 1_500.0 * max(0.0, float(metrics.get("top_day_profit_share", 1.0)) - 0.45)
    )


def _choose_threshold(
    decisions: Sequence,
    predictions: np.ndarray,
    *,
    permission_model: PermissionMLP,
    permission_scaler: FeatureScaler,
    cooldown_minutes: int,
    feature_names: Sequence[str],
) -> tuple[float, list[dict]]:
    sweep = []
    for threshold in THRESHOLD_GRID:
        trades, _ = _simulate_permission_policy(
            decisions,
            predictions,
            permission_model=permission_model,
            permission_scaler=permission_scaler,
            threshold=float(threshold),
            cooldown_minutes=cooldown_minutes,
            feature_names=feature_names,
            strategy=f"{LOOP_ID}:threshold_sweep",
        )
        metrics = metrics_with_concentration(trades)
        sweep.append({"threshold": float(threshold), "metrics": metrics, "selection_reward": _selection_reward(metrics)})
    ranked = sorted(
        sweep,
        key=lambda row: (
            row["selection_reward"],
            row["metrics"]["total_pnl"],
            min(float(row["metrics"]["profit_factor"]), 5.0) if np.isfinite(row["metrics"]["profit_factor"]) else 5.0,
            -abs(row["threshold"] - 0.55),
        ),
        reverse=True,
    )
    return float(ranked[0]["threshold"]), sweep


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize_split(seed_rows: list[dict], split: str) -> dict:
    metrics = [row["metrics_by_split"][split] for row in seed_rows]
    randoms = [row["random_baseline_by_split"][split] for row in seed_rows]
    pnl = [float(row["total_pnl"]) for row in metrics]
    pf = [float(row["profit_factor"]) for row in metrics]
    trades = [float(row["trades"]) for row in metrics]
    random_pnl = [float(row["total_pnl_median"]) for row in randoms]
    return {
        "pnl_by_seed": pnl,
        "pnl_median": _median(pnl),
        "profit_factor_by_seed": pf,
        "profit_factor_median": _median(pf),
        "trades_by_seed": trades,
        "trades_median": _median(trades),
        "positive_seed_fraction": float(np.mean([x > 0 for x in pnl])) if pnl else 0.0,
        "positive_day_fraction_median": _median([float(row["positive_day_fraction"]) for row in metrics]),
        "top_day_share_median": _median([float(row["top_day_profit_share"]) for row in metrics]),
        "random_pnl_by_seed": random_pnl,
        "random_pnl_median": _median(random_pnl),
        "edge_vs_random_median": _median(pnl) - _median(random_pnl),
    }


def _summarize_stress(seed_rows: list[dict], split: str) -> dict:
    out = {}
    for cost in ("25", "50", "100"):
        metrics = [row["slippage_stress_by_split"][split][cost] for row in seed_rows]
        pnl = [float(row["total_pnl"]) for row in metrics]
        pf = [float(row["profit_factor"]) for row in metrics]
        out[cost] = {
            "pnl_by_seed": pnl,
            "pnl_median": _median(pnl),
            "profit_factor_by_seed": pf,
            "profit_factor_median": _median(pf),
            "survives": bool(_median(pnl) > 0.0 and _median(pf) >= 1.05),
        }
    return out


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# A+ Permission Protocol 003C",
        "",
        payload["framing"],
        "",
        f"Passes permission gate: `{payload['passes_permission_gate']}`",
        "",
        "## Split Summary",
        "",
        "| Split | Median PnL | Median PF | Trades | Positive Seeds | Random PnL | Edge vs Random |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("selection", "march", "q4"):
        row = payload["summary_by_split"][split]
        lines.append(
            f"| {split} | {row['pnl_median']:.0f} | {row['profit_factor_median']:.3f} | "
            f"{row['trades_median']:.0f} | {row['positive_seed_fraction']:.2f} | "
            f"{row['random_pnl_median']:.0f} | {row['edge_vs_random_median']:.0f} |"
        )
    lines += [
        "",
        "## Extra Cost Stress",
        "",
        "| Split | Extra Cost | Median PnL | Median PF | Survives |",
        "|---|---:|---:|---:|---|",
    ]
    for split in ("march", "q4"):
        for cost, row in payload["stress_by_split"][split].items():
            lines.append(
                f"| {split} | {cost} | {row['pnl_median']:.0f} | {row['profit_factor_median']:.3f} | {row['survives']} |"
            )
    lines += [
        "",
        "## Thresholds",
        "",
        "| Seed | Threshold | Permission Train Positive | Permission Validation Positive |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["seed_rows"]:
        hist = row["permission_history"][-1] if row["permission_history"] else {}
        lines.append(
            f"| {row['seed']} | {row['chosen_threshold']:.2f} | "
            f"{hist.get('train_positive_fraction', 0.0):.2f} | {hist.get('validation_positive_fraction', 0.0):.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    q4_paths = sorted(args.q4_data_dir.glob("*.pkl"))
    if not q4_paths:
        raise SystemExit(f"no Q4 pkl files found under {args.q4_data_dir}")
    window = _protocol_window(paths, q4_paths)
    variant = _variant()
    policy_name, cooldown = POLICY_META[args.policy_index]
    market_cache = MarketStructureCache()
    cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    feature_names = token_feature_names(variant.token_mode)

    train_decisions = _load_surface_decisions_cached(paths["train"], policy_index=args.policy_index, variant=variant, market_cache=market_cache, split="train", cache_dir=cache_dir)
    validation_decisions = _load_surface_decisions_cached(paths["validation"], policy_index=args.policy_index, variant=variant, market_cache=market_cache, split="validation", cache_dir=cache_dir)
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    march_decisions = _load_surface_decisions_cached(paths["test"], policy_index=args.policy_index, variant=variant, market_cache=market_cache, split="march", cache_dir=cache_dir)
    q4_decisions = _load_surface_decisions_cached(q4_paths, policy_index=args.policy_index, variant=variant, market_cache=market_cache, split="q4", cache_dir=cache_dir)
    decision_sets = {
        "train": train_decisions,
        "calibration": calibration_decisions,
        "selection": selection_decisions,
        "march": march_decisions,
        "q4": q4_decisions,
    }

    seed_rows = []
    selected_trades = {"selection": [], "march": [], "q4": []}
    for seed in args.seeds:
        effective_seed = window_seed(seed, window.window_id)
        print(f"{LOOP_ID} seed={seed}", flush=True)
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        base_model, standardizer, base_history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=variant,
        )
        predictions = {
            split: predict_surface_actions(base_model, standardizer, decisions, target_scale=config.target_scale)
            for split, decisions in decision_sets.items()
        }
        train_examples = _proposal_examples(train_decisions, predictions["train"], feature_names=feature_names)
        validation_examples = _proposal_examples(calibration_decisions, predictions["calibration"], feature_names=feature_names)
        permission_model, permission_scaler, permission_history = _train_permission_model(
            train_examples,
            validation_examples,
            seed=effective_seed,
            epochs=args.permission_epochs,
            batch_size=args.permission_batch_size,
        )
        threshold, threshold_sweep = _choose_threshold(
            selection_decisions,
            predictions["selection"],
            permission_model=permission_model,
            permission_scaler=permission_scaler,
            cooldown_minutes=cooldown,
            feature_names=feature_names,
        )
        metrics_by_split = {}
        random_baseline_by_split = {}
        slippage_stress_by_split = {}
        bootstrap_by_split = {}
        for split in ("selection", "march", "q4"):
            trades, rows = _simulate_permission_policy(
                decision_sets[split],
                predictions[split],
                permission_model=permission_model,
                permission_scaler=permission_scaler,
                threshold=threshold,
                cooldown_minutes=cooldown,
                feature_names=feature_names,
                strategy=f"{LOOP_ID}:permission_threshold_{threshold:.2f}",
            )
            for row in rows:
                selected_trades[split].append({"seed": int(seed), **row})
            metrics_by_split[split] = metrics_with_concentration(trades)
            random_baseline_by_split[split] = summarize_random_baseline(
                decision_sets[split],
                trial=PERMISSION_TRIAL,
                cooldown_minutes=cooldown,
                seed=effective_seed,
                target_trade_count=len(trades),
            )
            if split in {"march", "q4"}:
                bootstrap_by_split[split] = bootstrap_trade_pnl(trades, seed=effective_seed)
                slippage_stress_by_split[split] = {
                    str(extra_cost): metrics_with_concentration(
                        stress_trades(trades, extra_cost_per_trade=float(extra_cost))
                    )
                    for extra_cost in (25, 50, 100)
                }
        seed_rows.append(
            {
                "seed": int(seed),
                "effective_seed": int(effective_seed),
                "chosen_threshold": float(threshold),
                "base_best_epoch": next((x["epoch"] for x in base_history if x["is_best"]), None),
                "permission_best_epoch": next((x["epoch"] for x in permission_history if x["is_best"]), None),
                "base_history": base_history,
                "permission_history": permission_history,
                "threshold_sweep": threshold_sweep,
                "proposal_counts": {
                    "train": len(train_examples),
                    "calibration": len(validation_examples),
                },
                "metrics_by_split": metrics_by_split,
                "random_baseline_by_split": random_baseline_by_split,
                "slippage_stress_by_split": slippage_stress_by_split,
                "bootstrap_by_split": bootstrap_by_split,
            }
        )

    summary_by_split = {split: _summarize_split(seed_rows, split) for split in ("selection", "march", "q4")}
    stress_by_split = {split: _summarize_stress(seed_rows, split) for split in ("march", "q4")}
    passes = bool(
        summary_by_split["selection"]["pnl_median"] > 0
        and summary_by_split["march"]["pnl_median"] > 0
        and summary_by_split["march"]["profit_factor_median"] >= 1.05
        and summary_by_split["march"]["positive_seed_fraction"] >= 2 / 3
        and summary_by_split["q4"]["pnl_median"] > 0
        and summary_by_split["q4"]["profit_factor_median"] >= 1.05
        and summary_by_split["q4"]["positive_seed_fraction"] >= 2 / 3
        and stress_by_split["march"]["25"]["survives"]
        and stress_by_split["q4"]["25"]["survives"]
    )
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Pre-registered learned entry-permission layer for the Protocol 003 A+ champion. "
            "The permission layer trains on pre-March base proposals and selects its threshold on late February only."
        ),
        "window": asdict(window) | {"window_id": window.window_id},
        "variant": asdict(variant) | {"variant_id": variant.variant_id},
        "policy_index": int(args.policy_index),
        "policy_name": policy_name,
        "trial": asdict(PERMISSION_TRIAL) | {"config_id": PERMISSION_TRIAL.config_id},
        "threshold_grid": list(THRESHOLD_GRID),
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "out_dir": str(args.out_dir),
            "seeds": args.seeds,
            "epochs": args.epochs,
            "permission_epochs": args.permission_epochs,
            "batch_size": args.batch_size,
            "permission_batch_size": args.permission_batch_size,
        },
        "summary_by_split": summary_by_split,
        "stress_by_split": stress_by_split,
        "passes_permission_gate": passes,
        "interpretation": (
            "A pass would mean a learned permission layer improved entry selectivity without using March/Q4. "
            "A fail means the current A+ lead still lacks a stable first-entry permission rule."
        ),
        "seed_rows": seed_rows,
        "selected_trades": selected_trades,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, payload)
    for split, rows in selected_trades.items():
        (args.out_dir / f"selected_trades_{split}.json").write_text(
            json.dumps(rows, indent=2, allow_nan=True) + "\n"
        )
    print(json_path)
    print(md_path)
    print(json.dumps({"passes_permission_gate": passes, "summary_by_split": summary_by_split}, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
