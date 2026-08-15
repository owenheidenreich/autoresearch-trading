"""Decision-level SPXW 0DTE action pilot.

This module is deliberately narrower than the candidate-ranking pilot. Each
minute becomes one action decision:

    no trade, nearest ATM call, nearest ATM put

The point is to test whether a simpler side/risk decision can recover obvious
structure that the candidate-level scorer missed.
"""
from __future__ import annotations

import copy
import json
import math
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import (
    FeatureScaler,
    PilotConfig,
    Trade,
    candidate_feature_vector,
    metrics_for_trades,
    session_from_path,
    split_name,
)


ACTION_NAMES = ("no_trade", "call", "put")
FEATURE_VERSION = "action_v1"
DECISION_AWARE_FEATURE_VERSION = "action_v2_decision_aware"
ACTION_DECISION_LOSS_CONFIG = {
    "regression_loss": "huber",
    "decision_ce_weight": 0.20,
    "false_trade_margin_weight": 1.25,
    "missed_trade_margin_weight": 0.35,
    "no_trade_margin_dollars": 25.0,
    "positive_trade_margin_dollars": 10.0,
    "logit_temperature": 0.75,
}
ACTION_LOSS_MODES = ("huber", "decision_aware")


@dataclass
class ActionDecision:
    """One minute-level action example."""

    session: str
    decision_time: datetime
    features: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    market_last: np.ndarray


class ActionMLP(nn.Module):
    """Small MLP that predicts Q-values for no-trade/call/put actions."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, len(ACTION_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _nearest_action_index(row: dict, *, right: str, policy_index: int) -> tuple[int, int] | None:
    mask = np.asarray(row["candidate_mask"], dtype=bool)
    labels = np.asarray(row["labels_net_pnl"], dtype=np.float32)[:, :, policy_index]
    rights = tuple(row["rights"])
    if right not in rights:
        return None
    right_idx = rights.index(right)
    offsets = np.asarray(row["strike_offsets"], dtype=np.float32)
    valid = mask[:, right_idx] & np.isfinite(labels[:, right_idx])
    if not valid.any():
        return None
    strike_candidates = np.where(valid)[0]
    strike_idx = int(strike_candidates[np.argmin(np.abs(offsets[strike_candidates]))])
    return strike_idx, right_idx


def action_feature_vector(row: dict, call_idx: tuple[int, int], put_idx: tuple[int, int]) -> np.ndarray:
    """Build one decision-level feature vector from ATM call/put candidates."""
    call_features = candidate_feature_vector(row, call_idx[0], call_idx[1])
    put_features = candidate_feature_vector(row, put_idx[0], put_idx[1])
    spread = call_features - put_features
    product = call_features * put_features
    return np.concatenate([call_features, put_features, spread, product]).astype(np.float32)


def action_decision_from_row(
    *,
    session: str,
    row: dict,
    policy_index: int,
) -> ActionDecision | None:
    """Convert a neural row into a no-trade/call/put action row."""
    call_idx = _nearest_action_index(row, right="C", policy_index=policy_index)
    put_idx = _nearest_action_index(row, right="P", policy_index=policy_index)
    if call_idx is None or put_idx is None:
        return None
    labels = np.asarray(row["labels_net_pnl"], dtype=np.float32)
    market_window = np.asarray(row["market_window"], dtype=np.float32)
    offsets = np.asarray(row["strike_offsets"], dtype=np.float32)
    return ActionDecision(
        session=session,
        decision_time=row["decision_time"],
        features=action_feature_vector(row, call_idx, put_idx),
        labels=np.asarray(
            [
                0.0,
                float(labels[call_idx[0], call_idx[1], policy_index]),
                float(labels[put_idx[0], put_idx[1], policy_index]),
            ],
            dtype=np.float32,
        ),
        offsets=np.asarray([0.0, offsets[call_idx[0]], offsets[put_idx[0]]], dtype=np.float32),
        market_last=market_window[-1].astype(np.float32),
    )


def load_action_decisions(paths: Sequence[Path], *, policy_index: int) -> list[ActionDecision]:
    decisions: list[ActionDecision] = []
    for path in sorted(paths):
        session = session_from_path(path)
        with path.open("rb") as f:
            rows = pickle.load(f)
        for row in rows:
            decision = action_decision_from_row(
                session=session,
                row=row,
                policy_index=policy_index,
            )
            if decision is not None:
                decisions.append(decision)
    return decisions


def collect_action_examples(decisions: Sequence[ActionDecision]) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack([d.features for d in decisions]).astype(np.float32)
    labels = np.vstack([d.labels for d in decisions]).astype(np.float32)
    return features, labels


def _target(labels: np.ndarray, config: PilotConfig) -> np.ndarray:
    clipped = np.clip(labels, -config.target_clip, config.target_clip)
    return (clipped / config.target_scale).astype(np.float32)


def _mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.sum() * 0.0


def decision_aware_action_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PnL regression plus direct penalties for bad trade/no-trade decisions.

    The regression term still teaches executable dollar outcomes. The additional
    terms make the model care about the actual action boundary:

    * cross entropy for the best action among no-trade/call/put
    * larger margin penalty when no-trade is best but a trade scores higher
    * smaller margin penalty when a profitable trade is best but no-trade wins
    """
    regression = F.huber_loss(predictions, targets, delta=1.0)
    best_action = torch.argmax(targets, dim=1)
    temperature = ACTION_DECISION_LOSS_CONFIG["logit_temperature"]
    decision_ce = F.cross_entropy(predictions / temperature, best_action)

    no_trade_score = predictions[:, 0]
    trade_scores = predictions[:, 1:]
    best_trade_score = trade_scores.max(dim=1).values
    best_action_score = predictions.gather(1, best_action.unsqueeze(1)).squeeze(1)
    no_trade_is_best = best_action == 0
    trade_is_best = best_action != 0
    no_trade_margin = ACTION_DECISION_LOSS_CONFIG["no_trade_margin_dollars"] / target_scale
    positive_trade_margin = (
        ACTION_DECISION_LOSS_CONFIG["positive_trade_margin_dollars"] / target_scale
    )
    false_trade_margin = _mean_or_zero(
        F.relu(best_trade_score - no_trade_score + no_trade_margin),
        no_trade_is_best,
    )
    missed_trade_margin = _mean_or_zero(
        F.relu(no_trade_score - best_action_score + positive_trade_margin),
        trade_is_best,
    )

    total = (
        regression
        + ACTION_DECISION_LOSS_CONFIG["decision_ce_weight"] * decision_ce
        + ACTION_DECISION_LOSS_CONFIG["false_trade_margin_weight"] * false_trade_margin
        + ACTION_DECISION_LOSS_CONFIG["missed_trade_margin_weight"] * missed_trade_margin
    )
    return total, {
        "total": total,
        "huber": regression,
        "decision_ce": decision_ce,
        "false_trade_margin": false_trade_margin,
        "missed_trade_margin": missed_trade_margin,
    }


def action_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_scale: float,
    loss_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the configured action-model loss."""
    if loss_mode == "huber":
        huber = F.huber_loss(predictions, targets, delta=1.0)
        zero = huber * 0.0
        return huber, {
            "total": huber,
            "huber": huber,
            "decision_ce": zero,
            "false_trade_margin": zero,
            "missed_trade_margin": zero,
        }
    if loss_mode == "decision_aware":
        return decision_aware_action_loss(
            predictions,
            targets,
            target_scale=target_scale,
        )
    raise ValueError(f"unknown action loss mode: {loss_mode}")


def action_feature_version(loss_mode: str) -> str:
    if loss_mode == "huber":
        return FEATURE_VERSION
    if loss_mode == "decision_aware":
        return DECISION_AWARE_FEATURE_VERSION
    raise ValueError(f"unknown action loss mode: {loss_mode}")


def train_action_model(
    train_decisions: Sequence[ActionDecision],
    validation_decisions: Sequence[ActionDecision],
    *,
    config: PilotConfig,
    loss_mode: str = "huber",
) -> tuple[ActionMLP, FeatureScaler, list[dict]]:
    if loss_mode not in ACTION_LOSS_MODES:
        raise ValueError(f"unknown action loss mode: {loss_mode}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    x_train_raw, y_train_raw = collect_action_examples(train_decisions)
    x_val_raw, y_val_raw = collect_action_examples(validation_decisions)
    scaler = FeatureScaler.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    y_train = _target(y_train_raw, config)
    y_val = _target(y_val_raw, config)

    model = ActionMLP(input_dim=x_train.shape[1], hidden_dim=max(config.hidden_dim, 128))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(config.batch_size, len(x_train)),
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    history: list[dict] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        parts_by_name: dict[str, list[float]] = {
            "total": [],
            "huber": [],
            "decision_ce": [],
            "false_trade_margin": [],
            "missed_trade_margin": [],
        }
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss, parts = action_loss(
                pred,
                batch_y,
                target_scale=config.target_scale,
                loss_mode=loss_mode,
            )
            loss.backward()
            optimizer.step()
            for name, value in parts.items():
                parts_by_name[name].append(float(value.detach().cpu()))
        model.eval()
        with torch.no_grad():
            _, val_parts_t = action_loss(
                model(x_val_t),
                y_val_t,
                target_scale=config.target_scale,
                loss_mode=loss_mode,
            )
        val_parts = {name: float(value.detach().cpu()) for name, value in val_parts_t.items()}
        val_loss = val_parts["total"]
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "loss_mode": loss_mode,
                "train_loss": float(np.mean(parts_by_name["total"])) if parts_by_name["total"] else math.nan,
                "train_huber": float(np.mean(parts_by_name["huber"])) if parts_by_name["huber"] else math.nan,
                "train_decision_ce": float(np.mean(parts_by_name["decision_ce"])) if parts_by_name["decision_ce"] else math.nan,
                "train_false_trade_margin": float(np.mean(parts_by_name["false_trade_margin"])) if parts_by_name["false_trade_margin"] else math.nan,
                "train_missed_trade_margin": float(np.mean(parts_by_name["missed_trade_margin"])) if parts_by_name["missed_trade_margin"] else math.nan,
                "validation_loss": val_loss,
                "validation_huber": val_parts["huber"],
                "validation_decision_ce": val_parts["decision_ce"],
                "validation_false_trade_margin": val_parts["false_trade_margin"],
                "validation_missed_trade_margin": val_parts["missed_trade_margin"],
                "is_best": is_best,
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def predict_actions(
    model: ActionMLP,
    scaler: FeatureScaler,
    decisions: Sequence[ActionDecision],
    *,
    target_scale: float,
) -> np.ndarray:
    if not decisions:
        return np.empty((0, len(ACTION_NAMES)), dtype=np.float32)
    x_raw = np.vstack([d.features for d in decisions]).astype(np.float32)
    x = scaler.transform(x_raw)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).cpu().numpy().astype(np.float32)
    return pred * target_scale


def simulate_action_policy(
    decisions: Sequence[ActionDecision],
    predictions: np.ndarray,
    *,
    threshold: float,
    cooldown_minutes: int,
    strategy: str,
) -> list[Trade]:
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    for decision, pred in zip(decisions, predictions):
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        # Use only call/put scores; no-trade is governed by the threshold.
        action = int(np.argmax(pred[1:]) + 1)
        score = float(pred[action])
        if score < threshold:
            continue
        pnl = float(decision.labels[action])
        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=pnl,
                score=score,
                right="C" if action == 1 else "P",
                offset=float(decision.offsets[action]),
                strategy=strategy,
            )
        )
        next_time_by_session[decision.session] = decision.decision_time + timedelta(
            minutes=cooldown_minutes
        )
    return trades


def simulate_action_baseline(
    decisions: Sequence[ActionDecision],
    *,
    kind: str,
    cooldown_minutes: int,
    seed: int = 42,
) -> list[Trade]:
    rng = np.random.default_rng(seed)
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    for decision in decisions:
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        if kind == "atm_call":
            action = 1
        elif kind == "atm_put":
            action = 2
        elif kind == "random_atm_side":
            action = int(rng.integers(1, 3))
        elif kind == "vwap_omar":
            spx_close = float(decision.market_last[0])
            spx_vwap = float(decision.market_last[2])
            omar = float(decision.market_last[3])
            if not all(np.isfinite([spx_close, spx_vwap, omar])):
                continue
            if spx_close > spx_vwap and omar > 0:
                action = 1
            elif spx_close < spx_vwap and omar < 0:
                action = 2
            else:
                continue
        else:
            raise ValueError(f"unknown baseline kind: {kind}")
        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=float(decision.labels[action]),
                score=None,
                right="C" if action == 1 else "P",
                offset=float(decision.offsets[action]),
                strategy=kind,
            )
        )
        next_time_by_session[decision.session] = decision.decision_time + timedelta(
            minutes=cooldown_minutes
        )
    return trades


def choose_action_threshold(
    decisions: Sequence[ActionDecision],
    predictions: np.ndarray,
    *,
    config: PilotConfig,
) -> tuple[float, list[dict]]:
    if len(predictions) == 0:
        return float("inf"), []
    top = np.max(predictions[:, 1:], axis=1)
    thresholds = sorted(set(np.quantile(top, np.linspace(0.0, 0.98, 50)).round(4).tolist() + [0.0]))
    sweep = []
    for threshold in thresholds:
        trades = simulate_action_policy(
            decisions,
            predictions,
            threshold=float(threshold),
            cooldown_minutes=config.cooldown_minutes,
            strategy="action_neural",
        )
        sweep.append({"threshold": float(threshold), **metrics_for_trades(trades)})
    eligible = [x for x in sweep if x["trades"] >= config.min_validation_trades]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda x: (x["total_pnl"], x["profit_factor"], x["trades"]))
    return float(best["threshold"]), sweep


def summarize_random_action_baseline(
    decisions: Sequence[ActionDecision],
    *,
    config: PilotConfig,
) -> dict:
    metrics = []
    for seed in range(config.seed, config.seed + config.random_seeds):
        trades = simulate_action_baseline(
            decisions,
            kind="random_atm_side",
            cooldown_minutes=config.cooldown_minutes,
            seed=seed,
        )
        metrics.append(metrics_for_trades(trades))
    totals = np.asarray([m["total_pnl"] for m in metrics], dtype=float)
    trades = np.asarray([m["trades"] for m in metrics], dtype=float)
    return {
        "runs": config.random_seeds,
        "trades_mean": float(trades.mean()) if len(trades) else 0.0,
        "total_pnl_mean": float(totals.mean()) if len(totals) else 0.0,
        "total_pnl_std": float(totals.std()) if len(totals) else 0.0,
        "best_total_pnl": float(totals.max()) if len(totals) else 0.0,
        "worst_total_pnl": float(totals.min()) if len(totals) else 0.0,
    }


def action_model_state_dict(
    *,
    model: ActionMLP,
    scaler: FeatureScaler,
    config: PilotConfig,
    history: Sequence[dict],
    input_dim: int,
    loss_mode: str = "huber",
) -> dict:
    return {
        "feature_version": action_feature_version(loss_mode),
        "loss_mode": loss_mode,
        "loss_config": ACTION_DECISION_LOSS_CONFIG if loss_mode == "decision_aware" else None,
        "action_names": ACTION_NAMES,
        "input_dim": input_dim,
        "config": asdict(config),
        "scaler": scaler.to_dict(),
        "history": list(history),
        "model_state": model.state_dict(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
