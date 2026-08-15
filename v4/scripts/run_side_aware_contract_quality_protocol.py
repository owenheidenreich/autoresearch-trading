"""Run Protocol 022: side-aware contract-quality calibration.

This protocol keeps the frozen A+ timing/value surface model from Protocol 018
and adds one learned gate over its proposed trades. The gate is trained only on
pre-holdout data and asks the next practical question:

    this setup may be good, but is this call/put contract too expensive for it?

No paid data is downloaded. Q1 2025 is audit-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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
    SurfaceDecision,
    SurfaceVariant,
    predict_surface_actions,
    registered_aplus_surface_variants,
    registered_protocol_trials,
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


LOOP_ID = "v4_aplus_hypothesis_022_side_aware_contract_quality"
_NY = ZoneInfo("America/New_York")
_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
_FEATURES_OF_INTEREST = (
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
    "pattern_count_norm",
    "pattern_side_gap_atr",
    "pattern_side_sigma",
    "pattern_side_move1_atr",
    "pattern_side_move5_atr",
    "pattern_side_move15_atr",
)


@dataclass
class Proposal:
    decision: SurfaceDecision
    split: str
    seed: int
    effective_seed: int
    token_idx: int
    edge: float
    flat_score: float
    action_score: float
    pnl: float
    features: np.ndarray


class QualityMLP(nn.Module):
    """Small side-aware proposal gate; intentionally lower-capacity."""

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class Calibrator:
    model: QualityMLP
    scaler: FeatureScaler
    history: list[dict]

    def predict_proba(self, features: np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        x = self.scaler.transform(features)
        out = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                logits = self.model(torch.from_numpy(x[start : start + batch_size]))
                out.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
        return np.concatenate(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q1-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025"))
    parser.add_argument("--q2-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025"))
    parser.add_argument("--q3-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--seed-q4-data-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_022_side_aware_contract_quality"),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--variant-name", default="surface_structure_aplus_teacher_margin")
    parser.add_argument("--trial-name", default="post_open_late_edge25_max2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--calibrator-epochs", type=int, default=14)
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--loop-id", default=LOOP_ID)
    parser.add_argument("--thresholds", nargs="*", type=float, default=list(_THRESHOLDS))
    parser.add_argument(
        "--min-selection-trade-retention",
        type=float,
        default=0.0,
        help="minimum fraction of baseline selection trades a threshold must preserve",
    )
    return parser.parse_args()


def _paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("*.pkl"))
    if not paths:
        raise SystemExit(f"no pkl files found under {data_dir}")
    return paths


def _find_variant(name: str) -> SurfaceVariant:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise SystemExit(f"unknown A+ variant: {name}")


def _find_trial(name: str) -> ProtocolTrial:
    for trial in registered_protocol_trials():
        if trial.name == name:
            return trial
    raise SystemExit(f"unknown protocol trial: {name}")


def _finite(value: float, default: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else default


def _token_map(feature_names: Sequence[str], token: np.ndarray) -> dict[str, float]:
    return {name: _finite(float(value), 0.0) for name, value in zip(feature_names, token)}


def _bucket_features(bucket: str) -> list[float]:
    return [
        float(bucket == "post_open_morning"),
        float(bucket == "late_afternoon"),
        float(bucket == "midday"),
        float(bucket == "first_30"),
    ]


def _proposal_features(
    decision: SurfaceDecision,
    token_idx: int,
    *,
    pred: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    token = np.asarray(decision.token_features[token_idx], dtype=np.float32)
    values = _token_map(feature_names, token)
    is_put = float(str(decision.rights[token_idx]) == "P")
    is_call = float(str(decision.rights[token_idx]) == "C")
    bucket = time_bucket(decision.decision_time)
    flat_score = float(pred[0])
    action_score = float(pred[token_idx + 1])
    edge = action_score - flat_score
    selected = [
        edge / 100.0,
        action_score / 100.0,
        flat_score / 100.0,
        float(decision.offsets[token_idx]) / 50.0,
        abs(float(decision.offsets[token_idx])) / 50.0,
        is_call,
        is_put,
        *_bucket_features(bucket),
    ]
    for name in _FEATURES_OF_INTEREST:
        selected.append(values.get(name, 0.0))
    interaction_names = (
        "theta_burden_hold",
        "spread_tax",
        "breakeven_atr",
        "gamma_theta_ratio_scaled",
        "contract_value_score",
        "pattern_count_norm",
        "pattern_side_sigma",
        "pattern_side_move5_atr",
        "convexity_per_premium",
    )
    for name in interaction_names:
        selected.append(is_put * values.get(name, 0.0))
    selected.append(is_put * max(0.0, 2.0 - values.get("breakeven_atr", 0.0)))
    selected.append(is_put * max(0.0, 0.006 - values.get("gamma_theta_ratio_scaled", 0.0)) * 100.0)
    selected.append(is_put * max(0.0, values.get("theta_burden_hold", 0.0) - 0.06) * 10.0)
    selected.append(is_put * max(0.0, values.get("spread_tax", 0.0) - 0.012) * 100.0)
    return np.nan_to_num(np.asarray(selected, dtype=np.float32), nan=0.0, posinf=8.0, neginf=-8.0)


def _feature_names() -> list[str]:
    names = [
        "edge_scaled",
        "action_score_scaled",
        "flat_score_scaled",
        "offset_norm",
        "abs_offset_norm",
        "is_call",
        "is_put",
        "bucket_post_open_morning",
        "bucket_late_afternoon",
        "bucket_midday",
        "bucket_first_30",
        *_FEATURES_OF_INTEREST,
    ]
    for name in (
        "theta_burden_hold",
        "spread_tax",
        "breakeven_atr",
        "gamma_theta_ratio_scaled",
        "contract_value_score",
        "pattern_count_norm",
        "pattern_side_sigma",
        "pattern_side_move5_atr",
        "convexity_per_premium",
    ):
        names.append(f"put_x_{name}")
    names.extend(
        [
            "put_x_low_breakeven_room",
            "put_x_weak_gamma_theta",
            "put_x_theta_over_006",
            "put_x_spread_over_0012",
        ]
    )
    return names


def _collect_proposals(
    decisions: Sequence[SurfaceDecision],
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    split: str,
    seed: int,
    effective_seed: int,
    feature_names: Sequence[str],
) -> list[Proposal]:
    proposals = []
    allowed = set(trial.allowed_buckets)
    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in allowed:
            continue
        action_mask = np.concatenate([[True], decision.token_mask])
        masked = np.asarray(pred, dtype=float).copy()
        masked[~action_mask] = -np.inf
        if not np.isfinite(masked).any():
            continue
        action = int(np.nanargmax(masked))
        if action == 0:
            continue
        token_idx = action - 1
        edge = float(masked[action] - masked[0])
        if not np.isfinite(edge) or edge < trial.min_edge_vs_no_trade:
            continue
        pnl = float(decision.labels[token_idx])
        if not np.isfinite(pnl):
            continue
        proposals.append(
            Proposal(
                decision=decision,
                split=split,
                seed=seed,
                effective_seed=effective_seed,
                token_idx=token_idx,
                edge=edge,
                flat_score=float(masked[0]),
                action_score=float(masked[action]),
                pnl=pnl,
                features=_proposal_features(
                    decision,
                    token_idx,
                    pred=masked,
                    feature_names=feature_names,
                ),
            )
        )
    return proposals


def _quality_weight(proposals: Sequence[Proposal], feature_names: Sequence[str]) -> np.ndarray:
    weights = []
    for proposal in proposals:
        values = _token_map(feature_names, proposal.decision.token_features[proposal.token_idx])
        is_put = str(proposal.decision.rights[proposal.token_idx]) == "P"
        bad_contract = bool(
            values.get("worth_spread_flag", 0.0) < 0.5
            or values.get("obvious_overpay_flag", 0.0) > 0.5
            or values.get("breakeven_atr", 0.0) > 2.25
            or values.get("spread_tax", 0.0) > 0.014
            or values.get("theta_burden_hold", 0.0) > 0.14
            or values.get("gamma_theta_ratio_scaled", 0.0) < 0.003
        )
        weight = 1.0
        if proposal.pnl < 0.0:
            weight += 0.50
        if is_put:
            weight += 0.35
        if bad_contract:
            weight += 0.45
        if is_put and bad_contract:
            weight += 0.80
        weights.append(weight)
    return np.asarray(weights, dtype=np.float32)


def _train_calibrator(
    train_proposals: Sequence[Proposal],
    validation_proposals: Sequence[Proposal],
    *,
    seed: int,
    feature_names: Sequence[str],
    token_feature_names_: Sequence[str],
    epochs: int,
    batch_size: int,
) -> Calibrator:
    if not train_proposals:
        raise ValueError("no train proposals for calibrator")
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train = np.vstack([proposal.features for proposal in train_proposals]).astype(np.float32)
    y_train = np.asarray([proposal.pnl >= 50.0 for proposal in train_proposals], dtype=np.float32)
    w_train = _quality_weight(train_proposals, token_feature_names_)
    x_val = np.vstack([proposal.features for proposal in validation_proposals]).astype(np.float32) if validation_proposals else x_train
    y_val = (
        np.asarray([proposal.pnl >= 50.0 for proposal in validation_proposals], dtype=np.float32)
        if validation_proposals
        else y_train
    )
    w_val = _quality_weight(validation_proposals, token_feature_names_) if validation_proposals else w_train
    scaler = FeatureScaler.fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    model = QualityMLP(input_dim=len(feature_names), hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_s),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
        ),
        batch_size=min(batch_size, len(x_train_s)),
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val_s)
    val_y = torch.from_numpy(y_val)
    val_w = torch.from_numpy(w_val)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x_b, y_b, w_b in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_b)
            raw_loss = F.binary_cross_entropy_with_logits(logits, y_b, reduction="none")
            loss = (raw_loss * w_b).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = (F.binary_cross_entropy_with_logits(val_logits, val_y, reduction="none") * val_w).mean()
        is_best = float(val_loss.detach().cpu()) < best_val
        if is_best:
            best_val = float(val_loss.detach().cpu())
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": float(val_loss.detach().cpu()),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
                "is_best": is_best,
            }
        )
    model.load_state_dict(best_state)
    return Calibrator(model=model, scaler=scaler, history=history)


def _trade_from_proposal(
    proposal: Proposal,
    *,
    quality_score: float,
    strategy: str,
) -> Trade:
    decision = proposal.decision
    return Trade(
        session=decision.session,
        decision_time=decision.decision_time.isoformat(),
        pnl=float(proposal.pnl),
        score=float(quality_score),
        right=str(decision.rights[proposal.token_idx]),
        offset=float(decision.offsets[proposal.token_idx]),
        strategy=strategy,
    )


def _simulate_calibrated_policy(
    proposals: Sequence[Proposal],
    quality_scores: np.ndarray,
    *,
    threshold: float,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    strategy: str,
) -> list[Trade]:
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    for proposal, quality_score in zip(proposals, quality_scores):
        session = proposal.decision.session
        if session in halted_sessions:
            continue
        if trades_by_session.get(session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(session)
        if next_time is not None and proposal.decision.decision_time < next_time:
            continue
        if float(quality_score) < threshold:
            continue
        trades.append(_trade_from_proposal(proposal, quality_score=float(quality_score), strategy=strategy))
        trades_by_session[session] = trades_by_session.get(session, 0) + 1
        pnl_by_session[session] = pnl_by_session.get(session, 0.0) + float(proposal.pnl)
        next_time_by_session[session] = proposal.decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[session] <= trial.daily_loss_stop:
            halted_sessions.add(session)
    return trades


def _selection_reward(metrics: dict) -> float:
    trades = float(metrics["trades"])
    if trades < 12:
        return -1_000_000.0 + trades
    pf = float(metrics["profit_factor"])
    if not math.isfinite(pf):
        pf = 5.0
    return (
        float(metrics["total_pnl"])
        + 1_000.0 * (min(pf, 5.0) - 1.0)
        + 1_000.0 * (float(metrics["positive_day_fraction"]) - 0.50)
        + 0.15 * float(metrics["max_drawdown"])
        - 1_500.0 * max(0.0, float(metrics.get("top_day_profit_share", 1.0)) - 0.45)
    )


def _select_threshold(
    proposals: Sequence[Proposal],
    scores: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    seed: int,
    thresholds: Sequence[float] = _THRESHOLDS,
    min_trade_retention: float = 0.0,
    loop_id: str = LOOP_ID,
) -> tuple[float, list[dict]]:
    rows = []
    baseline_trades = _simulate_calibrated_policy(
        proposals,
        scores,
        threshold=-1.0,
        trial=trial,
        cooldown_minutes=cooldown_minutes,
        strategy=f"{loop_id}:threshold_baseline",
    )
    min_retained_trades = int(math.ceil(max(0.0, min_trade_retention) * len(baseline_trades)))
    for threshold in thresholds:
        trades = _simulate_calibrated_policy(
            proposals,
            scores,
            threshold=threshold,
            trial=trial,
            cooldown_minutes=cooldown_minutes,
            strategy=f"{loop_id}:threshold_{threshold:.2f}",
        )
        metrics = metrics_with_concentration(trades)
        stress50 = metrics_with_concentration(stress_trades(trades, extra_cost_per_trade=50.0))
        row = {
            "threshold": threshold,
            "baseline_trade_count": len(baseline_trades),
            "min_retained_trades": min_retained_trades,
            "metrics": metrics,
            "stress50": stress50,
            "reward": _selection_reward(stress50),
            "passes_floor": bool(
                metrics["trades"] >= max(12, min_retained_trades)
                and metrics["total_pnl"] > 0
                and metrics["profit_factor"] >= 1.05
                and stress50["total_pnl"] > 0
                and float(metrics.get("top_day_profit_share", 1.0)) <= 0.75
            ),
        }
        rows.append(row)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["passes_floor"],
            row["reward"],
            row["metrics"]["total_pnl"],
            -abs(row["threshold"] - 0.70),
        ),
        reverse=True,
    )
    return float(ranked[0]["threshold"]), rows


def _summarize_seed_rows(rows: Sequence[dict], split: str, metric_key: str) -> dict:
    metrics = [row[metric_key] for row in rows if row["split"] == split]
    if not metrics:
        return {
            "split": split,
            "pnl_median": 0.0,
            "pf_median": 0.0,
            "trades_median": 0.0,
            "positive_seed_fraction": 0.0,
            "top_day_share_median": 1.0,
        }
    return {
        "split": split,
        "pnl_median": float(np.median([m["total_pnl"] for m in metrics])),
        "pf_median": float(np.median([m["profit_factor"] for m in metrics])),
        "trades_median": float(np.median([m["trades"] for m in metrics])),
        "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in metrics])),
        "top_day_share_median": float(np.median([m["top_day_profit_share"] for m in metrics])),
    }


def _trade_lens(trades: Sequence[Trade], split: str, seed: int) -> list[dict]:
    rows = []
    for lens, predicate in (
        ("post_open_put", lambda t: time_bucket(datetime.fromisoformat(t.decision_time)) == "post_open_morning" and t.right == "P"),
        ("post_open_call", lambda t: time_bucket(datetime.fromisoformat(t.decision_time)) == "post_open_morning" and t.right == "C"),
        ("late_afternoon_all", lambda t: time_bucket(datetime.fromisoformat(t.decision_time)) == "late_afternoon"),
    ):
        group = [t for t in trades if predicate(t)]
        metrics = metrics_with_concentration(group)
        rows.append(
            {
                "split": split,
                "seed": seed,
                "lens": lens,
                "trades": metrics["trades"],
                "total_pnl": metrics["total_pnl"],
                "profit_factor": metrics["profit_factor"],
                "win_rate": float(np.mean([t.pnl > 0.0 for t in group])) if group else 0.0,
            }
        )
    return rows


def _aggregate_lens_rows(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["split"], row["lens"]), []).append(row)
    out = []
    for (split, lens), group in sorted(groups.items()):
        out.append(
            {
                "split": split,
                "lens": lens,
                "seed_pnl_median": float(np.median([row["total_pnl"] for row in group])),
                "seed_pf_median": float(np.median([row["profit_factor"] for row in group])),
                "seed_trades_median": float(np.median([row["trades"] for row in group])),
                "win_rate_median": float(np.median([row["win_rate"] for row in group])),
                "all_seed_pnl": float(np.sum([row["total_pnl"] for row in group])),
                "all_seed_trades": int(np.sum([row["trades"] for row in group])),
            }
        )
    return out


def _proposal_dump(
    proposals: Sequence[Proposal],
    scores: np.ndarray,
    *,
    threshold: float,
    token_feature_names_: Sequence[str],
) -> list[dict]:
    rows = []
    for proposal, score in zip(proposals, scores):
        decision = proposal.decision
        token_values = _token_map(token_feature_names_, decision.token_features[proposal.token_idx])
        row = {
            "seed": int(proposal.seed),
            "effective_seed": int(proposal.effective_seed),
            "split": proposal.split,
            "session": decision.session,
            "decision_time": decision.decision_time.isoformat(),
            "bucket": time_bucket(decision.decision_time),
            "contract_id": str(decision.contract_ids[proposal.token_idx]),
            "right": str(decision.rights[proposal.token_idx]),
            "offset": float(decision.offsets[proposal.token_idx]),
            "pnl": float(proposal.pnl),
            "edge": float(proposal.edge),
            "quality_score": float(score),
            "threshold": float(threshold),
            "passes_quality_threshold": bool(float(score) >= threshold),
        }
        for feature in _FEATURES_OF_INTEREST:
            row[f"feature_{feature}"] = token_values.get(feature, 0.0)
        rows.append(row)
    return rows


def _sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "inf" if value > 0 else "-inf"
    return value


def _fmt(value: object, digits: int = 0) -> str:
    if isinstance(value, str):
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.{digits}f}"


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 022: Side-Aware Contract-Quality Calibration",
        "",
        payload["framing"],
        "",
        "## Pre-Registration",
        "",
    ]
    for item in payload["pre_registration"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Split Summary",
        "",
        "| Split | PnL | PF | Trades | Positive Seeds | +50 PnL | +100 PnL | Random PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]["summary"]:
        lines.append(
            f"| {row['split']} | {_fmt(row['pnl_median'])} | {_fmt(row['pf_median'], 3)} | "
            f"{_fmt(row['trades_median'])} | {_fmt(row['positive_seed_fraction'], 2)} | "
            f"{_fmt(row['stress50_pnl_median'])} | {_fmt(row['stress100_pnl_median'])} | "
            f"{_fmt(row['random_pnl_median'])} |"
        )
    lines += [
        "",
        "## Baseline Comparison",
        "",
        "| Split | Baseline PnL | Protocol 022 PnL | Delta | Baseline Trades | Protocol 022 Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]["baseline_comparison"]:
        lines.append(
            f"| {row['split']} | {_fmt(row['baseline_pnl_median'])} | {_fmt(row['candidate_pnl_median'])} | "
            f"{_fmt(row['pnl_delta'])} | {_fmt(row['baseline_trades_median'])} | {_fmt(row['candidate_trades_median'])} |"
        )
    lines += [
        "",
        "## Q1 / Side Lens",
        "",
        "| Split | Lens | Seed Median Trades | Seed Median PnL | Seed Median PF | Win Rate | All-Seed PnL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]["lens_summary"]:
        lines.append(
            f"| {row['split']} | {row['lens']} | {_fmt(row['seed_trades_median'])} | "
            f"{_fmt(row['seed_pnl_median'])} | {_fmt(row['seed_pf_median'], 3)} | "
            f"{_fmt(row['win_rate_median'], 2)} | {_fmt(row['all_seed_pnl'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        payload["decision"]["text"],
        "",
        "Next gate:",
        "",
        payload["decision"]["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    policy_name, cooldown = POLICY_META[args.policy_index]
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    train_paths = _paths_by_split(args.data_dir)
    seed_q4_dir = args.seed_q4_data_dir or args.q4_data_dir
    seed_q4_paths = _paths(seed_q4_dir)
    window = _protocol_window(train_paths, seed_q4_paths)
    market_cache = MarketStructureCache()
    token_names = token_feature_names(variant.token_mode)
    proposal_feature_names = _feature_names()

    train_decisions = _load_surface_decisions_cached(
        train_paths["train"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="train",
        cache_dir=decision_cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        train_paths["validation"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="validation",
        cache_dir=decision_cache_dir,
    )
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    decision_sets = {
        "selection": selection_decisions,
        "march_2026": _load_surface_decisions_cached(
            train_paths["test"],
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="march",
            cache_dir=decision_cache_dir,
        ),
        "q1_2025": _load_surface_decisions_cached(
            _paths(args.q1_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q1_2025",
            cache_dir=decision_cache_dir,
        ),
        "q2_2025": _load_surface_decisions_cached(
            _paths(args.q2_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q2_2025",
            cache_dir=decision_cache_dir,
        ),
        "q3_2025": _load_surface_decisions_cached(
            _paths(args.q3_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q3_2025",
            cache_dir=decision_cache_dir,
        ),
        "q4_2025": _load_surface_decisions_cached(
            _paths(args.q4_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q4_2025",
            cache_dir=decision_cache_dir,
        ),
    }

    seed_rows = []
    all_lens_rows = []
    selected_thresholds = []
    selected_dir = args.out_dir / "selected_trades"
    selected_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir = args.out_dir / "proposal_attribution"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        effective_seed = window_seed(seed, window.window_id)
        print(
            f"{args.loop_id} policy={args.policy_index} variant={variant.name} "
            f"trial={trial.name} seed={seed}",
            flush=True,
        )
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        surface_model, standardizer, surface_history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=variant,
        )

        prediction_sets = {
            "calibrator_train": predict_surface_actions(
                surface_model,
                standardizer,
                [*train_decisions, *calibration_decisions],
                target_scale=config.target_scale,
            ),
            "selection": predict_surface_actions(surface_model, standardizer, selection_decisions, target_scale=config.target_scale),
        }
        train_proposals = _collect_proposals(
            [*train_decisions, *calibration_decisions],
            prediction_sets["calibrator_train"],
            trial=trial,
            split="calibrator_train",
            seed=seed,
            effective_seed=effective_seed,
            feature_names=token_names,
        )
        selection_proposals = _collect_proposals(
            selection_decisions,
            prediction_sets["selection"],
            trial=trial,
            split="selection",
            seed=seed,
            effective_seed=effective_seed,
            feature_names=token_names,
        )
        calibrator = _train_calibrator(
            train_proposals,
            selection_proposals,
            seed=effective_seed,
            feature_names=proposal_feature_names,
            token_feature_names_=token_names,
            epochs=args.calibrator_epochs,
            batch_size=args.batch_size,
        )
        selection_scores = calibrator.predict_proba(
            np.vstack([proposal.features for proposal in selection_proposals]).astype(np.float32)
            if selection_proposals
            else np.empty((0, len(proposal_feature_names)), dtype=np.float32)
        )
        threshold, threshold_rows = _select_threshold(
            selection_proposals,
            selection_scores,
            trial=trial,
            cooldown_minutes=cooldown,
            seed=effective_seed,
            thresholds=args.thresholds,
            min_trade_retention=args.min_selection_trade_retention,
            loop_id=args.loop_id,
        )
        selected_thresholds.append(
            {
                "seed": seed,
                "effective_seed": effective_seed,
                "threshold": threshold,
                "threshold_rows": threshold_rows,
                "surface_best_epoch": next((x["epoch"] for x in surface_history if x["is_best"]), None),
                "calibrator_best_epoch": next((x["epoch"] for x in calibrator.history if x["is_best"]), None),
                "calibrator_train_proposals": len(train_proposals),
                "selection_proposals": len(selection_proposals),
            }
        )

        for split, decisions in decision_sets.items():
            predictions = prediction_sets.get(split)
            if predictions is None:
                predictions = predict_surface_actions(
                    surface_model,
                    standardizer,
                    decisions,
                    target_scale=config.target_scale,
                )
            proposals = (
                selection_proposals
                if split == "selection"
                else _collect_proposals(
                    decisions,
                    predictions,
                    trial=trial,
                    split=split,
                    seed=seed,
                    effective_seed=effective_seed,
                    feature_names=token_names,
                )
            )
            scores = (
                selection_scores
                if split == "selection"
                else calibrator.predict_proba(
                    np.vstack([proposal.features for proposal in proposals]).astype(np.float32)
                    if proposals
                    else np.empty((0, len(proposal_feature_names)), dtype=np.float32)
                )
            )
            proposal_dump = _proposal_dump(
                proposals,
                scores,
                threshold=threshold,
                token_feature_names_=token_names,
            )
            (proposal_dir / f"seed{seed}_{split}.json").write_text(
                json.dumps(_sanitize(proposal_dump), indent=2, allow_nan=False) + "\n"
            )
            trades = _simulate_calibrated_policy(
                proposals,
                scores,
                threshold=threshold,
                trial=trial,
                cooldown_minutes=cooldown,
                strategy=f"{args.loop_id}:threshold_{threshold:.2f}",
            )
            metrics = metrics_with_concentration(trades)
            stress50 = metrics_with_concentration(stress_trades(trades, extra_cost_per_trade=50.0))
            stress100 = metrics_with_concentration(stress_trades(trades, extra_cost_per_trade=100.0))
            random_baseline = summarize_random_baseline(
                decisions,
                trial=trial,
                cooldown_minutes=cooldown,
                seed=effective_seed,
                target_trade_count=int(metrics["trades"]),
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "effective_seed": effective_seed,
                    "split": split,
                    "threshold": threshold,
                    "proposal_count": len(proposals),
                    "metrics": metrics,
                    "stress50": stress50,
                    "stress100": stress100,
                    "random_baseline": random_baseline,
                }
            )
            all_lens_rows.extend(_trade_lens(trades, split, seed))
            trade_dump = [
                {
                    "seed": seed,
                    "split": split,
                    "session": trade.session,
                    "decision_time": trade.decision_time,
                    "right": trade.right,
                    "offset": trade.offset,
                    "pnl": trade.pnl,
                    "quality_score": trade.score,
                    "threshold": threshold,
                }
                for trade in trades
            ]
            (selected_dir / f"seed{seed}_{split}.json").write_text(
                json.dumps(_sanitize(trade_dump), indent=2, allow_nan=False) + "\n"
            )

    splits = ["selection", "march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025"]
    summary = []
    for split in splits:
        base = _summarize_seed_rows(seed_rows, split, "metrics")
        stress50 = _summarize_seed_rows(seed_rows, split, "stress50")
        stress100 = _summarize_seed_rows(seed_rows, split, "stress100")
        random_pnls = [
            row["random_baseline"]["total_pnl_median"]
            for row in seed_rows
            if row["split"] == split
        ]
        summary.append(
            {
                **base,
                "stress50_pnl_median": stress50["pnl_median"],
                "stress50_pf_median": stress50["pf_median"],
                "stress100_pnl_median": stress100["pnl_median"],
                "stress100_pf_median": stress100["pf_median"],
                "random_pnl_median": float(np.median(random_pnls)) if random_pnls else 0.0,
            }
        )

    baseline = {
        "march_2026": {"pnl": 3962.0, "trades": 39.0},
        "q1_2025": {"pnl": 2190.0, "trades": 117.0},
        "q2_2025": {"pnl": 13492.0, "trades": 110.0},
        "q3_2025": {"pnl": 11394.0, "trades": 128.0},
        "q4_2025": {"pnl": 20432.0, "trades": 126.0},
        "selection": {"pnl": 2380.0, "trades": 20.0},
    }
    baseline_comparison = []
    for row in summary:
        b = baseline[row["split"]]
        baseline_comparison.append(
            {
                "split": row["split"],
                "baseline_pnl_median": b["pnl"],
                "candidate_pnl_median": row["pnl_median"],
                "pnl_delta": row["pnl_median"] - b["pnl"],
                "baseline_trades_median": b["trades"],
                "candidate_trades_median": row["trades_median"],
            }
        )

    q1 = next(row for row in summary if row["split"] == "q1_2025")
    q2 = next(row for row in summary if row["split"] == "q2_2025")
    q3 = next(row for row in summary if row["split"] == "q3_2025")
    q4 = next(row for row in summary if row["split"] == "q4_2025")
    march = next(row for row in summary if row["split"] == "march_2026")
    improves_q1 = q1["pnl_median"] > baseline["q1_2025"]["pnl"] and q1["stress50_pnl_median"] > -3810.0
    survives_all = all(row["pnl_median"] > 0.0 and row["positive_seed_fraction"] >= 2 / 3 for row in (march, q1, q2, q3, q4))
    stress_improves = q1["stress50_pnl_median"] > -3810.0 and q2["stress50_pnl_median"] > 0 and q4["stress50_pnl_median"] > 0
    if improves_q1 and survives_all and stress_improves:
        decision_text = (
            "Keep Protocol 022 as a research improvement: it improves Q1 while preserving broad positive "
            "audit behavior and the learned side-aware contract-quality framing."
        )
        next_gate = (
            "Run a narrow 1s/tick path audit on Protocol 022 selected trades using existing 1s data first; "
            "request approval before any new paid audit slice."
        )
    else:
        decision_text = (
            "Reject Protocol 022 as the next champion. It is a valid diagnostic implementation, but the "
            "side-aware contract-quality gate did not improve Q1/stress without damaging another audit block."
        )
        next_gate = (
            "Keep the frozen Protocol 018 candidate and use Protocol 022 diagnostics to design a better "
            "in-network side/value objective rather than a proposal-level gate."
        )

    payload = {
        "loop_id": args.loop_id,
        "framing": (
            "Pre-registered no-paid-data model change: keep the A+ teacher-margin surface model, then train "
            "a small side-aware contract-quality neural gate on pre-holdout proposals only."
        ),
        "pre_registration": [
            "No paid market data download.",
            "No Q1 2025 labels in training or threshold selection.",
            "Base surface candidate remains surface_structure_aplus_teacher_margin / policy1 / post_open_late_edge25_max2.",
            "The learned gate sees the proposed contract, side, base-model edge, A+ timing/value features, and explicit put/value interactions.",
            "The gate trains on January 2026 plus early-February calibration proposals; threshold selection uses late-February only.",
            "March 2026 and Q1/Q2/Q3/Q4 2025 are audit-only.",
            "Keep only if Q1 improves without sacrificing broad positive audits and slippage-stress evidence.",
        ],
        "candidate": {
            "variant": asdict(variant) | {"variant_id": variant.variant_id},
            "policy_index": args.policy_index,
            "policy_name": policy_name,
            "trial": asdict(trial) | {"config_id": trial.config_id},
            "seeds": args.seeds,
            "threshold_grid": list(_THRESHOLDS),
            "thresholds": list(args.thresholds),
            "min_selection_trade_retention": float(args.min_selection_trade_retention),
            "window": asdict(window) | {"window_id": window.window_id},
        },
        "selected_thresholds": selected_thresholds,
        "feature_manifest": {
            "proposal_features": proposal_feature_names,
            "token_features": token_names,
        },
        "seed_rows": seed_rows,
        "aggregate": {
            "summary": summary,
            "baseline_comparison": baseline_comparison,
            "lens_summary": _aggregate_lens_rows(all_lens_rows),
            "lens_seed_rows": all_lens_rows,
        },
        "decision": {
            "keep": bool(improves_q1 and survives_all and stress_improves),
            "improves_q1": bool(improves_q1),
            "survives_all": bool(survives_all),
            "stress_improves": bool(stress_improves),
            "text": decision_text,
            "next_gate": next_gate,
        },
        "cost": "$0 incremental paid data",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = _sanitize(payload)
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_markdown(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.json")
    print(args.out_dir / "report.md")
    print(json.dumps(payload["decision"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
