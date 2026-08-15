"""Protocol 057: recovery-confirmed lifecycle exits on frozen Protocol 054.

This is one lifecycle-only hypothesis:

    If Protocol 054 wants to exit on model_exit_loss or model_exit_giveback,
    require a separate causal recovery model to agree that recovery is unlikely.

No entry-side knobs are changed. The Protocol 051 entry variant/trial remain
frozen, and the Protocol 054 lifecycle configs are fixed by split from the
prior 10-seed validation. No paid data is downloaded.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.dataset.spxw_0dte_neural import OPTION_FEATURE_NAMES
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceVariant,
    predict_surface_actions,
    stress_trades,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import FeatureScaler, PilotConfig, Trade
from v4.scripts.run_aplus_neural_protocol import _load_surface_decisions_cached
from v4.scripts.run_protocol052_sequential_lifecycle_walkforward import (
    DEFAULT_TRIAL,
    DEFAULT_VARIANT,
    NormalizedPathStore,
    _bounded_entries,
    _find_trial,
    _find_variant,
    _simulate_lifecycle_exit,
    _split_name,
    _summary_by_split,
    _trade_metrics,
)
from v4.scripts.run_sequential_risk_protocol import (
    _CONTRACT_MULTIPLIER,
    _FEATURE_INDEX,
    _POLICY,
    _RISK_TARGET_SCALE,
    RISK_FEATURE_NAMES,
    EntryProposal,
    PathPoint,
    RiskConfig,
    _causal_state_features,
    _constraint_allows_model_exit,
    _contract_path,
    _fit_risk_model,
    _load_session_rows,
    _metrics,
    _option_entry_features,
    _path_samples,
    _predict_headroom,
    _risk_grid,
    _select_entry_proposals,
    _trade_from_entry,
    _training_candidate_proposals,
)
from v4.scripts.run_soft_quality_walkforward_protocol import FoldSpec, _folds
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_057_lifecycle_recovery_confirmation_screen"
RECOVERY_TARGET_DELTA = 200.0
RECOVERY_PROBABILITY_THRESHOLD = 0.60
BASELINE_REGRET_TARGET_CAP = 600.0
PROTOCOL054_CONFIG_BY_SPLIT = {
    "q2_2025": "headroom_le_150_minhold_1_gb50_gbfrac50",
    "q3_2025": "headroom_le_150_minhold_1_gb50_gbfrac35",
    "q4_2025": "headroom_le_150_minhold_1_gb50_gbfrac35",
    "q1_2026": "headroom_le_150_minhold_1_gb50_gbfrac35",
}

RECOVERY_FEATURE_NAMES = tuple(RISK_FEATURE_NAMES) + (
    "mfe_age_frac",
    "post_mfe_velocity_norm",
    "pnl_velocity_3_norm",
    "pnl_accel_3_norm",
    "gamma_theta_change",
    "spread_change_frac",
    "bid_size_log",
    "ask_size_log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_official_context"))
    parser.add_argument("--q2-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_official_context"))
    parser.add_argument("--q3-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_official_context"))
    parser.add_argument("--q4-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_official_context"))
    parser.add_argument("--q1-2026-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2026_official_context"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"v4/audit/autoresearch/{LOOP_ID}"))
    parser.add_argument("--loop-id", default=LOOP_ID)
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions_official_context"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized_official_context"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--variant-name", default=DEFAULT_VARIANT)
    parser.add_argument("--trial-name", default=DEFAULT_TRIAL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--risk-epochs", type=int, default=10)
    parser.add_argument("--recovery-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-risk-teacher-entries", type=int, default=4000)
    parser.add_argument(
        "--lifecycle-mode",
        choices=("recovery_confirmation", "baseline_regret_objective", "giveback_onebar_confirmation"),
        default="recovery_confirmation",
        help=(
            "recovery_confirmation tests Protocol 057's second-stage recovery classifier; "
            "baseline_regret_objective tests Protocol 058's direct exit-regret objective; "
            "giveback_onebar_confirmation tests Protocol 059's one-bar confirmation for giveback exits."
        ),
    )
    parser.add_argument("--recovery-target-delta", type=float, default=RECOVERY_TARGET_DELTA)
    parser.add_argument("--recovery-probability-threshold", type=float, default=RECOVERY_PROBABILITY_THRESHOLD)
    parser.add_argument("--market-structure-source", choices=("v2_cache", "index_bars"), default="index_bars")
    parser.add_argument("--market-spx-dir", type=Path, default=Path("data/vendor/thetadata/index/spx_1m"))
    parser.add_argument("--market-vix-dir", type=Path, default=Path("data/vendor/thetadata/index/vix_1m"))
    parser.add_argument("--es-vwap-dir", type=Path, default=None)
    return parser.parse_args()


class RecoveryModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.06),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _risk_config_for_split(split: str) -> RiskConfig:
    name = PROTOCOL054_CONFIG_BY_SPLIT[split]
    configs = {config.name: config for config in _risk_grid("loss_or_giveback")}
    return configs[name]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return 0.0
    return float(numerator / denominator)


def _recovery_state_features(entry: EntryProposal, path: Sequence[PathPoint], idx: int) -> np.ndarray:
    base = _causal_state_features(entry, path, idx)
    entry_features = _option_entry_features(entry)
    now = path[idx].features
    pnls = np.asarray([point.pnl for point in path[: idx + 1]], dtype=np.float32)
    current_pnl = float(pnls[-1])
    mfe = float(np.max(pnls))
    denom = max(float(entry_features[_FEATURE_INDEX["ask"]]) * _CONTRACT_MULTIPLIER, 1.0)
    mfe_idx = int(np.argmax(pnls))
    mfe_age = idx - mfe_idx
    post_mfe_velocity = 0.0 if mfe_age <= 0 else (current_pnl - mfe) / mfe_age
    lookback_3 = min(3, idx)
    velocity_3 = 0.0 if lookback_3 == 0 else (pnls[-1] - pnls[-1 - lookback_3]) / lookback_3
    prev_velocity_3 = 0.0
    if idx >= 4:
        prev_velocity_3 = (pnls[-2] - pnls[-5]) / 3.0
    entry_gamma_theta = _safe_ratio(
        abs(float(entry_features[_FEATURE_INDEX["gamma"]])),
        abs(float(entry_features[_FEATURE_INDEX["theta"]])),
    )
    now_gamma_theta = _safe_ratio(abs(float(now[_FEATURE_INDEX["gamma"]])), abs(float(now[_FEATURE_INDEX["theta"]])))
    entry_spread = float(entry_features[_FEATURE_INDEX["spread_frac"]])
    now_spread = float(now[_FEATURE_INDEX["spread_frac"]])
    extras = np.asarray(
        [
            mfe_age / _POLICY.max_hold_minutes,
            post_mfe_velocity / denom,
            velocity_3 / denom,
            (velocity_3 - prev_velocity_3) / denom,
            now_gamma_theta - entry_gamma_theta,
            now_spread - entry_spread,
            np.log1p(max(0.0, float(now[_FEATURE_INDEX["bid_size"]]))),
            np.log1p(max(0.0, float(now[_FEATURE_INDEX["ask_size"]]))),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(np.concatenate([base, extras]), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def _recovery_samples_for(
    entries: Sequence[EntryProposal],
    *,
    path_for: Callable[[EntryProposal], list[PathPoint]],
    recovery_target_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for entry in entries:
        path = path_for(entry)
        if not path:
            continue
        pnls = np.asarray([point.pnl for point in path], dtype=np.float32)
        for idx, pnl in enumerate(pnls):
            future = pnls[idx:]
            future_best_delta = float(np.max(future) - pnl)
            future_final_delta = float(future[-1] - pnl)
            recovered = (
                future_best_delta >= recovery_target_delta
                or (future_best_delta >= 0.5 * recovery_target_delta and future_final_delta >= 0.25 * recovery_target_delta)
            )
            xs.append(_recovery_state_features(entry, path, idx))
            ys.append(float(recovered))
    if not xs:
        return (
            np.empty((0, len(RECOVERY_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def _fit_recovery_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[RecoveryModel, FeatureScaler, list[dict]]:
    if len(train_x) == 0:
        raise ValueError("cannot train recovery model with zero samples")
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = FeatureScaler.fit(train_x.astype(np.float32))
    x_train = scaler.transform(train_x.astype(np.float32))
    x_val = scaler.transform(val_x.astype(np.float32)) if len(val_x) else x_train
    y_train = train_y.astype(np.float32)
    y_val = val_y.astype(np.float32) if len(val_y) else y_train
    model = RecoveryModel(input_dim=x_train.shape[1])
    positives = float(np.sum(y_train > 0.5))
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(x_val))
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, torch.from_numpy(y_val), pos_weight=pos_weight).cpu())
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
                "validation_positive_rate": float(np.mean(y_val > 0.5)),
                "validation_predicted_positive_rate": float(np.mean(val_prob >= RECOVERY_PROBABILITY_THRESHOLD)),
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def _predict_recovery_probability(model: RecoveryModel, scaler: FeatureScaler, x: np.ndarray) -> float:
    x_scaled = scaler.transform(x.reshape(1, -1).astype(np.float32))
    model.eval()
    with torch.no_grad():
        return float(torch.sigmoid(model(torch.from_numpy(x_scaled))).cpu().numpy()[0])


def _simulate_recovery_confirmed_exit(
    entry: EntryProposal,
    path: Sequence[PathPoint],
    *,
    headroom_model,
    headroom_scaler,
    recovery_model: RecoveryModel,
    recovery_scaler: FeatureScaler,
    config: RiskConfig,
    recovery_probability_threshold: float,
    loop_id: str,
) -> tuple[Trade, dict]:
    if not path:
        trade = _trade_from_entry(entry, entry.baseline_pnl, f"{loop_id}:path_missing")
        return trade, {
            "exit_reason": "path_missing",
            "hold_minutes": None,
            "predicted_headroom": None,
            "predicted_recovery_probability": None,
            "blocked_recovery_exits": 0,
            "blocked_model_exits": 0,
        }

    entry_features = _option_entry_features(entry)
    entry_ask = float(entry_features[_FEATURE_INDEX["ask"]])
    stop_pnl = -_POLICY.stop_loss_pct * entry_ask * _CONTRACT_MULTIPLIER
    target_pnl = _POLICY.take_profit_pct * entry_ask * _CONTRACT_MULTIPLIER
    exit_point = path[-1]
    reason = "time_flat"
    predicted_headroom = None
    predicted_recovery = None
    blocked_model_exits = 0
    blocked_recovery_exits = 0
    constraint_details: dict = {}

    for idx, point in enumerate(path):
        hold_minutes = max(1.0, (point.time - entry.decision_time).total_seconds() / 60.0)
        if point.pnl <= stop_pnl:
            exit_point = point
            reason = "hard_stop"
            break
        if point.pnl >= target_pnl:
            exit_point = point
            reason = "target"
            break
        if hold_minutes < config.min_hold_minutes:
            continue
        x = _causal_state_features(entry, path, idx)
        predicted_headroom = _predict_headroom(headroom_model, headroom_scaler, x)
        if predicted_headroom <= config.exit_headroom_threshold:
            allowed, constraint_reason, details = _constraint_allows_model_exit(path, idx, config)
            constraint_details = details
            if allowed:
                recovery_x = _recovery_state_features(entry, path, idx)
                predicted_recovery = _predict_recovery_probability(recovery_model, recovery_scaler, recovery_x)
                if predicted_recovery >= recovery_probability_threshold:
                    blocked_recovery_exits += 1
                    continue
                exit_point = point
                reason = constraint_reason
                break
            blocked_model_exits += 1

    trade = _trade_from_entry(entry, exit_point.pnl, f"{loop_id}:{config.name}:{reason}")
    return trade, {
        "exit_reason": reason,
        "exit_time": exit_point.time.isoformat(),
        "hold_minutes": float((exit_point.time - entry.decision_time).total_seconds() / 60.0),
        "predicted_headroom": predicted_headroom,
        "predicted_recovery_probability": predicted_recovery,
        "blocked_recovery_exits": int(blocked_recovery_exits),
        "blocked_model_exits": int(blocked_model_exits),
        **constraint_details,
    }


def _simulate_giveback_onebar_confirmed_exit(
    entry: EntryProposal,
    path: Sequence[PathPoint],
    *,
    model,
    scaler,
    config: RiskConfig,
    loop_id: str,
) -> tuple[Trade, dict]:
    if not path:
        trade = _trade_from_entry(entry, entry.baseline_pnl, f"{loop_id}:path_missing")
        return trade, {
            "exit_reason": "path_missing",
            "exit_time": None,
            "hold_minutes": None,
            "predicted_headroom": None,
            "blocked_model_exits": 0,
            "pending_giveback_confirmations": 0,
            "canceled_giveback_confirmations": 0,
        }

    entry_features = _option_entry_features(entry)
    entry_ask = float(entry_features[_FEATURE_INDEX["ask"]])
    stop_pnl = -_POLICY.stop_loss_pct * entry_ask * _CONTRACT_MULTIPLIER
    target_pnl = _POLICY.take_profit_pct * entry_ask * _CONTRACT_MULTIPLIER
    exit_point = path[-1]
    reason = "time_flat"
    predicted_headroom = None
    blocked_model_exits = 0
    pending_giveback_pnl: float | None = None
    pending_giveback_confirmations = 0
    canceled_giveback_confirmations = 0
    constraint_details: dict = {}

    for idx, point in enumerate(path):
        hold_minutes = max(1.0, (point.time - entry.decision_time).total_seconds() / 60.0)
        if point.pnl <= stop_pnl:
            exit_point = point
            reason = "hard_stop"
            break
        if point.pnl >= target_pnl:
            exit_point = point
            reason = "target"
            break
        if hold_minutes < config.min_hold_minutes:
            continue
        x = _causal_state_features(entry, path, idx)
        predicted_headroom = _predict_headroom(model, scaler, x)
        if predicted_headroom > config.exit_headroom_threshold:
            if pending_giveback_pnl is not None:
                canceled_giveback_confirmations += 1
            pending_giveback_pnl = None
            continue
        allowed, constraint_reason, details = _constraint_allows_model_exit(path, idx, config)
        constraint_details = details
        if not allowed:
            blocked_model_exits += 1
            continue
        if constraint_reason != "model_exit_giveback":
            exit_point = point
            reason = constraint_reason
            break
        if pending_giveback_pnl is not None and point.pnl <= pending_giveback_pnl:
            exit_point = point
            reason = constraint_reason
            break
        if pending_giveback_pnl is not None:
            canceled_giveback_confirmations += 1
        pending_giveback_pnl = float(point.pnl)
        pending_giveback_confirmations += 1

    trade = _trade_from_entry(entry, exit_point.pnl, f"{loop_id}:{config.name}:{reason}")
    return trade, {
        "exit_reason": reason,
        "exit_time": exit_point.time.isoformat(),
        "hold_minutes": float((exit_point.time - entry.decision_time).total_seconds() / 60.0),
        "predicted_headroom": predicted_headroom,
        "blocked_model_exits": int(blocked_model_exits),
        "pending_giveback_confirmations": int(pending_giveback_confirmations),
        "canceled_giveback_confirmations": int(canceled_giveback_confirmations),
        **constraint_details,
    }


def _samples_for(
    entries: Sequence[EntryProposal],
    *,
    path_for: Callable[[EntryProposal], list[PathPoint]],
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for entry in entries:
        x, y = _path_samples(entry, path_for(entry))
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (
            np.empty((0, len(RISK_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def _baseline_regret_samples_for(
    entries: Sequence[EntryProposal],
    *,
    path_for: Callable[[EntryProposal], list[PathPoint]],
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for entry in entries:
        path = path_for(entry)
        if not path:
            continue
        for idx, point in enumerate(path):
            regret = max(0.0, float(entry.baseline_pnl) - float(point.pnl))
            xs.append(_causal_state_features(entry, path, idx))
            ys.append(min(regret, BASELINE_REGRET_TARGET_CAP) / _RISK_TARGET_SCALE)
    if not xs:
        return (
            np.empty((0, len(RISK_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def _risk_samples_for_mode(
    entries: Sequence[EntryProposal],
    *,
    path_for: Callable[[EntryProposal], list[PathPoint]],
    lifecycle_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if lifecycle_mode == "baseline_regret_objective":
        return _baseline_regret_samples_for(entries, path_for=path_for)
    return _samples_for(entries, path_for=path_for)


def _write_report(path: Path, payload: dict) -> None:
    p054 = {row["split"]: row for row in payload["protocol054_summary"]}
    entry = {row["split"]: row for row in payload["entry_summary"]}
    lines = [
        f"# {payload['loop_id']}",
        "",
        "No paid data was downloaded. Protocol 051 entries and Protocol 054 lifecycle configs are frozen.",
        "",
        "## Hypothesis",
        "",
        payload["pre_registration"]["hypothesis"],
        "",
        "## Walk-Forward Summary",
        "",
        "| Split | Candidate PnL | Protocol 054 PnL | Delta vs 054 | Entry PnL | PF | Trades | +50 | +100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["recovery_summary"]:
        split = row["split"]
        p054_pnl = float(p054.get(split, {}).get("pnl_median", 0.0))
        entry_pnl = float(entry.get(split, {}).get("pnl_median", 0.0))
        lines.append(
            f"| {split} | {row['pnl_median']:.0f} | {p054_pnl:.0f} | {row['pnl_median'] - p054_pnl:.0f} | "
            f"{entry_pnl:.0f} | {row['pf_median']:.3f} | {row['trades_median']:.0f} | "
            f"{row['stress50_pnl_median']:.0f} | {row['stress100_pnl_median']:.0f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "This is not paper/live approval.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _run_fold(
    *,
    fold: FoldSpec,
    variant: SurfaceVariant,
    trial: ProtocolTrial,
    market_cache: MarketStructureCache,
    decision_cache_dir: Path | None,
    normalized_store: NormalizedPathStore,
    args: argparse.Namespace,
) -> dict:
    policy_name, cooldown = POLICY_META[args.policy_index]
    split = _split_name(fold)
    risk_config = _risk_config_for_split(split)
    session_rows = _load_session_rows(
        {
            "train": fold.train_paths,
            "validation": fold.validation_paths,
            "test": fold.test_paths,
        }
    )
    train_decisions = _load_surface_decisions_cached(
        fold.train_paths,
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_train",
        cache_dir=decision_cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        fold.validation_paths,
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_validation",
        cache_dir=decision_cache_dir,
    )
    test_decisions = _load_surface_decisions_cached(
        fold.test_paths,
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_test",
        cache_dir=decision_cache_dir,
    )

    recovery_rows = []
    protocol054_rows = []
    entry_rows = []
    selected_trades = []
    seed_artifacts = []

    for seed in args.seeds:
        effective_seed = window_seed(seed, fold.window_id)
        print(f"{args.loop_id} fold={fold.name} seed={seed}", flush=True)
        entry_config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        entry_model, entry_standardizer, entry_history = train_surface_model(
            train_decisions,
            validation_decisions,
            config=entry_config,
            variant=variant,
        )
        predictions_by_split = {
            "train": predict_surface_actions(entry_model, entry_standardizer, train_decisions, target_scale=entry_config.target_scale),
            "validation": predict_surface_actions(entry_model, entry_standardizer, validation_decisions, target_scale=entry_config.target_scale),
            "test": predict_surface_actions(entry_model, entry_standardizer, test_decisions, target_scale=entry_config.target_scale),
        }
        proposals_by_split = {
            "train": _select_entry_proposals(
                train_decisions,
                predictions_by_split["train"],
                trial=trial,
                cooldown_minutes=cooldown,
                split="train",
                seed=seed,
                effective_seed=effective_seed,
            ),
            "validation": _select_entry_proposals(
                validation_decisions,
                predictions_by_split["validation"],
                trial=trial,
                cooldown_minutes=cooldown,
                split="validation",
                seed=seed,
                effective_seed=effective_seed,
            ),
            "test": _select_entry_proposals(
                test_decisions,
                predictions_by_split["test"],
                trial=trial,
                cooldown_minutes=cooldown,
                split=split,
                seed=seed,
                effective_seed=effective_seed,
            ),
        }
        path_cache: dict[tuple[str, str, object], list[PathPoint]] = {}

        def path_for(entry: EntryProposal) -> list[PathPoint]:
            key = (entry.session, entry.decision_time.isoformat(), entry.contract_id)
            if key not in path_cache:
                if normalized_store is not None:
                    path_cache[key] = normalized_store.contract_path(entry)
                else:
                    path_cache[key] = _contract_path(entry, session_rows.get(entry.session, []))
            return path_cache[key]

        train_teacher = _bounded_entries(
            _training_candidate_proposals(
                train_decisions,
                trial=trial,
                split="train_teacher_candidates",
                seed=seed,
                effective_seed=effective_seed,
            ),
            limit=args.max_risk_teacher_entries,
            seed=effective_seed,
        )
        validation_teacher = _bounded_entries(
            _training_candidate_proposals(
                validation_decisions,
                trial=trial,
                split="validation_teacher_candidates",
                seed=seed,
                effective_seed=effective_seed,
            ),
            limit=max(250, args.max_risk_teacher_entries // 4),
            seed=effective_seed + 1,
        )
        train_entries = list(proposals_by_split["train"]) + train_teacher
        validation_entries = list(proposals_by_split["validation"]) + validation_teacher
        protocol054_train_x, protocol054_train_y = _samples_for(train_entries, path_for=path_for)
        protocol054_val_x, protocol054_val_y = _samples_for(validation_entries, path_for=path_for)
        candidate_train_x, candidate_train_y = _risk_samples_for_mode(
            train_entries,
            path_for=path_for,
            lifecycle_mode=args.lifecycle_mode,
        )
        candidate_val_x, candidate_val_y = _risk_samples_for_mode(
            validation_entries,
            path_for=path_for,
            lifecycle_mode=args.lifecycle_mode,
        )
        recovery_train_x = np.empty((0, len(RECOVERY_FEATURE_NAMES)), dtype=np.float32)
        recovery_train_y = np.empty((0,), dtype=np.float32)
        recovery_val_x = np.empty((0, len(RECOVERY_FEATURE_NAMES)), dtype=np.float32)
        recovery_val_y = np.empty((0,), dtype=np.float32)
        if args.lifecycle_mode == "recovery_confirmation":
            recovery_train_x, recovery_train_y = _recovery_samples_for(
                train_entries,
                path_for=path_for,
                recovery_target_delta=args.recovery_target_delta,
            )
            recovery_val_x, recovery_val_y = _recovery_samples_for(
                validation_entries,
                path_for=path_for,
                recovery_target_delta=args.recovery_target_delta,
            )
        if len(protocol054_train_x) == 0 or len(candidate_train_x) == 0:
            raise SystemExit(f"{fold.name} seed {seed}: no lifecycle training samples")
        if args.lifecycle_mode == "recovery_confirmation" and len(recovery_train_x) == 0:
            raise SystemExit(f"{fold.name} seed {seed}: no recovery training samples")
        protocol054_model, protocol054_scaler, protocol054_risk_history = _fit_risk_model(
            protocol054_train_x,
            protocol054_train_y,
            protocol054_val_x,
            protocol054_val_y,
            seed=effective_seed,
            epochs=args.risk_epochs,
            batch_size=args.batch_size,
        )
        candidate_model = protocol054_model
        candidate_scaler = protocol054_scaler
        candidate_risk_history = protocol054_risk_history
        if args.lifecycle_mode == "baseline_regret_objective":
            candidate_model, candidate_scaler, candidate_risk_history = _fit_risk_model(
                candidate_train_x,
                candidate_train_y,
                candidate_val_x,
                candidate_val_y,
                seed=effective_seed + 19,
                epochs=args.risk_epochs,
                batch_size=args.batch_size,
            )
        recovery_model = None
        recovery_scaler = None
        recovery_history = []
        if args.lifecycle_mode == "recovery_confirmation":
            recovery_model, recovery_scaler, recovery_history = _fit_recovery_model(
                recovery_train_x,
                recovery_train_y,
                recovery_val_x,
                recovery_val_y,
                seed=effective_seed + 17,
                epochs=args.recovery_epochs,
                batch_size=args.batch_size,
            )

        dynamic_trades = []
        protocol054_trades = []
        entry_trades = []
        for entry in proposals_by_split["test"]:
            path = path_for(entry)
            if args.lifecycle_mode == "recovery_confirmation":
                recovery_trade, recovery_info = _simulate_recovery_confirmed_exit(
                    entry,
                    path,
                    headroom_model=protocol054_model,
                    headroom_scaler=protocol054_scaler,
                    recovery_model=recovery_model,
                    recovery_scaler=recovery_scaler,
                    config=risk_config,
                    recovery_probability_threshold=args.recovery_probability_threshold,
                    loop_id=args.loop_id,
                )
            elif args.lifecycle_mode == "giveback_onebar_confirmation":
                recovery_trade, recovery_info = _simulate_giveback_onebar_confirmed_exit(
                    entry,
                    path,
                    model=protocol054_model,
                    scaler=protocol054_scaler,
                    config=risk_config,
                    loop_id=args.loop_id,
                )
            else:
                recovery_trade, recovery_info = _simulate_lifecycle_exit(
                    entry,
                    path,
                    model=candidate_model,
                    scaler=candidate_scaler,
                    config=risk_config,
                    loop_id=args.loop_id,
                )
            protocol054_trade, protocol054_info = _simulate_lifecycle_exit(
                entry,
                path,
                model=protocol054_model,
                scaler=protocol054_scaler,
                config=risk_config,
                loop_id="protocol054_fixed_config",
            )
            entry_trade = _trade_from_entry(entry, entry.baseline_pnl, f"{args.loop_id}:protocol051_entry_baseline")
            dynamic_trades.append(recovery_trade)
            protocol054_trades.append(protocol054_trade)
            entry_trades.append(entry_trade)
            selected_trades.append(
                {
                    "fold": fold.name,
                    "split": split,
                    "seed": int(seed),
                    "session": entry.session,
                    "decision_time": entry.decision_time.isoformat(),
                    "contract_id": str(entry.contract_id),
                    "right": entry.right,
                    "offset": entry.offset,
                    "edge": entry.edge,
                    "entry_baseline_pnl": entry.baseline_pnl,
                    "protocol054_pnl": protocol054_trade.pnl,
                    "candidate_pnl": recovery_trade.pnl,
                    "recovery_pnl": recovery_trade.pnl,
                    "protocol054_exit_reason": protocol054_info.get("exit_reason"),
                    **{f"recovery_{key}": value for key, value in recovery_info.items()},
                }
            )

        row = {
            "fold": fold.name,
            "split": split,
            "seed": seed,
            "metrics": _trade_metrics(dynamic_trades),
            "stress50_metrics": _trade_metrics(stress_trades(dynamic_trades, extra_cost_per_trade=50.0)),
            "stress100_metrics": _trade_metrics(stress_trades(dynamic_trades, extra_cost_per_trade=100.0)),
        }
        p054_row = {
            "fold": fold.name,
            "split": split,
            "seed": seed,
            "metrics": _trade_metrics(protocol054_trades),
            "stress50_metrics": _trade_metrics(stress_trades(protocol054_trades, extra_cost_per_trade=50.0)),
            "stress100_metrics": _trade_metrics(stress_trades(protocol054_trades, extra_cost_per_trade=100.0)),
        }
        entry_row = {
            "fold": fold.name,
            "split": split,
            "seed": seed,
            "metrics": _trade_metrics(entry_trades),
            "stress50_metrics": _trade_metrics(stress_trades(entry_trades, extra_cost_per_trade=50.0)),
            "stress100_metrics": _trade_metrics(stress_trades(entry_trades, extra_cost_per_trade=100.0)),
        }
        recovery_rows.append(row)
        protocol054_rows.append(p054_row)
        entry_rows.append(entry_row)
        if split == "q1_2026":
            march_dynamic = [trade for trade in dynamic_trades if trade.session >= "2026-03-01"]
            march_p054 = [trade for trade in protocol054_trades if trade.session >= "2026-03-01"]
            march_entry = [trade for trade in entry_trades if trade.session >= "2026-03-01"]
            recovery_rows.append(
                {
                    "fold": fold.name,
                    "split": "march_2026",
                    "seed": seed,
                    "metrics": _trade_metrics(march_dynamic),
                    "stress50_metrics": _trade_metrics(stress_trades(march_dynamic, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _trade_metrics(stress_trades(march_dynamic, extra_cost_per_trade=100.0)),
                }
            )
            protocol054_rows.append(
                {
                    "fold": fold.name,
                    "split": "march_2026",
                    "seed": seed,
                    "metrics": _trade_metrics(march_p054),
                    "stress50_metrics": _trade_metrics(stress_trades(march_p054, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _trade_metrics(stress_trades(march_p054, extra_cost_per_trade=100.0)),
                }
            )
            entry_rows.append(
                {
                    "fold": fold.name,
                    "split": "march_2026",
                    "seed": seed,
                    "metrics": _trade_metrics(march_entry),
                    "stress50_metrics": _trade_metrics(stress_trades(march_entry, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _trade_metrics(stress_trades(march_entry, extra_cost_per_trade=100.0)),
                }
            )

        seed_artifacts.append(
            {
                "seed": int(seed),
                "effective_seed": int(effective_seed),
                "entry_history": entry_history,
                "risk_history": candidate_risk_history,
                "candidate_risk_history": candidate_risk_history,
                "protocol054_risk_history": protocol054_risk_history,
                "recovery_history": recovery_history,
                "risk_train_samples": int(len(candidate_train_x)),
                "risk_validation_samples": int(len(candidate_val_x)),
                "protocol054_risk_train_samples": int(len(protocol054_train_x)),
                "protocol054_risk_validation_samples": int(len(protocol054_val_x)),
                "recovery_train_samples": int(len(recovery_train_x)),
                "recovery_validation_samples": int(len(recovery_val_x)),
                "recovery_train_positive_rate": float(np.mean(recovery_train_y > 0.5)) if len(recovery_train_y) else None,
                "recovery_validation_positive_rate": float(np.mean(recovery_val_y > 0.5)) if len(recovery_val_y) else None,
                "test_entries": int(len(proposals_by_split["test"])),
            }
        )

    return {
        "fold": fold.summary(),
        "fold_name": fold.name,
        "split": split,
        "risk_config": asdict(risk_config),
        "seed_artifacts": seed_artifacts,
        "recovery_seed_rows": recovery_rows,
        "protocol054_seed_rows": protocol054_rows,
        "entry_seed_rows": entry_rows,
        "selected_trades": selected_trades,
    }


def main() -> int:
    args = parse_args()
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    folds = _folds(args)
    market_cache = MarketStructureCache(
        source=args.market_structure_source,
        index_spx_dir=args.market_spx_dir,
        index_vix_dir=args.market_vix_dir,
        es_vwap_dir=args.es_vwap_dir,
    )
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    normalized_store = NormalizedPathStore(args.normalized_dir)
    fold_payloads = [
        _run_fold(
            fold=fold,
            variant=variant,
            trial=trial,
            market_cache=market_cache,
            decision_cache_dir=decision_cache_dir,
            normalized_store=normalized_store,
            args=args,
        )
        for fold in folds
    ]

    recovery_rows = [row for payload in fold_payloads for row in payload["recovery_seed_rows"]]
    protocol054_rows = [row for payload in fold_payloads for row in payload["protocol054_seed_rows"]]
    entry_rows = [row for payload in fold_payloads for row in payload["entry_seed_rows"]]
    selected_trades = [row for payload in fold_payloads for row in payload["selected_trades"]]
    splits = ["q2_2025", "q3_2025", "q4_2025", "q1_2026", "march_2026"]
    recovery_summary = _summary_by_split(recovery_rows, splits=splits)
    protocol054_summary = _summary_by_split(protocol054_rows, splits=splits)
    entry_summary = _summary_by_split(entry_rows, splits=splits)
    recovery_lookup = {row["split"]: row for row in recovery_summary}
    p054_lookup = {row["split"]: row for row in protocol054_summary}
    scored_splits = ["q2_2025", "q3_2025", "q4_2025", "q1_2026"]
    beat_folds = sum(
        1
        for split in scored_splits
        if recovery_lookup.get(split, {}).get("pnl_median", -1e9)
        > p054_lookup.get(split, {}).get("pnl_median", 1e9)
    )
    stress_positive = all(
        recovery_lookup.get(split, {}).get("stress50_pnl_median", -1.0) > 0.0
        for split in scored_splits
    )
    q2_not_worse = (
        recovery_lookup.get("q2_2025", {}).get("pnl_median", -1e9)
        >= p054_lookup.get("q2_2025", {}).get("pnl_median", 1e9)
    )
    march_not_worse = (
        recovery_lookup.get("march_2026", {}).get("pnl_median", -1e9)
        >= p054_lookup.get("march_2026", {}).get("pnl_median", 1e9)
    )
    if beat_folds >= 3 and stress_positive and q2_not_worse and march_not_worse:
        decision = (
            f"Keep {args.loop_id} as a lifecycle challenger and advance to 10-seed validation. "
            "It improves Protocol 054 in at least 3 of 4 folds, preserves Q2/March, and keeps +50 stress positive."
        )
    else:
        decision = (
            f"Reject {args.loop_id} as a replacement for Protocol 054 at this gate. "
            "Do not tune this lifecycle hypothesis without a new diagnosis."
        )
    if args.lifecycle_mode == "baseline_regret_objective":
        hypothesis = (
            "A direct causal baseline-regret objective should teach the lifecycle model the expected cost "
            "of exiting now versus the frozen stop/target/time path. This targets the Protocol 056 failure "
            "mode directly: temporary pullbacks that later recover to the frozen baseline should retain "
            "higher hold value, while true continuation decay should retain low hold value. Entries, hard "
            "stops, targets, time-flat, and Protocol 054 split configs remain unchanged."
        )
    elif args.lifecycle_mode == "giveback_onebar_confirmation":
        hypothesis = (
            "A fixed one-bar confirmation should reduce clipped winners from model_exit_giveback without "
            "touching model_exit_loss. When Protocol 054 wants a giveback exit, the candidate waits for one "
            "additional same-contract minute and exits only if the exit signal persists without PnL recovery. "
            "Hard stops, targets, loss exits, time-flat, entries, and Protocol 054 split configs remain unchanged."
        )
    else:
        hypothesis = (
            "A separate causal recovery-confirmation model should block model_exit_loss/model_exit_giveback "
            "only when the current post-entry state looks likely to recover by at least the fixed recovery "
            "target. Hard stops, targets, time-flat, and entries remain unchanged."
        )

    payload = {
        "loop_id": args.loop_id,
        "pre_registration": {
            "paid_data_downloaded": False,
            "lifecycle_mode": args.lifecycle_mode,
            "entry_variant_frozen": args.variant_name,
            "trial_frozen": args.trial_name,
            "protocol054_configs_frozen": PROTOCOL054_CONFIG_BY_SPLIT,
            "baseline_regret_target_cap": BASELINE_REGRET_TARGET_CAP,
            "recovery_target_delta": args.recovery_target_delta,
            "recovery_probability_threshold": args.recovery_probability_threshold,
            "hypothesis": hypothesis,
            "acceptance_rule": (
                "Advance only if the candidate beats Protocol 054 in at least 3 of 4 quarterly folds, "
                "matches or improves Q2, does not damage March, and keeps +50 stress median positive."
            ),
        },
        "args": {
            "out_dir": str(args.out_dir),
            "seeds": args.seeds,
            "epochs": args.epochs,
            "risk_epochs": args.risk_epochs,
            "recovery_epochs": args.recovery_epochs,
            "lifecycle_mode": args.lifecycle_mode,
            "batch_size": args.batch_size,
            "normalized_dir": str(args.normalized_dir),
            "decision_cache_dir": None if decision_cache_dir is None else str(decision_cache_dir),
            "market_structure_source": args.market_structure_source,
        },
        "entry": {
            "variant_name": variant.name,
            "variant": asdict(variant) | {"variant_id": variant.variant_id},
            "policy_index": args.policy_index,
            "policy_name": POLICY_META[args.policy_index][0],
            "trial_name": trial.name,
            "trial": asdict(trial) | {"config_id": trial.config_id},
        },
        "candidate_summary": recovery_summary,
        "recovery_feature_names": list(RECOVERY_FEATURE_NAMES),
        "fold_results": [
            {
                key: value
                for key, value in payload.items()
                if key not in {"selected_trades"}
            }
            for payload in fold_payloads
        ],
        "recovery_summary": recovery_summary,
        "protocol054_summary": protocol054_summary,
        "entry_summary": entry_summary,
        "selected_trades_file": str(args.out_dir / "selected_trades_with_recovery_confirmation.json"),
        "decision": decision,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_trades_with_recovery_confirmation.json").write_text(
        json.dumps(selected_trades, indent=2, allow_nan=False) + "\n"
    )
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps(recovery_summary, indent=2, sort_keys=True), flush=True)
    print(args.out_dir / "report.md", flush=True)
    print(decision, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
