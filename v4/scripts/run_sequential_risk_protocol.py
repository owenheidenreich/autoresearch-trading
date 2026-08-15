"""Sequential risk-management protocol for v4 SPXW 0DTE.

This freezes an entry baseline and trains a separate causal
exit/hold model on already-collected one-minute option paths. The entry model
still decides "enter now or wait"; the risk model only acts after entry:
hold, exit, or hard stop/forced flat.

No paid data is downloaded here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
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

from v4.dataset.spxw_0dte_neural import LabelPolicy, OPTION_FEATURE_NAMES
from v4.model.environment_diagnostics import time_bucket
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceDecision,
    SurfaceVariant,
    registered_aplus_surface_variants,
    registered_protocol_trials,
    train_surface_model,
    predict_surface_actions,
    stress_trades,
    summarize_random_baseline,
    token_feature_names,
    window_seed,
)
from v4.model.supervised_pilot import FeatureScaler, PilotConfig, Trade, session_from_path
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


DEFAULT_LOOP_ID = "v4_aplus_hypothesis_029_sequential_risk_layer"
DEFAULT_REPORT_TITLE = "Protocol 029: Sequential Risk Layer"
_NY = ZoneInfo("America/New_York")
_CONTRACT_MULTIPLIER = 100.0
_RISK_TARGET_SCALE = 100.0
_POLICY = LabelPolicy(0.50, 1.00, 25)
_FEATURE_INDEX = {name: idx for idx, name in enumerate(OPTION_FEATURE_NAMES)}


RISK_FEATURE_NAMES = (
    "hold_frac",
    "remaining_frac",
    "minutes_to_forced_flat_frac",
    "current_pnl_norm",
    "mfe_norm",
    "mae_norm",
    "giveback_norm",
    "pnl_velocity_5_norm",
    "realized_vol_10_norm",
    "mfe_decay_norm",
    "current_return",
    "bid_over_entry_ask",
    "mid_over_entry_ask",
    "spread_frac",
    "entry_spread_frac",
    "abs_delta",
    "delta_change",
    "gamma_abs",
    "gamma_change",
    "theta_abs",
    "theta_change",
    "gamma_theta_ratio",
    "theta_over_mid",
    "time_theta_burden",
    "entry_edge_norm",
    "entry_pattern_target",
    "entry_value_target",
    "is_call",
    "is_put",
    "offset_norm",
)


@dataclass
class EntryProposal:
    split: str
    seed: int
    effective_seed: int
    decision: SurfaceDecision
    token_idx: int
    edge: float
    baseline_pnl: float

    @property
    def session(self) -> str:
        return self.decision.session

    @property
    def decision_time(self) -> datetime:
        return self.decision.decision_time

    @property
    def contract_id(self) -> object:
        return self.decision.contract_ids[self.token_idx]

    @property
    def right(self) -> str:
        return str(self.decision.rights[self.token_idx])

    @property
    def offset(self) -> float:
        return float(self.decision.offsets[self.token_idx])


@dataclass
class PathPoint:
    time: datetime
    features: np.ndarray
    pnl: float


@dataclass(frozen=True)
class RiskConfig:
    name: str
    exit_headroom_threshold: float
    min_hold_minutes: int
    exit_constraint: str = "unconstrained"
    min_mfe: float = 0.0
    giveback_trigger: float = 0.0
    giveback_fraction: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived_nofee"))
    p.add_argument("--q1-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_nofee"))
    p.add_argument("--q2-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_nofee"))
    p.add_argument("--q3-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_nofee"))
    p.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_nofee"))
    p.add_argument("--seed-q4-data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_029_sequential_risk_layer"))
    p.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions_nofee"))
    p.add_argument("--risk-cache-dir", type=Path, default=Path("data/cache/v4_sequential_risk_paths"))
    p.add_argument("--loop-id", default=DEFAULT_LOOP_ID)
    p.add_argument("--report-title", default=DEFAULT_REPORT_TITLE)
    p.add_argument("--exit-constraint", choices=["unconstrained", "loss_or_giveback", "loss_only"], default="unconstrained")
    p.add_argument("--no-decision-cache", action="store_true")
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    p.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    p.add_argument("--variant-name", default="surface_structure_aplus_side_value_multitask")
    p.add_argument("--trial-name", default="post_open_late_edge25_max2")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--risk-epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--market-structure-source", choices=("v2_cache", "index_bars"), default="v2_cache")
    p.add_argument("--market-spx-dir", type=Path, default=None)
    p.add_argument("--market-vix-dir", type=Path, default=None)
    p.add_argument("--es-vwap-dir", type=Path, default=None)
    return p.parse_args()


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


def _risk_grid(exit_constraint: str) -> list[RiskConfig]:
    out = []
    if exit_constraint == "unconstrained":
        for threshold in (0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0):
            for min_hold in (1, 3, 5):
                out.append(
                    RiskConfig(
                        name=f"headroom_le_{int(threshold)}_minhold_{min_hold}",
                        exit_headroom_threshold=threshold,
                        min_hold_minutes=min_hold,
                        exit_constraint=exit_constraint,
                    )
                )
        return out
    if exit_constraint == "loss_only":
        for threshold in (0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0):
            for min_hold in (1, 3, 5):
                out.append(
                    RiskConfig(
                        name=f"headroom_le_{int(threshold)}_minhold_{min_hold}_lossonly",
                        exit_headroom_threshold=threshold,
                        min_hold_minutes=min_hold,
                        exit_constraint=exit_constraint,
                    )
                )
        return out
    if exit_constraint != "loss_or_giveback":
        raise ValueError(f"unknown exit constraint: {exit_constraint}")
    for threshold in (25.0, 50.0, 100.0, 150.0):
        for min_hold in (1, 3, 5):
            for giveback_trigger in (50.0, 100.0):
                for giveback_fraction in (0.35, 0.50):
                    out.append(
                        RiskConfig(
                            name=(
                                f"headroom_le_{int(threshold)}_minhold_{min_hold}"
                                f"_gb{int(giveback_trigger)}_gbfrac{int(giveback_fraction * 100)}"
                            ),
                            exit_headroom_threshold=threshold,
                            min_hold_minutes=min_hold,
                            exit_constraint=exit_constraint,
                            min_mfe=100.0,
                            giveback_trigger=giveback_trigger,
                            giveback_fraction=giveback_fraction,
                        )
                    )
    return out


def _constraint_allows_model_exit(
    path: Sequence[PathPoint],
    idx: int,
    config: RiskConfig,
) -> tuple[bool, str, dict]:
    if config.exit_constraint == "unconstrained":
        return True, "model_exit", {}
    if config.exit_constraint not in {"loss_or_giveback", "loss_only"}:
        raise ValueError(f"unknown exit constraint: {config.exit_constraint}")
    pnls = np.asarray([point.pnl for point in path[: idx + 1]], dtype=np.float32)
    current_pnl = float(pnls[-1])
    mfe = float(np.max(pnls))
    mae = float(np.min(pnls))
    giveback = max(0.0, mfe - current_pnl)
    giveback_fraction = giveback / max(mfe, 1.0) if mfe > 0.0 else 0.0
    details = {
        "current_pnl": current_pnl,
        "mfe": mfe,
        "mae": mae,
        "giveback": giveback,
        "giveback_fraction": giveback_fraction,
    }
    if current_pnl < 0.0:
        return True, "model_exit_loss", details
    if config.exit_constraint == "loss_only":
        return False, "model_blocked_constraint", details
    if (
        mfe >= config.min_mfe
        and giveback >= config.giveback_trigger
        and giveback_fraction >= config.giveback_fraction
    ):
        return True, "model_exit_giveback", details
    return False, "model_blocked_constraint", details


def _risk_framing(exit_constraint: str) -> str:
    if exit_constraint == "loss_only":
        return (
            "Freeze the entry baseline and test a loss-only causal sequential "
            "risk layer. The risk model learns future headroom from minute-by-minute position "
            "state: PnL, MFE/MAE, PnL velocity, MFE decay, Greeks/theta burden, time left, "
            "and entry contract overpay/value context. It can exit early only while current "
            "bid-to-entry PnL is negative; winners remain under the original stop/target/time "
            "lifecycle so the convex right tail is not clipped by generic giveback rules."
        )
    if exit_constraint == "loss_or_giveback":
        return (
            "Freeze the entry baseline and test a constrained causal sequential "
            "risk layer. The risk model learns future headroom from minute-by-minute position "
            "state: PnL, MFE/MAE, PnL velocity, MFE decay, Greeks/theta burden, time left, "
            "and entry contract overpay/value context. It can exit early only when the trade "
            "is losing or when a prior winner has given back meaningful MFE; otherwise winners "
            "remain under the original stop/target/time lifecycle."
        )
    return (
        "Freeze the entry baseline and test the first causal sequential "
        "risk layer. The risk model learns future headroom from minute-by-minute position "
        "state: PnL, MFE/MAE, PnL velocity, MFE decay, Greeks/theta burden, time left, "
        "and entry contract overpay/value context. It can exit early; hard stop and "
        "forced flat remain mandatory."
    )


def _trade_from_entry(entry: EntryProposal, pnl: float, strategy: str) -> Trade:
    return Trade(
        session=entry.session,
        decision_time=entry.decision_time.isoformat(),
        pnl=float(pnl),
        score=float(entry.edge),
        right=entry.right,
        offset=float(entry.offset),
        strategy=strategy,
    )


def _select_entry_proposals(
    decisions: Sequence[SurfaceDecision],
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    split: str,
    seed: int,
    effective_seed: int,
) -> list[EntryProposal]:
    proposals: list[EntryProposal] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(trial.allowed_buckets)
    for decision, pred in zip(decisions, predictions):
        bucket = time_bucket(decision.decision_time)
        if bucket not in allowed:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        action_mask = np.concatenate([[True], decision.token_mask])
        masked = np.asarray(pred, dtype=float).copy()
        masked[~action_mask] = -np.inf
        if not np.isfinite(masked).any():
            continue
        action = int(np.nanargmax(masked))
        if action == 0:
            continue
        edge = float(masked[action] - masked[0])
        if not np.isfinite(edge) or edge < trial.min_edge_vs_no_trade:
            continue
        token_idx = action - 1
        pnl = float(decision.labels[token_idx])
        if not np.isfinite(pnl):
            continue
        proposal = EntryProposal(
            split=split,
            seed=int(seed),
            effective_seed=int(effective_seed),
            decision=decision,
            token_idx=int(token_idx),
            edge=edge,
            baseline_pnl=pnl,
        )
        proposals.append(proposal)
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
    return proposals


def _training_candidate_proposals(
    decisions: Sequence[SurfaceDecision],
    *,
    trial: ProtocolTrial,
    split: str,
    seed: int,
    effective_seed: int,
) -> list[EntryProposal]:
    """Extra train/calibration-only paths for the exit model.

    The frozen entry policy remains the only source of evaluation trades. This
    function merely gives the risk model enough causal paths to learn from by
    sampling A+ pattern/value candidate contracts in allowed time buckets from
    the training periods.
    """

    proposals: list[EntryProposal] = []
    allowed = set(trial.allowed_buckets)
    for decision in decisions:
        if time_bucket(decision.decision_time) not in allowed:
            continue
        valid = decision.token_mask & np.isfinite(decision.labels)
        if decision.pattern_targets is not None:
            valid &= decision.pattern_targets > 0.5
        if decision.value_targets is not None:
            valid |= (decision.token_mask & np.isfinite(decision.labels) & (decision.value_targets > 0.5))
        idxs = np.where(valid)[0]
        if len(idxs) == 0:
            continue
        labels = decision.labels[idxs]
        chosen: list[int] = []
        positives = idxs[labels > 0.0]
        negatives = idxs[labels <= 0.0]
        if len(positives):
            chosen.append(int(positives[np.argmax(decision.labels[positives])]))
        if len(negatives):
            chosen.append(int(negatives[np.argmin(decision.labels[negatives])]))
        for token_idx in dict.fromkeys(chosen):
            proposals.append(
                EntryProposal(
                    split=split,
                    seed=int(seed),
                    effective_seed=int(effective_seed),
                    decision=decision,
                    token_idx=int(token_idx),
                    edge=float(decision.labels[token_idx]),
                    baseline_pnl=float(decision.labels[token_idx]),
                )
            )
    return proposals


def _load_session_rows(paths_by_split: dict[str, Sequence[Path]]) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for paths in paths_by_split.values():
        for path in paths:
            session = session_from_path(path)
            if session in sessions:
                continue
            with path.open("rb") as handle:
                rows = pickle.load(handle)
            sessions[session] = sorted(rows, key=lambda row: row["decision_time"])
    return sessions


def _token_entry_features(entry: EntryProposal) -> np.ndarray:
    return np.asarray(entry.decision.token_features[entry.token_idx], dtype=np.float32)


def _option_entry_features(entry: EntryProposal) -> np.ndarray:
    # token_idx flattening follows strike-major, right-minor order.
    rights = np.asarray(("C", "P"), dtype=object)
    right_count = len(rights)
    strike_idx = int(entry.token_idx // right_count)
    right_idx = int(entry.token_idx % right_count)
    # Surface decisions are flattened from the source row; the raw option
    # feature vector is the prefix for all aplus token modes.
    raw = entry.decision.token_features[entry.token_idx]
    if len(raw) >= len(OPTION_FEATURE_NAMES):
        return np.asarray(raw[: len(OPTION_FEATURE_NAMES)], dtype=np.float32)
    raise ValueError(f"entry token has too few features: {len(raw)}")


def _deadline(decision_time: datetime) -> datetime:
    max_hold = decision_time + timedelta(minutes=_POLICY.max_hold_minutes)
    local = decision_time.astimezone(_NY)
    forced = datetime.combine(local.date(), datetime.strptime("15:55", "%H:%M").time(), tzinfo=_NY).astimezone(decision_time.tzinfo)
    return min(max_hold, forced)


def _contract_path(entry: EntryProposal, session_rows: Sequence[dict]) -> list[PathPoint]:
    deadline = _deadline(entry.decision_time)
    contract_id = entry.contract_id
    entry_features = _option_entry_features(entry)
    entry_ask = float(entry_features[_FEATURE_INDEX["ask"]])
    if not np.isfinite(entry_ask) or entry_ask <= 0:
        return []
    points: list[PathPoint] = []
    for row in session_rows:
        ts = row["decision_time"]
        if ts <= entry.decision_time or ts > deadline:
            continue
        ids = np.asarray(row["contract_ids"], dtype=object).reshape(-1)
        matches = np.where(ids == contract_id)[0]
        if len(matches) == 0:
            continue
        token_idx = int(matches[0])
        features = np.asarray(row["option_ladder"], dtype=np.float32).reshape(-1, len(OPTION_FEATURE_NAMES))[token_idx]
        bid = float(features[_FEATURE_INDEX["bid"]])
        if not np.isfinite(bid):
            continue
        points.append(
            PathPoint(
                time=ts,
                features=features,
                pnl=(bid - entry_ask) * _CONTRACT_MULTIPLIER,
            )
        )
    return points


def _truncate_at_stop(entry: EntryProposal, path: Sequence[PathPoint]) -> list[PathPoint]:
    entry_features = _option_entry_features(entry)
    entry_ask = float(entry_features[_FEATURE_INDEX["ask"]])
    stop_pnl = -_POLICY.stop_loss_pct * entry_ask * _CONTRACT_MULTIPLIER
    out: list[PathPoint] = []
    for point in path:
        out.append(point)
        if point.pnl <= stop_pnl:
            break
    return out


def _causal_state_features(entry: EntryProposal, path: Sequence[PathPoint], idx: int) -> np.ndarray:
    entry_features = _option_entry_features(entry)
    now = path[idx].features
    pnls = np.asarray([point.pnl for point in path[: idx + 1]], dtype=np.float32)
    current_pnl = float(pnls[-1])
    mfe = float(np.max(pnls))
    mae = float(np.min(pnls))
    denom = max(float(entry_features[_FEATURE_INDEX["ask"]]) * _CONTRACT_MULTIPLIER, 1.0)
    hold_minutes = max(1.0, (path[idx].time - entry.decision_time).total_seconds() / 60.0)
    remaining_minutes = max(0.0, (_deadline(entry.decision_time) - path[idx].time).total_seconds() / 60.0)
    local = path[idx].time.astimezone(_NY)
    forced_local = datetime.combine(local.date(), datetime.strptime("15:55", "%H:%M").time(), tzinfo=_NY)
    minutes_to_forced = max(0.0, (forced_local - local).total_seconds() / 60.0)
    lookback = min(5, idx)
    velocity = 0.0 if lookback == 0 else float((pnls[-1] - pnls[-1 - lookback]) / lookback)
    vol_window = pnls[max(0, idx - 9) : idx + 1]
    realized_vol = float(np.std(vol_window, ddof=0)) if len(vol_window) >= 2 else 0.0
    mfe_idx = int(np.argmax(pnls))
    mfe_age = idx - mfe_idx
    mfe_decay = 0.0 if mfe_age <= 0 else float((current_pnl - mfe) / mfe_age)

    bid = float(now[_FEATURE_INDEX["bid"]])
    ask = float(now[_FEATURE_INDEX["ask"]])
    mid = float(now[_FEATURE_INDEX["mid"]])
    entry_ask = float(entry_features[_FEATURE_INDEX["ask"]])
    entry_mid = float(entry_features[_FEATURE_INDEX["mid"]])
    delta = float(now[_FEATURE_INDEX["delta"]])
    gamma = abs(float(now[_FEATURE_INDEX["gamma"]]))
    theta = abs(float(now[_FEATURE_INDEX["theta"]]))
    entry_delta = float(entry_features[_FEATURE_INDEX["delta"]])
    entry_gamma = abs(float(entry_features[_FEATURE_INDEX["gamma"]]))
    entry_theta = abs(float(entry_features[_FEATURE_INDEX["theta"]]))
    theta_over_mid = theta / max(abs(mid), 1e-6)
    gamma_theta = gamma / max(theta, 1e-6)
    pattern_target = 0.0
    value_target = 0.0
    if entry.decision.pattern_targets is not None:
        pattern_target = float(entry.decision.pattern_targets[entry.token_idx])
    if entry.decision.value_targets is not None:
        value_target = float(entry.decision.value_targets[entry.token_idx])

    out = np.asarray(
        [
            hold_minutes / _POLICY.max_hold_minutes,
            remaining_minutes / _POLICY.max_hold_minutes,
            minutes_to_forced / 390.0,
            current_pnl / denom,
            mfe / denom,
            mae / denom,
            (current_pnl - mfe) / denom,
            velocity / denom,
            realized_vol / denom,
            mfe_decay / denom,
            (bid / max(entry_ask, 1e-6)) - 1.0,
            bid / max(entry_ask, 1e-6),
            mid / max(entry_ask, 1e-6),
            float(now[_FEATURE_INDEX["spread_frac"]]),
            float(entry_features[_FEATURE_INDEX["spread_frac"]]),
            abs(delta),
            delta - entry_delta,
            gamma,
            gamma - entry_gamma,
            theta,
            theta - entry_theta,
            gamma_theta,
            theta_over_mid,
            theta_over_mid * remaining_minutes,
            entry.edge / _RISK_TARGET_SCALE,
            pattern_target,
            value_target,
            float(entry.right == "C"),
            float(entry.right == "P"),
            entry.offset / 50.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0)


def _path_samples(entry: EntryProposal, path: Sequence[PathPoint]) -> tuple[np.ndarray, np.ndarray]:
    stopped = _truncate_at_stop(entry, path)
    if not stopped:
        return np.empty((0, len(RISK_FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    pnls = np.asarray([point.pnl for point in stopped], dtype=np.float32)
    x_rows = []
    y_rows = []
    for idx, pnl in enumerate(pnls):
        future_best = float(np.max(pnls[idx:]))
        headroom = max(0.0, future_best - float(pnl))
        x_rows.append(_causal_state_features(entry, stopped, idx))
        y_rows.append(min(headroom, 600.0) / _RISK_TARGET_SCALE)
    return np.vstack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.float32)


class RiskHeadroomModel(nn.Module):
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


def _fit_risk_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[RiskHeadroomModel, FeatureScaler, list[dict]]:
    if len(train_x) == 0:
        raise ValueError("cannot train risk model with zero samples")
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = FeatureScaler.fit(train_x.astype(np.float32))
    x_train = scaler.transform(train_x.astype(np.float32))
    x_val = scaler.transform(val_x.astype(np.float32)) if len(val_x) else x_train
    y_val = val_y.astype(np.float32) if len(val_y) else train_y.astype(np.float32)
    model = RiskHeadroomModel(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train_y.astype(np.float32))),
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
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.huber_loss(model(torch.from_numpy(x_val)), torch.from_numpy(y_val), delta=1.0).cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
    model.load_state_dict(best_state)
    return model, scaler, history


def _predict_headroom(model: RiskHeadroomModel, scaler: FeatureScaler, x: np.ndarray) -> float:
    x_scaled = scaler.transform(x.reshape(1, -1).astype(np.float32))
    model.eval()
    with torch.no_grad():
        return float(model(torch.from_numpy(x_scaled)).cpu().numpy()[0] * _RISK_TARGET_SCALE)


def _simulate_risk_exit(
    entry: EntryProposal,
    path: Sequence[PathPoint],
    *,
    model: RiskHeadroomModel,
    scaler: FeatureScaler,
    config: RiskConfig,
    loop_id: str = DEFAULT_LOOP_ID,
) -> tuple[Trade, dict]:
    stopped = _truncate_at_stop(entry, path)
    if not stopped:
        trade = _trade_from_entry(entry, entry.baseline_pnl, f"{loop_id}:path_missing")
        return trade, {
            "exit_reason": "path_missing",
            "hold_minutes": None,
            "predicted_headroom": None,
            "blocked_model_exits": 0,
        }
    entry_features = _option_entry_features(entry)
    entry_ask = float(entry_features[_FEATURE_INDEX["ask"]])
    stop_pnl = -_POLICY.stop_loss_pct * entry_ask * _CONTRACT_MULTIPLIER
    exit_point = stopped[-1]
    reason = "time_flat"
    predicted_headroom = None
    blocked_model_exits = 0
    constraint_details: dict = {}
    for idx, point in enumerate(stopped):
        hold_minutes = max(1.0, (point.time - entry.decision_time).total_seconds() / 60.0)
        if point.pnl <= stop_pnl:
            exit_point = point
            reason = "hard_stop"
            break
        if hold_minutes < config.min_hold_minutes:
            continue
        x = _causal_state_features(entry, stopped, idx)
        predicted_headroom = _predict_headroom(model, scaler, x)
        if predicted_headroom <= config.exit_headroom_threshold:
            allowed, constraint_reason, details = _constraint_allows_model_exit(stopped, idx, config)
            constraint_details = details
            if allowed:
                exit_point = point
                reason = constraint_reason
                break
            blocked_model_exits += 1
    trade = _trade_from_entry(entry, exit_point.pnl, f"{loop_id}:{config.name}:{reason}")
    return trade, {
        "exit_reason": reason,
        "hold_minutes": float((exit_point.time - entry.decision_time).total_seconds() / 60.0),
        "predicted_headroom": predicted_headroom,
        "blocked_model_exits": int(blocked_model_exits),
        **constraint_details,
    }


def _metrics(trades: Sequence[Trade]) -> dict:
    out = metrics_with_concentration(trades)
    clean = {}
    for key, value in out.items():
        if isinstance(value, (int, float, np.generic)):
            f = float(value)
            if math.isfinite(f):
                clean[key] = f
            elif f > 0:
                clean[key] = 999.0
            elif f < 0:
                clean[key] = -999.0
            else:
                clean[key] = 0.0
        else:
            clean[key] = value
    return clean


def _summarize_by_split(rows: Sequence[dict], *, split_names: Sequence[str]) -> list[dict]:
    out = []
    for split in split_names:
        group = [row for row in rows if row["split"] == split]
        seed_metrics = [row["metrics"] for row in group]
        stress50 = [row["stress50_metrics"] for row in group]
        stress100 = [row["stress100_metrics"] for row in group]
        if not seed_metrics:
            continue
        out.append(
            {
                "split": split,
                "seed_pnl_median": float(np.median([m["total_pnl"] for m in seed_metrics])),
                "seed_pf_median": float(np.median([m["profit_factor"] for m in seed_metrics])),
                "seed_trades_median": float(np.median([m["trades"] for m in seed_metrics])),
                "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0 for m in seed_metrics])),
                "seed_stress50_pnl_median": float(np.median([m["total_pnl"] for m in stress50])),
                "seed_stress100_pnl_median": float(np.median([m["total_pnl"] for m in stress100])),
            }
        )
    return out


def _selection_reward(metrics: dict) -> float:
    trades = float(metrics["trades"])
    if trades < 8:
        return -1_000_000.0 + trades
    pf = float(metrics["profit_factor"])
    if not np.isfinite(pf):
        pf = 5.0
    return float(metrics["total_pnl"]) + 800.0 * (min(pf, 5.0) - 1.0) + 0.10 * float(metrics["max_drawdown"])


def _write_report_md(path: Path, payload: dict) -> None:
    lines = [
        f"# {payload['report_title']}",
        "",
        payload["framing"],
        "",
        "## Frozen Entry Baseline",
        "",
        f"- Variant: `{payload['baseline']['variant_name']}`",
        f"- Policy: `policy{payload['baseline']['policy_index']}` / `{payload['baseline']['policy_name']}`",
        f"- Trial: `{payload['baseline']['trial_name']}`",
        f"- Selected risk config: `{payload['selected_risk_config']['name']}`",
        "",
        "## Split Summary",
        "",
        "| Split | Dynamic PnL | Dynamic PF | Trades | Positive Seeds | +50 PnL | +100 PnL | Baseline PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_lookup = {row["split"]: row for row in payload["baseline_summary_by_split"]}
    for row in payload["dynamic_summary_by_split"]:
        base = baseline_lookup.get(row["split"], {})
        lines.append(
            f"| {row['split']} | {row['seed_pnl_median']:.0f} | {row['seed_pf_median']:.3f} | "
            f"{row['seed_trades_median']:.0f} | {row['positive_seed_fraction']:.2f} | "
            f"{row['seed_stress50_pnl_median']:.0f} | {row['seed_stress100_pnl_median']:.0f} | "
            f"{float(base.get('seed_pnl_median', 0.0)):.0f} |"
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


def main() -> int:
    args = parse_args()
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    policy_name, cooldown = POLICY_META[args.policy_index]
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    market_cache = MarketStructureCache(
        source=args.market_structure_source,
        index_spx_dir=args.market_spx_dir,
        index_vix_dir=args.market_vix_dir,
        es_vwap_dir=args.es_vwap_dir,
    )

    train_paths = _paths_by_split(args.data_dir)
    seed_q4_dir = args.seed_q4_data_dir or args.q4_data_dir
    seed_q4_paths = _paths(seed_q4_dir)
    window = _protocol_window(train_paths, seed_q4_paths)
    split_paths: dict[str, Sequence[Path]] = {
        "train": train_paths["train"],
        "validation": train_paths["validation"],
        "march_2026": train_paths["test"],
        "q1_2025": _paths(args.q1_data_dir),
        "q2_2025": _paths(args.q2_data_dir),
        "q3_2025": _paths(args.q3_data_dir),
        "q4_2025": _paths(args.q4_data_dir),
    }
    session_rows = _load_session_rows(split_paths)

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
        "train": train_decisions,
        "calibration": calibration_decisions,
        "selection": selection_decisions,
        "march_2026": _load_surface_decisions_cached(
            train_paths["test"],
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="march",
            cache_dir=decision_cache_dir,
        ),
    }
    for split, data_dir in (
        ("q1_2025", args.q1_data_dir),
        ("q2_2025", args.q2_data_dir),
        ("q3_2025", args.q3_data_dir),
        ("q4_2025", args.q4_data_dir),
    ):
        decision_sets[split] = _load_surface_decisions_cached(
            _paths(data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split=split,
            cache_dir=decision_cache_dir,
        )

    risk_grid = _risk_grid(args.exit_constraint)
    seed_artifacts = []
    config_scores: dict[str, list[float]] = {config.name: [] for config in risk_grid}
    config_by_name = {config.name: config for config in risk_grid}
    for seed in args.seeds:
        effective_seed = window_seed(seed, window.window_id)
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        print(f"{args.loop_id} baseline seed={seed}", flush=True)
        entry_model, entry_standardizer, entry_history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=variant,
        )
        proposals_by_split: dict[str, list[EntryProposal]] = {}
        for split, decisions in decision_sets.items():
            predictions = predict_surface_actions(
                entry_model,
                entry_standardizer,
                decisions,
                target_scale=config.target_scale,
            )
            proposals_by_split[split] = _select_entry_proposals(
                decisions,
                predictions,
                trial=trial,
                cooldown_minutes=cooldown,
                split=split,
                seed=seed,
                effective_seed=effective_seed,
            )

        path_cache: dict[tuple[str, str, object], list[PathPoint]] = {}

        def path_for(entry: EntryProposal) -> list[PathPoint]:
            key = (entry.session, entry.decision_time.isoformat(), entry.contract_id)
            if key not in path_cache:
                path_cache[key] = _contract_path(entry, session_rows.get(entry.session, []))
            return path_cache[key]

        def samples_for(entries: Sequence[EntryProposal]) -> tuple[np.ndarray, np.ndarray]:
            xs = []
            ys = []
            for entry in entries:
                x, y = _path_samples(entry, path_for(entry))
                if len(x):
                    xs.append(x)
                    ys.append(y)
            if not xs:
                return np.empty((0, len(RISK_FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.float32)
            return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)

        risk_train_entries = list(proposals_by_split["train"])
        risk_train_entries.extend(
            _training_candidate_proposals(
                train_decisions,
                trial=trial,
                split="train_teacher_candidates",
                seed=seed,
                effective_seed=effective_seed,
            )
        )
        risk_calibration_entries = list(proposals_by_split["calibration"])
        risk_calibration_entries.extend(
            _training_candidate_proposals(
                calibration_decisions,
                trial=trial,
                split="calibration_teacher_candidates",
                seed=seed,
                effective_seed=effective_seed,
            )
        )

        train_x, train_y = samples_for(risk_train_entries)
        val_x, val_y = samples_for(risk_calibration_entries)
        if len(train_x) == 0:
            raise SystemExit(f"seed {seed}: no risk training samples")
        risk_model, risk_scaler, risk_history = _fit_risk_model(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=effective_seed,
            epochs=args.risk_epochs,
            batch_size=args.batch_size,
        )

        selection_entries = proposals_by_split["selection"]
        per_config_selection = {}
        for risk_config in risk_grid:
            trades = [
                _simulate_risk_exit(
                    entry,
                    path_for(entry),
                    model=risk_model,
                    scaler=risk_scaler,
                    config=risk_config,
                    loop_id=args.loop_id,
                )[0]
                for entry in selection_entries
            ]
            metrics = _metrics(trades)
            per_config_selection[risk_config.name] = metrics
            config_scores[risk_config.name].append(_selection_reward(metrics))
        seed_artifacts.append(
            {
                "seed": seed,
                "effective_seed": effective_seed,
                "entry_history": entry_history,
                "risk_history": risk_history,
                "risk_train_samples": int(len(train_x)),
                "risk_validation_samples": int(len(val_x)),
                "risk_train_entries": int(len(risk_train_entries)),
                "risk_validation_entries": int(len(risk_calibration_entries)),
                "proposals_by_split": {split: entries for split, entries in proposals_by_split.items()},
                "path_for": path_for,
                "risk_model": risk_model,
                "risk_scaler": risk_scaler,
                "per_config_selection": per_config_selection,
            }
        )

    selected_config_name = max(
        config_scores,
        key=lambda name: float(np.median(config_scores[name])) if config_scores[name] else -1e9,
    )
    selected_config = config_by_name[selected_config_name]
    print(f"{args.loop_id} selected risk config {selected_config.name}", flush=True)

    split_order = ["selection", "march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025"]
    dynamic_rows = []
    baseline_rows = []
    exit_reason_rows = []
    selected_trade_rows = []
    for artifact in seed_artifacts:
        seed = int(artifact["seed"])
        risk_model = artifact["risk_model"]
        risk_scaler = artifact["risk_scaler"]
        path_for = artifact["path_for"]
        for split in split_order:
            entries = artifact["proposals_by_split"][split]
            dynamic_trades = []
            baseline_trades = []
            for entry in entries:
                trade, info = _simulate_risk_exit(
                    entry,
                    path_for(entry),
                    model=risk_model,
                    scaler=risk_scaler,
                    config=selected_config,
                    loop_id=args.loop_id,
                )
                dynamic_trades.append(trade)
                baseline_trades.append(_trade_from_entry(entry, entry.baseline_pnl, f"{args.loop_id}:baseline_entry"))
                exit_reason_rows.append({"split": split, "seed": seed, **info})
                selected_trade_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "session": entry.session,
                        "decision_time": entry.decision_time.isoformat(),
                        "contract_id": str(entry.contract_id),
                        "right": entry.right,
                        "offset": entry.offset,
                        "edge": entry.edge,
                        "baseline_pnl": entry.baseline_pnl,
                        "dynamic_pnl": trade.pnl,
                        **info,
                    }
                )
            dynamic_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "metrics": _metrics(dynamic_trades),
                    "stress50_metrics": _metrics(stress_trades(dynamic_trades, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _metrics(stress_trades(dynamic_trades, extra_cost_per_trade=100.0)),
                }
            )
            baseline_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "metrics": _metrics(baseline_trades),
                    "stress50_metrics": _metrics(stress_trades(baseline_trades, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _metrics(stress_trades(baseline_trades, extra_cost_per_trade=100.0)),
                }
            )

    dynamic_summary = _summarize_by_split(dynamic_rows, split_names=split_order)
    baseline_summary = _summarize_by_split(baseline_rows, split_names=split_order)
    dynamic_lookup = {row["split"]: row for row in dynamic_summary}
    protocol_name = args.report_title.split(":", 1)[0]
    decision = f"Reject {protocol_name} as a replacement for the frozen entry baseline."
    if all(dynamic_lookup.get(split, {}).get("seed_pnl_median", -1.0) > 0 for split in ("march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025")) and all(dynamic_lookup.get(split, {}).get("seed_stress50_pnl_median", -1.0) > 0 for split in ("march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025")):
        decision = f"Keep {protocol_name} as a risk-management candidate; it survives the broad split and +50 stress gate."

    exit_reason_summary = {}
    for row in exit_reason_rows:
        key = (row["split"], row["exit_reason"])
        exit_reason_summary[key] = exit_reason_summary.get(key, 0) + 1
    exit_reason_summary_rows = [
        {"split": split, "exit_reason": reason, "count": count}
        for (split, reason), count in sorted(exit_reason_summary.items())
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.out_dir / "selected_trades_with_dynamic_exits.json"
    selected_path.write_text(json.dumps(selected_trade_rows, indent=2, allow_nan=False) + "\n")

    payload = {
        "loop_id": args.loop_id,
        "report_title": args.report_title,
        "framing": _risk_framing(args.exit_constraint),
        "pre_registration": [
            "No paid data downloads.",
            "Do not change the entry variant, policy, trial, seeds, labels, or holdout splits.",
            "Train the risk model only on train/calibration entry paths.",
            "Select one exit-headroom/min-hold config on February selection only.",
            f"Exit constraint: {args.exit_constraint}.",
            "Score March 2026 and Q1/Q2/Q3/Q4 2025 once with the selected config.",
        ],
        "baseline": {
            "variant_name": variant.name,
            "variant": asdict(variant) | {"variant_id": variant.variant_id},
            "policy_index": args.policy_index,
            "policy_name": policy_name,
            "trial_name": trial.name,
            "trial": asdict(trial) | {"config_id": trial.config_id},
            "seeds": args.seeds,
        },
        "window": asdict(window) | {"window_id": window.window_id},
        "risk_feature_names": list(RISK_FEATURE_NAMES),
        "risk_grid": [asdict(config) for config in risk_grid],
        "selected_risk_config": asdict(selected_config),
        "config_selection_scores": {
            name: {
                "seed_rewards": scores,
                "median_reward": float(np.median(scores)) if scores else None,
            }
            for name, scores in sorted(config_scores.items())
        },
        "risk_sample_counts": [
            {
                "seed": int(artifact["seed"]),
                "train_entries": int(artifact["risk_train_entries"]),
                "validation_entries": int(artifact["risk_validation_entries"]),
                "train_samples": int(artifact["risk_train_samples"]),
                "validation_samples": int(artifact["risk_validation_samples"]),
            }
            for artifact in seed_artifacts
        ],
        "dynamic_summary_by_split": dynamic_summary,
        "baseline_summary_by_split": baseline_summary,
        "dynamic_seed_rows": dynamic_rows,
        "baseline_seed_rows": baseline_rows,
        "exit_reason_summary": exit_reason_summary_rows,
        "selected_trades_file": str(selected_path),
        "decision": decision,
    }
    report_json = args.out_dir / "report.json"
    report_md = args.out_dir / "report.md"
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report_md(report_md, payload)
    print(report_json)
    print(report_md)
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
