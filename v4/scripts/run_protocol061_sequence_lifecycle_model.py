"""Protocol 061: first sequence lifecycle model on Protocol 060.

This is the first architectural move after stopping small exit-rule tweaks.
It trains a causal GRU over each post-entry same-contract path and evaluates a
frozen exit rule against Protocol 054:

    exit when predicted risk-adjusted continuation value is <= 0

Mandatory hard stop, target, and flat-before-close behavior remain enforced via
the frozen baseline path labels. Entries are unchanged. No paid data is
downloaded.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.model.hypothesis_protocol import stress_trades
from v4.model.supervised_pilot import FeatureScaler, Trade
from v4.scripts.build_lifecycle_sequence_dataset import CAUSAL_STEP_FEATURE_COLUMNS
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration


LOOP_ID = "v4_aplus_hypothesis_061_sequence_lifecycle_model_screen"
SPLIT_ORDER = ("q1_2025", "q2_2025", "q3_2025", "q4_2025", "q1_2026")
FOLD_SPECS = (
    {"fold": "train_q2_test_q3", "train_splits": ("q2_2025",), "validation_source": "q2_2025", "test_split": "q3_2025"},
    {
        "fold": "train_q2_q3_test_q4",
        "train_splits": ("q2_2025", "q3_2025"),
        "validation_source": "q3_2025",
        "test_split": "q4_2025",
    },
    {
        "fold": "train_q2_q3_q4_test_q1",
        "train_splits": ("q2_2025", "q3_2025", "q4_2025"),
        "validation_source": "q4_2025",
        "test_split": "q1_2026",
    },
)
TARGET_SCALE = 100.0
TARGET_CLIP = 600.0
RISK_PENALTY = 0.75
# Prevent live/replay drift when a prediction is numerically equal to a
# validation-selected threshold after model serialization.
OVERRIDE_THRESHOLD_EPSILON = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_060_lifecycle_sequence_dataset"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path(f"v4/audit/autoresearch/{LOOP_ID}"))
    parser.add_argument("--loop-id", default=LOOP_ID)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument(
        "--sequence-mode",
        choices=(
            "risk_adjusted_continuation",
            "protocol054_residual_override",
            "protocol054_residual_recovery_penalty",
        ),
        default="risk_adjusted_continuation",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=72)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-train-trades", type=int, default=0)
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument(
        "--save-model-artifacts",
        action="store_true",
        help="Persist per-fold/per-seed model, scaler, threshold, and manifest artifacts.",
    )
    return parser.parse_args()


@dataclass
class SequenceBundle:
    trade_uids: list[str]
    features: np.ndarray
    target_value: np.ndarray
    target_weight: np.ndarray
    target_recovery: np.ndarray
    target_decay: np.ndarray
    mask: np.ndarray
    step_frames: list[pd.DataFrame]
    trade_frame: pd.DataFrame


class LifecycleSequenceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.recovery_head = nn.Linear(hidden_dim, 1)
        self.decay_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.input(x)
        h, _ = self.gru(z)
        value = self.value_head(h).squeeze(-1)
        recovery = self.recovery_head(h).squeeze(-1)
        decay = self.decay_head(h).squeeze(-1)
        return value, recovery, decay


def _load_tables(sequence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = sequence_dir / "protocol054_lifecycle_trades.parquet"
    steps_path = sequence_dir / "protocol054_lifecycle_steps.parquet"
    if not trades_path.exists() or not steps_path.exists():
        raise SystemExit(f"missing Protocol 060 parquet files under {sequence_dir}")
    trades = pd.read_parquet(trades_path)
    steps = pd.read_parquet(steps_path)
    trades["session"] = trades["session"].astype(str)
    steps["session"] = steps["session"].astype(str)
    return trades, steps


def _fold_specs_for(trades: pd.DataFrame) -> tuple[dict, ...]:
    available = {
        str(split)
        for split in trades.loc[trades["path_status"] == "ok", "split"].dropna().unique().tolist()
    }
    ordered = [split for split in SPLIT_ORDER if split in available]
    if len(ordered) < 2:
        raise SystemExit(f"need at least two chronological splits with ok paths, found {ordered}")
    specs = []
    for idx in range(1, len(ordered)):
        train_splits = tuple(ordered[:idx])
        validation_source = ordered[idx - 1]
        test_split = ordered[idx]
        specs.append(
            {
                "fold": f"train_{'_'.join(train_splits)}_test_{test_split}",
                "train_splits": train_splits,
                "validation_source": validation_source,
                "test_split": test_split,
            }
        )
    return tuple(specs)


def _validation_sessions(trades: pd.DataFrame, source_split: str, validation_days: int) -> set[str]:
    sessions = sorted(trades.loc[(trades["split"] == source_split) & (trades["path_status"] == "ok"), "session"].unique())
    if validation_days <= 0:
        return set()
    return set(sessions[-validation_days:])


def _trade_uids_for(
    trades: pd.DataFrame,
    *,
    splits: Sequence[str],
    exclude_sessions: set[str] | None = None,
    include_sessions: set[str] | None = None,
    max_trades: int = 0,
) -> list[str]:
    frame = trades[(trades["path_status"] == "ok") & (trades["split"].isin(splits))].copy()
    if exclude_sessions:
        frame = frame[~frame["session"].isin(exclude_sessions)]
    if include_sessions is not None:
        frame = frame[frame["session"].isin(include_sessions)]
    frame = frame.sort_values(["split", "session", "seed", "decision_time", "trade_uid"])
    if max_trades > 0:
        frame = frame.head(max_trades)
    return frame["trade_uid"].astype(str).tolist()


def _risk_adjusted_target(step_frame: pd.DataFrame) -> np.ndarray:
    future_upside = pd.to_numeric(step_frame["future_max_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    future_downside = np.maximum(
        0.0,
        -pd.to_numeric(step_frame["future_min_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
    )
    value = future_upside - RISK_PENALTY * future_downside
    return (np.clip(value, -TARGET_CLIP, TARGET_CLIP) / TARGET_SCALE).astype(np.float32)


def _protocol054_residual_target(step_frame: pd.DataFrame) -> np.ndarray:
    current = pd.to_numeric(step_frame["current_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    protocol054 = pd.to_numeric(step_frame["protocol054_recorded_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    value = current - protocol054
    return (np.clip(value, -TARGET_CLIP, TARGET_CLIP) / TARGET_SCALE).astype(np.float32)


def _is_residual_mode(sequence_mode: str) -> bool:
    return sequence_mode in {"protocol054_residual_override", "protocol054_residual_recovery_penalty"}


def _target_for_mode(step_frame: pd.DataFrame, sequence_mode: str) -> np.ndarray:
    if _is_residual_mode(sequence_mode):
        return _protocol054_residual_target(step_frame)
    return _risk_adjusted_target(step_frame)


def _target_weight_for_mode(step_frame: pd.DataFrame, sequence_mode: str) -> np.ndarray:
    if sequence_mode != "protocol054_residual_recovery_penalty":
        return np.ones(len(step_frame), dtype=np.float32)
    residual = _protocol054_residual_target(step_frame) * TARGET_SCALE
    future_max_delta = pd.to_numeric(step_frame["future_max_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    exit_regret = pd.to_numeric(step_frame["exit_now_regret_to_baseline"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    minutes_since_entry = pd.to_numeric(step_frame["minutes_since_entry"], errors="coerce").fillna(99.0).to_numpy(dtype=np.float32)
    negative_residual = (residual < 0.0).astype(np.float32)

    recovery_strength = 1.0 / (1.0 + np.exp(-(future_max_delta - 250.0) / 200.0))
    regret_strength = 1.0 / (1.0 + np.exp(-(exit_regret - 100.0) / 200.0))
    early_strength = 1.0 / (1.0 + np.exp((minutes_since_entry - 5.0) / 2.0))
    penalty = negative_residual * recovery_strength * regret_strength * early_strength
    return (1.0 + 4.0 * penalty).astype(np.float32)


def _bundle_for(
    trades: pd.DataFrame,
    steps: pd.DataFrame,
    trade_uids: Sequence[str],
    *,
    scaler: FeatureScaler | None = None,
    sequence_mode: str = "risk_adjusted_continuation",
) -> tuple[SequenceBundle, FeatureScaler]:
    if not trade_uids:
        raise SystemExit("cannot build sequence bundle with zero trades")
    uid_set = set(map(str, trade_uids))
    step_subset = steps[steps["trade_uid"].astype(str).isin(uid_set)].copy()
    trade_subset = trades[trades["trade_uid"].astype(str).isin(uid_set)].copy()
    grouped = {uid: frame.sort_values("step_idx").reset_index(drop=True) for uid, frame in step_subset.groupby("trade_uid", sort=False)}
    ordered_uids = [uid for uid in trade_uids if uid in grouped]
    if not ordered_uids:
        raise SystemExit("no step rows matched requested trade ids")
    max_len = int(max(len(grouped[uid]) for uid in ordered_uids))
    feature_dim = len(CAUSAL_STEP_FEATURE_COLUMNS)
    features = np.zeros((len(ordered_uids), max_len, feature_dim), dtype=np.float32)
    target_value = np.zeros((len(ordered_uids), max_len), dtype=np.float32)
    target_weight = np.ones((len(ordered_uids), max_len), dtype=np.float32)
    target_recovery = np.zeros((len(ordered_uids), max_len), dtype=np.float32)
    target_decay = np.zeros((len(ordered_uids), max_len), dtype=np.float32)
    mask = np.zeros((len(ordered_uids), max_len), dtype=np.float32)
    step_frames: list[pd.DataFrame] = []
    flat_features = []
    for i, uid in enumerate(ordered_uids):
        frame = grouped[uid]
        length = len(frame)
        x = frame[CAUSAL_STEP_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        features[i, :length, :] = x
        target_value[i, :length] = _target_for_mode(frame, sequence_mode)
        target_weight[i, :length] = _target_weight_for_mode(frame, sequence_mode)
        target_recovery[i, :length] = frame["future_recovery_200"].astype(float).to_numpy(dtype=np.float32)
        target_decay[i, :length] = frame["future_decay_200"].astype(float).to_numpy(dtype=np.float32)
        if _is_residual_mode(sequence_mode):
            mask[i, :length] = (~frame["is_after_protocol054_exit"].astype(bool)).astype(float).to_numpy(dtype=np.float32)
        else:
            mask[i, :length] = 1.0
        flat_features.append(x)
        step_frames.append(frame)
    if scaler is None:
        scaler = FeatureScaler.fit(np.vstack(flat_features).astype(np.float32))
    flat = features.reshape(-1, feature_dim)
    features = scaler.transform(flat).reshape(features.shape)
    trade_subset = trade_subset.set_index("trade_uid").loc[ordered_uids].reset_index()
    return (
        SequenceBundle(
            trade_uids=ordered_uids,
            features=features,
            target_value=target_value,
            target_weight=target_weight,
            target_recovery=target_recovery,
            target_decay=target_decay,
            mask=mask,
            step_frames=step_frames,
            trade_frame=trade_subset,
        ),
        scaler,
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _fit_model(
    train: SequenceBundle,
    validation: SequenceBundle,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
) -> tuple[LifecycleSequenceModel, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LifecycleSequenceModel(input_dim=train.features.shape[-1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.features),
            torch.from_numpy(train.target_value),
            torch.from_numpy(train.target_weight),
            torch.from_numpy(train.target_recovery),
            torch.from_numpy(train.target_decay),
            torch.from_numpy(train.mask),
        ),
        batch_size=min(batch_size, len(train.trade_uids)),
        shuffle=True,
    )
    val_tensors = (
        torch.from_numpy(validation.features),
        torch.from_numpy(validation.target_value),
        torch.from_numpy(validation.target_weight),
        torch.from_numpy(validation.target_recovery),
        torch.from_numpy(validation.target_decay),
        torch.from_numpy(validation.mask),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, y_value, y_weight, y_recovery, y_decay, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            pred_value, pred_recovery, pred_decay = model(xb)
            value_loss = _masked_mean(F.huber_loss(pred_value, y_value, reduction="none", delta=1.0) * y_weight, mask)
            recovery_loss = _masked_mean(F.binary_cross_entropy_with_logits(pred_recovery, y_recovery, reduction="none"), mask)
            decay_loss = _masked_mean(F.binary_cross_entropy_with_logits(pred_decay, y_decay, reduction="none"), mask)
            loss = value_loss + 0.20 * recovery_loss + 0.20 * decay_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vx, vy_value, vy_weight, vy_recovery, vy_decay, vmask = val_tensors
            pv, pr, pd_ = model(vx)
            val_value = _masked_mean(F.huber_loss(pv, vy_value, reduction="none", delta=1.0) * vy_weight, vmask)
            val_recovery = _masked_mean(F.binary_cross_entropy_with_logits(pr, vy_recovery, reduction="none"), vmask)
            val_decay = _masked_mean(F.binary_cross_entropy_with_logits(pd_, vy_decay, reduction="none"), vmask)
            val_loss = val_value + 0.20 * val_recovery + 0.20 * val_decay
        if float(val_loss.cpu()) < best_val:
            best_val = float(val_loss.cpu())
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": float(val_loss.cpu()),
                "validation_value_loss": float(val_value.cpu()),
                "validation_recovery_loss": float(val_recovery.cpu()),
                "validation_decay_loss": float(val_decay.cpu()),
            }
        )
    model.load_state_dict(best_state)
    return model, history


def _predict(model: LifecycleSequenceModel, bundle: SequenceBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        value, recovery, decay = model(torch.from_numpy(bundle.features))
    return (
        value.cpu().numpy() * TARGET_SCALE,
        torch.sigmoid(recovery).cpu().numpy(),
        torch.sigmoid(decay).cpu().numpy(),
    )


def _save_model_artifact(
    *,
    model: LifecycleSequenceModel,
    scaler: FeatureScaler,
    history: list[dict],
    threshold: float,
    threshold_sweep: list[dict],
    spec: dict,
    seed: int,
    args: argparse.Namespace,
    train_bundle: SequenceBundle,
    validation_bundle: SequenceBundle,
    test_bundle: SequenceBundle,
    validation_sessions: set[str],
) -> dict:
    artifact_dir = args.out_dir / "model_artifacts" / str(spec["fold"]) / f"seed_{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    scaler_path = artifact_dir / "scaler.json"
    threshold_sweep_path = artifact_dir / "threshold_sweep.json"
    manifest_path = artifact_dir / "manifest.json"
    threshold_value: float | str = float(threshold) if np.isfinite(threshold) else "inf"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(train_bundle.features.shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "target_scale": TARGET_SCALE,
            "target_clip": TARGET_CLIP,
            "risk_penalty": RISK_PENALTY,
            "sequence_mode": args.sequence_mode,
            "feature_columns": list(CAUSAL_STEP_FEATURE_COLUMNS),
        },
        model_path,
    )
    scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2, allow_nan=False) + "\n")
    threshold_sweep_path.write_text(json.dumps(threshold_sweep, indent=2, allow_nan=False) + "\n")
    manifest = {
        "artifact_type": "protocol066_lifecycle_sequence_model",
        "fold": spec["fold"],
        "train_splits": list(spec["train_splits"]),
        "validation_source": spec["validation_source"],
        "test_split": spec["test_split"],
        "seed": int(seed),
        "sequence_mode": args.sequence_mode,
        "calibrate_threshold": bool(args.calibrate_threshold),
        "selected_override_threshold": threshold_value,
        "feature_columns": list(CAUSAL_STEP_FEATURE_COLUMNS),
        "model_class": "v4.scripts.run_protocol061_sequence_lifecycle_model.LifecycleSequenceModel",
        "scaler_class": "v4.model.supervised_pilot.FeatureScaler",
        "target_scale": TARGET_SCALE,
        "target_clip": TARGET_CLIP,
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "validation_days": int(args.validation_days),
        "validation_sessions": sorted(validation_sessions),
        "train_trades": len(train_bundle.trade_uids),
        "validation_trades": len(validation_bundle.trade_uids),
        "test_trades": len(test_bundle.trade_uids),
        "history": history,
        "files": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "threshold_sweep": str(threshold_sweep_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {
        "artifact_dir": str(artifact_dir),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "threshold_sweep_path": str(threshold_sweep_path),
        "manifest_path": str(manifest_path),
    }


def _trade_metrics(trades: Sequence[Trade]) -> dict:
    metrics = metrics_with_concentration(trades)
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.generic)):
            value = float(value)
            if np.isfinite(value):
                out[key] = value
            elif value > 0:
                out[key] = 999.0
            elif value < 0:
                out[key] = -999.0
            else:
                out[key] = 0.0
        else:
            out[key] = value
    return out


def _candidate_thresholds(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    fixed = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, float("inf")]
    if len(finite) == 0:
        return [float("inf")]
    quantiles = np.quantile(finite, [0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.975])
    candidates = fixed + [float(x) for x in quantiles]
    return sorted(set(round(x, 4) if np.isfinite(x) else x for x in candidates))


def _select_residual_threshold(
    bundle: SequenceBundle,
    value_pred: np.ndarray,
    recovery_pred: np.ndarray,
    decay_pred: np.ndarray,
    *,
    loop_id: str,
    seed: int,
    sequence_mode: str,
) -> tuple[float, list[dict]]:
    sweep = []
    best_threshold = float("inf")
    best_key = (-1e18, -1e18, 0.0)
    for threshold in _candidate_thresholds(value_pred):
        candidate_trades, p054_trades, rows = _simulate_sequence_exits(
            bundle,
            value_pred,
            recovery_pred,
            decay_pred,
            loop_id=loop_id,
            seed=seed,
            sequence_mode=sequence_mode,
            override_threshold=threshold,
        )
        metrics = _trade_metrics(candidate_trades)
        p054_metrics = _trade_metrics(p054_trades)
        override_fraction = float(np.mean([row["candidate_exit_reason"] == "sequence_residual_override" for row in rows])) if rows else 0.0
        delta = float(metrics["total_pnl"] - p054_metrics["total_pnl"])
        pf = float(metrics["profit_factor"])
        if not np.isfinite(pf):
            pf = 999.0
        key = (delta, float(metrics["total_pnl"]), -override_fraction)
        sweep.append(
            {
                "threshold": threshold if np.isfinite(threshold) else "inf",
                "metrics": metrics,
                "protocol054_metrics": p054_metrics,
                "delta_vs_protocol054": delta,
                "override_fraction": override_fraction,
                "profit_factor_for_selection": pf,
            }
        )
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold, sweep


def _simulate_sequence_exits(
    bundle: SequenceBundle,
    value_pred: np.ndarray,
    recovery_pred: np.ndarray,
    decay_pred: np.ndarray,
    *,
    loop_id: str,
    seed: int,
    sequence_mode: str,
    override_threshold: float = 0.0,
) -> tuple[list[Trade], list[Trade], list[dict]]:
    candidate_trades: list[Trade] = []
    protocol054_trades: list[Trade] = []
    selected_rows: list[dict] = []
    for i, (uid, steps) in enumerate(zip(bundle.trade_uids, bundle.step_frames)):
        trade = bundle.trade_frame.iloc[i]
        exit_idx = len(steps) - 1
        reason = "time_flat"
        protocol054_exit_step = int(trade["protocol054_exit_step"])
        if _is_residual_mode(sequence_mode):
            exit_idx = protocol054_exit_step
            reason = "protocol054_fallback"
            for idx, row in steps.iloc[: protocol054_exit_step + 1].iterrows():
                baseline_reason = str(row.get("baseline_exit_reason"))
                if bool(row.get("is_baseline_exit_step")) and baseline_reason in {"hard_stop", "target"}:
                    exit_idx = int(idx)
                    reason = baseline_reason
                    break
                if (
                    int(idx) < protocol054_exit_step
                    and float(value_pred[i, int(idx)]) > override_threshold + OVERRIDE_THRESHOLD_EPSILON
                ):
                    exit_idx = int(idx)
                    reason = "sequence_residual_override"
                    break
        else:
            for idx, row in steps.iterrows():
                baseline_reason = str(row.get("baseline_exit_reason"))
                if bool(row.get("is_baseline_exit_step")) and baseline_reason in {"hard_stop", "target"}:
                    exit_idx = int(idx)
                    reason = baseline_reason
                    break
                if float(value_pred[i, int(idx)]) <= 0.0:
                    exit_idx = int(idx)
                    reason = "sequence_model_exit"
                    break
                if bool(row.get("is_baseline_exit_step")) and baseline_reason == "time_flat":
                    exit_idx = int(idx)
                    reason = "time_flat"
                    break
        exit_row = steps.iloc[exit_idx]
        pnl = float(exit_row["current_pnl"])
        p054_pnl = float(trade["protocol054_pnl"])
        candidate_trades.append(
            Trade(
                session=str(trade["session"]),
                decision_time=str(trade["decision_time"]),
                pnl=pnl,
                score=float(value_pred[i, exit_idx]),
                right=str(trade["right"]),
                offset=float(trade["offset"]),
                strategy=f"{loop_id}:seed{seed}:{reason}",
            )
        )
        protocol054_trades.append(
            Trade(
                session=str(trade["session"]),
                decision_time=str(trade["decision_time"]),
                pnl=p054_pnl,
                score=float(trade.get("protocol054_predicted_headroom", np.nan)),
                right=str(trade["right"]),
                offset=float(trade["offset"]),
                strategy="protocol054_frozen",
            )
        )
        selected_rows.append(
            {
                "trade_uid": uid,
                "canonical_entry_uid": trade["canonical_entry_uid"],
                "split": trade["split"],
                "seed": int(seed),
                "entry_seed": int(trade["seed"]),
                "session": trade["session"],
                "decision_time": trade["decision_time"],
                "contract_id": trade["contract_id"],
                "right": trade["right"],
                "offset": float(trade["offset"]),
                "candidate_pnl": pnl,
                "protocol054_pnl": p054_pnl,
                "delta_vs_protocol054": pnl - p054_pnl,
                "candidate_exit_reason": reason,
                "candidate_exit_step": int(exit_idx),
                "candidate_exit_time": exit_row["quote_time"],
                "protocol054_exit_reason": trade["protocol054_exit_reason"],
                "protocol054_exit_step": protocol054_exit_step,
                "predicted_continuation_value": float(value_pred[i, exit_idx]),
                "override_threshold": float(override_threshold) if np.isfinite(override_threshold) else "inf",
                "predicted_recovery_probability": float(recovery_pred[i, exit_idx]),
                "predicted_decay_probability": float(decay_pred[i, exit_idx]),
                "current_pnl_at_exit": pnl,
                "mfe_to_exit": float(exit_row["mfe_to_now"]),
                "mae_to_exit": float(exit_row["mae_to_now"]),
                "future_max_delta_at_exit": float(exit_row["future_max_delta"]),
                "future_min_delta_at_exit": float(exit_row["future_min_delta"]),
            }
        )
    return candidate_trades, protocol054_trades, selected_rows


def _summary_by_split(rows: Sequence[dict], *, splits: Sequence[str]) -> list[dict]:
    out = []
    for split in splits:
        group = [row for row in rows if row["split"] == split]
        if not group:
            continue
        metrics = [row["metrics"] for row in group]
        stress50 = [row["stress50_metrics"] for row in group]
        stress100 = [row["stress100_metrics"] for row in group]
        out.append(
            {
                "split": split,
                "pnl_median": float(np.median([m["total_pnl"] for m in metrics])),
                "pf_median": float(np.median([m["profit_factor"] for m in metrics])),
                "trades_median": float(np.median([m["trades"] for m in metrics])),
                "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in metrics])),
                "stress50_pnl_median": float(np.median([m["total_pnl"] for m in stress50])),
                "stress100_pnl_median": float(np.median([m["total_pnl"] for m in stress100])),
                "stress50_positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in stress50])),
            }
        )
    return out


def _run_fold(
    *,
    spec: dict,
    trades: pd.DataFrame,
    steps: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    validation_sessions = _validation_sessions(trades, spec["validation_source"], args.validation_days)
    train_uids = _trade_uids_for(
        trades,
        splits=spec["train_splits"],
        exclude_sessions=validation_sessions,
        max_trades=args.max_train_trades,
    )
    validation_uids = _trade_uids_for(trades, splits=(spec["validation_source"],), include_sessions=validation_sessions)
    test_uids = _trade_uids_for(trades, splits=(spec["test_split"],))
    train_bundle, scaler = _bundle_for(trades, steps, train_uids, sequence_mode=args.sequence_mode)
    validation_bundle, _ = _bundle_for(trades, steps, validation_uids, scaler=scaler, sequence_mode=args.sequence_mode)
    test_bundle, _ = _bundle_for(trades, steps, test_uids, scaler=scaler, sequence_mode=args.sequence_mode)
    candidate_rows = []
    p054_rows = []
    selected_rows = []
    seed_artifacts = []
    for seed in args.seeds:
        print(f"{args.loop_id} fold={spec['fold']} seed={seed}", flush=True)
        model, history = _fit_model(
            train_bundle,
            validation_bundle,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
        )
        threshold = 0.0
        threshold_sweep: list[dict] = []
        if _is_residual_mode(args.sequence_mode) and args.calibrate_threshold:
            val_value, val_recovery, val_decay = _predict(model, validation_bundle)
            threshold, threshold_sweep = _select_residual_threshold(
                validation_bundle,
                val_value,
                val_recovery,
                val_decay,
                loop_id=f"{args.loop_id}:validation_threshold",
                seed=seed,
                sequence_mode=args.sequence_mode,
            )
        saved_artifact = None
        if args.save_model_artifacts:
            saved_artifact = _save_model_artifact(
                model=model,
                scaler=scaler,
                history=history,
                threshold=threshold,
                threshold_sweep=threshold_sweep,
                spec=spec,
                seed=seed,
                args=args,
                train_bundle=train_bundle,
                validation_bundle=validation_bundle,
                test_bundle=test_bundle,
                validation_sessions=validation_sessions,
            )
        value_pred, recovery_pred, decay_pred = _predict(model, test_bundle)
        candidate_trades, p054_trades, seed_selected = _simulate_sequence_exits(
            test_bundle,
            value_pred,
            recovery_pred,
            decay_pred,
            loop_id=args.loop_id,
            seed=seed,
            sequence_mode=args.sequence_mode,
            override_threshold=threshold,
        )
        candidate_rows.append(
            {
                "fold": spec["fold"],
                "split": spec["test_split"],
                "seed": seed,
                "metrics": _trade_metrics(candidate_trades),
                "stress50_metrics": _trade_metrics(stress_trades(candidate_trades, extra_cost_per_trade=50.0)),
                "stress100_metrics": _trade_metrics(stress_trades(candidate_trades, extra_cost_per_trade=100.0)),
            }
        )
        p054_rows.append(
            {
                "fold": spec["fold"],
                "split": spec["test_split"],
                "seed": seed,
                "metrics": _trade_metrics(p054_trades),
                "stress50_metrics": _trade_metrics(stress_trades(p054_trades, extra_cost_per_trade=50.0)),
                "stress100_metrics": _trade_metrics(stress_trades(p054_trades, extra_cost_per_trade=100.0)),
            }
        )
        if spec["test_split"] == "q1_2026":
            march_candidate = [trade for trade in candidate_trades if trade.session >= "2026-03-01"]
            march_p054 = [trade for trade in p054_trades if trade.session >= "2026-03-01"]
            candidate_rows.append(
                {
                    "fold": spec["fold"],
                    "split": "march_2026",
                    "seed": seed,
                    "metrics": _trade_metrics(march_candidate),
                    "stress50_metrics": _trade_metrics(stress_trades(march_candidate, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _trade_metrics(stress_trades(march_candidate, extra_cost_per_trade=100.0)),
                }
            )
            p054_rows.append(
                {
                    "fold": spec["fold"],
                    "split": "march_2026",
                    "seed": seed,
                    "metrics": _trade_metrics(march_p054),
                    "stress50_metrics": _trade_metrics(stress_trades(march_p054, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _trade_metrics(stress_trades(march_p054, extra_cost_per_trade=100.0)),
                }
            )
        selected_rows.extend(seed_selected)
        seed_artifacts.append(
            {
                "seed": seed,
                "history": history,
                "train_trades": len(train_bundle.trade_uids),
                "validation_trades": len(validation_bundle.trade_uids),
                "test_trades": len(test_bundle.trade_uids),
                "validation_sessions": sorted(validation_sessions),
                "selected_override_threshold": threshold if np.isfinite(threshold) else "inf",
                "threshold_sweep": threshold_sweep,
                "saved_artifact": saved_artifact,
            }
        )
    return {
        "fold": spec["fold"],
        "train_splits": list(spec["train_splits"]),
        "validation_source": spec["validation_source"],
        "test_split": spec["test_split"],
        "seed_artifacts": seed_artifacts,
        "candidate_rows": candidate_rows,
        "protocol054_rows": p054_rows,
        "selected_trades": selected_rows,
    }


def _write_report(path: Path, payload: dict) -> None:
    p054 = {row["split"]: row for row in payload["protocol054_summary"]}
    lines = [
        f"# {payload['loop_id']}",
        "",
        "No paid data was downloaded. Protocol 051 entries and Protocol 054 baseline remain frozen.",
        "",
        "## Hypothesis",
        "",
        payload["pre_registration"]["hypothesis"],
        "",
        "## Walk-Forward Summary",
        "",
        "| Split | Sequence PnL | Protocol 054 PnL | Delta | PF | Trades | +50 | +100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidate_summary"]:
        split = row["split"]
        p054_pnl = float(p054.get(split, {}).get("pnl_median", 0.0))
        lines.append(
            f"| {split} | {row['pnl_median']:.0f} | {p054_pnl:.0f} | {row['pnl_median'] - p054_pnl:.0f} | "
            f"{row['pf_median']:.3f} | {row['trades_median']:.0f} | "
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


def main() -> int:
    args = parse_args()
    trades, steps = _load_tables(args.sequence_dir)
    all_specs = _fold_specs_for(trades)
    specs = all_specs[: args.max_folds] if args.max_folds > 0 else all_specs
    fold_payloads = [_run_fold(spec=spec, trades=trades, steps=steps, args=args) for spec in specs]
    candidate_rows = [row for payload in fold_payloads for row in payload["candidate_rows"]]
    p054_rows = [row for payload in fold_payloads for row in payload["protocol054_rows"]]
    selected_rows = [row for payload in fold_payloads for row in payload["selected_trades"]]
    scored = [spec["test_split"] for spec in specs]
    splits = [*scored]
    if "q1_2026" in scored:
        splits.append("march_2026")
    candidate_summary = _summary_by_split(candidate_rows, splits=splits)
    p054_summary = _summary_by_split(p054_rows, splits=splits)
    candidate_lookup = {row["split"]: row for row in candidate_summary}
    p054_lookup = {row["split"]: row for row in p054_summary}
    scored = [split for split in scored if split in candidate_lookup]
    beat_folds = sum(
        candidate_lookup[split]["pnl_median"] > p054_lookup.get(split, {}).get("pnl_median", 1e9)
        for split in scored
    )
    stress_positive = all(candidate_lookup[split]["stress50_pnl_median"] > 0.0 for split in scored)
    march_not_worse = (
        "march_2026" not in candidate_lookup
        or candidate_lookup["march_2026"]["pnl_median"] >= p054_lookup.get("march_2026", {}).get("pnl_median", 1e9)
    )
    required_beats = max(2, len(scored) - 1)
    if len(scored) >= 3 and beat_folds >= required_beats and stress_positive and march_not_worse:
        decision = (
            f"Keep {args.loop_id} as a sequence-lifecycle challenger and advance to stricter validation. "
            f"It beats Protocol 054 in at least {required_beats} of {len(scored)} scored folds, keeps +50 stress positive, and does not damage March."
        )
    else:
        decision = (
            f"Reject {args.loop_id} as a replacement for Protocol 054 at this gate. "
            "Use the result diagnostically before changing the sequence objective."
        )
    if args.sequence_mode == "protocol054_residual_recovery_penalty":
        target_description = (
            "protocol054_residual = current_executable_pnl - frozen_protocol054_pnl, clipped to +/-600 "
            "and scaled by 100; value loss is up-weighted for early negative-residual steps where "
            "future recovery and baseline-regret labels show recoverable convexity"
        )
        if args.calibrate_threshold:
            exit_rule = (
                "mandatory hard stop/target first; otherwise exit before Protocol 054 only when "
                "predicted residual value exceeds a threshold selected on validation sessions; "
                "if no override fires, use the frozen Protocol 054 exit"
            )
        else:
            exit_rule = (
                "mandatory hard stop/target first; otherwise exit before Protocol 054 only when "
                "predicted residual value > 0; if no override fires, use the frozen Protocol 054 exit"
            )
        hypothesis = (
            "Protocol 064 showed March damage comes from false early residual overrides on recoverable "
            "convex pullbacks. A causal GRU trained with an asymmetric recovery-aware residual loss should "
            "preserve useful Q3/Q4 early exits while lowering confidence on those false early exits."
        )
    elif args.sequence_mode == "protocol054_residual_override":
        target_description = (
            "protocol054_residual = current_executable_pnl - frozen_protocol054_pnl, "
            "clipped to +/-600 and scaled by 100"
        )
        if args.calibrate_threshold:
            exit_rule = (
                "mandatory hard stop/target first; otherwise exit before Protocol 054 only when "
                "predicted residual value exceeds a threshold selected on validation sessions; "
                "if no override fires, use the frozen Protocol 054 exit"
            )
            hypothesis = (
                "A causal GRU residual override needs validation-calibrated abstention: it should only "
                "override Protocol 054 when the predicted executable advantage is strong enough, with "
                "no-override available as a validation-selected outcome."
            )
        else:
            exit_rule = (
                "mandatory hard stop/target first; otherwise exit before Protocol 054 only when "
                "predicted residual value > 0; if no override fires, use the frozen Protocol 054 exit"
            )
            hypothesis = (
                "A causal GRU can improve Protocol 054 as a residual override by learning when the current "
                "executable bid is better than waiting for the frozen lifecycle exit, while leaving Protocol 054 "
                "unchanged when the model is not confident."
            )
    else:
        target_description = (
            "risk_adjusted_continuation = future_max_delta - 0.75 * max(0, -future_min_delta), "
            "clipped to +/-600 and scaled by 100"
        )
        exit_rule = "mandatory hard stop/target first; otherwise exit when predicted continuation value <= 0"
        hypothesis = (
            "A causal GRU over the post-entry same-contract path can learn temporary pullback versus "
            "true continuation decay better than small hand-written exit-rule tweaks."
        )
    payload = {
        "loop_id": args.loop_id,
        "pre_registration": {
            "paid_data_downloaded": False,
            "sequence_dir": str(args.sequence_dir),
            "sequence_mode": args.sequence_mode,
            "calibrate_threshold": bool(args.calibrate_threshold),
            "entry_protocol_frozen": "Protocol 051",
            "baseline_lifecycle_protocol": "Protocol 054",
            "training_rule": "train only on earlier split sequences; validation uses trailing sessions from the latest training split",
            "target": target_description,
            "exit_rule": exit_rule,
            "hypothesis": hypothesis,
            "acceptance_rule": (
                "Advance only if the candidate beats Protocol 054 in all but at most one chronological scored fold, "
                "keeps +50 stress median positive, and does not damage March when Q1 2026 is present."
            ),
        },
        "args": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "validation_days": args.validation_days,
            "max_train_trades": args.max_train_trades,
            "sequence_mode": args.sequence_mode,
            "calibrate_threshold": bool(args.calibrate_threshold),
            "save_model_artifacts": bool(args.save_model_artifacts),
        },
        "feature_columns": list(CAUSAL_STEP_FEATURE_COLUMNS),
        "fold_results": [
            {key: value for key, value in payload.items() if key != "selected_trades"} for payload in fold_payloads
        ],
        "candidate_summary": candidate_summary,
        "protocol054_summary": p054_summary,
        "selected_trades_file": str(args.out_dir / "selected_trades_sequence_exits.json"),
        "decision": decision,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_trades_sequence_exits.json").write_text(json.dumps(selected_rows, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps(candidate_summary, indent=2, sort_keys=True), flush=True)
    print(args.out_dir / "report.md", flush=True)
    print(decision, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
