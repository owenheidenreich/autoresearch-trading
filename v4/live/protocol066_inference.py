"""Inference helpers for the frozen Protocol 066 lifecycle model.

Protocol 066 is a post-entry sequence model. A live caller must feed it the
full causal feature path observed since entry, not a single isolated row.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol061_sequence_lifecycle_model import LifecycleSequenceModel


# Keep this in sync with the training simulator. It avoids action flips when a
# prediction sits exactly on a validation-selected threshold after serialization.
OVERRIDE_THRESHOLD_EPSILON = 1e-4


@dataclass(frozen=True)
class Protocol066Prediction:
    step_index: int
    predicted_continuation_value: float
    predicted_recovery_probability: float
    predicted_decay_probability: float
    override_threshold: float
    action: str
    reason: str


@dataclass
class Protocol066Artifact:
    manifest_path: Path
    model_path: Path
    scaler_path: Path
    fold: str
    seed: int
    feature_columns: list[str]
    selected_override_threshold: float
    target_scale: float
    model: LifecycleSequenceModel
    scaler: FeatureScaler

    @property
    def artifact_ref(self) -> str:
        return str(self.model_path)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _scaler_from_dict(payload: dict[str, Any]) -> FeatureScaler:
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def _threshold(value: Any) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    out = float(value)
    return out if math.isfinite(out) else float("inf")


def load_protocol066_artifact(manifest_path: str | Path) -> Protocol066Artifact:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    model_path = Path(manifest["files"]["model"])
    scaler_path = Path(manifest["files"]["scaler"])
    checkpoint = _torch_load(model_path)
    feature_columns = list(checkpoint.get("feature_columns") or manifest["feature_columns"])
    model = LifecycleSequenceModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaler = _scaler_from_dict(json.loads(scaler_path.read_text()))
    return Protocol066Artifact(
        manifest_path=manifest_path,
        model_path=model_path,
        scaler_path=scaler_path,
        fold=str(manifest["fold"]),
        seed=int(manifest["seed"]),
        feature_columns=feature_columns,
        selected_override_threshold=_threshold(manifest["selected_override_threshold"]),
        target_scale=float(checkpoint.get("target_scale", manifest.get("target_scale", 100.0))),
        model=model,
        scaler=scaler,
    )


def predict_protocol066_sequence(
    artifact: Protocol066Artifact,
    feature_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict value/recovery/decay for a causal feature sequence."""

    missing = [column for column in artifact.feature_columns if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"missing Protocol 066 feature columns: {missing}")
    raw = feature_frame[artifact.feature_columns].to_numpy(dtype=np.float32)
    scaled = artifact.scaler.transform(raw)[None, :, :]
    with torch.no_grad():
        value, recovery, decay = artifact.model(torch.from_numpy(scaled))
    return (
        value.squeeze(0).cpu().numpy() * artifact.target_scale,
        torch.sigmoid(recovery).squeeze(0).cpu().numpy(),
        torch.sigmoid(decay).squeeze(0).cpu().numpy(),
    )


def protocol066_action(
    *,
    predicted_continuation_value: float,
    override_threshold: float,
    is_baseline_exit_step: bool = False,
    baseline_exit_reason: str | None = None,
    minutes_to_forced_flat: float | None = None,
) -> tuple[str, str]:
    """Map model output and hard lifecycle constraints into a no-order action."""

    baseline_reason = str(baseline_exit_reason or "")
    if is_baseline_exit_step and baseline_reason == "hard_stop":
        return "stop", "mandatory_hard_stop"
    if is_baseline_exit_step and baseline_reason == "target":
        return "exit", "mandatory_target"
    if is_baseline_exit_step and baseline_reason == "time_flat":
        return "forced_flat", "mandatory_time_flat"
    if is_baseline_exit_step and baseline_reason:
        return "exit", "protocol054_fallback"
    if minutes_to_forced_flat is not None and float(minutes_to_forced_flat) <= 0.0:
        return "forced_flat", "forced_flat_deadline"
    if math.isfinite(override_threshold) and predicted_continuation_value > override_threshold + OVERRIDE_THRESHOLD_EPSILON:
        return "exit", "sequence_residual_override"
    return "hold", "protocol054_fallback_hold"


def prediction_for_step(
    *,
    step_index: int,
    value: np.ndarray,
    recovery: np.ndarray,
    decay: np.ndarray,
    override_threshold: float,
    step_row: pd.Series | None = None,
) -> Protocol066Prediction:
    row = step_row if step_row is not None else pd.Series(dtype=object)
    action, reason = protocol066_action(
        predicted_continuation_value=float(value[step_index]),
        override_threshold=override_threshold,
        is_baseline_exit_step=bool(row.get("is_baseline_exit_step", False)),
        baseline_exit_reason=str(row.get("baseline_exit_reason", "")),
        minutes_to_forced_flat=float(row.get("minutes_to_forced_flat", 999.0)),
    )
    return Protocol066Prediction(
        step_index=int(step_index),
        predicted_continuation_value=float(value[step_index]),
        predicted_recovery_probability=float(recovery[step_index]),
        predicted_decay_probability=float(decay[step_index]),
        override_threshold=float(override_threshold) if math.isfinite(override_threshold) else float("inf"),
        action=action,
        reason=reason,
    )
