from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v4.live.protocol066_inference import (
    load_protocol066_artifact,
    protocol066_action,
    predict_protocol066_sequence,
)
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol061_sequence_lifecycle_model import LifecycleSequenceModel


def _artifact(tmp_path: Path) -> Path:
    feature_columns = ["bid", "ask", "gamma"]
    scaler = FeatureScaler.fit(np.array([[1.0, 1.2, 0.01], [1.1, 1.3, 0.02]], dtype=np.float32))
    model = LifecycleSequenceModel(input_dim=3, hidden_dim=8)
    model_path = tmp_path / "model.pt"
    scaler_path = tmp_path / "scaler.json"
    manifest_path = tmp_path / "manifest.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": 3,
            "hidden_dim": 8,
            "target_scale": 100.0,
            "feature_columns": feature_columns,
        },
        model_path,
    )
    scaler_path.write_text(json.dumps(scaler.to_dict()))
    manifest_path.write_text(
        json.dumps(
            {
                "fold": "test_fold",
                "seed": 1,
                "selected_override_threshold": 25.0,
                "feature_columns": feature_columns,
                "files": {
                    "model": str(model_path),
                    "scaler": str(scaler_path),
                },
            }
        )
    )
    return manifest_path


def test_load_protocol066_artifact_and_predict_sequence(tmp_path: Path) -> None:
    artifact = load_protocol066_artifact(_artifact(tmp_path))
    frame = pd.DataFrame({"bid": [1.0, 1.2], "ask": [1.1, 1.3], "gamma": [0.01, 0.02]})

    value, recovery, decay = predict_protocol066_sequence(artifact, frame)

    assert value.shape == (2,)
    assert recovery.shape == (2,)
    assert decay.shape == (2,)
    assert np.isfinite(value).all()


def test_protocol066_action_enforces_hard_stop_before_model() -> None:
    action, reason = protocol066_action(
        predicted_continuation_value=1000.0,
        override_threshold=25.0,
        is_baseline_exit_step=True,
        baseline_exit_reason="hard_stop",
    )

    assert action == "stop"
    assert reason == "mandatory_hard_stop"


def test_protocol066_action_uses_override_threshold() -> None:
    action, reason = protocol066_action(
        predicted_continuation_value=30.0,
        override_threshold=25.0,
    )

    assert action == "exit"
    assert reason == "sequence_residual_override"


def test_protocol066_action_exits_on_protocol054_fallback_step() -> None:
    action, reason = protocol066_action(
        predicted_continuation_value=1000.0,
        override_threshold=25.0,
        is_baseline_exit_step=True,
        baseline_exit_reason="model_exit_giveback",
    )

    assert action == "exit"
    assert reason == "protocol054_fallback"
