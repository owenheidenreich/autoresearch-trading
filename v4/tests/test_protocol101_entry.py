from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v4.dataset.spxw_0dte_neural import OPTION_FEATURE_NAMES
from v4.live.protocol101_entry import (
    Protocol101HistoryState,
    load_protocol101_entry_artifact,
    predict_protocol101_entry,
    protocol101_candidate_gate_diagnostics,
    protocol101_candidate_frame_from_surface,
)
from v4.model.hypothesis_protocol import SurfaceDecision
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol097_sequential_event_policy import EventSetPolicy
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS


def _decision() -> SurfaceDecision:
    tokens = np.zeros((3, len(OPTION_FEATURE_NAMES)), dtype=np.float32)
    for i in range(3):
        tokens[i, OPTION_FEATURE_NAMES.index("bid")] = 1.0 + i
        tokens[i, OPTION_FEATURE_NAMES.index("ask")] = 1.2 + i
        tokens[i, OPTION_FEATURE_NAMES.index("mid")] = 1.1 + i
        tokens[i, OPTION_FEATURE_NAMES.index("spread")] = 0.2
        tokens[i, OPTION_FEATURE_NAMES.index("spread_frac")] = 0.05
        tokens[i, OPTION_FEATURE_NAMES.index("bid_size")] = 10
        tokens[i, OPTION_FEATURE_NAMES.index("ask_size")] = 12
        tokens[i, OPTION_FEATURE_NAMES.index("iv")] = 0.20
        tokens[i, OPTION_FEATURE_NAMES.index("delta")] = 0.35 if i != 1 else -0.35
        tokens[i, OPTION_FEATURE_NAMES.index("gamma")] = 0.012 + i * 0.001
        tokens[i, OPTION_FEATURE_NAMES.index("theta")] = -0.25
    return SurfaceDecision(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc),
        scalar_features=np.zeros(2, dtype=np.float32),
        token_features=tokens,
        token_mask=np.asarray([True, True, False]),
        labels=np.asarray([1.0, 2.0, np.nan], dtype=np.float32),
        offsets=np.asarray([0.0, 5.0, 10.0], dtype=np.float32),
        rights=np.asarray(["C", "P", "C"], dtype=object),
        contract_ids=np.asarray(
            ["SPXW-20260102-06000.000-C", "SPXW-20260102-06005.000-P", "SPXW-20260102-06010.000-C"],
            dtype=object,
        ),
        market_last=np.asarray([6000.0, 18.0, 6000.0, 0.0, 10.0, 1.0, 2.0], dtype=np.float32),
    )


def test_protocol101_candidate_frame_uses_surface_edges_and_history() -> None:
    history = Protocol101HistoryState()
    first = protocol101_candidate_frame_from_surface(
        _decision(),
        np.asarray([0.0, 30.0, 50.0, 1000.0], dtype=np.float32),
        history,
        min_edge=25.0,
    )
    history.update(first, pd.Timestamp(_decision().decision_time))
    second = protocol101_candidate_frame_from_surface(
        _decision(),
        np.asarray([0.0, 26.0, 10.0, 1000.0], dtype=np.float32),
        history,
        min_edge=25.0,
    )

    assert list(first["contract_id"]) == ["SPXW-20260102-06005.000-P", "SPXW-20260102-06000.000-C"]
    assert first["edge"].tolist() == [50.0, 30.0]
    assert set(FEATURE_COLUMNS).issubset(first.columns)
    assert second["hist_prev_candidate_count"].iloc[0] == 2.0
    assert second["hist_prev_max_edge"].iloc[0] == 50.0


def test_protocol101_candidate_frame_respects_time_bucket() -> None:
    decision = _decision()
    decision.decision_time = datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc)

    frame = protocol101_candidate_frame_from_surface(decision, np.asarray([0.0, 50.0, 40.0, 30.0]), Protocol101HistoryState())

    assert frame.empty


def test_protocol101_candidate_gate_diagnostics_explains_below_edge_gate() -> None:
    diagnostics = protocol101_candidate_gate_diagnostics(
        _decision(),
        np.asarray([0.0, 20.0, 24.0, 1000.0], dtype=np.float32),
        min_edge=25.0,
    )

    assert diagnostics["filter_reason"] == "below_min_edge"
    assert diagnostics["time_bucket"] == "post_open_morning"
    assert diagnostics["eligible_token_count"] == 2
    assert diagnostics["above_min_edge_count"] == 0
    assert diagnostics["max_edge"] == 24.0
    assert diagnostics["best_call_edge"] == 20.0
    assert diagnostics["best_put_edge"] == 24.0
    assert diagnostics["top_rejected_contracts"][0]["contract_id"] == "SPXW-20260102-06005.000-P"


def test_protocol101_candidate_gate_diagnostics_explains_time_bucket_block() -> None:
    decision = _decision()
    decision.decision_time = datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc)

    diagnostics = protocol101_candidate_gate_diagnostics(
        decision,
        np.asarray([0.0, 50.0, 40.0, 30.0], dtype=np.float32),
        min_edge=25.0,
    )

    assert diagnostics["filter_reason"] == "outside_time_bucket"
    assert diagnostics["time_bucket"] == "midday"
    assert diagnostics["allowed_time_bucket"] is False
    assert diagnostics["above_min_edge_count"] == 2


def _artifact(tmp_path: Path) -> Path:
    feature_count = len(FEATURE_COLUMNS)
    model = EventSetPolicy(input_dim=feature_count, hidden_dim=8)
    model_path = tmp_path / "model.pt"
    scaler_path = tmp_path / "scaler.json"
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.json"
    torch.save(model.state_dict(), model_path)
    scaler = FeatureScaler.fit(np.zeros((2, feature_count), dtype=np.float32))
    scaler_path.write_text(json.dumps(scaler.to_dict()))
    manifest_path.write_text(
        json.dumps(
            {
                "config": {"hidden_dim": 8},
                "feature_columns": FEATURE_COLUMNS,
                "fold": "fold3_train_q1_q2_q3_validate_q4_test_q1_2026",
                "seed": 1,
            }
        )
    )
    summary_path.write_text(
        json.dumps(
            {
                "fold_results": [
                    {
                        "fold": "fold3_train_q1_q2_q3_validate_q4_test_q1_2026",
                        "seed": 1,
                        "threshold": -999.0,
                    }
                ]
            }
        )
    )
    return manifest_path


def test_load_and_predict_protocol101_entry_artifact(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path)
    artifact = load_protocol101_entry_artifact(manifest, manifest.parent / "summary.json")
    frame = protocol101_candidate_frame_from_surface(
        _decision(),
        np.asarray([0.0, 30.0, 50.0, 0.0], dtype=np.float32),
        Protocol101HistoryState(),
        min_edge=25.0,
    )

    result = predict_protocol101_entry(artifact, frame)

    assert result["action"] in {"enter", "no_entry"}
    assert "margin" in result
