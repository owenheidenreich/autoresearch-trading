from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from v4.live.protocol051_surface_edge import (
    load_surface_edge_artifact,
    score_surface_decisions,
    surface_edge_row,
)
from v4.model.hypothesis_protocol import SurfaceActionModel, SurfaceDecision, SurfaceStandardizer
from v4.model.supervised_pilot import FeatureScaler


def _artifact(tmp_path: Path) -> Path:
    model = SurfaceActionModel(scalar_dim=2, token_dim=3, hidden_dim=8, token_hidden_dim=4, dropout=0.0)
    model_path = tmp_path / "entry_model.pt"
    standardizer_path = tmp_path / "entry_standardizer.json"
    manifest_path = tmp_path / "manifest.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class": "SurfaceActionModel",
            "scalar_dim": 2,
            "token_dim": 3,
            "hidden_dim": 8,
            "target_scale": 100.0,
            "policy_index": 1,
            "variant_name": "surface_structure_aplus_side_value_rank",
        },
        model_path,
    )
    standardizer = SurfaceStandardizer(
        scalar=FeatureScaler.fit(np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)),
        token=FeatureScaler.fit(np.asarray([[1.0, 0.0, 0.1], [2.0, 1.0, 0.2]], dtype=np.float32)),
    )
    standardizer_path.write_text(json.dumps(standardizer.to_dict()))
    manifest_path.write_text(
        json.dumps(
            {
                "variant_name": "surface_structure_aplus_side_value_rank",
                "trial_name": "post_open_late_edge25_max2",
                "policy_index": 1,
                "files": {
                    "entry_model": str(model_path),
                    "entry_standardizer": str(standardizer_path),
                },
            }
        )
    )
    return manifest_path


def _decision() -> SurfaceDecision:
    return SurfaceDecision(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
        scalar_features=np.asarray([1.0, 2.0], dtype=np.float32),
        token_features=np.asarray([[1.0, 0.0, 0.1], [2.0, 1.0, 0.2]], dtype=np.float32),
        token_mask=np.asarray([True, False]),
        labels=np.asarray([100.0, -50.0], dtype=np.float32),
        offsets=np.asarray([0.0, 5.0], dtype=np.float32),
        rights=np.asarray(["C", "P"], dtype=object),
        contract_ids=np.asarray(["SPXW-20260102-06000.000-C", "SPXW-20260102-06005.000-P"], dtype=object),
        market_last=np.zeros(4, dtype=np.float32),
    )


def test_load_surface_edge_artifact_and_score_decision(tmp_path: Path) -> None:
    artifact = load_surface_edge_artifact(_artifact(tmp_path))
    decisions = [_decision()]

    scores = score_surface_decisions(artifact, decisions)

    assert scores.shape == (1, 3)
    assert np.isfinite(scores).all()
    assert artifact.variant_name == "surface_structure_aplus_side_value_rank"


def test_surface_edge_row_ignores_masked_tokens() -> None:
    decision = _decision()
    row = surface_edge_row(decision, np.asarray([0.0, 1.0, 999.0], dtype=np.float32))

    assert row["best_token_idx"] == 0
    assert row["contract_id"] == "SPXW-20260102-06000.000-C"
    assert row["edge"] == 1.0


def test_load_surface_edge_artifact_requires_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {"entry_model": str(tmp_path / "missing.pt"), "entry_standardizer": str(tmp_path / "missing.json")}}))

    try:
        load_surface_edge_artifact(manifest)
    except FileNotFoundError as exc:
        assert "entry_model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing artifact to raise")
