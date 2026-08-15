"""Reusable frozen Protocol 051/A+ surface edge scorer.

Protocol 101 expects the `edge` feature produced by the upstream A+ surface
entry model. This module loads the frozen Protocol 051 entry artifact and
scores `SurfaceDecision` objects without constructing broker orders.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from v4.model.hypothesis_protocol import SurfaceActionModel, SurfaceDecision, SurfaceStandardizer
from v4.model.supervised_pilot import FeatureScaler


@dataclass(frozen=True)
class SurfaceEdgeArtifact:
    manifest_path: Path
    model_path: Path
    standardizer_path: Path
    model: SurfaceActionModel
    standardizer: SurfaceStandardizer
    target_scale: float
    variant_name: str
    trial_name: str
    policy_index: int


def load_surface_edge_artifact(manifest_path: Path) -> SurfaceEdgeArtifact:
    """Load a frozen Protocol 051 entry surface artifact from its manifest."""

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    model_path = Path(files.get("entry_model", ""))
    standardizer_path = Path(files.get("entry_standardizer", ""))
    if not model_path.exists():
        raise FileNotFoundError(f"missing entry_model: {model_path}")
    if not standardizer_path.exists():
        raise FileNotFoundError(f"missing entry_standardizer: {standardizer_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{model_path} must contain a checkpoint dictionary")
    scalar_dim = int(checkpoint["scalar_dim"])
    token_dim = int(checkpoint["token_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", manifest.get("entry_config", {}).get("hidden_dim", 128)))
    state_dict = checkpoint["state_dict"]
    token_hidden_dim = int(checkpoint.get("token_hidden_dim", state_dict["token_proj.0.weight"].shape[0]))
    model = SurfaceActionModel(
        scalar_dim=scalar_dim,
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        token_hidden_dim=token_hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()

    standardizer = _standardizer_from_json(standardizer_path)
    return SurfaceEdgeArtifact(
        manifest_path=manifest_path,
        model_path=model_path,
        standardizer_path=standardizer_path,
        model=model,
        standardizer=standardizer,
        target_scale=float(checkpoint.get("target_scale", manifest.get("entry_config", {}).get("target_scale", 100.0))),
        variant_name=str(manifest.get("variant_name", checkpoint.get("variant_name", ""))),
        trial_name=str(manifest.get("trial_name", "")),
        policy_index=int(manifest.get("policy_index", checkpoint.get("policy_index", -1))),
    )


def score_surface_decisions(
    artifact: SurfaceEdgeArtifact,
    decisions: Sequence[SurfaceDecision],
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return flat-plus-token action scores in executable-dollar units."""

    if not decisions:
        return np.empty((0, 1), dtype=np.float32)
    scalar, tokens, _ = artifact.standardizer.transform(decisions)
    out = []
    with torch.no_grad():
        for start in range(0, len(scalar), batch_size):
            pred = artifact.model(
                torch.from_numpy(scalar[start : start + batch_size]),
                torch.from_numpy(tokens[start : start + batch_size]),
            )
            out.append(pred.cpu().numpy().astype(np.float32) * artifact.target_scale)
    return np.vstack(out)


def surface_edge_rows(
    artifact: SurfaceEdgeArtifact,
    decisions: Sequence[SurfaceDecision],
    *,
    batch_size: int = 4096,
) -> list[dict[str, Any]]:
    """Compute one causal best-edge row for each decision."""

    predictions = score_surface_decisions(artifact, decisions, batch_size=batch_size)
    rows = []
    for decision, pred in zip(decisions, predictions):
        rows.append(surface_edge_row(decision, pred))
    return rows


def surface_edge_row(decision: SurfaceDecision, prediction: np.ndarray) -> dict[str, Any]:
    """Compute flat score, best contract score, and best edge for one decision."""

    scores = np.asarray(prediction, dtype=float).copy()
    if scores.shape[0] != len(decision.token_mask) + 1:
        raise ValueError("prediction length must equal flat plus token count")
    mask = np.concatenate([[True], np.asarray(decision.token_mask, dtype=bool)])
    scores[~mask] = -np.inf
    flat_score = float(scores[0])
    token_scores = scores[1:]
    if not np.isfinite(token_scores).any():
        return {
            "session": decision.session,
            "decision_time": decision.decision_time.isoformat(),
            "flat_score": flat_score,
            "best_token_idx": None,
            "best_action_score": None,
            "edge": None,
            "contract_id": None,
            "right": None,
            "offset": None,
            "valid_token_count": int(np.asarray(decision.token_mask, dtype=bool).sum()),
        }
    token_idx = int(np.nanargmax(token_scores))
    action_score = float(token_scores[token_idx])
    return {
        "session": decision.session,
        "decision_time": decision.decision_time.isoformat(),
        "flat_score": flat_score,
        "best_token_idx": token_idx,
        "best_action_score": action_score,
        "edge": float(action_score - flat_score),
        "contract_id": str(decision.contract_ids[token_idx]),
        "right": str(decision.rights[token_idx]),
        "offset": float(decision.offsets[token_idx]),
        "valid_token_count": int(np.asarray(decision.token_mask, dtype=bool).sum()),
    }


def _standardizer_from_json(path: Path) -> SurfaceStandardizer:
    payload = json.loads(path.read_text())
    return SurfaceStandardizer(
        scalar=_scaler_from_dict(payload["scalar"]),
        token=_scaler_from_dict(payload["token"]),
    )


def _scaler_from_dict(payload: dict[str, Any]) -> FeatureScaler:
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )
