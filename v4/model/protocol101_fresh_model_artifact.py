"""Prediction-neutral artifact binding for freshly fitted Protocol101 models."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping


MODEL_ARTIFACT_BINDING_SCHEMA = (
    "Protocol101FreshCampaignModelArtifactBindingV1"
)
MODEL_ARTIFACT_BINDING_ATTRIBUTE = (
    "_protocol101_fresh_campaign_artifact_binding"
)


def bind_model_artifact(
    model: Any,
    *,
    campaign_namespace: str,
    unit_identity: str,
    binding_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Attach artifact provenance without changing estimator computation."""

    if not campaign_namespace or not unit_identity:
        raise ValueError("fresh model artifact identity is required")
    if hasattr(model, MODEL_ARTIFACT_BINDING_ATTRIBUTE):
        raise ValueError("fresh model artifact is already bound")
    binding = {
        "schema_version": MODEL_ARTIFACT_BINDING_SCHEMA,
        "campaign_namespace": campaign_namespace,
        "unit_identity": unit_identity,
        "binding_sources": dict(sorted(binding_sources.items())),
        "prediction_semantics_changed": False,
        "scientific_model_state_changed": False,
    }
    setattr(model, MODEL_ARTIFACT_BINDING_ATTRIBUTE, binding)
    return binding


def write_bound_model(
    path: Path,
    model: Any,
    *,
    allowed_root: Path,
    campaign_namespace: str,
    binding_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Write one campaign-bound pickle inside the authorized fitted root."""

    path = Path(path)
    allowed_root = Path(allowed_root).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    try:
        unit_identity = resolved.parent.relative_to(allowed_root).as_posix()
    except ValueError as exc:
        raise ValueError("fresh model artifact path is outside fitted root") from exc
    if resolved.name != "model.pkl" or not unit_identity:
        raise ValueError("fresh model artifact path is not a unit model.pkl")
    binding = bind_model_artifact(
        model,
        campaign_namespace=campaign_namespace,
        unit_identity=unit_identity,
        binding_sources=binding_sources,
    )
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, resolved)
    return binding
