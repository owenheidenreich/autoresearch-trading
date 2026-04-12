"""Artifact save/load helpers for v3."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import torch

from v3.core.schema import RLArtifactManifest, StepTrace


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def save_traces(path: str, traces: list[StepTrace]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    df = pd.DataFrame([t.to_dict() for t in traces])
    df.to_parquet(path, index=False)


def save_manifest(path: str, manifest: RLArtifactManifest) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def load_manifest(path: str) -> RLArtifactManifest:
    with open(path) as f:
        return RLArtifactManifest(**json.load(f))


def resolve_checkpoint_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        return os.path.join(path_or_dir, "checkpoint.pt")
    return path_or_dir


def create_artifact(
    *,
    artifact_dir: str,
    checkpoint_payload: dict[str, Any],
    manifest: RLArtifactManifest,
    traces: list[StepTrace],
) -> None:
    ensure_dir(artifact_dir)
    save_checkpoint(os.path.join(artifact_dir, "checkpoint.pt"), checkpoint_payload)
    save_manifest(os.path.join(artifact_dir, "manifest.json"), manifest)
    save_traces(os.path.join(artifact_dir, "replay_traces.parquet"), traces)
