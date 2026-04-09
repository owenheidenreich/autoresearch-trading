"""Experiment artifact bundles: self-describing, reproducible model packages.

Each kept experiment produces an artifact directory containing everything
needed to reproduce the evaluation:
  - model.pt (checkpoint weights + model spec)
  - policy.json (DecisionPolicy snapshot)
  - manifest.json (all fingerprints, git SHA, score, timestamp)
  - train.py.snapshot (exact source that produced the model)

Artifacts prevent the "best model becomes unloadable" problem by storing
the model spec alongside the weights, and hard-failing on any mismatch
during load.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time

import torch

from v2.core.policy import DecisionPolicy, DEFAULT_POLICY
from v2.core.metrics import score_config_fingerprint


ARTIFACTS_DIR = os.path.join("v2", "artifacts")


def _get_git_sha() -> str:
    """Get current git HEAD SHA (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _file_fingerprint(path: str) -> str:
    """SHA-256 fingerprint of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_artifact(
    experiment_id: str,
    model_path: str,
    score: float,
    policy: DecisionPolicy = DEFAULT_POLICY,
    dataset_fingerprint: str = "unknown",
    train_source_path: str = "v2/train.py",
    extra_metadata: dict | None = None,
) -> str:
    """Save an experiment artifact bundle.

    Returns the artifact directory path.
    """
    artifact_dir = os.path.join(ARTIFACTS_DIR, experiment_id)
    os.makedirs(artifact_dir, exist_ok=True)

    # Copy model checkpoint
    dst_model = os.path.join(artifact_dir, "model.pt")
    shutil.copy2(model_path, dst_model)

    # Save policy
    dst_policy = os.path.join(artifact_dir, "policy.json")
    with open(dst_policy, "w") as f:
        f.write(policy.to_json())

    # Snapshot mutable source files
    dst_train = os.path.join(artifact_dir, "train.py.snapshot")
    if os.path.exists(train_source_path):
        shutil.copy2(train_source_path, dst_train)

    policy_source_path = "v2/core/policy.py"
    dst_policy_src = os.path.join(artifact_dir, "policy.py.snapshot")
    if os.path.exists(policy_source_path):
        shutil.copy2(policy_source_path, dst_policy_src)

    # Load checkpoint to get hyperparams
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    hyperparams = checkpoint.get('hyperparams', {})

    # Build manifest
    manifest = {
        "experiment_id": experiment_id,
        "git_sha": _get_git_sha(),
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "score": score,
        "promoted": False,
        "dataset_fingerprint": dataset_fingerprint,
        "evaluator_fingerprint": score_config_fingerprint(),
        "policy_fingerprint": policy.fingerprint(),
        "model_fingerprint": _file_fingerprint(model_path),
        "hyperparams": hyperparams,
        "checkpoint_epoch": checkpoint.get('epoch', -1),
        "checkpoint_val_loss": checkpoint.get('val_loss', -1),
    }
    if extra_metadata:
        manifest["extra"] = extra_metadata

    dst_manifest = os.path.join(artifact_dir, "manifest.json")
    with open(dst_manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return artifact_dir


def load_artifact(
    artifact_dir: str,
    current_dataset_fingerprint: str | None = None,
    device: str = "cpu",
) -> dict:
    """Load an artifact bundle and verify fingerprints.

    Returns dict with: model, policy, manifest, artifact_dir.
    Hard-fails on fingerprint mismatch.
    """
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.json in {artifact_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Verify dataset fingerprint
    if current_dataset_fingerprint is not None:
        saved_fp = manifest.get("dataset_fingerprint", "unknown")
        if saved_fp != "unknown" and saved_fp != current_dataset_fingerprint:
            raise ValueError(
                f"Dataset fingerprint mismatch: "
                f"artifact={saved_fp}, current={current_dataset_fingerprint}. "
                f"Model was trained on different data."
            )

    # Verify evaluator fingerprint
    current_eval_fp = score_config_fingerprint()
    saved_eval_fp = manifest.get("evaluator_fingerprint", "unknown")
    if saved_eval_fp != "unknown" and saved_eval_fp != current_eval_fp:
        raise ValueError(
            f"Evaluator fingerprint mismatch: "
            f"artifact={saved_eval_fp}, current={current_eval_fp}. "
            f"Score formula has changed since this model was evaluated."
        )

    # Load policy
    policy_path = os.path.join(artifact_dir, "policy.json")
    if os.path.exists(policy_path):
        with open(policy_path) as f:
            policy = DecisionPolicy.from_json(f.read())
    else:
        policy = DEFAULT_POLICY

    # Load model using saved hyperparams
    model_path = os.path.join(artifact_dir, "model.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Reconstruct model from saved hyperparams
    hyperparams = manifest.get("hyperparams", checkpoint.get("hyperparams", {}))

    from v2.train import TradingModel
    model = TradingModel(
        d_model=hyperparams.get('d_model', 64),
        depth=hyperparams.get('depth', 3),
        n_heads=hyperparams.get('n_heads', 4),
        dropout=hyperparams.get('dropout', 0.1),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return {
        "model": model,
        "policy": policy,
        "manifest": manifest,
        "artifact_dir": artifact_dir,
    }


def mark_promoted(artifact_dir: str):
    """Mark an artifact as promoted (kept). Only promoted artifacts are
    eligible for ``get_best_artifact``."""
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["promoted"] = True
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def mark_reverted(artifact_dir: str):
    """Mark an artifact as reverted. Reverted artifacts are excluded from
    ``get_best_artifact``."""
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["promoted"] = False
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def iter_artifacts_by_score(promoted_only: bool = True) -> list[str]:
    """Return eligible artifact directories sorted by descending score.

    Args:
        promoted_only: if True (default), only consider artifacts that have
            been explicitly promoted. Legacy manifests without a
            ``promoted`` field are treated as ineligible so pre-repair
            artifacts never become the implicit "best" model again.

    Returns a possibly-empty list of artifact directory paths.
    """
    if not os.path.exists(ARTIFACTS_DIR):
        return []

    manifests: list[tuple[float, str, object]] = []
    for name in os.listdir(ARTIFACTS_DIR):
        manifest_path = os.path.join(ARTIFACTS_DIR, name, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            promoted = manifest.get("promoted")
            score = manifest.get("score", float('-inf'))
            manifests.append((score, os.path.join(ARTIFACTS_DIR, name), promoted))
        except (json.JSONDecodeError, KeyError):
            continue

    eligible: list[tuple[float, str]] = []
    for score, artifact_dir, promoted in manifests:
        if promoted_only and promoted is not True:
            continue
        eligible.append((score, artifact_dir))
    eligible.sort(key=lambda item: item[0], reverse=True)
    return [artifact_dir for _, artifact_dir in eligible]


def get_best_artifact(promoted_only: bool = True) -> str | None:
    """Find the best eligible artifact, or None if no candidate exists."""
    artifact_dirs = iter_artifacts_by_score(promoted_only=promoted_only)
    return artifact_dirs[0] if artifact_dirs else None
