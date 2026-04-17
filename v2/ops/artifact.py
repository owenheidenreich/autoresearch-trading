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

from v2.core.artifact_kind import ArtifactKind, is_promotable
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
    kind: ArtifactKind = ArtifactKind.FINAL_TRAIN,
) -> str:
    """Save an experiment artifact bundle with a deployable model.

    The `kind` field determines promotability. Only FINAL_TRAIN artifacts may
    be promoted to v2/models/model.pt via v2.ops.model_manage.keep().
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
        "artifact_kind": ArtifactKind(kind).value,
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


def save_cv_eval_artifact(
    experiment_id: str,
    cv_report,  # CVReport (avoid circular import in type hints)
    policy: DecisionPolicy = DEFAULT_POLICY,
    train_source_path: str = "v2/train.py",
    kind: ArtifactKind = ArtifactKind.CV_EVAL,
) -> str:
    """Save a walk-forward CV artifact.

    CV_EVAL artifacts do NOT carry a top-level `model.pt` — there is no single
    deployable model from walk-forward CV. Per-fold checkpoints remain under
    {artifact_dir}/folds/<window_id>/model.pt as debug artifacts.

    This artifact is not promotable. Running `model_manage keep` with a CV_EVAL
    artifact is a hard error.
    """
    if kind == ArtifactKind.FINAL_TRAIN:
        raise ValueError(
            "save_cv_eval_artifact refuses FINAL_TRAIN kind. "
            "Use save_artifact() for final-train outputs."
        )

    artifact_dir = os.path.join(ARTIFACTS_DIR, experiment_id)
    os.makedirs(artifact_dir, exist_ok=True)

    # Save the CVReport JSON alongside per-fold checkpoints
    cv_path = os.path.join(artifact_dir, "cv_report.json")
    cv_report.to_json(cv_path)

    # Policy + source snapshots
    with open(os.path.join(artifact_dir, "policy.json"), "w") as f:
        f.write(policy.to_json())
    if os.path.exists(train_source_path):
        shutil.copy2(train_source_path, os.path.join(artifact_dir, "train.py.snapshot"))
    policy_source_path = "v2/core/policy.py"
    if os.path.exists(policy_source_path):
        shutil.copy2(policy_source_path, os.path.join(artifact_dir, "policy.py.snapshot"))

    manifest = {
        "experiment_id": experiment_id,
        "artifact_kind": ArtifactKind(kind).value,
        "schema_version": cv_report.schema_version,
        "screening_mode": cv_report.screening_mode,
        "git_sha": _get_git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "score": cv_report.stability.mean_fold_score,
        "promoted": False,
        "dataset_fingerprint": cv_report.dataset_fingerprint,
        "evaluator_fingerprint": cv_report.evaluator_fingerprint,
        "policy_fingerprint": cv_report.policy_fingerprint,
        "training_config_fingerprint": cv_report.training_config_fingerprint,
        "training_env_overrides": cv_report.training_env_overrides,
        "pooled_profit_factor": cv_report.pooled.profit_factor,
        "pooled_max_account_drawdown": cv_report.pooled.max_account_drawdown,
        "per_fold_scores": cv_report.stability.per_fold_scores,
        "any_fold_gate_failure": cv_report.stability.any_fold_gate_failure,
        "fold_window_ids": [fs.window_id for fs in cv_report.folds],
    }
    with open(os.path.join(artifact_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return artifact_dir


def write_model_pt_manifest(model_pt_path: str, manifest: dict) -> None:
    """Write a sibling manifest next to v2/models/model.pt.

    Promotion interlock: replay / plot tools read this banner at load time.
    If `artifact_kind != FINAL_TRAIN`, the tool should log a loud warning.
    """
    sibling = os.path.splitext(model_pt_path)[0] + ".manifest.json"
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **manifest}
    if "artifact_kind" not in payload:
        raise ValueError("Model manifest requires artifact_kind")
    with open(sibling, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_model_pt_manifest(model_pt_path: str) -> dict | None:
    sibling = os.path.splitext(model_pt_path)[0] + ".manifest.json"
    if not os.path.exists(sibling):
        return None
    with open(sibling) as f:
        return json.load(f)


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
