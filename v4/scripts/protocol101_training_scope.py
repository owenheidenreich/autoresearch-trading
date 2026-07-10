"""Helpers for Protocol101 fair-contract training-scope scripts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from v4.model.protocol101_governed_loader import (
    load_governed_loader_artifacts,
    resolve_governed_expanding_fold_paths,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    DEFAULT_ACCEPTANCE_REGISTRY,
    DEFAULT_DESIGN,
    DEFAULT_ERA_MANIFEST,
    DEFAULT_PROTECTED_HOLDOUT,
    DEFAULT_ROLE_POLICY,
)


@dataclass(frozen=True)
class TrainingScope:
    """Resolved 5-fold training design paths and session metadata."""

    sessions: list[tuple[str, Path]]
    folds: list[dict[str, object]]
    design_path: Path
    manifest_path: Path
    acceptance_registry_path: Path
    fold_governance_hash: str
    acceptance_registry_hash: str


def session_from_path(path: Path) -> str:
    return path.stem


def load_training_scope(
    *,
    design_path: Path = DEFAULT_DESIGN,
    acceptance_registry_path: Path = DEFAULT_ACCEPTANCE_REGISTRY,
    era_manifest_path: Path = DEFAULT_ERA_MANIFEST,
    role_policy_path: Path = DEFAULT_ROLE_POLICY,
    protected_holdout_path: Path = DEFAULT_PROTECTED_HOLDOUT,
) -> TrainingScope:
    """Resolve the owner-approved 5-fold expanding-window training design.

    The returned ``sessions`` are the unique train/validation fold sessions,
    excluding protected holdout, report-only, recorder/parity, and embargo-only
    rows according to the design and governed loader.
    """

    design = json.loads(design_path.read_text())
    manifest_path = Path(str(design["allowed_data"]["canonical_manifest"]))
    manifest = json.loads(manifest_path.read_text())
    artifacts = load_governed_loader_artifacts(
        acceptance_registry_path=acceptance_registry_path,
        era_manifest_path=era_manifest_path,
        role_policy_path=role_policy_path,
        protected_holdout_path=protected_holdout_path,
    )
    fold_paths, blockers, governance = resolve_governed_expanding_fold_paths(
        design=design,
        manifest=manifest,
        artifacts=artifacts,
    )
    if blockers:
        raise SystemExit(f"training-scope fold blockers: {blockers}")

    unique: dict[str, Path] = {}
    folds: list[dict[str, object]] = []
    for fold_number, fold_id in enumerate(sorted(fold_paths), start=0):
        train_sessions = [session_from_path(path) for path in fold_paths[fold_id]["train"]]
        validation_sessions = [session_from_path(path) for path in fold_paths[fold_id]["validation"]]
        for path in [*fold_paths[fold_id]["train"], *fold_paths[fold_id]["validation"]]:
            unique[session_from_path(path)] = path
        folds.append(
            {
                "fold": fold_number,
                "fold_id": fold_id,
                "train_sessions": sorted(train_sessions),
                "test_sessions": sorted(validation_sessions),
                "validation_sessions": sorted(validation_sessions),
            }
        )
    sessions = [(session, unique[session]) for session in sorted(unique)]
    return TrainingScope(
        sessions=sessions,
        folds=folds,
        design_path=design_path,
        manifest_path=manifest_path,
        acceptance_registry_path=acceptance_registry_path,
        fold_governance_hash=str(governance.get("fold_governance_hash") or ""),
        acceptance_registry_hash=str(artifacts.acceptance_registry.get("registry_hash") or ""),
    )
