"""Tests for governed Protocol101 fair-contract loading."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from v4.model.protocol101_governed_loader import (
    GovernedLoaderArtifacts,
    PROTECTED_HOLDOUT_OWNER_OVERRIDE_TOKEN,
    _manifest_by_session,
    file_sha256,
    resolve_governed_split_paths,
    validate_governance_artifacts,
    validate_session_for_role,
)
from v4.model.protocol101_regimen_repair import (
    Protocol101DuplicateSessionMembershipError,
)
from v4.scripts.build_protocol101_protected_holdout_artifact import build_artifact
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import compute_registry_hash


def _artifacts(tmp_path: Path, *, verifier_version: int = 35, status: str = "pass") -> tuple[Path, GovernedLoaderArtifacts]:
    processed = tmp_path / "2024-10-01.pkl"
    processed.write_bytes(b"accepted")
    acceptance = {
        "status": "pass",
        "verifier_version": verifier_version,
        "thresholds_are_defaults": True,
        "fee_model": {"fee_model": "gross_no_fees", "fee_per_contract": 0.0},
        "sessions": [
            {
                "session": "2024-10-01",
                "status": status,
                "verifier_version": verifier_version,
                "early_close_session": False,
                "processed": {
                    "processed_exists": True,
                    "neural_rows": 360,
                    "processed_path": str(processed),
                    "processed_sha256": file_sha256(processed),
                },
            }
        ],
    }
    acceptance["registry_hash"] = compute_registry_hash(acceptance)
    era_manifest = {
        "status": "pass",
        "sessions": [{"session": "2024-10-01", "era": "pre_program_oct2024_jun2025"}],
    }
    role_policy = {
        "status": "pass",
        "policy": {"pre_program_oct2024_jun2025": {"permitted_roles": ["train", "test", "diagnostics_only"]}},
    }
    protected_holdout = build_artifact(sessions=["2099-12-31"], owner_note="unit-test placeholder")
    artifacts = GovernedLoaderArtifacts(
        acceptance_registry_path=tmp_path / "acceptance.json",
        acceptance_registry=acceptance,
        era_manifest_path=tmp_path / "era.json",
        era_manifest=era_manifest,
        role_policy_path=tmp_path / "policy.json",
        role_policy=role_policy,
        protected_holdout_path=tmp_path / "protected_holdout.json",
        protected_holdout=protected_holdout,
    )
    return processed, artifacts


def test_validate_session_for_role_accepts_passed_v35_record(tmp_path: Path) -> None:
    processed, artifacts = _artifacts(tmp_path)

    result = validate_session_for_role(
        session="2024-10-01",
        role="train",
        processed_path=processed,
        artifacts=artifacts,
    )

    assert result["placeable"] is True
    assert result["blockers"] == []


def test_validate_session_for_role_rechecks_processed_hash(tmp_path: Path) -> None:
    processed, artifacts = _artifacts(tmp_path)
    processed.write_bytes(b"mutated")

    result = validate_session_for_role(
        session="2024-10-01",
        role="train",
        processed_path=processed,
        artifacts=artifacts,
    )

    assert result["placeable"] is False
    assert "processed_hash_mismatch" in result["blockers"]


def test_validate_session_for_role_rejects_old_verifier_and_holdout(tmp_path: Path) -> None:
    processed, artifacts = _artifacts(tmp_path, verifier_version=3)
    artifacts = replace(artifacts, protected_holdout_sessions=frozenset({"2024-10-01"}))

    result = validate_session_for_role(
        session="2024-10-01",
        role="train",
        processed_path=processed,
        artifacts=artifacts,
    )

    assert result["placeable"] is False
    assert "acceptance_registry_verifier_too_old" in result["blockers"]
    assert "protected_holdout_session_requested" in result["blockers"]


def test_resolve_governed_split_paths_blocks_wrong_role(tmp_path: Path) -> None:
    processed, artifacts = _artifacts(tmp_path)
    design = {
        "split_policy": {
            "train_sessions": ["2024-10-01"],
            "validation_sessions": [],
            "diagnostic_test_sessions": [],
        }
    }
    manifest = {"included_sessions": [{"session": "2024-10-01", "processed_file": str(processed)}]}
    artifacts = replace(
        artifacts,
        role_policy={
            "status": "pass",
            "policy": {"pre_program_oct2024_jun2025": {"permitted_roles": ["diagnostics_only"]}},
        },
    )

    paths, blockers, governance = resolve_governed_split_paths(
        design=design,
        manifest=manifest,
        artifacts=artifacts,
    )

    assert paths["train"] == []
    assert "session_not_placeable:train:2024-10-01" in blockers
    assert governance["governance_hash"]


def test_validate_governance_artifacts_rejects_registry_hash_mismatch(tmp_path: Path) -> None:
    _, artifacts = _artifacts(tmp_path)
    tampered = dict(artifacts.acceptance_registry)
    tampered["thresholds_are_defaults"] = False
    artifacts = replace(artifacts, acceptance_registry=tampered)

    blockers = validate_governance_artifacts(artifacts)

    assert "acceptance_registry_hash_mismatch" in blockers
    assert "acceptance_thresholds_not_defaults" in blockers


def test_validate_governance_artifacts_requires_passing_protected_holdout(tmp_path: Path) -> None:
    _, artifacts = _artifacts(tmp_path)
    artifacts = replace(artifacts, protected_holdout=build_artifact(sessions=[], owner_note="not declared"))

    blockers = validate_governance_artifacts(artifacts)

    assert "protected_holdout_artifact_status_not_pass" in blockers


def test_protected_holdout_artifact_blocks_session_without_owner_override(tmp_path: Path) -> None:
    processed, artifacts = _artifacts(tmp_path)
    artifacts = replace(
        artifacts,
        protected_holdout=build_artifact(sessions=["2024-10-01"], owner_note="lockbox"),
    )

    blocked = validate_session_for_role(
        session="2024-10-01",
        role="train",
        processed_path=processed,
        artifacts=artifacts,
    )
    overridden = replace(
        artifacts,
        protected_holdout_owner_override_token=PROTECTED_HOLDOUT_OWNER_OVERRIDE_TOKEN,
    )
    train_despite_override = validate_session_for_role(
        session="2024-10-01",
        role="train",
        processed_path=processed,
        artifacts=overridden,
    )
    test_with_override = validate_session_for_role(
        session="2024-10-01",
        role="test",
        processed_path=processed,
        artifacts=overridden,
    )

    assert "protected_holdout_session_requested" in blocked["blockers"]
    assert blocked["protected_holdout_override_used"] is False
    assert train_despite_override["placeable"] is False
    assert (
        "protected_holdout_override_only_permits_test_role"
        in train_despite_override["blockers"]
    )
    assert test_with_override["placeable"] is True
    assert test_with_override["protected_holdout_override_used"] is True


def test_manifest_duplicate_fails_before_dictionary_overwrite() -> None:
    manifest = {
        "included_sessions": [
            {"session": "2025-01-02", "processed_file": "first.pkl"},
            {"session": "2025-01-02", "processed_file": "second.pkl"},
        ]
    }
    with pytest.raises(Protocol101DuplicateSessionMembershipError):
        _manifest_by_session(manifest)
