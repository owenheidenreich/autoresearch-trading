"""Governed Protocol101 fair-contract data loading.

This module is the enforcement layer between acceptance artifacts and model
training. It refuses sessions that are not placeable for the requested role,
re-verifies the accepted processed file hash at load time, and emits the hashes
that make an experiment traceable to the governance state it used.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    MIN_FOLD_PLACEMENT_VERIFIER_VERSION,
    compute_registry_hash,
    fold_placement_predicate,
    stable_hash,
)
from v4.scripts.build_protocol101_protected_holdout_artifact import (
    compute_protected_holdout_hash,
)


SCHEMA_VERSION = "Protocol101GovernedLoaderV1"
DEFAULT_PROTECTED_HOLDOUT_PATH = Path("v4/audit/autoresearch/protocol101_protected_holdout/summary.json")
PROTECTED_HOLDOUT_OWNER_OVERRIDE_TOKEN = "OWNER_APPROVED_PROTECTED_HOLDOUT_EVALUATION"
DEFAULT_SPLIT_ROLE_MAP = {
    "train": "train",
    "validation": "test",
    "diagnostic_test": "diagnostics_only",
}


@dataclass(frozen=True)
class GovernedLoaderArtifacts:
    acceptance_registry_path: Path
    acceptance_registry: dict[str, Any]
    era_manifest_path: Path
    era_manifest: dict[str, Any]
    role_policy_path: Path
    role_policy: dict[str, Any]
    protected_holdout_path: Path | None = None
    protected_holdout: dict[str, Any] | None = None
    protected_holdout_sessions: frozenset[str] = frozenset()
    protected_holdout_owner_override_token: str = ""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_governed_loader_artifacts(
    *,
    acceptance_registry_path: Path,
    era_manifest_path: Path,
    role_policy_path: Path,
    protected_holdout_path: Path | None = DEFAULT_PROTECTED_HOLDOUT_PATH,
    protected_holdout_sessions: set[str] | frozenset[str] | None = None,
    protected_holdout_owner_override_token: str = "",
) -> GovernedLoaderArtifacts:
    return GovernedLoaderArtifacts(
        acceptance_registry_path=acceptance_registry_path,
        acceptance_registry=load_json(acceptance_registry_path),
        era_manifest_path=era_manifest_path,
        era_manifest=load_json(era_manifest_path),
        role_policy_path=role_policy_path,
        role_policy=load_json(role_policy_path),
        protected_holdout_path=protected_holdout_path,
        protected_holdout=load_json(protected_holdout_path) if protected_holdout_path else {},
        protected_holdout_sessions=frozenset(protected_holdout_sessions or set()),
        protected_holdout_owner_override_token=str(protected_holdout_owner_override_token),
    )


def _manifest_by_session(manifest: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in manifest.get("included_sessions") or []:
        session = str(item.get("session") or "")
        processed_file = item.get("processed_file")
        if session and processed_file:
            out[session] = Path(str(processed_file))
    return out


def _acceptance_by_session(acceptance_registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("session")): item
        for item in acceptance_registry.get("sessions", [])
        if item.get("session")
    }


def _protected_holdout_sessions(artifacts: GovernedLoaderArtifacts) -> frozenset[str]:
    artifact_sessions = set()
    holdout = artifacts.protected_holdout if isinstance(artifacts.protected_holdout, dict) else {}
    for session in holdout.get("sessions") or []:
        session_text = str(session).strip()
        if session_text:
            artifact_sessions.add(session_text)
    return frozenset(set(artifacts.protected_holdout_sessions) | artifact_sessions)


def _has_protected_holdout_owner_override(artifacts: GovernedLoaderArtifacts) -> bool:
    return artifacts.protected_holdout_owner_override_token == PROTECTED_HOLDOUT_OWNER_OVERRIDE_TOKEN


def validate_governance_artifacts(artifacts: GovernedLoaderArtifacts) -> list[str]:
    blockers: list[str] = []
    if artifacts.acceptance_registry.get("status") != "pass":
        blockers.append("acceptance_registry_status_not_pass")
    embedded_registry_hash = str(artifacts.acceptance_registry.get("registry_hash") or "")
    if not embedded_registry_hash:
        blockers.append("acceptance_registry_hash_missing")
    elif compute_registry_hash(artifacts.acceptance_registry) != embedded_registry_hash:
        blockers.append("acceptance_registry_hash_mismatch")
    if artifacts.era_manifest.get("status") != "pass":
        blockers.append("era_manifest_status_not_pass")
    if artifacts.role_policy.get("status") != "pass":
        blockers.append("role_policy_status_not_pass")
    holdout = artifacts.protected_holdout if isinstance(artifacts.protected_holdout, dict) else {}
    if holdout.get("status") != "pass":
        blockers.append("protected_holdout_artifact_status_not_pass")
    else:
        embedded_holdout_hash = str(holdout.get("protected_holdout_hash") or "")
        if not embedded_holdout_hash:
            blockers.append("protected_holdout_hash_missing")
        elif compute_protected_holdout_hash(holdout) != embedded_holdout_hash:
            blockers.append("protected_holdout_hash_mismatch")
    try:
        verifier_version = int(artifacts.acceptance_registry.get("verifier_version") or 0)
    except (TypeError, ValueError):
        verifier_version = 0
    if verifier_version < MIN_FOLD_PLACEMENT_VERIFIER_VERSION:
        blockers.append("acceptance_registry_verifier_too_old")
    if artifacts.acceptance_registry.get("thresholds_are_defaults") is not True:
        blockers.append("acceptance_thresholds_not_defaults")
    if artifacts.acceptance_registry.get("fee_model", {}).get("fee_model") != "gross_no_fees":
        blockers.append("unexpected_acceptance_fee_model")
    return blockers


def validate_session_for_role(
    *,
    session: str,
    role: str,
    processed_path: Path,
    artifacts: GovernedLoaderArtifacts,
) -> dict[str, Any]:
    blockers = validate_governance_artifacts(artifacts)
    acceptance_records = _acceptance_by_session(artifacts.acceptance_registry)
    record = acceptance_records.get(str(session), {})
    predicate = fold_placement_predicate(
        session=str(session),
        role=str(role),
        era_manifest=artifacts.era_manifest,
        role_policy=artifacts.role_policy,
        acceptance_registry=artifacts.acceptance_registry,
    )
    if not predicate.get("placeable"):
        blockers.append("fold_placement_predicate_not_placeable")
    if (
        str(session) in _protected_holdout_sessions(artifacts)
        and not _has_protected_holdout_owner_override(artifacts)
    ):
        blockers.append("protected_holdout_session_requested")
    if record.get("status") != "pass":
        blockers.append("session_acceptance_not_pass")
    if record.get("early_close_session"):
        blockers.append("early_close_report_only_session")
    if not processed_path.exists():
        blockers.append("processed_file_missing")
    else:
        actual_hash = file_sha256(processed_path)
        accepted_hash = str(record.get("processed", {}).get("processed_sha256") or "")
        if not accepted_hash:
            blockers.append("accepted_processed_hash_missing")
        elif actual_hash != accepted_hash:
            blockers.append("processed_hash_mismatch")
    accepted_path = str(record.get("processed", {}).get("processed_path") or "")
    if accepted_path and Path(accepted_path) != processed_path:
        blockers.append("processed_path_differs_from_acceptance_record")
    return {
        "schema_version": "Protocol101GovernedSessionValidationV1",
        "session": str(session),
        "role": str(role),
        "processed_path": str(processed_path),
        "accepted_processed_sha256": str(record.get("processed", {}).get("processed_sha256") or ""),
        "placeable": not blockers,
        "predicate": predicate,
        "blockers": sorted(set(blockers)),
    }


def resolve_governed_split_paths(
    *,
    design: dict[str, Any],
    manifest: dict[str, Any],
    artifacts: GovernedLoaderArtifacts,
    split_role_map: dict[str, str] | None = None,
) -> tuple[dict[str, list[Path]], list[str], dict[str, Any]]:
    role_map = split_role_map or DEFAULT_SPLIT_ROLE_MAP
    split_policy = design.get("split_policy") or {}
    by_session = _manifest_by_session(manifest)
    paths = {"train": [], "validation": [], "diagnostic_test": []}
    blockers: list[str] = []
    validations: dict[str, list[dict[str, Any]]] = {name: [] for name in paths}
    for split_name, key in (
        ("train", "train_sessions"),
        ("validation", "validation_sessions"),
        ("diagnostic_test", "diagnostic_test_sessions"),
    ):
        role = role_map[split_name]
        for session in split_policy.get(key) or []:
            session_text = str(session)
            path = by_session.get(session_text)
            if path is None:
                blockers.append(f"manifest_missing_session:{split_name}:{session_text}")
                continue
            validation = validate_session_for_role(
                session=session_text,
                role=role,
                processed_path=path,
                artifacts=artifacts,
            )
            validations[split_name].append(validation)
            if validation["placeable"]:
                paths[split_name].append(path)
            else:
                blockers.append(f"session_not_placeable:{split_name}:{session_text}")
                blockers.extend(f"{session_text}:{item}" for item in validation["blockers"])
    for split_name, split_paths in paths.items():
        if not split_paths:
            blockers.append(f"empty_split:{split_name}")
    governance_payload = {
        "schema_version": SCHEMA_VERSION,
        "acceptance_registry_path": str(artifacts.acceptance_registry_path),
        "era_manifest_path": str(artifacts.era_manifest_path),
        "role_policy_path": str(artifacts.role_policy_path),
        "protected_holdout_path": str(artifacts.protected_holdout_path) if artifacts.protected_holdout_path else "",
        "acceptance_registry_hash": stable_hash(artifacts.acceptance_registry),
        "acceptance_registry_embedded_hash": str(artifacts.acceptance_registry.get("registry_hash") or ""),
        "era_manifest_hash": stable_hash(artifacts.era_manifest),
        "role_policy_hash": stable_hash(artifacts.role_policy),
        "protected_holdout_hash": stable_hash(artifacts.protected_holdout or {}),
        "protected_holdout_embedded_hash": str((artifacts.protected_holdout or {}).get("protected_holdout_hash") or ""),
        "minimum_verifier_version": MIN_FOLD_PLACEMENT_VERIFIER_VERSION,
        "split_role_map": role_map,
        "validations": validations,
        "protected_holdout_sessions": sorted(_protected_holdout_sessions(artifacts)),
        "protected_holdout_owner_override": _has_protected_holdout_owner_override(artifacts),
    }
    governance_payload["governance_hash"] = stable_hash(governance_payload)
    return paths, sorted(set(blockers)), governance_payload
