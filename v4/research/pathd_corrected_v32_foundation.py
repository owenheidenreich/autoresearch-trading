"""Write-once corrected-v3.2 Path-D foundation freeze.

This module is deliberately independent of the fit/evidence entrypoints.  It may
only copy the v3.1 scientific contract, replace the source-policy/generation
bindings, and return the release to a Claude-verification pause.  It never reads
the research corpus and cannot create fold, evidence, or holdout namespaces.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from v4.research import pathd_entry_exit as base


V31_ROOT = base.REPO_ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_1_2026_08_01"
)
V32_ROOT = base.REPO_ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_2_2026_08_01"
)
V31_PREREGISTRATION_SHA256 = (
    "7fa621d63086a6dd28fb10cb731f229471051951875f0c663d53838e7771b370"
)
V31_FOUNDATION_GENERATION_SHA256 = (
    "1b6eb620668be3655facac71467513d50eafa706a1aedc64dc507d0dbce873b2"
)
V31_FILE_SHA256S = {
    "feature_lineage.json": "4521b3e62982f672daaba73d98ce5a9ce72df170b083cd40df7dc979983d1ef0",
    "foundation_restoration_receipt.json": "3cecec8bda2c829153200e7b081beccf29f39474384f05e23d0dff2b024d00cc",
    "preregistration.json": V31_PREREGISTRATION_SHA256,
    "preregistration.sha256": "ce3a76c74f1b6a13dc078a900e50513a7487224f7cfe01f55eb860105062e4b8",
    "preregistration_freeze_receipt.json": "6eb38341c6f99b42fc318b9644d25d7d0a9e5ca98e63d1cad278338a7f700f98",
    "session_assignments.json": "431cd14879ad6a14b278cb2683e4f3b860eef960e6b74a4f0edb3e620cf82475",
}
EXECUTABLE_SOURCE_PATHS = (
    "v4/research/pathd_entry_features.py",
    "v4/research/pathd_entry_dataset.py",
    "v4/research/pathd_entry_models.py",
    "v4/research/pathd_entry_execution_v32.py",
    "v4/research/pathd_evidence_gate.py",
    "v4/research/pathd_corrected_v32_foundation.py",
    "v4/path_d/execution/research_fill_law.py",
    "v4/path_d/execution/research_replay.py",
    "v4/scripts/run_pathd_entry_exit_research.py",
)
EXECUTABLE_TEST_PATHS = (
    "v4/tests/test_pathd_evidence_gate.py",
    "v4/tests/test_pathd_corrected_v32_executable_build.py",
)


def _deepcopy(value: Any) -> Any:
    return base.strict_json_loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _assert_v31_bytes() -> None:
    if V31_ROOT.is_symlink() or not V31_ROOT.is_dir():
        raise RuntimeError("corrected-v3.1 foundation root drift")
    if {path.name for path in V31_ROOT.iterdir()} != set(V31_FILE_SHA256S):
        raise RuntimeError("corrected-v3.1 foundation inventory drift")
    for name, expected in V31_FILE_SHA256S.items():
        if base.sha256_path(V31_ROOT / name) != expected:
            raise RuntimeError(f"corrected-v3.1 foundation byte drift: {name}")


def _source_hashes(v31: dict[str, Any]) -> dict[str, str]:
    labels = set(v31["source_hashes_at_freeze"])
    labels.update(EXECUTABLE_SOURCE_PATHS)
    labels.update(EXECUTABLE_TEST_PATHS)
    hashes: dict[str, str] = {}
    for label in sorted(labels):
        path = base.REPO_ROOT / label
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v3.2 frozen source absent or noncanonical: {label}")
        hashes[label] = base.sha256_path(path)
    return hashes


def _science_projection(payload: dict[str, Any]) -> dict[str, Any]:
    projected = _deepcopy(payload)
    projected.pop("supersedes", None)
    projected.pop("source_hash_policy", None)
    projected.pop("source_hashes_at_freeze", None)
    pause = projected["foundation_correction"]["prefit_pause"]
    for key in (
        "claude_verification_pending",
        "corpus_decode_authorized",
        "foundation_stability_receipt_sealed",
        "foundation_stability_seal_authorized",
        "machinery_seal_authorized",
        "model_fit_authorized",
        "nested_or_outer_evidence_open_authorized",
        "protected_holdout_open_authorized",
        "separate_post_verification_release_required",
    ):
        pause.pop(key, None)
    return projected


def corrected_v32_payload() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reconstruct the exact v3.2 preregistration without writing anything."""

    _assert_v31_bytes()
    v31 = base.read_json(V31_ROOT / "preregistration.json")
    sessions = base.read_json(V31_ROOT / "session_assignments.json")
    lineage = base.read_json(V31_ROOT / "feature_lineage.json")
    payload = _deepcopy(v31)
    payload["supersedes"] = {
        "audit_root": base.repo_path_label(V31_ROOT),
        "foundation_generation_sha256": V31_FOUNDATION_GENERATION_SHA256,
        "preregistration_sha256": V31_PREREGISTRATION_SHA256,
        "relationship": "GATE_HARDENING_SOURCE_POLICY_ONLY_SCIENCE_IDENTICAL",
    }
    pause = payload["foundation_correction"]["prefit_pause"]
    pause.update(
        {
            "claude_verification_pending": True,
            "corpus_decode_authorized": False,
            "foundation_stability_receipt_sealed": False,
            "foundation_stability_seal_authorized": False,
            "machinery_seal_authorized": False,
            "model_fit_authorized": False,
            "nested_or_outer_evidence_open_authorized": False,
            "protected_holdout_open_authorized": False,
            "separate_post_verification_release_required": True,
        }
    )
    source_hashes = _source_hashes(v31)
    payload["source_hashes_at_freeze"] = source_hashes
    policy = _deepcopy(v31["source_hash_policy"])
    immutable = list(policy["immutable_and_rehashed_at_every_fit"])
    for label in (*EXECUTABLE_SOURCE_PATHS, *EXECUTABLE_TEST_PATHS):
        if label not in immutable:
            immutable.append(label)
    policy["immutable_and_rehashed_at_every_fit"] = immutable
    owners = dict(policy["extensible_path_owners"])
    owners.update(
        {
            "v4/research/pathd_entry_execution_v32.py": "machinery",
            "v4/research/pathd_corrected_v32_foundation.py": "machinery",
        }
    )
    policy["extensible_path_owners"] = owners
    policy["authorized_in_repo_dependency_paths"] = list(
        base.AUTHORIZED_IN_REPO_DEPENDENCY_PATHS
    )
    policy["authorized_runtime_dependency_paths"] = list(
        base.AUTHORIZED_RUNTIME_DEPENDENCY_PATHS
    )
    policy["required_post_build"] = [
        {
            "path": label,
            "owner": owners[label],
            "existed_at_freeze": True,
            "freeze_sha256": source_hashes[label],
        }
        for label in owners
    ]
    policy["extensible_at_freeze"] = [
        {
            "path": label,
            "owner": owners[label],
            "freeze_sha256": source_hashes[label],
        }
        for label in owners
    ]
    policy["executable_generation_enforcement"] = {
        "schema_version": "pathd.corrected_v32.executable_generation_policy.v1",
        "foundation_root": base.repo_path_label(V32_ROOT),
        "source_paths": list(EXECUTABLE_SOURCE_PATHS),
        "test_paths": list(EXECUTABLE_TEST_PATHS),
        "required_receipts": [
            "executable_generation.json",
            "synthetic_test_receipt.json",
            "implementation_receipt.json",
        ],
        "gate_rule": "the exact six-file v3.2 foundation, executable generation, implementation receipt, and passing test receipt are rehashed before every fit authorization, corpus decode, evidence capability issue, evidence open, and estimator/calibrator call",
        "bootstrap_rule": "build-only bootstrap is allowed only while no v3.2 foundation file exists; any partial generation fails closed",
        "br_before_state_rule": "the immutable evidence API validates the exact VALID B-R scope receipt before lock creation, access-count write, capability issue, dataset receipt, decode, or result write",
        "claude_release_rule": "owner authorization is carried from v3.1 but fit/decode/evidence remain false until a separate Claude gate-hardening verification release",
    }
    payload["source_hash_policy"] = policy
    if _science_projection(payload) != _science_projection(v31):
        raise RuntimeError("corrected-v3.2 scientific contract drift")
    return payload, sessions, lineage


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": base.repo_path_label(path),
        "bytes": path.stat().st_size,
        "sha256": base.sha256_path(path),
    }


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_corrected_v32_foundation() -> dict[str, Any]:
    """Write the six v3.2 foundation files and leave every execution gate closed."""

    if V32_ROOT.exists():
        raise RuntimeError("corrected-v3.2 foundation root already exists")
    payload, sessions, lineage = corrected_v32_payload()
    V32_ROOT.mkdir(parents=True, exist_ok=False)
    prereg_path = V32_ROOT / "preregistration.json"
    session_path = V32_ROOT / "session_assignments.json"
    lineage_path = V32_ROOT / "feature_lineage.json"
    base._write_canonical_json_exclusive(prereg_path, payload)
    base._write_canonical_json_exclusive(session_path, sessions)
    base._write_canonical_json_exclusive(lineage_path, lineage)
    prereg_sha = base.sha256_path(prereg_path)
    _write_text_exclusive(
        V32_ROOT / "preregistration.sha256",
        f"{prereg_sha}  preregistration.json\n",
    )
    science = {
        "feature_lineage_sha256": base.sha256_path(lineage_path),
        "session_assignments_sha256": base.sha256_path(session_path),
        "entry17_sha256": base.stable_hash(payload["entry"]["feature_names"]),
        "exit47_sha256": base.stable_hash(payload["exit"]["feature_contract"]),
        "aref_sha256": base.stable_hash(
            payload["exit"]["aref_decision_critical_topology"]
        ),
        "calibration_sha256": base.stable_hash(payload["calibration_and_statistics"]),
        "br_sha256": base.stable_hash(payload["composite_calibration_terminal_rule"]),
    }
    freeze_semantic = {
        "schema_version": "pathd.entry_exit.corrected_v32_freeze_receipt.v1",
        "status": "FROZEN_GATE_HARDENING_PENDING_CLAUDE_VERIFICATION",
        "frozen_at_utc": _utc_now(),
        "release_audit_root": base.repo_path_label(V32_ROOT),
        "preregistration_sha256": prereg_sha,
        "source_hash_policy_sha256": base.stable_hash(payload["source_hash_policy"]),
        "science_identity_vs_corrected_v31": science,
        "supersedes": payload["supersedes"],
        "owner_v31_authorization_carried": True,
        "release_state": {
            "claude_verification_pending": True,
            "machinery_seal_authorized": False,
            "foundation_stability_seal_authorized": False,
            "corpus_decode_authorized": False,
            "model_fit_authorized": False,
            "nested_or_outer_evidence_open_authorized": False,
            "protected_holdout_open_authorized": False,
            "separate_post_verification_release_required": True,
        },
        "no_action_state": {
            "model_fit_executed": False,
            "corpus_decoded": False,
            "nested_or_outer_evidence_opened": False,
            "protected_holdout_opened": False,
            "live_or_broker_action_executed": False,
            "promotion_or_default_changed": False,
        },
        "holdout_open_count": 0,
        "terminal_marker": "STOP_FOR_CLAUDE_VERIFICATION",
    }
    freeze_receipt = {
        **freeze_semantic,
        "receipt_sha256": base.stable_hash(freeze_semantic),
    }
    freeze_path = V32_ROOT / "preregistration_freeze_receipt.json"
    base._write_canonical_json_exclusive(freeze_path, freeze_receipt)
    base_files = [
        _file_binding(path)
        for path in (
            prereg_path,
            V32_ROOT / "preregistration.sha256",
            session_path,
            lineage_path,
            freeze_path,
        )
    ]
    generation_semantic = {
        "schema_version": "pathd.entry_exit.corrected_v32_generation.v1",
        "audit_root": base.repo_path_label(V32_ROOT),
        "foundation_files": base_files,
        "superseded_v31_file_sha256s": dict(V31_FILE_SHA256S),
        "science_identity_vs_corrected_v31": science,
        "source_hash_policy_sha256": freeze_semantic["source_hash_policy_sha256"],
        "restored_outer_folds": [1, 2, 3, 4, 5],
        "fold_namespace_state": "PRISTINE_ABSENT",
        "holdout_open_count": 0,
    }
    generation_sha = base.stable_hash(generation_semantic)
    restoration_semantic = {
        "schema_version": "pathd.entry_exit.corrected_v32_restoration_receipt.v1",
        "status": "RESTORED_FIVE_FOLDS_GATE_HARDENING_NO_EXECUTION",
        "restored_at_utc": freeze_semantic["frozen_at_utc"],
        "foundation_generation_sha256": generation_sha,
        "corrected_foundation_files": base_files,
        "preserved_v31_file_sha256s": dict(V31_FILE_SHA256S),
        "fold_namespaces": [
            {
                "outer_fold": fold,
                "path": base.repo_path_label(V32_ROOT / "entry_outer_folds" / f"fold_{fold}"),
                "state": "PRISTINE_ABSENT",
            }
            for fold in range(1, 6)
        ],
        "release_state": freeze_semantic["release_state"],
        "no_action_state": freeze_semantic["no_action_state"],
        "holdout_open_count": 0,
        "terminal_marker": "STOP_FOR_CLAUDE_VERIFICATION",
    }
    restoration = {
        **restoration_semantic,
        "receipt_sha256": base.stable_hash(restoration_semantic),
    }
    base._write_canonical_json_exclusive(
        V32_ROOT / "foundation_restoration_receipt.json", restoration
    )
    return assert_corrected_v32_foundation()


def assert_corrected_v32_foundation() -> dict[str, Any]:
    """Reconstruct and byte-verify the complete frozen v3.2 foundation."""

    _assert_v31_bytes()
    expected_names = set(V31_FILE_SHA256S)
    if V32_ROOT.is_symlink() or not V32_ROOT.is_dir():
        raise RuntimeError("corrected-v3.2 foundation is absent")
    if {path.name for path in V32_ROOT.iterdir()} != expected_names:
        raise RuntimeError("corrected-v3.2 foundation inventory drift")
    payload, sessions, lineage = corrected_v32_payload()
    if base.read_json(V32_ROOT / "preregistration.json") != payload:
        raise RuntimeError("corrected-v3.2 preregistration reconstruction drift")
    if base.read_json(V32_ROOT / "session_assignments.json") != sessions:
        raise RuntimeError("corrected-v3.2 session assignment drift")
    if base.read_json(V32_ROOT / "feature_lineage.json") != lineage:
        raise RuntimeError("corrected-v3.2 feature lineage drift")
    prereg_sha = base.sha256_path(V32_ROOT / "preregistration.json")
    if (V32_ROOT / "preregistration.sha256").read_text(encoding="utf-8") != (
        f"{prereg_sha}  preregistration.json\n"
    ):
        raise RuntimeError("corrected-v3.2 preregistration hash sidecar drift")
    freeze = base.read_json(V32_ROOT / "preregistration_freeze_receipt.json")
    freeze_semantic = dict(freeze)
    freeze_digest = freeze_semantic.pop("receipt_sha256", None)
    if freeze_digest != base.stable_hash(freeze_semantic):
        raise RuntimeError("corrected-v3.2 freeze receipt self-hash drift")
    if (
        freeze.get("preregistration_sha256") != prereg_sha
        or freeze.get("status")
        != "FROZEN_GATE_HARDENING_PENDING_CLAUDE_VERIFICATION"
        or freeze.get("release_state", {}).get("claude_verification_pending") is not True
        or any(
            freeze.get("release_state", {}).get(key) is not False
            for key in (
                "machinery_seal_authorized",
                "foundation_stability_seal_authorized",
                "corpus_decode_authorized",
                "model_fit_authorized",
                "nested_or_outer_evidence_open_authorized",
                "protected_holdout_open_authorized",
            )
        )
        or freeze.get("holdout_open_count") != 0
    ):
        raise RuntimeError("corrected-v3.2 release state drift")
    restoration = base.read_json(V32_ROOT / "foundation_restoration_receipt.json")
    restoration_semantic = dict(restoration)
    restoration_digest = restoration_semantic.pop("receipt_sha256", None)
    if restoration_digest != base.stable_hash(restoration_semantic):
        raise RuntimeError("corrected-v3.2 restoration receipt self-hash drift")
    if any(
        (V32_ROOT / "entry_outer_folds" / f"fold_{fold}").exists()
        for fold in range(1, 6)
    ):
        raise RuntimeError("corrected-v3.2 fold namespace is not pristine")
    return {
        "preregistration_sha256": prereg_sha,
        "foundation_generation_sha256": restoration["foundation_generation_sha256"],
        "source_hash_policy_sha256": freeze["source_hash_policy_sha256"],
        "science_identity_vs_corrected_v31": freeze[
            "science_identity_vs_corrected_v31"
        ],
        "release_state": freeze["release_state"],
        "holdout_open_count": 0,
    }
