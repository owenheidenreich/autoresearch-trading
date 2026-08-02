"""Sealed causal entry examples and datasets for the Path-D Tier-S study.

The public readers hash every authorized source byte before invoking a decoder.  The
module intentionally has no broker, network, registry, or runtime dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v4.path_d.contracts import ContractIdentityV1
from v4.research import pathd_entry_exit as prereg
from v4.research.pathd_entry_features import (
    EntrySnapshotV1,
    Signed17HistoryV1,
    build_identity_joined_history,
    hgb_signed17_summaries,
    historical_snapshot_from_processed_row,
    official_spx_market_window_from_rows,
    signed17_from_snapshot,
)


CORPUS_ROOT = prereg.CORPUS_ROOT
assert_fit_authorization_current = prereg.assert_fit_authorization_current
assert_entry_evidence_authorization_current = (
    prereg.assert_entry_evidence_authorization_current
)

CORRECTED_V32_EXECUTABLE_ROOT = prereg.REPO_ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_2_"
    "executable_2026_08_01"
)
CORRECTED_V32_EXECUTABLE_GENERATION_PATH = (
    CORRECTED_V32_EXECUTABLE_ROOT / "executable_generation.json"
)
CORRECTED_V32_EXECUTABLE_BRIDGE_PATH = (
    CORRECTED_V32_EXECUTABLE_ROOT / "implementation_receipt.json"
)
CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH = (
    CORRECTED_V32_EXECUTABLE_ROOT / "registered_test_reconciliation_authorization.json"
)
CORRECTED_V32_TEST_RECONCILIATION_RECEIPT_PATH = (
    CORRECTED_V32_EXECUTABLE_ROOT / "registered_test_reconciliation_receipt.json"
)
CORRECTED_V32_CLAUDE_RELEASE_PATH = (
    CORRECTED_V32_EXECUTABLE_ROOT / "claude_verification_release.json"
)
CORRECTED_V32_REGISTERED_TEST_PATHS = (
    "v4/tests/test_pathd_entry_exit_gate_frozen.py",
    "v4/tests/test_pathd_entry_exit_research.py",
    "v4/tests/test_pathd_evidence_gate.py",
    "v4/tests/test_pathd_holdout_gate.py",
    "v4/tests/test_pathd_fixed_science_contract.py",
    "v4/tests/test_pathd_feature_live_twin.py",
    "v4/tests/test_pathd_evidence_gate_capability_hardening.py",
    "v4/tests/test_pathd_foundation_correction.py",
    "v4/tests/test_pathd_corrected_v32_executable_build.py",
)
CORRECTED_V32_RECONCILER_SOURCE_PATH = (
    "v4/research/pathd_v32_test_reconciliation.py"
)


def _validate_self_hashed_mapping(value: dict[str, Any], field: str) -> None:
    semantic = dict(value)
    digest = semantic.pop(field, None)
    if digest != prereg.stable_hash(semantic):
        raise RuntimeError(f"corrected-v3.2 {field} self-hash drift")


def _v32_foundation_from_frozen_generation(
    generation: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = prereg.REPO_ROOT / generation["foundation_root"]
    expected_names = {
        "feature_lineage.json", "foundation_restoration_receipt.json",
        "preregistration.json", "preregistration.sha256",
        "preregistration_freeze_receipt.json", "session_assignments.json",
    }
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != expected_names
        or prereg.sha256_path(root / "preregistration.json")
        != generation["foundation_preregistration_sha256"]
    ):
        raise RuntimeError("corrected-v3.2 frozen foundation byte inventory drift")
    restoration = prereg.read_json(root / "foundation_restoration_receipt.json")
    freeze = prereg.read_json(root / "preregistration_freeze_receipt.json")
    _validate_self_hashed_mapping(restoration, "receipt_sha256")
    _validate_self_hashed_mapping(freeze, "receipt_sha256")
    if (
        restoration.get("foundation_generation_sha256")
        != generation["foundation_generation_sha256"]
        or freeze.get("preregistration_sha256")
        != generation["foundation_preregistration_sha256"]
        or freeze.get("source_hash_policy_sha256")
        != generation["source_hash_policy_sha256"]
    ):
        raise RuntimeError("corrected-v3.2 frozen foundation semantic drift")
    return root, prereg.read_json(root / "preregistration.json")


def _validate_v32_test_reconciliation(
    *, preregistration: dict[str, Any], frozen_implementations: list[dict[str, str]],
    current_implementations: list[dict[str, str]], base_bridge_sha256: str,
    foundation_generation_sha256: str, source_hash_policy_sha256: str,
) -> dict[str, Any]:
    authorization = prereg.read_json(
        CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH
    )
    _validate_self_hashed_mapping(authorization, "receipt_sha256")
    source_changes = [
        {"path": frozen["path"], "frozen_sha256": frozen["sha256"], "current_sha256": current["sha256"]}
        for frozen, current in zip(
            frozen_implementations, current_implementations, strict=True
        )
        if frozen != current
    ]
    test_labels = authorization.get("registered_test_paths")
    if test_labels != list(CORRECTED_V32_REGISTERED_TEST_PATHS):
        raise RuntimeError("corrected-v3.2 reconciliation test registry drift")
    test_changes = []
    for label in test_labels:
        frozen = preregistration["source_hashes_at_freeze"].get(label)
        current = prereg.sha256_path(prereg.REPO_ROOT / label)
        if frozen != current:
            test_changes.append(
                {"path": label, "frozen_sha256": frozen, "current_sha256": current}
            )
    if (
        authorization.get("schema_version")
        != "pathd.corrected_v32.registered_test_reconciliation_authorization.v1"
        or authorization.get("status")
        != "AUTHORIZED_TEST_ONLY_UNRELEASED_NO_EXECUTION"
        or authorization.get("base_implementation_receipt_sha256")
        != base_bridge_sha256
        or authorization.get("foundation_generation_sha256")
        != foundation_generation_sha256
        or authorization.get("source_hash_policy_sha256")
        != source_hash_policy_sha256
        or authorization.get("reconciler_source_path")
        != CORRECTED_V32_RECONCILER_SOURCE_PATH
        or authorization.get("reconciler_source_sha256")
        != prereg.sha256_path(prereg.REPO_ROOT / CORRECTED_V32_RECONCILER_SOURCE_PATH)
        or authorization.get("source_changes") != source_changes
        or authorization.get("test_changes") != test_changes
        or any(authorization.get(key) is not False for key in (
            "model_fit_executed", "corpus_decoded", "evidence_opened",
            "foundation_or_machinery_sealed_against_corpus",
            "holdout_opened", "live_or_broker_action_executed",
        ))
        or authorization.get("holdout_open_count") != 0
    ):
        raise RuntimeError("corrected-v3.2 test reconciliation authorization drift")
    if not CORRECTED_V32_TEST_RECONCILIATION_RECEIPT_PATH.exists():
        return {
            "schema_version": authorization["schema_version"],
            "status": "TEST_RECONCILIATION_AUTHORIZED_UNRELEASED",
            "foundation_generation_sha256": authorization[
                "foundation_generation_sha256"
            ],
            "test_reconciliation_authorization_sha256": prereg.sha256_path(
                CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH
            ),
            "holdout_open_count": 0,
        }
    receipt = prereg.read_json(CORRECTED_V32_TEST_RECONCILIATION_RECEIPT_PATH)
    _validate_self_hashed_mapping(receipt, "receipt_sha256")
    junit_path = prereg.REPO_ROOT / receipt.get("junit_path", "")
    if not junit_path.is_file() or junit_path.is_symlink():
        raise RuntimeError("corrected-v3.2 reconciliation JUnit is absent or unsafe")
    suite = ET.parse(junit_path).getroot()
    cases = suite.findall(".//testcase")
    failures = sum(len(case.findall("failure")) for case in cases)
    errors = sum(len(case.findall("error")) for case in cases)
    skipped = sum(len(case.findall("skipped")) for case in cases)
    if (
        receipt.get("schema_version")
        != "pathd.corrected_v32.registered_test_reconciliation_receipt.v1"
        or receipt.get("status") != "PASS_FULL_REGISTERED_PATHD_SUITE_UNRELEASED"
        or receipt.get("authorization_sha256")
        != prereg.sha256_path(CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH)
        or receipt.get("source_changes") != source_changes
        or receipt.get("test_changes") != test_changes
        or receipt.get("registered_test_paths")
        != list(CORRECTED_V32_REGISTERED_TEST_PATHS)
        or receipt.get("junit_sha256") != prereg.sha256_path(junit_path)
        or receipt.get("summary")
        != {
            "tests": len(cases), "passed": len(cases), "failures": failures,
            "errors": errors, "skipped": skipped,
        }
        or failures != 0 or errors != 0 or skipped != 0
        or any(receipt.get(key) is not False for key in (
            "model_fit_executed", "corpus_decoded", "evidence_opened",
            "foundation_or_machinery_sealed_against_corpus",
            "holdout_opened", "live_or_broker_action_executed",
        ))
        or receipt.get("holdout_open_count") != 0
    ):
        raise RuntimeError("corrected-v3.2 registered test reconciliation drift")
    return {
        **receipt,
        "foundation_generation_sha256": authorization[
            "foundation_generation_sha256"
        ],
    }


def _validate_v32_released_reconciliation(
    *, generation: dict[str, Any], preregistration: dict[str, Any],
    current_implementations: list[dict[str, str]], base_bridge_sha256: str,
) -> dict[str, Any]:
    """Bind post-verification source/test corrections without mutating v3.2."""

    release = prereg.read_json(CORRECTED_V32_CLAUDE_RELEASE_PATH)
    _validate_self_hashed_mapping(release, "receipt_sha256")
    current_tests = [
        {"path": label, "sha256": prereg.sha256_path(prereg.REPO_ROOT / label)}
        for label in CORRECTED_V32_REGISTERED_TEST_PATHS
    ]
    pause = release.get("foundation_correction", {}).get("prefit_pause", {})
    if (
        release.get("schema_version")
        != "pathd.corrected_v32.claude_fit_release.v1"
        or release.get("status")
        != "CLAUDE_VERIFIED_GATE_HARDENING_FIT_RELEASE"
        or release.get("verification_outcome") != "PASS"
        or release.get("foundation_generation_sha256")
        != generation["foundation_generation_sha256"]
        or release.get("preregistration_sha256")
        != generation["foundation_preregistration_sha256"]
        or release.get("source_hash_policy_sha256")
        != generation["source_hash_policy_sha256"]
        or release.get("implementation_receipt_sha256") != base_bridge_sha256
        or release.get("prior_test_reconciliation_authorization_sha256")
        != prereg.sha256_path(CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH)
        or release.get("prior_test_reconciliation_receipt_sha256")
        != prereg.sha256_path(CORRECTED_V32_TEST_RECONCILIATION_RECEIPT_PATH)
        or release.get("released_implementations") != current_implementations
        or release.get("released_registered_tests") != current_tests
        or any(release.get(key) is not True for key in (
            "machinery_seal_authorized", "foundation_stability_seal_authorized",
            "corpus_decode_authorized", "model_fit_authorized",
            "nested_or_outer_evidence_open_authorized",
        ))
        or release.get("protected_holdout_open_authorized") is not False
        or release.get("live_or_broker_action_authorized") is not False
        or release.get("holdout_open_count") != 0
        or pause.get("foundation_stability_gate_implemented") is not True
        or pause.get("machinery_seal_authorized") is not True
        or pause.get("foundation_stability_seal_authorized") is not True
        or pause.get("model_fit_authorized") is not True
        or pause.get("corpus_decode_authorized") is not True
        or pause.get("nested_or_outer_evidence_open_authorized") is not True
        or pause.get("protected_holdout_open_authorized") is not False
        or pause.get("claude_verification_pending") is not False
        or pause.get("separate_post_verification_release_required") is not False
        or any(release.get(key) is not False for key in (
            "model_fit_executed", "corpus_decoded", "evidence_opened",
            "foundation_or_machinery_sealed_against_corpus", "holdout_opened",
            "live_or_broker_action_executed",
        ))
    ):
        raise RuntimeError("corrected-v3.2 released reconciliation drift")
    return release


def assert_corrected_v32_executable_bridge() -> dict[str, Any]:
    """Rehash the v3.2 foundation and implementation chain before decode."""

    generation = prereg.read_json(CORRECTED_V32_EXECUTABLE_GENERATION_PATH)
    generation_semantic = dict(generation)
    generation_digest = generation_semantic.pop("generation_sha256", None)
    if generation_digest != prereg.stable_hash(generation_semantic):
        raise RuntimeError("corrected-v3.2 executable generation drift")
    V32_ROOT, preregistration = _v32_foundation_from_frozen_generation(generation)
    if not V32_ROOT.exists() and not CORRECTED_V32_EXECUTABLE_ROOT.exists():
        return {"status": "PRE_V32_FOUNDATION_BUILD_ONLY"}
    if (
        not CORRECTED_V32_EXECUTABLE_GENERATION_PATH.is_file()
        or CORRECTED_V32_EXECUTABLE_GENERATION_PATH.is_symlink()
        or not CORRECTED_V32_EXECUTABLE_BRIDGE_PATH.is_file()
        or CORRECTED_V32_EXECUTABLE_BRIDGE_PATH.is_symlink()
    ):
        raise RuntimeError("corrected-v3.2 executable generation is incomplete")
    policy = preregistration["source_hash_policy"]
    enforcement = policy["executable_generation_enforcement"]
    frozen_implementations = [
        {"path": label, "sha256": preregistration["source_hashes_at_freeze"][label]}
        for label in enforcement["source_paths"]
    ]
    current_implementations = [
        {"path": label, "sha256": prereg.sha256_path(prereg.REPO_ROOT / label)}
        for label in enforcement["source_paths"]
    ]
    test_path = CORRECTED_V32_EXECUTABLE_ROOT / "synthetic_test_receipt.json"
    test_receipt = prereg.read_json(test_path)
    test_semantic = dict(test_receipt)
    test_digest = test_semantic.pop("receipt_sha256", None)
    receipt = prereg.read_json(CORRECTED_V32_EXECUTABLE_BRIDGE_PATH)
    semantic = dict(receipt)
    digest = semantic.pop("receipt_sha256", None)
    if (
        generation_digest != prereg.stable_hash(generation_semantic)
        or generation.get("foundation_generation_sha256")
        != prereg.read_json(V32_ROOT / "foundation_restoration_receipt.json")["foundation_generation_sha256"]
        or generation.get("source_hash_policy_sha256")
        != prereg.stable_hash(preregistration["source_hash_policy"])
        or test_digest != prereg.stable_hash(test_semantic)
        or test_receipt.get("status") != "PASS_SYNTHETIC_PRODUCTION_PATH"
        or test_receipt.get("pytest_exit_code") != 0
        or receipt.get("schema_version")
        != "pathd.corrected_v32.executable_implementation_receipt.v1"
        or receipt.get("status") != "FROZEN_EXECUTABLE_BUILT_NOT_RUN"
        or receipt.get("foundation_generation_sha256")
        != generation["foundation_generation_sha256"]
        or receipt.get("executable_generation_sha256") != generation_digest
        or receipt.get("implementations") != frozen_implementations
        or receipt.get("implementations_sha256") != prereg.stable_hash(frozen_implementations)
        or receipt.get("synthetic_test_receipt_sha256") != prereg.sha256_path(test_path)
        or any(receipt.get(key) is not False for key in (
            "model_fit_executed", "corpus_decoded", "evidence_opened",
        ))
        or receipt.get("holdout_open_count") != 0
        or digest != prereg.stable_hash(semantic)
    ):
        raise RuntimeError("corrected-v3.2 executable bridge drift")
    if current_implementations != frozen_implementations:
        if CORRECTED_V32_CLAUDE_RELEASE_PATH.exists():
            return _validate_v32_released_reconciliation(
                generation=generation,
                preregistration=preregistration,
                current_implementations=current_implementations,
                base_bridge_sha256=prereg.sha256_path(
                    CORRECTED_V32_EXECUTABLE_BRIDGE_PATH
                ),
            )
        return _validate_v32_test_reconciliation(
            preregistration=preregistration,
            frozen_implementations=frozen_implementations,
            current_implementations=current_implementations,
            base_bridge_sha256=prereg.sha256_path(CORRECTED_V32_EXECUTABLE_BRIDGE_PATH),
            foundation_generation_sha256=generation["foundation_generation_sha256"],
            source_hash_policy_sha256=generation["source_hash_policy_sha256"],
        )
    return receipt



_MINUTE_NS = 60_000_000_000
_SECOND_NS = 1_000_000_000
_ENTRY_SOURCE_PATHS = (
    "aligned/processed/minute_entry/{session}.pkl",
    "raw/databento/opra_spxw_cbbo_1s/{session}.cbbo-1s.parquet",
    "vendor/thetadata/index/spx_1m/{session}.parquet",
)
_PROCESSED_PREFIX = "aligned/processed/minute_entry/"
_RAW_CBBO_1S_PREFIX = "raw/databento/opra_spxw_cbbo_1s/"
_OFFICIAL_SPX_PREFIX = "vendor/thetadata/index/spx_1m/"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": _jsonable(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "+inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _exact_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _contract_dict(contract: ContractIdentityV1) -> dict[str, Any]:
    if type(contract) is not ContractIdentityV1:
        raise TypeError("contract must be ContractIdentityV1")
    return {field.name: getattr(contract, field.name) for field in fields(contract)}


def _contract_for_action(
    *, session: str, strike: float, right: str
) -> ContractIdentityV1:
    expiry_compact = session.replace("-", "")
    strike_milli = int(round(strike * 1_000.0))
    osi = f"SPXW  {expiry_compact[2:]}{right}{strike_milli:08d}"
    return ContractIdentityV1(
        osi_symbol=osi,
        underlying="SPX",
        trading_class="SPXW",
        expiry=session,
        strike_milli=strike_milli,
        right=right,
    )


def _timestamp_ns(value: Any, *, name: str) -> int:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def _source_neutral_contract_id(contract: ContractIdentityV1) -> str:
    if type(contract) is not ContractIdentityV1:
        raise TypeError("verified selector contract must be ContractIdentityV1")
    return (
        f"SPXW-{contract.expiry.replace('-', '')}-"
        f"{contract.strike_milli / 1_000.0:09.3f}-{contract.right}"
    )


def _osi_symbol_from_source_neutral(value: str, *, session: str) -> str:
    match = re.fullmatch(
        r"SPXW-(\d{8})-(\d{5}\.\d{3})-([CP])", str(value)
    )
    if match is None or match.group(1) != session.replace("-", ""):
        raise ValueError("source-neutral contract identity drift")
    strike_milli = int(round(float(match.group(2)) * 1_000.0))
    return f"SPXW  {match.group(1)[2:]}{match.group(3)}{strike_milli:08d}"


def _manifest_source_rows_for_session(session: str) -> tuple[dict[str, Any], ...]:
    if type(session) is not str or re.fullmatch(r"\d{4}-\d{2}-\d{2}", session) is None:
        raise RuntimeError("authorized entry session identity is malformed")
    manifest = {
        row["relative_path"]: row
        for row in prereg.integrity_manifest_entries_for_partition(
            prereg.CORE_INTEGRITY_PARTITION
        )
    }
    expected = [template.format(session=session) for template in _ENTRY_SOURCE_PATHS]
    if len(expected) != len(set(expected)):
        raise RuntimeError("entry source path contract is duplicated")
    rows: list[dict[str, Any]] = []
    for relative_path in expected:
        row = manifest.get(relative_path)
        if type(row) is not dict or set(row) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError(
                f"authorized core source is absent from exact manifest: {relative_path}"
            )
        rows.append(dict(row))
    rows.sort(key=lambda row: row["relative_path"])
    return tuple(rows)


def _receipt_source_rows(rows: Any) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        relative_path = row.get("relative_path")
        size = row.get("bytes", row.get("size"))
        if (
            type(relative_path) is not str
            or type(size) is not int
            or size < 0
        ):
            raise RuntimeError("verified source row schema drift")
        digest = _exact_sha256(row.get("sha256"), name="verified source sha256")
        normalized.append(
            {"relative_path": relative_path, "bytes": size, "sha256": digest}
        )
    normalized.sort(key=lambda row: row["relative_path"])
    if len(normalized) != len({row["relative_path"] for row in normalized}):
        raise RuntimeError("verified source row duplication")
    return tuple(normalized)


@dataclass(frozen=True)
class EntryFuturePathV1:
    SCHEMA_VERSION = "pathd.entry_future_path.v1"

    schema_version: str
    arrival_time_ns: int
    arrival_contract_id: str
    arrival_bid_micros: int
    arrival_ask_micros: int
    arrival_watermark_ns: int
    marks_by_horizon: dict[str, list[int]]
    future_audit_nonce: str


@dataclass(frozen=True)
class EntryActionExecutionFactV1:
    SCHEMA_VERSION = "pathd.entry_action_execution_fact.v1"

    schema_version: str
    session: str
    decision_time_ns: int
    source_neutral_contract_id: str
    contract: ContractIdentityV1
    represented_interval_end_ns: int
    ts_recv_ns: int
    available_at_ns: int
    bid_micros: int
    ask_micros: int
    source_receipt_sha256: str
    source_record_sha256: str
    physical_eligible: bool
    fact_sha256: str


@dataclass(frozen=True)
class EntryExampleV1:
    SCHEMA_VERSION = "pathd.entry_example.v1"

    schema_version: str
    model_input: Any
    action_execution_facts: tuple[EntryActionExecutionFactV1, ...]
    targets: dict[str, float]
    target_validity: dict[str, bool]
    audit: dict[str, Any]

    def canonical_sha256(self) -> str:
        return _stable_hash(self)


@dataclass(frozen=True)
class EntrySessionV1:
    SCHEMA_VERSION = "pathd.entry_session.v1"

    schema_version: str
    session: str
    source_files: tuple[dict[str, Any], ...]
    source_files_sha256: str
    examples: tuple[EntryExampleV1, ...]
    content_sha256: str


@dataclass(frozen=True)
class EntryFitDatasetV1:
    SCHEMA_VERSION = "pathd.entry_fit_dataset.v1"

    schema_version: str
    authorization_sha256: str
    role: str
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    source_receipts: tuple[dict[str, Any], ...]
    examples: tuple[EntryExampleV1, ...]
    dataset_sha256: str


@dataclass(frozen=True)
class EntryEvidenceDatasetV1:
    SCHEMA_VERSION = "pathd.entry_evidence_dataset.v1"

    schema_version: str
    authorization_sha256: str
    role: str
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    source_receipts: tuple[dict[str, Any], ...]
    examples: tuple[EntryExampleV1, ...]
    dataset_sha256: str


# A fitting command intentionally reuses one already-decoded, sealed role dataset
# across the positive family and its mandatory controls.  Every public cache hit
# still rehashes every authorized source byte; only the expensive deterministic
# decode and history materialization are reused.  Evidence/holdout capabilities
# are never cached here.
_FIT_DATASET_CACHE: dict[str, tuple[Any, EntryFitDatasetV1]] = {}


@dataclass(frozen=True)
class VerifiedResearchQuoteRowV1:
    SCHEMA_VERSION = "pathd.verified_research_quote_row.v1"

    schema_version: str
    session: str
    source_relative_path: str
    source_file_sha256: str
    row_group: int
    row_index: int
    canonical_row_sha256: str
    contract: ContractIdentityV1
    source_vendor: str
    represented_interval_end_ns: int
    ts_recv_ns: int
    available_at_ns: int
    bid_micros: int
    ask_micros: int
    source_receipt_sha256: str
    query_at_or_before_ns: int
    query_maximum_age_ms: int
    query_require_actionable: bool
    eligible_row_count: int
    selection_key: tuple[Any, ...]
    selection_proof_sha256: str
    record_sha256: str


@dataclass(frozen=True)
class VerifiedOfficialSpxRowV1:
    SCHEMA_VERSION = "pathd.verified_official_spx_row.v1"

    schema_version: str
    session: str
    source_relative_path: str
    source_file_sha256: str
    row_group: int
    row_index: int
    canonical_row_sha256: str
    source_vendor: str
    represented_interval_end_ns: int
    ts_recv_ns: int
    available_at_ns: int
    spx_micros: int
    source_receipt_sha256: str
    query_at_or_before_ns: int
    query_maximum_age_ms: int
    eligible_row_count: int
    selection_key: tuple[Any, ...]
    selection_proof_sha256: str
    record_sha256: str


@dataclass(frozen=True)
class _VerifiedEntrySessionRead:
    session: str
    source_files: tuple[dict[str, Any], ...]
    payload: Any


@dataclass(frozen=True)
class _RawQuoteSeries:
    timestamps_ns: np.ndarray
    rows: tuple[tuple[int, int, int, int, int], ...]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1 or len(timestamps) != len(self.rows):
            raise ValueError("raw quote series geometry drift")
        if len(timestamps) and np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("raw quote series clock order drift")
        timestamps = np.array(timestamps, copy=True)
        timestamps.setflags(write=False)
        object.__setattr__(self, "timestamps_ns", timestamps)


def _validate_future_path(path: EntryFuturePathV1, *, decision_time_ns: int) -> None:
    if type(path) is not EntryFuturePathV1 or path.schema_version != path.SCHEMA_VERSION:
        raise TypeError("future path must be exact EntryFuturePathV1")
    if type(path.arrival_time_ns) is not int or path.arrival_time_ns != decision_time_ns + 60_000_000_000:
        raise ValueError("entry arrival must be the next completed minute")
    if type(path.arrival_watermark_ns) is not int or path.arrival_watermark_ns > path.arrival_time_ns:
        raise ValueError("future path arrival watermark drift")
    if not isinstance(path.arrival_contract_id, str) or not path.arrival_contract_id:
        raise ValueError("future path contract identity is missing")
    if type(path.arrival_bid_micros) is not int or type(path.arrival_ask_micros) is not int:
        raise TypeError("arrival quote must use exact integer micros")
    if path.arrival_bid_micros < 0 or path.arrival_ask_micros < 0:
        raise ValueError("arrival quote cannot be negative")
    if list(path.marks_by_horizon) != list(prereg.ENTRY_FUTURE_PATH_KEYS):
        raise ValueError("future path horizon keys/order drift")
    for key, values in path.marks_by_horizon.items():
        if type(values) is not list or any(type(value) is not int for value in values):
            raise TypeError(f"future marks for {key} must be exact integer list")
        if any(value < 0 for value in values):
            raise ValueError("future executable mark cannot be negative")
    if not isinstance(path.future_audit_nonce, str) or not path.future_audit_nonce:
        raise ValueError("future audit nonce is mandatory")


def available_entry_horizons(
    *, decision_time_ns: int, terminal_time_ns: int
) -> tuple[str, ...]:
    if type(decision_time_ns) is not int or type(terminal_time_ns) is not int:
        raise TypeError("entry horizon clocks must be exact integers")
    arrival = decision_time_ns + 60_000_000_000
    if arrival >= terminal_time_ns:
        return ()
    rows: list[str] = []
    for horizon in (10, 20, 45, 90):
        if arrival + horizon * 60_000_000_000 <= terminal_time_ns:
            rows.append(f"h{horizon}")
    rows.append("remaining_session")
    return tuple(rows)


def entry_targets_from_executable_marks(
    *,
    entry_fill_price_micros: int,
    round_trip_fee_micros: int,
    marks_by_horizon: Any,
    available_horizons: Any,
) -> dict[str, float]:
    if type(entry_fill_price_micros) is not int or entry_fill_price_micros <= 0:
        raise ValueError("entry fill price must be positive integer option-price micros")
    if type(round_trip_fee_micros) is not int or round_trip_fee_micros < 0:
        raise ValueError("round-trip fee must be nonnegative cash micros")
    if type(marks_by_horizon) is not dict:
        raise TypeError("marks_by_horizon must be an exact mapping")
    horizons = tuple(available_horizons)
    if len(set(horizons)) != len(horizons):
        raise ValueError("available horizons must be unique")
    if any(value not in prereg.ENTRY_FUTURE_PATH_KEYS for value in horizons):
        raise ValueError("unknown entry horizon")
    premium_dollars = entry_fill_price_micros * 100.0 / 1_000_000.0
    result: dict[str, float] = {}
    for horizon in horizons:
        marks = marks_by_horizon.get(horizon)
        if type(marks) is not list or not marks:
            raise ValueError(f"available horizon {horizon} has no full mark window")
        if any(type(value) is not int or value < 0 for value in marks):
            raise TypeError("executable marks must be nonnegative integer micros")
        pnl = np.asarray(
            [
                ((value - entry_fill_price_micros) * 100 - round_trip_fee_micros)
                / 1_000_000.0
                for value in marks
            ],
            dtype=np.float64,
        )
        mfe = float(np.max(pnl))
        area = float(np.maximum(pnl, 0.0).sum())
        prefix = "session" if horizon == "remaining_session" else horizon
        result[f"{prefix}_mfe_dollars"] = mfe
        result[f"{prefix}_mfe_return"] = mfe / premium_dollars
        result[f"{prefix}_profit_area_dollars"] = area
        result[f"{prefix}_profit_area_return"] = area / premium_dollars
    return result


def _terminal_time_ns(session: str) -> int:
    return int(
        pd.Timestamp(f"{session} 15:55:00", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )


def _action_execution_facts(snapshot: EntrySnapshotV1) -> tuple[EntryActionExecutionFactV1, ...]:
    try:
        bid_index = snapshot.option_feature_names.index("bid")
        ask_index = snapshot.option_feature_names.index("ask")
    except ValueError as exc:
        raise ValueError("entry action facts require bid and ask") from exc
    rows: list[EntryActionExecutionFactV1] = []
    lineage_root = _stable_hash(snapshot.declared_alpha_sources)
    for strike_index, offset in enumerate(snapshot.strike_offsets):
        for right_index, right in enumerate(snapshot.rights):
            identity = str(snapshot.contract_ids[strike_index, right_index])
            contract = _contract_for_action(
                session=snapshot.session,
                strike=snapshot.atm_strike + float(offset),
                right=right,
            )
            bid_value = float(snapshot.option_ladder[strike_index, right_index, bid_index])
            ask_value = float(snapshot.option_ladder[strike_index, right_index, ask_index])
            bid_micros = int(round(bid_value * 1_000_000.0)) if math.isfinite(bid_value) else 0
            ask_micros = int(round(ask_value * 1_000_000.0)) if math.isfinite(ask_value) else 0
            available_at = int(snapshot.contract_quote_watermark_ns[strike_index, right_index])
            eligible = bool(
                math.isfinite(bid_value)
                and math.isfinite(ask_value)
                and bid_micros > 0
                and ask_micros > bid_micros
                and ask_micros >= 1_000_000
                and available_at <= snapshot.decision_time_ns
                and snapshot.decision_time_ns - available_at <= 90_000_000_000
            )
            semantic = {
                "schema_version": EntryActionExecutionFactV1.SCHEMA_VERSION,
                "session": snapshot.session,
                "decision_time_ns": snapshot.decision_time_ns,
                "source_neutral_contract_id": identity,
                "contract": _contract_dict(contract),
                "represented_interval_end_ns": available_at,
                "ts_recv_ns": available_at,
                "available_at_ns": available_at,
                "bid_micros": bid_micros,
                "ask_micros": ask_micros,
                "source_receipt_sha256": lineage_root,
                "source_record_sha256": _stable_hash(
                    {
                        "identity": identity,
                        "available_at_ns": available_at,
                        "bid_micros": bid_micros,
                        "ask_micros": ask_micros,
                    }
                ),
                "physical_eligible": eligible,
            }
            rows.append(
                EntryActionExecutionFactV1(
                    **{**semantic, "contract": contract},
                    fact_sha256=_stable_hash(semantic),
                )
            )
    if len(rows) != 42:
        raise ValueError("entry action fact ladder must contain exactly 42 actions")
    return tuple(rows)


def build_entry_example(
    snapshot: Any, /, *, future_path: Any, fill_law: Any
) -> EntryExampleV1:
    if type(snapshot) is not EntrySnapshotV1:
        raise TypeError("entry example requires EntrySnapshotV1")
    _validate_future_path(future_path, decision_time_ns=snapshot.decision_time_ns)
    if getattr(fill_law, "fill_law_hash", None) != prereg.fill_law()["fill_law_hash"]:
        raise ValueError("entry example fill-law drift")
    facts = _action_execution_facts(snapshot)
    identities = [fact.source_neutral_contract_id for fact in facts]
    if future_path.arrival_contract_id not in identities:
        raise ValueError("future path contract is not in the exact action ladder")
    chosen = facts[identities.index(future_path.arrival_contract_id)]
    from v4.path_d.execution.research_fill_law import option_tick_micros
    from v4.research.pathd_entry_models import EntryModelInputV1

    hard_limit = chosen.ask_micros + option_tick_micros(chosen.ask_micros)
    filled = bool(
        chosen.physical_eligible
        and future_path.arrival_bid_micros > 0
        and future_path.arrival_ask_micros > future_path.arrival_bid_micros
        and future_path.arrival_ask_micros <= hard_limit
    )
    terminal = _terminal_time_ns(snapshot.session)
    gate_horizons = available_entry_horizons(
        decision_time_ns=snapshot.decision_time_ns,
        terminal_time_ns=terminal,
    )
    diagnostic = tuple(
        f"h{value}"
        for value in prereg.DIAGNOSTIC_HORIZONS
        if future_path.arrival_time_ns + value * 60_000_000_000 <= terminal
    )
    all_available = (*diagnostic, *gate_horizons)
    expected_lengths = {
        "h3": 3,
        "h5": 5,
        "h10": 10,
        "h20": 20,
        "h45": 45,
        "h90": 90,
        "remaining_session": (terminal - future_path.arrival_time_ns)
        // 60_000_000_000,
    }
    for key, expected in expected_lengths.items():
        observed = len(future_path.marks_by_horizon[key])
        if key in all_available:
            if observed != expected:
                raise ValueError(f"future path {key} length drift: {observed} != {expected}")
        elif observed != 0:
            raise ValueError(f"unavailable future path {key} must be empty")
    if filled:
        targets = entry_targets_from_executable_marks(
            entry_fill_price_micros=hard_limit,
            round_trip_fee_micros=fill_law.entry_fee_micros + fill_law.exit_fee_micros,
            marks_by_horizon=future_path.marks_by_horizon,
            available_horizons=all_available,
        )
    else:
        keys: list[str] = []
        for horizon in all_available:
            prefix = "session" if horizon == "remaining_session" else horizon
            keys.extend(
                (
                    f"{prefix}_mfe_dollars",
                    f"{prefix}_mfe_return",
                    f"{prefix}_profit_area_dollars",
                    f"{prefix}_profit_area_return",
                )
            )
        targets = {key: 0.0 for key in keys}
    validity = {key: True for key in targets}
    frame = signed17_from_snapshot(snapshot)
    history = build_identity_joined_history(
        [(snapshot.session, frame)],
        current_session=snapshot.session,
        current_decision_time_ns=snapshot.decision_time_ns,
        current_contract_ids=snapshot.contract_ids,
        history_minutes=90,
    )
    offsets = np.repeat(snapshot.strike_offsets, len(snapshot.rights)).astype(np.float64)
    rights = np.tile(np.asarray(snapshot.rights, dtype=object), len(snapshot.strike_offsets))
    model_input = EntryModelInputV1(
        schema_version=EntryModelInputV1.SCHEMA_VERSION,
        session=snapshot.session,
        decision_time_ns=snapshot.decision_time_ns,
        signed17_frame=frame,
        signed17_history=history,
        hgb_summaries=hgb_signed17_summaries(
            history, current_offsets=offsets, current_rights=rights
        ),
        current_offsets=offsets,
        current_rights=tuple(rights.tolist()),
        contract_ids=tuple(identities),
        physical_action_mask=np.asarray(
            [fact.physical_eligible for fact in facts], dtype=np.bool_
        ),
    )
    return EntryExampleV1(
        schema_version=EntryExampleV1.SCHEMA_VERSION,
        model_input=model_input,
        action_execution_facts=facts,
        targets=targets,
        target_validity=validity,
        audit={
            "future_audit_nonce": future_path.future_audit_nonce,
            "arrival_contract_id": future_path.arrival_contract_id,
            "entry_filled": filled,
            "submitted_hard_limit_micros": hard_limit,
            "available_horizons": list(all_available),
            "fill_law_hash": fill_law.fill_law_hash,
        },
    )


def canonical_entry_dataset_sha256(
    *,
    schema_version: str,
    authorization_sha256: str,
    role: str,
    sessions: Any,
    sessions_sha256_newline: str,
    source_receipts: Any,
    examples: Any,
) -> str:
    ordered_hashes = [example.canonical_sha256() for example in examples]
    return prereg.stable_hash(
        {
            "schema_version": schema_version,
            "authorization_sha256": authorization_sha256,
            "role": role,
            "sessions": list(sessions),
            "sessions_sha256_newline": sessions_sha256_newline,
            "source_receipts": list(source_receipts),
            "ordered_example_hashes": ordered_hashes,
        }
    )


def _validate_source_receipts(
    source_receipts: tuple[dict[str, Any], ...],
    *,
    sessions: tuple[str, ...],
    examples: tuple[EntryExampleV1, ...],
) -> None:
    if [row.get("session") for row in source_receipts] != list(sessions):
        raise ValueError("source receipt sessions/order drift")
    for session, receipt in zip(sessions, source_receipts, strict=True):
        required = {
            "session",
            "source_files",
            "source_files_sha256",
            "example_count",
            "ordered_example_list_sha256",
            "session_content_sha256",
            "receipt_sha256",
        }
        if type(receipt) is not dict or set(receipt) != required:
            raise ValueError("source receipt schema drift")
        session_examples = [item for item in examples if item.model_input.session == session]
        hashes = [item.canonical_sha256() for item in session_examples]
        files = receipt["source_files"]
        if type(files) is not list or files != sorted(files, key=lambda row: row["relative_path"]):
            raise ValueError("source file order drift")
        if len({row["relative_path"] for row in files}) != len(files):
            raise ValueError("duplicate source file path")
        for row in files:
            if set(row) != {"relative_path", "bytes", "sha256"}:
                raise ValueError("source file row schema drift")
            if type(row["bytes"]) is not int or row["bytes"] < 0:
                raise ValueError("source byte count invalid")
            _exact_sha256(row["sha256"], name="source file sha256")
        if receipt["source_files_sha256"] != prereg.stable_hash(files):
            raise ValueError("source file root drift")
        if receipt["example_count"] != len(hashes) or not hashes:
            raise ValueError("source receipt example count drift")
        if receipt["ordered_example_list_sha256"] != prereg.stable_hash(hashes):
            raise ValueError("source receipt example root drift")
        content = prereg.stable_hash(
            {
                "schema_version": EntrySessionV1.SCHEMA_VERSION,
                "session": session,
                "source_files": files,
                "source_files_sha256": receipt["source_files_sha256"],
                "ordered_example_hashes": hashes,
            }
        )
        if receipt["session_content_sha256"] != content:
            raise ValueError("entry session content root drift")
        semantic = dict(receipt)
        observed = semantic.pop("receipt_sha256")
        if observed != prereg.stable_hash(semantic):
            raise ValueError("source receipt self-hash drift")


def _validate_dataset(
    dataset: Any,
    *,
    authorization: Any,
    expected_type: type[EntryFitDatasetV1] | type[EntryEvidenceDatasetV1],
) -> Any:
    if type(dataset) is not expected_type or dataset.schema_version != expected_type.SCHEMA_VERSION:
        raise TypeError("entry dataset type/schema drift")
    expected_auth = prereg.stable_hash(authorization.to_dict())
    sessions = tuple(authorization.sessions)
    if (
        dataset.authorization_sha256 != expected_auth
        or dataset.role != authorization.role
        or tuple(dataset.sessions) != sessions
        or dataset.sessions_sha256_newline != authorization.sessions_sha256_newline
        or dataset.sessions_sha256_newline != prereg.canonical_session_hash(sessions)
    ):
        raise ValueError("entry dataset authorization/partition drift")
    examples = tuple(dataset.examples)
    if not examples or any(type(value) is not EntryExampleV1 for value in examples):
        raise ValueError("entry dataset examples missing or mistyped")
    receipts = tuple(dataset.source_receipts)
    _validate_source_receipts(receipts, sessions=sessions, examples=examples)
    receipt_by_session = {row["session"]: row for row in receipts}
    identities: list[tuple[str, int]] = []
    all_action_target_keys = [
        f"{'session' if horizon == 'remaining_session' else f'h{horizon}'}_{axis}"
        for horizon in (*prereg.DIAGNOSTIC_HORIZONS, *prereg.GATE_HORIZONS)
        for axis in prereg.TARGET_AXES
    ]
    for example in examples:
        if example.schema_version != EntryExampleV1.SCHEMA_VERSION:
            raise ValueError("entry example schema drift")
        session = example.model_input.session
        decision = example.model_input.decision_time_ns
        if session not in sessions:
            raise ValueError("cross-session entry example")
        if len(example.action_execution_facts) != 42:
            raise ValueError("entry example action-fact count drift")
        if tuple(example.model_input.contract_ids) != tuple(
            fact.source_neutral_contract_id for fact in example.action_execution_facts
        ):
            raise ValueError("entry example action identity drift")
        if type(example.targets) is not dict or type(example.target_validity) is not dict:
            raise TypeError("entry example target bundle drift")
        sequence_mode = any(
            type(value) in (tuple, list) for value in example.targets.values()
        )
        if sequence_mode:
            source_files_sha256 = receipt_by_session[session][
                "source_files_sha256"
            ]
            for action_index, fact in enumerate(
                example.action_execution_facts
            ):
                if (
                    type(fact) is not EntryActionExecutionFactV1
                    or fact.schema_version
                    != EntryActionExecutionFactV1.SCHEMA_VERSION
                    or fact.session != session
                    or fact.decision_time_ns != decision
                    or type(fact.contract) is not ContractIdentityV1
                    or fact.contract.expiry != session
                    or fact.source_neutral_contract_id
                    != _source_neutral_contract_id(fact.contract)
                    or any(
                        type(value) is not int
                        for value in (
                            fact.represented_interval_end_ns,
                            fact.ts_recv_ns,
                            fact.available_at_ns,
                            fact.bid_micros,
                            fact.ask_micros,
                        )
                    )
                    or fact.represented_interval_end_ns > fact.available_at_ns
                    or fact.ts_recv_ns > fact.available_at_ns
                    or fact.available_at_ns > decision
                    or fact.bid_micros < 0
                    or fact.ask_micros < 0
                    or type(fact.physical_eligible) is not bool
                    or fact.source_receipt_sha256 != source_files_sha256
                ):
                    raise ValueError("all-action execution-fact provenance drift")
                _exact_sha256(
                    fact.source_record_sha256,
                    name="action execution source-record sha256",
                )
                if (
                    fact.physical_eligible
                    and not (
                        fact.bid_micros > 0
                        and fact.ask_micros > fact.bid_micros
                        and fact.ask_micros >= 1_000_000
                        and decision - fact.ts_recv_ns <= 90_000_000_000
                    )
                ):
                    raise ValueError("all-action executable fact drift")
                if (
                    bool(example.model_input.physical_action_mask[action_index])
                    is not fact.physical_eligible
                ):
                    raise ValueError("all-action physical mask drift")
                semantic = {
                    field.name: getattr(fact, field.name)
                    for field in fields(fact)
                    if field.name != "fact_sha256"
                }
                semantic["contract"] = _contract_dict(fact.contract)
                if fact.fact_sha256 != prereg.stable_hash(semantic):
                    raise ValueError("all-action execution-fact hash drift")
            if (
                list(example.targets) != all_action_target_keys
                or list(example.target_validity) != all_action_target_keys
            ):
                raise ValueError("all-action entry target key/order drift")
            for key in all_action_target_keys:
                values = example.targets[key]
                valid = example.target_validity[key]
                if (
                    type(values) is not list
                    or len(values) != 42
                    or any(
                        type(value) not in (int, float)
                        or not math.isfinite(float(value))
                        for value in values
                    )
                    or type(valid) is not list
                    or len(valid) != 42
                    or any(type(flag) is not bool for flag in valid)
                ):
                    raise ValueError("all-action entry target component drift")
            future = example.audit.get("future_audit") if type(example.audit) is dict else None
            required_future = {
                "entry_filled_by_action",
                "buy_hard_limit_micros_by_action",
                "entry_cash_debit_if_filled_micros_by_action",
                "arrival_bid_micros_by_action",
                "arrival_ask_micros_by_action",
                "arrival_available_at_ns_by_action",
                "terminal_executable_bid_micros_by_action",
                "hold_to_flat_net_pnl_micros_by_action",
                "future_marks_roots_by_action",
                "available_horizons",
            }
            if type(future) is not dict or set(future) != required_future:
                raise ValueError("all-action future audit schema drift")
            for name in required_future - {"available_horizons"}:
                if type(future[name]) is not list or len(future[name]) != 42:
                    raise ValueError("all-action future audit geometry drift")
        identities.append((session, decision))
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError("entry examples must be unique chronological frames")
    expected_hash = canonical_entry_dataset_sha256(
        schema_version=dataset.schema_version,
        authorization_sha256=dataset.authorization_sha256,
        role=dataset.role,
        sessions=dataset.sessions,
        sessions_sha256_newline=dataset.sessions_sha256_newline,
        source_receipts=receipts,
        examples=examples,
    )
    if dataset.dataset_sha256 != expected_hash:
        raise ValueError("entry dataset seal drift")
    return dataset


def validate_entry_fit_dataset(dataset: Any, /, *, authorization: Any) -> Any:
    assert_corrected_v32_executable_bridge()
    current = assert_fit_authorization_current(authorization)
    return _validate_dataset(dataset, authorization=current, expected_type=EntryFitDatasetV1)


def validate_entry_evidence_dataset(dataset: Any, /, *, authorization: Any) -> Any:
    assert_corrected_v32_executable_bridge()
    current = assert_entry_evidence_authorization_current(authorization)
    return _validate_dataset(
        dataset, authorization=current, expected_type=EntryEvidenceDatasetV1
    )


def validate_protected_holdout_dataset(dataset: Any, /, *, authorization: Any) -> Any:
    from v4.research.pathd_holdout_gate import ActiveProtectedHoldoutAuthorizationV1

    if type(authorization) is not ActiveProtectedHoldoutAuthorizationV1:
        raise TypeError("protected dataset requires an active holdout capability")
    return _validate_dataset(
        dataset, authorization=authorization, expected_type=EntryEvidenceDatasetV1
    )


def _stream_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _verify_resolved_source_rows(
    rows: tuple[dict[str, Any], ...], /, *, owner: str
) -> tuple[dict[str, Any], ...]:
    verified: list[dict[str, Any]] = []
    root = CORPUS_ROOT.resolve()
    for row in rows:
        path = Path(row["path"])
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"{owner} source file is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{owner} source must be a regular non-symlink file")
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"{owner} source escaped corpus root") from exc
        observed_hash, observed_size = _stream_sha256_and_size(path)
        if observed_hash != row["sha256"] or observed_size != row["size"]:
            raise RuntimeError(f"{owner} source bytes drifted before decode")
        verified.append(dict(row))
    return tuple(verified)


def _resolve_authorized_session_files(authorization: Any, session: str) -> tuple[dict[str, Any], ...]:
    """Resolve one session from the current immutable core receipt without decoding."""

    if session not in tuple(getattr(authorization, "sessions", ())):
        raise RuntimeError("session is outside exact authorization")
    prereg.assert_no_holdout_sessions((session,))
    selected = []
    for row in _manifest_source_rows_for_session(session):
        selected.append(
            {
                "path": CORPUS_ROOT / row["relative_path"],
                "sha256": row["sha256"],
                "size": row["bytes"],
                "relative_path": row["relative_path"],
            }
        )
    return tuple(selected)


def _processed_bar_open_ns(row: dict[str, Any]) -> int:
    if type(row) is not dict or "decision_time" not in row:
        raise ValueError("processed minute row schema drift")
    bar_open_ns = _timestamp_ns(row["decision_time"], name="processed minute timestamp")
    for name in ("source_quote_time", "source_context_time"):
        if name in row and _timestamp_ns(row[name], name=name) != bar_open_ns:
            raise ValueError("processed minute source clocks do not share BAR_OPEN")
    return bar_open_ns


def _processed_quote_watermarks(
    row: dict[str, Any], *, decision_time_ns: int
) -> np.ndarray:
    identities = np.asarray(row.get("contract_ids"), dtype=object)
    metadata = row.get("contract_quote_metadata")
    if identities.shape != (21, 2) or type(metadata) is not dict:
        raise ValueError("processed minute contract metadata geometry drift")
    clocks = np.empty((21, 2), dtype=np.int64)
    for strike_index in range(21):
        for right_index in range(2):
            identity = str(identities[strike_index, right_index])
            record = metadata.get(identity)
            if type(record) is not dict or record.get("contract_id") != identity:
                raise ValueError("processed minute contract metadata identity drift")
            clock = _timestamp_ns(
                record.get("received_timestamp_utc"),
                name="processed option receipt",
            )
            if clock > decision_time_ns:
                raise ValueError("processed option receipt is post-decision")
            clocks[strike_index, right_index] = clock
    return clocks


def _causal_snapshot_from_processed_minute(
    row: dict[str, Any],
    *,
    session: str,
    official_spx_rows: pd.DataFrame,
) -> EntrySnapshotV1:
    bar_open_ns = _processed_bar_open_ns(row)
    decision_time_ns = bar_open_ns + _MINUTE_NS
    local_session = (
        pd.Timestamp(decision_time_ns, unit="ns", tz="UTC")
        .tz_convert("America/New_York")
        .strftime("%Y-%m-%d")
    )
    if local_session != session:
        raise ValueError("processed minute crossed the authorized session")
    quote_clocks = _processed_quote_watermarks(
        row, decision_time_ns=decision_time_ns
    )
    market_window, market_names, spx_available_at_ns = (
        official_spx_market_window_from_rows(
            official_spx_rows,
            session=session,
            decision_time_ns=decision_time_ns,
            history_minutes=30,
        )
    )
    spx_close = float(market_window[-1, market_names.index("spx_close")])
    causal_atm = int(round(spx_close / 5.0) * 5)
    if type(row.get("atm_strike")) not in (int, np.int64) or int(
        row["atm_strike"]
    ) != causal_atm:
        raise ValueError(
            "persisted 42-action ladder is not centered on causal +60 official SPX"
        )
    decision_watermark_ns = max(
        int(np.max(quote_clocks)), int(spx_available_at_ns)
    )
    return historical_snapshot_from_processed_row(
        {
            "session": session,
            "decision_time_ns": decision_time_ns,
            "decision_watermark_ns": decision_watermark_ns,
            "atm_strike": int(row["atm_strike"]),
            "strike_offsets": row.get("strike_offsets"),
            "rights": row.get("rights"),
            "option_ladder": row.get("option_ladder"),
            "option_feature_names": row.get("feature_names"),
            "market_window": market_window,
            "market_feature_names": market_names,
            "contract_ids": row.get("contract_ids"),
            "contract_quote_watermark_ns": quote_clocks,
            "declared_alpha_sources": tuple(prereg.feature_lineage()["features"]),
        }
    )


def _action_execution_facts_from_processed_minute(
    snapshot: EntrySnapshotV1,
    row: dict[str, Any],
    *,
    source_provenance_sha256: str,
) -> tuple[EntryActionExecutionFactV1, ...]:
    try:
        bid_index = snapshot.option_feature_names.index("bid")
        ask_index = snapshot.option_feature_names.index("ask")
    except ValueError as exc:
        raise ValueError("processed entry ladder requires bid and ask") from exc
    candidate_mask = np.asarray(row.get("candidate_mask"))
    metadata = row.get("contract_quote_metadata")
    if candidate_mask.shape != (21, 2) or candidate_mask.dtype != np.bool_:
        raise ValueError("persisted candidate mask geometry/type drift")
    if type(metadata) is not dict:
        raise ValueError("processed contract metadata is absent")
    complete = True
    for strike_index in range(21):
        for right_index in range(2):
            bid = float(snapshot.option_ladder[strike_index, right_index, bid_index])
            ask = float(snapshot.option_ladder[strike_index, right_index, ask_index])
            receipt = int(snapshot.contract_quote_watermark_ns[strike_index, right_index])
            complete = complete and bool(
                math.isfinite(bid)
                and math.isfinite(ask)
                and bid >= 0.0
                and ask > 0.0
                and bid <= ask
                and receipt <= snapshot.decision_time_ns
                and snapshot.decision_time_ns - receipt <= 90_000_000_000
            )
    bar_open_ns = snapshot.decision_time_ns - _MINUTE_NS
    rows: list[EntryActionExecutionFactV1] = []
    for strike_index, offset in enumerate(snapshot.strike_offsets):
        for right_index, right in enumerate(snapshot.rights):
            identity = str(snapshot.contract_ids[strike_index, right_index])
            record = metadata.get(identity)
            if type(record) is not dict or record.get("contract_id") != identity:
                raise ValueError("processed execution fact metadata drift")
            strike = snapshot.atm_strike + float(offset)
            if (
                float(record.get("strike")) != strike
                or record.get("right") != right
                or float(record.get("offset")) != float(offset)
            ):
                raise ValueError("processed execution fact geometry drift")
            contract = _contract_for_action(
                session=snapshot.session, strike=strike, right=right
            )
            if _osi_symbol_from_source_neutral(identity, session=snapshot.session) != (
                contract.osi_symbol
            ):
                raise ValueError("processed execution fact OSI drift")
            bid = float(snapshot.option_ladder[strike_index, right_index, bid_index])
            ask = float(snapshot.option_ladder[strike_index, right_index, ask_index])
            bid_micros = int(round(bid * 1_000_000.0)) if math.isfinite(bid) else 0
            ask_micros = int(round(ask * 1_000_000.0)) if math.isfinite(ask) else 0
            receipt_ns = int(
                snapshot.contract_quote_watermark_ns[strike_index, right_index]
            )
            physical = bool(
                complete
                and candidate_mask[strike_index, right_index]
                and bid_micros > 0
                and ask_micros > bid_micros
                and ask_micros >= 1_000_000
            )
            source_record_sha256 = prereg.stable_hash(record)
            semantic = {
                "schema_version": EntryActionExecutionFactV1.SCHEMA_VERSION,
                "session": snapshot.session,
                "decision_time_ns": snapshot.decision_time_ns,
                "source_neutral_contract_id": identity,
                "contract": _contract_dict(contract),
                "represented_interval_end_ns": bar_open_ns,
                "ts_recv_ns": receipt_ns,
                "available_at_ns": snapshot.decision_time_ns,
                "bid_micros": bid_micros,
                "ask_micros": ask_micros,
                "source_receipt_sha256": source_provenance_sha256,
                "source_record_sha256": source_record_sha256,
                "physical_eligible": physical,
            }
            rows.append(
                EntryActionExecutionFactV1(
                    **{**semantic, "contract": contract},
                    fact_sha256=prereg.stable_hash(semantic),
                )
            )
    if len(rows) != 42:
        raise ValueError("processed execution fact ladder is not exactly 42")
    return tuple(rows)


def _raw_quote_index(
    path: Path, *, session: str, symbols: set[str]
) -> dict[str, _RawQuoteSeries]:
    parquet = pq.ParquetFile(path)
    collected: dict[str, list[tuple[int, int, int, int, int]]] = {
        symbol: [] for symbol in symbols
    }
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(
            row_group,
            columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"],
        ).to_pandas()
        if "ts_recv" not in frame.columns:
            frame = frame.reset_index()
        for row_index, record in enumerate(frame.itertuples(index=False)):
            values = record._asdict()
            symbol = str(values.get("symbol"))
            if symbol not in collected:
                continue
            ts_recv_ns = _timestamp_ns(values.get("ts_recv"), name="raw CBBO ts_recv")
            bid = values.get("bid_px_00")
            ask = values.get("ask_px_00")
            bid_micros = (
                int(round(float(bid) * 1_000_000.0))
                if bid is not None and math.isfinite(float(bid))
                else 0
            )
            ask_micros = (
                int(round(float(ask) * 1_000_000.0))
                if ask is not None and math.isfinite(float(ask))
                else 0
            )
            collected[symbol].append(
                (ts_recv_ns, row_group, row_index, bid_micros, ask_micros)
            )
    result: dict[str, _RawQuoteSeries] = {}
    for symbol, values in collected.items():
        values.sort(key=lambda value: value[:3])
        frozen = tuple(values)
        result[symbol] = _RawQuoteSeries(
            timestamps_ns=np.asarray([value[0] for value in frozen], dtype=np.int64),
            rows=frozen,
        )
    return result


def _latest_indexed_quote(
    index: dict[str, _RawQuoteSeries],
    *,
    symbol: str,
    at_or_before_ns: int,
    maximum_age_ns: int,
    require_actionable: bool,
) -> tuple[int, int, int, int, int] | None:
    series = index.get(symbol)
    if series is None or not len(series.rows):
        return None
    position = int(
        np.searchsorted(series.timestamps_ns, at_or_before_ns, side="right") - 1
    )
    while position >= 0:
        value = series.rows[position]
        ts_recv_ns, _row_group, _row_index, bid_micros, ask_micros = value
        if at_or_before_ns - ts_recv_ns > maximum_age_ns:
            break
        actionable = bid_micros > 0 and ask_micros > 0 and bid_micros < ask_micros
        if not require_actionable or actionable:
            return value
        position -= 1
    return None


def _sell_until_filled_mark_micros(
    index: dict[str, _RawQuoteSeries],
    *,
    symbol: str,
    decision_time_ns: int,
    terminal_time_ns: int,
    cache: dict[tuple[str, int, int], int | None],
) -> int | None:
    from v4.path_d.execution.research_fill_law import option_tick_micros

    cache_key = (symbol, decision_time_ns, terminal_time_ns)
    if cache_key in cache:
        return cache[cache_key]
    cursor = decision_time_ns
    while cursor < terminal_time_ns:
        reference = _latest_indexed_quote(
            index,
            symbol=symbol,
            at_or_before_ns=cursor,
            maximum_age_ns=2_000_000_000,
            require_actionable=True,
        )
        if reference is None:
            cache[cache_key] = None
            return None
        hard_limit = max(0, reference[3] - option_tick_micros(reference[3]))
        arrival_time_ns = cursor + _SECOND_NS
        if arrival_time_ns >= terminal_time_ns:
            break
        arrival = _latest_indexed_quote(
            index,
            symbol=symbol,
            at_or_before_ns=arrival_time_ns,
            maximum_age_ns=2_000_000_000,
            require_actionable=True,
        )
        if arrival is not None and arrival[3] >= hard_limit:
            cache[cache_key] = hard_limit
            return hard_limit
        cursor = arrival_time_ns + _SECOND_NS
    boundary = _latest_indexed_quote(
        index,
        symbol=symbol,
        at_or_before_ns=terminal_time_ns,
        maximum_age_ns=2_000_000_000,
        require_actionable=True,
    )
    value = 0 if boundary is None else boundary[3]
    cache[cache_key] = value
    return value


def _model_input_from_causal_history(
    snapshot: EntrySnapshotV1,
    *,
    prior_frames: list[tuple[str, Any]],
    facts: tuple[EntryActionExecutionFactV1, ...],
) -> Any:
    from v4.research.pathd_entry_models import EntryModelInputV1

    frame = signed17_from_snapshot(snapshot)
    history = build_identity_joined_history(
        [*prior_frames, (snapshot.session, frame)],
        current_session=snapshot.session,
        current_decision_time_ns=snapshot.decision_time_ns,
        current_contract_ids=snapshot.contract_ids,
        history_minutes=90,
    )
    offsets = np.repeat(snapshot.strike_offsets, len(snapshot.rights)).astype(
        np.float64
    )
    rights = np.tile(
        np.asarray(snapshot.rights, dtype=object), len(snapshot.strike_offsets)
    )
    return EntryModelInputV1(
        schema_version=EntryModelInputV1.SCHEMA_VERSION,
        session=snapshot.session,
        decision_time_ns=snapshot.decision_time_ns,
        signed17_frame=frame,
        signed17_history=history,
        hgb_summaries=hgb_signed17_summaries(
            history, current_offsets=offsets, current_rights=rights
        ),
        current_offsets=offsets,
        current_rights=tuple(rights.tolist()),
        contract_ids=tuple(fact.source_neutral_contract_id for fact in facts),
        physical_action_mask=np.asarray(
            [fact.physical_eligible for fact in facts], dtype=np.bool_
        ),
    )


def _entry_action_target_bundles(
    *,
    snapshot: EntrySnapshotV1,
    facts: tuple[EntryActionExecutionFactV1, ...],
    next_processed_row: dict[str, Any] | None,
    raw_quotes: dict[str, _RawQuoteSeries],
    sell_mark_cache: dict[tuple[str, int, int], int | None],
) -> tuple[dict[str, list[float]], dict[str, list[bool]], dict[str, Any]]:
    from v4.path_d.execution.research_fill_law import option_tick_micros

    horizons = (*prereg.DIAGNOSTIC_HORIZONS, *prereg.GATE_HORIZONS)
    keys = [
        f"{'session' if horizon == 'remaining_session' else f'h{horizon}'}_{axis}"
        for horizon in horizons
        for axis in prereg.TARGET_AXES
    ]
    targets = {key: [0.0] * 42 for key in keys}
    validity = {key: [False] * 42 for key in keys}
    terminal = _terminal_time_ns(snapshot.session)
    arrival_time = snapshot.decision_time_ns + _MINUTE_NS
    availability = {
        horizon: (
            arrival_time
            + (int(horizon) * _MINUTE_NS if horizon != "remaining_session" else 0)
            <= terminal
        )
        for horizon in horizons
    }
    availability["remaining_session"] = arrival_time < terminal
    next_metadata = (
        next_processed_row.get("contract_quote_metadata")
        if type(next_processed_row) is dict
        else None
    )
    next_bar_open = (
        _processed_bar_open_ns(next_processed_row)
        if type(next_processed_row) is dict
        else None
    )
    if next_bar_open is not None and next_bar_open + _MINUTE_NS != arrival_time:
        next_metadata = None
    filled_flags: list[bool] = [False] * 42
    hard_limits: list[int | None] = [None] * 42
    cash_debits_if_filled: list[int | None] = [None] * 42
    arrival_bids: list[int | None] = [None] * 42
    arrival_asks: list[int | None] = [None] * 42
    arrival_available: list[int | None] = [None] * 42
    terminal_bids: list[int | None] = [None] * 42
    hold_to_flat_net_pnl: list[int | None] = [None] * 42
    marks_roots: list[str | None] = [None] * 42
    for action_index, fact in enumerate(facts):
        if not fact.physical_eligible:
            continue
        hard_limit = fact.ask_micros + option_tick_micros(fact.ask_micros)
        hard_limits[action_index] = hard_limit
        cash_debits_if_filled[action_index] = hard_limit * 100 + 1_500_000
        for horizon in horizons:
            if availability[horizon]:
                prefix = "session" if horizon == "remaining_session" else f"h{horizon}"
                for axis in prereg.TARGET_AXES:
                    validity[f"{prefix}_{axis}"][action_index] = True
        arrival_record = (
            next_metadata.get(fact.source_neutral_contract_id)
            if type(next_metadata) is dict
            else None
        )
        if type(arrival_record) is not dict:
            continue
        arrival_receipt = _timestamp_ns(
            arrival_record.get("received_timestamp_utc"),
            name="entry arrival option receipt",
        )
        arrival_bid = arrival_record.get("bid")
        arrival_ask = arrival_record.get("ask")
        if type(arrival_bid) in (int, float, np.int64, np.float64) and math.isfinite(
            float(arrival_bid)
        ):
            arrival_bids[action_index] = int(round(float(arrival_bid) * 1_000_000.0))
        if type(arrival_ask) in (int, float, np.int64, np.float64) and math.isfinite(
            float(arrival_ask)
        ):
            arrival_asks[action_index] = int(round(float(arrival_ask) * 1_000_000.0))
        arrival_available[action_index] = arrival_time
        actionable = bool(
            arrival_receipt <= arrival_time
            and arrival_time - arrival_receipt <= 90_000_000_000
            and type(arrival_bid) in (int, float, np.int64, np.float64)
            and type(arrival_ask) in (int, float, np.int64, np.float64)
            and math.isfinite(float(arrival_bid))
            and math.isfinite(float(arrival_ask))
            and float(arrival_bid) > 0.0
            and float(arrival_ask) > float(arrival_bid)
            and float(arrival_ask) >= 1.0
            and int(round(float(arrival_ask) * 1_000_000.0)) <= hard_limit
        )
        filled_flags[action_index] = actionable
        if not actionable:
            continue
        mark_count = int((terminal - arrival_time) // _MINUTE_NS)
        symbol = fact.contract.osi_symbol
        marks: list[int | None] = []
        for offset in range(1, mark_count + 1):
            marks.append(
                _sell_until_filled_mark_micros(
                    raw_quotes,
                    symbol=symbol,
                    decision_time_ns=arrival_time + offset * _MINUTE_NS,
                    terminal_time_ns=terminal,
                    cache=sell_mark_cache,
                )
            )
        marks_roots[action_index] = prereg.stable_hash(marks)
        boundary = _latest_indexed_quote(
            raw_quotes,
            symbol=symbol,
            at_or_before_ns=terminal,
            maximum_age_ns=2_000_000_000,
            require_actionable=True,
        )
        terminal_bid = 0 if boundary is None else boundary[3]
        terminal_bids[action_index] = terminal_bid
        hold_to_flat_net_pnl[action_index] = (
            (terminal_bid - hard_limit) * 100 - 3_000_000
        )
        for horizon in horizons:
            if not availability[horizon]:
                continue
            count = mark_count if horizon == "remaining_session" else int(horizon)
            selected = marks[:count]
            prefix = "session" if horizon == "remaining_session" else f"h{horizon}"
            if len(selected) != count or any(value is None for value in selected):
                for axis in prereg.TARGET_AXES:
                    validity[f"{prefix}_{axis}"][action_index] = False
                continue
            values = entry_targets_from_executable_marks(
                entry_fill_price_micros=hard_limit,
                round_trip_fee_micros=3_000_000,
                marks_by_horizon={
                    "remaining_session" if horizon == "remaining_session" else f"h{horizon}": selected
                },
                available_horizons=(
                    "remaining_session"
                    if horizon == "remaining_session"
                    else f"h{horizon}",
                ),
            )
            for axis in prereg.TARGET_AXES:
                targets[f"{prefix}_{axis}"][action_index] = float(
                    values[f"{prefix}_{axis}"]
                )
    return targets, validity, {
        "entry_filled_by_action": filled_flags,
        "buy_hard_limit_micros_by_action": hard_limits,
        "entry_cash_debit_if_filled_micros_by_action": cash_debits_if_filled,
        "arrival_bid_micros_by_action": arrival_bids,
        "arrival_ask_micros_by_action": arrival_asks,
        "arrival_available_at_ns_by_action": arrival_available,
        "terminal_executable_bid_micros_by_action": terminal_bids,
        "hold_to_flat_net_pnl_micros_by_action": hold_to_flat_net_pnl,
        "future_marks_roots_by_action": marks_roots,
        "available_horizons": [
            "remaining_session" if horizon == "remaining_session" else f"h{horizon}"
            for horizon in horizons
            if availability[horizon]
        ],
    }


def _memmap_compact_examples(
    examples: list[EntryExampleV1], /
) -> tuple[list[EntryExampleV1], int]:
    """Move repeated full histories/summaries to unlinked read-only memmaps."""

    if not examples:
        return [], 0
    count = len(examples)
    specifications = {
        "values": (np.float64, (count, 90, 42, 17)),
        "finite": (np.bool_, (count, 90, 42, 17)),
        "present": (np.bool_, (count, 90, 42)),
        "available": (np.bool_, (count, 90)),
        "summaries": (np.float64, (count, 42, 444)),
    }
    paths: dict[str, Path] = {}
    writable: dict[str, np.memmap] = {}
    try:
        for name, (dtype, shape) in specifications.items():
            path = Path("/tmp") / f"pathd-{name}-{uuid.uuid4().hex}.mmap"
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            os.close(descriptor)
            paths[name] = path
            writable[name] = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        for index, example in enumerate(examples):
            history = example.model_input.signed17_history
            writable["values"][index] = history.values
            writable["finite"][index] = history.finite
            writable["present"][index] = history.contract_present
            writable["available"][index] = history.minute_available
            writable["summaries"][index] = example.model_input.hgb_summaries
        for mapping in writable.values():
            mapping.flush()
        writable.clear()
        readonly = {
            name: np.memmap(path, mode="r", dtype=dtype, shape=shape)
            for name, path in paths.items()
            for dtype, shape in [specifications[name]]
        }
        compacted: list[EntryExampleV1] = []
        for index, example in enumerate(examples):
            prior = example.model_input
            history = Signed17HistoryV1(
                schema_version=Signed17HistoryV1.SCHEMA_VERSION,
                values=readonly["values"][index],
                finite=readonly["finite"][index],
                contract_present=readonly["present"][index],
                minute_available=readonly["available"][index],
                current_contract_ids=prior.signed17_history.current_contract_ids,
                decision_time_ns=prior.signed17_history.decision_time_ns,
            )
            summary_view = readonly["summaries"][index]
            model_input = replace(
                prior,
                signed17_history=history,
                hgb_summaries=summary_view,
            )
            # EntryModelInputV1 defensively copies.  The source was just validated by
            # its constructor, so replace only that copy with the same read-only view.
            object.__setattr__(model_input, "hgb_summaries", summary_view)
            compacted.append(replace(example, model_input=model_input))
        storage_bytes = sum(
            int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
            for dtype, shape in specifications.values()
        )
        for path in paths.values():
            path.unlink()
        return compacted, storage_bytes
    except Exception:
        for path in paths.values():
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _decode_verified_entry_files(rows: tuple[dict[str, Any], ...]) -> Any:
    source_files = _receipt_source_rows(rows)
    if len(source_files) != len(_ENTRY_SOURCE_PATHS):
        raise RuntimeError("verified entry decoder source coverage drift")
    paths = {row["relative_path"]: Path(row["path"]) for row in rows}
    sessions = {
        match.group(1)
        for relative_path in paths
        for match in [re.search(r"(\d{4}-\d{2}-\d{2})", Path(relative_path).name)]
        if match is not None
    }
    if len(sessions) != 1:
        raise RuntimeError("verified entry decoder crossed sessions")
    session = next(iter(sessions))
    expected = {template.format(session=session) for template in _ENTRY_SOURCE_PATHS}
    if set(paths) != expected:
        raise RuntimeError("verified entry decoder path authority drift")
    processed_path = paths[f"{_PROCESSED_PREFIX}{session}.pkl"]
    raw_path = paths[f"{_RAW_CBBO_1S_PREFIX}{session}.cbbo-1s.parquet"]
    spx_path = paths[f"{_OFFICIAL_SPX_PREFIX}{session}.parquet"]
    processed_rows = pd.read_pickle(processed_path)
    if type(processed_rows) is not list or any(
        type(row) is not dict for row in processed_rows
    ):
        raise RuntimeError("verified processed minute payload schema drift")
    official_spx = pd.read_parquet(spx_path)
    source_provenance_sha256 = prereg.stable_hash(list(source_files))
    causal_rows: list[tuple[int, dict[str, Any], EntrySnapshotV1]] = []
    rejection_counts: dict[str, int] = {}
    seen_decisions: set[int] = set()
    for row_index, row in enumerate(processed_rows):
        try:
            snapshot = _causal_snapshot_from_processed_minute(
                row, session=session, official_spx_rows=official_spx
            )
            if snapshot.decision_time_ns in seen_decisions:
                raise ValueError("duplicate canonical completed-minute decision")
            seen_decisions.add(snapshot.decision_time_ns)
            causal_rows.append((row_index, row, snapshot))
        except (TypeError, ValueError) as exc:
            reason = str(exc)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    causal_rows.sort(key=lambda value: value[2].decision_time_ns)
    if not causal_rows:
        raise RuntimeError(
            f"verified entry decoder retained no causal minute rows: {rejection_counts}"
        )
    symbols = {
        _osi_symbol_from_source_neutral(str(identity), session=session)
        for _row_index, _row, snapshot in causal_rows
        for identity in snapshot.contract_ids.reshape(-1).tolist()
    }
    raw_quotes = _raw_quote_index(raw_path, session=session, symbols=symbols)
    sell_mark_cache: dict[tuple[str, int, int], int | None] = {}
    frame_history: list[tuple[str, Any]] = []
    examples: list[EntryExampleV1] = []
    by_original_index = {index: row for index, row, _snapshot in causal_rows}
    for original_index, row, snapshot in causal_rows:
        facts = _action_execution_facts_from_processed_minute(
            snapshot,
            row,
            source_provenance_sha256=source_provenance_sha256,
        )
        model_input = _model_input_from_causal_history(
            snapshot, prior_frames=frame_history, facts=facts
        )
        frame_history.append((session, model_input.signed17_frame))
        local = pd.Timestamp(
            snapshot.decision_time_ns, unit="ns", tz="UTC"
        ).tz_convert("America/New_York")
        if row.get("context_ready") is not True or (local.hour, local.minute) >= (15, 30):
            continue
        next_row = by_original_index.get(original_index + 1)
        targets, target_validity, future_audit = _entry_action_target_bundles(
            snapshot=snapshot,
            facts=facts,
            next_processed_row=next_row,
            raw_quotes=raw_quotes,
            sell_mark_cache=sell_mark_cache,
        )
        examples.append(
            EntryExampleV1(
                schema_version=EntryExampleV1.SCHEMA_VERSION,
                model_input=model_input,
                action_execution_facts=facts,
                targets=targets,
                target_validity=target_validity,
                audit={
                    "canonical_clock_rule": "persisted_BAR_OPEN_plus_60_seconds",
                    "source_bar_open_ns": snapshot.decision_time_ns - _MINUTE_NS,
                    "future_audit": future_audit,
                },
            )
        )
    if not examples:
        raise RuntimeError("verified entry decoder retained no eligible decision frames")
    examples, memmap_storage_bytes = _memmap_compact_examples(examples)
    ordered_hashes = [example.canonical_sha256() for example in examples]
    source_files_sha256 = prereg.stable_hash(list(source_files))
    content_sha256 = prereg.stable_hash(
        {
            "schema_version": EntrySessionV1.SCHEMA_VERSION,
            "session": session,
            "source_files": list(source_files),
            "source_files_sha256": source_files_sha256,
            "ordered_example_hashes": ordered_hashes,
        }
    )
    entry_session = EntrySessionV1(
        schema_version=EntrySessionV1.SCHEMA_VERSION,
        session=session,
        source_files=source_files,
        source_files_sha256=source_files_sha256,
        examples=tuple(examples),
        content_sha256=content_sha256,
    )
    return {
        "entry_session": entry_session,
        "decoder_audit": {
            "source_row_count": len(processed_rows),
            "causal_row_count": len(causal_rows),
            "example_count": len(examples),
            "rejected_frame_count": sum(rejection_counts.values()),
            "rejected_frames_by_reason": dict(sorted(rejection_counts.items())),
            "canonical_clock_rule": "persisted_BAR_OPEN_plus_60_seconds",
            "memmap_storage_bytes": memmap_storage_bytes,
            "in_memory_history_policy": "unlinked_read_only_memmap_views",
        },
    }


def read_verified_entry_session(authorization: Any, /, *, session: str) -> _VerifiedEntrySessionRead:
    assert_corrected_v32_executable_bridge()
    current = assert_fit_authorization_current(authorization)
    if session not in current.sessions:
        raise RuntimeError("session is outside fit authorization")
    prereg.assert_no_evidence_firewall_sessions((session,))
    resolved = _resolve_authorized_session_files(current, session)
    verified = _verify_resolved_source_rows(resolved, owner="authorized")
    payload = _decode_verified_entry_files(verified)
    return _VerifiedEntrySessionRead(session=session, source_files=verified, payload=payload)


def read_verified_entry_evidence_session(
    authorization: Any, /, *, session: str
) -> _VerifiedEntrySessionRead:
    assert_corrected_v32_executable_bridge()
    current = assert_entry_evidence_authorization_current(authorization)
    if session not in current.sessions:
        raise RuntimeError("session is outside evidence authorization")
    resolved = _resolve_authorized_session_files(current, session)
    verified = _verify_resolved_source_rows(resolved, owner="evidence")
    payload = _decode_verified_entry_files(verified)
    return _VerifiedEntrySessionRead(session=session, source_files=verified, payload=payload)


def _assemble_dataset(authorization: Any, reads: tuple[_VerifiedEntrySessionRead, ...], *, evidence: bool) -> Any:
    sessions: list[EntrySessionV1] = []
    for read in reads:
        if type(read.payload) is EntrySessionV1:
            sessions.append(read.payload)
        elif type(read.payload) is dict and type(read.payload.get("entry_session")) is EntrySessionV1:
            sessions.append(read.payload["entry_session"])
        else:
            raise RuntimeError("verified decoder did not return EntrySessionV1")
    examples = tuple(example for session in sessions for example in session.examples)
    receipts = []
    for item in sessions:
        hashes = [value.canonical_sha256() for value in item.examples]
        receipt = {
            "session": item.session,
            "source_files": list(item.source_files),
            "source_files_sha256": item.source_files_sha256,
            "example_count": len(hashes),
            "ordered_example_list_sha256": prereg.stable_hash(hashes),
            "session_content_sha256": item.content_sha256,
        }
        receipt["receipt_sha256"] = prereg.stable_hash(receipt)
        receipts.append(receipt)
    dataset_type = EntryEvidenceDatasetV1 if evidence else EntryFitDatasetV1
    authorization_sha = prereg.stable_hash(authorization.to_dict())
    dataset_sha = canonical_entry_dataset_sha256(
        schema_version=dataset_type.SCHEMA_VERSION,
        authorization_sha256=authorization_sha,
        role=authorization.role,
        sessions=authorization.sessions,
        sessions_sha256_newline=authorization.sessions_sha256_newline,
        source_receipts=tuple(receipts),
        examples=examples,
    )
    return dataset_type(
        schema_version=dataset_type.SCHEMA_VERSION,
        authorization_sha256=authorization_sha,
        role=authorization.role,
        sessions=tuple(authorization.sessions),
        sessions_sha256_newline=authorization.sessions_sha256_newline,
        source_receipts=tuple(receipts),
        examples=examples,
        dataset_sha256=dataset_sha,
    )


def load_authorized_entry_dataset(authorization: Any, /) -> EntryFitDatasetV1:
    assert_corrected_v32_executable_bridge()
    current = assert_fit_authorization_current(authorization)
    authorization_sha256 = prereg.stable_hash(current.to_dict())
    cached = _FIT_DATASET_CACHE.get(authorization_sha256)
    if cached is not None:
        if cached[0] != current:
            raise RuntimeError("fit dataset cache authorization collision")
        dataset = cached[1]
        receipts = {
            row["session"]: row for row in dataset.source_receipts
        }
        if set(receipts) != set(current.sessions):
            raise RuntimeError("fit dataset cache session receipt drift")
        for session in current.sessions:
            resolved = _resolve_authorized_session_files(current, session)
            verified = _verify_resolved_source_rows(resolved, owner="authorized")
            if list(_receipt_source_rows(verified)) != receipts[session][
                "source_files"
            ]:
                raise RuntimeError("fit dataset cached source bytes drifted")
        if (
            dataset.authorization_sha256 != authorization_sha256
            or dataset.role != current.role
            or tuple(dataset.sessions) != tuple(current.sessions)
        ):
            raise RuntimeError("fit dataset cache identity drift")
        return dataset
    reads = tuple(
        read_verified_entry_session(current, session=session) for session in current.sessions
    )
    dataset = _assemble_dataset(current, reads, evidence=False)
    validated = validate_entry_fit_dataset(dataset, authorization=current)
    _FIT_DATASET_CACHE[authorization_sha256] = (current, validated)
    return validated


def release_authorized_entry_fit_dataset(authorization: Any, /) -> None:
    """Release one process-local fit-role memmap population after its artifacts seal."""

    if type(authorization) is not prereg.FrozenFitAuthorization:
        raise TypeError("fit dataset release requires FrozenFitAuthorization")
    authorization_sha256 = prereg.stable_hash(authorization.to_dict())
    cached = _FIT_DATASET_CACHE.get(authorization_sha256)
    if cached is None:
        return
    if cached[0] != authorization:
        raise RuntimeError("fit dataset release authorization collision")
    del _FIT_DATASET_CACHE[authorization_sha256]


def load_authorized_entry_evidence_dataset(authorization: Any, /) -> EntryEvidenceDatasetV1:
    assert_corrected_v32_executable_bridge()
    from v4.research.pathd_evidence_gate import claim_entry_evidence_decode_once

    current = assert_entry_evidence_authorization_current(authorization)
    claim_entry_evidence_decode_once(current)
    reads = tuple(
        read_verified_entry_evidence_session(current, session=session)
        for session in current.sessions
    )
    dataset = _assemble_dataset(current, reads, evidence=True)
    return validate_entry_evidence_dataset(dataset, authorization=current)


def load_authorized_protected_holdout_dataset(authorization: Any, /) -> EntryEvidenceDatasetV1:
    raise RuntimeError("protected holdout dataset loading is transaction-private")


def _source_receipt_for_session(dataset: Any, session: str) -> dict[str, Any]:
    matches = [row for row in dataset.source_receipts if row["session"] == session]
    if len(matches) != 1:
        raise RuntimeError("dataset source receipt partition drift")
    return matches[0]


def _validate_quote_record(row: VerifiedResearchQuoteRowV1) -> None:
    if row.schema_version != row.SCHEMA_VERSION or row.source_vendor != "DATABENTO_OPRA":
        raise ValueError("verified option row schema/vendor drift")
    semantic = {field.name: getattr(row, field.name) for field in fields(row) if field.name != "record_sha256"}
    semantic["contract"] = _contract_dict(row.contract)
    if row.record_sha256 != prereg.stable_hash(semantic):
        raise ValueError("verified option row hash drift")
    if row.available_at_ns > row.query_at_or_before_ns:
        raise ValueError("verified option row is post-query")
    if row.query_at_or_before_ns - row.available_at_ns > row.query_maximum_age_ms * 1_000_000:
        raise ValueError("verified option row is stale")
    selection_key = tuple(row.selection_key)
    if selection_key != (
        row.available_at_ns,
        row.ts_recv_ns,
        row.represented_interval_end_ns,
        row.source_relative_path,
        row.row_group,
        row.row_index,
    ):
        raise ValueError("verified option row selection key drift")
    if not (
        row.represented_interval_end_ns <= row.ts_recv_ns <= row.available_at_ns
    ):
        raise ValueError("verified option row clock order drift")
    if type(row.eligible_row_count) is not int or row.eligible_row_count < 1:
        raise ValueError("verified option row eligible count drift")
    if re.fullmatch(r"[0-9a-f]{64}", row.selection_proof_sha256) is None:
        raise ValueError("verified option row selection proof drift")


def _validate_spx_record(row: VerifiedOfficialSpxRowV1) -> None:
    if row.schema_version != row.SCHEMA_VERSION or row.source_vendor != "THETADATA":
        raise ValueError("verified SPX row schema/vendor drift")
    semantic = {field.name: getattr(row, field.name) for field in fields(row) if field.name != "record_sha256"}
    if row.record_sha256 != prereg.stable_hash(semantic):
        raise ValueError("verified SPX row hash drift")
    if row.available_at_ns > row.query_at_or_before_ns:
        raise ValueError("verified SPX row is post-query")
    if row.query_at_or_before_ns - row.available_at_ns > row.query_maximum_age_ms * 1_000_000:
        raise ValueError("verified SPX row is stale")
    selection_key = tuple(row.selection_key)
    if selection_key != (
        row.available_at_ns,
        row.ts_recv_ns,
        row.represented_interval_end_ns,
        row.source_relative_path,
        row.row_group,
        row.row_index,
    ):
        raise ValueError("verified SPX row selection key drift")
    if not (
        row.represented_interval_end_ns <= row.ts_recv_ns <= row.available_at_ns
    ):
        raise ValueError("verified SPX row clock order drift")
    if type(row.eligible_row_count) is not int or row.eligible_row_count < 1:
        raise ValueError("verified SPX row eligible count drift")
    if re.fullmatch(r"[0-9a-f]{64}", row.selection_proof_sha256) is None:
        raise ValueError("verified SPX row selection proof drift")


def _validated_dataset_and_session_sources(
    authorization: Any,
    dataset: Any,
    *,
    session: str,
) -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]]]:
    if type(dataset) is EntryFitDatasetV1:
        current = assert_fit_authorization_current(authorization)
        validate_entry_fit_dataset(dataset, authorization=current)
    elif type(dataset) is EntryEvidenceDatasetV1:
        current = assert_entry_evidence_authorization_current(authorization)
        validate_entry_evidence_dataset(dataset, authorization=current)
    else:
        raise TypeError("verified source selector requires a sealed entry dataset")
    if session not in tuple(current.sessions):
        raise RuntimeError("verified source selector crossed authorization sessions")
    receipt = _source_receipt_for_session(dataset, session)
    expected = list(_manifest_source_rows_for_session(session))
    if receipt.get("source_files") != expected:
        raise RuntimeError("verified source selector receipt/manifest drift")
    resolved = tuple(
        {
            "path": CORPUS_ROOT / row["relative_path"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "size": row["bytes"],
        }
        for row in expected
    )
    verified = _verify_resolved_source_rows(resolved, owner="selector")
    return current, receipt, {row["relative_path"]: row for row in verified}


def _selection_proof(candidate_keys: tuple[tuple[Any, ...], ...]) -> str:
    if not candidate_keys:
        raise ValueError("selection proof requires at least one candidate")
    ordered = tuple(sorted(candidate_keys))
    selected_key = ordered[-1]
    return prereg.stable_hash(
        {
            "eligible_row_count": len(ordered),
            "selected_key": list(selected_key),
            "candidate_keys_sha256": prereg.stable_hash(
                [list(key) for key in ordered]
            ),
        }
    )


def _processed_quote_candidates(
    path: Path,
    *,
    relative_path: str,
    contract_id: str,
    at_or_before_ns: int,
    maximum_age_ns: int,
    require_actionable: bool,
) -> list[tuple[tuple[Any, ...], str, int, int]]:
    payload = pd.read_pickle(path)
    if type(payload) is not list:
        raise RuntimeError("processed quote selector payload drift")
    candidates: list[tuple[tuple[Any, ...], str, int, int]] = []
    for frame_index, frame in enumerate(payload):
        if type(frame) is not dict:
            raise RuntimeError("processed quote selector frame drift")
        bar_open_ns = _processed_bar_open_ns(frame)
        available_at_ns = bar_open_ns + _MINUTE_NS
        if (
            available_at_ns > at_or_before_ns
            or at_or_before_ns - available_at_ns > maximum_age_ns
        ):
            continue
        identities = np.asarray(frame.get("contract_ids"), dtype=object)
        if identities.shape != (21, 2):
            raise RuntimeError("processed quote selector identity geometry drift")
        locations = np.argwhere(identities == contract_id)
        if len(locations) != 1:
            continue
        strike_index, right_index = (int(value) for value in locations[0])
        record = frame.get("contract_quote_metadata", {}).get(contract_id)
        if type(record) is not dict or record.get("contract_id") != contract_id:
            raise RuntimeError("processed quote selector metadata drift")
        ts_recv_ns = _timestamp_ns(
            record.get("received_timestamp_utc"), name="processed quote receipt"
        )
        if ts_recv_ns > available_at_ns:
            raise RuntimeError("processed quote receipt exceeds completed-minute boundary")
        bid = record.get("bid")
        ask = record.get("ask")
        if type(bid) not in (int, float, np.int64, np.float64) or type(ask) not in (
            int,
            float,
            np.int64,
            np.float64,
        ):
            continue
        bid_value, ask_value = float(bid), float(ask)
        if not math.isfinite(bid_value) or not math.isfinite(ask_value):
            continue
        bid_micros = int(round(bid_value * 1_000_000.0))
        ask_micros = int(round(ask_value * 1_000_000.0))
        if bid_micros < 0 or ask_micros < 0:
            continue
        if require_actionable and not (
            bid_micros > 0 and ask_micros > 0 and bid_micros < ask_micros
        ):
            continue
        flat_index = frame_index * 42 + strike_index * 2 + right_index
        key = (
            available_at_ns,
            ts_recv_ns,
            bar_open_ns,
            relative_path,
            0,
            flat_index,
        )
        candidates.append(
            (
                key,
                prereg.stable_hash(
                    {
                        "bar_open_ns": bar_open_ns,
                        "contract_quote_metadata": record,
                    }
                ),
                bid_micros,
                ask_micros,
            )
        )
    return candidates


def _raw_quote_candidates(
    path: Path,
    *,
    relative_path: str,
    osi_symbol: str,
    at_or_before_ns: int,
    maximum_age_ns: int,
    require_actionable: bool,
) -> list[tuple[tuple[Any, ...], str, int, int]]:
    parquet = pq.ParquetFile(path)
    candidates: list[tuple[tuple[Any, ...], str, int, int]] = []
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(
            row_group,
            columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"],
        ).to_pandas()
        if "ts_recv" not in frame.columns:
            frame = frame.reset_index()
        for row_index, record in enumerate(frame.to_dict("records")):
            if str(record.get("symbol")) != osi_symbol:
                continue
            ts_recv_ns = _timestamp_ns(record.get("ts_recv"), name="raw quote receipt")
            if (
                ts_recv_ns > at_or_before_ns
                or at_or_before_ns - ts_recv_ns > maximum_age_ns
            ):
                continue
            bid, ask = record.get("bid_px_00"), record.get("ask_px_00")
            if bid is None or ask is None:
                continue
            bid_value, ask_value = float(bid), float(ask)
            if not math.isfinite(bid_value) or not math.isfinite(ask_value):
                continue
            bid_micros = int(round(bid_value * 1_000_000.0))
            ask_micros = int(round(ask_value * 1_000_000.0))
            if bid_micros < 0 or ask_micros < 0:
                continue
            if require_actionable and not (
                bid_micros > 0 and ask_micros > 0 and bid_micros < ask_micros
            ):
                continue
            represented_interval_end_ns = ts_recv_ns - _SECOND_NS
            key = (
                ts_recv_ns,
                ts_recv_ns,
                represented_interval_end_ns,
                relative_path,
                row_group,
                row_index,
            )
            candidates.append(
                (
                    key,
                    prereg.stable_hash(
                        {
                            "ts_recv_ns": ts_recv_ns,
                            "symbol": osi_symbol,
                            "bid_micros": bid_micros,
                            "ask_micros": ask_micros,
                            "row_group": row_group,
                            "row_index": row_index,
                        }
                    ),
                    bid_micros,
                    ask_micros,
                )
            )
    return candidates


def read_verified_research_quote_row(
    authorization: Any,
    /,
    *,
    dataset: Any,
    session: str,
    contract: ContractIdentityV1,
    at_or_before_ns: int,
    maximum_age_ms: int,
    require_actionable: bool,
) -> VerifiedResearchQuoteRowV1:
    if type(at_or_before_ns) is not int or type(maximum_age_ms) is not int:
        raise TypeError("verified quote selector clocks must be exact integers")
    if maximum_age_ms not in (2_000, 90_000) or type(require_actionable) is not bool:
        raise ValueError("verified quote selector registered query drift")
    _current, receipt, sources = _validated_dataset_and_session_sources(
        authorization, dataset, session=session
    )
    contract_id = _source_neutral_contract_id(contract)
    maximum_age_ns = maximum_age_ms * 1_000_000
    if maximum_age_ms == 90_000:
        relative_path = f"{_PROCESSED_PREFIX}{session}.pkl"
        candidates = _processed_quote_candidates(
            Path(sources[relative_path]["path"]),
            relative_path=relative_path,
            contract_id=contract_id,
            at_or_before_ns=at_or_before_ns,
            maximum_age_ns=maximum_age_ns,
            require_actionable=require_actionable,
        )
    else:
        relative_path = f"{_RAW_CBBO_1S_PREFIX}{session}.cbbo-1s.parquet"
        candidates = _raw_quote_candidates(
            Path(sources[relative_path]["path"]),
            relative_path=relative_path,
            osi_symbol=contract.osi_symbol,
            at_or_before_ns=at_or_before_ns,
            maximum_age_ns=maximum_age_ns,
            require_actionable=require_actionable,
        )
    if not candidates:
        raise RuntimeError("verified quote selector found no eligible row")
    candidates.sort(key=lambda value: value[0])
    selected = candidates[-1]
    if sum(value[0] == selected[0] for value in candidates) != 1:
        raise RuntimeError("verified quote selector latest key is duplicated")
    key, canonical_row_sha256, bid_micros, ask_micros = selected
    candidate_keys = tuple(value[0] for value in candidates)
    semantic = {
        "schema_version": VerifiedResearchQuoteRowV1.SCHEMA_VERSION,
        "session": session,
        "source_relative_path": relative_path,
        "source_file_sha256": sources[relative_path]["sha256"],
        "row_group": key[4],
        "row_index": key[5],
        "canonical_row_sha256": canonical_row_sha256,
        "contract": _contract_dict(contract),
        "source_vendor": "DATABENTO_OPRA",
        "represented_interval_end_ns": key[2],
        "ts_recv_ns": key[1],
        "available_at_ns": key[0],
        "bid_micros": bid_micros,
        "ask_micros": ask_micros,
        "source_receipt_sha256": receipt["receipt_sha256"],
        "query_at_or_before_ns": at_or_before_ns,
        "query_maximum_age_ms": maximum_age_ms,
        "query_require_actionable": require_actionable,
        "eligible_row_count": len(candidates),
        "selection_key": key,
        "selection_proof_sha256": _selection_proof(candidate_keys),
    }
    row = VerifiedResearchQuoteRowV1(
        **{
            **semantic,
            "contract": contract,
            "record_sha256": prereg.stable_hash(semantic),
        }
    )
    _validate_quote_record(row)
    return row


def read_verified_official_spx_row(
    authorization: Any,
    /,
    *,
    dataset: Any,
    session: str,
    at_or_before_ns: int,
    maximum_age_ms: int,
) -> VerifiedOfficialSpxRowV1:
    if type(at_or_before_ns) is not int or type(maximum_age_ms) is not int:
        raise TypeError("verified SPX selector clocks must be exact integers")
    if maximum_age_ms != 90_000:
        raise ValueError("verified SPX selector registered query drift")
    _current, receipt, sources = _validated_dataset_and_session_sources(
        authorization, dataset, session=session
    )
    relative_path = f"{_OFFICIAL_SPX_PREFIX}{session}.parquet"
    parquet = pq.ParquetFile(Path(sources[relative_path]["path"]))
    required_invariants = prereg.feature_lineage()["raw_leaf_registry"][
        "official_spx"
    ]["row_invariants"]
    candidates: list[tuple[tuple[Any, ...], str, int]] = []
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group).to_pandas()
        for row_index, record in enumerate(frame.to_dict("records")):
            if any(record.get(name) != value for name, value in required_invariants.items()):
                raise RuntimeError("official SPX selector row invariant drift")
            event_time_ns = _timestamp_ns(
                record.get("event_time"), name="official SPX event_time"
            )
            available_at_ns = event_time_ns + _MINUTE_NS
            if (
                available_at_ns > at_or_before_ns
                or at_or_before_ns - available_at_ns > maximum_age_ms * 1_000_000
            ):
                continue
            close = record.get("close")
            if type(close) not in (int, float, np.int64, np.float64) or not math.isfinite(
                float(close)
            ) or float(close) <= 0.0:
                raise RuntimeError("official SPX selector close drift")
            spx_micros = int(round(float(close) * 1_000_000.0))
            key = (
                available_at_ns,
                available_at_ns,
                event_time_ns,
                relative_path,
                row_group,
                row_index,
            )
            canonical = {
                "event_time_ns": event_time_ns,
                "symbol": record["symbol"],
                "open_float64_hex": float(record["open"]).hex(),
                "high_float64_hex": float(record["high"]).hex(),
                "low_float64_hex": float(record["low"]).hex(),
                "close_float64_hex": float(record["close"]).hex(),
                "volume": int(record["volume"]),
                "count": int(record["count"]),
                "vwap_float64_hex": float(record["vwap"]).hex(),
                "context_source": record["context_source"],
                "is_derived": record["is_derived"],
                "is_proxy": record["is_proxy"],
                "is_official_index_data": record["is_official_index_data"],
            }
            candidates.append((key, prereg.stable_hash(canonical), spx_micros))
    if not candidates:
        raise RuntimeError("verified SPX selector found no fresh official row")
    candidates.sort(key=lambda value: value[0])
    selected = candidates[-1]
    if sum(value[0] == selected[0] for value in candidates) != 1:
        raise RuntimeError("verified SPX selector latest key is duplicated")
    key, canonical_row_sha256, spx_micros = selected
    candidate_keys = tuple(value[0] for value in candidates)
    semantic = {
        "schema_version": VerifiedOfficialSpxRowV1.SCHEMA_VERSION,
        "session": session,
        "source_relative_path": relative_path,
        "source_file_sha256": sources[relative_path]["sha256"],
        "row_group": key[4],
        "row_index": key[5],
        "canonical_row_sha256": canonical_row_sha256,
        "source_vendor": "THETADATA",
        "represented_interval_end_ns": key[2],
        "ts_recv_ns": key[1],
        "available_at_ns": key[0],
        "spx_micros": spx_micros,
        "source_receipt_sha256": receipt["receipt_sha256"],
        "query_at_or_before_ns": at_or_before_ns,
        "query_maximum_age_ms": maximum_age_ms,
        "eligible_row_count": len(candidates),
        "selection_key": key,
        "selection_proof_sha256": _selection_proof(candidate_keys),
    }
    row = VerifiedOfficialSpxRowV1(
        **semantic, record_sha256=prereg.stable_hash(semantic)
    )
    _validate_spx_record(row)
    return row


__all__ = [
    "EntryFuturePathV1",
    "EntryExampleV1",
    "EntryActionExecutionFactV1",
    "EntrySessionV1",
    "EntryFitDatasetV1",
    "EntryEvidenceDatasetV1",
    "VerifiedResearchQuoteRowV1",
    "VerifiedOfficialSpxRowV1",
    "build_entry_example",
    "available_entry_horizons",
    "entry_targets_from_executable_marks",
    "read_verified_entry_session",
    "load_authorized_entry_dataset",
    "release_authorized_entry_fit_dataset",
    "canonical_entry_dataset_sha256",
    "validate_entry_fit_dataset",
    "read_verified_entry_evidence_session",
    "load_authorized_entry_evidence_dataset",
    "validate_entry_evidence_dataset",
    "load_authorized_protected_holdout_dataset",
    "validate_protected_holdout_dataset",
    "read_verified_research_quote_row",
    "read_verified_official_spx_row",
]
