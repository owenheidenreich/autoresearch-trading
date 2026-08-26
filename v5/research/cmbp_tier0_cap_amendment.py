"""Job-55 exact-$2.00 cap amendment for the frozen CMBP Tier-0 acquisition.

This module is an authority overlay.  It imports the sealed Job-51/52 code as
read-only libraries, independently adopts one exact terminal Job-52 state, and
keeps every Job-55 byte in a sibling external namespace.
"""

from __future__ import annotations

import dataclasses
import math
import os
import stat
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from v5.research import cmbp_tier0 as base
from v5.research import cmbp_tier0_paid as paid


JOB_ID = 55
TARGET_JOB_ID = 51
PREDECESSOR_JOB_ID = 52
CONTRACT_ARTIFACT = "JOB55_CMBP_TIER0_CAP_AMENDMENT_PROGRAM_CONTRACT_V1"
READINESS_ARTIFACT = "JOB55_CMBP_TIER0_CAP_AMENDMENT_LOCAL_READINESS_RECEIPT_V1"
READINESS_STATUS = "JOB55_CMBP_TIER0_CAP_AMENDMENT_LOCAL_READY_ONLY"
ADOPTION_ARTIFACT = "JOB55_CMBP_TIER0_JOB52_TERMINAL_ADOPTION_V1"
SESSION_QC_ARTIFACT = "JOB55_CMBP_TIER0_CAP_AMENDMENT_SESSION_QC_V1"
AGGREGATE_RECEIPT_ARTIFACT = "JOB55_CMBP_TIER0_CAP_AMENDMENT_ACQUISITION_QC_RECEIPT_V1"
ATTEMPT_STOP_ARTIFACT = "JOB55_CMBP_TIER0_CAP_AMENDMENT_ATTEMPT_STOP_V1"

EXPECTED_PLAN_FILE_SHA256 = "b59da7b859f1b6175ccfda111dee904ece2ee14aa514e7ffc26a9bcfa53c4aa7"
EXPECTED_PROGRAM_CONTRACT_SHA256 = "60011bd6a2eba526f3690fe01326fb1074dcba14c00f0b72fc0ed2ff3d8ada78"
EXPECTED_PROGRAM_CONTRACT_FILE_SHA256 = "11b7576dfa47f3e951cf5717ab26a4d53355bfb61d4305cb15786e24d4c2a1de"
EXPECTED_JOB52_READINESS_SHA256 = "6e4e4977d5ff74f06ff866409cbaacdc833a58dc85039d51d7c83ae7e9face6e"
EXPECTED_JOB52_READINESS_FILE_SHA256 = "46ddc5631a69e26a9ec0284a1ec5d30723ebc737eb2762f1db2cab35ceed9c0c"
EXPECTED_JOB52_CONTRACT_SHA256 = "094733cb8e6145214161cf4fd3765b3667cc8fabac53eea9abc08796d00600cf"
EXPECTED_JOB52_CONTRACT_FILE_SHA256 = "befa3f3fa2e5d7f5d4b667da78061a2a5def7c1b4802a8c6b7a2cf801af277ac"

PER_SESSION_LIFETIME_CAP_USD = Decimal("2.00")
TOTAL_COMMITTED_CAP_USD = Decimal("32.00")
OPENING_COMMITMENT_USD = Decimal("0.950392448902")
FAILED_SESSION = "2025-08-22"
FAILED_SESSION_JOB55_COST_CALL_ALLOWANCE = 1
FAILED_SESSION_JOB55_START_ALLOWANCE = 1

# Explicit Job55 aliases make the overlay identity unambiguous to receipts and tests.
EXPECTED_JOB55_PLAN_FILE_SHA256 = EXPECTED_PLAN_FILE_SHA256
EXPECTED_JOB55_PROGRAM_CONTRACT_SHA256 = EXPECTED_PROGRAM_CONTRACT_SHA256
EXPECTED_JOB55_PROGRAM_CONTRACT_FILE_SHA256 = EXPECTED_PROGRAM_CONTRACT_FILE_SHA256
JOB55_PER_SESSION_LIFETIME_CAP_USD = PER_SESSION_LIFETIME_CAP_USD
JOB55_TOTAL_COMMITTED_CAP_USD = TOTAL_COMMITTED_CAP_USD

JOB55_ROOT_RELATIVE = Path("cmbp-tier0/job55")
JOB55_NAMESPACE = JOB55_ROOT_RELATIVE.as_posix()
JOB51_ROOT_RELATIVE = Path("cmbp-tier0/job51")
JOB55_ADOPTION_NAME = "JOB52_TERMINAL_ADOPTION_V1.json"
JOB55_LOCK_BINDING_NAME = "RUN_LOCK_JOB55_BINDING_V1.json"
JOB55_ATTEMPT_ANCHOR_NAME = "JOB55_ATTEMPT_SET_ANCHOR_V1.json"
JOB55_SESSION_QC_NAME = "SESSION_QC_V3.json"
JOB55_AGGREGATE_NAME = "JOB55_CMBP_TIER0_CAP_AMENDMENT_ACQUISITION_QC_RECEIPT_V1.json"
JOB55_ATTEMPT_STOP_NAME = "ATTEMPT_STOP_JOB55_V1.json"
JOB55_AGGREGATE_SEAL_NAME = "JOB55_AGGREGATE_SEAL_V1.json"

# Frozen from the exact sterile four-target population after the Job-55
# runner, recovery, QC, aggregate, and sealer cases settled.
EXPECTED_FOCUSED_TEST_COUNT = 373
EXPECTED_FOCUSED_TEST_IDENTITY_SHA256 = "fbb7c038794224f05ee8bc1dcc51f7d56f8831192a6ce0d445f110ecb3798090"

Tier0Error = base.Tier0Error
SessionRequest = base.SessionRequest
VolumeIdentity = base.VolumeIdentity
AttemptJournal = base.AttemptJournal


def _paths(repo_root: Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    return {
        "plan": root / "v5/work/cmbp-tier0-cap-amendment/PLAN.md",
        "contract": root / "v5/work/cmbp-tier0-cap-amendment/PROGRAM_CONTRACT_V1.json",
        "readiness": root / "v5/work/cmbp-tier0-cap-amendment/LOCAL_READINESS_RECEIPT_V1.json",
        "test_report": root / "v5/work/cmbp-tier0-cap-amendment/TEST_RESULTS_V1.xml",
        "job52_plan": root / "v5/work/cmbp-tier0-paid-resume/PLAN.md",
        "job52_contract": root / "v5/work/cmbp-tier0-paid-resume/PROGRAM_CONTRACT_V1.json",
        "job52_readiness": root / "v5/work/cmbp-tier0-paid-resume/LOCAL_READINESS_RECEIPT_V1.json",
        "job52_test_report": root / "v5/work/cmbp-tier0-paid-resume/TEST_RESULTS_V1.xml",
    }


def _private_regular(path: Path, *, status: str) -> os.stat_result:
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError as exc:
        raise Tier0Error(f"required file is absent: {Path(path).name}", status=status) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise Tier0Error(f"required file is unsafe: {Path(path).name}", status=status)
    return metadata


@dataclass(frozen=True)
class CapAmendmentScopeBundle:
    """Exact Job-51 scope plus frozen Job-52 and Job-55 authority packets."""

    paid_bundle: paid.PaidScopeBundle
    contract: Mapping[str, Any]
    contract_file_sha256: str
    plan_file_sha256: str

    @property
    def base_bundle(self) -> base.ScopeBundle:
        return self.paid_bundle.base_bundle

    @property
    def sessions(self) -> tuple[SessionRequest, ...]:
        return self.paid_bundle.sessions


def load_cap_amendment_scope_bundle(repo_root: Path) -> CapAmendmentScopeBundle:
    """Reconstruct the exact frozen scope and both authority epochs."""

    root = Path(repo_root).resolve()
    paths = _paths(root)
    for name in ("plan", "contract", "job52_plan", "job52_contract", "job52_readiness", "job52_test_report"):
        _private_regular(paths[name], status="STOP_JOB55_CONTRACT_DRIFT")
    paid_bundle = paid.load_paid_scope_bundle(root)
    if base.file_sha256(paths["plan"]) != EXPECTED_PLAN_FILE_SHA256:
        raise Tier0Error("Job-55 plan raw hash drifted", status="STOP_JOB55_CONTRACT_DRIFT")
    contract = base.strict_json(paths["contract"])
    contract_file_sha = base.file_sha256(paths["contract"])
    if (
        contract.get("artifact_type") != CONTRACT_ARTIFACT
        or contract.get("schema_version") != "v5.job55-cmbp-tier0-cap-amendment-program-contract.v1"
        or contract.get("job_id") != JOB_ID
        or contract.get("target_job_id") != TARGET_JOB_ID
        or contract.get("predecessor_job_id") != PREDECESSOR_JOB_ID
        or contract.get("contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or base.self_hash(contract, "contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or contract_file_sha != EXPECTED_PROGRAM_CONTRACT_FILE_SHA256
        or contract.get("plan_file_sha256") != EXPECTED_PLAN_FILE_SHA256
    ):
        raise Tier0Error("Job-55 contract identity/hash drifted", status="STOP_JOB55_CONTRACT_DRIFT")
    authority = contract.get("authority", {})
    commitment = contract.get("commitment_law", {})
    scope = contract.get("scope", {})
    destination = contract.get("destination", {})
    if (
        authority.get("owner_authorized_in_current_conversation") is not True
        or authority.get("per_session_lifetime_committed_quote_cap_usd") != "2.00"
        or authority.get("total_committed_quote_cap_usd") != "32.00"
        or authority.get("singular_job55_retry_session") != FAILED_SESSION
        or authority.get("singular_job55_retry_cost_call_allowance") != 1
        or authority.get("singular_job55_retry_time_series_start_allowance") != 1
        or commitment.get("maximum_per_session_lifetime_usd") != "2.00"
        or commitment.get("maximum_total_usd") != "32.00"
        or commitment.get("job52_opening_total_usd") != "0.950392448902"
        or commitment.get("job52_opening_by_session_usd") != {FAILED_SESSION: "0.950392448902"}
        or commitment.get("job52_opening_commitment_count") != 1
        or commitment.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
        or destination.get("job51_52_tree_mutated_by_job55") is not False
        or destination.get("job55_root_relative") != str(JOB55_ROOT_RELATIVE)
        or destination.get("job55_final_sessions_relative") != str(JOB55_ROOT_RELATIVE / "sessions")
        or scope.get("semantic_sha256") != base.EXPECTED_SCOPE_SHA256
        or scope.get("raw_file_sha256") != base.EXPECTED_SCOPE_FILE_SHA256
        or scope.get("session_count") != base.EXPECTED_SESSION_COUNT
        or scope.get("total_record_count") != base.EXPECTED_RECORD_COUNT
        or scope.get("total_session_symbols") != base.EXPECTED_SESSION_SYMBOLS
        or scope.get("dataset") != base.EXPECTED_DATASET
        or scope.get("schema") != base.EXPECTED_SCHEMA
        or scope.get("stype_in") != base.EXPECTED_STYPE_IN
        or scope.get("stype_out") != base.EXPECTED_STYPE_OUT
    ):
        raise Tier0Error("Job-55 authority/scope law drifted", status="STOP_JOB55_CONTRACT_DRIFT")
    if (
        base.file_sha256(paths["job52_contract"]) != EXPECTED_JOB52_CONTRACT_FILE_SHA256
        or paid_bundle.contract.get("contract_sha256") != EXPECTED_JOB52_CONTRACT_SHA256
        or base.file_sha256(paths["job52_readiness"]) != EXPECTED_JOB52_READINESS_FILE_SHA256
        or base.strict_json(paths["job52_readiness"]).get("receipt_sha256") != EXPECTED_JOB52_READINESS_SHA256
    ):
        raise Tier0Error("Job-52 immutable authority drifted", status="STOP_JOB55_PREDECESSOR_DRIFT")
    return CapAmendmentScopeBundle(
        paid_bundle=paid_bundle,
        contract=contract,
        contract_file_sha256=contract_file_sha,
        plan_file_sha256=base.file_sha256(paths["plan"]),
    )


@dataclass(frozen=True)
class Job52EvidencePins:
    """Exact external predecessor population accepted by the amendment."""

    directories: tuple[str, ...]
    file_sha256: Mapping[str, str]
    anchor_sha256: str
    failed_partial_bytes: int
    failed_partial_sha256: str


def _pins_from_contract(contract: Mapping[str, Any]) -> Job52EvidencePins:
    evidence = contract.get("opening_job52_evidence", {})
    directories = evidence.get("exact_directory_population")
    files = evidence.get("exact_file_sha256")
    if not isinstance(directories, list) or not all(isinstance(item, str) for item in directories):
        raise Tier0Error("Job-52 directory pins are malformed", status="STOP_JOB55_CONTRACT_DRIFT")
    if not isinstance(files, dict) or not all(
        isinstance(key, str) and base.SHA256_RE.fullmatch(str(value)) is not None for key, value in files.items()
    ):
        raise Tier0Error("Job-52 file pins are malformed", status="STOP_JOB55_CONTRACT_DRIFT")
    return Job52EvidencePins(
        directories=tuple(directories),
        file_sha256=MappingProxyType(dict(files)),
        anchor_sha256=str(evidence.get("anchor_sha256")),
        failed_partial_bytes=int(evidence.get("failed_partial_bytes", -1)),
        failed_partial_sha256=str(evidence.get("failed_partial_file_sha256")),
    )


# This immutable value is filled from the separately self-hashed contract at import.
EXPECTED_JOB52_EVIDENCE_PINS = _pins_from_contract(
    base.strict_json(Path(__file__).resolve().parents[1] / "work/cmbp-tier0-cap-amendment/PROGRAM_CONTRACT_V1.json")
)


@dataclass(frozen=True)
class Job52OpeningEvidence:
    job51_root: str
    manifest_sha256: str
    anchor_sha256: str
    opening_commitment_total_usd: Decimal
    opening_commitment_by_session_usd: Mapping[str, Decimal]
    opening_commitment_count: int
    cost_call_starts: int
    cost_call_results: int
    time_series_call_starts: int
    time_series_call_results: int
    failed_partial_bytes: int
    failed_partial_sha256: str
    superseded_terminal_attempt_id: str
    superseded_terminal_cost_result_record_hash: str
    predecessor_summary: Mapping[str, Any]


def _external_tree_manifest(job51_root: Path, *, volume: VolumeIdentity) -> dict[str, Any]:
    root = Path(job51_root)
    try:
        root_meta = root.lstat()
    except FileNotFoundError as exc:
        raise Tier0Error("Job-51 predecessor root is absent", status="STOP_JOB55_PREDECESSOR_DRIFT") from exc
    if stat.S_ISLNK(root_meta.st_mode) or not stat.S_ISDIR(root_meta.st_mode) or root_meta.st_dev != volume.st_dev:
        raise Tier0Error("Job-51 predecessor root is unsafe", status="STOP_JOB55_PREDECESSOR_DRIFT")
    directories: list[str] = ["."]
    files: dict[str, str] = {}
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        current_meta = current.lstat()
        if (
            stat.S_ISLNK(current_meta.st_mode)
            or not stat.S_ISDIR(current_meta.st_mode)
            or current_meta.st_dev != volume.st_dev
        ):
            raise Tier0Error("Job-51 predecessor directory is unsafe", status="STOP_JOB55_PREDECESSOR_DRIFT")
        names.sort()
        filenames.sort()
        for name in names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or metadata.st_dev != volume.st_dev:
                raise Tier0Error("Job-51 predecessor child is unsafe", status="STOP_JOB55_PREDECESSOR_DRIFT")
            directories.append(child.relative_to(root).as_posix())
        for name in filenames:
            child = current / name
            metadata = child.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_dev != volume.st_dev
            ):
                raise Tier0Error("Job-51 predecessor file is unsafe", status="STOP_JOB55_PREDECESSOR_DRIFT")
            files[child.relative_to(root).as_posix()] = base.file_sha256(child)
    return {"directories": sorted(set(directories)), "file_sha256": dict(sorted(files.items()))}


def _assert_current_production_volume(
    volume: VolumeIdentity,
    *,
    unpublished_records: int = base.EXPECTED_RECORD_COUNT,
) -> VolumeIdentity:
    """Replace caller trust with a fresh inspection of the one authorized mount."""

    if (
        volume.mount_point != str(base.EXPECTED_VOLUME_ROOT)
        or volume.volume_uuid != base.EXPECTED_VOLUME_UUID
        or volume.filesystem != base.EXPECTED_FILESYSTEM
    ):
        raise Tier0Error("Job-55 volume identity is not the authorized mount", status="STOP_JOB55_EXTERNAL_VOLUME")
    observed = base.inspect_destination_volume(
        base.EXPECTED_VOLUME_ROOT,
        unpublished_records=unpublished_records,
        expected_device_identifier=volume.device_identifier,
    )
    if _stable_volume_fields(observed) != _stable_volume_fields(volume):
        raise Tier0Error("Job-55 inspected volume identity changed", status="STOP_JOB55_EXTERNAL_VOLUME")
    return observed


def _validate_job52_opening_evidence_with_pins(
    job51_root: Path,
    *,
    volume: VolumeIdentity,
    pins: Job52EvidencePins,
) -> Job52OpeningEvidence:
    """Adopt only the one exact stopped Job-52 state authorized by Job 55."""

    if Path(job51_root) != Path(volume.mount_point) / JOB51_ROOT_RELATIVE:
        raise Tier0Error("Job-51 predecessor root is not canonical", status="STOP_JOB55_PREDECESSOR_DRIFT")
    try:
        parent_meta = (Path(volume.mount_point) / "cmbp-tier0").lstat()
    except FileNotFoundError as exc:
        raise Tier0Error("Tier-0 destination parent is absent", status="STOP_JOB55_PREDECESSOR_DRIFT") from exc
    if stat.S_ISLNK(parent_meta.st_mode) or not stat.S_ISDIR(parent_meta.st_mode) or parent_meta.st_dev != volume.st_dev:
        raise Tier0Error("Tier-0 destination parent is unsafe", status="STOP_JOB55_PREDECESSOR_DRIFT")
    repo_root = Path(__file__).resolve().parents[2]
    bundle = load_cap_amendment_scope_bundle(repo_root)
    manifest = _external_tree_manifest(Path(job51_root), volume=volume)
    if manifest["directories"] != list(pins.directories) or manifest["file_sha256"] != dict(pins.file_sha256):
        raise Tier0Error("Job-51/52 predecessor tree population or bytes drifted", status="STOP_JOB55_PREDECESSOR_DRIFT")
    anchor = base.strict_json(Path(job51_root) / paid.PAID_ATTEMPT_ANCHOR_NAME)
    if (
        anchor.get("anchor_sha256") != pins.anchor_sha256
        or base.self_hash(anchor, "anchor_sha256") != pins.anchor_sha256
    ):
        raise Tier0Error("Job-52 predecessor anchor drifted", status="STOP_JOB55_PREDECESSOR_DRIFT")
    readiness = paid.validate_paid_readiness_receipt(repo_root)
    sdk_sha = base.json_sha256(paid.paid_sdk_identity())
    summary = paid.summarize_paid_attempts(
        Path(job51_root),
        bundle.sessions,
        require_legacy_stop=True,
        expected_readiness_sha256=EXPECTED_JOB52_READINESS_SHA256,
        expected_readiness_file_sha256=EXPECTED_JOB52_READINESS_FILE_SHA256,
        require_client_constructed=True,
        expected_sdk_identity_sha256=sdk_sha,
    )
    contract_evidence = bundle.contract.get("opening_job52_evidence", {})
    commitments = summary.get("commitments")
    terminal = summary.get("terminal_authority_failures")
    partials = summary.get("partial_or_failed_staging")
    exact_commitment = commitments[0] if isinstance(commitments, list) and len(commitments) == 1 else None
    exact_terminal = terminal[0] if isinstance(terminal, list) and len(terminal) == 1 else None
    exact_terminal_observation = next(
        (
            item
            for item in summary.get("quote_observations", [])
            if isinstance(exact_terminal, dict)
            and item.get("attempt_id") == exact_terminal.get("attempt_id")
            and item.get("cost_result_sequence") == exact_terminal.get("cost_result_sequence")
        ),
        None,
    )
    exact_partial = partials[0] if isinstance(partials, list) and len(partials) == 1 else None
    if (
        readiness.get("receipt_sha256") != EXPECTED_JOB52_READINESS_SHA256
        or summary.get("paid_attempt_count") != 2
        or summary.get("cost_call_starts") != 2
        or summary.get("cost_call_results") != 2
        or summary.get("cost_call_errors") != 0
        or summary.get("timeseries_call_starts") != 1
        or summary.get("timeseries_call_results") != 0
        or summary.get("failed_timeseries_starts") != 1
        or summary.get("pending_timeseries_starts") != 0
        or summary.get("committed_quote_total_usd") != "0.950392448902"
        or summary.get("committed_quote_by_session_usd") != {FAILED_SESSION: "0.950392448902"}
        or summary.get("terminal_authority_failure_observed") is not True
        or summary.get("successful_stream_recovery_required") is not False
        or summary.get("published_sessions_in_journals") != []
        or summary.get("aggregate_seal_count") != 0
        or exact_commitment is None
        or exact_commitment.get("commitment_index") != 1
        or exact_commitment.get("attempt_id") != contract_evidence.get("job52_attempt_1_id")
        or exact_commitment.get("timeseries_start_record_hash")
        != contract_evidence.get("job52_attempt_1_time_series_start_record_hash")
        or exact_commitment.get("fresh_quote_usd") != "0.950392448902"
        or exact_terminal is None
        or exact_terminal.get("attempt_id") != contract_evidence.get("old_cap_stop_superseded_only_for_attempt_id")
        or not isinstance(exact_terminal_observation, dict)
        or exact_terminal_observation.get("cost_result_record_hash")
        != contract_evidence.get("old_cap_stop_superseded_only_for_cost_result_record_hash")
        or exact_terminal.get("session") != FAILED_SESSION
        or exact_terminal.get("observed_sdk_quote") != "0.950392448902"
        or exact_partial is None
        or exact_partial.get("compressed_bytes") != pins.failed_partial_bytes
        or exact_partial.get("file_sha256") != pins.failed_partial_sha256
        or exact_partial.get("timeseries_start_record_hash")
        != contract_evidence.get("job52_attempt_1_time_series_start_record_hash")
    ):
        raise Tier0Error("Job-52 semantic opening ledger drifted", status="STOP_JOB55_PREDECESSOR_DRIFT")
    return Job52OpeningEvidence(
        job51_root=str(Path(job51_root)),
        manifest_sha256=base.json_sha256(manifest),
        anchor_sha256=pins.anchor_sha256,
        opening_commitment_total_usd=OPENING_COMMITMENT_USD,
        opening_commitment_by_session_usd=MappingProxyType({FAILED_SESSION: OPENING_COMMITMENT_USD}),
        opening_commitment_count=1,
        cost_call_starts=2,
        cost_call_results=2,
        time_series_call_starts=1,
        time_series_call_results=0,
        failed_partial_bytes=pins.failed_partial_bytes,
        failed_partial_sha256=pins.failed_partial_sha256,
        superseded_terminal_attempt_id=str(contract_evidence["old_cap_stop_superseded_only_for_attempt_id"]),
        superseded_terminal_cost_result_record_hash=str(
            contract_evidence["old_cap_stop_superseded_only_for_cost_result_record_hash"]
        ),
        predecessor_summary=MappingProxyType(dict(summary)),
    )


def validate_job52_opening_evidence(
    job51_root: Path,
    *,
    volume: VolumeIdentity,
) -> Job52OpeningEvidence:
    """Production predecessor adoption with non-overridable frozen pins."""

    observed = _assert_current_production_volume(volume)
    return _validate_job52_opening_evidence_with_pins(
        job51_root,
        volume=observed,
        pins=EXPECTED_JOB52_EVIDENCE_PINS,
    )


def normalize_amended_quote(value: Any) -> str:
    """Normalize a finite unsigned numeric SDK quote without rounding it."""

    return paid.normalize_paid_quote(value)


def _exact_add(left: Decimal, right: Decimal) -> Decimal:
    return paid._exact_nonnegative_add(left, right)


def project_amended_commitment(
    *,
    committed_total_usd: Decimal,
    committed_session_usd: Decimal,
    fresh_quote_usd: Decimal,
    session: str | None = None,
    job55_start_count_for_session: int = 0,
) -> dict[str, Any]:
    """Precision-independent projection for both caps and the singular retry."""

    for value in (committed_total_usd, committed_session_usd, fresh_quote_usd):
        if not isinstance(value, Decimal) or not value.is_finite() or value.is_signed() or value < 0:
            raise Tier0Error("Job-55 commitment projection operand is invalid", status="STOP_JOB55_BUDGET_INVALID")
    if (
        isinstance(job55_start_count_for_session, bool)
        or not isinstance(job55_start_count_for_session, int)
        or job55_start_count_for_session < 0
    ):
        raise Tier0Error("Job-55 session start count is invalid", status="STOP_JOB55_BUDGET_INVALID")
    session_after = _exact_add(committed_session_usd, fresh_quote_usd)
    total_after = _exact_add(committed_total_usd, fresh_quote_usd)
    retry_allowance_pass = not (
        session == FAILED_SESSION and job55_start_count_for_session >= FAILED_SESSION_JOB55_START_ALLOWANCE
    )
    return {
        "committed_quote_session_after_usd": session_after,
        "committed_quote_total_after_usd": total_after,
        "per_session_lifetime_cap_pass": session_after <= PER_SESSION_LIFETIME_CAP_USD,
        "total_cap_pass": total_after <= TOTAL_COMMITTED_CAP_USD,
        "job55_retry_start_allowance_pass": retry_allowance_pass,
        "time_series_start_permitted": bool(
            session_after <= PER_SESSION_LIFETIME_CAP_USD
            and total_after <= TOTAL_COMMITTED_CAP_USD
            and retry_allowance_pass
        ),
    }


@dataclass
class AmendedBudgetState:
    """Combined immutable Job-52 prefix plus disk-derived Job-55 commitments."""

    job55_committed_total_usd: Decimal = Decimal("0")
    job55_committed_by_session_usd: dict[str, Decimal] = field(default_factory=dict)
    job55_start_count_by_session: dict[str, int] = field(default_factory=dict)
    job55_commitment_count: int = 0
    job55_cost_call_count_by_session: dict[str, int] = field(default_factory=dict)
    published_sessions: set[str] = field(default_factory=set)
    current_attempt_cost_calls: int = 0
    current_attempt_time_series_starts: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.job55_committed_total_usd, Decimal):
            self.job55_committed_total_usd = Decimal(str(self.job55_committed_total_usd))
        if (
            not self.job55_committed_total_usd.is_finite()
            or self.job55_committed_total_usd.is_signed()
            or self.job55_committed_total_usd < 0
        ):
            raise Tier0Error("Job-55 budget total is invalid", status="STOP_JOB55_BUDGET_INVALID")
        self.job55_committed_by_session_usd = {
            str(session): value if isinstance(value, Decimal) else Decimal(str(value))
            for session, value in self.job55_committed_by_session_usd.items()
        }
        reconstructed = Decimal("0")
        for session, value in self.job55_committed_by_session_usd.items():
            if not session or not value.is_finite() or value.is_signed() or value < 0:
                raise Tier0Error("Job-55 budget has an invalid session value", status="STOP_JOB55_BUDGET_INVALID")
            reconstructed = _exact_add(reconstructed, value)
        if reconstructed != self.job55_committed_total_usd:
            raise Tier0Error("Job-55 budget total does not reconcile", status="STOP_JOB55_BUDGET_INVALID")
        if isinstance(self.job55_commitment_count, bool) or not isinstance(self.job55_commitment_count, int):
            raise Tier0Error("Job-55 commitment count is invalid", status="STOP_JOB55_BUDGET_INVALID")
        if self.job55_commitment_count < 0 or self.job55_commitment_count != sum(self.job55_start_count_by_session.values()):
            raise Tier0Error("Job-55 commitment count does not reconcile", status="STOP_JOB55_BUDGET_INVALID")
        for session, count in self.job55_start_count_by_session.items():
            if not session or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise Tier0Error("Job-55 session start count is invalid", status="STOP_JOB55_BUDGET_INVALID")
        if self.job55_start_count_by_session.get(FAILED_SESSION, 0) > FAILED_SESSION_JOB55_START_ALLOWANCE:
            raise Tier0Error("Job-55 singular retry allowance was exceeded", status="STOP_JOB55_RETRY_EXHAUSTED")
        self.job55_cost_call_count_by_session = dict(self.job55_cost_call_count_by_session)
        for session, count in self.job55_cost_call_count_by_session.items():
            if not session or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise Tier0Error("Job-55 cost-call count is invalid", status="STOP_JOB55_BUDGET_INVALID")
        if self.job55_cost_call_count_by_session.get(FAILED_SESSION, 0) > FAILED_SESSION_JOB55_COST_CALL_ALLOWANCE:
            raise Tier0Error("Job-55 singular cost-call allowance was exceeded", status="STOP_JOB55_RETRY_EXHAUSTED")
        self.published_sessions = {str(session) for session in self.published_sessions}
        for value, label in (
            (self.current_attempt_cost_calls, "cost calls"),
            (self.current_attempt_time_series_starts, "time-series starts"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 1:
                raise Tier0Error(f"Job-55 current-attempt {label} are invalid", status="STOP_JOB55_BUDGET_INVALID")
        for session, value in self.committed_by_session_usd.items():
            if value > PER_SESSION_LIFETIME_CAP_USD:
                raise Tier0Error("combined session cap is exceeded", status="STOP_JOB55_SESSION_CAP")
        if self.committed_total_usd > TOTAL_COMMITTED_CAP_USD:
            raise Tier0Error("combined total cap is exceeded", status="STOP_JOB55_TOTAL_CAP")

    @classmethod
    def from_opening_evidence(cls, opening: Job52OpeningEvidence | Mapping[str, Any]) -> "AmendedBudgetState":
        if isinstance(opening, Job52OpeningEvidence):
            if (
                opening.opening_commitment_total_usd != OPENING_COMMITMENT_USD
                or dict(opening.opening_commitment_by_session_usd) != {FAILED_SESSION: OPENING_COMMITMENT_USD}
                or isinstance(opening.opening_commitment_count, bool)
                or not isinstance(opening.opening_commitment_count, int)
                or opening.opening_commitment_count != 1
            ):
                raise Tier0Error("Job-52 opening ledger is not exact", status="STOP_JOB55_PREDECESSOR_DRIFT")
        else:
            total = opening.get("opening_commitment_total_usd")
            by_session = opening.get("opening_commitment_by_session_usd")
            count = opening.get("opening_commitment_count")
            if str(total) != "0.950392448902" or by_session not in (
                {FAILED_SESSION: "0.950392448902"},
                {FAILED_SESSION: OPENING_COMMITMENT_USD},
            ) or isinstance(count, bool) or not isinstance(count, int) or count != 1:
                raise Tier0Error("Job-52 opening ledger is not exact", status="STOP_JOB55_PREDECESSOR_DRIFT")
        return cls()

    @property
    def committed_by_session_usd(self) -> dict[str, Decimal]:
        combined = {FAILED_SESSION: OPENING_COMMITMENT_USD}
        for session, value in self.job55_committed_by_session_usd.items():
            combined[session] = _exact_add(combined.get(session, Decimal("0")), value)
        return combined

    @property
    def committed_total_usd(self) -> Decimal:
        return _exact_add(OPENING_COMMITMENT_USD, self.job55_committed_total_usd)

    @property
    def commitment_count(self) -> int:
        return 1 + self.job55_commitment_count

    @property
    def job55_cost_call_count(self) -> int:
        return sum(self.job55_cost_call_count_by_session.values())

    def assert_cost_call_permitted(self, session: str) -> None:
        if self.current_attempt_cost_calls >= 1 or self.current_attempt_time_series_starts >= 1:
            raise Tier0Error("a Job-55 process attempt may make only one vendor pair", status="STOP_JOB55_ATTEMPT_PAIR_LIMIT")
        if self.job55_cost_call_count == 0 and session != FAILED_SESSION:
            raise Tier0Error("the failed session must be the first Job-55 target", status="STOP_JOB55_FIRST_TARGET")
        if session != FAILED_SESSION and FAILED_SESSION not in self.published_sessions:
            raise Tier0Error("later session precedes failed-session publication", status="STOP_JOB55_FIRST_TARGET")
        if (
            session == FAILED_SESSION
            and self.job55_cost_call_count_by_session.get(session, 0) >= FAILED_SESSION_JOB55_COST_CALL_ALLOWANCE
        ):
            raise Tier0Error("Job-55 singular cost-call allowance is exhausted", status="STOP_JOB55_RETRY_EXHAUSTED")

    def note_cost_call(self, session: str) -> None:
        self.assert_cost_call_permitted(session)
        self.job55_cost_call_count_by_session[session] = self.job55_cost_call_count_by_session.get(session, 0) + 1
        self.current_attempt_cost_calls += 1

    def projection(self, session: str, quote: Decimal) -> dict[str, Any]:
        return project_amended_commitment(
            committed_total_usd=self.committed_total_usd,
            committed_session_usd=self.committed_by_session_usd.get(session, Decimal("0")),
            fresh_quote_usd=quote,
            session=session,
            job55_start_count_for_session=self.job55_start_count_by_session.get(session, 0),
        )

    def commit(self, session: str, quote: Decimal, projection: Mapping[str, Any]) -> None:
        expected = self.projection(session, quote)
        if dict(projection) != expected or expected["time_series_start_permitted"] is not True:
            raise Tier0Error("Job-55 commitment cannot be applied", status="STOP_JOB55_BUDGET_INVALID")
        self.job55_committed_by_session_usd[session] = _exact_add(
            self.job55_committed_by_session_usd.get(session, Decimal("0")), quote
        )
        self.job55_committed_total_usd = _exact_add(self.job55_committed_total_usd, quote)
        self.job55_start_count_by_session[session] = self.job55_start_count_by_session.get(session, 0) + 1
        self.job55_commitment_count += 1
        self.current_attempt_time_series_starts += 1


def _safe_quote_observation(value: Any) -> str:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return repr(value)
    return type(value).__name__


def _amended_cost_payload(state: AmendedBudgetState, session: str, quote_text: str) -> dict[str, Any]:
    quote = Decimal(quote_text)
    projection = state.projection(session, quote)
    return {
        "observed_sdk_quote_usd": quote_text,
        "quote_valid": True,
        "job52_carried_commitment_usd": "0.950392448902",
        "combined_commitment_count_before": state.commitment_count,
        "job55_commitment_count_before": state.job55_commitment_count,
        "job55_session_start_count_before": state.job55_start_count_by_session.get(session, 0),
        "combined_committed_quote_session_before_usd": format(
            state.committed_by_session_usd.get(session, Decimal("0")), "f"
        ),
        "combined_committed_quote_session_projected_usd": format(
            projection["committed_quote_session_after_usd"], "f"
        ),
        "combined_committed_quote_total_before_usd": format(state.committed_total_usd, "f"),
        "combined_committed_quote_total_projected_usd": format(
            projection["committed_quote_total_after_usd"], "f"
        ),
        "job55_committed_quote_session_before_usd": format(
            state.job55_committed_by_session_usd.get(session, Decimal("0")), "f"
        ),
        "job55_committed_quote_session_projected_usd": format(
            _exact_add(state.job55_committed_by_session_usd.get(session, Decimal("0")), quote), "f"
        ),
        "job55_committed_quote_total_before_usd": format(state.job55_committed_total_usd, "f"),
        "job55_committed_quote_total_projected_usd": format(
            _exact_add(state.job55_committed_total_usd, quote), "f"
        ),
        "per_session_lifetime_cap_usd": "2.00",
        "total_cap_usd": "32.00",
        "per_session_lifetime_cap_pass": projection["per_session_lifetime_cap_pass"],
        "total_cap_pass": projection["total_cap_pass"],
        "job55_retry_start_allowance_pass": projection["job55_retry_start_allowance_pass"],
        "time_series_start_permitted": projection["time_series_start_permitted"],
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }


def _amended_start_payload(
    *,
    request: SessionRequest,
    output_path: Path,
    quote_text: str,
    cost_result_record_hash: str,
    state: AmendedBudgetState,
    projection: Mapping[str, Any],
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
    ordinal: int,
) -> dict[str, Any]:
    quote = Decimal(quote_text)
    combined_session_before = state.committed_by_session_usd.get(request.session, Decimal("0"))
    job55_session_before = state.job55_committed_by_session_usd.get(request.session, Decimal("0"))
    return {
        "method": "timeseries.get_range",
        "parameters": {**request.market_parameters, "stype_out": base.EXPECTED_STYPE_OUT, "limit": None},
        "output_relative": output_path.name,
        "fresh_quote_usd": quote_text,
        "immediately_preceding_quote_usd": quote_text,
        "cost_result_record_hash": cost_result_record_hash,
        "combined_commitment_index": state.commitment_count + 1,
        "job55_commitment_index": state.job55_commitment_count + 1,
        "ordinal": ordinal,
        "job52_carried_commitment_usd": "0.950392448902",
        "combined_committed_quote_session_before_usd": format(combined_session_before, "f"),
        "combined_committed_quote_session_after_usd": format(
            projection["committed_quote_session_after_usd"], "f"
        ),
        "combined_committed_quote_total_before_usd": format(state.committed_total_usd, "f"),
        "combined_committed_quote_total_after_usd": format(projection["committed_quote_total_after_usd"], "f"),
        "job55_committed_quote_session_before_usd": format(job55_session_before, "f"),
        "job55_committed_quote_session_after_usd": format(_exact_add(job55_session_before, quote), "f"),
        "job55_committed_quote_total_before_usd": format(state.job55_committed_total_usd, "f"),
        "job55_committed_quote_total_after_usd": format(_exact_add(state.job55_committed_total_usd, quote), "f"),
        "job55_session_start_count_before": state.job55_start_count_by_session.get(request.session, 0),
        "per_session_lifetime_cap_usd": "2.00",
        "total_cap_usd": "32.00",
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "readiness_receipt_file_sha256": readiness_receipt_file_sha256,
        "adoption_receipt_sha256": adoption_receipt_sha256,
    }


def acquire_amended_session_bytes(
    client: Any,
    request: SessionRequest,
    *,
    output_path: Path,
    journal: Any,
    pre_pair_gate: Callable[[], None],
    budget_state: AmendedBudgetState,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
    ordinal: int,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Make exactly one fresh capped vendor pair and stream one frozen session."""

    volume = _assert_current_production_volume(volume)
    frozen_bundle = load_cap_amendment_scope_bundle(Path(__file__).resolve().parents[2])
    frozen_by_session = {item.session: item for item in frozen_bundle.sessions}
    frozen_request = frozen_by_session.get(request.session)
    frozen_ordinal = next(
        (index for index, item in enumerate(frozen_bundle.sessions, start=1) if item.session == request.session), None
    )
    if (
        frozen_request is None
        or request != frozen_request
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal != frozen_ordinal
    ):
        raise Tier0Error("Job-55 request is not the exact frozen session request", status="STOP_JOB55_SCOPE_WIDENING")
    if any(
        not isinstance(value, str) or base.SHA256_RE.fullmatch(value) is None
        for value in (readiness_receipt_sha256, readiness_receipt_file_sha256, adoption_receipt_sha256)
    ):
        raise Tier0Error("Job-55 execution evidence identity is invalid", status="STOP_JOB55_READINESS_DRIFT")
    if not isinstance(journal, base.AttemptJournal):
        raise Tier0Error("Job-55 acquisition requires a durable attempt journal", status="STOP_JOB55_JOURNAL_INVALID")
    verified_before = base.verify_attempt_journal(journal.attempt_dir)
    if any(record.get("event") in base.CALL_EVENTS for record in verified_before["records"]):
        raise Tier0Error("Job-55 process attempt already contains a vendor pair", status="STOP_JOB55_ATTEMPT_PAIR_LIMIT")
    output_path = Path(output_path)
    attempt_dir = journal.attempt_dir
    expected_job55_root = Path(volume.mount_point) / JOB55_ROOT_RELATIVE
    expected_attempt = expected_job55_root / "attempts" / journal.attempt_id
    expected_output = attempt_dir / "sessions" / f"{request.session}.bundle.part" / "data.cmbp-1.dbn.zst"
    if (
        attempt_dir != expected_attempt
        or output_path != expected_output
        or base.UUID4_RE.fullmatch(journal.attempt_id) is None
    ):
        raise Tier0Error("Job-55 output path is outside the exact staging namespace", status="STOP_JOB55_EXTERNAL_PATH")
    try:
        parent_meta = (Path(volume.mount_point) / "cmbp-tier0").lstat()
        root_meta = expected_job55_root.lstat()
        attempts_meta = (expected_job55_root / "attempts").lstat()
        attempt_meta = attempt_dir.lstat()
        sessions_meta = (attempt_dir / "sessions").lstat()
        staging_meta = output_path.parent.lstat()
    except FileNotFoundError as exc:
        raise Tier0Error("Job-55 staging directory is absent", status="STOP_JOB55_EXTERNAL_PATH") from exc
    if any(
        stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode) or item.st_dev != attempt_meta.st_dev
        for item in (parent_meta, root_meta, attempts_meta, attempt_meta, sessions_meta, staging_meta)
    ):
        raise Tier0Error("Job-55 staging path is unsafe or cross-device", status="STOP_JOB55_EXTERNAL_PATH")
    if any(
        item.st_dev != volume.st_dev
        for item in (parent_meta, root_meta, attempts_meta, attempt_meta, sessions_meta, staging_meta)
    ):
        raise Tier0Error("Job-55 staging path is not on the pinned volume", status="STOP_JOB55_EXTERNAL_PATH")
    adoption_path = expected_job55_root / JOB55_ADOPTION_NAME
    _private_regular(adoption_path, status="STOP_JOB55_ADOPTION_DRIFT")
    adoption = base.strict_json(adoption_path)
    if (
        adoption.get("adoption_sha256") != adoption_receipt_sha256
        or base.self_hash(adoption, "adoption_sha256") != adoption_receipt_sha256
    ):
        raise Tier0Error("Job-55 adoption receipt is not bound to this pair", status="STOP_JOB55_ADOPTION_DRIFT")
    anchor = validate_job55_attempt_anchor(
        expected_job55_root,
        volume=volume,
        readiness_receipt_sha256=readiness_receipt_sha256,
        readiness_receipt_file_sha256=readiness_receipt_file_sha256,
        adoption_receipt_sha256=adoption_receipt_sha256,
        allow_initialize=False,
        repair_header_only=False,
    )
    if journal.attempt_id not in {item["attempt_id"] for item in anchor["job55_attempts"]}:
        raise Tier0Error("Job-55 current attempt is not anchored", status="STOP_JOB55_ATTEMPT_SET")
    if output_path.exists() or output_path.is_symlink():
        raise Tier0Error("Job-55 output path already exists", status="STOP_JOB55_EXTERNAL_PATH")
    budget_state.assert_cost_call_permitted(request.session)
    if progress is not None:
        progress(f"{request.session}: requesting fresh amended-cap quote")
    pre_pair_gate()
    canonical = reconstruct_job55_local_state(volume=volume, require_client_constructed=True)
    disk_state = canonical["budget_state"]
    if (
        disk_state.job55_committed_total_usd != budget_state.job55_committed_total_usd
        or disk_state.job55_committed_by_session_usd != budget_state.job55_committed_by_session_usd
        or disk_state.job55_start_count_by_session != budget_state.job55_start_count_by_session
        or disk_state.job55_commitment_count != budget_state.job55_commitment_count
        or disk_state.job55_cost_call_count_by_session != budget_state.job55_cost_call_count_by_session
        or disk_state.published_sessions != budget_state.published_sessions
        or canonical["readiness"]["receipt_sha256"] != readiness_receipt_sha256
        or canonical["readiness_file_sha256"] != readiness_receipt_file_sha256
        or canonical["adoption"]["adoption_sha256"] != adoption_receipt_sha256
        or request.session in canonical["final_qc"]
    ):
        raise Tier0Error("Job-55 caller state does not match canonical disk reconstruction", status="STOP_JOB55_BUDGET_INVALID")
    request_sha = request.market_request_sha256
    journal.append(
        "COST_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload={"method": "metadata.get_cost", "parameters": request.market_parameters},
    )
    budget_state.note_cost_call(request.session)
    try:
        observed = client.metadata.get_cost(**request.market_parameters)
    except Exception as exc:  # noqa: BLE001 - vendor text is deliberately redacted
        journal.append(
            "COST_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={"error_class": type(exc).__name__, "actual_vendor_invoice_cost_usd": "UNKNOWN"},
        )
        raise Tier0Error("fresh Job-55 SDK cost request failed", status="STOP_JOB55_VENDOR_COST_CALL") from exc
    session_before = budget_state.committed_by_session_usd.get(request.session, Decimal("0"))
    try:
        quote_text = normalize_amended_quote(observed)
        quote_decimal = Decimal(quote_text)
    except Tier0Error:
        journal.append(
            "COST_CALL_RESULT",
            session=request.session,
            request_sha256=request_sha,
            payload={
                "observed_sdk_quote": _safe_quote_observation(observed),
                "quote_valid": False,
                "combined_commitment_count_before": budget_state.commitment_count,
                "job55_commitment_count_before": budget_state.job55_commitment_count,
                "job55_session_start_count_before": budget_state.job55_start_count_by_session.get(request.session, 0),
                "combined_committed_quote_session_before_usd": format(session_before, "f"),
                "combined_committed_quote_total_before_usd": format(budget_state.committed_total_usd, "f"),
                "job55_committed_quote_session_before_usd": format(
                    budget_state.job55_committed_by_session_usd.get(request.session, Decimal("0")), "f"
                ),
                "job55_committed_quote_total_before_usd": format(budget_state.job55_committed_total_usd, "f"),
                "per_session_lifetime_cap_usd": "2.00",
                "total_cap_usd": "32.00",
                "per_session_lifetime_cap_pass": False,
                "total_cap_pass": False,
                "job55_retry_start_allowance_pass": False,
                "time_series_start_permitted": False,
                "actual_vendor_invoice_cost_usd": "UNKNOWN",
            },
        )
        raise
    projection = budget_state.projection(request.session, quote_decimal)
    cost_result = journal.append(
        "COST_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload=_amended_cost_payload(budget_state, request.session, quote_text),
    )
    if not projection["job55_retry_start_allowance_pass"]:
        raise Tier0Error("Job-55 singular retry allowance is exhausted", status="STOP_JOB55_RETRY_EXHAUSTED")
    if not projection["per_session_lifetime_cap_pass"]:
        raise Tier0Error("session lifetime commitment would exceed USD 2.00", status="STOP_JOB55_SESSION_CAP")
    if not projection["total_cap_pass"]:
        raise Tier0Error("combined commitment would exceed USD 32.00", status="STOP_JOB55_TOTAL_CAP")
    # The result record and the irrevocable start record are adjacent.  No
    # callback, stdout, local work, or second vendor call may intervene.
    start = journal.append(
        "TIMESERIES_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload=_amended_start_payload(
            request=request,
            output_path=output_path,
            quote_text=quote_text,
            cost_result_record_hash=str(cost_result["record_hash"]),
            state=budget_state,
            projection=projection,
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            adoption_receipt_sha256=adoption_receipt_sha256,
            ordinal=ordinal,
        ),
    )
    budget_state.commit(request.session, quote_decimal, projection)
    result_context = {
        "fresh_quote_usd": quote_text,
        "combined_commitment_index": budget_state.commitment_count,
        "job55_commitment_index": budget_state.job55_commitment_count,
        "timeseries_start_record_hash": start["record_hash"],
        "combined_committed_quote_session_usd": format(
            budget_state.committed_by_session_usd[request.session], "f"
        ),
        "combined_committed_quote_total_usd": format(budget_state.committed_total_usd, "f"),
        "job55_committed_quote_session_usd": format(
            budget_state.job55_committed_by_session_usd[request.session], "f"
        ),
        "job55_committed_quote_total_usd": format(budget_state.job55_committed_total_usd, "f"),
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }
    try:
        client.timeseries.get_range(
            **request.market_parameters,
            stype_out=base.EXPECTED_STYPE_OUT,
            limit=None,
            path=output_path,
        )
    except Exception as exc:  # noqa: BLE001 - redact vendor exception text
        journal.append(
            "TIMESERIES_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={**result_context, "error_class": type(exc).__name__},
        )
        raise Tier0Error("Job-55 time-series stream failed", status="STOP_JOB55_VENDOR_TIMESERIES_CALL") from exc
    metadata = output_path.lstat() if output_path.exists() or output_path.is_symlink() else None
    parent_metadata = output_path.parent.lstat()
    if (
        metadata is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_dev != parent_metadata.st_dev
        or metadata.st_size <= 0
    ):
        journal.append(
            "TIMESERIES_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={**result_context, "error_class": "MissingOrEmptyOutput"},
        )
        raise Tier0Error("Job-55 stream produced no safe file", status="STOP_JOB55_VENDOR_TIMESERIES_CALL")
    descriptor = os.open(output_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    base.fsync_directory(output_path.parent)
    journal.append(
        "TIMESERIES_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload={
            **result_context,
            "compressed_bytes": metadata.st_size,
            "dbn_file_sha256": base.file_sha256(output_path),
        },
    )
    if progress is not None:
        progress(f"{request.session}: amended-cap DBN stream durably recorded")
    return quote_text


Job55RunLock = paid.PaidRunLock


def initialize_job55_destination_tree(volume: VolumeIdentity) -> Path:
    """Create only the authorized sibling namespace on the verified mount."""

    volume = _assert_current_production_volume(volume)
    mount = Path(volume.mount_point)
    parent = mount / "cmbp-tier0"
    job55_root = mount / JOB55_ROOT_RELATIVE
    base.ensure_nofollow_directory(parent, volume=volume, create=False)
    for path in (job55_root, job55_root / "attempts", job55_root / "sessions", job55_root / "receipts"):
        base.ensure_nofollow_directory(path, volume=volume, create=True)
    return job55_root


def _stable_volume_fields(volume: VolumeIdentity) -> dict[str, Any]:
    return {
        "mount_point": volume.mount_point,
        "volume_uuid": volume.volume_uuid,
        "device_identifier": volume.device_identifier,
        "filesystem": volume.filesystem,
        "bus_protocol": volume.bus_protocol,
        "st_dev": volume.st_dev,
        "total_bytes": volume.total_bytes,
    }


def _validate_volume_payload(payload: Any, *, current: VolumeIdentity, status: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(dataclasses.asdict(current)):
        raise Tier0Error("Job-55 volume payload fields drifted", status=status)
    int_fields = ("st_dev", "free_bytes", "total_bytes")
    str_fields = ("mount_point", "volume_uuid", "device_identifier", "filesystem", "bus_protocol")
    if any(isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) or payload[key] < 0 for key in int_fields):
        raise Tier0Error("Job-55 volume numeric field is invalid", status=status)
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in str_fields):
        raise Tier0Error("Job-55 volume string field is invalid", status=status)
    if {key: payload[key] for key in _stable_volume_fields(current)} != _stable_volume_fields(current):
        raise Tier0Error("Job-55 stable volume identity drifted", status=status)
    return dict(payload)


def build_job52_terminal_adoption_receipt(
    opening: Job52OpeningEvidence,
    *,
    volume: VolumeIdentity,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "artifact_type": ADOPTION_ARTIFACT,
        "schema_version": "v5.job55-cmbp-tier0-job52-terminal-adoption.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "predecessor_job_id": PREDECESSOR_JOB_ID,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "job52_program_contract_sha256": EXPECTED_JOB52_CONTRACT_SHA256,
        "job52_program_contract_file_sha256": EXPECTED_JOB52_CONTRACT_FILE_SHA256,
        "job52_readiness_receipt_sha256": EXPECTED_JOB52_READINESS_SHA256,
        "job52_readiness_receipt_file_sha256": EXPECTED_JOB52_READINESS_FILE_SHA256,
        "job51_root_relative": str(JOB51_ROOT_RELATIVE),
        "job51_52_tree_manifest_sha256": opening.manifest_sha256,
        "job52_anchor_sha256": opening.anchor_sha256,
        "opening_commitment_count": opening.opening_commitment_count,
        "opening_commitment_total_usd": format(opening.opening_commitment_total_usd, "f"),
        "opening_commitment_by_session_usd": {
            session: format(value, "f") for session, value in opening.opening_commitment_by_session_usd.items()
        },
        "job52_cost_call_starts": opening.cost_call_starts,
        "job52_cost_call_results": opening.cost_call_results,
        "job52_time_series_call_starts": opening.time_series_call_starts,
        "job52_time_series_call_results": opening.time_series_call_results,
        "failed_partial_bytes": opening.failed_partial_bytes,
        "failed_partial_file_sha256": opening.failed_partial_sha256,
        "superseded_terminal_attempt_id": opening.superseded_terminal_attempt_id,
        "superseded_terminal_cost_result_record_hash": opening.superseded_terminal_cost_result_record_hash,
        "supersession_scope": "ONLY_THIS_EXACT_JOB52_USD_1_50_CAP_RESULT_UNDER_THE_CURRENT_JOB55_USD_2_00_AUTHORITY",
        "prior_partial_reuse_permitted": False,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "volume_snapshot": dataclasses.asdict(volume),
        "threat_boundary": "DETECTS_INCONSISTENT_OR_MISSING_DESTINATION_COMPONENTS; COORDINATED_REWRITE_REQUIRES_OUTSIDE_RETAINED_HEAD_OR_SIGNATURE",
    }
    receipt["adoption_sha256"] = base.self_hash(receipt, "adoption_sha256")
    return receipt


def ensure_job52_terminal_adoption(
    job55_root: Path,
    opening: Job52OpeningEvidence,
    *,
    volume: VolumeIdentity,
    allow_initialize: bool = False,
) -> dict[str, Any]:
    volume = _assert_current_production_volume(volume)
    if Path(job55_root) != Path(volume.mount_point) / JOB55_ROOT_RELATIVE:
        raise Tier0Error("Job-55 adoption target is not canonical", status="STOP_JOB55_ADOPTION_DRIFT")
    path = Path(job55_root) / JOB55_ADOPTION_NAME
    if not path.exists() and not path.is_symlink():
        attempts_root = Path(job55_root) / "attempts"
        controls_present = any(
            (Path(job55_root) / name).exists() or (Path(job55_root) / name).is_symlink()
            for name in (JOB55_LOCK_BINDING_NAME, JOB55_ATTEMPT_ANCHOR_NAME, JOB55_AGGREGATE_SEAL_NAME)
        )
        if not allow_initialize or controls_present or any(attempts_root.iterdir()):
            raise Tier0Error("Job-55 adoption receipt is absent after state exists", status="STOP_JOB55_ADOPTION_DRIFT")
        base.write_canonical_exclusive(path, build_job52_terminal_adoption_receipt(opening, volume=volume))
    _private_regular(path, status="STOP_JOB55_ADOPTION_DRIFT")
    if path.stat().st_dev != volume.st_dev:
        raise Tier0Error("Job-55 adoption receipt is cross-device", status="STOP_JOB55_ADOPTION_DRIFT")
    actual = base.strict_json(path)
    if actual.get("adoption_sha256") != base.self_hash(actual, "adoption_sha256"):
        raise Tier0Error("Job-55 adoption self-hash drifted", status="STOP_JOB55_ADOPTION_DRIFT")
    recorded_volume = _validate_volume_payload(
        actual.get("volume_snapshot"), current=volume, status="STOP_JOB55_ADOPTION_DRIFT"
    )
    recorded_identity = VolumeIdentity(**recorded_volume)
    expected = build_job52_terminal_adoption_receipt(opening, volume=recorded_identity)
    if actual != expected:
        raise Tier0Error("Job-55 adoption receipt does not reconstruct", status="STOP_JOB55_ADOPTION_DRIFT")
    return actual


def ensure_job55_lock_binding(
    job55_root: Path,
    *,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
    allow_initialize: bool = False,
) -> dict[str, Any]:
    volume = _assert_current_production_volume(volume)
    if Path(job55_root) != Path(volume.mount_point) / JOB55_ROOT_RELATIVE:
        raise Tier0Error("Job-55 lock-binding target is not canonical", status="STOP_JOB55_LOCK_BINDING")
    for value in (readiness_receipt_sha256, readiness_receipt_file_sha256, adoption_receipt_sha256):
        if base.SHA256_RE.fullmatch(str(value)) is None:
            raise Tier0Error("Job-55 lock identity is invalid", status="STOP_JOB55_LOCK_BINDING")
    receipt: dict[str, Any] = {
        "artifact_type": "JOB55_CMBP_TIER0_CAP_AMENDMENT_RUN_LOCK_BINDING_V1",
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-run-lock-binding.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "predecessor_job_id": PREDECESSOR_JOB_ID,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "readiness_receipt_file_sha256": readiness_receipt_file_sha256,
        "adoption_receipt_sha256": adoption_receipt_sha256,
        "shared_flock_relative": str(JOB51_ROOT_RELATIVE / "RUN_LOCK_V1"),
        "shared_flock_file_sha256": EXPECTED_JOB52_EVIDENCE_PINS.file_sha256["RUN_LOCK_V1"],
        "shared_flock_bytes_mutated": False,
    }
    receipt["binding_sha256"] = base.self_hash(receipt, "binding_sha256")
    path = Path(job55_root) / JOB55_LOCK_BINDING_NAME
    if not path.exists() and not path.is_symlink():
        attempts = Path(job55_root) / "attempts"
        state_exists = any(attempts.iterdir()) or (Path(job55_root) / JOB55_ATTEMPT_ANCHOR_NAME).exists()
        if not allow_initialize or state_exists:
            raise Tier0Error("Job-55 lock binding is absent after state exists", status="STOP_JOB55_LOCK_BINDING")
        base.write_canonical_exclusive(path, receipt)
    _private_regular(path, status="STOP_JOB55_LOCK_BINDING")
    if path.stat().st_dev != volume.st_dev or base.strict_json(path) != receipt:
        raise Tier0Error("Job-55 lock binding drifted", status="STOP_JOB55_LOCK_BINDING")
    return receipt


def build_job55_attempt_header_payload(
    *,
    attempt_id: str,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
    job55_attempt_ordinal: int,
    previous_job55_attempt_id: str | None,
    previous_job55_attempt_header_record_hash: str | None,
) -> dict[str, Any]:
    return {
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "predecessor_job_id": PREDECESSOR_JOB_ID,
        "authority": "OWNER_CURRENT_CONVERSATION_EXACT_USD_2_SESSION_CAP_AMENDMENT",
        "attempt_id": attempt_id,
        "job55_attempt_ordinal": job55_attempt_ordinal,
        "previous_job55_attempt_id": previous_job55_attempt_id,
        "previous_job55_attempt_header_record_hash": previous_job55_attempt_header_record_hash,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "readiness_receipt_file_sha256": readiness_receipt_file_sha256,
        "adoption_receipt_sha256": adoption_receipt_sha256,
        "opening_combined_commitment_count": 1,
        "opening_combined_commitment_total_usd": "0.950392448902",
        "per_session_lifetime_cap_usd": "2.00",
        "total_cap_usd": "32.00",
        "maximum_time_series_starts_this_attempt": 1,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "volume_identity": dataclasses.asdict(volume),
    }


def _validate_job55_header(
    record: Mapping[str, Any],
    *,
    attempt_dir: Path,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
) -> dict[str, Any]:
    if (
        record.get("event") != "ATTEMPT_START"
        or record.get("session") is not None
        or record.get("request_sha256") is not None
        or record.get("attempt_id") != attempt_dir.name
    ):
        raise Tier0Error("Job-55 attempt header record drifted", status="STOP_JOB55_ATTEMPT_SET")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise Tier0Error("Job-55 attempt header payload is absent", status="STOP_JOB55_ATTEMPT_SET")
    recorded = _validate_volume_payload(
        payload.get("volume_identity"), current=volume, status="STOP_JOB55_ATTEMPT_SET"
    )
    ordinal = payload.get("job55_attempt_ordinal")
    previous_id = payload.get("previous_job55_attempt_id")
    previous_hash = payload.get("previous_job55_attempt_header_record_hash")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise Tier0Error("Job-55 attempt ordinal is invalid", status="STOP_JOB55_ATTEMPT_SET")
    if previous_id is not None and base.UUID4_RE.fullmatch(str(previous_id)) is None:
        raise Tier0Error("Job-55 previous attempt ID is invalid", status="STOP_JOB55_ATTEMPT_SET")
    if previous_hash is not None and base.SHA256_RE.fullmatch(str(previous_hash)) is None:
        raise Tier0Error("Job-55 previous header hash is invalid", status="STOP_JOB55_ATTEMPT_SET")
    expected = build_job55_attempt_header_payload(
        attempt_id=attempt_dir.name,
        volume=VolumeIdentity(**recorded),
        readiness_receipt_sha256=readiness_receipt_sha256,
        readiness_receipt_file_sha256=readiness_receipt_file_sha256,
        adoption_receipt_sha256=adoption_receipt_sha256,
        job55_attempt_ordinal=ordinal,
        previous_job55_attempt_id=previous_id,
        previous_job55_attempt_header_record_hash=previous_hash,
    )
    if payload != expected:
        raise Tier0Error("Job-55 attempt header fields drifted", status="STOP_JOB55_ATTEMPT_SET")
    return payload


def _validate_job55_attempt_lineage(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = sorted(entries, key=lambda item: item.get("job55_attempt_ordinal", -1))
    if [item.get("job55_attempt_ordinal") for item in ordered] != list(range(1, len(ordered) + 1)):
        raise Tier0Error("Job-55 attempt ordinals are not contiguous", status="STOP_JOB55_ATTEMPT_SET")
    previous: Mapping[str, Any] | None = None
    for entry in ordered:
        if entry.get("previous_job55_attempt_id") != (None if previous is None else previous.get("attempt_id")):
            raise Tier0Error("Job-55 attempt ID lineage is broken", status="STOP_JOB55_ATTEMPT_SET")
        if entry.get("previous_job55_attempt_header_record_hash") != (
            None if previous is None else previous.get("header_record_hash")
        ):
            raise Tier0Error("Job-55 header-hash lineage is broken", status="STOP_JOB55_ATTEMPT_SET")
        previous = entry
    return ordered


def _job55_anchor_receipt(
    *,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "artifact_type": "JOB55_CMBP_TIER0_CAP_AMENDMENT_ATTEMPT_SET_ANCHOR_V1",
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-attempt-set-anchor.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "readiness_receipt_file_sha256": readiness_receipt_file_sha256,
        "adoption_receipt_sha256": adoption_receipt_sha256,
        "job55_attempt_count": len(attempts),
        "job55_attempts": sorted((dict(item) for item in attempts), key=lambda item: item["job55_attempt_ordinal"]),
        "threat_boundary": "DETECTS_MISSING_OR_ROLLED_BACK_SINGLE_DESTINATION_COMPONENT; COORDINATED_REWRITE_REQUIRES_OUTSIDE_ANCHOR",
    }
    receipt["anchor_sha256"] = base.self_hash(receipt, "anchor_sha256")
    return receipt


def validate_job55_attempt_anchor(
    job55_root: Path,
    *,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
    allow_initialize: bool = False,
    repair_header_only: bool = False,
) -> dict[str, Any]:
    volume = _assert_current_production_volume(volume)
    if Path(job55_root) != Path(volume.mount_point) / JOB55_ROOT_RELATIVE:
        raise Tier0Error("Job-55 anchor target is not canonical", status="STOP_JOB55_ATTEMPT_SET")
    root = Path(job55_root)
    attempts_root = root / "attempts"
    base.ensure_nofollow_directory(attempts_root, volume=volume, create=False)
    actual: dict[str, dict[str, Any]] = {}
    for attempt_dir in sorted(attempts_root.iterdir()):
        if attempt_dir.is_symlink() or not attempt_dir.is_dir() or base.UUID4_RE.fullmatch(attempt_dir.name) is None:
            raise Tier0Error("Job-55 attempt set has an unsafe child", status="STOP_JOB55_ATTEMPT_SET")
        if attempt_dir.stat().st_dev != volume.st_dev:
            raise Tier0Error("Job-55 attempt is cross-device", status="STOP_JOB55_ATTEMPT_SET")
        verified = base.verify_attempt_journal(attempt_dir)
        paid._validate_marker_semantics(attempt_dir, verified["records"])
        header = _validate_job55_header(
            verified["records"][0],
            attempt_dir=attempt_dir,
            volume=volume,
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            adoption_receipt_sha256=adoption_receipt_sha256,
        )
        actual[attempt_dir.name] = {
            "attempt_id": attempt_dir.name,
            "header_record_hash": verified["records"][0]["record_hash"],
            "job55_attempt_ordinal": header["job55_attempt_ordinal"],
            "previous_job55_attempt_id": header["previous_job55_attempt_id"],
            "previous_job55_attempt_header_record_hash": header[
                "previous_job55_attempt_header_record_hash"
            ],
            "readiness_receipt_sha256": readiness_receipt_sha256,
            "adoption_receipt_sha256": adoption_receipt_sha256,
        }
    _validate_job55_attempt_lineage(list(actual.values()))
    anchor_path = root / JOB55_ATTEMPT_ANCHOR_NAME
    if not anchor_path.exists() and not anchor_path.is_symlink():
        if not allow_initialize:
            raise Tier0Error("Job-55 attempt-set anchor is absent", status="STOP_JOB55_ATTEMPT_SET")
        advanced = [
            attempt_id
            for attempt_id in actual
            if len(base.verify_attempt_journal(attempts_root / attempt_id)["records"]) != 1
        ]
        if advanced:
            raise Tier0Error("Job-55 anchor vanished after an attempt advanced", status="STOP_JOB55_ATTEMPT_SET")
        base.write_canonical_exclusive(
            anchor_path,
            _job55_anchor_receipt(
                readiness_receipt_sha256=readiness_receipt_sha256,
                readiness_receipt_file_sha256=readiness_receipt_file_sha256,
                adoption_receipt_sha256=adoption_receipt_sha256,
                attempts=list(actual.values()),
            ),
        )
    _private_regular(anchor_path, status="STOP_JOB55_ATTEMPT_SET")
    if anchor_path.stat().st_dev != volume.st_dev:
        raise Tier0Error("Job-55 anchor is cross-device", status="STOP_JOB55_ATTEMPT_SET")
    anchor = base.strict_json(anchor_path)
    if anchor.get("anchor_sha256") != base.self_hash(anchor, "anchor_sha256"):
        raise Tier0Error("Job-55 anchor self-hash drifted", status="STOP_JOB55_ATTEMPT_SET")
    entries = anchor.get("job55_attempts")
    expected = _job55_anchor_receipt(
        readiness_receipt_sha256=readiness_receipt_sha256,
        readiness_receipt_file_sha256=readiness_receipt_file_sha256,
        adoption_receipt_sha256=adoption_receipt_sha256,
        attempts=entries if isinstance(entries, list) else [],
    )
    if anchor != expected:
        raise Tier0Error("Job-55 anchor fields drifted", status="STOP_JOB55_ATTEMPT_SET")
    anchored = {str(item.get("attempt_id")): item for item in entries if isinstance(item, dict)}
    if len(anchored) != len(entries):
        raise Tier0Error("Job-55 anchor has duplicate/malformed entries", status="STOP_JOB55_ATTEMPT_SET")
    _validate_job55_attempt_lineage(list(anchored.values()))
    missing = set(anchored) - set(actual)
    extra = set(actual) - set(anchored)
    if missing:
        raise Tier0Error("a Job-55 attempt disappeared", status="STOP_JOB55_ATTEMPT_SET")
    if extra:
        if not repair_header_only:
            raise Tier0Error("an unanchored Job-55 attempt exists", status="STOP_JOB55_ATTEMPT_SET")
        for attempt_id in extra:
            if len(base.verify_attempt_journal(attempts_root / attempt_id)["records"]) != 1:
                raise Tier0Error("an advanced Job-55 attempt is unanchored", status="STOP_JOB55_ATTEMPT_SET")
        repaired = _job55_anchor_receipt(
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            adoption_receipt_sha256=adoption_receipt_sha256,
            attempts=list(actual.values()),
        )
        base.write_canonical_replace(anchor_path, repaired)
        anchor = repaired
        anchored = {item["attempt_id"]: item for item in repaired["job55_attempts"]}
    if anchored != actual:
        raise Tier0Error("Job-55 anchor/header binding drifted", status="STOP_JOB55_ATTEMPT_SET")
    return anchor


_JOB55_EVENTS = frozenset(
    {
        "ATTEMPT_START",
        "CLIENT_CONSTRUCTED",
        "COST_CALL_START",
        "COST_CALL_RESULT",
        "COST_CALL_ERROR",
        "TIMESERIES_CALL_START",
        "TIMESERIES_CALL_RESULT",
        "TIMESERIES_CALL_ERROR",
        "SESSION_PUBLISHED",
        "SESSION_RECOVERED",
        "ATTEMPT_STOP",
    }
)


def _validate_job55_attempt_tree(
    attempt_dir: Path,
    *,
    allowed_sessions: set[str],
    volume: VolumeIdentity,
) -> dict[str, list[str]]:
    expected_files = {"ACQUISITION_JOURNAL_V1.jsonl", "JOURNAL_WATERMARK_V1.json"}
    allowed_root = expected_files | {"call-markers", "sessions", JOB55_ATTEMPT_STOP_NAME}
    children = {path.name: path for path in attempt_dir.iterdir()}
    if set(children) - allowed_root or not expected_files.issubset(children) or not {"call-markers", "sessions"}.issubset(children):
        raise Tier0Error("Job-55 attempt tree population drifted", status="STOP_JOB55_JOURNAL_INVALID")
    for name, path in children.items():
        metadata = path.lstat()
        if metadata.st_dev != volume.st_dev or stat.S_ISLNK(metadata.st_mode):
            raise Tier0Error("Job-55 attempt child is unsafe", status="STOP_JOB55_JOURNAL_INVALID")
        if name in {"call-markers", "sessions"}:
            if not stat.S_ISDIR(metadata.st_mode):
                raise Tier0Error("Job-55 attempt directory child drifted", status="STOP_JOB55_JOURNAL_INVALID")
        elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Tier0Error("Job-55 attempt evidence file is unsafe", status="STOP_JOB55_JOURNAL_INVALID")
    staging: dict[str, list[str]] = {}
    sessions_root = attempt_dir / "sessions"
    for child in sorted(sessions_root.iterdir()):
        if child.is_symlink() or not child.is_dir() or child.stat().st_dev != volume.st_dev:
            raise Tier0Error("Job-55 staging child is unsafe", status="STOP_JOB55_JOURNAL_INVALID")
        suffix = ".bundle.part"
        if not child.name.endswith(suffix) or child.name[: -len(suffix)] not in allowed_sessions:
            raise Tier0Error("Job-55 staging session is out of scope", status="STOP_JOB55_SCOPE_WIDENING")
        names: list[str] = []
        for item in sorted(child.iterdir()):
            metadata = item.lstat()
            if (
                item.name not in {"data.cmbp-1.dbn.zst", JOB55_SESSION_QC_NAME}
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_dev != volume.st_dev
            ):
                raise Tier0Error("Job-55 staging file is unsafe or unexpected", status="STOP_JOB55_JOURNAL_INVALID")
            names.append(item.name)
        staging[child.name[: -len(suffix)]] = names
    return staging


def build_job55_attempt_stop_receipt(
    *,
    attempt_id: str,
    terminal_record: Mapping[str, Any],
    readiness_receipt_sha256: str,
    adoption_receipt_sha256: str,
) -> dict[str, Any]:
    payload = terminal_record.get("payload", {})
    receipt: dict[str, Any] = {
        "artifact_type": ATTEMPT_STOP_ARTIFACT,
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-attempt-stop.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "attempt_id": attempt_id,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "adoption_receipt_sha256": adoption_receipt_sha256,
        "status": payload.get("status"),
        "error_class": payload.get("error_class"),
        "journal_terminal_sequence": terminal_record.get("sequence"),
        "journal_terminal_head": terminal_record.get("record_hash"),
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }
    receipt["stop_sha256"] = base.self_hash(receipt, "stop_sha256")
    return receipt


def write_job55_attempt_stop_receipt(
    attempt_dir: Path,
    *,
    terminal_record: Mapping[str, Any],
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    adoption_receipt_sha256: str,
) -> Path:
    volume = _assert_current_production_volume(volume)
    expected_attempts = Path(volume.mount_point) / JOB55_ROOT_RELATIVE / "attempts"
    if Path(attempt_dir).parent != expected_attempts or base.UUID4_RE.fullmatch(Path(attempt_dir).name) is None:
        raise Tier0Error("Job-55 stop target is not canonical", status="STOP_JOB55_JOURNAL_INVALID")
    path = Path(attempt_dir) / JOB55_ATTEMPT_STOP_NAME
    receipt = build_job55_attempt_stop_receipt(
        attempt_id=Path(attempt_dir).name,
        terminal_record=terminal_record,
        readiness_receipt_sha256=readiness_receipt_sha256,
        adoption_receipt_sha256=adoption_receipt_sha256,
    )
    base.write_canonical_exclusive(path, receipt)
    return path


def _validate_job55_stop(
    attempt_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    readiness_receipt_sha256: str,
    adoption_receipt_sha256: str,
) -> str | None:
    path = Path(attempt_dir) / JOB55_ATTEMPT_STOP_NAME
    terminal = records[-1]
    if terminal.get("event") != "ATTEMPT_STOP":
        if path.exists() or path.is_symlink():
            raise Tier0Error("Job-55 stop receipt lacks a terminal stop", status="STOP_JOB55_JOURNAL_INVALID")
        return None
    _private_regular(path, status="STOP_JOB55_JOURNAL_INVALID")
    actual = base.strict_json(path)
    expected = build_job55_attempt_stop_receipt(
        attempt_id=Path(attempt_dir).name,
        terminal_record=terminal,
        readiness_receipt_sha256=readiness_receipt_sha256,
        adoption_receipt_sha256=adoption_receipt_sha256,
    )
    if actual != expected:
        raise Tier0Error("Job-55 stop receipt drifted", status="STOP_JOB55_JOURNAL_INVALID")
    return base.file_sha256(path)


def _invalid_amended_cost_payload(state: AmendedBudgetState, session: str, observed: Any) -> dict[str, Any]:
    return {
        "observed_sdk_quote": _safe_quote_observation(observed),
        "quote_valid": False,
        "combined_commitment_count_before": state.commitment_count,
        "job55_commitment_count_before": state.job55_commitment_count,
        "job55_session_start_count_before": state.job55_start_count_by_session.get(session, 0),
        "combined_committed_quote_session_before_usd": format(
            state.committed_by_session_usd.get(session, Decimal("0")), "f"
        ),
        "combined_committed_quote_total_before_usd": format(state.committed_total_usd, "f"),
        "job55_committed_quote_session_before_usd": format(
            state.job55_committed_by_session_usd.get(session, Decimal("0")), "f"
        ),
        "job55_committed_quote_total_before_usd": format(state.job55_committed_total_usd, "f"),
        "per_session_lifetime_cap_usd": "2.00",
        "total_cap_usd": "32.00",
        "per_session_lifetime_cap_pass": False,
        "total_cap_pass": False,
        "job55_retry_start_allowance_pass": False,
        "time_series_start_permitted": False,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }


def _result_context_from_start(start: Mapping[str, Any]) -> dict[str, Any]:
    payload = start.get("payload", {})
    return {
        "fresh_quote_usd": payload.get("fresh_quote_usd"),
        "combined_commitment_index": payload.get("combined_commitment_index"),
        "job55_commitment_index": payload.get("job55_commitment_index"),
        "timeseries_start_record_hash": start.get("record_hash"),
        "combined_committed_quote_session_usd": payload.get("combined_committed_quote_session_after_usd"),
        "combined_committed_quote_total_usd": payload.get("combined_committed_quote_total_after_usd"),
        "job55_committed_quote_session_usd": payload.get("job55_committed_quote_session_after_usd"),
        "job55_committed_quote_total_usd": payload.get("job55_committed_quote_total_after_usd"),
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }


def summarize_job55_attempts(
    job55_root: Path,
    allowed_requests: Sequence[SessionRequest],
    *,
    opening: Job52OpeningEvidence,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    adoption_receipt_sha256: str,
    require_client_constructed: bool = False,
    expected_sdk_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Rebuild the composite ledger and recovery state from Job-55 disk evidence."""

    root = Path(job55_root)
    attempts_root = root / "attempts"
    base.ensure_nofollow_directory(attempts_root, volume=volume, create=False)
    allowed = {request.session: request for request in allowed_requests}
    if len(allowed) != len(allowed_requests):
        raise Tier0Error("Job-55 allowed request population is duplicated", status="STOP_JOB55_SCOPE_WIDENING")
    ordinal_by_session = {request.session: index for index, request in enumerate(allowed_requests, start=1)}
    if require_client_constructed and base.SHA256_RE.fullmatch(str(expected_sdk_identity_sha256)) is None:
        raise Tier0Error("Job-55 expected SDK identity is absent", status="STOP_JOB55_JOURNAL_INVALID")
    verified_attempts: list[dict[str, Any]] = []
    for attempt_dir in sorted(attempts_root.iterdir()):
        if attempt_dir.is_symlink() or not attempt_dir.is_dir() or base.UUID4_RE.fullmatch(attempt_dir.name) is None:
            raise Tier0Error("Job-55 attempts root has an unexpected child", status="STOP_JOB55_JOURNAL_INVALID")
        if attempt_dir.stat().st_dev != volume.st_dev:
            raise Tier0Error("Job-55 attempt is cross-device", status="STOP_JOB55_JOURNAL_INVALID")
        staging = _validate_job55_attempt_tree(
            attempt_dir, allowed_sessions=set(allowed), volume=volume
        )
        verified = base.verify_attempt_journal(attempt_dir)
        paid._validate_marker_semantics(attempt_dir, verified["records"])
        header = _validate_job55_header(
            verified["records"][0],
            attempt_dir=attempt_dir,
            volume=volume,
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            adoption_receipt_sha256=adoption_receipt_sha256,
        )
        verified_attempts.append(
            {"attempt_dir": attempt_dir, "verified": verified, "records": verified["records"], "header": header, "staging": staging}
        )
    lineage = _validate_job55_attempt_lineage(
        [
            {
                "attempt_id": item["attempt_dir"].name,
                "header_record_hash": item["records"][0]["record_hash"],
                "job55_attempt_ordinal": item["header"]["job55_attempt_ordinal"],
                "previous_job55_attempt_id": item["header"]["previous_job55_attempt_id"],
                "previous_job55_attempt_header_record_hash": item["header"][
                    "previous_job55_attempt_header_record_hash"
                ],
            }
            for item in verified_attempts
        ]
    )
    by_id = {item["attempt_dir"].name: item for item in verified_attempts}
    verified_attempts = [by_id[str(entry["attempt_id"])] for entry in lineage]

    state = AmendedBudgetState.from_opening_evidence(opening)
    cost_starts = cost_results = cost_errors = starts = results = errors = 0
    quote_observations: list[dict[str, Any]] = []
    commitments: list[dict[str, Any]] = []
    start_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    publication_records: list[dict[str, Any]] = []
    partial_staging: list[dict[str, Any]] = []
    recoverable_staging: list[dict[str, Any]] = []
    attempts_summary: list[dict[str, Any]] = []
    successful_results: dict[tuple[str, int], Mapping[str, Any]] = {}
    published_sessions: set[str] = set()
    recovery_barrier: Mapping[str, Any] | None = None
    authority_terminal: dict[str, Any] | None = None

    for item in verified_attempts:
        records = item["records"]
        header = item["header"]
        attempt_dir = item["attempt_dir"]
        if authority_terminal is not None:
            raise Tier0Error("Job-55 attempt exists after terminal authority evidence", status="STOP_JOB55_AUTHORITY_TERMINAL")
        state.current_attempt_cost_calls = 0
        state.current_attempt_time_series_starts = 0
        client_seen = False
        pending_cost: Mapping[str, Any] | None = None
        last_cost_result: Mapping[str, Any] | None = None
        pending_start: Mapping[str, Any] | None = None
        awaiting_publication: Mapping[str, Any] | None = None
        call_error_seen = False
        attempt_cost_results = attempt_starts = attempt_results = 0
        for index, record in enumerate(records):
            event = record.get("event")
            if event not in _JOB55_EVENTS:
                raise Tier0Error("Job-55 journal has an unknown event", status="STOP_JOB55_JOURNAL_INVALID")
            if index == 0:
                continue
            if event == "ATTEMPT_START":
                raise Tier0Error("Job-55 journal has a duplicate header", status="STOP_JOB55_JOURNAL_INVALID")
            if record.get("attempt_id") != attempt_dir.name:
                raise Tier0Error("Job-55 journal attempt identity drifted", status="STOP_JOB55_JOURNAL_INVALID")
            if call_error_seen and event != "ATTEMPT_STOP":
                raise Tier0Error("Job-55 process continued after a call error", status="STOP_JOB55_JOURNAL_INVALID")
            if event == "ATTEMPT_STOP" and index != len(records) - 1:
                raise Tier0Error("Job-55 attempt stop is not terminal", status="STOP_JOB55_JOURNAL_INVALID")
            if recovery_barrier is not None and event != "ATTEMPT_STOP":
                payload = record.get("payload", {})
                exact_recovery = (
                    event == "SESSION_RECOVERED"
                    and record.get("session") == recovery_barrier.get("session")
                    and record.get("request_sha256") == recovery_barrier.get("request_sha256")
                    and payload.get("source_attempt_id") == recovery_barrier.get("attempt_id")
                    and payload.get("source_timeseries_result_sequence") == recovery_barrier.get("sequence")
                    and payload.get("source_timeseries_result_record_hash") == recovery_barrier.get("record_hash")
                )
                if not exact_recovery:
                    raise Tier0Error("Job-55 continued before local recovery", status="STOP_JOB55_RECOVERY_REQUIRED")
            if awaiting_publication is not None and event not in {"SESSION_PUBLISHED", "ATTEMPT_STOP"}:
                raise Tier0Error("Job-55 continued before publication", status="STOP_JOB55_RECOVERY_REQUIRED")
            if event == "CLIENT_CONSTRUCTED":
                payload = record.get("payload")
                if (
                    client_seen
                    or pending_cost is not None
                    or not isinstance(payload, dict)
                    or set(payload) != {"credential_source", "sdk_identity_sha256"}
                    or payload.get("credential_source") not in {"process_environment", "repository_root_dotenv"}
                    or base.SHA256_RE.fullmatch(str(payload.get("sdk_identity_sha256"))) is None
                    or (
                        expected_sdk_identity_sha256 is not None
                        and payload.get("sdk_identity_sha256") != expected_sdk_identity_sha256
                    )
                ):
                    raise Tier0Error("Job-55 client construction payload drifted", status="STOP_JOB55_JOURNAL_INVALID")
                client_seen = True
                continue
            if event in base.CALL_EVENTS:
                if require_client_constructed and not client_seen:
                    raise Tier0Error("Job-55 vendor event precedes client construction", status="STOP_JOB55_JOURNAL_INVALID")
                request = allowed.get(str(record.get("session")))
                if request is None or record.get("request_sha256") != request.market_request_sha256:
                    raise Tier0Error("Job-55 call is out of scope", status="STOP_JOB55_SCOPE_WIDENING")
                if request.session in published_sessions:
                    raise Tier0Error("Job-55 call follows session completion", status="STOP_JOB55_JOURNAL_INVALID")
            if event == "COST_CALL_START":
                if pending_cost is not None or last_cost_result is not None or pending_start is not None or state.current_attempt_cost_calls:
                    raise Tier0Error("Job-55 process has overlapping or repeated cost calls", status="STOP_JOB55_JOURNAL_INVALID")
                request = allowed[str(record["session"])]
                state.assert_cost_call_permitted(request.session)
                if record.get("payload") != {"method": "metadata.get_cost", "parameters": request.market_parameters}:
                    raise Tier0Error("Job-55 cost request widened", status="STOP_JOB55_SCOPE_WIDENING")
                state.note_cost_call(request.session)
                pending_cost = record
                cost_starts += 1
            elif event == "COST_CALL_RESULT":
                if (
                    pending_cost is None
                    or pending_cost.get("session") != record.get("session")
                    or pending_cost.get("request_sha256") != record.get("request_sha256")
                ):
                    raise Tier0Error("Job-55 cost result lacks a start", status="STOP_JOB55_JOURNAL_INVALID")
                pending_cost = None
                last_cost_result = record
                cost_results += 1
                attempt_cost_results += 1
                payload = record.get("payload", {})
                if payload.get("quote_valid") is True:
                    quote_text = payload.get("observed_sdk_quote_usd")
                    try:
                        if not isinstance(quote_text, str) or normalize_amended_quote(Decimal(quote_text)) != quote_text:
                            raise ValueError("noncanonical")
                    except Exception as exc:  # noqa: BLE001
                        raise Tier0Error("Job-55 quote string is malformed", status="STOP_JOB55_JOURNAL_INVALID") from exc
                    if payload != _amended_cost_payload(state, str(record["session"]), quote_text):
                        raise Tier0Error("Job-55 cost arithmetic drifted", status="STOP_JOB55_JOURNAL_INVALID")
                else:
                    expected_invalid = _invalid_amended_cost_payload(state, str(record["session"]), 0)
                    observed_text = payload.get("observed_sdk_quote")
                    expected_invalid["observed_sdk_quote"] = observed_text
                    if not isinstance(observed_text, str) or not observed_text or payload != expected_invalid:
                        raise Tier0Error("Job-55 invalid quote payload drifted", status="STOP_JOB55_JOURNAL_INVALID")
                quote_observations.append(
                    {
                        "attempt_id": attempt_dir.name,
                        "job55_attempt_ordinal": header["job55_attempt_ordinal"],
                        "session": record["session"],
                        "request_sha256": record["request_sha256"],
                        "cost_result_sequence": record["sequence"],
                        "cost_result_record_hash": record["record_hash"],
                        "observed_sdk_quote": payload.get("observed_sdk_quote_usd", payload.get("observed_sdk_quote")),
                        "quote_valid": payload.get("quote_valid"),
                        "time_series_start_permitted": payload.get("time_series_start_permitted"),
                        "committed_to_time_series_start": False,
                    }
                )
                if payload.get("time_series_start_permitted") is not True:
                    authority_terminal = {
                        "status": "STOP_JOB55_QUOTE_AUTHORITY",
                        "attempt_id": attempt_dir.name,
                        "session": record["session"],
                        "record_hash": record["record_hash"],
                    }
            elif event == "COST_CALL_ERROR":
                payload = record.get("payload")
                if (
                    pending_cost is None
                    or pending_cost.get("session") != record.get("session")
                    or pending_cost.get("request_sha256") != record.get("request_sha256")
                    or not isinstance(payload, dict)
                    or set(payload) != {"error_class", "actual_vendor_invoice_cost_usd"}
                    or not isinstance(payload.get("error_class"), str)
                    or not payload["error_class"]
                    or payload.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
                ):
                    raise Tier0Error("Job-55 cost error payload drifted", status="STOP_JOB55_JOURNAL_INVALID")
                pending_cost = None
                cost_errors += 1
                call_error_seen = True
                if record.get("session") == FAILED_SESSION:
                    authority_terminal = {
                        "status": "STOP_JOB55_SINGULAR_RETRY_COST_ERROR",
                        "attempt_id": attempt_dir.name,
                        "session": FAILED_SESSION,
                        "record_hash": record["record_hash"],
                    }
            elif event == "TIMESERIES_CALL_START":
                if (
                    last_cost_result is None
                    or last_cost_result.get("sequence") != record.get("sequence", -2) - 1
                    or last_cost_result.get("session") != record.get("session")
                    or last_cost_result.get("request_sha256") != record.get("request_sha256")
                    or pending_start is not None
                    or state.current_attempt_time_series_starts
                ):
                    raise Tier0Error("Job-55 start lacks an adjacent quote", status="STOP_JOB55_JOURNAL_INVALID")
                request = allowed[str(record["session"])]
                quote_text = last_cost_result.get("payload", {}).get("observed_sdk_quote_usd")
                projection = state.projection(request.session, Decimal(str(quote_text)))
                expected_start = _amended_start_payload(
                    request=request,
                    output_path=Path("data.cmbp-1.dbn.zst"),
                    quote_text=str(quote_text),
                    cost_result_record_hash=str(last_cost_result["record_hash"]),
                    state=state,
                    projection=projection,
                    readiness_receipt_sha256=readiness_receipt_sha256,
                    readiness_receipt_file_sha256=readiness_receipt_file_sha256,
                    adoption_receipt_sha256=adoption_receipt_sha256,
                    ordinal=ordinal_by_session[request.session],
                )
                if record.get("payload") != expected_start or projection["time_series_start_permitted"] is not True:
                    raise Tier0Error("Job-55 start commitment payload drifted", status="STOP_JOB55_JOURNAL_INVALID")
                state.commit(request.session, Decimal(str(quote_text)), projection)
                quote_observations[-1]["committed_to_time_series_start"] = True
                commitments.append(
                    {
                        "combined_commitment_index": state.commitment_count,
                        "job55_commitment_index": state.job55_commitment_count,
                        "attempt_id": attempt_dir.name,
                        "job55_attempt_ordinal": header["job55_attempt_ordinal"],
                        "session": request.session,
                        "request_sha256": request.market_request_sha256,
                        "fresh_quote_usd": str(quote_text),
                        "timeseries_start_sequence": record["sequence"],
                        "timeseries_start_record_hash": record["record_hash"],
                        "combined_committed_quote_session_usd": format(
                            state.committed_by_session_usd[request.session], "f"
                        ),
                        "combined_committed_quote_total_usd": format(state.committed_total_usd, "f"),
                    }
                )
                last_cost_result = None
                pending_start = record
                start_records.append(dict(record))
                starts += 1
                attempt_starts += 1
            elif event in {"TIMESERIES_CALL_RESULT", "TIMESERIES_CALL_ERROR"}:
                if (
                    pending_start is None
                    or pending_start.get("session") != record.get("session")
                    or pending_start.get("request_sha256") != record.get("request_sha256")
                ):
                    raise Tier0Error("Job-55 time-series terminal lacks a start", status="STOP_JOB55_JOURNAL_INVALID")
                context = _result_context_from_start(pending_start)
                payload = record.get("payload")
                if event == "TIMESERIES_CALL_RESULT":
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != set(context) | {"compressed_bytes", "dbn_file_sha256"}
                        or {key: payload.get(key) for key in context} != context
                        or isinstance(payload.get("compressed_bytes"), bool)
                        or not isinstance(payload.get("compressed_bytes"), int)
                        or payload["compressed_bytes"] <= 0
                        or base.SHA256_RE.fullmatch(str(payload.get("dbn_file_sha256"))) is None
                    ):
                        raise Tier0Error("Job-55 result payload drifted", status="STOP_JOB55_JOURNAL_INVALID")
                    results += 1
                    attempt_results += 1
                    awaiting_publication = record
                    successful_results[(attempt_dir.name, int(record["sequence"]))] = record
                    result_records.append(dict(record))
                else:
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != set(context) | {"error_class"}
                        or {key: payload.get(key) for key in context} != context
                        or not isinstance(payload.get("error_class"), str)
                        or not payload["error_class"]
                    ):
                        raise Tier0Error("Job-55 stream error payload drifted", status="STOP_JOB55_JOURNAL_INVALID")
                    errors += 1
                    call_error_seen = True
                    if record.get("session") == FAILED_SESSION:
                        authority_terminal = {
                            "status": "STOP_JOB55_SINGULAR_RETRY_STREAM_ERROR",
                            "attempt_id": attempt_dir.name,
                            "session": FAILED_SESSION,
                            "record_hash": record["record_hash"],
                        }
                pending_start = None
            elif event in {"SESSION_PUBLISHED", "SESSION_RECOVERED"}:
                request = allowed.get(str(record.get("session")))
                payload = record.get("payload")
                expected_fields = {
                    "ordinal",
                    "publication_mode",
                    "session_qc_sha256",
                    "decoded_records",
                    "compressed_bytes",
                    "dbn_file_sha256",
                    "source_attempt_id",
                    "source_timeseries_result_sequence",
                    "source_timeseries_result_record_hash",
                }
                if request is None or record.get("request_sha256") != request.market_request_sha256:
                    raise Tier0Error("Job-55 publication is out of scope", status="STOP_JOB55_SCOPE_WIDENING")
                source_key = (
                    str(payload.get("source_attempt_id")) if isinstance(payload, dict) else "",
                    payload.get("source_timeseries_result_sequence") if isinstance(payload, dict) else -1,
                )
                source = successful_results.get(source_key) if isinstance(source_key[1], int) else None
                if (
                    not isinstance(payload, dict)
                    or set(payload) != expected_fields
                    or payload.get("ordinal") != ordinal_by_session[request.session]
                    or payload.get("publication_mode") != event
                    or base.SHA256_RE.fullmatch(str(payload.get("session_qc_sha256"))) is None
                    or payload.get("decoded_records") != request.expected_record_count
                    or isinstance(payload.get("compressed_bytes"), bool)
                    or not isinstance(payload.get("compressed_bytes"), int)
                    or payload["compressed_bytes"] <= 0
                    or base.SHA256_RE.fullmatch(str(payload.get("dbn_file_sha256"))) is None
                    or source is None
                    or source.get("record_hash") != payload.get("source_timeseries_result_record_hash")
                    or source.get("session") != request.session
                    or source.get("request_sha256") != request.market_request_sha256
                    or source.get("payload", {}).get("compressed_bytes") != payload.get("compressed_bytes")
                    or source.get("payload", {}).get("dbn_file_sha256") != payload.get("dbn_file_sha256")
                    or request.session in published_sessions
                    or (event == "SESSION_PUBLISHED" and payload.get("source_attempt_id") != attempt_dir.name)
                    or (event == "SESSION_RECOVERED" and payload.get("source_attempt_id") == attempt_dir.name)
                ):
                    raise Tier0Error("Job-55 publication/source binding drifted", status="STOP_JOB55_JOURNAL_INVALID")
                if event == "SESSION_PUBLISHED":
                    if awaiting_publication is None or awaiting_publication.get("record_hash") != source.get("record_hash"):
                        raise Tier0Error("Job-55 publication is not adjacent to its result", status="STOP_JOB55_JOURNAL_INVALID")
                    awaiting_publication = None
                elif recovery_barrier is None or recovery_barrier.get("record_hash") != source.get("record_hash"):
                    raise Tier0Error("Job-55 recovery lacks its global barrier", status="STOP_JOB55_JOURNAL_INVALID")
                if event == "SESSION_RECOVERED":
                    recovery_barrier = None
                publication_records.append(dict(record))
                published_sessions.add(request.session)
                state.published_sessions.add(request.session)
            elif event == "ATTEMPT_STOP":
                payload = record.get("payload")
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"status", "error_class"}
                    or not isinstance(payload.get("status"), str)
                    or not payload["status"]
                    or not isinstance(payload.get("error_class"), str)
                    or not payload["error_class"]
                ):
                    raise Tier0Error("Job-55 stop payload drifted", status="STOP_JOB55_JOURNAL_INVALID")
            else:
                raise Tier0Error("Job-55 journal state is invalid", status="STOP_JOB55_JOURNAL_INVALID")
        if pending_cost is not None:
            if pending_cost.get("session") == FAILED_SESSION:
                authority_terminal = {
                    "status": "STOP_JOB55_SINGULAR_RETRY_COST_INDETERMINATE",
                    "attempt_id": attempt_dir.name,
                    "session": FAILED_SESSION,
                    "record_hash": pending_cost["record_hash"],
                }
        if last_cost_result is not None and last_cost_result.get("session") == FAILED_SESSION:
            authority_terminal = {
                "status": "STOP_JOB55_SINGULAR_RETRY_QUOTE_WITHOUT_START",
                "attempt_id": attempt_dir.name,
                "session": FAILED_SESSION,
                "record_hash": last_cost_result["record_hash"],
            }
        if pending_start is not None and pending_start.get("session") == FAILED_SESSION:
            authority_terminal = {
                "status": "STOP_JOB55_SINGULAR_RETRY_START_INDETERMINATE",
                "attempt_id": attempt_dir.name,
                "session": FAILED_SESSION,
                "record_hash": pending_start["record_hash"],
            }
        if awaiting_publication is not None:
            if recovery_barrier is not None:
                raise Tier0Error("multiple Job-55 successful streams await recovery", status="STOP_JOB55_RECOVERY_REQUIRED")
            recovery_barrier = awaiting_publication
        stop_sha = _validate_job55_stop(
            attempt_dir,
            records,
            readiness_receipt_sha256=readiness_receipt_sha256,
            adoption_receipt_sha256=adoption_receipt_sha256,
        )
        attempts_summary.append(
            {
                "attempt_id": attempt_dir.name,
                "job55_attempt_ordinal": header["job55_attempt_ordinal"],
                "terminal_event": records[-1]["event"],
                "journal_file_sha256": item["verified"]["journal_file_sha256"],
                "watermark_file_sha256": item["verified"]["watermark_file_sha256"],
                "marker_files": item["verified"]["marker_files"],
                "attempt_stop_file_sha256": stop_sha,
                "cost_results": attempt_cost_results,
                "timeseries_starts": attempt_starts,
                "timeseries_results": attempt_results,
            }
        )

    # Classify every staged file only after the journal grammar and ledger pass.
    result_by_attempt_session = {
        (record["attempt_id"], record["session"]): record for record in result_records
    }
    start_by_attempt_session: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in verified_attempts:
        for record in item["records"]:
            if record.get("event") == "TIMESERIES_CALL_START":
                key = (item["attempt_dir"].name, str(record["session"]))
                if key in start_by_attempt_session:
                    raise Tier0Error("Job-55 attempt repeated a staged session", status="STOP_JOB55_JOURNAL_INVALID")
                start_by_attempt_session[key] = record
        for session, names in item["staging"].items():
            key = (item["attempt_dir"].name, session)
            start = start_by_attempt_session.get(key)
            data_path = item["attempt_dir"] / "sessions" / f"{session}.bundle.part" / "data.cmbp-1.dbn.zst"
            qc_path = data_path.parent / JOB55_SESSION_QC_NAME
            if names and start is None:
                raise Tier0Error("Job-55 staging has no start record", status="STOP_JOB55_JOURNAL_INVALID")
            if JOB55_SESSION_QC_NAME in names and "data.cmbp-1.dbn.zst" not in names:
                raise Tier0Error("Job-55 staging QC lacks raw data", status="STOP_JOB55_JOURNAL_INVALID")
            if "data.cmbp-1.dbn.zst" not in names:
                continue
            metadata = data_path.lstat()
            staged = {
                "attempt_id": item["attempt_dir"].name,
                "session": session,
                "request_sha256": start["request_sha256"] if start else None,
                "data_path": str(data_path),
                "compressed_bytes": metadata.st_size,
                "file_sha256": base.file_sha256(data_path),
                "timeseries_start_record_hash": start["record_hash"] if start else None,
            }
            result = result_by_attempt_session.get(key)
            if result is not None:
                if (
                    staged["compressed_bytes"] != result["payload"]["compressed_bytes"]
                    or staged["file_sha256"] != result["payload"]["dbn_file_sha256"]
                ):
                    raise Tier0Error("Job-55 result-bound staging drifted", status="STOP_JOB55_RECOVERY_REQUIRED")
                staged["timeseries_result_sequence"] = result["sequence"]
                staged["timeseries_result_record_hash"] = result["record_hash"]
                recoverable_staging.append(staged)
            else:
                if JOB55_SESSION_QC_NAME in names:
                    raise Tier0Error("Job-55 partial staging has unbound QC", status="STOP_JOB55_JOURNAL_INVALID")
                partial_staging.append(staged)

    combined_by_session = {
        session: format(value, "f") for session, value in sorted(state.committed_by_session_usd.items())
    }
    job55_by_session = {
        session: format(value, "f") for session, value in sorted(state.job55_committed_by_session_usd.items())
    }
    return {
        "job52_opening_commitment_count": 1,
        "job52_opening_commitment_total_usd": "0.950392448902",
        "job52_opening_commitment_by_session_usd": {FAILED_SESSION: "0.950392448902"},
        "job55_attempt_count": len(verified_attempts),
        "job55_cost_call_starts": cost_starts,
        "job55_cost_call_results": cost_results,
        "job55_cost_call_errors": cost_errors,
        "job55_timeseries_call_starts": starts,
        "job55_timeseries_call_results": results,
        "job55_timeseries_call_errors": errors,
        "job55_commitment_count": state.job55_commitment_count,
        "combined_commitment_count": state.commitment_count,
        "job55_committed_quote_total_usd": format(state.job55_committed_total_usd, "f"),
        "job55_committed_quote_by_session_usd": job55_by_session,
        "combined_committed_quote_total_usd": format(state.committed_total_usd, "f"),
        "combined_committed_quote_by_session_usd": combined_by_session,
        "job55_cost_call_count_by_session": dict(sorted(state.job55_cost_call_count_by_session.items())),
        "job55_start_count_by_session": dict(sorted(state.job55_start_count_by_session.items())),
        "commitments": commitments,
        "start_records": start_records,
        "quote_observations": quote_observations,
        "result_records": result_records,
        "publication_records": publication_records,
        "published_sessions_in_journals": sorted(published_sessions),
        "successful_stream_recovery_required": recovery_barrier is not None,
        "successful_stream_recovery_record": dict(recovery_barrier) if recovery_barrier is not None else None,
        "recoverable_staging": recoverable_staging,
        "partial_or_failed_staging": partial_staging,
        "terminal_authority_failure_observed": authority_terminal is not None,
        "terminal_authority_failure": authority_terminal,
        "per_session_lifetime_cap_usd": "2.00",
        "total_committed_quote_cap_usd": "32.00",
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "attempts": attempts_summary,
    }


def amended_budget_state_from_summary(summary: Mapping[str, Any]) -> AmendedBudgetState:
    try:
        total = Decimal(str(summary["job55_committed_quote_total_usd"]))
        by_session = {
            str(session): Decimal(str(value))
            for session, value in summary["job55_committed_quote_by_session_usd"].items()
        }
        starts = {str(session): int(value) for session, value in summary["job55_start_count_by_session"].items()}
        costs = {str(session): int(value) for session, value in summary["job55_cost_call_count_by_session"].items()}
        count = summary["job55_commitment_count"]
        published = set(summary["published_sessions_in_journals"])
    except Exception as exc:  # noqa: BLE001
        raise Tier0Error("Job-55 summary budget fields are malformed", status="STOP_JOB55_BUDGET_INVALID") from exc
    return AmendedBudgetState(
        job55_committed_total_usd=total,
        job55_committed_by_session_usd=by_session,
        job55_start_count_by_session=starts,
        job55_commitment_count=count,
        job55_cost_call_count_by_session=costs,
        published_sessions=published,
    )


def _request_projection(request: SessionRequest) -> dict[str, Any]:
    return paid._request_projection(request)


def build_job55_session_qc(
    *,
    request: SessionRequest,
    data_path: Path,
    metadata_summary: Mapping[str, Any],
    decoder_summary: Mapping[str, Any],
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    adoption: Mapping[str, Any],
    start_record: Mapping[str, Any],
    result_record: Mapping[str, Any],
) -> dict[str, Any]:
    start_payload = start_record.get("payload", {})
    result_payload = result_record.get("payload", {})
    if (
        start_record.get("event") != "TIMESERIES_CALL_START"
        or result_record.get("event") != "TIMESERIES_CALL_RESULT"
        or start_record.get("attempt_id") != result_record.get("attempt_id")
        or start_record.get("session") != request.session
        or result_record.get("session") != request.session
        or start_record.get("request_sha256") != request.market_request_sha256
        or result_record.get("request_sha256") != request.market_request_sha256
        or result_payload.get("timeseries_start_record_hash") != start_record.get("record_hash")
        or start_payload.get("program_contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or start_payload.get("readiness_receipt_sha256") != readiness.get("receipt_sha256")
        or start_payload.get("adoption_receipt_sha256") != adoption.get("adoption_sha256")
    ):
        raise Tier0Error("Job-55 session QC source pair is invalid", status="STOP_JOB55_SESSION_QC")
    receipt: dict[str, Any] = {
        "artifact_type": SESSION_QC_ARTIFACT,
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-session-qc.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "predecessor_job_id": PREDECESSOR_JOB_ID,
        "status": "JOB55_CMBP_TIER0_CAP_AMENDMENT_SESSION_QC_PASS",
        "session": request.session,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "readiness_receipt_file_sha256": readiness_file_sha256,
        "adoption_receipt_sha256": adoption["adoption_sha256"],
        "request": _request_projection(request),
        "cost_commitment": {
            "fresh_observed_sdk_quote_usd": start_payload["fresh_quote_usd"],
            "job52_carried_commitment_usd": "0.950392448902",
            "combined_commitment_index": start_payload["combined_commitment_index"],
            "job55_commitment_index": start_payload["job55_commitment_index"],
            "combined_committed_quote_session_before_usd": start_payload[
                "combined_committed_quote_session_before_usd"
            ],
            "combined_committed_quote_session_after_usd": start_payload[
                "combined_committed_quote_session_after_usd"
            ],
            "combined_committed_quote_total_before_usd": start_payload[
                "combined_committed_quote_total_before_usd"
            ],
            "combined_committed_quote_total_after_usd": start_payload[
                "combined_committed_quote_total_after_usd"
            ],
            "job55_committed_quote_session_before_usd": start_payload[
                "job55_committed_quote_session_before_usd"
            ],
            "job55_committed_quote_session_after_usd": start_payload[
                "job55_committed_quote_session_after_usd"
            ],
            "job55_committed_quote_total_before_usd": start_payload[
                "job55_committed_quote_total_before_usd"
            ],
            "job55_committed_quote_total_after_usd": start_payload[
                "job55_committed_quote_total_after_usd"
            ],
            "job55_session_start_count_before": start_payload["job55_session_start_count_before"],
            "per_session_lifetime_cap_usd": "2.00",
            "total_cap_usd": "32.00",
            "acquisition_initiated": True,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "quote_is_atomic_invoice_lock": False,
        },
        "raw_dbn": {
            "file_name": data_path.name,
            "file_sha256": base.file_sha256(data_path),
            "compressed_bytes": data_path.stat().st_size,
        },
        "dbn_metadata": dict(metadata_summary),
        "decoder": dict(decoder_summary),
        "source_attempt": {
            "attempt_id": start_record["attempt_id"],
            "timeseries_start_sequence": start_record["sequence"],
            "timeseries_start_record_hash": start_record["record_hash"],
            "timeseries_result_sequence": result_record["sequence"],
            "timeseries_result_record_hash": result_record["record_hash"],
        },
        "qc_law": {
            "streaming_no_full_session_pandas": True,
            "causal_clock": "ts_recv",
            "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
            "receive_time_ties_excluded": True,
            "historical_connection_telemetry": "UNKNOWN_NOT_PRESENT_IN_FILE_TRANSPORT",
        },
    }
    receipt["session_qc_sha256"] = base.self_hash(receipt, "session_qc_sha256")
    return receipt


def validate_job55_session_bundle(
    bundle_dir: Path,
    *,
    request: SessionRequest,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    adoption: Mapping[str, Any],
    volume: VolumeIdentity,
    source_record_lookup: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    base.ensure_nofollow_directory(bundle_dir, volume=volume)
    if sorted(path.name for path in bundle_dir.iterdir()) != [JOB55_SESSION_QC_NAME, "data.cmbp-1.dbn.zst"]:
        raise Tier0Error(f"{request.session}: Job-55 bundle population drifted", status="STOP_JOB55_SESSION_QC")
    data_path = bundle_dir / "data.cmbp-1.dbn.zst"
    qc_path = bundle_dir / JOB55_SESSION_QC_NAME
    for path in (data_path, qc_path):
        metadata = _private_regular(path, status="STOP_JOB55_SESSION_QC")
        if metadata.st_dev != volume.st_dev:
            raise Tier0Error(f"{request.session}: Job-55 bundle is cross-device", status="STOP_JOB55_SESSION_QC")
    receipt = base.strict_json(qc_path)
    expected_fields = {
        "artifact_type",
        "schema_version",
        "job_id",
        "target_job_id",
        "predecessor_job_id",
        "status",
        "session",
        "scope_sha256",
        "scope_file_sha256",
        "program_contract_sha256",
        "program_contract_file_sha256",
        "readiness_receipt_sha256",
        "readiness_receipt_file_sha256",
        "adoption_receipt_sha256",
        "request",
        "cost_commitment",
        "raw_dbn",
        "dbn_metadata",
        "decoder",
        "source_attempt",
        "qc_law",
        "session_qc_sha256",
    }
    expected_scalars = {
        "artifact_type": SESSION_QC_ARTIFACT,
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-session-qc.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "predecessor_job_id": PREDECESSOR_JOB_ID,
        "status": "JOB55_CMBP_TIER0_CAP_AMENDMENT_SESSION_QC_PASS",
        "session": request.session,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness.get("receipt_sha256"),
        "readiness_receipt_file_sha256": readiness_file_sha256,
        "adoption_receipt_sha256": adoption.get("adoption_sha256"),
    }
    if set(receipt) != expected_fields or any(receipt.get(key) != value for key, value in expected_scalars.items()):
        raise Tier0Error(f"{request.session}: Job-55 QC identity drifted", status="STOP_JOB55_SESSION_QC")
    if receipt.get("session_qc_sha256") != base.self_hash(receipt, "session_qc_sha256"):
        raise Tier0Error(f"{request.session}: Job-55 QC self-hash mismatch", status="STOP_JOB55_SESSION_QC")
    if receipt.get("request") != _request_projection(request):
        raise Tier0Error(f"{request.session}: Job-55 request projection drifted", status="STOP_JOB55_SESSION_QC")
    expected_raw = {
        "file_name": data_path.name,
        "file_sha256": base.file_sha256(data_path),
        "compressed_bytes": data_path.stat().st_size,
    }
    if receipt.get("raw_dbn") != expected_raw:
        raise Tier0Error(f"{request.session}: Job-55 raw DBN binding drifted", status="STOP_JOB55_SESSION_QC")
    cost = receipt.get("cost_commitment")
    cost_fields = {
        "fresh_observed_sdk_quote_usd",
        "job52_carried_commitment_usd",
        "combined_commitment_index",
        "job55_commitment_index",
        "combined_committed_quote_session_before_usd",
        "combined_committed_quote_session_after_usd",
        "combined_committed_quote_total_before_usd",
        "combined_committed_quote_total_after_usd",
        "job55_committed_quote_session_before_usd",
        "job55_committed_quote_session_after_usd",
        "job55_committed_quote_total_before_usd",
        "job55_committed_quote_total_after_usd",
        "job55_session_start_count_before",
        "per_session_lifetime_cap_usd",
        "total_cap_usd",
        "acquisition_initiated",
        "actual_vendor_invoice_cost_usd",
        "quote_is_atomic_invoice_lock",
    }
    if not isinstance(cost, dict) or set(cost) != cost_fields:
        raise Tier0Error(f"{request.session}: Job-55 cost fields drifted", status="STOP_JOB55_SESSION_QC")
    try:
        quote = Decimal(str(cost["fresh_observed_sdk_quote_usd"]))
        combined_session_before = Decimal(str(cost["combined_committed_quote_session_before_usd"]))
        combined_session_after = Decimal(str(cost["combined_committed_quote_session_after_usd"]))
        combined_total_before = Decimal(str(cost["combined_committed_quote_total_before_usd"]))
        combined_total_after = Decimal(str(cost["combined_committed_quote_total_after_usd"]))
        job55_session_before = Decimal(str(cost["job55_committed_quote_session_before_usd"]))
        job55_session_after = Decimal(str(cost["job55_committed_quote_session_after_usd"]))
        job55_total_before = Decimal(str(cost["job55_committed_quote_total_before_usd"]))
        job55_total_after = Decimal(str(cost["job55_committed_quote_total_after_usd"]))
    except Exception as exc:  # noqa: BLE001
        raise Tier0Error(f"{request.session}: Job-55 cost decimal is invalid", status="STOP_JOB55_SESSION_QC") from exc
    if (
        normalize_amended_quote(quote) != cost["fresh_observed_sdk_quote_usd"]
        or _exact_add(combined_session_before, quote) != combined_session_after
        or _exact_add(combined_total_before, quote) != combined_total_after
        or _exact_add(job55_session_before, quote) != job55_session_after
        or _exact_add(job55_total_before, quote) != job55_total_after
        or combined_session_after > PER_SESSION_LIFETIME_CAP_USD
        or combined_total_after > TOTAL_COMMITTED_CAP_USD
        or cost.get("job52_carried_commitment_usd") != "0.950392448902"
        or isinstance(cost.get("combined_commitment_index"), bool)
        or not isinstance(cost.get("combined_commitment_index"), int)
        or cost["combined_commitment_index"] < 2
        or isinstance(cost.get("job55_commitment_index"), bool)
        or not isinstance(cost.get("job55_commitment_index"), int)
        or cost["job55_commitment_index"] < 1
        or cost["combined_commitment_index"] != cost["job55_commitment_index"] + 1
        or isinstance(cost.get("job55_session_start_count_before"), bool)
        or not isinstance(cost.get("job55_session_start_count_before"), int)
        or cost["job55_session_start_count_before"] < 0
        or (request.session == FAILED_SESSION and cost["job55_session_start_count_before"] != 0)
        or cost.get("per_session_lifetime_cap_usd") != "2.00"
        or cost.get("total_cap_usd") != "32.00"
        or cost.get("acquisition_initiated") is not True
        or cost.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
        or cost.get("quote_is_atomic_invoice_lock") is not False
    ):
        raise Tier0Error(f"{request.session}: Job-55 cost law drifted", status="STOP_JOB55_SESSION_QC")
    expected_qc_law = {
        "streaming_no_full_session_pandas": True,
        "causal_clock": "ts_recv",
        "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
        "receive_time_ties_excluded": True,
        "historical_connection_telemetry": "UNKNOWN_NOT_PRESENT_IN_FILE_TRANSPORT",
    }
    if receipt.get("qc_law") != expected_qc_law:
        raise Tier0Error(f"{request.session}: Job-55 QC law drifted", status="STOP_JOB55_SESSION_QC")
    decoder = receipt.get("decoder")
    if not isinstance(decoder, dict):
        raise Tier0Error(f"{request.session}: Job-55 decoder is absent", status="STOP_JOB55_SESSION_QC")
    paid._validate_decoder_summary(decoder, request)
    source = receipt.get("source_attempt")
    source_fields = {
        "attempt_id",
        "timeseries_start_sequence",
        "timeseries_start_record_hash",
        "timeseries_result_sequence",
        "timeseries_result_record_hash",
    }
    if (
        not isinstance(source, dict)
        or set(source) != source_fields
        or base.UUID4_RE.fullmatch(str(source.get("attempt_id"))) is None
        or any(
            isinstance(source.get(field_name), bool)
            or not isinstance(source.get(field_name), int)
            or source[field_name] < 0
            for field_name in ("timeseries_start_sequence", "timeseries_result_sequence")
        )
        or base.SHA256_RE.fullmatch(str(source.get("timeseries_start_record_hash"))) is None
        or base.SHA256_RE.fullmatch(str(source.get("timeseries_result_record_hash"))) is None
    ):
        raise Tier0Error(f"{request.session}: Job-55 source locator is invalid", status="STOP_JOB55_SESSION_QC")
    if source_record_lookup is not None:
        start = source_record_lookup.get((source["attempt_id"], source["timeseries_start_sequence"]))
        result = source_record_lookup.get((source["attempt_id"], source["timeseries_result_sequence"]))
        if (
            start is None
            or result is None
            or start.get("record_hash") != source["timeseries_start_record_hash"]
            or result.get("record_hash") != source["timeseries_result_record_hash"]
            or start.get("event") != "TIMESERIES_CALL_START"
            or result.get("event") != "TIMESERIES_CALL_RESULT"
            or start.get("session") != request.session
            or result.get("session") != request.session
            or start.get("request_sha256") != request.market_request_sha256
            or result.get("request_sha256") != request.market_request_sha256
            or result.get("payload", {}).get("timeseries_start_record_hash") != start.get("record_hash")
            or result.get("payload", {}).get("dbn_file_sha256") != expected_raw["file_sha256"]
            or result.get("payload", {}).get("compressed_bytes") != expected_raw["compressed_bytes"]
        ):
            raise Tier0Error(f"{request.session}: Job-55 source journal drifted", status="STOP_JOB55_SESSION_QC")
        rebuilt = build_job55_session_qc(
            request=request,
            data_path=data_path,
            metadata_summary=receipt["dbn_metadata"],
            decoder_summary=decoder,
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha256,
            adoption=adoption,
            start_record=start,
            result_record=result,
        )
        if receipt != rebuilt:
            raise Tier0Error(f"{request.session}: Job-55 QC does not reconstruct", status="STOP_JOB55_SESSION_QC")
    import databento

    store = databento.DBNStore.from_file(data_path)
    if base.validate_dbn_metadata(store, request) != receipt.get("dbn_metadata"):
        raise Tier0Error(f"{request.session}: Job-55 DBN metadata drifted", status="STOP_JOB55_SESSION_QC")
    return receipt


def validate_job55_existing_session_population(
    job55_root: Path,
    *,
    bundle: CapAmendmentScopeBundle,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    adoption: Mapping[str, Any],
    volume: VolumeIdentity,
    source_record_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    published_sessions_in_journals: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    sessions_root = Path(job55_root) / "sessions"
    base.ensure_nofollow_directory(sessions_root, volume=volume)
    expected = {request.session: request for request in bundle.sessions}
    result: dict[str, dict[str, Any]] = {}
    for child in sorted(sessions_root.iterdir()):
        if child.is_symlink() or not child.is_dir() or child.stat().st_dev != volume.st_dev or child.name not in expected:
            raise Tier0Error("Job-55 final session population is unexpected", status="STOP_JOB55_SESSION_QC")
        result[child.name] = validate_job55_session_bundle(
            child,
            request=expected[child.name],
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha256,
            adoption=adoption,
            volume=volume,
            source_record_lookup=source_record_lookup,
        )
    if set(published_sessions_in_journals) - set(result):
        raise Tier0Error("a durably published Job-55 bundle disappeared", status="STOP_JOB55_SESSION_QC")
    return result


def build_job55_publication_payload(
    *,
    request: SessionRequest,
    qc: Mapping[str, Any],
    mode: str,
    ordinal: int,
) -> dict[str, Any]:
    if mode not in {"SESSION_PUBLISHED", "SESSION_RECOVERED"}:
        raise Tier0Error("Job-55 publication mode is invalid", status="STOP_JOB55_SESSION_QC")
    source = qc["source_attempt"]
    return {
        "ordinal": ordinal,
        "publication_mode": mode,
        "session_qc_sha256": qc["session_qc_sha256"],
        "decoded_records": request.expected_record_count,
        "compressed_bytes": qc["raw_dbn"]["compressed_bytes"],
        "dbn_file_sha256": qc["raw_dbn"]["file_sha256"],
        "source_attempt_id": source["attempt_id"],
        "source_timeseries_result_sequence": source["timeseries_result_sequence"],
        "source_timeseries_result_record_hash": source["timeseries_result_record_hash"],
    }


def job55_source_record_lookup(summary: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in [*summary.get("start_records", []), *summary.get("result_records", [])]:
        key = (str(record.get("attempt_id")), record.get("sequence"))
        if (
            base.UUID4_RE.fullmatch(key[0]) is None
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or key[1] < 0
            or key in lookup
        ):
            raise Tier0Error("Job-55 source-record population is invalid", status="STOP_JOB55_JOURNAL_INVALID")
        lookup[key] = record
    return lookup


def validate_job55_job_tree(job55_root: Path, *, volume: VolumeIdentity) -> None:
    root = Path(job55_root)
    if root != Path(volume.mount_point) / JOB55_ROOT_RELATIVE:
        raise Tier0Error("Job-55 root is not canonical", status="STOP_JOB55_EXTERNAL_PATH")
    expected_dirs = {"attempts", "sessions", "receipts"}
    required_files = {JOB55_ADOPTION_NAME, JOB55_LOCK_BINDING_NAME, JOB55_ATTEMPT_ANCHOR_NAME}
    optional_files = {JOB55_AGGREGATE_SEAL_NAME}
    children = {item.name: item for item in root.iterdir()}
    if set(children) - expected_dirs - required_files - optional_files or not expected_dirs.issubset(children) or not required_files.issubset(children):
        raise Tier0Error("Job-55 root population drifted", status="STOP_JOB55_EXTERNAL_PATH")
    parent = Path(volume.mount_point) / "cmbp-tier0"
    for path in (parent, root):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or metadata.st_dev != volume.st_dev:
            raise Tier0Error("Job-55 root ancestry is unsafe", status="STOP_JOB55_EXTERNAL_PATH")
    for name, path in children.items():
        metadata = path.lstat()
        if metadata.st_dev != volume.st_dev or stat.S_ISLNK(metadata.st_mode):
            raise Tier0Error("Job-55 root child is unsafe", status="STOP_JOB55_EXTERNAL_PATH")
        if name in expected_dirs:
            if not stat.S_ISDIR(metadata.st_mode):
                raise Tier0Error("Job-55 required directory drifted", status="STOP_JOB55_EXTERNAL_PATH")
        elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Tier0Error("Job-55 control file is unsafe", status="STOP_JOB55_EXTERNAL_PATH")
    receipt_children = list((root / "receipts").iterdir())
    if any(
        item.name != JOB55_AGGREGATE_NAME
        or item.is_symlink()
        or not item.is_file()
        or item.stat().st_nlink != 1
        or item.stat().st_dev != volume.st_dev
        for item in receipt_children
    ):
        raise Tier0Error("Job-55 receipts population drifted", status="STOP_JOB55_EXTERNAL_PATH")


_AGGREGATE_DECODER_SUM_KEYS = (
    "mapping_records",
    "total_records_accepted",
    "cmbp1_records",
    "trade_records",
    "strict_prior_trades",
    "tied_prior_trades_excluded",
    "no_prior_trades",
    "signed_trades",
    "at_bid_trades",
    "at_ask_trades",
    "inside_trades",
    "outside_trades",
    "ambiguous_trades",
    "undefined_trade_price_excluded",
    "missing_prior_book_excluded",
    "undefined_prior_book_excluded",
    "locked_prior_book_excluded",
    "crossed_prior_book_excluded",
    "trade_bad_ts_recv_excluded",
    "prior_bad_ts_recv_excluded",
    "prior_maybe_bad_book_excluded",
    "global_receive_ties",
    "global_receive_regressions",
    "global_event_regressions",
    "instrument_event_regressions",
    "disconnect_count",
    "reconnect_count",
    "gap_count",
    "book_state_clear_count",
    "gaps_with_known_bounds",
    "total_known_gap_ns",
    "system_records",
    "heartbeat_records",
    "rows_with_flags",
    "unknown_flag_rows",
)


def _final_session_manifest(final_qc: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for session, qc in sorted(final_qc.items()):
        decoder = qc["decoder"]
        result.append(
            {
                "session": session,
                "session_qc_sha256": qc["session_qc_sha256"],
                "raw_dbn_file_sha256": qc["raw_dbn"]["file_sha256"],
                "compressed_bytes": qc["raw_dbn"]["compressed_bytes"],
                "decoded_records": decoder["cmbp1_records"],
                "mapping_count": len(qc["request"]["expected_mappings"]),
                "source_attempt": dict(qc["source_attempt"]),
            }
        )
    return result


def _validate_final_publication_binding(
    summary: Mapping[str, Any],
    final_qc: Mapping[str, Mapping[str, Any]],
) -> None:
    publications = summary.get("publication_records")
    if not isinstance(publications, list) or len(publications) != base.EXPECTED_SESSION_COUNT:
        raise Tier0Error("Job-55 publication population is incomplete", status="STOP_JOB55_AGGREGATE_QC")
    by_session: dict[str, Mapping[str, Any]] = {}
    for record in publications:
        session = str(record.get("session"))
        if session in by_session:
            raise Tier0Error("Job-55 publication population is duplicated", status="STOP_JOB55_AGGREGATE_QC")
        by_session[session] = record
    if set(by_session) != set(final_qc):
        raise Tier0Error("Job-55 publications do not match finals", status="STOP_JOB55_AGGREGATE_QC")
    for session, qc in final_qc.items():
        publication = by_session[session]
        payload = publication["payload"]
        source = qc["source_attempt"]
        if (
            payload.get("session_qc_sha256") != qc["session_qc_sha256"]
            or payload.get("compressed_bytes") != qc["raw_dbn"]["compressed_bytes"]
            or payload.get("dbn_file_sha256") != qc["raw_dbn"]["file_sha256"]
            or payload.get("decoded_records") != qc["decoder"]["cmbp1_records"]
            or payload.get("source_attempt_id") != source["attempt_id"]
            or payload.get("source_timeseries_result_sequence") != source["timeseries_result_sequence"]
            or payload.get("source_timeseries_result_record_hash") != source["timeseries_result_record_hash"]
        ):
            raise Tier0Error("Job-55 publication/final source triplet drifted", status="STOP_JOB55_AGGREGATE_QC")


def build_job55_aggregate_seal(
    *,
    job55_root: Path,
    summary: Mapping[str, Any],
    final_qc: Mapping[str, Mapping[str, Any]],
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    adoption: Mapping[str, Any],
    volume: VolumeIdentity,
) -> dict[str, Any]:
    _validate_final_publication_binding(summary, final_qc)
    anchor_path = Path(job55_root) / JOB55_ATTEMPT_ANCHOR_NAME
    receipt: dict[str, Any] = {
        "artifact_type": "JOB55_CMBP_TIER0_CAP_AMENDMENT_AGGREGATE_SEAL_V1",
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-aggregate-seal.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "readiness_receipt_file_sha256": readiness_file_sha256,
        "adoption_receipt_sha256": adoption["adoption_sha256"],
        "job55_attempt_anchor_file_sha256": base.file_sha256(anchor_path),
        "job55_attempt_summary_sha256": base.json_sha256(summary),
        "job55_attempt_count": summary["job55_attempt_count"],
        "combined_commitment_count": summary["combined_commitment_count"],
        "combined_committed_quote_total_usd": summary["combined_committed_quote_total_usd"],
        "published_session_count": len(final_qc),
        "decoded_records": sum(int(qc["decoder"]["cmbp1_records"]) for qc in final_qc.values()),
        "final_session_manifest": _final_session_manifest(final_qc),
        "stable_volume_identity": _stable_volume_fields(volume),
    }
    receipt["seal_sha256"] = base.self_hash(receipt, "seal_sha256")
    return receipt


def validate_job55_aggregate_seal(
    path: Path,
    *,
    job55_root: Path,
    summary: Mapping[str, Any],
    final_qc: Mapping[str, Mapping[str, Any]],
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    adoption: Mapping[str, Any],
    volume: VolumeIdentity,
) -> dict[str, Any]:
    _private_regular(path, status="STOP_JOB55_AGGREGATE_QC")
    if Path(path).stat().st_dev != volume.st_dev:
        raise Tier0Error("Job-55 aggregate seal is cross-device", status="STOP_JOB55_AGGREGATE_QC")
    actual = base.strict_json(path)
    expected = build_job55_aggregate_seal(
        job55_root=job55_root,
        summary=summary,
        final_qc=final_qc,
        readiness=readiness,
        readiness_file_sha256=readiness_file_sha256,
        adoption=adoption,
        volume=volume,
    )
    if actual != expected:
        raise Tier0Error("Job-55 aggregate seal does not reconstruct", status="STOP_JOB55_AGGREGATE_QC")
    return actual


def build_job55_aggregate_receipt(
    *,
    bundle: CapAmendmentScopeBundle,
    opening: Job52OpeningEvidence,
    summary: Mapping[str, Any],
    final_qc: Mapping[str, Mapping[str, Any]],
    seal: Mapping[str, Any],
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    adoption: Mapping[str, Any],
    volume: VolumeIdentity,
) -> dict[str, Any]:
    if (
        len(final_qc) != base.EXPECTED_SESSION_COUNT
        or set(final_qc) != {request.session for request in bundle.sessions}
        or summary.get("terminal_authority_failure_observed") is not False
        or summary.get("successful_stream_recovery_required") is not False
        or summary.get("job55_timeseries_call_results") != base.EXPECTED_SESSION_COUNT
        or summary.get("job55_timeseries_call_starts", 0) < base.EXPECTED_SESSION_COUNT
        or Decimal(str(summary.get("combined_committed_quote_total_usd"))) > TOTAL_COMMITTED_CAP_USD
    ):
        raise Tier0Error("Job-55 aggregate prerequisites are incomplete", status="STOP_JOB55_AGGREGATE_QC")
    for value in summary.get("combined_committed_quote_by_session_usd", {}).values():
        if Decimal(str(value)) > PER_SESSION_LIFETIME_CAP_USD:
            raise Tier0Error("Job-55 aggregate session cap is exceeded", status="STOP_JOB55_AGGREGATE_QC")
    _validate_final_publication_binding(summary, final_qc)
    decoder_totals = {
        key: sum(int(qc["decoder"][key]) for qc in final_qc.values()) for key in _AGGREGATE_DECODER_SUM_KEYS
    }
    decoder_totals["max_stream_silence_ns"] = max(
        int(qc["decoder"]["max_stream_silence_ns"]) for qc in final_qc.values()
    )
    decoder_totals["max_instrument_silence_ns"] = max(
        (int(qc["decoder"]["max_instrument_silence_ns"]) for qc in final_qc.values() if qc["decoder"]["max_instrument_silence_ns"] is not None),
        default=None,
    )
    predecessor_quotes = opening.predecessor_summary.get("all_quote_observations", [])
    receipt: dict[str, Any] = {
        "artifact_type": AGGREGATE_RECEIPT_ARTIFACT,
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-acquisition-qc-receipt.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "predecessor_job_id": PREDECESSOR_JOB_ID,
        "status": "JOB55_CMBP_TIER0_CAP_AMENDMENT_ACQUISITION_QC_PASS",
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "readiness_receipt_file_sha256": readiness_file_sha256,
        "adoption_receipt_sha256": adoption["adoption_sha256"],
        "aggregate_seal_sha256": seal["seal_sha256"],
        "aggregate_seal_file_sha256": base.file_sha256(Path(volume.mount_point) / JOB55_ROOT_RELATIVE / JOB55_AGGREGATE_SEAL_NAME),
        "volume_snapshot": dataclasses.asdict(volume),
        "scope_reconciliation": {
            "dataset": base.EXPECTED_DATASET,
            "schema": base.EXPECTED_SCHEMA,
            "stype_in": base.EXPECTED_STYPE_IN,
            "stype_out": base.EXPECTED_STYPE_OUT,
            "session_count": len(final_qc),
            "decoded_records": sum(int(qc["decoder"]["cmbp1_records"]) for qc in final_qc.values()),
            "expected_records": base.EXPECTED_RECORD_COUNT,
            "session_symbol_memberships": sum(len(request.expected_mappings) for request in bundle.sessions),
            "expected_session_symbol_memberships": base.EXPECTED_SESSION_SYMBOLS,
            "out_of_scope_calls": 0,
        },
        "call_accounting": {
            "job51_legacy_cost_call_starts": 1,
            "job51_legacy_timeseries_call_starts": 0,
            "job52_cost_call_starts": opening.cost_call_starts,
            "job52_timeseries_call_starts": opening.time_series_call_starts,
            "job52_timeseries_call_results": opening.time_series_call_results,
            "job52_failed_timeseries_starts": 1,
            "job55_cost_call_starts": summary["job55_cost_call_starts"],
            "job55_cost_call_results": summary["job55_cost_call_results"],
            "job55_cost_call_errors": summary["job55_cost_call_errors"],
            "job55_timeseries_call_starts": summary["job55_timeseries_call_starts"],
            "job55_timeseries_call_results": summary["job55_timeseries_call_results"],
            "job55_timeseries_call_errors": summary["job55_timeseries_call_errors"],
            "all_cost_call_starts": 1 + opening.cost_call_starts + summary["job55_cost_call_starts"],
            "all_timeseries_call_starts": opening.time_series_call_starts + summary["job55_timeseries_call_starts"],
        },
        "commitment_reconciliation": {
            "job52_opening_commitment_count": 1,
            "job52_opening_total_usd": "0.950392448902",
            "job55_commitment_count": summary["job55_commitment_count"],
            "job55_total_usd": summary["job55_committed_quote_total_usd"],
            "combined_commitment_count": summary["combined_commitment_count"],
            "combined_total_usd": summary["combined_committed_quote_total_usd"],
            "combined_by_session_usd": summary["combined_committed_quote_by_session_usd"],
            "per_session_lifetime_cap_usd": "2.00",
            "total_cap_usd": "32.00",
            "all_caps_pass": True,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
        },
        "quote_observations": [*predecessor_quotes, *summary["quote_observations"]],
        "job55_commitments": list(summary["commitments"]),
        "job55_attempts": list(summary["attempts"]),
        "predecessor_failure_disclosure": {
            "job52_failed_partial_bytes": opening.failed_partial_bytes,
            "job52_failed_partial_file_sha256": opening.failed_partial_sha256,
            "job52_failed_start_remains_committed": True,
            "job52_old_cap_terminal_superseded_only_by_job55": True,
        },
        "final_sessions": _final_session_manifest(final_qc),
        "decoder_aggregate": decoder_totals,
        "qc_law": {
            "all_record_counts_reconciled": True,
            "all_mappings_reconciled": True,
            "all_causal_priors_strict": True,
            "all_receive_time_ties_excluded": True,
            "all_global_receive_regressions_zero": decoder_totals["global_receive_regressions"] == 0,
            "streaming_no_full_session_pandas": True,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
        },
    }
    if (
        receipt["scope_reconciliation"]["decoded_records"] != base.EXPECTED_RECORD_COUNT
        or receipt["scope_reconciliation"]["session_symbol_memberships"] != base.EXPECTED_SESSION_SYMBOLS
        or decoder_totals["cmbp1_records"] != base.EXPECTED_RECORD_COUNT
        or decoder_totals["global_receive_regressions"] != 0
    ):
        raise Tier0Error("Job-55 aggregate totals do not reconcile", status="STOP_JOB55_AGGREGATE_QC")
    receipt["receipt_sha256"] = base.self_hash(receipt, "receipt_sha256")
    return receipt


def validate_job55_aggregate_receipt(
    path: Path,
    *,
    bundle: CapAmendmentScopeBundle,
    opening: Job52OpeningEvidence,
    summary: Mapping[str, Any],
    final_qc: Mapping[str, Mapping[str, Any]],
    seal: Mapping[str, Any],
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    adoption: Mapping[str, Any],
    volume: VolumeIdentity,
) -> dict[str, Any]:
    _private_regular(path, status="STOP_JOB55_AGGREGATE_QC")
    if Path(path).stat().st_dev != volume.st_dev:
        raise Tier0Error("Job-55 aggregate receipt is cross-device", status="STOP_JOB55_AGGREGATE_QC")
    actual = base.strict_json(path)
    if actual.get("receipt_sha256") != base.self_hash(actual, "receipt_sha256"):
        raise Tier0Error("Job-55 aggregate receipt self-hash drifted", status="STOP_JOB55_AGGREGATE_QC")
    recorded_volume = _validate_volume_payload(
        actual.get("volume_snapshot"), current=volume, status="STOP_JOB55_AGGREGATE_QC"
    )
    expected = build_job55_aggregate_receipt(
        bundle=bundle,
        opening=opening,
        summary=summary,
        final_qc=final_qc,
        seal=seal,
        readiness=readiness,
        readiness_file_sha256=readiness_file_sha256,
        adoption=adoption,
        volume=VolumeIdentity(**recorded_volume),
    )
    if actual != expected:
        raise Tier0Error("Job-55 aggregate receipt does not reconstruct", status="STOP_JOB55_AGGREGATE_QC")
    return actual


def _job55_junit_counts(path: Path, *, enforce_frozen_population: bool = True) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:  # noqa: BLE001
        raise Tier0Error("Job-55 focused JUnit is unreadable", status="STOP_JOB55_LOCAL_TESTS") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise Tier0Error("Job-55 JUnit has no suite", status="STOP_JOB55_LOCAL_TESTS")
    counts = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    cases = list(root.findall(".//testcase"))
    identities = [
        {"classname": classname, "name": name}
        for classname, name in sorted(
            (str(case.attrib.get("classname", "")), str(case.attrib.get("name", ""))) for case in cases
        )
    ]
    classnames = sorted(set(item["classname"] for item in identities))
    allowed = (
        "v5.tests.test_cmbp_stream",
        "v5.tests.test_cmbp_tier0",
        "v5.tests.test_cmbp_tier0_paid",
        "v5.tests.test_cmbp_tier0_cap_amendment",
    )
    identity_sha = base.json_sha256(identities)
    if (
        counts["tests"] < 12
        or counts["failures"]
        or counts["errors"]
        or counts["skipped"]
        or len(identities) != counts["tests"]
        or not classnames
        or any(not classname.startswith(allowed) for classname in classnames)
    ):
        raise Tier0Error("Job-55 tests are not a clean exact-target pass", status="STOP_JOB55_LOCAL_TESTS")
    if enforce_frozen_population and (
        EXPECTED_FOCUSED_TEST_COUNT <= 0
        or counts["tests"] != EXPECTED_FOCUSED_TEST_COUNT
        or identity_sha != EXPECTED_FOCUSED_TEST_IDENTITY_SHA256
    ):
        raise Tier0Error("Job-55 focused population/identity drifted", status="STOP_JOB55_LOCAL_TESTS")
    return {
        **counts,
        "test_names": sorted(item["name"] for item in identities if item["name"]),
        "test_classnames": classnames,
        "test_case_identity_sha256": identity_sha,
    }


def job55_sdk_identity() -> dict[str, Any]:
    """Reuse the exact sealed Job-52 SDK/HTTP/compression/TLS closure."""

    return paid.paid_sdk_identity()


def build_job55_readiness_receipt(repo_root: Path, *, test_report_path: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    bundle = load_cap_amendment_scope_bundle(root)
    required = bundle.contract.get("required_bound_files")
    if not isinstance(required, list) or len(required) != len(set(required)):
        raise Tier0Error("Job-55 bound-file list is malformed", status="STOP_JOB55_CONTRACT_DRIFT")
    bound_files: dict[str, str] = {}
    for relative in required:
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise Tier0Error("Job-55 bound path is unsafe", status="STOP_JOB55_CONTRACT_DRIFT")
        path = root / relative
        _private_regular(path, status="STOP_JOB55_CONTRACT_DRIFT")
        bound_files[relative] = base.file_sha256(path)
    report = Path(os.path.abspath(test_report_path))
    expected_report = Path(os.path.abspath(_paths(root)["test_report"]))
    if report != expected_report or Path(test_report_path).is_symlink():
        raise Tier0Error("Job-55 JUnit path is not canonical", status="STOP_JOB55_LOCAL_TESTS")
    _private_regular(report, status="STOP_JOB55_LOCAL_TESTS")
    predecessor_readiness = paid.validate_paid_readiness_receipt(root)
    receipt: dict[str, Any] = {
        "artifact_type": READINESS_ARTIFACT,
        "schema_version": "v5.job55-cmbp-tier0-cap-amendment-local-readiness.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "predecessor_job_id": PREDECESSOR_JOB_ID,
        "status": READINESS_STATUS,
        "status_meaning": "LOCAL_BUILD_AND_TEST_SEAL_ONLY; NO_VENDOR_OR_ACQUISITION_RESULT",
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "plan_file_sha256": EXPECTED_PLAN_FILE_SHA256,
        "job52_program_contract_sha256": EXPECTED_JOB52_CONTRACT_SHA256,
        "job52_program_contract_file_sha256": EXPECTED_JOB52_CONTRACT_FILE_SHA256,
        "job52_readiness_receipt_sha256": predecessor_readiness["receipt_sha256"],
        "job52_readiness_receipt_file_sha256": EXPECTED_JOB52_READINESS_FILE_SHA256,
        "opening_evidence_pins_sha256": base.json_sha256(
            {
                "directories": list(EXPECTED_JOB52_EVIDENCE_PINS.directories),
                "file_sha256": dict(EXPECTED_JOB52_EVIDENCE_PINS.file_sha256),
                "anchor_sha256": EXPECTED_JOB52_EVIDENCE_PINS.anchor_sha256,
                "failed_partial_bytes": EXPECTED_JOB52_EVIDENCE_PINS.failed_partial_bytes,
                "failed_partial_sha256": EXPECTED_JOB52_EVIDENCE_PINS.failed_partial_sha256,
            }
        ),
        "money_boundary": {
            "job52_opening_commitment_usd": "0.950392448902",
            "per_session_lifetime_committed_quote_cap_usd": "2.00",
            "total_committed_quote_cap_usd": "32.00",
            "failed_session_job55_cost_call_allowance": 1,
            "failed_session_job55_time_series_start_allowance": 1,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
        },
        "bound_files": bound_files,
        "focused_test_report": {
            "path": str(report.relative_to(root)),
            "file_sha256": base.file_sha256(report),
            "runner": "./.venv/bin/python -m pytest",
            "runner_flags": [
                "-c",
                "/dev/null",
                "--rootdir=<REPOSITORY_ROOT>",
                "-p",
                "no:cacheprovider",
                "-q",
            ],
            "test_targets": [
                "v5/tests/test_cmbp_stream.py",
                "v5/tests/test_cmbp_tier0.py",
                "v5/tests/test_cmbp_tier0_paid.py",
                "v5/tests/test_cmbp_tier0_cap_amendment.py",
            ],
            "environment_allowlist": [
                "PATH",
                "LANG",
                "LC_ALL",
                "PYTHONHASHSEED",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            ],
            "pytest_plugin_autoload_disabled": True,
            "repository_and_environment_pytest_addopts_ignored": True,
            "report_published_atomically_after_subprocess_exit_zero": True,
            **_job55_junit_counts(report),
        },
        "sdk_identity": job55_sdk_identity(),
        "integrity": {
            "credential_read": False,
            "authenticated_client_constructed": False,
            "external_calls": 0,
            "metadata_calls": 0,
            "timeseries_calls": 0,
            "data_downloaded": False,
            "acquisition_initiated": False,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "models_fit": 0,
            "broker_calls": 0,
            "orders_submitted": 0,
        },
        "authorization_effect": "LOCAL_READINESS_ONLY; EXECUTION_REMAINS_BOUND_TO_EXACT_JOB55_OWNER_CAP_AUTHORITY",
    }
    receipt["receipt_sha256"] = base.self_hash(receipt, "receipt_sha256")
    return receipt


def validate_job55_readiness_receipt(repo_root: Path, path: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    canonical = Path(os.path.abspath(_paths(root)["readiness"]))
    unresolved = Path(canonical if path is None else path)
    target = Path(os.path.abspath(unresolved))
    if target != canonical or unresolved.is_symlink():
        raise Tier0Error("Job-55 readiness path is not canonical", status="STOP_JOB55_READINESS")
    _private_regular(target, status="STOP_JOB55_READINESS")
    receipt = base.strict_json(target)
    if (
        receipt.get("artifact_type") != READINESS_ARTIFACT
        or receipt.get("schema_version") != "v5.job55-cmbp-tier0-cap-amendment-local-readiness.v1"
        or receipt.get("status") != READINESS_STATUS
        or receipt.get("receipt_sha256") != base.self_hash(receipt, "receipt_sha256")
    ):
        raise Tier0Error("Job-55 readiness identity drifted", status="STOP_JOB55_READINESS")
    expected = build_job55_readiness_receipt(root, test_report_path=_paths(root)["test_report"])
    if receipt != expected:
        raise Tier0Error("Job-55 readiness does not reconstruct", status="STOP_JOB55_READINESS")
    return receipt


def seal_job55_readiness(repo_root: Path) -> Path:
    root = Path(repo_root).resolve()
    paths = _paths(root)
    receipt = build_job55_readiness_receipt(root, test_report_path=paths["test_report"])
    if paths["readiness"].exists() or paths["readiness"].is_symlink():
        raise Tier0Error("Job-55 readiness V1 already exists", status="STOP_JOB55_READINESS")
    base.write_canonical_exclusive(paths["readiness"], receipt)
    validate_job55_readiness_receipt(root, paths["readiness"])
    return paths["readiness"]


def reconstruct_job55_local_state(
    *,
    volume: VolumeIdentity,
    require_client_constructed: bool = False,
) -> dict[str, Any]:
    """Canonical precredential/prepair reconstruction used by runner and primitive."""

    volume = _assert_current_production_volume(volume)
    repo_root = Path(__file__).resolve().parents[2]
    bundle = load_cap_amendment_scope_bundle(repo_root)
    readiness = validate_job55_readiness_receipt(repo_root)
    readiness_file_sha = base.file_sha256(_paths(repo_root)["readiness"])
    job51_root = Path(volume.mount_point) / JOB51_ROOT_RELATIVE
    job55_root = Path(volume.mount_point) / JOB55_ROOT_RELATIVE
    opening = validate_job52_opening_evidence(job51_root, volume=volume)
    adoption = ensure_job52_terminal_adoption(job55_root, opening, volume=volume, allow_initialize=False)
    ensure_job55_lock_binding(
        job55_root,
        volume=volume,
        readiness_receipt_sha256=readiness["receipt_sha256"],
        readiness_receipt_file_sha256=readiness_file_sha,
        adoption_receipt_sha256=adoption["adoption_sha256"],
        allow_initialize=False,
    )
    validate_job55_attempt_anchor(
        job55_root,
        volume=volume,
        readiness_receipt_sha256=readiness["receipt_sha256"],
        readiness_receipt_file_sha256=readiness_file_sha,
        adoption_receipt_sha256=adoption["adoption_sha256"],
        allow_initialize=False,
        repair_header_only=False,
    )
    validate_job55_job_tree(job55_root, volume=volume)
    sdk_sha = base.json_sha256(job55_sdk_identity())
    summary = summarize_job55_attempts(
        job55_root,
        bundle.sessions,
        opening=opening,
        volume=volume,
        readiness_receipt_sha256=readiness["receipt_sha256"],
        readiness_receipt_file_sha256=readiness_file_sha,
        adoption_receipt_sha256=adoption["adoption_sha256"],
        require_client_constructed=require_client_constructed,
        expected_sdk_identity_sha256=sdk_sha,
    )
    if summary["terminal_authority_failure_observed"]:
        raise Tier0Error("Job-55 authority is terminal", status="STOP_JOB55_AUTHORITY_TERMINAL")
    if summary["successful_stream_recovery_required"]:
        raise Tier0Error("Job-55 successful stream requires local recovery", status="STOP_JOB55_RECOVERY_REQUIRED")
    if (job55_root / JOB55_AGGREGATE_SEAL_NAME).exists() or (job55_root / "receipts" / JOB55_AGGREGATE_NAME).exists():
        raise Tier0Error("Job-55 is already sealed or complete", status="STOP_JOB55_ALREADY_COMPLETE")
    source_lookup = job55_source_record_lookup(summary)
    final_qc = validate_job55_existing_session_population(
        job55_root,
        bundle=bundle,
        readiness=readiness,
        readiness_file_sha256=readiness_file_sha,
        adoption=adoption,
        volume=volume,
        source_record_lookup=source_lookup,
        published_sessions_in_journals=summary["published_sessions_in_journals"],
    )
    if set(final_qc) != set(summary["published_sessions_in_journals"]):
        raise Tier0Error("Job-55 final/publication population is inconsistent", status="STOP_JOB55_SESSION_QC")
    return {
        "bundle": bundle,
        "readiness": readiness,
        "readiness_file_sha256": readiness_file_sha,
        "opening": opening,
        "adoption": adoption,
        "summary": summary,
        "budget_state": amended_budget_state_from_summary(summary),
        "source_record_lookup": source_lookup,
        "final_qc": final_qc,
        "job51_root": job51_root,
        "job55_root": job55_root,
        "sdk_identity_sha256": sdk_sha,
    }
