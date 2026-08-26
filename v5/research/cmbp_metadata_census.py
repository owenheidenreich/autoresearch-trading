"""Fail-closed execution primitives for the Job-50 CMBP metadata census.

The executor accepts an already-constructed client.  It cannot construct a
Databento client, read a credential, request time-series data, or download
data.  Its only executable call surface is the four methods frozen by the
Job-49 catalogue declaration.

Every attempted SDK invocation is recorded to an exclusive, hash-chained
JSONL journal before and after the call, with an ``fsync`` after each record.
A final response artifact is written exclusively only after the complete
2,440-call census succeeds.  An interrupted attempt is evidence of an
interrupted attempt; V1 never resumes or launders it into a complete result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from v5.research import cmbp_catalogue_preflight as catalogue


ARTIFACT_VERSION = "v5.cmbp-metadata-census.v1"
CALL_JOURNAL_ARTIFACT = "JOB50_CMBP_CALL_JOURNAL_V1"
CALL_JOURNAL_SCHEMA_VERSION = "v5.cmbp-metadata-call-journal.v1"
EXECUTION_AUDIT_ARTIFACT = "JOB50_METADATA_EXECUTION_AUDIT_V1"
VENDOR_AUTH_ARTIFACT = "JOB50_VENDOR_RUN_AUTHORIZATION_V2"
VENDOR_AUTH_SCHEMA_VERSION = "v5.job50-vendor-run-authorization.v2"
EXPECTED_SDK_PACKAGE = "databento"
EXPECTED_SDK_VERSION = "0.77.0"
BASE_PROGRAM_CONTRACT_SHA256 = "0d581abd9edfcc015d3d2f1574da3be15a3fcd62b275e31dd7485f7d279ca75d"
BASE_PROGRAM_CONTRACT_FILE_SHA256 = "5ba7fdf54d44e683cfe0a663d6929593eff7c9ea897aeb855734adc139837ae2"
INTERMEDIATE_PROGRAM_CONTRACT_SHA256 = "3a1cdff65beb11439be93dcf0a0dfa0554cf5e665c339b9f4bdd0ff6eb882b48"
INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256 = "dc6563c0efd2b4281f5907e7f45fbea059abf31937d529965dd0f596ce48679d"
PRIOR_PROGRAM_CONTRACT_SHA256 = "a8528c0311d38a2539c7493cd380967beca707b24ef83ac7eae440662c824451"
PRIOR_PROGRAM_CONTRACT_FILE_SHA256 = "e739605ea4d4c96a258d651503de073351e12bb6b258c9ab76630d5b7f56a3ef"
PROGRAM_CONTRACT_SHA256 = "01dbb71e7e58d08adf05114c2e89c74fa8870b653e527a66df083ae5b2408794"
PROGRAM_CONTRACT_FILE_SHA256 = "34bc793896c29e6675e539b7bec9cb3d03cf4e189fc5bc053e4f3f1b524829d1"
EXPECTED_CALL_COUNT = 2_440
GENESIS_HASH = "0" * 64
SHA256 = re.compile(r"^[0-9a-f]{64}$")

METHOD_ORDER = (
    "metadata.get_dataset_range",
    "symbology.resolve",
    "metadata.get_cost",
    "metadata.get_record_count",
)
ALLOWED_METHODS = METHOD_ORDER
SESSION_METHOD_ORDER = METHOD_ORDER[1:]
EXPECTED_METHOD_COUNTS = {
    "metadata.get_dataset_range": 1,
    "symbology.resolve": 813,
    "metadata.get_cost": 813,
    "metadata.get_record_count": 813,
}
AUTHORIZATION_EFFECT = "METADATA_ONLY_VENDOR_CONTACT; NO_TIMESERIES_DOWNLOAD_ACQUISITION_SPEND_OR_OUTCOMES"
EXTERNAL_RECEIPT_AUTHORIZATION_EFFECT = (
    "METADATA_PREFLIGHT_EVIDENCE_ONLY; "
    "NO_DOWNLOAD_ACQUISITION_SPEND_OUTCOME_MODEL_BROKER_OR_ORDER_AUTHORITY"
)
EXTERNAL_PASS_INTEGRITY = {
    "credential_read": True,
    "authenticated_client_constructed": True,
    "sdk_metadata_method_invocations": EXPECTED_CALL_COUNT,
    "low_level_network_request_count": "UNOBSERVED",
    "timeseries_calls": 0,
    "download_calls": 0,
    "data_downloaded": False,
    "data_acquired": False,
    "purchase_or_acquisition_spend_initiated_usd": 0,
    "strategy_outcomes_read": False,
    "reserved_economics_read": False,
    "broker_calls": 0,
    "models_fit": 0,
    "orders_submitted": 0,
}

JOB50_STOP_STATUSES = frozenset(
    {
        "STOP_CONTRACT_OR_DECLARATION_DRIFT",
        "STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
        "STOP_AUTHORIZATION_MISSING_OR_INVALID",
        "STOP_CREDENTIAL_ROTATION_UNRESOLVED",
        "STOP_QUOTED_CEILING_MISSING_OR_INVALID",
        "STOP_FORBIDDEN_METHOD_OR_SCOPE",
        "STOP_ATTEMPT_PATH_EXISTS",
        "STOP_CALL_JOURNAL_INVALID",
        "STOP_PARTIAL_OR_CRASHED_ATTEMPT",
        "STOP_VENDOR_AUTH_OR_ENTITLEMENT",
        "STOP_SCHEMA_UNAVAILABLE",
        "STOP_SYMBOL_OR_EXPIRY_MISMATCH",
        "STOP_ZERO_RECORDS",
        "STOP_NONFINITE_COST",
        "STOP_MISSING_SESSION_RESPONSE",
        "STOP_OVER_QUOTED_CEILING",
        "STOP_RESPONSE_OR_RECEIPT_INVALID",
        "STOP_LOCAL_READINESS_RECEIPT_INVALID",
        "JOB50_AUTHORITY_VIOLATION",
    }
)
JOB49_STOP_NORMALIZATION = {
    catalogue.STOP_SCOPE_OR_CODE_DRIFT: "STOP_CONTRACT_OR_DECLARATION_DRIFT",
    catalogue.STOP_MISSING_KEY_OR_ENTITLEMENT: "STOP_VENDOR_AUTH_OR_ENTITLEMENT",
    catalogue.STOP_OVER_HARD_CAP: "STOP_OVER_QUOTED_CEILING",
    catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH: "STOP_SYMBOL_OR_EXPIRY_MISMATCH",
    catalogue.STOP_SCHEMA_UNAVAILABLE: "STOP_SCHEMA_UNAVAILABLE",
    catalogue.STOP_ZERO_RECORDS: "STOP_ZERO_RECORDS",
    catalogue.STOP_NONFINITE_COST: "STOP_NONFINITE_COST",
    catalogue.STOP_MISSING_SESSION_RESPONSE: "STOP_MISSING_SESSION_RESPONSE",
}


def normalize_job50_status(status: Any) -> str:
    """Map every inherited/known stop idempotently into the Job-50 vocabulary."""

    # Treat the value as untrusted diagnostic input.  In particular, checking
    # membership on a caller-supplied list/dict would itself raise TypeError
    # and escape the promised total public-boundary normalization.
    if isinstance(status, str):
        if status in JOB50_STOP_STATUSES:
            return status
        if status in JOB49_STOP_NORMALIZATION:
            return JOB49_STOP_NORMALIZATION[status]
    return "STOP_CONTRACT_OR_DECLARATION_DRIFT"


class MetadataCensusError(ValueError):
    """The census cannot continue without violating its frozen contract."""

    def __init__(self, message: str, *, status: str = catalogue.STOP_SCOPE_OR_CODE_DRIFT) -> None:
        super().__init__(message)
        self.status = normalize_job50_status(status)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically and reject NaN or unsupported objects."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_no_symlink_components(path: Path, *, anchor: Path) -> None:
    """Reject every existing symlink component under one lexical anchor."""

    root = Path(os.path.abspath(anchor))
    target = Path(os.path.abspath(path))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise MetadataCensusError(
            f"path escapes its canonical anchor: {target}",
            status="STOP_ATTEMPT_PATH_EXISTS",
        ) from exc
    cursor = root
    _require(
        not cursor.is_symlink(),
        f"canonical anchor is a symlink: {cursor}",
        status="STOP_ATTEMPT_PATH_EXISTS",
    )
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise MetadataCensusError(
                f"canonical production path contains a symlink: {cursor}",
                status="STOP_ATTEMPT_PATH_EXISTS",
            )


def self_hash(value: Mapping[str, Any], field: str) -> str:
    semantic = dict(value)
    semantic.pop(field, None)
    return json_sha256(semantic)


def load_json(path: Path) -> dict[str, Any]:
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MetadataCensusError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def _constant(value: str) -> Any:
        raise MetadataCensusError(f"nonfinite JSON constant {value!r} in {path}")

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_object,
            parse_constant=_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataCensusError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MetadataCensusError(f"JSON artifact is not an object: {path}")
    return value


def _strict_json_bytes(raw: bytes, *, origin: str) -> Any:
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MetadataCensusError(f"duplicate JSON key {key!r} in {origin}")
            result[key] = value
        return result

    def _constant(value: str) -> Any:
        raise MetadataCensusError(f"nonfinite JSON constant {value!r} in {origin}")

    try:
        return json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataCensusError(f"invalid JSON in {origin}: {exc}") from exc


def _load_canonical_json_output(path: Path) -> dict[str, Any]:
    """Strictly reread one runner-owned compact JSON artifact from disk."""

    value = load_json(path)
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise MetadataCensusError(f"cannot reread JSON artifact {path}: {exc}") from exc
    _require(
        raw == canonical_json_bytes(value) + b"\n",
        f"runner-owned JSON artifact is not canonical or complete: {path}",
        status="STOP_RESPONSE_OR_RECEIPT_INVALID",
    )
    return value


def _require(condition: bool, message: str, *, status: str = catalogue.STOP_SCOPE_OR_CODE_DRIFT) -> None:
    if not condition:
        raise MetadataCensusError(message, status=status)


def _require_sha(value: Any, name: str) -> str:
    _require(isinstance(value, str) and SHA256.fullmatch(value) is not None, f"{name} is not a lowercase SHA-256")
    return value


def _decimal_text(value: Any, name: str) -> str:
    _require(not isinstance(value, bool) and value is not None, f"{name} is not a decimal", status=catalogue.STOP_NONFINITE_COST)
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name} is nonfinite", status=catalogue.STOP_NONFINITE_COST)
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MetadataCensusError(f"{name} is not a decimal", status=catalogue.STOP_NONFINITE_COST) from exc
    _require(parsed.is_finite() and parsed >= 0, f"{name} must be finite and nonnegative", status=catalogue.STOP_NONFINITE_COST)
    rendered = format(parsed, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _authorization_now_utc() -> datetime:
    """Private deterministic-test seam; production APIs expose no clock input."""

    return datetime.now(timezone.utc)


def _sealed_contract_sha256(
    contract: Mapping[str, Any],
    *,
    artifact_type: str,
    schema_version: str,
    expected_sha256: str,
) -> str:
    _require(contract.get("artifact_type") == artifact_type, "wrong Job-50 program contract artifact")
    _require(contract.get("schema_version") == schema_version, "wrong Job-50 program contract schema")
    hash_block = contract.get("self_hash")
    _require(isinstance(hash_block, Mapping), "program contract has no semantic self-hash")
    claimed = _require_sha(hash_block.get("value"), "program contract self_hash.value")
    semantic = json.loads(canonical_json_bytes(contract))
    semantic["self_hash"]["value"] = None
    semantic["self_hash"]["status"] = "NORMALIZED_FOR_HASH"
    _require(claimed == json_sha256(semantic), "program contract semantic self-hash mismatch")
    _require(claimed == expected_sha256, "program contract identity is not the frozen Job-50 contract")
    return claimed


def _contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    """Verify and return the effective sealed V4 contract identity."""

    return _sealed_contract_sha256(
        contract,
        artifact_type="JOB50_PROGRAM_CONTRACT_V4",
        schema_version="v5.cmbp-metadata-census-program-contract.v4",
        expected_sha256=PROGRAM_CONTRACT_SHA256,
    )


def _prior_contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    return _sealed_contract_sha256(
        contract,
        artifact_type="JOB50_PROGRAM_CONTRACT_V3",
        schema_version="v5.cmbp-metadata-census-program-contract.v3",
        expected_sha256=PRIOR_PROGRAM_CONTRACT_SHA256,
    )


def _intermediate_contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    return _sealed_contract_sha256(
        contract,
        artifact_type="JOB50_PROGRAM_CONTRACT_V2",
        schema_version="v5.cmbp-metadata-census-program-contract.v2",
        expected_sha256=INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
    )


def _base_contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    return _sealed_contract_sha256(
        contract,
        artifact_type="JOB50_PROGRAM_CONTRACT_V1",
        schema_version="v5.cmbp-metadata-census-program-contract.v1",
        expected_sha256=BASE_PROGRAM_CONTRACT_SHA256,
    )


def _expected_requests(declaration: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        catalogue.validate_catalogue_declaration(declaration)
    except catalogue.CataloguePreflightError as exc:
        raise MetadataCensusError(str(exc), status=exc.status) from exc
    requests: list[dict[str, Any]] = []
    global_requests = declaration.get("global_requests")
    _require(isinstance(global_requests, list) and len(global_requests) == 1, "declaration must contain exactly one global request")
    requests.append({"session": None, "request": global_requests[0]})
    for row in declaration["sessions"]:
        if not row["requests"]:
            continue
        _require(
            [request.get("method") for request in row["requests"]] == list(SESSION_METHOD_ORDER),
            f"{row['session']}: frozen per-session method order drifted",
        )
        requests.extend(
            {"session": row["session"], "request": request}
            for request in row["requests"]
        )
    _require(len(requests) == EXPECTED_CALL_COUNT, f"expected {EXPECTED_CALL_COUNT} calls, declaration yields {len(requests)}")
    counts = Counter(item["request"]["method"] for item in requests)
    _require(dict(counts) == EXPECTED_METHOD_COUNTS, f"frozen method counts drifted: {dict(counts)}")
    return requests


def inspect_declaration(
    declaration: Mapping[str, Any],
    *,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a local-only execution summary; no client or credential is touched."""

    requests = _expected_requests(declaration)
    if expected_contract_sha256 is not None:
        _require_sha(expected_contract_sha256, "expected_contract_sha256")
    return {
        "artifact_type": "JOB50_METADATA_CENSUS_INSPECTION_V1",
        "artifact_version": ARTIFACT_VERSION,
        "declaration_sha256": declaration["declaration_sha256"],
        "request_manifest_sha256": declaration["request_manifest_sha256"],
        "contract_sha256": expected_contract_sha256,
        "dataset": declaration["dataset"],
        "schema": declaration["schema"],
        "stype_in": declaration["stype_in"],
        "source_session_count": declaration["source_session_count"],
        "request_session_count": declaration["request_session_count"],
        "expected_call_count": len(requests),
        "expected_method_counts": dict(EXPECTED_METHOD_COUNTS),
        "expected_sdk_package": EXPECTED_SDK_PACKAGE,
        "expected_sdk_version": EXPECTED_SDK_VERSION,
        "network_calls_performed": 0,
        "credential_read": False,
        "authorization_effect": "NONE",
    }


def verify_vendor_run_authorization(
    authorization: Mapping[str, Any],
    declaration: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    prior_contract: Mapping[str, Any],
    intermediate_contract: Mapping[str, Any],
    base_contract: Mapping[str, Any],
    local_readiness_receipt: Mapping[str, Any],
    local_readiness_receipt_file_sha256: str,
    authorization_path: Path,
    repo_root: Path,
    attempt_id: str,
    now_utc: datetime | None = None,
    declaration_file_sha256: str,
    enforce_current_freshness: bool = True,
) -> Decimal:
    """Validate the V4 projection of the fresh, one-attempt V2 authorization."""

    def auth_require(
        condition: bool,
        message: str,
        *,
        status: str = "STOP_AUTHORIZATION_MISSING_OR_INVALID",
    ) -> None:
        if not condition:
            raise MetadataCensusError(message, status=status)

    def canonical_uuid4(value: Any, name: str) -> str:
        auth_require(isinstance(value, str), f"{name} is not canonical UUID text")
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError) as exc:
            raise MetadataCensusError(
                f"{name} is not a canonical UUID",
                status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
            ) from exc
        auth_require(parsed.version == 4 and str(parsed) == value, f"{name} must be lowercase canonical UUIDv4")
        return value

    def canonical_time(value: Any, name: str) -> datetime:
        auth_require(isinstance(value, str) and value.endswith("Z"), f"{name} is not canonical UTC text")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MetadataCensusError(
                f"{name} is not a valid UTC timestamp",
                status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
            ) from exc
        auth_require(parsed.tzinfo == timezone.utc, f"{name} must be UTC")
        auth_require(parsed.isoformat(timespec="microseconds").replace("+00:00", "Z") == value, f"{name} is not canonical microsecond UTC")
        return parsed

    try:
        catalogue.validate_catalogue_declaration(declaration)
    except catalogue.CataloguePreflightError as exc:
        raise MetadataCensusError(str(exc), status="STOP_CONTRACT_OR_DECLARATION_DRIFT") from exc
    contract_sha = _contract_semantic_sha256(contract)
    _prior_contract_semantic_sha256(prior_contract)
    _intermediate_contract_semantic_sha256(intermediate_contract)
    _base_contract_semantic_sha256(base_contract)
    exact_fields = {
        "artifact_type", "schema_version", "authorization_id", "attempt_id",
        "issued_at_utc", "expires_at_utc", "program_contract_sha256",
        "prior_program_contract_sha256", "prior_program_contract_file_sha256",
        "intermediate_program_contract_sha256",
        "intermediate_program_contract_file_sha256",
        "base_program_contract_sha256", "base_program_contract_file_sha256",
        "local_readiness_receipt_sha256", "local_readiness_receipt_file_sha256",
        "declaration_sha256", "declaration_file_sha256",
        "credential_rotation_attested", "credential_attestation_recorded_at_utc",
        "quoted_acquisition_ceiling_usd", "metadata_only", "authorized_methods",
        "current_conversation_authorization_sha256", "authorization_effect",
        "one_attempt_only", "authorization_sha256",
    }
    auth_require(set(authorization) == exact_fields, "vendor authorization exact-field set drifted")
    auth_require(authorization.get("artifact_type") == VENDOR_AUTH_ARTIFACT, "wrong vendor authorization artifact type")
    auth_require(authorization.get("schema_version") == VENDOR_AUTH_SCHEMA_VERSION, "wrong vendor authorization schema version")
    authorization_id = canonical_uuid4(authorization.get("authorization_id"), "authorization_id")
    authorized_attempt = canonical_uuid4(authorization.get("attempt_id"), "attempt_id")
    auth_require(authorization_id != authorized_attempt, "authorization and attempt identities must differ")
    auth_require(attempt_id == authorized_attempt, "authorization attempt identity drifted")
    expected_path = (
        Path(repo_root).resolve()
        / "v5/work/cmbp-metadata-census/authorizations"
        / attempt_id
        / "VENDOR_RUN_AUTHORIZATION_V2.json"
    )
    auth_require(Path(authorization_path).resolve() == expected_path, "authorization path is not canonical")
    issued = canonical_time(authorization.get("issued_at_utc"), "issued_at_utc")
    expires = canonical_time(authorization.get("expires_at_utc"), "expires_at_utc")
    now = now_utc or _authorization_now_utc()
    auth_require(now.tzinfo is not None, "authorization validation clock is timezone-naive")
    now = now.astimezone(timezone.utc)
    auth_require(issued < expires <= issued + timedelta(hours=24), "authorization lifetime exceeds 24 hours or is not increasing")
    if enforce_current_freshness:
        auth_require(issued <= now < expires, "authorization is not currently fresh")

    auth_require(authorization.get("program_contract_sha256") == contract_sha, "vendor authorization V2 contract link drifted")
    auth_require(
        authorization.get("prior_program_contract_sha256")
        == PRIOR_PROGRAM_CONTRACT_SHA256,
        "vendor authorization V2 prior V3 semantic link drifted",
    )
    auth_require(
        authorization.get("prior_program_contract_file_sha256")
        == PRIOR_PROGRAM_CONTRACT_FILE_SHA256,
        "vendor authorization V2 prior V3 raw-file link drifted",
    )
    auth_require(
        authorization.get("intermediate_program_contract_sha256")
        == INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "vendor authorization V2 intermediate semantic link drifted",
    )
    auth_require(
        authorization.get("intermediate_program_contract_file_sha256")
        == INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256,
        "vendor authorization V2 intermediate raw-file link drifted",
    )
    auth_require(authorization.get("base_program_contract_sha256") == BASE_PROGRAM_CONTRACT_SHA256, "vendor authorization V1 contract link drifted")
    auth_require(authorization.get("base_program_contract_file_sha256") == BASE_PROGRAM_CONTRACT_FILE_SHA256, "vendor authorization V1 raw-file link drifted")
    local_receipt_sha = local_readiness_receipt.get("receipt_sha256")
    auth_require(isinstance(local_receipt_sha, str) and SHA256.fullmatch(local_receipt_sha) is not None, "local readiness receipt self-hash is missing")
    auth_require(authorization.get("local_readiness_receipt_sha256") == local_receipt_sha, "authorization/local-readiness semantic link drifted")
    auth_require(authorization.get("local_readiness_receipt_file_sha256") == local_readiness_receipt_file_sha256, "authorization/local-readiness raw link drifted")
    auth_require(authorization.get("declaration_sha256") == declaration["declaration_sha256"], "vendor authorization declaration link drifted")
    auth_require(authorization.get("declaration_file_sha256") == declaration_file_sha256, "vendor authorization declaration-file link drifted")
    auth_require(
        authorization.get("credential_rotation_attested") is True,
        "credential rotation is not attested",
        status="STOP_CREDENTIAL_ROTATION_UNRESOLVED",
    )
    canonical_time(authorization.get("credential_attestation_recorded_at_utc"), "credential_attestation_recorded_at_utc")
    cap_text = authorization.get("quoted_acquisition_ceiling_usd")
    auth_require(
        isinstance(cap_text, str)
        and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", cap_text) is not None,
        "quoted acquisition ceiling is not canonical decimal text",
        status="STOP_QUOTED_CEILING_MISSING_OR_INVALID",
    )
    try:
        cap = Decimal(cap_text)
    except InvalidOperation as exc:
        raise MetadataCensusError(
            "quoted acquisition ceiling is invalid",
            status="STOP_QUOTED_CEILING_MISSING_OR_INVALID",
        ) from exc
    auth_require(
        cap.is_finite() and cap >= 0,
        "quoted acquisition ceiling is nonfinite or negative",
        status="STOP_QUOTED_CEILING_MISSING_OR_INVALID",
    )
    auth_require(
        authorization.get("metadata_only") is True,
        "authorization is not metadata-only",
        status="STOP_FORBIDDEN_METHOD_OR_SCOPE",
    )
    auth_require(
        authorization.get("authorized_methods") == list(METHOD_ORDER),
        "authorized method allowlist drifted",
        status="STOP_FORBIDDEN_METHOD_OR_SCOPE",
    )
    auth_require(isinstance(authorization.get("current_conversation_authorization_sha256"), str) and SHA256.fullmatch(authorization["current_conversation_authorization_sha256"]) is not None, "current-conversation authorization hash is invalid")
    auth_require(authorization.get("authorization_effect") == AUTHORIZATION_EFFECT, "authorization effect drifted")
    auth_require(authorization.get("one_attempt_only") is True, "authorization is not one-attempt-only")
    claimed = authorization.get("authorization_sha256")
    auth_require(isinstance(claimed, str) and SHA256.fullmatch(claimed) is not None, "authorization self-hash is invalid")
    auth_require(claimed == self_hash(authorization, "authorization_sha256"), "vendor authorization self-hash mismatch")
    return cap


def consume_vendor_run_authorization(
    authorization: Mapping[str, Any],
    *,
    authorization_path: Path,
    repo_root: Path,
    local_readiness_receipt: Mapping[str, Any],
    consumed_at_utc: str | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Durably spend one authorization before any credential is read."""

    root = Path(repo_root).resolve()
    attempt_id = str(authorization["attempt_id"])
    authorization_sha = str(authorization["authorization_sha256"])
    marker_path = (
        root
        / "v5/work/cmbp-metadata-census/authorization-consumptions"
        / f"{authorization_sha}.json"
    )
    attempt_relative = f"v5/work/cmbp-metadata-census/external-attempts/{attempt_id}"
    attempt_directory = root / attempt_relative
    _require_no_symlink_components(marker_path.parent, anchor=root)
    _require_no_symlink_components(attempt_directory.parent, anchor=root)
    timestamp = consumed_at_utc or _utc_now()
    record = {
        "artifact_type": "JOB50_VENDOR_AUTHORIZATION_CONSUMPTION_V1",
        "schema_version": "v5.job50-vendor-authorization-consumption.v1",
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization_sha,
        "authorization_file_sha256": file_sha256(authorization_path),
        "attempt_id": attempt_id,
        "attempt_directory": attempt_relative,
        "program_contract_sha256": PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_sha256": PRIOR_PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_file_sha256": PRIOR_PROGRAM_CONTRACT_FILE_SHA256,
        "intermediate_program_contract_sha256": INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_file_sha256": INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256,
        "base_program_contract_sha256": BASE_PROGRAM_CONTRACT_SHA256,
        "base_program_contract_file_sha256": BASE_PROGRAM_CONTRACT_FILE_SHA256,
        "local_readiness_receipt_sha256": local_readiness_receipt["receipt_sha256"],
        "declaration_sha256": authorization["declaration_sha256"],
        "issued_at_utc": authorization["issued_at_utc"],
        "expires_at_utc": authorization["expires_at_utc"],
        "consumed_at_utc": timestamp,
    }
    if marker_path.is_symlink():
        raise MetadataCensusError(
            f"authorization consumption marker is a symlink: {marker_path}",
            status="STOP_ATTEMPT_PATH_EXISTS",
        )
    if marker_path.exists():
        raise MetadataCensusError(
            "vendor authorization was already consumed",
            status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
        )
    try:
        _write_json_exclusive(marker_path, record, mode=0o600)
    except MetadataCensusError as exc:
        raise MetadataCensusError(
            "vendor authorization consumption marker could not be created exclusively",
            status=(
                "STOP_AUTHORIZATION_MISSING_OR_INVALID"
                if marker_path.exists()
                else "STOP_PARTIAL_OR_CRASHED_ATTEMPT"
            ),
        ) from exc
    _require_no_symlink_components(marker_path, anchor=root)
    attempt_directory.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(attempt_directory.parent, anchor=root)
    try:
        attempt_directory.mkdir(mode=0o700, exist_ok=False)
    except OSError as exc:
        raise MetadataCensusError(
            f"authorization was consumed but exclusive attempt path cannot be created: {attempt_directory}",
            status="STOP_ATTEMPT_PATH_EXISTS",
        ) from exc
    directory = os.open(attempt_directory.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return record, marker_path, attempt_directory


def verify_authorization_consumption(
    record: Mapping[str, Any],
    *,
    record_path: Path,
    authorization: Mapping[str, Any],
    authorization_path: Path,
    repo_root: Path,
    local_readiness_receipt: Mapping[str, Any],
) -> Path:
    exact_fields = {
        "artifact_type", "schema_version", "authorization_id", "authorization_sha256",
        "authorization_file_sha256", "attempt_id", "attempt_directory",
        "program_contract_sha256", "prior_program_contract_sha256",
        "prior_program_contract_file_sha256", "intermediate_program_contract_sha256",
        "intermediate_program_contract_file_sha256",
        "base_program_contract_sha256", "base_program_contract_file_sha256",
        "local_readiness_receipt_sha256", "declaration_sha256", "issued_at_utc",
        "expires_at_utc", "consumed_at_utc",
    }
    _require(set(record) == exact_fields, "authorization consumption field set drifted", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    _require(record.get("artifact_type") == "JOB50_VENDOR_AUTHORIZATION_CONSUMPTION_V1", "wrong authorization consumption artifact", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    _require(record.get("schema_version") == "v5.job50-vendor-authorization-consumption.v1", "wrong authorization consumption schema", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    auth_sha = authorization.get("authorization_sha256")
    expected_marker = Path(repo_root).resolve() / "v5/work/cmbp-metadata-census/authorization-consumptions" / f"{auth_sha}.json"
    _require(Path(record_path).resolve() == expected_marker, "authorization consumption path drifted", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    _require((Path(record_path).stat().st_mode & 0o777) == 0o600, "authorization consumption mode drifted", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    expected_attempt_relative = f"v5/work/cmbp-metadata-census/external-attempts/{authorization['attempt_id']}"
    expected = {
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": auth_sha,
        "authorization_file_sha256": file_sha256(authorization_path),
        "attempt_id": authorization["attempt_id"],
        "attempt_directory": expected_attempt_relative,
        "program_contract_sha256": PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_sha256": PRIOR_PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_file_sha256": PRIOR_PROGRAM_CONTRACT_FILE_SHA256,
        "intermediate_program_contract_sha256": INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_file_sha256": INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256,
        "base_program_contract_sha256": BASE_PROGRAM_CONTRACT_SHA256,
        "base_program_contract_file_sha256": BASE_PROGRAM_CONTRACT_FILE_SHA256,
        "local_readiness_receipt_sha256": local_readiness_receipt["receipt_sha256"],
        "declaration_sha256": authorization["declaration_sha256"],
        "issued_at_utc": authorization["issued_at_utc"],
        "expires_at_utc": authorization["expires_at_utc"],
    }
    for field, value in expected.items():
        _require(record.get(field) == value, f"authorization consumption {field} drifted", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    _require(isinstance(record.get("consumed_at_utc"), str) and str(record["consumed_at_utc"]).endswith("Z"), "authorization consumption timestamp drifted", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    attempt_directory = Path(repo_root).resolve() / expected_attempt_relative
    _require(attempt_directory.is_dir(), "authorization-bound attempt directory is missing", status="STOP_AUTHORIZATION_MISSING_OR_INVALID")
    return attempt_directory


_AUTHORIZED_CAPABILITY_FIELDS = frozenset(
    {
        "authorization_id",
        "authorization_sha256",
        "authorization_file_sha256",
        "attempt_id",
        "attempt_directory",
        "consumption_record_path",
        "consumption_marker_sha256",
        "local_readiness_receipt_sha256",
        "local_readiness_receipt_file_sha256",
        "base_program_contract_sha256",
        "intermediate_program_contract_sha256",
        "prior_program_contract_sha256",
        "program_contract_sha256",
        "declaration_sha256",
        "declaration_file_sha256",
        "sdk_identity_sha256",
        "quoted_acquisition_ceiling_usd",
        "repo_root",
        "declaration_path",
        "program_contract_path",
        "prior_program_contract_path",
        "intermediate_program_contract_path",
        "base_program_contract_path",
        "local_readiness_receipt_path",
        "readiness_test_report_path",
        "synthetic_journal_path",
        "synthetic_response_path",
        "authorization_path",
    }
)


class _MetadataOnlyClientAdapter:
    """Narrow dispatch surface; it retains no unrestricted root-client handle."""

    class _Metadata:
        __slots__ = ("get_dataset_range", "get_cost", "get_record_count")

        def __init__(self, client: Any) -> None:
            self.get_dataset_range = client.metadata.get_dataset_range
            self.get_cost = client.metadata.get_cost
            self.get_record_count = client.metadata.get_record_count

    class _Symbology:
        __slots__ = ("resolve",)

        def __init__(self, client: Any) -> None:
            self.resolve = client.symbology.resolve

    __slots__ = ("metadata", "symbology")

    def __init__(self, client: Any) -> None:
        try:
            self.metadata = self._Metadata(client)
            self.symbology = self._Symbology(client)
        except AttributeError as exc:
            raise MetadataCensusError(
                "external client lacks the exact four-method surface",
                status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
            ) from exc
        for method in (
            self.metadata.get_dataset_range,
            self.metadata.get_cost,
            self.metadata.get_record_count,
            self.symbology.resolve,
        ):
            _require(
                callable(method),
                "external client four-method surface is not callable",
                status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
            )


def _bind_authorized_capability_type() -> tuple[type[Any], Callable[..., Any], Callable[..., Any]]:
    """Create production and refusal-only capability mints with lexical state."""

    construction_seal = object()
    production_usable = object()
    refusal_only = object()

    class AuthorizedExternalClient:
        """Opaque exact-class capability; raw client handles are never exposed."""

        __slots__ = (
            "__adapter",
            "__bindings",
            "__construction_seal",
            "__production_state",
        )

        def __init__(
            self,
            client: Any,
            bindings: Mapping[str, Any],
            *,
            _construction_seal: object,
            _production_state: object,
        ) -> None:
            if _construction_seal is not construction_seal:
                raise MetadataCensusError(
                    "AuthorizedExternalClient cannot be caller-constructed",
                    status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
                )
            _require(
                _production_state in {production_usable, refusal_only},
                "authorized-client provenance state drifted",
                status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
            )
            _require(
                set(bindings) == _AUTHORIZED_CAPABILITY_FIELDS,
                "authorized-client capability binding set drifted",
                status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
            )
            self.__adapter = _MetadataOnlyClientAdapter(client)
            self.__bindings = MappingProxyType(dict(bindings))
            self.__construction_seal = _construction_seal
            self.__production_state = _production_state

        def __repr__(self) -> str:
            return "AuthorizedExternalClient(<opaque>)"

        def __reduce__(self) -> Any:
            raise TypeError("AuthorizedExternalClient is not serializable")

        def __copy__(self) -> Any:
            raise TypeError("AuthorizedExternalClient is not copyable")

        def __deepcopy__(self, memo: Any) -> Any:
            del memo
            raise TypeError("AuthorizedExternalClient is not copyable")

        def _dispatch_surface_for_core(
            self,
        ) -> tuple[_MetadataOnlyClientAdapter, Mapping[str, Any]]:
            if self.__construction_seal is not construction_seal:
                raise MetadataCensusError(
                    "authorized-client capability construction drifted",
                    status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
                )
            return self.__adapter, self.__bindings

        def _bindings_for_receipt(self) -> Mapping[str, Any]:
            if self.__construction_seal is not construction_seal:
                raise MetadataCensusError(
                    "authorized-client capability construction drifted",
                    status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
                )
            return self.__bindings

        def _require_production_usable(self) -> None:
            if (
                self.__construction_seal is not construction_seal
                or self.__production_state is not production_usable
            ):
                raise MetadataCensusError(
                    "non-production capability cannot emit external evidence",
                    status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
                )

    def production_mint(
        client: Any,
        *,
        bindings: Mapping[str, Any],
    ) -> AuthorizedExternalClient:
        """Mint only an already-loaded exact pinned Historical client."""

        client_module = sys.modules.get("databento.historical.client")
        metadata_module = sys.modules.get("databento.historical.api.metadata")
        symbology_module = sys.modules.get("databento.historical.api.symbology")
        historical = getattr(client_module, "Historical", None)
        metadata_type = getattr(metadata_module, "MetadataHttpAPI", None)
        symbology_type = getattr(symbology_module, "SymbologyHttpAPI", None)
        _require(
            isinstance(historical, type)
            and isinstance(metadata_type, type)
            and isinstance(symbology_type, type)
            and historical.__module__ == "databento.historical.client"
            and historical.__qualname__ == "Historical"
            and type(client) is historical
            and type(getattr(client, "metadata", None)) is metadata_type
            and type(getattr(client, "symbology", None)) is symbology_type,
            "production capability requires the exact already-loaded pinned client",
            status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
        )
        for bound_method, endpoint_type, method_name in (
            (client.metadata.get_dataset_range, metadata_type, "get_dataset_range"),
            (client.metadata.get_cost, metadata_type, "get_cost"),
            (client.metadata.get_record_count, metadata_type, "get_record_count"),
            (client.symbology.resolve, symbology_type, "resolve"),
        ):
            _require(
                getattr(bound_method, "__self__", None)
                is (
                    client.symbology
                    if method_name == "resolve"
                    else client.metadata
                )
                and getattr(bound_method, "__func__", None)
                is getattr(endpoint_type, method_name, None),
                "production endpoint method provenance drifted",
                status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
            )
        return AuthorizedExternalClient(
            client,
            bindings,
            _construction_seal=construction_seal,
            _production_state=production_usable,
        )

    def refusal_test_mint(
        client: Any,
        *,
        bindings: Mapping[str, Any],
    ) -> AuthorizedExternalClient:
        """Exact-class negative fixture that can never emit external evidence."""

        return AuthorizedExternalClient(
            client,
            bindings,
            _construction_seal=construction_seal,
            _production_state=refusal_only,
        )

    AuthorizedExternalClient.__qualname__ = "AuthorizedExternalClient"
    production_mint.__name__ = "_mint_authorized_external_client"
    refusal_test_mint.__name__ = (
        "_mint_nonproduction_authorized_client_for_refusal_test"
    )
    return AuthorizedExternalClient, production_mint, refusal_test_mint


(
    AuthorizedExternalClient,
    _mint_authorized_external_client,
    _mint_nonproduction_authorized_client_for_refusal_test,
) = _bind_authorized_capability_type()
del _bind_authorized_capability_type


def _readiness_sdk_identity_sha256(receipt: Mapping[str, Any]) -> str:
    bindings = receipt.get("bindings")
    _require(
        isinstance(bindings, Mapping),
        "local readiness receipt bindings are missing",
        status="STOP_LOCAL_READINESS_RECEIPT_INVALID",
    )
    identity = bindings.get("sdk_source_identity")
    _require(
        isinstance(identity, Mapping),
        "local readiness SDK identity is missing",
        status="STOP_LOCAL_READINESS_RECEIPT_INVALID",
    )
    claimed = _require_sha(identity.get("identity_sha256"), "local readiness SDK identity")
    semantic = dict(identity)
    semantic.pop("identity_sha256")
    _require(
        claimed == json_sha256(semantic),
        "local readiness SDK identity aggregate drifted",
        status="STOP_LOCAL_READINESS_RECEIPT_INVALID",
    )
    return claimed


def sdk_parameters(request: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a frozen descriptor, then adapt only the SDK's symbology names."""

    _require(isinstance(request, Mapping), "request descriptor is not an object")
    method = request.get("method")
    _require(method in METHOD_ORDER, f"method is not allowlisted: {method!r}")
    parameters = request.get("parameters")
    _require(isinstance(parameters, Mapping), f"{method}: parameters are not an object")
    descriptor = {"method": method, "parameters": dict(parameters)}
    expected_hash = json_sha256(descriptor)
    _require(request.get("request_sha256") == expected_hash, f"{method}: frozen request hash mismatch")
    adapted = dict(parameters)
    expected_keys = {
        "metadata.get_dataset_range": {"dataset"},
        "symbology.resolve": {"dataset", "symbols", "stype_in", "stype_out", "start", "end"},
        "metadata.get_cost": {"dataset", "schema", "symbols", "stype_in", "start", "end"},
        "metadata.get_record_count": {"dataset", "schema", "symbols", "stype_in", "start", "end"},
    }[str(method)]
    _require(set(adapted) == expected_keys, f"{method}: forbidden argument or request shape drift")
    _require(adapted.get("dataset") == catalogue.DATASET, f"{method}: dataset scope widened")
    if method != "metadata.get_dataset_range":
        _require(adapted.get("stype_in") == catalogue.STYPE_IN, f"{method}: stype_in scope widened")
        symbols = adapted.get("symbols")
        _require(isinstance(symbols, list) and bool(symbols), f"{method}: symbol scope is missing")
        session = str(adapted.get("start"))[:10]
        for symbol in symbols:
            try:
                catalogue.validate_raw_spxw_osi(symbol, session)
            except catalogue.CataloguePreflightError as exc:
                raise MetadataCensusError(f"{method}: symbol scope is not exact SPXW 0DTE") from exc
    if method in {"metadata.get_cost", "metadata.get_record_count"}:
        _require(adapted.get("schema") == catalogue.SCHEMA, f"{method}: schema scope widened")
    if method == "symbology.resolve":
        _require(adapted.get("stype_out") == "instrument_id", "symbology.resolve: output scope widened")
    if method == "symbology.resolve":
        _require("start" in adapted and "end" in adapted, "symbology descriptor lacks frozen start/end")
        _require("start_date" not in adapted and "end_date" not in adapted, "symbology descriptor already contains SDK-only date names")
        adapted["start_date"] = adapted.pop("start")
        adapted["end_date"] = adapted.pop("end")
    return adapted


def _iso_day(value: Any, name: str) -> str:
    _require(isinstance(value, str), f"{name} is not text", status=catalogue.STOP_SCHEMA_UNAVAILABLE)
    text = value[:10]
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise MetadataCensusError(f"{name} is not an ISO date", status=catalogue.STOP_SCHEMA_UNAVAILABLE) from exc
    _require(parsed.isoformat() == text, f"{name} is not canonical", status=catalogue.STOP_SCHEMA_UNAVAILABLE)
    return text


def _normalize_dataset_range(raw: Any) -> dict[str, str]:
    _require(isinstance(raw, Mapping), "dataset-range SDK result is not an object", status=catalogue.STOP_MISSING_KEY_OR_ENTITLEMENT)
    start = raw.get("start", raw.get("dataset_start"))
    end = raw.get("end", raw.get("dataset_end"))
    start_day = _iso_day(start, "dataset_start")
    end_day = _iso_day(end, "dataset_end")
    _require(start_day < end_day, "dataset range is not increasing", status=catalogue.STOP_SCHEMA_UNAVAILABLE)
    return {"dataset_start": start_day, "dataset_end": end_day}


def _flatten_unresolved(value: Any, name: str) -> list[str]:
    if value in (None, [], {}):
        return []
    if isinstance(value, Mapping):
        items = list(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        raise MetadataCensusError(f"symbology {name} is not a collection", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)
    _require(all(isinstance(item, str) for item in items), f"symbology {name} contains a non-symbol", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)
    return sorted(set(items))


def _normalize_symbology(raw: Any, *, session: str, symbols: Sequence[str]) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), f"{session}: symbology SDK result is not an object", status=catalogue.STOP_MISSING_KEY_OR_ENTITLEMENT)
    if "result" in raw:
        result = raw.get("result")
        partial = _flatten_unresolved(raw.get("partial"), "partial")
        not_found = _flatten_unresolved(raw.get("not_found"), "not_found")
    else:
        result = raw
        partial = []
        not_found = []
    _require(isinstance(result, Mapping), f"{session}: symbology result mapping is missing", status=catalogue.STOP_MISSING_KEY_OR_ENTITLEMENT)
    _require(set(result) == set(symbols), f"{session}: resolved symbol set is incomplete or widened", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)
    unresolved = sorted(set(partial + not_found))
    _require(not unresolved, f"{session}: unresolved or partial symbols: {unresolved[:3]}", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)

    normalized: dict[str, list[str | int]] = {}
    for symbol in symbols:
        value = result[symbol]
        entries = value if isinstance(value, list) else [value]
        identifiers: list[str | int] = []
        for entry in entries:
            identifier: Any
            if isinstance(entry, Mapping):
                _require("s" in entry, f"{session}: interval mapping for {symbol!r} lacks 's'", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)
                identifier = entry["s"]
                if "d0" in entry:
                    _require(_iso_day(entry["d0"], f"{session}.{symbol}.d0") <= session, f"{session}: resolution starts after session", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)
                if "d1" in entry:
                    _require(_iso_day(entry["d1"], f"{session}.{symbol}.d1") > session, f"{session}: resolution ends before session", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)
            else:
                identifier = entry
            _require(
                (isinstance(identifier, int) and not isinstance(identifier, bool))
                or (isinstance(identifier, str) and bool(identifier)),
                f"{session}: invalid instrument identifier for {symbol!r}",
                status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH,
            )
            identifiers.append(identifier)
        _require(bool(identifiers), f"{session}: no identifiers for {symbol!r}", status=catalogue.STOP_SYMBOL_OR_EXPIRY_MISMATCH)
        normalized[symbol] = sorted(set(identifiers), key=lambda item: (type(item).__name__, str(item)))
    return {"resolved_symbols": normalized, "unresolved_symbols": []}


def _normalize_record_count(raw: Any, session: str) -> int:
    if isinstance(raw, bool):
        raise MetadataCensusError(f"{session}: record count is boolean", status=catalogue.STOP_ZERO_RECORDS)
    if isinstance(raw, int):
        count = raw
    elif isinstance(raw, str) and raw.isdigit():
        count = int(raw)
    else:
        raise MetadataCensusError(f"{session}: record count is not an integer", status=catalogue.STOP_ZERO_RECORDS)
    _require(count > 0, f"{session}: record count is zero", status=catalogue.STOP_ZERO_RECORDS)
    return count


class _CallJournal:
    def __init__(
        self,
        path: Path,
        *,
        attempt_id: str,
        declaration: Mapping[str, Any],
        contract_sha256: str,
        source: str,
        sdk_version: str,
        authorization_sha256: str | None,
        clock: Callable[[], str],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise MetadataCensusError(f"call journal must be a new exclusive file: {self.path}: {exc}") from exc
        os.fchmod(descriptor, 0o600)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._handle = os.fdopen(descriptor, "wb", buffering=0)
        self.sequence = -1
        self.head = GENESIS_HASH
        self.last_record_type: str | None = None
        self.clock = clock
        self.append(
            {
                "record_type": "ATTEMPT_HEADER",
                "attempt_id": attempt_id,
                "declaration_sha256": declaration["declaration_sha256"],
                "request_manifest_sha256": declaration["request_manifest_sha256"],
                "contract_sha256": contract_sha256,
                "response_source": source,
                "sdk_package": EXPECTED_SDK_PACKAGE,
                "sdk_version": sdk_version,
                "expected_call_count": EXPECTED_CALL_COUNT,
                "authorized_methods": list(METHOD_ORDER),
                "authorization_sha256": authorization_sha256,
                "resume_allowed": False,
                "automatic_retries_allowed": False,
            }
        )

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        record = {
            "artifact_type": CALL_JOURNAL_ARTIFACT,
            "schema_version": CALL_JOURNAL_SCHEMA_VERSION,
            "sequence": self.sequence,
            "recorded_at_utc": self.clock(),
            "previous_hash": self.head,
            **dict(payload),
        }
        record["record_hash"] = json_sha256(record)
        raw = canonical_json_bytes(record) + b"\n"
        _write_all(self._handle.fileno(), raw)
        os.fsync(self._handle.fileno())
        self.head = record["record_hash"]
        self.last_record_type = str(record.get("record_type"))
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def _write_all(descriptor: int, raw: bytes) -> None:
    """Write every byte or fail; short writes are never promoted to success."""

    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(descriptor, view[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError(f"short durable write stopped at {offset} of {len(view)} bytes")
        offset += written


def _write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    mode: int = 0o644,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, mode)
    except OSError as exc:
        raise MetadataCensusError(f"output must be a new exclusive file: {target}: {exc}") from exc
    try:
        raw = canonical_json_bytes(value) + b"\n"
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        reread = _load_canonical_json_output(target)
        _require(
            reread == dict(value),
            f"runner-owned JSON artifact changed during durable reread: {target}",
            status="STOP_RESPONSE_OR_RECEIPT_INVALID",
        )
        return reread
    except Exception:
        # The exclusive artifact may remain visibly incomplete; never overwrite
        # or resume it as a completed V1 attempt.
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _invoke(client: Any, method: str, parameters: Mapping[str, Any]) -> Any:
    """Dispatch over the exact allowlist without exposing a generic call path."""

    if method == "metadata.get_dataset_range":
        return client.metadata.get_dataset_range(**parameters)
    if method == "symbology.resolve":
        return client.symbology.resolve(**parameters)
    if method == "metadata.get_cost":
        return client.metadata.get_cost(**parameters)
    if method == "metadata.get_record_count":
        return client.metadata.get_record_count(**parameters)
    raise MetadataCensusError(f"unreachable non-allowlisted method: {method!r}")


def _response_core(response: Mapping[str, Any]) -> dict[str, Any]:
    core = json.loads(canonical_json_bytes(response))
    audit = core.get("execution_audit")
    if isinstance(audit, dict):
        audit.pop("call_journal", None)
    return core


def _execute_metadata_census_engine(
    client: Any,
    declaration: Mapping[str, Any],
    *,
    call_journal_path: Path,
    response_path: Path,
    source: str,
    sdk_version: str = EXPECTED_SDK_VERSION,
    contract_sha256: str,
    authorization_sha256: str | None = None,
    quoted_acquisition_ceiling_usd: str = "0",
    clock: Callable[[], str] | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Closure-captured shared loop; deleted from module globals after binding."""

    _require(source in {"synthetic", "externally_supplied"}, "response source is invalid")
    _require(sdk_version == EXPECTED_SDK_VERSION, f"SDK version drifted: {sdk_version!r} != {EXPECTED_SDK_VERSION!r}")
    _require_sha(contract_sha256, "contract_sha256")
    _require(contract_sha256 == PROGRAM_CONTRACT_SHA256, "program contract identity drifted")
    if source == "externally_supplied":
        _require_sha(authorization_sha256, "authorization_sha256")
    else:
        _require(authorization_sha256 is None, "synthetic execution cannot claim vendor authorization")
    cap_text = _decimal_text(quoted_acquisition_ceiling_usd, "quoted_acquisition_ceiling_usd")
    _require(cap_text == quoted_acquisition_ceiling_usd, "quoted acquisition ceiling must be canonical decimal text")
    _require(Path(call_journal_path).resolve() != Path(response_path).resolve(), "journal and response paths must differ")
    _require(not Path(response_path).exists(), f"response output already exists: {response_path}")
    requests = _expected_requests(declaration)
    run_id = attempt_id or uuid.uuid4().hex
    _require(isinstance(run_id, str) and bool(run_id) and "/" not in run_id, "attempt_id is invalid")
    now = clock or _utc_now

    journal = _CallJournal(
        call_journal_path,
        attempt_id=run_id,
        declaration=declaration,
        contract_sha256=contract_sha256,
        source=source,
        sdk_version=sdk_version,
        authorization_sha256=authorization_sha256,
        clock=now,
    )
    call_order: list[dict[str, Any]] = []
    normalized_by_call: list[dict[str, Any]] = []
    raw_results_by_call: list[dict[str, Any]] = []
    dataset_range: dict[str, Any] | None = None
    sessions: dict[str, dict[str, Any]] = {}
    method_counts: Counter[str] = Counter()
    try:
        for call_index, item in enumerate(requests, start=1):
            session = item["session"]
            request = item["request"]
            method = request["method"]
            parameters = sdk_parameters(request)
            order_row = {
                "call_index": call_index,
                "method": method,
                "session": session,
                "request_sha256": request["request_sha256"],
            }
            call_order.append(order_row)
            journal.append(
                {
                    "record_type": "CALL_START",
                    "attempt_id": run_id,
                    "call_ordinal": call_index,
                    "method": method,
                    "session": session,
                    "request_sha256": request["request_sha256"],
                    "sdk_parameters_sha256": json_sha256(parameters),
                    "retry_number": 0,
                }
            )
            try:
                raw = _invoke(client, method, parameters)
                raw_result_sha256 = json_sha256(raw)
                if method == "metadata.get_dataset_range":
                    normalized = _normalize_dataset_range(raw)
                elif method == "symbology.resolve":
                    declared_row = next(row for row in declaration["sessions"] if row["session"] == session)
                    normalized = _normalize_symbology(raw, session=session, symbols=declared_row["symbols"])
                elif method == "metadata.get_cost":
                    normalized = {"cost_usd": _decimal_text(raw, f"{session}.cost_usd")}
                else:
                    normalized = {"record_count": _normalize_record_count(raw, str(session))}
            except Exception as exc:
                journal.append(
                    {
                        "record_type": "CALL_ERROR",
                        "attempt_id": run_id,
                        "call_ordinal": call_index,
                        "method": method,
                        "session": session,
                        "request_sha256": request["request_sha256"],
                        "closed_error_class": (
                            exc.status if isinstance(exc, MetadataCensusError)
                            else "STOP_VENDOR_AUTH_OR_ENTITLEMENT"
                        ),
                        "retry_performed": False,
                    }
                )
                journal.append(
                    {
                        "record_type": "ATTEMPT_STOP",
                        "attempt_id": run_id,
                        "failed_call_ordinal": call_index,
                        "closed_error_class": (
                            exc.status if isinstance(exc, MetadataCensusError)
                            else "STOP_VENDOR_AUTH_OR_ENTITLEMENT"
                        ),
                        "success_response_emitted": False,
                        "no_resume": True,
                    }
                )
                if isinstance(exc, MetadataCensusError):
                    raise
                raise MetadataCensusError(
                    f"{method} failed at call {call_index}; see sealed journal",
                    status="STOP_VENDOR_AUTH_OR_ENTITLEMENT",
                ) from exc
            journal.append(
                {
                    "record_type": "CALL_RESULT",
                    "attempt_id": run_id,
                    "call_ordinal": call_index,
                    "method": method,
                    "session": session,
                    "request_sha256": request["request_sha256"],
                    "result_sha256": raw_result_sha256,
                    "raw_result_sha256": raw_result_sha256,
                    "normalized_result_sha256": json_sha256(normalized),
                    "retry_performed": False,
                }
            )
            method_counts[method] += 1
            normalized_by_call.append({**order_row, "normalized": normalized})
            raw_results_by_call.append({**order_row, "result_sha256": raw_result_sha256})

            if method == "metadata.get_dataset_range":
                dataset_range = {
                    "method": method,
                    "status": "OK",
                    "request_sha256": request["request_sha256"],
                    "raw_result_sha256": raw_result_sha256,
                    "raw_value": json.loads(canonical_json_bytes(raw)),
                    **normalized,
                }
                continue
            assert session is not None
            declared_row = next(row for row in declaration["sessions"] if row["session"] == session)
            row = sessions.setdefault(
                session,
                {
                    "session": session,
                    "status": "OK",
                    "schema": declaration["schema"],
                    "rth_open_utc": declared_row["rth_open_utc"],
                    "rth_close_utc": declared_row["rth_close_utc"],
                },
            )
            if method == "symbology.resolve":
                row["symbology_resolve_request_sha256"] = request["request_sha256"]
                row["symbology_resolve_raw_result_sha256"] = raw_result_sha256
                row.update(normalized)
            elif method == "metadata.get_cost":
                row["cost_request_sha256"] = request["request_sha256"]
                row["cost_raw_result_sha256"] = raw_result_sha256
                row.update(normalized)
            elif method == "metadata.get_record_count":
                row["record_count_request_sha256"] = request["request_sha256"]
                row["record_count_raw_result_sha256"] = raw_result_sha256
                row.update(normalized)

        _require(dataset_range is not None, "complete attempt lacks dataset range")
        _require(dict(method_counts) == EXPECTED_METHOD_COUNTS, f"actual method ledger drifted: {dict(method_counts)}")
        expected_sessions = [row["session"] for row in declaration["sessions"] if row["requests"]]
        _require(list(sessions) == expected_sessions, "response session order or completeness drifted")
        for session in expected_sessions:
            _require(
                set(sessions[session])
                == {
                    "session", "status", "schema", "rth_open_utc", "rth_close_utc",
                    "symbology_resolve_request_sha256", "resolved_symbols", "unresolved_symbols",
                    "symbology_resolve_raw_result_sha256", "cost_request_sha256", "cost_raw_result_sha256",
                    "cost_usd", "record_count_request_sha256", "record_count_raw_result_sha256", "record_count",
                },
                f"{session}: normalized response is incomplete",
            )

        response: dict[str, Any] = {
            "artifact_type": catalogue.RESPONSE_ARTIFACT,
            "artifact_version": catalogue.ARTIFACT_VERSION,
            "declaration_sha256": declaration["declaration_sha256"],
            "source": source,
            "method_attestation": {
                "methods_used": [] if source == "synthetic" else list(METHOD_ORDER),
                "request_shapes_exercised": list(METHOD_ORDER),
                "timeseries_calls": 0,
                "download_calls": 0,
            },
            "dataset_range": dataset_range,
            "sessions": [sessions[session] for session in expected_sessions],
            "execution_audit": {
                "artifact_type": EXECUTION_AUDIT_ARTIFACT,
                "artifact_version": ARTIFACT_VERSION,
                "attempt_id": run_id,
                "contract_sha256": contract_sha256,
                "authorization_sha256": authorization_sha256,
                "declaration_sha256": declaration["declaration_sha256"],
                "request_manifest_sha256": declaration["request_manifest_sha256"],
                "sdk_package": EXPECTED_SDK_PACKAGE,
                "sdk_version": sdk_version,
                "response_source": source,
                "expected_call_count": EXPECTED_CALL_COUNT,
                "completed_call_count": len(normalized_by_call),
                "method_counts": dict(method_counts),
                "call_order_sha256": json_sha256(call_order),
                "raw_results_sha256": json_sha256(raw_results_by_call),
                "normalized_results_sha256": json_sha256(normalized_by_call),
                "quoted_acquisition_ceiling_usd": cap_text,
                "automatic_retries": 0,
                "timeseries_calls": 0,
                "download_calls": 0,
                "data_downloaded": False,
                "outcomes_read": False,
                "authorization_effect": "NONE" if source == "synthetic" else AUTHORIZATION_EFFECT,
            },
        }
        total_cost = sum(Decimal(row["cost_usd"]) for row in response["sessions"])
        structural_receipt = catalogue.build_preflight_receipt(
            declaration,
            response,
            hard_cap_usd=total_cost,
        )
        if structural_receipt.get("status") != catalogue.STATUS_PREFLIGHT_PASS_ONLY:
            stopped_status = normalize_job50_status(
                structural_receipt.get("status", "STOP_RESPONSE_OR_RECEIPT_INVALID")
            )
            journal.append(
                {
                    "record_type": "ATTEMPT_STOP",
                    "attempt_id": run_id,
                    "closed_error_class": stopped_status,
                    "success_response_emitted": False,
                    "no_resume": True,
                }
            )
            raise MetadataCensusError(
                f"completed SDK responses fail the frozen Job-49 parser: {stopped_status}",
                status=stopped_status,
            )
        if source == "externally_supplied" and total_cost > Decimal(cap_text):
            journal.append(
                {
                    "record_type": "ATTEMPT_STOP",
                    "attempt_id": run_id,
                    "closed_error_class": "STOP_OVER_QUOTED_CEILING",
                    "exact_total_cost_usd": catalogue.decimal_text(total_cost),
                    "quoted_acquisition_ceiling_usd": cap_text,
                    "success_response_emitted": False,
                    "no_resume": True,
                }
            )
            raise MetadataCensusError(
                f"exact quote {catalogue.decimal_text(total_cost)} exceeds frozen ceiling {cap_text}",
                status="STOP_OVER_QUOTED_CEILING",
            )
        core_sha = json_sha256(_response_core(response))
        terminal = journal.append(
            {
                "record_type": "ATTEMPT_COMPLETE",
                "attempt_id": run_id,
                "completed_call_count": len(normalized_by_call),
                "method_counts": dict(method_counts),
                "call_order_sha256": json_sha256(call_order),
                "raw_results_sha256": json_sha256(raw_results_by_call),
                "normalized_results_sha256": json_sha256(normalized_by_call),
                "response_core_sha256": core_sha,
                "no_resume": True,
            }
        )
    except Exception as exc:
        if journal.last_record_type not in {"ATTEMPT_STOP", "ATTEMPT_COMPLETE"}:
            stopped_status = (
                exc.status if isinstance(exc, MetadataCensusError)
                else "STOP_PARTIAL_OR_CRASHED_ATTEMPT"
            )
            try:
                journal.append(
                    {
                        "record_type": "ATTEMPT_STOP",
                        "attempt_id": run_id,
                        "closed_error_class": stopped_status,
                        "success_response_emitted": False,
                        "no_resume": True,
                    }
                )
            except OSError:
                # The earlier fsynced prefix remains the authoritative evidence
                # when the storage failure also prevents a terminal record.
                pass
        raise
    finally:
        journal.close()

    response["execution_audit"]["call_journal"] = {
        "artifact_type": CALL_JOURNAL_ARTIFACT,
        "file_name": Path(call_journal_path).name,
        "file_sha256": file_sha256(call_journal_path),
        "terminal_sequence": terminal["sequence"],
        "terminal_head": terminal["record_hash"],
        "response_core_sha256": core_sha,
    }
    validate_execution_response(response, declaration, call_journal_path=call_journal_path)
    reread_response = _write_json_exclusive(response_path, response)
    validate_execution_response(
        reread_response,
        declaration,
        call_journal_path=call_journal_path,
    )
    return reread_response


def _read_journal(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise MetadataCensusError(f"cannot read call journal {path}: {exc}") from exc
    _require(raw.endswith(b"\n"), "call journal lacks a final newline")
    records: list[dict[str, Any]] = []
    previous = GENESIS_HASH
    for index, line in enumerate(raw.splitlines()):
        _require(bool(line), f"call journal has empty record {index}")
        record = _strict_json_bytes(line, origin=f"call journal record {index}")
        _require(isinstance(record, dict), f"call journal record {index} is not an object")
        _require(line == canonical_json_bytes(record), f"call journal record {index} is not canonical JSON")
        _require(record.get("artifact_type") == CALL_JOURNAL_ARTIFACT, f"call journal record {index} artifact drifted")
        _require(record.get("schema_version") == CALL_JOURNAL_SCHEMA_VERSION, f"call journal record {index} version drifted")
        _require(record.get("sequence") == index, f"call journal sequence drifted at {index}")
        _require(record.get("previous_hash") == previous, f"call journal chain broke at {index}")
        claimed = _require_sha(record.get("record_hash"), f"journal record {index} hash")
        semantic = dict(record)
        semantic.pop("record_hash")
        _require(claimed == json_sha256(semantic), f"call journal record {index} hash mismatch")
        previous = claimed
        records.append(record)
    return records, raw


def validate_execution_response(
    response: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    call_journal_path: Path,
) -> None:
    """Verify completeness, call order, result bindings, and the journal tail."""

    requests = _expected_requests(declaration)
    _require(isinstance(response, Mapping), "execution response is not an object")
    audit = response.get("execution_audit")
    _require(isinstance(audit, Mapping), "execution audit is missing")
    _require(audit.get("artifact_type") == EXECUTION_AUDIT_ARTIFACT, "execution audit artifact drifted")
    _require(audit.get("artifact_version") == ARTIFACT_VERSION, "execution audit version drifted")
    _require(audit.get("declaration_sha256") == declaration["declaration_sha256"], "execution audit declaration link drifted")
    _require(audit.get("request_manifest_sha256") == declaration["request_manifest_sha256"], "execution audit request link drifted")
    _require(audit.get("sdk_package") == EXPECTED_SDK_PACKAGE and audit.get("sdk_version") == EXPECTED_SDK_VERSION, "execution audit SDK binding drifted")
    _require(audit.get("expected_call_count") == EXPECTED_CALL_COUNT, "expected call count drifted")
    _require(audit.get("completed_call_count") == EXPECTED_CALL_COUNT, "attempt is incomplete")
    _require(audit.get("method_counts") == EXPECTED_METHOD_COUNTS, "method count ledger drifted")
    _require(audit.get("automatic_retries") == 0, "attempt used retries")
    _require(audit.get("timeseries_calls") == 0 and audit.get("download_calls") == 0, "forbidden call family recorded")
    _require(audit.get("data_downloaded") is False and audit.get("outcomes_read") is False, "execution audit widened scope")

    expected_order = [
        {"call_index": index, "method": item["request"]["method"], "session": item["session"], "request_sha256": item["request"]["request_sha256"]}
        for index, item in enumerate(requests, start=1)
    ]
    _require(audit.get("call_order_sha256") == json_sha256(expected_order), "call-order hash drifted")
    journal_binding = audit.get("call_journal")
    _require(isinstance(journal_binding, Mapping), "call-journal binding is missing")
    _require(journal_binding.get("artifact_type") == CALL_JOURNAL_ARTIFACT, "call-journal artifact binding drifted")
    _require(journal_binding.get("file_name") == Path(call_journal_path).name, "call-journal filename binding drifted")
    _require(journal_binding.get("file_sha256") == file_sha256(call_journal_path), "call-journal file hash mismatch")

    records, _ = _read_journal(call_journal_path)
    _require(len(records) == 2 * EXPECTED_CALL_COUNT + 2, "call journal record count is incomplete")
    header = records[0]
    terminal = records[-1]
    _require(header.get("record_type") == "ATTEMPT_HEADER", "call journal header missing")
    _require(header.get("attempt_id") == audit.get("attempt_id"), "journal attempt identity drifted")
    _require(header.get("declaration_sha256") == declaration["declaration_sha256"], "journal declaration link drifted")
    _require(header.get("contract_sha256") == audit.get("contract_sha256"), "journal contract link drifted")
    _require(header.get("authorization_sha256") == audit.get("authorization_sha256"), "journal authorization link drifted")
    _require(header.get("expected_call_count") == EXPECTED_CALL_COUNT, "journal expected-call count drifted")
    _require(header.get("authorized_methods") == list(METHOD_ORDER), "journal allowlist drifted")
    _require(header.get("resume_allowed") is False and header.get("automatic_retries_allowed") is False, "journal permits resume or retries")

    normalized_results: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    response_sessions = {row["session"]: row for row in response.get("sessions", []) if isinstance(row, Mapping) and isinstance(row.get("session"), str)}
    _require(len(response_sessions) == declaration["request_session_count"], "response session set is incomplete")
    for call_index, (item, expected_order_row) in enumerate(zip(requests, expected_order, strict=True), start=1):
        start = records[2 * call_index - 1]
        result = records[2 * call_index]
        _require(start.get("record_type") == "CALL_START", f"call {call_index} start record missing")
        _require(result.get("record_type") == "CALL_RESULT", f"call {call_index} result record missing")
        _require(start.get("call_ordinal") == call_index and result.get("call_ordinal") == call_index, f"call {call_index} ordinal binding drifted")
        for field in ("method", "session", "request_sha256"):
            expected = expected_order_row[field]
            _require(start.get(field) == expected and result.get(field) == expected, f"call {call_index} {field} binding drifted")
        _require(start.get("attempt_id") == audit.get("attempt_id") and result.get("attempt_id") == audit.get("attempt_id"), f"call {call_index} attempt binding drifted")
        _require(start.get("sdk_parameters_sha256") == json_sha256(sdk_parameters(item["request"])), f"call {call_index} SDK adapter hash drifted")
        method = expected_order_row["method"]
        session = expected_order_row["session"]
        if method == "metadata.get_dataset_range":
            range_row = response.get("dataset_range")
            _require(isinstance(range_row, Mapping), "response dataset range is missing")
            normalized = {"dataset_start": range_row.get("dataset_start"), "dataset_end": range_row.get("dataset_end")}
            raw_result_sha256 = range_row.get("raw_result_sha256")
            _require(
                json_sha256(range_row.get("raw_value")) == raw_result_sha256,
                "dataset-range raw value/digest binding drifted",
            )
        else:
            _require(session in response_sessions, f"{session}: normalized response missing")
            row = response_sessions[session]
            if method == "symbology.resolve":
                normalized = {"resolved_symbols": row.get("resolved_symbols"), "unresolved_symbols": row.get("unresolved_symbols")}
                raw_result_sha256 = row.get("symbology_resolve_raw_result_sha256")
            elif method == "metadata.get_cost":
                normalized = {"cost_usd": row.get("cost_usd")}
                raw_result_sha256 = row.get("cost_raw_result_sha256")
            else:
                normalized = {"record_count": row.get("record_count")}
                raw_result_sha256 = row.get("record_count_raw_result_sha256")
        _require_sha(raw_result_sha256, f"call {call_index} raw result hash")
        _require(result.get("result_sha256") == raw_result_sha256, f"call {call_index} raw result hash does not bind response")
        _require(result.get("raw_result_sha256") == raw_result_sha256, f"call {call_index} explicit raw result hash drifted")
        _require(result.get("normalized_result_sha256") == json_sha256(normalized), f"call {call_index} normalized result hash does not bind response")
        raw_results.append({**expected_order_row, "result_sha256": raw_result_sha256})
        normalized_results.append({**expected_order_row, "normalized": normalized})

    _require(audit.get("raw_results_sha256") == json_sha256(raw_results), "raw-results hash drifted")
    _require(audit.get("normalized_results_sha256") == json_sha256(normalized_results), "normalized-results hash drifted")
    core_sha = json_sha256(_response_core(response))
    _require(terminal.get("record_type") == "ATTEMPT_COMPLETE", "call journal terminal completion missing")
    _require(terminal.get("response_core_sha256") == core_sha, "journal terminal does not bind response core")
    _require(terminal.get("completed_call_count") == EXPECTED_CALL_COUNT, "journal terminal is incomplete")
    _require(terminal.get("method_counts") == EXPECTED_METHOD_COUNTS, "journal terminal method counts drifted")
    _require(terminal.get("call_order_sha256") == audit.get("call_order_sha256"), "journal terminal call order drifted")
    _require(terminal.get("raw_results_sha256") == audit.get("raw_results_sha256"), "journal terminal raw-result binding drifted")
    _require(terminal.get("normalized_results_sha256") == audit.get("normalized_results_sha256"), "journal terminal result binding drifted")
    _require(journal_binding.get("terminal_sequence") == terminal["sequence"], "journal terminal sequence binding drifted")
    _require(journal_binding.get("terminal_head") == terminal["record_hash"], "journal terminal head binding drifted")
    _require(journal_binding.get("response_core_sha256") == core_sha, "response core binding drifted")

    structural_receipt = catalogue.build_preflight_receipt(
        declaration,
        response,
        hard_cap_usd=sum(Decimal(str(row["cost_usd"])) for row in response_sessions.values()),
    )
    _require(structural_receipt.get("status") == catalogue.STATUS_PREFLIGHT_PASS_ONLY, f"response fails frozen offline parser: {structural_receipt.get('status')}")


def classify_metadata_attempt(
    declaration: Mapping[str, Any],
    *,
    call_journal_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    """Classify retained on-disk evidence without repairing or resuming it.

    This is deliberately an offline diagnostic.  A terminal COMPLETE record
    is not a pass by itself: the canonical response must exist and reconstruct
    against the entire retained journal.  Every other tail is a named STOP.
    """

    try:
        records, _ = _read_journal(call_journal_path)
    except Exception as exc:
        return {
            "artifact_type": "JOB50_METADATA_ATTEMPT_CLASSIFICATION_V1",
            "status": "STOP_CALL_JOURNAL_INVALID",
            "complete": False,
            "success_response_valid": False,
            "diagnostic_detail": type(exc).__name__,
            "authorization_effect": "NONE",
        }
    if not records or records[0].get("record_type") != "ATTEMPT_HEADER":
        return {
            "artifact_type": "JOB50_METADATA_ATTEMPT_CLASSIFICATION_V1",
            "status": "STOP_CALL_JOURNAL_INVALID",
            "complete": False,
            "success_response_valid": False,
            "diagnostic_detail": "missing_attempt_header",
            "authorization_effect": "NONE",
        }
    header = records[0]
    terminal = records[-1]
    base = {
        "artifact_type": "JOB50_METADATA_ATTEMPT_CLASSIFICATION_V1",
        "attempt_id": header.get("attempt_id"),
        "terminal_sequence": terminal.get("sequence"),
        "terminal_head": terminal.get("record_hash"),
        "authorization_effect": "NONE",
    }
    terminal_type = terminal.get("record_type")
    if terminal_type == "ATTEMPT_STOP":
        return {
            **base,
            "status": normalize_job50_status(terminal.get("closed_error_class")),
            "complete": False,
            "success_response_valid": False,
        }
    if terminal_type != "ATTEMPT_COMPLETE":
        return {
            **base,
            "status": "STOP_PARTIAL_OR_CRASHED_ATTEMPT",
            "complete": False,
            "success_response_valid": False,
        }
    if not Path(response_path).is_file():
        return {
            **base,
            "status": "STOP_PARTIAL_OR_CRASHED_ATTEMPT",
            "complete": False,
            "success_response_valid": False,
        }
    try:
        response = _load_canonical_json_output(response_path)
        validate_execution_response(
            response,
            declaration,
            call_journal_path=call_journal_path,
        )
    except Exception as exc:
        return {
            **base,
            "status": "STOP_RESPONSE_OR_RECEIPT_INVALID",
            "complete": False,
            "success_response_valid": False,
            "diagnostic_detail": type(exc).__name__,
        }
    return {
        **base,
        "status": "JOB50_METADATA_RESPONSE_COMPLETE_ONLY",
        "complete": True,
        "success_response_valid": True,
        "response_source": response.get("source"),
        "response_file_sha256": file_sha256(response_path),
    }


EXTERNAL_RECEIPT_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "job_id",
        "status",
        "status_meaning",
        "next_state",
        "attempt_id",
        "program_contract_sha256",
        "prior_program_contract_sha256",
        "prior_program_contract_file_sha256",
        "intermediate_program_contract_sha256",
        "intermediate_program_contract_file_sha256",
        "base_program_contract_sha256",
        "base_program_contract_file_sha256",
        "local_readiness_receipt_sha256",
        "local_readiness_receipt_file_sha256",
        "authorization_sha256",
        "authorization_file_sha256",
        "authorization_consumption",
        "declaration_sha256",
        "declaration_file_sha256",
        "request_manifest_sha256",
        "response",
        "call_journal",
        "sdk_identity",
        "dataset_range",
        "request_session_count",
        "exact_total_record_count",
        "exact_total_cost_usd",
        "quoted_acquisition_ceiling_usd",
        "within_quoted_acquisition_ceiling",
        "method_counts",
        "nested_structural_parser_receipt",
        "claims",
        "integrity",
        "authorization_effect",
        "receipt_sha256",
    }
)


def _repository_relative(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    _require(
        resolved.is_relative_to(root),
        f"artifact path escapes repository: {resolved}",
        status="STOP_RESPONSE_OR_RECEIPT_INVALID",
    )
    return resolved.relative_to(root).as_posix()


def _external_receipt_context_engine(
    *,
    repo_root: Path,
    attempt_id: str,
    enforce_current_authorization_freshness: bool,
) -> dict[str, Any]:
    """Strictly reconstruct every retained input to one external wrapper."""

    from v5.research import cmbp_metadata_census_receipt_v2 as readiness

    root = Path(repo_root).resolve()
    work = root / "v5/work/cmbp-metadata-census"
    paths = {
        "program_contract": work / "PROGRAM_CONTRACT_V4.json",
        "prior_contract": work / "PROGRAM_CONTRACT_V3.json",
        "intermediate_contract": work / "PROGRAM_CONTRACT_V2.json",
        "base_contract": work / "PROGRAM_CONTRACT_V1.json",
        "declaration": root / "v5/work/human-policy-foundation/CMBP_CATALOGUE_DECLARATION_V1.json",
        "local_readiness_receipt": work / "LOCAL_READINESS_RECEIPT_V2.json",
        "readiness_test_report": work / "TEST_RESULTS_V2.xml",
        "synthetic_journal": work / "SYNTHETIC_CALL_JOURNAL_V2.jsonl",
        "synthetic_response": work / "SYNTHETIC_METADATA_RESPONSES_V2.json",
        "authorization": work / "authorizations" / attempt_id / "VENDOR_RUN_AUTHORIZATION_V2.json",
        "attempt_directory": work / "external-attempts" / attempt_id,
    }
    contract = load_json(paths["program_contract"])
    prior_contract = load_json(paths["prior_contract"])
    intermediate_contract = load_json(paths["intermediate_contract"])
    base_contract = load_json(paths["base_contract"])
    declaration = load_json(paths["declaration"])
    _contract_semantic_sha256(contract)
    _prior_contract_semantic_sha256(prior_contract)
    _intermediate_contract_semantic_sha256(intermediate_contract)
    _base_contract_semantic_sha256(base_contract)
    _require(file_sha256(paths["program_contract"]) == PROGRAM_CONTRACT_FILE_SHA256, "V4 contract raw-file drifted")
    _require(file_sha256(paths["prior_contract"]) == PRIOR_PROGRAM_CONTRACT_FILE_SHA256, "V3 contract raw-file drifted")
    _require(file_sha256(paths["intermediate_contract"]) == INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256, "V2 contract raw-file drifted")
    _require(file_sha256(paths["base_contract"]) == BASE_PROGRAM_CONTRACT_FILE_SHA256, "V1 contract raw-file drifted")
    try:
        local_receipt = readiness.strict_json(paths["local_readiness_receipt"])
        readiness.validate_local_readiness_receipt(
            local_receipt,
            repo_root=root,
            test_report_path=paths["readiness_test_report"],
            synthetic_journal_path=paths["synthetic_journal"],
            synthetic_response_path=paths["synthetic_response"],
            require_vendor_authorization_absent=False,
        )
    except Exception as exc:
        raise MetadataCensusError(
            "external receipt no longer binds a valid local-readiness receipt",
            status="STOP_LOCAL_READINESS_RECEIPT_INVALID",
        ) from exc

    authorization = load_json(paths["authorization"])
    cap = verify_vendor_run_authorization(
        authorization,
        declaration,
        contract,
        prior_contract=prior_contract,
        intermediate_contract=intermediate_contract,
        base_contract=base_contract,
        local_readiness_receipt=local_receipt,
        local_readiness_receipt_file_sha256=file_sha256(paths["local_readiness_receipt"]),
        authorization_path=paths["authorization"],
        repo_root=root,
        attempt_id=attempt_id,
        declaration_file_sha256=file_sha256(paths["declaration"]),
        enforce_current_freshness=enforce_current_authorization_freshness,
    )
    paths["authorization_consumption"] = (
        work
        / "authorization-consumptions"
        / f"{authorization['authorization_sha256']}.json"
    )
    consumption = load_json(paths["authorization_consumption"])
    verified_attempt = verify_authorization_consumption(
        consumption,
        record_path=paths["authorization_consumption"],
        authorization=authorization,
        authorization_path=paths["authorization"],
        repo_root=root,
        local_readiness_receipt=local_receipt,
    )
    _require(verified_attempt == paths["attempt_directory"], "external attempt path drifted")
    paths["journal"] = verified_attempt / "CALL_JOURNAL_V1.jsonl"
    paths["response"] = verified_attempt / "EXTERNAL_METADATA_RESPONSES_V1.json"
    paths["nested_receipt"] = verified_attempt / "NESTED_STRUCTURAL_PARSER_RECEIPT_V1.json"
    paths["external_receipt"] = verified_attempt / "EXTERNAL_METADATA_RECEIPT_V1.json"
    response = _load_canonical_json_output(paths["response"])
    validate_execution_response(response, declaration, call_journal_path=paths["journal"])
    audit = response.get("execution_audit")
    _require(response.get("source") == "externally_supplied", "external response source drifted")
    _require(isinstance(audit, Mapping), "external execution audit is missing")
    _require(audit.get("attempt_id") == attempt_id, "external response attempt identity drifted")
    _require(audit.get("authorization_sha256") == authorization["authorization_sha256"], "external response authorization link drifted")
    _require(Decimal(str(audit.get("quoted_acquisition_ceiling_usd"))) == cap, "external response ceiling drifted")
    return {
        "root": root,
        "paths": paths,
        "contract": contract,
        "prior_contract": prior_contract,
        "intermediate_contract": intermediate_contract,
        "base_contract": base_contract,
        "declaration": declaration,
        "local_receipt": local_receipt,
        "authorization": authorization,
        "consumption": consumption,
        "response": response,
        "audit": audit,
        "cap": cap,
    }


def _build_nested_structural_receipt_engine(
    context: Mapping[str, Any],
    *,
    write_new: bool,
) -> tuple[dict[str, Any], str]:
    declaration = context["declaration"]
    response = context["response"]
    cap = context["cap"]
    path = context["paths"]["nested_receipt"]
    expected = catalogue.build_preflight_receipt(
        declaration,
        response,
        hard_cap_usd=cap,
    )
    stopped = normalize_job50_status(expected.get("status"))
    _require(
        expected.get("status") == catalogue.STATUS_PREFLIGHT_PASS_ONLY,
        f"nested structural parser stopped: {expected.get('status')}",
        status=(
            "STOP_OVER_QUOTED_CEILING"
            if expected.get("status") == catalogue.STOP_OVER_HARD_CAP
            else stopped
        ),
    )
    if write_new:
        nested = _write_json_exclusive(path, expected)
    else:
        nested = _load_canonical_json_output(path)
    catalogue.validate_preflight_receipt(nested, declaration=declaration)
    _require(nested == expected, "nested Job-49 parser receipt reconstruction drifted")
    _require(nested.get("job49_integration_disposition") == "VALIDATOR_REHEARSAL_ONLY", "nested Job-49 classification drifted")
    _require(nested.get("external_preflight_achieved_by_job49") is False, "nested receipt overclaims Job 49")
    claims = nested.get("claims")
    _require(isinstance(claims, Mapping) and claims.get("actual_vendor_availability") is False, "nested receipt overclaims vendor availability")
    return nested, file_sha256(path)


def _external_wrapper_from_context_engine(
    context: Mapping[str, Any],
    *,
    nested_receipt: Mapping[str, Any],
    nested_file_sha256: str,
) -> dict[str, Any]:
    root = context["root"]
    paths = context["paths"]
    declaration = context["declaration"]
    local_receipt = context["local_receipt"]
    authorization = context["authorization"]
    consumption = context["consumption"]
    response = context["response"]
    audit = context["audit"]
    cap = context["cap"]
    sdk_source_identity = local_receipt["bindings"]["sdk_source_identity"]
    total_cost = Decimal(str(nested_receipt["exact_total_cost_usd"]))
    within_ceiling = total_cost <= cap
    _require(within_ceiling, "nested receipt total exceeds quoted ceiling", status="STOP_OVER_QUOTED_CEILING")
    dataset_range = response["dataset_range"]
    wrapper: dict[str, Any] = {
        "artifact_type": "JOB50_EXTERNAL_METADATA_RECEIPT_V1",
        "schema_version": "v5.job50-external-metadata-receipt.v1",
        "job_id": 50,
        "status": "JOB50_EXTERNAL_METADATA_PREFLIGHT_PASS_ONLY",
        "status_meaning": (
            "The authenticated metadata census completed for exactly the sealed declaration. "
            "This is availability-and-quote evidence only and authorizes no acquisition or trade."
        ),
        "next_state": "BLOCKED_AWAITING_SEPARATE_ACQUISITION_DECISION",
        "attempt_id": authorization["attempt_id"],
        "program_contract_sha256": PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_sha256": PRIOR_PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_file_sha256": PRIOR_PROGRAM_CONTRACT_FILE_SHA256,
        "intermediate_program_contract_sha256": INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_file_sha256": INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256,
        "base_program_contract_sha256": BASE_PROGRAM_CONTRACT_SHA256,
        "base_program_contract_file_sha256": BASE_PROGRAM_CONTRACT_FILE_SHA256,
        "local_readiness_receipt_sha256": local_receipt["receipt_sha256"],
        "local_readiness_receipt_file_sha256": file_sha256(paths["local_readiness_receipt"]),
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": file_sha256(paths["authorization"]),
        "authorization_consumption": {
            "path": _repository_relative(paths["authorization_consumption"], root),
            "file_sha256": file_sha256(paths["authorization_consumption"]),
            "payload_sha256": json_sha256(consumption),
            "authorization_id": consumption["authorization_id"],
            "attempt_id": consumption["attempt_id"],
            "consumed_at_utc": consumption["consumed_at_utc"],
        },
        "declaration_sha256": declaration["declaration_sha256"],
        "declaration_file_sha256": file_sha256(paths["declaration"]),
        "request_manifest_sha256": declaration["request_manifest_sha256"],
        "response": {
            "artifact_type": response["artifact_type"],
            "path": _repository_relative(paths["response"], root),
            "semantic_sha256": json_sha256(response),
            "file_sha256": file_sha256(paths["response"]),
            "source": response["source"],
            "attempt_id": audit["attempt_id"],
            "dataset_range_raw_result_sha256": dataset_range["raw_result_sha256"],
        },
        "call_journal": {
            "artifact_type": CALL_JOURNAL_ARTIFACT,
            "schema_version": CALL_JOURNAL_SCHEMA_VERSION,
            "path": _repository_relative(paths["journal"], root),
            "file_sha256": audit["call_journal"]["file_sha256"],
            "terminal_sequence": audit["call_journal"]["terminal_sequence"],
            "terminal_head": audit["call_journal"]["terminal_head"],
            "completed_call_count": audit["completed_call_count"],
            "method_counts": audit["method_counts"],
            "call_order_sha256": audit["call_order_sha256"],
            "raw_results_sha256": audit["raw_results_sha256"],
            "normalized_results_sha256": audit["normalized_results_sha256"],
        },
        "sdk_identity": {
            "package": EXPECTED_SDK_PACKAGE,
            "version": EXPECTED_SDK_VERSION,
            "source_identity_sha256": _readiness_sdk_identity_sha256(local_receipt),
            "source_identity": sdk_source_identity,
        },
        "dataset_range": {
            "request_sha256": dataset_range["request_sha256"],
            "dataset_start": dataset_range["dataset_start"],
            "dataset_end": dataset_range["dataset_end"],
            "raw_result_sha256": dataset_range["raw_result_sha256"],
            "claim_boundary": "NORMALIZED_UTC_CALENDAR_DATES_NOT_SCHEMA_SPECIFIC",
        },
        "request_session_count": declaration["request_session_count"],
        "exact_total_record_count": nested_receipt["exact_total_record_count"],
        "exact_total_cost_usd": nested_receipt["exact_total_cost_usd"],
        "quoted_acquisition_ceiling_usd": authorization["quoted_acquisition_ceiling_usd"],
        "within_quoted_acquisition_ceiling": within_ceiling,
        "method_counts": audit["method_counts"],
        "nested_structural_parser_receipt": {
            "classification": "NESTED_JOB49_STRUCTURAL_VALIDATION_ONLY",
            "artifact_type": catalogue.RECEIPT_ARTIFACT,
            "status": catalogue.STATUS_PREFLIGHT_PASS_ONLY,
            "path": _repository_relative(paths["nested_receipt"], root),
            "receipt_sha256": nested_receipt["receipt_sha256"],
            "file_sha256": nested_file_sha256,
        },
        "claims": {
            "actual_vendor_availability": True,
            "actual_vendor_availability_scope": "EXACT_SEALED_DECLARATION_AND_AUTHENTICATED_ENTITLEMENT_AT_THIS_ATTEMPT_ONLY",
            "universal_or_population_availability": False,
            "schema_specific_availability_inferred_from_dataset_range": False,
            "zero_price_boundary": False,
            "population_event_prevalence": False,
            "strategy_economics": False,
            "execution_quality": False,
            "acquisition_authorized": False,
            "download_authorized": False,
            "model_or_trading_authority": False,
            "owner_attestation_verification": "OWNER_ATTESTED_NOT_CRYPTOGRAPHICALLY_VERIFIED",
            "tamper_evidence_boundary": "BOUNDED_LOCAL_TAMPER_EVIDENCE_NOT_ABSOLUTE_TAMPER_PROOFING_OR_VENDOR_EXACTLY_ONCE",
        },
        "integrity": dict(EXTERNAL_PASS_INTEGRITY),
        "authorization_effect": EXTERNAL_RECEIPT_AUTHORIZATION_EFFECT,
        "receipt_sha256": None,
    }
    _require(set(wrapper) == EXTERNAL_RECEIPT_FIELDS, "external wrapper field set drifted")
    wrapper["receipt_sha256"] = self_hash(wrapper, "receipt_sha256")
    return wrapper


def _bind_external_receipt_entrypoints(
    context_loader: Callable[..., dict[str, Any]],
    nested_builder: Callable[..., tuple[dict[str, Any], str]],
    wrapper_builder: Callable[..., dict[str, Any]],
) -> tuple[
    Callable[[AuthorizedExternalClient], dict[str, Any]],
    Callable[..., None],
    Callable[[AuthorizedExternalClient], dict[str, Any]],
]:
    """Hide every provenance-bearing receipt context in lexical scope."""

    class ReceiptContext(dict[str, Any]):
        """Closure-local exact context created only by canonical disk replay."""

    def reconstruct_context(
        *,
        repo_root: Path,
        attempt_id: str,
        enforce_current_authorization_freshness: bool,
    ) -> ReceiptContext:
        return ReceiptContext(
            context_loader(
                repo_root=repo_root,
                attempt_id=attempt_id,
                enforce_current_authorization_freshness=(
                    enforce_current_authorization_freshness
                ),
            )
        )

    def require_context(context: ReceiptContext) -> None:
        _require(
            type(context) is ReceiptContext,
            "external receipt context was not reconstructed from canonical disk evidence",
            status="STOP_RESPONSE_OR_RECEIPT_INVALID",
        )

    def validate_object(
        receipt: Mapping[str, Any],
        *,
        context: ReceiptContext,
    ) -> None:
        require_context(context)
        _require(
            set(receipt) == EXTERNAL_RECEIPT_FIELDS,
            "external receipt exact-field set drifted",
        )
        _require(
            receipt.get("artifact_type") == "JOB50_EXTERNAL_METADATA_RECEIPT_V1",
            "wrong external receipt artifact",
        )
        _require(
            receipt.get("schema_version")
            == "v5.job50-external-metadata-receipt.v1",
            "wrong external receipt schema",
        )
        _require(
            receipt.get("status") == "JOB50_EXTERNAL_METADATA_PREFLIGHT_PASS_ONLY",
            "external receipt is not a pass",
        )
        _require(
            receipt.get("integrity") == EXTERNAL_PASS_INTEGRITY,
            "external pass integrity object drifted",
        )
        _require(
            receipt.get("authorization_effect")
            == EXTERNAL_RECEIPT_AUTHORIZATION_EFFECT,
            "external authorization effect drifted",
        )
        claims = receipt.get("claims")
        _require(
            isinstance(claims, Mapping)
            and claims.get("owner_attestation_verification")
            == "OWNER_ATTESTED_NOT_CRYPTOGRAPHICALLY_VERIFIED",
            "owner-attestation epistemic disclosure drifted",
        )
        claimed = _require_sha(
            receipt.get("receipt_sha256"),
            "external receipt self-hash",
        )
        _require(
            claimed == self_hash(receipt, "receipt_sha256"),
            "external receipt self-hash mismatch",
        )
        nested, nested_raw_sha = nested_builder(context, write_new=False)
        expected = wrapper_builder(
            context,
            nested_receipt=nested,
            nested_file_sha256=nested_raw_sha,
        )
        _require(
            dict(receipt) == expected,
            "external Job-50 wrapper reconstruction mismatch",
        )

    def build(
        capability: AuthorizedExternalClient,
    ) -> dict[str, Any]:
        if type(capability) is not AuthorizedExternalClient:
            raise MetadataCensusError(
                "external receipt build requires the exact authorized-client capability",
                status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
            )
        capability._require_production_usable()
        bound = capability._bindings_for_receipt()
        context = reconstruct_context(
            repo_root=Path(bound["repo_root"]),
            attempt_id=str(bound["attempt_id"]),
            enforce_current_authorization_freshness=True,
        )
        for field in (
            "authorization_sha256",
            "authorization_file_sha256",
            "attempt_id",
            "local_readiness_receipt_sha256",
            "local_readiness_receipt_file_sha256",
            "declaration_sha256",
            "declaration_file_sha256",
            "sdk_identity_sha256",
            "quoted_acquisition_ceiling_usd",
        ):
            expected = {
                "authorization_sha256": context["authorization"][
                    "authorization_sha256"
                ],
                "authorization_file_sha256": file_sha256(
                    context["paths"]["authorization"]
                ),
                "attempt_id": context["authorization"]["attempt_id"],
                "local_readiness_receipt_sha256": context["local_receipt"][
                    "receipt_sha256"
                ],
                "local_readiness_receipt_file_sha256": file_sha256(
                    context["paths"]["local_readiness_receipt"]
                ),
                "declaration_sha256": context["declaration"][
                    "declaration_sha256"
                ],
                "declaration_file_sha256": file_sha256(
                    context["paths"]["declaration"]
                ),
                "sdk_identity_sha256": _readiness_sdk_identity_sha256(
                    context["local_receipt"]
                ),
                "quoted_acquisition_ceiling_usd": str(
                    context["authorization"]["quoted_acquisition_ceiling_usd"]
                ),
            }[field]
            _require(
                bound[field] == expected,
                f"receipt capability {field} drifted",
                status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
            )
        nested, nested_raw_sha = nested_builder(context, write_new=True)
        wrapper = wrapper_builder(
            context,
            nested_receipt=nested,
            nested_file_sha256=nested_raw_sha,
        )
        validate_object(wrapper, context=context)
        return wrapper

    def validate(
        receipt: Mapping[str, Any],
        *,
        repo_root: Path,
        receipt_path: Path | None = None,
    ) -> None:
        try:
            _require(isinstance(receipt, Mapping), "external receipt is not an object")
            _require(
                set(receipt) == EXTERNAL_RECEIPT_FIELDS,
                "external receipt exact-field set drifted",
            )
            context = reconstruct_context(
                repo_root=repo_root,
                attempt_id=str(receipt.get("attempt_id")),
                enforce_current_authorization_freshness=False,
            )
            expected_path = context["paths"]["external_receipt"]
            if receipt_path is not None:
                _require(
                    Path(receipt_path).resolve() == expected_path,
                    "external receipt path is not canonical",
                )
                disk_receipt = _load_canonical_json_output(expected_path)
                _require(
                    dict(receipt) == disk_receipt,
                    "external receipt object/disk mismatch",
                )
            validate_object(receipt, context=context)
        except MetadataCensusError:
            raise
        except Exception as exc:
            raise MetadataCensusError(
                "external metadata receipt validation failed closed",
                status="STOP_RESPONSE_OR_RECEIPT_INVALID",
            ) from exc

    def finalize(
        capability: AuthorizedExternalClient,
    ) -> dict[str, Any]:
        wrapper = build(capability)
        bound = capability._bindings_for_receipt()
        path = (
            Path(bound["attempt_directory"]).resolve()
            / "EXTERNAL_METADATA_RECEIPT_V1.json"
        )
        reread = _write_json_exclusive(path, wrapper)
        validate(
            reread,
            repo_root=Path(bound["repo_root"]),
            receipt_path=path,
        )
        return reread

    build.__name__ = "build_external_metadata_receipt"
    build.__qualname__ = "build_external_metadata_receipt"
    validate.__name__ = "validate_external_metadata_receipt"
    validate.__qualname__ = "validate_external_metadata_receipt"
    finalize.__name__ = "finalize_external_metadata_receipt"
    finalize.__qualname__ = "finalize_external_metadata_receipt"
    return build, validate, finalize


(
    build_external_metadata_receipt,
    validate_external_metadata_receipt,
    finalize_external_metadata_receipt,
) = _bind_external_receipt_entrypoints(
    _external_receipt_context_engine,
    _build_nested_structural_receipt_engine,
    _external_wrapper_from_context_engine,
)
del _external_receipt_context_engine
del _build_nested_structural_receipt_engine
del _external_wrapper_from_context_engine
del _bind_external_receipt_entrypoints


class SyntheticMetadataClient:
    """Deterministic no-network fake exercising all four request shapes.

    Tests may mutate ``overrides`` with keys ``(method, session_or_none)``.  A
    callable override receives the SDK keyword arguments; an exception is
    raised; any other value is returned directly.
    """

    class _Metadata:
        def __init__(self, owner: "SyntheticMetadataClient") -> None:
            self.owner = owner

        def get_dataset_range(self, **parameters: Any) -> Any:
            return self.owner._answer("metadata.get_dataset_range", None, parameters)

        def get_cost(self, **parameters: Any) -> Any:
            return self.owner._answer("metadata.get_cost", self.owner._session_from_start(parameters.get("start")), parameters)

        def get_record_count(self, **parameters: Any) -> Any:
            return self.owner._answer("metadata.get_record_count", self.owner._session_from_start(parameters.get("start")), parameters)

    class _Symbology:
        def __init__(self, owner: "SyntheticMetadataClient") -> None:
            self.owner = owner

        def resolve(self, **parameters: Any) -> Any:
            return self.owner._answer("symbology.resolve", str(parameters.get("start_date")), parameters)

    def __init__(
        self,
        declaration: Mapping[str, Any],
        *,
        overrides: Mapping[tuple[str, str | None], Any] | None = None,
    ) -> None:
        try:
            catalogue.validate_catalogue_declaration(declaration)
        except catalogue.CataloguePreflightError as exc:
            raise MetadataCensusError(str(exc), status=exc.status) from exc
        self.declaration = declaration
        self.overrides = dict(overrides or {})
        self.metadata = self._Metadata(self)
        self.symbology = self._Symbology(self)
        self.calls: list[dict[str, Any]] = []
        self._rows = {row["session"]: row for row in declaration["sessions"]}

    @staticmethod
    def _session_from_start(value: Any) -> str:
        _require(isinstance(value, str) and len(value) >= 10, "synthetic call lacks start time")
        return value[:10]

    def _answer(self, method: str, session: str | None, parameters: Mapping[str, Any]) -> Any:
        self.calls.append({"method": method, "session": session, "parameters": dict(parameters)})
        key = (method, session)
        if key in self.overrides:
            override = self.overrides[key]
            if isinstance(override, BaseException):
                raise override
            return override(**parameters) if callable(override) else override
        if method == "metadata.get_dataset_range":
            return {"start": catalogue.COVERAGE_START.isoformat(), "end": "2026-08-01"}
        assert session is not None
        row = self._rows[session]
        if method == "symbology.resolve":
            return {
                "result": {symbol: [{"d0": session, "d1": str(parameters["end_date"]), "s": str(index + 1)}] for index, symbol in enumerate(row["symbols"])},
                "partial": [],
                "not_found": [],
            }
        if method == "metadata.get_cost":
            return 0.0
        if method == "metadata.get_record_count":
            return max(1, len(row["symbols"]))
        raise AssertionError(method)


_TEST_CAPABILITY_SEAL = object()


class _TestOnlyAuthorizedExternalClient:
    """Private fake capability that can produce synthetic evidence only."""

    __slots__ = ("_client", "_seal")

    def __init__(self, client: SyntheticMetadataClient, seal: object) -> None:
        if seal is not _TEST_CAPABILITY_SEAL or type(client) is not SyntheticMetadataClient:
            raise MetadataCensusError(
                "invalid test-only authorized-client fixture",
                status="STOP_FORBIDDEN_METHOD_OR_SCOPE",
            )
        self._client = client
        self._seal = seal


def _construct_test_only_authorized_external_client(
    client: SyntheticMetadataClient,
) -> _TestOnlyAuthorizedExternalClient:
    return _TestOnlyAuthorizedExternalClient(client, _TEST_CAPABILITY_SEAL)


def _run_test_only_authorized_external_client(
    capability: _TestOnlyAuthorizedExternalClient,
    declaration: Mapping[str, Any],
    *,
    call_journal_path: Path,
    response_path: Path,
    attempt_id: str,
) -> dict[str, Any]:
    """Exercise accounting with a fake while refusing all external provenance."""

    _require(
        type(capability) is _TestOnlyAuthorizedExternalClient
        and capability._seal is _TEST_CAPABILITY_SEAL,
        "invalid test-only capability",
        status="STOP_FORBIDDEN_METHOD_OR_SCOPE",
    )
    return _execute_metadata_census(
        capability._client,
        declaration,
        call_journal_path=call_journal_path,
        response_path=response_path,
        sdk_version=EXPECTED_SDK_VERSION,
        contract_sha256=PROGRAM_CONTRACT_SHA256,
        attempt_id=attempt_id,
    )


def _run_metadata_census_with_injected_synthetic_client(
    client: SyntheticMetadataClient,
    declaration: Mapping[str, Any],
    *,
    call_journal_path: Path,
    response_path: Path,
    contract_sha256: str,
    clock: Callable[[], str] | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Private fake-only injection seam used by the no-network test suite."""

    _require(
        isinstance(client, SyntheticMetadataClient),
        "injected synthetic execution requires the repository-owned fake client",
        status="STOP_FORBIDDEN_METHOD_OR_SCOPE",
    )
    return _execute_metadata_census(
        client,
        declaration,
        call_journal_path=call_journal_path,
        response_path=response_path,
        sdk_version=EXPECTED_SDK_VERSION,
        contract_sha256=contract_sha256,
        clock=clock,
        attempt_id=attempt_id,
    )


def run_synthetic_metadata_census(
    declaration: Mapping[str, Any],
    *,
    call_journal_path: Path,
    response_path: Path,
    clock: Callable[[], str] | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Run the public no-network rehearsal with no caller-supplied client."""

    return _run_metadata_census_with_injected_synthetic_client(
        SyntheticMetadataClient(declaration),
        declaration,
        call_journal_path=call_journal_path,
        response_path=response_path,
        contract_sha256=PROGRAM_CONTRACT_SHA256,
        clock=clock,
        attempt_id=attempt_id,
    )


def _prepare_authorized_external_metadata_census(
    capability: AuthorizedExternalClient,
) -> dict[str, Any]:
    """Reconstruct every authority binding without making an SDK call."""

    if type(capability) is not AuthorizedExternalClient:
        raise MetadataCensusError(
            "external execution requires the exact authorized-client capability",
            status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
        )
    client, bound = capability._dispatch_surface_for_core()
    from v5.research import cmbp_metadata_census_receipt_v2 as readiness

    root = Path(bound["repo_root"]).resolve()
    work = root / "v5/work/cmbp-metadata-census"
    canonical_paths = {
        "declaration_path": root / "v5/work/human-policy-foundation/CMBP_CATALOGUE_DECLARATION_V1.json",
        "program_contract_path": work / "PROGRAM_CONTRACT_V4.json",
        "prior_program_contract_path": work / "PROGRAM_CONTRACT_V3.json",
        "intermediate_program_contract_path": work / "PROGRAM_CONTRACT_V2.json",
        "base_program_contract_path": work / "PROGRAM_CONTRACT_V1.json",
        "local_readiness_receipt_path": work / "LOCAL_READINESS_RECEIPT_V2.json",
        "readiness_test_report_path": work / "TEST_RESULTS_V2.xml",
        "synthetic_journal_path": work / "SYNTHETIC_CALL_JOURNAL_V2.jsonl",
        "synthetic_response_path": work / "SYNTHETIC_METADATA_RESPONSES_V2.json",
        "authorization_path": (
            work
            / "authorizations"
            / str(bound["attempt_id"])
            / "VENDOR_RUN_AUTHORIZATION_V2.json"
        ),
        "consumption_record_path": (
            work
            / "authorization-consumptions"
            / f"{bound['authorization_sha256']}.json"
        ),
        "attempt_directory": work / "external-attempts" / str(bound["attempt_id"]),
    }
    for field, expected in canonical_paths.items():
        _require(
            Path(bound[field]).resolve() == expected,
            f"authorized-client {field} is not canonical",
            status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
        )

    declaration = load_json(canonical_paths["declaration_path"])
    contract = load_json(canonical_paths["program_contract_path"])
    prior_contract = load_json(canonical_paths["prior_program_contract_path"])
    intermediate_contract = load_json(
        canonical_paths["intermediate_program_contract_path"]
    )
    base_contract = load_json(canonical_paths["base_program_contract_path"])
    _contract_semantic_sha256(contract)
    _prior_contract_semantic_sha256(prior_contract)
    _intermediate_contract_semantic_sha256(intermediate_contract)
    _base_contract_semantic_sha256(base_contract)
    _require(
        file_sha256(canonical_paths["program_contract_path"])
        == PROGRAM_CONTRACT_FILE_SHA256,
        "effective V4 contract raw-file identity drifted",
    )
    _require(
        file_sha256(canonical_paths["prior_program_contract_path"])
        == PRIOR_PROGRAM_CONTRACT_FILE_SHA256,
        "preserved V3 contract raw-file identity drifted",
    )
    _require(
        file_sha256(canonical_paths["intermediate_program_contract_path"])
        == INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256,
        "preserved V2 contract raw-file identity drifted",
    )
    _require(
        file_sha256(canonical_paths["base_program_contract_path"])
        == BASE_PROGRAM_CONTRACT_FILE_SHA256,
        "preserved V1 contract raw-file identity drifted",
    )
    try:
        local_receipt = readiness.strict_json(
            canonical_paths["local_readiness_receipt_path"]
        )
        readiness.validate_local_readiness_receipt(
            local_receipt,
            repo_root=root,
            test_report_path=canonical_paths["readiness_test_report_path"],
            synthetic_journal_path=canonical_paths["synthetic_journal_path"],
            synthetic_response_path=canonical_paths["synthetic_response_path"],
            require_vendor_authorization_absent=False,
        )
    except Exception as exc:
        raise MetadataCensusError(
            "sealed local readiness receipt no longer reconstructs",
            status="STOP_LOCAL_READINESS_RECEIPT_INVALID",
        ) from exc

    authorization = load_json(canonical_paths["authorization_path"])
    cap = verify_vendor_run_authorization(
        authorization,
        declaration,
        contract,
        prior_contract=prior_contract,
        intermediate_contract=intermediate_contract,
        base_contract=base_contract,
        local_readiness_receipt=local_receipt,
        local_readiness_receipt_file_sha256=file_sha256(
            canonical_paths["local_readiness_receipt_path"]
        ),
        authorization_path=canonical_paths["authorization_path"],
        repo_root=root,
        attempt_id=str(bound["attempt_id"]),
        declaration_file_sha256=file_sha256(canonical_paths["declaration_path"]),
    )
    consumption = load_json(canonical_paths["consumption_record_path"])
    attempt_directory = verify_authorization_consumption(
        consumption,
        record_path=canonical_paths["consumption_record_path"],
        authorization=authorization,
        authorization_path=canonical_paths["authorization_path"],
        repo_root=root,
        local_readiness_receipt=local_receipt,
    )

    expected_bindings = {
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": file_sha256(canonical_paths["authorization_path"]),
        "attempt_id": authorization["attempt_id"],
        "attempt_directory": attempt_directory,
        "consumption_marker_sha256": file_sha256(canonical_paths["consumption_record_path"]),
        "local_readiness_receipt_sha256": local_receipt["receipt_sha256"],
        "local_readiness_receipt_file_sha256": file_sha256(canonical_paths["local_readiness_receipt_path"]),
        "base_program_contract_sha256": BASE_PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_sha256": INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_sha256": PRIOR_PROGRAM_CONTRACT_SHA256,
        "program_contract_sha256": PROGRAM_CONTRACT_SHA256,
        "declaration_sha256": declaration["declaration_sha256"],
        "declaration_file_sha256": file_sha256(canonical_paths["declaration_path"]),
        "sdk_identity_sha256": _readiness_sdk_identity_sha256(local_receipt),
        "quoted_acquisition_ceiling_usd": str(authorization["quoted_acquisition_ceiling_usd"]),
    }
    for field, expected in expected_bindings.items():
        actual = bound[field]
        if field == "attempt_directory":
            actual = Path(actual).resolve()
        _require(
            actual == expected,
            f"authorized-client capability {field} drifted",
            status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
        )
    _require(
        cap == Decimal(str(bound["quoted_acquisition_ceiling_usd"])),
        "authorized-client quoted ceiling drifted",
        status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
    )
    capability._require_production_usable()

    return {
        "client": client,
        "declaration": declaration,
        "attempt_directory": attempt_directory,
        "authorization_sha256": str(authorization["authorization_sha256"]),
        "quoted_acquisition_ceiling_usd": str(
            authorization["quoted_acquisition_ceiling_usd"]
        ),
        "attempt_id": str(bound["attempt_id"]),
    }


def _bind_metadata_execution_entrypoints(
    engine: Callable[..., dict[str, Any]],
    prepare_external: Callable[[AuthorizedExternalClient], dict[str, Any]],
) -> tuple[Callable[..., dict[str, Any]], Callable[[AuthorizedExternalClient], dict[str, Any]]]:
    """Capture the only external selector lexically, then erase its globals."""

    def synthetic_execute(
        client: Any,
        declaration: Mapping[str, Any],
        *,
        call_journal_path: Path,
        response_path: Path,
        sdk_version: str = EXPECTED_SDK_VERSION,
        contract_sha256: str,
        clock: Callable[[], str] | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        return engine(
            client,
            declaration,
            call_journal_path=call_journal_path,
            response_path=response_path,
            source="synthetic",
            sdk_version=sdk_version,
            contract_sha256=contract_sha256,
            authorization_sha256=None,
            quoted_acquisition_ceiling_usd="0",
            clock=clock,
            attempt_id=attempt_id,
        )

    def authorized_execute(
        capability: AuthorizedExternalClient,
    ) -> dict[str, Any]:
        # No provenance-bearing object exists until this full reconstruction
        # returns.  The captured engine and its external selector are not
        # module globals and have no separately callable external entrypoint.
        prepared = prepare_external(capability)
        return engine(
            prepared["client"],
            prepared["declaration"],
            call_journal_path=(
                prepared["attempt_directory"] / "CALL_JOURNAL_V1.jsonl"
            ),
            response_path=(
                prepared["attempt_directory"]
                / "EXTERNAL_METADATA_RESPONSES_V1.json"
            ),
            source="externally_supplied",
            sdk_version=EXPECTED_SDK_VERSION,
            contract_sha256=PROGRAM_CONTRACT_SHA256,
            authorization_sha256=prepared["authorization_sha256"],
            quoted_acquisition_ceiling_usd=prepared[
                "quoted_acquisition_ceiling_usd"
            ],
            attempt_id=prepared["attempt_id"],
        )

    synthetic_execute.__name__ = "_execute_metadata_census"
    synthetic_execute.__qualname__ = "_execute_metadata_census"
    synthetic_execute.__doc__ = (
        "Synthetic-only census executor. It exposes no external provenance selector."
    )
    authorized_execute.__name__ = "run_authorized_external_metadata_census"
    authorized_execute.__qualname__ = "run_authorized_external_metadata_census"
    authorized_execute.__doc__ = (
        "Reverify the opaque production capability and execute one external census."
    )
    return synthetic_execute, authorized_execute


(
    _execute_metadata_census,
    run_authorized_external_metadata_census,
) = _bind_metadata_execution_entrypoints(
    _execute_metadata_census_engine,
    _prepare_authorized_external_metadata_census,
)
del _execute_metadata_census_engine
del _prepare_authorized_external_metadata_census
del _bind_metadata_execution_entrypoints
