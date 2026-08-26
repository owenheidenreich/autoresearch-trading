"""Fail-closed Job-51 Tier-0 CMBP acquisition and QC primitives.

The production entrypoint lives in :mod:`v5.ops.acquire_cmbp_tier0`.  This
module owns the frozen-scope reconstruction, external-volume guard, durable
call accounting, readiness seal, historical DBN QC, and aggregate receipt.
It does not construct a vendor client or read a credential.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import plistlib
import re
import stat
import subprocess
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


JOB_ID = 51
SCOPE_ARTIFACT = "JOB51_TIER0_ACQUISITION_SCOPE_V1"
CONTRACT_ARTIFACT = "JOB51_PROGRAM_CONTRACT_V1"
READINESS_ARTIFACT = "JOB51_LOCAL_READINESS_RECEIPT_V1"
READINESS_STATUS = "JOB51_LOCAL_READY_ONLY"
SESSION_QC_ARTIFACT = "JOB51_SESSION_QC_V1"
AGGREGATE_RECEIPT_ARTIFACT = "JOB51_ACQUISITION_QC_RECEIPT_V1"
JOURNAL_ARTIFACT = "JOB51_ACQUISITION_JOURNAL_V1"
WATERMARK_ARTIFACT = "JOB51_JOURNAL_WATERMARK_V1"
CALL_MARKER_ARTIFACT = "JOB51_VENDOR_CALL_MARKER_V1"
EXPECTED_SCOPE_SHA256 = "87050c2b3ad248d4d96c5f6945b883241cdd89a53a2f9902d6e70baf5218a099"
EXPECTED_SCOPE_FILE_SHA256 = "1760ec585c23f5213a70a46c5eb6a76400f8cad27569fb1e4ef64abfe41b2463"
EXPECTED_PROGRAM_CONTRACT_SHA256 = "98a5273aad45f98133f16e2fd45158caea1eb46a76571c8a40a009708fd90e15"
EXPECTED_PROGRAM_CONTRACT_FILE_SHA256 = "5d170653c08f45e2035667cb5ee1e8fb34d3fc113146ec75311ae9a0e596cd1f"
EXPECTED_JOB50_RECEIPT_FILE_SHA256 = "319c05bd1ca06edd14f6c4a847debb4a64880dd2f3a5c80450af86fa41dce076"
EXPECTED_JOB50_RESPONSE_FILE_SHA256 = "b759b48641ff529196356ce795eec8f53434452203c8b628cdd73bfa342cef71"
EXPECTED_JOB49_DECLARATION_FILE_SHA256 = "51066b09a0a6add8b2bed407c2a8b585f0689825a1065339abb986831b1b59ba"
EXPECTED_DATASET = "OPRA.PILLAR"
EXPECTED_SCHEMA = "cmbp-1"
EXPECTED_STYPE_IN = "raw_symbol"
EXPECTED_STYPE_OUT = "instrument_id"
EXPECTED_SESSION_COUNT = 21
EXPECTED_RECORD_COUNT = 2_373_877_845
EXPECTED_SESSION_SYMBOLS = 1_001
EXPECTED_VOLUME_ROOT = Path("/Volumes/AR_TRADING_DATA")
EXPECTED_VOLUME_UUID = "8CBA2FD2-1446-4439-866C-3BEA6C297E30"
EXPECTED_FILESYSTEM = "apfs"
EXPECTED_DATABENTO_VERSION = "0.77.0"
EXPECTED_DATABENTO_DBN_VERSION = "0.56.0"
EXPECTED_FOCUSED_TEST_COUNT = 78
EXPECTED_FOCUSED_TEST_IDENTITY_SHA256 = "4c76628aa6e802ed843970761f5645f5adb5737426fe16f6cf061d2290ca1e6e"
MIN_FREE_BYTES = 100_000_000_000
PLANNING_BYTES_PER_RECORD = 32
GENESIS_HASH = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
OSI_RE = re.compile(r"^SPXW\s{2}(\d{6})([CP])(\d{8})$")
MARKER_NAME_RE = re.compile(r"^(\d{6})-([a-z0-9_]+)\.json$")
ALLOWED_JOURNAL_EVENTS = frozenset(
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
        "SESSION_REUSED",
        "ATTEMPT_SEALED_FOR_AGGREGATE",
        "ATTEMPT_STOP",
    }
)


class Tier0Error(RuntimeError):
    """Named fail-closed Job-51 stop."""

    def __init__(self, message: str, *, status: str = "STOP_JOB51_INVALID") -> None:
        super().__init__(message)
        self.status = status


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Tier0Error(f"duplicate JSON key: {key}", status="STOP_SCOPE_OR_SEAL_DRIFT")
        result[key] = value
    return result


def strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Tier0Error(f"nonfinite JSON token: {token}", status="STOP_SCOPE_OR_SEAL_DRIFT")
            ),
        )
    except Tier0Error:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed evidence is a named stop
        raise Tier0Error(f"cannot parse JSON evidence {path.name}", status="STOP_SCOPE_OR_SEAL_DRIFT") from exc
    if not isinstance(value, dict):
        raise Tier0Error(f"JSON evidence {path.name} is not an object", status="STOP_SCOPE_OR_SEAL_DRIFT")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    return json_sha256({key: item for key, item in value.items() if key != field})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_canonical_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, raw)
        if written != len(raw):
            raise Tier0Error("short exclusive artifact write", status="STOP_DURABLE_WRITE_FAILURE")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    if path.read_bytes() != raw:
        raise Tier0Error("exclusive artifact reread mismatch", status="STOP_DURABLE_WRITE_FAILURE")


def write_canonical_replace(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or metadata.st_nlink != 1:
            raise Tier0Error("unsafe replacement artifact target", status="STOP_EXTERNAL_PATH")
    raw = canonical_json_bytes(value) + b"\n"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temp, flags, 0o600)
    try:
        written = os.write(descriptor, raw)
        if written != len(raw):
            raise Tier0Error("short replacement artifact write", status="STOP_DURABLE_WRITE_FAILURE")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temp, path)
    fsync_directory(path.parent)
    if path.read_bytes() != raw:
        raise Tier0Error("replacement artifact reread mismatch", status="STOP_DURABLE_WRITE_FAILURE")


@dataclass(frozen=True)
class SessionRequest:
    session: str
    start: str
    end: str
    symbols: tuple[str, ...]
    expected_mappings: tuple[tuple[int, str], ...]
    expected_record_count: int
    expected_cost_usd: str
    cost_request_sha256: str
    record_count_request_sha256: str
    symbology_request_sha256: str

    @property
    def market_parameters(self) -> dict[str, Any]:
        return {
            "dataset": EXPECTED_DATASET,
            "start": self.start,
            "end": self.end,
            "symbols": list(self.symbols),
            "schema": EXPECTED_SCHEMA,
            "stype_in": EXPECTED_STYPE_IN,
        }

    @property
    def market_request_sha256(self) -> str:
        return json_sha256(self.market_parameters)

    @property
    def mapping_by_instrument(self) -> dict[int, str]:
        return dict(self.expected_mappings)


@dataclass(frozen=True)
class ScopeBundle:
    contract: Mapping[str, Any]
    scope: Mapping[str, Any]
    job50_receipt: Mapping[str, Any]
    job50_response: Mapping[str, Any]
    job49_declaration: Mapping[str, Any]
    sessions: tuple[SessionRequest, ...]
    contract_file_sha256: str
    plan_file_sha256: str


def _repo_paths(repo_root: Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    return {
        "plan": root / "v5/work/cmbp-tier0-acquisition/PLAN.md",
        "contract": root / "v5/work/cmbp-tier0-acquisition/PROGRAM_CONTRACT_V1.json",
        "scope": root / "v5/work/cmbp-metadata-census/TIER0_ACQUISITION_SCOPE_V1.json",
        "job50_receipt": root / "v5/work/cmbp-metadata-census/external-attempts/ac3249b5-b593-4114-958d-e8fad67aa47f/EXTERNAL_METADATA_RECEIPT_V1.json",
        "job50_response": root / "v5/work/cmbp-metadata-census/external-attempts/ac3249b5-b593-4114-958d-e8fad67aa47f/EXTERNAL_METADATA_RESPONSES_V1.json",
        "job49_declaration": root / "v5/work/human-policy-foundation/CMBP_CATALOGUE_DECLARATION_V1.json",
        "readiness": root / "v5/work/cmbp-tier0-acquisition/LOCAL_READINESS_RECEIPT_V1.json",
        "test_report": root / "v5/work/cmbp-tier0-acquisition/TEST_RESULTS_V1.xml",
    }


def _request_for(session: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    matches = [item for item in session.get("requests", []) if item.get("method") == method]
    if len(matches) != 1:
        raise Tier0Error(
            f"{session.get('session')}: expected exactly one {method} descriptor",
            status="STOP_SCOPE_OR_SEAL_DRIFT",
        )
    return matches[0]


def _expiry_from_symbol(symbol: str) -> str:
    match = OSI_RE.fullmatch(symbol)
    if match is None:
        raise Tier0Error(f"invalid frozen OSI raw symbol {symbol!r}", status="STOP_CONTRACT_MAPPING")
    raw = match.group(1)
    return f"20{raw[:2]}-{raw[2:4]}-{raw[4:6]}"


def _selection_dates(response_sessions: Sequence[Mapping[str, Any]]) -> list[str]:
    zero = sorted(
        str(item["session"])
        for item in response_sessions
        if item.get("status") == "OK" and item.get("cost_usd") == "0"
    )
    if len(zero) != 228:
        raise Tier0Error(f"Job-50 zero-cost population drifted to {len(zero)}", status="STOP_SCOPE_OR_SEAL_DRIFT")
    selected = [zero[round(index * (len(zero) - 1) / 19)] for index in range(20)]
    early = "2026-07-02"
    if early not in selected:
        selected.append(early)
    return sorted(selected)


def load_scope_bundle(repo_root: Path) -> ScopeBundle:
    paths = _repo_paths(repo_root)
    for name, path in paths.items():
        if name in {"readiness", "test_report"}:
            continue
        if not path.is_file() or path.is_symlink():
            raise Tier0Error(f"required frozen input missing or symlinked: {name}", status="STOP_SCOPE_OR_SEAL_DRIFT")

    contract = strict_json(paths["contract"])
    if contract.get("artifact_type") != CONTRACT_ARTIFACT or contract.get("job_id") != JOB_ID:
        raise Tier0Error("Job-51 program contract identity drifted", status="STOP_SCOPE_OR_SEAL_DRIFT")
    if (
        contract.get("contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or self_hash(contract, "contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or file_sha256(paths["contract"]) != EXPECTED_PROGRAM_CONTRACT_FILE_SHA256
    ):
        raise Tier0Error("Job-51 program contract self-hash mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")

    scope = strict_json(paths["scope"])
    if file_sha256(paths["scope"]) != EXPECTED_SCOPE_FILE_SHA256:
        raise Tier0Error("Tier-0 scope raw file hash mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
    if scope.get("artifact_type") != SCOPE_ARTIFACT:
        raise Tier0Error("Tier-0 scope artifact type drifted", status="STOP_SCOPE_OR_SEAL_DRIFT")
    if scope.get("scope_sha256") != EXPECTED_SCOPE_SHA256 or self_hash(scope, "scope_sha256") != EXPECTED_SCOPE_SHA256:
        raise Tier0Error("Tier-0 scope semantic hash mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")

    expected_top = {
        "dataset": EXPECTED_DATASET,
        "schema": EXPECTED_SCHEMA,
        "stype_in": EXPECTED_STYPE_IN,
        "destination_root": str(EXPECTED_VOLUME_ROOT),
        "quoted_cost_usd": "0",
        "session_count": EXPECTED_SESSION_COUNT,
        "total_record_count": EXPECTED_RECORD_COUNT,
        "total_session_symbols": EXPECTED_SESSION_SYMBOLS,
        "source_census_receipt_sha256": EXPECTED_JOB50_RECEIPT_FILE_SHA256,
        "source_responses_file_sha256": EXPECTED_JOB50_RESPONSE_FILE_SHA256,
    }
    for field, expected in expected_top.items():
        if scope.get(field) != expected:
            raise Tier0Error(f"Tier-0 scope {field} drifted", status="STOP_SCOPE_OR_SEAL_DRIFT")

    if file_sha256(paths["job50_receipt"]) != EXPECTED_JOB50_RECEIPT_FILE_SHA256:
        raise Tier0Error("Job-50 receipt raw hash mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
    if file_sha256(paths["job50_response"]) != EXPECTED_JOB50_RESPONSE_FILE_SHA256:
        raise Tier0Error("Job-50 response raw hash mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
    if file_sha256(paths["job49_declaration"]) != EXPECTED_JOB49_DECLARATION_FILE_SHA256:
        raise Tier0Error("Job-49 declaration raw hash mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
    job50_receipt = strict_json(paths["job50_receipt"])
    job50_response = strict_json(paths["job50_response"])
    declaration = strict_json(paths["job49_declaration"])
    if job50_receipt.get("status") != "JOB50_EXTERNAL_METADATA_PREFLIGHT_PASS_ONLY":
        raise Tier0Error("Job-50 external receipt is not PASS-only", status="STOP_SCOPE_OR_SEAL_DRIFT")
    response_binding = job50_receipt.get("response", {})
    if response_binding.get("file_sha256") != EXPECTED_JOB50_RESPONSE_FILE_SHA256:
        raise Tier0Error("Job-50 receipt no longer binds source response", status="STOP_SCOPE_OR_SEAL_DRIFT")
    declaration_sha = declaration.get("declaration_sha256")
    manifest_sha = declaration.get("request_manifest_sha256")
    if (
        job50_receipt.get("declaration_file_sha256") != EXPECTED_JOB49_DECLARATION_FILE_SHA256
        or job50_receipt.get("declaration_sha256") != declaration_sha
        or job50_receipt.get("request_manifest_sha256") != manifest_sha
        or job50_response.get("declaration_sha256") != declaration_sha
    ):
        raise Tier0Error("Job-50 evidence no longer binds the frozen Job-49 declaration", status="STOP_SCOPE_OR_SEAL_DRIFT")
    if job50_receipt.get("integrity", {}).get("timeseries_calls") != 0:
        raise Tier0Error("Job-50 source attempt unexpectedly made time-series calls", status="STOP_SCOPE_OR_SEAL_DRIFT")

    response_sessions = job50_response.get("sessions")
    declaration_sessions = declaration.get("sessions")
    scope_sessions = scope.get("sessions")
    if not all(isinstance(item, list) for item in (response_sessions, declaration_sessions, scope_sessions)):
        raise Tier0Error("frozen session arrays are malformed", status="STOP_SCOPE_OR_SEAL_DRIFT")
    expected_dates = _selection_dates(response_sessions)
    actual_dates = [str(item.get("session")) for item in scope_sessions]
    if actual_dates != expected_dates or actual_dates != sorted(set(actual_dates)):
        raise Tier0Error("Tier-0 outcome-blind selection does not reconstruct", status="STOP_SCOPE_OR_SEAL_DRIFT")

    response_by_date = {str(item.get("session")): item for item in response_sessions}
    declaration_by_date = {str(item.get("session")): item for item in declaration_sessions}
    requests: list[SessionRequest] = []
    total_records = 0
    total_symbols = 0
    all_memberships: set[tuple[str, str]] = set()
    for scope_item in scope_sessions:
        session = str(scope_item["session"])
        response = response_by_date.get(session)
        declared = declaration_by_date.get(session)
        if not isinstance(response, dict) or not isinstance(declared, dict):
            raise Tier0Error(f"{session}: missing source response/declaration", status="STOP_SCOPE_OR_SEAL_DRIFT")
        if declared.get("coverage_disposition") != "REQUEST_CANDIDATE" or response.get("status") != "OK":
            raise Tier0Error(f"{session}: source availability disposition drifted", status="STOP_SCOPE_OR_SEAL_DRIFT")

        symbols = tuple(declared.get("symbols", []))
        if not symbols or list(symbols) != sorted(set(symbols)):
            raise Tier0Error(f"{session}: symbols are empty, duplicate, or unordered", status="STOP_CONTRACT_MAPPING")
        if int(scope_item.get("symbol_count", -1)) != len(symbols) or int(declared.get("symbol_count", -1)) != len(symbols):
            raise Tier0Error(f"{session}: symbol count mismatch", status="STOP_CONTRACT_MAPPING")
        if any(_expiry_from_symbol(symbol) != session for symbol in symbols):
            raise Tier0Error(f"{session}: cross-expiry symbol in frozen request", status="STOP_CONTRACT_MAPPING")

        resolved = response.get("resolved_symbols")
        if not isinstance(resolved, dict) or sorted(resolved) != list(symbols) or response.get("unresolved_symbols") != []:
            raise Tier0Error(f"{session}: resolved-symbol keys do not match declaration", status="STOP_CONTRACT_MAPPING")
        mapping: list[tuple[int, str]] = []
        seen_ids: set[int] = set()
        for symbol in symbols:
            values = resolved.get(symbol)
            if not isinstance(values, list) or len(values) != 1 or not str(values[0]).isdigit():
                raise Tier0Error(f"{session}: {symbol} is not one-to-one resolved", status="STOP_CONTRACT_MAPPING")
            instrument_id = int(values[0])
            if instrument_id in seen_ids:
                raise Tier0Error(f"{session}: duplicate resolved instrument ID", status="STOP_CONTRACT_MAPPING")
            seen_ids.add(instrument_id)
            mapping.append((instrument_id, symbol))
            all_memberships.add((session, symbol))

        cost_descriptor = _request_for(declared, "metadata.get_cost")
        count_descriptor = _request_for(declared, "metadata.get_record_count")
        resolve_descriptor = _request_for(declared, "symbology.resolve")
        cost_parameters = cost_descriptor.get("parameters")
        count_parameters = count_descriptor.get("parameters")
        if cost_parameters != count_parameters:
            raise Tier0Error(f"{session}: cost/count market requests differ", status="STOP_SCOPE_OR_SEAL_DRIFT")
        expected_parameters = {
            "dataset": EXPECTED_DATASET,
            "start": scope_item.get("rth_open_utc"),
            "end": scope_item.get("rth_close_utc"),
            "symbols": list(symbols),
            "schema": EXPECTED_SCHEMA,
            "stype_in": EXPECTED_STYPE_IN,
        }
        if cost_parameters != expected_parameters:
            raise Tier0Error(f"{session}: request parameters widened or drifted", status="STOP_SCOPE_OR_SEAL_DRIFT")
        for descriptor in (cost_descriptor, count_descriptor, resolve_descriptor):
            if descriptor.get("request_sha256") != json_sha256(
                {"method": descriptor.get("method"), "parameters": descriptor.get("parameters")}
            ):
                raise Tier0Error(f"{session}: source request self-hash mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
        if response.get("cost_request_sha256") != cost_descriptor.get("request_sha256"):
            raise Tier0Error(f"{session}: cost response request binding mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
        if response.get("record_count_request_sha256") != count_descriptor.get("request_sha256"):
            raise Tier0Error(f"{session}: record response request binding mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
        if response.get("symbology_resolve_request_sha256") != resolve_descriptor.get("request_sha256"):
            raise Tier0Error(f"{session}: resolve response request binding mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")

        record_count = int(response.get("record_count", -1))
        if record_count != int(scope_item.get("record_count", -2)) or record_count <= 0:
            raise Tier0Error(f"{session}: census record count mismatch", status="STOP_SCOPE_OR_SEAL_DRIFT")
        if response.get("cost_usd") != "0" or scope_item.get("cost_usd") != "0":
            raise Tier0Error(f"{session}: frozen quote is not exact zero", status="STOP_SCOPE_OR_SEAL_DRIFT")
        requests.append(
            SessionRequest(
                session=session,
                start=str(scope_item["rth_open_utc"]),
                end=str(scope_item["rth_close_utc"]),
                symbols=symbols,
                expected_mappings=tuple(sorted(mapping)),
                expected_record_count=record_count,
                expected_cost_usd="0",
                cost_request_sha256=str(cost_descriptor["request_sha256"]),
                record_count_request_sha256=str(count_descriptor["request_sha256"]),
                symbology_request_sha256=str(resolve_descriptor["request_sha256"]),
            )
        )
        total_records += record_count
        total_symbols += len(symbols)

    if len(requests) != EXPECTED_SESSION_COUNT or total_records != EXPECTED_RECORD_COUNT:
        raise Tier0Error("Tier-0 aggregate session/record total drifted", status="STOP_SCOPE_OR_SEAL_DRIFT")
    if total_symbols != EXPECTED_SESSION_SYMBOLS or len(all_memberships) != EXPECTED_SESSION_SYMBOLS:
        raise Tier0Error("Tier-0 aggregate symbol-membership total drifted", status="STOP_CONTRACT_MAPPING")

    return ScopeBundle(
        contract=contract,
        scope=scope,
        job50_receipt=job50_receipt,
        job50_response=job50_response,
        job49_declaration=declaration,
        sessions=tuple(requests),
        contract_file_sha256=file_sha256(paths["contract"]),
        plan_file_sha256=file_sha256(paths["plan"]),
    )


@dataclass(frozen=True)
class VolumeIdentity:
    mount_point: str
    volume_uuid: str
    device_identifier: str
    filesystem: str
    bus_protocol: str
    st_dev: int
    free_bytes: int
    total_bytes: int


def inspect_destination_volume(
    root: Path = EXPECTED_VOLUME_ROOT,
    *,
    unpublished_records: int = EXPECTED_RECORD_COUNT,
    expected_device_identifier: str | None = None,
) -> VolumeIdentity:
    root = Path(root)
    if str(root) != str(EXPECTED_VOLUME_ROOT) or not root.exists() or root.is_symlink() or not root.is_dir():
        raise Tier0Error("external destination root is absent, wrong, or symlinked", status="STOP_EXTERNAL_VOLUME")
    if not os.path.ismount(root):
        raise Tier0Error("external destination is not a mounted volume", status="STOP_EXTERNAL_VOLUME")
    try:
        completed = subprocess.run(
            ["/usr/sbin/diskutil", "info", "-plist", str(root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C", "LC_ALL": "C"},
        )
        info = plistlib.loads(completed.stdout)
    except Exception as exc:  # noqa: BLE001 - an unverifiable mount is a stop
        raise Tier0Error("cannot verify external volume identity", status="STOP_EXTERNAL_VOLUME") from exc
    required = {
        "MountPoint": str(EXPECTED_VOLUME_ROOT),
        "VolumeUUID": EXPECTED_VOLUME_UUID,
        "FilesystemType": EXPECTED_FILESYSTEM,
        "Internal": False,
        "RemovableMediaOrExternalDevice": True,
        "WritableVolume": True,
    }
    for field, expected in required.items():
        if info.get(field) != expected:
            raise Tier0Error(f"external volume {field} mismatch", status="STOP_EXTERNAL_VOLUME")
    device = str(info.get("DeviceIdentifier", ""))
    if not device or (expected_device_identifier is not None and device != expected_device_identifier):
        raise Tier0Error("external volume device identifier changed", status="STOP_EXTERNAL_VOLUME")
    stat_result = root.stat()
    usage = os.statvfs(root)
    free_bytes = int(usage.f_bavail * usage.f_frsize)
    total_bytes = int(usage.f_blocks * usage.f_frsize)
    required_free = max(MIN_FREE_BYTES, max(0, int(unpublished_records)) * PLANNING_BYTES_PER_RECORD)
    if free_bytes < required_free:
        raise Tier0Error(
            f"external volume free-space guard failed ({free_bytes} < {required_free})",
            status="STOP_EXTERNAL_VOLUME",
        )
    return VolumeIdentity(
        mount_point=str(root),
        volume_uuid=str(info["VolumeUUID"]),
        device_identifier=device,
        filesystem=str(info["FilesystemType"]),
        bus_protocol=str(info.get("BusProtocol", "UNKNOWN")),
        st_dev=int(stat_result.st_dev),
        free_bytes=free_bytes,
        total_bytes=total_bytes,
    )


def ensure_nofollow_directory(path: Path, *, volume: VolumeIdentity, create: bool = False) -> None:
    path = Path(path)
    if create and not path.exists():
        try:
            os.mkdir(path, 0o700)
            fsync_directory(path.parent)
        except FileExistsError:
            pass
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Tier0Error(f"required destination directory missing: {path.name}", status="STOP_EXTERNAL_PATH") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or metadata.st_dev != volume.st_dev:
        raise Tier0Error(f"unsafe or cross-device destination directory: {path}", status="STOP_EXTERNAL_PATH")


def initialize_destination_tree(volume: VolumeIdentity) -> Path:
    root = Path(volume.mount_point)
    parent = root / "cmbp-tier0"
    job_root = parent / "job51"
    for path in (parent, job_root, job_root / "attempts", job_root / "sessions", job_root / "receipts"):
        ensure_nofollow_directory(path, volume=volume, create=True)
    return job_root


class RunLock:
    """Whole-process nonblocking lock held on the pinned external volume."""

    def __init__(self, job_root: Path, *, volume: VolumeIdentity, scope_sha256: str, readiness_sha256: str) -> None:
        self.path = Path(job_root) / "RUN_LOCK_V1"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.path, flags, 0o600)
        metadata = os.fstat(self.fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_dev != volume.st_dev:
            os.close(self.fd)
            raise Tier0Error("Job-51 run lock is unsafe or cross-device", status="STOP_CONCURRENT_RUNNER")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.fd)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise Tier0Error("another Job-51 runner holds the external lock", status="STOP_CONCURRENT_RUNNER") from exc
            raise
        binding = canonical_json_bytes(
            {"artifact_type": "JOB51_RUN_LOCK_V1", "scope_sha256": scope_sha256, "readiness_sha256": readiness_sha256}
        ) + b"\n"
        os.ftruncate(self.fd, 0)
        os.lseek(self.fd, 0, os.SEEK_SET)
        if os.write(self.fd, binding) != len(binding):
            self.close()
            raise Tier0Error("short run-lock binding write", status="STOP_DURABLE_WRITE_FAILURE")
        os.fsync(self.fd)
        fsync_directory(self.path.parent)

    def close(self) -> None:
        if getattr(self, "fd", None) is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self) -> RunLock:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def sdk_identity() -> dict[str, Any]:
    """Return exact local SDK source/binary identities without constructing a client."""

    try:
        databento_version = importlib.metadata.version("databento")
        dbn_version = importlib.metadata.version("databento-dbn")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Tier0Error("pinned Databento packages are not installed", status="STOP_SDK_DRIFT") from exc
    if databento_version != EXPECTED_DATABENTO_VERSION or dbn_version != EXPECTED_DATABENTO_DBN_VERSION:
        raise Tier0Error("pinned Databento package version drifted", status="STOP_SDK_DRIFT")

    import databento
    import databento.common.dbnstore as dbnstore_module
    import databento.common.http as http_module
    import databento.common.parsing as parsing_module
    import databento.common.validation as validation_module
    import databento.historical.api.metadata as metadata_module
    import databento.historical.api.timeseries as timeseries_module
    import databento.historical.client as client_module
    import databento_dbn
    import databento_dbn._lib as dbn_lib_module
    import databento_dbn.metadata as dbn_metadata_module
    import requests.adapters as requests_adapters
    import requests.api as requests_api
    import requests.auth as requests_auth
    import requests.models as requests_models
    import requests.sessions as requests_sessions
    import urllib3.connectionpool as urllib3_connectionpool
    import urllib3.util.retry as urllib3_retry

    expected_signatures = {
        "MetadataHttpAPI.get_cost": (
            "self", "dataset", "start", "end", "mode", "symbols", "schema", "stype_in", "limit"
        ),
        "TimeseriesHttpAPI.get_range": (
            "self", "dataset", "start", "end", "symbols", "schema", "stype_in", "stype_out", "limit", "path"
        ),
    }
    actual_methods = {
        "MetadataHttpAPI.get_cost": metadata_module.MetadataHttpAPI.get_cost,
        "TimeseriesHttpAPI.get_range": timeseries_module.TimeseriesHttpAPI.get_range,
    }
    signatures: dict[str, list[str]] = {}
    for name, method in actual_methods.items():
        actual = tuple(inspect.signature(method).parameters)
        if actual != expected_signatures[name]:
            raise Tier0Error(f"pinned SDK signature drifted: {name}", status="STOP_SDK_DRIFT")
        signatures[name] = list(actual)

    modules = {
        "databento.__init__": databento,
        "databento.historical.client": client_module,
        "databento.historical.api.metadata": metadata_module,
        "databento.historical.api.timeseries": timeseries_module,
        "databento.common.http": http_module,
        "databento.common.dbnstore": dbnstore_module,
        "databento.common.parsing": parsing_module,
        "databento.common.validation": validation_module,
        "databento_dbn": databento_dbn,
        "databento_dbn._lib": dbn_lib_module,
        "databento_dbn.metadata": dbn_metadata_module,
        "requests.api": requests_api,
        "requests.auth": requests_auth,
        "requests.adapters": requests_adapters,
        "requests.models": requests_models,
        "requests.sessions": requests_sessions,
        "urllib3.connectionpool": urllib3_connectionpool,
        "urllib3.util.retry": urllib3_retry,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise Tier0Error(f"SDK source identity unavailable: {name}", status="STOP_SDK_DRIFT")
        unresolved_path = Path(raw_path)
        if unresolved_path.is_symlink():
            raise Tier0Error(f"SDK source identity unsafe: {name}", status="STOP_SDK_DRIFT")
        path = unresolved_path.resolve()
        if not path.is_file():
            raise Tier0Error(f"SDK source identity unsafe: {name}", status="STOP_SDK_DRIFT")
        files[name] = {"path": str(path), "file_sha256": file_sha256(path)}
    distributions: dict[str, Any] = {}
    for distribution_name in ("databento", "databento-dbn", "requests", "urllib3"):
        distribution = importlib.metadata.distribution(distribution_name)
        manifest: dict[str, str] = {}
        for relative in distribution.files or ():
            relative_text = str(relative)
            located = Path(distribution.locate_file(relative))
            if located.is_symlink():
                raise Tier0Error(
                    f"SDK distribution contains a symlink: {distribution_name}",
                    status="STOP_SDK_DRIFT",
                )
            located = located.resolve()
            if not located.is_file():
                raise Tier0Error(
                    f"SDK distribution file is absent: {distribution_name}",
                    status="STOP_SDK_DRIFT",
                )
            manifest[relative_text] = file_sha256(located)
        if not manifest:
            raise Tier0Error(f"SDK distribution manifest is empty: {distribution_name}", status="STOP_SDK_DRIFT")
        distributions[distribution_name] = {
            "version": distribution.version,
            "file_count": len(manifest),
            "manifest_sha256": json_sha256(manifest),
            "files": manifest,
        }
    return {
        "databento_version": databento_version,
        "databento_dbn_version": dbn_version,
        "signatures": signatures,
        "files": files,
        "distributions": distributions,
    }


def _junit_counts(path: Path, *, enforce_frozen_population: bool = True) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:  # noqa: BLE001 - an invalid report cannot seal readiness
        raise Tier0Error("focused JUnit report is unreadable", status="STOP_LOCAL_TESTS") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise Tier0Error("focused JUnit report has no testsuite", status="STOP_LOCAL_TESTS")
    counts = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    cases = list(root.findall(".//testcase"))
    names = sorted(str(case.attrib.get("name")) for case in cases if case.attrib.get("name"))
    classnames = sorted(set(str(case.attrib.get("classname", "")) for case in cases))
    case_identities = [
        {"classname": classname, "name": name}
        for classname, name in sorted(
            (str(case.attrib.get("classname", "")), str(case.attrib.get("name", "")))
            for case in cases
        )
    ]
    case_identity_sha256 = json_sha256(case_identities)
    if counts["tests"] < 12 or counts["failures"] or counts["errors"] or counts["skipped"]:
        raise Tier0Error(f"focused tests are not a clean pass: {counts}", status="STOP_LOCAL_TESTS")
    if len(names) != counts["tests"]:
        raise Tier0Error("focused JUnit case count does not match suite count", status="STOP_LOCAL_TESTS")
    allowed = ("v5.tests.test_cmbp_stream", "v5.tests.test_cmbp_tier0")
    if not classnames or any(not name.startswith(allowed) for name in classnames):
        raise Tier0Error("JUnit report contains a non-Job51 test class", status="STOP_LOCAL_TESTS")
    if enforce_frozen_population and (
        counts["tests"] != EXPECTED_FOCUSED_TEST_COUNT
        or case_identity_sha256 != EXPECTED_FOCUSED_TEST_IDENTITY_SHA256
    ):
        raise Tier0Error("focused JUnit case identity/population drifted", status="STOP_LOCAL_TESTS")
    return {
        **counts,
        "test_names": names,
        "test_classnames": classnames,
        "test_case_identity_sha256": case_identity_sha256,
    }


def build_readiness_receipt(repo_root: Path, *, test_report_path: Path) -> dict[str, Any]:
    """Build the precredential seal after a clean focused offline test run."""

    root = Path(repo_root).resolve()
    bundle = load_scope_bundle(root)
    contract = bundle.contract
    bound_files: dict[str, str] = {}
    for relative in contract.get("required_bound_files", []):
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise Tier0Error("contract contains unsafe readiness path", status="STOP_SCOPE_OR_SEAL_DRIFT")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise Tier0Error(f"required readiness file absent or symlinked: {relative}", status="STOP_SCOPE_OR_SEAL_DRIFT")
        bound_files[relative] = file_sha256(path)
    report = Path(os.path.abspath(test_report_path))
    expected_report = Path(os.path.abspath(root / "v5/work/cmbp-tier0-acquisition/TEST_RESULTS_V1.xml"))
    if report != expected_report:
        raise Tier0Error("focused JUnit report path is not canonical", status="STOP_LOCAL_TESTS")
    report_metadata = report.lstat() if report.exists() or report.is_symlink() else None
    if (
        report_metadata is None
        or stat.S_ISLNK(report_metadata.st_mode)
        or not stat.S_ISREG(report_metadata.st_mode)
        or report_metadata.st_nlink != 1
    ):
        raise Tier0Error("focused JUnit report is missing or symlinked", status="STOP_LOCAL_TESTS")
    receipt: dict[str, Any] = {
        "artifact_type": READINESS_ARTIFACT,
        "schema_version": "v5.job51-local-readiness.v1",
        "job_id": JOB_ID,
        "status": READINESS_STATUS,
        "status_meaning": "LOCAL_BUILD_AND_TEST_SEAL_ONLY; NO_VENDOR_OR_ACQUISITION_RESULT",
        "scope_sha256": EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": contract["contract_sha256"],
        "program_contract_file_sha256": bundle.contract_file_sha256,
        "plan_file_sha256": bundle.plan_file_sha256,
        "bound_files": bound_files,
        "focused_test_report": {
            "path": str(report.relative_to(root)),
            "file_sha256": file_sha256(report),
            "runner": "./.venv/bin/python -m pytest",
            "runner_flags": [
                "-c",
                "/dev/null",
                "--rootdir=<REPOSITORY_ROOT>",
                "-p",
                "no:cacheprovider",
                "-q",
            ],
            "test_targets": ["v5/tests/test_cmbp_stream.py", "v5/tests/test_cmbp_tier0.py"],
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
            **_junit_counts(report),
        },
        "sdk_identity": sdk_identity(),
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
        "authorization_effect": "LOCAL_READINESS_ONLY; OWNER_JOB51_AUTHORITY_STILL_REQUIRED_AT_EXECUTION",
    }
    receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
    return receipt


def validate_readiness_receipt(repo_root: Path, path: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = _repo_paths(root)
    target = Path(os.path.abspath(paths["readiness"] if path is None else Path(path)))
    canonical_target = Path(os.path.abspath(paths["readiness"]))
    target_metadata = target.lstat() if target.exists() or target.is_symlink() else None
    if (
        target != canonical_target
        or target_metadata is None
        or stat.S_ISLNK(target_metadata.st_mode)
        or not stat.S_ISREG(target_metadata.st_mode)
        or target_metadata.st_nlink != 1
    ):
        raise Tier0Error("canonical Job-51 readiness receipt is absent or unsafe", status="STOP_READINESS_SEAL")
    receipt = strict_json(target)
    if receipt.get("artifact_type") != READINESS_ARTIFACT or receipt.get("status") != READINESS_STATUS:
        raise Tier0Error("Job-51 readiness status is not local-ready-only", status="STOP_READINESS_SEAL")
    if receipt.get("receipt_sha256") != self_hash(receipt, "receipt_sha256"):
        raise Tier0Error("Job-51 readiness self-hash mismatch", status="STOP_READINESS_SEAL")
    reconstructed = build_readiness_receipt(root, test_report_path=paths["test_report"])
    if receipt != reconstructed:
        raise Tier0Error("Job-51 readiness receipt does not reconstruct exactly", status="STOP_READINESS_SEAL")
    bundle = load_scope_bundle(root)
    expected_scalars = {
        "scope_sha256": EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": bundle.contract["contract_sha256"],
        "program_contract_file_sha256": bundle.contract_file_sha256,
        "plan_file_sha256": bundle.plan_file_sha256,
    }
    for field, expected in expected_scalars.items():
        if receipt.get(field) != expected:
            raise Tier0Error(f"readiness {field} drifted", status="STOP_READINESS_SEAL")
    if receipt.get("integrity", {}).get("external_calls") != 0 or receipt.get("integrity", {}).get("credential_read") is not False:
        raise Tier0Error("readiness receipt is not precredential/local-only", status="STOP_READINESS_SEAL")
    bound = receipt.get("bound_files")
    expected_bound = set(bundle.contract.get("required_bound_files", []))
    if not isinstance(bound, dict) or set(bound) != expected_bound:
        raise Tier0Error("readiness bound-file population drifted", status="STOP_READINESS_SEAL")
    for relative, expected_hash in bound.items():
        path_item = root / relative
        if not path_item.is_file() or path_item.is_symlink() or file_sha256(path_item) != expected_hash:
            raise Tier0Error(f"readiness-bound file drifted: {relative}", status="STOP_READINESS_SEAL")
    report = receipt.get("focused_test_report", {})
    report_path = root / str(report.get("path", ""))
    expected_report_path = root / "v5/work/cmbp-tier0-acquisition/TEST_RESULTS_V1.xml"
    report_fields = (
        "tests",
        "failures",
        "errors",
        "skipped",
        "test_names",
        "test_classnames",
        "test_case_identity_sha256",
    )
    if (
        report_path.resolve() != expected_report_path.resolve()
        or report_path.is_symlink()
        or file_sha256(report_path) != report.get("file_sha256")
        or _junit_counts(report_path) != {key: report[key] for key in report_fields}
    ):
        raise Tier0Error("readiness focused-test report drifted", status="STOP_READINESS_SEAL")
    if sdk_identity() != receipt.get("sdk_identity"):
        raise Tier0Error("readiness SDK identity drifted", status="STOP_SDK_DRIFT")
    return receipt


def seal_readiness(repo_root: Path) -> Path:
    root = Path(repo_root).resolve()
    paths = _repo_paths(root)
    receipt = build_readiness_receipt(root, test_report_path=paths["test_report"])
    if paths["readiness"].exists():
        raise Tier0Error("readiness receipt already exists; V1 is immutable", status="STOP_READINESS_SEAL")
    write_canonical_exclusive(paths["readiness"], receipt)
    validate_readiness_receipt(root, paths["readiness"])
    return paths["readiness"]


CALL_EVENTS = frozenset(
    {
        "COST_CALL_START",
        "COST_CALL_RESULT",
        "COST_CALL_ERROR",
        "TIMESERIES_CALL_START",
        "TIMESERIES_CALL_RESULT",
        "TIMESERIES_CALL_ERROR",
    }
)


class AttemptJournal:
    """Append-only journal with independent watermark and exclusive call markers."""

    def __init__(self, attempt_dir: Path, *, attempt_id: str, volume: VolumeIdentity) -> None:
        if UUID4_RE.fullmatch(attempt_id) is None:
            raise Tier0Error("attempt ID is not canonical UUIDv4", status="STOP_ATTEMPT_ID")
        self.attempt_dir = Path(attempt_dir)
        ensure_nofollow_directory(self.attempt_dir, volume=volume, create=False)
        self.journal_path = self.attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl"
        self.watermark_path = self.attempt_dir / "JOURNAL_WATERMARK_V1.json"
        self.marker_dir = self.attempt_dir / "call-markers"
        ensure_nofollow_directory(self.marker_dir, volume=volume, create=True)
        if self.journal_path.exists() or self.watermark_path.exists():
            raise Tier0Error("attempt journal path already exists", status="STOP_ATTEMPT_PATH_EXISTS")
        self.attempt_id = attempt_id
        self.sequence = -1
        self.head = GENESIS_HASH
        self.last_record: dict[str, Any] | None = None
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.journal_path, flags, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        fsync_directory(self.attempt_dir)

    def append(self, event: str, *, session: str | None = None, request_sha256: str | None = None,
               payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(event, str) or not event:
            raise Tier0Error("journal event is invalid", status="STOP_JOURNAL_INVALID")
        sequence = self.sequence + 1
        record: dict[str, Any] = {
            "artifact_type": JOURNAL_ARTIFACT,
            "schema_version": "v5.job51-acquisition-journal.v1",
            "attempt_id": self.attempt_id,
            "sequence": sequence,
            "recorded_at_utc": utc_now(),
            "event": event,
            "session": session,
            "request_sha256": request_sha256,
            "payload": dict(payload or {}),
            "previous_hash": self.head,
        }
        record["record_hash"] = self_hash(record, "record_hash")
        raw = canonical_json_bytes(record) + b"\n"
        descriptor = os.open(self.journal_path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise Tier0Error("attempt journal is not a private regular file", status="STOP_JOURNAL_INVALID")
            if os.write(descriptor, raw) != len(raw):
                raise Tier0Error("short journal append", status="STOP_DURABLE_WRITE_FAILURE")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.sequence = sequence
        self.head = str(record["record_hash"])
        self.last_record = record

        if event in CALL_EVENTS:
            marker: dict[str, Any] = {
                "artifact_type": CALL_MARKER_ARTIFACT,
                "schema_version": "v5.job51-vendor-call-marker.v1",
                "attempt_id": self.attempt_id,
                "sequence": sequence,
                "event": event,
                "session": session,
                "request_sha256": request_sha256,
                "journal_record_hash": self.head,
                "recorded_at_utc": record["recorded_at_utc"],
                "payload": dict(payload or {}),
            }
            marker["marker_sha256"] = self_hash(marker, "marker_sha256")
            marker_path = self.marker_dir / f"{sequence:06d}-{event.lower()}.json"
            write_canonical_exclusive(marker_path, marker)

        watermark: dict[str, Any] = {
            "artifact_type": WATERMARK_ARTIFACT,
            "schema_version": "v5.job51-journal-watermark.v1",
            "attempt_id": self.attempt_id,
            "terminal_sequence": self.sequence,
            "terminal_head": self.head,
            "journal_file_sha256": file_sha256(self.journal_path),
            "updated_at_utc": utc_now(),
        }
        watermark["watermark_sha256"] = self_hash(watermark, "watermark_sha256")
        write_canonical_replace(self.watermark_path, watermark)
        return record


def create_attempt(job_root: Path, *, attempt_id: str, volume: VolumeIdentity) -> tuple[Path, AttemptJournal]:
    attempts = Path(job_root) / "attempts"
    ensure_nofollow_directory(attempts, volume=volume)
    attempt_dir = attempts / attempt_id
    if attempt_dir.exists() or attempt_dir.is_symlink():
        raise Tier0Error("attempt directory already exists", status="STOP_ATTEMPT_PATH_EXISTS")
    os.mkdir(attempt_dir, 0o700)
    fsync_directory(attempts)
    ensure_nofollow_directory(attempt_dir, volume=volume)
    sessions_dir = attempt_dir / "sessions"
    os.mkdir(sessions_dir, 0o700)
    fsync_directory(attempt_dir)
    return attempt_dir, AttemptJournal(attempt_dir, attempt_id=attempt_id, volume=volume)


def verify_attempt_journal(attempt_dir: Path) -> dict[str, Any]:
    attempt_dir = Path(attempt_dir)
    journal_path = attempt_dir / "ACQUISITION_JOURNAL_V1.jsonl"
    watermark_path = attempt_dir / "JOURNAL_WATERMARK_V1.json"
    marker_dir = attempt_dir / "call-markers"
    if any(path.is_symlink() for path in (attempt_dir, journal_path, watermark_path, marker_dir)):
        raise Tier0Error("attempt evidence contains a symlink", status="STOP_JOURNAL_INVALID")
    attempt_metadata = attempt_dir.lstat()
    marker_dir_metadata = marker_dir.lstat()
    for path in (journal_path, watermark_path):
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_dev != attempt_metadata.st_dev
        ):
            raise Tier0Error("attempt journal evidence is not a private regular file", status="STOP_JOURNAL_INVALID")
    if not stat.S_ISDIR(marker_dir_metadata.st_mode) or marker_dir_metadata.st_dev != attempt_metadata.st_dev:
        raise Tier0Error("attempt marker directory is unsafe", status="STOP_JOURNAL_INVALID")
    raw_lines = journal_path.read_bytes().splitlines()
    previous = GENESIS_HASH
    records: list[dict[str, Any]] = []
    for sequence, raw in enumerate(raw_lines):
        try:
            record = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
        except Exception as exc:  # noqa: BLE001
            raise Tier0Error("attempt journal JSON is invalid", status="STOP_JOURNAL_INVALID") from exc
        if canonical_json_bytes(record) != raw:
            raise Tier0Error("attempt journal record is not canonical", status="STOP_JOURNAL_INVALID")
        if (
            record.get("artifact_type") != JOURNAL_ARTIFACT
            or record.get("schema_version") != "v5.job51-acquisition-journal.v1"
            or record.get("attempt_id") != attempt_dir.name
        ):
            raise Tier0Error("attempt journal identity drifted", status="STOP_JOURNAL_INVALID")
        if record.get("sequence") != sequence or record.get("previous_hash") != previous:
            raise Tier0Error("attempt journal sequence/hash chain is broken", status="STOP_JOURNAL_INVALID")
        if record.get("record_hash") != self_hash(record, "record_hash"):
            raise Tier0Error("attempt journal record hash mismatch", status="STOP_JOURNAL_INVALID")
        previous = str(record["record_hash"])
        records.append(record)
    if not records:
        raise Tier0Error("attempt journal is empty", status="STOP_JOURNAL_INVALID")
    watermark = strict_json(watermark_path)
    if (
        watermark.get("artifact_type") != WATERMARK_ARTIFACT
        or watermark.get("schema_version") != "v5.job51-journal-watermark.v1"
        or watermark.get("attempt_id") != records[0].get("attempt_id")
        or watermark.get("attempt_id") != attempt_dir.name
        or watermark.get("watermark_sha256") != self_hash(watermark, "watermark_sha256")
    ):
        raise Tier0Error("attempt watermark self-hash mismatch", status="STOP_JOURNAL_INVALID")
    if watermark.get("terminal_sequence") != len(records) - 1 or watermark.get("terminal_head") != previous:
        raise Tier0Error("attempt journal valid-prefix/watermark mismatch", status="STOP_JOURNAL_INVALID")
    if watermark.get("journal_file_sha256") != file_sha256(journal_path):
        raise Tier0Error("attempt journal file/watermark mismatch", status="STOP_JOURNAL_INVALID")

    marker_paths = sorted(marker_dir.iterdir())
    marker_records: list[dict[str, Any]] = []
    for marker_path in marker_paths:
        marker_metadata = marker_path.lstat()
        if (
            stat.S_ISLNK(marker_metadata.st_mode)
            or not stat.S_ISREG(marker_metadata.st_mode)
            or marker_metadata.st_nlink != 1
            or marker_metadata.st_dev != attempt_metadata.st_dev
            or MARKER_NAME_RE.fullmatch(marker_path.name) is None
        ):
            raise Tier0Error("call marker directory contains an unsafe entry", status="STOP_JOURNAL_INVALID")
        marker = strict_json(marker_path)
        if marker.get("artifact_type") != CALL_MARKER_ARTIFACT or marker.get("schema_version") != "v5.job51-vendor-call-marker.v1":
            raise Tier0Error("call marker identity drifted", status="STOP_JOURNAL_INVALID")
        if marker.get("marker_sha256") != self_hash(marker, "marker_sha256"):
            raise Tier0Error("call marker self-hash mismatch", status="STOP_JOURNAL_INVALID")
        sequence = marker.get("sequence")
        if not isinstance(sequence, int) or not (0 <= sequence < len(records)):
            raise Tier0Error("call marker sequence is invalid", status="STOP_JOURNAL_INVALID")
        source = records[sequence]
        if marker_path.name != f"{sequence:06d}-{str(marker.get('event')).lower()}.json":
            raise Tier0Error("call marker filename does not bind its record", status="STOP_JOURNAL_INVALID")
        for field in ("attempt_id", "event", "session", "request_sha256"):
            if marker.get(field) != source.get(field):
                raise Tier0Error("call marker/journal mismatch", status="STOP_JOURNAL_INVALID")
        if marker.get("journal_record_hash") != source.get("record_hash") or marker.get("event") not in CALL_EVENTS:
            raise Tier0Error("call marker does not bind a vendor-call record", status="STOP_JOURNAL_INVALID")
        marker_records.append(marker)
    expected_call_records = [record for record in records if record.get("event") in CALL_EVENTS]
    expected_sequences = {int(record["sequence"]) for record in expected_call_records}
    marker_sequences = [int(marker["sequence"]) for marker in marker_records]
    if len(marker_sequences) != len(set(marker_sequences)) or set(marker_sequences) != expected_sequences:
        raise Tier0Error("call marker population differs from journal", status="STOP_JOURNAL_INVALID")
    return {
        "attempt_id": records[0].get("attempt_id"),
        "records": records,
        "watermark": watermark,
        "marker_count": len(marker_records),
        "journal_file_sha256": file_sha256(journal_path),
        "watermark_file_sha256": file_sha256(watermark_path),
        "marker_files": {path.name: file_sha256(path) for path in marker_paths},
    }


def exact_zero_quote(value: Any) -> str:
    """Normalize the SDK numeric quote, rejecting bools, strings and signed zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise Tier0Error("fresh SDK cost quote is not numeric", status="STOP_NONZERO_OR_INVALID_COST")
    if isinstance(value, float) and not math.isfinite(value):
        raise Tier0Error("fresh SDK cost quote is nonfinite", status="STOP_NONZERO_OR_INVALID_COST")
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal.is_finite() or decimal.is_signed() or decimal != Decimal("0"):
        raise Tier0Error("fresh SDK cost quote is not exact unsigned zero", status="STOP_NONZERO_OR_INVALID_COST")
    return "0"


def acquire_session_bytes(
    client: Any,
    request: SessionRequest,
    *,
    output_path: Path,
    journal: AttemptJournal,
    pre_pair_gate: Callable[[], None],
    progress: Callable[[str], None] | None = None,
) -> str:
    """Execute the sole allowed cost/get-range pair for one session."""

    if output_path.exists() or output_path.is_symlink():
        raise Tier0Error("session output path already exists", status="STOP_EXTERNAL_PATH")
    if progress is not None:
        progress(f"{request.session}: requesting fresh exact cost quote")
    pre_pair_gate()
    request_sha = request.market_request_sha256
    journal.append(
        "COST_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload={"method": "metadata.get_cost", "parameters": request.market_parameters},
    )
    try:
        quote = client.metadata.get_cost(**request.market_parameters)
    except Exception as exc:  # noqa: BLE001 - redact vendor exception text
        journal.append(
            "COST_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={"error_class": type(exc).__name__},
        )
        raise Tier0Error("fresh SDK cost request failed", status="STOP_VENDOR_COST_CALL") from exc
    try:
        quote_text = exact_zero_quote(quote)
    except Tier0Error:
        safe_observed = repr(quote) if isinstance(quote, (int, float, Decimal)) else type(quote).__name__
        journal.append(
            "COST_CALL_RESULT",
            session=request.session,
            request_sha256=request_sha,
            payload={"observed_sdk_quote": safe_observed, "zero_gate_pass": False},
        )
        raise
    journal.append(
        "COST_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload={"observed_sdk_quote_usd": quote_text, "zero_gate_pass": True},
    )
    # Keep only the required durable result/start markers between the quote and
    # request. In particular, do not risk blocking on stdout in this interval.
    journal.append(
        "TIMESERIES_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload={
            "method": "timeseries.get_range",
            "parameters": {**request.market_parameters, "stype_out": EXPECTED_STYPE_OUT, "limit": None},
            "output_relative": str(output_path.name),
            "immediately_preceding_quote_usd": quote_text,
        },
    )
    try:
        client.timeseries.get_range(
            **request.market_parameters,
            stype_out=EXPECTED_STYPE_OUT,
            limit=None,
            path=output_path,
        )
    except Exception as exc:  # noqa: BLE001 - redact vendor exception text
        journal.append(
            "TIMESERIES_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={"error_class": type(exc).__name__, "actual_vendor_invoice_cost_usd": "UNKNOWN"},
        )
        raise Tier0Error("time-series stream failed", status="STOP_VENDOR_TIMESERIES_CALL") from exc
    output_metadata = output_path.lstat() if output_path.exists() or output_path.is_symlink() else None
    parent_metadata = output_path.parent.lstat()
    if (
        output_metadata is None
        or not stat.S_ISREG(output_metadata.st_mode)
        or stat.S_ISLNK(output_metadata.st_mode)
        or output_metadata.st_nlink != 1
        or output_metadata.st_dev != parent_metadata.st_dev
        or output_metadata.st_size <= 0
    ):
        journal.append(
            "TIMESERIES_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={"error_class": "MissingOrEmptyOutput", "actual_vendor_invoice_cost_usd": "UNKNOWN"},
        )
        raise Tier0Error("time-series stream did not produce a regular file", status="STOP_VENDOR_TIMESERIES_CALL")
    descriptor = os.open(output_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(output_path.parent)
    journal.append(
        "TIMESERIES_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload={
            "observed_sdk_quote_usd": quote_text,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "compressed_bytes": output_path.stat().st_size,
            "dbn_file_sha256": file_sha256(output_path),
        },
    )
    if progress is not None:
        progress(f"{request.session}: exact zero-gated DBN stream durably recorded")
    return quote_text


def _timestamp_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Tier0Error("request timestamp is not timezone-aware", status="STOP_SCOPE_OR_SEAL_DRIFT")
    seconds = int(parsed.timestamp())
    return seconds * 1_000_000_000 + parsed.microsecond * 1_000


def validate_dbn_metadata(store: Any, request: SessionRequest) -> dict[str, Any]:
    """Validate only the file header/mappings; record streaming happens separately."""

    metadata = store.metadata
    scalars = {
        "dbn_version": int(metadata.version),
        "ts_out": bool(metadata.ts_out),
        "dataset": str(store.dataset),
        "schema": str(store.schema),
        "stype_in": str(store.stype_in),
        "stype_out": str(store.stype_out),
        "start_ns": int(metadata.start),
        "end_ns": int(metadata.end),
        "limit": store.limit,
        "partial": list(metadata.partial),
        "not_found": list(metadata.not_found),
    }
    expected = {
        "ts_out": False,
        "dataset": EXPECTED_DATASET,
        "schema": EXPECTED_SCHEMA,
        "stype_in": EXPECTED_STYPE_IN,
        "stype_out": EXPECTED_STYPE_OUT,
        "start_ns": _timestamp_ns(request.start),
        "end_ns": _timestamp_ns(request.end),
        "limit": None,
        "partial": [],
        "not_found": [],
    }
    if not (1 <= scalars["dbn_version"] <= 3):
        raise Tier0Error(f"{request.session}: unsupported DBN metadata version", status="STOP_DBN_METADATA")
    if {key: value for key, value in scalars.items() if key != "dbn_version"} != expected:
        raise Tier0Error(f"{request.session}: DBN metadata scalar/request mismatch", status="STOP_DBN_METADATA")
    if list(store.symbols) != list(request.symbols):
        raise Tier0Error(f"{request.session}: DBN metadata symbol list mismatch", status="STOP_DBN_METADATA")
    raw_mappings = store.mappings
    if not isinstance(raw_mappings, dict) or sorted(raw_mappings) != list(request.symbols):
        raise Tier0Error(f"{request.session}: DBN mapping key population mismatch", status="STOP_DBN_METADATA")
    expected_by_symbol = {symbol: instrument_id for instrument_id, symbol in request.expected_mappings}
    flattened: dict[str, int] = {}
    mapping_intervals: dict[str, list[dict[str, str]]] = {}
    for symbol in request.symbols:
        intervals = raw_mappings.get(symbol)
        if not isinstance(intervals, list) or len(intervals) != 1:
            raise Tier0Error(f"{request.session}: DBN mapping is not one interval", status="STOP_DBN_METADATA")
        interval = intervals[0]
        instrument = str(interval.get("symbol", ""))
        start_date = str(interval.get("start_date", ""))
        end_date = str(interval.get("end_date", ""))
        if not instrument.isdigit() or int(instrument) != expected_by_symbol[symbol]:
            raise Tier0Error(f"{request.session}: DBN instrument mapping differs from Job-50", status="STOP_DBN_METADATA")
        if not (start_date <= request.session < end_date):
            raise Tier0Error(f"{request.session}: DBN mapping interval misses session", status="STOP_DBN_METADATA")
        flattened[symbol] = int(instrument)
        mapping_intervals[symbol] = [{"start_date": start_date, "end_date": end_date, "instrument_id": instrument}]
    if len(set(flattened.values())) != len(flattened):
        raise Tier0Error(f"{request.session}: DBN metadata mapping is not one-to-one", status="STOP_DBN_METADATA")
    return {
        **scalars,
        "symbols": list(request.symbols),
        "symbol_count": len(request.symbols),
        "mappings": mapping_intervals,
        "mapping_sha256": json_sha256(mapping_intervals),
    }


def stream_dbn_qc(
    path: Path,
    request: SessionRequest,
    *,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 10_000_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode a DBN file record-by-record with no session-sized frame."""

    try:
        import databento
        from databento.common.error import BentoWarning
        from v5.research.cmbp_stream import CmbpStreamDecoder
    except Exception as exc:  # noqa: BLE001
        raise Tier0Error("streaming decoder dependencies unavailable", status="STOP_SDK_DRIFT") from exc
    try:
        store = databento.DBNStore.from_file(path)
        metadata_summary = validate_dbn_metadata(store, request)
        decoder = CmbpStreamDecoder(
            session=request.session,
            expected_mappings=request.mapping_by_instrument,
            source_kind="historical",
            window_start_ns=_timestamp_ns(request.start),
            window_end_ns=_timestamp_ns(request.end),
        )
        decoder.preload_historical_mappings(store.mappings)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BentoWarning)
            if progress is None:
                decoder.consume(store)
            else:
                seen = 0
                next_progress = progress_every
                for record in store:
                    decoder.accept(record)
                    seen += 1
                    if seen >= next_progress:
                        progress(seen)
                        next_progress += progress_every
        summary = decoder.finalize(
            expected_record_count=request.expected_record_count,
            require_all_expected_mappings=True,
            require_all_expected_instruments=True,
        )
    except Tier0Error:
        raise
    except Exception as exc:  # noqa: BLE001 - decode exception text can be huge/vendor-derived
        raise Tier0Error(
            f"{request.session}: DBN streaming QC failed ({type(exc).__name__})",
            status="STOP_DBN_STREAM_QC",
        ) from exc
    summary_payload = asdict(summary) if hasattr(summary, "__dataclass_fields__") else dict(summary)
    if int(summary_payload.get("cmbp1_records", -1)) != request.expected_record_count:
        raise Tier0Error(f"{request.session}: decoder/census record count mismatch", status="STOP_RECORD_RECONCILIATION")
    return metadata_summary, summary_payload


def build_session_qc(
    *,
    request: SessionRequest,
    data_path: Path,
    quote_usd: str,
    metadata_summary: Mapping[str, Any],
    decoder_summary: Mapping[str, Any],
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    attempt_id: str,
    journal_record: Mapping[str, Any],
) -> dict[str, Any]:
    if quote_usd != "0":
        raise Tier0Error("session QC cannot bind a nonzero quote", status="STOP_NONZERO_OR_INVALID_COST")
    receipt: dict[str, Any] = {
        "artifact_type": SESSION_QC_ARTIFACT,
        "schema_version": "v5.job51-session-qc.v1",
        "job_id": JOB_ID,
        "status": "JOB51_SESSION_QC_PASS",
        "session": request.session,
        "scope_sha256": EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": EXPECTED_SCOPE_FILE_SHA256,
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "readiness_receipt_file_sha256": readiness_file_sha256,
        "program_contract_sha256": readiness["program_contract_sha256"],
        "request": {
            "market_parameters": request.market_parameters,
            "market_request_sha256": request.market_request_sha256,
            "cost_request_sha256": request.cost_request_sha256,
            "record_count_request_sha256": request.record_count_request_sha256,
            "symbology_request_sha256": request.symbology_request_sha256,
            "expected_record_count": request.expected_record_count,
            "expected_mappings": [
                {"instrument_id": instrument_id, "raw_symbol": symbol}
                for instrument_id, symbol in request.expected_mappings
            ],
        },
        "cost": {
            "fresh_observed_sdk_quote_usd": quote_usd,
            "zero_gate_pass": True,
            "acquisition_initiated": True,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "zero_quote_is_atomic_invoice_lock": False,
        },
        "raw_dbn": {
            "file_name": data_path.name,
            "file_sha256": file_sha256(data_path),
            "compressed_bytes": data_path.stat().st_size,
        },
        "dbn_metadata": dict(metadata_summary),
        "decoder": dict(decoder_summary),
        "source_attempt": {
            "attempt_id": attempt_id,
            "journal_sequence": journal_record["sequence"],
            "journal_record_hash": journal_record["record_hash"],
            "event": journal_record["event"],
        },
        "qc_law": {
            "streaming_no_full_session_pandas": True,
            "causal_clock": "ts_recv",
            "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
            "receive_time_ties_excluded": True,
            "historical_connection_telemetry": "UNKNOWN_NOT_PRESENT_IN_FILE_TRANSPORT",
        },
    }
    receipt["session_qc_sha256"] = self_hash(receipt, "session_qc_sha256")
    return receipt


def validate_session_bundle(
    bundle_dir: Path,
    *,
    request: SessionRequest,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    volume: VolumeIdentity,
    source_record_lookup: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    ensure_nofollow_directory(bundle_dir, volume=volume)
    children = sorted(path.name for path in bundle_dir.iterdir())
    if children != ["SESSION_QC_V1.json", "data.cmbp-1.dbn.zst"]:
        raise Tier0Error(f"{request.session}: final bundle file population mismatch", status="STOP_SESSION_BUNDLE")
    data_path = bundle_dir / "data.cmbp-1.dbn.zst"
    qc_path = bundle_dir / "SESSION_QC_V1.json"
    for path in (data_path, qc_path):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_dev != volume.st_dev:
            raise Tier0Error(f"{request.session}: unsafe final bundle file", status="STOP_SESSION_BUNDLE")
    receipt = strict_json(qc_path)
    expected_top_level = {
        "artifact_type",
        "schema_version",
        "job_id",
        "status",
        "session",
        "scope_sha256",
        "scope_file_sha256",
        "readiness_receipt_sha256",
        "readiness_receipt_file_sha256",
        "program_contract_sha256",
        "request",
        "cost",
        "raw_dbn",
        "dbn_metadata",
        "decoder",
        "source_attempt",
        "qc_law",
        "session_qc_sha256",
    }
    if set(receipt) != expected_top_level:
        raise Tier0Error(f"{request.session}: session QC field population invalid", status="STOP_SESSION_BUNDLE")
    if (
        receipt.get("artifact_type") != SESSION_QC_ARTIFACT
        or receipt.get("schema_version") != "v5.job51-session-qc.v1"
        or receipt.get("job_id") != JOB_ID
        or receipt.get("status") != "JOB51_SESSION_QC_PASS"
    ):
        raise Tier0Error(f"{request.session}: session QC status invalid", status="STOP_SESSION_BUNDLE")
    if receipt.get("session_qc_sha256") != self_hash(receipt, "session_qc_sha256"):
        raise Tier0Error(f"{request.session}: session QC self-hash mismatch", status="STOP_SESSION_BUNDLE")
    expected = {
        "session": request.session,
        "scope_sha256": EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": EXPECTED_SCOPE_FILE_SHA256,
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "readiness_receipt_file_sha256": readiness_file_sha256,
        "program_contract_sha256": readiness["program_contract_sha256"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise Tier0Error(f"{request.session}: session QC {field} mismatch", status="STOP_SESSION_BUNDLE")
    raw = receipt.get("raw_dbn", {})
    expected_raw = {
        "file_name": data_path.name,
        "file_sha256": file_sha256(data_path),
        "compressed_bytes": data_path.stat().st_size,
    }
    if raw != expected_raw:
        raise Tier0Error(f"{request.session}: session QC raw DBN binding mismatch", status="STOP_SESSION_BUNDLE")
    expected_request_projection = {
        "market_parameters": request.market_parameters,
        "market_request_sha256": request.market_request_sha256,
        "cost_request_sha256": request.cost_request_sha256,
        "record_count_request_sha256": request.record_count_request_sha256,
        "symbology_request_sha256": request.symbology_request_sha256,
        "expected_record_count": request.expected_record_count,
        "expected_mappings": [
            {"instrument_id": instrument_id, "raw_symbol": symbol}
            for instrument_id, symbol in request.expected_mappings
        ],
    }
    if receipt.get("request") != expected_request_projection:
        raise Tier0Error(f"{request.session}: session QC request projection mismatch", status="STOP_SESSION_BUNDLE")
    expected_cost = {
        "fresh_observed_sdk_quote_usd": "0",
        "zero_gate_pass": True,
        "acquisition_initiated": True,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "zero_quote_is_atomic_invoice_lock": False,
    }
    if receipt.get("cost") != expected_cost:
        raise Tier0Error(f"{request.session}: session QC quote is not zero", status="STOP_SESSION_BUNDLE")
    expected_qc_law = {
        "streaming_no_full_session_pandas": True,
        "causal_clock": "ts_recv",
        "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
        "receive_time_ties_excluded": True,
        "historical_connection_telemetry": "UNKNOWN_NOT_PRESENT_IN_FILE_TRANSPORT",
    }
    if receipt.get("qc_law") != expected_qc_law:
        raise Tier0Error(f"{request.session}: session QC law drifted", status="STOP_SESSION_BUNDLE")
    decoder = receipt.get("decoder", {})
    decoded = decoder.get("cmbp1_records")
    decoder_invariants = {
        "artifact_type": "JOB51_CMBP_STREAM_SUMMARY_V1",
        "session": request.session,
        "source_kind": "historical",
        "window_start_ns": _timestamp_ns(request.start),
        "window_end_ns": _timestamp_ns(request.end),
        "cmbp1_records": request.expected_record_count,
        "expected_cmbp1_records": request.expected_record_count,
        "record_count_reconciled": True,
        "mapping_reconciled": True,
        "all_expected_instruments_seen": True,
        "all_causal_priors_strict": True,
        "tied_receive_priors_excluded": True,
        "global_receive_regressions": 0,
        "classification_reconciled": True,
    }
    for field, value in decoder_invariants.items():
        if decoder.get(field) != value:
            raise Tier0Error(f"{request.session}: decoder invariant {field} mismatch", status="STOP_SESSION_BUNDLE")

    count_fields = (
        "mapping_records",
        "total_records_accepted",
        "cmbp1_records",
        "expected_cmbp1_records",
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
    if any(isinstance(decoder.get(field), bool) or not isinstance(decoder.get(field), int) or decoder[field] < 0 for field in count_fields):
        raise Tier0Error(f"{request.session}: decoder contains an invalid counter", status="STOP_SESSION_BUNDLE")
    expected_mapping_payload = [
        {"instrument_id": instrument_id, "raw_symbol": symbol}
        for instrument_id, symbol in request.expected_mappings
    ]
    if decoder.get("expected_mappings") != expected_mapping_payload or decoder.get("active_mappings") != expected_mapping_payload:
        raise Tier0Error(f"{request.session}: decoder mapping summary mismatch", status="STOP_SESSION_BUNDLE")
    seen = decoder.get("seen_instrument_ids")
    if seen != [instrument_id for instrument_id, _symbol in request.expected_mappings]:
        raise Tier0Error(f"{request.session}: decoder seen-instrument population mismatch", status="STOP_SESSION_BUNDLE")
    trade_total = int(decoder.get("trade_records", -1))
    if trade_total != (
        int(decoder.get("strict_prior_trades", -1))
        + int(decoder.get("tied_prior_trades_excluded", -1))
        + int(decoder.get("no_prior_trades", -1))
    ):
        raise Tier0Error(f"{request.session}: decoder causal trade counters do not reconcile", status="STOP_SESSION_BUNDLE")
    exclusion_fields = (
        "undefined_trade_price_excluded",
        "missing_prior_book_excluded",
        "undefined_prior_book_excluded",
        "locked_prior_book_excluded",
        "crossed_prior_book_excluded",
        "trade_bad_ts_recv_excluded",
        "prior_bad_ts_recv_excluded",
        "prior_maybe_bad_book_excluded",
    )
    exclusions = sum(int(decoder[field]) for field in exclusion_fields)
    if int(decoder["signed_trades"]) != int(decoder["at_bid_trades"]) + int(decoder["at_ask_trades"]):
        raise Tier0Error(f"{request.session}: decoder signed counters do not reconcile", status="STOP_SESSION_BUNDLE")
    if trade_total != sum(
        int(decoder[field])
        for field in ("at_bid_trades", "at_ask_trades", "inside_trades", "outside_trades", "ambiguous_trades")
    ):
        raise Tier0Error(f"{request.session}: decoder classification counters do not reconcile", status="STOP_SESSION_BUNDLE")
    if int(decoder["strict_prior_trades"]) != (
        int(decoder["at_bid_trades"])
        + int(decoder["at_ask_trades"])
        + int(decoder["inside_trades"])
        + int(decoder["outside_trades"])
        + exclusions
    ):
        raise Tier0Error(f"{request.session}: decoder strict-prior counters do not reconcile", status="STOP_SESSION_BUNDLE")
    if int(decoder["ambiguous_trades"]) != (
        int(decoder["tied_prior_trades_excluded"]) + int(decoder["no_prior_trades"]) + exclusions
    ):
        raise Tier0Error(f"{request.session}: decoder ambiguous counters do not reconcile", status="STOP_SESSION_BUNDLE")
    if int(decoder["total_records_accepted"]) != (
        int(decoder["cmbp1_records"]) + int(decoder["mapping_records"]) + int(decoder["system_records"])
    ):
        raise Tier0Error(f"{request.session}: decoder total record counters do not reconcile", status="STOP_SESSION_BUNDLE")
    if (
        decoder.get("disconnect_count") != 0
        or decoder.get("reconnect_count") != 0
        or decoder.get("gap_count") != 0
        or decoder.get("book_state_clear_count") != 0
        or decoder.get("gaps_with_known_bounds") != 0
        or decoder.get("total_known_gap_ns") != 0
        or decoder.get("max_known_gap_ns") is not None
        or decoder.get("explicit_connection_telemetry") != "UNKNOWN"
    ):
        raise Tier0Error(f"{request.session}: historical connection telemetry is invalid", status="STOP_SESSION_BUNDLE")
    first_recv = decoder.get("first_ts_recv")
    last_recv = decoder.get("last_ts_recv")
    if (
        isinstance(first_recv, bool)
        or not isinstance(first_recv, int)
        or isinstance(last_recv, bool)
        or not isinstance(last_recv, int)
        or not (_timestamp_ns(request.start) <= first_recv <= last_recv < _timestamp_ns(request.end))
    ):
        raise Tier0Error(f"{request.session}: decoder receive bounds are invalid", status="STOP_SESSION_BUNDLE")

    def named_count_sum(field: str, *, expected_names: set[str] | None = None) -> int:
        values = decoder.get(field)
        if not isinstance(values, list):
            raise Tier0Error(f"{request.session}: decoder {field} is not a list", status="STOP_SESSION_BUNDLE")
        names: set[str] = set()
        total = 0
        for item in values:
            if not isinstance(item, dict) or set(item) != {"name", "count"}:
                raise Tier0Error(f"{request.session}: decoder {field} entry invalid", status="STOP_SESSION_BUNDLE")
            name = item.get("name")
            count = item.get("count")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise Tier0Error(f"{request.session}: decoder {field} counter invalid", status="STOP_SESSION_BUNDLE")
            names.add(name)
            total += count
        if expected_names is not None and names != expected_names:
            raise Tier0Error(f"{request.session}: decoder {field} names drifted", status="STOP_SESSION_BUNDLE")
        return total

    if (
        named_count_sum("action_counts") != request.expected_record_count
        or named_count_sum("side_counts") != request.expected_record_count
        or named_count_sum("flag_value_counts") != request.expected_record_count
        or named_count_sum("system_code_counts") != int(decoder["system_records"])
    ):
        raise Tier0Error(f"{request.session}: decoder named counters do not reconcile", status="STOP_SESSION_BUNDLE")
    known_flag_names = {"LAST", "TOB", "SNAPSHOT", "MBP", "BAD_TS_RECV", "MAYBE_BAD_BOOK", "PUBLISHER_SPECIFIC"}
    named_count_sum("flag_counts", expected_names=known_flag_names)
    if int(decoder["heartbeat_records"]) > int(decoder["system_records"]):
        raise Tier0Error(f"{request.session}: decoder heartbeat count is impossible", status="STOP_SESSION_BUNDLE")
    source = receipt.get("source_attempt", {})
    if (
        not isinstance(source, dict)
        or set(source) != {"attempt_id", "journal_sequence", "journal_record_hash", "event"}
        or source.get("event") != "TIMESERIES_CALL_RESULT"
        or UUID4_RE.fullmatch(str(source.get("attempt_id"))) is None
        or isinstance(source.get("journal_sequence"), bool)
        or not isinstance(source.get("journal_sequence"), int)
        or source["journal_sequence"] < 0
        or SHA256_RE.fullmatch(str(source.get("journal_record_hash"))) is None
    ):
        raise Tier0Error(f"{request.session}: source-attempt locator is invalid", status="STOP_SESSION_BUNDLE")
    if source_record_lookup is not None:
        key = (str(source.get("attempt_id")), int(source.get("journal_sequence")))
        actual_source = source_record_lookup.get(key)
        expected_source_payload = {
            "observed_sdk_quote_usd": "0",
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "compressed_bytes": raw["compressed_bytes"],
            "dbn_file_sha256": raw["file_sha256"],
        }
        if (
            actual_source is None
            or actual_source.get("event") != "TIMESERIES_CALL_RESULT"
            or actual_source.get("record_hash") != source.get("journal_record_hash")
            or actual_source.get("session") != request.session
            or actual_source.get("request_sha256") != request.market_request_sha256
            or actual_source.get("payload") != expected_source_payload
        ):
            raise Tier0Error(f"{request.session}: source-attempt journal binding mismatch", status="STOP_SESSION_BUNDLE")
    # Header revalidation is cheap and catches a file substituted under the same path.
    import databento

    store = databento.DBNStore.from_file(data_path)
    if validate_dbn_metadata(store, request) != receipt.get("dbn_metadata"):
        raise Tier0Error(f"{request.session}: session QC DBN header no longer reconstructs", status="STOP_SESSION_BUNDLE")
    return receipt


def validate_existing_session_population(
    job_root: Path,
    *,
    bundle: ScopeBundle,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    volume: VolumeIdentity,
    source_record_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fail closed on every pre-existing final before any resume vendor call."""

    sessions_root = Path(job_root) / "sessions"
    ensure_nofollow_directory(sessions_root, volume=volume)
    expected = {request.session: request for request in bundle.sessions}
    result: dict[str, dict[str, Any]] = {}
    for child in sorted(sessions_root.iterdir()):
        if child.is_symlink() or not child.is_dir() or child.name not in expected:
            raise Tier0Error("published sessions root contains an unexpected entry", status="STOP_SESSION_BUNDLE")
        request = expected[child.name]
        result[child.name] = validate_session_bundle(
            child,
            request=request,
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha256,
            volume=volume,
            source_record_lookup=source_record_lookup,
        )
    return result


def publish_session_bundle(staging_dir: Path, final_dir: Path, *, volume: VolumeIdentity) -> None:
    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)
    ensure_nofollow_directory(staging_dir, volume=volume)
    if final_dir.exists() or final_dir.is_symlink():
        raise Tier0Error("final session bundle already exists", status="STOP_SESSION_BUNDLE")
    if staging_dir.parent.stat().st_dev != final_dir.parent.stat().st_dev:
        raise Tier0Error("session publication would cross devices", status="STOP_EXTERNAL_PATH")
    os.rename(staging_dir, final_dir)
    fsync_directory(final_dir.parent)
    ensure_nofollow_directory(final_dir, volume=volume)


def summarize_attempts(job_root: Path, allowed_requests: Sequence[SessionRequest]) -> dict[str, Any]:
    """Reconstruct every attempt under the frozen Job-51 destination."""

    attempts_root = Path(job_root) / "attempts"
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise Tier0Error("attempt root is unsafe", status="STOP_JOURNAL_INVALID")
    summaries: list[dict[str, Any]] = []
    cost_starts = timeseries_starts = timeseries_results = 0
    session_attempts: dict[str, int] = {}
    nonzero_or_invalid_observed = False
    allowed = {request.session: request for request in allowed_requests}
    if set(allowed) != {request.session for request in allowed_requests} or len(allowed) != EXPECTED_SESSION_COUNT:
        raise Tier0Error("aggregate allowed-request population is invalid", status="STOP_AGGREGATE_QC")
    children = sorted(attempts_root.iterdir())
    for child in children:
        if child.is_symlink() or not child.is_dir() or UUID4_RE.fullmatch(child.name) is None:
            raise Tier0Error("attempt root contains an unexpected entry", status="STOP_JOURNAL_INVALID")
    all_timeseries_results: list[dict[str, Any]] = []
    for attempt_dir in children:
        allowed_attempt_entries = {
            "ACQUISITION_JOURNAL_V1.jsonl",
            "JOURNAL_WATERMARK_V1.json",
            "call-markers",
            "sessions",
            "ATTEMPT_STOP_V1.json",
        }
        attempt_entries = sorted(attempt_dir.iterdir())
        if any(path.name not in allowed_attempt_entries or path.is_symlink() for path in attempt_entries):
            raise Tier0Error("attempt contains an unexpected or symlinked entry", status="STOP_JOURNAL_INVALID")
        required_attempt_entries = {
            "ACQUISITION_JOURNAL_V1.jsonl",
            "JOURNAL_WATERMARK_V1.json",
            "call-markers",
            "sessions",
        }
        if not required_attempt_entries <= {path.name for path in attempt_entries}:
            raise Tier0Error("attempt evidence population is incomplete", status="STOP_JOURNAL_INVALID")
        for entry in attempt_entries:
            metadata = entry.lstat()
            if entry.name in {"call-markers", "sessions"}:
                valid_type = stat.S_ISDIR(metadata.st_mode)
            else:
                valid_type = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
            if not valid_type or metadata.st_dev != attempt_dir.stat().st_dev:
                raise Tier0Error("attempt contains an unsafe evidence entry", status="STOP_JOURNAL_INVALID")
        staging_root = attempt_dir / "sessions"
        if not staging_root.is_dir():
            raise Tier0Error("attempt staging root is unsafe", status="STOP_JOURNAL_INVALID")
        for staging in sorted(staging_root.iterdir()):
            if staging.is_symlink() or not staging.is_dir() or not staging.name.endswith(".bundle.part"):
                raise Tier0Error("attempt staging root contains an unexpected entry", status="STOP_JOURNAL_INVALID")
            staged_session = staging.name.removesuffix(".bundle.part")
            if staged_session not in allowed:
                raise Tier0Error("attempt contains an out-of-scope staging bundle", status="STOP_JOURNAL_INVALID")
            for staged_file in sorted(staging.iterdir()):
                metadata = staged_file.lstat()
                if (
                    staged_file.name not in {"data.cmbp-1.dbn.zst", "SESSION_QC_V1.json"}
                    or stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_dev != staging.stat().st_dev
                ):
                    raise Tier0Error("attempt staging bundle contains an unsafe entry", status="STOP_JOURNAL_INVALID")
        verified = verify_attempt_journal(attempt_dir)
        records = verified.pop("records")
        stop_path = attempt_dir / "ATTEMPT_STOP_V1.json"
        if stop_path.exists() or stop_path.is_symlink():
            stop_receipt = strict_json(stop_path)
            expected_stop_fields = {
                "artifact_type",
                "schema_version",
                "attempt_id",
                "scope_sha256",
                "status",
                "error_class",
                "actual_vendor_invoice_cost_usd",
                "journal_terminal_sequence",
                "journal_terminal_head",
                "stop_sha256",
            }
            terminal = records[-1]
            if (
                set(stop_receipt) != expected_stop_fields
                or stop_receipt.get("artifact_type") != "JOB51_ATTEMPT_STOP_V1"
                or stop_receipt.get("schema_version") != "v5.job51-attempt-stop.v1"
                or stop_receipt.get("attempt_id") != attempt_dir.name
                or stop_receipt.get("scope_sha256") != EXPECTED_SCOPE_SHA256
                or stop_receipt.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
                or stop_receipt.get("stop_sha256") != self_hash(stop_receipt, "stop_sha256")
                or terminal.get("event") != "ATTEMPT_STOP"
                or stop_receipt.get("status") != terminal.get("payload", {}).get("status")
                or stop_receipt.get("error_class") != terminal.get("payload", {}).get("error_class")
                or stop_receipt.get("journal_terminal_sequence") != terminal.get("sequence")
                or stop_receipt.get("journal_terminal_head") != terminal.get("record_hash")
            ):
                raise Tier0Error("attempt stop receipt does not bind its terminal journal", status="STOP_JOURNAL_INVALID")
            verified["attempt_stop_file_sha256"] = file_sha256(stop_path)
        header = records[0]
        if header.get("event") != "ATTEMPT_START" or header.get("payload", {}).get("scope_sha256") != EXPECTED_SCOPE_SHA256:
            raise Tier0Error("attempt root contains a different/unbound scope", status="STOP_JOURNAL_INVALID")
        pending_cost: dict[str, Any] | None = None
        last_cost_result: dict[str, Any] | None = None
        pending_timeseries: dict[str, Any] | None = None
        per_attempt = {"cost_starts": 0, "timeseries_starts": 0, "timeseries_results": 0}
        for record_index, record in enumerate(records):
            event = record.get("event")
            session = record.get("session")
            if event not in ALLOWED_JOURNAL_EVENTS:
                raise Tier0Error("attempt contains an unknown journal event", status="STOP_JOURNAL_INVALID")
            if event == "ATTEMPT_START" and record.get("sequence") != 0:
                raise Tier0Error("attempt contains a duplicate start event", status="STOP_JOURNAL_INVALID")
            if event == "ATTEMPT_STOP" and record is not records[-1]:
                raise Tier0Error("attempt stop is not terminal", status="STOP_JOURNAL_INVALID")
            if event == "ATTEMPT_SEALED_FOR_AGGREGATE" and record is not records[-1]:
                if record_index != len(records) - 2 or records[-1].get("event") != "ATTEMPT_STOP":
                    raise Tier0Error("aggregate seal has an invalid successor", status="STOP_JOURNAL_INVALID")
            if event in CALL_EVENTS:
                request = allowed.get(str(session))
                if request is None or record.get("request_sha256") != request.market_request_sha256:
                    raise Tier0Error("attempt contains an out-of-scope vendor call", status="STOP_JOURNAL_INVALID")
            if event == "COST_CALL_START":
                if pending_cost is not None or last_cost_result is not None or pending_timeseries is not None:
                    raise Tier0Error("cost call began before prior call state closed", status="STOP_JOURNAL_INVALID")
                request = allowed[str(session)]
                if record.get("payload") != {"method": "metadata.get_cost", "parameters": request.market_parameters}:
                    raise Tier0Error("cost call payload widened or drifted", status="STOP_JOURNAL_INVALID")
                cost_starts += 1
                per_attempt["cost_starts"] += 1
                pending_cost = record
            elif event == "COST_CALL_RESULT":
                if (
                    pending_cost is None
                    or pending_cost.get("session") != session
                    or pending_cost.get("request_sha256") != record.get("request_sha256")
                ):
                    raise Tier0Error("cost result lacks a matching start", status="STOP_JOURNAL_INVALID")
                pending_cost = None
                last_cost_result = record
                payload = record.get("payload", {})
                if payload.get("zero_gate_pass") is True:
                    if payload != {"observed_sdk_quote_usd": "0", "zero_gate_pass": True}:
                        raise Tier0Error("zero cost result payload is malformed", status="STOP_JOURNAL_INVALID")
                else:
                    if set(payload) != {"observed_sdk_quote", "zero_gate_pass"} or payload.get("zero_gate_pass") is not False:
                        raise Tier0Error("nonzero cost result payload is malformed", status="STOP_JOURNAL_INVALID")
                    nonzero_or_invalid_observed = True
            elif event == "COST_CALL_ERROR":
                if (
                    pending_cost is None
                    or pending_cost.get("session") != session
                    or pending_cost.get("request_sha256") != record.get("request_sha256")
                ):
                    raise Tier0Error("cost error lacks a matching start", status="STOP_JOURNAL_INVALID")
                error_payload = record.get("payload")
                if (
                    not isinstance(error_payload, dict)
                    or set(error_payload) != {"error_class"}
                    or not isinstance(error_payload.get("error_class"), str)
                    or not error_payload["error_class"]
                ):
                    raise Tier0Error("cost error payload is malformed", status="STOP_JOURNAL_INVALID")
                pending_cost = None
                last_cost_result = None
            elif event == "TIMESERIES_CALL_START":
                timeseries_starts += 1
                per_attempt["timeseries_starts"] += 1
                session = str(record.get("session"))
                session_attempts[session] = session_attempts.get(session, 0) + 1
                if (
                    last_cost_result is None
                    or last_cost_result.get("session") != record.get("session")
                    or last_cost_result.get("request_sha256") != record.get("request_sha256")
                    or last_cost_result.get("payload", {}).get("zero_gate_pass") is not True
                    or last_cost_result.get("sequence") != record.get("sequence") - 1
                ):
                    raise Tier0Error("time-series attempt lacks an immediate identical zero quote", status="STOP_JOURNAL_INVALID")
                request = allowed[str(session)]
                expected_payload = {
                    "method": "timeseries.get_range",
                    "parameters": {**request.market_parameters, "stype_out": EXPECTED_STYPE_OUT, "limit": None},
                    "output_relative": "data.cmbp-1.dbn.zst",
                    "immediately_preceding_quote_usd": "0",
                }
                if record.get("payload") != expected_payload or pending_timeseries is not None:
                    raise Tier0Error("time-series call payload widened or call overlapped", status="STOP_JOURNAL_INVALID")
                pending_timeseries = record
                last_cost_result = None
            elif event == "TIMESERIES_CALL_RESULT":
                result_payload = record.get("payload")
                if (
                    pending_timeseries is None
                    or pending_timeseries.get("session") != session
                    or pending_timeseries.get("request_sha256") != record.get("request_sha256")
                    or not isinstance(result_payload, dict)
                    or set(result_payload) != {
                        "observed_sdk_quote_usd",
                        "actual_vendor_invoice_cost_usd",
                        "compressed_bytes",
                        "dbn_file_sha256",
                    }
                    or result_payload.get("observed_sdk_quote_usd") != "0"
                    or result_payload.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
                    or isinstance(result_payload.get("compressed_bytes"), bool)
                    or not isinstance(result_payload.get("compressed_bytes"), int)
                    or result_payload["compressed_bytes"] <= 0
                    or SHA256_RE.fullmatch(str(result_payload.get("dbn_file_sha256"))) is None
                ):
                    raise Tier0Error("time-series result lacks a matching zero-gated start", status="STOP_JOURNAL_INVALID")
                pending_timeseries = None
                timeseries_results += 1
                per_attempt["timeseries_results"] += 1
                all_timeseries_results.append(record)
            elif event == "TIMESERIES_CALL_ERROR":
                if (
                    pending_timeseries is None
                    or pending_timeseries.get("session") != session
                    or pending_timeseries.get("request_sha256") != record.get("request_sha256")
                ):
                    raise Tier0Error("time-series error lacks a matching start", status="STOP_JOURNAL_INVALID")
                error_payload = record.get("payload")
                if (
                    not isinstance(error_payload, dict)
                    or set(error_payload) != {"error_class", "actual_vendor_invoice_cost_usd"}
                    or not isinstance(error_payload.get("error_class"), str)
                    or not error_payload["error_class"]
                    or error_payload.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
                ):
                    raise Tier0Error("time-series error payload is malformed", status="STOP_JOURNAL_INVALID")
                pending_timeseries = None
            elif event in {"SESSION_PUBLISHED", "SESSION_REUSED"}:
                request = allowed.get(str(session))
                if request is None or record.get("request_sha256") != request.market_request_sha256:
                    raise Tier0Error("attempt cites an out-of-scope session publication", status="STOP_JOURNAL_INVALID")
        summaries.append({**verified, **per_attempt, "terminal_event": records[-1].get("event")})
    return {
        "attempt_count": len(summaries),
        "cost_call_starts": cost_starts,
        "timeseries_call_starts": timeseries_starts,
        "timeseries_call_results": timeseries_results,
        "session_timeseries_attempt_counts": dict(sorted(session_attempts.items())),
        "duplicate_session_attempts": {key: value for key, value in sorted(session_attempts.items()) if value > 1},
        "nonzero_or_invalid_quote_observed": nonzero_or_invalid_observed,
        "timeseries_result_records": all_timeseries_results,
        "attempts": summaries,
    }


def build_aggregate_receipt(
    job_root: Path,
    *,
    bundle: ScopeBundle,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    volume: VolumeIdentity,
) -> dict[str, Any]:
    sessions_root = Path(job_root) / "sessions"
    expected_names = [request.session for request in bundle.sessions]
    session_children = sorted(sessions_root.iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in session_children):
        raise Tier0Error("published sessions root contains an unexpected entry", status="STOP_AGGREGATE_QC")
    actual_names = [path.name for path in session_children]
    if actual_names != expected_names:
        raise Tier0Error("published session directory population mismatch", status="STOP_AGGREGATE_QC")
    attempts = summarize_attempts(job_root, bundle.sessions)
    source_record_lookup = {
        (str(record["attempt_id"]), int(record["sequence"])): record
        for record in attempts["timeseries_result_records"]
    }
    receipts: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    total_records = total_symbols = total_bytes = 0
    source_attempt_ids: set[str] = set()
    decoder_totals = {
        field: 0
        for field in (
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
        )
    }
    max_stream_silence_ns: int | None = None
    max_instrument_silence_ns: int | None = None
    for request in bundle.sessions:
        directory = sessions_root / request.session
        qc = validate_session_bundle(
            directory,
            request=request,
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha256,
            volume=volume,
            source_record_lookup=source_record_lookup,
        )
        receipts.append(qc)
        total_records += request.expected_record_count
        total_symbols += len(request.symbols)
        total_bytes += int(qc["raw_dbn"]["compressed_bytes"])
        source_attempt_ids.add(str(qc["source_attempt"]["attempt_id"]))
        decoder = qc["decoder"]
        for field in decoder_totals:
            decoder_totals[field] += int(decoder[field])
        session_stream_silence = decoder.get("max_stream_silence_ns")
        if isinstance(session_stream_silence, int) and not isinstance(session_stream_silence, bool):
            max_stream_silence_ns = (
                session_stream_silence
                if max_stream_silence_ns is None
                else max(max_stream_silence_ns, session_stream_silence)
            )
        session_instrument_silence = decoder.get("max_instrument_silence_ns")
        if isinstance(session_instrument_silence, int) and not isinstance(session_instrument_silence, bool):
            max_instrument_silence_ns = (
                session_instrument_silence
                if max_instrument_silence_ns is None
                else max(max_instrument_silence_ns, session_instrument_silence)
            )
        files.append(
            {
                "session": request.session,
                "relative_bundle": f"sessions/{request.session}",
                "dbn_file_sha256": qc["raw_dbn"]["file_sha256"],
                "compressed_bytes": qc["raw_dbn"]["compressed_bytes"],
                "session_qc_sha256": qc["session_qc_sha256"],
                "session_qc_file_sha256": file_sha256(directory / "SESSION_QC_V1.json"),
                "decoded_records": request.expected_record_count,
                "symbol_count": len(request.symbols),
                "source_attempt_id": qc["source_attempt"]["attempt_id"],
                "decoder_summary_sha256": json_sha256(decoder),
                "trade_records": decoder["trade_records"],
                "strict_prior_trades": decoder["strict_prior_trades"],
                "tied_prior_trades_excluded": decoder["tied_prior_trades_excluded"],
                "signed_trades": decoder["signed_trades"],
                "global_receive_regressions": decoder["global_receive_regressions"],
                "max_stream_silence_ns": decoder["max_stream_silence_ns"],
            }
        )
    if total_records != EXPECTED_RECORD_COUNT or total_symbols != EXPECTED_SESSION_SYMBOLS:
        raise Tier0Error("published aggregate census totals do not reconcile", status="STOP_AGGREGATE_QC")
    if attempts["nonzero_or_invalid_quote_observed"]:
        raise Tier0Error("a Job-51 attempt observed a nonzero/invalid quote", status="STOP_NONZERO_OR_INVALID_COST")
    if attempts["cost_call_starts"] < EXPECTED_SESSION_COUNT or attempts["timeseries_call_starts"] < EXPECTED_SESSION_COUNT:
        raise Tier0Error("aggregate vendor call population is incomplete", status="STOP_AGGREGATE_QC")
    known_attempt_ids = {str(item["attempt_id"]) for item in attempts["attempts"]}
    if not source_attempt_ids <= known_attempt_ids:
        raise Tier0Error("session QC cites an absent source attempt", status="STOP_AGGREGATE_QC")
    receipt: dict[str, Any] = {
        "artifact_type": AGGREGATE_RECEIPT_ARTIFACT,
        "schema_version": "v5.job51-acquisition-qc-receipt.v1",
        "job_id": JOB_ID,
        "status": "JOB51_TIER0_ACQUISITION_AND_QC_PASS",
        "status_meaning": "EXACT_FROZEN_ACQUISITION_AND_STREAMING_DECODER_QC_ONLY; NO_STRATEGY_OR_TRADING_CLAIM",
        "scope_sha256": EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": EXPECTED_SCOPE_FILE_SHA256,
        "program_contract_sha256": readiness["program_contract_sha256"],
        "program_contract_file_sha256": readiness["program_contract_file_sha256"],
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "readiness_receipt_file_sha256": readiness_file_sha256,
        "volume_identity": asdict(volume),
        "completeness": {
            "expected_sessions": EXPECTED_SESSION_COUNT,
            "published_sessions": len(files),
            "expected_records": EXPECTED_RECORD_COUNT,
            "decoded_records": total_records,
            "expected_session_symbol_memberships": EXPECTED_SESSION_SYMBOLS,
            "mapped_session_symbol_memberships": total_symbols,
            "compressed_bytes": total_bytes,
            "per_session_census_reconciliation": True,
        },
        "cost_boundary": {
            "all_observed_sdk_quotes_usd": "0",
            "no_timeseries_request_after_nonzero_or_invalid_quote": True,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "zero_quote_is_atomic_invoice_lock": False,
            "acquisition_initiated": True,
        },
        "causal_ordering": {
            "clock": "ts_recv",
            "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
            "receive_time_ties_excluded": True,
            "per_session_decoder_summaries_bound": True,
            "trade_records": decoder_totals["trade_records"],
            "strict_prior_trades": decoder_totals["strict_prior_trades"],
            "tied_prior_trades_excluded": decoder_totals["tied_prior_trades_excluded"],
            "no_prior_trades": decoder_totals["no_prior_trades"],
            "signed_trades": decoder_totals["signed_trades"],
            "at_bid_trades": decoder_totals["at_bid_trades"],
            "at_ask_trades": decoder_totals["at_ask_trades"],
            "inside_trades": decoder_totals["inside_trades"],
            "outside_trades": decoder_totals["outside_trades"],
            "ambiguous_trades": decoder_totals["ambiguous_trades"],
            "global_receive_ties": decoder_totals["global_receive_ties"],
            "global_receive_regressions": decoder_totals["global_receive_regressions"],
            "global_event_regressions_diagnostic": decoder_totals["global_event_regressions"],
            "instrument_event_regressions_diagnostic": decoder_totals["instrument_event_regressions"],
        },
        "gaps_and_reconnects": {
            "historical_transport_telemetry": "UNKNOWN_NOT_PRESENT_IN_DBN_FILES",
            "explicit_decoder_gap_reconnect_controls": True,
            "per_session_observed_silence_diagnostics_bound": True,
            "live_connection_exercised": False,
            "disconnect_count": decoder_totals["disconnect_count"],
            "reconnect_count": decoder_totals["reconnect_count"],
            "gap_count": decoder_totals["gap_count"],
            "book_state_clear_count": decoder_totals["book_state_clear_count"],
            "gaps_with_known_bounds": decoder_totals["gaps_with_known_bounds"],
            "total_known_gap_ns": decoder_totals["total_known_gap_ns"],
            "max_observed_stream_silence_ns": max_stream_silence_ns,
            "max_observed_instrument_silence_ns": max_instrument_silence_ns,
            "system_records": decoder_totals["system_records"],
            "heartbeat_records": decoder_totals["heartbeat_records"],
        },
        "attempt_accounting": attempts,
        "sessions": files,
        "claims": {
            "full_session_pandas_loads": 0,
            "live_vendor_subscriptions": 0,
            "models_fit": 0,
            "strategy_outcomes_read": False,
            "broker_calls": 0,
            "orders_submitted": 0,
            "profitability_or_edge": False,
        },
    }
    receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
    return receipt
