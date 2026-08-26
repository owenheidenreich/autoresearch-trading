"""Offline-only CMBP catalogue scope and metadata-receipt primitives.

This module is deliberately unable to contact a data service.  It freezes a
session/raw-symbol scope, constructs the two metadata request *descriptions*
needed for each covered session, and validates responses supplied later as
plain JSON.  The method names below are data, not executable client calls.

No quote, trade, outcome, return, label, fill, or P&L column is read here.  The
optional ladder scanner reads only each file name and its ``raw_symbol``
column.  Dates before the explicitly declared CMBP event-era boundary remain
in the source population but receive no request rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo


ARTIFACT_VERSION = "v5.cmbp-catalogue-preflight.v1"
SCOPE_ARTIFACT = "CMBP_CATALOGUE_SCOPE_V1"
RESPONSE_ARTIFACT = "CMBP_CATALOGUE_METADATA_RESPONSES_V1"
RECEIPT_ARTIFACT = "CMBP_CATALOGUE_PREFLIGHT_RECEIPT_V1"

DATASET = "OPRA.PILLAR"
SCHEMA = "cmbp-1"
STYPE_IN = "raw_symbol"
COVERAGE_START = date(2023, 3, 28)
CALENDAR_NAME = "XNYS"
CALENDAR_TIMEZONE = "America/New_York"
BOUNDS_SEMANTICS = "[rth_open_utc,rth_close_utc)"

# Modern XNYS early closes that intersect the frozen 2022-06 through 2026-07
# lifecycle ladder.  A direct ladder build fails outside these supported years
# unless the caller supplies an explicit replacement set.
KNOWN_EARLY_CLOSE_SESSIONS = frozenset(
    {
        "2022-11-25",
        "2023-07-03",
        "2023-11-24",
        "2024-07-03",
        "2024-11-29",
        "2024-12-24",
        "2025-07-03",
        "2025-11-28",
        "2025-12-24",
        "2026-07-02",
        "2026-11-27",
        "2026-12-24",
    }
)
SUPPORTED_CALENDAR_YEARS = frozenset(range(2022, 2027))

ALLOWED_FUTURE_METHODS = (
    "metadata.get_dataset_range",
    "symbology.resolve",
    "metadata.get_cost",
    "metadata.get_record_count",
)
GLOBAL_REQUEST_METHODS = ("metadata.get_dataset_range",)
SESSION_REQUEST_METHODS = (
    "symbology.resolve",
    "metadata.get_cost",
    "metadata.get_record_count",
)
# All four shapes must be exercised by the offline rehearsal.  Only the latter
# three are repeated for every event-era session.
REQUEST_METHODS = ALLOWED_FUTURE_METHODS
FORBIDDEN_METHOD_FAMILIES = ("timeseries", "download")

STATUS_CATALOGUE_READY_LOCAL = "CATALOGUE_READY_LOCAL"
LOCAL_GATE_STATUS = "CMBP_CATALOGUE_READY_LOCAL"
STATUS_PREFLIGHT_PASS_ONLY = "PREFLIGHT_PASS_ONLY"
STOP_MISSING_KEY_OR_ENTITLEMENT = "STOP_MISSING_KEY_OR_ENTITLEMENT"
STOP_SYMBOL_OR_EXPIRY_MISMATCH = "STOP_SYMBOL_OR_EXPIRY_MISMATCH"
STOP_SCHEMA_UNAVAILABLE = "STOP_SCHEMA_UNAVAILABLE"
STOP_ZERO_RECORDS = "STOP_ZERO_RECORDS"
STOP_NONFINITE_COST = "STOP_NONFINITE_COST"
STOP_SCOPE_OR_CODE_DRIFT = "STOP_SCOPE_OR_CODE_DRIFT"
STOP_OVER_HARD_CAP = "STOP_OVER_HARD_CAP"
STOP_MISSING_SESSION_RESPONSE = "STOP_MISSING_SESSION_RESPONSE"

STOP_STATUSES = frozenset(
    {
        STOP_MISSING_KEY_OR_ENTITLEMENT,
        STOP_SYMBOL_OR_EXPIRY_MISMATCH,
        STOP_SCHEMA_UNAVAILABLE,
        STOP_ZERO_RECORDS,
        STOP_NONFINITE_COST,
        STOP_SCOPE_OR_CODE_DRIFT,
        STOP_OVER_HARD_CAP,
        STOP_MISSING_SESSION_RESPONSE,
    }
)
RECEIPT_STATUSES = STOP_STATUSES | {STATUS_PREFLIGHT_PASS_ONLY}

RAW_OSI = re.compile(r"^SPXW  (?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")
LADDER_FILE = re.compile(r"^(?P<session>20\d{2}-\d{2}-\d{2})\.parquet$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CataloguePreflightError(ValueError):
    """A local scope or receipt cannot be accepted without widening policy."""

    def __init__(self, status: str, message: str) -> None:
        if status not in STOP_STATUSES:
            raise ValueError(f"unknown catalogue stop status: {status}")
        super().__init__(message)
        self.status = status


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's deterministic JSON encoding."""

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


def self_hash(payload: Mapping[str, Any], field: str) -> str:
    semantic = dict(payload)
    semantic.pop(field, None)
    return json_sha256(semantic)


def _require(condition: bool, status: str, message: str) -> None:
    if not condition:
        raise CataloguePreflightError(status, message)


def _session_date(value: Any) -> date:
    _require(isinstance(value, str), STOP_SCOPE_OR_CODE_DRIFT, "session must be ISO text")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CataloguePreflightError(
            STOP_SCOPE_OR_CODE_DRIFT, f"invalid ISO session {value!r}"
        ) from exc
    _require(parsed.isoformat() == value, STOP_SCOPE_OR_CODE_DRIFT, f"non-canonical session {value!r}")
    _require(parsed.weekday() < 5, STOP_SCOPE_OR_CODE_DRIFT, f"weekend session {value}")
    return parsed


def validate_raw_spxw_osi(raw_symbol: Any, session: str) -> str:
    """Validate one byte-exact 21-character same-day SPXW OSI symbol.

    The two padding spaces after ``SPXW`` are semantically significant.  This
    function never strips them.
    """

    _session_date(session)
    _require(
        isinstance(raw_symbol, str),
        STOP_SYMBOL_OR_EXPIRY_MISMATCH,
        f"{session}: raw symbol is not text",
    )
    _require(
        len(raw_symbol) == 21,
        STOP_SYMBOL_OR_EXPIRY_MISMATCH,
        f"{session}: raw symbol must be exactly 21 characters: {raw_symbol!r}",
    )
    match = RAW_OSI.fullmatch(raw_symbol)
    _require(
        match is not None,
        STOP_SYMBOL_OR_EXPIRY_MISMATCH,
        f"{session}: invalid byte-exact SPXW OSI symbol {raw_symbol!r}",
    )
    assert match is not None
    expected_expiry = _session_date(session).strftime("%y%m%d")
    _require(
        match.group("expiry") == expected_expiry,
        STOP_SYMBOL_OR_EXPIRY_MISMATCH,
        f"{session}: symbol expiry {match.group('expiry')} is not the session",
    )
    _require(
        int(match.group("strike")) > 0,
        STOP_SYMBOL_OR_EXPIRY_MISMATCH,
        f"{session}: OSI strike must be a positive eight-digit integer",
    )
    return raw_symbol


def rth_utc_bounds(session: str, *, early_close: bool) -> tuple[str, str]:
    """Return exact UTC bounds for the half-open XNYS regular-hours interval."""

    day = _session_date(session)
    eastern = ZoneInfo(CALENDAR_TIMEZONE)
    local_open = datetime.combine(day, time(9, 30), tzinfo=eastern)
    local_close = datetime.combine(day, time(13 if early_close else 16, 0), tzinfo=eastern)

    def _utc_text(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return _utc_text(local_open), _utc_text(local_close)


def _validate_hash(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and SHA256.fullmatch(value) is not None,
        STOP_SCOPE_OR_CODE_DRIFT,
        f"{name} is not a lowercase SHA-256",
    )
    return value


def _validate_code_hashes(code_hashes: Mapping[str, Any]) -> dict[str, str]:
    _require(bool(code_hashes), STOP_SCOPE_OR_CODE_DRIFT, "code hashes are empty")
    normalized: dict[str, str] = {}
    for name, value in sorted(code_hashes.items()):
        _require(isinstance(name, str) and bool(name), STOP_SCOPE_OR_CODE_DRIFT, "invalid code path")
        normalized[name] = _validate_hash(value, f"code hash for {name}")
    return normalized


def _validate_dependency_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(bool(value), STOP_SCOPE_OR_CODE_DRIFT, "dependency manifest is empty")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise CataloguePreflightError(
            STOP_SCOPE_OR_CODE_DRIFT, "dependency manifest is not canonical-JSON serializable"
        ) from exc
    return dict(value)


def _calendar_manifest(early_close_sessions: Sequence[str]) -> dict[str, Any]:
    early = sorted(early_close_sessions)
    for session in early:
        _session_date(session)
    _require(len(early) == len(set(early)), STOP_SCOPE_OR_CODE_DRIFT, "duplicate early-close session")
    return {
        "name": CALENDAR_NAME,
        "timezone": CALENDAR_TIMEZONE,
        "open_local": "09:30:00",
        "regular_close_local": "16:00:00",
        "early_close_local": "13:00:00",
        "early_close_sessions": early,
        "bounds_semantics": BOUNDS_SEMANTICS,
    }


def _global_request(method: str) -> dict[str, Any]:
    _require(
        method in GLOBAL_REQUEST_METHODS,
        STOP_SCOPE_OR_CODE_DRIFT,
        f"forbidden global request method {method}",
    )
    request = {"method": method, "parameters": {"dataset": DATASET}}
    request["request_sha256"] = json_sha256(request)
    return request


def _request(session_row: Mapping[str, Any], method: str) -> dict[str, Any]:
    _require(
        method in SESSION_REQUEST_METHODS,
        STOP_SCOPE_OR_CODE_DRIFT,
        f"forbidden per-session request method {method}",
    )
    if method == "symbology.resolve":
        session = _session_date(session_row["session"])
        request = {
            "method": method,
            "parameters": {
                "dataset": DATASET,
                "symbols": list(session_row["symbols"]),
                "stype_in": STYPE_IN,
                "stype_out": "instrument_id",
                "start": session.isoformat(),
                "end": date.fromordinal(session.toordinal() + 1).isoformat(),
            },
        }
        request["request_sha256"] = json_sha256(request)
        return request
    request = {
        "method": method,
        "parameters": {
            "dataset": DATASET,
            "schema": SCHEMA,
            "symbols": list(session_row["symbols"]),
            "stype_in": STYPE_IN,
            "start": session_row["rth_open_utc"],
            "end": session_row["rth_close_utc"],
        },
    }
    request["request_sha256"] = json_sha256(request)
    return request


def _normalize_source_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(manifest, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "manifest is not an object")
    _require(
        manifest.get("coverage_start") == COVERAGE_START.isoformat(),
        STOP_SCHEMA_UNAVAILABLE,
        f"cmbp-1 coverage_start must be explicitly {COVERAGE_START.isoformat()}",
    )
    expected = manifest.get("expected_source_session_count")
    _require(
        isinstance(expected, int) and not isinstance(expected, bool) and expected > 0,
        STOP_SCOPE_OR_CODE_DRIFT,
        "expected_source_session_count must be a positive integer",
    )
    raw_sessions = manifest.get("sessions")
    _require(isinstance(raw_sessions, list) and raw_sessions, STOP_SCOPE_OR_CODE_DRIFT, "sessions are empty")

    calendar = manifest.get("calendar")
    _require(isinstance(calendar, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "calendar declaration is missing")
    expected_calendar = _calendar_manifest(calendar.get("early_close_sessions", []))
    for key, expected_value in expected_calendar.items():
        _require(
            calendar.get(key) == expected_value,
            STOP_SCOPE_OR_CODE_DRIFT,
            f"calendar field {key} drifted",
        )
    early = set(expected_calendar["early_close_sessions"])

    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in raw_sessions:
        _require(isinstance(raw_row, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "session row is not an object")
        session = raw_row.get("session")
        day = _session_date(session)
        _require(session not in seen, STOP_SCOPE_OR_CODE_DRIFT, f"duplicate source session {session}")
        seen.add(session)
        symbols_value = raw_row.get("symbols")
        _require(isinstance(symbols_value, list) and symbols_value, STOP_SYMBOL_OR_EXPIRY_MISMATCH, f"{session}: no raw symbols")
        symbols = [validate_raw_spxw_osi(value, session) for value in symbols_value]
        _require(
            symbols == sorted(symbols) and len(symbols) == len(set(symbols)),
            STOP_SYMBOL_OR_EXPIRY_MISMATCH,
            f"{session}: symbols must be unique and sorted byte-exactly",
        )
        expected_open, expected_close = rth_utc_bounds(session, early_close=session in early)
        _require(
            raw_row.get("rth_open_utc") == expected_open,
            STOP_SCOPE_OR_CODE_DRIFT,
            f"{session}: RTH open is not the frozen XNYS UTC bound",
        )
        _require(
            raw_row.get("rth_close_utc") == expected_close,
            STOP_SCOPE_OR_CODE_DRIFT,
            f"{session}: RTH close is not the frozen exclusive XNYS UTC bound",
        )
        source_filename = raw_row.get("source_filename")
        _require(
            source_filename is None or source_filename == f"{session}.parquet",
            STOP_SCOPE_OR_CODE_DRIFT,
            f"{session}: source filename does not bind the session",
        )
        covered = day >= COVERAGE_START
        sessions.append(
            {
                "session": session,
                "source_filename": source_filename,
                "rth_open_utc": expected_open,
                "rth_close_utc": expected_close,
                "coverage_disposition": (
                    "REQUEST_CANDIDATE"
                    if covered
                    else "EXCLUDED_KNOWN_PRE_COVERAGE"
                ),
                "symbols": symbols,
                "symbol_count": len(symbols),
                "symbol_sha256": json_sha256(symbols),
            }
        )

    sessions.sort(key=lambda row: row["session"])
    _require(
        len(sessions) == expected,
        STOP_SCOPE_OR_CODE_DRIFT,
        f"source-session count drifted: {len(sessions)} != {expected}",
    )
    _require(
        early.issubset(seen),
        STOP_SCOPE_OR_CODE_DRIFT,
        f"early-close declaration contains absent sessions: {sorted(early - seen)}",
    )
    return {
        "coverage_start": COVERAGE_START.isoformat(),
        "expected_source_session_count": expected,
        "calendar": expected_calendar,
        "sessions": sessions,
    }


def prepare_catalogue_declaration(
    manifest: Mapping[str, Any],
    *,
    code_hashes: Mapping[str, str],
    dependency_manifest: Mapping[str, Any],
    source_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze a local scope and its future metadata request descriptions."""

    normalized = _normalize_source_manifest(manifest)
    normalized_code = _validate_code_hashes(code_hashes)
    dependencies = _validate_dependency_manifest(dependency_manifest)
    if source_manifest_file_sha256 is not None:
        _validate_hash(source_manifest_file_sha256, "source manifest file hash")

    rows: list[dict[str, Any]] = []
    for source_row in normalized["sessions"]:
        row = dict(source_row)
        if row["coverage_disposition"] == "REQUEST_CANDIDATE":
            row["requests"] = [_request(row, method) for method in SESSION_REQUEST_METHODS]
        else:
            row["requests"] = []
        rows.append(row)

    session_projection = [
        {
            "session": row["session"],
            "source_filename": row["source_filename"],
            "rth_open_utc": row["rth_open_utc"],
            "rth_close_utc": row["rth_close_utc"],
            "coverage_disposition": row["coverage_disposition"],
        }
        for row in rows
    ]
    symbol_projection = [
        {"session": row["session"], "symbols": row["symbols"]} for row in rows
    ]
    global_requests = [_global_request(method) for method in GLOBAL_REQUEST_METHODS]
    request_projection = {
        "global_requests": global_requests,
        "session_requests": [
            {
                "session": row["session"],
                "requests": row["requests"],
            }
            for row in rows
            if row["requests"]
        ],
    }
    request_sessions = sum(bool(row["requests"]) for row in rows)
    excluded_sessions = len(rows) - request_sessions

    declaration: dict[str, Any] = {
        "artifact_type": SCOPE_ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "status": STATUS_CATALOGUE_READY_LOCAL,
        "local_gate_status": LOCAL_GATE_STATUS,
        "status_meaning": (
            "Local scope and offline request descriptions validate; no external call, "
            "download, acquisition, or outcome access is authorized."
        ),
        "dataset": DATASET,
        "schema": SCHEMA,
        "stype_in": STYPE_IN,
        "coverage_start": normalized["coverage_start"],
        "calendar": normalized["calendar"],
        "network_policy": {
            "offline_processor_network_calls": 0,
            "live_client_imported": False,
            "allowed_future_methods": list(ALLOWED_FUTURE_METHODS),
            "forbidden_method_families": list(FORBIDDEN_METHOD_FAMILIES),
            "timeseries_allowed": False,
            "download_allowed": False,
        },
        "outcome_policy": {
            "outcome_columns_read": False,
            "labels_read": False,
            "returns_read": False,
            "fills_read": False,
            "pnl_read": False,
        },
        "claim_policy": {
            "actual_vendor_availability": "UNKNOWN",
            "exact_available_session_count": "UNKNOWN",
            "exact_vendor_cost": "UNKNOWN",
            "zero_price_boundary": "UNKNOWN",
            "population_event_prevalence": "UNKNOWN",
            "strategy_economics": "NOT_READ",
        },
        "selected_semantic_fixture": {
            "known_session_count": 64,
            "known_session_symbol_pair_count": 248,
            "use": "PARSER_ONLY",
            "population_evidence": False,
            "economic_evidence": False,
            "rerun_performed": False,
        },
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "source_session_count": len(rows),
        "request_session_count": request_sessions,
        "excluded_pre_event_era_session_count": excluded_sessions,
        "session_symbol_membership_count": sum(row["symbol_count"] for row in rows),
        "session_manifest_sha256": json_sha256(session_projection),
        "symbol_manifest_sha256": json_sha256(symbol_projection),
        "request_manifest_sha256": json_sha256(request_projection),
        "code_hashes": normalized_code,
        "dependency_manifest": dependencies,
        "dependency_hashes": {"dependency_manifest_sha256": json_sha256(dependencies)},
        "global_requests": global_requests,
        "sessions": rows,
    }
    declaration["declaration_sha256"] = self_hash(declaration, "declaration_sha256")
    validate_catalogue_declaration(declaration)
    return declaration


def _read_ladder_symbols_default(path: Path) -> Sequence[Any]:
    # Local import keeps the scientific core importable in stdlib-only contexts.
    # ``columns`` is deliberately literal: no other parquet column can enter.
    import pyarrow.parquet as parquet  # type: ignore[import-not-found]

    table = parquet.read_table(Path(path), columns=["raw_symbol"], use_threads=False)
    _require(
        table.column_names == ["raw_symbol"],
        STOP_SCOPE_OR_CODE_DRIFT,
        f"{path.name}: parquet projection widened beyond raw_symbol",
    )
    return table.column("raw_symbol").to_pylist()


def manifest_from_ladder_directory(
    ladder_root: Path,
    *,
    coverage_start: str,
    expected_source_session_count: int = 1_014,
    early_close_sessions: Sequence[str] | None = None,
    raw_symbol_reader: Callable[[Path], Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Build a frozen source manifest from file names and ``raw_symbol`` only."""

    _require(
        coverage_start == COVERAGE_START.isoformat(),
        STOP_SCHEMA_UNAVAILABLE,
        f"coverage_start must be explicitly {COVERAGE_START.isoformat()}",
    )
    root = Path(ladder_root)
    _require(root.is_dir(), STOP_SCOPE_OR_CODE_DRIFT, f"ladder root is not a directory: {root}")
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".parquet")
    _require(bool(files), STOP_SCOPE_OR_CODE_DRIFT, f"no parquet ladder files in {root}")
    invalid_names = [path.name for path in files if LADDER_FILE.fullmatch(path.name) is None]
    _require(
        not invalid_names,
        STOP_SCOPE_OR_CODE_DRIFT,
        f"ladder has non-session parquet filenames: {invalid_names[:3]}",
    )
    _require(
        len(files) == expected_source_session_count,
        STOP_SCOPE_OR_CODE_DRIFT,
        f"source-session count drifted: {len(files)} != {expected_source_session_count}",
    )

    sessions = [LADDER_FILE.fullmatch(path.name).group("session") for path in files]  # type: ignore[union-attr]
    parsed = [_session_date(session) for session in sessions]
    if early_close_sessions is None:
        unsupported = sorted({value.year for value in parsed} - SUPPORTED_CALENDAR_YEARS)
        _require(
            not unsupported,
            STOP_SCOPE_OR_CODE_DRIFT,
            f"calendar years require an explicit early-close set: {unsupported}",
        )
        early_set = set(KNOWN_EARLY_CLOSE_SESSIONS) & set(sessions)
    else:
        early_set = set(early_close_sessions)
        _require(
            len(early_set) == len(list(early_close_sessions)),
            STOP_SCOPE_OR_CODE_DRIFT,
            "duplicate explicit early-close session",
        )

    reader = raw_symbol_reader or _read_ladder_symbols_default
    rows: list[dict[str, Any]] = []
    for path, session in zip(files, sessions, strict=True):
        raw_values = reader(path)
        _require(
            isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)),
            STOP_SCOPE_OR_CODE_DRIFT,
            f"{session}: raw-symbol reader did not return a sequence",
        )
        symbols: list[str] = []
        for value in raw_values:
            symbols.append(validate_raw_spxw_osi(value, session))
        symbols = sorted(set(symbols))
        _require(bool(symbols), STOP_SYMBOL_OR_EXPIRY_MISMATCH, f"{session}: no raw symbols")
        rth_open, rth_close = rth_utc_bounds(session, early_close=session in early_set)
        rows.append(
            {
                "session": session,
                "source_filename": path.name,
                "rth_open_utc": rth_open,
                "rth_close_utc": rth_close,
                "symbols": symbols,
            }
        )

    manifest = {
        "manifest_type": "CMBP_CATALOGUE_SESSION_SYMBOL_MANIFEST_V1",
        "coverage_start": coverage_start,
        "expected_source_session_count": expected_source_session_count,
        "calendar": _calendar_manifest(sorted(early_set)),
        "source_policy": {
            "source": "lifecycle_ladder",
            "filename_fields_read": ["name"],
            "parquet_columns_read": ["raw_symbol"],
            "quote_columns_read": [],
            "outcome_columns_read": [],
        },
        "sessions": rows,
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    return manifest


def _expected_network_policy() -> dict[str, Any]:
    return {
        "offline_processor_network_calls": 0,
        "live_client_imported": False,
        "allowed_future_methods": list(ALLOWED_FUTURE_METHODS),
        "forbidden_method_families": list(FORBIDDEN_METHOD_FAMILIES),
        "timeseries_allowed": False,
        "download_allowed": False,
    }


def validate_catalogue_declaration(
    declaration: Mapping[str, Any],
    *,
    expected_code_hashes: Mapping[str, str] | None = None,
    expected_dependency_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed if any scope, request, code, dependency, or self-hash drifts."""

    _require(isinstance(declaration, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "declaration is not an object")
    _require(declaration.get("artifact_type") == SCOPE_ARTIFACT, STOP_SCOPE_OR_CODE_DRIFT, "wrong scope artifact type")
    _require(declaration.get("artifact_version") == ARTIFACT_VERSION, STOP_SCOPE_OR_CODE_DRIFT, "wrong scope artifact version")
    _require(declaration.get("status") == STATUS_CATALOGUE_READY_LOCAL, STOP_SCOPE_OR_CODE_DRIFT, "scope is not locally ready")
    _require(declaration.get("local_gate_status") == LOCAL_GATE_STATUS, STOP_SCOPE_OR_CODE_DRIFT, "local catalogue gate status drifted")
    _require(declaration.get("dataset") == DATASET, STOP_SCOPE_OR_CODE_DRIFT, "dataset drifted")
    _require(declaration.get("schema") == SCHEMA, STOP_SCHEMA_UNAVAILABLE, "schema must be cmbp-1")
    _require(declaration.get("stype_in") == STYPE_IN, STOP_SCOPE_OR_CODE_DRIFT, "stype_in must be raw_symbol")
    _require(declaration.get("coverage_start") == COVERAGE_START.isoformat(), STOP_SCHEMA_UNAVAILABLE, "coverage boundary drifted")
    _require(declaration.get("network_policy") == _expected_network_policy(), STOP_SCOPE_OR_CODE_DRIFT, "network allowlist or prohibition drifted")
    _require(
        declaration.get("outcome_policy")
        == {
            "outcome_columns_read": False,
            "labels_read": False,
            "returns_read": False,
            "fills_read": False,
            "pnl_read": False,
        },
        STOP_SCOPE_OR_CODE_DRIFT,
        "outcome firewall drifted",
    )
    _require(
        declaration.get("claim_policy")
        == {
            "actual_vendor_availability": "UNKNOWN",
            "exact_available_session_count": "UNKNOWN",
            "exact_vendor_cost": "UNKNOWN",
            "zero_price_boundary": "UNKNOWN",
            "population_event_prevalence": "UNKNOWN",
            "strategy_economics": "NOT_READ",
        },
        STOP_SCOPE_OR_CODE_DRIFT,
        "catalogue claim boundary drifted",
    )
    _require(
        declaration.get("selected_semantic_fixture")
        == {
            "known_session_count": 64,
            "known_session_symbol_pair_count": 248,
            "use": "PARSER_ONLY",
            "population_evidence": False,
            "economic_evidence": False,
            "rerun_performed": False,
        },
        STOP_SCOPE_OR_CODE_DRIFT,
        "selected semantic fixture boundary drifted",
    )

    calendar = declaration.get("calendar")
    _require(isinstance(calendar, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "calendar is missing")
    expected_calendar = _calendar_manifest(calendar.get("early_close_sessions", []))
    _require(dict(calendar) == expected_calendar, STOP_SCOPE_OR_CODE_DRIFT, "calendar declaration drifted")
    early = set(expected_calendar["early_close_sessions"])

    rows = declaration.get("sessions")
    _require(isinstance(rows, list) and rows, STOP_SCOPE_OR_CODE_DRIFT, "scope sessions are empty")
    seen: set[str] = set()
    session_projection: list[dict[str, Any]] = []
    symbol_projection: list[dict[str, Any]] = []
    global_requests = declaration.get("global_requests")
    expected_global_requests = [
        _global_request(method) for method in GLOBAL_REQUEST_METHODS
    ]
    _require(
        global_requests == expected_global_requests,
        STOP_SCOPE_OR_CODE_DRIFT,
        "global dataset-range request drifted",
    )
    session_request_projection: list[dict[str, Any]] = []
    request_sessions = 0
    symbol_memberships = 0
    for row in rows:
        _require(isinstance(row, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "scope session row is invalid")
        session = row.get("session")
        day = _session_date(session)
        _require(session not in seen, STOP_SCOPE_OR_CODE_DRIFT, f"duplicate scope session {session}")
        seen.add(session)
        symbols = row.get("symbols")
        _require(isinstance(symbols, list) and symbols, STOP_SYMBOL_OR_EXPIRY_MISMATCH, f"{session}: symbols missing")
        checked = [validate_raw_spxw_osi(value, session) for value in symbols]
        _require(checked == sorted(checked) and len(checked) == len(set(checked)), STOP_SYMBOL_OR_EXPIRY_MISMATCH, f"{session}: symbols drifted")
        _require(row.get("symbol_count") == len(checked), STOP_SCOPE_OR_CODE_DRIFT, f"{session}: symbol count drifted")
        _require(row.get("symbol_sha256") == json_sha256(checked), STOP_SCOPE_OR_CODE_DRIFT, f"{session}: symbol hash drifted")
        expected_open, expected_close = rth_utc_bounds(session, early_close=session in early)
        _require(row.get("rth_open_utc") == expected_open, STOP_SCOPE_OR_CODE_DRIFT, f"{session}: open bound drifted")
        _require(row.get("rth_close_utc") == expected_close, STOP_SCOPE_OR_CODE_DRIFT, f"{session}: close bound drifted")
        source_filename = row.get("source_filename")
        _require(source_filename is None or source_filename == f"{session}.parquet", STOP_SCOPE_OR_CODE_DRIFT, f"{session}: filename drifted")

        covered = day >= COVERAGE_START
        expected_disposition = "REQUEST_CANDIDATE" if covered else "EXCLUDED_KNOWN_PRE_COVERAGE"
        _require(row.get("coverage_disposition") == expected_disposition, STOP_SCHEMA_UNAVAILABLE, f"{session}: coverage disposition drifted")
        requests = row.get("requests")
        _require(isinstance(requests, list), STOP_SCOPE_OR_CODE_DRIFT, f"{session}: requests missing")
        if covered:
            expected_requests = [_request(row, method) for method in SESSION_REQUEST_METHODS]
            _require(requests == expected_requests, STOP_SCOPE_OR_CODE_DRIFT, f"{session}: request rows drifted")
            request_sessions += 1
            session_request_projection.append({"session": session, "requests": requests})
        else:
            _require(requests == [], STOP_SCHEMA_UNAVAILABLE, f"{session}: pre-event session has request rows")

        symbol_memberships += len(checked)
        session_projection.append(
            {
                "session": session,
                "source_filename": source_filename,
                "rth_open_utc": expected_open,
                "rth_close_utc": expected_close,
                "coverage_disposition": expected_disposition,
            }
        )
        symbol_projection.append({"session": session, "symbols": checked})

    _require([row["session"] for row in rows] == sorted(seen), STOP_SCOPE_OR_CODE_DRIFT, "scope sessions are not sorted")
    _require(declaration.get("source_session_count") == len(rows), STOP_SCOPE_OR_CODE_DRIFT, "source session count drifted")
    _require(declaration.get("request_session_count") == request_sessions, STOP_SCOPE_OR_CODE_DRIFT, "request session count drifted")
    _require(declaration.get("excluded_pre_event_era_session_count") == len(rows) - request_sessions, STOP_SCOPE_OR_CODE_DRIFT, "excluded session count drifted")
    _require(declaration.get("session_symbol_membership_count") == symbol_memberships, STOP_SCOPE_OR_CODE_DRIFT, "symbol-membership count drifted")
    _require(declaration.get("session_manifest_sha256") == json_sha256(session_projection), STOP_SCOPE_OR_CODE_DRIFT, "session manifest hash drifted")
    _require(declaration.get("symbol_manifest_sha256") == json_sha256(symbol_projection), STOP_SCOPE_OR_CODE_DRIFT, "symbol manifest hash drifted")
    request_projection = {
        "global_requests": expected_global_requests,
        "session_requests": session_request_projection,
    }
    _require(declaration.get("request_manifest_sha256") == json_sha256(request_projection), STOP_SCOPE_OR_CODE_DRIFT, "request manifest hash drifted")

    code_hashes = declaration.get("code_hashes")
    _require(isinstance(code_hashes, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "code hashes missing")
    normalized_code = _validate_code_hashes(code_hashes)
    if expected_code_hashes is not None:
        _require(normalized_code == _validate_code_hashes(expected_code_hashes), STOP_SCOPE_OR_CODE_DRIFT, "implementation code drifted")
    dependencies = declaration.get("dependency_manifest")
    _require(isinstance(dependencies, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "dependency manifest missing")
    normalized_dependencies = _validate_dependency_manifest(dependencies)
    _require(
        declaration.get("dependency_hashes")
        == {"dependency_manifest_sha256": json_sha256(normalized_dependencies)},
        STOP_SCOPE_OR_CODE_DRIFT,
        "dependency hash drifted",
    )
    if expected_dependency_manifest is not None:
        _require(normalized_dependencies == _validate_dependency_manifest(expected_dependency_manifest), STOP_SCOPE_OR_CODE_DRIFT, "implementation dependencies drifted")

    claimed = declaration.get("declaration_sha256")
    _validate_hash(claimed, "declaration self-hash")
    _require(claimed == self_hash(declaration, "declaration_sha256"), STOP_SCOPE_OR_CODE_DRIFT, "declaration self-hash mismatch")


def _decimal(value: Any, *, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise CataloguePreflightError(STOP_NONFINITE_COST, f"{name} is not a cost")
    if isinstance(value, float) and not math.isfinite(value):
        raise CataloguePreflightError(STOP_NONFINITE_COST, f"{name} is nonfinite")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CataloguePreflightError(STOP_NONFINITE_COST, f"{name} is not decimal") from exc
    _require(parsed.is_finite() and parsed >= 0, STOP_NONFINITE_COST, f"{name} must be finite and nonnegative")
    return parsed


def decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _record_count(value: Any, session: str) -> int:
    if isinstance(value, bool):
        raise CataloguePreflightError(STOP_ZERO_RECORDS, f"{session}: invalid record count")
    if isinstance(value, int):
        count = value
    elif isinstance(value, str) and value.isdigit():
        count = int(value)
    else:
        raise CataloguePreflightError(STOP_ZERO_RECORDS, f"{session}: record count is not an integer")
    _require(count > 0, STOP_ZERO_RECORDS, f"{session}: zero records")
    return count


def _iso_date(value: Any, *, name: str) -> str:
    _require(isinstance(value, str), STOP_SCOPE_OR_CODE_DRIFT, f"{name} is not ISO text")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CataloguePreflightError(
            STOP_SCOPE_OR_CODE_DRIFT, f"{name} is not an ISO date"
        ) from exc
    _require(parsed.isoformat() == value, STOP_SCOPE_OR_CODE_DRIFT, f"{name} is not canonical")
    return value


def _validate_resolution(
    value: Any,
    *,
    session: str,
    expected_symbols: Sequence[str],
) -> tuple[int, str]:
    _require(
        isinstance(value, Mapping),
        STOP_MISSING_KEY_OR_ENTITLEMENT,
        f"{session}: resolved_symbols mapping is missing",
    )
    _require(
        set(value) == set(expected_symbols),
        STOP_SYMBOL_OR_EXPIRY_MISMATCH,
        f"{session}: symbology response is incomplete or widened",
    )
    normalized: dict[str, list[str | int]] = {}
    for symbol in expected_symbols:
        validate_raw_spxw_osi(symbol, session)
        identifiers = value[symbol]
        _require(
            isinstance(identifiers, list) and bool(identifiers),
            STOP_MISSING_KEY_OR_ENTITLEMENT,
            f"{session}: {symbol!r} has no resolved instrument ID",
        )
        checked: list[str | int] = []
        for identifier in identifiers:
            _require(
                (isinstance(identifier, int) and not isinstance(identifier, bool))
                or (isinstance(identifier, str) and bool(identifier)),
                STOP_SCOPE_OR_CODE_DRIFT,
                f"{session}: invalid resolved instrument ID for {symbol!r}",
            )
            checked.append(identifier)
        normalized[symbol] = checked
    return len(normalized), json_sha256(normalized)


def _response_sha256(responses: Mapping[str, Any]) -> str | None:
    try:
        return json_sha256(responses)
    except (TypeError, ValueError):
        return None


def _receipt_base(
    declaration: Mapping[str, Any],
    responses: Mapping[str, Any],
    *,
    hard_cap_usd: Decimal,
) -> dict[str, Any]:
    return {
        "artifact_type": RECEIPT_ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "declaration_sha256": declaration.get("declaration_sha256"),
        "session_manifest_sha256": declaration.get("session_manifest_sha256"),
        "symbol_manifest_sha256": declaration.get("symbol_manifest_sha256"),
        "request_manifest_sha256": declaration.get("request_manifest_sha256"),
        "code_hashes": declaration.get("code_hashes"),
        "dependency_hashes": declaration.get("dependency_hashes"),
        "response_payload_sha256": _response_sha256(responses),
        "response_source": responses.get("source"),
        "hard_cap_usd": decimal_text(hard_cap_usd),
        "integrity": {
            "offline_processor_network_calls": 0,
            "live_client_imported": False,
            "timeseries_requested": False,
            "download_requested": False,
            "data_downloaded": False,
            "money_spent_by_offline_processor": False,
            "broker_contacted": False,
            "outcomes_read": False,
            "fills_read": False,
            "pnl_read": False,
        },
        "authorization_effect": "NONE",
    }


def _stop_receipt(
    declaration: Mapping[str, Any],
    responses: Mapping[str, Any],
    *,
    hard_cap_usd: Decimal,
    status: str,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(status in STOP_STATUSES, STOP_SCOPE_OR_CODE_DRIFT, f"unknown stop {status}")
    receipt = _receipt_base(declaration, responses, hard_cap_usd=hard_cap_usd)
    receipt.update(
        status=status,
        status_meaning="The offline catalogue preflight failed closed; no acquisition is authorized.",
        failure={"reason": reason, "details": dict(details or {})},
        source_session_count=declaration.get("source_session_count"),
        request_session_count=declaration.get("request_session_count"),
        sessions=[],
    )
    receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
    return receipt


def build_preflight_receipt(
    declaration: Mapping[str, Any],
    responses: Mapping[str, Any],
    *,
    hard_cap_usd: Any,
    expected_code_hashes: Mapping[str, str] | None = None,
    expected_dependency_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate supplied metadata responses and return a self-hashed receipt.

    The function returns a named STOP receipt for response/cap failures.  A
    malformed or drifted declaration also fails closed as
    ``STOP_SCOPE_OR_CODE_DRIFT``.  Nothing in this function can execute a
    method recorded in a request description.
    """

    try:
        cap = _decimal(hard_cap_usd, name="hard_cap_usd")
    except CataloguePreflightError as exc:
        safe_responses = responses if isinstance(responses, Mapping) else {}
        return _stop_receipt(
            declaration,
            safe_responses,
            hard_cap_usd=Decimal(0),
            status=exc.status,
            reason=str(exc),
        )
    try:
        validate_catalogue_declaration(
            declaration,
            expected_code_hashes=expected_code_hashes,
            expected_dependency_manifest=expected_dependency_manifest,
        )
    except CataloguePreflightError as exc:
        return _stop_receipt(
            declaration,
            responses,
            hard_cap_usd=cap,
            status=STOP_SCOPE_OR_CODE_DRIFT,
            reason=str(exc),
        )

    if not isinstance(responses, Mapping):
        return _stop_receipt(declaration, {}, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="responses are not an object")
    if responses.get("artifact_type") != RESPONSE_ARTIFACT:
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="wrong response artifact type")
    if responses.get("declaration_sha256") != declaration.get("declaration_sha256"):
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="responses do not bind the declaration")
    source = responses.get("source")
    if source not in {"synthetic", "externally_supplied"}:
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="response source must be synthetic or externally_supplied")

    attestation = responses.get("method_attestation")
    if not isinstance(attestation, Mapping):
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="method attestation is missing")
    methods_used = attestation.get("methods_used")
    if not isinstance(methods_used, list) or any(not isinstance(value, str) for value in methods_used):
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="method attestation is invalid")
    forbidden = sorted(
        method
        for method in methods_used
        if method not in ALLOWED_FUTURE_METHODS
        or any(family in method.lower() for family in FORBIDDEN_METHOD_FAMILIES)
    )
    if forbidden or attestation.get("timeseries_calls") != 0 or attestation.get("download_calls") != 0:
        return _stop_receipt(
            declaration,
            responses,
            hard_cap_usd=cap,
            status=STOP_SCOPE_OR_CODE_DRIFT,
            reason="forbidden or non-allowlisted method attested",
            details={"forbidden_methods": forbidden},
        )
    exercised = attestation.get("request_shapes_exercised")
    if exercised != list(ALLOWED_FUTURE_METHODS):
        return _stop_receipt(
            declaration,
            responses,
            hard_cap_usd=cap,
            status=STOP_SCOPE_OR_CODE_DRIFT,
            reason="offline rehearsal does not exercise all four allowlisted request shapes",
        )
    if source == "externally_supplied" and set(methods_used) != set(ALLOWED_FUTURE_METHODS):
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_MISSING_KEY_OR_ENTITLEMENT, reason="external response lacks all-four-method attestation")
    if source == "synthetic" and methods_used != []:
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="synthetic rehearsal falsely attests external method use")

    global_response = responses.get("dataset_range")
    if not isinstance(global_response, Mapping):
        return _stop_receipt(
            declaration,
            responses,
            hard_cap_usd=cap,
            status=STOP_MISSING_KEY_OR_ENTITLEMENT,
            reason="dataset-range response is missing",
        )
    range_status = global_response.get("status")
    if range_status in STOP_STATUSES:
        return _stop_receipt(
            declaration,
            responses,
            hard_cap_usd=cap,
            status=range_status,
            reason=f"dataset range: {global_response.get('reason', range_status)}",
        )
    if range_status != "OK":
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_MISSING_KEY_OR_ENTITLEMENT, reason="dataset-range response status is not OK")
    expected_range_request = declaration["global_requests"][0]
    if (
        global_response.get("method") != "metadata.get_dataset_range"
        or global_response.get("request_sha256")
        != expected_range_request["request_sha256"]
    ):
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="dataset-range request binding drifted")
    try:
        dataset_start = _iso_date(global_response.get("dataset_start"), name="dataset_start")
        dataset_end = _iso_date(global_response.get("dataset_end"), name="dataset_end")
        _require(
            dataset_start < dataset_end,
            STOP_SCOPE_OR_CODE_DRIFT,
            "dataset-range response is not half-open increasing",
        )
    except CataloguePreflightError as exc:
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=exc.status, reason=str(exc))

    raw_rows = responses.get("sessions")
    if not isinstance(raw_rows, list):
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_MISSING_SESSION_RESPONSE, reason="response sessions are missing")
    response_by_session: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in raw_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("session"), str):
            return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="response session row is invalid")
        session = row["session"]
        if session in response_by_session:
            duplicates.append(session)
        response_by_session[session] = row
    if duplicates:
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="duplicate response sessions", details={"sessions": sorted(set(duplicates))})

    declared_rows = [row for row in declaration["sessions"] if row["requests"]]
    declared_by_session = {row["session"]: row for row in declared_rows}
    expected_sessions = set(declared_by_session)
    observed_sessions = set(response_by_session)
    missing = sorted(expected_sessions - observed_sessions)
    extra = sorted(observed_sessions - expected_sessions)
    if missing:
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_MISSING_SESSION_RESPONSE, reason="one or more request sessions have no response", details={"missing_sessions": missing, "extra_sessions": extra})
    if extra:
        return _stop_receipt(declaration, responses, hard_cap_usd=cap, status=STOP_SCOPE_OR_CODE_DRIFT, reason="response contains undeclared sessions", details={"extra_sessions": extra})

    checked_rows: list[dict[str, Any]] = []
    total_cost = Decimal(0)
    total_records = 0
    try:
        for session in sorted(expected_sessions):
            declared = declared_by_session[session]
            response = response_by_session[session]
            response_status = response.get("status")
            if response_status in STOP_STATUSES:
                raise CataloguePreflightError(response_status, f"{session}: {response.get('reason', response_status)}")
            _require(response_status == "OK", STOP_MISSING_KEY_OR_ENTITLEMENT, f"{session}: response status is not OK")
            _require(response.get("schema") == SCHEMA, STOP_SCHEMA_UNAVAILABLE, f"{session}: response schema is not cmbp-1")
            _require(response.get("rth_open_utc") == declared["rth_open_utc"], STOP_SCOPE_OR_CODE_DRIFT, f"{session}: response open bound drifted")
            _require(response.get("rth_close_utc") == declared["rth_close_utc"], STOP_SCOPE_OR_CODE_DRIFT, f"{session}: response close bound drifted")
            expected_requests = {request["method"]: request for request in declared["requests"]}
            _require(
                response.get("symbology_resolve_request_sha256")
                == expected_requests["symbology.resolve"]["request_sha256"],
                STOP_SCOPE_OR_CODE_DRIFT,
                f"{session}: symbology request hash drifted",
            )
            _require(
                response.get("unresolved_symbols") == [],
                STOP_SYMBOL_OR_EXPIRY_MISMATCH,
                f"{session}: one or more raw symbols did not resolve",
            )
            resolved_count, resolution_sha256 = _validate_resolution(
                response.get("resolved_symbols"),
                session=session,
                expected_symbols=declared["symbols"],
            )
            _require(
                response.get("cost_request_sha256") == expected_requests["metadata.get_cost"]["request_sha256"],
                STOP_SCOPE_OR_CODE_DRIFT,
                f"{session}: cost request hash drifted",
            )
            _require(
                response.get("record_count_request_sha256")
                == expected_requests["metadata.get_record_count"]["request_sha256"],
                STOP_SCOPE_OR_CODE_DRIFT,
                f"{session}: record-count request hash drifted",
            )
            cost = _decimal(response.get("cost_usd"), name=f"{session}.cost_usd")
            count = _record_count(response.get("record_count"), session)
            # The count rule is deliberately applied even when cost is exactly zero.
            _require(not (cost == 0 and count <= 0), STOP_ZERO_RECORDS, f"{session}: zero-cost response has no records")
            checked_rows.append(
                {
                    "session": session,
                    "schema": SCHEMA,
                    "rth_open_utc": declared["rth_open_utc"],
                    "rth_close_utc": declared["rth_close_utc"],
                    "symbology_resolve_request_sha256": response[
                        "symbology_resolve_request_sha256"
                    ],
                    "resolved_symbol_count": resolved_count,
                    "symbology_resolution_sha256": resolution_sha256,
                    "cost_request_sha256": response["cost_request_sha256"],
                    "record_count_request_sha256": response["record_count_request_sha256"],
                    "cost_usd": decimal_text(cost),
                    "record_count": count,
                }
            )
            total_cost += cost
            total_records += count
    except CataloguePreflightError as exc:
        return _stop_receipt(
            declaration,
            responses,
            hard_cap_usd=cap,
            status=exc.status,
            reason=str(exc),
        )

    if total_cost > cap:
        return _stop_receipt(
            declaration,
            responses,
            hard_cap_usd=cap,
            status=STOP_OVER_HARD_CAP,
            reason=f"exact total {decimal_text(total_cost)} exceeds cap {decimal_text(cap)}",
            details={"exact_total_cost_usd": decimal_text(total_cost)},
        )

    receipt = _receipt_base(declaration, responses, hard_cap_usd=cap)
    receipt.update(
        status=STATUS_PREFLIGHT_PASS_ONLY,
        local_gate_status=LOCAL_GATE_STATUS,
        job49_integration_disposition="VALIDATOR_REHEARSAL_ONLY",
        external_preflight_achieved_by_job49=False,
        status_meaning=(
            "All declared event-era metadata responses are present, positive-count, and within "
            "the cap. This is not download, acquisition, outcome, paper, or trading authority."
        ),
        source_session_count=declaration["source_session_count"],
        excluded_pre_event_era_session_count=declaration["excluded_pre_event_era_session_count"],
        request_session_count=len(checked_rows),
        response_session_count=len(checked_rows),
        dataset_range={
            "request_sha256": expected_range_request["request_sha256"],
            "dataset_start": dataset_start,
            "dataset_end": dataset_end,
            "bounds_semantics": "[dataset_start,dataset_end)",
        },
        exact_total_cost_usd=decimal_text(total_cost),
        exact_total_record_count=total_records,
        claims={
            "actual_vendor_availability": False,
            "zero_price_boundary": False,
            "population_event_prevalence": False,
            "strategy_economics": False,
            "values_are_synthetic": responses.get("source") == "synthetic",
        },
        sessions=checked_rows,
    )
    receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
    validate_preflight_receipt(receipt, declaration=declaration)
    return receipt


def validate_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any] | None = None,
) -> None:
    """Validate a pass or named-STOP receipt without performing external work."""

    _require(isinstance(receipt, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "receipt is not an object")
    _require(receipt.get("artifact_type") == RECEIPT_ARTIFACT, STOP_SCOPE_OR_CODE_DRIFT, "wrong receipt artifact type")
    _require(receipt.get("artifact_version") == ARTIFACT_VERSION, STOP_SCOPE_OR_CODE_DRIFT, "wrong receipt artifact version")
    status = receipt.get("status")
    _require(status in RECEIPT_STATUSES, STOP_SCOPE_OR_CODE_DRIFT, f"unknown receipt status {status}")
    integrity = receipt.get("integrity")
    _require(
        integrity
        == {
            "offline_processor_network_calls": 0,
            "live_client_imported": False,
            "timeseries_requested": False,
            "download_requested": False,
            "data_downloaded": False,
            "money_spent_by_offline_processor": False,
            "broker_contacted": False,
            "outcomes_read": False,
            "fills_read": False,
            "pnl_read": False,
        },
        STOP_SCOPE_OR_CODE_DRIFT,
        "receipt safety attestations drifted",
    )
    _require(receipt.get("authorization_effect") == "NONE", STOP_SCOPE_OR_CODE_DRIFT, "receipt claims authority")
    claimed = receipt.get("receipt_sha256")
    _validate_hash(claimed, "receipt self-hash")
    _require(claimed == self_hash(receipt, "receipt_sha256"), STOP_SCOPE_OR_CODE_DRIFT, "receipt self-hash mismatch")

    if declaration is not None:
        validate_catalogue_declaration(declaration)
        _require(receipt.get("declaration_sha256") == declaration.get("declaration_sha256"), STOP_SCOPE_OR_CODE_DRIFT, "receipt/declaration link drifted")
        for field in ("session_manifest_sha256", "symbol_manifest_sha256", "request_manifest_sha256", "code_hashes", "dependency_hashes"):
            _require(receipt.get(field) == declaration.get(field), STOP_SCOPE_OR_CODE_DRIFT, f"receipt {field} drifted")

    if status != STATUS_PREFLIGHT_PASS_ONLY:
        _require(isinstance(receipt.get("failure"), Mapping), STOP_SCOPE_OR_CODE_DRIFT, "stop receipt lacks failure detail")
        _require(receipt.get("sessions") == [], STOP_SCOPE_OR_CODE_DRIFT, "stop receipt presents partial sessions as a pass")
        return

    _require(receipt.get("local_gate_status") == LOCAL_GATE_STATUS, STOP_SCOPE_OR_CODE_DRIFT, "receipt local gate status drifted")
    _require(receipt.get("job49_integration_disposition") == "VALIDATOR_REHEARSAL_ONLY", STOP_SCOPE_OR_CODE_DRIFT, "receipt overclaims the Job-49 result")
    _require(receipt.get("external_preflight_achieved_by_job49") is False, STOP_SCOPE_OR_CODE_DRIFT, "receipt claims an external Job-49 preflight")
    claims = receipt.get("claims")
    _require(isinstance(claims, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "receipt claims boundary is missing")
    _require(
        claims.get("actual_vendor_availability") is False
        and claims.get("zero_price_boundary") is False
        and claims.get("population_event_prevalence") is False
        and claims.get("strategy_economics") is False,
        STOP_SCOPE_OR_CODE_DRIFT,
        "receipt emits a forbidden availability, boundary, population, or economics claim",
    )

    rows = receipt.get("sessions")
    _require(isinstance(rows, list) and rows, STOP_SCOPE_OR_CODE_DRIFT, "pass receipt sessions are empty")
    _require(receipt.get("request_session_count") == len(rows), STOP_SCOPE_OR_CODE_DRIFT, "receipt request count drifted")
    _require(receipt.get("response_session_count") == len(rows), STOP_SCOPE_OR_CODE_DRIFT, "receipt response count drifted")
    _require(len({row.get("session") for row in rows if isinstance(row, Mapping)}) == len(rows), STOP_SCOPE_OR_CODE_DRIFT, "receipt session duplicate")
    total_cost = Decimal(0)
    total_records = 0
    dataset_range = receipt.get("dataset_range")
    _require(isinstance(dataset_range, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "receipt dataset range is missing")
    dataset_start = _iso_date(dataset_range.get("dataset_start"), name="dataset_start")
    dataset_end = _iso_date(dataset_range.get("dataset_end"), name="dataset_end")
    _require(dataset_start < dataset_end, STOP_SCOPE_OR_CODE_DRIFT, "receipt dataset range is invalid")
    _require(dataset_range.get("bounds_semantics") == "[dataset_start,dataset_end)", STOP_SCOPE_OR_CODE_DRIFT, "dataset-range bounds semantics drifted")
    if declaration is not None:
        _require(
            dataset_range.get("request_sha256")
            == declaration["global_requests"][0]["request_sha256"],
            STOP_SCOPE_OR_CODE_DRIFT,
            "receipt dataset-range request hash drifted",
        )
    for row in rows:
        _require(isinstance(row, Mapping), STOP_SCOPE_OR_CODE_DRIFT, "receipt session row invalid")
        _require(row.get("schema") == SCHEMA, STOP_SCHEMA_UNAVAILABLE, "receipt schema drifted")
        _require(
            isinstance(row.get("resolved_symbol_count"), int)
            and row.get("resolved_symbol_count") > 0,
            STOP_SYMBOL_OR_EXPIRY_MISMATCH,
            "receipt symbology count is invalid",
        )
        _validate_hash(row.get("symbology_resolution_sha256"), "symbology resolution hash")
        total_cost += _decimal(row.get("cost_usd"), name=f"{row.get('session')}.cost_usd")
        total_records += _record_count(row.get("record_count"), str(row.get("session")))
    _require(receipt.get("exact_total_cost_usd") == decimal_text(total_cost), STOP_SCOPE_OR_CODE_DRIFT, "exact cost sum drifted")
    _require(receipt.get("exact_total_record_count") == total_records, STOP_SCOPE_OR_CODE_DRIFT, "record-count sum drifted")
    cap = _decimal(receipt.get("hard_cap_usd"), name="hard_cap_usd")
    _require(total_cost <= cap, STOP_OVER_HARD_CAP, "pass receipt exceeds its hard cap")


def response_template(declaration: Mapping[str, Any], *, source: str = "synthetic") -> dict[str, Any]:
    """Create an offline response skeleton bound to every request session."""

    validate_catalogue_declaration(declaration)
    _require(source in {"synthetic", "externally_supplied"}, STOP_SCOPE_OR_CODE_DRIFT, "invalid response source")
    rows: list[dict[str, Any]] = []
    for declared in declaration["sessions"]:
        if not declared["requests"]:
            continue
        requests = {request["method"]: request for request in declared["requests"]}
        rows.append(
            {
                "session": declared["session"],
                "status": "FILL_OFFLINE",
                "schema": SCHEMA,
                "rth_open_utc": declared["rth_open_utc"],
                "rth_close_utc": declared["rth_close_utc"],
                "symbology_resolve_request_sha256": requests["symbology.resolve"][
                    "request_sha256"
                ],
                "resolved_symbols": None,
                "unresolved_symbols": None,
                "cost_request_sha256": requests["metadata.get_cost"]["request_sha256"],
                "record_count_request_sha256": requests["metadata.get_record_count"]["request_sha256"],
                "cost_usd": None,
                "record_count": None,
            }
        )
    return {
        "artifact_type": RESPONSE_ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "declaration_sha256": declaration["declaration_sha256"],
        "source": source,
        "method_attestation": {
            "methods_used": [] if source == "synthetic" else list(REQUEST_METHODS),
            "request_shapes_exercised": list(ALLOWED_FUTURE_METHODS),
            "timeseries_calls": 0,
            "download_calls": 0,
        },
        "dataset_range": {
            "method": "metadata.get_dataset_range",
            "status": "FILL_OFFLINE",
            "request_sha256": declaration["global_requests"][0]["request_sha256"],
            "dataset_start": None,
            "dataset_end": None,
        },
        "sessions": rows,
    }
