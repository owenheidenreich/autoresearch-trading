"""Local, outcome-blind journal for prospective human trading decisions.

This module is deliberately boring infrastructure.  It has no market-data,
broker, vendor, account, or network dependency.  It writes one local JSON
record per line and makes every record commit to the preceding record's hash.

Three distinctions are load-bearing:

* ``WAIT`` is an explicit decision made while monitoring is on.  Silence after
  a ``PROMPT`` is reported as no response; time outside a monitoring interval
  is ``UNOBSERVED``.  Neither is silently converted into a training label.
* A ``DECISION`` records both when it occurred and when it was appended.  It
  must be appended within 30 seconds, before it can be rewritten by hindsight.
* A ``CORRECTION`` is an annotation pointing to an earlier event.  It never
changes that event or the state reconstructed from it.

Every successful initialization and append also advances a self-hashed,
adjacent high-water watermark after the journal bytes are durable.  Normal
verification requires exact journal/watermark equality, so retaining the
sidecar exposes deletion of a complete tail.  An optional caller-retained head
or minimum sequence is still required to detect coordinated rollback of both
mutable local files.

The writer accepts no arbitrary payload mapping or free-text decision field.
Its fixed schema and closed vocabularies refuse typed fill, return, future-path,
and P&L content.  A separate strict projection exposes only verified DECISION
fields to future training; audit annotations and chain metadata stay outside it.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "v5.human-decision-log.v2"
WATERMARK_SCHEMA_VERSION = "v5.human-decision-log-watermark.v1"
WATERMARK_ARTIFACT_TYPE = "HUMAN_DECISION_LOG_WATERMARK_V1"
HEADER_RECORD = "HEADER"
EVENT_RECORD = "EVENT"
GENESIS_HASH = "0" * 64

MONITORING_ON = "MONITORING_ON"
MONITORING_OFF = "MONITORING_OFF"
PROMPT = "PROMPT"
DECISION = "DECISION"
CORRECTION = "CORRECTION"
EVENT_KINDS = frozenset(
    {MONITORING_ON, MONITORING_OFF, PROMPT, DECISION, CORRECTION}
)

WAIT = "WAIT"
OPEN_CALL = "OPEN_CALL"
OPEN_PUT = "OPEN_PUT"
HOLD = "HOLD"
EXIT = "EXIT"
DECISIONS = frozenset({WAIT, OPEN_CALL, OPEN_PUT, HOLD, EXIT})

MONITORING_STATES = frozenset({"ON", "OFF"})
POSITION_STATES = frozenset({"FLAT", "OPEN"})
INTENDED_ORDER_TYPES = frozenset({"LIMIT", "MARKETABLE_LIMIT"})

INFORMATION_SOURCE_VOCABULARY = frozenset(
    {
        "ES_CHART",
        "NEWS",
        "OPTION_CHAIN",
        "OPTION_POSITION",
        "ORDER_FLOW",
        "SPX_CHART",
        "SYNTHETIC_FIXTURE",
    }
)
REASON_CODE_VOCABULARY = frozenset(
    {
        "BREAKDOWN",
        "BREAKOUT",
        "MOMENTUM",
        "NO_SETUP",
        "PROFIT_PROTECTION",
        "RESISTANCE",
        "REVERSAL",
        "RISK_LIMIT",
        "STOP_TRIGGER",
        "SUPPORT",
        "SYNTHETIC_FIXTURE",
        "TIME_OF_DAY",
        "TREND_CONTINUATION",
        "VOLATILITY",
    }
)
OPEN_OWNER_INTENTS = {
    OPEN_CALL: frozenset({"ENTER_LONG_CALL"}),
    OPEN_PUT: frozenset({"ENTER_LONG_PUT"}),
}
EXIT_OWNER_INTENTS = frozenset(
    {
        "CUT_LOSS",
        "MANUAL_EXIT",
        "PROFIT_TARGET",
        "RISK_EXIT",
        "SETUP_INVALIDATED",
        "TIME_EXIT",
    }
)
OWNER_INTENT_VOCABULARY = frozenset().union(
    *OPEN_OWNER_INTENTS.values(), EXIT_OWNER_INTENTS
)
CORRECTION_ANNOTATION_VOCABULARY = frozenset(
    {
        "ACTION_TYPO",
        "AUDIT_TEST_FIXTURE",
        "CONFIDENCE_TYPO",
        "CONTRACT_TYPO",
        "INFORMATION_SOURCE_TYPO",
        "REASON_CODE_TYPO",
        "TIMESTAMP_TYPO",
    }
)

# Only these fields can leave a verified DECISION for a future training table.
# Append/audit clocks, hash-chain material, identifiers, correction links, and
# notes are deliberately excluded. Identifiers remain only in the verified
# journal as audit/linkage metadata and are never learner-facing fields.
TRAINING_DECISION_FIELDS = (
    "schema_version",
    "session",
    "occurred_at",
    "monitoring_state",
    "position_before",
    "position_after",
    "action",
    "contract_osi",
    "market_state_sha256",
    "universe_sha256",
    "spontaneous",
    "information_sources",
    "confidence",
    "reason_codes",
    "program_contract_sha256",
    "risk_contract_sha256",
    "expiry",
    "right",
    "strike",
    "quantity",
    "intended_order_type",
    "intended_limit_price",
    "estimated_entry_debit_usd",
    "declared_stop_fraction",
    "owner_intent",
)

# Owner-rulable causal clock, frozen for this local instrumentation foundation.
MAX_DECISION_APPEND_LAG_SECONDS = 30
PROMPT_RESPONSE_WINDOW_SECONDS = 30

# SHA-256 identities of the sealed V2 integrity overlay and the unchanged
# canonical ``risk_contract_v1`` object in its bound V1 base contract. A
# decision carries both; the logger validates identity without simulating
# economics.
PROGRAM_CONTRACT_SHA256 = (
    "7726845bd62fbcd8e48643872079c18ee5a1a8832c0fe1e13d0c1795c50b7d7e"
)
RISK_CONTRACT_SHA256 = (
    "cf17dddd5bf6379be3c893c891e19c76581b34680999bb2c5d6f5183d989b116"
)
ENTRY_QUANTITY_CONTRACTS = 1
MAX_TOTAL_DEBIT_USD = 2500.0
MAXIMUM_DECLARED_STOP_FRACTION = -0.4

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# Byte-exact: SPXW root padded to six characters, not whitespace-normalised.
OSI_RE = re.compile(
    r"^SPXW  (?P<expiry>[0-9]{6})(?P<right>[CP])(?P<strike>[0-9]{8})$"
)

_HEADER_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "sequence",
        "log_id",
        "created_at",
        "previous_hash",
        "record_hash",
    }
)
_EVENT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "sequence",
        "event_id",
        "decision_id",
        "kind",
        "session",
        "occurred_at",
        "appended_at",
        "local_monotonic_ns",
        "monitoring_state",
        "position_before",
        "position_after",
        "action",
        "contract_osi",
        "market_state_sha256",
        "universe_sha256",
        "prompt_event_id",
        "spontaneous",
        "information_sources",
        "confidence",
        "reason_codes",
        "program_contract_sha256",
        "risk_contract_sha256",
        "expiry",
        "right",
        "strike",
        "quantity",
        "intended_order_type",
        "intended_limit_price",
        "estimated_entry_debit_usd",
        "declared_stop_fraction",
        "owner_intent",
        "corrects_event_id",
        "note",
        "previous_hash",
        "record_hash",
    }
)
_WATERMARK_KEYS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "journal_schema_version",
        "program_contract_sha256",
        "log_id",
        "terminal_sequence",
        "terminal_head",
        "watermark_sha256",
    }
)


class HumanDecisionLogError(RuntimeError):
    """The journal would stop being causal, local, or append-only."""


def utc_now() -> datetime:
    """The production writer's current UTC wall clock."""

    return datetime.now(timezone.utc)


def _parse_timestamp(value: datetime | str, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise HumanDecisionLogError(
                f"{field_name} must be an ISO-8601 timestamp with a UTC offset"
            ) from exc
    else:
        raise HumanDecisionLogError(f"{field_name} must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HumanDecisionLogError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def canonical_utc(value: datetime | str, *, field_name: str = "timestamp") -> str:
    """UTC with fixed microseconds and ``Z``; the only timestamp representation."""

    parsed = _parse_timestamp(value, field_name=field_name)
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _record_hash(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("record_hash", None)
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _watermark_hash(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("watermark_sha256", None)
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _require_exact_keys(
    record: Mapping[str, Any], expected: frozenset[str], *, where: str
) -> None:
    actual = frozenset(record)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise HumanDecisionLogError(
            f"{where} schema mismatch: missing={missing}, extra={extra}"
        )


def _require_identifier(value: object, *, field_name: str) -> str:
    text = str(value) if value is not None else ""
    if not EVENT_ID_RE.fullmatch(text):
        raise HumanDecisionLogError(
            f"{field_name} must match {EVENT_ID_RE.pattern!r}"
        )
    return text


def _require_sha256(value: object, *, field_name: str) -> str:
    text = str(value) if value is not None else ""
    if not SHA256_RE.fullmatch(text):
        raise HumanDecisionLogError(f"{field_name} must be a lowercase SHA-256")
    return text


def _require_session(value: object) -> str:
    text = str(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise HumanDecisionLogError("session must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise HumanDecisionLogError("session must be canonical YYYY-MM-DD")
    if parsed.weekday() >= 5:
        raise HumanDecisionLogError("session must be a Monday-through-Friday date")
    return text


def _require_inventory(
    value: object,
    *,
    field_name: str,
    vocabulary: frozenset[str],
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise HumanDecisionLogError(f"{field_name} must be an explicit non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise HumanDecisionLogError(
            f"{field_name} entries must be non-empty strings"
        )
    if value != sorted(set(value)):
        raise HumanDecisionLogError(
            f"{field_name} must be sorted and contain no duplicates"
        )
    unknown = sorted(set(value) - vocabulary)
    if unknown:
        raise HumanDecisionLogError(
            f"{field_name} contains values outside the frozen vocabulary: {unknown}"
        )
    return value


def _inventory_argument(value: Iterable[str] | None, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise HumanDecisionLogError(f"{field_name} must be a list, not text")
    return sorted(value)


def _require_finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HumanDecisionLogError(f"{field_name} must be a finite number")
    number = float(value)
    if not float("-inf") < number < float("inf"):
        raise HumanDecisionLogError(f"{field_name} must be a finite number")
    return number


def validate_spxw_osi(
    value: object,
    *,
    session: str,
    required_right: str | None = None,
) -> str:
    """Validate an exact 21-byte same-session SPXW raw OSI symbol."""

    if not isinstance(value, str) or len(value.encode("ascii", errors="ignore")) != 21:
        raise HumanDecisionLogError(
            "contract_osi must be the byte-exact 21-character raw SPXW OSI symbol"
        )
    match = OSI_RE.fullmatch(value)
    if match is None:
        raise HumanDecisionLogError(
            "contract_osi must match '^SPXW  [0-9]{6}[CP][0-9]{8}$' exactly"
        )
    try:
        expiry = datetime.strptime(match.group("expiry"), "%y%m%d").date()
    except ValueError as exc:
        raise HumanDecisionLogError("contract_osi carries an invalid expiry") from exc
    if expiry.isoformat() != session:
        raise HumanDecisionLogError(
            f"contract_osi expiry {expiry.isoformat()} does not equal session {session}"
        )
    if required_right is not None and match.group("right") != required_right:
        raise HumanDecisionLogError(
            f"contract_osi right {match.group('right')} does not match {required_right}"
        )
    if int(match.group("strike")) <= 0:
        raise HumanDecisionLogError("contract_osi strike must be positive")
    return value


def _safe_jsonl_path(path: Path, *, must_exist: bool) -> Path:
    path = Path(path)
    raw = str(path)
    if raw == "-" or "://" in raw or path.suffix != ".jsonl":
        raise HumanDecisionLogError("journal path must be a local .jsonl file")
    if path.is_symlink():
        raise HumanDecisionLogError("journal path may not be a symbolic link")
    if must_exist:
        if not path.is_file():
            raise HumanDecisionLogError(f"journal does not exist: {path}")
    else:
        if path.exists():
            raise HumanDecisionLogError(
                f"journal already exists and will not be overwritten: {path}"
            )
        if not path.parent.is_dir():
            raise HumanDecisionLogError(
                f"journal parent must already exist: {path.parent}"
            )
    return path


def watermark_path(journal_path: Path) -> Path:
    """Return the one adjacent high-water sidecar path for ``journal_path``."""

    journal_path = Path(journal_path)
    return journal_path.with_name(f"{journal_path.name}.watermark.json")


def _safe_watermark_path(journal_path: Path, *, must_exist: bool) -> Path:
    path = watermark_path(journal_path)
    if path.is_symlink():
        raise HumanDecisionLogError("journal watermark may not be a symbolic link")
    if must_exist:
        if not path.is_file():
            raise HumanDecisionLogError(f"journal watermark is missing: {path}")
    elif path.exists():
        raise HumanDecisionLogError(
            f"journal watermark already exists and will not be overwritten: {path}"
        )
    return path


def _open_flags(base: int) -> int:
    return base | getattr(os, "O_NOFOLLOW", 0)


def _read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _append_bytes(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise HumanDecisionLogError("local append did not complete")
        view = view[written:]
    os.fsync(fd)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, _open_flags(flags))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_watermark_bytes(
    path: Path,
    payload: bytes,
    *,
    create: bool,
) -> None:
    """Durably publish a complete sidecar without exposing partial JSON.

    A new sidecar is linked into place only after its temporary inode is
    fsync'd.  Updates use same-directory atomic replacement.  In both cases the
    parent directory is fsync'd before success is reported.
    """

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(
        temporary,
        _open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
        0o600,
    )
    published = False
    try:
        _append_bytes(fd, payload)
        os.close(fd)
        fd = -1
        if create:
            # link(2) has no-overwrite semantics, unlike rename(2).
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            if path.is_symlink() or not path.is_file():
                raise HumanDecisionLogError(
                    "journal watermark is missing or not a regular file"
                )
            os.replace(temporary, path)
        published = True
        _fsync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        if not published and temporary.exists() and not temporary.is_symlink():
            os.unlink(temporary)


@dataclass
class _ReplayState:
    log_id: str
    created_at: datetime
    header_head: str
    head: str
    sequence: int = 0
    monitoring: str = "OFF"
    position: str = "FLAT"
    held_contract_osi: str | None = None
    current_session: str | None = None
    last_occurred_at: datetime | None = None
    last_appended_at: datetime | None = None
    last_monotonic_ns: int | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    event_ids: set[str] = field(default_factory=set)
    decision_ids: set[str] = field(default_factory=set)
    events_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    answered_prompts: set[str] = field(default_factory=set)
    prompts_closed_without_response: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class JournalWatermark:
    artifact_type: str
    schema_version: str
    journal_schema_version: str
    program_contract_sha256: str
    log_id: str
    terminal_sequence: int
    terminal_head: str
    watermark_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "journal_schema_version": self.journal_schema_version,
            "program_contract_sha256": self.program_contract_sha256,
            "log_id": self.log_id,
            "terminal_sequence": self.terminal_sequence,
            "terminal_head": self.terminal_head,
            "watermark_sha256": self.watermark_sha256,
        }


@dataclass(frozen=True)
class VerificationResult:
    log_id: str
    events: int
    decisions: int
    explicit_waits: int
    monitoring_state: str
    observation_status: str
    position_state: str
    held_contract_osi: str | None
    answered_prompts: int
    pending_prompts: int
    no_response_prompts: int
    head: str
    terminal_sequence: int
    watermark_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": "PASS",
            "log_id": self.log_id,
            "events": self.events,
            "decisions": self.decisions,
            "explicit_waits": self.explicit_waits,
            "monitoring_state": self.monitoring_state,
            "observation_status": self.observation_status,
            "position_state": self.position_state,
            "held_contract_osi": self.held_contract_osi,
            "answered_prompts": self.answered_prompts,
            "pending_prompts": self.pending_prompts,
            "no_response_prompts": self.no_response_prompts,
            "head": self.head,
            "terminal_sequence": self.terminal_sequence,
            "watermark_sha256": self.watermark_sha256,
        }


def _validate_header(record: Mapping[str, Any], *, raw_line: str) -> _ReplayState:
    _require_exact_keys(record, _HEADER_KEYS, where="header")
    if record["record_type"] != HEADER_RECORD:
        raise HumanDecisionLogError("first JSONL record is not a header")
    if record["schema_version"] != SCHEMA_VERSION:
        raise HumanDecisionLogError(
            f"unknown journal schema {record['schema_version']!r}"
        )
    if record["sequence"] != 0:
        raise HumanDecisionLogError("header sequence must be zero")
    log_id = _require_identifier(record["log_id"], field_name="log_id")
    created_text = canonical_utc(record["created_at"], field_name="created_at")
    if created_text != record["created_at"]:
        raise HumanDecisionLogError("created_at is not in canonical UTC form")
    if record["previous_hash"] != GENESIS_HASH:
        raise HumanDecisionLogError("header does not point to the genesis hash")
    if record["record_hash"] != _record_hash(record):
        raise HumanDecisionLogError("header hash mismatch")
    if raw_line != _canonical_json(record):
        raise HumanDecisionLogError("header is not deterministically serialized")
    return _ReplayState(
        log_id=log_id,
        created_at=_parse_timestamp(created_text, field_name="created_at"),
        header_head=str(record["record_hash"]),
        head=str(record["record_hash"]),
    )


def _validate_event_envelope(
    record: Mapping[str, Any],
    *,
    raw_line: str,
    state: _ReplayState,
) -> tuple[datetime, datetime]:
    _require_exact_keys(record, _EVENT_KEYS, where=f"record {state.sequence + 1}")
    if record["record_type"] != EVENT_RECORD:
        raise HumanDecisionLogError("non-header records must have record_type EVENT")
    if record["schema_version"] != SCHEMA_VERSION:
        raise HumanDecisionLogError("event schema version differs from the header")
    expected_sequence = state.sequence + 1
    if record["sequence"] != expected_sequence:
        raise HumanDecisionLogError(
            f"event sequence is not append-only: expected {expected_sequence}, "
            f"got {record['sequence']}"
        )
    event_id = _require_identifier(record["event_id"], field_name="event_id")
    if event_id in state.event_ids:
        raise HumanDecisionLogError(f"duplicate event_id: {event_id}")
    if record["kind"] not in EVENT_KINDS:
        raise HumanDecisionLogError(f"unknown event kind: {record['kind']!r}")
    _require_session(record["session"])
    occurred_text = canonical_utc(record["occurred_at"], field_name="occurred_at")
    appended_text = canonical_utc(record["appended_at"], field_name="appended_at")
    if occurred_text != record["occurred_at"] or appended_text != record["appended_at"]:
        raise HumanDecisionLogError("event timestamps are not in canonical UTC form")
    occurred = _parse_timestamp(occurred_text, field_name="occurred_at")
    appended = _parse_timestamp(appended_text, field_name="appended_at")
    if occurred.astimezone(ZoneInfo("America/New_York")).date().isoformat() != record[
        "session"
    ]:
        raise HumanDecisionLogError(
            "session must equal the occurred_at America/New_York calendar date"
        )
    monotonic_ns = record["local_monotonic_ns"]
    if (
        isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns < 0
    ):
        raise HumanDecisionLogError("local_monotonic_ns must be a non-negative integer")
    if occurred > appended:
        raise HumanDecisionLogError("event occurred_at is in the future of appended_at")
    if state.last_occurred_at is not None and occurred < state.last_occurred_at:
        raise HumanDecisionLogError("retroactive/backdated event timestamp")
    if state.last_appended_at is not None and appended < state.last_appended_at:
        raise HumanDecisionLogError("appended_at moved backward")
    if (
        state.last_monotonic_ns is not None
        and monotonic_ns <= state.last_monotonic_ns
    ):
        raise HumanDecisionLogError("local monotonic clock did not advance")
    if record["previous_hash"] != state.head:
        raise HumanDecisionLogError(
            f"hash chain broken at sequence {expected_sequence}"
        )
    if record["record_hash"] != _record_hash(record):
        raise HumanDecisionLogError(
            f"record hash mismatch at sequence {expected_sequence}"
        )
    if raw_line != _canonical_json(record):
        raise HumanDecisionLogError(
            f"record {expected_sequence} is not deterministically serialized"
        )
    return occurred, appended


def _validate_common_nullable_fields(record: Mapping[str, Any]) -> None:
    note = record["note"]
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise HumanDecisionLogError("note must be null or a non-empty string")
    if record["kind"] != CORRECTION and note is not None:
        raise HumanDecisionLogError(
            "note is audit-only and may appear only on CORRECTION"
        )
    for name in ("position_before", "position_after"):
        if record[name] not in POSITION_STATES:
            raise HumanDecisionLogError(f"{name} must be FLAT or OPEN")
    if record["monitoring_state"] not in MONITORING_STATES:
        raise HumanDecisionLogError("monitoring_state must be ON or OFF")


def _validate_session_transition(record: Mapping[str, Any], state: _ReplayState) -> None:
    session = str(record["session"])
    if state.current_session is None or session == state.current_session:
        return
    if state.monitoring != "OFF" or state.position != "FLAT":
        raise HumanDecisionLogError(
            "a new session cannot begin while monitoring or a position remains open"
        )
    if record["kind"] != MONITORING_ON:
        raise HumanDecisionLogError("a new session must begin with MONITORING_ON")


def _close_unanswered_prompts(state: _ReplayState) -> None:
    for event_id, event in state.events_by_id.items():
        if event["kind"] == PROMPT and event_id not in state.answered_prompts:
            state.prompts_closed_without_response.add(event_id)


def _validate_and_apply_event(
    record: Mapping[str, Any],
    *,
    occurred: datetime,
    appended: datetime,
    state: _ReplayState,
) -> None:
    _validate_common_nullable_fields(record)
    _validate_session_transition(record, state)
    kind = str(record["kind"])
    before = state.position

    if record["position_before"] != before:
        raise HumanDecisionLogError(
            f"position_before {record['position_before']} disagrees with chain state {before}"
        )

    if kind == MONITORING_ON:
        if state.monitoring != "OFF":
            raise HumanDecisionLogError("MONITORING_ON while monitoring is already on")
        if record["monitoring_state"] != "ON":
            raise HumanDecisionLogError("MONITORING_ON must record monitoring_state ON")
        state.monitoring = "ON"
    elif kind == MONITORING_OFF:
        if state.monitoring != "ON":
            raise HumanDecisionLogError("MONITORING_OFF while monitoring is already off")
        if state.position != "FLAT":
            raise HumanDecisionLogError("monitoring cannot turn off while a position is open")
        if record["monitoring_state"] != "OFF":
            raise HumanDecisionLogError("MONITORING_OFF must record monitoring_state OFF")
        _close_unanswered_prompts(state)
        state.monitoring = "OFF"
    elif kind == PROMPT:
        if state.monitoring != "ON" or record["monitoring_state"] != "ON":
            raise HumanDecisionLogError("PROMPT requires monitoring_state ON")
        _require_sha256(
            record["market_state_sha256"], field_name="market_state_sha256"
        )
        _require_sha256(record["universe_sha256"], field_name="universe_sha256")
    elif kind == DECISION:
        if state.monitoring != "ON" or record["monitoring_state"] != "ON":
            raise HumanDecisionLogError(
                "DECISION requires explicit monitoring_state ON"
            )
        decision_id = _require_identifier(
            record["decision_id"], field_name="decision_id"
        )
        if decision_id in state.decision_ids:
            raise HumanDecisionLogError(f"duplicate decision_id: {decision_id}")
        action = record["action"]
        if action not in DECISIONS:
            raise HumanDecisionLogError(f"unknown decision action: {action!r}")
        _require_sha256(
            record["market_state_sha256"], field_name="market_state_sha256"
        )
        _require_sha256(record["universe_sha256"], field_name="universe_sha256")
        _require_inventory(
            record["information_sources"],
            field_name="information_sources",
            vocabulary=INFORMATION_SOURCE_VOCABULARY,
        )
        _require_inventory(
            record["reason_codes"],
            field_name="reason_codes",
            vocabulary=REASON_CODE_VOCABULARY,
        )
        confidence = _require_finite_number(
            record["confidence"], field_name="confidence"
        )
        if not 0.0 <= confidence <= 1.0:
            raise HumanDecisionLogError("confidence must be between 0 and 1")
        if record["program_contract_sha256"] != PROGRAM_CONTRACT_SHA256:
            raise HumanDecisionLogError(
                "program_contract_sha256 does not identify the sealed Job 49 contract"
            )
        if record["risk_contract_sha256"] != RISK_CONTRACT_SHA256:
            raise HumanDecisionLogError(
                "risk_contract_sha256 does not identify frozen risk_contract_v1"
            )
        lag = appended - occurred
        if lag > timedelta(seconds=MAX_DECISION_APPEND_LAG_SECONDS):
            raise HumanDecisionLogError(
                "DECISION was appended retroactively: append lag exceeds "
                f"{MAX_DECISION_APPEND_LAG_SECONDS} seconds"
            )

        prompt_event_id = record["prompt_event_id"]
        if prompt_event_id is not None:
            if record["spontaneous"] is not False:
                raise HumanDecisionLogError(
                    "a prompt-linked DECISION must explicitly set spontaneous false"
                )
            prompt_event_id = _require_identifier(
                prompt_event_id, field_name="prompt_event_id"
            )
            prompt = state.events_by_id.get(prompt_event_id)
            if prompt is None or prompt["kind"] != PROMPT:
                raise HumanDecisionLogError(
                    "prompt_event_id must point backward to a PROMPT"
                )
            if prompt_event_id in state.answered_prompts:
                raise HumanDecisionLogError("PROMPT already has a DECISION response")
            if prompt["session"] != record["session"]:
                raise HumanDecisionLogError("PROMPT and DECISION must share a session")
            prompt_at = _parse_timestamp(prompt["occurred_at"], field_name="occurred_at")
            if occurred < prompt_at:
                raise HumanDecisionLogError("DECISION cannot precede its PROMPT")
            if occurred - prompt_at > timedelta(seconds=PROMPT_RESPONSE_WINDOW_SECONDS):
                raise HumanDecisionLogError("DECISION arrived after the PROMPT response window")
            state.answered_prompts.add(prompt_event_id)
        elif record["spontaneous"] is not True:
            raise HumanDecisionLogError(
                "an unprompted DECISION must explicitly set spontaneous true"
            )

        session = str(record["session"])
        contract = record["contract_osi"]
        if action == WAIT:
            if before != "FLAT":
                raise HumanDecisionLogError("WAIT is a flat-state decision; use HOLD when open")
            if contract is not None:
                raise HumanDecisionLogError("WAIT cannot carry contract_osi")
        elif action in {OPEN_CALL, OPEN_PUT}:
            if before != "FLAT":
                raise HumanDecisionLogError("OPEN requires position_state FLAT")
            required_right = "C" if action == OPEN_CALL else "P"
            contract = validate_spxw_osi(
                contract, session=session, required_right=required_right
            )
            match = OSI_RE.fullmatch(contract)
            assert match is not None  # validated immediately above
            expected_expiry = datetime.strptime(
                match.group("expiry"), "%y%m%d"
            ).date().isoformat()
            expected_strike = int(match.group("strike")) / 1000.0
            if record["expiry"] != expected_expiry:
                raise HumanDecisionLogError("expiry does not match raw OSI symbol")
            if record["right"] != required_right:
                raise HumanDecisionLogError("right does not match action/raw OSI symbol")
            strike = _require_finite_number(record["strike"], field_name="strike")
            if strike != expected_strike:
                raise HumanDecisionLogError("strike does not match raw OSI symbol")
            if record["quantity"] != ENTRY_QUANTITY_CONTRACTS:
                raise HumanDecisionLogError(
                    f"quantity must equal frozen entry quantity {ENTRY_QUANTITY_CONTRACTS}"
                )
            if record["intended_order_type"] not in INTENDED_ORDER_TYPES:
                raise HumanDecisionLogError(
                    "intended_order_type must be LIMIT or MARKETABLE_LIMIT"
                )
            limit_price = _require_finite_number(
                record["intended_limit_price"], field_name="intended_limit_price"
            )
            if limit_price <= 0:
                raise HumanDecisionLogError("intended_limit_price must be positive")
            debit = _require_finite_number(
                record["estimated_entry_debit_usd"],
                field_name="estimated_entry_debit_usd",
            )
            if not 0 < debit <= MAX_TOTAL_DEBIT_USD:
                raise HumanDecisionLogError(
                    "estimated_entry_debit_usd exceeds frozen $2,500 intent cap"
                )
            stop = _require_finite_number(
                record["declared_stop_fraction"],
                field_name="declared_stop_fraction",
            )
            if stop > MAXIMUM_DECLARED_STOP_FRACTION:
                raise HumanDecisionLogError(
                    "declared_stop_fraction must be -0.4 or more negative"
                )
            if record["owner_intent"] not in OPEN_OWNER_INTENTS[action]:
                raise HumanDecisionLogError(
                    f"{action} owner_intent must be one of "
                    f"{sorted(OPEN_OWNER_INTENTS[action])}"
                )
            state.position = "OPEN"
            state.held_contract_osi = contract
        elif action in {HOLD, EXIT}:
            if before != "OPEN":
                raise HumanDecisionLogError(
                    f"{action} requires explicit position_state OPEN"
                )
            contract = validate_spxw_osi(contract, session=session)
            if contract != state.held_contract_osi:
                raise HumanDecisionLogError(
                    "position decision contract_osi differs from the held contract"
                )
            if action == EXIT:
                if record["owner_intent"] not in EXIT_OWNER_INTENTS:
                    raise HumanDecisionLogError(
                        "EXIT owner_intent must be one of "
                        f"{sorted(EXIT_OWNER_INTENTS)}"
                    )
                state.position = "FLAT"
                state.held_contract_osi = None
        state.decision_ids.add(decision_id)
    elif kind == CORRECTION:
        target_id = _require_identifier(
            record["corrects_event_id"], field_name="corrects_event_id"
        )
        if target_id not in state.events_by_id:
            raise HumanDecisionLogError(
                "CORRECTION must point backward to an existing event_id"
            )
        if record["note"] not in CORRECTION_ANNOTATION_VOCABULARY:
            raise HumanDecisionLogError(
                "CORRECTION note must be one frozen audit annotation: "
                f"{sorted(CORRECTION_ANNOTATION_VOCABULARY)}"
            )
        if record["monitoring_state"] != state.monitoring:
            raise HumanDecisionLogError(
                "CORRECTION must record the actual current monitoring state"
            )

    non_decision_fields = (
        "decision_id",
        "action",
        "contract_osi",
        "prompt_event_id",
        "spontaneous",
        "confidence",
        "program_contract_sha256",
        "risk_contract_sha256",
        "expiry",
        "right",
        "strike",
        "quantity",
        "intended_order_type",
        "intended_limit_price",
        "estimated_entry_debit_usd",
        "declared_stop_fraction",
        "owner_intent",
    )
    if kind != DECISION and any(record[name] is not None for name in non_decision_fields):
        raise HumanDecisionLogError(
            f"{kind} cannot carry decision/action/contract/prompt-response fields"
        )
    if kind != DECISION and (
        record["information_sources"] != [] or record["reason_codes"] != []
    ):
        raise HumanDecisionLogError(f"{kind} cannot carry decision inventories")
    entry_only_fields = (
        "expiry",
        "right",
        "strike",
        "quantity",
        "intended_order_type",
        "intended_limit_price",
        "estimated_entry_debit_usd",
        "declared_stop_fraction",
    )
    if kind == DECISION and record["action"] not in {OPEN_CALL, OPEN_PUT} and any(
        record[name] is not None for name in entry_only_fields
    ):
        raise HumanDecisionLogError(
            f"{record['action']} cannot carry entry-intent risk fields"
        )
    if kind == DECISION and record["action"] not in {OPEN_CALL, OPEN_PUT, EXIT}:
        if record["owner_intent"] is not None:
            raise HumanDecisionLogError(
                f"{record['action']} cannot carry owner_intent"
            )
    if kind not in {PROMPT, DECISION} and (
        record["market_state_sha256"] is not None
        or record["universe_sha256"] is not None
    ):
        raise HumanDecisionLogError(f"{kind} cannot carry market snapshot hashes")
    if kind != CORRECTION and record["corrects_event_id"] is not None:
        raise HumanDecisionLogError(f"{kind} cannot carry corrects_event_id")

    if record["position_after"] != state.position:
        raise HumanDecisionLogError(
            f"position_after {record['position_after']} disagrees with resulting state "
            f"{state.position}"
        )

    state.current_session = str(record["session"])
    state.last_occurred_at = occurred
    state.last_appended_at = appended
    state.last_monotonic_ns = int(record["local_monotonic_ns"])


def _decode_and_verify(data: bytes) -> _ReplayState:
    if not data:
        raise HumanDecisionLogError("journal is empty or was truncated")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HumanDecisionLogError("journal is not UTF-8 JSONL") from exc
    if not text.endswith("\n"):
        raise HumanDecisionLogError("journal has an incomplete final JSONL record")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise HumanDecisionLogError("journal contains an empty JSONL record")
    decoded: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanDecisionLogError(
                f"journal line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise HumanDecisionLogError(
                f"journal line {line_number} is not a JSON object"
            )
        decoded.append(record)

    state = _validate_header(decoded[0], raw_line=lines[0])
    for record, raw_line in zip(decoded[1:], lines[1:]):
        occurred, appended = _validate_event_envelope(
            record, raw_line=raw_line, state=state
        )
        _validate_and_apply_event(
            record, occurred=occurred, appended=appended, state=state
        )
        event = dict(record)
        state.sequence = int(record["sequence"])
        state.head = str(record["record_hash"])
        state.records.append(event)
        state.event_ids.add(str(record["event_id"]))
        state.events_by_id[str(record["event_id"])] = event
    return state


def _build_watermark(path: Path, state: _ReplayState) -> JournalWatermark:
    payload: dict[str, Any] = {
        "artifact_type": WATERMARK_ARTIFACT_TYPE,
        "schema_version": WATERMARK_SCHEMA_VERSION,
        "journal_schema_version": SCHEMA_VERSION,
        "program_contract_sha256": PROGRAM_CONTRACT_SHA256,
        "log_id": state.log_id,
        "terminal_sequence": state.sequence,
        "terminal_head": state.head,
    }
    payload["watermark_sha256"] = _watermark_hash(payload)
    return JournalWatermark(**payload)


def _decode_watermark(
    data: bytes,
    *,
    journal_path: Path,
    expected_log_id: str,
) -> JournalWatermark:
    if not data:
        raise HumanDecisionLogError("journal watermark is empty or truncated")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HumanDecisionLogError("journal watermark is not UTF-8 JSON") from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise HumanDecisionLogError(
            "journal watermark must contain one complete canonical JSON record"
        )
    raw_line = text[:-1]
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise HumanDecisionLogError("journal watermark is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise HumanDecisionLogError("journal watermark is not a JSON object")
    _require_exact_keys(payload, _WATERMARK_KEYS, where="journal watermark")
    if payload["artifact_type"] != WATERMARK_ARTIFACT_TYPE:
        raise HumanDecisionLogError("unknown journal watermark artifact_type")
    if payload["schema_version"] != WATERMARK_SCHEMA_VERSION:
        raise HumanDecisionLogError("unknown journal watermark schema version")
    if payload["journal_schema_version"] != SCHEMA_VERSION:
        raise HumanDecisionLogError(
            "journal watermark does not bind the active journal schema"
        )
    if payload["program_contract_sha256"] != PROGRAM_CONTRACT_SHA256:
        raise HumanDecisionLogError(
            "journal watermark does not bind the active program contract"
        )
    log_id = _require_identifier(payload["log_id"], field_name="watermark log_id")
    if log_id != expected_log_id:
        raise HumanDecisionLogError("journal watermark log_id does not match journal")
    sequence = payload["terminal_sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise HumanDecisionLogError(
            "journal watermark terminal_sequence must be a non-negative integer"
        )
    terminal_head = _require_sha256(
        payload["terminal_head"], field_name="watermark terminal_head"
    )
    self_hash = _require_sha256(
        payload["watermark_sha256"], field_name="watermark_sha256"
    )
    if self_hash != _watermark_hash(payload):
        raise HumanDecisionLogError("journal watermark self-hash mismatch")
    if raw_line != _canonical_json(payload):
        raise HumanDecisionLogError(
            "journal watermark is not deterministically serialized"
        )
    return JournalWatermark(
        artifact_type=WATERMARK_ARTIFACT_TYPE,
        schema_version=WATERMARK_SCHEMA_VERSION,
        journal_schema_version=SCHEMA_VERSION,
        program_contract_sha256=PROGRAM_CONTRACT_SHA256,
        log_id=log_id,
        terminal_sequence=sequence,
        terminal_head=terminal_head,
        watermark_sha256=self_hash,
    )


def _read_watermark_locked(path: Path, state: _ReplayState) -> JournalWatermark:
    sidecar = _safe_watermark_path(path, must_exist=True)
    fd = os.open(sidecar, _open_flags(os.O_RDONLY))
    try:
        return _decode_watermark(
            _read_all(fd), journal_path=path, expected_log_id=state.log_id
        )
    finally:
        os.close(fd)


def _head_at_sequence(state: _ReplayState, sequence: int) -> str | None:
    if sequence == 0:
        return state.header_head
    if 1 <= sequence <= len(state.records):
        return str(state.records[sequence - 1]["record_hash"])
    return None


def _assert_watermark_matches_terminal(
    state: _ReplayState,
    watermark: JournalWatermark,
) -> None:
    anchored_head = _head_at_sequence(state, watermark.terminal_sequence)
    if anchored_head is None:
        raise HumanDecisionLogError(
            "journal was tail-truncated: terminal sequence is below its watermark"
        )
    if anchored_head != watermark.terminal_head:
        raise HumanDecisionLogError(
            "journal terminal head does not match its recorded watermark"
        )
    if watermark.terminal_sequence != state.sequence:
        raise HumanDecisionLogError(
            "journal watermark terminal sequence regressed behind the journal head"
        )
    if watermark.terminal_head != state.head:
        raise HumanDecisionLogError(
            "journal watermark terminal head differs from the journal head"
        )


def _assert_expected_terminal(
    state: _ReplayState,
    *,
    expected_head: str | None,
    expected_min_sequence: int | None,
) -> None:
    if expected_head is not None:
        expected = _require_sha256(expected_head, field_name="expected_head")
        if state.head != expected:
            raise HumanDecisionLogError(
                "journal head differs from the caller's expected terminal head"
            )
    if expected_min_sequence is not None:
        if (
            isinstance(expected_min_sequence, bool)
            or not isinstance(expected_min_sequence, int)
            or expected_min_sequence < 0
        ):
            raise HumanDecisionLogError(
                "expected minimum sequence must be a non-negative integer"
            )
        if state.sequence < expected_min_sequence:
            raise HumanDecisionLogError(
                "journal terminal sequence is below the expected minimum sequence"
            )


def _verified_state_and_watermark_locked(
    path: Path,
    fd: int,
    *,
    expected_head: str | None = None,
    expected_min_sequence: int | None = None,
) -> tuple[_ReplayState, JournalWatermark]:
    state = _decode_and_verify(_read_all(fd))
    watermark = _read_watermark_locked(path, state)
    _assert_watermark_matches_terminal(state, watermark)
    _assert_expected_terminal(
        state,
        expected_head=expected_head,
        expected_min_sequence=expected_min_sequence,
    )
    return state, watermark


def _result_from_state(
    state: _ReplayState,
    watermark: JournalWatermark,
    *,
    now: datetime | str | None = None,
) -> VerificationResult:
    instant = _parse_timestamp(now or utc_now(), field_name="status now")
    prompts = {
        event_id: event
        for event_id, event in state.events_by_id.items()
        if event["kind"] == PROMPT
    }
    pending = 0
    no_response = 0
    for prompt_id, prompt in prompts.items():
        if prompt_id in state.answered_prompts:
            continue
        prompt_at = _parse_timestamp(prompt["occurred_at"], field_name="occurred_at")
        expired = instant - prompt_at > timedelta(
            seconds=PROMPT_RESPONSE_WINDOW_SECONDS
        )
        if prompt_id in state.prompts_closed_without_response or expired:
            no_response += 1
        else:
            pending += 1
    decisions = [record for record in state.records if record["kind"] == DECISION]
    waits = [record for record in decisions if record["action"] == WAIT]
    return VerificationResult(
        log_id=state.log_id,
        events=len(state.records),
        decisions=len(decisions),
        explicit_waits=len(waits),
        monitoring_state=state.monitoring,
        observation_status="OBSERVED" if state.monitoring == "ON" else "UNOBSERVED",
        position_state=state.position,
        held_contract_osi=state.held_contract_osi,
        answered_prompts=len(state.answered_prompts),
        pending_prompts=pending,
        no_response_prompts=no_response,
        head=state.head,
        terminal_sequence=state.sequence,
        watermark_sha256=watermark.watermark_sha256,
    )


def initialize_log(
    path: Path,
    *,
    log_id: str | None = None,
    created_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Create one new local journal; an existing path is never overwritten."""

    path = _safe_jsonl_path(path, must_exist=False)
    sidecar = _safe_watermark_path(path, must_exist=False)
    log_id = _require_identifier(log_id or str(uuid.uuid4()), field_name="log_id")
    created = canonical_utc(created_at or utc_now(), field_name="created_at")
    header: dict[str, Any] = {
        "record_type": HEADER_RECORD,
        "schema_version": SCHEMA_VERSION,
        "sequence": 0,
        "log_id": log_id,
        "created_at": created,
        "previous_hash": GENESIS_HASH,
    }
    header["record_hash"] = _record_hash(header)
    encoded = (_canonical_json(header) + "\n").encode("utf-8")
    try:
        fd = os.open(
            path,
            _open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
        )
    except FileExistsError as exc:
        raise HumanDecisionLogError(
            f"journal already exists and will not be overwritten: {path}"
        ) from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _append_bytes(fd, encoded)
        state = _decode_and_verify(encoded)
        watermark = _build_watermark(path, state)
        _write_watermark_bytes(
            sidecar,
            (_canonical_json(watermark.payload()) + "\n").encode("utf-8"),
            create=True,
        )
    finally:
        os.close(fd)
    return header


def _event_after_state(
    *, state: _ReplayState, kind: str, action: str | None
) -> tuple[str, str, str]:
    monitoring = state.monitoring
    position_before = state.position
    position_after = position_before
    if kind == MONITORING_ON:
        monitoring = "ON"
    elif kind == MONITORING_OFF:
        monitoring = "OFF"
    elif kind == DECISION:
        if action in {OPEN_CALL, OPEN_PUT}:
            position_after = "OPEN"
        elif action == EXIT:
            position_after = "FLAT"
    return monitoring, position_before, position_after


def _append_event_with_clocks(
    path: Path,
    *,
    kind: str,
    event_id: str,
    session: str,
    occurred_at: datetime | str,
    appended_at: datetime | str | None,
    local_monotonic_ns: int | None,
    decision_id: str | None = None,
    monitoring_state: str | None = None,
    position_state: str | None = None,
    action: str | None = None,
    contract_osi: str | None = None,
    market_state_sha256: str | None = None,
    universe_sha256: str | None = None,
    prompt_event_id: str | None = None,
    spontaneous: bool | None = None,
    information_sources: Iterable[str] | None = None,
    confidence: float | None = None,
    reason_codes: Iterable[str] | None = None,
    program_contract_sha256: str | None = None,
    risk_contract_sha256: str | None = None,
    quantity: int | None = None,
    intended_order_type: str | None = None,
    intended_limit_price: float | None = None,
    estimated_entry_debit_usd: float | None = None,
    declared_stop_fraction: float | None = None,
    owner_intent: str | None = None,
    corrects_event_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Private deterministic writer used by production and offline tests."""

    path = _safe_jsonl_path(path, must_exist=True)
    fd = os.open(path, _open_flags(os.O_RDWR | os.O_APPEND))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        state, _ = _verified_state_and_watermark_locked(path, fd)
        if appended_at is None:
            appended_at = utc_now()
        if local_monotonic_ns is None:
            local_monotonic_ns = time.monotonic_ns()
        derived_monitoring, position_before, position_after = _event_after_state(
            state=state, kind=kind, action=action
        )
        if kind == DECISION:
            # These are intentionally not inferred: each training label must say
            # that the owner was watching and what position state they saw.
            if monitoring_state is None:
                raise HumanDecisionLogError(
                    "DECISION requires explicit monitoring_state ON"
                )
            if position_state is None:
                raise HumanDecisionLogError(
                    "DECISION requires explicit position_state FLAT or OPEN"
                )
            recorded_monitoring = monitoring_state
            position_before = position_state
        elif kind == CORRECTION:
            recorded_monitoring = monitoring_state or state.monitoring
        else:
            recorded_monitoring = derived_monitoring
        expiry: str | None = None
        right: str | None = None
        strike: float | None = None
        if kind == DECISION and action in {OPEN_CALL, OPEN_PUT}:
            match = OSI_RE.fullmatch(contract_osi or "")
            if match is not None:
                try:
                    expiry = datetime.strptime(
                        match.group("expiry"), "%y%m%d"
                    ).date().isoformat()
                except ValueError:
                    # The verifier reports the authoritative raw-OSI error.
                    expiry = None
                right = match.group("right")
                strike = int(match.group("strike")) / 1000.0
        event: dict[str, Any] = {
            "record_type": EVENT_RECORD,
            "schema_version": SCHEMA_VERSION,
            "sequence": state.sequence + 1,
            "event_id": event_id,
            "decision_id": decision_id,
            "kind": kind,
            "session": session,
            "occurred_at": canonical_utc(occurred_at, field_name="occurred_at"),
            "appended_at": canonical_utc(appended_at, field_name="appended_at"),
            "local_monotonic_ns": local_monotonic_ns,
            "monitoring_state": recorded_monitoring,
            "position_before": position_before,
            "position_after": position_after,
            "action": action,
            "contract_osi": contract_osi,
            "market_state_sha256": market_state_sha256,
            "universe_sha256": universe_sha256,
            "prompt_event_id": prompt_event_id,
            "spontaneous": spontaneous,
            "information_sources": _inventory_argument(
                information_sources, field_name="information_sources"
            ),
            "confidence": confidence,
            "reason_codes": _inventory_argument(
                reason_codes, field_name="reason_codes"
            ),
            "program_contract_sha256": program_contract_sha256,
            "risk_contract_sha256": risk_contract_sha256,
            "expiry": expiry,
            "right": right,
            "strike": strike,
            "quantity": quantity,
            "intended_order_type": intended_order_type,
            "intended_limit_price": intended_limit_price,
            "estimated_entry_debit_usd": estimated_entry_debit_usd,
            "declared_stop_fraction": declared_stop_fraction,
            "owner_intent": owner_intent,
            "corrects_event_id": corrects_event_id,
            "note": note,
            "previous_hash": state.head,
        }
        event["record_hash"] = _record_hash(event)
        # Run the same verifier used on reopened bytes before any write occurs.
        line = _canonical_json(event)
        occurred, appended = _validate_event_envelope(
            event, raw_line=line, state=state
        )
        _validate_and_apply_event(
            event, occurred=occurred, appended=appended, state=state
        )
        # Commit order is deliberate: durable journal record first, then an
        # atomic durable watermark advance, all under the journal's exclusive
        # advisory lock.  A crash between the two leaves a detectable
        # journal-ahead mismatch; it can never make a truncated journal pass.
        _append_bytes(fd, (line + "\n").encode("utf-8"))
        state.sequence = int(event["sequence"])
        state.head = str(event["record_hash"])
        watermark = _build_watermark(path, state)
        _write_watermark_bytes(
            watermark_path(path),
            (_canonical_json(watermark.payload()) + "\n").encode("utf-8"),
            create=False,
        )
        return event
    finally:
        os.close(fd)


def append_event(
    path: Path,
    *,
    kind: str,
    event_id: str,
    session: str,
    occurred_at: datetime | str,
    decision_id: str | None = None,
    monitoring_state: str | None = None,
    position_state: str | None = None,
    action: str | None = None,
    contract_osi: str | None = None,
    market_state_sha256: str | None = None,
    universe_sha256: str | None = None,
    prompt_event_id: str | None = None,
    spontaneous: bool | None = None,
    information_sources: Iterable[str] | None = None,
    confidence: float | None = None,
    reason_codes: Iterable[str] | None = None,
    program_contract_sha256: str | None = None,
    risk_contract_sha256: str | None = None,
    quantity: int | None = None,
    intended_order_type: str | None = None,
    intended_limit_price: float | None = None,
    estimated_entry_debit_usd: float | None = None,
    declared_stop_fraction: float | None = None,
    owner_intent: str | None = None,
    corrects_event_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Append using writer-owned wall and monotonic clocks.

    Production callers cannot inject either append clock. Deterministic clock
    injection exists only on the private ``_append_event_with_clocks`` test
    helper, so the 30-second decision seal cannot be forged through this API.
    """

    return _append_event_with_clocks(
        path,
        kind=kind,
        event_id=event_id,
        session=session,
        occurred_at=occurred_at,
        appended_at=None,
        local_monotonic_ns=None,
        decision_id=decision_id,
        monitoring_state=monitoring_state,
        position_state=position_state,
        action=action,
        contract_osi=contract_osi,
        market_state_sha256=market_state_sha256,
        universe_sha256=universe_sha256,
        prompt_event_id=prompt_event_id,
        spontaneous=spontaneous,
        information_sources=information_sources,
        confidence=confidence,
        reason_codes=reason_codes,
        program_contract_sha256=program_contract_sha256,
        risk_contract_sha256=risk_contract_sha256,
        quantity=quantity,
        intended_order_type=intended_order_type,
        intended_limit_price=intended_limit_price,
        estimated_entry_debit_usd=estimated_entry_debit_usd,
        declared_stop_fraction=declared_stop_fraction,
        owner_intent=owner_intent,
        corrects_event_id=corrects_event_id,
        note=note,
    )


def verify_log(
    path: Path,
    *,
    now: datetime | str | None = None,
    expected_head: str | None = None,
    expected_min_sequence: int | None = None,
) -> VerificationResult:
    """Fail closed on chain, state, watermark, or optional external anchors.

    The adjacent watermark is mandatory by default. ``expected_head`` is an
    exact out-of-band terminal anchor; ``expected_min_sequence`` refuses a
    sequence regression while allowing later legitimate appends.
    """

    path = _safe_jsonl_path(path, must_exist=True)
    fd = os.open(path, _open_flags(os.O_RDONLY))
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        state, watermark = _verified_state_and_watermark_locked(
            path,
            fd,
            expected_head=expected_head,
            expected_min_sequence=expected_min_sequence,
        )
    finally:
        os.close(fd)
    return _result_from_state(state, watermark, now=now)


def verify_watermark(path: Path) -> JournalWatermark:
    """Validate and return the self-hashed watermark bound to a full journal."""

    path = _safe_jsonl_path(path, must_exist=True)
    fd = os.open(path, _open_flags(os.O_RDONLY))
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        _, watermark = _verified_state_and_watermark_locked(path, fd)
        return watermark
    finally:
        os.close(fd)


def read_training_decisions(path: Path) -> tuple[dict[str, Any], ...]:
    """Return the closed, structured projection of verified DECISION records.

    The complete journal is verified before projection. Correction annotations,
    append clocks, sequence/hash-chain metadata, and every non-DECISION event are
    deliberately unavailable to a future training consumer.
    """

    path = _safe_jsonl_path(path, must_exist=True)
    fd = os.open(path, _open_flags(os.O_RDONLY))
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        state, _ = _verified_state_and_watermark_locked(path, fd)
    finally:
        os.close(fd)
    projected: list[dict[str, Any]] = []
    for record in state.records:
        if record["kind"] != DECISION:
            continue
        projected.append(
            {
                field_name: (
                    list(record[field_name])
                    if isinstance(record[field_name], list)
                    else record[field_name]
                )
                for field_name in TRAINING_DECISION_FIELDS
            }
        )
    return tuple(projected)


def log_status(
    path: Path,
    *,
    now: datetime | str | None = None,
    expected_head: str | None = None,
    expected_min_sequence: int | None = None,
) -> dict[str, Any]:
    """A concise verified status; silence is never reported as ``WAIT``."""

    return verify_log(
        path,
        now=now,
        expected_head=expected_head,
        expected_min_sequence=expected_min_sequence,
    ).payload()


class HumanDecisionLog:
    """Small path-bound facade for callers that prefer an object interface."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def initialize(
        cls,
        path: Path,
        *,
        log_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> "HumanDecisionLog":
        initialize_log(path, log_id=log_id, created_at=created_at)
        return cls(path)

    def append(self, **event: Any) -> dict[str, Any]:
        return append_event(self.path, **event)

    def _append_for_test(
        self,
        *,
        appended_at: datetime | str,
        local_monotonic_ns: int | None = None,
        **event: Any,
    ) -> dict[str, Any]:
        """Private deterministic-clock hook for offline fixtures and tests."""

        return _append_event_with_clocks(
            self.path,
            appended_at=appended_at,
            local_monotonic_ns=(
                time.monotonic_ns()
                if local_monotonic_ns is None
                else local_monotonic_ns
            ),
            **event,
        )

    def verify(
        self,
        *,
        now: datetime | str | None = None,
        expected_head: str | None = None,
        expected_min_sequence: int | None = None,
    ) -> VerificationResult:
        return verify_log(
            self.path,
            now=now,
            expected_head=expected_head,
            expected_min_sequence=expected_min_sequence,
        )

    def watermark(self) -> JournalWatermark:
        return verify_watermark(self.path)

    def status(
        self,
        *,
        now: datetime | str | None = None,
        expected_head: str | None = None,
        expected_min_sequence: int | None = None,
    ) -> dict[str, Any]:
        return log_status(
            self.path,
            now=now,
            expected_head=expected_head,
            expected_min_sequence=expected_min_sequence,
        )

    def training_decisions(self) -> tuple[dict[str, Any], ...]:
        return read_training_decisions(self.path)


def event_source_imports() -> tuple[str, ...]:
    """Auditable dependency surface used by the no-network unit test."""

    return (
        "fcntl",
        "hashlib",
        "json",
        "os",
        "re",
        "time",
        "uuid",
        "dataclasses",
        "datetime",
        "pathlib",
        "typing",
        "zoneinfo",
    )
