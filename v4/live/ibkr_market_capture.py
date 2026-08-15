"""Append-only IBKR market capture primitives for Protocol101 parity work."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import uuid


SCHEMA_VERSION = "IBKRMarketCaptureV1"
MANIFEST_VERSION = "IBKRMarketCaptureManifestV1"
QUALITY_VERSION = "IBKRMarketCaptureQualityV1"
COLLECTION_GATE_VERSION = "Protocol101CollectionGateV1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None = None) -> str:
    current = value or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(clean_json(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def stable_hash(value: Any) -> str:
    encoded = json.dumps(clean_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_capture_rows(path: Path) -> Iterable[tuple[int, dict[str, Any] | None]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None
                continue
            yield line_number, row if isinstance(row, dict) else None


def last_valid_sequence(path: Path) -> int:
    sequence = 0
    for _, row in iter_capture_rows(path):
        if row is None:
            continue
        try:
            sequence = max(sequence, int(row.get("sequence", 0)))
        except (TypeError, ValueError):
            continue
    return sequence


def option_snapshot_delta(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    """Return a lossless changed-field payload for an option ticker update."""
    if previous is None:
        return dict(current)
    return {
        "contract_id": current.get("contract_id"),
        "changes": {key: value for key, value in current.items() if previous.get(key) != value},
    }


def apply_option_capture_event(state: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    """Apply a full or delta option event to an in-memory quote state."""
    event_type = row.get("event_type")
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    contract_id = str(payload.get("contract_id") or "")
    if not contract_id:
        return
    if event_type == "option_update":
        state[contract_id] = dict(payload)
    elif event_type == "option_delta" and isinstance(payload.get("changes"), dict):
        state.setdefault(contract_id, {"contract_id": contract_id}).update(payload["changes"])


@dataclass(frozen=True)
class CapturePaths:
    root: Path
    events: Path
    state: Path
    manifest: Path
    checksums: Path

    @classmethod
    def for_capture(cls, capture_root: Path, session: str, capture_id: str) -> "CapturePaths":
        root = capture_root.expanduser() / session / capture_id
        return cls(
            root=root,
            events=root / "market_events.jsonl",
            state=root / "capture_state.json",
            manifest=root / "capture_manifest.json",
            checksums=root / "checksums.sha256",
        )


class CaptureWriter:
    """Append-only writer. Existing market rows are never rewritten."""

    def __init__(self, paths: CapturePaths, *, session: str, capture_id: str) -> None:
        self.paths = paths
        self.session = session
        self.capture_id = capture_id
        self.producer_instance_id = str(uuid.uuid4())
        self.connection_epoch = 0
        self.sequence = last_valid_sequence(paths.events)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.paths.root / ".writer.lock").open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            raise RuntimeError(f"capture writer already active for {self.paths.root}") from exc
        self._handle = self.paths.events.open("a", encoding="utf-8", buffering=1)
        self.event_counts: dict[str, int] = {}
        self.started_at = iso_utc()
        self.last_received_at: str | None = None
        self.last_heartbeat_at: str | None = None
        self.write_errors = 0

    def new_connection_epoch(self) -> int:
        self.connection_epoch += 1
        return self.connection_epoch

    def append(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        event_timestamp_utc: str | datetime | None = None,
        flush_to_disk: bool = False,
    ) -> dict[str, Any]:
        self.sequence += 1
        received = iso_utc()
        source_time = event_timestamp_utc
        if isinstance(source_time, datetime):
            source_time = iso_utc(source_time)
        row = {
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "producer_instance_id": self.producer_instance_id,
            "connection_epoch": self.connection_epoch,
            "session": self.session,
            "capture_id": self.capture_id,
            "event_type": str(event_type),
            "event_timestamp_utc": source_time,
            "received_timestamp_utc": received,
            "payload": clean_json(payload or {}),
        }
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        try:
            self._handle.write(encoded + "\n")
            self._handle.flush()
            if flush_to_disk:
                os.fsync(self._handle.fileno())
        except OSError:
            self.write_errors += 1
            raise
        self.last_received_at = received
        if event_type == "heartbeat":
            self.last_heartbeat_at = received
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        return row

    def write_state(self, **extra: Any) -> dict[str, Any]:
        state = {
            "schema_version": SCHEMA_VERSION,
            "status": extra.pop("status", "running"),
            "session": self.session,
            "capture_id": self.capture_id,
            "producer_instance_id": self.producer_instance_id,
            "connection_epoch": self.connection_epoch,
            "last_sequence": self.sequence,
            "started_at_utc": self.started_at,
            "updated_at_utc": iso_utc(),
            "last_received_at_utc": self.last_received_at,
            "last_heartbeat_at_utc": self.last_heartbeat_at,
            "events_path": str(self.paths.events),
            "events_bytes": self.paths.events.stat().st_size if self.paths.events.exists() else 0,
            "event_counts_process": dict(sorted(self.event_counts.items())),
            "write_errors": self.write_errors,
            "broker_order_endpoint_called": False,
            "real_money_trading": False,
            **clean_json(extra),
        }
        atomic_write_json(self.paths.state, state)
        return state

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
        if not self._lock_handle.closed:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()

    def __enter__(self) -> "CaptureWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
