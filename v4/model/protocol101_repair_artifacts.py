"""Immutable two-clock replay artifacts for Protocol101 simulator v5.

The writer is deliberately economics-agnostic.  It serializes already-created
payloads, verifies all identities and hashes, writes each payload atomically,
and commits the root manifest last.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    Protocol101MixedSimulatorVersionError,
)


SCHEMA_VERSION = "Protocol101ImmutableReplayArtifactSchemaV2TwoClockExit"
REQUIRED_PAYLOAD_FILES = (
    "candidate_intents.jsonl",
    "exit_quote_age_report.json",
    "fold_metrics.json",
    "pooled_metrics.json",
    "session_metrics.json",
    "skipped_events.jsonl",
    "trades.jsonl",
)
REQUIRED_MANIFEST_HASHES = (
    "processed_corpus_hash",
    "fold_governance_hash",
    "acceptance_registry_hash",
    "feature_contract_hash",
    "policy_contract_hash",
    "two_clock_exit_contract_hash",
    "model_or_equivalence_certificate_hash",
    "threshold_hash",
    "epsilon_hash",
    "selection_contract_hash",
    "simulator_source_hash",
    "simulator_config_hash",
    "candidate_stream_hash",
    "candidate_payload_hash",
    "trade_identity_hash",
    "exit_quote_age_report_hash",
)
JSONL_FILES = frozenset(
    {
        "candidate_intents.jsonl",
        "skipped_events.jsonl",
        "trades.jsonl",
    }
)


class Protocol101ArtifactError(RuntimeError):
    blocker_code = "P101_ARTIFACT_ERROR"

    def __init__(self, message: str, **payload: Any) -> None:
        super().__init__(message)
        self.payload = {"blocker_code": self.blocker_code, **payload}


class Protocol101ArtifactHashMismatchError(Protocol101ArtifactError):
    blocker_code = "P101_ARTIFACT_HASH_MISMATCH"


class Protocol101ArtifactSchemaError(Protocol101ArtifactError):
    blocker_code = "P101_ARTIFACT_SCHEMA_ERROR"


@dataclass(frozen=True)
class ArtifactWriteResult:
    status: str
    packet_dir: str
    voided_partial_packet: str | None
    manifest_hash: str
    write_order: tuple[str, ...]


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Protocol101ArtifactSchemaError(
                "canonical artifacts prohibit NaN and infinity"
            )
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _canonical(asdict(value))
    raise Protocol101ArtifactSchemaError(
        f"unsupported canonical artifact value: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any, *, trailing_lf: bool = True) -> bytes:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_lf else b"")


def canonical_jsonl_bytes(rows: Iterable[Any]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array_hash(rows: Iterable[Any]) -> str:
    return sha256_bytes(canonical_json_bytes(list(rows), trailing_lf=False))


def semantic_payload_hashes(
    payloads: Mapping[str, Any],
) -> dict[str, str]:
    """Derive the four payload-governed hashes required by the v2 manifest."""

    candidates = list(payloads["candidate_intents.jsonl"])
    candidate_stream = [
        {
            name: row[name]
            for name in (
                "split",
                "session",
                "decision_time_ns",
                "contract_id",
                "right",
                "canonical_strike_slot",
                "policy_index",
            )
        }
        for row in candidates
    ]
    candidate_payload = [
        {
            **candidate_stream[index],
            **{
                name: row[name]
                for name in (
                    "entry_ask",
                    "score",
                    "raw_label_pnl_after_campaign_fee",
                    "label_mid_pnl_before_campaign_fee",
                    "label_realized_exit_time_ns",
                    "label_source_exit_quote_time_ns",
                    "label_exit_quote_age_ms",
                    "label_exit_reason_code",
                    "label_executable_exit_bid",
                    "label_policy_deadline_ns",
                    "label_invalid_reason_code",
                    "feature_hash",
                    "source_quote_time_ns",
                    "source_context_time_ns",
                    "strategy",
                    "metadata",
                )
            },
        }
        for index, row in enumerate(candidates)
    ]
    trade_identity = [
        {
            name: row.get(name)
            for name in (
                "split",
                "fold",
                "session",
                "decision_time_ns",
                "contract_id",
                "policy_index",
                "label_source_exit_quote_time_ns",
                "label_realized_exit_time_ns",
            )
        }
        for row in payloads["trades.jsonl"]
    ]
    return {
        "candidate_stream_hash": _canonical_array_hash(candidate_stream),
        "candidate_payload_hash": _canonical_array_hash(candidate_payload),
        "trade_identity_hash": _canonical_array_hash(trade_identity),
        "exit_quote_age_report_hash": sha256_bytes(
            canonical_json_bytes(
                payloads["exit_quote_age_report.json"], trailing_lf=False
            )
        ),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _payload_bytes(
    payloads: Mapping[str, Any],
) -> dict[str, bytes]:
    missing = sorted(set(REQUIRED_PAYLOAD_FILES) - set(payloads))
    extra = sorted(set(payloads) - set(REQUIRED_PAYLOAD_FILES))
    if missing or extra:
        raise Protocol101ArtifactSchemaError(
            "replay payload file set does not match signed schema",
            missing=missing,
            extra=extra,
        )
    return {
        name: (
            canonical_jsonl_bytes(payloads[name])
            if name in JSONL_FILES
            else canonical_json_bytes(payloads[name])
        )
        for name in REQUIRED_PAYLOAD_FILES
    }


def _assert_v5_only(
    payloads: Mapping[str, Any],
    manifest_fields: Mapping[str, Any],
) -> None:
    declared = str(manifest_fields.get("simulator_version") or "")
    if declared != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
        raise Protocol101MixedSimulatorVersionError(
            "immutable repaired packet requires the exact v5 simulator marker",
            canonical_key=declared,
            boundary="artifact preflight before hashing",
        )
    for name in ("candidate_intents.jsonl", "trades.jsonl"):
        for index, row in enumerate(payloads[name]):
            observed = str(
                row.get("source_simulator_version")
                or row.get("simulator_version")
                or declared
            )
            if observed != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
                raise Protocol101MixedSimulatorVersionError(
                    "mixed simulator version in immutable replay payload",
                    canonical_key=(name, index, observed),
                    boundary="artifact preflight before hashing",
                )


def _validate_manifest_fields(fields: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_MANIFEST_HASHES if name not in fields]
    malformed = [
        name
        for name in REQUIRED_MANIFEST_HASHES
        if name in fields
        and (
            not isinstance(fields[name], str)
            or len(fields[name]) != 64
            or any(character not in "0123456789abcdef" for character in fields[name])
        )
    ]
    if missing or malformed:
        raise Protocol101ArtifactSchemaError(
            "root manifest is missing required canonical SHA-256 identities",
            missing=missing,
            malformed=malformed,
        )


def _parse_hashes_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in parsed:
            raise Protocol101ArtifactHashMismatchError(
                "malformed or duplicate hashes.sha256 entry", path=str(path)
            )
        parsed[name] = digest
    return parsed


def verify_replay_packet(packet_dir: Path) -> dict[str, Any]:
    manifest_path = packet_dir / "manifest.json"
    hashes_path = packet_dir / "hashes.sha256"
    required = {
        *REQUIRED_PAYLOAD_FILES,
        "hashes.sha256",
        "manifest.json",
    }
    missing = sorted(name for name in required if not (packet_dir / name).is_file())
    if missing:
        raise Protocol101ArtifactHashMismatchError(
            "existing packet is incomplete", path=str(packet_dir), missing=missing
        )
    hashes = _parse_hashes_file(hashes_path)
    if set(hashes) != {*REQUIRED_PAYLOAD_FILES, "manifest.json"}:
        raise Protocol101ArtifactHashMismatchError(
            "existing byte-hash manifest has the wrong file set",
            path=str(hashes_path),
        )
    for name, expected in hashes.items():
        observed = sha256_file(packet_dir / name)
        if observed != expected:
            raise Protocol101ArtifactHashMismatchError(
                "existing replay payload hash mismatch",
                path=str(packet_dir / name),
                expected=expected,
                observed=observed,
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise Protocol101ArtifactHashMismatchError(
            "existing root manifest schema mismatch", path=str(manifest_path)
        )
    payload_hashes = {
        name: digest for name, digest in hashes.items() if name != "manifest.json"
    }
    if manifest.get("file_hashes") != payload_hashes:
        raise Protocol101ArtifactHashMismatchError(
            "root manifest and hashes.sha256 disagree", path=str(manifest_path)
        )
    return manifest


def _void_partial_packet(packet_dir: Path) -> Path:
    void_root = packet_dir.parent / "void_outputs"
    void_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = void_root / f"{packet_dir.name}_partial_attempt{index:03d}"
        if not destination.exists():
            break
        index += 1
    packet_dir.rename(destination)
    receipt = {
        "status": "VOID_PARTIAL_PACKET",
        "reason_code": "P101_ARTIFACT_PARTIAL_PACKET_RESTART",
        "original_path": str(packet_dir),
        "void_path": str(destination),
        "observed_files": sorted(
            path.name for path in destination.iterdir() if path.is_file()
        ),
    }
    _atomic_write(destination / "void_receipt.json", canonical_json_bytes(receipt))
    return destination


def write_immutable_replay_packet(
    packet_dir: Path,
    *,
    payloads: Mapping[str, Any],
    manifest_fields: Mapping[str, Any],
) -> ArtifactWriteResult:
    """Write or deterministically resume one v2 packet.

    A packet with a root manifest is treated as committed: any discrepancy is
    a hard hash failure.  A directory without a manifest is partial and is
    moved aside before a fresh namespace is written.
    """

    packet_dir = Path(packet_dir)
    _assert_v5_only(payloads, manifest_fields)
    _validate_manifest_fields(manifest_fields)
    semantic_hashes = semantic_payload_hashes(payloads)
    mismatches = {
        name: {
            "declared": manifest_fields[name],
            "observed": observed,
        }
        for name, observed in semantic_hashes.items()
        if manifest_fields[name] != observed
    }
    if mismatches:
        raise Protocol101ArtifactHashMismatchError(
            "root semantic hashes do not match canonical replay payloads",
            mismatches=mismatches,
        )
    encoded = _payload_bytes(payloads)
    file_hashes = {name: sha256_bytes(encoded[name]) for name in encoded}
    root_manifest = {
        **_canonical(dict(manifest_fields)),
        "schema_version": SCHEMA_VERSION,
        "status": "complete_pending_independent_acceptance",
        "file_hashes": dict(sorted(file_hashes.items())),
        "manifest_written_last": True,
    }
    manifest_payload = canonical_json_bytes(root_manifest)
    manifest_hash = sha256_bytes(manifest_payload)
    byte_hashes = {**file_hashes, "manifest.json": manifest_hash}
    hashes_payload = "".join(
        f"{byte_hashes[name]}  {name}\n" for name in sorted(byte_hashes)
    ).encode("utf-8")

    voided: Path | None = None
    if packet_dir.exists():
        if (packet_dir / "manifest.json").exists():
            existing = verify_replay_packet(packet_dir)
            if canonical_json_bytes(existing) != manifest_payload:
                raise Protocol101ArtifactHashMismatchError(
                    "complete packet differs from requested immutable packet",
                    path=str(packet_dir),
                )
            return ArtifactWriteResult(
                status="verified_existing_complete_packet_skipped",
                packet_dir=str(packet_dir),
                voided_partial_packet=None,
                manifest_hash=manifest_hash,
                write_order=(),
            )
        voided = _void_partial_packet(packet_dir)

    packet_dir.mkdir(parents=True, exist_ok=False)
    write_order: list[str] = []
    for name in REQUIRED_PAYLOAD_FILES:
        _atomic_write(packet_dir / name, encoded[name])
        write_order.append(name)
    _atomic_write(packet_dir / "hashes.sha256", hashes_payload)
    write_order.append("hashes.sha256")
    _atomic_write(packet_dir / "manifest.json", manifest_payload)
    write_order.append("manifest.json")
    verify_replay_packet(packet_dir)
    return ArtifactWriteResult(
        status="created_complete_packet",
        packet_dir=str(packet_dir),
        voided_partial_packet=str(voided) if voided is not None else None,
        manifest_hash=manifest_hash,
        write_order=tuple(write_order),
    )
