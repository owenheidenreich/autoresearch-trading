"""Lossless archive plus compact index for fresh Protocol101 unit summaries."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


COMPACT_SUMMARY_SCHEMA = "Protocol101FreshEntryUnitCompactSummaryV1"
ARCHIVE_RECEIPT_SCHEMA = "Protocol101FreshEntryFullSummaryArchiveReceiptV1"
ARCHIVE_NAME = "summary.full.json.gz"
ARCHIVE_RECEIPT_NAME = "summary.full.archive_receipt.json"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _write_bytes_immutable(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable_summary_archive_conflict:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    ).encode()
    _write_bytes_immutable(path, encoded)


def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _calibration_observations(
    validation: Mapping[str, Any],
    *,
    fee: float,
) -> list[dict[str, float]]:
    return [
        {
            "confidence": float(item["calibrated_confidence"]),
            "won": float(
                float(item["selected_label_before_fee"]) - float(fee) > 0.0
            ),
        }
        for item in validation["diagnostics"]
    ]


def _compact_payload(
    full: Mapping[str, Any],
    *,
    archive_path: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    compact = deepcopy(dict(full))
    unit = compact["unit"]
    validation = unit["validation"]
    observations = _calibration_observations(
        validation,
        fee=float(unit["config"]["fee"]),
    )
    unit["validation"] = {
        "primary_noise_scale": validation["primary_noise_scale"],
        "primary_noise_seed": validation["primary_noise_seed"],
        "decision_count": validation["decision_count"],
        "candidate_count": validation["candidate_count"],
        "metrics": validation["metrics"],
        "expected_calibration_error": validation[
            "expected_calibration_error"
        ],
        "calibration_observations": validation[
            "calibration_observations"
        ],
        "calibration_observations_compact": observations,
        "fee_sensitivity": validation["fee_sensitivity"],
        "fill_edge_band": validation["fill_edge_band"],
        "noise_diagnostics": validation["noise_diagnostics"],
        "simulator_semantics": validation["simulator_semantics"],
        "archived_fields": [
            "diagnostics",
            "entry_intents",
            "trades",
            "skipped_events",
        ],
        "immutable_replay_packet_is_candidate_trade_source": True,
    }
    compact["schema_version"] = COMPACT_SUMMARY_SCHEMA
    compact["full_summary_archive"] = {
        "path": str(archive_path),
        "sha256": receipt["compressed_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_path(receipt_path),
        "uncompressed_sha256": receipt["uncompressed_sha256"],
        "roundtrip_exact": True,
    }
    compact.pop("summary_hash", None)
    compact["summary_hash"] = stable_hash(compact)
    return compact


def validate_compact_summary(summary_path: Path) -> dict[str, Any]:
    summary_path = Path(summary_path)
    compact = json.loads(summary_path.read_text())
    if compact.get("schema_version") != COMPACT_SUMMARY_SCHEMA:
        raise RuntimeError(f"summary_not_compact:{summary_path}")
    expected_hash = compact.get("summary_hash")
    without_hash = dict(compact)
    without_hash.pop("summary_hash", None)
    if expected_hash != stable_hash(without_hash):
        raise RuntimeError(f"compact_summary_hash_mismatch:{summary_path}")
    binding = compact.get("full_summary_archive") or {}
    archive_path = Path(str(binding.get("path") or ""))
    receipt_path = Path(str(binding.get("receipt_path") or ""))
    if (
        not archive_path.is_file()
        or not receipt_path.is_file()
        or sha256_path(archive_path) != binding.get("sha256")
        or sha256_path(receipt_path) != binding.get("receipt_sha256")
    ):
        raise RuntimeError(f"compact_summary_archive_binding_mismatch:{summary_path}")
    receipt = json.loads(receipt_path.read_text())
    receipt_without_hash = dict(receipt)
    receipt_hash = receipt_without_hash.pop("receipt_sha256", None)
    if receipt_hash != stable_hash(receipt_without_hash):
        raise RuntimeError(f"summary_archive_receipt_hash_mismatch:{summary_path}")
    raw = gzip.decompress(archive_path.read_bytes())
    if (
        sha256_bytes(raw) != receipt["uncompressed_sha256"]
        or len(raw) != receipt["uncompressed_bytes"]
        or json.loads(raw).get("summary_hash")
        != receipt["original_summary_hash"]
    ):
        raise RuntimeError(f"summary_archive_roundtrip_mismatch:{summary_path}")
    return compact


def compact_summary(summary_path: Path) -> dict[str, Any]:
    """Archive the full JSON exactly and install a compact resumable index."""

    summary_path = Path(summary_path)
    current = json.loads(summary_path.read_text())
    if current.get("schema_version") == COMPACT_SUMMARY_SCHEMA:
        return validate_compact_summary(summary_path)
    raw = summary_path.read_bytes()
    if json.loads(raw) != current:
        raise RuntimeError(f"full_summary_parse_instability:{summary_path}")
    original_without_hash = dict(current)
    original_hash = original_without_hash.pop("summary_hash", None)
    if original_hash != stable_hash(original_without_hash):
        raise RuntimeError(f"full_summary_hash_mismatch:{summary_path}")
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    archive_path = summary_path.with_name(ARCHIVE_NAME)
    receipt_path = summary_path.with_name(ARCHIVE_RECEIPT_NAME)
    _write_bytes_immutable(archive_path, compressed)
    if gzip.decompress(archive_path.read_bytes()) != raw:
        raise RuntimeError(f"full_summary_archive_roundtrip_failed:{summary_path}")
    receipt = {
        "schema_version": ARCHIVE_RECEIPT_SCHEMA,
        "status": "full_summary_losslessly_archived",
        "summary_path": str(summary_path),
        "archive_path": str(archive_path),
        "uncompressed_sha256": sha256_bytes(raw),
        "compressed_sha256": sha256_path(archive_path),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": archive_path.stat().st_size,
        "original_summary_hash": original_hash,
        "compactor_source_sha256": sha256_path(Path(__file__)),
        "roundtrip_exact": True,
        "scientific_content_deleted": False,
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    _write_json_immutable(receipt_path, receipt)
    compact = _compact_payload(
        current,
        archive_path=archive_path,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    _write_json_replace(summary_path, compact)
    return validate_compact_summary(summary_path)
