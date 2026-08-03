"""Build a distinct corrected-v3.2 two-clock development foundation.

The corrected-v3.2 Path-D minute-entry rows are not byte-equivalent to the
older FT1D two-clock campaign.  This module therefore materializes additive
two-clock exit metadata into a new namespace and never mixes the generations.
It has no holdout-loading API.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import NeuralDatasetConfig
from v4.model.protocol101_regimen_repair import TWO_CLOCK_PROCESSED_ROW_SCHEMA
from v4.scripts.materialize_protocol101_ft1d_two_clock_rows import (
    ADDITIVE_FIELDS,
    _attach_two_clock_labels,
    values_equal,
)

from .dataset import ROOT, protected_sessions, sha256_path, stable_hash


CORRECTED_ROOT = Path(
    "/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31"
)
SESSION_MANIFEST = ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_2_2026_08_01/"
    "session_assignments.json"
)
OUTPUT_ROOT = ROOT / (
    "v4/audit/autoresearch/autoresearch_v2_corrected_v3_two_clock"
)
PROCESSED_ROOT = OUTPUT_ROOT / "processed"
RECEIPT_ROOT = OUTPUT_ROOT / "receipts"
FOUNDATION_PATH = ROOT / (
    "v4/research/autoresearch_v2/foundations/"
    "development_2025-08-01_2026-06-09_corrected_v3_two_clock.json"
)
SCHEMA_VERSION = "autoresearch_v2.corrected_v3_two_clock_receipt.v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def eligible_sessions() -> tuple[str, ...]:
    assignments = json.loads(SESSION_MANIFEST.read_text())
    forbidden = set(protected_sessions(SESSION_MANIFEST)) | set(
        assignments.get("opra_degraded_diagnostic_only", ())
    )
    sessions = tuple(
        str(session)
        for session in assignments["pre_holdout_session_indices_1_215"]
        if session not in forbidden
    )
    if len(sessions) != 214 or set(sessions) & forbidden:
        raise RuntimeError("corrected-v3 development session boundary drifted")
    return sessions


def _source_paths(session: str) -> dict[str, Path]:
    aligned = CORRECTED_ROOT / "aligned"
    return {
        "legacy": aligned / "processed/minute_entry" / f"{session}.pkl",
        "normalized": aligned
        / "normalized"
        / f"databento_spxw_0dte_{session}_official_context.parquet",
        "output": PROCESSED_ROOT / f"{session}.pkl",
        "receipt": RECEIPT_ROOT / f"{session}.json",
    }


def _validate_additive_parity(
    legacy: list[dict[str, Any]], materialized: list[dict[str, Any]], *, session: str
) -> None:
    if len(legacy) != len(materialized) or not legacy:
        raise RuntimeError(f"corrected_v3_row_count_mismatch:{session}")
    for index, (before, after) in enumerate(zip(legacy, materialized)):
        if set(after) - set(before) - ADDITIVE_FIELDS:
            raise RuntimeError(f"corrected_v3_unexpected_field:{session}:{index}")
        for key, value in before.items():
            if key not in after or not values_equal(value, after[key]):
                raise RuntimeError(
                    f"corrected_v3_legacy_field_mismatch:{session}:{index}:{key}"
                )
        if after.get("processed_row_schema_version") != TWO_CLOCK_PROCESSED_ROW_SCHEMA:
            raise RuntimeError(f"corrected_v3_schema_mismatch:{session}:{index}")


def _expected_receipt(session: str, paths: Mapping[str, Path]) -> dict[str, Any] | None:
    if not paths["output"].is_file() or not paths["receipt"].is_file():
        return None
    payload = json.loads(paths["receipt"].read_text())
    semantic = dict(payload)
    observed_receipt_hash = semantic.pop("receipt_sha256", None)
    if observed_receipt_hash != stable_hash(semantic):
        return None
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_additive_two_clock_materialization",
        "session": session,
        "role": "development",
        "holdout_access_count": 0,
        "source_hashes": {
            "legacy": sha256_path(paths["legacy"]),
            "normalized": sha256_path(paths["normalized"]),
            "session_manifest": sha256_path(SESSION_MANIFEST),
        },
        "output_path": str(paths["output"].relative_to(ROOT)),
        "output_sha256": sha256_path(paths["output"]),
        "row_count": payload.get("row_count"),
        "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "legacy_field_parity": True,
        "materialization_method": "corrected_v3_frozen_row_additive_label_attachment",
    }
    if semantic != expected:
        return None
    return payload


def materialize_session(session: str) -> dict[str, Any]:
    if session not in eligible_sessions():
        raise RuntimeError(f"session outside corrected-v3 development boundary:{session}")
    paths = _source_paths(session)
    for name in ("legacy", "normalized"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"corrected_v3_source_missing:{session}:{name}")
    existing = _expected_receipt(session, paths)
    if existing is not None:
        return existing
    if paths["output"].exists() or paths["receipt"].exists():
        raise RuntimeError(f"corrected_v3_partial_or_drifted_output:{session}")
    with paths["legacy"].open("rb") as handle:
        legacy = pickle.load(handle)
    if not isinstance(legacy, list):
        raise RuntimeError(f"corrected_v3_legacy_not_list:{session}")
    normalized = pq.read_table(paths["normalized"])
    config = NeuralDatasetConfig(
        feature_contract="protocol101-live-v2-microstructure-masked",
        compute_policy_labels=True,
        processed_row_schema_version=TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    )
    materialized = _attach_two_clock_labels(legacy, normalized, config=config)
    _validate_additive_parity(legacy, materialized, session=session)
    _atomic_pickle(paths["output"], materialized)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_additive_two_clock_materialization",
        "session": session,
        "role": "development",
        "holdout_access_count": 0,
        "source_hashes": {
            "legacy": sha256_path(paths["legacy"]),
            "normalized": sha256_path(paths["normalized"]),
            "session_manifest": sha256_path(SESSION_MANIFEST),
        },
        "output_path": str(paths["output"].relative_to(ROOT)),
        "output_sha256": sha256_path(paths["output"]),
        "row_count": len(materialized),
        "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "legacy_field_parity": True,
        "materialization_method": "corrected_v3_frozen_row_additive_label_attachment",
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    _atomic_json(paths["receipt"], receipt)
    return receipt


def freeze_foundation(receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(receipts, key=lambda item: str(item["session"]))
    expected = eligible_sessions()
    if tuple(str(item["session"]) for item in ordered) != expected:
        raise RuntimeError("corrected-v3 foundation receipt coverage mismatch")
    rows = []
    for receipt in ordered:
        session = str(receipt["session"])
        paths = _source_paths(session)
        verified = _expected_receipt(session, paths)
        if verified is None:
            raise RuntimeError(f"corrected-v3 receipt failed re-verification:{session}")
        rows.append(
            {
                "session": session,
                "path": str(paths["output"].relative_to(ROOT)),
                "sha256": str(receipt["output_sha256"]),
                "receipt_path": str(paths["receipt"].relative_to(ROOT)),
                "receipt_sha256": sha256_path(paths["receipt"]),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "autoresearch_v2.development_foundation.v1",
        "generation": "corrected-v3.2-distinct-two-clock",
        "role": "development",
        "holdout_access_count": 0,
        "session_manifest": str(SESSION_MANIFEST.relative_to(ROOT)),
        "session_manifest_sha256": sha256_path(SESSION_MANIFEST),
        "session_count": len(rows),
        "first_session": expected[0],
        "last_session": expected[-1],
        "sessions": rows,
    }
    payload["foundation_sha256"] = stable_hash(payload)
    if FOUNDATION_PATH.exists():
        if json.loads(FOUNDATION_PATH.read_text()) != payload:
            raise RuntimeError("corrected-v3 foundation already exists with different bytes")
    else:
        _atomic_json(FOUNDATION_PATH, payload)
    return payload


def materialize_all(*, max_workers: int) -> dict[str, Any]:
    sessions = eligible_sessions()
    receipts: dict[str, Mapping[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(materialize_session, session): session for session in sessions
        }
        for future in as_completed(futures):
            session = futures[future]
            receipts[session] = future.result()
            if len(receipts) % 20 == 0 or len(receipts) == len(sessions):
                print(f"materialized_or_verified={len(receipts)}/{len(sessions)}")
    foundation = freeze_foundation(list(receipts.values()))
    return {
        "status": "complete",
        "session_count": foundation["session_count"],
        "foundation_path": str(FOUNDATION_PATH),
        "foundation_sha256": foundation["foundation_sha256"],
        "holdout_access_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(materialize_all(max_workers=args.max_workers), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
