"""Materialize signed two-clock rows for the owner-authorized FT1D campaign.

The governed campaign manifest predates the additive two-clock row schema.
This adapter rebuilds those fields from the exact local normalized inputs and
admits a row only when every pre-existing field remains identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import (
    NeuralDatasetConfig,
    _contract_quote_path,
    _prepare_options,
    label_for_policy_two_clock_from_path,
)
from v4.model.protocol101_regimen_repair import (
    INT64_MISSING,
    ExitReason,
    InvalidReason,
    TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    assert_processed_row_identities,
)
from v4.scripts.protocol101_training_scope import TrainingScope, load_training_scope


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_entry_campaign_execution_attempt001"
)
OUTPUT_ROOT = EXECUTION_ROOT / "two_clock_processed"
SESSION_RECEIPT_ROOT = EXECUTION_ROOT / "two_clock_materialization_receipts"
GLOBAL_RECEIPT = EXECUTION_ROOT / "two_clock_materialization_receipt.json"
PROGRESS_PATH = EXECUTION_ROOT / "two_clock_materialization_progress.json"
MANIFEST_PATH = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_live_v2_microstructure_masked_15mo_training_preflight/"
    "canonical_processed_session_manifest.json"
)
FEATURE_CONTRACT = "protocol101-live-v2-microstructure-masked"
SCHEMA_VERSION = "Protocol101FT1DTwoClockMaterializationV1"
SESSION_SCHEMA_VERSION = "Protocol101FT1DTwoClockSessionMaterializationV1"
SLOW_FULL_REBUILD_PRODUCER_SHA256 = (
    "0677758ef50f8b4be689ca2b8359fa4c5f433c793c2bb1676977e20e5f91f354"
)
ADDITIVE_FIELDS = frozenset(
    {
        "processed_row_schema_version",
        "label_realized_exit_time_ns",
        "label_source_exit_quote_time_ns",
        "label_exit_quote_age_ms",
        "label_exit_reason_code",
        "label_executable_exit_bid",
        "label_policy_deadline_ns",
        "label_policy_index",
        "label_invalid_reason_code",
        "candidate_filter_trace",
        "ladder_context",
    }
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() <= right.keys()
            and all(values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(values_equal(a, b) for a, b in zip(left, right))
        )
    try:
        return bool(
            np.array_equal(
                np.asarray(left),
                np.asarray(right),
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        try:
            return bool(np.all(left == right))
        except Exception:
            return False


def assert_additive_parity(
    legacy_rows: list[dict[str, Any]],
    materialized_rows: list[dict[str, Any]],
    *,
    session: str,
) -> None:
    if len(legacy_rows) != 359 or len(materialized_rows) != len(legacy_rows):
        raise RuntimeError(f"two_clock_row_count_mismatch:{session}")
    for index, (legacy, materialized) in enumerate(
        zip(legacy_rows, materialized_rows)
    ):
        if not isinstance(legacy, dict) or not isinstance(materialized, dict):
            raise RuntimeError(f"two_clock_non_mapping_row:{session}:{index}")
        unexpected = set(materialized) - set(legacy) - ADDITIVE_FIELDS
        if unexpected:
            raise RuntimeError(
                f"two_clock_unexpected_additive_fields:{session}:{index}:"
                f"{sorted(unexpected)}"
            )
        for key, legacy_value in legacy.items():
            if key not in materialized or not values_equal(
                legacy_value,
                materialized[key],
            ):
                raise RuntimeError(
                    f"two_clock_legacy_field_mismatch:{session}:{index}:{key}"
                )
        if (
            materialized.get("processed_row_schema_version")
            != TWO_CLOCK_PROCESSED_ROW_SCHEMA
        ):
            raise RuntimeError(
                f"two_clock_schema_mismatch:{session}:{index}"
            )
    assert_processed_row_identities(
        materialized_rows,
        split="FT1D-TWO-CLOCK-MATERIALIZATION",
        session=session,
        boundary="campaign two-clock materialization before any fit",
    )


def _manifest_entries() -> dict[str, dict[str, Any]]:
    payload = json.loads(MANIFEST_PATH.read_text())
    entries = {
        str(item["session"]): dict(item)
        for item in payload.get("included_sessions", [])
    }
    if len(entries) != 301:
        raise RuntimeError("governed_manifest_session_grid_mismatch")
    return entries


def _official_index_path(kind: str, session: str) -> Path:
    return ROOT / f"data/raw/index/{kind}_1m/{session}.official_{kind}.parquet"


def _session_paths(
    session: str,
    legacy_path: Path,
    manifest_entry: dict[str, Any],
) -> dict[str, Path]:
    declared_legacy = ROOT / str(manifest_entry["processed_file"])
    if declared_legacy.resolve() != legacy_path.resolve():
        raise RuntimeError(f"governed_legacy_path_mismatch:{session}")
    return {
        "legacy": declared_legacy,
        "normalized_official_context": (
            ROOT / str(manifest_entry["normalized_official_context_file"])
        ),
        "spx": _official_index_path("spx", session),
        "vix": _official_index_path("vix", session),
        "output": OUTPUT_ROOT / f"{session}.pkl",
        "receipt": SESSION_RECEIPT_ROOT / f"{session}.json",
    }


def _verify_existing(
    *,
    session: str,
    paths: dict[str, Path],
) -> dict[str, Any] | None:
    output = paths["output"]
    receipt_path = paths["receipt"]
    if not output.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    source_hashes = {
        name: sha256_path(paths[name])
        for name in ("legacy", "normalized_official_context", "spx", "vix")
    }
    legacy_expected = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "verified_additive_two_clock_materialization",
        "session": session,
        "source_hashes": source_hashes,
        "output_path": str(output.relative_to(ROOT)),
        "output_sha256": sha256_path(output),
        "row_count": 359,
        "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "legacy_field_parity": True,
    }
    legacy_expected["receipt_sha256"] = stable_hash(legacy_expected)
    if receipt == legacy_expected:
        migrated = {
            key: value
            for key, value in legacy_expected.items()
            if key != "receipt_sha256"
        }
        migrated.update(
            {
                "materialization_method": "full_row_rebuild_exact_parity",
                "producer_source_sha256": (
                    SLOW_FULL_REBUILD_PRODUCER_SHA256
                ),
            }
        )
        migrated["receipt_sha256"] = stable_hash(migrated)
        write_json_atomic(receipt_path, migrated)
        return migrated
    without_hash = dict(receipt)
    receipt_hash = without_hash.pop("receipt_sha256", None)
    if (
        receipt_hash == stable_hash(without_hash)
        and all(
            receipt.get(key) == value
            for key, value in legacy_expected.items()
            if key != "receipt_sha256"
        )
        and receipt.get("materialization_method")
        in {
            "full_row_rebuild_exact_parity",
            "frozen_row_additive_label_attachment",
        }
        and isinstance(receipt.get("producer_source_sha256"), str)
    ):
        return receipt
    return None


def _void_partial(paths: dict[str, Path], session: str) -> None:
    existing = [paths[name] for name in ("output", "receipt") if paths[name].exists()]
    if not existing:
        return
    root = EXECUTION_ROOT / "voided_mechanical_artifacts" / "two_clock_materialization"
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = root / f"{session}_attempt{index:03d}"
        if not destination.exists():
            break
        index += 1
    destination.mkdir()
    for path in existing:
        shutil.move(str(path), destination / path.name)
    write_json_atomic(
        destination / "VOID.json",
        {
            "schema_version": "Protocol101FT1DVoidedMechanicalArtifactV1",
            "status": "void_not_campaign_evidence",
            "blocker_classification": "mechanical_non_scientific",
            "blocker_code": "partial_or_hash_mismatched_two_clock_materialization",
            "session": session,
        },
    )


def _materialize_one(
    session: str,
    legacy_path_text: str,
    manifest_entry: dict[str, Any],
) -> dict[str, Any]:
    legacy_path = Path(legacy_path_text)
    paths = _session_paths(session, legacy_path, manifest_entry)
    existing = _verify_existing(session=session, paths=paths)
    if existing is not None:
        return existing
    _void_partial(paths, session)
    for name in ("legacy", "normalized_official_context", "spx", "vix"):
        if not paths[name].is_file():
            raise RuntimeError(f"two_clock_source_missing:{session}:{name}")
    with paths["legacy"].open("rb") as handle:
        legacy_rows = pickle.load(handle)
    if not isinstance(legacy_rows, list):
        raise RuntimeError(f"two_clock_legacy_rows_invalid:{session}")
    normalized = pq.read_table(paths["normalized_official_context"])
    config = NeuralDatasetConfig(
        feature_contract=FEATURE_CONTRACT,
        compute_policy_labels=True,
        processed_row_schema_version=TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    )
    materialized_rows = _attach_two_clock_labels(
        legacy_rows,
        normalized,
        config=config,
    )
    assert_additive_parity(
        legacy_rows,
        materialized_rows,
        session=session,
    )
    write_pickle_atomic(paths["output"], materialized_rows)
    receipt = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "verified_additive_two_clock_materialization",
        "session": session,
        "source_hashes": {
            name: sha256_path(paths[name])
            for name in ("legacy", "normalized_official_context", "spx", "vix")
        },
        "output_path": str(paths["output"].relative_to(ROOT)),
        "output_sha256": sha256_path(paths["output"]),
        "row_count": len(materialized_rows),
        "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "legacy_field_parity": True,
        "materialization_method": "frozen_row_additive_label_attachment",
        "producer_source_sha256": sha256_path(Path(__file__)),
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    write_json_atomic(paths["receipt"], receipt)
    return receipt


def _attach_two_clock_labels(
    legacy_rows: list[dict[str, Any]],
    normalized: Any,
    *,
    config: NeuralDatasetConfig,
) -> list[dict[str, Any]]:
    """Attach only signed clock metadata to the frozen governed row grid."""

    options = _prepare_options(normalized, config)
    by_contract_path = {
        str(contract_id): _contract_quote_path(
            group,
            enforce_unique_path=True,
        )
        for contract_id, group in options.groupby("contract_id")
    }
    policies = config.label_policies
    materialized: list[dict[str, Any]] = []
    for row_index, legacy in enumerate(legacy_rows):
        decision_time = pd.Timestamp(legacy["decision_time"])
        if decision_time.tzinfo is None:
            decision_time = decision_time.tz_localize("UTC")
        else:
            decision_time = decision_time.tz_convert("UTC")
        net = np.asarray(legacy["labels_net_pnl"], dtype=float)
        mid = np.asarray(legacy["labels_mid_pnl"], dtype=float)
        if net.shape != mid.shape or net.shape[2] != len(policies):
            raise RuntimeError(
                f"two_clock_legacy_policy_axis_mismatch:{row_index}"
            )
        shape = net.shape
        realized = np.full(shape, INT64_MISSING, dtype=np.int64)
        source = np.full(shape, INT64_MISSING, dtype=np.int64)
        age = np.full(shape, np.nan, dtype=np.float64)
        reason = np.full(
            shape,
            int(ExitReason.INVALID),
            dtype=np.uint8,
        )
        exit_bid = np.full(shape, np.nan, dtype=np.float64)
        deadline = np.full(shape, INT64_MISSING, dtype=np.int64)
        invalid = np.full(
            shape,
            int(InvalidReason.AXIS_OR_POLICY_ALIGNMENT_FAILURE),
            dtype=np.uint8,
        )
        rebuilt_net = np.full(shape, np.nan, dtype=np.float64)
        rebuilt_mid = np.full(shape, np.nan, dtype=np.float64)
        mask = np.asarray(legacy["candidate_mask"], dtype=bool)
        contract_ids = np.asarray(legacy["contract_ids"], dtype=object)
        metadata = legacy.get("contract_quote_metadata") or {}
        for strike_index, right_index in np.argwhere(mask):
            contract_id = str(contract_ids[strike_index, right_index])
            quote_path = by_contract_path.get(contract_id)
            item = metadata.get(contract_id) or {}
            if quote_path is None:
                raise RuntimeError(
                    f"two_clock_frozen_contract_path_missing:"
                    f"{row_index}:{contract_id}"
                )
            try:
                entry_ask = float(item["ask"])
                entry_mid = float(item["mid"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"two_clock_frozen_entry_quote_missing:"
                    f"{row_index}:{contract_id}"
                ) from exc
            for policy_index, policy in enumerate(policies):
                label = label_for_policy_two_clock_from_path(
                    quote_path,
                    decision_time=decision_time,
                    entry_ask=entry_ask,
                    entry_mid=entry_mid,
                    policy=policy,
                    config=config,
                )
                index = (strike_index, right_index, policy_index)
                rebuilt_net[index] = label.net_pnl
                rebuilt_mid[index] = label.mid_pnl
                realized[index] = label.realized_exit_time_ns
                source[index] = label.source_exit_quote_time_ns
                age[index] = label.exit_quote_age_ms
                reason[index] = label.exit_reason_code
                exit_bid[index] = label.executable_exit_bid
                deadline[index] = label.policy_deadline_ns
                invalid[index] = label.invalid_reason_code
        if not np.array_equal(net, rebuilt_net, equal_nan=True):
            raise RuntimeError(
                f"two_clock_rebuilt_net_label_mismatch:{row_index}"
            )
        if not np.array_equal(mid, rebuilt_mid, equal_nan=True):
            raise RuntimeError(
                f"two_clock_rebuilt_mid_label_mismatch:{row_index}"
            )
        row = dict(legacy)
        row.update(
            {
                "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
                "label_realized_exit_time_ns": realized,
                "label_source_exit_quote_time_ns": source,
                "label_exit_quote_age_ms": age,
                "label_exit_reason_code": reason,
                "label_executable_exit_bid": exit_bid,
                "label_policy_deadline_ns": deadline,
                "label_policy_index": np.arange(
                    len(policies),
                    dtype=np.uint8,
                ),
                "label_invalid_reason_code": invalid,
            }
        )
        materialized.append(row)
    return materialized


def materialize_scope(
    *,
    max_workers: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[TrainingScope, dict[str, Any]]:
    scope = load_training_scope()
    entries = _manifest_entries()
    expected_sessions = [session for session, _path in scope.sessions]
    if len(expected_sessions) != 271 or len(set(expected_sessions)) != 271:
        raise RuntimeError("campaign_scope_session_grid_mismatch")
    jobs = [
        (session, str(path.resolve()), entries[session])
        for session, path in scope.sessions
    ]
    receipts: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_materialize_one, *job): job[0] for job in jobs
        }
        for future in as_completed(futures):
            session = futures[future]
            receipts[session] = future.result()
            completed = len(receipts)
            write_json_atomic(
                PROGRESS_PATH,
                {
                    "schema_version": "Protocol101FT1DTwoClockProgressV1",
                    "status": "materializing" if completed < 271 else "complete",
                    "completed_sessions": completed,
                    "expected_sessions": 271,
                    "last_completed_session": session,
                },
            )
            if progress_callback is not None:
                progress_callback(completed, 271, session)
    ordered_receipts = [receipts[session] for session in expected_sessions]
    producer_variants = [
        {
            "materialization_method": method,
            "producer_source_sha256": producer,
            "session_count": sum(
                1
                for item in ordered_receipts
                if item["materialization_method"] == method
                and item["producer_source_sha256"] == producer
            ),
        }
        for method, producer in sorted(
            {
                (
                    str(item["materialization_method"]),
                    str(item["producer_source_sha256"]),
                )
                for item in ordered_receipts
            }
        )
    ]
    global_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_verified_additive_two_clock_materialization",
        "campaign_namespace": (
            "protocol101_full_trader_stage1_entry_fresh_attempt001"
        ),
        "session_count": len(ordered_receipts),
        "sessions": ordered_receipts,
        "fold_governance_sha256": scope.fold_governance_hash,
        "acceptance_registry_sha256": scope.acceptance_registry_hash,
        "governed_manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "governed_manifest_sha256": sha256_path(MANIFEST_PATH),
        "materializer_source_sha256": sha256_path(Path(__file__)),
        "producer_variants": producer_variants,
        "side_effects": {
            "protected_holdout_read": False,
            "paid_data_downloaded": False,
            "broker_endpoint_called": False,
            "seed_45_or_G9_executed": False,
        },
    }
    global_receipt["receipt_sha256"] = stable_hash(global_receipt)
    write_json_atomic(GLOBAL_RECEIPT, global_receipt)
    materialized_paths = {
        item["session"]: ROOT / str(item["output_path"])
        for item in ordered_receipts
    }
    return (
        replace(
            scope,
            sessions=[
                (session, materialized_paths[session])
                for session in expected_sessions
            ],
        ),
        global_receipt,
    )
