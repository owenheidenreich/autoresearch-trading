"""Trusted local append-only journal for a Protocol101 Stage-1 campaign."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


JOURNAL_SCHEMA = "Protocol101Stage1ControllerJournalV1"
RECORD_SCHEMA = "Protocol101Stage1ControllerJournalRecordV1"
GENESIS_NODE = "GENESIS"
GENESIS_ROUTE = "owner_approved_option_a_controller_journal_genesis"
GENESIS_SENTINEL = (
    "0f5b7667ff1ecf404ed18957e4cb60ac7480e0f5bc6b53dfcc6414649969cab4"
)
JOURNAL_NODES = (
    "FRESH_420_UNIT_RUN",
    "EXECUTION_PROVENANCE_AUTHORITY",
    "REAL_V5_REFERENCES_D1_D5_D6",
    "CONTROL_AUTHORITY",
    "FROZEN_20000_REPLICATE_MAXT",
    "G1_G8_AGGREGATION",
    "INDEPENDENT_AUDIT",
    "MODEL_FREE_SELECTION_ROUTING",
)
JOURNAL_ROUTES = {
    "FRESH_420_UNIT_RUN": "fresh_420_unit_campaign_run_complete",
    "EXECUTION_PROVENANCE_AUTHORITY": (
        "fresh_execution_provenance_authority_frozen"
    ),
    "REAL_V5_REFERENCES_D1_D5_D6": "fresh_v5_reference_controls_complete",
    "CONTROL_AUTHORITY": "fresh_control_authority_frozen",
    "FROZEN_20000_REPLICATE_MAXT": "fresh_28_row_maxT_complete",
    "G1_G8_AGGREGATION": "fresh_G1_G8_aggregation_complete",
    "INDEPENDENT_AUDIT": "fresh_28_row_independent_audit_accepted",
    "MODEL_FREE_SELECTION_ROUTING": "fresh_stage1_selection_routed",
}
GENESIS_FIELDS = frozenset(
    {
        "schema_version",
        "journal_schema",
        "record_kind",
        "ordinal",
        "node",
        "routing_decision",
        "campaign_namespace",
        "campaign_execution_id",
        "owner_option_a_decision_path",
        "owner_option_a_decision_sha256",
        "offline_training_authorization_path",
        "offline_training_authorization_sha256",
        "campaign_goal_path",
        "campaign_goal_sha256",
        "campaign_preregistration_path",
        "campaign_preregistration_sha256",
        "signed_contract_bundle_path",
        "signed_contract_bundle_sha256",
        "owner_execution_authorization_sha256",
        "owner_identity",
        "owner_decision_date",
        "previous_record_sha256",
        "timestamp_utc",
        "record_sha256",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "record_kind",
        "ordinal",
        "node",
        "routing_decision",
        "campaign_namespace",
        "campaign_execution_id",
        "owner_genesis_sha256",
        "previous_record_sha256",
        "artifact_path",
        "artifact_sha256",
        "validator_route",
        "validator_receipt_path",
        "validator_receipt_sha256",
        "timestamp_utc",
        "record_sha256",
    }
)


class ControllerJournalError(ValueError):
    """The local controller journal failed closed."""


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _canonical_line(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _timestamp(value: str | None) -> str:
    if value is not None:
        if not _strict_text(value):
            raise ControllerJournalError("timestamp_invalid")
        return value
    return datetime.now(timezone.utc).isoformat()


def _workspace_file(
    workspace_root: Path,
    path: Path,
    *,
    field: str,
) -> tuple[Path, str]:
    root = workspace_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise ControllerJournalError(f"{field}_outside_workspace_or_missing") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ControllerJournalError(f"{field}_not_regular_file")
    return resolved, relative


def _owner_only(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def _seal(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload["record_sha256"] = None
    payload["record_sha256"] = stable_hash(payload)
    return payload


def _read_locked(handle) -> bytes:
    handle.seek(0)
    return handle.read()


def _parse_records(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        raise ControllerJournalError("journal_empty")
    if not raw.endswith(b"\n"):
        raise ControllerJournalError("journal_partial_final_line")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerJournalError(
                f"journal_line_malformed:{index}"
            ) from exc
        if not isinstance(record, dict):
            raise ControllerJournalError(f"journal_line_not_object:{index}")
        if _canonical_line(record) != line:
            raise ControllerJournalError(f"journal_line_not_canonical:{index}")
        records.append(record)
    return records


def _validate_record_hash(record: Mapping[str, Any], index: int) -> None:
    record_hash = record.get("record_sha256")
    if not _is_sha256(record_hash):
        raise ControllerJournalError(f"record_hash_invalid:{index}")
    payload = dict(record)
    payload["record_sha256"] = None
    if stable_hash(payload) != record_hash:
        raise ControllerJournalError(f"record_hash_mismatch:{index}")


def _validate_relative_binding(
    workspace_root: Path,
    relative_path: Any,
    expected_hash: Any,
    *,
    field: str,
) -> None:
    if not _strict_text(relative_path) or Path(str(relative_path)).is_absolute():
        raise ControllerJournalError(f"{field}_path_invalid")
    target, normalized = _workspace_file(
        workspace_root,
        Path(str(relative_path)),
        field=field,
    )
    if normalized != relative_path:
        raise ControllerJournalError(f"{field}_path_not_normalized")
    if not _is_sha256(expected_hash):
        raise ControllerJournalError(f"{field}_hash_invalid")
    if file_sha256(target) != expected_hash:
        raise ControllerJournalError(f"{field}_hash_mismatch")


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    journal_path: Path,
    workspace_root: Path,
    expected_campaign_namespace: str | None,
    expected_campaign_execution_id: str | None,
    expected_owner_authorization_sha256: str | None,
) -> dict[str, Any]:
    if not records:
        raise ControllerJournalError("journal_empty")
    genesis = records[0]
    if set(genesis) != GENESIS_FIELDS:
        raise ControllerJournalError("genesis_fields_not_exact")
    if genesis.get("schema_version") != RECORD_SCHEMA:
        raise ControllerJournalError("genesis_record_schema_mismatch")
    if genesis.get("journal_schema") != JOURNAL_SCHEMA:
        raise ControllerJournalError("journal_schema_mismatch")
    if genesis.get("record_kind") != "genesis":
        raise ControllerJournalError("genesis_kind_mismatch")
    if genesis.get("ordinal") != 0 or genesis.get("node") != GENESIS_NODE:
        raise ControllerJournalError("genesis_position_mismatch")
    if genesis.get("routing_decision") != GENESIS_ROUTE:
        raise ControllerJournalError("genesis_route_mismatch")
    if genesis.get("previous_record_sha256") != GENESIS_SENTINEL:
        raise ControllerJournalError("genesis_sentinel_mismatch")
    for field in (
        "campaign_namespace",
        "campaign_execution_id",
        "owner_identity",
        "owner_decision_date",
        "timestamp_utc",
    ):
        if not _strict_text(genesis.get(field)):
            raise ControllerJournalError(f"genesis_{field}_invalid")
    if (
        expected_campaign_namespace is not None
        and genesis["campaign_namespace"] != expected_campaign_namespace
    ):
        raise ControllerJournalError("journal_campaign_namespace_mismatch")
    if (
        expected_campaign_execution_id is not None
        and genesis["campaign_execution_id"] != expected_campaign_execution_id
    ):
        raise ControllerJournalError("journal_campaign_execution_id_mismatch")
    if (
        expected_owner_authorization_sha256 is not None
        and genesis["owner_execution_authorization_sha256"]
        != expected_owner_authorization_sha256
    ):
        raise ControllerJournalError("journal_owner_authorization_mismatch")
    for prefix in (
        "owner_option_a_decision",
        "offline_training_authorization",
        "campaign_goal",
        "campaign_preregistration",
        "signed_contract_bundle",
    ):
        _validate_relative_binding(
            workspace_root,
            genesis.get(f"{prefix}_path"),
            genesis.get(f"{prefix}_sha256"),
            field=prefix,
        )
    if not _is_sha256(genesis.get("owner_execution_authorization_sha256")):
        raise ControllerJournalError("owner_execution_authorization_hash_invalid")
    _validate_record_hash(genesis, 0)

    genesis_hash = str(genesis["record_sha256"])
    previous_hash = genesis_hash
    bindings: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records[1:], start=1):
        if index > len(JOURNAL_NODES):
            raise ControllerJournalError("journal_has_extra_checkpoint")
        expected_node = JOURNAL_NODES[index - 1]
        if set(record) != CHECKPOINT_FIELDS:
            raise ControllerJournalError(f"checkpoint_fields_not_exact:{index}")
        if record.get("schema_version") != RECORD_SCHEMA:
            raise ControllerJournalError(f"checkpoint_schema_mismatch:{index}")
        if record.get("record_kind") != "checkpoint":
            raise ControllerJournalError(f"checkpoint_kind_mismatch:{index}")
        if record.get("ordinal") != index:
            raise ControllerJournalError(f"checkpoint_ordinal_mismatch:{index}")
        if record.get("node") != expected_node:
            raise ControllerJournalError(f"checkpoint_node_or_order_mismatch:{index}")
        if record.get("routing_decision") != JOURNAL_ROUTES[expected_node]:
            raise ControllerJournalError(f"checkpoint_route_mismatch:{index}")
        if record.get("campaign_namespace") != genesis["campaign_namespace"]:
            raise ControllerJournalError(f"checkpoint_campaign_mismatch:{index}")
        if record.get("campaign_execution_id") != genesis["campaign_execution_id"]:
            raise ControllerJournalError(f"checkpoint_execution_mismatch:{index}")
        if record.get("owner_genesis_sha256") != genesis_hash:
            raise ControllerJournalError(f"checkpoint_genesis_mismatch:{index}")
        if record.get("previous_record_sha256") != previous_hash:
            raise ControllerJournalError(f"checkpoint_parent_mismatch:{index}")
        if not _strict_text(record.get("validator_route")):
            raise ControllerJournalError(f"validator_route_invalid:{index}")
        if not _strict_text(record.get("timestamp_utc")):
            raise ControllerJournalError(f"checkpoint_timestamp_invalid:{index}")
        _validate_relative_binding(
            workspace_root,
            record.get("artifact_path"),
            record.get("artifact_sha256"),
            field=f"artifact:{expected_node}",
        )
        _validate_relative_binding(
            workspace_root,
            record.get("validator_receipt_path"),
            record.get("validator_receipt_sha256"),
            field=f"validator_receipt:{expected_node}",
        )
        _validate_record_hash(record, index)
        previous_hash = str(record["record_sha256"])
        bindings[expected_node] = {
            "artifact_path": record["artifact_path"],
            "artifact_sha256": record["artifact_sha256"],
            "validator_route": record["validator_route"],
            "validator_receipt_path": record["validator_receipt_path"],
            "validator_receipt_sha256": record[
                "validator_receipt_sha256"
            ],
            "record_sha256": record["record_sha256"],
        }
    return {
        "schema_version": JOURNAL_SCHEMA,
        "journal_path": journal_path.resolve().as_posix(),
        "campaign_namespace": genesis["campaign_namespace"],
        "campaign_execution_id": genesis["campaign_execution_id"],
        "owner_authorization_sha256": genesis[
            "owner_execution_authorization_sha256"
        ],
        "genesis_sha256": genesis_hash,
        "journal_head_sha256": previous_hash,
        "checkpoint_count": len(records) - 1,
        "completed_prefix": list(JOURNAL_NODES[: len(records) - 1]),
        "next_node": (
            "STOP"
            if len(records) - 1 == len(JOURNAL_NODES)
            else JOURNAL_NODES[len(records) - 1]
        ),
        "artifact_bindings": bindings,
        "records": [dict(record) for record in records],
        "valid": True,
    }


def validate_journal(
    journal_path: Path,
    *,
    workspace_root: Path,
    expected_campaign_namespace: str | None = None,
    expected_campaign_execution_id: str | None = None,
    expected_owner_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    path = journal_path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ControllerJournalError("journal_missing_or_not_regular")
    if not _owner_only(path):
        raise ControllerJournalError("journal_permissions_not_owner_only")
    with path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            records = _parse_records(_read_locked(handle))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return _validate_records(
        records,
        journal_path=path,
        workspace_root=workspace_root,
        expected_campaign_namespace=expected_campaign_namespace,
        expected_campaign_execution_id=expected_campaign_execution_id,
        expected_owner_authorization_sha256=(
            expected_owner_authorization_sha256
        ),
    )


def create_journal(
    journal_path: Path,
    *,
    workspace_root: Path,
    campaign_namespace: str,
    campaign_execution_id: str,
    owner_option_a_decision_path: Path,
    offline_training_authorization_path: Path,
    campaign_goal_path: Path,
    campaign_preregistration_path: Path,
    signed_contract_bundle_path: Path,
    owner_execution_authorization_sha256: str,
    owner_identity: str,
    owner_decision_date: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    for field, value in (
        ("campaign_namespace", campaign_namespace),
        ("campaign_execution_id", campaign_execution_id),
        ("owner_identity", owner_identity),
        ("owner_decision_date", owner_decision_date),
    ):
        if not _strict_text(value):
            raise ControllerJournalError(f"{field}_invalid")
    if not _is_sha256(owner_execution_authorization_sha256):
        raise ControllerJournalError("owner_execution_authorization_hash_invalid")
    paths = {
        "owner_option_a_decision": owner_option_a_decision_path,
        "offline_training_authorization": offline_training_authorization_path,
        "campaign_goal": campaign_goal_path,
        "campaign_preregistration": campaign_preregistration_path,
        "signed_contract_bundle": signed_contract_bundle_path,
    }
    bindings: dict[str, tuple[str, str]] = {}
    for name, supplied in paths.items():
        target, relative = _workspace_file(
            workspace_root,
            supplied,
            field=name,
        )
        bindings[name] = (relative, file_sha256(target))
    genesis = _seal(
        {
            "schema_version": RECORD_SCHEMA,
            "journal_schema": JOURNAL_SCHEMA,
            "record_kind": "genesis",
            "ordinal": 0,
            "node": GENESIS_NODE,
            "routing_decision": GENESIS_ROUTE,
            "campaign_namespace": campaign_namespace,
            "campaign_execution_id": campaign_execution_id,
            "owner_option_a_decision_path": bindings[
                "owner_option_a_decision"
            ][0],
            "owner_option_a_decision_sha256": bindings[
                "owner_option_a_decision"
            ][1],
            "offline_training_authorization_path": bindings[
                "offline_training_authorization"
            ][0],
            "offline_training_authorization_sha256": bindings[
                "offline_training_authorization"
            ][1],
            "campaign_goal_path": bindings["campaign_goal"][0],
            "campaign_goal_sha256": bindings["campaign_goal"][1],
            "campaign_preregistration_path": bindings[
                "campaign_preregistration"
            ][0],
            "campaign_preregistration_sha256": bindings[
                "campaign_preregistration"
            ][1],
            "signed_contract_bundle_path": bindings[
                "signed_contract_bundle"
            ][0],
            "signed_contract_bundle_sha256": bindings[
                "signed_contract_bundle"
            ][1],
            "owner_execution_authorization_sha256": (
                owner_execution_authorization_sha256
            ),
            "owner_identity": owner_identity,
            "owner_decision_date": owner_decision_date,
            "previous_record_sha256": GENESIS_SENTINEL,
            "timestamp_utc": _timestamp(timestamp_utc),
            "record_sha256": None,
        }
    )
    path = journal_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, _canonical_line(genesis))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
    return validate_journal(
        path,
        workspace_root=workspace_root,
        expected_campaign_namespace=campaign_namespace,
        expected_campaign_execution_id=campaign_execution_id,
        expected_owner_authorization_sha256=(
            owner_execution_authorization_sha256
        ),
    )


def append_checkpoint(
    journal_path: Path,
    *,
    workspace_root: Path,
    node: str,
    artifact_path: Path,
    validator_route: str,
    validator_receipt_path: Path,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    if node not in JOURNAL_NODES:
        raise ControllerJournalError("checkpoint_node_forbidden_or_unknown")
    if not _strict_text(validator_route):
        raise ControllerJournalError("validator_route_invalid")
    artifact, artifact_relative = _workspace_file(
        workspace_root,
        artifact_path,
        field="artifact",
    )
    validator, validator_relative = _workspace_file(
        workspace_root,
        validator_receipt_path,
        field="validator_receipt",
    )
    path = journal_path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ControllerJournalError("journal_missing_or_not_regular")
    with path.open("r+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            records = _parse_records(_read_locked(handle))
            state = _validate_records(
                records,
                journal_path=path,
                workspace_root=workspace_root,
                expected_campaign_namespace=None,
                expected_campaign_execution_id=None,
                expected_owner_authorization_sha256=None,
            )
            if state["next_node"] == "STOP":
                raise ControllerJournalError("journal_already_complete")
            if node != state["next_node"]:
                raise ControllerJournalError("checkpoint_duplicate_skipped_or_reordered")
            genesis = records[0]
            record = _seal(
                {
                    "schema_version": RECORD_SCHEMA,
                    "record_kind": "checkpoint",
                    "ordinal": len(records),
                    "node": node,
                    "routing_decision": JOURNAL_ROUTES[node],
                    "campaign_namespace": genesis["campaign_namespace"],
                    "campaign_execution_id": genesis[
                        "campaign_execution_id"
                    ],
                    "owner_genesis_sha256": genesis["record_sha256"],
                    "previous_record_sha256": records[-1]["record_sha256"],
                    "artifact_path": artifact_relative,
                    "artifact_sha256": file_sha256(artifact),
                    "validator_route": validator_route,
                    "validator_receipt_path": validator_relative,
                    "validator_receipt_sha256": file_sha256(validator),
                    "timestamp_utc": _timestamp(timestamp_utc),
                    "record_sha256": None,
                }
            )
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_line(record))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    os.chmod(path, 0o600)
    return validate_journal(path, workspace_root=workspace_root)
