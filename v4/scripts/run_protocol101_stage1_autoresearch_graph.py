"""Protocol101 Stage-1 graph rooted in one strict owner receipt chain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from v4.model.protocol101_stage1_controller_journal import (
    ControllerJournalError,
    JOURNAL_NODES,
    JOURNAL_SCHEMA,
    validate_journal,
)

ROOT = Path(__file__).resolve().parents[2]
GRAPH_SCHEMA = "Protocol101FT1CGraphStateV3"
GRAPH_CONTRACT_SCHEMA = "Protocol101FT1CGraphContractV3"
AUTHORIZATION_SCHEMA = "Protocol101Fresh420UnitOwnerExecutionAuthorizationV2"
RECEIPT_SCHEMA = "Protocol101FT1CGraphReceiptV2"
RECEIPT_CHAIN_SCHEMA = "Protocol101FT1CReceiptChainV2"
CAMPAIGN_NAMESPACE = "protocol101_full_trader_stage1_entry_fresh_attempt001"
NO_OWNER_ROUTE = (
    "full_trader_stage1_machinery_ready_owner_execution_"
    "authorization_required"
)
RUN_PENDING_ROUTE = "owner_authorized_fresh_420_unit_RUN_pending_external_execution"
INVALID_CHAIN_ROUTE = "full_trader_stage1_receipt_chain_invalid"
STOP_ROUTE = "full_trader_stage1_selection_routed_STOP_before_G9"

OWNER_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_namespace",
        "campaign_execution_id",
        "authorized",
        "routing_decision",
        "owner_signature",
        "owner_decision_date",
        "goal_sha256",
        "preregistration_sha256",
        "contract_bundle_sha256",
        "seed_45_or_G9_authorized",
    }
)
RECEIPT_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "node",
        "routing_decision",
        "campaign_namespace",
        "campaign_execution_id",
        "owner_authorization_sha256",
        "parent_receipt_sha256",
        "artifact_sha256",
        "receipt_sha256",
    }
)
CHAIN_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_namespace",
        "campaign_execution_id",
        "owner_authorization_sha256",
        "receipts",
        "chain_sha256",
    }
)

NODE_DEFINITIONS = (
    {
        "node": "OWNER_EXECUTION_AUTHORIZATION",
        "requires": [],
        "receipt_route": "owner_authorized_fresh_420_unit_campaign_execution",
    },
    {
        "node": "FRESH_420_UNIT_RUN",
        "requires": ["OWNER_EXECUTION_AUTHORIZATION"],
        "receipt_route": "fresh_420_unit_campaign_run_complete",
    },
    {
        "node": "EXECUTION_PROVENANCE_AUTHORITY",
        "requires": ["FRESH_420_UNIT_RUN"],
        "receipt_route": "fresh_execution_provenance_authority_frozen",
    },
    {
        "node": "REAL_V5_REFERENCES_D1_D5_D6",
        "requires": ["EXECUTION_PROVENANCE_AUTHORITY"],
        "receipt_route": "fresh_v5_reference_controls_complete",
    },
    {
        "node": "CONTROL_AUTHORITY",
        "requires": ["REAL_V5_REFERENCES_D1_D5_D6"],
        "receipt_route": "fresh_control_authority_frozen",
    },
    {
        "node": "FROZEN_20000_REPLICATE_MAXT",
        "requires": ["CONTROL_AUTHORITY"],
        "receipt_route": "fresh_28_row_maxT_complete",
    },
    {
        "node": "G1_G8_AGGREGATION",
        "requires": ["FROZEN_20000_REPLICATE_MAXT"],
        "receipt_route": "fresh_G1_G8_aggregation_complete",
    },
    {
        "node": "INDEPENDENT_AUDIT",
        "requires": ["G1_G8_AGGREGATION"],
        "receipt_route": "fresh_28_row_independent_audit_accepted",
    },
    {
        "node": "MODEL_FREE_SELECTION_ROUTING",
        "requires": ["INDEPENDENT_AUDIT"],
        "receipt_route": "fresh_stage1_selection_routed",
    },
    {
        "node": "STOP",
        "requires": ["MODEL_FREE_SELECTION_ROUTING"],
        "receipt_route": None,
    },
)
RECEIPT_NODE_DEFINITIONS = NODE_DEFINITIONS[1:-1]
RECEIPT_NODES = tuple(item["node"] for item in RECEIPT_NODE_DEFINITIONS)
RECEIPT_ROUTES = {
    item["node"]: item["receipt_route"] for item in RECEIPT_NODE_DEFINITIONS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--owner-authorization", type=Path)
    parser.add_argument("--controller-journal", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--trusted-artifact-bindings", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def owner_authorization_sha256(authorization: Mapping[str, Any]) -> str:
    """Hash the exact external owner payload; the graph never creates it."""
    return stable_hash(dict(authorization))


def seal_graph_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(receipt)
    payload["receipt_sha256"] = None
    payload["receipt_sha256"] = stable_hash(payload)
    return payload


def seal_receipt_chain(chain: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(chain)
    payload["receipts"] = [
        dict(receipt) for receipt in payload.get("receipts", [])
    ]
    payload["chain_sha256"] = None
    payload["chain_sha256"] = stable_hash(payload)
    return payload


def build_receipt_chain(
    *,
    owner_authorization: Mapping[str, Any],
    artifact_sha256_by_node: Mapping[str, str],
    prefix_length: int,
) -> dict[str, Any]:
    """Build a strict synthetic/external chain from already-frozen artifacts."""
    authorized, blockers = _authorization_valid(owner_authorization)
    if not authorized:
        raise ValueError("invalid owner authorization: " + ",".join(blockers))
    if not isinstance(prefix_length, int) or not 0 <= prefix_length <= len(
        RECEIPT_NODES
    ):
        raise ValueError("prefix_length out of range")
    expected_nodes = RECEIPT_NODES[:prefix_length]
    if set(artifact_sha256_by_node) != set(expected_nodes):
        raise ValueError("artifact bindings must equal the completed prefix")
    owner_hash = owner_authorization_sha256(owner_authorization)
    execution_id = owner_authorization["campaign_execution_id"]
    parent_hash = owner_hash
    receipts: list[dict[str, Any]] = []
    for node in expected_nodes:
        artifact_hash = artifact_sha256_by_node[node]
        if not _is_sha256(artifact_hash):
            raise ValueError(f"invalid artifact hash for {node}")
        receipt = seal_graph_receipt(
            {
                "schema_version": RECEIPT_SCHEMA,
                "node": node,
                "routing_decision": RECEIPT_ROUTES[node],
                "campaign_namespace": CAMPAIGN_NAMESPACE,
                "campaign_execution_id": execution_id,
                "owner_authorization_sha256": owner_hash,
                "parent_receipt_sha256": parent_hash,
                "artifact_sha256": artifact_hash,
                "receipt_sha256": None,
            }
        )
        receipts.append(receipt)
        parent_hash = receipt["receipt_sha256"]
    return seal_receipt_chain(
        {
            "schema_version": RECEIPT_CHAIN_SCHEMA,
            "campaign_namespace": CAMPAIGN_NAMESPACE,
            "campaign_execution_id": execution_id,
            "owner_authorization_sha256": owner_hash,
            "receipts": receipts,
            "chain_sha256": None,
        }
    )


def graph_nodes() -> list[dict[str, Any]]:
    return [dict(node) for node in NODE_DEFINITIONS]


def graph_definition() -> dict[str, Any]:
    return {
        "schema_version": GRAPH_CONTRACT_SCHEMA,
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "authorization_schema": AUTHORIZATION_SCHEMA,
        "receipt_schema": RECEIPT_SCHEMA,
        "receipt_chain_schema": RECEIPT_CHAIN_SCHEMA,
        "owner_allowed_fields": sorted(OWNER_ALLOWED_FIELDS),
        "receipt_allowed_fields": sorted(RECEIPT_ALLOWED_FIELDS),
        "chain_allowed_fields": sorted(CHAIN_ALLOWED_FIELDS),
        "nodes": graph_nodes(),
        "canonical_receipt_nodes": list(RECEIPT_NODES),
        "controller_journal_schema": JOURNAL_SCHEMA,
        "controller_journal_nodes": list(JOURNAL_NODES),
        "controller_journal_required": True,
        "autonomous_execution": False,
        "permission_escalation": False,
        "owner_authorization_created_by_graph": False,
        "trusted_artifact_bindings_are_authorization": False,
        "terminal_boundary": "STOP_before_seed45_G9_holdout_learned_exits_transfer_or_paper",
        "forbidden_downstream_nodes": [
            "G9_seed45",
            "protected_holdout",
            "learned_exits",
            "transfer",
            "paper",
        ],
        "no_owner_route": NO_OWNER_ROUTE,
        "invalid_chain_route": INVALID_CHAIN_ROUTE,
    }


def _authorization_valid(
    authorization: Mapping[str, Any] | None,
) -> tuple[bool, list[str]]:
    if authorization is None:
        return False, ["owner_execution_authorization_missing"]
    if not isinstance(authorization, Mapping):
        return False, ["owner_authorization_not_mapping"]
    blockers: list[str] = []
    if set(authorization) != OWNER_ALLOWED_FIELDS:
        blockers.append("owner_authorization_fields_not_exact")
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        blockers.append("owner_authorization_schema_mismatch")
    if authorization.get("campaign_namespace") != CAMPAIGN_NAMESPACE:
        blockers.append("owner_authorization_campaign_mismatch")
    if not _strict_nonempty_string(
        authorization.get("campaign_execution_id")
    ):
        blockers.append("owner_campaign_execution_id_invalid")
    if authorization.get("authorized") is not True:
        blockers.append("owner_authorization_not_true")
    if (
        authorization.get("routing_decision")
        != "owner_authorized_fresh_420_unit_campaign_execution"
    ):
        blockers.append("owner_authorization_route_mismatch")
    if not _strict_nonempty_string(authorization.get("owner_signature")):
        blockers.append("owner_signature_invalid")
    if not _strict_nonempty_string(authorization.get("owner_decision_date")):
        blockers.append("owner_decision_date_invalid")
    for field in (
        "goal_sha256",
        "preregistration_sha256",
        "contract_bundle_sha256",
    ):
        if not _is_sha256(authorization.get(field)):
            blockers.append(f"owner_{field}_invalid")
    if authorization.get("seed_45_or_G9_authorized") is not False:
        blockers.append("owner_authorization_improperly_includes_G9")
    return not blockers, blockers


def _receipt_chain_blockers(
    *,
    chain: Mapping[str, Any] | None,
    authorization: Mapping[str, Any],
    trusted_artifact_sha256_by_node: Mapping[str, str] | None,
) -> tuple[list[str], list[str], dict[str, str], str | None]:
    if chain is None:
        return [], [], {}, None
    blockers: list[str] = []
    if not isinstance(chain, Mapping):
        return ["receipt_chain_not_mapping"], [], {}, None
    if set(chain) != CHAIN_ALLOWED_FIELDS:
        blockers.append("receipt_chain_fields_not_exact")
    if chain.get("schema_version") != RECEIPT_CHAIN_SCHEMA:
        blockers.append("receipt_chain_schema_mismatch")
    owner_hash = owner_authorization_sha256(authorization)
    execution_id = authorization["campaign_execution_id"]
    if chain.get("campaign_namespace") != CAMPAIGN_NAMESPACE:
        blockers.append("receipt_chain_campaign_mismatch")
    if chain.get("campaign_execution_id") != execution_id:
        blockers.append("receipt_chain_execution_id_mismatch")
    if chain.get("owner_authorization_sha256") != owner_hash:
        blockers.append("receipt_chain_owner_hash_mismatch")
    receipts = chain.get("receipts")
    if not isinstance(receipts, list):
        return blockers + ["receipt_chain_receipts_not_list"], [], {}, None
    chain_hash = chain.get("chain_sha256")
    if not _is_sha256(chain_hash):
        blockers.append("receipt_chain_hash_invalid")
    else:
        payload = dict(chain)
        payload["receipts"] = [
            dict(receipt) if isinstance(receipt, Mapping) else receipt
            for receipt in receipts
        ]
        payload["chain_sha256"] = None
        if stable_hash(payload) != chain_hash:
            blockers.append("receipt_chain_hash_mismatch")
    if len(receipts) > len(RECEIPT_NODES):
        blockers.append("receipt_chain_too_long")
    prefix_nodes: list[str] = []
    accepted_bindings: dict[str, str] = {}
    parent_hash = owner_hash
    for index, receipt in enumerate(receipts):
        if index >= len(RECEIPT_NODES):
            blockers.append(f"receipt_{index}_unknown_trailing")
            continue
        expected_node = RECEIPT_NODES[index]
        prefix_nodes.append(expected_node)
        if not isinstance(receipt, Mapping):
            blockers.append(f"receipt_{index}_not_mapping")
            continue
        if set(receipt) != RECEIPT_ALLOWED_FIELDS:
            blockers.append(f"receipt_{index}_fields_not_exact")
        if receipt.get("schema_version") != RECEIPT_SCHEMA:
            blockers.append(f"receipt_{index}_schema_mismatch")
        if receipt.get("node") != expected_node:
            blockers.append(f"receipt_{index}_node_or_order_mismatch")
        if receipt.get("routing_decision") != RECEIPT_ROUTES[expected_node]:
            blockers.append(f"receipt_{index}_route_mismatch")
        if receipt.get("campaign_namespace") != CAMPAIGN_NAMESPACE:
            blockers.append(f"receipt_{index}_campaign_mismatch")
        if receipt.get("campaign_execution_id") != execution_id:
            blockers.append(f"receipt_{index}_execution_id_mismatch")
        if receipt.get("owner_authorization_sha256") != owner_hash:
            blockers.append(f"receipt_{index}_owner_hash_mismatch")
        if receipt.get("parent_receipt_sha256") != parent_hash:
            blockers.append(f"receipt_{index}_parent_hash_mismatch")
        artifact_hash = receipt.get("artifact_sha256")
        if not _is_sha256(artifact_hash):
            blockers.append(f"receipt_{index}_artifact_hash_invalid")
        else:
            accepted_bindings[expected_node] = artifact_hash
        receipt_hash = receipt.get("receipt_sha256")
        if not _is_sha256(receipt_hash):
            blockers.append(f"receipt_{index}_hash_invalid")
        else:
            payload = dict(receipt)
            payload["receipt_sha256"] = None
            if stable_hash(payload) != receipt_hash:
                blockers.append(f"receipt_{index}_hash_mismatch")
            parent_hash = receipt_hash
    expected_binding_keys = set(RECEIPT_NODES[: len(receipts)])
    if receipts:
        if not isinstance(trusted_artifact_sha256_by_node, Mapping):
            blockers.append("trusted_artifact_bindings_missing")
        elif set(trusted_artifact_sha256_by_node) != expected_binding_keys:
            blockers.append("trusted_artifact_binding_fields_not_exact")
        else:
            for node in RECEIPT_NODES[: len(receipts)]:
                expected_hash = trusted_artifact_sha256_by_node.get(node)
                if not _is_sha256(expected_hash):
                    blockers.append(f"trusted_artifact_binding_invalid:{node}")
                elif accepted_bindings.get(node) != expected_hash:
                    blockers.append(f"trusted_artifact_binding_mismatch:{node}")
    elif trusted_artifact_sha256_by_node not in (None, {}):
        blockers.append("trusted_artifact_bindings_before_RUN")
    return blockers, prefix_nodes, accepted_bindings, chain_hash


def _node_states(
    *,
    authorized: bool,
    completed_prefix: Sequence[str],
    next_node: str,
    invalid: bool,
) -> list[dict[str, Any]]:
    states = [
        {"node": node["node"], "status": "BLOCKED", "receipt_route": None}
        for node in NODE_DEFINITIONS
    ]
    states[0]["status"] = "COMPLETE" if authorized else "AWAITING_OWNER"
    completed = set(completed_prefix)
    for index, definition in enumerate(NODE_DEFINITIONS[1:-1], start=1):
        node = definition["node"]
        if node in completed:
            states[index]["status"] = "COMPLETE"
            states[index]["receipt_route"] = definition["receipt_route"]
        elif node == next_node and not invalid:
            states[index]["status"] = "AWAITING_EXTERNAL_RECEIPT"
    if next_node == "STOP" and not invalid:
        states[-1]["status"] = "COMPLETE"
    return states


def _finalize_state(payload: dict[str, Any]) -> dict[str, Any]:
    payload["graph_sha256"] = None
    payload["graph_sha256"] = stable_hash(payload)
    return payload


def _evaluate_legacy_receipt_graph(
    *,
    owner_authorization: Mapping[str, Any] | None = None,
    receipts: Mapping[str, Any] | None = None,
    trusted_artifact_sha256_by_node: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    authorized, authorization_blockers = _authorization_valid(
        owner_authorization
    )
    if owner_authorization is None:
        return _finalize_state(
            {
                "schema_version": GRAPH_SCHEMA,
                "routing_decision": NO_OWNER_ROUTE,
                "next_node": "OWNER_EXECUTION_AUTHORIZATION",
                "nodes": _node_states(
                    authorized=False,
                    completed_prefix=[],
                    next_node="OWNER_EXECUTION_AUTHORIZATION",
                    invalid=False,
                ),
                "blockers": authorization_blockers,
                "owner_authorization_sha256": None,
                "campaign_execution_id": None,
                "completed_prefix": [],
                "accepted_artifact_bindings": {},
                "receipt_chain_sha256": None,
                "RUN_executed": False,
                "commands_executed": [],
                "owner_authorization_created": False,
                "G9_executed": False,
                "stopped_before_RUN": True,
            }
        )
    if not authorized:
        return _finalize_state(
            {
                "schema_version": GRAPH_SCHEMA,
                "routing_decision": INVALID_CHAIN_ROUTE,
                "next_node": "OWNER_EXECUTION_AUTHORIZATION",
                "nodes": _node_states(
                    authorized=False,
                    completed_prefix=[],
                    next_node="OWNER_EXECUTION_AUTHORIZATION",
                    invalid=True,
                ),
                "blockers": authorization_blockers,
                "owner_authorization_sha256": None,
                "campaign_execution_id": None,
                "completed_prefix": [],
                "accepted_artifact_bindings": {},
                "receipt_chain_sha256": None,
                "RUN_executed": False,
                "commands_executed": [],
                "owner_authorization_created": False,
                "G9_executed": False,
                "stopped_before_RUN": True,
            }
        )
    assert owner_authorization is not None
    chain_blockers, completed_prefix, accepted_bindings, chain_hash = (
        _receipt_chain_blockers(
            chain=receipts,
            authorization=owner_authorization,
            trusted_artifact_sha256_by_node=trusted_artifact_sha256_by_node,
        )
    )
    owner_hash = owner_authorization_sha256(owner_authorization)
    if chain_blockers:
        return _finalize_state(
            {
                "schema_version": GRAPH_SCHEMA,
                "routing_decision": INVALID_CHAIN_ROUTE,
                "next_node": "INVALID_RECEIPT_CHAIN",
                "nodes": _node_states(
                    authorized=True,
                    completed_prefix=[],
                    next_node="INVALID_RECEIPT_CHAIN",
                    invalid=True,
                ),
                "blockers": chain_blockers,
                "owner_authorization_sha256": owner_hash,
                "campaign_execution_id": owner_authorization[
                    "campaign_execution_id"
                ],
                "completed_prefix": [],
                "accepted_artifact_bindings": {},
                "receipt_chain_sha256": chain_hash,
                "RUN_executed": False,
                "commands_executed": [],
                "owner_authorization_created": False,
                "G9_executed": False,
                "stopped_before_RUN": True,
            }
        )
    if len(completed_prefix) == len(RECEIPT_NODES):
        next_node = "STOP"
        route = STOP_ROUTE
    else:
        next_node = RECEIPT_NODES[len(completed_prefix)]
        route = RUN_PENDING_ROUTE
    return _finalize_state(
        {
            "schema_version": GRAPH_SCHEMA,
            "routing_decision": route,
            "next_node": next_node,
            "nodes": _node_states(
                authorized=True,
                completed_prefix=completed_prefix,
                next_node=next_node,
                invalid=False,
            ),
            "blockers": [],
            "owner_authorization_sha256": owner_hash,
            "campaign_execution_id": owner_authorization[
                "campaign_execution_id"
            ],
            "completed_prefix": completed_prefix,
            "accepted_artifact_bindings": accepted_bindings,
            "receipt_chain_sha256": chain_hash,
            "RUN_executed": False,
            "commands_executed": [],
            "owner_authorization_created": False,
            "G9_executed": False,
            "stopped_before_RUN": not completed_prefix,
        }
    )


def evaluate_graph(
    *,
    owner_authorization: Mapping[str, Any] | None = None,
    controller_journal_path: Path | None = None,
    workspace_root: Path = ROOT,
    receipts: Mapping[str, Any] | None = None,
    trusted_artifact_sha256_by_node: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read graph state only from the trusted local controller journal."""
    authorized, authorization_blockers = _authorization_valid(
        owner_authorization
    )
    if owner_authorization is None:
        return _finalize_state(
            {
                "schema_version": GRAPH_SCHEMA,
                "routing_decision": NO_OWNER_ROUTE,
                "next_node": "OWNER_EXECUTION_AUTHORIZATION",
                "nodes": _node_states(
                    authorized=False,
                    completed_prefix=[],
                    next_node="OWNER_EXECUTION_AUTHORIZATION",
                    invalid=False,
                ),
                "blockers": authorization_blockers,
                "owner_authorization_sha256": None,
                "campaign_execution_id": None,
                "completed_prefix": [],
                "accepted_artifact_bindings": {},
                "receipt_chain_sha256": None,
                "controller_journal_sha256": None,
                "RUN_executed": False,
                "commands_executed": [],
                "owner_authorization_created": False,
                "G9_executed": False,
                "stopped_before_RUN": True,
            }
        )
    if not authorized:
        return _finalize_state(
            {
                "schema_version": GRAPH_SCHEMA,
                "routing_decision": INVALID_CHAIN_ROUTE,
                "next_node": "OWNER_EXECUTION_AUTHORIZATION",
                "nodes": _node_states(
                    authorized=False,
                    completed_prefix=[],
                    next_node="OWNER_EXECUTION_AUTHORIZATION",
                    invalid=True,
                ),
                "blockers": authorization_blockers,
                "owner_authorization_sha256": None,
                "campaign_execution_id": None,
                "completed_prefix": [],
                "accepted_artifact_bindings": {},
                "receipt_chain_sha256": None,
                "controller_journal_sha256": None,
                "RUN_executed": False,
                "commands_executed": [],
                "owner_authorization_created": False,
                "G9_executed": False,
                "stopped_before_RUN": True,
            }
        )
    assert owner_authorization is not None
    owner_hash = owner_authorization_sha256(owner_authorization)
    blockers: list[str] = []
    if receipts is not None:
        blockers.append("caller_supplied_receipt_chain_not_authoritative")
    if trusted_artifact_sha256_by_node is not None:
        blockers.append("caller_supplied_trusted_artifact_map_not_authoritative")
    if controller_journal_path is None:
        blockers.append("controller_journal_required")
    journal_state: dict[str, Any] | None = None
    if not blockers:
        try:
            journal_state = validate_journal(
                controller_journal_path,
                workspace_root=workspace_root,
                expected_campaign_namespace=CAMPAIGN_NAMESPACE,
                expected_campaign_execution_id=owner_authorization[
                    "campaign_execution_id"
                ],
                expected_owner_authorization_sha256=owner_hash,
            )
        except ControllerJournalError as exc:
            blockers.append(f"controller_journal_invalid:{exc}")
    if blockers or journal_state is None:
        return _finalize_state(
            {
                "schema_version": GRAPH_SCHEMA,
                "routing_decision": INVALID_CHAIN_ROUTE,
                "next_node": "INVALID_CONTROLLER_JOURNAL",
                "nodes": _node_states(
                    authorized=True,
                    completed_prefix=[],
                    next_node="INVALID_CONTROLLER_JOURNAL",
                    invalid=True,
                ),
                "blockers": blockers,
                "owner_authorization_sha256": owner_hash,
                "campaign_execution_id": owner_authorization[
                    "campaign_execution_id"
                ],
                "completed_prefix": [],
                "accepted_artifact_bindings": {},
                "receipt_chain_sha256": None,
                "controller_journal_sha256": None,
                "RUN_executed": False,
                "commands_executed": [],
                "owner_authorization_created": False,
                "G9_executed": False,
                "stopped_before_RUN": True,
            }
        )
    completed_prefix = journal_state["completed_prefix"]
    accepted_bindings = {
        node: journal_state["artifact_bindings"][node]["artifact_sha256"]
        for node in completed_prefix
    }
    next_node = journal_state["next_node"]
    route = STOP_ROUTE if next_node == "STOP" else RUN_PENDING_ROUTE
    return _finalize_state(
        {
            "schema_version": GRAPH_SCHEMA,
            "routing_decision": route,
            "next_node": next_node,
            "nodes": _node_states(
                authorized=True,
                completed_prefix=completed_prefix,
                next_node=next_node,
                invalid=False,
            ),
            "blockers": [],
            "owner_authorization_sha256": owner_hash,
            "campaign_execution_id": owner_authorization[
                "campaign_execution_id"
            ],
            "completed_prefix": completed_prefix,
            "accepted_artifact_bindings": accepted_bindings,
            "receipt_chain_sha256": None,
            "controller_journal_sha256": journal_state[
                "journal_head_sha256"
            ],
            "RUN_executed": False,
            "commands_executed": [],
            "owner_authorization_created": False,
            "G9_executed": False,
            "stopped_before_RUN": not completed_prefix,
        }
    )


def initial_state(definition: Mapping[str, Any] | None = None) -> dict[str, Any]:
    del definition
    return evaluate_graph()


def run_graph(
    *,
    definition: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    state_dir: Path | None = None,
    max_attempts: int = 1,
) -> dict[str, Any]:
    """Legacy-compatible, deliberately non-executing graph entrypoint."""
    del definition, state, max_attempts
    result = evaluate_graph()
    if state_dir is not None:
        write_json_atomic(state_dir / "graph_state.json", result)
    return result


def main() -> int:
    args = parse_args()
    authorization = (
        load_json(args.owner_authorization) if args.owner_authorization else None
    )
    receipts = load_json(args.receipts) if args.receipts else None
    bindings = (
        load_json(args.trusted_artifact_bindings)
        if args.trusted_artifact_bindings
        else None
    )
    result = evaluate_graph(
        owner_authorization=authorization,
        controller_journal_path=args.controller_journal,
        workspace_root=args.workspace_root,
        receipts=receipts,
        trusted_artifact_sha256_by_node=bindings,
    )
    write_json_atomic(args.out_dir / "graph_state.json", result)
    write_json_atomic(args.out_dir / "graph_contract.json", graph_definition())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
