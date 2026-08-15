"""Deterministic controller for the Protocol101 Full Trader research graph.

The controller records externally completed work. It never launches training,
touches protected evidence, contacts IBKR, or changes the scientific contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = (
    ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V1.json"
)
DEFAULT_STATE_DIR = (
    ROOT / "v4/audit/autoresearch/protocol101_full_trader_graph_v1"
)

GRAPH_SCHEMA = "Protocol101FullTraderGraphV1"
STATE_SCHEMA = "Protocol101FullTraderGraphStateV1"
RECEIPT_SCHEMA = "Protocol101FullTraderNodeReceiptV1"
START_AUTHORIZATION_SCHEMA = "Protocol101FullTraderNodeStartAuthorizationV1"
RESET_NODE = "FT-00-GRAPH-RESET"
ENTRY_NODE = "FT-10-FULL-LADDER-MODEL-DESIGN"
PARKED_STATUS = "parked_before_node"
READY_STATUS = "ready_for_external_goal"
TERMINAL_STATUS = "terminal"

REQUIRED_SCOPE_ASSERTIONS = {
    "flat_state_can_wait": True,
    "flat_state_selects_from_full_governed_ladder": True,
    "direction_not_permanently_hard_coded": True,
    "strike_not_permanently_hard_coded": True,
    "entry_plus_exit_required_for_paper": True,
}


class GraphContractError(ValueError):
    """Raised when graph, state, or receipt authority is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "validate",
            "initialize",
            "status",
            "dry-run-routes",
            "authorize-start",
            "advance",
        ),
        required=True,
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GraphContractError(f"expected JSON object: {path}")
    return payload


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def node_map(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["id"]): dict(node)
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping) and "id" in node
    }


def outgoing_edges(
    graph: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        if isinstance(edge, Mapping):
            outgoing[str(edge.get("from"))].append(dict(edge))
    return dict(outgoing)


def reachable_nodes(
    graph: Mapping[str, Any],
    *,
    start: str,
    blocked: Iterable[str] = (),
) -> set[str]:
    blocked_set = set(blocked)
    if start in blocked_set:
        return set()
    outgoing = outgoing_edges(graph)
    reached = {start}
    queue = deque([start])
    while queue:
        source = queue.popleft()
        for edge in outgoing.get(source, []):
            target = str(edge["to"])
            if target in blocked_set or target in reached:
                continue
            reached.add(target)
            queue.append(target)
    return reached


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    _require(
        graph.get("schema_version") == GRAPH_SCHEMA,
        "graph_schema_mismatch",
        errors,
    )
    _require(graph.get("reset_node") == RESET_NODE, "reset_node_mismatch", errors)
    _require(graph.get("entry_node") == ENTRY_NODE, "entry_node_mismatch", errors)

    product = graph.get("product_contract")
    _require(isinstance(product, Mapping), "product_contract_missing", errors)
    if isinstance(product, Mapping):
        flat_actions = set(product.get("flat_state_learned_actions", []))
        open_actions = set(product.get("open_state_learned_actions", []))
        _require("WAIT" in flat_actions, "flat_WAIT_missing", errors)
        _require(
            "BUY_SPECIFIC_ELIGIBLE_CALL_OR_PUT_CONTRACT" in flat_actions,
            "full_ladder_specific_contract_action_missing",
            errors,
        )
        _require(open_actions == {"HOLD", "EXIT"}, "open_actions_not_exact", errors)
        _require(
            product.get("paper_readiness_requires_complete_entry_plus_exit")
            is True,
            "paper_complete_trader_requirement_missing",
            errors,
        )

    raw_nodes = graph.get("nodes")
    _require(isinstance(raw_nodes, list) and bool(raw_nodes), "nodes_missing", errors)
    nodes = node_map(graph)
    _require(
        isinstance(raw_nodes, list) and len(nodes) == len(raw_nodes),
        "node_ids_not_unique_or_invalid",
        errors,
    )
    _require(RESET_NODE in nodes, "reset_node_not_defined", errors)
    _require(ENTRY_NODE in nodes, "entry_node_not_defined", errors)

    edges = graph.get("edges")
    _require(isinstance(edges, list) and bool(edges), "edges_missing", errors)
    seen_routes: set[tuple[str, str]] = set()
    loop_budgets = graph.get("controller_policy", {}).get("loop_budgets", {})
    for index, edge in enumerate(edges if isinstance(edges, list) else []):
        if not isinstance(edge, Mapping):
            errors.append(f"edge_{index}_not_mapping")
            continue
        source = edge.get("from")
        target = edge.get("to")
        outcome = edge.get("outcome")
        _require(source in nodes, f"edge_{index}_source_invalid", errors)
        _require(target in nodes, f"edge_{index}_target_invalid", errors)
        _require(
            isinstance(outcome, str) and bool(outcome),
            f"edge_{index}_outcome_invalid",
            errors,
        )
        route_key = (str(source), str(outcome))
        _require(route_key not in seen_routes, f"duplicate_route_{route_key}", errors)
        seen_routes.add(route_key)
        if "loop_budget" in edge:
            budget_name = edge["loop_budget"]
            _require(
                budget_name in loop_budgets,
                f"edge_{index}_loop_budget_unknown",
                errors,
            )
            _require(
                isinstance(loop_budgets.get(budget_name), int)
                and loop_budgets[budget_name] > 0,
                f"edge_{index}_loop_budget_invalid",
                errors,
            )

    outgoing = outgoing_edges(graph)
    for node_id, node in nodes.items():
        is_terminal = node.get("kind") == "terminal"
        _require(
            (not outgoing.get(node_id)) if is_terminal else bool(outgoing.get(node_id)),
            f"terminal_edge_rule_failed_{node_id}",
            errors,
        )
        if node.get("model_training") is True:
            _require(
                node.get("action_space")
                in {
                    "flat_state_full_governed_ladder",
                    "open_state_HOLD_EXIT",
                },
                f"training_action_space_invalid_{node_id}",
                errors,
            )

    reached = reachable_nodes(graph, start=RESET_NODE)
    _require(set(nodes) == reached, "unreachable_nodes_present", errors)

    for target, required_gates in graph.get(
        "gate_dominance_requirements", {}
    ).items():
        _require(target in nodes, f"gate_target_invalid_{target}", errors)
        for gate in required_gates:
            _require(gate in nodes, f"gate_node_invalid_{gate}", errors)
            without_gate = reachable_nodes(
                graph,
                start=RESET_NODE,
                blocked={gate},
            )
            _require(
                target not in without_gate,
                f"gate_does_not_dominate_{gate}_to_{target}",
                errors,
            )

    review_nodes = [
        node
        for node in nodes.values()
        if node.get("independent_acceptance") is True
    ]
    for node in review_nodes:
        _require(
            node.get("actor") != "executor",
            f"producer_reviews_own_work_{node['id']}",
            errors,
        )

    initial = graph.get("initial_state")
    _require(isinstance(initial, Mapping), "initial_state_missing", errors)
    if isinstance(initial, Mapping):
        _require(
            initial.get("current_node") == ENTRY_NODE,
            "initial_state_not_parked_at_design",
            errors,
        )
        _require(
            initial.get("execution_status") == PARKED_STATUS,
            "initial_state_not_parked",
            errors,
        )
        _require(
            initial.get("owner_start_required") is True,
            "initial_owner_start_not_required",
            errors,
        )
        _require(
            initial.get("completed_nodes") == [RESET_NODE],
            "initial_completed_nodes_invalid",
            errors,
        )

    if errors:
        raise GraphContractError(";".join(errors))
    return {
        "valid": True,
        "graph_id": graph["graph_id"],
        "graph_sha256": stable_hash(graph),
        "product_contract_sha256": stable_hash(graph["product_contract"]),
        "node_count": len(nodes),
        "edge_count": len(graph["edges"]),
        "reachable_node_count": len(reached),
        "terminal_nodes": sorted(
            node_id
            for node_id, node in nodes.items()
            if node.get("kind") == "terminal"
        ),
    }


def build_initial_state(graph: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_graph(graph)
    initial = deepcopy(graph["initial_state"])
    return {
        "schema_version": STATE_SCHEMA,
        "graph_id": graph["graph_id"],
        "graph_sha256": validation["graph_sha256"],
        "product_contract_sha256": validation["product_contract_sha256"],
        "execution_status": initial["execution_status"],
        "current_node": initial["current_node"],
        "completed_nodes": list(initial["completed_nodes"]),
        "loop_counts": dict(initial["loop_counts"]),
        "accepted_full_ladder_flat_model": None,
        "accepted_lifecycle_model": None,
        "complete_full_trader_candidate": None,
        "protected_resources_spent": [],
        "owner_start_required": True,
        "stop_reason": initial["stop_reason"],
        "start_authorizations": [],
        "receipts": [],
        "transition_count": 0,
        "commands_executed": [],
        "training_executed": False,
        "protected_resource_accessed": False,
        "broker_contacted": False,
        "paper_order_submitted": False,
        "initialized_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }


def validate_state(
    state: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_graph(graph)
    errors: list[str] = []
    _require(state.get("schema_version") == STATE_SCHEMA, "state_schema", errors)
    _require(state.get("graph_id") == graph["graph_id"], "state_graph_id", errors)
    _require(
        state.get("graph_sha256") == validation["graph_sha256"],
        "state_graph_hash",
        errors,
    )
    _require(
        state.get("product_contract_sha256")
        == validation["product_contract_sha256"],
        "state_product_hash",
        errors,
    )
    _require(state.get("current_node") in node_map(graph), "state_node", errors)
    _require(
        state.get("execution_status")
        in {PARKED_STATUS, READY_STATUS, TERMINAL_STATUS},
        "state_execution_status",
        errors,
    )
    _require(state.get("commands_executed") == [], "commands_were_executed", errors)
    _require(state.get("broker_contacted") is False, "broker_contacted", errors)
    _require(
        state.get("paper_order_submitted") is False,
        "paper_order_submitted",
        errors,
    )
    if errors:
        raise GraphContractError(";".join(errors))
    return {
        "valid": True,
        "current_node": state["current_node"],
        "execution_status": state["execution_status"],
        "transition_count": state["transition_count"],
    }


def dry_run_routes(graph: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_graph(graph)
    routes = []
    for edge in graph["edges"]:
        routes.append(
            {
                "source": edge["from"],
                "outcome": edge["outcome"],
                "target": edge["to"],
                "bounded_loop": edge.get("loop_budget"),
                "validated_without_execution": True,
            }
        )
    return {
        "schema_version": "Protocol101FullTraderRouteDryRunV1",
        "graph_id": graph["graph_id"],
        "graph_sha256": validation["graph_sha256"],
        "product_contract_sha256": validation["product_contract_sha256"],
        "all_routes_valid": True,
        "routes_checked": len(routes),
        "nodes_reachable": validation["reachable_node_count"],
        "terminal_nodes": validation["terminal_nodes"],
        "commands_executed": [],
        "training_executed": False,
        "protected_resource_accessed": False,
        "broker_contacted": False,
        "paper_order_submitted": False,
        "routes": routes,
    }


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    current_node = str(state["current_node"])
    nodes = node_map(graph)
    node = nodes[current_node]
    _require(
        receipt.get("schema_version") == RECEIPT_SCHEMA,
        "receipt_schema",
        errors,
    )
    _require(receipt.get("node_id") == current_node, "receipt_node", errors)
    _require(
        receipt.get("graph_sha256") == state["graph_sha256"],
        "receipt_graph_hash",
        errors,
    )
    _require(
        receipt.get("product_contract_sha256")
        == state["product_contract_sha256"],
        "receipt_product_hash",
        errors,
    )
    _require(
        receipt.get("scope_assertions") == REQUIRED_SCOPE_ASSERTIONS,
        "product_drift_detected",
        errors,
    )
    producer_id = receipt.get("producer_id")
    reviewer_id = receipt.get("independent_reviewer_id")
    _require(
        isinstance(producer_id, str) and bool(producer_id.strip()),
        "producer_id",
        errors,
    )
    if node.get("independent_acceptance") is True:
        required_reviewers = node.get("reviewers")
        if isinstance(required_reviewers, list) and required_reviewers:
            reviewer_ids = receipt.get("reviewer_ids")
            _require(
                isinstance(reviewer_ids, Mapping)
                and set(reviewer_ids) == set(required_reviewers),
                "parallel_reviewer_ids_not_exact",
                errors,
            )
            if isinstance(reviewer_ids, Mapping):
                identities = list(reviewer_ids.values())
                _require(
                    all(
                        isinstance(identity, str) and bool(identity.strip())
                        for identity in identities
                    ),
                    "parallel_reviewer_identity_invalid",
                    errors,
                )
                _require(
                    len(set(identities)) == len(identities),
                    "parallel_reviewers_not_distinct",
                    errors,
                )
                _require(
                    producer_id not in identities,
                    "producer_may_not_independently_accept_own_work",
                    errors,
                )
        else:
            _require(
                isinstance(reviewer_id, str) and bool(reviewer_id.strip()),
                "independent_reviewer_id",
                errors,
            )
            _require(
                reviewer_id != producer_id,
                "producer_may_not_independently_accept_own_work",
                errors,
            )
    if node.get("owner_gate") is True:
        _require(
            receipt.get("owner_approved") is True,
            "owner_approval_required",
            errors,
        )
    artifact_hashes = receipt.get("artifact_hashes")
    _require(
        isinstance(artifact_hashes, Mapping) and bool(artifact_hashes),
        "artifact_hashes_missing",
        errors,
    )
    if isinstance(artifact_hashes, Mapping):
        for key, value in artifact_hashes.items():
            _require(
                isinstance(key, str)
                and isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                "artifact_hash_invalid",
                errors,
            )
    if errors:
        raise GraphContractError(";".join(errors))
    return dict(receipt)


def authorize_node_start(
    state: Mapping[str, Any],
    authorization: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    validate_state(state, graph)
    errors: list[str] = []
    _require(
        state.get("execution_status") == PARKED_STATUS,
        "node_is_not_parked",
        errors,
    )
    _require(
        state.get("owner_start_required") is True,
        "owner_start_is_not_required",
        errors,
    )
    _require(
        authorization.get("schema_version") == START_AUTHORIZATION_SCHEMA,
        "start_authorization_schema",
        errors,
    )
    _require(
        authorization.get("graph_sha256") == state["graph_sha256"],
        "start_authorization_graph_hash",
        errors,
    )
    _require(
        authorization.get("product_contract_sha256")
        == state["product_contract_sha256"],
        "start_authorization_product_hash",
        errors,
    )
    _require(
        authorization.get("node_id") == state["current_node"],
        "start_authorization_node",
        errors,
    )
    _require(
        authorization.get("authorized") is True,
        "start_authorization_not_true",
        errors,
    )
    _require(
        isinstance(authorization.get("owner_identity"), str)
        and bool(authorization["owner_identity"].strip()),
        "start_authorization_owner_identity",
        errors,
    )
    _require(
        authorization.get("scope_assertions") == REQUIRED_SCOPE_ASSERTIONS,
        "product_drift_detected",
        errors,
    )
    if errors:
        raise GraphContractError(";".join(errors))

    updated = deepcopy(dict(state))
    updated["execution_status"] = READY_STATUS
    updated["owner_start_required"] = False
    updated["stop_reason"] = None
    updated["start_authorizations"] = list(updated["start_authorizations"]) + [
        {
            "node_id": state["current_node"],
            "authorization_sha256": stable_hash(authorization),
        }
    ]
    updated["updated_at_utc"] = utc_now()
    validate_state(updated, graph)
    return updated


def advance_state(
    state: Mapping[str, Any],
    receipt: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    validate_state(state, graph)
    if state.get("execution_status") != READY_STATUS:
        raise GraphContractError("node_not_authorized_for_external_goal")
    accepted_receipt = validate_receipt(receipt, state=state, graph=graph)
    source = str(state["current_node"])
    outcome = accepted_receipt.get("outcome")
    matching = [
        edge
        for edge in graph["edges"]
        if edge["from"] == source and edge["outcome"] == outcome
    ]
    if len(matching) != 1:
        raise GraphContractError("receipt_outcome_not_an_allowed_unique_route")
    edge = matching[0]

    updated = deepcopy(dict(state))
    loop_counts = dict(updated.get("loop_counts", {}))
    budget_name = edge.get("loop_budget")
    if budget_name:
        limit = graph["controller_policy"]["loop_budgets"][budget_name]
        next_count = int(loop_counts.get(budget_name, 0)) + 1
        if next_count > limit:
            raise GraphContractError(f"loop_budget_exhausted:{budget_name}")
        loop_counts[budget_name] = next_count

    updated["loop_counts"] = loop_counts
    updated["completed_nodes"] = list(updated["completed_nodes"]) + [source]
    updated["current_node"] = edge["to"]
    target_node = node_map(graph)[edge["to"]]
    if target_node.get("kind") == "terminal":
        updated["execution_status"] = TERMINAL_STATUS
        updated["owner_start_required"] = False
        updated["stop_reason"] = target_node["purpose"]
    elif target_node.get("owner_gate") is True:
        updated["execution_status"] = PARKED_STATUS
        updated["owner_start_required"] = True
        updated["stop_reason"] = f"owner decision required before {edge['to']}"
    else:
        updated["execution_status"] = READY_STATUS
        updated["owner_start_required"] = False
        updated["stop_reason"] = None
    updated["transition_count"] = int(updated["transition_count"]) + 1
    updated["receipts"] = list(updated["receipts"]) + [
        {
            "node_id": source,
            "outcome": outcome,
            "receipt_sha256": stable_hash(accepted_receipt),
        }
    ]
    updated["updated_at_utc"] = utc_now()
    validate_state(updated, graph)
    return updated


def initialize_state_dir(
    graph: Mapping[str, Any],
    state_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    state_path = state_dir / "state.json"
    if state_path.exists() and not force:
        raise GraphContractError(f"state already exists: {state_path}")
    state_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_graph(graph)
    state = build_initial_state(graph)
    write_json_atomic(state_dir / "graph_definition.json", graph)
    write_json_atomic(state_dir / "graph_validation.json", validation)
    write_json_atomic(state_dir / "route_dry_run.json", dry_run_routes(graph))
    write_json_atomic(state_path, state)
    write_json_atomic(
        state_dir / "reset_receipt.json",
        {
            "schema_version": "Protocol101FullTraderGraphResetReceiptV1",
            "graph_id": graph["graph_id"],
            "graph_sha256": validation["graph_sha256"],
            "product_contract_sha256": validation["product_contract_sha256"],
            "completed_node": RESET_NODE,
            "next_node": ENTRY_NODE,
            "execution_status": PARKED_STATUS,
            "training_executed": False,
            "protected_resource_accessed": False,
            "broker_contacted": False,
            "paper_order_submitted": False,
            "created_at_utc": utc_now(),
        },
    )
    events_path = state_dir / "events.jsonl"
    if force and events_path.exists():
        events_path.unlink()
    append_jsonl(
        events_path,
        {
            "event": "graph_initialized",
            "graph_sha256": validation["graph_sha256"],
            "current_node": ENTRY_NODE,
            "execution_status": PARKED_STATUS,
            "timestamp_utc": utc_now(),
        },
    )
    return state


def main() -> int:
    args = parse_args()
    graph = read_json(args.graph)
    validation = validate_graph(graph)

    if args.mode == "validate":
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    if args.mode == "initialize":
        state = initialize_state_dir(graph, args.state_dir, force=args.force)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    if args.mode == "dry-run-routes":
        print(json.dumps(dry_run_routes(graph), indent=2, sort_keys=True))
        return 0

    state_path = args.state_dir / "state.json"
    state = read_json(state_path)
    validate_state(state, graph)

    if args.mode == "status":
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    if args.receipt is None:
        raise GraphContractError("--receipt is required")
    receipt = read_json(args.receipt)
    if args.mode == "authorize-start":
        updated = authorize_node_start(state, receipt, graph)
        write_json_atomic(state_path, updated)
        append_jsonl(
            args.state_dir / "events.jsonl",
            {
                "event": "node_start_authorized",
                "node": updated["current_node"],
                "authorization_sha256": updated["start_authorizations"][-1][
                    "authorization_sha256"
                ],
                "timestamp_utc": utc_now(),
            },
        )
        print(json.dumps(updated, indent=2, sort_keys=True))
        return 0

    updated = advance_state(state, receipt, graph)
    write_json_atomic(state_path, updated)
    append_jsonl(
        args.state_dir / "events.jsonl",
        {
            "event": "node_transition_recorded",
            "from": state["current_node"],
            "to": updated["current_node"],
            "receipt_sha256": updated["receipts"][-1]["receipt_sha256"],
            "timestamp_utc": utc_now(),
        },
    )
    print(json.dumps(updated, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
