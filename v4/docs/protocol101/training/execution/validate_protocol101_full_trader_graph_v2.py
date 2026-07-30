"""Read-only structural validator for Protocol101 Full Trader Graph V2.

This script never trains, never touches protected evidence, never contacts a
broker, and never mutates any file. It only reads the graph JSON and prints a
validation report. It is the FT2-01 consolidation validator; it is not the
runtime controller (that is v4/scripts/run_protocol101_full_trader_graph.py,
which targets the V1 schema).

Checks:
  1. JSON parses and is an object.
  2. schema_version == "Protocol101FullTraderGraphV2".
  3. Node ids are unique.
  4. Every edge from/to references a defined node; outcome is a non-empty
     string; (from, outcome) is unique (deterministic routing).
  5. Every non-terminal node has at least one outgoing edge.
  6. Every terminal node has zero outgoing edges.
  7. Every node is reachable from the reachability root (reset node).
  8. The set of kind=="terminal" nodes equals the declared terminal_states.
  9. Every model_training node declares a valid action_space.
 10. No independent_acceptance node is run by the executor role.
 11. Every gate in gate_dominance_requirements dominates its target
     (target unreachable from root when the gate node is removed).
 12. Every edge loop_budget names a defined controller_policy.loop_budgets key.
 13. initial_state matches the consolidation contract: FT2-01 complete,
     current_node FT2-04, parked, owner_start_required true.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

SCHEMA = "Protocol101FullTraderGraphV2"
EXPECTED_COMPLETED = ["FT2-00-GRAPH-RESET", "FT2-01-CONSOLIDATED-AUTHORITY"]
EXPECTED_CURRENT = "FT2-04-PATH-LABEL-FREEZE"
VALID_ACTION_SPACES = {
    "flat_state_full_governed_ladder",
    "open_state_HOLD_EXIT",
}


def reachable(nodes, edges, start, blocked=frozenset()):
    if start in blocked:
        return set()
    outgoing = defaultdict(list)
    for edge in edges:
        outgoing[edge["from"]].append(edge["to"])
    reached = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for target in outgoing.get(node, []):
            if target in blocked or target in reached:
                continue
            reached.add(target)
            queue.append(target)
    return reached


def validate(path):
    errors = []
    graph = json.loads(Path(path).read_text())
    if not isinstance(graph, dict):
        return {"valid": False, "errors": ["graph_not_object"]}

    if graph.get("schema_version") != SCHEMA:
        errors.append(f"schema_version_mismatch:{graph.get('schema_version')}")

    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", [])
    node_ids = [n["id"] for n in raw_nodes]
    nodes = {n["id"]: n for n in raw_nodes}
    if len(node_ids) != len(nodes):
        errors.append("node_ids_not_unique")

    # Edge referential integrity + deterministic routing.
    seen_routes = set()
    for i, edge in enumerate(raw_edges):
        if edge.get("from") not in nodes:
            errors.append(f"edge_{i}_source_undefined:{edge.get('from')}")
        if edge.get("to") not in nodes:
            errors.append(f"edge_{i}_target_undefined:{edge.get('to')}")
        outcome = edge.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            errors.append(f"edge_{i}_outcome_invalid")
        key = (edge.get("from"), outcome)
        if key in seen_routes:
            errors.append(f"duplicate_route:{key}")
        seen_routes.add(key)
        loop_budgets = graph.get("controller_policy", {}).get("loop_budgets", {})
        if "loop_budget" in edge and edge["loop_budget"] not in loop_budgets:
            errors.append(f"edge_{i}_loop_budget_undefined:{edge['loop_budget']}")

    # Outgoing-edge rules by terminality.
    outgoing = defaultdict(list)
    for edge in raw_edges:
        outgoing[edge["from"]].append(edge)
    declared_terminals = set(graph.get("terminal_states", []))
    kind_terminals = {nid for nid, n in nodes.items() if n.get("kind") == "terminal"}
    for nid, node in nodes.items():
        is_terminal = node.get("kind") == "terminal"
        has_out = bool(outgoing.get(nid))
        if is_terminal and has_out:
            errors.append(f"terminal_has_outgoing:{nid}")
        if not is_terminal and not has_out:
            errors.append(f"nonterminal_has_no_outgoing:{nid}")
        if node.get("model_training") is True:
            if node.get("action_space") not in VALID_ACTION_SPACES:
                errors.append(f"training_action_space_invalid:{nid}")
        if node.get("independent_acceptance") is True and node.get("actor") == "executor":
            errors.append(f"producer_reviews_own_work:{nid}")

    if kind_terminals != declared_terminals:
        errors.append(
            "terminal_set_mismatch:"
            f"kind={sorted(kind_terminals)} declared={sorted(declared_terminals)}"
        )

    # Reachability from the reset root.
    root = graph.get("reachability_root") or graph.get("reset_node")
    if root not in nodes:
        errors.append(f"reachability_root_undefined:{root}")
    else:
        reached = reachable(nodes, raw_edges, root)
        unreachable = set(nodes) - reached
        if unreachable:
            errors.append(f"unreachable_nodes:{sorted(unreachable)}")

    # Gate dominance.
    for target, gates in graph.get("gate_dominance_requirements", {}).items():
        if target not in nodes:
            errors.append(f"gate_target_undefined:{target}")
            continue
        for gate in gates:
            if gate not in nodes:
                errors.append(f"gate_node_undefined:{gate}")
                continue
            without = reachable(nodes, raw_edges, root, blocked={gate})
            if target in without:
                errors.append(f"gate_does_not_dominate:{gate}->{target}")

    # initial_state contract.
    initial = graph.get("initial_state", {})
    if initial.get("completed_nodes") != EXPECTED_COMPLETED:
        errors.append(f"initial_completed_nodes:{initial.get('completed_nodes')}")
    if initial.get("current_node") != EXPECTED_CURRENT:
        errors.append(f"initial_current_node:{initial.get('current_node')}")
    if initial.get("execution_status") != "parked_before_node":
        errors.append(f"initial_execution_status:{initial.get('execution_status')}")
    if initial.get("owner_start_required") is not True:
        errors.append("initial_owner_start_required_not_true")

    return {
        "valid": not errors,
        "errors": errors,
        "schema_version": graph.get("schema_version"),
        "graph_id": graph.get("graph_id"),
        "node_count": len(nodes),
        "edge_count": len(raw_edges),
        "reachable_from_root": len(reachable(nodes, raw_edges, root)) if root in nodes else 0,
        "reachability_root": root,
        "terminal_states": sorted(kind_terminals),
        "model_training_nodes": sorted(
            nid for nid, n in nodes.items() if n.get("model_training") is True
        ),
        "owner_gate_nodes": sorted(
            nid for nid, n in nodes.items() if n.get("owner_gate") is True
        ),
        "independent_acceptance_nodes": sorted(
            nid for nid, n in nodes.items() if n.get("independent_acceptance") is True
        ),
        "gate_dominance_checked": sorted(
            graph.get("gate_dominance_requirements", {}).keys()
        ),
    }


def main():
    default = (
        Path(__file__).resolve().parent
        if False
        else Path(
            "/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/"
            "training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
        )
    )
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    report = validate(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
