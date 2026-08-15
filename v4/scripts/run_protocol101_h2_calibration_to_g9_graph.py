"""Run the durable bounded H2 calibration-to-G9 autoresearch graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


AUDIT = Path("v4/audit/autoresearch")
GRAPH_DIR = AUDIT / "protocol101_h2_policy5_calibration_to_g9_graph"
CAL_AUDIT = (
    AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001_audit"
)
G9_AUDIT = AUDIT / "protocol101_h2_policy5_g9_seed45_attempt001_audit"
MAX_ATTEMPTS = 3
SOURCE_PATHS = (
    Path(__file__),
    Path("v4/model/protocol101_h2_calibration_repair.py"),
    Path("v4/scripts/run_protocol101_h2_policy5_calibration_repair.py"),
    Path("v4/scripts/run_protocol101_h2_policy5_calibration_gate.py"),
    Path("v4/scripts/run_protocol101_h2_policy5_calibration_audit.py"),
    Path("v4/scripts/run_protocol101_h2_policy5_g9.py"),
    Path("v4/scripts/run_protocol101_h2_policy5_g9_audit.py"),
    Path("v4/model/protocol101_scoped_stage1_hgb.py"),
    Path("v4/model/protocol101_canonical_stage1_contract.py"),
    Path("v4/model/protocol101_serial_simulator.py"),
    Path("v4/scripts/run_protocol101_scoped_stage1_independent_audit.py"),
)
NODE_NAMES = (
    "CAL-PREREGISTER",
    "CAL-MACHINERY-SMOKE",
    "CAL-RUN",
    "CAL-G1-G8-GATE",
    "CAL-G1-G8-INDEPENDENT-AUDIT",
    "C1-G9-MACHINERY",
    "C2-G9-RUN-SEED45",
    "C3-G9-INDEPENDENT-AUDIT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--owner-approved-offline-training", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def source_hashes() -> dict[str, str]:
    return {str(path): sha256_path(path) for path in SOURCE_PATHS}


def route_after_calibration(verdict: str) -> str:
    return (
        "continue_to_g9"
        if verdict == "accepted_eligible_G1_G8"
        else "stop_scientific_calibration_result"
    )


def route_after_g9(verdict: str) -> str:
    if verdict == "g9_pass_candidate_eligible_for_final_fit":
        return "complete_g9_pass"
    if verdict == "g9_fail_candidate_burned":
        return "complete_g9_scientific_fail"
    return "stop_g9_non_scientific_terminal"


def initial_state() -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101H2CalibrationToG9GraphV1",
        "created_at_utc": now(),
        "updated_at_utc": now(),
        "status": "running",
        "current_node": NODE_NAMES[0],
        "nodes": {
            name: {
                "status": "pending",
                "attempts": 0,
                "started_at_utc": None,
                "completed_at_utc": None,
                "last_error": None,
            }
            for name in NODE_NAMES
        },
        "source_hashes": source_hashes(),
        "max_mechanical_attempts_per_node": MAX_ATTEMPTS,
        "calibration_route": None,
        "g9_route": None,
        "terminal_verdict": None,
        "side_effect_boundary": {
            "protected_holdout_read": False,
            "recorder_or_confirmation_data_read": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    payload["state_hash"] = stable_hash(payload)
    return payload


def update_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = now()
    state.pop("state_hash", None)
    state["state_hash"] = stable_hash(state)
    write_json(path, state)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def verify_frozen_sources(state: dict[str, Any]) -> None:
    current = source_hashes()
    if current != state.get("source_hashes"):
        changed = sorted(
            path
            for path in set(current) | set(state.get("source_hashes") or {})
            if current.get(path) != (state.get("source_hashes") or {}).get(path)
        )
        raise RuntimeError(f"graph_source_changed_after_freeze:{changed}")


def command_for(node: str) -> list[str]:
    python = sys.executable
    commands = {
        "CAL-PREREGISTER": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_calibration_repair",
            "--mode",
            "preregister",
        ],
        "CAL-MACHINERY-SMOKE": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_calibration_repair",
            "--mode",
            "smoke",
        ],
        "CAL-RUN": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_calibration_repair",
            "--mode",
            "run",
        ],
        "CAL-G1-G8-GATE": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_calibration_gate",
        ],
        "CAL-G1-G8-INDEPENDENT-AUDIT": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_calibration_audit",
        ],
        "C1-G9-MACHINERY": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_g9",
            "--mode",
            "smoke",
        ],
        "C2-G9-RUN-SEED45": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_g9",
            "--mode",
            "run",
            "--owner-approved-offline-training",
        ],
        "C3-G9-INDEPENDENT-AUDIT": [
            python,
            "-m",
            "v4.scripts.run_protocol101_h2_policy5_g9_audit",
        ],
    }
    return commands[node]


def output_log_name(node: str, attempt: int) -> str:
    safe = node.lower().replace("-", "_")
    return f"{safe}_attempt{attempt:02d}.log"


def run_node(
    *,
    node: str,
    state: dict[str, Any],
    graph_dir: Path,
    state_path: Path,
    events_path: Path,
) -> None:
    node_state = state["nodes"][node]
    while int(node_state["attempts"]) < MAX_ATTEMPTS:
        verify_frozen_sources(state)
        node_state["attempts"] = int(node_state["attempts"]) + 1
        node_state["status"] = "running"
        node_state["started_at_utc"] = node_state["started_at_utc"] or now()
        node_state["last_error"] = None
        state["current_node"] = node
        update_state(state_path, state)
        attempt = int(node_state["attempts"])
        command = command_for(node)
        started = now()
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONHASHSEED": "0"},
            check=False,
        )
        log_path = graph_dir / "logs" / output_log_name(node, attempt)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout)
        event = {
            "event_at_utc": now(),
            "node": node,
            "attempt": attempt,
            "started_at_utc": started,
            "returncode": completed.returncode,
            "command": command,
            "log": {
                "path": str(log_path),
                "sha256": sha256_path(log_path),
            },
        }
        append_event(events_path, event)
        if completed.returncode == 0:
            node_state["status"] = "complete"
            node_state["completed_at_utc"] = now()
            update_state(state_path, state)
            return
        node_state["status"] = "mechanical_retry_pending"
        node_state["last_error"] = completed.stdout[-4000:]
        update_state(state_path, state)
    node_state["status"] = "blocked_after_three_mechanical_attempts"
    state["status"] = "blocked_mechanical"
    state["terminal_verdict"] = f"{node}:mechanical_attempt_budget_exhausted"
    update_state(state_path, state)
    raise RuntimeError(state["terminal_verdict"])


def mark_remaining_skipped(
    state: dict[str, Any],
    *,
    after_node: str,
    reason: str,
) -> None:
    start = NODE_NAMES.index(after_node) + 1
    for name in NODE_NAMES[start:]:
        if state["nodes"][name]["status"] == "pending":
            state["nodes"][name]["status"] = "skipped"
            state["nodes"][name]["last_error"] = reason


def main() -> int:
    args = parse_args()
    if not args.owner_approved_offline_training:
        raise RuntimeError("owner approval missing for bounded offline graph")
    args.graph_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.graph_dir / "state.json"
    events_path = args.graph_dir / "events.jsonl"
    if state_path.exists():
        state = load_json(state_path)
        stored_hash = state.pop("state_hash", None)
        if stored_hash != stable_hash(state):
            raise RuntimeError("graph state self-hash mismatch")
        state["state_hash"] = stored_hash
    else:
        state = initial_state()
        write_json(state_path, state)
    if state.get("status") in {"complete", "scientific_stop"}:
        print(json.dumps(state, indent=2, sort_keys=True, default=str))
        return 0
    verify_frozen_sources(state)
    for node in NODE_NAMES:
        if state["nodes"][node]["status"] == "complete":
            continue
        run_node(
            node=node,
            state=state,
            graph_dir=args.graph_dir,
            state_path=state_path,
            events_path=events_path,
        )
        if node == "CAL-G1-G8-INDEPENDENT-AUDIT":
            verdict = load_json(CAL_AUDIT / "summary.json")["verdict"]
            route = route_after_calibration(str(verdict))
            state["calibration_route"] = route
            if route != "continue_to_g9":
                mark_remaining_skipped(state, after_node=node, reason=verdict)
                state["status"] = "scientific_stop"
                state["terminal_verdict"] = verdict
                update_state(state_path, state)
                break
        if node == "C3-G9-INDEPENDENT-AUDIT":
            verdict = load_json(G9_AUDIT / "summary.json")["verdict"]
            state["g9_route"] = route_after_g9(str(verdict))
            state["status"] = "complete"
            state["terminal_verdict"] = verdict
            update_state(state_path, state)
    report = [
        "# H2 Calibration-to-G9 Graph",
        "",
        f"- Status: `{state['status']}`",
        f"- Terminal verdict: `{state['terminal_verdict']}`",
        f"- Calibration route: `{state['calibration_route']}`",
        f"- G9 route: `{state['g9_route']}`",
        "",
        "## Nodes",
        "",
    ]
    report.extend(
        f"- {name}: `{state['nodes'][name]['status']}` "
        f"(attempts={state['nodes'][name]['attempts']})"
        for name in NODE_NAMES
    )
    (args.graph_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
