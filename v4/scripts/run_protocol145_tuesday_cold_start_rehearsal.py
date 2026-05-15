"""Protocol 145: Tuesday cold-start rehearsal for IBKR paper mode.

This is an operational test, not a trading test. It starts IB Gateway through
the same startup script used by launchd, waits for a local API port, and records
the expected blocker if credentials/2FA prevent the API listener from opening.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

from v4.scripts.run_protocol140_ibkr_autostart_prep import GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_145_tuesday_cold_start_rehearsal")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_START_SCRIPT = Path("v4/ops/ibkr/start_ib_gateway_paper_ibc.sh")
DEFAULT_PORTS = (4002, 4000, 7497, 7496, 4001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--start-script", type=Path, default=DEFAULT_START_SCRIPT)
    parser.add_argument("--ports", default=",".join(str(port) for port in DEFAULT_PORTS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--start-wait-seconds", type=int, default=45)
    parser.add_argument("--post-start-wait-seconds", type=int, default=10)
    parser.add_argument("--skip-start", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ports = candidate_ports(args.ports)
    before = port_status(args.host, ports)
    launchd = {
        GATEWAY_LABEL: launchd_status(GATEWAY_LABEL),
        PREFLIGHT_LABEL: launchd_status(PREFLIGHT_LABEL),
        SESSION_LABEL: launchd_status(SESSION_LABEL),
    }
    start_result = {"skipped": True, "returncode": None, "stdout": "", "stderr": ""}
    if not args.skip_start:
        start_result = run_start_script(args.start_script, ports=ports, wait_seconds=int(args.start_wait_seconds))
    after = wait_for_ports(args.host, ports, seconds=max(0, int(args.post_start_wait_seconds)))
    decision = decide(before=before, after=after, launchd=launchd, start_result=start_result, skip_start=bool(args.skip_start))
    payload = {
        "protocol": "145_tuesday_cold_start_rehearsal",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": False,
        "broker_order_endpoint_called": False,
        "market_data_endpoint_called": False,
        "host": args.host,
        "ports": ports,
        "launchd": launchd,
        "port_status_before": before,
        "start_result": start_result,
        "port_status_after": after,
        "interpretation": interpretation(decision),
        "next_gate": next_gate(decision),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if decision.startswith(("pass_", "expected_blocker_")) else 1


def candidate_ports(text: str) -> list[int]:
    ports: list[int] = []
    for value in str(text).split(","):
        try:
            port = int(value.strip())
        except ValueError:
            continue
        if port > 0 and port not in ports:
            ports.append(port)
    return ports


def port_status(host: str, ports: list[int]) -> dict[str, Any]:
    rows = []
    for port in ports:
        rows.append({"port": port, "open": can_connect(host, port, timeout=1.0)})
    return {
        "any_open": any(row["open"] for row in rows),
        "open_ports": [row["port"] for row in rows if row["open"]],
        "rows": rows,
    }


def wait_for_ports(host: str, ports: list[int], *, seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, seconds)
    latest = port_status(host, ports)
    while not latest["any_open"] and time.monotonic() < deadline:
        time.sleep(1.0)
        latest = port_status(host, ports)
    latest["wait_seconds"] = seconds
    return latest


def can_connect(host: str, port: int, *, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def launchd_status(label: str) -> dict[str, Any]:
    uid = subprocess.run(["id", "-u"], text=True, capture_output=True, check=True).stdout.strip()
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{label}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "label": label,
        "loaded": result.returncode == 0,
        "returncode": result.returncode,
        "header": result.stdout.splitlines()[0] if result.stdout else result.stderr.splitlines()[0] if result.stderr else "",
    }


def run_start_script(script: Path, *, ports: list[int], wait_seconds: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["IB_GATEWAY_WAIT_SECONDS"] = str(max(1, wait_seconds))
    env["IB_GATEWAY_API_PORTS"] = ",".join(str(port) for port in ports)
    result = subprocess.run(
        ["/bin/bash", str(script)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=max(10, wait_seconds + 20),
    )
    return {
        "skipped": False,
        "script": str(script),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def decide(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    launchd: dict[str, Any],
    start_result: dict[str, Any],
    skip_start: bool,
) -> str:
    if not all(item.get("loaded") for item in launchd.values()):
        return "blocked_launchagents_not_loaded"
    if before.get("any_open"):
        return "pass_api_already_running_before_rehearsal"
    if after.get("any_open") and int(start_result.get("returncode") or 0) == 0:
        return "pass_cold_start_api_ready"
    if skip_start and not after.get("any_open"):
        return "expected_blocker_api_closed_start_skipped"
    if not after.get("any_open"):
        return "expected_blocker_gateway_login_required_or_api_port_closed"
    return "blocked_unclassified_cold_start_state"


def interpretation(decision: str) -> str:
    if decision == "pass_cold_start_api_ready":
        return "IB Gateway opened and the paper API became reachable. Tuesday can move to account probe and live-data parity."
    if decision == "pass_api_already_running_before_rehearsal":
        return "The paper API was already reachable before the cold-start rehearsal."
    if decision == "expected_blocker_gateway_login_required_or_api_port_closed":
        return (
            "This is the expected failure when IBC cannot complete credentials/2FA or no API listener opens. "
            "IBC can enter the stored username/password, but it cannot bypass IBKR Mobile/2FA approval."
        )
    if decision == "expected_blocker_api_closed_start_skipped":
        return "Start was skipped and no API port was open."
    return "Cold-start state needs manual inspection."


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return "Run Protocol141 account probe, then Protocol119/124 live-data parity, then paper executor dry-run."
    if decision == "expected_blocker_gateway_login_required_or_api_port_closed":
        return (
            "Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally, approve any IBKR Mobile/2FA challenge during startup, "
            "and rerun this rehearsal. The next passing state should expose port 4002 or 4000."
        )
    return "Fix LaunchAgent/startup state before Tuesday."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 145: Tuesday Cold-Start Rehearsal",
        "",
        "No paid data was downloaded. No market-data endpoint was called. No order endpoint was called.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Ports before: `{payload['port_status_before']['open_ports']}`",
        f"- Ports after: `{payload['port_status_after']['open_ports']}`",
        f"- Start returncode: `{payload['start_result']['returncode']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Start Output",
        "",
        "```text",
        str(payload["start_result"].get("stdout") or payload["start_result"].get("stderr") or ""),
        "```",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 145 Tuesday Cold-Start Rehearsal"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Rehearsed Tuesday's cold-start path with IB Gateway closed, using the same startup script as the morning LaunchAgent.
Reason: User wanted to know whether the unattended startup path works and expected failure at the missing username/password login stage.
Data Used: Local app/launchd/API-port checks only. No paid data was downloaded, no market-data endpoint was called, and no order endpoint was called.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Ports before={payload['port_status_before']['open_ports']}; ports after={payload['port_status_after']['open_ports']}. Report: {report_path}
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    existing = ledger.read_text() if ledger.exists() else ""
    if marker not in existing:
        ledger.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


if __name__ == "__main__":
    raise SystemExit(main())
