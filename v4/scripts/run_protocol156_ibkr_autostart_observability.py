"""Protocol 156: IBKR autostart observability.

Summarize launchd state, IBKR API port reachability, and the morning automation
logs in one place. This script does not call broker order endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from v4.scripts.run_protocol140_ibkr_autostart_prep import (
    DEFAULT_LOG_DIR,
    DEFAULT_LAUNCHD_RUNTIME_DIR,
    GATEWAY_LABEL,
    PREFLIGHT_LABEL,
    SESSION_LABEL,
    candidate_api_ports,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_156_ibkr_autostart_observability")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
PACIFIC = ZoneInfo("America/Los_Angeles")
KNOWN_LOG_FILES = {
    "gateway_stdout": "ibgateway-paper.out.log",
    "gateway_stderr": "ibgateway-paper.err.log",
    "preflight_stdout": "protocol101-paper-preflight.out.log",
    "preflight_stderr": "protocol101-paper-preflight.err.log",
    "session_stdout": "protocol101-paper-session.out.log",
    "session_stderr": "protocol101-paper-session.err.log",
}
LABELS = [GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL]


PATTERNS = {
    "pythonpath_missing_for_session_runner": "ModuleNotFoundError: No module named 'v4'",
    "missing_live_market_data_entitlements": "missing_live_market_data_entitlements",
    "ibkr_market_data_not_subscribed": "Requested market data is not subscribed",
    "ibkr_competing_live_session": "No market data during competing live session",
    "ibkr_keepalive_socket_disconnect": "ibkr_keepalive_failed",
    "socket_disconnect": "Socket disconnect",
    "launchd_python_runtime_failed": "Fatal Python error: init_fs_encoding",
    "launchd_permission_denied": "PermissionError: [Errno 1] Operation not permitted",
    "api_port_open_detected": '"status": "port_open"',
    "ibkr_api_connected_detected": '"connected": true',
    "pass_status_detected": '"status": "pass"',
    "holding_status_detected": '"status": "holding"',
}


@dataclass(frozen=True)
class PortProbe:
    port: int
    open: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--ibkr-port", type=int, default=int(os.environ.get("IB_GATEWAY_API_PORT", "4002")))
    parser.add_argument("--ibkr-auto-ports", default=os.environ.get("IB_GATEWAY_API_PORTS", "4002,4000,7497,7496,4001"))
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(PACIFIC)
    session_date = args.session_date or now.date().isoformat()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    uid = os.getuid()
    launchd = {label: launchd_status(label, uid=uid) for label in LABELS}
    runtime_wrappers = collect_runtime_wrappers(DEFAULT_LAUNCHD_RUNTIME_DIR)
    logs = collect_logs(args.log_dir, tail_lines=args.tail_lines)
    ports = probe_ports(ports_from_args(args.ibkr_port, args.ibkr_auto_ports))
    live_logs = collect_live_trade_logs(Path("v4/logs/paper_trading"), session_date=session_date)
    signals = aggregate_signals(logs)
    decision = decide(launchd=launchd, logs=logs, ports=ports, signals=signals, runtime_wrappers=runtime_wrappers)
    payload = {
        "protocol": "156_ibkr_autostart_observability",
        "generated_at_pacific": now.isoformat(),
        "session_date": session_date,
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders": False,
        "broker_order_endpoint_called": False,
        "launchd": launchd,
        "runtime_wrappers": runtime_wrappers,
        "ports": [probe.__dict__ for probe in ports],
        "log_dir": str(args.log_dir.expanduser()),
        "logs": logs,
        "signals": dict(sorted(signals.items())),
        "live_trade_logs": live_logs,
        "interpretation": interpretation(decision, signals),
        "next_action": next_action(decision),
    }
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(report_path, payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, report_path)

    print(
        json.dumps(
            {
                "decision": decision,
                "report": str(report_path),
                "summary": str(summary_path),
                "next_action": payload["next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if decision.startswith("pass_") or decision.startswith("observe_") else 1


def ports_from_args(primary: int, auto_ports: str) -> list[int]:
    parsed: list[int] = []
    for raw in auto_ports.split(","):
        try:
            value = int(raw.strip())
        except ValueError:
            continue
        if value > 0:
            parsed.append(value)
    configured = parsed[0] if parsed else None
    return candidate_api_ports(primary, configured_port=configured)


def launchd_status(label: str, *, uid: int) -> dict[str, Any]:
    target = f"gui/{uid}/{label}"
    try:
        result = subprocess.run(
            ["launchctl", "print", target],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return {
            "label": label,
            "target": target,
            "installed": False,
            "loaded": False,
            "error": "launchctl_not_available",
        }
    return parse_launchctl_print(label=label, target=target, stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)


def parse_launchctl_print(*, label: str, target: str, stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    installed = returncode == 0
    status: dict[str, Any] = {
        "label": label,
        "target": target,
        "installed": installed,
        "loaded": installed,
        "returncode": returncode,
        "state": None,
        "runs": None,
        "last_exit_code": None,
        "path": None,
        "stdout_path": None,
        "stderr_path": None,
        "stderr": stderr.strip()[-500:] if stderr.strip() else None,
    }
    if not installed:
        return status

    for raw in stdout.splitlines():
        line = raw.strip()
        if line.startswith("state = "):
            status["state"] = line.split("=", 1)[1].strip()
        elif line.startswith("runs = "):
            status["runs"] = parse_int(line.split("=", 1)[1].strip())
        elif line.startswith("last exit code = "):
            status["last_exit_code"] = parse_int(line.split("=", 1)[1].strip())
        elif line.startswith("path = "):
            status["path"] = line.split("=", 1)[1].strip()
        elif line.startswith("stdout path = "):
            status["stdout_path"] = line.split("=", 1)[1].strip()
        elif line.startswith("stderr path = "):
            status["stderr_path"] = line.split("=", 1)[1].strip()
    return status


def parse_int(value: str) -> int | None:
    match = re.search(r"-?\d+", value)
    return int(match.group(0)) if match else None


def collect_logs(log_dir: Path, *, tail_lines: int) -> dict[str, Any]:
    expanded = log_dir.expanduser()
    logs = {
        name: file_digest(expanded / filename, tail_lines=tail_lines)
        for name, filename in KNOWN_LOG_FILES.items()
    }
    ibc_dir = expanded / "ibc"
    if ibc_dir.exists():
        ibc_logs = sorted(path for path in ibc_dir.glob("*.log") if path.is_file())
        logs["ibc_recent"] = [file_digest(path, tail_lines=min(tail_lines, 40)) for path in ibc_logs[-5:]]
    else:
        logs["ibc_recent"] = []
    return logs


def collect_runtime_wrappers(runtime_dir: Path) -> dict[str, Any]:
    expanded = runtime_dir.expanduser()
    wrappers: dict[str, Any] = {"runtime_dir": str(expanded)}
    for name in ("run_protocol101_paper_session.sh", "run_protocol101_paper_preflight.sh", "run_ibkr_autostart_status.sh"):
        path = expanded / name
        text = path.read_text(errors="replace") if path.exists() else ""
        wrappers[name] = {
            "path": str(path),
            "exists": path.exists(),
            "exports_pythonpath": "PYTHONPATH" in text and "REPO_ROOT" in text,
            "prefers_project_venv": ".venv/bin/python" in text and '== "/usr/bin/python3"' in text,
            "modified_at_pacific": datetime.fromtimestamp(path.stat().st_mtime, PACIFIC).isoformat() if path.exists() else None,
        }
    return wrappers


def file_digest(path: Path, *, tail_lines: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": 0,
        "modified_at_pacific": None,
        "tail": [],
        "signals": [],
        "json_events": [],
    }
    if not path.exists():
        return out
    stat = path.stat()
    out["size_bytes"] = int(stat.st_size)
    out["modified_at_pacific"] = datetime.fromtimestamp(stat.st_mtime, PACIFIC).isoformat()
    lines = tail_file(path, tail_lines=tail_lines)
    out["tail"] = lines
    text = "\n".join(lines)
    out["signals"] = sorted(name for name, needle in PATTERNS.items() if needle in text)
    out["json_events"] = parse_json_events(lines)
    return out


def tail_file(path: Path, *, tail_lines: int) -> list[str]:
    # Log files here are small enough for a normal read, and preserving line
    # order is more useful than clever byte-window tailing.
    return path.read_text(errors="replace").splitlines()[-tail_lines:]


def parse_json_events(lines: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        redacted = redact_event(event)
        events.append(redacted)
    return events[-20:]


def redact_event(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if any(token in lowered for token in ("password", "token", "secret", "apikey", "api_key")):
                out[key] = "<redacted>"
            else:
                out[key] = redact_event(item)
        return out
    if isinstance(value, list):
        return [redact_event(item) for item in value]
    return value


def aggregate_signals(logs: dict[str, Any]) -> Counter[str]:
    signals: Counter[str] = Counter()
    for name, digest in logs.items():
        if name == "ibc_recent":
            for ibc_digest in digest:
                signals.update(ibc_digest.get("signals", []))
            continue
        signals.update(digest.get("signals", []))
    return signals


def probe_ports(ports: list[int]) -> list[PortProbe]:
    probes: list[PortProbe] = []
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.35):
                probes.append(PortProbe(port=int(port), open=True))
        except OSError as exc:
            probes.append(PortProbe(port=int(port), open=False, error=exc.__class__.__name__))
    return probes


def collect_live_trade_logs(root: Path, *, session_date: str) -> dict[str, Any]:
    date_dir = root / session_date
    if not date_dir.exists():
        return {"session_date": session_date, "root": str(root), "exists": False, "jsonl_files": []}
    files = sorted(path for path in date_dir.rglob("*.jsonl") if path.is_file())
    return {
        "session_date": session_date,
        "root": str(root),
        "exists": True,
        "jsonl_files": [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "modified_at_pacific": datetime.fromtimestamp(path.stat().st_mtime, PACIFIC).isoformat(),
            }
            for path in files
        ],
    }


def decide(
    *,
    launchd: dict[str, Any],
    logs: dict[str, Any],
    ports: list[PortProbe],
    signals: Counter[str],
    runtime_wrappers: dict[str, Any] | None = None,
) -> str:
    loaded = [label for label, status in launchd.items() if status.get("loaded")]
    if len(loaded) < len(LABELS):
        return "blocked_launchagents_not_loaded"
    session_wrapper = (runtime_wrappers or {}).get("run_protocol101_paper_session.sh", {})
    preflight_wrapper = (runtime_wrappers or {}).get("run_protocol101_paper_preflight.sh", {})
    if (signals.get("launchd_python_runtime_failed") or signals.get("launchd_permission_denied")) and not (
        session_wrapper.get("prefers_project_venv") and preflight_wrapper.get("prefers_project_venv")
    ):
        return "blocked_launchd_python_runtime_failed"
    if signals.get("pythonpath_missing_for_session_runner") and not session_wrapper.get("exports_pythonpath"):
        return "blocked_session_runner_pythonpath_missing"
    if signals.get("missing_live_market_data_entitlements"):
        return "blocked_live_market_data_entitlements"
    if signals.get("ibkr_market_data_not_subscribed") or signals.get("ibkr_competing_live_session"):
        return "blocked_live_market_data_entitlements"
    if signals.get("ibkr_keepalive_socket_disconnect") or signals.get("socket_disconnect"):
        return "blocked_gateway_keepalive_disconnect"
    if any(probe.open for probe in ports):
        return "pass_ibkr_api_port_reachable"
    if signals.get("ibkr_api_connected_detected") or signals.get("pass_status_detected"):
        return "observe_previous_ibkr_api_connection_no_current_port"
    return "blocked_ibkr_autostart_no_api_confirmation"


def interpretation(decision: str, signals: Counter[str]) -> list[str]:
    notes = {
        "blocked_launchagents_not_loaded": "One or more launchd jobs are missing or unloaded, so the morning automation may not run.",
        "blocked_session_runner_pythonpath_missing": "The session job reached Python but could not import the local v4 package. Reinstall/copy the patched runtime wrappers before tomorrow's run.",
        "blocked_launchd_python_runtime_failed": "The scheduled launchd job reached Apple Command Line Tools Python and crashed during interpreter startup. Use the project venv wrapper for scheduled jobs.",
        "blocked_live_market_data_entitlements": "Gateway/API startup reached the market-data probe, but IBKR refused at least one live data request. This is a data entitlement/session issue, not a model issue.",
        "blocked_gateway_keepalive_disconnect": "Gateway connected and then disconnected during keepalive. The next check is whether Gateway stayed logged in and API settings remained enabled.",
        "pass_ibkr_api_port_reachable": "An IBKR API port is currently reachable from localhost.",
        "observe_previous_ibkr_api_connection_no_current_port": "Recent logs show a prior successful API connection, but no API port is currently reachable.",
        "blocked_ibkr_autostart_no_api_confirmation": "The logs and current port probes do not yet prove that IB Gateway reached the API listener.",
    }
    out = [notes.get(decision, "No interpretation rule is defined for this decision.")]
    if signals:
        out.append("Detected signals: " + ", ".join(f"{name}={count}" for name, count in sorted(signals.items())))
    return out


def next_action(decision: str) -> str:
    if decision == "blocked_session_runner_pythonpath_missing":
        return "Copy/reinstall the patched launchd runtime wrapper so the paper-session job exports PYTHONPATH before running python -m v4..."
    if decision == "blocked_launchd_python_runtime_failed":
        return "Copy/reinstall the patched launchd runtime wrappers so scheduled jobs prefer the project .venv Python instead of Apple Command Line Tools Python."
    if decision == "blocked_live_market_data_entitlements":
        return "Confirm the required IBKR paper market-data subscriptions/session state, then run the no-order shadow path again during market hours."
    if decision == "blocked_gateway_keepalive_disconnect":
        return "Verify Gateway remains logged in after automatic startup and rerun the status report after the scheduled launch."
    if decision == "blocked_launchagents_not_loaded":
        return "Run the generated install script from Protocol140, then rerun this status report."
    if decision.startswith("pass_"):
        return "Run Protocol101 no-order/paper session and inspect the live JSONL plus this status report after the session."
    return "Rerun this status report after Gateway is expected to be open, then inspect the report's log tails for the first failing stage."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 156: IBKR Autostart Observability",
        "",
        "No paid data was downloaded. No broker order endpoint was called. No orders were placed.",
        "",
        f"- Generated: `{payload['generated_at_pacific']}`",
        f"- Session date: `{payload['session_date']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Log directory: `{payload['log_dir']}`",
        f"- Next action: {payload['next_action']}",
        "",
        "## What It Means",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["interpretation"])
    lines.extend(
        [
            "",
            "## LaunchAgents",
            "",
            "| label | loaded | state | runs | last exit | stdout | stderr |",
            "| --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for label, status in payload["launchd"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"`{status.get('loaded')}`",
                    f"`{status.get('state')}`",
                    f"`{status.get('runs')}`",
                    f"`{status.get('last_exit_code')}`",
                    f"`{status.get('stdout_path')}`",
                    f"`{status.get('stderr_path')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Current IBKR API Ports", "", "| port | open | error |", "| ---: | ---: | --- |"])
    for probe in payload["ports"]:
        lines.append(f"| {probe['port']} | `{probe['open']}` | `{probe.get('error')}` |")
    lines.extend(["", "## Signal Counts", "", "| signal | count |", "| --- | ---: |"])
    if payload["signals"]:
        for name, count in payload["signals"].items():
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Runtime Wrappers", "", "| wrapper | exists | exports PYTHONPATH | prefers project venv | modified |", "| --- | ---: | ---: | ---: | --- |"])
    wrappers = payload.get("runtime_wrappers", {})
    for name, status in wrappers.items():
        if name == "runtime_dir":
            continue
        lines.append(
            f"| {name} | `{status.get('exists')}` | `{status.get('exports_pythonpath')}` | `{status.get('prefers_project_venv')}` | `{status.get('modified_at_pacific')}` |"
        )
    lines.extend(["", "## Log Files", ""])
    for name, digest in payload["logs"].items():
        if name == "ibc_recent":
            lines.append("### Recent IBC Logs")
            if not digest:
                lines.append("")
                lines.append("- No recent IBC log files found.")
                lines.append("")
                continue
            for item in digest:
                lines.append(f"- `{item['path']}` size={item['size_bytes']} modified=`{item['modified_at_pacific']}` signals=`{item['signals']}`")
            lines.append("")
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Path: `{digest['path']}`")
        lines.append(f"- Exists: `{digest['exists']}`")
        lines.append(f"- Size bytes: `{digest['size_bytes']}`")
        lines.append(f"- Modified: `{digest['modified_at_pacific']}`")
        lines.append(f"- Signals: `{digest['signals']}`")
        if digest["json_events"]:
            lines.append("- Recent JSON events:")
            for event in digest["json_events"][-5:]:
                lines.append(f"  - `{json.dumps(event, sort_keys=True)}`")
        if digest["tail"]:
            lines.append("")
            lines.append("```text")
            lines.extend(digest["tail"][-20:])
            lines.append("```")
        lines.append("")
    live = payload["live_trade_logs"]
    lines.extend(
        [
            "## Live Paper JSONL Logs",
            "",
            f"- Date directory exists: `{live['exists']}`",
            f"- Root: `{live['root']}`",
        ]
    )
    if live["jsonl_files"]:
        for item in live["jsonl_files"]:
            lines.append(f"- `{item['path']}` size={item['size_bytes']} modified=`{item['modified_at_pacific']}`")
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-18 Protocol 156 IBKR Autostart Observability"
    entry = f"""

{marker}

```text
Date: 2026-05-18
Decision / Experiment: Added a single IBKR morning autostart observability report covering launchd state, log tails, API port probes, and known failure signals.
Reason: The user needs to know whether the automatic IBKR login/session path is working and where it fails before Tuesday's paper-trading run.
Data Used: Local launchd state and local logs only. No paid data was downloaded, no broker order endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: Use the report after the scheduled 6:28/6:29/6:30 Pacific automation to diagnose startup, preflight, session, and market-data blockers.
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
