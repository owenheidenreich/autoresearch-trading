"""Protocol 140: IB Gateway paper-mode autostart preparation.

This prepares launchd assets and a preflight for unattended paper-mode startup.
It does not place orders and does not store or print credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_140_ibkr_autostart_prep")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_JTS_INI = Path.home() / "Jts/jts.ini"
DEFAULT_LAUNCHD_DIR = Path("v4/ops/launchd")
DEFAULT_LOG_DIR = Path.home() / "Library/Logs/autoresearch-trading"
GATEWAY_LABEL = "com.autoresearch.ibgateway.paper"
PREFLIGHT_LABEL = "com.autoresearch.protocol101.paper-preflight"
DEFAULT_IBKR_PAPER_API_PORT = 4002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--jts-ini", type=Path, default=DEFAULT_JTS_INI)
    parser.add_argument("--launchd-dir", type=Path, default=DEFAULT_LAUNCHD_DIR)
    parser.add_argument("--app-path", type=Path, default=None)
    parser.add_argument("--gateway-hour", type=int, default=6)
    parser.add_argument("--gateway-minute", type=int, default=20)
    parser.add_argument("--preflight-hour", type=int, default=6)
    parser.add_argument("--preflight-minute", type=int, default=30)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.launchd_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    app_path = args.app_path or discover_ib_gateway_app()
    jts = parse_jts_ini(args.jts_ini)
    jts_api_port = int(jts.get("IBGateway", {}).get("LocalServerPort") or 0)
    api_port = int(os.environ.get("IB_GATEWAY_API_PORT") or DEFAULT_IBKR_PAPER_API_PORT)
    api_ports = candidate_api_ports(api_port, configured_port=jts_api_port)
    checks = readiness_checks(app_path=app_path, jts=jts, jts_path=args.jts_ini)
    assets = write_launchd_assets(
        launchd_dir=args.launchd_dir,
        app_path=app_path,
        api_port=api_port,
        api_ports=api_ports,
        gateway_hour=int(args.gateway_hour),
        gateway_minute=int(args.gateway_minute),
        preflight_hour=int(args.preflight_hour),
        preflight_minute=int(args.preflight_minute),
    )
    write_install_scripts(args.launchd_dir)
    decision = "ready_to_install_ib_gateway_paper_autostart" if all(check["passed"] for check in checks) else "blocked_ib_gateway_autostart_config"
    payload = {
        "protocol": "140_ibkr_autostart_prep",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders": False,
        "broker_order_endpoint_called": False,
        "app_path": str(app_path) if app_path else None,
        "jts_ini": str(args.jts_ini),
        "jts_config_redacted": redact_jts(jts),
        "api_port": api_port,
        "api_ports": api_ports,
        "checks": checks,
        "launchd_assets": assets,
        "autostart_assumption": (
            "launchd can open IB Gateway in paper mode, but IBKR may still require saved credentials/2FA. "
            "If Gateway pauses at login, finish that login once; the preflight will keep reporting port-not-open until the API listener is available."
        ),
        "next_gate": next_gate(decision),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if decision.startswith("ready_") else 1


def discover_ib_gateway_app() -> Path | None:
    apps = sorted(
        path
        for path in (Path.home() / "Applications").glob("IB Gateway*/IB Gateway*.app")
        if "Uninstaller" not in path.name
    )
    return apps[-1] if apps else None


def parse_jts_ini(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    section = ""
    if not path.exists():
        return out
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip().strip("\r")
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            out.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out.setdefault(section, {})[key.strip()] = value.strip()
    return out


def readiness_checks(*, app_path: Path | None, jts: dict[str, dict[str, str]], jts_path: Path) -> list[dict[str, Any]]:
    ibgateway = jts.get("IBGateway", {})
    logon = jts.get("Logon", {})
    return [
        {"name": "ib_gateway_app_exists", "passed": bool(app_path and app_path.exists()), "detail": str(app_path) if app_path else "not found"},
        {"name": "jts_ini_exists", "passed": jts_path.exists(), "detail": str(jts_path)},
        {"name": "paper_mode_configured", "passed": logon.get("tradingMode") == "p", "detail": f"tradingMode={logon.get('tradingMode')}"},
        {"name": "api_only_configured", "passed": str(ibgateway.get("ApiOnly", "")).lower() == "true", "detail": f"ApiOnly={ibgateway.get('ApiOnly')}"},
        {"name": "api_port_configured", "passed": bool(ibgateway.get("LocalServerPort")), "detail": f"LocalServerPort={ibgateway.get('LocalServerPort')}"},
        {"name": "trusted_localhost", "passed": "127.0.0.1" in str(ibgateway.get("TrustedIPs", "")), "detail": f"TrustedIPs={ibgateway.get('TrustedIPs')}"},
    ]


def write_launchd_assets(
    *,
    launchd_dir: Path,
    app_path: Path | None,
    api_port: int,
    api_ports: list[int],
    gateway_hour: int,
    gateway_minute: int,
    preflight_hour: int,
    preflight_minute: int,
) -> dict[str, str]:
    app = str(app_path or "")
    gateway_plist = launchd_dir / f"{GATEWAY_LABEL}.plist"
    preflight_plist = launchd_dir / f"{PREFLIGHT_LABEL}.plist"
    gateway_payload = launchd_payload(
        label=GATEWAY_LABEL,
        program_arguments=[
            "/bin/bash",
            str(REPO_ROOT / "v4/ops/ibkr/start_ib_gateway_paper_ibc.sh"),
        ],
        hour=gateway_hour,
        minute=gateway_minute,
        stdout=DEFAULT_LOG_DIR / "ibgateway-paper.out.log",
        stderr=DEFAULT_LOG_DIR / "ibgateway-paper.err.log",
        environment={
            "IB_GATEWAY_APP": app,
            "IB_GATEWAY_API_PORT": str(api_port),
            "IB_GATEWAY_API_PORTS": ",".join(str(port) for port in api_ports),
            "REPO_ROOT": str(REPO_ROOT),
        },
    )
    preflight_payload = launchd_payload(
        label=PREFLIGHT_LABEL,
        program_arguments=[
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "v4/ops/ibkr/wait_for_ibkr_api.py"),
            "--port",
            str(api_port),
            "--auto-ports",
            ",".join(str(port) for port in api_ports),
            "--timeout-seconds",
            "600",
            "--run-entitlement-probe",
            "--repo-root",
            str(REPO_ROOT),
        ],
        hour=preflight_hour,
        minute=preflight_minute,
        stdout=DEFAULT_LOG_DIR / "protocol101-paper-preflight.out.log",
        stderr=DEFAULT_LOG_DIR / "protocol101-paper-preflight.err.log",
        environment={},
    )
    gateway_plist.write_bytes(plistlib.dumps(gateway_payload, sort_keys=True))
    preflight_plist.write_bytes(plistlib.dumps(preflight_payload, sort_keys=True))
    return {
        "gateway_plist": str(gateway_plist),
        "preflight_plist": str(preflight_plist),
        "install_script": str(launchd_dir / "install_ibkr_paper_autostart.sh"),
        "uninstall_script": str(launchd_dir / "uninstall_ibkr_paper_autostart.sh"),
    }


def launchd_payload(
    *,
    label: str,
    program_arguments: list[str],
    hour: int,
    minute: int,
    stdout: Path,
    stderr: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    return {
        "Label": label,
        "ProgramArguments": program_arguments,
        "StartCalendarInterval": {"Hour": int(hour), "Minute": int(minute)},
        "RunAtLoad": False,
        "StandardOutPath": str(stdout),
        "StandardErrorPath": str(stderr),
        "EnvironmentVariables": environment,
    }


def candidate_api_ports(primary_port: int, configured_port: int | None = None) -> list[int]:
    ports: list[int] = []
    for port in (primary_port, configured_port or 0, 4002, 4000, 7497, 7496, 4001):
        if int(port) > 0 and int(port) not in ports:
            ports.append(int(port))
    return ports


def write_install_scripts(launchd_dir: Path) -> None:
    install = launchd_dir / "install_ibkr_paper_autostart.sh"
    uninstall = launchd_dir / "uninstall_ibkr_paper_autostart.sh"
    install.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/autoresearch-trading"
launchctl bootout "gui/$UID/{GATEWAY_LABEL}" 2>/dev/null || true
launchctl bootout "gui/$UID/{PREFLIGHT_LABEL}" 2>/dev/null || true
cp "{launchd_dir / (GATEWAY_LABEL + '.plist')}" "$HOME/Library/LaunchAgents/"
cp "{launchd_dir / (PREFLIGHT_LABEL + '.plist')}" "$HOME/Library/LaunchAgents/"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/{GATEWAY_LABEL}.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/{PREFLIGHT_LABEL}.plist"
launchctl enable "gui/$UID/{GATEWAY_LABEL}"
launchctl enable "gui/$UID/{PREFLIGHT_LABEL}"
echo "Installed IB Gateway paper autostart and Protocol101 preflight LaunchAgents."
"""
    )
    uninstall.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
launchctl bootout "gui/$UID/{GATEWAY_LABEL}" 2>/dev/null || true
launchctl bootout "gui/$UID/{PREFLIGHT_LABEL}" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/{GATEWAY_LABEL}.plist"
rm -f "$HOME/Library/LaunchAgents/{PREFLIGHT_LABEL}.plist"
echo "Removed IB Gateway paper autostart LaunchAgents."
"""
    )
    install.chmod(0o755)
    uninstall.chmod(0o755)


def redact_jts(jts: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    sensitive = ("user", "account", "pass", "token", "key")
    redacted: dict[str, dict[str, str]] = {}
    for section, values in jts.items():
        redacted[section] = {}
        for key, value in values.items():
            if any(part in key.lower() for part in sensitive):
                redacted[section][key] = "<redacted>"
            else:
                redacted[section][key] = value
    return redacted


def next_gate(decision: str) -> str:
    if decision.startswith("ready_"):
        return (
            "Install the LaunchAgents when desired, then run the IBKR preflight during market hours. "
            "Paper-order submission remains blocked until the paper-order guard and live Protocol101 shadow parity pass."
        )
    return "Fix the failed local IB Gateway app/JTS paper-mode checks before installing morning autostart."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 140: IBKR Paper Autostart Prep",
        "",
        "No paid data was downloaded. No broker order endpoint was called. No orders were placed.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- IB Gateway app: `{payload['app_path']}`",
        f"- Configured API port: `{payload['api_port']}`",
        f"- Candidate API ports: `{payload['api_ports']}`",
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "| --- | ---: | --- |",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['name']} | `{check['passed']}` | `{check['detail']}` |")
    lines.extend(
        [
            "",
            "## Assets",
            "",
            f"- Gateway LaunchAgent: `{payload['launchd_assets']['gateway_plist']}`",
            f"- Preflight LaunchAgent: `{payload['launchd_assets']['preflight_plist']}`",
            f"- Install script: `{payload['launchd_assets']['install_script']}`",
            f"- Uninstall script: `{payload['launchd_assets']['uninstall_script']}`",
            "",
            "## Autostart Assumption",
            "",
            payload["autostart_assumption"],
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 140 IBKR Paper Autostart Prep"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Prepared IB Gateway paper-mode morning autostart and Protocol101 API preflight assets.
Reason: User wants the system to start IB Gateway paper mode automatically before the market opens, so the project needs OS-level startup and a broker API readiness check before paper trading.
Data Used: Local IB Gateway app path and redacted JTS config only. No paid data was downloaded, no broker order endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: Install LaunchAgents when desired, then run paper-order guard and live shadow parity before enabling paper orders.
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
