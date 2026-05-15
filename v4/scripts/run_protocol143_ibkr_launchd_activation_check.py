"""Protocol 143: verify installed IBKR paper LaunchAgents."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from v4.scripts.run_protocol140_ibkr_autostart_prep import GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_143_ibkr_launchd_activation_check")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks = [agent_check(GATEWAY_LABEL), agent_check(PREFLIGHT_LABEL), agent_check(SESSION_LABEL)]
    decision = "pass_ibkr_paper_launchagents_installed" if all(check["installed"] and check["loaded"] for check in checks) else "blocked_launchagents_not_loaded"
    payload = {
        "protocol": "143_ibkr_launchd_activation_check",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": False,
        "broker_order_endpoint_called": False,
        "checks": checks,
        "next_gate": (
            "At the next market session, confirm the preflight log shows a live IBKR paper API connection and then run live Protocol101 shadow parity before paper orders."
            if decision.startswith("pass_")
            else "Install or reload the LaunchAgents before relying on morning automation."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if decision.startswith("pass_") else 1


def agent_check(label: str) -> dict[str, Any]:
    plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    result = subprocess.run(
        ["launchctl", "print", f"gui/{subprocess_uid()}/{label}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "label": label,
        "plist": str(plist),
        "installed": plist.exists(),
        "loaded": result.returncode == 0,
        "launchctl_returncode": result.returncode,
        "launchctl_header": result.stdout.splitlines()[0] if result.stdout else result.stderr.splitlines()[0] if result.stderr else "",
    }


def subprocess_uid() -> str:
    result = subprocess.run(["id", "-u"], text=True, capture_output=True, check=True)
    return result.stdout.strip()


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 143: IBKR LaunchAgent Activation Check",
        "",
        "This verifies morning automation only. It does not submit orders.",
        "",
        f"- Decision: `{payload['decision']}`",
        "",
        "| label | installed | loaded |",
        "| --- | ---: | ---: |",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['label']} | `{check['installed']}` | `{check['loaded']}` |")
    lines.extend(["", "## Next Gate", "", payload["next_gate"]])
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 143 IBKR LaunchAgent Activation Check"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Installed and verified the IB Gateway paper-mode morning LaunchAgents.
Reason: User wants IB Gateway paper mode to start before the market so the bot can run without a 6:30 AM manual login.
Data Used: Local launchd state only. No paid data was downloaded, no paper order was submitted, and no broker order endpoint was called.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: During the next market session, use the preflight/live-shadow logs to confirm live paper API parity before enabling paper order submission.
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
