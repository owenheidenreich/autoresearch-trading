"""Protocol 146: IBC credential-backed login readiness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from v4.ops.ibkr.write_ibc_runtime_config import (
    DEFAULT_OUT,
    DEFAULT_PASSWORD_SERVICE,
    DEFAULT_USERNAME_SERVICE,
    read_keychain_secret,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_146_ibc_credential_readiness")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_IBC_PATH = Path.home() / ".autoresearch-trading/ibc/3.23.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ibc-path", type=Path, default=DEFAULT_IBC_PATH)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--username-service", default=DEFAULT_USERNAME_SERVICE)
    parser.add_argument("--password-service", default=DEFAULT_PASSWORD_SERVICE)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    install = install_status(args.ibc_path)
    credential_status = {
        "username_present": read_keychain_secret(args.username_service) is not None,
        "password_present": read_keychain_secret(args.password_service) is not None,
        "username_service": args.username_service,
        "password_service": args.password_service,
    }
    config_result = generate_runtime_config(args) if all((credential_status["username_present"], credential_status["password_present"])) else {
        "status": "skipped_missing_credentials",
        "out": str(args.runtime_config),
    }
    decision = decide(install, credential_status, config_result)
    payload = {
        "protocol": "146_ibc_credential_readiness",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": False,
        "broker_order_endpoint_called": False,
        "market_data_endpoint_called": False,
        "ibc": install,
        "credentials": credential_status,
        "runtime_config": sanitize_config_result(config_result),
        "next_gate": next_gate(decision),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if decision.startswith(("pass_", "expected_blocker_")) else 1


def install_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "installed": (path / "IBC.jar").exists() and (path / "scripts/ibcstart.sh").exists(),
        "ibc_jar": str(path / "IBC.jar"),
        "ibcstart": str(path / "scripts/ibcstart.sh"),
        "version_file": (path / "version").read_text().strip() if (path / "version").exists() else None,
    }


def generate_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "v4/ops/ibkr/write_ibc_runtime_config.py"),
            "--out",
            str(args.runtime_config),
            "--username-service",
            args.username_service,
            "--password-service",
            args.password_service,
            "--api-port",
            "4002",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"status": "unparseable_stdout", "stdout": result.stdout}
    payload["returncode"] = result.returncode
    payload["stderr"] = result.stderr
    return payload


def decide(install: dict[str, Any], credentials: dict[str, Any], config_result: dict[str, Any]) -> str:
    if not install.get("installed"):
        return "blocked_ibc_not_installed"
    if not credentials.get("username_present") or not credentials.get("password_present"):
        return "expected_blocker_missing_keychain_credentials"
    if config_result.get("status") != "written":
        return "blocked_ibc_runtime_config_not_written"
    return "pass_ibc_credentials_ready_for_cold_start_rehearsal"


def sanitize_config_result(result: dict[str, Any]) -> dict[str, Any]:
    clean = dict(result)
    clean.pop("stdout", None)
    clean.pop("stderr", None)
    return clean


def next_gate(decision: str) -> str:
    if decision == "pass_ibc_credentials_ready_for_cold_start_rehearsal":
        return "Run Protocol145 cold-start rehearsal again; IBC should enter credentials and expose the paper API port after any required 2FA approval."
    if decision == "expected_blocker_missing_keychain_credentials":
        return "Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally, then rerun Protocol146. Do not paste credentials into chat."
    return "Fix IBC installation/config generation before Tuesday."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 146: IBC Credential Readiness",
        "",
        "No paid data was downloaded. No market-data endpoint was called. No order endpoint was called.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- IBC installed: `{payload['ibc']['installed']}`",
        f"- Username in Keychain: `{payload['credentials']['username_present']}`",
        f"- Password in Keychain: `{payload['credentials']['password_present']}`",
        f"- Runtime config status: `{payload['runtime_config']['status']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 146 IBC Credential Readiness"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Added IBC credential-backed login readiness for IB Gateway paper mode.
Reason: IB Gateway logs out daily, so the morning startup path must enter username/password from local secure storage instead of requiring manual wake-up.
Data Used: Local IBC installation and macOS Keychain presence checks only. No credentials were committed or printed, no paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. IBC installed={payload['ibc']['installed']}; username_present={payload['credentials']['username_present']}; password_present={payload['credentials']['password_present']}. Report: {report_path}
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
