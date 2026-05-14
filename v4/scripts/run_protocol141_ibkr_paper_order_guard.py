"""Protocol 141: IBKR paper-order guard and account preflight.

This is the bridge between research artifacts and broker-connected paper
trading. By default it only validates guardrails. It never submits orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.ibkr_paper_guard import (
    PAPER_PERMISSION_ENV,
    PaperOrderGuardConfig,
    PaperOrderIntent,
    paper_order_permission,
    redact_account_id,
    validate_order_intent,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_141_ibkr_paper_order_guard")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("guard-smoke", "account-probe", "paper-order-dry-run"), default="guard-smoke")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4000)
    parser.add_argument("--ibkr-auto-ports", default="4000,4002,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=141)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = PaperOrderGuardConfig(starting_paper_cash=float(args.paper_cash))
    account_probe = {}
    account_id = args.account_id
    broker_connection_endpoint_called = False
    if args.mode == "account-probe":
        account_probe = probe_ibkr_account(args)
        broker_connection_endpoint_called = bool(account_probe.get("broker_connection_endpoint_called"))
        account_id = account_probe.get("primary_account_id") or account_id
    stored_account_probe = redact_account_probe(account_probe)

    permission = paper_order_permission(
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        account_id=account_id,
        config=config,
    )
    dry_run_intent = sample_order_intent()
    intent_validation = validate_order_intent(
        dry_run_intent,
        account_cash=float(args.paper_cash),
        open_positions=0,
        quote={"bid": 9.90, "ask": 10.00, "reference_ask": 10.00, "quote_age_ms": 100},
        context={"context_age_ms": 100},
        config=config,
    )
    decision = decide(args.mode, permission=permission, account_probe=account_probe, intent_validation=intent_validation)
    payload = {
        "protocol": "141_ibkr_paper_order_guard",
        "decision": decision,
        "mode": args.mode,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": False,
        "broker_connection_endpoint_called": broker_connection_endpoint_called,
        "broker_order_endpoint_called": False,
        "paper_permission_env_required": PAPER_PERMISSION_ENV,
        "account_id_redacted": redact_account_id(account_id),
        "config": config.__dict__,
        "permission": permission,
        "account_probe": stored_account_probe,
        "sample_intent_validation": intent_validation,
        "sample_intent": dry_run_intent.__dict__,
        "next_gate": next_gate(decision),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if not decision.startswith("blocked_") else 1


def probe_ibkr_account(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from ib_insync import IB  # type: ignore
    except ImportError:
        return {"broker_connection_endpoint_called": False, "connected": False, "blocked_reason": "missing_ib_insync"}
    attempts: list[dict[str, Any]] = []
    for port in candidate_ports(args):
        ib = IB()
        try:
            ib.connect(args.ibkr_host, port, clientId=args.ibkr_client_id, timeout=8)
            accounts = list(ib.managedAccounts() or [])
            ib.disconnect()
            return {
                "broker_connection_endpoint_called": True,
                "connected": True,
                "host": args.ibkr_host,
                "port": port,
                "attempts": attempts + [{"port": port, "status": "connected"}],
                "managed_account_count": len(accounts),
                "primary_account_id": accounts[0] if accounts else None,
                "managed_accounts_redacted": [redact_account_id(account) for account in accounts],
            }
        except Exception as exc:
            attempts.append({"port": port, "status": "failed", "error": str(exc)})
            if ib.isConnected():
                ib.disconnect()
    return {
        "broker_connection_endpoint_called": True,
        "connected": False,
        "blocked_reason": "ibkr_connection_failed",
        "attempts": attempts,
    }


def redact_account_probe(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "primary_account_id" in out:
        out["primary_account_id_redacted"] = redact_account_id(out.pop("primary_account_id"))
    return out


def candidate_ports(args: argparse.Namespace) -> list[int]:
    ports: list[int] = []
    for value in [str(args.ibkr_port), *str(args.ibkr_auto_ports).split(",")]:
        try:
            port = int(value.strip())
        except ValueError:
            continue
        if port > 0 and port not in ports:
            ports.append(port)
    return ports


def sample_order_intent() -> PaperOrderIntent:
    return PaperOrderIntent(
        action="BUY",
        symbol="SPX",
        expiry="20260320",
        strike=6700.0,
        right="C",
        quantity=1,
        limit_price=10.0,
    )


def decide(
    mode: str,
    *,
    permission: dict[str, Any],
    account_probe: dict[str, Any],
    intent_validation: dict[str, Any],
) -> str:
    if mode == "account-probe" and not account_probe.get("connected"):
        return f"blocked_{account_probe.get('blocked_reason', 'ibkr_account_probe_failed')}"
    if not intent_validation.get("passed"):
        return "blocked_sample_order_intent_validation_failed"
    if permission.get("passed"):
        return "ready_for_guarded_paper_order_submission_after_live_shadow_parity"
    if mode == "guard-smoke":
        return "pass_default_blocks_paper_orders_until_explicitly_enabled"
    return "blocked_paper_order_permission_not_enabled"


def next_gate(decision: str) -> str:
    if decision.startswith("ready_"):
        return (
            "Keep this guard around any paper-order executor. The next requirement is a live Protocol101 shadow stream "
            "that passes schema/freshness/parity before a BUY or SELL order is submitted."
        )
    if decision == "pass_default_blocks_paper_orders_until_explicitly_enabled":
        return "Run account-probe with IB Gateway open; keep order submission disabled until live shadow parity passes."
    return "Fix the blocked paper-order guard input before broker-connected paper trading."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 141: IBKR Paper-Order Guard",
        "",
        "No paid data was downloaded. No paper order was submitted. No broker order endpoint was called.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Account: `{payload['account_id_redacted']}`",
        f"- Permission reason: `{payload['permission']['reason']}`",
        f"- Sample order validation: `{payload['sample_intent_validation']['reason']}`",
        "",
        "## Required To Enable Paper Orders",
        "",
        "- `--enable-paper-orders`",
        "- `--acknowledge-paper-loss`",
        f"- `{payload['paper_permission_env_required']}=YES`",
        "- Paper account detected, normally with a `DU` prefix.",
        "- Live Protocol101 shadow parity and risk gates pass.",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 141 IBKR Paper-Order Guard"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Added and ran an IBKR paper-order permission guard around future Protocol101 broker-connected paper trading.
Reason: User approved paper trades, but the project needs an explicit paper-only permission layer so no real-money order path can be reached by accident.
Data Used: Local guard configuration only unless account-probe mode was requested. No paid data was downloaded and no broker order endpoint was called.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: Run account-probe with IB Gateway open, then run no-order live shadow parity before enabling paper-order submission.
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
