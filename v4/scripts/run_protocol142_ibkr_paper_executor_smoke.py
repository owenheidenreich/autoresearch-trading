"""Protocol 142: guarded IBKR paper-order executor smoke.

Dry-run mode validates the end-to-end executor without submitting an order.
Submit mode exists for future paper trading, but requires an explicit intent
file plus the paper-order guard flags/environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.ibkr_paper_executor import PaperExecutionConfig, execute_guarded_paper_order
from v4.live.ibkr_paper_guard import PaperOrderIntent
from v4.live.paper_trade_log import DEFAULT_TRADE_LOG_ROOT, export_trade_log_csv, trade_log_path
from v4.scripts.run_protocol141_ibkr_paper_order_guard import candidate_ports, probe_ibkr_account


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_142_ibkr_paper_executor_smoke")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "paper-submit"), default="dry-run")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--intent-json", type=Path, default=None)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4000)
    parser.add_argument("--ibkr-auto-ports", default="4000,4002,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=142)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--trade-log-run-id", default="protocol142_executor_smoke")
    parser.add_argument("--no-trade-log", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    intent = load_intent(args.intent_json) if args.intent_json else sample_intent()
    dry_run = args.mode == "dry-run"
    if args.mode == "paper-submit" and args.intent_json is None:
        payload = blocked_payload(args, intent, "paper_submit_requires_intent_json")
        return finish(args, payload)

    result = run_executor(args, intent=intent, dry_run=dry_run)
    decision = decide(args.mode, result)
    payload = {
        "protocol": "142_ibkr_paper_executor_smoke",
        "decision": decision,
        "mode": args.mode,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": bool(result.get("paper_order_submitted")),
        "broker_order_endpoint_called": bool(result.get("broker_order_endpoint_called")),
        "executor_result": result,
        "trade_log": trade_log_outputs(args),
        "next_gate": next_gate(decision),
    }
    return finish(args, payload)


def run_executor(args: argparse.Namespace, *, intent: PaperOrderIntent, dry_run: bool) -> dict[str, Any]:
    try:
        from ib_insync import IB, LimitOrder, Option  # type: ignore
    except ImportError:
        return {"status": "blocked", "reason": "missing_ib_insync", "broker_order_endpoint_called": False}
    account_probe = probe_ibkr_account(args)
    account_id = args.account_id or account_probe.get("primary_account_id")
    if not account_probe.get("connected"):
        return {
            "status": "blocked",
            "reason": account_probe.get("blocked_reason", "ibkr_connection_failed"),
            "account_probe": redact_account_probe_for_executor(account_probe),
            "broker_order_endpoint_called": False,
        }
    ib = IB()
    try:
        port = int(account_probe.get("port") or first_probe_port(args))
        ib.connect(args.ibkr_host, port, clientId=args.ibkr_client_id + 1000, timeout=8)
        result = execute_guarded_paper_order(
            ib=ib,
            option_cls=Option,
            order_cls=LimitOrder,
            intent=intent,
            account_id=account_id,
            account_cash=float(args.paper_cash),
            open_positions=0,
            quote={"bid": max(0.01, intent.limit_price - 0.10), "ask": intent.limit_price, "reference_ask": intent.limit_price, "quote_age_ms": 100},
            context={"context_age_ms": 100},
            enable_paper_orders=bool(args.enable_paper_orders),
            acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
            dry_run=dry_run,
            config=PaperExecutionConfig(),
            trade_log_root=None if args.no_trade_log else args.trade_log_root,
            trade_log_run_id=args.trade_log_run_id,
        )
        result["account_probe"] = redact_account_probe_for_executor(account_probe)
        return result
    finally:
        if ib.isConnected():
            ib.disconnect()


def first_probe_port(args: argparse.Namespace) -> int:
    return candidate_ports(args)[0]


def load_intent(path: Path) -> PaperOrderIntent:
    payload = json.loads(path.read_text())
    return PaperOrderIntent(
        action=str(payload["action"]),
        symbol=str(payload["symbol"]),
        expiry=str(payload["expiry"]),
        strike=float(payload["strike"]),
        right=str(payload["right"]),
        quantity=int(payload["quantity"]),
        limit_price=float(payload["limit_price"]),
        trading_class=str(payload.get("trading_class", "SPXW")),
        exchange=str(payload.get("exchange", "SMART")),
        currency=str(payload.get("currency", "USD")),
    )


def sample_intent() -> PaperOrderIntent:
    return PaperOrderIntent(
        action="BUY",
        symbol="SPX",
        expiry="20260320",
        strike=6700.0,
        right="C",
        quantity=1,
        limit_price=10.0,
    )


def decide(mode: str, result: dict[str, Any]) -> str:
    if mode == "dry-run" and result.get("status") == "dry_run_pass":
        return "pass_paper_executor_validates_without_order_submission"
    if result.get("status") == "submitted" and result.get("paper_order_submitted") is True:
        return "paper_order_submitted"
    return f"blocked_{result.get('reason', 'paper_executor_not_ready')}"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Use this executor only after a live Protocol101 decision stream passes schema, quote freshness, account, "
            "and risk gates. Paper-submit mode requires an explicit order-intent JSON."
        )
    if decision == "paper_order_submitted":
        return "Immediately verify order state and fill/cancel handling in the paper account ledger."
    return "Fix the executor blocker before using broker-connected paper orders."


def blocked_payload(args: argparse.Namespace, intent: PaperOrderIntent, reason: str) -> dict[str, Any]:
    return {
        "protocol": "142_ibkr_paper_executor_smoke",
        "decision": f"blocked_{reason}",
        "mode": args.mode,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": False,
        "broker_order_endpoint_called": False,
        "executor_result": {"status": "blocked", "reason": reason, "intent": intent.__dict__},
        "next_gate": "Pass an explicit Protocol101 intent JSON before paper-submit mode can place an order.",
    }


def finish(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    if payload.get("trade_log", {}).get("jsonl_path"):
        jsonl = Path(payload["trade_log"]["jsonl_path"])
        csv_path = Path(payload["trade_log"]["csv_path"])
        if jsonl.exists():
            payload["trade_log"]["csv_export"] = export_trade_log_csv(jsonl, csv_path)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if not str(payload["decision"]).startswith("blocked_") else 1


def write_report(path: Path, payload: dict[str, Any]) -> None:
    result = payload["executor_result"]
    log = payload.get("trade_log", {})
    lines = [
        "# Protocol 142: IBKR Paper Executor Smoke",
        "",
        "No paid data was downloaded. Dry-run mode does not submit any order.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Executor status: `{result.get('status')}`",
        f"- Reason: `{result.get('reason')}`",
        f"- Broker order endpoint called: `{payload['broker_order_endpoint_called']}`",
        f"- Paper orders submitted: `{payload['paper_orders_submitted']}`",
        f"- Trade log JSONL: `{log.get('jsonl_path')}`",
        f"- Trade log CSV: `{log.get('csv_path')}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 142 IBKR Paper Executor Smoke"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Added a guarded IBKR paper-order executor and smoke-tested the non-submitting path.
Reason: User approved eventual paper trades, so the project needs an explicit executor that can submit only validated paper-order intents after live shadow parity passes.
Data Used: Local IBKR paper connection in dry-run mode. No paid data was downloaded and no paper order was submitted unless paper-submit mode is explicitly used.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: Feed this executor only from a live Protocol101 order-intent stream that has passed schema/freshness/risk checks.
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


def redact_account_probe_for_executor(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "primary_account_id" in out:
        value = str(out.pop("primary_account_id") or "")
        out["primary_account_id_redacted"] = f"{value[:2]}***{value[-2:]}" if len(value) > 4 else "***"
    return out


def trade_log_outputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_trade_log:
        return {"enabled": False, "jsonl_path": None, "csv_path": None}
    path = trade_log_path(root=args.trade_log_root, run_id=args.trade_log_run_id)
    return {
        "enabled": True,
        "jsonl_path": str(path),
        "csv_path": str(path.with_suffix(".csv")),
        "run_id": args.trade_log_run_id,
    }


if __name__ == "__main__":
    raise SystemExit(main())
