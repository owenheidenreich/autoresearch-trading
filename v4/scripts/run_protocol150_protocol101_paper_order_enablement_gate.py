"""Protocol 150: controlled paper-order enablement gate.

This script does not submit orders. It answers whether a specific session is
allowed to move from no-order shadow into broker-connected IBKR paper order
testing, and optionally writes a small runtime flag consumed by future runners.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from v4.live.ibkr_paper_guard import PAPER_PERMISSION_ENV, PaperOrderGuardConfig, paper_order_permission, redact_account_id
from v4.live.paper_trade_log import DEFAULT_TRADE_LOG_ROOT, load_trade_log
from v4.scripts.run_protocol148_protocol101_post_session_analyzer import analyze_session_rows, resolve_trade_log


DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_150_protocol101_paper_order_enablement_gate")
DEFAULT_RUNTIME_FLAG = Path("v4/runtime/protocol101_paper_order_enablement.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    parser.add_argument("--write-runtime-flag", action="store_true")
    parser.add_argument("--runtime-flag", type=Path, default=DEFAULT_RUNTIME_FLAG)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trade_log = resolve_trade_log(
        trade_log=args.trade_log,
        root=args.trade_log_root,
        session=args.session,
        run_id=args.run_id,
    )
    rows = load_trade_log(trade_log)
    if not rows:
        raise SystemExit(f"no rows found in {trade_log}")
    analysis = analyze_session_rows(rows, trade_log=trade_log)
    permission = paper_order_permission(
        enable_paper_orders=bool(args.enable_paper_orders),
        acknowledge_paper_loss=bool(args.acknowledge_paper_loss),
        account_id=args.account_id,
        config=PaperOrderGuardConfig(starting_paper_cash=float(args.paper_cash)),
    )
    gate = evaluate_gate(analysis, permission)
    out_dir = args.out_root / str(analysis["session"]) / str(analysis["run_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_written = False
    if args.write_runtime_flag and gate["passed"]:
        args.runtime_flag.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_flag.write_text(
            json.dumps(
                {
                    "paper_orders_enabled": True,
                    "real_money_trading": False,
                    "enabled_at": datetime.now(timezone.utc).isoformat(),
                    "session": analysis["session"],
                    "run_id": analysis["run_id"],
                    "account_id_redacted": redact_account_id(args.account_id),
                    "source_trade_log": str(trade_log),
                    "required_env": f"{PAPER_PERMISSION_ENV}=YES",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        runtime_written = True
    payload = {
        "protocol": "150_protocol101_paper_order_enablement_gate",
        "decision": "ready_for_guarded_ibkr_paper_orders" if gate["passed"] else "blocked_paper_order_enablement_gate",
        "paid_data_downloaded": False,
        "live_orders": False,
        "real_money_trading": False,
        "broker_order_endpoint_called": False,
        "source_trade_log": str(trade_log),
        "account_id_redacted": redact_account_id(args.account_id),
        "permission": permission,
        "gate": gate,
        "runtime_flag_written": runtime_written,
        "runtime_flag": str(args.runtime_flag),
        "next_gate": next_gate(gate),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(out_dir / "report.md")}, indent=2))
    return 0 if gate["passed"] else 1


def evaluate_gate(analysis: dict[str, Any], permission: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if analysis["validation"]["status"] != "pass":
        reasons.append("session_log_validation_failed")
    if analysis["broker_order_endpoint_called_rows"] > 0:
        reasons.append("existing_broker_endpoint_rows_require_manual_review")
    if not analysis["startup_and_data"]["live_capture_pass"]:
        reasons.append("live_shadow_capture_not_passed")
    if not analysis["startup_and_data"]["live_parity_ready"]:
        reasons.append("live_parity_not_ready")
    if not permission.get("passed"):
        reasons.extend(str(reason) for reason in permission.get("reasons", []))
    return {
        "passed": not reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "reasons": reasons,
        "required_runtime_mode": "paper",
        "required_env": f"{PAPER_PERMISSION_ENV}=YES",
        "max_initial_contracts": 1,
    }


def next_gate(gate: dict[str, Any]) -> str:
    if gate["passed"]:
        return "Switch the morning session to paper mode and feed only risk-gated Protocol101 order intents to the guarded executor."
    return "Keep the morning session in no-order-shadow mode until every listed blocker is cleared."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 150: Protocol101 Paper-Order Enablement Gate",
        "",
        "This gate does not submit orders. It controls whether the next run may use the guarded IBKR paper executor.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Account: `{payload['account_id_redacted']}`",
        f"- Gate passed: `{payload['gate']['passed']}`",
        f"- Gate reason: `{payload['gate']['reason']}`",
        f"- Runtime flag written: `{payload['runtime_flag_written']}`",
        f"- Runtime flag: `{payload['runtime_flag']}`",
        "",
        "## Required",
        "",
        "- Passing live shadow capture",
        "- Passing live parity",
        "- Clean session log",
        "- No unexplained existing broker endpoint rows",
        "- Explicit `--enable-paper-orders` and `--acknowledge-paper-loss` flags",
        f"- `{payload['gate']['required_env']}`",
        "- Paper account ID, normally `DU...`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
