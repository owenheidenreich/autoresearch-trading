"""Protocol 147: unattended Protocol101 morning paper-session runner.

This is the launchd target that starts at the equity market open. It is built
to fail closed: it can analyze and log automatically, but broker order
submission remains disabled unless a later executor explicitly clears the live
parity and paper-order gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, time as wall_time
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from zoneinfo import ZoneInfo

from v4.live.paper_trade_log import (
    DEFAULT_TRADE_LOG_ROOT,
    append_trade_event,
    export_trade_log_csv,
    make_trade_log_event,
    trade_log_path,
    validate_trade_log,
    load_trade_log,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
LA = ZoneInfo("America/Los_Angeles")
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_147_protocol101_morning_session")
DEFAULT_LIVE_SHADOW_ROOT = Path("v4/logs/live_shadow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("no-order-shadow", "paper"), default="no-order-shadow")
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--live-shadow-root", type=Path, default=DEFAULT_LIVE_SHADOW_ROOT)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-auto-ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=147)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--cycle-seconds", type=float, default=60.0)
    parser.add_argument("--capture-seconds", type=float, default=45.0)
    parser.add_argument("--max-cycles", type=int, default=390)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--skip-market-clock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = args.session_date or datetime.now(tz=LA).date().isoformat()
    run_id = args.run_id or f"protocol101_{args.mode}_{session}"
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_log = trade_log_path(root=args.trade_log_root, session=session, run_id=run_id)
    csv_log = trade_log.with_suffix(".csv")
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    append_session_event(
        trade_log,
        event_type="heartbeat",
        session=session,
        run_id=run_id,
        mode=args.mode,
        paper_cash=args.paper_cash,
        reason="session_started",
        extra={"session_config": public_config(args), "orders_fail_closed": True},
    )
    commands: list[dict[str, Any]] = []
    decision = "completed_no_order_analysis"
    cycles_run = 0
    live_capture_passes = 0
    live_capture_blocks = 0

    if args.mode == "paper":
        append_session_event(
            trade_log,
            event_type="paper_order_blocked",
            session=session,
            run_id=run_id,
            mode=args.mode,
            paper_cash=args.paper_cash,
            reason="paper_mode_requested_but_session_runner_starts_fail_closed",
            extra={
                "required_before_paper_orders": [
                    "Protocol101 live shadow parity passes on fresh SPX/VIX/SPXW rows",
                    "paper-order executor is explicitly enabled",
                    "V4_ALLOW_IBKR_PAPER_ORDERS=YES",
                ]
            },
        )

    if args.dry_run:
        decision = "dry_run_logged_no_commands"
    else:
        wait_until_market_open_if_needed(args)
        preflight = run_preflight(args, out_dir)
        commands.append(preflight)
        if preflight["returncode"] != 0:
            decision = "blocked_preflight_failed"
            append_session_event(
                trade_log,
                event_type="paper_error",
                session=session,
                run_id=run_id,
                mode=args.mode,
                paper_cash=args.paper_cash,
                reason="preflight_failed",
                extra={"command": redacted_command_result(preflight)},
            )

        for cycle in range(max(0, int(args.max_cycles))):
            if not args.skip_market_clock and not market_is_open():
                break
            cycles_run += 1
            cycle_dir = out_dir / f"cycle_{cycle:04d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            readiness = run_module(
                "v4.scripts.run_protocol119_protocol101_live_readiness",
                ["--out-dir", str(cycle_dir / "protocol119"), "--no-ledger"],
                out_dir=cycle_dir,
            )
            parity = run_module(
                "v4.scripts.run_protocol124_protocol101_live_data_parity_checkpoint",
                ["--out-dir", str(cycle_dir / "protocol124"), "--no-ledger"],
                out_dir=cycle_dir,
            )
            commands.extend([readiness, parity])
            parity_summary = load_json(cycle_dir / "protocol124" / "summary.json")
            parity_decision = str(parity_summary.get("decision") or "")
            append_session_event(
                trade_log,
                event_type="risk_gate",
                session=session,
                run_id=run_id,
                mode=args.mode,
                paper_cash=args.paper_cash,
                reason=parity_decision or "parity_unknown",
                extra={
                    "protocol119": command_brief(readiness),
                    "protocol124": command_brief(parity),
                    "required_actions": parity_summary.get("required_before_protocol101_live_shadow", []),
                },
            )
            if parity_decision == "ready_for_protocol101_no_order_live_capture":
                capture = run_live_capture(args, cycle_dir)
                commands.append(capture)
                capture_summary = load_json(cycle_dir / "live_capture" / "ibkr-live-capture_summary.json")
                capture_decision = str(capture_summary.get("decision") or "unknown")
                live_capture_passes += int(capture_decision == "pass")
                live_capture_blocks += int(capture_decision != "pass")
                append_session_event(
                    trade_log,
                    event_type="model_decision",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason=f"live_capture_{capture_decision}",
                    extra={
                        "live_capture": command_brief(capture),
                        "captured_rows": capture_summary.get("captured_rows", 0),
                        "shadow_log": capture_summary.get("shadow_log"),
                        "blocked_reason": capture_summary.get("blocked_reason"),
                    },
                )
            else:
                live_capture_blocks += 1
                append_session_event(
                    trade_log,
                    event_type="paper_order_blocked",
                    session=session,
                    run_id=run_id,
                    mode=args.mode,
                    paper_cash=args.paper_cash,
                    reason=parity_decision or "live_parity_not_ready",
                    extra={"paper_order_submission": "disabled_until_live_shadow_parity_passes"},
                )
            if args.max_cycles <= 1:
                break
            time.sleep(max(0.0, float(args.cycle_seconds)))

    rows = load_trade_log(trade_log)
    validation = validate_trade_log(rows)
    csv_summary = export_trade_log_csv(trade_log, csv_log) if trade_log.exists() else {"rows": 0}
    payload = {
        "protocol": "147_protocol101_morning_session",
        "decision": decision,
        "mode": args.mode,
        "session": session,
        "run_id": run_id,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": False,
        "real_money_trading": False,
        "broker_order_endpoint_called": False,
        "cycles_run": cycles_run,
        "live_capture_passes": live_capture_passes,
        "live_capture_blocks": live_capture_blocks,
        "trade_log": {
            "jsonl": str(trade_log),
            "csv": str(csv_log),
            "validation": validation,
            "csv_summary": csv_summary,
        },
        "command_count": len(commands),
        "commands": [command_brief(item) for item in commands[-20:]],
        "next_gate": next_gate(decision, live_capture_passes),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(report_path, payload)
    print(json.dumps({"decision": decision, "report": str(report_path), "trade_log": str(trade_log)}, indent=2))
    return 0 if validation["status"] == "pass" else 1


def wait_until_market_open_if_needed(args: argparse.Namespace) -> None:
    if args.skip_market_clock:
        return
    now = datetime.now(tz=NY)
    open_dt = datetime.combine(now.date(), wall_time(9, 30), tzinfo=NY)
    if now < open_dt:
        time.sleep(min(max((open_dt - now).total_seconds(), 0.0), max(float(args.cycle_seconds), 1.0)))


def market_is_open() -> bool:
    now = datetime.now(tz=NY)
    if now.weekday() >= 5:
        return False
    return wall_time(9, 30) <= now.time() <= wall_time(16, 0)


def run_preflight(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    return run_command(
        [
            os.environ.get("PYTHON_BIN", "/usr/bin/python3"),
            str(REPO_ROOT / "v4/ops/ibkr/wait_for_ibkr_api.py"),
            "--host",
            args.ibkr_host,
            "--port",
            str(args.ibkr_port),
            "--auto-ports",
            args.ibkr_auto_ports,
            "--timeout-seconds",
            str(args.preflight_timeout_seconds),
            "--run-entitlement-probe",
            "--repo-root",
            str(REPO_ROOT),
        ],
        out_dir=out_dir,
        name="preflight",
    )


def run_live_capture(args: argparse.Namespace, cycle_dir: Path) -> dict[str, Any]:
    return run_module(
        "v4.scripts.run_protocol081_live_shadow_router",
        [
            "--mode",
            "ibkr-live-capture",
            "--out-dir",
            str(cycle_dir / "live_capture"),
            "--ibkr-host",
            args.ibkr_host,
            "--ibkr-port",
            str(args.ibkr_port),
            "--ibkr-auto-ports",
            args.ibkr_auto_ports,
            "--ibkr-client-id",
            str(args.ibkr_client_id),
            "--live-capture-seconds",
            str(args.capture_seconds),
            "--min-live-shadow-rows",
            "1",
        ],
        out_dir=cycle_dir,
    )


def run_module(module: str, extra_args: list[str], *, out_dir: Path) -> dict[str, Any]:
    return run_command(
        [os.environ.get("PYTHON_BIN", "/usr/bin/python3"), "-m", module, *extra_args],
        out_dir=out_dir,
        name=module.rsplit(".", 1)[-1],
    )


def run_command(cmd: list[str], *, out_dir: Path, name: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    stdout_path = out_dir / f"{name}.out.log"
    stderr_path = out_dir / f"{name}.err.log"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    return {
        "name": name,
        "cmd": public_cmd(cmd),
        "returncode": result.returncode,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def append_session_event(
    path: Path,
    *,
    event_type: str,
    session: str,
    run_id: str,
    mode: str,
    paper_cash: float,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    row = make_trade_log_event(
        event_type=event_type,
        session=session,
        run_id=run_id,
        mode=mode,
        paper_trading=True,
        real_money_trading=False,
        broker_order_endpoint_called=False,
        account={
            "account_id_redacted": None,
            "starting_cash": float(paper_cash),
            "cash": float(paper_cash),
            "equity": float(paper_cash),
            "realized_daily_pnl": 0.0,
            "open_positions": 0,
        },
        market_snapshot={
            "underlying": {},
            "option_nbbo": {},
            "context": {},
        },
        model_decision={"action": "blocked" if "blocked" in reason or "failed" in reason else "wait", "reason": reason},
        risk_gate={"passed": not ("blocked" in reason or "failed" in reason), "reason": reason},
        extra=extra,
    )
    append_trade_event(path, row)


def public_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "ibkr_host": args.ibkr_host,
        "ibkr_port": args.ibkr_port,
        "ibkr_auto_ports": args.ibkr_auto_ports,
        "paper_cash": args.paper_cash,
        "cycle_seconds": args.cycle_seconds,
        "capture_seconds": args.capture_seconds,
        "max_cycles": args.max_cycles,
    }


def command_brief(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item.get("name"),
        "returncode": item.get("returncode"),
        "stdout_log": item.get("stdout_log"),
        "stderr_log": item.get("stderr_log"),
    }


def redacted_command_result(item: dict[str, Any]) -> dict[str, Any]:
    return command_brief(item)


def public_cmd(cmd: list[str]) -> list[str]:
    return [part if "key" not in part.lower() and "password" not in part.lower() else "<redacted>" for part in cmd]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def next_gate(decision: str, live_capture_passes: int) -> str:
    if live_capture_passes > 0:
        return "Review the live shadow rows and trade log, then run order-state rehearsal before enabling paper orders."
    if decision.startswith("blocked_"):
        return "Fix preflight/live-data blockers; the session will keep logging blocked paper-order state until parity passes."
    return "Use the generated JSONL/CSV trade logs to diagnose why live parity did or did not clear."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 147: Protocol101 Morning Session",
        "",
        "This is the unattended morning session runner. It starts fail-closed: no real-money trading and no broker order endpoint.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Session: `{payload['session']}`",
        f"- Run ID: `{payload['run_id']}`",
        f"- Cycles run: `{payload['cycles_run']}`",
        f"- Live capture passes: `{payload['live_capture_passes']}`",
        f"- Live capture blocks: `{payload['live_capture_blocks']}`",
        f"- Trade log JSONL: `{payload['trade_log']['jsonl']}`",
        f"- Trade log CSV: `{payload['trade_log']['csv']}`",
        f"- Trade log validation: `{payload['trade_log']['validation']['status']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
