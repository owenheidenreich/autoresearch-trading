"""Dispatch the daily paper autopilot to the current paper-approved model."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

from v4.live.paper_model_registry import (
    DEFAULT_PAPER_TRADING_DEFAULT,
    PaperModelSelection,
    load_paper_model_selection,
    selection_summary,
)


NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-default-registry", type=Path, default=DEFAULT_PAPER_TRADING_DEFAULT)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--print-selection", action="store_true")
    parser.add_argument("--mode", choices=("intent-shadow", "paper-dry-run", "paper-submit"), default="paper-submit")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--trade-log-root", type=Path, default=None)
    parser.add_argument("--runtime-state", type=Path, default=None)
    parser.add_argument("--live-index-context-log", type=Path, default=None)
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-auto-ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=160)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--paper-cash", type=float, default=10_000.0)
    parser.add_argument("--open-positions", type=int, default=0)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--order-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--market-close-time", default="16:00")
    parser.add_argument("--min-edge", type=float, default=25.0)
    parser.add_argument("--live-strikes-around-atm", type=int, default=10)
    parser.add_argument("--quote-loop-seconds", type=float, default=1.0)
    parser.add_argument("--entry-decision-interval-seconds", type=float, default=60.0)
    parser.add_argument("--entry-decision-mode", choices=("minute", "interval"), default="minute")
    parser.add_argument("--decision-interval-seconds", type=float, default=None)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--contract-refresh-seconds", type=float, default=60.0)
    parser.add_argument("--refresh-contracts-drift-points", type=float, default=15.0)
    parser.add_argument("--min-live-context-minutes", type=float, default=30.0)
    parser.add_argument("--reconnect-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-reconnects", type=int, default=200)
    parser.add_argument("--max-decisions", type=int, default=100_000)
    parser.add_argument("--allow-delayed-market-data", action="store_true")
    parser.add_argument("--enable-paper-orders", action="store_true")
    parser.add_argument("--acknowledge-paper-loss", action="store_true")
    parser.add_argument("--skip-market-clock", action="store_true")
    parser.add_argument("--wait-for-market-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = load_paper_model_selection(args.paper_default_registry, model_id=args.model_id)
    session = str(args.session_date or datetime.now(tz=NY).date().isoformat())
    run_id = str(args.run_id or selection.default_run_id(session))
    summary = selection_summary(selection)
    summary.update({"session": session, "run_id": run_id})
    if args.print_selection:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    child_args = build_child_args(selection, args=args, session=session, run_id=run_id)
    command = [sys.executable, "-m", selection.entrypoint, *child_args]
    return subprocess.run(command, check=False).returncode


def build_child_args(
    selection: PaperModelSelection,
    *,
    args: argparse.Namespace,
    session: str,
    run_id: str,
) -> list[str]:
    if selection.runner_kind != "protocol101_persistent":
        raise ValueError(f"unsupported runner kind: {selection.runner_kind}")

    model_args = selection.arguments
    out: list[str] = [
        "--mode",
        str(args.mode),
        "--session-date",
        session,
        "--run-id",
        run_id,
        "--surface-manifest",
        model_args["surface_manifest"],
        "--protocol101-manifest",
        model_args["protocol101_manifest"],
        "--protocol101-summary",
        model_args["protocol101_summary"],
        "--runtime-flag",
        model_args["runtime_flag"],
        "--runtime-state",
        str(getattr(args, "runtime_state", None) or model_args["runtime_state"]),
        "--live-index-context-log",
        str(getattr(args, "live_index_context_log", None) or model_args["live_index_context_log"]),
        "--ibkr-host",
        str(args.ibkr_host),
        "--ibkr-port",
        str(args.ibkr_port),
        "--ibkr-auto-ports",
        str(args.ibkr_auto_ports),
        "--ibkr-client-id",
        str(args.ibkr_client_id),
        "--paper-cash",
        str(args.paper_cash),
        "--open-positions",
        str(args.open_positions),
        "--quantity",
        str(args.quantity),
        "--order-timeout-seconds",
        str(args.order_timeout_seconds),
        "--forced-flat-time",
        str(args.forced_flat_time),
        "--market-close-time",
        str(args.market_close_time),
        "--min-edge",
        str(args.min_edge),
        "--live-strikes-around-atm",
        str(args.live_strikes_around_atm),
        "--quote-loop-seconds",
        str(args.quote_loop_seconds),
        "--entry-decision-interval-seconds",
        str(args.entry_decision_interval_seconds),
        "--entry-decision-mode",
        str(args.entry_decision_mode),
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
        "--contract-refresh-seconds",
        str(args.contract_refresh_seconds),
        "--refresh-contracts-drift-points",
        str(args.refresh_contracts_drift_points),
        "--min-live-context-minutes",
        str(args.min_live_context_minutes),
        "--reconnect-sleep-seconds",
        str(args.reconnect_sleep_seconds),
        "--max-reconnects",
        str(args.max_reconnects),
        "--max-decisions",
        str(args.max_decisions),
    ]
    if args.out_root is not None:
        out.extend(["--out-root", str(args.out_root)])
    if args.trade_log_root is not None:
        out.extend(["--trade-log-root", str(args.trade_log_root)])
    if args.account_id:
        out.extend(["--account-id", str(args.account_id)])
    if args.decision_interval_seconds is not None:
        out.extend(["--decision-interval-seconds", str(args.decision_interval_seconds)])
    if args.allow_delayed_market_data:
        out.append("--allow-delayed-market-data")
    if args.enable_paper_orders:
        out.append("--enable-paper-orders")
    if args.acknowledge_paper_loss:
        out.append("--acknowledge-paper-loss")
    if args.skip_market_clock:
        out.append("--skip-market-clock")
    if getattr(args, "wait_for_market_open", False):
        out.append("--wait-for-market-open")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
