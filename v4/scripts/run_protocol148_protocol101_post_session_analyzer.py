"""Protocol 148: post-session analyzer for Protocol101 live/paper logs.

The analyzer turns one append-only session JSONL into a plain evidence report:
startup, connection/parity state, blocked reasons, order activity, and any
paper-fill PnL that can be reconstructed from the logged events.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from v4.live.paper_trade_log import DEFAULT_TRADE_LOG_ROOT, load_trade_log, validate_trade_log


DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_148_protocol101_post_session_analyzer")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session", default=None)
    parser.add_argument("--run-id", default=None)
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
    out_dir = args.out_root / str(analysis["session"]) / str(analysis["run_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "148_protocol101_post_session_analyzer",
        "decision": decide(analysis),
        "paid_data_downloaded": False,
        "live_orders": False,
        "real_money_trading": False,
        "source_trade_log": str(trade_log),
        "analysis": analysis,
        "outputs": {
            "summary": str(out_dir / "summary.json"),
            "report": str(out_dir / "report.md"),
        },
        "next_gate": next_gate(analysis),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(out_dir / "report.md")}, indent=2))
    return 0


def resolve_trade_log(
    *,
    trade_log: Path | None,
    root: Path,
    session: str | None,
    run_id: str | None,
) -> Path:
    if trade_log is not None:
        return trade_log
    if session and run_id:
        return root / session / f"{run_id}.jsonl"
    candidates = sorted(root.glob("*/*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no session logs found under {root}")
    if session:
        candidates = [path for path in candidates if path.parent.name == session]
    if run_id:
        candidates = [path for path in candidates if path.stem == run_id]
    if not candidates:
        raise FileNotFoundError(f"no matching session logs found under {root}")
    return candidates[-1]


def analyze_session_rows(rows: list[dict[str, Any]], *, trade_log: Path) -> dict[str, Any]:
    validation = validate_trade_log(rows)
    session = first_value(rows, "session") or trade_log.parent.name
    run_id = first_value(rows, "run_id") or trade_log.stem
    event_counts = Counter(str(row.get("event_type", "missing")) for row in rows)
    risk_reasons = Counter(
        str((_object(row.get("risk_gate")).get("reason") or "missing"))
        for row in rows
        if str(row.get("event_type")) in {"risk_gate", "paper_order_blocked", "paper_error"}
    )
    model_actions = Counter(
        str((_object(row.get("model_decision")).get("action") or "missing"))
        for row in rows
        if str(row.get("event_type")) in {"model_decision", "risk_gate", "paper_order_blocked", "paper_error", "heartbeat"}
    )
    broker_rows = [row for row in rows if bool(row.get("broker_order_endpoint_called"))]
    paper_order_rows = [
        row
        for row in rows
        if str(row.get("event_type", "")).startswith("paper_order")
        or str(row.get("event_type", "")).startswith("paper_entry")
        or str(row.get("event_type", "")).startswith("paper_exit")
    ]
    fill_summary = reconstruct_paper_fill_pnl(rows)
    command_refs = collect_command_refs(rows)
    command_signals = summarize_command_logs(command_refs)
    required_actions = sorted(
        {
            str(item)
            for row in rows
            for item in (_list(row.get("required_actions")) or _list(row.get("required_before_protocol101_live_shadow")))
            if item
        }
    )
    linked_shadow_logs = sorted(
        {
            str(row.get("shadow_log"))
            for row in rows
            if row.get("shadow_log")
        }
    )
    return {
        "session": session,
        "run_id": run_id,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "trade_log": str(trade_log),
        "rows": len(rows),
        "validation": validation,
        "event_counts": dict(sorted(event_counts.items())),
        "model_action_counts": dict(sorted(model_actions.items())),
        "risk_reason_counts": dict(risk_reasons.most_common()),
        "broker_order_endpoint_called_rows": len(broker_rows),
        "paper_order_event_rows": len(paper_order_rows),
        "paper_orders_submitted": int(event_counts.get("paper_order_submitted", 0)),
        "paper_entry_fills": int(event_counts.get("paper_entry_fill", 0)),
        "paper_exit_fills": int(event_counts.get("paper_exit_fill", 0)),
        "paper_fill_pnl": fill_summary,
        "startup_and_data": command_signals,
        "required_actions": required_actions,
        "linked_shadow_logs": linked_shadow_logs,
        "headline": headline(validation, command_signals, fill_summary, broker_rows),
    }


def summarize_command_logs(command_refs: list[dict[str, Any]]) -> dict[str, Any]:
    preflight_returncodes: list[int] = []
    readiness_returncodes: list[int] = []
    parity_returncodes: list[int] = []
    port_open = False
    ibkr_connected = False
    live_parity_ready = False
    live_capture_pass = False
    blocked_reasons: Counter[str] = Counter()
    parsed_json_rows = 0
    for ref in command_refs:
        name = str(ref.get("name") or "")
        returncode = _int(ref.get("returncode"))
        if name == "preflight" and returncode is not None:
            preflight_returncodes.append(returncode)
        if "protocol119" in name and returncode is not None:
            readiness_returncodes.append(returncode)
        if "protocol124" in name and returncode is not None:
            parity_returncodes.append(returncode)
        for key in ("stdout_log", "stderr_log"):
            raw_path = str(ref.get(key) or "")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.exists():
                continue
            for row in parse_json_lines(path):
                parsed_json_rows += 1
                port_open = port_open or row.get("status") == "port_open"
                ibkr_connected = ibkr_connected or bool(row.get("ibkr_connected"))
                if row.get("decision") == "ready_for_protocol101_no_order_live_capture":
                    live_parity_ready = True
                if row.get("decision") == "pass":
                    live_capture_pass = True
                reason = row.get("blocked_reason") or row.get("decision")
                if reason and str(reason).startswith(("blocked_", "missing_", "ibkr_")):
                    blocked_reasons[str(reason)] += 1
    return {
        "command_refs": len(command_refs),
        "parsed_json_rows": parsed_json_rows,
        "gateway_or_api_port_open": port_open,
        "ibkr_connected": ibkr_connected,
        "preflight_returncodes": preflight_returncodes,
        "readiness_returncodes": readiness_returncodes,
        "parity_returncodes": parity_returncodes,
        "live_parity_ready": live_parity_ready,
        "live_capture_pass": live_capture_pass,
        "blocked_reason_counts": dict(blocked_reasons.most_common()),
    }


def reconstruct_paper_fill_pnl(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("trade_uid"):
            grouped[str(row["trade_uid"])].append(row)
    closed = []
    open_count = 0
    for trade_uid, trade_rows in grouped.items():
        buys = [row for row in trade_rows if row.get("event_type") == "paper_entry_fill"]
        sells = [row for row in trade_rows if row.get("event_type") == "paper_exit_fill"]
        if not buys:
            continue
        if not sells:
            open_count += 1
            continue
        buy = buys[-1]
        sell = sells[-1]
        qty = _number(_object(buy.get("order")).get("filled") or _object(buy.get("order")).get("quantity")) or 1.0
        entry = _number(_object(buy.get("order")).get("avg_fill_price") or _object(buy.get("order")).get("limit_price"))
        exit_ = _number(_object(sell.get("order")).get("avg_fill_price") or _object(sell.get("order")).get("limit_price"))
        if entry is None or exit_ is None:
            continue
        pnl = (exit_ - entry) * CONTRACT_MULTIPLIER * qty
        closed.append({"trade_uid": trade_uid, "quantity": qty, "entry": entry, "exit": exit_, "pnl": pnl})
    total = sum(row["pnl"] for row in closed)
    return {
        "available": bool(closed),
        "closed_trades": len(closed),
        "open_trades": open_count,
        "total_pnl": round(float(total), 6),
        "wins": sum(1 for row in closed if row["pnl"] > 0),
        "losses": sum(1 for row in closed if row["pnl"] < 0),
        "sample": closed[:10],
    }


def collect_command_refs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for row in rows:
        for key in ("command", "protocol119", "protocol124", "live_capture"):
            value = row.get(key)
            if isinstance(value, dict):
                refs.append(value)
    return refs


def parse_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def decide(analysis: dict[str, Any]) -> str:
    if analysis["validation"]["status"] != "pass":
        return "blocked_session_log_validation_failed"
    if analysis["broker_order_endpoint_called_rows"] > 0 and analysis["paper_orders_submitted"] <= 0:
        return "blocked_unexplained_broker_endpoint_call"
    if analysis["startup_and_data"]["live_capture_pass"]:
        return "pass_live_shadow_capture_available_for_review"
    if analysis["startup_and_data"]["gateway_or_api_port_open"]:
        return "blocked_live_data_or_parity_not_ready"
    return "blocked_gateway_or_api_not_confirmed"


def headline(
    validation: dict[str, Any],
    command_signals: dict[str, Any],
    fill_summary: dict[str, Any],
    broker_rows: list[dict[str, Any]],
) -> str:
    if validation.get("status") != "pass":
        return "Session log failed schema validation."
    if broker_rows:
        return "Broker endpoint rows are present; inspect order events before trusting the run."
    if fill_summary["available"]:
        return f"Paper fills were logged; reconstructed PnL is ${fill_summary['total_pnl']:.2f}."
    if command_signals["live_capture_pass"]:
        return "Live shadow capture passed; review model decisions and order-state rehearsal before paper orders."
    if command_signals["gateway_or_api_port_open"]:
        return "Gateway/API appears reachable, but live-data parity or capture did not pass."
    return "Session log is valid, but Gateway/API was not confirmed from the captured logs."


def next_gate(analysis: dict[str, Any]) -> str:
    if analysis["paper_fill_pnl"]["available"]:
        return "Review fills, slippage, and realized PnL against intended model decisions."
    if analysis["startup_and_data"]["live_capture_pass"]:
        return "Run order-state rehearsal on the captured live shadow rows, then decide whether paper-submit mode is allowed."
    return "Fix the listed startup/live-data blockers, then rerun the morning session."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    analysis = payload["analysis"]
    startup = analysis["startup_and_data"]
    pnl = analysis["paper_fill_pnl"]
    lines = [
        "# Protocol 148: Protocol101 Post-Session Analyzer",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Session: `{analysis['session']}`",
        f"- Run ID: `{analysis['run_id']}`",
        f"- Trade log: `{analysis['trade_log']}`",
        f"- Headline: {analysis['headline']}",
        "",
        "## Session Health",
        "",
        f"- Log validation: `{analysis['validation']['status']}`",
        f"- Rows: `{analysis['rows']}`",
        f"- Gateway/API port open: `{startup['gateway_or_api_port_open']}`",
        f"- IBKR connected signal: `{startup['ibkr_connected']}`",
        f"- Live parity ready: `{startup['live_parity_ready']}`",
        f"- Live capture pass: `{startup['live_capture_pass']}`",
        f"- Broker endpoint rows: `{analysis['broker_order_endpoint_called_rows']}`",
        "",
        "## Event Counts",
        "",
        "```json",
        json.dumps(analysis["event_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Blocked Reasons",
        "",
        "```json",
        json.dumps(analysis["risk_reason_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Paper PnL",
        "",
        f"- Available: `{pnl['available']}`",
        f"- Closed trades: `{pnl['closed_trades']}`",
        f"- Open trades: `{pnl['open_trades']}`",
        f"- Total PnL: `${pnl['total_pnl']:.2f}`",
        "",
        "## Required Actions",
        "",
    ]
    if analysis["required_actions"]:
        lines.extend(f"- {item}" for item in analysis["required_actions"])
    else:
        lines.append("- None captured in this log.")
    lines.extend(["", "## Next Gate", "", payload["next_gate"]])
    path.write_text("\n".join(lines) + "\n")


def first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


if __name__ == "__main__":
    raise SystemExit(main())
