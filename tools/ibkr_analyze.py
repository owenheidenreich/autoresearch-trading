#!/usr/bin/env python3
"""
IBKR Session Analyzer — parse audit.jsonl and produce structured metrics.

Extracts per-trade details, computes aggregate metrics (PF, win rate, stop rate),
and outputs JSON compatible with ART² briefing integration.

Usage:
  python3 tools/ibkr_analyze.py                                 # default audit path
  python3 tools/ibkr_analyze.py --audit results/live/audit.jsonl
  python3 tools/ibkr_analyze.py --output results/ibkr_sessions/2026-03-23.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT = PROJECT_ROOT / "results" / "live" / "audit.jsonl"
SESSIONS_DIR = PROJECT_ROOT / "results" / "ibkr_sessions"

ACTION_NAMES = {
    0: "DO_NOTHING", 1: "BUY_CALL_ATM", 2: "BUY_CALL_OTM5",
    3: "BUY_CALL_OTM10", 4: "BUY_PUT_ATM", 5: "BUY_PUT_OTM5",
    6: "BUY_PUT_OTM10", 7: "EXIT",
}


def parse_audit(path: Path) -> dict[str, Any]:
    """Parse audit.jsonl into sessions, trades, and inferences."""
    if not path.exists():
        return {"error": f"No audit file at {path}", "sessions": [], "trades": []}

    sessions: list[dict] = []
    trades: list[dict] = []          # closed positions
    entries: dict[str, dict] = {}    # intent_id → entry data
    dry_runs: dict[str, dict] = {}   # intent_id → dry_run state
    inferences: list[dict] = []
    current_session: dict | None = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = d.get("ts", "")
            event = d.get("event", "")
            p = d.get("payload", d)

            if event == "session_start":
                current_session = {
                    "session_id": p.get("session_id", ""),
                    "ts": ts,
                    "seed_spx": p.get("seed_spx", 0),
                    "dry_run": p.get("dry_run", False),
                    "account": p.get("account", ""),
                }
                sessions.append(current_session)

            elif event == "session_end":
                sid = p.get("session_id", "")
                for s in sessions:
                    if s["session_id"] == sid:
                        s["end_ts"] = ts
                        s["processed_minutes"] = p.get("processed_minutes", 0)
                        s["signals_generated"] = p.get("signals_generated", 0)
                        s["entries_applied"] = p.get("entries_applied", 0)
                        s["exits_applied"] = p.get("exits_applied", 0)
                        s["open_positions_final"] = p.get("open_positions_final", 0)

            elif event == "entry_intent":
                intent_id = p.get("intent_id", "")
                contract = p.get("contract", {})
                action_id = p.get("action", 0)
                entries[intent_id] = {
                    "ts": ts,
                    "session_id": p.get("session_id", ""),
                    "intent_id": intent_id,
                    "action": action_id,
                    "action_name": ACTION_NAMES.get(action_id, f"ACTION_{action_id}"),
                    "confidence": p.get("confidence", 0),
                    "stop_price": p.get("stop_price", 0),
                    "tp_price": p.get("take_profit_price", 0),
                    "ref_price": p.get("reference_price", 0),
                    "strike": contract.get("strike", 0),
                    "right": contract.get("right", ""),
                    "expiry": contract.get("lastTradeDateOrContractMonth", ""),
                    "symbol": contract.get("localSymbol", ""),
                    "applied": False,
                }

            elif event == "entry_intent_applied":
                intent_id = p.get("intent_id", "")
                if intent_id in entries:
                    entries[intent_id]["applied"] = True
                    entries[intent_id]["applied_ts"] = ts
                    entries[intent_id]["position_id"] = p.get("position_id", "")

            elif event == "entry_dry_run":
                intent_id = p.get("intent_id", "")
                state = p.get("state", {})
                meta = state.get("metadata", {})
                intent_data = p.get("intent", {})
                intent_meta = intent_data.get("metadata", {})
                gate_prob = meta.get("gate_trade_prob",
                                     intent_meta.get("gate_trade_prob", 0))
                dry_runs[intent_id] = {
                    "position_id": p.get("position_id", ""),
                    "fill_price": state.get("fill_price", 0),
                    "gate_trade_prob": gate_prob,
                    "direction_probs": meta.get("direction_probs",
                                                intent_meta.get("direction_probs", [])),
                    "entry_limit_price": intent_data.get("entry_limit_price",
                                                         meta.get("entry_limit_price", 0)),
                    "stop_price": state.get("current_stop", 0),
                    "tp_price": state.get("current_take_profit", 0),
                    "contract": state.get("contract", {}),
                    "status": state.get("status", ""),
                }
                if intent_id in entries:
                    entries[intent_id]["gate_trade_prob"] = gate_prob
                    entries[intent_id]["fill_price"] = state.get("fill_price", 0)
                    entries[intent_id]["position_id"] = p.get("position_id", "")

            elif event == "fill":
                # Real IBKR fill (not dry-run)
                pass  # fills enriched into entries/trades via position_closed

            elif event == "position_closed":
                trade = {
                    "ts": ts,
                    "intent_id": p.get("intent_id", ""),
                    "position_id": p.get("position_id", ""),
                    "pnl_pct": p.get("pnl_pct", 0),
                    "pnl_dollar": p.get("pnl_dollar", 0),
                    "bars_held": p.get("bars_held", 0),
                    "exit_reason": p.get("exit_reason", ""),
                    "direction": p.get("direction", ""),
                    "strike": p.get("strike", 0),
                    "entry_price": p.get("entry_price", 0),
                    "exit_price": p.get("exit_price", 0),
                }
                # Enrich from entry/dry_run
                iid = trade["intent_id"]
                entry = entries.get(iid, {})
                dr = dry_runs.get(iid, {})
                trade["action"] = entry.get("action", 0)
                trade["action_name"] = entry.get("action_name", "")
                trade["gate_trade_prob"] = entry.get("gate_trade_prob",
                                                     dr.get("gate_trade_prob", 0))
                trade["entry_ts"] = entry.get("ts", "")
                trade["symbol"] = entry.get("symbol", "")
                trades.append(trade)

            elif event == "model_inference":
                inferences.append({
                    "ts": ts,
                    "action": p.get("action", 0),
                    "gate_trade_prob": p.get("gate_trade_prob", 0),
                })

    # Detect open positions (entries applied but not closed)
    closed_intent_ids = {t["intent_id"] for t in trades}
    closed_position_ids = {t["position_id"] for t in trades}
    open_positions = []
    for iid, entry in entries.items():
        if entry.get("applied") and iid not in closed_intent_ids:
            pid = entry.get("position_id", "")
            if not pid or pid not in closed_position_ids:
                dr = dry_runs.get(iid, {})
                open_positions.append({
                    **entry,
                    "fill_price": entry.get("fill_price", dr.get("fill_price", 0)),
                    "gate_trade_prob": entry.get("gate_trade_prob",
                                                 dr.get("gate_trade_prob", 0)),
                    "stop_price": dr.get("stop_price", entry.get("stop_price", 0)),
                    "tp_price": dr.get("tp_price", entry.get("tp_price", 0)),
                })

    return {
        "sessions": sessions,
        "trades": trades,
        "open_positions": open_positions,
        "inferences": inferences,
        "entries": entries,
        "dry_runs": dry_runs,
    }


def compute_metrics(parsed: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate trading metrics from parsed audit data."""
    trades = parsed["trades"]
    sessions = parsed["sessions"]
    inferences = parsed["inferences"]
    open_positions = parsed["open_positions"]

    # Session summary
    total_minutes = sum(s.get("processed_minutes", 0) for s in sessions)
    total_signals = sum(s.get("signals_generated", 0) for s in sessions)
    num_sessions = len(sessions)
    session_dates = sorted(set(
        s.get("ts", "")[:10] for s in sessions if s.get("ts")
    ))

    # Trade metrics
    num_closed = len(trades)
    num_open = len(open_positions)
    if num_closed == 0:
        return {
            "session_summary": {
                "num_sessions": num_sessions,
                "total_minutes": total_minutes,
                "total_signals": total_signals,
                "dates": session_dates,
            },
            "trade_summary": {
                "num_closed": 0,
                "num_open": num_open,
                "open_positions": [_fmt_open(op) for op in open_positions],
            },
            "inference_summary": _inference_summary(inferences),
            "verdict": "INSUFFICIENT_DATA",
            "verdict_reason": f"No closed trades ({num_open} open)",
        }

    pnls = [t["pnl_pct"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(winners) / num_closed

    # Exit breakdown
    exit_reasons: dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    stop_rate = exit_reasons.get("stop_loss", 0) / num_closed

    # Direction breakdown
    directions: dict[str, int] = {}
    for t in trades:
        d = t.get("direction", t.get("action_name", "unknown"))
        directions[d] = directions.get(d, 0) + 1

    # Time-of-day breakdown
    tod_buckets: dict[str, list[float]] = {}
    for t in trades:
        hour = _extract_hour(t.get("entry_ts", t.get("ts", "")))
        bucket = _tod_bucket(hour)
        tod_buckets.setdefault(bucket, []).append(t["pnl_pct"])
    tod_summary = {
        k: {"trades": len(v), "total_pnl_pct": round(sum(v), 4),
             "avg_pnl_pct": round(sum(v) / len(v), 4)}
        for k, v in sorted(tod_buckets.items())
    }

    # Per-trade detail
    trade_details = []
    for t in trades:
        trade_details.append({
            "entry_ts": t.get("entry_ts", ""),
            "exit_ts": t.get("ts", ""),
            "action": t.get("action_name", ""),
            "strike": t.get("strike", 0),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "pnl_pct": round(t["pnl_pct"], 4),
            "pnl_dollar": round(t.get("pnl_dollar", 0), 2),
            "bars_held": t.get("bars_held", 0),
            "exit_reason": t.get("exit_reason", ""),
            "gate_trade_prob": round(t.get("gate_trade_prob", 0), 4),
        })

    return {
        "session_summary": {
            "num_sessions": num_sessions,
            "total_minutes": total_minutes,
            "total_signals": total_signals,
            "dates": session_dates,
        },
        "trade_summary": {
            "num_closed": num_closed,
            "num_open": num_open,
            "pf": round(pf, 4),
            "win_rate": round(win_rate, 4),
            "total_pnl_pct": round(sum(pnls), 4),
            "avg_pnl_pct": round(sum(pnls) / num_closed, 4),
            "avg_winner_pct": round(sum(winners) / len(winners), 4) if winners else 0,
            "avg_loser_pct": round(sum(losers) / len(losers), 4) if losers else 0,
            "stop_rate": round(stop_rate, 4),
            "avg_bars_held": round(sum(t.get("bars_held", 0) for t in trades) / num_closed, 1),
            "exit_breakdown": exit_reasons,
            "direction_breakdown": directions,
            "open_positions": [_fmt_open(op) for op in open_positions],
        },
        "time_of_day": tod_summary,
        "trade_details": trade_details,
        "inference_summary": _inference_summary(inferences),
        "verdict": _compute_verdict(pf, win_rate, num_closed, stop_rate),
        "verdict_reason": _verdict_reason(pf, win_rate, num_closed, stop_rate),
    }


def _fmt_open(op: dict) -> dict:
    return {
        "action": op.get("action_name", ACTION_NAMES.get(op.get("action", 0), "?")),
        "entry_ts": op.get("ts", ""),
        "fill_price": op.get("fill_price", 0),
        "gate_trade_prob": round(op.get("gate_trade_prob", 0), 4),
        "strike": op.get("strike", 0),
        "stop_price": round(op.get("stop_price", 0), 2),
    }


def _inference_summary(inferences: list[dict]) -> dict:
    if not inferences:
        return {"total_bars": 0, "trade_signals": 0, "signal_rate": 0}
    trade_signals = sum(1 for i in inferences if i.get("gate_trade_prob", 0) > 0.5)
    return {
        "total_bars": len(inferences),
        "trade_signals": trade_signals,
        "signal_rate": round(trade_signals / len(inferences), 4),
    }


def _extract_hour(ts_str: str) -> int:
    """Extract hour from ISO timestamp (UTC). Returns -1 on failure."""
    try:
        return int(ts_str[11:13])
    except (ValueError, IndexError):
        return -1


def _tod_bucket(hour_utc: int) -> str:
    """Map UTC hour to ET market time bucket."""
    # UTC to ET is -4 (EDT) or -5 (EST). Approximate with -4.
    hour_et = (hour_utc - 4) % 24
    if hour_et < 10:
        return "09:30-10:00"
    elif hour_et < 11:
        return "10:00-11:00"
    elif hour_et < 12:
        return "11:00-12:00"
    elif hour_et < 14:
        return "12:00-14:00"
    elif hour_et < 15:
        return "14:00-15:00"
    else:
        return "15:00-16:00"


def _compute_verdict(pf: float, win_rate: float, num_trades: int,
                     stop_rate: float) -> str:
    if num_trades < 5:
        return "INSUFFICIENT_DATA"
    if pf >= 1.5 and win_rate >= 0.45 and stop_rate < 0.5:
        return "VIABLE"
    if pf >= 1.0 and win_rate >= 0.35:
        return "PROMISING"
    if pf >= 0.7:
        return "INCONCLUSIVE"
    return "NOT_VIABLE"


def _verdict_reason(pf: float, win_rate: float, num_trades: int,
                    stop_rate: float) -> str:
    parts = []
    if num_trades < 5:
        parts.append(f"Only {num_trades} closed trades — need ≥5 for assessment")
    else:
        parts.append(f"PF={pf:.2f}, WR={win_rate:.0%}, stop_rate={stop_rate:.0%}, N={num_trades}")
    return "; ".join(parts)


def format_report(metrics: dict[str, Any]) -> str:
    """Format metrics as human-readable markdown report."""
    lines = ["# IBKR Paper Trading Session Report\n"]

    ss = metrics["session_summary"]
    lines.append(f"**Sessions:** {ss['num_sessions']} | "
                 f"**Total minutes:** {ss['total_minutes']} | "
                 f"**Signals:** {ss['total_signals']}")
    if ss.get("dates"):
        lines.append(f"**Dates:** {', '.join(ss['dates'])}")
    lines.append("")

    ts = metrics["trade_summary"]
    lines.append(f"## Trade Summary")
    lines.append(f"- Closed: {ts['num_closed']} | Open: {ts['num_open']}")
    if ts["num_closed"] > 0:
        lines.append(f"- PF: {ts['pf']:.2f} | Win rate: {ts['win_rate']:.0%}")
        lines.append(f"- Total P&L: {ts['total_pnl_pct']:.2%} | "
                     f"Avg: {ts['avg_pnl_pct']:.2%}")
        lines.append(f"- Avg winner: {ts.get('avg_winner_pct', 0):.2%} | "
                     f"Avg loser: {ts.get('avg_loser_pct', 0):.2%}")
        lines.append(f"- Stop rate: {ts['stop_rate']:.0%} | "
                     f"Avg hold: {ts['avg_bars_held']:.0f} bars")
        if ts.get("exit_breakdown"):
            lines.append(f"- Exits: {ts['exit_breakdown']}")
        if ts.get("direction_breakdown"):
            lines.append(f"- Directions: {ts['direction_breakdown']}")
    if ts.get("open_positions"):
        lines.append(f"\n### Open Positions")
        for op in ts["open_positions"]:
            lines.append(f"- {op['action']} strike={op['strike']} "
                        f"fill={op['fill_price']} gate={op['gate_trade_prob']:.2f} "
                        f"stop={op['stop_price']}")
    lines.append("")

    tod = metrics.get("time_of_day", {})
    if tod:
        lines.append("## Time of Day")
        lines.append("| Period | Trades | Total P&L | Avg P&L |")
        lines.append("|--------|--------|-----------|---------|")
        for period, data in sorted(tod.items()):
            lines.append(f"| {period} | {data['trades']} | "
                        f"{data['total_pnl_pct']:.2%} | {data['avg_pnl_pct']:.2%} |")
        lines.append("")

    inf = metrics.get("inference_summary", {})
    if inf.get("total_bars"):
        lines.append(f"## Model Behavior")
        lines.append(f"- Bars processed: {inf['total_bars']}")
        lines.append(f"- Trade signals (gate > 0.5): {inf['trade_signals']} "
                    f"({inf['signal_rate']:.0%})")
        lines.append("")

    lines.append(f"## Verdict: **{metrics['verdict']}**")
    lines.append(f"{metrics['verdict_reason']}")
    lines.append("")

    details = metrics.get("trade_details", [])
    if details:
        lines.append("## Trade Log")
        lines.append("| # | Entry | Action | Strike | Entry$ | Exit$ | P&L% | Bars | Exit |")
        lines.append("|---|-------|--------|--------|--------|-------|------|------|------|")
        for i, td in enumerate(details, 1):
            lines.append(
                f"| {i} | {td['entry_ts'][11:16] if td['entry_ts'] else '?'} | "
                f"{td['action']} | {td['strike']} | "
                f"{td['entry_price']:.2f} | {td['exit_price']:.2f} | "
                f"{td['pnl_pct']:.2%} | {td['bars_held']} | {td['exit_reason']} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="IBKR Paper Trading Session Analyzer")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT,
                        help="Path to audit.jsonl")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: results/ibkr_sessions/<date>.json)")
    parser.add_argument("--report", action="store_true",
                        help="Also print markdown report to stdout")
    args = parser.parse_args()

    parsed = parse_audit(args.audit)
    if "error" in parsed:
        print(f"Error: {parsed['error']}", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(parsed)

    # Determine output path
    out_path = args.output
    if out_path is None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        # Use the latest session date, or today
        dates = metrics["session_summary"].get("dates", [])
        date_str = dates[-1] if dates else datetime.now().strftime("%Y-%m-%d")
        out_path = SESSIONS_DIR / f"{date_str}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Session metrics → {out_path}")

    if args.report:
        print()
        print(format_report(metrics))

    # Always print summary to stdout
    ts = metrics["trade_summary"]
    v = metrics["verdict"]
    if ts["num_closed"] > 0:
        print(f"  {ts['num_closed']} trades | PF={ts['pf']:.2f} | "
              f"WR={ts['win_rate']:.0%} | Verdict: {v}")
    else:
        print(f"  No closed trades ({ts['num_open']} open) | Verdict: {v}")


if __name__ == "__main__":
    main()
