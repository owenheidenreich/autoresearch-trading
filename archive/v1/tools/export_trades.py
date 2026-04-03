#!/usr/bin/env python3
"""Convert JSONL audit trail from paper trading sessions into daily CSV + summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Action names matching training/prepare.py action space
ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "CALL_ATM",
    2: "CALL_OTM5",
    3: "CALL_OTM10",
    4: "PUT_ATM",
    5: "PUT_OTM5",
    6: "PUT_OTM10",
    7: "EXIT",
}

CSV_COLUMNS = [
    "trade_num",
    "date",
    "entry_time",
    "exit_time",
    "direction",
    "strike",
    "expiry",
    "entry_price",
    "exit_price",
    "qty",
    "pnl_pct",
    "pnl_usd",
    "hold_minutes",
    "exit_reason",
    "confidence",
    "gate_trade_prob",
    "position_id",
    "session_id",
]


def parse_audit_jsonl(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed line {line_num}", file=sys.stderr)
    return events


def reconstruct_trades(events: list[dict]) -> list[dict]:
    """Walk audit events and pair entries with exits to build trade rows."""
    # Index events by position_id
    positions: dict[str, dict] = {}  # position_id -> accumulated data

    for ev in events:
        event_type = ev.get("event", "")
        payload = ev.get("payload", {})
        ts = ev.get("ts", "")

        if event_type == "entry_intent":
            pid = payload.get("intent_id", "")
            # We'll match by intent_id later, but store contract info keyed by session+intent
            # The actual position_id comes from entry_intent_applied
            # Store intent data temporarily keyed by intent_id
            positions.setdefault(f"_intent_{payload.get('intent_id', '')}", {})["intent"] = {
                "ts": ts,
                "action": payload.get("action"),
                "qty": payload.get("qty"),
                "confidence": payload.get("confidence"),
                "stop_price": payload.get("stop_price"),
                "take_profit_price": payload.get("take_profit_price"),
                "reference_price": payload.get("reference_price"),
                "reason_codes": payload.get("reason_codes", []),
                "contract": payload.get("contract", {}),
                "session_id": payload.get("session_id"),
            }

        elif event_type == "entry_intent_applied":
            pid = payload.get("position_id", "")
            intent_id = payload.get("intent_id", "")
            intent_key = f"_intent_{intent_id}"
            intent_data = positions.pop(intent_key, {}).get("intent", {})

            positions[pid] = {
                "position_id": pid,
                "session_id": payload.get("session_id") or intent_data.get("session_id", ""),
                "entry_time": intent_data.get("ts", ts),
                "action": payload.get("action") or intent_data.get("action"),
                "qty": intent_data.get("qty", 1),
                "confidence": payload.get("confidence") or intent_data.get("confidence"),
                "reference_price": intent_data.get("reference_price"),
                "stop_price": intent_data.get("stop_price"),
                "contract": intent_data.get("contract", {}),
                "entry_fill_price": None,
                "exit_fill_price": None,
                "exit_time": None,
                "exit_reason": None,
                "gate_trade_prob": None,
            }

        elif event_type == "entry_dry_run":
            pid = payload.get("position_id", "")
            intent = payload.get("intent", {})
            state = payload.get("state", {})
            contract = payload.get("contract", {})
            positions[pid] = {
                "position_id": pid,
                "session_id": state.get("session_id", ""),
                "entry_time": ts,
                "action": intent.get("action"),
                "qty": intent.get("qty", 1),
                "confidence": intent.get("confidence"),
                "reference_price": intent.get("reference_price"),
                "stop_price": intent.get("stop_price"),
                "contract": contract,
                "entry_fill_price": state.get("fill_price"),
                "exit_fill_price": None,
                "exit_time": None,
                "exit_reason": None,
                "gate_trade_prob": intent.get("metadata", {}).get("gate_trade_prob"),
            }

        elif event_type == "entry_live":
            pid = payload.get("position_id", "")
            intent = payload.get("intent", {})
            contract = payload.get("contract", {})
            if pid not in positions:
                positions[pid] = {
                    "position_id": pid,
                    "session_id": payload.get("state", {}).get("session_id", ""),
                    "entry_time": ts,
                    "action": intent.get("action"),
                    "qty": intent.get("qty", 1),
                    "confidence": intent.get("confidence"),
                    "reference_price": intent.get("reference_price"),
                    "stop_price": intent.get("stop_price"),
                    "contract": contract,
                    "entry_fill_price": None,
                    "exit_fill_price": None,
                    "exit_time": None,
                    "exit_reason": None,
                    "gate_trade_prob": intent.get("metadata", {}).get("gate_trade_prob"),
                }

        elif event_type == "ib_order_status":
            pid = payload.get("position_id", "")
            role = payload.get("order_role", "")
            status = (payload.get("status") or "").lower()
            fill_price = payload.get("avg_fill_price")
            if pid in positions and status in ("filled", "partiallyfilled"):
                if role == "parent" and fill_price is not None:
                    positions[pid]["entry_fill_price"] = float(fill_price)
                elif role in ("stop", "take_profit") and fill_price is not None:
                    positions[pid]["exit_fill_price"] = float(fill_price)
                    positions[pid]["exit_time"] = positions[pid].get("exit_time") or ts
                    if role == "stop":
                        positions[pid]["exit_reason"] = positions[pid].get("exit_reason") or "stop_loss"
                    elif role == "take_profit":
                        positions[pid]["exit_reason"] = positions[pid].get("exit_reason") or "take_profit"

        elif event_type == "ib_exec_details":
            pid = payload.get("position_id", "")
            role = payload.get("order_role", "")
            price = payload.get("price")
            if pid in positions and price is not None:
                if role == "parent" and positions[pid]["entry_fill_price"] is None:
                    positions[pid]["entry_fill_price"] = float(price)

        elif event_type in ("model_exit", "flatten_dry_run", "flatten_live", "eod_flatten"):
            pid = payload.get("position_id", "")
            if pid in positions:
                positions[pid]["exit_time"] = positions[pid].get("exit_time") or ts
                reason_map = {
                    "model_exit": "model_exit",
                    "flatten_dry_run": payload.get("reason", "model_exit"),
                    "flatten_live": payload.get("reason", "model_exit"),
                    "eod_flatten": "eod_flatten",
                }
                positions[pid]["exit_reason"] = positions[pid].get("exit_reason") or reason_map.get(event_type, event_type)
                # For dry-run, exit price = entry price (simulated)
                if event_type == "flatten_dry_run" and positions[pid]["exit_fill_price"] is None:
                    positions[pid]["exit_fill_price"] = positions[pid].get("entry_fill_price") or positions[pid].get("reference_price")

        elif event_type == "model_inference":
            # Capture gate_trade_prob for the next trade if there is one
            # We'll attach it in post-processing
            pass

    # Attach gate_trade_prob from model_inference events
    # Walk events again and find the inference that triggered each entry
    inference_by_decision_id: dict[str, float] = {}
    for ev in events:
        if ev.get("event") == "model_inference":
            p = ev.get("payload", {})
            did = p.get("decision_id", "")
            if did:
                inference_by_decision_id[did] = p.get("gate_trade_prob", 0.0)

    for ev in events:
        if ev.get("event") in ("entry_intent", "entry_intent_applied"):
            p = ev.get("payload", {})
            did = p.get("decision_id", "")
            pid = p.get("position_id", "")
            if pid and pid in positions and did in inference_by_decision_id:
                positions[pid]["gate_trade_prob"] = inference_by_decision_id[did]

    # Build trade rows from positions that have entries
    trades = []
    # Filter out temporary _intent_ keys
    real_positions = {k: v for k, v in positions.items() if not k.startswith("_intent_")}

    for i, (pid, pos) in enumerate(sorted(real_positions.items(), key=lambda x: x[1].get("entry_time", "")), 1):
        entry_price = pos.get("entry_fill_price") or pos.get("reference_price")
        exit_price = pos.get("exit_fill_price")

        # Compute P&L
        pnl_pct = None
        pnl_usd = None
        if entry_price and exit_price and entry_price > 0:
            pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)
            # SPX options: 1 contract = 100 multiplier
            pnl_usd = round((exit_price - entry_price) * 100 * (pos.get("qty") or 1), 2)

        # Compute hold time
        hold_minutes = None
        entry_ts = pos.get("entry_time", "")
        exit_ts = pos.get("exit_time", "")
        if entry_ts and exit_ts:
            try:
                t0 = datetime.fromisoformat(entry_ts)
                t1 = datetime.fromisoformat(exit_ts)
                hold_minutes = round((t1 - t0).total_seconds() / 60.0, 1)
            except (ValueError, TypeError):
                pass

        # Extract contract details
        contract = pos.get("contract", {})
        strike = contract.get("strike")
        expiry = contract.get("lastTradeDateOrExpiry") or contract.get("expiry")

        # Parse date from entry_time
        trade_date = entry_ts[:10] if entry_ts and len(entry_ts) >= 10 else ""

        action = pos.get("action")
        direction = ACTION_NAMES.get(action, f"action_{action}") if action is not None else ""

        trades.append({
            "trade_num": i,
            "date": trade_date,
            "entry_time": entry_ts,
            "exit_time": exit_ts or "",
            "direction": direction,
            "strike": strike or "",
            "expiry": expiry or "",
            "entry_price": round(entry_price, 2) if entry_price is not None else "",
            "exit_price": round(exit_price, 2) if exit_price is not None else "",
            "qty": pos.get("qty", 1),
            "pnl_pct": pnl_pct if pnl_pct is not None else "",
            "pnl_usd": pnl_usd if pnl_usd is not None else "",
            "hold_minutes": hold_minutes if hold_minutes is not None else "",
            "exit_reason": pos.get("exit_reason", ""),
            "confidence": round(pos.get("confidence", 0), 4) if pos.get("confidence") is not None else "",
            "gate_trade_prob": round(pos.get("gate_trade_prob", 0), 4) if pos.get("gate_trade_prob") is not None else "",
            "position_id": pid,
            "session_id": pos.get("session_id", ""),
        })

    return trades


def write_csv(trades: list[dict], output_path: Path) -> None:
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(trades)
    print(f"  CSV: {output_path} ({len(trades)} trades)")


def write_summary(trades: list[dict], events: list[dict], output_path: Path) -> None:
    os.makedirs(output_path.parent, exist_ok=True)
    lines = []

    # Session info
    sessions = [e for e in events if e.get("event") == "session_start"]
    for s in sessions:
        p = s.get("payload", {})
        lines.append(f"Session: {p.get('session_id', '?')}  dry_run={p.get('dry_run', '?')}  account={p.get('account', '?')}")
    lines.append("")

    # Trade summary
    n = len(trades)
    lines.append(f"Total trades: {n}")
    if n == 0:
        lines.append("No trades taken.")
    else:
        pnls = [t["pnl_pct"] for t in trades if isinstance(t["pnl_pct"], (int, float))]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl_usd = sum(t["pnl_usd"] for t in trades if isinstance(t["pnl_usd"], (int, float)))
        hold_times = [t["hold_minutes"] for t in trades if isinstance(t["hold_minutes"], (int, float))]

        lines.append(f"Win rate: {len(wins)}/{len(pnls)} ({len(wins)/len(pnls)*100:.0f}%)" if pnls else "Win rate: N/A")
        lines.append(f"Total P&L: ${total_pnl_usd:,.2f}")
        if pnls:
            lines.append(f"Avg P&L: {sum(pnls)/len(pnls):.2f}%")
            lines.append(f"Best: {max(pnls):.2f}%  Worst: {min(pnls):.2f}%")
        if hold_times:
            lines.append(f"Avg hold: {sum(hold_times)/len(hold_times):.1f} min")
        # Exit reasons
        reasons = {}
        for t in trades:
            r = t.get("exit_reason") or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        lines.append(f"Exit reasons: {reasons}")

    # Inference stats
    inferences = [e for e in events if e.get("event") == "model_inference"]
    trade_signals = [e for e in inferences if e["payload"].get("action", 0) != 0]
    lines.append("")
    lines.append(f"Model inferences: {len(inferences)}")
    lines.append(f"Trade signals: {len(trade_signals)}")
    if inferences:
        gate_probs = [e["payload"].get("gate_trade_prob", 0) for e in inferences]
        lines.append(f"Gate trade prob: mean={sum(gate_probs)/len(gate_probs):.4f}  max={max(gate_probs):.4f}")

    text = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(text)
    print(f"  Summary: {output_path}")
    print()
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper trading audit JSONL to CSV")
    parser.add_argument("audit_path", help="Path to audit JSONL file")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as audit file)")
    args = parser.parse_args()

    audit_path = Path(args.audit_path)
    if not audit_path.exists():
        print(f"ERROR: {audit_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {audit_path}...")
    events = parse_audit_jsonl(audit_path)
    print(f"  {len(events)} events")

    trades = reconstruct_trades(events)

    # Determine output paths
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = audit_path.parent

    stem = audit_path.stem  # e.g. "audit-2026-03-20" or "audit"
    csv_path = out_dir / f"trades-{stem.replace('audit', '').strip('-') or 'all'}.csv"
    summary_path = out_dir / f"summary-{stem.replace('audit', '').strip('-') or 'all'}.txt"

    # If stem is just "audit", use today's date
    if csv_path.name == "trades-all.csv":
        # Try to extract date from session_start events
        for ev in events:
            if ev.get("event") == "session_start":
                ts = ev.get("ts", "")
                if len(ts) >= 10:
                    date_str = ts[:10]
                    csv_path = out_dir / f"trades-{date_str}.csv"
                    summary_path = out_dir / f"summary-{date_str}.txt"
                    break

    write_csv(trades, csv_path)
    write_summary(trades, events, summary_path)


if __name__ == "__main__":
    main()
