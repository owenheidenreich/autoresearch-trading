#!/usr/bin/env python3
"""
Live trading dashboard — browser-based monitor for IBKR paper trading.

Reads audit.jsonl and optionally connects to IBKR for independent position
verification (--ibkr flag).

Usage:
  python3 tools/live_dash.py                          # Audit-only (default)
  python3 tools/live_dash.py --ibkr                   # With IBKR live verification
  python3 tools/live_dash.py --audit results/live/audit.jsonl --port 8421
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Optional: ib_insync for live IBKR verification
try:
    from ib_insync import IB, util
    HAS_IB = True
except ImportError:
    HAS_IB = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT = PROJECT_ROOT / "results" / "live" / "audit.jsonl"

# Action ID → human label
ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "BUY_CALL_ATM",
    2: "BUY_CALL_OTM5",
    3: "BUY_CALL_OTM10",
    4: "BUY_PUT_ATM",
    5: "BUY_PUT_OTM5",
    6: "BUY_PUT_OTM10",
    7: "EXIT",
}

DIRECTION_LABELS = ["CALL_ATM", "CALL_OTM5", "CALL_OTM10", "PUT_ATM", "PUT_OTM5", "PUT_OTM10"]

# ── IBKR Live Reader ─────────────────────────────────────────────────────

class IBKRDashReader:
    """Read-only IBKR connection running in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 80):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._data: dict = {"connected": False}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ibkr-dash")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        return self._data

    def _loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ib = IB()
        while not self._stop.is_set():
            try:
                if not ib.isConnected():
                    ib.connect(self.host, self.port, clientId=self.client_id,
                               timeout=10, readonly=True)
                    print(f"[ibkr-dash] Connected (client_id={self.client_id})")
                self._poll(ib)
                ib.sleep(5)
            except Exception as exc:
                self._data = {"connected": False, "error": str(exc)}
                if ib.isConnected():
                    try:
                        ib.disconnect()
                    except Exception:
                        pass
                self._stop.wait(10)
        if ib.isConnected():
            ib.disconnect()

    def _poll(self, ib: IB) -> None:
        now = datetime.now(timezone.utc).isoformat()

        # Portfolio positions
        positions = []
        for item in ib.portfolio():
            c = item.contract
            mid = None
            if item.marketPrice and math.isfinite(item.marketPrice):
                mid = float(item.marketPrice)
            positions.append({
                "symbol": c.localSymbol or f"{c.symbol} {getattr(c, 'lastTradeDateOrContractMonth', '')} {getattr(c, 'strike', '')} {getattr(c, 'right', '')}",
                "secType": c.secType,
                "qty": int(item.position),
                "avgCost": round(float(item.averageCost), 2),
                "marketPrice": mid,
                "marketValue": round(float(item.marketValue), 2),
                "unrealizedPNL": round(float(item.unrealizedPNL), 2),
                "realizedPNL": round(float(item.realizedPNL), 2),
            })

        # Account summary
        account: dict[str, str] = {}
        try:
            for row in ib.accountSummary():
                if row.tag in ("NetLiquidation", "AvailableFunds", "TotalCashValue",
                               "BuyingPower", "GrossPositionValue"):
                    key = row.tag if not row.currency else f"{row.tag}"
                    account[key] = row.value
        except Exception:
            pass

        # Open orders
        open_orders = []
        for trade in ib.openTrades():
            o = trade.order
            c = trade.contract
            open_orders.append({
                "symbol": c.localSymbol or c.symbol,
                "action": o.action,
                "qty": int(o.totalQuantity),
                "orderType": o.orderType,
                "lmtPrice": float(o.lmtPrice) if o.lmtPrice else None,
                "auxPrice": float(o.auxPrice) if o.auxPrice else None,
                "status": trade.orderStatus.status if trade.orderStatus else "",
            })

        self._data = {
            "connected": True,
            "last_update": now,
            "positions": positions,
            "account": account,
            "open_orders": open_orders,
        }


# ── Parse audit.jsonl ────────────────────────────────────────────────────

def _collect_audit_files(path: Path) -> list[Path]:
    """Collect all .jsonl files to parse. Reads the given file plus all
    sibling .jsonl files in the same directory so session-specific audit
    files (e.g. v6_bugfix4_session.jsonl) are included."""
    if path.is_dir():
        directory = path
    else:
        directory = path.parent

    files = sorted(directory.glob("*.jsonl"))
    if not files and path.is_file() and path.exists():
        files = [path]
    return files


def parse_audit(path: Path) -> dict:
    """Parse all .jsonl files in audit directory into structured dashboard data."""
    files = _collect_audit_files(path)
    if not files:
        return {"error": "No audit .jsonl files found", "sessions": [], "trades": [],
                "inferences": [], "positions": [], "open_positions": [],
                "current_session": None}

    sessions = []
    trades = []
    inferences = []
    entries = {}       # intent_id -> entry data
    dry_runs = {}      # intent_id -> dry_run state (has fill info)
    positions = []     # closed positions
    pnl_updates = []   # cumulative P&L snapshots for equity curve
    current_session = None
    bar_snapshots = []
    session_ends = {}  # session_id -> end data
    context_refreshes = []
    feature_projections = []

    for audit_file in files:
      with open(audit_file) as f:
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
                    "ts": ts,
                    "session_id": p.get("session_id", ""),
                    "seed_spx": p.get("seed_spx", 0),
                    "dry_run": p.get("dry_run", False),
                    "num_features": p.get("context_num_features", 0),
                    "model_features": p.get("model_num_features", 0),
                    "account": p.get("account", ""),
                }
                sessions.append(current_session)

            elif event == "session_end":
                sid = p.get("session_id", "")
                session_ends[sid] = {
                    "ts": ts,
                    "processed_minutes": p.get("processed_minutes", 0),
                    "signals": p.get("signals_generated", 0),
                    "entries": p.get("entries_applied", 0),
                    "exits": p.get("exits_applied", 0),
                    "open_final": p.get("open_positions_final", 0),
                }

            elif event == "entry_intent":
                intent_id = p.get("intent_id", "")
                contract = p.get("contract", {})
                action_id = p.get("action", 0)
                entries[intent_id] = {
                    "ts": ts,
                    "action": action_id,
                    "action_name": ACTION_NAMES.get(action_id, f"ACTION_{action_id}"),
                    "confidence": p.get("confidence", 0),
                    "strike": contract.get("strike", ""),
                    "right": contract.get("right", ""),
                    "expiry": contract.get("lastTradeDateOrContractMonth", ""),
                    "symbol": contract.get("localSymbol", contract.get("tradingClass", "")),
                    "stop_price": p.get("stop_price", 0),
                    "tp_price": p.get("take_profit_price", 0),
                    "ref_price": p.get("reference_price", 0),
                    "intent_id": intent_id,
                    "applied": False,
                    "reason_codes": p.get("reason_codes", []),
                    "qty": p.get("qty", 1),
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
                gate_prob = meta.get("gate_trade_prob", intent_meta.get("gate_trade_prob", 0))
                direction_probs = meta.get("direction_probs", intent_meta.get("direction_probs", []))
                entry_limit = intent_data.get("entry_limit_price", meta.get("entry_limit_price", 0))
                dry_runs[intent_id] = {
                    "position_id": p.get("position_id", ""),
                    "fill_price": state.get("fill_price", 0),
                    "fill_status": state.get("fill_status", ""),
                    "gate_trade_prob": gate_prob,
                    "direction_probs": direction_probs,
                    "entry_limit": entry_limit,
                    "stop_price": state.get("current_stop", 0),
                    "tp_price": state.get("current_take_profit", 0),
                    "contract": state.get("contract", {}),
                    "status": state.get("status", ""),
                }
                if intent_id in entries:
                    entries[intent_id]["gate_trade_prob"] = gate_prob
                    entries[intent_id]["fill_price"] = state.get("fill_price", 0)
                    entries[intent_id]["position_id"] = p.get("position_id", "")
                    entries[intent_id]["direction_probs"] = direction_probs
                    entries[intent_id]["entry_limit"] = entry_limit

            elif event == "fill":
                fill_data = {
                    "ts": ts,
                    "fill_price": p.get("fill_price", 0),
                    "fill_qty": p.get("fill_qty", 0),
                    "side": p.get("side", ""),
                    "slippage_bps": p.get("slippage_bps", 0),
                }
                trades.append(fill_data)

            elif event == "position_closed":
                pos = {
                    "ts": ts,
                    "pnl_pct": p.get("pnl_pct", 0),
                    "pnl_dollar": p.get("pnl_dollar", 0),
                    "bars_held": p.get("bars_held", 0),
                    "exit_reason": p.get("exit_reason", ""),
                    "direction": p.get("direction", ""),
                    "strike": p.get("strike", ""),
                    "entry_price": p.get("entry_price", 0),
                    "exit_price": p.get("exit_price", 0),
                    "entry_ts": p.get("entry_ts", ""),
                    "intent_id": p.get("intent_id", ""),
                    "position_id": p.get("position_id", ""),
                    "gate_confidence": p.get("gate_confidence", 0),
                    "qty": p.get("qty", 1),
                }
                positions.append(pos)

            elif event == "pnl_update":
                # Track cumulative P&L for equity curve
                pnl_updates.append({
                    "ts": ts,
                    "session_pnl_dollars": p.get("session_pnl_dollars", 0),
                    "realized_pnl_pct": p.get("realized_pnl_pct", 0),
                    "trade_pnl_pct": p.get("trade_pnl_pct", 0),
                    "trade_pnl_dollars": p.get("trade_pnl_dollars", 0),
                    "position_id": p.get("position_id", ""),
                    "entry_price": p.get("entry_price", 0),
                    "exit_price": p.get("exit_price", 0),
                    "reason": p.get("reason", ""),
                })

            elif event == "model_inference":
                inf = {
                    "ts": ts,
                    "action": p.get("action", 0),
                    "action_name": ACTION_NAMES.get(p.get("action", 0), "?"),
                    "gate_trade_prob": p.get("gate_trade_prob", 0),
                    "confidence": p.get("confidence", 0),
                    "decision_id": p.get("decision_id", ""),
                    "reason_codes": p.get("reason_codes", []),
                    "direction_probs": p.get("direction_probs", []),
                }
                inferences.append(inf)

            elif event == "bar_snapshot":
                bar_snapshots.append({
                    "ts": ts,
                    "completeness": p.get("completeness", 0),
                    "present": p.get("present_feature_count", 0),
                    "missing": p.get("missing_feature_count", 0),
                    "missing_names": p.get("missing_feature_names", []),
                    "staleness": p.get("staleness_seconds", {}),
                })

            elif event == "context_refresh":
                context_refreshes.append({
                    "ts": ts,
                    "bars": p.get("bars", 0),
                    "as_of_date": p.get("as_of_date", ""),
                })

            elif event == "feature_projection":
                feature_projections.append({
                    "ts": ts,
                    "context_features": p.get("context_num_features", 0),
                    "model_features": p.get("model_num_features", 0),
                    "mode": p.get("mode", ""),
                })


    # Fallback: reconstruct closed positions from pnl_updates for sessions
    # that didn't emit position_closed events. Deduplicate against real closes.
    closed_pos_ids_real = {p["position_id"] for p in positions if p.get("position_id")}
    for pu in pnl_updates:
        pos_id = pu.get("position_id", "")
        if not pos_id or pos_id in closed_pos_ids_real:
            continue
        matching_entry = next(
            (e for e in entries.values() if e.get("position_id") == pos_id), None)
        positions.append({
            "ts": pu.get("ts", ""),
            "pnl_pct": pu.get("trade_pnl_pct", 0),
            "pnl_dollar": pu.get("trade_pnl_dollars", 0),
            "bars_held": 0,
            "exit_reason": pu.get("reason", ""),
            "direction": matching_entry.get("action_name", "") if matching_entry else "",
            "strike": matching_entry.get("strike", "") if matching_entry else "",
            "entry_price": pu.get("entry_price", 0),
            "exit_price": pu.get("exit_price", 0),
            "entry_ts": matching_entry.get("ts", "") if matching_entry else "",
            "intent_id": matching_entry.get("intent_id", "") if matching_entry else "",
            "position_id": pos_id,
            "gate_confidence": matching_entry.get("gate_trade_prob", 0) if matching_entry else 0,
            "qty": matching_entry.get("qty", 1) if matching_entry else 1,
        })
        closed_pos_ids_real.add(pos_id)  # prevent double-add from multiple pnl_updates

    # Determine open positions: entries that were applied but not closed
    closed_position_ids = {p["position_id"] for p in positions if p.get("position_id")}
    closed_intent_ids = {p["intent_id"] for p in positions if p.get("intent_id")}
    open_positions = []
    for eid, entry in entries.items():
        if entry.get("applied") and eid not in closed_intent_ids:
            pos_id = entry.get("position_id", "")
            if pos_id and pos_id not in closed_position_ids:
                dr = dry_runs.get(eid, {})
                open_positions.append({
                    **entry,
                    "gate_trade_prob": entry.get("gate_trade_prob", dr.get("gate_trade_prob", 0)),
                    "fill_price": entry.get("fill_price", dr.get("fill_price", 0)),
                    "stop_price": dr.get("stop_price", entry.get("stop_price", 0)),
                    "tp_price": dr.get("tp_price", entry.get("tp_price", 0)),
                    "contract_symbol": dr.get("contract", {}).get("localSymbol", ""),
                    "contract_strike": dr.get("contract", {}).get("strike", entry.get("strike", "")),
                    "contract_right": dr.get("contract", {}).get("right", entry.get("right", "")),
                    "direction_probs": entry.get("direction_probs", dr.get("direction_probs", [])),
                    "entry_limit": entry.get("entry_limit", dr.get("entry_limit", 0)),
                })

    summary = _compute_summary(positions, inferences, current_session, open_positions, entries, session_ends)

    # Most recent mtime across all audit files
    audit_mtime = ""
    try:
        mt = max(os.path.getmtime(f) for f in files)
        audit_mtime = datetime.fromtimestamp(mt, tz=timezone.utc).isoformat()
    except (OSError, ValueError):
        pass

    return {
        "sessions": sessions,
        "session_ends": session_ends,
        "trades": trades,
        "inferences": inferences[-80:],
        "all_gate_probs": [i["gate_trade_prob"] for i in inferences],
        "positions": positions,
        "open_positions": open_positions,
        "entries": list(entries.values()),
        "bar_snapshots": bar_snapshots[-15:],
        "current_session": current_session,
        "context_refreshes": context_refreshes,
        "feature_projections": feature_projections,
        "pnl_updates": pnl_updates,
        "summary": summary,
        "audit_mtime": audit_mtime,
    }


def _compute_summary(positions, inferences, session, open_positions, entries, session_ends):
    if not positions:
        total_pnl = 0; win_rate = 0; num_trades = 0
        avg_hold = 0; winners = 0; losers = 0; pf = 0
        gross_win = 0; gross_loss = 0
    else:
        pnls = [p["pnl_pct"] for p in positions]
        total_pnl = sum(pnls)
        winners = sum(1 for p in pnls if p > 0)
        losers = sum(1 for p in pnls if p <= 0)
        num_trades = len(pnls)
        win_rate = winners / num_trades if num_trades > 0 else 0
        avg_hold = sum(p.get("bars_held", 0) for p in positions) / num_trades
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0

    trade_signals = sum(1 for i in inferences if i.get("gate_trade_prob", 0) > 0.5)
    total_bars = len(inferences)
    signal_rate = trade_signals / total_bars if total_bars > 0 else 0
    avg_gate_prob = sum(i.get("gate_trade_prob", 0) for i in inferences) / total_bars if total_bars > 0 else 0

    entries_applied = sum(1 for e in entries.values() if e.get("applied"))
    conversion_rate = entries_applied / trade_signals if trade_signals > 0 else 0

    total_minutes = sum(v.get("processed_minutes", 0) for v in session_ends.values())
    num_sessions = len(session_ends)

    return {
        "num_closed": num_trades,
        "num_open": len(open_positions),
        "winners": winners,
        "losers": losers,
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl,
        "avg_hold_bars": avg_hold,
        "profit_factor": pf,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "trade_signals": trade_signals,
        "total_bars": total_bars,
        "signal_rate": signal_rate,
        "avg_gate_prob": avg_gate_prob,
        "entries_applied": entries_applied,
        "conversion_rate": conversion_rate,
        "total_minutes": total_minutes,
        "num_sessions": num_sessions,
        "dry_run": session.get("dry_run", True) if session else True,
    }


# ── HTML Dashboard ───────────────────────────────────────────────────────

def render_html():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ART² Live Trading</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #0a0a0f; color: #e0e0e0; padding: 16px; }
  h1 { color: #00ff88; font-size: 1.4em; margin-bottom: 4px; }
  .subtitle { color: #555; font-size: 0.8em; margin-bottom: 12px; }
  h2 { color: #888; font-size: 1.0em; margin: 12px 0 6px; border-bottom: 1px solid #222; padding-bottom: 4px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
  .card { background: #111118; border: 1px solid #222; border-radius: 8px; padding: 12px; }
  .card.full { grid-column: 1 / -1; }
  .card.highlight { border-color: #00ff8844; background: #0a1a0f; }
  .card.warn { border-color: #ffaa0044; background: #1a1a0a; }
  .stat { display: inline-block; margin-right: 20px; margin-bottom: 8px; }
  .stat .label { color: #666; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat .value { font-size: 1.25em; font-weight: bold; }
  .stat-sm { display: inline-block; margin-right: 14px; margin-bottom: 4px; }
  .stat-sm .label { color: #555; font-size: 0.65em; text-transform: uppercase; }
  .stat-sm .value { font-size: 0.95em; font-weight: bold; }
  .green { color: #00ff88; }
  .red { color: #ff4444; }
  .yellow { color: #ffaa00; }
  .blue { color: #4488ff; }
  .dim { color: #555; }
  .orange { color: #ff8800; }
  .dry-run-badge { background: #332200; color: #ffaa00; padding: 4px 10px; border-radius: 4px; display: inline-block; margin-left: 12px; font-size: 0.8em; }
  .live-badge { background: #003300; color: #00ff88; padding: 4px 10px; border-radius: 4px; display: inline-block; margin-left: 12px; font-size: 0.8em; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
  th { text-align: left; color: #666; font-weight: normal; padding: 4px 8px; border-bottom: 1px solid #333; }
  td { padding: 4px 8px; border-bottom: 1px solid #1a1a1a; }
  tr:hover { background: #1a1a22; }
  tr.trade-signal { border-left: 3px solid #00ff8866; }
  .pnl-pos { color: #00ff88; font-weight: bold; }
  .pnl-neg { color: #ff4444; font-weight: bold; }
  .action-trade { color: #00ff88; font-weight: bold; }
  .action-hold { color: #555; }
  .gate-bar { display: inline-block; height: 10px; background: #222; border-radius: 5px; width: 70px; position: relative; vertical-align: middle; margin-right: 4px; }
  .gate-fill { height: 100%; border-radius: 5px; position: absolute; left: 0; top: 0; }
  .no-data { color: #444; font-style: italic; padding: 20px; text-align: center; }
  .refresh-info { color: #444; font-size: 0.75em; float: right; }
  .pulse { animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
  .position-card { padding: 12px; }
  .position-card .contract { font-size: 1.2em; font-weight: bold; color: #00ff88; margin-bottom: 8px; }
  .position-card .detail-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
  .time-et { color: #4488ff; }
  .time-pt { color: #666; font-size: 0.8em; }
  .pill { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.75em; margin: 1px; }
  .pill-grey { background: #333; color: #888; }
  .pill-green { background: #0a2a0a; color: #00ff88; }
  .pill-yellow { background: #2a2a0a; color: #ffaa00; }
  .pill-orange { background: #2a1a0a; color: #ff8800; }
  .pill-red { background: #2a0a0a; color: #ff4444; }
  .dir-bar { display: inline-flex; height: 14px; border-radius: 3px; overflow: hidden; width: 120px; vertical-align: middle; }
  .dir-seg { height: 100%; min-width: 1px; }
  .progress-track { display: inline-block; width: 100px; height: 8px; background: #222; border-radius: 4px; position: relative; vertical-align: middle; margin: 0 6px; }
  .progress-fill { height: 100%; border-radius: 4px; position: absolute; left: 0; top: 0; }
  .progress-marker { width: 2px; height: 12px; position: absolute; top: -2px; background: #fff; }
  .sparkline { display: block; width: 100%; margin-bottom: 8px; }
  .tod-label { font-size: 0.85em; padding: 3px 8px; border-radius: 4px; display: inline-block; }
  .tod-opening { background: #0a2a0a; color: #00ff88; }
  .tod-midmorning { background: #0a1a2a; color: #4488ff; }
  .tod-lunch { background: #2a1a0a; color: #ff8800; }
  .tod-afternoon { background: #0a1a2a; color: #4488ff; }
  .tod-power { background: #2a0a0a; color: #ff4444; }
  .tod-premarket { background: #222; color: #888; }
  .tod-afterhours { background: #222; color: #888; }
  .session-active { color: #00ff88; }
  .session-ended { color: #555; }
  .staleness-good { color: #00ff88; }
  .staleness-warn { color: #ffaa00; }
  .staleness-bad { color: #ff4444; }
  .age-good { color: #00ff88; }
  .age-warn { color: #ffaa00; }
  .age-bad { color: #ff4444; }
  .histogram { display: flex; align-items: flex-end; height: 60px; gap: 2px; margin-bottom: 20px; }
  .hist-bar { border-radius: 2px 2px 0 0; flex: 1; min-width: 0; position: relative; }
  .hist-bar .hist-label { position: absolute; bottom: -16px; font-size: 0.6em; color: #666; text-align: center; width: 100%; white-space: nowrap; }
  .audit-stale { color: #ff4444; font-size: 0.7em; }
  .audit-fresh { color: #00ff88; font-size: 0.7em; }
</style>
</head>
<body>
<h1>ART² Live Trading <span id="mode-badge"></span> <span class="refresh-info" id="refresh-ts"></span></h1>
<div class="subtitle" id="clock"></div>

<div class="grid">
  <!-- Summary bar -->
  <div class="card full" id="summary-card"><div id="summary-stats"></div></div>

  <!-- Equity Curve -->
  <div class="card full" id="equity-card">
    <h2>Equity Curve</h2>
    <div id="equity-curve"><span class="no-data">Waiting for trades...</span></div>
  </div>

  <!-- Position + Market Context -->
  <div class="card highlight" id="position-card">
    <h2>Current Position</h2>
    <div id="current-position"><span class="no-data">Flat — no open position</span></div>
  </div>

  <div class="card" id="market-card">
    <h2>Market Context</h2>
    <div id="market-context"></div>
  </div>

  <!-- IBKR Open Orders -->
  <div class="card full" id="ibkr-card" style="display:none">
    <h2>IBKR Open Orders <span id="ibkr-status"></span></h2>
    <div id="ibkr-content"><span class="no-data">No open orders</span></div>
  </div>

  <!-- Session History -->
  <div class="card full">
    <h2>Session History</h2>
    <div id="session-history"><span class="no-data">No sessions</span></div>
  </div>

  <!-- Closed Trades -->
  <div class="card full">
    <h2>Closed Trades</h2>
    <div id="trade-history"><span class="no-data">No closed trades yet</span></div>
  </div>

  <!-- Entry Signals -->
  <div class="card full">
    <h2>Entry Signals</h2>
    <div id="entry-intents"><span class="no-data">No entries yet</span></div>
  </div>

  <!-- Model Behavior -->
  <div class="card full">
    <h2>Model Behavior</h2>
    <div id="gate-histogram"></div>
    <div id="gate-sparkline"></div>
    <div id="inference-stream" style="max-height: 400px; overflow-y: auto;"><span class="no-data">Waiting for inferences...</span></div>
  </div>

  <!-- Bar Quality -->
  <div class="card full">
    <h2>Data Quality</h2>
    <div id="bar-quality"><span class="no-data">No bar data yet</span></div>
  </div>
</div>

<script>
const ACTION_NAMES = {0:'DO_NOTHING',1:'CALL_ATM',2:'CALL_OTM5',3:'CALL_OTM10',4:'PUT_ATM',5:'PUT_OTM5',6:'PUT_OTM10',7:'EXIT'};
const DIR_LABELS = ['CALL_ATM','CALL_OTM5','CALL_OTM10','PUT_ATM','PUT_OTM5','PUT_OTM10'];
const DIR_COLORS = ['#00cc66','#00aa88','#008866','#cc4444','#aa4466','#884466'];

function fmt(v, d=2) { return v != null ? Number(v).toFixed(d) : '--'; }
function fmtPct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '--'; }
function fmtPctRaw(v) { return v != null ? Number(v).toFixed(1) + '%' : '--'; }
function fmtDollar(v) { return v != null && v !== 0 ? '$' + Number(v).toFixed(2) : '--'; }
function pnlClass(v) { return v > 0 ? 'pnl-pos' : v < 0 ? 'pnl-neg' : ''; }

function toET(utcStr) {
    if (!utcStr) return '--';
    const d = new Date(utcStr.endsWith('Z') ? utcStr : utcStr + 'Z');
    return d.toLocaleString('en-US', {timeZone:'America/New_York', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:true});
}
function toPT(utcStr) {
    if (!utcStr) return '--';
    const d = new Date(utcStr.endsWith('Z') ? utcStr : utcStr + 'Z');
    return d.toLocaleString('en-US', {timeZone:'America/Los_Angeles', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:true});
}
function timeCell(utcStr) {
    return `<span class="time-et">${toET(utcStr)}</span> <span class="time-pt">(${toPT(utcStr)} PT)</span>`;
}

function gateBar(prob) {
    if (prob == null || prob === 0) return '<span class="dim">--</span>';
    const pct = Math.min(100, Math.max(0, prob * 100)).toFixed(0);
    const color = prob > 0.55 ? '#00ff88' : prob > 0.45 ? '#ffaa00' : '#ff4444';
    return `<span class="gate-bar"><span class="gate-fill" style="width:${pct}%;background:${color}"></span></span>${(prob*100).toFixed(1)}%`;
}

function dirBar(probs) {
    if (!probs || probs.length < 6) return '';
    let html = '<span class="dir-bar" title="';
    for (let i = 0; i < 6; i++) html += `${DIR_LABELS[i]}: ${(probs[i]*100).toFixed(1)}% `;
    html += '">';
    for (let i = 0; i < 6; i++) {
        const w = Math.max(1, probs[i] * 100);
        html += `<span class="dir-seg" style="width:${w}%;background:${DIR_COLORS[i]}"></span>`;
    }
    html += '</span>';
    // Show top direction
    let maxI = 0;
    for (let i = 1; i < 6; i++) if (probs[i] > probs[maxI]) maxI = i;
    html += ` <span style="color:${DIR_COLORS[maxI]};font-size:0.8em">${DIR_LABELS[maxI]} ${(probs[maxI]*100).toFixed(0)}%</span>`;
    return html;
}

function actionLabel(action, name) {
    const n = name || ACTION_NAMES[action] || `ACTION_${action}`;
    if (action === 0) return `<span class="action-hold">${n}</span>`;
    if (action === 7) return `<span class="orange">${n}</span>`;
    return `<span class="action-trade">${n}</span>`;
}

function reasonPills(codes) {
    if (!codes || codes.length === 0) return '<span class="dim">—</span>';
    return codes.map(c => {
        if (c === 'gate_no_trade') return `<span class="pill pill-grey">${c}</span>`;
        if (c === 'trade_signal') return `<span class="pill pill-green">${c}</span>`;
        if (c === 'cooldown') return `<span class="pill pill-yellow">${c}</span>`;
        if (c === 'pre_10am' || c === 'pre_10am_gate') return `<span class="pill pill-orange">${c}</span>`;
        if (c === 'in_position') return `<span class="pill pill-yellow">${c}</span>`;
        return `<span class="pill pill-grey">${c}</span>`;
    }).join('');
}

function formatOptionSymbol(raw) {
    // Parse IBKR localSymbol like "SPXW  260325C06590000" into "SPXW 6590 C 03/25/2026"
    if (!raw) return raw;
    const m = raw.match(/^(SPXW?)\s+(\d{6})([CP])(\d{8})$/);
    if (!m) return raw;
    const [, root, dateStr, right, strikeRaw] = m;
    // dateStr = YYMMDD
    const yy = dateStr.slice(0, 2);
    const mm = dateStr.slice(2, 4);
    const dd = dateStr.slice(4, 6);
    const dateFormatted = `${mm}/${dd}/20${yy}`;
    // strikeRaw = 8 digits, last 3 are decimals (e.g. 06590000 = 6590.000)
    const strike = (parseInt(strikeRaw, 10) / 1000).toFixed(0);
    const rightLabel = right === 'C' ? 'Call' : 'Put';
    return `${root} ${strike} ${rightLabel} ${dateFormatted}`;
}

function stalenessClass(s) {
    if (typeof s !== 'number') return 'dim';
    if (s < 2) return 'staleness-good';
    if (s < 5) return 'staleness-warn';
    return 'staleness-bad';
}

function todRegime() {
    const now = new Date();
    const etStr = now.toLocaleString('en-US', {timeZone:'America/New_York', hour:'numeric', minute:'numeric', hour12:false});
    const [h, m] = etStr.split(':').map(Number);
    const mins = h * 60 + m;
    if (mins < 570) return {label: 'Pre-Market', cls: 'tod-premarket'};
    if (mins < 630) return {label: 'Opening Drive (9:30-10:30)', cls: 'tod-opening'};
    if (mins < 690) return {label: 'Mid-Morning (10:30-11:30)', cls: 'tod-midmorning'};
    if (mins < 810) return {label: 'Lunch Chop (11:30-13:30)', cls: 'tod-lunch'};
    if (mins < 930) return {label: 'Afternoon (13:30-15:30)', cls: 'tod-afternoon'};
    if (mins < 960) return {label: 'Power Hour (15:30-16:00)', cls: 'tod-power'};
    return {label: 'After Hours', cls: 'tod-afterhours'};
}

function minutesSinceOpen() {
    const now = new Date();
    const etStr = now.toLocaleString('en-US', {timeZone:'America/New_York', hour:'numeric', minute:'numeric', hour12:false});
    const [h, m] = etStr.split(':').map(Number);
    return h * 60 + m - 570; // 9:30 = 570 min
}

function elapsedSinceEntry(entryTs) {
    if (!entryTs) return {mins: 0, bars: 0};
    const entryDate = new Date(entryTs.endsWith('Z') ? entryTs : entryTs + 'Z');
    const nowMs = Date.now();
    const diffMin = Math.floor((nowMs - entryDate.getTime()) / 60000);
    return {mins: Math.max(0, diffMin), bars: Math.max(0, diffMin)};
}

function ageClass(mins) {
    if (mins < 15) return 'age-good';
    if (mins < 30) return 'age-warn';
    return 'age-bad';
}

function updateClock() {
    const now = new Date();
    const et = now.toLocaleString('en-US',{timeZone:'America/New_York',weekday:'short',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:true});
    const pt = now.toLocaleString('en-US',{timeZone:'America/Los_Angeles',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:true});
    document.getElementById('clock').textContent = `Market: ${et} ET  •  Local: ${pt} PT`;
}
updateClock();
setInterval(updateClock, 1000);

function sparklineSvg(values, width, height) {
    if (!values || values.length < 2) return '';
    const n = values.length;
    const max = Math.max(...values, 0.6);
    const min = Math.min(...values, 0);
    const range = max - min || 1;
    let path = '';
    for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * width;
        const y = height - ((values[i] - min) / range) * height;
        path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
    }
    // 0.5 threshold line
    const threshY = height - ((0.5 - min) / range) * height;
    return `<svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
        <line x1="0" y1="${threshY.toFixed(1)}" x2="${width}" y2="${threshY.toFixed(1)}" stroke="#ffaa0033" stroke-width="1" stroke-dasharray="4,4"/>
        <path d="${path}" fill="none" stroke="#00ff88" stroke-width="1.5"/>
    </svg>`;
}

function equityCurveSvg(pnlUpdates, closedPositions, width, height) {
    // Build equity data points from either pnl_updates or closed positions
    let points = [];
    if (pnlUpdates && pnlUpdates.length > 0) {
        points = pnlUpdates.map(p => ({ts: p.ts, value: p.session_pnl_dollars || 0}));
    } else if (closedPositions && closedPositions.length > 0) {
        let cum = 0;
        for (const p of closedPositions) {
            cum += (p.pnl_dollar || 0);
            points.push({ts: p.ts, value: cum});
        }
    }
    if (points.length < 2) return '';

    const values = points.map(p => p.value);
    const maxVal = Math.max(...values, 0);
    const minVal = Math.min(...values, 0);
    const range = (maxVal - minVal) || 1;
    const pad = 30;
    const chartW = width - pad;
    const chartH = height - 10;
    const n = values.length;

    let path = '';
    let areaPath = '';
    const zeroY = chartH - ((0 - minVal) / range) * chartH + 5;

    for (let i = 0; i < n; i++) {
        const x = pad + (i / (n - 1)) * chartW;
        const y = chartH - ((values[i] - minVal) / range) * chartH + 5;
        path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
        areaPath += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
    }
    const lastX = pad + chartW;
    const firstX = pad;
    areaPath += `L${lastX.toFixed(1)},${zeroY.toFixed(1)}L${firstX.toFixed(1)},${zeroY.toFixed(1)}Z`;

    const lastVal = values[values.length - 1];
    const lineColor = lastVal >= 0 ? '#00ff88' : '#ff4444';
    const fillColor = lastVal >= 0 ? '#00ff8815' : '#ff444415';

    // Y-axis labels
    let labels = '';
    labels += `<text x="${pad-4}" y="${(chartH - ((maxVal - minVal) / range) * chartH + 5).toFixed(1)}" fill="#666" font-size="10" text-anchor="end">${maxVal >= 0 ? '+' : ''}$${maxVal.toFixed(0)}</text>`;
    labels += `<text x="${pad-4}" y="${zeroY.toFixed(1)}" fill="#666" font-size="10" text-anchor="end">$0</text>`;
    if (minVal < 0) {
        labels += `<text x="${pad-4}" y="${(chartH + 5).toFixed(1)}" fill="#666" font-size="10" text-anchor="end">$${minVal.toFixed(0)}</text>`;
    }
    // Final value label
    const lastY = chartH - ((lastVal - minVal) / range) * chartH + 5;
    labels += `<text x="${(lastX+4).toFixed(1)}" y="${lastY.toFixed(1)}" fill="${lineColor}" font-size="11" font-weight="bold">${lastVal >= 0 ? '+' : ''}$${lastVal.toFixed(0)}</text>`;

    return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="display:block">
        <line x1="${pad}" y1="${zeroY.toFixed(1)}" x2="${lastX}" y2="${zeroY.toFixed(1)}" stroke="#333" stroke-width="1" stroke-dasharray="4,4"/>
        <path d="${areaPath}" fill="${fillColor}"/>
        <path d="${path}" fill="none" stroke="${lineColor}" stroke-width="2"/>
        <circle cx="${lastX.toFixed(1)}" cy="${lastY.toFixed(1)}" r="3" fill="${lineColor}"/>
        ${labels}
    </svg>`;
}

function gateHistogram(allProbs) {
    if (!allProbs || allProbs.length === 0) return '';
    const buckets = new Array(10).fill(0);
    for (const p of allProbs) {
        const idx = Math.min(9, Math.floor(p * 10));
        buckets[idx]++;
    }
    const maxCount = Math.max(...buckets, 1);
    const labels = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'];
    let html = '<div style="font-size:0.7em;color:#666;margin-bottom:2px">Gate P(trade) distribution ('+allProbs.length+' bars)</div><div class="histogram">';
    for (let i = 0; i < 10; i++) {
        const h = Math.max(2, (buckets[i] / maxCount) * 60);
        const color = i < 5 ? '#ff444488' : '#00ff8888';
        html += `<div class="hist-bar" style="height:${h}px;background:${color}" title="${labels[i]}%: ${buckets[i]} bars"><div class="hist-label">${labels[i]}</div></div>`;
    }
    html += '</div>';
    return html;
}

async function refresh() {
    try {
        const resp = await fetch('/api/data');
        const data = await resp.json();

        // Dynamic title
        const s = data.summary || {};
        document.title = `ART² Live — ${s.num_open||0} Open | ${s.num_closed||0} Closed`;

        // Mode badge
        const isDry = s.dry_run;
        document.getElementById('mode-badge').innerHTML = isDry
            ? '<span class="dry-run-badge">DRY RUN</span>'
            : '<span class="live-badge pulse">LIVE</span>';

        // Audit freshness
        let auditInfo = '';
        if (data.audit_mtime) {
            const mtimeDate = new Date(data.audit_mtime);
            const ageSec = (Date.now() - mtimeDate.getTime()) / 1000;
            const cls = ageSec < 120 ? 'audit-fresh' : 'audit-stale';
            const label = ageSec < 120 ? '● live' : `● stale (${Math.floor(ageSec/60)}m ago)`;
            auditInfo = `<span class="${cls}">${label}</span> `;
        }
        const now = new Date();
        document.getElementById('refresh-ts').innerHTML = auditInfo + 'Refreshed ' + now.toLocaleTimeString('en-US',{timeZone:'America/Los_Angeles',hour:'2-digit',minute:'2-digit',second:'2-digit'}) + ' PT';

        // Summary stats
        const pfStr = s.num_closed > 0 ? (s.profit_factor === Infinity ? '∞' : fmt(s.profit_factor, 2)) : '--';
        const pfColor = s.profit_factor > 1.5 ? 'green' : s.profit_factor > 1.0 ? 'yellow' : s.profit_factor > 0 ? 'red' : '';
        document.getElementById('summary-stats').innerHTML = `
            <div class="stat"><div class="label">Open</div><div class="value ${s.num_open > 0 ? 'green pulse' : ''}">${s.num_open || 0}</div></div>
            <div class="stat"><div class="label">Closed</div><div class="value blue">${s.num_closed || 0}</div></div>
            <div class="stat"><div class="label">P&L</div><div class="value ${pnlClass(s.total_pnl_pct)}">${s.num_closed > 0 ? fmtPct(s.total_pnl_pct) : '--'}</div></div>
            <div class="stat"><div class="label">Profit Factor</div><div class="value ${pfColor}">${pfStr}</div></div>
            <div class="stat"><div class="label">Win Rate</div><div class="value ${s.win_rate > 0.5 ? 'green' : s.win_rate > 0 ? 'yellow' : ''}">${s.num_closed > 0 ? fmtPct(s.win_rate) : '--'}</div></div>
            <div class="stat"><div class="label">W / L</div><div class="value">${s.winners || 0} / ${s.losers || 0}</div></div>
            <div class="stat"><div class="label">Avg Hold</div><div class="value">${s.num_closed > 0 ? fmt(s.avg_hold_bars, 1) + ' bars' : '--'}</div></div>
            <div class="stat"><div class="label">Sessions</div><div class="value">${s.num_sessions || 0} (${s.total_minutes || 0}m)</div></div>
            <br>
            <div class="stat-sm"><div class="label">Bars Processed</div><div class="value">${s.total_bars || 0}</div></div>
            <div class="stat-sm"><div class="label">Trade Signals</div><div class="value green">${s.trade_signals || 0}</div></div>
            <div class="stat-sm"><div class="label">Signal Rate</div><div class="value">${fmtPct(s.signal_rate)}</div></div>
            <div class="stat-sm"><div class="label">Entries Applied</div><div class="value">${s.entries_applied || 0}</div></div>
            <div class="stat-sm"><div class="label">Signal→Entry</div><div class="value">${fmtPct(s.conversion_rate)}</div></div>
            <div class="stat-sm"><div class="label">Avg Gate P</div><div class="value">${fmt(s.avg_gate_prob * 100, 1)}%</div></div>
        `;
        // Inject IBKR account balance into summary if available
        const ibkrAcct = data.ibkr && data.ibkr.connected && data.ibkr.account;
        if (ibkrAcct && ibkrAcct.NetLiquidation) {
            const nlv = parseFloat(ibkrAcct.NetLiquidation);
            const avail = ibkrAcct.AvailableFunds ? parseFloat(ibkrAcct.AvailableFunds) : null;
            let acctHtml = `<div class="stat" style="border-left:2px solid #00aaff;padding-left:12px;margin-left:8px"><div class="label">Net Liquidation</div><div class="value blue" style="font-size:1.3em">$${nlv.toLocaleString(undefined,{minimumFractionDigits:0,maximumFractionDigits:0})}</div></div>`;
            if (avail) acctHtml += `<div class="stat"><div class="label">Available</div><div class="value">$${avail.toLocaleString(undefined,{minimumFractionDigits:0,maximumFractionDigits:0})}</div></div>`;
            document.getElementById('summary-stats').innerHTML += acctHtml;
        }

        // Market context
        const tod = todRegime();
        const elapsed = minutesSinceOpen();
        const barsRemaining = Math.max(0, 390 - elapsed);
        const sess = data.current_session;
        const spx = sess ? fmt(sess.seed_spx, 2) : '--';
        const fpInfo = data.feature_projections?.length > 0 ? data.feature_projections[data.feature_projections.length - 1] : null;
        const ctxInfo = data.context_refreshes?.length > 0 ? data.context_refreshes[data.context_refreshes.length - 1] : null;
        document.getElementById('market-context').innerHTML = `
            <div class="stat"><div class="label">SPX (session start)</div><div class="value blue">${spx}</div></div>
            <div class="stat-sm"><div class="label">Market Elapsed</div><div class="value">${elapsed > 0 ? elapsed + ' min' : 'Pre-market'}</div></div>
            <div class="stat-sm"><div class="label">Bars Remaining</div><div class="value ${barsRemaining < 30 ? 'red' : barsRemaining < 60 ? 'yellow' : ''}">${elapsed > 0 && elapsed < 390 ? barsRemaining : '--'}</div></div>
            ${ctxInfo ? `<div class="stat-sm"><div class="label">Context Bars</div><div class="value">${ctxInfo.bars} (${ctxInfo.as_of_date})</div></div>` : ''}
            ${fpInfo ? `<div class="stat-sm"><div class="label">Features</div><div class="value">${fpInfo.context_features}→${fpInfo.model_features} (${fpInfo.mode})</div></div>` : ''}
            ${sess ? `<div class="stat-sm"><div class="label">Account</div><div class="value" style="font-size:0.8em">${sess.account || '—'}</div></div>` : ''}
        `;

        // Current position — IBKR is primary source, audit is fallback
        const ibkr = data.ibkr;
        const ibkrConnected = ibkr && ibkr.connected;
        const ibkrPos = ibkrConnected ? (ibkr.positions || []) : [];
        const ibkrOrders = ibkrConnected ? (ibkr.open_orders || []) : [];
        const openPos = data.open_positions || [];  // audit fallback

        if (ibkrPos.length > 0) {
            // IBKR live position — source of truth
            const ip = ibkrPos[0];
            const livePrice = ip.marketPrice ? ip.marketPrice : null;
            const entryPrice = ip.avgCost / 100;  // IBKR avgCost is per-share * multiplier
            const unrealPnl = ip.unrealizedPNL || 0;
            const unrealPct = entryPrice > 0 && livePrice ? ((livePrice - entryPrice) / entryPrice) : 0;
            const pctColor = unrealPct >= 0 ? '#00ff88' : '#ff4444';
            const pctSign = unrealPct >= 0 ? '+' : '';

            // Try to get audit metadata (stop/TP, gate, direction) if available
            const auditP = openPos.length > 0 ? openPos[openPos.length - 1] : null;
            const stopPrice = auditP ? (auditP.stop_price || 0) : 0;
            const tpPrice = auditP ? (auditP.tp_price || 0) : 0;

            let livePriceHtml = '';
            if (livePrice) {
                livePriceHtml = `<div style="margin-bottom:10px">
                    <span style="font-size:1.4em;font-weight:bold">${fmtDollar(livePrice)}</span>
                    <span style="color:${pctColor};font-size:1.1em;font-weight:bold;margin-left:8px">${pctSign}${(unrealPct * 100).toFixed(1)}%</span>
                    <span style="color:${pctColor};font-size:0.9em;margin-left:8px">(${unrealPnl >= 0 ? '+' : ''}$${unrealPnl.toFixed(0)})</span>
                    <span style="font-size:0.65em;color:#666;margin-left:6px">IBKR live</span>
                </div>`;
            }

            // Stop/TP distances
            let stopDistHtml = fmtDollar(stopPrice), tpDistHtml = fmtDollar(tpPrice);
            if (entryPrice > 0 && stopPrice > 0) {
                const stopDist = ((stopPrice - entryPrice) / entryPrice * 100);
                stopDistHtml = `${fmtDollar(stopPrice)} <span class="dim">(${stopDist.toFixed(0)}%)</span>`;
            }
            if (entryPrice > 0 && tpPrice > 0) {
                const tpDist = ((tpPrice - entryPrice) / entryPrice * 100);
                tpDistHtml = `${fmtDollar(tpPrice)} <span class="dim">(+${tpDist.toFixed(0)}%)</span>`;
            }

            // Progress bar
            let progressHtml = '';
            if (entryPrice > 0 && stopPrice > 0 && tpPrice > 0 && livePrice) {
                const range = tpPrice - stopPrice;
                const pct = range > 0 ? ((livePrice - stopPrice) / range * 100) : 50;
                const clampedPct = Math.max(0, Math.min(100, pct));
                const entryPct = range > 0 ? ((entryPrice - stopPrice) / range * 100) : 50;
                const barColor = pct >= entryPct ? '#00ff88' : '#ff4444';
                progressHtml = `
                    <div style="margin-top:10px">
                        <div style="display:flex;justify-content:space-between;font-size:0.65em;color:#666;margin-bottom:2px">
                            <span>STOP ${fmtDollar(stopPrice)}</span><span>ENTRY ${fmtDollar(entryPrice)}</span><span>TP ${fmtDollar(tpPrice)}</span>
                        </div>
                        <div style="background:#222;height:8px;border-radius:4px;position:relative;overflow:visible">
                            <div style="position:absolute;left:${entryPct.toFixed(1)}%;top:-1px;bottom:-1px;width:2px;background:#666;z-index:1" title="Entry"></div>
                            <div style="width:${clampedPct.toFixed(1)}%;height:100%;background:${barColor};border-radius:4px;transition:width 0.5s"></div>
                        </div>
                    </div>`;
            }

            const entryTs = auditP ? timeCell(auditP.ts) : '--';
            const elapsed = auditP ? elapsedSinceEntry(auditP.ts) : {mins: 0, bars: 0};
            const ageCls = auditP ? ageClass(elapsed.mins) : '';

            document.getElementById('current-position').innerHTML = `
                <div class="position-card">
                    <div class="contract">${formatOptionSymbol(ip.symbol)} <span style="font-size:0.7em;color:#888">qty: ${ip.qty}</span></div>
                    ${livePriceHtml}
                    <div class="detail-grid">
                        ${auditP ? `<div class="stat"><div class="label">Action</div><div class="value">${actionLabel(auditP.action, auditP.action_name)}</div></div>` : ''}
                        <div class="stat"><div class="label">Avg Cost</div><div class="value">${fmtDollar(entryPrice)}</div></div>
                        <div class="stat"><div class="label">Mkt Value</div><div class="value">${fmtDollar(ip.marketValue)}</div></div>
                        ${auditP ? `<div class="stat"><div class="label">Entry Time</div><div class="value" style="font-size:0.85em">${entryTs}</div></div>` : ''}
                        ${auditP ? `<div class="stat"><div class="label">Time Held</div><div class="value ${ageCls}">${elapsed.mins} min (~${elapsed.bars} bars)</div></div>` : ''}
                        ${auditP ? `<div class="stat"><div class="label">Gate P(trade)</div><div class="value">${gateBar(auditP.gate_trade_prob)}</div></div>` : ''}
                        ${stopPrice > 0 ? `<div class="stat"><div class="label">Stop</div><div class="value red">${stopDistHtml}</div></div>` : ''}
                        ${tpPrice > 0 ? `<div class="stat"><div class="label">Take Profit</div><div class="value green">${tpDistHtml}</div></div>` : ''}
                    </div>
                    ${progressHtml}
                    ${auditP && auditP.direction_probs && auditP.direction_probs.length >= 6 ? `<div style="margin-top:8px"><span class="label" style="color:#666;font-size:0.7em">DIRECTION PROBS</span> ${dirBar(auditP.direction_probs)}</div>` : ''}
                </div>
            `;
            document.getElementById('position-card').classList.add('highlight');
        } else if (openPos.length > 0 && !ibkrConnected) {
            // Fallback: no IBKR connection, use audit data
            const p = openPos[openPos.length - 1];
            const contractStr = formatOptionSymbol(p.contract_symbol) || `SPXW ${p.contract_strike} ${p.contract_right}`;
            const elapsed = elapsedSinceEntry(p.ts);
            const ageCls = ageClass(elapsed.mins);
            const entryPrice = p.fill_price || 0;
            const stopPrice = p.stop_price || 0;
            const tpPrice = p.tp_price || 0;

            let stopDistHtml = fmtDollar(stopPrice), tpDistHtml = fmtDollar(tpPrice);
            if (entryPrice > 0 && stopPrice > 0) {
                stopDistHtml = `${fmtDollar(stopPrice)} <span class="dim">(${((stopPrice - entryPrice) / entryPrice * 100).toFixed(0)}%)</span>`;
            }
            if (entryPrice > 0 && tpPrice > 0) {
                tpDistHtml = `${fmtDollar(tpPrice)} <span class="dim">(+${((tpPrice - entryPrice) / entryPrice * 100).toFixed(0)}%)</span>`;
            }

            document.getElementById('current-position').innerHTML = `
                <div class="position-card">
                    <div class="contract">${contractStr} <span style="font-size:0.7em;color:#888">qty: ${p.qty || 1}</span> <span style="font-size:0.6em;color:#ff8800">audit only — no IBKR</span></div>
                    <div class="detail-grid">
                        <div class="stat"><div class="label">Action</div><div class="value">${actionLabel(p.action, p.action_name)}</div></div>
                        <div class="stat"><div class="label">Entry Time</div><div class="value" style="font-size:0.85em">${timeCell(p.ts)}</div></div>
                        <div class="stat"><div class="label">Time Held</div><div class="value ${ageCls}">${elapsed.mins} min (~${elapsed.bars} bars)</div></div>
                        <div class="stat"><div class="label">Gate P(trade)</div><div class="value">${gateBar(p.gate_trade_prob)}</div></div>
                        <div class="stat"><div class="label">Fill Price</div><div class="value">${fmtDollar(entryPrice)}</div></div>
                        ${p.entry_limit ? `<div class="stat"><div class="label">Limit Price</div><div class="value dim">${fmtDollar(p.entry_limit)}</div></div>` : ''}
                        <div class="stat"><div class="label">Stop</div><div class="value red">${stopDistHtml}</div></div>
                        <div class="stat"><div class="label">Take Profit</div><div class="value green">${tpDistHtml}</div></div>
                    </div>
                    ${p.direction_probs && p.direction_probs.length >= 6 ? `<div style="margin-top:8px"><span class="label" style="color:#666;font-size:0.7em">DIRECTION PROBS</span> ${dirBar(p.direction_probs)}</div>` : ''}
                </div>
            `;
            document.getElementById('position-card').classList.add('highlight');
        } else {
            document.getElementById('current-position').innerHTML = ibkrConnected
                ? '<span class="no-data">Flat — no open position (IBKR live)</span>'
                : '<span class="no-data">Flat — no open position</span>';
            document.getElementById('position-card').classList.remove('highlight');
        }

        // IBKR Open Orders
        const ibkrCard = document.getElementById('ibkr-card');
        if (ibkrConnected) {
            if (ibkrOrders.length > 0) {
                ibkrCard.style.display = '';
                document.getElementById('ibkr-status').innerHTML = '<span class="audit-fresh">● Connected</span>';
                let orows = ibkrOrders.map(o => {
                    const price = o.orderType === 'STP' ? fmtDollar(o.auxPrice) : o.orderType === 'LMT' ? fmtDollar(o.lmtPrice) : '—';
                    return `<tr><td>${o.symbol}</td><td>${o.action}</td><td>${o.qty}</td><td>${o.orderType}</td><td>${price}</td><td>${o.status}</td></tr>`;
                }).join('');
                document.getElementById('ibkr-content').innerHTML = `<table><tr><th>Contract</th><th>Side</th><th>Qty</th><th>Type</th><th>Price</th><th>Status</th></tr>${orows}</table>`;
            } else {
                ibkrCard.style.display = 'none';
            }
        } else if (ibkr) {
            ibkrCard.style.display = '';
            document.getElementById('ibkr-status').innerHTML = `<span class="audit-stale">● Disconnected${ibkr.error ? ' — ' + ibkr.error : ''}</span>`;
            document.getElementById('ibkr-content').innerHTML = '<span class="no-data">IBKR Gateway not reachable — running in audit-only mode</span>';
        } else {
            ibkrCard.style.display = 'none';
        }

        // Session history
        const sessions = data.sessions || [];
        const sessionEnds = data.session_ends || {};
        if (sessions.length > 0) {
            let rows = sessions.slice().reverse().map(sess => {
                const end = sessionEnds[sess.session_id];
                const isActive = !end;
                const statusHtml = isActive ? '<span class="session-active pulse">● Active</span>' : '<span class="session-ended">Ended</span>';
                const dur = end ? end.processed_minutes + ' min' : '—';
                const signals = end ? end.signals : '—';
                const entries = end ? end.entries : '—';
                const exits = end ? end.exits : '—';
                const openFinal = end ? end.open_final : '—';
                return `<tr>
                    <td>${statusHtml}</td>
                    <td>${timeCell(sess.ts)}</td>
                    <td>${end ? timeCell(end.ts) : '—'}</td>
                    <td>${dur}</td>
                    <td>${signals}</td>
                    <td>${entries}</td>
                    <td>${exits}</td>
                    <td>${openFinal}</td>
                    <td>${fmt(sess.seed_spx, 0)}</td>
                </tr>`;
            }).join('');
            document.getElementById('session-history').innerHTML = `
                <table><tr><th>Status</th><th>Started</th><th>Ended</th><th>Duration</th><th>Signals</th><th>Entries</th><th>Exits</th><th>Open</th><th>SPX</th></tr>${rows}</table>`;
        }

        // Closed trades
        const positions = data.positions || [];
        if (positions.length > 0) {
            let cumPnl = 0;
            let cumDollar = 0;
            let rows = positions.map((p, i) => {
                cumPnl += p.pnl_pct || 0;
                cumDollar += p.pnl_dollar || 0;
                const rowColor = (p.pnl_pct || 0) > 0 ? 'background:rgba(0,255,136,0.06)' : (p.pnl_pct || 0) < 0 ? 'background:rgba(255,68,68,0.06)' : '';
                const entryTs = p.entry_ts ? timeCell(p.entry_ts) : '--';
                const priceArrow = `${fmtDollar(p.entry_price)} → ${fmtDollar(p.exit_price)}`;
                const dollarPnl = p.pnl_dollar != null ? (p.pnl_dollar >= 0 ? '+' : '') + '$' + Number(p.pnl_dollar).toFixed(0) : '--';
                const cumDollarStr = (cumDollar >= 0 ? '+' : '') + '$' + cumDollar.toFixed(0);
                return `<tr style="${rowColor}">
                    <td>${i+1}</td>
                    <td>${entryTs}</td>
                    <td>${timeCell(p.ts)}</td>
                    <td>${p.direction || ''}</td>
                    <td>${p.strike || ''}</td>
                    <td class="${pnlClass(p.pnl_pct)}">${fmtPct(p.pnl_pct)}</td>
                    <td class="${pnlClass(p.pnl_dollar)}" style="font-weight:bold">${dollarPnl}</td>
                    <td class="${pnlClass(cumPnl)}">${fmtPct(cumPnl)}</td>
                    <td class="${pnlClass(cumDollar)}">${cumDollarStr}</td>
                    <td>${p.bars_held || 0}</td>
                    <td>${reasonPills([p.exit_reason])}</td>
                    <td style="font-size:0.85em">${priceArrow}</td>
                    <td>${p.gate_confidence ? fmt(p.gate_confidence * 100, 0) + '%' : '--'}</td>
                </tr>`;
            }).reverse().join('');
            document.getElementById('trade-history').innerHTML = `
                <table><tr><th>#</th><th>Entry</th><th>Exit</th><th>Dir</th><th>Strike</th><th>P&L %</th><th>P&L $</th><th>Cum %</th><th>Cum $</th><th>Hold</th><th>Exit Reason</th><th>Price</th><th>Gate</th></tr>${rows}</table>`;
        }

        // Entry intents
        const intents = data.entries || [];
        if (intents.length > 0) {
            let rows = intents.slice().reverse().map(e => {
                const gp = e.gate_trade_prob || 0;
                const fillPrice = e.fill_price || 0;
                const limitPrice = e.entry_limit || 0;
                let slippage = '--';
                if (fillPrice > 0 && limitPrice > 0) {
                    slippage = ((fillPrice - limitPrice) / limitPrice * 10000).toFixed(0) + ' bps';
                }
                return `<tr>
                    <td>${timeCell(e.ts)}</td>
                    <td>${actionLabel(e.action, e.action_name)}</td>
                    <td>${e.strike ? e.strike + ' ' + (e.right||'') : '—'}</td>
                    <td>${gateBar(gp)}</td>
                    <td>${e.direction_probs ? dirBar(e.direction_probs) : '—'}</td>
                    <td>${fmtDollar(limitPrice)}</td>
                    <td>${fmtDollar(fillPrice)}</td>
                    <td>${slippage}</td>
                    <td>${fmtDollar(e.ref_price)}</td>
                    <td>${fmtDollar(e.stop_price)}</td>
                    <td>${e.applied ? '<span class="green">✓ FILLED</span>' : '<span class="red">✗</span>'}</td>
                </tr>`;
            }).join('');
            document.getElementById('entry-intents').innerHTML = `
                <table><tr><th>Time</th><th>Action</th><th>Strike</th><th>Gate</th><th>Direction</th><th>Limit$</th><th>Fill$</th><th>Slip</th><th>Ref$</th><th>Stop$</th><th>Status</th></tr>${rows}</table>`;
        }

        // Model behavior: histogram + sparkline + inference table
        const allGateProbs = data.all_gate_probs || [];
        document.getElementById('gate-histogram').innerHTML = gateHistogram(allGateProbs);

        // Equity curve
        const eqEl = document.getElementById('equity-curve');
        const eqW = Math.max(300, eqEl.clientWidth || 600);
        const eqSvg = equityCurveSvg(data.pnl_updates, data.positions, eqW, 120);
        eqEl.innerHTML = eqSvg || '<span class="no-data">Waiting for trades...</span>';

        const recentProbs = allGateProbs.slice(-60);
        const sparkEl = document.getElementById('gate-sparkline');
        const sparkW = Math.max(200, sparkEl.clientWidth || 400);
        sparkEl.innerHTML = recentProbs.length > 1
            ? '<div style="font-size:0.7em;color:#666;margin-bottom:2px">Gate P(trade) last 60 bars — yellow line = 0.5 threshold</div>' + sparklineSvg(recentProbs, sparkW, 40)
            : '';

        const infs = data.inferences || [];
        if (infs.length > 0) {
            let rows = infs.slice().reverse().map(i => {
                const gp = i.gate_trade_prob || 0;
                const isTrade = gp > 0.5;
                const trClass = isTrade ? 'trade-signal' : '';
                const dp = i.direction_probs && i.direction_probs.length >= 6 && isTrade ? dirBar(i.direction_probs) : '';
                return `<tr class="${trClass}">
                    <td>${timeCell(i.ts)}</td>
                    <td>${actionLabel(i.action, i.action_name)}</td>
                    <td>${gateBar(gp)}</td>
                    <td>${dp}</td>
                    <td>${reasonPills(i.reason_codes)}</td>
                </tr>`;
            }).join('');
            document.getElementById('inference-stream').innerHTML = `
                <table><tr><th>Time</th><th>Action</th><th>Gate</th><th>Direction</th><th>Reason</th></tr>${rows}</table>`;
        }

        // Bar quality
        const bars = data.bar_snapshots || [];
        if (bars.length > 0) {
            let rows = bars.slice().reverse().map(b => {
                const pct = (b.completeness * 100).toFixed(0);
                const color = b.completeness >= 0.9 ? 'green' : b.completeness >= 0.75 ? 'yellow' : 'red';
                const missingStr = b.missing_names && b.missing_names.length > 0
                    ? b.missing_names.map(n => `<span class="pill pill-red">${n}</span>`).join(' ')
                    : '<span class="dim">none</span>';
                let stalenessStr = '';
                if (b.staleness && typeof b.staleness === 'object') {
                    stalenessStr = Object.entries(b.staleness).map(([k, v]) =>
                        `<span class="${stalenessClass(v)}">${k}: ${typeof v === 'number' ? v.toFixed(1) : v}s</span>`
                    ).join(' ');
                } else if (typeof b.staleness === 'number') {
                    stalenessStr = `<span class="${stalenessClass(b.staleness)}">${b.staleness.toFixed(1)}s</span>`;
                }
                return `<tr>
                    <td>${timeCell(b.ts)}</td>
                    <td><span class="${color}">${pct}%</span> (${b.present}/${b.present + b.missing})</td>
                    <td>${missingStr}</td>
                    <td style="font-size:0.75em">${stalenessStr || '—'}</td>
                </tr>`;
            }).join('');
            document.getElementById('bar-quality').innerHTML = `
                <table><tr><th>Time</th><th>Completeness</th><th>Missing Features</th><th>Option Staleness</th></tr>${rows}</table>`;
        }

    } catch (e) {
        console.error('Refresh failed:', e);
    }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


# ── HTTP Server ──────────────────────────────────────────────────────────

class DashHandler(BaseHTTPRequestHandler):
    audit_path = DEFAULT_AUDIT
    ibkr_reader: IBKRDashReader | None = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._respond(200, "text/html", render_html())
        elif self.path == "/api/data":
            data = parse_audit(self.audit_path)
            if self.ibkr_reader:
                data["ibkr"] = self.ibkr_reader.snapshot()
            self._respond(200, "application/json", json.dumps(data))
        else:
            self._respond(404, "text/plain", "Not found")

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)


def main():
    parser = argparse.ArgumentParser(description="ART² Live Trading Dashboard")
    parser.add_argument("--audit", type=str, default=str(DEFAULT_AUDIT),
                        help="Path to audit.jsonl")
    parser.add_argument("--port", type=int, default=8421,
                        help="HTTP port (default: 8421)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    parser.add_argument("--ibkr", action="store_true",
                        help="Connect to IBKR for live position verification")
    parser.add_argument("--ibkr-port", type=int, default=4002,
                        help="IB Gateway port (default: 4002)")
    parser.add_argument("--ibkr-client-id", type=int, default=80,
                        help="IBKR client ID for dashboard (default: 80)")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    DashHandler.audit_path = audit_path

    ibkr_reader = None
    if args.ibkr:
        if not HAS_IB:
            print("[live_dash] WARNING: --ibkr requested but ib_insync not installed. Running without IBKR.")
        else:
            ibkr_reader = IBKRDashReader(port=args.ibkr_port, client_id=args.ibkr_client_id)
            ibkr_reader.start()
            print(f"[live_dash] IBKR verification enabled (port={args.ibkr_port}, client_id={args.ibkr_client_id})")
    DashHandler.ibkr_reader = ibkr_reader

    server = HTTPServer(("0.0.0.0", args.port), DashHandler)
    print(f"[live_dash] Serving on http://localhost:{args.port}")
    print(f"[live_dash] Reading: {audit_path}")
    print(f"[live_dash] Auto-refresh: 5s | Times shown in ET (market) + PT (local)")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if ibkr_reader:
            ibkr_reader.stop()
        print("\n[live_dash] Stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
