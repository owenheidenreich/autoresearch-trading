"""AUDIT_FILL_MODEL_READINESS_V1.

Inventory live-shadow and paper logs to decide whether v4 has enough observed
order/fill evidence to calibrate a fill model. If not, the audit explicitly
blocks calibration and keeps deterministic ask/bid plus stress replay as the
research fill assumption.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.foundation.execution_observation_contract import (
    coverage_for_frame,
    default_execution_observation_contract,
)


ROLE_LABEL = "AUDIT_FILL_MODEL_READINESS_V1"
HISTORICAL_ID = "Protocol272"
DEFAULT_LOG_ROOT = Path("v4/audit")
DEFAULT_PAPER_LOG_ROOT = Path("v4/logs/paper_trading")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness")
MIN_FILL_OBSERVATIONS = 30
MIN_EXECUTION_OBSERVATIONS = 8
MIN_ROUND_TRIP_FILLS = 1
PAPER_LIVE_LOG_TOKENS = ("paper", "live", "runtime", "shadow", "fill", "order", "ibkr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--paper-log-root", type=Path, default=DEFAULT_PAPER_LOG_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-fill-observations", type=int, default=MIN_FILL_OBSERVATIONS)
    parser.add_argument("--min-execution-observations", type=int, default=MIN_EXECUTION_OBSERVATIONS)
    parser.add_argument("--min-round-trip-fills", type=int, default=MIN_ROUND_TRIP_FILLS)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.log_root)
    rows.extend(collect_csv_rows(args.paper_log_root))
    frame = pd.DataFrame(rows)
    fills = extract_fill_observations(frame)
    if not fills.empty:
        fills.to_csv(args.out_dir / "fill_observations.csv", index=False)
    if not frame.empty:
        frame.to_csv(args.out_dir / "live_and_paper_event_inventory.csv", index=False)
    readiness = summarize_readiness(
        frame,
        fills,
        min_fill_observations=int(args.min_fill_observations),
        min_execution_observations=int(args.min_execution_observations),
        min_round_trip_fills=int(args.min_round_trip_fills),
    )
    packet_coverage = coverage_for_frame(frame)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / fill-model calibration readiness audit",
        "changes_paper_default": False,
        "candidate_label": "ALL_RESEARCH_CHALLENGERS",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": {"jsonl_log_root": str(args.log_root), "paper_csv_log_root": str(args.paper_log_root)},
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "row_counts": {
            "jsonl_events_read": int(len(frame)),
            "fill_observations": int(len(fills)),
            "log_files": int(frame["source_file"].nunique()) if not frame.empty else 0,
        },
        "readiness": readiness,
        "execution_observation_contract": default_execution_observation_contract().to_dict(),
        "execution_observation_packet_coverage": packet_coverage,
        "decision": decide(readiness),
        "next_experiment": next_experiment(readiness),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = ["**/*.jsonl"]
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not should_scan_jsonl(path):
                continue
            try:
                with path.open() as handle:
                    for line_no, line in enumerate(handle, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rows.append(flatten_event(payload, path, line_no))
            except OSError:
                continue
    return rows


def should_scan_jsonl(path: Path) -> bool:
    text = str(path).lower()
    if "databento" in text:
        return False
    return any(token in text for token in PAPER_LIVE_LOG_TOKENS)


def collect_csv_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("**/*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for line_no, row in frame.iterrows():
            rows.append(
                {
                    "source_file": str(path),
                    "line_no": int(line_no) + 2,
                    "event_type": str(row.get("event_type", "")),
                    "protocol_id": str(row.get("protocol_id", "")),
                    "session": str(row.get("session", "")),
                    "timestamp": str(row.get("timestamp", row.get("decision_emitted_at", ""))),
                    "decision_action": str(row.get("model_action", row.get("action", ""))),
                    "decision_reason": str(row.get("blocked_reason", row.get("risk_reason", ""))),
                    "position_state": "",
                    "contract_id": str(row.get("contract_id", "")),
                    "bid": finite(row.get("bid")),
                    "ask": finite(row.get("ask")),
                    "bid_size": finite(row.get("bid_size")),
                    "ask_size": finite(row.get("ask_size")),
                    "spread": finite(row.get("ask")) - finite(row.get("bid")) if math.isfinite(finite(row.get("ask"))) and math.isfinite(finite(row.get("bid"))) else math.nan,
                    "quote_gap_seconds": finite(row.get("quote_age_ms")) / 1000.0 if math.isfinite(finite(row.get("quote_age_ms"))) else math.nan,
                    "predicted_value": finite(row.get("model_score")),
                    "order_status": str(row.get("order_status", "")),
                    "order_side": str(row.get("action", "")),
                    "order_limit_price": finite(row.get("limit_price")),
                    "order_submit_time": str(row.get("broker_submit_at", "")),
                    "fill_status": str(row.get("order_status", "")),
                    "fill_price": finite(row.get("avg_fill_price")),
                    "fill_time": str(row.get("entry_fill_at", row.get("exit_fill_at", ""))),
                }
            )
    return rows


def flatten_event(payload: dict[str, Any], path: Path, line_no: int) -> dict[str, Any]:
    nbbo = payload.get("nbbo") if isinstance(payload.get("nbbo"), dict) else {}
    market = payload.get("market_snapshot") if isinstance(payload.get("market_snapshot"), dict) else {}
    option = market.get("option_nbbo") if isinstance(market.get("option_nbbo"), dict) else {}
    contract = payload.get("selected_contract") if isinstance(payload.get("selected_contract"), dict) else {}
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    model = payload.get("model") if isinstance(payload.get("model"), dict) else payload.get("model_decision", {})
    order = payload.get("order") if isinstance(payload.get("order"), dict) else payload.get("order_event", {})
    fill = payload.get("fill") if isinstance(payload.get("fill"), dict) else payload.get("fill_event", {})
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    bid = finite(payload.get("bid", nbbo.get("bid", option.get("bid", features.get("bid")))))
    ask = finite(payload.get("ask", nbbo.get("ask", option.get("ask", features.get("ask")))))
    spread = finite(payload.get("spread", features.get("spread")))
    if not math.isfinite(spread) and math.isfinite(bid) and math.isfinite(ask):
        spread = ask - bid
    fill_status = str(payload.get("fill_status", fill.get("status", payload.get("order_status", "")))) if isinstance(fill, dict) else str(payload.get("fill_status", payload.get("order_status", "")))
    order_status = str(order.get("status", payload.get("order_status", ""))) if isinstance(order, dict) else str(payload.get("order_status", ""))
    order_limit = finite(payload.get("submitted_limit", order.get("limit_price", payload.get("order_limit_price"))) if isinstance(order, dict) else payload.get("submitted_limit", payload.get("order_limit_price")))
    return {
        "source_file": str(path),
        "line_no": int(line_no),
        "event_type": str(payload.get("event_type", payload.get("type", ""))),
        "protocol_id": str(payload.get("protocol_id", "")),
        "session": str(payload.get("session", "")),
        "timestamp": str(payload.get("timestamp", payload.get("decision_time", ""))),
        "decision_timestamp": str(payload.get("decision_timestamp", payload.get("timestamp", payload.get("decision_time", "")))),
        "raw_quote_timestamp": str(payload.get("raw_quote_timestamp", payload.get("quote_timestamp", option.get("quote_timestamp", "")))),
        "received_timestamp": str(payload.get("received_timestamp", option.get("received_timestamp", ""))),
        "decision_action": str(decision.get("action", payload.get("selected_action", ""))),
        "decision_reason": str(decision.get("reason", payload.get("blocked_reason", ""))),
        "position_state": str(payload.get("position_state", "")),
        "contract_id": str(payload.get("contract_id", contract.get("contract_id", ""))),
        "bid": bid,
        "ask": ask,
        "bid_size": finite(payload.get("bid_size", option.get("bid_size", features.get("bid_size")))),
        "ask_size": finite(payload.get("ask_size", option.get("ask_size", features.get("ask_size")))),
        "spread": spread,
        "quote_age_ms": finite(payload.get("quote_age_ms", option.get("quote_age_ms"))),
        "quote_gap_seconds": finite(payload.get("quote_gap_seconds", features.get("quote_gap_seconds"))),
        "premium": finite(payload.get("premium", ask * 100.0 if math.isfinite(ask) else math.nan)),
        "side": str(payload.get("side", contract.get("right", payload.get("order_side", "")))),
        "moneyness": finite(payload.get("moneyness", payload.get("offset_points", payload.get("offset")))),
        "time_bucket": str(payload.get("time_bucket", payload.get("bucket", ""))),
        "intended_ask_entry": finite(payload.get("intended_ask_entry", payload.get("reference_ask", ask))),
        "submitted_limit": order_limit,
        "cancel_status": str(payload.get("cancel_status", payload.get("order_cancel_status", ""))),
        "timeout_status": str(payload.get("timeout_status", payload.get("order_timeout_status", ""))),
        "latency_ms": finite(payload.get("latency_ms", payload.get("latency_seconds"))),
        "exit_bid": finite(payload.get("exit_bid", bid)),
        "post_fill_pnl": finite(payload.get("post_fill_pnl", payload.get("pnl", payload.get("realized_pnl")))),
        "predicted_value": finite(model.get("predicted_continuation_value", model.get("score"))),
        "order_status": order_status,
        "order_side": str(order.get("side", payload.get("order_side", ""))) if isinstance(order, dict) else str(payload.get("order_side", "")),
        "order_limit_price": order_limit,
        "order_submit_time": str(order.get("submit_time", payload.get("order_submit_time", ""))) if isinstance(order, dict) else str(payload.get("order_submit_time", "")),
        "fill_status": fill_status,
        "fill_price": finite(fill.get("price", payload.get("fill_price", payload.get("entry_fill_price")))) if isinstance(fill, dict) else finite(payload.get("fill_price", payload.get("entry_fill_price"))),
        "fill_time": str(fill.get("time", payload.get("fill_time", ""))) if isinstance(fill, dict) else str(payload.get("fill_time", "")),
        "exit_fill_status": str(payload.get("exit_fill_status", "")),
        "exit_fill_price": finite(payload.get("exit_fill_price")),
        "open_position_risk": bool(payload.get("open_position_risk", False)),
    }


def extract_fill_observations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    fill_mask = frame["fill_price"].map(math.isfinite) | frame["fill_status"].astype(str).str.lower().isin(["filled", "partial", "cancelled", "canceled", "timeout", "missed"])
    fills = frame[fill_mask].copy()
    if fills.empty:
        return fills
    fills["filled"] = fills["fill_price"].map(math.isfinite) & fills["fill_status"].astype(str).str.lower().isin(["", "filled", "partial", "fill"])
    fills["mid_at_decision"] = (pd.to_numeric(fills["bid"], errors="coerce") + pd.to_numeric(fills["ask"], errors="coerce")) / 2.0
    fills["entry_buy_slippage_vs_ask"] = pd.to_numeric(fills["fill_price"], errors="coerce") - pd.to_numeric(fills["ask"], errors="coerce")
    fills["exit_sell_slippage_vs_bid"] = pd.to_numeric(fills["bid"], errors="coerce") - pd.to_numeric(fills["fill_price"], errors="coerce")
    fills["latency_seconds"] = (
        pd.to_datetime(fills["fill_time"], utc=True, errors="coerce") - pd.to_datetime(fills["order_submit_time"], utc=True, errors="coerce")
    ).dt.total_seconds()
    return fills


def summarize_readiness(
    frame: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    min_fill_observations: int,
    min_execution_observations: int = MIN_EXECUTION_OBSERVATIONS,
    min_round_trip_fills: int = MIN_ROUND_TRIP_FILLS,
) -> dict[str, Any]:
    action_counts = frame["decision_action"].value_counts(dropna=False).to_dict() if not frame.empty else {}
    if fills.empty:
        return {
            "status": "blocked_insufficient_fill_observations",
            "reason": "no paper/live fill rows found in existing logs",
            "fill_observations": 0,
            "required_fill_observations": int(min_fill_observations),
            "execution_observations": 0,
            "required_execution_observations": int(min_execution_observations),
            "round_trip_fills": 0,
            "required_round_trip_fills": int(min_round_trip_fills),
            "deterministic_replay_assumption": "ask_entry_bid_exit_plus_slippage_stress",
            "decision_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        }
    filled = fills[fills["filled"].astype(bool)]
    round_trip = fills[
        fills["fill_status"].astype(str).str.lower().eq("filled")
        & fills.get("exit_fill_status", pd.Series(index=fills.index, dtype=str)).astype(str).str.lower().eq("filled")
    ]
    fill_rate = float(len(filled) / len(fills)) if len(fills) else 0.0
    enough_for_calibration = int(len(fills)) >= int(min_fill_observations)
    enough_for_truth_packet = int(len(fills)) >= int(min_execution_observations) and int(len(round_trip)) >= int(min_round_trip_fills)
    status = (
        "ready_for_simple_empirical_fill_model"
        if enough_for_calibration
        else "execution_truth_packet_ready_for_conservative_fill_stress"
        if enough_for_truth_packet
        else "blocked_insufficient_fill_observations"
    )
    return {
        "status": status,
        "fill_observations": int(len(fills)),
        "filled_observations": int(len(filled)),
        "required_fill_observations": int(min_fill_observations),
        "execution_observations": int(len(fills)),
        "required_execution_observations": int(min_execution_observations),
        "round_trip_fills": int(len(round_trip)),
        "required_round_trip_fills": int(min_round_trip_fills),
        "fill_rate": fill_rate,
        "median_buy_slippage_vs_ask": finite(filled["entry_buy_slippage_vs_ask"].median()) if not filled.empty else None,
        "median_exit_slippage_vs_bid": finite(filled["exit_sell_slippage_vs_bid"].median()) if not filled.empty else None,
        "median_latency_seconds": finite(filled["latency_seconds"].median()) if not filled.empty else None,
        "decision_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "deterministic_replay_assumption": "eligible_for_calibration_but_not_auto_enabled" if enough_for_calibration else "ask_entry_bid_exit_plus_slippage_stress",
    }


def decide(readiness: dict[str, Any]) -> str:
    if readiness.get("status") == "ready_for_simple_empirical_fill_model":
        return "fill_model_calibration_ready_but_requires_separate_validation"
    if readiness.get("status") == "execution_truth_packet_ready_for_conservative_fill_stress":
        return "execution_truth_packet_ready_keep_stress_replay"
    return "blocked_insufficient_fill_observations_keep_stress_replay"


def next_experiment(readiness: dict[str, Any]) -> str:
    if readiness.get("status") == "ready_for_simple_empirical_fill_model":
        return "Fit a simple fill-probability/slippage model, then compare deterministic ask/bid, stress, and calibrated replay."
    if readiness.get("status") == "execution_truth_packet_ready_for_conservative_fill_stress":
        return "Use the paper execution packet to falsify obvious stale/fill artifacts; keep conservative stress replay until more observations exist."
    return "Keep collecting paper fill logs; do not silently assume calibrated fills."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    readiness = payload["readiness"]
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Readiness",
        "",
        f"- Status: `{readiness['status']}`",
        f"- Fill observations: `{readiness['fill_observations']}`",
        f"- Required observations: `{readiness['required_fill_observations']}`",
        f"- Execution observations: `{readiness.get('execution_observations', 0)}` / `{readiness.get('required_execution_observations', 0)}`",
        f"- Round-trip fills: `{readiness.get('round_trip_fills', 0)}` / `{readiness.get('required_round_trip_fills', 0)}`",
        f"- Replay assumption: `{readiness['deterministic_replay_assumption']}`",
        f"- Observation packet coverage: `{payload['execution_observation_packet_coverage']['status']}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    if marker in ledger.read_text():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


if __name__ == "__main__":
    raise SystemExit(main())
