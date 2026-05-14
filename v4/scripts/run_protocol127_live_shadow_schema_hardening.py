"""Protocol 127: expanded no-order live shadow schema hardening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_shadow_schema import (
    EVENT_TYPES,
    SHADOW_SCHEMA_VERSION,
    schema_contract,
    validate_shadow_stream,
)


DEFAULT_CHARTS_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_127_protocol101_live_shadow_schema_hardening")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts-dir", type=Path, default=DEFAULT_CHARTS_DIR)
    parser.add_argument("--input-shadow-log", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(args.charts_dir / "trades.csv")
    examples = build_examples(trades[0])
    example_validation = validate_shadow_stream(examples)
    live_rows = load_jsonl(args.input_shadow_log) if args.input_shadow_log else []
    live_validation = validate_shadow_stream(live_rows) if live_rows else None
    decision = decide(example_validation, live_validation)
    payload = {
        "protocol": "127_protocol101_live_shadow_schema_hardening",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "schema": schema_contract(),
        "example_validation": example_validation,
        "input_shadow_log": str(args.input_shadow_log) if args.input_shadow_log else None,
        "input_shadow_validation": live_validation,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "schema_contract": str(args.out_dir / "protocol101_shadow_v2_schema.json"),
            "examples": str(args.out_dir / "protocol101_shadow_v2_examples.jsonl"),
        },
        "next_gate": (
            "Tuesday live shadow capture must emit this schema and pass validation before order-state rehearsal "
            "or any request for paper-order approval."
        ),
    }
    (args.out_dir / "protocol101_shadow_v2_schema.json").write_text(json_dumps(payload["schema"]))
    (args.out_dir / "protocol101_shadow_v2_examples.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in examples) + "\n"
    )
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload)
    print(json.dumps({"decision": decision, "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"no trades found in {path}")
    frame = frame.sort_values(["decision_time", "trade_number"]).reset_index(drop=True)
    return [clean(row) for row in frame.to_dict("records")]


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clean(row: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def build_examples(trade: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for event_type, action, gate_passed, reason in (
        ("market_snapshot", "wait", True, ""),
        ("candidate_set", "wait", True, ""),
        ("model_decision", "enter", True, ""),
        ("risk_gate", "blocked", False, "stale_option_quote"),
        ("paper_account_state", "hold", True, ""),
        ("exit_decision", "exit", True, ""),
    ):
        rows.append(example_event(trade, event_type=event_type, action=action, gate_passed=gate_passed, reason=reason))
    return rows


def example_event(
    trade: dict[str, Any],
    *,
    event_type: str,
    action: str,
    gate_passed: bool,
    reason: str,
) -> dict[str, Any]:
    is_exit = event_type == "exit_decision"
    timestamp = str(trade["exit_time"] if is_exit else trade["decision_time"])
    bid = float(trade["exit_bid"] if is_exit else trade["entry_bid"])
    ask = float(trade["exit_ask"] if is_exit else trade["entry_ask"])
    selected_contract = None if action in {"wait", "blocked"} else {
        "contract_id": str(trade["contract_id"]),
        "root": "SPXW",
        "settlement_style": "PM",
        "right": str(trade["right"]),
        "side": str(trade["side"]),
        "offset": float(trade["offset"]),
        "quantity": 1,
        "multiplier": 100.0,
    }
    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "protocol_id": "protocol101",
        "event_type": event_type,
        "timestamp": timestamp,
        "session": str(trade["session"]),
        "live_orders_enabled": False,
        "broker_endpoint_called": False,
        "market_snapshot": {
            "underlying": {
                "spx": float(trade["exit_spx"] if is_exit else trade["entry_spx"]),
                "spx_timestamp": timestamp,
                "spx_source": "historical_example",
                "vix": 20.0,
                "vix_timestamp": timestamp,
                "vix_source": "historical_example",
                "context_age_ms": 0,
            },
            "option_nbbo": {
                "contract_id": str(trade["contract_id"]),
                "bid": bid,
                "ask": ask,
                "bid_size": float(trade["entry_bid_size"] or 1),
                "ask_size": float(trade["entry_ask_size"] or 1),
                "timestamp": timestamp,
                "quote_age_ms": 0,
                "source": "historical_example",
            },
        },
        "candidate_set": {
            "candidate_count": 1,
            "spxw_only": True,
            "pm_settled_only": True,
        },
        "model_decision": {
            "score": float(trade["score"]),
            "threshold": float(trade["threshold"]),
            "features": {"entry_ask": float(trade["entry_ask"]), "entry_bid": float(trade["entry_bid"])},
            "artifact": "frozen_protocol101",
        },
        "selected_action": action,
        "selected_contract": selected_contract,
        "risk_gate": {
            "passed": bool(gate_passed),
            "reason": reason,
            "reasons": [] if gate_passed else [reason],
        },
        "paper_account_state": {
            "starting_cash": 10_000.0,
            "cash": float(trade["paper_cash_before"]),
            "equity": float(trade["paper_cash_before"]),
            "daily_realized_pnl": 0.0,
            "open_positions": 1 if action == "hold" else 0,
            "max_concurrent_positions": 1,
            "premium_required": float(trade["premium_paid"]),
            "affordable": True,
        },
    }


def decide(example_validation: dict[str, Any], live_validation: dict[str, Any] | None) -> str:
    if example_validation.get("status") != "pass":
        return "blocked_schema_examples_failed"
    if live_validation is not None and live_validation.get("status") != "pass":
        return "blocked_input_shadow_schema_failed"
    if live_validation is not None:
        return "pass_live_shadow_schema_validated"
    return "pass_schema_hardening_ready_for_live_capture"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    example = payload["example_validation"]
    live = payload["input_shadow_validation"]
    lines = [
        "# Protocol 127: Protocol101 Live Shadow Schema Hardening",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Schema version: `{payload['schema']['schema_version']}`",
        f"- Event types: `{', '.join(EVENT_TYPES)}`",
        f"- Example validation: `{example['status']}` over `{example['rows']}` rows",
        f"- Input live-shadow validation: `{live['status'] if live else 'not_run'}`",
        "",
        "## Outputs",
        "",
        f"- Schema: `{payload['outputs']['schema_contract']}`",
        f"- Examples: `{payload['outputs']['examples']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 127 Protocol101 Live Shadow Schema Hardening"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Defined and validated the expanded Protocol101 no-order live shadow schema.
Reason: Tuesday needs to capture all decisions, risk blocks, market snapshots, and account state before any paper-order rehearsal.
Data Used: Existing Protocol113 trade replay only for examples. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Schema {payload['outputs']['schema_contract']}; examples {payload['outputs']['examples']}.
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    existing = path.read_text() if path.exists() else ""
    if marker not in existing:
        path.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
