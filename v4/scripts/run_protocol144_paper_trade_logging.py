"""Protocol 144: paper trade logging contract.

This validates the append-only JSONL/CSV trade journal that future live paper
sessions will use for analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v4.live.paper_trade_log import (
    append_trade_event,
    export_trade_log_csv,
    load_trade_log,
    make_trade_log_event,
    validate_trade_log,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_144_paper_trade_logging")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "sample_protocol101_paper_trade_log.jsonl"
    csv_path = args.out_dir / "sample_protocol101_paper_trade_log.csv"
    if jsonl_path.exists():
        jsonl_path.unlink()
    for event in sample_events():
        append_trade_event(jsonl_path, event)
    rows = load_trade_log(jsonl_path)
    validation = validate_trade_log(rows)
    csv_summary = export_trade_log_csv(jsonl_path, csv_path)
    decision = "pass_paper_trade_logging_ready_for_live_paper_analysis" if validation["status"] == "pass" else "blocked_paper_trade_log_schema_failed"
    payload = {
        "protocol": "144_paper_trade_logging",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": False,
        "broker_order_endpoint_called": False,
        "validation": validation,
        "outputs": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "csv_summary": csv_summary,
            "report": str(args.out_dir / "report.md"),
        },
        "next_gate": next_gate(decision),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if decision.startswith("pass_") else 1


def sample_events() -> list[dict[str, Any]]:
    common = {
        "session": "2026-03-20",
        "run_id": "protocol144_sample",
        "trade_uid": "protocol101-20260320-6700-C-BUY",
        "selected_contract": {
            "symbol": "SPX",
            "root": "SPXW",
            "trading_class": "SPXW",
            "expiry": "20260320",
            "strike": 6700.0,
            "right": "C",
            "exchange": "SMART",
            "currency": "USD",
        },
        "order": {"action": "BUY", "quantity": 1, "limit_price": 10.0},
        "account": {
            "account_id_redacted": "DU***40",
            "starting_cash": 10_000.0,
            "cash": 10_000.0,
            "equity": 10_000.0,
            "realized_daily_pnl": 0.0,
            "open_positions": 0,
        },
        "market_snapshot": {
            "underlying": {"spx": 6700.0, "vix": 16.0, "spx_timestamp": "2026-03-20T14:35:00+00:00"},
            "option_nbbo": {"bid": 9.9, "ask": 10.0, "bid_size": 10, "ask_size": 12, "quote_age_ms": 100},
            "context": {"context_age_ms": 100},
        },
        "model_decision": {"action": "enter", "score": 3.1, "threshold": 1.2, "reason": "sample"},
        "risk_gate": {"passed": True, "reason": "pass"},
    }
    return [
        make_trade_log_event(event_type="model_decision", timestamp="2026-03-20T14:35:00+00:00", **common),
        make_trade_log_event(event_type="risk_gate", timestamp="2026-03-20T14:35:00+00:00", **common),
        make_trade_log_event(event_type="paper_order_dry_run", timestamp="2026-03-20T14:35:01+00:00", **common),
    ]


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Wire the live Protocol101 router to append model_decision, risk_gate, order, fill, exit, and account_state "
            "events to this JSONL stream during every paper session."
        )
    return "Fix the paper trade log schema before enabling paper sessions."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    validation = payload["validation"]
    lines = [
        "# Protocol 144: Paper Trade Logging",
        "",
        "No paid data was downloaded. No order endpoint was called. This validates the log contract only.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Rows: `{validation['rows']}`",
        f"- Status: `{validation['status']}`",
        f"- Event counts: `{validation['event_counts']}`",
        f"- JSONL: `{payload['outputs']['jsonl']}`",
        f"- CSV: `{payload['outputs']['csv']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 144 Paper Trade Logging"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Added and validated the append-only Protocol101 paper trade log contract.
Reason: User requested live/paper trades be logged for later analysis, so every future paper session needs a durable JSONL source of truth plus CSV export.
Data Used: Synthetic local sample events only. No paid data was downloaded, no broker order endpoint was called, and no paper order was submitted.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Rows={payload['validation']['rows']}; event_counts={payload['validation']['event_counts']}. Report: {report_path}
Next Gate: Wire live Protocol101 decisions/orders/fills/exits/account state into this trade journal during paper sessions.
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


if __name__ == "__main__":
    raise SystemExit(main())
