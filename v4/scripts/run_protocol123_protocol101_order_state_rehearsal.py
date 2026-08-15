"""Protocol 123: Protocol101 $10k order-state rehearsal.

This is not live or paper trading. It converts the frozen Protocol101
paper-account replay into auditable order-state records using the v4 order
state machine. The capital baseline is explicitly $10,000; the $500 IBKR cash
is only an account/data-access reserve.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.export_protocol101_trade_charts import (
    DEFAULT_NORMALIZED_DIRS,
    DEFAULT_PROTOCOL101_DIR,
    DEFAULT_PROTOCOL107_DIR,
    DEFAULT_SPX_DIR,
    build_paper_account_trades,
    none_or_float,
    premium_dollars,
)
from v4.scripts.run_protocol122_protocol101_capital_realism import (
    IBKR_ACCESS_RESERVE,
    TRADING_CAPITAL_BASELINE,
    _load_research_trades,
)
from v4.sim.order_state import OrderRecord, OrderState


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_123_protocol101_order_state_rehearsal")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_STRESS_PER_SIDE = (0.0, 0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--protocol107-dir", type=Path, default=DEFAULT_PROTOCOL107_DIR)
    parser.add_argument("--spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--normalized-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--paper-seed", type=int, default=1)
    parser.add_argument("--starting-cash", type=float, default=TRADING_CAPITAL_BASELINE)
    parser.add_argument("--stress-per-side", action="append", type=float, default=None)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalized_dirs = tuple(args.normalized_dir or DEFAULT_NORMALIZED_DIRS)
    stress_values = tuple(args.stress_per_side or DEFAULT_STRESS_PER_SIDE)

    research_trades = _load_research_trades(
        protocol101_dir=args.protocol101_dir,
        protocol107_dir=args.protocol107_dir,
        protocol112_dir=None,
        spx_dir=args.spx_dir,
        normalized_dirs=normalized_dirs,
    )
    paper_trades, skipped = build_paper_account_trades(
        research_trades,
        starting_equity=float(args.starting_cash),
        paper_seed=int(args.paper_seed),
    )
    order_rows, order_summary = build_order_state_rows(paper_trades)
    stress_rows = [stress_replay(paper_trades, starting_cash=float(args.starting_cash), stress_per_side=value) for value in stress_values]
    decision = decide(order_summary, stress_rows, skipped=skipped, starting_cash=float(args.starting_cash))
    payload = {
        "protocol": "123_protocol101_order_state_rehearsal",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "paper_seed": int(args.paper_seed),
        "capital_assumption": {
            "ibkr_access_reserve": IBKR_ACCESS_RESERVE,
            "paper_trading_starting_equity": float(args.starting_cash),
            "intended_real_money_bankroll": TRADING_CAPITAL_BASELINE,
        },
        "source_trade_count": len([row for row in research_trades if int(row.get("seed", -1)) == int(args.paper_seed)]),
        "paper_trade_count": len(paper_trades),
        "skipped_trade_count": len(skipped),
        "skipped_reasons": _skip_reasons(skipped),
        "order_state": order_summary,
        "stress_results": stress_rows,
        "next_gate": next_gate(decision),
    }

    order_rows_path = args.out_dir / "order_state_rows.json"
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    order_rows_path.write_text(json.dumps(order_rows, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(report_path, payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, report_path)
    print(json.dumps({"decision": decision, "report": str(report_path)}, indent=2, sort_keys=True))
    return 0


def build_order_state_rows(trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_state_counts: dict[str, int] = {}
    transition_counts: dict[str, int] = {}
    errors: list[str] = []
    for index, trade in enumerate(trades, start=1):
        try:
            record = build_order_record(trade, order_id=f"protocol101-paper10k-{index}")
        except Exception as exc:
            errors.append(f"{trade.get('candidate_uid', index)}: {exc}")
            continue
        final = record.final_state.value if record.final_state else "missing"
        final_state_counts[final] = final_state_counts.get(final, 0) + 1
        for event in record.history:
            key = f"{event.from_state.value}->{event.to_state.value}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        rows.append(order_row(record, trade))
    summary = {
        "records": len(rows),
        "errors": errors,
        "final_state_counts": final_state_counts,
        "transition_counts": transition_counts,
        "all_exit_filled": bool(rows and final_state_counts == {OrderState.EXIT_FILLED.value: len(rows)}),
        "all_one_contract": bool(rows and all(row["intended_size"] == 1 for row in rows)),
        "all_spxw": bool(rows and all(str(row["contract_id"]).startswith("SPXW-") for row in rows)),
        "max_concurrent_positions": max_concurrent_positions(trades),
    }
    return rows, summary


def build_order_record(trade: dict[str, Any], *, order_id: str) -> OrderRecord:
    decision_time = timestamp(trade["decision_time"])
    exit_time = timestamp(trade["exit_time"])
    bid = require_float(trade, "entry_bid")
    ask = require_float(trade, "entry_ask")
    exit_bid = require_float(trade, "exit_bid")
    exit_ask = require_float(trade, "exit_ask")
    record = OrderRecord(
        order_id=order_id,
        contract_id=str(trade["contract_id"]),
        side="BUY",
        intended_size=1,
        decision_time=decision_time,
        submit_time=decision_time,
        ack_time=decision_time,
        fill_times=[decision_time],
        limit_price=ask,
        fill_prices=[ask],
        fill_sizes=[1],
        nbbo_at_decision=(bid, ask),
        nbbo_at_submit=(bid, ask),
        nbbo_at_fill=(bid, ask),
        quote_age_ms_at_decision=quote_age_ms(trade.get("quote_gap_seconds")),
        quote_age_ms_at_fill=0,
        spread_at_submit=ask - bid,
        option_price_at_submit=ask,
    )
    for state in (
        OrderState.DECISION_MADE,
        OrderState.ORDER_SUBMITTED,
        OrderState.BROKER_ACKNOWLEDGED,
        OrderState.WORKING,
        OrderState.FILLED,
    ):
        record.transition(t=decision_time, to_state=state)
    record.nbbo_at_fill = (exit_bid, exit_ask)
    record.transition(t=exit_time, to_state=OrderState.EXIT_SUBMITTED)
    record.transition(t=exit_time, to_state=OrderState.EXIT_FILLED)
    return record


def order_row(record: OrderRecord, trade: dict[str, Any]) -> dict[str, Any]:
    entry_ask = require_float(trade, "entry_ask")
    exit_bid = require_float(trade, "exit_bid")
    return {
        "order_id": record.order_id,
        "candidate_uid": trade.get("candidate_uid"),
        "contract_id": record.contract_id,
        "side": record.side,
        "intended_size": record.intended_size,
        "decision_time": trade.get("decision_time"),
        "exit_time": trade.get("exit_time"),
        "entry_ask": entry_ask,
        "exit_bid": exit_bid,
        "premium": premium_dollars(trade),
        "realized_pnl": (exit_bid - entry_ask) * 100.0,
        "reported_pnl": none_or_float(trade.get("pnl")),
        "final_state": record.final_state.value if record.final_state else None,
        "history": [
            {
                **asdict(event),
                "timestamp": event.timestamp.isoformat(),
                "from_state": event.from_state.value,
                "to_state": event.to_state.value,
            }
            for event in record.history
        ],
    }


def stress_replay(trades: list[dict[str, Any]], *, starting_cash: float, stress_per_side: float) -> dict[str, Any]:
    cash = float(starting_cash)
    peak = cash
    max_drawdown = 0.0
    unaffordable = 0
    daily: dict[str, float] = {}
    for trade in sorted(trades, key=lambda row: (row["decision_ms"], row["exit_ms"], row["candidate_uid"])):
        premium = premium_dollars(trade)
        if premium is None:
            unaffordable += 1
            continue
        stressed_entry_cost = premium + stress_per_side * 100.0
        if stressed_entry_cost > cash + 1e-9:
            unaffordable += 1
            continue
        pnl = float(trade["pnl"]) - 2.0 * stress_per_side * 100.0
        cash += pnl
        peak = max(peak, cash)
        max_drawdown = min(max_drawdown, cash - peak)
        session = str(trade["session"])
        daily[session] = daily.get(session, 0.0) + pnl
    worst_day = min(daily.values()) if daily else 0.0
    return {
        "stress_per_side": float(stress_per_side),
        "trades": len(trades) - unaffordable,
        "unaffordable_trades": unaffordable,
        "ending_cash": round(cash, 2),
        "total_pnl": round(cash - float(starting_cash), 2),
        "max_drawdown": round(max_drawdown, 2),
        "worst_day_pnl": round(worst_day, 2),
        "positive_ending_cash": bool(cash > 0.0),
    }


def decide(order_summary: dict[str, Any], stress_rows: list[dict[str, Any]], *, skipped: list[dict[str, Any]], starting_cash: float) -> str:
    if abs(float(starting_cash) - TRADING_CAPITAL_BASELINE) > 1e-9:
        return "blocked_wrong_paper_capital_baseline"
    if skipped:
        return "blocked_10000_paper_replay_skipped_trades"
    if order_summary.get("errors"):
        return "fail_order_state_errors"
    if not order_summary.get("all_exit_filled") or not order_summary.get("all_one_contract") or not order_summary.get("all_spxw"):
        return "fail_order_state_invariants"
    if int(order_summary.get("max_concurrent_positions", 99)) > 1:
        return "fail_overlapping_positions"
    stress_025 = next((row for row in stress_rows if abs(float(row["stress_per_side"]) - 0.25) < 1e-9), None)
    if stress_025 and float(stress_025["total_pnl"]) <= 0:
        return "blocked_025_stress_not_positive"
    return "pass_10000_order_state_rehearsal_live_data_pending"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Run no-order Protocol101 live shadow capture once live SPX/VIX and OPRA/SPXW subscriptions are active. "
            "Do not place paper orders until live-data parity passes."
        )
    return "Fix the failed order-state or capital invariant before any live-shadow or paper-order step."


def max_concurrent_positions(trades: list[dict[str, Any]]) -> int:
    events: list[tuple[int, int]] = []
    for trade in trades:
        events.append((int(trade["decision_ms"]), 1))
        events.append((int(trade["exit_ms"]), -1))
    current = 0
    max_seen = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        max_seen = max(max_seen, current)
    return max_seen


def require_float(row: dict[str, Any], key: str) -> float:
    value = none_or_float(row.get(key))
    if value is None or not math.isfinite(value):
        raise ValueError(f"{key} is missing or non-finite")
    return value


def quote_age_ms(value: Any) -> int | None:
    number = none_or_float(value)
    return None if number is None else int(round(number * 1000.0))


def timestamp(value: Any):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _skip_reasons(skipped: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in skipped:
        reason = str(row.get("paper_skip_reason", "unknown") or "unknown")
        out[reason] = out.get(reason, 0) + 1
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    capital = payload["capital_assumption"]
    lines = [
        "# Protocol 123: Protocol101 Order-State Rehearsal",
        "",
        "No paid market data was downloaded. No broker endpoint was called. No orders were placed.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- IBKR access reserve: `${capital['ibkr_access_reserve']:,.0f}`",
        f"- Paper trading starting cash: `${capital['paper_trading_starting_equity']:,.0f}`",
        f"- Paper trades rehearsed: `{payload['paper_trade_count']}`",
        f"- Skipped trades: `{payload['skipped_trade_count']}`",
        f"- Max concurrent positions: `{payload['order_state']['max_concurrent_positions']}`",
        f"- All exit filled: `{payload['order_state']['all_exit_filled']}`",
        f"- All one contract: `{payload['order_state']['all_one_contract']}`",
        "",
        "## Slippage Stress",
        "",
        "| extra_per_side | trades | unaffordable | ending_cash | total_pnl | max_drawdown | worst_day |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["stress_results"]:
        lines.append(
            "| "
            f"${row['stress_per_side']:.2f} | "
            f"{row['trades']} | "
            f"{row['unaffordable_trades']} | "
            f"${row['ending_cash']:,.0f} | "
            f"${row['total_pnl']:,.0f} | "
            f"${row['max_drawdown']:,.0f} | "
            f"${row['worst_day_pnl']:,.0f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["next_gate"],
        "",
        "This validates order-state accounting around the $10,000 paper baseline. It does not validate live market data, broker fills, or order placement.",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 123 Protocol101 Order-State Rehearsal"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Rehearsed frozen Protocol101 paper trades through the v4 order-state machine using $10,000 starting paper cash and treating $500 only as the IBKR access reserve.
Reason: Cash settlement blocks paper orders, so the next no-cost promotion-readiness work is order-state accounting around the intended $10,000 bankroll.
Data Used: Existing Protocol101/107 trade artifacts and local normalized quote files only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Rehearsed {payload['paper_trade_count']} trades; skipped {payload['skipped_trade_count']}; all_exit_filled={payload['order_state']['all_exit_filled']}; max_concurrent_positions={payload['order_state']['max_concurrent_positions']}. Report: {report_path}
Next Gate: {payload['next_gate']}
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
