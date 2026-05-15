"""Protocol 155: one-contract live paper timing evidence.

This protocol is the bridge from historical replay to real paper sessions. It
does not train models or buy data. It analyzes one Protocol101 paper/live log,
enforces that operational paper orders are one contract only, extracts timing
and fill evidence, and replays closed paper trades through the one-contract
baseline and the account-aware multi-contract candidate offline.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.paper_trade_log import DEFAULT_TRADE_LOG_ROOT, load_trade_log, validate_trade_log
from v4.scripts.run_protocol148_protocol101_post_session_analyzer import resolve_trade_log
from v4.sim.protocol101_position_sizing import (
    account_aware_sizer_policy,
    baseline_one_contract_policy,
    simulate_position_sizing,
)


DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_155_protocol101_live_timing_evidence")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
CONTRACT_MULTIPLIER = 100.0
DELAY_STRESS_SECONDS = (1, 5, 15, 30)
MAX_OPERATIONAL_QUANTITY = 1
MAX_ENTRY_FILL_LATENCY_MS = 5_000.0
MAX_QUOTE_AGE_MS = 1_500.0
MAX_CONTEXT_AGE_MS = 5_000.0
MAX_ENTRY_SLIPPAGE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
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
    analysis = analyze_timing_evidence(rows, trade_log=trade_log)
    session = str(analysis["session"])
    run_id = str(analysis["run_id"])
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = decide(analysis)
    payload = {
        "protocol": "155_protocol101_one_contract_live_timing_evidence",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_orders_submitted": bool(analysis["paper_orders_submitted"]),
        "real_money_trading": False,
        "broker_endpoint_called": bool(analysis["broker_order_endpoint_called_rows"]),
        "operational_mode": "one_contract_paper_only",
        "multi_contract_enabled": False,
        "source_trade_log": str(trade_log),
        "analysis": analysis,
        "outputs": {
            "summary": str(out_dir / "summary.json"),
            "report": str(out_dir / "report.md"),
            "decision_rows": str(out_dir / "decision_rows.csv"),
            "closed_trade_rows": str(out_dir / "closed_trade_rows.csv"),
            "delay_stress_rows": str(out_dir / "delay_stress_rows.csv"),
            "replay_summary": str(out_dir / "replay_summary.csv"),
        },
        "next_gate": next_gate(decision, analysis),
    }
    pd.DataFrame(analysis["decision_rows"]).to_csv(out_dir / "decision_rows.csv", index=False)
    pd.DataFrame(analysis["closed_trade_rows"]).to_csv(out_dir / "closed_trade_rows.csv", index=False)
    pd.DataFrame(analysis["delay_stress_rows"]).to_csv(out_dir / "delay_stress_rows.csv", index=False)
    pd.DataFrame(analysis["replay_summary_rows"]).to_csv(out_dir / "replay_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload)
    print(json.dumps({"decision": decision, "report": str(out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def analyze_timing_evidence(rows: list[dict[str, Any]], *, trade_log: Path) -> dict[str, Any]:
    validation = validate_trade_log(rows)
    session = first_value(rows, "session") or trade_log.parent.name
    run_id = first_value(rows, "run_id") or trade_log.stem
    event_counts = Counter(str(row.get("event_type", "missing")) for row in rows)
    grouped = group_by_trade_uid(rows)
    decision_rows = [decision_row(trade_uid, trade_rows) for trade_uid, trade_rows in sorted(grouped.items())]
    decision_rows = [row for row in decision_rows if row is not None]
    closed_trade_rows = [closed_trade_row(trade_uid, trade_rows) for trade_uid, trade_rows in sorted(grouped.items())]
    closed_trade_rows = [row for row in closed_trade_rows if row is not None]
    delay_stress_rows = build_delay_stress_rows(decision_rows, closed_trade_rows)
    replay_summary_rows = replay_closed_trades(closed_trade_rows)
    operational = operational_checks(rows, decision_rows, closed_trade_rows)
    timing = timing_checks(decision_rows, closed_trade_rows, delay_stress_rows)
    return {
        "session": session,
        "run_id": run_id,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "trade_log": str(trade_log),
        "validation": validation,
        "event_counts": dict(sorted(event_counts.items())),
        "paper_orders_submitted": int(event_counts.get("paper_order_submitted", 0)),
        "broker_order_endpoint_called_rows": sum(1 for row in rows if bool(row.get("broker_order_endpoint_called"))),
        "decision_count": len(decision_rows),
        "closed_trade_count": len(closed_trade_rows),
        "decision_rows": decision_rows,
        "closed_trade_rows": closed_trade_rows,
        "delay_stress_rows": delay_stress_rows,
        "replay_summary_rows": replay_summary_rows,
        "operational_checks": operational,
        "timing_checks": timing,
    }


def group_by_trade_uid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anonymous = 0
    for row in rows:
        trade_uid = str(row.get("trade_uid") or "")
        if not trade_uid:
            if str(row.get("event_type")) not in {"heartbeat", "market_snapshot", "paper_account_state"}:
                anonymous += 1
                trade_uid = f"anonymous_{anonymous:04d}"
            else:
                continue
        grouped[trade_uid].append(row)
    return grouped


def decision_row(trade_uid: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    decision = first_event(rows, "model_decision") or first_event(rows, "risk_gate") or first_trade_event(rows)
    if decision is None:
        return None
    contract = obj(decision.get("selected_contract"))
    market = obj(decision.get("market_snapshot"))
    option = obj(market.get("option_nbbo"))
    context = obj(market.get("context"))
    underlying = obj(market.get("underlying"))
    model = obj(decision.get("model_decision"))
    risk = obj(decision.get("risk_gate"))
    timing = obj(decision.get("timing"))
    entry_fill = first_event(rows, "paper_entry_fill")
    exit_fill = first_event(rows, "paper_exit_fill")
    order_event = first_event(rows, "paper_order_submitted") or first_event(rows, "paper_order_dry_run")
    return {
        "trade_uid": trade_uid,
        "session": decision.get("session"),
        "decision_time": decision.get("timestamp"),
        "selected_action": model.get("action"),
        "contract_id": contract_id(contract),
        "side": side_from_contract(contract),
        "quantity": order_quantity(decision),
        "score": number(model.get("score")),
        "threshold": number(model.get("threshold")),
        "score_margin": score_margin(model),
        "bid": number(option.get("bid")),
        "ask": number(option.get("ask")),
        "bid_size": number(option.get("bid_size")),
        "ask_size": number(option.get("ask_size")),
        "quote_age_ms": number(option.get("quote_age_ms")),
        "context_age_ms": number(context.get("context_age_ms")),
        "spx_timestamp": underlying.get("spx_timestamp"),
        "vix_timestamp": underlying.get("vix_timestamp"),
        "risk_passed": bool(risk.get("passed")),
        "blocked_reason": risk.get("reason") or decision.get("blocked_reason"),
        "intended_entry_time": timing.get("intended_entry_time") or decision.get("timestamp"),
        "intended_exit_time": timing.get("intended_exit_time"),
        "broker_submit_at": event_timestamp(order_event),
        "entry_fill_at": timing.get("entry_fill_at") or event_timestamp(entry_fill),
        "exit_fill_at": timing.get("exit_fill_at") or event_timestamp(exit_fill),
        "decision_to_submit_ms": milliseconds_between(decision.get("timestamp"), event_timestamp(order_event)),
        "decision_to_entry_fill_ms": milliseconds_between(decision.get("timestamp"), timing.get("entry_fill_at") or event_timestamp(entry_fill)),
        "exit_decision_to_exit_fill_ms": milliseconds_between(timing.get("exit_decision_at"), timing.get("exit_fill_at") or event_timestamp(exit_fill)),
        "entry_slippage": entry_slippage(decision, entry_fill),
        "closed": entry_fill is not None and exit_fill is not None,
        "delay_quote_scenarios": obj(decision.get("delay_quote_scenarios") or timing.get("delay_quote_scenarios")),
    }


def closed_trade_row(trade_uid: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    decision = first_event(rows, "model_decision") or first_event(rows, "risk_gate") or first_trade_event(rows)
    entry = first_event(rows, "paper_entry_fill")
    exit_ = first_event(rows, "paper_exit_fill")
    if decision is None or entry is None or exit_ is None:
        return None
    contract = obj(entry.get("selected_contract")) or obj(decision.get("selected_contract"))
    entry_order = obj(entry.get("order"))
    exit_order = obj(exit_.get("order"))
    model = obj(decision.get("model_decision"))
    entry_price = number(entry_order.get("avg_fill_price") or entry_order.get("limit_price"))
    exit_price = number(exit_order.get("avg_fill_price") or exit_order.get("limit_price"))
    quantity = int(number(entry_order.get("filled") or entry_order.get("quantity")) or 1)
    if entry_price is None or exit_price is None:
        return None
    one_contract_pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER
    premium = entry_price * CONTRACT_MULTIPLIER
    account = obj(decision.get("account"))
    return {
        "trade_uid": trade_uid,
        "trade_number": len([1 for row in rows if row.get("event_type") == "paper_entry_fill"]),
        "session": decision.get("session"),
        "decision_time": decision.get("timestamp"),
        "exit_time": exit_.get("timestamp"),
        "contract_id": contract_id(contract),
        "side": side_from_contract(contract),
        "score": number(model.get("score")),
        "threshold": number(model.get("threshold")),
        "score_margin": score_margin(model),
        "premium_paid": premium,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "pnl": round(one_contract_pnl, 6),
        "paper_realized_pnl": round(one_contract_pnl * quantity, 6),
        "cash_before": number(account.get("cash") or account.get("equity")) or 10_000.0,
    }


def operational_checks(
    rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    closed_trade_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    quantities = [
        int(number(obj(row.get("order")).get("quantity") or obj(row.get("order")).get("totalQuantity") or obj(row.get("order")).get("filled")) or 0)
        for row in rows
        if str(row.get("event_type", "")).startswith("paper_order")
        or str(row.get("event_type", "")).startswith("paper_entry")
        or str(row.get("event_type", "")).startswith("paper_exit")
    ]
    bad_qty = [qty for qty in quantities if qty > MAX_OPERATIONAL_QUANTITY]
    real_money_rows = [row for row in rows if bool(row.get("real_money_trading"))]
    return {
        "one_contract_operational_only": not bad_qty,
        "max_observed_order_quantity": max(quantities or [0]),
        "quantity_violations": len(bad_qty),
        "multi_contract_orders_enabled": bool(bad_qty),
        "real_money_rows": len(real_money_rows),
        "closed_trades_all_one_contract": all(int(row.get("quantity", 0)) <= 1 for row in closed_trade_rows),
        "decisions_logged": len(decision_rows),
        "closed_trades_logged": len(closed_trade_rows),
    }


def timing_checks(
    decision_rows: list[dict[str, Any]],
    closed_trade_rows: list[dict[str, Any]],
    delay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    entry_latencies = [float(row["decision_to_entry_fill_ms"]) for row in decision_rows if row.get("decision_to_entry_fill_ms") is not None]
    quote_ages = [float(row["quote_age_ms"]) for row in decision_rows if row.get("quote_age_ms") is not None]
    context_ages = [float(row["context_age_ms"]) for row in decision_rows if row.get("context_age_ms") is not None]
    slippages = [float(row["entry_slippage"]) for row in decision_rows if row.get("entry_slippage") is not None]
    delay_coverage = delay_stress_coverage(decision_rows, delay_rows)
    return {
        "entry_latency_rows": len(entry_latencies),
        "max_decision_to_entry_fill_ms": max(entry_latencies) if entry_latencies else None,
        "entry_latency_inside_budget": bool(entry_latencies) and max(entry_latencies) <= MAX_ENTRY_FILL_LATENCY_MS,
        "quote_age_rows": len(quote_ages),
        "max_quote_age_ms": max(quote_ages) if quote_ages else None,
        "quote_age_inside_budget": bool(quote_ages) and max(quote_ages) <= MAX_QUOTE_AGE_MS,
        "context_age_rows": len(context_ages),
        "max_context_age_ms": max(context_ages) if context_ages else None,
        "context_age_inside_budget": bool(context_ages) and max(context_ages) <= MAX_CONTEXT_AGE_MS,
        "entry_slippage_rows": len(slippages),
        "max_entry_slippage": max(slippages) if slippages else None,
        "entry_slippage_inside_budget": bool(slippages) and max(slippages) <= MAX_ENTRY_SLIPPAGE,
        "closed_trades_for_replay": len(closed_trade_rows),
        "delay_stress_rows": len(delay_rows),
        "delay_stress_coverage": delay_coverage,
        "delay_stress_available": bool(delay_rows) and all(
            float(row.get("coverage", 0.0)) >= 1.0 for row in delay_coverage
        ),
    }


def build_delay_stress_rows(
    decision_rows: list[dict[str, Any]],
    closed_trade_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    closed_by_uid = {row["trade_uid"]: row for row in closed_trade_rows}
    out: list[dict[str, Any]] = []
    for row in decision_rows:
        closed = closed_by_uid.get(row["trade_uid"])
        scenarios = obj(row.get("delay_quote_scenarios"))
        if not closed or not scenarios:
            continue
        original_entry = number(closed.get("entry_price"))
        exit_price = number(closed.get("exit_price"))
        if original_entry is None or exit_price is None:
            continue
        for delay in DELAY_STRESS_SECONDS:
            scenario = obj(scenarios.get(str(delay)) or scenarios.get(delay))
            delayed_entry = number(scenario.get("entry_ask") or scenario.get("ask"))
            delayed_exit = number(scenario.get("exit_bid") or scenario.get("bid") or exit_price)
            if delayed_entry is None or delayed_exit is None:
                continue
            delayed_pnl = (delayed_exit - delayed_entry) * CONTRACT_MULTIPLIER
            out.append(
                {
                    "trade_uid": row["trade_uid"],
                    "session": row.get("session"),
                    "delay_seconds": delay,
                    "original_pnl": closed["pnl"],
                    "delayed_pnl": round(delayed_pnl, 6),
                    "pnl_delta": round(delayed_pnl - float(closed["pnl"]), 6),
                }
            )
    return out


def delay_stress_coverage(decision_rows: list[dict[str, Any]], delay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed_decisions = [row for row in decision_rows if bool(row.get("closed"))]
    denominator = max(len(closed_decisions), 1)
    out: list[dict[str, Any]] = []
    for delay in DELAY_STRESS_SECONDS:
        rows = [row for row in delay_rows if int(row.get("delay_seconds", -1)) == delay]
        out.append(
            {
                "delay_seconds": delay,
                "covered_rows": len(rows),
                "closed_trade_rows": len(closed_decisions),
                "coverage": round(len(rows) / denominator, 6),
                "total_delayed_pnl": round(sum(float(row.get("delayed_pnl", 0.0)) for row in rows), 6),
                "total_original_pnl": round(sum(float(row.get("original_pnl", 0.0)) for row in rows), 6),
            }
        )
    return out


def replay_closed_trades(closed_trade_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not closed_trade_rows:
        return []
    trades = []
    for index, row in enumerate(closed_trade_rows, start=1):
        trades.append(
            {
                "trade_number": index,
                "session": row["session"],
                "decision_time": row["decision_time"],
                "exit_time": row["exit_time"],
                "contract_id": row["contract_id"],
                "side": row.get("side"),
                "score": row.get("score"),
                "threshold": row.get("threshold"),
                "premium_paid": row.get("premium_paid"),
                "pnl": row.get("pnl"),
            }
        )
    baseline = simulate_position_sizing(trades, baseline_one_contract_policy())
    multi = simulate_position_sizing(trades, account_aware_sizer_policy(10_000.0))
    return [
        {
            "policy": "one_contract_baseline",
            "total_pnl": baseline["summary"]["total_pnl"],
            "ending_cash": baseline["summary"]["ending_cash"],
            "max_drawdown": baseline["summary"]["max_drawdown"],
            "max_quantity": baseline["summary"]["max_quantity"],
            "taken_trades": baseline["summary"]["taken_trades"],
        },
        {
            "policy": "account_aware_sizer_v1_offline_only",
            "total_pnl": multi["summary"]["total_pnl"],
            "ending_cash": multi["summary"]["ending_cash"],
            "max_drawdown": multi["summary"]["max_drawdown"],
            "max_quantity": multi["summary"]["max_quantity"],
            "taken_trades": multi["summary"]["taken_trades"],
        },
    ]


def decide(analysis: dict[str, Any]) -> str:
    validation = analysis["validation"]
    operational = analysis["operational_checks"]
    timing = analysis["timing_checks"]
    if validation["status"] != "pass":
        return "blocked_protocol155_trade_log_validation_failed"
    if operational["real_money_rows"] > 0:
        return "reject_protocol155_real_money_row_detected"
    if not operational["one_contract_operational_only"] or not operational["closed_trades_all_one_contract"]:
        return "reject_protocol155_multi_contract_operational_violation"
    if operational["closed_trades_logged"] <= 0:
        return "blocked_protocol155_no_closed_one_contract_paper_trades_yet"
    blockers = []
    for key in (
        "entry_latency_inside_budget",
        "quote_age_inside_budget",
        "context_age_inside_budget",
        "entry_slippage_inside_budget",
        "delay_stress_available",
    ):
        if not timing[key]:
            blockers.append(key)
    if blockers:
        return "blocked_protocol155_timing_evidence_incomplete"
    return "pass_protocol155_live_timing_evidence_ready_for_multi_contract_review"


def next_gate(decision: str, analysis: dict[str, Any]) -> str:
    if decision.startswith("pass_"):
        return (
            "Continue one-contract paper trading and accumulate sessions. Replay the same logs through "
            "account_aware_sizer_v1 offline; multi-contract remains disabled until repeated sessions pass."
        )
    if decision == "blocked_protocol155_no_closed_one_contract_paper_trades_yet":
        return "Run the next market session in one-contract paper mode and capture closed entry/exit fill rows."
    if decision == "blocked_protocol155_timing_evidence_incomplete":
        return "Keep one-contract paper mode; add missing latency, quote freshness, slippage, or delay-stress observations to the live log."
    if decision.startswith("reject_"):
        return "Do not continue paper execution until the one-contract or real-money safety violation is fixed."
    return "Fix the trade log schema before using the run for promotion evidence."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    analysis = payload["analysis"]
    operational = analysis["operational_checks"]
    timing = analysis["timing_checks"]
    lines = [
        "# Protocol 155: One-Contract Live Paper Timing Evidence",
        "",
        "No paid data was downloaded. No real-money trading is allowed. Multi-contract execution is disabled.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source log: `{payload['source_trade_log']}`",
        f"- Paper orders submitted: `{payload['paper_orders_submitted']}`",
        f"- Operational mode: `{payload['operational_mode']}`",
        f"- Multi-contract enabled: `{payload['multi_contract_enabled']}`",
        f"- Decisions logged: `{operational['decisions_logged']}`",
        f"- Closed trades logged: `{operational['closed_trades_logged']}`",
        f"- Max observed order quantity: `{operational['max_observed_order_quantity']}`",
        f"- One-contract operational only: `{operational['one_contract_operational_only']}`",
        "",
        "## Timing Evidence",
        "",
        f"- Max decision-to-entry-fill latency ms: `{timing['max_decision_to_entry_fill_ms']}`",
        f"- Entry latency inside budget: `{timing['entry_latency_inside_budget']}`",
        f"- Max quote age ms: `{timing['max_quote_age_ms']}`",
        f"- Quote age inside budget: `{timing['quote_age_inside_budget']}`",
        f"- Max context age ms: `{timing['max_context_age_ms']}`",
        f"- Context age inside budget: `{timing['context_age_inside_budget']}`",
        f"- Max entry slippage: `{timing['max_entry_slippage']}`",
        f"- Entry slippage inside budget: `{timing['entry_slippage_inside_budget']}`",
        f"- Delay stress available: `{timing['delay_stress_available']}`",
        "",
        "## Offline Replay",
        "",
    ]
    if analysis["replay_summary_rows"]:
        lines.extend(
            [
                "| policy | total_pnl | ending_cash | max_drawdown | max_quantity | taken_trades |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in analysis["replay_summary_rows"]:
            lines.append(
                f"| {row['policy']} | {money(row['total_pnl'])} | {money(row['ending_cash'])} | "
                f"{money(row['max_drawdown'])} | {row['max_quantity']} | {row['taken_trades']} |"
            )
    else:
        lines.append("No closed paper trades are available for offline replay yet.")
    lines.extend(["", "## Next Gate", "", payload["next_gate"], ""])
    path.write_text("\n".join(lines))


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = f"""
## 2026-05-15 Protocol 155 One-Contract Live Paper Timing Evidence

```text
Date: 2026-05-15
Decision / Experiment: Added the one-contract live paper timing evidence protocol for Protocol101.
Reason: Multi-contract sizing is economically promising but blocked by timing evidence; live paper should collect timing and fill data while executing only one contract.
Data Used: Existing paper/live trade log rows only. No paid data was downloaded.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {payload['outputs']['report']}
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    with path.open("a") as handle:
        handle.write(entry)


def first_event(rows: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("event_type") == event_type:
            return row
    return None


def first_trade_event(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("event_type", "")).startswith("paper_") or row.get("event_type") in {"model_decision", "risk_gate"}:
            return row
    return None


def first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def event_timestamp(row: dict[str, Any] | None) -> str | None:
    return None if row is None else row.get("timestamp")


def milliseconds_between(start: Any, end: Any) -> float | None:
    left = parse_time(start)
    right = parse_time(end)
    if left is None or right is None:
        return None
    return round((right - left).total_seconds() * 1000.0, 6)


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def entry_slippage(decision: dict[str, Any], entry_fill: dict[str, Any] | None) -> float | None:
    if entry_fill is None:
        return None
    decision_ask = number(obj(obj(decision.get("market_snapshot")).get("option_nbbo")).get("ask"))
    fill = number(obj(entry_fill.get("order")).get("avg_fill_price") or obj(entry_fill.get("order")).get("limit_price"))
    if decision_ask is None or fill is None:
        return None
    return round(fill - decision_ask, 6)


def order_quantity(row: dict[str, Any]) -> int:
    order = obj(row.get("order"))
    return int(number(order.get("quantity") or order.get("totalQuantity") or order.get("filled")) or 0)


def contract_id(contract: dict[str, Any]) -> str:
    return str(contract.get("contract_id") or contract.get("local_symbol") or contract.get("localSymbol") or "")


def side_from_contract(contract: dict[str, Any]) -> str:
    right = str(contract.get("right") or "").upper()
    if right == "C":
        return "CALL"
    if right == "P":
        return "PUT"
    return str(contract.get("side") or "")


def score_margin(model: dict[str, Any]) -> float | None:
    score = number(model.get("score"))
    threshold = number(model.get("threshold"))
    if score is None or threshold is None:
        return None
    return round(score - threshold, 6)


def obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def money(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):,.0f}"


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
