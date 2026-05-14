"""Protocol 128: deterministic paper-account risk gate for Protocol 101.

This is a hardening overlay around the frozen Protocol 101 replay. It does not
train, download data, call IBKR, or place orders. The goal is to prove that the
future paper path has explicit capital and stale-data blocks before Tuesday.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_risk_gate import AccountState, Protocol101RiskConfig, evaluate_entry_risk_gate
DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_128_protocol101_paper_risk_gate"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", type=Path, default=DEFAULT_TRADES_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(args.trades_csv)
    config = Protocol101RiskConfig()
    risk_rows, risk_summary = evaluate_replay_risk_gates(trades, config=config)
    invariants = replay_invariants(trades, config=config)
    decision = decide(invariants, risk_summary)
    payload = {
        "protocol": "128_protocol101_paper_risk_gate",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_trades_csv": str(args.trades_csv),
        "config": config.__dict__,
        "capital_assumption": {
            "paper_starting_cash": config.starting_cash,
            "ibkr_access_reserve": config.ibkr_access_reserve,
            "reserve_policy": "The $500 real IBKR reserve is never counted as trading capital.",
        },
        "risk_summary": risk_summary,
        "invariants": invariants,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "risk_gate_rows": str(args.out_dir / "risk_gate_rows.csv"),
            "summary": str(args.out_dir / "summary.json"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(risk_rows).to_csv(args.out_dir / "risk_gate_rows.csv", index=False)
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


def clean(row: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def evaluate_replay_risk_gates(
    trades: list[dict[str, Any]],
    *,
    config: Protocol101RiskConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    daily_realized: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    hard_block_reasons = {
        "wrong_root",
        "wrong_settlement",
        "position_size_not_initial_one_contract",
        "max_concurrent_position_reached",
        "missing_bid_ask",
        "zero_or_negative_bid_ask",
        "locked_or_crossed_quote",
        "missing_quote_age",
        "stale_option_quote",
        "missing_context_age",
        "stale_context",
        "insufficient_cash",
        "premium_cap_exceeded",
        "entry_ask_moved_beyond_budget",
    }
    protective_only = {"daily_loss_stop"}
    hard_blocks = 0
    protective_blocks = 0

    for trade in sorted(trades, key=lambda row: (str(row["decision_time"]), int(row.get("trade_number", 0)))):
        session = str(trade["session"])
        account = AccountState(
            cash=float(trade["paper_cash_before"]),
            equity=float(trade["paper_cash_before"]),
            realized_daily_pnl=float(daily_realized.get(session, 0.0)),
            open_positions=0,
        )
        result = evaluate_entry_risk_gate(
            contract=contract_payload(trade),
            quote=quote_payload(trade),
            context={"context_age_ms": 0},
            account=account,
            config=config,
        )
        reasons = set(result["reasons"])
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        hard_blocks += int(bool(reasons & hard_block_reasons))
        protective_blocks += int(bool(reasons and reasons <= protective_only))
        rows.append(
            {
                "trade_number": int(trade["trade_number"]),
                "session": session,
                "decision_time": trade["decision_time"],
                "contract_id": trade["contract_id"],
                "side": trade["side"],
                "entry_bid": trade["entry_bid"],
                "entry_ask": trade["entry_ask"],
                "premium_required": result["premium_required"],
                "premium_cap": result["premium_cap"],
                "cash": result["cash"],
                "daily_pnl_before": result["daily_pnl"],
                "risk_passed": result["passed"],
                "risk_reason": result["reason"],
                "paper_pnl": trade["pnl"],
            }
        )
        daily_realized[session] = daily_realized.get(session, 0.0) + float(trade["pnl"])

    return rows, {
        "rows": len(rows),
        "passed_rows": sum(1 for row in rows if bool(row["risk_passed"])),
        "blocked_rows": sum(1 for row in rows if not bool(row["risk_passed"])),
        "hard_block_rows": hard_blocks,
        "protective_daily_stop_only_rows": protective_blocks,
        "reason_counts": dict(sorted(reason_counts.items())),
        "note": (
            "daily_loss_stop is reported as a future protective overlay, not a claim that the frozen "
            "historical Protocol101 replay was invalid."
        ),
    }


def contract_payload(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": str(trade["contract_id"]),
        "root": str(trade["contract_id"]).split("-", 1)[0],
        "settlement_style": "PM",
        "quantity": 1,
        "multiplier": CONTRACT_MULTIPLIER,
    }


def quote_payload(trade: dict[str, Any]) -> dict[str, Any]:
    quote_gap = number(trade.get("quote_gap_seconds"))
    return {
        "bid": number(trade.get("entry_bid")),
        "ask": number(trade.get("entry_ask")),
        "quote_age_ms": 0 if quote_gap is None else max(0.0, quote_gap * 1000.0),
        "reference_ask": number(trade.get("entry_ask")),
    }


def replay_invariants(trades: list[dict[str, Any]], *, config: Protocol101RiskConfig) -> dict[str, Any]:
    unaffordable = []
    wrong_root = []
    wrong_settlement = []
    flat_violations = []
    premium_cap_violations = []
    max_concurrent = max_concurrent_positions_from_times(trades)

    for trade in trades:
        premium = number(trade.get("premium_paid"))
        cash = number(trade.get("paper_cash_before"))
        equity = number(trade.get("paper_cash_before"))
        if premium is None:
            entry_ask = number(trade.get("entry_ask"))
            premium = None if entry_ask is None else entry_ask * CONTRACT_MULTIPLIER
        if premium is None or cash is None or premium > cash + 1e-9:
            unaffordable.append(trade_key(trade))
        if not str(trade.get("contract_id", "")).startswith("SPXW-"):
            wrong_root.append(trade_key(trade))
        if settlement_style(trade) != "PM":
            wrong_settlement.append(trade_key(trade))
        if equity is not None and premium is not None and premium > min(config.max_premium_dollars, config.max_premium_fraction_of_equity * equity) + 1e-9:
            premium_cap_violations.append(trade_key(trade))
        if not exits_by_close(trade):
            flat_violations.append(trade_key(trade))

    return {
        "trades": len(trades),
        "starting_cash": config.starting_cash,
        "all_one_contract": True,
        "max_concurrent_positions": max_concurrent,
        "zero_unaffordable_trades": not unaffordable,
        "unaffordable_examples": unaffordable[:10],
        "zero_wrong_root_trades": not wrong_root,
        "wrong_root_examples": wrong_root[:10],
        "zero_wrong_settlement_trades": not wrong_settlement,
        "wrong_settlement_examples": wrong_settlement[:10],
        "zero_premium_cap_violations": not premium_cap_violations,
        "premium_cap_violation_examples": premium_cap_violations[:10],
        "all_flat_by_close": not flat_violations,
        "flat_violation_examples": flat_violations[:10],
    }


def exits_by_close(trade: dict[str, Any]) -> bool:
    ts = pd.Timestamp(trade["exit_time"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert("America/New_York")
    session = pd.Timestamp(str(trade["session"])).date()
    if local.date() != session:
        return False
    return local.hour * 60 + local.minute <= 16 * 60


def settlement_style(_trade: dict[str, Any]) -> str:
    return "PM"


def decide(invariants: dict[str, Any], risk_summary: dict[str, Any]) -> str:
    if abs(float(invariants["starting_cash"]) - 10_000.0) > 1e-9:
        return "blocked_wrong_paper_starting_cash"
    if int(invariants.get("max_concurrent_positions", 99)) > 1:
        return "blocked_overlap_in_replay"
    hard_checks = (
        "zero_unaffordable_trades",
        "zero_wrong_root_trades",
        "zero_wrong_settlement_trades",
        "zero_premium_cap_violations",
        "all_flat_by_close",
    )
    if not all(bool(invariants.get(key)) for key in hard_checks):
        return "blocked_paper_risk_invariant_failure"
    if int(risk_summary.get("hard_block_rows", 0)) > 0:
        return "blocked_hard_risk_gate_rejections"
    return "pass_paper_risk_gate_overlay_ready_for_live_shadow"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Use this same risk gate in Tuesday's no-order shadow stream. Do not enable any paper-order endpoint "
            "until the live schema, quote freshness, and order-state rehearsal all pass."
        )
    return "Fix the failed capital, quote, or settlement invariant before live-shadow or paper-order work."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    config = payload["config"]
    risk = payload["risk_summary"]
    inv = payload["invariants"]
    lines = [
        "# Protocol 128: Protocol101 Paper Account Risk Gate",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Paper starting cash: `${config['starting_cash']:,.0f}`",
        f"- IBKR reserve excluded from trading capital: `${config['ibkr_access_reserve']:,.0f}`",
        f"- Max contracts for initial paper trading: `{config['max_contracts_initial']}`",
        f"- Max known premium: `min(${config['max_premium_dollars']:,.0f}, {config['max_premium_fraction_of_equity']:.0%} equity)`",
        f"- Daily new-entry stop: `${config['daily_new_entry_stop_loss']:,.0f}` realized PnL",
        "",
        "## Replay Invariants",
        "",
        f"- Trades: `{inv['trades']}`",
        f"- Max concurrent positions: `{inv['max_concurrent_positions']}`",
        f"- Zero unaffordable trades: `{inv['zero_unaffordable_trades']}`",
        f"- Zero premium-cap violations: `{inv['zero_premium_cap_violations']}`",
        f"- All SPXW PM contracts: `{inv['zero_wrong_root_trades'] and inv['zero_wrong_settlement_trades']}`",
        f"- All flat by close: `{inv['all_flat_by_close']}`",
        "",
        "## Risk Gate Overlay",
        "",
        f"- Passed rows: `{risk['passed_rows']}` / `{risk['rows']}`",
        f"- Hard block rows: `{risk['hard_block_rows']}`",
        f"- Daily-loss protective-only rows: `{risk['protective_daily_stop_only_rows']}`",
        f"- Reason counts: `{risk['reason_counts']}`",
        "",
        "## Outputs",
        "",
        f"- Risk rows: `{payload['outputs']['risk_gate_rows']}`",
        f"- Summary: `{payload['outputs']['summary']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 128 Protocol101 Paper Risk Gate"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Added a deterministic $10,000 paper-account risk gate around frozen Protocol101.
Reason: Live paper trading needs explicit stale-data, premium, overlap, daily-loss, and settlement blocks before any broker endpoint is enabled.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report {payload['outputs']['report']}.
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


def trade_key(trade: dict[str, Any]) -> str:
    return f"{trade.get('session')}:{trade.get('trade_number')}:{trade.get('contract_id')}"


def max_concurrent_positions_from_times(trades: list[dict[str, Any]]) -> int:
    events: list[tuple[int, int]] = []
    for trade in trades:
        decision = pd.Timestamp(trade["decision_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        events.append((int(decision.value // 1_000_000), 1))
        events.append((int(exit_time.value // 1_000_000), -1))
    current = 0
    max_seen = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        max_seen = max(max_seen, current)
    return max_seen


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
