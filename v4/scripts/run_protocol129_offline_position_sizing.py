"""Protocol 129: offline-only position sizing research for Protocol 101.

This keeps Protocol 101 entries and exits frozen. It only changes simulated
quantity after the one-contract path is already known, so it is not a Tuesday
paper-trading feature and it never touches a broker endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_129_protocol101_offline_position_sizing"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
CONTRACT_MULTIPLIER = 100.0
STARTING_CASH = 10_000.0
DAILY_LOSS_STOP = -750.0
PREMIUM_EXPOSURE_CAP_FRACTION = 0.20


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
    baseline = simulate_one_contract_baseline(trades)
    capped_one_contract = simulate_sizing(trades, mode="one_contract_20pct_cap")
    scaled = simulate_sizing(trades, mode="adaptive_1_to_3_contracts")
    decision = decide_scaling(baseline, scaled)
    payload = {
        "protocol": "129_protocol101_offline_position_sizing",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_trades_csv": str(args.trades_csv),
        "starting_cash": STARTING_CASH,
        "risk_rules": {
            "initial_paper_max_contracts": 1,
            "offline_only": True,
            "equity_below_20k": "max 1 contract",
            "equity_20k_to_50k": "max 2 contracts only if current drawdown is below 5%",
            "equity_above_50k": "max 3 contracts only if drawdown is below 5% and recent pnl is positive",
            "premium_exposure_cap": f"{PREMIUM_EXPOSURE_CAP_FRACTION:.0%} of equity",
            "daily_loss_stop": DAILY_LOSS_STOP,
        },
        "results": [baseline["summary"], capped_one_contract["summary"], scaled["summary"]],
        "baseline_daily": baseline["daily"],
        "scaled_daily": scaled["daily"],
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "sizing_trade_rows": str(args.out_dir / "sizing_trade_rows.csv"),
            "summary": str(args.out_dir / "summary.json"),
        },
        "next_gate": next_gate(decision),
    }
    trade_rows = baseline["rows"] + capped_one_contract["rows"] + scaled["rows"]
    pd.DataFrame(trade_rows).to_csv(args.out_dir / "sizing_trade_rows.csv", index=False)
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


def simulate_one_contract_baseline(trades: list[dict[str, Any]]) -> dict[str, Any]:
    cash = STARTING_CASH
    peak = cash
    rows: list[dict[str, Any]] = []
    daily: dict[str, float] = {}
    for trade in sorted(trades, key=lambda row: (str(row["decision_time"]), int(row.get("trade_number", 0)))):
        pnl = float(trade["pnl"])
        before = cash
        cash += pnl
        peak = max(peak, cash)
        session = str(trade["session"])
        daily[session] = daily.get(session, 0.0) + pnl
        rows.append(
            row_payload(
                mode="one_contract_baseline",
                trade=trade,
                quantity=1,
                cash_before=before,
                cash_after=cash,
                realized_pnl=pnl,
                skip_reason="",
            )
        )
    return {
        "summary": summarize("one_contract_baseline", rows, daily, starting_cash=STARTING_CASH),
        "rows": rows,
        "daily": daily_rows("one_contract_baseline", daily),
    }


def simulate_sizing(trades: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    cash = STARTING_CASH
    peak = cash
    recent_pnls: list[float] = []
    daily: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for trade in sorted(trades, key=lambda row: (str(row["decision_time"]), int(row.get("trade_number", 0)))):
        session = str(trade["session"])
        daily_before = daily.get(session, 0.0)
        premium = premium_dollars(trade)
        qty = 0
        reason = ""
        if daily_before <= DAILY_LOSS_STOP:
            reason = "daily_loss_stop"
        elif premium is None:
            reason = "missing_premium"
        else:
            drawdown_pct = 0.0 if peak <= 0 else max(0.0, (peak - cash) / peak)
            max_qty = 1 if mode == "one_contract_20pct_cap" else max_contracts_for_state(
                equity=cash,
                drawdown_pct=drawdown_pct,
                recent_pnls=recent_pnls,
            )
            exposure_cap = cash * PREMIUM_EXPOSURE_CAP_FRACTION
            affordable_qty = int(min(max_qty, math.floor(cash / premium), math.floor(exposure_cap / premium)))
            if affordable_qty <= 0:
                reason = "premium_exposure_cap"
            else:
                qty = affordable_qty
        pnl = float(trade["pnl"]) * qty
        before = cash
        cash += pnl
        peak = max(peak, cash)
        if qty > 0:
            daily[session] = daily.get(session, 0.0) + pnl
            recent_pnls.append(pnl)
            recent_pnls = recent_pnls[-10:]
        rows.append(
            row_payload(
                mode=mode,
                trade=trade,
                quantity=qty,
                cash_before=before,
                cash_after=cash,
                realized_pnl=pnl,
                skip_reason=reason,
            )
        )
    return {
        "summary": summarize(mode, rows, daily, starting_cash=STARTING_CASH),
        "rows": rows,
        "daily": daily_rows(mode, daily),
    }


def max_contracts_for_state(*, equity: float, drawdown_pct: float, recent_pnls: list[float]) -> int:
    if equity < 20_000.0:
        return 1
    if equity < 50_000.0:
        return 2 if drawdown_pct < 0.05 else 1
    recent_positive = sum(recent_pnls[-10:]) > 0 if recent_pnls else False
    if drawdown_pct < 0.05 and recent_positive:
        return 3
    return 2 if drawdown_pct < 0.05 else 1


def row_payload(
    *,
    mode: str,
    trade: dict[str, Any],
    quantity: int,
    cash_before: float,
    cash_after: float,
    realized_pnl: float,
    skip_reason: str,
) -> dict[str, Any]:
    premium = premium_dollars(trade)
    return {
        "mode": mode,
        "trade_number": int(trade["trade_number"]),
        "session": str(trade["session"]),
        "decision_time": trade["decision_time"],
        "contract_id": trade["contract_id"],
        "side": trade["side"],
        "quantity": int(quantity),
        "one_contract_premium": premium,
        "premium_exposure": None if premium is None else premium * quantity,
        "cash_before": round(cash_before, 6),
        "cash_after": round(cash_after, 6),
        "realized_pnl": round(realized_pnl, 6),
        "skip_reason": skip_reason,
        "one_contract_pnl": float(trade["pnl"]),
    }


def summarize(mode: str, rows: list[dict[str, Any]], daily: dict[str, float], *, starting_cash: float) -> dict[str, Any]:
    taken = [row for row in rows if int(row["quantity"]) > 0]
    cash_values = [starting_cash] + [float(row["cash_after"]) for row in rows]
    peak = starting_cash
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    underwater_start: int | None = None
    longest_recovery = 0
    for index, value in enumerate(cash_values):
        if value >= peak:
            if underwater_start is not None:
                longest_recovery = max(longest_recovery, index - underwater_start)
                underwater_start = None
            peak = value
        else:
            if underwater_start is None:
                underwater_start = index
        drawdown = value - peak
        max_drawdown = min(max_drawdown, drawdown)
        max_drawdown_pct = min(max_drawdown_pct, drawdown / peak if peak else 0.0)
    if underwater_start is not None:
        longest_recovery = max(longest_recovery, len(cash_values) - underwater_start)
    skip_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["skip_reason"] or "")
        if reason:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
    ending = cash_values[-1] if cash_values else starting_cash
    return {
        "mode": mode,
        "starting_cash": round(starting_cash, 2),
        "ending_cash": round(ending, 2),
        "total_pnl": round(ending - starting_cash, 2),
        "return_on_starting_cash": round((ending - starting_cash) / starting_cash, 6),
        "candidate_trades": len(rows),
        "taken_trades": len(taken),
        "total_contracts": int(sum(int(row["quantity"]) for row in rows)),
        "skipped_trades": len(rows) - len(taken),
        "skip_counts": skip_counts,
        "max_quantity": int(max([row["quantity"] for row in rows] or [0])),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 6),
        "worst_day_pnl": round(min(daily.values()) if daily else 0.0, 2),
        "best_day_pnl": round(max(daily.values()) if daily else 0.0, 2),
        "risk_of_ruin": bool(any(value <= 0.0 for value in cash_values)),
        "min_cash": round(min(cash_values), 2),
        "longest_recovery_trades": int(longest_recovery),
    }


def decide_scaling(baseline: dict[str, Any], scaled: dict[str, Any]) -> str:
    base = baseline["summary"]
    test = scaled["summary"]
    if test["risk_of_ruin"]:
        return "reject_scaling_risk_of_ruin"
    if float(test["total_pnl"]) <= float(base["total_pnl"]):
        return "reject_scaling_no_return_improvement"
    if abs(float(test["max_drawdown_pct"])) > abs(float(base["max_drawdown_pct"])) + 0.05:
        return "reject_scaling_drawdown_materially_worse"
    if float(test["worst_day_pnl"]) < float(base["worst_day_pnl"]) * 1.25:
        return "reject_scaling_loss_clustering_worse"
    if int(test["skipped_trades"]) > int(base["candidate_trades"]) * 0.20:
        return "reject_scaling_skips_too_many_frozen_entries"
    return "pass_offline_scaling_candidate_research_only"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Keep Tuesday paper trading at one contract anyway. Scaling remains offline-only until live shadow "
            "and one-contract paper behavior are proven."
        )
    return (
        "Reject scaling for initial paper trading. Keep one contract as the live-paper constraint and revisit sizing "
        "only after the one-contract path survives live shadow and paper replay."
    )


def premium_dollars(trade: dict[str, Any]) -> float | None:
    premium = number(trade.get("premium_paid"))
    if premium is not None:
        return premium
    ask = number(trade.get("entry_ask"))
    return None if ask is None else ask * CONTRACT_MULTIPLIER


def daily_rows(mode: str, daily: dict[str, float]) -> list[dict[str, Any]]:
    return [{"mode": mode, "session": key, "daily_pnl": round(value, 6)} for key, value in sorted(daily.items())]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 129: Protocol101 Offline Position Sizing",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 entries/exits remain frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Starting cash: `${payload['starting_cash']:,.0f}`",
        "- Tuesday live paper remains capped at one contract regardless of this offline result.",
        "",
        "## Results",
        "",
        "| mode | ending_cash | total_pnl | return | taken | contracts | skipped | max_qty | max_dd | worst_day |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        lines.append(
            "| "
            f"{row['mode']} | {money(row['ending_cash'])} | {money(row['total_pnl'])} | {pct(row['return_on_starting_cash'])} | "
            f"{row['taken_trades']} | {row['total_contracts']} | {row['skipped_trades']} | {row['max_quantity']} | "
            f"{money(row['max_drawdown'])} | {money(row['worst_day_pnl'])} |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "This protocol is intentionally isolated from Tuesday paper trading. The live-paper gate remains one contract, max one open position, and no broker endpoint without explicit approval.",
            "",
            "## Outputs",
            "",
            f"- Trade rows: `{payload['outputs']['sizing_trade_rows']}`",
            f"- Summary: `{payload['outputs']['summary']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 129 Protocol101 Offline Position Sizing"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Tested adaptive 1-to-3 contract sizing offline while keeping Protocol101 entries and exits frozen.
Reason: Position scaling should not be introduced into Tuesday paper trading until it proves it improves return without materially worsening drawdown or loss clustering.
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


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def money(value: Any) -> str:
    number_value = number(value)
    if number_value is None:
        return "n/a"
    sign = "-" if number_value < 0 else ""
    return f"{sign}${abs(number_value):,.0f}"


def pct(value: Any) -> str:
    number_value = number(value)
    return "n/a" if number_value is None else f"{number_value * 100:.1f}%"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
