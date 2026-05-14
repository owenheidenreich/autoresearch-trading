"""Protocol 135: starting-cash sensitivity for one always-on sizing policy."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from v4.sim.protocol101_position_sizing import (
    PositionSizingPolicy,
    baseline_one_contract_policy,
    high_conviction_profit_cushion_policy,
    simulate_position_sizing,
)


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_135_protocol101_starting_cash_sensitivity"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
STARTING_CASH_VALUES = (10_000.0, 25_000.0, 30_000.0, 50_000.0, 100_000.0)
STRESS_PER_SIDE = (0.0, 0.25, 0.50)


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
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    policies = [baseline_policy, active_sizing_policy]
    for starting_cash in STARTING_CASH_VALUES:
        for stress in STRESS_PER_SIDE:
            stressed = stress_trades(trades, stress)
            baseline = simulate_position_sizing(stressed, baseline_policy(starting_cash))
            baseline_summary = enrich_summary(baseline["summary"], stress=stress, policy_kind="baseline")
            rows.append(baseline_summary)
            daily_rows.extend(enrich_daily(baseline["daily"], starting_cash=starting_cash, stress=stress))
            candidate = simulate_position_sizing(stressed, active_sizing_policy(starting_cash))
            rows.append(
                {
                    **enrich_summary(candidate["summary"], stress=stress, policy_kind="candidate"),
                    **compare_to_baseline(candidate["summary"], baseline["summary"]),
                }
            )
            daily_rows.extend(enrich_daily(candidate["daily"], starting_cash=starting_cash, stress=stress))

    decision = decide(rows)
    payload = {
        "protocol": "135_protocol101_starting_cash_sensitivity",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "source_trades_csv": str(args.trades_csv),
        "starting_cash_values": list(STARTING_CASH_VALUES),
        "stress_per_side": list(STRESS_PER_SIDE),
        "active_policy": active_sizing_policy(10_000.0).__dict__,
        "rows": rows,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "starting_cash_summary": str(args.out_dir / "starting_cash_summary.csv"),
            "daily_summary": str(args.out_dir / "daily_summary.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(rows).to_csv(args.out_dir / "starting_cash_summary.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(args.out_dir / "daily_summary.csv", index=False)
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
    return [{key: (None if pd.isna(value) else value) for key, value in row.items()} for row in frame.to_dict("records")]


def baseline_policy(starting_cash: float) -> PositionSizingPolicy:
    return replace(baseline_one_contract_policy(), starting_cash=float(starting_cash))


def active_sizing_policy(starting_cash: float) -> PositionSizingPolicy:
    return replace(
        high_conviction_profit_cushion_policy(),
        name="always_on_account_aware_sizer",
        starting_cash=float(starting_cash),
        min_score_margin_for_two=2.0,
    )


def stress_trades(trades: list[dict[str, Any]], stress_per_side: float) -> list[dict[str, Any]]:
    cost = 2.0 * float(stress_per_side) * 100.0
    out = []
    for row in trades:
        copy = dict(row)
        copy["pnl"] = float(copy["pnl"]) - cost
        out.append(copy)
    return out


def enrich_summary(row: dict[str, Any], *, stress: float, policy_kind: str) -> dict[str, Any]:
    starting = float(row["starting_cash"])
    out = dict(row)
    out["policy_kind"] = policy_kind
    out["stress_per_side"] = float(stress)
    out["max_drawdown_pct_start"] = round(abs(float(row["max_drawdown"])) / starting, 6)
    out["worst_day_pct_start"] = round(abs(float(row["worst_day_pnl"])) / starting, 6)
    out["return_over_max_drawdown"] = round(float(row["total_pnl"]) / max(abs(float(row["max_drawdown"])), 1.0), 6)
    out["return_over_worst_day"] = round(float(row["total_pnl"]) / max(abs(float(row["worst_day_pnl"])), 1.0), 6)
    return out


def compare_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    total_improved = float(candidate["total_pnl"]) > float(baseline["total_pnl"])
    dd_pct_ok = abs(float(candidate["max_drawdown_pct"])) <= abs(float(baseline["max_drawdown_pct"])) + 0.04
    worst_day_pct_ok = (
        abs(float(candidate["worst_day_pnl"])) / float(candidate["starting_cash"])
        <= abs(float(baseline["worst_day_pnl"])) / float(baseline["starting_cash"]) + 0.03
    )
    return {
        "baseline_total_pnl": baseline["total_pnl"],
        "incremental_pnl": round(float(candidate["total_pnl"]) - float(baseline["total_pnl"]), 2),
        "total_improved": total_improved,
        "drawdown_pct_ok": dd_pct_ok,
        "worst_day_pct_ok": worst_day_pct_ok,
        "passes_cash_level_gate": bool(total_improved and dd_pct_ok and worst_day_pct_ok and not bool(candidate["risk_of_ruin"])),
    }


def enrich_daily(rows: list[dict[str, Any]], *, starting_cash: float, stress: float) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "starting_cash": float(starting_cash),
            "stress_per_side": float(stress),
            "daily_pnl_pct_start": round(float(row["daily_pnl"]) / float(starting_cash), 6),
        }
        for row in rows
    ]


def decide(rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in rows if row.get("policy_kind") == "candidate"]
    if not candidates:
        return "reject_starting_cash_no_candidate_rows"
    hard_failures = [
        row for row in candidates if not bool(row.get("passes_cash_level_gate")) or bool(row.get("risk_of_ruin"))
    ]
    stress025_failures = [
        row for row in candidates if abs(float(row["stress_per_side"]) - 0.25) < 1e-9 and not bool(row.get("passes_cash_level_gate"))
    ]
    if stress025_failures:
        return "fragile_sizer_fails_025_starting_cash_sensitivity"
    if hard_failures:
        return "fragile_sizer_has_starting_cash_failures"
    return "pass_always_on_sizer_starting_cash_sensitivity_not_live"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Promote the account-aware sizer to an offline research candidate artifact, then test equity visualizations "
            "and live-shadow-compatible account-state serialization."
        )
    return (
        "Do not promote the sizer. Find whether the failure comes from starting cash, stress, drawdown, or worst-day loss."
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 135: Starting-Cash Sensitivity",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Active policy: `{payload['active_policy']['name']}`",
        "",
        "## Summary",
        "",
        "| start | stress | policy | total_pnl | inc_pnl | max_dd | worst_day | contracts | pass |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{money(row['starting_cash'])} | ${float(row['stress_per_side']):.2f} | {row['policy_kind']} | "
            f"{money(row['total_pnl'])} | {money(row.get('incremental_pnl', 0.0))} | {money(row['max_drawdown'])} | "
            f"{money(row['worst_day_pnl'])} | {row['total_contracts']} | `{row.get('passes_cash_level_gate', True)}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Starting cash summary: `{payload['outputs']['starting_cash_summary']}`",
            f"- Daily summary: `{payload['outputs']['daily_summary']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 135 Protocol101 Starting-Cash Sensitivity"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Tested one always-on account-aware sizing policy across multiple starting paper account sizes and slippage stresses.
Reason: The production goal is a turn-it-on bot that scales risk from account state, not manual model selection.
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


def money(value: Any) -> str:
    number = float(value)
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
