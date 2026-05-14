"""Protocol 131: offline multi-contract P&L tracking experiment.

This is deliberately post-commit research only. It never mutates Protocol 101,
never downloads data, and never touches IBKR. It tests whether simple
profit-ladder sizing can improve the frozen trade stream without wrecking
drawdown, worst-day loss, or skipped-entry behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.sim.protocol101_position_sizing import (
    default_position_sizing_policies,
    simulate_position_sizing,
)


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_131_protocol101_multi_contract_pnl_tracking"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


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
    runs = [simulate_position_sizing(trades, policy) for policy in default_position_sizing_policies()]
    summary_rows = [run["summary"] for run in runs]
    trade_rows = [row for run in runs for row in run["rows"]]
    daily_rows = [row for run in runs for row in run["daily"]]
    decision = decide(summary_rows)
    payload = {
        "protocol": "131_protocol101_multi_contract_pnl_tracking",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "source_trades_csv": str(args.trades_csv),
        "summary": summary_rows,
        "acceptance_rule": {
            "scope": "offline research only",
            "must_improve_total_pnl_vs_baseline": True,
            "must_not_increase_max_drawdown_pct_by_more_than": 0.03,
            "must_not_make_worst_day_more_than_150pct_of_baseline": True,
            "must_not_skip_more_than_20pct_of_candidates": True,
            "must_have_no_risk_of_ruin": True,
        },
        "best_candidate": best_candidate(summary_rows),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "sizing_summary": str(args.out_dir / "sizing_summary.csv"),
            "sizing_trade_rows": str(args.out_dir / "sizing_trade_rows.csv"),
            "sizing_daily_rows": str(args.out_dir / "sizing_daily_rows.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "sizing_summary.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(args.out_dir / "sizing_trade_rows.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(args.out_dir / "sizing_daily_rows.csv", index=False)
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


def decide(summary_rows: list[dict[str, Any]]) -> str:
    candidate = best_candidate(summary_rows)
    if not candidate:
        return "reject_multi_contract_no_candidate"
    if candidate["risk_of_ruin"]:
        return "reject_multi_contract_risk_of_ruin"
    if not candidate["passes_acceptance"]:
        return "reject_multi_contract_risk_not_improved_enough"
    return "pass_multi_contract_offline_research_candidate_not_live"


def best_candidate(summary_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = next((row for row in summary_rows if row["policy"] == "one_contract_baseline"), None)
    if baseline is None:
        return None
    candidates = [row for row in summary_rows if row["policy"] != "one_contract_baseline"]
    if not candidates:
        return None
    enriched = [with_acceptance(row, baseline) for row in candidates]
    return max(enriched, key=lambda row: (row["passes_acceptance"], row["total_pnl"], -abs(row["max_drawdown_pct"])))


def with_acceptance(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    total_improved = float(row["total_pnl"]) > float(baseline["total_pnl"])
    drawdown_ok = abs(float(row["max_drawdown_pct"])) <= abs(float(baseline["max_drawdown_pct"])) + 0.03
    worst_day_ok = float(row["worst_day_pnl"]) >= float(baseline["worst_day_pnl"]) * 1.5
    skip_ok = int(row["skipped_trades"]) <= int(row["candidate_trades"]) * 0.20
    risk_ok = not bool(row["risk_of_ruin"])
    return {
        **row,
        "baseline_total_pnl": baseline["total_pnl"],
        "baseline_max_drawdown_pct": baseline["max_drawdown_pct"],
        "baseline_worst_day_pnl": baseline["worst_day_pnl"],
        "passes_acceptance": bool(total_improved and drawdown_ok and worst_day_ok and skip_ok and risk_ok),
        "acceptance_checks": {
            "total_improved": total_improved,
            "drawdown_ok": drawdown_ok,
            "worst_day_ok": worst_day_ok,
            "skip_ok": skip_ok,
            "risk_ok": risk_ok,
        },
    }


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Keep this as offline research only. Do not enable multi-contract paper trading until one-contract live "
            "shadow and one-contract paper behavior are proven."
        )
    return (
        "Keep Tuesday and initial paper trading at one contract. Multi-contract sizing remains rejected until a safer "
        "policy improves PnL without worsening drawdown or loss clustering."
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 131: Protocol101 Multi-Contract P&L Tracking",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        "- Scope: offline research only; Tuesday paper path remains one contract.",
        "",
        "## Policy Results",
        "",
        "| policy | ending_cash | total_pnl | return | taken | skipped | contracts | max_qty | max_dd | worst_day |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| "
            f"{row['policy']} | {money(row['ending_cash'])} | {money(row['total_pnl'])} | {pct(row['return_on_starting_cash'])} | "
            f"{row['taken_trades']} | {row['skipped_trades']} | {row['total_contracts']} | {row['max_quantity']} | "
            f"{money(row['max_drawdown'])} | {money(row['worst_day_pnl'])} |"
        )
    candidate = payload.get("best_candidate") or {}
    lines.extend(
        [
            "",
            "## Best Offline Candidate",
            "",
            f"- Policy: `{candidate.get('policy', 'none')}`",
            f"- Passes acceptance: `{candidate.get('passes_acceptance', False)}`",
            f"- Checks: `{candidate.get('acceptance_checks', {})}`",
            "",
            "## Outputs",
            "",
            f"- Summary CSV: `{payload['outputs']['sizing_summary']}`",
            f"- Trade rows: `{payload['outputs']['sizing_trade_rows']}`",
            f"- Daily rows: `{payload['outputs']['sizing_daily_rows']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 131 Protocol101 Multi-Contract P&L Tracking"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Tested offline multi-contract P&L tracking policies around frozen Protocol101.
Reason: User asked whether the model could scale lots only after profits; this requires account-state tracking before any leverage touches paper trading.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Best candidate {payload.get('best_candidate', {}).get('policy')}. Report {payload['outputs']['report']}.
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


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
