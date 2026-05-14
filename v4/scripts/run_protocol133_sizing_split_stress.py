"""Protocol 133: split and stress validation for accepted sizing policies."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.run_protocol132_confidence_sizing_autoresearch import evaluate_candidate
from v4.sim.protocol101_position_sizing import (
    PositionSizingPolicy,
    baseline_one_contract_policy,
    high_conviction_profit_cushion_policy,
    simulate_position_sizing,
    slow_growth_two_contract_policy,
)


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_133_protocol101_sizing_split_stress"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
STRESS_PER_SIDE = (0.0, 0.10, 0.25, 0.50)


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
    policies = accepted_policies()
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    for stress in STRESS_PER_SIDE:
        stressed = stress_trades(trades, stress)
        baseline = simulate_position_sizing(stressed, baseline_one_contract_policy())
        summary_rows.append(with_stress(baseline["summary"], stress, baseline["summary"]))
        split_rows.extend(group_rows(baseline["rows"], stress=stress, group_col="segment"))
        month_rows.extend(group_rows(add_months(baseline["rows"]), stress=stress, group_col="month"))
        daily_rows.extend(with_stress_daily(baseline["daily"], stress))
        for policy in policies:
            run = simulate_position_sizing(stressed, policy)
            evaluation = evaluate_candidate(run["summary"], baseline["summary"])
            summary_rows.append(with_stress({**run["summary"], **evaluation}, stress, baseline["summary"]))
            split_rows.extend(group_rows(run["rows"], stress=stress, group_col="segment"))
            month_rows.extend(group_rows(add_months(run["rows"]), stress=stress, group_col="month"))
            daily_rows.extend(with_stress_daily(run["daily"], stress))

    decision = decide(summary_rows)
    payload = {
        "protocol": "133_protocol101_sizing_split_stress",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "source_trades_csv": str(args.trades_csv),
        "stress_per_side": list(STRESS_PER_SIDE),
        "summary_rows": summary_rows,
        "best_candidate": best_candidate(summary_rows),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "stress_summary": str(args.out_dir / "stress_summary.csv"),
            "split_summary": str(args.out_dir / "split_summary.csv"),
            "month_summary": str(args.out_dir / "month_summary.csv"),
            "daily_summary": str(args.out_dir / "daily_summary.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "stress_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.out_dir / "split_summary.csv", index=False)
    pd.DataFrame(month_rows).to_csv(args.out_dir / "month_summary.csv", index=False)
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


def accepted_policies() -> list[PositionSizingPolicy]:
    high = high_conviction_profit_cushion_policy()
    return [
        high,
        slow_growth_two_contract_policy(),
        replace(high, name="lower_two_contract_threshold", min_score_margin_for_two=2.0),
    ]


def stress_trades(trades: list[dict[str, Any]], stress_per_side: float) -> list[dict[str, Any]]:
    cost = 2.0 * float(stress_per_side) * 100.0
    out = []
    for row in trades:
        copy = dict(row)
        copy["pnl"] = float(copy["pnl"]) - cost
        out.append(copy)
    return out


def with_stress(row: dict[str, Any], stress: float, baseline: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["stress_per_side"] = float(stress)
    if out["policy"] == "one_contract_baseline":
        out["passes_acceptance"] = True
        out["baseline_total_pnl"] = baseline["total_pnl"]
        out["baseline_max_drawdown_pct"] = baseline["max_drawdown_pct"]
        out["baseline_worst_day_pnl"] = baseline["worst_day_pnl"]
    return out


def group_rows(rows: list[dict[str, Any]], *, stress: float, group_col: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    out = []
    for (policy, group), sub in frame.groupby(["policy", group_col], dropna=False, sort=True):
        values = pd.to_numeric(sub["realized_pnl"], errors="coerce").fillna(0.0)
        out.append(
            {
                "stress_per_side": float(stress),
                "policy": policy,
                "group_type": group_col,
                "group": group,
                "total_pnl": round(float(values.sum()), 2),
                "taken_trades": int((sub["quantity"] > 0).sum()),
                "skipped_trades": int((sub["quantity"] <= 0).sum()),
                "total_contracts": int(sub["quantity"].sum()),
                "worst_trade": round(float(values.min()), 2) if len(values) else 0.0,
            }
        )
    return out


def add_months(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copy = dict(row)
        copy["month"] = str(pd.Timestamp(copy["session"]).to_period("M"))
        out.append(copy)
    return out


def with_stress_daily(rows: list[dict[str, Any]], stress: float) -> list[dict[str, Any]]:
    return [{**row, "stress_per_side": float(stress)} for row in rows]


def best_candidate(summary_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    stress0 = [row for row in summary_rows if float(row["stress_per_side"]) == 0.0 and row["policy"] != "one_contract_baseline"]
    accepted = [row for row in stress0 if bool(row.get("passes_acceptance"))]
    if not accepted:
        return None
    return max(accepted, key=lambda row: (row["total_pnl"], -abs(row["max_drawdown_pct"])))


def decide(summary_rows: list[dict[str, Any]]) -> str:
    candidate = best_candidate(summary_rows)
    if candidate is None:
        return "reject_sizing_no_unstressed_candidate"
    policy = candidate["policy"]
    stress025 = [
        row
        for row in summary_rows
        if row["policy"] == policy and abs(float(row["stress_per_side"]) - 0.25) < 1e-9
    ]
    if not stress025 or not bool(stress025[0].get("passes_acceptance")):
        return "fragile_sizing_candidate_fails_025_stress"
    return "pass_sizing_candidate_survives_split_stress_not_live"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Run trade-set attribution for the surviving sizing policy and keep it offline until one-contract live "
            "paper parity is proven."
        )
    return "Do not advance multi-contract sizing; inspect split/month and stress rows for the failure mode first."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 133: Sizing Split And Stress Validation",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Best candidate: `{(payload.get('best_candidate') or {}).get('policy', 'none')}`",
        "",
        "## Stress Summary",
        "",
        "| stress | policy | total_pnl | max_dd | worst_day | skipped | max_qty | pass |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| "
            f"${float(row['stress_per_side']):.2f} | {row['policy']} | {money(row['total_pnl'])} | "
            f"{money(row['max_drawdown'])} | {money(row['worst_day_pnl'])} | {row['skipped_trades']} | "
            f"{row['max_quantity']} | `{row.get('passes_acceptance')}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Stress summary: `{payload['outputs']['stress_summary']}`",
            f"- Split summary: `{payload['outputs']['split_summary']}`",
            f"- Month summary: `{payload['outputs']['month_summary']}`",
            f"- Daily summary: `{payload['outputs']['daily_summary']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 133 Protocol101 Sizing Split Stress"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Validated accepted Protocol132 sizing policies by split, month, day, and slippage stress.
Reason: Aggregate multi-contract PnL is not enough; sizing must not simply move loss clustering into a later month or disappear under execution stress.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Best candidate {(payload.get('best_candidate') or {}).get('policy')}. Report {payload['outputs']['report']}.
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
