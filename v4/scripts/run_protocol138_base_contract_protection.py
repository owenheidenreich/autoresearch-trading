"""Protocol 138: protect the base one-contract entry from scaling caps."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.run_protocol134_sizing_attribution import build_attribution, concentration_checks
from v4.scripts.run_protocol135_starting_cash_sensitivity import baseline_policy, stress_trades
from v4.sim.protocol101_position_sizing import account_aware_sizer_policy, simulate_position_sizing


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_138_protocol101_base_contract_protection"
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
    old_policy = replace(account_aware_sizer_policy(10_000.0), name="account_aware_sizer_v1_cap_blocks_base", exposure_cap_applies_to_initial_contract=True)
    new_policy = account_aware_sizer_policy(10_000.0)
    baseline = simulate_position_sizing(trades, baseline_policy(10_000.0))
    old = simulate_position_sizing(trades, old_policy)
    new = simulate_position_sizing(trades, new_policy)
    attribution = build_attribution(baseline["rows"], new["rows"])
    segment_rows = group_incremental(attribution, "segment")
    month_rows = group_incremental(attribution, "month")
    concentration = concentration_checks(attribution)
    stress_rows = stress_compare(trades, baseline_policy(10_000.0), new_policy)
    decision = decide(segment_rows, stress_rows, new["summary"], old["summary"])
    payload = {
        "protocol": "138_protocol101_base_contract_protection",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "old_policy": old_policy.__dict__,
        "new_policy": new_policy.__dict__,
        "baseline_summary": baseline["summary"],
        "old_summary": old["summary"],
        "new_summary": new["summary"],
        "concentration": concentration,
        "segment_rows": segment_rows,
        "month_rows": month_rows,
        "stress_rows": stress_rows,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "segment_summary": str(args.out_dir / "segment_summary.csv"),
            "month_summary": str(args.out_dir / "month_summary.csv"),
            "stress_summary": str(args.out_dir / "stress_summary.csv"),
            "attribution_rows": str(args.out_dir / "attribution_rows.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(segment_rows).to_csv(args.out_dir / "segment_summary.csv", index=False)
    pd.DataFrame(month_rows).to_csv(args.out_dir / "month_summary.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(args.out_dir / "stress_summary.csv", index=False)
    pd.DataFrame(attribution).to_csv(args.out_dir / "attribution_rows.csv", index=False)
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


def group_incremental(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    out = []
    for key, group in frame.groupby(column, dropna=False, sort=True):
        out.append(
            {
                column: key,
                "incremental_pnl": round(float(group["incremental_pnl"].sum()), 2),
                "scaled_trades": int(group["is_scaled"].sum()),
                "skipped_trades": int(group["is_skipped"].sum()),
            }
        )
    return out


def stress_compare(trades: list[dict[str, Any]], baseline_policy_obj: Any, candidate_policy_obj: Any) -> list[dict[str, Any]]:
    out = []
    for stress in (0.0, 0.25, 0.50):
        stressed = stress_trades(trades, stress)
        baseline = simulate_position_sizing(stressed, baseline_policy_obj)
        candidate = simulate_position_sizing(stressed, candidate_policy_obj)
        out.append(
            {
                "stress_per_side": stress,
                "baseline_pnl": baseline["summary"]["total_pnl"],
                "candidate_pnl": candidate["summary"]["total_pnl"],
                "incremental_pnl": round(candidate["summary"]["total_pnl"] - baseline["summary"]["total_pnl"], 2),
                "candidate_max_drawdown": candidate["summary"]["max_drawdown"],
                "candidate_worst_day": candidate["summary"]["worst_day_pnl"],
            }
        )
    return out


def decide(
    segment_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    new_summary: dict[str, Any],
    old_summary: dict[str, Any],
) -> str:
    q4_external = next((row for row in segment_rows if row.get("segment") == "q4_2024_external"), None)
    if q4_external is not None and float(q4_external["incremental_pnl"]) < 0:
        return "reject_base_contract_protection_still_external_negative"
    if any(float(row["incremental_pnl"]) <= 0 for row in stress_rows):
        return "reject_base_contract_protection_stress_negative"
    if abs(float(new_summary["max_drawdown"])) > abs(float(old_summary["max_drawdown"])) + 1_000.0:
        return "reject_base_contract_protection_drawdown_worse"
    return "pass_base_contract_protection_candidate_not_live"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return "Replace the prior account-aware sizer artifact with this safer semantics and rerun starting-cash validation."
    return "Do not keep this change; inspect the segment summary for why base-contract protection failed."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 138: Base Contract Protection",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Old total PnL: `{money(payload['old_summary']['total_pnl'])}`",
        f"- New total PnL: `{money(payload['new_summary']['total_pnl'])}`",
        f"- New skipped trades: `{payload['new_summary']['skipped_trades']}`",
        "",
        "## Segment Incremental vs One Contract",
        "",
        "| segment | incremental | scaled | skipped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["segment_rows"]:
        lines.append(f"| {row['segment']} | {money(row['incremental_pnl'])} | {row['scaled_trades']} | {row['skipped_trades']} |")
    lines.extend(
        [
            "",
            "## Stress",
            "",
            "| stress | incremental | max_dd | worst_day |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["stress_rows"]:
        lines.append(
            f"| ${row['stress_per_side']:.2f} | {money(row['incremental_pnl'])} | "
            f"{money(row['candidate_max_drawdown'])} | {money(row['candidate_worst_day'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Segment summary: `{payload['outputs']['segment_summary']}`",
            f"- Stress summary: `{payload['outputs']['stress_summary']}`",
            f"- Attribution rows: `{payload['outputs']['attribution_rows']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 138 Protocol101 Base Contract Protection"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Tested whether premium exposure caps should protect scaling without blocking the base one-contract trade.
Reason: The prior account-aware sizer underperformed the Q4 2024 external block partly because a scaling cap could skip ordinary one-contract trades.
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
