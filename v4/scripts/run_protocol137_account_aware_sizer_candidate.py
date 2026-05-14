"""Protocol 137: consolidate the always-on account-aware sizing candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.run_protocol134_sizing_attribution import build_attribution, concentration_checks
from v4.scripts.run_protocol135_starting_cash_sensitivity import (
    STARTING_CASH_VALUES,
    baseline_policy,
    compare_to_baseline,
    enrich_summary,
    stress_trades,
)
from v4.sim.protocol101_position_sizing import account_aware_sizer_policy, simulate_position_sizing


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_137_protocol101_account_aware_sizer_candidate"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
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
    for starting_cash in STARTING_CASH_VALUES:
        for stress in STRESS_PER_SIDE:
            stressed = stress_trades(trades, stress)
            baseline = simulate_position_sizing(stressed, baseline_policy(starting_cash))
            candidate = simulate_position_sizing(stressed, account_aware_sizer_policy(starting_cash))
            rows.append({**enrich_summary(baseline["summary"], stress=stress, policy_kind="baseline"), "passes_cash_level_gate": True})
            rows.append(
                {
                    **enrich_summary(candidate["summary"], stress=stress, policy_kind="candidate"),
                    **compare_to_baseline(candidate["summary"], baseline["summary"]),
                }
            )

    baseline_10 = simulate_position_sizing(trades, baseline_policy(10_000.0))
    candidate_10 = simulate_position_sizing(trades, account_aware_sizer_policy(10_000.0))
    attribution = build_attribution(baseline_10["rows"], candidate_10["rows"])
    concentration = concentration_checks(attribution)
    decision = decide(rows, concentration)
    policy = account_aware_sizer_policy(10_000.0)
    payload = {
        "protocol": "137_protocol101_account_aware_sizer_candidate",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "source_trades_csv": str(args.trades_csv),
        "policy": policy.__dict__,
        "summary_rows": rows,
        "concentration": concentration,
        "baseline_10000": baseline_10["summary"],
        "candidate_10000": candidate_10["summary"],
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "candidate_config": str(args.out_dir / "account_aware_sizer_v1.json"),
            "validation_summary": str(args.out_dir / "validation_summary.csv"),
            "attribution_rows": str(args.out_dir / "attribution_rows.csv"),
        },
        "next_gate": next_gate(decision),
    }
    (args.out_dir / "account_aware_sizer_v1.json").write_text(json_dumps(policy.__dict__))
    pd.DataFrame(rows).to_csv(args.out_dir / "validation_summary.csv", index=False)
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


def decide(rows: list[dict[str, Any]], concentration: dict[str, Any]) -> str:
    candidate_rows = [row for row in rows if row["policy_kind"] == "candidate"]
    stress025 = [row for row in candidate_rows if abs(float(row["stress_per_side"]) - 0.25) < 1e-9]
    stress050 = [row for row in candidate_rows if abs(float(row["stress_per_side"]) - 0.50) < 1e-9]
    if not all(bool(row["passes_cash_level_gate"]) for row in stress025):
        return "fragile_sizer_candidate_fails_025_gate"
    if not all(float(row["incremental_pnl"]) > 0 for row in stress050):
        return "fragile_sizer_candidate_fails_050_positive_incremental"
    if float(concentration["top_day_share_of_positive"]) > 0.35:
        return "fragile_sizer_candidate_top_day_concentrated"
    if float(concentration["top_month_share_of_positive"]) > 0.55:
        return "fragile_sizer_candidate_top_month_concentrated"
    return "pass_account_aware_sizer_v1_research_candidate_not_live"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Use this config for offline visual/account-state reports. It is still not paper/live multi-contract approval."
        )
    return "Keep the prior fixed-stop sizing candidate and inspect the failed validation rows."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    c = payload["concentration"]
    lines = [
        "# Protocol 137: Account-Aware Sizer V1 Candidate",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Policy artifact: `{payload['outputs']['candidate_config']}`",
        f"- 10k candidate PnL: `{money(payload['candidate_10000']['total_pnl'])}`",
        f"- 10k baseline PnL: `{money(payload['baseline_10000']['total_pnl'])}`",
        f"- Incremental PnL: `{money(c['incremental_pnl'])}`",
        f"- Scaled trades: `{c['scaled_trades']}`",
        f"- Skipped trades: `{c['skipped_trades']}`",
        f"- Top day share: `{c['top_day_share_of_positive']:.1%}`",
        f"- Top month share: `{c['top_month_share_of_positive']:.1%}`",
        "",
        "## Validation",
        "",
        "| start | stress | policy | total_pnl | inc_pnl | max_dd | worst_day | pass |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| "
            f"{money(row['starting_cash'])} | ${float(row['stress_per_side']):.2f} | {row['policy_kind']} | "
            f"{money(row['total_pnl'])} | {money(row.get('incremental_pnl', 0.0))} | {money(row['max_drawdown'])} | "
            f"{money(row['worst_day_pnl'])} | `{row.get('passes_cash_level_gate', True)}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Config: `{payload['outputs']['candidate_config']}`",
            f"- Validation summary: `{payload['outputs']['validation_summary']}`",
            f"- Attribution rows: `{payload['outputs']['attribution_rows']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 137 Protocol101 Account-Aware Sizer V1"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Consolidated the best current account-aware multi-contract sizing rule into one offline candidate artifact.
Reason: The product goal is a turn-it-on policy that scales risk from account state, confidence, and drawdown without manual model selection.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Config {payload['outputs']['candidate_config']}. Report {payload['outputs']['report']}.
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
