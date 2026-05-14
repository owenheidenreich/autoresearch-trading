"""Protocol 132: confidence-aware multi-contract sizing autoresearch.

This is an offline sizing loop, not a live/paper trading change. It makes one
conceptual change after Protocol 131: extra contracts require account growth and
entry-time confidence, with profit-cushion and drawdown locks.
"""
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
    confidence_ladder_policy,
    default_position_sizing_policies,
    high_conviction_profit_cushion_policy,
    simulate_position_sizing,
    slow_growth_two_contract_policy,
)


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_132_protocol101_confidence_sizing_autoresearch"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
MAX_FAILED_HYPOTHESES = 3


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
    baseline = simulate_position_sizing(trades, baseline_one_contract_policy())
    experiments = build_experiments()
    loop_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    failed_in_row = 0
    all_trade_rows: list[dict[str, Any]] = baseline["rows"]
    all_daily_rows: list[dict[str, Any]] = baseline["daily"]

    baseline_summary = baseline["summary"]
    for index, hypothesis in enumerate(experiments, start=1):
        run = simulate_position_sizing(trades, hypothesis["policy"])
        all_trade_rows.extend(run["rows"])
        all_daily_rows.extend(run["daily"])
        evaluation = evaluate_candidate(run["summary"], baseline_summary)
        row = {
            "index": index,
            "hypothesis": hypothesis["name"],
            "change": hypothesis["change"],
            **run["summary"],
            **evaluation,
        }
        loop_rows.append(row)
        if evaluation["passes_acceptance"]:
            accepted.append(row)
            failed_in_row = 0
        else:
            failed_in_row += 1
        if failed_in_row >= MAX_FAILED_HYPOTHESES:
            break

    decision = decide(accepted, failed_in_row)
    payload = {
        "protocol": "132_protocol101_confidence_sizing_autoresearch",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "source_trades_csv": str(args.trades_csv),
        "baseline": baseline_summary,
        "max_failed_hypotheses": MAX_FAILED_HYPOTHESES,
        "loop_rows": loop_rows,
        "accepted": accepted,
        "stopped_after_failed_hypotheses": failed_in_row >= MAX_FAILED_HYPOTHESES,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "loop_results": str(args.out_dir / "loop_results.csv"),
            "trade_rows": str(args.out_dir / "sizing_trade_rows.csv"),
            "daily_rows": str(args.out_dir / "sizing_daily_rows.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(loop_rows).to_csv(args.out_dir / "loop_results.csv", index=False)
    pd.DataFrame(all_trade_rows).to_csv(args.out_dir / "sizing_trade_rows.csv", index=False)
    pd.DataFrame(all_daily_rows).to_csv(args.out_dir / "sizing_daily_rows.csv", index=False)
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


def build_experiments() -> list[dict[str, Any]]:
    base = confidence_ladder_policy()
    high = high_conviction_profit_cushion_policy()
    slow = slow_growth_two_contract_policy()
    return [
        {
            "name": "132a_confidence_ladder",
            "change": "Add score-margin thresholds plus 3% drawdown lock and profit cushion.",
            "policy": base,
        },
        {
            "name": "132b_high_conviction_profit_cushion",
            "change": "Raise score thresholds, require larger profit cushion, cap scalable premiums.",
            "policy": high,
        },
        {
            "name": "132c_slow_growth_two_contract",
            "change": "Allow only two contracts after larger account growth with stricter exposure cap.",
            "policy": slow,
        },
        {
            "name": "132d_lower_two_contract_threshold",
            "change": "Relax the two-contract score threshold while keeping profit/drawdown locks.",
            "policy": replace(high, name="lower_two_contract_threshold", min_score_margin_for_two=2.0),
        },
        {
            "name": "132e_delayed_three_contract_unlock",
            "change": "Use confidence ladder but delay three-contract sizing until $125k equity.",
            "policy": replace(base, name="delayed_three_contract_unlock", equity_for_three_contracts=125_000.0),
        },
        {
            "name": "132f_prior_131_policy_regression_check",
            "change": "Retest the previous non-confidence policies after adding the same acceptance gate.",
            "policy": next(policy for policy in default_position_sizing_policies() if policy.name == "recovery_lock_ladder"),
        },
    ]


def evaluate_candidate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    total_improved = float(candidate["total_pnl"]) > float(baseline["total_pnl"])
    drawdown_ok = abs(float(candidate["max_drawdown_pct"])) <= abs(float(baseline["max_drawdown_pct"])) + 0.03
    worst_day_ok = float(candidate["worst_day_pnl"]) >= float(baseline["worst_day_pnl"]) * 1.5
    skip_ok = int(candidate["skipped_trades"]) <= int(candidate["candidate_trades"]) * 0.20
    risk_ok = not bool(candidate["risk_of_ruin"])
    return {
        "baseline_total_pnl": baseline["total_pnl"],
        "baseline_max_drawdown_pct": baseline["max_drawdown_pct"],
        "baseline_worst_day_pnl": baseline["worst_day_pnl"],
        "total_improved": total_improved,
        "drawdown_ok": drawdown_ok,
        "worst_day_ok": worst_day_ok,
        "skip_ok": skip_ok,
        "risk_ok": risk_ok,
        "passes_acceptance": bool(total_improved and drawdown_ok and worst_day_ok and skip_ok and risk_ok),
    }


def decide(accepted: list[dict[str, Any]], failed_in_row: int) -> str:
    if accepted:
        return "pass_confidence_sizing_research_candidate_not_live"
    if failed_in_row >= MAX_FAILED_HYPOTHESES:
        return "pause_after_three_failed_sizing_hypotheses"
    return "reject_confidence_sizing_no_candidate"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Treat the accepted policy as offline research only. It needs split-by-split attribution and live "
            "one-contract parity before any multi-contract paper test."
        )
    if decision.startswith("pause_"):
        return (
            "Pause sizing tweaks and inspect why confidence/account-aware policies still worsen loss clustering. "
            "Do not continue adding knobs without attribution."
        )
    return "Keep initial paper/live path one contract only."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 132: Confidence-Aware Sizing Autoresearch",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Baseline total PnL: `{money(payload['baseline']['total_pnl'])}`",
        f"- Stopped after 3 failed hypotheses: `{payload['stopped_after_failed_hypotheses']}`",
        "",
        "## Loop Results",
        "",
        "| # | hypothesis | total_pnl | max_dd | worst_day | skipped | max_qty | pass |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["loop_rows"]:
        lines.append(
            "| "
            f"{row['index']} | {row['hypothesis']} | {money(row['total_pnl'])} | {money(row['max_drawdown'])} | "
            f"{money(row['worst_day_pnl'])} | {row['skipped_trades']} | {row['max_quantity']} | `{row['passes_acceptance']}` |"
        )
    lines.extend(
        [
            "",
            "## Accepted",
            "",
            "`" + ", ".join(row["hypothesis"] for row in payload["accepted"]) + "`" if payload["accepted"] else "`none`",
            "",
            "## Outputs",
            "",
            f"- Loop results: `{payload['outputs']['loop_results']}`",
            f"- Trade rows: `{payload['outputs']['trade_rows']}`",
            f"- Daily rows: `{payload['outputs']['daily_rows']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 132 Protocol101 Confidence Sizing Autoresearch"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Ran a confidence/account-aware sizing autoresearch loop around frozen Protocol101.
Reason: Long-term bot behavior may scale contracts as a $10,000 account grows, but only if account state and model confidence reduce the loss-clustering problem.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Accepted {len(payload['accepted'])} sizing hypotheses. Report {payload['outputs']['report']}.
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
