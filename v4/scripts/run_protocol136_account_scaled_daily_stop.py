"""Protocol 136: account-scaled daily loss stop for the always-on sizer."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.run_protocol135_starting_cash_sensitivity import (
    STARTING_CASH_VALUES,
    active_sizing_policy,
    baseline_policy,
    compare_to_baseline,
    enrich_summary,
    stress_trades,
)
from v4.sim.protocol101_position_sizing import PositionSizingPolicy, simulate_position_sizing


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_136_protocol101_account_scaled_daily_stop"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
STRESS_PER_SIDE = (0.0, 0.25, 0.50)
FRACTIONS = (0.0025, 0.005, 0.0075, 0.01)
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
    current_policy_rows = evaluate_policy_grid(trades, "current_fixed_750", active_sizing_policy)
    current_candidate_rows = [row for row in current_policy_rows if row["policy_kind"] == "candidate"]
    experiments: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    failed = 0
    all_rows = list(current_policy_rows)
    for fraction in FRACTIONS:
        name = f"daily_stop_{fraction:.2%}_equity"
        rows = evaluate_policy_grid(
            trades,
            name,
            lambda starting, fraction=fraction, name=name: scaled_daily_stop_policy(starting, fraction, name),
        )
        all_rows.extend(rows)
        evaluation = evaluate_experiment(name, rows, current_candidate_rows)
        experiments.append(evaluation)
        if evaluation["passes"]:
            accepted.append(evaluation)
            failed = 0
        else:
            failed += 1
        if failed >= MAX_FAILED_HYPOTHESES:
            break

    decision = decide(accepted, failed)
    payload = {
        "protocol": "136_protocol101_account_scaled_daily_stop",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "source_trades_csv": str(args.trades_csv),
        "baseline_policy": active_sizing_policy(10_000.0).__dict__,
        "experiments": experiments,
        "accepted": accepted,
        "stopped_after_failed_hypotheses": failed >= MAX_FAILED_HYPOTHESES,
        "rows": all_rows,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "daily_stop_summary": str(args.out_dir / "daily_stop_summary.csv"),
            "experiment_summary": str(args.out_dir / "experiment_summary.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(all_rows).to_csv(args.out_dir / "daily_stop_summary.csv", index=False)
    pd.DataFrame(experiments).to_csv(args.out_dir / "experiment_summary.csv", index=False)
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


def scaled_daily_stop_policy(starting_cash: float, fraction: float, name: str) -> PositionSizingPolicy:
    base = active_sizing_policy(starting_cash)
    return replace(base, name=name, daily_new_entry_stop_fraction_of_equity=float(fraction))


def evaluate_policy_grid(
    trades: list[dict[str, Any]],
    experiment: str,
    policy_factory: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for starting_cash in STARTING_CASH_VALUES:
        for stress in STRESS_PER_SIDE:
            stressed = stress_trades(trades, stress)
            baseline = simulate_position_sizing(stressed, baseline_policy(starting_cash))
            candidate = simulate_position_sizing(stressed, policy_factory(starting_cash))
            out.append({**enrich_summary(baseline["summary"], stress=stress, policy_kind="baseline"), "experiment": experiment})
            out.append(
                {
                    **enrich_summary(candidate["summary"], stress=stress, policy_kind="candidate"),
                    **compare_to_baseline(candidate["summary"], baseline["summary"]),
                    "experiment": experiment,
                }
            )
    return out


def evaluate_experiment(
    name: str,
    rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [row for row in rows if row["policy_kind"] == "candidate"]
    current_by_key = {
        (float(row["starting_cash"]), float(row["stress_per_side"])): row for row in current_rows
    }
    improvements = []
    risk_ok = []
    for row in candidates:
        current = current_by_key[(float(row["starting_cash"]), float(row["stress_per_side"]))]
        improvements.append(float(row["total_pnl"]) - float(current["total_pnl"]))
        risk_ok.append(
            bool(row["passes_cash_level_gate"])
            and abs(float(row["max_drawdown"])) <= abs(float(current["max_drawdown"])) + 1_000.0
            and abs(float(row["worst_day_pnl"])) <= abs(float(current["worst_day_pnl"])) + 500.0
        )
    stress025 = [row for row in candidates if abs(float(row["stress_per_side"]) - 0.25) < 1e-9]
    passes = bool(
        improvements
        and sum(improvements) > 0
        and all(risk_ok)
        and all(float(row["incremental_pnl"]) > 0 for row in stress025)
    )
    return {
        "experiment": name,
        "passes": passes,
        "total_incremental_vs_current": round(sum(improvements), 2),
        "min_incremental_vs_current": round(min(improvements) if improvements else 0.0, 2),
        "risk_ok_all": all(risk_ok),
        "candidate_rows": len(candidates),
    }


def decide(accepted: list[dict[str, Any]], failed: int) -> str:
    if accepted:
        return "pass_account_scaled_daily_stop_candidate_not_live"
    if failed >= MAX_FAILED_HYPOTHESES:
        return "pause_after_three_failed_daily_stop_hypotheses"
    return "reject_account_scaled_daily_stop_no_candidate"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Fold the accepted daily-stop rule into the single offline sizing candidate and rerun attribution/visualization."
        )
    return "Keep the fixed-stop sizer and inspect skipped recovery trades before adding another sizing knob."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 136: Account-Scaled Daily Stop",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Stopped after 3 failed hypotheses: `{payload['stopped_after_failed_hypotheses']}`",
        "",
        "## Experiments",
        "",
        "| experiment | pass | total_inc_vs_current | min_inc | risk_ok |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["experiments"]:
        lines.append(
            f"| {row['experiment']} | `{row['passes']}` | {money(row['total_incremental_vs_current'])} | "
            f"{money(row['min_incremental_vs_current'])} | `{row['risk_ok_all']}` |"
        )
    lines.extend(
        [
            "",
            "## Accepted",
            "",
            "`" + ", ".join(row["experiment"] for row in payload["accepted"]) + "`" if payload["accepted"] else "`none`",
            "",
            "## Outputs",
            "",
            f"- Daily-stop summary: `{payload['outputs']['daily_stop_summary']}`",
            f"- Experiment summary: `{payload['outputs']['experiment_summary']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 136 Protocol101 Account-Scaled Daily Stop"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Tested account-scaled daily loss stops for the always-on Protocol101 sizing policy.
Reason: A fixed daily stop can be too small after account growth; the real bot needs risk controls that scale with equity without amplifying drawdown.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Accepted {len(payload['accepted'])} daily-stop hypotheses. Report {payload['outputs']['report']}.
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
