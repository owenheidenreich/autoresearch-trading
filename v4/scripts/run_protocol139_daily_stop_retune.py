"""Protocol 139: retune the account-aware sizer daily stop."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.run_protocol134_sizing_attribution import build_attribution
from v4.scripts.run_protocol135_starting_cash_sensitivity import baseline_policy, stress_trades
from v4.sim.protocol101_position_sizing import account_aware_sizer_policy, simulate_position_sizing


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_139_protocol101_daily_stop_retune"
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
    baseline = simulate_position_sizing(trades, baseline_policy(10_000.0))
    current = simulate_position_sizing(trades, account_aware_sizer_policy(10_000.0))
    experiments = [evaluate_variant(trades, fixed, fraction) for fixed, fraction in ((-1500.0, 0.005), (-1500.0, 0.01), (-2000.0, 0.005))]
    best = select_best(experiments)
    best_run = simulate_position_sizing(trades, variant_policy(10_000.0, best["fixed_stop"], best["fraction"]))
    attribution = build_attribution(baseline["rows"], best_run["rows"])
    segment_rows = segment_summary(attribution)
    decision = decide(best, segment_rows, current["summary"])
    payload = {
        "protocol": "139_protocol101_daily_stop_retune",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "current_summary": current["summary"],
        "best_summary": best_run["summary"],
        "best_variant": best,
        "experiments": experiments,
        "segment_rows": segment_rows,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "experiment_summary": str(args.out_dir / "experiment_summary.csv"),
            "segment_summary": str(args.out_dir / "segment_summary.csv"),
            "attribution_rows": str(args.out_dir / "attribution_rows.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(experiments).to_csv(args.out_dir / "experiment_summary.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(args.out_dir / "segment_summary.csv", index=False)
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


def variant_policy(starting_cash: float, fixed_stop: float, fraction: float):
    return replace(
        account_aware_sizer_policy(starting_cash),
        name=f"daily_stop_{abs(int(fixed_stop))}_{fraction:.3%}",
        daily_new_entry_stop_loss=float(fixed_stop),
        daily_new_entry_stop_fraction_of_equity=float(fraction),
    )


def evaluate_variant(trades: list[dict[str, Any]], fixed_stop: float, fraction: float) -> dict[str, Any]:
    stress_rows = []
    for stress in (0.0, 0.25, 0.50):
        stressed = stress_trades(trades, stress)
        baseline = simulate_position_sizing(stressed, baseline_policy(10_000.0))
        run = simulate_position_sizing(stressed, variant_policy(10_000.0, fixed_stop, fraction))
        stress_rows.append(
            {
                "stress": stress,
                "incremental": run["summary"]["total_pnl"] - baseline["summary"]["total_pnl"],
                "max_drawdown": run["summary"]["max_drawdown"],
                "worst_day": run["summary"]["worst_day_pnl"],
                "total_pnl": run["summary"]["total_pnl"],
            }
        )
    return {
        "fixed_stop": float(fixed_stop),
        "fraction": float(fraction),
        "total_pnl_unstressed": next(row["total_pnl"] for row in stress_rows if row["stress"] == 0.0),
        "incremental_unstressed": next(row["incremental"] for row in stress_rows if row["stress"] == 0.0),
        "incremental_025": next(row["incremental"] for row in stress_rows if row["stress"] == 0.25),
        "incremental_050": next(row["incremental"] for row in stress_rows if row["stress"] == 0.50),
        "max_drawdown_unstressed": next(row["max_drawdown"] for row in stress_rows if row["stress"] == 0.0),
        "worst_day_unstressed": next(row["worst_day"] for row in stress_rows if row["stress"] == 0.0),
        "passes": all(row["incremental"] > 0 for row in stress_rows) and abs(next(row["max_drawdown"] for row in stress_rows if row["stress"] == 0.0)) <= 3_500.0,
    }


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [row for row in rows if row["passes"]]
    pool = passing or rows
    return max(pool, key=lambda row: (row["passes"], -abs(row["worst_day_unstressed"]), row["incremental_unstressed"]))


def segment_summary(attribution: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(attribution)
    return [
        {
            "segment": key,
            "incremental_pnl": round(float(group["incremental_pnl"].sum()), 2),
            "scaled_trades": int(group["is_scaled"].sum()),
            "skipped_trades": int(group["is_skipped"].sum()),
        }
        for key, group in frame.groupby("segment", sort=True)
    ]


def decide(best: dict[str, Any], segment_rows: list[dict[str, Any]], current_summary: dict[str, Any]) -> str:
    if not best["passes"]:
        return "reject_daily_stop_retune_no_passing_variant"
    if float(best["total_pnl_unstressed"]) <= float(current_summary["total_pnl"]):
        return "reject_daily_stop_retune_no_current_improvement"
    if any(float(row["incremental_pnl"]) <= 0 for row in segment_rows):
        return "reject_daily_stop_retune_segment_negative"
    return "pass_daily_stop_retune_candidate_not_live"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return "Update account_aware_sizer_v1 to use the retuned daily stop and rerun consolidated validation."
    return "Keep the previous account-aware sizer and inspect daily-stop attribution."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 139: Daily Stop Retune",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Best fixed stop: `{payload['best_variant']['fixed_stop']}`",
        f"- Best equity fraction: `{payload['best_variant']['fraction']}`",
        f"- Best unstressed PnL: `{money(payload['best_variant']['total_pnl_unstressed'])}`",
        f"- Current unstressed PnL: `{money(payload['current_summary']['total_pnl'])}`",
        "",
        "## Experiments",
        "",
        "| fixed_stop | fraction | pass | pnl | inc_025 | inc_050 | max_dd | worst_day |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["experiments"]:
        lines.append(
            f"| {row['fixed_stop']} | {row['fraction']:.3%} | `{row['passes']}` | {money(row['total_pnl_unstressed'])} | "
            f"{money(row['incremental_025'])} | {money(row['incremental_050'])} | {money(row['max_drawdown_unstressed'])} | {money(row['worst_day_unstressed'])} |"
        )
    lines.extend(["", "## Segment Incremental", "", "| segment | incremental | scaled | skipped |", "| --- | ---: | ---: | ---: |"])
    for row in payload["segment_rows"]:
        lines.append(f"| {row['segment']} | {money(row['incremental_pnl'])} | {row['scaled_trades']} | {row['skipped_trades']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Experiment summary: `{payload['outputs']['experiment_summary']}`",
            f"- Segment summary: `{payload['outputs']['segment_summary']}`",
            f"- Attribution rows: `{payload['outputs']['attribution_rows']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 139 Protocol101 Daily Stop Retune"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Retuned the always-on account-aware sizer daily stop after base-contract protection.
Reason: The prior fixed stop skipped recovery winners in the external block; the sizer needs account-aware risk control without blocking ordinary one-contract recovery trades too aggressively.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Best stop {payload['best_variant']['fixed_stop']} plus {payload['best_variant']['fraction']:.3%} equity. Report {payload['outputs']['report']}.
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
