"""Protocol 134: attribution for the surviving Protocol 133 sizing policy."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from v4.sim.protocol101_position_sizing import (
    baseline_one_contract_policy,
    high_conviction_profit_cushion_policy,
    simulate_position_sizing,
)


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_134_protocol101_sizing_attribution"
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
    baseline = simulate_position_sizing(trades, baseline_one_contract_policy())
    policy = replace(high_conviction_profit_cushion_policy(), name="lower_two_contract_threshold", min_score_margin_for_two=2.0)
    candidate = simulate_position_sizing(trades, policy)
    attribution = build_attribution(baseline["rows"], candidate["rows"])
    concentration = concentration_checks(attribution)
    by_group = build_group_summaries(attribution)
    decision = decide(concentration, candidate["summary"], baseline["summary"])
    payload = {
        "protocol": "134_protocol101_sizing_attribution",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "policy": policy.__dict__,
        "baseline_summary": baseline["summary"],
        "candidate_summary": candidate["summary"],
        "concentration": concentration,
        "group_summary_rows": len(by_group),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "attribution_rows": str(args.out_dir / "attribution_rows.csv"),
            "group_summary": str(args.out_dir / "group_summary.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(attribution).to_csv(args.out_dir / "attribution_rows.csv", index=False)
    pd.DataFrame(by_group).to_csv(args.out_dir / "group_summary.csv", index=False)
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


def build_attribution(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_trade = {int(row["trade_number"]): row for row in baseline_rows}
    out: list[dict[str, Any]] = []
    for row in candidate_rows:
        trade_no = int(row["trade_number"])
        base = base_by_trade[trade_no]
        baseline_pnl = float(base["realized_pnl"])
        candidate_pnl = float(row["realized_pnl"])
        quantity = int(row["quantity"])
        month = str(pd.Timestamp(row["session"]).to_period("M"))
        out.append(
            {
                "trade_number": trade_no,
                "session": row["session"],
                "month": month,
                "segment": row.get("segment"),
                "stage": row.get("stage"),
                "decision_time": row["decision_time"],
                "side": row.get("side"),
                "contract_id": row["contract_id"],
                "score_margin": row.get("score_margin"),
                "one_contract_premium": row.get("one_contract_premium"),
                "baseline_quantity": int(base["quantity"]),
                "candidate_quantity": quantity,
                "quantity_delta": quantity - int(base["quantity"]),
                "baseline_pnl": baseline_pnl,
                "candidate_pnl": candidate_pnl,
                "incremental_pnl": candidate_pnl - baseline_pnl,
                "skip_reason": row.get("skip_reason", ""),
                "is_scaled": quantity > int(base["quantity"]),
                "is_skipped": quantity == 0,
            }
        )
    return out


def concentration_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    incremental = pd.to_numeric(frame["incremental_pnl"], errors="coerce").fillna(0.0)
    total = float(incremental.sum())
    positive_total = float(incremental.clip(lower=0).sum())
    top_trade = float(incremental.max()) if len(incremental) else 0.0
    top_day = group_incremental(frame, "session")[0] if not frame.empty else {"incremental_pnl": 0.0, "session": None}
    top_month = group_incremental(frame, "month")[0] if not frame.empty else {"incremental_pnl": 0.0, "month": None}
    scaled = frame[frame["is_scaled"]]
    skipped = frame[frame["is_skipped"]]
    return {
        "incremental_pnl": round(total, 2),
        "positive_incremental_pnl": round(positive_total, 2),
        "top_trade_incremental_pnl": round(top_trade, 2),
        "top_trade_share_of_positive": share(top_trade, positive_total),
        "top_day": top_day,
        "top_day_share_of_positive": share(float(top_day["incremental_pnl"]), positive_total),
        "top_month": top_month,
        "top_month_share_of_positive": share(float(top_month["incremental_pnl"]), positive_total),
        "scaled_trades": int(len(scaled)),
        "skipped_trades": int(len(skipped)),
        "scaled_positive_fraction": float((scaled["incremental_pnl"] > 0).mean()) if len(scaled) else 0.0,
        "skipped_pnl_given_up": round(float(skipped["baseline_pnl"].sum()), 2) if len(skipped) else 0.0,
    }


def group_incremental(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    grouped = []
    for key, group in frame.groupby(column, dropna=False, sort=True):
        grouped.append({column: key, "incremental_pnl": round(float(group["incremental_pnl"].sum()), 2)})
    return sorted(grouped, key=lambda row: row["incremental_pnl"], reverse=True)


def build_group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    out = []
    for column in ("segment", "month", "side", "is_scaled", "is_skipped"):
        for key, group in frame.groupby(column, dropna=False, sort=True):
            out.append(
                {
                    "dimension": column,
                    "value": str(key),
                    "rows": int(len(group)),
                    "incremental_pnl": round(float(group["incremental_pnl"].sum()), 2),
                    "scaled_trades": int(group["is_scaled"].sum()),
                    "skipped_trades": int(group["is_skipped"].sum()),
                    "positive_incremental_fraction": float((group["incremental_pnl"] > 0).mean()),
                }
            )
    return out


def decide(concentration: dict[str, Any], candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    if float(candidate["total_pnl"]) <= float(baseline["total_pnl"]):
        return "reject_sizing_no_incremental_pnl"
    if float(concentration["top_day_share_of_positive"]) > 0.35:
        return "fragile_sizing_top_day_concentrated"
    if float(concentration["top_month_share_of_positive"]) > 0.55:
        return "fragile_sizing_top_month_concentrated"
    if float(concentration["scaled_positive_fraction"]) < 0.55:
        return "fragile_sizing_scaled_trades_not_reliable"
    return "pass_sizing_attribution_research_candidate_not_live"


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Keep the sizing policy as an offline research candidate. Next test should add split-by-split equity curves "
            "and paper-account visualization, not live multi-contract trading."
        )
    return "Do not continue tuning sizing knobs until the concentration failure is explained."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    c = payload["concentration"]
    lines = [
        "# Protocol 134: Sizing Attribution",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Policy: `{payload['policy']['name']}`",
        f"- Incremental PnL vs one contract: `{money(c['incremental_pnl'])}`",
        f"- Scaled trades: `{c['scaled_trades']}`",
        f"- Skipped trades: `{c['skipped_trades']}`",
        f"- Scaled positive fraction: `{c['scaled_positive_fraction']:.1%}`",
        f"- Top day share of positive incremental PnL: `{c['top_day_share_of_positive']:.1%}`",
        f"- Top month share of positive incremental PnL: `{c['top_month_share_of_positive']:.1%}`",
        "",
        "## Concentration",
        "",
        f"- Top day: `{c['top_day']}`",
        f"- Top month: `{c['top_month']}`",
        f"- Skipped baseline PnL given up: `{money(c['skipped_pnl_given_up'])}`",
        "",
        "## Outputs",
        "",
        f"- Attribution rows: `{payload['outputs']['attribution_rows']}`",
        f"- Group summary: `{payload['outputs']['group_summary']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 134 Protocol101 Sizing Attribution"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Attributed the surviving Protocol133 sizing policy trade-by-trade against the one-contract baseline.
Reason: Multi-contract sizing must prove it is not just a few large scaled winners masking concentrated risk.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Incremental PnL {payload['concentration']['incremental_pnl']}. Report {payload['outputs']['report']}.
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


def share(value: float, total: float) -> float:
    return 0.0 if total <= 0 else round(float(value) / float(total), 6)


def money(value: Any) -> str:
    number = float(value)
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
