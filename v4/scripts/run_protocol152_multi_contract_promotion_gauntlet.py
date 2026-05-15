"""Protocol 152: promotion gauntlet for the multi-contract challenger.

This keeps Protocol 101 entries/exits frozen and tests one challenger:
``account_aware_sizer_v1``. It does not search knobs. If the challenger fails a
policy-fixable gate, later protocols can test the next hypothesis. Incomplete
high-resolution coverage is reported as a promotion blocker, not as a sizing
model failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.run_protocol151_account_aware_sizing_validation import (
    build_attribution,
    build_validation_rows,
    concentration_checks,
    evaluate_gate,
    group_attribution,
    load_trades,
)
from v4.sim.protocol101_position_sizing import (
    account_aware_sizer_policy,
    baseline_one_contract_policy,
    simulate_position_sizing,
)


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_TIMING_ROWS = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/timing_delay_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_152_protocol101_multi_contract_promotion_gauntlet"
)
CRITICAL_DELAYS = (1, 5)
REPORT_DELAYS = (0, 1, 5, 15, 30, 60)
MIN_CRITICAL_COVERAGE = 0.95
MAX_FAILED_HYPOTHESES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", type=Path, default=DEFAULT_TRADES_CSV)
    parser.add_argument("--timing-rows", type=Path, default=DEFAULT_TIMING_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trades = load_trades(args.trades_csv)
    baseline = simulate_position_sizing(trades, baseline_one_contract_policy())
    challenger = simulate_position_sizing(trades, account_aware_sizer_policy(10_000.0))
    validation_rows = build_validation_rows(trades)
    attribution = build_attribution(baseline["rows"], challenger["rows"])
    segment_rows = group_attribution(attribution, "segment")
    side_rows = group_attribution(attribution, "side")
    concentration = concentration_checks(attribution)
    base_gate = evaluate_gate(
        validation_rows,
        segment_rows,
        concentration,
        baseline["summary"],
        challenger["summary"],
    )
    timing_rows = build_timing_rows(
        trades=trades,
        baseline_rows=baseline["rows"],
        challenger_rows=challenger["rows"],
        timing_path=args.timing_rows,
    )
    timing_summary = summarize_timing(timing_rows)
    timing_gate = evaluate_timing_gate(timing_summary)
    decision = decide(base_gate, timing_gate)
    failed_hypotheses = 0 if decision.startswith("pass_") or decision.startswith("blocked_") else 1

    payload = {
        "protocol": "152_protocol101_multi_contract_promotion_gauntlet",
        "decision": decision,
        "failed_hypotheses": failed_hypotheses,
        "max_failed_hypotheses": MAX_FAILED_HYPOTHESES,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "challenger": "account_aware_sizer_v1",
        "baseline": "one_contract_baseline",
        "source_trades_csv": str(args.trades_csv),
        "source_timing_rows": str(args.timing_rows),
        "base_gate": base_gate,
        "timing_gate": timing_gate,
        "baseline_summary": baseline["summary"],
        "challenger_summary": challenger["summary"],
        "concentration": concentration,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "timing_trade_rows": str(args.out_dir / "timing_trade_rows.csv"),
            "timing_summary": str(args.out_dir / "timing_summary.csv"),
            "validation_summary": str(args.out_dir / "validation_summary.csv"),
            "attribution_rows": str(args.out_dir / "attribution_rows.csv"),
            "segment_summary": str(args.out_dir / "segment_summary.csv"),
            "side_summary": str(args.out_dir / "side_summary.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(timing_rows).to_csv(args.out_dir / "timing_trade_rows.csv", index=False)
    pd.DataFrame(timing_summary).to_csv(args.out_dir / "timing_summary.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(args.out_dir / "validation_summary.csv", index=False)
    pd.DataFrame(attribution).to_csv(args.out_dir / "attribution_rows.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(args.out_dir / "segment_summary.csv", index=False)
    pd.DataFrame(side_rows).to_csv(args.out_dir / "side_summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload, timing_summary, segment_rows, side_rows)
    print(json.dumps({"decision": decision, "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def build_timing_rows(
    *,
    trades: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
    timing_path: Path,
) -> list[dict[str, Any]]:
    raw_by_trade = {int(row["trade_number"]): row for row in trades}
    quantity_by_uid = {}
    baseline_qty_by_uid = {}
    for row in challenger_rows:
        raw = raw_by_trade[int(row["trade_number"])]
        quantity_by_uid[str(raw["candidate_uid"])] = int(row["quantity"])
    for row in baseline_rows:
        raw = raw_by_trade[int(row["trade_number"])]
        baseline_qty_by_uid[str(raw["candidate_uid"])] = int(row["quantity"])

    timing = pd.read_csv(timing_path)
    timing = timing[timing["candidate_uid"].astype(str).isin(quantity_by_uid)].copy()
    timing = timing[timing["delay_seconds"].isin(REPORT_DELAYS)].copy()
    out: list[dict[str, Any]] = []
    for row in timing.to_dict("records"):
        uid = str(row["candidate_uid"])
        baseline_qty = int(baseline_qty_by_uid.get(uid, 1))
        challenger_qty = int(quantity_by_uid.get(uid, 0))
        delayed_pnl = number(row.get("delayed_pnl"))
        original_pnl = number(row.get("original_pnl"))
        status = str(row.get("status", "missing"))
        ok = status == "ok" and delayed_pnl is not None and original_pnl is not None
        out.append(
            {
                **row,
                "baseline_quantity": baseline_qty,
                "challenger_quantity": challenger_qty,
                "baseline_original_pnl": original_pnl * baseline_qty if ok else None,
                "baseline_delayed_pnl": delayed_pnl * baseline_qty if ok else None,
                "challenger_original_pnl": original_pnl * challenger_qty if ok else None,
                "challenger_delayed_pnl": delayed_pnl * challenger_qty if ok else None,
                "incremental_original_pnl": original_pnl * (challenger_qty - baseline_qty) if ok else None,
                "incremental_delayed_pnl": delayed_pnl * (challenger_qty - baseline_qty) if ok else None,
                "is_scaled": challenger_qty > baseline_qty,
                "is_skipped": challenger_qty == 0,
            }
        )
    return out


def summarize_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    for (split, delay), group in frame.groupby(["split", "delay_seconds"], dropna=False, sort=True):
        ok = group[group["status"].eq("ok")].copy()
        out.append(
            {
                "split": split,
                "delay_seconds": int(delay),
                "rows": int(len(group)),
                "coverage": round(float(len(ok) / max(len(group), 1)), 6),
                "baseline_original_pnl": sum_numeric(ok, "baseline_original_pnl"),
                "baseline_delayed_pnl": sum_numeric(ok, "baseline_delayed_pnl"),
                "challenger_original_pnl": sum_numeric(ok, "challenger_original_pnl"),
                "challenger_delayed_pnl": sum_numeric(ok, "challenger_delayed_pnl"),
                "incremental_original_pnl": sum_numeric(ok, "incremental_original_pnl"),
                "incremental_delayed_pnl": sum_numeric(ok, "incremental_delayed_pnl"),
                "scaled_trades": int(ok["is_scaled"].sum()) if not ok.empty else 0,
                "skipped_trades": int(ok["is_skipped"].sum()) if not ok.empty else 0,
            }
        )
    return out


def evaluate_timing_gate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    critical = [row for row in summary_rows if int(row["delay_seconds"]) in CRITICAL_DELAYS]
    if not critical:
        reasons.append("missing_critical_delay_rows")
    if any(float(row["coverage"]) < MIN_CRITICAL_COVERAGE for row in critical):
        reasons.append("incomplete_highres_timing_coverage")
    if any(float(row["challenger_delayed_pnl"]) <= 0 for row in critical):
        reasons.append("challenger_negative_under_critical_delay")
    if any(float(row["incremental_delayed_pnl"]) <= 0 for row in critical):
        reasons.append("challenger_not_incremental_under_critical_delay")
    policy_failures = [reason for reason in reasons if reason != "incomplete_highres_timing_coverage"]
    return {
        "passed_for_available_coverage": not policy_failures,
        "promotion_coverage_passed": not any(reason == "incomplete_highres_timing_coverage" for reason in reasons),
        "promotion_passed": not reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "reasons": reasons,
    }


def decide(base_gate: dict[str, Any], timing_gate: dict[str, Any]) -> str:
    if not base_gate["passed"]:
        return "reject_multi_contract_challenger_base_gate_failed"
    if not timing_gate["passed_for_available_coverage"]:
        return "reject_multi_contract_challenger_timing_fragile"
    if not timing_gate["promotion_coverage_passed"]:
        return "blocked_multi_contract_promotion_incomplete_highres_coverage"
    return "pass_multi_contract_challenger_promotable_research_only"


def next_gate(decision: str) -> str:
    if decision == "pass_multi_contract_challenger_promotable_research_only":
        return "Replay live paper logs through this sizer before enabling multi-contract paper quantities."
    if decision.startswith("blocked_"):
        return "Do not replace the one-contract operational baseline yet; collect live-shadow/paper evidence or more high-resolution coverage for the same frozen challenger."
    return "Treat this as one failed hypothesis and test the next sizing hypothesis, stopping after five failures."


def write_report(
    path: Path,
    payload: dict[str, Any],
    timing_summary: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
) -> None:
    baseline = payload["baseline_summary"]
    challenger = payload["challenger_summary"]
    lines = [
        "# Protocol 152: Multi-Contract Promotion Gauntlet",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 entries and exits remain frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Failed hypotheses: `{payload['failed_hypotheses']}` of `{payload['max_failed_hypotheses']}`",
        f"- Baseline PnL: `{money(baseline['total_pnl'])}`",
        f"- Challenger PnL: `{money(challenger['total_pnl'])}`",
        f"- Incremental PnL: `{money(float(challenger['total_pnl']) - float(baseline['total_pnl']))}`",
        f"- Base gate: `{payload['base_gate']['reason']}`",
        f"- Timing gate: `{payload['timing_gate']['reason']}`",
        "",
        "## Timing Delay Gate",
        "",
        "| split | delay_s | coverage | baseline_delayed | challenger_delayed | incremental_delayed | scaled | skipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in timing_summary:
        lines.append(
            "| "
            f"{row['split']} | {row['delay_seconds']} | {pct(row['coverage'])} | "
            f"{money(row['baseline_delayed_pnl'])} | {money(row['challenger_delayed_pnl'])} | "
            f"{money(row['incremental_delayed_pnl'])} | {row['scaled_trades']} | {row['skipped_trades']} |"
        )
    lines.extend(
        [
            "",
            "## Segment Incremental",
            "",
            "| segment | incremental | scaled | skipped |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in segment_rows:
        lines.append(f"| {row['segment']} | {money(row['incremental_pnl'])} | {row['scaled_trades']} | {row['skipped_trades']} |")
    lines.extend(["", "## Side Incremental", "", "| side | incremental | scaled | skipped |", "| --- | ---: | ---: | ---: |"])
    for row in side_rows:
        lines.append(f"| {row['side']} | {money(row['incremental_pnl'])} | {row['scaled_trades']} | {row['skipped_trades']} |")
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum()), 6)


def number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


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
