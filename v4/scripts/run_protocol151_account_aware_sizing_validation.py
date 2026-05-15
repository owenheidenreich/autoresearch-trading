"""Protocol 151: consolidated account-aware multi-contract validation.

This is an offline simulator only. It keeps Protocol 101 entries/exits frozen
and asks whether the current account-aware sizer is actually better than the
one-contract replay after cash, confidence, stress, and concentration checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.sim.protocol101_position_sizing import (
    PositionSizingPolicy,
    account_aware_sizer_policy,
    baseline_one_contract_policy,
    simulate_position_sizing,
)


DEFAULT_TRADES_CSV = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation"
)
STARTING_CASH_VALUES = (10_000.0, 25_000.0, 30_000.0, 50_000.0, 100_000.0)
STRESS_PER_SIDE = (0.0, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", type=Path, default=DEFAULT_TRADES_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(args.trades_csv)
    baseline_policy = with_starting_cash(baseline_one_contract_policy(), 10_000.0)
    candidate_policy = account_aware_sizer_policy(10_000.0)

    validation_rows = build_validation_rows(trades)
    baseline_10 = simulate_position_sizing(trades, baseline_policy)
    candidate_10 = simulate_position_sizing(trades, candidate_policy)
    attribution = build_attribution(baseline_10["rows"], candidate_10["rows"])
    segment_rows = group_attribution(attribution, "segment")
    month_rows = group_attribution(attribution, "month")
    side_rows = group_attribution(attribution, "side")
    concentration = concentration_checks(attribution)
    gate = evaluate_gate(validation_rows, segment_rows, concentration, baseline_10["summary"], candidate_10["summary"])
    decision = "pass_account_aware_sizing_beats_one_contract_research_only" if gate["passed"] else "reject_account_aware_sizing_not_proven"

    payload = {
        "protocol": "151_protocol101_account_aware_sizing_validation",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "protocol101_frozen": True,
        "source_trades_csv": str(args.trades_csv),
        "policy": candidate_policy.__dict__,
        "starting_cash_values": list(STARTING_CASH_VALUES),
        "stress_per_side": list(STRESS_PER_SIDE),
        "baseline_10000": baseline_10["summary"],
        "candidate_10000": candidate_10["summary"],
        "gate": gate,
        "concentration": concentration,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "policy_config": str(args.out_dir / "account_aware_sizer_v1.json"),
            "validation_summary": str(args.out_dir / "validation_summary.csv"),
            "attribution_rows": str(args.out_dir / "attribution_rows.csv"),
            "segment_summary": str(args.out_dir / "segment_summary.csv"),
            "month_summary": str(args.out_dir / "month_summary.csv"),
            "side_summary": str(args.out_dir / "side_summary.csv"),
            "candidate_trade_rows": str(args.out_dir / "candidate_trade_rows.csv"),
        },
        "next_gate": next_gate(decision),
    }
    pd.DataFrame(validation_rows).to_csv(args.out_dir / "validation_summary.csv", index=False)
    pd.DataFrame(attribution).to_csv(args.out_dir / "attribution_rows.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(args.out_dir / "segment_summary.csv", index=False)
    pd.DataFrame(month_rows).to_csv(args.out_dir / "month_summary.csv", index=False)
    pd.DataFrame(side_rows).to_csv(args.out_dir / "side_summary.csv", index=False)
    pd.DataFrame(candidate_10["rows"]).to_csv(args.out_dir / "candidate_trade_rows.csv", index=False)
    (args.out_dir / "account_aware_sizer_v1.json").write_text(json_dumps(candidate_policy.__dict__))
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload, validation_rows, segment_rows, side_rows)
    print(json.dumps({"decision": decision, "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"no trades found in {path}")
    frame = frame.sort_values(["decision_time", "trade_number"]).reset_index(drop=True)
    return [{key: (None if pd.isna(value) else value) for key, value in row.items()} for row in frame.to_dict("records")]


def with_starting_cash(policy: PositionSizingPolicy, starting_cash: float) -> PositionSizingPolicy:
    return PositionSizingPolicy(**{**policy.__dict__, "starting_cash": float(starting_cash)})


def build_validation_rows(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for starting_cash in STARTING_CASH_VALUES:
        for stress in STRESS_PER_SIDE:
            stressed = stress_trades(trades, stress)
            baseline = simulate_position_sizing(stressed, with_starting_cash(baseline_one_contract_policy(), starting_cash))
            candidate = simulate_position_sizing(stressed, account_aware_sizer_policy(starting_cash))
            rows.append(compare_summaries(baseline["summary"], candidate["summary"], stress))
    return rows


def stress_trades(trades: list[dict[str, Any]], stress_per_side: float) -> list[dict[str, Any]]:
    round_trip_cost = 2.0 * float(stress_per_side) * 100.0
    out: list[dict[str, Any]] = []
    for row in trades:
        copy = dict(row)
        copy["pnl"] = float(copy["pnl"]) - round_trip_cost
        out.append(copy)
    return out


def compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any], stress: float) -> dict[str, Any]:
    baseline_romd = return_over_drawdown(baseline)
    candidate_romd = return_over_drawdown(candidate)
    return {
        "starting_cash": float(baseline["starting_cash"]),
        "stress_per_side": float(stress),
        "baseline_total_pnl": float(baseline["total_pnl"]),
        "candidate_total_pnl": float(candidate["total_pnl"]),
        "incremental_pnl": round(float(candidate["total_pnl"]) - float(baseline["total_pnl"]), 2),
        "baseline_max_drawdown": float(baseline["max_drawdown"]),
        "candidate_max_drawdown": float(candidate["max_drawdown"]),
        "baseline_max_drawdown_pct": float(baseline["max_drawdown_pct"]),
        "candidate_max_drawdown_pct": float(candidate["max_drawdown_pct"]),
        "baseline_worst_day_pnl": float(baseline["worst_day_pnl"]),
        "candidate_worst_day_pnl": float(candidate["worst_day_pnl"]),
        "baseline_return_over_drawdown": baseline_romd,
        "candidate_return_over_drawdown": candidate_romd,
        "return_over_drawdown_improved": candidate_romd > baseline_romd,
        "candidate_risk_of_ruin": bool(candidate["risk_of_ruin"]),
        "candidate_max_quantity": int(candidate["max_quantity"]),
        "candidate_taken_trades": int(candidate["taken_trades"]),
        "candidate_skipped_trades": int(candidate["skipped_trades"]),
        "candidate_total_contracts": int(candidate["total_contracts"]),
        "incremental_positive": float(candidate["total_pnl"]) > float(baseline["total_pnl"]),
        "worst_day_not_materially_worse": float(candidate["worst_day_pnl"]) >= float(baseline["worst_day_pnl"]) - 500.0,
    }


def return_over_drawdown(summary: dict[str, Any]) -> float:
    return round(float(summary["total_pnl"]) / max(abs(float(summary["max_drawdown"])), 1.0), 6)


def build_attribution(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_trade = {int(row["trade_number"]): row for row in baseline_rows}
    out: list[dict[str, Any]] = []
    for row in candidate_rows:
        trade_no = int(row["trade_number"])
        base = base_by_trade[trade_no]
        quantity = int(row["quantity"])
        baseline_quantity = int(base["quantity"])
        candidate_pnl = float(row["realized_pnl"])
        baseline_pnl = float(base["realized_pnl"])
        out.append(
            {
                "trade_number": trade_no,
                "session": row["session"],
                "month": str(pd.Timestamp(row["session"]).to_period("M")),
                "segment": row.get("segment"),
                "stage": row.get("stage"),
                "decision_time": row["decision_time"],
                "exit_time": row["exit_time"],
                "side": row.get("side"),
                "contract_id": row["contract_id"],
                "score_margin": row.get("score_margin"),
                "one_contract_premium": row.get("one_contract_premium"),
                "baseline_quantity": baseline_quantity,
                "candidate_quantity": quantity,
                "quantity_delta": quantity - baseline_quantity,
                "baseline_pnl": baseline_pnl,
                "candidate_pnl": candidate_pnl,
                "incremental_pnl": round(candidate_pnl - baseline_pnl, 6),
                "candidate_cash_before": row.get("cash_before"),
                "candidate_cash_after": row.get("cash_after"),
                "candidate_drawdown_pct_before": row.get("drawdown_pct_before"),
                "candidate_daily_pnl_before": row.get("daily_pnl_before"),
                "premium_exposure": row.get("premium_exposure"),
                "skip_reason": row.get("skip_reason", ""),
                "is_scaled": quantity > baseline_quantity,
                "is_skipped": quantity == 0,
            }
        )
    return out


def group_attribution(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    for key, group in frame.groupby(column, dropna=False, sort=True):
        out.append(
            {
                column: key,
                "rows": int(len(group)),
                "incremental_pnl": round(float(group["incremental_pnl"].sum()), 2),
                "scaled_trades": int(group["is_scaled"].sum()),
                "skipped_trades": int(group["is_skipped"].sum()),
                "avg_score_margin_scaled": mean_or_none(group.loc[group["is_scaled"], "score_margin"]),
            }
        )
    return out


def concentration_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    incremental = pd.to_numeric(frame["incremental_pnl"], errors="coerce").fillna(0.0)
    positive = incremental.clip(lower=0)
    positive_total = float(positive.sum())
    top_trade = float(incremental.max()) if len(incremental) else 0.0
    by_day = group_positive_share(frame, "session")
    by_month = group_positive_share(frame, "month")
    scaled = frame[frame["is_scaled"]]
    return {
        "incremental_pnl": round(float(incremental.sum()), 2),
        "positive_incremental_pnl": round(positive_total, 2),
        "top_trade_incremental_pnl": round(top_trade, 2),
        "top_trade_share_of_positive": share(top_trade, positive_total),
        "top_day": by_day[0] if by_day else {},
        "top_day_share_of_positive": by_day[0]["share_of_positive"] if by_day else 0.0,
        "top_month": by_month[0] if by_month else {},
        "top_month_share_of_positive": by_month[0]["share_of_positive"] if by_month else 0.0,
        "scaled_trades": int(len(scaled)),
        "scaled_positive_fraction": round(float((scaled["incremental_pnl"] > 0).mean()), 6) if len(scaled) else 0.0,
        "skipped_trades": int(frame["is_skipped"].sum()),
        "quantity_counts": {str(int(k)): int(v) for k, v in frame["candidate_quantity"].value_counts().sort_index().items()},
    }


def group_positive_share(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    positive_total = float(pd.to_numeric(frame["incremental_pnl"], errors="coerce").clip(lower=0).sum())
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(column, dropna=False, sort=True):
        incremental = float(group["incremental_pnl"].sum())
        positive = float(pd.to_numeric(group["incremental_pnl"], errors="coerce").clip(lower=0).sum())
        rows.append({column: key, "incremental_pnl": round(incremental, 2), "share_of_positive": share(positive, positive_total)})
    return sorted(rows, key=lambda row: row["share_of_positive"], reverse=True)


def evaluate_gate(
    validation_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    concentration: dict[str, Any],
    baseline_10: dict[str, Any],
    candidate_10: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if float(candidate_10["total_pnl"]) <= float(baseline_10["total_pnl"]):
        reasons.append("candidate_does_not_beat_10000_one_contract_baseline")
    if int(candidate_10["max_quantity"]) <= 1:
        reasons.append("candidate_never_scales_beyond_one_contract")
    if any(not row["incremental_positive"] for row in validation_rows):
        reasons.append("not_positive_incremental_on_every_cash_stress_row")
    if any(bool(row["candidate_risk_of_ruin"]) for row in validation_rows):
        reasons.append("risk_of_ruin")
    if any(not row["return_over_drawdown_improved"] for row in validation_rows):
        reasons.append("return_over_drawdown_not_improved_everywhere")
    if any(not row["worst_day_not_materially_worse"] for row in validation_rows):
        reasons.append("worst_day_materially_worse")
    if any(float(row["incremental_pnl"]) <= 0 for row in segment_rows):
        reasons.append("negative_incremental_segment")
    if float(concentration["top_day_share_of_positive"]) > 0.35:
        reasons.append("top_day_too_concentrated")
    if float(concentration["top_month_share_of_positive"]) > 0.55:
        reasons.append("top_month_too_concentrated")
    if float(concentration["scaled_positive_fraction"]) < 0.55:
        reasons.append("scaled_trades_not_reliable")
    return {
        "passed": not reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "reasons": reasons,
        "acceptance": {
            "beats_10000_baseline": float(candidate_10["total_pnl"]) > float(baseline_10["total_pnl"]),
            "incremental_positive_every_cash_stress": all(row["incremental_positive"] for row in validation_rows),
            "positive_every_segment": all(float(row["incremental_pnl"]) > 0 for row in segment_rows),
            "return_over_drawdown_improved_everywhere": all(row["return_over_drawdown_improved"] for row in validation_rows),
            "worst_day_not_materially_worse_everywhere": all(row["worst_day_not_materially_worse"] for row in validation_rows),
            "top_day_share_le_35pct": float(concentration["top_day_share_of_positive"]) <= 0.35,
            "top_month_share_le_55pct": float(concentration["top_month_share_of_positive"]) <= 0.55,
            "scaled_positive_fraction_ge_55pct": float(concentration["scaled_positive_fraction"]) >= 0.55,
        },
    }


def next_gate(decision: str) -> str:
    if decision.startswith("pass_"):
        return (
            "Keep account-aware sizing as an offline research candidate. Use one-contract paper first, then replay live "
            "paper logs through this sizer before enabling multi-contract paper order quantities."
        )
    return "Reject multi-contract sizing for now and diagnose the failed gate before adding new sizing knobs."


def write_report(
    path: Path,
    payload: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
) -> None:
    baseline = payload["baseline_10000"]
    candidate = payload["candidate_10000"]
    c = payload["concentration"]
    lines = [
        "# Protocol 151: Account-Aware Multi-Contract Validation",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 entries and exits remain frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Gate: `{payload['gate']['reason']}`",
        f"- Baseline one-contract PnL from $10,000: `{money(baseline['total_pnl'])}`",
        f"- Candidate account-aware PnL from $10,000: `{money(candidate['total_pnl'])}`",
        f"- Incremental PnL: `{money(float(candidate['total_pnl']) - float(baseline['total_pnl']))}`",
        f"- Candidate max quantity: `{candidate['max_quantity']}`",
        f"- Candidate skipped trades: `{candidate['skipped_trades']}`",
        f"- Quantity counts: `{c['quantity_counts']}`",
        f"- Top day share of positive incremental PnL: `{c['top_day_share_of_positive']:.1%}`",
        f"- Top month share of positive incremental PnL: `{c['top_month_share_of_positive']:.1%}`",
        "",
        "## Policy Logic",
        "",
        "Extra contracts require cash/account growth, low drawdown, recent positive PnL, nonnegative same-day PnL, profit cushion, premium exposure limits, and score margin above the frozen Protocol101 threshold. The base one-contract trade is protected from the scaling cap when cash can afford it.",
        "",
        "## Cash And Stress",
        "",
        "| start | stress | baseline_pnl | candidate_pnl | incremental | baseline_dd | candidate_dd | romd_improved | worst_day_ok |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in validation_rows:
        lines.append(
            "| "
            f"{money(row['starting_cash'])} | ${float(row['stress_per_side']):.2f} | "
            f"{money(row['baseline_total_pnl'])} | {money(row['candidate_total_pnl'])} | {money(row['incremental_pnl'])} | "
            f"{money(row['baseline_max_drawdown'])} | {money(row['candidate_max_drawdown'])} | "
            f"`{row['return_over_drawdown_improved']}` | `{row['worst_day_not_materially_worse']}` |"
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
    lines.extend(
        [
            "",
            "## Side Incremental",
            "",
            "| side | incremental | scaled | skipped |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in side_rows:
        lines.append(f"| {row['side']} | {money(row['incremental_pnl'])} | {row['scaled_trades']} | {row['skipped_trades']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Policy config: `{payload['outputs']['policy_config']}`",
            f"- Validation summary: `{payload['outputs']['validation_summary']}`",
            f"- Attribution rows: `{payload['outputs']['attribution_rows']}`",
            f"- Candidate trade rows: `{payload['outputs']['candidate_trade_rows']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else round(float(values.mean()), 6)


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
