"""Pre-live sanity gate for Protocol101 live-style historical validation.

The gate is intentionally conservative and offline-only. It inspects replay and
chart/export artifacts produced by historical validation before any repaired
IBKR paper-submit session is allowed to count as synchronization evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_pre_live_historical_sanity_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol161-summary", type=Path, required=True)
    parser.add_argument("--chart-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-decision-rows", type=int, default=1)
    parser.add_argument("--min-entry-signals", type=int, default=1)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument(
        "--allow-zero-entry-signals-with-reason",
        action="store_true",
        help="Use only for scoped diagnostic dates where zero entries are an expected result, not for Q1-style sanity validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol161 = load_json(args.protocol161_summary)
    chart = load_json(args.chart_summary) if args.chart_summary else {}
    errors: list[str] = []
    warnings: list[str] = []

    replay_summary = protocol161.get("summary", {})
    decision_rows = int_number(replay_summary.get("decision_rows"))
    entry_signals = int_number(replay_summary.get("entry_signals"))
    if decision_rows < int(args.min_decision_rows):
        errors.append(f"decision_rows_below_min:{decision_rows}<{args.min_decision_rows}")
    if entry_signals < int(args.min_entry_signals):
        reason = str(protocol161.get("decision") or "")
        if args.allow_zero_entry_signals_with_reason and reason:
            warnings.append(f"entry_signals_below_min_but_allowed:{entry_signals}<{args.min_entry_signals}:{reason}")
        else:
            errors.append(f"entry_signals_below_min:{entry_signals}<{args.min_entry_signals}")

    chart_metrics = summarize_chart_metrics(chart)
    if args.chart_summary:
        if chart_metrics["trade_count"] is not None and chart_metrics["trade_count"] < int(args.min_trades):
            errors.append(f"trade_count_below_min:{chart_metrics['trade_count']}<{args.min_trades}")
        if chart_metrics["ending_equity"] is not None and not math.isfinite(float(chart_metrics["ending_equity"])):
            errors.append("ending_equity_not_finite")
        if chart_metrics["starting_equity"] is not None and chart_metrics["ending_equity"] is not None:
            if float(chart_metrics["starting_equity"]) <= 0:
                errors.append("starting_equity_not_positive")
        if chart_metrics["trade_count"] is None:
            warnings.append("chart_summary_trade_count_unknown")
        if chart_metrics["ending_equity"] is None:
            warnings.append("chart_summary_ending_equity_unknown")
    else:
        warnings.append("chart_summary_not_provided")

    payload = {
        "protocol": "protocol101_pre_live_historical_sanity_gate",
        "status": "pass" if not errors else "fail",
        "paper_submit_allowed_by_gate": not errors,
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "protocol161_summary": str(args.protocol161_summary),
            "chart_summary": str(args.chart_summary) if args.chart_summary else None,
        },
        "thresholds": {
            "min_decision_rows": int(args.min_decision_rows),
            "min_entry_signals": int(args.min_entry_signals),
            "min_trades": int(args.min_trades),
            "allow_zero_entry_signals_with_reason": bool(args.allow_zero_entry_signals_with_reason),
        },
        "replay_summary": replay_summary,
        "chart_metrics": chart_metrics,
        "next_gate": "prospective_ibkr_live_trace" if not errors else "repair_live_style_historical_contract",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"status": payload["status"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text())


def summarize_chart_metrics(chart: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_count": first_int(chart, ("trade_count", "total_trades", "trades", "closed_trades", "num_trades")),
        "starting_equity": first_float(chart, ("starting_equity", "starting_cash", "start_cash", "initial_equity")),
        "ending_equity": first_float(chart, ("ending_equity", "final_equity", "ending_cash", "end_equity")),
        "max_drawdown": first_float(chart, ("max_drawdown", "max_drawdown_dollars", "max_dd")),
        "win_rate": first_float(chart, ("win_rate", "win_rate_pct")),
    }


def first_int(row: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = nested_value(row, name)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def first_float(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = nested_value(row, name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def nested_value(row: dict[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    for value in row.values():
        if isinstance(value, dict):
            found = nested_value(value, name)
            if found is not None:
                return found
    return None


def int_number(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Pre-Live Historical Sanity Gate",
        "",
        f"- Status: `{payload['status']}`",
        f"- Paper-submit allowed by gate: `{payload['paper_submit_allowed_by_gate']}`",
        f"- Next gate: `{payload['next_gate']}`",
        f"- Errors: `{payload['errors']}`",
        f"- Warnings: `{payload['warnings']}`",
        "",
        "## Replay Summary",
        "",
    ]
    for key, value in payload.get("replay_summary", {}).items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Chart Metrics", ""])
    for key, value in payload.get("chart_metrics", {}).items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
