"""Export a simple equity/trade sanity packet from fair-contract replay output.

This is an offline reporting step only.  It consumes selected candidates from a
completed fair-contract replay, applies their already-built causal label P&L in
timestamp order, and writes reconstructable equity/trade artifacts.  It does
not train, tune thresholds, contact vendors, call broker endpoints, submit
orders, change defaults, or promote models.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_historical_sanity_report"
)
SCHEMA_VERSION = "Protocol101FairContractHistoricalSanityReportV1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--selected-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument(
        "--stress-per-trade",
        type=float,
        default=0.0,
        help="Optional extra dollar stress subtracted from each selected trade.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_selected(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (str(row.get("decision_time") or ""), str(row.get("contract_id") or "")))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_trade_rows(
    selected: list[dict[str, Any]],
    *,
    starting_equity: float,
    stress_per_trade: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cash = float(starting_equity)
    peak = cash
    max_drawdown = 0.0
    wins = 0
    losses = 0
    gross_wins = 0.0
    gross_losses = 0.0
    premium_deployed = 0.0
    daily_pnl: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        raw_pnl = safe_float(row.get("label_net_pnl"), 0.0)
        stressed_pnl = raw_pnl - float(stress_per_trade)
        entry_ask = safe_float(row.get("entry_ask"), 0.0)
        premium = max(entry_ask, 0.0) * 100.0
        cash_before = cash
        cash += stressed_pnl
        peak = max(peak, cash)
        drawdown = cash - peak
        max_drawdown = min(max_drawdown, drawdown)
        if stressed_pnl > 0:
            wins += 1
            gross_wins += stressed_pnl
        elif stressed_pnl < 0:
            losses += 1
            gross_losses += abs(stressed_pnl)
        premium_deployed += premium
        session = str(row.get("session") or "")
        daily_pnl[session] = daily_pnl.get(session, 0.0) + stressed_pnl
        rows.append(
            {
                "trade_index": idx,
                "session": session,
                "decision_time": row.get("decision_time"),
                "contract_id": row.get("contract_id"),
                "right": row.get("right"),
                "offset": safe_float(row.get("offset")),
                "entry_ask": entry_ask,
                "premium_at_risk": premium,
                "score": safe_float(row.get("score")),
                "raw_label_pnl": raw_pnl,
                "stress_per_trade": float(stress_per_trade),
                "stressed_pnl": stressed_pnl,
                "cash_before": cash_before,
                "cash_after": cash,
                "drawdown": drawdown,
                "feature_contract_version": row.get("feature_contract_version"),
                "feature_hash": row.get("feature_hash"),
                "model_feature_hash": row.get("model_feature_hash"),
                "source_quote_time": row.get("source_quote_time"),
                "source_context_time": row.get("source_context_time"),
            }
        )
    pnl_values = [float(row["stressed_pnl"]) for row in rows]
    top_trade_abs = sorted((abs(value) for value in pnl_values), reverse=True)
    total_pnl = cash - float(starting_equity)
    top_10_abs = sum(top_trade_abs[:10])
    daily_abs = sorted((abs(value) for value in daily_pnl.values()), reverse=True)
    summary = {
        "starting_equity": float(starting_equity),
        "ending_equity": cash,
        "total_pnl": total_pnl,
        "trades": len(rows),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(rows) if rows else 0.0,
        "profit_factor": (
            gross_wins / gross_losses
            if gross_losses > 0
            else (float("inf") if gross_wins > 0 else 0.0)
        ),
        "max_drawdown": max_drawdown,
        "premium_deployed": premium_deployed,
        "return_on_premium": total_pnl / premium_deployed if premium_deployed > 0 else 0.0,
        "sessions_with_trades": len([value for value in daily_pnl.values() if value != 0.0]),
        "top_10_trade_abs_pnl_share": top_10_abs / abs(total_pnl) if total_pnl else 0.0,
        "top_5_day_abs_pnl_share": sum(daily_abs[:5]) / abs(total_pnl) if total_pnl else 0.0,
        "min_cash": min((float(row["cash_after"]) for row in rows), default=float(starting_equity)),
    }
    return rows, summary


def render_report(summary: dict[str, Any], replay_summary: dict[str, Any]) -> str:
    replay = replay_summary.get("summary") or {}
    outputs = summary.get("outputs") or {}
    lines = [
        "# Protocol101 Fair-Contract Historical Sanity Report",
        "",
        "## Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Feature contract: `{summary['feature_contract']}`",
        f"- Same-input replay exact: `{str(summary['same_input_exact']).lower()}`",
        f"- Broker endpoint called: `{str(summary['broker_endpoint_called']).lower()}`",
        f"- Paper-submit allowed: `{str(summary['paper_submit_allowed']).lower()}`",
        f"- Model training executed here: `{str(summary['model_training_executed_here']).lower()}`",
        f"- Threshold tuning executed here: `{str(summary['threshold_tuning_executed_here']).lower()}`",
        "",
        "## Replay Inputs",
        "",
        f"- Decisions: `{replay.get('decision_rows')}`",
        f"- Candidate rows: `{replay.get('candidate_rows')}`",
        f"- Selected entries: `{replay.get('selected_entries')}`",
        f"- Feature transform: `{replay.get('feature_transform')}`",
        f"- Threshold: `{replay.get('threshold')}`",
        f"- Entry filter: `{replay.get('entry_filter')}`",
        "",
        "## Equity Sanity",
        "",
    ]
    metrics = summary["metrics"]
    for key in (
        "trades",
        "total_pnl",
        "ending_equity",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "premium_deployed",
        "return_on_premium",
        "sessions_with_trades",
        "top_10_trade_abs_pnl_share",
        "top_5_day_abs_pnl_share",
    ):
        lines.append(f"- {key}: `{metrics.get(key)}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_equity_html(summary: dict[str, Any], trade_rows: list[dict[str, Any]]) -> str:
    width = 1100
    height = 520
    left = 78
    right = width - 28
    top = 44
    bottom = height - 72
    values = [float(summary["metrics"]["starting_equity"])] + [float(row["cash_after"]) for row in trade_rows]
    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    if math.isclose(low, high):
        low -= 1.0
        high += 1.0

    def x_at(idx: int) -> float:
        if len(values) <= 1:
            return float(left)
        return left + (right - left) * idx / (len(values) - 1)

    def y_at(value: float) -> float:
        return bottom - (bottom - top) * (value - low) / (high - low)

    points = " ".join(f"{x_at(idx):.2f},{y_at(value):.2f}" for idx, value in enumerate(values))
    rows = []
    for key in ("trades", "total_pnl", "ending_equity", "profit_factor", "win_rate", "max_drawdown"):
        rows.append(
            "<tr>"
            f"<th>{html.escape(key)}</th>"
            f"<td>{html.escape(str(summary['metrics'].get(key)))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Protocol101 Fair-Contract Historical Sanity</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; margin: 28px; color: #111827; background: #f9fafb; }}
main {{ max-width: 1180px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin: 0 0 6px; }}
.sub {{ color: #4b5563; margin-bottom: 20px; }}
.panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin: 16px 0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 9px 10px; text-align: left; font-size: 13px; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
<h1>Protocol101 Fair-Contract Historical Sanity</h1>
<div class=\"sub\">Offline audit under <code>{html.escape(str(summary['feature_contract']))}</code>. No broker calls, no training, no threshold tuning.</div>
<div class=\"panel\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Equity curve\">
<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#ffffff\"/>
<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{bottom}\" stroke=\"#9ca3af\"/>
<line x1=\"{left}\" y1=\"{bottom}\" x2=\"{right}\" y2=\"{bottom}\" stroke=\"#9ca3af\"/>
<text x=\"{left - 10}\" y=\"{y_at(high) + 4:.2f}\" text-anchor=\"end\" font-size=\"12\" fill=\"#4b5563\">${high:,.0f}</text>
<text x=\"{left - 10}\" y=\"{y_at(low) + 4:.2f}\" text-anchor=\"end\" font-size=\"12\" fill=\"#4b5563\">${low:,.0f}</text>
<polyline points=\"{points}\" fill=\"none\" stroke=\"#0f766e\" stroke-width=\"2.5\"/>
</svg></div>
<div class=\"panel\"><table><tbody>{''.join(rows)}</tbody></table></div>
</main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_summary = read_json(args.replay_summary)
    selected = read_selected(args.selected_candidates)
    trade_rows, metrics = build_trade_rows(
        selected,
        starting_equity=float(args.starting_equity),
        stress_per_trade=float(args.stress_per_trade),
    )
    replay = replay_summary.get("summary") or {}
    same_input_exact = bool((replay_summary.get("repeat_check") or {}).get("same_input_exact"))
    status = "pass" if same_input_exact and metrics["trades"] > 0 and metrics["min_cash"] > 0 else "fail"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": (
            "historical_sanity_passed_hill_climb_parity_gate_candidate"
            if status == "pass"
            else "historical_sanity_failed_keep_hill_climbing_blocked"
        ),
        "feature_contract": replay.get("feature_contract"),
        "same_input_exact": same_input_exact,
        "broker_endpoint_called": bool(replay.get("broker_endpoint_called")),
        "paper_submit_allowed": bool(replay.get("paper_submit_allowed")),
        "model_training_executed_here": bool(replay.get("model_training_executed_here")),
        "threshold_tuning_executed_here": bool(replay.get("threshold_tuning_executed_here")),
        "paid_data_downloaded": bool(replay.get("paid_data_downloaded")),
        "metrics": metrics,
        "replay_summary_path": str(args.replay_summary),
        "selected_candidates_path": str(args.selected_candidates),
        "outputs": {
            "equity_html": str(args.out_dir / "equity.html"),
            "trades_csv": str(args.out_dir / "trades.csv"),
            "summary_json": str(args.out_dir / "summary.json"),
            "report_md": str(args.out_dir / "report.md"),
        },
    }
    write_csv(args.out_dir / "trades.csv", trade_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_report(summary, replay_summary))
    (args.out_dir / "equity.html").write_text(render_equity_html(summary, trade_rows))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "trades": metrics["trades"],
                "total_pnl": metrics["total_pnl"],
                "ending_equity": metrics["ending_equity"],
                "report": summary["outputs"]["report_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
