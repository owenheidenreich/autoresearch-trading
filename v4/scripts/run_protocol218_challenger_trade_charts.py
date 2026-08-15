"""DIAGNOSTIC_CHALLENGER_FULL_ACTION_HISTORY_TRADE_CHARTS_V1.

Historically Protocol218. This exports visual inspection charts for
CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.

The charts are research-only. They do not change PAPER_DEFAULT_PROTOCOL101 and
do not download data or call broker endpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.export_protocol101_trade_charts import (
    DEFAULT_NORMALIZED_DIRS,
    DEFAULT_SPX_DIR,
    EQUITY_BODY,
    TRADE_BODY,
    add_equity_fields,
    attach_spx_prices,
    backfill_option_quote_accounting,
    block_summary_rows,
    build_paper_account_summary,
    build_paper_account_trades,
    cleanup_legacy_duplicate_outputs,
    equity_after_stress,
    html_shell,
    iso_utc,
    load_spx_bars,
    none_or_float,
    profit_concentration,
    write_equity_html,
    write_trades_csv,
)


ROLE_LABEL = "DIAGNOSTIC_CHALLENGER_FULL_ACTION_HISTORY_TRADE_CHARTS_V1"
HISTORICAL_ID = "Protocol218"
DEFAULT_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_215_full_action_surface_edge_history_5seed_confirmation/model_trades_5seed.csv"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_218_challenger_full_action_history_trade_charts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--normalized-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument("--paper-seed", type=int, default=1)
    parser.add_argument("--stress-per-side", type=float, default=0.10)
    parser.add_argument("--skip-option-quote-backfill", action="store_true")
    parser.add_argument("--role-label", default=ROLE_LABEL)
    parser.add_argument("--historical-id", default=HISTORICAL_ID)
    parser.add_argument("--candidate-label", default="CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1")
    parser.add_argument("--chart-title", default="Challenger Full-Action History")
    parser.add_argument(
        "--subtitle",
        default=(
            "Research-only historical replay for CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1: "
            "one seed, one affordable SPXW 0DTE contract at a time, ask-entry/bid-exit."
        ),
    )
    parser.add_argument("--source-protocol", default="215_full_action_surface_edge_history_5seed_confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_duplicate_outputs(args.out_dir)
    normalized_dirs = tuple(args.normalized_dir or DEFAULT_NORMALIZED_DIRS)
    research_trades = load_challenger_trades(args.trades, source_protocol=str(args.source_protocol))
    spx = load_spx_bars_for_trade_sessions(
        args.spx_dir,
        sessions=sorted({str(row["session"]) for row in research_trades if row.get("session")}),
    )
    research_trades = attach_spx_prices(research_trades, spx)
    research_trades = add_equity_fields(research_trades)
    paper_trades, skipped_trades = build_paper_account_trades(
        research_trades,
        starting_equity=float(args.starting_equity),
        paper_seed=int(args.paper_seed),
    )
    if not paper_trades:
        raise SystemExit(f"no affordable paper trades found for seed {args.paper_seed}")
    # The full five-seed research trade set is large; quote-path backfill is only
    # needed for the displayed one-account replay, not for every diagnostic row.
    if args.skip_option_quote_backfill:
        for row in paper_trades:
            row["quote_backfill_status"] = "skipped_uses_source_trade_file_accounting"
    else:
        paper_trades = backfill_option_quote_accounting(paper_trades, normalized_dirs)
    paper_summary = build_paper_account_summary(
        paper_trades,
        float(args.starting_equity),
        skipped_trades=skipped_trades,
    )
    write_trades_csv(args.out_dir / "trades.csv", paper_trades)
    write_trades_csv(args.out_dir / "skipped_trades.csv", skipped_trades)
    write_trades_csv(args.out_dir / "research_all_seed_trades.csv", research_trades)
    (args.out_dir / "paper_account_summary.json").write_text(json.dumps(paper_summary, indent=2, sort_keys=True) + "\n")
    write_challenger_trades_html(
        args.out_dir / "trades.html",
        trades=paper_trades,
        spx=spx,
        starting_equity=float(args.starting_equity),
        skipped_trades=skipped_trades,
        chart_title=str(args.chart_title),
    )
    write_equity_html(
        args.out_dir / "equity.html",
        trades=paper_trades,
        starting_equity=float(args.starting_equity),
        skipped_trades=skipped_trades,
        stress_per_side=float(args.stress_per_side),
        chart_title=f"{args.chart_title} Equity",
        subtitle=str(args.subtitle),
    )
    payload = {
        "role_label": str(args.role_label),
        "historical_protocol": str(args.historical_id),
        "what_is_this": "diagnostic / visual inspection artifact",
        "changes_paper_default": False,
        "candidate_label": str(args.candidate_label),
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "paper_seed": int(args.paper_seed),
        "starting_equity": float(args.starting_equity),
        "paper_trades": len(paper_trades),
        "skipped_trades": len(skipped_trades),
        "research_trades": len(research_trades),
        "sessions": len({row["session"] for row in paper_trades}),
        "first_trade": min(row["decision_time"] for row in paper_trades),
        "last_trade": max(row["decision_time"] for row in paper_trades),
        "source_of_truth": "equity.html",
        "outputs": {
            "trades_html": str(args.out_dir / "trades.html"),
            "equity_html": str(args.out_dir / "equity.html"),
            "trades_csv": str(args.out_dir / "trades.csv"),
            "skipped_trades_csv": str(args.out_dir / "skipped_trades.csv"),
            "research_all_seed_trades_csv": str(args.out_dir / "research_all_seed_trades.csv"),
            "paper_account_summary": str(args.out_dir / "paper_account_summary.json"),
            "report": str(args.out_dir / "report.md"),
        },
        "decision": "visual_inspection_artifacts_ready_paper_default_unchanged",
        "next_experiment": "Inspect biggest winners, biggest losers, worst days, and churn chains before any replacement decision.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(
        args.out_dir / "report.md",
        payload=payload,
        trades=paper_trades,
        research_trades=research_trades,
        skipped_trades=skipped_trades,
        paper_summary=paper_summary,
        stress_per_side=float(args.stress_per_side),
    )
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_challenger_trades(path: Path, *, source_protocol: str = "215_full_action_surface_edge_history_5seed_confirmation") -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if {"split", "stressed_pnl", "synthetic_exit_time"}.issubset(set(frame.columns)):
        return load_fair_contract_strict_replay_trades(frame, source_protocol=source_protocol)
    # March is a report slice inside q1_2026; keep q1_2026 only to avoid duplicated equity.
    if "reported_split" in frame.columns:
        frame = frame[~frame["reported_split"].astype(str).eq("march_2026")].copy()
    rows = []
    for row in frame.to_dict("records"):
        right = str(row.get("right", "")).upper()
        pnl = float(row.get("pnl", row.get("raw_candidate_pnl", 0.0)) or 0.0)
        item = {
            "candidate_uid": str(row.get("candidate_uid", "")),
            "trade_uid": str(row.get("trade_uid", "")),
            "contract_id": str(row.get("contract_id", "")),
            "seed": int(row.get("seed", 0) or 0),
            "entry_seed": int(row.get("seed", 0) or 0),
            "session": str(row.get("session", "")),
            "decision_time": iso_utc(row.get("decision_time")),
            "exit_time": iso_utc(row.get("exit_time")),
            "right": right,
            "side": "CALL" if right == "C" else "PUT" if right == "P" else right,
            "offset": float(row.get("offset", 0.0) or 0.0),
            "score": float(row.get("score", 0.0) or 0.0),
            "threshold": float(row.get("threshold", 0.0) or 0.0),
            "pnl": pnl,
            "raw_candidate_pnl": float(row.get("raw_candidate_pnl", pnl) or 0.0),
            "slippage_per_side": float(row.get("slippage_per_side", 0.0) or 0.0),
            "exit_reason": str(row.get("exit_reason", "")),
            "label_source": str(row.get("label_source", "")),
            "entry_ask": none_or_float(row.get("entry_ask")),
            "entry_bid": none_or_float(row.get("entry_bid")),
            "premium_paid": none_or_float(row.get("entry_premium")),
            "fold": str(row.get("fold", "")),
            "stage": "research_challenger",
            "segment": str(row.get("reported_split", "")),
            "source_protocol": str(source_protocol),
        }
        rows.append(item)
    rows.sort(key=lambda item: (int(item["seed"]), item["decision_time"], item["exit_time"], item["candidate_uid"]))
    return rows


def load_fair_contract_strict_replay_trades(
    frame: pd.DataFrame,
    *,
    source_protocol: str,
) -> list[dict[str, Any]]:
    """Load Protocol101 fair-contract strict replay rows for visual inspection."""
    rows = []
    for row in frame.to_dict("records"):
        right = str(row.get("right", "")).upper()
        decision_time = iso_utc(row.get("decision_time"))
        exit_time = iso_utc(row.get("synthetic_exit_time", row.get("exit_time")))
        contract_id = str(row.get("contract_id", ""))
        session = str(row.get("session", ""))
        candidate_uid = f"{session}:{decision_time}:{contract_id}"
        pnl = float(row.get("stressed_pnl", row.get("raw_label_pnl", 0.0)) or 0.0)
        raw_pnl = float(row.get("raw_label_pnl", pnl) or 0.0)
        item = {
            "candidate_uid": candidate_uid,
            "trade_uid": candidate_uid,
            "contract_id": contract_id,
            "seed": 1,
            "entry_seed": 1,
            "session": session,
            "decision_time": decision_time,
            "exit_time": exit_time,
            "right": right,
            "side": "CALL" if right == "C" else "PUT" if right == "P" else right,
            "offset": float(row.get("offset", 0.0) or 0.0),
            "score": float(row.get("score", 0.0) or 0.0),
            "threshold": float(row.get("threshold", 0.0) or 0.0),
            "pnl": pnl,
            "raw_candidate_pnl": raw_pnl,
            "slippage_per_side": 0.0,
            "exit_reason": "synthetic_policy_exit",
            "label_source": "protocol101_fair_contract_strict_replay",
            "entry_ask": none_or_float(row.get("entry_ask")),
            "entry_bid": None,
            "premium_paid": none_or_float(row.get("premium_at_risk")),
            "fold": str(row.get("split", "")),
            "stage": "protocol101_fair_contract_strict_replay",
            "segment": str(row.get("split", "")),
            "source_protocol": str(source_protocol),
            "feature_hash": str(row.get("feature_hash", "")),
            "source_quote_time": str(row.get("source_quote_time", "")),
            "source_context_time": str(row.get("source_context_time", "")),
        }
        rows.append(item)
    rows.sort(key=lambda item: (int(item["seed"]), item["decision_time"], item["exit_time"], item["candidate_uid"]))
    return rows


def load_spx_bars_for_trade_sessions(spx_dir: Path, *, sessions: list[str]) -> list[dict[str, Any]]:
    if not sessions:
        return load_spx_bars(spx_dir)
    if not spx_dir.exists():
        raise SystemExit(f"missing SPX directory: {spx_dir}")
    frames = []
    for session in sessions:
        path = spx_dir / f"{session}.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["event_time", "open", "high", "low", "close"])
            price_columns = ["open", "high", "low", "close"]
        except Exception:
            try:
                frame = pd.read_parquet(path, columns=["event_time", "close"])
                price_columns = ["close"]
            except Exception:
                continue
        if "event_time" not in frame.columns or "close" not in frame.columns or frame.empty:
            continue
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        for column in price_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ("open", "high", "low"):
            if column not in frame.columns:
                frame[column] = frame["close"]
        frame = frame.dropna(subset=["event_time", "close"])
        frame["session"] = session
        frames.append(frame)
    if not frames:
        return load_spx_bars(spx_dir)
    spx = pd.concat(frames, ignore_index=True).drop_duplicates("event_time").sort_values("event_time")
    return [
        {
            "bar": i,
            "t": int(row.event_time.value // 1_000_000),
            "iso": row.event_time.isoformat(),
            "session": str(row.session),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for i, row in enumerate(spx.itertuples(index=False))
    ]


def write_challenger_trades_html(
    path: Path,
    *,
    trades: list[dict[str, Any]],
    spx: list[dict[str, Any]],
    starting_equity: float,
    skipped_trades: list[dict[str, Any]],
    chart_title: str = "Challenger Full-Action History",
) -> None:
    body = TRADE_BODY.replace("Protocol 101 Trade Overlay", f"{chart_title} Trade Overlay")
    body = body.replace("Protocol 101 Trades", f"{chart_title} Trades")
    body = body.replace("Single-account replay: one frozen seed", "Research-only single-account replay: one frozen seed")
    payload = {
        "title": f"{chart_title} Trade Overlay",
        "startingEquity": float(starting_equity),
        "paperSeed": int(trades[0]["seed"]) if trades else None,
        "skippedTrades": len(skipped_trades),
        "unaffordableSkippedTrades": sum(1 for row in skipped_trades if row.get("paper_skip_reason") == "insufficient_cash"),
        "spx": spx,
        "trades": trades,
    }
    path.write_text(html_shell(title=f"{chart_title} Trades", body=body, payload=payload))


def write_report(
    path: Path,
    *,
    payload: dict[str, Any],
    trades: list[dict[str, Any]],
    research_trades: list[dict[str, Any]],
    skipped_trades: list[dict[str, Any]],
    paper_summary: list[dict[str, Any]],
    stress_per_side: float,
) -> None:
    concentration = profit_concentration(trades)
    ending_stress = equity_after_stress(trades, float(payload["starting_equity"]), stress_per_side)
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Paper Replay",
        "",
        f"- Paper seed: `{payload['paper_seed']}`",
        f"- Starting equity: `${payload['starting_equity']:,.0f}`",
        f"- Trades: `{len(trades)}`",
        f"- Skipped: `{len(skipped_trades)}`",
        f"- Research trades before paper filtering: `{len(research_trades)}`",
        f"- Stressed ending equity with ${stress_per_side:.2f}/side: `${ending_stress:,.0f}`",
        f"- First trade: `{payload['first_trade']}`",
        f"- Last trade: `{payload['last_trade']}`",
        "",
        "## Account Summary",
        "",
        "```json",
        json.dumps(paper_summary, indent=2, sort_keys=True),
        "```",
    ]
    if concentration:
        best = concentration["best_trade"]
        worst = concentration["worst_trade"]
        lines.extend(
            [
                "",
                "## Visual Inspection Targets",
                "",
                f"- Best trade: `{best['session']}` `{best['side']}` `{best['contract_id']}` `${float(best['pnl']):,.0f}`",
                f"- Worst trade: `{worst['session']}` `{worst['side']}` `{worst['contract_id']}` `${float(worst['pnl']):,.0f}`",
                f"- Best day: `{concentration['best_day_session']}` `${concentration['best_day_pnl']:,.0f}`",
                f"- Worst day: `{concentration['worst_day_session']}` `${concentration['worst_day_pnl']:,.0f}`",
                f"- Top 20 trades: `${concentration['top_20']:,.0f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Block PnL",
            "",
            "| stage | segment | first | last | trades | pnl | stressed_pnl | win_rate |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in block_summary_rows(trades, stress_per_side):
        lines.append(
            f"| {row['stage']} | {row['segment']} | {row['first']} | {row['last']} | "
            f"{row['trades']} | ${row['pnl']:,.0f} | ${row['stressed_pnl']:,.0f} | {row['win_rate'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Trades chart: `{payload['outputs']['trades_html']}`",
            f"- Equity chart: `{payload['outputs']['equity_html']}`",
            f"- Trade CSV: `{payload['outputs']['trades_csv']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
