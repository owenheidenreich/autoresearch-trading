"""Protocol 122: capital realism audit for frozen Protocol 101.

This experiment does not retrain, download data, call a broker, or place
orders. It asks a narrower question than the historical equity curve:

Can the frozen Protocol 101 one-contract SPXW replay actually operate at the
intended paper/live capital baseline? The user's $500 IBKR cash is an account
access and market-data reserve, not the intended trading bankroll.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.export_protocol101_trade_charts import (
    DEFAULT_NORMALIZED_DIRS,
    DEFAULT_PROTOCOL101_DIR,
    DEFAULT_PROTOCOL107_DIR,
    DEFAULT_PROTOCOL112_DIR,
    DEFAULT_SPX_DIR,
    add_equity_fields,
    attach_spx_prices,
    backfill_option_quote_accounting,
    build_paper_account_summary,
    build_paper_account_trades,
    load_canonical_trades,
    load_spx_bars,
    none_or_float,
    premium_dollars,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_122_protocol101_capital_realism")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_STARTING_EQUITIES = (500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0)
TRADING_CAPITAL_BASELINE = 10_000.0
IBKR_ACCESS_RESERVE = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--protocol107-dir", type=Path, default=DEFAULT_PROTOCOL107_DIR)
    parser.add_argument("--protocol112-dir", type=Path, default=DEFAULT_PROTOCOL112_DIR)
    parser.add_argument("--spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--normalized-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--paper-seed", type=int, default=1)
    parser.add_argument("--starting-equity", action="append", type=float, default=None)
    parser.add_argument("--include-train-validation", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    starting_equities = tuple(args.starting_equity or DEFAULT_STARTING_EQUITIES)
    normalized_dirs = tuple(args.normalized_dir or DEFAULT_NORMALIZED_DIRS)

    research_trades = _load_research_trades(
        protocol101_dir=args.protocol101_dir,
        protocol107_dir=args.protocol107_dir,
        protocol112_dir=args.protocol112_dir if args.include_train_validation else None,
        spx_dir=args.spx_dir,
        normalized_dirs=normalized_dirs,
    )
    if not research_trades:
        raise SystemExit("no Protocol 101 trades found for capital realism audit")

    rows: list[dict[str, Any]] = []
    skipped_examples: list[dict[str, Any]] = []
    for starting_equity in starting_equities:
        taken, skipped = build_paper_account_trades(
            research_trades,
            starting_equity=float(starting_equity),
            paper_seed=int(args.paper_seed),
        )
        summary = _single_summary(
            taken,
            skipped,
            starting_equity=float(starting_equity),
            paper_seed=int(args.paper_seed),
        )
        rows.append(summary)
        skipped_examples.extend(_insufficient_cash_examples(skipped, starting_equity=float(starting_equity), limit=8))

    premium_summary = _premium_summary(
        [row for row in research_trades if int(row.get("seed", -1)) == int(args.paper_seed)]
    )
    decision = decide(rows)
    payload = {
        "protocol": "122_protocol101_capital_realism",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "paper_seed": int(args.paper_seed),
        "include_train_validation": bool(args.include_train_validation),
        "source_mode": "includes_train_validation" if args.include_train_validation else "holdout_only",
        "capital_assumption": {
            "ibkr_access_reserve": IBKR_ACCESS_RESERVE,
            "paper_trading_starting_equity": TRADING_CAPITAL_BASELINE,
            "why": (
                "$500 is kept in IBKR to maintain account/data access. The paper account starts at $10,000, "
                "matching the intended future real-money bankroll."
            ),
        },
        "starting_equities": [float(value) for value in starting_equities],
        "research_trade_count": len([row for row in research_trades if int(row.get("seed", -1)) == int(args.paper_seed)]),
        "premium_summary": premium_summary,
        "capital_results": rows,
        "insufficient_cash_examples": skipped_examples,
        "next_hypothesis": next_hypothesis(rows),
    }

    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    csv_path = args.out_dir / "capital_results.csv"
    examples_path = args.out_dir / "insufficient_cash_examples.csv"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, rows)
    write_csv(examples_path, skipped_examples)
    write_report(report_path, payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, report_path)

    print(json.dumps({"decision": decision, "report": str(report_path)}, indent=2, sort_keys=True))
    return 0


def _load_research_trades(
    *,
    protocol101_dir: Path,
    protocol107_dir: Path,
    protocol112_dir: Path | None,
    spx_dir: Path,
    normalized_dirs: tuple[Path, ...],
) -> list[dict[str, Any]]:
    spx = load_spx_bars(spx_dir)
    trades = load_canonical_trades(
        protocol101_dir=protocol101_dir,
        protocol107_dir=protocol107_dir,
        protocol112_dir=protocol112_dir,
    )
    trades = attach_spx_prices(trades, spx)
    trades = backfill_option_quote_accounting(trades, normalized_dirs)
    return add_equity_fields(trades)


def _single_summary(
    taken: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    *,
    starting_equity: float,
    paper_seed: int,
) -> dict[str, Any]:
    if taken:
        summary_rows = build_paper_account_summary(taken, starting_equity, skipped_trades=skipped)
        base = dict(summary_rows[0])
    else:
        base = {
            "seed": int(paper_seed),
            "starting_equity": round(float(starting_equity), 2),
            "trades": 0,
            "skipped_trades": len(skipped),
            "skip_reasons": _skip_reasons(skipped),
            "winning_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "ending_equity": round(float(starting_equity), 2),
            "return_on_starting_equity": 0.0,
            "min_equity": round(float(starting_equity), 2),
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "worst_day_pnl": 0.0,
            "max_daily_loss": 0.0,
            "known_premium_trades": 0,
            "premium_coverage": 0.0,
            "max_known_buying_power": None,
            "avg_known_buying_power": None,
            "max_known_buying_power_pct_equity": None,
            "known_path_trades": 0,
            "path_coverage": 0.0,
            "worst_intratrade_mae": None,
            "max_intratrade_loss": None,
            "min_intratrade_equity": None,
            "best_intratrade_mfe": None,
        }
    base["insufficient_cash_skips"] = int(base.get("skip_reasons", {}).get("insufficient_cash", 0))
    base["overlap_skips"] = int(base.get("skip_reasons", {}).get("overlap_open_position", 0))
    base["missing_premium_skips"] = int(base.get("skip_reasons", {}).get("missing_entry_premium", 0))
    base["total_candidate_events"] = int(base["trades"]) + int(base["skipped_trades"])
    base["trade_capture_rate"] = round(
        int(base["trades"]) / max(1, int(base["total_candidate_events"])),
        6,
    )
    bp_fracs = [
        float(value)
        for value in (none_or_float(row.get("paper_buying_power_pct_cash")) for row in taken)
        if value is not None
    ]
    base["max_buying_power_pct_current_cash"] = round(max(bp_fracs), 6) if bp_fracs else None
    base["avg_buying_power_pct_current_cash"] = round(sum(bp_fracs) / len(bp_fracs), 6) if bp_fracs else None
    base["trades_over_50pct_cash"] = sum(1 for value in bp_fracs if value > 0.50)
    base["trades_over_75pct_cash"] = sum(1 for value in bp_fracs if value > 0.75)
    base["trades_over_90pct_cash"] = sum(1 for value in bp_fracs if value > 0.90)
    base["first_taken_trade"] = taken[0]["decision_time"] if taken else None
    base["first_skipped_insufficient_cash"] = next(
        (row["decision_time"] for row in skipped if row.get("paper_skip_reason") == "insufficient_cash"),
        None,
    )
    return base


def _skip_reasons(skipped: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in skipped:
        reason = str(row.get("paper_skip_reason", "unknown") or "unknown")
        out[reason] = out.get(reason, 0) + 1
    return out


def _premium_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    premiums = sorted(value for value in (premium_dollars(row) for row in trades) if value is not None)
    if not premiums:
        return {"count": 0}
    series = pd.Series(premiums)
    return {
        "count": int(len(premiums)),
        "min": round(float(series.min()), 2),
        "p10": round(float(series.quantile(0.10)), 2),
        "median": round(float(series.median()), 2),
        "p90": round(float(series.quantile(0.90)), 2),
        "max": round(float(series.max()), 2),
        "pct_at_or_below_500": round(float((series <= 500.0).mean()), 6),
        "pct_at_or_below_1000": round(float((series <= 1_000.0).mean()), 6),
        "pct_at_or_below_2500": round(float((series <= 2_500.0).mean()), 6),
        "pct_at_or_below_5000": round(float((series <= 5_000.0).mean()), 6),
    }


def _insufficient_cash_examples(
    skipped: list[dict[str, Any]],
    *,
    starting_equity: float,
    limit: int,
) -> list[dict[str, Any]]:
    out = []
    for row in skipped:
        if row.get("paper_skip_reason") != "insufficient_cash":
            continue
        out.append(
            {
                "starting_equity": float(starting_equity),
                "decision_time": row.get("decision_time"),
                "session": row.get("session"),
                "side": row.get("side"),
                "offset": row.get("offset"),
                "contract_id": row.get("contract_id"),
                "cash_before": row.get("paper_cash_before"),
                "premium": none_or_float(row.get("paper_premium")),
                "pnl_if_taken": none_or_float(row.get("pnl")),
                "score": none_or_float(row.get("score")),
                "threshold": none_or_float(row.get("threshold")),
            }
        )
        if len(out) >= limit:
            break
    return out


def decide(rows: list[dict[str, Any]]) -> str:
    by_cash = {float(row["starting_equity"]): row for row in rows}
    reference = by_cash.get(TRADING_CAPITAL_BASELINE) or rows[-1]
    if int(reference["trades"]) <= 0:
        return "blocked_no_affordable_reference_trades"
    if int(reference["insufficient_cash_skips"]) > 0:
        return "blocked_10000_baseline_has_unaffordable_trades"
    return "pass_10000_paper_capital_baseline"


def next_hypothesis(rows: list[dict[str, Any]]) -> str:
    by_cash = {float(row["starting_equity"]): row for row in rows}
    reference = by_cash.get(TRADING_CAPITAL_BASELINE) or rows[-1]
    if int(reference.get("insufficient_cash_skips", 0)) == 0:
        return (
            "Use $10,000 as the paper-account and eventual real-money capital baseline. The $500 IBKR cash is only "
            "an access reserve. Next research should focus on live-data parity and order-state rehearsal, not shrinking "
            "the strategy to fit a $500 trading account."
        )
    return (
        "The $10,000 baseline still has unaffordable trades. Before paper orders, add an affordability-aware entry gate "
        "that rejects any contract whose ask premium exceeds available paper cash."
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    premium = payload["premium_summary"]
    rows = payload["capital_results"]
    lines = [
        "# Protocol 122: Protocol 101 Capital Realism",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source mode: `{payload['source_mode']}`",
        f"- Paper seed: `{payload['paper_seed']}`",
        f"- Research candidate events: `{payload['research_trade_count']}`",
        f"- IBKR access reserve: `${payload['capital_assumption']['ibkr_access_reserve']:,.0f}`",
        f"- Paper trading starting equity: `${payload['capital_assumption']['paper_trading_starting_equity']:,.0f}`",
        "",
        payload["capital_assumption"]["why"],
        "",
        "## Premium Reality",
        "",
        f"- Known premium rows: `{premium.get('count', 0)}`",
        f"- Premium min / median / max: `${premium.get('min')}` / `${premium.get('median')}` / `${premium.get('max')}`",
        f"- Percent affordable at $500 access reserve: `{_pct(premium.get('pct_at_or_below_500'))}`",
        f"- Percent affordable at $1,000: `{_pct(premium.get('pct_at_or_below_1000'))}`",
        f"- Percent affordable at $2,500: `{_pct(premium.get('pct_at_or_below_2500'))}`",
        "",
        "## Capital Sweep",
        "",
        "| starting_cash | trades | skipped | insufficient_cash | ending_equity | total_pnl | max_drawdown | min_intratrade_equity | max_premium | capture_rate | max_bp/current_cash | >75% cash trades |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"${row['starting_equity']:,.0f} | "
            f"{row['trades']} | "
            f"{row['skipped_trades']} | "
            f"{row['insufficient_cash_skips']} | "
            f"${row['ending_equity']:,.0f} | "
            f"${row['total_pnl']:,.0f} | "
            f"${row['max_drawdown']:,.0f} | "
            f"{_money(row.get('min_intratrade_equity'))} | "
            f"{_money(row.get('max_known_buying_power'))} | "
            f"{float(row['trade_capture_rate']) * 100:.1f}% |"
            f" {_pct(row.get('max_buying_power_pct_current_cash'))} | "
            f"{row.get('trades_over_75pct_cash')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["next_hypothesis"],
        "",
        "The $10,000 equity curve remains the clean inspection baseline. The lower-cash rows are sensitivity checks only; they are not the intended paper/live bankroll. This audit is a capital-assumption check, not live-trading approval.",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 122 Protocol101 Capital Realism"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Ran a no-training capital realism audit for frozen Protocol 101 with $10,000 as the intended paper/live trading baseline and $500 treated only as the IBKR access reserve.
Reason: Live paper execution is blocked by cash settlement and live market-data subscriptions. The project needed to prevent confusing the $500 account-access cash with the $10,000 paper/live trading bankroll.
Data Used: Existing Protocol 101/107 trade artifacts, local official SPX bars, and normalized local option quote files only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: {payload['next_hypothesis']}
Owner: Codex
```
"""
    existing = ledger.read_text() if ledger.exists() else ""
    if marker not in existing:
        ledger.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def _money(value: Any) -> str:
    number = none_or_float(value)
    return "n/a" if number is None else f"${number:,.0f}"


def _pct(value: Any) -> str:
    number = none_or_float(value)
    return "n/a" if number is None else f"{number * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
