"""AUDIT_CHALLENGER_FAILURE_SURFACE_V1.

Historically Protocol248. This is a strict metric-scope audit for the frozen
premium-leaning challenger versus PAPER_DEFAULT_PROTOCOL101.

It intentionally uses one seed and one-account serial replay only. Five-seed
medians, five-seed totals, overlapping diagnostics, and directional attribution
subsets are not allowed to masquerade as headline equity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v4.scripts.run_protocol216_full_action_history_vs_protocol101_attribution as p216


ROLE_LABEL = "AUDIT_CHALLENGER_FAILURE_SURFACE_V1"
HISTORICAL_ID = "Protocol248"
DEFAULT_ENRICHED_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_242_premium_blend_vs_protocol101_attribution/enriched_policy_trades.csv"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_248_challenger_failure_surface")
STARTING_EQUITY = 10_000.0
REQUIRED_SEGMENTS = ("q3_2025", "q4_2025", "q1_2026", "recent_2026")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-trades", type=Path, default=DEFAULT_ENRICHED_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--starting-equity", type=float, default=STARTING_EQUITY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_scoped_trades(args.enriched_trades, seed=int(args.seed))
    frame = add_failure_surface_fields(frame)

    headline = summarize_group(frame, ["policy"])
    by_split = summarize_group(frame, ["reported_split", "policy"])
    by_side = summarize_group(frame, ["reported_split", "policy", "right"])
    by_moneyness = summarize_group(frame, ["reported_split", "policy", "moneyness"])
    by_premium = summarize_group(frame, ["reported_split", "policy", "premium_bucket"])
    by_time = summarize_group(frame, ["reported_split", "policy", "time_bucket"])
    by_exit = summarize_group(frame, ["reported_split", "policy", "exit_reason"])
    by_day = summarize_days(frame, starting_equity=float(args.starting_equity))
    drawdowns = summarize_drawdowns(frame, starting_equity=float(args.starting_equity))
    concentration = summarize_concentration(frame)
    directional = summarize_directional(frame)
    churn_summary, churn_chains = p216.summarize_churn(frame, gap_minutes=30.0)
    quality_deltas = summarize_quality_deltas(frame)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / failure-surface audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "metric_scope": "single_seed_strict_one_account_serial_replay",
        "seed": int(args.seed),
        "starting_equity": float(args.starting_equity),
        "data_used": str(args.enriched_trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "row_counts": count_rows(frame),
        "headline": headline,
        "by_split": by_split,
        "by_side": by_side,
        "by_moneyness": by_moneyness,
        "by_premium": by_premium,
        "by_time_bucket": by_time,
        "by_exit_reason": by_exit,
        "by_day": by_day,
        "drawdowns": drawdowns,
        "concentration": concentration,
        "directional_move_capture": directional,
        "churn_summary": churn_summary,
        "quality_deltas": quality_deltas,
        "scope_guards": scope_guards(frame),
        "decision": decide(headline, drawdowns, quality_deltas),
        "next_experiment": (
            "Run EXP_ENTRY_QUALITY_CALIBRATOR_V1 first. The main failure surface is lower hit rate, "
            "higher churn/drawdown, and weak low-premium/OTM expectancy rather than insufficient total PnL."
        ),
    }
    write_csv(args.out_dir / "headline.csv", headline)
    write_csv(args.out_dir / "by_split.csv", by_split)
    write_csv(args.out_dir / "by_side.csv", by_side)
    write_csv(args.out_dir / "by_moneyness.csv", by_moneyness)
    write_csv(args.out_dir / "by_premium.csv", by_premium)
    write_csv(args.out_dir / "by_time_bucket.csv", by_time)
    write_csv(args.out_dir / "by_exit_reason.csv", by_exit)
    write_csv(args.out_dir / "by_day.csv", by_day)
    write_csv(args.out_dir / "drawdowns.csv", drawdowns)
    write_csv(args.out_dir / "concentration.csv", concentration)
    write_csv(args.out_dir / "directional_move_capture.csv", directional)
    write_csv(args.out_dir / "churn_summary.csv", churn_summary)
    churn_chains.to_csv(args.out_dir / "churn_chains.csv", index=False)
    write_csv(args.out_dir / "quality_deltas.csv", quality_deltas)
    frame.to_csv(args.out_dir / "scoped_seed_trades.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_scoped_trades(path: Path, *, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "seed" not in frame.columns:
        raise ValueError("enriched trade file must include seed column")
    frame = frame[pd.to_numeric(frame["seed"], errors="coerce").fillna(-1).astype(int).eq(int(seed))].copy()
    frame = frame[frame["policy"].isin(["challenger", "protocol101"])].copy()
    frame = frame[frame["reported_split"].isin(REQUIRED_SEGMENTS)].copy()
    for column in ["decision_time", "exit_time", "decision_ts", "exit_ts"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if "decision_ts" not in frame.columns or frame["decision_ts"].isna().all():
        frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    if "exit_ts" not in frame.columns or frame["exit_ts"].isna().all():
        frame["exit_ts"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    for column in ["pnl", "entry_premium", "entry_ask", "entry_bid", "offset", "directional_underlying_move", "path_mfe", "path_mae", "duration_minutes"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "entry_ask_live" in frame.columns:
        frame["entry_ask_live"] = pd.to_numeric(frame["entry_ask_live"], errors="coerce")
        frame["entry_ask"] = frame["entry_ask"].where(frame["entry_ask"].notna(), frame["entry_ask_live"])
    repaired_premium = frame["entry_ask"] * 100.0
    frame["entry_premium"] = frame["entry_premium"].where(frame["entry_premium"].notna(), repaired_premium)
    frame = frame[frame["decision_ts"].notna() & frame["exit_ts"].notna()].copy()
    return frame.sort_values(["policy", "reported_split", "session", "decision_ts", "contract_id"]).reset_index(drop=True)


def add_failure_surface_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["moneyness"] = [moneyness(right, offset) for right, offset in zip(out["right"], out["offset"])]
    out["premium_bucket"] = pd.cut(
        out["entry_premium"].fillna(-1.0),
        bins=[-1.0, 0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 1_000_000.0],
        labels=["missing", "0-500", "500-1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000", "3000-4000", "4000+"],
    ).astype(str)
    out["abs_offset_bucket"] = pd.cut(
        out["offset"].abs().fillna(-1.0),
        bins=[-1.0, 0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 1_000_000.0],
        labels=["0", "<=5", "<=10", "<=20", "<=35", "<=50", ">50"],
    ).astype(str)
    out["trade_return_on_premium"] = np.where(out["entry_premium"] > 0.0, out["pnl"] / out["entry_premium"], np.nan)
    out["mfe_capture"] = np.where(out["path_mfe"] > 0.0, out["pnl"] / out["path_mfe"], np.nan)
    return out


def moneyness(right: Any, offset: Any) -> str:
    try:
        value = float(offset)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value):
        return "unknown"
    if abs(value) <= 2.5:
        return "ATM"
    side = str(right)
    if side == "C":
        return "ITM" if value < 0.0 else "OTM"
    if side == "P":
        return "ITM" if value > 0.0 else "OTM"
    return "unknown"


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True, observed=False):
        if not isinstance(key, tuple):
            key = (key,)
        pnl = group["pnl"].fillna(0.0)
        wins = pnl[pnl > 0.0]
        losses = pnl[pnl < 0.0]
        premium = group["entry_premium"].dropna()
        row = {column: str(value) for column, value in zip(group_cols, key)}
        row.update(
            {
                "trades": int(len(group)),
                "pnl": finite(pnl.sum()),
                "win_rate": finite((pnl >= 0.0).mean()),
                "positive_win_rate": finite((pnl > 0.0).mean()),
                "avg_pnl": finite(pnl.mean()),
                "median_pnl": finite(pnl.median()),
                "profit_factor": profit_factor(pnl),
                "median_premium": finite(premium.median()) if len(premium) else 0.0,
                "pnl_per_premium": finite(pnl.sum() / premium.sum()) if len(premium) and abs(float(premium.sum())) > 1e-9 else 0.0,
                "median_return_on_premium": finite(group["trade_return_on_premium"].median()),
                "median_duration_minutes": finite(group["duration_minutes"].median()),
                "median_mfe_capture": finite(group["mfe_capture"].median()),
                "gross_profit": finite(wins.sum()),
                "gross_loss": finite(losses.sum()),
            }
        )
        rows.append(row)
    return rows


def summarize_days(frame: pd.DataFrame, *, starting_equity: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (policy, split, session), group in frame.groupby(["policy", "reported_split", "session"], sort=True):
        pnl = group["pnl"].fillna(0.0)
        rows.append(
            {
                "policy": str(policy),
                "reported_split": str(split),
                "session": str(session),
                "trades": int(len(group)),
                "pnl": finite(pnl.sum()),
                "win_rate": finite((pnl >= 0.0).mean()),
                "pnl_pct_starting_equity": finite(pnl.sum() / starting_equity),
            }
        )
    return rows


def summarize_drawdowns(frame: pd.DataFrame, *, starting_equity: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, group in frame.sort_values(["session", "decision_ts"]).groupby("policy", sort=True):
        pnl = group["pnl"].fillna(0.0).to_numpy(dtype=float)
        equity = starting_equity + np.cumsum(pnl)
        running_high = np.maximum.accumulate(np.r_[starting_equity, equity])[:-1]
        drawdown = equity - running_high
        drawdown_pct = np.divide(drawdown, running_high, out=np.zeros_like(drawdown), where=running_high > 0.0)
        day = pd.DataFrame({"session": group["session"].astype(str).to_numpy(), "pnl": pnl}).groupby("session")["pnl"].sum()
        rows.append(
            {
                "policy": str(policy),
                "starting_equity": float(starting_equity),
                "ending_equity": finite(starting_equity + pnl.sum()),
                "total_pnl": finite(pnl.sum()),
                "max_drawdown": finite(drawdown.min()) if len(drawdown) else 0.0,
                "max_drawdown_pct": finite(drawdown_pct.min()) if len(drawdown_pct) else 0.0,
                "max_drawdown_pct_of_start": finite(drawdown.min() / starting_equity) if len(drawdown) and starting_equity > 0.0 else 0.0,
                "min_equity": finite(equity.min()) if len(equity) else starting_equity,
                "worst_day_pnl": finite(day.min()) if len(day) else 0.0,
                "best_day_pnl": finite(day.max()) if len(day) else 0.0,
                "losing_days": int((day < 0.0).sum()) if len(day) else 0,
                "trading_days": int(len(day)),
            }
        )
    return rows


def summarize_concentration(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, group in frame.groupby("policy", sort=True):
        pnl = group["pnl"].fillna(0.0)
        net = finite(pnl.sum())
        positive = pnl[pnl > 0.0].sort_values(ascending=False)
        day = group.groupby("session")["pnl"].sum().sort_values(ascending=False)
        for n in [5, 10, 20]:
            top_trade = finite(positive.head(n).sum())
            top_day = finite(day.head(n).sum())
            rows.append(
                {
                    "policy": str(policy),
                    "top_n": int(n),
                    "top_trade_pnl": top_trade,
                    "top_trade_share_of_net": finite(top_trade / net) if abs(net) > 1e-9 else 0.0,
                    "top_day_pnl": top_day,
                    "top_day_share_of_net": finite(top_day / net) if abs(net) > 1e-9 else 0.0,
                }
            )
    return rows


def summarize_directional(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, policy), group in frame.groupby(["reported_split", "policy"], sort=True):
        for threshold in [10.0, 20.0, 30.0]:
            moves = group[group["directional_underlying_move"] >= threshold]
            rows.append(
                {
                    "reported_split": str(split),
                    "policy": str(policy),
                    "directional_move_threshold_spx": float(threshold),
                    "trades": int(len(moves)),
                    "share_of_policy_trades": finite(len(moves) / max(len(group), 1)),
                    "pnl": finite(moves["pnl"].sum()),
                    "win_rate": finite((moves["pnl"] >= 0.0).mean()) if len(moves) else 0.0,
                    "median_duration_minutes": finite(moves["duration_minutes"].median()) if len(moves) else 0.0,
                }
            )
    return rows


def summarize_quality_deltas(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = {row["policy"]: row for row in summarize_group(frame, ["policy"])}
    challenger = metrics.get("challenger", {})
    protocol101 = metrics.get("protocol101", {})
    for metric in ["pnl", "win_rate", "avg_pnl", "profit_factor", "median_premium", "pnl_per_premium"]:
        rows.append(
            {
                "metric": metric,
                "challenger": finite(challenger.get(metric)),
                "protocol101": finite(protocol101.get(metric)),
                "delta_challenger_minus_protocol101": finite(challenger.get(metric)) - finite(protocol101.get(metric)),
            }
        )
    drawdowns = {row["policy"]: row for row in summarize_drawdowns(frame, starting_equity=STARTING_EQUITY)}
    for metric in ["max_drawdown", "max_drawdown_pct", "max_drawdown_pct_of_start", "worst_day_pnl"]:
        rows.append(
            {
                "metric": metric,
                "challenger": finite(drawdowns.get("challenger", {}).get(metric)),
                "protocol101": finite(drawdowns.get("protocol101", {}).get(metric)),
                "delta_challenger_minus_protocol101": finite(drawdowns.get("challenger", {}).get(metric)) - finite(drawdowns.get("protocol101", {}).get(metric)),
            }
        )
    return rows


def scope_guards(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "metric_scope": "single_seed_strict_one_account_serial_replay",
        "allowed_policies": sorted(frame["policy"].dropna().astype(str).unique().tolist()),
        "seeds": sorted(pd.to_numeric(frame["seed"], errors="coerce").dropna().astype(int).unique().tolist()),
        "forbidden_scopes_excluded": [
            "five_seed_total",
            "five_seed_median",
            "overlapping_candidate_pnl",
            "directional_subset_as_headline_equity",
        ],
    }


def count_rows(frame: pd.DataFrame) -> dict[str, int]:
    return {str(policy): int(len(group)) for policy, group in frame.groupby("policy", sort=True)}


def decide(headline: list[dict[str, Any]], drawdowns: list[dict[str, Any]], quality_deltas: list[dict[str, Any]]) -> str:
    by_policy = {row["policy"]: row for row in headline}
    dd = {row["policy"]: row for row in drawdowns}
    challenger = by_policy.get("challenger", {})
    protocol101 = by_policy.get("protocol101", {})
    higher_pnl = finite(challenger.get("pnl")) > finite(protocol101.get("pnl"))
    lower_win = finite(challenger.get("win_rate")) < finite(protocol101.get("win_rate"))
    worse_dd = finite(dd.get("challenger", {}).get("max_drawdown")) < finite(dd.get("protocol101", {}).get("max_drawdown"))
    if higher_pnl and (lower_win or worse_dd):
        return "failure_surface_confirms_research_only_higher_pnl_lower_quality"
    if higher_pnl:
        return "failure_surface_supports_candidate_but_requires_runtime_parity"
    return "failure_surface_rejects_challenger_vs_protocol101"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Metric scope: `{payload['metric_scope']}`",
        f"Seed: `{payload['seed']}`",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Headline",
        "",
        "| policy | trades | PnL | ending equity | win rate | avg PnL | PF | median premium | PnL/premium |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["headline"]:
        ending = float(payload["starting_equity"]) + finite(row["pnl"])
        lines.append(
            f"| {row['policy']} | {row['trades']} | {money(row['pnl'])} | {money(ending)} | "
            f"{pct(row['win_rate'])} | {money(row['avg_pnl'])} | {row['profit_factor']:.2f} | "
            f"{money(row['median_premium'])} | {row['pnl_per_premium']:.4f} |"
        )
    lines.extend(["", "## Drawdown / Day Risk", "", "| policy | max drawdown | max DD % | max DD / start | worst day | best day | losing days | trading days |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in payload["drawdowns"]:
        lines.append(
            f"| {row['policy']} | {money(row['max_drawdown'])} | {pct(row['max_drawdown_pct'])} | "
            f"{pct(row['max_drawdown_pct_of_start'])} | {money(row['worst_day_pnl'])} | {money(row['best_day_pnl'])} | "
            f"{row['losing_days']} | {row['trading_days']} |"
        )
    lines.extend(["", "## Split Scorecard", "", "| split | policy | trades | PnL | win rate | avg PnL | PF | median premium |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["by_split"]:
        lines.append(
            f"| {row['reported_split']} | {row['policy']} | {row['trades']} | {money(row['pnl'])} | "
            f"{pct(row['win_rate'])} | {money(row['avg_pnl'])} | {row['profit_factor']:.2f} | {money(row['median_premium'])} |"
        )
    lines.extend(["", "## Moneyness", "", "| split | policy | moneyness | trades | PnL | win rate | avg PnL | median premium |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in payload["by_moneyness"]:
        lines.append(
            f"| {row['reported_split']} | {row['policy']} | {row['moneyness']} | {row['trades']} | "
            f"{money(row['pnl'])} | {pct(row['win_rate'])} | {money(row['avg_pnl'])} | {money(row['median_premium'])} |"
        )
    lines.extend(["", "## Quality Deltas", "", "| metric | challenger | Protocol101 | delta |", "|---|---:|---:|---:|"])
    for row in payload["quality_deltas"]:
        lines.append(
            f"| {row['metric']} | {fmt_metric(row['metric'], row['challenger'])} | "
            f"{fmt_metric(row['metric'], row['protocol101'])} | {fmt_metric(row['metric'], row['delta_challenger_minus_protocol101'])} |"
        )
    lines.extend(
        [
            "",
            "## Scope Guard",
            "",
            "This audit is a single-seed strict-serial scorecard. It does not use five-seed totals, five-seed medians, overlapping candidate PnL, or directional subset PnL as headline equity.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Scoped trades: `{path.parent / 'scoped_seed_trades.csv'}`",
            f"- Split scorecard: `{path.parent / 'by_split.csv'}`",
            f"- Churn chains: `{path.parent / 'churn_chains.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def profit_factor(pnl: pd.Series) -> float:
    wins = float(pnl[pnl > 0.0].sum())
    losses = float(-pnl[pnl < 0.0].sum())
    if losses <= 1e-9:
        return float("inf") if wins > 0.0 else 0.0
    return finite(wins / losses)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def money(value: Any) -> str:
    return f"${finite(value):,.0f}"


def pct(value: Any) -> str:
    return f"{finite(value) * 100.0:.1f}%"


def fmt_metric(metric: str, value: Any) -> str:
    if "rate" in metric or "pct" in metric or metric == "pnl_per_premium":
        return pct(value)
    if metric == "profit_factor":
        return f"{finite(value):.2f}"
    return money(value)


if __name__ == "__main__":
    raise SystemExit(main())
