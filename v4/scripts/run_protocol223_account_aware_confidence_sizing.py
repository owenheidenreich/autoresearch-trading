"""EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2.

Historically Protocol223. This is an offline-only sizing experiment for the
return-on-premium trade stream. It tests whether confidence-based multi-contract
position sizing can scale with account equity without becoming a full-port
leverage overlay.

The entry and exit decisions are frozen from Protocol221. This runner changes
only quantity. No paid data is downloaded and no broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_LABEL = "EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2"
HISTORICAL_ID = "Protocol223"
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_221_return_on_premium_full_action_policy/model_trades.csv")
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_BASELINE_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_221_return_on_premium_full_action_policy/summary.json")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_223_account_aware_confidence_sizing")
CONTRACT_MULTIPLIER = 100.0
REQUIRED_SPLITS = ("q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=str, default="10000,25000,50000,100000,250000,1000000")
    parser.add_argument("--min-risk-frac", type=float, default=0.01)
    parser.add_argument("--max-risk-frac", type=float, default=0.10)
    parser.add_argument("--hard-premium-cap-frac", type=float, default=0.12)
    parser.add_argument("--confidence-scale", type=float, default=0.75)
    parser.add_argument("--liquidity-fraction", type=float, default=0.25)
    parser.add_argument("--absolute-max-contracts", type=int, default=100)
    parser.add_argument("--ruin-stop-frac", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    starting_cash_values = parse_cash_values(args.starting_cash)
    trades = load_trades(args.trades, dataset_path=args.dataset)
    protocol101 = load_protocol101_baselines(args.baseline_summary)

    all_rows: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    for cash in starting_cash_values:
        for slippage in (0.0, 0.10, 0.25, 0.50):
            replay = replay_account_aware(
                trades,
                starting_cash=cash,
                min_risk_frac=float(args.min_risk_frac),
                max_risk_frac=float(args.max_risk_frac),
                hard_premium_cap_frac=float(args.hard_premium_cap_frac),
                confidence_scale=float(args.confidence_scale),
                liquidity_fraction=float(args.liquidity_fraction),
                absolute_max_contracts=int(args.absolute_max_contracts),
                ruin_stop_frac=float(args.ruin_stop_frac),
                extra_slippage_per_side=float(slippage),
            )
            replay["starting_cash"] = cash
            replay["extra_slippage_per_side"] = slippage
            all_rows.append(replay)
            all_summary.append(summarize_replay(replay, protocol101=protocol101))

    rows = pd.concat(all_rows, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    rows.to_csv(args.out_dir / "account_aware_sized_trades.csv", index=False)
    summary.to_csv(args.out_dir / "split_balance_stress_summary.csv", index=False)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / offline account-aware sizing simulator",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1_WITH_ACCOUNT_AWARE_SIZING",
        "entry_exit_source": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "starting_cash_values": starting_cash_values,
        "sizing_rule": {
            "min_risk_frac": float(args.min_risk_frac),
            "max_risk_frac": float(args.max_risk_frac),
            "hard_premium_cap_frac": float(args.hard_premium_cap_frac),
            "confidence_scale": float(args.confidence_scale),
            "liquidity_fraction": float(args.liquidity_fraction),
            "absolute_max_contracts": int(args.absolute_max_contracts),
            "ruin_stop_frac": float(args.ruin_stop_frac),
            "confidence_formula": "margin / (margin + confidence_scale), squared before risk interpolation",
            "drawdown_throttle": "risk multiplier decreases linearly to 25% by 15% account drawdown",
            "hard_cap": "skip entry if one contract would exceed hard_premium_cap_frac of current equity",
        },
        "decision": decide(summary),
        "aggregate": aggregate_for_json(summary),
        "risk_summary": risk_summary(rows),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "sized_trades": str(args.out_dir / "account_aware_sized_trades.csv"),
            "split_balance_stress_summary": str(args.out_dir / "split_balance_stress_summary.csv"),
        },
        "next_experiment": (
            "If the account-aware sizing overlay survives stress without excessive drawdown, test it against the "
            "ITM dollar challenger. If it remains inferior or too stress-sensitive, train a blended dollar-plus-premium "
            "utility before adding scale-in/scale-out."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload, summary)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def parse_cash_values(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        value = float(part.strip())
        if value <= 0.0:
            raise ValueError(f"starting cash must be positive: {part}")
        values.append(value)
    return values


def load_trades(path: Path, *, dataset_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if dataset_path.exists() and "entry_ask_size" not in frame.columns:
        sizes = pd.read_parquet(dataset_path, columns=["candidate_uid", "entry_ask_size"])
        sizes = sizes.drop_duplicates("candidate_uid")
        frame = frame.merge(sizes, on="candidate_uid", how="left", validate="many_to_one")
    for column in ["decision_time", "exit_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    numeric = ["score", "threshold", "entry_premium", "entry_premium_with_slippage", "entry_ask_size", "pnl"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = ["reported_split", "session", "decision_time", "exit_time", "entry_premium", "pnl", "score", "threshold"]
    return frame.dropna(subset=required).sort_values(["reported_split", "decision_time", "exit_time", "candidate_uid"]).reset_index(drop=True)


def load_protocol101_baselines(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    aggregate = payload.get("aggregate", {})
    out: dict[str, float] = {}
    if isinstance(aggregate, dict):
        for split, row in aggregate.items():
            if isinstance(row, dict) and "frozen_protocol101_total_pnl" in row:
                out[str(split)] = finite_float(row["frozen_protocol101_total_pnl"], 0.0)
    return out


def replay_account_aware(
    trades: pd.DataFrame,
    *,
    starting_cash: float,
    min_risk_frac: float,
    max_risk_frac: float,
    hard_premium_cap_frac: float,
    confidence_scale: float,
    liquidity_fraction: float,
    absolute_max_contracts: int,
    ruin_stop_frac: float,
    extra_slippage_per_side: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in trades.groupby("reported_split", sort=True):
        equity = float(starting_cash)
        peak = equity
        halted = False
        for item in group.sort_values(["decision_time", "exit_time", "candidate_uid"]).to_dict("records"):
            before = equity
            peak = max(peak, equity)
            drawdown_frac = max(0.0, (peak - equity) / peak) if peak > 0.0 else 1.0
            premium = finite_float(item.get("entry_premium_with_slippage"), finite_float(item.get("entry_premium"), 0.0))
            premium += float(extra_slippage_per_side) * CONTRACT_MULTIPLIER
            unit_pnl = finite_float(item["pnl"], 0.0) - float(extra_slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER
            margin = max(finite_float(item.get("score"), 0.0) - finite_float(item.get("threshold"), 0.0), 0.0)
            confidence = margin / (margin + max(float(confidence_scale), 1e-9)) if margin > 0.0 else 0.0
            confidence = min(max(confidence, 0.0), 1.0)
            risk_frac_raw = min_risk_frac + (max_risk_frac - min_risk_frac) * confidence * confidence
            throttle = drawdown_multiplier(drawdown_frac)
            risk_frac = min(float(max_risk_frac), max(0.0, risk_frac_raw * throttle))
            hard_budget = max(equity, 0.0) * hard_premium_cap_frac
            risk_budget = min(max(equity, 0.0) * risk_frac, hard_budget)
            ask_size = finite_int(item.get("entry_ask_size"), default=1)
            liquidity_cap = max(1, int(math.floor(ask_size * liquidity_fraction))) if ask_size > 0 else 1
            max_contracts = max(1, min(int(absolute_max_contracts), liquidity_cap))
            raw_contracts = int(math.floor(risk_budget / premium)) if premium > 0.0 and risk_budget > 0.0 else 0

            skip_reason = ""
            if halted:
                contracts = 0
                skip_reason = "account_halted"
            elif premium <= 0.0 or not math.isfinite(premium):
                contracts = 0
                skip_reason = "invalid_premium"
            elif premium > hard_budget + 1e-9:
                contracts = 0
                skip_reason = "one_contract_exceeds_hard_premium_cap"
            else:
                contracts = min(max_contracts, raw_contracts)
                if contracts < 1:
                    contracts = 1
                if contracts * premium > hard_budget + 1e-9:
                    contracts = int(math.floor(hard_budget / premium))
                if contracts * premium > equity + 1e-9:
                    contracts = int(math.floor(equity / premium))
                if contracts <= 0:
                    contracts = 0
                    skip_reason = "unaffordable_after_caps"

            pnl = unit_pnl * contracts
            equity += pnl
            if equity <= starting_cash * ruin_stop_frac:
                halted = True
            peak = max(peak, equity)
            out = dict(item)
            out.update(
                {
                    "starting_cash": float(starting_cash),
                    "extra_slippage_per_side": float(extra_slippage_per_side),
                    "aa_contracts": int(contracts),
                    "aa_confidence": float(confidence),
                    "aa_margin": float(margin),
                    "aa_risk_frac_raw": float(risk_frac_raw),
                    "aa_drawdown_frac_before": float(drawdown_frac),
                    "aa_drawdown_multiplier": float(throttle),
                    "aa_risk_frac_after_throttle": float(risk_frac),
                    "aa_hard_premium_budget": float(hard_budget),
                    "aa_risk_budget": float(risk_budget),
                    "aa_entry_premium_after_stress": float(premium),
                    "aa_displayed_ask_size": int(ask_size),
                    "aa_liquidity_cap_contracts": int(liquidity_cap),
                    "aa_absolute_max_contracts": int(absolute_max_contracts),
                    "aa_max_contracts_after_caps": int(max_contracts),
                    "aa_raw_contracts": int(raw_contracts),
                    "aa_premium_at_risk": float(contracts * premium),
                    "aa_premium_frac_realized": float((contracts * premium) / before) if before > 0.0 else 0.0,
                    "aa_unit_pnl_after_stress": float(unit_pnl),
                    "aa_equity_before": float(before),
                    "aa_pnl": float(pnl),
                    "aa_equity_after": float(equity),
                    "aa_drawdown": float(equity - peak),
                    "aa_drawdown_frac_after": float(max(0.0, (peak - equity) / peak)) if peak > 0.0 else 1.0,
                    "aa_skip_reason": skip_reason,
                    "aa_account_halted_after": bool(halted),
                    "reported_split": split,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def drawdown_multiplier(drawdown_frac: float) -> float:
    if drawdown_frac <= 0.0:
        return 1.0
    if drawdown_frac >= 0.15:
        return 0.25
    return max(0.25, 1.0 - 0.75 * (drawdown_frac / 0.15))


def summarize_replay(replay: pd.DataFrame, *, protocol101: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in replay.groupby("reported_split", sort=True):
        gains = float(group.loc[group["aa_pnl"] > 0.0, "aa_pnl"].sum())
        losses = float(group.loc[group["aa_pnl"] < 0.0, "aa_pnl"].sum())
        sessions = group.groupby("session", sort=True)["aa_pnl"].sum()
        starting_cash = float(group["starting_cash"].iloc[0])
        slippage = float(group["extra_slippage_per_side"].iloc[0])
        total = float(group["aa_pnl"].sum())
        executed = group[group["aa_contracts"] > 0]
        rows.append(
            {
                "reported_split": split,
                "starting_cash": starting_cash,
                "extra_slippage_per_side": slippage,
                "trades": int(len(group)),
                "executed_trades": int(len(executed)),
                "skipped_trades": int((group["aa_contracts"] <= 0).sum()),
                "skip_hard_cap": int(group["aa_skip_reason"].eq("one_contract_exceeds_hard_premium_cap").sum()),
                "skip_halted": int(group["aa_skip_reason"].eq("account_halted").sum()),
                "total_pnl": total,
                "ending_equity": float(group["aa_equity_after"].iloc[-1]),
                "return_on_starting_cash": total / starting_cash if starting_cash > 0.0 else 0.0,
                "max_drawdown": float(group["aa_drawdown"].min()),
                "max_drawdown_pct": float(group["aa_drawdown_frac_after"].max()),
                "worst_day_pnl": float(sessions.min()) if not sessions.empty else 0.0,
                "best_day_pnl": float(sessions.max()) if not sessions.empty else 0.0,
                "profit_factor": gains / abs(losses) if losses < 0.0 else (999.0 if gains > 0.0 else 0.0),
                "win_rate_executed": float((executed["aa_pnl"] > 0.0).mean()) if not executed.empty else 0.0,
                "median_contracts": float(executed["aa_contracts"].median()) if not executed.empty else 0.0,
                "p90_contracts": float(executed["aa_contracts"].quantile(0.90)) if not executed.empty else 0.0,
                "max_contracts": int(executed["aa_contracts"].max()) if not executed.empty else 0,
                "median_premium_frac": float(executed["aa_premium_frac_realized"].median()) if not executed.empty else 0.0,
                "p90_premium_frac": float(executed["aa_premium_frac_realized"].quantile(0.90)) if not executed.empty else 0.0,
                "max_premium_frac": float(executed["aa_premium_frac_realized"].max()) if not executed.empty else 0.0,
                "liquidity_cap_binding_fraction": float((executed["aa_contracts"] >= executed["aa_liquidity_cap_contracts"]).mean()) if not executed.empty else 0.0,
                "raw_cap_binding_fraction": float((executed["aa_contracts"] < executed["aa_raw_contracts"]).mean()) if not executed.empty else 0.0,
                "halted": bool(group["aa_account_halted_after"].any()),
                "protocol101_total_pnl": float(protocol101.get(str(split), 0.0)),
                "delta_vs_protocol101": total - float(protocol101.get(str(split), 0.0)),
            }
        )
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame) -> str:
    base = summary[(summary["starting_cash"].eq(10_000.0)) & (summary["extra_slippage_per_side"].eq(0.0))]
    stress10 = summary[(summary["starting_cash"].eq(10_000.0)) & (summary["extra_slippage_per_side"].eq(0.10))]
    stress25 = summary[(summary["starting_cash"].eq(10_000.0)) & (summary["extra_slippage_per_side"].eq(0.25))]
    if not required_present(base) or not required_present(stress10) or not required_present(stress25):
        return "research_only_account_aware_sizing_missing_required_splits"
    if not all_positive(stress10) or not all_positive(stress25):
        return "research_only_account_aware_sizing_fails_slippage_stress"
    if bool(base["halted"].any()) or float(base["max_drawdown_pct"].max()) > 0.25:
        return "research_only_account_aware_sizing_drawdown_or_halt_blocker"
    if (base["delta_vs_protocol101"] > 0.0).all():
        return "account_aware_sizing_candidate_clears_initial_offline_gate"
    return "research_only_account_aware_sizing_improves_capital_efficiency_but_not_protocol101"


def required_present(frame: pd.DataFrame) -> bool:
    return set(REQUIRED_SPLITS).issubset({str(value) for value in frame["reported_split"].tolist()})


def all_positive(frame: pd.DataFrame) -> bool:
    required = frame[frame["reported_split"].astype(str).isin(REQUIRED_SPLITS)]
    return bool((required["total_pnl"] > 0.0).all())


def aggregate_for_json(summary: pd.DataFrame) -> list[dict[str, Any]]:
    keep = summary[summary["extra_slippage_per_side"].isin([0.0, 0.10, 0.25])]
    keep = keep[keep["starting_cash"].isin([10_000.0, 100_000.0, 1_000_000.0])]
    return keep.sort_values(["starting_cash", "extra_slippage_per_side", "reported_split"]).to_dict("records")


def risk_summary(rows: pd.DataFrame) -> dict[str, Any]:
    executed = rows[rows["aa_contracts"] > 0]
    if executed.empty:
        return {}
    return {
        "executed_rows": int(len(executed)),
        "skipped_rows": int((rows["aa_contracts"] <= 0).sum()),
        "median_contracts": float(executed["aa_contracts"].median()),
        "p90_contracts": float(executed["aa_contracts"].quantile(0.90)),
        "max_contracts": int(executed["aa_contracts"].max()),
        "median_premium_frac": float(executed["aa_premium_frac_realized"].median()),
        "p90_premium_frac": float(executed["aa_premium_frac_realized"].quantile(0.90)),
        "max_premium_frac": float(executed["aa_premium_frac_realized"].max()),
        "max_drawdown_pct": float(rows["aa_drawdown_frac_after"].max()),
    }


def write_report(path: Path, payload: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Entry/exit source: {payload['entry_exit_source']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Sizing Rule",
        "",
        f"- Min risk fraction: `{payload['sizing_rule']['min_risk_frac']}`",
        f"- Max risk fraction: `{payload['sizing_rule']['max_risk_frac']}`",
        f"- Hard premium cap fraction: `{payload['sizing_rule']['hard_premium_cap_frac']}`",
        f"- Liquidity fraction of displayed ask size: `{payload['sizing_rule']['liquidity_fraction']}`",
        f"- Absolute max contracts: `{payload['sizing_rule']['absolute_max_contracts']}`",
        f"- Ruin stop fraction: `{payload['sizing_rule']['ruin_stop_frac']}`",
        "",
        "## 10k Paper-Account Gate",
        "",
        "| split | stress | PnL | Protocol101 | delta | return | max DD | max DD % | worst day | executed/skipped | median contracts | p90 contracts | max contracts | p90 premium % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ten = summary[summary["starting_cash"].eq(10_000.0)].sort_values(["extra_slippage_per_side", "reported_split"])
    for row in ten.to_dict("records"):
        lines.append(summary_row(row))
    lines.extend(
        [
            "",
            "## Account-Scale Snapshot",
            "",
            "| cash | stress | total PnL | return | max DD % | max contracts | p90 contracts | skipped | halted |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    snap = (
        summary.groupby(["starting_cash", "extra_slippage_per_side"], as_index=False)
        .agg(
            total_pnl=("total_pnl", "sum"),
            return_on_starting_cash=("return_on_starting_cash", "sum"),
            max_drawdown_pct=("max_drawdown_pct", "max"),
            max_contracts=("max_contracts", "max"),
            p90_contracts=("p90_contracts", "max"),
            skipped_trades=("skipped_trades", "sum"),
            halted=("halted", "any"),
        )
        .sort_values(["starting_cash", "extra_slippage_per_side"])
    )
    for row in snap.to_dict("records"):
        lines.append(
            f"| ${row['starting_cash']:,.0f} | ${row['extra_slippage_per_side']:.2f} | {money(row['total_pnl'])} | "
            f"{pct(row['return_on_starting_cash'])} | {pct(row['max_drawdown_pct'])} | {int(row['max_contracts'])} | "
            f"{float(row['p90_contracts']):.1f} | {int(row['skipped_trades'])} | {bool(row['halted'])} |"
        )
    lines.extend(
        [
            "",
            "## Risk Summary",
            "",
            "```json",
            json.dumps(payload["risk_summary"], indent=2, sort_keys=True),
            "```",
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Sized trades: `{payload['outputs']['sized_trades']}`",
            f"- Split/balance/stress summary: `{payload['outputs']['split_balance_stress_summary']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def summary_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['reported_split']} | ${float(row['extra_slippage_per_side']):.2f} | {money(row['total_pnl'])} | "
        f"{money(row['protocol101_total_pnl'])} | {money(row['delta_vs_protocol101'])} | "
        f"{pct(row['return_on_starting_cash'])} | {money(row['max_drawdown'])} | {pct(row['max_drawdown_pct'])} | "
        f"{money(row['worst_day_pnl'])} | {int(row['executed_trades'])}/{int(row['skipped_trades'])} | "
        f"{float(row['median_contracts']):.1f} | {float(row['p90_contracts']):.1f} | {int(row['max_contracts'])} | "
        f"{pct(row['p90_premium_frac'])} |"
    )


def finite_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def finite_int(value: Any, *, default: int) -> int:
    number = finite_float(value, float(default))
    if number <= 0.0:
        return int(default)
    return int(math.floor(number))


def money(value: Any) -> str:
    number = finite_float(value, 0.0)
    sign = "-" if number < 0.0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    return f"{finite_float(value, 0.0) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
