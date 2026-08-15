"""EXP_2026_05_22_CONFIDENCE_SCALED_PREMIUM_SIZING_V1.

Historically Protocol222. This offline-only sizing experiment tests the user's
hypothesis that a return-on-premium model may deserve more than one contract
when it selects cheap contracts with high confidence.

It does not change entries, exits, or the paper default. It replays the frozen
Protocol221 selected trades with deterministic quantity sizing:

    confidence = clamp((score - threshold) / 1.0, 0, 1)
    premium_at_risk = equity * (10% + 25% * confidence)
    contracts = floor(premium_at_risk / entry_premium)

Quantity is additionally capped by displayed ask size when available and by a
current research max order size of 20 contracts. This prevents the replay from
pretending that exponential compounding can buy impossible option size.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_LABEL = "EXP_2026_05_22_CONFIDENCE_SCALED_PREMIUM_SIZING_V1"
HISTORICAL_ID = "Protocol222"
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_221_return_on_premium_full_action_policy/model_trades.csv")
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_222_confidence_scaled_premium_sizing")
STARTING_CASH = 10_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--base-risk-frac", type=float, default=0.10)
    parser.add_argument("--confidence-risk-frac", type=float, default=0.25)
    parser.add_argument("--confidence-scale", type=float, default=1.0)
    parser.add_argument("--max-order-contracts", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(args.trades, dataset_path=args.dataset)
    sized = replay_sized(
        trades,
        starting_cash=float(args.starting_cash),
        base_risk_frac=float(args.base_risk_frac),
        confidence_risk_frac=float(args.confidence_risk_frac),
        confidence_scale=float(args.confidence_scale),
        max_order_contracts=int(args.max_order_contracts),
        extra_slippage_per_side=0.0,
    )
    sized_10 = replay_sized(
        trades,
        starting_cash=float(args.starting_cash),
        base_risk_frac=float(args.base_risk_frac),
        confidence_risk_frac=float(args.confidence_risk_frac),
        confidence_scale=float(args.confidence_scale),
        max_order_contracts=int(args.max_order_contracts),
        extra_slippage_per_side=0.10,
    )
    sized_25 = replay_sized(
        trades,
        starting_cash=float(args.starting_cash),
        base_risk_frac=float(args.base_risk_frac),
        confidence_risk_frac=float(args.confidence_risk_frac),
        confidence_scale=float(args.confidence_scale),
        max_order_contracts=int(args.max_order_contracts),
        extra_slippage_per_side=0.25,
    )
    one_contract = replay_one_contract(trades, starting_cash=float(args.starting_cash))
    sized.to_csv(args.out_dir / "sized_trades.csv", index=False)
    sized_10.to_csv(args.out_dir / "sized_trades_stress_0_10.csv", index=False)
    sized_25.to_csv(args.out_dir / "sized_trades_stress_0_25.csv", index=False)
    one_contract.to_csv(args.out_dir / "one_contract_replay.csv", index=False)
    split_summary = summarize_by_split(sized, one_contract, stress_10=sized_10, stress_25=sized_25)
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / offline sizing simulator",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1_WITH_CONFIDENCE_SIZING",
        "entry_exit_source": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "sizing_rule": {
            "base_risk_frac": float(args.base_risk_frac),
            "confidence_risk_frac": float(args.confidence_risk_frac),
            "confidence_scale": float(args.confidence_scale),
            "max_order_contracts": int(args.max_order_contracts),
            "formula": "floor(equity * (base + confidence_extra * clamp((score-threshold)/scale,0,1)) / entry_premium)",
            "displayed_ask_size_cap": True,
        },
        "aggregate": split_summary.to_dict("records"),
        "risk_summary": risk_summary(sized),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
                "sized_trades": str(args.out_dir / "sized_trades.csv"),
                "sized_trades_stress_0_10": str(args.out_dir / "sized_trades_stress_0_10.csv"),
                "sized_trades_stress_0_25": str(args.out_dir / "sized_trades_stress_0_25.csv"),
            "one_contract_replay": str(args.out_dir / "one_contract_replay.csv"),
            "split_summary": str(args.out_dir / "split_summary.csv"),
        },
        "decision": decide(split_summary),
        "next_experiment": "If confidence sizing improves dollars without unacceptable drawdown, compare it against the ITM dollar challenger; otherwise test a blended dollar-plus-premium utility before more sizing.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path, *, dataset_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if dataset_path.exists() and "entry_ask_size" not in frame.columns:
        sizes = pd.read_parquet(dataset_path, columns=["candidate_uid", "entry_ask_size"])
        sizes = sizes.drop_duplicates("candidate_uid")
        frame = frame.merge(sizes, on="candidate_uid", how="left", validate="many_to_one")
    for column in ["decision_time", "exit_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ["score", "threshold", "entry_premium", "entry_premium_with_slippage", "entry_ask_size", "pnl"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["decision_time", "exit_time", "entry_premium", "pnl"]).sort_values(["reported_split", "decision_time", "exit_time", "candidate_uid"]).reset_index(drop=True)


def replay_sized(
    trades: pd.DataFrame,
    *,
    starting_cash: float,
    base_risk_frac: float,
    confidence_risk_frac: float,
    confidence_scale: float,
    max_order_contracts: int,
    extra_slippage_per_side: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in trades.groupby("reported_split", sort=True):
        equity = float(starting_cash)
        peak = equity
        for item in group.sort_values(["decision_time", "exit_time", "candidate_uid"]).to_dict("records"):
            premium = float(item.get("entry_premium_with_slippage", item.get("entry_premium", 0.0)) or 0.0)
            margin = max(float(item.get("score", 0.0)) - float(item.get("threshold", 0.0)), 0.0)
            confidence = min(max(margin / max(confidence_scale, 1e-9), 0.0), 1.0)
            risk_frac = base_risk_frac + confidence_risk_frac * confidence
            premium_budget = max(equity, 0.0) * risk_frac
            raw_contracts = int(math.floor(premium_budget / premium)) if premium > 0.0 and equity > 0.0 else 0
            ask_size = finite_int(item.get("entry_ask_size"), default=max_order_contracts)
            max_contracts = max(1, min(int(max_order_contracts), ask_size))
            contracts = min(raw_contracts, max_contracts)
            contracts = max(contracts, 1) if premium <= equity and contracts < 1 else contracts
            affordable = contracts > 0 and contracts * premium <= equity + 1e-9
            stressed_unit_pnl = float(item["pnl"]) - float(extra_slippage_per_side) * 2.0 * 100.0
            pnl = stressed_unit_pnl * contracts if affordable else 0.0
            before = equity
            equity += pnl
            peak = max(peak, equity)
            out = dict(item)
            out.update(
                {
                    "sized_contracts": contracts if affordable else 0,
                    "sized_confidence": confidence,
                    "sized_risk_frac": risk_frac,
                    "sized_premium_budget": premium_budget,
                    "sized_premium_at_risk": contracts * premium if affordable else 0.0,
                    "sized_raw_contracts": raw_contracts,
                    "sized_displayed_ask_size": ask_size,
                    "sized_max_contracts_after_caps": max_contracts,
                    "sized_equity_before": before,
                    "sized_pnl": pnl,
                    "sized_unit_pnl_after_stress": stressed_unit_pnl,
                    "sized_extra_slippage_per_side": float(extra_slippage_per_side),
                    "sized_equity_after": equity,
                    "sized_drawdown": equity - peak,
                    "sized_skip_reason": "" if affordable else "unaffordable",
                    "reported_split": split,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def replay_one_contract(trades: pd.DataFrame, *, starting_cash: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in trades.groupby("reported_split", sort=True):
        equity = float(starting_cash)
        peak = equity
        for item in group.sort_values(["decision_time", "exit_time", "candidate_uid"]).to_dict("records"):
            before = equity
            equity += float(item["pnl"])
            peak = max(peak, equity)
            out = dict(item)
            out.update(
                {
                    "one_contract_equity_before": before,
                    "one_contract_pnl": float(item["pnl"]),
                    "one_contract_equity_after": equity,
                    "one_contract_drawdown": equity - peak,
                    "reported_split": split,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_by_split(
    sized: pd.DataFrame,
    one_contract: pd.DataFrame,
    *,
    stress_10: pd.DataFrame,
    stress_25: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split, group in sized.groupby("reported_split", sort=True):
        one = one_contract[one_contract["reported_split"].astype(str).eq(str(split))]
        stress10 = stress_10[stress_10["reported_split"].astype(str).eq(str(split))]
        stress25 = stress_25[stress_25["reported_split"].astype(str).eq(str(split))]
        total_pnl = float(group["sized_pnl"].sum())
        one_pnl = float(one["one_contract_pnl"].sum())
        losses = group.loc[group["sized_pnl"] < 0, "sized_pnl"].sum()
        gains = group.loc[group["sized_pnl"] > 0, "sized_pnl"].sum()
        rows.append(
            {
                "reported_split": split,
                "trades": int(len(group)),
                "sized_total_pnl": total_pnl,
                "sized_stress_0_10_total_pnl": float(stress10["sized_pnl"].sum()),
                "sized_stress_0_25_total_pnl": float(stress25["sized_pnl"].sum()),
                "one_contract_total_pnl": one_pnl,
                "delta_vs_one_contract": total_pnl - one_pnl,
                "sized_ending_equity": float(group["sized_equity_after"].iloc[-1]),
                "one_contract_ending_equity": float(one["one_contract_equity_after"].iloc[-1]),
                "sized_max_drawdown": float(group["sized_drawdown"].min()),
                "one_contract_max_drawdown": float(one["one_contract_drawdown"].min()),
                "sized_profit_factor": float(gains / abs(losses)) if losses < 0 else (999.0 if gains > 0 else 0.0),
                "sized_win_rate": float((group["sized_pnl"] > 0).mean()),
                "median_contracts": float(group["sized_contracts"].median()),
                "p90_contracts": float(group["sized_contracts"].quantile(0.90)),
                "max_contracts": int(group["sized_contracts"].max()),
                "median_premium_at_risk": float(group["sized_premium_at_risk"].median()),
                "p90_premium_at_risk": float(group["sized_premium_at_risk"].quantile(0.90)),
                "max_premium_at_risk": float(group["sized_premium_at_risk"].max()),
                "median_displayed_ask_size": float(group["sized_displayed_ask_size"].median()),
                "cap_binding_fraction": float((group["sized_contracts"] < group["sized_raw_contracts"]).mean()),
                "skipped": int((group["sized_contracts"] <= 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def risk_summary(sized: pd.DataFrame) -> dict[str, Any]:
    if sized.empty:
        return {}
    return {
        "median_contracts": float(sized["sized_contracts"].median()),
        "p90_contracts": float(sized["sized_contracts"].quantile(0.90)),
        "max_contracts": int(sized["sized_contracts"].max()),
        "median_premium_at_risk": float(sized["sized_premium_at_risk"].median()),
        "p90_premium_at_risk": float(sized["sized_premium_at_risk"].quantile(0.90)),
        "max_premium_at_risk": float(sized["sized_premium_at_risk"].max()),
        "max_risk_frac_realized": float((sized["sized_premium_at_risk"] / sized["sized_equity_before"].replace(0, np.nan)).max()),
        "cap_binding_fraction": float((sized["sized_contracts"] < sized["sized_raw_contracts"]).mean()),
    }


def decide(summary: pd.DataFrame) -> str:
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    available = {str(value) for value in summary["reported_split"]}
    if not set(required).issubset(available):
        return "research_only_confidence_sizing_missing_required_splits"
    checks = []
    for split in required:
        row = summary[summary["reported_split"].astype(str).eq(split)].iloc[0]
        checks.append(float(row["sized_total_pnl"]) > 0.0)
        checks.append(float(row["sized_stress_0_10_total_pnl"]) > 0.0)
        checks.append(float(row["sized_total_pnl"]) > float(row["one_contract_total_pnl"]))
        checks.append(float(row["sized_ending_equity"]) > 0.0)
    if all(checks):
        return "confidence_sizing_improves_return_on_premium_candidate_offline"
    return "research_only_confidence_sizing_not_consistently_better"


def write_report(path: Path, payload: dict[str, Any]) -> None:
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
        f"- Formula: `{payload['sizing_rule']['formula']}`",
        f"- Base risk fraction: `{payload['sizing_rule']['base_risk_frac']}`",
        f"- Confidence risk fraction: `{payload['sizing_rule']['confidence_risk_frac']}`",
        f"- Max order contracts for current research gate: `{payload['sizing_rule']['max_order_contracts']}`",
        f"- Displayed ask-size cap: `{payload['sizing_rule']['displayed_ask_size_cap']}`",
        "",
        "## Split Results",
        "",
        "| split | trades | sized pnl | stress 0.10 | stress 0.25 | one-contract pnl | delta | sized max DD | one max DD | median contracts | p90 contracts | max contracts | cap binding | median premium risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['reported_split']} | {row['trades']} | ${float(row['sized_total_pnl']):,.0f} | "
            f"${float(row['sized_stress_0_10_total_pnl']):,.0f} | ${float(row['sized_stress_0_25_total_pnl']):,.0f} | "
            f"${float(row['one_contract_total_pnl']):,.0f} | ${float(row['delta_vs_one_contract']):,.0f} | "
            f"${float(row['sized_max_drawdown']):,.0f} | ${float(row['one_contract_max_drawdown']):,.0f} | "
            f"{float(row['median_contracts']):.1f} | {float(row['p90_contracts']):.1f} | {int(row['max_contracts'])} | "
            f"{float(row['cap_binding_fraction']) * 100:.1f}% | "
            f"${float(row['median_premium_at_risk']):,.0f} |"
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
            f"- Split summary: `{payload['outputs']['split_summary']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def finite_int(value: Any, *, default: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return int(default)
    if not math.isfinite(number) or number <= 0:
        return int(default)
    return int(math.floor(number))


if __name__ == "__main__":
    raise SystemExit(main())
