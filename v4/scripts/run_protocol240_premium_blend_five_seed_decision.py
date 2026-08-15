"""DECISION_2026_05_22_PREMIUM_LEANING_BLEND_FIVE_SEED_V1.

Historically Protocol240. This decision packet combines the 3-seed and 2-seed
screens for the same premium-leaning blended utility challenger. It does not
train a model, download data, call a broker, or change the paper default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol165_full_action_space_policy import fmt


ROLE_LABEL = "DECISION_2026_05_22_PREMIUM_LEANING_BLEND_FIVE_SEED_V1"
HISTORICAL_ID = "Protocol240"
DEFAULT_SUMMARIES = [
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_238_premium_leaning_blend_seed_stability/summary.json"),
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_239_premium_leaning_blend_seed_holdout_4_5/summary.json"),
]
DEFAULT_TRADE_FILES = [
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_238_premium_leaning_blend_seed_stability/model_trades.csv"),
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_239_premium_leaning_blend_seed_holdout_4_5/model_trades.csv"),
]
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision")
REQUIRED_SPLITS = ("q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", dest="summaries", type=Path, action="append", default=[])
    parser.add_argument("--trades", dest="trade_files", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = args.summaries or DEFAULT_SUMMARIES
    trade_files = args.trade_files or DEFAULT_TRADE_FILES
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payloads = [json.loads(path.read_text()) for path in summaries]
    rows = per_seed_rows(payloads)
    trades = pd.concat([pd.read_csv(path) for path in trade_files], ignore_index=True)
    rows.to_csv(args.out_dir / "five_seed_split_rows.csv", index=False)
    trades.to_csv(args.out_dir / "five_seed_model_trades.csv", index=False)

    aggregate = aggregate_rows(rows)
    decision = decide(aggregate)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "freeze / promotion decision packet",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_to_beat": "Strict one-account serial Protocol101 replay",
        "data_used": sorted({payload.get("data_used", "") for payload in payloads}),
        "source_summaries": [str(path) for path in summaries],
        "source_trade_files": [str(path) for path in trade_files],
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "utility": payloads[0].get("utility", {}),
        "aggregate": aggregate,
        "trade_profile": trade_profile(trades),
        "decision": decision,
        "next_experiment": (
            "Build no-order runtime parity for this full-action feature set and run a trade-level attribution "
            "against Protocol101 before any paper-default replacement."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def per_seed_rows(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        baselines = payload.get("frozen_protocol101_baselines", {})
        for result in payload.get("fold_results", []):
            if result.get("skipped"):
                continue
            seed = int(result["seed"])
            for split, split_payload in result.get("splits", {}).items():
                model = split_payload["model"]
                stress10 = split_payload.get("model_stress_0_10", {})
                stress25 = split_payload.get("model_stress_0_25", {})
                baseline = float(baselines.get(split, payload.get("aggregate", {}).get(split, {}).get("frozen_protocol101_total_pnl", 0.0)))
                rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "total_pnl": float(model.get("total_pnl", 0.0)),
                        "protocol101_total_pnl": baseline,
                        "delta_vs_protocol101": float(model.get("total_pnl", 0.0)) - baseline,
                        "profit_factor": float(model.get("profit_factor", 0.0)),
                        "stress_0_10_total_pnl": float(stress10.get("total_pnl", 0.0)),
                        "stress_0_25_total_pnl": float(stress25.get("total_pnl", 0.0)),
                        "trades": int(model.get("trades", 0)),
                        "pnl_per_premium": float(model.get("pnl_per_premium", 0.0)),
                        "median_entry_premium": float(model.get("median_entry_premium", 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_rows(rows: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for split, group in rows.groupby("split", sort=True):
        item = {
            "seeds": int(group["seed"].nunique()),
            "median_total_pnl": float(group["total_pnl"].median()),
            "protocol101_total_pnl": float(group["protocol101_total_pnl"].median()),
            "median_delta_vs_protocol101": float(group["delta_vs_protocol101"].median()),
            "positive_seed_fraction": float((group["total_pnl"] > 0).mean()),
            "beats_protocol101_seed_fraction": float((group["delta_vs_protocol101"] > 0).mean()),
            "median_profit_factor": float(group["profit_factor"].median()),
            "median_stress_0_10_total_pnl": float(group["stress_0_10_total_pnl"].median()),
            "median_stress_0_25_total_pnl": float(group["stress_0_25_total_pnl"].median()),
            "median_trades": float(group["trades"].median()),
            "median_pnl_per_premium": float(group["pnl_per_premium"].median()),
            "median_entry_premium": float(group["median_entry_premium"].median()),
            "seed_pnls": {str(int(row.seed)): float(row.total_pnl) for row in group.itertuples(index=False)},
        }
        out[str(split)] = item
        if split in REQUIRED_SPLITS:
            checks.extend(
                [
                    {"split": split, "name": "five_seeds_available", "pass": item["seeds"] == 5, "value": item["seeds"]},
                    {"split": split, "name": "positive_median_pnl", "pass": item["median_total_pnl"] > 0.0, "value": item["median_total_pnl"]},
                    {"split": split, "name": "positive_seed_fraction_ge_0_80", "pass": item["positive_seed_fraction"] >= 0.80, "value": item["positive_seed_fraction"]},
                    {"split": split, "name": "beats_protocol101_median", "pass": item["median_delta_vs_protocol101"] > 0.0, "value": item["median_delta_vs_protocol101"]},
                    {"split": split, "name": "beats_protocol101_seed_fraction_ge_0_80", "pass": item["beats_protocol101_seed_fraction"] >= 0.80, "value": item["beats_protocol101_seed_fraction"]},
                    {"split": split, "name": "median_pf_ge_1_15", "pass": item["median_profit_factor"] >= 1.15, "value": item["median_profit_factor"]},
                    {"split": split, "name": "stress_0_10_positive", "pass": item["median_stress_0_10_total_pnl"] > 0.0, "value": item["median_stress_0_10_total_pnl"]},
                    {"split": split, "name": "stress_0_25_positive", "pass": item["median_stress_0_25_total_pnl"] > 0.0, "value": item["median_stress_0_25_total_pnl"]},
                ]
            )
    out["promotion_checks"] = checks
    out["research_challenger_ready"] = bool(checks and all(check["pass"] for check in checks))
    return out


def trade_profile(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"rows": 0}
    by_offset = (
        trades.assign(offset_bucket=pd.cut(pd.to_numeric(trades["offset"], errors="coerce"), bins=[-999, -25, -5, 5, 25, 999], labels=["far_put_side", "near_put_side", "atm_band", "near_call_side", "far_call_side"]))
        .groupby("offset_bucket", observed=False)
        .agg(trades=("pnl", "size"), pnl=("pnl", "sum"), median_premium=("entry_premium", "median"))
        .reset_index()
    )
    by_right = trades.groupby("right").agg(trades=("pnl", "size"), pnl=("pnl", "sum"), median_premium=("entry_premium", "median")).reset_index()
    return {
        "rows": int(len(trades)),
        "total_pnl": float(trades["pnl"].sum()),
        "median_entry_premium": float(trades["entry_premium"].median()),
        "pnl_per_entry_premium": float(trades["pnl"].sum() / trades["entry_premium"].sum()) if float(trades["entry_premium"].sum()) else 0.0,
        "by_offset_bucket": by_offset.to_dict("records"),
        "by_right": by_right.to_dict("records"),
    }


def decide(aggregate: dict[str, Any]) -> str:
    if aggregate.get("research_challenger_ready"):
        return "freeze_research_challenger_premium_leaning_blend_five_seed_survives"
    return "do_not_freeze_premium_leaning_blend_five_seed_gate_failed"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Baseline it must beat: {payload['baseline_to_beat']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Five-Seed Gate",
        "",
        "| split | seeds | median PnL | Protocol101 | delta | beat seed frac | PF | stress 0.10 | stress 0.25 | pnl/premium | median premium | trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in REQUIRED_SPLITS:
        item = payload["aggregate"].get(split, {})
        if not item:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | {fmt(item['protocol101_total_pnl'])} | "
            f"{fmt(item['median_delta_vs_protocol101'])} | {item['beats_protocol101_seed_fraction']:.2f} | "
            f"{fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} | "
            f"{fmt(item['median_stress_0_25_total_pnl'])} | {item['median_pnl_per_premium']:.4f} | "
            f"{fmt(item['median_entry_premium'])} | {fmt(item['median_trades'])} |"
        )
    lines.extend(["", "## Trade Profile", ""])
    profile = payload["trade_profile"]
    lines.append(f"- Combined rows: {profile['rows']}")
    lines.append(f"- Combined PnL: {fmt(profile['total_pnl'])}")
    lines.append(f"- Median entry premium: {fmt(profile['median_entry_premium'])}")
    lines.append(f"- PnL per entry premium: {profile['pnl_per_entry_premium']:.4f}")
    lines.append("")
    lines.append("Side profile:")
    for row in profile["by_right"]:
        lines.append(f"- {row['right']}: {int(row['trades'])} trades, PnL {fmt(row['pnl'])}, median premium {fmt(row['median_premium'])}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Five-seed split rows: `{path.parent / 'five_seed_split_rows.csv'}`",
            f"- Five-seed model trades: `{path.parent / 'five_seed_model_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
