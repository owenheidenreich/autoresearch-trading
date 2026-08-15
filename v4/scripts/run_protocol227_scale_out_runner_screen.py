"""EXP_2026_05_22_SCALE_OUT_RUNNER_SCREEN_V1.

Historically Protocol227. Protocol226 showed that blindly extending whole
positions after the baseline exit does not improve the account-aware stream.
This experiment tests a more trader-like scale-out idea: close most contracts
at the baseline exit, leave a small runner only when the account-aware quantity
is large enough, and force the runner flat before close.

This is not a final model. It is a validation-selected lifecycle structure
screen to decide whether scale-out runners deserve a neural action model.
No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v4.scripts.run_protocol225_account_aware_lifecycle_exit_policy import (
    DEFAULT_DATASET,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_TRADES,
    REQUIRED_SPLITS,
    STARTING_CASH,
    build_path_records,
    compare_summaries,
    compute_quantity,
    count_by,
    fold_specs,
    load_entries,
    metrics_for_rows,
    serial_invariants,
    simulate_baseline_serial,
    summarize_replay,
)


ROLE_LABEL = "EXP_2026_05_22_SCALE_OUT_RUNNER_SCREEN_V1"
HISTORICAL_ID = "Protocol227"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_227_scale_out_runner_screen")
RUNNER_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:30")
    parser.add_argument("--min-risk-frac", type=float, default=0.01)
    parser.add_argument("--max-risk-frac", type=float, default=0.10)
    parser.add_argument("--hard-premium-cap-frac", type=float, default=0.12)
    parser.add_argument("--confidence-scale", type=float, default=0.75)
    parser.add_argument("--liquidity-fraction", type=float, default=0.25)
    parser.add_argument("--absolute-max-contracts", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.trades, dataset_path=args.dataset)
    records, skips = build_path_records(entries, normalized_dir=args.normalized_dir, forced_flat_time=str(args.forced_flat_time))
    sizing = {
        "min_risk_frac": float(args.min_risk_frac),
        "max_risk_frac": float(args.max_risk_frac),
        "hard_premium_cap_frac": float(args.hard_premium_cap_frac),
        "confidence_scale": float(args.confidence_scale),
        "liquidity_fraction": float(args.liquidity_fraction),
        "absolute_max_contracts": int(args.absolute_max_contracts),
    }
    baseline_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    for spec in fold_specs():
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not validation_records:
            continue
        runner_fraction, sweep = select_runner_fraction(validation_records, sizing=sizing, split_name=str(spec["validation_split"]))
        sweep_rows.extend({**row, "fold": spec["fold"]} for row in sweep)
        for split in spec["test_splits"]:
            split_records = [r for r in records if r.reported_split == split]
            model_rows.extend(
                simulate_runner_serial(
                    split_records,
                    sizing=sizing,
                    runner_fraction=runner_fraction,
                    strategy=f"protocol227:{spec['fold']}:runner{runner_fraction}",
                )
            )
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                model_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_runner_serial(
                        march_records,
                        sizing=sizing,
                        runner_fraction=runner_fraction,
                        strategy=f"protocol227:{spec['fold']}:runner{runner_fraction}:march_subset",
                    )
                )
        fold_payloads.append(
            {
                "fold": spec["fold"],
                "validation_split": spec["validation_split"],
                "test_splits": list(spec["test_splits"]),
                "selected_runner_fraction": float(runner_fraction),
                "validation_records": len(validation_records),
            }
        )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, sizing=sizing, strategy="protocol223_account_aware_baseline"))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(
                        march_records,
                        sizing=sizing,
                        strategy="protocol223_account_aware_baseline:march_subset",
                    )
                )
    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    sweep_frame = pd.DataFrame(sweep_rows)
    baseline_summary = summarize_replay(baseline_frame, seed_col=None)
    model_summary = summarize_replay(model_frame, seed_col=None)
    comparison = compare_summaries(model_summary, baseline_summary)
    invariants = serial_invariants(model_frame.assign(model_seed=0))
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / validation-selected scale-out runner screen",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_SCALE_OUT_RUNNER_V1",
        "entry_source": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "sizing_source": "EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "pre_registration": {
            "hypothesis": "A small runner after the baseline exit may capture continuation without risking the full position.",
            "candidate_runner_fractions": list(RUNNER_FRACTIONS),
            "runner_exit": "mandatory forced-flat time from the same historical/live contract",
            "threshold_selection": "runner fraction selected only on validation split",
            "no_hindsight_exit": True,
        },
        "row_counts": {
            "entries": int(len(entries)),
            "path_records": int(len(records)),
            "path_skips": int(len(skips)),
            "baseline_trade_rows": int(len(baseline_frame)),
            "model_trade_rows": int(len(model_frame)),
        },
        "folds": fold_payloads,
        "baseline_summary": baseline_summary,
        "model_summary": model_summary,
        "comparison": comparison,
        "runner_sweep": sweep_frame.to_dict("records"),
        "invariants": invariants,
        "path_skip_counts": count_by(skips, "skip_reason"),
        "decision": decide(comparison, invariants),
        "next_experiment": next_experiment(comparison, sweep_frame),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "model_trades": str(args.out_dir / "protocol227_model_serial_trades.csv"),
            "baseline_trades": str(args.out_dir / "protocol223_baseline_serial_trades.csv"),
            "runner_sweep": str(args.out_dir / "runner_fraction_sweep.csv"),
            "path_skips": str(args.out_dir / "path_skips.csv"),
        },
    }
    baseline_frame.to_csv(args.out_dir / "protocol223_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol227_model_serial_trades.csv", index=False)
    sweep_frame.to_csv(args.out_dir / "runner_fraction_sweep.csv", index=False)
    pd.DataFrame(skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def select_runner_fraction(records: Sequence[Any], *, sizing: dict[str, Any], split_name: str) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_fraction = 0.0
    best_key = (-1e18, -1e18)
    for fraction in RUNNER_FRACTIONS:
        trades = simulate_runner_serial(records, sizing=sizing, runner_fraction=float(fraction), strategy="validation")
        metrics = metrics_for_rows(pd.DataFrame(trades))
        key = (float(metrics["total_pnl"]), float(metrics["profit_factor_for_selection"]))
        rows.append({"validation_split": split_name, "runner_fraction": float(fraction), **metrics})
        if key > best_key:
            best_key = key
            best_fraction = float(fraction)
    return best_fraction, rows


def simulate_runner_serial(
    records: Sequence[Any],
    *,
    sizing: dict[str, Any],
    runner_fraction: float,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity = STARTING_CASH
    peak = equity
    open_until_by_session: dict[str, pd.Timestamp] = {}
    for record in sorted(records, key=lambda r: (r.session, r.decision_ts, r.contract_id)):
        if record.decision_ts < open_until_by_session.get(record.session, pd.Timestamp.min.tz_localize("UTC")):
            continue
        quantity, sizing_info = compute_quantity(record, equity=equity, peak=peak, sizing=sizing)
        if quantity <= 0:
            continue
        runner_qty = int(math.floor(quantity * float(runner_fraction)))
        close_qty = quantity - runner_qty
        runner_unit_pnl = float(record.unit_pnl_path[-1]) if runner_qty > 0 else 0.0
        baseline_unit_pnl = float(record.baseline_unit_pnl)
        pnl = close_qty * baseline_unit_pnl + runner_qty * runner_unit_pnl
        exit_ts = pd.Timestamp(record.quote_times[-1]) if runner_qty > 0 else pd.Timestamp(record.baseline_exit_ts)
        before = equity
        equity += pnl
        peak = max(peak, equity)
        rows.append(
            {
                "reported_split": record.reported_split,
                "fold": record.fold,
                "model_seed": 0,
                "session": record.session,
                "decision_time": record.decision_ts.isoformat(),
                "exit_time": exit_ts.isoformat(),
                "contract_id": record.contract_id,
                "right": record.right,
                "offset": float(record.offset),
                "score": float(record.score),
                "threshold": math.nan,
                "entry_ask": float(record.entry_ask),
                "entry_premium": float(record.entry_premium),
                "quantity": int(quantity),
                "close_qty_at_baseline": int(close_qty),
                "runner_qty": int(runner_qty),
                "runner_fraction": float(runner_fraction),
                "premium_at_risk": float(quantity * record.entry_premium),
                "premium_frac": float((quantity * record.entry_premium) / before) if before > 0.0 else 0.0,
                "unit_pnl": float(pnl / max(quantity, 1)),
                "baseline_unit_pnl": baseline_unit_pnl,
                "runner_unit_pnl": runner_unit_pnl,
                "pnl": float(pnl),
                "account_equity_before": float(before),
                "account_equity_after": float(equity),
                "exit_reason": "baseline_with_runner" if runner_qty > 0 else "baseline_no_runner",
                "sizing_confidence": float(sizing_info.get("confidence", math.nan)),
                "sizing_risk_frac": float(sizing_info.get("risk_frac", math.nan)),
                "sizing_drawdown_frac": float(sizing_info.get("drawdown_frac", math.nan)),
                "strategy": strategy,
            }
        )
        open_until_by_session[record.session] = exit_ts
    return rows


def decide(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_scale_out_runner_invariant_failure"
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "keep_scale_out_runner_research_candidate"
    if any(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "mixed_scale_out_runner_requires_attribution"
    return "reject_scale_out_runner_no_improvement"


def next_experiment(comparison: list[dict[str, Any]], sweep: pd.DataFrame) -> str:
    del comparison
    if not sweep.empty and (sweep["runner_fraction"].astype(float) > 0.0).any():
        return "If runner fractions helped validation but failed tests, attribute runner losses by side/time and test learned reduce/exit instead of forced-flat runners."
    return "Runner screen selected no runner or did not generalize; move to a learned pre-baseline scale-out/reduce classifier."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Entry source: {payload['entry_source']}",
        f"Sizing source: {payload['sizing_source']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Comparison To Account-Aware Baseline",
        "",
        "| split | model PnL | baseline PnL | delta | model PF | baseline PF | model trades | baseline trades | model DD | baseline DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | {money(row['baseline_median_total_pnl'])} | "
            f"{money(row['delta_vs_baseline'])} | {row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{money(row['model_median_max_drawdown'])} | {money(row['baseline_median_max_drawdown'])} |"
        )
    lines.extend(["", "## Runner Sweep", ""])
    if payload["runner_sweep"]:
        lines.extend(["| fold | validation | runner | PnL | PF | trades |", "|---|---|---:|---:|---:|---:|"])
        for row in payload["runner_sweep"]:
            lines.append(
                f"| {row['fold']} | {row['validation_split']} | {float(row['runner_fraction']):.2f} | "
                f"{money(row['total_pnl'])} | {float(row['profit_factor']):.3f} | {int(row['trades'])} |"
            )
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            f"- Overlap violations: `{payload['invariants']['overlap_violations']}`",
            f"- Unaffordable violations: `{payload['invariants']['unaffordable_violations']}`",
            f"- NaN time rows: `{payload['invariants']['nan_time_rows']}`",
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Model trades: `{payload['outputs']['model_trades']}`",
            f"- Baseline trades: `{payload['outputs']['baseline_trades']}`",
            f"- Runner sweep: `{payload['outputs']['runner_sweep']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    sign = "-" if number < 0.0 else ""
    return f"{sign}${abs(number):,.0f}"


if __name__ == "__main__":
    raise SystemExit(main())
