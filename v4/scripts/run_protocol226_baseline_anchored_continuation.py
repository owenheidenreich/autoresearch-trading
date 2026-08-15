"""EXP_2026_05_22_BASELINE_ANCHORED_CONTINUATION_V1.

Historically Protocol226. Protocol225 failed by exiting too early. This
experiment keeps the same account-aware entry and sizing stream, but treats the
frozen baseline exit as a safety anchor. The neural model may extend a trade
past the baseline exit when continuation value is predicted, but it cannot
prematurely scalp out before that anchor.

This is a conservative lifecycle test, not a final unified policy. No paid data
is downloaded and no broker endpoint is called.
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
    THRESHOLD_CANDIDATES,
    build_path_records,
    compare_summaries,
    compute_quantity,
    count_by,
    fit_training_matrix,
    fold_specs,
    load_entries,
    metrics_for_rows,
    predict_records,
    row_for_trade,
    serial_invariants,
    simulate_baseline_serial,
    summarize_replay,
    threshold_summary,
    train_model,
    write_report as write_protocol225_style_report,
)


ROLE_LABEL = "EXP_2026_05_22_BASELINE_ANCHORED_CONTINUATION_V1"
HISTORICAL_ID = "Protocol226"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_226_baseline_anchored_continuation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:30")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
    threshold_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_x, train_y, scaler = fit_training_matrix(train_records, max_train_steps=int(args.max_train_steps), seed=226)
        for seed in args.seeds:
            model, history = train_model(
                train_x,
                train_y,
                seed=int(seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                learning_rate=float(args.learning_rate),
            )
            validation_predictions = predict_records(model, scaler, validation_records)
            threshold, sweep = select_threshold(
                validation_records,
                validation_predictions,
                sizing=sizing,
                split_name=str(spec["validation_split"]),
                model_seed=int(seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                predictions = predict_records(model, scaler, split_records)
                model_rows.extend(
                    simulate_anchor_serial(
                        split_records,
                        predictions,
                        sizing=sizing,
                        threshold=threshold,
                        model_seed=int(seed),
                        strategy=f"protocol226:{spec['fold']}:seed{seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_predictions = {r.uid: predictions[r.uid] for r in march_records if r.uid in predictions}
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_anchor_serial(
                            march_records,
                            march_predictions,
                            sizing=sizing,
                            threshold=threshold,
                            model_seed=int(seed),
                            strategy=f"protocol226:{spec['fold']}:seed{seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(seed),
                    "threshold": float(threshold),
                    "history": history,
                    "train_records": int(len(train_records)),
                    "validation_records": int(len(validation_records)),
                    "train_steps_used": int(len(train_y)),
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
    threshold_frame = pd.DataFrame(threshold_rows)
    baseline_summary = summarize_replay(baseline_frame, seed_col=None)
    model_summary = summarize_replay(model_frame, seed_col="model_seed")
    comparison = compare_summaries(model_summary, baseline_summary)
    invariants = serial_invariants(model_frame)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / baseline-anchored causal continuation model",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_BASELINE_ANCHORED_CONTINUATION_V1",
        "entry_source": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "sizing_source": "EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "pre_registration": {
            "hypothesis": "Use the baseline exit as a safety anchor, then learn continuation past it without enabling premature scalp exits.",
            "flat_entries": "frozen Protocol221 trade stream",
            "holding_action_space": "baseline exit or model-extended hold",
            "sizing": sizing,
            "threshold_selection": "validation split only",
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
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(skips, "skip_reason"),
        "decision": decide(comparison, invariants),
        "next_experiment": next_experiment(comparison, invariants),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "model_trades": str(args.out_dir / "protocol226_model_serial_trades.csv"),
            "baseline_trades": str(args.out_dir / "protocol223_baseline_serial_trades.csv"),
            "threshold_sweep": str(args.out_dir / "threshold_sweep.csv"),
            "path_skips": str(args.out_dir / "path_skips.csv"),
        },
    }
    baseline_frame.to_csv(args.out_dir / "protocol223_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol226_model_serial_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def select_threshold(
    records: Sequence[Any],
    predictions: dict[str, np.ndarray],
    *,
    sizing: dict[str, Any],
    split_name: str,
    model_seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = float(THRESHOLD_CANDIDATES[0])
    best_key = (-1e18, -1e18, 0.0)
    for threshold in THRESHOLD_CANDIDATES:
        trades = simulate_anchor_serial(
            records,
            predictions,
            sizing=sizing,
            threshold=float(threshold),
            model_seed=model_seed,
            strategy="threshold_selection",
        )
        metrics = metrics_for_rows(pd.DataFrame(trades))
        key = (float(metrics["total_pnl"]), float(metrics["profit_factor_for_selection"]), -float(metrics["trades"]))
        rows.append({"validation_split": split_name, "threshold": float(threshold), **metrics})
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, rows


def simulate_anchor_serial(
    records: Sequence[Any],
    predictions: dict[str, np.ndarray],
    *,
    sizing: dict[str, Any],
    threshold: float,
    model_seed: int,
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
        pred = predictions.get(record.uid)
        if pred is None or len(pred) != len(record.unit_pnl_path):
            continue
        exit_idx = anchored_exit_index(record, pred, threshold)
        anchor_idx = baseline_anchor_idx(record)
        extended = exit_idx > anchor_idx
        unit_pnl = float(record.unit_pnl_path[exit_idx]) if extended else float(record.baseline_unit_pnl)
        pnl = unit_pnl * quantity
        exit_ts = pd.Timestamp(record.quote_times[exit_idx]) if extended else pd.Timestamp(record.baseline_exit_ts)
        before = equity
        equity += pnl
        peak = max(peak, equity)
        row = row_for_trade(
            record,
            quantity,
            before,
            equity,
            pnl,
            unit_pnl,
            exit_idx,
            exit_ts,
            strategy,
            model_seed,
            sizing_info,
            pred[exit_idx],
            threshold,
        )
        row["baseline_anchor_idx"] = int(anchor_idx)
        row["extended_beyond_baseline"] = bool(extended)
        row["exit_reason"] = "model_extended_exit" if row["extended_beyond_baseline"] else "baseline_anchor_exit"
        rows.append(row)
        open_until_by_session[record.session] = exit_ts
    return rows


def anchored_exit_index(record: Any, prediction: np.ndarray, threshold: float) -> int:
    anchor = baseline_anchor_idx(record)
    tail = np.asarray(prediction, dtype=float)[anchor:]
    eligible = np.where(tail <= float(threshold))[0]
    if len(eligible):
        return int(anchor + eligible[0])
    return int(len(prediction) - 1)


def baseline_anchor_idx(record: Any) -> int:
    times = pd.to_datetime(record.quote_times, utc=True, format="ISO8601")
    anchor = int(np.searchsorted(np.array([ts.value for ts in times], dtype=np.int64), pd.Timestamp(record.baseline_exit_ts).value, side="left"))
    return min(max(anchor, 0), len(record.unit_pnl_path) - 1)


def decide(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_baseline_anchored_continuation_invariant_failure"
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "keep_baseline_anchored_continuation_research_candidate"
    if any(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "mixed_baseline_anchored_continuation_requires_attribution"
    return "reject_baseline_anchored_continuation_no_improvement"


def next_experiment(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "Fix simulator invariants before more lifecycle work."
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "Stress and attribute extension behavior, then test partial reduce/scale-out actions."
    return "Attribution: if extensions lose, move to scale-out/reduce-at-baseline instead of holding longer."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    write_protocol225_style_report(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
