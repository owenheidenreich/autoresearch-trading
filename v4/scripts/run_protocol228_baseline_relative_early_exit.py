"""EXP_2026_05_22_BASELINE_RELATIVE_EARLY_EXIT_V1.

Historically Protocol228. Protocol225 exited too early because it tried to
replace the lifecycle. Protocol226/227 showed that holding longer or holding
runners did not help. This experiment reframes scale-out as a baseline-relative
decision:

    While holding before the baseline exit, is the current executable bid better
    than the baseline exit value we would otherwise expect?

The model can exit early only when its predicted advantage over the baseline
exit clears a validation-selected threshold. If not, the trade falls back to the
baseline exit. No add actions and no runner-to-close behavior are introduced.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.scripts.run_protocol225_account_aware_lifecycle_exit_policy import (
    ContinuationMLP,
    DEFAULT_DATASET,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_TRADES,
    REQUIRED_SPLITS,
    STARTING_CASH,
    TARGET_SCALE,
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
)
from v4.model.supervised_pilot import FeatureScaler


ROLE_LABEL = "EXP_2026_05_22_BASELINE_RELATIVE_EARLY_EXIT_V1"
HISTORICAL_ID = "Protocol228"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_228_baseline_relative_early_exit")
TARGET_CLIP = 3_000.0
THRESHOLD_CANDIDATES = (-250, -100, 0, 50, 100, 150, 200, 300, 500, 750, 1_000, 1_500)


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
    apply_baseline_relative_targets(records)
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
        train_x, train_y, scaler = fit_training_matrix(train_records, max_train_steps=int(args.max_train_steps), seed=228)
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
                    simulate_early_exit_serial(
                        split_records,
                        predictions,
                        sizing=sizing,
                        threshold=threshold,
                        model_seed=int(seed),
                        strategy=f"protocol228:{spec['fold']}:seed{seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_predictions = {r.uid: predictions[r.uid] for r in march_records if r.uid in predictions}
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_early_exit_serial(
                            march_records,
                            march_predictions,
                            sizing=sizing,
                            threshold=threshold,
                            model_seed=int(seed),
                            strategy=f"protocol228:{spec['fold']}:seed{seed}:march_subset",
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
        "what_is_this": "experiment / baseline-relative early scale-out model",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_BASELINE_RELATIVE_EARLY_EXIT_V1",
        "entry_source": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "sizing_source": "EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "pre_registration": {
            "hypothesis": "Learn when current executable bid is better than the baseline exit value, then exit/scale out early.",
            "target": "current_unit_pnl - baseline_unit_pnl, clipped, validation threshold selected chronologically",
            "fallback": "baseline exit if predicted advantage never clears threshold before baseline",
            "no_runner_or_add": True,
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
            "model_trades": str(args.out_dir / "protocol228_model_serial_trades.csv"),
            "baseline_trades": str(args.out_dir / "protocol223_baseline_serial_trades.csv"),
            "threshold_sweep": str(args.out_dir / "threshold_sweep.csv"),
            "path_skips": str(args.out_dir / "path_skips.csv"),
        },
    }
    baseline_frame.to_csv(args.out_dir / "protocol223_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol228_model_serial_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
) -> tuple[ContinuationMLP, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContinuationMLP(input_dim=train_x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=min(batch_size, len(train_y)),
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            # Penalize sign errors more than small magnitude errors; this is
            # an early-exit decision, not just a smooth value estimate.
            loss = F.huber_loss(pred, yb, delta=1.0) + 0.25 * F.binary_cross_entropy_with_logits(pred, (yb > 0.0).float())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": float(epoch), "train_loss": epoch_loss})
    model.load_state_dict(best_state)
    return model, history


def apply_baseline_relative_targets(records: Sequence[Any]) -> None:
    for record in records:
        baseline = float(record.baseline_unit_pnl)
        target = np.clip(record.unit_pnl_path.astype(np.float32) - baseline, -TARGET_CLIP, TARGET_CLIP)
        record.target = target.astype(np.float32)


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
        trades = simulate_early_exit_serial(
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


def simulate_early_exit_serial(
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
        exit_idx, early = early_exit_index(record, pred, threshold)
        unit_pnl = float(record.unit_pnl_path[exit_idx]) if early else float(record.baseline_unit_pnl)
        exit_ts = pd.Timestamp(record.quote_times[exit_idx]) if early else pd.Timestamp(record.baseline_exit_ts)
        pnl = unit_pnl * quantity
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
        row["early_exit_before_baseline"] = bool(early)
        row["exit_reason"] = "baseline_relative_early_exit" if early else "baseline_fallback_exit"
        rows.append(row)
        open_until_by_session[record.session] = exit_ts
    return rows


def early_exit_index(record: Any, prediction: np.ndarray, threshold: float) -> tuple[int, bool]:
    anchor = baseline_anchor_idx(record)
    head = np.asarray(prediction, dtype=float)[: anchor + 1]
    eligible = np.where(head >= float(threshold))[0]
    if len(eligible):
        return int(eligible[0]), True
    return anchor, False


def baseline_anchor_idx(record: Any) -> int:
    times = pd.to_datetime(record.quote_times, utc=True, format="ISO8601")
    anchor = int(np.searchsorted(np.array([ts.value for ts in times], dtype=np.int64), pd.Timestamp(record.baseline_exit_ts).value, side="left"))
    return min(max(anchor, 0), len(record.unit_pnl_path) - 1)


def decide(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_baseline_relative_early_exit_invariant_failure"
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "keep_baseline_relative_early_exit_research_candidate"
    if any(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "mixed_baseline_relative_early_exit_requires_attribution"
    return "reject_baseline_relative_early_exit_no_improvement"


def next_experiment(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "Fix simulator invariants before more lifecycle work."
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "Stress and attribute early exits, then test partial reduce actions instead of all-out exits."
    return "If early exits fail, build a true action dataset with wait/exit/reduce labels from a session DP rather than path-local labels."


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
        "| split | model PnL | baseline PnL | delta | model PF | baseline PF | model trades | baseline trades | model DD | baseline DD | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | {money(row['baseline_median_total_pnl'])} | "
            f"{money(row['delta_vs_baseline'])} | {row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{money(row['model_median_max_drawdown'])} | {money(row['baseline_median_max_drawdown'])} | "
            f"{pct(row['model_positive_seed_fraction'])} |"
        )
    lines.extend(["", "## Thresholds", ""])
    if payload["threshold_summary"]:
        lines.extend(["| fold | seed | threshold | validation PnL | PF | trades |", "|---|---:|---:|---:|---:|---:|"])
        for row in payload["threshold_summary"]:
            lines.append(
                f"| {row['fold']} | {row['model_seed']} | {row['selected_threshold']:.0f} | "
                f"{money(row['validation_total_pnl'])} | {row['validation_profit_factor']:.3f} | {row['validation_trades']} |"
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
            f"- Threshold sweep: `{payload['outputs']['threshold_sweep']}`",
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


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


if __name__ == "__main__":
    raise SystemExit(main())
