"""EXP_2026_05_22_LIFECYCLE_CONTEXT_CALIBRATION_V1.

Historically Protocol207. This is a research experiment, not a paper-trading
promotion.

Hypothesis:
    CHALLENGER_LIFECYCLE_SLOT_AWARE_V1 was directionally useful, but the recent
    gap showed context-specific lifecycle miscalibration, especially side/time
    behavior. Instead of hardcoding hold times, profit targets, or percentage
    moves, calibrate the model's predicted continuation value by side/time
    residuals learned only from the chronological validation split.

The paper default remains PAPER_DEFAULT_PROTOCOL101. No paid data is downloaded
and no broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct
from v4.scripts.run_protocol200_lifecycle_continuation_policy import (
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_REPLAY_DIRS,
    STARTING_CASH,
    THRESHOLD_CANDIDATES,
    build_path_records,
    compare_summaries,
    count_by,
    fit_training_matrix,
    fold_specs,
    load_candidate_entries,
    metrics_for_rows,
    predict_records,
    select_threshold,
    serial_invariants,
    simulate_baseline_serial,
    simulate_serial,
    summarize_replay,
    threshold_summary,
    train_model,
)
from v4.scripts.run_protocol202_slot_aware_lifecycle_policy import apply_slot_aware_targets


ROLE_LABEL = "EXP_2026_05_22_LIFECYCLE_CONTEXT_CALIBRATION_V1"
CANDIDATE_LABEL = "CHALLENGER_LIFECYCLE_CONTEXT_CALIBRATED_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
OTHER_BASELINE_LABEL = "CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1_WITH_FROZEN_PROTOCOL081_EXITS"
DECISION_LABEL = "DECISION_RESEARCH_ONLY_CONTEXT_CALIBRATION"
LOOP_ID = "v4_aplus_hypothesis_207_lifecycle_context_calibration"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
PAPER_DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_181_protocol101_anchor_overlay/protocol101_baseline_trades.csv")
NY = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", action="append", type=Path, default=None)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--paper-default-trades", type=Path, default=PAPER_DEFAULT_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=650_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-context-steps", type=int, default=1_000)
    parser.add_argument("--max-context-offset", type=float, default=750.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_dirs = args.replay_dir or DEFAULT_REPLAY_DIRS
    candidates = load_candidate_entries(replay_dirs)
    records, path_skips = build_path_records(candidates, normalized_dir=args.normalized_dir, forced_flat_time=args.forced_flat_time)
    apply_slot_aware_targets(records)

    baseline_rows = []
    model_rows = []
    threshold_rows = []
    calibration_rows = []
    fold_payloads = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        calibration_records, threshold_records = split_validation_records(validation_records)
        if not train_records or not calibration_records or not threshold_records:
            continue
        train_x, train_y, scaler = fit_training_matrix(train_records, max_train_steps=args.max_train_steps, seed=207)
        for model_seed in args.seeds:
            model, history = train_model(
                train_x,
                train_y,
                scaler=scaler,
                seed=int(model_seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                learning_rate=float(args.learning_rate),
            )
            calibration_predictions = predict_records(model, scaler, calibration_records)
            calibration = calibrate_context_offsets(
                calibration_records,
                calibration_predictions,
                min_context_steps=int(args.min_context_steps),
                max_context_offset=float(args.max_context_offset),
            )
            threshold_predictions_raw = predict_records(model, scaler, threshold_records)
            threshold_predictions = apply_context_offsets(threshold_records, threshold_predictions_raw, calibration)
            threshold, sweep = select_threshold(
                threshold_records,
                threshold_predictions,
                split_name=f"{spec['validation_split']}:threshold_half",
                model_seed=int(model_seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(model_seed)} for row in sweep)
            calibration_rows.extend(calibration_records_for_csv(spec["fold"], int(model_seed), calibration))
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                raw_predictions = predict_records(model, scaler, split_records)
                predictions = apply_context_offsets(split_records, raw_predictions, calibration)
                model_rows.extend(
                    simulate_serial(
                        split_records,
                        predictions,
                        threshold=threshold,
                        model_seed=int(model_seed),
                        strategy=f"{CANDIDATE_LABEL}:{spec['fold']}:seed{model_seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_predictions = {r.uid: predictions[r.uid] for r in march_records if r.uid in predictions}
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_serial(
                            march_records,
                            march_predictions,
                            threshold=threshold,
                            model_seed=int(model_seed),
                            strategy=f"{CANDIDATE_LABEL}:{spec['fold']}:seed{model_seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "calibration_records": len(calibration_records),
                    "threshold_records": len(threshold_records),
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(model_seed),
                    "threshold": float(threshold),
                    "history": history,
                    "train_records": len(train_records),
                    "train_steps_used": int(len(train_y)),
                    "context_offsets": calibration["context_offsets"],
                    "side_offsets": calibration["side_offsets"],
                }
            )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, strategy=OTHER_BASELINE_LABEL))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(march_records, strategy=f"{OTHER_BASELINE_LABEL}:march_subset")
                )

    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "entry_seed", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    calibration_frame = pd.DataFrame(calibration_rows)
    paper_default_frame = load_paper_default_trades(args.paper_default_trades)

    baseline_summary = summarize_replay(baseline_frame, seed_col="entry_seed")
    model_summary = summarize_replay(model_frame, seed_col="combo_seed")
    paper_default_summary = summarize_replay(paper_default_frame, seed_col="seed")
    lifecycle_baseline_comparison = compare_summaries(model_summary, baseline_summary)
    paper_default_comparison = compare_against_named_baseline(model_summary, paper_default_summary, "paper_default")
    invariants = serial_invariants(model_frame)
    stress_010 = summarize_stress(model_frame, seed_col="combo_seed", extra_per_side=0.10)
    stress_025 = summarize_stress(model_frame, seed_col="combo_seed", extra_per_side=0.25)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": "Protocol207",
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "other_baseline_label": OTHER_BASELINE_LABEL,
        "decision_label": DECISION_LABEL,
        "what_is_this": "experiment / research challenger training run",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "source_replay_dirs": [str(path) for path in replay_dirs],
        "normalized_dir": str(args.normalized_dir),
        "paper_default_trades": str(args.paper_default_trades),
        "row_counts": {
            "candidate_entries": int(len(candidates)),
            "path_records": int(len(records)),
            "path_skips": int(len(path_skips)),
            "model_trade_rows": int(len(model_frame)),
            "paper_default_rows": int(len(paper_default_frame)),
        },
        "pre_registration": {
            "hypothesis": (
                "Slot-aware lifecycle continuation is useful, but context-specific prediction miscalibration "
                "can make exits too early or too late. Validation-only side/time residual calibration should "
                "improve hold/exit decisions without hardcoded hold durations, profit targets, or percentage moves."
            ),
            "candidate": CANDIDATE_LABEL,
            "paper_default_baseline": PAPER_DEFAULT_LABEL,
            "other_baseline": OTHER_BASELINE_LABEL,
            "entry_stream": "frozen full-action surface-edge entry stream from the Protocol194 challenger lineage",
            "holding_action_space": "hold or exit",
            "flat_action_space": "unchanged for this experiment; no new entry-side knob",
            "validation_discipline": (
                "Each chronological validation split is divided by session: first half calibrates side/time "
                "residual offsets, second half selects the exit threshold."
            ),
            "no_hardcoded_exit_rules": True,
            "starting_cash": STARTING_CASH,
            "max_contracts": 1,
            "max_concurrent_positions": 1,
        },
        "folds": fold_payloads,
        "model_summary": model_summary,
        "paper_default_summary": paper_default_summary,
        "lifecycle_baseline_summary": baseline_summary,
        "paper_default_comparison": paper_default_comparison,
        "lifecycle_baseline_comparison": lifecycle_baseline_comparison,
        "stress_0_10_per_side_summary": stress_010,
        "stress_0_25_per_side_summary": stress_025,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": decide(paper_default_comparison, lifecycle_baseline_comparison, invariants, stress_010),
        "next_experiment": next_experiment(paper_default_comparison, lifecycle_baseline_comparison),
    }
    model_frame.to_csv(args.out_dir / "challenger_lifecycle_context_calibrated_trades.csv", index=False)
    baseline_frame.to_csv(args.out_dir / "lifecycle_baseline_trades.csv", index=False)
    paper_default_frame.to_csv(args.out_dir / "paper_default_protocol101_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    calibration_frame.to_csv(args.out_dir / "context_calibration_offsets.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def split_validation_records(records: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    sessions = sorted({record.session for record in records})
    if len(sessions) < 2:
        return list(records), list(records)
    midpoint = max(1, len(sessions) // 2)
    calibration_sessions = set(sessions[:midpoint])
    threshold_sessions = set(sessions[midpoint:])
    calibration = [record for record in records if record.session in calibration_sessions]
    threshold = [record for record in records if record.session in threshold_sessions]
    return calibration or list(records), threshold or list(records)


def calibrate_context_offsets(
    records: Sequence[Any],
    predictions: dict[str, np.ndarray],
    *,
    min_context_steps: int,
    max_context_offset: float,
) -> dict[str, Any]:
    global_residuals = []
    by_context: dict[str, list[np.ndarray]] = {}
    by_side: dict[str, list[np.ndarray]] = {}
    for record in records:
        pred = predictions.get(record.uid)
        if pred is None or len(pred) != len(record.target):
            continue
        residual = np.asarray(record.target, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        residual = residual[np.isfinite(residual)]
        if len(residual) == 0:
            continue
        global_residuals.append(residual)
        by_context.setdefault(context_key(record), []).append(residual)
        by_side.setdefault(str(record.right), []).append(residual)
    global_values = np.concatenate(global_residuals) if global_residuals else np.array([0.0], dtype=np.float32)
    global_median = float(np.median(global_values))
    side_offsets = residual_offsets(by_side, global_median, min_context_steps=min_context_steps, max_offset=max_context_offset)
    context_offsets = residual_offsets(by_context, global_median, min_context_steps=min_context_steps, max_offset=max_context_offset)
    return {
        "global_median_residual": global_median,
        "context_offsets": context_offsets,
        "side_offsets": side_offsets,
        "min_context_steps": int(min_context_steps),
        "max_context_offset": float(max_context_offset),
    }


def residual_offsets(
    groups: dict[str, list[np.ndarray]],
    global_median: float,
    *,
    min_context_steps: int,
    max_offset: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, chunks in sorted(groups.items()):
        values = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        if len(values) < int(min_context_steps):
            continue
        median_residual = float(np.median(values))
        offset = float(np.clip(median_residual - global_median, -max_offset, max_offset))
        out[key] = {
            "offset": offset,
            "steps": float(len(values)),
            "median_residual": median_residual,
        }
    return out


def apply_context_offsets(
    records: Sequence[Any],
    predictions: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for record in records:
        pred = predictions.get(record.uid)
        if pred is None:
            continue
        out[record.uid] = np.asarray(pred, dtype=np.float32) + offset_for_record(record, calibration)
    return out


def offset_for_record(record: Any, calibration: dict[str, Any]) -> float:
    context = calibration.get("context_offsets", {}).get(context_key(record))
    if context is not None:
        return float(context.get("offset", 0.0))
    side = calibration.get("side_offsets", {}).get(str(record.right))
    if side is not None:
        return float(side.get("offset", 0.0))
    return 0.0


def calibration_records_for_csv(fold: str, model_seed: int, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scope, offsets in [("context", calibration.get("context_offsets", {})), ("side", calibration.get("side_offsets", {}))]:
        for key, value in offsets.items():
            rows.append(
                {
                    "fold": fold,
                    "model_seed": int(model_seed),
                    "scope": scope,
                    "key": key,
                    "offset": float(value.get("offset", 0.0)),
                    "steps": int(value.get("steps", 0.0)),
                    "median_residual": float(value.get("median_residual", 0.0)),
                    "global_median_residual": float(calibration.get("global_median_residual", 0.0)),
                }
            )
    return rows


def context_key(record: Any) -> str:
    return f"{record.right}|{time_bucket(record.decision_ts)}"


def time_bucket(timestamp: pd.Timestamp) -> str:
    local = pd.Timestamp(timestamp).tz_convert(NY)
    minutes = local.hour * 60 + local.minute
    if minutes < 10 * 60:
        return "first_30"
    if minutes < 11 * 60 + 30:
        return "post_open_morning"
    if minutes < 13 * 60 + 30:
        return "midday"
    return "late_afternoon"


def load_paper_default_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame = frame.copy()
    frame["reported_split"] = frame["reported_split"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int)
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    return frame


def compare_against_named_baseline(
    model_summary: list[dict[str, Any]],
    baseline_summary: list[dict[str, Any]],
    baseline_name: str,
) -> list[dict[str, Any]]:
    baseline = {row["reported_split"]: row for row in baseline_summary}
    rows = []
    for model in model_summary:
        split = model["reported_split"]
        base = baseline.get(split, {})
        base_pnl = finite_float(base.get("median_total_pnl"), 0.0)
        model_pnl = finite_float(model.get("median_total_pnl"), 0.0)
        rows.append(
            {
                "reported_split": split,
                "baseline_name": baseline_name,
                "model_median_total_pnl": model_pnl,
                "baseline_median_total_pnl": base_pnl,
                "delta_vs_baseline": model_pnl - base_pnl,
                "model_median_profit_factor": finite_float(model.get("median_profit_factor"), 0.0),
                "baseline_median_profit_factor": finite_float(base.get("median_profit_factor"), 0.0),
                "model_median_trades": finite_float(model.get("median_trades"), 0.0),
                "baseline_median_trades": finite_float(base.get("median_trades"), 0.0),
                "model_positive_seed_fraction": finite_float(model.get("positive_seed_fraction"), 0.0),
                "baseline_positive_seed_fraction": finite_float(base.get("positive_seed_fraction"), 0.0),
                "beats_baseline": bool(model_pnl > base_pnl),
            }
        )
    return rows


def summarize_stress(frame: pd.DataFrame, *, seed_col: str, extra_per_side: float) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    stressed = frame.copy()
    stressed["pnl"] = pd.to_numeric(stressed["pnl"], errors="coerce").fillna(0.0) - float(extra_per_side) * 2.0 * 100.0
    return summarize_replay(stressed, seed_col=seed_col)


def decide(
    paper_default_comparison: list[dict[str, Any]],
    lifecycle_baseline_comparison: list[dict[str, Any]],
    invariants: dict[str, Any],
    stress_010: list[dict[str, Any]],
) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_context_calibration_invariant_failure"
    common_required = ["q1_2026", "march_2026", "recent_2026"]
    paper = {row["reported_split"]: row for row in paper_default_comparison}
    lifecycle = {row["reported_split"]: row for row in lifecycle_baseline_comparison}
    stress = {row["reported_split"]: row for row in stress_010}
    beats_paper = all(paper.get(split, {}).get("beats_baseline") for split in common_required)
    beats_lifecycle = all(lifecycle.get(split, {}).get("beats_baseline") for split in common_required)
    stress_positive = all(finite_float(stress.get(split, {}).get("median_total_pnl"), 0.0) > 0.0 for split in common_required)
    if beats_paper and beats_lifecycle and stress_positive:
        return "keep_research_challenger_context_calibrated_lifecycle_common_splits_only"
    if beats_paper:
        return "mixed_context_calibration_beats_paper_default_but_not_lifecycle_baseline"
    return "reject_context_calibration_does_not_beat_paper_default"


def next_experiment(
    paper_default_comparison: list[dict[str, Any]],
    lifecycle_baseline_comparison: list[dict[str, Any]],
) -> str:
    paper = {row["reported_split"]: row for row in paper_default_comparison}
    lifecycle = {row["reported_split"]: row for row in lifecycle_baseline_comparison}
    recent_lifecycle = lifecycle.get("recent_2026", {})
    if paper and all(row.get("beats_baseline") for row in paper.values()) and recent_lifecycle.get("beats_baseline"):
        return (
            "Build a no-order runtime parity harness for this challenger and separately solve the missing "
            "Q3/Q4 out-of-sample lifecycle validation coverage before any paper-default decision."
        )
    if paper and all(row.get("beats_baseline") for row in paper.values()):
        return (
            "Do attribution against the lifecycle baseline. If the only miss is the strong Protocol194/081 "
            "baseline, keep this as a paper-default challenger but do not replace runtime."
        )
    return (
        "Reject this calibration path and move to a unified entry-plus-lifecycle sequence objective, because "
        "post-entry calibration alone did not clear the paper-default gate."
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        f"Does it change the paper-trading default: {'yes' if payload['changes_paper_default'] else 'no'}",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        "Data used: existing Protocol194-lineage candidate entries, existing normalized official-context SPXW rows, and existing Protocol101 paper-default trades.",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Paper Default Comparison",
        "",
        "| split | challenger | paper default | delta | challenger PF | paper PF | challenger trades | paper trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paper_default_comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{pct(row['model_positive_seed_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Lifecycle Baseline Comparison",
            "",
            "| split | challenger | lifecycle baseline | delta | beats baseline |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["lifecycle_baseline_comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['beats_baseline']} |"
        )
    lines.extend(
        [
            "",
            "## Stress",
            "",
            "| split | stress $0.10/side median PnL | stress $0.25/side median PnL |",
            "|---|---:|---:|",
        ]
    )
    stress_010 = {row["reported_split"]: row for row in payload["stress_0_10_per_side_summary"]}
    stress_025 = {row["reported_split"]: row for row in payload["stress_0_25_per_side_summary"]}
    for split in sorted(stress_010):
        lines.append(
            f"| {split} | {money(stress_010[split].get('median_total_pnl'))} | "
            f"{money(stress_025.get(split, {}).get('median_total_pnl'))} |"
        )
    lines.extend(
        [
            "",
            "## Calibration Discipline",
            "",
            "- Validation sessions are split chronologically: first half for residual calibration, second half for threshold selection.",
            "- Calibration changes predicted continuation value only; it does not impose a minimum hold time, fixed profit target, or percentage stop.",
            "- The paper default remains unchanged regardless of this result.",
            "",
            "## Invariants",
            "",
            f"- Overlap violations: `{payload['invariants']['overlap_violations']}`",
            f"- Unaffordable violations: `{payload['invariants']['unaffordable_violations']}`",
            f"- NaN time rows: `{payload['invariants']['nan_time_rows']}`",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Challenger trades: `{path.parent / 'challenger_lifecycle_context_calibrated_trades.csv'}`",
            f"- Calibration offsets: `{path.parent / 'context_calibration_offsets.csv'}`",
            f"- Paper default trades: `{path.parent / 'paper_default_protocol101_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


if __name__ == "__main__":
    raise SystemExit(main())

