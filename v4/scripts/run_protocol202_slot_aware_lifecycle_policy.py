"""Protocol202: slot-aware lifecycle continuation target.

Protocol200 learned continuation value for the current contract but ignored the
single-position opportunity cost. Protocol201 showed that it over-held and
blocked profitable later baseline entries. Protocol202 changes the training
target, not the entry stream:

    hold only when continuing this contract is better than exiting now and
    preserving the single account slot for future candidate entries.

The target is built with a per-session dynamic program over the frozen
Protocol194 candidate stream. It is hindsight supervision, not a tradable rule.
Evaluation remains serial one-account replay with model exits.

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

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct
from v4.scripts.run_protocol200_lifecycle_continuation_policy import (
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_REPLAY_DIRS,
    STARTING_CASH,
    TARGET_CLIP,
    build_path_records,
    compare_summaries,
    count_by,
    fit_training_matrix,
    fold_specs,
    load_candidate_entries,
    predict_records,
    select_threshold,
    serial_invariants,
    simulate_baseline_serial,
    simulate_serial,
    summarize_replay,
    threshold_summary,
    train_model,
)


LOOP_ID = "v4_aplus_hypothesis_202_slot_aware_lifecycle_policy"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", action="append", type=Path, default=None)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=650_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
    fold_payloads = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_x, train_y, scaler = fit_training_matrix(train_records, max_train_steps=args.max_train_steps, seed=202)
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
            validation_predictions = predict_records(model, scaler, validation_records)
            threshold, sweep = select_threshold(
                validation_records,
                validation_predictions,
                split_name=str(spec["validation_split"]),
                model_seed=int(model_seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(model_seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                predictions = predict_records(model, scaler, split_records)
                model_rows.extend(
                    simulate_serial(
                        split_records,
                        predictions,
                        threshold=threshold,
                        model_seed=int(model_seed),
                        strategy=f"protocol202:{spec['fold']}:seed{model_seed}",
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
                            strategy=f"protocol202:{spec['fold']}:seed{model_seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(model_seed),
                    "threshold": float(threshold),
                    "history": history,
                    "train_records": len(train_records),
                    "validation_records": len(validation_records),
                    "train_steps_used": int(len(train_y)),
                }
            )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, strategy="protocol194_protocol081_baseline"))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(march_records, strategy="protocol194_protocol081_baseline:march_subset")
                )
    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "entry_seed", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    baseline_summary = summarize_replay(baseline_frame, seed_col="entry_seed")
    model_summary = summarize_replay(model_frame, seed_col="combo_seed")
    comparison = compare_summaries(model_summary, baseline_summary)
    invariants = serial_invariants(model_frame)
    payload = {
        "protocol": "202_slot_aware_lifecycle_policy",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": True,
        "source_replay_dirs": [str(path) for path in replay_dirs],
        "normalized_dir": str(args.normalized_dir),
        "row_counts": {
            "candidate_entries": int(len(candidates)),
            "path_records": int(len(records)),
            "path_skips": int(len(path_skips)),
            "baseline_trade_rows": int(len(baseline_frame)),
            "model_trade_rows": int(len(model_frame)),
        },
        "pre_registration": {
            "hypothesis": (
                "A slot-aware lifecycle target should reduce Protocol200's over-holding by valuing the "
                "current contract against the future candidate stream available after exit."
            ),
            "entry_stream": "frozen Protocol194 candidate entries",
            "holding_action_space": "hold or exit",
            "flat_action_space": "unchanged for this protocol; no new entry-side knob",
            "starting_cash": STARTING_CASH,
            "target": "dynamic-programming opportunity-cost target over the single-position candidate stream",
            "threshold_selection": "validation split only",
        },
        "folds": fold_payloads,
        "baseline_summary": baseline_summary,
        "model_summary": model_summary,
        "comparison": comparison,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": decide_protocol202(comparison, invariants),
        "next_gate": next_gate(comparison),
    }
    baseline_frame.to_csv(args.out_dir / "protocol194_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol202_model_serial_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def apply_slot_aware_targets(records: Sequence[Any]) -> None:
    groups: dict[tuple[str, int, str], list[Any]] = {}
    for record in records:
        groups.setdefault((record.reported_split, int(record.entry_seed), record.session), []).append(record)
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (r.decision_ts, r.contract_id))
        decision_ns = np.array([pd.Timestamp(record.decision_ts).value for record in ordered], dtype=np.int64)
        flat_value = np.zeros(len(ordered) + 1, dtype=np.float32)
        target_by_uid: dict[str, np.ndarray] = {}
        for idx in range(len(ordered) - 1, -1, -1):
            record = ordered[idx]
            path_ns = np.array([pd.Timestamp(value).value for value in record.quote_times], dtype=np.int64)
            next_indices = np.searchsorted(decision_ns, path_ns, side="left")
            next_values = flat_value[np.clip(next_indices, 0, len(ordered))]
            step_total_value = record.path_pnl.astype(np.float32) + next_values
            best_from_step = np.maximum.accumulate(step_total_value[::-1])[::-1]
            target = np.clip(best_from_step - step_total_value, -TARGET_CLIP, TARGET_CLIP).astype(np.float32)
            target_by_uid[record.uid] = target
            enter_value = float(best_from_step[0]) if len(best_from_step) else -math.inf
            wait_value = float(flat_value[idx + 1])
            flat_value[idx] = max(wait_value, enter_value)
        for record in ordered:
            record.target = target_by_uid[record.uid]


def next_gate(comparison: list[dict[str, Any]]) -> str:
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in ["q1_2026", "march_2026", "recent_2026"]):
        return "Run Protocol202 five-seed confirmation and churn/blocked-entry attribution."
    if any(by_split.get(split, {}).get("beats_baseline") for split in ["q1_2026", "march_2026", "recent_2026"]):
        return "Attribute the mixed slot-aware result before changing architecture."
    return "Reject this slot-aware MLP form; next hypothesis should use a recurrent holding-state model or return to entry-side opportunity-cost training."


def decide_protocol202(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_protocol202_invariant_failure"
    by_split = {row["reported_split"]: row for row in comparison}
    required = ["q1_2026", "march_2026", "recent_2026"]
    if all(by_split.get(split, {}).get("beats_baseline") for split in required):
        return "keep_research_candidate_protocol202_beats_lifecycle_baseline_on_tests"
    if any(by_split.get(split, {}).get("beats_baseline") for split in required):
        return "mixed_protocol202_slot_aware_lifecycle_signal_requires_attribution"
    return "reject_protocol202_no_test_improvement"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol202 Slot-Aware Lifecycle Policy",
        "",
        "No paid data was downloaded. No broker endpoint was called. No live or paper orders were placed.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Next gate: {payload['next_gate']}",
        "",
        "## Comparison",
        "",
        "| split | model PnL | baseline PnL | delta | model PF | baseline PF | model trades | baseline trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
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
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'protocol202_model_serial_trades.csv'}`",
            f"- Baseline trades: `{path.parent / 'protocol194_baseline_serial_trades.csv'}`",
            f"- Threshold sweep: `{path.parent / 'threshold_sweep.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
