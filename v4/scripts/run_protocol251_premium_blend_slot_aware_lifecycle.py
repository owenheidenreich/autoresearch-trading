"""EXP_PREMIUM_BLEND_SLOT_AWARE_LIFECYCLE_V1.

Historically Protocol251. This experiment applies the slot-aware lifecycle
continuation idea to the current premium-leaning blended challenger entry
stream, rather than to the older Protocol194 lineage.

Scope:
* entry stream stays frozen to CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1
* model only learns post-entry hold/exit behavior
* no fixed hold time, percentage target, or percentage stop is introduced
* evaluation remains strict one-account serial replay

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v4.scripts.run_protocol200_lifecycle_continuation_policy as p200
from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct
from v4.scripts.run_protocol200_lifecycle_continuation_policy import (
    DEFAULT_NORMALIZED_DIR,
    STARTING_CASH,
    build_path_records,
    compare_summaries,
    count_by,
    fit_training_matrix,
    fold_specs,
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
from v4.scripts.run_protocol207_lifecycle_context_calibration import (
    PAPER_DEFAULT_TRADES,
    compare_against_named_baseline,
    load_paper_default_trades,
    summarize_stress,
)


ROLE_LABEL = "EXP_PREMIUM_BLEND_SLOT_AWARE_LIFECYCLE_V1"
HISTORICAL_ID = "Protocol251"
CANDIDATE_LABEL = "CHALLENGER_PREMIUM_BLEND_SLOT_AWARE_LIFECYCLE_V1"
BASE_CHALLENGER_LABEL = "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision/five_seed_model_trades.csv")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_251_premium_blend_slot_aware_lifecycle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
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
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_premium_blend_candidates(args.trades)
    p200.find_normalized_path = safe_find_normalized_path
    records, path_skips = build_path_records(candidates, normalized_dir=args.normalized_dir, forced_flat_time=str(args.forced_flat_time))
    if not records:
        raise SystemExit("no executable premium-blend lifecycle path records were built")
    apply_slot_aware_targets(records)

    baseline_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    for spec in fold_specs():
        train_records = [record for record in records if record.reported_split in spec["train_splits"]]
        validation_records = [record for record in records if record.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_x, train_y, scaler = fit_training_matrix(train_records, max_train_steps=int(args.max_train_steps), seed=251)
        for model_seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": spec["fold"], "seed": int(model_seed), "train_records": len(train_records)}), flush=True)
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
            threshold, sweep = select_threshold(validation_records, validation_predictions, split_name=str(spec["validation_split"]), model_seed=int(model_seed))
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(model_seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [record for record in records if record.reported_split == split]
                predictions = predict_records(model, scaler, split_records)
                model_rows.extend(
                    simulate_serial(
                        split_records,
                        predictions,
                        threshold=float(threshold),
                        model_seed=int(model_seed),
                        strategy=f"{CANDIDATE_LABEL}:{spec['fold']}:seed{model_seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [record for record in split_records if record.session >= "2026-03-01"]
                    march_predictions = {record.uid: predictions[record.uid] for record in march_records if record.uid in predictions}
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_serial(
                            march_records,
                            march_predictions,
                            threshold=float(threshold),
                            model_seed=int(model_seed),
                            strategy=f"{CANDIDATE_LABEL}:{spec['fold']}:seed{model_seed}:march_subset",
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
            print(json.dumps({"stage": "seed_done", "fold": spec["fold"], "seed": int(model_seed), "threshold": float(threshold)}), flush=True)
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [record for record in records if record.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, strategy=BASE_CHALLENGER_LABEL))
            if split == "q1_2026":
                march_records = [record for record in split_records if record.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(march_records, strategy=f"{BASE_CHALLENGER_LABEL}:march_subset")
                )

    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "entry_seed", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    paper_default_frame = load_paper_default_trades(args.paper_default_trades)
    baseline_summary = summarize_replay(baseline_frame, seed_col="entry_seed")
    model_summary = summarize_replay(model_frame, seed_col="combo_seed")
    paper_default_summary = summarize_replay(paper_default_frame, seed_col="seed")
    baseline_comparison = compare_summaries(model_summary, baseline_summary)
    paper_default_comparison = compare_against_named_baseline(model_summary, paper_default_summary, "paper_default")
    invariants = serial_invariants(model_frame)
    stress_010 = summarize_stress(model_frame, seed_col="combo_seed", extra_per_side=0.10)
    stress_025 = summarize_stress(model_frame, seed_col="combo_seed", extra_per_side=0.25)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / model change",
        "changes_paper_default": False,
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "baseline_challenger_label": BASE_CHALLENGER_LABEL,
        "data_used": {
            "premium_blend_trades": str(args.trades),
            "normalized_dir": str(args.normalized_dir),
            "paper_default_trades": str(args.paper_default_trades),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "pre_registration": {
            "hypothesis": "The premium-blend entries are useful, but hardcoded exits/churn clip winners; a slot-aware lifecycle model can improve hold/exit decisions.",
            "entry_stream": "frozen premium-blend challenger trades",
            "holding_action_space": "hold or exit",
            "flat_action_space": "unchanged; no new entry-side knob",
            "threshold_selection": "validation split only",
            "starting_cash": STARTING_CASH,
        },
        "row_counts": {
            "candidate_entries": int(len(candidates)),
            "path_records": int(len(records)),
            "path_skips": int(len(path_skips)),
            "baseline_trade_rows": int(len(baseline_frame)),
            "model_trade_rows": int(len(model_frame)),
        },
        "folds": fold_payloads,
        "model_summary": model_summary,
        "baseline_challenger_summary": baseline_summary,
        "paper_default_summary": paper_default_summary,
        "baseline_challenger_comparison": baseline_comparison,
        "paper_default_comparison": paper_default_comparison,
        "stress_0_10_per_side_summary": stress_010,
        "stress_0_25_per_side_summary": stress_025,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": "",
        "next_experiment": "",
    }
    payload["decision"] = decide(payload)
    payload["next_experiment"] = next_experiment(payload)
    baseline_frame.to_csv(args.out_dir / "premium_blend_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "premium_blend_slot_aware_lifecycle_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_premium_blend_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["reported_split"].astype(str) != "march_2026"].copy()
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["candidate_exit_ts"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["candidate_exit_time"] = frame["candidate_exit_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    frame["candidate_pnl"] = pd.to_numeric(frame.get("raw_candidate_pnl", frame.get("pnl")), errors="coerce")
    frame["candidate_exit_reason"] = frame.get("exit_reason", "").astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int)
    frame["entry_premium"] = pd.to_numeric(frame.get("entry_premium"), errors="coerce")
    frame["entry_ask"] = pd.to_numeric(frame.get("entry_ask"), errors="coerce")
    frame["entry_premium"] = frame["entry_premium"].where(frame["entry_premium"].notna(), frame["entry_ask"] * 100.0)
    for column in ["score", "offset", "entry_bid", "entry_mid"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["session"] = frame["session"].astype(str)
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame["right"] = frame["right"].astype(str)
    frame["fold"] = frame["fold"].astype(str)
    frame["reported_split"] = frame["reported_split"].astype(str)
    frame = frame[frame["decision_ts"].notna() & frame["candidate_exit_ts"].notna()].copy()
    return frame.drop_duplicates(["reported_split", "seed", "session", "decision_time", "contract_id"], keep="last").sort_values(
        ["reported_split", "seed", "session", "decision_ts", "contract_id"]
    )


def safe_find_normalized_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    for path in [*preferred, *fallback]:
        try:
            if path.stat().st_size <= 0:
                continue
            pd.read_parquet(path, columns=[])
        except OSError:
            continue
        except Exception:
            continue
        return path
    return None


def decide(payload: dict[str, Any]) -> str:
    if any(int(payload["invariants"].get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_premium_blend_lifecycle_invariant_failure"
    required = ["q1_2026", "march_2026", "recent_2026"]
    paper = {row["reported_split"]: row for row in payload["paper_default_comparison"]}
    base = {row["reported_split"]: row for row in payload["baseline_challenger_comparison"]}
    stress = {row["reported_split"]: row for row in payload["stress_0_10_per_side_summary"]}
    beats_paper = all(paper.get(split, {}).get("beats_baseline") for split in required)
    beats_base = all(base.get(split, {}).get("beats_baseline") for split in required)
    stress_positive = all(float(stress.get(split, {}).get("median_total_pnl", 0.0)) > 0.0 for split in required)
    if beats_paper and beats_base and stress_positive:
        return "research_candidate_premium_blend_lifecycle_beats_paper_and_base_on_common_splits"
    if beats_paper:
        return "research_only_premium_blend_lifecycle_beats_paper_but_not_base"
    return "rejected_premium_blend_lifecycle_does_not_beat_paper_default"


def next_experiment(payload: dict[str, Any]) -> str:
    if payload["decision"].startswith("research_candidate"):
        return "Run churn/re-entry attribution and runtime parity before any paper-default replacement discussion."
    if payload["decision"].startswith("research_only"):
        return "Attribute where lifecycle changes gave back premium-blend winners; do not add entry knobs."
    return "Reject this lifecycle formulation and move to a risk-adjusted utility objective or broader sequence architecture."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Baseline challenger: {payload['baseline_challenger_label']}",
        f"Data used: `{payload['data_used']['premium_blend_trades']}` plus normalized quote paths",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Paper Default Comparison",
        "",
        "| split | model | Protocol101 | delta | model PF | paper PF | model trades | paper trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paper_default_comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | {money(row['baseline_median_total_pnl'])} | "
            f"{money(row['delta_vs_baseline'])} | {row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | {pct(row['model_positive_seed_fraction'])} |"
        )
    lines.extend(["", "## Premium-Blend Baseline Comparison", "", "| split | model | base | delta | model PF | base PF |", "|---|---:|---:|---:|---:|---:|"])
    for row in payload["baseline_challenger_comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | {money(row['baseline_median_total_pnl'])} | "
            f"{money(row['delta_vs_baseline'])} | {row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} |"
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
            f"- Model trades: `{path.parent / 'premium_blend_slot_aware_lifecycle_trades.csv'}`",
            f"- Baseline trades: `{path.parent / 'premium_blend_baseline_serial_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    f"## {HISTORICAL_ID} - {ROLE_LABEL}",
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    f"- Candidate: {payload['candidate_label']}",
                    f"- Baseline: {payload['paper_default_label']}",
                    f"- Data used: `{payload['data_used']['premium_blend_trades']}` plus normalized quote paths",
                    "- Paid data downloaded: false",
                    "- Broker endpoint called: false",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
