"""EXP_BASELINE_PRESERVING_LIFECYCLE_EXTENSION_V1.

Historically Protocol252. Protocol251 showed that fully replacing premium-blend
exits was too aggressive and clipped the March/Q1 edge. This experiment is a
more conservative lifecycle test:

* keep the premium-blend entry stream fixed
* keep the premium-blend baseline exit as the default
* allow a learned hold/exit model to extend after the baseline exit only when
  post-entry state supports continuing to occupy the single account slot

No hardcoded hold time, profit target, or percentage exit is introduced. The
model sees causal post-entry state; hindsight is used only for supervised
training labels. No paid data is downloaded and no broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct
from v4.scripts.run_protocol200_lifecycle_continuation_policy import (
    DEFAULT_NORMALIZED_DIR,
    STARTING_CASH,
    TARGET_SCALE,
    THRESHOLD_CANDIDATES,
    ContinuationMLP,
    build_path_records,
    compare_summaries,
    count_by,
    fit_training_matrix,
    fold_specs,
    metrics_for_rows,
    predict_records,
    serial_invariants,
    simulate_baseline_serial,
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
import v4.scripts.run_protocol251_premium_blend_slot_aware_lifecycle as p251


ROLE_LABEL = "EXP_BASELINE_PRESERVING_LIFECYCLE_EXTENSION_V1"
HISTORICAL_ID = "Protocol252"
CANDIDATE_LABEL = "CHALLENGER_BASELINE_PRESERVING_LIFECYCLE_EXTENSION_V1"
BASE_CHALLENGER_LABEL = "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
DEFAULT_TRADES = p251.DEFAULT_TRADES
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_252_baseline_preserving_lifecycle_extension")


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
    candidates = p251.load_premium_blend_candidates(args.trades)
    p251.p200.find_normalized_path = p251.safe_find_normalized_path
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
        train_x, train_y, scaler = fit_extension_training_matrix(
            train_records,
            max_train_steps=int(args.max_train_steps),
            seed=252,
        )
        if len(train_y) == 0:
            continue
        for model_seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": spec["fold"], "seed": int(model_seed), "train_steps": int(len(train_y))}), flush=True)
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
            threshold, sweep = select_extension_threshold(
                validation_records,
                validation_predictions,
                split_name=str(spec["validation_split"]),
                model_seed=int(model_seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(model_seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [record for record in records if record.reported_split == split]
                predictions = predict_records(model, scaler, split_records)
                model_rows.extend(
                    simulate_extension_serial(
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
                        for row in simulate_extension_serial(
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
    extension_summary = summarize_extensions(model_frame)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / conservative lifecycle model change",
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
            "hypothesis": "Preserve the premium-blend baseline exit, then learn only whether extending after that exit captures additional value without creating churn.",
            "entry_stream": "frozen premium-blend challenger trades",
            "default_exit": "premium-blend baseline exit",
            "allowed_lifecycle_change": "extend after baseline exit only",
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
        "extension_summary": extension_summary,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": "",
        "next_experiment": "",
    }
    payload["decision"] = decide(payload)
    payload["next_experiment"] = next_experiment(payload)
    baseline_frame.to_csv(args.out_dir / "premium_blend_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "baseline_preserving_lifecycle_extension_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def fit_extension_training_matrix(records: Sequence[Any], *, max_train_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray, Any]:
    extension_features = []
    extension_targets = []
    for record in records:
        idx = baseline_index(record)
        if idx >= len(record.features):
            continue
        extension_features.append(record.features[idx:])
        extension_targets.append(record.target[idx:])
    if not extension_targets:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.float32), fit_training_matrix(records, max_train_steps=1, seed=seed)[2]
    x = np.vstack(extension_features).astype(np.float32)
    y = np.concatenate(extension_targets).astype(np.float32) / TARGET_SCALE
    if max_train_steps > 0 and len(y) > max_train_steps:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(y), size=max_train_steps, replace=False))
        x = x[idx]
        y = y[idx]
    scaler = fit_training_matrix(records, max_train_steps=max_train_steps, seed=seed)[2]
    return scaler.transform(x), y, scaler


def baseline_index(record: Any) -> int:
    times = np.asarray([pd.Timestamp(value).value for value in record.quote_times], dtype=np.int64)
    idx = int(np.searchsorted(times, pd.Timestamp(record.baseline_exit_ts).value, side="left"))
    return min(max(idx, 0), len(times) - 1)


def select_extension_threshold(
    records: Sequence[Any],
    predictions: dict[str, np.ndarray],
    *,
    split_name: str,
    model_seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = float(THRESHOLD_CANDIDATES[0])
    best_key = (-1e18, -1e18, 0.0)
    for threshold in THRESHOLD_CANDIDATES:
        trades = simulate_extension_serial(records, predictions, threshold=float(threshold), model_seed=model_seed, strategy="threshold_selection")
        metrics = metrics_for_rows(pd.DataFrame(trades))
        key = (float(metrics["total_pnl"]), float(metrics["profit_factor_for_selection"]), -float(metrics["trades"]))
        rows.append({"validation_split": split_name, "threshold": float(threshold), **metrics})
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, rows


def simulate_extension_serial(
    records: Sequence[Any],
    predictions: dict[str, np.ndarray],
    *,
    threshold: float,
    model_seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity_by_seed: dict[int, float] = {}
    open_until_by_seed_session: dict[tuple[int, str], pd.Timestamp] = {}
    ordered = sorted(records, key=lambda r: (r.entry_seed, r.session, r.decision_ts, r.contract_id))
    for record in ordered:
        equity = equity_by_seed.get(record.entry_seed, STARTING_CASH)
        session_key = (record.entry_seed, record.session)
        if record.decision_ts < open_until_by_seed_session.get(session_key, pd.Timestamp.min.tz_localize("UTC")):
            continue
        if record.entry_premium <= 0.0 or record.entry_premium > equity:
            continue
        pred = predictions.get(record.uid)
        baseline_idx = baseline_index(record)
        if pred is None or len(pred) != len(record.path_pnl):
            exit_idx = baseline_idx
            extended = False
        else:
            tail = np.asarray(pred[baseline_idx:], dtype=float)
            eligible = np.where(tail <= float(threshold))[0]
            if len(eligible):
                exit_idx = baseline_idx + int(eligible[0])
            else:
                exit_idx = len(record.path_pnl) - 1
            extended = bool(exit_idx > baseline_idx)
        if exit_idx == baseline_idx:
            pnl = float(record.baseline_pnl)
            exit_ts = pd.Timestamp(record.baseline_exit_ts)
        else:
            pnl = float(record.path_pnl[exit_idx])
            exit_ts = pd.Timestamp(record.quote_times[exit_idx])
        combo_seed = int(model_seed * 100 + record.entry_seed)
        rows.append(
            {
                "reported_split": record.reported_split,
                "fold": record.fold,
                "model_seed": int(model_seed),
                "entry_seed": int(record.entry_seed),
                "combo_seed": combo_seed,
                "session": record.session,
                "decision_time": record.decision_ts.isoformat(),
                "exit_time": exit_ts.isoformat(),
                "contract_id": record.contract_id,
                "right": record.right,
                "offset": float(record.offset),
                "score": float(record.score),
                "threshold": float(threshold),
                "entry_ask": float(record.entry_ask),
                "entry_premium": float(record.entry_premium),
                "pnl": pnl,
                "account_equity_before": float(equity),
                "account_equity_after": float(equity + pnl),
                "exit_step": int(exit_idx),
                "baseline_exit_step": int(baseline_idx),
                "path_points": int(len(record.path_pnl)),
                "exit_reason": "extended_model_exit" if extended and exit_idx < len(record.path_pnl) - 1 else ("extended_forced_flat" if extended else record.baseline_exit_reason),
                "extended_after_baseline": bool(extended),
                "extension_minutes": max(0.0, (exit_ts - pd.Timestamp(record.baseline_exit_ts)).total_seconds() / 60.0),
                "baseline_exit_time": record.baseline_exit_ts.isoformat(),
                "baseline_pnl": float(record.baseline_pnl),
                "baseline_exit_reason": record.baseline_exit_reason,
                "strategy": strategy,
            }
        )
        equity_by_seed[record.entry_seed] = float(equity + pnl)
        open_until_by_seed_session[session_key] = exit_ts
    return rows


def summarize_extensions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    tmp = frame.copy()
    tmp["pnl"] = pd.to_numeric(tmp["pnl"], errors="coerce").fillna(0.0)
    tmp["baseline_pnl"] = pd.to_numeric(tmp["baseline_pnl"], errors="coerce").fillna(0.0)
    tmp["extended_after_baseline"] = tmp["extended_after_baseline"].astype(bool)
    tmp["extension_delta"] = tmp["pnl"] - tmp["baseline_pnl"]
    for split, group in tmp.groupby("reported_split", sort=True):
        ext = group[group["extended_after_baseline"]]
        rows.append(
            {
                "reported_split": str(split),
                "trades": int(len(group)),
                "extended_trades": int(len(ext)),
                "extended_fraction": float(len(ext) / max(len(group), 1)),
                "extension_delta_pnl": float(ext["extension_delta"].sum()) if len(ext) else 0.0,
                "median_extension_minutes": float(ext["extension_minutes"].median()) if len(ext) else 0.0,
            }
        )
    return rows


def decide(payload: dict[str, Any]) -> str:
    if any(int(payload["invariants"].get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_baseline_preserving_extension_invariant_failure"
    required = ["q1_2026", "march_2026", "recent_2026"]
    paper = {row["reported_split"]: row for row in payload["paper_default_comparison"]}
    base = {row["reported_split"]: row for row in payload["baseline_challenger_comparison"]}
    stress = {row["reported_split"]: row for row in payload["stress_0_10_per_side_summary"]}
    beats_paper = all(paper.get(split, {}).get("beats_baseline") for split in required)
    beats_base = all(base.get(split, {}).get("beats_baseline") for split in required)
    stress_positive = all(float(stress.get(split, {}).get("median_total_pnl", 0.0)) > 0.0 for split in required)
    if beats_paper and beats_base and stress_positive:
        return "research_candidate_baseline_preserving_extension_beats_paper_and_base"
    if beats_base:
        return "research_only_extension_improves_base_but_not_paper_gate"
    if beats_paper:
        return "research_only_extension_beats_paper_but_not_base"
    return "rejected_baseline_preserving_extension_no_clear_improvement"


def next_experiment(payload: dict[str, Any]) -> str:
    if payload["decision"].startswith("research_candidate"):
        return "Run churn/re-entry and extension attribution before any runtime parity discussion."
    if payload["decision"].startswith("research_only"):
        return "Attribute whether extensions added value only in recent_2026 or only in specific side/time buckets."
    return "Move to risk-adjusted utility on the original full-action objective; lifecycle replacement and extension did not beat the baseline."


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
    lines.extend(["", "## Extension Summary", ""])
    for row in payload["extension_summary"]:
        lines.append(
            f"- {row['reported_split']}: {row['extended_trades']} / {row['trades']} extended, "
            f"extension delta {money(row['extension_delta_pnl'])}, median extension {row['median_extension_minutes']:.1f}m"
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
            f"- Model trades: `{path.parent / 'baseline_preserving_lifecycle_extension_trades.csv'}`",
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
