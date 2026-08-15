"""EXP_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1.

Historically Protocol265. Protocol263/264 showed the current lifecycle learners
beat PAPER_DEFAULT_PROTOCOL101, but give back recent 2026 edge versus the frozen
Protocol261 stream by exiting early and suppressing later base-stream trades.

This experiment keeps the Protocol261 entry stream and original baseline exit
as an anchor. The neural lifecycle model may extend a trade beyond that anchor,
but it cannot exit earlier. This directly tests the trader hypothesis: preserve
the base trade, then only let the model continue when the post-entry path still
looks worth holding.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

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
    serial_invariants,
    simulate_baseline_serial,
    summarize_replay,
    threshold_summary,
    train_model,
)
from v4.scripts.run_protocol207_lifecycle_context_calibration import (
    PAPER_DEFAULT_TRADES,
    compare_against_named_baseline,
    load_paper_default_trades,
    summarize_stress,
)
from v4.scripts.run_protocol251_premium_blend_slot_aware_lifecycle import (
    load_premium_blend_candidates,
    safe_find_normalized_path,
)


ROLE_LABEL = "EXP_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1"
HISTORICAL_ID = "Protocol265"
CANDIDATE_LABEL = "CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1"
BASE_CHALLENGER_LABEL = "CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration/model_trades.csv")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation")


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p200.find_normalized_path = safe_find_normalized_path
    candidates = load_premium_blend_candidates(args.trades)
    records, path_skips = build_path_records(candidates, normalized_dir=args.normalized_dir, forced_flat_time=str(args.forced_flat_time))
    if not records:
        raise SystemExit("no executable source-penalty lifecycle path records were built")

    baseline_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    for spec in fold_specs():
        train_records = [record for record in records if record.reported_split in spec["train_splits"]]
        validation_records = [record for record in records if record.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_x, train_y, scaler = fit_training_matrix(train_records, max_train_steps=int(args.max_train_steps), seed=265)
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
            threshold, sweep = select_threshold(validation_records, validation_predictions, split_name=str(spec["validation_split"]), model_seed=int(model_seed))
            model_dir = args.out_dir / "model_artifacts" / str(spec["fold"]) / f"seed_{model_seed}"
            save_model_artifact(
                model_dir=model_dir,
                model=model,
                scaler=scaler,
                spec=spec,
                model_seed=int(model_seed),
                threshold=float(threshold),
                history=history,
                args=args,
                train_records=len(train_records),
                validation_records=len(validation_records),
                train_steps_used=len(train_y),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(model_seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [record for record in records if record.reported_split == split]
                predictions = predict_records(model, scaler, split_records)
                model_rows.extend(
                    simulate_anchor_serial(
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
                        for row in simulate_anchor_serial(
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
        "what_is_this": "experiment / baseline-anchored lifecycle continuation model",
        "changes_paper_default": False,
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "baseline_challenger_label": BASE_CHALLENGER_LABEL,
        "data_used": {
            "source_penalty_trades": str(args.trades),
            "normalized_dir": str(args.normalized_dir),
            "paper_default_trades": str(args.paper_default_trades),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "pre_registration": {
            "hypothesis": "Preserve the Protocol261 baseline exit, then let the lifecycle model continue only when the post-entry path remains worth holding.",
            "entry_stream": "frozen Protocol261 source-penalty router trades",
            "holding_action_space": "baseline exit or model-extended hold",
            "threshold_selection": "validation split only",
            "starting_cash": STARTING_CASH,
            "no_early_model_exit": True,
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
        "extension_summary": extension_summary,
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": "",
        "next_experiment": "",
    }
    payload["decision"] = decide(payload)
    payload["next_experiment"] = next_experiment(payload)
    payload["artifact_manifest"] = write_artifact_manifest(args.out_dir)
    baseline_frame.to_csv(args.out_dir / "source_penalty_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "source_penalty_baseline_anchored_continuation_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def save_model_artifact(
    *,
    model_dir: Path,
    model: Any,
    scaler: Any,
    spec: dict[str, Any],
    model_seed: int,
    threshold: float,
    history: list[dict[str, float]],
    args: argparse.Namespace,
    train_records: int,
    validation_records: int,
    train_steps_used: int,
) -> None:
    """Persist a Protocol265 fold/seed model as a reproducible artifact."""

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    scaler_path = model_dir / "scaler.json"
    manifest_path = model_dir / "manifest.json"
    import torch

    torch.save(model.state_dict(), model_path)
    scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
    manifest = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "candidate_label": CANDIDATE_LABEL,
        "baseline_challenger_label": BASE_CHALLENGER_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "model_class": "ContinuationMLP",
        "feature_columns": list(p200.FEATURE_COLUMNS),
        "fold": spec["fold"],
        "train_splits": list(spec["train_splits"]),
        "validation_split": spec["validation_split"],
        "test_splits": list(spec["test_splits"]),
        "seed": int(model_seed),
        "threshold": float(threshold),
        "history": history,
        "config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "hidden_dim": int(args.hidden_dim),
            "max_train_steps": int(args.max_train_steps),
            "learning_rate": float(args.learning_rate),
            "forced_flat_time": str(args.forced_flat_time),
            "starting_cash": float(STARTING_CASH),
            "contract_multiplier": float(p200.CONTRACT_MULTIPLIER),
            "target_scale": float(p200.TARGET_SCALE),
            "target_clip": float(p200.TARGET_CLIP),
            "risk_penalty": float(p200.RISK_PENALTY),
            "threshold_candidates": [float(value) for value in p200.THRESHOLD_CANDIDATES],
        },
        "data_used": {
            "source_penalty_trades": str(args.trades),
            "normalized_dir": str(args.normalized_dir),
        },
        "training_row_counts": {
            "train_records": int(train_records),
            "validation_records": int(validation_records),
            "train_steps_used": int(train_steps_used),
        },
        "policy_semantics": {
            "entry_stream": "frozen Protocol261 source-penalty router trades",
            "entry_action_changed": False,
            "lifecycle_action": "baseline exit or model-extended hold",
            "no_early_model_exit": True,
            "entry_price": "ask",
            "exit_price": "bid",
            "max_contracts": 1,
            "max_concurrent_positions": 1,
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")


def write_artifact_manifest(out_dir: Path) -> dict[str, Any]:
    """Write a run-level manifest with hashes for every saved Protocol265 artifact."""

    artifact_root = out_dir / "model_artifacts"
    files: list[dict[str, Any]] = []
    if artifact_root.exists():
        for path in sorted(artifact_root.rglob("*")):
            if path.is_file():
                files.append({"path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)})
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "candidate_label": CANDIDATE_LABEL,
        "artifact_root": str(artifact_root),
        "files": files,
    }
    (out_dir / "artifact_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_threshold(
    records: Sequence[Any],
    predictions: dict[str, np.ndarray],
    *,
    split_name: str,
    model_seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = float(p200.THRESHOLD_CANDIDATES[0])
    best_key = (-1e18, -1e18, 0.0)
    for threshold in p200.THRESHOLD_CANDIDATES:
        trades = simulate_anchor_serial(records, predictions, threshold=float(threshold), model_seed=model_seed, strategy="threshold_selection")
        item = metrics_for_rows(pd.DataFrame(trades))
        key = (float(item["total_pnl"]), float(item["profit_factor_for_selection"]), -float(item["trades"]))
        rows.append({"validation_split": split_name, "threshold": float(threshold), **item})
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, rows


def simulate_anchor_serial(
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
        if pred is None or len(pred) != len(record.path_pnl):
            continue
        anchor_idx = baseline_anchor_idx(record)
        exit_idx = anchored_exit_index(pred, threshold, anchor_idx)
        extended = exit_idx > anchor_idx
        pnl = float(record.path_pnl[exit_idx]) if extended else float(record.baseline_pnl)
        exit_ts = pd.Timestamp(record.quote_times[exit_idx]) if extended else pd.Timestamp(record.baseline_exit_ts)
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
                "baseline_anchor_idx": int(anchor_idx),
                "path_points": int(len(record.path_pnl)),
                "exit_reason": "model_extended_exit" if extended else "baseline_anchor_exit",
                "extended_beyond_baseline": bool(extended),
                "predicted_continuation_value": float(pred[exit_idx]),
                "baseline_exit_time": record.baseline_exit_ts.isoformat(),
                "baseline_pnl": float(record.baseline_pnl),
                "baseline_exit_reason": record.baseline_exit_reason,
                "strategy": strategy,
            }
        )
        equity_by_seed[record.entry_seed] = float(equity + pnl)
        open_until_by_seed_session[session_key] = exit_ts
    return rows


def baseline_anchor_idx(record: Any) -> int:
    times = pd.to_datetime(record.quote_times, utc=True, format="ISO8601")
    values = np.array([pd.Timestamp(value).value for value in times], dtype=np.int64)
    idx = int(np.searchsorted(values, pd.Timestamp(record.baseline_exit_ts).value, side="left"))
    return min(max(idx, 0), len(record.path_pnl) - 1)


def anchored_exit_index(prediction: np.ndarray, threshold: float, anchor_idx: int) -> int:
    tail = np.asarray(prediction, dtype=float)[anchor_idx:]
    eligible = np.where(tail <= float(threshold))[0]
    if len(eligible):
        return int(anchor_idx + eligible[0])
    return int(len(prediction) - 1)


def summarize_extensions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for (split, reason), group in frame.groupby(["reported_split", "exit_reason"], sort=True):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        base = pd.to_numeric(group["baseline_pnl"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "reported_split": str(split),
                "exit_reason": str(reason),
                "rows": int(len(group)),
                "pnl": float(pnl.sum()),
                "baseline_pnl": float(base.sum()),
                "delta_vs_baseline_path": float((pnl - base).sum()),
                "median_extra_steps": float((pd.to_numeric(group["exit_step"], errors="coerce") - pd.to_numeric(group["baseline_anchor_idx"], errors="coerce")).median()),
            }
        )
    return rows


def decide(payload: dict[str, Any]) -> str:
    if any(int(payload["invariants"].get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_source_penalty_baseline_anchor_invariant_failure"
    required = ["q1_2026", "march_2026", "recent_2026"]
    paper = {row["reported_split"]: row for row in payload["paper_default_comparison"]}
    base = {row["reported_split"]: row for row in payload["baseline_challenger_comparison"]}
    stress = {row["reported_split"]: row for row in payload["stress_0_10_per_side_summary"]}
    beats_paper = all(paper.get(split, {}).get("beats_baseline") for split in required)
    beats_base = all(base.get(split, {}).get("beats_baseline") for split in required)
    stress_positive = all(float(stress.get(split, {}).get("median_total_pnl", 0.0)) > 0.0 for split in required)
    if beats_paper and beats_base and stress_positive:
        return "research_candidate_source_penalty_baseline_anchor_beats_paper_and_base"
    if beats_paper:
        return "research_only_source_penalty_baseline_anchor_beats_paper_but_not_base"
    return "rejected_source_penalty_baseline_anchor_does_not_beat_paper_default"


def next_experiment(payload: dict[str, Any]) -> str:
    if payload["decision"].startswith("research_candidate"):
        return "Attribute extension behavior and run no-order runtime parity before any paper-default replacement discussion."
    if payload["decision"].startswith("research_only"):
        return "If anchored extension fails recent/base, test learned reduce/scale-out or return to serial full-action entry selection."
    return "Reject baseline-anchored continuation and investigate whether Protocol261 base exits are already close to the local optimum."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Baseline challenger: {payload['baseline_challenger_label']}",
        f"Data used: `{payload['data_used']['source_penalty_trades']}` plus normalized quote paths",
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
    lines.extend(["", "## Protocol261 Baseline Comparison", "", "| split | model | base | delta | model PF | base PF |", "|---|---:|---:|---:|---:|---:|"])
    for row in payload["baseline_challenger_comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | {money(row['baseline_median_total_pnl'])} | "
            f"{money(row['delta_vs_baseline'])} | {row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} |"
        )
    lines.extend(["", "## Extension Summary", "", "| split | exit reason | rows | pnl | baseline pnl | delta | median extra steps |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in payload["extension_summary"]:
        lines.append(
            f"| {row['reported_split']} | {row['exit_reason']} | {row['rows']} | {money(row['pnl'])} | "
            f"{money(row['baseline_pnl'])} | {money(row['delta_vs_baseline_path'])} | {row['median_extra_steps']:.0f} |"
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
            f"- Model trades: `{path.parent / 'source_penalty_baseline_anchored_continuation_trades.csv'}`",
            f"- Baseline trades: `{path.parent / 'source_penalty_baseline_serial_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    if marker in ledger.read_text():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    f"- Candidate: {payload['candidate_label']}",
                    f"- Baseline: {payload['paper_default_label']}; secondary baseline {payload['baseline_challenger_label']}",
                    f"- Data used: `{payload['data_used']['source_penalty_trades']}` plus normalized quote paths",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
