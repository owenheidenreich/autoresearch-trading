"""Attach Protocol101 baseline actions to unified trajectory events."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.protocol101_baseline_attachment import (
    ROLE_LABEL,
    build_protocol101_baseline_event_actions,
    summarize_protocol101_baseline_attachment,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment")
DEFAULT_DOC_PATH = Path(
    "v4/docs/protocol101/training/research/UNIFIED_PROTOCOL101_BASELINE_ATTACHMENT.md"
)
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_BASELINE_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_209_unified_entry_lifecycle_sequence/paper_default_protocol101_trades.csv")
DEFAULT_TRAINING_SCOPE_SPLITS = ("q1_2026", "q3_2025", "q4_2025", "recent_2026")
DEFAULT_BASELINE_ALIAS_SPLITS = ("march_2026",)

FLAT_COLUMNS = ["split", "session", "decision_time", "decision_dt", "candidate_uid", "contract_id"]
BASELINE_COLUMNS = ["reported_split", "seed", "session", "decision_time", "exit_time", "contract_id", "candidate_uid", "pnl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--baseline-trades", type=Path, default=DEFAULT_BASELINE_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--training-splits", default=",".join(DEFAULT_TRAINING_SCOPE_SPLITS))
    parser.add_argument("--baseline-alias-splits", default=",".join(DEFAULT_BASELINE_ALIAS_SPLITS))
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flat = pd.read_parquet(args.flat_dataset, columns=FLAT_COLUMNS)
    baseline = pd.read_csv(args.baseline_trades, usecols=BASELINE_COLUMNS)
    actions = build_protocol101_baseline_event_actions(flat, baseline)
    summary = summarize_protocol101_baseline_attachment(actions, baseline)
    training_splits = parse_csv_arg(args.training_splits)
    alias_splits = parse_csv_arg(args.baseline_alias_splits)
    training_actions = actions[actions["split"].astype(str).isin(training_splits)].copy()
    training_baseline = baseline[
        baseline["reported_split"].astype(str).isin(training_splits)
        & ~baseline["reported_split"].astype(str).isin(alias_splits)
    ].copy()
    training_summary = summarize_protocol101_baseline_attachment(training_actions, training_baseline)
    out_path = args.out_dir / "protocol101_baseline_event_actions.parquet"
    actions.to_parquet(out_path, index=False)
    training_out_path = args.out_dir / "protocol101_baseline_event_actions_training_scope.parquet"
    training_actions.to_parquet(training_out_path, index=False)
    action_counts = pd.DataFrame(
        [{"protocol101_action": key, "rows": value} for key, value in summary.action_counts.items()]
    )
    action_counts.to_csv(args.out_dir / "action_counts.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "dataset foundation / Protocol101 baseline action attachment",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "data_used": {"flat_dataset": str(args.flat_dataset), "baseline_trades": str(args.baseline_trades)},
        "output_dataset": str(out_path),
        "summary": summary.to_dict(),
        "training_scope": {
            "included_splits": list(training_splits),
            "excluded_flat_splits": sorted(set(summary.splits_with_flat_events) - set(training_splits)),
            "baseline_alias_splits_excluded": list(alias_splits),
            "event_actions": str(training_out_path),
            "summary": training_summary.to_dict(),
            "decision": decide(training_summary),
        },
        "decision": decide_with_training_scope(summary, training_summary),
        "next_required_evidence": next_required_evidence(summary, training_summary),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "event_actions": str(out_path),
            "training_scope_event_actions": str(training_out_path),
            "action_counts": str(args.out_dir / "action_counts.csv"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }
    write_json(args.out_dir / "summary.json", payload)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def decide(summary: Any) -> str:
    if summary.flat_events <= 0 or summary.baseline_trades <= 0:
        return "protocol101_baseline_attachment_blocked_missing_inputs"
    if summary.splits_without_baseline:
        return "protocol101_baseline_attachment_partial_missing_trajectory_split_baselines"
    if summary.baseline_splits_without_flat_events:
        return "protocol101_baseline_attachment_partial_baseline_split_not_in_trajectory"
    if summary.baseline_trades_without_event:
        return "protocol101_baseline_attachment_partial_baseline_trades_without_events"
    if summary.unmatched_entry_trades:
        return "protocol101_baseline_attachment_partial_unmatched_surface_entries"
    return "protocol101_baseline_attachment_ready_all_trajectory_splits"


def decide_with_training_scope(summary: Any, training_summary: Any) -> str:
    full_decision = decide(summary)
    training_decision = decide(training_summary)
    if training_decision == "protocol101_baseline_attachment_ready_all_trajectory_splits":
        if full_decision == training_decision:
            return full_decision
        return "protocol101_baseline_attachment_ready_for_baseline_aligned_training_scope"
    return full_decision


def next_required_evidence(summary: Any, training_summary: Any) -> list[str]:
    items: list[str] = []
    if summary.splits_without_baseline:
        items.append(f"Add Protocol101 baseline actions for trajectory splits without baseline coverage: {list(summary.splits_without_baseline)}.")
    if summary.baseline_splits_without_flat_events:
        items.append(
            f"Reconcile baseline splits that are not represented as trajectory splits: {list(summary.baseline_splits_without_flat_events)}."
        )
    if summary.baseline_trades_without_event:
        items.append("Explain baseline trades that did not land on a full-surface trajectory decision event.")
    if summary.unmatched_entry_trades:
        items.append("Explain or repair baseline entry trades whose contracts are absent from the full candidate surface.")
    training_decision = decide(training_summary)
    if training_decision == "protocol101_baseline_attachment_ready_all_trajectory_splits":
        items.append("Use the baseline-aligned training scope for conservative neural training; keep excluded splits diagnostic until baseline coverage exists.")
    if not items:
        items.append("Use the event-level baseline actions as the conservative defer action in the unified training dataset.")
    return items


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    training = payload["training_scope"]
    training_summary = training["summary"]
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Coverage",
        "",
        f"- Flat events: `{summary['flat_events']}`",
        f"- Event-seed rows: `{summary['event_seed_rows']}`",
        f"- Baseline trades: `{summary['baseline_trades']}`",
        f"- Matched entry trades: `{summary['matched_entry_trades']}`",
        f"- Unmatched entry trades: `{summary['unmatched_entry_trades']}`",
        f"- Baseline trades without event: `{summary['baseline_trades_without_event']}`",
        f"- Splits with flat events: `{summary['splits_with_flat_events']}`",
        f"- Splits with baseline: `{summary['splits_with_baseline']}`",
        f"- Splits without baseline: `{summary['splits_without_baseline']}`",
        f"- Baseline splits without flat events: `{summary['baseline_splits_without_flat_events']}`",
        "",
        "## Baseline-Aligned Training Scope",
        "",
        f"- Included splits: `{training['included_splits']}`",
        f"- Excluded flat splits: `{training['excluded_flat_splits']}`",
        f"- Baseline alias splits excluded: `{training['baseline_alias_splits_excluded']}`",
        f"- Scope decision: `{training['decision']}`",
        f"- Scope flat events: `{training_summary['flat_events']}`",
        f"- Scope event-seed rows: `{training_summary['event_seed_rows']}`",
        f"- Scope baseline trades: `{training_summary['baseline_trades']}`",
        f"- Scope matched entry trades: `{training_summary['matched_entry_trades']}`",
        f"- Scope unmatched entry trades: `{training_summary['unmatched_entry_trades']}`",
        f"- Scope baseline trades without event: `{training_summary['baseline_trades_without_event']}`",
        "",
        "## Action Counts",
        "",
        "| action | rows |",
        "|---|---:|",
    ]
    for action, rows in summary["action_counts"].items():
        lines.append(f"| `{action}` | {rows} |")
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Report: `{payload['outputs']['report']}`",
            f"- Event actions: `{payload['outputs']['event_actions']}`",
            f"- Training-scope event actions: `{payload['outputs']['training_scope_event_actions']}`",
            f"- Action counts: `{payload['outputs']['action_counts']}`",
            f"- Docs copy: `{payload['outputs']['doc']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
    text = ledger.read_text()
    if marker in text:
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
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
