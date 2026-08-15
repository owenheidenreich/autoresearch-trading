"""Materialize the unified serial DP oracle training-scope manifest."""
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

from v4.model.unified_serial_dp_oracle import ROLE_LABEL, build_serial_dp_oracle_scope


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_serial_dp_oracle")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_SERIAL_DP_ORACLE.md")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_HOLDING_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet")
DEFAULT_PATH_SKIPS = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/path_skips.csv")
DEFAULT_BASELINE_ACTIONS = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet")
DEFAULT_TRAINING_SPLITS = ("q1_2026", "q3_2025", "q4_2025", "recent_2026")

FLAT_COLUMNS = [
    "split",
    "session",
    "decision_time",
    "decision_dt",
    "candidate_uid",
    "contract_id",
    "entry_premium",
    "q_wait",
    "q_enter",
    "a_enter",
    "session_oracle_value",
    "best_enter_value_at_decision",
    "oracle_action",
    "oracle_action_uid",
]
HOLDING_COLUMNS = [
    "split",
    "session",
    "candidate_uid",
    "trade_uid",
    "entry_time",
    "state_time",
    "oracle_holding_action",
    "q_exit",
    "q_hold",
    "a_hold",
    "a_switch",
    "current_pnl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--holding-dataset", type=Path, default=DEFAULT_HOLDING_DATASET)
    parser.add_argument("--path-skips", type=Path, default=DEFAULT_PATH_SKIPS)
    parser.add_argument("--baseline-actions", type=Path, default=DEFAULT_BASELINE_ACTIONS)
    parser.add_argument("--training-splits", default=",".join(DEFAULT_TRAINING_SPLITS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flat = pd.read_parquet(args.flat_dataset, columns=FLAT_COLUMNS)
    holding = pd.read_parquet(args.holding_dataset, columns=HOLDING_COLUMNS)
    baseline_actions = pd.read_parquet(args.baseline_actions)
    path_skips = pd.read_csv(args.path_skips) if args.path_skips.exists() else pd.DataFrame()
    scope = build_serial_dp_oracle_scope(
        flat,
        holding,
        baseline_actions,
        path_skips,
        training_splits=parse_csv_arg(args.training_splits),
    )
    manifest = scope["manifest"]
    flat_events = scope["flat_events"]
    holding_manifest = scope["holding_trade_manifest"]
    session_manifest = scope["session_manifest"]
    flat_events_path = args.out_dir / "serial_dp_flat_events.parquet"
    holding_manifest_path = args.out_dir / "serial_dp_holding_trade_manifest.parquet"
    session_manifest_path = args.out_dir / "serial_dp_session_manifest.csv"
    flat_events.to_parquet(flat_events_path, index=False)
    holding_manifest.to_parquet(holding_manifest_path, index=False)
    session_manifest.to_csv(session_manifest_path, index=False)
    split_summary = summarize_by_split(flat_events, holding_manifest, session_manifest)
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "dataset foundation / unified serial DP oracle training-scope manifest",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "data_used": {
            "flat_dataset": str(args.flat_dataset),
            "holding_dataset": str(args.holding_dataset),
            "baseline_actions": str(args.baseline_actions),
            "path_skips": str(args.path_skips),
        },
        "manifest": manifest.to_dict(),
        "decision": manifest.decision,
        "split_summary": split_summary.to_dict("records"),
        "label_semantics": {
            "flat": "wait/enter candidate values from Protocol270 full-surface serial flat DP",
            "holding": "hold/exit values from Protocol274 holding-state opportunity-cost DP",
            "baseline": "Protocol101 seed-specific defer actions from the baseline-aligned training scope",
            "session_filter": "sessions with missing holding paths are excluded from the training scope",
        },
        "next_required_evidence": next_required_evidence(manifest.to_dict()),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "flat_events": str(flat_events_path),
            "holding_trade_manifest": str(holding_manifest_path),
            "session_manifest": str(session_manifest_path),
            "split_summary": str(args.out_dir / "split_summary.csv"),
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


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def summarize_by_split(flat_events: pd.DataFrame, holding_manifest: pd.DataFrame, session_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    splits = sorted(set(flat_events.get("split", pd.Series(dtype=str)).astype(str)))
    for split in splits:
        flat_split = flat_events[flat_events["split"].astype(str).eq(split)]
        holding_split = holding_manifest[holding_manifest["split"].astype(str).eq(split)]
        sessions = session_manifest[session_manifest["split"].astype(str).eq(split)]
        rows.append(
            {
                "split": split,
                "included_sessions": int(sessions["included"].astype(bool).sum()) if not sessions.empty else 0,
                "excluded_sessions": int((~sessions["included"].astype(bool)).sum()) if not sessions.empty else 0,
                "flat_events": int(len(flat_split)),
                "flat_oracle_enters": int(flat_split["oracle_action"].astype(str).eq("enter").sum()),
                "holding_trades": int(len(holding_split)),
                "holding_state_rows": int(holding_split["state_rows"].sum()) if "state_rows" in holding_split.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def next_required_evidence(manifest: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if manifest["decision"] != "unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope":
        if manifest["holding_coverage_of_oracle_entries"] < 1.0:
            items.append("Exclude or repair sessions whose oracle-enter paths lack holding-state labels.")
        if manifest["baseline_coverage_of_flat_events"] < 1.0:
            items.append("Attach Protocol101 baseline actions to every included flat event.")
        if manifest["disallowed_baseline_actions"]:
            items.append("Remove baseline_not_available rows from the training scope.")
    else:
        items.append("This oracle scope may be used for the first preregistered conservative neural training run.")
    return items


def render_report(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
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
        "## Manifest",
        "",
        f"- Training splits: `{manifest['training_splits']}`",
        f"- Included sessions: `{manifest['included_sessions']}`",
        f"- Excluded sessions with missing holding paths: `{manifest['excluded_sessions_with_missing_holding_paths']}`",
        f"- Flat candidate rows: `{manifest['flat_candidate_rows']}`",
        f"- Flat decision events: `{manifest['flat_decision_events']}`",
        f"- Flat oracle enter events: `{manifest['flat_oracle_enter_events']}`",
        f"- Holding state rows: `{manifest['holding_state_rows']}`",
        f"- Holding oracle trades: `{manifest['holding_oracle_trades']}`",
        f"- Baseline event-seed rows: `{manifest['baseline_event_seed_rows']}`",
        f"- Baseline seed count: `{manifest['baseline_seed_count']}`",
        f"- Holding coverage of oracle entries: `{manifest['holding_coverage_of_oracle_entries']:.6f}`",
        f"- Baseline coverage of flat events: `{manifest['baseline_coverage_of_flat_events']:.6f}`",
        f"- Disallowed baseline actions: `{manifest['disallowed_baseline_actions']}`",
        "",
        "## Split Summary",
        "",
        "| split | included sessions | excluded sessions | flat events | oracle enters | holding trades | holding rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['split']} | {row['included_sessions']} | {row['excluded_sessions']} | {row['flat_events']} | "
            f"{row['flat_oracle_enters']} | {row['holding_trades']} | {row['holding_state_rows']} |"
        )
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
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
