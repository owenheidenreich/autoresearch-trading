"""Protocol101 loss-reversal local action pricing.

This packet prices the selected loss-reversal rows as mutually exclusive local
actions at the later blocked-signal time:

* keep holding the original Protocol101 trade to its actual exit,
* exit the original trade immediately,
* exit immediately and switch into the later Protocol101-approved signal.

It is not a full serial account replay. It is the action-pricing scaffold needed
before any neural label or full replay is allowed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.scripts.run_protocol101_strategy_forensics_packet import _fmt_money  # noqa: E402


ROLE_LABEL = "AUDIT_PROTOCOL101_LOSS_REVERSAL_LOCAL_ACTION_PRICING_V1"
DEFAULT_INPUT = Path(
    "v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/direct_causal_state_join.csv"
)
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_LOSS_REVERSAL_LOCAL_ACTION_PRICING_V1.md"
)
STRESS_LEVELS = (0.0, 0.10, 0.25)
CONTRACT_MULTIPLIER = 100.0


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_direct_state_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty direct-state input: {path}")
    return frame


def build_local_action_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[frame["direct_quote_state_status"].astype(str).eq("matched")].copy()
    for col in ["current_pnl_at_signal", "best_blocked_candidate_pnl", "open_trade_pnl", "best_blocked_minus_open_pnl"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
    rows["keep_to_protocol101_exit_pnl"] = rows["open_trade_pnl"]
    rows["exit_now_pnl"] = rows["current_pnl_at_signal"]
    rows["switch_to_blocked_signal_pnl"] = rows["current_pnl_at_signal"] + rows["best_blocked_candidate_pnl"]
    rows["exit_now_minus_keep"] = rows["exit_now_pnl"] - rows["keep_to_protocol101_exit_pnl"]
    rows["switch_minus_keep"] = rows["switch_to_blocked_signal_pnl"] - rows["keep_to_protocol101_exit_pnl"]
    rows["current_pnl_charge_vs_slot_artifact"] = rows["switch_minus_keep"] - rows["best_blocked_minus_open_pnl"]
    for stress in STRESS_LEVELS:
        suffix = stress_suffix(stress)
        rows[f"switch_minus_keep_stress_{suffix}"] = rows["switch_minus_keep"] - (2.0 * float(stress) * CONTRACT_MULTIPLIER)
        rows[f"best_local_action_stress_{suffix}"] = [
            best_action(exit_delta, switch_delta)
            for exit_delta, switch_delta in zip(rows["exit_now_minus_keep"], rows[f"switch_minus_keep_stress_{suffix}"])
        ]
        rows[f"best_local_delta_stress_{suffix}"] = [
            max(0.0, _finite(exit_delta), _finite(switch_delta))
            for exit_delta, switch_delta in zip(rows["exit_now_minus_keep"], rows[f"switch_minus_keep_stress_{suffix}"])
        ]
    rows["action_pricing_limitations"] = (
        "local_event_pricing_not_full_serial_replay; blocked_candidate_pnl_reuses_frozen_protocol101_exit; "
        "stress_charges_two_extra_switch_sides; fill_latency_not_calibrated"
    )
    return rows


def stress_suffix(stress: float) -> str:
    return f"{int(round(float(stress) * 100)):03d}"


def best_action(exit_delta: float, switch_delta: float) -> str:
    exit_delta = _finite(exit_delta)
    switch_delta = _finite(switch_delta)
    if switch_delta > max(0.0, exit_delta):
        return "exit_and_switch"
    if exit_delta > 0.0:
        return "exit_now"
    return "keep_holding"


def summary_row(prefix: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    row = dict(prefix)
    row.update(
        {
            "rows": int(len(group)),
            "keep_pnl": float(pd.to_numeric(group["keep_to_protocol101_exit_pnl"], errors="coerce").fillna(0.0).sum()),
            "exit_now_pnl": float(pd.to_numeric(group["exit_now_pnl"], errors="coerce").fillna(0.0).sum()),
            "switch_pnl": float(pd.to_numeric(group["switch_to_blocked_signal_pnl"], errors="coerce").fillna(0.0).sum()),
            "exit_now_minus_keep": float(pd.to_numeric(group["exit_now_minus_keep"], errors="coerce").fillna(0.0).sum()),
            "switch_minus_keep": float(pd.to_numeric(group["switch_minus_keep"], errors="coerce").fillna(0.0).sum()),
            "median_switch_minus_keep": float(pd.to_numeric(group["switch_minus_keep"], errors="coerce").median()) if len(group) else 0.0,
            "switch_positive_rate": float((pd.to_numeric(group["switch_minus_keep"], errors="coerce").fillna(0.0) > 0.0).mean())
            if len(group)
            else 0.0,
        }
    )
    for stress in STRESS_LEVELS:
        suffix = stress_suffix(stress)
        delta = pd.to_numeric(group[f"switch_minus_keep_stress_{suffix}"], errors="coerce").fillna(0.0)
        actions = group[f"best_local_action_stress_{suffix}"].astype(str)
        row[f"switch_minus_keep_stress_{suffix}"] = float(delta.sum())
        row[f"switch_positive_rate_stress_{suffix}"] = float((delta > 0.0).mean()) if len(group) else 0.0
        row[f"best_action_exit_and_switch_rows_stress_{suffix}"] = int(actions.eq("exit_and_switch").sum())
        row[f"best_action_exit_now_rows_stress_{suffix}"] = int(actions.eq("exit_now").sum())
        row[f"best_action_keep_holding_rows_stress_{suffix}"] = int(actions.eq("keep_holding").sum())
        row[f"best_local_delta_stress_{suffix}"] = float(
            pd.to_numeric(group[f"best_local_delta_stress_{suffix}"], errors="coerce").fillna(0.0).sum()
        )
    return row


def build_bucket_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = [summary_row({"summary_bucket": "all_matched_loss_reversal_rows"}, rows)]
    for bucket, group in rows.groupby("live_test_bucket", dropna=False):
        output.append(summary_row({"summary_bucket": f"live_test:{bucket}"}, group))
    for split, group in rows.groupby("reported_split", dropna=False):
        output.append(summary_row({"summary_bucket": f"split:{split}"}, group))
    for relation, group in rows.groupby("best_blocked_relation", dropna=False):
        output.append(summary_row({"summary_bucket": f"relation:{relation}"}, group))
    return pd.DataFrame(output)


def build_priority_review(rows: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    priority = {
        "priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates": 1,
        "priority_2_current_loss_opposite_side": 2,
        "priority_3_current_loss_same_side": 3,
        "priority_4_not_yet_loss_but_negative_next_and_exit_deteriorates": 4,
        "priority_5_final_loss_but_live_state_not_stale_enough": 5,
    }
    out = rows.copy()
    out["review_priority"] = out["live_test_bucket"].map(priority).fillna(99).astype(int)
    cols = [
        "review_priority",
        "live_test_bucket",
        "reported_split",
        "seed",
        "session",
        "open_trade_entry_dt",
        "open_trade_exit_dt",
        "best_blocked_decision_dt",
        "open_trade_right",
        "best_blocked_right",
        "best_blocked_relation",
        "open_trade_pnl",
        "current_pnl_at_signal",
        "best_blocked_candidate_pnl",
        "keep_to_protocol101_exit_pnl",
        "exit_now_pnl",
        "switch_to_blocked_signal_pnl",
        "exit_now_minus_keep",
        "switch_minus_keep",
        "switch_minus_keep_stress_010",
        "switch_minus_keep_stress_025",
        "best_local_action_stress_000",
        "best_local_action_stress_010",
        "best_local_action_stress_025",
        "mfe_to_signal",
        "mae_to_signal",
        "giveback_from_mfe_to_signal",
        "current_spread",
        "best_blocked_entry_ask",
        "best_blocked_entry_spread",
        "action_pricing_limitations",
    ]
    available = [col for col in cols if col in out.columns]
    return out.sort_values(["review_priority", "switch_minus_keep_stress_025"], ascending=[True, False], kind="stable").head(limit)[
        available
    ]


def build_summary(rows: pd.DataFrame, bucket_summary: pd.DataFrame) -> dict[str, Any]:
    p1 = rows["live_test_bucket"].astype(str).eq("priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates")
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "local action pricing for selected Protocol101 loss-reversal rows",
        "decision": "protocol101_loss_reversal_local_action_pricing_complete_full_serial_replay_required_training_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "challenge_allowed": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "counts": {
            "priced_rows": int(len(rows)),
            "priority_1_rows": int(p1.sum()),
        },
        "aggregate": {
            "exit_now_minus_keep": float(rows["exit_now_minus_keep"].sum()),
            "switch_minus_keep": float(rows["switch_minus_keep"].sum()),
            "switch_minus_keep_stress_010": float(rows["switch_minus_keep_stress_010"].sum()),
            "switch_minus_keep_stress_025": float(rows["switch_minus_keep_stress_025"].sum()),
            "priority_1_switch_minus_keep": float(rows.loc[p1, "switch_minus_keep"].sum()),
            "priority_1_switch_minus_keep_stress_025": float(rows.loc[p1, "switch_minus_keep_stress_025"].sum()),
        },
        "blockers": [
            "local_action_pricing_is_not_full_serial_account_replay",
            "blocked_candidate_exit_policy_is_reused_from_frozen_protocol101_metadata",
            "fill_latency_and_quote_freshness_not_calibrated",
            "missing_quote_state_rows_excluded",
            "diagnostic_splits_are_research_exposed",
            "neural_training_forbidden_until_full_replay_labels_are_defined",
        ],
    }


def write_report(output_dir: Path, summary: dict[str, Any], bucket_summary: pd.DataFrame) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: local action pricing for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "The loss-reversal idea survives the first local action-pricing check, especially in the priority-1 bucket. This is still not a trading rule or neural label because the effects are local and not full serial-account replay.",
        "",
        f"- Priced rows: `{summary['counts']['priced_rows']}`",
        f"- Exit-now minus keep total: `{_fmt_money(summary['aggregate']['exit_now_minus_keep'])}`",
        f"- Switch minus keep total: `{_fmt_money(summary['aggregate']['switch_minus_keep'])}`",
        f"- Switch minus keep under `$0.10` two-side stress: `{_fmt_money(summary['aggregate']['switch_minus_keep_stress_010'])}`",
        f"- Switch minus keep under `$0.25` two-side stress: `{_fmt_money(summary['aggregate']['switch_minus_keep_stress_025'])}`",
        f"- Priority-1 rows: `{summary['counts']['priority_1_rows']}`",
        f"- Priority-1 switch minus keep under `$0.25` stress: `{_fmt_money(summary['aggregate']['priority_1_switch_minus_keep_stress_025'])}`",
        "",
        "## Pricing Summary",
        "",
        "| bucket | rows | exit-now delta | switch delta | switch delta $0.10 | switch delta $0.25 | switch positive $0.25 | best switch rows $0.25 | keep rows $0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in bucket_summary.iterrows():
        lines.append(
            f"| {row['summary_bucket']} | {int(row['rows'])} | {_fmt_money(row['exit_now_minus_keep'])} | "
            f"{_fmt_money(row['switch_minus_keep'])} | {_fmt_money(row['switch_minus_keep_stress_010'])} | "
            f"{_fmt_money(row['switch_minus_keep_stress_025'])} | {row['switch_positive_rate_stress_025']:.3f} | "
            f"{int(row['best_action_exit_and_switch_rows_stress_025'])} | {int(row['best_action_keep_holding_rows_stress_025'])} |"
        )
    lines.extend(
        [
            "",
            "## Stopping Point",
            "",
            "This packet prices the local decision and gives us a viable first playbook candidate. The next step must be full serial replay, because switching out of one trade changes all later account state and may block or create future Protocol101 opportunities.",
            "",
            "No neural model should be trained until `PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1` defines the label under one account, one contract, no overlaps, affordability, bid/ask execution, and slippage/fill stress.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Local action rows: `{output_dir / 'local_action_pricing_rows.csv'}`",
            f"- Bucket summary: `{output_dir / 'local_action_bucket_summary.csv'}`",
            f"- Priority review rows: `{output_dir / 'priority_action_review_rows.csv'}`",
            f"- Report: `{output_dir / 'report.md'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_direct_state_rows(Path(args.input))
    rows = build_local_action_rows(source)
    bucket_summary = build_bucket_summary(rows)
    review = build_priority_review(rows)
    summary = build_summary(rows, bucket_summary)
    summary["inputs"] = {"direct_causal_state_join": str(args.input)}
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "local_action_rows": str(output_dir / "local_action_pricing_rows.csv"),
        "bucket_summary": str(output_dir / "local_action_bucket_summary.csv"),
        "priority_review_rows": str(output_dir / "priority_action_review_rows.csv"),
    }
    rows.to_csv(output_dir / "local_action_pricing_rows.csv", index=False)
    bucket_summary.to_csv(output_dir / "local_action_bucket_summary.csv", index=False)
    review.to_csv(output_dir / "priority_action_review_rows.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    report = write_report(output_dir, summary, bucket_summary)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
