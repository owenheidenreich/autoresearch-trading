"""Protocol101 loss-reversal exit-gate diagnostic.

This packet tests whether the selected Track A hypothesis is causally testable:
Protocol101 final losers sometimes occupy the only slot while a later
Protocol101-approved signal appears. The crucial audit question is whether the
trade was already losing or stale at the later signal time, not merely at final
exit. No model is trained.
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
from v4.scripts.run_protocol101_strategy_selection_packet import attach_best_blocked_event  # noqa: E402


ROLE_LABEL = "AUDIT_PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_DIAGNOSTIC_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_DIAGNOSTIC_V1.md"
)
DEFAULT_SLOT_COST_DIR = Path("v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1")
DEFAULT_HOLD_EXIT_DATASET = Path(
    "v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/protocol101_hold_exit_action_advantage.parquet"
)


HOLD_STATE_CAUSAL_COLUMNS = [
    "reported_split",
    "seed",
    "session",
    "candidate_uid",
    "contract_id",
    "state_time",
    "minutes_since_entry",
    "minutes_to_protocol101_exit",
    "minutes_to_forced_flat",
    "bid",
    "ask",
    "mid",
    "spread",
    "underlying_price",
    "current_pnl",
    "mfe_to_now",
    "mae_to_now",
    "giveback_from_mfe",
    "giveback_fraction",
    "pnl_velocity_1",
    "pnl_velocity_3",
    "pnl_velocity_5",
    "time_since_mfe_minutes",
]

HOLD_STATE_DIAGNOSTIC_LABEL_COLUMNS = [
    "a_hold",
    "a_exit",
    "a_hold_one_step_realized",
    "q_exit",
    "q_hold",
    "oracle_holding_action",
    "oracle_one_step_action",
]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty required csv: {path}")
    return frame


def load_loss_reversal_candidates(slot_cost_dir: Path) -> pd.DataFrame:
    open_slots = _read_required_csv(slot_cost_dir / "enriched_open_trade_slot_summary.csv")
    blocked = _read_required_csv(slot_cost_dir / "enriched_blocked_slot_events.csv")
    rows = attach_best_blocked_event(open_slots, blocked)
    open_pnl = pd.to_numeric(rows["open_trade_pnl"], errors="coerce").fillna(0.0)
    blocked_pnl = pd.to_numeric(rows["best_blocked_candidate_pnl"], errors="coerce").fillna(0.0)
    out = rows[(open_pnl < 0.0) & (blocked_pnl > 0.0)].copy()
    out["loss_reversal_subcase"] = np.where(
        out["best_blocked_relation"].astype(str).eq("opposite_side"),
        "final_loss_plus_opposite_side_signal",
        np.where(
            out["best_blocked_relation"].astype(str).isin(["same_contract", "same_side_different_contract"]),
            "final_loss_plus_same_side_signal",
            "final_loss_plus_unknown_relation",
        ),
    )
    out["candidate_label_limitations"] = (
        "final_open_trade_pnl_used_for_selection; live rule must use current state at blocked signal time"
    )
    return out.sort_values("best_blocked_minus_open_pnl", ascending=False, kind="stable").reset_index(drop=True)


def load_hold_state_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    keep = [col for col in HOLD_STATE_CAUSAL_COLUMNS + HOLD_STATE_DIAGNOSTIC_LABEL_COLUMNS if col in frame.columns]
    out = frame[keep].copy()
    if out.empty:
        raise ValueError(f"empty hold-state dataset: {path}")
    return out


def attach_causal_hold_state(candidates: pd.DataFrame, hold_state: pd.DataFrame) -> pd.DataFrame:
    left = candidates.copy()
    right = hold_state.copy()
    left["blocked_state_dt"] = pd.to_datetime(left["best_blocked_decision_dt"], utc=True, errors="coerce")
    right["blocked_state_dt"] = pd.to_datetime(right["state_time"], utc=True, errors="coerce")
    left["join_split"] = left["reported_split"].replace({"march_2026": "q1_2026"})
    right["join_split"] = right["reported_split"].replace({"march_2026": "q1_2026"})
    merged = left.merge(
        right,
        left_on=[
            "join_split",
            "seed",
            "session",
            "open_trade_candidate_uid",
            "open_trade_contract_id",
            "blocked_state_dt",
        ],
        right_on=["join_split", "seed", "session", "candidate_uid", "contract_id", "blocked_state_dt"],
        how="left",
        suffixes=("", "_hold_state"),
    )
    merged["causal_hold_state_join_status"] = np.where(merged["current_pnl"].notna(), "matched", "missing")
    merged["current_loss_at_blocked_signal"] = pd.to_numeric(merged["current_pnl"], errors="coerce").fillna(np.nan) < 0.0
    merged["current_positive_at_blocked_signal"] = pd.to_numeric(merged["current_pnl"], errors="coerce").fillna(np.nan) >= 0.0
    merged["exit_advantage_positive_at_blocked_signal"] = pd.to_numeric(merged.get("a_exit", 0.0), errors="coerce").fillna(0.0) > 0.0
    merged["hold_advantage_positive_at_blocked_signal"] = pd.to_numeric(merged.get("a_hold", 0.0), errors="coerce").fillna(0.0) > 0.0
    merged["one_step_hold_negative_at_blocked_signal"] = (
        pd.to_numeric(merged.get("a_hold_one_step_realized", 0.0), errors="coerce").fillna(0.0) < 0.0
    )
    merged["loss_reversal_live_test_bucket"] = merged.apply(classify_live_test_bucket, axis=1)
    return merged


def classify_live_test_bucket(row: pd.Series) -> str:
    if str(row.get("causal_hold_state_join_status")) != "matched":
        return "blocked_missing_causal_state_at_signal"
    relation = str(row.get("best_blocked_relation", ""))
    current_loss = bool(row.get("current_loss_at_blocked_signal", False))
    one_step_negative = bool(row.get("one_step_hold_negative_at_blocked_signal", False))
    exit_positive = bool(row.get("exit_advantage_positive_at_blocked_signal", False))
    if current_loss and relation == "opposite_side" and one_step_negative:
        return "priority_1_current_loss_opposite_side_one_step_negative"
    if current_loss and relation == "opposite_side":
        return "priority_2_current_loss_opposite_side"
    if current_loss and relation in {"same_contract", "same_side_different_contract"}:
        return "priority_3_current_loss_same_side"
    if exit_positive:
        return "priority_4_hindsight_exit_advantage_positive"
    return "priority_5_final_loss_but_not_current_loss"


def summarize_bucket(prefix: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    slot = pd.to_numeric(group["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0)
    matched = group["causal_hold_state_join_status"].astype(str).eq("matched")
    matched_group = group[matched]
    current_pnl = pd.to_numeric(matched_group.get("current_pnl", pd.Series(dtype=float)), errors="coerce")
    out = dict(prefix)
    out.update(
        {
            "rows": int(len(group)),
            "positive_slot_cost_total": float(slot.clip(lower=0.0).sum()),
            "median_slot_cost": float(slot.median()) if len(group) else 0.0,
            "matched_causal_state_rows": int(matched.sum()),
            "causal_state_match_rate": float(matched.mean()) if len(group) else 0.0,
            "current_loss_rows": int(matched_group["current_loss_at_blocked_signal"].sum()) if not matched_group.empty else 0,
            "current_positive_rows": int(matched_group["current_positive_at_blocked_signal"].sum()) if not matched_group.empty else 0,
            "exit_advantage_positive_rows": int(matched_group["exit_advantage_positive_at_blocked_signal"].sum())
            if not matched_group.empty
            else 0,
            "hold_advantage_positive_rows": int(matched_group["hold_advantage_positive_at_blocked_signal"].sum())
            if not matched_group.empty
            else 0,
            "one_step_hold_negative_rows": int(matched_group["one_step_hold_negative_at_blocked_signal"].sum())
            if not matched_group.empty
            else 0,
            "median_current_pnl_at_signal": float(current_pnl.median()) if current_pnl.notna().any() else 0.0,
        }
    )
    return out


def build_causal_state_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_bucket({"summary_bucket": "all_loss_reversal_candidates"}, joined)]
    for subcase, group in joined.groupby("loss_reversal_subcase", dropna=False):
        rows.append(summarize_bucket({"summary_bucket": f"subcase:{subcase}"}, group))
    for bucket, group in joined.groupby("loss_reversal_live_test_bucket", dropna=False):
        rows.append(summarize_bucket({"summary_bucket": f"live_test:{bucket}"}, group))
    for split, group in joined.groupby("reported_split", dropna=False):
        rows.append(summarize_bucket({"summary_bucket": f"split:{split}"}, group))
    return pd.DataFrame(rows)


def build_manual_review_priority_rows(joined: pd.DataFrame, limit: int = 150) -> pd.DataFrame:
    priority_order = {
        "priority_1_current_loss_opposite_side_one_step_negative": 1,
        "priority_2_current_loss_opposite_side": 2,
        "priority_3_current_loss_same_side": 3,
        "priority_4_hindsight_exit_advantage_positive": 4,
        "priority_5_final_loss_but_not_current_loss": 5,
        "blocked_missing_causal_state_at_signal": 6,
    }
    frame = joined.copy()
    frame["review_priority"] = frame["loss_reversal_live_test_bucket"].map(priority_order).fillna(99).astype(int)
    cols = [
        "review_priority",
        "loss_reversal_live_test_bucket",
        "reported_split",
        "seed",
        "session",
        "open_trade_entry_dt",
        "open_trade_exit_dt",
        "best_blocked_decision_dt",
        "open_trade_right",
        "best_blocked_right",
        "best_blocked_relation",
        "open_trade_exit_reason",
        "open_trade_pnl",
        "current_pnl",
        "mfe_to_now",
        "mae_to_now",
        "giveback_from_mfe",
        "a_hold_one_step_realized",
        "a_hold",
        "a_exit",
        "best_blocked_candidate_pnl",
        "best_blocked_minus_open_pnl",
        "best_blocked_entry_ask",
        "best_blocked_entry_spread",
        "open_duration_minutes",
        "best_blocked_minutes_after_open_entry",
        "best_blocked_minutes_before_open_exit",
        "causal_hold_state_join_status",
        "candidate_label_limitations",
    ]
    available = [col for col in cols if col in frame.columns]
    return frame.sort_values(["review_priority", "best_blocked_minus_open_pnl"], ascending=[True, False], kind="stable").head(limit)[
        available
    ]


def build_gate_checklist(summary: dict[str, Any]) -> pd.DataFrame:
    match_rate = float(summary["causal_state"]["match_rate"])
    current_loss_rows = int(summary["causal_state"]["current_loss_rows"])
    rows = [
        {
            "gate": "selected_hypothesis_loaded",
            "status": "pass",
            "requirement": "Loss-reversal candidates are present.",
            "evidence": f"{summary['counts']['loss_reversal_candidates']} candidates",
        },
        {
            "gate": "causal_state_at_blocked_signal",
            "status": "pass" if match_rate >= 0.90 else "blocked",
            "requirement": "Need causal holding state for nearly all selected rows before replay/model work.",
            "evidence": f"match_rate={match_rate:.3f}",
        },
        {
            "gate": "live_loss_condition_verified",
            "status": "pass" if current_loss_rows > 0 else "blocked",
            "requirement": "Some final-loss rows must also be losing at the later signal time.",
            "evidence": f"current_loss_rows={current_loss_rows}",
        },
        {
            "gate": "mutually_exclusive_replay",
            "status": "blocked",
            "requirement": "Need keep-hold versus exit/switch replay before labels or training.",
            "evidence": "not_built",
        },
        {
            "gate": "execution_realism",
            "status": "blocked",
            "requirement": "Need switching cost, latency, quote freshness, and fill stress.",
            "evidence": "not_calibrated",
        },
    ]
    return pd.DataFrame(rows)


def build_summary(joined: pd.DataFrame, causal_summary: pd.DataFrame, review_rows: pd.DataFrame) -> dict[str, Any]:
    total = int(len(joined))
    matched = joined["causal_hold_state_join_status"].astype(str).eq("matched")
    matched_rows = int(matched.sum())
    matched_group = joined[matched]
    current_loss_rows = int(matched_group["current_loss_at_blocked_signal"].sum()) if not matched_group.empty else 0
    current_positive_rows = int(matched_group["current_positive_at_blocked_signal"].sum()) if not matched_group.empty else 0
    opposite = joined["best_blocked_relation"].astype(str).eq("opposite_side")
    current_loss_opposite = matched & joined["current_loss_at_blocked_signal"].astype(bool) & opposite
    slot = pd.to_numeric(joined["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0)
    decision = (
        "protocol101_loss_reversal_exit_gate_diagnostic_partial_causal_state_coverage_training_blocked"
        if matched_rows < total
        else "protocol101_loss_reversal_exit_gate_diagnostic_complete_replay_still_required"
    )
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "Track A diagnostic for the selected Protocol101 loss-reversal exit-gate hypothesis",
        "decision": decision,
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "challenge_allowed": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "counts": {
            "loss_reversal_candidates": total,
            "manual_review_priority_rows": int(len(review_rows)),
        },
        "slot_cost": {
            "positive_slot_cost_total": float(slot.clip(lower=0.0).sum()),
            "median_slot_cost": float(slot.median()) if len(joined) else 0.0,
        },
        "causal_state": {
            "matched_rows": matched_rows,
            "missing_rows": int(total - matched_rows),
            "match_rate": float(matched_rows / total) if total else 0.0,
            "current_loss_rows": current_loss_rows,
            "current_positive_rows": current_positive_rows,
            "current_loss_opposite_side_rows": int(current_loss_opposite.sum()),
            "current_loss_opposite_side_slot_cost": float(slot[current_loss_opposite].clip(lower=0.0).sum()),
            "exit_advantage_positive_rows": int(matched_group["exit_advantage_positive_at_blocked_signal"].sum())
            if not matched_group.empty
            else 0,
            "hold_advantage_positive_rows": int(matched_group["hold_advantage_positive_at_blocked_signal"].sum())
            if not matched_group.empty
            else 0,
            "one_step_hold_negative_rows": int(matched_group["one_step_hold_negative_at_blocked_signal"].sum())
            if not matched_group.empty
            else 0,
        },
        "foundational_truth": (
            "The selected slot-cost hypothesis is not yet a valid ML label: final loser status is not the same as live "
            "loss/staleness at the blocked signal time."
        ),
        "blockers": [
            "full_causal_hold_state_at_blocked_signal_missing",
            "mutually_exclusive_exit_switch_replay_not_built",
            "switching_cost_fill_latency_quote_freshness_not_calibrated",
            "diagnostic_splits_are_research_exposed",
            "neural_training_forbidden_until_label_semantics_are_live_observable",
        ],
    }


def write_report(output_dir: Path, summary: dict[str, Any], causal_summary: pd.DataFrame) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: no-training diagnostic for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "The loss-reversal hypothesis is the right first question, but it is not ready for neural training. The slot-cost evidence used final open-trade PnL; a live rule needs the current holding state at the later blocked signal time.",
        "",
        f"- Loss-reversal candidates: `{summary['counts']['loss_reversal_candidates']}`",
        f"- Positive slot-cost total: `{_fmt_money(summary['slot_cost']['positive_slot_cost_total'])}`",
        f"- Causal state matched at blocked signal: `{summary['causal_state']['matched_rows']}` / `{summary['counts']['loss_reversal_candidates']}`",
        f"- Current-loss rows among matched: `{summary['causal_state']['current_loss_rows']}`",
        f"- Current-positive rows among matched: `{summary['causal_state']['current_positive_rows']}`",
        f"- Current-loss plus opposite-side rows: `{summary['causal_state']['current_loss_opposite_side_rows']}`",
        f"- One-step hold-negative rows among matched: `{summary['causal_state']['one_step_hold_negative_rows']}`",
        "",
        "This is the important truth: some final losers were not actually losing at the later signal time. A model trained on final-loss slot-cost rows would learn a hindsight label unless the row is redefined around live state.",
        "",
        "## Causal State Summary",
        "",
        "| bucket | rows | slot-cost total | matched | match rate | current loss | current positive | exit advantage + | hold advantage + | one-step hold negative | median current PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in causal_summary.iterrows():
        lines.append(
            f"| {row['summary_bucket']} | {int(row['rows'])} | {_fmt_money(row['positive_slot_cost_total'])} | "
            f"{int(row['matched_causal_state_rows'])} | {row['causal_state_match_rate']:.3f} | "
            f"{int(row['current_loss_rows'])} | {int(row['current_positive_rows'])} | "
            f"{int(row['exit_advantage_positive_rows'])} | {int(row['hold_advantage_positive_rows'])} | "
            f"{int(row['one_step_hold_negative_rows'])} | {_fmt_money(row['median_current_pnl_at_signal'])} |"
        )
    lines.extend(
        [
            "",
            "## Stopping Point",
            "",
            "Do not train a loss-reversal neural gate from the current slot-cost rows. First build the full causal state attachment for every candidate row, then run a mutually exclusive keep-hold versus exit/switch replay. Only rows where the live state supports loss/staleness should become labels.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Loss-reversal candidates: `{output_dir / 'loss_reversal_candidates.csv'}`",
            f"- Causal state join: `{output_dir / 'causal_state_join.csv'}`",
            f"- Causal state summary: `{output_dir / 'causal_state_summary.csv'}`",
            f"- Manual review priority rows: `{output_dir / 'manual_review_priority_rows.csv'}`",
            f"- Diagnostic gate checklist: `{output_dir / 'diagnostic_gate_checklist.csv'}`",
            f"- Report: `{output_dir / 'report.md'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_loss_reversal_candidates(Path(args.slot_cost_dir))
    hold_state = load_hold_state_dataset(Path(args.hold_exit_dataset))
    joined = attach_causal_hold_state(candidates, hold_state)
    causal_summary = build_causal_state_summary(joined)
    review_rows = build_manual_review_priority_rows(joined)
    summary = build_summary(joined, causal_summary, review_rows)
    gate_checklist = build_gate_checklist(summary)
    summary["inputs"] = {
        "slot_cost_dir": str(args.slot_cost_dir),
        "hold_exit_dataset": str(args.hold_exit_dataset),
    }
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "loss_reversal_candidates": str(output_dir / "loss_reversal_candidates.csv"),
        "causal_state_join": str(output_dir / "causal_state_join.csv"),
        "causal_state_summary": str(output_dir / "causal_state_summary.csv"),
        "manual_review_priority_rows": str(output_dir / "manual_review_priority_rows.csv"),
        "diagnostic_gate_checklist": str(output_dir / "diagnostic_gate_checklist.csv"),
    }
    candidates.to_csv(output_dir / "loss_reversal_candidates.csv", index=False)
    joined.to_csv(output_dir / "causal_state_join.csv", index=False)
    causal_summary.to_csv(output_dir / "causal_state_summary.csv", index=False)
    review_rows.to_csv(output_dir / "manual_review_priority_rows.csv", index=False)
    gate_checklist.to_csv(output_dir / "diagnostic_gate_checklist.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    report = write_report(output_dir, summary, causal_summary)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--slot-cost-dir", type=Path, default=DEFAULT_SLOT_COST_DIR)
    parser.add_argument("--hold-exit-dataset", type=Path, default=DEFAULT_HOLD_EXIT_DATASET)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
