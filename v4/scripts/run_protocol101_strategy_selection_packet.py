"""Protocol101 strategy-selection packet.

This Track A packet turns the slot-cost/archetype diagnostics into a single
next hypothesis recommendation. It is intentionally research-only: no model is
trained, no thresholds are tuned, and Protocol101 remains the paper default.
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


ROLE_LABEL = "AUDIT_PROTOCOL101_STRATEGY_SELECTION_PACKET_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_strategy_selection_packet_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_STRATEGY_SELECTION_PACKET_V1.md"
)
DEFAULT_SLOT_COST_DIR = Path("v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1")
DEFAULT_FOUNDATIONAL_TRUTH = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_TRACK_A_FOUNDATIONAL_TRUTH_V1.md"
)


HYPOTHESIS_REGISTRY: list[dict[str, Any]] = [
    {
        "hypothesis_id": "PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1",
        "primary_thesis": "losing_open_trade_blocks_positive_later_signal",
        "supporting_theses": "losing_open_trade_blocks_positive_later_signal",
        "mechanism": "The current Protocol101 position is failing while a later Protocol101-approved signal appears.",
        "decision_change": "Exit or refuse to keep occupying the slot when causal loss/reversal evidence appears.",
        "why_not_generic_ml": "This is not a raw loss classifier; it must distinguish avoidable stale positions from the normal cost of the playbook.",
        "first_diagnostic": "Classify top loss rows into pre-entry avoidable, path-management avoidable, execution artifact, or unavoidable.",
        "label_shape": "A_exit_or_defer = Q(exit/release slot) - Q(hold current position), charged for missed Protocol101 opportunity.",
        "causal_identifiability": 0.78,
        "specificity": 0.72,
        "low_blast_radius": 0.64,
        "notes": "Largest raw slot-cost bucket. Keep it narrow by treating opposite-side later signals as the first subcase.",
    },
    {
        "hypothesis_id": "PROTOCOL101_REGIME_FLIP_EXIT_GATE_V1",
        "primary_thesis": "opposite_side_later_signal_blocked",
        "supporting_theses": "opposite_side_later_signal_blocked",
        "mechanism": "The market may have flipped while Protocol101 still holds the original side.",
        "decision_change": "When an opposite-side Protocol101 flat-mode signal appears, decide whether to exit, switch, or keep holding.",
        "why_not_generic_ml": "The signal is a live-observable trader event: a baseline-approved opposite-side setup while one slot is occupied.",
        "first_diagnostic": "Build a mutually exclusive regime-flip replay with switching cost, no overlap, ask-entry/bid-exit, and fill stress.",
        "label_shape": "A_switch_or_exit = Q(exit/switch on opposite-side signal) - Q(continue current position), net of spread and latency.",
        "causal_identifiability": 0.90,
        "specificity": 0.88,
        "low_blast_radius": 0.72,
        "notes": "Not the largest raw bucket, but the cleanest first strategy question because the trigger is observable live.",
    },
    {
        "hypothesis_id": "PROTOCOL101_SAME_SIDE_SWITCH_OR_RUNNER_V1",
        "primary_thesis": "same_side_later_signal_blocked",
        "supporting_theses": "same_side_later_signal_blocked,high_mfe_giveback_blocks_later_signal",
        "mechanism": "Same-side signals may mean the current trade should keep running or switch to a better same-direction contract.",
        "decision_change": "Choose between hold current, exit/re-enter same side, or keep the baseline exit.",
        "why_not_generic_ml": "This is the disciplined version of 'hold longer': hold only when continuation/switching advantage is positive.",
        "first_diagnostic": "Separate same-contract continuation from same-side contract replacement and charge explicit switching cost.",
        "label_shape": "A_hold_or_switch = Q(hold/switch same-side) - Q(exit baseline), with giveback and slot-cost penalties.",
        "causal_identifiability": 0.78,
        "specificity": 0.76,
        "low_blast_radius": 0.62,
        "notes": "Important lifecycle question, but more ambiguous than regime flip because the correct action may be hold, switch, or exit.",
    },
    {
        "hypothesis_id": "PROTOCOL101_LONG_DURATION_FALLBACK_EXIT_V1",
        "primary_thesis": "long_duration_slot_cost",
        "supporting_theses": "long_duration_slot_cost,fallback_exit_slot_cost",
        "mechanism": "Long fallback/sequence holds may keep the slot occupied after the trade's continuation value has decayed.",
        "decision_change": "Exit stale long-duration holds earlier only when continuation advantage is negative after slot cost.",
        "why_not_generic_ml": "This rejects blanket max-duration rules and follows the engineer-response action-advantage formulation.",
        "first_diagnostic": "Run mutually exclusive replay for long-duration/fallback rows before training any duration gate.",
        "label_shape": "A_exit = Q(exit now at bid and free slot) - Q(hold), including future baseline opportunity value.",
        "causal_identifiability": 0.82,
        "specificity": 0.80,
        "low_blast_radius": 0.68,
        "notes": "Good first-principles lifecycle patch, but some evidence overlaps with the broader loss/regime-flip rows.",
    },
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


def attach_best_blocked_event(open_slots: pd.DataFrame, blocked_events: pd.DataFrame) -> pd.DataFrame:
    """Attach best blocked-event details to each open-trade slot row."""

    required_open = {
        "reported_split",
        "fold",
        "seed",
        "session",
        "open_trade_candidate_uid",
        "best_blocked_candidate_uid",
        "best_blocked_decision_dt",
    }
    missing_open = sorted(required_open.difference(open_slots.columns))
    if missing_open:
        raise KeyError(f"open slot rows missing columns: {missing_open}")

    required_blocked = {
        "reported_split",
        "fold",
        "seed",
        "session",
        "open_trade_candidate_uid",
        "blocked_candidate_uid",
        "blocked_decision_dt",
        "blocked_relation",
    }
    missing_blocked = sorted(required_blocked.difference(blocked_events.columns))
    if missing_blocked:
        raise KeyError(f"blocked event rows missing columns: {missing_blocked}")

    details = blocked_events[
        [
            col
            for col in [
                "reported_split",
                "fold",
                "seed",
                "session",
                "open_trade_candidate_uid",
                "blocked_candidate_uid",
                "blocked_decision_dt",
                "blocked_right",
                "blocked_contract_id",
                "blocked_relation",
                "blocked_candidate_pnl",
                "blocked_entry_ask",
                "blocked_entry_spread",
                "blocked_time_bucket",
                "minutes_after_open_entry",
                "minutes_before_open_exit",
            ]
            if col in blocked_events.columns
        ]
    ].rename(
        columns={
            "blocked_candidate_uid": "best_blocked_candidate_uid",
            "blocked_decision_dt": "best_blocked_decision_dt",
            "blocked_right": "best_blocked_right",
            "blocked_contract_id": "best_blocked_contract_id",
            "blocked_relation": "best_blocked_relation",
            "blocked_candidate_pnl": "best_blocked_event_pnl",
            "blocked_entry_ask": "best_blocked_entry_ask",
            "blocked_entry_spread": "best_blocked_entry_spread",
            "blocked_time_bucket": "best_blocked_time_bucket",
            "minutes_after_open_entry": "best_blocked_minutes_after_open_entry",
            "minutes_before_open_exit": "best_blocked_minutes_before_open_exit",
        }
    )
    return open_slots.merge(
        details,
        on=[
            "reported_split",
            "fold",
            "seed",
            "session",
            "open_trade_candidate_uid",
            "best_blocked_candidate_uid",
            "best_blocked_decision_dt",
        ],
        how="left",
    )


def hypothesis_mask(hypothesis_id: str, rows: pd.DataFrame) -> pd.Series:
    slot = pd.to_numeric(rows.get("best_blocked_minus_open_pnl", pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0.0)
    open_pnl = pd.to_numeric(rows.get("open_trade_pnl", pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0.0)
    best_pnl = pd.to_numeric(rows.get("best_blocked_candidate_pnl", pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0.0)
    relation = rows.get("best_blocked_relation", pd.Series("", index=rows.index)).astype(str)
    exit_reason = rows.get("open_trade_exit_reason", pd.Series("", index=rows.index)).astype(str)
    duration = pd.to_numeric(rows.get("open_duration_minutes", pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0.0)
    large_giveback = rows.get("open_large_uncaptured_mfe", pd.Series(False, index=rows.index)).fillna(False).astype(bool)

    if hypothesis_id == "PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1":
        return (open_pnl < 0.0) & (best_pnl > 0.0)
    if hypothesis_id == "PROTOCOL101_REGIME_FLIP_EXIT_GATE_V1":
        return relation.eq("opposite_side") & (slot > 0.0)
    if hypothesis_id == "PROTOCOL101_SAME_SIDE_SWITCH_OR_RUNNER_V1":
        return relation.isin(["same_contract", "same_side_different_contract"]) & (slot > 0.0)
    if hypothesis_id == "PROTOCOL101_LONG_DURATION_FALLBACK_EXIT_V1":
        return ((duration > 15.0) | exit_reason.eq("protocol054_fallback") | large_giveback) & (slot > 0.0)
    raise KeyError(f"unknown hypothesis id: {hypothesis_id}")


def _share(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(values.max() / total)


def _top_n_share(series: pd.Series, n: int) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).sort_values(ascending=False)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(values.head(n).sum() / total)


def _weighted_positive_rate(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    slot = pd.to_numeric(rows["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0)
    return float((slot > 0.0).mean())


def build_strategy_selection_matrix(thesis_tests: pd.DataFrame, evidence_rows: pd.DataFrame) -> pd.DataFrame:
    thesis_lookup = thesis_tests.set_index("thesis").to_dict(orient="index") if not thesis_tests.empty else {}
    rows: list[dict[str, Any]] = []
    for meta in HYPOTHESIS_REGISTRY:
        mask = hypothesis_mask(meta["hypothesis_id"], evidence_rows)
        group = evidence_rows[mask].copy()
        slot = pd.to_numeric(group.get("best_blocked_minus_open_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        positive_slot = slot.clip(lower=0.0)
        raw_thesis_rows = []
        for thesis in str(meta["supporting_theses"]).split(","):
            thesis = thesis.strip()
            if thesis in thesis_lookup:
                raw_thesis_rows.append(thesis_lookup[thesis])
        raw_thesis_total = float(sum(_finite(row.get("best_blocked_minus_open_total")) for row in raw_thesis_rows))
        raw_thesis_trades = int(sum(int(_finite(row.get("open_trades"))) for row in raw_thesis_rows))
        top_session_share = _share(positive_slot.groupby(group["session"]).sum()) if not group.empty and "session" in group else 0.0
        top_split_share = _share(positive_slot.groupby(group["reported_split"]).sum()) if not group.empty and "reported_split" in group else 0.0
        top5_share = _top_n_share(positive_slot, 5)
        concentration_health = max(0.0, 1.0 - max(top5_share / 0.70, top_session_share / 0.35, top_split_share / 0.75))
        rows.append(
            {
                **meta,
                "evidence_open_trades": int(len(group)),
                "dedup_positive_slot_cost_total": float(positive_slot.sum()),
                "dedup_net_slot_cost_total": float(slot.sum()),
                "raw_supporting_thesis_open_trades": raw_thesis_trades,
                "raw_supporting_thesis_slot_cost_total": raw_thesis_total,
                "median_slot_cost": float(slot.median()) if len(group) else 0.0,
                "positive_slot_cost_rate": _weighted_positive_rate(group),
                "sessions": int(group["session"].nunique()) if not group.empty and "session" in group else 0,
                "splits": int(group["reported_split"].nunique()) if not group.empty and "reported_split" in group else 0,
                "top_session_positive_share": top_session_share,
                "top_split_positive_share": top_split_share,
                "top5_trade_positive_share": top5_share,
                "concentration_health": concentration_health,
            }
        )
    out = pd.DataFrame(rows)
    max_cost = max(float(out["dedup_positive_slot_cost_total"].max()), 1.0)
    max_rows = max(float(out["evidence_open_trades"].max()), 1.0)
    out["normalized_evidence_cost"] = out["dedup_positive_slot_cost_total"] / max_cost
    out["normalized_evidence_rows"] = out["evidence_open_trades"] / max_rows
    out["strategy_selection_score"] = (
        0.45 * out["normalized_evidence_cost"]
        + 0.15 * out["normalized_evidence_rows"]
        + 0.15 * out["causal_identifiability"]
        + 0.10 * out["specificity"]
        + 0.10 * out["low_blast_radius"]
        + 0.05 * out["concentration_health"]
    )
    out["rank"] = out["strategy_selection_score"].rank(method="first", ascending=False).astype(int)
    cols = [
        "rank",
        "hypothesis_id",
        "strategy_selection_score",
        "primary_thesis",
        "evidence_open_trades",
        "dedup_positive_slot_cost_total",
        "raw_supporting_thesis_slot_cost_total",
        "median_slot_cost",
        "positive_slot_cost_rate",
        "sessions",
        "splits",
        "top_session_positive_share",
        "top_split_positive_share",
        "top5_trade_positive_share",
        "causal_identifiability",
        "specificity",
        "low_blast_radius",
        "mechanism",
        "decision_change",
        "first_diagnostic",
        "label_shape",
        "notes",
    ]
    return out[cols].sort_values(["rank"], kind="stable")


def build_hypothesis_overlap_matrix(evidence_rows: pd.DataFrame) -> pd.DataFrame:
    masks = {meta["hypothesis_id"]: hypothesis_mask(meta["hypothesis_id"], evidence_rows) for meta in HYPOTHESIS_REGISTRY}
    slot = pd.to_numeric(
        evidence_rows.get("best_blocked_minus_open_pnl", pd.Series(0.0, index=evidence_rows.index)),
        errors="coerce",
    ).fillna(0.0)
    rows: list[dict[str, Any]] = []
    for left_id, left_mask in masks.items():
        left_total = float(slot[left_mask].clip(lower=0.0).sum())
        for right_id, right_mask in masks.items():
            overlap = left_mask & right_mask
            overlap_total = float(slot[overlap].clip(lower=0.0).sum())
            rows.append(
                {
                    "hypothesis_id": left_id,
                    "overlap_hypothesis_id": right_id,
                    "overlap_open_trades": int(overlap.sum()),
                    "overlap_positive_slot_cost_total": overlap_total,
                    "share_of_hypothesis_positive_slot_cost": float(overlap_total / left_total) if left_total > 0.0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_hypothesis_evidence_examples(evidence_rows: pd.DataFrame, *, per_hypothesis: int = 25) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base_cols = [
        "reported_split",
        "seed",
        "session",
        "open_trade_entry_dt",
        "open_trade_exit_dt",
        "open_trade_right",
        "open_trade_contract_id",
        "open_trade_exit_reason",
        "open_time_bucket",
        "open_moneyness",
        "open_premium_bucket",
        "open_duration_minutes",
        "open_score_margin",
        "open_trade_pnl",
        "open_path_mfe",
        "open_path_mae",
        "open_giveback_from_mfe",
        "best_blocked_candidate_pnl",
        "best_blocked_minus_open_pnl",
        "best_blocked_candidate_uid",
        "best_blocked_decision_dt",
        "best_blocked_right",
        "best_blocked_contract_id",
        "best_blocked_relation",
        "best_blocked_entry_ask",
        "best_blocked_entry_spread",
        "best_blocked_time_bucket",
        "best_blocked_minutes_after_open_entry",
        "best_blocked_minutes_before_open_exit",
    ]
    for meta in HYPOTHESIS_REGISTRY:
        group = evidence_rows[hypothesis_mask(meta["hypothesis_id"], evidence_rows)].copy()
        if group.empty:
            continue
        group["hypothesis_id"] = meta["hypothesis_id"]
        group["review_reason"] = meta["decision_change"]
        cols = ["hypothesis_id", "review_reason"] + [col for col in base_cols if col in group.columns]
        frames.append(group.sort_values("best_blocked_minus_open_pnl", ascending=False, kind="stable").head(per_hypothesis)[cols])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_recommended_first_hypothesis(selection: pd.DataFrame) -> dict[str, Any]:
    if selection.empty:
        return {
            "hypothesis_id": "",
            "status": "blocked_no_strategy_selection_rows",
            "why": "No strategy-selection evidence rows were available.",
        }
    top = selection.sort_values("rank", kind="stable").iloc[0].to_dict()
    return {
        "hypothesis_id": top["hypothesis_id"],
        "status": "recommended_first_diagnostic_not_training",
        "why": (
            "Selected by weighted evidence, causal identifiability, specificity, low blast radius, "
            "and concentration health. This recommendation authorizes a preregistered diagnostic only."
        ),
        "mechanism": top["mechanism"],
        "decision_change": top["decision_change"],
        "first_diagnostic": top["first_diagnostic"],
        "label_shape": top["label_shape"],
        "score": float(top["strategy_selection_score"]),
    }


def build_selection_gate_checklist(summary: dict[str, Any]) -> pd.DataFrame:
    rec_id = summary["recommended_first_hypothesis"]["hypothesis_id"]
    rows = [
        {
            "gate": "paper_default_unchanged",
            "status": "pass",
            "requirement": "Protocol101 remains PAPER_DEFAULT_PROTOCOL101.",
            "evidence": str(summary["paper_default_baseline"]),
        },
        {
            "gate": "no_training_or_threshold_tuning",
            "status": "pass",
            "requirement": "This packet chooses a hypothesis only.",
            "evidence": f"model_training={summary['model_training']}; challenge_allowed={summary['challenge_allowed']}",
        },
        {
            "gate": "single_named_hypothesis",
            "status": "pass" if rec_id else "blocked",
            "requirement": "Exactly one hypothesis is recommended for manual review.",
            "evidence": rec_id,
        },
        {
            "gate": "manual_trade_review",
            "status": "blocked",
            "requirement": "Top rows must be reviewed as trading situations, not accepted as labels.",
            "evidence": "manual_chart_and_trade_review_required",
        },
        {
            "gate": "mutually_exclusive_replay",
            "status": "blocked",
            "requirement": "Selected hypothesis must be tested in a no-overlap serial replay.",
            "evidence": "mutually_exclusive_replay_not_built_for_selected_hypothesis",
        },
        {
            "gate": "execution_realism",
            "status": "blocked",
            "requirement": "Switching cost, latency, quote freshness, and fill evidence must be charged before promotion.",
            "evidence": "switching_cost_fill_latency_and_quote_freshness_not_calibrated",
        },
    ]
    return pd.DataFrame(rows)


def build_input_artifact_manifest(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    slot_dir = Path(args.slot_cost_dir)
    paths = {
        "slot_cost_dir": slot_dir,
        "thesis_tests": slot_dir / "slot_cost_thesis_tests.csv",
        "enriched_open_slots": slot_dir / "enriched_open_trade_slot_summary.csv",
        "enriched_blocked_events": slot_dir / "enriched_blocked_slot_events.csv",
        "foundational_truth_doc": Path(args.foundational_truth_doc),
    }
    return {
        "role_label": ROLE_LABEL,
        "artifacts": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
            for name, path in paths.items()
        },
        "outputs_root": str(output_dir),
    }


def build_summary(
    selection: pd.DataFrame,
    examples: pd.DataFrame,
    overlap: pd.DataFrame,
    foundational_truth_exists: bool,
) -> dict[str, Any]:
    recommended = build_recommended_first_hypothesis(selection)
    loss_regime_overlap = overlap[
        overlap["hypothesis_id"].eq("PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1")
        & overlap["overlap_hypothesis_id"].eq("PROTOCOL101_REGIME_FLIP_EXIT_GATE_V1")
    ]
    loss_regime = loss_regime_overlap.iloc[0].to_dict() if not loss_regime_overlap.empty else {}
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "Track A strategy-selection packet for the next Protocol101 improvement hypothesis",
        "decision": "protocol101_strategy_selection_complete_first_diagnostic_recommended_training_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "challenge_allowed": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "foundational_truth_doc_seen": bool(foundational_truth_exists),
        "recommended_first_hypothesis": recommended,
        "important_overlap": {
            "loss_reversal_and_regime_flip_open_trades": int(_finite(loss_regime.get("overlap_open_trades", 0))),
            "loss_reversal_and_regime_flip_positive_slot_cost": float(
                _finite(loss_regime.get("overlap_positive_slot_cost_total", 0.0))
            ),
            "share_of_loss_reversal_positive_slot_cost": float(
                _finite(loss_regime.get("share_of_hypothesis_positive_slot_cost", 0.0))
            ),
        },
        "counts": {
            "hypotheses_ranked": int(len(selection)),
            "evidence_examples": int(len(examples)),
        },
        "blockers": [
            "manual_chart_and_trade_review_required",
            "mutually_exclusive_replay_not_built_for_selected_hypothesis",
            "switching_cost_fill_latency_and_quote_freshness_not_calibrated",
            "counterfactual_slot_cost_evidence_is_research_exposed",
            "formal_strategy_matrix_and_untouched_holdout_required_before_challenge",
        ],
        "next_step": (
            "Build PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1 as a diagnostic packet if the human review accepts the "
            "recommended first hypothesis; treat losing-plus-opposite-side rows as the first subcase; do not train a neural model yet."
        ),
    }


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    selection: pd.DataFrame,
    examples: pd.DataFrame,
) -> str:
    rec = summary["recommended_first_hypothesis"]
    overlap = summary.get("important_overlap", {})
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: Track A strategy-selection packet",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "Track A has enough evidence to stop broad diagnostics and pick one narrow trader hypothesis for the next diagnostic. It does not have enough evidence to train a neural model.",
        "",
        f"Recommended first diagnostic: `{rec['hypothesis_id']}`.",
        "",
        "Why this one: it is the largest raw slot-cost bucket, and most of the regime-flip damage appears inside it. The first test should stay narrow: a current Protocol101 trade is losing, and a later Protocol101-approved signal appears while the slot is still occupied. The opposite-side subcase is the first manual review bucket.",
        "",
        f"Loss-reversal / regime-flip overlap: `{int(overlap.get('loss_reversal_and_regime_flip_open_trades', 0))}` rows, `{_fmt_money(overlap.get('loss_reversal_and_regime_flip_positive_slot_cost', 0.0))}` positive slot-cost total, `{float(overlap.get('share_of_loss_reversal_positive_slot_cost', 0.0)):.3f}` of loss-reversal positive slot cost.",
        "",
        "## Strategy Ranking",
        "",
        "| rank | hypothesis | score | rows | slot-cost total | top-day share | top-5-trade share | causal | specificity | first diagnostic |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in selection.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['hypothesis_id']} | {row['strategy_selection_score']:.3f} | "
            f"{int(row['evidence_open_trades'])} | {_fmt_money(row['dedup_positive_slot_cost_total'])} | "
            f"{row['top_session_positive_share']:.3f} | {row['top5_trade_positive_share']:.3f} | "
            f"{row['causal_identifiability']:.2f} | {row['specificity']:.2f} | {row['first_diagnostic']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` is first because it has the largest raw evidence and captures most of the opposite-side regime-flip damage.",
            "- `PROTOCOL101_REGIME_FLIP_EXIT_GATE_V1` remains the most important subcase because a baseline-approved opposite-side setup while holding is live-observable.",
            "- `PROTOCOL101_SAME_SIDE_SWITCH_OR_RUNNER_V1` is the disciplined version of the runner question, but it must distinguish hold-current from switch-contract economics.",
            "- `PROTOCOL101_LONG_DURATION_FALLBACK_EXIT_V1` is a useful lifecycle patch, but it overlaps with the loss and regime-flip evidence and should not be first unless manual review rejects loss-reversal.",
            "",
            "## Selected Hypothesis Contract",
            "",
            f"- Mechanism: {rec['mechanism']}",
            f"- Decision change: {rec['decision_change']}",
            f"- First diagnostic: {rec['first_diagnostic']}",
            f"- Label shape: {rec['label_shape']}",
            "",
            "The selected diagnostic should answer a trader question, not a generic ML question:",
            "",
            "> When Protocol101 is losing while holding the only slot, and another Protocol101-approved event appears, is the current trade stale enough to release the slot, or is the later signal hindsight bait/noise?",
            "",
            "## Required Next Diagnostic",
            "",
            "Build `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` as a no-training diagnostic with:",
            "",
            "- all losing-open-trade rows from the internal slot-cost counterfactual, with losing-plus-opposite-side rows reviewed first,",
            "- causal in-trade state at the blocked signal time, including current bid, PnL, MFE, MAE, giveback, duration, spread, quote age if available, and score margin,",
            "- mutually exclusive serial replay comparing keep-holding, exit-now, and exit/switch-to-later-signal where feasible,",
            "- explicit switching cost: exit spread, new entry spread, fill uncertainty, latency risk, slot opportunity cost, and model uncertainty penalty,",
            "- side/time/premium/moneyness buckets, day concentration, delay/slippage stress, and failure examples,",
            "- a hard stop that no neural model is trained until this diagnostic shows a causal, live-observable rule survives costs.",
            "",
            "## Evidence Examples",
            "",
            f"The full examples file contains `{len(examples)}` rows across hypotheses. The first rows are review targets, not training labels.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Strategy selection matrix: `{output_dir / 'strategy_selection_matrix.csv'}`",
            f"- Selected strategy: `{output_dir / 'selected_strategy.csv'}`",
            f"- Rejected alternatives: `{output_dir / 'rejected_strategy_alternatives.csv'}`",
            f"- Hypothesis evidence examples: `{output_dir / 'hypothesis_evidence_examples.csv'}`",
            f"- Hypothesis overlap matrix: `{output_dir / 'hypothesis_overlap_matrix.csv'}`",
            f"- Selection gate checklist: `{output_dir / 'selection_gate_checklist.csv'}`",
            f"- Input artifact manifest: `{output_dir / 'input_artifact_manifest.json'}`",
            f"- Report: `{output_dir / 'report.md'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slot_dir = Path(args.slot_cost_dir)
    thesis_tests = _read_required_csv(slot_dir / "slot_cost_thesis_tests.csv")
    open_slots = _read_required_csv(slot_dir / "enriched_open_trade_slot_summary.csv")
    blocked_events = _read_required_csv(slot_dir / "enriched_blocked_slot_events.csv")
    evidence_rows = attach_best_blocked_event(open_slots, blocked_events)
    selection = build_strategy_selection_matrix(thesis_tests, evidence_rows)
    examples = build_hypothesis_evidence_examples(evidence_rows)
    overlap = build_hypothesis_overlap_matrix(evidence_rows)
    summary = build_summary(selection, examples, overlap, Path(args.foundational_truth_doc).exists())
    gate_checklist = build_selection_gate_checklist(summary)
    selected = selection[selection["rank"].eq(1)].copy()
    rejected = selection[selection["rank"].ne(1)].copy()
    manifest = build_input_artifact_manifest(args, output_dir)
    summary["inputs"] = {
        "slot_cost_dir": str(slot_dir),
        "thesis_tests": str(slot_dir / "slot_cost_thesis_tests.csv"),
        "enriched_open_slots": str(slot_dir / "enriched_open_trade_slot_summary.csv"),
        "enriched_blocked_events": str(slot_dir / "enriched_blocked_slot_events.csv"),
        "foundational_truth_doc": str(args.foundational_truth_doc),
    }
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "strategy_selection_matrix": str(output_dir / "strategy_selection_matrix.csv"),
        "selected_strategy": str(output_dir / "selected_strategy.csv"),
        "rejected_strategy_alternatives": str(output_dir / "rejected_strategy_alternatives.csv"),
        "hypothesis_evidence_examples": str(output_dir / "hypothesis_evidence_examples.csv"),
        "hypothesis_overlap_matrix": str(output_dir / "hypothesis_overlap_matrix.csv"),
        "selection_gate_checklist": str(output_dir / "selection_gate_checklist.csv"),
        "input_artifact_manifest": str(output_dir / "input_artifact_manifest.json"),
    }
    selection.to_csv(output_dir / "strategy_selection_matrix.csv", index=False)
    selected.to_csv(output_dir / "selected_strategy.csv", index=False)
    rejected.to_csv(output_dir / "rejected_strategy_alternatives.csv", index=False)
    examples.to_csv(output_dir / "hypothesis_evidence_examples.csv", index=False)
    overlap.to_csv(output_dir / "hypothesis_overlap_matrix.csv", index=False)
    gate_checklist.to_csv(output_dir / "selection_gate_checklist.csv", index=False)
    (output_dir / "input_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    report = write_report(output_dir, summary, selection, examples)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--slot-cost-dir", type=Path, default=DEFAULT_SLOT_COST_DIR)
    parser.add_argument("--foundational-truth-doc", type=Path, default=DEFAULT_FOUNDATIONAL_TRUTH)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
