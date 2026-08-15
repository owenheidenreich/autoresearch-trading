"""Evaluate whether the unified conservative policy is ready for training.

This is a gate, not a training run. It reads the current foundation artifacts
and emits a single readiness packet for two stages:

1. preregistered neural training readiness;
2. stricter Protocol101 challenge readiness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.neural_training_readiness import (
    BLOCKED,
    PARTIAL,
    PASS,
    PAPER_DEFAULT_BASELINE,
    ROLE_LABEL,
    evaluate_neural_training_readiness,
    gate,
    gate_summary,
    validate_untouched_holdout_reservation,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_neural_training_readiness")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_NEURAL_TRAINING_READINESS.md")
FOUNDATION = Path("v4/audit/autoresearch/unified_conservative_offline_policy_foundation/summary.json")
TRAJECTORY = Path("v4/audit/autoresearch/unified_policy_trajectory_foundation/summary.json")
PROTOCOL276_ATTRIBUTION = Path("v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution/summary.json")
PROTOCOL272_FILL = Path("v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/summary.json")
PROTOCOL273_OVERFIT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/summary.json")
FORMAL_VALIDATION = Path("v4/audit/autoresearch/formal_validation_governance/summary.json")
PROTOCOL269_PARITY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_269_protocol265_no_order_runtime_parity/summary.json")
LIVE_PARITY_READINESS = Path("v4/audit/autoresearch/live_no_order_full_action_parity_readiness/summary.json")
HOLDOUT = Path("v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json")
HOLDOUT_AVAILABILITY = Path("v4/audit/autoresearch/untouched_holdout_availability/summary.json")
BASELINE_ATTACHMENT = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment/summary.json")
SERIAL_DP_ORACLE = Path("v4/audit/autoresearch/unified_serial_dp_oracle/summary.json")
TRAINED_POLICY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_v1/summary.json")
STRICT_REPLAY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1/summary.json")
FLAT_GATE_DIAGNOSTIC = Path("v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic/summary.json")
FLAT_CALIBRATED_POLICY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1/summary.json")
FLAT_CALIBRATED_REPLAY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1/summary.json")
FLAT_CALIBRATED_ATTRIBUTION = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_override_attribution_v1/summary.json")
Q1_Q3_UNDERPERFORMANCE = Path("v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/summary.json")
SLOT_OPPORTUNITY_OVERLAY = Path("v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation/summary.json")
SLOT_OPPORTUNITY_LABELS = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/summary.json")
SLOT_OPPORTUNITY_ESTIMATOR = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/summary.json")
SLOT_OPPORTUNITY_LEARNED_OVERLAY_REPLAY = Path("v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w025_e3/summary.json")
PREREGISTERED_LEARNED_DEFER_POLICY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/summary.json")
PREREGISTERED_LEARNED_DEFER_REPLAY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        "foundation": load_json(FOUNDATION),
        "trajectory": load_json(TRAJECTORY),
        "protocol276_attribution": load_json(PROTOCOL276_ATTRIBUTION),
        "fill": load_json(PROTOCOL272_FILL),
        "overfit": load_json(PROTOCOL273_OVERFIT),
        "formal_validation": load_json(FORMAL_VALIDATION),
        "parity": load_json(PROTOCOL269_PARITY),
        "live_parity_readiness": load_json(LIVE_PARITY_READINESS),
        "holdout": load_json(HOLDOUT),
        "holdout_availability": load_json(HOLDOUT_AVAILABILITY),
        "baseline_attachment": load_json(BASELINE_ATTACHMENT),
        "serial_dp_oracle": load_json(SERIAL_DP_ORACLE),
        "trained_policy": load_json(TRAINED_POLICY),
        "strict_replay": load_json(STRICT_REPLAY),
        "flat_gate_diagnostic": load_json(FLAT_GATE_DIAGNOSTIC),
        "flat_calibrated_policy": load_json(FLAT_CALIBRATED_POLICY),
        "flat_calibrated_replay": load_json(FLAT_CALIBRATED_REPLAY),
        "flat_calibrated_attribution": load_json(FLAT_CALIBRATED_ATTRIBUTION),
        "q1_q3_underperformance": load_json(Q1_Q3_UNDERPERFORMANCE),
        "slot_opportunity_overlay": load_json(SLOT_OPPORTUNITY_OVERLAY),
        "slot_opportunity_labels": load_json(SLOT_OPPORTUNITY_LABELS),
        "slot_opportunity_estimator": load_json(SLOT_OPPORTUNITY_ESTIMATOR),
        "slot_opportunity_learned_overlay_replay": load_json(SLOT_OPPORTUNITY_LEARNED_OVERLAY_REPLAY),
        "preregistered_learned_defer_policy": load_json(PREREGISTERED_LEARNED_DEFER_POLICY),
        "preregistered_learned_defer_replay": load_json(PREREGISTERED_LEARNED_DEFER_REPLAY),
    }
    gates = build_gates(summaries)
    evaluation = evaluate_neural_training_readiness(gates)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "foundation gate / unified conservative policy neural-training readiness",
        "changes_paper_default": False,
        "paper_default_baseline": PAPER_DEFAULT_BASELINE,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        **evaluation.to_dict(),
        "next_training_decision": next_training_decision(summaries),
        "latest_training_artifacts": latest_training_artifacts(summaries),
        "gate_summary": gate_summary(gates),
        "source_artifacts": {
            "foundation": str(FOUNDATION),
            "trajectory": str(TRAJECTORY),
            "protocol276_attribution": str(PROTOCOL276_ATTRIBUTION),
            "fill": str(PROTOCOL272_FILL),
            "overfit": str(PROTOCOL273_OVERFIT),
            "formal_validation": str(FORMAL_VALIDATION),
            "parity": str(PROTOCOL269_PARITY),
            "live_parity_readiness": str(LIVE_PARITY_READINESS),
            "holdout": str(HOLDOUT),
            "holdout_availability": str(HOLDOUT_AVAILABILITY),
            "baseline_attachment": str(BASELINE_ATTACHMENT),
            "serial_dp_oracle": str(SERIAL_DP_ORACLE),
            "trained_policy": str(TRAINED_POLICY),
            "strict_replay": str(STRICT_REPLAY),
            "flat_gate_diagnostic": str(FLAT_GATE_DIAGNOSTIC),
            "flat_calibrated_policy": str(FLAT_CALIBRATED_POLICY),
            "flat_calibrated_replay": str(FLAT_CALIBRATED_REPLAY),
            "flat_calibrated_attribution": str(FLAT_CALIBRATED_ATTRIBUTION),
            "q1_q3_underperformance": str(Q1_Q3_UNDERPERFORMANCE),
            "slot_opportunity_overlay": str(SLOT_OPPORTUNITY_OVERLAY),
            "slot_opportunity_labels": str(SLOT_OPPORTUNITY_LABELS),
            "slot_opportunity_estimator": str(SLOT_OPPORTUNITY_ESTIMATOR),
            "slot_opportunity_learned_overlay_replay": str(SLOT_OPPORTUNITY_LEARNED_OVERLAY_REPLAY),
            "preregistered_learned_defer_policy": str(PREREGISTERED_LEARNED_DEFER_POLICY),
            "preregistered_learned_defer_replay": str(PREREGISTERED_LEARNED_DEFER_REPLAY),
        },
        "next_allowed_work": next_allowed_work(evaluation.training_blockers, evaluation.challenge_blockers, summaries),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
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
    print(
        json.dumps(
            {
                "training_decision": payload["training_decision"],
                "protocol101_challenge_decision": payload["protocol101_challenge_decision"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_gates(summaries: dict[str, dict[str, Any]]) -> list[Any]:
    foundation = summaries["foundation"]
    trajectory = summaries["trajectory"]
    p276_attr = summaries["protocol276_attribution"]
    fill = summaries["fill"].get("readiness", {})
    parity = summaries["parity"]
    live_parity = summaries.get("live_parity_readiness", {})
    holdout = summaries["holdout"]
    holdout_availability = summaries.get("holdout_availability", {})
    baseline_attachment = summaries["baseline_attachment"]
    serial_dp_oracle = summaries["serial_dp_oracle"]
    holdout_validation = validate_holdout_summary(holdout)
    holdout_data_available = bool(holdout_availability.get("data_available", holdout_validation.get("data_available")))
    holdout_data_status = str(holdout_availability.get("data_status", holdout_validation.get("data_status", "missing")))

    row_counts = trajectory.get("row_counts", {})
    feature_contract = trajectory.get("feature_contract", {})
    trajectory_rows_ready = int(row_counts.get("flat_candidate_rows", 0)) > 0 and int(row_counts.get("holding_state_rows", 0)) > 0
    trajectory_features_ready = feature_contract.get("status") == "pass"
    fill_status = str(fill.get("status", "missing"))
    parity_decision = str(parity.get("decision", "missing"))
    live_parity_decision = str(live_parity.get("decision", parity_decision))
    formal_validation = summaries.get("formal_validation", {})
    formal_validation_status = str(
        formal_validation.get(
            "pbo_cscv_status",
            summaries["overfit"].get("pbo_cscv_status", summaries["overfit"].get("decision", "missing")),
        )
    )
    baseline_decision = str(baseline_attachment.get("decision", "missing"))
    oracle_decision = str(serial_dp_oracle.get("decision", "missing"))
    slot_labels = summaries.get("slot_opportunity_labels", {})
    slot_estimator = summaries.get("slot_opportunity_estimator", {})
    slot_replay = summaries.get("slot_opportunity_learned_overlay_replay", {})
    prereg_policy = summaries.get("preregistered_learned_defer_policy", {})
    prereg_replay = summaries.get("preregistered_learned_defer_replay", {})
    slot_labels_decision = str(slot_labels.get("decision", "missing"))
    slot_estimator_decision = str(slot_estimator.get("decision", "missing"))
    slot_replay_decision = str(slot_replay.get("decision", "missing"))
    prereg_policy_decision = str(prereg_policy.get("decision", "missing"))
    prereg_replay_decision = str(prereg_replay.get("decision", "missing"))

    return [
        gate(
            "Unified state/action/execution contract",
            PASS if foundation.get("decision") == "unified_conservative_offline_policy_foundation_frozen_no_model_training" else BLOCKED,
            str(foundation.get("decision", "missing")),
            "Freeze UnifiedDecisionStateV1 / ExecutionModelV1 / ActionAdvantageLabelV1 before training.",
        ),
        gate(
            "Trajectory dataset foundation",
            PASS if trajectory_rows_ready and trajectory_features_ready else BLOCKED,
            f"flat_rows={row_counts.get('flat_candidate_rows', 0)}, holding_rows={row_counts.get('holding_state_rows', 0)}, feature_status={feature_contract.get('status', 'missing')}",
            "Materialize causal flat and holding state rows with no future/path columns in model inputs.",
        ),
        gate(
            "Full serial wait/enter/hold/exit DP oracle",
            PASS if oracle_decision == "unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope" else BLOCKED,
            oracle_decision,
            "Compute wait/enter/hold/exit action values on the exact frozen simulator and same account trajectory before training.",
        ),
        gate(
            "Protocol101 baseline action attachment",
            PASS
            if baseline_decision
            in {
                "protocol101_baseline_attachment_ready_all_trajectory_splits",
                "protocol101_baseline_attachment_ready_for_baseline_aligned_training_scope",
            }
            else PARTIAL
            if baseline_decision.startswith("protocol101_baseline_attachment_partial")
            else BLOCKED,
            baseline_decision,
            "Use the baseline-aligned training scope, and add Protocol101 action/candidate/lifecycle fields for any newly admitted splits.",
        ),
        gate(
            "Protocol276 failure attribution",
            PASS if str(p276_attr.get("decision", "")).startswith("protocol276_failure_attribution_complete") else PARTIAL,
            str(p276_attr.get("decision", "missing")),
            "Keep Protocol276 abandoned as a candidate and use its attribution to shape the new labels.",
            blocks_training=False,
            blocks_protocol101_challenge=True,
        ),
        gate(
            "Causal slot opportunity-cost labels",
            PASS if slot_labels_decision == "slot_opportunity_cost_labels_ready_for_causal_estimator" else BLOCKED,
            f"{slot_labels_decision}, rows={slot_labels.get('row_counts', {}).get('candidate_rows', 0)}",
            "Materialize candidate-level blocked-Protocol101 opportunity-cost labels without future/runtime-forbidden feature leakage.",
        ),
        gate(
            "Causal slot opportunity-cost estimator",
            PASS if slot_estimator_decision == "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay" else BLOCKED,
            slot_estimator_decision,
            "Train/calibrate the slot opportunity-cost defer estimator before another neural policy run.",
        ),
        gate(
            "Learned slot opportunity-cost defer replay",
            PASS if slot_replay_decision == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training" else BLOCKED,
            slot_replay_decision,
            "Replay the learned estimator as a strict defer overlay and require Q1/Q3 nonnegative stress deltas before another neural policy run.",
        ),
        gate(
            "Preregistered learned-defer neural policy run",
            PASS if prereg_policy_decision == "conservative_neural_policy_trained_replay_and_challenge_still_blocked" else BLOCKED,
            prereg_policy_decision,
            "Run exactly one preregistered neural policy after the learned defer overlay is frozen.",
        ),
        gate(
            "Preregistered learned-defer neural replay",
            PASS if prereg_replay_decision == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training" else BLOCKED,
            prereg_replay_decision,
            "Replay the preregistered neural policy with the frozen learned defer overlay under all deterministic stress levels.",
        ),
        gate(
            "Additional neural training pause",
            BLOCKED if prereg_replay_decision == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training" else PASS,
            "single preregistered run complete" if prereg_replay_decision == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training" else "pending preregistered run",
            "Do not run additional neural experiments until fill, untouched holdout data, live parity, and formal validation gates are addressed.",
            blocks_training=True,
            blocks_protocol101_challenge=False,
        ),
        gate(
            "Deterministic execution replay",
            PASS,
            "ExecutionModelV1 uses ask-entry, bid-exit, one account, one contract, affordability, forced flat, and stress-ready slippage.",
            "Keep deterministic ask/bid plus $0.10/$0.25 per-side stress as the approved replay assumption until fills exist.",
            blocks_training=False,
            blocks_protocol101_challenge=True,
        ),
        gate(
            "Execution realism fill evidence",
            PASS if fill_status in {"ready_for_simple_empirical_fill_model", "execution_truth_packet_ready_for_conservative_fill_stress"} else BLOCKED,
            f"fill_status={fill_status}, observations={fill.get('fill_observations', 0)}/{fill.get('required_fill_observations', 30)}",
            "Collect bounded paper fill/cancel/timeout observations; keep conservative stress replay unless a larger calibrated fill model is separately validated.",
            blocks_training=False,
            blocks_protocol101_challenge=True,
        ),
        gate(
            "Untouched holdout reservation",
            PASS if holdout_validation["status"] == "pass" else BLOCKED,
            str(holdout.get("decision", "missing")),
            "Reserve a new unseen final evaluation block before model selection.",
        ),
        gate(
            "Untouched holdout data availability",
            PASS if holdout_data_available else BLOCKED,
            f"data_status={holdout_data_status}; decision={holdout_availability.get('decision', 'reservation_only')}",
            "Collect/freeze the reserved block before any final better-than-Protocol101 claim.",
            blocks_training=False,
            blocks_protocol101_challenge=True,
        ),
        gate(
            "Live no-order full-action parity",
            PASS if live_parity_decision == "live_full_action_no_order_parity_passed" else PARTIAL if parity.get("historical_replay_proxy") or live_parity else BLOCKED,
            f"{live_parity_decision}; historical_proxy={parity.get('historical_replay_proxy', False)}",
            "Upgrade historical proxy parity to live no-order parity for full candidate breadth, freshness, Greeks, masks, account state, and latency.",
            blocks_training=False,
            blocks_protocol101_challenge=True,
        ),
        gate(
            "Formal validation controls",
            PASS if formal_validation_status == "formal_validation_controls_ready" else BLOCKED,
            f"{formal_validation_status}; comparable_strategies={formal_validation.get('comparable_strategy_count', 0)}; cscv_folds={formal_validation.get('cscv_proxy', {}).get('folds', 0)}",
            "Build the strategy-matrix/PBO-CSCV or equivalent false-discovery control before promotion-grade claims.",
            blocks_training=False,
            blocks_protocol101_challenge=True,
        ),
    ]


def validate_holdout_summary(summary: dict[str, Any]) -> dict[str, Any]:
    reservation = summary.get("reservation") if isinstance(summary.get("reservation"), dict) else {}
    if not reservation:
        return {
            "status": BLOCKED,
            "errors": ["missing_holdout_reservation"],
            "warnings": [],
            "data_available": False,
            "data_status": "missing",
            "split_label": "",
            "is_exposed_diagnostic_split": False,
        }
    return validate_untouched_holdout_reservation(reservation)


def next_training_decision(summaries: dict[str, dict[str, Any]]) -> str:
    if summaries.get("preregistered_learned_defer_replay", {}).get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        return "next_training_paused_preregistered_run_complete_protocol101_challenge_blocked"
    if summaries.get("slot_opportunity_learned_overlay_replay", {}).get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        return "next_training_allowed_preregistered_unified_policy_after_learned_overlay"
    if summaries.get("slot_opportunity_estimator", {}).get("decision") == "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay":
        return "next_training_requires_learned_slot_defer_overlay_strict_replay"
    if summaries.get("slot_opportunity_labels", {}).get("decision") == "slot_opportunity_cost_labels_ready_for_causal_estimator":
        return "next_training_requires_causal_slot_opportunity_cost_estimator"
    if summaries.get("slot_opportunity_overlay", {}).get("decision") == "slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator":
        return "next_training_requires_causal_slot_opportunity_cost_labels"
    if summaries.get("q1_q3_underperformance", {}).get("decision") == "q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost":
        return "next_training_requires_slot_opportunity_cost_defer_overlay"
    if summaries.get("flat_calibrated_attribution", {}).get("decision") == "override_attribution_mixed_split_research_only":
        return "next_training_requires_split_stability_defer_repair"
    if summaries.get("flat_gate_diagnostic", {}).get("decision") == "flat_entry_gate_overconservative_zero_model_overrides":
        return "next_training_requires_preregistered_flat_entry_calibration_repair"
    if summaries.get("strict_replay", {}).get("strict_replay_run"):
        return "next_training_requires_strict_replay_attribution_before_retraining"
    return "first_preregistered_training_run_allowed"


def latest_training_artifacts(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "trained_policy_decision": summaries.get("trained_policy", {}).get("decision", "missing"),
        "strict_replay_decision": summaries.get("strict_replay", {}).get("decision", "missing"),
        "flat_gate_decision": summaries.get("flat_gate_diagnostic", {}).get("decision", "missing"),
        "flat_gate_diagnosis": summaries.get("flat_gate_diagnostic", {}).get("diagnosis", []),
        "flat_calibrated_policy_decision": summaries.get("flat_calibrated_policy", {}).get("decision", "missing"),
        "flat_calibrated_replay_decision": summaries.get("flat_calibrated_replay", {}).get("decision", "missing"),
        "flat_calibrated_attribution_decision": summaries.get("flat_calibrated_attribution", {}).get("decision", "missing"),
        "flat_calibrated_attribution_diagnosis": summaries.get("flat_calibrated_attribution", {}).get("diagnosis", []),
        "q1_q3_underperformance_decision": summaries.get("q1_q3_underperformance", {}).get("decision", "missing"),
        "q1_q3_underperformance_diagnosis": summaries.get("q1_q3_underperformance", {}).get("q1_q3_diagnosis", []),
        "slot_opportunity_overlay_decision": summaries.get("slot_opportunity_overlay", {}).get("decision", "missing"),
        "slot_opportunity_overlay_diagnosis": summaries.get("slot_opportunity_overlay", {}).get("diagnosis", []),
        "slot_opportunity_labels_decision": summaries.get("slot_opportunity_labels", {}).get("decision", "missing"),
        "slot_opportunity_label_rows": summaries.get("slot_opportunity_labels", {}).get("row_counts", {}).get("candidate_rows", 0),
        "slot_opportunity_estimator_decision": summaries.get("slot_opportunity_estimator", {}).get("decision", "missing"),
        "slot_opportunity_estimator_q1_auc": summaries.get("slot_opportunity_estimator", {}).get("metrics", {}).get("q1_2026", {}).get("positive_auc"),
        "slot_opportunity_learned_overlay_replay_decision": summaries.get("slot_opportunity_learned_overlay_replay", {}).get("decision", "missing"),
        "preregistered_learned_defer_policy_decision": summaries.get("preregistered_learned_defer_policy", {}).get("decision", "missing"),
        "preregistered_learned_defer_replay_decision": summaries.get("preregistered_learned_defer_replay", {}).get("decision", "missing"),
        "preregistered_learned_defer_replay_totals": [
            item.get("totals", {})
            for item in summaries.get("preregistered_learned_defer_replay", {}).get("stress_results", [])
        ],
    }


def next_allowed_work(
    training_blockers: tuple[str, ...],
    challenge_blockers: tuple[str, ...],
    summaries: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    work: list[str] = []
    summaries = summaries or {}
    if summaries.get("preregistered_learned_defer_replay", {}).get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        work.append("Stop neural training after the completed preregistered run; focus only on Protocol101 challenge blockers.")
    elif summaries.get("slot_opportunity_learned_overlay_replay", {}).get("decision") == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        work.append("Proceed only to a preregistered next neural policy run with the learned defer overlay fixed in advance.")
    elif summaries.get("slot_opportunity_estimator", {}).get("decision") == "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay":
        work.append("Replay the learned slot-opportunity-cost defer overlay under strict one-account serial conditions before any broader neural training.")
    elif summaries.get("slot_opportunity_labels", {}).get("decision") == "slot_opportunity_cost_labels_ready_for_causal_estimator":
        work.append("Train/calibrate the causal slot-opportunity-cost defer estimator using current-state features only.")
    elif summaries.get("slot_opportunity_overlay", {}).get("decision") == "slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator":
        work.append("Materialize causal blocked-Protocol101 opportunity-cost labels and train/calibrate the defer estimator before retraining.")
    elif summaries.get("q1_q3_underperformance", {}).get("decision") == "q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost":
        work.append("Build a slot-opportunity-cost/defer overlay before retraining; require nonnegative Q1/Q3 replay under all deterministic stress levels.")
    elif summaries.get("flat_calibrated_attribution", {}).get("decision") == "override_attribution_mixed_split_research_only":
        work.append("Explain Q1/Q3 underperformance and pre-register split-stability/defer constraints before any further training.")
    elif summaries.get("flat_gate_diagnostic", {}).get("decision") == "flat_entry_gate_overconservative_zero_model_overrides":
        work.append("Pre-register one flat-entry calibration/loss-balance repair, then rerun gate-component diagnostics before strict replay.")
    if "Full serial wait/enter/hold/exit DP oracle" in training_blockers:
        work.append("Materialize the full serial DP oracle over the frozen trajectory dataset before any neural training.")
    if "Protocol101 baseline action attachment" in training_blockers:
        work.append("Attach Protocol101 baseline/defer actions to each trajectory event for conservative improvement learning.")
    if "Execution realism fill evidence" in challenge_blockers:
        work.append("Continue collecting paper/live fill observations; keep deterministic ask/bid stress replay for offline work.")
    if "Live no-order full-action parity" in challenge_blockers:
        work.append("Build a live no-order full-action parity run before paper-default discussion.")
    if "Untouched holdout data availability" in challenge_blockers:
        work.append("Do not score final claims until the reserved unseen block is collected/frozen.")
    if "Formal validation controls" in challenge_blockers:
        work.append("Add PBO/CSCV-style false-discovery controls for the final challenge packet.")
    return work


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Training decision: `{payload['training_decision']}`",
        f"Next training decision: `{payload['next_training_decision']}`",
        f"Protocol101 challenge decision: `{payload['protocol101_challenge_decision']}`",
        "",
        "## Gate Summary",
        "",
        f"- Training blockers: `{list(payload['training_blockers'])}`",
        f"- Protocol101 challenge blockers: `{list(payload['challenge_blockers'])}`",
        "",
        "| gate | status | blocks training | blocks challenge | evidence | required action |",
        "|---|---|---|---|---|---|",
    ]
    for gate_payload in payload["gates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(gate_payload["name"]),
                    f"`{gate_payload['status']}`",
                    str(bool(gate_payload["blocks_training"])),
                    str(bool(gate_payload["blocks_protocol101_challenge"])),
                    str(gate_payload["evidence"]).replace("|", "\\|"),
                    str(gate_payload["required_action"]).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Allowed Work", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_allowed_work"], start=1))
    latest = payload["latest_training_artifacts"]
    lines.extend(
        [
            "",
            "## Latest Training Artifacts",
            "",
            f"- Trained policy decision: `{latest['trained_policy_decision']}`",
            f"- Strict replay decision: `{latest['strict_replay_decision']}`",
            f"- Flat-gate decision: `{latest['flat_gate_decision']}`",
            f"- Flat-calibrated policy decision: `{latest['flat_calibrated_policy_decision']}`",
            f"- Flat-calibrated replay decision: `{latest['flat_calibrated_replay_decision']}`",
            f"- Flat-calibrated attribution decision: `{latest['flat_calibrated_attribution_decision']}`",
            f"- Q1/Q3 underperformance decision: `{latest['q1_q3_underperformance_decision']}`",
            f"- Slot opportunity overlay decision: `{latest['slot_opportunity_overlay_decision']}`",
            f"- Slot opportunity labels decision: `{latest['slot_opportunity_labels_decision']}` over `{latest['slot_opportunity_label_rows']}` rows",
            f"- Slot opportunity estimator decision: `{latest['slot_opportunity_estimator_decision']}` (Q1 AUC `{latest['slot_opportunity_estimator_q1_auc']}`)",
            f"- Learned slot overlay replay decision: `{latest['slot_opportunity_learned_overlay_replay_decision']}`",
            f"- Preregistered learned-defer policy decision: `{latest['preregistered_learned_defer_policy_decision']}`",
            f"- Preregistered learned-defer replay decision: `{latest['preregistered_learned_defer_replay_decision']}`",
            f"- Preregistered learned-defer replay totals: `{latest['preregistered_learned_defer_replay_totals']}`",
        ]
    )
    for item in latest.get("flat_gate_diagnosis", []):
        lines.append(f"- {item}")
    for item in latest.get("flat_calibrated_attribution_diagnosis", []):
        lines.append(f"- {item}")
    for item in latest.get("q1_q3_underperformance_diagnosis", []):
        lines.append(f"- {item}")
    for item in latest.get("slot_opportunity_overlay_diagnosis", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for name, path in payload["source_artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Report: `{payload['outputs']['report']}`",
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
                    f"- Training decision: `{payload['training_decision']}`",
                    f"- Protocol101 challenge decision: `{payload['protocol101_challenge_decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                    f"- Training blockers: `{list(payload['training_blockers'])}`",
                ]
            )
            + "\n"
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
