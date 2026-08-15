"""Seal compact action-value inference and economics before a fit can run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.evaluate_causal_day_action_value import (
    BOOTSTRAP_SEED,
    DECLARED_FAMILY_SIZE,
    MAX_PREMIUM_LOSS_SHARE,
    STARTING_EQUITY_USD,
)
from v5.research.causal_day_action_advantage import HORIZON_MINUTES, LABEL_NAME
from v5.research.causal_day_action_value_attainability import (
    action_value_selector_attainability,
)
from v5.research.causal_day_architectures import computed_parameter_counts
from v5.research.causal_day_compact_interaction import ARCHITECTURE_NAME
from v5.research.causal_day_declaration_reseal import (
    PINNED_RESEARCH_LAW_SHA256,
    SOURCE_EVALUATION_DECLARATION,
    SOURCE_FIT_DECLARATION,
    assert_mechanical_reseal,
    load_declaration,
    load_pinned_source_pair,
)
from v5.research.causal_day_policy_gate import (
    REOPENED_CORPUS,
    REOPENED_RESEARCH_LAW_SHA256,
    REQUIRED_KILL_CONDITIONS,
    fit_blockers,
    load_reopening,
)


FIT_DECLARATION = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_FIT_DECLARATION_V3.json"
)
EXPECTED_OUTPUT = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_EVALUATION_DECLARATION_V3.json"
)
FEATURE_AUDIT = Path(
    "v4/audit/autoresearch/causal_day_fit_feature_audit_2026_08_14_attempt001/receipt.json"
)
ACTION_VALUE_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_action_advantage_2026_08_14_attempt002/receipt.json"
)
FIT_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_action_value_fit_2026_08_14_attempt003/receipt.json"
)
OUT_ROOT = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_action_value_economics_v3")
EVIDENCE_DIR = Path(
    "v4/audit/autoresearch/causal_day_action_value_economics_2026_08_14_attempt003"
)
IMPLEMENTATION = (
    Path("v5/research/causal_day_architectures.py"),
    Path("v5/research/causal_day_policy_gate.py"),
    Path("v5/research/causal_day_action_value_attainability.py"),
    Path("v5/research/causal_day_action_value_selection.py"),
    Path("v5/research/causal_day_declaration_reseal.py"),
    Path("v5/ops/evaluate_causal_day_action_value.py"),
    Path("v5/ops/write_causal_day_action_value_evaluation_declaration.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output != EXPECTED_OUTPUT:
        raise RuntimeError(f"V3 evaluation declaration must be written to {EXPECTED_OUTPUT}")
    if args.output.exists() or OUT_ROOT.exists() or EVIDENCE_DIR.exists():
        raise RuntimeError("refusing to overwrite evaluation declaration, output, or evidence")
    fit_declaration = load_declaration(FIT_DECLARATION)
    source_fit, source_evaluation = load_pinned_source_pair()
    if PINNED_RESEARCH_LAW_SHA256 != REOPENED_RESEARCH_LAW_SHA256:
        raise RuntimeError("reseal law and signed gate law differ")
    action_receipt = json.loads(ACTION_VALUE_RECEIPT.read_text())
    candidate = action_receipt["artifacts"]["candidate_action_values"]
    parameter_counts = computed_parameter_counts()
    parameters = parameter_counts[ARCHITECTURE_NAME]
    attainability = action_value_selector_attainability()
    reopening = load_reopening()
    blockers = fit_blockers(
        ARCHITECTURE_NAME,
        sessions=243,
        trainable_parameters=parameters,
        reopening=reopening,
        label=LABEL_NAME,
        horizon=HORIZON_MINUTES,
        corpus=REOPENED_CORPUS,
        declared_kill_conditions=REQUIRED_KILL_CONDITIONS,
    )
    if blockers:
        raise RuntimeError("signed action-value scope is not fit-permitted: " + "; ".join(blockers))
    payload = {
        "schema_version": "v5.causal-day-action-value-evaluation-declaration.v3",
        "supersedes": str(SOURCE_EVALUATION_DECLARATION),
        "declared_on": "2026-08-14",
        "purpose": "complete outcome-firewalled evaluation of the single compact WAIT-versus-ENTER fit",
        "fit_declaration": {
            "path": str(FIT_DECLARATION),
            "sha256": file_sha256(FIT_DECLARATION),
            "self_hash": fit_declaration["receipt_sha256"],
        },
        "future_fit_receipt": str(FIT_RECEIPT),
        "architecture": {
            "name": ARCHITECTURE_NAME,
            "computed_parameters": parameters,
            "parameter_count_source": "computed_parameter_counts()['compact_interaction_entry']",
            "canonical_parameter_counts": parameter_counts,
            "label": LABEL_NAME,
            "horizon_minutes": HORIZON_MINUTES,
        },
        "selector_attainability": attainability.to_dict(),
        "mechanical_reseal": {
            "source_fit_declaration": str(SOURCE_FIT_DECLARATION),
            "source_fit_self_hash": source_fit["receipt_sha256"],
            "source_evaluation_declaration": str(SOURCE_EVALUATION_DECLARATION),
            "source_evaluation_self_hash": source_evaluation["receipt_sha256"],
            "research_law_sha256": PINNED_RESEARCH_LAW_SHA256,
            "allowed_differences": "mechanical fields only; enforced before fit/evaluation",
        },
        "inference": {
            "trade_cap": 1,
            "law": "walk each score session from 09:35 through 15:00; enter the first minute where the largest current predicted Q(enter) strictly exceeds max(0, predicted Q(wait)); otherwise abstain",
            "no_trade_floor_usd": 0.0,
            "no_trade_floor_reason": "doing nothing is always available for zero dollars; regression error cannot make abstention economically negative",
            "contract_tie_break": "lexicographically first contract_id at the same maximum current prediction",
            "external_threshold": False,
            "current_day_rank": False,
            "forced_time": False,
            "later_same_day_prediction_used": False,
        },
        "outcome_firewall": {
            "primary_first": "mean gross mid-to-mid P&L per all 150 scored sessions, with abstentions zero",
            "stop_if": "no selected trade or primary mean <= 0",
            "only_after_primary_positive": [
                "ask-entry/bid-exit net outcomes including measured fee",
                "outcome-blind composition-matched control",
                "identically fitted shuffled-label policy",
            ],
            "operating_point_search": False,
        },
        "matched_control": {
            "law": "one alternate per real selected session; exact session, opening regime and side; exclude the selected action; minimize minute distance, then absolute-delta distance, then log-premium distance, then contract_id",
            "same_minute_preferred": True,
            "fallback": "nearest eligible minute in the same session, opening regime and side",
            "payoff_read_during_match": False,
            "trade_count_preserved": True,
            "report_maximum_minute_distance": True,
        },
        "shuffled_control": {
            "law": fit_declaration["null"]["law"],
            "training_and_inference_identical_to_real": True,
            "comparison_unit": "paired all-session vectors; abstentions are zero",
        },
        "inference_standard": {
            "declared_family_size": DECLARED_FAMILY_SIZE,
            "family_reason": "retain Job 39's 648-member family and add exactly one compact successor",
            "one_sided_alpha": 0.05,
            "moving_block_sessions": 5,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "positive_chronological_folds_required": 4,
            "required_positive_vectors": [
                "real executable net",
                "real minus matched executable net",
                "real minus shuffled executable net",
            ],
            "corrected_lower_bound_above_zero_required_for_all_vectors": True,
        },
        "risk": {
            "starting_equity_usd": STARTING_EQUITY_USD,
            "maximum_loss_share": MAX_PREMIUM_LOSS_SHARE,
            "maximum_loss_usd": STARTING_EQUITY_USD * MAX_PREMIUM_LOSS_SHARE,
            "requirements": [
                "largest selected ask ticket plus fee <= $500",
                "worst realised selected trade >= -$500",
            ],
            "reason": "the fixed 120-minute policy has no proven early stop, so the signed 5% daily breaker must bind to premium-at-risk rather than an assumed exit",
        },
        "kill_conditions": list(REQUIRED_KILL_CONDITIONS),
        "additional_completion_requirements": [
            "executable net corrected lower bound above zero",
            "risk compatible with the $10,000 account",
        ],
        "failure_consequence": "any primary, executable, control, chronology, confidence or risk failure closes this long-selector branch; no target, horizon, seed, architecture or operating-point retry",
        "feature_audit": {"path": str(FEATURE_AUDIT), "sha256": file_sha256(FEATURE_AUDIT)},
        "action_values": {
            "receipt_path": str(ACTION_VALUE_RECEIPT),
            "receipt_sha256": file_sha256(ACTION_VALUE_RECEIPT),
            "candidate_path": candidate["path"],
            "candidate_sha256": candidate["sha256"],
            "candidate_rows": candidate["rows"],
        },
        "implementation_hashes": {str(path): file_sha256(path) for path in IMPLEMENTATION},
        "current_fit_blockers": list(blockers),
        "gate_law": {
            "corpus": REOPENED_CORPUS,
            "reopening_sha256": reopening.document_sha256,
            "fit_and_evaluation_both_call_gate": True,
            "canonical_architecture_registered": ARCHITECTURE_NAME,
            "new_label_requires_signed_scope_widening": True,
        },
        "outputs": {"root": str(OUT_ROOT), "evidence_dir": str(EVIDENCE_DIR)},
        "forbidden": [
            "run while any fit blocker remains",
            "alter the structural zero-dollar WAIT floor after fit",
            "select a scored-session threshold or top-N rank",
            "read executable economics if midpoint primary is non-positive",
            "retry a neighboring policy after failure",
            "open reserved sessions, promote, paper trade, or submit an order",
        ],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    reseal_hash = assert_mechanical_reseal(
        source_fit,
        source_evaluation,
        fit_declaration,
        payload,
    )
    if reseal_hash != PINNED_RESEARCH_LAW_SHA256:
        raise RuntimeError("candidate declarations do not reproduce the signed research law")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(payload["receipt_sha256"])
    print(json.dumps({"fit_blockers": list(blockers)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
