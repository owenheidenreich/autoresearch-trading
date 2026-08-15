"""Freeze the single compact action-value fit before the gate can release it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.train_causal_day_action_value import (
    BATCH_MINUTES,
    GRADIENT_CLIP,
    INITIAL_EPOCHS,
    LEARNING_RATE,
    SEED,
    UPDATE_EPOCHS,
    WEIGHT_DECAY,
)
from v5.research.causal_day_action_advantage import HORIZON_MINUTES, LABEL_NAME
from v5.research.causal_day_action_value_attainability import (
    action_value_selector_attainability,
)
from v5.research.causal_day_action_value_targets import (
    ENTER_LOSS_WEIGHT,
    SMOOTH_L1_BETA,
    TARGET_SCALE_USD,
    WAIT_LOSS_WEIGHT,
)
from v5.research.causal_day_architectures import computed_parameter_counts
from v5.research.causal_day_compact_interaction import ARCHITECTURE_NAME
from v5.research.causal_day_declaration_reseal import (
    PINNED_RESEARCH_LAW_SHA256,
    SOURCE_EVALUATION_DECLARATION,
    SOURCE_FIT_DECLARATION,
    load_pinned_source_pair,
)
from v5.research.causal_day_policy_gate import (
    REOPENED_CORPUS,
    REOPENED_RESEARCH_LAW_SHA256,
    REQUIRED_KILL_CONDITIONS,
    fit_blockers,
    load_reopening,
)


FEATURE_CACHE_RECEIPT = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_magnitude_cache_v3/receipt.json")
TARGET_CACHE_RECEIPT = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_action_value_targets_v1/receipt.json")
ACTION_VALUE_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_action_advantage_2026_08_14_attempt002/receipt.json"
)
EXPECTED_OUTPUT = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_FIT_DECLARATION_V3.json"
)
PAIRED_EVALUATION_DECLARATION = Path(
    "v5/work/entry-exit-attribution/ACTION_VALUE_EVALUATION_DECLARATION_V3.json"
)
OUT_ROOT = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_action_value_fit_v3")
EVIDENCE_DIR = Path(
    "v4/audit/autoresearch/causal_day_action_value_fit_2026_08_14_attempt003"
)
IMPLEMENTATION = (
    Path("v5/research/causal_day_architectures.py"),
    Path("v5/research/causal_day_policy_gate.py"),
    Path("v5/research/causal_day_compact_interaction.py"),
    Path("v5/research/causal_day_action_value_attainability.py"),
    Path("v5/research/causal_day_action_value_targets.py"),
    Path("v5/research/causal_day_declaration_reseal.py"),
    Path("v5/ops/train_causal_day_action_value.py"),
    Path("v5/ops/write_causal_day_action_value_fit_declaration.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output != EXPECTED_OUTPUT:
        raise RuntimeError(f"V3 fit declaration must be written to {EXPECTED_OUTPUT}")
    if args.output.exists() or OUT_ROOT.exists() or EVIDENCE_DIR.exists():
        raise RuntimeError("refusing to overwrite declaration, fit, or evidence")
    parameter_counts = computed_parameter_counts()
    parameters = parameter_counts[ARCHITECTURE_NAME]
    attainability = action_value_selector_attainability()
    source_fit, source_evaluation = load_pinned_source_pair()
    if PINNED_RESEARCH_LAW_SHA256 != REOPENED_RESEARCH_LAW_SHA256:
        raise RuntimeError("reseal law and signed gate law differ")
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
        "schema_version": "v5.causal-day-action-value-fit-declaration.v3",
        "supersedes": str(SOURCE_FIT_DECLARATION),
        "declared_on": "2026-08-14",
        "purpose": "one decisive canonically counted WAIT-versus-ENTER fit; no neighboring retry",
        "architecture": {
            "name": ARCHITECTURE_NAME,
            "computed_parameters": parameters,
            "parameter_count_source": "computed_parameter_counts()['compact_interaction_entry']",
            "canonical_parameter_counts": parameter_counts,
            "state_contract_interactions": ["state_x_is_call", "state_x_moneyness"],
            "trained_actions": "WAIT and every eligible affordable contract",
        },
        "label": {
            "name": LABEL_NAME,
            "horizon_minutes": HORIZON_MINUTES,
            "trade_cap": 1,
            "target_scale_usd": TARGET_SCALE_USD,
            "target_transform": "q_value_usd / 1000.0",
            "target_clip_bounds": None,
            "prediction_clip_bounds": None,
            "common_scale_preserves_action_argmax": True,
        },
        "training": {
            "seed": SEED,
            "batch_minutes": BATCH_MINUTES,
            "initial_epochs": INITIAL_EPOCHS,
            "update_epochs": UPDATE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "optimizer": "AdamW",
            "loss": f"{ENTER_LOSS_WEIGHT} mean-per-minute ENTER Smooth-L1 + {WAIT_LOSS_WEIGHT} WAIT Smooth-L1",
            "smooth_l1_beta_scaled_units": SMOOTH_L1_BETA,
            "feature_scaler": "mean/std from initial 93-session prefix only, frozen forward, clipped [-10,10]",
            "folds": "existing 93-session initial prefix plus five chronological 30-session score blocks",
            "updates": "fold 1 trains on 93; each later fold updates only on the immediately prior 30 scored sessions",
        },
        "null": {
            "law": "deterministically permute all executable Q(enter) values within each training session and recompute Q(wait) from the shuffled surface",
            "preserves": ["session payoff distribution", "candidate counts", "minute clock", "one-trade action-value law"],
            "removes": "association between causal state/contract and its actual future payoff",
        },
        "inference": {
            "action": "first minute whose best current ENTER is strictly greater than max($0, predicted WAIT)",
            "external_threshold": False,
            "current_day_rank": False,
            "forced_time": False,
        },
        "selector_attainability": attainability.to_dict(),
        "mechanical_reseal": {
            "source_fit_declaration": str(SOURCE_FIT_DECLARATION),
            "source_fit_self_hash": source_fit["receipt_sha256"],
            "source_evaluation_declaration": str(SOURCE_EVALUATION_DECLARATION),
            "source_evaluation_self_hash": source_evaluation["receipt_sha256"],
            "paired_evaluation_declaration": str(PAIRED_EVALUATION_DECLARATION),
            "research_law_sha256": PINNED_RESEARCH_LAW_SHA256,
            "allowed_differences": "mechanical fields only; enforced before fit",
        },
        "evaluation": {
            "mid_to_mid_first": True,
            "bid_and_fee_only_after_positive_mid": True,
            "chronological_positive_folds_required": 4,
            "folds": 5,
            "matched_control": "session opportunity, regime, minute, side, delta, premium and trade count",
            "kill_conditions": list(REQUIRED_KILL_CONDITIONS),
            "failure_consequence": "non-positive mid gross or less than 4/5 chronology closes the long-selector branch; no retry",
        },
        "inputs": {
            "feature_cache_receipt_sha256": file_sha256(FEATURE_CACHE_RECEIPT),
            "target_cache_receipt_sha256": file_sha256(TARGET_CACHE_RECEIPT),
            "action_value_receipt_sha256": file_sha256(ACTION_VALUE_RECEIPT),
        },
        "implementation_hashes": {str(path): file_sha256(path) for path in IMPLEMENTATION},
        "current_fit_blockers": list(blockers),
        "gate_law": {
            "reopening_sha256": reopening.document_sha256,
            "corpus": REOPENED_CORPUS,
            "assert_fit_permitted_before_target_load_or_optimizer": True,
            "canonical_architecture_registered": ARCHITECTURE_NAME,
            "new_label_requires_signed_scope_widening": True,
        },
        "outputs": {"root": str(OUT_ROOT), "evidence_dir": str(EVIDENCE_DIR)},
        "forbidden": [
            "run while any fit blocker remains",
            "change architecture, target, cap, horizon, seed, loss, epochs, or null after fit",
            "threshold or operating-point search",
            "open reserved sessions",
        ],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(payload["receipt_sha256"])
    print(json.dumps({"fit_blockers": list(blockers)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
