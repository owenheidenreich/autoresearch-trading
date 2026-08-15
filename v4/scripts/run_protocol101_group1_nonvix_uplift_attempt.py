"""Run Protocol101 Group 1 non-VIX index/context uplift attempt001.

This is an offline-only preregistered uplift harness for the certified v2
masked contract. It consumes the governed 5-fold expanding-window scaffold,
keeps the required masked model-facing transform, appends only parity-certified
stable index/context features, and evaluates the G1-G9 gates.

It does not contact brokers, submit paper orders, download paid data, change
promotion/defaults, edit runtime flags, touch launchd, or use real-money paths.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

from v4.model.protocol101_governed_loader import (
    file_sha256,
    load_governed_loader_artifacts,
    resolve_governed_expanding_fold_paths,
    stable_hash,
)
from v4.model.supervised_pilot import (
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    PilotConfig,
    jitter_decision_features,
    load_decisions,
    metrics_for_trades,
    simulate_baseline,
    simulate_model_policy,
    summarize_random_baseline,
    top_prediction,
    transform_decision_features,
)
from v4.scripts import run_protocol101_fair_contract_training_runner as runner
from v4.scripts import run_protocol101_stage1_bounded_hgb_search as stage1


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_live_v2_group1_nonvix_uplift_attempt001"
DEFAULT_ATTEMPT002_OUT_DIR = DEFAULT_OUT_DIR
DEFAULT_DESIGN = BASE_AUDIT / "protocol101_live_v2_microstructure_masked_15mo_training_design" / "summary.json"
DEFAULT_ACCEPTANCE_REGISTRY = (
    BASE_AUDIT / "protocol101_live_v2_microstructure_masked_15mo_training_scope_acceptance" / "summary.json"
)
DEFAULT_ERA_MANIFEST = BASE_AUDIT / "protocol101_session_era_manifest" / "summary.json"
DEFAULT_ROLE_POLICY = (
    BASE_AUDIT / "protocol101_live_v2_microstructure_masked_training_role_policy" / "summary.json"
)
DEFAULT_PROTECTED_HOLDOUT = BASE_AUDIT / "protocol101_protected_holdout" / "summary.json"
DEFAULT_STATIC_POLICY_SUMMARY = (
    BASE_AUDIT / "protocol101_live_v2_static_ladder_boundary_stable_policy_audit" / "summary.json"
)
DEFAULT_STATIC_POLICY_REPORT = (
    BASE_AUDIT / "protocol101_live_v2_static_ladder_boundary_stable_policy_audit" / "report.md"
)
DEFAULT_GROUP1_NONVIX_PARITY_SUMMARY = (
    BASE_AUDIT / "protocol101_live_v2_group1_index_context_parity_resolution" / "summary.json"
)
DEFAULT_GROUP1_NONVIX_PARITY_REPORT = (
    BASE_AUDIT / "protocol101_live_v2_group1_index_context_parity_resolution" / "report.md"
)
DEFAULT_GROUP1_NONVIX_PARITY_RESULT = (
    BASE_AUDIT / "protocol101_live_v2_group1_index_context_parity_resolution" / "parity_result.json"
)
DEFAULT_GROUP1_NONVIX_FEATURE_COVERAGE = (
    BASE_AUDIT / "protocol101_live_v2_group1_index_context_parity_resolution" / "feature_coverage.csv"
)
DEFAULT_GROUP1_NONVIX_DEFINITION = (
    BASE_AUDIT / "protocol101_live_v2_feature_recovery_group1_stable_index_context_plan" / "feature_group_definition.json"
)
DEFAULT_STAGE1_PRIMARY = BASE_AUDIT / "protocol101_live_v2_microstructure_masked_stage1_attempt001" / "primary_evaluation.json"
DEFAULT_STAGE1_CONSERVATIVE = (
    BASE_AUDIT / "protocol101_live_v2_microstructure_masked_stage1_attempt001" / "conservative_evaluation.json"
)
DEFAULT_GATES_DOC = Path(
    "v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md"
)
DEFAULT_FEATURE_RECOVERY_PLAN = Path(
    "v4/docs/protocol101/synchronization/history/PROTOCOL101_FAIR_CONTRACT_TRAINING_AND_FEATURE_RECOVERY_PLAN_2026_07_08.md"
)
DEFAULT_NORMALIZED_DIR = BASE_AUDIT / "protocol101_live_v2_microstructure_masked_15mo_build" / "normalized"

CONTRACT = "protocol101-live-v2-microstructure-masked"
TRANSFORM = FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE
ATTEMPT_ID = "protocol101_live_v2_group1_nonvix_uplift_attempt001"
ATTEMPT002_ID = ATTEMPT_ID
POLICY_ID = "boundary_stable_tradability_plus_static_ladder_for_non_quote_features_v1"
MODEL_FAMILY = "sklearn_hist_gradient_boosting"
SELECTION_SEEDS = [42, 43, 44]
CONFIRMATION_SEED = 1042
ROUND_TRIP_FEE = 3.0
FEE_SENSITIVITIES = [2.0, 5.0]
ADVERSE_STRESS_PER_TRADE = 20.0
STARTING_CASH = 10_000.0
GROUP1_NONVIX_FEATURE_NAMES = [
    "spx_vwap_gap_points",
    "spx_vwap_gap_bps",
    "spx_vwap_gap_over_session_range",
    "session_range_bps",
    "momentum_5m_bps",
    "momentum_15m_bps",
    "momentum_5m_over_session_range",
    "momentum_15m_over_session_range",
    "omar_clipped_neg3_pos3",
    "vwap_side_alignment_flag",
    "omar_side_alignment_flag",
    "momentum15_side_alignment_flag",
]
EXCLUDED_VIX_FEATURES = [
    "vix_change_5m",
    "vix_change_15m",
    "vix_change_5m_bps",
    "vix_change_15m_bps",
]
RECORDER_PARITY_SESSIONS = {"2026-06-30", "2026-07-01", "2026-07-02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--attempt-id", default=ATTEMPT_ID)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--acceptance-registry", type=Path, default=DEFAULT_ACCEPTANCE_REGISTRY)
    parser.add_argument("--era-manifest", type=Path, default=DEFAULT_ERA_MANIFEST)
    parser.add_argument("--role-policy", type=Path, default=DEFAULT_ROLE_POLICY)
    parser.add_argument("--protected-holdout", type=Path, default=DEFAULT_PROTECTED_HOLDOUT)
    parser.add_argument("--static-policy-summary", type=Path, default=DEFAULT_STATIC_POLICY_SUMMARY)
    parser.add_argument("--static-policy-report", type=Path, default=DEFAULT_STATIC_POLICY_REPORT)
    parser.add_argument("--group1-parity-summary", type=Path, default=DEFAULT_GROUP1_NONVIX_PARITY_SUMMARY)
    parser.add_argument("--group1-parity-report", type=Path, default=DEFAULT_GROUP1_NONVIX_PARITY_REPORT)
    parser.add_argument("--group1-parity-result", type=Path, default=DEFAULT_GROUP1_NONVIX_PARITY_RESULT)
    parser.add_argument("--group1-feature-coverage", type=Path, default=DEFAULT_GROUP1_NONVIX_FEATURE_COVERAGE)
    parser.add_argument("--group1-definition", type=Path, default=DEFAULT_GROUP1_NONVIX_DEFINITION)
    parser.add_argument("--stage1-primary", type=Path, default=DEFAULT_STAGE1_PRIMARY)
    parser.add_argument("--stage1-conservative", type=Path, default=DEFAULT_STAGE1_CONSERVATIVE)
    parser.add_argument("--gates-doc", type=Path, default=DEFAULT_GATES_DOC)
    parser.add_argument("--feature-recovery-plan", type=Path, default=DEFAULT_FEATURE_RECOVERY_PLAN)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--max-train-examples", type=int, default=350_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-learning-curve", action="store_true")
    parser.add_argument(
        "--instrumented",
        action="store_true",
        help="Persist per-fold checkpoints, prediction/trade/path artifacts, progress, and reproducibility outputs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 1-fold/1-policy/1-seed smoke persistence check and write smoke/runtime artifacts.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Diagnostic limiter for smoke/tests. 0 means all governed folds.",
    )
    parser.add_argument(
        "--policy-indexes",
        default="",
        help="Comma-separated policy indexes for diagnostic smoke/tests. Empty means all 7 policies.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated selection seeds for diagnostic smoke/tests. Empty means the preregistered seeds.",
    )
    parser.add_argument(
        "--batches",
        default="",
        help="Comma-separated batches for diagnostic smoke/tests. Empty preserves preregistered primary-then-conservative routing.",
    )
    parser.add_argument("--top-n-predictions", type=int, default=5)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n")


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def fold_hash(design: dict[str, Any]) -> str:
    split = design.get("split_policy") or {}
    return stable_hash(
        {
            "required_expanding_window_cv": split.get("required_expanding_window_cv"),
            "fold_count": split.get("fold_count"),
            "expanding_window_embargo_sessions": split.get("expanding_window_embargo_sessions"),
            "folds": split.get("folds") or [],
        }
    )


def feature_set_payload() -> dict[str, Any]:
    return {
        "base_transform": TRANSFORM,
        "feature_addback_group": "group1_stable_index_context_non_vix_subset",
        "static_ladder_policy_id": POLICY_ID,
        "group1_nonvix_feature_names": GROUP1_NONVIX_FEATURE_NAMES,
        "excluded_vix_features": EXCLUDED_VIX_FEATURES,
        "feature_expression_version": "stable_index_context_refinements_v1_non_vix_subset",
        "deterministic_guards": {
            "minimum_denominator_abs": 1.0,
            "session_range_floor_points": 1.0,
            "omar_clip_min": -3.0,
            "omar_clip_max": 3.0,
            "missing_or_nonfinite_value": "zero_without_backfill",
        },
        "forbidden_alpha_sources": [
            "VIX-change features",
            "Group 2 geometry/moneyness features",
            "Greeks/IV",
            "raw bid/ask/mid/spread/size quote microstructure",
            "liquidity/spread alpha",
            "volume/open-interest alpha",
        ],
    }


def feature_set_hash() -> str:
    return stable_hash(feature_set_payload())


def policy_hash(args: argparse.Namespace) -> str:
    return stable_hash(
        {
            "policy_summary": artifact(args.static_policy_summary),
            "group1_nonvix_parity_summary": artifact(args.group1_parity_summary),
            "group1_nonvix_parity_report": artifact(args.group1_parity_report),
            "group1_nonvix_parity_result": artifact(args.group1_parity_result),
            "group1_nonvix_feature_coverage": artifact(args.group1_feature_coverage),
            "policy_id": POLICY_ID,
        }
    )


def group1_parity_artifact_hash(args: argparse.Namespace) -> str:
    return stable_hash(
        {
            "summary": artifact(args.group1_parity_summary),
            "report": artifact(args.group1_parity_report),
            "parity_result": artifact(args.group1_parity_result),
            "feature_coverage": artifact(args.group1_feature_coverage),
        }
    )


def null_canary_refs() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for policy_index in range(7):
        path = BASE_AUDIT / f"protocol101_live_v2_microstructure_masked_null_canary_15mo_cv_policy{policy_index}" / "summary.json"
        refs.append(
            {
                "policy_index": policy_index,
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else "",
            }
        )
    return refs


def load_manifest(design: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str((design.get("allowed_data") or {}).get("canonical_manifest") or ""))
    return load_json(manifest_path)


def resolve_fold_paths(args: argparse.Namespace) -> tuple[dict[str, dict[str, list[Path]]], dict[str, Any], list[str]]:
    design = load_json(args.design)
    manifest = load_manifest(design)
    artifacts = load_governed_loader_artifacts(
        acceptance_registry_path=args.acceptance_registry,
        era_manifest_path=args.era_manifest,
        role_policy_path=args.role_policy,
        protected_holdout_path=args.protected_holdout,
    )
    fold_paths, blockers, governance = resolve_governed_expanding_fold_paths(
        design=design,
        manifest=manifest,
        artifacts=artifacts,
    )
    return fold_paths, governance, blockers


def split_sessions_from_fold_paths(fold_paths: dict[str, dict[str, list[Path]]]) -> dict[str, dict[str, list[str]]]:
    return {
        fold_id: {
            split: [path.name.removesuffix(".pkl") for path in paths]
            for split, paths in sorted(splits.items())
        }
        for fold_id, splits in sorted(fold_paths.items())
    }


def input_readiness(args: argparse.Namespace, fold_paths: dict[str, dict[str, list[Path]]], governance_blockers: list[str]) -> dict[str, Any]:
    design = load_json(args.design)
    acceptance = load_json(args.acceptance_registry)
    static_policy = load_json(args.static_policy_summary)
    parity_summary = load_json(args.group1_parity_summary)
    parity_result = load_json(args.group1_parity_result)
    fold_sessions = split_sessions_from_fold_paths(fold_paths)
    used_sessions = {
        session
        for folds in fold_sessions.values()
        for sessions in folds.values()
        for session in sessions
    }
    blockers = list(governance_blockers)
    if design.get("selected_feature_contract") != CONTRACT:
        blockers.append("unexpected_contract")
    allowed = design.get("allowed_data") or {}
    if allowed.get("require_manifest_loading") is not True:
        blockers.append("manifest_loading_not_required")
    if allowed.get("glob_loading_allowed") is not False:
        blockers.append("glob_loading_allowed")
    split = design.get("split_policy") or {}
    if split.get("required_expanding_window_cv") is not True:
        blockers.append("expanding_window_cv_not_required")
    if len(split.get("folds") or []) != 5:
        blockers.append(f"unexpected_fold_count:{len(split.get('folds') or [])}")
    if acceptance.get("status") != "pass":
        blockers.append(f"training_scope_acceptance_not_pass:{acceptance.get('status')}")
    if int(acceptance.get("fail_count") or 0) != 0:
        blockers.append("training_scope_has_fail_sessions")
    if int(acceptance.get("report_only_count") or 0) != 0:
        blockers.append("training_scope_has_report_only_sessions")
    if static_policy.get("status") != "complete":
        blockers.append("static_ladder_policy_not_complete")
    if parity_summary.get("status") != "complete":
        blockers.append(f"group1_parity_resolution_not_complete:{parity_summary.get('status')}")
    if parity_summary.get("contract") != CONTRACT or parity_result.get("contract") != CONTRACT:
        blockers.append("group1_parity_resolution_wrong_contract")
    if parity_summary.get("required_transform") != TRANSFORM or parity_result.get("required_transform") != TRANSFORM:
        blockers.append("group1_parity_resolution_wrong_transform")
    if parity_summary.get("subset_eligibility") is not True or parity_result.get("subset_eligibility") is not True:
        blockers.append("group1_nonvix_subset_not_parity_eligible")
    if parity_summary.get("full_group_eligibility") is not False or parity_result.get("full_group_eligibility") is not False:
        blockers.append("group1_full_group_unexpectedly_eligible")
    parity_features = list(parity_summary.get("eligible_subset_features") or [])
    if parity_features != GROUP1_NONVIX_FEATURE_NAMES:
        blockers.append("group1_nonvix_feature_list_mismatch")
    if sorted(str(item) for item in parity_summary.get("excluded_features") or []) != sorted(EXCLUDED_VIX_FEATURES):
        blockers.append("group1_excluded_vix_feature_list_mismatch")
    coverage = pd.read_csv(args.group1_feature_coverage)
    coverage_by_feature = {str(row["feature"]): str(row["after_status"]) for _, row in coverage.iterrows()}
    bad_coverage = [
        feature
        for feature in GROUP1_NONVIX_FEATURE_NAMES
        if coverage_by_feature.get(feature) != "parity_eligible_in_subset"
    ]
    bad_excluded = [
        feature
        for feature in EXCLUDED_VIX_FEATURES
        if coverage_by_feature.get(feature) != "excluded_pending_new_evidence"
    ]
    if bad_coverage:
        blockers.append(f"group1_nonvix_feature_coverage_not_eligible:{bad_coverage}")
    if bad_excluded:
        blockers.append(f"group1_vix_feature_coverage_not_excluded:{bad_excluded}")
    if used_sessions & RECORDER_PARITY_SESSIONS:
        blockers.append(f"recorder_parity_sessions_used:{sorted(used_sessions & RECORDER_PARITY_SESSIONS)}")
    protected_sessions = set(load_json(args.protected_holdout).get("sessions") or [])
    protected_conflicts = sorted(used_sessions & protected_sessions)
    if protected_conflicts:
        blockers.append(f"protected_holdout_sessions_used:{protected_conflicts}")
    if len(fold_paths) != 5:
        blockers.append(f"governed_fold_path_count_not_5:{len(fold_paths)}")
    for fold_id, splits in fold_paths.items():
        if not splits.get("train") or not splits.get("validation"):
            blockers.append(f"empty_governed_fold_split:{fold_id}")
    return {
        "schema_version": "Protocol101Group1NonVixUpliftInputReadinessV1",
        "status": "pass" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "training_scope_registry": {
            "path": str(args.acceptance_registry),
            "registry_hash": acceptance.get("registry_hash"),
            "session_count": acceptance.get("session_count"),
            "pass_count": acceptance.get("pass_count"),
            "fail_count": acceptance.get("fail_count"),
            "report_only_count": acceptance.get("report_only_count"),
        },
        "fold_count": len(fold_paths),
        "fold_sessions": fold_sessions,
        "group1_nonvix_parity": {
            "summary_path": str(args.group1_parity_summary),
            "report_path": str(args.group1_parity_report),
            "parity_result_path": str(args.group1_parity_result),
            "feature_coverage_path": str(args.group1_feature_coverage),
            "subset_eligibility": parity_summary.get("subset_eligibility"),
            "full_group_eligibility": parity_summary.get("full_group_eligibility"),
            "eligible_subset_features": parity_features,
            "excluded_features": parity_summary.get("excluded_features") or [],
        },
        "excluded_session_checks": {
            "recorder_parity_sessions_in_used_scope": sorted(used_sessions & RECORDER_PARITY_SESSIONS),
            "protected_holdout_sessions_in_used_scope": protected_conflicts,
        },
    }


def preregistration_payload(args: argparse.Namespace, input_gate: dict[str, Any]) -> dict[str, Any]:
    design = load_json(args.design)
    acceptance = load_json(args.acceptance_registry)
    return {
        "schema_version": "Protocol101Group1NonVixUpliftPreregistrationV1",
        "status": "preregistered",
        "attempt_id": args.attempt_id,
        "created_at_utc": now_utc(),
        "results_inspected_before_preregistration": False,
        "scope": "offline_preregistered_group1_nonvix_uplift_only",
        "contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "corpus": {
            "training_scope_registry_hash": str(acceptance.get("registry_hash") or ""),
            "session_count": int(acceptance.get("session_count") or 0),
            "pass_count": int(acceptance.get("pass_count") or 0),
            "fail_count": int(acceptance.get("fail_count") or 0),
            "report_only_count": int(acceptance.get("report_only_count") or 0),
            "included_first_session": (design.get("allowed_data") or {}).get("included_first_session"),
            "included_last_session": (design.get("allowed_data") or {}).get("included_last_session"),
            "canonical_manifest": (design.get("allowed_data") or {}).get("canonical_manifest"),
        },
        "artifacts": {
            "design": artifact(args.design),
            "acceptance_registry": artifact(args.acceptance_registry),
            "static_ladder_policy_summary": artifact(args.static_policy_summary),
            "static_ladder_policy_report": artifact(args.static_policy_report),
            "group1_nonvix_parity_summary": artifact(args.group1_parity_summary),
            "group1_nonvix_parity_report": artifact(args.group1_parity_report),
            "group1_nonvix_parity_result": artifact(args.group1_parity_result),
            "group1_nonvix_feature_coverage": artifact(args.group1_feature_coverage),
            "group1_definition": artifact(args.group1_definition),
            "stage1_masked_primary_baseline": artifact(args.stage1_primary),
            "stage1_masked_conservative_baseline": artifact(args.stage1_conservative),
            "gates_doc": artifact(args.gates_doc),
            "feature_recovery_plan": artifact(args.feature_recovery_plan),
        },
        "hashes": {
            "corpus_hash": str(acceptance.get("registry_hash") or ""),
            "split_hash": fold_hash(design),
            "feature_set_hash": feature_set_hash(),
            "static_ladder_policy_hash": policy_hash(args),
            "group1_nonvix_parity_artifact_hash": group1_parity_artifact_hash(args),
        },
        "feature_set": feature_set_payload(),
        "model": {
            "family": MODEL_FAMILY,
            "target": "return_on_premium_regression",
            "target_clip": 5.0,
            "threshold_rule": "max_validation_stressed_pnl",
            "threshold_selection_split": "train_tail20_calibration_only",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "max_train_examples": int(args.max_train_examples),
            "selection_mode": "top_score",
            "seeds": SELECTION_SEEDS,
            "confirmation_seed": CONFIRMATION_SEED,
        },
        "policies": {
            str(policy): {"policy_name": name, "cooldown_minutes": cooldown}
            for policy, (name, cooldown) in sorted(runner.POLICY_META.items())
        },
        "fees_and_stress": {
            "round_trip_fee_dollars": ROUND_TRIP_FEE,
            "fee_sensitivity_round_trips": FEE_SENSITIVITIES,
            "adverse_bid_ask_stress_per_trade_dollars": ADVERSE_STRESS_PER_TRADE,
        },
        "excluded_features": {
            "vix_change_features": EXCLUDED_VIX_FEATURES,
            "group2_geometry_features_added": False,
            "greeks_iv_added": False,
            "quote_liquidity_spread_alpha_added": False,
            "raw_microstructure_alpha_added": False,
            "volume_oi_alpha_added": False,
        },
        "gates": {
            "G1_profitability": "fee_adjusted_net_pnl_gt_0_on_at_least_4_of_5_folds_and_pooled_gt_0",
            "G2_beats_no_skill": "pooled_top_selection_pnl_z_score_gte_3_vs_matched_random_null",
            "G3_beats_heuristic": "pooled_fee_adjusted_pnl_gt_fixed_heuristic_baseline_same_folds",
            "G4_drawdown": "max_strict_serial_drawdown_lte_25pct_peak_equity_each_fold",
            "G5_seed_robustness": "three_seeds_worst_seed_satisfies_G1_and_G2_z_gte_2",
            "G6_era_guard": "no_era_with_systematically_negative_test_folds",
            "G7_frequency": "0.3_to_6.0_trades_per_day_average_per_fold",
            "G8_calibration": "pooled_ece_lte_0.10_on_payoff_score_confidence",
            "G9_confirmation": "fresh_seed_1042_satisfies_G1_G2_G4",
        },
        "planned_batches": {
            "primary": {"max_trades_per_session": 3, "policies": list(range(7)), "seeds": SELECTION_SEEDS},
            "nearby_conservative_if_needed": {
                "max_trades_per_session": 1,
                "policies": list(range(7)),
                "seeds": SELECTION_SEEDS,
                "trigger": "primary_batch_has_no_real_signal",
            },
        },
        "null_canary_references": null_canary_refs(),
        "input_readiness_status_at_preregistration": input_gate.get("status"),
        "side_effect_policy": side_effect_policy(),
        "highest_allowed_claim": "Group 1 non-VIX uplift testing complete",
    }


def render_preregistration(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Protocol101 Group 1 Non-VIX Uplift Attempt 001 Preregistration",
            "",
            f"- Status: `{payload['status']}`",
            f"- Attempt: `{payload['attempt_id']}`",
            f"- Contract: `{payload['contract']}`",
            f"- Transform: `{payload['model_facing_transform']}`",
            f"- Feature set hash: `{payload['hashes']['feature_set_hash']}`",
            f"- Static-ladder policy hash: `{payload['hashes']['static_ladder_policy_hash']}`",
            f"- Split hash: `{payload['hashes']['split_hash']}`",
            f"- Model family: `{payload['model']['family']}`",
            f"- Policies: `{sorted(payload['policies'])}`",
            f"- Seeds: `{payload['model']['seeds']}`",
            "",
            "This preregistration was written before Group 1 non-VIX uplift results were inspected.",
        ]
    ) + "\n"


def side_effect_policy(*, model_training: bool = False, threshold_selection: bool = False) -> dict[str, bool]:
    return {
        "model_training_executed": bool(model_training),
        "threshold_selection_executed": bool(threshold_selection),
        "feature_uplift_cv_executed": bool(model_training),
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_download": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
        "real_money_path_changed": False,
    }


def selected_ints(raw: str, *, default: list[int], allowed: set[int] | None = None) -> list[int]:
    if not raw:
        return list(default)
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if allowed is not None:
        unexpected = sorted(set(values) - set(allowed))
        if unexpected:
            raise ValueError(f"unexpected selection values: {unexpected}")
    return values


def selected_batches(args: argparse.Namespace, *, primary_real_signal: bool | None = None) -> list[str]:
    if args.smoke:
        return ["primary"]
    if args.batches:
        values = [part.strip() for part in str(args.batches).split(",") if part.strip()]
        unexpected = sorted(set(values) - {"primary", "conservative"})
        if unexpected:
            raise ValueError(f"unexpected batches: {unexpected}")
        return values
    if primary_real_signal is None:
        return ["primary"]
    return ["conservative"] if not primary_real_signal else []


def selected_policy_indexes(args: argparse.Namespace) -> list[int]:
    default = [0] if args.smoke else list(range(7))
    return selected_ints(args.policy_indexes, default=default, allowed=set(range(7)))


def selected_seeds(args: argparse.Namespace) -> list[int]:
    default = [SELECTION_SEEDS[0]] if args.smoke else list(SELECTION_SEEDS)
    return selected_ints(args.seeds, default=default)


def limited_fold_decisions(
    fold_decisions: dict[str, dict[str, list[Any]]],
    *,
    max_folds: int,
) -> dict[str, dict[str, list[Any]]]:
    if max_folds <= 0:
        return fold_decisions
    return {fold_id: fold_decisions[fold_id] for fold_id in sorted(fold_decisions)[:max_folds]}


def strike_from_contract_id(contract_id: str | None) -> float:
    if not contract_id:
        return float("nan")
    try:
        return float(str(contract_id).rsplit("-", 2)[-2])
    except (IndexError, TypeError, ValueError):
        return float("nan")


def strike_idx_from_offset(offset: float) -> int:
    diagnostic_static_offsets = np.arange(-50.0, 55.0, 5.0, dtype=np.float32)
    matches = np.where(np.isclose(diagnostic_static_offsets, float(offset)))[0]
    return int(matches[0]) if len(matches) else -1


def right_idx_from_right(right: str) -> int:
    return 0 if str(right) == "C" else 1 if str(right) == "P" else -1


def candidate_feature_hash(decision: Any, idx: int) -> str:
    values = np.asarray(decision.features[int(idx)], dtype=np.float32)
    rounded = np.round(np.nan_to_num(values, nan=0.0, posinf=999.0, neginf=-999.0), 8)
    return stable_hash(rounded.tolist())


def candidate_payload(decision: Any, idx: int) -> dict[str, Any]:
    contract_id = (
        str(decision.contract_ids[int(idx)])
        if getattr(decision, "contract_ids", None) is not None
        else ""
    )
    right = str(decision.rights[int(idx)])
    offset = float(decision.offsets[int(idx)])
    entry_ask = (
        float(decision.entry_asks[int(idx)])
        if getattr(decision, "entry_asks", None) is not None
        else float("nan")
    )
    return {
        "contract_id": contract_id,
        "right": right,
        "strike": strike_from_contract_id(contract_id),
        "offset": offset,
        "strike_idx": strike_idx_from_offset(offset),
        "right_idx": right_idx_from_right(right),
        "entry_ask": entry_ask,
        "feature_hash": candidate_feature_hash(decision, idx),
        "feature_group_version": "group1_stable_index_context_non_vix_subset_v1",
    }


def _policy_exit_deadline_utc(decision_time: datetime, policy_index: int) -> pd.Timestamp:
    from v4.dataset.spxw_0dte_neural import NeuralDatasetConfig

    cfg = NeuralDatasetConfig()
    policy = cfg.label_policies[int(policy_index)]
    decision_ts = pd.Timestamp(decision_time)
    if decision_ts.tzinfo is None:
        decision_ts = decision_ts.tz_localize("UTC")
    else:
        decision_ts = decision_ts.tz_convert("UTC")
    max_hold = decision_ts + pd.Timedelta(minutes=int(policy.max_hold_minutes))
    local_day = decision_ts.tz_convert("America/New_York").date()
    forced = pd.Timestamp.combine(local_day, cfg.forced_flat_before).tz_localize("America/New_York").tz_convert("UTC")
    return min(max_hold, forced)


class PathDiagnosticsCache:
    """Load normalized session quote paths lazily for selected-trade diagnostics."""

    def __init__(self, normalized_dir: Path) -> None:
        self.normalized_dir = Path(normalized_dir)
        self._session_cache: dict[str, pd.DataFrame] = {}

    def clear(self) -> None:
        self._session_cache.clear()

    def _path_for(self, session: str) -> Path:
        return self.normalized_dir / f"databento_spxw_0dte_{session}.parquet"

    def _session_frame(self, session: str) -> pd.DataFrame:
        if session in self._session_cache:
            return self._session_cache[session]
        path = self._path_for(session)
        if not path.exists():
            self._session_cache[session] = pd.DataFrame()
            return self._session_cache[session]
        columns = ["contract_id", "quote_time", "event_time", "bid", "ask", "mid"]
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception:
            frame = pd.read_parquet(path)
            frame = frame[[col for col in columns if col in frame.columns]]
        if frame.empty:
            self._session_cache[session] = frame
            return frame
        frame = frame.copy()
        source_time = frame["quote_time"] if "quote_time" in frame else frame["event_time"]
        frame["quote_time"] = pd.to_datetime(source_time, utc=True)
        for col in ("bid", "ask", "mid"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        self._session_cache[session] = frame.sort_values(["contract_id", "quote_time"]).reset_index(drop=True)
        return self._session_cache[session]

    def diagnostics(
        self,
        *,
        session: str,
        contract_id: str,
        decision_time: datetime,
        entry_ask: float,
        policy_index: int,
    ) -> dict[str, Any]:
        from v4.dataset.spxw_0dte_neural import NeuralDatasetConfig

        cfg = NeuralDatasetConfig()
        policy = cfg.label_policies[int(policy_index)]
        frame = self._session_frame(session)
        base = {
            "path_status": "missing",
            "exit_timestamp": None,
            "exit_bid": None,
            "gross_pnl_recomputed": None,
            "exit_reason": "path_unavailable",
            "forced_flat": False,
            "mfe": None,
            "mae": None,
            "max_favorable_timestamp": None,
            "max_adverse_timestamp": None,
            "exit_efficiency": None,
            "profitable_before_ending_negative": None,
            "time_to_best_minutes": None,
            "path_quote_count": 0,
        }
        if frame.empty or not contract_id or not np.isfinite(entry_ask):
            return base
        decision_ts = pd.Timestamp(decision_time)
        if decision_ts.tzinfo is None:
            decision_ts = decision_ts.tz_localize("UTC")
        else:
            decision_ts = decision_ts.tz_convert("UTC")
        deadline = _policy_exit_deadline_utc(decision_time, int(policy_index))
        path = frame[
            (frame["contract_id"].astype(str) == str(contract_id))
            & (frame["quote_time"] > decision_ts)
            & (frame["quote_time"] <= deadline)
        ].sort_values("quote_time")
        if path.empty:
            return {**base, "path_status": "empty"}
        bids = pd.to_numeric(path["bid"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        quote_times = list(path["quote_time"])
        gross_path = (bids - float(entry_ask)) * float(cfg.contract_multiplier)
        stop_bid = float(entry_ask) * (1.0 - float(policy.stop_loss_pct))
        target_bid = float(entry_ask) * (1.0 + float(policy.take_profit_pct))
        hit_mask = (bids <= stop_bid) | (bids >= target_bid)
        if bool(hit_mask.any()):
            exit_rel = int(np.flatnonzero(hit_mask)[0])
        else:
            exit_rel = len(path) - 1
        exit_bid = float(bids[exit_rel])
        exit_ts = pd.Timestamp(quote_times[exit_rel])
        if exit_bid <= stop_bid:
            exit_reason = "stop_loss"
        elif exit_bid >= target_bid:
            exit_reason = "take_profit"
        elif exit_ts >= deadline:
            forced_deadline = deadline.tz_convert("America/New_York").time() == cfg.forced_flat_before
            exit_reason = "forced_flat" if forced_deadline else "max_hold"
        else:
            exit_reason = "max_hold"
        mfe_idx = int(np.nanargmax(gross_path))
        mae_idx = int(np.nanargmin(gross_path))
        mfe = float(gross_path[mfe_idx])
        mae = float(gross_path[mae_idx])
        gross = float((exit_bid - float(entry_ask)) * float(cfg.contract_multiplier))
        return {
            **base,
            "path_status": "ok",
            "exit_timestamp": exit_ts.isoformat(),
            "exit_bid": exit_bid,
            "gross_pnl_recomputed": gross,
            "exit_reason": exit_reason,
            "forced_flat": bool(exit_reason == "forced_flat"),
            "mfe": mfe,
            "mae": mae,
            "max_favorable_timestamp": pd.Timestamp(quote_times[mfe_idx]).isoformat(),
            "max_adverse_timestamp": pd.Timestamp(quote_times[mae_idx]).isoformat(),
            "exit_efficiency": float(gross / mfe) if mfe > 0.0 else None,
            "profitable_before_ending_negative": bool(mfe > 0.0 and gross < 0.0),
            "time_to_best_minutes": float((pd.Timestamp(quote_times[mfe_idx]) - decision_ts).total_seconds() / 60.0),
            "path_quote_count": int(len(path)),
        }


def group1_nonvix_feature_matrix(decision) -> np.ndarray:
    rights = np.asarray([str(right) for right in decision.rights], dtype=object)
    is_call = rights == "C"
    is_put = rights == "P"
    market = np.asarray(decision.market_last, dtype=np.float32)

    def value(index: int) -> float:
        if index >= len(market):
            return 0.0
        raw = float(market[index])
        return raw if np.isfinite(raw) else 0.0

    spx_close = value(0)
    spx_vwap = value(2)
    omar = value(3)
    session_range = value(4)
    momentum_5m = value(5)
    momentum_15m = value(6)
    spx_denom = max(abs(spx_close), 1.0)
    range_denom = max(abs(session_range), 1.0)
    vwap_gap = spx_close - spx_vwap
    count = len(rights)
    spx_vwap_gap_points = np.full(count, vwap_gap, dtype=np.float32)
    spx_vwap_gap_bps = np.full(count, vwap_gap / spx_denom * 10_000.0, dtype=np.float32)
    spx_vwap_gap_over_session_range = np.full(count, vwap_gap / range_denom, dtype=np.float32)
    session_range_bps = np.full(count, session_range / spx_denom * 10_000.0, dtype=np.float32)
    momentum_5m_bps = np.full(count, momentum_5m / spx_denom * 10_000.0, dtype=np.float32)
    momentum_15m_bps = np.full(count, momentum_15m / spx_denom * 10_000.0, dtype=np.float32)
    momentum_5m_over_session_range = np.full(count, momentum_5m / range_denom, dtype=np.float32)
    momentum_15m_over_session_range = np.full(count, momentum_15m / range_denom, dtype=np.float32)
    omar_clipped = np.full(count, min(max(omar, -3.0), 3.0), dtype=np.float32)
    vwap_side_alignment = ((is_call & (vwap_gap > 0.0)) | (is_put & (vwap_gap < 0.0))).astype(np.float32)
    omar_side_alignment = ((is_call & (omar > 0.0)) | (is_put & (omar < 0.0))).astype(np.float32)
    momentum15_side_alignment = (
        (is_call & (momentum_15m > 0.0)) | (is_put & (momentum_15m < 0.0))
    ).astype(np.float32)
    return np.column_stack(
        [
            spx_vwap_gap_points,
            spx_vwap_gap_bps,
            spx_vwap_gap_over_session_range,
            session_range_bps,
            momentum_5m_bps,
            momentum_15m_bps,
            momentum_5m_over_session_range,
            momentum_15m_over_session_range,
            omar_clipped,
            vwap_side_alignment,
            omar_side_alignment,
            momentum15_side_alignment,
        ]
    ).astype(np.float32)


def append_group1_nonvix_features(decisions: list[Any]) -> list[Any]:
    return [
        replace(
            decision,
            features=np.hstack([np.asarray(decision.features, dtype=np.float32), group1_nonvix_feature_matrix(decision)]).astype(
                np.float32
            ),
        )
        for decision in decisions
    ]


def model_facing_decisions(decisions: list[Any], *, add_group1_nonvix: bool) -> list[Any]:
    masked = transform_decision_features(decisions, TRANSFORM)
    return append_group1_nonvix_features(masked) if add_group1_nonvix else masked


def load_fold_decisions(
    fold_paths: dict[str, dict[str, list[Path]]],
    *,
    policy_index: int,
    add_group1_nonvix: bool,
) -> dict[str, dict[str, list[Any]]]:
    out: dict[str, dict[str, list[Any]]] = {}
    for fold_id, splits in sorted(fold_paths.items()):
        out[fold_id] = {}
        for split in ("train", "validation"):
            raw = load_decisions(splits[split], policy_index=policy_index)
            out[fold_id][split] = model_facing_decisions(raw, add_group1_nonvix=add_group1_nonvix)
    return out


def config_for(policy_index: int, *, seed: int, max_trades_per_session: int, args: argparse.Namespace) -> PilotConfig:
    policy_name, cooldown = runner.POLICY_META[int(policy_index)]
    return replace(
        PilotConfig(),
        policy_index=int(policy_index),
        policy_name=policy_name,
        cooldown_minutes=cooldown,
        max_train_examples=int(args.max_train_examples),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        target_mode="return_on_premium_regression",
        target_clip=5.0,
        entry_filter="none",
        selection_mode="top_score",
        max_trades_per_session=int(max_trades_per_session),
        feature_transform=TRANSFORM,
        feature_noise_augmentation="none",
        seed=int(seed),
    )


def simulate_model_policy_diagnostics(
    decisions: list[Any],
    predictions: list[np.ndarray],
    *,
    threshold: float,
    config: PilotConfig,
    batch: str,
    fold_id: str,
    path_cache: PathDiagnosticsCache,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    skipped = {"positive_untraded_decisions": 0, "best_untraded_label_sum": 0.0}
    cash_gross = float(STARTING_CASH)
    equity_fee_adjusted = float(STARTING_CASH)
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    realized_pnl_by_session: dict[str, float] = {}
    ordered_pairs = sorted(zip(decisions, predictions), key=lambda item: (item[0].decision_time, item[0].session))
    for decision, pred in ordered_pairs:
        pred = np.asarray(pred, dtype=np.float32)
        finite = np.where(np.isfinite(pred))[0]
        decision_block_reasons: list[str] = []
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            decision_block_reasons.append("cooldown")
        if config.max_trades_per_session > 0 and trades_by_session.get(decision.session, 0) >= config.max_trades_per_session:
            decision_block_reasons.append("max_trades_per_session")
        if config.max_daily_loss > 0.0 and realized_pnl_by_session.get(decision.session, 0.0) <= -float(config.max_daily_loss):
            decision_block_reasons.append("max_daily_loss")
        affordable_mask = None
        if decision.entry_asks is not None:
            asks = np.asarray(decision.entry_asks, dtype=np.float32)
            affordable_mask = np.asarray(
                np.isfinite(asks) & (asks > 0.0) & ((asks * 100.0) <= cash_gross + 1e-9),
                dtype=bool,
            )
        ranked = sorted(finite, key=lambda idx: float(pred[int(idx)]), reverse=True)
        top = top_prediction(
            decision,
            pred,
            entry_filter=config.entry_filter,
            extra_allowed_mask=affordable_mask,
            selection_mode=config.selection_mode,
        )
        selected_idx = int(top[0]) if top is not None else None
        selected_score = float(top[1]) if top is not None else None
        selected_margin = float(top[2]) if top is not None else None
        for rank, idx in enumerate(ranked[: max(int(top_n), 1)], start=1):
            info = candidate_payload(decision, int(idx))
            score = float(pred[int(idx)])
            label = float(decision.labels[int(idx)])
            prediction_rows.append(
                {
                    "batch": batch,
                    "fold_id": fold_id,
                    "policy_index": int(config.policy_index),
                    "policy_name": config.policy_name,
                    "seed": int(config.seed),
                    "session": decision.session,
                    "decision_timestamp": decision.decision_time.isoformat(),
                    "candidate_rank": rank,
                    "is_selected_candidate": bool(selected_idx == int(idx) and not decision_block_reasons and score >= float(threshold)),
                    "selection_block_reasons": ";".join(decision_block_reasons),
                    "model_score": score,
                    "calibrated_confidence": score,
                    "abstention_signal": bool(score < float(threshold)),
                    "score_minus_threshold": score - float(threshold),
                    "threshold": float(threshold),
                    "label_net_pnl": label,
                    "selected_menu_v2_exit_shape": config.policy_name,
                    **info,
                }
            )
        if decision_block_reasons or len(pred) == 0 or top is None:
            if len(decision.labels) and np.isfinite(decision.labels).any():
                best_label = float(np.nanmax(decision.labels))
                if best_label > 0.0:
                    skipped["positive_untraded_decisions"] += 1
                    skipped["best_untraded_label_sum"] += best_label
            continue
        idx = int(selected_idx)
        score = float(selected_score)
        if config.max_score_ceiling > 0.0 and score >= float(config.max_score_ceiling):
            continue
        if score < float(threshold):
            if len(decision.labels) and np.isfinite(decision.labels).any():
                best_label = float(np.nanmax(decision.labels))
                if best_label > 0.0:
                    skipped["positive_untraded_decisions"] += 1
                    skipped["best_untraded_label_sum"] += best_label
            continue
        if config.min_score_margin > 0.0 and float(selected_margin or 0.0) < float(config.min_score_margin):
            continue
        info = candidate_payload(decision, idx)
        gross_pnl = float(decision.labels[idx])
        fees = float(ROUND_TRIP_FEE)
        fee_pnl = gross_pnl - fees
        equity_before = float(equity_fee_adjusted)
        cash_before = float(cash_gross)
        entry_ask = float(info["entry_ask"])
        path = path_cache.diagnostics(
            session=decision.session,
            contract_id=str(info["contract_id"]),
            decision_time=decision.decision_time,
            entry_ask=entry_ask,
            policy_index=int(config.policy_index),
        )
        equity_fee_adjusted += fee_pnl
        cash_gross += gross_pnl
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        realized_pnl_by_session[decision.session] = realized_pnl_by_session.get(decision.session, 0.0) + gross_pnl
        next_time_by_session[decision.session] = decision.decision_time + pd.Timedelta(minutes=config.cooldown_minutes).to_pytimedelta()
        premium = entry_ask * 100.0 if np.isfinite(entry_ask) else float("nan")
        trade_row = {
            "batch": batch,
            "fold_id": fold_id,
            "policy_index": int(config.policy_index),
            "policy_name": config.policy_name,
            "seed": int(config.seed),
            "session": decision.session,
            "entry_timestamp": decision.decision_time.isoformat(),
            "exit_timestamp": path.get("exit_timestamp"),
            "entry_ask": entry_ask,
            "exit_bid": path.get("exit_bid"),
            "fees": fees,
            "gross_pnl": gross_pnl,
            "fee_adjusted_pnl": fee_pnl,
            "return_on_premium": float(fee_pnl / premium) if np.isfinite(premium) and premium > 0.0 else None,
            "hold_duration_minutes": (
                float((pd.Timestamp(path["exit_timestamp"]) - pd.Timestamp(decision.decision_time)).total_seconds() / 60.0)
                if path.get("exit_timestamp")
                else None
            ),
            "exit_reason": path.get("exit_reason"),
            "forced_flat": bool(path.get("forced_flat")),
            "affordable": bool(affordable_mask[idx]) if affordable_mask is not None else True,
            "cash_before": cash_before,
            "cash_after_gross": cash_gross,
            "serial_account_equity_before": equity_before,
            "serial_account_equity_after": equity_fee_adjusted,
            "score": score,
            "model_score": score,
            "calibrated_confidence": score,
            "threshold": float(threshold),
            "score_margin": selected_margin,
            "strategy": "group1_nonvix_hgb_threshold",
            "selected_menu_v2_exit_shape": config.policy_name,
            **info,
        }
        trades.append({**trade_row, "pnl": gross_pnl, "decision_time": decision.decision_time.isoformat()})
        path_rows.append(
            {
                "batch": batch,
                "fold_id": fold_id,
                "policy_index": int(config.policy_index),
                "policy_name": config.policy_name,
                "seed": int(config.seed),
                "session": decision.session,
                "entry_timestamp": decision.decision_time.isoformat(),
                "contract_id": info["contract_id"],
                "right": info["right"],
                "strike": info["strike"],
                "offset": info["offset"],
                "strike_idx": info["strike_idx"],
                "gross_pnl_label": gross_pnl,
                "fee_adjusted_pnl": fee_pnl,
                "drawdown_contribution": min(fee_pnl, 0.0),
                **path,
            }
        )
    return trades, prediction_rows, path_rows, skipped


def train_fold(
    *,
    fold_id: str,
    decisions: dict[str, list[Any]],
    config: PilotConfig,
) -> dict[str, Any]:
    fit_decisions, calibration_decisions, fit_summary = runner._split_fit_and_calibration_decisions(
        decisions["train"],
        fit_mode="train_tail20_calibration",
    )
    state, predictions, preview = runner._train_sklearn_hist_gradient_boosting(
        fit_decisions,
        calibration_decisions,
        decisions["validation"],
        config=config,
    )
    threshold, threshold_sweep = runner.choose_threshold_with_rule(
        calibration_decisions,
        predictions["validation"],
        config=config,
        threshold_rule="max_validation_stressed_pnl",
        stress_per_trade=ADVERSE_STRESS_PER_TRADE,
    )
    validation_predictions = predictions["diagnostic_test"]
    trades = simulate_model_policy(
        decisions["validation"],
        validation_predictions,
        threshold=float(threshold),
        cooldown_minutes=config.cooldown_minutes,
        strategy="group1_nonvix_hgb_threshold",
        max_trades_per_session=config.max_trades_per_session,
        selection_mode=config.selection_mode,
    )
    baselines = baseline_metrics(decisions["validation"], config=config)
    return {
        "fold_id": fold_id,
        "chosen_threshold": float(threshold),
        "fit_calibration_summary": fit_summary,
        "model_family_preview": preview,
        "history": state.get("history") or [],
        "validation": {
            "metrics": metrics_for_trades(trades),
            "trades": [asdict(trade) for trade in trades],
        },
        "baselines": baselines,
        "threshold_sweep": threshold_sweep,
        "split_summary": {
            "train": {
                "sessions": len({decision.session for decision in decisions["train"]}),
                "decisions": len(decisions["train"]),
                "candidates": int(sum(len(decision.labels) for decision in decisions["train"])),
            },
            "validation": {
                "sessions": len({decision.session for decision in decisions["validation"]}),
                "decisions": len(decisions["validation"]),
                "candidates": int(sum(len(decision.labels) for decision in decisions["validation"])),
            },
        },
    }


def unit_key(*, batch: str, policy_index: int, seed: int, fold_id: str) -> str:
    return f"{batch}/policy{int(policy_index)}/seed{int(seed)}/{fold_id}"


def unit_checkpoint_dir(args: argparse.Namespace, *, batch: str, policy_index: int, seed: int, fold_id: str) -> Path:
    return args.out_dir / "checkpoints" / str(batch) / f"policy{int(policy_index)}" / f"seed{int(seed)}" / str(fold_id)


def checkpoint_config_hash(args: argparse.Namespace, *, batch: str, policy_index: int, seed: int, fold_id: str) -> str:
    return stable_hash(
        {
            "attempt_id": args.attempt_id,
            "batch": batch,
            "policy_index": int(policy_index),
            "seed": int(seed),
            "fold_id": fold_id,
            "contract": CONTRACT,
            "transform": TRANSFORM,
            "feature_set_hash": feature_set_hash(),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_train_examples": int(args.max_train_examples),
            "top_n_predictions": int(args.top_n_predictions),
        }
    )


def write_progress(
    args: argparse.Namespace,
    *,
    status: str,
    completed_units: int,
    total_units: int,
    last_unit: str | None = None,
    started_at_utc: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": "Protocol101Group1NonVixAttempt001ProgressV1",
        "attempt_id": args.attempt_id,
        "status": status,
        "updated_at_utc": now_utc(),
        "started_at_utc": started_at_utc,
        "completed_units": int(completed_units),
        "total_units": int(total_units),
        "last_completed_unit": last_unit,
        "model_training_executed": status not in {"initialized", "blocked"},
        "threshold_selection_executed": status not in {"initialized", "blocked"},
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "extra": extra or {},
    }
    write_json(args.out_dir / "progress.json", payload)


def _write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _trade_metrics_for_dicts(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trades": len(trades),
        "total_pnl": float(sum(float(row.get("gross_pnl") or row.get("pnl") or 0.0) for row in trades)),
        "fee_adjusted_pnl": float(sum(float(row.get("fee_adjusted_pnl") or 0.0) for row in trades)),
    }


def train_fold_instrumented(
    *,
    batch: str,
    fold_id: str,
    decisions: dict[str, list[Any]],
    config: PilotConfig,
    args: argparse.Namespace,
    path_cache: PathDiagnosticsCache,
) -> dict[str, Any]:
    checkpoint_dir = unit_checkpoint_dir(
        args,
        batch=batch,
        policy_index=config.policy_index,
        seed=config.seed,
        fold_id=fold_id,
    )
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    config_hash = checkpoint_config_hash(
        args,
        batch=batch,
        policy_index=config.policy_index,
        seed=config.seed,
        fold_id=fold_id,
    )
    if args.skip_completed and checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        if checkpoint.get("config_hash") == config_hash and checkpoint.get("status") == "complete":
            return checkpoint["fold_result"]

    started = time.perf_counter()
    fit_decisions, calibration_decisions, fit_summary = runner._split_fit_and_calibration_decisions(
        decisions["train"],
        fit_mode="train_tail20_calibration",
    )
    state, predictions, preview = runner._train_sklearn_hist_gradient_boosting(
        fit_decisions,
        calibration_decisions,
        decisions["validation"],
        config=config,
    )
    threshold, threshold_sweep = runner.choose_threshold_with_rule(
        calibration_decisions,
        predictions["validation"],
        config=config,
        threshold_rule="max_validation_stressed_pnl",
        stress_per_trade=ADVERSE_STRESS_PER_TRADE,
    )
    validation_predictions = predictions["diagnostic_test"]
    trades, prediction_rows, path_rows, skipped = simulate_model_policy_diagnostics(
        decisions["validation"],
        validation_predictions,
        threshold=float(threshold),
        config=config,
        batch=batch,
        fold_id=fold_id,
        path_cache=path_cache,
        top_n=int(args.top_n_predictions),
    )
    baselines = baseline_metrics(decisions["validation"], config=config)
    prediction_path = checkpoint_dir / "predictions.csv"
    trades_path = checkpoint_dir / "trades.csv"
    path_diagnostics_path = checkpoint_dir / "path_diagnostics.csv"
    _write_frame(prediction_path, prediction_rows)
    _write_frame(trades_path, trades)
    _write_frame(path_diagnostics_path, path_rows)
    result = {
        "fold_id": fold_id,
        "chosen_threshold": float(threshold),
        "fit_calibration_summary": fit_summary,
        "model_family_preview": preview,
        "history": state.get("history") or [],
        "validation": {
            "metrics": _trade_metrics_for_dicts(trades),
            "trades": trades,
        },
        "baselines": baselines,
        "threshold_sweep": threshold_sweep,
        "split_summary": {
            "train": {
                "sessions": len({decision.session for decision in decisions["train"]}),
                "decisions": len(decisions["train"]),
                "candidates": int(sum(len(decision.labels) for decision in decisions["train"])),
            },
            "validation": {
                "sessions": len({decision.session for decision in decisions["validation"]}),
                "decisions": len(decisions["validation"]),
                "candidates": int(sum(len(decision.labels) for decision in decisions["validation"])),
            },
        },
        "diagnostic_artifacts": {
            "predictions": artifact(prediction_path),
            "trades": artifact(trades_path),
            "path_diagnostics": artifact(path_diagnostics_path),
        },
        "skipped_opportunity": skipped,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    checkpoint = {
        "schema_version": "Protocol101Group1NonVixAttempt001FoldCheckpointV1",
        "status": "complete",
        "attempt_id": args.attempt_id,
        "unit": {
            "batch": batch,
            "policy_index": int(config.policy_index),
            "policy_name": config.policy_name,
            "seed": int(config.seed),
            "fold_id": fold_id,
        },
        "config_hash": config_hash,
        "completed_at_utc": now_utc(),
        "fold_result": result,
        "side_effect_policy": side_effect_policy(model_training=True, threshold_selection=True),
    }
    write_json(checkpoint_path, checkpoint)
    return result


def baseline_metrics(decisions: list[Any], *, config: PilotConfig) -> dict[str, Any]:
    out = {"random_valid": summarize_random_baseline(decisions, config=config)}
    for kind in ("atm_call", "atm_put", "vwap_omar"):
        trades = simulate_baseline(
            decisions,
            kind=kind,
            cooldown_minutes=config.cooldown_minutes,
            seed=config.seed,
        )
        out[kind] = metrics_for_trades(trades)
    return out


def adjusted_trade_pnls(trades: list[dict[str, Any]], *, fee: float) -> list[float]:
    return [float(trade.get("pnl") or 0.0) - float(fee) for trade in trades]


def strict_drawdown(pnls: list[float], *, starting_cash: float = STARTING_CASH) -> tuple[float, float, float]:
    equity = float(starting_cash)
    peak = float(starting_cash)
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    min_equity = float(starting_cash)
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        min_equity = min(min_equity, equity)
        dd = max(peak - equity, 0.0)
        max_dd_abs = max(max_dd_abs, dd)
        max_dd_pct = max(max_dd_pct, dd / peak if peak > 0.0 else float("inf"))
    return max_dd_abs, max_dd_pct, min_equity


def fold_fee_metrics(trades: list[dict[str, Any]], *, validation_sessions: int, fee: float) -> dict[str, Any]:
    pnls = adjusted_trade_pnls(trades, fee=fee)
    total = float(sum(pnls))
    dd_abs, dd_pct, min_equity = strict_drawdown(pnls)
    by_day: dict[str, float] = defaultdict(float)
    side_counts: Counter[str] = Counter()
    hour_counts: Counter[str] = Counter()
    offsets: list[float] = []
    wins = 0
    for trade, pnl in zip(trades, pnls):
        session = str(trade.get("session") or "")
        by_day[session] += float(pnl)
        side_counts[str(trade.get("right") or "UNKNOWN")] += 1
        try:
            hour_counts[str(pd.Timestamp(trade.get("decision_time")).tz_convert("America/New_York").strftime("%H"))] += 1
        except Exception:
            hour_counts["UNKNOWN"] += 1
        try:
            offsets.append(float(trade.get("offset") or 0.0))
        except (TypeError, ValueError):
            pass
        wins += int(float(pnl) > 0.0)
    gross_win = sum(pnl for pnl in pnls if pnl > 0.0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0.0))
    sessions = max(int(validation_sessions), 1)
    return {
        "trades": len(trades),
        "total_pnl": total,
        "avg_pnl": total / len(pnls) if pnls else 0.0,
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_win > 0 else 0.0),
        "max_drawdown_abs": float(dd_abs),
        "max_drawdown_pct_of_peak": float(dd_pct),
        "min_equity": float(min_equity),
        "no_ruin": bool(min_equity > 0.0),
        "trades_per_day": len(trades) / sessions,
        "sessions": sessions,
        "worst_day_pnl": min(by_day.values()) if by_day else 0.0,
        "side_counts": dict(sorted(side_counts.items())),
        "hour_counts": dict(sorted(hour_counts.items())),
        "offset_mean": float(sum(offsets) / len(offsets)) if offsets else 0.0,
        "top_day_profit_share": (
            max(by_day.values()) / total
            if by_day and total > 0.0
            else 0.0
        ),
    }


def fee_adjusted_baseline(metric: dict[str, Any], *, fee: float = ROUND_TRIP_FEE) -> float:
    return float(metric.get("total_pnl") or 0.0) - float(fee) * float(metric.get("trades") or 0.0)


def fee_adjusted_random(metric: dict[str, Any], *, fee: float = ROUND_TRIP_FEE) -> tuple[float, float]:
    mean = float(metric.get("total_pnl_mean") or 0.0) - float(fee) * float(metric.get("trades_mean") or 0.0)
    return mean, float(metric.get("total_pnl_std") or 0.0)


def ece_from_scores(trades: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    normalized_trades = []
    for trade in trades:
        if trade.get("score") is None and trade.get("model_score") is not None:
            normalized_trades.append({**trade, "score": trade.get("model_score")})
        else:
            normalized_trades.append(trade)
    return stage1.ece_from_trade_scores(normalized_trades, bins=bins)


def era_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = load_json(path)
    return {str(row.get("session")): str(row.get("era") or "UNKNOWN") for row in payload.get("sessions") or []}


def evaluate_attempt_result(
    *,
    batch: str,
    policy_index: int,
    seed: int,
    fold_results: dict[str, Any],
    eras: dict[str, str],
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    pooled_trades: list[dict[str, Any]] = []
    pooled_pnl = 0.0
    pooled_stress = 0.0
    pooled_random_mean = 0.0
    pooled_random_var = 0.0
    pooled_heuristic = 0.0
    fee_sensitivity = {str(fee): 0.0 for fee in [ROUND_TRIP_FEE, *FEE_SENSITIVITIES]}
    era_fold_pnls: dict[str, list[float]] = defaultdict(list)
    skipped_opportunity = {"positive_untraded_decisions": 0, "best_untraded_label_sum": 0.0}
    for fold_id, result in sorted(fold_results.items()):
        trades = list((result.get("validation") or {}).get("trades") or [])
        validation_sessions = int(((result.get("split_summary") or {}).get("validation") or {}).get("sessions") or 0)
        fee_metrics = fold_fee_metrics(trades, validation_sessions=validation_sessions, fee=ROUND_TRIP_FEE)
        stress_metrics = fold_fee_metrics(
            trades,
            validation_sessions=validation_sessions,
            fee=ROUND_TRIP_FEE + ADVERSE_STRESS_PER_TRADE,
        )
        pooled_trades.extend(trades)
        pooled_pnl += float(fee_metrics["total_pnl"])
        pooled_stress += float(stress_metrics["total_pnl"])
        for fee in fee_sensitivity:
            fee_sensitivity[fee] += fold_fee_metrics(trades, validation_sessions=validation_sessions, fee=float(fee))[
                "total_pnl"
            ]
        baselines = result.get("baselines") or {}
        random_mean, random_std = fee_adjusted_random(baselines.get("random_valid") or {})
        pooled_random_mean += random_mean
        pooled_random_var += random_std**2
        heuristic_values = [
            fee_adjusted_baseline(baselines.get(kind) or {})
            for kind in ("vwap_omar", "atm_call", "atm_put")
        ]
        pooled_heuristic += max(heuristic_values) if heuristic_values else 0.0
        for trade, pnl in zip(trades, adjusted_trade_pnls(trades, fee=ROUND_TRIP_FEE)):
            era_fold_pnls[eras.get(str(trade.get("session") or ""), "UNKNOWN")].append(float(pnl))
        fold_rows.append(
            {
                "fold_id": fold_id,
                "chosen_threshold": result.get("chosen_threshold"),
                "fee_metrics": fee_metrics,
                "stress_metrics": stress_metrics,
                "random_valid": baselines.get("random_valid") or {},
                "heuristic_best_fee_adjusted_pnl": max(heuristic_values) if heuristic_values else 0.0,
                "split_summary": result.get("split_summary") or {},
            }
        )
    random_std = math.sqrt(pooled_random_var)
    z_score = (pooled_pnl - pooled_random_mean) / random_std if random_std > 1e-9 else (float("inf") if pooled_pnl > pooled_random_mean else 0.0)
    positive_folds = sum(1 for row in fold_rows if float(row["fee_metrics"]["total_pnl"]) > 0.0)
    drawdown_max = max((float(row["fee_metrics"]["max_drawdown_pct_of_peak"]) for row in fold_rows), default=0.0)
    frequency_values = [float(row["fee_metrics"]["trades_per_day"]) for row in fold_rows]
    ece = ece_from_scores(pooled_trades)
    era_summary = {
        era: {
            "trade_count": len(values),
            "total_pnl": float(sum(values)),
            "median_trade_pnl": float(pd.Series(values).median()) if values else 0.0,
        }
        for era, values in sorted(era_fold_pnls.items())
    }
    negative_eras = [era for era, row in era_summary.items() if int(row["trade_count"]) > 0 and float(row["total_pnl"]) < 0.0]
    gates = {
        "G1_profitability": {
            "pass": bool(positive_folds >= 4 and pooled_pnl > 0.0),
            "positive_folds": positive_folds,
            "pooled_fee_adjusted_pnl": pooled_pnl,
        },
        "G2_beats_no_skill": {
            "pass": bool(z_score >= 3.0),
            "z_score": float(z_score),
            "random_fee_adjusted_mean": pooled_random_mean,
            "random_std": random_std,
        },
        "G3_beats_heuristic": {
            "pass": bool(pooled_pnl > pooled_heuristic),
            "heuristic_fee_adjusted_pnl": pooled_heuristic,
        },
        "G4_drawdown": {
            "pass": bool(drawdown_max <= 0.25),
            "owner_review_stage2_zone": bool(0.25 < drawdown_max <= 0.35),
            "hard_fail": bool(drawdown_max > 0.35),
            "max_drawdown_pct_of_peak": drawdown_max,
        },
        "G6_era_guard": {
            "pass": not negative_eras,
            "negative_eras": negative_eras,
            "decision": "regime_bound_requires_owner_review" if negative_eras else "pass",
        },
        "G7_frequency": {
            "pass": bool(frequency_values and min(frequency_values) >= 0.3 and max(frequency_values) <= 6.0),
            "min_trades_per_day": min(frequency_values) if frequency_values else 0.0,
            "max_trades_per_day": max(frequency_values) if frequency_values else 0.0,
            "mean_trades_per_day": float(sum(frequency_values) / len(frequency_values)) if frequency_values else 0.0,
            "conservative_rail_3_per_day_all_folds": bool(frequency_values and max(frequency_values) <= 3.0),
        },
        "G8_calibration": {
            "pass": bool(float(ece.get("ece") or 1.0) <= 0.10),
            **ece,
        },
    }
    return {
        "batch": batch,
        "policy_index": int(policy_index),
        "policy_name": runner.POLICY_META[int(policy_index)][0],
        "seed": int(seed),
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "pooled": {
            "fee_adjusted_pnl": pooled_pnl,
            "stress_fee_adjusted_pnl": pooled_stress,
            "fee_sensitivity_pnl": fee_sensitivity,
            "trades": len(pooled_trades),
            "random_z_score": float(z_score),
            "heuristic_fee_adjusted_pnl": pooled_heuristic,
            "no_ruin_all_folds": all(row["fee_metrics"]["no_ruin"] for row in fold_rows),
            "worst_day_pnl": min((row["fee_metrics"]["worst_day_pnl"] for row in fold_rows), default=0.0),
            "max_drawdown_pct_of_peak": drawdown_max,
            "side_counts": dict(sum((Counter(row["fee_metrics"]["side_counts"]) for row in fold_rows), Counter())),
            "hour_counts": dict(sum((Counter(row["fee_metrics"]["hour_counts"]) for row in fold_rows), Counter())),
            "top_day_profit_share_max": max((row["fee_metrics"]["top_day_profit_share"] for row in fold_rows), default=0.0),
        },
        "era_summary": era_summary,
        "skipped_opportunity": skipped_opportunity,
        "gates": gates,
    }


def run_attempt(
    *,
    batch: str,
    policy_index: int,
    seed: int,
    max_trades_per_session: int,
    fold_decisions: dict[str, dict[str, list[Any]]],
    eras: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = config_for(
        policy_index,
        seed=seed,
        max_trades_per_session=max_trades_per_session,
        args=args,
    )
    fold_results: dict[str, Any] = {}
    for fold_id, decisions in sorted(fold_decisions.items()):
        fold_results[fold_id] = train_fold(
            fold_id=fold_id,
            decisions=decisions,
            config=config,
        )
    evaluated = evaluate_attempt_result(
        batch=batch,
        policy_index=policy_index,
        seed=seed,
        fold_results=fold_results,
        eras=eras,
    )
    evaluated["config"] = asdict(config)
    evaluated["model_training_executed"] = True
    evaluated["threshold_selection_executed"] = True
    evaluated["side_effect_policy"] = side_effect_policy(model_training=True, threshold_selection=True)
    return evaluated


def run_attempt_instrumented(
    *,
    batch: str,
    policy_index: int,
    seed: int,
    max_trades_per_session: int,
    fold_decisions: dict[str, dict[str, list[Any]]],
    eras: dict[str, str],
    args: argparse.Namespace,
    progress: dict[str, Any],
) -> dict[str, Any]:
    config = config_for(
        policy_index,
        seed=seed,
        max_trades_per_session=max_trades_per_session,
        args=args,
    )
    path_cache = PathDiagnosticsCache(args.normalized_dir)
    fold_results: dict[str, Any] = {}
    for fold_id, decisions in sorted(fold_decisions.items()):
        fold_results[fold_id] = train_fold_instrumented(
            batch=batch,
            fold_id=fold_id,
            decisions=decisions,
            config=config,
            args=args,
            path_cache=path_cache,
        )
        path_cache.clear()
        progress["completed_units"] = int(progress.get("completed_units") or 0) + 1
        progress["last_unit"] = unit_key(batch=batch, policy_index=policy_index, seed=seed, fold_id=fold_id)
        write_progress(
            args,
            status="running",
            completed_units=int(progress["completed_units"]),
            total_units=int(progress["total_units"]),
            last_unit=str(progress["last_unit"]),
            started_at_utc=str(progress["started_at_utc"]),
        )
    evaluated = evaluate_attempt_result(
        batch=batch,
        policy_index=policy_index,
        seed=seed,
        fold_results=fold_results,
        eras=eras,
    )
    evaluated["config"] = asdict(config)
    evaluated["model_training_executed"] = True
    evaluated["threshold_selection_executed"] = True
    evaluated["side_effect_policy"] = side_effect_policy(model_training=True, threshold_selection=True)
    return evaluated


def evaluate_batch(attempts: list[dict[str, Any]], *, batch: str) -> dict[str, Any]:
    by_policy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_policy[int(row["policy_index"])].append(row)
    policy_rows: list[dict[str, Any]] = []
    for policy, rows in sorted(by_policy.items()):
        rows = sorted(rows, key=lambda item: int(item["seed"]))
        seed_gate_pairs = [
            bool(item["gates"]["G1_profitability"]["pass"])
            and float(item["gates"]["G2_beats_no_skill"]["z_score"]) >= 2.0
            for item in rows
        ]
        all_gates_except_g5_g9 = all(
            all(
                bool(item["gates"][gate]["pass"])
                for gate in (
                    "G1_profitability",
                    "G2_beats_no_skill",
                    "G3_beats_heuristic",
                    "G4_drawdown",
                    "G6_era_guard",
                    "G7_frequency",
                    "G8_calibration",
                )
            )
            for item in rows
        )
        policy_rows.append(
            {
                "policy_index": policy,
                "policy_name": runner.POLICY_META[policy][0],
                "seed_count": len(rows),
                "G5_seed_robustness": {
                    "pass": bool(len(rows) >= 3 and all(seed_gate_pairs)),
                    "seed_results": [
                        {
                            "seed": item["seed"],
                            "G1": item["gates"]["G1_profitability"]["pass"],
                            "G2_z": item["gates"]["G2_beats_no_skill"]["z_score"],
                            "G2_z_gte_2": item["gates"]["G2_beats_no_skill"]["z_score"] >= 2.0,
                            "fee_adjusted_pnl": item["pooled"]["fee_adjusted_pnl"],
                            "drawdown_pct": item["pooled"]["max_drawdown_pct_of_peak"],
                        }
                        for item in rows
                    ],
                },
                "eligible_before_G9": bool(all_gates_except_g5_g9 and len(rows) >= 3 and all(seed_gate_pairs)),
                "selection_metric_fee_adjusted_pnl_mean": float(
                    sum(float(item["pooled"]["fee_adjusted_pnl"]) for item in rows) / len(rows)
                )
                if rows
                else 0.0,
                "best_seed_fee_adjusted_pnl": max((float(item["pooled"]["fee_adjusted_pnl"]) for item in rows), default=0.0),
                "worst_seed_fee_adjusted_pnl": min((float(item["pooled"]["fee_adjusted_pnl"]) for item in rows), default=0.0),
            }
        )
    eligible = [row for row in policy_rows if row["eligible_before_G9"]]
    any_real_signal = any(
        bool(row["gates"]["G1_profitability"]["pass"]) and bool(row["gates"]["G2_beats_no_skill"]["pass"])
        for row in attempts
    )
    any_signal_drawdown_fail = any(
        bool(row["gates"]["G1_profitability"]["pass"])
        and bool(row["gates"]["G2_beats_no_skill"]["pass"])
        and not bool(row["gates"]["G4_drawdown"]["pass"])
        for row in attempts
    )
    best_policy = max(eligible, key=lambda item: item["selection_metric_fee_adjusted_pnl_mean"]) if eligible else None
    return {
        "schema_version": "Protocol101Group1NonVixBatchEvaluationV1",
        "batch": batch,
        "status": "pass" if eligible else "fail",
        "attempt_count": len(attempts),
        "policy_count": len(policy_rows),
        "policies": policy_rows,
        "attempts": attempts,
        "eligible_policies_before_G9": eligible,
        "best_policy_before_G9": best_policy,
        "real_signal_observed": bool(any_real_signal),
        "real_signal_but_drawdown_failed": bool(any_signal_drawdown_fail),
        "gate_counts": {
            gate: sum(1 for row in attempts if bool(row["gates"].get(gate, {}).get("pass")))
            for gate in (
                "G1_profitability",
                "G2_beats_no_skill",
                "G3_beats_heuristic",
                "G4_drawdown",
                "G6_era_guard",
                "G7_frequency",
                "G8_calibration",
            )
        },
    }


def run_batch_instrumented(
    *,
    batch: str,
    max_trades_per_session: int,
    fold_paths: dict[str, dict[str, list[Path]]],
    eras: dict[str, str],
    args: argparse.Namespace,
    registry_rows: list[dict[str, Any]],
    progress: dict[str, Any],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for policy_index in selected_policy_indexes(args):
        fold_decisions = load_fold_decisions(fold_paths, policy_index=policy_index, add_group1_nonvix=True)
        fold_decisions = limited_fold_decisions(fold_decisions, max_folds=int(args.max_folds))
        for seed in selected_seeds(args):
            attempt = run_attempt_instrumented(
                batch=batch,
                policy_index=policy_index,
                seed=seed,
                max_trades_per_session=max_trades_per_session,
                fold_decisions=fold_decisions,
                eras=eras,
                args=args,
                progress=progress,
            )
            attempts.append(attempt)
            registry_rows.append(registry_entry(attempt))
    return evaluate_batch(attempts, batch=batch)


def checkpoint_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((args.out_dir / "checkpoints").glob("*/*/*/*/checkpoint.json")):
        checkpoint = load_json(path)
        unit = checkpoint.get("unit") or {}
        artifacts = ((checkpoint.get("fold_result") or {}).get("diagnostic_artifacts") or {})
        rows.append(
            {
                "checkpoint_path": str(path),
                "checkpoint_sha256": file_sha256(path),
                "status": checkpoint.get("status"),
                "config_hash": checkpoint.get("config_hash"),
                **unit,
                "prediction_path": (artifacts.get("predictions") or {}).get("path"),
                "prediction_sha256": (artifacts.get("predictions") or {}).get("sha256"),
                "trades_path": (artifacts.get("trades") or {}).get("path"),
                "trades_sha256": (artifacts.get("trades") or {}).get("sha256"),
                "path_diagnostics_path": (artifacts.get("path_diagnostics") or {}).get("path"),
                "path_diagnostics_sha256": (artifacts.get("path_diagnostics") or {}).get("sha256"),
            }
        )
    return rows


def assemble_csv_from_checkpoints(args: argparse.Namespace, *, field: str, out_name: str) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    for row in checkpoint_rows(args):
        path = row.get(field)
        if path and Path(str(path)).exists():
            frames.append(pd.read_csv(path, low_memory=False))
    out_path = args.out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
    return {"path": str(out_path), "sha256": file_sha256(out_path), "rows": int(sum(len(frame) for frame in frames))}


def write_checkpoint_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = checkpoint_rows(args)
    manifest = {
        "schema_version": "Protocol101Group1NonVixAttempt001CheckpointManifestV1",
        "attempt_id": args.attempt_id,
        "generated_at_utc": now_utc(),
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }
    write_json(args.out_dir / "checkpoint_manifest.json", manifest)
    return manifest


def write_fold_month_attribution(args: argparse.Namespace, trades_artifact: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(trades_artifact["path"]))
    if not path.exists() or int(trades_artifact.get("rows") or 0) == 0:
        payload = {"schema_version": "Protocol101Group1NonVixFoldMonthAttributionV1", "rows": []}
        write_json(args.out_dir / "fold_month_attribution.json", payload)
        return payload
    frame = pd.read_csv(path)
    if frame.empty:
        rows: list[dict[str, Any]] = []
    else:
        frame["month"] = frame["session"].astype(str).str.slice(0, 7)
        rows = []
        group_cols = ["batch", "policy_index", "seed", "fold_id", "month"]
        for keys, group in frame.groupby(group_cols, dropna=False):
            batch, policy_index, seed, fold_id, month = keys
            pnl = pd.to_numeric(group["fee_adjusted_pnl"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "batch": batch,
                    "policy_index": int(policy_index),
                    "seed": int(seed),
                    "fold_id": fold_id,
                    "month": month,
                    "trades": int(len(group)),
                    "fee_adjusted_pnl": float(pnl.sum()),
                    "mean_fee_adjusted_pnl": float(pnl.mean()) if len(pnl) else 0.0,
                }
            )
    payload = {
        "schema_version": "Protocol101Group1NonVixFoldMonthAttributionV1",
        "attempt_id": args.attempt_id,
        "generated_at_utc": now_utc(),
        "rows": rows,
    }
    write_json(args.out_dir / "fold_month_attribution.json", payload)
    return payload


def write_reproducibility_manifest(
    args: argparse.Namespace,
    *,
    prereg: dict[str, Any],
    input_gate: dict[str, Any],
    assembled: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
) -> dict[str, Any]:
    files = {
        name: artifact(args.out_dir / name)
        for name in (
            "summary.json",
            "gate_results.json",
            "uplift_comparison.json",
            "experiment_registry.jsonl",
            "fold_predictions.csv",
            "fold_trades.csv",
            "path_diagnostics.csv",
            "fold_month_attribution.json",
            "checkpoint_manifest.json",
            "progress.json",
            "routing_decision.json",
        )
        if (args.out_dir / name).exists()
    }
    payload = {
        "schema_version": "Protocol101Group1NonVixAttempt001ReproducibilityManifestV1",
        "attempt_id": args.attempt_id,
        "generated_at_utc": now_utc(),
        "command": [sys.executable, *sys.argv],
        "contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "hashes": prereg.get("hashes") or {},
        "input_readiness_status": input_gate.get("status"),
        "assembled_artifacts": assembled,
        "checkpoint_count": checkpoint_manifest.get("checkpoint_count"),
        "files": files,
        "side_effect_policy": side_effect_policy(model_training=True, threshold_selection=True),
    }
    write_json(args.out_dir / "reproducibility_manifest.json", payload)
    return payload


def assemble_instrumented_artifacts(
    args: argparse.Namespace,
    *,
    prereg: dict[str, Any],
    input_gate: dict[str, Any],
) -> dict[str, Any]:
    predictions = assemble_csv_from_checkpoints(args, field="prediction_path", out_name="fold_predictions.csv")
    trades = assemble_csv_from_checkpoints(args, field="trades_path", out_name="fold_trades.csv")
    path_diagnostics = assemble_csv_from_checkpoints(
        args,
        field="path_diagnostics_path",
        out_name="path_diagnostics.csv",
    )
    checkpoint_manifest = write_checkpoint_manifest(args)
    month = write_fold_month_attribution(args, trades)
    assembled = {
        "fold_predictions": predictions,
        "fold_trades": trades,
        "path_diagnostics": path_diagnostics,
        "fold_month_attribution": artifact(args.out_dir / "fold_month_attribution.json"),
        "checkpoint_manifest": artifact(args.out_dir / "checkpoint_manifest.json"),
    }
    repro = write_reproducibility_manifest(
        args,
        prereg=prereg,
        input_gate=input_gate,
        assembled=assembled,
        checkpoint_manifest=checkpoint_manifest,
    )
    assembled["reproducibility_manifest"] = artifact(args.out_dir / "reproducibility_manifest.json")
    assembled["fold_month_rows"] = len(month.get("rows") or [])
    assembled["reproducibility_side_effect_policy"] = repro.get("side_effect_policy")
    return assembled


def feature_jitter_stability_sample(fold_paths: dict[str, dict[str, list[Path]]]) -> dict[str, Any]:
    first_fold = sorted(fold_paths)[0]
    first_path = fold_paths[first_fold]["validation"][0]
    raw = load_decisions([first_path], policy_index=0)
    sample = raw[: min(len(raw), 25)]
    baseline = model_facing_decisions(sample, add_group1_nonvix=True)
    jittered = model_facing_decisions(
        jitter_decision_features(sample, iv_delta=0.002, spread_delta=0.05, spread_frac_delta=0.005),
        add_group1_nonvix=True,
    )
    max_abs = 0.0
    compared = 0
    for left, right in zip(baseline, jittered):
        delta = np.asarray(left.features) - np.asarray(right.features)
        max_abs = max(max_abs, float(np.max(np.abs(delta))) if delta.size else 0.0)
        compared += int(delta.size)
    return {
        "status": "pass" if max_abs <= 1e-9 else "fail",
        "method": "raw_vendor_jitter_then_required_mask_plus_static_group1_nonvix",
        "sample_file": str(first_path),
        "sample_decisions": len(sample),
        "feature_values_compared": compared,
        "max_abs_feature_delta": max_abs,
    }


def run_learning_curve(
    *,
    best: dict[str, Any],
    fold_paths: dict[str, dict[str, list[Path]]],
    eras: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch = str(best["batch"])
    max_trades = 1 if batch == "conservative" else 3
    for fraction in (0.50, 0.75, 1.00):
        fraction_args = argparse.Namespace(**vars(args))
        fraction_args.max_train_examples = max(1, int(int(args.max_train_examples) * fraction))
        fold_decisions = load_fold_decisions(
            fold_paths,
            policy_index=int(best["policy_index"]),
            add_group1_nonvix=True,
        )
        attempt = run_attempt(
            batch=f"learning_curve_{fraction:.2f}",
            policy_index=int(best["policy_index"]),
            seed=int(best["seed"]),
            max_trades_per_session=max_trades,
            fold_decisions=fold_decisions,
            eras=eras,
            args=fraction_args,
        )
        rows.append(
            {
                "train_fraction": fraction,
                "max_train_examples": int(fraction_args.max_train_examples),
                "fee_adjusted_pnl": attempt["pooled"]["fee_adjusted_pnl"],
                "trades": attempt["pooled"]["trades"],
                "G1": attempt["gates"]["G1_profitability"]["pass"],
                "G2_z": attempt["gates"]["G2_beats_no_skill"]["z_score"],
                "G4_drawdown_pct": attempt["gates"]["G4_drawdown"]["max_drawdown_pct_of_peak"],
            }
        )
    return rows


def run_batch(
    *,
    batch: str,
    max_trades_per_session: int,
    fold_paths: dict[str, dict[str, list[Path]]],
    eras: dict[str, str],
    args: argparse.Namespace,
    registry_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for policy_index in range(7):
        fold_decisions = load_fold_decisions(fold_paths, policy_index=policy_index, add_group1_nonvix=True)
        for seed in SELECTION_SEEDS:
            attempt = run_attempt(
                batch=batch,
                policy_index=policy_index,
                seed=seed,
                max_trades_per_session=max_trades_per_session,
                fold_decisions=fold_decisions,
                eras=eras,
                args=args,
            )
            attempts.append(attempt)
            registry_rows.append(registry_entry(attempt))
    return evaluate_batch(attempts, batch=batch)


def registry_entry(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101Group1NonVixExperimentRegistryEntryV1",
        "attempt_id": f"{attempt['batch']}_policy{attempt['policy_index']}_seed{attempt['seed']}",
        "batch": attempt["batch"],
        "policy_index": attempt["policy_index"],
        "seed": attempt["seed"],
        "feature_set_hash": feature_set_hash(),
        "model_family": MODEL_FAMILY,
        "config": attempt["config"],
        "pooled": attempt["pooled"],
        "gates": attempt["gates"],
        "model_training_executed": True,
        "threshold_selection_executed": True,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
    }


def confirmation_run(
    *,
    best_policy: dict[str, Any],
    batch: str,
    fold_paths: dict[str, dict[str, list[Path]]],
    eras: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy = int(best_policy["policy_index"])
    max_trades = 1 if batch == "conservative" else 3
    fold_decisions = load_fold_decisions(fold_paths, policy_index=policy, add_group1_nonvix=True)
    attempt = run_attempt(
        batch="confirmation",
        policy_index=policy,
        seed=CONFIRMATION_SEED,
        max_trades_per_session=max_trades,
        fold_decisions=fold_decisions,
        eras=eras,
        args=args,
    )
    pass_g9 = (
        bool(attempt["gates"]["G1_profitability"]["pass"])
        and bool(attempt["gates"]["G2_beats_no_skill"]["pass"])
        and bool(attempt["gates"]["G4_drawdown"]["pass"])
    )
    return {
        "status": "pass" if pass_g9 else "fail",
        "required_seed": CONFIRMATION_SEED,
        "attempt": attempt,
    }


def outcome_from(primary: dict[str, Any], conservative: dict[str, Any] | None, confirmation: dict[str, Any] | None) -> dict[str, Any]:
    final_batch = conservative if conservative is not None else primary
    eligible = list(final_batch.get("eligible_policies_before_G9") or [])
    if eligible and confirmation and confirmation.get("status") == "pass":
        return {
            "outcome": "Outcome 1",
            "status": "pass",
            "decision": "eligible_offline_candidate",
            "paper_readiness_claim_allowed": False,
        }
    all_batches = [primary] + ([conservative] if conservative is not None else [])
    if any(batch.get("real_signal_but_drawdown_failed") for batch in all_batches):
        return {
            "outcome": "Outcome 2",
            "status": "stage2_route",
            "decision": "stage2_learned_exits_candidate",
            "paper_readiness_claim_allowed": False,
        }
    if conservative is not None and not primary.get("real_signal_observed") and not conservative.get("real_signal_observed"):
        return {
            "outcome": "Outcome 3",
            "status": "fail",
            "decision": "group1_nonvix_rejected_no_real_signal",
            "paper_readiness_claim_allowed": False,
        }
    return {
        "outcome": "Outcome 3",
        "status": "fail",
        "decision": "group1_nonvix_rejected_no_real_signal",
        "paper_readiness_claim_allowed": False,
    }


def comparison_payload(
    *,
    primary: dict[str, Any],
    conservative: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101Group1NonVixUpliftComparisonV1",
        "masked_v2_baseline_comparison": {
            "primary": load_json(args.stage1_primary),
            "conservative": load_json(args.stage1_conservative),
            "primary_artifact": artifact(args.stage1_primary),
            "conservative_artifact": artifact(args.stage1_conservative),
        },
        "group1_nonvix_uplift_comparison": {
            "primary": {
                "status": primary["status"],
                "gate_counts": primary["gate_counts"],
                "real_signal_observed": primary["real_signal_observed"],
                "best_policy_before_G9": primary.get("best_policy_before_G9"),
            },
            "conservative": (
                {
                    "status": conservative["status"],
                    "gate_counts": conservative["gate_counts"],
                    "real_signal_observed": conservative["real_signal_observed"],
                    "best_policy_before_G9": conservative.get("best_policy_before_G9"),
                }
                if conservative
                else None
            ),
        },
        "fixed_heuristic_comparison": "per_attempt_G3_uses_best_of_vwap_omar_atm_call_atm_put_same_folds",
        "random_null_canary_comparison": {
            "matched_random_null": "per_attempt_G2_uses_validation_random_valid_same_fold_policy_cooldown",
            "null_canary_references": null_canary_refs(),
        },
    }


def gate_results_payload(
    *,
    primary: dict[str, Any],
    conservative: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101Group1NonVixGateResultsV1",
        "status": outcome["status"],
        "outcome": outcome["outcome"],
        "decision": outcome["decision"],
        "primary": {
            "status": primary["status"],
            "gate_counts": primary["gate_counts"],
            "policy_summary": primary["policies"],
        },
        "conservative": (
            {
                "status": conservative["status"],
                "gate_counts": conservative["gate_counts"],
                "policy_summary": conservative["policies"],
            }
            if conservative
            else None
        ),
        "G9_confirmation": confirmation,
    }


def routing_decision_payload(
    *,
    summary: dict[str, Any],
    primary: dict[str, Any],
    conservative: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
    assembled: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_batches = [primary] + ([conservative] if conservative is not None else [])
    attempts = [attempt for batch in all_batches for attempt in (batch.get("attempts") or [])]
    positive_concentrated = any(
        float((attempt.get("pooled") or {}).get("fee_adjusted_pnl") or 0.0) > 0.0
        and float((attempt.get("pooled") or {}).get("top_day_profit_share_max") or 0.0) > 0.50
        for attempt in attempts
    )
    path_status_counts: dict[str, int] = {}
    path_summary: dict[str, Any] = {"path_status_counts": path_status_counts}
    path_artifact = (assembled or {}).get("path_diagnostics") or {}
    path_path = Path(str(path_artifact.get("path") or ""))
    if path_path.exists() and int(path_artifact.get("rows") or 0) > 0:
        frame = pd.read_csv(path_path, low_memory=False)
        path_status_counts = {
            str(key): int(value)
            for key, value in frame.get("path_status", pd.Series(dtype=object)).value_counts(dropna=False).items()
        }
        if "profitable_before_ending_negative" in frame:
            losing = frame[pd.to_numeric(frame.get("fee_adjusted_pnl"), errors="coerce").fillna(0.0) < 0.0]
            if len(losing):
                path_summary["losing_trade_profitable_before_ending_negative_pct"] = float(
                    losing["profitable_before_ending_negative"].astype(bool).mean()
                )
        path_summary["path_status_counts"] = path_status_counts
    decision = str((summary.get("outcome") or {}).get("decision") or "")
    if decision not in {
        "eligible_offline_candidate",
        "stage2_learned_exits_candidate",
        "group1_nonvix_rejected_no_real_signal",
        "rerun_required_artifact_or_reporting_issue",
    }:
        decision = "rerun_required_artifact_or_reporting_issue"
    if decision == "stage2_learned_exits_candidate" and positive_concentrated:
        decision = "group1_nonvix_rejected_no_real_signal"
    return {
        "schema_version": "Protocol101Group1NonVixRoutingDecisionV1",
        "attempt_id": summary.get("attempt_id"),
        "generated_at_utc": now_utc(),
        "routing_decision": decision,
        "status": summary.get("status"),
        "outcome": (summary.get("outcome") or {}).get("outcome"),
        "highest_allowed_claim": summary.get("highest_allowed_claim"),
        "decision_basis": [
            "G1-G9 gate results from five-fold chronological expanding-window CV",
            "matched random/null and fixed heuristic comparisons embedded in gate results",
            "strict serial drawdown, no-ruin, frequency, era, calibration, and seed robustness gates",
            "path diagnostics, concentration, side/time exposure, churn, skipped opportunity, and fee/stress sensitivity artifacts",
        ],
        "gate_summary": {
            "primary": (primary or {}).get("gate_counts") or {},
            "conservative": (conservative or {}).get("gate_counts") if conservative else None,
        },
        "positive_result_concentration_flag": bool(positive_concentrated),
        "path_diagnostic_summary": path_summary,
        "confirmation": confirmation,
        "required_artifacts": {
            "summary_json": "summary.json",
            "report_md": "report.md",
            "gate_results_json": "gate_results.json",
            "uplift_comparison_json": "uplift_comparison.json",
            "experiment_registry_jsonl": "experiment_registry.jsonl",
            "fold_predictions_csv": assembled.get("fold_predictions") if assembled else None,
            "fold_trades_csv": assembled.get("fold_trades") if assembled else None,
            "path_diagnostics_csv": assembled.get("path_diagnostics") if assembled else None,
            "fold_month_attribution_json": assembled.get("fold_month_attribution") if assembled else None,
            "reproducibility_manifest_json": assembled.get("reproducibility_manifest") if assembled else None,
            "routing_decision_json": "routing_decision.json",
        },
        "claims_not_made": {
            "paper_readiness": False,
            "promotion_readiness": False,
            "live_trading_readiness": False,
            "real_money_readiness": False,
        },
        "side_effect_policy": summary.get("side_effect_policy") or {},
    }


def policy_usage_distribution(*batches: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for batch in batches:
        if not batch:
            continue
        for attempt in batch.get("attempts") or []:
            key = (str(attempt["batch"]), int(attempt["policy_index"]))
            row = rows.setdefault(
                key,
                {
                    "batch": key[0],
                    "policy_index": key[1],
                    "policy_name": attempt.get("policy_name"),
                    "seed_count": 0,
                    "total_trades": 0,
                    "fee_adjusted_pnl_sum": 0.0,
                    "side_counts": Counter(),
                    "hour_counts": Counter(),
                },
            )
            row["seed_count"] += 1
            row["total_trades"] += int((attempt.get("pooled") or {}).get("trades") or 0)
            row["fee_adjusted_pnl_sum"] += float((attempt.get("pooled") or {}).get("fee_adjusted_pnl") or 0.0)
            row["side_counts"].update(Counter((attempt.get("pooled") or {}).get("side_counts") or {}))
            row["hour_counts"].update(Counter((attempt.get("pooled") or {}).get("hour_counts") or {}))
    out: list[dict[str, Any]] = []
    for row in rows.values():
        seed_count = max(int(row["seed_count"]), 1)
        out.append(
            {
                "batch": row["batch"],
                "policy_index": row["policy_index"],
                "policy_name": row["policy_name"],
                "seed_count": row["seed_count"],
                "total_trades": row["total_trades"],
                "mean_trades_per_seed": row["total_trades"] / seed_count,
                "mean_fee_adjusted_pnl": row["fee_adjusted_pnl_sum"] / seed_count,
                "side_counts": dict(sorted(row["side_counts"].items())),
                "hour_counts": dict(sorted(row["hour_counts"].items())),
            }
        )
    return sorted(out, key=lambda item: (item["batch"], item["policy_index"]))


def render_report(summary: dict[str, Any]) -> str:
    artifacts = summary.get("diagnostic_artifacts") or {}
    routing = summary.get("routing_decision") or {}
    lines = [
        f"# Protocol101 Group 1 Non-VIX Uplift {summary.get('attempt_id', ATTEMPT_ID)}",
        "",
        f"- Status: `{summary['status']}`",
        f"- Outcome: `{summary['outcome']['outcome']}`",
        f"- Decision: `{summary['outcome']['decision']}`",
        f"- Routing decision: `{routing.get('routing_decision', summary['outcome']['decision'])}`",
        f"- Highest allowed claim: `{summary['highest_allowed_claim']}`",
        f"- Contract: `{summary['contract']}`",
        f"- Transform: `{summary['model_facing_transform']}`",
        f"- Feature set hash: `{summary['hashes']['feature_set_hash']}`",
        "",
        "## Masked-v2 Baseline Comparison",
        "",
        f"- Baseline artifacts are embedded in `{summary['uplift_comparison_path']}`.",
        f"- Stage-1 masked primary/conservative comparison is unchanged and used only as the frozen control.",
        "",
        "## Group 1 Non-VIX Uplift Comparison",
        "",
        f"- Added features: `{GROUP1_NONVIX_FEATURE_NAMES}`",
        f"- Excluded VIX features: `{EXCLUDED_VIX_FEATURES}`",
        f"- Uplift comparison artifact: `{summary['uplift_comparison_path']}`",
        "",
        "## Fixed Heuristic Comparison",
        "",
        "- G3 compares pooled fee-adjusted PnL against the best fixed heuristic baseline on the same folds.",
        f"- Gate details: `{summary['gate_results_path']}`",
        "",
        "## Random/Null/Canary Comparison",
        "",
        "- G2 uses the matched random-selection null for the same folds, policy cooldown, and selection game.",
        f"- Null/canary references: `{len(summary['null_canary_references'])}` all policies.",
        "",
        "## Batch Results",
        "",
    ]
    for name in ("primary", "conservative"):
        batch = summary["batches"].get(name)
        if not batch:
            lines.append(f"- `{name}`: not run.")
            continue
        lines.append(
            f"- `{name}`: status=`{batch['status']}` attempts=`{batch['attempt_count']}` "
            f"real_signal=`{batch['real_signal_observed']}` gate_counts=`{batch['gate_counts']}`"
        )
    lines.extend(["", "## Policy Usage", ""])
    lines.append("| Batch | Policy | Mean trades/seed | Mean fee PnL |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["policy_usage_distribution"]:
        lines.append(
            f"| {row['batch']} | {row['policy_index']} | {row['mean_trades_per_seed']:.1f} | {row['mean_fee_adjusted_pnl']:.2f} |"
        )
    lines.extend(["", "## All 7 Policy Results", ""])
    for name in ("primary", "conservative"):
        batch = summary["batches"].get(name)
        if not batch:
            continue
        for policy in batch.get("policies") or []:
            lines.append(
                f"- `{name}` policy `{policy['policy_index']}`: eligible_before_G9="
                f"`{str(policy.get('eligible_before_G9')).lower()}`, mean_fee_pnl="
                f"`{policy.get('selection_metric_fee_adjusted_pnl_mean')}`."
            )
    lines.extend(["", "## HGB Learning Curve", ""])
    if summary.get("learning_curve"):
        lines.append("| Train fraction | Max train examples | Fee PnL | Trades | G1 | G2 z | G4 DD |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary.get("learning_curve") or []:
            lines.append(
                f"| `{row['train_fraction']}` | `{row['max_train_examples']}` | `{row['fee_adjusted_pnl']}` "
                f"| `{row['trades']}` | `{row['G1']}` | `{row['G2_z']}` | `{row['G4_drawdown_pct']}` |"
            )
    else:
        lines.append("- Learning curve rows: `0`.")
    lines.extend(
        [
            "",
            "## Fee And Stress",
            "",
            f"- Primary fee overlay: `${ROUND_TRIP_FEE}` round trip.",
            f"- Fee sensitivity overlays: `{FEE_SENSITIVITIES}`.",
            f"- Adverse bid/ask stress per trade: `${ADVERSE_STRESS_PER_TRADE}`.",
            "- Per-attempt fee sensitivity and stressed PnL are persisted in `experiment_registry.jsonl` and `gate_results.json`.",
            "",
            "## Risk Diagnostics",
            "",
            "- No-ruin / no cash exhaustion, worst day, drawdown, concentration, churn, side/time exposure, and skipped opportunity are evaluated inside the per-attempt pooled/fold gate rows.",
            f"- Fold trades artifact: `{(artifacts.get('fold_trades') or {}).get('path', 'fold_trades.csv')}`",
            f"- Fold month attribution artifact: `{(artifacts.get('fold_month_attribution') or {}).get('path', 'fold_month_attribution.json')}`",
            "",
            "## MFE/MAE Path Diagnostics",
            "",
            f"- Path diagnostics artifact: `{(artifacts.get('path_diagnostics') or {}).get('path', 'path_diagnostics.csv')}`",
            "",
            "## Feature-Jitter Stability",
            "",
            f"- Feature jitter stability: `{summary['feature_jitter_stability']['status']}` "
            f"max_abs_delta=`{summary['feature_jitter_stability']['max_abs_feature_delta']}`",
            "",
            "## Churn And Skipped Opportunity",
            "",
            "- Trade frequency, cooldown churn, and skipped positive-label opportunity are persisted per attempt and fold.",
            "",
            "## Forbidden Feature Status",
            "",
            "- VIX-change features remained excluded.",
            "- Group 2 geometry/moneyness, Greeks/IV, quote/liquidity/spread alpha, raw microstructure alpha, and volume/OI alpha remained blocked.",
            "",
            "## Side Effects",
            "",
        ]
    )
    for key, value in summary["side_effect_policy"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    return "\n".join(lines) + "\n"


def write_smoke_and_runtime_reports(
    args: argparse.Namespace,
    *,
    assembled: dict[str, Any],
    completed_units: int,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = assembled.get("fold_predictions") or {}
    trades = assembled.get("fold_trades") or {}
    paths = assembled.get("path_diagnostics") or {}
    path_status_counts: dict[str, int] = {}
    if paths.get("path") and Path(str(paths["path"])).exists() and int(paths.get("rows") or 0) > 0:
        frame = pd.read_csv(str(paths["path"]))
        path_status_counts = {
            str(key): int(value)
            for key, value in frame.get("path_status", pd.Series(dtype=object)).value_counts(dropna=False).items()
        }
    persistence_complete = (
        int(predictions.get("rows") or 0) > 0
        and int(trades.get("rows") or 0) > 0
        and int(paths.get("rows") or 0) == int(trades.get("rows") or -1)
        and bool(path_status_counts)
        and set(path_status_counts) <= {"ok"}
    )
    smoke = {
        "schema_version": "Protocol101Group1NonVixAttempt001SmokeTestReportV1",
        "attempt_id": args.attempt_id,
        "generated_at_utc": now_utc(),
        "status": "pass" if persistence_complete else "fail",
        "smoke_scope": {
            "folds": int(args.max_folds or 1),
            "policies": selected_policy_indexes(args),
            "seeds": selected_seeds(args),
            "batches": selected_batches(args),
        },
        "artifact_rows": {
            "fold_predictions": int(predictions.get("rows") or 0),
            "fold_trades": int(trades.get("rows") or 0),
            "path_diagnostics": int(paths.get("rows") or 0),
        },
        "path_status_counts": path_status_counts,
        "persistence_complete": bool(persistence_complete),
        "side_effect_policy": side_effect_policy(model_training=True, threshold_selection=True),
    }
    write_json(args.out_dir / "smoke_test_report.json", smoke)
    full_units = 2 * 7 * len(SELECTION_SEEDS) * 5
    checkpoint_elapsed = 0.0
    checkpoint_count = 0
    for row in checkpoint_rows(args):
        path = row.get("checkpoint_path")
        if not path:
            continue
        checkpoint = load_json(Path(str(path)))
        elapsed = ((checkpoint.get("fold_result") or {}).get("elapsed_seconds"))
        if elapsed is not None:
            checkpoint_elapsed += float(elapsed)
            checkpoint_count += 1
    seconds_per_unit = float(checkpoint_elapsed / max(checkpoint_count, 1))
    observed_non_unit_overhead = max(float(elapsed_seconds) - float(checkpoint_elapsed), 0.0)
    full_policy_batch_loads = 2 * 7
    modeled_full_seconds = seconds_per_unit * full_units + observed_non_unit_overhead * full_policy_batch_loads
    estimate = {
        "schema_version": "Protocol101Group1NonVixAttempt001RuntimeEstimateV1",
        "attempt_id": args.attempt_id,
        "generated_at_utc": now_utc(),
        "smoke_elapsed_seconds": float(elapsed_seconds),
        "completed_units": int(completed_units),
        "checkpoint_elapsed_seconds": float(checkpoint_elapsed),
        "checkpoint_count": int(checkpoint_count),
        "seconds_per_completed_fold_unit": seconds_per_unit,
        "observed_non_unit_overhead_seconds": float(observed_non_unit_overhead),
        "estimated_full_policy_batch_loads": int(full_policy_batch_loads),
        "estimated_full_units_primary_plus_conservative": int(full_units),
        "estimated_full_attempt_seconds": float(modeled_full_seconds),
        "estimated_full_attempt_hours": float(modeled_full_seconds / 3600.0),
        "simple_wall_clock_linear_hours": float(float(elapsed_seconds) * full_units / 3600.0),
        "full_attempt_allowed_under_4h": bool(modeled_full_seconds <= 4 * 3600),
        "estimation_method": (
            "checkpoint_fold_compute_seconds_per_unit_times_210_plus_observed_smoke_non_unit_overhead_times_14_policy_batch_loads"
        ),
    }
    write_json(args.out_dir / "runtime_estimate.json", estimate)
    return smoke, estimate


def write_performance_blocker_packet(
    args: argparse.Namespace,
    *,
    smoke: dict[str, Any],
    estimate: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if smoke.get("status") != "pass":
        reasons.append("smoke_artifact_persistence_incomplete")
    if not bool(estimate.get("full_attempt_allowed_under_4h")):
        reasons.append("estimated_full_attempt_runtime_exceeds_4h")
    payload = {
        "schema_version": "Protocol101Group1NonVixAttempt001PerformanceBlockerPacketV1",
        "attempt_id": args.attempt_id,
        "generated_at_utc": now_utc(),
        "status": "blocked",
        "decision": "do_not_run_full_attempt001_until_runtime_or_persistence_blocker_is_resolved",
        "blocker_reasons": reasons,
        "smoke_test_report": artifact(args.out_dir / "smoke_test_report.json"),
        "runtime_estimate": artifact(args.out_dir / "runtime_estimate.json"),
        "persistence_complete": bool(smoke.get("persistence_complete")),
        "estimated_full_attempt_hours": estimate.get("estimated_full_attempt_hours"),
        "estimated_full_attempt_seconds": estimate.get("estimated_full_attempt_seconds"),
        "runtime_limit_hours": 4.0,
        "recommended_next_action": (
            "Optimize deterministic orchestration/checkpoint reuse/parallel independent fold-units, "
            "or obtain explicit owner approval to exceed the 4h runtime stop before full attempt001."
        ),
        "side_effect_policy": side_effect_policy(model_training=True, threshold_selection=True),
    }
    write_json(args.out_dir / "performance_blocker_packet.json", payload)
    lines = [
        "# Protocol101 Group 1 Non-VIX Attempt001 Performance Blocker Packet",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Persistence complete: `{str(payload['persistence_complete']).lower()}`",
        f"- Estimated full attempt hours: `{payload['estimated_full_attempt_hours']}`",
        f"- Runtime limit hours: `{payload['runtime_limit_hours']}`",
        "",
        "## Blocker Reasons",
        "",
    ]
    for reason in reasons:
        lines.append(f"- `{reason}`")
    lines.extend(
        [
            "",
            "## Side Effects",
            "",
        ]
    )
    for key, value in payload["side_effect_policy"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    (args.out_dir / "performance_blocker_packet.md").write_text("\n".join(lines) + "\n")
    return payload


def run_instrumented_main(args: argparse.Namespace) -> int:
    if args.smoke:
        args.instrumented = True
        args.max_folds = 1 if int(args.max_folds or 0) <= 0 else int(args.max_folds)
        args.out_dir = DEFAULT_ATTEMPT002_OUT_DIR if args.out_dir == DEFAULT_OUT_DIR else args.out_dir
        args.attempt_id = ATTEMPT002_ID if args.attempt_id == ATTEMPT_ID else args.attempt_id
        args.skip_learning_curve = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in (
            "summary.json",
            "report.md",
            "experiment_registry.jsonl",
            "gate_results.json",
            "uplift_comparison.json",
            "fold_predictions.csv",
            "fold_trades.csv",
            "path_diagnostics.csv",
            "fold_month_attribution.json",
            "reproducibility_manifest.json",
            "progress.json",
            "checkpoint_manifest.json",
            "smoke_test_report.json",
            "runtime_estimate.json",
        ):
            target = args.out_dir / path
            if target.exists():
                target.unlink()

    fold_paths, governance, governance_blockers = resolve_fold_paths(args)
    input_gate = input_readiness(args, fold_paths, governance_blockers)
    write_json(args.out_dir / "input_readiness.json", input_gate)
    prereg_path = args.out_dir / "preregistration.json"
    if not prereg_path.exists() or args.force:
        prereg = preregistration_payload(args, input_gate)
        prereg["attempt_id"] = args.attempt_id
        prereg["highest_allowed_claim"] = "Group 1 non-VIX uplift testing complete"
        write_json(prereg_path, prereg)
        (args.out_dir / "preregistration.md").write_text(render_preregistration(prereg))
    else:
        prereg = load_json(prereg_path)
    if input_gate["status"] != "pass":
        summary = {
            "schema_version": "Protocol101Group1NonVixInstrumentedUpliftSummaryV1",
            "status": "blocked",
            "blockers": input_gate["blockers"],
            "attempt_id": args.attempt_id,
            "preregistration": str(prereg_path),
            "side_effect_policy": side_effect_policy(),
        }
        write_json(args.out_dir / "summary.json", summary)
        write_progress(args, status="blocked", completed_units=0, total_units=0, last_unit=None)
        return 2

    policy_indexes = selected_policy_indexes(args)
    seeds = selected_seeds(args)
    fold_count = min(int(args.max_folds or 0), len(fold_paths)) if int(args.max_folds or 0) > 0 else len(fold_paths)
    started_at = now_utc()
    started = time.perf_counter()
    total_units = max(1, len(policy_indexes) * len(seeds) * fold_count)
    if not args.smoke and not args.batches:
        total_units *= 2
    progress = {"completed_units": 0, "total_units": total_units, "started_at_utc": started_at, "last_unit": None}
    write_progress(args, status="initialized", completed_units=0, total_units=total_units, started_at_utc=started_at)

    registry_rows: list[dict[str, Any]] = []
    eras = era_lookup(args.era_manifest)
    jitter = feature_jitter_stability_sample(fold_paths)
    batches_to_run = selected_batches(args)
    primary = None
    conservative = None
    if "primary" in batches_to_run:
        primary = run_batch_instrumented(
            batch="primary",
            max_trades_per_session=3,
            fold_paths=fold_paths,
            eras=eras,
            args=args,
            registry_rows=registry_rows,
            progress=progress,
        )
    if args.batches:
        run_conservative = "conservative" in batches_to_run
    else:
        run_conservative = bool(primary is not None and not primary.get("real_signal_observed") and not args.smoke)
    if run_conservative:
        conservative = run_batch_instrumented(
            batch="conservative",
            max_trades_per_session=1,
            fold_paths=fold_paths,
            eras=eras,
            args=args,
            registry_rows=registry_rows,
            progress=progress,
        )
    if primary is None:
        primary = {
            "schema_version": "Protocol101Group1NonVixBatchEvaluationV1",
            "batch": "primary",
            "status": "not_run",
            "attempt_count": 0,
            "policy_count": 0,
            "policies": [],
            "attempts": [],
            "eligible_policies_before_G9": [],
            "best_policy_before_G9": None,
            "real_signal_observed": False,
            "real_signal_but_drawdown_failed": False,
            "gate_counts": {},
        }
    best_batch_name = "conservative" if conservative and conservative.get("eligible_policies_before_G9") else "primary"
    best_batch = conservative if best_batch_name == "conservative" else primary
    confirmation = None
    if not args.smoke and best_batch.get("best_policy_before_G9"):
        confirmation = confirmation_run(
            best_policy=best_batch["best_policy_before_G9"],
            batch=best_batch_name,
            fold_paths=fold_paths,
            eras=eras,
            args=args,
        )
        registry_rows.append(registry_entry(confirmation["attempt"]))
    outcome = (
        {
            "outcome": "smoke_only",
            "status": "smoke_complete",
            "decision": "instrumented_smoke_completed_no_full_search_executed",
            "paper_readiness_claim_allowed": False,
        }
        if args.smoke
        else outcome_from(primary, conservative, confirmation)
    )
    best_attempt_for_lc = None
    all_attempts = list(primary.get("attempts") or []) + list((conservative or {}).get("attempts") or [])
    if all_attempts:
        best_attempt_for_lc = max(all_attempts, key=lambda item: float(item["pooled"]["fee_adjusted_pnl"]))
    learning_curve = []
    if best_attempt_for_lc and not args.skip_learning_curve and not args.smoke:
        learning_curve = run_learning_curve(best=best_attempt_for_lc, fold_paths=fold_paths, eras=eras, args=args)
    registry = args.out_dir / "experiment_registry.jsonl"
    registry.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=True, default=str) + "\n" for row in registry_rows))
    comparison = comparison_payload(primary=primary, conservative=conservative, args=args)
    gates = gate_results_payload(primary=primary, conservative=conservative, confirmation=confirmation, outcome=outcome)
    summary = {
        "schema_version": "Protocol101Group1NonVixInstrumentedUpliftSummaryV1",
        "generated_at_utc": now_utc(),
        "status": outcome["status"],
        "attempt_id": args.attempt_id,
        "contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "feature_group": "stable_index_context_non_vix_subset",
        "highest_allowed_claim": "Group 1 non-VIX uplift testing complete",
        "outcome": outcome,
        "hashes": prereg["hashes"],
        "input_readiness": input_gate,
        "governance": governance,
        "batches": {
            "primary": {k: v for k, v in primary.items() if k != "attempts"},
            "conservative": ({k: v for k, v in conservative.items() if k != "attempts"} if conservative else None),
        },
        "G9_confirmation": confirmation,
        "policy_usage_distribution": policy_usage_distribution(primary, conservative),
        "learning_curve": learning_curve,
        "feature_jitter_stability": jitter,
        "null_canary_references": null_canary_refs(),
        "uplift_comparison_path": str(args.out_dir / "uplift_comparison.json"),
        "gate_results_path": str(args.out_dir / "gate_results.json"),
        "experiment_registry": str(registry),
        "side_effect_policy": side_effect_policy(model_training=True, threshold_selection=True),
        "forbidden_alpha_status": {
            "group1_nonvix_features_added": True,
            "vix_change_features_added": False,
            "group2_geometry_features_added": False,
            "greeks_iv_added": False,
            "quote_liquidity_spread_alpha_added": False,
            "raw_microstructure_alpha_added": False,
            "volume_oi_alpha_added": False,
        },
    }
    write_json(args.out_dir / "uplift_comparison.json", comparison)
    write_json(args.out_dir / "gate_results.json", gates)
    write_json(args.out_dir / "summary.json", summary)
    assembled = assemble_instrumented_artifacts(args, prereg=prereg, input_gate=input_gate)
    summary["diagnostic_artifacts"] = assembled
    routing = routing_decision_payload(
        summary=summary,
        primary=primary,
        conservative=conservative,
        confirmation=confirmation,
        assembled=assembled,
    )
    if routing["routing_decision"] != summary["outcome"]["decision"]:
        summary["outcome"]["decision"] = routing["routing_decision"]
        summary["status"] = "fail" if routing["routing_decision"] == "group1_nonvix_rejected_no_real_signal" else summary["status"]
        gates["decision"] = summary["outcome"]["decision"]
        gates["status"] = summary["status"]
    summary["routing_decision"] = routing
    summary["routing_decision_path"] = str(args.out_dir / "routing_decision.json")
    write_json(args.out_dir / "routing_decision.json", routing)
    write_json(args.out_dir / "gate_results.json", gates)
    checkpoint_manifest = load_json(args.out_dir / "checkpoint_manifest.json")
    repro = write_reproducibility_manifest(
        args,
        prereg=prereg,
        input_gate=input_gate,
        assembled=assembled,
        checkpoint_manifest=checkpoint_manifest,
    )
    assembled["reproducibility_manifest"] = artifact(args.out_dir / "reproducibility_manifest.json")
    assembled["reproducibility_side_effect_policy"] = repro.get("side_effect_policy")
    summary["diagnostic_artifacts"] = assembled
    write_json(args.out_dir / "summary.json", summary)
    if args.smoke:
        elapsed = time.perf_counter() - started
        smoke, estimate = write_smoke_and_runtime_reports(
            args,
            assembled=assembled,
            completed_units=int(progress["completed_units"]),
            elapsed_seconds=float(elapsed),
        )
        summary["smoke_test_report"] = smoke
        summary["runtime_estimate"] = estimate
        if smoke["status"] != "pass" or not estimate["full_attempt_allowed_under_4h"]:
            blocker = write_performance_blocker_packet(args, smoke=smoke, estimate=estimate)
            summary["status"] = "blocked_runtime_or_persistence"
            summary["outcome"] = {
                "outcome": "performance_or_persistence_blocker",
                "status": "blocked_runtime_or_persistence",
                "decision": "do_not_run_full_attempt001_until_smoke_persistence_and_runtime_gate_pass",
                "paper_readiness_claim_allowed": False,
            }
            summary["performance_blocker_packet"] = artifact(args.out_dir / "performance_blocker_packet.json")
            summary["performance_blocker_reasons"] = blocker.get("blocker_reasons")
        write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "report.md").write_text(render_report(summary))
    write_progress(
        args,
        status=str(summary["status"]),
        completed_units=int(progress["completed_units"]),
        total_units=int(progress["total_units"]),
        last_unit=progress.get("last_unit"),
        started_at_utc=started_at,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "outcome": summary["outcome"]["outcome"],
                "decision": summary["outcome"]["decision"],
                "diagnostic_artifacts": assembled,
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] not in {"blocked", "blocked_runtime_or_persistence"} else 2


def main() -> int:
    args = parse_args()
    if args.instrumented or args.smoke:
        return run_instrumented_main(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in (
            "summary.json",
            "report.md",
            "experiment_registry.jsonl",
            "gate_results.json",
            "uplift_comparison.json",
        ):
            target = args.out_dir / path
            if target.exists():
                target.unlink()

    fold_paths, governance, governance_blockers = resolve_fold_paths(args)
    input_gate = input_readiness(args, fold_paths, governance_blockers)
    write_json(args.out_dir / "input_readiness.json", input_gate)

    prereg_path = args.out_dir / "preregistration.json"
    if not prereg_path.exists() or args.force:
        prereg = preregistration_payload(args, input_gate)
        write_json(prereg_path, prereg)
        (args.out_dir / "preregistration.md").write_text(render_preregistration(prereg))
    else:
        prereg = load_json(prereg_path)
    if input_gate["status"] != "pass":
        summary = {
            "schema_version": "Protocol101Group1NonVixUpliftSummaryV1",
            "status": "blocked",
            "blockers": input_gate["blockers"],
            "preregistration": str(prereg_path),
            "side_effect_policy": side_effect_policy(),
        }
        write_json(args.out_dir / "summary.json", summary)
        return 2

    registry_rows: list[dict[str, Any]] = []
    eras = era_lookup(args.era_manifest)
    jitter = feature_jitter_stability_sample(fold_paths)
    primary = run_batch(
        batch="primary",
        max_trades_per_session=3,
        fold_paths=fold_paths,
        eras=eras,
        args=args,
        registry_rows=registry_rows,
    )
    conservative = None
    if not primary["real_signal_observed"]:
        conservative = run_batch(
            batch="conservative",
            max_trades_per_session=1,
            fold_paths=fold_paths,
            eras=eras,
            args=args,
            registry_rows=registry_rows,
        )
    best_batch_name = "conservative" if conservative and conservative.get("eligible_policies_before_G9") else "primary"
    best_batch = conservative if best_batch_name == "conservative" else primary
    confirmation = None
    if best_batch.get("best_policy_before_G9"):
        confirmation = confirmation_run(
            best_policy=best_batch["best_policy_before_G9"],
            batch=best_batch_name,
            fold_paths=fold_paths,
            eras=eras,
            args=args,
        )
        registry_rows.append(registry_entry(confirmation["attempt"]))
    outcome = outcome_from(primary, conservative, confirmation)
    best_attempt_for_lc = None
    all_attempts = list(primary.get("attempts") or []) + list((conservative or {}).get("attempts") or [])
    if all_attempts:
        best_attempt_for_lc = max(all_attempts, key=lambda item: float(item["pooled"]["fee_adjusted_pnl"]))
    learning_curve = []
    if best_attempt_for_lc and not args.skip_learning_curve:
        learning_curve = run_learning_curve(best=best_attempt_for_lc, fold_paths=fold_paths, eras=eras, args=args)

    registry = args.out_dir / "experiment_registry.jsonl"
    registry.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=True, default=str) + "\n" for row in registry_rows))
    comparison = comparison_payload(primary=primary, conservative=conservative, args=args)
    gates = gate_results_payload(primary=primary, conservative=conservative, confirmation=confirmation, outcome=outcome)
    summary = {
        "schema_version": "Protocol101Group1NonVixUpliftSummaryV1",
        "generated_at_utc": now_utc(),
        "status": outcome["status"],
        "attempt_id": ATTEMPT_ID,
        "contract": CONTRACT,
        "model_facing_transform": TRANSFORM,
        "feature_group": "stable_index_context_non_vix_subset",
        "highest_allowed_claim": "Group 1 non-VIX uplift testing complete",
        "outcome": outcome,
        "hashes": prereg["hashes"],
        "input_readiness": input_gate,
        "governance": governance,
        "batches": {
            "primary": {k: v for k, v in primary.items() if k != "attempts"},
            "conservative": ({k: v for k, v in conservative.items() if k != "attempts"} if conservative else None),
        },
        "G9_confirmation": confirmation,
        "policy_usage_distribution": policy_usage_distribution(primary, conservative),
        "learning_curve": learning_curve,
        "feature_jitter_stability": jitter,
        "null_canary_references": null_canary_refs(),
        "uplift_comparison_path": str(args.out_dir / "uplift_comparison.json"),
        "gate_results_path": str(args.out_dir / "gate_results.json"),
        "experiment_registry": str(registry),
        "side_effect_policy": side_effect_policy(model_training=True, threshold_selection=True),
        "forbidden_alpha_status": {
            "group1_nonvix_features_added": True,
            "vix_change_features_added": False,
            "group2_geometry_features_added": False,
            "greeks_iv_added": False,
            "quote_liquidity_spread_alpha_added": False,
            "raw_microstructure_alpha_added": False,
            "volume_oi_alpha_added": False,
        },
    }
    write_json(args.out_dir / "uplift_comparison.json", comparison)
    write_json(args.out_dir / "gate_results.json", gates)
    routing = routing_decision_payload(
        summary=summary,
        primary=primary,
        conservative=conservative,
        confirmation=confirmation,
        assembled=None,
    )
    if routing["routing_decision"] != summary["outcome"]["decision"]:
        summary["outcome"]["decision"] = routing["routing_decision"]
        summary["status"] = "fail" if routing["routing_decision"] == "group1_nonvix_rejected_no_real_signal" else summary["status"]
        gates["decision"] = summary["outcome"]["decision"]
        gates["status"] = summary["status"]
    summary["routing_decision"] = routing
    summary["routing_decision_path"] = str(args.out_dir / "routing_decision.json")
    write_json(args.out_dir / "routing_decision.json", routing)
    write_json(args.out_dir / "gate_results.json", gates)
    write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "report.md").write_text(render_report(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "outcome": outcome["outcome"],
                "decision": outcome["decision"],
                "primary_real_signal": primary["real_signal_observed"],
                "conservative_real_signal": conservative["real_signal_observed"] if conservative else None,
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
