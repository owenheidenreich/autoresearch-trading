"""Run the exact-contract Protocol101 bounded-HGB path.

``dry-run`` and ``plumbing-smoke`` are preparation modes. The smoke fits one
small disposable model through the same canonical unit used by the future
research batch and is never edge evidence. ``train-hypothesis`` is present but
requires a separate explicit owner flag and is not invoked by readiness work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.model.protocol101_canonical_stage1_contract import (
    CONTRACT_ID,
    FEATURE_NAMES,
    HYPOTHESES,
)
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_scoped_stage1_hgb import (
    HGBUnitConfig,
    load_decisions,
    load_repaired_decisions,
    run_hgb_unit,
    run_hgb_unit_v5,
    split_fit_calibration_sessions,
)
from v4.model.protocol101_serial_simulator import (
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
)
from v4.model.protocol101_regimen_repair import (
    TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    assert_fold_session_identities,
    assert_manifest_session_identities,
    build_exit_quote_age_report,
)
from v4.model.protocol101_repair_artifacts import (
    SCHEMA_VERSION as IMMUTABLE_REPLAY_SCHEMA_VERSION,
    semantic_payload_hashes,
    verify_replay_packet,
    write_immutable_replay_packet,
)
from v4.scripts.protocol101_training_scope import load_training_scope


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_READINESS = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_readiness/summary.json"
)
DEFAULT_OUT = BASE_AUDIT / "protocol101_scoped_canonical_stage1_plumbing_smoke"
FRESH_CAMPAIGN_NAMESPACE = (
    "protocol101_full_trader_stage1_entry_fresh_attempt001"
)
DEFAULT_FRESH_OUT = BASE_AUDIT / FRESH_CAMPAIGN_NAMESPACE
CAMPAIGN_ROOT = (
    BASE_AUDIT
    / "protocol101_full_trader_entry_campaign_preregistration_attempt001"
)
ACCEPTANCE_ROOT = (
    BASE_AUDIT
    / "protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001"
)
CAMPAIGN_PREREGISTRATION = CAMPAIGN_ROOT / "preregistration.json"
CAMPAIGN_CONTRACT = CAMPAIGN_ROOT / "campaign_contract.json"
CAMPAIGN_ROUTING = CAMPAIGN_ROOT / "routing_decision.json"
CAMPAIGN_HASHES = CAMPAIGN_ROOT / "hashes.sha256"
ACCEPTANCE_DECISION = ACCEPTANCE_ROOT / "acceptance_decision.json"
ACCEPTANCE_IDENTITY = ACCEPTANCE_ROOT / "identity_positive_controls.json"
ACCEPTANCE_BOUNDARY = ACCEPTANCE_ROOT / "boundary_attestation.json"
ACCEPTANCE_HASHES = ACCEPTANCE_ROOT / "hashes.sha256"
INTERSECTION_SUMMARY = (
    BASE_AUDIT / "protocol101_canonical_v1_intersection_guard_audit/summary.json"
)
NOISE_DISTRIBUTION = (
    BASE_AUDIT
    / "protocol101_canonical_v1_l0_l2_design_audit_attempt001/"
    "divergence_distributions.parquet"
)
SCIENTIFIC_RUNNER_PATH = Path(__file__)
RUNNER_PATH = SCIENTIFIC_RUNNER_PATH
CORE_PATH = Path("v4/model/protocol101_scoped_stage1_hgb.py")
CONTRACT_PATH = Path("v4/model/protocol101_canonical_stage1_contract.py")
SIMULATOR_PATH = Path("v4/model/protocol101_serial_simulator.py")
SIMULATOR_V5_PATH = Path("v4/model/protocol101_serial_simulator_v5.py")
IDENTITY_PATH = Path("v4/model/protocol101_regimen_repair.py")
ARTIFACT_PATH = Path("v4/model/protocol101_repair_artifacts.py")
GATE_AGGREGATOR_PATH = Path(
    "v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py"
)
SMOKE_VALIDATOR_PATH = Path(
    "v4/scripts/validate_protocol101_scoped_stage1_plumbing_smoke.py"
)
SEEDS = (42, 43, 44)
POLICIES = tuple(range(7))
CAMPAIGN_INPUT_HASHES = {
    CAMPAIGN_PREREGISTRATION: (
        "40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5"
    ),
    CAMPAIGN_CONTRACT: (
        "7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0"
    ),
    CAMPAIGN_ROOT / "runner_readiness_matrix.csv": (
        "a54e01419e470de30b6fea6a80af5af103ee035040352536411d6bcd8ef2d9f6"
    ),
    CAMPAIGN_ROOT / "runner_call_graph.json": (
        "d9373ba0bec3128e25420c70ef36c57adba833e08da357c22c1cde346429471c"
    ),
    CAMPAIGN_ROOT / "runner_gap_packet.json": (
        "eb62bcbf310842932976417de1f27c1882b3d3edfce04b9b6906861d2bcddf98"
    ),
    CAMPAIGN_ROUTING: (
        "056537be6312051420b51390070fe04800fb2046d0371dd05ce7a9384d661c6d"
    ),
    CAMPAIGN_HASHES: (
        "5a4f5d91489902b135dc39efd0bd50e26346e44dc84e3561291859a9355b3345"
    ),
}
DEFERRED_STACK_BLOCKERS = (
    "random_null_canary_reference_generation",
    "fixed_heuristic_baseline_generation",
    "G1_G8_aggregation",
    "hard_campaign_maxT",
    "D1_reporting",
    "D5_reporting",
    "D6_reporting",
    "independent_runner_acceptance",
    "cross_hypothesis_selection",
    "G9",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "plumbing-smoke", "train-hypothesis"),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FRESH_OUT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--hypothesis", choices=tuple(HYPOTHESES), default="H0")
    parser.add_argument("--owner-approved-plumbing-smoke", action="store_true")
    parser.add_argument("--owner-approved-offline-training", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-rows-per-session", type=int, default=80)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def side_effects(*, disposable_fit: bool = False, research_fit: bool = False) -> dict[str, bool]:
    return {
        "disposable_plumbing_model_fit_executed": bool(disposable_fit),
        "research_model_training_executed": bool(research_fit),
        "protected_holdout_read": False,
        "recorder_or_confirmation_data_read": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_downloaded": False,
        "promotion_or_default_changed": False,
        "runtime_or_launchd_changed": False,
        "real_money_path_changed": False,
    }


def code_hashes() -> dict[str, str]:
    return {
        str(path): sha256_path(path)
        for path in (
            RUNNER_PATH,
            CORE_PATH,
            CONTRACT_PATH,
            SIMULATOR_PATH,
            SIMULATOR_V5_PATH,
            IDENTITY_PATH,
            ARTIFACT_PATH,
            GATE_AGGREGATOR_PATH,
            SMOKE_VALIDATOR_PATH,
        )
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_relative_hash_manifest(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in parsed:
            raise RuntimeError(f"invalid checksum manifest: {path}")
        parsed[name] = digest
    return parsed


def _verify_relative_hash_manifest(root: Path, path: Path) -> dict[str, Any]:
    entries = _parse_relative_hash_manifest(path)
    mismatches: list[dict[str, str]] = []
    for name, expected in entries.items():
        target = root / name
        observed = sha256_path(target) if target.is_file() else "MISSING"
        if observed != expected:
            mismatches.append(
                {
                    "path": str(target),
                    "expected": expected,
                    "observed": observed,
                }
            )
    return {
        "path": str(path),
        "entry_count": len(entries),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def verify_fresh_campaign_inputs() -> dict[str, Any]:
    pinned: list[dict[str, Any]] = []
    for path, expected in CAMPAIGN_INPUT_HASHES.items():
        observed = sha256_path(path) if path.is_file() else "MISSING"
        pinned.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    campaign_manifest = _verify_relative_hash_manifest(
        CAMPAIGN_ROOT,
        CAMPAIGN_HASHES,
    )
    acceptance_manifest = _verify_relative_hash_manifest(
        ACCEPTANCE_ROOT,
        ACCEPTANCE_HASHES,
    )
    campaign = load_json(CAMPAIGN_CONTRACT)
    routing = load_json(CAMPAIGN_ROUTING)
    acceptance = load_json(ACCEPTANCE_DECISION)
    predicates = {
        "pinned_campaign_inputs_exact": all(
            row["status"] == "PASS" for row in pinned
        ),
        "campaign_manifest_exact": campaign_manifest["status"] == "PASS",
        "acceptance_manifest_exact": acceptance_manifest["status"] == "PASS",
        "campaign_route_exact": routing.get("terminal_route")
        == "fresh_entry_campaign_preregistered_runner_repair_required",
        "campaign_namespace_exact": campaign.get("campaign_namespace")
        == FRESH_CAMPAIGN_NAMESPACE,
        "campaign_sessions_exact": (
            campaign.get("corpus_and_folds", {}).get("campaign_sessions")
            == 271
        ),
        "campaign_units_exact": (
            campaign.get("registered_search", {}).get(
                "total_fresh_fitted_units"
            )
            == 420
        ),
        "old_model_reuse_forbidden": (
            campaign.get("registered_search", {}).get("model_reuse")
            == "forbidden_all_420_units_must_be_freshly_fit"
        ),
        "seed_45_protected": (
            campaign.get("registered_search", {}).get("g9_seed_status")
            == "protected_not_executable"
        ),
        "independent_acceptance_passed": (
            acceptance.get("terminal_route")
            == "repair_machinery_independently_accepted"
            and acceptance.get("status") == "PASS"
        ),
    }
    return {
        "schema_version": "Protocol101FreshCampaignInputReceiptV1",
        "status": (
            "PASS" if all(predicates.values()) else "FAIL"
        ),
        "predicates": predicates,
        "pinned_inputs": pinned,
        "campaign_manifest": campaign_manifest,
        "acceptance_manifest": acceptance_manifest,
    }


def fresh_campaign_identity_receipt(scope: Any) -> dict[str, Any]:
    manifest_receipt = assert_manifest_session_identities(
        [{"session": session} for session, _path in scope.sessions],
        boundary="fresh campaign manifest preflight before dataset load",
    )
    folds: list[dict[str, Any]] = []
    for fold in scope.folds:
        fit, calibration = split_fit_calibration_sessions(
            list(fold["train_sessions"])
        )
        folds.append(
            {
                "fold_id": str(fold["fold_id"]),
                "fit_sessions": fit,
                "calibration_sessions": calibration,
                "validation_sessions": list(fold["validation_sessions"]),
            }
        )
    fold_receipt = assert_fold_session_identities(
        folds,
        attempt_id=FRESH_CAMPAIGN_NAMESPACE,
        boundary="fresh campaign fold preflight before dataset load",
    )
    upstream = load_json(ACCEPTANCE_IDENTITY)
    controls = list(upstream.get("controls") or [])
    path_quote_control = next(
        (
            row
            for row in controls
            if row.get("id") == "duplicate_path"
        ),
        {},
    )
    if (
        upstream.get("status") != "PASS"
        or not controls
        or not all(bool(row.get("passed")) for row in controls)
        or not bool(path_quote_control.get("passed"))
    ):
        raise RuntimeError(
            "accepted upstream identity positive controls are not complete"
        )
    return {
        "schema_version": "Protocol101FreshCampaignIdentityReceiptV1",
        "status": "PASS",
        "boundary": "before any dataset load, fit, score, hash, or replay",
        "manifest": manifest_receipt,
        "folds": fold_receipt,
        "processed_row_identity_check": (
            "load_repaired_decisions calls assert_processed_row_identities "
            "before target construction"
        ),
        "path_quote_identity_authority": {
            "path": str(ACCEPTANCE_IDENTITY),
            "sha256": sha256_path(ACCEPTANCE_IDENTITY),
            "status": upstream["status"],
            "duplicate_path_positive_control": path_quote_control,
        },
    }


def fresh_campaign_provenance(scope: Any) -> dict[str, Any]:
    return {
        "campaign_namespace": FRESH_CAMPAIGN_NAMESPACE,
        "campaign_contract_sha256": sha256_path(CAMPAIGN_CONTRACT),
        "campaign_preregistration_sha256": sha256_path(
            CAMPAIGN_PREREGISTRATION
        ),
        "fold_governance_sha256": scope.fold_governance_hash,
        "acceptance_registry_sha256": scope.acceptance_registry_hash,
        "feature_contract_id": CONTRACT_ID,
        "feature_contract_source_sha256": sha256_path(CONTRACT_PATH),
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "simulator_source_sha256": sha256_path(SIMULATOR_V5_PATH),
        "two_clock_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "identity_contract_version": "Protocol101RegimenRepairIdentityV1",
        "identity_source_sha256": sha256_path(IDENTITY_PATH),
        "immutable_artifact_schema_version": (
            IMMUTABLE_REPLAY_SCHEMA_VERSION
        ),
        "immutable_artifact_source_sha256": sha256_path(ARTIFACT_PATH),
        "runner_core_source_sha256": sha256_path(CORE_PATH),
        "runner_source_sha256": sha256_path(RUNNER_PATH),
        "scientific_runner_source_sha256": sha256_path(
            SCIENTIFIC_RUNNER_PATH
        ),
    }


def readiness_blockers(readiness_path: Path) -> list[str]:
    if not readiness_path.exists():
        return [f"readiness_missing:{readiness_path}"]
    summary = load_json(readiness_path)
    blockers = list(summary.get("blockers") or [])
    if summary.get("status") != "ready_for_plumbing_smoke":
        blockers.append(f"readiness_status:{summary.get('status')}")
    if summary.get("contract_id") != CONTRACT_ID:
        blockers.append("readiness_contract_mismatch")
    if tuple(summary.get("feature_names") or ()) != FEATURE_NAMES:
        blockers.append("readiness_feature_list_mismatch")
    if any(bool(value) for value in (summary.get("side_effects") or {}).values()):
        blockers.append("readiness_forbidden_side_effect")
    return sorted(set(blockers))


def runner_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode in {"dry-run", "train-hypothesis"}:
        return fresh_runner_plan(args)
    scope = load_training_scope()
    blockers = readiness_blockers(args.readiness)
    if args.mode == "plumbing-smoke" and not args.owner_approved_plumbing_smoke:
        blockers.append("owner_approval_missing:plumbing_smoke")
    if args.mode == "train-hypothesis" and not args.owner_approved_offline_training:
        blockers.append("owner_approval_missing:offline_training")
    return {
        "schema_version": "Protocol101ScopedStage1HGBRunnerPlanV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "ready" if not blockers else "blocked",
        "mode": args.mode,
        "blockers": sorted(set(blockers)),
        "contract_id": CONTRACT_ID,
        "feature_names": list(FEATURE_NAMES),
        "hypotheses": {key: list(value) for key, value in HYPOTHESES.items()},
        "selected_hypothesis": args.hypothesis,
        "fold_governance_hash": scope.fold_governance_hash,
        "acceptance_registry_hash": scope.acceptance_registry_hash,
        "fold_count": len(scope.folds),
        "fold_geometry": [
            {
                "fold": item["fold"],
                "train": len(item["train_sessions"]),
                "validation": len(item["validation_sessions"]),
            }
            for item in scope.folds
        ],
        "fold_eligible_sessions": len(scope.sessions),
        "policies": list(POLICIES),
        "seeds": list(SEEDS),
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_VERSION,
        "model_family": "bounded_hist_gradient_boosting_regressor",
        "target": "fee_adjusted_return_on_premium",
        "threshold_source": "chronological_training_tail_only",
        "epsilon_source": "training_tail_0x_vs_1x_measured_noise_p95",
        "code_hashes": code_hashes(),
        "side_effects": side_effects(),
    }


def fresh_runner_call_graph() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FreshRunnerCallGraphV1",
        "public_entry": (
            "run_protocol101_scoped_stage1_hgb_runner."
            "run_fresh_hypothesis"
        ),
        "unit_entry": "protocol101_scoped_stage1_hgb.run_hgb_unit_v5",
        "replay_rungs": {
            "calibration_threshold_sweep": (
                "choose_threshold_v5 -> replay_candidates_v5 -> "
                "simulate_serial_candidates_v5"
            ),
            "primary_validation": (
                "selection_rows_v5 -> replay_candidates_v5 -> "
                "simulate_serial_candidates_v5"
            ),
            "fee_sensitivity": (
                "replay_candidates_v5_at_fee -> replay_candidates_v5 -> "
                "simulate_serial_candidates_v5"
            ),
            "noise_diagnostics": (
                "selection_rows_v5 -> replay_candidates_v5 -> "
                "simulate_serial_candidates_v5"
            ),
        },
        "forbidden_fresh_edges": [],
        "legacy_v4_helpers_dormant_on_fresh_path": True,
    }


def fresh_runner_plan(args: argparse.Namespace) -> dict[str, Any]:
    scope = load_training_scope()
    blockers: list[str] = []
    try:
        input_receipt = verify_fresh_campaign_inputs()
        if input_receipt["status"] != "PASS":
            blockers.append("frozen_campaign_input_mismatch")
    except Exception as exc:
        input_receipt = {"status": "FAIL", "error": str(exc)}
        blockers.append("frozen_campaign_input_verification_failed")
    try:
        identity_receipt = fresh_campaign_identity_receipt(scope)
    except Exception as exc:
        identity_receipt = {"status": "FAIL", "error": str(exc)}
        blockers.append("fresh_campaign_identity_preflight_failed")
    provenance = fresh_campaign_provenance(scope)
    if len(scope.sessions) != 271:
        blockers.append("campaign_session_count_mismatch")
    if len(scope.folds) != 5:
        blockers.append("campaign_fold_count_mismatch")
    if tuple(SEEDS) != (42, 43, 44):
        blockers.append("campaign_seed_mismatch")
    if 45 in SEEDS:
        blockers.append("protected_seed_45_exposed")
    fresh_root = DEFAULT_FRESH_OUT.resolve()
    requested_out = Path(args.out_dir).resolve()
    if requested_out != fresh_root and fresh_root not in requested_out.parents:
        blockers.append("fresh_output_namespace_mismatch")
    if args.mode == "train-hypothesis":
        if not args.owner_approved_offline_training:
            blockers.append("owner_approval_missing:offline_training")
        blockers.append("fresh_campaign_execution_authorization_not_present")
    return {
        "schema_version": "Protocol101FreshEntryRunnerV5PlanV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "v5_core_ready_pending_independent_acceptance"
            if not blockers
            else "blocked"
        ),
        "mode": args.mode,
        "core_blockers": sorted(set(blockers)),
        "blockers": sorted(set(blockers)),
        "deferred_stack_blockers": list(DEFERRED_STACK_BLOCKERS),
        "campaign_ready": False,
        "training_authorized": False,
        "contract_id": CONTRACT_ID,
        "feature_names": list(FEATURE_NAMES),
        "hypotheses": {key: list(value) for key, value in HYPOTHESES.items()},
        "selected_hypothesis": args.hypothesis,
        "fold_count": len(scope.folds),
        "fold_eligible_sessions": len(scope.sessions),
        "policies": list(POLICIES),
        "seeds": list(SEEDS),
        "seed_45_status": "protected_not_executable",
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "two_clock_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "target": "fee_adjusted_return_on_premium",
        "model_family": "bounded_hist_gradient_boosting_regressor",
        "fresh_call_graph": fresh_runner_call_graph(),
        "input_receipt": input_receipt,
        "identity_receipt": identity_receipt,
        "provenance": provenance,
        "code_hashes": code_hashes(),
        "side_effects": side_effects(),
    }


def preregistration(
    args: argparse.Namespace,
    *,
    plan: dict[str, Any],
    smoke_sessions: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101ScopedStage1HGBPreregistrationV1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "claim_scope": (
            "plumbing_only_no_edge_claim"
            if args.mode == "plumbing-smoke"
            else "owner_approved_offline_hypothesis_research"
        ),
        "contract_id": CONTRACT_ID,
        "selected_hypothesis": args.hypothesis,
        "feature_names": list(HYPOTHESES[args.hypothesis]),
        "unexpected_model_features_allowed": False,
        "quarantined_fields_available_only_for_guards_fills_labels_pnl_audit": True,
        "fold_governance_hash": plan["fold_governance_hash"],
        "acceptance_registry_hash": plan["acceptance_registry_hash"],
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_VERSION,
        "fee": 3.0,
        "fee_sensitivity": [2.60, 4.00],
        "daily_loss_fraction_of_session_start_equity": 0.05,
        "training_noise_scale": 1.0,
        "validation_noise": {
            "primary": 1.0,
            "diagnostics": [0.0, 0.5, 2.0],
            "drift_mining_rule": (
                "G1-G8 primary evidence is the 1.0x validation replay; "
                "clean 0x is diagnostic only"
            ),
        },
        "selection": {
            "k_action": 2.0,
            "k_slot": 2.0,
            "fallback": "nearest_atm_score_independent",
            "deterministic_tie_break": "score_then_strike_idx_then_right_idx",
        },
        "gate_aggregation": {
            "G1": "all_three_seeds_positive_on_at_least_4_of_5_folds_and_pooled_positive",
            "G2": "median_seed_pooled_pnl_z_gte_3_vs_matched_policy_null",
            "G3": "median_seed_pooled_pnl_gt_fixed_policy5_heuristic_baseline",
            "G4": "every_seed_pooled_calmar_gte_1_and_every_fold_min_equity_gte_5000",
            "G5": "all_seeds_pass_G1_and_worst_seed_null_z_gte_2",
            "G6": "no_seed_has_negative_median_test_fold_pnl_in_any_governed_era",
            "G7": "every_seed_fold_frequency_between_0.3_and_6.0_trades_per_day",
            "G8": "every_seed_weighted_oof_ece_lte_0.10",
            "G9": "not_run_in_initial_batch_requires_fresh_seed_after_selection",
            "seed_aggregation": "median_for_G2_G3_and_worst_seed_for_G1_G4_G5_G6_G7_G8",
        },
        "smoke_sessions": smoke_sessions,
        "code_hashes": plan["code_hashes"],
        "results_inspected_before_preregistration": False,
        "side_effects": side_effects(),
    }
    payload["preregistration_hash"] = stable_hash(payload)
    return payload


def fresh_preregistration(
    args: argparse.Namespace,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101FreshEntryRunnerPreregistrationV1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "campaign_namespace": FRESH_CAMPAIGN_NAMESPACE,
        "campaign_contract_sha256": plan["provenance"][
            "campaign_contract_sha256"
        ],
        "campaign_preregistration_sha256": plan["provenance"][
            "campaign_preregistration_sha256"
        ],
        "contract_id": CONTRACT_ID,
        "selected_hypothesis": args.hypothesis,
        "feature_names": list(HYPOTHESES[args.hypothesis]),
        "unexpected_model_features_allowed": False,
        "model_target": "fee_adjusted_payoff_return_on_premium",
        "fold_governance_sha256": plan["provenance"][
            "fold_governance_sha256"
        ],
        "acceptance_registry_sha256": plan["provenance"][
            "acceptance_registry_sha256"
        ],
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "simulator_source_sha256": plan["provenance"][
            "simulator_source_sha256"
        ],
        "two_clock_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "identity_receipt": plan["identity_receipt"],
        "fee": 3.0,
        "fee_sensitivity": [2.60, 3.00, 4.00],
        "training_noise_scale": 1.0,
        "validation_noise": {
            "primary": 1.0,
            "diagnostics": [0.0, 0.5, 2.0],
        },
        "selection": {
            "k_action": 2.0,
            "k_slot": 2.0,
            "fallback": "nearest_atm_score_independent",
            "deterministic_ordering": [
                "score",
                "strike_idx",
                "right_idx",
            ],
        },
        "gate_contract": {
            "G1_through_G7": "hard_downstream_not_executed_here",
            "G8": "required_report_only",
            "G9": "protected_not_executable",
            "maxT": "hard_downstream_not_executed_here",
        },
        "fresh_fit_only": True,
        "old_model_reuse": "forbidden",
        "seed_45": "protected",
        "deferred_stack_blockers": list(DEFERRED_STACK_BLOCKERS),
        "results_inspected_before_preregistration": False,
        "code_hashes": plan["code_hashes"],
        "side_effects": side_effects(),
    }
    payload["preregistration_hash"] = stable_hash(payload)
    return payload


def _scope_path_map(scope: Any) -> dict[str, Path]:
    return {session: path for session, path in scope.sessions}


def _session_paths(path_map: dict[str, Path], sessions: list[str]) -> list[tuple[str, Path]]:
    return [(session, path_map[session]) for session in sessions]


def guard_margins() -> dict[str, float]:
    summary = load_json(INTERSECTION_SUMMARY)
    if summary.get("status") != "pass":
        raise RuntimeError("intersection guard artifact not passed")
    return dict(
        summary["vendor_only_training_guard_policy"]["use_boundary_stable_margins"]
    )


def _write_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def _fresh_unit_payloads(result: dict[str, Any]) -> dict[str, Any]:
    validation = result["validation"]
    intents = list(validation["entry_intents"])
    trades = list(validation["trades"])
    skipped = list(validation["skipped_events"])
    quote_age_records = [
        {
            "policy_index": row["policy_index"],
            "session": row["session"],
            "valid": int(row.get("label_invalid_reason_code", 0)) == 0,
            "label_realized_exit_time_ns": row[
                "label_realized_exit_time_ns"
            ],
            "label_policy_deadline_ns": row["label_policy_deadline_ns"],
            "label_exit_quote_age_ms": row["label_exit_quote_age_ms"],
        }
        for row in intents
    ]
    session_metrics: dict[str, dict[str, Any]] = {}
    for trade in trades:
        session = str(trade["session"])
        row = session_metrics.setdefault(
            session,
            {"session": session, "trades": 0, "net_pnl": 0.0},
        )
        row["trades"] += 1
        row["net_pnl"] += float(
            trade["raw_label_pnl_after_campaign_fee"]
        )
    metrics = dict(validation["metrics"])
    return {
        "candidate_intents.jsonl": intents,
        "exit_quote_age_report.json": build_exit_quote_age_report(
            quote_age_records
        ),
        "fold_metrics.json": {
            "fold": result["fold"],
            "metrics": metrics,
        },
        "pooled_metrics.json": metrics,
        "session_metrics.json": {
            "sessions": [
                session_metrics[key] for key in sorted(session_metrics)
            ]
        },
        "skipped_events.jsonl": skipped,
        "trades.jsonl": trades,
    }


def commit_fresh_unit(
    unit_dir: Path,
    *,
    result: dict[str, Any],
    model_path: Path,
    fold: dict[str, Any],
    fit_sessions: list[str],
    calibration_sessions: list[str],
    validation_sessions: list[str],
    preregistration_payload: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    payloads = _fresh_unit_payloads(result)
    semantic = semantic_payload_hashes(payloads)
    model_hash = sha256_path(model_path)
    threshold = result["calibration"]["threshold"]
    epsilon = result["calibration"]["epsilon"]
    simulator_semantics = result["validation"]["simulator_semantics"]
    manifest_fields = {
        "campaign_namespace": FRESH_CAMPAIGN_NAMESPACE,
        "campaign_contract_sha256": provenance[
            "campaign_contract_sha256"
        ],
        "campaign_preregistration_sha256": provenance[
            "campaign_preregistration_sha256"
        ],
        "runner_preregistration_sha256": preregistration_payload[
            "preregistration_hash"
        ],
        "contract_id": CONTRACT_ID,
        "hypothesis": result["hypothesis"],
        "policy_index": int(result["policy_index"]),
        "seed": int(result["seed"]),
        "fold": int(fold["fold"]),
        "fold_id": str(fold["fold_id"]),
        "fit_sessions": fit_sessions,
        "calibration_sessions": calibration_sessions,
        "validation_sessions": validation_sessions,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "two_clock_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "identity_contract_version": provenance[
            "identity_contract_version"
        ],
        "processed_corpus_hash": stable_hash(
            {
                "fit": fit_sessions,
                "calibration": calibration_sessions,
                "validation": validation_sessions,
            }
        ),
        "fold_governance_hash": provenance[
            "fold_governance_sha256"
        ],
        "acceptance_registry_hash": provenance[
            "acceptance_registry_sha256"
        ],
        "feature_contract_hash": provenance[
            "feature_contract_source_sha256"
        ],
        "policy_contract_hash": stable_hash(
            {
                "policy_index": int(result["policy_index"]),
                "fee": result["config"]["fee"],
            }
        ),
        "two_clock_exit_contract_hash": stable_hash(
            {
                "schema": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
                "semantics": result["two_clock_exit_semantics"],
            }
        ),
        "model_or_equivalence_certificate_hash": model_hash,
        "threshold_hash": stable_hash({"threshold": threshold}),
        "epsilon_hash": stable_hash({"epsilon": epsilon}),
        "selection_contract_hash": stable_hash(
            {
                "k_action": result["config"]["k_action"],
                "k_slot": result["config"]["k_slot"],
                "fallback": "nearest_atm_score_independent",
                "ordering": ["score", "strike_idx", "right_idx"],
            }
        ),
        "simulator_source_hash": provenance[
            "simulator_source_sha256"
        ],
        "simulator_config_hash": simulator_semantics[
            "simulator_config_hash"
        ],
        **semantic,
    }
    packet_dir = unit_dir / "replay_packet"
    write_result = write_immutable_replay_packet(
        packet_dir,
        payloads=payloads,
        manifest_fields=manifest_fields,
    )
    verified_manifest = verify_replay_packet(packet_dir)
    unit_summary = {
        "schema_version": "Protocol101FreshEntryUnitSummaryV1",
        "status": "complete_pending_independent_acceptance",
        "campaign_namespace": FRESH_CAMPAIGN_NAMESPACE,
        "campaign_contract_sha256": provenance[
            "campaign_contract_sha256"
        ],
        "campaign_preregistration_sha256": provenance[
            "campaign_preregistration_sha256"
        ],
        "runner_preregistration_sha256": preregistration_payload[
            "preregistration_hash"
        ],
        "fold": int(fold["fold"]),
        "fold_id": str(fold["fold_id"]),
        "fit_sessions": fit_sessions,
        "calibration_sessions": calibration_sessions,
        "validation_sessions": validation_sessions,
        "unit": result,
        "model_artifact": {
            "path": str(model_path),
            "sha256": model_hash,
        },
        "replay_packet": {
            "path": str(packet_dir),
            "manifest_sha256": write_result.manifest_hash,
            "schema_version": verified_manifest["schema_version"],
            "simulator_version": verified_manifest[
                "simulator_version"
            ],
        },
        "provenance": provenance,
    }
    unit_summary["summary_hash"] = stable_hash(unit_summary)
    write_json(unit_dir / "summary.json", unit_summary)
    return unit_summary


def run_smoke(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    scope = load_training_scope()
    first_fold = scope.folds[0]
    fit_sessions = list(first_fold["train_sessions"][:2])
    calibration_sessions = list(first_fold["train_sessions"][2:3])
    validation_sessions = list(first_fold["validation_sessions"][:1])
    smoke_sessions = {
        "fit": fit_sessions,
        "calibration": calibration_sessions,
        "validation": validation_sessions,
    }
    prereg = preregistration(args, plan=plan, smoke_sessions=smoke_sessions)
    write_json(args.out_dir / "preregistration.json", prereg)
    write_json(
        args.out_dir / "progress.json",
        {"status": "loading_smoke_inputs", "preregistration_hash": prereg["preregistration_hash"]},
    )

    path_map = _scope_path_map(scope)
    margins = guard_margins()
    noise_model = DivergenceNoiseModel.from_parquet(NOISE_DISTRIBUTION)
    config = HGBUnitConfig(
        hypothesis="H3",
        policy_index=0,
        seed=42,
        max_iter=20,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=1.0,
        max_train_examples=5_000,
        training_noise_scale=1.0,
    )
    common = {
        "hypothesis": config.hypothesis,
        "policy_index": config.policy_index,
        "guard_margins": margins,
        "max_rows_per_session": int(args.smoke_rows_per_session),
    }
    fit = load_decisions(_session_paths(path_map, fit_sessions), **common)
    calibration = load_decisions(
        _session_paths(path_map, calibration_sessions),
        **common,
    )
    validation = load_decisions(
        _session_paths(path_map, validation_sessions),
        **common,
    )
    write_json(
        args.out_dir / "progress.json",
        {
            "status": "fitting_disposable_smoke_model",
            "fit_decisions": len(fit),
            "calibration_decisions": len(calibration),
            "validation_decisions": len(validation),
        },
    )
    model, unit = run_hgb_unit(
        fit_decisions=fit,
        calibration_decisions=calibration,
        validation_decisions=validation,
        noise_model=noise_model,
        config=config,
    )
    model_path = args.out_dir / "disposable_model.pkl"
    _write_model(model_path, model)
    summary = {
        "schema_version": "Protocol101ScopedStage1PlumbingSmokeV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "plumbing_smoke_complete_pending_independent_validation",
        "highest_allowed_claim": "exact-contract disposable plumbing smoke completed",
        "edge_claim_allowed": False,
        "paper_readiness_claim_allowed": False,
        "contract_id": CONTRACT_ID,
        "preregistration_hash": prereg["preregistration_hash"],
        "fold_governance_hash": scope.fold_governance_hash,
        "acceptance_registry_hash": scope.acceptance_registry_hash,
        "input_sessions": smoke_sessions,
        "input_counts": {
            "fit_decisions": len(fit),
            "calibration_decisions": len(calibration),
            "validation_decisions": len(validation),
            "fit_candidates": int(sum(len(item.labels) for item in fit)),
            "calibration_candidates": int(sum(len(item.labels) for item in calibration)),
            "validation_candidates": int(sum(len(item.labels) for item in validation)),
        },
        "unit": unit,
        "model_artifact": {
            "path": str(model_path),
            "sha256": sha256_path(model_path),
        },
        "code_hashes": code_hashes(),
        "side_effects": side_effects(disposable_fit=True),
    }
    summary["summary_hash"] = stable_hash(summary)
    write_json(args.out_dir / "summary.json", summary)
    write_json(
        args.out_dir / "progress.json",
        {"status": "complete", "summary_hash": summary["summary_hash"]},
    )
    (args.out_dir / "report.md").write_text(
        "# Protocol101 Scoped Stage-1 Plumbing Smoke\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Contract: `{CONTRACT_ID}`\n"
        f"- Features exercised: `{len(unit['feature_names'])}`\n"
        f"- Disposable fit examples: `{unit['fit']['fit_candidates_after_cap']}`\n"
        f"- Validation decisions: `{unit['validation']['decision_count']}`\n"
        f"- Validation trades: `{unit['validation']['metrics']['trades']}`\n"
        "- This packet is plumbing evidence only and makes no edge claim.\n"
    )
    return summary


def run_hypothesis(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    """Execute all governed units for one owner-approved hypothesis.

    This path writes resumable per-unit packets. It intentionally stops at raw
    unit evidence; the separately preregistered gate aggregator owns G1-G9.
    """
    scope = load_training_scope()
    prereg = preregistration(args, plan=plan)
    write_json(args.out_dir / "preregistration.json", prereg)
    path_map = _scope_path_map(scope)
    margins = guard_margins()
    noise_model = DivergenceNoiseModel.from_parquet(NOISE_DISTRIBUTION)
    unit_refs: list[dict[str, Any]] = []
    total = len(scope.folds) * len(POLICIES) * len(SEEDS)
    completed = 0
    for policy in POLICIES:
        for seed in SEEDS:
            for fold in scope.folds:
                fold_id = str(fold["fold_id"])
                train_sessions = list(fold["train_sessions"])
                fit_sessions, calibration_sessions = split_fit_calibration_sessions(
                    train_sessions
                )
                validation_sessions = list(fold["validation_sessions"])
                unit_dir = (
                    args.out_dir
                    / "units"
                    / f"policy{policy}"
                    / f"seed{seed}"
                    / fold_id
                )
                summary_path = unit_dir / "summary.json"
                if summary_path.exists() and not args.force:
                    unit_refs.append(
                        {
                            "path": str(summary_path),
                            "sha256": sha256_path(summary_path),
                        }
                    )
                    completed += 1
                    continue
                common = {
                    "hypothesis": args.hypothesis,
                    "policy_index": policy,
                    "guard_margins": margins,
                }
                fit = load_decisions(
                    _session_paths(path_map, fit_sessions),
                    **common,
                )
                calibration = load_decisions(
                    _session_paths(path_map, calibration_sessions),
                    **common,
                )
                validation = load_decisions(
                    _session_paths(path_map, validation_sessions),
                    **common,
                )
                config = HGBUnitConfig(
                    hypothesis=args.hypothesis,
                    policy_index=policy,
                    seed=seed,
                )
                model, result = run_hgb_unit(
                    fit_decisions=fit,
                    calibration_decisions=calibration,
                    validation_decisions=validation,
                    noise_model=noise_model,
                    config=config,
                )
                unit_dir.mkdir(parents=True, exist_ok=True)
                model_path = unit_dir / "model.pkl"
                _write_model(model_path, model)
                unit_summary = {
                    "fold": int(fold["fold"]),
                    "fold_id": fold_id,
                    "fit_sessions": fit_sessions,
                    "calibration_sessions": calibration_sessions,
                    "validation_sessions": validation_sessions,
                    "unit": result,
                    "model_artifact": {
                        "path": str(model_path),
                        "sha256": sha256_path(model_path),
                    },
                }
                write_json(summary_path, unit_summary)
                unit_refs.append(
                    {
                        "path": str(summary_path),
                        "sha256": sha256_path(summary_path),
                    }
                )
                completed += 1
                write_json(
                    args.out_dir / "progress.json",
                    {
                        "status": "running_owner_approved_hypothesis",
                        "completed_units": completed,
                        "total_units": total,
                    },
                )
    summary = {
        "schema_version": "Protocol101ScopedStage1HGBHypothesisUnitsV1",
        "status": "unit_execution_complete_pending_preregistered_gate_aggregation",
        "contract_id": CONTRACT_ID,
        "hypothesis": args.hypothesis,
        "preregistration_hash": prereg["preregistration_hash"],
        "unit_count": len(unit_refs),
        "expected_unit_count": total,
        "unit_artifacts": unit_refs,
        "code_hashes": code_hashes(),
        "side_effects": side_effects(research_fit=True),
    }
    summary["summary_hash"] = stable_hash(summary)
    write_json(args.out_dir / "summary.json", summary)
    return summary


def run_fresh_hypothesis(
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Execute fresh units through repaired rows and simulator v5 only."""

    if plan.get("identity_receipt", {}).get("status") != "PASS":
        raise RuntimeError("fresh identity preflight did not pass")
    if plan.get("input_receipt", {}).get("status") != "PASS":
        raise RuntimeError("fresh frozen-input verification did not pass")
    scope = load_training_scope()
    prereg = fresh_preregistration(args, plan=plan)
    write_json(args.out_dir / "preregistration.json", prereg)
    path_map = _scope_path_map(scope)
    margins = guard_margins()
    noise_model = DivergenceNoiseModel.from_parquet(NOISE_DISTRIBUTION)
    unit_refs: list[dict[str, Any]] = []
    total = len(scope.folds) * len(POLICIES) * len(SEEDS)
    completed = 0
    for policy in POLICIES:
        for seed in SEEDS:
            for fold in scope.folds:
                fold_id = str(fold["fold_id"])
                train_sessions = list(fold["train_sessions"])
                fit_sessions, calibration_sessions = (
                    split_fit_calibration_sessions(train_sessions)
                )
                validation_sessions = list(fold["validation_sessions"])
                unit_dir = (
                    args.out_dir
                    / "units"
                    / args.hypothesis
                    / f"policy{policy}"
                    / f"seed{seed}"
                    / fold_id
                )
                summary_path = unit_dir / "summary.json"
                if summary_path.exists() and not args.force:
                    raise RuntimeError(
                        "fresh unit resume must be verified by the durable "
                        f"v2 wrapper: {summary_path}"
                    )
                common = {
                    "hypothesis": args.hypothesis,
                    "policy_index": policy,
                    "guard_margins": margins,
                }
                fit = load_repaired_decisions(
                    _session_paths(path_map, fit_sessions),
                    split=f"{fold_id}:fit",
                    **common,
                )
                calibration = load_repaired_decisions(
                    _session_paths(path_map, calibration_sessions),
                    split=f"{fold_id}:calibration",
                    **common,
                )
                validation = load_repaired_decisions(
                    _session_paths(path_map, validation_sessions),
                    split=f"{fold_id}:validation",
                    **common,
                )
                config = HGBUnitConfig(
                    hypothesis=args.hypothesis,
                    policy_index=policy,
                    seed=seed,
                )
                model, result = run_hgb_unit_v5(
                    fit_decisions=fit,
                    calibration_decisions=calibration,
                    validation_decisions=validation,
                    noise_model=noise_model,
                    config=config,
                    fold=fold_id,
                )
                unit_dir.mkdir(parents=True, exist_ok=True)
                model_path = unit_dir / "model.pkl"
                _write_model(model_path, model)
                unit_summary = commit_fresh_unit(
                    unit_dir,
                    result=result,
                    model_path=model_path,
                    fold=fold,
                    fit_sessions=fit_sessions,
                    calibration_sessions=calibration_sessions,
                    validation_sessions=validation_sessions,
                    preregistration_payload=prereg,
                    provenance=plan["provenance"],
                )
                unit_refs.append(
                    {
                        "path": str(summary_path),
                        "sha256": sha256_path(summary_path),
                        "replay_manifest_sha256": unit_summary[
                            "replay_packet"
                        ]["manifest_sha256"],
                    }
                )
                completed += 1
                write_json(
                    args.out_dir / "progress.json",
                    {
                        "status": "running_fresh_v5_hypothesis",
                        "completed_units": completed,
                        "total_units": total,
                        "campaign_namespace": FRESH_CAMPAIGN_NAMESPACE,
                        "simulator_version": (
                            PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
                        ),
                    },
                )
    summary = {
        "schema_version": "Protocol101FreshEntryHypothesisUnitsV1",
        "status": "unit_execution_complete_pending_evidence_stack",
        "campaign_namespace": FRESH_CAMPAIGN_NAMESPACE,
        "campaign_contract_sha256": plan["provenance"][
            "campaign_contract_sha256"
        ],
        "campaign_preregistration_sha256": plan["provenance"][
            "campaign_preregistration_sha256"
        ],
        "runner_preregistration_sha256": prereg[
            "preregistration_hash"
        ],
        "contract_id": CONTRACT_ID,
        "hypothesis": args.hypothesis,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "two_clock_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "unit_count": len(unit_refs),
        "expected_unit_count": total,
        "unit_artifacts": unit_refs,
        "provenance": plan["provenance"],
        "code_hashes": code_hashes(),
        "deferred_stack_blockers": list(DEFERRED_STACK_BLOCKERS),
        "side_effects": side_effects(research_fit=True),
    }
    summary["summary_hash"] = stable_hash(summary)
    write_json(args.out_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = runner_plan(args)
    write_json(args.out_dir / "runner_plan.json", plan)
    if args.mode == "dry-run":
        print(json.dumps({"status": plan["status"], "blockers": plan["blockers"]}, indent=2))
        return 0
    if plan["blockers"]:
        raise SystemExit(f"runner blocked: {plan['blockers']}")
    if args.out_dir.exists() and (args.out_dir / "summary.json").exists() and not args.force:
        raise SystemExit(f"{args.out_dir}/summary.json exists; pass --force")
    if args.mode == "plumbing-smoke":
        summary = run_smoke(args, plan)
    else:
        summary = run_fresh_hypothesis(args, plan)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "contract_id": summary["contract_id"],
                "side_effects": summary["side_effects"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
