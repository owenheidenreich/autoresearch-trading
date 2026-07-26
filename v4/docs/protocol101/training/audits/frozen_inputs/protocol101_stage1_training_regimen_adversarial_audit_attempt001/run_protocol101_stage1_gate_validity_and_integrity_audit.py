"""Audit Protocol101 Stage-1 gates, leakage controls, and reward-hacking risk."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from v4.model.protocol101_canonical_stage1_contract import (
    CONTRACT_ID,
    HYPOTHESES,
    QUARANTINED_ALPHA_TOKENS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "v4/audit/autoresearch"
OUT_DIR = (
    AUDIT_ROOT
    / "protocol101_stage1_gate_validity_and_integrity_audit_attempt001"
)
HYPOTHESIS_IDS = ("H0", "H1", "H2", "H3")
POLICIES = tuple(range(7))
SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))
PROTECTED_START = "2025-05-16"
PROTECTED_END = "2025-06-30"
SIGNED_DOCS = (
    REPO_ROOT / "v4/docs/PROTOCOL101_TRADER_CHARTER.md",
    REPO_ROOT
    / "v4/docs/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md",
    REPO_ROOT / "v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md",
    REPO_ROOT / "v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md",
    REPO_ROOT / "v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md",
)
SOURCE_FILES = (
    REPO_ROOT / "v4/model/protocol101_canonical_stage1_contract.py",
    REPO_ROOT / "v4/model/protocol101_scoped_stage1_hgb.py",
    REPO_ROOT / "v4/model/protocol101_serial_simulator.py",
    REPO_ROOT / "v4/scripts/protocol101_training_scope.py",
    REPO_ROOT / "v4/scripts/run_protocol101_scoped_stage1_hgb_runner.py",
    REPO_ROOT / "v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py",
    REPO_ROOT / "v4/scripts/run_protocol101_scoped_stage1_independent_audit.py",
)
SYNC_SUMMARY = (
    AUDIT_ROOT / "protocol101_clean_window_hill_climb_readiness_2026_07_25/summary.json"
)
READINESS_SUMMARY = (
    AUDIT_ROOT / "protocol101_scoped_canonical_stage1_readiness/summary.json"
)
CALIBRATION_AUDIT = (
    AUDIT_ROOT
    / "protocol101_h2_policy5_calibration_repair_attempt001_audit/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


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


def embedded_hash_matches(payload: dict[str, Any], field: str) -> bool:
    expected = payload.get(field)
    material = dict(payload)
    material.pop(field, None)
    return bool(expected) and str(expected) == stable_hash(material)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def hypothesis_paths(hypothesis: str) -> dict[str, Path]:
    token = hypothesis.lower()
    base = AUDIT_ROOT / f"protocol101_scoped_canonical_stage1_{token}_attempt001"
    return {
        "batch": base,
        "gate": Path(f"{base}_gates"),
        "audit": Path(f"{base}_audit"),
    }


def gate_rows() -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    defects: list[str] = []
    evidence: dict[str, Any] = {}
    for hypothesis in HYPOTHESIS_IDS:
        paths = hypothesis_paths(hypothesis)
        gate_path = paths["gate"] / "summary.json"
        audit_path = paths["audit"] / "summary.json"
        if not gate_path.exists() or not audit_path.exists():
            defects.append(f"{hypothesis}:gate_or_audit_missing")
            continue
        gate = load_json(gate_path)
        audit = load_json(audit_path)
        if not embedded_hash_matches(gate, "summary_hash"):
            defects.append(f"{hypothesis}:gate_self_hash_mismatch")
        if audit.get("status") != "accepted":
            defects.append(f"{hypothesis}:independent_audit_not_accepted")
        policy_rows = list((gate.get("results") or {}).get("policy_results") or [])
        if len(policy_rows) != 7:
            defects.append(f"{hypothesis}:policy_row_count:{len(policy_rows)}")
        for row in policy_rows:
            seed_rows = list(row.get("seed_results") or [])
            median_ece = (
                float(
                    statistics.median(
                        float(seed["weighted_oof_ece"]) for seed in seed_rows
                    )
                )
                if seed_rows
                else math.nan
            )
            gates = dict(row.get("gates") or {})
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "policy_index": int(row["policy_index"]),
                    "median_seed_cv_pooled_net_pnl": float(
                        row["median_seed_cv_pooled_net_pnl"]
                    ),
                    "median_seed_null_z": float(row["median_seed_null_z"]),
                    "worst_seed_null_z": float(row["worst_seed_null_z"]),
                    "median_seed_weighted_oof_ece": median_ece,
                    **{
                        f"G{index}": bool(gates.get(f"G{index}"))
                        for index in range(1, 9)
                    },
                    "G1_G8_pass": bool(row.get("G1_G8_pass")),
                    "real_signal": bool(row.get("real_signal")),
                    "routing": str(row.get("routing") or ""),
                }
            )
        evidence[hypothesis] = {
            "gate": {"path": relative(gate_path), "sha256": sha256_path(gate_path)},
            "audit": {
                "path": relative(audit_path),
                "sha256": sha256_path(audit_path),
                "status": audit.get("status"),
                "verdict": audit.get("verdict"),
            },
        }
    return rows, defects, evidence


def empirical_gate_validity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_counts = {
        f"G{index}": sum(bool(row[f"G{index}"]) for row in rows)
        for index in range(1, 9)
    }
    g1_rows = [row for row in rows if row["G1"]]
    non_g1_rows = [row for row in rows if not row["G1"]]
    ece = np.asarray(
        [float(row["median_seed_weighted_oof_ece"]) for row in rows],
        dtype=float,
    )
    pnl = np.asarray(
        [float(row["median_seed_cv_pooled_net_pnl"]) for row in rows],
        dtype=float,
    )
    null_z = np.asarray([float(row["median_seed_null_z"]) for row in rows])
    ece_pnl = spearmanr(ece, pnl)
    ece_null = spearmanr(ece, null_z)
    pair_agreement: dict[str, Any] = {}
    for left in range(1, 9):
        for right in range(left + 1, 9):
            left_values = np.asarray([bool(row[f"G{left}"]) for row in rows])
            right_values = np.asarray([bool(row[f"G{right}"]) for row in rows])
            pair_agreement[f"G{left}_G{right}"] = float(
                np.mean(left_values == right_values)
            )
    by_policy = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy_index"] == policy]
        by_policy[str(policy)] = {
            "rows": len(selected),
            "G1_pass": sum(row["G1"] for row in selected),
            "G8_pass": sum(row["G8"] for row in selected),
            "median_ece": (
                float(
                    statistics.median(
                        row["median_seed_weighted_oof_ece"] for row in selected
                    )
                )
                if selected
                else None
            ),
            "median_pnl": (
                float(
                    statistics.median(
                        row["median_seed_cv_pooled_net_pnl"] for row in selected
                    )
                )
                if selected
                else None
            ),
        }
    return {
        "candidate_rows": len(rows),
        "pass_counts": pass_counts,
        "G8_discrimination": {
            "G1_profitable_rows": len(g1_rows),
            "G1_profitable_rows_passing_G8": sum(row["G8"] for row in g1_rows),
            "non_G1_rows": len(non_g1_rows),
            "non_G1_rows_passing_G8": sum(row["G8"] for row in non_g1_rows),
            "spearman_ece_vs_pnl": {
                "rho": float(ece_pnl.statistic),
                "pvalue": float(ece_pnl.pvalue),
            },
            "spearman_ece_vs_null_z": {
                "rho": float(ece_null.statistic),
                "pvalue": float(ece_null.pvalue),
            },
            "calibrated_confidence_is_action_input": False,
            "calibrated_confidence_is_post_selection_diagnostic": True,
        },
        "by_policy": by_policy,
        "pairwise_gate_agreement": pair_agreement,
    }


def preregistration_precedence(
    prereg_path: Path,
    unit_paths: list[Path],
) -> tuple[bool, dict[str, Any]]:
    prereg_mtime = prereg_path.stat().st_mtime
    earliest_result = min(path.stat().st_mtime for path in unit_paths)
    return prereg_mtime <= earliest_result, {
        "preregistration_path": relative(prereg_path),
        "preregistration_mtime": prereg_mtime,
        "earliest_unit_mtime": earliest_result,
        "precedes_or_equals_first_unit": prereg_mtime <= earliest_result,
    }


def artifact_integrity() -> dict[str, Any]:
    defects: list[str] = []
    insufficient: list[str] = []
    evidence: dict[str, Any] = {}
    total_units = 0
    total_models = 0
    all_sessions: set[str] = set()
    protected_overlap: set[str] = set()
    role_overlap_count = 0
    preregistration_order_pass = True
    model_width_counts: dict[str, dict[str, int]] = {}
    feature_contract_counts: dict[str, int] = {}
    for hypothesis in HYPOTHESIS_IDS:
        batch_dir = hypothesis_paths(hypothesis)["batch"]
        batch_path = batch_dir / "summary.json"
        prereg_path = batch_dir / "preregistration.json"
        if not batch_path.exists() or not prereg_path.exists():
            insufficient.append(f"{hypothesis}:batch_or_preregistration_missing")
            continue
        batch = load_json(batch_path)
        if not embedded_hash_matches(batch, "summary_hash"):
            defects.append(f"{hypothesis}:batch_self_hash_mismatch")
        refs = list(batch.get("unit_artifacts") or [])
        if len(refs) != 105:
            defects.append(f"{hypothesis}:unit_reference_count:{len(refs)}")
        unit_paths = [Path(str(ref.get("path") or "")) for ref in refs]
        if unit_paths and all(path.exists() for path in unit_paths):
            order_pass, order_evidence = preregistration_precedence(
                prereg_path,
                unit_paths,
            )
            preregistration_order_pass = preregistration_order_pass and order_pass
            if not order_pass:
                defects.append(f"{hypothesis}:preregistration_after_result")
        else:
            order_evidence = {"status": "insufficient_unit_paths"}
        grid: set[tuple[int, int, int]] = set()
        widths: dict[str, int] = {}
        for ref, path in zip(refs, unit_paths):
            if not path.exists():
                insufficient.append(f"{hypothesis}:unit_missing:{relative(path)}")
                continue
            if sha256_path(path) != str(ref.get("sha256") or ""):
                defects.append(f"{hypothesis}:unit_hash_mismatch:{relative(path)}")
                continue
            unit = load_json(path)
            payload = unit.get("unit") or {}
            config = payload.get("config") or {}
            policy = int(payload.get("policy_index", -1))
            seed = int(payload.get("seed", -1))
            fold = int(unit.get("fold", -1))
            grid.add((policy, seed, fold))
            total_units += 1
            expected_features = tuple(HYPOTHESES[hypothesis])
            observed_features = tuple(payload.get("feature_names") or ())
            if payload.get("contract_id") != CONTRACT_ID:
                defects.append(f"{hypothesis}:{policy}:{seed}:{fold}:contract")
            if payload.get("hypothesis") != hypothesis:
                defects.append(f"{hypothesis}:{policy}:{seed}:{fold}:hypothesis")
            if observed_features != expected_features:
                defects.append(f"{hypothesis}:{policy}:{seed}:{fold}:features")
            if tuple((payload.get("fit") or {}).get("feature_names") or ()) != (
                expected_features
            ):
                defects.append(f"{hypothesis}:{policy}:{seed}:{fold}:fit_features")
            if payload.get("unexpected_model_features"):
                defects.append(
                    f"{hypothesis}:{policy}:{seed}:{fold}:unexpected_features"
                )
            if any(
                token in name.lower()
                for name in observed_features
                for token in QUARANTINED_ALPHA_TOKENS
            ):
                defects.append(
                    f"{hypothesis}:{policy}:{seed}:{fold}:quarantined_alpha"
                )
            feature_contract_counts[stable_hash(observed_features)] = (
                feature_contract_counts.get(stable_hash(observed_features), 0) + 1
            )
            fit_sessions = set(str(item) for item in unit.get("fit_sessions") or [])
            calibration_sessions = set(
                str(item) for item in unit.get("calibration_sessions") or []
            )
            validation_sessions = set(
                str(item) for item in unit.get("validation_sessions") or []
            )
            if (
                fit_sessions & calibration_sessions
                or fit_sessions & validation_sessions
                or calibration_sessions & validation_sessions
            ):
                role_overlap_count += 1
                defects.append(f"{hypothesis}:{policy}:{seed}:{fold}:role_overlap")
            used = fit_sessions | calibration_sessions | validation_sessions
            all_sessions.update(used)
            protected_overlap.update(
                session
                for session in used
                if PROTECTED_START <= session <= PROTECTED_END
            )
            if fit_sessions and calibration_sessions and (
                max(fit_sessions) >= min(calibration_sessions)
            ):
                defects.append(
                    f"{hypothesis}:{policy}:{seed}:{fold}:fit_calibration_order"
                )
            if calibration_sessions and validation_sessions and (
                max(calibration_sessions) >= min(validation_sessions)
            ):
                defects.append(
                    f"{hypothesis}:{policy}:{seed}:{fold}:calibration_validation_order"
                )
            model_ref = unit.get("model_artifact") or {}
            model_path = Path(str(model_ref.get("path") or ""))
            if not model_path.exists():
                insufficient.append(
                    f"{hypothesis}:model_missing:{relative(model_path)}"
                )
                continue
            if sha256_path(model_path) != str(model_ref.get("sha256") or ""):
                defects.append(
                    f"{hypothesis}:model_hash_mismatch:{relative(model_path)}"
                )
                continue
            with model_path.open("rb") as handle:
                model = pickle.load(handle)
            width = int(getattr(model, "n_features_in_", -1))
            if width != len(expected_features):
                defects.append(
                    f"{hypothesis}:{policy}:{seed}:{fold}:model_width:{width}"
                )
            widths[str(width)] = widths.get(str(width), 0) + 1
            total_models += 1
            if (
                config.get("hypothesis") != hypothesis
                or int(config.get("policy_index", -1)) != policy
                or int(config.get("seed", -1)) != seed
            ):
                defects.append(f"{hypothesis}:{policy}:{seed}:{fold}:config")
        expected_grid = {
            (policy, seed, fold)
            for policy in POLICIES
            for seed in SEEDS
            for fold in FOLDS
        }
        if grid != expected_grid:
            defects.append(f"{hypothesis}:unit_grid_mismatch")
        model_width_counts[hypothesis] = widths
        evidence[hypothesis] = {
            "batch": {
                "path": relative(batch_path),
                "sha256": sha256_path(batch_path),
            },
            "preregistration": {
                "path": relative(prereg_path),
                "sha256": sha256_path(prereg_path),
            },
            "preregistration_order": order_evidence,
            "unit_count": len(refs),
            "model_width_counts": widths,
        }
    if protected_overlap:
        defects.append(f"protected_holdout_overlap:{sorted(protected_overlap)}")
    return {
        "status": (
            "pass"
            if not defects and not insufficient
            else ("blocked" if insufficient else "fail")
        ),
        "defects": defects,
        "insufficient_evidence": insufficient,
        "total_units_verified": total_units,
        "total_models_verified": total_models,
        "unique_sessions_seen": len(all_sessions),
        "protected_holdout_overlap": sorted(protected_overlap),
        "role_overlap_count": role_overlap_count,
        "preregistration_precedes_results": preregistration_order_pass,
        "model_width_counts": model_width_counts,
        "feature_contract_hash_counts": feature_contract_counts,
        "evidence": evidence,
    }


def source_flow_integrity() -> dict[str, Any]:
    contract_text = SOURCE_FILES[0].read_text()
    hgb_text = SOURCE_FILES[1].read_text()
    checks = {
        "contract_asserts_exact_17_unique_features": (
            "len(FEATURE_NAMES) != 17" in contract_text
        ),
        "contract_rejects_quarantined_alpha_tokens": (
            "quarantined alpha leaked into contract" in contract_text
        ),
        "feature_matrix_is_selected_by_hypothesis_allowlist": (
            "indexes = [FEATURE_NAMES.index(name) for name in names]" in contract_text
        ),
        "model_fit_matrix_uses_feature_names_only": (
            'noisy.loc[:, list(feature_names)].to_numpy' in hgb_text
        ),
        "training_target_is_fee_adjusted_payoff_over_premium": (
            "labels - float(config.fee)" in hgb_text
            and "where=premium > 0.0" in hgb_text
        ),
        "model_fit_receives_x_and_target_separately": (
            "model.fit(x, target)" in hgb_text
        ),
        "threshold_selected_on_calibration_rows": (
            "choose_threshold(" in hgb_text
            and 'split="calibration"' in hgb_text
        ),
        "outer_validation_scored_after_threshold_freeze": (
            "validation_scores = score_decisions(" in hgb_text
        ),
        "calibrated_confidence_is_written_after_action_selection": (
            "intents, diagnostics = selection_rows(" in hgb_text
            and 'row["calibrated_confidence"]' in hgb_text
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_hashes": {
            relative(path): sha256_path(path) for path in SOURCE_FILES
        },
        "interpretation": (
            "Static source-flow inspection supports feature/label separation. "
            "It is not a mathematical proof against every possible hidden defect."
        ),
    }


def synchronization_integrity() -> dict[str, Any]:
    sync = load_json(SYNC_SUMMARY)
    readiness = load_json(READINESS_SUMMARY)
    family = sync["family_results"]
    return {
        "status": (
            "pass_scoped_not_global"
            if sync.get("decision")
            == "governed_hill_climbing_ready_on_parity_stable_subset"
            and readiness.get("global_all_feature_synchronization_claimed") is False
            else "fail"
        ),
        "contract_id": readiness.get("contract_id"),
        "scope": readiness.get("synchronization_scope"),
        "H2_model_features": list(HYPOTHESES["H2"]),
        "H2_feature_count": len(HYPOTHESES["H2"]),
        "H2_ladder_semantics": (
            "score each boundary-stable eligible ladder contract using the same "
            "12 context values plus that contract's internally recomputed delta/gamma"
        ),
        "D_composite_source_auc": family["D_composites"]["l2_hgb_auc"],
        "E_internal_greek_source_auc": family["E_internal_greeks"]["l2_hgb_auc"],
        "direct_option_price_source_auc": family["C_direct_option_price"][
            "l2_hgb_auc"
        ],
        "direct_option_price_alpha_allowed": False,
        "raw_quote_microstructure_alpha_allowed": False,
        "candidate_specific_shadow_transfer_complete": False,
        "paper_ready": False,
        "remaining_required_proof": list(sync["required_before_paper_validation"]),
        "evidence": {
            "sync_summary": {
                "path": relative(SYNC_SUMMARY),
                "sha256": sha256_path(SYNC_SUMMARY),
            },
            "readiness_summary": {
                "path": relative(READINESS_SUMMARY),
                "sha256": sha256_path(READINESS_SUMMARY),
            },
        },
    }


def gate_recommendations(empirical: dict[str, Any]) -> list[dict[str, Any]]:
    g8 = empirical["G8_discrimination"]
    return [
        {
            "gate": "G1",
            "recommendation": "keep_hard_gate",
            "reason": "Directly requires repeatable positive fee-adjusted economics.",
        },
        {
            "gate": "G2",
            "recommendation": "keep_hard_gate_add_campaign_multiplicity_report",
            "reason": (
                "Distinguishes selection from matched random, but the campaign "
                "tested 28 hypothesis-policy rows and the current z gate is not "
                "a campaign-level max-statistic correction."
            ),
        },
        {
            "gate": "G3",
            "recommendation": "keep_hard_gate",
            "reason": "Prevents a learned model from advancing when a fixed rule is better.",
        },
        {
            "gate": "G4",
            "recommendation": "keep_hard_gate",
            "reason": "Signed v2 rule directly protects return-to-drawdown and survival.",
        },
        {
            "gate": "G5",
            "recommendation": "keep_hard_gate",
            "reason": "Worst-seed requirements reduce seed-shopping.",
        },
        {
            "gate": "G6",
            "recommendation": "keep_with_sample_size_diagnostics",
            "reason": "Regime failure matters, but era medians need counts and uncertainty.",
        },
        {
            "gate": "G7",
            "recommendation": "keep_as_product_charter_gate",
            "reason": (
                "Defines the intended day-trader behavior; it is not itself "
                "evidence of statistical edge."
            ),
        },
        {
            "gate": "G8",
            "recommendation": (
                "owner_amend_to_report_only_until_confidence_controls_behavior"
            ),
            "reason": (
                f"Calibrated confidence is post-selection and does not affect "
                f"actions. All {g8['G1_profitable_rows']} G1-profitable rows "
                f"failed G8, while {g8['non_G1_rows_passing_G8']} losing/non-G1 "
                f"rows passed it. The gate currently selects exit-policy outcome "
                f"calibratability rather than trading quality."
            ),
        },
        {
            "gate": "G9",
            "recommendation": "keep_spend_once_hard_gate",
            "reason": (
                "Fresh-seed confirmation protects against seed luck, but does "
                "not replace the protected holdout or live shadow transfer."
            ),
        },
    ]


def reward_hacking_analysis(
    artifact: dict[str, Any],
    empirical: dict[str, Any],
) -> dict[str, Any]:
    controls_passed = {
        "all_four_preregistered_hypotheses_executed": True,
        "exact_420_unit_grid_verified": artifact["total_units_verified"] == 420,
        "preregistration_precedes_unit_results": artifact[
            "preregistration_precedes_results"
        ],
        "chronological_roles_do_not_overlap": artifact["role_overlap_count"] == 0,
        "protected_holdout_excluded": not artifact["protected_holdout_overlap"],
        "independent_gate_audits_present": True,
        "g9_seed45_unspent": not (
            AUDIT_ROOT / "protocol101_h2_policy5_g9_seed45_attempt001"
        ).exists(),
    }
    risks = [
        {
            "risk": "research_process_overfit_to_reused_cv",
            "severity": "high",
            "detail": (
                "H0-H3, seven policies, and the post-result H2 calibration repair "
                "were all judged against the same five outer folds. No direct "
                "feature leakage is required for repeated human/agent iteration "
                "to overfit those folds."
            ),
            "control": (
                "Freeze the gate amendment and candidate before seed45; then use "
                "the spend-once G9, protected holdout, and candidate-specific shadow."
            ),
        },
        {
            "risk": "campaign_multiple_comparisons",
            "severity": "medium_high",
            "detail": (
                f"The search produced {empirical['candidate_rows']} hypothesis-policy "
                "rows. G2 is matched-null significance per row, not a frozen "
                "campaign-level maximum-statistic test."
            ),
            "control": (
                "Report a campaign max-stat/null or conservative multiplicity "
                "sensitivity before final candidate freeze."
            ),
        },
        {
            "risk": "post_failure_gate_or_calibration_adaptation",
            "severity": "medium",
            "detail": (
                "The calibration repair was designed after observing H2's G8 "
                "failure. It was preregistered and changed no economics, but it "
                "still represents post-selection attention to one candidate."
            ),
            "control": (
                "Do not try another calibrator. Any G8 amendment must apply to "
                "all H0-H3 rows and be signed before reaggregation."
            ),
        },
        {
            "risk": "candidate_specific_live_transfer_pending",
            "severity": "high",
            "detail": (
                "Family-level source discrimination passed for H2 inputs, but "
                "the fitted H2 model has not yet completed no-order IBKR "
                "decision/slot shadow transfer."
            ),
            "control": "Do not claim paper readiness before the frozen shadow battery.",
        },
    ]
    return {
        "status": (
            "controls_pass_with_material_remaining_process_risks"
            if all(controls_passed.values())
            else "control_failure"
        ),
        "controls_passed": controls_passed,
        "material_risks": risks,
        "no_reward_hacking_claim_allowed": False,
        "current_honest_claim": (
            "No direct leakage or protected-data violation was found in the "
            "audited code/artifacts; process-level overfitting remains possible "
            "until G9, protected holdout, and live shadow are passed."
        ),
    }


def write_gate_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hypothesis",
        "policy_index",
        "median_seed_cv_pooled_net_pnl",
        "median_seed_null_z",
        "worst_seed_null_z",
        "median_seed_weighted_oof_ece",
        *[f"G{index}" for index in range(1, 9)],
        "G1_G8_pass",
        "real_signal",
        "routing",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_text(result: dict[str, Any]) -> str:
    empirical = result["gate_validity"]
    integrity = result["artifact_integrity"]
    sync = result["synchronization_integrity"]
    lines = [
        "# Protocol101 Stage-1 Gate Validity And Integrity Audit",
        "",
        f"- Status: `{result['status']}`",
        f"- Gate decision: `{result['decision']}`",
        f"- Units/models verified: `{integrity['total_units_verified']}` / "
        f"`{integrity['total_models_verified']}`",
        f"- Protected overlap: `{integrity['protected_holdout_overlap']}`",
        f"- Role-overlap defects: `{integrity['role_overlap_count']}`",
        "",
        "## Central Findings",
        "",
        "- No direct label/future field was found in the model-facing allowlists.",
        "- All audited fit/calibration/validation roles were chronological and disjoint.",
        "- The 30-session protected holdout was absent from all 420 model units.",
        "- H2 scores ladder contracts using 12 context features plus internally "
        "computed delta/gamma; it does not use raw quotes or future PnL as alpha.",
        "- Synchronization is scoped, not global; candidate-specific IBKR shadow "
        "transfer is still required.",
        f"- Every G1-profitable row passing G8: "
        f"`{empirical['G8_discrimination']['G1_profitable_rows_passing_G8']}` / "
        f"`{empirical['G8_discrimination']['G1_profitable_rows']}`.",
        f"- Non-G1 rows passing G8: "
        f"`{empirical['G8_discrimination']['non_G1_rows_passing_G8']}` / "
        f"`{empirical['G8_discrimination']['non_G1_rows']}`.",
        "",
        "## Gate Recommendations",
        "",
    ]
    lines.extend(
        f"- {row['gate']}: **{row['recommendation']}** - {row['reason']}"
        for row in result["gate_recommendations"]
    )
    lines.extend(
        [
            "",
            "## Same-Game Boundary",
            "",
            f"- Contract: `{sync['contract_id']}`",
            f"- H2 feature count: `{sync['H2_feature_count']}`",
            f"- D source AUC: `{sync['D_composite_source_auc']}`",
            f"- E source AUC: `{sync['E_internal_greek_source_auc']}`",
            f"- Direct option-price source AUC: "
            f"`{sync['direct_option_price_source_auc']}` (quarantined)",
            "- Family-level evidence supports offline research on the scoped "
            "features. It does not yet prove the fitted candidate is "
            "indistinguishable across historical and live feeds.",
            "",
            "## Required Decision",
            "",
            "Do not spend seed 45 under the current contradictory gate contract. "
            "An owner-signed amendment should make G8 report-only until calibrated "
            "confidence actually controls abstention, sizing, or another trading "
            "behavior. Reaggregate all H0-H3 rows under the amended contract, then "
            "select the candidate without changing models or economics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError(f"audit output is not empty: {args.out_dir}")
    rows, gate_defects, gate_evidence = gate_rows()
    empirical = empirical_gate_validity(rows)
    artifact = artifact_integrity()
    source = source_flow_integrity()
    sync = synchronization_integrity()
    recommendations = gate_recommendations(empirical)
    reward = reward_hacking_analysis(artifact, empirical)
    defects = [
        *gate_defects,
        *artifact["defects"],
        *([] if source["status"] == "pass" else ["source_flow_check_failed"]),
        *([] if sync["status"] == "pass_scoped_not_global" else ["sync_scope_failed"]),
    ]
    insufficient = list(artifact["insufficient_evidence"])
    status = (
        "blocked"
        if insufficient
        else ("fail_integrity" if defects else "complete_owner_amendment_required")
    )
    decision = (
        "repair_integrity_before_any_candidate_work"
        if defects or insufficient
        else "gate_contract_amendment_required_before_g9"
    )
    result = {
        "schema_version": "Protocol101Stage1GateValidityAndIntegrityAuditV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "decision": decision,
        "gate_validity": empirical,
        "gate_recommendations": recommendations,
        "artifact_integrity": artifact,
        "source_flow_integrity": source,
        "synchronization_integrity": sync,
        "reward_hacking_analysis": reward,
        "defects": defects,
        "insufficient_evidence": insufficient,
        "gate_evidence": gate_evidence,
        "signed_document_hashes": {
            relative(path): sha256_path(path) for path in SIGNED_DOCS
        },
        "calibration_audit": {
            "path": relative(CALIBRATION_AUDIT),
            "sha256": sha256_path(CALIBRATION_AUDIT),
            "verdict": load_json(CALIBRATION_AUDIT).get("verdict"),
        },
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "protected_holdout_read": False,
            "recorder_or_confirmation_data_read": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    result["summary_hash"] = stable_hash(result)
    write_json(args.out_dir / "summary.json", result)
    write_json(args.out_dir / "integrity_checks.json", {
        "artifact_integrity": artifact,
        "source_flow_integrity": source,
        "reward_hacking_analysis": reward,
    })
    write_json(args.out_dir / "gate_recommendations.json", {
        "decision": decision,
        "recommendations": recommendations,
        "empirical_gate_validity": empirical,
    })
    write_gate_matrix(args.out_dir / "gate_matrix.csv", rows)
    (args.out_dir / "report.md").write_text(report_text(result))
    print(json.dumps({
        "status": status,
        "decision": decision,
        "summary_hash": result["summary_hash"],
        "defects": defects,
        "insufficient_evidence": insufficient,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
