"""Verify the exact scoped Protocol101 Stage-1 contract without training.

This preflight is intentionally fail-closed. It verifies the governed folds,
the 17-feature adapter, synchronization evidence, guard/noise artifacts, owner
signatures, and exact-contract null/baseline artifacts. It never trains a
model, selects a threshold, reads protected/recorder data, or touches runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from v4.model.protocol101_canonical_stage1_contract import (
    CONTRACT_ID,
    FEATURE_NAMES,
    HYPOTHESES,
    boundary_stable_mask,
    hypothesis_matrix,
)
from v4.model.protocol101_serial_simulator import (
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
)
from v4.scripts.protocol101_training_scope import load_training_scope


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_scoped_canonical_stage1_readiness"
READINESS_SUMMARY = (
    BASE_AUDIT
    / "protocol101_clean_window_hill_climb_readiness_2026_07_25/summary.json"
)
INTERSECTION_SUMMARY = (
    BASE_AUDIT / "protocol101_canonical_v1_intersection_guard_audit/summary.json"
)
NOISE_SUMMARY = (
    BASE_AUDIT / "protocol101_canonical_v1_divergence_noise_calibration/summary.json"
)
NOISE_DISTRIBUTION = (
    BASE_AUDIT
    / "protocol101_canonical_v1_l0_l2_design_audit_attempt001/"
    "divergence_distributions.parquet"
)
SELECTION_CONTRACT = (
    BASE_AUDIT
    / "protocol101_canonical_v1_4_near_atm_band_restriction_attempt005/"
    "selection_contract_v1_4.json"
)
EXACT_NULL = BASE_AUDIT / "protocol101_scoped_canonical_stage1_null_canary/summary.json"
EXACT_HEURISTIC = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_heuristic_baseline/summary.json"
)
SCOPED_DECISION_DOC = Path(
    "v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md"
)
CHARTER_DOC = Path(
    "v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md"
)
TRAINING_DESIGN_DOC = Path(
    "v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md"
)
GATES_DOC = Path(
    "v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md"
)
G4_DOC = Path(
    "v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_document(path: Path) -> bool:
    text = path.read_text()
    signature_lines = [
        line.strip() for line in text.splitlines() if "signature:" in line.lower()
    ]
    return any(
        line
        and "________________" not in line
        and not line.lower().endswith("signature:")
        for line in signature_lines
    )


def feature_smoke(session_path: Path) -> dict[str, Any]:
    rows = pickle.load(session_path.open("rb"))
    margins = load_json(INTERSECTION_SUMMARY)["vendor_only_training_guard_policy"][
        "use_boundary_stable_margins"
    ]
    tested_rows = 0
    eligible_slots = 0
    finite_by_hypothesis = {name: 0 for name in HYPOTHESES}
    expected_widths = {name: len(features) for name, features in HYPOTHESES.items()}
    for row in rows[:30]:
        tested_rows += 1
        mask = boundary_stable_mask(row, margins)
        eligible_slots += int(mask.sum())
        for hypothesis in HYPOTHESES:
            matrix = hypothesis_matrix(row, hypothesis)
            if matrix.shape[:2] != mask.shape or matrix.shape[2] != expected_widths[hypothesis]:
                raise ValueError(
                    f"{hypothesis} feature shape {matrix.shape} incompatible with {mask.shape}"
                )
            finite_by_hypothesis[hypothesis] += int(
                np.isfinite(matrix[mask]).all(axis=1).sum()
            )
    return {
        "session_path": str(session_path),
        "rows_tested": tested_rows,
        "boundary_stable_eligible_slots": eligible_slots,
        "expected_widths": expected_widths,
        "fully_finite_eligible_rows_by_hypothesis": finite_by_hypothesis,
    }


def reference_packet_blockers(
    *,
    scope: Any,
    null_summary: dict[str, Any],
    heuristic_summary: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    for name, payload in (
        ("null_canary", null_summary),
        ("heuristic_baseline", heuristic_summary),
    ):
        if payload.get("status") != "pass":
            blockers.append(f"exact_contract_{name}_status_not_pass")
        if payload.get("contract_id") != CONTRACT_ID:
            blockers.append(f"exact_contract_{name}_contract_mismatch")
        if payload.get("fold_governance_hash") != scope.fold_governance_hash:
            blockers.append(f"exact_contract_{name}_fold_hash_mismatch")
        if payload.get("acceptance_registry_hash") != scope.acceptance_registry_hash:
            blockers.append(f"exact_contract_{name}_registry_hash_mismatch")
        if int(payload.get("fold_eligible_sessions") or 0) != len(scope.sessions):
            blockers.append(f"exact_contract_{name}_session_count_mismatch")
        effects = payload.get("side_effects") or {}
        if any(bool(value) for value in effects.values()):
            blockers.append(f"exact_contract_{name}_forbidden_side_effect")

    if null_summary.get("preregistration_hash") != heuristic_summary.get(
        "preregistration_hash"
    ):
        blockers.append("exact_contract_reference_preregistration_hash_mismatch")

    g3 = heuristic_summary.get("g3_fixed_baseline") or {}
    folds = g3.get("folds") or []
    if len(folds) != 5:
        blockers.append("exact_contract_heuristic_not_five_folds")
    for fold in folds:
        semantics = fold.get("simulator_semantics") or {}
        if semantics.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_VERSION:
            blockers.append("exact_contract_heuristic_simulator_version_mismatch")
        if float(semantics.get("affordability_reserve_per_trade") or 0.0) != 3.0:
            blockers.append("exact_contract_heuristic_fee_reserve_mismatch")
        if float(fold.get("min_equity", 0.0)) < 0.0:
            blockers.append("exact_contract_heuristic_negative_equity")
    return blockers


def build_summary() -> dict[str, Any]:
    blockers: list[str] = []
    required = [
        READINESS_SUMMARY,
        INTERSECTION_SUMMARY,
        NOISE_SUMMARY,
        NOISE_DISTRIBUTION,
        SELECTION_CONTRACT,
        SCOPED_DECISION_DOC,
        CHARTER_DOC,
        TRAINING_DESIGN_DOC,
        GATES_DOC,
        G4_DOC,
    ]
    missing = [str(path) for path in required if not path.exists()]
    blockers.extend(f"missing_required_artifact:{path}" for path in missing)
    if missing:
        return {
            "schema_version": "Protocol101ScopedCanonicalStage1ReadinessV1",
            "status": "blocked",
            "blockers": blockers,
            "side_effects": side_effects(),
        }

    readiness = load_json(READINESS_SUMMARY)
    if readiness.get("decision") != "governed_hill_climbing_ready_on_parity_stable_subset":
        blockers.append("scoped_synchronization_evidence_not_passed")
    if tuple(readiness.get("initial_model_alpha_features") or ()) != FEATURE_NAMES:
        blockers.append("readiness_feature_list_does_not_match_exact_contract")

    intersection = load_json(INTERSECTION_SUMMARY)
    if intersection.get("status") != "pass":
        blockers.append("intersection_guard_audit_not_passed")
    noise = load_json(NOISE_SUMMARY)
    if noise.get("status") != "pass":
        blockers.append("divergence_noise_calibration_not_passed")

    scope = load_training_scope()
    registry = load_json(scope.acceptance_registry_path)
    if int(registry.get("session_count") or 0) != 301:
        blockers.append("training_scope_registry_not_301_sessions")
    if len(scope.sessions) != 271:
        blockers.append("fold_eligible_session_union_not_271")
    expected_folds = [(45, 45), (90, 45), (135, 45), (180, 45), (225, 45)]
    observed_folds = [
        (len(fold["train_sessions"]), len(fold["validation_sessions"]))
        for fold in scope.folds
    ]
    if observed_folds != expected_folds:
        blockers.append(f"unexpected_fold_geometry:{observed_folds}")
    protected = {
        session
        for session, _path in scope.sessions
        if "2025-05-16" <= session <= "2025-06-30"
    }
    if protected:
        blockers.append(f"protected_holdout_present:{sorted(protected)}")

    signatures = {
        "scoped_synchronization_decision": signed_document(SCOPED_DECISION_DOC),
        "trader_charter": signed_document(CHARTER_DOC),
        "training_design": signed_document(TRAINING_DESIGN_DOC),
        "g4_revision": signed_document(G4_DOC),
    }
    for name in ("scoped_synchronization_decision", "trader_charter", "training_design"):
        if not signatures[name]:
            blockers.append(f"owner_signature_missing:{name}")
    if not signatures["g4_revision"]:
        blockers.append("signed_g4_revision_not_detected")

    exact_contract_artifacts = {
        "null_canary": EXACT_NULL.exists(),
        "heuristic_baseline": EXACT_HEURISTIC.exists(),
    }
    for name, present in exact_contract_artifacts.items():
        if not present:
            blockers.append(f"exact_contract_{name}_missing")
    if all(exact_contract_artifacts.values()):
        blockers.extend(
            reference_packet_blockers(
                scope=scope,
                null_summary=load_json(EXACT_NULL),
                heuristic_summary=load_json(EXACT_HEURISTIC),
            )
        )

    smoke = feature_smoke(scope.sessions[0][1])
    if smoke["boundary_stable_eligible_slots"] <= 0:
        blockers.append("feature_smoke_has_no_boundary_stable_candidates")
    if smoke["fully_finite_eligible_rows_by_hypothesis"]["H0"] <= 0:
        blockers.append("feature_smoke_H0_has_no_finite_candidates")

    hashes = {
        str(path): sha256_path(path)
        for path in required
        if path.exists()
    }
    return {
        "schema_version": "Protocol101ScopedCanonicalStage1ReadinessV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "ready_for_plumbing_smoke" if not blockers else "blocked",
        "contract_id": CONTRACT_ID,
        "feature_names": list(FEATURE_NAMES),
        "hypotheses": {name: list(features) for name, features in HYPOTHESES.items()},
        "synchronization_scope": "17_feature_offline_stage1_only",
        "global_all_feature_synchronization_claimed": False,
        "training_scope": {
            "registry_sessions": int(registry.get("session_count") or 0),
            "fold_eligible_unique_sessions": len(scope.sessions),
            "protected_holdout_sessions_present": sorted(protected),
            "fold_geometry": observed_folds,
            "fold_governance_hash": scope.fold_governance_hash,
            "acceptance_registry_hash": scope.acceptance_registry_hash,
        },
        "signatures": signatures,
        "exact_contract_artifacts": exact_contract_artifacts,
        "feature_smoke": smoke,
        "artifact_hashes": hashes,
        "blockers": sorted(set(blockers)),
        "side_effects": side_effects(),
    }


def side_effects() -> dict[str, bool]:
    return {
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
    }


def write_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Scoped Canonical Stage-1 Readiness",
        "",
        f"- Status: `{summary['status']}`",
        f"- Contract: `{summary.get('contract_id', CONTRACT_ID)}`",
        "- This is a no-training preflight.",
        "",
        "## Blockers",
        "",
    ]
    blockers = summary.get("blockers") or []
    lines.extend(f"- `{blocker}`" for blocker in blockers)
    if not blockers:
        lines.append("- None.")
    if "training_scope" in summary:
        scope = summary["training_scope"]
        lines += [
            "",
            "## Governed Scope",
            "",
            f"- Registry sessions: `{scope['registry_sessions']}`",
            f"- Fold-eligible sessions after protected holdout exclusion: `{scope['fold_eligible_unique_sessions']}`",
            f"- Fold geometry: `{scope['fold_geometry']}`",
            f"- Protected sessions present: `{scope['protected_holdout_sessions_present']}`",
            "",
            "## Exact Feature Contract",
            "",
            f"- Features: `{len(summary['feature_names'])}`",
            f"- Hypotheses: `{list(summary['hypotheses'])}`",
            f"- Smoke: `{summary['feature_smoke']}`",
        ]
    lines += [
        "",
        "No model training, threshold selection, holdout read, recorder read, broker call, paper-submit, or runtime mutation occurred.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    (args.out_dir / "report.md").write_text(write_report(summary))
    print(json.dumps({"status": summary["status"], "blockers": summary["blockers"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
