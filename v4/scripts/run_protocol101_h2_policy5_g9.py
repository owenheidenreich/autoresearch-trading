"""Preregister, smoke, and run the spend-once H2 policy-5 seed-45 G9 batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_h2_calibration_repair import (
    CalibrationRow,
    apply_repair,
    select_and_fit_map,
)
from v4.model.protocol101_scoped_stage1_hgb import (
    HGBUnitConfig,
    score_decisions,
)
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as base
from v4.scripts.protocol101_training_scope import load_training_scope


AUDIT = Path("v4/audit/autoresearch")
CAL_AUDIT = (
    AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001_audit"
)
DESIGN_DIR = AUDIT / "protocol101_h2_policy5_g9_seed45_design"
SMOKE_DIR = AUDIT / "protocol101_h2_policy5_g9_seed45_smoke"
OUT_DIR = AUDIT / "protocol101_h2_policy5_g9_seed45_attempt001"
STAGE1_DIRS = tuple(
    AUDIT / f"protocol101_scoped_canonical_stage1_h{name}_attempt001"
    for name in range(4)
)
SEED = 45
POLICY = 5
SOURCE_PATHS = (
    Path(__file__),
    Path("v4/model/protocol101_h2_calibration_repair.py"),
    Path("v4/model/protocol101_scoped_stage1_hgb.py"),
    Path("v4/model/protocol101_canonical_stage1_contract.py"),
    Path("v4/model/protocol101_serial_simulator.py"),
    Path("v4/scripts/run_protocol101_scoped_stage1_hgb_runner.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "smoke", "run"), required=True)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--smoke-dir", type=Path, default=SMOKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--owner-approved-offline-training", action="store_true")
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
    return bool(expected) and expected == stable_hash(material)


def seed45_exclusion_evidence() -> dict[str, Any]:
    matches: list[str] = []
    scanned: list[dict[str, Any]] = []
    for directory in STAGE1_DIRS:
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Stage-1 batch missing: {summary_path}")
        summary = load_json(summary_path)
        refs = list(summary.get("unit_artifacts") or [])
        for ref in refs:
            text = str(ref.get("path") or "")
            if "seed45" in text:
                matches.append(text)
        scanned.append(
            {
                "path": str(summary_path),
                "sha256": sha256_path(summary_path),
                "unit_references": len(refs),
            }
        )
    if matches:
        raise RuntimeError(f"seed45_already_used_in_stage1:{matches}")
    return {
        "status": "pass",
        "seed": SEED,
        "matches": matches,
        "scanned_batches": scanned,
    }


def accepted_calibration_freeze() -> dict[str, Any]:
    summary_path = CAL_AUDIT / "summary.json"
    freeze_path = CAL_AUDIT / "freeze.json"
    if not summary_path.exists() or not freeze_path.exists():
        raise RuntimeError("accepted calibration audit evidence is missing")
    summary = load_json(summary_path)
    freeze = load_json(freeze_path)
    if summary.get("status") != "accepted" or summary.get("verdict") != (
        "accepted_eligible_G1_G8"
    ):
        raise RuntimeError(
            f"calibration_not_eligible_for_g9:{summary.get('verdict')}"
        )
    if not embedded_hash_matches(summary, "summary_hash"):
        raise RuntimeError("calibration audit summary self-hash mismatch")
    if not embedded_hash_matches(freeze, "freeze_hash"):
        raise RuntimeError("calibration audit freeze self-hash mismatch")
    return {
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_path(summary_path),
            "summary_hash": summary["summary_hash"],
        },
        "freeze": {
            "path": str(freeze_path),
            "sha256": sha256_path(freeze_path),
            "freeze_hash": freeze["freeze_hash"],
        },
    }


def preregistration() -> dict[str, Any]:
    scope = load_training_scope()
    payload = {
        "schema_version": "Protocol101H2Policy5G9Seed45DesignV1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "contract_id": base.CONTRACT_ID,
        "hypothesis": "H2",
        "feature_names": list(base.HYPOTHESES["H2"]),
        "policy_index": POLICY,
        "seed": SEED,
        "seed_spend_rule": "one_nonvoid_scientific_result_only_no_replacement_seed",
        "fold_count": len(scope.folds),
        "fold_governance_hash": scope.fold_governance_hash,
        "acceptance_registry_hash": scope.acceptance_registry_hash,
        "calibration_rule": {
            "methods": [
                "training_tail_isotonic_v1",
                "training_tail_platt_v1",
            ],
            "selection": (
                "per_outer_fold_seed_minimum_observation_weighted_"
                "chronological_inner_10bin_ece"
            ),
            "tie_break": "training_tail_isotonic_v1",
        },
        "exact_g9_gates": ["G1", "G2_z3", "G4"],
        "g8_seed45_diagnostic_only": True,
        "seed_exclusion": seed45_exclusion_evidence(),
        "accepted_calibration_audit": accepted_calibration_freeze(),
        "source_hashes": {
            str(path): sha256_path(path) for path in SOURCE_PATHS
        },
        "results_inspected_before_preregistration": False,
        "side_effects": {
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
    payload["preregistration_hash"] = stable_hash(payload)
    return payload


def write_preregistration(design_dir: Path) -> dict[str, Any]:
    path = design_dir / "preregistration.json"
    proposed = preregistration()
    if path.exists():
        existing = load_json(path)
        if not embedded_hash_matches(existing, "preregistration_hash"):
            raise RuntimeError("existing G9 preregistration self-hash mismatch")
        comparable = lambda row: {
            key: value
            for key, value in row.items()
            if key not in {"registered_at_utc", "preregistration_hash"}
        }
        if comparable(existing) != comparable(proposed):
            raise RuntimeError("existing G9 preregistration contract mismatch")
        return existing
    design_dir.mkdir(parents=True, exist_ok=False)
    write_json(path, proposed)
    return proposed


def run_smoke(smoke_dir: Path, design: dict[str, Any]) -> dict[str, Any]:
    if smoke_dir.exists() and any(smoke_dir.iterdir()):
        existing_path = smoke_dir / "summary.json"
        if existing_path.exists():
            return load_json(existing_path)
        raise RuntimeError(f"G9 smoke output is not empty: {smoke_dir}")
    rows = [
        CalibrationRow(
            session=f"synthetic-{session:02d}",
            decision_time=f"2026-01-{session:02d}T10:{minute:02d}:00",
            score=float(minute - 4) / 4.0,
            outcome=float((minute + session) % 3 == 0),
            selected_contract_id=f"slot-{minute}",
        )
        for session in range(1, 10)
        for minute in range(10)
    ]
    selection = select_and_fit_map(rows)
    result = {
        "schema_version": "Protocol101H2Policy5G9Seed45SmokeV1",
        "status": "pass",
        "scientific_result": False,
        "seed45_spent": False,
        "fold_count": 5,
        "selected_method": selection["selected_method"],
        "seed_exclusion": design["seed_exclusion"],
        "preregistration_hash": design["preregistration_hash"],
        "source_hashes": design["source_hashes"],
    }
    result["summary_hash"] = stable_hash(result)
    write_json(smoke_dir / "summary.json", result)
    return result


def apply_accepted_calibration(
    *,
    model: Any,
    calibration: list[Any],
    validation: list[Any],
    unit: dict[str, Any],
    config: HGBUnitConfig,
    noise_model: DivergenceNoiseModel,
) -> dict[str, Any]:
    epsilon = float(unit["calibration"]["epsilon"])
    calibration_scores = score_decisions(
        model,
        calibration,
        feature_names=tuple(base.HYPOTHESES["H2"]),
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=SEED + 100_000,
    )
    validation_scores = score_decisions(
        model,
        validation,
        feature_names=tuple(base.HYPOTHESES["H2"]),
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=SEED + 200_000,
    )
    repair = apply_repair(
        calibration_decisions=calibration,
        calibration_scores=calibration_scores,
        validation_decisions=validation,
        validation_scores=validation_scores,
        epsilon=epsilon,
        config=config,
    )
    rows = repair.pop("validation_rows")
    diagnostics = unit["validation"]["diagnostics"]
    if len(rows) != len(diagnostics):
        raise RuntimeError("G9 calibration diagnostic row count mismatch")
    for row, diagnostic in zip(rows, diagnostics):
        if (
            str(row["session"]) != str(diagnostic["session"])
            or str(row["decision_time"]) != str(diagnostic["decision_time"])
            or str(row["selected_contract_id"])
            != str(diagnostic["selected_contract_id"])
        ):
            raise RuntimeError("G9 calibration diagnostic identity mismatch")
        diagnostic["calibrated_confidence"] = float(row["calibrated_confidence"])
    unit["calibration"]["confidence_map"] = repair["method_selection"]["final_state"]
    unit["calibration"]["confidence_method_selection"] = repair["method_selection"]
    unit["validation"]["expected_calibration_error"] = float(
        repair["validation_ece"]
    )
    unit["validation"]["calibration_observations"] = int(
        repair["validation_observations"]
    )
    return repair


def run_seed45(
    out_dir: Path,
    design: dict[str, Any],
    *,
    design_dir: Path,
) -> dict[str, Any]:
    terminal_path = out_dir / "summary.json"
    if terminal_path.exists():
        existing = load_json(terminal_path)
        if not embedded_hash_matches(existing, "summary_hash"):
            raise RuntimeError("existing G9 summary self-hash mismatch")
        return existing
    for path, expected in design["source_hashes"].items():
        if sha256_path(Path(path)) != expected:
            raise RuntimeError(f"G9 source changed after preregistration: {path}")
    scope = load_training_scope()
    path_map = base._scope_path_map(scope)
    margins = base.guard_margins()
    noise_model = DivergenceNoiseModel.from_parquet(base.NOISE_DISTRIBUTION)
    refs: list[dict[str, Any]] = []
    for fold in scope.folds:
        fold_id = str(fold["fold_id"])
        unit_dir = out_dir / "units" / f"seed{SEED}" / fold_id
        summary_path = unit_dir / "summary.json"
        if summary_path.exists():
            refs.append(
                {"path": str(summary_path), "sha256": sha256_path(summary_path)}
            )
            continue
        fit_sessions, calibration_sessions = base.split_fit_calibration_sessions(
            list(fold["train_sessions"])
        )
        validation_sessions = list(fold["validation_sessions"])
        common = {
            "hypothesis": "H2",
            "policy_index": POLICY,
            "guard_margins": margins,
        }
        fit = base.load_decisions(
            base._session_paths(path_map, fit_sessions),
            **common,
        )
        calibration = base.load_decisions(
            base._session_paths(path_map, calibration_sessions),
            **common,
        )
        validation = base.load_decisions(
            base._session_paths(path_map, validation_sessions),
            **common,
        )
        config = HGBUnitConfig(hypothesis="H2", policy_index=POLICY, seed=SEED)
        model, unit = base.run_hgb_unit(
            fit_decisions=fit,
            calibration_decisions=calibration,
            validation_decisions=validation,
            noise_model=noise_model,
            config=config,
        )
        calibration_evidence = apply_accepted_calibration(
            model=model,
            calibration=calibration,
            validation=validation,
            unit=unit,
            config=config,
            noise_model=noise_model,
        )
        unit_dir.mkdir(parents=True, exist_ok=True)
        model_path = unit_dir / "model.pkl"
        model_path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        unit_summary = {
            "fold": int(fold["fold"]),
            "fold_id": fold_id,
            "fit_sessions": fit_sessions,
            "calibration_sessions": calibration_sessions,
            "validation_sessions": validation_sessions,
            "unit": unit,
            "calibration_repair": calibration_evidence,
            "model_artifact": {
                "path": str(model_path),
                "sha256": sha256_path(model_path),
            },
            "preregistration_hash": design["preregistration_hash"],
        }
        unit_summary["unit_hash"] = stable_hash(unit_summary)
        write_json(summary_path, unit_summary)
        refs.append({"path": str(summary_path), "sha256": sha256_path(summary_path)})
        write_json(
            out_dir / "progress.json",
            {
                "status": "running_seed45_spend_once",
                "completed_folds": len(refs),
                "total_folds": 5,
                "seed": SEED,
            },
        )
    result = {
        "schema_version": "Protocol101H2Policy5G9Seed45RunV1",
        "status": "unit_execution_complete_pending_independent_G9_audit",
        "seed45_spent": True,
        "contract_id": base.CONTRACT_ID,
        "hypothesis": "H2",
        "policy_index": POLICY,
        "seed": SEED,
        "unit_count": len(refs),
        "unit_artifacts": refs,
        "preregistration": {
            "path": str(design_dir / "preregistration.json"),
            "sha256": sha256_path(design_dir / "preregistration.json"),
            "preregistration_hash": design["preregistration_hash"],
        },
        "source_hashes": design["source_hashes"],
        "economic_verdict_emitted": False,
        "side_effects": {
            "research_model_training_executed": True,
            "threshold_selection_executed": True,
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
    write_json(out_dir / "summary.json", result)
    write_json(out_dir / "progress.json", {"status": "complete", "summary_hash": result["summary_hash"]})
    return result


def main() -> int:
    args = parse_args()
    design = write_preregistration(args.design_dir)
    if args.mode == "preregister":
        result = {
            "status": "preregistered",
            "preregistration_hash": design["preregistration_hash"],
        }
    elif args.mode == "smoke":
        result = run_smoke(args.smoke_dir, design)
    else:
        if not args.owner_approved_offline_training:
            raise RuntimeError("owner approval missing for seed-45 offline training")
        result = run_seed45(
            args.out_dir,
            design,
            design_dir=args.design_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
