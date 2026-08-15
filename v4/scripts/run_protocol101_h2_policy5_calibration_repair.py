"""Preregister and run the bounded H2 policy-5 G8 calibration repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_h2_calibration_repair import (
    CalibrationRow,
    METHODS,
    apply_repair,
    select_and_fit_map,
)
from v4.model.protocol101_scoped_stage1_hgb import (
    HGBUnitConfig,
    replay_candidates,
    score_decisions,
    selection_rows,
)
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as base
from v4.scripts.protocol101_training_scope import load_training_scope


AUDIT = Path("v4/audit/autoresearch")
DESIGN_DIR = AUDIT / "protocol101_h2_policy5_calibration_repair_design"
OUT_DIR = AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001"
SMOKE_DIR = AUDIT / "protocol101_h2_policy5_calibration_repair_smoke"
H2_DIR = AUDIT / "protocol101_scoped_canonical_stage1_h2_attempt001"
H2_GATE = AUDIT / "protocol101_scoped_canonical_stage1_h2_attempt001_gates"
H2_AUDIT = AUDIT / "protocol101_scoped_canonical_stage1_h2_attempt001_audit"
SELECTION = (
    AUDIT / "protocol101_scoped_canonical_stage1_cross_hypothesis_selection"
)
ATTRIBUTION = AUDIT / "protocol101_stage1_fixed_exit_binding_attribution_attempt001"
STAGE1_GRAPH = AUDIT / "protocol101_stage1_autoresearch_graph"
POLICY = 5
SEEDS = (42, 43, 44)
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
    parser.add_argument(
        "--mode",
        choices=("preregister", "smoke", "run"),
        required=True,
    )
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke-dir", type=Path, default=SMOKE_DIR)
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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _embedded_hash(payload: dict[str, Any], field: str) -> bool:
    expected = payload.get(field)
    material = dict(payload)
    material.pop(field, None)
    return bool(expected) and expected == stable_hash(material)


def _policy5(gate: dict[str, Any]) -> dict[str, Any]:
    rows = list((gate.get("results") or {}).get("policy_results") or [])
    selected = [row for row in rows if int(row.get("policy_index", -1)) == POLICY]
    if len(selected) != 1:
        raise RuntimeError("H2 gate does not contain exactly one policy-5 row")
    return selected[0]


def starting_checkpoint() -> dict[str, Any]:
    required = {
        "h2_summary": H2_DIR / "summary.json",
        "h2_gate": H2_GATE / "summary.json",
        "h2_audit": H2_AUDIT / "summary.json",
        "selection": SELECTION / "summary.json",
        "attribution": ATTRIBUTION / "summary.json",
        "graph": STAGE1_GRAPH / "state.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"starting checkpoint evidence missing: {missing}")
    payloads = {name: load_json(path) for name, path in required.items()}
    h2_summary = payloads["h2_summary"]
    h2_gate = payloads["h2_gate"]
    h2_audit = payloads["h2_audit"]
    selection = payloads["selection"]
    attribution = payloads["attribution"]
    graph = payloads["graph"]
    blockers: list[str] = []
    if h2_summary.get("status") != (
        "unit_execution_complete_pending_preregistered_gate_aggregation"
    ):
        blockers.append("H2 producing packet is not complete")
    if h2_gate.get("status") != "pass":
        blockers.append("H2 gate packet is not pass")
    if h2_audit.get("status") != "accepted" or h2_audit.get("verdict") != (
        "accepted_real_entry_signal_fixed_exit_or_other_gate_failure"
    ):
        blockers.append("H2 independent audit is not accepted")
    if selection.get("status") != "pass" or selection.get("routing") != (
        "real_signal_attribution_required"
    ):
        blockers.append("cross-hypothesis selection route mismatch")
    if attribution.get("status") != "complete" or attribution.get("decision") != (
        "fixed_exit_composition_requires_separate_design"
    ):
        blockers.append("fixed-exit attribution decision mismatch")
    if graph.get("status") != "complete_pending_owner_review":
        blockers.append("Stage-1 graph is not complete")
    policy = _policy5(h2_gate)
    for gate in (f"G{index}" for index in range(1, 8)):
        if not bool((policy.get("gates") or {}).get(gate)):
            blockers.append(f"H2 policy 5 does not pass {gate}")
    if bool((policy.get("gates") or {}).get("G8")):
        blockers.append("H2 policy 5 unexpectedly passes G8")
    observed = {
        int(row["seed"]): float(row["weighted_oof_ece"])
        for row in policy.get("seed_results") or []
    }
    expected = {
        42: 0.11403412278039929,
        43: 0.10586361094648139,
        44: 0.10780966005499511,
    }
    if observed != expected:
        blockers.append(f"H2 policy-5 starting ECE mismatch: {observed}")
    if blockers:
        raise RuntimeError(f"starting_checkpoint_invalid: {blockers}")
    return {
        "status": "pass",
        "files": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in required.items()
        },
        "original_weighted_ece": {str(key): value for key, value in expected.items()},
        "h2_preregistration_hash": h2_summary["preregistration_hash"],
        "h2_summary_hash": h2_summary["summary_hash"],
        "h2_gate_summary_hash": h2_gate["summary_hash"],
        "h2_audit_freeze_hash": load_json(H2_AUDIT / "freeze.json")["freeze_hash"],
    }


def preregistration() -> dict[str, Any]:
    scope = load_training_scope()
    checkpoint = starting_checkpoint()
    payload = {
        "schema_version": "Protocol101H2Policy5CalibrationRepairDesignV1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "contract_id": base.CONTRACT_ID,
        "hypothesis": "H2",
        "policy_index": POLICY,
        "repair_seeds": list(SEEDS),
        "fresh_g9_seed_reserved": 45,
        "feature_names": list(base.HYPOTHESES["H2"]),
        "fold_governance_hash": scope.fold_governance_hash,
        "acceptance_registry_hash": scope.acceptance_registry_hash,
        "fold_count": len(scope.folds),
        "calibration_only": True,
        "method_menu": {
            "methods": list(METHODS),
            "selection": (
                "per_outer_fold_seed_minimum_observation_weighted_"
                "chronological_inner_10bin_ece"
            ),
            "tie_break": "training_tail_isotonic_v1",
            "platt": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": None,
                "random_state": 0,
            },
            "outer_validation_used_for_selection": False,
        },
        "invariants": [
            "model_hash",
            "epsilon",
            "threshold",
            "selected_slots",
            "actions",
            "entry_intents",
            "trades",
            "pnl",
            "G1_through_G7",
        ],
        "G8": {
            "bins": 10,
            "every_seed_weighted_oof_ece_lte": 0.10,
        },
        "starting_checkpoint": checkpoint,
        "source_hashes": {
            str(path): sha256_path(path) for path in SOURCE_PATHS
        },
        "results_inspected_before_preregistration": False,
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
    payload["preregistration_hash"] = stable_hash(payload)
    return payload


def write_preregistration(design_dir: Path) -> dict[str, Any]:
    path = design_dir / "preregistration.json"
    proposed = preregistration()
    if path.exists():
        existing = load_json(path)
        if not _embedded_hash(existing, "preregistration_hash"):
            raise RuntimeError("existing calibration preregistration self-hash mismatch")
        without_time = lambda row: {
            key: value
            for key, value in row.items()
            if key not in {"registered_at_utc", "preregistration_hash"}
        }
        if without_time(existing) != without_time(proposed):
            raise RuntimeError("existing calibration preregistration contract mismatch")
        return existing
    design_dir.mkdir(parents=True, exist_ok=False)
    write_json(path, proposed)
    return proposed


def run_smoke(smoke_dir: Path, design: dict[str, Any]) -> dict[str, Any]:
    if smoke_dir.exists() and any(smoke_dir.iterdir()):
        raise RuntimeError(f"smoke output is not empty: {smoke_dir}")
    rows = [
        CalibrationRow(
            session=f"synthetic-{session:02d}",
            decision_time=f"2026-01-{session:02d}T10:{minute:02d}:00",
            score=-1.0 + 0.1 * minute,
            outcome=float(minute >= 5),
            selected_contract_id=f"slot-{minute}",
        )
        for session in range(1, 10)
        for minute in range(10)
    ]
    selection = select_and_fit_map(rows)
    payload = {
        "schema_version": "Protocol101H2Policy5CalibrationRepairSmokeV1",
        "status": "pass",
        "scientific_result": False,
        "preregistration_hash": design["preregistration_hash"],
        "selected_method": selection["selected_method"],
        "method_results": selection["method_results"],
        "source_hashes": design["source_hashes"],
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "protected_holdout_read": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
        },
    }
    payload["summary_hash"] = stable_hash(payload)
    write_json(smoke_dir / "summary.json", payload)
    return payload


def _strip_confidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "calibrated_confidence"}
        for row in rows
    ]


def _unit_path(seed: int, fold_id: str) -> Path:
    return H2_DIR / "units" / "policy5" / f"seed{seed}" / fold_id / "summary.json"


def repair_unit(
    *,
    seed: int,
    fold: dict[str, Any],
    path_map: dict[str, Path],
    margins: dict[str, float],
    noise_model: DivergenceNoiseModel,
    out_dir: Path,
) -> dict[str, Any]:
    fold_id = str(fold["fold_id"])
    original_path = _unit_path(seed, fold_id)
    original = load_json(original_path)
    payload = original["unit"]
    config = HGBUnitConfig(**payload["config"])
    if (
        config.hypothesis != "H2"
        or config.policy_index != POLICY
        or config.seed != seed
    ):
        raise RuntimeError(f"original H2 unit config mismatch: {seed}/{fold_id}")
    model_path = Path(original["model_artifact"]["path"])
    if sha256_path(model_path) != original["model_artifact"]["sha256"]:
        raise RuntimeError(f"original model hash mismatch: {model_path}")
    model = pickle.loads(model_path.read_bytes())
    common = {
        "hypothesis": "H2",
        "policy_index": POLICY,
        "guard_margins": margins,
    }
    calibration = base.load_decisions(
        base._session_paths(path_map, list(original["calibration_sessions"])),
        **common,
    )
    validation = base.load_decisions(
        base._session_paths(path_map, list(original["validation_sessions"])),
        **common,
    )
    epsilon = float(payload["calibration"]["epsilon"])
    threshold = float(payload["calibration"]["threshold"])
    calibration_scores = score_decisions(
        model,
        calibration,
        feature_names=tuple(base.HYPOTHESES["H2"]),
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=seed + 100_000,
    )
    validation_scores = score_decisions(
        model,
        validation,
        feature_names=tuple(base.HYPOTHESES["H2"]),
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=seed + 200_000,
    )
    intents, diagnostics = selection_rows(
        validation,
        validation_scores,
        threshold=threshold,
        epsilon=epsilon,
        config=config,
        split="validation",
    )
    trades, _state, metrics = replay_candidates(intents, config=config)
    original_validation = payload["validation"]
    recomputed_intents = [asdict(row) for row in intents]
    recomputed_trades = [asdict(row) for row in trades]
    invariants = {
        "model_hash": {
            "original": original["model_artifact"]["sha256"],
            "recomputed": sha256_path(model_path),
        },
        "epsilon": {
            "original": epsilon,
            "recomputed": epsilon,
        },
        "threshold": {
            "original": threshold,
            "recomputed": threshold,
        },
        "selection_hash": {
            "original": stable_hash(
                _strip_confidence(original_validation["diagnostics"])
            ),
            "recomputed": stable_hash(diagnostics),
        },
        "entry_intents_hash": {
            "original": stable_hash(original_validation["entry_intents"]),
            "recomputed": stable_hash(recomputed_intents),
        },
        "trades_hash": {
            "original": stable_hash(original_validation["trades"]),
            "recomputed": stable_hash(recomputed_trades),
        },
        "metrics_hash": {
            "original": stable_hash(original_validation["metrics"]),
            "recomputed": stable_hash(metrics),
        },
    }
    mismatches = [
        name
        for name, values in invariants.items()
        if values["original"] != values["recomputed"]
    ]
    if mismatches:
        raise RuntimeError(
            f"void_not_calibration_only:{seed}:{fold_id}:{mismatches}"
        )
    repair = apply_repair(
        calibration_decisions=calibration,
        calibration_scores=calibration_scores,
        validation_decisions=validation,
        validation_scores=validation_scores,
        epsilon=epsilon,
        config=config,
    )
    validation_rows = pd.DataFrame(repair.pop("validation_rows"))
    validation_rows["action_enter"] = [
        bool(row["action_enter"]) for row in diagnostics
    ]
    validation_rows["selected_offset"] = [
        float(row["selected_offset"]) for row in diagnostics
    ]
    validation_rows["selected_right"] = [
        str(row["selected_right"]) for row in diagnostics
    ]
    unit_dir = out_dir / "units" / f"seed{seed}" / fold_id
    parquet_path = unit_dir / "validation_confidence.parquet"
    unit_dir.mkdir(parents=True, exist_ok=True)
    validation_rows.to_parquet(parquet_path, index=False)
    result = {
        "schema_version": "Protocol101H2Policy5CalibrationRepairUnitV1",
        "seed": seed,
        "fold": int(fold["fold"]),
        "fold_id": fold_id,
        "original_unit": {
            "path": str(original_path),
            "sha256": sha256_path(original_path),
        },
        "original_model": {
            "path": str(model_path),
            "sha256": sha256_path(model_path),
        },
        "fit_sessions": list(original["fit_sessions"]),
        "calibration_sessions": list(original["calibration_sessions"]),
        "validation_sessions": list(original["validation_sessions"]),
        "original_ece": float(original_validation["expected_calibration_error"]),
        "repaired_ece": float(repair["validation_ece"]),
        "validation_observations": int(repair["validation_observations"]),
        "method_selection": repair["method_selection"],
        "validation_confidence": {
            "path": str(parquet_path),
            "sha256": sha256_path(parquet_path),
            "rows": len(validation_rows),
        },
        "invariants": invariants,
        "invariant_mismatches": mismatches,
    }
    result["unit_hash"] = stable_hash(result)
    summary_path = unit_dir / "summary.json"
    write_json(summary_path, result)
    return {"path": str(summary_path), "sha256": sha256_path(summary_path)}


def run_repair(
    out_dir: Path,
    design: dict[str, Any],
    *,
    design_dir: Path = DESIGN_DIR,
) -> dict[str, Any]:
    terminal_summary = out_dir / "summary.json"
    if terminal_summary.exists():
        existing = load_json(terminal_summary)
        if not _embedded_hash(existing, "summary_hash"):
            raise RuntimeError("existing repair summary self-hash mismatch")
        return existing
    scope = load_training_scope()
    path_map = base._scope_path_map(scope)
    margins = base.guard_margins()
    noise_model = DivergenceNoiseModel.from_parquet(base.NOISE_DISTRIBUTION)
    refs: list[dict[str, Any]] = []
    completed = 0
    for seed in SEEDS:
        for fold in scope.folds:
            fold_id = str(fold["fold_id"])
            existing_path = (
                out_dir / "units" / f"seed{seed}" / fold_id / "summary.json"
            )
            if existing_path.exists():
                existing_unit = load_json(existing_path)
                if not _embedded_hash(existing_unit, "unit_hash"):
                    raise RuntimeError(
                        f"existing repair unit self-hash mismatch: {existing_path}"
                    )
                if existing_unit.get("invariant_mismatches"):
                    raise RuntimeError(
                        f"existing repair unit invariant mismatch: {existing_path}"
                    )
                ref = {
                    "path": str(existing_path),
                    "sha256": sha256_path(existing_path),
                }
            else:
                ref = repair_unit(
                        seed=seed,
                        fold=fold,
                        path_map=path_map,
                        margins=margins,
                        noise_model=noise_model,
                        out_dir=out_dir,
                    )
            refs.append(ref)
            completed += 1
            write_json(
                out_dir / "progress.json",
                {
                    "status": "running",
                    "completed_units": completed,
                    "total_units": 15,
                    "preregistration_hash": design["preregistration_hash"],
                },
            )
    summary = {
        "schema_version": "Protocol101H2Policy5CalibrationRepairRunV1",
        "status": "complete_pending_gate",
        "contract_id": base.CONTRACT_ID,
        "hypothesis": "H2",
        "policy_index": POLICY,
        "seeds": list(SEEDS),
        "unit_count": len(refs),
        "unit_artifacts": refs,
        "preregistration": {
            "path": str(design_dir / "preregistration.json"),
            "sha256": sha256_path(design_dir / "preregistration.json"),
            "preregistration_hash": design["preregistration_hash"],
        },
        "source_hashes": design["source_hashes"],
        "side_effects": {
            "existing_model_rescoring_executed": True,
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
    summary["summary_hash"] = stable_hash(summary)
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "progress.json",
        {"status": "complete", "summary_hash": summary["summary_hash"]},
    )
    return summary


def main() -> int:
    args = parse_args()
    design = write_preregistration(args.design_dir)
    if args.mode == "preregister":
        result = {
            "status": "preregistered",
            "preregistration_hash": design["preregistration_hash"],
            "path": str(args.design_dir / "preregistration.json"),
        }
    elif args.mode == "smoke":
        result = run_smoke(args.smoke_dir, design)
    else:
        result = run_repair(
            args.out_dir,
            design,
            design_dir=args.design_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
