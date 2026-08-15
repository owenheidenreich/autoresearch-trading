"""Independently audit H2 policy-5 G1-G8 after calibration repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts import run_protocol101_scoped_stage1_independent_audit as prior


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "v4/audit/autoresearch"
REPAIR_DIR = AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001"
GATE_DIR = AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001_gates"
ORIGINAL_H2_DIR = AUDIT / "protocol101_scoped_canonical_stage1_h2_attempt001"
OUT_DIR = AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001_audit"
POLICY = 5
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, default=REPAIR_DIR)
    parser.add_argument("--gate-dir", type=Path, default=GATE_DIR)
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
    return bool(expected) and expected == stable_hash(material)


def independent_ece(confidence: np.ndarray, outcomes: np.ndarray) -> float:
    finite = np.isfinite(confidence) & np.isfinite(outcomes)
    confidence = np.clip(confidence[finite], 0.0, 1.0)
    outcomes = outcomes[finite]
    if not len(confidence):
        return 1.0
    edges = np.linspace(0.0, 1.0, 11)
    result = 0.0
    for index in range(10):
        if index == 9:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            result += (
                float(mask.sum())
                / len(confidence)
                * abs(float(confidence[mask].mean()) - float(outcomes[mask].mean()))
            )
    return float(result)


def reference_inputs() -> tuple[dict[str, Any], float, dict[str, str]]:
    null_summary = load_json(prior.NULL_SUMMARY)
    heuristic = load_json(prior.HEURISTIC_SUMMARY)
    era = load_json(prior.ERA_MANIFEST)
    heuristic_pnl = float(heuristic["g3_fixed_baseline"]["pooled_net_pnl"])
    sessions_to_era: dict[str, str] = {}
    for row in era.get("sessions") or era.get("session_rows") or []:
        session = str(row.get("session") or row.get("session_date") or "")
        label = str(row.get("era") or row.get("era_id") or "")
        if session and label:
            sessions_to_era[session] = label
    if not sessions_to_era:
        mapping = era.get("session_to_era") or {}
        sessions_to_era = {str(key): str(value) for key, value in mapping.items()}
    return null_summary, heuristic_pnl, sessions_to_era


def audit(repair_dir: Path, gate_dir: Path) -> dict[str, Any]:
    repair_summary_path = repair_dir / "summary.json"
    gate_summary_path = gate_dir / "summary.json"
    repair_summary = load_json(repair_summary_path)
    producer_gate = load_json(gate_summary_path)
    defects: list[str] = []
    insufficient: list[str] = []
    if not embedded_hash_matches(repair_summary, "summary_hash"):
        defects.append("repair_summary_self_hash_mismatch")
    if not embedded_hash_matches(producer_gate, "summary_hash"):
        defects.append("producer_gate_self_hash_mismatch")

    repair_units: dict[tuple[int, int], dict[str, Any]] = {}
    ece_by_seed: dict[int, list[tuple[float, int]]] = {seed: [] for seed in SEEDS}
    unit_evidence: list[dict[str, Any]] = []
    for ref in repair_summary.get("unit_artifacts") or []:
        path = Path(str(ref.get("path") or ""))
        if not path.exists():
            insufficient.append(f"repair_unit_missing:{path}")
            continue
        actual_hash = sha256_path(path)
        if actual_hash != str(ref.get("sha256") or ""):
            defects.append(f"repair_unit_hash_mismatch:{path}")
            continue
        unit = load_json(path)
        if not embedded_hash_matches(unit, "unit_hash"):
            defects.append(f"repair_unit_self_hash_mismatch:{path}")
        if unit.get("invariant_mismatches"):
            defects.append(f"repair_invariant_mismatch:{path}")
        seed, fold = int(unit["seed"]), int(unit["fold"])
        confidence_ref = unit.get("validation_confidence") or {}
        confidence_path = Path(str(confidence_ref.get("path") or ""))
        if not confidence_path.exists():
            insufficient.append(f"confidence_rows_missing:{confidence_path}")
            continue
        confidence_hash = sha256_path(confidence_path)
        if confidence_hash != str(confidence_ref.get("sha256") or ""):
            defects.append(f"confidence_rows_hash_mismatch:{confidence_path}")
            continue
        frame = pd.read_parquet(confidence_path)
        required = {
            "calibrated_confidence",
            "outcome",
            "session",
            "decision_time",
            "selected_contract_id",
        }
        if not required.issubset(frame.columns):
            defects.append(f"confidence_rows_schema:{confidence_path}")
            continue
        ece = independent_ece(
            frame["calibrated_confidence"].to_numpy(dtype=float),
            frame["outcome"].to_numpy(dtype=float),
        )
        if not math.isclose(
            ece,
            float(unit["repaired_ece"]),
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            defects.append(f"unit_ece_mismatch:{seed}:{fold}")
        ece_by_seed[seed].append((ece, len(frame)))
        repair_units[(seed, fold)] = unit
        unit_evidence.append(
            {
                "seed": seed,
                "fold": fold,
                "summary_path": str(path),
                "summary_sha256": actual_hash,
                "confidence_path": str(confidence_path),
                "confidence_sha256": confidence_hash,
                "independent_ece": ece,
                "observations": len(frame),
            }
        )

    expected_grid = {(seed, fold) for seed in SEEDS for fold in range(5)}
    if set(repair_units) != expected_grid:
        insufficient.append("repair_seed_fold_grid_incomplete")

    null_summary, heuristic_pnl, sessions_to_era = reference_inputs()
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        original_units = [
            load_json(
                ORIGINAL_H2_DIR
                / "units"
                / "policy5"
                / f"seed{seed}"
                / f"expanding_fold_{fold + 1:02d}"
                / "summary.json"
            )
            for fold in range(5)
        ]
        economic, economic_defects = prior.recompute_seed(
            original_units,
            hypothesis="H2",
            policy_index=POLICY,
            seed=seed,
            null_stats=null_summary["policy_results"][str(POLICY)][
                "pooled_random_net_pnl"
            ],
            heuristic_pnl=heuristic_pnl,
            sessions_to_era=sessions_to_era,
        )
        defects.extend(f"economic_replay:{item}" for item in economic_defects)
        weighted_rows = ece_by_seed[seed]
        observations = sum(count for _ece, count in weighted_rows)
        weighted_ece = (
            sum(ece * count for ece, count in weighted_rows) / observations
            if observations
            else 1.0
        )
        economic["weighted_oof_ece"] = float(weighted_ece)
        economic["ece_observations"] = int(observations)
        economic["gates"]["G8"] = bool(weighted_ece <= 0.10)
        seed_rows.append(economic)

    ordered = sorted(seed_rows, key=lambda row: int(row["seed"]))
    pooled_pnls = [float(row["cv_pooled_net_pnl"]) for row in ordered]
    null_zs = [float(row["null_z"]) for row in ordered]
    gates = {
        "G1": all(bool(row["gates"]["G1"]) for row in ordered),
        "G2": statistics.median(null_zs) >= 3.0,
        "G3": statistics.median(pooled_pnls) > heuristic_pnl,
        "G4": all(bool(row["gates"]["G4"]) for row in ordered),
        "G5": all(bool(row["gates"]["G1"]) for row in ordered)
        and min(null_zs) >= 2.0,
        "G6": all(bool(row["gates"]["G6"]) for row in ordered),
        "G7": all(bool(row["gates"]["G7"]) for row in ordered),
        "G8": all(bool(row["gates"]["G8"]) for row in ordered),
        "G9": False,
    }
    g1_g8_pass = all(gates[f"G{index}"] for index in range(1, 9))
    producer_gates = producer_gate.get("gates") or {}
    for name in [f"G{index}" for index in range(1, 9)]:
        if bool(producer_gates.get(name)) != bool(gates[name]):
            defects.append(f"producer_gate_disagreement:{name}")
    producer_seed = {
        int(row["seed"]): row for row in producer_gate.get("seed_results") or []
    }
    for row in ordered:
        seed = int(row["seed"])
        expected = float(producer_seed.get(seed, {}).get("weighted_oof_ece", math.nan))
        if not math.isclose(
            float(row["weighted_oof_ece"]),
            expected,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            defects.append(f"producer_seed_ece_disagreement:{seed}")

    if insufficient:
        verdict = "blocked_insufficient_evidence"
    elif defects:
        verdict = "void_implementation_or_artifact_defect"
    elif g1_g8_pass:
        verdict = "accepted_eligible_G1_G8"
    else:
        verdict = "accepted_calibration_repair_failed_G8"
    return {
        "schema_version": "Protocol101H2Policy5CalibrationIndependentAuditV1",
        "status": (
            "accepted"
            if verdict.startswith("accepted_")
            else ("blocked" if verdict.startswith("blocked_") else "void")
        ),
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "hypothesis": "H2",
        "policy_index": POLICY,
        "seed_results": ordered,
        "median_seed_cv_pooled_net_pnl": float(statistics.median(pooled_pnls)),
        "median_seed_null_z": float(statistics.median(null_zs)),
        "worst_seed_null_z": float(min(null_zs)),
        "gates": gates,
        "G1_G8_pass": g1_g8_pass,
        "defects": defects,
        "insufficient_evidence": insufficient,
        "unit_evidence": unit_evidence,
        "inputs": {
            "repair_summary": {
                "path": str(repair_summary_path),
                "sha256": sha256_path(repair_summary_path),
            },
            "producer_gate_summary": {
                "path": str(gate_summary_path),
                "sha256": sha256_path(gate_summary_path),
            },
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


def report_text(result: dict[str, Any]) -> str:
    lines = [
        "# Independent H2 Policy-5 Calibration Audit",
        "",
        f"- Verdict: `{result['verdict']}`",
        f"- G1-G8 pass: `{result['G1_G8_pass']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- {name}: `{value}`" for name, value in result["gates"].items()
    )
    lines.extend(["", "## Seed ECE", ""])
    lines.extend(
        f"- Seed {row['seed']}: `{row['weighted_oof_ece']:.12f}`"
        for row in result["seed_results"]
    )
    if result["defects"]:
        lines.extend(["", "## Defects", ""])
        lines.extend(f"- `{item}`" for item in result["defects"])
    if result["insufficient_evidence"]:
        lines.extend(["", "## Insufficient Evidence", ""])
        lines.extend(f"- `{item}`" for item in result["insufficient_evidence"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError(f"audit output is not empty: {args.out_dir}")
    result = audit(args.repair_dir, args.gate_dir)
    result["independent_verification_hash"] = stable_hash(result)
    write_json(args.out_dir / "independent_verification.json", result)
    freeze = {
        "schema_version": "Protocol101H2Policy5CalibrationAuditFreezeV1",
        "verdict": result["verdict"],
        "independent_verification": {
            "path": str(args.out_dir / "independent_verification.json"),
            "sha256": sha256_path(args.out_dir / "independent_verification.json"),
        },
        "inputs": result["inputs"],
    }
    freeze["freeze_hash"] = stable_hash(freeze)
    write_json(args.out_dir / "freeze.json", freeze)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"unit_evidence", "seed_results"}
    }
    summary["freeze_hash"] = freeze["freeze_hash"]
    summary["summary_hash"] = stable_hash(summary)
    write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "report.md").write_text(report_text(result))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
