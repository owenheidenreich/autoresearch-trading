"""Aggregate the preregistered H2 policy-5 calibration repair through G1-G8."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


AUDIT = Path("v4/audit/autoresearch")
REPAIR_DIR = AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001"
ORIGINAL_GATE_DIR = (
    AUDIT / "protocol101_scoped_canonical_stage1_h2_attempt001_gates"
)
OUT_DIR = (
    AUDIT / "protocol101_h2_policy5_calibration_repair_attempt001_gates"
)
POLICY = 5
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, default=REPAIR_DIR)
    parser.add_argument("--original-gate-dir", type=Path, default=ORIGINAL_GATE_DIR)
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


def original_policy5(gate_summary: dict[str, Any]) -> dict[str, Any]:
    rows = list((gate_summary.get("results") or {}).get("policy_results") or [])
    matches = [row for row in rows if int(row.get("policy_index", -1)) == POLICY]
    if len(matches) != 1:
        raise RuntimeError("original H2 gate does not contain one policy-5 row")
    return matches[0]


def aggregate(
    repair_dir: Path,
    original_gate_dir: Path,
) -> dict[str, Any]:
    repair_summary_path = repair_dir / "summary.json"
    gate_summary_path = original_gate_dir / "summary.json"
    repair_summary = load_json(repair_summary_path)
    gate_summary = load_json(gate_summary_path)
    blockers: list[str] = []
    if not embedded_hash_matches(repair_summary, "summary_hash"):
        blockers.append("repair_summary_self_hash_mismatch")
    if not embedded_hash_matches(gate_summary, "summary_hash"):
        blockers.append("original_gate_summary_self_hash_mismatch")
    if repair_summary.get("status") != "complete_pending_gate":
        blockers.append("repair_run_not_complete")
    refs = list(repair_summary.get("unit_artifacts") or [])
    if len(refs) != 15:
        blockers.append(f"repair_unit_reference_count:{len(refs)}")

    units: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    fallback_states: Counter[str] = Counter()
    for ref in refs:
        path = Path(str(ref.get("path") or ""))
        if not path.exists():
            blockers.append(f"unit_missing:{path}")
            continue
        if sha256_path(path) != str(ref.get("sha256") or ""):
            blockers.append(f"unit_hash_mismatch:{path}")
            continue
        unit = load_json(path)
        if not embedded_hash_matches(unit, "unit_hash"):
            blockers.append(f"unit_self_hash_mismatch:{path}")
        if unit.get("invariant_mismatches"):
            blockers.append(f"calibration_only_invariant_mismatch:{path}")
        if int(unit.get("seed", -1)) not in SEEDS:
            blockers.append(f"unexpected_seed:{path}")
        if int(unit.get("fold", -1)) not in range(5):
            blockers.append(f"unexpected_fold:{path}")
        method = str(
            (unit.get("method_selection") or {}).get("selected_method") or ""
        )
        methods[method] += 1
        state_method = str(
            ((unit.get("method_selection") or {}).get("final_state") or {}).get(
                "method"
            )
            or ""
        )
        if state_method.startswith("constant_"):
            fallback_states[state_method] += 1
        confidence_ref = unit.get("validation_confidence") or {}
        confidence_path = Path(str(confidence_ref.get("path") or ""))
        if not confidence_path.exists():
            blockers.append(f"validation_confidence_missing:{confidence_path}")
        elif sha256_path(confidence_path) != str(confidence_ref.get("sha256") or ""):
            blockers.append(f"validation_confidence_hash_mismatch:{confidence_path}")
        units.append(unit)

    grid = sorted((int(row["seed"]), int(row["fold"])) for row in units)
    expected_grid = [(seed, fold) for seed in SEEDS for fold in range(5)]
    if grid != expected_grid:
        blockers.append("repair_seed_fold_grid_mismatch")

    original = original_policy5(gate_summary)
    original_seed_rows = {
        int(row["seed"]): row for row in original.get("seed_results") or []
    }
    repaired_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_units = [row for row in units if int(row["seed"]) == seed]
        observations = sum(int(row["validation_observations"]) for row in seed_units)
        weighted_ece = (
            sum(
                float(row["repaired_ece"]) * int(row["validation_observations"])
                for row in seed_units
            )
            / observations
            if observations
            else 1.0
        )
        original_seed = dict(original_seed_rows[seed])
        gates = dict(original_seed.get("gates") or {})
        gates["G8"] = bool(weighted_ece <= 0.10)
        original_seed["weighted_oof_ece"] = float(weighted_ece)
        original_seed["ece_observations"] = int(observations)
        original_seed["gates"] = gates
        repaired_seed_rows.append(original_seed)

    gates = {
        f"G{index}": bool((original.get("gates") or {}).get(f"G{index}"))
        for index in range(1, 8)
    }
    gates["G8"] = all(
        bool((row.get("gates") or {}).get("G8")) for row in repaired_seed_rows
    )
    gates["G9"] = False
    g1_g8_pass = all(gates[f"G{index}"] for index in range(1, 9))
    verdict = (
        "eligible_pending_independent_audit"
        if g1_g8_pass and not blockers
        else (
            "accepted_calibration_repair_failed_G8"
            if not gates["G8"] and not blockers
            else "void_not_calibration_only"
        )
    )
    return {
        "schema_version": "Protocol101H2Policy5CalibrationRepairGateV1",
        "status": "pass" if not blockers else "void",
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "hypothesis": "H2",
        "policy_index": POLICY,
        "seed_results": repaired_seed_rows,
        "median_seed_cv_pooled_net_pnl": original[
            "median_seed_cv_pooled_net_pnl"
        ],
        "median_seed_null_z": original["median_seed_null_z"],
        "worst_seed_null_z": original["worst_seed_null_z"],
        "gates": gates,
        "G1_G8_pass": g1_g8_pass,
        "verdict": verdict,
        "method_counts": dict(sorted(methods.items())),
        "fallback_state_counts": dict(sorted(fallback_states.items())),
        "blockers": blockers,
        "inputs": {
            "repair_summary": {
                "path": str(repair_summary_path),
                "sha256": sha256_path(repair_summary_path),
            },
            "original_gate_summary": {
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
        "# H2 Policy-5 Calibration Repair Gate",
        "",
        f"- Status: `{result['status']}`",
        f"- Verdict: `{result['verdict']}`",
        f"- G1-G8 pass: `{result['G1_G8_pass']}`",
        f"- Selected methods: `{result['method_counts']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- {name}: `{value}`" for name, value in result["gates"].items()
    )
    lines.extend(["", "## Repaired ECE", ""])
    lines.extend(
        f"- Seed {row['seed']}: `{row['weighted_oof_ece']:.12f}`"
        for row in result["seed_results"]
    )
    if result["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in result["blockers"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError(f"gate output is not empty: {args.out_dir}")
    result = aggregate(args.repair_dir, args.original_gate_dir)
    result["summary_hash"] = stable_hash(result)
    write_json(args.out_dir / "summary.json", result)
    write_json(args.out_dir / "gate_results.json", result)
    (args.out_dir / "report.md").write_text(report_text(result))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
