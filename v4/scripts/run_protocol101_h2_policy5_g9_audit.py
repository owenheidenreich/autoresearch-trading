"""Independently audit the spend-once H2 policy-5 seed-45 G9 result."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.scripts import run_protocol101_scoped_stage1_independent_audit as prior


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "v4/audit/autoresearch"
G9_DIR = AUDIT / "protocol101_h2_policy5_g9_seed45_attempt001"
OUT_DIR = AUDIT / "protocol101_h2_policy5_g9_seed45_attempt001_audit"
POLICY = 5
SEED = 45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g9-dir", type=Path, default=G9_DIR)
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


def audit(g9_dir: Path) -> dict[str, Any]:
    summary_path = g9_dir / "summary.json"
    if not summary_path.exists():
        return {
            "verdict": "g9_blocked_insufficient_evidence",
            "defects": [],
            "insufficient_evidence": ["g9_summary_missing"],
        }
    summary = load_json(summary_path)
    defects: list[str] = []
    insufficient: list[str] = []
    if not embedded_hash_matches(summary, "summary_hash"):
        defects.append("g9_summary_self_hash_mismatch")
    if summary.get("status") != (
        "unit_execution_complete_pending_independent_G9_audit"
    ):
        defects.append("g9_producer_status_mismatch")
    refs = list(summary.get("unit_artifacts") or [])
    if len(refs) != 5:
        insufficient.append(f"g9_unit_reference_count:{len(refs)}")
    units: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    protected_overlap: set[str] = set()
    for ref in refs:
        path = Path(str(ref.get("path") or ""))
        if not path.exists():
            insufficient.append(f"g9_unit_missing:{path}")
            continue
        unit_hash = sha256_path(path)
        if unit_hash != str(ref.get("sha256") or ""):
            defects.append(f"g9_unit_hash_mismatch:{path}")
            continue
        unit = load_json(path)
        if not embedded_hash_matches(unit, "unit_hash"):
            defects.append(f"g9_unit_self_hash_mismatch:{path}")
        payload = unit.get("unit") or {}
        config = payload.get("config") or {}
        if (
            payload.get("hypothesis") != "H2"
            or int(payload.get("policy_index", -1)) != POLICY
            or int(payload.get("seed", -1)) != SEED
            or int(config.get("seed", -1)) != SEED
        ):
            defects.append(f"g9_unit_contract_mismatch:{path}")
        model_ref = unit.get("model_artifact") or {}
        model_path = Path(str(model_ref.get("path") or ""))
        if not model_path.exists():
            insufficient.append(f"g9_model_missing:{model_path}")
        elif sha256_path(model_path) != str(model_ref.get("sha256") or ""):
            defects.append(f"g9_model_hash_mismatch:{model_path}")
        sessions = (
            list(unit.get("fit_sessions") or [])
            + list(unit.get("calibration_sessions") or [])
            + list(unit.get("validation_sessions") or [])
        )
        protected_overlap.update(
            session
            for session in sessions
            if "2025-05-16" <= str(session) <= "2025-06-30"
        )
        units.append(unit)
        evidence.append(
            {
                "path": str(path),
                "sha256": unit_hash,
                "model_path": str(model_path),
                "model_sha256": (
                    sha256_path(model_path) if model_path.exists() else None
                ),
            }
        )
    if protected_overlap:
        defects.append(f"protected_holdout_overlap:{sorted(protected_overlap)}")
    if sorted(int(unit["fold"]) for unit in units) != list(range(5)):
        insufficient.append("g9_fold_grid_not_exact_0_4")

    seed_result: dict[str, Any] | None = None
    if not defects and not insufficient:
        null = load_json(prior.NULL_SUMMARY)
        heuristic = load_json(prior.HEURISTIC_SUMMARY)
        era = load_json(prior.ERA_MANIFEST)
        sessions_to_era = {
            str(item["session"]): str(item["era"])
            for item in era.get("sessions") or []
        }
        seed_result, replay_defects = prior.recompute_seed(
            units,
            hypothesis="H2",
            policy_index=POLICY,
            seed=SEED,
            null_stats=null["policy_results"][str(POLICY)][
                "pooled_random_net_pnl"
            ],
            heuristic_pnl=float(
                heuristic["g3_fixed_baseline"]["pooled_net_pnl"]
            ),
            sessions_to_era=sessions_to_era,
        )
        defects.extend(f"independent_replay:{item}" for item in replay_defects)
    signed_gates = (
        {
            "G1": bool(seed_result["gates"]["G1"]),
            "G2": bool(seed_result["gates"]["G2_z3"]),
            "G4": bool(seed_result["gates"]["G4"]),
        }
        if seed_result is not None
        else {"G1": False, "G2": False, "G4": False}
    )
    if insufficient:
        verdict = "g9_blocked_insufficient_evidence"
    elif defects:
        verdict = "g9_void_implementation_or_artifact_defect"
    elif all(signed_gates.values()):
        verdict = "g9_pass_candidate_eligible_for_final_fit"
    else:
        verdict = "g9_fail_candidate_burned"
    return {
        "schema_version": "Protocol101H2Policy5G9IndependentAuditV1",
        "status": (
            "pass"
            if verdict.startswith("g9_pass_")
            else (
                "fail"
                if verdict.startswith("g9_fail_")
                else ("blocked" if "blocked" in verdict else "void")
            )
        ),
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "hypothesis": "H2",
        "policy_index": POLICY,
        "seed": SEED,
        "signed_g9_gates": signed_gates,
        "seed_result": seed_result,
        "g8_diagnostic": (
            bool(seed_result["gates"]["G8"]) if seed_result is not None else None
        ),
        "defects": defects,
        "insufficient_evidence": insufficient,
        "protected_holdout_overlap": sorted(protected_overlap),
        "unit_evidence": evidence,
        "input": {
            "path": str(summary_path),
            "sha256": sha256_path(summary_path),
        },
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


def report_text(result: dict[str, Any]) -> str:
    lines = [
        "# Independent H2 Policy-5 Seed-45 G9 Audit",
        "",
        f"- Verdict: `{result['verdict']}`",
        f"- Signed gates: `{result['signed_g9_gates']}`",
        f"- G8 diagnostic: `{result['g8_diagnostic']}`",
    ]
    seed = result.get("seed_result")
    if seed:
        lines.extend(
            [
                f"- CV pooled net PnL: `{seed['cv_pooled_net_pnl']}`",
                f"- Null z-score: `{seed['null_z']}`",
                f"- Pooled Calmar: `{seed['pooled_calmar']}`",
                f"- Weighted ECE: `{seed['weighted_oof_ece']}`",
            ]
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
        existing = args.out_dir / "summary.json"
        if existing.exists():
            print(existing.read_text())
            return 0
        raise RuntimeError(f"G9 audit output is not empty: {args.out_dir}")
    result = audit(args.g9_dir)
    result["independent_verification_hash"] = stable_hash(result)
    write_json(args.out_dir / "independent_verification.json", result)
    freeze = {
        "schema_version": "Protocol101H2Policy5G9AuditFreezeV1",
        "verdict": result["verdict"],
        "input": result["input"],
        "independent_verification": {
            "path": str(args.out_dir / "independent_verification.json"),
            "sha256": sha256_path(args.out_dir / "independent_verification.json"),
        },
    }
    freeze["freeze_hash"] = stable_hash(freeze)
    write_json(args.out_dir / "freeze.json", freeze)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"unit_evidence"}
    }
    summary["freeze_hash"] = freeze["freeze_hash"]
    summary["summary_hash"] = stable_hash(summary)
    write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "report.md").write_text(report_text(result))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
