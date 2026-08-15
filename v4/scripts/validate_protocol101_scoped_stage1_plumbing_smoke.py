"""Independently validate the exact-contract Stage-1 plumbing smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.model.protocol101_canonical_stage1_contract import (
    CONTRACT_ID,
    FEATURE_NAMES,
)
from v4.model.protocol101_serial_simulator import (
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
)
from v4.scripts.protocol101_training_scope import load_training_scope


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_SMOKE = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_plumbing_smoke/summary.json"
)
DEFAULT_OUT = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_plumbing_smoke_validation"
)
DEFAULT_READINESS = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_readiness/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-summary", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def validate(
    *,
    smoke: dict[str, Any],
    readiness: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    if readiness.get("status") != "ready_for_plumbing_smoke":
        blockers.append("readiness_not_green")
    if readiness.get("blockers"):
        blockers.append("readiness_has_blockers")
    if smoke.get("status") != "plumbing_smoke_complete_pending_independent_validation":
        blockers.append("unexpected_smoke_status")
    if smoke.get("contract_id") != CONTRACT_ID:
        blockers.append("contract_mismatch")
    if smoke.get("edge_claim_allowed") is not False:
        blockers.append("edge_claim_not_disabled")
    if smoke.get("paper_readiness_claim_allowed") is not False:
        blockers.append("paper_claim_not_disabled")

    expected_summary_hash = smoke.get("summary_hash")
    without_hash = dict(smoke)
    without_hash.pop("summary_hash", None)
    if expected_summary_hash != stable_hash(without_hash):
        blockers.append("summary_hash_mismatch")

    unit = smoke.get("unit") or {}
    if unit.get("feature_names") != list(FEATURE_NAMES):
        blockers.append("smoke_did_not_exercise_all_17_features")
    if unit.get("unexpected_model_features"):
        blockers.append("unexpected_model_features")
    if unit.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_VERSION:
        blockers.append("simulator_version_mismatch")
    fit = unit.get("fit") or {}
    if int(fit.get("fit_candidates_after_cap") or 0) <= 0:
        blockers.append("no_fit_candidates")
    calibration = unit.get("calibration") or {}
    if int(calibration.get("decision_count") or 0) <= 0:
        blockers.append("no_calibration_decisions")
    if not math.isfinite(float(calibration.get("threshold", math.nan))):
        blockers.append("nonfinite_threshold")
    epsilon_summary = calibration.get("epsilon_summary") or {}
    if (
        epsilon_summary.get("method")
        != "training_tail_0x_vs_1x_measured_noise_p95"
    ):
        blockers.append("epsilon_method_mismatch")
    confidence_map = calibration.get("confidence_map") or {}
    if confidence_map.get("method") not in {
        "training_tail_isotonic_payoff_score_to_realized_win",
        "constant_training_tail_base_rate",
    }:
        blockers.append("confidence_calibration_method_mismatch")
    validation = unit.get("validation") or {}
    if int(validation.get("decision_count") or 0) <= 0:
        blockers.append("no_validation_decisions")
    if not math.isfinite(
        float(validation.get("expected_calibration_error", math.nan))
    ):
        blockers.append("nonfinite_validation_ece")
    if float(validation.get("primary_noise_scale", math.nan)) != 1.0:
        blockers.append("validation_primary_noise_scale_not_1x")
    noise_diagnostics = validation.get("noise_diagnostics") or {}
    if set(noise_diagnostics) != {"0.0x", "0.5x", "2.0x"}:
        blockers.append("validation_noise_diagnostics_incomplete")
    fee_sensitivity = validation.get("fee_sensitivity") or {}
    if set(fee_sensitivity) != {"2.60", "3.00", "4.00"}:
        blockers.append("fee_sensitivity_incomplete")
    fill_edge_band = validation.get("fill_edge_band") or {}
    if (
        fill_edge_band.get("gating_rung")
        != "pessimistic_ask_in_bid_out"
    ):
        blockers.append("pessimistic_fill_rung_not_primary")
    if len(validation.get("entry_intents") or []) != int(
        (validation.get("metrics") or {}).get("entry_intents") or 0
    ):
        blockers.append("entry_intent_count_mismatch")
    semantics = validation.get("simulator_semantics") or {}
    if semantics.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_VERSION:
        blockers.append("validation_simulator_version_mismatch")
    if float(semantics.get("affordability_reserve_per_trade") or 0.0) != 3.0:
        blockers.append("fee_reserve_mismatch")
    if (
        float(
            semantics.get("max_daily_loss_fraction_of_session_start_equity") or 0.0
        )
        != 0.05
    ):
        blockers.append("daily_loss_fraction_mismatch")

    sessions = smoke.get("input_sessions") or {}
    fit_sessions = set(sessions.get("fit") or [])
    calibration_sessions = set(sessions.get("calibration") or [])
    validation_sessions = set(sessions.get("validation") or [])
    if fit_sessions & calibration_sessions:
        blockers.append("fit_calibration_overlap")
    if fit_sessions & validation_sessions:
        blockers.append("fit_validation_overlap")
    if calibration_sessions & validation_sessions:
        blockers.append("calibration_validation_overlap")
    if fit_sessions and calibration_sessions and max(fit_sessions) >= min(calibration_sessions):
        blockers.append("calibration_not_after_fit")
    if calibration_sessions and validation_sessions and max(calibration_sessions) >= min(
        validation_sessions
    ):
        blockers.append("validation_not_after_calibration")
    scope = load_training_scope()
    allowed = {session for session, _path in scope.sessions}
    used = fit_sessions | calibration_sessions | validation_sessions
    if not used.issubset(allowed):
        blockers.append("session_outside_governed_scope")
    if any("2025-05-16" <= session <= "2025-06-30" for session in used):
        blockers.append("protected_holdout_session_used")

    model_artifact = smoke.get("model_artifact") or {}
    model_path = Path(str(model_artifact.get("path") or ""))
    if not model_path.exists():
        blockers.append("model_artifact_missing")
    elif sha256_path(model_path) != model_artifact.get("sha256"):
        blockers.append("model_artifact_hash_mismatch")
    for path_text, expected in (smoke.get("code_hashes") or {}).items():
        path = Path(path_text)
        if not path.exists() or sha256_path(path) != expected:
            blockers.append(f"code_hash_mismatch:{path_text}")

    effects = smoke.get("side_effects") or {}
    if effects.get("disposable_plumbing_model_fit_executed") is not True:
        blockers.append("disposable_fit_not_recorded")
    for name, value in effects.items():
        if name == "disposable_plumbing_model_fit_executed":
            continue
        if bool(value):
            blockers.append(f"forbidden_side_effect:{name}")
    details = {
        "used_sessions": sorted(used),
        "feature_count": len(unit.get("feature_names") or []),
        "fit_candidates": int(fit.get("fit_candidates_after_cap") or 0),
        "calibration_decisions": int(calibration.get("decision_count") or 0),
        "validation_decisions": int(validation.get("decision_count") or 0),
        "validation_trades": int(
            ((validation.get("metrics") or {}).get("trades") or 0)
        ),
        "epsilon": calibration.get("epsilon"),
        "threshold": calibration.get("threshold"),
    }
    return sorted(set(blockers)), details


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    smoke = load_json(args.smoke_summary)
    readiness = load_json(args.readiness)
    blockers, details = validate(smoke=smoke, readiness=readiness)
    status = "pass" if not blockers else "blocked"
    summary = {
        "schema_version": "Protocol101ScopedStage1PlumbingSmokeValidationV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "blockers": blockers,
        "smoke_summary": str(args.smoke_summary),
        "smoke_summary_sha256": sha256_path(args.smoke_summary),
        "details": details,
        "highest_allowed_claim": (
            "canonical HGB core plumbing smoke independently validated"
            if not blockers
            else "plumbing smoke validation blocked"
        ),
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
    write_json(args.out_dir / "summary.json", summary)
    if not blockers:
        freeze = {
            "schema_version": "Protocol101ScopedStage1RunnerFreezeV1",
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "status": "core_runner_smoke_passed",
            "contract_id": CONTRACT_ID,
            "smoke_summary_hash": smoke["summary_hash"],
            "preregistration_hash": smoke["preregistration_hash"],
            "fold_governance_hash": smoke["fold_governance_hash"],
            "acceptance_registry_hash": smoke["acceptance_registry_hash"],
            "code_hashes": smoke["code_hashes"],
            "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_VERSION,
            "next_gate": "preregistered_gate_aggregator_and_owner_approved_H0",
        }
        freeze["freeze_hash"] = stable_hash(freeze)
        write_json(args.out_dir / "runner_freeze.json", freeze)
    (args.out_dir / "report.md").write_text(
        "# Protocol101 Scoped Stage-1 Plumbing Smoke Validation\n\n"
        f"- Status: `{status}`\n"
        f"- Blockers: `{blockers}`\n"
        f"- Details: `{details}`\n"
        "- No research model training or threshold selection occurred in validation.\n"
    )
    print(json.dumps({"status": status, "blockers": blockers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
