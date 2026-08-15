"""Finalize no-training readiness for the scoped Protocol101 H0 runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.model.protocol101_canonical_stage1_contract import CONTRACT_ID
from v4.model.protocol101_serial_simulator import (
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
)
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as runner


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_OUT = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_machinery_readiness"
)
PREFLIGHT = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_readiness/summary.json"
)
SMOKE = (
    BASE_AUDIT
    / "protocol101_scoped_canonical_stage1_plumbing_smoke/summary.json"
)
VALIDATION = (
    BASE_AUDIT
    / "protocol101_scoped_canonical_stage1_plumbing_smoke_validation/summary.json"
)
FREEZE = (
    BASE_AUDIT
    / "protocol101_scoped_canonical_stage1_plumbing_smoke_validation/"
    "runner_freeze.json"
)
NULL = (
    BASE_AUDIT / "protocol101_scoped_canonical_stage1_null_canary/summary.json"
)
HEURISTIC = (
    BASE_AUDIT
    / "protocol101_scoped_canonical_stage1_heuristic_baseline/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def runner_args(*, mode: str) -> Namespace:
    return Namespace(
        mode=mode,
        out_dir=Path("/tmp/protocol101-machinery-readiness-no-execution"),
        readiness=PREFLIGHT,
        hypothesis="H0",
        owner_approved_plumbing_smoke=False,
        owner_approved_offline_training=False,
        force=False,
        smoke_rows_per_session=20,
    )


def build_summary() -> dict[str, Any]:
    required = (PREFLIGHT, SMOKE, VALIDATION, FREEZE, NULL, HEURISTIC)
    blockers = [
        f"required_artifact_missing:{path}"
        for path in required
        if not path.exists()
    ]
    if blockers:
        return {
            "schema_version": "Protocol101ScopedStage1MachineryReadinessV1",
            "status": "blocked",
            "blockers": blockers,
            "side_effects": side_effects(),
        }

    preflight = load_json(PREFLIGHT)
    smoke = load_json(SMOKE)
    validation = load_json(VALIDATION)
    freeze = load_json(FREEZE)
    null = load_json(NULL)
    heuristic = load_json(HEURISTIC)
    if preflight.get("status") != "ready_for_plumbing_smoke":
        blockers.append("exact_contract_preflight_not_passed")
    if preflight.get("blockers"):
        blockers.append("exact_contract_preflight_has_blockers")
    if not all((preflight.get("signatures") or {}).values()):
        blockers.append("owner_signatures_not_all_detected")
    if validation.get("status") != "pass" or validation.get("blockers"):
        blockers.append("independent_smoke_validation_not_passed")
    if freeze.get("status") != "core_runner_smoke_passed":
        blockers.append("runner_freeze_not_passed")
    if freeze.get("contract_id") != CONTRACT_ID:
        blockers.append("runner_freeze_contract_mismatch")
    if freeze.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_VERSION:
        blockers.append("runner_freeze_simulator_mismatch")
    if freeze.get("smoke_summary_hash") != smoke.get("summary_hash"):
        blockers.append("runner_freeze_smoke_hash_mismatch")
    for path_text, expected in (freeze.get("code_hashes") or {}).items():
        path = Path(str(path_text))
        if not path.exists() or sha256_path(path) != expected:
            blockers.append(f"frozen_code_hash_mismatch:{path_text}")
    if str(runner.GATE_AGGREGATOR_PATH) not in (
        freeze.get("code_hashes") or {}
    ):
        blockers.append("gate_aggregator_absent_from_runner_freeze")
    for name, payload in (("null", null), ("heuristic", heuristic)):
        if payload.get("status") != "pass":
            blockers.append(f"{name}_reference_not_passed")
        if payload.get("contract_id") != CONTRACT_ID:
            blockers.append(f"{name}_reference_contract_mismatch")

    dry_plan = runner.runner_plan(runner_args(mode="dry-run"))
    locked_plan = runner.runner_plan(runner_args(mode="train-hypothesis"))
    if dry_plan.get("status") != "ready" or dry_plan.get("blockers"):
        blockers.append("runner_dry_plan_not_ready")
    if locked_plan.get("status") != "blocked":
        blockers.append("H0_runner_not_owner_locked")
    if locked_plan.get("blockers") != [
        "owner_approval_missing:offline_training"
    ]:
        blockers.append("H0_runner_has_unexpected_pretraining_blockers")

    return {
        "schema_version": "Protocol101ScopedStage1MachineryReadinessV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "ready_for_separate_owner_approved_H0"
            if not blockers
            else "blocked"
        ),
        "blockers": sorted(set(blockers)),
        "contract_id": CONTRACT_ID,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_VERSION,
        "preflight_sha256": sha256_path(PREFLIGHT),
        "smoke_summary_sha256": sha256_path(SMOKE),
        "smoke_validation_sha256": sha256_path(VALIDATION),
        "runner_freeze_sha256": sha256_path(FREEZE),
        "runner_freeze_hash": freeze.get("freeze_hash"),
        "dry_run_status": dry_plan.get("status"),
        "H0_without_owner_approval": {
            "status": locked_plan.get("status"),
            "blockers": locked_plan.get("blockers"),
        },
        "research_model_training_executed": False,
        "next_owner_action": "explicitly authorize the preregistered H0 offline batch",
        "highest_allowed_claim": (
            "scoped Stage-1 machinery is ready for separately authorized H0"
            if not blockers
            else "Stage-1 machinery readiness blocked"
        ),
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


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary["summary_hash"] = stable_hash(summary)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(
        "# Protocol101 Scoped Stage-1 Machinery Readiness\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Blockers: `{summary['blockers']}`\n"
        f"- Next owner action: `{summary.get('next_owner_action')}`\n"
        "- No research model training or threshold selection occurred.\n"
    )
    print(
        json.dumps(
            {"status": summary["status"], "blockers": summary["blockers"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
