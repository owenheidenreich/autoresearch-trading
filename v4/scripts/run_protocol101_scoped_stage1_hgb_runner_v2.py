"""Durable wrapper for fresh Protocol101 simulator-v5 entry units."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_scoped_stage1_hgb import HGBUnitConfig
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as base


RUNNER_PATH = Path(__file__)
ORIGINAL_RUNNER_PATH = Path(base.__file__)


def _without_registration_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("registered_at_utc", None)
    result.pop("preregistration_hash", None)
    return result


def load_or_create_preregistration(
    args: Any,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    path = args.out_dir / "preregistration.json"
    proposed = base.fresh_preregistration(args, plan=plan)
    if not path.exists():
        base.write_json(path, proposed)
        return proposed
    existing = base.load_json(path)
    expected_hash = existing.get("preregistration_hash")
    without_hash = dict(existing)
    without_hash.pop("preregistration_hash", None)
    if not expected_hash or expected_hash != base.stable_hash(without_hash):
        raise RuntimeError("existing preregistration self-hash mismatch")
    if _without_registration_identity(existing) != _without_registration_identity(
        proposed
    ):
        raise RuntimeError(
            "existing preregistration differs from the current frozen run contract"
        )
    return existing


def verify_resumable_unit(
    summary_path: Path,
    *,
    hypothesis: str,
    policy: int,
    seed: int,
    fold: dict[str, Any],
    fit_sessions: list[str],
    calibration_sessions: list[str],
    validation_sessions: list[str],
    preregistration_payload: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = base.load_json(summary_path)
    preregistration_payload = preregistration_payload or {}
    provenance = provenance or {}
    if summary.get("campaign_namespace") != base.FRESH_CAMPAIGN_NAMESPACE:
        raise RuntimeError(
            f"old or foreign campaign unit is not resumable: {summary_path}"
        )
    if (
        base.FRESH_CAMPAIGN_NAMESPACE
        not in Path(summary_path).resolve().parts
    ):
        raise RuntimeError(
            f"unit is outside the fresh campaign namespace: {summary_path}"
        )
    if (
        summary.get("campaign_contract_sha256")
        != provenance.get("campaign_contract_sha256")
        or summary.get("campaign_preregistration_sha256")
        != provenance.get("campaign_preregistration_sha256")
        or summary.get("runner_preregistration_sha256")
        != preregistration_payload.get("preregistration_hash")
    ):
        raise RuntimeError(
            f"completed unit campaign provenance mismatch: {summary_path}"
        )
    unit = summary.get("unit") or {}
    config = unit.get("config") or {}
    expected = {
        "fold": int(fold["fold"]),
        "fold_id": str(fold["fold_id"]),
        "hypothesis": hypothesis,
        "policy": int(policy),
        "seed": int(seed),
    }
    observed = {
        "fold": int(summary.get("fold", -1)),
        "fold_id": str(summary.get("fold_id") or ""),
        "hypothesis": str(unit.get("hypothesis") or ""),
        "policy": int(unit.get("policy_index", -1)),
        "seed": int(unit.get("seed", -1)),
    }
    if observed != expected:
        raise RuntimeError(
            f"completed unit identity mismatch at {summary_path}: "
            f"{observed} != {expected}"
        )
    if (
        list(summary.get("fit_sessions") or []) != fit_sessions
        or list(summary.get("calibration_sessions") or []) != calibration_sessions
        or list(summary.get("validation_sessions") or []) != validation_sessions
    ):
        raise RuntimeError(f"completed unit session membership mismatch: {summary_path}")
    if list(unit.get("feature_names") or []) != list(base.HYPOTHESES[hypothesis]):
        raise RuntimeError(f"completed unit feature mismatch: {summary_path}")
    if (
        config.get("hypothesis") != hypothesis
        or int(config.get("policy_index", -1)) != int(policy)
        or int(config.get("seed", -1)) != int(seed)
    ):
        raise RuntimeError(f"completed unit config mismatch: {summary_path}")
    model = summary.get("model_artifact") or {}
    model_path = Path(str(model.get("path") or ""))
    if not model_path.exists():
        raise RuntimeError(f"completed unit model missing: {model_path}")
    if base.sha256_path(model_path) != model.get("sha256"):
        raise RuntimeError(f"completed unit model hash mismatch: {model_path}")
    if unit.get("simulator_version") != (
        base.PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
    ):
        raise RuntimeError(
            f"completed unit is not simulator-v5-only: {summary_path}"
        )
    replay = summary.get("replay_packet") or {}
    packet_dir = Path(str(replay.get("path") or ""))
    manifest = base.verify_replay_packet(packet_dir)
    if manifest.get("simulator_version") != (
        base.PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
    ):
        raise RuntimeError(
            f"completed replay packet is not v5-only: {packet_dir}"
        )
    if base.sha256_path(packet_dir / "manifest.json") != replay.get(
        "manifest_sha256"
    ):
        raise RuntimeError(
            f"completed replay manifest hash mismatch: {packet_dir}"
        )
    if manifest.get("model_or_equivalence_certificate_hash") != model.get(
        "sha256"
    ):
        raise RuntimeError(
            f"replay packet model hash mismatch: {packet_dir}"
        )
    if (
        manifest.get("campaign_namespace")
        != base.FRESH_CAMPAIGN_NAMESPACE
        or manifest.get("campaign_contract_sha256")
        != provenance.get("campaign_contract_sha256")
        or manifest.get("campaign_preregistration_sha256")
        != provenance.get("campaign_preregistration_sha256")
        or manifest.get("runner_preregistration_sha256")
        != preregistration_payload.get("preregistration_hash")
    ):
        raise RuntimeError(
            f"replay packet campaign provenance mismatch: {packet_dir}"
        )
    observed_provenance = summary.get("provenance") or {}
    for key in (
        "campaign_contract_sha256",
        "campaign_preregistration_sha256",
        "fold_governance_sha256",
        "acceptance_registry_sha256",
        "feature_contract_source_sha256",
        "simulator_source_sha256",
        "runner_core_source_sha256",
        "runner_source_sha256",
        "scientific_runner_source_sha256",
    ):
        if observed_provenance.get(key) != provenance.get(key):
            raise RuntimeError(
                f"completed unit source provenance mismatch for {key}: "
                f"{summary_path}"
            )
    expected_summary_hash = summary.get("summary_hash")
    without_hash = dict(summary)
    without_hash.pop("summary_hash", None)
    if expected_summary_hash != base.stable_hash(without_hash):
        raise RuntimeError(
            f"completed unit summary self-hash mismatch: {summary_path}"
        )
    return summary


def run_fresh_hypothesis(args: Any, plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("identity_receipt", {}).get("status") != "PASS":
        raise RuntimeError("fresh identity preflight did not pass")
    if plan.get("input_receipt", {}).get("status") != "PASS":
        raise RuntimeError("fresh frozen-input verification did not pass")
    scope = base.load_training_scope()
    prereg = load_or_create_preregistration(args, plan=plan)
    path_map = base._scope_path_map(scope)
    margins = base.guard_margins()
    noise_model = DivergenceNoiseModel.from_parquet(base.NOISE_DISTRIBUTION)
    unit_refs: list[dict[str, Any]] = []
    total = len(scope.folds) * len(base.POLICIES) * len(base.SEEDS)
    completed = 0
    for policy in base.POLICIES:
        for seed in base.SEEDS:
            for fold in scope.folds:
                fold_id = str(fold["fold_id"])
                train_sessions = list(fold["train_sessions"])
                fit_sessions, calibration_sessions = base.split_fit_calibration_sessions(
                    train_sessions
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
                    verify_resumable_unit(
                        summary_path,
                        hypothesis=args.hypothesis,
                        policy=policy,
                        seed=seed,
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
                            "sha256": base.sha256_path(summary_path),
                        }
                    )
                    completed += 1
                    base.write_json(
                        args.out_dir / "progress.json",
                        {
                            "status": "resuming_owner_approved_hypothesis",
                            "completed_units": completed,
                            "total_units": total,
                            "preregistration_hash": prereg["preregistration_hash"],
                        },
                    )
                    continue
                common = {
                    "hypothesis": args.hypothesis,
                    "policy_index": policy,
                    "guard_margins": margins,
                }
                fit = base.load_repaired_decisions(
                    base._session_paths(path_map, fit_sessions),
                    split=f"{fold_id}:fit",
                    **common,
                )
                calibration = base.load_repaired_decisions(
                    base._session_paths(path_map, calibration_sessions),
                    split=f"{fold_id}:calibration",
                    **common,
                )
                validation = base.load_repaired_decisions(
                    base._session_paths(path_map, validation_sessions),
                    split=f"{fold_id}:validation",
                    **common,
                )
                config = HGBUnitConfig(
                    hypothesis=args.hypothesis,
                    policy_index=policy,
                    seed=seed,
                )
                model, result = base.run_hgb_unit_v5(
                    fit_decisions=fit,
                    calibration_decisions=calibration,
                    validation_decisions=validation,
                    noise_model=noise_model,
                    config=config,
                    fold=fold_id,
                )
                unit_dir.mkdir(parents=True, exist_ok=True)
                model_path = unit_dir / "model.pkl"
                base._write_model(model_path, model)
                unit_summary = base.commit_fresh_unit(
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
                        "sha256": base.sha256_path(summary_path),
                        "replay_manifest_sha256": unit_summary[
                            "replay_packet"
                        ]["manifest_sha256"],
                    }
                )
                completed += 1
                base.write_json(
                    args.out_dir / "progress.json",
                    {
                        "status": "running_owner_approved_hypothesis",
                        "completed_units": completed,
                        "total_units": total,
                        "preregistration_hash": prereg["preregistration_hash"],
                    },
                )
    summary = {
        "schema_version": "Protocol101FreshEntryHypothesisUnitsV1",
        "status": "unit_execution_complete_pending_evidence_stack",
        "campaign_namespace": base.FRESH_CAMPAIGN_NAMESPACE,
        "campaign_contract_sha256": plan["provenance"][
            "campaign_contract_sha256"
        ],
        "campaign_preregistration_sha256": plan["provenance"][
            "campaign_preregistration_sha256"
        ],
        "contract_id": base.CONTRACT_ID,
        "hypothesis": args.hypothesis,
        "runner_preregistration_sha256": prereg[
            "preregistration_hash"
        ],
        "simulator_version": (
            base.PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
        ),
        "two_clock_schema_version": base.TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "unit_count": len(unit_refs),
        "expected_unit_count": total,
        "unit_artifacts": unit_refs,
        "code_hashes": base.code_hashes(),
        "provenance": plan["provenance"],
        "deferred_stack_blockers": list(base.DEFERRED_STACK_BLOCKERS),
        "side_effects": base.side_effects(research_fit=True),
        "durable_resume_contract": {
            "version": (
                "fresh_campaign_v5_preregistration_source_model_and_"
                "packet_hash_verified_v3"
            ),
            "preregistration_rewritten_on_resume": False,
            "completed_model_hashes_verified_before_reuse": True,
            "old_campaign_reuse_forbidden": True,
            "mixed_v4_v5_reuse_forbidden": True,
        },
    }
    summary["summary_hash"] = base.stable_hash(summary)
    base.write_json(args.out_dir / "summary.json", summary)
    return summary


def code_hashes() -> dict[str, str]:
    """Freeze both the durable wrapper and the scientific runner it delegates to."""
    paths = (
        RUNNER_PATH,
        ORIGINAL_RUNNER_PATH,
        base.CORE_PATH,
        base.CONTRACT_PATH,
        base.SIMULATOR_PATH,
        base.SIMULATOR_V5_PATH,
        base.IDENTITY_PATH,
        base.ARTIFACT_PATH,
        base.GATE_AGGREGATOR_PATH,
        base.SMOKE_VALIDATOR_PATH,
    )
    return {str(path): base.sha256_path(path) for path in paths}


def main() -> int:
    base.RUNNER_PATH = RUNNER_PATH
    base.code_hashes = code_hashes
    base.run_fresh_hypothesis = run_fresh_hypothesis
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
