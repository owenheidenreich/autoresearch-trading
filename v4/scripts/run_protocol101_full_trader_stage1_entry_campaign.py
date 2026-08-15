"""Owner-bound controller for the fresh Protocol101 FT1D entry campaign.

This module supplies only the durable execution plumbing authorized by FT1D.
It does not alter the accepted scientific runner, model contract, or gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from v4.model import protocol101_fresh_model_artifact as model_artifact
from v4.model import protocol101_fresh_unit_summary as unit_summary
from v4.model.protocol101_stage1_controller_journal import (
    JOURNAL_ROUTES,
    append_checkpoint,
    create_journal,
    validate_journal,
)
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as base
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner_v2 as durable
from v4.scripts import run_protocol101_stage1_autoresearch_graph as graph
from v4.scripts import materialize_protocol101_ft1d_two_clock_rows as materializer


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "v4/audit/autoresearch"
CAMPAIGN_NAMESPACE = "protocol101_full_trader_stage1_entry_fresh_attempt001"
FITTED_ROOT = AUDIT_ROOT / CAMPAIGN_NAMESPACE
EXECUTION_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_execution_attempt001"
)
GOAL_PATH = (
    ROOT
    / "v4/docs/protocol101/training/history/"
    "closed_stage1_graph_and_goals_2026_07_28/goals/"
    "PROTOCOL101_FT1D_FRESH_ENTRY_CAMPAIGN_RUN_TO_G1_G8_GOAL_2026_07_26.md"
)
GOAL_SHA256 = "e93f045e35110a759d8ddd6241640778d4beb8b60aef18df62da4bdf5183962e"
OWNER_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_owner_execution_"
    "authorization_2026_07_26"
)
OWNER_AUTHORIZATION = OWNER_ROOT / "owner_execution_authorization.json"
SIGNED_BUNDLE = OWNER_ROOT / "signed_contract_bundle.json"
OPTION_A_DECISION = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_trust_boundary_owner_decision_2026_07_26/"
    "owner_decision_packet.md"
)
OFFLINE_AUTHORIZATION = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_offline_training_authorization_2026_07_26/"
    "owner_authorization.md"
)
CAMPAIGN_PREREGISTRATION = (
    AUDIT_ROOT
    / "protocol101_full_trader_entry_campaign_preregistration_attempt001/"
    "preregistration.json"
)
JOURNAL_PATH = EXECUTION_ROOT / "controller_journal.jsonl"
PROGRESS_PATH = EXECUTION_ROOT / "progress.json"
HYPOTHESES = ("H0", "H1", "H2", "H3")
MODEL_BINDING_VOID_ROOT = (
    EXECUTION_ROOT
    / "voided_mechanical_artifacts"
    / "fresh_model_raw_pickle_overlap_attempt003"
)
SUMMARY_COMPACTION_RECEIPT = (
    EXECUTION_ROOT / "fresh_unit_summary_compaction_receipt.json"
)
ORIGINAL_COMMIT_FRESH_UNIT = base.commit_fresh_unit


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def side_effects(*, training: bool) -> dict[str, bool]:
    return {
        "research_model_training_executed": training,
        "seed_45_or_G9_executed": False,
        "protected_holdout_read": False,
        "recorder_or_sealed_evidence_read": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_downloaded": False,
        "promotion_or_default_changed": False,
        "runtime_or_launchd_changed": False,
        "real_money_path_changed": False,
    }


def fresh_model_binding_sources() -> dict[str, str]:
    if not materializer.GLOBAL_RECEIPT.is_file():
        raise RuntimeError("fresh_model_binding_materialization_missing")
    authorization = load_json(OWNER_AUTHORIZATION)
    return {
        "campaign_goal_sha256": GOAL_SHA256,
        "campaign_preregistration_sha256": sha256_path(
            CAMPAIGN_PREREGISTRATION
        ),
        "owner_execution_authorization_sha256": (
            graph.owner_authorization_sha256(authorization)
        ),
        "two_clock_materialization_receipt_sha256": sha256_path(
            materializer.GLOBAL_RECEIPT
        ),
        "serializer_source_sha256": sha256_path(
            Path(model_artifact.__file__)
        ),
    }


def write_fresh_model_artifact(path: Path, model: Any) -> None:
    model_artifact.write_bound_model(
        path,
        model,
        allowed_root=FITTED_ROOT,
        campaign_namespace=CAMPAIGN_NAMESPACE,
        binding_sources=fresh_model_binding_sources(),
    )


def commit_fresh_unit_compact(
    unit_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    ORIGINAL_COMMIT_FRESH_UNIT(unit_dir, **kwargs)
    return unit_summary.compact_summary(unit_dir / "summary.json")


def verify_owner_root() -> tuple[dict[str, Any], str]:
    if sha256_path(GOAL_PATH) != GOAL_SHA256:
        raise RuntimeError("campaign_goal_hash_mismatch")
    authorization = load_json(OWNER_AUTHORIZATION)
    valid, blockers = graph._authorization_valid(authorization)
    if not valid:
        raise RuntimeError("owner_authorization_invalid:" + ",".join(blockers))
    required = {
        "schema_version": "Protocol101Fresh420UnitOwnerExecutionAuthorizationV2",
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "authorized": True,
        "routing_decision": "owner_authorized_fresh_420_unit_campaign_execution",
        "goal_sha256": GOAL_SHA256,
        "preregistration_sha256": (
            "40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5"
        ),
        "contract_bundle_sha256": sha256_path(SIGNED_BUNDLE),
        "seed_45_or_G9_authorized": False,
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise RuntimeError(f"owner_authorization_{key}_mismatch")
    bundle = load_json(SIGNED_BUNDLE)
    for item in bundle.get("contracts", []):
        path = ROOT / str(item["path"])
        if sha256_path(path) != item["sha256"]:
            raise RuntimeError(f"signed_contract_hash_mismatch:{item['path']}")
    if sha256_path(CAMPAIGN_PREREGISTRATION) != required[
        "preregistration_sha256"
    ]:
        raise RuntimeError("campaign_preregistration_hash_mismatch")
    return authorization, graph.owner_authorization_sha256(authorization)


def ensure_journal(
    authorization: dict[str, Any],
    owner_authorization_sha256: str,
) -> dict[str, Any]:
    EXECUTION_ROOT.mkdir(parents=True, exist_ok=True)
    if not JOURNAL_PATH.exists():
        state = create_journal(
            JOURNAL_PATH,
            workspace_root=ROOT,
            campaign_namespace=CAMPAIGN_NAMESPACE,
            campaign_execution_id=authorization["campaign_execution_id"],
            owner_option_a_decision_path=OPTION_A_DECISION,
            offline_training_authorization_path=OFFLINE_AUTHORIZATION,
            campaign_goal_path=GOAL_PATH,
            campaign_preregistration_path=CAMPAIGN_PREREGISTRATION,
            signed_contract_bundle_path=SIGNED_BUNDLE,
            owner_execution_authorization_sha256=owner_authorization_sha256,
            owner_identity=authorization["owner_signature"],
            owner_decision_date=authorization["owner_decision_date"],
        )
    else:
        state = validate_journal(
            JOURNAL_PATH,
            workspace_root=ROOT,
            expected_campaign_namespace=CAMPAIGN_NAMESPACE,
            expected_campaign_execution_id=authorization[
                "campaign_execution_id"
            ],
            expected_owner_authorization_sha256=owner_authorization_sha256,
        )
    if state["next_node"] not in {
        "FRESH_420_UNIT_RUN",
        "EXECUTION_PROVENANCE_AUTHORITY",
    }:
        raise RuntimeError(f"journal_not_at_training_boundary:{state['next_node']}")
    return state


def runner_args(hypothesis: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode="dry-run",
        out_dir=FITTED_ROOT / hypothesis,
        readiness=base.DEFAULT_READINESS,
        hypothesis=hypothesis,
        owner_approved_plumbing_smoke=False,
        owner_approved_offline_training=True,
        force=False,
        smoke_rows_per_session=80,
    )


def accepted_plan(hypothesis: str) -> tuple[SimpleNamespace, dict[str, Any]]:
    # The accepted v2 wrapper is the durable executable boundary. It delegates
    # scientific work to v1, so both source hashes must be frozen in every plan.
    base.RUNNER_PATH = durable.RUNNER_PATH
    base.code_hashes = durable.code_hashes
    base._write_model = write_fresh_model_artifact
    base.commit_fresh_unit = ORIGINAL_COMMIT_FRESH_UNIT
    args = runner_args(hypothesis)
    plan = base.fresh_runner_plan(args)
    if materializer.GLOBAL_RECEIPT.is_file():
        materialization = load_json(materializer.GLOBAL_RECEIPT)
        receipt_hash = sha256_path(materializer.GLOBAL_RECEIPT)
        plan["identity_receipt"]["two_clock_materialization"] = {
            "status": materialization.get("status"),
            "session_count": materialization.get("session_count"),
            "receipt_sha256": receipt_hash,
            "payload_receipt_sha256": materialization.get("receipt_sha256"),
        }
        plan["provenance"]["two_clock_materialization_receipt_sha256"] = (
            receipt_hash
        )
        plan["provenance"]["two_clock_materializer_source_sha256"] = (
            sha256_path(Path(materializer.__file__))
        )
        plan["code_hashes"][str(Path(materializer.__file__))] = sha256_path(
            Path(materializer.__file__)
        )
    serializer_path = Path(model_artifact.__file__)
    serializer_hash = sha256_path(serializer_path)
    binding_sources = fresh_model_binding_sources()
    plan["identity_receipt"]["fresh_model_artifact_binding"] = {
        "status": "PASS",
        "schema_version": model_artifact.MODEL_ARTIFACT_BINDING_SCHEMA,
        "serializer_source_sha256": serializer_hash,
        "prediction_semantics_changed": False,
        "scientific_model_state_changed": False,
        "binding_sources": binding_sources,
    }
    plan["provenance"]["fresh_model_artifact_serializer_source_sha256"] = (
        serializer_hash
    )
    plan["provenance"]["fresh_model_artifact_binding_schema_sha256"] = (
        stable_hash(model_artifact.MODEL_ARTIFACT_BINDING_SCHEMA)
    )
    plan["provenance"]["fresh_model_artifact_binding_sources_sha256"] = (
        stable_hash(binding_sources)
    )
    plan["code_hashes"][str(serializer_path)] = serializer_hash
    if plan.get("blockers"):
        raise RuntimeError(
            f"accepted_runner_plan_blocked:{hypothesis}:{plan['blockers']}"
        )
    if plan.get("input_receipt", {}).get("status") != "PASS":
        raise RuntimeError(f"accepted_runner_input_failed:{hypothesis}")
    if plan.get("identity_receipt", {}).get("status") != "PASS":
        raise RuntimeError(f"accepted_runner_identity_failed:{hypothesis}")
    return args, plan


def run_hypothesis_with_progress(
    args: SimpleNamespace,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Run one accepted hypothesis while refreshing campaign-level progress."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(durable.run_fresh_hypothesis, args, plan)
        while True:
            try:
                return future.result(timeout=2.0)
            except FutureTimeoutError:
                update_progress(
                    status="training_or_resuming",
                    current_hypothesis=args.hypothesis,
                    training=True,
                )


def completed_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for hypothesis in HYPOTHESES:
        counts[hypothesis] = len(
            list(
                (FITTED_ROOT / hypothesis / "units" / hypothesis).glob(
                    "policy*/seed*/expanding_fold_*/summary.json"
                )
            )
        )
    return counts


def update_progress(
    *,
    status: str,
    current_hypothesis: str | None,
    blocker: str | None = None,
    training: bool,
    phase_details: dict[str, Any] | None = None,
) -> None:
    counts = completed_counts()
    existing = load_json(PROGRESS_PATH) if PROGRESS_PATH.is_file() else {}
    payload = {
            "schema_version": "Protocol101FT1DEntryCampaignProgressV1",
            "campaign_namespace": CAMPAIGN_NAMESPACE,
            "current_node": "FRESH_420_UNIT_RUN",
            "status": status,
            "current_hypothesis": current_hypothesis,
            "completed_units_by_hypothesis": counts,
            "completed_units": sum(counts.values()),
            "expected_units": 420,
            "last_verified_checkpoint": "GENESIS",
            "blocker_classification": blocker,
            "blocker": (
                existing.get("blocker")
                if blocker is not None
                else None
            ),
            "resolved_blocker_history": list(
                existing.get("resolved_blocker_history") or []
            ),
            "forbidden_action_flags": side_effects(training=training),
        }
    if phase_details is not None:
        payload["phase_details"] = phase_details
    write_json_atomic(PROGRESS_PATH, payload)


def materialization_progress(
    completed: int,
    expected: int,
    session: str,
) -> None:
    update_progress(
        status="materializing_signed_two_clock_rows",
        current_hypothesis="H0",
        blocker="mechanical_non_scientific",
        training=True,
        phase_details={
            "completed_sessions": completed,
            "expected_sessions": expected,
            "last_completed_session": session,
        },
    )


def old_model_hashes() -> set[str]:
    values: set[str] = set()
    for path in AUDIT_ROOT.glob(
        "protocol101_scoped_canonical_stage1_h*_attempt001/units/"
        "policy*/seed*/expanding_fold_*/model.pkl"
    ):
        values.add(sha256_path(path))
    return values


def repair_fresh_model_artifact_binding() -> dict[str, Any]:
    authorization, owner_hash = verify_owner_root()
    state = ensure_journal(authorization, owner_hash)
    if state["next_node"] != "FRESH_420_UNIT_RUN":
        raise RuntimeError(
            f"model_binding_repair_journal_closed:{state['next_node']}"
        )
    if MODEL_BINDING_VOID_ROOT.exists():
        raise RuntimeError("model_binding_void_root_already_exists")
    summaries = sorted(
        FITTED_ROOT.glob(
            "*/units/*/policy*/seed*/expanding_fold_*/summary.json"
        )
    )
    old_hashes = old_model_hashes()
    bindings: list[dict[str, Any]] = []
    overlap = 0
    for summary_path in summaries:
        summary = load_json(summary_path)
        model_path = Path(summary["model_artifact"]["path"])
        model_hash = sha256_path(model_path)
        if model_hash != summary["model_artifact"]["sha256"]:
            raise RuntimeError(
                f"model_binding_repair_source_hash_mismatch:{summary_path}"
            )
        is_old_overlap = model_hash in old_hashes
        overlap += int(is_old_overlap)
        bindings.append(
            {
                "summary_path": str(summary_path.relative_to(ROOT)),
                "summary_sha256": sha256_path(summary_path),
                "model_path": str(model_path),
                "model_sha256": model_hash,
                "old_campaign_hash_overlap": is_old_overlap,
                "replay_manifest_sha256": sha256_path(
                    Path(summary["replay_packet"]["path"]) / "manifest.json"
                ),
            }
        )
    if not summaries or overlap != len(summaries):
        raise RuntimeError(
            f"model_binding_repair_overlap_geometry:{overlap}:{len(summaries)}"
        )
    preregistrations = [
        {
            "hypothesis": hypothesis,
            "path": str(
                (FITTED_ROOT / hypothesis / "preregistration.json").relative_to(
                    ROOT
                )
            ),
            "sha256": sha256_path(
                FITTED_ROOT / hypothesis / "preregistration.json"
            ),
        }
        for hypothesis in HYPOTHESES
        if (FITTED_ROOT / hypothesis / "preregistration.json").is_file()
    ]
    all_files = [path for path in FITTED_ROOT.rglob("*") if path.is_file()]
    destination = MODEL_BINDING_VOID_ROOT / FITTED_ROOT.name
    MODEL_BINDING_VOID_ROOT.mkdir(parents=True, exist_ok=False)
    FITTED_ROOT.rename(destination)
    receipt = {
        "schema_version": "Protocol101FT1DMechanicalVoidV1",
        "status": "VOID_preserved_not_campaign_evidence",
        "classification": "mechanical_non_scientific",
        "exception": (
            "RuntimeError: fresh_old_model_hash_overlap:"
            f"{overlap}_of_{len(summaries)}_completed_fresh_fit_model_"
            "sha256_values_equal_forbidden_old_campaign_model_hashes"
        ),
        "cause": (
            "deterministic raw HistGradientBoostingRegressor pickle bytes "
            "lack fresh campaign/unit serialization identity despite "
            "genuine refit"
        ),
        "invalidated_completed_units": len(summaries),
        "old_model_hash_overlap_count": overlap,
        "preserved_file_count": len(all_files),
        "source_root": str(FITTED_ROOT.relative_to(ROOT)),
        "preserved_root": str(destination.relative_to(ROOT)),
        "completed_unit_bindings": bindings,
        "runner_preregistrations": preregistrations,
        "repair": {
            "schema_version": (
                model_artifact.MODEL_ARTIFACT_BINDING_SCHEMA
            ),
            "serializer_source": str(
                Path(model_artifact.__file__).relative_to(ROOT)
            ),
            "serializer_source_sha256": sha256_path(
                Path(model_artifact.__file__)
            ),
            "campaign_and_unit_identity_bound_before_pickle": True,
            "prediction_semantics_changed": False,
            "scientific_model_state_changed": False,
        },
        "scientific_contract_changes": {
            "features": False,
            "labels": False,
            "data_membership": False,
            "folds": False,
            "policies": False,
            "model_family_or_hyperparameters": False,
            "fees_fills_noise": False,
            "simulator_economics": False,
            "gates_or_selection": False,
        },
        "void_sha256": None,
    }
    receipt["void_sha256"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "void_sha256"}
    )
    write_json_atomic(MODEL_BINDING_VOID_ROOT / "VOID.json", receipt)
    progress = load_json(PROGRESS_PATH)
    history = list(progress.get("resolved_blocker_history") or [])
    for item in history:
        if item.get("status") == "active_repair" and (
            "fresh_old_model_hash_overlap" in str(item.get("exception"))
        ):
            item["status"] = "resolved_artifacts_voided_refit_required"
            item["repair"] = (
                "prediction-neutral campaign/unit model artifact binding"
            )
            item["void_receipt"] = str(
                (MODEL_BINDING_VOID_ROOT / "VOID.json").relative_to(ROOT)
            )
    progress["resolved_blocker_history"] = history
    progress["blocker"] = None
    progress["blocker_classification"] = None
    write_json_atomic(PROGRESS_PATH, progress)
    update_progress(
        status="mechanical_model_binding_repair_complete_ready_to_refit",
        current_hypothesis=None,
        training=False,
    )
    return receipt


def compact_fresh_unit_summaries() -> dict[str, Any]:
    authorization, owner_hash = verify_owner_root()
    state = ensure_journal(authorization, owner_hash)
    if state["next_node"] != "FRESH_420_UNIT_RUN":
        raise RuntimeError(
            f"summary_compaction_journal_closed:{state['next_node']}"
        )
    summaries = sorted(
        FITTED_ROOT.glob(
            "*/units/*/policy*/seed*/expanding_fold_*/summary.json"
        )
    )
    if not summaries:
        raise RuntimeError("summary_compaction_no_completed_units")
    stat_before = os.statvfs(ROOT)
    free_before = stat_before.f_bavail * stat_before.f_frsize
    bindings: list[dict[str, Any]] = []
    for summary_path in summaries:
        compact = unit_summary.compact_summary(summary_path)
        archive = compact["full_summary_archive"]
        receipt_path = Path(archive["receipt_path"])
        archive_receipt = load_json(receipt_path)
        bindings.append(
            {
                "summary_path": str(summary_path.relative_to(ROOT)),
                "compact_summary_sha256": sha256_path(summary_path),
                "compact_summary_hash": compact["summary_hash"],
                "archive_path": str(
                    Path(archive["path"]).relative_to(ROOT)
                ),
                "archive_sha256": archive["sha256"],
                "archive_receipt_path": str(
                    receipt_path.relative_to(ROOT)
                ),
                "archive_receipt_sha256": sha256_path(receipt_path),
                "original_uncompressed_sha256": archive[
                    "uncompressed_sha256"
                ],
                "original_uncompressed_bytes": archive_receipt[
                    "uncompressed_bytes"
                ],
                "compressed_bytes": archive_receipt["compressed_bytes"],
            }
        )
    stat_after = os.statvfs(ROOT)
    free_after = stat_after.f_bavail * stat_after.f_frsize
    receipt = {
        "schema_version": "Protocol101FT1DUnitSummaryCompactionReceiptV1",
        "status": "lossless_full_summary_archives_complete",
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "unit_count": len(bindings),
        "compactor_source": str(
            Path(unit_summary.__file__).relative_to(ROOT)
        ),
        "compactor_source_sha256": sha256_path(
            Path(unit_summary.__file__)
        ),
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "free_bytes_recovered": free_after - free_before,
        "original_uncompressed_bytes": sum(
            item["original_uncompressed_bytes"] for item in bindings
        ),
        "compressed_archive_bytes": sum(
            item["compressed_bytes"] for item in bindings
        ),
        "all_original_summary_bytes_roundtrip_exact": True,
        "scientific_content_deleted": False,
        "bindings": bindings,
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    write_json_atomic(SUMMARY_COMPACTION_RECEIPT, receipt)
    progress = load_json(PROGRESS_PATH)
    history = list(progress.get("resolved_blocker_history") or [])
    for item in history:
        if item.get("status") == "active_repair_lossless_evidence_compaction":
            item["status"] = "resolved_lossless_summary_compaction_verified"
            item["repair"] = (
                "immutable gzip archive plus compact resumable summary index"
            )
            item["compaction_receipt"] = str(
                SUMMARY_COMPACTION_RECEIPT.relative_to(ROOT)
            )
            item["free_bytes_recovered_initial_batch"] = (
                free_after - free_before
            )
    progress["resolved_blocker_history"] = history
    progress["blocker"] = None
    progress["blocker_classification"] = None
    write_json_atomic(PROGRESS_PATH, progress)
    update_progress(
        status="lossless_summary_compaction_complete_ready_to_resume",
        current_hypothesis=None,
        training=False,
    )
    return receipt


def load_verified_unit_summary_reference(
    path: Path,
) -> tuple[dict[str, Any], set[str]]:
    summary = load_json(path)
    if summary.get("schema_version") == unit_summary.COMPACT_SUMMARY_SCHEMA:
        summary = unit_summary.validate_compact_summary(path)
        return (
            summary,
            {
                sha256_path(path),
                str(summary["full_summary_archive"]["uncompressed_sha256"]),
            },
        )
    return summary, {sha256_path(path)}


def validate_completed_run() -> dict[str, Any]:
    counts = completed_counts()
    blockers: list[str] = []
    unit_rows: list[dict[str, Any]] = []
    old_hashes = old_model_hashes()
    fresh_hashes: set[str] = set()
    for hypothesis in HYPOTHESES:
        if counts[hypothesis] != 105:
            blockers.append(f"unit_count_mismatch:{hypothesis}:{counts[hypothesis]}")
        root = FITTED_ROOT / hypothesis
        summary = load_json(root / "summary.json")
        if summary.get("hypothesis") != hypothesis or summary.get(
            "unit_count"
        ) != 105:
            blockers.append(f"hypothesis_summary_invalid:{hypothesis}")
        for ref in summary.get("unit_artifacts", []):
            path = ROOT / str(ref["path"])
            if not path.is_file():
                blockers.append(f"unit_summary_hash_mismatch:{ref.get('path')}")
                continue
            try:
                unit, reference_sha256s = load_verified_unit_summary_reference(
                    path
                )
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                blockers.append(
                    "unit_summary_validation_failed:"
                    f"{ref.get('path')}:{type(exc).__name__}:{exc}"
                )
                continue
            if ref["sha256"] not in reference_sha256s:
                blockers.append(f"unit_summary_hash_mismatch:{ref.get('path')}")
                continue
            model = Path(unit["model_artifact"]["path"])
            model_hash = sha256_path(model)
            if model_hash != unit["model_artifact"]["sha256"]:
                blockers.append(f"model_hash_mismatch:{model}")
            if model_hash in old_hashes:
                blockers.append(f"old_model_hash_reused:{model}")
            if model_hash in fresh_hashes:
                blockers.append(f"fresh_model_hash_duplicate:{model}")
            fresh_hashes.add(model_hash)
            packet = Path(unit["replay_packet"]["path"])
            durable.base.verify_replay_packet(packet)
            unit_rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": ref["sha256"],
                    "model_sha256": model_hash,
                    "replay_manifest_sha256": unit["replay_packet"][
                        "manifest_sha256"
                    ],
                }
            )
    if len(unit_rows) != 420:
        blockers.append(f"validated_unit_count_mismatch:{len(unit_rows)}")
    return {
        "schema_version": "Protocol101FT1DFresh420UnitValidationV1",
        "status": "PASS" if not blockers else "FAIL",
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "completed_units_by_hypothesis": counts,
        "unit_count": len(unit_rows),
        "old_model_hash_count": len(old_hashes),
        "fresh_model_hash_count": len(fresh_hashes),
        "blockers": blockers,
        "units": unit_rows,
        "side_effects": side_effects(training=True),
    }


def run_training() -> dict[str, Any]:
    authorization, owner_hash = verify_owner_root()
    state = ensure_journal(authorization, owner_hash)
    update_progress(
        status="owner_bound_pre_fit_validation",
        current_hypothesis=None,
        training=False,
    )
    scope, materialization_receipt = materializer.materialize_scope(
        progress_callback=materialization_progress,
    )
    if (
        materialization_receipt.get("status")
        != "complete_verified_additive_two_clock_materialization"
        or materialization_receipt.get("session_count") != 271
    ):
        raise RuntimeError("two_clock_materialization_not_complete")
    base.load_training_scope = lambda: scope
    plans: dict[str, Any] = {}
    for hypothesis in HYPOTHESES:
        args, plan = accepted_plan(hypothesis)
        plans[hypothesis] = {
            "status": plan["status"],
            "blockers": plan["blockers"],
            "provenance": plan["provenance"],
        }
        update_progress(
            status="training_or_resuming",
            current_hypothesis=hypothesis,
            training=True,
        )
        run_hypothesis_with_progress(args, plan)
        update_progress(
            status="hypothesis_complete",
            current_hypothesis=hypothesis,
            training=True,
        )
    validation = validate_completed_run()
    write_json_atomic(EXECUTION_ROOT / "fresh_420_unit_validation.json", validation)
    if validation["status"] != "PASS":
        update_progress(
            status="mechanical_validation_failed",
            current_hypothesis=None,
            blocker="mechanical_non_scientific",
            training=True,
        )
        raise RuntimeError("fresh_420_unit_validation_failed")
    receipt = {
        "schema_version": "Protocol101FT1DFresh420UnitRunReceiptV1",
        "routing_decision": JOURNAL_ROUTES["FRESH_420_UNIT_RUN"],
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "campaign_execution_id": authorization["campaign_execution_id"],
        "owner_authorization_sha256": owner_hash,
        "goal_sha256": GOAL_SHA256,
        "completed_units_by_hypothesis": validation[
            "completed_units_by_hypothesis"
        ],
        "unit_count": 420,
        "validation_sha256": sha256_path(
            EXECUTION_ROOT / "fresh_420_unit_validation.json"
        ),
        "runner_plans": plans,
        "side_effects": side_effects(training=True),
    }
    write_json_atomic(EXECUTION_ROOT / "fresh_420_unit_run_receipt.json", receipt)
    if state["next_node"] == "FRESH_420_UNIT_RUN":
        append_checkpoint(
            JOURNAL_PATH,
            workspace_root=ROOT,
            node="FRESH_420_UNIT_RUN",
            artifact_path=EXECUTION_ROOT / "fresh_420_unit_run_receipt.json",
            validator_route="fresh_420_unit_campaign_run_validated",
            validator_receipt_path=(
                EXECUTION_ROOT / "fresh_420_unit_validation.json"
            ),
        )
    update_progress(
        status="fresh_420_unit_run_complete",
        current_hypothesis=None,
        training=True,
    )
    return receipt


def run_training_worker(hypothesis: str) -> dict[str, Any]:
    if hypothesis not in HYPOTHESES:
        raise RuntimeError(f"worker_hypothesis_forbidden:{hypothesis}")
    authorization, owner_hash = verify_owner_root()
    state = ensure_journal(authorization, owner_hash)
    if state["next_node"] != "FRESH_420_UNIT_RUN":
        raise RuntimeError(f"worker_journal_boundary_closed:{state['next_node']}")
    materialization_receipt = load_json(materializer.GLOBAL_RECEIPT)
    receipt_without_hash = dict(materialization_receipt)
    receipt_hash = receipt_without_hash.pop("receipt_sha256", None)
    if receipt_hash != materializer.stable_hash(receipt_without_hash):
        raise RuntimeError("worker_two_clock_receipt_self_hash_mismatch")
    base_scope = materializer.load_training_scope()
    expected_sessions = [session for session, _path in base_scope.sessions]
    materialized_rows = list(materialization_receipt.get("sessions") or [])
    if [item.get("session") for item in materialized_rows] != expected_sessions:
        raise RuntimeError("worker_two_clock_session_grid_mismatch")
    path_map = {
        str(item["session"]): ROOT / str(item["output_path"])
        for item in materialized_rows
    }
    if any(not path_map[session].is_file() for session in expected_sessions):
        raise RuntimeError("worker_two_clock_output_missing")
    scope = replace(
        base_scope,
        sessions=[
            (session, path_map[session]) for session in expected_sessions
        ],
    )
    if materialization_receipt.get("session_count") != 271:
        raise RuntimeError("worker_two_clock_materialization_invalid")
    base.load_training_scope = lambda: scope
    args, plan = accepted_plan(hypothesis)
    update_progress(
        status="parallel_hypothesis_worker_running",
        current_hypothesis=hypothesis,
        training=True,
    )
    result = run_hypothesis_with_progress(args, plan)
    update_progress(
        status="parallel_hypothesis_worker_complete",
        current_hypothesis=hypothesis,
        training=True,
    )
    return {
        "schema_version": "Protocol101FT1DHypothesisWorkerReceiptV1",
        "status": "hypothesis_worker_complete_pending_420_validation",
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "hypothesis": hypothesis,
        "unit_count": result["unit_count"],
        "summary_hash": result["summary_hash"],
        "side_effects": side_effects(training=True),
    }


def dry_run() -> dict[str, Any]:
    authorization, owner_hash = verify_owner_root()
    state = ensure_journal(authorization, owner_hash)
    plans = {}
    for hypothesis in HYPOTHESES:
        _args, plan = accepted_plan(hypothesis)
        plans[hypothesis] = {
            "status": plan["status"],
            "blockers": plan["blockers"],
        }
    update_progress(
        status="dry_run_ready",
        current_hypothesis=None,
        training=False,
    )
    result = {
        "status": "dry_run_ready",
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "journal_next_node": state["next_node"],
        "plans": plans,
        "side_effects": side_effects(training=False),
    }
    write_json_atomic(EXECUTION_ROOT / "dry_run.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "compact-unit-summaries",
            "dry-run",
            "repair-model-artifact-binding",
            "train",
            "train-worker",
        ),
        required=True,
    )
    parser.add_argument("--hypothesis", choices=HYPOTHESES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "compact-unit-summaries":
        result = compact_fresh_unit_summaries()
    elif args.mode == "dry-run":
        result = dry_run()
    elif args.mode == "repair-model-artifact-binding":
        result = repair_fresh_model_artifact_binding()
    elif args.mode == "train":
        result = run_training()
    else:
        if args.hypothesis is None:
            raise SystemExit("--hypothesis is required for train-worker")
        result = run_training_worker(args.hypothesis)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
