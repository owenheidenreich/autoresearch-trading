from __future__ import annotations

import json
from pathlib import Path

from v4.live.protocol101_synchronization import (
    Protocol101CanonicalSemanticsAuditV1,
    Protocol101DataPlaneDecisionMapV1,
    Protocol101DataPlaneSelectionV1,
    Protocol101ExistingCaptureRepairabilityDiagnosticV1,
    Protocol101FairContractTrainingDryRunV1,
    Protocol101FairContractTrainingDesignV1,
    Protocol101FairContractTrainingPreflightV1,
    Protocol101FairGamePolicyV1,
    Protocol101PaperReadinessGateV2,
    Protocol101RepairabilityDecisionV1,
    Protocol101SingleDaySynchronizationGateV1,
    evaluate_historical_non_inferiority,
)
from v4.scripts.run_protocol101_fair_contract_training_preflight import (
    build_packet as build_fair_contract_training_preflight,
    processed_file_inventory,
)
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import compute_registry_hash
from v4.scripts.run_protocol101_fair_contract_training_design import (
    build_packet as build_fair_contract_training_design,
)
from v4.scripts.run_protocol101_fair_contract_training_dry_run import (
    split_lookup as fair_contract_split_lookup,
)
from v4.scripts.run_protocol101_data_plane_decision_map import classify_feature
from v4.scripts.run_protocol101_h1_canonical_semantics_audit import (
    classify_h1_feature_route,
    dominant_group_route,
)
from v4.scripts.run_protocol101_h1_top_example_inspection import example_signature
from v4.scripts.run_protocol101_h1_repairability_decision import route_decisions
from v4.scripts.run_protocol101_cross_vendor_feature_audit import _load, _token_names, _universe
from v4.scripts.run_protocol101_existing_capture_repairability_diagnostic import (
    build_repairability_routes,
    summarize_lost_trade_source_policy,
)

import pandas as pd

from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED


def test_historical_non_inferiority_rejects_current_live_contract_metrics() -> None:
    legacy = {
        "trades": 221,
        "total_pnl": 69180.0,
        "return_on_premium": 0.1065,
        "profit_factor": 2.07,
        "max_drawdown_dollars": -9650.0,
    }
    candidate = {
        "trades": 151,
        "total_pnl": 17480.0,
        "return_on_premium": 0.0381,
        "profit_factor": 1.29,
        "max_drawdown_dollars": -11910.0,
        "top_1_day_pnl_share": 0.30,
        "top_5_day_pnl_share": 1.37,
        "top_10_trade_pnl_share": 1.86,
    }

    result = evaluate_historical_non_inferiority(
        legacy=legacy,
        candidate=candidate,
        adverse_fill_total_pnl=None,
    )

    assert result["status"] == "fail"
    assert result["checks"]["trade_count_ge_75pct"] is False
    assert result["checks"]["positive_after_adverse_fill"] is False


def test_data_plane_selection_waits_for_evidence() -> None:
    selection = Protocol101DataPlaneSelectionV1.evaluate(
        enriched_ibkr={"evaluation_complete": False},
        same_vendor_model_feed={"evaluation_complete": False},
    )
    assert selection.status == "evidence_pending"
    assert selection.selected_data_plane is None
    assert selection.retraining_required is False


def test_fair_contract_training_preflight_contract() -> None:
    packet = Protocol101FairContractTrainingPreflightV1(
        status="blocked_dataset_cleanup_required",
        selected_feature_contract="protocol101-live-v1",
        model_training_authorized=False,
        paper_submit_allowed=False,
        decision="clean_canonical_live_v1_dataset_before_any_training_design_or_paper_submit",
        dataset_checks={"processed_exact_session_count": 1},
        synchronization_checks={"historical_non_inferiority_status": "fail"},
        blockers=["duplicate_or_ambiguous_processed_session_files"],
        next_actions=["clean duplicate files"],
    )

    assert packet.schema_version == "Protocol101FairContractTrainingPreflightV1"
    assert packet.paper_submit_allowed is False
    assert packet.to_dict()["blockers"] == ["duplicate_or_ambiguous_processed_session_files"]


def test_fair_contract_training_design_requires_owner_approval() -> None:
    packet = Protocol101FairContractTrainingDesignV1(
        status="owner_approval_required_before_training",
        decision="fair_contract_training_design_ready_no_training_executed",
        selected_feature_contract="protocol101-live-v1",
        model_training_authorized=False,
        threshold_tuning_authorized=False,
        paper_submit_allowed=False,
        allowed_data={"glob_loading_allowed": False},
        excluded_data=[],
        split_policy={"scope": "q1_2026_development_only_not_final_holdout"},
        training_target={"entry_model": "new_candidate_required"},
        success_gates=[],
        forbidden_actions=["Do not train without explicit owner approval."],
        required_owner_approvals=["approval_to_start_model_training_on_protocol101_live_v1_manifest"],
    )

    assert packet.schema_version == "Protocol101FairContractTrainingDesignV1"
    assert packet.model_training_authorized is False
    assert packet.paper_submit_allowed is False
    assert packet.allowed_data["glob_loading_allowed"] is False


def test_fair_contract_training_dry_run_contract() -> None:
    packet = Protocol101FairContractTrainingDryRunV1(
        status="blocked",
        decision="fix_manifest_ingestion_before_training_runner",
        selected_feature_contract="protocol101-live-v1",
        model_training_executed=False,
        threshold_tuning_executed=False,
        broker_endpoint_called=False,
        split_summary={"train": {"sessions": 1}},
        feature_summary={"feature_dimensions": {"71": 10}, "feature_imputation_required": True},
        label_summary={"target": {"n": 10, "positive": 0, "negative": 0}},
        blockers=["all_zero_label:target"],
        next_actions=["generate labels"],
    )

    assert packet.schema_version == "Protocol101FairContractTrainingDryRunV1"
    assert packet.model_training_executed is False
    assert packet.broker_endpoint_called is False
    assert packet.feature_summary["feature_dimensions"] == {"71": 10}
    assert packet.feature_summary["feature_imputation_required"] is True
    assert packet.blockers == ["all_zero_label:target"]


def test_fair_contract_dry_run_split_lookup_marks_embargoed_sessions() -> None:
    lookup = fair_contract_split_lookup(
        {
            "train_sessions": ["2026-01-02"],
            "validation_sessions": ["2026-03-03"],
            "diagnostic_test_sessions": ["2026-03-18"],
            "embargoed_sessions": [
                {"session": "2026-03-02", "boundary": "train_to_validation"},
                {"session": "2026-03-17", "boundary": "validation_to_diagnostic_test"},
            ],
        }
    )

    assert lookup["2026-01-02"] == "train"
    assert lookup["2026-03-03"] == "validation"
    assert lookup["2026-03-18"] == "diagnostic_test"
    assert lookup["2026-03-02"] == "embargoed"
    assert lookup["2026-03-17"] == "embargoed"


def test_fair_contract_training_design_builds_manifest_based_split(tmp_path: Path) -> None:
    preflight = {
        "status": "ready_for_owner_authorized_fair_contract_training_design",
        "selected_feature_contract": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    }
    sessions = [
        "2025-07-01",
        "2026-01-02",
        "2026-01-05",
        "2026-02-02",
        "2026-03-02",
        "2026-03-03",
    ]
    manifest = {
        "included_sessions": [
            {
                "session": session,
                "processed_file": f"data/{session}.pkl",
                "normalized_base_file": f"norm/{session}.parquet",
                "normalized_official_context_file": f"norm/{session}_official.parquet",
            }
            for session in sessions
        ],
        "excluded_processed_files": [{"file": "data/2026-01-02 2.pkl", "reason": "duplicate"}],
    }

    packet = build_fair_contract_training_design(
        preflight,
        manifest,
        manifest_path=tmp_path / "manifest.json",
    )

    assert packet.status == "owner_approval_required_before_training"
    assert packet.selected_feature_contract == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
    assert packet.model_training_authorized is False
    assert packet.allowed_data["included_session_count"] == 6
    assert packet.allowed_data["glob_loading_allowed"] is False
    assert packet.split_policy["same_timestamp_split_allowed"] is False
    assert "2025-07-01" in packet.split_policy["train_sessions"]
    assert packet.split_policy["embargoed_sessions"][0]["session"] == "2026-03-02"
    assert "2026-03-02" not in packet.split_policy["validation_sessions"]
    assert any(
        row.get("file") == "data/2026-01-02 2.pkl"
        and row.get("source") == "processed_duplicate"
        for row in packet.excluded_data
    )


def test_fair_contract_training_preflight_blocks_duplicate_processed_files(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    normalized = tmp_path / "normalized"
    sync = tmp_path / "sync"
    out = tmp_path / "out"
    processed.mkdir()
    normalized.mkdir()
    sync.mkdir()
    row = {
        "decision_time": pd.Timestamp("2026-01-02T14:31:00Z"),
        "source_quote_time": pd.Timestamp("2026-01-02T14:31:00Z"),
        "source_context_time": pd.Timestamp("2026-01-02T14:30:00Z"),
        "feature_contract_version": "protocol101-live-v1",
        "feature_names": ["bid", "ask", "market_delta.spx_close"],
    }
    for name in ("2026-01-02.pkl", "2026-01-02 2.pkl"):
        with (processed / name).open("wb") as handle:
            import pickle

            pickle.dump([row], handle)
    for name in (
        "databento_spxw_0dte_2026-01-02.parquet",
        "databento_spxw_0dte_2026-01-02_official_context.parquet",
    ):
        pd.DataFrame([{"event_time": "2026-01-02T14:31:00Z"}]).to_parquet(normalized / name)
    (sync / "historical_non_inferiority_gate.json").write_text(json.dumps({"status": "fail"}))
    (sync / "historical_non_inferiority_gate_fair_timing_context_lag0_decision_plus1.json").write_text(
        json.dumps({"status": "fail"})
    )
    (sync / "single_day_synchronization_gate.json").write_text(json.dumps({"status": "pass"}))
    (sync / "paper_readiness_gate.json").write_text(json.dumps({"status": "blocked"}))
    (sync / "fair_game_policy.json").write_text(
        json.dumps({"status": "reject_same_minute_timing_repair_continue_other_causal_routes"})
    )

    class Args:
        processed_dir = processed
        normalized_dir = normalized
        sync_root = sync
        out_dir = out
        feature_contract = "protocol101-live-v1"

    inventory = processed_file_inventory(processed)
    packet, rows = build_fair_contract_training_preflight(Args())

    assert inventory["duplicate_or_ambiguous"]["2026-01-02"]
    assert packet.status == "blocked_dataset_cleanup_required"
    assert "duplicate_or_ambiguous_processed_session_files" in packet.blockers
    assert packet.dataset_checks["canonical_processed_files"][0]["processed_file"].endswith(
        "2026-01-02.pkl"
    )
    assert packet.dataset_checks["excluded_processed_files"][0]["file"].endswith(
        "2026-01-02 2.pkl"
    )
    assert rows[0]["future_context_rows"] == 0
    assert packet.dataset_checks["row_checks"]["context_lag_minutes_median"] == 1.0


def test_fair_contract_training_preflight_can_filter_manifest_by_acceptance_registry(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    normalized = tmp_path / "normalized"
    sync = tmp_path / "sync"
    out = tmp_path / "out"
    acceptance_path = tmp_path / "acceptance.json"
    processed.mkdir()
    normalized.mkdir()
    sync.mkdir()
    for session in ("2025-01-02", "2025-01-03"):
        row = {
            "decision_time": pd.Timestamp(f"{session}T14:32:00Z"),
            "source_quote_time": pd.Timestamp(f"{session}T14:32:00Z"),
            "source_context_time": pd.Timestamp(f"{session}T14:31:00Z"),
            "feature_contract_version": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
            "feature_names": ["market_delta.spx_close"],
        }
        with (processed / f"{session}.pkl").open("wb") as handle:
            import pickle

            pickle.dump([row], handle)
        for suffix in ("", "_official_context"):
            pd.DataFrame([{"event_time": f"{session}T14:32:00Z"}]).to_parquet(
                normalized / f"databento_spxw_0dte_{session}{suffix}.parquet"
            )
    (sync / "historical_non_inferiority_gate.json").write_text(json.dumps({"status": "fail"}))
    (sync / "historical_non_inferiority_gate_fair_timing_context_lag0_decision_plus1.json").write_text(
        json.dumps({"status": "fail"})
    )
    (sync / "single_day_synchronization_gate.json").write_text(json.dumps({"status": "pass"}))
    (sync / "paper_readiness_gate.json").write_text(json.dumps({"status": "blocked"}))
    (sync / "fair_game_policy.json").write_text(
        json.dumps({"status": "reject_same_minute_timing_repair_continue_other_causal_routes"})
    )
    registry = {
        "status": "pass",
        "sessions": [
            {"session": "2025-01-02", "status": "pass"},
        ],
    }
    registry["registry_hash"] = compute_registry_hash(registry)
    acceptance_path.write_text(json.dumps(registry))

    class Args:
        processed_dir = processed
        normalized_dir = normalized
        sync_root = sync
        out_dir = out
        feature_contract = FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
        acceptance_registry = acceptance_path

    packet, _rows = build_fair_contract_training_preflight(Args())

    included = packet.dataset_checks["canonical_processed_files"]
    excluded = packet.dataset_checks["excluded_acceptance_sessions"]
    assert packet.status == "ready_for_owner_authorized_fair_contract_training_design"
    assert [row["session"] for row in included] == ["2025-01-02"]
    assert excluded == [
        {
            "session": "2025-01-03",
            "processed_file": str(processed / "2025-01-03.pkl"),
            "reason": "not_present_as_pass_in_acceptance_registry",
        }
    ]
    assert packet.dataset_checks["acceptance_registry"]["pass_session_count"] == 1


def test_data_plane_decision_map_contract_and_feature_classification() -> None:
    packet = Protocol101DataPlaneDecisionMapV1(
        headline={"net_degradation": -51700.0},
        feature_families=[],
        hypotheses=[],
        decisions=["do not tune threshold"],
        blockers=["historical non-inferiority fails"],
    )

    assert packet.schema_version == "Protocol101DataPlaneDecisionMapV1"
    assert packet.to_dict()["headline"]["net_degradation"] == -51700.0
    assert classify_feature("pattern_last10_breakout")["family"] == "pattern_tokens"
    assert classify_feature("log_open_interest")["family"] == "open_interest"
    assert classify_feature("market_delta.momentum_5m")["family"] == "index_context"


def test_h1_canonical_semantics_audit_contract_and_routes() -> None:
    packet = Protocol101CanonicalSemanticsAuditV1(
        headline={"status": "offline_h1_audit_complete"},
        repair_routes=[],
        field_routes=[],
        conclusions=["same-input fixtures pass"],
        next_actions=["inspect high-PnL examples"],
    )

    assert packet.schema_version == "Protocol101CanonicalSemanticsAuditV1"
    assert packet.to_dict()["headline"]["status"] == "offline_h1_audit_complete"
    assert (
        classify_h1_feature_route("pattern_last10_breakout")["repair_route"]
        == "canonical_pattern_semantics"
    )
    assert (
        classify_h1_feature_route("market_delta.momentum_5m")["repair_route"]
        == "canonical_index_context_semantics"
    )
    assert (
        classify_h1_feature_route("structure.last10_break_state")["repair_route"]
        == "canonical_structure_semantics"
    )
    assert (
        classify_h1_feature_route("log_open_interest")["repair_route"]
        == "runtime_zeroed_pressure_diagnostics"
    )
    assert (
        classify_h1_feature_route("distance_points")["repair_route"]
        == "canonical_candidate_geometry"
    )
    assert dominant_group_route("tokens") == "canonical_pattern_semantics"
    assert dominant_group_route("quote_spread") == "quote_tradability_guardrail"


def test_h1_top_example_signature_detects_same_quote_semantic_drift() -> None:
    signature = example_signature(
        legacy_trace={"selected_action": "enter"},
        live_trace={"selected_action": "wait"},
        legacy_candidate={"bid": 10.0, "ask": 10.5, "spread": 0.5},
        live_candidate={"bid": 10.0, "ask": 10.5, "spread": 0.5},
        pattern_diff_count=2,
        pressure_zeroed=True,
    )

    assert signature == "same_quote_pressure_zeroed_and_pattern_drift"


def test_h1_repairability_decision_blocks_contract_mutation_without_causal_proof() -> None:
    packet = Protocol101RepairabilityDecisionV1(
        status="contract_change_blocked_pending_causal_source_policy_proof",
        decision="do_not_update_shared_feature_contract_or_rerun_q1_as_promotion_claim_yet",
        immediate_contract_change_allowed=False,
        q1_rerun_required_now=False,
        route_decisions=[],
        next_actions=["collect July 6 confirmation evidence"],
    )
    routes = route_decisions(
        {
            "repair_routes": [
                {
                    "repair_route": "canonical_pattern_semantics",
                    "lost_trade_pnl_exposure": 58810.0,
                }
            ]
        },
        {
            "signatures": [
                {
                    "signature": "same_quote_pressure_zeroed_and_pattern_drift",
                    "examples": 25,
                }
            ],
            "diff_routes": [
                {"repair_route": "canonical_pattern_semantics", "diff_rows": 56},
                {"repair_route": "runtime_zeroed_pressure_diagnostics", "diff_rows": 96},
            ],
        },
    )

    assert packet.schema_version == "Protocol101RepairabilityDecisionV1"
    assert packet.immediate_contract_change_allowed is False
    assert routes[0]["route"] == "canonical_pattern_semantics"
    assert routes[0]["repairability"] == "not_safe_to_mutate_contract_yet"
    assert routes[2]["route"] == "runtime_zeroed_pressure_diagnostics"
    assert routes[2]["repairability"] == "diagnostic_only_until_live_equivalent_proven"


def test_h1_repairability_rejects_same_minute_completed_bar_after_fair_timing_failure() -> None:
    routes = route_decisions(
        {
            "repair_routes": [
                {
                    "repair_route": "canonical_pattern_semantics",
                    "lost_trade_pnl_exposure": 58810.0,
                }
            ]
        },
        {
            "signatures": [
                {
                    "signature": "same_quote_pressure_zeroed_and_pattern_drift",
                    "examples": 25,
                }
            ],
            "diff_routes": [
                {"repair_route": "canonical_pattern_semantics", "diff_rows": 56},
            ],
        },
        {"protocol101_live_v1": {"total_pnl": -14230.0}},
    )

    assert routes[0]["route"] == "canonical_pattern_semantics"
    assert routes[0]["repairability"] == "rejected_as_same_minute_completed_bar_repair"
    assert "timing privilege" in routes[0]["reason"]


def test_existing_capture_diagnostic_uses_existing_captures_without_contract_change() -> None:
    lost = pd.DataFrame(
        [
            {
                "pnl": 1000.0,
                "near_match": True,
                "near_minute_delta": 1.0,
                "near_strike_delta": 5.0,
                "dominant_gap_closure_group": "tokens",
            },
            {
                "pnl": 500.0,
                "near_match": False,
                "near_minute_delta": None,
                "near_strike_delta": None,
                "dominant_gap_closure_group": "open_interest",
            },
        ]
    )
    policy, _groups = summarize_lost_trade_source_policy(lost)
    routes = build_repairability_routes(
        q1_source_policy=policy,
        h1_top={"signatures": [{"signature": "same_quote_pressure_zeroed_and_pattern_drift"}]},
        h1_repairability={"decision": "do_not_update_shared_feature_contract_or_rerun_q1_as_promotion_claim_yet"},
        capture_timing=[
            {
                "same_input_exact": True,
                "context_lag_min_median": -1,
                "future_context_rows": 0,
            }
        ],
        vendor_fields=[{"option_volume_rows": 10, "option_open_interest_rows": 0}],
    )
    packet = Protocol101ExistingCaptureRepairabilityDiagnosticV1(
        headline={"production_contract_change_allowed": False},
        q1_lost_trade_policy=policy,
        capture_timing=[],
        capture_vendor_fields=[],
        repairability_routes=routes,
        conclusions=["use existing captures now"],
        next_actions=["diagnostic rebuild only"],
    )

    assert policy["one_minute_near_trades"] == 1
    assert routes[0]["route"] == "source_policy_timing_and_atm_geometry"
    assert routes[0]["status"] == "offline_diagnostic_rebuild_allowed"
    assert routes[-1]["route"] == "shared_contract_mutation"
    assert routes[-1]["status"] == "blocked"
    assert packet.schema_version == "Protocol101ExistingCaptureRepairabilityDiagnosticV1"
    assert packet.headline["production_contract_change_allowed"] is False


def test_fair_game_policy_accepts_serial_realism_and_blocks_old_edge_chasing() -> None:
    policy = Protocol101FairGamePolicyV1.evaluate(
        old_event_policy={"trades": 249, "total_pnl": 95010.0},
        legacy_serial={
            "trades": 221,
            "total_pnl": 69180.0,
            "return_on_premium": 0.1065,
            "profit_factor": 2.07,
            "max_drawdown_dollars": -9650.0,
        },
        live_contract={
            "trades": 151,
            "total_pnl": 17480.0,
            "return_on_premium": 0.0381,
            "profit_factor": 1.29,
            "max_drawdown_dollars": -11910.0,
        },
        diagnostic_source_policy={
            "trades": 191,
            "total_pnl": 48230.0,
            "return_on_premium": 0.0914,
            "profit_factor": 1.84,
            "max_drawdown_dollars": -10300.0,
        },
        live_non_inferiority={"status": "fail"},
        diagnostic_non_inferiority={"status": "fail"},
    )

    accepted = policy.accepted_degradation[0]
    repairable = policy.repairable_degradation[0]
    diagnostic = policy.repairable_degradation[1]

    assert policy.schema_version == "Protocol101FairGamePolicyV1"
    assert policy.status == "continue_causal_repair_before_retraining"
    assert accepted["classification"] == "accepted_trustworthiness_correction"
    assert accepted["restore_old_result_allowed"] is False
    assert repairable["classification"] == "investigate_for_needless_edge_loss"
    assert diagnostic["classification"] == "promising_but_incomplete"
    assert diagnostic["diagnostic_total_pnl_delta_vs_live"] == 30750.0
    assert diagnostic["production_contract_mutation_allowed"] is False
    assert any("precomputed" in path for path in policy.forbidden_recovery_paths)
    assert "retrain" in policy.retraining_trigger["action"]


def test_fair_game_policy_rejects_completed_minute_same_minute_timing_edge() -> None:
    policy = Protocol101FairGamePolicyV1.evaluate(
        old_event_policy={"trades": 249, "total_pnl": 95010.0},
        legacy_serial={
            "trades": 221,
            "total_pnl": 69180.0,
            "return_on_premium": 0.1065,
            "profit_factor": 2.07,
            "max_drawdown_dollars": -9650.0,
        },
        live_contract={
            "trades": 151,
            "total_pnl": 17480.0,
            "return_on_premium": 0.0381,
            "profit_factor": 1.29,
            "max_drawdown_dollars": -11910.0,
        },
        diagnostic_source_policy={
            "trades": 191,
            "total_pnl": 48230.0,
            "return_on_premium": 0.0914,
            "profit_factor": 1.84,
            "max_drawdown_dollars": -10300.0,
        },
        fair_timing_source_policy={
            "trades": 200,
            "total_pnl": -14230.0,
            "return_on_premium": -0.0232,
            "profit_factor": 0.85,
            "max_drawdown_dollars": -19920.0,
        },
        live_non_inferiority={"status": "fail"},
        diagnostic_non_inferiority={"status": "fail"},
        fair_timing_non_inferiority={"status": "fail"},
    )

    assert policy.status == "reject_same_minute_timing_repair_continue_other_causal_routes"
    timing = policy.repairable_degradation[2]
    assert timing["classification"] == "rejected_same_minute_timing_edge"
    assert timing["fair_timing_total_pnl"] == -14230.0
    assert timing["production_contract_mutation_allowed"] is False
    assert any("completed minute T features" in path for path in policy.forbidden_recovery_paths)


def test_paper_readiness_requires_shadow_and_lifecycle_evidence() -> None:
    gate = Protocol101PaperReadinessGateV2.evaluate(
        complete_shadow_sessions=3,
        entry_intents=9,
        complete_lifecycle_paths=0,
        same_input_exact=True,
        unclassified_nonthreshold_mismatches=None,
        unflattened_positions=1,
    )
    assert gate.status == "blocked"
    assert gate.checks["same_input_exact"]["pass"] is True
    assert gate.checks["complete_shadow_sessions"]["pass"] is False
    assert gate.checks["unclassified_nonthreshold_mismatches"]["value"] == "UNKNOWN"


def test_single_day_synchronization_gate_passes_july2_style_evidence() -> None:
    gate = Protocol101SingleDaySynchronizationGateV1.evaluate(
        session="2026-07-02",
        same_input_exact=True,
        entry_intents=9,
        action_mismatches=0,
        selected_contract_mismatches=0,
        historical_serial_trades=2,
        ibkr_terminal_paths=2,
        lifecycle_exit_time_matches=2,
        lifecycle_reason_matches=2,
        lifecycle_quote_missing_rows=0,
        lifecycle_terminal_stale_quote_rows=0,
        unflattened_positions=0,
    )

    assert gate.status == "pass"
    assert gate.evidence_scope == "single_threshold_crossing_development_day"


def test_single_day_synchronization_gate_blocks_on_unknown_action_mismatch() -> None:
    gate = Protocol101SingleDaySynchronizationGateV1.evaluate(
        session="2026-07-02",
        same_input_exact=True,
        entry_intents=9,
        action_mismatches=None,
        selected_contract_mismatches=0,
        historical_serial_trades=2,
        ibkr_terminal_paths=2,
        lifecycle_exit_time_matches=2,
        lifecycle_reason_matches=2,
        lifecycle_quote_missing_rows=0,
        lifecycle_terminal_stale_quote_rows=0,
        unflattened_positions=0,
    )

    assert gate.status == "blocked"
    assert gate.checks["action_mismatches"]["value"] == "UNKNOWN"


def test_feature_audit_loader_supports_multiple_sessions(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [
        {"session": "2026-01-02", "decision_ts": "2026-01-02T14:31:00+00:00"},
        {"session": "2026-01-05", "decision_ts": "2026-01-05T14:31:00+00:00"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = _load(path, None)

    assert len(loaded) == 2
    assert set(key.split("|", 1)[0] for key in loaded) == {"2026-01-02", "2026-01-05"}


def test_feature_audit_universe_merges_trace_payload_scores_and_features() -> None:
    row = {
        "candidate_universe": [
            {"contract_id": "SPXW-20260701-07510.000-P", "score": 12.5, "entry_ask": 4.1}
        ],
        "features": {
            "token_features": [
                {
                    "contract_id": "SPXW-20260701-07510.000-P",
                    "feature_hash": "abc123",
                    "features": [1.0, 2.0, 3.0],
                }
            ],
        },
        "model_scores": {
            "candidate_scores": [
                {"contract_id": "SPXW-20260701-07510.000-P", "score": 13.25}
            ],
        },
    }

    universe = _universe(row)

    candidate = universe["SPXW-20260701-07510.000-P"]
    assert candidate["entry_ask"] == 4.1
    assert candidate["token_features"] == [1.0, 2.0, 3.0]
    assert candidate["feature_hash"] == "abc123"
    assert candidate["edge"] == 13.25


def test_feature_audit_token_names_match_candidate_feature_vector_order() -> None:
    names = _token_names(71)

    assert len(names) == 71
    assert names[:15] == [
        "bid",
        "ask",
        "mid",
        "spread",
        "spread_frac",
        "bid_size",
        "ask_size",
        "option_ohlcv_volume",
        "stat_open_interest",
        "iv",
        "delta",
        "gamma",
        "theta",
        "distance_points",
        "breakeven_distance",
    ]
    assert names[15:22] == [
        "market_last.spx_close",
        "market_last.vix_close",
        "market_last.spx_vwap",
        "market_last.omar",
        "market_last.session_range",
        "market_last.momentum_5m",
        "market_last.momentum_15m",
    ]
    assert names[43:47] == [
        "side.is_call",
        "side.is_put",
        "shape.offset_norm",
        "shape.abs_offset_norm",
    ]
    assert names[-8:] == [
        "time.session_progress",
        "time.session_progress_remaining",
        "time.session_progress_sin",
        "time.session_progress_cos",
        "time.bucket_first_30",
        "time.bucket_post_open_morning",
        "time.bucket_midday",
        "time.bucket_late_afternoon",
    ]


def test_recorder_requests_volume_and_open_interest_generic_ticks() -> None:
    source = Path("v4/ops/ibkr/run_protocol101_ibkr_recorder.py").read_text()
    assert 'default="100,101"' in source
    assert 'ib.reqMktData(contract, str(args.option_generic_ticks), False, False)' in source
    assert "diagnostic_daily_aggregate_not_databento_minute_equivalent" in source
    assert 'default=90' in source
    assert "dict(option_tickers)" in source
