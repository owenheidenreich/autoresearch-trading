"""Versioned evidence contracts for Protocol101 synchronization resolution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Protocol101BaselineLineageV1:
    stages: list[dict[str, Any]]
    disclosures: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101BaselineLineageV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101FeatureAvailabilityMatrixV1:
    features: list[dict[str, Any]]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101FeatureAvailabilityMatrixV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalMarketMinuteV2:
    session: str
    completed_minute_et: str
    decision_time_et: str
    source: str
    source_quote_time_utc: str | None
    source_context_time_utc: str | None
    raw_input_hash: str
    canonical_feature_hash: str
    candidate_universe_hash: str
    feature_contract_version: str
    opening_context_ready: bool
    missing_opening_minutes: int
    schema_version: str = "CanonicalMarketMinuteV2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101ParityAttributionV2:
    session: str
    decision_time: str
    action_match: bool
    threshold_adjacent: bool
    categories: list[str]
    candidate_overlap: float | None
    live_max_edge: float | None
    historical_max_edge: float | None
    edge_delta: float | None
    selected_contract_match: bool | None
    explanation: str
    schema_version: str = "Protocol101ParityAttributionV2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101DataPlaneSelectionV1:
    status: str
    selected_data_plane: str | None
    reason: str
    enriched_ibkr: dict[str, Any]
    same_vendor_model_feed: dict[str, Any]
    retraining_required: bool
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101DataPlaneSelectionV1"

    @classmethod
    def evaluate(
        cls,
        *,
        enriched_ibkr: dict[str, Any],
        same_vendor_model_feed: dict[str, Any],
    ) -> "Protocol101DataPlaneSelectionV1":
        if enriched_ibkr.get("parity_pass") and enriched_ibkr.get("historical_non_inferiority_pass"):
            return cls(
                status="selected",
                selected_data_plane="enriched_ibkr",
                reason="Enriched IBKR passed parity and historical non-inferiority.",
                enriched_ibkr=enriched_ibkr,
                same_vendor_model_feed=same_vendor_model_feed,
                retraining_required=False,
            )
        if same_vendor_model_feed.get("parity_pass") and same_vendor_model_feed.get(
            "historical_non_inferiority_pass"
        ):
            return cls(
                status="selected",
                selected_data_plane="databento_thetadata_same_vendor",
                reason="Same-vendor live/history feed passed parity and historical non-inferiority.",
                enriched_ibkr=enriched_ibkr,
                same_vendor_model_feed=same_vendor_model_feed,
                retraining_required=False,
            )
        both_complete = all(
            item.get("evaluation_complete")
            for item in (enriched_ibkr, same_vendor_model_feed)
        )
        return cls(
            status="retrain_required" if both_complete else "evidence_pending",
            selected_data_plane=None,
            reason=(
                "Neither causal data plane preserved viable frozen-model behavior."
                if both_complete
                else "Feature attribution and prospective data-plane evidence are incomplete."
            ),
            enriched_ibkr=enriched_ibkr,
            same_vendor_model_feed=same_vendor_model_feed,
            retraining_required=both_complete,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101DataPlaneDecisionMapV1:
    """Offline hypothesis map for resolving frozen-model data-plane viability."""

    headline: dict[str, Any]
    feature_families: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    decisions: list[str]
    blockers: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101DataPlaneDecisionMapV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101CanonicalSemanticsAuditV1:
    """Offline H1 audit for causal feature-contract/data-policy repair routes."""

    headline: dict[str, Any]
    repair_routes: list[dict[str, Any]]
    field_routes: list[dict[str, Any]]
    conclusions: list[str]
    next_actions: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101CanonicalSemanticsAuditV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101RepairabilityDecisionV1:
    """Governance decision for whether to mutate the canonical contract now."""

    status: str
    decision: str
    immediate_contract_change_allowed: bool
    q1_rerun_required_now: bool
    route_decisions: list[dict[str, Any]]
    next_actions: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101RepairabilityDecisionV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101ExistingCaptureRepairabilityDiagnosticV1:
    """Offline diagnostic packet from existing recorder captures and Q1 attribution."""

    headline: dict[str, Any]
    q1_lost_trade_policy: dict[str, Any]
    capture_timing: list[dict[str, Any]]
    capture_vendor_fields: list[dict[str, Any]]
    repairability_routes: list[dict[str, Any]]
    conclusions: list[str]
    next_actions: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101ExistingCaptureRepairabilityDiagnosticV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101FairContractTrainingPreflightV1:
    """Offline preflight for preparing retraining on the accepted fair contract."""

    status: str
    selected_feature_contract: str
    model_training_authorized: bool
    paper_submit_allowed: bool
    decision: str
    dataset_checks: dict[str, Any]
    synchronization_checks: dict[str, Any]
    blockers: list[str]
    next_actions: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101FairContractTrainingPreflightV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101FairContractTrainingDesignV1:
    """Owner-approval packet for a future fair-contract retraining run."""

    status: str
    decision: str
    selected_feature_contract: str
    model_training_authorized: bool
    threshold_tuning_authorized: bool
    paper_submit_allowed: bool
    allowed_data: dict[str, Any]
    excluded_data: list[dict[str, Any]]
    split_policy: dict[str, Any]
    training_target: dict[str, Any]
    success_gates: list[dict[str, Any]]
    forbidden_actions: list[str]
    required_owner_approvals: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101FairContractTrainingDesignV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101FairContractTrainingDryRunV1:
    """Manifest-based ingestion dry run for fair-contract model training."""

    status: str
    decision: str
    selected_feature_contract: str
    model_training_executed: bool
    threshold_tuning_executed: bool
    broker_endpoint_called: bool
    split_summary: dict[str, Any]
    feature_summary: dict[str, Any]
    label_summary: dict[str, Any]
    blockers: list[str]
    next_actions: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101FairContractTrainingDryRunV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101FairContractCandidateValidationGateV1:
    """Gate a future fair-contract candidate before any paper-readiness claim."""

    status: str
    decision: str
    selected_feature_contract: str
    model_training_executed: bool
    threshold_selection_executed: bool
    broker_endpoint_called: bool
    paper_submit_allowed: bool
    checks: dict[str, dict[str, Any]]
    metrics: dict[str, Any]
    next_actions: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101FairContractCandidateValidationGateV1"

    @classmethod
    def evaluate(
        cls,
        *,
        runner_plan: dict[str, Any] | None,
        training_result: dict[str, Any] | None,
        selected_feature_contract: str = FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
        min_validation_trades: int = 20,
        min_diagnostic_trades: int = 20,
        min_profit_factor: float = 1.25,
    ) -> "Protocol101FairContractCandidateValidationGateV1":
        if training_result is None:
            return cls(
                status="waiting_for_owner_approved_training_result",
                decision="run_no_training_yet_keep_paper_submit_disabled",
                selected_feature_contract=selected_feature_contract,
                model_training_executed=False,
                threshold_selection_executed=False,
                broker_endpoint_called=False,
                paper_submit_allowed=False,
                checks={
                    "training_result_present": {
                        "value": False,
                        "required": True,
                        "pass": False,
                    }
                },
                metrics={},
                next_actions=[
                    "Do not paper-submit from a dry-run runner plan.",
                    "If owner-approved, train a candidate on the fair-contract manifest, then rerun this gate.",
                    "After this gate passes, run strict serial/lifecycle replay and recorder-shadow validation before paper-submit.",
                ],
            )

        plan = runner_plan or {}
        validation = ((training_result.get("neural") or {}).get("validation") or {}).get("metrics") or {}
        diagnostic = (
            ((training_result.get("neural") or {}).get("diagnostic_test") or {}).get("metrics")
            or {}
        )
        actual_contract = str(plan.get("selected_feature_contract") or selected_feature_contract)
        values = {
            "feature_contract": (
                actual_contract,
                selected_feature_contract,
                actual_contract == selected_feature_contract,
            ),
            "runner_mode_train": (plan.get("mode"), "train", plan.get("mode") == "train"),
            "model_training_executed": (
                bool(plan.get("model_training_executed")),
                True,
                bool(plan.get("model_training_executed")),
            ),
            "threshold_selection_executed": (
                bool(plan.get("threshold_selection_executed")),
                True,
                bool(plan.get("threshold_selection_executed")),
            ),
            "broker_endpoint_called": (
                bool(plan.get("broker_endpoint_called")),
                False,
                not bool(plan.get("broker_endpoint_called")),
            ),
            "paper_submit_allowed": (
                bool(plan.get("paper_submit_allowed")),
                False,
                not bool(plan.get("paper_submit_allowed")),
            ),
            "validation_trades": (
                int(validation.get("trades") or 0),
                min_validation_trades,
                int(validation.get("trades") or 0) >= min_validation_trades,
            ),
            "validation_positive_pnl": (
                float(validation.get("total_pnl") or 0.0),
                ">0",
                float(validation.get("total_pnl") or 0.0) > 0.0,
            ),
            "validation_profit_factor": (
                float(validation.get("profit_factor") or 0.0),
                min_profit_factor,
                float(validation.get("profit_factor") or 0.0) >= min_profit_factor,
            ),
            "diagnostic_trades": (
                int(diagnostic.get("trades") or 0),
                min_diagnostic_trades,
                int(diagnostic.get("trades") or 0) >= min_diagnostic_trades,
            ),
            "diagnostic_positive_pnl": (
                float(diagnostic.get("total_pnl") or 0.0),
                ">0",
                float(diagnostic.get("total_pnl") or 0.0) > 0.0,
            ),
            "diagnostic_profit_factor": (
                float(diagnostic.get("profit_factor") or 0.0),
                min_profit_factor,
                float(diagnostic.get("profit_factor") or 0.0) >= min_profit_factor,
            ),
        }
        checks = {
            name: {"value": value, "required": required, "pass": passed}
            for name, (value, required, passed) in values.items()
        }
        passed = all(item["pass"] for item in checks.values())
        return cls(
            status="pass" if passed else "fail",
            decision=(
                "candidate_can_proceed_to_strict_serial_lifecycle_replay"
                if passed
                else "candidate_not_ready_for_strict_serial_lifecycle_replay"
            ),
            selected_feature_contract=selected_feature_contract,
            model_training_executed=bool(plan.get("model_training_executed")),
            threshold_selection_executed=bool(plan.get("threshold_selection_executed")),
            broker_endpoint_called=bool(plan.get("broker_endpoint_called")),
            paper_submit_allowed=bool(plan.get("paper_submit_allowed")),
            checks=checks,
            metrics={
                "validation": validation,
                "diagnostic_test": diagnostic,
                "chosen_threshold": training_result.get("chosen_threshold"),
                "model_out": training_result.get("model_out"),
            },
            next_actions=[
                "If this gate passes, run strict serial/lifecycle replay on the same causal contract.",
                "If strict replay passes, run recorder-shadow same-input replay before any paper-submit request.",
                "If this gate fails, revise the training hypothesis instead of tuning against paper/live diagnostics.",
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101FairGamePolicyV1:
    """Governance policy for preserving only live-plausible Protocol101 edge."""

    status: str
    objective: str
    non_negotiable_laws: list[str]
    accepted_degradation: list[dict[str, Any]]
    repairable_degradation: list[dict[str, Any]]
    forbidden_recovery_paths: list[str]
    retraining_trigger: dict[str, Any]
    current_evidence: dict[str, Any]
    next_actions: list[str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101FairGamePolicyV1"

    @classmethod
    def evaluate(
        cls,
        *,
        old_event_policy: dict[str, Any],
        legacy_serial: dict[str, Any],
        live_contract: dict[str, Any],
        diagnostic_source_policy: dict[str, Any] | None,
        fair_timing_source_policy: dict[str, Any] | None = None,
        live_non_inferiority: dict[str, Any],
        diagnostic_non_inferiority: dict[str, Any] | None,
        fair_timing_non_inferiority: dict[str, Any] | None = None,
    ) -> "Protocol101FairGamePolicyV1":
        legacy_pnl = float(legacy_serial.get("total_pnl") or 0.0)
        live_pnl = float(live_contract.get("total_pnl") or 0.0)
        diagnostic_pnl = (
            float(diagnostic_source_policy.get("total_pnl") or 0.0)
            if diagnostic_source_policy
            else None
        )
        fair_timing_pnl = (
            float(fair_timing_source_policy.get("total_pnl") or 0.0)
            if fair_timing_source_policy
            else None
        )
        diagnostic_delta = (
            diagnostic_pnl - live_pnl
            if diagnostic_pnl is not None
            else None
        )
        fair_timing_delta = (
            fair_timing_pnl - live_pnl
            if fair_timing_pnl is not None
            else None
        )
        repair_status = (
            "promising_but_incomplete"
            if diagnostic_non_inferiority and diagnostic_non_inferiority.get("status") == "fail"
            else "not_run"
            if diagnostic_non_inferiority is None
            else "passes_current_gate"
        )
        fair_timing_status = (
            "not_run"
            if fair_timing_non_inferiority is None
            else "rejected_same_minute_timing_edge"
            if fair_timing_pnl is not None and fair_timing_pnl <= 0.0
            else "fails_current_gate"
            if fair_timing_non_inferiority.get("status") == "fail"
            else "passes_current_gate"
        )
        if fair_timing_status == "rejected_same_minute_timing_edge":
            status = "reject_same_minute_timing_repair_continue_other_causal_routes"
        elif repair_status == "promising_but_incomplete":
            status = "continue_causal_repair_before_retraining"
        else:
            status = "evaluate_next_repair_or_retrain"
        return cls(
            status=status,
            objective=(
                "Build a realistic training and validation game that Protocol101 can also "
                "play live, preserving causal edge while removing simulation privilege."
            ),
            non_negotiable_laws=[
                "Keep strict one-account serial replay; do not return to overlapping or precomputed-path headline metrics.",
                "Keep live-reproducibility as the law for model-facing features.",
                "Repair only feature semantics that can be proven causal and available before the decision.",
                "Accept lost edge when it depended on unavailable, stale, retrospective, or non-causal information.",
                "If the frozen model cannot pass under the final fair contract, retrain on that contract instead of tuning the old one around the problem.",
            ],
            accepted_degradation=[
                {
                    "transition": "older_event_policy_to_same_runner_legacy_serial",
                    "classification": "accepted_trustworthiness_correction",
                    "reason": (
                        "Strict serial replay removes simulation privilege: one account, "
                        "one open position, ask-entry, bid-exit, affordability, and flat-by-close."
                    ),
                    "old_trades": old_event_policy.get("trades"),
                    "old_total_pnl": old_event_policy.get("total_pnl"),
                    "serial_trades": legacy_serial.get("trades"),
                    "serial_total_pnl": legacy_serial.get("total_pnl"),
                    "restore_old_result_allowed": False,
                }
            ],
            repairable_degradation=[
                {
                    "transition": "same_runner_legacy_serial_to_protocol101_live_v1",
                    "classification": "investigate_for_needless_edge_loss",
                    "reason": (
                        "This stage changes model-facing feature semantics. Losses are acceptable "
                        "only after each missing or changed signal is classified as non-causal, "
                        "unavailable live, or too vendor-sensitive to trust."
                    ),
                    "legacy_trades": legacy_serial.get("trades"),
                    "legacy_total_pnl": legacy_serial.get("total_pnl"),
                    "live_trades": live_contract.get("trades"),
                    "live_total_pnl": live_contract.get("total_pnl"),
                    "live_non_inferiority_status": live_non_inferiority.get("status"),
                },
                {
                    "transition": "protocol101_live_v1_to_diagnostic_context_lag0",
                    "classification": repair_status,
                    "reason": (
                        "The diagnostic source-policy rebuild recovered material Q1 behavior "
                        "without changing weights or thresholds, but it used completed-minute "
                        "features without proving that the model could enter after that minute "
                        "under live-causal execution timing."
                    ),
                    "diagnostic_total_pnl_delta_vs_live": diagnostic_delta,
                    "diagnostic_non_inferiority_status": (
                        diagnostic_non_inferiority.get("status")
                        if diagnostic_non_inferiority
                        else "not_run"
                    ),
                    "production_contract_mutation_allowed": False,
                },
                {
                    "transition": "diagnostic_context_lag0_to_fair_decision_plus1",
                    "classification": fair_timing_status,
                    "reason": (
                        "A live-plausible completed-minute contract can observe minute T only "
                        "after it is complete and then act on minute T+1. The fair timing replay "
                        "tests exactly that. If it loses money, the recovered same-minute edge "
                        "is treated as timing privilege, not a production repair."
                    ),
                    "fair_timing_trades": (
                        fair_timing_source_policy.get("trades")
                        if fair_timing_source_policy
                        else None
                    ),
                    "fair_timing_total_pnl": fair_timing_pnl,
                    "fair_timing_total_pnl_delta_vs_live": fair_timing_delta,
                    "fair_timing_non_inferiority_status": (
                        fair_timing_non_inferiority.get("status")
                        if fair_timing_non_inferiority
                        else "not_run"
                    ),
                    "production_contract_mutation_allowed": False,
                },
            ],
            forbidden_recovery_paths=[
                "Do not restore precomputed candidate_pnl, candidate_exit_time, or best-exit values into runtime features.",
                "Do not use overlapping independent-entry totals as a promotion benchmark.",
                "Do not use completed minute T features to enter during minute T unless an intra-minute causal feed proves those features were available before the order decision.",
                "Do not restore volume/open-interest fields unless their live source and timestamp semantics are proven causal.",
                "Do not tune thresholds or model weights to exposed parity diagnostics.",
                "Do not change PAPER_TRADING_DEFAULT or enable paper-submit from this diagnostic evidence alone.",
            ],
            retraining_trigger={
                "condition": (
                    "After causal repair routes and data-plane choices are evaluated, the selected "
                    "fair contract still fails historical non-inferiority or cannot be reproduced live."
                ),
                "action": "freeze the canonical contract and retrain a replacement candidate on that same game",
                "current_state": "not_triggered_yet_pending_causal_repair_tests",
            },
            current_evidence={
                "same_runner_legacy_total_pnl": legacy_pnl,
                "protocol101_live_v1_total_pnl": live_pnl,
                "diagnostic_source_policy_total_pnl": diagnostic_pnl,
                "fair_timing_source_policy_total_pnl": fair_timing_pnl,
                "live_non_inferiority_status": live_non_inferiority.get("status"),
                "diagnostic_non_inferiority_status": (
                    diagnostic_non_inferiority.get("status")
                    if diagnostic_non_inferiority
                    else "not_run"
                ),
                "fair_timing_non_inferiority_status": (
                    fair_timing_non_inferiority.get("status")
                    if fair_timing_non_inferiority
                    else "not_run"
                ),
            },
            next_actions=[
                "Keep same-runner legacy as the provisional strict benchmark, not the old event-policy headline.",
                "Reject completed-minute same-minute entry as a production repair unless it is replaced by a truly intra-minute causal model/feed contract.",
                "Continue feature-family attribution for pattern/context, candidate geometry, and pressure fields.",
                "Use July 6+ recorder days as confirmation evidence, not as a reason to pause offline diagnosis.",
                "Declare retraining required if causal repairs cannot recover a viable fair contract.",
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101PaperReadinessGateV2:
    status: str
    checks: dict[str, dict[str, Any]]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101PaperReadinessGateV2"

    @classmethod
    def evaluate(
        cls,
        *,
        complete_shadow_sessions: int,
        entry_intents: int,
        complete_lifecycle_paths: int,
        same_input_exact: bool,
        unclassified_nonthreshold_mismatches: int | None,
        unflattened_positions: int,
    ) -> "Protocol101PaperReadinessGateV2":
        mismatch_value: int | str = (
            int(unclassified_nonthreshold_mismatches)
            if unclassified_nonthreshold_mismatches is not None
            else "UNKNOWN"
        )
        values = {
            "complete_shadow_sessions": (complete_shadow_sessions, 5, complete_shadow_sessions >= 5),
            "entry_intents": (entry_intents, 20, entry_intents >= 20),
            "complete_lifecycle_paths": (
                complete_lifecycle_paths,
                10,
                complete_lifecycle_paths >= 10,
            ),
            "same_input_exact": (same_input_exact, True, bool(same_input_exact)),
            "unclassified_nonthreshold_mismatches": (
                mismatch_value,
                0,
                unclassified_nonthreshold_mismatches is not None
                and unclassified_nonthreshold_mismatches == 0,
            ),
            "unflattened_positions": (unflattened_positions, 0, unflattened_positions == 0),
        }
        checks = {
            name: {"value": value, "required": required, "pass": passed}
            for name, (value, required, passed) in values.items()
        }
        return cls(
            status="pass" if all(item["pass"] for item in checks.values()) else "blocked",
            checks=checks,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Protocol101SingleDaySynchronizationGateV1:
    status: str
    session: str
    checks: dict[str, dict[str, Any]]
    evidence_scope: str
    confidence: str
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = "Protocol101SingleDaySynchronizationGateV1"

    @classmethod
    def evaluate(
        cls,
        *,
        session: str,
        same_input_exact: bool,
        entry_intents: int,
        action_mismatches: int | None,
        selected_contract_mismatches: int | None,
        historical_serial_trades: int,
        ibkr_terminal_paths: int,
        lifecycle_exit_time_matches: int,
        lifecycle_reason_matches: int,
        lifecycle_quote_missing_rows: int,
        lifecycle_terminal_stale_quote_rows: int,
        unflattened_positions: int,
    ) -> "Protocol101SingleDaySynchronizationGateV1":
        def known_zero(value: int | None) -> bool:
            return value is not None and int(value) == 0

        checks = {
            "same_input_exact": {
                "value": bool(same_input_exact),
                "required": True,
                "pass": bool(same_input_exact),
            },
            "threshold_crossing_entries": {
                "value": int(entry_intents),
                "required": ">=1",
                "pass": int(entry_intents) >= 1,
            },
            "action_mismatches": {
                "value": "UNKNOWN" if action_mismatches is None else int(action_mismatches),
                "required": 0,
                "pass": known_zero(action_mismatches),
            },
            "selected_contract_mismatches": {
                "value": "UNKNOWN"
                if selected_contract_mismatches is None
                else int(selected_contract_mismatches),
                "required": 0,
                "pass": known_zero(selected_contract_mismatches),
            },
            "terminal_lifecycle_paths": {
                "value": int(ibkr_terminal_paths),
                "required": int(historical_serial_trades),
                "pass": int(ibkr_terminal_paths) >= int(historical_serial_trades) > 0,
            },
            "lifecycle_exit_time_matches": {
                "value": int(lifecycle_exit_time_matches),
                "required": int(historical_serial_trades),
                "pass": int(lifecycle_exit_time_matches) >= int(historical_serial_trades) > 0,
            },
            "lifecycle_reason_matches": {
                "value": int(lifecycle_reason_matches),
                "required": int(historical_serial_trades),
                "pass": int(lifecycle_reason_matches) >= int(historical_serial_trades) > 0,
            },
            "lifecycle_quote_missing_rows": {
                "value": int(lifecycle_quote_missing_rows),
                "required": 0,
                "pass": int(lifecycle_quote_missing_rows) == 0,
            },
            "lifecycle_terminal_stale_quote_rows": {
                "value": int(lifecycle_terminal_stale_quote_rows),
                "required": 0,
                "pass": int(lifecycle_terminal_stale_quote_rows) == 0,
            },
            "unflattened_positions": {
                "value": int(unflattened_positions),
                "required": 0,
                "pass": int(unflattened_positions) == 0,
            },
        }
        return cls(
            status="pass" if all(item["pass"] for item in checks.values()) else "blocked",
            session=session,
            checks=checks,
            evidence_scope="single_threshold_crossing_development_day",
            confidence="development_evidence_not_statistical_proof",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_historical_non_inferiority(
    *,
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    adverse_fill_total_pnl: float | None = None,
) -> dict[str, Any]:
    """Apply the frozen Phase 6 hard gates without tuning any threshold."""

    legacy_trades = max(float(legacy.get("trades") or 0.0), 1.0)
    legacy_pnl = max(float(legacy.get("total_pnl") or 0.0), 1e-12)
    legacy_rop = max(float(legacy.get("return_on_premium") or 0.0), 1e-12)
    legacy_drawdown = abs(float(legacy.get("max_drawdown_dollars") or 0.0))
    candidate_drawdown = abs(float(candidate.get("max_drawdown_dollars") or 0.0))
    checks = {
        "trade_count_ge_75pct": float(candidate.get("trades") or 0.0) / legacy_trades >= 0.75,
        "pnl_ge_80pct": float(candidate.get("total_pnl") or 0.0) / legacy_pnl >= 0.80,
        "return_on_premium_ge_80pct": float(candidate.get("return_on_premium") or 0.0)
        / legacy_rop
        >= 0.80,
        "profit_factor_ge_1_75": float(candidate.get("profit_factor") or 0.0) >= 1.75,
        "drawdown_le_110pct": candidate_drawdown <= legacy_drawdown * 1.10,
        "positive_after_adverse_fill": adverse_fill_total_pnl is not None
        and float(adverse_fill_total_pnl) > 0.0,
        "top_1_day_share_le_25pct": float(candidate.get("top_1_day_pnl_share") or 1.0) <= 0.25,
        "top_5_day_share_le_75pct": float(candidate.get("top_5_day_pnl_share") or 1.0) <= 0.75,
        "top_10_trade_share_le_75pct": float(candidate.get("top_10_trade_pnl_share") or 1.0) <= 0.75,
    }
    return {
        "schema_version": "Protocol101HistoricalNonInferiorityGateV1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "ratios": {
            "trade_count": float(candidate.get("trades") or 0.0) / legacy_trades,
            "total_pnl": float(candidate.get("total_pnl") or 0.0) / legacy_pnl,
            "return_on_premium": float(candidate.get("return_on_premium") or 0.0) / legacy_rop,
            "drawdown": candidate_drawdown / max(legacy_drawdown, 1e-12),
        },
        "adverse_fill_total_pnl": adverse_fill_total_pnl,
    }
