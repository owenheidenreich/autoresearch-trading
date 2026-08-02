"""Immutable Path-D entry-machinery gate frozen before any model fitting.

This file intentionally imports future implementation modules only inside tests.  It is
hashed into the preregistration before those modules exist, and the exact ten-test JUnit
result is required before an entry fit authorization can be issued.
"""
from __future__ import annotations

import copy
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import io
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_deterministic_debug_mode(2)

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES, feature_matrix
from v4.path_d.contracts import (
    ContractIdentityV1,
    DecisionDirectiveV1,
    ExecutionIntentV1,
    GovernorDecisionV1,
    IntentClocksV1,
    PositionPreconditionV1,
    PositionV1,
    PriceBudgetV1,
    ProducerIdentityV1,
)
from v4.path_d.execution.simulated import (
    ExecutionScenario,
    SimulatedExecutor,
    SimulatedQuote,
    VirtualMonotonicClock,
)
from v4.path_d.risk.governor import (
    DeterministicGovernor,
    FeedHealthV1,
    GovernorConfigV1,
    LifecycleStateV1,
    fake_broker_state,
)
from v4.research import pathd_entry_exit as prereg


def _resolve(qualified_name: str) -> Any:
    parts = qualified_name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for name in parts[split:]:
            value = getattr(value, name)
        return value
    raise ModuleNotFoundError(qualified_name)


def _signature_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameter in inspect.signature(value).parameters.values():
        row: dict[str, Any] = {
            "name": parameter.name,
            "kind": parameter.kind.name,
            "required": parameter.default is inspect.Parameter.empty,
        }
        if parameter.default is not inspect.Parameter.empty:
            row["default"] = parameter.default
        rows.append(row)
    return rows


def _causal_inputs() -> dict[str, Any]:
    offsets = np.arange(-50.0, 55.0, 5.0, dtype=np.float64)
    rights = ("C", "P")
    ladder = np.empty((21, 2, 3), dtype=np.float64)
    for strike_index in range(21):
        call_mid = 8.0 + strike_index * 0.05
        put_mid = 9.0 - strike_index * 0.04
        ladder[strike_index, 0, :] = (call_mid - 0.05, call_mid + 0.05, call_mid)
        ladder[strike_index, 1, :] = (put_mid - 0.05, put_mid + 0.05, put_mid)
    market_names = (
        "spx_close",
        "spx_vwap",
        "omar",
        "session_range",
        "momentum_5m",
        "momentum_15m",
    )
    market = np.asarray(
        [[6000.0, 5998.0, 0.2, 30.0, 1.0, 2.0], [6002.0, 5999.0, 0.3, 32.0, 1.5, 3.0]],
        dtype=np.float64,
    )
    decision = pd.Timestamp("2026-01-02T15:00:00Z").value
    identities = np.asarray(
        [
            [
                f"SPXW-20260102-{6000.0 + offset:08.3f}-{right}"
                for right in rights
            ]
            for offset in offsets
        ],
        dtype=object,
    )
    watermarks = np.full((21, 2), decision - 1_000_000_000, dtype=np.int64)
    return {
        "session": "2026-01-02",
        "decision_time_ns": int(decision),
        "decision_watermark_ns": int(decision),
        "atm_strike": 6000.0,
        "strike_offsets": offsets,
        "rights": rights,
        "option_ladder": ladder,
        "option_feature_names": ("bid", "ask", "mid"),
        "market_window": market,
        "market_feature_names": market_names,
        "contract_ids": identities,
        "contract_quote_watermark_ns": watermarks,
        "declared_alpha_sources": tuple(prereg.feature_lineage()["features"]),
    }


def _processed_row() -> dict[str, Any]:
    values = _causal_inputs()
    return {
        **values,
        "decision_time": pd.Timestamp(values["decision_time_ns"], unit="ns", tz="UTC"),
        "feature_names": values["option_feature_names"],
        "post_decision_quotes": {"ignored": 1},
        "stored_label": 123.0,
        "future_exit_time_ns": values["decision_time_ns"] + 60_000_000_000,
    }


def _valid_future_path(
    dataset_module: Any,
    snapshot: Any,
    *,
    nonce: str = "valid-future-path",
    arrival_bid_micros: int = 2_000_000,
    arrival_ask_micros: int = 2_050_000,
) -> Any:
    minute_ns = 60_000_000_000
    arrival = snapshot.decision_time_ns + minute_ns
    terminal = pd.Timestamp(
        f"{snapshot.session} 15:55:00", tz="America/New_York"
    ).tz_convert("UTC").value
    remaining_count = (int(terminal) - arrival) // minute_ns
    assert remaining_count == 354
    marks = {
        "h3": [2_100_000] * 3,
        "h5": [2_100_000] * 5,
        "h10": [2_100_000] * 10,
        "h20": [2_100_000] * 20,
        "h45": [2_100_000] * 45,
        "h90": [2_100_000] * 90,
        "remaining_session": [2_100_000] * remaining_count,
    }
    return dataset_module.EntryFuturePathV1(
        schema_version=dataset_module.EntryFuturePathV1.SCHEMA_VERSION,
        arrival_time_ns=arrival,
        arrival_contract_id=str(snapshot.contract_ids.reshape(-1)[0]),
        arrival_bid_micros=arrival_bid_micros,
        arrival_ask_micros=arrival_ask_micros,
        arrival_watermark_ns=arrival,
        marks_by_horizon=marks,
        future_audit_nonce=nonce,
    )


def _normal_intent(*, side: str, hard_limit_micros: int) -> ExecutionIntentV1:
    sell = side == "SELL"
    reference_bid = 2_000_000
    reference_ask = 2_100_000
    if sell:
        hard_limit = min(hard_limit_micros, reference_bid)
    else:
        hard_limit = max(hard_limit_micros, reference_ask)
    contract = ContractIdentityV1(
        osi_symbol="SPXW  260102C06000000",
        underlying="SPX",
        trading_class="SPXW",
        expiry="2026-01-02",
        strike_milli=6_000_000,
        right="C",
    )
    clocks = IntentClocksV1(
        decision_clock="received_timestamp_utc",
        event_interval_end_utc="2026-01-02T15:00:00Z",
        option_received_watermark_utc="2026-01-02T15:00:00Z",
        spx_received_watermark_utc="2026-01-02T15:00:00Z",
        decision_available_at_utc="2026-01-02T15:00:00Z",
        model_started_at_utc="2026-01-02T15:00:00Z",
        model_finished_at_utc="2026-01-02T15:00:00Z",
        intent_emitted_at_utc="2026-01-02T15:00:00Z",
        valid_until_utc="2026-01-02T15:01:01Z",
    )
    return ExecutionIntentV1.create(
        trace_id="frozen-gate",
        parent_intent_id=None,
        origin="DETERMINISTIC_EXIT" if sell else "MODEL",
        producer=ProducerIdentityV1(
            component_version="pathd.frozen-gate.v1",
            strategy_id="pathd.frozen-gate.v1",
            artifact_sha256="sha256:" + "1" * 64,
            feature_contract_version="pathd.entry-signed17.v1",
            feature_snapshot_sha256="sha256:" + "2" * 64,
        ),
        decision=DecisionDirectiveV1(
            action="CLOSE_LONG" if sell else "OPEN_LONG",
            side=side,
            position_effect="CLOSE" if sell else "OPEN",
            urgency="NORMAL_EXIT" if sell else "NORMAL_ENTRY",
            quantity=1,
            reason_code="FROZEN_GATE",
        ),
        contract=contract,
        price_budget=PriceBudgetV1(
            unit="USD_OPTION_PRICE_MICROS",
            reference_vendor="DATABENTO_OPRA",
            reference_bid_micros=reference_bid,
            reference_ask_micros=reference_ask,
            max_adverse_move_micros=abs(hard_limit - (reference_bid if sell else reference_ask)),
            hard_limit_micros=hard_limit,
        ),
        clocks=clocks,
        state_precondition=PositionPreconditionV1(
            expected_position="LONG_ONE" if sell else "FLAT",
            position_snapshot_version="frozen-gate-state",
            held_osi_symbol=contract.osi_symbol if sell else None,
        ),
        execution_profile_version="pathd.governed-limit-profile.v1",
    )


def _authorization(intent: ExecutionIntentV1, config: GovernorConfigV1 | None = None):
    positions = ()
    if intent.decision.position_effect == "CLOSE":
        positions = (PositionV1(intent.contract.osi_symbol, 1, 2_100_000, "2026-01-02T14:59:00Z"),)
    broker = fake_broker_state(
        captured_at_utc="2026-01-02T15:00:00Z",
        snapshot_version="frozen-gate-state",
        positions=positions,
    )
    feed = FeedHealthV1("2026-01-02T15:00:00Z", "2026-01-02T15:00:00Z")
    governor = DeterministicGovernor(config)
    return broker, feed, governor.evaluate(intent, broker_state=broker, feed_health=feed, now_utc="2026-01-02T15:00:00Z")


def _foundation_executor(
    outcome: str,
    *,
    quantity: int = 1,
) -> tuple[SimulatedExecutor, ExecutionIntentV1, Any, Any]:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "path_d/contracts/fixtures/execution_intent_v1.json"
    )
    intent = ExecutionIntentV1.from_dict(
        json.loads(fixture.read_text(encoding="utf-8"))
    )
    if quantity != 1:
        intent = ExecutionIntentV1.create(
            trace_id=intent.trace_id,
            parent_intent_id=intent.parent_intent_id,
            origin=intent.origin,
            producer=intent.producer,
            decision=DecisionDirectiveV1(
                **{**intent.decision.to_dict(), "quantity": quantity}
            ),
            contract=intent.contract,
            price_budget=intent.price_budget,
            clocks=intent.clocks,
            state_precondition=intent.state_precondition,
            execution_profile_version=intent.execution_profile_version,
        )
    broker = fake_broker_state(
        captured_at_utc="2026-06-30T13:30:02Z",
        positions=(
            PositionV1(
                intent.contract.osi_symbol,
                1,
                2_500_000,
                "2026-06-30T13:30:00Z",
            ),
        ),
    )
    feed = FeedHealthV1("2026-06-30T13:30:02Z", "2026-06-30T13:30:01Z")
    authorization = DeterministicGovernor(
        GovernorConfigV1(max_quantity=max(1, quantity))
    ).evaluate(
        intent,
        broker_state=broker,
        feed_health=feed,
        now_utc="2026-06-30T13:30:02Z",
    )
    assert authorization.disposition == "ALLOW"
    executor = SimulatedExecutor(
        clock=VirtualMonotonicClock(
            datetime(2026, 6, 30, 13, 30, 2, tzinfo=timezone.utc)
        ),
        quote_tape=(
            SimulatedQuote(0, 2_000_000, 2_100_000),
            SimulatedQuote(120, 1_950_000, 2_050_000),
        ),
        scenario=ExecutionScenario(outcome=outcome, submit_latency_ms=100),
    )
    return executor, intent, broker, authorization


def _transcript_sha256(events: Any) -> str:
    return hashlib.sha256(
        "".join(event.to_json() + "\n" for event in events).encode("utf-8")
    ).hexdigest()


def _dummy_authorization(session: str = "2026-01-02") -> prereg.FrozenFitAuthorization:
    return prereg.FrozenFitAuthorization(
        role="outer_weights",
        outer_fold=1,
        inner_fold=None,
        sessions=(session,),
        sessions_sha256_newline=prereg.canonical_session_hash((session,)),
        preregistration_sha256="1" * 64,
        session_assignments_sha256="2" * 64,
        source_hash_policy_sha256="3" * 64,
        corpus_integrity_receipt_sha256="4" * 64,
        lineage_receipt_sha256="5" * 64,
        machinery_receipt_sha256="6" * 64,
        fit_environment_sha256="7" * 64,
        entry_pooled_acceptance_receipt_sha256=None,
    )


def _dummy_evidence_authorization(
    session: str = "2026-01-02",
) -> prereg.FrozenEvidenceAuthorization:
    return prereg.FrozenEvidenceAuthorization(
        role="outer_test_primary",
        outer_fold=1,
        inner_fold=None,
        sessions=(session,),
        sessions_sha256_newline=prereg.canonical_session_hash((session,)),
        preregistration_sha256="1" * 64,
        session_assignments_sha256="2" * 64,
        source_hash_policy_sha256="3" * 64,
        corpus_integrity_receipt_sha256="4" * 64,
        lineage_receipt_sha256="5" * 64,
        machinery_receipt_sha256="6" * 64,
        fit_environment_sha256="7" * 64,
        open_gate_receipts_sha256=("8" * 64,),
        access_scope_id="outer_1_primary",
        access_receipt_path=(
            "v4/audit/autoresearch/protocol101_pathd_entry_exit_model/"
            "outer_fold_1/outer_primary_access_receipt.json"
        ),
        access_receipt_sha256="9" * 64,
        transaction_id="frozen-gate-transaction",
    )


def _sealed_dataset_for_authorization(
    dataset_module: Any,
    authorization: prereg.FrozenFitAuthorization | prereg.FrozenEvidenceAuthorization,
    example: Any,
) -> Any:
    source_files = [
        {
            "relative_path": f"processed/{authorization.sessions[0]}.parquet",
            "bytes": 123,
            "sha256": "d" * 64,
        },
        {
            "relative_path": f"raw/databento/opra_spxw_cbbo_1s/{authorization.sessions[0]}.parquet",
            "bytes": 456,
            "sha256": "a" * 64,
        },
        {
            "relative_path": f"raw/thetadata/spx/{authorization.sessions[0]}.parquet",
            "bytes": 789,
            "sha256": "e" * 64,
        },
    ]
    source_files.sort(key=lambda row: row["relative_path"])
    example_hashes = [example.canonical_sha256()]
    session_content = prereg.stable_hash(
        {
            "schema_version": dataset_module.EntrySessionV1.SCHEMA_VERSION,
            "session": authorization.sessions[0],
            "source_files": source_files,
            "source_files_sha256": prereg.stable_hash(source_files),
            "ordered_example_hashes": example_hashes,
        }
    )
    receipt = {
        "session": authorization.sessions[0],
        "source_files": source_files,
        "source_files_sha256": prereg.stable_hash(source_files),
        "example_count": 1,
        "ordered_example_list_sha256": prereg.stable_hash(example_hashes),
        "session_content_sha256": session_content,
    }
    receipt["receipt_sha256"] = prereg.stable_hash(receipt)
    source_receipts = (receipt,)
    authorization_sha = prereg.stable_hash(authorization.to_dict())
    dataset_type = (
        dataset_module.EntryEvidenceDatasetV1
        if type(authorization) is prereg.FrozenEvidenceAuthorization
        else dataset_module.EntryFitDatasetV1
    )
    dataset_sha = dataset_module.canonical_entry_dataset_sha256(
        schema_version=dataset_type.SCHEMA_VERSION,
        authorization_sha256=authorization_sha,
        role=authorization.role,
        sessions=authorization.sessions,
        sessions_sha256_newline=authorization.sessions_sha256_newline,
        source_receipts=source_receipts,
        examples=(example,),
    )
    return dataset_type(
        schema_version=dataset_type.SCHEMA_VERSION,
        authorization_sha256=authorization_sha,
        role=authorization.role,
        sessions=authorization.sessions,
        sessions_sha256_newline=authorization.sessions_sha256_newline,
        source_receipts=source_receipts,
        examples=(example,),
        dataset_sha256=dataset_sha,
    )


def _sealed_quote(
    dataset_module: Any,
    fill_module: Any,
    contract: ContractIdentityV1,
    *,
    available_at_ns: int,
    bid_micros: int = 2_000_000,
    ask_micros: int = 2_100_000,
    source_receipt_sha256: str = "c" * 64,
) -> Any:
    expected_actionable = bid_micros > 0 and ask_micros > 0 and bid_micros < ask_micros
    selection_key = [
        available_at_ns,
        available_at_ns,
        available_at_ns - 1_000_000_000,
        "raw/databento/opra_spxw_cbbo_1s/2026-01-02.parquet",
        0,
        0,
    ]
    proof = prereg.stable_hash(
        {
            "eligible_row_count": 1,
            "selected_key": selection_key,
            "candidate_keys_sha256": prereg.stable_hash([selection_key]),
        }
    )
    values = {
        "schema_version": dataset_module.VerifiedResearchQuoteRowV1.SCHEMA_VERSION,
        "session": "2026-01-02",
        "source_relative_path": selection_key[3],
        "source_file_sha256": "a" * 64,
        "row_group": 0,
        "row_index": 0,
        "canonical_row_sha256": "b" * 64,
        "contract": contract,
        "source_vendor": "DATABENTO_OPRA",
        "represented_interval_end_ns": selection_key[2],
        "ts_recv_ns": available_at_ns,
        "available_at_ns": available_at_ns,
        "bid_micros": bid_micros,
        "ask_micros": ask_micros,
        "source_receipt_sha256": source_receipt_sha256,
        "query_at_or_before_ns": available_at_ns,
        "query_maximum_age_ms": 2_000,
        "query_require_actionable": expected_actionable,
        "eligible_row_count": 1,
        "selection_key": tuple(selection_key),
        "selection_proof_sha256": proof,
    }
    semantic = dict(values)
    semantic["contract"] = {
        field.name: getattr(contract, field.name) for field in fields(contract)
    }
    source_row = dataset_module.VerifiedResearchQuoteRowV1(
        **values, record_sha256=prereg.stable_hash(semantic)
    )
    quote = fill_module.seal_research_quote(source_row)
    assert quote.actionable is expected_actionable
    if expected_actionable:
        assert quote.invalid_reason is None
    elif bid_micros == 0:
        assert quote.invalid_reason == "ZERO_BID"
    return quote


def _verified_spx_row(
    dataset_module: Any,
    *,
    available_at_ns: int,
    source_receipt_sha256: str,
) -> Any:
    selection_key = [
        available_at_ns,
        available_at_ns,
        available_at_ns - 1_000_000_000,
        "raw/thetadata/spx/2026-01-02.parquet",
        0,
        0,
    ]
    proof = prereg.stable_hash(
        {
            "eligible_row_count": 1,
            "selected_key": selection_key,
            "candidate_keys_sha256": prereg.stable_hash([selection_key]),
        }
    )
    values = {
        "schema_version": dataset_module.VerifiedOfficialSpxRowV1.SCHEMA_VERSION,
        "session": "2026-01-02",
        "source_relative_path": selection_key[3],
        "source_file_sha256": "e" * 64,
        "row_group": 0,
        "row_index": 0,
        "canonical_row_sha256": "f" * 64,
        "source_vendor": "THETADATA",
        "represented_interval_end_ns": selection_key[2],
        "ts_recv_ns": available_at_ns,
        "available_at_ns": available_at_ns,
        "spx_micros": 6_000_000_000,
        "source_receipt_sha256": source_receipt_sha256,
        "query_at_or_before_ns": available_at_ns,
        "query_maximum_age_ms": 90_000,
        "eligible_row_count": 1,
        "selection_key": tuple(selection_key),
        "selection_proof_sha256": proof,
    }
    return dataset_module.VerifiedOfficialSpxRowV1(
        **values, record_sha256=prereg.stable_hash(values)
    )


def test_public_api_surface_exact() -> None:
    contract = prereg.entry_future_api_contract()
    for row in contract["functions"]:
        value = _resolve(row["qualified_name"])
        assert _signature_rows(value) == row["parameters"]

    class_locations = {
        "EntrySnapshotV1": "v4.research.pathd_entry_features.EntrySnapshotV1",
        "Signed17FrameV1": "v4.research.pathd_entry_features.Signed17FrameV1",
        "Signed17HistoryV1": "v4.research.pathd_entry_features.Signed17HistoryV1",
        "EntryFuturePathV1": "v4.research.pathd_entry_dataset.EntryFuturePathV1",
        "EntryExampleV1": "v4.research.pathd_entry_dataset.EntryExampleV1",
        "EntryActionExecutionFactV1": "v4.research.pathd_entry_dataset.EntryActionExecutionFactV1",
        "EntrySessionV1": "v4.research.pathd_entry_dataset.EntrySessionV1",
        "EntryFitDatasetV1": "v4.research.pathd_entry_dataset.EntryFitDatasetV1",
        "EntryEvidenceDatasetV1": "v4.research.pathd_entry_dataset.EntryEvidenceDatasetV1",
        "EntryModelInputV1": "v4.research.pathd_entry_models.EntryModelInputV1",
        "EntryPredictionV1": "v4.research.pathd_entry_models.EntryPredictionV1",
        "EntryModelBundleV1": "v4.research.pathd_entry_models.EntryModelBundleV1",
        "EntryCalibrationBundleV1": "v4.research.pathd_entry_models.EntryCalibrationBundleV1",
        "EntryComposerBundleV1": "v4.research.pathd_entry_models.EntryComposerBundleV1",
        "EntryNegativeControlBundleV1": "v4.research.pathd_entry_models.EntryNegativeControlBundleV1",
        "EntryNegativeControlCalibrationBundleV1": "v4.research.pathd_entry_models.EntryNegativeControlCalibrationBundleV1",
        "EntryNegativeControlComposerBundleV1": "v4.research.pathd_entry_models.EntryNegativeControlComposerBundleV1",
        "EntryNegativeControlManifestV1": "v4.research.pathd_entry_models.EntryNegativeControlManifestV1",
        "EntryReplayEvaluationCellV1": "v4.research.pathd_entry_models.EntryReplayEvaluationCellV1",
        "EntryActionCalibrationObservationV1": "v4.path_d.execution.research_replay.EntryActionCalibrationObservationV1",
        "EntryTime300TrajectoryEvidenceV1": "v4.path_d.execution.research_replay.EntryTime300TrajectoryEvidenceV1",
        "EntrySurvivalAuditRecordV1": "v4.path_d.execution.research_replay.EntrySurvivalAuditRecordV1",
        "EntryNegativeControlPanelV1": "v4.research.pathd_entry_models.EntryNegativeControlPanelV1",
        "EntryPolicyEvaluationV1": "v4.scripts.run_pathd_entry_exit_research.EntryPolicyEvaluationV1",
        "EntryNestedFamilyEvaluationV1": "v4.scripts.run_pathd_entry_exit_research.EntryNestedFamilyEvaluationV1",
        "EntryOuterPrimaryResultV1": "v4.scripts.run_pathd_entry_exit_research.EntryOuterPrimaryResultV1",
        "EntryPooledAcceptanceResultV1": "v4.scripts.run_pathd_entry_exit_research.EntryPooledAcceptanceResultV1",
        "FrozenContextDiagnosticsAuthorizationV1": "v4.research.pathd_entry_exit.FrozenContextDiagnosticsAuthorizationV1",
        "EntryContextSourceSelectionV1": "v4.scripts.run_pathd_entry_exit_research.EntryContextSourceSelectionV1",
        "EntryContextAnchorRowV1": "v4.scripts.run_pathd_entry_exit_research.EntryContextAnchorRowV1",
        "EntryOuterContextDiagnosticsV1": "v4.scripts.run_pathd_entry_exit_research.EntryOuterContextDiagnosticsV1",
        "EntryPooledContextDiagnosticsV1": "v4.scripts.run_pathd_entry_exit_research.EntryPooledContextDiagnosticsV1",
        "EntryControlReplayConfigV1": "v4.research.pathd_entry_models.EntryControlReplayConfigV1",
        "EntryControlReplayResultV1": "v4.research.pathd_entry_models.EntryControlReplayResultV1",
        "EntryControlExitSelectionV1": "v4.research.pathd_entry_models.EntryControlExitSelectionV1",
        "EntryNestedReplayConfigV1": "v4.research.pathd_entry_models.EntryNestedReplayConfigV1",
        "EntryActionDecisionV1": "v4.research.pathd_entry_models.EntryActionDecisionV1",
        "EntryActionObservationV1": "v4.path_d.execution.research_replay.EntryActionObservationV1",
        "EntryFrameCoverageV1": "v4.path_d.execution.research_replay.EntryFrameCoverageV1",
        "ResearchFillLawV1": "v4.path_d.execution.research_fill_law.ResearchFillLawV1",
        "VerifiedResearchQuoteRowV1": "v4.research.pathd_entry_dataset.VerifiedResearchQuoteRowV1",
        "VerifiedOfficialSpxRowV1": "v4.research.pathd_entry_dataset.VerifiedOfficialSpxRowV1",
        "ResearchQuoteV1": "v4.path_d.execution.research_fill_law.ResearchQuoteV1",
        "ResearchFillDecisionV1": "v4.path_d.execution.research_fill_law.ResearchFillDecisionV1",
        "ResearchExecutionResultV1": "v4.path_d.execution.research_replay.ResearchExecutionResultV1",
        "ResearchDecisionContextV1": "v4.path_d.execution.research_replay.ResearchDecisionContextV1",
        "ResearchLedgerStateV1": "v4.path_d.execution.research_replay.ResearchLedgerStateV1",
        "ResearchLedgerControlEventV1": "v4.path_d.execution.research_replay.ResearchLedgerControlEventV1",
        "ResearchLedgerTransitionV1": "v4.path_d.execution.research_replay.ResearchLedgerTransitionV1",
        "ResearchLedgerJournalV1": "v4.path_d.execution.research_replay.ResearchLedgerJournalV1",
        "ResearchPreparedExecutionV1": "v4.path_d.execution.research_replay.ResearchPreparedExecutionV1",
        "ResearchExecutionOutcomeV1": "v4.path_d.execution.research_replay.ResearchExecutionOutcomeV1",
        "ResearchResultEnvelopeV1": "v4.path_d.execution.research_replay.ResearchResultEnvelopeV1",
        "NonOrderSafetyEventV1": "v4.path_d.execution.research_replay.NonOrderSafetyEventV1",
        "ProtectedHoldoutStateV1": "v4.research.pathd_holdout_gate.ProtectedHoldoutStateV1",
        "ProtectedHoldoutEvaluationV1": "v4.scripts.run_pathd_entry_exit_research.ProtectedHoldoutEvaluationV1",
        "ProtectedHoldoutTraceRecordV1": "v4.scripts.run_pathd_entry_exit_research.ProtectedHoldoutTraceRecordV1",
    }
    assert set(class_locations) == set(contract["schema_versions"])
    for name, version in contract["schema_versions"].items():
        cls = _resolve(class_locations[name])
        assert cls.SCHEMA_VERSION == version
        expected_fields = contract["dataclass_fields"].get(name)
        if expected_fields is not None:
            assert [field.name for field in fields(cls)] == expected_fields


def test_signed17_historical_source_neutral_byte_equality() -> None:
    features = importlib.import_module("v4.research.pathd_entry_features")
    inputs = _causal_inputs()
    historical = features.historical_snapshot_from_processed_row(_processed_row())
    neutral = features.source_neutral_snapshot_from_causal_inputs(**inputs)
    assert historical.canonical_sha256() == neutral.canonical_sha256()
    left = features.signed17_from_snapshot(historical)
    right = features.signed17_from_snapshot(neutral)
    assert left.feature_names == tuple(FEATURE_NAMES)
    assert left.values.shape == (21, 2, 17)
    assert left.values.dtype == np.float64
    assert left.finite.dtype == np.bool_
    np.testing.assert_array_equal(left.values, right.values)
    reference = feature_matrix(_processed_row())
    np.testing.assert_array_equal(left.values, reference)
    np.testing.assert_array_equal(right.values, reference)
    assert hashlib.sha256(left.values.tobytes(order="C")).hexdigest() == (
        "36045553dfea1bce2712501e14ad4101ab78204ae1506eb8bf4c4a3d952bded0"
    )
    assert hashlib.sha256(left.finite.tobytes(order="C")).hexdigest() == (
        "c90c0e79b667f80ea85d59dd406be1714a02066a49266d0a8eb47ac4c0a0b007"
    )
    assert left.finite.sum(axis=(0, 1)).tolist() == [
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 14, 14, 14, 25, 25
    ]
    assert left.canonical_sha256() == right.canonical_sha256()


def test_signed17_lineage_fail_closed() -> None:
    features = importlib.import_module("v4.research.pathd_entry_features")
    inputs = _causal_inputs()
    for record in inputs["declared_alpha_sources"]:
        features.validate_alpha_source_record(record)
    base = copy.deepcopy(inputs["declared_alpha_sources"][0])
    bad_rows = []
    row = copy.deepcopy(base)
    row["source_vendors"] = ["IBKR"]
    row["transitive_source_leaves"] = ["ibkr_bid"]
    bad_rows.append(row)
    row = copy.deepcopy(base)
    row["raw_vendor_greek"] = True
    bad_rows.append(row)
    row = copy.deepcopy(base)
    row["available_after_decision"] = True
    bad_rows.append(row)
    row = copy.deepcopy(base)
    row["name"] = "unregistered_alpha"
    bad_rows.append(row)
    row = copy.deepcopy(base)
    row["future_live_twin_adapter"] = ""
    bad_rows.append(row)
    for row in bad_rows:
        with pytest.raises((TypeError, ValueError)):
            features.validate_alpha_source_record(row)

    later = dict(inputs)
    later["contract_quote_watermark_ns"] = np.full((21, 2), inputs["decision_time_ns"] + 1, dtype=np.int64)
    with pytest.raises((TypeError, ValueError)):
        features.source_neutral_snapshot_from_causal_inputs(**later)

    current = features.source_neutral_snapshot_from_causal_inputs(**inputs)
    changed_inputs = dict(inputs)
    changed_market = inputs["market_window"].copy()
    changed_market[-1, 0] += 5.0
    changed_inputs["market_window"] = changed_market
    changed = features.source_neutral_snapshot_from_causal_inputs(**changed_inputs)
    assert features.signed17_from_snapshot(current).canonical_sha256() != features.signed17_from_snapshot(changed).canonical_sha256()


def test_mutate_future_is_causal_and_live() -> None:
    features = importlib.import_module("v4.research.pathd_entry_features")
    dataset = importlib.import_module("v4.research.pathd_entry_dataset")
    models = importlib.import_module("v4.research.pathd_entry_models")
    fill_module = importlib.import_module("v4.path_d.execution.research_fill_law")
    payload = prereg.preregistration_payload()[0]
    law = fill_module.research_fill_law_from_preregistration(payload)
    snapshot = features.source_neutral_snapshot_from_causal_inputs(**_causal_inputs())
    decision = snapshot.decision_time_ns
    base = _valid_future_path(dataset, snapshot, nonce="base")
    changed_marks = copy.deepcopy(base.marks_by_horizon)
    changed_marks["h10"][1] = 3_000_000
    variants = [
        replace(base, arrival_ask_micros=2_150_000),
        replace(base, arrival_bid_micros=1_950_000),
        replace(base, marks_by_horizon=changed_marks),
        replace(base, future_audit_nonce="mutated-label-and-exit-audit"),
    ]
    examples = [dataset.build_entry_example(snapshot, future_path=value, fill_law=law) for value in (base, *variants)]
    inputs = [models.model_input_from_example(example) for example in examples]
    predictions = [models.causal_probe_predict(value) for value in inputs]
    assert len({value.canonical_sha256() for value in inputs}) == 1
    assert len({value.canonical_sha256() for value in predictions}) == 1
    target_hashes = {prereg.stable_hash(example.targets) for example in examples}
    validity_hashes = {prereg.stable_hash(example.target_validity) for example in examples}
    assert len(target_hashes | validity_hashes) > 1
    terminal = pd.Timestamp("2026-01-02 15:55:00", tz="America/New_York").tz_convert("UTC").value
    assert dataset.available_entry_horizons(
        decision_time_ns=decision, terminal_time_ns=int(terminal)
    ) == ("h10", "h20", "h45", "h90", "remaining_session")
    arithmetic = dataset.entry_targets_from_executable_marks(
        entry_fill_price_micros=2_050_000,
        round_trip_fee_micros=3_000_000,
        marks_by_horizon={
            "h3": [2_050_000] * 3,
            "h5": [2_050_000] * 5,
            "h10": [2_000_000, 2_200_000, 1_900_000] + [2_050_000] * 7,
            "h20": [2_050_000] * 20,
            "h45": [2_050_000] * 45,
            "h90": [2_050_000] * 90,
            "remaining_session": [2_050_000] * 354,
        },
        available_horizons=("h10", "h20", "h45", "h90", "remaining_session"),
    )
    assert arithmetic["h10_mfe_dollars"] == 12.0
    assert arithmetic["h10_profit_area_dollars"] == 12.0
    assert arithmetic["h10_mfe_return"] == 12.0 / 205.0
    assert arithmetic["h10_profit_area_return"] == 12.0 / 205.0
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        dataset.build_entry_example(
            snapshot,
            future_path=replace(
                base,
                marks_by_horizon={**base.marks_by_horizon, "h10": [2_100_000] * 9},
            ),
            fill_law=law,
        )


def test_frozen_roles_history_and_model_input_boundary() -> None:
    features = importlib.import_module("v4.research.pathd_entry_features")
    dataset_module = importlib.import_module("v4.research.pathd_entry_dataset")
    models = importlib.import_module("v4.research.pathd_entry_models")
    snapshot = features.source_neutral_snapshot_from_causal_inputs(**_causal_inputs())
    current = features.signed17_from_snapshot(snapshot)
    previous_ids = current.contract_ids.copy().reshape(-1)
    previous_values = current.values.copy().reshape(42, 17)
    previous_ids[[0, 1]] = previous_ids[[1, 0]]
    previous_values[0, :] = 111.0
    previous_values[1, :] = 222.0
    previous = replace(
        current,
        contract_ids=previous_ids.reshape(21, 2),
        values=previous_values.reshape(21, 2, 17),
        finite=np.isfinite(previous_values).reshape(21, 2, 17),
        decision_time_ns=current.decision_time_ns - 60_000_000_000,
        decision_watermark_ns=current.decision_watermark_ns - 60_000_000_000,
    )
    other_session = replace(
        previous,
        decision_time_ns=previous.decision_time_ns - 60_000_000_000,
        decision_watermark_ns=previous.decision_watermark_ns - 60_000_000_000,
    )
    history = features.build_identity_joined_history(
        [("1999-01-01", other_session), (snapshot.session, previous), (snapshot.session, current)],
        current_session=snapshot.session,
        current_decision_time_ns=current.decision_time_ns,
        current_contract_ids=current.contract_ids,
        history_minutes=3,
    )
    assert history.values.shape == (3, 42, 17)
    assert not history.minute_available[0]
    assert history.values[-2, 0, 0] == 222.0
    assert history.values[-2, 1, 0] == 111.0
    offsets = np.repeat(snapshot.strike_offsets, 2)
    rights = np.tile(np.asarray(snapshot.rights, dtype=object), 21)
    summary = features.hgb_signed17_summaries(history, current_offsets=offsets, current_rights=rights)
    assert summary.shape == (42, 444)
    np.testing.assert_array_equal(summary[:, -2], offsets / 50.0)
    np.testing.assert_array_equal(summary[:, -1], (rights == "C").astype(np.float64))

    assignments = prereg.session_assignments()
    sessions = prereg._resolve_frozen_fit_sessions(assignments, role="outer_weights", outer_fold=1, inner_fold=None)
    assert sessions == tuple(assignments["folds"][0]["model_fit"])
    assert prereg.canonical_session_hash(sessions) == prereg.canonical_session_hash(
        assignments["folds"][0]["model_fit"]
    )
    prereg.assert_no_holdout_sessions(sessions)
    prereg.assert_no_evidence_firewall_sessions(sessions)
    with pytest.raises(RuntimeError):
        prereg.assert_no_evidence_firewall_sessions((assignments["protected_holdout_30"][0],))

    weights_auth = _dummy_authorization()
    fill_module = importlib.import_module("v4.path_d.execution.research_fill_law")
    law = fill_module.research_fill_law_from_preregistration(prereg.preregistration_payload()[0])
    future = _valid_future_path(
        dataset_module, snapshot, nonce="fit-wrapper-provenance"
    )
    example = dataset_module.build_entry_example(snapshot, future_path=future, fill_law=law)
    assert len(example.action_execution_facts) == 42
    assert all(
        fact.schema_version == dataset_module.EntryActionExecutionFactV1.SCHEMA_VERSION
        and fact.session == snapshot.session
        and fact.decision_time_ns == snapshot.decision_time_ns
        and fact.fact_sha256 != ""
        for fact in example.action_execution_facts
    )
    model_input = models.model_input_from_example(example)
    np.testing.assert_array_equal(
        model_input.physical_action_mask,
        np.asarray(
            [fact.physical_eligible for fact in example.action_execution_facts],
            dtype=np.bool_,
        ),
    )
    fit_dataset = _sealed_dataset_for_authorization(dataset_module, weights_auth, example)
    original_dataset_fit_validator = dataset_module.assert_fit_authorization_current
    try:
        dataset_module.assert_fit_authorization_current = lambda value: value
        dataset_module.validate_entry_fit_dataset(
            fit_dataset, authorization=weights_auth
        )
    finally:
        dataset_module.assert_fit_authorization_current = (
            original_dataset_fit_validator
        )
    original_validator = models.assert_fit_authorization_current
    original_loader = models.load_authorized_entry_dataset
    original_hgb_impl = models._fit_hgb_entry_bundle_impl
    original_neural_impl = models._fit_neural_entry_bundle_impl
    original_cal_impl = models._fit_entry_calibrators_impl
    try:
        order: list[str] = []
        models.load_authorized_entry_dataset = lambda value: order.append("LOAD") or fit_dataset
        models.assert_fit_authorization_current = lambda value: order.append("AUTH") or value
        def fake_hgb_impl(authorization: Any, dataset: Any) -> Any:
            order.append("HGB_IMPL")
            return models.seal_entry_model_bundle(
                family="HGB",
                authorization=authorization,
                dataset_sha256=dataset.dataset_sha256,
                payload={"weights_sha256": "8" * 64},
            )

        models._fit_hgb_entry_bundle_impl = fake_hgb_impl
        hgb_bundle = models.fit_hgb_entry_bundle(weights_auth)
        models.validate_entry_model_bundle(hgb_bundle)
        assert hgb_bundle.family == "HGB"
        assert hgb_bundle.authorization_sha256 == prereg.stable_hash(
            weights_auth.to_dict()
        )
        assert hgb_bundle.dataset_sha256 == fit_dataset.dataset_sha256
        assert order == ["LOAD", "AUTH", "HGB_IMPL"]
        with pytest.raises(TypeError):
            models.fit_hgb_entry_bundle(weights_auth, dataset=fit_dataset)
        order.clear()
        models._fit_neural_entry_bundle_impl = lambda authorization, dataset, hgb_bundle: order.append("NEURAL_IMPL") or "NEURAL_OK"
        assert models.fit_neural_entry_bundle(
            weights_auth, hgb_bundle=hgb_bundle
        ) == "NEURAL_OK"
        assert order == ["LOAD", "AUTH", "NEURAL_IMPL"]
        wrong_family = models.seal_entry_model_bundle(
            family="NEURAL", authorization=weights_auth,
            dataset_sha256=fit_dataset.dataset_sha256,
            payload={"weights_sha256": "9" * 64},
        )
        wrong_authorization = replace(weights_auth, outer_fold=2)
        wrong_scope = models.seal_entry_model_bundle(
            family="HGB", authorization=wrong_authorization,
            dataset_sha256=fit_dataset.dataset_sha256,
            payload={"weights_sha256": "a" * 64},
        )
        wrong_sessions_authorization = replace(
            weights_auth,
            sessions=("2026-01-05",),
            sessions_sha256_newline=prereg.canonical_session_hash(("2026-01-05",)),
        )
        wrong_sessions = models.seal_entry_model_bundle(
            family="HGB", authorization=wrong_sessions_authorization,
            dataset_sha256=fit_dataset.dataset_sha256,
            payload={"weights_sha256": "d" * 64},
        )
        wrong_dataset = models.seal_entry_model_bundle(
            family="HGB", authorization=weights_auth,
            dataset_sha256="b" * 64,
            payload={"weights_sha256": "c" * 64},
        )
        for invalid_baseline in (
            {"family": "HGB"}, wrong_family, wrong_scope, wrong_sessions,
            wrong_dataset,
        ):
            order.clear()
            with pytest.raises((RuntimeError, TypeError, ValueError)):
                models.fit_neural_entry_bundle(
                    weights_auth, hgb_bundle=invalid_baseline
                )
            assert "LOAD" not in order
            assert "NEURAL_IMPL" not in order
        calibration_auth = replace(weights_auth, role="outer_calibration")
        calibration_dataset = _sealed_dataset_for_authorization(
            dataset_module, calibration_auth, example
        )
        model_bundle = models.seal_entry_model_bundle(
            family="HGB", authorization=weights_auth,
            dataset_sha256=fit_dataset.dataset_sha256,
            payload={"weights_sha256": "8" * 64},
        )
        models.validate_entry_model_bundle(model_bundle)
        models.load_authorized_entry_dataset = lambda value: order.append("LOAD_CAL") or calibration_dataset
        models._fit_entry_calibrators_impl = lambda authorization, model_bundle, dataset: order.append("CAL_IMPL") or "CAL_OK"
        order.clear()
        assert models.fit_entry_calibrators(
            calibration_auth, model_bundle=model_bundle
        ) == "CAL_OK"
        assert order == ["LOAD_CAL", "AUTH", "CAL_IMPL"]
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.fit_hgb_entry_bundle(calibration_auth)
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.fit_entry_calibrators(
                weights_auth, model_bundle=model_bundle
            )
        cross_session = replace(
            example,
            model_input=replace(example.model_input, session="2099-01-01"),
        )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            dataset_module.validate_entry_fit_dataset(
                replace(fit_dataset, examples=(cross_session,)), authorization=weights_auth
            )
        mutated_target = replace(
            example, targets={**example.targets, next(iter(example.targets)): 999.0}
        )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            dataset_module.validate_entry_fit_dataset(
                replace(fit_dataset, examples=(mutated_target,)), authorization=weights_auth
            )
        for changed in (
            replace(model_bundle, fit_role="full_weights"),
            replace(model_bundle, outer_fold=5),
            replace(model_bundle, authorization_sha256="0" * 64),
            replace(model_bundle, sessions_sha256_newline="0" * 64),
            replace(model_bundle, artifact_sha256="0" * 64),
        ):
            with pytest.raises((RuntimeError, TypeError, ValueError)):
                models.validate_entry_model_bundle(changed)
            with pytest.raises((RuntimeError, TypeError, ValueError)):
                models.fit_entry_calibrators(
                    calibration_auth, model_bundle=changed
                )
        wrong_outer = replace(calibration_auth, outer_fold=5)
        wrong_outer_dataset = replace(
            calibration_dataset,
            authorization_sha256=prereg.stable_hash(wrong_outer.to_dict()),
        )
        models.load_authorized_entry_dataset = lambda value: wrong_outer_dataset
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.fit_entry_calibrators(
                wrong_outer, model_bundle=model_bundle
            )
        nested_weights = replace(weights_auth, role="nested_weights", inner_fold=1)
        nested_model = models.seal_entry_model_bundle(
            family="HGB", authorization=nested_weights,
            dataset_sha256=fit_dataset.dataset_sha256,
            payload={"weights_sha256": "9" * 64},
        )
        wrong_inner_cal = replace(
            weights_auth, role="nested_calibration", inner_fold=2
        )
        wrong_inner_dataset = replace(
            fit_dataset,
            authorization_sha256=prereg.stable_hash(wrong_inner_cal.to_dict()),
            role="nested_calibration",
        )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.load_authorized_entry_dataset = lambda value: wrong_inner_dataset
            models.fit_entry_calibrators(
                wrong_inner_cal,
                model_bundle=nested_model,
            )
    finally:
        models.assert_fit_authorization_current = original_validator
        models.load_authorized_entry_dataset = original_loader
        models._fit_hgb_entry_bundle_impl = original_hgb_impl
        models._fit_neural_entry_bundle_impl = original_neural_impl
        models._fit_entry_calibrators_impl = original_cal_impl

    mean, std = models.session_balanced_population_mean_std(
        np.asarray([1.0, 3.0, 10.0], dtype=np.float64),
        sessions=("A", "A", "B"),
        valid=np.asarray([True, True, True], dtype=np.bool_),
    )
    assert mean == 6.0
    assert std == math.sqrt(16.5)
    assert models.weighted_lower_conformal_correction(
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 3.0, 10.0], dtype=np.float64),
        sessions=("A", "A", "B"),
        identities=("A-0", "A-1", "B-0"),
        alpha=0.1,
    ) == 1.0
    assert models.wait_raw_lower_statistic((-1.0, 2.0, 0.5)) == 2.0
    assert models.wait_raw_lower_statistic(()) == 0.0

    hgb = models.build_hgb_regressor(
        seed=101, loss="squared_error", quantile=None
    )
    hgb_params = hgb.get_params(deep=False)
    assert {
        name: hgb_params[name]
        for name in (
            "loss", "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
            "min_samples_leaf", "l2_regularization", "max_bins", "early_stopping",
            "random_state",
        )
    } == {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "max_depth": 3,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 101,
    }
    q10 = models.build_hgb_regressor(seed=102, loss="quantile", quantile=0.1)
    assert q10.loss == "quantile" and q10.quantile == 0.1

    synthetic_x = np.column_stack(
        (
            np.linspace(-2.0, 2.0, 96, dtype=np.float64),
            np.sin(np.linspace(0.0, 4.0, 96, dtype=np.float64)),
            np.cos(np.linspace(0.0, 2.0, 96, dtype=np.float64)),
            np.arange(96, dtype=np.float64) % 7.0,
        )
    )
    synthetic_y = (
        2.0 * synthetic_x[:, 0]
        - 0.5 * synthetic_x[:, 1]
        + 0.25 * synthetic_x[:, 3]
    )
    synthetic_weight = np.where(
        np.arange(96) < 48, 0.5, 1.5
    ).astype(np.float64)
    fitted_left = models.build_hgb_regressor(
        seed=101, loss="squared_error", quantile=None
    ).fit(synthetic_x, synthetic_y, sample_weight=synthetic_weight)
    fitted_right = models.build_hgb_regressor(
        seed=101, loss="squared_error", quantile=None
    ).fit(synthetic_x, synthetic_y, sample_weight=synthetic_weight)
    fitted_prediction = fitted_left.predict(synthetic_x)
    assert np.isfinite(fitted_prediction).all()
    assert np.ptp(fitted_prediction) > 0.0
    np.testing.assert_array_equal(
        fitted_prediction, fitted_right.predict(synthetic_x)
    )
    reloaded_hgb = pickle.loads(
        pickle.dumps(fitted_left, protocol=pickle.HIGHEST_PROTOCOL)
    )
    np.testing.assert_array_equal(
        fitted_prediction, reloaded_hgb.predict(synthetic_x)
    )
    q10.fit(synthetic_x, synthetic_y, sample_weight=synthetic_weight)
    q10_prediction = q10.predict(synthetic_x)
    q10_repeat = models.build_hgb_regressor(
        seed=102, loss="quantile", quantile=0.1
    ).fit(
        synthetic_x, synthetic_y, sample_weight=synthetic_weight
    ).predict(synthetic_x)
    assert np.isfinite(q10_prediction).all()
    np.testing.assert_array_equal(q10_prediction, q10_repeat)
    np.testing.assert_array_equal(
        q10_prediction,
        pickle.loads(
            pickle.dumps(q10, protocol=pickle.HIGHEST_PROTOCOL)
        ).predict(synthetic_x),
    )

    neural_left = models.build_entry_neural_module(seed=211)
    neural_right = models.build_entry_neural_module(seed=211)
    assert sum(parameter.numel() for parameter in neural_left.parameters()) == 45_768
    assert all(parameter.device.type == "cpu" for parameter in neural_left.parameters())
    assert tuple(neural_left.state_dict()) == tuple(neural_right.state_dict())
    for name in neural_left.state_dict():
        torch.testing.assert_close(
            neural_left.state_dict()[name], neural_right.state_dict()[name],
            rtol=0.0, atol=0.0,
        )

    history_tokens = torch.linspace(
        -1.0, 1.0, steps=2 * 90 * 42 * 34, dtype=torch.float32
    ).reshape(2, 90, 42, 34)
    history_tokens[..., 17:] = (
        history_tokens[..., 17:] >= 0.0
    ).to(dtype=torch.float32)
    geometry_one = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 42, dtype=torch.float32),
            torch.tensor(([1.0, 0.0] * 21), dtype=torch.float32),
        ),
        dim=-1,
    )
    geometry = geometry_one.unsqueeze(0).repeat(2, 1, 1)
    current_present = torch.ones((2, 42), dtype=torch.bool)
    target = torch.linspace(
        -0.5, 0.5, steps=2 * 42 * 20, dtype=torch.float32
    ).reshape(2, 42, 20)

    def synthetic_neural_step(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(9_001)
        module.train()
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0001,
            amsgrad=False,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        optimizer.zero_grad(set_to_none=True)
        raw = module(history_tokens, geometry, current_present)
        assert raw.shape == (2, 42, 40)
        assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
        conditional_mean = raw[..., :20]
        raw_q10 = raw[..., 20:]
        mean_loss = torch.square(conditional_mean - target).mean()
        q10_residual = target - raw_q10
        q10_loss = torch.maximum(
            0.10 * q10_residual, -0.90 * q10_residual
        ).mean()
        loss = 0.5 * mean_loss + 0.5 * q10_loss
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            module.parameters(),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=False,
        )
        assert torch.isfinite(gradient_norm)
        optimizer.step()
        module.eval()
        with torch.inference_mode():
            post = module(history_tokens, geometry, current_present)
        return loss.detach(), post, *tuple(
            value.detach().clone() for value in module.state_dict().values()
        )

    trained_left = synthetic_neural_step(neural_left)
    trained_right = synthetic_neural_step(neural_right)
    assert len(trained_left) == len(trained_right)
    for left_value, right_value in zip(trained_left, trained_right, strict=True):
        torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)

    serialized = io.BytesIO()
    torch.save(neural_left.state_dict(), serialized)
    serialized.seek(0)
    reloaded_neural = models.build_entry_neural_module(seed=999)
    reloaded_neural.load_state_dict(
        torch.load(serialized, map_location="cpu", weights_only=True), strict=True
    )
    reloaded_neural.eval()
    with torch.inference_mode():
        reloaded_prediction = reloaded_neural(
            history_tokens, geometry, current_present
        )
    torch.testing.assert_close(
        trained_left[1], reloaded_prediction, rtol=0.0, atol=0.0
    )

    # An absent current action is neither a key nor a value in set attention.
    # Its token/geometry bytes therefore cannot influence any present action.
    partial_present = torch.tensor(
        [
            [index % 3 != 0 for index in range(42)],
            [index % 4 not in (0, 1) for index in range(42)],
        ],
        dtype=torch.bool,
    )
    absent_history = (~partial_present)[:, None, :, None].expand_as(history_tokens)
    absent_geometry = (~partial_present)[:, :, None].expand_as(geometry)
    mutated_history = history_tokens.clone()
    mutated_geometry = geometry.clone()
    mutated_history[absent_history] = torch.linspace(
        -1_000_000.0,
        1_000_000.0,
        steps=int(absent_history.sum().item()),
        dtype=torch.float32,
    )
    mutated_geometry[absent_geometry] = torch.linspace(
        1_000_000.0,
        -1_000_000.0,
        steps=int(absent_geometry.sum().item()),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        masked_baseline = reloaded_neural(
            history_tokens, geometry, partial_present
        )
        masked_mutation = reloaded_neural(
            mutated_history, mutated_geometry, partial_present
        )
    present_outputs = partial_present[:, :, None].expand_as(masked_baseline)
    torch.testing.assert_close(
        masked_baseline[present_outputs],
        masked_mutation[present_outputs],
        rtol=0.0,
        atol=0.0,
    )

    # PyTorch attention has an all-key-masked numerical edge case.  The frozen
    # module must handle it explicitly: reject the frame or return a finite,
    # completely masked tensor.  NaNs or learned output bias are not admissible.
    all_absent = torch.zeros_like(partial_present)
    try:
        with torch.inference_mode():
            all_absent_prediction = reloaded_neural(
                mutated_history, mutated_geometry, all_absent
            )
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        assert any(token in message for token in ("action", "mask", "present"))
    else:
        assert all_absent_prediction.shape == (2, 42, 40)
        assert torch.isfinite(all_absent_prediction).all()
        torch.testing.assert_close(
            all_absent_prediction,
            torch.zeros_like(all_absent_prediction),
            rtol=0.0,
            atol=0.0,
        )


def test_shared_fill_law_exact() -> None:
    fill_module = importlib.import_module("v4.path_d.execution.research_fill_law")
    dataset_module = importlib.import_module("v4.research.pathd_entry_dataset")
    payload = prereg.preregistration_payload()[0]
    law = fill_module.research_fill_law_from_preregistration(payload)
    assert law.fill_law_hash == payload["fill_law"]["fill_law_hash"]
    assert fill_module.option_tick_micros(2_950_000) == 50_000
    assert fill_module.option_tick_micros(3_000_000) == 100_000
    assert tuple(law.delay_sensitivity_ms) == (0, 1_000, 2_000, 5_000)
    assert tuple(law.paired_quote_latency_bounds_ms) == (100, 250, 500, 1_000)
    assert law.entry_fee_micros == law.exit_fee_micros == 1_500_000
    assert law.fill_price_mode == "SUBMITTED_HARD_LIMIT"

    buy = _normal_intent(side="BUY", hard_limit_micros=2_150_000)
    arrival = _sealed_quote(
        dataset_module,
        fill_module,
        buy.contract,
        available_at_ns=pd.Timestamp("2026-01-02T15:01:00Z").value,
    )
    filled = fill_module.evaluate_research_arrival(buy, arrival=arrival, law=law)
    assert filled.filled and filled.fill_price_micros == buy.price_budget.hard_limit_micros
    no_fill = fill_module.evaluate_research_arrival(
        buy,
        arrival=_sealed_quote(
            dataset_module, fill_module, buy.contract,
            available_at_ns=arrival.available_at_ns,
            ask_micros=buy.price_budget.hard_limit_micros + 1,
        ),
        law=law,
    )
    assert not no_fill.filled and no_fill.position_unchanged
    invalid = fill_module.evaluate_research_arrival(
        buy,
        arrival=_sealed_quote(
            dataset_module, fill_module, buy.contract,
            available_at_ns=arrival.available_at_ns,
            bid_micros=0, ask_micros=0,
        ),
        law=law,
    )
    assert not invalid.filled and invalid.fill_price_micros is None
    locked = _sealed_quote(
        dataset_module,
        fill_module,
        buy.contract,
        available_at_ns=arrival.available_at_ns,
        bid_micros=2_000_000,
        ask_micros=2_000_000,
    )
    crossed = _sealed_quote(
        dataset_module,
        fill_module,
        buy.contract,
        available_at_ns=arrival.available_at_ns,
        bid_micros=2_100_000,
        ask_micros=2_000_000,
    )
    assert locked.invalid_reason == "LOCKED_BBO"
    assert crossed.invalid_reason == "CROSSED_BBO"

    sell = _normal_intent(side="SELL", hard_limit_micros=1_900_000)
    sell_arrival = _sealed_quote(
        dataset_module, fill_module, sell.contract,
        available_at_ns=arrival.available_at_ns,
    )
    sell_fill = fill_module.evaluate_research_arrival(sell, arrival=sell_arrival, law=law)
    assert sell_fill.filled and sell_fill.fill_price_micros == 1_900_000


def test_default_simulator_governor_unchanged_and_research_opt_in() -> None:
    expected_transcripts = {
        ("FULL", 1): "4ed445ada4634e77259215f7bc98449c6c5821e7bff520589bcdfcab6c93a257",
        ("NO_FILL", 1): "be052bfd9cbc3ceb3b58d6889a1d980a31488cea3e60b252c6d728ed6f92f31a",
        ("PARTIAL", 2): "374b385367a0bc56f8d8ce130929f91a6664617a85a2da1048e32e0fb68b9a70",
        ("LATE_FILL_AFTER_CANCEL", 1): "7334087e24dd74ab22abd8d5dbf0179e2826ab9607cce761daf6ccd1f768508f",
        ("REJECT", 1): "953488ed6b90a725eba32f4dbf945e6e6b98ae8ba4e458da5ca62f708e1f9869",
        ("DISCONNECT", 1): "7b1dee87eb1bd2f76e44f8a46a6ed123f03007e85989f9e45841530fa6a0605a",
    }
    for (outcome, quantity), expected in expected_transcripts.items():
        foundation_executor, foundation_intent, _, foundation_authorization = (
            _foundation_executor(outcome, quantity=quantity)
        )
        events = foundation_executor.submit(
            foundation_intent, foundation_authorization
        )
        assert _transcript_sha256(events) == expected

    unauthorized_executor, unauthorized_intent, unauthorized_broker, _ = (
        _foundation_executor("FULL")
    )
    blocked = GovernorDecisionV1.create(
        intent_id=unauthorized_intent.intent_id,
        disposition="BLOCK",
        reason_codes=("TEST_BLOCK",),
        evaluated_at_utc="2026-06-30T13:30:02Z",
        broker_state_version=unauthorized_broker.snapshot_version,
    )
    assert _transcript_sha256(
        unauthorized_executor.submit(unauthorized_intent, blocked)
    ) == "fb647e854a07ba11a85609813f7b5fe90e8501b93d4b1b61214bf7e5359494be"

    for positions, expected in (
        (
            (),
            "a43907ea52722bef73a77b3eebbe9b8dd75747f3a249ff09176644642a45730c",
        ),
        (
            (
                PositionV1(
                    unauthorized_intent.contract.osi_symbol,
                    1,
                    2_500_000,
                    "2026-06-30T13:30:00Z",
                ),
            ),
            "5bea5cfee4ae2214253e0531f02af63e69c247bcc777bfb9b15a984fb0c822e1",
        ),
    ):
        disconnect, disconnect_intent, _, disconnect_authorization = (
            _foundation_executor("DISCONNECT")
        )
        disconnected = disconnect.submit(
            disconnect_intent, disconnect_authorization
        )
        reconciled = disconnect.reconcile(
            disconnected[-1].order_id,
            fake_broker_state(
                captured_at_utc="2026-06-30T13:30:03Z",
                snapshot_version="fake-state-2",
                positions=positions,
            ),
        )
        assert _transcript_sha256(reconciled) == expected

    intent = _normal_intent(side="SELL", hard_limit_micros=1_900_000)
    broker, feed, authorization = _authorization(intent)
    assert authorization.disposition == "ALLOW"
    tape = (SimulatedQuote(0, 2_000_000, 2_100_000),)
    default_executor = SimulatedExecutor(
        clock=VirtualMonotonicClock(datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)),
        quote_tape=tape,
        scenario=ExecutionScenario(outcome="FULL", submit_latency_ms=0, acknowledge_latency_ms=0),
    )
    default_events = default_executor.submit(intent, authorization)
    assert default_events[-1].state_to == "FILLED"
    assert default_events[-1].fill_price_micros == 2_000_000

    research_executor = SimulatedExecutor(
        clock=VirtualMonotonicClock(datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)),
        quote_tape=tape,
        scenario=ExecutionScenario(
            outcome="FULL",
            submit_latency_ms=0,
            acknowledge_latency_ms=0,
            cancel_after_ms=0,
            cancel_confirm_latency_ms=0,
            fill_price_mode="SUBMITTED_HARD_LIMIT",
        ),
    )
    research_events = research_executor.submit(intent, authorization)
    assert research_events[-1].fill_price_micros == 1_900_000
    empty = SimulatedExecutor(
        clock=VirtualMonotonicClock(datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)),
        quote_tape=(),
        scenario=ExecutionScenario(
            outcome="FULL", submit_latency_ms=0, acknowledge_latency_ms=0,
            cancel_after_ms=0, cancel_confirm_latency_ms=0,
            fill_price_mode="SUBMITTED_HARD_LIMIT",
        ),
    ).submit(intent, authorization)
    assert empty[-1].state_to == "CANCEL_CONFIRMED"

    defaults = GovernorConfigV1()
    assert defaults.allowed_forced_flat_reference_vendors == ("IBKR_SAFETY",)
    assert defaults.daily_loss_blocks_close is True
    loss_broker = fake_broker_state(
        captured_at_utc="2026-01-02T15:00:00Z",
        snapshot_version="frozen-gate-state",
        daily_pnl_micros=-defaults.max_daily_loss_micros,
        positions=(PositionV1(intent.contract.osi_symbol, 1, 2_100_000, "2026-01-02T14:59:00Z"),),
    )
    assert DeterministicGovernor().evaluate(intent, broker_state=loss_broker, feed_health=feed, now_utc="2026-01-02T15:00:00Z").disposition == "BLOCK"
    close_config = GovernorConfigV1(daily_loss_blocks_close=False)
    assert DeterministicGovernor(close_config).evaluate(intent, broker_state=loss_broker, feed_health=feed, now_utc="2026-01-02T15:00:00Z").disposition == "ALLOW"

    lifecycle = LifecycleStateV1(
        contract=intent.contract,
        position_snapshot_version=broker.snapshot_version,
        entry_bid_micros=2_100_000,
        running_max_bid_micros=2_100_000,
        current_bid_micros=2_000_000,
        current_ask_micros=2_100_000,
        opened_at_utc="2026-01-02T14:59:00Z",
        feature_contract_version="pathd.entry-signed17.v1",
        feature_snapshot_sha256="sha256:" + "2" * 64,
    )
    terminal_governor = DeterministicGovernor(
        GovernorConfigV1(
            daily_loss_blocks_close=False,
            allowed_forced_flat_reference_vendors=("DATABENTO_OPRA",),
        )
    )
    forced = terminal_governor.forced_flat_intent(
        lifecycle,
        feed_health=feed,
        now_utc="2026-01-02T15:00:00Z",
        reference_vendor="DATABENTO_OPRA",
        reference_bid_micros=2_000_000,
        reference_ask_micros=2_100_000,
        reason_code="PATHD_RESEARCH_TERMINAL",
    )
    assert forced.origin == "RISK_GOVERNOR"
    assert forced.producer.component_version == "pathd.deterministic-governor.v1"
    assert forced.price_budget.hard_limit_micros == 1_950_000
    assert terminal_governor.evaluate(forced, broker_state=broker, feed_health=feed, now_utc="2026-01-02T15:00:00Z").disposition == "ALLOW"
    with pytest.raises(ValueError):
        terminal_governor.forced_flat_intent(
            lifecycle,
            feed_health=feed,
            now_utc="2026-01-02T15:00:00Z",
            reference_vendor="DATABENTO_OPRA",
            reference_bid_micros=0,
            reference_ask_micros=0,
            reason_code="PATHD_RESEARCH_TERMINAL",
        )
    boundary_forced = terminal_governor.forced_flat_intent(
        lifecycle,
        feed_health=feed,
        now_utc="2026-01-02T15:00:00Z",
        reference_vendor="DATABENTO_OPRA",
        reference_bid_micros=3_000_000,
        reference_ask_micros=3_100_000,
        reason_code="PATHD_RESEARCH_TERMINAL",
    )
    assert boundary_forced.price_budget.max_adverse_move_micros == 100_000
    assert boundary_forced.price_budget.hard_limit_micros == 2_900_000
    for bad in (
        {
            "reference_vendor": "DATABENTO_OPRA",
            "reference_bid_micros": 2_000_000,
            "reference_ask_micros": 2_000_000,
            "reason_code": "PATHD_RESEARCH_TERMINAL",
            "feed_health": feed,
        },
        {
            "reference_vendor": "DATABENTO_OPRA",
            "reference_bid_micros": 2_100_000,
            "reference_ask_micros": 2_000_000,
            "reason_code": "PATHD_RESEARCH_TERMINAL",
            "feed_health": feed,
        },
        {
            "reference_vendor": "IBKR_SAFETY",
            "reference_bid_micros": 2_000_000,
            "reference_ask_micros": 2_100_000,
            "reason_code": "PATHD_RESEARCH_TERMINAL",
            "feed_health": feed,
        },
        {
            "reference_vendor": "DATABENTO_OPRA",
            "reference_bid_micros": 2_000_000,
            "reference_ask_micros": 2_100_000,
            "reason_code": "NOT_THE_FROZEN_REASON",
            "feed_health": feed,
        },
        {
            "reference_vendor": "DATABENTO_OPRA",
            "reference_bid_micros": 2_000_000,
            "reference_ask_micros": 2_100_000,
            "reason_code": "PATHD_RESEARCH_TERMINAL",
            "feed_health": FeedHealthV1(
                "2026-01-02T14:59:00Z", "2026-01-02T15:00:00Z"
            ),
        },
    ):
        with pytest.raises(ValueError):
            terminal_governor.forced_flat_intent(
                lifecycle,
                feed_health=bad["feed_health"],
                now_utc="2026-01-02T15:00:00Z",
                reference_vendor=bad["reference_vendor"],
                reference_bid_micros=bad["reference_bid_micros"],
                reference_ask_micros=bad["reference_ask_micros"],
                reason_code=bad["reason_code"],
            )

    replay = importlib.import_module("v4.path_d.execution.research_replay")
    evidence_authorization = _dummy_evidence_authorization()
    original_current = prereg.assert_entry_evidence_authorization_current
    prereg.assert_entry_evidence_authorization_current = lambda value: value
    replay_current = getattr(
        replay, "assert_entry_evidence_authorization_current", None
    )
    dataset_current = None
    if replay_current is not None:
        replay.assert_entry_evidence_authorization_current = lambda value: value
    try:
        journal = replay.genesis_research_ledger_journal(
            evidence_authorization,
            policy_id="FROZEN_GATE_POLICY",
            fee_path=3,
        )
        replay.validate_research_ledger_journal(
            journal, authorization=evidence_authorization
        )
        tip = journal.tip_ledger
        assert len(journal.transitions) == 1
        assert tip.authorization_sha256 == prereg.stable_hash(
            evidence_authorization.to_dict()
        )
        assert tip.sessions_sha256_newline == (
            evidence_authorization.sessions_sha256_newline
        )
        assert tip.session == evidence_authorization.sessions[0]
        assert tip.session_start_equity_micros == 10_000_000_000
        assert tip.cash_micros == 10_000_000_000
        assert tip.realized_session_pnl_micros == 0
        assert tip.position is None and tip.pending_intent_id is None
        assert tip.sequence == 0 and tip.last_decision_time_ns is None
        assert tip.prior_transition_sha256 == "0" * 64
        genesis_transition = journal.transitions[0]
        assert genesis_transition.event_schema_version == (
            replay.ResearchLedgerControlEventV1.SCHEMA_VERSION
        )
        assert type(genesis_transition.event_payload) is dict
        assert genesis_transition.event_payload

        features = importlib.import_module("v4.research.pathd_entry_features")
        dataset_module = importlib.import_module(
            "v4.research.pathd_entry_dataset"
        )
        dataset_current = getattr(
            dataset_module, "assert_entry_evidence_authorization_current", None
        )
        if dataset_current is not None:
            dataset_module.assert_entry_evidence_authorization_current = (
                lambda value: value
            )
        fill_module = importlib.import_module(
            "v4.path_d.execution.research_fill_law"
        )
        coverage_snapshot = features.source_neutral_snapshot_from_causal_inputs(
            **_causal_inputs()
        )
        coverage_example = dataset_module.build_entry_example(
            coverage_snapshot,
            future_path=_valid_future_path(
                dataset_module,
                coverage_snapshot,
                nonce="frame-coverage",
            ),
            fill_law=fill_module.research_fill_law_from_preregistration(
                prereg.preregistration_payload()[0]
            ),
        )
        coverage_dataset = _sealed_dataset_for_authorization(
            dataset_module,
            evidence_authorization,
            coverage_example,
        )
        coverage = replay.entry_frame_coverage_from_verified_example(
            coverage_example,
            dataset=coverage_dataset,
            authorization=evidence_authorization,
            journal=journal,
        )
        assert type(coverage) is replay.EntryFrameCoverageV1
        assert coverage.disposition == "ACTION_DECISION"
        assert tuple(coverage.reason_codes) == ()
        covered_journal = replay.append_entry_frame_coverage(
            journal,
            coverage,
            dataset=coverage_dataset,
            authorization=evidence_authorization,
        )
        coverage_payloads = [
            transition.event_payload
            for transition in covered_journal.transitions
            if transition.event_schema_version
            == replay.EntryFrameCoverageV1.SCHEMA_VERSION
        ]
        assert [row["example_sha256"] for row in coverage_payloads] == [
            example.canonical_sha256()
            for example in coverage_dataset.examples
        ]
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.validate_research_ledger_journal_against_dataset(
                journal,
                authorization=evidence_authorization,
                dataset=coverage_dataset,
            )
        forged_coverage = replace(
            coverage,
            disposition="STATE_INELIGIBLE",
            reason_codes=("FORGED_STATE_INELIGIBLE",),
            coverage_sha256="0" * 64,
        )
        forged_coverage = replace(
            forged_coverage,
            coverage_sha256=prereg.stable_hash(
                {
                    field.name: getattr(forged_coverage, field.name)
                    for field in fields(forged_coverage)
                    if field.name != "coverage_sha256"
                }
            ),
        )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.append_entry_frame_coverage(
                journal,
                forged_coverage,
                dataset=coverage_dataset,
                authorization=evidence_authorization,
            )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.validate_research_ledger_journal(
                replace(
                    journal,
                    transitions=(
                        replace(
                            genesis_transition,
                            event_payload={"forged_hash_only_event": True},
                        ),
                    ),
                ),
                authorization=evidence_authorization,
            )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.validate_research_ledger_journal(
                replace(
                    journal,
                    tip_ledger=replace(tip, cash_micros=tip.cash_micros - 1),
                ),
                authorization=evidence_authorization,
            )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.reconstruct_entry_policy_evaluation_from_journal(
                journal,
                authorization=evidence_authorization,
                dataset_sha256="a" * 64,
                sessions=evidence_authorization.sessions,
                evidence_role=evidence_authorization.role,
                outer_fold=evidence_authorization.outer_fold,
                inner_fold=evidence_authorization.inner_fold,
                policy_id="FROZEN_GATE_POLICY",
            )
        terminal_journal = replay.mark_research_session_terminal(
            journal,
            authorization=evidence_authorization,
            reason_code="PATHD_RESEARCH_SESSION_COMPLETE",
        )
        replay.validate_research_ledger_journal(
            terminal_journal, authorization=evidence_authorization
        )
        evaluation = replay.reconstruct_entry_policy_evaluation_from_journal(
            terminal_journal,
            authorization=evidence_authorization,
            dataset_sha256="a" * 64,
            sessions=evidence_authorization.sessions,
            evidence_role=evidence_authorization.role,
            outer_fold=evidence_authorization.outer_fold,
            inner_fold=evidence_authorization.inner_fold,
            policy_id="FROZEN_GATE_POLICY",
        )
        assert type(evaluation) is dict
        assert evaluation["session_pnl_micros"] == [0]
        assert evaluation["completed_trade_ids"] == []
        assert evaluation["completed_trade_sessions"] == []
        assert evaluation["zero_trade_sessions"] == ["2026-01-02"]
        assert evaluation["holdout_caveat"] == prereg.HOLDOUT_CAVEAT
        prereg._validate_policy_evaluation_dict(
            evaluation,
            authorization=evidence_authorization,
            dataset_sha256="a" * 64,
            sessions=evidence_authorization.sessions,
            role=evidence_authorization.role,
            outer_fold=1,
            inner_fold=None,
        )

        def reseal_policy_evaluation(value: dict[str, Any]) -> dict[str, Any]:
            value["session_coverage_sha256"] = prereg.stable_hash(
                {
                    "sessions": value["sessions"],
                    "session_pnl_micros": value["session_pnl_micros"],
                    "session_terminal_journal_sha256s": value[
                        "session_terminal_journal_sha256s"
                    ],
                    "completed_trade_ids": value["completed_trade_ids"],
                    "completed_trade_sessions": value[
                        "completed_trade_sessions"
                    ],
                    "zero_trade_sessions": value["zero_trade_sessions"],
                    "execution_trace_root_sha256": value[
                        "execution_trace_root_sha256"
                    ],
                }
            )
            semantic = dict(value)
            semantic.pop("result_sha256", None)
            value["result_sha256"] = prereg.stable_hash(semantic)
            return value

        pnl_forgery = copy.deepcopy(evaluation)
        pnl_forgery["session_pnl_micros"] = [1]
        trade_forgery = copy.deepcopy(evaluation)
        trade_forgery["completed_trade_ids"] = ["fabricated-trade"]
        trade_forgery["completed_trade_sessions"] = ["2026-01-02"]
        trade_forgery["zero_trade_sessions"] = []
        metric_forgery = copy.deepcopy(evaluation)
        metric_forgery["metrics"] = {"fabricated_metric": 1.0}
        caveat_forgery = copy.deepcopy(evaluation)
        caveat_forgery["holdout_caveat"] = "FORGED_HOLDOUT_CAVEAT"
        for forged in (
            reseal_policy_evaluation(pnl_forgery),
            reseal_policy_evaluation(trade_forgery),
            reseal_policy_evaluation(metric_forgery),
            reseal_policy_evaluation(caveat_forgery),
        ):
            with pytest.raises((RuntimeError, TypeError, ValueError)):
                prereg._validate_policy_evaluation_dict(
                    forged,
                    authorization=evidence_authorization,
                    dataset_sha256="a" * 64,
                    sessions=evidence_authorization.sessions,
                    role=evidence_authorization.role,
                    outer_fold=1,
                    inner_fold=None,
                )
        with pytest.raises(TypeError):
            replay.prepare_research_execution(
                intent_kind="BUY",
                context=None,
                ledger=tip,
                contract=intent.contract,
                reference_bid_micros=2_000_000,
                reference_ask_micros=2_100_000,
                arrival_time_ns=pd.Timestamp("2026-01-02T15:01:00Z").value,
                law=None,
                governor=DeterministicGovernor(),
                reason_code="FORBIDDEN_RAW_REPLAY_ARGUMENTS",
            )
    finally:
        prereg.assert_entry_evidence_authorization_current = original_current
        if replay_current is not None:
            replay.assert_entry_evidence_authorization_current = replay_current
        if dataset_current is not None:
            dataset_module.assert_entry_evidence_authorization_current = dataset_current


def test_corpus_hash_precedes_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = importlib.import_module("v4.research.pathd_entry_dataset")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    file_path = corpus / "session.parquet"
    payload = b"verified-parquet-fixture"
    file_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    authorization = _dummy_authorization("2025-08-01")
    order: list[str] = []
    decoded: list[bool] = []

    monkeypatch.setattr(dataset, "CORPUS_ROOT", corpus)
    monkeypatch.setattr(dataset, "assert_fit_authorization_current", lambda value: order.append("authorization") or value)
    monkeypatch.setattr(
        dataset,
        "_resolve_authorized_session_files",
        lambda value, session: order.append("resolve") or ({"path": file_path, "sha256": digest, "size": len(payload)},),
    )
    original_hash = dataset._stream_sha256_and_size
    monkeypatch.setattr(dataset, "_stream_sha256_and_size", lambda path: order.append("hash") or original_hash(path))
    monkeypatch.setattr(dataset, "_decode_verified_entry_files", lambda rows: decoded.append(True) or {"rows": len(rows)})
    result = dataset.read_verified_entry_session(authorization, session="2025-08-01")
    assert result.payload == {"rows": 1}
    assert order == ["authorization", "resolve", "hash"]
    assert decoded == [True]

    decoded.clear()
    monkeypatch.setattr(
        dataset,
        "_resolve_authorized_session_files",
        lambda value, session: ({"path": file_path, "sha256": "0" * 64, "size": len(payload)},),
    )
    with pytest.raises(RuntimeError):
        dataset.read_verified_entry_session(authorization, session="2025-08-01")
    assert decoded == []
    with pytest.raises(RuntimeError):
        dataset.read_verified_entry_session(authorization, session="2025-08-04")
    assert decoded == []

    link = corpus / "linked.parquet"
    link.symlink_to(file_path)
    monkeypatch.setattr(
        dataset,
        "_resolve_authorized_session_files",
        lambda value, session: ({"path": link, "sha256": digest, "size": len(payload)},),
    )
    with pytest.raises(RuntimeError):
        dataset.read_verified_entry_session(authorization, session="2025-08-01")
    assert decoded == []


def test_result_envelope_and_tamper(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = importlib.import_module("v4.path_d.execution.research_replay")
    fill_module = importlib.import_module("v4.path_d.execution.research_fill_law")
    features = importlib.import_module("v4.research.pathd_entry_features")
    dataset_module = importlib.import_module("v4.research.pathd_entry_dataset")
    payload = prereg.preregistration_payload()[0]
    law = fill_module.research_fill_law_from_preregistration(payload)
    authorization = _dummy_evidence_authorization()
    snapshot = features.source_neutral_snapshot_from_causal_inputs(**_causal_inputs())
    example = dataset_module.build_entry_example(
        snapshot,
        future_path=_valid_future_path(
            dataset_module, snapshot, nonce="result-envelope"
        ),
        fill_law=law,
    )
    dataset = _sealed_dataset_for_authorization(
        dataset_module, authorization, example
    )
    monkeypatch.setattr(
        prereg,
        "assert_entry_evidence_authorization_current",
        lambda value: value,
    )
    if hasattr(replay, "assert_entry_evidence_authorization_current"):
        monkeypatch.setattr(
            replay,
            "assert_entry_evidence_authorization_current",
            lambda value: value,
        )
    original_dataset_validator = dataset_module.validate_entry_evidence_dataset
    monkeypatch.setattr(
        dataset_module,
        "validate_entry_evidence_dataset",
        lambda value, *, authorization: value,
    )
    if hasattr(replay, "validate_entry_evidence_dataset"):
        monkeypatch.setattr(
            replay,
            "validate_entry_evidence_dataset",
            lambda value, *, authorization: value,
        )
    envelope = replay.seal_research_result(
        authorization=authorization,
        dataset=dataset,
        fill_law=law,
        payload={"status": "PASS", "metric": 1.0},
    )
    replay.validate_research_result(
        envelope,
        authorization=authorization,
        dataset=dataset,
        fill_law=law,
    )
    assert envelope.holdout_open_count == 0
    assert envelope.holdout_caveat == prereg.HOLDOUT_CAVEAT
    assert envelope.claim_boundary == prereg.CLAIM_BOUNDARY
    assert envelope.fill_law_hash == law.fill_law_hash

    for name in fields(envelope):
        if name.name == "schema_version":
            changed = replace(envelope, schema_version="wrong")
        elif name.name == "holdout_open_count":
            changed = replace(envelope, holdout_open_count=1)
        elif name.name.endswith("sha256") or name.name.endswith("hash"):
            changed = replace(envelope, **{name.name: "0" * 64})
        elif name.name == "payload":
            changed = replace(envelope, payload={"status": "PASS", "metric": float("nan")})
        else:
            continue
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.validate_research_result(
                changed,
                authorization=authorization,
                dataset=dataset,
                fill_law=law,
            )

    stale = replace(authorization, sessions_sha256_newline="f" * 64)
    stale_validator = lambda value: (_ for _ in ()).throw(RuntimeError("stale"))
    monkeypatch.setattr(
        prereg, "assert_entry_evidence_authorization_current", stale_validator
    )
    if hasattr(replay, "assert_entry_evidence_authorization_current"):
        monkeypatch.setattr(
            replay, "assert_entry_evidence_authorization_current", stale_validator
        )
    with pytest.raises(RuntimeError):
        replay.validate_research_result(
            envelope, authorization=stale, dataset=dataset, fill_law=law
        )
    monkeypatch.setattr(
        dataset_module,
        "validate_entry_evidence_dataset",
        original_dataset_validator,
    )


def test_fit_gate_tamper_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload, assignments, lineage = prereg.preregistration_payload()
    prereg.validate_preregistration_payload(payload, assignments, lineage)
    altered = copy.deepcopy(payload)
    altered["source_hash_policy"]["fit_session_roles"]["outer_weights"] = "caller.sessions"
    with pytest.raises(ValueError):
        prereg.validate_preregistration_payload(altered, assignments, lineage)
    altered = copy.deepcopy(payload)
    altered["source_hash_policy"]["receipt_contracts"]["required_test_commands"][0]["argv"] = ["true"]
    with pytest.raises(ValueError):
        prereg.validate_preregistration_payload(altered, assignments, lineage)
    altered = copy.deepcopy(payload)
    altered["source_hash_policy"]["entry_future_api_contract"]["functions"].pop()
    with pytest.raises(ValueError):
        prereg.validate_preregistration_payload(altered, assignments, lineage)

    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    link = prereg.REPO_ROOT / "v4/tests/.pathd-frozen-gate-link.py"
    try:
        link.symlink_to(outside)
        with pytest.raises(RuntimeError):
            prereg._canonical_repo_regular_file("v4/tests/.pathd-frozen-gate-link.py")
    finally:
        link.unlink(missing_ok=True)
    with pytest.raises(RuntimeError):
        prereg._canonical_repo_regular_file("../outside.py")

    authorization = _dummy_authorization()
    monkeypatch.setattr(prereg, "assert_entry_fit_ready", lambda **kwargs: authorization)
    release_path = prereg.REPO_ROOT / (
        "v4/audit/autoresearch/"
        "protocol101_pathd_entry_exit_model_research_corrected_v3_2_"
        "executable_2026_08_01/claude_verification_release.json"
    )
    if release_path.is_file():
        prereg._assert_corrected_v32_release_if_present()
    else:
        with pytest.raises(
            RuntimeError,
            match="STOP_FOR_CLAUDE_VERIFICATION: corrected-v3.2 release chain is incomplete",
        ):
            prereg.assert_fit_authorization_current(authorization)
    # Test-scoped verified-release fixture: the production guard is exercised
    # above and remains fail-closed; this fixture reaches the original stale-auth
    # governance assertions without creating a real Claude release artifact.
    monkeypatch.setattr(
        prereg, "_assert_corrected_v32_release_if_present", lambda: None
    )
    assert prereg.assert_fit_authorization_current(authorization) == authorization
    for field_name, value in (
        ("role", "full_weights"),
        ("sessions", ("2025-08-04",)),
        ("machinery_receipt_sha256", "0" * 64),
    ):
        with pytest.raises(RuntimeError):
            prereg.assert_fit_authorization_current(replace(authorization, **{field_name: value}))

    with pytest.raises(ValueError):
        prereg.strict_json_loads('{"duplicate":1,"duplicate":2}')
    for nonfinite in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError):
            prereg.strict_json_loads('{"value":' + nonfinite + "}")
    altered = copy.deepcopy(payload)
    altered["holdout_open_count"] = False
    with pytest.raises(ValueError):
        prereg.validate_preregistration_payload(altered, assignments, lineage)
    fit_environment = prereg.entry_fit_environment_spec()
    assert fit_environment["active_python_hash_probe"] == {
        "input": "pathd-entry-fit-probe-v1",
        "expected_hash": -4808614846103707838,
    }

    holdout = importlib.import_module("v4.research.pathd_holdout_gate")
    assert holdout.PRE_HOLDOUT_PACKET_PATH == prereg.PRE_HOLDOUT_PACKET_PATH
    assert holdout.PROTECTED_HOLDOUT_ROOT == prereg.PROTECTED_HOLDOUT_ROOT
    assert holdout.HOLDOUT_ACCESS_RECEIPT_PATH == prereg.HOLDOUT_ACCESS_RECEIPT_PATH
    assert holdout.HOLDOUT_RESULT_PATH == prereg.HOLDOUT_RESULT_PATH
    assert holdout.HOLDOUT_SEAL_RECEIPT_PATH == prereg.HOLDOUT_SEAL_RECEIPT_PATH
    assert holdout.HOLDOUT_ABORT_RECEIPT_PATH == prereg.HOLDOUT_ABORT_RECEIPT_PATH
    assert _signature_rows(holdout.execute_protected_holdout_once) == []
    assert holdout.execute_protected_holdout_once.__closure__ is None
    assert not hasattr(holdout, "begin_protected_holdout_once")
    assert not hasattr(holdout, "_begin_protected_holdout_once")
    assert not hasattr(holdout, "abort_protected_holdout")
    assert "ActiveProtectedHoldoutAuthorizationV1" not in holdout.__all__
    assert "begin_protected_holdout_once" not in holdout.__all__
    assert "abort_protected_holdout" not in holdout.__all__
    assert "load_authorized_protected_holdout" not in holdout.__all__
    assert "seal_protected_holdout_result" not in holdout.__all__
    assert not hasattr(holdout, "load_authorized_protected_holdout")
    assert not hasattr(holdout, "seal_protected_holdout_result")
    with pytest.raises(TypeError):
        holdout.ActiveProtectedHoldoutAuthorizationV1(
            object(),
            transaction_token_sha256="0" * 64,
            lock_fd=-1,
            semantic={},
        )
    with pytest.raises(TypeError):
        holdout.execute_protected_holdout_once(object())
    for call in (
        lambda: holdout._load_authorized_protected_holdout(
            object(), "0" * 64
        ),
        lambda: holdout._seal_protected_holdout_result(
            object(), object(), "0" * 64
        ),
        lambda: holdout._abort_protected_holdout(
            object(), "0" * 64, reason_code="OPERATOR_ABORT"
        ),
    ):
        with pytest.raises(holdout.ProtectedHoldoutError):
            call()

    holdout_root = tmp_path / "holdout"
    monkeypatch.setattr(holdout, "AUDIT_ROOT", tmp_path)
    monkeypatch.setattr(
        holdout, "PRE_HOLDOUT_PACKET_PATH", tmp_path / "complete_pre_holdout_packet.json"
    )
    monkeypatch.setattr(holdout, "PROTECTED_HOLDOUT_ROOT", holdout_root)
    monkeypatch.setattr(holdout, "HOLDOUT_LOCK_PATH", holdout_root / ".transaction.lock")
    monkeypatch.setattr(
        holdout, "HOLDOUT_ACCESS_RECEIPT_PATH", holdout_root / "holdout_access_receipt.json"
    )
    monkeypatch.setattr(
        holdout, "HOLDOUT_RESULT_PATH", holdout_root / "holdout_result.json"
    )
    monkeypatch.setattr(
        holdout,
        "HOLDOUT_SEAL_RECEIPT_PATH",
        holdout_root / "holdout_seal_receipt.json",
    )
    monkeypatch.setattr(
        holdout,
        "HOLDOUT_ABORT_RECEIPT_PATH",
        holdout_root / "holdout_abort_receipt.json",
    )
    monkeypatch.setattr(
        holdout, "HOLDOUT_TRACE_PATH", holdout_root / "holdout_trace.jsonl"
    )
    monkeypatch.setattr(
        holdout,
        "HOLDOUT_EVALUATOR_RECEIPT_PATH",
        holdout_root / "holdout_evaluator_receipt.json",
    )
    state = holdout.inspect_protected_holdout_state()
    assert state.state == holdout.UNOPENED and state.holdout_open_count == 0

    execute_source = inspect.getsource(holdout.execute_protected_holdout_once)
    transaction_source = inspect.getsource(
        holdout._execute_protected_holdout_transaction
    )
    load_source = inspect.getsource(holdout._load_authorized_protected_holdout)
    seal_source = inspect.getsource(holdout._seal_protected_holdout_result)
    recovery_source = inspect.getsource(holdout.recover_protected_holdout_after_crash)
    assert execute_source.index(
        "write_json_exclusive_durable(HOLDOUT_ACCESS_RECEIPT_PATH"
    ) < execute_source.index("authorization = ActiveProtectedHoldoutAuthorizationV1(")
    assert execute_source.index(
        "authorization = ActiveProtectedHoldoutAuthorizationV1("
    ) < execute_source.index("return _execute_protected_holdout_transaction(")
    assert "transaction_token = uuid.uuid4().hex + uuid.uuid4().hex" in execute_source
    assert "authorization, transaction_token" in execute_source
    assert "return authorization" not in execute_source
    assert transaction_source.index(
        "_load_authorized_protected_holdout("
    ) < transaction_source.index("evaluation = evaluator(authorization, dataset)")
    assert transaction_source.index(
        "evaluation = evaluator(authorization, dataset)"
    ) < transaction_source.index(
        "_seal_protected_holdout_result("
    )
    assert "authorization, transaction_token" in transaction_source
    assert "_validate_live_authorization(authorization, transaction_token)" in load_source
    assert "_validate_live_authorization(authorization, transaction_token)" in seal_source
    assert "_execution_scope_active" not in load_source
    assert "_execution_scope_active" not in seal_source
    assert seal_source.index(
        "write_json_exclusive_durable(HOLDOUT_RESULT_PATH"
    ) < seal_source.index(
        "write_json_exclusive_durable(HOLDOUT_SEAL_RECEIPT_PATH"
    )
    assert "load_authorized_protected_holdout" not in recovery_source
    assert "_seal_receipt_payload" not in recovery_source
    assert "HOLDOUT_SEAL_RECEIPT_PATH" not in recovery_source
    assert '_recover_abort(access, reason_code="CORRUPT_BURNED")' in recovery_source
