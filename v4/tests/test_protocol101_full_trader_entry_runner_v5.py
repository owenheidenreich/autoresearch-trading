from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from v4.model import protocol101_scoped_stage1_hgb as core
from v4.model.protocol101_canonical_stage1_contract import HYPOTHESES
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
)
from v4.scripts import run_protocol101_full_trader_entry_runner_v5_validation
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as runner
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner_v2 as durable


class FakeModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values[:, 0], dtype=float)


def _base_decision(
    minute: int,
    *,
    session: str = "2025-01-02",
) -> core.CanonicalDecision:
    decision_time = pd.Timestamp(
        f"{session} 15:{30 + minute:02d}:00",
        tz="UTC",
    )
    labels = np.asarray([-30.0, 20.0, 80.0], dtype=float)
    asks = np.asarray([2.0, 2.5, 3.0], dtype=float)
    features = np.asarray(
        [
            [float(candidate + minute)] * len(HYPOTHESES["H0"])
            for candidate in range(3)
        ],
        dtype=float,
    )
    return core.CanonicalDecision(
        session=session,
        decision_time=decision_time,
        features=features,
        labels=labels,
        mid_labels=labels + 10.0,
        entry_asks=asks,
        offsets=np.asarray([-5.0, 0.0, 5.0], dtype=float),
        rights=np.asarray(["P", "C", "C"], dtype=object),
        contract_ids=np.asarray(
            [
                f"{session}-P-{minute}",
                f"{session}-ATM-{minute}",
                f"{session}-C-{minute}",
            ],
            dtype=object,
        ),
        strike_indices=np.asarray([9, 10, 11], dtype=int),
        right_indices=np.asarray([1, 0, 0], dtype=int),
    )


def _repaired_decision(
    minute: int,
    *,
    session: str = "2025-01-02",
) -> core.RepairedCanonicalDecision:
    base = _base_decision(minute, session=session)
    source = int(base.decision_time.value + 60 * 1_000_000_000)
    deadline = int(base.decision_time.value + 10 * 60 * 1_000_000_000)
    return core.RepairedCanonicalDecision(
        base=base,
        realized_exit_time_ns=np.asarray([deadline] * 3, dtype=np.int64),
        source_exit_quote_time_ns=np.asarray([source] * 3, dtype=np.int64),
        exit_quote_age_ms=np.asarray([540_000.0] * 3, dtype=float),
        exit_reason_codes=np.asarray([3] * 3, dtype=np.uint8),
        executable_exit_bids=base.entry_asks + base.labels / 100.0,
        policy_deadline_ns=np.asarray([deadline] * 3, dtype=np.int64),
        invalid_reason_codes=np.zeros(3, dtype=np.uint8),
        canonical_strike_slots=np.asarray([9, 10, 11], dtype=np.int64),
        source_quote_time_ns=np.asarray(
            [int(base.decision_time.value)] * 3,
            dtype=np.int64,
        ),
        source_context_time_ns=np.asarray(
            [int(base.decision_time.value - 60 * 1_000_000_000)] * 3,
            dtype=np.int64,
        ),
    )


def _candidate(
    minute: int,
    *,
    realized_after_minutes: int,
    contract_id: str,
) -> SerialCandidateV5:
    decision = pd.Timestamp(
        f"2025-01-02 15:{30 + minute:02d}:00",
        tz="UTC",
    )
    source = int(decision.value + 60 * 1_000_000_000)
    realized = int(
        decision.value + realized_after_minutes * 60 * 1_000_000_000
    )
    return SerialCandidateV5(
        split="validation",
        fold="fold1",
        session="2025-01-02",
        decision_time_ns=int(decision.value),
        contract_id=contract_id,
        right="C",
        canonical_strike_slot=10 + minute,
        policy_index=0,
        entry_ask=2.0,
        score=1.0,
        raw_label_pnl_after_campaign_fee=47.0,
        label_mid_pnl_before_campaign_fee=60.0,
        label_realized_exit_time_ns=realized,
        label_source_exit_quote_time_ns=source,
        label_exit_quote_age_ms=(realized - source) / 1_000_000.0,
        label_exit_reason_code=3,
        label_executable_exit_bid=2.5,
        label_policy_deadline_ns=realized,
        feature_hash=core.CONTRACT_ID,
        source_quote_time_ns=int(decision.value),
        source_context_time_ns=int(
            decision.value - 60 * 1_000_000_000
        ),
        strategy="test",
    )


def _patch_no_fit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        core,
        "fit_model",
        lambda decisions, noise_model, config: (
            FakeModel(),
            {
                "fit_candidates_before_cap": sum(
                    len(item.labels) for item in decisions
                ),
                "fit_candidates_after_cap": sum(
                    len(item.labels) for item in decisions
                ),
                "target_min": -1.0,
                "target_max": 1.0,
                "target_mean": 0.0,
                "feature_names": list(HYPOTHESES[config.hypothesis]),
                "feature_count": len(HYPOTHESES[config.hypothesis]),
                "contract_id": core.CONTRACT_ID,
                "model_family": "fake_no_fit",
                "model_iterations": 0,
            },
        ),
    )
    monkeypatch.setattr(
        core,
        "score_noise_epsilon",
        lambda *args, **kwargs: (
            0.0,
            {
                "method": "synthetic",
                "sample_count": 1,
                "p50_abs_score_drift": 0.0,
                "p95_abs_score_drift": 0.0,
                "p99_abs_score_drift": 0.0,
            },
        ),
    )

    def fake_scores(
        model: FakeModel,
        decisions: list[core.CanonicalDecision],
        **kwargs: object,
    ) -> list[np.ndarray]:
        del model, kwargs
        return [
            np.asarray([0.0, 0.5, 1.0 + index / 10.0], dtype=float)
            for index, _item in enumerate(decisions)
        ]

    monkeypatch.setattr(core, "score_decisions", fake_scores)


def test_fresh_public_unit_uses_v5_for_every_replay_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_fit_model(monkeypatch)
    calls = 0
    actual = core.simulate_serial_candidates_v5

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(core, "simulate_serial_candidates_v5", counted)
    monkeypatch.setattr(
        core,
        "simulate_serial_candidates",
        lambda *args, **kwargs: pytest.fail("fresh path reached v4"),
    )
    repaired = [_repaired_decision(index) for index in range(3)]
    config = core.HGBUnitConfig(
        hypothesis="H0",
        policy_index=0,
        seed=42,
    )
    _model, result = core.run_hgb_unit_v5(
        fit_decisions=repaired,
        calibration_decisions=repaired,
        validation_decisions=repaired,
        noise_model=DivergenceNoiseModel(
            samples={},
            feature_family={},
            source_path="test",
        ),
        config=config,
        fold="fold1",
    )
    assert calls >= 8
    assert result["simulator_version"] == (
        PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
    )
    assert all(
        row["simulator_version"]
        == PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
        for row in result["calibration"]["threshold_sweep"]
    )
    assert {
        row["simulator_version"]
        for row in result["validation"]["fee_sensitivity"].values()
    } == {PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION}
    assert {
        row["metrics"]["simulator_version"]
        for row in result["validation"]["noise_diagnostics"].values()
    } == {PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION}


def test_two_clocks_and_alpha_firewall_survive_fresh_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_fit_model(monkeypatch)
    repaired = [_repaired_decision(index) for index in range(3)]
    _model, result = core.run_hgb_unit_v5(
        fit_decisions=repaired,
        calibration_decisions=repaired,
        validation_decisions=repaired,
        noise_model=DivergenceNoiseModel(
            samples={},
            feature_family={},
            source_path="test",
        ),
        config=core.HGBUnitConfig(
            hypothesis="H0",
            policy_index=0,
            seed=42,
        ),
        fold="fold1",
    )
    intent = result["validation"]["entry_intents"][0]
    trade = result["validation"]["trades"][0]
    assert intent["label_source_exit_quote_time_ns"] < (
        intent["label_realized_exit_time_ns"]
    )
    assert trade["label_source_exit_quote_time_ns"] == (
        intent["label_source_exit_quote_time_ns"]
    )
    assert trade["label_realized_exit_time_ns"] == (
        intent["label_realized_exit_time_ns"]
    )
    assert result["feature_names"] == list(HYPOTHESES["H0"])
    assert all("label" not in name for name in result["feature_names"])


def test_v5_realized_exit_releases_before_same_timestamp_decision() -> None:
    first = _candidate(
        0,
        realized_after_minutes=2,
        contract_id="first",
    )
    second = replace(
        _candidate(
            2,
            realized_after_minutes=2,
            contract_id="second",
        ),
        canonical_strike_slot=12,
    )
    trades, _state, metrics = core.replay_candidates_v5(
        [first, second],
        config=core.HGBUnitConfig(
            hypothesis="H0",
            policy_index=0,
            seed=42,
        ),
    )
    assert len(trades) == 2
    assert metrics["skipped"]["overlap"] == 0


def test_fee_sensitivity_applies_alternate_fee_once() -> None:
    candidate = _candidate(
        0,
        realized_after_minutes=2,
        contract_id="fee",
    )
    config = core.HGBUnitConfig(
        hypothesis="H0",
        policy_index=0,
        seed=42,
        fee=3.0,
    )
    for fee, expected in ((2.6, 47.4), (3.0, 47.0), (4.0, 46.0)):
        trades, state, metrics = core.replay_candidates_v5_at_fee(
            [candidate],
            config=config,
            fee=fee,
        )
        assert trades[0].raw_label_pnl_after_campaign_fee == pytest.approx(
            expected
        )
        assert metrics["net_pnl"] == pytest.approx(expected)
        assert state.semantics["campaign_round_trip_fee_dollars"] == fee


def test_fresh_dry_run_is_truthful_and_keeps_deferred_blockers() -> None:
    args = Namespace(
        mode="dry-run",
        out_dir=runner.DEFAULT_FRESH_OUT,
        readiness=runner.DEFAULT_READINESS,
        hypothesis="H0",
        owner_approved_plumbing_smoke=False,
        owner_approved_offline_training=False,
        force=False,
        smoke_rows_per_session=20,
    )
    plan = runner.fresh_runner_plan(args)
    assert plan["status"] == "v5_core_ready_pending_independent_acceptance"
    assert plan["core_blockers"] == []
    assert plan["campaign_ready"] is False
    assert plan["training_authorized"] is False
    assert "G1_G8_aggregation" in plan["deferred_stack_blockers"]
    assert plan["identity_receipt"]["status"] == "PASS"
    assert plan["input_receipt"]["status"] == "PASS"


def test_call_graph_validation_detects_injected_v4_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = (
        run_protocol101_full_trader_entry_runner_v5_validation._called_names
    )

    def injected(function: object) -> set[str]:
        names = actual(function)
        if function is core.replay_candidates_v5:
            names.discard("simulate_serial_candidates_v5")
            names.add("simulate_serial_candidates")
        return names

    monkeypatch.setattr(
        run_protocol101_full_trader_entry_runner_v5_validation,
        "_called_names",
        injected,
    )
    result = (
        run_protocol101_full_trader_entry_runner_v5_validation
        .validate_call_graph()
    )
    assert result["status"] == "FAIL"
    assert (
        result["predicates"]["v5_replay_terminates_in_v5_simulator"]
        is False
    )


def test_old_campaign_unit_cannot_resume(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"campaign_namespace": "old_h0"}))
    with pytest.raises(RuntimeError, match="old or foreign"):
        durable.verify_resumable_unit(
            summary,
            hypothesis="H0",
            policy=0,
            seed=42,
            fold={"fold": 0, "fold_id": "fold1"},
            fit_sessions=["a"],
            calibration_sessions=["b"],
            validation_sessions=["c"],
            preregistration_payload={"preregistration_hash": "a" * 64},
            provenance={
                "campaign_contract_sha256": "b" * 64,
                "campaign_preregistration_sha256": "c" * 64,
            },
        )


def test_candidate_artifacts_are_v5_only() -> None:
    repaired = _repaired_decision(0)
    intents, _diagnostics = core.selection_rows_v5(
        [repaired],
        [np.asarray([0.0, 0.5, 1.0])],
        threshold=-1.0,
        epsilon=0.0,
        config=core.HGBUnitConfig(
            hypothesis="H0",
            policy_index=0,
            seed=42,
        ),
        split="validation",
        fold="fold1",
    )
    assert asdict(intents[0])["source_simulator_version"] == (
        PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
    )
    assert intents[0].label_source_exit_quote_time_ns < (
        intents[0].label_realized_exit_time_ns
    )


def test_duplicate_manifest_fails_before_any_model_work() -> None:
    scope = SimpleNamespace(
        sessions=[
            ("2025-01-02", Path("a")),
            ("2025-01-02", Path("b")),
        ],
        folds=[],
    )
    with pytest.raises(Exception) as error:
        runner.fresh_campaign_identity_receipt(scope)
    assert getattr(error.value, "blocker_code", "") == (
        "P101_ID_DUPLICATE_SESSION_MEMBERSHIP"
    )


def test_fresh_unit_packet_is_immutable_and_resume_hash_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_no_fit_model(monkeypatch)
    repaired = [_repaired_decision(index) for index in range(3)]
    _model, result = core.run_hgb_unit_v5(
        fit_decisions=repaired,
        calibration_decisions=repaired,
        validation_decisions=repaired,
        noise_model=DivergenceNoiseModel(
            samples={},
            feature_family={},
            source_path="test",
        ),
        config=core.HGBUnitConfig(
            hypothesis="H0",
            policy_index=0,
            seed=42,
        ),
        fold="fold1",
    )
    model_path = (
        tmp_path
        / runner.FRESH_CAMPAIGN_NAMESPACE
        / "units"
        / "H0"
        / "unit"
        / "model.pkl"
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fake-model")
    provenance = {
        "campaign_contract_sha256": "a" * 64,
        "campaign_preregistration_sha256": "b" * 64,
        "fold_governance_sha256": "c" * 64,
        "acceptance_registry_sha256": "d" * 64,
        "feature_contract_source_sha256": "e" * 64,
        "simulator_source_sha256": "f" * 64,
        "identity_contract_version": "identity-v1",
        "runner_core_source_sha256": "1" * 64,
        "runner_source_sha256": "2" * 64,
        "scientific_runner_source_sha256": "4" * 64,
    }
    prereg = {"preregistration_hash": "3" * 64}
    fold = {"fold": 0, "fold_id": "fold1"}
    summary = runner.commit_fresh_unit(
        model_path.parent,
        result=result,
        model_path=model_path,
        fold=fold,
        fit_sessions=["a"],
        calibration_sessions=["b"],
        validation_sessions=["c"],
        preregistration_payload=prereg,
        provenance=provenance,
    )
    verified = durable.verify_resumable_unit(
        model_path.parent / "summary.json",
        hypothesis="H0",
        policy=0,
        seed=42,
        fold=fold,
        fit_sessions=["a"],
        calibration_sessions=["b"],
        validation_sessions=["c"],
        preregistration_payload=prereg,
        provenance=provenance,
    )
    assert verified["replay_packet"]["manifest_sha256"] == (
        summary["replay_packet"]["manifest_sha256"]
    )
    packet = model_path.parent / "replay_packet"
    assert (packet / "manifest.json").is_file()
    assert json.loads((packet / "manifest.json").read_text())[
        "manifest_written_last"
    ]
    changed_provenance = {
        **provenance,
        "runner_core_source_sha256": "9" * 64,
    }
    with pytest.raises(RuntimeError, match="source provenance mismatch"):
        durable.verify_resumable_unit(
            model_path.parent / "summary.json",
            hypothesis="H0",
            policy=0,
            seed=42,
            fold=fold,
            fit_sessions=["a"],
            calibration_sessions=["b"],
            validation_sessions=["c"],
            preregistration_payload=prereg,
            provenance=changed_provenance,
        )
    model_path.write_bytes(b"tampered-model")
    with pytest.raises(RuntimeError, match="model hash mismatch"):
        durable.verify_resumable_unit(
            model_path.parent / "summary.json",
            hypothesis="H0",
            policy=0,
            seed=42,
            fold=fold,
            fit_sessions=["a"],
            calibration_sessions=["b"],
            validation_sessions=["c"],
            preregistration_payload=prereg,
            provenance=provenance,
        )
