from __future__ import annotations

import json
from pathlib import Path

import pytest

from v4.model.protocol101_repair_artifacts import (
    REQUIRED_MANIFEST_HASHES,
    Protocol101ArtifactHashMismatchError,
    semantic_payload_hashes,
    verify_replay_packet,
    write_immutable_replay_packet,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION,
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    Protocol101MixedSimulatorVersionError,
)


def _payloads() -> dict:
    candidate = {
        "split": "validation",
        "session": "2025-01-02",
        "decision_time_ns": 1,
        "contract_id": "c1",
        "right": "C",
        "canonical_strike_slot": 0,
        "policy_index": 0,
        "entry_ask": 2.0,
        "score": 1.0,
        "raw_label_pnl_after_campaign_fee": 7.0,
        "label_mid_pnl_before_campaign_fee": 10.0,
        "label_realized_exit_time_ns": 3,
        "label_source_exit_quote_time_ns": 2,
        "label_exit_quote_age_ms": 0.000001,
        "label_exit_reason_code": 3,
        "label_executable_exit_bid": 2.1,
        "label_policy_deadline_ns": 3,
        "label_invalid_reason_code": 0,
        "feature_hash": "f" * 64,
        "source_quote_time_ns": 1,
        "source_context_time_ns": 1,
        "strategy": "synthetic",
        "metadata": {},
        "source_simulator_version": (
            PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
        ),
    }
    trade = {
        **candidate,
        "fold": "f1",
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    }
    quote_report = {
        "schema_version": "Protocol101ExitQuoteAgeReportV1",
        "gate": False,
        "rejection_threshold_ms": None,
        "groups": [],
    }
    return {
        "candidate_intents.jsonl": [candidate],
        "exit_quote_age_report.json": quote_report,
        "fold_metrics.json": {"status": "synthetic_non_economic"},
        "pooled_metrics.json": {"status": "synthetic_non_economic"},
        "session_metrics.json": {"status": "synthetic_non_economic"},
        "skipped_events.jsonl": [],
        "trades.jsonl": [trade],
    }


def _manifest(payloads: dict) -> dict:
    result = {name: "a" * 64 for name in REQUIRED_MANIFEST_HASHES}
    result.update(semantic_payload_hashes(payloads))
    result.update(
        {
            "simulator_version": (
                PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
            ),
            "attempt_id": "synthetic_test",
        }
    )
    return result


def test_atomic_manifest_last_hashes_and_resume(tmp_path: Path) -> None:
    payloads = _payloads()
    manifest = _manifest(payloads)
    packet = tmp_path / "packet"
    created = write_immutable_replay_packet(
        packet, payloads=payloads, manifest_fields=manifest
    )
    assert created.status == "created_complete_packet"
    assert created.write_order[-1] == "manifest.json"
    verified = verify_replay_packet(packet)
    assert verified["status"] == "complete_pending_independent_acceptance"
    resumed = write_immutable_replay_packet(
        packet, payloads=payloads, manifest_fields=manifest
    )
    assert resumed.status == "verified_existing_complete_packet_skipped"
    assert resumed.write_order == ()


def test_partial_packet_moves_to_void_with_receipt(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "candidate_intents.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    payloads = _payloads()
    result = write_immutable_replay_packet(
        packet, payloads=payloads, manifest_fields=_manifest(payloads)
    )
    voided = Path(result.voided_partial_packet or "")
    assert voided.parent == tmp_path / "void_outputs"
    receipt = json.loads(
        (voided / "void_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "VOID_PARTIAL_PACKET"
    verify_replay_packet(packet)


def test_complete_hash_mismatch_fails_without_overwrite(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    payloads = _payloads()
    manifest = _manifest(payloads)
    write_immutable_replay_packet(
        packet, payloads=payloads, manifest_fields=manifest
    )
    path = packet / "fold_metrics.json"
    path.write_text('{"tampered":true}\\n', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(Protocol101ArtifactHashMismatchError):
        write_immutable_replay_packet(
            packet, payloads=payloads, manifest_fields=manifest
        )
    assert path.read_bytes() == before


def test_mixed_simulator_versions_fail_before_output(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["candidate_intents.jsonl"][0][
        "source_simulator_version"
    ] = PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION
    with pytest.raises(Protocol101MixedSimulatorVersionError):
        write_immutable_replay_packet(
            tmp_path / "packet",
            payloads=payloads,
            manifest_fields=_manifest(payloads),
        )
    assert not (tmp_path / "packet").exists()
