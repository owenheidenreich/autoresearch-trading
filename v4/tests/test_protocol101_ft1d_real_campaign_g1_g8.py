from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.model import protocol101_stage1_gate_contract as gate
from v4.model.protocol101_scoped_stage1_hgb import (
    CanonicalDecision,
    RepairedCanonicalDecision,
)
from v4.scripts import run_protocol101_ft1d_real_campaign_g1_g8 as campaign


def _write_test_fitted_summary(path: Path) -> str:
    payload = {
        "schema_version": "Protocol101FreshEntryUnitSummaryV1",
        "unit": {
            "config": {"fee": 3.0},
            "validation": {
                "primary_noise_scale": 1.0,
                "primary_noise_seed": 42,
                "decision_count": 1,
                "candidate_count": 1,
                "metrics": {"sentinel": float("-inf")},
                "expected_calibration_error": 0.0,
                "calibration_observations": [],
                "fee_sensitivity": {},
                "fill_edge_band": {},
                "noise_diagnostics": {},
                "simulator_semantics": {},
                "diagnostics": [
                    {
                        "calibrated_confidence": 0.75,
                        "selected_label_before_fee": 4.0,
                    }
                ],
                "entry_intents": [],
                "trades": [],
                "skipped_events": [],
            },
        },
    }
    payload["summary_hash"] = campaign.runner.stable_hash(payload)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return campaign.sha256_path(path)


def test_fitted_summary_reference_supports_both_compaction_epochs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unit" / "summary.json"
    original_sha256 = _write_test_fitted_summary(path)
    campaign.unit_summary.compact_summary(path)

    _summary, accepted = campaign._load_verified_fitted_summary_reference(path)

    assert accepted == {original_sha256, campaign.sha256_path(path)}


def test_compact_replay_uses_verified_immutable_candidate_stream() -> None:
    immutable = [{"split": "source-split", "fold": "source-fold"}]
    compact_result = {"validation": {}}
    full_result = {
        "validation": {
            "entry_intents": [{"split": "full-split", "fold": "full-fold"}]
        }
    }

    assert campaign._replay_source_candidates(
        compact_result,
        immutable,
        is_compact=True,
    ) == immutable
    assert campaign._replay_source_candidates(
        full_result,
        immutable,
        is_compact=False,
    ) == full_result["validation"]["entry_intents"]


def _abstention_packet_and_certification() -> tuple[dict, dict]:
    unit_id = "H0/P0/S42/F3"
    unit = {
        "unit_id": unit_id,
        "unit_sha256": "a" * 64,
        "candidates": [],
    }
    empty_file_sha256 = campaign.hashlib.sha256(b"").hexdigest()
    empty_list_sha256 = campaign.gate.stable_hash([])
    certification = {
        "schema_version": "Protocol101FT1DAllWaitUnitSourceCertificationV1",
        "status": "PASS",
        "units": [
            {
                "unit_id": unit_id,
                "status": "PASS",
                "blockers": [],
                "packet_unit_sha256": unit["unit_sha256"],
                "decision_count": 10,
                "candidate_frame_count": 100,
                "diagnostic_count": 10,
                "action_enter_count": 0,
                "entry_intent_count": 0,
                "trade_count": 0,
                "skipped_event_count": 0,
                "packet_candidate_count": 0,
                "replay_stream_bytes": {
                    "candidate_intents.jsonl": 0,
                    "trades.jsonl": 0,
                    "skipped_events.jsonl": 0,
                },
                "replay_stream_sha256": {
                    "candidate_intents.jsonl": empty_file_sha256,
                    "trades.jsonl": empty_file_sha256,
                    "skipped_events.jsonl": empty_file_sha256,
                },
                "empty_list_hashes": {
                    "candidate_stream_hash": empty_list_sha256,
                    "candidate_payload_hash": empty_list_sha256,
                    "trade_identity_hash": empty_list_sha256,
                },
                "packet_primary_economics": [0.0, 0.0, 0.0],
            }
        ],
        "receipt_sha256": None,
    }
    certification["receipt_sha256"] = campaign.stable_hash(
        {
            key: value
            for key, value in certification.items()
            if key != "receipt_sha256"
        }
    )
    return {"units": [unit]}, certification


def test_validator_accepts_source_certified_abstention_unit() -> None:
    packet, certification = _abstention_packet_and_certification()

    blockers, certified = campaign._filter_certified_abstention_blockers(
        ["unit_candidates_missing:H0/P0/S42/F3"],
        packet,
        certification,
    )

    assert blockers == []
    assert certified == {"H0/P0/S42/F3"}


def test_validator_fails_closed_for_empty_unit_without_source_evidence() -> None:
    packet, certification = _abstention_packet_and_certification()
    certification["units"] = []
    certification["receipt_sha256"] = campaign.stable_hash(
        {
            key: value
            for key, value in certification.items()
            if key != "receipt_sha256"
        }
    )

    blockers, certified = campaign._filter_certified_abstention_blockers(
        ["unit_candidates_missing:H0/P0/S42/F3"],
        packet,
        certification,
    )

    assert certified == set()
    assert "unit_candidates_missing:H0/P0/S42/F3" in blockers
    assert "uncertified_empty_candidate_unit:H0/P0/S42/F3" in blockers


def test_disk_packet_unit_rehydrates_signed_nested_field_order() -> None:
    provenance = {
        key: f"value-{key}"
        for key in reversed(campaign.gate.PROVENANCE_HASH_FIELDS)
    }
    diagnostics = {
        key: {"value": key}
        for key in reversed(campaign.gate.DIAGNOSTIC_KEYS)
    }

    unit = campaign._rehydrate_unit_gate_field_order(
        {
            "provenance": provenance,
            "diagnostics": diagnostics,
        }
    )

    assert tuple(unit["provenance"]) == campaign.gate.PROVENANCE_HASH_FIELDS
    assert tuple(unit["diagnostics"]) == campaign.gate.DIAGNOSTIC_KEYS


def test_campaign_validator_rehydrates_sorted_json_field_order(
    monkeypatch,
) -> None:
    packet, _certification = _abstention_packet_and_certification()
    packet = {
        "schema_version": gate.CAMPAIGN_SCHEMA,
        "campaign_namespace": gate.CAMPAIGN_NAMESPACE,
        "simulator_version": gate.PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "unit_count": gate.UNIT_COUNT,
        "forbidden_evidence": {
            key: False for key in gate.FORBIDDEN_EVIDENCE_FLAGS
        },
        "references": {
            "rows": [
                {
                    "row_id": "H0/P0",
                    "matched_null_z_by_seed": {
                        str(seed): 0.0 for seed in campaign.SEED_ORDER
                    },
                }
            ]
        },
        "controls": {
            "D1": {},
            "D5": {},
            "D6": {},
            "maxT": {},
        },
        "units": [
            {
                **packet["units"][0],
                "provenance": {
                    key: key
                    for key in gate.PROVENANCE_HASH_FIELDS
                },
                "diagnostics": {
                    key: {}
                    for key in gate.DIAGNOSTIC_KEYS
                },
            }
        ],
        "packet_sha256": "0" * 64,
    }
    sorted_packet = json.loads(json.dumps(packet, sort_keys=True))
    sorted_control_authority = json.loads(
        json.dumps(
            {
                "bindings": {
                    name: {}
                    for name in (
                        "references",
                        "D1",
                        "D5",
                        "D6",
                        "maxT",
                    )
                }
            },
            sort_keys=True,
        )
    )
    captured = {}

    def fake_validate(candidate, **kwargs):
        captured["packet_keys"] = tuple(candidate)
        captured["flag_keys"] = tuple(candidate["forbidden_evidence"])
        captured["control_keys"] = tuple(candidate["controls"])
        captured["provenance_keys"] = tuple(
            candidate["units"][0]["provenance"]
        )
        captured["diagnostic_keys"] = tuple(
            candidate["units"][0]["diagnostics"]
        )
        captured["seed_keys"] = tuple(
            candidate["references"]["rows"][0][
                "matched_null_z_by_seed"
            ]
        )
        captured["authority_binding_keys"] = tuple(
            kwargs["control_authority"]["bindings"]
        )
        return []

    monkeypatch.setattr(campaign.gate, "validate_campaign_packet", fake_validate)
    monkeypatch.setattr(
        campaign,
        "_filter_certified_abstention_blockers",
        lambda blockers, _packet: (list(blockers), set()),
    )

    assert campaign._validate_campaign_packet_with_abstentions(
        sorted_packet,
        control_authority=sorted_control_authority,
    ) == []
    assert captured == {
        "packet_keys": gate.CAMPAIGN_PACKET_FIELDS,
        "flag_keys": gate.FORBIDDEN_EVIDENCE_FLAGS,
        "control_keys": ("D1", "D5", "D6", "maxT"),
        "provenance_keys": gate.PROVENANCE_HASH_FIELDS,
        "diagnostic_keys": gate.DIAGNOSTIC_KEYS,
        "seed_keys": tuple(str(seed) for seed in campaign.SEED_ORDER),
        "authority_binding_keys": (
            "references",
            "D1",
            "D5",
            "D6",
            "maxT",
        ),
    }


def test_d5_rehydrates_unit_order_and_accepts_only_certified_abstention(
    tmp_path,
    monkeypatch,
) -> None:
    packet, certification = _abstention_packet_and_certification()
    unit = packet["units"][0]
    unit["provenance"] = {
        key: key for key in reversed(gate.PROVENANCE_HASH_FIELDS)
    }
    unit["diagnostics"] = {
        key: {} for key in reversed(gate.DIAGNOSTIC_KEYS)
    }
    unit["diagnostics"]["fee_sensitivity"] = {"3.00": 0.0}
    unit["diagnostics"]["fill_edge_band"] = {
        "pessimistic_executable": 0.0
    }
    unit["diagnostics"]["noise_diagnostics"] = {"1.0x": 0.0}
    captured = {}

    def fake_validate(candidate, _axis):
        captured["provenance_keys"] = tuple(candidate["provenance"])
        captured["diagnostic_keys"] = tuple(candidate["diagnostics"])
        return [f"unit_candidates_missing:{candidate['unit_id']}"]

    monkeypatch.setattr(
        campaign.gate,
        "expected_unit_axes",
        lambda: (("H0", "P0", 42, "F3"),),
    )
    monkeypatch.setattr(campaign.gate, "_validate_unit", fake_validate)
    monkeypatch.setattr(
        campaign,
        "_filter_certified_abstention_blockers",
        lambda blockers, _packet: (
            list(blockers),
            {unit["unit_id"]},
        ),
    )
    monkeypatch.setattr(
        campaign,
        "D5_DETAIL_PATH",
        tmp_path / "D5_detail.json",
    )

    detail, control = campaign.build_d5_control(
        json.loads(json.dumps(packet, sort_keys=True))
    )

    assert captured == {
        "provenance_keys": gate.PROVENANCE_HASH_FIELDS,
        "diagnostic_keys": gate.DIAGNOSTIC_KEYS,
    }
    assert detail["unit_count"] == 1
    assert detail["rows"][0]["candidate_count"] == 0
    assert detail["rows"][0]["trade_count"] == 0
    assert detail["rows"][0]["net_pnl"] == 0.0
    assert control["identity_complete"] is True


def test_d5_fails_closed_for_non_abstention_unit_error(
    monkeypatch,
) -> None:
    packet, _certification = _abstention_packet_and_certification()
    unit_id = packet["units"][0]["unit_id"]
    monkeypatch.setattr(
        campaign.gate,
        "expected_unit_axes",
        lambda: (("H0", "P0", 42, "F3"),),
    )
    monkeypatch.setattr(
        campaign.gate,
        "_validate_unit",
        lambda _unit, _axis: [f"unit_provenance_hash_mismatch:{unit_id}"],
    )
    monkeypatch.setattr(
        campaign,
        "_filter_certified_abstention_blockers",
        lambda blockers, _packet: (list(blockers), {unit_id}),
    )

    with pytest.raises(
        RuntimeError,
        match=f"unit_provenance_hash_mismatch:{unit_id}",
    ):
        campaign.build_d5_control(packet)


def test_real_final_packet_keeps_exact_compatibility_field_order() -> None:
    unit_packet = {"units": [{"unit_id": "real-unit"}]}
    reference_receipt = {
        "references": {"schema_version": "real-references"},
        "controls": {
            "D1": {"schema_version": "D1"},
            "D5": {"schema_version": "D5"},
            "D6": {"schema_version": "D6"},
        },
    }
    maxt = {"schema_version": "maxT"}

    packet = campaign._final_campaign_packet(
        unit_packet,
        reference_receipt,
        maxt,
    )

    assert tuple(packet) == gate.CAMPAIGN_PACKET_FIELDS
    assert packet["schema_version"] == gate.CAMPAIGN_SCHEMA
    assert packet["units"] == unit_packet["units"]
    assert tuple(packet["controls"]) == ("D1", "D5", "D6", "maxT")
    assert packet["packet_sha256"] == gate.stable_hash(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )


def test_maxT_session_grid_requires_225_unique_chronological_sessions() -> None:
    units = []
    for fold in campaign.FOLD_ORDER:
        units.append(
            {
                "hypothesis": "H0",
                "policy": "P0",
                "seed": 42,
                "fold": fold,
                "session_ids": [
                    f"2025-{fold:02d}-{index:02d}"
                    for index in range(1, 46)
                ],
            }
        )

    grid = campaign._maxT_session_grid({"units": units})

    assert len(grid) == 225
    assert grid[0][1] == "F1"
    assert grid[-1][1] == "F5"

    units[-1]["session_ids"][0] = units[0]["session_ids"][0]
    with pytest.raises(RuntimeError, match="225_unique"):
        campaign._maxT_session_grid({"units": units})


def _d1_test_destination() -> RepairedCanonicalDecision:
    base = CanonicalDecision(
        session="2025-03-14",
        decision_time=pd.Timestamp("2025-03-14T14:00:00Z"),
        features=np.asarray([[1.0] * 15, [2.0] * 15]),
        labels=np.asarray([11.0, 22.0]),
        mid_labels=np.asarray([12.0, 23.0]),
        entry_asks=np.asarray([1.1, 2.2]),
        offsets=np.asarray([-1.0, 1.0]),
        rights=np.asarray(["C", "P"], dtype=object),
        contract_ids=np.asarray(["C0", "P1"], dtype=object),
        strike_indices=np.asarray([0, 1]),
        right_indices=np.asarray([0, 1]),
    )
    first = np.asarray([101, 202])
    return RepairedCanonicalDecision(
        base=base,
        realized_exit_time_ns=first.copy(),
        source_exit_quote_time_ns=first.copy() + 1,
        exit_quote_age_ms=np.asarray([1.0, 2.0]),
        exit_reason_codes=np.asarray([1, 2], dtype=np.uint8),
        executable_exit_bids=np.asarray([1.0, 2.0]),
        policy_deadline_ns=first.copy() + 2,
        invalid_reason_codes=np.asarray([0, 0], dtype=np.uint8),
        canonical_strike_slots=np.asarray([0, 1]),
        source_quote_time_ns=first.copy() + 3,
        source_context_time_ns=first.copy() + 4,
    )


def test_D1_real_target_adapter_maps_slots_and_filters_invalid_source() -> None:
    destination = _d1_test_destination()
    net = np.full((21, 2), np.nan)
    mid = np.full((21, 2), np.nan)
    net[0, 0] = 101.0
    mid[0, 0] = 102.0

    moved = campaign._d1_apply_moved_targets(
        destination,
        net,
        mid,
        validation_unpermuted=False,
    )

    assert moved is not None
    assert moved.base.labels.tolist() == [101.0]
    assert moved.base.mid_labels.tolist() == [102.0]
    assert moved.base.contract_ids.tolist() == ["C0"]
    assert moved.base.features.tolist() == [[1.0] * 15]
    assert moved.realized_exit_time_ns.tolist() == [101]
    assert campaign._d1_apply_moved_targets(
        destination,
        np.full((21, 2), np.nan),
        np.full((21, 2), np.nan),
        validation_unpermuted=False,
    ) is None


def test_D1_validation_target_adapter_fails_closed_on_target_change() -> None:
    destination = _d1_test_destination()
    net = np.full((21, 2), np.nan)
    mid = np.full((21, 2), np.nan)
    net[0, 0], net[1, 1] = destination.base.labels
    mid[0, 0], mid[1, 1] = destination.base.mid_labels

    assert campaign._d1_apply_moved_targets(
        destination,
        net,
        mid,
        validation_unpermuted=True,
    ) is not None
    net[1, 1] += 1.0
    with pytest.raises(RuntimeError, match="validation_target_identity"):
        campaign._d1_apply_moved_targets(
            destination,
            net,
            mid,
            validation_unpermuted=True,
        )


def test_D1_non_candidate_threshold_never_calls_simulator_v5(
    monkeypatch,
) -> None:
    destination = _d1_test_destination()
    decision_ns = int(destination.base.decision_time.value)
    exit_ns = decision_ns + 60 * 1_000_000_000
    destination = replace(
        destination,
        realized_exit_time_ns=np.asarray([exit_ns, exit_ns]),
        source_exit_quote_time_ns=np.asarray([exit_ns, exit_ns]),
        exit_quote_age_ms=np.asarray([0.0, 0.0]),
        policy_deadline_ns=np.asarray([exit_ns, exit_ns]),
    )
    monkeypatch.setattr(
        campaign.hgb_core,
        "simulate_serial_candidates_v5",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("simulator v5 must not receive D1 target overrides")
        ),
    )

    threshold, rows = campaign._choose_D1_non_candidate_threshold(
        [destination],
        [np.asarray([1.0, 0.0])],
        epsilon=0.0,
        config=campaign.HGBUnitConfig(
            hypothesis="H2",
            policy_index=5,
            seed=8600,
        ),
        fold="D1_expanding_fold_01",
    )

    assert isinstance(threshold, float)
    assert rows
    assert all(row["non_candidate"] is True for row in rows)
    assert all(row["replay_permitted"] is False for row in rows)
    assert all(row["simulator_v5_called"] is False for row in rows)
    assert all(
        row["executable_quote_pnl_claim"] is False for row in rows
    )


def test_D1_non_candidate_schedule_rejects_bad_two_clock() -> None:
    destination = _d1_test_destination()
    intent = campaign._d1_non_candidate_intents(
        [destination],
        [np.asarray([1.0, 0.0])],
        threshold=0.0,
        epsilon=0.0,
        config=campaign.HGBUnitConfig(
            hypothesis="H2",
            policy_index=5,
            seed=8600,
        ),
        fold="D1_expanding_fold_01",
    )[0]

    with pytest.raises(RuntimeError, match="intent_contract_invalid"):
        campaign._d1_non_candidate_serial_target_evaluation(
            [intent],
            config=campaign.HGBUnitConfig(
                hypothesis="H2",
                policy_index=5,
                seed=8600,
            ),
        )


def test_atomic_progress_records_exact_blocker_without_losing_flags(
    tmp_path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "completed_units": 17,
                "forbidden_action_flags": {
                    "seed_45_or_G9_executed": False,
                },
            }
        )
    )
    monkeypatch.setattr(campaign, "PROGRESS_PATH", progress_path)

    campaign.update_progress(
        status="blocked_recorded_pending_repair",
        current_node="REAL_V5_REFERENCES_D1_D5_D6",
        blocker_classification="mechanical_non_scientific",
        blocker="ValueError: exact blocker",
    )

    payload = json.loads(progress_path.read_text())
    assert payload["completed_units"] == 17
    assert payload["forbidden_action_flags"]["seed_45_or_G9_executed"] is False
    assert payload["blocker_classification"] == "mechanical_non_scientific"
    assert payload["blocker"] == "ValueError: exact blocker"


def test_self_hash_verifier_rejects_mutation() -> None:
    payload = {"row_id": "H0/P0", "row_sha256": None}
    payload["row_sha256"] = campaign.stable_hash(
        {"row_id": payload["row_id"]}
    )
    campaign._verify_self_hash(
        payload,
        field="row_sha256",
        identity="reference:H0/P0",
    )

    payload["row_id"] = "H0/P1"
    with pytest.raises(RuntimeError, match="self_hash_mismatch"):
        campaign._verify_self_hash(
            payload,
            field="row_sha256",
            identity="reference:H0/P0",
        )


def test_immutable_json_reuses_exact_content_and_rejects_conflict(
    tmp_path,
) -> None:
    path = tmp_path / "receipt.json"
    campaign.write_json_immutable(path, {"status": "complete"})
    campaign.write_json_immutable(path, {"status": "complete"})

    with pytest.raises(RuntimeError, match="immutable_json_conflict"):
        campaign.write_json_immutable(path, {"status": "changed"})
