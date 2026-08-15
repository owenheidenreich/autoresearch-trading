from __future__ import annotations

from copy import deepcopy

from v4.model.protocol101_stage1_gate_contract import (
    ROWS,
    UNIT_COUNT,
    contract_payload,
    inclusive_lower_bound,
    inclusive_upper_bound,
    seal_campaign_packet,
    seal_unit,
    validate_campaign_packet,
    verify_frozen_acceptance_inputs,
)
from v4.scripts.run_protocol101_full_trader_stage1_gate_audit_selection_validation import (
    ROOT,
    _authorities,
    _validate,
    build_synthetic_campaign,
)


def test_exact_geometry_and_frozen_acceptance_inputs() -> None:
    packet = build_synthetic_campaign()
    assert UNIT_COUNT == 420
    assert len(ROWS) == 28
    assert len(packet["units"]) == 420
    assert _validate(packet) == []
    assert verify_frozen_acceptance_inputs(ROOT)["valid"] is True


def test_contract_freezes_G8_report_only_and_G9_false() -> None:
    contract = contract_payload()
    assert contract["hard_gates"] == [f"G{index}" for index in range(1, 8)]
    assert contract["G8"] == {
        "required": True,
        "role": "report_only",
        "benchmark": 0.10,
    }
    assert contract["G9"] == {"run": False, "authorized": False}
    assert contract["maxT"]["family_size"] == 28
    assert contract["maxT"]["replicates"] == 20_000


def test_packet_rejects_v4_seed45_protected_and_nonfinite() -> None:
    mutations = []
    v4 = build_synthetic_campaign()
    v4["simulator_version"] = "protocol101_serial_simulator_v4_account_continuity_fee_reserve"
    mutations.append(seal_campaign_packet(v4))
    seed45 = build_synthetic_campaign()
    seed45["units"][0]["seed"] = 45
    seed45["units"][0] = seal_unit(seed45["units"][0])
    mutations.append(seal_campaign_packet(seed45))
    protected = build_synthetic_campaign()
    protected["forbidden_evidence"]["protected_holdout_present"] = True
    mutations.append(seal_campaign_packet(protected))
    nonfinite = build_synthetic_campaign()
    nonfinite["units"][0]["diagnostics"]["worst_day"] = float("inf")
    mutations.append(nonfinite)
    assert all(_validate(packet) for packet in mutations)


def test_missing_reordered_duplicate_and_primary_substitution_fail() -> None:
    packets = []
    missing = build_synthetic_campaign()
    missing["units"].pop()
    packets.append(seal_campaign_packet(missing))
    reordered = build_synthetic_campaign()
    reordered["units"][0], reordered["units"][1] = (
        reordered["units"][1],
        reordered["units"][0],
    )
    packets.append(seal_campaign_packet(reordered))
    duplicate = build_synthetic_campaign()
    duplicate["units"][1] = deepcopy(duplicate["units"][0])
    packets.append(seal_campaign_packet(duplicate))
    no_primary = build_synthetic_campaign()
    no_primary["units"][0]["diagnostics"]["noise_diagnostics"].pop("1.0x")
    no_primary["units"][0] = seal_unit(no_primary["units"][0])
    packets.append(seal_campaign_packet(no_primary))
    assert all(_validate(packet) for packet in packets)


def test_external_execution_authority_rejects_self_resealed_model_hash() -> None:
    packet = build_synthetic_campaign()
    frozen = _authorities(packet)
    packet["units"][0]["provenance"]["model_hash"] = "a" * 64
    packet["units"][0] = seal_unit(packet["units"][0])
    packet = seal_campaign_packet(packet)
    blockers = validate_campaign_packet(packet, **frozen)
    assert "execution_authority_binding_mismatch" in blockers


def test_external_control_authority_rejects_self_resealed_D5_D6_hashes() -> None:
    for name in ("D5", "D6"):
        packet = build_synthetic_campaign()
        frozen = _authorities(packet)
        packet["controls"][name]["receipt_sha256"] = "b" * 64
        packet = seal_campaign_packet(packet)
        blockers = validate_campaign_packet(packet, **frozen)
        assert "control_authority_binding_mismatch" in blockers


def test_inclusive_boundary_policy_is_narrow_and_nonfinite_safe() -> None:
    assert inclusive_lower_bound(1.0, 1.0)
    assert inclusive_lower_bound(1.0 + 2e-9, 1.0)
    assert not inclusive_lower_bound(1.0 - 2e-9, 1.0)
    assert inclusive_lower_bound(-0.0, 0.0)
    assert inclusive_lower_bound(float("inf"), 1.0)
    assert not inclusive_lower_bound(float("-inf"), 1.0)
    assert not inclusive_lower_bound(float("nan"), 1.0)
    assert inclusive_upper_bound(6.0, 6.0)
    assert inclusive_upper_bound(6.0 - 2e-9, 6.0)
    assert not inclusive_upper_bound(6.0 + 2e-9, 6.0)
    assert inclusive_upper_bound(float("-inf"), 6.0)
    assert not inclusive_upper_bound(float("inf"), 6.0)
    assert not inclusive_upper_bound(float("nan"), 6.0)


def test_execution_authority_missing_reordered_duplicate_extra_and_malformed_fail() -> None:
    packet = build_synthetic_campaign()
    frozen = _authorities(packet)
    authorities = []
    missing = deepcopy(frozen["execution_authority"])
    missing["units"].pop()
    missing["unit_count"] -= 1
    authorities.append((missing, frozen["execution_authority_sha256"]))
    reordered = deepcopy(frozen["execution_authority"])
    reordered["units"][0], reordered["units"][1] = (
        reordered["units"][1],
        reordered["units"][0],
    )
    authorities.append((reordered, frozen["execution_authority_sha256"]))
    duplicate = deepcopy(frozen["execution_authority"])
    duplicate["units"][1] = deepcopy(duplicate["units"][0])
    authorities.append((duplicate, frozen["execution_authority_sha256"]))
    extra = deepcopy(frozen["execution_authority"])
    extra["unexpected"] = True
    authorities.append((extra, frozen["execution_authority_sha256"]))
    authorities.append((frozen["execution_authority"], "not-a-sha256"))
    for authority, expected_hash in authorities:
        blockers = validate_campaign_packet(
            packet,
            execution_authority=authority,
            execution_authority_sha256=expected_hash,
            control_authority=frozen["control_authority"],
            control_authority_sha256=frozen["control_authority_sha256"],
        )
        assert any("execution_authority" in blocker for blocker in blockers)


def test_control_authority_missing_reordered_extra_and_malformed_fail() -> None:
    packet = build_synthetic_campaign()
    frozen = _authorities(packet)
    authorities = []
    missing = deepcopy(frozen["control_authority"])
    missing["bindings"].pop("D5")
    authorities.append((missing, frozen["control_authority_sha256"]))
    reordered = deepcopy(frozen["control_authority"])
    reordered["bindings"] = {
        key: reordered["bindings"][key]
        for key in ("D1", "references", "D5", "D6", "maxT")
    }
    authorities.append((reordered, frozen["control_authority_sha256"]))
    extra = deepcopy(frozen["control_authority"])
    extra["bindings"]["unexpected"] = {}
    authorities.append((extra, frozen["control_authority_sha256"]))
    authorities.append((frozen["control_authority"], "not-a-sha256"))
    for authority, expected_hash in authorities:
        blockers = validate_campaign_packet(
            packet,
            execution_authority=frozen["execution_authority"],
            execution_authority_sha256=frozen[
                "execution_authority_sha256"
            ],
            control_authority=authority,
            control_authority_sha256=expected_hash,
        )
        assert any("control_authority" in blocker for blocker in blockers)
