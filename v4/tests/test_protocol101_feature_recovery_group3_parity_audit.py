from __future__ import annotations

import math

import pytest

from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as group1
from v4.scripts import run_protocol101_feature_recovery_group3_parity_audit as group3


def base_vector(spx: float) -> list[float]:
    names = group1.candidate_feature_names()
    by_name = {name: idx for idx, name in enumerate(names)}
    vector = [0.0] * len(names)
    vector[by_name["market_last.spx_close"]] = spx
    return vector


def test_group3_preregistration_is_machine_checkable() -> None:
    prereg = group1.load_json(group3.DEFAULT_PREREGISTRATION)
    definition = group1.load_json(group3.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    blockers = group1.validate_machine_checkable(prereg, definition)

    assert blockers == []
    assert definition["feature_group"] == "internally_computed_greeks_and_iv"
    assert definition["greek_repair_policy"]["vendor_greek_fallback_allowed"] is False
    assert "vendor-provided iv" in definition["forbidden_model_facing_sources"]
    assert "raw bid as a direct feature" in definition["forbidden_model_facing_sources"]


def test_group3_contract_terms_and_settlement_time_are_deterministic() -> None:
    row = {
        "decision_ts": "2026-07-01T19:00:00+00:00",
        "candidate_universe": [],
    }
    decision_ts = group1.parse_ts(row["decision_ts"])
    assert decision_ts is not None

    strike, right = group3.contract_terms("SPXW-20260701-06480.000-C")
    settlement = group3.settlement_time_utc(row, decision_ts)

    assert strike == 6480.0
    assert right == "C"
    assert settlement.isoformat() == "2026-07-01T20:00:00+00:00"


def test_group3_greek_derivation_uses_repair_formula_not_vendor_greeks() -> None:
    vector = base_vector(6500.0)
    row = {
        "decision_ts": "2026-07-01T19:00:00+00:00",
        "candidate_universe": [
            {
                "contract_id": "SPXW-20260701-06480.000-C",
                "right": "C",
                "entry_bid": 19.70,
                "entry_ask": 20.30,
                "entry_mid": 20.00,
                "iv": 0.90,
                "delta": 0.90,
                "gamma": 0.90,
                "theta": -0.90,
            }
        ],
    }
    definition = group1.load_json(group3.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    recovered, missing = group3.derive_internal_greek_features(
        row=row,
        contract_id="SPXW-20260701-06480.000-C",
        vector=vector,
        definition=definition,
    )

    assert recovered["repaired_success_flag"] == 1.0
    assert recovered["repaired_price_source_ask"] == 1.0
    assert recovered["repaired_price_source_mid"] == 0.0
    assert recovered["repaired_iv"] > 0.0
    assert not math.isclose(recovered["repaired_delta"], 0.90)
    assert not any(missing.values())


def test_group3_repair_failure_does_not_fall_back_to_vendor_greeks() -> None:
    vector = base_vector(6500.0)
    row = {
        "decision_ts": "2026-07-01T19:00:00+00:00",
        "candidate_universe": [
            {
                "contract_id": "SPXW-20260701-06480.000-C",
                "right": "C",
                "entry_bid": 18.80,
                "entry_ask": 19.20,
                "entry_mid": 19.00,
                "iv": 0.20,
                "delta": 0.99,
                "gamma": 0.001,
                "theta": -0.10,
            }
        ],
    }
    definition = group1.load_json(group3.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    recovered, missing = group3.derive_internal_greek_features(
        row=row,
        contract_id="SPXW-20260701-06480.000-C",
        vector=vector,
        definition=definition,
    )

    assert recovered["repaired_success_flag"] == 0.0
    assert recovered["repaired_iv"] == 0.0
    assert recovered["repaired_delta"] == 0.0
    assert missing["repaired_iv"] is True
    assert missing["repaired_delta"] is True
    assert missing["repaired_success_flag"] is False


def test_group3_bad_contract_id_marks_continuous_features_missing() -> None:
    definition = group1.load_json(group3.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    recovered, missing = group3.derive_internal_greek_features(
        row={"decision_ts": "2026-07-01T19:00:00+00:00", "candidate_universe": []},
        contract_id="bad-contract-id",
        vector=base_vector(6500.0),
        definition=definition,
    )

    assert recovered["repaired_success_flag"] == 0.0
    for name in definition["continuous_recovered_features"]:
        assert missing[name] is True
