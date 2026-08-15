from __future__ import annotations

import pytest

from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as group1
from v4.scripts import run_protocol101_feature_recovery_group4_parity_audit as group4


def base_vector(spx: float) -> list[float]:
    names = group1.candidate_feature_names()
    by_name = {name: idx for idx, name in enumerate(names)}
    vector = [0.0] * len(names)
    vector[by_name["market_last.spx_close"]] = spx
    return vector


def test_group4_preregistration_is_machine_checkable() -> None:
    prereg = group1.load_json(group4.DEFAULT_PREREGISTRATION)
    definition = group1.load_json(group4.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    blockers = group1.validate_machine_checkable(prereg, definition)

    assert blockers == []
    assert definition["feature_group"] == "normalized_liquidity_and_spread"
    assert "raw bid as a direct feature" in definition["forbidden_model_facing_sources"]
    assert "normalized_spread_over_mid" in definition["candidate_recovered_model_facing_features"]


def test_group4_liquidity_derivation_uses_normalized_quote_features() -> None:
    vector = base_vector(5000.0)
    row = {
        "candidate_universe": [
            {
                "contract_id": "SPXW-20260701-05000.000-C",
                "entry_bid": 9.5,
                "entry_ask": 10.0,
                "entry_mid": 9.75,
                "quote_age_ms": 1200.0,
            }
        ]
    }
    definition = group1.load_json(group4.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    recovered, missing = group4.derive_liquidity_features(
        row=row,
        contract_id="SPXW-20260701-05000.000-C",
        vector=vector,
        definition=definition,
    )

    assert recovered["normalized_spread_points"] == pytest.approx(0.5)
    assert recovered["normalized_spread_over_mid"] == pytest.approx(0.5 / 9.75)
    assert recovered["normalized_ask_bps_underlying"] == pytest.approx(20.0)
    assert recovered["quote_age_seconds"] == pytest.approx(1.2)
    assert recovered["spread_frac_bucket_0_05_0_10"] == 1.0
    assert recovered["quote_age_bucket_1s_15s"] == 1.0
    assert recovered["valid_quote_flag"] == 1.0
    assert not any(missing.values())


def test_group4_invalid_quote_marks_continuous_quote_features_missing() -> None:
    vector = base_vector(5000.0)
    row = {
        "candidate_universe": [
            {
                "contract_id": "SPXW-20260701-05000.000-C",
                "entry_bid": 10.5,
                "entry_ask": 10.0,
                "entry_mid": 10.25,
            }
        ]
    }
    definition = group1.load_json(group4.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    recovered, missing = group4.derive_liquidity_features(
        row=row,
        contract_id="SPXW-20260701-05000.000-C",
        vector=vector,
        definition=definition,
    )

    assert recovered["valid_quote_flag"] == 0.0
    assert recovered["normalized_spread_points"] == 0.0
    assert missing["normalized_spread_points"] is True
    assert missing["normalized_ask_bps_underlying"] is True
    assert missing["quote_age_seconds"] is True
