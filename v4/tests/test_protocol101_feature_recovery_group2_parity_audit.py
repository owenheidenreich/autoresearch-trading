from __future__ import annotations

from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as group1
from v4.scripts import run_protocol101_feature_recovery_group2_parity_audit as group2


def test_group2_preregistration_is_machine_checkable() -> None:
    prereg = group1.load_json(group2.DEFAULT_PREREGISTRATION)
    definition = group1.load_json(group2.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    blockers = group1.validate_machine_checkable(prereg, definition)

    assert blockers == []
    assert definition["feature_group"] == "candidate_geometry_and_moneyness"
    assert "bid" in definition["forbidden_model_facing_sources"]
    assert "offset_points" in definition["candidate_recovered_model_facing_features"]


def test_group2_geometry_derivation_uses_candidate_metadata_and_spx_normalizer() -> None:
    names = group1.candidate_feature_names()
    by_name = {name: idx for idx, name in enumerate(names)}
    vector = [0.0] * len(names)
    vector[by_name["market_last.spx_close"]] = 5000.0
    row = {
        "candidate_universe": [
            {"contract_id": "SPXW-20260701-04950.000-P", "right": "P", "offset": -50.0},
            {"contract_id": "SPXW-20260701-04975.000-P", "right": "P", "offset": -25.0},
            {"contract_id": "SPXW-20260701-05025.000-C", "right": "C", "offset": 25.0},
        ],
        "features": {
            "token_features": [
                {"contract_id": "SPXW-20260701-04950.000-P", "features": vector},
                {"contract_id": "SPXW-20260701-04975.000-P", "features": vector},
                {"contract_id": "SPXW-20260701-05025.000-C", "features": vector},
            ]
        },
    }
    definition = group1.load_json(group2.DEFAULT_PLAN_DIR / "feature_group_definition.json")

    recovered, missing = group2.derive_geometry_features(
        row=row,
        contract_id="SPXW-20260701-04950.000-P",
        vector=vector,
        definition=definition,
    )

    assert recovered["right_is_put"] == 1.0
    assert recovered["offset_points"] == -50.0
    assert recovered["abs_offset_bps_underlying"] == 100.0
    assert recovered["out_of_the_money_flag"] == 1.0
    assert recovered["same_right_candidate_count"] == 2.0
    assert recovered["candidate_count_total"] == 3.0
    assert recovered["abs_offset_bucket_25_50"] == 1.0
    assert not any(missing.values())
