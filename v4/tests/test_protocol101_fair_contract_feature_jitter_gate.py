"""Tests for the fair-contract option-feature jitter gate."""
from __future__ import annotations

import numpy as np

from v4.scripts import run_protocol101_fair_contract_feature_jitter_gate as gate


def test_apply_option_feature_jitter_copies_and_perturbs_vendor_sensitive_fields() -> None:
    original_features = np.arange(12, dtype=np.float32)
    records = [{"_features": original_features, "contract_id": "SPXW-X"}]
    scenario = gate.JitterScenario(
        name="synthetic",
        iv_delta=0.002,
        spread_delta=0.05,
        spread_frac_delta=0.005,
        bid_size_scale=0.5,
        ask_size_scale=2.0,
    )

    jittered = gate.apply_option_feature_jitter(records, scenario)

    assert jittered is not records
    assert jittered[0]["_features"] is not original_features
    assert original_features[3] == 3.0
    assert jittered[0]["_features"][3] == np.float32(3.05)
    assert jittered[0]["_features"][4] == np.float32(4.005)
    assert jittered[0]["_features"][5] == np.float32(2.5)
    assert jittered[0]["_features"][6] == np.float32(12.0)
    assert jittered[0]["_features"][9] == np.float32(9.002)
    assert jittered[0]["contract_id"] == "SPXW-X"


def test_apply_option_feature_jitter_baseline_is_copy_only() -> None:
    features = np.arange(12, dtype=np.float32)
    records = [{"_features": features}]

    baseline = gate.apply_option_feature_jitter(records, gate.JitterScenario(name="baseline"))

    assert baseline[0]["_features"] is not features
    np.testing.assert_allclose(baseline[0]["_features"], features)


def test_compare_selected_actions_classifies_presence_and_contract_drift() -> None:
    baseline = [
        {
            "split": "validation",
            "session": "2026-02-03",
            "decision_time": "2026-02-03T15:00:00+00:00",
            "contract_id": "A",
        },
        {
            "split": "validation",
            "session": "2026-02-03",
            "decision_time": "2026-02-03T16:00:00+00:00",
            "contract_id": "B",
        },
    ]
    scenario = [
        {
            "split": "validation",
            "session": "2026-02-03",
            "decision_time": "2026-02-03T15:00:00+00:00",
            "contract_id": "A",
        },
        {
            "split": "validation",
            "session": "2026-02-03",
            "decision_time": "2026-02-03T16:00:00+00:00",
            "contract_id": "C",
        },
        {
            "split": "validation",
            "session": "2026-02-03",
            "decision_time": "2026-02-03T17:00:00+00:00",
            "contract_id": "D",
        },
    ]

    comparison = gate.compare_selected_actions(baseline, scenario, total_decisions=10)

    assert comparison["action_presence_mismatches"] == 1
    assert comparison["selected_contract_mismatches"] == 1
    assert comparison["action_presence_match_rate"] == 0.9
    assert comparison["selected_contract_match_rate"] == 0.5
    assert comparison["mismatch_examples"][0]["decision_time"] == "2026-02-03T16:00:00+00:00"
