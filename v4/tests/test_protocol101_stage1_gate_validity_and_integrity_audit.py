from __future__ import annotations

from v4.scripts.run_protocol101_stage1_gate_validity_and_integrity_audit import (
    empirical_gate_validity,
    gate_recommendations,
)


def _row(policy: int, profitable: bool, calibrated: bool) -> dict[str, object]:
    return {
        "policy_index": policy,
        "median_seed_cv_pooled_net_pnl": 100.0 if profitable else -10.0,
        "median_seed_null_z": 4.0 if profitable else 0.0,
        "median_seed_weighted_oof_ece": 0.11 if profitable else 0.02,
        **{f"G{index}": profitable for index in range(1, 8)},
        "G8": calibrated,
    }


def test_empirical_gate_validity_detects_inverse_g8_pattern() -> None:
    rows = [_row(5, True, False), _row(0, False, True)]
    result = empirical_gate_validity(rows)
    assert result["G8_discrimination"]["G1_profitable_rows"] == 1
    assert result["G8_discrimination"]["G1_profitable_rows_passing_G8"] == 0
    assert result["G8_discrimination"]["non_G1_rows_passing_G8"] == 1


def test_g8_recommendation_does_not_retroactively_pass_candidate() -> None:
    rows = [_row(5, True, False), _row(0, False, True)]
    result = empirical_gate_validity(rows)
    recommendations = {
        row["gate"]: row["recommendation"]
        for row in gate_recommendations(result)
    }
    assert recommendations["G8"] == (
        "owner_amend_to_report_only_until_confidence_controls_behavior"
    )
    assert recommendations["G9"] == "keep_spend_once_hard_gate"
