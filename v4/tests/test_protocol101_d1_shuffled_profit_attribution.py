from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from v4.scripts import (
    run_protocol101_d1_shuffled_profit_attribution as attribution,
)


def _opportunity(session: str, decision_time_ns: int) -> SimpleNamespace:
    base = SimpleNamespace(
        session=session,
        decision_time=SimpleNamespace(value=decision_time_ns),
        rights=("C", "C", "P", "P"),
        entry_asks=(1.0, 2.0, 1.5, 2.5),
        contract_ids=("C0", "C1", "P0", "P1"),
    )
    repaired = SimpleNamespace(
        base=base,
        canonical_strike_slots=(0, 1, 0, 1),
    )
    return SimpleNamespace(repaired=repaired)


def test_profile_permutation_preserves_times_and_profile_margins(
    monkeypatch,
) -> None:
    opportunities = {
        ("2026-01-02", time): _opportunity("2026-01-02", time)
        for time in range(100, 108)
    }
    profiles = [
        ("C", 0, 1.0),
        ("C", 1, 2.0),
        ("P", 0, 1.5),
        ("P", 1, 2.5),
        ("C", 0, 1.0),
        ("C", 1, 2.0),
        ("P", 0, 1.5),
        ("P", 1, 2.5),
    ]
    model_candidates = [
        SimpleNamespace(
            session="2026-01-02",
            decision_time_ns=time,
            contract_id=f"source-{time}",
            right=right,
            canonical_strike_slot=slot,
            entry_ask=ask,
        )
        for time, (right, slot, ask) in zip(range(100, 108), profiles)
    ]

    def fake_candidate(opportunity, index, **_kwargs):
        return SimpleNamespace(
            session=opportunity.repaired.base.session,
            decision_time_ns=opportunity.repaired.base.decision_time.value,
            contract_id=opportunity.repaired.base.contract_ids[index],
            right=opportunity.repaired.base.rights[index],
            canonical_strike_slot=(
                opportunity.repaired.canonical_strike_slots[index]
            ),
            entry_ask=opportunity.repaired.base.entry_asks[index],
        )

    monkeypatch.setattr(attribution, "_candidate", fake_candidate)
    candidates, receipt = attribution.profile_permutation_at_model_times(
        model_candidates,
        opportunities,
        model_seed=8600,
        draw=0,
    )

    assert {item.decision_time_ns for item in candidates} == set(range(100, 108))
    assert Counter(
        (item.right, item.canonical_strike_slot)
        for item in candidates
    ) == Counter((right, slot) for right, slot, _ask in profiles)
    assert receipt["exact_time_set_preserved"] is True
    assert receipt["exact_right_slot_distribution_preserved"] is True
    assert receipt["exact_premium_bucket_distribution_preserved"] is True
    assert receipt["exact_right_slot_match_rate"] == 1.0
    assert receipt["time_profile_token_reassignment_rate"] >= 0.90


def test_paired_bootstrap_averages_seed_repetitions() -> None:
    rows = pd.DataFrame(
        [
            {"session": "A", "model_seed": 1, "difference": 10.0},
            {"session": "A", "model_seed": 2, "difference": 20.0},
            {"session": "B", "model_seed": 1, "difference": 30.0},
            {"session": "B", "model_seed": 2, "difference": 40.0},
        ]
    )
    result = attribution._paired_block_bootstrap(rows, contrast_index=99)
    assert result["observed_paired_pnl_difference"] == 50.0
    assert result["seed_count"] == 2


def test_paired_bootstrap_can_share_one_resample_matrix() -> None:
    rows = pd.DataFrame(
        [
            {"session": "A", "model_seed": 1, "difference": 10.0},
            {"session": "B", "model_seed": 1, "difference": 30.0},
        ]
    )
    indexes = attribution._shared_bootstrap_indexes(2)
    positive = attribution._paired_block_bootstrap(
        rows,
        bootstrap_indexes=indexes,
    )
    negative_rows = rows.copy()
    negative_rows["difference"] *= -1.0
    negative = attribution._paired_block_bootstrap(
        negative_rows,
        bootstrap_indexes=indexes,
    )
    assert positive["ci_p2_5"] == pytest.approx(
        -negative["ci_p97_5"]
    )
    assert positive["ci_p97_5"] == pytest.approx(
        -negative["ci_p2_5"]
    )


def test_profile_permutation_marks_unidentifiable_singleton(monkeypatch) -> None:
    opportunity = _opportunity("2026-01-02", 100)
    source = SimpleNamespace(
        session="2026-01-02",
        decision_time_ns=100,
        contract_id="C0",
        right="C",
        canonical_strike_slot=0,
        entry_ask=1.0,
    )

    def fake_candidate(opportunity, index, **_kwargs):
        return SimpleNamespace(
            session=opportunity.repaired.base.session,
            decision_time_ns=opportunity.repaired.base.decision_time.value,
            contract_id=opportunity.repaired.base.contract_ids[index],
            right=opportunity.repaired.base.rights[index],
            canonical_strike_slot=(
                opportunity.repaired.canonical_strike_slots[index]
            ),
            entry_ask=opportunity.repaired.base.entry_asks[index],
        )

    monkeypatch.setattr(attribution, "_candidate", fake_candidate)
    _candidates, receipt = attribution.profile_permutation_at_model_times(
        [source],
        {("2026-01-02", 100): opportunity},
        model_seed=8600,
        draw=0,
    )
    assert receipt["time_profile_token_reassignment_rate"] == 0.0
    assert receipt["token_reassignment_quality_pass"] is False


def test_preregistered_controls_include_valid_margin_matched_nulls() -> None:
    definitions = attribution.control_definitions()
    controls = definitions["controls"]["derived_matching_controls"]
    assert {"M_F", "M_G", "M_real"} <= set(controls)
    assert (
        "G_minus_M_G_valid_negative_control_increment"
        in definitions["primary_contrasts"]
    )
    plan = attribution.investigation_plan()
    assert plan["constraints"]["no_G9"] is True
    assert plan["constraints"]["no_protected_holdout"] is True
    assert plan["constraints"]["no_hold_exit_training"] is True


def _contrast(effect=0.0, low=-10.0, high=10.0, p=1.0):
    return {
        "observed_paired_pnl_difference": effect,
        "ci_p2_5": low,
        "ci_p97_5": high,
        "holm_adjusted_p": p,
    }


def _paired_payload():
    controls = {
        name: {
            "net_pnl": {"median": -1.0},
            "trades": {"median": 100.0},
            "joint_G1_G2_pass_count": 0,
        }
        for name in (
            "A",
            "B",
            "C",
            "D",
            "E",
            "E_common",
            "F",
            "G",
            "G_common",
            "H",
            "I",
            "C_G",
            "D_G",
            "M_F",
            "M_G",
            "B_real",
            "C_real",
            "D_real",
            "M_real",
        )
    }
    contrasts = {
        name: _contrast()
        for name in attribution.PRIMARY_CONTRASTS
    }
    return {
        "control_distributions": controls,
        "primary_contrasts": contrasts,
        "matched_G_M_G_exposure_quality": {"pass": True},
        "matched_H_M_real_exposure_quality": {"pass": True},
        "profile_matching_quality": {
            "D": {"valid_for_attribution": True},
            "D_G": {"valid_for_attribution": True},
            "M_F": {"valid_for_attribution": True},
            "M_G": {"valid_for_attribution": True},
        },
    }


def test_causal_attribution_does_not_call_uncertainty_random_luck() -> None:
    paired = _paired_payload()
    implementation = {
        "reproduced_weak_shuffle_defect": {"finding": "test"}
    }
    result = attribution._causal_attribution(paired, implementation)
    assert result["supported_mechanisms"] == []
    assert result["ordinary_random_luck_primary_explanation"] is False
    assert "not statistically distinguishable" in result["causal_conclusion"]
    assert result["strong_shuffle_equivalence_certified"] is True


def test_causal_attribution_uses_actual_g_minus_m_contrast() -> None:
    paired = _paired_payload()
    paired["primary_contrasts"][
        "G_minus_M_G_valid_negative_control_increment"
    ] = _contrast(effect=-123.0)
    implementation = {
        "reproduced_weak_shuffle_defect": {"finding": "test"}
    }
    result = attribution._causal_attribution(paired, implementation)
    assert (
        result["components_dollars_per_complete_campaign"][
            "valid_negative_control_G_minus_M_G"
        ]
        == -123.0
    )


def test_trust_rejects_excess_strong_shuffle_false_positives() -> None:
    paired = _paired_payload()
    paired["control_distributions"]["G"][
        "joint_G1_G2_pass_count"
    ] = 2
    causal = {
        "strong_negative_control_assessment": {
            "material": False,
            "holm_adjusted_p": 1.0,
        },
        "any_positive_material_effect_not_excluded": False,
    }
    decision = attribution._entry_trust_decision(paired, causal)
    assert decision["D1_revised_negative_control_pass"] is False
    assert (
        decision["status"]
        == "entry_campaign_remains_quarantined_negative_control_failed"
    )


def test_raw_d1_source_candidate_uses_destination_ladder_axes() -> None:
    contracts = np.asarray(
        [[f"S{strike}-{right}" for right in range(2)] for strike in range(21)],
        dtype=object,
    )
    ladder = np.zeros((21, 2, 15), dtype=float)
    ladder[:, :, 1] = np.arange(42, dtype=float).reshape(21, 2) + 1.0
    labels = np.zeros((21, 2, 7), dtype=float)
    labels[:, :, attribution.POLICY_INDEX] = (
        np.arange(42, dtype=float).reshape(21, 2) + 100.0
    )
    source = attribution._raw_source_candidate(
        {
            "contract_ids": contracts,
            "rights": ("C", "P"),
            "option_ladder": ladder,
            "labels_net_pnl": labels,
        },
        strike_index=7,
        right_index=1,
    )
    assert source == {
        "contract_id": "S7-1",
        "right": "P",
        "canonical_slot": 7,
        "entry_ask": 16.0,
        "dollar_pnl_label": 115.0,
    }


def test_matched_exposure_fails_if_one_seed_exceeds_limit() -> None:
    aggregated = []
    for seed in attribution.D1_SEEDS:
        aggregated.append(
            {
                "control": "G",
                "model_seed": seed,
                "trades": 100,
                "premium_at_risk_mean": 100.0,
                "candidate_holding_minutes_mean": 10.0,
                "executed_premium_at_risk_mean": 100.0,
                "executed_holding_minutes_mean": 10.0,
            }
        )
        for draw in range(attribution.RANDOM_DRAWS):
            aggregated.append(
                {
                    "control": "M_G",
                    "model_seed": seed,
                    "draw": draw,
                    "trades": (
                        120 if seed == attribution.D1_SEEDS[0] else 100
                    ),
                    "premium_at_risk_mean": 100.0,
                    "candidate_holding_minutes_mean": 10.0,
                    "executed_premium_at_risk_mean": 100.0,
                    "executed_holding_minutes_mean": 10.0,
                }
            )
    result = attribution._matched_exposure_quality(
        aggregated,
        left_control="G",
        right_control="M_G",
        seeds=attribution.D1_SEEDS,
    )
    assert result["median_executed_trade_count_drift"] == 0.0
    assert result["maximum_executed_trade_count_drift"] == pytest.approx(0.2)
    assert result["pass"] is False


def test_session_contrast_rejects_missing_session_axis() -> None:
    sessions = pd.DataFrame(
        [
            {
                "control": "G",
                "model_seed": 8600,
                "draw": 0,
                "fold": "F1",
                "session": "A",
                "net_pnl": 1.0,
            },
            {
                "control": "M_G",
                "model_seed": 8600,
                "draw": 0,
                "fold": "F1",
                "session": "B",
                "net_pnl": 1.0,
            },
        ]
    )
    with pytest.raises(
        attribution.AttributionError,
        match="paired contrast session loss",
    ):
        attribution._session_contrast(
            sessions,
            left="G",
            right="M_G",
            seeds=(8600,),
            random_right=True,
        )


def test_real_signal_requires_score_direction() -> None:
    paired = _paired_payload()
    paired["primary_contrasts"][
        "H_minus_M_real_margin_matched_real_increment"
    ] = _contrast(effect=1_000.0, low=500.0, high=1_500.0, p=0.01)
    paired["primary_contrasts"][
        "H_minus_B_real_total_real_model_increment"
    ] = _contrast(effect=1_000.0, low=500.0, high=1_500.0, p=0.01)
    paired["control_distributions"]["H"]["trades"]["median"] = 10.0
    causal = {
        "strong_negative_control_assessment": {
            "material": False,
            "holm_adjusted_p": 1.0,
        },
        "any_positive_material_effect_not_excluded": False,
    }
    decision = attribution._entry_trust_decision(paired, causal)
    assert decision["real_H2_P5_increment_credible"] is False


def test_unidentifiable_exact_margin_null_does_not_hide_primary_family() -> None:
    paired = _paired_payload()
    paired["profile_matching_quality"]["M_G"][
        "valid_for_attribution"
    ] = False
    paired["matched_G_M_G_exposure_quality"]["pass"] = False
    causal = {
        "strong_negative_control_assessment": {
            "material": False,
            "holm_adjusted_p": 1.0,
        },
        "any_positive_material_effect_not_excluded": False,
    }
    decision = attribution._entry_trust_decision(paired, causal)
    assert decision["D1_revised_negative_control_pass"] is True
