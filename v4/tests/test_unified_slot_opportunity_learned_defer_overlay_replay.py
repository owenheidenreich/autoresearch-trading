from __future__ import annotations

import pandas as pd

from v4.model.unified_conservative_neural_policy import ConservativeNeuralPolicyConfig
from v4.model.unified_slot_opportunity_defer import SlotOpportunityDeferConfig
from v4.scripts.run_unified_slot_opportunity_learned_defer_overlay_replay import (
    q1_q3_stress_pass,
    select_learned_defer_entry_candidate,
)


def test_select_learned_defer_candidate_charges_estimated_slot_cost() -> None:
    candidates = pd.DataFrame(
        {
            "predicted_advantage": [900.0, 700.0],
            "positive_probability": [0.9, 0.9],
            "tail_probability": [0.1, 0.1],
            "entry_premium": [100.0, 100.0],
            "estimated_blocked_protocol101_cost": [700.0, 100.0],
            "blocked_cost_uncertainty": [0.0, 0.0],
            "estimated_blocked_entries": [1.0, 1.0],
        },
        index=[10, 20],
    )

    selected, reason = select_learned_defer_entry_candidate(
        candidates,
        equity=10_000.0,
        policy_config=ConservativeNeuralPolicyConfig(min_advantage_margin=250.0),
        defer_config=SlotOpportunityDeferConfig(min_net_advantage_margin=250.0),
    )

    assert selected == 20
    assert reason == "challenger_entry_selected_after_learned_slot_defer"


def test_select_learned_defer_candidate_rejects_too_many_estimated_blocked_entries() -> None:
    candidates = pd.DataFrame(
        {
            "predicted_advantage": [2_000.0],
            "positive_probability": [0.9],
            "tail_probability": [0.1],
            "entry_premium": [100.0],
            "estimated_blocked_protocol101_cost": [0.0],
            "blocked_cost_uncertainty": [0.0],
            "estimated_blocked_entries": [3.0],
        }
    )

    selected, reason = select_learned_defer_entry_candidate(
        candidates,
        equity=10_000.0,
        policy_config=ConservativeNeuralPolicyConfig(min_advantage_margin=250.0),
        defer_config=SlotOpportunityDeferConfig(max_blocked_protocol101_entries=1),
    )

    assert selected is None
    assert reason == "no_candidate_clears_learned_slot_defer_gate"


def test_q1_q3_stress_pass_requires_nonnegative_protected_deltas() -> None:
    assert q1_q3_stress_pass(
        [
            {
                "splits": {
                    "q1_2026": {"delta_vs_protocol101_same_scope": 0.0},
                    "q3_2025": {"delta_vs_protocol101_same_scope": 1.0},
                }
            }
        ]
    )
    assert not q1_q3_stress_pass(
        [
            {
                "splits": {
                    "q1_2026": {"delta_vs_protocol101_same_scope": -0.01},
                    "q3_2025": {"delta_vs_protocol101_same_scope": 1.0},
                }
            }
        ]
    )
