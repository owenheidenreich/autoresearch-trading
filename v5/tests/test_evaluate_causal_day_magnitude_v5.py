from __future__ import annotations

import pandas as pd

from v5.ops.evaluate_causal_day_magnitude_v5 import (
    join_mid_only,
    select_composition_matched_control,
)
from v5.ops.write_causal_day_fit_declaration_v5 import build_declaration


def test_v5_declaration_recomputes_the_primary_count_and_reproduces_defect() -> None:
    declaration = build_declaration()
    assert declaration["fit_reuse"]["computed_trainable_parameters"] == 341
    assert declaration["fit_reuse"]["policy_fit_gate"].startswith("PERMITTED")
    assert declaration["v4_interpretation"]["negative_result_void"] is True
    assert declaration["primary_kill_cell"]["selector"]["whole_day_top_n_on_scored_session"] is False


def test_mid_only_join_does_not_require_or_read_a_bid_result() -> None:
    predictions = pd.DataFrame(
        {
            "session": ["2025-12-17"],
            "entry_minute": ["10:00"],
            "contract_id": ["C1"],
            "fold": [1],
            "predicted_depth_120m": [10.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "session": ["2025-12-17"],
            "entry_minute": ["10:00"],
            "contract_id": ["C1"],
            "right": ["C"],
            "entry_regime": ["morning"],
            "self_delta": [0.3],
            "entry_ask_usd": [110.0],
            "entry_mid_usd": [100.0],
            "spread_usd": [20.0],
            "moneyness_itm_points": [-5.0],
            "clock_exit_minute_120m": ["12:00"],
            "clock_exit_mid_value_120m": [2.0],
            "net_mid_120m_usd": [96.92],
        }
    )
    joined = join_mid_only(predictions, candidates)
    assert "net_bid_120m_usd" not in joined
    assert len(joined) == 1


def test_matched_control_uses_causal_exact_strata_and_excludes_model_trade() -> None:
    model = pd.DataFrame(
        {
            "session": ["2025-12-17"],
            "entry_minute": ["10:00"],
            "contract_id": ["MODEL"],
            "right": ["P"],
            "entry_regime": ["morning"],
            "self_delta": [-0.25],
            "entry_ask_usd": [200.0],
            "clock_exit_minute_120m": ["12:00"],
        }
    )
    population = pd.DataFrame(
        {
            "session": ["2025-12-17", "2025-12-17"],
            "entry_minute": ["10:00", "10:15"],
            "contract_id": ["MODEL", "CONTROL"],
            "right": ["P", "P"],
            "entry_regime": ["morning", "morning"],
            "self_delta": [-0.25, -0.24],
            "entry_ask_usd": [200.0, 205.0],
            "clock_exit_minute_120m": ["12:00", "12:15"],
        }
    )
    calibration = {
        "matched_control_quintiles": {
            "delta_internal_edges": [0.1, 0.2, 0.3, 0.4],
            "premium_internal_edges_usd": [100, 150, 250, 300],
        }
    }
    selected, failures = select_composition_matched_control(
        model, population, calibration=calibration
    )
    assert failures == []
    assert selected["contract_id"].tolist() == ["CONTROL"]
    assert "net_mid_120m_usd" not in selected
