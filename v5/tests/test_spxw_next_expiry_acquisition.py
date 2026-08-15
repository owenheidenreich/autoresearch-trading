from __future__ import annotations

import pandas as pd

from v5.ops.estimate_spxw_next_expiry_acquisition import estimate_costs


def test_estimate_uses_recorded_row_rate_and_next_expiry_symbol_count() -> None:
    recent = pd.DataFrame(
        {
            "session": ["a", "b"],
            "rows_0dte": [100, 100],
            "symbols_0dte": [10, 10],
            "symbols_next_expiry": [10, 20],
        }
    )

    got = estimate_costs(
        recent,
        cbbo_cost_by_session={"a": 1.0, "b": 1.0},
        definition_cost_by_session={"a": 0.5, "b": 0.5},
        target_sessions=2,
    )

    assert got["weighted_recorded_cbbo_usd_per_row"] == 0.01
    assert got["owned_0dte_rows_per_symbol"] == 10.0
    assert got["estimated_next_expiry_cbbo_usd"] == 3.0
    assert got["estimated_definitions_usd"] == 1.0
    assert got["estimated_combined_usd"] == 4.0
    assert got["recommended_hard_cap_usd"] == 50
