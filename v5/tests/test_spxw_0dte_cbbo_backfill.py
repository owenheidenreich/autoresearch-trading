from __future__ import annotations

from v5.ops.estimate_spxw_0dte_cbbo_backfill import estimate_backfill


def test_backfill_estimate_uses_recorded_daily_costs() -> None:
    got = estimate_backfill(
        sessions=10,
        cbbo_costs={"a": 1.0, "b": 3.0},
        definition_costs={"a": 0.5, "b": 1.5},
    )

    assert got["recorded_cbbo_daily_mean_usd"] == 2.0
    assert got["recorded_definition_daily_mean_usd"] == 1.0
    assert got["estimated_cbbo_usd"] == 20.0
    assert got["estimated_definitions_usd"] == 10.0
    assert got["estimated_combined_usd"] == 30.0
    assert got["recommended_hard_cap_usd"] == 75
