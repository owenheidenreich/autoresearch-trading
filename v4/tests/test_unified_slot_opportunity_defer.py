from __future__ import annotations

import numpy as np
import pandas as pd

from v4.model.unified_slot_opportunity_defer import (
    SlotOpportunityDeferConfig,
    blocked_protocol101_cost_for_interval,
    oracle_net_vs_blocked_protocol101,
    slot_opportunity_defer_decision,
)


def test_slot_opportunity_defer_charges_blocked_baseline_cost() -> None:
    decision = slot_opportunity_defer_decision(
        predicted_challenger_advantage=800.0,
        estimated_blocked_protocol101_cost=400.0,
        blocked_cost_uncertainty=100.0,
        config=SlotOpportunityDeferConfig(min_net_advantage_margin=250.0),
    )

    assert decision["allowed"]
    assert decision["adjusted_advantage"] == 300.0


def test_slot_opportunity_defer_rejects_when_cost_consumes_margin() -> None:
    decision = slot_opportunity_defer_decision(
        predicted_challenger_advantage=800.0,
        estimated_blocked_protocol101_cost=700.0,
        blocked_cost_uncertainty=0.0,
        config=SlotOpportunityDeferConfig(min_net_advantage_margin=250.0),
    )

    assert not decision["allowed"]
    assert decision["decision"] == "defer_to_protocol101"


def test_slot_opportunity_defer_rejects_too_many_blocked_entries() -> None:
    decision = slot_opportunity_defer_decision(
        predicted_challenger_advantage=2_000.0,
        estimated_blocked_protocol101_cost=0.0,
        estimated_blocked_entries=2,
        config=SlotOpportunityDeferConfig(max_blocked_protocol101_entries=1),
    )

    assert not decision["allowed"]


def test_oracle_net_vs_blocked_protocol101_is_diagnostic_difference() -> None:
    assert oracle_net_vs_blocked_protocol101(challenger_pnl=500.0, blocked_protocol101_pnl=700.0) == -200.0


def test_blocked_protocol101_cost_for_interval_includes_current_entry_and_excludes_exit_boundary() -> None:
    times = np.asarray(
        [
            np.datetime64("2026-01-02T15:00:00"),
            np.datetime64("2026-01-02T15:05:00"),
            np.datetime64("2026-01-02T15:10:00"),
        ]
    )
    pnl = np.asarray([100.0, 200.0, 300.0])

    result = blocked_protocol101_cost_for_interval(
        times,
        pnl,
        start=pd.Timestamp("2026-01-02T15:00:00"),
        end=pd.Timestamp("2026-01-02T15:10:00"),
        slippage_per_side=0.10,
    )

    assert result == {"blocked_entries": 2, "blocked_pnl": 260.0}
