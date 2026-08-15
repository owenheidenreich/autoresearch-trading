from __future__ import annotations

import pandas as pd

from v4.scripts.run_unified_slot_opportunity_cost_label_dataset import build_labels, decide


def test_build_labels_prices_blocked_protocol101_entries_with_stress_columns() -> None:
    flat = pd.DataFrame(
        {
            "split": ["q1_2026", "q1_2026"],
            "session": ["2026-01-02", "2026-01-02"],
            "decision_time": ["2026-01-02T15:00:00Z", "2026-01-02T15:11:00Z"],
            "decision_dt": pd.to_datetime(["2026-01-02T15:00:00Z", "2026-01-02T15:11:00Z"], utc=True),
            "candidate_uid": ["cand_a", "cand_b"],
            "contract_id": ["A", "B"],
            "right": ["C", "P"],
            "offset": [0, 5],
            "entry_premium": [10.0, 12.0],
            "candidate_exit_time": ["2026-01-02T15:10:00Z", "2026-01-02T15:20:00Z"],
            "candidate_exit_dt": pd.to_datetime(["2026-01-02T15:10:00Z", "2026-01-02T15:20:00Z"], utc=True),
        }
    )
    baseline = pd.DataFrame(
        {
            "split": ["q1_2026", "q1_2026", "q1_2026"],
            "session": ["2026-01-02", "2026-01-02", "2026-01-02"],
            "decision_dt": pd.to_datetime(
                ["2026-01-02T15:00:00Z", "2026-01-02T15:05:00Z", "2026-01-02T15:10:00Z"],
                utc=True,
            ),
            "protocol101_action": ["enter", "enter", "enter"],
            "contract_id": ["A", "B", "C"],
            "surface_candidate_uid": ["base_a", "base_b", "base_c"],
            "baseline_trade_pnl": [100.0, 200.0, 300.0],
        }
    )

    labels = build_labels(flat, baseline)

    first = labels.loc[labels["candidate_uid"].eq("cand_a")].iloc[0]
    assert first["blocked_protocol101_entries"] == 2
    assert first["blocked_protocol101_pnl_0_00"] == 300.0
    assert first["blocked_protocol101_pnl_0_10"] == 260.0
    assert first["blocked_protocol101_pnl_0_25"] == 200.0
    assert first["current_protocol101_action"] == "enter"
    assert first["current_protocol101_surface_candidate_uid"] == "base_a"

    second = labels.loc[labels["candidate_uid"].eq("cand_b")].iloc[0]
    assert second["blocked_protocol101_entries"] == 0
    assert second["current_protocol101_action"] == "wait"
    assert decide(labels) == "slot_opportunity_cost_labels_ready_for_causal_estimator"
