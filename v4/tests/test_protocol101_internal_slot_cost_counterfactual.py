from __future__ import annotations

import pandas as pd

from v4.scripts import run_protocol101_internal_slot_cost_counterfactual as slot_cost


def _hypothetical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["q1_2026", "q1_2026", "q1_2026", "q1_2026"],
            "fold": ["fold"] * 4,
            "seed": [1, 1, 1, 1],
            "session": ["2026-01-02"] * 4,
            "decision_dt": pd.to_datetime(
                [
                    "2026-01-02T15:00:00+00:00",
                    "2026-01-02T15:03:00+00:00",
                    "2026-01-02T15:05:00+00:00",
                    "2026-01-02T15:07:00+00:00",
                ],
                utc=True,
            ),
            "candidate_uid": ["same-entry", "blocked-a", "blocked-b", "at-exit"],
            "contract_id": ["SPXW-A-C", "SPXW-B-C", "SPXW-C-C", "SPXW-D-C"],
            "right": ["C", "C", "P", "P"],
            "offset": [0.0, -10.0, 10.0, 20.0],
            "score": [1.0, 2.0, 3.0, 4.0],
            "threshold": [0.5, 0.5, 0.5, 0.5],
            "candidate_pnl": [100.0, 400.0, -50.0, 900.0],
            "entry_ask": [10.0, 12.0, 8.0, 5.0],
            "entry_spread": [0.1, 0.2, 0.1, 0.1],
            "time_bucket": ["post_open_morning"] * 4,
        }
    )


def _actual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["q1_2026"],
            "fold": ["fold"],
            "seed": [1],
            "session": ["2026-01-02"],
            "entry_dt": pd.to_datetime(["2026-01-02T15:00:00+00:00"], utc=True),
            "exit_dt": pd.to_datetime(["2026-01-02T15:07:00+00:00"], utc=True),
            "candidate_uid": ["open"],
            "contract_id": ["SPXW-OPEN-C"],
            "right": ["C"],
            "pnl": [250.0],
            "exit_reason": ["sequence_residual_override"],
        }
    )


def test_attribute_blocked_entries_excludes_entry_and_exit_boundary() -> None:
    blocked = slot_cost.attribute_blocked_entries(_hypothetical(), _actual())

    assert blocked["blocked_candidate_uid"].tolist() == ["blocked-a", "blocked-b"]
    assert blocked.iloc[0]["open_minus_blocked_pnl"] == -150.0
    assert blocked.iloc[1]["open_minus_blocked_pnl"] == 300.0


def test_open_trade_slot_summary_uses_best_blocked_candidate_once() -> None:
    blocked = slot_cost.attribute_blocked_entries(_hypothetical(), _actual())
    summary = slot_cost.build_open_trade_slot_summary(blocked)

    assert len(summary) == 1
    assert summary.iloc[0]["blocked_entry_events"] == 2
    assert summary.iloc[0]["best_blocked_candidate_pnl"] == 400.0
    assert summary.iloc[0]["open_minus_best_blocked_pnl"] == -150.0
    assert bool(summary.iloc[0]["slot_cost_positive_vs_best_blocked"]) is True


def test_split_summary_reports_non_additive_blocked_view() -> None:
    hypothetical = _hypothetical()
    blocked = slot_cost.attribute_blocked_entries(hypothetical, _actual())
    open_slot = slot_cost.build_open_trade_slot_summary(blocked)
    summary = slot_cost.build_split_summary(hypothetical, blocked, open_slot)

    assert summary.iloc[0]["hypothetical_flat_entries"] == 4
    assert summary.iloc[0]["blocked_entry_events"] == 2
    assert summary.iloc[0]["open_trades_where_best_blocked_beats_open"] == 1
    assert summary.iloc[0]["best_blocked_minus_open_total"] == 150.0


def test_packet_decision_never_allows_training_or_challenge() -> None:
    hypothetical = _hypothetical()
    blocked = slot_cost.attribute_blocked_entries(hypothetical, _actual())
    open_slot = slot_cost.build_open_trade_slot_summary(blocked)
    split_summary = slot_cost.build_split_summary(hypothetical, blocked, open_slot)

    summary = slot_cost.build_summary(
        hypothetical,
        blocked,
        open_slot,
        split_summary,
        protocol092_dataset=pd.Timestamp("2026-01-02"),
        protocol101_dir=pd.Timestamp("2026-01-03"),
        actual_trades=pd.Timestamp("2026-01-04"),
    )

    assert summary["training_allowed"] is False
    assert summary["challenge_allowed"] is False
    assert "blocked_events_are_counterfactual_flat_upper_bound_not_additive_replay" in summary["blockers"]
