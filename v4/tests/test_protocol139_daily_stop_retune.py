from __future__ import annotations

from v4.scripts.run_protocol139_daily_stop_retune import decide, select_best


def test_select_best_prefers_lower_worst_day_then_incremental() -> None:
    rows = [
        {
            "passes": True,
            "worst_day_unstressed": -2000.0,
            "incremental_unstressed": 100.0,
        },
        {
            "passes": True,
            "worst_day_unstressed": -1800.0,
            "incremental_unstressed": 50.0,
        },
    ]

    assert select_best(rows)["worst_day_unstressed"] == -1800.0


def test_decide_rejects_negative_segment() -> None:
    assert (
        decide(
            {"passes": True, "total_pnl_unstressed": 200.0},
            [{"segment": "q", "incremental_pnl": -1.0}],
            {"total_pnl": 100.0},
        )
        == "reject_daily_stop_retune_segment_negative"
    )


def test_decide_passes_positive_segments_and_improvement() -> None:
    assert (
        decide(
            {"passes": True, "total_pnl_unstressed": 200.0},
            [{"segment": "q", "incremental_pnl": 1.0}],
            {"total_pnl": 100.0},
        )
        == "pass_daily_stop_retune_candidate_not_live"
    )

