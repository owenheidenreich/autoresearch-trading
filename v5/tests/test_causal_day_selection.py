from __future__ import annotations

import pandas as pd
import pytest

from v5.research.causal_day_selection import (
    SelectorSpecificationError,
    assert_absolute_threshold_attainable,
    assert_cutoff_precedes_score_sessions,
    calibrate_rank_cutoff,
    select_rank_clock_trades,
)


def test_absolute_selector_refuses_the_clip_ceiling_and_above() -> None:
    assert_absolute_threshold_attainable(29.999)
    with pytest.raises(SelectorSpecificationError, match="strictly below"):
        assert_absolute_threshold_attainable(30.0)
    with pytest.raises(SelectorSpecificationError, match="strictly below"):
        assert_absolute_threshold_attainable(31.0)


def test_rank_cutoff_uses_only_minute_maxima_from_training_sessions() -> None:
    rows = []
    for session_index, session in enumerate(("2025-01-02", "2025-01-03")):
        for minute_index in range(5):
            for contract_index in range(2):
                rows.append(
                    {
                        "session": session,
                        "entry_minute": f"10:0{minute_index}",
                        "contract_id": f"{session_index}-{minute_index}-{contract_index}",
                        "score": session_index * 10 + minute_index + contract_index / 10,
                    }
                )
    cutoff = calibrate_rank_cutoff(
        pd.DataFrame(rows),
        fold=1,
        score_column="score",
        target_signal_minutes_per_session=2,
    )
    assert cutoff.training_sessions == 2
    assert cutoff.training_minutes == 10
    assert cutoff.target_rank == 4
    assert cutoff.cutoff_points == pytest.approx(11.1)
    assert cutoff.cutoff_points < cutoff.training_max_points


def test_rank_cutoff_must_precede_the_score_block() -> None:
    rows = pd.DataFrame(
        {
            "session": ["2025-01-02"] * 3,
            "entry_minute": ["10:00", "10:01", "10:02"],
            "score": [3.0, 2.0, 1.0],
        }
    )
    cutoff = calibrate_rank_cutoff(
        rows, fold=1, score_column="score", target_signal_minutes_per_session=2
    )
    assert_cutoff_precedes_score_sessions(cutoff, ["2025-01-03"])
    with pytest.raises(SelectorSpecificationError, match="strictly before"):
        assert_cutoff_precedes_score_sessions(cutoff, ["2025-01-02"])


def _scored(minute: str, score: float, exit_minute: str) -> dict:
    return {
        "session": "2025-02-03",
        "entry_minute": minute,
        "contract_id": minute,
        "fold": 1,
        "predicted_depth_120m": score,
        "spread_usd": 5.0,
        "moneyness_itm_points": -5.0,
        "clock_exit_minute_120m": exit_minute,
        "net_bid_120m_usd": 1.0,
        "net_mid_120m_usd": 2.0,
    }


def _cutoffs() -> dict[int, float]:
    return {fold: 5.0 for fold in range(1, 6)}


def test_rank_walk_is_causal_and_respects_occupancy_and_cap() -> None:
    base = pd.DataFrame(
        [
            _scored("10:00", 6.0, "10:30"),
            _scored("10:15", 100.0, "10:45"),
            _scored("10:31", 7.0, "11:00"),
            _scored("11:01", 8.0, "11:30"),
        ]
    )
    chosen = select_rank_clock_trades(
        base, horizon=120, fold_cutoffs=_cutoffs(), trade_cap=2
    )
    assert chosen["entry_minute"].tolist() == ["10:00", "10:31"]

    changed_future = base.copy()
    changed_future.loc[changed_future["entry_minute"].eq("11:01"), "predicted_depth_120m"] = 1e9
    changed = select_rank_clock_trades(
        changed_future, horizon=120, fold_cutoffs=_cutoffs(), trade_cap=2
    )
    assert changed["entry_minute"].tolist() == ["10:00", "10:31"]
