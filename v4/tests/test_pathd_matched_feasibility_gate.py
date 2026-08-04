from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from v4.research.pathd_matched_feasibility_gate import (
    ES_FRICTION,
    NY,
    _es_outcome,
    _roll_exclusion,
    _session_terminal_ns,
    _tick,
)


def test_option_tick_and_forced_flat_are_dst_aware() -> None:
    assert _tick(2.95) == 0.05
    assert _tick(3.00) == 0.10
    summer = pd.Timestamp(_session_terminal_ns("2025-10-31"), unit="ns", tz="UTC").tz_convert(NY)
    winter = pd.Timestamp(_session_terminal_ns("2025-11-19"), unit="ns", tz="UTC").tz_convert(NY)
    assert (summer.hour, summer.minute) == (15, 55)
    assert (winter.hour, winter.minute) == (15, 55)


def test_roll_exclusion_matches_review_rule_change_plus_neighbors() -> None:
    files = []
    for index, instrument_id in enumerate((1, 1, 2, 2, 2)):
        session = f"2025-08-{index + 1:02d}"
        frame = pd.DataFrame({"instrument_id": [instrument_id], "close": [100.0]})
        files.append((session, Path(f"{session}.parquet"), frame))
    excluded, details = _roll_exclusion(files)
    assert excluded == {"2025-08-02", "2025-08-03", "2025-08-04"}
    assert details == [{"session": "2025-08-03", "prior_instrument_id": 1, "new_instrument_id": 2}]


def test_es_primary_is_two_tick_nonoverlapping_economics() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2025-10-06", "2025-10-06"],
            "outer_fold": [0, 0],
            "trade_index": [0, 1],
            "momentum_signal": [1.0, -1.0],
            "future_dollars": [50.0, -25.0],
        }
    )
    result = _es_outcome(
        {("2025-10-06", 15): frame},
        horizon=15,
        direction=1,
        friction=ES_FRICTION[2],
        label="test",
    )
    assert result["pnl"].tolist() == [20.5, -4.5]
