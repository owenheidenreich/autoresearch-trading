from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from v4.scripts.measure_pathd_es_bbo1s_spread import (
    _session_windows,
    _spread_summary,
    _weighted_quantile,
)


def test_fixed_utc_window_is_rejected_when_it_is_not_winter_rth() -> None:
    scope = {
        "sessions": ["2025-11-19"],
        "session_window_utc": {"start": "13:30", "end": "20:00"},
    }
    with pytest.raises(SystemExit, match="not full New York RTH"):
        _session_windows(scope)


def test_dst_aware_session_windows_accept_exact_new_york_rth() -> None:
    scope = {
        "sessions": ["2025-10-31", "2025-11-19"],
        "session_windows_utc": {
            "2025-10-31": {"start": "2025-10-31T13:30:00Z", "end": "2025-10-31T20:00:00Z"},
            "2025-11-19": {"start": "2025-11-19T14:30:00Z", "end": "2025-11-19T21:00:00Z"},
        },
    }
    windows = _session_windows(scope)
    assert windows["2025-10-31"][0] == datetime.fromisoformat("2025-10-31T13:30:00+00:00")
    assert windows["2025-11-19"][1] == datetime.fromisoformat("2025-11-19T21:00:00+00:00")


def test_spread_distribution_is_time_weighted() -> None:
    frame = pd.DataFrame(
        {
            "spread_ticks": [1.0, 2.0, 4.0],
            "duration_seconds": [8.0, 1.0, 1.0],
        }
    )
    summary = _spread_summary(frame)
    assert summary["mean_ticks"] == pytest.approx(1.4)
    assert summary["median_ticks"] == 1.0
    assert summary["share_one_tick"] == pytest.approx(0.8)
    assert summary["share_two_ticks"] == pytest.approx(0.1)
    assert summary["share_three_plus_ticks"] == pytest.approx(0.1)
    assert _weighted_quantile(frame["spread_ticks"].to_numpy(), frame["duration_seconds"].to_numpy(), 0.75) == 1.0
