from __future__ import annotations

import pandas as pd

from v4.scripts import run_protocol163_recent_data_integrity as p163_integrity


def test_expected_sessions_excludes_weekends_and_good_friday() -> None:
    assert p163_integrity.expected_sessions("2026-04-01", "2026-04-06") == [
        "2026-04-01",
        "2026-04-02",
        "2026-04-06",
    ]


def test_missing_minutes_detects_gap_without_flagging_duplicates() -> None:
    times = [
        pd.Timestamp("2026-05-18T13:31:00Z"),
        pd.Timestamp("2026-05-18T13:32:00Z"),
        pd.Timestamp("2026-05-18T13:34:00Z"),
    ]

    assert p163_integrity.missing_minutes(times) == ["2026-05-18T13:33:00+00:00"]
