from __future__ import annotations

import datetime as dt

from tools.run_ibkr_mock_training import _previous_business_day


def test_previous_business_day_from_weekday() -> None:
    # Tuesday -> Monday
    got = _previous_business_day(dt.date(2026, 3, 17))
    assert got == dt.date(2026, 3, 16)


def test_previous_business_day_from_monday() -> None:
    # Monday -> Friday
    got = _previous_business_day(dt.date(2026, 3, 16))
    assert got == dt.date(2026, 3, 13)

