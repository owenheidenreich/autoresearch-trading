"""Day-type classification must be right about days people can check."""
from __future__ import annotations

import pandas as pd
import pytest

from v5.research.session_calendar import (
    FOMC_ANNOUNCEMENTS,
    QUARTERLY_OPEX_MONTHS,
    SessionCalendarError,
    classify,
    coverage,
    last_friday_days,
    month_end_days,
    opex_days,
)


def _weekdays(start: str, end: str) -> list[str]:
    days = pd.bdate_range(start, end)
    return [d.strftime("%Y-%m-%d") for d in days]


# ------------------------------------------------------------------ the list


def test_the_fomc_list_has_eight_meetings_a_year() -> None:
    """The shape a complete list must have; a dropped date shows up here."""

    stamps = pd.to_datetime(pd.Series(FOMC_ANNOUNCEMENTS))
    counts = stamps.dt.year.value_counts().to_dict()
    assert len(FOMC_ANNOUNCEMENTS) == len(set(FOMC_ANNOUNCEMENTS)) == 34
    assert counts[2023] == counts[2024] == counts[2025] == 8
    assert counts[2022] == 5  # June onward
    assert counts[2026] == 5  # through July


def test_every_announcement_is_a_weekday_and_only_one_is_not_a_wednesday() -> None:
    """2024-11-07 is the exception and it is real, not a transcription slip.

    The November 2024 meeting moved to the 6th and 7th around the US general
    election on the 5th. Pinning it stops a future tidy-up "correcting" it.
    """

    stamps = pd.to_datetime(pd.Series(FOMC_ANNOUNCEMENTS))
    assert (stamps.dt.weekday < 5).all()
    exceptions = [
        date
        for date, day in zip(FOMC_ANNOUNCEMENTS, stamps.dt.day_name(), strict=True)
        if day != "Wednesday"
    ]
    assert exceptions == ["2024-11-07"]


# ------------------------------------------------------------------ arithmetic


def test_opex_is_the_third_friday() -> None:
    """Checkable by hand: March 2024's third Friday is the 15th."""

    sessions = _weekdays("2024-03-01", "2024-03-31")
    assert "2024-03-15" in opex_days(sessions)
    assert len(opex_days(sessions)) == 1


def test_opex_falls_back_to_the_prior_session_when_the_third_friday_is_shut() -> None:
    """Good Friday 2024 was 2024-03-29, not an expiration -- so use a month
    whose third Friday is a holiday: 2023-04-07 was Good Friday, and April
    2023's third Friday (the 21st) traded. Build the shut case explicitly."""

    sessions = [d for d in _weekdays("2024-03-01", "2024-03-31") if d != "2024-03-15"]
    assert opex_days(sessions) == ["2024-03-14"]


def test_quarterly_opex_is_flagged_only_in_the_witching_months() -> None:
    sessions = _weekdays("2024-01-01", "2024-12-31")
    frame = classify(sessions).set_index("session")
    quarterly = frame[frame["is_quarterly_opex"]]
    assert set(pd.to_datetime(pd.Series(quarterly.index)).dt.month) == set(QUARTERLY_OPEX_MONTHS)
    assert quarterly["is_opex"].all()


def test_month_end_is_the_last_session_not_the_last_calendar_day() -> None:
    """2024-06-30 was a Sunday; the month ends at Friday the 28th."""

    sessions = _weekdays("2024-06-01", "2024-06-30")
    assert month_end_days(sessions) == ["2024-06-28"]


def test_a_month_end_that_is_also_a_last_friday_is_flagged_as_both() -> None:
    """The owner's case: a last Friday sometimes coincides with something else."""

    sessions = _weekdays("2024-06-01", "2024-06-30")
    frame = classify(sessions).set_index("session")
    assert frame.loc["2024-06-28", "is_month_end"]
    assert frame.loc["2024-06-28", "is_last_friday"]


def test_last_friday_and_month_end_can_differ() -> None:
    """2024-07-31 was a Wednesday, so the last Friday is the 26th."""

    sessions = _weekdays("2024-07-01", "2024-07-31")
    assert month_end_days(sessions) == ["2024-07-31"]
    assert last_friday_days(sessions) == ["2024-07-26"]


# ------------------------------------------------------------------ coverage


def test_classification_emits_one_row_per_session() -> None:
    sessions = _weekdays("2024-01-01", "2024-03-31")
    frame = classify(sessions)
    assert len(frame) == len(sessions)
    assert frame["session"].is_unique
    assert frame["is_fomc"].sum() == 2  # 2024-01-31 and 2024-03-20


def test_an_absent_fomc_day_is_reported_rather_than_silently_zero() -> None:
    """2025-07-30 is an FOMC day the clock gate kept out of the corpus.

    An FOMC-excluded cut that did not know this would describe itself as
    excluding every FOMC day when one had already gone missing for an unrelated
    reason.
    """

    sessions = [d for d in _weekdays("2025-07-01", "2025-08-29") if d != "2025-07-30"]
    result = coverage(sessions)
    assert "2025-07-30" in result.fomc_absent
    assert result.fomc_present == result.fomc_in_range - len(result.fomc_absent)


def test_an_empty_session_list_is_refused() -> None:
    with pytest.raises(SessionCalendarError, match="no session supplied"):
        classify([])


def test_an_unparseable_session_is_refused_rather_than_dropped() -> None:
    with pytest.raises(SessionCalendarError, match="unparseable"):
        classify(["2024-01-02", "not-a-date"])
