"""Day-type classification for Phase 5 reporting. **Diagnostic only.**

**This is not a feature source and may not become one here.** The owner's
2026-08-19 ruling closed the event calendar: no calendar data is acquired and no
calendar feature is admitted. What the ruling *does* require is that Phase 5 be
able to slice its results by day type, so that assumptions are verified rather
than assumed. That is all this module does. Nothing here may be read by the
tensorizer, the adapter, the probe or the architecture.

**Two different kinds of fact live here, and conflating them would be the
mistake.**

* **FOMC announcement dates are recorded data.** They cannot be derived from a
  date and there is no source for them in this project, so they are transcribed
  from the Federal Reserve's published calendar, owner-supplied and
  owner-verified on 2026-08-19. Transcription, not acquisition.
* **Everything else is arithmetic on the session date.** OPEX, quarterly OPEX,
  month end, last Friday, month of year and weekday need no vendor, no purchase
  and no admission process. If any of them were ever wanted as a *feature*, the
  blocker would be a design decision about the parameter contract -- not data
  availability. That is a different question from the one the ruling closed.

**The FOMC list was verified rather than trusted.** Against the corpus's own
tape, the 33 listed dates present in it show **1.91x** the median one-minute
maximum step (13.74 against 7.18 index points) and **1.39x** the median session
range. The owner's stated reason for sitting out FOMC days -- elevated
volatility -- is measurable in this project's own data, independently of the
source the dates came from.

Design reference: `governance/WORK_ALLOCATION_MEMO_2026_08_18.md` section 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

SCHEMA_VERSION = "v5.session-calendar.v1"

#: FOMC policy-announcement dates, 2022-06 through 2026-07 inclusive.
#:
#: The **second day of each regularly scheduled meeting**, when the statement is
#: released -- not the later minutes-release dates. Source: the Federal Reserve's
#: published FOMC calendars; supplied and verified by the repository owner on
#: 2026-08-19.
#:
#: Eight meetings a year, and the counts here are 5/8/8/8/5 for 2022(from June)
#: /2023/2024/2025/2026(to July), which is what a complete list must look like.
#:
#: **2024-11-07 is a Thursday and that is correct**, not a transcription slip:
#: the November 2024 meeting was moved to the 6th and 7th around the US general
#: election on the 5th. Every other date is a Wednesday.
FOMC_ANNOUNCEMENTS: tuple[str, ...] = (
    "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
)

FOMC_SOURCE = (
    "Federal Reserve published FOMC calendars "
    "(federalreserve.gov/monetarypolicy/fomccalendars.htm); "
    "owner-supplied and owner-verified 2026-08-19"
)

QUARTERLY_OPEX_MONTHS = (3, 6, 9, 12)

DAY_TYPE_COLUMNS = (
    "is_fomc",
    "is_opex",
    "is_quarterly_opex",
    "is_month_end",
    "is_last_friday",
    "month",
    "weekday",
)


class SessionCalendarError(RuntimeError):
    """A session list cannot be classified as declared."""


def _ordered(sessions: Iterable[str]) -> list[str]:
    values = sorted({str(session) for session in sessions})
    if not values:
        raise SessionCalendarError("no session supplied")
    return values


def _stamps(sessions: Sequence[str]) -> pd.DatetimeIndex:
    stamps = pd.to_datetime(pd.Series(sessions), errors="coerce")
    if stamps.isna().any():
        bad = [s for s, ok in zip(sessions, stamps.notna(), strict=True) if not ok]
        raise SessionCalendarError(f"unparseable session dates: {bad[:5]}")
    return pd.DatetimeIndex(stamps)


def opex_days(sessions: Iterable[str]) -> list[str]:
    """The monthly option expiration -- what Pickles calls MOPEX.

    The third Friday of the month, **or the last session on or before it** when
    that Friday is a market holiday, which is how the expiration itself moves.
    Derived from the supplied session list rather than a holiday calendar, so it
    is exact for whatever calendar the corpus actually contains.
    """

    ordered = _ordered(sessions)
    stamps = _stamps(ordered)
    frame = pd.DataFrame({"session": ordered, "stamp": stamps})
    out: list[str] = []
    for (year, month), group in frame.groupby([stamps.year, stamps.month], sort=True):
        third_friday = _third_friday(int(year), int(month))
        eligible = group[group["stamp"] <= third_friday]
        if not eligible.empty:
            out.append(str(eligible.iloc[-1]["session"]))
    return out


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    # Friday is weekday 4; step to the first Friday, then two weeks on.
    offset = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 14)


def month_end_days(sessions: Iterable[str]) -> list[str]:
    """The last *session* of each month, not the last calendar day."""

    ordered = _ordered(sessions)
    stamps = _stamps(ordered)
    frame = pd.DataFrame({"session": ordered, "stamp": stamps})
    return [
        str(group.iloc[-1]["session"])
        for _, group in frame.groupby([stamps.year, stamps.month], sort=True)
    ]


def last_friday_days(sessions: Iterable[str]) -> list[str]:
    """The last Friday session of each month.

    Called out separately from month end because the owner names it separately:
    a last Friday is sometimes also the monthly expiration, and the point of
    reporting both is to see when they coincide.
    """

    ordered = _ordered(sessions)
    stamps = _stamps(ordered)
    frame = pd.DataFrame({"session": ordered, "stamp": stamps})
    fridays = frame[stamps.weekday == 4]
    if fridays.empty:
        return []
    friday_stamps = pd.DatetimeIndex(fridays["stamp"])
    return [
        str(group.iloc[-1]["session"])
        for _, group in fridays.groupby([friday_stamps.year, friday_stamps.month], sort=True)
    ]


@dataclass(frozen=True)
class CalendarCoverage:
    """What the classifier could and could not resolve, reported not assumed."""

    sessions: int
    fomc_in_range: int
    fomc_present: int
    fomc_absent: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "sessions": self.sessions,
            "fomc_announcements_in_range": self.fomc_in_range,
            "fomc_present_in_sessions": self.fomc_present,
            "fomc_absent_from_sessions": list(self.fomc_absent),
            "fomc_source": FOMC_SOURCE,
        }


def classify(sessions: Iterable[str]) -> pd.DataFrame:
    """One row per session carrying every declared day type."""

    ordered = _ordered(sessions)
    stamps = _stamps(ordered)
    opex = set(opex_days(ordered))
    frame = pd.DataFrame({"session": ordered})
    frame["is_fomc"] = frame["session"].isin(FOMC_ANNOUNCEMENTS)
    frame["is_opex"] = frame["session"].isin(opex)
    frame["is_quarterly_opex"] = frame["is_opex"] & pd.Series(
        stamps.month.isin(QUARTERLY_OPEX_MONTHS), index=frame.index
    )
    frame["is_month_end"] = frame["session"].isin(set(month_end_days(ordered)))
    frame["is_last_friday"] = frame["session"].isin(set(last_friday_days(ordered)))
    frame["month"] = stamps.month
    frame["weekday"] = stamps.day_name()
    return frame.loc[:, ["session", *DAY_TYPE_COLUMNS]]


def coverage(sessions: Iterable[str]) -> CalendarCoverage:
    """Which FOMC days the session set actually contains.

    Reported rather than assumed, because an absent one is not a null result:
    **2025-07-30 is an FOMC day the clock gate excluded from the corpus** for
    three missing interior minutes, so the all-sessions figures are already
    missing an FOMC day before anything is excluded on purpose.
    """

    ordered = _ordered(sessions)
    present = set(ordered)
    first, last = ordered[0], ordered[-1]
    in_range = [d for d in FOMC_ANNOUNCEMENTS if first <= d <= last]
    absent = tuple(d for d in in_range if d not in present)
    return CalendarCoverage(
        sessions=len(ordered),
        fomc_in_range=len(in_range),
        fomc_present=len(in_range) - len(absent),
        fomc_absent=absent,
    )
