"""Shared historical/live adapter for the Phase-0 Path-D contract clock.

Both paths are deliberately aliases of :func:`contract_clock_features`.  The
receipt generator hashes this file and proves value identity over the owned
same-session OPRA definition universe.  No vendor or broker access occurs here.
"""
from __future__ import annotations

from datetime import date, datetime, time
from types import MappingProxyType
from typing import Final, Mapping
from zoneinfo import ZoneInfo


NY: Final = ZoneInfo("America/New_York")
REGULAR_OPEN_ET: Final = time(9, 30)
REGULAR_CLOSE_ET: Final = time(16, 0)

# Frozen Cboe U.S. options RTH short sessions covering the owned development
# corpus and the 2026 live-certification year.  This is hashed as part of the
# exchange-calendar receipt; changes invalidate the receipt and ledger.
EARLY_CLOSE_TIMES_ET: Final[Mapping[str, time]] = MappingProxyType(
    {
        "2024-11-29": time(13, 0),
        "2024-12-24": time(13, 15),
        "2025-07-03": time(13, 0),
        "2025-11-28": time(13, 0),
        "2025-12-24": time(13, 15),
        "2026-07-02": time(13, 0),
        "2026-11-27": time(13, 0),
        "2026-12-24": time(13, 15),
    }
)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    return value.astimezone(ZoneInfo("UTC"))


def _parse_osi(osi_symbol: str) -> tuple[date, str, float]:
    if len(osi_symbol) != 21:
        raise ValueError(f"invalid padded OSI length: {osi_symbol!r}")
    root, expiry_code, right, strike_digits = (
        osi_symbol[:6].strip(),
        osi_symbol[6:12],
        osi_symbol[12],
        osi_symbol[13:21],
    )
    if root != "SPXW" or right not in {"C", "P"}:
        raise ValueError(f"unsupported SPXW OSI geometry: {osi_symbol!r}")
    if not expiry_code.isdigit() or not strike_digits.isdigit():
        raise ValueError(f"non-numeric SPXW OSI geometry: {osi_symbol!r}")
    expiry = datetime.strptime(expiry_code, "%y%m%d").date()
    return expiry, right, int(strike_digits) / 1000.0


def session_bounds(session: date) -> tuple[datetime, datetime, bool]:
    """Return the frozen Cboe options RTH bounds for one known session."""

    session_key = session.isoformat()
    close_time = EARLY_CLOSE_TIMES_ET.get(session_key, REGULAR_CLOSE_ET)
    opened = datetime.combine(session, REGULAR_OPEN_ET, tzinfo=NY)
    closed = datetime.combine(session, close_time, tzinfo=NY)
    return opened, closed, session_key in EARLY_CLOSE_TIMES_ET


def contract_clock_features(osi_symbol: str, decision_time: datetime) -> dict[str, float]:
    """Build all eight contract-clock fields from an OSI symbol and one clock."""

    expiry, right, strike = _parse_osi(osi_symbol)
    decision_utc = _as_utc(decision_time)
    opened, closed, early_close = session_bounds(expiry)
    opened_utc = opened.astimezone(ZoneInfo("UTC"))
    closed_utc = closed.astimezone(ZoneInfo("UTC"))
    if not opened_utc <= decision_utc <= closed_utc:
        raise ValueError("decision_time is outside the SPXW expiration session")
    seconds_from_open = (decision_utc - opened_utc).total_seconds()
    seconds_to_close = (closed_utc - decision_utc).total_seconds()
    return {
        "is_call": float(right == "C"),
        "strike": float(strike),
        "seconds_to_expiry": float(seconds_to_close),
        "seconds_from_open": float(seconds_from_open),
        "seconds_to_close": float(seconds_to_close),
        "minute_of_session": float(seconds_from_open // 60),
        "day_of_week": float(expiry.weekday()),
        "is_early_close": float(early_close),
    }


def historical_contract_clock_features(
    osi_symbol: str, decision_time: datetime
) -> dict[str, float]:
    """Historical path: intentionally delegates to the shared implementation."""

    return contract_clock_features(osi_symbol, decision_time)


def live_contract_clock_features(
    osi_symbol: str, decision_time: datetime
) -> dict[str, float]:
    """Live path: intentionally delegates to the shared implementation."""

    return contract_clock_features(osi_symbol, decision_time)
