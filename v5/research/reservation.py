"""The forward confirmation reservation, as code rather than a promise.

The protected historical holdout is spent, so sessions from the declared start
date onward are the project's only remaining confirmation evidence.  A
declaration in a document is only as strong as the next person's memory; this
module makes a research run refuse reserved sessions instead of averaging them
in.

Signed declaration:
``v5/governance/FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md``.

The rule this enforces is the declaration's rule 1: strategy and policy
*economics* on reserved sessions are off limits.  Infrastructure evidence that
evaluates no policy — arrival latency, feed parity, capture completeness — is
permitted, which is why ``assert_development_only`` is called by research code
and not by capture code.

Model-free and network-free.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


RESERVATION_START = "2026-08-06"
DEVELOPMENT_ENDS = "2026-08-05"
DECLARATION = "v5/governance/FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md"


class ReservationError(RuntimeError):
    """A research run tried to read sessions reserved for confirmation."""


def is_reserved(session: str) -> bool:
    """Whether a ``YYYY-MM-DD`` session falls inside the confirmation reserve."""

    return str(session) >= RESERVATION_START


def reserved_sessions(sessions: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(s) for s in sessions if is_reserved(s)}))


def assert_development_only(sessions: Iterable[str], *, purpose: str) -> None:
    """Refuse a research index that reaches into the confirmation reserve.

    ``purpose`` names the analysis, so the error says what was refused rather
    than only that something was.
    """

    reserved = reserved_sessions(sessions)
    if reserved:
        shown = ", ".join(reserved[:5])
        more = f" (+{len(reserved) - 5} more)" if len(reserved) > 5 else ""
        raise ReservationError(
            f"{purpose} would read {len(reserved)} reserved session(s): {shown}{more}. "
            f"Development data ends {DEVELOPMENT_ENDS}; sessions from {RESERVATION_START} "
            f"are confirmation-only and open once under a pre-registered protocol. "
            f"See {DECLARATION}."
        )


def development_frame(
    frame: pd.DataFrame, *, session_column: str = "session", purpose: str
) -> pd.DataFrame:
    """Return ``frame`` unchanged, after proving it holds no reserved session.

    Deliberately not a filter.  Silently dropping reserved rows would let a
    research run keep going with a quietly different index; the run should stop
    and its index be corrected upstream.
    """

    if session_column not in frame.columns:
        raise ReservationError(f"frame has no session column: {session_column}")
    assert_development_only(frame[session_column].astype(str), purpose=purpose)
    return frame
