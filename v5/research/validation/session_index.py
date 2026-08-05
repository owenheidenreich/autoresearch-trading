"""Collapse an authoritative trade ledger into the gate's per-session index.

The economic gate consumes one row per *calendar* session, including a zero on
every eligible day the policy did not trade.  Handing it only the sessions that
happened to trade silently inflates the per-session mean — the exact optimism
the gate exists to refuse.  This module is the missing seam between
``candidate_packet`` (per-trade rows) and ``replay_gate`` (per-session rows):
the eligible calendar and the fold assignment are declared up front, and every
trade must land inside them.

Model-free and network-free; it only sums columns it is given.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SESSION_INDEX_SCHEMA_VERSION = "v5.session-index.v1"


class SessionIndexError(RuntimeError):
    """The trade ledger and the declared calendar do not agree."""


def session_frame_from_trades(
    trades: pd.DataFrame,
    *,
    session_calendar: Sequence[str],
    fold_by_session: Mapping[str, str],
    net_column: str = "net_pnl",
    session_column: str = "session",
) -> pd.DataFrame:
    """One row per declared calendar session, zero-filled where no trade exists.

    ``session_calendar`` is the complete pre-declared eligible index — every
    session the policy *could* have traded, not merely the ones it did.  A trade
    on an undeclared session is refused rather than absorbed, and a declared
    session with no trades contributes an explicit ``0.0``.
    """

    calendar = [str(session) for session in session_calendar]
    if not calendar:
        raise SessionIndexError("the declared session calendar is empty")
    if len(set(calendar)) != len(calendar):
        raise SessionIndexError("the declared session calendar contains duplicates")
    if calendar != sorted(calendar):
        raise SessionIndexError("the declared session calendar is not chronological")
    missing_folds = [session for session in calendar if session not in fold_by_session]
    if missing_folds:
        raise SessionIndexError(
            "declared sessions have no fold assignment: "
            + ",".join(missing_folds[:5])
        )

    for column in (session_column, net_column):
        if column not in trades.columns:
            raise SessionIndexError(f"trade ledger missing column: {column}")
    net = pd.to_numeric(trades[net_column], errors="coerce")
    if len(trades) and (
        net.isna().any() or not np.isfinite(net.to_numpy(dtype=float)).all()
    ):
        raise SessionIndexError(f"trade ledger has invalid values in {net_column}")
    trade_sessions = trades[session_column].astype(str)
    undeclared = sorted(set(trade_sessions) - set(calendar))
    if undeclared:
        raise SessionIndexError(
            "trades occur on sessions outside the declared calendar: "
            + ",".join(undeclared[:5])
        )

    per_session = (
        pd.DataFrame({session_column: trade_sessions, net_column: net})
        .groupby(session_column)[net_column]
        .agg(["sum", "count"])
        if len(trades)
        else pd.DataFrame(columns=["sum", "count"])
    )
    rows = []
    for session in calendar:
        if session in per_session.index:
            total = float(per_session.loc[session, "sum"])
            count = int(per_session.loc[session, "count"])
        else:
            total, count = 0.0, 0
        rows.append(
            {
                "session": session,
                "fold": str(fold_by_session[session]),
                net_column: total,
                "trade_count": count,
            }
        )
    return pd.DataFrame(rows)
