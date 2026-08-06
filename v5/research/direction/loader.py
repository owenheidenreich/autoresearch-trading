"""Load owned ES bars and build the frozen G1 features, causally.

Two jobs, deliberately separated:

``load_sessions``
    Read the owned one-minute bars into one record per session, keeping the
    whole price path. The path is kept because the matched surrogate has to
    rebuild it and then re-derive *every* feature from the rebuilt path — a
    surrogate that only reshuffles a finished feature would not test the thing
    row 183 says must be tested.

``session_features``
    Turn those records into the causal fields declared in
    :mod:`v5.research.direction.family`. Nothing here may see its own session's
    future or any later session. The one cross-session field, the volume
    surprise, uses an expanding median over *strictly earlier* sessions.

This module reads bars but computes no economics: no side, no trade, no P&L.
That separation is what lets the surrogate campaign run the identical feature
code before any real outcome is inspected.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from v5.research import reservation
from v5.research.direction import family


class LoaderError(RuntimeError):
    """The owned bars do not support the frozen declaration."""


@dataclass(frozen=True)
class SessionBars:
    """One session's complete regular-hours path, indexed by ET minute."""

    session: str
    instrument_id: int
    minute_et: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def index_of(self, minute: str) -> int | None:
        try:
            return self.minute_et.index(minute)
        except ValueError:
            return None

    def close_at(self, minute: str) -> float | None:
        position = self.index_of(minute)
        return None if position is None else float(self.close[position])

    def open_at(self, minute: str) -> float | None:
        position = self.index_of(minute)
        return None if position is None else float(self.open[position])


def _minute_labels(index: pd.DatetimeIndex) -> tuple[str, ...]:
    return tuple(index.tz_convert("America/New_York").strftime("%H:%M"))


def load_sessions(root: str | Path = family.ES_BARS_ROOT) -> tuple[SessionBars, ...]:
    """Every non-empty owned session, chronologically, with its full path."""

    directory = Path(root)
    if not directory.is_dir():
        raise LoaderError(f"ES bar root not found: {directory}")
    sessions: list[SessionBars] = []
    for path in sorted(directory.glob("*.parquet")):
        session = path.name.split(".")[0]
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise LoaderError(f"{path} is not indexed by timestamp")
        frame = frame.sort_index()
        identifiers = frame["instrument_id"].unique()
        if len(identifiers) != 1:
            raise LoaderError(
                f"{session} mixes {len(identifiers)} instrument ids; the frozen "
                "declaration assumes one contract per session"
            )
        sessions.append(
            SessionBars(
                session=session,
                instrument_id=int(identifiers[0]),
                minute_et=_minute_labels(frame.index),
                open=frame["open"].to_numpy(float),
                high=frame["high"].to_numpy(float),
                low=frame["low"].to_numpy(float),
                close=frame["close"].to_numpy(float),
                volume=frame["volume"].to_numpy(float),
            )
        )
    if not sessions:
        raise LoaderError(f"no non-empty sessions under {directory}")

    # The confirmation reserve is enforced in code, not by memory. The owned
    # corpus ends well before it, so this should never fire -- which is exactly
    # why it is cheap to leave in place.
    reservation.assert_development_only(
        (s.session for s in sessions), purpose="G1 direction screen"
    )
    return tuple(sessions)


def _exit_minute(entry_minute: str, horizon_minutes: int) -> str:
    hours, minutes = (int(part) for part in entry_minute.split(":"))
    total = hours * 60 + minutes + horizon_minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def session_features(sessions: Sequence[SessionBars]) -> pd.DataFrame:
    """The frozen causal fields, one row per session, in chronological order.

    Every field is computable at 09:35:00 ET. ``volume_surprise`` is the only
    field that looks outside its own session, and it looks strictly backwards.
    """

    entry_minute = family.ENTRY_BAR_ET
    first_bar, last_feature_bar = family.FEATURE_BARS_ET[0], family.FEATURE_BARS_ET[-1]

    rows: list[dict[str, object]] = []
    previous_close: float | None = None
    previous_instrument: int | None = None

    for bars in sessions:
        open_0930 = bars.open_at(first_bar)
        close_0934 = bars.close_at(last_feature_bar)
        entry_price = bars.close_at(entry_minute)

        volume_5m = float("nan")
        positions = [bars.index_of(minute) for minute in family.FEATURE_BARS_ET]
        if all(position is not None for position in positions):
            volume_5m = float(sum(bars.volume[position] for position in positions))

        row: dict[str, object] = {
            "session": bars.session,
            "instrument_id": bars.instrument_id,
            "first_five_minute_return": (
                float("nan")
                if open_0930 is None or close_0934 is None
                else close_0934 - open_0930
            ),
            "first_five_minute_volume": volume_5m,
            "entry_price": float("nan") if entry_price is None else entry_price,
            "prior_session_final_close": (
                float("nan") if previous_close is None else previous_close
            ),
            "is_roll_boundary": (
                previous_instrument is not None
                and bars.instrument_id != previous_instrument
            ),
            "has_prior_session": previous_close is not None,
        }
        row["overnight_gap"] = (
            float("nan")
            if previous_close is None or open_0930 is None
            else open_0930 - previous_close
        )
        for horizon in family.HORIZON_MINUTES:
            exit_price = bars.close_at(_exit_minute(entry_minute, horizon))
            if exit_price is None:  # forced flat by the session close
                exit_price = float(bars.close[-1])
            row[f"exit_price_{horizon}m"] = float(exit_price)
        rows.append(row)

        previous_close = float(bars.close[-1])
        previous_instrument = bars.instrument_id

    frame = pd.DataFrame(rows)

    # Expanding median over STRICTLY earlier sessions. `shift(1)` is what keeps
    # a session out of its own baseline; without it the field would see itself.
    baseline = (
        frame["first_five_minute_volume"].expanding().median().shift(1)
    )
    frame["prior_volume_median"] = baseline
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["volume_surprise"] = frame["first_five_minute_volume"] / baseline
    frame.loc[~np.isfinite(frame["volume_surprise"]), "volume_surprise"] = float("nan")
    return frame


def eligible_sessions(features: pd.DataFrame, mechanism: str) -> pd.Series:
    """The declared calendar for a mechanism, as a boolean mask.

    M1 evaluates every non-empty session; its first session simply cannot trade
    because the volume baseline is undefined, and contributes an explicit zero.
    The gap mechanisms *exclude* roll boundaries and the first session outright,
    because there the gap is undefined rather than declined -- an undefined
    session is not a no-trade day and must not be averaged in as one.
    """

    if mechanism == "M1":
        return pd.Series(True, index=features.index)
    if mechanism in {"M3", "JOINT"}:
        return features["has_prior_session"] & ~features["is_roll_boundary"]
    raise LoaderError(f"unknown mechanism: {mechanism}")


def chronological_folds(count: int, fold_count: int) -> np.ndarray:
    """Contiguous chronological blocks, never shuffled.

    Earlier blocks absorb the remainder so the most recent fold is never the
    short one; a truncated final fold would quietly weight the newest regime
    least, and that is the regime a forward test will meet.
    """

    if count < fold_count:
        raise LoaderError(f"cannot split {count} sessions into {fold_count} folds")
    edges = np.linspace(0, count, fold_count + 1).round().astype(int)
    labels = np.empty(count, dtype=int)
    for fold in range(fold_count):
        labels[edges[fold] : edges[fold + 1]] = fold
    return labels
