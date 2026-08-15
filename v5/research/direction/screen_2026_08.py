"""Replay and gate for the second G1 attempt, judged on accuracy.

The first attempt's gate (:mod:`v5.research.direction.gate`) is untouched. It
judges eighteen members on net points against ES friction under a six-criterion
conjunction, and it remains the record of how the closed attempt was scored.

This module scores the two members frozen in
:mod:`v5.research.direction.hypothesis_2026_08` against a **directional accuracy**
threshold, because that is what the option layer requires and because the corpus
spans a 7.08x range of 60-minute volatility, over which no single points bar is
comparable.

It computes an accuracy and a bootstrap bound. It reads no option data, fits
nothing, and tunes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from v5.research import statistics
from v5.research.direction import hypothesis_2026_08 as hypothesis
from v5.research.direction import loader
from v5.research.validation.replay_gate import block_bootstrap_lower_bound


class ScreenError(RuntimeError):
    """The screen cannot run as declared."""


EQUITY_HOLIDAY_LAST_BAR = "13:00"
SCORE_FIELD = "overnight_gap"
BOOTSTRAP_SEED = 4_242


def is_equity_holiday(bars: loader.SessionBars) -> bool:
    """A session ES trades short and SPXW does not trade at all.

    Derived from the data rather than a holiday calendar: all seven sessions
    independently verified against the owned option corpus close at 12:59, and
    the 13:14/13:15 half-day closes are sessions on which SPXW does trade.
    """

    return bars.minute_et[-1] <= EQUITY_HOLIDAY_LAST_BAR


def eligible_mask(
    sessions: Sequence[loader.SessionBars], features: pd.DataFrame
) -> np.ndarray:
    """The frozen declared calendar, from structural rules only.

    Excluded sessions stay in the price chain: the next session's gap is still
    measured against the excluded day's close, because that is the close the
    live system would have seen. That is why this filters the feature frame
    rather than the session list.
    """

    if len(sessions) != len(features):
        raise ScreenError("session list and feature frame are not aligned")
    holiday = np.array([is_equity_holiday(b) for b in sessions], bool)
    ok = (
        ~holiday
        & features["has_prior_session"].to_numpy(bool)
        & ~features["is_roll_boundary"].to_numpy(bool)
        & np.isfinite(features[SCORE_FIELD].to_numpy(float))
        & np.isfinite(features["entry_price"].to_numpy(float))
        & np.isfinite(
            features[f"exit_price_{hypothesis.HORIZON_MINUTES}m"].to_numpy(float)
        )
    )
    return ok


@dataclass(frozen=True)
class MemberScore:
    member: str
    direction: str
    sessions: int
    traded: int
    correct: int
    accuracy: float
    accuracy_lower_bound: float
    required_accuracy: float
    passed: bool
    accuracy_by_fold: tuple[float, ...]
    folds_above_required: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "member": self.member,
            "direction": self.direction,
            "sessions": self.sessions,
            "traded": self.traded,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
            "accuracy_lower_bound": round(self.accuracy_lower_bound, 6),
            "required_accuracy": self.required_accuracy,
            "passed": self.passed,
            "accuracy_by_fold": [round(v, 6) for v in self.accuracy_by_fold],
            "folds_above_required": self.folds_above_required,
        }


def replay_member(
    features: pd.DataFrame, mask: np.ndarray, *, direction: str
) -> pd.DataFrame:
    """One row per declared session: side, correctness, and fold.

    Correctness, not profit, because the threshold is an accuracy. A zero move
    is not a correct call and is counted against the member rather than dropped.
    """

    if direction not in hypothesis.DIRECTIONS:
        raise ScreenError(f"undeclared direction: {direction}")
    frame = features.loc[mask].reset_index(drop=True)
    score = frame[SCORE_FIELD].to_numpy(float)
    entry = frame["entry_price"].to_numpy(float)
    exit_price = frame[f"exit_price_{hypothesis.HORIZON_MINUTES}m"].to_numpy(float)

    side = np.sign(score)
    if direction == "against":
        side = -side
    traded = side != 0.0
    realized = exit_price - entry
    correct = traded & (side * realized > 0.0)

    folds = loader.chronological_folds(len(frame), hypothesis.FOLD_COUNT)
    return pd.DataFrame(
        {
            "session": frame["session"].to_numpy(),
            "fold": folds,
            "side": side,
            "traded": traded,
            "correct": correct,
        }
    )


def score_member(
    features: pd.DataFrame, mask: np.ndarray, *, direction: str, seed_offset: int = 0
) -> MemberScore:
    """Judge one declared member on directional accuracy."""

    rows = replay_member(features, mask, direction=direction)
    traded = rows.loc[rows["traded"]]
    if traded.empty:
        raise ScreenError(f"member {direction} traded no sessions")

    hits = traded["correct"].to_numpy(float)
    accuracy = float(hits.mean())

    # Bonferroni across the declared family, applied inside the gate so the null
    # campaign and the real replay are judged by identical arithmetic.
    confidence = 1.0 - (
        (1.0 - float(hypothesis.declaration()["confidence_level"]))
        / hypothesis.FAMILY_SIZE
    )
    lower = float(
        block_bootstrap_lower_bound(
            hits, confidence=confidence, seed=BOOTSTRAP_SEED + seed_offset
        )
    )

    by_fold: list[float] = []
    for fold in sorted(set(traded["fold"].tolist())):
        cell = traded.loc[traded["fold"] == fold, "correct"].to_numpy(float)
        by_fold.append(float(cell.mean()) if len(cell) else float("nan"))

    required = hypothesis.REQUIRED_ACCURACY
    return MemberScore(
        member=f"{hypothesis.MECHANISM}.{direction}.{hypothesis.HORIZON_MINUTES}m",
        direction=direction,
        sessions=int(len(rows)),
        traded=int(len(traded)),
        correct=int(traded["correct"].sum()),
        accuracy=accuracy,
        accuracy_lower_bound=lower,
        required_accuracy=required,
        passed=bool(lower > required),
        accuracy_by_fold=tuple(by_fold),
        folds_above_required=int(sum(v > required for v in by_fold)),
    )


def score_family(
    features: pd.DataFrame, mask: np.ndarray, *, seed_offset: int = 0
) -> tuple[MemberScore, ...]:
    """Both declared members, in declaration order."""

    return tuple(
        score_member(features, mask, direction=d, seed_offset=seed_offset)
        for d in hypothesis.DIRECTIONS
    )


def any_member_passed(scores: Sequence[MemberScore]) -> bool:
    """The familywise event a surrogate campaign measures.

    A screen reporting the better of two directions is making two attempts at
    significance whether or not it says so, which is why the Bonferroni sits
    inside :func:`score_member` rather than being applied afterwards.
    """

    return any(s.passed for s in scores)


def build_inputs(root: str | None = None):
    """Load the declared corpus and return (sessions, features, mask)."""

    sessions = loader.load_sessions(root=root or hypothesis.ES_BARS_ROOT)
    features = loader.session_features(sessions)
    mask = eligible_mask(sessions, features)
    declared = int(mask.sum())
    if declared != hypothesis.ELIGIBLE_SESSIONS:
        raise ScreenError(
            f"eligible index is {declared}, declaration froze "
            f"{hypothesis.ELIGIBLE_SESSIONS}; the corpus or the rules moved"
        )
    return sessions, features, mask
