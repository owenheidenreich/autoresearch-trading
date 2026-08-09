"""Matched surrogates for the G1 known-answer campaign.

A surrogate session preserves everything about a real session **except the thing
under test**.  The whole screen's credibility rests on this being right, because
the campaign it feeds is what decides whether a positive G1 result means
anything.

Why matched, and why session shuffling is forbidden
---------------------------------------------------

Ledger row 183 recorded a screen whose headline ``+0.522`` rank correlation was
78.5% reproducible by surrogates with no predictability by construction, because
its features and its target shared the price level ``P(t)``.  A session shuffle
returns a clean null on that construction — it destroys the very pairing the
artifact lives in, so it cannot see the artifact at all.  The declaration
therefore lists ``session_shuffle`` and ``row_shuffle`` as forbidden nulls, and
this module refuses to implement them.

What is preserved, exactly
--------------------------

Per :data:`family.SURROGATE_PRESERVES`: the session's timestamps and bar count,
its per-bar volume, the magnitude of the overnight gap, the magnitude of each
bar's close-to-close change, and each bar's high-low range magnitude.  Only two
things are randomized — the sign of the overnight gap and the sign of each
close-to-close change — and from those the complete price path is rebuilt.

Everything downstream is then re-derived from the rebuilt path by the identical
feature code that runs on real bars.  A surrogate that reshuffled a finished
feature would not test what row 183 says must be tested.

This module reads no outcome and computes no economics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v5.research.direction import family, loader


class SurrogateError(RuntimeError):
    """A surrogate cannot be built as the frozen declaration requires."""


@dataclass(frozen=True)
class SurrogateAudit:
    """What the rebuild had to compromise, so it cannot hide.

    ``range_violations`` counts bars where the rebuilt body came out wider than
    the real high-low range, which makes preserving that range exactly
    impossible without an invalid bar.  It is reported rather than silently
    absorbed; on the owned corpus it is expected to be rare.
    """

    sessions: int
    bars: int
    range_violations: int

    @property
    def range_violation_rate(self) -> float:
        return 0.0 if self.bars == 0 else self.range_violations / self.bars


def _rebuild_session(
    bars: loader.SessionBars,
    real_prior_close: float | None,
    surrogate_prior_close: float | None,
    rng: np.random.Generator,
) -> tuple[loader.SessionBars, int]:
    """Rebuild one session's price path from preserved magnitudes.

    Two prior closes, deliberately. The *magnitude* of the overnight gap is a
    property of the real data and must be measured against the real previous
    close; the rebuilt path then hangs that magnitude off the *surrogate's*
    previous close so the surrogate corpus is internally consistent. Measuring
    the magnitude against the surrogate close instead silently fabricates gap
    sizes the market never produced.
    """

    close = bars.close
    open_ = bars.open
    count = close.size
    if count == 0:
        raise SurrogateError(f"{bars.session}: cannot rebuild an empty session")

    # --- the two randomized quantities ---------------------------------
    # sign(0) must stay 0: a zero move has no sign to flip, and inventing one
    # would add motion the real session did not have.
    delta_close = np.diff(close, prepend=close[0])
    delta_close[0] = 0.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    new_close = np.empty(count, dtype=float)

    if real_prior_close is None or surrogate_prior_close is None:
        # The first owned session has no prior close, so it has no gap to
        # randomize. Its opening level is carried, not invented.
        new_open0 = float(open_[0])
    else:
        gap_magnitude = abs(float(open_[0]) - float(real_prior_close))
        gap_sign = float(rng.choice(np.array([-1.0, 1.0]))) if gap_magnitude else 0.0
        new_open0 = float(surrogate_prior_close) + gap_sign * gap_magnitude

    # Bar 0's own move, then a sign-randomized walk on the preserved
    # close-to-close magnitudes.
    intrabar_0 = abs(float(close[0]) - float(open_[0]))
    new_close[0] = new_open0 + signs[0] * intrabar_0
    for i in range(1, count):
        step = abs(float(delta_close[i]))
        new_close[i] = new_close[i - 1] + (signs[i] * step if step else 0.0)

    # --- opens, from the preserved inter-bar move -----------------------
    new_open = np.empty(count, dtype=float)
    new_open[0] = new_open0
    inter_bar = np.abs(open_[1:] - close[:-1])
    inter_signs = rng.choice(np.array([-1.0, 1.0]), size=max(count - 1, 0))
    for i in range(1, count):
        step = float(inter_bar[i - 1])
        new_open[i] = new_close[i - 1] + (inter_signs[i - 1] * step if step else 0.0)

    # --- highs and lows, preserving each bar's range magnitude ----------
    real_range = bars.high - bars.low
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    above = bars.high - body_top
    below = body_bottom - bars.low

    new_top = np.maximum(new_open, new_close)
    new_bottom = np.minimum(new_open, new_close)
    slack = real_range - (new_top - new_bottom)
    violations = int(np.sum(slack < 0.0))

    total_excess = above + below
    share_above = np.divide(
        above, total_excess, out=np.full(count, 0.5), where=total_excess > 0
    )
    usable = np.maximum(slack, 0.0)
    new_high = new_top + usable * share_above
    new_low = new_high - np.where(slack >= 0.0, real_range, new_top - new_bottom)

    return (
        loader.SessionBars(
            session=bars.session,
            instrument_id=bars.instrument_id,
            minute_et=bars.minute_et,
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
            volume=bars.volume,
        ),
        violations,
    )


def matched_surrogate(
    sessions: tuple[loader.SessionBars, ...], *, seed: int
) -> tuple[tuple[loader.SessionBars, ...], SurrogateAudit]:
    """One complete surrogate corpus, matched session by session.

    The chronological order and the session calendar are untouched, so every
    downstream selection, abstention, fold assignment and target is rebuilt on
    the same index the real screen uses. Only the price path differs.
    """

    if not sessions:
        raise SurrogateError("cannot build a surrogate from no sessions")

    rng = np.random.default_rng(seed)
    rebuilt: list[loader.SessionBars] = []
    violations = 0
    bars = 0
    real_prior: float | None = None
    surrogate_prior: float | None = None
    for original in sessions:
        surrogate_session, session_violations = _rebuild_session(
            original, real_prior, surrogate_prior, rng
        )
        rebuilt.append(surrogate_session)
        violations += session_violations
        bars += surrogate_session.close.size
        real_prior = float(original.close[-1])
        surrogate_prior = float(surrogate_session.close[-1])

    return tuple(rebuilt), SurrogateAudit(
        sessions=len(rebuilt), bars=bars, range_violations=violations
    )


def assert_null_is_permitted(name: str) -> None:
    """Refuse a null the frozen declaration forbids.

    Named rather than assumed, because the forbidden nulls are the ones that
    look most reasonable: a session shuffle returns a clean, publishable null
    on exactly the artifact this campaign exists to detect.
    """

    if name in family.FORBIDDEN_NULLS:
        raise SurrogateError(
            f"{name} is a forbidden null (ledger row 183): it returns a clean "
            "result while a shared-term artifact goes unmeasured, because it "
            "destroys the pairing the artifact lives in. Use matched_surrogate."
        )
