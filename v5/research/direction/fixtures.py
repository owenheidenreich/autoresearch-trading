"""Two known-answer fixtures: one the gate must reject, one it must find.

A null that only ever says "no" is not evidence the gate works — it is
indistinguishable from a gate that is broken shut. The campaign therefore needs
both directions:

**The shared-term fixture** reproduces ledger row 183's artifact. Feature and
target are built from the same price level, so a naive statistic sees strong
structure where no predictability exists. The gate must pass it no more often
than :data:`family.SHARED_TERM_FIXTURE_MAX_PASS_RATE`. This is the fixture that
matters most, because it is the specific way this project has already fooled
itself once.

**The injected-effect fixture** adds a real, known directional edge of
minimum-detectable size to a matched surrogate. The gate must recover it at
least :data:`family.INJECTED_EFFECT_MIN_RECOVERY` of the time. Without this a
gate that never passes anything would score perfectly on the null and be
useless.

Neither fixture reads a real outcome.
"""
from __future__ import annotations

import numpy as np

from v5.research.direction import family, loader, surrogate


class FixtureError(RuntimeError):
    """A fixture cannot be constructed as declared."""


def shared_term_sessions(
    sessions: tuple[loader.SessionBars, ...], *, seed: int
) -> tuple[loader.SessionBars, ...]:
    """A corpus where feature and target share a price level but nothing predicts.

    Row 183's mechanism, made explicit: the session's path is a *bounded* random
    walk that reverts toward a slow-moving level. Both the opening-range feature
    and the horizon return are then measured against that same level, so they
    co-move strongly while carrying no information about direction.

    A session shuffle returns a clean null here, which is precisely why the
    declaration forbids it — the artifact lives in the within-session pairing
    that a shuffle destroys.
    """

    if not sessions:
        raise FixtureError("cannot build a shared-term fixture from no sessions")

    rng = np.random.default_rng(seed)
    built: list[loader.SessionBars] = []
    level = float(sessions[0].close[0])

    for original in sessions:
        count = original.close.size
        scale = float(np.abs(np.diff(original.close)).mean() or 1.0)
        # A large, slowly wandering level shared by every quantity measured on
        # the session, plus strictly independent increments. The shared term is
        # therefore real and dominant while the path stays a martingale, which
        # is the row-183 shape: a statistic that co-moves with the level looks
        # strong even though nothing observable predicts direction.
        #
        # Deliberately NOT mean-reverting. An earlier draft pulled the path back
        # toward the level, and that is genuine tradeable predictability rather
        # than an artifact -- the gate passed it 100% of the time, correctly.
        level += float(rng.normal(0.0, scale * 2.0))
        steps = rng.normal(0.0, scale, size=count)
        path = level + np.cumsum(steps)
        opens = np.concatenate(([path[0]], path[:-1]))
        built.append(
            loader.SessionBars(
                session=original.session,
                instrument_id=original.instrument_id,
                minute_et=original.minute_et,
                open=opens,
                high=np.maximum(opens, path) + scale,
                low=np.minimum(opens, path) - scale,
                close=path,
                volume=original.volume,
            )
        )
        level = float(path[-1])
    return tuple(built)


def inject_effect(
    sessions: tuple[loader.SessionBars, ...],
    *,
    points_per_session: float,
    mechanism: str = "M3",
    reference_horizon: int = 60,
) -> tuple[loader.SessionBars, ...]:
    """Add a real, known directional edge to an existing corpus.

    The edge is honest rather than magical: after the decision instant the path
    drifts in the direction the mechanism's score already points. Nothing before
    the entry bar is touched, so the edge is causal — a policy can only capture
    it by getting the direction right.

    The drift is calibrated so a position held for ``reference_horizon`` minutes
    collects exactly ``points_per_session``. An earlier draft spread the effect
    across the whole remaining session instead, which meant a 15-minute horizon
    collected only 15/355 of it — below friction — and made recovery
    non-monotonic in the injected size, which is how the error was caught.
    """

    if points_per_session <= 0.0:
        raise FixtureError("an injected effect must be positive to be recoverable")
    if reference_horizon <= 0:
        raise FixtureError("the reference horizon must be positive")

    entry_index = None
    built: list[loader.SessionBars] = []
    previous_close: float | None = None

    for original in sessions:
        minutes = original.minute_et
        try:
            entry_index = minutes.index(family.ENTRY_BAR_ET)
        except ValueError:
            built.append(original)
            previous_close = float(original.close[-1])
            continue

        if mechanism in {"M3", "JOINT"}:
            # JOINT scores on the same overnight gap as M3; it differs only in
            # when it is willing to trade, not in what it scores.
            score = (
                0.0 if previous_close is None
                else float(original.open[0]) - previous_close
            )
        elif mechanism == "M1":
            first_five = minutes.index(family.FEATURE_BARS_ET[-1])
            score = float(original.close[first_five]) - float(original.open[0])
        else:
            raise FixtureError(f"unknown mechanism for injection: {mechanism}")

        direction = float(np.sign(score))
        close = original.close.copy()
        after = np.arange(close.size) > entry_index
        steps = int(after.sum())
        if steps:
            # Per-minute drift, so holding `reference_horizon` minutes collects
            # exactly `points_per_session` and shorter horizons collect a
            # proportional share -- which is how a real drift behaves.
            per_minute = direction * points_per_session / float(reference_horizon)
            close[after] += per_minute * np.arange(1, steps + 1)

        built.append(
            loader.SessionBars(
                session=original.session,
                instrument_id=original.instrument_id,
                minute_et=minutes,
                open=original.open,
                high=np.maximum(original.high, close),
                low=np.minimum(original.low, close),
                close=close,
                volume=original.volume,
            )
        )
        previous_close = float(close[-1])
    return tuple(built)


def injected_surrogate(
    sessions: tuple[loader.SessionBars, ...],
    *,
    seed: int,
    points_per_session: float,
    mechanism: str = "M3",
) -> tuple[loader.SessionBars, ...]:
    """A matched surrogate carrying a known edge — the gate's positive control."""

    base, _ = surrogate.matched_surrogate(sessions, seed=seed)
    return inject_effect(
        base, points_per_session=points_per_session, mechanism=mechanism
    )
