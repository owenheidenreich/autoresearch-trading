"""The known-answer campaign: does the gate actually work, before it is trusted?

Three questions, all answered on synthetic data with a known truth, and all
answered **before** any real economic outcome is inspected:

1. On a matched surrogate with no predictability by construction, how often does
   the family falsely pass?  Declared limit
   :data:`family.SURROGATE_MAX_FALSE_PASS_RATE`, with a Wilson upper bound below
   :data:`family.SURROGATE_MAX_WILSON_UPPER` so a lucky campaign cannot claim
   control it has not demonstrated.
2. On the bounded shared-term fixture — row 183's shape — how often does it pass?
   Declared limit :data:`family.SHARED_TERM_FIXTURE_MAX_PASS_RATE`.
3. When a real edge of known size is injected, how often is it recovered?
   Declared floor :data:`family.INJECTED_EFFECT_MIN_RECOVERY`.

A gate that fails (1) or (2) reports edges that are not there.  A gate that
fails (3) is merely blind, which is safer but no more useful.  Both are
disqualifying, and the campaign is what turns "we believe the gate works" into
a measured claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Callable

import numpy as np

from v5.research.direction import family, fixtures, gate, loader, surrogate


class CampaignError(RuntimeError):
    """The campaign cannot be run as the frozen declaration requires."""


def wilson_upper(successes: int, trials: int, *, confidence: float = 0.95) -> float:
    """Upper bound on a rate, so a small sample cannot overstate control."""

    if trials <= 0:
        raise CampaignError("cannot bound a rate with no trials")
    # 95% one-sided normal quantile; kept explicit rather than importing scipy
    # for a single constant.
    z = 1.6448536269514722 if confidence == 0.95 else abs(
        math.sqrt(2.0) * _erfinv(2.0 * confidence - 1.0)
    )
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float((centre + margin) / denominator)


def _erfinv(x: float) -> float:
    # Winitzki's approximation; only used for non-default confidences.
    a = 0.147
    ln = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(first * first - ln / a) - first), x)


@dataclass(frozen=True)
class CampaignResult:
    name: str
    trials: int
    passes: int
    rate: float
    wilson_upper: float
    limit: float
    is_floor: bool

    @property
    def satisfied(self) -> bool:
        """Whether the declared criterion held.

        A ceiling criterion must also satisfy its Wilson bound; a floor
        criterion (recovery) is judged on the point estimate against its
        declared minimum.
        """

        if self.is_floor:
            return self.rate >= self.limit
        return self.rate <= self.limit and self.wilson_upper <= float(
            family.SURROGATE_MAX_WILSON_UPPER
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "satisfied": self.satisfied}


def _run(
    name: str,
    build: Callable[[int], tuple[loader.SessionBars, ...]],
    *,
    trials: int,
    limit: float,
    is_floor: bool,
    seed_base: int,
    member_prefix: str | None = None,
) -> CampaignResult:
    passes = 0
    for trial in range(trials):
        sessions = build(seed_base + trial)
        features = loader.session_features(sessions)
        results = gate.judge_family(features, seed_offset=trial)
        if member_prefix is None:
            passes += gate.any_member_passed(results)
        else:
            passes += any(
                r.passed and r.member.startswith(member_prefix) for r in results
            )
    rate = passes / trials
    return CampaignResult(
        name=name,
        trials=trials,
        passes=passes,
        rate=rate,
        wilson_upper=wilson_upper(passes, trials),
        limit=limit,
        is_floor=is_floor,
    )


def false_pass_campaign(
    sessions: tuple[loader.SessionBars, ...], *, trials: int, seed_base: int = 10_000
) -> CampaignResult:
    """Question 1: matched surrogates with no predictability by construction."""

    return _run(
        "matched_surrogate_false_pass",
        lambda seed: surrogate.matched_surrogate(sessions, seed=seed)[0],
        trials=trials,
        limit=float(family.SURROGATE_MAX_FALSE_PASS_RATE),
        is_floor=False,
        seed_base=seed_base,
    )


def shared_term_campaign(
    sessions: tuple[loader.SessionBars, ...], *, trials: int, seed_base: int = 20_000
) -> CampaignResult:
    """Question 2: row 183's shared-term shape."""

    return _run(
        "shared_term_fixture",
        lambda seed: fixtures.shared_term_sessions(sessions, seed=seed),
        trials=trials,
        limit=float(family.SHARED_TERM_FIXTURE_MAX_PASS_RATE),
        is_floor=False,
        seed_base=seed_base,
    )


def recovery_campaign(
    sessions: tuple[loader.SessionBars, ...],
    *,
    trials: int,
    points_per_session: float,
    mechanism: str = "M3",
    seed_base: int = 30_000,
) -> CampaignResult:
    """Question 3: a real edge of known size, injected causally."""

    return _run(
        f"injected_{mechanism}_{points_per_session:g}pts",
        lambda seed: fixtures.injected_surrogate(
            sessions, seed=seed, points_per_session=points_per_session,
            mechanism=mechanism,
        ),
        trials=trials,
        limit=float(family.INJECTED_EFFECT_MIN_RECOVERY),
        is_floor=True,
        seed_base=seed_base,
        member_prefix=f"{mechanism}.with",
    )


def operational_detection_floor(
    sessions: tuple[loader.SessionBars, ...],
    *,
    mechanism: str,
    trials: int,
    grid: tuple[float, ...],
    seed_base: int = 40_000,
) -> tuple[dict[float, float], float | None]:
    """The smallest injected edge the gate actually recovers at the declared floor.

    This is the number that matters when recovery fails: an analytic minimum
    detectable effect assumes a single z-test on the mean, while this gate is a
    conjunction of six criteria judged with a block bootstrap. The conjunction
    costs real power, and quoting the analytic figure would overstate what the
    screen can see.
    """

    curve: dict[float, float] = {}
    achieved: float | None = None
    for points in grid:
        result = recovery_campaign(
            sessions, trials=trials, points_per_session=points,
            mechanism=mechanism, seed_base=seed_base,
        )
        curve[points] = result.rate
        if achieved is None and result.rate >= float(
            family.INJECTED_EFFECT_MIN_RECOVERY
        ):
            achieved = points
    return curve, achieved


def surrogate_dispersion(
    sessions: tuple[loader.SessionBars, ...], *, member: family.Member, trials: int = 12
) -> tuple[float, float]:
    """Per-session net dispersion and occupancy, measured on surrogates only.

    Sizing an injected effect from *real* dispersion would touch the real
    outcome before the null gate has passed, which the frozen order of work
    forbids. Surrogates preserve the volatility structure, so they are the
    honest place to take this number.
    """

    from v5.research.direction import replay

    dispersions: list[float] = []
    occupancies: list[float] = []
    for trial in range(trials):
        fake, _ = surrogate.matched_surrogate(sessions, seed=50_000 + trial)
        frame = replay.replay_member(loader.session_features(fake), member)
        dispersions.append(float(frame["net_points"].std(ddof=1)))
        occupancies.append(float(frame["trades"].mean()))
    return float(np.mean(dispersions)), float(np.mean(occupancies))
