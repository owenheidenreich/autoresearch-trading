"""The small statistical facts this project keeps needing, in one place.

Two of these were previously duplicated: a normal quantile lived inline in
:mod:`v5.research.direction.campaign` and again in the job-15 feasibility
module, and the second copy sat inside a work packet that the file policy will
move to ``v5/history/jobs/`` when the job closes. Durable machinery must not
live somewhere scheduled for archival, so it lives here.

Nothing in this module reads data, fits anything, or computes a policy. It is
arithmetic about how much evidence a given amount of data can carry.
"""
from __future__ import annotations

import math


# One-sided normal quantiles, kept explicit rather than importing scipy for two
# constants. Z_95 is pinned to the literal already used by
# campaign.wilson_upper so the two cannot drift apart.
Z_95 = 1.6448536269514722
Z_POWER_80 = 0.8416212335729143

TRADING_DAYS_PER_YEAR = 252


class StatisticsError(ValueError):
    """Raised for a domain error rather than returning a silent nan."""


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, via Acklam's rational approximation.

    Accurate to roughly 1e-9 across the range this project uses.
    """

    if not 0.0 < p < 1.0:
        raise StatisticsError(f"normal_quantile needs 0 < p < 1, got {p!r}")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        return -normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def bonferroni_quantile(family_size: int, *, level: float = 0.95) -> float:
    """One-sided quantile for one member of a declared family of ``family_size``.

    Judging each member at the raw level and then reporting "any member passed"
    prices eighteen attempts as one; the 2026-08-09 pilot measured 11.7%
    familywise false-pass that way against a declared 5.0% limit.
    """

    if family_size < 1:
        raise StatisticsError("family_size must be at least 1")
    if not 0.0 < level < 1.0:
        raise StatisticsError(f"level must be in (0, 1), got {level!r}")
    return normal_quantile(1.0 - (1.0 - level) / family_size)


def detectable_sharpe(
    sessions: int, *, z_alpha: float, penalty: float = 1.0, power_z: float = Z_POWER_80
) -> float:
    """The smallest annualised Sharpe ratio a screen on ``sessions`` days can see.

    This is the central result of the 2026-08 measurement work, and the
    derivation is kept here because the cancellation is not obvious.

    A screen can just detect a marginally-profitable strategy when its minimum
    detectable per-trade effect equals the friction bar::

        MDE_per_trade = (z_alpha + z_power) * sigma_h * penalty / sqrt(n * k)

    where ``sigma_h`` is the standard deviation of an ``h``-minute move and
    ``k = M / h`` is the number of serial non-overlapping trades that fit in
    ``M`` tradeable minutes. Express that requirement as an annualised Sharpe,
    ``S = (effect / sigma_h) * sqrt(k * 252)``, and substitute::

        S = (z_alpha + z_power) * penalty * sqrt(252 / n)

    **The move dispersion, the horizon, the trades per session, the tradeable
    minutes and the friction bar all cancel.** What a screen can see depends on
    the session count and the statistical standard, and on nothing else.

    This is why job 15 found that higher occupancy could not rescue G1, and why
    a different instrument cannot either: both move a screen *along* this curve
    rather than moving the curve. ``n`` is the only lever.

    Verified against two independent results: it reproduces the job-15 surface's
    5/30/60-minute cells to within 1%, and it reproduces G1's separately
    measured floor of 8 net points per session.
    """

    if sessions <= 0:
        raise StatisticsError("sessions must be positive")
    if penalty <= 0:
        raise StatisticsError("penalty must be positive")
    return (z_alpha + power_z) * penalty * math.sqrt(TRADING_DAYS_PER_YEAR / sessions)


def sessions_for_sharpe(
    target: float, *, z_alpha: float, penalty: float = 1.0, power_z: float = Z_POWER_80
) -> int:
    """Inverse of :func:`detectable_sharpe`: sessions needed to see ``target``.

    This is the purchase question. It converts "the corpus is too small" into a
    number of trading days, and therefore into a price.
    """

    if target <= 0:
        raise StatisticsError("target Sharpe must be positive")
    if penalty <= 0:
        raise StatisticsError("penalty must be positive")
    return math.ceil(
        TRADING_DAYS_PER_YEAR * ((z_alpha + power_z) * penalty / target) ** 2
    )


def breakeven_accuracy(*, win: float, loss: float) -> float:
    """Directional accuracy at which a win/loss payoff pair breaks even."""

    if win <= 0 or loss <= 0:
        raise StatisticsError("win and loss must both be positive magnitudes")
    return loss / (win + loss)


def accuracy_payoff(accuracy: float, *, win: float, loss: float) -> float:
    """Expected payoff per trade at a given directional accuracy."""

    if not 0.0 <= accuracy <= 1.0:
        raise StatisticsError(f"accuracy must be in [0, 1], got {accuracy!r}")
    return accuracy * win - (1.0 - accuracy) * loss


def detectable_accuracy(
    sessions: int, *, win: float, loss: float, z_alpha: float, penalty: float = 1.0
) -> float:
    """Smallest directional accuracy visible above the noise on ``sessions`` days.

    Being profitable and being *measurable* are different bars, and for a 0DTE
    option the gap between them is wide because the per-trade payoff dispersion
    is enormous relative to the edge. Evaluated at break-even, where the
    dispersion is largest and the requirement therefore conservative.
    """

    if sessions <= 0:
        raise StatisticsError("sessions must be positive")
    p = breakeven_accuracy(win=win, loss=loss)
    spread = win + loss
    payoff_sd = math.sqrt(p * (1.0 - p)) * spread
    edge = (z_alpha + Z_POWER_80) * penalty * payoff_sd / math.sqrt(sessions)
    return (edge + loss) / spread


def sessions_for_accuracy_edge(
    target_accuracy: float,
    *,
    win: float,
    loss: float,
    z_alpha: float,
    penalty: float = 1.0,
    trades_per_session: float = 1.0,
) -> int:
    """Inverse of :func:`detectable_accuracy`: sessions needed to prove ``target_accuracy``.

    :func:`sessions_for_sharpe` answers the purchase question for a screen whose
    edge is expressed as a Sharpe ratio. This answers it for a screen whose edge
    is a **hit rate on a two-point payoff** — the shape a sparse option target
    actually has, where a rare large win is paid for by many small losses.

    Inverting the detectable-accuracy relation gives a form in which the payoff
    magnitudes cancel and only the *precision gap* survives::

        n_trades = [ (z_alpha + z_power) * penalty * sqrt(p0 * (1 - p0)) /
                     (target_accuracy - p0) ] ** 2

    with ``p0`` the break-even hit rate ``loss / (win + loss)``. So the cost of
    an answer is set by how far above break-even the policy must be proven to
    sit, not by how large the individual payoffs are. Halving the precision gap
    quadruples the sessions required.

    ``trades_per_session`` converts trades into calendar days, and therefore
    into a data purchase.
    """

    if not 0.0 < target_accuracy < 1.0:
        raise StatisticsError(
            f"target_accuracy must be in (0, 1), got {target_accuracy!r}"
        )
    if penalty <= 0:
        raise StatisticsError("penalty must be positive")
    if trades_per_session <= 0:
        raise StatisticsError("trades_per_session must be positive")
    p0 = breakeven_accuracy(win=win, loss=loss)
    gap = target_accuracy - p0
    if gap <= 0:
        raise StatisticsError(
            f"target_accuracy {target_accuracy!r} is at or below the {p0:.6f} "
            "break-even; there is no edge to prove"
        )
    payoff_sd = math.sqrt(p0 * (1.0 - p0))
    trades = ((z_alpha + Z_POWER_80) * penalty * payoff_sd / gap) ** 2
    return math.ceil(trades / trades_per_session)


def accuracy_to_underlying_points(accuracy: float, *, move_sd: float) -> float:
    """Gross underlying points per trade implied by a directional accuracy.

    Assumes the magnitude of the move is independent of whether the call was
    correct, so ``E[signed] = (2p - 1) * E|move|`` and ``E|move| = sd * sqrt(2/pi)``
    for a symmetric move. A strategy that is right more often on large moves
    would need less accuracy than this returns; one right more often on small
    moves would need more. The assumption must be declared by any screen that
    freezes a threshold from this, never inherited silently.
    """

    if not 0.0 <= accuracy <= 1.0:
        raise StatisticsError(f"accuracy must be in [0, 1], got {accuracy!r}")
    if move_sd <= 0:
        raise StatisticsError("move_sd must be positive")
    return (2.0 * accuracy - 1.0) * move_sd * math.sqrt(2.0 / math.pi)
