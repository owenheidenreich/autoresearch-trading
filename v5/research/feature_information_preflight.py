"""Phase 4a: does the declared feature set carry entry information at all?

**Why this exists, and why it runs before the fit.** Every feature family this
project has ever fitted is measured dead: chart/clock/geometry by the 375-cell
census (ledger row 338), price-and-volume selection by rows 339/340, greeks by
row 336. The one untested family is the option chain's own internals. The full
118-parameter fit is expensive and, worse, *interpretable in too many ways* --
a null result there can always be blamed on optimisation. A small probe cannot
be blamed on optimisation, so a null result from it is a statement about the
features.

**The known-answer twin is the part that makes a negative mean something.** Job
45 closed underpowered because a gate could not certify an edge the models had
demonstrably learned. A preflight that cannot recover its own planted edge proves
nothing about the real features, so the identical probe runs on synthetic data
carrying a planted lift and must find it before the real verdict is read.

**Generous by construction, so a failure is decisive.** The verdict statistic is
the *upper* bound on the lift, and where two upper bounds are available the
verdict takes the larger. If even the optimistic end of the interval cannot clear
half of the best entry effect this project has ever measured, the features are
not carrying what the target needs.

**What is deliberately not here.** No economics are opened: the probe reads
member P's binary label and never a dollar column. No score block is touched --
every fold lives inside the chronological training prefix. Nothing here fits the
declared architecture, tunes a threshold, or promotes anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v5.research.causal_day_chain_state_lifecycle import STATE_CANDLE_FEATURES
from v5.research.chain_internal_features import (
    CHAIN_STATE_FEATURES,
    CONTRACT_CHAIN_FEATURES,
)

SCHEMA_VERSION = "v5.feature-information-preflight.v1"

#: Exactly 25 parameters with the intercept -- the declared ceiling. What is in
#: and what is out, stated so a null result cannot later be explained away:
#:
#: * IN: the seven chain-internal fields (the hypothesis), the four tape channels
#:   the member reads as state, the five clock channels, the six per-contract
#:   fields, and the two ordering inputs the architecture multiplies state
#:   against.
#: * OUT, and why: the two account channels are constant at training time by
#:   declaration, so they can carry nothing; the six ladder-context mean/max
#:   aggregates are minute-common summaries of the same ladder the per-contract
#:   fields already come from, and including them would breach the parameter
#:   ceiling that makes this probe unblameable on optimisation.
CLOCK_FEATURES_NAMED = (
    "clock_elapsed",
    "clock_remaining",
    "clock_is_morning",
    "clock_sin",
    "clock_cos",
)
CONTRACT_BASE_FEATURES = ("ask", "spread", "self_iv", "self_theta_per_minute")
ORDERING_FEATURES = ("is_call", "moneyness_itm_points")

PROBE_FEATURES: tuple[str, ...] = (
    tuple(CHAIN_STATE_FEATURES)
    + tuple(STATE_CANDLE_FEATURES)
    + CLOCK_FEATURES_NAMED
    + CONTRACT_BASE_FEATURES
    + tuple(CONTRACT_CHAIN_FEATURES)
    + ORDERING_FEATURES
)
PROBE_PARAMETERS = len(PROBE_FEATURES) + 1
PARAMETER_CEILING = 25

#: The signed operating rate: roughly two trades a day (charter amendment
#: 2026-08-16 models the breaker at two trades a session).
SELECTIONS_PER_SESSION = 2

#: Pre-declared verdict constants. Stated here so the runner cannot pick them.
LIFT_BAR_PP = 4.0
PLANTED_LIFT_PP = 20.0
PLANT_RECOVERY_BAR_PP = 10.0
BOOTSTRAP_RESAMPLES = 10_000
INNER_FOLDS = 5


class PreflightError(RuntimeError):
    """The preflight cannot be run as declared."""


# --------------------------------------------------------------------------- #
# Interval estimates
# --------------------------------------------------------------------------- #


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963985) -> tuple[float, float]:
    """Two-sided Wilson score interval on a proportion."""

    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = (z / denominator) * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (float(centre - half), float(centre + half))


def cluster_bootstrap_lift(
    sessions: np.ndarray,
    selected: np.ndarray,
    label: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int,
) -> tuple[float, float]:
    """Resample whole sessions, not rows, because the session is the unit.

    **A measured correction to the obvious intuition.** Clustering is usually
    reached for because ignoring it *understates* an interval. That is not what
    happens here, and the reason is worth stating so nobody later "fixes" this
    into a row bootstrap. The statistic is a **within-session paired contrast** --
    precision among the session's own selections, minus that same session's own
    base rate -- so a session's regime largely cancels inside it. Resampling whole
    sessions preserves that pairing; resampling rows breaks it and *adds*
    variance. Measured on session-correlated synthetic outcomes: cluster width
    0.107 against row width 0.139.

    So this is used because it is the correct unit of independence, not because
    it is the conservative one. The verdict rule does not lean on it for
    conservatism either -- it takes the larger of this upper bound and Wilson's,
    precisely so the answer cannot depend on which interval happens to be wider.
    """

    unique = np.unique(sessions)
    if len(unique) < 2:
        return (float("nan"), float("nan"))
    index = {name: np.flatnonzero(sessions == name) for name in unique}
    rng = np.random.default_rng(seed)
    lifts = np.empty(resamples, dtype=float)
    for draw in range(resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[name] for name in picked])
        chosen = selected[rows]
        if not chosen.any():
            lifts[draw] = np.nan
            continue
        lifts[draw] = float(label[rows][chosen].mean() - label[rows].mean())
    lifts = lifts[np.isfinite(lifts)]
    if lifts.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(lifts, 0.025)), float(np.quantile(lifts, 0.975)))


# --------------------------------------------------------------------------- #
# The probe
# --------------------------------------------------------------------------- #


def _standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute and scale from the training fold only. Never from the holdout."""

    fill = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    filled_train = np.where(np.isfinite(train), train, fill)
    filled_other = np.where(np.isfinite(other), other, fill)
    centre = filled_train.mean(axis=0)
    scale = filled_train.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (filled_train - centre) / scale, (filled_other - centre) / scale


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=5000, random_state=seed)
    model.fit(x, y)
    return np.concatenate([model.coef_.ravel(), model.intercept_])


def _score(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return x @ weights[:-1] + weights[-1]


def _top_per_session(sessions: np.ndarray, scores: np.ndarray, per_session: int) -> np.ndarray:
    """The policy's operating rate: the best `per_session` actions of each day."""

    selected = np.zeros(len(scores), dtype=bool)
    order = np.lexsort((-scores, sessions))
    boundaries = np.flatnonzero(np.r_[True, sessions[order][1:] != sessions[order][:-1]])
    for start, stop in zip(boundaries, np.r_[boundaries[1:], len(order)], strict=True):
        selected[order[start : start + min(per_session, stop - start)]] = True
    return selected


@dataclass(frozen=True)
class ProbeResult:
    """One probe run, real or planted."""

    label: str
    rows: int
    sessions: int
    folds: int
    base_rate: float
    precision: float
    lift: float
    selected: int
    wilson_precision: tuple[float, float]
    wilson_lift_upper: float
    bootstrap_lift: tuple[float, float]
    lift_upper: float
    parameters: int
    fold_lifts: tuple[float, ...] = field(default_factory=tuple)

    def payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rows": self.rows,
            "sessions": self.sessions,
            "folds": self.folds,
            "parameters": self.parameters,
            "base_rate": self.base_rate,
            "precision_at_operating_rate": self.precision,
            "selected": self.selected,
            "lift": self.lift,
            "lift_pp": self.lift * 100.0,
            "wilson_precision_interval": list(self.wilson_precision),
            "wilson_lift_upper_pp": self.wilson_lift_upper * 100.0,
            "bootstrap_lift_interval_pp": [v * 100.0 for v in self.bootstrap_lift],
            "verdict_lift_upper_pp": self.lift_upper * 100.0,
            "fold_lifts_pp": [v * 100.0 for v in self.fold_lifts],
        }


def run_probe(
    frame: pd.DataFrame,
    folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    *,
    label_column: str,
    seed: int,
    name: str,
    features: Sequence[str] = PROBE_FEATURES,
) -> ProbeResult:
    """Nested chronological out-of-fold probe at the declared operating rate.

    Every fold trains on strictly earlier sessions than it scores, so no fold
    reads its own future. Scores are pooled across folds and the operating rate
    is applied per session, which is how the policy would actually spend it.
    """

    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise PreflightError(f"probe frame is missing declared features: {missing}")
    if len(features) + 1 > PARAMETER_CEILING:
        raise PreflightError(
            f"probe would use {len(features) + 1} parameters, over the declared "
            f"ceiling of {PARAMETER_CEILING}"
        )

    values = frame.loc[:, list(features)].to_numpy(float)
    label = frame[label_column].to_numpy(float)
    sessions = frame["session"].to_numpy()

    pooled_scores = np.full(len(frame), np.nan)
    fold_lifts: list[float] = []
    used = 0
    for index, (train_sessions, holdout_sessions) in enumerate(folds):
        train_mask = np.isin(sessions, train_sessions)
        test_mask = np.isin(sessions, holdout_sessions)
        if not train_mask.any() or not test_mask.any():
            continue
        if len(np.unique(label[train_mask])) < 2:
            continue
        x_train, x_test = _standardise(values[train_mask], values[test_mask])
        weights = _fit_logistic(x_train, label[train_mask], seed=seed + index)
        scores = _score(x_test, weights)
        pooled_scores[test_mask] = scores
        chosen = _top_per_session(sessions[test_mask], scores, SELECTIONS_PER_SESSION)
        fold_lifts.append(float(label[test_mask][chosen].mean() - label[test_mask].mean()))
        used += 1
    if used == 0:
        raise PreflightError("no usable fold produced a probe score")

    scored = np.isfinite(pooled_scores)
    sessions_scored = sessions[scored]
    labels_scored = label[scored]
    selected = _top_per_session(sessions_scored, pooled_scores[scored], SELECTIONS_PER_SESSION)

    successes = int(labels_scored[selected].sum())
    trials = int(selected.sum())
    base_rate = float(labels_scored.mean())
    precision = float(labels_scored[selected].mean())
    wilson = wilson_interval(successes, trials)
    bootstrap = cluster_bootstrap_lift(
        sessions_scored, selected, labels_scored, seed=seed + 9_000
    )
    # Generous by declaration: the verdict takes the larger of the two upper
    # bounds, so a failure cannot be attributed to the choice of interval.
    wilson_lift_upper = wilson[1] - base_rate
    candidates = [value for value in (wilson_lift_upper, bootstrap[1]) if np.isfinite(value)]
    return ProbeResult(
        label=name,
        rows=int(scored.sum()),
        sessions=int(len(np.unique(sessions_scored))),
        folds=used,
        base_rate=base_rate,
        precision=precision,
        lift=precision - base_rate,
        selected=trials,
        wilson_precision=wilson,
        wilson_lift_upper=wilson_lift_upper,
        bootstrap_lift=bootstrap,
        lift_upper=float(max(candidates)) if candidates else float("nan"),
        parameters=len(features) + 1,
        fold_lifts=tuple(fold_lifts),
    )


# --------------------------------------------------------------------------- #
# The known-answer twin
# --------------------------------------------------------------------------- #


def plant_known_answer(
    frame: pd.DataFrame,
    *,
    label_column: str,
    lift_pp: float = PLANTED_LIFT_PP,
    seed: int,
    features: Sequence[str] = PROBE_FEATURES,
) -> pd.DataFrame:
    """Regenerate the label so a linear function of the real features pays.

    The plant is a *linear* direction through the real feature matrix, at the
    real geometry -- same sessions, same candidates per session, same base rate,
    same two-per-day rate. It establishes that a probe of this size can find an
    edge of the size that matters here. If it cannot, nothing the probe says
    about the real label is worth reading.
    """

    rng = np.random.default_rng(seed)
    values = frame.loc[:, list(features)].to_numpy(float)
    fill = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    values = np.where(np.isfinite(values), values, np.where(np.isfinite(fill), fill, 0.0))
    centre = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    direction = rng.normal(size=values.shape[1])
    signal = ((values - centre) / scale) @ direction

    planted = frame.copy()
    base_rate = float(frame[label_column].mean())
    ranks = (
        pd.Series(signal, index=frame.index)
        .groupby(frame["session"].to_numpy())
        .rank(pct=True)
        .to_numpy()
    )
    # Percentile rank is uniform within a session, so `2 * rank - 1` averages to
    # zero and the planted base rate stays the real one. The top of each session
    # sits at rank ~1, so it carries very nearly the full planted lift.
    probability = np.clip(base_rate + (lift_pp / 100.0) * (2.0 * ranks - 1.0), 0.001, 0.999)
    planted[label_column] = (rng.random(len(frame)) < probability).astype(float)
    return planted


# --------------------------------------------------------------------------- #
# The pre-declared verdict
# --------------------------------------------------------------------------- #


def apply_verdict(real: ProbeResult, planted: ProbeResult) -> dict[str, Any]:
    """The rule exactly as declared before the probe ran. No interpretation."""

    recovered = bool(
        np.isfinite(planted.lift)
        and planted.lift * 100.0 >= PLANT_RECOVERY_BAR_PP
        and np.isfinite(planted.bootstrap_lift[0])
        and planted.bootstrap_lift[0] > 0.0
    )
    upper_pp = real.lift_upper * 100.0
    if not recovered:
        verdict = "ADVISORY_UNDERPOWERED"
        meaning = (
            "the probe did not recover its own planted edge, so it cannot certify "
            "the absence of one; the go/no-go is an owner decision, not an agent's"
        )
    elif upper_pp < LIFT_BAR_PP:
        verdict = "STOP_FEATURES_INSUFFICIENT"
        meaning = (
            "the probe is powered and even the generous upper bound on the real "
            "features' lift falls short of half the best entry effect ever measured "
            "here; the available features cannot close the +14-point gap on this corpus"
        )
    else:
        verdict = "PROCEED"
        meaning = (
            "the probe is powered and the real features clear the declared bar; the "
            "full fit is worth running and per-field information sets the shrink order"
        )
    return {
        "verdict": verdict,
        "meaning": meaning,
        "plant_recovered": recovered,
        "planted_lift_pp": planted.lift * 100.0,
        "plant_recovery_bar_pp": PLANT_RECOVERY_BAR_PP,
        "real_lift_pp": real.lift * 100.0,
        "real_lift_upper_pp": upper_pp,
        "lift_bar_pp": LIFT_BAR_PP,
    }
