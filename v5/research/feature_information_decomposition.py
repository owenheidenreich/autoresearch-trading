"""Phase 4b: split the Phase 4a lift into resolution and ordering.

**The question this settles.** Member P counts a win only when the option gains
+50% before losing -30% within 60 minutes, and a path that touches neither line
counts as a loss. So the hit rate factors:

    P(win) = P(the path resolves) x P(the gain line comes first | it resolved)

Only the second factor is the design's cross-sectional ordering hypothesis. The
first is magnitude -- picking minutes and contracts that move further in
percentage terms raises resolution mechanically, with no opinion about which
contract beats its neighbours. Ledger row 332 measured magnitude as already
priced. Phase 4a could not tell the two apart, so it was referred; this
decomposes them and calibrates the ordering half against its own null.

**Why a new module rather than an extension.**
`feature_information_preflight.py` is pinned by the Phase-4a declaration's
implementation hashes, and that declaration's result is already published.
Editing it would invalidate a declaration after the fact. So the Phase-4a probe
is imported and used unchanged, and the one behaviour Phase 4b needs that it
does not have -- an optional seeded tie-break -- lives here.

**The tie-break, and why it is opt-in.** Tape, clock and chain-state are
minute-common: all 16 of those features take one value across every contract in
a minute. A probe restricted to them scores a minute's contracts identically and
cannot rank them, so selection falls to the stable sort's tie-break on stored row
order, which is ascending `contract_id` -- the deepest-OTM puts. That contaminated
Phase 4a's per-family "alone" numbers. D3 randomises it. It cannot become the
default, because D1's precondition requires the Phase-4a headline to reproduce
bit-for-bit, which requires the deterministic tie-break.

**Label-side only.** This module reads the four first-touch columns and no dollar
column. Economics remain closed at this phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v5.research.feature_information_preflight import (
    PROBE_FEATURES,
    SELECTIONS_PER_SESSION,
    PreflightError,
    _fit_logistic,
    _score,
    _standardise,
    _top_per_session,
    cluster_bootstrap_lift,
)

SCHEMA_VERSION = "v5.feature-information-decomposition.v1"

PRIMARY_LABEL = "first_touch_50pct_before_loss_30pct_60m"
GAIN_MINUTE_COLUMN = "first_touch_50pct_minute_60m"
LOSS_MINUTE_COLUMN = "first_touch_loss_30pct_minute_50pct_60m"

#: The label-side columns permuted together as a block by the null. They are one
#: record about one action; permuting the binary label alone while leaving the
#: touch minutes in place would manufacture rows that are `resolved` but carry
#: another action's outcome, and the decomposition reads both.
LABEL_SIDE_COLUMNS = (PRIMARY_LABEL, GAIN_MINUTE_COLUMN, LOSS_MINUTE_COLUMN)

#: Pre-declared in `PHASE_4B_DECLARATION_V1.json`; restated here so the runner
#: cannot choose them.
PHASE_4A_HEADLINE_LIFT_PP = 8.649074789891253
ORDERING_MATERIAL_BAR_PP = 2.5
NULL_DRAWS = 120
NULL_UPPER_QUANTILE = 0.975
TIE_BREAK_DRAWS = 20


class DecompositionError(RuntimeError):
    """Phase 4b cannot be run as declared."""


# --------------------------------------------------------------------------- #
# Selection with an optional seeded tie-break
# --------------------------------------------------------------------------- #


def select_top(
    sessions: np.ndarray,
    scores: np.ndarray,
    per_session: int = SELECTIONS_PER_SESSION,
    *,
    tie_break_seed: int | None = None,
) -> np.ndarray:
    """The Phase-4a selector, with ties optionally broken at random.

    With `tie_break_seed=None` this delegates to the pinned Phase-4a function, so
    the default path is not merely equivalent but identical.
    """

    if tie_break_seed is None:
        return _top_per_session(sessions, scores, per_session)
    rng = np.random.default_rng(tie_break_seed)
    jitter = rng.permutation(len(scores))
    selected = np.zeros(len(scores), dtype=bool)
    order = np.lexsort((jitter, -scores, sessions))
    ordered_sessions = sessions[order]
    boundaries = np.flatnonzero(
        np.r_[True, ordered_sessions[1:] != ordered_sessions[:-1]]
    )
    for start, stop in zip(boundaries, np.r_[boundaries[1:], len(order)], strict=True):
        selected[order[start : start + min(per_session, stop - start)]] = True
    return selected


def out_of_fold_scores(
    frame: pd.DataFrame,
    folds: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    *,
    label_column: str,
    seed: int,
    features: Sequence[str] = PROBE_FEATURES,
) -> np.ndarray:
    """Pooled out-of-fold scores under the Phase-4a fit law, NaN where unscored."""

    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise DecompositionError(f"probe frame is missing declared features: {missing}")
    values = frame.loc[:, list(features)].to_numpy(float)
    label = frame[label_column].to_numpy(float)
    sessions = frame["session"].to_numpy()
    pooled = np.full(len(frame), np.nan)
    for index, (train_sessions, holdout_sessions) in enumerate(folds):
        train_mask = np.isin(sessions, train_sessions)
        test_mask = np.isin(sessions, holdout_sessions)
        if not train_mask.any() or not test_mask.any():
            continue
        if len(np.unique(label[train_mask])) < 2:
            continue
        x_train, x_test = _standardise(values[train_mask], values[test_mask])
        weights = _fit_logistic(x_train, label[train_mask], seed=seed + index)
        pooled[test_mask] = _score(x_test, weights)
    if not np.isfinite(pooled).any():
        raise DecompositionError("no usable fold produced a score")
    return pooled


# --------------------------------------------------------------------------- #
# D1 — the decomposition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decomposition:
    """One split of a lift into its resolution and ordering halves."""

    base_rate: float
    precision: float
    lift: float
    selected: int
    scored: int
    p_resolved_population: float
    p_resolved_selected: float
    p_gain_first_population: float
    p_gain_first_selected: float
    expected_without_ordering: float
    resolution_component: float
    ordering_component: float

    def payload(self) -> dict[str, Any]:
        return {
            "base_rate": self.base_rate,
            "precision_at_operating_rate": self.precision,
            "lift_pp": self.lift * 100.0,
            "selected": self.selected,
            "scored": self.scored,
            "p_resolved_population": self.p_resolved_population,
            "p_resolved_selected": self.p_resolved_selected,
            "p_gain_first_given_resolved_population": self.p_gain_first_population,
            "p_gain_first_given_resolved_selected": self.p_gain_first_selected,
            "expected_precision_without_ordering": self.expected_without_ordering,
            "resolution_component_pp": self.resolution_component * 100.0,
            "ordering_component_pp": self.ordering_component * 100.0,
        }


def resolved_mask(frame: pd.DataFrame) -> np.ndarray:
    """A path resolved when either barrier was touched inside the horizon."""

    for column in (GAIN_MINUTE_COLUMN, LOSS_MINUTE_COLUMN):
        if column not in frame.columns:
            raise DecompositionError(f"frame is missing the label-side column {column}")
    gain = np.isfinite(pd.to_numeric(frame[GAIN_MINUTE_COLUMN], errors="coerce").to_numpy(float))
    loss = np.isfinite(pd.to_numeric(frame[LOSS_MINUTE_COLUMN], errors="coerce").to_numpy(float))
    return gain | loss


def decompose(frame: pd.DataFrame, selected: np.ndarray, *, label_column: str) -> Decomposition:
    """Split precision into what resolution buys and what ordering adds.

    `expected_without_ordering` is what the selected set would score if it had
    the resolution rate it does and the *population's* gain-first share -- i.e. if
    selection had chosen paths that resolve more often while expressing no view
    about which barrier arrives first. Everything above that line is ordering.
    """

    label = frame[label_column].to_numpy(float)
    resolved = resolved_mask(frame)
    if not selected.any():
        raise DecompositionError("no action was selected")

    base_rate = float(label.mean())
    precision = float(label[selected].mean())
    p_res_pop = float(resolved.mean())
    p_res_sel = float(resolved[selected].mean())
    p_gain_pop = float(label[resolved].mean()) if resolved.any() else float("nan")
    p_gain_sel = (
        float(label[selected & resolved].mean()) if (selected & resolved).any() else float("nan")
    )
    expected = p_res_sel * p_gain_pop
    return Decomposition(
        base_rate=base_rate,
        precision=precision,
        lift=precision - base_rate,
        selected=int(selected.sum()),
        scored=int(len(frame)),
        p_resolved_population=p_res_pop,
        p_resolved_selected=p_res_sel,
        p_gain_first_population=p_gain_pop,
        p_gain_first_selected=p_gain_sel,
        expected_without_ordering=expected,
        resolution_component=expected - base_rate,
        ordering_component=precision - expected,
    )


def permute_label_side(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Permute the label-side record within session, as one block per action."""

    rng = np.random.default_rng(seed)
    out = frame.copy()
    columns = list(LABEL_SIDE_COLUMNS)
    for _, positions in out.groupby("session", sort=False).indices.items():
        shuffled = positions[rng.permutation(len(positions))]
        out.loc[out.index[positions], columns] = (
            frame.loc[frame.index[shuffled], columns].to_numpy()
        )
    return out


# --------------------------------------------------------------------------- #
# D2 — the matched consistency control
# --------------------------------------------------------------------------- #


def matched_precision(
    frame: pd.DataFrame,
    selected: np.ndarray,
    *,
    label_column: str,
    ask_deciles: int = 10,
    vol_quintiles: int = 5,
) -> dict[str, Any]:
    """Selected minus unselected precision inside price/volatility strata.

    Matching within session on entry-ask decile and `realised_vol_15m` quintile
    removes exactly the two channels a resolution-driven selection would exploit:
    leverage and volatility. What survives is comparable to D1's ordering half,
    and must agree with it in direction.
    """

    for column in ("ask", "realised_vol_15m", "session"):
        if column not in frame.columns:
            raise DecompositionError(f"frame is missing the matching column {column}")
    work = frame.loc[:, ["session", "ask", "realised_vol_15m", label_column]].copy()
    work["selected"] = selected

    def _bin(values: pd.Series, bins: int) -> pd.Series:
        try:
            return pd.qcut(values, bins, labels=False, duplicates="drop")
        except ValueError:  # pragma: no cover - degenerate session
            return pd.Series(np.zeros(len(values)), index=values.index)

    work["ask_bin"] = work.groupby("session")["ask"].transform(lambda v: _bin(v, ask_deciles))
    work["vol_bin"] = work.groupby("session")["realised_vol_15m"].transform(
        lambda v: _bin(v, vol_quintiles)
    )

    pairs: list[tuple[float, float]] = []
    for _, group in work.groupby(["session", "ask_bin", "vol_bin"], sort=False, dropna=False):
        chosen = group[group["selected"]]
        others = group[~group["selected"]]
        if chosen.empty or others.empty:
            continue
        pairs.append((float(chosen[label_column].mean()), float(others[label_column].mean())))
    if not pairs:
        raise DecompositionError("no selected action had a match inside its stratum")
    selected_rate = float(np.mean([p[0] for p in pairs]))
    matched_rate = float(np.mean([p[1] for p in pairs]))
    return {
        "strata_with_both": len(pairs),
        "selected_precision": selected_rate,
        "matched_precision": matched_rate,
        "difference_pp": (selected_rate - matched_rate) * 100.0,
        "matched_on": "within session: entry-ask decile x realised_vol_15m quintile",
    }


# --------------------------------------------------------------------------- #
# The pre-declared verdict
# --------------------------------------------------------------------------- #


def apply_verdict(
    observed: Decomposition, null_ordering_pp: Sequence[float]
) -> dict[str, Any]:
    """Fable's three-way rule, applied exactly. No interpretation."""

    values = np.asarray([v for v in null_ordering_pp if np.isfinite(v)], dtype=float)
    if len(values) < NULL_DRAWS:
        raise DecompositionError(
            f"the null needs at least {NULL_DRAWS} usable draws, got {len(values)}"
        )
    threshold = float(np.quantile(values, NULL_UPPER_QUANTILE))
    ordering_pp = observed.ordering_component * 100.0
    if ordering_pp <= threshold:
        verdict = "B_RESOLUTION_MECHANICS"
        meaning = (
            "the ordering component does not exceed its own null: the Phase 4a lift is "
            "resolution mechanics, the magnitude family already priced by ledger row 332. "
            "The fit does not run on this feature set and label as designed."
        )
    elif ordering_pp >= ORDERING_MATERIAL_BAR_PP:
        verdict = "A_ORDERING_INFORMATION"
        meaning = (
            "ordering information exists and is material: the fit proceeds as designed, "
            "with the volatility channel documented as conditioning"
        )
    else:
        verdict = "OWNER_DECISION_REAL_BUT_SMALL"
        meaning = (
            "the ordering component exceeds its null but falls short of the materiality "
            "bar; both components and their nulls go to the owner"
        )
    return {
        "verdict": verdict,
        "meaning": meaning,
        "ordering_component_pp": ordering_pp,
        "resolution_component_pp": observed.resolution_component * 100.0,
        "null_draws": int(len(values)),
        "null_p975_pp": threshold,
        "null_mean_pp": float(values.mean()),
        "null_max_pp": float(values.max()),
        "materiality_bar_pp": ORDERING_MATERIAL_BAR_PP,
        "draws_at_or_above_observed": int((values >= ordering_pp).sum()),
    }
