"""Machine-readable fit gate and architecture roles for the causal day trader.

This module intentionally contains no estimator.  It makes the current
governance conflict fail closed while preserving the architecture comparison
the owner wants if the evidence rules are later reopened legitimately.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from v5.research import knobs


class PolicyFitBlocked(RuntimeError):
    """A proposed fit violates at least one current evidence rule."""


class ReopeningInvalid(RuntimeError):
    """A supplied reopening does not match the signed document on disk."""


class SelectorUnattainable(RuntimeError):
    """A declared selector cannot fire inside its own target/output support."""


# The current signed release of this gate. The original ITM-depth reopening is
# spent by corrected V5. This successor is scoped to one compact action-value
# label and one research-law projection; anything outside it stays refused.
#
# The signed bytes are pinned here.  Computing and returning a digest is not by
# itself an integrity check: without an expected value, an edited document
# would simply acquire a new digest and still release the fit.  Any scope change
# therefore needs a new dated document and a new constant, never an edit to
# this one.
REOPENING_DOCUMENT = Path(
    "v5/governance/CAUSAL_DAY_ACTION_VALUE_SCOPE_ACTIVATION_2026_08_14.md"
)
REOPENING_DOCUMENT_SHA256 = (
    "4d45402344dedfdb2a0cb7ac8bc5714dbff8aab3f4aabd7ad42d9d322b7a04af"
)

# What the 2026-08-14 activation actually released, transcribed from its §3.
REOPENED_LABEL = "serial_action_advantage_120m"
REOPENED_HORIZONS = (120,)
REOPENED_CORPUS = "causal_day_quote_243"
REOPENED_RESEARCH_LAW_SHA256 = (
    "3a7e26e78ad8c6f1c972ba779fd7a6ac1b640e95979dda49ce74891248499627"
)
# The 2026-08-14 re-ruling charges the 20:1 ratio against MEASURED effective
# observations, not raw decision states. Section 4 originally used the 93,798
# raw states; the effective-sample-size measurement showed those are worth
# 2,879-7,557 independent observations by integrated autocorrelation and
# 590-1,016 by design effect. The generous figure is used here, for the sparsest
# scoped label, so the budget cannot be accused of resting on the harsher
# method — and the conservative figure is recorded as a stated limitation.
#
# 7,557 / 20 = 377 trainable parameters.
MEASURED_EFFECTIVE_OBSERVATIONS = 7_557
CONSERVATIVE_EFFECTIVE_OBSERVATIONS = 1_016
# The owner suspended §4 of the reopening on 2026-08-14 after every parameter
# count in it was found to be wrong by roughly 2.4x. While suspended the gate
# refuses every architecture through every route.
#
# This is deliberately not driven by the document's presence on disk: deleting a
# file must not be a way to start fitting. The suspension lifts only when a
# signed re-ruling is named here, which is a reviewable code change.
SUSPENSION_DOCUMENT = Path("v5/governance/CAUSAL_DAY_FIT_SUSPENSION_2026_08_14.md")
SUSPENSION_LIFTED_BY: str | None = (
    "v5/governance/CAUSAL_DAY_FIT_RERULING_2026_08_14.md"
)

REQUIRED_KILL_CONDITIONS = (
    "mid_to_mid_gross_positive",
    "beats_composition_matched_control",
    "beats_shuffled_label_null",
    "per_feature_timestamp_audit",
    "no_post_entry_slot_filter",
    "chronological_out_of_sample",
    "no_reserved_sessions",
)


@dataclass(frozen=True)
class Reopening:
    """A scoped, signed release of the general fit blockers.

    This is deliberately not a boolean. A reopening names the label, horizons
    and corpus it covers and carries the kill conditions the run has committed
    to, so the gate can refuse a fit that drifts outside what was signed.
    """

    label: str
    horizons: tuple[int, ...]
    corpus: str
    kill_conditions: tuple[str, ...]
    document_sha256: str
    research_law_sha256: str | None = None

    def covers(self, *, label: str, horizon: int, corpus: str) -> bool:
        return (
            self.label == label
            and horizon in self.horizons
            and self.corpus == corpus
        )


def load_reopening(document: Path | None = None) -> Reopening:
    """Read the signed ruling from disk and return its declared scope.

    Refuses if the document is missing or its bytes have changed since signing.
    """

    path = document or REOPENING_DOCUMENT
    if not path.exists():
        raise ReopeningInvalid(f"no signed reopening at {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REOPENING_DOCUMENT_SHA256:
        raise ReopeningInvalid(
            f"signed reopening digest mismatch at {path}: "
            f"{digest} != {REOPENING_DOCUMENT_SHA256}"
        )
    return Reopening(
        label=REOPENED_LABEL,
        horizons=REOPENED_HORIZONS,
        corpus=REOPENED_CORPUS,
        kill_conditions=REQUIRED_KILL_CONDITIONS,
        document_sha256=digest,
        research_law_sha256=REOPENED_RESEARCH_LAW_SHA256,
    )


@dataclass(frozen=True)
class Architecture:
    name: str
    sequence: bool
    shared_encoder: bool
    heads: tuple[str, ...]
    conditional_followup: bool = False


ROLES = (
    "morning_entry",
    "morning_exit",
    "afternoon_entry",
    "afternoon_exit",
)

ARCHITECTURES = {
    "shallow_joint": Architecture("shallow_joint", False, True, ("joint",)),
    "shallow_four_head": Architecture("shallow_four_head", False, True, ROLES),
    "neural_joint": Architecture("neural_joint", True, True, ("joint",)),
    "neural_four_head": Architecture("neural_four_head", True, True, ROLES),
    "four_independent": Architecture(
        "four_independent", True, False, ROLES, conditional_followup=True
    ),
    "compact_interaction_entry": Architecture(
        "compact_interaction_entry", False, True, ("morning_entry", "afternoon_entry")
    ),
}


@dataclass(frozen=True)
class SelectorAttainability:
    """Machine-checkable support/witness contract for a selection rule.

    ``None`` bounds mean the target or prediction is not clipped. Absolute
    rules are refused at or beyond a clipped target/output ceiling. Relative
    action rules must carry a concrete built-model witness that strictly
    satisfies the same comparison used at inference.
    """

    selector_name: str
    rule_kind: str
    target_clip_bounds: tuple[float, float] | None = None
    prediction_clip_bounds: tuple[float, float] | None = None
    absolute_threshold: float | None = None
    no_trade_floor: float | None = None
    witness_enter: float | None = None
    witness_wait: float | None = None
    proof_source: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _valid_bounds(
    name: str, bounds: tuple[float, float] | None
) -> tuple[float, float] | None:
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise SelectorUnattainable(f"{name} must contain lower and upper bounds")
    lower, upper = map(float, bounds)
    if not (lower < upper):
        raise SelectorUnattainable(f"{name} must be strictly increasing")
    return lower, upper


def assert_selector_attainable(proof: SelectorAttainability) -> None:
    """Refuse support-boundary selectors before a fit or outcome can be read."""

    if not proof.selector_name or not proof.proof_source:
        raise SelectorUnattainable("selector attainability needs a named rule and proof source")
    target_bounds = _valid_bounds("target_clip_bounds", proof.target_clip_bounds)
    prediction_bounds = _valid_bounds(
        "prediction_clip_bounds", proof.prediction_clip_bounds
    )
    if proof.rule_kind == "absolute_threshold":
        if proof.absolute_threshold is None:
            raise SelectorUnattainable("absolute selector lacks its threshold")
        threshold = float(proof.absolute_threshold)
        if target_bounds is not None and threshold >= target_bounds[1]:
            raise SelectorUnattainable(
                "absolute selector threshold must be strictly below the clipped "
                f"target ceiling: {threshold:g} >= {target_bounds[1]:g}"
            )
        if prediction_bounds is not None and threshold >= prediction_bounds[1]:
            raise SelectorUnattainable(
                "absolute selector threshold must be strictly below the clipped "
                f"prediction ceiling: {threshold:g} >= {prediction_bounds[1]:g}"
            )
        return
    if proof.rule_kind == "relative_action_value":
        if proof.absolute_threshold is not None:
            raise SelectorUnattainable("relative action selector may not carry an absolute threshold")
        values = (proof.no_trade_floor, proof.witness_enter, proof.witness_wait)
        if any(value is None for value in values):
            raise SelectorUnattainable(
                "relative action selector needs a no-trade floor and built-model witness"
            )
        floor, enter, wait = map(float, values)
        if not all(value == value and abs(value) != float("inf") for value in (floor, enter, wait)):
            raise SelectorUnattainable("relative action witness must be finite")
        if prediction_bounds is not None and not all(
            prediction_bounds[0] <= value <= prediction_bounds[1]
            for value in (enter, wait)
        ):
            raise SelectorUnattainable("relative action witness lies outside prediction support")
        if enter <= max(floor, wait):
            raise SelectorUnattainable(
                "relative action witness cannot fire: "
                f"enter={enter:g} <= feasible_wait={max(floor, wait):g}"
            )
        return
    raise SelectorUnattainable(f"unknown selector rule kind: {proof.rule_kind!r}")


def route_head(minute: str, origin_regime: str | None) -> str:
    """Same four-state router as the simulator, without importing runtime code."""

    if origin_regime is not None:
        if origin_regime not in ("morning", "afternoon"):
            raise ValueError(f"unknown origin regime: {origin_regime}")
        return f"{origin_regime}_exit"
    if "09:30" <= minute < "12:46":
        return "morning_entry"
    if "12:46" <= minute <= "16:00":
        return "afternoon_entry"
    raise ValueError(f"minute outside declared router: {minute}")


def fit_blockers(
    architecture: str,
    *,
    sessions: int,
    trainable_parameters: int = 0,
    g1_passed: bool = False,
    do_not_retest_reopened: bool = False,
    shared_specialization_justified: bool = False,
    reopening: Reopening | None = None,
    label: str | None = None,
    horizon: int | None = None,
    corpus: str | None = None,
    declared_kill_conditions: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Why this fit may not run, as a tuple of reasons. Empty means permitted.

    Two routes exist. The boolean route is the historical one and is kept so
    the pre-reopening record still reproduces. The ``reopening`` route is the
    only legitimate way to actually run: it carries the signed ruling's scope,
    so a fit that drifts to a different label, horizon or corpus is refused
    even though a release exists.
    """

    if architecture not in ARCHITECTURES:
        return (f"architecture {architecture!r} is not in the hashed family",)
    spec = ARCHITECTURES[architecture]
    blocked: list[str] = []

    # The suspension is checked here rather than in load_reopening, so a
    # hand-constructed Reopening cannot route around it.
    if SUSPENSION_LIFTED_BY is None:
        blocked.append(
            f"{SUSPENSION_DOCUMENT}: section 4 of the reopening is suspended pending a "
            "signed re-ruling; every architecture is refused"
        )

    # A declared parameter count is only worth as much as the model it was
    # measured from. Section 4 was signed against transcribed counts that were
    # wrong by roughly 2.4x on every row, so the gate now checks the claim.
    if trainable_parameters > 0:
        # Imported lazily: causal_day_architectures imports ROLES from this
        # module, so a module-level import here is circular.
        from v5.research import causal_day_architectures as architectures

        try:
            computed = architectures.computed_parameter_counts()[architecture]
        except Exception as error:  # pragma: no cover - defensive
            blocked.append(f"could not compute a parameter count to verify against: {error}")
        else:
            if trainable_parameters != computed:
                blocked.append(
                    f"declared parameter count {trainable_parameters} does not match the "
                    f"{computed} parameters {architecture!r} actually builds at the declared "
                    "tensorizer dimensions"
                )

    per_parameter = int(knobs.frozen_value("minimum_sessions_per_neural_parameter"))
    minimum_sessions = int(knobs.frozen_value("minimum_sessions_for_neural_comparison"))
    # What the ratio is charged against, and what to call it in a refusal.
    ratio_sample, ratio_unit = sessions, "sessions"

    if reopening is not None:
        # A reopening releases the two general blockers, but only inside the
        # scope it was signed for, and only if the run has committed to every
        # kill condition the ruling requires.
        if label is None or horizon is None or corpus is None:
            blocked.append(
                "a reopening requires the run to declare its label, horizon and corpus"
            )
        elif not reopening.covers(label=label, horizon=horizon, corpus=corpus):
            blocked.append(
                f"outside the signed reopening: label={label!r} horizon={horizon!r} "
                f"corpus={corpus!r} is not covered by {reopening.label!r} "
                f"{reopening.horizons} on {reopening.corpus!r}"
            )
        else:
            g1_passed = True
            do_not_retest_reopened = True
        missing = [
            name
            for name in reopening.kill_conditions
            if name not in declared_kill_conditions
        ]
        if missing:
            blocked.append(
                "the reopening's pre-committed kill conditions are not declared: "
                + ", ".join(missing)
            )
        # The ratio of 20 is kept and charged against MEASURED effective
        # observations rather than raw states or sessions. The knob is untouched.
        ratio_sample = MEASURED_EFFECTIVE_OBSERVATIONS
        ratio_unit = "measured effective observations"
        minimum_sessions = 0

    if not g1_passed:
        blocked.append("STATUS.md: G1 is UNDERPOWERED, not passed; option-model fitting is prohibited")
    if not do_not_retest_reopened:
        blocked.append(
            "DO_NOT_RETEST.md: further selective long-side entry models on this quote corpus are closed"
        )
    # Under a reopening the evidence budget binds EVERY architecture, not only
    # the sequence ones. The measurement that set it is a property of the label
    # and the corpus, so a shallow model spends the same evidence per parameter
    # that a sequence model does.
    if reopening is not None and trainable_parameters > 0:
        if ratio_sample < trainable_parameters * per_parameter:
            blocked.append(
                f"evidence budget: {architecture!r} needs "
                f"{trainable_parameters * per_parameter:,} {ratio_unit} for "
                f"{trainable_parameters:,} parameters, and only {ratio_sample:,} were "
                f"measured (budget {ratio_sample // per_parameter:,} parameters)"
            )

    if spec.sequence:
        if sessions < minimum_sessions:
            blocked.append(
                f"neural session floor: {sessions} available < {minimum_sessions} required"
            )
        if trainable_parameters <= 0:
            blocked.append("neural trainable parameter count must be declared before fitting")
        elif reopening is None and ratio_sample < trainable_parameters * per_parameter:
            # The pre-reopening route, kept so the historical record reproduces.
            blocked.append(
                f"neural parameter ratio: {ratio_sample} {ratio_unit} < "
                f"{trainable_parameters * per_parameter} required for {trainable_parameters} parameters"
            )
    if spec.conditional_followup and not shared_specialization_justified:
        blocked.append(
            "four independent specialists are conditional on prior out-of-sample shared-head specialization"
        )
    return tuple(blocked)


def assert_fit_permitted(
    architecture: str,
    *,
    selector_attainability: SelectorAttainability | None = None,
    **kwargs: object,
) -> None:
    if selector_attainability is not None:
        assert_selector_attainable(selector_attainability)
    blocked = fit_blockers(architecture, **kwargs)
    if blocked:
        raise PolicyFitBlocked("fit refused:\n  " + "\n  ".join(blocked))
