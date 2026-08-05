"""What a bounded search may vary, what is frozen, and what is not certified.

A hillclimbing loop is only honest if the set of things it may change is
declared before it runs.  This module is that declaration in machine-readable
form, so a search cannot quietly widen itself and a reviewer can see the exact
evidence behind every fixed number.

Three classes:

``FROZEN``
    Measured or contractually fixed.  A search may read it and may never vary
    it.  Each entry names the evidence that fixed it and the exact condition
    that would legitimately unfreeze it.

``UNCERTIFIED``
    A number exists but its evidence does not support the use it would be put
    to.  It may be neither varied nor relied on.  Anything depending on it is
    blocked until the named evidence arrives.  This is deliberately stricter
    than ``FROZEN``: a frozen knob is trustworthy, an uncertified one is not.

``SEARCHABLE``
    May be varied inside a declared range, but only once its releasing gate has
    passed.  Every searchable knob names that gate.

The module is model-free and network-free.  It computes nothing about markets;
it only answers "may this run touch this?".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


class KnobError(RuntimeError):
    """A proposed run would vary something it is not allowed to vary."""


FROZEN = "FROZEN"
UNCERTIFIED = "UNCERTIFIED"
SEARCHABLE = "SEARCHABLE"


@dataclass(frozen=True)
class Knob:
    name: str
    knob_class: str
    summary: str
    evidence: str
    value: Any = None
    choices: tuple[Any, ...] | None = None
    low: float | None = None
    high: float | None = None
    unfreeze_condition: str = ""
    released_by_gate: str = ""

    def __post_init__(self) -> None:
        if self.knob_class not in {FROZEN, UNCERTIFIED, SEARCHABLE}:
            raise KnobError(f"unknown knob class for {self.name}: {self.knob_class}")
        if self.knob_class in {FROZEN, UNCERTIFIED} and not self.unfreeze_condition:
            raise KnobError(f"{self.name} must state what would unfreeze it")
        if self.knob_class == SEARCHABLE:
            if not self.released_by_gate:
                raise KnobError(f"{self.name} must name the gate that releases it")
            if self.choices is None and (self.low is None or self.high is None):
                raise KnobError(f"{self.name} must declare choices or a numeric range")
            if self.low is not None and self.high is not None and self.low >= self.high:
                raise KnobError(f"{self.name} has an empty range")

    def permits(self, value: Any) -> bool:
        """Whether ``value`` is inside this knob's declared search space.

        A collection value is permitted when every element is a declared
        choice — this is how a composition knob such as ``feature_families``
        stays a finite declaration without enumerating every subset.
        """

        if self.knob_class != SEARCHABLE:
            return value == self.value
        if self.choices is not None:
            if isinstance(value, (list, tuple, set, frozenset)):
                return len(value) > 0 and all(item in self.choices for item in value)
            return value in self.choices
        return self.low is not None and self.high is not None and self.low <= value <= self.high


_KNOBS: tuple[Knob, ...] = (
    # --- economics -------------------------------------------------------
    Knob(
        name="es_round_trip_friction_points",
        knob_class=FROZEN,
        value=0.358,
        summary="Measured ES futures round-trip friction, equal to $17.92.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="A new measured spread, fee and slippage study on owned ES data.",
    ),
    Knob(
        name="option_round_trip_fee_dollars",
        knob_class=FROZEN,
        value=3.08,
        summary="Measured SPXW round-trip cost, $1.54 per side.",
        evidence="v4/audit/autoresearch/pathd_phase0b_trackc_paper_transitions_2026_08_04/trackc_transition_evidence.json",
        unfreeze_condition="A new guarded paper round trip measuring per-side cost again.",
    ),
    # --- clock and causality --------------------------------------------
    Knob(
        name="feature_interval_law",
        knob_class=FROZEN,
        value="[t-60s, t)",
        summary="A minute boundary t means the closed interval [t-60s, t). The next bar is never used.",
        evidence="v5/research/training_twin.py",
        unfreeze_condition="Never for an existing result; a different cadence is a different experiment.",
    ),
    Knob(
        name="warm_up_intervals_discarded",
        knob_class=FROZEN,
        value=1,
        summary="The first subscription interval is partial and is discarded, including after a reconnect.",
        evidence="v5/research/training_twin.py",
        unfreeze_condition="Evidence that a first partial interval is complete, which contradicts the feed.",
    ),
    Knob(
        name="contract_identity",
        knob_class=FROZEN,
        value="raw_osi_symbol_plus_current_session_definitions",
        summary="Contracts are keyed by raw OSI symbol; instrument IDs are re-read every session.",
        evidence="v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/comparison_same_session_attempt002/comparison_result.json",
        unfreeze_condition="Evidence that instrument IDs are stable across sessions; all 492 observed changed.",
    ),
    # --- parity ----------------------------------------------------------
    Knob(
        name="feature_absolute_tolerance",
        knob_class=FROZEN,
        value=1e-12,
        summary="Offline and live feature cells must agree to this absolute tolerance.",
        evidence="v4/research/autoresearch_v2/runtime_decision_parity.py",
        unfreeze_condition="Never widened. A mismatch is a mechanism bug, not a tolerance problem.",
    ),
    Knob(
        name="feature_relative_tolerance",
        knob_class=FROZEN,
        value=0.0,
        summary="No relative tolerance is permitted on the same sealed input.",
        evidence="v4/research/autoresearch_v2/runtime_decision_parity.py",
        unfreeze_condition="Never widened.",
    ),
    # --- statistics ------------------------------------------------------
    Knob(
        name="fold_count",
        knob_class=FROZEN,
        value=5,
        summary="Five chronological, session-clustered folds. No fold trains on its evaluation sessions.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="A pre-registered change made before any outcome is seen.",
    ),
    Knob(
        name="fold_pass_rule",
        knob_class=FROZEN,
        value="4_of_5",
        summary="Both absolute and paired results must be positive in at least four of five folds.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="Never loosened. It was already too weak once, at one fold of five.",
    ),
    Knob(
        name="confidence_level",
        knob_class=FROZEN,
        value=0.95,
        summary="One-sided 95% lower bounds, session-block bootstrap.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="A pre-registered change made before any outcome is seen.",
    ),
    Knob(
        name="minimum_sessions_per_neural_parameter",
        knob_class=FROZEN,
        value=20,
        summary="At least 20 independent sessions per trainable parameter before a neural comparison.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="A published argument for a different ratio, pre-registered.",
    ),
    Knob(
        name="minimum_sessions_for_neural_comparison",
        knob_class=FROZEN,
        value=1140,
        summary="Absolute floor before any neural candidate may even be compared. Owned today: 254.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="Acquiring enough independent sessions, which is a data purchase decision.",
    ),
    # --- not certified ---------------------------------------------------
    Knob(
        name="emission_lag_ms",
        knob_class=UNCERTIFIED,
        value=2336,
        summary=(
            "The only available shared emission allowance. Derived from five ThetaData samples, which is "
            "the wrong source family for an OPRA stream and far too thin to fit against."
        ),
        evidence="v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/shared_emission_lag.json",
        unfreeze_condition=(
            "A multi-session Track-A OPRA capture re-deriving the allowance from the same stream the "
            "features come from. It may not be lowered unless the lowering rule is pre-registered first."
        ),
    ),
    Knob(
        name="historical_arrival_lag_ms",
        knob_class=UNCERTIFIED,
        value=0,
        summary=(
            "The stored arrival lag in the training corpus, which is zero on all 47,707,186 rows. It is "
            "not a measurement; the Historical API stamps a completed bar at its interval close."
        ),
        evidence="v5/research/findings/HISTORICAL_ARRIVAL_PARITY_2026_08_05.md",
        unfreeze_condition=(
            "Replace it with a signed OPRA latency receipt and inject arrival through "
            "v5.research.training_twin.simulate_historical_arrival."
        ),
    ),
    # --- searchable, once a gate releases them ---------------------------
    Knob(
        name="model_class",
        knob_class=SEARCHABLE,
        choices=("hist_gradient_boosting", "random_forest"),
        summary="Shallow rankers only. A neural candidate is barred by the two session-count knobs above.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        released_by_gate="G4",
    ),
    Knob(
        name="horizon_minutes",
        knob_class=SEARCHABLE,
        choices=(15, 30, 60),
        summary="The three pre-declared decision horizons. Only 60 can reopen the option wrapper.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        released_by_gate="G1",
    ),
    Knob(
        name="hold_minutes",
        knob_class=SEARCHABLE,
        low=1,
        high=390,
        summary="Position hold length, bounded by one trading session.",
        evidence="v5/research/training_twin.py",
        released_by_gate="G4",
    ),
    Knob(
        name="entry_threshold",
        knob_class=SEARCHABLE,
        low=0.0,
        high=1.0,
        summary="Score cutoff for taking a trade. Tuning it is threshold search and is owner-gated.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        released_by_gate="G4",
    ),
    # --- declared before any gate opens, so a search cannot widen itself -----
    Knob(
        name="feature_families",
        knob_class=SEARCHABLE,
        choices=(
            "entry.contract_clock.v1",
            "entry.opra_cbbo1m_native.v1",
            "entry.opra_cbbo1m_cross_section.v1",
            "entry.opra_implied_spot.v1",
            "entry.opra_implied_volatility.v1",
            "entry.self_computed_greeks.v1",
            "entry.opra_cbbo1s_rolling.v1",
            "entry.causal_account_state.v1",
            "entry.opra_ohlcv1m_sparse.v1",
        ),
        summary=(
            "Which certifiable feature families a fit may draw from, proposed as a subset "
            "of these nine. Admission still gates every individual feature; the ten "
            "no-live-twin rows are not offered at all."
        ),
        evidence="v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json",
        released_by_gate="G4",
    ),
    Knob(
        name="abstention_rule",
        knob_class=SEARCHABLE,
        choices=("no_trade_below_entry_threshold", "two_sided_score_band"),
        summary="How the policy declines to trade. Standing down is a first-class outcome.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        released_by_gate="G4",
    ),
    Knob(
        name="quote_age_cap_seconds",
        knob_class=FROZEN,
        value=90,
        summary="Maximum quote age at decision time, as used by the closed H1 screen.",
        evidence="v5/research/history/DO_NOT_RETEST.md",
        unfreeze_condition=(
            "A multi-session Track-A freshness receipt plus a pre-registered rule, "
            "declared before any search opens."
        ),
    ),
    Knob(
        name="fold_assignment",
        knob_class=FROZEN,
        value="chronological_contiguous_sessions",
        summary="Folds are contiguous chronological session blocks; never shuffled.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="A pre-registered change made before any outcome is seen.",
    ),
    Knob(
        name="bootstrap_seed",
        knob_class=FROZEN,
        value=0,
        summary="The gate's bootstrap seed. Reseeding until a bound clears is refused.",
        evidence="v5/research/validation/replay_gate.py",
        unfreeze_condition="Never for an existing result; a pre-registered change only.",
    ),
    Knob(
        name="random_forest_spec",
        knob_class=FROZEN,
        value="n_estimators=200,max_depth=5,min_samples_leaf=20",
        summary=(
            "The honest shallow benchmark specification, declared now so hyperparameter "
            "search cannot begin silently on the day training is authorized."
        ),
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="A pre-registered ablation protocol declared before any search.",
    ),
    Knob(
        name="hist_gradient_boosting_spec",
        knob_class=FROZEN,
        value="max_iter=200,max_depth=5,min_samples_leaf=20",
        summary="The declared boosting twin of the random-forest specification.",
        evidence="v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        unfreeze_condition="A pre-registered ablation protocol declared before any search.",
    ),
)

REGISTRY: Mapping[str, Knob] = {knob.name: knob for knob in _KNOBS}


def get(name: str) -> Knob:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KnobError(f"unknown knob: {name}") from None


def frozen_value(name: str) -> Any:
    """Read a frozen constant. Uncertified knobs refuse to be read this way."""

    knob = get(name)
    if knob.knob_class == UNCERTIFIED:
        raise KnobError(
            f"{name} is UNCERTIFIED and must not be used: {knob.summary} "
            f"Required: {knob.unfreeze_condition}"
        )
    if knob.knob_class != FROZEN:
        raise KnobError(f"{name} is {knob.knob_class}, not a frozen constant")
    return knob.value


def by_class(knob_class: str) -> tuple[Knob, ...]:
    return tuple(knob for knob in _KNOBS if knob.knob_class == knob_class)


def assert_search_space(
    params: Mapping[str, Any], *, released_gates: Iterable[str] = ()
) -> None:
    """Refuse a proposed search that touches anything it may not touch.

    ``params`` is what a hypothesis proposes to vary.  A run is permitted only
    when every named knob is SEARCHABLE, its releasing gate has passed, and the
    proposed value sits inside the declared space.  An unknown name is refused
    rather than ignored, so a search cannot smuggle in a new degree of freedom.
    """

    released = set(released_gates)
    problems: list[str] = []
    for name, value in sorted(params.items()):
        knob = REGISTRY.get(name)
        if knob is None:
            problems.append(f"{name}: not in the knob registry; declare it before searching it")
            continue
        if knob.knob_class == FROZEN:
            problems.append(
                f"{name}: FROZEN at {knob.value!r} ({knob.evidence}). "
                f"Unfreeze condition: {knob.unfreeze_condition}"
            )
            continue
        if knob.knob_class == UNCERTIFIED:
            problems.append(
                f"{name}: UNCERTIFIED and unusable. Required: {knob.unfreeze_condition}"
            )
            continue
        if knob.released_by_gate not in released:
            problems.append(
                f"{name}: SEARCHABLE but gate {knob.released_by_gate} has not passed"
            )
            continue
        if not knob.permits(value):
            space = knob.choices if knob.choices is not None else (knob.low, knob.high)
            problems.append(f"{name}: {value!r} is outside the declared space {space!r}")
    if problems:
        raise KnobError("proposed search is not permitted:\n  " + "\n  ".join(problems))


def blocking_uncertified(names: Sequence[str]) -> tuple[Knob, ...]:
    """Uncertified knobs among ``names``, for reporting why a run is blocked."""

    return tuple(
        knob
        for knob in (REGISTRY.get(name) for name in names)
        if knob is not None and knob.knob_class == UNCERTIFIED
    )
