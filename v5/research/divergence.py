"""Every way historical training data can differ from what the live system sees.

Every gate in the chain tests the *model*.  None of them systematically tests
whether the training data describes the same game the live system plays.  G6
comes closest and is still not it: G6 is a **same-input** parity test — identical
inputs in, identical decisions out.  Every divergence of this class lives
*upstream* of G6, in how the inputs are constructed.  If historical and live
inputs are built differently, G6 passes perfectly and the model is still learning
a game nobody is playing.

Two instances were already known before this register existed, both recorded as
``UNCERTIFIED`` knobs: the corpus stores zero arrival lag on all 47,707,186 rows
while the live stream measured 227 ms at the median, and the shared emission
allowance comes from the wrong source family.  Both were found by stumbling into
them.  This module exists because they cannot be the only two, and because
"look harder" is not a control.

Three statuses:

``PROVEN_EQUAL``
    Historical and live are known to agree on this axis, with evidence.  This is
    a claim about what was checked, not a promise about what was not.

``MEASURED_DIFFERENT``
    They are known to differ, by a measured amount.  This is not fatal on its
    own — it blocks only until the axis carries a declared repair, the way
    arrival latency is repaired by injecting a measured receipt through
    ``training_twin.simulate_historical_arrival``.

``UNKNOWN``
    Nobody has checked.  This is the dangerous state and the reason the module
    is written as code rather than prose: an unchecked axis silently passes a
    review and silently fails a deployment.

Model-free and network-free.  It computes nothing about markets and admits
nothing; it only refuses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from v5.research import knobs


class DivergenceError(RuntimeError):
    """A proposed fit depends on an axis nobody has checked."""


PROVEN_EQUAL = "PROVEN_EQUAL"
MEASURED_DIFFERENT = "MEASURED_DIFFERENT"
UNKNOWN = "UNKNOWN"

INPUT_CONSTRUCTION = "input_construction"
TIMING = "timing"
EXECUTION = "execution"

_STATUSES = frozenset({PROVEN_EQUAL, MEASURED_DIFFERENT, UNKNOWN})
_GROUPS = frozenset({INPUT_CONSTRUCTION, TIMING, EXECUTION})


@dataclass(frozen=True)
class Axis:
    """One way the training environment can differ from the live one."""

    name: str
    group: str
    status: str
    summary: str
    evidence: str
    settles_when: str
    binds_at_gate: str
    repair: str = ""
    # Knobs this axis is the divergence explanation for.  The link is explicit
    # rather than textual so the two registries cannot drift apart quietly:
    # a knob may be UNCERTIFIED only if some axis here says why.
    knob_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.group not in _GROUPS:
            raise DivergenceError(f"unknown group for {self.name}: {self.group}")
        if self.status not in _STATUSES:
            raise DivergenceError(f"unknown status for {self.name}: {self.status}")
        for knob_name in self.knob_names:
            if knob_name not in knobs.REGISTRY:
                raise DivergenceError(
                    f"{self.name} names knob {knob_name}, which is not in the "
                    "knob registry"
                )
        # An axis with no exit is a parked problem, so the exit is mandatory.
        if not self.settles_when:
            raise DivergenceError(f"{self.name} must state what would settle it")
        if not self.binds_at_gate:
            raise DivergenceError(f"{self.name} must name the gate it binds at")
        if self.status != MEASURED_DIFFERENT and self.repair:
            raise DivergenceError(
                f"{self.name} names a repair but is {self.status}; a repair only "
                "means something for a measured difference"
            )

    @property
    def blocks_a_fit(self) -> bool:
        """Whether depending on this axis today should refuse a fit.

        An unchecked axis blocks, obviously.  A *measured* difference blocks too
        unless it names its repair — knowing the exact size of a gap you have
        not closed is no safer than not knowing, and the first version of this
        module got that wrong by treating any measurement as reassuring.
        """

        return self.status == UNKNOWN or (
            self.status == MEASURED_DIFFERENT and not self.repair
        )


_AXES: tuple[Axis, ...] = (
    # --- A. input construction -------------------------------------------
    Axis(
        name="bar_interval_labelling",
        group=INPUT_CONSTRUCTION,
        status=PROVEN_EQUAL,
        summary=(
            "Bars are start-labelled on ts_event: a bar labelled 09:35 covers "
            "[09:35:00, 09:36:00) and is complete at 09:36:00. training_twin "
            "converts with represented_end = ts_event + 60s."
        ),
        evidence=(
            "Owned ES corpus read 2026-08-09: 390 regular-hours bars running "
            "09:30 to 15:59, which is the start-labelled count. End-labelling "
            "would run 09:31 to 16:00."
        ),
        settles_when=(
            "The loader asserts this convention from the data instead of "
            "assuming it; a flipped convention shifts every feature by one "
            "minute and still looks plausible."
        ),
        binds_at_gate="G4",
    ),
    Axis(
        name="bar_completeness",
        group=INPUT_CONSTRUCTION,
        status=PROVEN_EQUAL,
        summary=(
            "A forming bar cannot reach the feature path. "
            "training_twin.select_option_feature_quote matches the exact minute "
            "boundary and raises unless exactly one row matches, so a partial "
            "bar is refused rather than silently averaged in."
        ),
        evidence="v5/research/training_twin.py select_option_feature_quote, exact_boundary=True",
        settles_when=(
            "A test proves no feature path builds its own bars from a "
            "sub-minute stream and bypasses the selector. That bypass is the "
            "one route by which live code could still mistake ticks for bars."
        ),
        binds_at_gate="G6",
    ),
    Axis(
        name="revisions_vs_first_print",
        group=INPUT_CONSTRUCTION,
        status=UNKNOWN,
        summary=(
            "Whether the Historical API serves corrected or revised bars that "
            "the live first-print never showed. If it does, the corpus contains "
            "prices no live system could have acted on."
        ),
        evidence="UNKNOWN - never checked",
        settles_when=(
            "A paired comparison of live-captured rows against the same "
            "session re-requested from the Historical API, on the identical "
            "instrument and interval."
        ),
        binds_at_gate="G4",
    ),
    Axis(
        name="universe_composition",
        group=INPUT_CONSTRUCTION,
        status=PROVEN_EQUAL,
        summary=(
            "Contracts are keyed by raw OSI symbol and instrument ids are "
            "re-read every session, so a stale id cannot silently select a "
            "different contract."
        ),
        evidence=(
            "knobs.contract_identity (FROZEN); all 492 prior-session instrument "
            "ids changed in the observed session; the 0DTE listing measured 510 "
            "symbols on 2026-08-05 and 574 on 2026-08-06."
        ),
        settles_when=(
            "Already settled for identity. Kept in the register because the "
            "daily-varying universe size is what broke the 08-06 capture."
        ),
        binds_at_gate="G3",
    ),
    Axis(
        name="definitions_survivorship",
        group=INPUT_CONSTRUCTION,
        status=UNKNOWN,
        summary=(
            "Whether a historical definitions file lists contracts that only "
            "came into existence later in the session. Training on those would "
            "select strikes the live system could not have seen at decision "
            "time."
        ),
        evidence="UNKNOWN - never checked",
        settles_when=(
            "Compare a live definitions capture taken at the open against the "
            "historical definitions for the same session, and show the "
            "historical set adds nothing before the decision instant."
        ),
        binds_at_gate="G3",
    ),
    Axis(
        name="sparse_minutes",
        group=INPUT_CONSTRUCTION,
        status=MEASURED_DIFFERENT,
        summary=(
            "An option chain is sparse: an illiquid strike quotes in one minute "
            "and not the next. A feature computed as though every instrument "
            "reports every minute assumes coverage the live feed does not give."
        ),
        evidence=(
            "2026-08-05 midday capture: 1,361 of 1,530 expected CBBO-1m "
            "instrument-minutes observed, so 169 were missing."
        ),
        settles_when=(
            "The Track-A freshness receipt signs the observed coverage envelope "
            "across the banked sessions."
        ),
        binds_at_gate="G3",
        repair=(
            "arrival_analysis.build_cbbo1m_freshness_receipt signs the measured "
            "coverage, and reissue_ledger refuses to admit the native family "
            "without it."
        ),
    ),
    # --- B. timing --------------------------------------------------------
    Axis(
        name="arrival_latency",
        group=TIMING,
        status=MEASURED_DIFFERENT,
        summary=(
            "The corpus claims every row arrived the instant it existed. The "
            "live stream does not. A model trained on the corpus unrepaired "
            "learns to act on information the live system cannot deliver in "
            "time."
        ),
        evidence=(
            "receive_time equals event_time on all 47,707,186 corpus rows across "
            "251 sessions, while live CBBO-1m measured 226.932 ms at the median "
            "and 527.622 ms at p99. "
            "v5/research/findings/HISTORICAL_ARRIVAL_PARITY_2026_08_05.md"
        ),
        settles_when=(
            "A signed Track-A latency receipt is injected into historical rows "
            "before any fit."
        ),
        binds_at_gate="G4",
        repair=(
            "training_twin.simulate_historical_arrival stamps a causal arrival "
            "time from a signed receipt; assert_no_zero_lag refuses a frame that "
            "still claims zero lag."
        ),
        knob_names=("historical_arrival_lag_ms",),
    ),
    Axis(
        name="emission_lag",
        group=TIMING,
        status=UNKNOWN,
        summary=(
            "How long after a minute closes the decision may legitimately be "
            "emitted. The only available number comes from five ThetaData "
            "samples, which is the wrong source family for an OPRA stream."
        ),
        evidence="knobs.emission_lag_ms (UNCERTIFIED)",
        settles_when=(
            "A multi-session Track-A capture re-derives the allowance from the "
            "same stream the features come from. It may not be lowered unless "
            "the lowering rule is pre-registered first."
        ),
        binds_at_gate="G4",
        knob_names=("emission_lag_ms",),
    ),
    Axis(
        name="session_boundaries",
        group=TIMING,
        status=MEASURED_DIFFERENT,
        summary=(
            "Half-days, holidays and early closes. A session assumed to be 390 "
            "minutes that is actually 210 will silently mislabel every horizon "
            "that runs past the close."
        ),
        evidence=(
            "Enumerated from the owned ES corpus 2026-08-09: 9 of 254 sessions "
            "are short. Seven close at 13:00 (2025-09-01, 2025-11-27, "
            "2026-01-19, 2026-02-16, 2026-05-25, 2026-06-19, 2026-07-03) and "
            "two at 13:15 (2025-11-28, 2025-12-24). All 254 still open at "
            "09:30."
        ),
        settles_when=(
            "Any horizon that could run past a short close is either handled or "
            "excluded by a rule declared before outcomes are read."
        ),
        binds_at_gate="G4",
        repair=(
            "G1's declared horizons exit at 09:50, 10:05 and 10:35, all of "
            "which clear the earliest short close of 12:59 by more than two "
            "hours, so the frozen family is unaffected. A longer horizon would "
            "need this rule before it could be declared."
        ),
    ),
    Axis(
        name="product_session_existence",
        group=TIMING,
        status=MEASURED_DIFFERENT,
        summary=(
            "Whether the instrument the bot would actually trade exists on a "
            "session the research measures. G1 measures ES direction in order "
            "to justify buying SPXW options, but ES trades a shortened session "
            "on US equity-market holidays when SPXW does not trade at all. An "
            "edge measured on those days cannot be taken in the product."
        ),
        evidence=(
            "Measured 2026-08-09 against the owned corpus: 7 of G1's 254 "
            "M1-eligible sessions (2.8%) have no SPXW option session at all, "
            "and 6 of the 249 gap-eligible ones. They are exactly the seven "
            "13:00-close ES sessions. The two 13:15 early closes do have SPXW "
            "sessions."
        ),
        settles_when=(
            "The owner decides whether the G1 index excludes sessions on which "
            "the traded product does not exist. No G1 outcome has been "
            "inspected, so deciding now is still a pre-outcome narrowing rather "
            "than a post-hoc one -- but it changes the frozen family hash and "
            "is the owner's call, not the agent's."
        ),
        binds_at_gate="G2",
    ),
    Axis(
        name="clock_and_dst",
        group=TIMING,
        status=UNKNOWN,
        summary=(
            "Conversions between UTC storage, Eastern market time and Pacific "
            "machine time, across both daylight-saving transitions. An "
            "off-by-one-hour error looks exactly like a real regime change."
        ),
        evidence="UNKNOWN - no fixture covers a transition date",
        settles_when=(
            "Fixtures cover both 2025-11 and 2026-03 transitions and prove the "
            "same wall-clock minute maps to the same bar on either side."
        ),
        binds_at_gate="G4",
    ),
    # --- C. execution, after the decision ---------------------------------
    # These bind at G7/G8 and mostly cannot be settled without live shadow
    # evidence that does not exist yet. They are named now so they are not
    # discovered late, which is the whole purpose of the register.
    Axis(
        name="fill_law",
        group=EXECUTION,
        status=MEASURED_DIFFERENT,
        summary=(
            "Research fills at a bar close; a live order fills against a quote "
            "at some later instant. These are different prices by construction, "
            "not by error."
        ),
        evidence=(
            "G1 declares entry at the close of the 09:35 bar "
            "(v5/research/direction/family.py ENTRY_PRICE), while live "
            "execution selects an executable quote at most 2 s old "
            "(training_twin.DEFAULT_MAX_ENTRY_QUOTE_AGE_NS)."
        ),
        settles_when=(
            "A declared and tested mapping from the research fill to the live "
            "fill, measured on shadow evidence."
        ),
        binds_at_gate="G7",
        repair=(
            "G1 deliberately enters a full minute after its decision instant "
            "rather than assume timing the live system has not demonstrated, so "
            "the research fill is conservative rather than optimistic."
        ),
    ),
    Axis(
        name="slippage_accounting",
        group=EXECUTION,
        status=UNKNOWN,
        summary=(
            "Whether slippage is counted once, twice or not at all. The frozen "
            "friction knob is documented as a spread, fee and slippage study, "
            "while the candidate packet takes a separate per-side slippage "
            "argument that defaults to zero."
        ),
        evidence=(
            "knobs.es_round_trip_friction_points unfreeze_condition names "
            "slippage; v5/research/validation/candidate_packet.py takes "
            "slippage_per_side_points with default 0.0."
        ),
        settles_when=(
            "The friction knob's composition is decomposed into spread, fee and "
            "slippage, and the packet either consumes that decomposition or "
            "documents that its argument must stay zero to avoid double "
            "counting."
        ),
        binds_at_gate="G5",
    ),
    Axis(
        name="quote_age_at_execution",
        group=EXECUTION,
        status=PROVEN_EQUAL,
        summary=(
            "How stale a quote may be when an order is priced against it. Both "
            "sides read the same frozen cap through the same selector."
        ),
        evidence=(
            "training_twin.DEFAULT_MAX_ENTRY_QUOTE_AGE_NS is 2 s; "
            "knobs.quote_age_cap_seconds is FROZEN."
        ),
        settles_when=(
            "A boundary fixture pins the exact comparison for a quote sitting "
            "precisely on the age limit, so both sides round the same way."
        ),
        binds_at_gate="G6",
    ),
    Axis(
        name="reconnects_and_gaps",
        group=EXECUTION,
        status=UNKNOWN,
        summary=(
            "Reconnects, duplicates, corrections and sequence gaps legitimately "
            "change what input has arrived. They do not waive parity: live must "
            "fail closed to WAIT, preserve the sealed decision, and log the "
            "divergence."
        ),
        evidence=(
            "Specified in v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md "
            "section 8; StreamReadiness discards the first interval after any "
            "reconnect. Never tested end to end."
        ),
        settles_when=(
            "Boundary fixtures for a reconnect, a duplicate, a correction and a "
            "sequence gap each show the live path failing closed rather than "
            "emitting."
        ),
        binds_at_gate="G6",
    ),
    Axis(
        name="partial_fills_and_rejects",
        group=EXECUTION,
        status=UNKNOWN,
        summary=(
            "Research assumes a position is taken whole or not at all. A live "
            "order can fill partially, queue behind others, or be rejected."
        ),
        evidence="UNKNOWN - no shadow evidence exists; G7 is not built",
        settles_when=(
            "Twenty consecutive no-order shadow sessions record what the broker "
            "would have done with each intended order."
        ),
        binds_at_gate="G7",
    ),
)


REGISTRY: Mapping[str, Axis] = {axis.name: axis for axis in _AXES}


def get(name: str) -> Axis:
    try:
        return REGISTRY[name]
    except KeyError:
        raise DivergenceError(f"unknown divergence axis: {name}") from None


def by_group(group: str) -> tuple[Axis, ...]:
    if group not in _GROUPS:
        raise DivergenceError(f"unknown group: {group}")
    return tuple(axis for axis in _AXES if axis.group == group)


def by_status(status: str) -> tuple[Axis, ...]:
    if status not in _STATUSES:
        raise DivergenceError(f"unknown status: {status}")
    return tuple(axis for axis in _AXES if axis.status == status)


def unchecked() -> tuple[Axis, ...]:
    """Every axis nobody has checked. The register's headline number."""

    return by_status(UNKNOWN)


def assert_no_unknown_on_path(
    axis_names: Sequence[str], *, purpose: str, settled: Iterable[str] = ()
) -> tuple[Axis, ...]:
    """Refuse work that depends on an axis nobody has checked.

    ``settled`` lets a caller name axes that this particular run has since
    settled with evidence, so the register does not have to be edited in the
    same breath as the run that settles it.  An unknown *name* is refused
    rather than ignored: the failure mode this module exists to prevent is a
    divergence nobody wrote down.
    """

    settled_names = set(settled)
    unknown_names = [name for name in settled_names if name not in REGISTRY]
    if unknown_names:
        raise DivergenceError(
            "settled names not in the register: " + ", ".join(sorted(unknown_names))
        )

    problems: list[str] = []
    axes: list[Axis] = []
    for name in axis_names:
        axis = REGISTRY.get(name)
        if axis is None:
            problems.append(
                f"{name}: not in the divergence register; declare it before "
                "depending on it"
            )
            continue
        axes.append(axis)
        if axis.blocks_a_fit and name not in settled_names:
            problems.append(
                f"{name}: {UNKNOWN} - {axis.summary} Settles when: {axis.settles_when}"
            )
    if problems:
        raise DivergenceError(
            f"{purpose} depends on unchecked train/live divergence:\n  "
            + "\n  ".join(problems)
        )
    return tuple(axes)


def summary() -> dict[str, int]:
    """Counts by status, for a status page or a report."""

    return {status: len(by_status(status)) for status in sorted(_STATUSES)}
