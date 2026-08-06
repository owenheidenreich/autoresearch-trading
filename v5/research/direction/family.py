"""The frozen G1 direction family — declared before any outcome is inspected.

This module is a *declaration*, not an experiment. It computes nothing about
returns and reads no bars. Everything the screen is allowed to do is written
down here first and content-hashed, so the search space cannot widen after a
result is seen and a reviewer can diff the exact bytes that were frozen.

Why this file exists at all: ledger row 183 recorded a screen whose headline
`+0.522` rank correlation was 78.5% reproducible by surrogates with no
predictability by construction, because its features and its target shared the
price level `P(t)`. The lesson recorded there is that the *control* has to be
designed before the result is attractive. So does the family.

The three mechanisms, both directions, and three horizons make **18** members.
That is the multiplicity the surrogate campaign must control. No unregistered
variant may be substituted afterwards; adding one is a new experiment with a new
freeze.

Scope note: this screen answers one question — can a raw, no-model rule on owned
ES bars clear 0.358 points per round trip. It fits nothing, tunes nothing, and
touches no option data. Per the gate-chain audit §2.4, no barred option feature
is on its critical path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

from v5.research import knobs


FREEZE_SCHEMA_VERSION = "v5.g1-direction-family.v1"

# --- the eligible index, verified structurally from the owned bars -----------
# Counts reproduced 2026-08-06 from
# ~/.autoresearch-trading/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m
# by reading only session dates and instrument ids -- never a return.
ES_BARS_ROOT = (
    "/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31"
    "/raw/databento/glbx_es_ohlcv_1m"
)
CORPUS_FIRST_SESSION = "2025-08-01"
CORPUS_LAST_SESSION = "2026-07-31"
M1_ELIGIBLE_SESSIONS = 254
GAP_ELIGIBLE_SESSIONS = 249

# Cross-session contract changes. Excluded by instrument id *before* any outcome
# is read, never "adjusted": the expiring/new-contract spread is not owned, so a
# gap across a roll is a bookkeeping artifact rather than an overnight move.
ROLL_BOUNDARY_SESSIONS = (
    "2025-09-22",
    "2025-12-22",
    "2026-03-23",
    "2026-06-19",
)

# --- the causal clock --------------------------------------------------------
# At 09:35:00 ET the 09:30-09:34 bars are complete and the 09:35 bar is not.
# Entry is the CLOSE of the 09:35 bar, a full minute after the decision instant.
# That is deliberately conservative: the ES emission lag is UNCERTIFIED
# (knobs.emission_lag_ms), so the screen must not depend on filling inside the
# decision minute. It costs a minute of any real edge rather than assuming
# timing the live system has never been shown to deliver.
DECISION_TIME_ET = "09:35:00"
FEATURE_BARS_ET = ("09:30", "09:31", "09:32", "09:33", "09:34")
ENTRY_BAR_ET = "09:35"
ENTRY_PRICE = "close_of_09:35_bar"
EXIT_PRICE = "close_of_the_bar_horizon_minutes_after_the_entry_bar"
FORCED_FLAT_BY_SESSION_CLOSE = True

HORIZON_MINUTES = (15, 30, 60)

# --- economics ---------------------------------------------------------------
# One position, no overlap, at most one trade per session, forced flat by close.
#   session_net = side * (exit_price - entry_price) - friction * round_trips
# Every eligible calendar session contributes a number, including 0.0 on a
# no-trade day. Dropping no-trade days is the exact optimism the gate refuses.
MAX_TRADES_PER_SESSION = 1
FRICTION_KNOB = "es_round_trip_friction_points"

# --- statistics --------------------------------------------------------------
FOLD_COUNT_KNOB = "fold_count"
FOLD_RULE_KNOB = "fold_pass_rule"
CONFIDENCE_KNOB = "confidence_level"
BOOTSTRAP_SEED_KNOB = "bootstrap_seed"

# Comparator selection is causal and frozen as a *rule*, so it can never be
# reselected on the rows it is judged against: for fold k the comparator is the
# best constant side measured on sessions strictly before fold k begins. Fold 0
# has no prior sessions, so it takes the no-trade comparator by declaration.
COMPARATOR_CONTROLS = ("always_long", "always_short", "no_trade")
COMPARATOR_SELECTION = "best_constant_side_on_sessions_strictly_before_the_fold"
COMPARATOR_FOLD0 = "no_trade"

# --- surrogate known-answer gates -------------------------------------------
SURROGATE_CAMPAIGNS = 1000
SURROGATE_MAX_FALSE_PASS_RATE = 0.050
SURROGATE_MAX_WILSON_UPPER = 0.075
SHARED_TERM_FIXTURE_MAX_PASS_RATE = 0.050
INJECTED_EFFECT_MIN_RECOVERY = 0.800
NULL_REPAIRS_PERMITTED = 1

# A matched surrogate preserves everything about a session except the thing
# under test. Session shuffling is explicitly forbidden: row 183 recorded that
# it returns a clean null while a shared-term artifact goes unmeasured, because
# shuffling destroys the very pairing the artifact lives in.
SURROGATE_PRESERVES = (
    "session timestamps and bar count",
    "per-bar volume, exactly",
    "overnight gap magnitude",
    "per-bar absolute close-to-close change",
    "per-bar high-low range magnitude",
)
SURROGATE_RANDOMIZES = (
    "the sign of the overnight gap",
    "the sign of each bar's close-to-close change",
)
SURROGATE_REBUILDS = (
    "the complete price path",
    "every feature",
    "every selection and abstention",
    "every target",
)
FORBIDDEN_NULLS = ("session_shuffle", "row_shuffle", "any null reusing real-data signals")


@dataclass(frozen=True)
class Member:
    """One frozen family member. Occupancy and side are rules, not parameters."""

    name: str
    mechanism: str
    occupancy: str
    score: str
    side_rule: str
    horizon_minutes: int
    eligible_sessions: int


# --- the three mechanisms ----------------------------------------------------
# Each produces a signed score from causal fields only. "Direction" is whether
# the policy trades with the score or against it; both are declared, so neither
# can be chosen after the fact.
#
# M1  first-five-minute acceptance. Opening inventory accepted on unusual volume
#     may continue; the same move on thin volume may be rejected and revert.
#     Occupancy uses a volume surprise at or above the expanding median of
#     EARLIER sessions only -- 1.0 is the median itself, not a tuned threshold,
#     and the expanding median cannot see its own session or any later one.
# M3  overnight revaluation. The close-to-open move, on non-roll sessions.
# JOINT  the single predeclared interaction: the gap, taken only when the first
#     five minutes confirm its sign. This is one member of the M1/M3 family, not
#     a third mechanism.
MECHANISMS: Mapping[str, Mapping[str, str]] = {
    "M1": {
        "occupancy": "volume_surprise >= 1.0",
        "score": "first_five_minute_return = close(09:34) - open(09:30)",
        "eligible": "all non-empty sessions",
    },
    "M3": {
        "occupancy": "non-roll session with a prior session close",
        "score": "overnight_gap = open(09:30) - prior_session_final_close",
        "eligible": "non-roll sessions with a prior close",
    },
    "JOINT": {
        "occupancy": (
            "non-roll session with a prior close, and "
            "sign(overnight_gap) == sign(first_five_minute_return)"
        ),
        "score": "overnight_gap",
        "eligible": "non-roll sessions with a prior close",
    },
}
DIRECTIONS: Mapping[str, str] = {
    "with": "side = sign(score)",
    "against": "side = -sign(score)",
}

# --- causal feature definitions ---------------------------------------------
# Every field is computable at 09:35:00 ET from data that had already arrived.
# The expanding median is the one field that looks across sessions, and it uses
# strictly earlier sessions, which is what keeps it out of the shared-term trap
# described in the gate-chain audit's feature table.
FEATURE_LAW: Mapping[str, str] = {
    "first_five_minute_return": "close(09:34) - open(09:30), same session",
    "first_five_minute_volume": "sum of volume over 09:30..09:34, same session",
    "volume_surprise": (
        "first_five_minute_volume / expanding median of the same quantity over "
        "STRICTLY EARLIER sessions; undefined on the first session, which "
        "therefore cannot trade an M1 member"
    ),
    "overnight_gap": (
        "open(09:30) - final close of the previous eligible session, same "
        "instrument id; undefined across a contract roll"
    ),
    "prior_session_final_close": "last bar close of the previous eligible session",
}


def _members() -> tuple[Member, ...]:
    built: list[Member] = []
    for mechanism, spec in MECHANISMS.items():
        eligible = (
            M1_ELIGIBLE_SESSIONS if mechanism == "M1" else GAP_ELIGIBLE_SESSIONS
        )
        for direction, side_rule in DIRECTIONS.items():
            for horizon in HORIZON_MINUTES:
                built.append(
                    Member(
                        name=f"{mechanism}.{direction}.{horizon}m",
                        mechanism=mechanism,
                        occupancy=spec["occupancy"],
                        score=spec["score"],
                        side_rule=side_rule,
                        horizon_minutes=horizon,
                        eligible_sessions=eligible,
                    )
                )
    return tuple(built)


FAMILY: tuple[Member, ...] = _members()
FAMILY_SIZE = len(FAMILY)


# --- pass criteria -----------------------------------------------------------
# All six must hold for a member to pass. Criterion 4 is what makes this a
# *cost-clearing* test rather than a drift test; criterion 6 is what stops a
# no-trade day from being quietly dropped.
PASS_CRITERIA: tuple[str, ...] = (
    "one-sided 95% whole-session block-bootstrap lower bound on mean net "
    "points/session is above 0.000",
    "the paired lower bound versus the frozen causal constant-side comparator "
    "is above 0.000",
    "net points/session and the paired delta are each positive in at least 4 of "
    "5 chronological folds",
    "gross points per executed trade exceed 0.358 at the point estimate, and "
    "the absolute net criterion still passes with every no-trade day included",
    "the whole 18-member family is familywise-controlled by the surrogate "
    "campaign, with no unregistered variant substituted after outcomes are seen",
    "the full eligible index is evaluated: 254 sessions for M1 members and 249 "
    "for M3 and joint members",
)

# Only an exact 60-minute pass may reopen the option wrapper, and only for one
# locked option-dollar replay -- never model search (ledger row 181).
OPTION_REOPENING_HORIZON = 60

# Honest outcomes. "Underpowered" is decided from the MDE recomputed on the
# frozen realized occupancy, before the economic outcome is read -- not by
# counting fold labels afterwards.
VERDICTS = ("PASS", "NO_LARGE_EDGE", "UNDERPOWERED")


def declaration() -> dict[str, Any]:
    """The complete frozen declaration, as plain data for hashing and review."""

    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "question": (
            "Can a raw, no-model rule on owned ES bars predict direction over "
            "15-60 minutes well enough to clear 0.358 points per round trip?"
        ),
        "corpus": {
            "root": ES_BARS_ROOT,
            "first_session": CORPUS_FIRST_SESSION,
            "last_session": CORPUS_LAST_SESSION,
            "m1_eligible_sessions": M1_ELIGIBLE_SESSIONS,
            "gap_eligible_sessions": GAP_ELIGIBLE_SESSIONS,
            "roll_boundary_sessions": list(ROLL_BOUNDARY_SESSIONS),
            "verified": (
                "session count and roll boundaries reproduced 2026-08-06 from "
                "session dates and instrument ids only"
            ),
        },
        "clock": {
            "decision_time_et": DECISION_TIME_ET,
            "feature_bars_et": list(FEATURE_BARS_ET),
            "entry_bar_et": ENTRY_BAR_ET,
            "entry_price": ENTRY_PRICE,
            "exit_price": EXIT_PRICE,
            "forced_flat_by_close": FORCED_FLAT_BY_SESSION_CLOSE,
            "horizons_minutes": list(HORIZON_MINUTES),
        },
        "features": dict(FEATURE_LAW),
        "mechanisms": {k: dict(v) for k, v in MECHANISMS.items()},
        "directions": dict(DIRECTIONS),
        "family": [
            {
                "name": m.name,
                "mechanism": m.mechanism,
                "occupancy": m.occupancy,
                "score": m.score,
                "side_rule": m.side_rule,
                "horizon_minutes": m.horizon_minutes,
                "eligible_sessions": m.eligible_sessions,
            }
            for m in FAMILY
        ],
        "family_size": FAMILY_SIZE,
        "economics": {
            "max_trades_per_session": MAX_TRADES_PER_SESSION,
            "friction_points_per_round_trip": knobs.frozen_value(FRICTION_KNOB),
            "session_net": "side * (exit - entry) - friction * round_trips",
            "no_trade_day": "contributes 0.0 and is never dropped",
        },
        "statistics": {
            "folds": knobs.frozen_value(FOLD_COUNT_KNOB),
            "fold_assignment": knobs.frozen_value("fold_assignment"),
            "fold_pass_rule": knobs.frozen_value(FOLD_RULE_KNOB),
            "confidence_level": knobs.frozen_value(CONFIDENCE_KNOB),
            "bootstrap_seed": knobs.frozen_value(BOOTSTRAP_SEED_KNOB),
            "comparator_controls": list(COMPARATOR_CONTROLS),
            "comparator_selection": COMPARATOR_SELECTION,
            "comparator_fold0": COMPARATOR_FOLD0,
        },
        "surrogate_gate": {
            "campaigns": SURROGATE_CAMPAIGNS,
            "max_false_pass_rate": SURROGATE_MAX_FALSE_PASS_RATE,
            "max_wilson_upper": SURROGATE_MAX_WILSON_UPPER,
            "shared_term_fixture_max_pass_rate": SHARED_TERM_FIXTURE_MAX_PASS_RATE,
            "injected_effect_min_recovery": INJECTED_EFFECT_MIN_RECOVERY,
            "null_repairs_permitted": NULL_REPAIRS_PERMITTED,
            "preserves": list(SURROGATE_PRESERVES),
            "randomizes": list(SURROGATE_RANDOMIZES),
            "rebuilds": list(SURROGATE_REBUILDS),
            "forbidden_nulls": list(FORBIDDEN_NULLS),
        },
        "pass_criteria": list(PASS_CRITERIA),
        "option_reopening_horizon_minutes": OPTION_REOPENING_HORIZON,
        "verdicts": list(VERDICTS),
        "order_of_work": (
            "freeze this declaration -> run the surrogate known-answer campaign "
            "without inspecting real economics -> repair the null at most once "
            "-> recompute the MDE on the frozen realized occupancy -> one raw "
            "economic replay -> report PASS, NO_LARGE_EDGE, or UNDERPOWERED"
        ),
    }


def freeze_sha256() -> str:
    """Content hash of the declaration. Any widening changes these bytes."""

    return hashlib.sha256(
        json.dumps(declaration(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assert_member_registered(name: str) -> Member:
    """Refuse a member that was not declared before outcomes were seen."""

    for member in FAMILY:
        if member.name == name:
            return member
    raise ValueError(
        f"{name} is not in the frozen G1 family of {FAMILY_SIZE} members. "
        "Adding a variant after the freeze is a new experiment with a new "
        "freeze, not a member of this one."
    )
