"""The second G1 attempt: one mechanism, both directions, frozen before any outcome.

This is a *declaration*, not an experiment. It reads no bars and computes nothing
about returns. Everything the screen may do is written down here first and
content-hashed, so the search space cannot widen after a result is seen.

Why a second attempt exists at all. The first (:mod:`v5.research.direction.family`,
eighteen members, hash ``157fe437...``) stopped on 2026-08-09 as ``UNDERPOWERED``:
its detection floor was 22-88x the cost bar and its real economics were never
computed. Do-not-retest ledger row 186 records the verdict and names what would
be genuinely new -- **"more sessions, and essentially nothing else"**. The corpus
went from 247 to 2,435 eligible sessions on 2026-08-12 at a cost of $3.20, which
is exactly that condition and nothing else. That module is untouched and remains
the record of what was frozen for the closed attempt.

What changed, and why each change is legitimate:

* **More sessions.** 2,435 against 247. The named reopening condition.
* **One mechanism instead of three, one horizon instead of three.** M3 and the
  60-minute horizon were selected on *power and occupancy*, never on outcome --
  see :data:`SELECTION_RATIONALE`. The first attempt's economics do not exist, so
  no member's profitability is known to anyone.
* **A threshold in accuracy rather than points.** The first attempt was judged
  against 0.358 ES points, which is ES friction and answers "profitable in ES?".
  That was never the question: G1 exists only to reopen the option class that
  ledger row 181 closed structurally. The bar is what the *option layer* needs.
* **The fold rule demoted from pass criterion to reported diagnostic.** This buys
  the power the attempt needs. It is not a free choice and the risk is named in
  :data:`FOLD_DIAGNOSTIC`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from v5.research import knobs, statistics


FREEZE_SCHEMA_VERSION = "v5.g1-hypothesis.v2"
DECLARED_ON = "2026-08-12"

# --- the corpus -------------------------------------------------------------
ES_BARS_ROOT = str(
    Path.home()
    / ".autoresearch-trading/es_1m_2016-08-01_2026-07-31"
    / "raw/databento/glbx_es_ohlcv_1m"
)
CORPUS_FIRST_SESSION = "2016-08-01"
CORPUS_LAST_SESSION = "2026-07-31"
ACQUISITION_RECEIPT = (
    "v4/audit/autoresearch/es_history_acquisition_2026_08_12/acquisition_receipt.json"
)

# Built 2026-08-12 by structural rules only, before any outcome existed.
ELIGIBLE_SESSIONS = 2435
ELIGIBLE_INDEX_SHA256 = (
    "ade16417c6aca672b60d4adbeb16c6508ce529df4dd49d3a85f51e7f28d50e0a"
)

# Every exclusion is structural and was applied before any return was computed.
EXCLUSIONS: Mapping[str, Any] = {
    "equity_holiday_no_spxw": {
        "count": 67,
        "rule": "the session's last regular-hours bar is at or before 13:00",
        "why": (
            "On US equity market holidays ES trades a shortened session and SPXW "
            "does not trade at all, so an edge measured there cannot be taken in "
            "the product. The rule is derived from the data rather than from a "
            "holiday calendar: all seven sessions independently verified against "
            "the owned option corpus on 2026-08-09 close at 12:59, and every "
            "12:59/13:00 close in the corpus is such a holiday. The 13:14/13:15 "
            "closes are equity half-days on which SPXW does trade, and they are "
            "kept -- which matches the two verified kept sessions 2025-11-28 and "
            "2025-12-24."
        ),
    },
    "roll_boundary": {
        "count": 37,
        "rule": "the session's instrument id differs from the prior session's",
        "why": (
            "The expiring/new-contract spread is not owned, so a gap across a roll "
            "is a bookkeeping artifact rather than an overnight move."
        ),
    },
    "no_prior_session": {"count": 1, "rule": "first session in the corpus"},
    "missing_required_bar": {
        "count": 1,
        "rule": "the 09:30 open, 09:35 entry or 10:35 exit bar is absent",
    },
}
PRICE_CHAIN_LAW = (
    "Excluded sessions remain in the price chain: the next session's overnight gap "
    "is still measured against the excluded day's ES close, because that is the "
    "close the live system would actually have seen."
)

# --- the mechanism ----------------------------------------------------------
MECHANISM = "M3"
MECHANISM_SPEC: Mapping[str, str] = {
    "occupancy": "non-roll session with a prior session close",
    "score": "overnight_gap = open(09:30) - prior_session_final_close",
    "entry": "close(09:35)",
    "exit": "close(10:35)",
}
HORIZON_MINUTES = 60
DIRECTIONS: Mapping[str, str] = {
    "with": "side = sign(overnight_gap)",
    "against": "side = -sign(overnight_gap)",
}

SELECTION_RATIONALE: Mapping[str, str] = {
    "why_m3": (
        "Selected over M1 on occupancy and power, both of which are properties of "
        "the null campaign rather than of any outcome: M3 occupies 99.2% of "
        "sessions against M1's 56.0%, and its measured detection floor was 8 "
        "points/session against M1's 16. The first attempt's economics were never "
        "computed, so no member's profitability is known to anyone and this "
        "selection cannot be outcome-contaminated."
    ),
    "why_60_minutes": (
        "The option layer's requirement falls with horizon, and 60 minutes is the "
        "longest horizon the frozen family declared and the one row 181 names as "
        "the option-reopening horizon. It is where a plausible edge could clear "
        "option friction."
    ),
    "why_both_directions": (
        "Declared on 2026-08-12 after pricing the alternative. The mechanism "
        "argument is genuinely balanced -- an overnight gap is both information "
        "arriving (which should persist) and a thin-liquidity move (which should "
        "correct) -- and the project holds no evidence favouring either sign. "
        "Declaring one would be a 50/50 guess that wastes the attempt if wrong. "
        "Bonferroni across two members costs 0.86 accuracy points of a 8.93-point "
        "margin, so the guess is not worth its price. Owner decision."
    ),
}

MEMBERS: tuple[str, ...] = tuple(
    f"{MECHANISM}.{direction}.{HORIZON_MINUTES}m" for direction in DIRECTIONS
)
FAMILY_SIZE = len(MEMBERS)

# --- the bar ----------------------------------------------------------------
# Set by what the option layer needs, not by ES friction. Stated in accuracy
# because the corpus spans a 7.08x range of 60-minute volatility, so the same
# skill implies 0.93 ES points in 2017 and 6.84 in 2026 and no single points
# figure is comparable across it.
REQUIRED_ACCURACY = 0.6573
REQUIRED_ACCURACY_EVIDENCE = (
    "v5/research/findings/G1_THRESHOLD_FROM_THE_OPTION_LAYER_2026_08_12.md"
)
OPTION_BREAKEVEN_ACCURACY = 0.5799

FOLD_COUNT = 5
FOLD_DIAGNOSTIC = (
    "The 4-of-5 chronological fold rule is REPORTED, not required. Demoting it is "
    "what buys the power this attempt needs, and the cost is named here rather "
    "than hidden: the corpus spans a 7.08x range of 60-minute volatility (3.88 "
    "points in 2017 against 27.51 in 2026), so an edge that lives only in the "
    "violent years would be a volatility artifact rather than a strategy, and "
    "pooling can hide exactly that. The per-fold result must be read on every "
    "run. A pooled pass with a single positive fold is not a pass in substance "
    "and must be reported as such."
)

PASS_CRITERIA: tuple[str, ...] = (
    "the known-answer campaign passed all three of its criteria BEFORE any real "
    "economics were computed",
    "the block-bootstrap lower bound on directional accuracy, at 95% familywise "
    "confidence Bonferroni-corrected across the two declared members, exceeds "
    "the required accuracy",
    "the full declared eligible index was evaluated, with no session dropped",
    "the per-fold accuracy is reported alongside the pooled figure",
)

KNOWN_ANSWER_CRITERIA: Mapping[str, float] = {
    "matched_surrogate_false_pass_max": 0.05,
    "shared_term_fixture_false_pass_max": 0.05,
    "recovery_of_a_detectable_effect_min": 0.80,
}

STOPPING_RULE = (
    "Pre-committed 2026-08-12, before any outcome existed: if this screen returns "
    "no edge, the research programme closes and the work is written up as a "
    "negative result. Explicitly not a reason to reopen: another instrument, "
    "another horizon, another mechanism on this corpus, or a second null repair."
)


@dataclass(frozen=True)
class Power:
    """What this index can and cannot see, computed before it is run."""

    eligible_sessions: int
    family_size: int
    required_accuracy: float
    detectable_accuracy: float
    margin_points: float
    detectable_sharpe: float


def power() -> Power:
    """The power statement, recomputed from the declaration rather than quoted."""

    import math

    z = statistics.bonferroni_quantile(FAMILY_SIZE)
    # Residual conjunction penalty: G1's measured 2-4x with its eighteen-member
    # Bonferroni divided out, since this module applies Bonferroni explicitly.
    penalty = 4.0 / (
        (statistics.bonferroni_quantile(18) + statistics.Z_POWER_80)
        / (statistics.Z_95 + statistics.Z_POWER_80)
    )
    detectable = 0.5 + (z + statistics.Z_POWER_80) * penalty * 0.5 / math.sqrt(
        ELIGIBLE_SESSIONS
    )
    return Power(
        eligible_sessions=ELIGIBLE_SESSIONS,
        family_size=FAMILY_SIZE,
        required_accuracy=REQUIRED_ACCURACY,
        detectable_accuracy=round(detectable, 6),
        margin_points=round(100.0 * (REQUIRED_ACCURACY - detectable), 4),
        detectable_sharpe=round(
            statistics.detectable_sharpe(ELIGIBLE_SESSIONS, z_alpha=z), 4
        ),
    )


def declaration() -> dict[str, Any]:
    """Everything frozen, as one hashable payload."""

    p = power()
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "declared_on": DECLARED_ON,
        "question": (
            "Can a raw, no-model overnight-gap rule on owned ES bars predict "
            "60-minute direction accurately enough that the SPXW option built on "
            "it would be both profitable and measurable?"
        ),
        "supersedes": {
            "module": "v5/research/direction/family.py",
            "verdict": "UNDERPOWERED, 2026-08-09",
            "reopening_condition_met": "more sessions (247 -> 2,435)",
            "ledger_row": 186,
        },
        "corpus": {
            "root": ES_BARS_ROOT,
            "first_session": CORPUS_FIRST_SESSION,
            "last_session": CORPUS_LAST_SESSION,
            "acquisition_receipt": ACQUISITION_RECEIPT,
            "eligible_sessions": ELIGIBLE_SESSIONS,
            "eligible_index_sha256": ELIGIBLE_INDEX_SHA256,
            "exclusions": EXCLUSIONS,
            "price_chain_law": PRICE_CHAIN_LAW,
        },
        "mechanism": {MECHANISM: dict(MECHANISM_SPEC)},
        "horizon_minutes": HORIZON_MINUTES,
        "directions": dict(DIRECTIONS),
        "members": list(MEMBERS),
        "family_size": FAMILY_SIZE,
        "selection_rationale": dict(SELECTION_RATIONALE),
        "threshold": {
            "required_accuracy": REQUIRED_ACCURACY,
            "option_breakeven_accuracy": OPTION_BREAKEVEN_ACCURACY,
            "units": "directional accuracy, not points",
            "why_not_points": (
                "The corpus spans a 7.08x range of 60-minute volatility, so a fixed "
                "points bar is far too strict in calm years and far too lax in "
                "violent ones."
            ),
            "evidence": REQUIRED_ACCURACY_EVIDENCE,
        },
        "power": {
            "eligible_sessions": p.eligible_sessions,
            "detectable_accuracy": p.detectable_accuracy,
            "required_accuracy": p.required_accuracy,
            "margin_accuracy_points": p.margin_points,
            "detectable_annualised_sharpe": p.detectable_sharpe,
        },
        "folds": {"count": FOLD_COUNT, "rule": "chronological", "status": "DIAGNOSTIC"},
        "fold_diagnostic_note": FOLD_DIAGNOSTIC,
        "confidence_level": knobs.frozen_value("confidence_level"),
        "pass_criteria": list(PASS_CRITERIA),
        "known_answer_criteria": dict(KNOWN_ANSWER_CRITERIA),
        "stopping_rule": STOPPING_RULE,
        "computes_no_economics": True,
    }


def freeze_sha256() -> str:
    """Content hash of the declaration, so a later diff is mechanical."""

    payload = json.dumps(declaration(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
