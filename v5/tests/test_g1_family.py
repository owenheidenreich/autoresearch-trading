"""The frozen G1 family must stay frozen, and must stay honest.

These tests are deliberately adversarial about the ways a search widens itself
quietly: a member appearing that was never declared, a knob drifting away from
the registry, a forbidden null sneaking back, or the declaration's bytes
changing without anyone noticing.
"""
from __future__ import annotations

import pytest

from v5.research import knobs
from v5.research.direction import family


def test_the_family_is_exactly_eighteen_declared_members() -> None:
    """3 mechanisms x 2 directions x 3 horizons. This is the multiplicity the
    surrogate campaign has to control; it may not grow after a result."""

    assert family.FAMILY_SIZE == 18
    assert len({m.name for m in family.FAMILY}) == 18
    assert {m.mechanism for m in family.FAMILY} == {"M1", "M3", "JOINT"}
    assert {m.horizon_minutes for m in family.FAMILY} == {15, 30, 60}
    # Both directions exist for every mechanism and horizon, so neither can be
    # selected after the fact.
    for mechanism in ("M1", "M3", "JOINT"):
        for horizon in (15, 30, 60):
            names = {
                m.name
                for m in family.FAMILY
                if m.mechanism == mechanism and m.horizon_minutes == horizon
            }
            assert names == {
                f"{mechanism}.with.{horizon}m",
                f"{mechanism}.against.{horizon}m",
            }


def test_an_undeclared_member_is_refused() -> None:
    family.assert_member_registered("M1.with.15m")
    with pytest.raises(ValueError, match="not in the frozen G1 family"):
        family.assert_member_registered("M1.with.45m")
    with pytest.raises(ValueError, match="not in the frozen G1 family"):
        family.assert_member_registered("M2.with.15m")


def test_the_eligible_index_matches_the_verified_corpus() -> None:
    """247 tradeable sessions, 4 roll boundaries, 243 gap-eligible. From the
    bars on 2026-08-06 using session dates and instrument ids only."""

    assert family.M1_ELIGIBLE_SESSIONS == 247
    assert family.GAP_ELIGIBLE_SESSIONS == 243
    assert len(family.ROLL_BOUNDARY_SESSIONS) == 4
    assert len(family.NO_OPTION_SESSIONS) == 7
    # The two exclusion sets overlap: 2026-06-19 is both a contract roll and a
    # session on which SPXW does not trade, so it must be subtracted once, not
    # twice. Gap eligibility is the 247 tradeable sessions, less the first
    # (no prior close), less the rolls that are not already excluded.
    overlapping = set(family.ROLL_BOUNDARY_SESSIONS) & set(family.NO_OPTION_SESSIONS)
    assert overlapping == {"2026-06-19"}
    remaining_rolls = len(family.ROLL_BOUNDARY_SESSIONS) - len(overlapping)
    assert (
        family.M1_ELIGIBLE_SESSIONS - 1 - remaining_rolls
        == family.GAP_ELIGIBLE_SESSIONS
    )
    for member in family.FAMILY:
        expected = 247 if member.mechanism == "M1" else 243
        assert member.eligible_sessions == expected


def test_the_clock_never_uses_the_decision_bar_or_later_information() -> None:
    """Features end at 09:34; the 09:35 bar is incomplete at the 09:35:00
    decision instant and is used only for the entry fill, at its close."""

    assert family.DECISION_TIME_ET == "09:35:00"
    assert family.FEATURE_BARS_ET[-1] == "09:34"
    assert family.ENTRY_BAR_ET not in family.FEATURE_BARS_ET
    assert family.ENTRY_PRICE == "close_of_09:35_bar"
    assert family.FORCED_FLAT_BY_SESSION_CLOSE is True


def test_economics_clear_the_measured_cost_bar_from_the_registry() -> None:
    declaration = family.declaration()
    assert declaration["economics"]["friction_points_per_round_trip"] == 0.358
    assert declaration["economics"]["friction_points_per_round_trip"] == knobs.frozen_value(
        "es_round_trip_friction_points"
    )
    assert family.MAX_TRADES_PER_SESSION == 1
    assert "never dropped" in declaration["economics"]["no_trade_day"]


def test_statistics_come_from_the_knob_registry_not_local_copies() -> None:
    """A local constant that drifts from the registry is how a gate loosens
    without anyone deciding to loosen it."""

    stats = family.declaration()["statistics"]
    assert stats["folds"] == knobs.frozen_value("fold_count") == 5
    assert stats["fold_pass_rule"] == knobs.frozen_value("fold_pass_rule") == "4_of_5"
    assert stats["confidence_level"] == knobs.frozen_value("confidence_level") == 0.95
    assert stats["bootstrap_seed"] == knobs.frozen_value("bootstrap_seed") == 0


def test_the_comparator_cannot_be_reselected_on_the_rows_it_is_judged_against() -> None:
    assert family.COMPARATOR_SELECTION.endswith("strictly_before_the_fold")
    assert family.COMPARATOR_FOLD0 == "no_trade"
    assert set(family.COMPARATOR_CONTROLS) == {"always_long", "always_short", "no_trade"}


def test_session_shuffling_is_forbidden_as_a_null() -> None:
    """Ledger row 183: a session-shuffle control returns a clean null while a
    shared-term artifact goes unmeasured, because shuffling destroys the very
    pairing the artifact lives in. Matched surrogates are mandatory."""

    assert "session_shuffle" in family.FORBIDDEN_NULLS
    assert any("sign" in item for item in family.SURROGATE_RANDOMIZES)
    assert any("volume" in item for item in family.SURROGATE_PRESERVES)
    assert any("absolute close-to-close" in item for item in family.SURROGATE_PRESERVES)
    # The surrogate must rebuild the whole chain, not just the target.
    rebuilt = " ".join(family.SURROGATE_REBUILDS)
    for stage in ("price path", "feature", "selection", "target"):
        assert stage in rebuilt


def test_the_surrogate_gate_matches_the_audit_thresholds() -> None:
    assert family.SURROGATE_CAMPAIGNS == 1000
    assert family.SURROGATE_MAX_FALSE_PASS_RATE == 0.050
    assert family.SURROGATE_MAX_WILSON_UPPER == 0.075
    assert family.SHARED_TERM_FIXTURE_MAX_PASS_RATE == 0.050
    assert family.INJECTED_EFFECT_MIN_RECOVERY == 0.800
    assert family.NULL_REPAIRS_PERMITTED == 1


def test_only_a_sixty_minute_pass_can_reopen_the_option_wrapper() -> None:
    """Ledger row 181 closed 0DTE long premium structurally. Its one compatible
    door is demonstrated directional skill on the underlying, at 60 minutes."""

    assert family.OPTION_REOPENING_HORIZON == 60


def test_the_three_verdicts_include_an_honest_underpowered() -> None:
    assert set(family.VERDICTS) == {"PASS", "NO_LARGE_EDGE", "UNDERPOWERED"}


def test_the_declaration_hash_is_stable_and_sensitive() -> None:
    """The same declaration hashes the same; a widened family does not."""

    assert family.freeze_sha256() == family.freeze_sha256()
    assert len(family.freeze_sha256()) == 64

    widened = family.declaration()
    widened["family"].append({"name": "M1.with.45m"})
    import hashlib
    import json

    changed = hashlib.sha256(
        json.dumps(widened, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert changed != family.freeze_sha256()


def test_the_declaration_states_the_order_of_work_null_before_economics() -> None:
    order = family.declaration()["order_of_work"]
    assert order.index("surrogate") < order.index("economic replay")
    assert "recompute the MDE" in order
