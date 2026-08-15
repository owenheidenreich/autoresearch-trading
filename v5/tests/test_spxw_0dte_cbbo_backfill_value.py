from __future__ import annotations

from v5.ops.assess_spxw_0dte_cbbo_backfill_value import project_value


def test_backfill_value_requires_a_smaller_open_architecture() -> None:
    got = project_value(
        current_complete_sessions=100,
        current_quote_sessions=110,
        additional_nonempty_sessions=300,
        effective_ns=[200.0, 400.0],
        parameter_counts={"closed_compact": 20, "shared": 90},
        closed_architectures={"closed_compact"},
    )

    assert got["projected_additional_complete_sessions"] == 272
    assert got["scenarios"]["observed_completeness"]["projected_complete_sessions"] == 372
    assert got["scenarios"]["observed_completeness"]["conservative_parameter_budget"] == [37, 74]
    assert got["architecture_support"]["closed_compact"]["closed_on_current_corpus"] is True
    assert got["architecture_support"]["shared"]["supported_all_nonempty_complete"] is False
    assert got["new_architecture_requirement"]["parameters_at_most_for_full_conservative_support"] == 37
    assert got["decision"].endswith("NEW_SMALLER_SHARED_LIFECYCLE_DESIGN")


def test_backfill_value_can_admit_an_existing_open_architecture() -> None:
    got = project_value(
        current_complete_sessions=100,
        current_quote_sessions=100,
        additional_nonempty_sessions=900,
        effective_ns=[500.0, 500.0],
        parameter_counts={"closed_compact": 20, "shared": 100},
        closed_architectures={"closed_compact"},
    )

    assert got["architecture_support"]["shared"]["supported_all_nonempty_complete"] is True
    assert got["decision"] == "BACKFILL_SUPPORTS_AN_EXISTING_OPEN_ARCHITECTURE"
