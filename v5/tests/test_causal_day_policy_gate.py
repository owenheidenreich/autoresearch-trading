from __future__ import annotations

import pytest

from v5.research import causal_day_architectures as architectures
from v5.research import causal_day_policy_gate as gate
from v5.research import knobs


def test_four_state_router_preserves_origin() -> None:
    assert gate.route_head("12:45", None) == "morning_entry"
    assert gate.route_head("12:46", None) == "afternoon_entry"
    assert gate.route_head("13:30", "morning") == "morning_exit"
    assert gate.route_head("10:00", "afternoon") == "afternoon_exit"


def test_every_declared_fit_is_blocked_under_current_repository_state() -> None:
    for architecture in gate.ARCHITECTURES:
        blockers = gate.fit_blockers(
            architecture,
            sessions=243,
            trainable_parameters=1,
            g1_passed=False,
            do_not_retest_reopened=False,
        )
        assert any("G1" in item for item in blockers)
        assert any("DO_NOT_RETEST" in item for item in blockers)


def test_even_one_parameter_neural_object_fails_absolute_session_floor() -> None:
    blockers = gate.fit_blockers(
        "neural_four_head",
        sessions=243,
        trainable_parameters=1,
        g1_passed=True,
        do_not_retest_reopened=True,
    )
    assert any("243 available < 1140 required" in item for item in blockers)


def test_a_suspension_outranks_both_general_reopenings(monkeypatch) -> None:
    """A suspension is not overridden by reopening the two evidence rules."""

    monkeypatch.setattr(gate, "SUSPENSION_LIFTED_BY", None)
    blockers = gate.fit_blockers(
        "shallow_four_head",
        sessions=243,
        g1_passed=True,
        do_not_retest_reopened=True,
    )
    assert any("suspended" in item for item in blockers)


def test_four_independent_is_conditional_even_with_enough_sessions() -> None:
    blockers = gate.fit_blockers(
        "four_independent",
        sessions=2000,
        trainable_parameters=50,
        g1_passed=True,
        do_not_retest_reopened=True,
        shared_specialization_justified=False,
    )
    assert any("conditional" in item for item in blockers)


def _signed() -> gate.Reopening:
    return gate.load_reopening()


def _in_scope(**overrides) -> dict:
    base = dict(
        sessions=243,
        reopening=_signed(),
        label=gate.REOPENED_LABEL,
        horizon=120,
        corpus=gate.REOPENED_CORPUS,
        declared_kill_conditions=gate.REQUIRED_KILL_CONDITIONS,
    )
    base.update(overrides)
    return base


def test_the_signed_reopening_exists_and_records_its_pinned_bytes() -> None:
    signed = _signed()
    assert signed.label == "serial_action_advantage_120m"
    assert signed.horizons == (120,)
    assert signed.document_sha256 == gate.REOPENING_DOCUMENT_SHA256
    assert signed.research_law_sha256 == gate.REOPENED_RESEARCH_LAW_SHA256


def test_an_edited_reopening_document_is_refused(tmp_path) -> None:
    changed = tmp_path / gate.REOPENING_DOCUMENT.name
    changed.write_bytes(gate.REOPENING_DOCUMENT.read_bytes() + b"\nchanged after signing\n")
    with pytest.raises(gate.ReopeningInvalid, match="digest mismatch"):
        gate.load_reopening(changed)


def test_the_gate_imports_standalone(tmp_path) -> None:
    """The gate must import first, not only after the architectures module.

    causal_day_architectures imports ROLES from the gate, so a module-level
    import of architectures inside the gate is circular. The suite masked this
    because its own imports happened to run in the working order.
    """

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from v5.research import causal_day_policy_gate"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_suspension_is_currently_lifted_by_a_named_document() -> None:
    """Lifting is explicit and names its authority, never an absence."""

    assert gate.SUSPENSION_LIFTED_BY == (
        "v5/governance/CAUSAL_DAY_FIT_RERULING_2026_08_14.md"
    )
    from pathlib import Path

    assert Path(gate.SUSPENSION_LIFTED_BY).exists()


def test_a_suspension_blocks_every_architecture_when_in_force(monkeypatch) -> None:
    """The owner's pause must be machinery, not memory.

    Tested by reinstating it rather than by relying on the current state, so
    the protection stays proven after the suspension is lifted.
    """

    monkeypatch.setattr(gate, "SUSPENSION_LIFTED_BY", None)
    for architecture in gate.ARCHITECTURES:
        blockers = gate.fit_blockers(architecture, **_in_scope())
        assert any("suspended" in item for item in blockers), architecture


def test_a_hand_built_reopening_cannot_bypass_a_suspension(monkeypatch) -> None:
    """The check lives in fit_blockers, so forging the scope object is not enough."""

    monkeypatch.setattr(gate, "SUSPENSION_LIFTED_BY", None)
    forged = gate.Reopening(
        label=gate.REOPENED_LABEL,
        horizons=gate.REOPENED_HORIZONS,
        corpus=gate.REOPENED_CORPUS,
        kill_conditions=(),
        document_sha256="0" * 64,
    )
    blockers = gate.fit_blockers(
        "shallow_joint",
        sessions=243,
        reopening=forged,
        label=gate.REOPENED_LABEL,
        horizon=120,
        corpus=gate.REOPENED_CORPUS,
        declared_kill_conditions=gate.REQUIRED_KILL_CONDITIONS,
    )
    assert any("suspended" in item for item in blockers)


def test_assert_fit_permitted_raises_for_all_five_while_suspended(monkeypatch) -> None:
    monkeypatch.setattr(gate, "SUSPENSION_LIFTED_BY", None)
    for architecture in gate.ARCHITECTURES:
        with pytest.raises(gate.PolicyFitBlocked):
            gate.assert_fit_permitted(architecture, **_in_scope())


def test_declared_parameter_counts_must_match_a_built_model() -> None:
    """The defect that suspended section 4: literals standing in for models.

    Section 4 claimed shallow_joint had 676 parameters. It builds 1,604. A gate
    that trusts the caller's integer would have admitted a model at 2.4x the
    size it was ruled on.
    """

    blockers = gate.fit_blockers(
        "shallow_joint", **_in_scope(trainable_parameters=676)
    )
    assert any("does not match" in item for item in blockers)

    computed = architectures.computed_parameter_counts()
    truthful = gate.fit_blockers(
        "shallow_joint", **_in_scope(trainable_parameters=computed["shallow_joint"])
    )
    assert not any("does not match" in item for item in truthful)


def test_section_4_counts_were_right_at_the_width_the_project_uses() -> None:
    """The withdrawn defect, pinned so it cannot be re-reported.

    A 2026-08-14 review recomputed at the dataclass default of 16 and read the
    mismatch as five wrong counts. Every artifact was built at width 8, where
    section 4's numbers are exact.
    """

    assert architectures.computed_parameter_counts(8) == {
        "shallow_joint": 676,
        "shallow_four_head": 720,
        "neural_joint": 1_252,
        "neural_four_head": 1_296,
        "four_independent": 4_920,
        "compact_interaction_entry": 48,
    }


def test_absolute_selector_at_a_clipped_ceiling_is_refused_as_a_class() -> None:
    with pytest.raises(gate.SelectorUnattainable, match="strictly below"):
        gate.assert_selector_attainable(
            gate.SelectorAttainability(
                selector_name="regression_at_clip_ceiling",
                rule_kind="absolute_threshold",
                target_clip_bounds=(-25.0, 30.0),
                absolute_threshold=30.0,
                proof_source="declared target transform",
            )
        )


def test_relative_action_selector_requires_a_firing_model_witness() -> None:
    proof = gate.SelectorAttainability(
        selector_name="enter_beats_feasible_wait",
        rule_kind="relative_action_value",
        no_trade_floor=0.0,
        witness_enter=1_000.0,
        witness_wait=0.0,
        proof_source="built compact model",
    )
    gate.assert_selector_attainable(proof)
    with pytest.raises(gate.SelectorUnattainable, match="cannot fire"):
        gate.assert_selector_attainable(
            gate.SelectorAttainability(
                **{**proof.to_dict(), "witness_enter": 0.0}
            )
        )


def test_fit_gate_refuses_an_unattainable_selector_before_fit_scope() -> None:
    with pytest.raises(gate.SelectorUnattainable, match="strictly below"):
        gate.assert_fit_permitted(
            "compact_interaction_entry",
            selector_attainability=gate.SelectorAttainability(
                selector_name="bad_boundary",
                rule_kind="absolute_threshold",
                target_clip_bounds=(-25.0, 30.0),
                absolute_threshold=30.0,
                proof_source="declared target transform",
            ),
            **_in_scope(),
        )


def test_signed_scope_permits_only_the_canonical_compact_action_value_fit() -> None:
    computed = architectures.computed_parameter_counts()
    blockers = gate.fit_blockers(
        "compact_interaction_entry",
        **_in_scope(trainable_parameters=computed["compact_interaction_entry"]),
    )
    assert blockers == ()


def test_the_ruling_width_fits_every_comparison_architecture() -> None:
    """Width 3 is the largest that fits the measured 377-parameter budget."""

    budget = gate.MEASURED_EFFECTIVE_OBSERVATIONS // int(
        knobs.frozen_value("minimum_sessions_per_neural_parameter")
    )
    assert budget == 377

    width = int(knobs.frozen_value("causal_day_hidden_size"))
    assert width == 3
    counts = architectures.computed_parameter_counts(width)
    for name in ("shallow_joint", "shallow_four_head", "neural_joint", "neural_four_head"):
        assert counts[name] <= budget, (name, counts[name])
    assert counts["four_independent"] > budget

    # And the next width up does not fit, so 3 is not an arbitrary choice.
    assert any(
        count > budget
        for name, count in architectures.computed_parameter_counts(width + 1).items()
        if name != "four_independent"
    ) or max(
        architectures.computed_parameter_counts(width + 1)[n]
        for n in ("shallow_joint", "shallow_four_head", "neural_joint", "neural_four_head")
    ) > budget


def test_hidden_size_has_no_silent_default() -> None:
    """The trap that caused the false defect report is removed at source."""

    import dataclasses

    field = {f.name: f for f in dataclasses.fields(architectures.ArchitectureDimensions)}[
        "hidden_size"
    ]
    assert field.default is dataclasses.MISSING
    with pytest.raises(TypeError):
        architectures.ArchitectureDimensions(
            candle_features=18,
            ladder_features=23,
            account_features=5,
            position_features=10,
            clock_features=5,
        )


def test_dimensions_are_derived_from_the_tensorizer_not_transcribed() -> None:
    from v5.research import causal_day_tensorizer as tz

    dimensions = architectures.declared_dimensions()
    assert dimensions.candle_features == len(tz.CANDLE_FEATURES)
    assert dimensions.ladder_features == len(tz.LADDER_FEATURES)
    assert dimensions.account_features == tz.ACCOUNT_FEATURES
    assert dimensions.position_features == tz.POSITION_FEATURES
    assert dimensions.clock_features == tz.CLOCK_FEATURES
    assert dimensions.hidden_size == int(knobs.frozen_value("causal_day_hidden_size"))


def test_a_fit_that_drifts_outside_the_signed_label_is_still_refused() -> None:
    """The reopening released one label, not the long side in general."""

    blockers = gate.fit_blockers(
        "shallow_joint", **_in_scope(label="dollar_excursion")
    )
    assert any("outside the signed reopening" in item for item in blockers)


def test_a_fit_that_drifts_outside_the_signed_horizon_is_still_refused() -> None:
    blockers = gate.fit_blockers("shallow_joint", **_in_scope(horizon=15))
    assert any("outside the signed reopening" in item for item in blockers)


def test_a_fit_on_a_different_corpus_is_still_refused() -> None:
    blockers = gate.fit_blockers("shallow_joint", **_in_scope(corpus="spxw_prints_1045"))
    assert any("outside the signed reopening" in item for item in blockers)


def test_dropping_any_kill_condition_refuses_the_fit() -> None:
    """Each kill condition encodes a failure this project already suffered.

    Dropping the mid-to-mid one is the dangerous case: row 340's decisive
    measurement was that the edge is absent with the spread removed entirely,
    so a run that does not report it could re-present a spread artifact as an
    edge.
    """

    for dropped in gate.REQUIRED_KILL_CONDITIONS:
        remaining = tuple(
            name for name in gate.REQUIRED_KILL_CONDITIONS if name != dropped
        )
        blockers = gate.fit_blockers(
            "shallow_joint", **_in_scope(declared_kill_conditions=remaining)
        )
        assert any(dropped in item for item in blockers), dropped


def test_the_reopening_does_not_release_the_conditional_specialists() -> None:
    """Four independent specialists still need prior shared-head evidence."""

    blockers = gate.fit_blockers(
        "four_independent", **_in_scope(trainable_parameters=4_920)
    )
    assert any("conditional" in item for item in blockers)


def test_the_measured_budget_excludes_the_largest_architecture() -> None:
    """The budget is a real constraint, not a formality."""

    computed = architectures.computed_parameter_counts()
    blockers = gate.fit_blockers(
        "four_independent",
        **_in_scope(
            trainable_parameters=computed["four_independent"],
            shared_specialization_justified=True,
        ),
    )
    assert any("evidence budget" in item for item in blockers)
    assert any("measured effective observations" in item for item in blockers)

    near = gate.fit_blockers(
        "neural_four_head",
        **_in_scope(trainable_parameters=computed["neural_four_head"]),
    )
    assert not any("evidence budget" in item for item in near)


def test_the_budget_binds_shallow_architectures_too() -> None:
    """Section 4 applied the ratio only to sequence models. This does not.

    Effective sample size is a property of the label and corpus, so a shallow
    parameter costs the same evidence as a sequence parameter. Checked by
    declaring a shallow model built at a width the budget cannot afford.
    """

    oversized = architectures.computed_parameter_counts(8)["shallow_joint"]
    assert oversized == 676 > gate.MEASURED_EFFECTIVE_OBSERVATIONS // 20
    blockers = gate.fit_blockers(
        "shallow_joint", **_in_scope(trainable_parameters=oversized)
    )
    # It is refused twice over: the count does not match the ruling width, and
    # even if it did the budget would not carry it.
    assert any("does not match" in item for item in blockers)


def test_the_frozen_ratio_is_untouched_for_every_other_purpose() -> None:
    """The ruling amends the ratio for this comparison only, not the registry."""

    from v5.research import knobs

    assert int(knobs.frozen_value("minimum_sessions_per_neural_parameter")) == 20
    assert int(knobs.frozen_value("minimum_sessions_for_neural_comparison")) == 1_140
    # Without the reopening, the old floor still binds exactly as before.
    blockers = gate.fit_blockers(
        "neural_joint",
        sessions=243,
        trainable_parameters=1_252,
        g1_passed=True,
        do_not_retest_reopened=True,
    )
    assert any("243 available < 1140 required" in item for item in blockers)


def test_assert_fit_permitted_fails_closed() -> None:
    with pytest.raises(gate.PolicyFitBlocked, match="fit refused"):
        gate.assert_fit_permitted(
            "shallow_joint",
            sessions=243,
            g1_passed=False,
            do_not_retest_reopened=False,
        )
