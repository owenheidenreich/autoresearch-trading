from __future__ import annotations

from pathlib import Path

import pytest

from v5.research import knobs


ROOT = Path(__file__).resolve().parents[2]


def test_every_knob_names_evidence_that_exists() -> None:
    for knob in knobs.REGISTRY.values():
        assert knob.evidence, knob.name
        assert (ROOT / knob.evidence).exists(), f"{knob.name} cites missing {knob.evidence}"


def test_frozen_and_uncertified_knobs_state_how_to_unfreeze() -> None:
    for knob in knobs.by_class(knobs.FROZEN) + knobs.by_class(knobs.UNCERTIFIED):
        assert knob.unfreeze_condition.strip(), knob.name


def test_searchable_knobs_name_a_releasing_gate() -> None:
    for knob in knobs.by_class(knobs.SEARCHABLE):
        assert knob.released_by_gate.startswith("G"), knob.name


def test_frozen_constants_carry_the_measured_values() -> None:
    assert knobs.frozen_value("es_round_trip_friction_points") == 0.358
    assert knobs.frozen_value("option_round_trip_fee_dollars") == 3.08
    assert knobs.frozen_value("feature_absolute_tolerance") == 1e-12
    assert knobs.frozen_value("feature_relative_tolerance") == 0.0
    assert knobs.frozen_value("fold_pass_rule") == "4_of_5"


def test_uncertified_values_cannot_be_read_as_constants() -> None:
    """The 2,336 ms allowance must not become permission by being readable."""

    with pytest.raises(knobs.KnobError, match="UNCERTIFIED"):
        knobs.frozen_value("emission_lag_ms")
    with pytest.raises(knobs.KnobError, match="UNCERTIFIED"):
        knobs.frozen_value("historical_arrival_lag_ms")


def test_a_search_may_not_vary_a_frozen_knob() -> None:
    with pytest.raises(knobs.KnobError, match="FROZEN"):
        knobs.assert_search_space(
            {"es_round_trip_friction_points": 0.2}, released_gates={"G1", "G4"}
        )


def test_a_search_may_not_vary_an_uncertified_knob() -> None:
    with pytest.raises(knobs.KnobError, match="UNCERTIFIED"):
        knobs.assert_search_space({"emission_lag_ms": 500}, released_gates={"G1", "G4"})


def test_an_undeclared_parameter_is_refused_rather_than_ignored() -> None:
    with pytest.raises(knobs.KnobError, match="not in the knob registry"):
        knobs.assert_search_space({"secret_new_dial": 7}, released_gates={"G4"})


def test_a_searchable_knob_is_blocked_until_its_gate_passes() -> None:
    with pytest.raises(knobs.KnobError, match="G4 has not passed"):
        knobs.assert_search_space({"model_class": "random_forest"}, released_gates={"G1"})
    knobs.assert_search_space({"model_class": "random_forest"}, released_gates={"G1", "G4"})


def test_a_searchable_value_outside_its_declared_space_is_refused() -> None:
    with pytest.raises(knobs.KnobError, match="outside the declared space"):
        knobs.assert_search_space({"horizon_minutes": 5}, released_gates={"G1"})
    with pytest.raises(knobs.KnobError, match="outside the declared space"):
        knobs.assert_search_space({"hold_minutes": 9999}, released_gates={"G4"})
    knobs.assert_search_space({"horizon_minutes": 60}, released_gates={"G1"})


def test_a_neural_model_class_is_not_even_declared() -> None:
    """Neural candidates are barred by sample size, so the class is not offered."""

    assert "neural" not in str(knobs.get("model_class").choices).lower()
    assert knobs.frozen_value("minimum_sessions_for_neural_comparison") == 1140


def test_every_error_message_says_what_would_unblock_it() -> None:
    with pytest.raises(knobs.KnobError) as excinfo:
        knobs.assert_search_space(
            {"emission_lag_ms": 500, "feature_absolute_tolerance": 1e-6},
            released_gates={"G4"},
        )
    message = str(excinfo.value)
    assert "Required:" in message and "Unfreeze condition:" in message


def test_feature_family_composition_is_a_declared_subset() -> None:
    """A search proposes a subset of the nine certifiable families, nothing else."""

    knobs.assert_search_space(
        {"feature_families": ("entry.contract_clock.v1", "entry.opra_cbbo1m_native.v1")},
        released_gates={"G4"},
    )
    with pytest.raises(knobs.KnobError, match="outside the declared space"):
        knobs.assert_search_space(
            {"feature_families": ("entry.contract_clock.v1", "entry.barred_no_live_twin.v1")},
            released_gates={"G4"},
        )
    with pytest.raises(knobs.KnobError, match="outside the declared space"):
        knobs.assert_search_space({"feature_families": ()}, released_gates={"G4"})
    assert "entry.barred_no_live_twin.v1" not in knobs.get("feature_families").choices


def test_pre_declared_search_dimensions_exist_and_are_gated() -> None:
    """The realistic search dimensions are declared now, not on training day."""

    with pytest.raises(knobs.KnobError, match="G4 has not passed"):
        knobs.assert_search_space(
            {"abstention_rule": "two_sided_score_band"}, released_gates=set()
        )
    with pytest.raises(knobs.KnobError, match="FROZEN"):
        knobs.assert_search_space({"quote_age_cap_seconds": 30}, released_gates={"G4"})
    with pytest.raises(knobs.KnobError, match="FROZEN"):
        knobs.assert_search_space({"bootstrap_seed": 7}, released_gates={"G4"})
    with pytest.raises(knobs.KnobError, match="FROZEN"):
        knobs.assert_search_space(
            {"random_forest_spec": "n_estimators=5000"}, released_gates={"G4"}
        )
    assert knobs.frozen_value("quote_age_cap_seconds") == 90
    assert knobs.frozen_value("fold_assignment") == "chronological_contiguous_sessions"


def test_uncertified_knobs_can_be_reported_for_a_blocked_run() -> None:
    blocking = knobs.blocking_uncertified(
        ["emission_lag_ms", "fold_count", "historical_arrival_lag_ms"]
    )
    assert {knob.name for knob in blocking} == {
        "emission_lag_ms",
        "historical_arrival_lag_ms",
    }
