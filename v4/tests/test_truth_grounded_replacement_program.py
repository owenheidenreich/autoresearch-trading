from __future__ import annotations

import pandas as pd

from v4.scripts import run_truth_grounded_replacement_program as program


def test_strategy_registry_is_complete_and_blocks_training() -> None:
    registry = program.build_strategy_hypothesis_registry()
    errors = program.validate_strategy_registry(registry)

    assert errors == []
    assert set(program.REGISTRY_COLUMNS).issubset(registry.columns)
    assert registry["playbook_id"].is_unique
    assert registry["model_training_allowed"].eq(False).all()
    assert "PLAYBOOK_AWARE_REPLACEMENT_POLICY_V1" in set(registry["playbook_id"])


def test_replacement_spec_preserves_protocol101_and_forbids_leakage() -> None:
    registry = program.build_strategy_hypothesis_registry()
    rulebook = program.build_stage_gate_rulebook()
    spec = program.build_replacement_candidate_protocol_spec(registry, rulebook)

    assert spec["paper_default_baseline"] == "PAPER_DEFAULT_PROTOCOL101"
    assert spec["training_allowed"] is False
    assert "G4_execution_realism" in spec["blocked_by_gates"]
    assert "future PnL" in spec["forbidden_inputs"]
    assert "oracle action" in spec["forbidden_inputs"]
    assert spec["model_shape"]["rowwise_candidate_scorer_only"] == "forbidden"
    assert spec["protocol_id"] == "PLAYBOOK_AWARE_REPLACEMENT_CANDIDATE_PROTOCOL_V1"
    assert spec["optimization_target"] == "incremental_utility_over_protocol101_not_raw_pnl"
    assert "slot_opportunity_cost" in spec["must_include"]
    assert "timing_and_fill_penalty" in spec["must_include"]


def test_program_validation_requires_blocked_gates() -> None:
    registry = program.build_strategy_hypothesis_registry()
    weakness = program.build_protocol101_weakness_matrix(
        {
            "headline": {"hard_stop_trades": 1, "hard_stop_pnl": -100.0},
            "score_calibration": {"spearman_score_margin_vs_pnl": 0.1, "spearman_score_margin_vs_mfe": 0.2},
            "slot_opportunity_status": {"status": "blocked", "reason": "missing"},
        }
    )
    data_layer = program.build_research_data_layer(
        {"readiness": {"status": "blocked_insufficient_fill_observations", "fill_observations": 0, "required_fill_observations": 30}},
        {"reservation": {"status": "reserved_pending_collection"}},
    )
    rulebook = program.build_stage_gate_rulebook()
    spec = program.build_replacement_candidate_protocol_spec(registry, rulebook)

    assert program.validate_replacement_program(registry, weakness, data_layer, rulebook, spec) == []


def test_validation_catches_training_enabled_too_early() -> None:
    registry = program.build_strategy_hypothesis_registry()
    registry.loc[0, "model_training_allowed"] = True

    errors = program.validate_strategy_registry(registry)

    assert "model_training_allowed_before_diagnostics" in errors


def test_data_layer_marks_broad_purchase_blocked() -> None:
    data_layer = program.build_research_data_layer(
        {"readiness": {"status": "blocked_insufficient_fill_observations", "fill_observations": 0, "required_fill_observations": 30}},
        {"reservation": {"status": "reserved_pending_collection"}},
    )

    assert not data_layer.empty
    assert data_layer["broad_purchase_allowed"].eq(False).all()
    assert "fill_observations" in set(data_layer["data_domain"])


def test_validation_detects_missing_required_registry_columns() -> None:
    registry = pd.DataFrame({"playbook_id": ["x"]})

    errors = program.validate_strategy_registry(registry)

    assert any(error.startswith("missing_registry_columns:") for error in errors)
