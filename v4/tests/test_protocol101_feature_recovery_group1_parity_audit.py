from __future__ import annotations

import json
from pathlib import Path

from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as audit


def test_group1_preregistration_is_machine_checkable() -> None:
    prereg = audit.load_json(audit.DEFAULT_PREREGISTRATION)
    definition = audit.load_json(Path(prereg["feature_group_definition_path"]))

    blockers = audit.validate_machine_checkable(prereg, definition)

    assert blockers == []
    assert set(definition["feature_expressions"]) == set(definition["candidate_recovered_model_facing_features"])
    assert not (
        set(definition["forbidden_model_facing_sources"])
        & set(definition["candidate_recovered_model_facing_features"])
    )


def test_window_filter_keeps_training_convention_only() -> None:
    assert not audit.in_window(audit.parse_ts("2026-06-30T13:31:00+00:00"), "09:32", "15:30")
    assert audit.in_window(audit.parse_ts("2026-06-30T13:32:00+00:00"), "09:32", "15:30")
    assert audit.in_window(audit.parse_ts("2026-06-30T19:30:00+00:00"), "09:32", "15:30")
    assert not audit.in_window(audit.parse_ts("2026-06-30T19:31:00+00:00"), "09:32", "15:30")


def _definition() -> dict:
    return audit.load_json(audit.DEFAULT_PLAN_DIR / "feature_group_definition.json")


def _vector(*, right: str = "P") -> list[float]:
    names = audit.candidate_feature_names()
    values = [0.0] * len(names)
    by_name = {name: idx for idx, name in enumerate(names)}
    values[by_name["market_last.spx_close"]] = 5000.0
    values[by_name["market_last.vix_close"]] = 20.0
    values[by_name["market_last.spx_vwap"]] = 5002.0
    values[by_name["market_last.omar"]] = -0.4
    values[by_name["market_last.session_range"]] = 40.0
    values[by_name["market_last.momentum_5m"]] = -4.0
    values[by_name["market_last.momentum_15m"]] = -8.0
    values[by_name["side.is_call"]] = 1.0 if right == "C" else 0.0
    values[by_name["side.is_put"]] = 1.0 if right == "P" else 0.0
    return values


def test_derived_group1_features_do_not_need_forbidden_option_microstructure() -> None:
    row = {
        "decision_ts": "2026-06-30T13:47:00+00:00",
        "candidate_universe": [{"contract_id": "SPXW-20260630-05000.000-P", "right": "P"}],
    }
    now = audit.parse_ts(row["decision_ts"])
    vix_lookup = {
        now: 20.0,
        now - audit.timedelta(minutes=5): 19.5,
        now - audit.timedelta(minutes=15): 19.0,
    }

    recovered, missing, _sources = audit.derive_recovered_features(
        row=row,
        contract_id="SPXW-20260630-05000.000-P",
        vector=_vector(right="P"),
        vix_lookup=vix_lookup,
        definition=_definition(),
    )

    assert recovered["spx_vwap_gap_points"] == -2.0
    assert recovered["vwap_side_alignment_flag"] == 1.0
    assert recovered["omar_side_alignment_flag"] == 1.0
    assert recovered["momentum15_side_alignment_flag"] == 1.0
    assert recovered["vix_change_5m"] == 0.5
    assert not any(missing.values())


def test_sign_mismatch_threshold_adjacency_is_feature_specific() -> None:
    rules = {
        "spx_vwap_gap_points_abs_max_for_threshold_adjacent": 0.5,
        "omar_abs_max_for_threshold_adjacent": 0.02,
        "momentum_points_abs_max_for_threshold_adjacent": 0.5,
        "vix_change_abs_max_for_threshold_adjacent": 0.02,
    }

    assert audit.sign_threshold_adjacent(
        "vwap_side_alignment_flag",
        {"spx_vwap_gap_points": 0.25},
        {"spx_vwap_gap_points": 0.75},
        rules,
    )
    assert not audit.sign_threshold_adjacent(
        "momentum15_side_alignment_flag",
        {"momentum_15m": 1.0},
        {"momentum_15m": 2.0},
        rules,
    )


def test_machine_check_detects_feature_hash_mismatch(tmp_path: Path) -> None:
    definition = _definition()
    definition_path = tmp_path / "feature_group_definition.json"
    definition_path.write_text(json.dumps(definition, indent=2) + "\n")
    prereg = {
        "feature_group_definition_path": str(definition_path),
        "feature_group_definition_sha256": "not-the-real-hash",
        "base_contract": definition["base_contract"],
        "base_model_facing_transform": definition["base_transform"],
    }

    blockers = audit.validate_machine_checkable(prereg, definition)

    assert "feature_group_definition_hash_mismatch" in blockers
