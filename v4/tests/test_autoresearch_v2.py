from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from v4.research.autoresearch_v2.builtin import REFERENCE_EXIT, builtin_hypotheses
from v4.research.autoresearch_v2.cache import prediction_cache_key
from v4.research.autoresearch_v2.causality import assert_mutate_future_invariant
from v4.research.autoresearch_v2.compiler import ExperimentCompileError, compile_hypothesis
from v4.research.autoresearch_v2.confirmation import _confirmation_sessions, _sign_flip_p
from v4.research.autoresearch_v2.corrected_v3_foundation import eligible_sessions
from v4.research.autoresearch_v2.dataset import FoundationMismatch, rolling_folds, verify_foundation
from v4.research.autoresearch_v2.registry import DuplicateSemanticHypothesis, register
from v4.research.autoresearch_v2.screens import direction_screen, timing_screen
from v4.research.autoresearch_v2.signal_policy_experiment import hypothesis_payload
from v4.research.autoresearch_v2.statistics import session_blocked_max_t


def _valid_spec():
    spec = deepcopy(builtin_hypotheses()["entry_direction_call_vs_put_v1"])
    spec["feature_families"] = {"add": ["contract_clock"], "remove": []}
    spec["features"] = [
        {
            "name": "is_call",
            "family": "contract_clock",
            "available_at": "decision_time",
            "live_twin": "entry.contract_clock.v1",
        }
    ]
    return spec


def test_future_session_threshold_fails_compilation() -> None:
    spec = _valid_spec()
    spec["threshold"] = {"kind": "full_session_quantile", "value": 0.6, "fit_role": "outer_test"}
    with pytest.raises(ExperimentCompileError, match="future_session_threshold"):
        compile_hypothesis(spec)


def test_intraday_open_interest_without_live_twin_fails() -> None:
    spec = _valid_spec()
    spec["features"].append(
        {
            "name": "last_causal_open_interest",
            "family": "microstructure",
            "available_at": "decision_time",
            "live_twin": "historical OPRA statistics only",
        }
    )
    with pytest.raises(ExperimentCompileError, match="last_causal_open_interest:intraday"):
        compile_hypothesis(spec)


def test_prose_claim_is_not_an_executable_live_twin() -> None:
    spec = _valid_spec()
    spec["features"][0]["live_twin"] = "Protocol101 live option ladder"
    with pytest.raises(ExperimentCompileError, match="feature_without_executable_live_twin"):
        compile_hypothesis(spec)


def test_observed_but_unproved_cbbo_feature_is_not_fit_ready() -> None:
    spec = _valid_spec()
    spec["features"] = [
        {
            "name": "size_imbalance",
            "family": "live_safe_microstructure",
            "available_at": "interval_end_plus_frozen_lag",
            "live_twin": "entry.opra_cbbo1m_native.v1",
        }
    ]
    with pytest.raises(ExperimentCompileError, match=r"size_imbalance:BARRED"):
        compile_hypothesis(spec)


def test_renaming_cannot_evade_semantic_registry(tmp_path) -> None:
    first = compile_hypothesis(_valid_spec(), require_fit_ready=False)
    renamed = _valid_spec()
    renamed["hypothesis_id"] = "same_mechanics_new_goal_name"
    renamed["claim"] = "rewritten prose"
    second = compile_hypothesis(renamed, require_fit_ready=False)
    assert first.semantic_hash == second.semantic_hash
    path = tmp_path / "registry.jsonl"
    register(path, first, status="NO_INCREMENTAL_EDGE", result_path="first.json")
    with pytest.raises(DuplicateSemanticHypothesis):
        register(path, second, status="PROVISIONAL_EDGE", result_path="second.json")
    row = register(
        path,
        second,
        status="NO_INCREMENTAL_EDGE",
        result_path="reproduction.json",
        engine_source_hash="engine-v2",
        allow_reexecution=True,
    )
    assert row["reexecution_of"] == "first.json"
    assert row["previous_status"] == "NO_INCREMENTAL_EDGE"


def test_prediction_cache_key_has_no_threshold_dimension() -> None:
    common = dict(
        features=["a", "b"],
        target={"fields": ["y"]},
        model={"family": "hgb"},
        fold={"fold": 1},
        seed=101,
        foundation_hash="foundation",
    )
    assert prediction_cache_key(**common) == prediction_cache_key(**common)
    assert "threshold" not in prediction_cache_key.__annotations__


def test_mutate_future_invariance_executes_against_scorer() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "future_pnl": [10.0, -10.0]})
    receipt = assert_mutate_future_invariant(
        frame,
        feature_columns=["x"],
        future_columns=["future_pnl"],
        scorer=lambda values: values["x"].to_numpy() * 2.0,
    )
    assert receipt["status"] == "pass"
    assert receipt["mutated_future_columns"] == ["future_pnl"]


def test_foundation_hash_fails_before_any_source_decode(tmp_path) -> None:
    path = tmp_path / "foundation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoresearch_v2.development_foundation.v1",
                "role": "development",
                "holdout_access_count": 0,
                "foundation_sha256": "wrong",
                "session_manifest": "does-not-exist",
                "sessions": [],
            }
        )
    )
    with pytest.raises(FoundationMismatch, match="manifest hash mismatch"):
        verify_foundation(path)


def test_rolling_folds_are_expanding_and_disjoint() -> None:
    sessions = pd.bdate_range("2025-01-01", periods=164).strftime("%Y-%m-%d")
    folds = rolling_folds(sessions)
    assert len(folds) == 5
    assert [len(fold["model_fit"]) for fold in folds] == [44, 68, 92, 116, 140]
    assert all(set(fold["model_fit"]).isdisjoint(fold["outer_test"]) for fold in folds)


def test_max_t_is_family_corrected_and_deterministic() -> None:
    contrasts = {
        "a": {f"s{i}": float(i % 3) for i in range(30)},
        "b": {f"s{i}": float((i + 1) % 4) for i in range(30)},
    }
    first = session_blocked_max_t(contrasts, permutations=999, seed=7)
    second = session_blocked_max_t(contrasts, permutations=999, seed=7)
    assert first == second
    assert first["a"]["maxT_p_one_sided"] >= first["a"]["p_one_sided"]


def test_direction_and_timing_screens_preserve_exact_pairs() -> None:
    target = "pnl_p4"
    rows = [
        {"session": "2025-01-02", "decision_time_ns": 0, "contract_id": "c", "right": "C", "offset": 5, "entry_ask": 4.0, "_pred": 30.0, target: 20.0},
        {"session": "2025-01-02", "decision_time_ns": 0, "contract_id": "p", "right": "P", "offset": -5, "entry_ask": 4.2, "_pred": 10.0, target: -5.0},
        {"session": "2025-01-02", "decision_time_ns": 300_000_000_000, "contract_id": "c", "right": "C", "offset": 5, "entry_ask": 4.1, "_pred": 0.0, target: 25.0},
    ]
    frame = pd.DataFrame(rows)
    deltas, pairs, _ = direction_screen(frame.iloc[:2], threshold=20.0, policy=REFERENCE_EXIT)
    assert len(pairs) == 1
    assert deltas["2025-01-02"] == 25.0
    intents = frame.iloc[:1]
    timing, timing_pairs, meta = timing_screen(
        frame, intents, delay_minutes=5, policy=REFERENCE_EXIT
    )
    assert len(timing_pairs) == 1
    assert timing["2025-01-02"] == 5.0
    assert meta["paired_coverage"] == 1.0


def test_corrected_development_and_confirmation_boundaries_are_disjoint() -> None:
    development = set(eligible_sessions())
    confirmation = set(_confirmation_sessions())
    assert len(development) == 214
    assert len(confirmation) == 29
    assert development.isdisjoint(confirmation)


def test_invalidated_integrated_entry_policy_cannot_compile_on_prose_twins() -> None:
    with pytest.raises(ExperimentCompileError, match="feature_without_executable_live_twin"):
        compile_hypothesis(hypothesis_payload("signed18_model_side_nearest"))


def test_confirmation_sign_flip_canary_detects_uniform_positive_effect() -> None:
    values = {f"session-{index}": 1.0 for index in range(29)}
    assert _sign_flip_p(values, permutations=9_999, seed=1) <= 0.001
