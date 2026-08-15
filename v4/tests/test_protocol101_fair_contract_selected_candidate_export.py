"""Tests for fair-contract selected candidate reconstruction."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch

from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    LoadedFairModel,
    LoadedFairModelMember,
    SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
    build_waiting_payload,
    candidate_allowed_by_entry_filter,
    candidate_records_from_row,
    feature_hash,
    load_model,
    score_candidate_records,
    select_records_for_split,
    top_candidate_with_margin,
)
from v4.model.supervised_pilot import FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE
from v4.model.supervised_pilot import SELECTION_MODE_STABLE_ABS_OFFSET_15


def _row() -> dict:
    option_ladder = np.zeros((1, 2, 15), dtype=np.float32)
    option_ladder[0, 0, :] = np.arange(15, dtype=np.float32)
    option_ladder[0, 1, :] = np.arange(15, dtype=np.float32) + 1
    return {
        "decision_time": datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
        "source_quote_time": datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
        "source_context_time": datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        "feature_contract_version": "protocol101-live-v1",
        "candidate_mask": np.array([[True, False]]),
        "labels_net_pnl": np.array([[[125.0], [-50.0]]], dtype=np.float32),
        "contract_ids": np.array([["SPXW-20260102-06500.000-C", "SPXW-20260102-06500.000-P"]], dtype=object),
        "strike_offsets": np.array([0]),
        "rights": ("C", "P"),
        "option_ladder": option_ladder,
        "market_window": np.ones((30, 7), dtype=np.float32),
        "contract_quote_metadata": {
            "SPXW-20260102-06500.000-C": {
                "bid": 2.9,
                "ask": 3.1,
                "mid": 3.0,
                "quote_age_ms": 0.0,
            }
        },
    }


def test_candidate_records_from_row_are_reconstructable() -> None:
    records = candidate_records_from_row(
        _row(),
        session="2026-01-02",
        split="validation",
        policy_index=0,
    )

    assert len(records) == 1
    row = records[0]
    assert row["session"] == "2026-01-02"
    assert row["split"] == "validation"
    assert row["contract_id"] == "SPXW-20260102-06500.000-C"
    assert row["right"] == "C"
    assert row["label_net_pnl"] == 125.0
    assert row["entry_ask"] == 3.1
    assert row["source_context_time"] == "2026-01-02T14:31:00+00:00"
    assert isinstance(row["feature_hash"], str)
    assert len(row["feature_hash"]) == 64


def test_feature_hash_is_stable_for_same_vector() -> None:
    vector = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)

    assert feature_hash(vector) == feature_hash(vector.copy())


def test_waiting_payload_is_safe() -> None:
    payload = build_waiting_payload("training_result_missing")

    assert payload["status"] == "waiting_for_owner_approved_training_result"
    assert payload["implementation_version"] == SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION
    assert payload["model_training_executed_here"] is False
    assert payload["threshold_tuning_executed_here"] is False
    assert payload["broker_endpoint_called"] is False
    assert payload["paper_submit_allowed"] is False
    assert payload["blockers"] == ["training_result_missing"]


class _FakeRightEstimator:
    def __init__(self, positive_probability: float) -> None:
        self.classes_ = np.asarray([0, 1])
        self.positive_probability = float(positive_probability)

    def predict_proba(self, features):
        p = np.full(len(features), self.positive_probability, dtype=np.float32)
        return np.column_stack([1.0 - p, p])


class _FeatureRecordingRegressor:
    def __init__(self) -> None:
        self.seen = None

    def predict(self, features):
        self.seen = np.asarray(features, dtype=np.float32)
        return np.asarray([1.0] * len(features), dtype=np.float32)


def test_score_candidate_records_routes_to_right_specialists() -> None:
    records = [
        {"right": "C", "_features": np.asarray([1.0, 0.0], dtype=np.float32)},
        {"right": "P", "_features": np.asarray([0.0, 1.0], dtype=np.float32)},
    ]
    loaded = LoadedFairModel(
        model=None,
        scaler=None,
        target_mode="decision_top_profit_classifier",
        target_scale=1.0,
        threshold=0.0,
        policy_index=0,
        policy_name="test",
        cooldown_minutes=10,
        model_path="memory://model.pt",
        model_family="sklearn_hist_gradient_boosting_by_right",
        sklearn_models_by_right={
            "C": _FakeRightEstimator(0.25),
            "P": _FakeRightEstimator(0.75),
        },
    )

    score_candidate_records(records, loaded)

    assert records[0]["score"] == np.float32(0.25)
    assert records[1]["score"] == np.float32(0.75)


def test_score_candidate_records_applies_saved_feature_transform() -> None:
    estimator = _FeatureRecordingRegressor()
    records = [{"right": "P", "_features": np.arange(71, dtype=np.float32)}]
    loaded = LoadedFairModel(
        model=None,
        scaler=None,
        target_mode="regression",
        target_scale=100.0,
        threshold=0.0,
        policy_index=0,
        policy_name="test",
        cooldown_minutes=10,
        model_path="memory://model.pt",
        model_family="sklearn_hist_gradient_boosting",
        sklearn_model=estimator,
        feature_transform=FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    )

    score_candidate_records(records, loaded)

    assert estimator.seen is not None
    assert np.array_equal(estimator.seen[0, [3, 4, 5, 6, 7, 8, 9]], np.zeros(7, dtype=np.float32))
    assert records[0]["score"] == 100.0
    assert records[0]["model_feature_hash"]


def test_top_candidate_tie_break_matches_training_simulator() -> None:
    candidates = [
        {"contract_id": "first", "score": 1.0},
        {"contract_id": "last", "score": 1.0},
    ]

    selected, margin = top_candidate_with_margin(candidates)

    assert selected["contract_id"] == "last"
    assert margin == 0.0


def test_top_candidate_stable_offset_selection_ignores_top_score_for_contract_choice() -> None:
    candidates = [
        {"contract_id": "far_high_score", "score": 100.0, "offset": -20.0, "right": "P"},
        {"contract_id": "target_low_score", "score": 1.0, "offset": -15.0, "right": "P"},
        {"contract_id": "near_high_score", "score": 80.0, "offset": -10.0, "right": "P"},
    ]

    selected, margin = top_candidate_with_margin(
        candidates,
        selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_15,
    )

    assert selected["contract_id"] == "target_low_score"
    assert margin == -99.0


def test_load_model_preserves_policy_index_zero(tmp_path) -> None:
    model_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_family": "sklearn_hist_gradient_boosting",
            "config": {
                "target_mode": "profit_classifier",
                "target_scale": 1.0,
                "policy_index": 0,
                "policy_name": "ask_to_bid_stop35_target60_hold10m",
                "cooldown_minutes": 10,
            },
            "sklearn_model": _FakeRightEstimator(0.5),
        },
        model_path,
    )

    loaded = load_model({"model_out": str(model_path), "chosen_threshold": 0.25})

    assert loaded.policy_index == 0
    assert loaded.policy_name == "ask_to_bid_stop35_target60_hold10m"
    assert loaded.cooldown_minutes == 10


class _AlwaysEnterModel:
    policy_index = 0
    target_mode = "regression"
    threshold = -1.0
    policy_name = "ask_to_bid_stop35_target60_hold10m"
    cooldown_minutes = 10
    model_path = "memory://model.pt"
    entry_filter = "none"
    min_score_margin = 0.0
    max_score_ceiling = 0.0


def test_selected_records_include_cooldown_for_replay(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    rows = [_row()]

    def fake_load_rows(_path):
        return rows

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=_AlwaysEnterModel(),
    )

    assert summary["selected_entries"] == 1
    assert selected[0]["cooldown_minutes"] == 10


def test_selected_records_score_candidates_in_one_batch_per_split(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    first = _row()
    second = _row()
    second["decision_time"] = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    calls = []

    def fake_load_rows(_path):
        return [first, second]

    def fake_score(records, _loaded):
        calls.append(len(records))
        for record in records:
            record["score"] = 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=_AlwaysEnterModel(),
    )

    assert calls == [2]
    assert summary["score_batch_calls"] == 1
    assert summary["selected_entries"] == 2
    assert len(selected) == 2


def test_selected_records_apply_vwap_aligned_entry_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    row = _row()
    row["candidate_mask"] = np.array([[True, True]])
    row["market_window"] = np.ones((30, 7), dtype=np.float32)
    row["market_window"][-1, 0] = 100.0
    row["market_window"][-1, 2] = 99.0

    def fake_load_rows(_path):
        return [row]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 100.0 if record["right"] == "P" else 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "vwap_aligned"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert summary["entry_filter"] == "vwap_aligned"
    assert summary["entry_filter_blocks"] == 0
    assert selected[0]["right"] == "C"
    assert selected[0]["entry_filter"] == "vwap_aligned"


def test_selected_records_apply_premium_floor_entry_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    row = _row()
    row["candidate_mask"] = np.array([[True, True]])
    row["contract_quote_metadata"]["SPXW-20260102-06500.000-C"]["ask"] = 2.5
    row["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }

    def fake_load_rows(_path):
        return [row]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 100.0 if record["right"] == "C" else 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "premium_floor_3"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert selected[0]["right"] == "P"
    assert selected[0]["entry_filter"] == "premium_floor_3"


def test_selected_records_apply_account_affordability_before_top_selection(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    row = _row()
    row["candidate_mask"] = np.array([[True, True]])
    row["contract_quote_metadata"]["SPXW-20260102-06500.000-C"]["ask"] = 200.0
    row["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 0.9,
        "ask": 1.0,
        "mid": 0.95,
        "quote_age_ms": 0.0,
    }

    def fake_load_rows(_path):
        return [row]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 100.0 if record["right"] == "C" else 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=_AlwaysEnterModel(),
    )

    assert summary["selected_entries"] == 1
    assert summary["affordability_blocks"] == 0
    assert selected[0]["right"] == "P"
    assert selected[0]["entry_ask"] == 1.0


def test_selected_records_apply_near_offset_entry_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    row = _row()
    row["candidate_mask"] = np.array([[True, True]])
    row["strike_offsets"] = np.array([15])
    row["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }

    def fake_load_rows(_path):
        return [row]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 100.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "near_10_20_offset"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert selected[0]["offset"] == 15
    assert selected[0]["entry_filter"] == "near_10_20_offset"


def test_selected_records_apply_put_near_offset_entry_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    row = _row()
    row["candidate_mask"] = np.array([[True, True]])
    row["strike_offsets"] = np.array([15])
    row["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }

    def fake_load_rows(_path):
        return [row]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 100.0 if record["right"] == "C" else 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "put_near_10_20_offset"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert selected[0]["right"] == "P"
    assert selected[0]["offset"] == 15
    assert selected[0]["entry_filter"] == "put_near_10_20_offset"


def test_selected_records_apply_put_near_after_0940_entry_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    early = _row()
    early["decision_time"] = datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)
    early["candidate_mask"] = np.array([[True, True]])
    early["strike_offsets"] = np.array([15])
    early["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }
    later = _row()
    later["decision_time"] = datetime(2026, 1, 2, 14, 40, tzinfo=timezone.utc)
    later["candidate_mask"] = np.array([[True, True]])
    later["strike_offsets"] = np.array([15])
    later["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }

    def fake_load_rows(_path):
        return [early, later]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 100.0 if record["right"] == "C" else 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "put_near_after_0940"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert summary["entry_filter_blocks"] == 1
    assert selected[0]["decision_time"] == "2026-01-02T14:40:00+00:00"
    assert selected[0]["right"] == "P"
    assert selected[0]["offset"] == 15
    assert selected[0]["entry_filter"] == "put_near_after_0940"


def test_selected_records_apply_put_near_after_0940_vwap_gap_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    early = _row()
    early["decision_time"] = datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)
    early["candidate_mask"] = np.array([[True, True]])
    early["strike_offsets"] = np.array([15])
    early["market_window"] = np.ones((30, 7), dtype=np.float32)
    early["market_window"][-1, 0] = 105.0
    early["market_window"][-1, 2] = 100.0
    early["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }
    stable_gap = _row()
    stable_gap["decision_time"] = datetime(2026, 1, 2, 14, 40, tzinfo=timezone.utc)
    stable_gap["candidate_mask"] = np.array([[True, True]])
    stable_gap["strike_offsets"] = np.array([15])
    stable_gap["market_window"] = np.ones((30, 7), dtype=np.float32)
    stable_gap["market_window"][-1, 0] = 105.0
    stable_gap["market_window"][-1, 2] = 100.0
    stable_gap["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }
    too_far_above = _row()
    too_far_above["decision_time"] = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    too_far_above["candidate_mask"] = np.array([[True, True]])
    too_far_above["strike_offsets"] = np.array([15])
    too_far_above["market_window"] = np.ones((30, 7), dtype=np.float32)
    too_far_above["market_window"][-1, 0] = 115.0
    too_far_above["market_window"][-1, 2] = 100.0
    too_far_above["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }

    def fake_load_rows(_path):
        return [early, stable_gap, too_far_above]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 100.0 if record["right"] == "C" else 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "put_near_after_0940_vwap_m2_10"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert summary["entry_filter_blocks"] == 2
    assert selected[0]["decision_time"] == "2026-01-02T14:40:00+00:00"
    assert selected[0]["right"] == "P"
    assert selected[0]["entry_filter"] == "put_near_after_0940_vwap_m2_10"


def test_candidate_allowed_applies_compound_put_near_filters() -> None:
    row = _row()
    row["decision_time"] = datetime(2026, 1, 2, 16, 40, tzinfo=timezone.utc)
    row["market_window"] = np.ones((30, 7), dtype=np.float32)
    row["market_window"][-1, 0] = 101.0
    row["market_window"][-1, 2] = 100.0
    row["market_window"][-1, 3] = -1.0
    row["market_window"][-1, 4] = 35.0
    row["market_window"][-1, 6] = -2.0
    candidate = {"right": "P", "offset": 15.0, "entry_ask": 8.0}

    assert candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_omar_neg",
    )
    assert candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_range_20_45",
    )
    assert candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_near_vwap",
    )
    assert candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_premium_gte_7_5",
    )
    assert candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_mom15_nonpos",
    )
    assert candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos",
    )

    row["market_window"][-1, 3] = 1.0
    assert not candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_omar_neg",
    )
    assert candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos",
    )
    row["market_window"][-1, 6] = 2.0
    assert not candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_mom15_nonpos",
    )
    assert not candidate_allowed_by_entry_filter(
        candidate,
        row,
        "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos",
    )
    assert not candidate_allowed_by_entry_filter(
        candidate,
        row,
        "near_after_0940_vwap_m2_10_mom15_side",
    )
    assert candidate_allowed_by_entry_filter(
        {"right": "C", "offset": 15.0, "entry_ask": 6.0},
        row,
        "near_after_0940_vwap_m2_10_mom15_side",
    )
    assert not candidate_allowed_by_entry_filter(
        {"right": "C", "offset": 15.0, "entry_ask": 6.0},
        row,
        "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5",
    )


def test_selected_records_apply_score_ceiling_abstention(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    rows = [_row()]

    def fake_load_rows(_path):
        return rows

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 50.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class CeilingModel(_AlwaysEnterModel):
        max_score_ceiling = 50.0

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=CeilingModel(),
    )

    assert selected == []
    assert summary["selected_entries"] == 0
    assert summary["score_ceiling_waits"] == 1
    assert summary["max_score_ceiling"] == 50.0


def test_selected_records_apply_morning_entry_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    morning = _row()
    morning["decision_time"] = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)
    afternoon = _row()
    afternoon["decision_time"] = datetime(2026, 1, 2, 19, 5, tzinfo=timezone.utc)

    def fake_load_rows(_path):
        return [morning, afternoon]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "morning_1000_1129"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert selected[0]["decision_time"] == "2026-01-02T15:05:00+00:00"
    assert selected[0]["entry_filter"] == "morning_1000_1129"


def test_selected_records_apply_positive_context_entry_filter(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    row = _row()
    row["decision_time"] = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)
    row["market_window"] = np.ones((30, 7), dtype=np.float32)
    row["market_window"][-1, 0] = 100.0
    row["market_window"][-1, 2] = 99.0
    row["market_window"][-1, 3] = 1.0

    def fake_load_rows(_path):
        return [row]

    def fake_score(records, _loaded):
        for record in records:
            record["score"] = 5.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score,
    )

    class FilteredModel(_AlwaysEnterModel):
        entry_filter = "above_vwap_omar_pos_after_open"

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=FilteredModel(),
    )

    assert summary["selected_entries"] == 1
    assert selected[0]["entry_filter"] == "above_vwap_omar_pos_after_open"


def test_selected_records_apply_score_margin_abstention(monkeypatch, tmp_path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    row = _row()
    row["candidate_mask"] = np.array([[True, True]])
    row["contract_quote_metadata"]["SPXW-20260102-06500.000-P"] = {
        "bid": 3.1,
        "ask": 3.25,
        "mid": 3.175,
        "quote_age_ms": 0.0,
    }

    def fake_load_rows(_path):
        return [row]

    def fake_score_close(records, _loaded):
        for record in records:
            record["score"] = 10.0 if record["right"] == "C" else 7.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score_close,
    )

    class MarginModel(_AlwaysEnterModel):
        min_score_margin = 5.0

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=MarginModel(),
    )

    assert selected == []
    assert summary["score_margin_waits"] == 1

    def fake_score_wide(records, _loaded):
        for record in records:
            record["score"] = 10.0 if record["right"] == "C" else 3.0

    monkeypatch.setattr(
        "v4.scripts.run_protocol101_fair_contract_selected_candidate_export.score_candidate_records",
        fake_score_wide,
    )

    selected, summary = select_records_for_split(
        split="validation",
        paths=[path],
        loaded=MarginModel(),
    )

    assert summary["score_margin_waits"] == 0
    assert selected[0]["right"] == "C"
    assert selected[0]["min_score_margin"] == 5.0
    assert selected[0]["score_margin"] == 7.0


class _ZeroLogitModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0])


class _ConstantModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full((features.shape[0],), self.value)


class _IdentityScaler:
    def transform(self, features):
        return features


def test_profit_classifier_export_scores_are_probabilities() -> None:
    records = candidate_records_from_row(
        _row(),
        session="2026-01-02",
        split="validation",
        policy_index=0,
    )
    loaded = LoadedFairModel(
        model=_ZeroLogitModel(),
        scaler=_IdentityScaler(),
        target_mode="profit_classifier",
        target_scale=100.0,
        threshold=0.5,
        policy_index=0,
        policy_name="ask_to_bid_stop35_target60_hold10m",
        cooldown_minutes=10,
        model_path="memory://model.pt",
    )

    score_candidate_records(records, loaded)

    assert records[0]["score"] == 0.5


def test_ensemble_export_scores_are_member_average() -> None:
    records = candidate_records_from_row(
        _row(),
        session="2026-01-02",
        split="validation",
        policy_index=0,
    )
    loaded = LoadedFairModel(
        model=None,
        scaler=None,
        target_mode="regression",
        target_scale=100.0,
        threshold=0.0,
        policy_index=0,
        policy_name="ask_to_bid_stop35_target60_hold10m",
        cooldown_minutes=10,
        model_path="memory://ensemble.pt",
        ensemble_members=(
            LoadedFairModelMember(
                model=_ConstantModel(1.0),
                scaler=_IdentityScaler(),
                target_scale=10.0,
            ),
            LoadedFairModelMember(
                model=_ConstantModel(3.0),
                scaler=_IdentityScaler(),
                target_scale=20.0,
            ),
        ),
    )

    score_candidate_records(records, loaded)

    assert records[0]["score"] == 35.0
