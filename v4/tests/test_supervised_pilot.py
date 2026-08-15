"""Tests for the supervised pilot harness."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from v4.model.supervised_pilot import (
    DecisionCandidates,
    FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1,
    SELECTION_MODE_STABLE_ABS_OFFSET_15,
    FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    PilotConfig,
    apply_candidate_feature_transform,
    augment_decision_features_with_noise,
    candidate_feature_matrix_from_context,
    candidate_feature_vector,
    collect_training_examples,
    entry_filter_mask,
    metrics_for_trades,
    predict_decisions,
    simulate_baseline,
    simulate_model_policy,
    top_prediction,
    train_model,
    row_candidate_feature_context,
)


def _row() -> dict:
    option_ladder = np.full((2, 2, 15), np.nan, dtype=float)
    option_ladder[0, 0, :] = np.arange(15, dtype=float)
    return {
        "option_ladder": option_ladder,
        "market_window": np.ones((30, 7), dtype=float),
        "strike_offsets": np.array([-5, 0]),
        "rights": ("C", "P"),
        "decision_time": datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc),
    }


def test_candidate_feature_vector_is_stable_width() -> None:
    features = candidate_feature_vector(_row(), 0, 0)

    assert features.shape == (71,)
    assert np.allclose(features[43:47], [1.0, 0.0, -0.1, 0.1])
    assert np.isclose(features[-8], 5 / 360)


def test_candidate_feature_matrix_matches_single_candidate_vector() -> None:
    row = _row()
    context = row_candidate_feature_context(row)

    matrix = candidate_feature_matrix_from_context(
        row,
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        context,
    )

    np.testing.assert_allclose(matrix[0], candidate_feature_vector(row, 0, 0))


def test_feature_transform_masks_vendor_sensitive_option_microstructure_only() -> None:
    features = np.arange(71, dtype=np.float32)

    transformed = apply_candidate_feature_transform(
        features,
        FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    )

    masked = {3, 4, 5, 6, 7, 8, 9}
    for idx, value in enumerate(transformed):
        if idx in masked:
            assert value == 0.0
        else:
            assert value == features[idx]


def test_feature_transform_masks_vendor_sensitive_option_quote_greek_microstructure() -> None:
    features = np.arange(71, dtype=np.float32)

    transformed = apply_candidate_feature_transform(
        features,
        FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    )

    masked = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14}
    for idx, value in enumerate(transformed):
        if idx in masked:
            assert value == 0.0
        else:
            assert value == features[idx]


def test_feature_transform_buckets_vendor_sensitive_option_microstructure() -> None:
    features = np.arange(71, dtype=np.float32)
    features[3] = 0.11
    features[4] = 0.004
    features[5] = 2.0
    features[6] = 18.0
    features[7] = 1234.0
    features[8] = 5678.0
    features[9] = 0.1518

    transformed = apply_candidate_feature_transform(
        features,
        FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    )

    assert transformed[3] == np.float32(0.25)
    assert transformed[4] == np.float32(0.01)
    assert transformed[5] == np.float32(1.0)
    assert transformed[6] == np.float32(3.0)
    assert transformed[7] == 0.0
    assert transformed[8] == 0.0
    assert np.isclose(transformed[9], np.float32(0.15))
    assert transformed[0] == features[0]
    assert transformed[10] == features[10]


def test_feature_noise_augmentation_duplicates_fit_rows_with_microstructure_jitter() -> None:
    features = np.tile(np.arange(12, dtype=np.float32), (2, 1))
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
        features=features,
        labels=np.array([10.0, -5.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0], dtype=np.float32),
        rights=np.array(["P", "P"], dtype=object),
        market_last=np.zeros(7, dtype=np.float32),
    )

    augmented = augment_decision_features_with_noise(
        [decision],
        FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1,
    )

    assert len(augmented) == 7
    np.testing.assert_allclose(augmented[0].features, features)
    assert augmented[0].labels is decision.labels
    assert augmented[1].features[0, 9] == np.float32(features[0, 9] + 0.002)
    assert augmented[2].features[0, 9] == np.float32(features[0, 9] - 0.002)
    assert augmented[3].features[0, 3] == np.float32(features[0, 3] + 0.05)
    assert augmented[4].features[0, 3] == np.float32(features[0, 3] - 0.05)
    assert augmented[5].features[0, 5] == np.float32(features[0, 5] * 0.5)
    assert augmented[6].features[0, 6] == np.float32(features[0, 6] * 0.5)


def test_baseline_simulation_enforces_cooldown() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31 + i, tzinfo=timezone.utc),
            features=np.zeros((1, 71), dtype=np.float32),
            labels=np.array([10.0], dtype=np.float32),
            offsets=np.array([0.0], dtype=np.float32),
            rights=np.array(["C"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        )
        for i in range(3)
    ]

    trades = simulate_baseline(decisions, kind="atm_call", cooldown_minutes=25)
    metrics = metrics_for_trades(trades)

    assert metrics["trades"] == 1
    assert metrics["total_pnl"] == 10.0


def test_decision_relative_training_examples_center_each_decision() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
            features=np.zeros((3, 71), dtype=np.float32),
            labels=np.array([-100.0, 0.0, 200.0], dtype=np.float32),
            offsets=np.array([-5.0, 0.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        )
    ]
    config = PilotConfig(target_mode="decision_relative_regression", target_scale=100.0)

    _features, targets = collect_training_examples(decisions, config=config)

    assert np.allclose(targets, np.array([-1.0, 0.0, 2.0], dtype=np.float32))


def test_blended_relative_training_examples_mix_absolute_and_relative_targets() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
            features=np.zeros((3, 71), dtype=np.float32),
            labels=np.array([0.0, 100.0, 300.0], dtype=np.float32),
            offsets=np.array([-5.0, 0.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        )
    ]
    config = PilotConfig(
        target_mode="blended_relative_regression",
        target_scale=100.0,
        relative_target_weight=0.5,
    )

    _features, targets = collect_training_examples(decisions, config=config)

    assert np.allclose(targets, np.array([-0.5, 0.5, 2.5], dtype=np.float32))


def test_decision_top_profit_classifier_marks_only_best_profitable_candidate() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
            features=np.zeros((3, 71), dtype=np.float32),
            labels=np.array([25.0, 100.0, 80.0], dtype=np.float32),
            offsets=np.array([-5.0, 0.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        ),
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
            features=np.zeros((2, 71), dtype=np.float32),
            labels=np.array([-10.0, 10.0], dtype=np.float32),
            offsets=np.array([0.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        ),
    ]
    config = PilotConfig(
        target_mode="decision_top_profit_classifier",
        positive_label_threshold=20.0,
    )

    _features, targets = collect_training_examples(decisions, config=config)

    assert np.array_equal(targets, np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_decision_aggregate_targets_mark_whole_profitable_minute() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
            features=np.zeros((3, 71), dtype=np.float32),
            labels=np.array([25.0, 300.0, 80.0], dtype=np.float32),
            offsets=np.array([-5.0, 0.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        ),
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
            features=np.zeros((2, 71), dtype=np.float32),
            labels=np.array([-40.0, 10.0], dtype=np.float32),
            offsets=np.array([0.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        ),
    ]

    _features, classifier_targets = collect_training_examples(
        decisions,
        config=PilotConfig(
            target_mode="decision_profit_presence_classifier",
            positive_label_threshold=20.0,
        ),
    )
    _features, regression_targets = collect_training_examples(
        decisions,
        config=PilotConfig(
            target_mode="decision_best_profit_regression",
            target_scale=100.0,
        ),
    )

    assert np.array_equal(
        classifier_targets,
        np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.allclose(
        regression_targets,
        np.array([3.0, 3.0, 3.0, 0.1, 0.1], dtype=np.float32),
    )


def test_protocol101_teacher_profitable_classifier_uses_teacher_and_fair_label() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
            features=np.zeros((4, 71), dtype=np.float32),
            labels=np.array([100.0, -50.0, 200.0, 10.0], dtype=np.float32),
            offsets=np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32),
            rights=np.array(["C", "C", "P", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
            teacher_labels=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
    ]
    config = PilotConfig(
        target_mode="protocol101_teacher_profitable_classifier",
        positive_label_threshold=20.0,
    )

    _features, targets = collect_training_examples(decisions, config=config)

    assert np.array_equal(targets, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_protocol101_teacher_edge_regression_scores_teacher_fair_edge_only() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
            features=np.zeros((4, 71), dtype=np.float32),
            labels=np.array([100.0, -50.0, 200.0, 10.0], dtype=np.float32),
            offsets=np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32),
            rights=np.array(["C", "C", "P", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
            teacher_labels=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
    ]
    config = PilotConfig(
        target_mode="protocol101_teacher_edge_regression",
        positive_label_threshold=20.0,
        target_scale=100.0,
    )

    _features, targets = collect_training_examples(decisions, config=config)

    assert np.allclose(targets, np.array([0.8, -0.7, 0.0, -0.1], dtype=np.float32))


def test_decision_top_profit_regression_marks_best_profitable_candidate_value() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
            features=np.zeros((3, 71), dtype=np.float32),
            labels=np.array([25.0, 300.0, 80.0], dtype=np.float32),
            offsets=np.array([-5.0, 0.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        )
    ]
    config = PilotConfig(
        target_mode="decision_top_profit_regression",
        target_scale=100.0,
        positive_label_threshold=20.0,
    )

    _features, targets = collect_training_examples(decisions, config=config)

    assert np.array_equal(targets, np.array([0.0, 3.0, 0.0], dtype=np.float32))


def test_decision_top_profit_listwise_can_rank_within_decision() -> None:
    train_decisions = []
    validation_decisions = []
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    for idx in range(12):
        features = np.zeros((2, 8), dtype=np.float32)
        features[0, 0] = -1.0
        features[1, 0] = 1.0
        decision = DecisionCandidates(
            session="2026-01-02",
            decision_time=base_time,
            features=features,
            labels=np.array([-25.0, 125.0], dtype=np.float32),
            offsets=np.array([-5.0, 5.0], dtype=np.float32),
            rights=np.array(["C", "P"], dtype=object),
            market_last=np.array([1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        )
        if idx < 8:
            train_decisions.append(decision)
        else:
            validation_decisions.append(decision)
    config = PilotConfig(
        target_mode="decision_top_profit_listwise",
        hidden_dim=12,
        epochs=8,
        batch_size=128,
        positive_label_threshold=20.0,
        seed=7,
    )

    model, scaler, _history = train_model(train_decisions, validation_decisions, config=config)
    predictions = predict_decisions(
        model,
        scaler,
        validation_decisions[:1],
        target_scale=1.0,
        prediction_transform="identity_unit",
    )

    assert int(np.argmax(predictions[0])) == 1


def test_vwap_aligned_entry_filter_limits_policy_side() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.zeros((2, 71), dtype=np.float32),
        labels=np.array([40.0, 500.0], dtype=np.float32),
        offsets=np.array([0.0, 0.0], dtype=np.float32),
        rights=np.array(["C", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert np.array_equal(entry_filter_mask(decision, "vwap_aligned"), np.array([True, False]))

    trades = simulate_model_policy(
        [decision],
        [np.array([10.0, 100.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="filtered",
        entry_filter="vwap_aligned",
    )

    assert len(trades) == 1
    assert trades[0].right == "C"
    assert trades[0].pnl == 40.0


def test_premium_floor_entry_filter_blocks_low_ask_candidates() -> None:
    features = np.zeros((2, 71), dtype=np.float32)
    features[:, 1] = np.array([2.5, 3.25], dtype=np.float32)
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=features,
        labels=np.array([500.0, 30.0], dtype=np.float32),
        offsets=np.array([0.0, 5.0], dtype=np.float32),
        rights=np.array(["C", "C"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert np.array_equal(entry_filter_mask(decision, "premium_floor_3"), np.array([False, True]))

    trades = simulate_model_policy(
        [decision],
        [np.array([100.0, 5.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="filtered",
        entry_filter="premium_floor_3",
    )

    assert len(trades) == 1
    assert trades[0].offset == 5.0
    assert trades[0].pnl == 30.0


def test_near_offset_entry_filter_limits_candidate_geometry() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([0.0, 10.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "C", "P", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert np.array_equal(
        entry_filter_mask(decision, "near_10_20_offset"),
        np.array([False, True, True, False]),
    )

    trades = simulate_model_policy(
        [decision],
        [np.array([100.0, 8.0, 7.0, 99.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="filtered",
        entry_filter="near_10_20_offset",
    )

    assert len(trades) == 1
    assert trades[0].offset == 10.0


def test_put_near_offset_entry_filter_limits_side_and_geometry() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert np.array_equal(
        entry_filter_mask(decision, "put_near_10_20_offset"),
        np.array([False, True, True, False]),
    )

    trades = simulate_model_policy(
        [decision],
        [np.array([100.0, 8.0, 7.0, 99.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="filtered",
        entry_filter="put_near_10_20_offset",
    )

    assert len(trades) == 1
    assert trades[0].right == "P"
    assert trades[0].offset == 15.0


def test_put_near_after_0940_filter_requires_causal_warmup_side_and_geometry() -> None:
    open_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    later_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 40, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert not entry_filter_mask(open_decision, "put_near_after_0940").any()
    assert np.array_equal(
        entry_filter_mask(later_decision, "put_near_after_0940"),
        np.array([False, True, True, False]),
    )

    trades = simulate_model_policy(
        [open_decision, later_decision],
        [
            np.array([100.0, 8.0, 7.0, 99.0], dtype=np.float32),
            np.array([100.0, 8.0, 7.0, 99.0], dtype=np.float32),
        ],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="filtered",
        entry_filter="put_near_after_0940",
    )

    assert len(trades) == 1
    assert trades[0].decision_time == "2026-01-02T14:40:00+00:00"
    assert trades[0].right == "P"
    assert trades[0].offset == 15.0


def test_put_near_after_0940_vwap_gap_filter_requires_stable_gap_bucket() -> None:
    early_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 104.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    stable_gap_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 40, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([105.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    too_far_above_vwap = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 41, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([115.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert not entry_filter_mask(early_decision, "put_near_after_0940_vwap_m2_10").any()
    assert np.array_equal(
        entry_filter_mask(stable_gap_decision, "put_near_after_0940_vwap_m2_10"),
        np.array([False, True, True, False]),
    )
    assert not entry_filter_mask(too_far_above_vwap, "put_near_after_0940_vwap_m2_10").any()


def test_put_near_after_0940_vwap_gap_compound_filters() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 16, 40, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([500.0, 80.0, 70.0, 600.0], dtype=np.float32),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([101.0, 0.0, 100.0, -1.0, 35.0, 0.0, -2.0], dtype=np.float32),
        entry_asks=np.array([6.0, 8.0, 16.0, 20.0], dtype=np.float32),
    )

    expected = np.array([False, True, True, False])
    assert np.array_equal(
        entry_filter_mask(decision, "put_near_after_0940_vwap_m2_10_omar_neg"),
        expected,
    )
    assert np.array_equal(
        entry_filter_mask(decision, "put_near_after_0940_vwap_m2_10_range_20_45"),
        expected,
    )
    assert np.array_equal(
        entry_filter_mask(decision, "put_near_after_0940_vwap_m2_10_near_vwap"),
        expected,
    )
    assert np.array_equal(
        entry_filter_mask(decision, "put_near_after_0940_vwap_m2_10_premium_gte_7_5"),
        expected,
    )
    assert np.array_equal(
        entry_filter_mask(decision, "put_near_after_0940_vwap_m2_10_mom15_nonpos"),
        expected,
    )
    assert np.array_equal(
        entry_filter_mask(decision, "put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos"),
        expected,
    )
    assert np.array_equal(
        entry_filter_mask(decision, "near_after_0940_vwap_m2_10_mom15_side"),
        expected,
    )
    assert np.array_equal(
        entry_filter_mask(decision, "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5"),
        expected,
    )

    omar_positive = DecisionCandidates(
        session=decision.session,
        decision_time=decision.decision_time,
        features=decision.features,
        labels=decision.labels,
        offsets=decision.offsets,
        rights=decision.rights,
        market_last=np.array([101.0, 0.0, 100.0, 1.0, 35.0, 0.0, -2.0], dtype=np.float32),
        entry_asks=decision.entry_asks,
    )

    assert not entry_filter_mask(
        omar_positive,
        "put_near_after_0940_vwap_m2_10_omar_neg",
    ).any()
    assert np.array_equal(
        entry_filter_mask(omar_positive, "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos"),
        expected,
    )

    momentum_positive = DecisionCandidates(
        session=decision.session,
        decision_time=decision.decision_time,
        features=decision.features,
        labels=decision.labels,
        offsets=decision.offsets,
        rights=decision.rights,
        market_last=np.array([101.0, 0.0, 100.0, 1.0, 35.0, 0.0, 2.0], dtype=np.float32),
        entry_asks=decision.entry_asks,
    )
    assert not entry_filter_mask(
        momentum_positive,
        "put_near_after_0940_vwap_m2_10_mom15_nonpos",
    ).any()
    assert not entry_filter_mask(
        momentum_positive,
        "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos",
    ).any()
    assert np.array_equal(
        entry_filter_mask(momentum_positive, "near_after_0940_vwap_m2_10_mom15_side"),
        np.array([True, False, False, False]),
    )
    assert not entry_filter_mask(
        momentum_positive,
        "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5",
    ).any()


def test_morning_entry_filter_limits_decision_time() -> None:
    morning_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc),
        features=np.zeros((1, 71), dtype=np.float32),
        labels=np.array([40.0], dtype=np.float32),
        offsets=np.array([0.0], dtype=np.float32),
        rights=np.array(["C"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    afternoon_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 19, 5, tzinfo=timezone.utc),
        features=np.zeros((1, 71), dtype=np.float32),
        labels=np.array([40.0], dtype=np.float32),
        offsets=np.array([0.0], dtype=np.float32),
        rights=np.array(["C"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert entry_filter_mask(morning_decision, "morning_1000_1129").all()
    assert not entry_filter_mask(afternoon_decision, "morning_1000_1129").any()


def test_positive_context_entry_filter_requires_after_open_above_vwap_and_positive_omar() -> None:
    open_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 45, tzinfo=timezone.utc),
        features=np.zeros((2, 71), dtype=np.float32),
        labels=np.array([40.0, 20.0], dtype=np.float32),
        offsets=np.array([0.0, 5.0], dtype=np.float32),
        rights=np.array(["C", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    later_decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc),
        features=np.zeros((2, 71), dtype=np.float32),
        labels=np.array([40.0, 20.0], dtype=np.float32),
        offsets=np.array([0.0, 5.0], dtype=np.float32),
        rights=np.array(["C", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert not entry_filter_mask(open_decision, "above_vwap_omar_pos_after_open").any()
    assert entry_filter_mask(later_decision, "above_vwap_omar_pos_after_open").all()


def test_score_margin_abstention_requires_clear_winner() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.zeros((2, 71), dtype=np.float32),
        labels=np.array([40.0, 20.0], dtype=np.float32),
        offsets=np.array([0.0, 5.0], dtype=np.float32),
        rights=np.array(["C", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    top = top_prediction(decision, np.array([10.0, 7.0], dtype=np.float32))

    assert top == (0, 10.0, 3.0)
    assert simulate_model_policy(
        [decision],
        [np.array([10.0, 7.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="margin",
        min_score_margin=5.0,
    ) == []
    trades = simulate_model_policy(
        [decision],
        [np.array([10.0, 4.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="margin",
        min_score_margin=5.0,
    )
    assert len(trades) == 1


def test_stable_offset_selection_chooses_contract_geometry_not_top_score() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc),
        features=np.zeros((4, 71), dtype=np.float32),
        labels=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        offsets=np.array([-20.0, -15.0, -10.0, 15.0], dtype=np.float32),
        rights=np.array(["P", "P", "P", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    scores = np.array([100.0, 1.0, 90.0, 80.0], dtype=np.float32)

    top = top_prediction(
        decision,
        scores,
        selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_15,
    )

    assert top == (1, 1.0, -99.0)
    trades = simulate_model_policy(
        [decision],
        [scores],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="stable_offset",
        selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_15,
    )
    assert len(trades) == 1
    assert trades[0].offset == -15.0
    assert trades[0].pnl == 20.0
    assert trades[0].right == "P"


def test_score_ceiling_abstention_blocks_overconfident_top_score() -> None:
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc),
        features=np.zeros((2, 71), dtype=np.float32),
        labels=np.array([40.0, 20.0], dtype=np.float32),
        offsets=np.array([0.0, 5.0], dtype=np.float32),
        rights=np.array(["C", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert simulate_model_policy(
        [decision],
        [np.array([50.0, 7.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="score_ceiling",
        max_score_ceiling=50.0,
    ) == []
    trades = simulate_model_policy(
        [decision],
        [np.array([49.999, 7.0], dtype=np.float32)],
        threshold=0.0,
        cooldown_minutes=10,
        strategy="score_ceiling",
        max_score_ceiling=50.0,
    )

    assert len(trades) == 1
    assert trades[0].score == float(np.float32(49.999))


def test_model_policy_enforces_session_trade_cap_and_daily_loss_stop() -> None:
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=base_time + timedelta(minutes=idx * 15),
            features=np.zeros((1, 71), dtype=np.float32),
            labels=np.array([pnl], dtype=np.float32),
            offsets=np.array([0.0], dtype=np.float32),
            rights=np.array(["C"], dtype=object),
            market_last=np.array([100.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        for idx, pnl in enumerate([-600.0, 900.0, 900.0])
    ]
    predictions = [np.array([1.0], dtype=np.float32) for _ in decisions]

    capped = simulate_model_policy(
        decisions,
        predictions,
        threshold=0.0,
        cooldown_minutes=1,
        strategy="risk_controlled",
        max_trades_per_session=2,
    )
    loss_stopped = simulate_model_policy(
        decisions,
        predictions,
        threshold=0.0,
        cooldown_minutes=1,
        strategy="risk_controlled",
        max_daily_loss=500.0,
    )

    assert [trade.pnl for trade in capped] == [-600.0, 900.0]
    assert [trade.pnl for trade in loss_stopped] == [-600.0]
