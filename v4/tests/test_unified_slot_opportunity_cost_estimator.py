from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v4.model.unified_slot_opportunity_cost_estimator import (
    SlotOpportunityCostEstimatorConfig,
    build_slot_cost_targets,
    sample_training_rows,
    validate_slot_cost_feature_columns,
)


def test_validate_slot_cost_feature_columns_rejects_label_and_exit_leaks() -> None:
    with pytest.raises(ValueError, match="feature leak"):
        validate_slot_cost_feature_columns(["entry_premium", "blocked_protocol101_pnl_0_00"])

    with pytest.raises(ValueError):
        validate_slot_cost_feature_columns(["entry_premium", "candidate_exit_dt"])


def test_build_slot_cost_targets_clips_negative_and_large_costs() -> None:
    frame = pd.DataFrame(
        {
            "blocked_protocol101_pnl_0_00": [-200.0, 150.0, 20_000.0],
            "blocked_protocol101_entries": [0, 2, 5],
        }
    )

    targets = build_slot_cost_targets(frame, SlotOpportunityCostEstimatorConfig(target_clip=1_000.0))

    assert targets["target_cost"].tolist() == [0.0, 150.0, 1_000.0]
    assert targets["target_positive_cost"].tolist() == [False, True, True]
    assert np.allclose(targets["target_log_cost"], np.log1p([0.0, 150.0, 1_000.0]))


def test_sample_training_rows_preserves_requested_positive_fraction() -> None:
    frame = pd.DataFrame(
        {
            "value": range(100),
            "target_positive_cost": [True] * 20 + [False] * 80,
        }
    )

    sample = sample_training_rows(
        frame,
        positive_column="target_positive_cost",
        limit=40,
        positive_fraction=0.50,
        seed=1,
    )

    assert len(sample) == 40
    assert int(sample["target_positive_cost"].sum()) == 20
