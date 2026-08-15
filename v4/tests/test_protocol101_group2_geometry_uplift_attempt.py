from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from v4.model.supervised_pilot import DecisionCandidates
from v4.scripts import run_protocol101_group2_geometry_uplift_attempt as uplift


def _decision() -> DecisionCandidates:
    return DecisionCandidates(
        session="2025-07-09",
        decision_time=datetime(2025, 7, 9, 13, 32, tzinfo=UTC),
        features=np.zeros((3, 71), dtype=np.float32),
        labels=np.asarray([1.0, -1.0, 2.0], dtype=np.float32),
        offsets=np.asarray([-50.0, 0.0, 10.0], dtype=np.float32),
        rights=np.asarray(["P", "C", "C"], dtype=object),
        market_last=np.asarray([5000.0, 20.0, 5001.0, 0.0, 40.0, 0.0, 0.0], dtype=np.float32),
        contract_ids=np.asarray(
            [
                "SPXW-20250709-04950.000-P",
                "SPXW-20250709-05000.000-C",
                "SPXW-20250709-05010.000-C",
            ],
            dtype=object,
        ),
        entry_asks=np.asarray([5.0, 10.0, 7.5], dtype=np.float32),
    )


def test_group2_feature_matrix_uses_static_ladder_counts_not_accepted_subset() -> None:
    matrix = uplift.group2_feature_matrix(_decision())
    by_name = {name: idx for idx, name in enumerate(uplift.GROUP2_FEATURE_NAMES)}

    assert matrix.shape == (3, len(uplift.GROUP2_FEATURE_NAMES))
    assert matrix[:, by_name["candidate_count_total"]].tolist() == [42.0, 42.0, 42.0]
    assert matrix[:, by_name["same_right_candidate_count"]].tolist() == [21.0, 21.0, 21.0]
    assert matrix[0, by_name["local_same_right_neighbor_count_10_points"]] == 3.0
    assert matrix[1, by_name["local_same_right_neighbor_count_10_points"]] == 5.0
    assert matrix[0, by_name["out_of_the_money_flag"]] == 1.0
    assert matrix[1, by_name["abs_offset_bucket_0_10"]] == 1.0


def test_append_group2_features_preserves_existing_columns_and_appends_only_group2() -> None:
    decision = _decision()
    augmented = uplift.append_group2_features([decision])[0]

    assert augmented.features.shape[0] == decision.features.shape[0]
    assert augmented.features.shape[1] == decision.features.shape[1] + len(uplift.GROUP2_FEATURE_NAMES)
    np.testing.assert_array_equal(augmented.features[:, : decision.features.shape[1]], decision.features)


class _PathCache:
    def diagnostics(self, **kwargs):
        return {
            "path_status": "ok",
            "exit_timestamp": "2025-07-09T13:42:00+00:00",
            "exit_bid": 8.0,
            "gross_pnl_recomputed": 50.0,
            "exit_reason": "max_hold",
            "forced_flat": False,
            "mfe": 75.0,
            "mae": -25.0,
            "max_favorable_timestamp": "2025-07-09T13:38:00+00:00",
            "max_adverse_timestamp": "2025-07-09T13:35:00+00:00",
            "exit_efficiency": 2 / 3,
            "profitable_before_ending_negative": False,
            "time_to_best_minutes": 6.0,
            "path_quote_count": 10,
        }


def test_diagnostic_simulation_persists_predictions_trades_and_path_rows() -> None:
    decision = _decision()
    config = uplift.config_for(
        0,
        seed=42,
        max_trades_per_session=1,
        args=argparse.Namespace(
            max_train_examples=100,
            epochs=1,
            learning_rate=0.05,
            weight_decay=0.01,
        ),
    )

    trades, predictions, paths, skipped = uplift.simulate_model_policy_diagnostics(
        [decision],
        [np.asarray([0.1, 0.2, 9.0], dtype=np.float32)],
        threshold=1.0,
        config=config,
        batch="primary",
        fold_id="expanding_fold_01",
        path_cache=_PathCache(),
        top_n=2,
    )

    assert len(predictions) == 2
    assert len(trades) == 1
    assert len(paths) == 1
    assert trades[0]["contract_id"] == "SPXW-20250709-05010.000-C"
    assert trades[0]["fee_adjusted_pnl"] == -1.0
    assert paths[0]["mfe"] == 75.0
    assert skipped["positive_untraded_decisions"] == 0


def test_smoke_report_fails_when_path_diagnostics_are_missing(tmp_path: Path) -> None:
    out = tmp_path / "attempt002"
    out.mkdir()
    pd.DataFrame([{"x": 1}]).to_csv(out / "fold_predictions.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(out / "fold_trades.csv", index=False)
    pd.DataFrame().to_csv(out / "path_diagnostics.csv", index=False)
    args = argparse.Namespace(
        out_dir=out,
        attempt_id="attempt002",
        max_folds=1,
        policy_indexes="0",
        seeds="42",
        batches="primary",
        smoke=True,
    )
    assembled = {
        "fold_predictions": {"path": str(out / "fold_predictions.csv"), "rows": 1},
        "fold_trades": {"path": str(out / "fold_trades.csv"), "rows": 1},
        "path_diagnostics": {"path": str(out / "path_diagnostics.csv"), "rows": 0},
    }

    smoke, estimate = uplift.write_smoke_and_runtime_reports(
        args,
        assembled=assembled,
        completed_units=1,
        elapsed_seconds=1.0,
    )

    assert smoke["status"] == "fail"
    assert estimate["estimated_full_units_primary_plus_conservative"] == 210
