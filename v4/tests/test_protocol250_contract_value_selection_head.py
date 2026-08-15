from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts import run_protocol249_entry_quality_calibrator as p249
from v4.scripts import run_protocol250_contract_value_selection_head as p250


def _base() -> p249.BaseArtifact:
    return p249.BaseArtifact(
        artifact_dir=Path("unused"),
        feature_columns=[],
        threshold=0.0,
        model=None,
        scaler=None,
    )


def test_value_tensors_encode_contract_value_targets() -> None:
    frame = pd.DataFrame(
        {
            "entry_minutes_since_open": [10.0, 20.0],
            "candidate_pnl": [500.0, -200.0],
            "entry_premium": [1000.0, 1000.0],
            "candidate_exit_reason": ["target", "hard_stop"],
        }
    )
    scaler = FeatureScaler.fit(frame[["entry_minutes_since_open"]].to_numpy(dtype=np.float32))

    _, targets = p250.value_tensors(
        frame,
        scaler,
        ["entry_minutes_since_open"],
        p250.ContractValueConfig(),
    )

    assert targets["value"][0] > 0.0
    assert targets["value"][1] < 0.0
    assert targets["target_flag"].tolist() == [1.0, 0.0]
    assert targets["stop_flag"].tolist() == [0.0, 1.0]


def test_contract_value_selection_can_replace_base_contract_choice() -> None:
    decision_ts = pd.Timestamp("2025-07-01 14:00:00", tz="UTC")
    candidates = pd.DataFrame(
        [
            {
                "candidate_uid": "bad",
                "trade_uid": "bad",
                "split": "q3_2025",
                "session": "2025-07-01",
                "decision_dt": decision_ts,
                "candidate_exit_dt": pd.Timestamp("2025-07-01 14:10:00", tz="UTC"),
                "contract_id": "SPXW-bad",
                "right": "P",
                "offset": -5.0,
                "entry_ask": 10.0,
                "entry_affordable_10k": 1.0,
                "candidate_pnl": -100.0,
                "candidate_exit_reason": "hard_stop",
                "label_source": "synthetic",
            },
            {
                "candidate_uid": "good",
                "trade_uid": "good",
                "split": "q3_2025",
                "session": "2025-07-01",
                "decision_dt": decision_ts,
                "candidate_exit_dt": pd.Timestamp("2025-07-01 14:10:00", tz="UTC"),
                "contract_id": "SPXW-good",
                "right": "P",
                "offset": -10.0,
                "entry_ask": 9.0,
                "entry_affordable_10k": 1.0,
                "candidate_pnl": 300.0,
                "candidate_exit_reason": "target",
                "label_source": "synthetic",
            },
        ]
    )
    event = {"split": "q3_2025", "session": "2025-07-01", "decision_dt": decision_ts, "candidates": candidates}

    result = p250.simulate_value_selection(
        [event],
        [{"action": 1, "score": 1.0, "valid_action": True}],
        [np.asarray([-1.0, 2.0])],
        _base(),
        value_threshold=0.0,
        slippage_per_side=0.0,
        starting_cash=10_000.0,
        strategy="synthetic_contract_value",
    )

    assert result.summary["trades"] == 1
    assert result.trades[0]["trade_uid"] == "good"
    assert result.summary["total_pnl"] == 300.0
