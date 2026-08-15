from __future__ import annotations

import numpy as np
import pandas as pd

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol165_full_action_space_policy import (
    FULL_ACTION_FEATURE_COLUMNS,
    FullActionPolicyConfig,
    add_oracle_actions,
    simulate_first_affordable,
    tensors,
)


def test_protocol165_oracle_waits_when_early_trade_blocks_better_later_winner() -> None:
    events = [
        _event("2026-01-02T14:00:00+00:00", "2026-01-02T14:30:00+00:00", pnl=100.0),
        _event("2026-01-02T14:10:00+00:00", "2026-01-02T14:20:00+00:00", pnl=500.0),
    ]

    add_oracle_actions(events)

    assert events[0]["oracle_action"] == 0
    assert events[1]["oracle_action"] == 1


def test_protocol165_oracle_takes_early_trade_when_it_beats_later_opportunity() -> None:
    events = [
        _event("2026-01-02T14:00:00+00:00", "2026-01-02T14:30:00+00:00", pnl=700.0),
        _event("2026-01-02T14:10:00+00:00", "2026-01-02T14:20:00+00:00", pnl=500.0),
    ]

    add_oracle_actions(events)

    assert events[0]["oracle_action"] == 1


def test_protocol165_oracle_waits_when_all_available_candidates_are_negative() -> None:
    events = [_event("2026-01-02T14:00:00+00:00", "2026-01-02T14:10:00+00:00", pnl=-50.0)]

    add_oracle_actions(events)

    assert events[0]["oracle_action"] == 0


def test_protocol165_tensors_mask_unaffordable_candidates() -> None:
    events = [_event("2026-01-02T14:00:00+00:00", "2026-01-02T14:10:00+00:00", pnl=500.0, ask=125.0)]
    events[0]["oracle_action"] = 1
    scaler = FeatureScaler.fit(events[0]["candidates"][FULL_ACTION_FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    _, mask, y, _, _ = tensors(events, scaler, FullActionPolicyConfig())

    assert mask[0, 0] == np.False_
    assert y[0] == 0


def test_protocol165_first_affordable_enforces_one_open_position_and_affordability() -> None:
    events = [
        _event("2026-01-02T14:00:00+00:00", "2026-01-02T14:10:00+00:00", pnl=500.0, ask=120.0),
        _event("2026-01-02T14:01:00+00:00", "2026-01-02T14:20:00+00:00", pnl=100.0, ask=10.0),
        _event("2026-01-02T14:05:00+00:00", "2026-01-02T14:25:00+00:00", pnl=1000.0, ask=10.0),
    ]

    result = simulate_first_affordable(events, slippage_per_side=0.0, starting_cash=10_000.0)

    assert result.summary["trades"] == 1
    assert result.summary["total_pnl"] == 100.0
    assert result.summary["skipped_unaffordable_candidates"] == 1
    assert result.summary["skipped_overlap_candidates"] == 1
    assert result.summary["max_concurrent_positions"] == 1


def _event(decision_time: str, exit_time: str, *, pnl: float, ask: float = 10.0) -> dict:
    row = {name: 0.0 for name in FULL_ACTION_FEATURE_COLUMNS}
    row.update(
        {
            "entry_ask": ask,
            "entry_bid": max(ask - 0.2, 0.01),
            "entry_mid": max(ask - 0.1, 0.01),
            "entry_spread": 0.2,
            "entry_bid_size": 10.0,
            "entry_ask_size": 12.0,
            "entry_premium": ask * 100.0,
            "entry_affordable_10k": float(ask * 100.0 <= 10_000.0),
        }
    )
    frame = pd.DataFrame(
        [
            {
                **row,
                "candidate_uid": f"candidate-{decision_time}",
                "trade_uid": f"trade-{decision_time}",
                "split": "unit",
                "session": "2026-01-02",
                "decision_dt": pd.Timestamp(decision_time),
                "candidate_exit_dt": pd.Timestamp(exit_time),
                "contract_id": "SPXW-20260102-04000.000-C",
                "right": "C",
                "offset": 0.0,
                "candidate_pnl": pnl,
                "candidate_exit_reason": "unit",
                "label_source": "unit",
            }
        ]
    )
    return {
        "split": "unit",
        "session": "2026-01-02",
        "decision_dt": pd.Timestamp(decision_time),
        "candidates": frame,
    }
