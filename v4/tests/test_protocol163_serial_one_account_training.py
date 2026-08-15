from __future__ import annotations

import pandas as pd
import pytest

from v4.scripts.run_protocol163_serial_one_account_training import (
    recent_candidates_to_dataset,
    strict_one_account_baseline,
)


def test_recent_candidates_to_dataset_attaches_sizes_and_affordability(tmp_path):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pd.DataFrame(
        [
            {
                "contract_id": "SPXW-20260520-07350.000-P",
                "quote_time": "2026-05-20T14:01:00+00:00",
                "bid_size": 4,
                "ask_size": 7,
            }
        ]
    ).to_parquet(normalized / "databento_spxw_0dte_2026-05-20_official_context.parquet", index=False)
    candidates = pd.DataFrame(
        [
            {
                "path_status": "ok",
                "trade_uid": "trade1",
                "canonical_entry_uid": "entry1",
                "session": "2026-05-20",
                "decision_time": "2026-05-20T14:01:00+00:00",
                "contract_id": "SPXW-20260520-07350.000-P",
                "right": "P",
                "offset": 10.0,
                "edge": 31.0,
                "entry_quote_time": "2026-05-20T14:01:00+00:00",
                "entry_bid": 20.0,
                "entry_ask": 20.4,
                "entry_mid": 20.2,
                "entry_spread": 0.4,
                "entry_underlying_price": 7340.0,
                "entry_iv": 0.25,
                "entry_delta": -0.55,
                "entry_gamma": 0.01,
                "entry_theta": -18.0,
                "candidate_pnl": 150.0,
                "candidate_exit_time": "2026-05-20T14:26:00+00:00",
                "candidate_exit_reason": "mandatory_time_flat",
                "candidate_exit_step": 24,
            }
        ]
    )

    dataset, audit = recent_candidates_to_dataset(
        candidates,
        normalized_dir=normalized,
        seeds=[1],
        starting_cash=10_000.0,
    )

    assert audit["missing_entry_sizes_after_join"] == 0
    assert float(dataset["entry_bid_size"].iloc[0]) == 4.0
    assert float(dataset["entry_ask_size"].iloc[0]) == 7.0
    assert float(dataset["entry_premium"].iloc[0]) == pytest.approx(2040.0)
    assert float(dataset["entry_affordable_10k"].iloc[0]) == 1.0


def test_strict_one_account_baseline_enforces_affordability_and_overlap():
    events = [
        _event("2026-05-20T14:00:00+00:00", "2026-05-20T14:10:00+00:00", ask=120.0, pnl=500.0),
        _event("2026-05-20T14:01:00+00:00", "2026-05-20T14:20:00+00:00", ask=10.0, pnl=100.0),
        _event("2026-05-20T14:05:00+00:00", "2026-05-20T14:25:00+00:00", ask=10.0, pnl=1000.0),
    ]

    result = strict_one_account_baseline(events, seed=1, slippage_per_side=0.0, starting_cash=10_000.0)

    assert result.summary["trades"] == 1
    assert result.summary["total_pnl"] == 100.0
    assert result.summary["skipped_unaffordable_candidates"] == 1
    assert result.summary["skipped_overlap_candidates"] == 1


def _event(decision_time: str, exit_time: str, *, ask: float, pnl: float) -> dict:
    frame = pd.DataFrame(
        [
            {
                "candidate_uid": f"candidate-{decision_time}",
                "trade_uid": f"trade-{decision_time}",
                "split": "recent_2026",
                "seed": 1,
                "entry_seed": 1,
                "session": "2026-05-20",
                "decision_dt": pd.Timestamp(decision_time),
                "candidate_exit_dt": pd.Timestamp(exit_time),
                "contract_id": "SPXW-20260520-07350.000-P",
                "right": "P",
                "offset": 10.0,
                "candidate_pnl": pnl,
                "entry_ask": ask,
                "candidate_exit_reason": "mandatory_time_flat",
                "label_source": "test",
            }
        ]
    )
    return {
        "split": "recent_2026",
        "seed": 1,
        "session": "2026-05-20",
        "decision_time": decision_time,
        "decision_dt": pd.Timestamp(decision_time),
        "candidates": frame,
    }
