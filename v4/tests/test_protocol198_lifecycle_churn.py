from __future__ import annotations

import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import (
    build_hold_counterfactuals,
    build_same_side_reentry_chains,
    classify_churn,
    normalize_trade_frame,
    summarize_horizons,
)


def test_same_side_reentry_chain_ignores_opposite_side() -> None:
    trades = normalize_trade_frame(
        pd.DataFrame(
            [
                _trade("2026-05-20T14:00:00+00:00", "2026-05-20T14:05:00+00:00", "P1", "P", 50),
                _trade("2026-05-20T14:10:00+00:00", "2026-05-20T14:15:00+00:00", "P2", "P", 25),
                _trade("2026-05-20T14:18:00+00:00", "2026-05-20T14:25:00+00:00", "C1", "C", 100),
            ]
        )
    )

    chains = build_same_side_reentry_chains(trades, gap_minutes=10)

    assert len(chains) == 1
    assert chains[0].right == "P"
    assert len(chains[0].trade_indices) == 2


def test_horizon_summary_counts_only_close_reentries() -> None:
    trades = normalize_trade_frame(
        pd.DataFrame(
            [
                _trade("2026-05-20T14:00:00+00:00", "2026-05-20T14:05:00+00:00", "P1", "P", 50),
                _trade("2026-05-20T14:11:00+00:00", "2026-05-20T14:15:00+00:00", "P2", "P", 25),
            ]
        )
    )

    summary = summarize_horizons(trades, [5, 10])

    assert summary[0]["gap_minutes"] == 5
    assert summary[0]["chains"] == 0
    assert summary[1]["gap_minutes"] == 10
    assert summary[1]["chains"] == 1


def test_hold_counterfactual_uses_first_contract_ask_and_final_bid(tmp_path) -> None:
    trades = normalize_trade_frame(
        pd.DataFrame(
            [
                _trade("2026-05-20T14:00:00+00:00", "2026-05-20T14:05:00+00:00", "P1", "P", 50, entry_ask=10.0),
                _trade("2026-05-20T14:10:00+00:00", "2026-05-20T14:20:00+00:00", "P2", "P", 100, entry_ask=11.0),
            ]
        )
    )
    session_path = tmp_path / "databento_spxw_0dte_2026-05-20_official_context.parquet"
    pd.DataFrame(
        {
            "quote_time": pd.to_datetime(
                [
                    "2026-05-20T14:00:00+00:00",
                    "2026-05-20T14:10:00+00:00",
                    "2026-05-20T14:20:00+00:00",
                ],
                utc=True,
            ),
            "contract_id": ["P1", "P1", "P1"],
            "bid": [9.8, 11.0, 14.0],
            "ask": [10.0, 11.2, 14.2],
            "underlying_price": [7400.0, 7390.0, 7375.0],
        }
    ).to_parquet(session_path, index=False)
    chains = build_same_side_reentry_chains(trades, gap_minutes=30)

    rows, skips = build_hold_counterfactuals(trades, chains, normalized_dir=tmp_path)

    assert skips == []
    assert len(rows) == 1
    assert rows[0]["actual_sequence_pnl"] == 150.0
    assert rows[0]["counterfactual_hold_pnl"] == 400.0
    assert rows[0]["hold_minus_sequence_pnl"] == 250.0
    assert rows[0]["classification"] == "counterfactual_hold_better"
    assert rows[0]["directional_underlying_move"] == 25.0


def test_classify_churn() -> None:
    assert classify_churn(200.0, 100.0) == "counterfactual_hold_better"
    assert classify_churn(50.0, 100.0) == "exit_reentry_sequence_better"
    assert classify_churn(100.0, 100.0) == "neutral"


def _trade(
    decision_time: str,
    exit_time: str,
    contract_id: str,
    right: str,
    pnl: float,
    *,
    entry_ask: float = 10.0,
) -> dict:
    return {
        "fold": "fold1",
        "seed": 1,
        "reported_split": "recent_2026",
        "session": "2026-05-20",
        "decision_time": decision_time,
        "candidate_exit_time": exit_time,
        "contract_id": contract_id,
        "right": right,
        "pnl": pnl,
        "entry_ask": entry_ask,
        "candidate_exit_reason": "model_exit",
    }
