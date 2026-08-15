from __future__ import annotations

import pandas as pd

from v5.ops.evaluate_causal_day_magnitude import evaluate_primary_cell


def _row(*, session: str, minute: str, score: float, net_mid: float, fold: int) -> dict:
    return {
        "session": session,
        "entry_minute": minute,
        "contract_id": f"{session}-{minute}",
        "architecture": "neural_four_head",
        "horizon_minutes": 120,
        "shuffled_label_null": False,
        "fold": fold,
        "predicted_depth_120m": score,
        "spread_usd": 10.0,
        "moneyness_itm_points": -5.0,
        "clock_exit_minute_120m": "15:00",
        "net_bid_120m_usd": net_mid - 10.0,
        "net_mid_120m_usd": net_mid,
    }


def test_primary_cell_stops_on_non_positive_mid_gross() -> None:
    trades, result = evaluate_primary_cell(
        pd.DataFrame([_row(session="2025-01-02", minute="10:00", score=31, net_mid=-3.08, fold=1)])
    )
    assert len(trades) == 1
    assert result["mean_gross_mid_to_mid_usd_per_trade"] == 0.0
    assert result["passed"] is False


def test_primary_cell_requires_a_trade_and_strictly_positive_gross() -> None:
    _, empty = evaluate_primary_cell(
        pd.DataFrame([_row(session="2025-01-02", minute="10:00", score=29, net_mid=100, fold=1)])
    )
    _, positive = evaluate_primary_cell(
        pd.DataFrame([_row(session="2025-01-02", minute="10:00", score=31, net_mid=0, fold=1)])
    )
    assert empty["trades"] == 0 and empty["passed"] is False
    assert positive["mean_gross_mid_to_mid_usd_per_trade"] == 3.08
    assert positive["passed"] is True
