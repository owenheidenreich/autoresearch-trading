"""Tests for broad data purchase signal helpers."""
from __future__ import annotations

from v4.scripts.evaluate_broad_data_purchase_signal import _select_filter


def test_select_filter_uses_validation_metrics_only() -> None:
    metrics = {
        "bad": {"trades": 50, "total_pnl": -10.0, "profit_factor": 0.9, "max_drawdown": -100.0},
        "good": {"trades": 30, "total_pnl": 500.0, "profit_factor": 1.2, "max_drawdown": -200.0},
        "best_pf": {"trades": 25, "total_pnl": 300.0, "profit_factor": 1.5, "max_drawdown": -100.0},
    }

    assert _select_filter(metrics) == "best_pf"
