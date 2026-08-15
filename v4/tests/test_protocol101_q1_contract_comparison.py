from __future__ import annotations

import pandas as pd

from v4.scripts.run_protocol101_q1_contract_comparison import _decision_comparison, _trade_metrics


def test_decision_comparison_classifies_action_and_score_drift() -> None:
    legacy = pd.DataFrame([{"session": "2026-01-02", "decision_time": "t", "action": "enter", "max_edge": 26.0, "candidate_count": 2, "selected_contract_id": "A"}])
    live = pd.DataFrame([{"session": "2026-01-02", "decision_time": "t", "action": "wait", "max_edge": 24.0, "candidate_count": 1, "selected_contract_id": None}])

    out = _decision_comparison(legacy, live).iloc[0]

    assert not bool(out["action_match"])
    assert out["max_edge_delta"] == -2.0
    assert out["candidate_count_delta"] == -1


def test_trade_metrics_report_serial_risk_and_concentration() -> None:
    frame = pd.DataFrame(
        [
            {"session": "2026-01-02", "decision_time": "a", "serial_status": "taken", "entry_ask": 10.0, "candidate_pnl": 500.0},
            {"session": "2026-01-02", "decision_time": "b", "serial_status": "taken", "entry_ask": 8.0, "candidate_pnl": -300.0},
            {"session": "2026-01-05", "decision_time": "c", "serial_status": "taken", "entry_ask": 5.0, "candidate_pnl": 200.0},
        ]
    )

    metrics = _trade_metrics(frame, starting_equity=10_000.0)

    assert metrics["trades"] == 3
    assert metrics["total_pnl"] == 400.0
    assert metrics["ending_equity"] == 10_400.0
    assert metrics["max_drawdown_dollars"] == -300.0
    assert metrics["profit_factor"] == 700.0 / 300.0
    assert metrics["premium_deployed"] == 2300.0
