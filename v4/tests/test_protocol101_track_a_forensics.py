from __future__ import annotations

import pandas as pd

from v4.scripts import run_protocol101_track_a_forensics as track_a
from v4.scripts.run_protocol101_strategy_forensics_packet import enrich_protocol101_trades


def _sample_trade(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": 1,
        "trade_number": 1,
        "stage": "test",
        "segment": "q1_2026",
        "source_protocol": "test",
        "session": "2026-01-02",
        "decision_time": "2026-01-02T15:00:00+00:00",
        "exit_time": "2026-01-02T15:07:00+00:00",
        "right": "P",
        "side": "PUT",
        "offset": 20.0,
        "contract_id": "SPXW-test-P",
        "entry_spx": 5900.0,
        "exit_spx": 5910.0,
        "entry_bar": 1,
        "exit_bar": 8,
        "entry_quote_time": "2026-01-02T15:00:00+00:00",
        "exit_quote_time": "2026-01-02T15:07:00+00:00",
        "entry_ask": 25.0,
        "entry_bid": 24.8,
        "exit_bid": 10.0,
        "exit_ask": 10.4,
        "entry_bid_size": 3,
        "entry_ask_size": 3,
        "premium_paid": 2500.0,
        "quote_gap_seconds": 0.0,
        "path_mfe": 400.0,
        "path_mae": -1500.0,
        "path_final_pnl": -1500.0,
        "path_points": 8,
        "quote_backfill_status": "ok",
        "paper_seed": 1,
        "paper_selected": True,
        "paper_skip_reason": "",
        "paper_cash_before": 10000.0,
        "paper_cash_after": 8500.0,
        "paper_equity_after": 8500.0,
        "paper_cumulative_pnl": -1500.0,
        "paper_premium": 2500.0,
        "paper_buying_power_used": 2500.0,
        "paper_buying_power_pct_cash": 0.25,
        "paper_intratrade_low_equity": 8500.0,
        "paper_intratrade_high_equity": 10400.0,
        "pnl": -1500.0,
        "cumulative_pnl": -1500.0,
        "score": 1.0,
        "threshold": 0.5,
        "exit_reason": "hard_stop",
        "candidate_uid": "candidate-1",
    }
    row.update(overrides)
    return row


def test_hard_stop_classification_identifies_path_management_candidate() -> None:
    trades = enrich_protocol101_trades(pd.DataFrame([_sample_trade()]))

    classified = track_a.build_hard_stop_classification(trades)

    assert len(classified) == 1
    assert classified.iloc[0]["mechanism"] == "path_management_candidate"
    assert "lifecycle" in classified.iloc[0]["next_test"]


def test_hard_stop_classification_prioritizes_quote_fragility() -> None:
    trades = enrich_protocol101_trades(
        pd.DataFrame([_sample_trade(entry_ask=25.0, entry_bid=23.5, quote_gap_seconds=10.0)])
    )

    classified = track_a.build_hard_stop_classification(trades)

    assert classified.iloc[0]["mechanism"] == "execution_or_quote_fragility_candidate"


def test_losing_day_mechanism_detects_hard_stop_dominated_day() -> None:
    trades = enrich_protocol101_trades(
        pd.DataFrame(
            [
                _sample_trade(candidate_uid="loss-1", pnl=-1500.0, exit_reason="hard_stop"),
                _sample_trade(
                    candidate_uid="win-1",
                    decision_time="2026-01-02T15:10:00+00:00",
                    exit_time="2026-01-02T15:15:00+00:00",
                    pnl=100.0,
                    path_mfe=100.0,
                    path_mae=-50.0,
                    exit_reason="target",
                ),
            ]
        )
    )

    days = track_a.build_losing_day_mechanisms(trades)

    assert len(days) == 1
    assert days.iloc[0]["mechanism"] == "hard_stop_dominated"


def test_runner_state_requires_giveback_guard_when_forced_flat_loses_value() -> None:
    row = pd.Series(
        {
            "post_frozen_best_minus_frozen_pnl": 1000.0,
            "forced_flat_minus_frozen_pnl": -600.0,
            "oracle_minus_frozen_pnl": 1200.0,
        }
    )

    assert track_a.classify_runner_state(row) == "giveback_guard_required"


def test_selected_trade_runner_join_reports_partial_coverage() -> None:
    trades = enrich_protocol101_trades(pd.DataFrame([_sample_trade()]))
    full_path = pd.DataFrame(
        {
            "seed": [1],
            "session": ["2026-01-02"],
            "decision_time": ["2026-01-02T15:00:00+00:00"],
            "contract_id": ["SPXW-test-P"],
            "right": ["P"],
            "frozen_pnl": [-1500.0],
            "post_frozen_best_minus_frozen_pnl": [800.0],
            "forced_flat_minus_frozen_pnl": [-200.0],
            "oracle_minus_frozen_pnl": [900.0],
        }
    )

    joined, summary = track_a.build_selected_trade_runner_join(trades, full_path)

    assert summary["status"] == "partial_selected_trade_post_exit_path_coverage"
    assert summary["matched_unique_trade_keys"] == 1
    assert summary["match_rate"] == 1.0
    assert joined.iloc[0]["runner_state"] == "giveback_guard_required"


def test_internal_slot_cost_proxy_blocks_without_counterfactual_flat_actions() -> None:
    actions = pd.DataFrame(
        {
            "split": ["q1_2026", "q1_2026", "q1_2026"],
            "protocol101_action": ["enter", "holding", "wait"],
        }
    )
    all_seed = pd.DataFrame(
        {
            "seed": [1],
            "segment": ["q1_2026"],
            "decision_time": ["2026-01-02T15:00:00+00:00"],
            "exit_time": ["2026-01-02T15:05:00+00:00"],
        }
    )

    proxy = track_a.build_internal_slot_cost_proxy(actions, all_seed)

    assert proxy.iloc[0]["status"] == "blocked_missing_counterfactual_flat_protocol101_actions"
    assert proxy.iloc[0]["observable_blocked_enter_rows"] == 0
    assert "lower bound" in proxy.iloc[0]["why_blocked"]


def test_summary_never_allows_challenge() -> None:
    trades = enrich_protocol101_trades(pd.DataFrame([_sample_trade()]))
    hard = track_a.build_hard_stop_classification(trades)
    runner = pd.DataFrame(
        {
            "runner_state": ["runner_extension_candidate"],
            "rows": [3],
            "post_exit_best_delta": [1000.0],
            "forced_flat_delta": [100.0],
        }
    )
    slot = pd.DataFrame({"split": ["q1_2026"], "status": ["blocked_missing_counterfactual_flat_protocol101_actions"]})

    summary = track_a.summarize_packet(trades, hard, runner, slot)

    assert summary["challenge_allowed"] is False
    assert summary["model_training"] is False
    assert "counterfactual_flat_protocol101_actions_missing" in summary["blockers"]
