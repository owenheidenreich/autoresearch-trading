from __future__ import annotations

import pandas as pd

from v4.scripts import run_protocol101_strategy_forensics_packet as packet


def test_enrich_protocol101_trades_adds_strategy_buckets() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": ["2026-01-02T15:05:00+00:00"],
            "exit_time": ["2026-01-02T15:15:00+00:00"],
            "right": ["C"],
            "offset": [-25.0],
            "entry_ask": [31.0],
            "entry_bid": [30.7],
            "score": [1.2],
            "threshold": [0.7],
            "path_mfe": [1_000.0],
            "path_mae": [-100.0],
            "pnl": [700.0],
            "premium_paid": [3_100.0],
            "exit_reason": ["sequence_residual_override"],
        }
    )

    enriched = packet.enrich_protocol101_trades(frame)

    assert enriched.loc[0, "time_bucket"] == "post_open_morning"
    assert enriched.loc[0, "moneyness"] == "ITM"
    assert enriched.loc[0, "premium_bucket"] == "3000_3500"
    assert enriched.loc[0, "duration_bucket"] == "6_10m"
    assert enriched.loc[0, "score_margin_bucket"] == "0.50_1.00"
    assert enriched.loc[0, "mfe_capture"] == 0.7


def test_hard_stop_autopsy_keeps_only_stop_rows_sorted_by_pnl() -> None:
    trades = pd.DataFrame(
        {
            "segment": ["q1_2026", "q1_2026"],
            "session": ["2026-01-02", "2026-01-02"],
            "decision_time": ["2026-01-02T15:00:00+00:00", "2026-01-02T15:05:00+00:00"],
            "exit_time": ["2026-01-02T15:10:00+00:00", "2026-01-02T15:15:00+00:00"],
            "right": ["P", "C"],
            "offset": [20.0, -25.0],
            "premium_paid": [2_500.0, 3_000.0],
            "entry_ask": [25.0, 30.0],
            "entry_bid": [24.8, 29.7],
            "score": [1.0, 1.5],
            "threshold": [0.8, 0.8],
            "pnl": [-1_000.0, 500.0],
            "path_mfe": [300.0, 600.0],
            "path_mae": [-1_000.0, -100.0],
            "quote_gap_seconds": [0.0, 0.0],
            "exit_reason": ["hard_stop", "target"],
            "contract_id": ["p", "c"],
            "candidate_uid": ["stop", "target"],
        }
    )
    enriched = packet.enrich_protocol101_trades(trades)

    autopsy = packet.build_hard_stop_autopsy(enriched)

    assert len(autopsy) == 1
    assert autopsy.iloc[0]["candidate_uid"] == "stop"
    assert bool(autopsy.iloc[0]["early_mfe_then_loss"]) is True


def test_slot_opportunity_readiness_blocks_deployed_state_only_actions() -> None:
    actions = pd.DataFrame(
        {
            "protocol101_action": ["wait", "enter", "holding", "holding", "exit_then_wait"],
            "split": ["q1_2026"] * 5,
        }
    )

    status = packet.build_slot_opportunity_readiness(actions)

    assert status["status"] == "blocked_missing_counterfactual_flat_protocol101_actions"
    assert status["action_counts"]["holding"] == 2
    assert status["observable_enter_rows_while_holding"] == 0


def test_delay_fragility_groups_by_trade_archetype() -> None:
    trades = pd.DataFrame(
        {
            "candidate_uid": ["a"],
            "segment": ["q1_2026"],
            "seed": [1],
            "session": ["2026-01-02"],
            "decision_time": ["2026-01-02T15:05:00+00:00"],
            "exit_time": ["2026-01-02T15:15:00+00:00"],
            "right": ["C"],
            "offset": [-25.0],
            "premium_paid": [3_100.0],
            "entry_ask": [31.0],
            "entry_bid": [30.8],
            "score": [1.2],
            "threshold": [0.7],
            "pnl": [500.0],
            "path_mfe": [600.0],
            "path_mae": [-100.0],
            "exit_reason": ["target"],
        }
    )
    delays = pd.DataFrame(
        {
            "candidate_uid": ["a"],
            "reported_split": ["q1_2026"],
            "seed": [1],
            "session": ["2026-01-02"],
            "pnl": [500.0],
            "entry_delay_pnl": [-100.0],
            "both_delay_pnl": [-200.0],
        }
    )

    fragility = packet.build_delay_fragility_by_archetype(trades, delays)

    assert len(fragility) == 1
    assert fragility.iloc[0]["entry_delay_delta"] == -600.0
    assert fragility.iloc[0]["entry_delay_loss_rate"] == 1.0


def test_policy_quality_comparison_summarizes_protocol101_vs_challenger() -> None:
    frame = pd.DataFrame(
        {
            "policy": ["protocol101", "challenger"],
            "reported_split": ["q1_2026", "q1_2026"],
            "right": ["C", "C"],
            "offset": [-25.0, 25.0],
            "entry_premium": [3_000.0, 500.0],
            "decision_time": ["2026-01-02T15:00:00+00:00", "2026-01-02T15:00:00+00:00"],
            "exit_time": ["2026-01-02T15:10:00+00:00", "2026-01-02T15:10:00+00:00"],
            "pnl": [300.0, -200.0],
            "path_mfe": [400.0, 100.0],
            "path_mae": [-50.0, -200.0],
            "exit_reason": ["target", "hard_stop"],
        }
    )

    comparison = packet.build_policy_quality_comparison(frame)

    assert set(comparison["policy"]) == {"protocol101", "challenger"}
    assert comparison.loc[comparison["policy"].eq("protocol101"), "pnl"].iloc[0] == 300.0
    assert comparison.loc[comparison["policy"].eq("challenger"), "pnl"].iloc[0] == -200.0
