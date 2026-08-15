from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from v4.scripts import run_protocol101_loss_reversal_exit_gate_diagnostic as diag


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["q1_2026", "q1_2026", "q1_2026"],
            "fold": ["fold", "fold", "fold"],
            "seed": [1, 1, 1],
            "session": ["2026-03-02", "2026-03-03", "2026-03-04"],
            "open_trade_candidate_uid": ["open-1", "open-2", "open-3"],
            "open_trade_contract_id": ["SPXW-C", "SPXW-P", "SPXW-C2"],
            "open_trade_entry_dt": [
                "2026-03-02T15:00:00+00:00",
                "2026-03-03T15:00:00+00:00",
                "2026-03-04T15:00:00+00:00",
            ],
            "open_trade_exit_dt": [
                "2026-03-02T15:10:00+00:00",
                "2026-03-03T15:10:00+00:00",
                "2026-03-04T15:10:00+00:00",
            ],
            "best_blocked_decision_dt": [
                "2026-03-02T15:05:00+00:00",
                "2026-03-03T15:05:00+00:00",
                "2026-03-04T15:05:00+00:00",
            ],
            "open_trade_right": ["C", "P", "C"],
            "best_blocked_right": ["P", "P", "P"],
            "best_blocked_relation": ["opposite_side", "same_contract", "opposite_side"],
            "open_trade_exit_reason": ["hard_stop", "target", "hard_stop"],
            "open_trade_pnl": [-700.0, -300.0, -500.0],
            "best_blocked_candidate_pnl": [800.0, 400.0, 600.0],
            "best_blocked_minus_open_pnl": [1500.0, 700.0, 1100.0],
            "best_blocked_entry_ask": [30.0, 28.0, 25.0],
            "best_blocked_entry_spread": [0.4, 0.3, 0.4],
            "open_duration_minutes": [10.0, 10.0, 10.0],
            "best_blocked_minutes_after_open_entry": [5.0, 5.0, 5.0],
            "best_blocked_minutes_before_open_exit": [5.0, 5.0, 5.0],
            "loss_reversal_subcase": [
                "final_loss_plus_opposite_side_signal",
                "final_loss_plus_same_side_signal",
                "final_loss_plus_opposite_side_signal",
            ],
            "candidate_label_limitations": ["final_pnl_not_live_state"] * 3,
        }
    )


def _hold_state_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["q1_2026", "q1_2026"],
            "seed": [1, 1],
            "session": ["2026-03-02", "2026-03-03"],
            "candidate_uid": ["open-1", "open-2"],
            "contract_id": ["SPXW-C", "SPXW-P"],
            "state_time": ["2026-03-02T15:05:00+00:00", "2026-03-03T15:05:00+00:00"],
            "minutes_since_entry": [5.0, 5.0],
            "minutes_to_protocol101_exit": [5.0, 5.0],
            "minutes_to_forced_flat": [300.0, 300.0],
            "bid": [20.0, 32.0],
            "ask": [20.4, 32.4],
            "mid": [20.2, 32.2],
            "spread": [0.4, 0.4],
            "underlying_price": [6000.0, 6001.0],
            "current_pnl": [-500.0, 200.0],
            "mfe_to_now": [100.0, 300.0],
            "mae_to_now": [-700.0, -100.0],
            "giveback_from_mfe": [600.0, 100.0],
            "giveback_fraction": [6.0, 0.33],
            "pnl_velocity_1": [-100.0, 50.0],
            "pnl_velocity_3": [-80.0, 30.0],
            "pnl_velocity_5": [-60.0, 25.0],
            "time_since_mfe_minutes": [4.0, 1.0],
            "a_hold": [-200.0, 500.0],
            "a_exit": [200.0, -500.0],
            "a_hold_one_step_realized": [-50.0, 40.0],
            "q_exit": [-500.0, 200.0],
            "q_hold": [-700.0, 700.0],
            "oracle_holding_action": ["exit", "hold"],
            "oracle_one_step_action": ["exit", "hold"],
        }
    )


def test_attach_causal_hold_state_separates_final_loss_from_live_loss() -> None:
    joined = diag.attach_causal_hold_state(_candidate_rows(), _hold_state_rows())

    first = joined[joined["open_trade_candidate_uid"].eq("open-1")].iloc[0]
    second = joined[joined["open_trade_candidate_uid"].eq("open-2")].iloc[0]
    third = joined[joined["open_trade_candidate_uid"].eq("open-3")].iloc[0]

    assert first["loss_reversal_live_test_bucket"] == "priority_1_current_loss_opposite_side_one_step_negative"
    assert second["loss_reversal_live_test_bucket"] == "priority_5_final_loss_but_not_current_loss"
    assert third["loss_reversal_live_test_bucket"] == "blocked_missing_causal_state_at_signal"


def test_summary_blocks_training_when_causal_state_partial() -> None:
    joined = diag.attach_causal_hold_state(_candidate_rows(), _hold_state_rows())
    causal_summary = diag.build_causal_state_summary(joined)
    review = diag.build_manual_review_priority_rows(joined)
    summary = diag.build_summary(joined, causal_summary, review)

    assert summary["model_training"] is False
    assert summary["challenge_allowed"] is False
    assert summary["decision"] == "protocol101_loss_reversal_exit_gate_diagnostic_partial_causal_state_coverage_training_blocked"
    assert summary["causal_state"]["matched_rows"] == 2
    assert summary["causal_state"]["current_loss_rows"] == 1


def test_run_writes_packet_outputs(tmp_path) -> None:
    slot_dir = tmp_path / "slot"
    out_dir = tmp_path / "out"
    doc = tmp_path / "doc.md"
    slot_dir.mkdir()

    open_slots = _candidate_rows().rename(
        columns={
            "best_blocked_right": "ignored_best_blocked_right",
            "best_blocked_relation": "ignored_best_blocked_relation",
        }
    )
    open_slots["best_blocked_candidate_uid"] = ["blocked-1", "blocked-2", "blocked-3"]
    blocked = pd.DataFrame(
        {
            "reported_split": ["q1_2026", "q1_2026", "q1_2026"],
            "fold": ["fold", "fold", "fold"],
            "seed": [1, 1, 1],
            "session": ["2026-03-02", "2026-03-03", "2026-03-04"],
            "open_trade_candidate_uid": ["open-1", "open-2", "open-3"],
            "blocked_candidate_uid": ["blocked-1", "blocked-2", "blocked-3"],
            "blocked_decision_dt": [
                "2026-03-02T15:05:00+00:00",
                "2026-03-03T15:05:00+00:00",
                "2026-03-04T15:05:00+00:00",
            ],
            "blocked_right": ["P", "P", "P"],
            "blocked_contract_id": ["blocked-P1", "blocked-P2", "blocked-P3"],
            "blocked_relation": ["opposite_side", "same_contract", "opposite_side"],
            "blocked_candidate_pnl": [800.0, 400.0, 600.0],
        }
    )
    open_slots.to_csv(slot_dir / "enriched_open_trade_slot_summary.csv", index=False)
    blocked.to_csv(slot_dir / "enriched_blocked_slot_events.csv", index=False)
    hold_path = tmp_path / "hold.parquet"
    _hold_state_rows().to_parquet(hold_path)

    args = SimpleNamespace(output_dir=out_dir, doc=doc, slot_cost_dir=slot_dir, hold_exit_dataset=hold_path)
    summary = diag.run(args)

    assert summary["counts"]["loss_reversal_candidates"] == 3
    for path in summary["outputs"].values():
        assert pd.io.common.file_exists(path)
    assert (out_dir / "report.md").read_text() == doc.read_text()
