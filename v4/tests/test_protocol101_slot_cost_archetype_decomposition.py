from __future__ import annotations

import pandas as pd

from v4.scripts import run_protocol101_slot_cost_archetype_decomposition as decomp


def _open_slots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["march_2026", "q3_2025"],
            "fold": ["fold", "fold"],
            "seed": [1, 1],
            "session": ["2026-03-02", "2025-07-01"],
            "open_trade_candidate_uid": ["open-a", "open-b"],
            "open_trade_contract_id": ["SPXW-A-C", "SPXW-B-P"],
            "open_trade_entry_dt": ["2026-03-02T15:00:00+00:00", "2025-07-01T15:00:00+00:00"],
            "open_trade_exit_dt": ["2026-03-02T15:10:00+00:00", "2025-07-01T15:20:00+00:00"],
            "open_trade_right": ["C", "P"],
            "open_trade_exit_reason": ["protocol054_fallback", "sequence_residual_override"],
            "blocked_entry_events": [2, 1],
            "positive_blocked_entry_events": [1, 1],
            "sum_blocked_candidate_pnl_non_additive": [450.0, 600.0],
            "best_blocked_candidate_pnl": [500.0, 600.0],
            "best_blocked_candidate_uid": ["blocked-a", "blocked-b"],
            "best_blocked_decision_dt": ["2026-03-02T15:05:00+00:00", "2025-07-01T15:07:00+00:00"],
            "open_trade_pnl": [-100.0, 900.0],
            "open_minus_best_blocked_pnl": [-600.0, 300.0],
            "slot_cost_positive_vs_best_blocked": [True, False],
        }
    )


def _blocked_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["march_2026", "march_2026", "q3_2025"],
            "fold": ["fold", "fold", "fold"],
            "seed": [1, 1, 1],
            "session": ["2026-03-02", "2026-03-02", "2025-07-01"],
            "blocked_decision_dt": [
                "2026-03-02T15:05:00+00:00",
                "2026-03-02T15:06:00+00:00",
                "2025-07-01T15:07:00+00:00",
            ],
            "blocked_candidate_uid": ["blocked-a", "blocked-other", "blocked-b"],
            "blocked_contract_id": ["SPXW-A-C", "SPXW-Z-P", "SPXW-C-P"],
            "blocked_right": ["C", "P", "P"],
            "blocked_offset": [-20.0, 20.0, 10.0],
            "blocked_score": [1.0, 0.5, 2.0],
            "blocked_threshold": [0.0, 0.0, 0.0],
            "blocked_candidate_pnl": [500.0, -50.0, 600.0],
            "blocked_entry_ask": [10.0, 8.0, 11.0],
            "blocked_entry_spread": [0.2, 0.2, 0.3],
            "blocked_time_bucket": ["post_open_morning", "post_open_morning", "post_open_morning"],
            "open_trade_entry_dt": [
                "2026-03-02T15:00:00+00:00",
                "2026-03-02T15:00:00+00:00",
                "2025-07-01T15:00:00+00:00",
            ],
            "open_trade_exit_dt": [
                "2026-03-02T15:10:00+00:00",
                "2026-03-02T15:10:00+00:00",
                "2025-07-01T15:20:00+00:00",
            ],
            "open_trade_candidate_uid": ["open-a", "open-a", "open-b"],
            "open_trade_contract_id": ["SPXW-A-C", "SPXW-A-C", "SPXW-B-P"],
            "open_trade_right": ["C", "C", "P"],
            "open_trade_pnl": [-100.0, -100.0, 900.0],
            "open_trade_exit_reason": ["protocol054_fallback", "protocol054_fallback", "sequence_residual_override"],
            "minutes_after_open_entry": [5.0, 6.0, 7.0],
            "minutes_before_open_exit": [5.0, 4.0, 13.0],
            "open_minus_blocked_pnl": [-600.0, -50.0, 300.0],
        }
    )


def _actual_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [1, 1],
            "trade_number": [1, 2],
            "stage": ["test", "test"],
            "segment": ["q1_2026", "q3_2025"],
            "source_protocol": ["test", "test"],
            "session": ["2026-03-02", "2025-07-01"],
            "decision_time": ["2026-03-02T15:00:00+00:00", "2025-07-01T15:00:00+00:00"],
            "exit_time": ["2026-03-02T15:10:00+00:00", "2025-07-01T15:20:00+00:00"],
            "right": ["C", "P"],
            "side": ["CALL", "PUT"],
            "offset": [-20.0, 10.0],
            "contract_id": ["SPXW-A-C", "SPXW-B-P"],
            "entry_ask": [30.0, 20.0],
            "entry_bid": [29.5, 19.8],
            "premium_paid": [3000.0, 2000.0],
            "path_mfe": [700.0, 1000.0],
            "path_mae": [-300.0, -200.0],
            "pnl": [-100.0, 900.0],
            "score": [1.0, 2.0],
            "threshold": [0.5, 0.5],
            "exit_reason": ["protocol054_fallback", "sequence_residual_override"],
            "candidate_uid": ["open-a", "open-b"],
        }
    )


def test_enrich_open_slots_maps_march_to_q1_actual_trades() -> None:
    actual = decomp.enrich_protocol101_trades(_actual_trades())
    actual["join_split"] = actual["segment"].map(decomp.split_for_join)

    enriched = decomp.enrich_open_slots(_open_slots(), actual)

    assert enriched.iloc[0]["open_trade_join_status"] == "matched_actual_trade"
    assert enriched.iloc[0]["open_time_bucket"] == "post_open_morning"
    assert enriched.iloc[0]["best_blocked_minus_open_pnl"] == 600.0


def test_blocked_relation_identifies_same_contract_and_opposite_side() -> None:
    actual = decomp.enrich_protocol101_trades(_actual_trades())
    actual["join_split"] = actual["segment"].map(decomp.split_for_join)
    enriched = decomp.enrich_blocked_events(_blocked_events(), actual)

    assert enriched.iloc[0]["blocked_relation"] == "same_contract"
    assert enriched.iloc[1]["blocked_relation"] == "opposite_side"
    assert enriched.iloc[2]["blocked_relation"] == "same_side_different_contract"


def test_thesis_tests_find_losing_open_trade_slot_cost() -> None:
    actual = decomp.enrich_protocol101_trades(_actual_trades())
    actual["join_split"] = actual["segment"].map(decomp.split_for_join)
    open_slots = decomp.enrich_open_slots(_open_slots(), actual)
    blocked = decomp.enrich_blocked_events(_blocked_events(), actual)
    best = decomp.build_best_blocked_event_rows(open_slots, blocked)

    tests = decomp.build_thesis_tests(open_slots, best)
    row = tests[tests["thesis"].eq("losing_open_trade_blocks_positive_later_signal")].iloc[0]

    assert row["open_trades"] == 1
    assert row["best_blocked_minus_open_total"] == 600.0


def test_summary_blocks_training_and_challenge() -> None:
    actual = decomp.enrich_protocol101_trades(_actual_trades())
    actual["join_split"] = actual["segment"].map(decomp.split_for_join)
    open_slots = decomp.enrich_open_slots(_open_slots(), actual)
    blocked = decomp.enrich_blocked_events(_blocked_events(), actual)
    archetypes = decomp.summarize_open_archetypes(open_slots)
    best = decomp.build_best_blocked_event_rows(open_slots, blocked)
    tests = decomp.build_thesis_tests(open_slots, best)

    summary = decomp.build_summary(open_slots, blocked, archetypes, tests)

    assert summary["training_allowed"] is False
    assert summary["challenge_allowed"] is False
    assert summary["counts"]["actual_trade_join_failures"] == 0
