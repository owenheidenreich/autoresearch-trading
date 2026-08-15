from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from v4.scripts import run_protocol101_loss_reversal_causal_state_attachment as attach


def _open_slots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["q1_2026"],
            "fold": ["fold"],
            "seed": [1],
            "session": ["2026-03-02"],
            "open_trade_candidate_uid": ["open-1"],
            "open_trade_contract_id": ["SPXW-C"],
            "open_trade_entry_dt": ["2026-03-02T15:00:00+00:00"],
            "open_trade_exit_dt": ["2026-03-02T15:10:00+00:00"],
            "open_trade_right": ["C"],
            "open_trade_exit_reason": ["hard_stop"],
            "best_blocked_candidate_uid": ["blocked-1"],
            "best_blocked_decision_dt": ["2026-03-02T15:05:00+00:00"],
            "best_blocked_candidate_pnl": [700.0],
            "best_blocked_minus_open_pnl": [1600.0],
            "open_trade_pnl": [-900.0],
            "open_premium_paid": [3000.0],
        }
    )


def _blocked_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reported_split": ["q1_2026"],
            "fold": ["fold"],
            "seed": [1],
            "session": ["2026-03-02"],
            "open_trade_candidate_uid": ["open-1"],
            "blocked_candidate_uid": ["blocked-1"],
            "blocked_decision_dt": ["2026-03-02T15:05:00+00:00"],
            "blocked_right": ["P"],
            "blocked_contract_id": ["SPXW-P"],
            "blocked_relation": ["opposite_side"],
            "blocked_candidate_pnl": [700.0],
            "blocked_entry_ask": [28.0],
            "blocked_entry_spread": [0.4],
        }
    )


def _quotes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quote_time": [
                "2026-03-02T15:00:00+00:00",
                "2026-03-02T15:04:30+00:00",
                "2026-03-02T15:05:30+00:00",
                "2026-03-02T15:10:00+00:00",
            ],
            "contract_id": ["SPXW-C"] * 4,
            "bid": [29.8, 25.0, 24.0, 21.0],
            "ask": [30.0, 25.4, 24.4, 21.4],
            "underlying_price": [6000.0, 5995.0, 5990.0, 5980.0],
        }
    )


def test_quote_state_uses_last_quote_before_signal_not_future_quote() -> None:
    row = _open_slots().merge(
        _blocked_events().rename(
            columns={
                "blocked_candidate_uid": "best_blocked_candidate_uid",
                "blocked_decision_dt": "best_blocked_decision_dt",
                "blocked_right": "best_blocked_right",
                "blocked_contract_id": "best_blocked_contract_id",
                "blocked_relation": "best_blocked_relation",
            }
        ),
        on=[
            "reported_split",
            "fold",
            "seed",
            "session",
            "open_trade_candidate_uid",
            "best_blocked_candidate_uid",
            "best_blocked_decision_dt",
        ],
    ).iloc[0]
    quotes = _quotes().copy()
    quotes["quote_time"] = pd.to_datetime(quotes["quote_time"], utc=True)

    state = attach.quote_state_for_candidate(row, quotes)

    assert state["direct_quote_state_status"] == "matched"
    assert state["current_quote_time"] == "2026-03-02T15:04:30+00:00"
    assert state["quote_lag_seconds"] == 30.0
    assert state["current_pnl_at_signal"] == -500.0
    assert state["one_step_pnl_delta_after_signal"] == -100.0


def test_run_uses_fallback_normalized_dir_and_blocks_training(tmp_path) -> None:
    slot_dir = tmp_path / "slot"
    official_dir = tmp_path / "official"
    fallback_dir = tmp_path / "fallback"
    out_dir = tmp_path / "out"
    doc = tmp_path / "doc.md"
    slot_dir.mkdir()
    official_dir.mkdir()
    fallback_dir.mkdir()
    _open_slots().to_csv(slot_dir / "enriched_open_trade_slot_summary.csv", index=False)
    _blocked_events().to_csv(slot_dir / "enriched_blocked_slot_events.csv", index=False)
    (official_dir / "databento_spxw_0dte_2026-03-02_official_context.parquet").write_text("not parquet")
    _quotes().to_parquet(fallback_dir / "databento_spxw_0dte_2026-03-02.parquet")

    args = SimpleNamespace(output_dir=out_dir, doc=doc, slot_cost_dir=slot_dir, normalized_dir=[official_dir, fallback_dir])
    summary = attach.run(args)

    assert summary["model_training"] is False
    assert summary["counts"]["matched_direct_quote_state_rows"] == 1
    assert summary["live_state"]["priority_1_rows"] == 1
    assert (out_dir / "report.md").read_text() == doc.read_text()
