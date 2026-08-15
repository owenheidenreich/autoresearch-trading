from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from v4.scripts import run_protocol101_loss_reversal_local_action_pricing as pricing


def _direct_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "direct_quote_state_status": ["matched", "matched", "missing"],
            "live_test_bucket": [
                "priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates",
                "priority_3_current_loss_same_side",
                "blocked_no_readable_session_quotes",
            ],
            "reported_split": ["q1_2026", "q1_2026", "q1_2026"],
            "seed": [1, 1, 1],
            "session": ["2026-03-02", "2026-03-03", "2026-03-04"],
            "open_trade_entry_dt": ["", "", ""],
            "open_trade_exit_dt": ["", "", ""],
            "best_blocked_decision_dt": ["", "", ""],
            "open_trade_right": ["C", "P", "C"],
            "best_blocked_right": ["P", "P", "P"],
            "best_blocked_relation": ["opposite_side", "same_contract", "opposite_side"],
            "open_trade_pnl": [-900.0, -300.0, -500.0],
            "current_pnl_at_signal": [-500.0, -100.0, 0.0],
            "best_blocked_candidate_pnl": [700.0, 50.0, 100.0],
            "best_blocked_minus_open_pnl": [1600.0, 350.0, 600.0],
            "mfe_to_signal": [0.0, 100.0, 0.0],
            "mae_to_signal": [-500.0, -200.0, 0.0],
            "giveback_from_mfe_to_signal": [500.0, 200.0, 0.0],
            "current_spread": [0.4, 0.3, 0.0],
            "best_blocked_entry_ask": [28.0, 20.0, 10.0],
            "best_blocked_entry_spread": [0.4, 0.2, 0.1],
        }
    )


def test_local_action_rows_price_current_exit_and_switch() -> None:
    rows = pricing.build_local_action_rows(_direct_rows())

    assert len(rows) == 2
    first = rows.iloc[0]
    assert first["exit_now_minus_keep"] == 400.0
    assert first["switch_minus_keep"] == 1100.0
    assert first["switch_minus_keep_stress_025"] == 1050.0
    assert first["best_local_action_stress_025"] == "exit_and_switch"


def test_summary_blocks_training_and_counts_priority_one() -> None:
    rows = pricing.build_local_action_rows(_direct_rows())
    buckets = pricing.build_bucket_summary(rows)
    summary = pricing.build_summary(rows, buckets)

    assert summary["model_training"] is False
    assert summary["challenge_allowed"] is False
    assert summary["counts"]["priced_rows"] == 2
    assert summary["counts"]["priority_1_rows"] == 1
    assert summary["aggregate"]["switch_minus_keep_stress_025"] == 1250.0


def test_run_writes_doc_mirror(tmp_path) -> None:
    input_path = tmp_path / "direct.csv"
    out_dir = tmp_path / "out"
    doc = tmp_path / "doc.md"
    _direct_rows().to_csv(input_path, index=False)

    args = SimpleNamespace(input=input_path, output_dir=out_dir, doc=doc)
    summary = pricing.run(args)

    assert summary["counts"]["priced_rows"] == 2
    for path in summary["outputs"].values():
        assert pd.io.common.file_exists(path)
    assert (out_dir / "report.md").read_text() == doc.read_text()
