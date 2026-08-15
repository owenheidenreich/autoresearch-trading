from __future__ import annotations

from pathlib import Path

import pandas as pd

from v4.scripts.run_protocol114_skeptical_falsification import (
    chart_source_of_truth_checks,
    concentration_summary,
    delay_stress_for_trade,
    paper_account_checks,
    stressed_pnl,
)


def test_stressed_pnl_charges_both_entry_and_exit_side() -> None:
    assert stressed_pnl(100.0, 0.25) == 50.0


def test_paper_account_checks_detect_unaffordable_and_overlap() -> None:
    headline = pd.DataFrame(
        [
            {
                "seed": 1,
                "candidate_uid": "held",
                "session": "2026-03-06",
                "decision_ts": pd.Timestamp("2026-03-06T15:00:00Z"),
                "exit_ts": pd.Timestamp("2026-03-06T15:10:00Z"),
                "entry_premium": 1_000.0,
                "pnl": 100.0,
            },
            {
                "seed": 1,
                "candidate_uid": "overlap",
                "session": "2026-03-06",
                "decision_ts": pd.Timestamp("2026-03-06T15:05:00Z"),
                "exit_ts": pd.Timestamp("2026-03-06T15:08:00Z"),
                "entry_premium": 1_000.0,
                "pnl": 200.0,
            },
            {
                "seed": 1,
                "candidate_uid": "too_expensive",
                "session": "2026-03-06",
                "decision_ts": pd.Timestamp("2026-03-06T15:15:00Z"),
                "exit_ts": pd.Timestamp("2026-03-06T15:20:00Z"),
                "entry_premium": 12_000.0,
                "pnl": 300.0,
            },
        ]
    )

    [check] = paper_account_checks(headline, starting_equity=10_000.0)

    assert check.overlap_errors == 1
    assert check.skipped_unaffordable == 1
    assert check.ending_equity == 10_300.0
    assert check.all_flat_by_session_end is True


def test_paper_account_checks_detect_after_close_exits() -> None:
    headline = pd.DataFrame(
        [
            {
                "seed": 1,
                "candidate_uid": "late_exit",
                "session": "2026-03-06",
                "decision_ts": pd.Timestamp("2026-03-06T20:55:00Z"),
                "exit_ts": pd.Timestamp("2026-03-06T21:01:00Z"),
                "entry_premium": 1_000.0,
                "pnl": 100.0,
            }
        ]
    )

    [check] = paper_account_checks(headline, starting_equity=10_000.0)

    assert check.all_flat_by_session_end is False


def test_delay_stress_for_trade_reprices_with_one_minute_lag() -> None:
    trade = {
        "candidate_uid": "u1",
        "reported_split": "q1_2026",
        "seed": 1,
        "session": "2026-03-06",
        "decision_ts": pd.Timestamp("2026-03-06T15:00:00Z"),
        "exit_ts": pd.Timestamp("2026-03-06T15:03:00Z"),
        "entry_ask": 10.0,
        "pnl": 200.0,
    }
    quotes = pd.DataFrame(
        [
            {"quote_time": pd.Timestamp("2026-03-06T15:00:00Z"), "ask": 10.0, "bid": 9.9},
            {"quote_time": pd.Timestamp("2026-03-06T15:01:00Z"), "ask": 10.5, "bid": 10.4},
            {"quote_time": pd.Timestamp("2026-03-06T15:04:00Z"), "ask": 11.4, "bid": 11.25},
        ]
    )

    row = delay_stress_for_trade(trade, quotes)

    assert row["entry_delay_pnl"] == 150.0
    assert row["exit_delay_pnl"] == 125.0
    assert row["both_delay_pnl"] == 75.0


def test_concentration_summary_flags_top20_majority() -> None:
    rows = []
    for i in range(25):
        rows.append(
            {
                "pnl": 1_000.0 if i < 20 else 10.0,
                "day": f"2026-03-{1 + (i % 5):02d}",
                "month": "2026-03",
                "week": "2026-W10",
            }
        )
    summary = concentration_summary(pd.DataFrame(rows))

    assert summary["top20_majority"] is True
    assert summary["single_trade_majority"] is False


def test_chart_source_of_truth_requires_one_equity_and_no_train_rows(tmp_path: Path) -> None:
    (tmp_path / "equity.html").write_text("<html></html>")
    pd.DataFrame([{"stage": "test"}]).to_csv(tmp_path / "trades.csv", index=False)

    ok = chart_source_of_truth_checks(tmp_path)

    assert ok["single_equity_source"] is True
    assert ok["no_train_validation_in_headline"] is True

    (tmp_path / "holdout_only_equity.html").write_text("<html></html>")
    pd.DataFrame([{"stage": "train"}]).to_csv(tmp_path / "trades.csv", index=False)
    bad = chart_source_of_truth_checks(tmp_path)

    assert bad["single_equity_source"] is False
    assert bad["no_train_validation_in_headline"] is False
