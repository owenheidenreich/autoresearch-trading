"""Tests for put-near fair-contract regime diagnostics."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

from v4.scripts import run_protocol101_fair_contract_put_near_regime_diagnostic as diag


def test_market_context_fields_classify_live_observable_context() -> None:
    window = np.ones((30, 7), dtype=np.float32)
    window[-1, 0] = 100.0
    window[-1, 2] = 112.0
    window[-1, 3] = -3.0
    window[-1, 4] = 55.0
    window[-1, 5] = -1.5
    window[-1, 6] = -7.0

    fields = diag.market_context_fields({"market_window": window})

    assert fields["spx_close"] == 100.0
    assert fields["spx_vwap"] == 112.0
    assert fields["vwap_gap"] == -12.0
    assert fields["vwap_gap_bucket"] == "below_vwap_gte_10"
    assert fields["put_momentum_alignment"] == "put_aligned_mom15_down"


def test_enrich_attempt_trades_joins_context_and_sequences_session(monkeypatch, tmp_path) -> None:
    search_dir = tmp_path / "search"
    attempt_id = "attempt_092_policy0_hgb_teacher_put_near_cap2_s42"
    attempt_dir = search_dir / "attempts" / attempt_id
    (attempt_dir / "training_runner").mkdir(parents=True)
    (attempt_dir / "selected_candidate_replay_gate").mkdir(parents=True)
    row_path = tmp_path / "2026-01-02.pkl"
    decision_time = datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)
    second_time = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)
    (attempt_dir / "training_runner" / "runner_plan.json").write_text(
        json.dumps({"split_files": {"diagnostic_test": [str(row_path)]}})
    )
    (attempt_dir / "training_runner" / "training_result.json").write_text(
        json.dumps({"config": {"target_mode": "teacher", "entry_filter": "put_near_10_20_offset"}})
    )
    with (attempt_dir / "selected_candidate_replay_gate" / "strict_replay_trades.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session",
                "decision_time",
                "right",
                "offset",
                "entry_ask",
                "stressed_pnl",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "diagnostic_test",
                "session": "2026-01-02",
                "decision_time": decision_time.isoformat(),
                "right": "P",
                "offset": "15",
                "entry_ask": "3.25",
                "stressed_pnl": "-120",
            }
        )
        writer.writerow(
            {
                "split": "diagnostic_test",
                "session": "2026-01-02",
                "decision_time": second_time.isoformat(),
                "right": "P",
                "offset": "15",
                "entry_ask": "3.50",
                "stressed_pnl": "80",
            }
        )
    window = np.ones((30, 7), dtype=np.float32)
    window[-1, 0] = 100.0
    window[-1, 2] = 104.0
    window[-1, 3] = -1.0
    window[-1, 4] = 35.0
    window[-1, 5] = -2.0
    window[-1, 6] = -5.0
    rows = [
        {"decision_time": decision_time, "market_window": window},
        {"decision_time": second_time, "market_window": window},
    ]

    monkeypatch.setattr(diag, "load_rows", lambda _path: rows)

    enriched, blockers = diag.enrich_attempt_trades(search_dir, attempt_id)

    assert blockers == []
    assert len(enriched) == 2
    assert enriched[0]["target_mode"] == "teacher"
    assert enriched[0]["entry_filter"] == "put_near_10_20_offset"
    assert enriched[0]["vwap_side"] == "below_vwap"
    assert enriched[0]["omar_side"] == "omar_neg"
    assert enriched[0]["momentum15_side"] == "mom15_neg"
    assert enriched[0]["put_momentum_alignment"] == "put_aligned_mom15_down"
    assert enriched[0]["trade_index_in_session"] == 1
    assert enriched[1]["trade_index_in_session"] == 2
    assert enriched[0]["session_stressed_pnl"] == -40.0
    assert enriched[0]["first_trade_outcome"] == "first_trade_loss"


def test_session_and_bucket_summary_expose_diagnostic_losses() -> None:
    rows = [
        {
            "attempt_id": "a",
            "split": "diagnostic_test",
            "session": "2026-01-02",
            "decision_time": "2026-01-02T14:35:00+00:00",
            "local_time": "09:35",
            "stressed_pnl": "-120",
            "vwap_side": "below_vwap",
            "omar_side": "omar_neg",
            "momentum15_side": "mom15_neg",
            "put_momentum_alignment": "put_aligned_mom15_down",
            "vwap_gap_bucket": "below_vwap_2_10",
            "time_bucket": "open_0931_0959",
            "trade_index_in_session": 1,
        },
        {
            "attempt_id": "a",
            "split": "diagnostic_test",
            "session": "2026-01-02",
            "decision_time": "2026-01-02T15:05:00+00:00",
            "local_time": "10:05",
            "stressed_pnl": "80",
            "vwap_side": "below_vwap",
            "omar_side": "omar_neg",
            "momentum15_side": "mom15_neg",
            "put_momentum_alignment": "put_aligned_mom15_down",
            "vwap_gap_bucket": "below_vwap_2_10",
            "time_bucket": "morning_1000_1129",
            "trade_index_in_session": 2,
        },
    ]

    sessions = diag.session_summary(rows)
    buckets = diag.bucket_summary(rows)
    first_trade_bucket = [
        row
        for row in buckets
        if row["dimension"] == "trade_index_in_session" and row["bucket"] == "1"
    ][0]

    assert sessions[0]["total_pnl"] == -40.0
    assert sessions[0]["first_trade_stressed_pnl"] == -120.0
    assert sessions[0]["outcome"] == "negative_day"
    assert first_trade_bucket["total_pnl"] == -120.0


def test_top_negative_diagnostic_buckets_classify_split_instability() -> None:
    buckets = [
        {
            "attempt_id": "a",
            "split": "diagnostic_test",
            "dimension": "omar_side",
            "bucket": "omar_neg",
            "trades": 4,
            "total_pnl": -500.0,
            "profit_factor": 0.25,
        },
        {
            "attempt_id": "a",
            "split": "validation",
            "dimension": "omar_side",
            "bucket": "omar_neg",
            "trades": 5,
            "total_pnl": 300.0,
            "profit_factor": 1.5,
        },
        {
            "attempt_id": "a",
            "split": "diagnostic_test",
            "dimension": "vwap_side",
            "bucket": "below_vwap",
            "trades": 2,
            "total_pnl": -1000.0,
            "profit_factor": 0.0,
        },
    ]

    rows = diag.top_negative_diagnostic_buckets(buckets, min_trades=3)

    assert len(rows) == 1
    assert rows[0]["dimension"] == "omar_side"
    assert rows[0]["relation"] == "diagnostic_negative_validation_positive"
    assert rows[0]["validation_total_pnl"] == 300.0
