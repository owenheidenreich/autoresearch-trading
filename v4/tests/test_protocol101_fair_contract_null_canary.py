"""Tests for Protocol101 fair-contract null/canary plumbing."""
from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from v4.model.supervised_pilot import DecisionCandidates, PilotConfig
from v4.scripts.run_protocol101_fair_contract_null_canary import (
    label_canaries,
    label_summary,
    summarize_mask,
    summarize_timing,
)


def _processed_row(decision_time: datetime) -> dict:
    return {
        "decision_time": decision_time,
        "source_context_time": decision_time - timedelta(minutes=1),
        "source_quote_time": decision_time - timedelta(minutes=1),
    }


def test_summarize_timing_accepts_0932_359_convention(tmp_path: Path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    first = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    rows = [_processed_row(first + timedelta(minutes=idx)) for idx in range(359)]
    with path.open("wb") as handle:
        pickle.dump(rows, handle)

    timing_rows, summary, blockers = summarize_timing(
        {"train": [path]},
        expected_first="09:32",
        expected_last="15:30",
        expected_rows=359,
    )

    assert blockers == []
    assert summary["session_count"] == 1
    assert summary["row_count_min"] == 359
    assert summary["first_decision_et_values"] == ["09:32"]
    assert summary["last_decision_et_values"] == ["15:30"]
    assert timing_rows[0]["context_lag_min"] == 1.0


def test_summarize_timing_blocks_wrong_first_decision(tmp_path: Path) -> None:
    path = tmp_path / "2026-01-02.pkl"
    first = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    rows = [_processed_row(first + timedelta(minutes=idx)) for idx in range(360)]
    with path.open("wb") as handle:
        pickle.dump(rows, handle)

    _timing_rows, _summary, blockers = summarize_timing(
        {"train": [path]},
        expected_first="09:32",
        expected_last="15:30",
        expected_rows=359,
    )

    assert "unexpected_row_count:train:2026-01-02:360" in blockers
    assert "unexpected_first_decision_et:train:2026-01-02:09:31" in blockers


def test_summarize_mask_zeroes_sensitive_features_only() -> None:
    features = np.ones((2, 24), dtype=np.float32)
    decision = DecisionCandidates(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
        features=features,
        labels=np.asarray([25.0, -10.0], dtype=np.float32),
        offsets=np.asarray([0.0, 10.0], dtype=np.float32),
        rights=np.asarray(["C", "P"], dtype=object),
        market_last=np.ones(7, dtype=np.float32),
    )

    summary = summarize_mask({"train": [decision]})

    assert summary["sensitive_zeroed"] is True
    assert summary["sensitive_max_abs"] == 0.0
    assert summary["non_sensitive_nonzero"] is True


def test_label_summary_and_canaries_are_non_degenerate() -> None:
    decisions = [
        DecisionCandidates(
            session="2026-01-02",
            decision_time=datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
            features=np.ones((2, 24), dtype=np.float32),
            labels=np.asarray([100.0, -50.0], dtype=np.float32),
            offsets=np.asarray([0.0, 10.0], dtype=np.float32),
            rights=np.asarray(["C", "P"], dtype=object),
            market_last=np.ones(7, dtype=np.float32),
            entry_asks=np.asarray([1.0, 1.0], dtype=np.float32),
        )
    ]

    labels = label_summary(decisions)
    canaries = label_canaries(decisions, config=PilotConfig(cooldown_minutes=10))

    assert labels["positive_labels"] == 1
    assert labels["negative_labels"] == 1
    assert labels["label_std"] > 0.0
    assert canaries["label_oracle_positive"]["trades"] == 1
    assert canaries["label_oracle_positive"]["total_pnl"] > 0.0
    assert canaries["inverted_label_negative"]["trades"] == 1
