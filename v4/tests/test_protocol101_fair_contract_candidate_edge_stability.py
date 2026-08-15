"""Tests for fair-contract candidate-edge stability diagnostics."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from v4.model.supervised_pilot import DecisionCandidates
from v4.scripts import run_protocol101_fair_contract_candidate_edge_stability as diag


def _decision(decision_time: datetime, labels: np.ndarray | None = None) -> DecisionCandidates:
    return DecisionCandidates(
        session="2026-03-03",
        decision_time=decision_time,
        features=np.zeros((4, 71), dtype=np.float32),
        labels=(
            np.array([100.0, -50.0, 80.0, 20.0], dtype=np.float32)
            if labels is None
            else labels.astype(np.float32)
        ),
        offsets=np.array([10.0, 15.0, -20.0, 35.0], dtype=np.float32),
        rights=np.array(["C", "P", "P", "P"], dtype=object),
        market_last=np.array([100.0, 0.0, 104.0, -1.0, 35.0, -2.0, -5.0], dtype=np.float32),
        entry_asks=np.array([2.0, 3.5, 4.0, 5.0], dtype=np.float32),
    )


def test_candidate_buckets_use_live_causal_context() -> None:
    row = diag.candidate_buckets(_decision(datetime(2026, 3, 3, 14, 45, tzinfo=timezone.utc)), 1)

    assert row["right"] == "P"
    assert row["time_bucket"] == "open_0931_0959"
    assert row["offset_bucket"] == "near_10_20"
    assert row["premium_bucket"] == "3_to_7_5"
    assert row["vwap_side"] == "below_vwap"
    assert row["omar_side"] == "omar_neg"
    assert row["momentum15_side"] == "mom15_neg"
    assert row["vwap_gap_bucket"] == "below_vwap_2_10"
    assert row["range_bucket"] == "range_20_45"


def test_oracle_trades_apply_filter_cooldown_and_session_cap() -> None:
    first = _decision(datetime(2026, 3, 3, 14, 35, tzinfo=timezone.utc))
    second = _decision(
        datetime(2026, 3, 3, 14, 40, tzinfo=timezone.utc),
        labels=np.array([40.0, 200.0, 180.0, 20.0], dtype=np.float32),
    )
    third = _decision(
        datetime(2026, 3, 3, 14, 51, tzinfo=timezone.utc),
        labels=np.array([40.0, 190.0, 180.0, 20.0], dtype=np.float32),
    )

    trades = diag.oracle_trades_for_filter(
        [first, second, third],
        entry_filter="put_near_after_0940",
        positive_label_threshold=20.0,
        cooldown_minutes=10,
        max_trades_per_session=1,
    )

    assert len(trades) == 1
    assert trades[0].decision_time == "2026-03-03T14:40:00+00:00"
    assert trades[0].right == "P"
    assert trades[0].pnl == 200.0


def test_stability_pairs_classify_validation_diagnostic_signs() -> None:
    stats = {
        ("near_10_20_offset", "validation", "right", "P"): diag.LabelStats(),
        ("near_10_20_offset", "diagnostic_test", "right", "P"): diag.LabelStats(),
        ("near_10_20_offset", "validation", "right", "C"): diag.LabelStats(),
        ("near_10_20_offset", "diagnostic_test", "right", "C"): diag.LabelStats(),
    }
    for value in (100.0, 50.0, -20.0):
        stats[("near_10_20_offset", "validation", "right", "P")].add(value, positive_threshold=20.0)
    for value in (80.0, 40.0, -10.0):
        stats[("near_10_20_offset", "diagnostic_test", "right", "P")].add(value, positive_threshold=20.0)
    for value in (100.0, -20.0, -10.0):
        stats[("near_10_20_offset", "validation", "right", "C")].add(value, positive_threshold=20.0)
    for value in (-80.0, -40.0, 10.0):
        stats[("near_10_20_offset", "diagnostic_test", "right", "C")].add(value, positive_threshold=20.0)

    rows = diag.stability_pairs(stats, min_candidates=3)
    by_bucket = {row["bucket"]: row for row in rows}

    assert by_bucket["P"]["status"] == "stable_positive_label_bucket"
    assert by_bucket["C"]["status"] == "validation_positive_diagnostic_negative"
