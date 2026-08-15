"""Tests for compound fair-contract selector diagnostics."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from v4.model.supervised_pilot import DecisionCandidates
from v4.scripts import run_protocol101_fair_contract_two_stage_selector_diagnostic as diag


def _decision(
    decision_time: datetime,
    *,
    market_last: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    entry_asks: np.ndarray | None = None,
) -> DecisionCandidates:
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
        market_last=(
            np.array([105.0, 0.0, 100.0, -1.0, 35.0, -2.0, -5.0], dtype=np.float32)
            if market_last is None
            else market_last.astype(np.float32)
        ),
        entry_asks=(
            np.array([2.0, 8.0, 16.0, 5.0], dtype=np.float32)
            if entry_asks is None
            else entry_asks.astype(np.float32)
        ),
    )


def test_gate_mask_applies_base_filter_and_compound_predicates() -> None:
    spec = diag.gate_specs_by_name()["base_plus_omar_neg_premium_7_5_to_15"]
    decision = _decision(datetime(2026, 3, 3, 16, 45, tzinfo=timezone.utc))

    assert np.array_equal(
        diag.gate_mask(decision, spec),
        np.array([False, True, False, False]),
    )

    omar_positive = _decision(
        datetime(2026, 3, 3, 16, 45, tzinfo=timezone.utc),
        market_last=np.array([105.0, 0.0, 100.0, 1.0, 35.0, -2.0, -5.0], dtype=np.float32),
    )

    assert not diag.gate_mask(omar_positive, spec).any()


def test_premium_gte_7_5_predicate_combines_mid_and_high_premium_buckets() -> None:
    assert diag.predicate_matches("premium_gte_7_5", {"premium_bucket": "7_5_to_15"})
    assert diag.predicate_matches("premium_gte_7_5", {"premium_bucket": "gte_15"})
    assert not diag.predicate_matches("premium_gte_7_5", {"premium_bucket": "3_to_7_5"})


def test_oracle_trades_for_gate_obeys_cooldown_and_session_cap() -> None:
    spec = diag.gate_specs_by_name()["base_plus_omar_neg"]
    first = _decision(datetime(2026, 3, 3, 16, 40, tzinfo=timezone.utc))
    second = _decision(
        datetime(2026, 3, 3, 16, 45, tzinfo=timezone.utc),
        labels=np.array([40.0, 200.0, 180.0, 20.0], dtype=np.float32),
    )
    third = _decision(
        datetime(2026, 3, 3, 16, 56, tzinfo=timezone.utc),
        labels=np.array([40.0, 190.0, 180.0, 20.0], dtype=np.float32),
    )

    trades = diag.oracle_trades_for_gate(
        [first, second, third],
        gate=spec,
        positive_label_threshold=20.0,
        cooldown_minutes=10,
        max_trades_per_session=1,
    )

    assert len(trades) == 1
    assert trades[0].decision_time == "2026-03-03T16:40:00+00:00"
    assert trades[0].right == "P"


def test_stability_rows_rank_stable_compound_gate() -> None:
    label_rows = [
        {
            "split": "validation",
            "gate": "stable",
            "candidate_count": 120,
            "avg_label_pnl": 30.0,
            "profit_factor": 1.5,
        },
        {
            "split": "diagnostic_test",
            "gate": "stable",
            "candidate_count": 130,
            "avg_label_pnl": 25.0,
            "profit_factor": 1.4,
        },
        {
            "split": "validation",
            "gate": "flip",
            "candidate_count": 120,
            "avg_label_pnl": 30.0,
            "profit_factor": 1.5,
        },
        {
            "split": "diagnostic_test",
            "gate": "flip",
            "candidate_count": 130,
            "avg_label_pnl": -10.0,
            "profit_factor": 0.8,
        },
    ]
    oracle_rows = [
        {"split": "validation", "gate": "stable", "trades": 10, "total_pnl": 1000.0, "profit_factor": 2.0},
        {"split": "diagnostic_test", "gate": "stable", "trades": 10, "total_pnl": 900.0, "profit_factor": 1.8},
        {"split": "validation", "gate": "flip", "trades": 10, "total_pnl": 1000.0, "profit_factor": 2.0},
        {"split": "diagnostic_test", "gate": "flip", "trades": 10, "total_pnl": -100.0, "profit_factor": 0.8},
    ]

    rows = diag.stability_rows(label_rows, oracle_rows, min_candidates=100)
    by_gate = {row["gate"]: row for row in rows}

    assert by_gate["stable"]["status"] == "stable_positive_compound_gate"
    assert by_gate["flip"]["status"] == "validation_positive_diagnostic_negative"
    assert rows[0]["gate"] == "stable"
