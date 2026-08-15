from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from v4.dataset.spxw_0dte_neural import (
    ContractQuotePath,
    LabelPolicy,
    NeuralDatasetConfig,
    _contract_quote_path,
    label_for_policy_two_clock_from_path,
    label_for_policy_two_clock_reference,
)
from v4.model.protocol101_regimen_repair import (
    INT64_MISSING,
    ExitReason,
    InvalidReason,
    Protocol101DuplicatePathQuoteIdentityError,
    build_exit_quote_age_report,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="America/New_York").tz_convert("UTC")


def _quotes(rows: list[tuple[str, float | None, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_id": ["c1"] * len(rows),
            "quote_time": [_ts(row[0]) for row in rows],
            "bid": [row[1] for row in rows],
            "mid": [row[2] for row in rows],
        }
    )


def _labels(
    rows: list[tuple[str, float | None, float]],
    *,
    decision: str = "2025-01-02 10:00:00",
    policy: LabelPolicy = LabelPolicy(0.50, 1.00, 10),
    forced_flat: time = time(15, 55),
):
    frame = _quotes(rows)
    decision_time = _ts(decision)
    entry = pd.Series(
        {"contract_id": "c1", "ask": 2.0, "mid": 1.9}
    )
    config = NeuralDatasetConfig(
        label_policies=(policy,),
        forced_flat_before=forced_flat,
    )
    reference = label_for_policy_two_clock_reference(
        frame,
        decision_time=decision_time,
        entry_row=entry,
        policy=policy,
        config=config,
    )
    vector = label_for_policy_two_clock_from_path(
        _contract_quote_path(frame, enforce_unique_path=True),
        decision_time=decision_time,
        entry_ask=2.0,
        entry_mid=1.9,
        policy=policy,
        config=config,
    )
    return reference, vector


def _assert_exact(left, right) -> None:
    for name in left.__dataclass_fields__:
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, float) and np.isnan(a):
            assert np.isnan(b), name
        else:
            assert a == b, name


def test_stop_clock_and_pnl_RX_STOP_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 10:01:00", 0.90, 1.00)]
    )
    _assert_exact(reference, vector)
    assert reference.exit_reason_code == ExitReason.STOP_LOSS
    assert reference.source_exit_quote_time_ns == reference.realized_exit_time_ns
    assert reference.exit_quote_age_ms == 0.0
    assert reference.net_pnl == pytest.approx(-110.0, abs=1e-12)


def test_target_clock_and_pnl_RX_TARGET_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 10:01:00", 4.10, 4.20)]
    )
    _assert_exact(reference, vector)
    assert reference.exit_reason_code == ExitReason.TAKE_PROFIT
    assert reference.net_pnl == pytest.approx(210.0, abs=1e-12)


@pytest.mark.parametrize("bid", [None, 0.0], ids=["absent", "zero"])
def test_no_bid_clock_RX_NOBID_001(bid: float | None) -> None:
    reference, vector = _labels(
        [("2025-01-02 10:01:00", bid, 0.10)]
    )
    _assert_exact(reference, vector)
    assert reference.exit_reason_code == ExitReason.NO_BID_STOP
    assert reference.executable_exit_bid == 0.0
    assert reference.exit_quote_age_ms == 0.0


def test_max_hold_uses_latest_prior_quote_RX_MAXHOLD_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 10:09:00", 1.50, 1.55)]
    )
    _assert_exact(reference, vector)
    assert reference.exit_reason_code == ExitReason.MAX_HOLD
    assert reference.exit_quote_age_ms == 60_000.0
    assert reference.net_pnl == -50.0


def test_forced_flat_uses_latest_prior_quote_RX_FLAT_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 15:54:00", 1.50, 1.55)],
        decision="2025-01-02 15:40:00",
        policy=LabelPolicy(1.0, 99.0, 384),
    )
    _assert_exact(reference, vector)
    assert reference.exit_reason_code == ExitReason.FORCED_FLAT
    assert reference.exit_quote_age_ms == 60_000.0


def test_exact_deadline_quote_RX_DEADLINE_QUOTE_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 10:10:00", 1.50, 1.55)]
    )
    _assert_exact(reference, vector)
    assert reference.source_exit_quote_time_ns == reference.realized_exit_time_ns
    assert reference.exit_quote_age_ms == 0.0


def test_no_exact_deadline_remains_valid_RX_NO_EXACT_DEADLINE_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 10:09:59", 1.50, 1.55)]
    )
    _assert_exact(reference, vector)
    assert reference.valid
    assert reference.exit_quote_age_ms == 1_000.0


def test_missing_future_path_RX_MISSING_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 10:00:00", 1.50, 1.55)]
    )
    _assert_exact(reference, vector)
    assert not reference.valid
    assert (
        reference.invalid_reason_code
        == InvalidReason.NO_CAUSAL_FUTURE_QUOTE_AT_OR_BEFORE_DEADLINE
    )
    assert reference.policy_deadline_ns == INT64_MISSING


def test_entry_timestamp_quote_excluded_RX_SAMETS_001() -> None:
    reference, vector = _labels(
        [
            ("2025-01-02 10:00:00", 0.10, 0.20),
            ("2025-01-02 10:01:00", 1.50, 1.55),
        ]
    )
    _assert_exact(reference, vector)
    assert reference.source_exit_quote_time_ns == _ts(
        "2025-01-02 10:01:00"
    ).value


def test_next_session_path_is_invalid_RX_SESSION_001() -> None:
    reference, vector = _labels(
        [("2025-01-03 09:30:00", 1.50, 1.55)]
    )
    _assert_exact(reference, vector)
    assert not reference.valid


def test_duplicate_path_fails_closed_RX_DUPPATH_001() -> None:
    frame = _quotes(
        [
            ("2025-01-02 10:01:00", 1.50, 1.55),
            ("2025-01-02 10:01:00", 1.40, 1.45),
        ]
    )
    with pytest.raises(Protocol101DuplicatePathQuoteIdentityError):
        _contract_quote_path(frame, enforce_unique_path=True)


def test_pnl_uses_source_not_occupancy_clock_RX_PNL_CLOCK_001() -> None:
    reference, vector = _labels(
        [
            ("2025-01-02 10:09:00", 1.50, 1.55),
            ("2025-01-02 10:11:00", 9.00, 9.00),
        ]
    )
    _assert_exact(reference, vector)
    assert reference.net_pnl == -50.0
    assert reference.realized_exit_time_ns == _ts(
        "2025-01-02 10:10:00"
    ).value


def test_reference_vector_and_age_formula_RX_REF_VEC_001_RX_AGE_FORMULA_001() -> None:
    scenarios = [
        [("2025-01-02 10:01:00", 0.90, 1.00)],
        [("2025-01-02 10:01:00", 4.10, 4.20)],
        [("2025-01-02 10:09:00", 1.50, 1.55)],
    ]
    for rows in scenarios:
        reference, vector = _labels(rows)
        _assert_exact(reference, vector)
        if reference.valid:
            expected = (
                reference.realized_exit_time_ns
                - reference.source_exit_quote_time_ns
            ) / 1_000_000.0
            assert reference.exit_quote_age_ms == expected >= 0.0


def test_large_quote_age_is_diagnostic_only_RX_AGE_NOGATE_001() -> None:
    reference, vector = _labels(
        [("2025-01-02 10:00:01", 1.50, 1.55)]
    )
    _assert_exact(reference, vector)
    assert reference.valid
    assert reference.exit_quote_age_ms == 599_000.0


def test_quote_age_report_dimensions_RX_AGE_REPORT_001() -> None:
    report = build_exit_quote_age_report(
        [
            {
                "policy_index": 0,
                "session": "2025-01-02",
                "valid": True,
                "label_realized_exit_time_ns": _ts(
                    "2025-01-02 10:00:00"
                ).value,
                "label_policy_deadline_ns": _ts(
                    "2025-01-02 10:00:00"
                ).value,
                "label_exit_quote_age_ms": 0.0,
            },
            {
                "policy_index": 0,
                "session": "2025-01-02",
                "valid": True,
                "label_realized_exit_time_ns": _ts(
                    "2025-01-02 10:01:00"
                ).value,
                "label_policy_deadline_ns": _ts(
                    "2025-01-02 10:01:00"
                ).value,
                "label_exit_quote_age_ms": 60_000.0,
            },
            {
                "policy_index": 1,
                "session": "2025-01-03",
                "valid": False,
                "label_policy_deadline_ns": INT64_MISSING,
                "label_exit_quote_age_ms": np.nan,
            },
        ]
    )
    assert report["gate"] is False
    assert report["rejection_threshold_ms"] is None
    assert report["record_count"] == 3
    first = report["groups"][0]
    assert first["valid_count"] == 2
    assert first["zero_age_share"] == 0.5
    assert first["minimum_ms"] == 0.0
    assert first["maximum_ms"] == 60_000.0
