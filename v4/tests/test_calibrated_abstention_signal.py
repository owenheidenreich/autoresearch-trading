"""Tests for validation-calibrated abstention helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from v4.scripts.evaluate_calibrated_abstention_signal import (
    ActionPredictionRecord,
    CalibratedConfig,
    ScoreCalibration,
    fit_score_calibration,
    simulate_calibrated_policy,
    split_validation_by_session,
)


def _record(
    *,
    minute_utc: int,
    score: float,
    pnl: float,
    session: str = "2026-03-02",
) -> ActionPredictionRecord:
    return ActionPredictionRecord(
        session=session,
        decision_time=datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        + timedelta(minutes=minute_utc),
        action=1,
        right="C",
        offset=0.0,
        pnl=pnl,
        top_score=score,
        no_trade_score=0.0,
        edge_vs_no_trade=score,
        directional_margin=score,
        edge_plus_directional_margin=score,
        worst_case_margin=score,
    )


def test_score_calibration_is_monotonic() -> None:
    records = [
        _record(minute_utc=65, score=0.1, pnl=-100.0),
        _record(minute_utc=66, score=0.2, pnl=-50.0),
        _record(minute_utc=67, score=0.8, pnl=75.0),
        _record(minute_utc=68, score=0.9, pnl=125.0),
    ]

    calibration = fit_score_calibration(records, "top_score", bins=2)

    assert calibration.expected_pnl(0.1) <= calibration.expected_pnl(0.9)
    assert calibration.win_rate(0.1) <= calibration.win_rate(0.9)


def test_calibrated_policy_checks_time_before_cooldown() -> None:
    config = CalibratedConfig(
        score_name="top_score",
        min_calibrated_pnl=10.0,
        min_calibrated_win_rate=0.50,
        min_raw_edge=0.0,
        time_filter="post_open_only",
        allowed_buckets=("post_open_morning",),
        max_trades_per_day=99,
        daily_loss_stop=None,
    )
    calibration = ScoreCalibration(
        score_name="top_score",
        edges=(-1.0, 100.0),
        expected_pnl_by_bin=(50.0,),
        win_rate_by_bin=(0.75,),
        count_by_bin=(2,),
    )
    records = [
        _record(minute_utc=45, score=10.0, pnl=-100.0),  # 09:45 ET, disallowed.
        _record(minute_utc=65, score=10.0, pnl=50.0),  # 10:05 ET, allowed.
    ]

    trades = simulate_calibrated_policy(
        records,
        {"top_score": calibration},
        config=config,
        cooldown_minutes=45,
        strategy="test",
    )

    assert len(trades) == 1
    assert trades[0].pnl == 50.0
    assert trades[0].decision_time.endswith("15:05:00+00:00")


def test_split_validation_by_session_preserves_chronology() -> None:
    records = [
        _record(minute_utc=65, score=1.0, pnl=1.0, session="2026-02-02"),
        _record(minute_utc=65, score=1.0, pnl=1.0, session="2026-02-03"),
        _record(minute_utc=65, score=1.0, pnl=1.0, session="2026-02-04"),
        _record(minute_utc=65, score=1.0, pnl=1.0, session="2026-02-05"),
    ]

    calibration, selection = split_validation_by_session(records)  # type: ignore[arg-type]

    assert {x.session for x in calibration} == {"2026-02-02", "2026-02-03"}
    assert {x.session for x in selection} == {"2026-02-04", "2026-02-05"}
