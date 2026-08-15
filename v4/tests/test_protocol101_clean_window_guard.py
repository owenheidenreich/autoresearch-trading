from __future__ import annotations

from datetime import datetime, timedelta, timezone

from v4.live.protocol101_clean_window_guard import Protocol101CleanWindowGuard


def test_guard_requires_full_warmup_before_entry() -> None:
    guard = Protocol101CleanWindowGuard(warmup_minutes=15)
    start = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)

    statuses = [
        guard.observe(decision_time=start + timedelta(minutes=index), healthy=True)
        for index in range(16)
    ]

    assert not statuses[14].eligible
    assert statuses[14].warmup_remaining_minutes == 1
    assert statuses[15].eligible


def test_guard_resets_after_unhealthy_minute() -> None:
    guard = Protocol101CleanWindowGuard(warmup_minutes=2)
    start = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)
    guard.observe(decision_time=start, healthy=True)
    guard.observe(decision_time=start + timedelta(minutes=1), healthy=True)

    blocked = guard.observe(
        decision_time=start + timedelta(minutes=2),
        healthy=False,
        reasons=("stale_quote",),
    )
    restarted = guard.observe(
        decision_time=start + timedelta(minutes=3), healthy=True
    )

    assert not blocked.eligible
    assert blocked.reasons == ("stale_quote",)
    assert restarted.healthy_streak_minutes == 1
    assert not restarted.eligible


def test_guard_treats_missing_minute_as_interruption() -> None:
    guard = Protocol101CleanWindowGuard(warmup_minutes=1)
    start = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)
    guard.observe(decision_time=start, healthy=True)

    status = guard.observe(
        decision_time=start + timedelta(minutes=2), healthy=True
    )

    assert status.healthy_streak_minutes == 1
    assert not status.eligible


def test_repeated_observation_does_not_advance_warmup() -> None:
    guard = Protocol101CleanWindowGuard(warmup_minutes=1)
    minute = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)

    first = guard.observe(decision_time=minute, healthy=True)
    repeated = guard.observe(
        decision_time=minute + timedelta(seconds=40), healthy=True
    )

    assert first.healthy_streak_minutes == 1
    assert repeated.healthy_streak_minutes == 1
    assert not repeated.eligible
