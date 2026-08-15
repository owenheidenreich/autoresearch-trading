"""Causal entry abstention after unhealthy live-data minutes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


@dataclass(frozen=True)
class CleanWindowStatus:
    eligible: bool
    healthy: bool
    healthy_streak_minutes: int
    warmup_remaining_minutes: int
    reasons: tuple[str, ...]
    decision_minute_utc: str


class Protocol101CleanWindowGuard:
    """Require a fresh consecutive-minute warm-up before allowing entries."""

    def __init__(self, *, warmup_minutes: int = 15) -> None:
        if warmup_minutes < 0:
            raise ValueError("warmup_minutes must be non-negative")
        self.warmup_minutes = int(warmup_minutes)
        self.healthy_streak_minutes = 0
        self.last_observed_minute: datetime | None = None
        self.last_reasons: tuple[str, ...] = ("runtime_start",)
        self.interruption_count = 0

    @staticmethod
    def _minute(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)

    def interrupt(self, reason: str) -> None:
        self.healthy_streak_minutes = 0
        self.last_observed_minute = None
        self.last_reasons = (str(reason),)
        self.interruption_count += 1

    def observe(
        self,
        *,
        decision_time: datetime,
        healthy: bool,
        reasons: Iterable[str] = (),
    ) -> CleanWindowStatus:
        minute = self._minute(decision_time)
        normalized_reasons = tuple(sorted({str(reason) for reason in reasons if reason}))

        if not healthy:
            self.healthy_streak_minutes = 0
            self.last_observed_minute = minute
            self.last_reasons = normalized_reasons or ("unhealthy_market_data",)
            self.interruption_count += 1
        elif self.last_observed_minute == minute:
            self.last_reasons = ()
        elif (
            self.last_observed_minute is not None
            and minute - self.last_observed_minute == timedelta(minutes=1)
            and not self.last_reasons
        ):
            self.healthy_streak_minutes += 1
            self.last_observed_minute = minute
            self.last_reasons = ()
        else:
            self.healthy_streak_minutes = 1
            self.last_observed_minute = minute
            self.last_reasons = ()

        eligible = bool(
            healthy and self.healthy_streak_minutes > self.warmup_minutes
        )
        remaining = max(
            0, self.warmup_minutes + 1 - self.healthy_streak_minutes
        )
        status_reasons = self.last_reasons
        if healthy and not eligible:
            status_reasons = ("clean_window_warmup",)
        return CleanWindowStatus(
            eligible=eligible,
            healthy=bool(healthy),
            healthy_streak_minutes=int(self.healthy_streak_minutes),
            warmup_remaining_minutes=int(remaining),
            reasons=status_reasons,
            decision_minute_utc=minute.isoformat(),
        )
