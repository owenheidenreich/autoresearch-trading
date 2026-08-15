"""Tests for environment diagnostics."""
from __future__ import annotations

from datetime import datetime, timezone

from v4.model.environment_diagnostics import time_bucket


def test_time_bucket_reflects_post_open_and_late_afternoon() -> None:
    assert time_bucket(datetime(2026, 3, 2, 14, 45, tzinfo=timezone.utc)) == "first_30"
    assert time_bucket(datetime(2026, 3, 2, 15, 5, tzinfo=timezone.utc)) == "post_open_morning"
    assert time_bucket(datetime(2026, 3, 2, 17, 0, tzinfo=timezone.utc)) == "midday"
    assert time_bucket(datetime(2026, 3, 2, 19, 30, tzinfo=timezone.utc)) == "late_afternoon"
