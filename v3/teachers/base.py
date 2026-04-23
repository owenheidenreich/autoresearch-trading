"""Teacher base: action enum, bar context, abstract base class.

Teachers are pure functions over `BarContext` that return a `TeacherAction`.
They do NOT:
- pick contracts (Layer 0 handles this given the signal direction)
- hold session state (callers precompute first-15 range, VWAP, etc.)
- reference account equity or guardrails (they are signal-only)

This keeps the logger's job simple: for every eligible bar, call
`teacher.evaluate(bar)` on each registered teacher and record the result.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TeacherAction(Enum):
    NO_SIGNAL = "no_signal"
    BUY_CALL = "buy_call"
    BUY_PUT = "buy_put"


@dataclass(frozen=True)
class BarContext:
    """Per-bar inputs visible to a Teacher.

    All session-level aggregates (first-15 opening range, VWAP, volume ratio,
    break-recency counters) are expected to be precomputed by the harness and
    passed in. This makes teachers trivially testable and keeps state-
    management responsibility out of the teacher itself.

    The two `bars_since_break_*` counters use `-1` as a sentinel meaning
    "no such break has occurred this session." Zero means the current bar is
    the break bar; positive values count bars since the most recent break.
    """

    minute_of_session: int
    close: float
    vwap: float
    vwap_slope: float
    volume_ratio: float
    first15_high: float
    first15_low: float
    first15_range_pct: float
    bars_since_break_above_first15: int = -1
    bars_since_break_below_first15: int = -1
    sigma_pos: Optional[float] = None


class Teacher(ABC):
    """Abstract base for Stage 1 teacher playbooks.

    Subclasses declare `name`, `entry_window_start_min`, `entry_window_end_min`,
    and implement `evaluate`. Minutes are measured from 09:30 ET (minute 0).
    """

    name: str
    entry_window_start_min: int
    entry_window_end_min: int

    def in_window(self, minute_of_session: int) -> bool:
        return self.entry_window_start_min <= minute_of_session <= self.entry_window_end_min

    @abstractmethod
    def evaluate(self, bar: BarContext) -> TeacherAction: ...
