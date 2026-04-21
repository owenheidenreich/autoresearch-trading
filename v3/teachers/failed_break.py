"""Failed-Break Reversal teacher.

Fires when the underlying recently broke the first-15 opening range in one
direction, then reclaimed the range. The signal is OPPOSITE the direction of
the failed break: an upside break that failed is a long-put setup; a downside
break that failed is a long-call setup.

v1 is purely structural — it relies on break-recency and current-bar range
position, nothing else. No VWAP filter, no volume filter, no range-size band.
The logger records all those features as soft inputs so their contribution
can be measured post-hoc via ablation; the plan explicitly warns against
piling gates onto the teacher at design time.

Why the lookback of 5 bars: a break that fails immediately (1-2 bars) may be
noise; a break that fails after 10+ bars is no longer a coherent reversal
pattern — too much price action has intervened. 5 bars is the middle of that
range and can be revisited after the opportunity surface is logged.
"""
from __future__ import annotations

from dataclasses import dataclass

from v3.teachers.base import BarContext, Teacher, TeacherAction


@dataclass
class FailedBreakTeacher(Teacher):
    name: str = "failed_break"
    entry_window_start_min: int = 30   # 10:00 ET — need some time for an initial break to form
    entry_window_end_min: int = 120    # 11:30 ET — stop before lunch theta zone
    lookback_bars: int = 5

    def evaluate(self, bar: BarContext) -> TeacherAction:
        if not self.in_window(bar.minute_of_session):
            return TeacherAction.NO_SIGNAL

        inside_range = bar.first15_low <= bar.close <= bar.first15_high
        if not inside_range:
            return TeacherAction.NO_SIGNAL

        had_upside_break = 1 <= bar.bars_since_break_above_first15 <= self.lookback_bars
        had_downside_break = 1 <= bar.bars_since_break_below_first15 <= self.lookback_bars

        # If both sides broke recently, the range is contested — no clean reversal thesis.
        if had_upside_break and had_downside_break:
            return TeacherAction.NO_SIGNAL
        if had_upside_break:
            return TeacherAction.BUY_PUT
        if had_downside_break:
            return TeacherAction.BUY_CALL
        return TeacherAction.NO_SIGNAL
