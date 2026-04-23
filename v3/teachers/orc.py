"""Opening Range Continuation teacher.

Triggers a directional signal when the underlying breaks the first-15 opening
range in the direction that agrees with VWAP trend. Deliberately minimal:

- NO VIX regime filter
- NO first-15 range-size band
- NO volume threshold

Those features remain in `BarContext` so the logger records them as soft
inputs, but they do not gate the teacher's signal in v1. The plan explicitly
calls out that soft features earn gate status only by proving their worth
in out-of-sample ablations, not by being crammed into a brittle gate stack
upfront.
"""
from __future__ import annotations

from dataclasses import dataclass

from v3.teachers.base import BarContext, Teacher, TeacherAction


@dataclass
class ORCTeacher(Teacher):
    name: str = "orc"
    entry_window_start_min: int = 15  # 09:45 ET, first bar after the first-15 range is fixed
    entry_window_end_min: int = 90     # 11:00 ET

    def evaluate(self, bar: BarContext) -> TeacherAction:
        if not self.in_window(bar.minute_of_session):
            return TeacherAction.NO_SIGNAL

        breaks_high = bar.close > bar.first15_high
        breaks_low = bar.close < bar.first15_low
        vwap_trend_up = bar.close > bar.vwap and bar.vwap_slope > 0.0
        vwap_trend_dn = bar.close < bar.vwap and bar.vwap_slope < 0.0

        if breaks_high and vwap_trend_up:
            if bar.sigma_pos is not None and bar.sigma_pos > 0.0:
                return TeacherAction.NO_SIGNAL
            return TeacherAction.BUY_CALL
        if breaks_low and vwap_trend_dn:
            if bar.sigma_pos is not None and bar.sigma_pos < 0.0:
                return TeacherAction.NO_SIGNAL
            return TeacherAction.BUY_PUT
        return TeacherAction.NO_SIGNAL
