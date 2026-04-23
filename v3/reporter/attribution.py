"""Attribution taxonomy reporter.

Decomposes the gap between an ideal policy (opportunity oracle) and the
actual teacher-based policy into the four named diagnostics from the plan:

1. `guardrail_suppression` — oracle had an opportunity on a bar, but zero
   contracts passed the hard rails. Policy couldn't have entered even if it
   wanted to. (Already quantified by feasibility reporter; shown here for
   completeness on a per-session basis.)

2. `abstention_gap` — oracle says this bar was the session's best trade,
   but no teacher triggered on that bar. Policy stayed out of a real setup.

3. `side_error_gap` — oracle bar had a teacher trigger, but the teacher's
   direction (call/put) was opposite the oracle's direction. Policy took
   the wrong side.

4. `exit_gap` — policy entered on the oracle bar in the right direction,
   but its exit (here: hold to session end) underperformed the best
   hindsight exit on the same contract.

Also reports:
- `selection_gap` — policy entered the right bar and direction, but the
  teacher's contract tiebreak picked a different strike than the oracle's.
  Measured as (opportunity oracle realized PnL) − (exit headroom best PnL
  on the actual teacher-selected contract).

The "policy" in v1 = teacher-driven: enter on first trigger, select contract
via guardrail + gamma/theta tiebreak, exit at session-end time stop. This is
not the target Layer-3 learned policy — it's a mechanical baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from v3.logger.schema import BarRecord, DayLog


@dataclass
class AttributionCounts:
    """Per-session bucket counts and dollar accumulations."""

    n_sessions: int = 0
    n_sessions_with_oracle: int = 0

    # Session-level outcome buckets (each session falls into exactly one)
    guardrail_suppression_count: int = 0
    abstention_count: int = 0
    side_error_count: int = 0
    entered_right_count: int = 0   # policy entered oracle bar in right direction
    entered_same_bar_wrong_contract_count: int = 0  # contract-selection miss

    # Dollar totals (all on oracle bar + right direction entries only)
    oracle_total_pnl: float = 0.0
    exit_headroom_total_pnl: float = 0.0  # hindsight best exit on selected contract
    time_stop_total_pnl: float = 0.0       # naive hold-to-end baseline
    stop_target_total_pnl: float = 0.0     # -35%/+60% mechanical baseline

    # Aggregate gaps (dollars)
    selection_gap_total: float = 0.0       # oracle − exit_headroom
    exit_gap_vs_time_stop: float = 0.0     # exit_headroom − time_stop
    exit_gap_vs_stop_target: float = 0.0   # exit_headroom − stop_target
    stop_target_vs_time_stop: float = 0.0  # stop_target − time_stop (does the mechanical rule help?)


@dataclass
class AttributionReport:
    counts: AttributionCounts = field(default_factory=AttributionCounts)
    per_session: list[dict] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        c = self.counts
        lines = [
            f"Sessions processed:               {c.n_sessions}",
            f"  with opportunity oracle entry:  {c.n_sessions_with_oracle}",
            "",
            "Outcome on oracle bar:",
            f"  guardrail_suppression:          {c.guardrail_suppression_count}",
            f"  abstention (no teacher trig):   {c.abstention_count}",
            f"  side_error (wrong direction):   {c.side_error_count}",
            f"  entered (right bar+direction):  {c.entered_right_count}",
            "",
            "Dollar totals (on right-bar+right-direction entries only):",
            f"  oracle session PnL:                ${c.oracle_total_pnl:>12,.0f}",
            f"  exit-headroom on actual selection: ${c.exit_headroom_total_pnl:>12,.0f}",
            f"  stop/target baseline (-35/+60):    ${c.stop_target_total_pnl:>12,.0f}",
            f"  time-stop baseline (hold to end):  ${c.time_stop_total_pnl:>12,.0f}",
            "",
            "Gaps (positive = room for improvement):",
            f"  selection_gap:       ${c.selection_gap_total:>12,.0f}   "
            f"(oracle − exit-headroom: hindsight contract choice)",
            f"  exit_gap vs hold:    ${c.exit_gap_vs_time_stop:>12,.0f}   "
            f"(exit-headroom − time-stop: hindsight vs naive hold)",
            f"  exit_gap vs -35/+60: ${c.exit_gap_vs_stop_target:>12,.0f}   "
            f"(exit-headroom − stop/target: hindsight vs mechanical rule)",
            f"  stop/target − hold:  ${c.stop_target_vs_time_stop:>12,.0f}   "
            f"(does -35/+60 rule help or hurt vs naive hold?)",
        ]
        return lines


def _find_oracle_bar(day_log: DayLog) -> Optional[BarRecord]:
    for bar in day_log.bars:
        if bar.labels.opportunity_oracle_entry:
            return bar
    return None


def _selection_for_direction(
    bar: BarRecord, direction: str
) -> Optional["SelectedContract"]:  # type: ignore[name-defined]
    for sel in bar.selections:
        if sel.direction == direction:
            return sel
    return None


def build_report(logs: Iterable[DayLog]) -> AttributionReport:
    report = AttributionReport()
    c = report.counts

    for day_log in logs:
        if not day_log.bars:
            continue
        c.n_sessions += 1
        oracle_bar = _find_oracle_bar(day_log)
        if oracle_bar is None:
            continue
        c.n_sessions_with_oracle += 1

        oracle_direction = oracle_bar.labels.opportunity_oracle_direction
        oracle_realized = oracle_bar.labels.opportunity_oracle_realized_pnl or 0.0
        c.oracle_total_pnl += oracle_realized

        session_row: dict = {
            "day": day_log.day,
            "oracle_bar": oracle_bar.bar_index,
            "oracle_direction": oracle_direction,
            "oracle_pnl": oracle_realized,
            "outcome": None,
        }

        # Classify: what did the policy do on the oracle bar?
        if not oracle_bar.surface.any_contract_passes:
            c.guardrail_suppression_count += 1
            session_row["outcome"] = "guardrail_suppression"
            report.per_session.append(session_row)
            continue

        teacher_triggered_bars = [t for t in oracle_bar.teachers if t.triggered]
        if not teacher_triggered_bars:
            c.abstention_count += 1
            session_row["outcome"] = "abstention"
            report.per_session.append(session_row)
            continue

        # Teacher triggered; did any teacher trigger match the oracle direction?
        triggered_directions = {
            "call" if t.action == "buy_call" else "put"
            for t in teacher_triggered_bars
        }
        if oracle_direction not in triggered_directions:
            c.side_error_count += 1
            session_row["outcome"] = "side_error"
            report.per_session.append(session_row)
            continue

        # Right bar + right direction. Find the selection in that direction.
        sel = _selection_for_direction(oracle_bar, oracle_direction)
        if sel is None:
            # Teacher triggered correct direction but no selection passed guardrails.
            c.guardrail_suppression_count += 1
            session_row["outcome"] = "guardrail_suppression_post_trigger"
            report.per_session.append(session_row)
            continue

        c.entered_right_count += 1
        key = f"{sel.teacher_name}_{sel.direction}"
        eh = oracle_bar.labels.exit_headroom_by_selection.get(key) or {}
        exit_headroom_best = eh.get("best_exit_pnl")
        time_stop_pnl = eh.get("time_stop_pnl")
        stop_target_pnl = eh.get("stop_target_pnl")

        if exit_headroom_best is not None:
            c.exit_headroom_total_pnl += exit_headroom_best
            selection_gap = oracle_realized - exit_headroom_best
            c.selection_gap_total += selection_gap
            session_row["selection_gap"] = selection_gap
            if sel.strike != oracle_bar.labels.opportunity_oracle_strike:
                c.entered_same_bar_wrong_contract_count += 1
        if time_stop_pnl is not None:
            c.time_stop_total_pnl += time_stop_pnl
            if exit_headroom_best is not None:
                c.exit_gap_vs_time_stop += (exit_headroom_best - time_stop_pnl)
                session_row["exit_gap_vs_time_stop"] = exit_headroom_best - time_stop_pnl
        if stop_target_pnl is not None:
            c.stop_target_total_pnl += stop_target_pnl
            if exit_headroom_best is not None:
                c.exit_gap_vs_stop_target += (exit_headroom_best - stop_target_pnl)
            if time_stop_pnl is not None:
                c.stop_target_vs_time_stop += (stop_target_pnl - time_stop_pnl)
        session_row["outcome"] = "entered_right"
        report.per_session.append(session_row)

    return report


def format_report(report: AttributionReport) -> str:
    return "\n".join(report.summary_lines())
