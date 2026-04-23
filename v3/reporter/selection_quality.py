"""Selection-quality reporter.

Unlike `attribution.py` which only measures selection_gap on oracle bars
(117 of 986 sessions), this reporter walks EVERY teacher-entered bar in
every session and asks: "on this specific entry, did the teacher's
gamma/theta tiebreak pick the forward-PnL-maximizing contract among all
passing contracts of the chosen direction, or did it leave money on the
table?"

Why this matters: attribution's selection_gap was structurally pinned near
zero because oracle bars are clean-directional days by definition — and on
clean directional days, highest-delta-under-cap (≈ gamma/theta winner) is
also the dollar-maximizing contract. We need the selection gap measured on
ALL entries to know if the heuristic is robust outside that biased sample.

Metrics (per stratum: overall, per teacher, per direction, per session block):

- `n_entries`: teacher-triggered selections in this stratum
- `mean_gap`: mean (best_possible − actual_selection) in dollars
- `median_gap`: median gap in dollars
- `p95_gap`: 95th percentile gap
- `pct_suboptimal`: share of entries where gap > $1
- `pct_significantly_suboptimal`: share of entries where gap > $50

Positive gap ⇒ the teacher's tiebreak left money on the table. Large mean gap
with high `pct_suboptimal` ⇒ the selection heuristic is NOT robust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from v3.logger.schema import BarRecord, DayLog


@dataclass
class SelectionSample:
    """Raw per-entry record used to compute aggregates."""

    day: str
    bar_index: int
    teacher_name: str
    direction: str
    gap: float           # best_possible − actual_selection (dollars)
    actual: float        # exit_headroom best on actual selection
    best_possible: float  # opportunity oracle best across all passing contracts of this direction


@dataclass
class StratumStats:
    name: str
    samples: list[SelectionSample] = field(default_factory=list)

    def n(self) -> int:
        return len(self.samples)

    def _gaps(self) -> np.ndarray:
        return np.asarray([s.gap for s in self.samples], dtype=float)

    def mean_gap(self) -> Optional[float]:
        return float(self._gaps().mean()) if self.samples else None

    def median_gap(self) -> Optional[float]:
        return float(np.median(self._gaps())) if self.samples else None

    def p95_gap(self) -> Optional[float]:
        return float(np.percentile(self._gaps(), 95)) if self.samples else None

    def max_gap(self) -> Optional[float]:
        return float(self._gaps().max()) if self.samples else None

    def pct_suboptimal(self, threshold: float = 1.0) -> Optional[float]:
        if not self.samples:
            return None
        return float(np.mean(self._gaps() > threshold))


@dataclass
class SelectionQualityReport:
    strata: dict[str, StratumStats] = field(default_factory=dict)

    def get(self, name: str) -> StratumStats:
        if name not in self.strata:
            self.strata[name] = StratumStats(name=name)
        return self.strata[name]


def _session_block(bar_index: int, width: int = 30) -> str:
    lo = (bar_index // width) * width
    return f"block_{lo:03d}_{lo + width:03d}"


def _process_bar(bar: BarRecord, report: SelectionQualityReport) -> None:
    for sel in bar.selections:
        key = f"{sel.teacher_name}_{sel.direction}"
        eh = bar.labels.exit_headroom_by_selection.get(key) or {}
        actual = eh.get("best_exit_pnl")
        if actual is None:
            continue
        if sel.direction == "call":
            best_possible = bar.labels.best_forward_pnl_call
        else:
            best_possible = bar.labels.best_forward_pnl_put
        if best_possible is None:
            continue
        sample = SelectionSample(
            day=bar.day,
            bar_index=bar.bar_index,
            teacher_name=sel.teacher_name,
            direction=sel.direction,
            gap=best_possible - actual,
            actual=actual,
            best_possible=best_possible,
        )
        report.get("all").samples.append(sample)
        report.get(f"teacher_{sel.teacher_name}").samples.append(sample)
        report.get(f"direction_{sel.direction}").samples.append(sample)
        report.get(_session_block(bar.bar_index)).samples.append(sample)


def build_report(logs: Iterable[DayLog]) -> SelectionQualityReport:
    report = SelectionQualityReport()
    for log in logs:
        for bar in log.bars:
            if not bar.selections:
                continue
            _process_bar(bar, report)
    return report


def format_report(report: SelectionQualityReport) -> str:
    lines = []
    header = (
        f"{'stratum':<28}{'n':>8}{'mean':>10}{'median':>10}{'p95':>10}{'max':>10}"
        f"{'%sub>$1':>10}{'%sub>$50':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def _sort_key(name: str) -> tuple[int, str]:
        if name == "all":
            return (0, "")
        if name.startswith("teacher_"):
            return (1, name)
        if name.startswith("direction_"):
            return (2, name)
        return (3, name)

    for name in sorted(report.strata.keys(), key=_sort_key):
        s = report.strata[name]

        def _fmt(x: Optional[float]) -> str:
            return f"{x:.2f}" if x is not None else "   ---"

        lines.append(
            f"{name:<28}{s.n():>8}"
            f"{_fmt(s.mean_gap()):>10}{_fmt(s.median_gap()):>10}"
            f"{_fmt(s.p95_gap()):>10}{_fmt(s.max_gap()):>10}"
            f"{_fmt(s.pct_suboptimal(1.0)):>10}{_fmt(s.pct_suboptimal(50.0)):>10}"
        )
    return "\n".join(lines)
