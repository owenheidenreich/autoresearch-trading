"""Four-diagnostic feasibility reporter.

Computes the SPX/$25k feasibility metrics defined in the plan:

1. `contract_surface_coverage_rate`
   numer = eligible bars with ≥1 guardrail-passing contract
   denom = all eligible bars

2. `premium_cap_bind_severity` (replaces the old `premium_cap_bind_rate`)
   For each signal-positive bar, compute cap_blocked_fraction =
       n_blocked_solely_by_premium_cap / max(n_contracts_valid, 1).
   Report the mean and p95 across signal-positive bars.
   The old "share of bars with ≥1 cap-blocked contract" was always ~1.0
   because any real SPX chain has deep-ITM contracts above any reasonable
   cap. The severity metric measures how much of the chain the cap is
   actually eating, which varies meaningfully with market conditions.

3. `guardrail_suppression_rate`
   numer = signal-positive bars with zero passing contracts
   denom = signal-positive bars

4. `low_delta_forcing_rate`
   numer = entered trades where chosen abs_delta ≤ q25 passing AND a
           higher-abs_delta contract existed and was blocked solely by cap
   denom = entered trades

"Signal-positive" is currently `any teacher triggered`. When the opportunity
oracle lands, it will expand to `any teacher triggered OR opportunity oracle
positive at this bar` — the signature here already anticipates that by
reading `BarRecord.labels.opportunity_oracle_entry` if present.

Stratification:
- full sample
- per teacher (only bars where a named teacher triggered)
- per session block (15-minute buckets within the eligible window)

All alarm thresholds come straight from the plan. The reporter returns raw
numerators/denominators alongside the rates so borderline sliced results
(e.g. a block with 3 denominator bars) can be filtered or flagged at
analysis time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from v3.logger.schema import BarRecord, DayLog


# Alarm thresholds. `cap_bind_severity` threshold recalibrated after the
# metric switched from presence (always ~1.0) to severity (mean fraction of
# chain blocked on signal-bars). 40% fraction-blocked is the alarm.
COVERAGE_ALARM_BELOW = 0.60
CAP_BIND_SEVERITY_ALARM_ABOVE = 0.40
SUPPRESSION_ALARM_ABOVE = 0.20
LOW_DELTA_FORCING_ALARM_ABOVE = 0.40


@dataclass
class Counts:
    """Raw counts behind every diagnostic. Rates are numer/denom."""

    eligible_bars: int = 0
    bars_with_coverage: int = 0
    signal_positive_bars: int = 0
    signal_and_suppressed: int = 0
    entered_trades: int = 0
    low_delta_forced_entries: int = 0

    # Severity: sum of (n_solely_cap_blocked / max(n_valid, 1)) across
    # signal-positive bars. Mean = sum / signal_positive_bars.
    cap_blocked_fraction_sum: float = 0.0
    # Per-bar cap_blocked_fraction samples for percentile aggregation.
    cap_blocked_fraction_samples: list[float] = field(default_factory=list)

    def coverage_rate(self) -> Optional[float]:
        return _safe_div(self.bars_with_coverage, self.eligible_bars)

    def cap_bind_severity_mean(self) -> Optional[float]:
        if self.signal_positive_bars == 0:
            return None
        return self.cap_blocked_fraction_sum / self.signal_positive_bars

    def cap_bind_severity_p95(self) -> Optional[float]:
        if not self.cap_blocked_fraction_samples:
            return None
        import numpy as np
        return float(np.percentile(self.cap_blocked_fraction_samples, 95))

    def suppression_rate(self) -> Optional[float]:
        return _safe_div(self.signal_and_suppressed, self.signal_positive_bars)

    def low_delta_forcing_rate(self) -> Optional[float]:
        return _safe_div(self.low_delta_forced_entries, self.entered_trades)


@dataclass
class StratumReport:
    """One stratum's counts and derived rates."""

    name: str
    counts: Counts = field(default_factory=Counts)

    def verdict(self) -> str:
        """Textual interpretation using the plan's thresholds."""
        flags = []
        c = self.counts
        cov = c.coverage_rate()
        cap_mean = c.cap_bind_severity_mean()
        sup = c.suppression_rate()
        ldf = c.low_delta_forcing_rate()
        if cov is not None and cov < COVERAGE_ALARM_BELOW:
            flags.append(f"coverage<{COVERAGE_ALARM_BELOW}")
        if cap_mean is not None and cap_mean > CAP_BIND_SEVERITY_ALARM_ABOVE:
            flags.append(f"cap_bind_severity>{CAP_BIND_SEVERITY_ALARM_ABOVE}")
        if sup is not None and sup > SUPPRESSION_ALARM_ABOVE:
            flags.append(f"suppression>{SUPPRESSION_ALARM_ABOVE}")
        if ldf is not None and ldf > LOW_DELTA_FORCING_ALARM_ABOVE:
            flags.append(f"low_delta_forcing>{LOW_DELTA_FORCING_ALARM_ABOVE}")
        if not flags:
            return "within envelope"
        if len(flags) >= 2:
            return f"OVERCONSTRAINED ({', '.join(flags)})"
        return f"borderline ({flags[0]})"


@dataclass
class FeasibilityReport:
    """Full four-diagnostic report, sliced across strata."""

    strata: dict[str, StratumReport] = field(default_factory=dict)

    def get(self, name: str) -> StratumReport:
        if name not in self.strata:
            self.strata[name] = StratumReport(name=name)
        return self.strata[name]


def _safe_div(n: int, d: int) -> Optional[float]:
    return (n / d) if d > 0 else None


def _bar_is_signal_positive(bar: BarRecord) -> bool:
    """Signal OR opportunity-oracle-positive. Oracle path kicks in when
    labels are populated; for now, we fall through to teacher-trigger only.
    """
    if bar.labels.opportunity_oracle_entry:
        return True
    return any(t.triggered for t in bar.teachers)


def _bar_forces_low_delta(bar: BarRecord) -> bool:
    """One entered trade counts as low-delta-forced when BOTH:
    - the chosen contract's abs_delta is at or below the 25th percentile of
      the passing-delta distribution on this bar;
    - the surface had at least one higher-abs_delta contract blocked solely
      by the premium cap.

    If the bar has multiple selections (multiple teachers triggered in the
    same direction-direction pair), all selections must satisfy the rule for
    the bar to count — but we count PER-SELECTION, since each corresponds to
    a distinct entered trade in the Stage 1 contract.
    """
    raise RuntimeError("use _count_low_delta_forced_selections")


def _count_low_delta_forced_selections(bar: BarRecord) -> tuple[int, int]:
    """Return (entered_trades, low_delta_forced_entries) for one bar."""
    if not bar.selections:
        return 0, 0
    s = bar.surface
    q25 = s.passing_abs_delta_q25
    max_cap_blocked = s.max_abs_delta_blocked_solely_by_cap
    if q25 < 0 or max_cap_blocked < 0:
        # Degenerate surface — no passing contracts, or no cap-blocked contracts.
        # Selections shouldn't exist in that case, but be defensive.
        return len(bar.selections), 0
    forced = 0
    for sel in bar.selections:
        if sel.abs_delta <= q25 and max_cap_blocked > sel.abs_delta:
            forced += 1
    return len(bar.selections), forced


def _session_block_name(bar_index: int, block_size: int = 30) -> str:
    """15-min wide buckets indexed by their floor, e.g. minute 47 -> 'block_30_60'.
    Block width is 30 min by default (two 15-min buckets joined) to keep
    sample-size-per-block reasonable; override via `block_size` if needed.
    """
    lo = (bar_index // block_size) * block_size
    hi = lo + block_size
    return f"block_{lo:03d}_{hi:03d}"


def _update_stratum(stratum: StratumReport, bar: BarRecord) -> None:
    c = stratum.counts
    c.eligible_bars += 1
    if bar.surface.any_contract_passes:
        c.bars_with_coverage += 1
    if _bar_is_signal_positive(bar):
        c.signal_positive_bars += 1
        denom = max(bar.surface.n_contracts_valid, 1)
        fraction_blocked = bar.surface.n_blocked_solely_by_premium_cap / denom
        c.cap_blocked_fraction_sum += fraction_blocked
        c.cap_blocked_fraction_samples.append(fraction_blocked)
        if not bar.surface.any_contract_passes:
            c.signal_and_suppressed += 1
    entered, forced = _count_low_delta_forced_selections(bar)
    c.entered_trades += entered
    c.low_delta_forced_entries += forced


def build_report(
    logs: Iterable[DayLog],
    session_block_size: int = 30,
) -> FeasibilityReport:
    report = FeasibilityReport()
    all_stratum = report.get("all")
    for log in logs:
        for bar in log.bars:
            _update_stratum(all_stratum, bar)
            # Per session block
            _update_stratum(
                report.get(_session_block_name(bar.bar_index, session_block_size)),
                bar,
            )
            # Per teacher (only the teachers that triggered on this bar)
            for t in bar.teachers:
                if t.triggered:
                    _update_stratum(report.get(f"teacher_{t.teacher_name}"), bar)
    return report


def format_report(report: FeasibilityReport) -> str:
    """Pretty-print all strata sorted so 'all' appears first."""
    lines = []
    header = (
        f"{'stratum':<28}{'n_elig':>8}{'n_sig':>8}"
        f"{'cov':>8}{'cap_mean':>10}{'cap_p95':>10}{'supp':>8}{'low_d_f':>10}  verdict"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def _sort_key(name: str) -> tuple[int, str]:
        if name == "all":
            return (0, "")
        if name.startswith("teacher_"):
            return (1, name)
        return (2, name)

    for name in sorted(report.strata.keys(), key=_sort_key):
        s = report.strata[name]
        c = s.counts
        cov = c.coverage_rate()
        cap_mean = c.cap_bind_severity_mean()
        cap_p95 = c.cap_bind_severity_p95()
        sup = c.suppression_rate()
        ldf = c.low_delta_forcing_rate()

        def _fmt(x: Optional[float]) -> str:
            return f"{x:.3f}" if x is not None else "   ---"

        lines.append(
            f"{name:<28}{c.eligible_bars:>8}{c.signal_positive_bars:>8}"
            f"{_fmt(cov):>8}{_fmt(cap_mean):>10}{_fmt(cap_p95):>10}"
            f"{_fmt(sup):>8}{_fmt(ldf):>10}  {s.verdict()}"
        )
    return "\n".join(lines)
