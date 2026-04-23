"""Opportunity-surface record types.

The four layers of per-bar truth the plan requires the logger to separate:

1. bar-level context (underlying state, session structure, regime)
2. per-teacher direction outputs (action + in-window + triggered)
3. per-contract guardrail outcomes (per-gate pass/fail booleans, raw values)
4. oracle labels (opportunity-oracle positive flag, forward-PnL surface)

A precomputed `SurfaceSummary` is stored alongside the raw contract list so
the SPX/$25k feasibility diagnostics can be computed cheaply without
re-scanning every contract. The raw `contracts` list is still canonical and
is the ground truth anything downstream must agree with — `SurfaceSummary`
is strictly derivable from it.

Design invariants enforced by the schema shape:

- NO_SIGNAL is intentionally overloaded on the teacher action axis, so the
  teacher record carries `in_window` and `triggered` explicitly. Anything
  finer-grained (contested-chop, no-setup-in-window, etc.) is derivable from
  the bar-level state fields.
- Teacher output is logged per bar without deduplication. ORC-style state
  signals that re-assert across consecutive bars are preserved as-is;
  "first-trigger" vs "all-signaled-bars" is a downstream query, not a
  logger decision.
- Oracle labels are optional at logging time and filled in by a later pass.
  This keeps the logger itself a single forward sweep over the dataset.

Serialization format is deferred (pickle for v1 parity with v2 sidecars;
parquet planned once the schema stabilizes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ContractRecord:
    """One contract as observed at one bar.

    The five `*_ok` fields mirror `ContractCheck` in `v3.guardrails` so
    every feasibility counterfactual is answerable from this record alone.
    Raw `delta` is signed; `abs_delta` is pre-computed for query efficiency.
    """

    strike: float
    right: str  # "C" or "P"
    mid: float
    delta: float
    abs_delta: float
    spread_fraction: float
    premium: float  # mid * 100, $ per contract

    # Per-gate flags (mirrors ContractCheck)
    contract_valid: bool
    premium_cap_ok: bool
    premium_floor_ok: bool
    delta_floor_ok: bool
    spread_cap_ok: bool
    passed: bool

    # Optional contract-quality fields for scoring/tiebreak later
    gamma_dollar: Optional[float] = None
    theta_to_premium: Optional[float] = None


@dataclass(frozen=True)
class TeacherOutcome:
    """One teacher's output at one bar.

    `action` carries the raw TeacherAction value. `in_window` and `triggered`
    are explicit booleans so NO_SIGNAL doesn't have to do double duty.
    Teacher-specific state (e.g. failed-break "contested") is derivable from
    BarRecord.bars_since_break_above / below.
    """

    teacher_name: str
    action: str  # TeacherAction.value
    in_window: bool
    triggered: bool


@dataclass(frozen=True)
class SelectedContract:
    """Contract the harness would pick under the guardrails for a given
    (teacher, direction) combination at this bar. Populated only when the
    teacher triggered and at least one contract of that right passed the
    guardrails. Tiebreak: max(gamma_dollar / theta_to_premium).
    """

    teacher_name: str
    direction: str  # "call" or "put"
    strike: float
    mid: float
    delta: float
    abs_delta: float
    premium: float
    spread_fraction: float
    score: float  # gamma_dollar / theta_to_premium


@dataclass(frozen=True)
class SurfaceSummary:
    """Per-bar derived summary of the chain surface.

    Strictly redundant with the `contracts` list — every field here is
    derivable — but precomputed for cheap aggregation at diagnostic time.
    Downstream diagnostics MUST be computable from this struct alone;
    if they can't, add the necessary field here rather than re-scanning
    the full chain.
    """

    n_contracts_total: int
    n_contracts_valid: int         # contract_valid == True
    n_contracts_passing: int        # all gates == True
    n_blocked_solely_by_premium_cap: int

    # Coverage: true iff at least one contract passes all gates.
    any_contract_passes: bool
    # Counterfactual: true iff at least one contract would pass if cap were removed.
    any_passes_without_premium_cap: bool

    # Delta distribution over PASSING contracts (call + put unified, on abs_delta).
    # -1.0 sentinel when no contract passes.
    passing_abs_delta_q25: float
    passing_abs_delta_q50: float
    passing_abs_delta_q75: float

    # Among contracts blocked SOLELY by premium cap: the highest abs_delta
    # seen. Supports `low_delta_forcing_rate`: if the chosen contract's delta
    # is below q25 of passing AND this value is higher than the chosen
    # delta, the cap is forcing us into the low-delta tail.
    # -1.0 sentinel when no contract is solely-cap-blocked.
    max_abs_delta_blocked_solely_by_cap: float


@dataclass
class OracleLabels:
    """Forward labels produced by a post-log oracle pass.

    Filled in after the raw bar log exists. All fields are Optional so the
    logger can emit a bar record without waiting for the oracle pass. The
    oracle reader sets them in-place once labels exist.
    """

    # Opportunity oracle: is this bar the entry on the hindsight-optimal
    # (bar, direction, contract, exit) tuple for this session?
    opportunity_oracle_entry: Optional[bool] = None
    # If the opportunity oracle picks this bar, which direction and contract?
    opportunity_oracle_direction: Optional[str] = None
    opportunity_oracle_strike: Optional[float] = None
    opportunity_oracle_realized_pnl: Optional[float] = None

    # Bar-local forward-PnL surface at this entry, best exit within the
    # teacher's session constraints. Separate for each direction so the
    # "side error gap" diagnostic can be computed.
    best_forward_pnl_call: Optional[float] = None
    best_forward_pnl_put: Optional[float] = None
    worst_forward_pnl_call: Optional[float] = None
    worst_forward_pnl_put: Optional[float] = None

    # Max favorable / adverse excursion over fixed horizons. Per direction.
    mfe_5min_call: Optional[float] = None
    mae_5min_call: Optional[float] = None
    mfe_10min_call: Optional[float] = None
    mae_10min_call: Optional[float] = None
    mfe_20min_call: Optional[float] = None
    mae_20min_call: Optional[float] = None
    mfe_5min_put: Optional[float] = None
    mae_5min_put: Optional[float] = None
    mfe_10min_put: Optional[float] = None
    mae_10min_put: Optional[float] = None
    mfe_20min_put: Optional[float] = None
    mae_20min_put: Optional[float] = None

    # Exit-headroom oracle: conditional on the actual selected contract at
    # this bar, what's the best forward exit PnL the policy could have
    # achieved? Keyed by f"{teacher_name}_{direction}" so we can match
    # against `BarRecord.selections` at attribution time.
    exit_headroom_by_selection: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BarRecord:
    """One eligible bar's complete record.

    Identity is (day, bar_index). `bar_index` is minutes since 09:30 ET
    (0..389). `timestamp_ms` is the exchange-provided bar timestamp in ms
    epoch — kept alongside bar_index so the record is self-describing.

    The `eligible_reason` field is populated even when no teacher triggered
    and zero contracts pass the guardrails, because the plan explicitly
    wants those bars logged as NO_TRADE.
    """

    day: str                  # "YYYY-MM-DD"
    bar_index: int            # 0..389 minutes since 09:30 ET
    timestamp_ms: int

    # --- Bar-level context (fields beyond BarContext are surface/regime) ---
    underlying_close: float
    vwap: float
    vwap_slope: float
    volume_ratio: float
    first15_high: float
    first15_low: float
    first15_range_pct: float
    bars_since_break_above_first15: int
    bars_since_break_below_first15: int
    vix: float
    atm_iv: float
    iv_percentile: float

    # --- Eligibility reason ---
    # True if this bar falls in the union of teacher windows AND has valid
    # context + chain snapshot. False means the bar is NOT logged; this
    # field is always True in stored rows and documents what "eligible" meant.
    eligible: bool

    # --- Teacher outputs ---
    teachers: tuple[TeacherOutcome, ...]

    # --- Full chain (ground truth; SurfaceSummary is derived from this) ---
    contracts: tuple[ContractRecord, ...]

    # --- Precomputed surface summary ---
    surface: SurfaceSummary

    # --- Selected contracts under guardrails (per teacher that triggered) ---
    selections: tuple[SelectedContract, ...]

    # --- Forward labels (filled by oracle pass) ---
    labels: OracleLabels = field(default_factory=OracleLabels)


@dataclass
class DayLog:
    """All eligible bars for one session."""

    day: str
    bars: list[BarRecord] = field(default_factory=list)

    def eligible_count(self) -> int:
        return sum(1 for b in self.bars if b.eligible)
