"""Builder: composes schema records from per-bar inputs.

This module is pure composition logic. It takes a bar's worth of input
(market context, full option chain, registered teachers, current equity) and
produces a `BarRecord` matching the schema. No I/O, no data-source coupling
— the harness is responsible for translating whatever upstream format
(v2 data.pt, a future v3 dataset, or synthetic test input) into the
`ChainRowInput` dataclass before calling `build_bar_record`.

The non-obvious correctness constraints enforced here:

1. `build_contract_records` runs `filter_contract` on EVERY row in the
   chain, never short-circuiting. This is required for `premium_cap_bind_rate`
   and `low_delta_forcing_rate` to be answerable from the log alone.

2. `summarize_surface` is strictly a derivation from the contract list; any
   future diagnostic that can't be computed from `SurfaceSummary` alone
   must land here as a new field, not be computed by re-scanning contracts
   at analysis time.

3. `select_contract` is a SEPARATE step from teacher evaluation. The teacher
   outputs a direction; the harness (here) picks the contract. This keeps
   "teacher signaled" and "harness had a contract" independently queryable
   in the log.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from v3.config import GuardrailConfig
from v3.guardrails import filter_contract
from v3.logger.schema import (
    BarRecord,
    ContractRecord,
    SelectedContract,
    SurfaceSummary,
    TeacherOutcome,
)
from v3.teachers.base import BarContext, Teacher, TeacherAction


@dataclass(frozen=True)
class ChainRowInput:
    """Canonical per-contract input to the builder.

    The harness translates upstream data (v2 chain_data fields, data.pt
    arrays, etc.) into this dataclass before calling `build_bar_record`.
    """

    strike: float
    right: str  # "C" or "P"
    mid: float
    delta: float  # signed; put deltas stay negative
    spread_fraction: float
    contract_valid: bool
    gamma_dollar: Optional[float] = None
    theta_to_premium: Optional[float] = None


def build_contract_records(
    chain: Sequence[ChainRowInput],
    equity: float,
    cfg: GuardrailConfig,
) -> tuple[ContractRecord, ...]:
    """Evaluate every contract against the guardrails. Never short-circuits."""
    out: list[ContractRecord] = []
    for row in chain:
        check = filter_contract(
            mid=row.mid,
            delta=row.delta,
            spread_fraction=row.spread_fraction,
            contract_valid=row.contract_valid,
            equity=equity,
            cfg=cfg,
        )
        out.append(
            ContractRecord(
                strike=row.strike,
                right=row.right,
                mid=row.mid,
                delta=row.delta,
                abs_delta=check.abs_delta,
                spread_fraction=check.spread_fraction,
                premium=check.premium,
                contract_valid=check.contract_valid,
                premium_cap_ok=check.premium_cap_ok,
                premium_floor_ok=check.premium_floor_ok,
                delta_floor_ok=check.delta_floor_ok,
                spread_cap_ok=check.spread_cap_ok,
                passed=check.passed,
                gamma_dollar=row.gamma_dollar,
                theta_to_premium=row.theta_to_premium,
            )
        )
    return tuple(out)


def _blocked_solely_by_premium_cap(c: ContractRecord) -> bool:
    return (
        not c.premium_cap_ok
        and c.contract_valid
        and c.premium_floor_ok
        and c.delta_floor_ok
        and c.spread_cap_ok
    )


def summarize_surface(contracts: Sequence[ContractRecord]) -> SurfaceSummary:
    """Derive the per-bar surface summary.

    Must be the ONLY summarizing path in the codebase: downstream diagnostic
    code reads from `SurfaceSummary`, not from `contracts`. If a new
    diagnostic needs more info, extend this function AND the schema, do not
    re-scan contracts elsewhere.
    """
    n_total = len(contracts)
    n_valid = sum(1 for c in contracts if c.contract_valid)
    n_passing = sum(1 for c in contracts if c.passed)
    n_solely_cap = sum(1 for c in contracts if _blocked_solely_by_premium_cap(c))

    any_passes = n_passing > 0
    any_would_pass_without_cap = any(
        c.passed or _blocked_solely_by_premium_cap(c) for c in contracts
    )

    passing_deltas = [c.abs_delta for c in contracts if c.passed]
    if passing_deltas:
        arr = np.asarray(passing_deltas, dtype=float)
        q25 = float(np.percentile(arr, 25))
        q50 = float(np.percentile(arr, 50))
        q75 = float(np.percentile(arr, 75))
    else:
        q25 = q50 = q75 = -1.0

    cap_blocked_deltas = [
        c.abs_delta for c in contracts if _blocked_solely_by_premium_cap(c)
    ]
    max_cap_blocked = max(cap_blocked_deltas) if cap_blocked_deltas else -1.0

    return SurfaceSummary(
        n_contracts_total=n_total,
        n_contracts_valid=n_valid,
        n_contracts_passing=n_passing,
        n_blocked_solely_by_premium_cap=n_solely_cap,
        any_contract_passes=any_passes,
        any_passes_without_premium_cap=any_would_pass_without_cap,
        passing_abs_delta_q25=q25,
        passing_abs_delta_q50=q50,
        passing_abs_delta_q75=q75,
        max_abs_delta_blocked_solely_by_cap=max_cap_blocked,
    )


def _selection_score(c: ContractRecord) -> float:
    """gamma_dollar / theta_to_premium — higher is better.

    Missing or zero theta_to_premium falls back to 0.0 rather than NaN so
    the selection is deterministic even when chain quality varies. Negative
    theta_to_premium (which should not occur for a long option) also maps
    to 0.0 — callers should investigate if they see it in logs.
    """
    if c.gamma_dollar is None or c.theta_to_premium is None:
        return 0.0
    if c.theta_to_premium <= 0.0:
        return 0.0
    return c.gamma_dollar / c.theta_to_premium


def select_contract(
    contracts: Sequence[ContractRecord],
    direction: str,
    teacher_name: str,
) -> Optional[SelectedContract]:
    """Pick the best guardrail-passing contract for the given direction."""
    right = "C" if direction == "call" else "P"
    candidates = [c for c in contracts if c.passed and c.right == right]
    if not candidates:
        return None
    best = max(candidates, key=_selection_score)
    return SelectedContract(
        teacher_name=teacher_name,
        direction=direction,
        strike=best.strike,
        mid=best.mid,
        delta=best.delta,
        abs_delta=best.abs_delta,
        premium=best.premium,
        spread_fraction=best.spread_fraction,
        score=_selection_score(best),
    )


def evaluate_teachers(
    bar: BarContext,
    teachers: Sequence[Teacher],
) -> tuple[TeacherOutcome, ...]:
    out: list[TeacherOutcome] = []
    for t in teachers:
        action = t.evaluate(bar)
        out.append(
            TeacherOutcome(
                teacher_name=t.name,
                action=action.value,
                in_window=t.in_window(bar.minute_of_session),
                triggered=action != TeacherAction.NO_SIGNAL,
            )
        )
    return tuple(out)


def build_bar_record(
    day: str,
    bar_index: int,
    timestamp_ms: int,
    bar_ctx: BarContext,
    chain: Sequence[ChainRowInput],
    teachers: Sequence[Teacher],
    equity: float,
    cfg: GuardrailConfig,
    vix: float,
    atm_iv: float,
    iv_percentile: float,
) -> BarRecord:
    """Assemble one BarRecord from the four input streams."""
    contracts = build_contract_records(chain, equity, cfg)
    surface = summarize_surface(contracts)
    teacher_outcomes = evaluate_teachers(bar_ctx, teachers)

    selections: list[SelectedContract] = []
    for outcome in teacher_outcomes:
        if not outcome.triggered:
            continue
        direction = "call" if outcome.action == "buy_call" else "put"
        sel = select_contract(contracts, direction, outcome.teacher_name)
        if sel is not None:
            selections.append(sel)

    return BarRecord(
        day=day,
        bar_index=bar_index,
        timestamp_ms=timestamp_ms,
        underlying_close=bar_ctx.close,
        vwap=bar_ctx.vwap,
        vwap_slope=bar_ctx.vwap_slope,
        volume_ratio=bar_ctx.volume_ratio,
        first15_high=bar_ctx.first15_high,
        first15_low=bar_ctx.first15_low,
        first15_range_pct=bar_ctx.first15_range_pct,
        bars_since_break_above_first15=bar_ctx.bars_since_break_above_first15,
        bars_since_break_below_first15=bar_ctx.bars_since_break_below_first15,
        vix=vix,
        atm_iv=atm_iv,
        iv_percentile=iv_percentile,
        eligible=True,
        teachers=teacher_outcomes,
        contracts=contracts,
        surface=surface,
        selections=tuple(selections),
    )
