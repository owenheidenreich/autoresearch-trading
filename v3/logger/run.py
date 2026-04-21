"""Top-level Stage 1 runner.

Orchestrates the logger: for each requested day, iterates over the union of
teacher windows, builds a `BarRecord` per bar, and returns a `DayLog`. No
I/O-side effects yet — this module returns in-memory objects so the caller
(a test, a notebook, or a future `write_day_log` function) decides what to
persist.
"""
from __future__ import annotations

from typing import Sequence

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import (
    V2Dataset,
    bar_context_from_dataset,
    chain_from_sidecar,
    extra_context_from_dataset,
    has_chain_snapshot,
    load_day_sidecar,
)
from v3.logger.builder import build_bar_record
from v3.logger.schema import DayLog
from v3.teachers.base import Teacher


def _eligible_window(teachers: Sequence[Teacher]) -> tuple[int, int]:
    """Compute the union of teacher windows, in minutes since 09:30 ET."""
    if not teachers:
        return 0, 0
    start = min(t.entry_window_start_min for t in teachers)
    end = max(t.entry_window_end_min for t in teachers)
    return start, end


def run_day(
    dataset: V2Dataset,
    day: str,
    teachers: Sequence[Teacher],
    cfg: GuardrailConfig,
    equity: float,
) -> DayLog:
    """Produce a `DayLog` for one session.

    Loops over every bar in the union-of-teacher-windows, not only bars where
    some teacher triggers — this is required for `NO_TRADE` to stay a first-
    class learnable action per the plan. Zero-contract bars are still logged.
    """
    log = DayLog(day=day)
    sidecar = load_day_sidecar(dataset, day)
    if sidecar is None:
        return log

    day_start_abs, day_end_abs = dataset.day_bar_range(day)
    window_start, window_end = _eligible_window(teachers)

    for abs_idx in range(day_start_abs, day_end_abs):
        local_bar = abs_idx - day_start_abs
        minute = int(dataset.bar_of_day[abs_idx])
        if minute < window_start or minute > window_end:
            continue
        # Eligibility rule (plan): valid context AND a chain snapshot must
        # exist for the bar. "Chain snapshot exists" = sidecar has any rows
        # at this bar, regardless of contract validity. Skipping chain-less
        # bars is what keeps the feasibility diagnostics measuring rail-
        # tightness, not v2 pipeline coverage gaps.
        if not has_chain_snapshot(sidecar, local_bar):
            continue
        bar_ctx = bar_context_from_dataset(dataset, abs_idx)
        extra = extra_context_from_dataset(dataset, abs_idx)
        chain = chain_from_sidecar(sidecar, local_bar)
        ts_ms = 0  # timestamp_ms unused in v1; keep placeholder until parquet writer needs it
        record = build_bar_record(
            day=day,
            bar_index=minute,
            timestamp_ms=ts_ms,
            bar_ctx=bar_ctx,
            chain=chain,
            teachers=teachers,
            equity=equity,
            cfg=cfg,
            vix=extra["vix"],
            atm_iv=extra["atm_iv"],
            iv_percentile=extra["iv_percentile"],
        )
        log.bars.append(record)
    return log


def run_days(
    dataset: V2Dataset,
    days: Sequence[str],
    teachers: Sequence[Teacher],
    cfg: GuardrailConfig,
    equity: float,
) -> list[DayLog]:
    return [run_day(dataset, d, teachers, cfg, equity) for d in days]


def iter_days(
    dataset: V2Dataset,
    days: Sequence[str],
    teachers: Sequence[Teacher],
    cfg: GuardrailConfig,
    equity: float,
):
    """Generator version — yields one `DayLog` at a time so the caller can
    consume and discard. Use for full-dataset runs where holding every bar's
    contract list in memory would be ~10 GB.
    """
    for day in days:
        yield run_day(dataset, day, teachers, cfg, equity)
