"""Exit-headroom oracle: conditional on the actual teacher-selected
contract at a bar, the best hindsight exit PnL achievable on that contract.

This is the SECONDARY oracle. Primary = opportunity oracle (best contract
across the candidate set). The difference between the two is the *selection
gap*: how much the teacher's contract choice left on the table vs the best
possible contract. The difference between exit-headroom and the policy's
*actual* exit is the *exit gap*: how much a better exit model could have
recovered on the already-chosen contract.

Keyed by `f"{teacher_name}_{direction}"` to match `BarRecord.selections`.

Also records `time_stop_pnl` — what the trade realizes if held to
session_end_bar (the simplest "naive policy" baseline). Attribution's
exit_gap = best_exit_pnl − time_stop_pnl.
"""
from __future__ import annotations

import numpy as np

from v3.logger.schema import DayLog
from v3.oracles.opportunity import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
    _best_and_worst_forward_pnl,
    _build_contract_paths,
    _contract_idx_for_record,
)


def _time_stop_pnl(
    mids: np.ndarray,
    entry_bar: int,
    entry_mid: float,
    entry_spread_frac: float,
    session_end_bar: int,
    commission: float,
) -> float | None:
    """PnL if we hold the position from entry_bar until the last observed bar
    in the forward window [entry_bar+1, session_end_bar], treating that last
    observation as the forced time-stop exit. Returns None if no observation.
    """
    end = min(session_end_bar, len(mids) - 1)
    window = mids[entry_bar + 1 : end + 1]
    finite = np.where(np.isfinite(window))[0]
    if finite.size == 0:
        return None
    exit_mid = float(window[finite[-1]])
    entry_ask = entry_mid * (1.0 + entry_spread_frac / 2.0)
    exit_bid = exit_mid * (1.0 - entry_spread_frac / 2.0)
    return 100.0 * (exit_bid - entry_ask) - commission


def _stop_target_pnl(
    mids: np.ndarray,
    entry_bar: int,
    entry_mid: float,
    entry_spread_frac: float,
    session_end_bar: int,
    commission: float,
    stop_pct: float = -0.35,
    target_pct: float = 0.60,
) -> float | None:
    """PnL under a -35%/+60%/time-stop mechanical exit rule.

    Triggers checked on mid_pct_change bar-by-bar. Whichever fires first is
    the exit bar; if nothing fires, exit at the last observation (same as
    time_stop). Used as an attribution baseline — NOT a recommended live
    policy (see decay analysis: a fixed stop cuts ~5-9% of recoverable
    drawdowns).
    """
    end = min(session_end_bar, len(mids) - 1)
    window = mids[entry_bar + 1 : end + 1]
    finite_mask = np.isfinite(window)
    if not finite_mask.any():
        return None

    entry_ask = entry_mid * (1.0 + entry_spread_frac / 2.0)

    # Scan forward for stop or target trigger
    exit_mid = None
    for i, mid in enumerate(window):
        if not np.isfinite(mid):
            continue
        pct_change = (mid - entry_mid) / entry_mid
        if pct_change <= stop_pct or pct_change >= target_pct:
            exit_mid = float(mid)
            break
    if exit_mid is None:
        # Neither stop nor target triggered — fall back to last observation (time stop)
        exit_mid = float(window[np.where(finite_mask)[0][-1]])

    exit_bid = exit_mid * (1.0 - entry_spread_frac / 2.0)
    return 100.0 * (exit_bid - entry_ask) - commission


def apply_exit_headroom_oracle(
    day_log: DayLog,
    sidecar: dict,
    bars_per_day: int,
    session_end_bar: int = DEFAULT_SESSION_END_BAR,
    commission: float = DEFAULT_COMMISSION_PER_CONTRACT,
) -> None:
    """Populate `BarRecord.labels.exit_headroom_by_selection` on each bar.

    For each `SelectedContract` in `bar.selections`, look up the matching
    `ContractRecord` (by strike+right) in `bar.contracts`, find its forward
    path, and compute the hindsight-best forward PnL on that specific
    contract. Stores a dict keyed by `f"{teacher_name}_{direction}"`.
    """
    paths = _build_contract_paths(sidecar, bars_per_day)

    for bar in day_log.bars:
        if not bar.selections:
            continue
        local_bar = bar.bar_index
        for sel in bar.selections:
            match = None
            for c in bar.contracts:
                if c.strike == sel.strike and c.right == (
                    "P" if sel.direction == "put" else "C"
                ):
                    match = c
                    break
            if match is None:
                continue
            cid = _contract_idx_for_record(sidecar, local_bar, match)
            if cid is None or cid not in paths:
                continue
            path = paths[cid]
            best_pnl, worst_pnl = _best_and_worst_forward_pnl(
                path,
                entry_bar=local_bar,
                entry_mid=match.mid,
                entry_spread_frac=match.spread_fraction,
                session_end_bar=session_end_bar,
                commission=commission,
            )
            if best_pnl is None:
                continue
            ts_pnl = _time_stop_pnl(
                path.mids,
                entry_bar=local_bar,
                entry_mid=match.mid,
                entry_spread_frac=match.spread_fraction,
                session_end_bar=session_end_bar,
                commission=commission,
            )
            st_pnl = _stop_target_pnl(
                path.mids,
                entry_bar=local_bar,
                entry_mid=match.mid,
                entry_spread_frac=match.spread_fraction,
                session_end_bar=session_end_bar,
                commission=commission,
            )
            key = f"{sel.teacher_name}_{sel.direction}"
            bar.labels.exit_headroom_by_selection[key] = {
                "best_exit_pnl": float(best_pnl) if np.isfinite(best_pnl) else None,
                "worst_exit_pnl": float(worst_pnl) if np.isfinite(worst_pnl) else None,
                "time_stop_pnl": float(ts_pnl) if ts_pnl is not None and np.isfinite(ts_pnl) else None,
                "stop_target_pnl": float(st_pnl) if st_pnl is not None and np.isfinite(st_pnl) else None,
            }
