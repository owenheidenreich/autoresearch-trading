"""Opportunity oracle: hindsight-best (entry, direction, contract, exit).

Populates `BarRecord.labels` on every bar of a `DayLog`:

- `best_forward_pnl_call` / `best_forward_pnl_put`: for this entry bar, the
  maximum realized PnL achievable by buying the best guardrail-passing
  contract of each direction and selling at the best forward exit bar
  (subject to spread + commission + session-end time stop).
- `worst_forward_pnl_call` / `worst_forward_pnl_put`: same but worst case.
- `mfe_* / mae_*`: max favorable / adverse excursion at 5-, 10-, 20-minute
  horizons per direction.
- `opportunity_oracle_entry`: True on exactly one bar per session — the bar
  whose best-forward-PnL (across both directions) is the session maximum.
- `opportunity_oracle_direction` / `opportunity_oracle_strike` /
  `opportunity_oracle_realized_pnl`: details of that single hindsight-optimal
  trade.

Fills model (keep in sync with the plan's "opportunity oracle" definition):
- Buy at ask ≈ mid × (1 + spread_fraction / 2)
- Sell at bid ≈ mid × (1 - spread_fraction / 2)
- Per-round-trip commission: $1.00 (SPX 0DTE realistic, configurable)
- Session-end cutoff: bar 375 by default (15:45 ET), avoids last-15-min chaos
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from v2.core.chain_data import contract_snapshot
from v3.logger.schema import BarRecord, ContractRecord, DayLog


# CONTRACT_FEATURE_FIELDS column indices (same as in v3/harness/v2_adapter.py)
CIDX_VALID = 0
CIDX_MID = 3
CIDX_SPREAD_FRAC = 4

DEFAULT_SESSION_END_BAR = 375
DEFAULT_COMMISSION_PER_CONTRACT = 1.0  # round-trip $


@dataclass
class _ContractPath:
    """Cached forward-mid path for one contract across a day."""

    contract_idx: int
    mids: np.ndarray         # [n_bars_total] with NaN where not observed
    spread_fracs: np.ndarray  # same shape
    # Precomputed suffix max/min of mids (NaN-aware); index k = max/min of mids[k:]
    suffix_max: np.ndarray
    suffix_min: np.ndarray


def _build_contract_paths(sidecar: dict, n_bars: int) -> dict[int, _ContractPath]:
    """Pivot the sidecar's flat rows into per-contract forward paths.

    Returns a dict keyed by contract_idx with one `_ContractPath` per unique
    contract observed anywhere in the session.
    """
    ptrs = sidecar["bar_ptrs"]
    row_idx = sidecar["row_contract_idx"]
    feats = sidecar["row_features"]

    observed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # First pass: allocate arrays per contract.
    # Partial sessions exist — cap iteration by the sidecar's actual bar count.
    usable_bars = min(n_bars, len(ptrs) - 1)
    for b in range(usable_bars):
        s = int(ptrs[b])
        e = int(ptrs[b + 1])
        if e <= s:
            continue
        for j in range(s, e):
            c = feats[j]
            if c[CIDX_VALID] < 0.5:
                continue
            cid = int(row_idx[j])
            if cid not in observed:
                mids = np.full(n_bars, np.nan, dtype=np.float64)
                sfs = np.full(n_bars, np.nan, dtype=np.float64)
                observed[cid] = (mids, sfs)
            observed[cid][0][b] = float(c[CIDX_MID])
            observed[cid][1][b] = float(c[CIDX_SPREAD_FRAC])

    paths: dict[int, _ContractPath] = {}
    for cid, (mids, sfs) in observed.items():
        suffix_max = _nan_aware_suffix_reduce(mids, np.fmax)
        suffix_min = _nan_aware_suffix_reduce(mids, np.fmin)
        paths[cid] = _ContractPath(
            contract_idx=cid,
            mids=mids,
            spread_fracs=sfs,
            suffix_max=suffix_max,
            suffix_min=suffix_min,
        )
    return paths


def _nan_aware_suffix_reduce(a: np.ndarray, op) -> np.ndarray:
    """Reverse accumulate `op` from the end, skipping NaN.

    suffix[k] = op over a[k:], ignoring NaN. NaN where no finite value.
    np.fmax/np.fmin ignore NaN element-wise, so reversed accumulation works.
    """
    rev = a[::-1]
    out = np.empty_like(rev)
    out[0] = rev[0]
    for i in range(1, len(rev)):
        out[i] = op(out[i - 1], rev[i])
    return out[::-1]


def _contract_idx_for_record(
    sidecar: dict, local_bar: int, contract: ContractRecord
) -> Optional[int]:
    """Look up the sidecar contract_idx matching a ContractRecord at a bar.

    Matches on (strike, right). Returns None if not found (shouldn't happen —
    the ContractRecord was built from this sidecar — but be defensive).
    """
    feats, _, cids = contract_snapshot(sidecar, local_bar)
    if feats.shape[0] == 0:
        return None
    right_is_put = 1.0 if contract.right == "P" else 0.0
    for i in range(feats.shape[0]):
        r = feats[i]
        if r[CIDX_VALID] < 0.5:
            continue
        if abs(float(r[1]) - contract.strike) < 1e-6 and abs(float(r[2]) - right_is_put) < 1e-6:
            return int(cids[i])
    return None


def _best_and_worst_forward_pnl(
    path: _ContractPath,
    entry_bar: int,
    entry_mid: float,
    entry_spread_frac: float,
    session_end_bar: int,
    commission: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return (best_pnl, worst_pnl) from entering long at entry_bar.

    PnL = (exit_bid - entry_ask) × 100 − commission.
    Bid/ask approximated via spread_fraction as a symmetric half-spread.
    Ignores exit bars where the contract isn't observed (NaN in suffix).
    Returns (None, None) if no valid forward exit exists.
    """
    search_end = min(session_end_bar, len(path.mids) - 1)
    if entry_bar + 1 > search_end:
        return None, None
    best_exit_mid = path.suffix_max[entry_bar + 1] if entry_bar + 1 <= search_end else np.nan
    worst_exit_mid = path.suffix_min[entry_bar + 1] if entry_bar + 1 <= search_end else np.nan
    # Suffix includes bars after session_end too; clip by searching explicitly.
    window = path.mids[entry_bar + 1 : search_end + 1]
    if window.size == 0 or np.all(np.isnan(window)):
        return None, None
    best_exit_mid = float(np.nanmax(window))
    worst_exit_mid = float(np.nanmin(window))

    entry_ask = entry_mid * (1.0 + entry_spread_frac / 2.0)
    # Approximation: exit spread ≈ entry spread (keeps the model simple).
    best_exit_bid = best_exit_mid * (1.0 - entry_spread_frac / 2.0)
    worst_exit_bid = worst_exit_mid * (1.0 - entry_spread_frac / 2.0)
    best_pnl = 100.0 * (best_exit_bid - entry_ask) - commission
    worst_pnl = 100.0 * (worst_exit_bid - entry_ask) - commission
    return best_pnl, worst_pnl


def _mfe_mae_at_horizon(
    path: _ContractPath,
    entry_bar: int,
    entry_mid: float,
    horizon_bars: int,
    session_end_bar: int,
) -> tuple[Optional[float], Optional[float]]:
    """Max favorable / adverse excursion over `horizon_bars` after entry.

    Returns (mfe_pnl_pct, mae_pnl_pct) as percentage-of-entry-premium.
    """
    if entry_mid <= 0:
        return None, None
    end = min(entry_bar + horizon_bars, session_end_bar, len(path.mids) - 1)
    window = path.mids[entry_bar + 1 : end + 1]
    if window.size == 0 or np.all(np.isnan(window)):
        return None, None
    max_mid = float(np.nanmax(window))
    min_mid = float(np.nanmin(window))
    mfe = (max_mid - entry_mid) / entry_mid * 100.0
    mae = (min_mid - entry_mid) / entry_mid * 100.0
    return mfe, mae


def apply_opportunity_oracle(
    day_log: DayLog,
    sidecar: dict,
    day_start_abs: int,
    bars_per_day: int,
    session_end_bar: int = DEFAULT_SESSION_END_BAR,
    commission: float = DEFAULT_COMMISSION_PER_CONTRACT,
) -> None:
    """Populate OracleLabels on every BarRecord in day_log, in place.

    `day_start_abs` is the absolute-index offset of bar 0 for this day in the
    dataset. `bars_per_day` is the session length (typically 390).
    """
    paths = _build_contract_paths(sidecar, bars_per_day)

    # Session-level tracker for the single opportunity-oracle entry.
    best_session_pnl = -np.inf
    best_entry_bar: Optional[int] = None
    best_direction: Optional[str] = None
    best_strike: Optional[float] = None
    best_realized: Optional[float] = None

    for bar in day_log.bars:
        local_bar = bar.bar_index  # bar_index is minute-of-session = local bar
        # --- Per-direction forward best/worst ---
        best_call = -np.inf
        worst_call = np.inf
        best_put = -np.inf
        worst_put = np.inf
        best_call_strike = None
        best_put_strike = None
        # --- MFE/MAE over passing contracts, per direction ---
        call_mfe_5 = call_mae_5 = call_mfe_10 = call_mae_10 = call_mfe_20 = call_mae_20 = None
        put_mfe_5 = put_mae_5 = put_mfe_10 = put_mae_10 = put_mfe_20 = put_mae_20 = None

        for c in bar.contracts:
            if not c.passed:
                continue
            cid = _contract_idx_for_record(sidecar, local_bar, c)
            if cid is None or cid not in paths:
                continue
            path = paths[cid]
            best_pnl, worst_pnl = _best_and_worst_forward_pnl(
                path,
                entry_bar=local_bar,
                entry_mid=c.mid,
                entry_spread_frac=c.spread_fraction,
                session_end_bar=session_end_bar,
                commission=commission,
            )
            if best_pnl is None:
                continue

            if c.right == "C":
                if best_pnl > best_call:
                    best_call = best_pnl
                    best_call_strike = c.strike
                    # Horizon excursions for the call winner
                    for hb, (mfe_slot, mae_slot) in (
                        (5, ("call_mfe_5", "call_mae_5")),
                        (10, ("call_mfe_10", "call_mae_10")),
                        (20, ("call_mfe_20", "call_mae_20")),
                    ):
                        mfe, mae = _mfe_mae_at_horizon(path, local_bar, c.mid, hb, session_end_bar)
                        if mfe_slot == "call_mfe_5": call_mfe_5, call_mae_5 = mfe, mae
                        elif mfe_slot == "call_mfe_10": call_mfe_10, call_mae_10 = mfe, mae
                        else: call_mfe_20, call_mae_20 = mfe, mae
                if worst_pnl < worst_call:
                    worst_call = worst_pnl
            else:
                if best_pnl > best_put:
                    best_put = best_pnl
                    best_put_strike = c.strike
                    for hb, (mfe_slot, mae_slot) in (
                        (5, ("put_mfe_5", "put_mae_5")),
                        (10, ("put_mfe_10", "put_mae_10")),
                        (20, ("put_mfe_20", "put_mae_20")),
                    ):
                        mfe, mae = _mfe_mae_at_horizon(path, local_bar, c.mid, hb, session_end_bar)
                        if mfe_slot == "put_mfe_5": put_mfe_5, put_mae_5 = mfe, mae
                        elif mfe_slot == "put_mfe_10": put_mfe_10, put_mae_10 = mfe, mae
                        else: put_mfe_20, put_mae_20 = mfe, mae
                if worst_pnl < worst_put:
                    worst_put = worst_pnl

        # Write to labels (None where no passing contract of that direction existed)
        lbl = bar.labels
        lbl.best_forward_pnl_call = best_call if np.isfinite(best_call) else None
        lbl.best_forward_pnl_put = best_put if np.isfinite(best_put) else None
        lbl.worst_forward_pnl_call = worst_call if np.isfinite(worst_call) else None
        lbl.worst_forward_pnl_put = worst_put if np.isfinite(worst_put) else None
        lbl.mfe_5min_call, lbl.mae_5min_call = call_mfe_5, call_mae_5
        lbl.mfe_10min_call, lbl.mae_10min_call = call_mfe_10, call_mae_10
        lbl.mfe_20min_call, lbl.mae_20min_call = call_mfe_20, call_mae_20
        lbl.mfe_5min_put, lbl.mae_5min_put = put_mfe_5, put_mae_5
        lbl.mfe_10min_put, lbl.mae_10min_put = put_mfe_10, put_mae_10
        lbl.mfe_20min_put, lbl.mae_20min_put = put_mfe_20, put_mae_20

        # Track the session maximum for opportunity_oracle_entry
        bar_best = max(
            best_call if np.isfinite(best_call) else -np.inf,
            best_put if np.isfinite(best_put) else -np.inf,
        )
        if bar_best > best_session_pnl:
            best_session_pnl = bar_best
            best_entry_bar = bar.bar_index
            if np.isfinite(best_call) and best_call >= best_put:
                best_direction = "call"
                best_strike = best_call_strike
                best_realized = best_call
            else:
                best_direction = "put"
                best_strike = best_put_strike
                best_realized = best_put

    # Second pass: flag the single session-best entry bar
    if best_entry_bar is not None:
        for bar in day_log.bars:
            if bar.bar_index == best_entry_bar:
                bar.labels.opportunity_oracle_entry = True
                bar.labels.opportunity_oracle_direction = best_direction
                bar.labels.opportunity_oracle_strike = best_strike
                bar.labels.opportunity_oracle_realized_pnl = best_realized
            else:
                bar.labels.opportunity_oracle_entry = False
