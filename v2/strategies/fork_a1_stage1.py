"""Fork A1 Stage 1 runner — entry validity only, no exits.

Walks the 4-year RTH dataset bar-by-bar. For each bar in the trading window
(09:45-12:00 ET, bars 15-150), checks Pickles Row-1 preconditions + entry
trigger across the ablation matrix. Logs every candidate bar (even skipped)
and measures forward MFE/MAE on SPX spot and on the delta-targeted option
contract at 5/10/20/30-minute horizons.

**Does not** fire exits. **Does not** route through ``simulate_trade`` or
``simulate_day``. Those entry-point masks would silently drop bars 15-29
and apply premium-based exits incompatible with Row 1.

Outputs:
    v2/artifacts/fork_a1_stage1/candidates_<git_sha>.parquet
    v2/artifacts/fork_a1_stage1/summary_<git_sha>.json
    v2/artifacts/fork_a1_stage1/provenance_<git_sha>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v2.core.chain_data import load_sidecar_cached
from v2.core.provenance import build_provenance, write_provenance
from v2.strategies.pickles_row1 import (
    AblationConfig,
    CandidateRecord,
    ContractPick,
    ALL_PRECONDITION_CELLS,
    P_OPEN,
    VWAP_RALLY_LOOKBACK_BARS,
    assert_feature_layout,
    entry_trigger_fires,
    is_in_window,
    pick_delta_targeted_call,
    preconditions_ok,
    _FEAT_IDX_VWAP_DIST,
    _FEAT_IDX_FIRST15_CLOSE_POS,
)
from v2.strategies.session_state import (
    BARS_PER_DAY,
    FIRST15_BARS,
    SessionState,
    WINDOW_FIRST_BAR,
    WINDOW_LAST_BAR,
    compute_daily_vwap_sigma,
    compute_session_vwap,
    in_halfhour_cooldown,
)


FORWARD_HORIZONS = (5, 10, 20, 30)
DEFAULT_DELTA_GRID = (0.40, 0.50)
EXTENDED_DELTA_GRID = (0.30, 0.40, 0.50)
FULL_DELTA_GRID = (0.20, 0.30, 0.40, 0.50)


@dataclass
class ForwardMetrics:
    """Per-(candidate × horizon) forward excursion on SPX spot and the option.

    All values are fractions (not bps). NaN when the horizon extends past
    session close and no valid prints exist.
    """
    horizon: int
    spot_mfe: float
    spot_mae: float
    spot_end_ret: float
    opt_mfe: float
    opt_mae: float
    opt_end_ret: float
    n_valid_spot_bars: int
    n_valid_opt_bars: int


def _compute_forward_metrics(
    entry_global_bar: int,
    day_last_bar_global: int,
    spot_prices: np.ndarray,
    entry_spot: float,
    option_mid: np.ndarray | None,
    local_bar: int,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> list[ForwardMetrics]:
    """Forward MFE/MAE/end-return per horizon, capped at session close.

    Entry fills conceptually at the close of entry_global_bar; measurement
    starts at the NEXT bar and runs for ``horizon`` bars, capped at
    day_last_bar_global (inclusive).

    option_mid is the full-day contract-mid array (shape (n_bars,)); indexed
    by local bar. None if contract selection skipped.
    """
    out: list[ForwardMetrics] = []
    for h in horizons:
        start = entry_global_bar + 1
        end = min(entry_global_bar + h, day_last_bar_global)
        if end < start:
            out.append(ForwardMetrics(h, *([float("nan")] * 6), 0, 0))
            continue
        spot_path = spot_prices[start:end + 1].astype(np.float64)
        spot_valid = np.isfinite(spot_path) & (spot_path > 0)
        if spot_valid.any() and np.isfinite(entry_spot) and entry_spot > 0:
            sp = spot_path[spot_valid]
            spot_mfe = float(sp.max() - entry_spot) / entry_spot
            spot_mae = float(sp.min() - entry_spot) / entry_spot
            spot_end_ret = float(sp[-1] - entry_spot) / entry_spot
        else:
            spot_mfe = spot_mae = spot_end_ret = float("nan")
        n_spot = int(spot_valid.sum())

        if option_mid is not None:
            local_start = local_bar + 1
            local_end = local_bar + (end - entry_global_bar)
            opt_path = option_mid[local_start:local_end + 1].astype(np.float64)
            opt_valid = np.isfinite(opt_path) & (opt_path > 0)
            if opt_valid.any():
                op = opt_path[opt_valid]
                entry_mid = float(option_mid[local_bar])
                if np.isfinite(entry_mid) and entry_mid > 0:
                    opt_mfe = float(op.max() - entry_mid) / entry_mid
                    opt_mae = float(op.min() - entry_mid) / entry_mid
                    opt_end_ret = float(op[-1] - entry_mid) / entry_mid
                else:
                    opt_mfe = opt_mae = opt_end_ret = float("nan")
            else:
                opt_mfe = opt_mae = opt_end_ret = float("nan")
            n_opt = int(opt_valid.sum())
        else:
            opt_mfe = opt_mae = opt_end_ret = float("nan")
            n_opt = 0

        out.append(ForwardMetrics(h, spot_mfe, spot_mae, spot_end_ret,
                                  opt_mfe, opt_mae, opt_end_ret, n_spot, n_opt))
    return out


def _group_by_day(dates: list[str], bar_of_day: np.ndarray) -> list[tuple[str, int, int]]:
    """Return list of (date, start_idx, end_idx_inclusive) per session.

    Groups by consecutive equal ``dates[i]`` — tolerates half-day sessions
    (e.g. 2023-07-03, 2023-11-24) that run 210 bars ending 13:00 ET instead
    of 390. Still asserts each session begins at ``bar_of_day == 0``.
    """
    days: list[tuple[str, int, int]] = []
    n = len(dates)
    i = 0
    while i < n:
        if int(bar_of_day[i]) != 0:
            raise RuntimeError(
                f"Unexpected bar_of_day at row {i} ({dates[i]}): "
                f"{int(bar_of_day[i])} (expected 0)."
            )
        j = i
        while j + 1 < n and dates[j + 1] == dates[i]:
            j += 1
        days.append((dates[i], i, j))
        i = j + 1
    return days


def _load_volume_per_bar_for_day(sidecar: dict[str, Any]) -> np.ndarray:
    """Approximate per-bar SPX volume from the chain transactions aggregate.

    The sidecar doesn't carry SPX volume directly; it carries contract-level
    volume. For session VWAP we ideally want SPX minute volume, but that is
    not preserved into v2/data.pt's tensor.

    Workaround: use the total near-ATM transactions per bar as a proxy (matches
    the feature-pipeline convention used by ``vwap_dist``). If the sidecar
    spot_series plus a uniform volume is used, cumulative VWAP collapses to
    a simple cumulative price mean, which is what the pipeline's vwap_arr
    actually computes when volume is ~constant.

    To stay strictly point-in-time faithful to the pipeline's vwap_dist, we
    simply back-derive VWAP from raw spot + the ``X_sim[:, vwap_dist]`` column:
    ``vwap_t = spot_t / (1 - vwap_dist_frac_t)``. The Stage 1 runner does this
    directly in the main loop (see ``_recover_vwap_from_features``), making
    this function unused at runtime but retained for documentation.
    """
    return np.ones(sidecar["n_bars"], dtype=np.float64)


def _recover_vwap_from_features(spot: float, vwap_dist_frac: float) -> float:
    """Invert ``vwap_dist_frac = (spot - vwap) / spot`` to recover raw VWAP.

    Using the pipeline's already-computed VWAP avoids any drift between our
    runtime VWAP and the feature pipeline's (which is what ``X_sim[vwap_dist]``
    encodes). Returns 0.0 on degenerate input.
    """
    if not np.isfinite(spot) or not np.isfinite(vwap_dist_frac) or spot <= 0:
        return 0.0
    return float(spot * (1.0 - vwap_dist_frac))


def run_stage1(
    data_path: str,
    delta_grid: tuple[float, ...],
    out_dir: str,
    screen_days: int | None = None,
) -> dict[str, Any]:
    """Run the Stage 1 entry-validity pass across the dataset.

    Parameters
    ----------
    data_path:
        Path to ``v2/data.pt``.
    delta_grid:
        Target-delta buckets to test. Default: {0.40, 0.50}.
    out_dir:
        Directory for candidates parquet + summary JSON + provenance JSON.
    screen_days:
        If set, run only the last ``screen_days`` sessions (fast dev loop).
    """
    t0 = time.time()
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    feature_names = data["feature_names"]
    assert_feature_layout(feature_names)

    X_sim = data["X_sim"].numpy()  # (N, 79) raw features
    bar_of_day = data["bar_of_day"].numpy()
    dates = list(data["dates"])
    spot_prices = data["spot_prices"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dataset_fingerprint = str(data["metadata"].get("fingerprint", "unknown"))
    sidecar_schema_version = str(data["metadata"].get("chain_schema_version", "unknown"))

    sessions = _group_by_day(dates, bar_of_day)
    if screen_days is not None:
        sessions = sessions[-screen_days:]
    n_sessions = len(sessions)
    print(f"Stage 1 runner: {n_sessions} sessions, deltas={delta_grid}, "
          f"window=bars[{WINDOW_FIRST_BAR}..{WINDOW_LAST_BAR}]")

    # Precompute σ-bands per day from prior sessions (point-in-time).
    spot_by_day: list[np.ndarray] = []
    vol_by_day: list[np.ndarray] = []
    for _, start, end in sessions:
        spot_by_day.append(spot_prices[start:end + 1].astype(np.float64))
        # Volume proxy: use a uniform array since we'll recover VWAP from
        # X_sim[vwap_dist] directly during the loop. The sigma computation
        # below uses spot-only dispersion, which is approximately the same
        # as dispersion of (spot - vwap) / spot given how vwap is computed.
        vol_by_day.append(np.ones_like(spot_by_day[-1]))
    sigmas = compute_daily_vwap_sigma(spot_by_day, vol_by_day)

    # Main loop. For each session, for each bar in the window, check
    # trigger + preconditions across cells, log candidates, measure forward.
    all_records: list[dict[str, Any]] = []
    all_forward: list[dict[str, Any]] = []
    state = SessionState(date="")

    # Prior-session close spot (for ovn_proxy). First session has no prior.
    prior_close_by_session = [None] * n_sessions
    for i in range(1, n_sessions):
        prev_start, prev_end = sessions[i - 1][1], sessions[i - 1][2]
        prior_close_by_session[i] = float(spot_prices[prev_end])

    cand_count = 0
    for si, (day, day_start, day_end) in enumerate(sessions):
        state.reset_for_day(day, prior_close_by_session[si] or 0.0)
        state.session_open_spot = float(spot_prices[day_start])
        state.vwap_sigma_frac = float(sigmas[si]) if np.isfinite(sigmas[si]) else 0.0

        # OVN proxy: sign of (today_open - prior_close). +1 long, -1 short, 0 flat/unknown.
        if prior_close_by_session[si] is not None and state.session_open_spot > 0:
            diff = state.session_open_spot - prior_close_by_session[si]
            state.ovn_proxy_direction = int(np.sign(diff))
        else:
            state.ovn_proxy_direction = 0

        # Lazy-load sidecar.
        sidecar_path = os.path.join(sidecar_dir, f"{day}.pt")
        if not os.path.exists(sidecar_path):
            continue
        sidecar = load_sidecar_cached(sidecar_path)
        n_bars_sc = int(sidecar.get("n_bars", BARS_PER_DAY))

        # Walk every bar in the day, updating state, and within the window
        # log candidates. Session length may be < BARS_PER_DAY on half-days.
        session_length = day_end - day_start + 1
        for local_bar in range(session_length):
            global_bar = day_start + local_bar

            # Update prior_bar_vwap_dist AFTER we finish using it for this bar.
            # At bar 0 there's no prior; we can only start firing at bar >= 1.
            vwap_dist_frac = float(X_sim[global_bar, _FEAT_IDX_VWAP_DIST])

            # Cache first-15m stats at the close of bar 14.
            if local_bar == FIRST15_BARS - 1:
                seg = spot_prices[day_start:day_start + FIRST15_BARS].astype(np.float64)
                if seg.size > 0 and np.isfinite(seg).all():
                    state.first15_high = float(seg.max())
                    state.first15_low = float(seg.min())
                    state.first15_close = float(seg[-1])
                    state.first15_open = float(seg[0])
                    span = state.first15_high - state.first15_low
                    if span > 0:
                        state.first15_close_position = (state.first15_close - state.first15_low) / span
                    else:
                        state.first15_close_position = 0.5
                    state.first15_ready = True

            if not is_in_window(local_bar):
                # Still update the rolling buffer so that when we enter the
                # window we can evaluate the "session recently above VWAP" cond.
                if np.isfinite(vwap_dist_frac):
                    state.recent_vwap_dist_buffer.append(float(vwap_dist_frac))
                    if len(state.recent_vwap_dist_buffer) > VWAP_RALLY_LOOKBACK_BARS:
                        state.recent_vwap_dist_buffer.pop(0)
                    state.recent_window_ready = True
                continue

            # At this point, we're in the window. Compute the recent-max BEFORE
            # appending the current bar so the trigger only consults prior bars.
            if state.recent_vwap_dist_buffer:
                recent_max = max(state.recent_vwap_dist_buffer)
            else:
                recent_max = float("-inf")

            # Entry trigger (same check for all cells).
            X_row = X_sim[global_bar]
            trigger_fires = entry_trigger_fires(
                bar_of_day=local_bar,
                X_sim_row=X_row,
                recent_max_vwap_dist_frac=recent_max,
                recent_window_ready=state.recent_window_ready,
            )

            spot_t = float(spot_prices[global_bar])
            vwap_t = _recover_vwap_from_features(spot_t, vwap_dist_frac)
            in_cd = in_halfhour_cooldown(local_bar)

            if trigger_fires:
                for cell in ALL_PRECONDITION_CELLS:
                    ok, skip = preconditions_ok(local_bar, X_row, state, cell)
                    for td in delta_grid:
                        rec = CandidateRecord(
                            date=day,
                            bar_of_day=local_bar,
                            global_bar=global_bar,
                            ablation_cell=cell.label,
                            target_delta=td,
                            spx_spot=spot_t,
                            session_vwap=vwap_t,
                            vwap_dist_frac=vwap_dist_frac,
                            vwap_sigma_frac=state.vwap_sigma_frac,
                            first15_high=state.first15_high,
                            first15_close_position=state.first15_close_position or float("nan"),
                            ovn_proxy_direction=state.ovn_proxy_direction,
                            in_cooldown=in_cd,
                            skip_reason="" if ok else skip,
                        )
                        pick: ContractPick | None = None
                        opt_mid_arr: np.ndarray | None = None
                        if ok:
                            pick, pick_skip = pick_delta_targeted_call(sidecar, local_bar, td)
                            if pick is None:
                                rec.skip_reason = pick_skip or "no_delta_match"
                            else:
                                rec.contract_idx = pick.contract_idx
                                rec.contract_strike = pick.strike
                                rec.contract_right = pick.right
                                rec.contract_delta = pick.delta
                                rec.contract_mid_at_entry = pick.mid_at_entry
                                opt_mid_arr = np.asarray(
                                    sidecar["contract_mid"][pick.contract_idx], dtype=np.float64
                                )

                        all_records.append(rec.to_dict())
                        cand_count += 1

                        # Forward MFE/MAE only when the trade would have been taken.
                        if ok and pick is not None:
                            fwd = _compute_forward_metrics(
                                entry_global_bar=global_bar,
                                day_last_bar_global=day_end,
                                spot_prices=spot_prices,
                                entry_spot=spot_t,
                                option_mid=opt_mid_arr,
                                local_bar=local_bar,
                            )
                            for fm in fwd:
                                all_forward.append({
                                    "date": day, "bar_of_day": local_bar, "global_bar": global_bar,
                                    "ablation_cell": cell.label, "target_delta": td,
                                    "horizon": fm.horizon,
                                    "spot_mfe": fm.spot_mfe, "spot_mae": fm.spot_mae,
                                    "spot_end_ret": fm.spot_end_ret,
                                    "opt_mfe": fm.opt_mfe, "opt_mae": fm.opt_mae,
                                    "opt_end_ret": fm.opt_end_ret,
                                    "n_valid_spot_bars": fm.n_valid_spot_bars,
                                    "n_valid_opt_bars": fm.n_valid_opt_bars,
                                    "contract_idx": pick.contract_idx,
                                    "contract_strike": pick.strike,
                                    "contract_delta": pick.delta,
                                    "contract_mid_at_entry": pick.mid_at_entry,
                                })

            # Roll the lookback buffer AFTER all checks (so this bar's
            # vwap_dist is available to the NEXT bar's trigger evaluation).
            if np.isfinite(vwap_dist_frac):
                state.recent_vwap_dist_buffer.append(float(vwap_dist_frac))
                if len(state.recent_vwap_dist_buffer) > VWAP_RALLY_LOOKBACK_BARS:
                    state.recent_vwap_dist_buffer.pop(0)
                state.recent_window_ready = True

        if (si + 1) % 100 == 0:
            print(f"  session {si + 1}/{n_sessions}: {day}, candidates so far={cand_count}, "
                  f"elapsed={time.time() - t0:.1f}s")

    print(f"Stage 1 loop done: {cand_count} records, "
          f"{len(all_forward)} forward rows, elapsed={time.time() - t0:.1f}s")

    # Emit artifacts.
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cand_df = pd.DataFrame(all_records)
    fwd_df = pd.DataFrame(all_forward)

    cand_path = os.path.join(out_dir, "candidates.csv")
    fwd_path = os.path.join(out_dir, "forward_metrics.csv")
    cand_df.to_csv(cand_path, index=False)
    fwd_df.to_csv(fwd_path, index=False)

    # Summary: per-cell × per-delta × per-horizon hit-rate tables.
    summary = _build_summary(cand_df, fwd_df)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    # Provenance.
    prov = build_provenance(
        data_path=data_path,
        feature_names=list(feature_names),
        label_mode="fork_a1_stage1_row1_spx_proxy",
        label_thresholds={
            "vwap_touch_band_frac": 5e-4,
            "vwap_prior_above_frac": 1e-3,
            "volume_ratio_min": 1.0,
            "delta_tolerance": 0.05,
            "min_contract_mid": 0.50,
        },
        screen_mode="full" if screen_days is None else f"last_{screen_days}",
        n_folds=0,  # not a fold-based experiment
        research_tier=False,  # mechanical backtest, not a learned research label
        dataset_fingerprint=dataset_fingerprint,
        sidecar_schema_version=sidecar_schema_version,
    )
    prov_path = os.path.join(out_dir, "provenance.json")
    write_provenance(prov_path, prov, extra={
        "delta_grid": list(delta_grid),
        "n_sessions": n_sessions,
        "n_candidate_records": len(cand_df),
        "n_forward_records": len(fwd_df),
        "elapsed_sec": round(time.time() - t0, 1),
    })

    print(f"Written: {cand_path}, {fwd_path}, {summary_path}, {prov_path}")
    return {"summary": summary, "paths": {
        "candidates": cand_path, "forward": fwd_path,
        "summary": summary_path, "provenance": prov_path,
    }}


def _build_summary(cand_df: pd.DataFrame, fwd_df: pd.DataFrame) -> dict[str, Any]:
    """Per-cell × per-delta × per-horizon rollups."""
    out: dict[str, Any] = {
        "n_candidate_records": int(len(cand_df)),
        "n_forward_records": int(len(fwd_df)),
    }
    if cand_df.empty:
        return out

    # Skip-reason histogram.
    out["skip_reason_counts"] = (
        cand_df.groupby(["ablation_cell", "skip_reason"]).size().unstack(fill_value=0).to_dict()
    )

    # Per-cell × per-delta: count of actual (taken) trades.
    taken = cand_df[cand_df["skip_reason"] == ""]
    out["n_trades_by_cell_delta"] = (
        taken.groupby(["ablation_cell", "target_delta"]).size().unstack(fill_value=0).to_dict()
    )

    if fwd_df.empty:
        return out

    # Hit-rate table: for each (cell, delta, horizon), the fraction of trades
    # whose spot_mfe exceeds +0.3% / +0.5% / +1.0% / +1.5% and whose spot_mae
    # exceeds -0.3%. Also median opt return at each horizon.
    rows: list[dict[str, Any]] = []
    groups = fwd_df.groupby(["ablation_cell", "target_delta", "horizon"])
    for (cell, delta, horizon), g in groups:
        n = int(len(g))
        def frac(s: pd.Series) -> float:
            return float(s.mean()) if n > 0 else float("nan")
        rows.append({
            "ablation_cell": cell,
            "target_delta": float(delta),
            "horizon": int(horizon),
            "n": n,
            "spot_mfe_ge_30bps": frac(g["spot_mfe"] >= 3e-3),
            "spot_mfe_ge_50bps": frac(g["spot_mfe"] >= 5e-3),
            "spot_mfe_ge_100bps": frac(g["spot_mfe"] >= 1e-2),
            "spot_mfe_ge_150bps": frac(g["spot_mfe"] >= 1.5e-2),
            "spot_mae_le_neg30bps": frac(g["spot_mae"] <= -3e-3),
            "spot_end_ret_pos": frac(g["spot_end_ret"] > 0),
            "opt_end_ret_pos": frac(g["opt_end_ret"] > 0),
            "spot_mfe_median": float(g["spot_mfe"].median()),
            "spot_mae_median": float(g["spot_mae"].median()),
            "opt_mfe_median": float(g["opt_mfe"].median()),
            "opt_mae_median": float(g["opt_mae"].median()),
            "opt_end_ret_median": float(g["opt_end_ret"].median()),
        })
    out["hit_rate_table"] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default="v2/artifacts/fork_a1_stage1")
    ap.add_argument("--deltas", default="0.40,0.50",
                    help="Comma-separated delta targets. Default: primary grid {0.40, 0.50}.")
    ap.add_argument("--screen-days", type=int, default=None,
                    help="If set, run only the last N sessions (for quick iteration).")
    args = ap.parse_args()
    delta_grid = tuple(float(x.strip()) for x in args.deltas.split(",") if x.strip())
    run_stage1(args.data, delta_grid, args.out_dir, screen_days=args.screen_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
