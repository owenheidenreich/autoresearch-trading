"""Sanity check: sweep over fixed exit-bar strategies on chosen trades.

If the L3 oracle's per-trade exit predictions are valuable, they should
beat any constant exit-bar strategy. If a fixed exit-bar (e.g., always
exit at bar 90, or always at session-end-30) matches or beats the
oracle, the per-trade prediction adds no value.

Builds per-bar pnl trajectory using v3/oracles/opportunity._build_contract_paths
and walks each chosen trade bar-by-bar from entry to session end.

Output: v3/artifacts/research/fw_trade_trajectories.pkl + console sweep.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import build_labeled_day
from v3.oracles.opportunity import _build_contract_paths, _contract_idx_for_record


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def build_trade_trajectory(trade_row, log, sidecar, paths_cache, n_bars=390, session_end_bar=375):
    """Per-bar pnl trajectory matching v3/oracles/exit_headroom._time_stop_pnl."""
    bar_index = int(trade_row["bar_index"])
    chosen_strike = float(trade_row.get("chosen_strike", 0))
    chosen_side = str(trade_row.get("chosen_side", ""))
    if chosen_strike <= 0 or chosen_side not in ("call", "put"):
        return None
    side_char = "C" if chosen_side == "call" else "P"

    bar = next((b for b in log.bars if b.bar_index == bar_index), None)
    if bar is None: return None
    match = next((c for c in bar.contracts
                  if abs(float(c.strike) - chosen_strike) < 0.01 and c.right == side_char), None)
    if match is None: return None

    cache_key = id(sidecar)
    if cache_key not in paths_cache:
        paths_cache[cache_key] = _build_contract_paths(sidecar, n_bars)
    paths = paths_cache[cache_key]
    cid = _contract_idx_for_record(sidecar, bar_index, match)
    if cid is None or cid not in paths: return None
    path = paths[cid]

    entry_mid = float(match.mid)
    entry_sf = float(match.spread_fraction)
    if not np.isfinite(entry_mid) or entry_mid <= 0: return None
    if not np.isfinite(entry_sf): entry_sf = 0.05

    # Pnl matching _time_stop_pnl: entry_ask = mid*(1+sf/2), exit_bid = mid*(1-sf/2)
    # PnL = 100*(exit_bid - entry_ask) - commission
    entry_ask = entry_mid * (1.0 + entry_sf / 2.0)
    commission = 1.0  # round-trip
    end = min(session_end_bar, n_bars - 1)
    pnl_per_bar = []
    for b in range(bar_index + 1, end + 1):
        mid_b = path.mids[b]
        if not np.isfinite(mid_b): continue
        sf_b = path.spread_fracs[b] if np.isfinite(path.spread_fracs[b]) else entry_sf
        exit_bid = mid_b * (1.0 - sf_b / 2.0)
        pnl = float(100.0 * (exit_bid - entry_ask) - commission)
        pnl_per_bar.append({"bar": b, "bar_offset": b - bar_index, "pnl": pnl})

    if not pnl_per_bar: return None
    return {
        "trade_id": f"{trade_row['day']}_{bar_index}_{chosen_side}",
        "day": str(trade_row["day"]),
        "bar_index": bar_index,
        "chosen_side": chosen_side,
        "chosen_strike": chosen_strike,
        "entry_mid": entry_mid,
        "entry_sf": entry_sf,
        "pnl_per_bar": pnl_per_bar,
    }


def main():
    seeds = [42, 43, 44, 45, 46]
    cfg = GuardrailConfig()
    ds = V2Dataset.load()

    day_cache = {}
    paths_cache = {}
    all_trajectories = []
    skipped = 0

    for s in seeds:
        try:
            df = pd.read_pickle(f"v3/artifacts/forward_walk/forward_walk_chosen_seed{s}.pkl")
        except FileNotFoundError:
            continue
        if df.empty: continue
        df = df.copy(); df["seed"] = s
        for _, row in df.iterrows():
            day = str(row["day"])
            if day not in day_cache:
                log_tmp, sc_tmp = build_labeled_day(ds, day, cfg, equity=25000.0)
                day_cache[day] = (log_tmp, sc_tmp)
            log, sidecar = day_cache[day]
            if log is None or sidecar is None:
                skipped += 1; continue
            traj = build_trade_trajectory(row, log, sidecar, paths_cache)
            if traj is None:
                skipped += 1; continue
            traj["seed"] = s
            traj["fwd_pnl_hybrid_with_oracle"] = float(row.get("fwd_pnl_hybrid_with_oracle", 0))
            traj["fwd_pnl_time_stop"] = float(row.get("fwd_pnl_time_stop", 0))
            all_trajectories.append(traj)

    print(f"Built {len(all_trajectories)} forward-walk trajectories, {skipped} skipped")
    if not all_trajectories:
        print("No trajectories — abort.")
        return

    max_offset = max(max(b["bar_offset"] for b in t["pnl_per_bar"]) for t in all_trajectories)
    print(f"Max bar_offset across trades: {max_offset}")
    print()

    # Fixed exit-bar sweep — note: time_stop is "hold to last bar before 375"
    print(f"=== Fixed-exit-bar sweep (hold for at least N bars from entry) ===")
    print(f"{'exit_at+':>10} {'n':>4} {'PF':>7} {'sum':>10} {'mean':>8} {'median':>8}")
    rows = []
    for offset in [10, 20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 180, 210, 240, 270, 300, max_offset]:
        pnls = []
        for t in all_trajectories:
            best_bar = None
            for b in t["pnl_per_bar"]:
                if b["bar_offset"] >= offset:
                    best_bar = b; break
            if best_bar is None and t["pnl_per_bar"]:
                best_bar = t["pnl_per_bar"][-1]
            if best_bar is not None:
                pnls.append(best_bar["pnl"])
        if not pnls: continue
        pnls = np.asarray(pnls, dtype=float)
        rows.append({"offset": offset, "n": len(pnls), "pf": float(pf(pnls)),
                     "sum": float(pnls.sum()), "mean": float(pnls.mean()),
                     "median": float(np.median(pnls))})
        print(f"{offset:>10} {len(pnls):>4} {pf(pnls):>7.3f} {pnls.sum():>10.0f} {pnls.mean():>8.0f} {np.median(pnls):>8.0f}")
    print()

    # Comparisons
    oracle_pnls = [t["fwd_pnl_hybrid_with_oracle"] for t in all_trajectories]
    ts_pnls_label = [t["fwd_pnl_time_stop"] for t in all_trajectories]
    print(f"=== Comparison vs reported pnls ===")
    print(f"  reported oracle:        PF {pf(oracle_pnls):.3f}, sum ${sum(oracle_pnls):.0f}")
    print(f"  reported time_stop:     PF {pf(ts_pnls_label):.3f}, sum ${sum(ts_pnls_label):.0f}")
    best_fixed = max(rows, key=lambda r: r["pf"])
    print(f"  best fixed at bar+{best_fixed['offset']}:    PF {best_fixed['pf']:.3f}, sum ${best_fixed['sum']:.0f}")

    # Per-trade peak (perfect ceiling)
    perfect_pnls = []
    for t in all_trajectories:
        if t["pnl_per_bar"]:
            perfect_pnls.append(max(b["pnl"] for b in t["pnl_per_bar"]))
    if perfect_pnls:
        print(f"  perfect (per-trade peak): PF {pf(perfect_pnls):.3f}, sum ${sum(perfect_pnls):.0f}")

    # Save
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/fw_trade_trajectories.pkl", "wb") as f:
        pickle.dump(all_trajectories, f)
    print(f"\nWrote v3/artifacts/research/fw_trade_trajectories.pkl ({len(all_trajectories)} trajectories)")


if __name__ == "__main__":
    main()
