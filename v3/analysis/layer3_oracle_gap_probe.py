"""Stage 1 of the Layer 3 (learned exits) workstream.

Question: how much PnL does the time-stop exit leave on the table
relative to a hindsight-optimal exit on each chosen trade's contract?
If the gap is small, exit timing isn't a lever and the workstream dies
at this gate.

For each of the 275 chosen trades in the detach-side baseline:
1. Recover the chosen contract via select_contract().
2. Pull the contract's mid + spread_frac trajectory over the day.
3. Compute three PnLs per trade:
   - asymmetric_time_stop: existing pipeline (uses entry's spread_frac
     for the exit leg too — matches the published 1.455 PF). Sanity
     check: must match layer2_trades.csv pnl values.
   - corrected_time_stop: hold to last finite bar; apply the EXIT bar's
     spread_frac to the exit leg. Honest baseline.
   - oracle_exit: max over all valid exit bars [entry+1, session_end]
     of the per-bar PnL using exit-bar spread. The hindsight ceiling.
4. Aggregate per-fold + overall PF/DD; report per-trade gap distribution
   and exit-bar timing histogram for the oracle.

Verdict gates (per the Layer 3 plan):
- GO: oracle_PF − corrected_time_stop_PF >= +0.30
- STOP: gap < +0.10
- MARGINAL: gap in [+0.10, +0.30]

Run:
    python -m v3.analysis.layer3_oracle_gap_probe \
        --baseline-run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_labeled_day,
    load_export_bundle,
    replay_metrics_from_pnls,
)
from v3.logger.builder import select_contract
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
    _time_stop_pnl,
)
from v3.oracles.opportunity import (
    _build_contract_paths,
    _contract_idx_for_record,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_oracle_gap_probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer 3 oracle-gap probe (Stage 1).")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--session-end-bar", type=int, default=DEFAULT_SESSION_END_BAR)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_CONTRACT)
    return p.parse_args()


def _corrected_time_stop_pnl(
    mids: np.ndarray,
    spread_fracs: np.ndarray,
    entry_bar: int,
    entry_mid: float,
    entry_spread_frac: float,
    session_end_bar: int,
    commission: float,
) -> tuple[float | None, int | None]:
    """Hold-to-end PnL using the EXIT bar's spread_frac for the exit leg.
    Returns (pnl, exit_bar) or (None, None) if no valid exit observation."""
    end = min(session_end_bar, len(mids) - 1)
    window_mids = mids[entry_bar + 1 : end + 1]
    window_sfs = spread_fracs[entry_bar + 1 : end + 1]
    finite = np.where(np.isfinite(window_mids))[0]
    if finite.size == 0:
        return None, None
    last = int(finite[-1])
    exit_mid = float(window_mids[last])
    exit_sf = float(window_sfs[last]) if np.isfinite(window_sfs[last]) else float(entry_spread_frac)
    entry_ask = entry_mid * (1.0 + entry_spread_frac / 2.0)
    exit_bid = exit_mid * (1.0 - exit_sf / 2.0)
    pnl = 100.0 * (exit_bid - entry_ask) - commission
    return float(pnl), int(entry_bar + 1 + last)


def _oracle_exit_pnl(
    mids: np.ndarray,
    spread_fracs: np.ndarray,
    entry_bar: int,
    entry_mid: float,
    entry_spread_frac: float,
    session_end_bar: int,
    commission: float,
) -> tuple[float | None, int | None]:
    """Hindsight-best exit: scan every valid post-entry bar and return
    the highest PnL using that bar's spread_frac. Returns (pnl, exit_bar)
    or (None, None)."""
    end = min(session_end_bar, len(mids) - 1)
    if entry_bar + 1 > end:
        return None, None
    entry_ask = entry_mid * (1.0 + entry_spread_frac / 2.0)
    best_pnl: float | None = None
    best_exit_bar: int | None = None
    for t in range(entry_bar + 1, end + 1):
        m = mids[t]
        if not np.isfinite(m):
            continue
        sf = spread_fracs[t]
        if not np.isfinite(sf):
            sf = entry_spread_frac
        exit_bid = float(m) * (1.0 - float(sf) / 2.0)
        pnl = 100.0 * (exit_bid - entry_ask) - commission
        if best_pnl is None or pnl > best_pnl:
            best_pnl = float(pnl)
            best_exit_bar = int(t)
    return best_pnl, best_exit_bar


def _process_trade(
    trade: pd.Series,
    log,
    sidecar: dict,
    paths_cache: dict,
    session_end_bar: int,
    commission: float,
) -> dict[str, Any] | None:
    """For one trade, compute the three PnLs + oracle exit-bar timing."""
    bar_index = int(trade["bar_index"])
    direction = str(trade["direction"])
    bar = next((b for b in log.bars if b.bar_index == bar_index), None)
    if bar is None:
        return None
    selection = select_contract(bar.contracts, direction, "layer2")
    if selection is None:
        return None
    right = "P" if direction == "put" else "C"
    match = next(
        (c for c in bar.contracts if c.strike == selection.strike and c.right == right),
        None,
    )
    if match is None:
        return None
    paths = paths_cache.get(id(sidecar))
    if paths is None:
        paths = _build_contract_paths(sidecar, 390)
        paths_cache[id(sidecar)] = paths
    cid = _contract_idx_for_record(sidecar, bar_index, match)
    if cid is None or cid not in paths:
        return None
    path = paths[cid]

    asym = _time_stop_pnl(
        path.mids, entry_bar=bar_index, entry_mid=match.mid,
        entry_spread_frac=match.spread_fraction,
        session_end_bar=session_end_bar, commission=commission,
    )
    corrected, ts_exit_bar = _corrected_time_stop_pnl(
        path.mids, path.spread_fracs, entry_bar=bar_index,
        entry_mid=match.mid, entry_spread_frac=match.spread_fraction,
        session_end_bar=session_end_bar, commission=commission,
    )
    oracle, oracle_exit_bar = _oracle_exit_pnl(
        path.mids, path.spread_fracs, entry_bar=bar_index,
        entry_mid=match.mid, entry_spread_frac=match.spread_fraction,
        session_end_bar=session_end_bar, commission=commission,
    )
    if asym is None or corrected is None or oracle is None:
        return None
    return {
        "fold_idx": int(trade["fold_idx"]),
        "day": str(trade["day"]),
        "bar_index": bar_index,
        "direction": direction,
        "csv_pnl": float(trade["pnl"]),
        "asym_pnl": float(asym),
        "corrected_pnl": float(corrected),
        "oracle_pnl": float(oracle),
        "ts_exit_bar": ts_exit_bar,
        "oracle_exit_bar": oracle_exit_bar,
        "bars_held_oracle": (oracle_exit_bar - bar_index) if oracle_exit_bar is not None else None,
        "entry_mid": float(match.mid),
        "entry_spread_frac": float(match.spread_fraction),
    }


def _agg_metrics(trades_df: pd.DataFrame, pnl_col: str, equity: float, total_days: int) -> dict[str, float]:
    if trades_df.empty:
        return {"pf": 0.0, "max_dd_pct": 0.0, "mean_pnl": 0.0, "trades": 0.0, "trades_per_day": 0.0}
    sorted_df = trades_df.sort_values(["day", "bar_index"])
    pnls = sorted_df[pnl_col].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    m["trades"] = float(len(pnls))
    m["trades_per_day"] = float(len(pnls) / max(total_days, 1))
    m["mean_pnl"] = float(np.mean(pnls))
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades_path = os.path.join(args.baseline_run_dir, "layer2_trades.csv")
    trades = pd.read_csv(trades_path)
    print(f"Loaded {len(trades)} trades from {trades_path}")

    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds = list(bundle["meta"]["folds"])
    total_days = sum(len(f["test_days"]) for f in folds)

    print("Loading V2Dataset + per-day labeling (this is the slow part)...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    day_cache: dict[str, tuple] = {}
    paths_cache: dict[int, dict] = {}
    rows: list[dict[str, Any]] = []
    skipped = 0
    for i, trade in trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            day_cache[day] = build_labeled_day(ds, day, cfg, equity=args.equity)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            skipped += 1
            continue
        result = _process_trade(trade, log, sidecar, paths_cache,
                                args.session_end_bar, args.commission)
        if result is None:
            skipped += 1
            continue
        rows.append(result)
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(trades)}...")

    df = pd.DataFrame(rows)
    print(f"Processed {len(df)} trades; {skipped} skipped")
    if df.empty:
        print("ERROR: no trades successfully processed")
        return 1

    # --- Sanity check: asym_pnl should match csv_pnl for every trade ---
    diff = (df["asym_pnl"] - df["csv_pnl"]).abs()
    sanity_max = float(diff.max())
    sanity_mean = float(diff.mean())
    print()
    print(f"Sanity check (asym_pnl vs csv_pnl): max diff ${sanity_max:.4f}, mean diff ${sanity_mean:.4f}")
    if sanity_max > 1.0:
        print(f"  WARNING: large asym/csv discrepancy; pipeline may have drifted")

    # --- Aggregate per-PnL-flavor metrics ---
    print()
    print("=" * 100)
    print("Aggregate metrics by exit policy (chronological sort)")
    print("=" * 100)
    print(f"{'policy':<25}{'PF':>8}{'DD%':>8}{'mean$':>10}{'trades':>8}")
    flavors = [
        ("asym_pnl (csv repro)", "asym_pnl"),
        ("corrected_time_stop", "corrected_pnl"),
        ("oracle_exit", "oracle_pnl"),
    ]
    overall: dict[str, dict[str, float]] = {}
    for label, col in flavors:
        m = _agg_metrics(df, col, args.equity, total_days)
        overall[label] = m
        print(f"{label:<25}{m['pf']:>8.3f}{m['max_dd_pct']:>8.1f}{m['mean_pnl']:>10.1f}{int(m['trades']):>8d}")

    # --- Per-fold breakdown ---
    print()
    print("=" * 100)
    print("Per-fold PF (chronological sort)")
    print("=" * 100)
    print(f"{'fold':<6}{'asym':>10}{'corrected':>12}{'oracle':>10}{'gap_corr_oracle':>18}{'n_trades':>10}")
    fold_breakdown: dict[int, dict[str, float]] = {}
    fold_days = {int(f["fold_idx"]): len(f["test_days"]) for f in folds}
    for fold_idx in sorted(df["fold_idx"].unique()):
        sub = df[df["fold_idx"] == fold_idx]
        n_days = fold_days.get(int(fold_idx), len(sub))
        m_a = _agg_metrics(sub, "asym_pnl", args.equity, n_days)
        m_c = _agg_metrics(sub, "corrected_pnl", args.equity, n_days)
        m_o = _agg_metrics(sub, "oracle_pnl", args.equity, n_days)
        gap = m_o["pf"] - m_c["pf"]
        fold_breakdown[int(fold_idx)] = {
            "asym": m_a["pf"], "corrected": m_c["pf"], "oracle": m_o["pf"],
            "gap_corrected_to_oracle": float(gap),
            "asym_dd": m_a["max_dd_pct"], "corrected_dd": m_c["max_dd_pct"], "oracle_dd": m_o["max_dd_pct"],
            "asym_mean": m_a["mean_pnl"], "corrected_mean": m_c["mean_pnl"], "oracle_mean": m_o["mean_pnl"],
            "n_trades": int(len(sub)),
        }
        print(f"{int(fold_idx):<6}{m_a['pf']:>10.3f}{m_c['pf']:>12.3f}{m_o['pf']:>10.3f}{gap:>18.3f}{int(len(sub)):>10d}")

    # --- Per-trade gap distribution ---
    df["gap_oracle_minus_corrected"] = df["oracle_pnl"] - df["corrected_pnl"]
    print()
    print("=" * 100)
    print("Per-trade oracle - corrected_time_stop gap distribution ($)")
    print("=" * 100)
    g = df["gap_oracle_minus_corrected"]
    print(f"  mean ${g.mean():>+8.1f}  median ${g.median():>+8.1f}  std ${g.std():.1f}")
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
        print(f"  p{int(q*100):>2d}: ${float(g.quantile(q)):>+8.1f}")
    print(f"  n_zero_gap (oracle = corrected): {int((g == 0).sum())}")
    print(f"  n_pos_gap (oracle better): {int((g > 0).sum())}")

    # --- Oracle exit-bar timing distribution ---
    print()
    print("=" * 100)
    print("Oracle exit timing — bars held distribution (oracle_exit_bar - entry_bar)")
    print("=" * 100)
    bh = df["bars_held_oracle"].dropna()
    print(f"  mean {bh.mean():.1f} bars  median {bh.median():.1f}  std {bh.std():.1f}")
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
        print(f"  p{int(q*100):>2d}: {int(bh.quantile(q))} bars")
    # Bin into early / mid / late buckets
    ts_bh = (df["ts_exit_bar"] - df["bar_index"]).dropna()
    print(f"  (compare time_stop bars held: mean {ts_bh.mean():.1f})")

    # --- Verdict ---
    print()
    print("=" * 100)
    print("Verdict criteria (Stage 1)")
    print("=" * 100)
    pf_corrected = overall["corrected_time_stop"]["pf"]
    pf_oracle = overall["oracle_exit"]["pf"]
    pf_gap = pf_oracle - pf_corrected
    print(f"  Corrected time-stop PF: {pf_corrected:.3f}")
    print(f"  Oracle exit PF:        {pf_oracle:.3f}")
    print(f"  PF gap:                {pf_gap:+.3f}")
    if pf_gap >= 0.30:
        verdict = f"GO -- oracle PF gap {pf_gap:+.3f} >= +0.30; learned exits have room to capture lift"
    elif pf_gap < 0.10:
        verdict = f"STOP -- oracle PF gap {pf_gap:+.3f} < +0.10; exit timing is not a lever"
    else:
        verdict = f"MARGINAL -- oracle PF gap {pf_gap:+.3f} in [+0.10, +0.30); proceed with caution"
    print(f"  VERDICT: {verdict}")

    # --- Save artifact ---
    df.to_csv(os.path.join(args.out_dir, "per_trade.csv"), index=False)
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_trades": int(len(df)),
            "n_skipped": int(skipped),
            "session_end_bar": int(args.session_end_bar),
            "commission": float(args.commission),
            "sanity_max_diff_dollars": float(sanity_max),
            "sanity_mean_diff_dollars": float(sanity_mean),
        },
        "overall": overall,
        "per_fold": fold_breakdown,
        "gap_distribution_dollars": {
            "mean": float(g.mean()),
            "median": float(g.median()),
            "std": float(g.std()),
            "p05": float(g.quantile(0.05)),
            "p25": float(g.quantile(0.25)),
            "p50": float(g.quantile(0.50)),
            "p75": float(g.quantile(0.75)),
            "p95": float(g.quantile(0.95)),
            "n_zero_gap": int((g == 0).sum()),
            "n_pos_gap": int((g > 0).sum()),
        },
        "oracle_bars_held_distribution": {
            "mean": float(bh.mean()),
            "median": float(bh.median()),
            "std": float(bh.std()),
            "p05": int(bh.quantile(0.05)),
            "p25": int(bh.quantile(0.25)),
            "p50": int(bh.quantile(0.50)),
            "p75": int(bh.quantile(0.75)),
            "p95": int(bh.quantile(0.95)),
            "ts_mean_bars_held": float(ts_bh.mean()),
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "oracle_gap_probe.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print()
    print(f"Saved: {out}")
    print(f"Saved: {os.path.join(args.out_dir, 'per_trade.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
