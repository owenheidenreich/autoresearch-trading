"""H3e oracle evaluation: spread-minimization is the primary gate.

Compares H3e oracle (target reshape + 4 chosen-contract Greeks) against:
  - balanced_fresh baseline (7-feature trade_state, current production)
  - h3a (10-feature, regime-conditional)

Gates (in priority order, per the user-approved plan):
  1. PRIMARY:   cross-cell PF spread (h3e) <= 1.00  (vs baseline 1.30, h3a 1.76)
  2. SECONDARY: floor cell PF >= 1.40  (vs baseline 1.18 in chop x low)
  3. TERTIARY:  aggregate PF >= 1.831  (baseline 1.881 - 0.05 tolerance)
  4. DISCIPLINE: per-seed delta_pf >= -0.10 (no seed crashes)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/research_eval_h3e_oracle.py [--seeds 42]

Default --seeds runs all 5 (42-46) for the 5-seed gate. Pass --seeds 42 for
the cheap seed-42 first check before parallel rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle


BASE_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_balanced_fresh.npz"
H3A_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_h3a.npz"
H3E_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_h3e.npz"
CHOSEN_PATTERN = "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{seed}/seed_{seed}/chosen_trades.pkl"
SPX_1MIN = "/Users/gduby/.cache/autoresearch-trading/data/spx_1min.pkl"


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def hl(pnl_raw, em, ef, eb, so, exit_bar):
    return hybrid_live_utility(
        pnl_raw if np.isfinite(pnl_raw) else None,
        entry_mid=em, spread_fraction=ef if np.isfinite(ef) else 0.0,
        stopout_risk=so if np.isfinite(so) else 0.0,
        entry_bar=int(eb),
        exit_bar=int(exit_bar) if exit_bar >= 0 else None,
        session_end_bar=375,
    )


def build_daily_returns():
    with open(SPX_1MIN, "rb") as f:
        spx = pickle.load(f)
    daily = spx.groupby("date")["spx_close"].last().sort_index()
    daily.index = daily.index.astype(str)
    return daily.pct_change(20).to_dict()


def trend_regime(ret20d: float) -> str:
    if not np.isfinite(ret20d): return "unknown"
    if ret20d <= -0.02: return "bear"
    if ret20d >= 0.02: return "bull"
    return "chop"


def vol_regime(iv_pct: float) -> str:
    if not np.isfinite(iv_pct): return "unknown"
    if iv_pct < 0.33: return "low"
    if iv_pct < 0.66: return "mid"
    return "high"


def gather_records(seeds, oracle_pattern, bundle, key_to_row, al, daily_returns):
    """Load chosen trades + oracle predictions for the given seeds, return DF."""
    records = []
    for seed in seeds:
        opath = oracle_pattern.format(seed=seed)
        cpath = CHOSEN_PATTERN.format(seed=seed)
        if not os.path.exists(opath):
            print(f"  seed {seed}: oracle missing at {opath}")
            continue
        if not os.path.exists(cpath):
            print(f"  seed {seed}: chosen_trades missing at {cpath}")
            continue
        oracle = np.load(opath, allow_pickle=True)
        ex_pnl, ex_bar = oracle["l3_exit_pnl"], oracle["l3_exit_bar"]

        fw = pd.read_pickle(cpath)
        fw = fw[fw["chosen_action_id"] > 0].reset_index(drop=True)

        for _, row in fw.iterrows():
            key = (row["day"], row["bar_index"])
            r = key_to_row.get(key)
            if r is None: continue
            a = int(row["chosen_action_id"])
            if a <= 0: continue
            em = float(al["entry_fill_mid"][r, a])
            ef = float(al["entry_spread_fraction"][r, a])
            eb_ = float(al["entry_fill_bar"][r, a])
            so = float(al["stopout_risk"][r, a])
            if not (np.isfinite(em) and np.isfinite(eb_)): continue

            day = str(row["day"])
            ret20 = daily_returns.get(day, np.nan)
            iv_pct = float(row.get("iv_percentile", np.nan))

            pnl, bar = float(ex_pnl[r, a]), int(ex_bar[r, a])
            records.append({
                "seed": seed,
                "day": day,
                "side": row["chosen_side"],
                "trend": trend_regime(ret20),
                "vol": vol_regime(iv_pct),
                "best": float(row["best_forward_pnl_call"]) if row["chosen_side"] == "call"
                        else float(row["best_forward_pnl_put"]),
                "hl": hl(pnl, em, ef, eb_, so, bar),
                "exit_bar": bar,
                "exit_pnl_raw": pnl,
            })
    return pd.DataFrame(records)


def regime_cells(df):
    cells = []
    for t in ["bear", "chop", "bull"]:
        for v in ["low", "mid", "high"]:
            sub = df[(df["trend"] == t) & (df["vol"] == v)]
            if len(sub) >= 30:
                cells.append({"trend": t, "vol": v, "n": int(len(sub)),
                              "pf": float(pf(sub["hl"]))})
    return cells


def boot_delta_pf(base_hl, h3e_hl, n_iter=2000):
    rng = np.random.default_rng(42)
    n = len(base_hl)
    if n < 5: return float("nan"), float("nan"), float("nan")
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        deltas.append(pf(h3e_hl.iloc[idx]) - pf(base_hl.iloc[idx]))
    deltas = np.asarray(deltas)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float((deltas <= 0).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--out", default="v3/artifacts/research/h3e_eval.json")
    args = parser.parse_args()

    seeds = args.seeds
    print(f"=== H3e evaluation (spread-minimization gate) — seeds {seeds} ===\n")

    daily_returns = build_daily_returns()
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    df_base = gather_records(seeds, BASE_PATTERN, bundle, key_to_row, al, daily_returns)
    df_h3a = gather_records(seeds, H3A_PATTERN, bundle, key_to_row, al, daily_returns)
    df_h3e = gather_records(seeds, H3E_PATTERN, bundle, key_to_row, al, daily_returns)

    # Align indices on (seed, day, side, exit-key) — actually on row order; same trades
    if not (len(df_base) == len(df_h3a) == len(df_h3e)):
        print(f"WARNING: row counts differ. base={len(df_base)} h3a={len(df_h3a)} h3e={len(df_h3e)}")
    print(f"Trades: base={len(df_base)}  h3a={len(df_h3a)}  h3e={len(df_h3e)}")
    print()

    def report_variant(label, df):
        agg_pf = pf(df["hl"])
        cells = regime_cells(df)
        if cells:
            cell_pfs = [c["pf"] for c in cells]
            spread = max(cell_pfs) - min(cell_pfs)
            floor = min(cell_pfs)
        else:
            spread = float("nan"); floor = float("nan")
        return agg_pf, spread, floor, cells

    base_pf, base_spread, base_floor, base_cells = report_variant("baseline", df_base)
    h3a_pf, h3a_spread, h3a_floor, h3a_cells = report_variant("h3a", df_h3a)
    h3e_pf, h3e_spread, h3e_floor, h3e_cells = report_variant("h3e", df_h3e)

    print(f"{'variant':>10} {'agg_pf':>8} {'spread':>8} {'floor':>8}")
    print(f"{'baseline':>10} {base_pf:>8.3f} {base_spread:>8.3f} {base_floor:>8.3f}")
    print(f"{'h3a':>10} {h3a_pf:>8.3f} {h3a_spread:>8.3f} {h3a_floor:>8.3f}")
    print(f"{'h3e':>10} {h3e_pf:>8.3f} {h3e_spread:>8.3f} {h3e_floor:>8.3f}")
    print()

    print("=== Per-cell comparison (baseline vs h3e) ===")
    print(f"{'trend':>6} {'vol':>6} {'n':>5} {'base':>8} {'h3a':>8} {'h3e':>8}  {'d_h3e_vs_base':>15}")
    base_by = {(c["trend"], c["vol"]): c for c in base_cells}
    h3a_by = {(c["trend"], c["vol"]): c for c in h3a_cells}
    for c in h3e_cells:
        key = (c["trend"], c["vol"])
        bp = base_by.get(key, {}).get("pf", float("nan"))
        hp = h3a_by.get(key, {}).get("pf", float("nan"))
        delta = c["pf"] - bp if np.isfinite(bp) else float("nan")
        print(f"{c['trend']:>6} {c['vol']:>6} {c['n']:>5} {bp:>8.3f} {hp:>8.3f} {c['pf']:>8.3f}  {delta:>+15.3f}")
    print()

    # Per-seed (discipline check)
    print("=== Per-seed delta_pf vs baseline ===")
    print(f"{'seed':>5} {'n':>5} {'base_pf':>9} {'h3e_pf':>9} {'d_pf':>8}")
    per_seed_deltas = []
    for s in seeds:
        sb = df_base[df_base["seed"] == s]
        sh = df_h3e[df_h3e["seed"] == s]
        if len(sh) == 0: continue
        bp = pf(sb["hl"]); hp = pf(sh["hl"])
        per_seed_deltas.append(hp - bp)
        print(f"{s:>5} {len(sh):>5} {bp:>9.3f} {hp:>9.3f} {hp-bp:>+8.3f}")
    print()

    # Bootstrap delta-PF on aggregate
    if len(df_base) == len(df_h3e):
        ci_lo, ci_hi, p_neg = boot_delta_pf(df_base["hl"], df_h3e["hl"])
        print(f"Bootstrap delta-PF (h3e - base) 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]  p[d<=0]={p_neg:.3f}")
        print()

    # Gates
    print("=== Decision gates ===")
    primary = h3e_spread <= 1.00
    secondary = h3e_floor >= 1.40
    tertiary = h3e_pf >= 1.831
    no_seed_crash = all(d >= -0.10 for d in per_seed_deltas) if per_seed_deltas else False

    print(f"  PRIMARY   spread <= 1.00:  {h3e_spread:.3f}  -> {'PASS' if primary else 'FAIL'}")
    print(f"  SECONDARY floor  >= 1.40:  {h3e_floor:.3f}  -> {'PASS' if secondary else 'FAIL'}")
    print(f"  TERTIARY  agg   >= 1.831:  {h3e_pf:.3f}     -> {'PASS' if tertiary else 'FAIL'}")
    print(f"  DISCIPLINE no seed crash:  {'PASS' if no_seed_crash else 'FAIL'}")
    print()

    if primary and secondary and tertiary and no_seed_crash:
        verdict = "PASS"
        print("✓ H3e PASSES all gates. Recommend forward-walk validation.")
    elif primary and secondary and not tertiary and h3e_pf >= 1.5:
        verdict = "PARTIAL"
        print("≈ H3e PARTIAL: spread + floor pass, aggregate degraded. Per user")
        print("  framing (regime adaptability over aggregate), still proceed to FW.")
    elif primary and not secondary:
        verdict = "WEAK"
        print("≈ H3e WEAK: spread narrowed but floor didn't lift to 1.40. Marginal.")
    else:
        verdict = "FAIL"
        print("✗ H3e FAILS primary gate. Recommend git revert; consider H3e-clean variant")
        print("  (drop H3a) as a secondary attribution test.")

    summary = {
        "seeds": list(seeds),
        "n_trades_per_variant": {"base": len(df_base), "h3a": len(df_h3a), "h3e": len(df_h3e)},
        "agg_pf": {"base": base_pf, "h3a": h3a_pf, "h3e": h3e_pf},
        "spread": {"base": base_spread, "h3a": h3a_spread, "h3e": h3e_spread},
        "floor": {"base": base_floor, "h3a": h3a_floor, "h3e": h3e_floor},
        "h3e_cells": h3e_cells,
        "per_seed_deltas_base_to_h3e": per_seed_deltas,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0 if verdict in ("PASS", "PARTIAL") else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
