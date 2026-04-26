"""Phase 1: L2 dollar-score gate threshold sweep.

L2's decision_margin only enforces composite_utility > flat. It does NOT
enforce pred_dollar_score >= 0. The L2 audit found 245 trades in chop×low
entered at mean pred_dollar_score = -0.14 (model expected to lose money).

This script tests the hypothesis: a runtime gate on pred_dollar_score
narrows the regime spread and lifts the floor without retraining.

Method (pure analysis on existing data):
  - Load 5-seed chosen_trades + baseline oracle predictions
  - For each candidate threshold, filter trades where pred_dollar_score < t
  - Recompute aggregate PF, per-cell PF (sigma_pos × iv_percentile),
    regime spread, per-seed deltas
  - Report: threshold-vs-PF curve, n_trades dropped, per-cell impact

Decision gate:
  - If a threshold lifts agg PF AND narrows spread AND keeps >=70% trades:
    proceed to Phase 2 (integrate into _select_daily_trades)
  - If improves PF but loses >30% trades: investigate retraining
  - If no threshold helps: gate hypothesis falsified
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle


SEEDS = [42, 43, 44, 45, 46]
BASE_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_balanced_fresh.npz"
CHOSEN_PATTERN = "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{seed}/seed_{seed}/chosen_trades.pkl"

THRESHOLDS = [-0.30, -0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20, +0.30]


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


def gather() -> pd.DataFrame:
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    records = []
    for seed in SEEDS:
        cp = CHOSEN_PATTERN.format(seed=seed)
        bp = BASE_PATTERN.format(seed=seed)
        if not (os.path.exists(cp) and os.path.exists(bp)):
            continue
        fw = pd.read_pickle(cp)
        fw = fw[fw["chosen_action_id"] > 0].reset_index(drop=True)
        oracle = np.load(bp, allow_pickle=True)
        ex_pnl, ex_bar = oracle["l3_exit_pnl"], oracle["l3_exit_bar"]

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
            if int(row["bar_index"]) >= 120: continue  # drop boundary

            pnl, bar = float(ex_pnl[r, a]), int(ex_bar[r, a])
            hl_val = hl(pnl, em, ef, eb_, so, bar)
            if not np.isfinite(hl_val): continue

            records.append({
                "seed": seed,
                "day": str(row["day"]),
                "side": row["chosen_side"],
                "bar_index": int(row["bar_index"]),
                "pred_dollar_score": float(row.get("pred_dollar_score", np.nan)),
                "pred_return_score": float(row.get("pred_return_score", np.nan)),
                "pred_win_prob": float(row.get("pred_win_prob", np.nan)),
                "pred_clean_entry_prob": float(row.get("pred_clean_entry_prob", np.nan)),
                "pred_stopout_risk": float(row.get("pred_stopout_risk", np.nan)),
                "decision_margin": float(row.get("decision_margin", np.nan)),
                "sigma_pos": float(row.get("sigma_pos", np.nan)),
                "iv_percentile": float(row.get("iv_percentile", np.nan)),
                "hl": hl_val,
            })
    df = pd.DataFrame(records).dropna(subset=["pred_dollar_score", "sigma_pos", "iv_percentile"])
    df["sigma_b"] = pd.qcut(df["sigma_pos"], 3, labels=False, duplicates="drop")
    df["iv_b"] = pd.qcut(df["iv_percentile"], 3, labels=False, duplicates="drop")
    df["cell_id"] = df["sigma_b"].astype(int) * 3 + df["iv_b"].astype(int)
    return df


def cell_label(cell_id: int) -> str:
    return f"s{cell_id // 3}_iv{cell_id % 3}"


def evaluate(df: pd.DataFrame, threshold: float, baseline_n: int) -> dict:
    kept = df[df["pred_dollar_score"] >= threshold].copy()
    n = len(kept)
    if n < 50:
        return None
    agg = pf(kept["hl"])
    cells = []
    for cid, group in kept.groupby("cell_id"):
        if len(group) < 30:
            continue
        cells.append({
            "cell": cell_label(cid),
            "n": int(len(group)),
            "pf": float(pf(group["hl"])),
        })
    if not cells:
        return None
    cell_pfs = [c["pf"] for c in cells]
    spread = max(cell_pfs) - min(cell_pfs)
    floor = min(cell_pfs)

    # Per-seed
    per_seed = {}
    for s in SEEDS:
        sub = kept[kept["seed"] == s]
        if len(sub) >= 30:
            per_seed[s] = {"n": int(len(sub)), "pf": float(pf(sub["hl"]))}

    return {
        "threshold": float(threshold),
        "n_trades": n,
        "n_dropped": baseline_n - n,
        "frac_kept": n / baseline_n,
        "agg_pf": float(agg),
        "spread": float(spread),
        "floor_pf": float(floor),
        "n_cells_eval": len(cells),
        "cells": cells,
        "per_seed": per_seed,
    }


def main():
    print("=== Phase 1: L2 dollar-score gate threshold sweep ===\n")
    df = gather()
    baseline_n = len(df)
    print(f"Baseline trades (5 seeds, dropped bar>=120 + NaN): {baseline_n}")
    print(f"pred_dollar_score: mean={df['pred_dollar_score'].mean():+.3f}, "
          f"std={df['pred_dollar_score'].std():.3f}, "
          f"range [{df['pred_dollar_score'].min():+.3f}, {df['pred_dollar_score'].max():+.3f}]")
    n_neg = (df["pred_dollar_score"] < 0).sum()
    n_neg_05 = (df["pred_dollar_score"] < -0.05).sum()
    print(f"Trades with pred_dollar_score < 0:    {n_neg} ({n_neg/baseline_n:.1%})")
    print(f"Trades with pred_dollar_score < -0.05: {n_neg_05} ({n_neg_05/baseline_n:.1%})")
    print()
    print(f"Baseline aggregate PF (no gate): {pf(df['hl']):.3f}")
    print()

    # ----- Threshold sweep -----
    print("=" * 90)
    print("Threshold sweep — agg PF, spread, floor, coverage\n")
    print(f"{'thr':>6} {'n_kept':>7} {'frac_kept':>10} {'agg_pf':>8} {'spread':>8} {'floor_pf':>9} {'min_seed_pf':>12}")

    results = []
    for t in THRESHOLDS:
        r = evaluate(df, t, baseline_n)
        if r is None:
            continue
        seed_pfs = [s["pf"] for s in r["per_seed"].values()]
        min_seed_pf = min(seed_pfs) if seed_pfs else float("nan")
        print(f"{t:>+6.2f} {r['n_trades']:>7} {r['frac_kept']:>10.1%} "
              f"{r['agg_pf']:>8.3f} {r['spread']:>8.3f} {r['floor_pf']:>9.3f} "
              f"{min_seed_pf:>12.3f}")
        results.append(r)
    print()

    # ----- Per-cell impact at the optimal threshold -----
    # Optimal = best (spread, agg_pf) tradeoff
    baseline_eval = evaluate(df, -1e9, baseline_n)
    print("=" * 90)
    print(f"Baseline (no gate, threshold = -inf):  agg_pf {baseline_eval['agg_pf']:.3f}  "
          f"spread {baseline_eval['spread']:.3f}  floor {baseline_eval['floor_pf']:.3f}\n")

    # Find best by aggregate criterion: (spread - 0.5 * agg_pf) — smaller is better
    # (we want narrow spread without sacrificing aggregate)
    rated = [(r, r["spread"] - 0.5 * r["agg_pf"]) for r in results]
    rated.sort(key=lambda x: x[1])
    best = rated[0][0]
    print(f"Best by (spread - 0.5*agg_pf): threshold = {best['threshold']:+.2f}")
    print(f"  agg_pf {best['agg_pf']:.3f} (vs baseline {baseline_eval['agg_pf']:.3f}, delta {best['agg_pf']-baseline_eval['agg_pf']:+.3f})")
    print(f"  spread {best['spread']:.3f} (vs baseline {baseline_eval['spread']:.3f}, delta {best['spread']-baseline_eval['spread']:+.3f})")
    print(f"  floor  {best['floor_pf']:.3f} (vs baseline {baseline_eval['floor_pf']:.3f}, delta {best['floor_pf']-baseline_eval['floor_pf']:+.3f})")
    print(f"  trades kept: {best['frac_kept']:.1%}\n")

    # Also report by simpler "max agg_pf with spread <= baseline" criterion
    valid = [r for r in results if r["spread"] <= baseline_eval["spread"]]
    if valid:
        by_agg = max(valid, key=lambda r: r["agg_pf"])
        print(f"Best agg_pf with spread <= baseline:  threshold = {by_agg['threshold']:+.2f}")
        print(f"  agg_pf {by_agg['agg_pf']:.3f} (delta {by_agg['agg_pf']-baseline_eval['agg_pf']:+.3f})")
        print(f"  spread {by_agg['spread']:.3f} (delta {by_agg['spread']-baseline_eval['spread']:+.3f})")
        print(f"  floor  {by_agg['floor_pf']:.3f} (delta {by_agg['floor_pf']-baseline_eval['floor_pf']:+.3f})")
        print(f"  trades kept: {by_agg['frac_kept']:.1%}\n")
    else:
        by_agg = None

    # ----- Per-cell deltas at the chosen threshold -----
    chosen = best
    print("=" * 90)
    print(f"Per-cell impact at threshold = {chosen['threshold']:+.2f}\n")
    base_cells = {c["cell"]: c for c in baseline_eval["cells"]}
    print(f"  {'cell':>10} {'baseline_n':>11} {'baseline_pf':>12} {'gated_n':>8} {'gated_pf':>9} {'delta_pf':>10} {'frac_kept':>10}")
    for c in sorted(chosen["cells"], key=lambda x: x["cell"]):
        base_c = base_cells.get(c["cell"], {"n": 0, "pf": float("nan")})
        delta = c["pf"] - base_c["pf"]
        frac = c["n"] / max(base_c["n"], 1)
        print(f"  {c['cell']:>10} {base_c['n']:>11} {base_c['pf']:>12.3f} "
              f"{c['n']:>8} {c['pf']:>9.3f} {delta:>+10.3f} {frac:>10.1%}")
    print()

    # ----- Decision -----
    print("=" * 90)
    print("Decision gate (per plan)\n")
    delta_agg = chosen["agg_pf"] - baseline_eval["agg_pf"]
    delta_spread = chosen["spread"] - baseline_eval["spread"]
    delta_floor = chosen["floor_pf"] - baseline_eval["floor_pf"]
    frac_kept = chosen["frac_kept"]

    if delta_agg > 0 and delta_spread < 0 and frac_kept >= 0.70:
        verdict = "PROCEED to Phase 2"
        msg = "Gate lifts agg AND narrows spread AND keeps >=70% trades."
    elif delta_agg > 0 and frac_kept < 0.70:
        verdict = "INVESTIGATE retraining (Phase 4)"
        msg = "Gate lifts PF but drops too many trades. Calibration miss is broader than chop x low."
    elif delta_agg <= 0 and delta_spread >= 0:
        verdict = "FALSIFIED"
        msg = "No threshold helps. Gate hypothesis is wrong."
    else:
        verdict = "MIXED"
        msg = "Tradeoff is ambiguous. Manual review of threshold-vs-PF curve recommended."

    print(f"  Best threshold: {chosen['threshold']:+.2f}")
    print(f"  delta agg_pf: {delta_agg:+.3f}")
    print(f"  delta spread: {delta_spread:+.3f}")
    print(f"  delta floor:  {delta_floor:+.3f}")
    print(f"  frac kept:    {frac_kept:.1%}")
    print(f"  Verdict: {verdict}")
    print(f"  {msg}")
    print()

    summary = {
        "baseline": {
            "n_trades": baseline_n,
            "agg_pf": baseline_eval["agg_pf"],
            "spread": baseline_eval["spread"],
            "floor_pf": baseline_eval["floor_pf"],
            "cells": baseline_eval["cells"],
        },
        "thresholds": results,
        "best_threshold": chosen,
        "verdict": verdict,
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/l2_dollar_gate_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv("v3/artifacts/research/l2_dollar_gate_sweep_picks.csv", index=False)
    print(f"Wrote v3/artifacts/research/l2_dollar_gate_sweep.json + picks.csv")


if __name__ == "__main__":
    main()
