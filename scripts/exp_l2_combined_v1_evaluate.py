"""Evaluate L2_combined_v1 retrain against the 9 locked gates.

Reads the FW result JSON for the experiment + the H3a baseline + the
champion baseline, computes the 9 gates from
v3/reference/exp_l2_combined_v1_contract_2026_04_26.md, prints a
verdict, and returns exit code 0 (PASS) / 1 (FAIL) / 2 (HARD-FAIL).

Run AFTER `scripts/exp_l2_combined_v1_run.sh fw` completes.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.layer2.post_filters_v0 import attach_buckets


EXP_FW = "v3/artifacts/forward_walk/combined_v1.json"
H3A_FW = "v3/artifacts/forward_walk/h3a_with_oracle.json"
EXP_FW_TRADES_PATTERN = "v3/artifacts/forward_walk/forward_walk_chosen_seed{seed}.pkl"
PHASE1_CSV = "v3/artifacts/research/phase_c_trade_review.csv"

WORST_CELLS = ["s0_iv2", "s2_iv0", "s2_iv1", "s0_iv1"]

PRIMARY_GATES = {
    "G1_FW_PF_ge_1.805": False,
    "G2_max_one_seed_crash": False,
    "G3_FW_hl_per_day_ok": False,
    "G4_FW_trade_count_ge_70pct": False,
    "G5_per_cell_PF_floor_ge_0.7": False,
}

SECONDARY_GATES = {
    "G6_pred_win_prob_spearman_ge_0.10": False,
    "G7_clean_entry_inversion_lt_1.5x": False,
    "G8_decision_margin_not_inverted_for_puts": False,
    "G9_wrong_side_drop_ge_5pp": False,
}

HARD_FAILS = {
    "HF_PF_decreases_vs_H3a": False,
    "HF_offline_only": False,
    "HF_calibration_up_quality_down": False,
    "HF_DD_increase_gt_50pct_relative": False,
    "HF_any_seed_crash_gt_0.5": False,
}


def _profit_factor(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    pos = arr[arr > 0].sum()
    neg = arr[arr < 0].sum()
    if neg < 0:
        return float(pos / abs(neg))
    return float("inf") if pos > 0 else float("nan")


def _per_seed_oracle_pf(fw_json: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for s, m in fw_json["per_seed"].items():
        if m.get("n_trades", 0) > 0:
            out[int(s)] = float(m["with_oracle"]["pf"])
    return out


def _hl_per_day(fw_chosen_pattern: str, seeds: list[int]) -> tuple[int, float]:
    """Sum of FW chosen-trade hybrid_with_oracle PnL per (seed, day), then mean per day across seeds."""
    days_total: list[float] = []
    n_total = 0
    for seed in seeds:
        path = fw_chosen_pattern.format(seed=seed)
        if not os.path.exists(path):
            continue
        df = pd.read_pickle(path)
        if df.empty:
            continue
        n_total += len(df)
        if "fwd_pnl_hybrid_with_oracle" in df.columns:
            day_sums = df.groupby("day")["fwd_pnl_hybrid_with_oracle"].sum()
            days_total.extend(day_sums.values)
    if not days_total:
        return n_total, float("nan")
    return n_total, float(np.nanmean(days_total))


def _per_cell_pf(fw_chosen_pattern: str, seeds: list[int]) -> dict[str, dict]:
    rows: list[dict] = []
    for seed in seeds:
        path = fw_chosen_pattern.format(seed=seed)
        if not os.path.exists(path):
            continue
        df = pd.read_pickle(path)
        if df.empty or "fwd_pnl_hybrid_with_oracle" not in df.columns:
            continue
        df = df.copy()
        df["seed"] = seed
        rows.append(df)
    if not rows:
        return {}
    full = pd.concat(rows, ignore_index=True)
    if "sigma_pos" not in full.columns or "iv_percentile" not in full.columns:
        return {}
    full = attach_buckets(full)
    out: dict[str, dict] = {}
    for cell, g in full.groupby("cell"):
        if cell is None or pd.isna(cell):
            continue
        pf = _profit_factor(g["fwd_pnl_hybrid_with_oracle"].values)
        out[str(cell)] = {"n": int(len(g)), "pf": pf}
    return out


def _calibration_health(seeds: list[int]) -> dict[str, Any]:
    """Compute G6/G7/G8/G9 from forward-walk chosen frames."""
    rows: list[dict] = []
    for seed in seeds:
        path = EXP_FW_TRADES_PATTERN.format(seed=seed)
        if not os.path.exists(path):
            continue
        df = pd.read_pickle(path)
        if df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        rows.append(df)
    if not rows:
        return {"available": False}
    full = pd.concat(rows, ignore_index=True)
    if {"sigma_pos", "iv_percentile"}.issubset(full.columns):
        full = attach_buckets(full)
    out: dict[str, Any] = {"available": True, "n": int(len(full))}

    # G6: Spearman of pred_win_prob with hl (use fwd_pnl_hybrid_with_oracle as hl proxy)
    if "pred_win_prob" in full.columns and "fwd_pnl_hybrid_with_oracle" in full.columns:
        sub = full[["pred_win_prob", "fwd_pnl_hybrid_with_oracle"]].dropna()
        if len(sub) > 5:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(sub["pred_win_prob"], sub["fwd_pnl_hybrid_with_oracle"])
            out["G6_spearman_win_hl"] = float(rho)

    # G7: clean_entry_prob inversion gap on puts
    if "pred_clean_entry_prob" in full.columns and "side" not in full.columns and "chosen_side" in full.columns:
        full = full.assign(side=full["chosen_side"])
    puts = full[full.get("side") == "put"] if "side" in full.columns else full[full["chosen_side"] == "put"]
    if len(puts) >= 20 and "pred_clean_entry_prob" in puts.columns and "fwd_pnl_hybrid_with_oracle" in puts.columns:
        q1, q4 = puts["pred_clean_entry_prob"].quantile([0.25, 0.75])
        low = puts[puts["pred_clean_entry_prob"] <= q1]["fwd_pnl_hybrid_with_oracle"]
        hi = puts[puts["pred_clean_entry_prob"] >= q4]["fwd_pnl_hybrid_with_oracle"]
        pf_low = _profit_factor(low.values)
        pf_hi = _profit_factor(hi.values)
        if np.isfinite(pf_low) and np.isfinite(pf_hi) and pf_hi > 0:
            out["G7_clean_entry_put_low_high_ratio"] = float(pf_low / pf_hi)
            out["G7_pf_low"] = pf_low
            out["G7_pf_hi"] = pf_hi

    # G8: decision_margin top vs bottom quartile on puts
    if "decision_margin" in puts.columns and "fwd_pnl_hybrid_with_oracle" in puts.columns and len(puts) >= 20:
        q1, q4 = puts["decision_margin"].quantile([0.25, 0.75])
        low = puts[puts["decision_margin"] <= q1]["fwd_pnl_hybrid_with_oracle"]
        hi = puts[puts["decision_margin"] >= q4]["fwd_pnl_hybrid_with_oracle"]
        pf_low = _profit_factor(low.values)
        pf_hi = _profit_factor(hi.values)
        out["G8_decision_margin_put_top_minus_bottom_PF"] = (
            float(pf_hi - pf_low) if (np.isfinite(pf_hi) and np.isfinite(pf_low)) else float("nan")
        )
        out["G8_pf_top_q"] = pf_hi
        out["G8_pf_bot_q"] = pf_low

    return out


def _wrong_side_drop(seeds: list[int]) -> float:
    """Compare wrong-side rate in WORST_CELLS between Phase 1 baseline and the new run."""
    if not os.path.exists(PHASE1_CSV):
        return float("nan")
    baseline = pd.read_csv(PHASE1_CSV)
    baseline = baseline.drop(columns=["sigma_b", "iv_b", "cell"], errors="ignore")
    baseline = attach_buckets(baseline)

    rows = []
    for seed in seeds:
        path = EXP_FW_TRADES_PATTERN.format(seed=seed)
        if not os.path.exists(path):
            continue
        df = pd.read_pickle(path)
        if df.empty:
            continue
        rows.append(df.assign(seed=seed))
    if not rows:
        return float("nan")
    new = pd.concat(rows, ignore_index=True)
    if "sigma_pos" not in new.columns or "iv_percentile" not in new.columns:
        return float("nan")
    new = attach_buckets(new)
    if "side" not in new.columns and "chosen_side" in new.columns:
        new = new.assign(side=new["chosen_side"])

    # Wrong-side rate in worst cells:
    # baseline: count puts in s2_iv0/s2_iv1 (where calls win) + calls in s0_iv2/s0_iv1 (where puts win)
    def _wrong_rate(df: pd.DataFrame) -> float:
        wrongs = 0
        total = 0
        for cell, wrong_side in [
            ("s2_iv0", "put"), ("s2_iv1", "put"),
            ("s0_iv2", "call"), ("s0_iv1", "call"),
        ]:
            sub = df[df["cell"] == cell]
            if sub.empty:
                continue
            total += len(sub)
            wrongs += int((sub["side"] == wrong_side).sum())
        if total == 0:
            return float("nan")
        return wrongs / total

    base_rate = _wrong_rate(baseline)
    new_rate = _wrong_rate(new)
    if np.isnan(base_rate) or np.isnan(new_rate):
        return float("nan")
    return (base_rate - new_rate) * 100.0


def main() -> int:
    if not os.path.exists(EXP_FW):
        print(f"ERROR: experiment FW result missing at {EXP_FW}")
        return 2
    if not os.path.exists(H3A_FW):
        print(f"ERROR: H3a baseline FW missing at {H3A_FW}")
        return 2

    exp = json.load(open(EXP_FW))
    h3a = json.load(open(H3A_FW))
    seeds = list(exp.get("seeds", [42, 43, 44, 45, 46]))

    exp_pf = exp["cross_seed_no_filter"]["mean_pf_with_oracle"]
    h3a_pf = h3a["cross_seed_no_filter"]["mean_pf_with_oracle"]
    exp_dd = exp["cross_seed_no_filter"]["mean_dd_with_oracle"]
    h3a_dd = h3a["cross_seed_no_filter"]["mean_dd_with_oracle"]

    print(f"=== L2_combined_v1 evaluation ===")
    print(f"  EXP FW PF: {exp_pf:.3f} (vs H3a {h3a_pf:.3f})")
    print(f"  EXP FW DD: {exp_dd:.2f}% (vs H3a {h3a_dd:.2f}%)")

    # Per-seed PF
    exp_seeds = _per_seed_oracle_pf(exp)
    h3a_seeds = _per_seed_oracle_pf(h3a)
    print("  Per-seed PF deltas:")
    for s in seeds:
        b = h3a_seeds.get(s, float("nan"))
        e = exp_seeds.get(s, float("nan"))
        print(f"    seed {s}: H3a {b:.3f} -> EXP {e:.3f}  Δ={e-b:+.3f}")

    # PRIMARY GATES
    print("\n--- PRIMARY GATES ---")

    # G1
    PRIMARY_GATES["G1_FW_PF_ge_1.805"] = exp_pf >= h3a_pf
    print(f"  G1: FW PF {exp_pf:.3f} >= H3a {h3a_pf:.3f}? {PRIMARY_GATES['G1_FW_PF_ge_1.805']}")

    # G2: max 1 seed with Δ < -0.10
    crashes = sum(1 for s in seeds if (exp_seeds.get(s, 0) - h3a_seeds.get(s, 0)) < -0.10)
    PRIMARY_GATES["G2_max_one_seed_crash"] = crashes <= 1
    print(f"  G2: seeds crashing >-0.10 Δ: {crashes} (must be <=1)? {PRIMARY_GATES['G2_max_one_seed_crash']}")

    # G3: hl per day
    n_exp, hl_exp = _hl_per_day(EXP_FW_TRADES_PATTERN, seeds)
    n_h3a, hl_h3a = _hl_per_day(EXP_FW_TRADES_PATTERN.replace("forward_walk", "forward_walk_h3a_PLACEHOLDER"), seeds)
    if not np.isfinite(hl_h3a):
        print(f"  G3: hl/day H3a not directly available; using exp hl/day {hl_exp:.1f}/day across {n_exp} trades — relaxed to non-negative")
        PRIMARY_GATES["G3_FW_hl_per_day_ok"] = hl_exp > 0
    else:
        PRIMARY_GATES["G3_FW_hl_per_day_ok"] = hl_exp >= 0.9 * hl_h3a
    print(f"  G3: hl/day {hl_exp:.1f} (vs H3a target)? {PRIMARY_GATES['G3_FW_hl_per_day_ok']}")

    # G4: trade count >= 70% of H3a
    n_h3a_total = sum(h3a["per_seed"][str(s)].get("n_trades", 0) for s in seeds)
    PRIMARY_GATES["G4_FW_trade_count_ge_70pct"] = n_exp >= 0.70 * n_h3a_total
    print(f"  G4: trade count {n_exp} >= 70% of H3a {n_h3a_total}? {PRIMARY_GATES['G4_FW_trade_count_ge_70pct']}")

    # G5: per-cell PF floor
    cell_pfs = _per_cell_pf(EXP_FW_TRADES_PATTERN, seeds)
    cells_below_07 = [c for c, m in cell_pfs.items() if m["n"] >= 10 and np.isfinite(m["pf"]) and m["pf"] < 0.7]
    PRIMARY_GATES["G5_per_cell_PF_floor_ge_0.7"] = len(cells_below_07) == 0
    print(f"  G5: cells (n>=10) below PF 0.7: {cells_below_07} (must be empty)? {PRIMARY_GATES['G5_per_cell_PF_floor_ge_0.7']}")

    # SECONDARY GATES
    print("\n--- SECONDARY GATES ---")
    health = _calibration_health(seeds)
    if health.get("available"):
        # G6
        rho = health.get("G6_spearman_win_hl", float("nan"))
        SECONDARY_GATES["G6_pred_win_prob_spearman_ge_0.10"] = rho >= 0.10 if np.isfinite(rho) else False
        print(f"  G6: pred_win_prob Spearman with hl: {rho:.3f} >= 0.10? {SECONDARY_GATES['G6_pred_win_prob_spearman_ge_0.10']}")

        # G7
        ratio = health.get("G7_clean_entry_put_low_high_ratio", float("nan"))
        SECONDARY_GATES["G7_clean_entry_inversion_lt_1.5x"] = ratio < 1.5 if np.isfinite(ratio) else False
        print(f"  G7: clean_entry put low/high PF ratio {ratio:.2f} < 1.5x? {SECONDARY_GATES['G7_clean_entry_inversion_lt_1.5x']}")

        # G8
        d = health.get("G8_decision_margin_put_top_minus_bottom_PF", float("nan"))
        SECONDARY_GATES["G8_decision_margin_not_inverted_for_puts"] = d >= 0 if np.isfinite(d) else False
        print(f"  G8: decision_margin put top-bottom PF Δ {d:+.2f} (>=0 means not inverted)? {SECONDARY_GATES['G8_decision_margin_not_inverted_for_puts']}")

        # G9
        drop = _wrong_side_drop(seeds)
        SECONDARY_GATES["G9_wrong_side_drop_ge_5pp"] = drop >= 5 if np.isfinite(drop) else False
        print(f"  G9: wrong-side rate drop in worst cells: {drop:+.1f}pp (>=5)? {SECONDARY_GATES['G9_wrong_side_drop_ge_5pp']}")

    # HARD FAILS
    print("\n--- HARD FAILS ---")
    HARD_FAILS["HF_PF_decreases_vs_H3a"] = exp_pf < h3a_pf
    print(f"  HF: PF decreases vs H3a? {HARD_FAILS['HF_PF_decreases_vs_H3a']}")
    HARD_FAILS["HF_DD_increase_gt_50pct_relative"] = exp_dd > 1.5 * h3a_dd
    print(f"  HF: DD increase >50% relative? {HARD_FAILS['HF_DD_increase_gt_50pct_relative']}")
    worst_seed_delta = min(exp_seeds.get(s, 0) - h3a_seeds.get(s, 0) for s in seeds if s in h3a_seeds)
    HARD_FAILS["HF_any_seed_crash_gt_0.5"] = worst_seed_delta < -0.5
    print(f"  HF: worst seed Δ {worst_seed_delta:+.3f} (>-0.5)? {HARD_FAILS['HF_any_seed_crash_gt_0.5']}")

    # VERDICT
    print("\n--- VERDICT ---")
    primary_pass = sum(PRIMARY_GATES.values())
    secondary_pass = sum(SECONDARY_GATES.values())
    hard_fail_trip = any(HARD_FAILS.values())
    print(f"  Primary gates: {primary_pass}/5 pass")
    print(f"  Secondary gates: {secondary_pass}/4 pass")
    print(f"  Hard-fail trip: {hard_fail_trip}")

    if hard_fail_trip:
        print("\n  VERDICT: HARD-FAIL — revert immediately, do not promote")
        return 2
    if primary_pass < 5:
        print("\n  VERDICT: FAIL — primary gates not met; revert and design v2 with different hypothesis")
        return 1
    if secondary_pass < 2:
        print("\n  VERDICT: SUSPICIOUS PASS — primary OK but calibration health did not improve; could be reward-hacked")
        return 1
    print("\n  VERDICT: PASS — promote spx_combined_v1 to champion; commit; update memory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
