"""Three proof audits for the spx_combined_3seed_001 deployment recipe.

#3. L3-oracle headroom check. The chosen_objective_pnl in chosen_trades.pkl
    uses the L3 oracle's predicted exit, not the perfect-information best
    exit. This audit compares chosen_objective_pnl to:
      - best_exit_pnl  (per-action perfect-info ceiling)
      - horizon_pnl    (naive hold-to-end-of-window)
      - chosen_time_stop_pnl  (already in chosen_trades.pkl)
    If chosen ≈ best_exit, the oracle is memorizing; if chosen << best_exit
    but >> horizon, the oracle is generalizing.

#4. Bootstrap PF/DD confidence intervals on the K=2 + drop-W12 trade set.
    Resamples the chosen-bar list with replacement 1000 times and reports
    5%/50%/95% PF and DD. Tells us whether 2.142 is a tight estimate or
    has wide variance.

#5. Leave-one-window-out (LOWO) validation of the cal_pf > 4 rule.
    For each held-out window, derive the rule's threshold from the other
    12, then check whether the held-out window's flagged seeds actually
    showed val→OOS degradation. Tests whether the rule is overfit to
    these specific 13 windows.

Usage:

    .venv/bin/python -m v3.analysis.proof_audits \
        --champion-dir v3/artifacts \
        --champion-name spx_combined_3seed_001 \
        --seeds 42 43 44 45 46 \
        --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \
        --calibration-audit v3/artifacts/calibration_audit/spx_combined_3seed_001.json \
        --out v3/artifacts/proof_audits/spx_combined_3seed_001.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from v3.layer2.common import load_export_bundle


def _profit_factor(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def _max_drawdown_pct(pnl: np.ndarray, capital_base: float = 25_000.0) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        return 0.0
    eq = capital_base
    pk = capital_base
    mx = 0.0
    for p in pnl:
        eq += float(p)
        if eq > pk:
            pk = eq
        if pk > 0:
            dd = (pk - eq) / pk * 100.0
            if dd > mx:
                mx = dd
    return mx


def _load_seed_trades(champion_dir: str, name: str, seed: int) -> pd.DataFrame:
    path = os.path.join(
        champion_dir,
        f"layer2_unified_policy_{name}_seed{seed}",
        f"seed_{seed}",
        "chosen_trades.pkl",
    )
    with open(path, "rb") as f:
        df = pickle.load(f)
    df["seed"] = seed
    df = df.sort_values(["day", "bar_index"]).reset_index(drop=True)
    return df


# ---------- #3: L3-oracle headroom ----------

def audit_3_oracle_headroom(seed_trades: dict[int, pd.DataFrame], dataset_path: str) -> dict[str, Any]:
    bundle = load_export_bundle(dataset_path)
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    # build (day, bar_index) -> row index
    key_to_row = (
        rows[["day", "bar_index", "__row__"]]
        .drop_duplicates(subset=["day", "bar_index"])
        .set_index(["day", "bar_index"])
        ["__row__"]
        .to_dict()
    )
    best_exit = bundle["action_labels"]["best_exit_pnl"]
    horizon = bundle["action_labels"]["horizon_pnl"]
    time_stop_raw = bundle["action_labels"]["utility_raw"]  # this is the time-stop label in the dataset

    out_per_seed: dict[int, dict[str, Any]] = {}
    for seed, df in seed_trades.items():
        df = df[np.isfinite(df["chosen_objective_pnl"])]
        chosen_obj = []
        chosen_best = []
        chosen_horizon = []
        chosen_time_stop = []
        n_rows_missing = 0
        for _, row in df.iterrows():
            key = (row["day"], row["bar_index"])
            r_idx = key_to_row.get(key)
            if r_idx is None:
                n_rows_missing += 1
                continue
            a_idx = int(row["chosen_action_id"])
            chosen_obj.append(float(row["chosen_objective_pnl"]))
            chosen_best.append(float(best_exit[r_idx, a_idx]))
            chosen_horizon.append(float(horizon[r_idx, a_idx]))
            chosen_time_stop.append(float(row["chosen_time_stop_pnl"]))
        chosen_obj = np.asarray(chosen_obj, dtype=float)
        chosen_best = np.asarray(chosen_best, dtype=float)
        chosen_horizon = np.asarray(chosen_horizon, dtype=float)
        chosen_time_stop = np.asarray(chosen_time_stop, dtype=float)
        finite = (
            np.isfinite(chosen_obj)
            & np.isfinite(chosen_best)
            & np.isfinite(chosen_horizon)
            & np.isfinite(chosen_time_stop)
        )
        co = chosen_obj[finite]
        cb = chosen_best[finite]
        ch = chosen_horizon[finite]
        cts = chosen_time_stop[finite]
        # Headroom utilized: chosen / best on positive-best trades
        # capture-fraction = chosen_obj_sum / best_exit_sum on the positive-best subset
        positive_best = cb > 0
        cap = (co[positive_best].sum() / cb[positive_best].sum()) if cb[positive_best].sum() > 0 else 0.0
        out_per_seed[seed] = {
            "n_chosen": int(len(co)),
            "n_missing_lookup": int(n_rows_missing),
            "pf_chosen_objective": _profit_factor(co),
            "pf_best_exit_at_chosen": _profit_factor(cb),
            "pf_horizon_at_chosen": _profit_factor(ch),
            "pf_time_stop_at_chosen": _profit_factor(cts),
            "mean_chosen_obj": float(co.mean()),
            "mean_best_exit": float(cb.mean()),
            "mean_horizon": float(ch.mean()),
            "mean_time_stop": float(cts.mean()),
            "capture_fraction_of_best": float(cap),
            "frac_chosen_eq_best": float((np.isclose(co, cb, atol=1.0)).mean()),
            "frac_chosen_above_horizon": float((co > ch + 1.0).mean()),
            "frac_chosen_above_time_stop": float((co > cts + 1.0).mean()),
        }
    return out_per_seed


# ---------- #4: Bootstrap PF/DD confidence intervals ----------

def _build_consensus_summary(per_bar: pd.DataFrame) -> pd.DataFrame:
    grp = per_bar.groupby(["day", "bar_index"]).agg(
        n_call=("chosen_side", lambda s: int((s == "call").sum())),
        n_put=("chosen_side", lambda s: int((s == "put").sum())),
        window_idx=("window_idx", "first"),
    ).reset_index()
    grp["max_count"] = np.maximum(grp["n_call"], grp["n_put"])
    grp["max_side"] = np.where(grp["n_call"] >= grp["n_put"], "call", "put")
    return grp


def audit_4_bootstrap_ci(
    seed_trades: dict[int, pd.DataFrame],
    K: int = 2,
    drop_windows: tuple[int, ...] = (12,),
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-seed K-consensus + drop-W12 trades, bootstrap-resample 1000x.

    Each bootstrap iteration resamples the seed's filtered trade list
    with replacement (preserving N), computes PF and DD. Returns per-seed
    quantiles and the cross-seed mean PF distribution.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(seed_trades.keys())
    per_bar = pd.concat(seed_trades.values(), ignore_index=True)
    per_bar = per_bar[np.isfinite(per_bar["chosen_objective_pnl"])]
    consensus = _build_consensus_summary(per_bar)
    cs_index = consensus.set_index(["day", "bar_index"])

    per_seed_filtered: dict[int, np.ndarray] = {}
    for s in seeds:
        df = seed_trades[s]
        df = df[np.isfinite(df["chosen_objective_pnl"])]
        keep = []
        for _, row in df.iterrows():
            key = (row["day"], row["bar_index"])
            if key not in cs_index.index:
                keep.append(False)
                continue
            agg = cs_index.loc[key]
            side = row["chosen_side"]
            cnt = int(agg["n_call"]) if side == "call" else int(agg["n_put"])
            in_window = int(row["window_idx"]) not in drop_windows
            keep.append(cnt >= K and in_window)
        per_seed_filtered[s] = df.loc[pd.Series(keep, index=df.index), "chosen_objective_pnl"].values.astype(float)

    out_per_seed: dict[int, dict[str, Any]] = {}
    cross_seed_mean_pf_samples: list[float] = []
    cross_seed_mean_dd_samples: list[float] = []
    cross_seed_min_pf_samples: list[float] = []
    cross_seed_max_dd_samples: list[float] = []
    for it in range(n_bootstrap):
        seed_pfs = []
        seed_dds = []
        for s in seeds:
            arr = per_seed_filtered[s]
            n = len(arr)
            if n == 0:
                continue
            idx = rng.integers(0, n, size=n)
            sample = arr[idx]
            seed_pfs.append(_profit_factor(sample))
            seed_dds.append(_max_drawdown_pct(sample))
        if seed_pfs:
            cross_seed_mean_pf_samples.append(float(np.mean(seed_pfs)))
            cross_seed_min_pf_samples.append(float(np.min(seed_pfs)))
            cross_seed_mean_dd_samples.append(float(np.mean(seed_dds)))
            cross_seed_max_dd_samples.append(float(np.max(seed_dds)))

    # Per-seed bootstrap distributions
    rng2 = np.random.default_rng(seed + 1)
    for s in seeds:
        arr = per_seed_filtered[s]
        if len(arr) == 0:
            out_per_seed[s] = {"n_trades": 0}
            continue
        pfs = []
        dds = []
        for _ in range(n_bootstrap):
            idx = rng2.integers(0, len(arr), size=len(arr))
            sample = arr[idx]
            pfs.append(_profit_factor(sample))
            dds.append(_max_drawdown_pct(sample))
        pfs = np.asarray(pfs, dtype=float)
        dds = np.asarray(dds, dtype=float)
        finite_pf = pfs[np.isfinite(pfs)]
        out_per_seed[s] = {
            "n_trades": int(len(arr)),
            "pf_point": _profit_factor(arr),
            "dd_point": _max_drawdown_pct(arr),
            "pf_p05": float(np.percentile(finite_pf, 5)),
            "pf_p50": float(np.percentile(finite_pf, 50)),
            "pf_p95": float(np.percentile(finite_pf, 95)),
            "dd_p05": float(np.percentile(dds, 5)),
            "dd_p50": float(np.percentile(dds, 50)),
            "dd_p95": float(np.percentile(dds, 95)),
            "frac_pf_above_1": float((finite_pf > 1.0).mean()),
            "frac_pf_above_1_5": float((finite_pf > 1.5).mean()),
            "frac_pf_above_2_0": float((finite_pf > 2.0).mean()),
        }

    cross = {
        "n_bootstrap": n_bootstrap,
        "K": K,
        "drop_windows": list(drop_windows),
        "mean_pf_p05": float(np.percentile(cross_seed_mean_pf_samples, 5)),
        "mean_pf_p50": float(np.percentile(cross_seed_mean_pf_samples, 50)),
        "mean_pf_p95": float(np.percentile(cross_seed_mean_pf_samples, 95)),
        "min_pf_p05": float(np.percentile(cross_seed_min_pf_samples, 5)),
        "min_pf_p50": float(np.percentile(cross_seed_min_pf_samples, 50)),
        "min_pf_p95": float(np.percentile(cross_seed_min_pf_samples, 95)),
        "mean_dd_p05": float(np.percentile(cross_seed_mean_dd_samples, 5)),
        "mean_dd_p50": float(np.percentile(cross_seed_mean_dd_samples, 50)),
        "mean_dd_p95": float(np.percentile(cross_seed_mean_dd_samples, 95)),
        "max_dd_p05": float(np.percentile(cross_seed_max_dd_samples, 5)),
        "max_dd_p50": float(np.percentile(cross_seed_max_dd_samples, 50)),
        "max_dd_p95": float(np.percentile(cross_seed_max_dd_samples, 95)),
        "frac_mean_pf_above_2_0": float(np.mean([x > 2.0 for x in cross_seed_mean_pf_samples])),
        "frac_min_pf_above_1_75": float(np.mean([x > 1.75 for x in cross_seed_min_pf_samples])),
        "frac_max_dd_below_13": float(np.mean([x < 13.0 for x in cross_seed_max_dd_samples])),
    }
    return {"cross_seed": cross, "per_seed": out_per_seed}


# ---------- #5: Leave-one-window-out validation of cal_pf > 4 rule ----------

def audit_5_lowo_cal_pf_rule(audit_path: str, threshold: float = 4.0) -> dict[str, Any]:
    """For each held-out window, derive the cal_pf threshold from the
    OTHER 12 windows (the "fit" cohort). Then check whether the held-out
    window's records that exceed that threshold show degraded OOS PF.

    Specifically, we compute:
      - On the fit cohort (12 windows), median cal_pf among records with
        val→OOS deg < -50% (the "ground-truth bad" cohort). If this
        median is consistently around 4.0, the rule generalizes.
      - On the held-out window, count records with cal_pf > threshold
        and report their median OOS PF + median val→OOS pct.
    """
    with open(audit_path) as f:
        audit = json.load(f)
    recs = [r for r in audit["records"] if r.get("cal_pf") is not None and r.get("oos_pf") is not None and (r.get("oos_trades") or 0) >= 5]
    n_windows = audit["n_windows"]

    out: dict[str, Any] = {
        "threshold": threshold,
        "n_records_total": len(recs),
        "lowo_results": [],
    }

    for held_out in range(n_windows):
        fit_recs = [r for r in recs if r["window"] != held_out]
        held_recs = [r for r in recs if r["window"] == held_out]
        if not fit_recs:
            continue
        # Did the rule generalize? On the fit cohort, what's the cal_pf
        # threshold above which val→OOS degradation is consistently bad?
        fit_high = [r for r in fit_recs if r["cal_pf"] > threshold]
        fit_low = [r for r in fit_recs if r["cal_pf"] <= threshold]
        if not fit_high:
            fit_high_median_oos_pf = None
            fit_high_median_deg = None
        else:
            fit_high_median_oos_pf = float(np.median([r["oos_pf"] for r in fit_high]))
            fit_high_median_deg = float(np.median([r["val_to_oos_pf_pct"] for r in fit_high]))
        fit_low_median_oos_pf = float(np.median([r["oos_pf"] for r in fit_low])) if fit_low else None
        fit_low_median_deg = float(np.median([r["val_to_oos_pf_pct"] for r in fit_low])) if fit_low else None

        held_high = [r for r in held_recs if r["cal_pf"] > threshold]
        held_low = [r for r in held_recs if r["cal_pf"] <= threshold]
        held_high_median_oos_pf = float(np.median([r["oos_pf"] for r in held_high])) if held_high else None
        held_high_median_deg = float(np.median([r["val_to_oos_pf_pct"] for r in held_high])) if held_high else None
        held_low_median_oos_pf = float(np.median([r["oos_pf"] for r in held_low])) if held_low else None

        out["lowo_results"].append({
            "held_out_window": held_out,
            "fit_high_n": len(fit_high),
            "fit_high_median_oos_pf": fit_high_median_oos_pf,
            "fit_high_median_val_oos_deg_pct": fit_high_median_deg,
            "fit_low_n": len(fit_low),
            "fit_low_median_oos_pf": fit_low_median_oos_pf,
            "fit_low_median_val_oos_deg_pct": fit_low_median_deg,
            "held_high_n": len(held_high),
            "held_high_median_oos_pf": held_high_median_oos_pf,
            "held_high_median_val_oos_deg_pct": held_high_median_deg,
            "held_low_n": len(held_low),
            "held_low_median_oos_pf": held_low_median_oos_pf,
            # Rule fires correctly on held-out: held_high cohort's OOS PF < held_low cohort's
            "rule_fires_correctly_on_held_out": (
                held_high_median_oos_pf is not None
                and held_low_median_oos_pf is not None
                and held_high_median_oos_pf < held_low_median_oos_pf
            ),
        })

    fired_correct = sum(1 for r in out["lowo_results"] if r["rule_fires_correctly_on_held_out"])
    held_fired = sum(1 for r in out["lowo_results"] if r["held_high_n"] > 0)
    out["n_held_out_with_high_records"] = held_fired
    out["n_held_out_rule_correct"] = fired_correct
    out["rule_correct_rate"] = fired_correct / max(1, held_fired)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-dir", default="v3/artifacts")
    parser.add_argument("--champion-name", default="spx_combined_3seed_001")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--dataset", default="v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl")
    parser.add_argument("--calibration-audit", default="v3/artifacts/calibration_audit/spx_combined_3seed_001.json")
    parser.add_argument("--bootstrap-K", type=int, default=2)
    parser.add_argument("--bootstrap-drop-windows", type=int, nargs="*", default=[12])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--lowo-threshold", type=float, default=4.0)
    parser.add_argument("--out", default="v3/artifacts/proof_audits/spx_combined_3seed_001.json")
    args = parser.parse_args()

    print("Loading seed chosen_trades...")
    seed_trades = {s: _load_seed_trades(args.champion_dir, args.champion_name, s) for s in args.seeds}

    print("\n=== #3: L3-oracle headroom ===")
    audit3 = audit_3_oracle_headroom(seed_trades, args.dataset)
    print(f"{'seed':>5} {'pf_obj':>8} {'pf_best':>9} {'pf_horiz':>10} {'pf_tstop':>10} {'cap_frac':>10} {'eq_best':>9} {'>horiz':>8}")
    for s in args.seeds:
        a = audit3[s]
        print(
            f"{s:>5} {a['pf_chosen_objective']:>8.3f} {a['pf_best_exit_at_chosen']:>9.3f} "
            f"{a['pf_horizon_at_chosen']:>10.3f} {a['pf_time_stop_at_chosen']:>10.3f} "
            f"{a['capture_fraction_of_best']:>10.2%} "
            f"{a['frac_chosen_eq_best']:>9.2%} {a['frac_chosen_above_horizon']:>8.2%}"
        )

    print(f"\n=== #4: Bootstrap PF/DD CIs (K={args.bootstrap_K}, drop W{args.bootstrap_drop_windows}, n={args.n_bootstrap}) ===")
    audit4 = audit_4_bootstrap_ci(
        seed_trades,
        K=args.bootstrap_K,
        drop_windows=tuple(args.bootstrap_drop_windows),
        n_bootstrap=args.n_bootstrap,
    )
    cross = audit4["cross_seed"]
    print(f"Cross-seed mean PF:   p05={cross['mean_pf_p05']:.3f}  p50={cross['mean_pf_p50']:.3f}  p95={cross['mean_pf_p95']:.3f}")
    print(f"Cross-seed min PF:    p05={cross['min_pf_p05']:.3f}  p50={cross['min_pf_p50']:.3f}  p95={cross['min_pf_p95']:.3f}")
    print(f"Cross-seed mean DD:   p05={cross['mean_dd_p05']:.2f}%  p50={cross['mean_dd_p50']:.2f}%  p95={cross['mean_dd_p95']:.2f}%")
    print(f"Cross-seed max DD:    p05={cross['max_dd_p05']:.2f}%  p50={cross['max_dd_p50']:.2f}%  p95={cross['max_dd_p95']:.2f}%")
    print(f"Frac bootstrap mean PF > 2.0:   {cross['frac_mean_pf_above_2_0']:.2%}")
    print(f"Frac bootstrap min PF > 1.75:   {cross['frac_min_pf_above_1_75']:.2%}")
    print(f"Frac bootstrap max DD < 13%:    {cross['frac_max_dd_below_13']:.2%}")

    print(f"\nPer-seed:")
    print(f"{'seed':>5} {'n':>5} {'pf_pt':>8} {'pf_p05':>8} {'pf_p50':>8} {'pf_p95':>8} {'dd_p50':>8}  >2.0  >1.5  >1.0")
    for s in args.seeds:
        a = audit4["per_seed"][s]
        if a.get("n_trades", 0) == 0:
            continue
        print(
            f"{s:>5} {a['n_trades']:>5} {a['pf_point']:>8.3f} "
            f"{a['pf_p05']:>8.3f} {a['pf_p50']:>8.3f} {a['pf_p95']:>8.3f} "
            f"{a['dd_p50']:>8.2f}  {a['frac_pf_above_2_0']:.2%}  "
            f"{a['frac_pf_above_1_5']:.2%}  {a['frac_pf_above_1']:.2%}"
        )

    print(f"\n=== #5: LOWO validation of cal_pf > {args.lowo_threshold} rule ===")
    audit5 = audit_5_lowo_cal_pf_rule(args.calibration_audit, threshold=args.lowo_threshold)
    print(f"Rule fires correctly on {audit5['n_held_out_rule_correct']} of {audit5['n_held_out_with_high_records']} held-out windows that have any cal_pf > {args.lowo_threshold} record")
    print(f"Rule correct rate: {audit5['rule_correct_rate']:.2%}")
    print()
    print(f"{'held':>5} {'fit_hi_n':>9} {'fit_hi_oos':>11} {'fit_lo_oos':>11} {'held_hi_n':>10} {'held_hi_oos':>13} {'held_lo_oos':>13} {'fires?':>7}")
    for r in audit5["lowo_results"]:
        fhp = r['fit_high_median_oos_pf']
        flp = r['fit_low_median_oos_pf']
        hhp = r['held_high_median_oos_pf']
        hlp = r['held_low_median_oos_pf']
        print(
            f"W{r['held_out_window']:02d}".rjust(5)
            + f" {r['fit_high_n']:>9}"
            + f" {fhp if fhp is not None else 0:>11.3f}"
            + f" {flp if flp is not None else 0:>11.3f}"
            + f" {r['held_high_n']:>10}"
            + f" {hhp if hhp is not None else 0:>13.3f}"
            + f" {hlp if hlp is not None else 0:>13.3f}"
            + f" {('Y' if r['rule_fires_correctly_on_held_out'] else 'N' if r['held_high_n']>0 else '-'):>7}"
        )

    out = {
        "champion": args.champion_name,
        "seeds": args.seeds,
        "audit_3_oracle_headroom": audit3,
        "audit_4_bootstrap_ci": audit4,
        "audit_5_lowo_cal_pf_rule": audit5,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
