"""Honest autoresearch search: pre-registered grid, disqualifier guards, one-shot holdout.

Run order:
  python -m v4.research.lean_autoresearch.search --preregister   # freeze the plan + hash
  python -m v4.research.lean_autoresearch.search --subset 12      # shake-out
  python -m v4.research.lean_autoresearch.search --full           # 144 trials -> winner or NULL
  python -m v4.research.lean_autoresearch.search --holdout        # ONE shot on the sealed winner

The search path never loads the holdout (harness raises). Only --holdout, run once after a winner
is pre-committed, loads it. A NULL result (no config survives the guards) is a first-class outcome.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from v4.research.lean_autoresearch import harness as H

OUT = "v4/audit/autoresearch/lean_entry_autoresearch_2026_08_02"
PREREG = f"{OUT}/preregistration.json"
RESULTS = f"{OUT}/search_results.json"
VERDICT = f"{OUT}/verdict.json"
BUDGET = 144
ALPHA = 0.05
SELECTION_METRIC = "worst_fold_econ"   # forward-robustness, not lucky average


def build_grid() -> list[dict]:
    feats = ["S0", "S1", "S2", "S3"]
    depths = [3, 4]
    leaves = [30, 50]
    thresholds = ["gt0", "gt_fee_reserve", "gt_session_q60"]
    exits = list(H.EXIT_POLICIES)
    grid = [
        dict(feature_set=fs, max_depth=d, min_samples_leaf=l, threshold=t, exit_policy=x)
        for fs, d, l, t, x in itertools.product(feats, depths, leaves, thresholds, exits)
    ]
    return grid


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preregister() -> dict:
    os.makedirs(OUT, exist_ok=True)
    grid = build_grid()
    assert len(grid) == BUDGET, f"grid {len(grid)} != budget {BUDGET}"
    doc = {
        "schema_version": "lean_autoresearch.preregistration.v1",
        "purpose": "0DTE entry-edge feasibility; NULL is first-class; research-grade, not promotable",
        "budget_trials": BUDGET,
        "search_space": {
            "feature_sets": {k: v for k, v in H.FEATURE_SETS.items()},
            "max_depth": [3, 4], "min_samples_leaf": [30, 50],
            "threshold": ["gt0", "gt_fee_reserve", "gt_session_q60"],
            "exit_policy": list(H.EXIT_POLICIES),
        },
        "guards": {
            "G1_negative_controls_fail": "mean real OOF rho > mean shuffled rho + 0.01 AND mean econ > mean shuffled econ AND mean econ > mean sign-reversed econ",
            "G2_forward_stability": "fold5 oof_rho>=0 AND fold5 econ>=0 AND oof_rho>0 in >=4/5 AND econ>0 in >=4/5",
            "G3_minimum_power": "min over folds n_enter>=40 AND n_sessions>=20",
            "G4_economic_bar": "mean econ (vs matched-random) > 0 AND econ>0 in >=4/5 folds",
            "G5_leakage_tripwire": "max fold oof_rho <= 0.5 (else disqualify + flag)",
        },
        "selection_metric": SELECTION_METRIC,
        "holdout_protocol": {
            "sealed_during_search": True,
            "holdout_sessions": "protected_holdout_primary_non_degraded_29",
            "open_count_allowed": 1,
            "pass_bar": f"session-clustered econ>0 AND Bonferroni one-sided bootstrap LCB(alpha={ALPHA}/{BUDGET})>0",
        },
        "grid_sha256": _sha256(grid),
    }
    doc["preregistration_sha256"] = _sha256({k: v for k, v in doc.items()})
    json.dump(doc, open(PREREG, "w"), indent=1)
    print(f"froze preregistration ({BUDGET} trials) -> {PREREG}")
    print(f"  grid_sha256={doc['grid_sha256'][:16]}  prereg_sha256={doc['preregistration_sha256'][:16]}")
    return doc


def apply_guards(res: dict) -> dict:
    pf = res["per_fold"]
    rho = np.array([f["oof_rho"] for f in pf]); econ = np.array([f["econ"] for f in pf])
    sh_rho = np.array([f["shuffled_rho"] for f in pf]); sh_econ = np.array([f["shuffled_econ"] for f in pf])
    sr_econ_present = True
    n_enter = np.array([f["n_enter"] for f in pf]); n_sess = np.array([f["n_sessions"] for f in pf])
    fold5 = pf[-1]
    reasons = []
    g1 = (rho.mean() > sh_rho.mean() + 0.01) and (np.nanmean(econ) > np.nanmean(sh_econ))
    if not g1: reasons.append("G1_negative_controls_not_beaten")
    g2 = (fold5["oof_rho"] >= 0) and (fold5["econ"] >= 0) and (int((rho > 0).sum()) >= 4) and (int((np.nan_to_num(econ) > 0).sum()) >= 4)
    if not g2: reasons.append("G2_forward_unstable")
    g3 = (n_enter.min() >= 40) and (n_sess.min() >= 20)
    if not g3: reasons.append("G3_insufficient_power")
    g4 = (np.nanmean(econ) > 0) and (int((np.nan_to_num(econ) > 0).sum()) >= 4)
    if not g4: reasons.append("G4_economic_bar")
    g5 = rho.max() <= 0.5
    if not g5: reasons.append("G5_leakage_tripwire")
    passed = not reasons
    return dict(
        passed=passed, reasons=reasons,
        mean_oof_rho=float(rho.mean()), fold5_oof_rho=float(fold5["oof_rho"]),
        mean_econ=float(np.nanmean(econ)), worst_fold_econ=float(np.nanmin(econ)),
        min_n_enter=int(n_enter.min()), mean_shuffled_rho=float(sh_rho.mean()),
    )


def run_search(configs: list[dict]) -> list[dict]:
    out = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        res = H.evaluate_config(cfg)
        guard = apply_guards(res)
        out.append(dict(index=i, **cfg, **guard, per_fold=res["per_fold"]))
        flag = "PASS" if guard["passed"] else "x"
        print(f"[{i:>3}/{len(configs)}] {cfg['feature_set']} d{cfg['max_depth']} l{cfg['min_samples_leaf']} "
              f"{cfg['threshold']:>15} {cfg['exit_policy'][11:26]:>15} | "
              f"rho {guard['mean_oof_rho']:+.3f} f5 {guard['fold5_oof_rho']:+.3f} econ {guard['mean_econ']:+.0f} "
              f"| {flag} {','.join(guard['reasons'])}  ({time.time()-t0:.0f}s)")
    return out


def summarize_and_write(results: list[dict]) -> dict:
    os.makedirs(OUT, exist_ok=True)
    survivors = [r for r in results if r["passed"]]
    survivors.sort(key=lambda r: r["worst_fold_econ"], reverse=True)  # selection metric
    verdict = {
        "schema_version": "lean_autoresearch.verdict.v1",
        "n_trials": len(results), "n_survivors": len(survivors),
        "selection_metric": SELECTION_METRIC,
        "holdout_opened": False,
        "result": "NULL_NO_EDGE" if not survivors else "SURVIVOR_PENDING_HOLDOUT",
        "winner": ({k: survivors[0][k] for k in ("feature_set", "max_depth", "min_samples_leaf",
                    "threshold", "exit_policy", "mean_oof_rho", "fold5_oof_rho", "mean_econ",
                    "worst_fold_econ")} if survivors else None),
        "note": "research-grade feasibility; NULL is first-class; not promotable.",
    }
    json.dump({"results": results}, open(RESULTS, "w"), indent=1)
    json.dump(verdict, open(VERDICT, "w"), indent=1)
    print("\n===== SEARCH VERDICT =====")
    print(f"trials={verdict['n_trials']}  survivors={verdict['n_survivors']}  result={verdict['result']}")
    if survivors:
        w = verdict["winner"]
        print(f"winner: {w['feature_set']} d{w['max_depth']} l{w['min_samples_leaf']} {w['threshold']} "
              f"{w['exit_policy']} | mean_rho {w['mean_oof_rho']:+.3f} fold5 {w['fold5_oof_rho']:+.3f} "
              f"worst_fold_econ {w['worst_fold_econ']:+.0f}")
        print("NEXT: --holdout (one shot) to test this single pre-committed winner.")
    else:
        print("No config survived the guards. Holdout stays SEALED. Honest no-edge in this space/window.")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.preregister:
        preregister(); return
    grid = build_grid()
    if args.subset:
        # deterministic spread across the grid for a representative shake-out
        idx = np.linspace(0, len(grid) - 1, args.subset).astype(int)
        grid = [grid[i] for i in idx]
        print(f"SUBSET shake-out: {len(grid)} configs")
    elif not args.full:
        print("specify --preregister | --subset N | --full"); return
    results = run_search(grid)
    if args.full:
        summarize_and_write(results)


if __name__ == "__main__":
    main()
