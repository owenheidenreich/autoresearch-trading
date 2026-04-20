"""V2-pruned-gated with gated-train-only retrain.

User-locked (2026-04-20): narrow experiment — align the training
distribution to the deployed regime. Train the RF only on gate-passing
train days/bars; keep V1B gate selection, test-time gate, candidate
universe shape, contract selection, exits, controls, and folds
identical to V2-pruned-gated.

Hypothesis:
    Because the true edge lives in the `first15_range_pct >= 20 bps`
    regime, training the RF only on gate-passing train samples will
    reduce train/test regime-mismatch noise and improve held-out gated
    performance vs V2-pruned-gated.

Only change vs V2-pruned-gated:
- `collect_training_samples` filters to days where the first-15 range
  meets the gate, BEFORE enumerating admissible (bar, side) candidates
- V1B gate selection still uses the full original train_days (per
  user spec)
- Test-time logic is identical (same fixed gate, same model class,
  same seeds, same pop-A/pop-B methodology)

Plan: lab_notebook entry 2026-04-20 (V2-pruned-gated promotion).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from v2.core.chain_data import load_sidecar_cached, sidecar_path
from v2.analysis.mechanical_baseline_opening_reversion import (
    BAR_LO,
    BAR_HI,
    SkipRecord,
    SKIP_FIELDS,
    build_folds,
    day_ranges,
    feature_index_map,
    load_data,
    summarize_trades,
    write_csv,
    write_json,
)
from v2.analysis.mechanical_baseline_v1b_opening_reversion import V1BGateConfig
from v2.analysis import mechanical_baseline_v2_learned_scorer as v2_mod
from v2.analysis.mechanical_baseline_v2_pruned_gated import (
    FIRST15_RANGE_GATE,
    FIRST15_RANGE_GATE_BPS,
    GatedTrade,
    GATED_TRADE_FIELDS,
    _activate_pruned_feature_set,
    aggregate,
    day_passes_gate,
    paired_delta_bootstrap,
    pop_control_A,
    pop_control_B,
    secondary_wide_and_vix_mid,
)


EXPERIMENT_ID = "mechbase_opening_reversion_v2_pruned_gated_trained"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated_trained"
N_BOOTSTRAP = 1000
BASE_SEED = 42


class _FilteredFoldSpec:
    """Minimal fold_spec with a shortened train_days list."""
    def __init__(self, fs, train_days: list[str]):
        self.fold_idx = fs.fold_idx
        self.window_id = fs.window_id
        self.train_days = train_days
        self.val_days = fs.val_days
        self.test_days = fs.test_days


def split_train_by_gate(
    *,
    train_days: list[str],
    day_to_range: dict[str, tuple[int, int]],
    X_sim: np.ndarray,
    idx_first15_range: int,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for day in train_days:
        if day not in day_to_range:
            dropped.append(day)
            continue
        ds, de = day_to_range[day]
        ok, _ = day_passes_gate(X_sim[ds:de], idx_first15_range)
        if ok:
            kept.append(day)
        else:
            dropped.append(day)
    return kept, dropped


# ---------------------------------------------------------------------------
# Per-fold driver — identical to V2-pruned-gated's run_fold except for the
# training-sample-day filter between V1B gate selection and training.
# ---------------------------------------------------------------------------

def run_fold(
    *,
    fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
) -> dict:
    t0 = time.time()
    fold_idx = fold_spec.fold_idx
    print(f"\n=== Fold {fold_idx} | window={fold_spec.window_id} ===", flush=True)

    idx_first15_range = idx_map["first15_range_pct"]

    # --- V1B gate selection — UNCHANGED, uses full train_days ---
    config, _diag, _n_tr, _n_ent = v2_mod.select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    if config is None:
        print("  NO VIABLE GATE CONFIG", flush=True)
        return {"fold_idx": fold_idx, "strategy_trades": [], "skips": []}

    # --- Train-day filter BEFORE training-sample collection ---
    train_kept, train_dropped = split_train_by_gate(
        train_days=fold_spec.train_days, day_to_range=day_to_range,
        X_sim=X_sim, idx_first15_range=idx_first15_range,
    )
    n_train_kept = len(train_kept)
    n_train_dropped = len(train_dropped)
    n_train_total = n_train_kept + n_train_dropped
    print(f"  train_days filter: kept {n_train_kept}/{n_train_total} "
          f"({100.0 * n_train_kept / max(n_train_total, 1):.1f}%) as gate-passing", flush=True)

    filtered_spec = _FilteredFoldSpec(fold_spec, train_kept)
    feats, labels, _n_days = v2_mod.collect_training_samples(
        fold_spec=filtered_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    n_train_samples = int(feats.shape[0])
    print(f"  train_samples: n={n_train_samples} (gated-train only)", flush=True)

    model = v2_mod.train_scorer(feats, labels, seed=fold_idx * 10007 + 1)
    if model is None:
        print("  NO TRAINING SAMPLES", flush=True)
        return {"fold_idx": fold_idx, "strategy_trades": [], "skips": []}

    # Feature importance snapshot
    fi = [
        {"feature": (v2_mod.CORE_FEATURE_NAMES + ("side_indicator",))[i],
         "importance": float(model.feature_importances_[i])}
        for i in range(v2_mod.N_MODEL_FEATURES)
    ]

    # --- Test-time logic — identical to V2-pruned-gated ---
    gate_passing_days: list[str] = []
    abstains: list[SkipRecord] = []
    for day in fold_spec.test_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        ds, de = day_to_range[day]
        ok, val = day_passes_gate(X_sim[ds:de], idx_first15_range)
        if ok:
            gate_passing_days.append(day)
        else:
            abstains.append(SkipRecord(day, fold_idx, "strategy", -1, "X",
                                       f"gated_narrow_first15_{val:.5f}"))
    n_all_test = sum(1 for d in fold_spec.test_days if d in day_to_range)
    print(f"  test: {len(gate_passing_days)}/{n_all_test} days pass gate", flush=True)

    strategy_trades: list[GatedTrade] = []
    for day in gate_passing_days:
        sc = load_sidecar_cached(sidecar_path(sidecar_dir, day))
        ds, de = day_to_range[day]
        X_day = X[ds:de]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        cands = v2_mod._enumerate_candidates_on_day(
            sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, core_indices=core_indices, config=config,
        )
        if not cands:
            continue
        feat_mat = np.vstack([c["features"] for c in cands]).astype(np.float32)
        preds = model.predict(feat_mat)
        pos = int(np.argmax(preds))
        chosen = cands[pos]
        v2trade, _ = v2_mod._realize_trade(
            day=day, fold_idx=fold_idx,
            strategy_label="strategy", paired_trade_id=-1,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map, sc=sc,
            entry_local=chosen["local_i"], side=chosen["side"],
            picked=chosen["picked"], context_spread=chosen["context_spread"],
            predicted_net_pct=float(preds[pos]),
            score_rank=1, n_candidates=len(cands),
        )
        if v2trade is None:
            continue
        pop_a, pop_a_n = pop_control_A(
            sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, core_indices=core_indices, config=config,
            strategy_side=chosen["side"],
        )
        pop_b, pop_b_n = pop_control_B(
            strategy_date=day, gate_passing_dates=gate_passing_days,
            entry_local=chosen["local_i"], strategy_side=chosen["side"],
            day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, config=config,
        )
        first15_val = float(X_sim_day[15, idx_first15_range]) if X_sim_day.shape[0] > 15 else 0.0
        strategy_trades.append(GatedTrade(
            **{**asdict(v2trade),
               "first15_range_pct": first15_val,
               "pop_ctrl_a_mean_net_pct": pop_a,
               "pop_ctrl_a_n_bars": pop_a_n,
               "pop_ctrl_b_mean_net_pct": pop_b,
               "pop_ctrl_b_n_days": pop_b_n,
               "delta_vs_pop_A": float(v2trade.net_pct - pop_a),
               "delta_vs_pop_B": float(v2trade.net_pct - pop_b),
               },
        ))

    s = summarize_trades(strategy_trades)
    print(f"  strategy: n={s['n']}, mean_net_pct={s['mean_net_pct']:+.5f}, "
          f"target={s['target_hit_frac']:.3f}, stop={s['stop_hit_frac']:.3f}, "
          f"dollar_pf={s['dollar_pf']:.3f}", flush=True)
    if strategy_trades:
        a_vals = np.array([t.pop_ctrl_a_mean_net_pct for t in strategy_trades])
        print(f"  pop_A mean: {a_vals.mean():+.5f}, gap: {s['mean_net_pct'] - a_vals.mean():+.5f}",
              flush=True)

    return {
        "fold_idx": fold_idx, "window_id": fold_spec.window_id,
        "gate_config": asdict(config),
        "n_train_days_total": n_train_total,
        "n_train_days_kept": n_train_kept,
        "n_train_days_dropped": n_train_dropped,
        "n_train_samples": n_train_samples,
        "n_all_test_days": n_all_test,
        "n_gate_passing_days": len(gate_passing_days),
        "gate_passing_days": gate_passing_days,
        "strategy_trades": strategy_trades,
        "skips": abstains,
        "feature_importance": fi,
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    _activate_pruned_feature_set()
    print(f"[gated-trained] Active CORE_FEATURE_NAMES: {list(v2_mod.CORE_FEATURE_NAMES)}")
    print(f"[gated-trained] Regime gate (train filter + test gate): "
          f"first15_range_pct >= {FIRST15_RANGE_GATE:.4f} ({FIRST15_RANGE_GATE_BPS} bps)")

    t0 = time.time()
    data = load_data(args.data)
    dates = list(data["dates"])
    feature_names = list(data["feature_names"])
    idx_map = feature_index_map(feature_names)
    required = (list(v2_mod.CORE_FEATURE_NAMES)
                + ["vix_regime", "iv_percentile", "vrp", "first15_range_pct"])
    missing = [n for n in required if n not in idx_map]
    if missing:
        print(f"FATAL: feature_names missing: {missing}", file=sys.stderr)
        return 2
    core_idx = v2_mod.core_feature_indices(idx_map)

    X = data["X"].numpy() if isinstance(data["X"], torch.Tensor) else np.asarray(data["X"])
    X_sim = data["X_sim"].numpy() if isinstance(data["X_sim"], torch.Tensor) else np.asarray(data["X_sim"])
    spot_prices = data["spot_prices"].numpy() if isinstance(data["spot_prices"], torch.Tensor) else np.asarray(data["spot_prices"])

    day_to_range, _ = day_ranges(dates)
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dataset_fp = str(data["metadata"].get("fingerprint", "unknown"))

    folds = build_folds(dates)
    os.makedirs(args.out_dir, exist_ok=True)

    per_fold_results: list[dict] = []
    for fs in folds:
        per_fold_results.append(run_fold(
            fold_spec=fs, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X=X, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, core_indices=core_idx,
        ))

    all_trades: list[GatedTrade] = []
    all_skips: list[SkipRecord] = []
    for r in per_fold_results:
        all_trades.extend(r["strategy_trades"])
        all_skips.extend(r["skips"])
    write_csv(os.path.join(args.out_dir, "trades.csv"), all_trades, GATED_TRADE_FIELDS)
    write_csv(os.path.join(args.out_dir, "skips.csv"), all_skips, SKIP_FIELDS)

    agg = aggregate(all_trades)
    boot = paired_delta_bootstrap(all_trades, n_boot=args.n_boot, seed=BASE_SEED)
    sec = secondary_wide_and_vix_mid(all_trades, n_boot=args.n_boot, seed=BASE_SEED + 1)

    print("\n=== Gated-trained aggregate (PRIMARY) ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:+.5f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Bootstrap CIs (day-level, paired delta) ===")
    for k, s in boot.items():
        print(f"  {k}: p50={s['p50']:+.5f}, 95%CI=[{s['p2_5']:+.5f}, {s['p97_5']:+.5f}], "
              f"frac>0={s['frac_gt_0']:.3f}")

    print("\n=== Per-fold gated-trained ===")
    for r in per_fold_results:
        st = r["strategy_trades"]
        ra = aggregate(st)
        if "n" in ra and ra["n"] > 0:
            print(f"  fold {r['fold_idx']}: "
                  f"train_days kept {r['n_train_days_kept']}/{r['n_train_days_total']} "
                  f"({100.0 * r['n_train_days_kept']/max(r['n_train_days_total'],1):.1f}%), "
                  f"n_train_samples={r['n_train_samples']}, "
                  f"test n={ra['n']}, mean_net_pct={ra['strategy_mean_net_pct']:+.5f}, "
                  f"gap_vs_A={ra.get('gap_vs_A', 0):+.5f}, PF={ra['strategy_dollar_pf']:.3f}")

    print("\n=== Feature importance (averaged across folds) ===")
    all_fi: dict[str, list[float]] = {}
    for r in per_fold_results:
        for item in r.get("feature_importance", []):
            all_fi.setdefault(item["feature"], []).append(item["importance"])
    fi_agg = [
        {"feature": name, "importance_mean": float(np.mean(vals)),
         "per_fold": [round(v, 4) for v in vals]}
        for name, vals in sorted(all_fi.items(), key=lambda kv: -float(np.mean(kv[1])))
    ]
    for row in fi_agg[:6]:
        print(f"  {row['feature']:<26} mean={row['importance_mean']:.4f}, "
              f"per_fold={row['per_fold']}")

    print("\n=== Secondary: wide AND vix_regime=mid ===")
    for k, v in sec.items():
        if k == "bootstrap":
            for bk, bs in v.items():
                print(f"  boot {bk}: p50={bs['p50']:+.5f}, 95%CI=[{bs['p2_5']:+.5f}, "
                      f"{bs['p97_5']:+.5f}], frac>0={bs['frac_gt_0']:.3f}")
        elif isinstance(v, float):
            print(f"  {k}: {v:+.5f}")
        else:
            print(f"  {k}: {v}")

    write_json(os.path.join(args.out_dir, "summary.json"), {
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint": dataset_fp,
        "gate_threshold_bps": FIRST15_RANGE_GATE_BPS,
        "gate_threshold_fraction": FIRST15_RANGE_GATE,
        "aggregate": agg,
        "bootstrap": boot,
        "per_fold": [
            {
                "fold_idx": r["fold_idx"],
                "window_id": r["window_id"],
                "n_train_days_total": r.get("n_train_days_total", 0),
                "n_train_days_kept": r.get("n_train_days_kept", 0),
                "n_train_days_dropped": r.get("n_train_days_dropped", 0),
                "n_train_samples": r.get("n_train_samples", 0),
                "n_all_test_days": r.get("n_all_test_days", 0),
                "n_gate_passing_days": r.get("n_gate_passing_days", 0),
                "aggregate": aggregate(r["strategy_trades"]),
                "feature_importance": r.get("feature_importance", []),
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in per_fold_results
        ],
        "feature_importance_agg": fi_agg,
        "secondary_appendix_vix_mid": sec,
        "note": "Train-sample collection filtered to gate-passing days; V1B gate "
                "selection and test-time logic unchanged from V2-pruned-gated.",
    })
    print(f"\nWrote summary to {args.out_dir}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
