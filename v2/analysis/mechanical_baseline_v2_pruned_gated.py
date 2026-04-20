"""V2-pruned with fixed first15_range_pct >= 20 bps regime gate.

User-locked (2026-04-20): single, pre-declared, prospectively-specifiable
opening-regime gate. Training pipeline unchanged; gate applied at test
time only. No train-picked selection across candidate gates — that would
reintroduce the bounded-search overfit the thresholding run exposed.

Hypothesis:
    V2-pruned's edge is materially stronger when the opening auction is
    wide enough to indicate real displacement/acceptance dynamics.
    Narrow first-15 regimes (< 20 bps range) are low-opportunity / chop
    for this thesis and should be skipped.

Pipeline (freeze except test-time gate):
- V1B gate selection unchanged
- Candidate universe unchanged (all admissible bars in [BAR_LO, BAR_HI])
- RF training data collection unchanged (trains on ALL admissible
  train-day samples including narrow days — model sees the full
  feature space)
- RF training unchanged (same hparams, seed, OOB disabled since not
  needed here)
- Exit engine unchanged
- One trade per day rule unchanged
- Folds unchanged

Only change:
- At test time, before RF scoring, check `first15_range_pct[entry_bar]`.
  If < 0.002, skip the day with reason `gated_narrow_first15`.

Comparators:
- Primary: population-mean Control A (per strategy trade, mean over all
  V1B-admissible same-day same-side bars). RNG-free.
- Secondary: population-mean Control B (per strategy trade, mean over
  the realized trade at the SAME bar and side on every other
  gate-passing test day in the same fold; attempts fail when V1B gates
  don't pass on that other day or no contract).

Plan: regime-audit lab_notebook entry, 2026-04-20.
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
    BaselineTrade,
    SkipRecord,
    SKIP_FIELDS,
    build_folds,
    context_spread_at_bar,
    day_ranges,
    feature_index_map,
    load_data,
    summarize_trades,
    write_csv,
    write_json,
)
from v2.analysis.mechanical_baseline_v1b_opening_reversion import (
    check_gates,
    V1BGateConfig,
)
from v2.analysis import mechanical_baseline_v2_learned_scorer as v2_mod


# ---------------------------------------------------------------------------
# Locked params
# ---------------------------------------------------------------------------

DROPPED_FEATURES = ("vwap_reclaim_state",)
FIRST15_RANGE_GATE_BPS = 20                            # pinned from regime audit
FIRST15_RANGE_GATE = FIRST15_RANGE_GATE_BPS / 10000.0  # 0.0020 as a fraction

EXPERIMENT_ID = "mechbase_opening_reversion_v2_pruned_gated"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated"
N_BOOTSTRAP = 1000
BASE_SEED = 42


def _activate_pruned_feature_set() -> None:
    active = tuple(n for n in v2_mod.CORE_FEATURE_NAMES if n not in DROPPED_FEATURES)
    v2_mod.CORE_FEATURE_NAMES = active
    v2_mod.N_MODEL_FEATURES = len(active) + 1
    v2_mod.SIDE_INDICATOR_IDX = len(active)


# ---------------------------------------------------------------------------
# Extended trade record — carries regime value and population-control stats
# ---------------------------------------------------------------------------

@dataclass
class GatedTrade(v2_mod.V2Trade):
    first15_range_pct: float = 0.0
    pop_ctrl_a_mean_net_pct: float = 0.0
    pop_ctrl_a_n_bars: int = 0
    pop_ctrl_b_mean_net_pct: float = 0.0
    pop_ctrl_b_n_days: int = 0
    delta_vs_pop_A: float = 0.0
    delta_vs_pop_B: float = 0.0


GATED_TRADE_FIELDS = list(GatedTrade.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# Population-mean Control A
# ---------------------------------------------------------------------------

def pop_control_A(
    *,
    sc: dict[str, Any],
    X_day: np.ndarray,
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config: V1BGateConfig,
    strategy_side: str,
) -> tuple[float, int]:
    cands = v2_mod._enumerate_candidates_on_day(
        sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    side_cands = [c for c in cands if c["side"] == strategy_side]
    if not side_cands:
        return 0.0, 0
    nps: list[float] = []
    for cand in side_cands:
        t, _ = v2_mod._realize_trade(
            day="_pop", fold_idx=-1,
            strategy_label="pop_A", paired_trade_id=-1,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map, sc=sc,
            entry_local=cand["local_i"], side=cand["side"],
            picked=cand["picked"], context_spread=cand["context_spread"],
        )
        if t is not None:
            nps.append(t.net_pct)
    if not nps:
        return 0.0, 0
    return float(np.mean(nps)), len(nps)


# ---------------------------------------------------------------------------
# Population-mean Control B (over all OTHER gate-passing test days in the fold)
# ---------------------------------------------------------------------------

def pop_control_B(
    *,
    strategy_date: str,
    gate_passing_dates: list[str],
    entry_local: int,
    strategy_side: str,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    config: V1BGateConfig,
) -> tuple[float, int]:
    """For each gate-passing other day, attempt to realize a trade at the
    same bar and side. Returns mean net_pct across successful attempts."""
    idx_iv = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]
    idx_spread = idx_map["option_spread_pct"]
    nps: list[float] = []
    for d in gate_passing_dates:
        if d == strategy_date or d not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, d)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[d]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        if entry_local >= X_sim_day.shape[0]:
            continue
        ok, _ = check_gates(X_sim_day, entry_local, idx_iv, idx_vrp, config)
        if not ok:
            continue
        cs = context_spread_at_bar(X_sim_day, entry_local, idx_spread)
        picked, _ = v2_mod.select_contract(sc, entry_local, strategy_side, cs)
        if picked is None:
            continue
        t, _ = v2_mod._realize_trade(
            day="_popB", fold_idx=-1,
            strategy_label="pop_B", paired_trade_id=-1,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map, sc=sc,
            entry_local=entry_local, side=strategy_side,
            picked=picked, context_spread=cs,
        )
        if t is not None:
            nps.append(t.net_pct)
    if not nps:
        return 0.0, 0
    return float(np.mean(nps)), len(nps)


# ---------------------------------------------------------------------------
# Per-fold driver
# ---------------------------------------------------------------------------

def day_passes_gate(X_sim_day: np.ndarray, idx_first15_range: int) -> tuple[bool, float]:
    if X_sim_day.shape[0] <= 15:
        return False, 0.0
    val = float(X_sim_day[15, idx_first15_range])
    return (val >= FIRST15_RANGE_GATE), val


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

    # Gate selection + RF fit (unchanged V2-pruned pipeline)
    config, _diag, _n_tr, _n_ent = v2_mod.select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    if config is None:
        print("  NO VIABLE GATE CONFIG", flush=True)
        return {"fold_idx": fold_idx, "strategy_trades": [], "skips": []}
    feats, labels, _ = v2_mod.collect_training_samples(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    model = v2_mod.train_scorer(feats, labels, seed=fold_idx * 10007 + 1)
    print(f"  train: n_samples={feats.shape[0]}", flush=True)

    idx_first15_range = idx_map["first15_range_pct"]

    # First pass — identify gate-passing test days (deterministic from data)
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
    n_all_test_days = sum(1 for d in fold_spec.test_days if d in day_to_range)
    print(f"  gate: {len(gate_passing_days)}/{n_all_test_days} "
          f"({100.0 * len(gate_passing_days) / max(n_all_test_days, 1):.1f}%) days pass "
          f"first15_range_pct >= {FIRST15_RANGE_GATE:.4f}", flush=True)

    # Second pass — strategy argmax on gate-passing days + pop A/B
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
        b_vals = np.array([t.pop_ctrl_b_mean_net_pct for t in strategy_trades
                           if t.pop_ctrl_b_n_days > 0])
        print(f"  pop_A: mean={a_vals.mean():+.5f}, gap={s['mean_net_pct'] - a_vals.mean():+.5f}",
              flush=True)
        if len(b_vals):
            print(f"  pop_B: mean={b_vals.mean():+.5f} (from {len(b_vals)} trades), "
                  f"gap={s['mean_net_pct'] - b_vals.mean():+.5f}", flush=True)

    return {
        "fold_idx": fold_idx, "window_id": fold_spec.window_id,
        "gate_config": asdict(config),
        "n_all_test_days": n_all_test_days,
        "n_gate_passing_days": len(gate_passing_days),
        "gate_passing_days": gate_passing_days,
        "strategy_trades": strategy_trades,
        "skips": abstains,
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Aggregation + paired-delta bootstrap (day-level)
# ---------------------------------------------------------------------------

def paired_delta_bootstrap(
    trades: list[GatedTrade],
    n_boot: int,
    seed: int,
) -> dict:
    if not trades:
        return {"n": 0}
    # Group by (fold, date) for day-level resample
    by_day: dict[tuple[int, str], list[GatedTrade]] = {}
    for t in trades:
        by_day.setdefault((t.fold, t.date), []).append(t)
    keys = list(by_day.keys())
    rng = np.random.default_rng(seed)
    reps_strat: list[float] = []
    reps_gap_A: list[float] = []
    reps_gap_B: list[float] = []
    reps_pf: list[float] = []
    for _ in range(n_boot):
        sampled = [keys[i] for i in rng.integers(0, len(keys), size=len(keys))]
        sampled_trades = [t for k in sampled for t in by_day[k]]
        if not sampled_trades:
            reps_strat.append(0.0)
            reps_gap_A.append(0.0)
            reps_gap_B.append(0.0)
            reps_pf.append(0.0)
            continue
        s = np.array([t.net_pct for t in sampled_trades])
        a = np.array([t.pop_ctrl_a_mean_net_pct for t in sampled_trades])
        b_subset = [t for t in sampled_trades if t.pop_ctrl_b_n_days > 0]
        reps_strat.append(float(s.mean()))
        reps_gap_A.append(float((s - a).mean()))
        if b_subset:
            s_b = np.array([t.net_pct for t in b_subset])
            b_arr = np.array([t.pop_ctrl_b_mean_net_pct for t in b_subset])
            reps_gap_B.append(float((s_b - b_arr).mean()))
        else:
            reps_gap_B.append(0.0)
        dollars = np.array([t.net_pnl_dollars for t in sampled_trades])
        gp = float(dollars[dollars > 0].sum())
        gl = float(-dollars[dollars < 0].sum())
        reps_pf.append((gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else 0.0))

    def _summary(vals: list[float]) -> dict:
        arr = np.array(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]  # skip non-finite (inf PF)
        if len(arr) == 0:
            return {"p2_5": 0.0, "p50": 0.0, "p97_5": 0.0, "mean": 0.0, "frac_gt_0": 0.0}
        return {
            "p2_5": float(np.quantile(arr, 0.025)),
            "p50": float(np.quantile(arr, 0.50)),
            "p97_5": float(np.quantile(arr, 0.975)),
            "mean": float(arr.mean()),
            "frac_gt_0": float((arr > 0).mean()),
        }

    return {
        "strategy_mean_net_pct": _summary(reps_strat),
        "gap_vs_A": _summary(reps_gap_A),
        "gap_vs_B": _summary(reps_gap_B),
        "strategy_dollar_pf": _summary(reps_pf),
    }


def aggregate(trades: list[GatedTrade]) -> dict:
    if not trades:
        return {"n": 0}
    s = summarize_trades(trades)
    a_vals = np.array([t.pop_ctrl_a_mean_net_pct for t in trades])
    b_subset = [t for t in trades if t.pop_ctrl_b_n_days > 0]
    out = {
        "n": s["n"],
        "strategy_mean_net_pct": s["mean_net_pct"],
        "strategy_dollar_pf": s["dollar_pf"],
        "target_hit_frac": s["target_hit_frac"],
        "stop_hit_frac": s["stop_hit_frac"],
        "time_stop_frac": s["time_stop_frac"],
        "pop_ctrl_a_mean": float(a_vals.mean()),
        "gap_vs_A": float(s["mean_net_pct"] - a_vals.mean()),
    }
    if b_subset:
        b_arr = np.array([t.pop_ctrl_b_mean_net_pct for t in b_subset])
        s_b = summarize_trades(b_subset)
        out["pop_ctrl_b_mean"] = float(b_arr.mean())
        out["gap_vs_B"] = float(s_b["mean_net_pct"] - b_arr.mean())
        out["n_with_popB"] = len(b_subset)
    return out


# ---------------------------------------------------------------------------
# Secondary appendix: wide AND vix_regime=mid
# ---------------------------------------------------------------------------

def secondary_wide_and_vix_mid(trades: list[GatedTrade], n_boot: int, seed: int) -> dict:
    """Filter to `vix_regime_at_entry == +0.33` (mid) within already-gated trades."""
    sub = [t for t in trades if abs(t.vix_regime_at_entry - 0.33) < 0.01]
    agg = aggregate(sub)
    if sub:
        boot = paired_delta_bootstrap(sub, n_boot=n_boot, seed=seed)
        agg["bootstrap"] = boot
    return agg


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
    print(f"[gated] Active CORE_FEATURE_NAMES: {list(v2_mod.CORE_FEATURE_NAMES)}")
    print(f"[gated] Regime gate: first15_range_pct >= {FIRST15_RANGE_GATE:.4f} "
          f"({FIRST15_RANGE_GATE_BPS} bps)")

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

    print("\n=== Gated aggregate (PRIMARY) ===")
    agg = aggregate(all_trades)
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:+.5f}")
        else:
            print(f"  {k}: {v}")

    boot_primary = paired_delta_bootstrap(all_trades, n_boot=args.n_boot, seed=BASE_SEED)
    print("\n=== Bootstrap CIs (day-level, paired delta) ===")
    for k, s in boot_primary.items():
        print(f"  {k}: p50={s['p50']:+.5f}, 95%CI=[{s['p2_5']:+.5f}, {s['p97_5']:+.5f}], "
              f"frac>0={s['frac_gt_0']:.3f}")

    # Secondary: vix_regime = mid subset
    print(f"\n=== Secondary appendix: wide AND vix_regime=mid ===")
    sec = secondary_wide_and_vix_mid(all_trades, n_boot=args.n_boot, seed=BASE_SEED + 1)
    for k, v in sec.items():
        if k == "bootstrap":
            for bk, bs in v.items():
                print(f"  boot {bk}: p50={bs['p50']:+.5f}, 95%CI=[{bs['p2_5']:+.5f}, "
                      f"{bs['p97_5']:+.5f}], frac>0={bs['frac_gt_0']:.3f}")
        elif isinstance(v, float):
            print(f"  {k}: {v:+.5f}")
        else:
            print(f"  {k}: {v}")

    # Coverage / abstention summary
    n_all_test = sum(r.get("n_all_test_days", 0) for r in per_fold_results)
    n_gate_passing = sum(r.get("n_gate_passing_days", 0) for r in per_fold_results)
    n_abstain = len(all_skips)
    print(f"\n=== Coverage summary ===")
    print(f"  total test days: {n_all_test}")
    print(f"  gate-passing days: {n_gate_passing} ({100 * n_gate_passing / max(n_all_test, 1):.1f}%)")
    print(f"  gated-out (narrow): {n_abstain} ({100 * n_abstain / max(n_all_test, 1):.1f}%)")
    print(f"  strategy trades produced: {agg['n']}")

    # Per-fold table
    print(f"\n=== Per-fold gated results ===")
    per_fold_rows: list[dict] = []
    for r in per_fold_results:
        st = r["strategy_trades"]
        ra = aggregate(st)
        if "n" in ra and ra["n"] > 0:
            print(f"  fold {r['fold_idx']}: n={ra['n']:3d}, mean_net_pct={ra['strategy_mean_net_pct']:+.5f}, "
                  f"pop_A_mean={ra.get('pop_ctrl_a_mean', 0):+.5f}, "
                  f"gap_vs_A={ra.get('gap_vs_A', 0):+.5f}, PF={ra['strategy_dollar_pf']:.3f}, "
                  f"days_gated_out={len(r['skips'])}")
        per_fold_rows.append({
            "fold_idx": r["fold_idx"],
            "window_id": r["window_id"],
            "n_gate_passing_days": r.get("n_gate_passing_days", 0),
            "n_all_test_days": r.get("n_all_test_days", 0),
            "aggregate": ra,
            "elapsed_sec": r["elapsed_sec"],
        })

    write_json(os.path.join(args.out_dir, "summary.json"), {
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint": dataset_fp,
        "gate_threshold_bps": FIRST15_RANGE_GATE_BPS,
        "gate_threshold_fraction": FIRST15_RANGE_GATE,
        "aggregate": agg,
        "bootstrap": boot_primary,
        "per_fold": per_fold_rows,
        "coverage": {
            "n_all_test_days": n_all_test,
            "n_gate_passing_days": n_gate_passing,
            "gate_pass_rate": n_gate_passing / max(n_all_test, 1),
            "n_gated_out": n_abstain,
        },
        "secondary_appendix_vix_mid": sec,
        "note": "Fixed pre-declared first15_range_pct gate, test-time only. "
                "Training pipeline unchanged from V2-pruned. Pop-A is RNG-free "
                "population mean over admissible same-day same-side bars.",
    })

    print(f"\nWrote summary to {args.out_dir}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
