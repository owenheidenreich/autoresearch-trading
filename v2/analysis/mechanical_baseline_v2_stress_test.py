"""Stress-test the V2 learned-scorer baseline (frozen pipeline).

Answers the question: is V2's +2.28pp gap vs Control A a real learned edge,
or an artifact that could arise from a sufficiently flexible RF fit on
noise? And is the top-importance story (opening_gap_pct + first15_*)
the actual driver?

Four diagnostics — no changes to folds, gates, candidate universe,
contract selection, exits, RF model class / hparams, or controls:

1. **Day-level bootstrap CIs.** Resample test_days per fold with
   replacement, recompute aggregate metrics. Respects the day-level
   dependence structure (intra-day trades aren't independent).
2. **Null/permutation test.** Shuffle train labels within fold before
   RF fit, re-score test, re-run controls. Repeat N_PERM times to build
   a null distribution of strategy mean and strategy-vs-Control-A gap.
3. **Feature ablations.** Zero out one (or several) of the 13 model
   inputs in both train and test feature matrices, refit RF, re-score,
   compare aggregate.
4. (Optional, secondary) Shorter-window fold structure. Not run by
   default — kept as a diagnostic that the current 5-fold result is
   stable under a denser walk-forward.

Architecture: precompute a FoldCache per fold (gate config + train
features/labels + test candidates with realized trades). Everything else
reuses that cache so RF refits are fast. Null and ablation runs are
minutes, not hours.

Plan: [v2/docs/mechanical_baseline_v2_plan.md] (V2 pipeline).
Implementation references: mechanical_baseline_v2_learned_scorer.py for
the real V2 pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from v2.core.chain_data import load_sidecar_cached, sidecar_path
from v2.core.walkforward import CANONICAL_N_FOLDS
from v2.analysis.mechanical_baseline_opening_reversion import (
    BAR_LO,
    BAR_HI,
    build_folds,
    context_spread_at_bar,
    day_ranges,
    feature_index_map,
    load_data,
    summarize_trades,
    write_json,
)
from v2.analysis.mechanical_baseline_v1b_opening_reversion import (
    check_gates,
)
from v2.analysis.mechanical_baseline_v2_learned_scorer import (
    CORE_FEATURE_NAMES,
    N_MODEL_FEATURES,
    SIDE_INDICATOR_IDX,
    V2Trade,
    _enumerate_candidates_on_day,
    _realize_trade,
    core_feature_indices,
    select_gates_for_fold,
    train_scorer,
)


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 100
BASE_SEED = 42
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_stress"
CACHE_FILENAME = "fold_caches.pkl"


# ---------------------------------------------------------------------------
# Fold cache
# ---------------------------------------------------------------------------

@dataclass
class FoldCache:
    """Everything needed to rerun the RF step quickly under modifications.

    - `train_features` / `train_labels`: one row per (train day, bar, side)
      admissible sample. Labels are realized net_pct.
    - `test_candidates_by_day`: mapping date -> list of admissible
      (bar, side) candidates on that day, each with pre-simulated trade.
      The realized trade is fixed regardless of RF output; only which
      candidate gets picked depends on RF.
    - `test_trade_lookup`: flat index for control lookups. Keyed by
      (date, local_i, side).
    """
    fold_idx: int
    window_id: str
    test_days: list[str]
    gate_config: dict                                   # V1BGateConfig asdict
    train_features: np.ndarray                          # (n_train, 13)
    train_labels: np.ndarray                            # (n_train,)
    test_candidates_by_day: dict[str, list[dict]]      # day -> list of cand dicts
    test_trade_lookup: dict[tuple[str, int, str], dict]  # (day, bar, side) -> cand dict


def _cand_to_serial(cand: dict) -> dict:
    """Ensure the candidate dict is picklable (V2Trade dataclass + numpy)."""
    out = dict(cand)
    out["features"] = np.asarray(cand["features"], dtype=np.float32)
    out["trade"] = asdict(cand["trade"]) if isinstance(cand["trade"], V2Trade) else cand["trade"]
    return out


def _cand_from_serial(cand: dict) -> dict:
    """Inverse of _cand_to_serial — rebuild a V2Trade from its dict form."""
    out = dict(cand)
    trade_d = cand["trade"]
    if isinstance(trade_d, dict):
        out["trade"] = V2Trade(**trade_d)
    return out


def _enumerate_train_samples(
    *,
    fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for day in fold_spec.train_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        cands = _enumerate_candidates_on_day(
            sc=sc,
            X_day=X[ds:de],
            X_sim_day=X_sim[ds:de],
            spot_day=spot_prices[ds:de],
            idx_map=idx_map,
            core_indices=core_indices,
            config=config,
        )
        for cand in cands:
            trade, status = _realize_trade(
                day=day, fold_idx=fold_spec.fold_idx,
                strategy_label="train_sample", paired_trade_id=-1,
                X_sim_day=X_sim[ds:de], spot_day=spot_prices[ds:de],
                idx_map=idx_map, sc=sc,
                entry_local=cand["local_i"], side=cand["side"],
                picked=cand["picked"], context_spread=cand["context_spread"],
            )
            if trade is None:
                continue
            rows.append(cand["features"])
            labels.append(trade.net_pct)
    if not rows:
        return (np.zeros((0, N_MODEL_FEATURES), dtype=np.float32),
                np.zeros(0, dtype=np.float32))
    return np.vstack(rows).astype(np.float32), np.array(labels, dtype=np.float32)


def _enumerate_test_cache(
    *,
    fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config,
) -> tuple[dict[str, list[dict]], dict[tuple[str, int, str], dict]]:
    by_day: dict[str, list[dict]] = {}
    lookup: dict[tuple[str, int, str], dict] = {}
    for day in fold_spec.test_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        cands = _enumerate_candidates_on_day(
            sc=sc,
            X_day=X[ds:de],
            X_sim_day=X_sim[ds:de],
            spot_day=spot_prices[ds:de],
            idx_map=idx_map,
            core_indices=core_indices,
            config=config,
        )
        day_cands: list[dict] = []
        for cand in cands:
            trade, status = _realize_trade(
                day=day, fold_idx=fold_spec.fold_idx,
                strategy_label="test_cand", paired_trade_id=-1,
                X_sim_day=X_sim[ds:de], spot_day=spot_prices[ds:de],
                idx_map=idx_map, sc=sc,
                entry_local=cand["local_i"], side=cand["side"],
                picked=cand["picked"], context_spread=cand["context_spread"],
            )
            if trade is None:
                continue
            full_cand = {
                "local_i": int(cand["local_i"]),
                "side": cand["side"],
                "features": np.asarray(cand["features"], dtype=np.float32),
                "trade": trade,
            }
            day_cands.append(full_cand)
            lookup[(day, int(cand["local_i"]), cand["side"])] = full_cand
        if day_cands:
            by_day[day] = day_cands
    return by_day, lookup


def build_fold_cache(
    *,
    fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
) -> FoldCache | None:
    t0 = time.time()
    config, _grid, _ntrig, _nent = select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    if config is None:
        print(f"  fold {fold_spec.fold_idx}: no viable gate config, skipping")
        return None
    t1 = time.time()
    train_feats, train_labels = _enumerate_train_samples(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    t2 = time.time()
    test_by_day, test_lookup = _enumerate_test_cache(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    t3 = time.time()
    n_test_days = len(test_by_day)
    n_test_cands = sum(len(v) for v in test_by_day.values())
    print(f"  fold {fold_spec.fold_idx}: gates {t1 - t0:.1f}s, "
          f"train_samples n={train_feats.shape[0]} in {t2 - t1:.1f}s, "
          f"test_cache days={n_test_days} cands={n_test_cands} in {t3 - t2:.1f}s")
    return FoldCache(
        fold_idx=fold_spec.fold_idx,
        window_id=fold_spec.window_id,
        test_days=list(fold_spec.test_days),
        gate_config=asdict(config),
        train_features=train_feats,
        train_labels=train_labels,
        test_candidates_by_day=test_by_day,
        test_trade_lookup=test_lookup,
    )


def save_caches(caches: list[FoldCache], path: str) -> None:
    serial = []
    for c in caches:
        serial.append({
            "fold_idx": c.fold_idx,
            "window_id": c.window_id,
            "test_days": c.test_days,
            "gate_config": c.gate_config,
            "train_features": c.train_features,
            "train_labels": c.train_labels,
            "test_candidates_by_day": {
                day: [_cand_to_serial(cand) for cand in cands]
                for day, cands in c.test_candidates_by_day.items()
            },
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(serial, f)


def load_caches(path: str) -> list[FoldCache]:
    with open(path, "rb") as f:
        serial = pickle.load(f)
    caches: list[FoldCache] = []
    for d in serial:
        test_by_day: dict[str, list[dict]] = {}
        lookup: dict[tuple[str, int, str], dict] = {}
        for day, cands in d["test_candidates_by_day"].items():
            rebuilt = [_cand_from_serial(c) for c in cands]
            test_by_day[day] = rebuilt
            for c in rebuilt:
                lookup[(day, c["local_i"], c["side"])] = c
        caches.append(FoldCache(
            fold_idx=d["fold_idx"], window_id=d["window_id"],
            test_days=d["test_days"], gate_config=d["gate_config"],
            train_features=d["train_features"], train_labels=d["train_labels"],
            test_candidates_by_day=test_by_day, test_trade_lookup=lookup,
        ))
    return caches


# ---------------------------------------------------------------------------
# Pipeline rerun against a cache (supports ablation / label permutation)
# ---------------------------------------------------------------------------

def _zero_columns(mat: np.ndarray, cols: tuple[int, ...]) -> np.ndarray:
    if not cols:
        return mat
    out = mat.copy()
    for c in cols:
        out[:, c] = 0.0
    return out


def run_one_pipeline_on_cache(
    cache: FoldCache,
    *,
    permute_seed: int | None = None,
    ablate_cols: tuple[int, ...] = (),
    control_rng_seed: int | None = None,
) -> dict:
    """Refit RF, pick argmax per day, simulate controls against the
    precomputed trade cache. Returns summaries for strategy + controls.
    """
    feats = cache.train_features
    labels = cache.train_labels
    if feats.shape[0] == 0:
        return {"strategy": summarize_trades([]), "control_A": summarize_trades([]),
                "control_B": summarize_trades([]), "strategy_trades": [],
                "control_A_trades": [], "control_B_trades": []}

    feats_mod = _zero_columns(feats, ablate_cols)
    labels_mod = labels.copy()
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        idx = np.arange(labels_mod.shape[0])
        rng.shuffle(idx)
        labels_mod = labels_mod[idx]

    fit_seed = cache.fold_idx * 10007 + 1
    model = train_scorer(feats_mod, labels_mod, seed=fit_seed)
    if model is None:
        return {"strategy": summarize_trades([]), "control_A": summarize_trades([]),
                "control_B": summarize_trades([]), "strategy_trades": [],
                "control_A_trades": [], "control_B_trades": []}

    strategy_trades: list[V2Trade] = []
    for day, cands in cache.test_candidates_by_day.items():
        feat_mat = np.vstack([_zero_columns(c["features"][None, :], ablate_cols)[0]
                              for c in cands]).astype(np.float32)
        preds = model.predict(feat_mat)
        pos = int(np.argmax(preds))
        chosen = cands[pos]
        t = chosen["trade"]
        # Re-label the trade as a strategy trade (preserve the rest)
        new_trade = V2Trade(
            **{**asdict(t), "strategy_label": "strategy",
               "predicted_net_pct": float(preds[pos]),
               "score_rank_in_day": 1,
               "n_candidates_in_day": len(cands)}
        )
        strategy_trades.append(new_trade)

    # Controls
    ctrl_rng = np.random.default_rng(
        control_rng_seed if control_rng_seed is not None else (cache.fold_idx * 10007 + 11)
    )
    ctrl_A_trades: list[V2Trade] = []
    ctrl_B_trades: list[V2Trade] = []
    available_days = [d for d in cache.test_days if d in cache.test_candidates_by_day]

    for pid, st in enumerate(strategy_trades):
        # Control A: same day, random bar in [BAR_LO, BAR_HI], same side
        bar_a = int(ctrl_rng.integers(BAR_LO, BAR_HI + 1))
        ca = cache.test_trade_lookup.get((st.date, bar_a, st.side))
        if ca is not None:
            t = ca["trade"]
            ctrl_A_trades.append(V2Trade(
                **{**asdict(t), "strategy_label": "control_A", "paired_trade_id": pid}
            ))
        # Control B: random other test day, same bar as strategy, same side
        pool = [d for d in available_days if d != st.date]
        if pool:
            pick_day = pool[int(ctrl_rng.integers(0, len(pool)))]
            cb = cache.test_trade_lookup.get((pick_day, st.bar_entry, st.side))
            if cb is not None:
                t = cb["trade"]
                ctrl_B_trades.append(V2Trade(
                    **{**asdict(t), "strategy_label": "control_B", "paired_trade_id": pid,
                       "date": pick_day}
                ))

    return {
        "strategy": summarize_trades(strategy_trades),
        "control_A": summarize_trades(ctrl_A_trades),
        "control_B": summarize_trades(ctrl_B_trades),
        "strategy_trades": strategy_trades,
        "control_A_trades": ctrl_A_trades,
        "control_B_trades": ctrl_B_trades,
    }


def _combine_fold_results(per_fold: list[dict]) -> dict:
    all_s = [t for r in per_fold for t in r["strategy_trades"]]
    all_a = [t for r in per_fold for t in r["control_A_trades"]]
    all_b = [t for r in per_fold for t in r["control_B_trades"]]
    s = summarize_trades(all_s)
    a = summarize_trades(all_a)
    b = summarize_trades(all_b)
    return {
        "strategy": s,
        "control_A": a,
        "control_B": b,
        "strategy_minus_A": s["mean_net_pct"] - a["mean_net_pct"],
        "strategy_minus_B": s["mean_net_pct"] - b["mean_net_pct"],
    }


# ---------------------------------------------------------------------------
# (1) Day-level block bootstrap on real V2 output
# ---------------------------------------------------------------------------

def run_real_v2_over_caches(caches: list[FoldCache]) -> list[dict]:
    """Recompute the real V2 (no modifications) from caches so we can
    bootstrap off identical trade lists.
    """
    out: list[dict] = []
    for c in caches:
        r = run_one_pipeline_on_cache(c)
        out.append(r)
    return out


def bootstrap_day_level(
    per_fold: list[dict],
    caches: list[FoldCache],
    n_boot: int = N_BOOTSTRAP,
    seed: int = BASE_SEED,
) -> dict:
    """Day-level block bootstrap. Resample each fold's test_days with
    replacement, reconstruct strategy+control aggregates from the
    matching trades.
    """
    rng = np.random.default_rng(seed)

    # Group trades by (fold_idx, date) for fast resampling
    fold_groups: list[dict[str, dict[str, list[V2Trade]]]] = []
    for fr, cache in zip(per_fold, caches):
        grp: dict[str, dict[str, list[V2Trade]]] = {}
        for t in fr["strategy_trades"]:
            grp.setdefault(t.date, {"strategy": [], "control_A": [], "control_B": []})["strategy"].append(t)
        for t in fr["control_A_trades"]:
            grp.setdefault(t.date, {"strategy": [], "control_A": [], "control_B": []})["control_A"].append(t)
        for t in fr["control_B_trades"]:
            # Control B lives on a *different* date than the paired strategy trade.
            # For day-level bootstrap we sample strategy days and then pull in the
            # Control B trade paired to the resampled strategy trades, not trades
            # keyed by Control B's own calendar date. Easier: key Control B by
            # its paired strategy trade's date.
            paired_date = fr["strategy_trades"][t.paired_trade_id].date if t.paired_trade_id >= 0 and t.paired_trade_id < len(fr["strategy_trades"]) else t.date
            grp.setdefault(paired_date, {"strategy": [], "control_A": [], "control_B": []})["control_B"].append(t)
        fold_groups.append(grp)

    metrics: dict[str, list[float]] = {
        "strategy_mean_net_pct": [], "strategy_dollar_pf": [],
        "control_A_mean_net_pct": [], "control_A_dollar_pf": [],
        "control_B_mean_net_pct": [],
        "gap_vs_A": [], "gap_vs_B": [],
    }
    for _rep in range(n_boot):
        all_s: list[V2Trade] = []
        all_a: list[V2Trade] = []
        all_b: list[V2Trade] = []
        for cache, grp in zip(caches, fold_groups):
            days = cache.test_days
            if not days:
                continue
            sampled = rng.choice(days, size=len(days), replace=True)
            for d in sampled:
                if d not in grp:
                    continue
                all_s.extend(grp[d]["strategy"])
                all_a.extend(grp[d]["control_A"])
                all_b.extend(grp[d]["control_B"])
        s = summarize_trades(all_s)
        a = summarize_trades(all_a)
        b = summarize_trades(all_b)
        metrics["strategy_mean_net_pct"].append(s["mean_net_pct"])
        metrics["strategy_dollar_pf"].append(s["dollar_pf"])
        metrics["control_A_mean_net_pct"].append(a["mean_net_pct"])
        metrics["control_A_dollar_pf"].append(a["dollar_pf"])
        metrics["control_B_mean_net_pct"].append(b["mean_net_pct"])
        metrics["gap_vs_A"].append(s["mean_net_pct"] - a["mean_net_pct"])
        metrics["gap_vs_B"].append(s["mean_net_pct"] - b["mean_net_pct"])

    summary: dict[str, dict[str, float]] = {}
    for k, v in metrics.items():
        arr = np.array(v, dtype=np.float64)
        summary[k] = {
            "n_boot": int(len(arr)),
            "p2_5": float(np.quantile(arr, 0.025)),
            "p50": float(np.quantile(arr, 0.50)),
            "p97_5": float(np.quantile(arr, 0.975)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "frac_gt_0": float((arr > 0).mean()),
        }
    return summary


# ---------------------------------------------------------------------------
# (2) Null permutation test
# ---------------------------------------------------------------------------

def run_null_permutations(
    caches: list[FoldCache],
    n_perms: int = N_PERMUTATIONS,
    base_seed: int = BASE_SEED,
) -> dict:
    reps: list[dict] = []
    t0 = time.time()
    for i in range(n_perms):
        per_fold: list[dict] = []
        for cache in caches:
            r = run_one_pipeline_on_cache(
                cache,
                permute_seed=base_seed + i * 997 + cache.fold_idx,
                control_rng_seed=base_seed + i * 131 + cache.fold_idx + 1,
            )
            per_fold.append(r)
        agg = _combine_fold_results(per_fold)
        reps.append({
            "strategy_mean_net_pct": agg["strategy"]["mean_net_pct"],
            "strategy_dollar_pf": agg["strategy"]["dollar_pf"],
            "control_A_mean_net_pct": agg["control_A"]["mean_net_pct"],
            "gap_vs_A": agg["strategy_minus_A"],
            "strategy_n": agg["strategy"]["n"],
            "control_A_n": agg["control_A"]["n"],
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_perms - i - 1)
            print(f"  null perm {i + 1}/{n_perms} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    return {"reps": reps, "elapsed_sec": time.time() - t0, "n_perms": n_perms}


def _null_summary(reps: list[dict], observed: dict) -> dict:
    def _pct(arr: np.ndarray, v: float) -> float:
        return float((arr >= v).mean())

    def _summarize(key: str) -> dict:
        arr = np.array([r[key] for r in reps], dtype=np.float64)
        obs = float(observed.get(key, 0.0))
        return {
            "null_mean": float(arr.mean()),
            "null_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "null_p2_5": float(np.quantile(arr, 0.025)),
            "null_p50": float(np.quantile(arr, 0.50)),
            "null_p97_5": float(np.quantile(arr, 0.975)),
            "observed": obs,
            "p_value_one_sided_ge": _pct(arr, obs),
        }

    return {
        "strategy_mean_net_pct": _summarize("strategy_mean_net_pct"),
        "gap_vs_A": _summarize("gap_vs_A"),
        "strategy_dollar_pf": _summarize("strategy_dollar_pf"),
    }


# ---------------------------------------------------------------------------
# (3) Feature ablations
# ---------------------------------------------------------------------------

ABLATION_PLAN: list[tuple[str, tuple[str, ...]]] = [
    ("baseline_full", ()),
    ("ablate_opening_gap_pct", ("opening_gap_pct",)),
    ("ablate_first15_close_position", ("first15_close_position",)),
    ("ablate_first15_acceptance", ("first15_acceptance",)),
    ("ablate_gap_and_both_first15", ("opening_gap_pct", "first15_close_position", "first15_acceptance")),
    ("ablate_vwap_reclaim_state", ("vwap_reclaim_state",)),
    ("ablate_bar_delta", ("bar_delta",)),
]


def _feature_name_to_col(name: str) -> int:
    if name == "side_indicator":
        return SIDE_INDICATOR_IDX
    return CORE_FEATURE_NAMES.index(name)


def run_ablations(caches: list[FoldCache]) -> list[dict]:
    results: list[dict] = []
    for name, names in ABLATION_PLAN:
        cols = tuple(_feature_name_to_col(n) for n in names)
        t0 = time.time()
        per_fold = [run_one_pipeline_on_cache(c, ablate_cols=cols) for c in caches]
        agg = _combine_fold_results(per_fold)
        elapsed = time.time() - t0
        per_fold_strat = [{
            "fold_idx": c.fold_idx,
            "strategy_mean_net_pct": r["strategy"]["mean_net_pct"],
            "strategy_dollar_pf": r["strategy"]["dollar_pf"],
            "control_A_mean_net_pct": r["control_A"]["mean_net_pct"],
        } for c, r in zip(caches, per_fold)]
        results.append({
            "name": name,
            "zeroed_features": list(names),
            "strategy_mean_net_pct": agg["strategy"]["mean_net_pct"],
            "strategy_dollar_pf": agg["strategy"]["dollar_pf"],
            "strategy_target_hit_frac": agg["strategy"]["target_hit_frac"],
            "strategy_stop_hit_frac": agg["strategy"]["stop_hit_frac"],
            "strategy_n": agg["strategy"]["n"],
            "control_A_mean_net_pct": agg["control_A"]["mean_net_pct"],
            "control_A_n": agg["control_A"]["n"],
            "gap_vs_A": agg["strategy_minus_A"],
            "per_fold": per_fold_strat,
            "elapsed_sec": elapsed,
        })
        print(f"  ablation '{name}': "
              f"strategy mean={agg['strategy']['mean_net_pct']:+.5f}, "
              f"gap_vs_A={agg['strategy_minus_A']:+.5f}, "
              f"elapsed={elapsed:.1f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--cache-path", default="",
                    help="If non-empty, load (or save) FoldCache to this path")
    ap.add_argument("--force-rebuild-cache", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(args.out_dir, CACHE_FILENAME)

    data = load_data(args.data)
    dates = list(data["dates"])
    feature_names = list(data["feature_names"])
    idx_map = feature_index_map(feature_names)
    required = list(CORE_FEATURE_NAMES) + ["vix_regime", "iv_percentile", "vrp"]
    missing = [n for n in required if n not in idx_map]
    if missing:
        print(f"FATAL: feature_names missing: {missing}", file=sys.stderr)
        return 2
    core_idx = core_feature_indices(idx_map)

    X = data["X"].numpy() if isinstance(data["X"], torch.Tensor) else np.asarray(data["X"])
    X_sim = data["X_sim"].numpy() if isinstance(data["X_sim"], torch.Tensor) else np.asarray(data["X_sim"])
    spot_prices = data["spot_prices"].numpy() if isinstance(data["spot_prices"], torch.Tensor) else np.asarray(data["spot_prices"])

    day_to_range, _ = day_ranges(dates)
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dataset_fp = str(data["metadata"].get("fingerprint", "unknown"))

    folds = build_folds(dates)

    # --- Build or load caches ---
    caches: list[FoldCache]
    if os.path.exists(cache_path) and not args.force_rebuild_cache:
        print(f"Loading FoldCache from {cache_path}")
        caches = load_caches(cache_path)
    else:
        print("Building fold caches (gates + train features/labels + test candidates)")
        t0 = time.time()
        caches = []
        for fs in folds:
            c = build_fold_cache(
                fold_spec=fs, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
                X=X, X_sim=X_sim, spot_prices=spot_prices,
                idx_map=idx_map, core_indices=core_idx,
            )
            if c is not None:
                caches.append(c)
        print(f"Cache build elapsed: {time.time() - t0:.1f}s")
        save_caches(caches, cache_path)
        print(f"Saved caches to {cache_path}")

    # --- Real V2 re-run (for bootstrap baseline + observed metrics) ---
    print("\nReal V2 run from caches (sanity check)")
    per_fold_real = run_real_v2_over_caches(caches)
    observed = _combine_fold_results(per_fold_real)
    observed_flat = {
        "strategy_mean_net_pct": observed["strategy"]["mean_net_pct"],
        "strategy_dollar_pf": observed["strategy"]["dollar_pf"],
        "strategy_n": observed["strategy"]["n"],
        "control_A_mean_net_pct": observed["control_A"]["mean_net_pct"],
        "gap_vs_A": observed["strategy_minus_A"],
        "control_B_mean_net_pct": observed["control_B"]["mean_net_pct"],
        "gap_vs_B": observed["strategy_minus_B"],
    }
    print(f"  observed: strategy_mean={observed_flat['strategy_mean_net_pct']:+.5f}, "
          f"gap_vs_A={observed_flat['gap_vs_A']:+.5f}, "
          f"gap_vs_B={observed_flat['gap_vs_B']:+.5f}")

    # --- Bootstrap ---
    print("\nDay-level block bootstrap")
    boot = bootstrap_day_level(per_fold_real, caches, n_boot=args.n_boot, seed=BASE_SEED)
    print(f"  strategy_mean_net_pct: p50={boot['strategy_mean_net_pct']['p50']:+.5f} "
          f"95%CI=[{boot['strategy_mean_net_pct']['p2_5']:+.5f}, "
          f"{boot['strategy_mean_net_pct']['p97_5']:+.5f}] "
          f"frac>0={boot['strategy_mean_net_pct']['frac_gt_0']:.3f}")
    print(f"  gap_vs_A:            p50={boot['gap_vs_A']['p50']:+.5f} "
          f"95%CI=[{boot['gap_vs_A']['p2_5']:+.5f}, {boot['gap_vs_A']['p97_5']:+.5f}] "
          f"frac>0={boot['gap_vs_A']['frac_gt_0']:.3f}")
    print(f"  gap_vs_B:            p50={boot['gap_vs_B']['p50']:+.5f} "
          f"95%CI=[{boot['gap_vs_B']['p2_5']:+.5f}, {boot['gap_vs_B']['p97_5']:+.5f}] "
          f"frac>0={boot['gap_vs_B']['frac_gt_0']:.3f}")

    # --- Null permutation ---
    print(f"\nNull permutation test ({args.n_perms} perms)")
    null_reps = run_null_permutations(caches, n_perms=args.n_perms, base_seed=BASE_SEED)
    null_summary = _null_summary(null_reps["reps"], observed_flat)
    for key, s in null_summary.items():
        obs = s["observed"]
        print(f"  {key}: observed={obs:+.5f} null_p50={s['null_p50']:+.5f} "
              f"null_95%CI=[{s['null_p2_5']:+.5f}, {s['null_p97_5']:+.5f}] "
              f"p_one_sided={s['p_value_one_sided_ge']:.3f}")

    # --- Feature ablations ---
    print("\nFeature ablations")
    abl = run_ablations(caches)

    # --- Write outputs ---
    write_json(os.path.join(args.out_dir, "stress_test_summary.json"), {
        "dataset_fingerprint": dataset_fp,
        "observed": observed_flat,
        "bootstrap": boot,
        "null_permutation_summary": null_summary,
        "null_permutation_reps": null_reps["reps"],
        "null_permutation_elapsed_sec": null_reps["elapsed_sec"],
        "ablations": abl,
    })
    print(f"\nWrote stress_test_summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
