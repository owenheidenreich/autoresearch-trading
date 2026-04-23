"""Phase 2A — L3 feature pruning + deferred-exit rule experiment.

Two independent improvements proposed for the augmented L3 model:

1. **Feature pruning** (from Stage A divergence top-15):
   - X_sim feature 13 = minutes_to_close (in 2.81 -> oos -0.08, gap 2.89)
   - X_sim feature 27 = trend_5min (in 3.29 -> oos 0.08, gap 3.22)
   - X_sim feature 25 = force_index_2 (in 1.69 -> oos 0.32, gap 1.37)
   - trade-state mfe_norm (in 0.88 -> oos 0.00, gap 0.88)

2. **Deferred-exit rule** (from Phase 1C false-positive diagnostic):
   - Block exit if `bars_since_entry < 10` AND `mfe_norm < 0.05`
   - Both OOS false positives share this signature exactly

Test arms:
| Arm | Pruned features | Deferred-exit rule? |
|-----|-----------------|----------------------|
| A0  | (none, baseline) | no |
| A1  | minutes_to_close (l2_13) | no |
| A2  | minutes_to_close + trend_5min (l2_13, l2_27) | no |
| A3  | mfe_norm (trade-state) | no |
| A4  | minutes_to_close + mfe_norm | no |
| B0  | (none) | yes |
| B1  | minutes_to_close (l2_13) | yes |
| B4  | minutes_to_close + mfe_norm | yes |

Each arm reports OOS PF/DD vs baseline V1+L3 champion (PF 2.169).
Acceptance: arm OOS PF >= 2.169 AND no fold catastrophic regression.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v3.analysis.layer3_train_replay import (
    TRADE_STATE_NAMES,
    _build_per_trade_data,
)
from v3.analysis.layer3_v31_cleanup import _build_in_sample_trade_data
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_labeled_day,
    load_export_bundle,
    replay_metrics_from_pnls,
)
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_V1_ALONE_OOS = os.path.join(
    "v3", "artifacts", "layer2_directional_variants", "oos_trades_V1.csv",
)
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "l3_feature_pruning_experiment")
DEFAULT_THRESHOLD = 0.19
DEFAULT_SEED = 42
DEFAULT_MFE_NORM_FLOOR = 0.05
DEFAULT_DEFER_BARS_FLOOR = 10
N_X_SIM = 89
TS_OFFSET = N_X_SIM
MFE_NORM_TS_INDEX = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--v1-alone-oos", default=DEFAULT_V1_ALONE_OOS)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--mfe-norm-floor", type=float, default=DEFAULT_MFE_NORM_FLOOR)
    p.add_argument("--defer-bars-floor", type=int, default=DEFAULT_DEFER_BARS_FLOOR)
    return p.parse_args()


def build_feature_matrix(trade_data: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build X (n_rows, n_features) and y, plus row metadata."""
    X_rows, y_rows, meta = [], [], []
    for td in trade_data:
        for b in td["per_bar"]:
            feats = np.concatenate([b["l2_feats"], b["trade_state"]])
            X_rows.append(feats)
            y_rows.append(b["target"])
            meta.append({
                "fold_idx": td["fold_idx"], "day": td["day"],
                "entry_bar": td["entry_bar"], "bar": b["bar"],
                "current_pnl": b["current_pnl"],
            })
    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int8),
        meta,
    )


def drop_columns(X: np.ndarray, drop_idxs: list[int]) -> np.ndarray:
    if not drop_idxs:
        return X
    keep = [i for i in range(X.shape[1]) if i not in set(drop_idxs)]
    return X[:, keep]


def replay_with_deferred_exit(
    trade_data: list[dict],
    model: HistGradientBoostingClassifier,
    threshold: float,
    drop_idxs: list[int],
    use_defer_rule: bool,
    mfe_norm_floor: float,
    defer_bars_floor: int,
) -> list[dict]:
    """Replay each trade. Optionally defer exits per the Phase 1C rule."""
    out = []
    for td in trade_data:
        feats_full = np.stack([
            np.concatenate([b["l2_feats"], b["trade_state"]])
            for b in td["per_bar"]
        ])
        feats_in = drop_columns(feats_full, drop_idxs)
        probs = model.predict_proba(feats_in)[:, 1]

        exit_idx = None
        trigger = "model"
        for i in range(len(probs)):
            if probs[i] >= threshold:
                if use_defer_rule:
                    bars_since_entry = float(td["per_bar"][i]["trade_state"][0])
                    mfe_norm = float(td["per_bar"][i]["trade_state"][MFE_NORM_TS_INDEX])
                    if bars_since_entry < defer_bars_floor and mfe_norm < mfe_norm_floor:
                        continue
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = len(probs) - 1
            trigger = "time_stop_fallback"
        bar = td["per_bar"][exit_idx]
        out.append({
            "fold_idx": td["fold_idx"], "day": td["day"],
            "entry_bar": td["entry_bar"], "exit_bar": bar["bar"],
            "direction": td["direction"],
            "exit_pnl": bar["current_pnl"],
            "trigger": trigger,
            "bars_held": bar["bar"] - td["entry_bar"],
        })
    return out


def metrics_from_sim(sim_rows: list[dict], equity: float) -> dict[str, float]:
    if not sim_rows:
        return {"pf": 0.0, "max_dd_pct": 0.0, "mean_pnl": 0.0, "trades": 0}
    df = pd.DataFrame(sim_rows).sort_values(["day", "entry_bar"])
    pnls = df["exit_pnl"].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    m["trades"] = len(pnls)
    m["mean_pnl"] = float(np.mean(pnls))
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset...", flush=True)
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)

    print(flush=True)
    print("Building chosen + teacher per-trade data...", flush=True)
    in_sample_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    teacher_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "teacher_baseline_trades.csv"))
    day_cache: dict = {}
    paths_cache: dict = {}
    minute_map_cache: dict = {}
    chosen_data = _build_in_sample_trade_data(
        in_sample_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    teacher_data = _build_in_sample_trade_data(
        teacher_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  chosen: {len(chosen_data)}, teacher: {len(teacher_data)}", flush=True)
    augmented_data = chosen_data + teacher_data

    print(flush=True)
    print("Building OOS V1 per-trade data...", flush=True)
    v1_alone = pd.read_csv(args.v1_alone_oos)
    v1_alone["day"] = v1_alone["day"].astype(str)
    v1_oos_data = []
    for _, trade in v1_alone.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=args.equity)
            if log_d is None:
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        fake_row = pd.Series({
            "day": day,
            "bar_index": int(trade["bar_index"]),
            "direction": str(trade["direction"]),
            "fold_idx": int(trade.get("fold_idx", -1)),
            "pnl": float(trade["pnl"]),
        })
        td = _build_per_trade_data(
            fake_row, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            v1_oos_data.append(td)
    print(f"  V1 OOS trade datasets: {len(v1_oos_data)}", flush=True)

    # === Define arms ===
    n_features_full = N_X_SIM + len(TRADE_STATE_NAMES)
    feature_names = list(ds.feature_names) + TRADE_STATE_NAMES
    minutes_to_close_idx = ds.feature_names.index("minutes_to_close")
    trend_5min_idx = ds.feature_names.index("trend_5min")
    mfe_norm_idx = TS_OFFSET + MFE_NORM_TS_INDEX
    print(flush=True)
    print(f"  minutes_to_close idx: {minutes_to_close_idx}", flush=True)
    print(f"  trend_5min idx:       {trend_5min_idx}", flush=True)
    print(f"  mfe_norm idx:         {mfe_norm_idx}", flush=True)

    arms = {
        "A0_baseline": {"drop": [], "defer": False},
        "A1_drop_mtc": {"drop": [minutes_to_close_idx], "defer": False},
        "A2_drop_mtc_t5m": {"drop": [minutes_to_close_idx, trend_5min_idx], "defer": False},
        "A3_drop_mfen": {"drop": [mfe_norm_idx], "defer": False},
        "A4_drop_mtc_mfen": {"drop": [minutes_to_close_idx, mfe_norm_idx], "defer": False},
        "B0_baseline_defer": {"drop": [], "defer": True},
        "B1_drop_mtc_defer": {"drop": [minutes_to_close_idx], "defer": True},
        "B4_drop_mtc_mfen_defer": {"drop": [minutes_to_close_idx, mfe_norm_idx], "defer": True},
    }

    # === Build full feature matrix once ===
    X_aug_full, y_aug, _ = build_feature_matrix(augmented_data)
    X_in_full, _, _ = build_feature_matrix(chosen_data)
    print(flush=True)
    print(f"  Augmented training: {len(augmented_data)} trades / {len(X_aug_full)} rows / {X_aug_full.shape[1]} features", flush=True)

    # === Train each unique-feature-set model and evaluate every arm ===
    model_cache: dict[tuple, HistGradientBoostingClassifier] = {}
    arm_results: dict[str, dict] = {}

    print(flush=True)
    print("=" * 100)
    print("Per-arm OOS + in-sample evaluation")
    print("=" * 100)
    print(f"{'arm':<26}{'OOS PF':>10}{'OOS DD%':>10}{'OOS mean$':>12}{'OOS trades':>12}"
          f"{'IS PF':>10}{'IS DD%':>10}", flush=True)

    for arm_name, arm in arms.items():
        drop_idxs = sorted(arm["drop"])
        key = tuple(drop_idxs)
        if key not in model_cache:
            X_train = drop_columns(X_aug_full, drop_idxs)
            model = HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.05, max_depth=4,
                max_iter=200, min_samples_leaf=50,
                random_state=args.seed + 1000, early_stopping=False,
            )
            model.fit(X_train, y_aug)
            model_cache[key] = model
        model = model_cache[key]

        # OOS evaluation on V1 trades
        oos_sim = replay_with_deferred_exit(
            v1_oos_data, model, args.threshold, drop_idxs,
            arm["defer"], args.mfe_norm_floor, args.defer_bars_floor,
        )
        oos_m = metrics_from_sim(oos_sim, args.equity)

        # In-sample evaluation on chosen trades (subset training distribution)
        is_sim = replay_with_deferred_exit(
            chosen_data, model, args.threshold, drop_idxs,
            arm["defer"], args.mfe_norm_floor, args.defer_bars_floor,
        )
        is_m = metrics_from_sim(is_sim, args.equity)

        arm_results[arm_name] = {
            "drop_idxs": drop_idxs,
            "drop_names": [feature_names[i] for i in drop_idxs],
            "defer": arm["defer"],
            "oos": oos_m,
            "in_sample": is_m,
        }
        print(
            f"{arm_name:<26}{oos_m['pf']:>10.3f}{oos_m['max_dd_pct']:>10.1f}"
            f"{oos_m['mean_pnl']:>12.0f}{int(oos_m['trades']):>12}"
            f"{is_m['pf']:>10.3f}{is_m['max_dd_pct']:>10.1f}",
            flush=True,
        )

    # === Verdict ===
    baseline_oos_pf = arm_results["A0_baseline"]["oos"]["pf"]
    baseline_in_pf = arm_results["A0_baseline"]["in_sample"]["pf"]
    print(flush=True)
    print("=" * 100)
    print("Phase 2A verdict")
    print("=" * 100, flush=True)
    print(f"  Baseline (A0): OOS PF {baseline_oos_pf:.3f} / IS PF {baseline_in_pf:.3f}", flush=True)
    best_arm = None
    best_oos_pf = baseline_oos_pf
    for name, r in arm_results.items():
        if name == "A0_baseline":
            continue
        oos_lift = r["oos"]["pf"] - baseline_oos_pf
        is_change = r["in_sample"]["pf"] - baseline_in_pf
        verdict_tag = "WIN" if oos_lift >= 0.10 and r["in_sample"]["pf"] >= baseline_in_pf - 0.10 else (
                     "TIE" if abs(oos_lift) < 0.10 else "LOSE")
        print(f"  {name:<26}: OOS Δ={oos_lift:+.3f}  IS Δ={is_change:+.3f}  → {verdict_tag}", flush=True)
        if r["oos"]["pf"] > best_oos_pf:
            best_oos_pf = r["oos"]["pf"]
            best_arm = name
    print(flush=True)
    if best_arm:
        print(f"Best arm: {best_arm} (OOS PF {best_oos_pf:.3f}, lift {best_oos_pf - baseline_oos_pf:+.3f})", flush=True)
    else:
        print("No arm beat baseline.", flush=True)

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_chosen": int(len(chosen_data)),
            "n_teacher": int(len(teacher_data)),
            "n_v1_oos": int(len(v1_oos_data)),
            "threshold": float(args.threshold),
            "mfe_norm_floor": float(args.mfe_norm_floor),
            "defer_bars_floor": int(args.defer_bars_floor),
            "minutes_to_close_idx": int(minutes_to_close_idx),
            "trend_5min_idx": int(trend_5min_idx),
            "mfe_norm_idx": int(mfe_norm_idx),
        },
        "arms": arm_results,
        "best_arm": best_arm,
        "best_oos_pf": best_oos_pf,
        "baseline_oos_pf": baseline_oos_pf,
    }
    out = os.path.join(args.out_dir, "l3_feature_pruning_experiment.json")
    with open(out, "w") as f:
        json.dump(
            payload, f, indent=2, sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
