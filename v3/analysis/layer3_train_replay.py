"""Stage 3 of the Layer 3 (learned exits) workstream.

Builds a HistGradientBoostingClassifier that, at each post-entry bar,
predicts P(exit_now). For each chosen trade, walk bars from entry+1
to session_end; exit at the first bar where P(exit_now) >= threshold,
otherwise fall back to time-stop. Compose with Layer-2's frozen
entries; report PF/DD/per-fold vs the corrected baseline (1.472) and
the heuristic ceiling (time_of_day_90 PF 1.709).

Acceptance gate (per Stage 2 verdict):
- PF >= 1.709 AND no fold below 0.808 (don't break fold 0 worse than
  the heuristic), OR
- DD <= 30.6% at PF >= corrected baseline (1.472)

Walk-forward semantics:
- Fold k Layer-3 trains on chosen trades from days in folds 0..k-1
  (chronologically prior).
- Fold 0 has zero prior chosen trades -> falls back to time_of_day_90
  heuristic, documented as a known limitation.
- Threshold = 0.5 default; per-fold calibration is a Stage 3 ablation
  if the default doesn't hit acceptance.

Run:
    python -m v3.analysis.layer3_train_replay \
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
from sklearn.ensemble import HistGradientBoostingClassifier

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
)
from v3.oracles.opportunity import (
    _build_contract_paths,
    _contract_idx_for_record,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_learned_v3_0")

TRADE_STATE_NAMES = [
    "bars_since_entry",
    "bars_to_session_end",
    "current_pnl_norm",
    "mfe_norm",
    "mae_norm",
    "mfe_bar_age",
    "direction_is_call",
]

# Stage 2 anchor results
HEURISTIC_BASELINE_PF = 1.709           # time_of_day_90
HEURISTIC_BASELINE_DD = 31.3
HEURISTIC_BASELINE_FOLD0 = 0.808        # time_of_day_90 fold 0
CORRECTED_BASELINE_PF = 1.472
CORRECTED_BASELINE_DD = 35.6
CORRECTED_BASELINE_FOLD0 = 0.875


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer 3 learned exits (Stage 3).")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--session-end-bar", type=int, default=DEFAULT_SESSION_END_BAR)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_CONTRACT)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Exit if P(exit_now) >= threshold.")
    p.add_argument("--fold0-fallback", default="time_of_day_90",
                   choices=["time_of_day_90", "corrected_time_stop"],
                   help="Exit policy for fold 0 (no prior training data).")
    p.add_argument("--fold0-fallback-bars", type=int, default=90,
                   help="If fallback is time_of_day, exit at entry+K.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_per_trade_data(
    trade: pd.Series,
    log,
    sidecar: dict,
    paths_cache: dict,
    minute_map_cache: dict,
    ds: V2Dataset,
    session_end_bar: int,
    commission: float,
) -> dict[str, Any] | None:
    """For one trade, build per-bar (current_pnl, layer2_features, target)."""
    bar_index = int(trade["bar_index"])
    direction = str(trade["direction"])
    day = str(trade["day"])
    bar = next((b for b in log.bars if b.bar_index == bar_index), None)
    if bar is None:
        return None
    sel = select_contract(bar.contracts, direction, "layer2")
    if sel is None:
        return None
    right = "P" if direction == "put" else "C"
    match = next(
        (c for c in bar.contracts if c.strike == sel.strike and c.right == right),
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

    if day not in minute_map_cache:
        start, end = ds.day_bar_range(day)
        minute_map_cache[day] = {int(ds.bar_of_day[i]): i for i in range(start, end)}
    minute_map = minute_map_cache[day]

    entry_mid = float(match.mid)
    entry_sf = float(match.spread_fraction)
    entry_premium_dollars = entry_mid * 100.0
    entry_ask = entry_mid * (1.0 + entry_sf / 2.0)

    end_bar = min(session_end_bar, len(path.mids) - 1)
    per_bar: list[dict[str, Any]] = []
    for t in range(bar_index + 1, end_bar + 1):
        m = path.mids[t]
        if not np.isfinite(m):
            continue
        sf = path.spread_fracs[t]
        if not np.isfinite(sf):
            sf = entry_sf
        exit_bid = float(m) * (1.0 - float(sf) / 2.0)
        current_pnl = 100.0 * (exit_bid - entry_ask) - commission
        abs_idx = minute_map.get(t)
        if abs_idx is None:
            continue
        l2_feats = ds.X_sim[abs_idx].astype(np.float32)
        per_bar.append({"bar": t, "current_pnl": current_pnl, "l2_feats": l2_feats})

    if len(per_bar) < 2:
        return None

    # Suffix max for target: target_i = 1 iff current_pnl_i >= max(current_pnl_{i+1..end})
    n = len(per_bar)
    suffix_max = np.full(n, -np.inf, dtype=np.float64)
    if n >= 2:
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(per_bar[i + 1]["current_pnl"], suffix_max[i + 1])
    suffix_max[n - 1] = -np.inf  # last bar trivially exits

    # Trade-state features running
    mfe = -np.inf
    mae = np.inf
    mfe_bar = bar_index
    for i, b in enumerate(per_bar):
        cp = b["current_pnl"]
        t = b["bar"]
        if cp > mfe:
            mfe = cp
            mfe_bar = t
        if cp < mae:
            mae = cp
        denom = max(abs(entry_premium_dollars), 1e-9)
        trade_state = np.array([
            t - bar_index,                # bars_since_entry
            end_bar - t,                  # bars_to_session_end
            cp / denom,                   # current_pnl_norm
            mfe / denom,                  # mfe_norm
            mae / denom,                  # mae_norm
            t - mfe_bar,                  # mfe_bar_age
            1.0 if direction == "call" else 0.0,  # direction_is_call
        ], dtype=np.float32)
        target = int(cp >= suffix_max[i]) if np.isfinite(suffix_max[i]) else 1
        b["trade_state"] = trade_state
        b["target"] = target

    return {
        "fold_idx": int(trade["fold_idx"]),
        "day": day,
        "entry_bar": bar_index,
        "direction": direction,
        "entry_premium_dollars": entry_premium_dollars,
        "per_bar": per_bar,
        "csv_pnl": float(trade["pnl"]),
    }


def _flatten_to_rows(trade_data: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X_rows = []
    y_rows = []
    meta_rows = []
    for td in trade_data:
        for b in td["per_bar"]:
            feats = np.concatenate([b["l2_feats"], b["trade_state"]])
            X_rows.append(feats)
            y_rows.append(b["target"])
            meta_rows.append({
                "fold_idx": td["fold_idx"], "day": td["day"],
                "entry_bar": td["entry_bar"], "bar": b["bar"],
                "current_pnl": b["current_pnl"],
            })
    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int8),
        meta_rows,
    )


def _replay_with_model(
    trade_data: list[dict],
    model: HistGradientBoostingClassifier,
    threshold: float,
) -> list[dict]:
    """For each trade, walk bars and exit at first P(exit_now) >= threshold."""
    out = []
    for td in trade_data:
        exit_pnl = None
        exit_bar = None
        trigger = "model"
        # Build all per-bar features for this trade in one batch
        feats = np.stack([np.concatenate([b["l2_feats"], b["trade_state"]]) for b in td["per_bar"]])
        probs = model.predict_proba(feats)[:, 1]
        for i, b in enumerate(td["per_bar"]):
            if probs[i] >= threshold:
                exit_pnl = b["current_pnl"]
                exit_bar = b["bar"]
                break
        if exit_pnl is None:
            # Never triggered — fall back to last bar (time-stop)
            last = td["per_bar"][-1]
            exit_pnl = last["current_pnl"]
            exit_bar = last["bar"]
            trigger = "time_stop_fallback"
        out.append({
            "fold_idx": td["fold_idx"], "day": td["day"],
            "entry_bar": td["entry_bar"], "exit_bar": exit_bar,
            "direction": td["direction"], "exit_pnl": exit_pnl,
            "trigger": trigger,
            "bars_held": exit_bar - td["entry_bar"],
            "max_prob": float(np.max(probs)) if len(probs) else float("nan"),
        })
    return out


def _replay_fold0_fallback(trade_data: list[dict], fallback: str, fallback_bars: int) -> list[dict]:
    """Fold 0 fallback: time_of_day_90 or corrected_time_stop."""
    out = []
    for td in trade_data:
        exit_pnl = None
        exit_bar = None
        trigger = fallback
        for b in td["per_bar"]:
            if fallback == "time_of_day_90" and (b["bar"] - td["entry_bar"]) >= fallback_bars:
                exit_pnl = b["current_pnl"]
                exit_bar = b["bar"]
                break
        if exit_pnl is None:
            last = td["per_bar"][-1]
            exit_pnl = last["current_pnl"]
            exit_bar = last["bar"]
            trigger = "time_stop_fallback"
        out.append({
            "fold_idx": td["fold_idx"], "day": td["day"],
            "entry_bar": td["entry_bar"], "exit_bar": exit_bar,
            "direction": td["direction"], "exit_pnl": exit_pnl,
            "trigger": trigger, "bars_held": exit_bar - td["entry_bar"],
        })
    return out


def _agg(trades_df: pd.DataFrame, equity: float, total_days: int) -> dict[str, float]:
    if trades_df.empty:
        return {"pf": 0.0, "max_dd_pct": 0.0, "mean_pnl": 0.0, "trades": 0.0}
    sorted_df = trades_df.sort_values(["day", "entry_bar"])
    pnls = sorted_df["exit_pnl"].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    m["trades"] = float(len(pnls))
    m["mean_pnl"] = float(np.mean(pnls))
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    print(f"Loaded {len(trades)} trades")
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds_meta = list(bundle["meta"]["folds"])
    fold_test_days = {int(f["fold_idx"]): set(f["test_days"]) for f in folds_meta}

    print("Loading V2Dataset + per-day labeling (slow part, ~3-5 min)...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    # --- Build per-trade data ---
    day_cache: dict[str, tuple] = {}
    paths_cache: dict[int, dict] = {}
    minute_map_cache: dict[str, dict] = {}
    trade_data: list[dict] = []
    skipped = 0
    for i, trade in trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            day_cache[day] = build_labeled_day(ds, day, cfg, equity=args.equity)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            skipped += 1
            continue
        td = _build_per_trade_data(
            trade, log, sidecar, paths_cache, minute_map_cache, ds,
            args.session_end_bar, args.commission,
        )
        if td is None:
            skipped += 1
            continue
        trade_data.append(td)
        if (i + 1) % 50 == 0:
            print(f"  built {i+1}/{len(trades)} trades...")
    print(f"Built {len(trade_data)} trade datasets; {skipped} skipped")

    # --- Per-fold training and replay ---
    print()
    print("=" * 100)
    print(f"Per-fold training (threshold={args.threshold:.2f}, fold0_fallback={args.fold0_fallback})")
    print("=" * 100)
    fold_results: dict[int, dict] = {}
    all_simulated = []
    feature_names = list(bundle["meta"]["feature_names"])

    for fold_idx in sorted(set(td["fold_idx"] for td in trade_data)):
        train_data = [td for td in trade_data if td["fold_idx"] < fold_idx]
        test_data = [td for td in trade_data if td["fold_idx"] == fold_idx]

        if len(train_data) == 0:
            # Fold 0 — use fallback
            sim_rows = _replay_fold0_fallback(test_data, args.fold0_fallback, args.fold0_fallback_bars)
            sim_df = pd.DataFrame(sim_rows)
            n_test = len(test_data)
            print(f"  fold {fold_idx}: fallback={args.fold0_fallback} (no prior training data); n_test={n_test}")
        else:
            X_train, y_train, _ = _flatten_to_rows(train_data)
            print(f"  fold {fold_idx}: training on {len(train_data)} trades / {len(X_train)} bar-rows; pos_rate={y_train.mean():.3f}")
            model = HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.05, max_depth=4,
                max_iter=200, min_samples_leaf=50,
                random_state=args.seed + fold_idx, early_stopping=False,
            )
            model.fit(X_train, y_train)
            sim_rows = _replay_with_model(test_data, model, args.threshold)
            sim_df = pd.DataFrame(sim_rows)
            print(f"    test n={len(test_data)}; early_exit_share={float((sim_df['trigger']=='model').mean()):.2%}; "
                  f"mean bars held={sim_df['bars_held'].mean():.1f}")

        fold_test_n_days = len(fold_test_days.get(int(fold_idx), set()))
        m = _agg(sim_df, args.equity, fold_test_n_days)
        fold_results[int(fold_idx)] = m
        all_simulated.append(sim_df)

    # --- Aggregate ---
    full_df = pd.concat(all_simulated, ignore_index=True) if all_simulated else pd.DataFrame()
    total_days = sum(len(d) for d in fold_test_days.values())
    overall = _agg(full_df, args.equity, total_days)

    print()
    print("=" * 100)
    print("Aggregate results")
    print("=" * 100)
    print(f"  Layer-3 composed:  PF={overall['pf']:.3f}  DD={overall['max_dd_pct']:.1f}%  mean=${overall['mean_pnl']:.0f}  trades={int(overall['trades'])}")
    print(f"  Corrected baseline: PF={CORRECTED_BASELINE_PF:.3f}  DD={CORRECTED_BASELINE_DD:.1f}%  (Layer-2 alone, time-stop)")
    print(f"  Heuristic ceiling:  PF={HEURISTIC_BASELINE_PF:.3f}  DD={HEURISTIC_BASELINE_DD:.1f}%  (time_of_day_90)")

    print()
    print("Per-fold")
    print(f"{'fold':<6}{'PF':>10}{'DD%':>10}{'mean$':>12}{'trades':>10}")
    for fi in sorted(fold_results):
        r = fold_results[fi]
        print(f"{fi:<6}{r['pf']:>10.3f}{r['max_dd_pct']:>10.1f}{r['mean_pnl']:>12.1f}{int(r['trades']):>10}")

    # --- Verdict ---
    pf = overall["pf"]
    dd = overall["max_dd_pct"]
    fold_pfs = {fi: fold_results[fi]["pf"] for fi in fold_results}
    min_fold_pf = float(min(fold_pfs.values())) if fold_pfs else float("nan")
    min_fold_idx = int(min(fold_pfs, key=fold_pfs.get)) if fold_pfs else -1
    fold_floor = HEURISTIC_BASELINE_FOLD0 - 0.01  # 0.01 PF tolerance (vs 0.808)

    print()
    print("=" * 100)
    print("Verdict (Stage 3)")
    print("=" * 100)
    pf_pass = pf >= HEURISTIC_BASELINE_PF
    dd_pass = dd <= 30.6 and pf >= CORRECTED_BASELINE_PF
    fold_ok = min_fold_pf >= fold_floor
    if pf_pass and fold_ok:
        verdict = (f"PASS-PF -- PF {pf:.3f} >= {HEURISTIC_BASELINE_PF}; "
                   f"min fold PF {min_fold_pf:.3f} (fold {min_fold_idx}) >= {fold_floor:.3f}")
    elif dd_pass and fold_ok:
        verdict = (f"PASS-DD -- DD {dd:.1f}% <= 30.6 at PF {pf:.3f} >= {CORRECTED_BASELINE_PF}; "
                   f"min fold PF {min_fold_pf:.3f} ok")
    elif (pf_pass or dd_pass) and not fold_ok:
        verdict = (f"FAIL-FOLD -- aggregate ok but fold {min_fold_idx} PF {min_fold_pf:.3f} < {fold_floor:.3f}")
    elif pf < CORRECTED_BASELINE_PF:
        verdict = f"FAIL-WORSE-THAN-BASELINE -- PF {pf:.3f} < {CORRECTED_BASELINE_PF} corrected baseline"
    else:
        verdict = (f"INCONCLUSIVE -- PF {pf:.3f}, DD {dd:.1f}%, min_fold {min_fold_pf:.3f} (fold {min_fold_idx}); "
                   f"doesn't beat heuristic on PF or DD")
    print(f"  {verdict}")

    # --- Save ---
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_trades": int(len(trade_data)),
            "n_skipped": int(skipped),
            "threshold": float(args.threshold),
            "fold0_fallback": args.fold0_fallback,
            "session_end_bar": int(args.session_end_bar),
            "commission": float(args.commission),
            "model": "HistGradientBoostingClassifier",
            "trade_state_features": TRADE_STATE_NAMES,
            "n_layer2_features": int(ds.X_sim.shape[1]),
        },
        "overall": overall,
        "per_fold": fold_results,
        "anchors": {
            "corrected_baseline_pf": CORRECTED_BASELINE_PF,
            "corrected_baseline_dd": CORRECTED_BASELINE_DD,
            "corrected_baseline_fold0": CORRECTED_BASELINE_FOLD0,
            "heuristic_baseline_pf": HEURISTIC_BASELINE_PF,
            "heuristic_baseline_dd": HEURISTIC_BASELINE_DD,
            "heuristic_baseline_fold0": HEURISTIC_BASELINE_FOLD0,
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "layer3_train_replay.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    full_df.to_csv(os.path.join(args.out_dir, "layer3_trades.csv"), index=False)
    print()
    print(f"Saved: {out}")
    print(f"Saved: {os.path.join(args.out_dir, 'layer3_trades.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
