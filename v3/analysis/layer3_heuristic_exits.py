"""Stage 2 of the Layer 3 (learned exits) workstream.

Question: what's the PF/DD ceiling from simple rule-based exits?
Anchors what a learned model has to beat in Stage 3.

Heuristics tested on each of the 275 chosen trades from the detach-side
baseline, all using the corrected exit-bar spread fill model:

- trailing_N: once MFE > 0, exit if current_pnl <= MFE * (1 - N/100).
  N ∈ {25, 40, 60} (give back N% of MFE gain). Higher N = looser trail.
- take_profit_X: exit if current_pnl >= X% * entry_premium ($).
  X ∈ {50, 80, 120}.
- bail_out_Y: exit if current_pnl <= -Y% * entry_premium ($).
  Y ∈ {30, 50, 70}.
- time_of_day_K: exit at bar entry+K (ignores price action).
  K ∈ {30, 60, 90} (minutes since entry).
- composite_X_N: arm trailing_N AFTER take_profit_X reached.
  Common practitioner pattern. (X=50, N=40)

For each heuristic, walk per-bar PnLs through [entry+1, session_end];
exit at first trigger or fall back to time-stop. Aggregate PF/DD/mean
overall and per-fold; report fraction of trades exited early.

Verdict gates (per the Layer 3 plan):
- GO: best heuristic PF >= corrected baseline (1.472) + 0.15 (i.e. >= 1.622)
- STOP-LITE: best heuristic improves DD without PF lift -> use heuristic as production
- STOP: no heuristic moves the needle

Run:
    python -m v3.analysis.layer3_heuristic_exits \
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
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_heuristic_exits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer 3 heuristic-exit ablation (Stage 2).")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--session-end-bar", type=int, default=DEFAULT_SESSION_END_BAR)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_CONTRACT)
    return p.parse_args()


def _per_bar_pnls(
    path,
    entry_bar: int,
    entry_mid: float,
    entry_sf: float,
    session_end_bar: int,
    commission: float,
) -> list[tuple[int, float]]:
    """Return (bar, current_pnl) pairs for every finite post-entry bar."""
    end = min(session_end_bar, len(path.mids) - 1)
    entry_ask = entry_mid * (1.0 + entry_sf / 2.0)
    out: list[tuple[int, float]] = []
    for t in range(entry_bar + 1, end + 1):
        m = path.mids[t]
        if not np.isfinite(m):
            continue
        sf = path.spread_fracs[t]
        if not np.isfinite(sf):
            sf = entry_sf
        exit_bid = float(m) * (1.0 - float(sf) / 2.0)
        out.append((int(t), 100.0 * (exit_bid - entry_ask) - commission))
    return out


def _apply_heuristic(
    pnls: list[tuple[int, float]],
    entry_bar: int,
    entry_premium_dollars: float,
    heuristic: str,
    params: dict,
) -> tuple[float, int, str]:
    """Walk per-bar PnLs and apply the heuristic. Returns (exit_pnl, exit_bar, trigger).
    If never triggered, falls back to the last finite bar (time-stop)."""
    if not pnls:
        return float("nan"), -1, "no_data"

    mfe = -np.inf
    armed = False  # for composite
    last_t, last_pnl = pnls[-1]

    for t, pnl in pnls:
        if pnl > mfe:
            mfe = pnl

        if heuristic == "trailing":
            n_pct = params["n"] / 100.0
            if mfe > 0 and pnl <= mfe * (1.0 - n_pct):
                return float(pnl), int(t), "trailing"
        elif heuristic == "take_profit":
            target = (params["x"] / 100.0) * entry_premium_dollars
            if pnl >= target:
                return float(pnl), int(t), "take_profit"
        elif heuristic == "bail_out":
            stop = -(params["y"] / 100.0) * entry_premium_dollars
            if pnl <= stop:
                return float(pnl), int(t), "bail_out"
        elif heuristic == "time_of_day":
            if t >= entry_bar + params["k"]:
                return float(pnl), int(t), "time_of_day"
        elif heuristic == "composite":
            arm_target = (params["x"] / 100.0) * entry_premium_dollars
            n_pct = params["n"] / 100.0
            if not armed and pnl >= arm_target:
                armed = True
            if armed and mfe > 0 and pnl <= mfe * (1.0 - n_pct):
                return float(pnl), int(t), "composite"
        else:
            raise ValueError(f"unknown heuristic {heuristic}")

    return float(last_pnl), int(last_t), "time_stop_fallback"


def _process_trade_paths(
    trade: pd.Series,
    log,
    sidecar: dict,
    paths_cache: dict,
    session_end_bar: int,
    commission: float,
) -> dict[str, Any] | None:
    bar_index = int(trade["bar_index"])
    direction = str(trade["direction"])
    bar = next((b for b in log.bars if b.bar_index == bar_index), None)
    if bar is None:
        return None
    selection = select_contract(bar.contracts, direction, "layer2")
    if selection is None:
        return None
    right = "P" if direction == "put" else "C"
    match = next(
        (c for c in bar.contracts if c.strike == selection.strike and c.right == right),
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
    pnls = _per_bar_pnls(path, bar_index, match.mid, match.spread_fraction,
                        session_end_bar, commission)
    if not pnls:
        return None
    return {
        "fold_idx": int(trade["fold_idx"]),
        "day": str(trade["day"]),
        "bar_index": bar_index,
        "direction": direction,
        "csv_pnl": float(trade["pnl"]),
        "entry_mid": float(match.mid),
        "entry_premium_dollars": float(match.mid * 100.0),
        "pnls": pnls,  # list of (bar, pnl)
    }


def _agg(trades_df: pd.DataFrame, equity: float, total_days: int) -> dict[str, float]:
    if trades_df.empty:
        return {"pf": 0.0, "max_dd_pct": 0.0, "mean_pnl": 0.0, "trades": 0.0, "trades_per_day": 0.0}
    sorted_df = trades_df.sort_values(["day", "bar_index"])
    pnls = sorted_df["exit_pnl"].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    m["trades"] = float(len(pnls))
    m["trades_per_day"] = float(len(pnls) / max(total_days, 1))
    m["mean_pnl"] = float(np.mean(pnls))
    return m


def _early_exit_share(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    triggered = (trades_df["trigger"] != "time_stop_fallback").sum()
    return float(triggered / len(trades_df))


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    print(f"Loaded {len(trades)} trades")
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds = list(bundle["meta"]["folds"])
    total_days = sum(len(f["test_days"]) for f in folds)
    fold_days = {int(f["fold_idx"]): len(f["test_days"]) for f in folds}

    print("Loading V2Dataset + per-day labeling...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    day_cache: dict[str, tuple] = {}
    paths_cache: dict[int, dict] = {}
    cached_trades: list[dict[str, Any]] = []
    skipped = 0
    for i, trade in trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            day_cache[day] = build_labeled_day(ds, day, cfg, equity=args.equity)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            skipped += 1
            continue
        result = _process_trade_paths(trade, log, sidecar, paths_cache,
                                       args.session_end_bar, args.commission)
        if result is None:
            skipped += 1
            continue
        cached_trades.append(result)
        if (i + 1) % 50 == 0:
            print(f"  cached {i+1}/{len(trades)}...")
    print(f"Cached {len(cached_trades)} trade paths; {skipped} skipped")

    # --- Heuristic configurations ---
    configs = [
        ("baseline_corrected_time_stop", "time_of_day", {"k": 10_000}),  # k > session always falls back
        ("trailing_25", "trailing", {"n": 25}),
        ("trailing_40", "trailing", {"n": 40}),
        ("trailing_60", "trailing", {"n": 60}),
        ("take_profit_50", "take_profit", {"x": 50}),
        ("take_profit_80", "take_profit", {"x": 80}),
        ("take_profit_120", "take_profit", {"x": 120}),
        ("bail_out_30", "bail_out", {"y": 30}),
        ("bail_out_50", "bail_out", {"y": 50}),
        ("bail_out_70", "bail_out", {"y": 70}),
        ("time_of_day_30", "time_of_day", {"k": 30}),
        ("time_of_day_60", "time_of_day", {"k": 60}),
        ("time_of_day_90", "time_of_day", {"k": 90}),
        ("composite_tp50_trail40", "composite", {"x": 50, "n": 40}),
    ]

    # --- Run each config ---
    results_by_config: dict[str, dict[str, Any]] = {}
    for label, heur, params in configs:
        rows = []
        for ct in cached_trades:
            exit_pnl, exit_bar, trigger = _apply_heuristic(
                ct["pnls"], ct["bar_index"], ct["entry_premium_dollars"], heur, params,
            )
            rows.append({
                "fold_idx": ct["fold_idx"], "day": ct["day"], "bar_index": ct["bar_index"],
                "direction": ct["direction"], "exit_pnl": exit_pnl, "exit_bar": exit_bar,
                "trigger": trigger, "bars_held": exit_bar - ct["bar_index"],
            })
        df = pd.DataFrame(rows)
        overall = _agg(df, args.equity, total_days)
        per_fold = {
            int(fi): _agg(df[df["fold_idx"] == fi], args.equity, fold_days.get(int(fi), 1))
            for fi in sorted(df["fold_idx"].unique())
        }
        results_by_config[label] = {
            "heuristic": heur, "params": params,
            "overall": overall, "per_fold": per_fold,
            "early_exit_share": _early_exit_share(df),
            "mean_bars_held": float(df["bars_held"].mean()),
        }

    # --- Print summary ---
    baseline_pf = results_by_config["baseline_corrected_time_stop"]["overall"]["pf"]
    baseline_dd = results_by_config["baseline_corrected_time_stop"]["overall"]["max_dd_pct"]
    baseline_mean = results_by_config["baseline_corrected_time_stop"]["overall"]["mean_pnl"]

    print()
    print("=" * 110)
    print(f"Aggregate (corrected-spread baseline: PF={baseline_pf:.3f} DD={baseline_dd:.1f}% mean=${baseline_mean:.0f})")
    print("=" * 110)
    print(f"{'config':<32}{'PF':>8}{'PF Δ':>8}{'DD%':>8}{'DD Δ':>8}{'mean$':>10}{'mean Δ':>10}{'early%':>9}{'bars':>7}")
    for label, _, _ in configs:
        r = results_by_config[label]
        o = r["overall"]
        print(
            f"{label:<32}{o['pf']:>8.3f}{o['pf']-baseline_pf:>8.3f}"
            f"{o['max_dd_pct']:>8.1f}{o['max_dd_pct']-baseline_dd:>8.1f}"
            f"{o['mean_pnl']:>10.1f}{o['mean_pnl']-baseline_mean:>10.1f}"
            f"{100*r['early_exit_share']:>8.1f}%{r['mean_bars_held']:>7.1f}"
        )

    # --- Per-fold for top-3 by PF ---
    ranked = sorted(
        [(label, r) for label, r in results_by_config.items() if label != "baseline_corrected_time_stop"],
        key=lambda kv: -kv[1]["overall"]["pf"],
    )
    top3 = ranked[:3]
    print()
    print("=" * 110)
    print("Per-fold PF for top-3 heuristics by aggregate PF (vs baseline corrected_time_stop)")
    print("=" * 110)
    base_pf_by_fold = {fi: results_by_config["baseline_corrected_time_stop"]["per_fold"][fi]["pf"]
                      for fi in results_by_config["baseline_corrected_time_stop"]["per_fold"]}
    print(f"{'fold':<6}{'baseline':>12}", end="")
    for label, _ in top3:
        print(f"{label:>30}", end="")
    print()
    for fi in sorted(base_pf_by_fold):
        print(f"{fi:<6}{base_pf_by_fold[fi]:>12.3f}", end="")
        for label, r in top3:
            pf = r["per_fold"].get(fi, {}).get("pf", float("nan"))
            print(f"{pf:>30.3f}", end="")
        print()

    # --- Verdict ---
    best_label, best_r = ranked[0]
    best_pf = best_r["overall"]["pf"]
    best_dd = best_r["overall"]["max_dd_pct"]
    pf_delta = best_pf - baseline_pf
    dd_delta = best_dd - baseline_dd

    print()
    print("=" * 110)
    print("Verdict (Stage 2)")
    print("=" * 110)
    print(f"  Best heuristic by PF: {best_label}  PF={best_pf:.3f} (Δ{pf_delta:+.3f}) DD={best_dd:.1f}% (Δ{dd_delta:+.1f})")
    if pf_delta >= 0.15:
        verdict = f"GO -- best heuristic '{best_label}' PF lift {pf_delta:+.3f} >= +0.15; learned model has room"
    elif dd_delta <= -5.0 and pf_delta >= -0.05:
        verdict = f"STOP-LITE -- best heuristic '{best_label}' tightens DD {dd_delta:+.1f}pt without PF regression; use as production exit"
    else:
        verdict = f"STOP -- no heuristic moves PF >= +0.15 OR DD <= -5pt without PF regression; exit decision may not be learnable"
    print(f"  VERDICT: {verdict}")

    # --- Save ---
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_trades": int(len(cached_trades)),
            "n_skipped": int(skipped),
            "session_end_bar": int(args.session_end_bar),
            "commission": float(args.commission),
            "baseline_corrected_pf": float(baseline_pf),
            "baseline_corrected_dd": float(baseline_dd),
            "baseline_corrected_mean": float(baseline_mean),
        },
        "results": {label: r for label, r in results_by_config.items()},
        "verdict": verdict,
        "best_heuristic": best_label,
    }
    out = os.path.join(args.out_dir, "heuristic_exits.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
