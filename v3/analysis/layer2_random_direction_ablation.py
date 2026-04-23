"""Layer-2 random-direction ablation (Check 2 from the reality-checks plan).

Question: is the Layer-2 direction head actually earning its keep, or
is the edge entirely in the entry gate (fixed_quantile on entry_score
+ product ranking + teacher-conditioned direction fallback)?

Approach: take the exact trades the winning run made in
`v3/artifacts/layer2_shared_enc_fixedq_detach/layer2_trades.csv`,
keep the same (day, bar_index) selections, and override direction in
one of two ways:

- all-trades mode: randomize every chosen trade
- fallback-only mode: keep teacher-triggered bars teacher-directed and
  randomize only fallback bars

Recompute PnL using the same simulator path as replay.py. Report PF /
DD / TPD and compare to the winning run.

Disqualifying signal (per plan §2b):
- Random-direction PF ≥ 1.0 → direction head is cosmetic; the "win"
  is entry gate + teacher-conditioned fallback.
- Random-direction PF ≤ 0.85 → direction head contributes real lift.

Run:
    python -m v3.analysis.layer2_random_direction_ablation \
        --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    build_labeled_day,
    compute_time_stop_pnl_for_direction,
    replay_metrics_from_pnls,
    teacher_direction_hint_from_row,
    load_pickle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-2 random-direction ablation.")
    p.add_argument("--run-dir", default="v3/artifacts/layer2_shared_enc_fixedq_detach")
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--n-seeds", type=int, default=10,
                   help="Number of random seeds to average over (each seed is a fresh call/put coin-flip pass).")
    p.add_argument(
        "--mode",
        default="all_trades",
        choices=("all_trades", "fallback_only"),
        help="Randomize every chosen trade or only non-teacher fallback trades.",
    )
    return p.parse_args()


def _load_trades(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "layer2_trades.csv")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} trades from {path}")
    return df


def _trade_route_sources(run_dir: str, trades: pd.DataFrame) -> dict[tuple[str, int], str]:
    if "route_source" in trades.columns:
        return {
            (str(row.day), int(row.bar_index)): str(row.route_source)
            for row in trades.itertuples(index=False)
        }
    oof_path = os.path.join(run_dir, "oof_predictions.pkl")
    if not os.path.exists(oof_path):
        return {(str(row.day), int(row.bar_index)): "fallback" for row in trades.itertuples(index=False)}
    oof = load_pickle(oof_path)
    route_lookup: dict[tuple[str, int], str] = {}
    for row in oof.itertuples(index=False):
        key = (str(row.day), int(row.bar_index))
        teacher_dir = teacher_direction_hint_from_row(pd.Series(row._asdict()))
        route_lookup[key] = "teacher" if teacher_dir else "fallback"
    return {
        (str(row.day), int(row.bar_index)): route_lookup.get((str(row.day), int(row.bar_index)), "fallback")
        for row in trades.itertuples(index=False)
    }


def _recompute_one_seed(
    trades: pd.DataFrame,
    route_sources: dict[tuple[str, int], str],
    ds: V2Dataset,
    cfg: GuardrailConfig,
    equity: float,
    seed: int,
    mode: str,
) -> tuple[list[float], dict]:
    """Pick a random direction for each trade, recompute PnL."""
    rng = np.random.default_rng(seed)
    pnls: list[float] = []
    day_cache: dict[str, Any] = {}
    call_count = 0
    put_count = 0
    for _, row in trades.iterrows():
        day = str(row["day"])
        bar_index = int(row["bar_index"])
        if day not in day_cache:
            day_cache[day] = build_labeled_day(ds, day, cfg, equity=equity)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            continue
        bar = next((b for b in log.bars if b.bar_index == bar_index), None)
        if bar is None:
            continue
        route_source = route_sources.get((day, bar_index), "fallback")
        if mode == "fallback_only" and route_source == "teacher":
            direction = str(row["direction"])
        else:
            direction = "call" if rng.random() < 0.5 else "put"
        if direction == "call":
            call_count += 1
        else:
            put_count += 1
        pnl = compute_time_stop_pnl_for_direction(bar, sidecar, direction)
        if pnl is not None:
            pnls.append(float(pnl))
    return pnls, {"call_count": call_count, "put_count": put_count}


def main() -> int:
    args = parse_args()
    trades = _load_trades(args.run_dir)
    route_sources = _trade_route_sources(args.run_dir, trades)

    print("Loading v2 dataset + sidecars for PnL recomputation...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    # Collect PF / DD / metrics across n_seeds to average out coin-flip noise
    runs = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        pnls, counts = _recompute_one_seed(
            trades,
            route_sources,
            ds,
            cfg,
            args.equity,
            seed,
            args.mode,
        )
        m = replay_metrics_from_pnls(pnls, args.equity)
        m["seed"] = seed
        m["call_count"] = counts["call_count"]
        m["put_count"] = counts["put_count"]
        m["mean_pnl"] = float(np.mean(pnls)) if pnls else 0.0
        runs.append(m)
        print(
            f"  seed {seed}: n={len(pnls)}  PF={m['pf']:.3f}  DD={m['max_dd_pct']:.1f}%  "
            f"mean={m['mean_pnl']:+.1f}$  call={counts['call_count']} put={counts['put_count']}"
        )

    runs_df = pd.DataFrame(runs)
    print()
    print("=" * 80)
    print(f"Random-direction ablation ({args.mode}) — averaged over {args.n_seeds} seeds")
    print("=" * 80)
    print(f"PF:        mean={runs_df['pf'].mean():.3f}  std={runs_df['pf'].std():.3f}  "
          f"min={runs_df['pf'].min():.3f}  max={runs_df['pf'].max():.3f}")
    print(f"DD%:       mean={runs_df['max_dd_pct'].mean():.1f}  std={runs_df['max_dd_pct'].std():.1f}  "
          f"min={runs_df['max_dd_pct'].min():.1f}  max={runs_df['max_dd_pct'].max():.1f}")
    print(f"mean_pnl:  mean={runs_df['mean_pnl'].mean():+.1f}  std={runs_df['mean_pnl'].std():.1f}")
    print()
    print("Reference — winning run (model direction):")
    winning_pnls = trades["pnl"].astype(float).tolist()
    wm = replay_metrics_from_pnls(winning_pnls, args.equity)
    print(f"  PF={wm['pf']:.3f}  DD={wm['max_dd_pct']:.1f}%  mean={float(np.mean(winning_pnls)):+.1f}$  "
          f"trades={len(trades)}")
    print()
    print("Verdict criteria (plan §2b):")
    mean_pf_random = float(runs_df["pf"].mean())
    if mean_pf_random >= 1.0:
        print(f"  ⚠ HARD STOP: random-direction mean PF = {mean_pf_random:.3f} ≥ 1.0.")
        print(f"     The direction head is cosmetic; the edge is in the entry gate +")
        print(f"     teacher-conditioned direction fallback, not in Layer-2's side head.")
    elif mean_pf_random <= 0.85:
        print(f"  ✓ Direction head contributes real lift: random PF = {mean_pf_random:.3f} ≤ 0.85")
        print(f"    while model PF = {wm['pf']:.3f}. Side head is earning its keep.")
    else:
        print(f"  ~ Middling: random PF = {mean_pf_random:.3f} (between 0.85 and 1.0).")
        print(f"    Side head contributes some lift but the signal is modest. Not a hard stop.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
