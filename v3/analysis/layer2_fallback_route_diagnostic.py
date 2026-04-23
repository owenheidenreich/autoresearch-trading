"""Layer-2 fallback-route diagnostic for a chosen run.

Question: for the exact bars a Layer-2 run selected, how much of the
PnL comes from fallback routing versus teacher-directed bars?

Controls evaluated on the SAME chosen bars:
- teacher unchanged, fallback = put
- teacher unchanged, fallback = call
- teacher unchanged, fallback = flat
- teacher unchanged, fallback = random call/put
- teacher unchanged, fallback = oracle-best-of-call/put/flat

Run:
    .venv/bin/python -m v3.analysis.layer2_fallback_route_diagnostic \
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
    DEFAULT_DATASET_PATH,
    build_labeled_day,
    compute_time_stop_pnl_for_direction,
    load_export_bundle,
    load_pickle,
    replay_metrics_from_pnls,
    teacher_direction_hint_from_row,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-2 fallback-route diagnostic.")
    p.add_argument("--run-dir", default="v3/artifacts/layer2_shared_enc_fixedq_detach")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--random-seed", type=int, default=101)
    p.add_argument("--random-n-seeds", type=int, default=10)
    return p.parse_args()


def _load_trade_context(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(os.path.join(run_dir, "layer2_trades.csv"))
    oof = load_pickle(os.path.join(run_dir, "oof_predictions.pkl"))
    return trades, oof


def _chosen_with_context(trades: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "day",
        "bar_index",
        "fold_idx",
        "teacher_any_triggered",
        "orc_buy_call",
        "orc_buy_put",
        "failed_break_buy_call",
        "failed_break_buy_put",
        "time_stop_pnl_call",
        "time_stop_pnl_put",
    ]
    merged = trades.merge(
        oof[keep_cols],
        on=["day", "bar_index", "fold_idx"],
        how="left",
        validate="one_to_one",
    )
    merged["teacher_direction"] = merged.apply(teacher_direction_hint_from_row, axis=1)
    merged["route_source"] = np.where(merged["teacher_direction"] != "", "teacher", "fallback")
    return merged


def _augment_contract_pnls(chosen: pd.DataFrame, ds: V2Dataset, cfg: GuardrailConfig, equity: float) -> pd.DataFrame:
    day_cache: dict[str, Any] = {}
    records = []
    for row in chosen.itertuples(index=False):
        day = str(row.day)
        bar_index = int(row.bar_index)
        if day not in day_cache:
            day_cache[day] = build_labeled_day(ds, day, cfg, equity=equity)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            continue
        bar = next((b for b in log.bars if b.bar_index == bar_index), None)
        if bar is None:
            continue
        call_pnl = compute_time_stop_pnl_for_direction(bar, sidecar, "call")
        put_pnl = compute_time_stop_pnl_for_direction(bar, sidecar, "put")
        rec = row._asdict()
        rec["call_pnl_live"] = call_pnl
        rec["put_pnl_live"] = put_pnl
        records.append(rec)
    return pd.DataFrame(records)


def _control_direction(row: pd.Series, control: str, rng: np.random.Generator | None = None) -> str:
    teacher_direction = str(row.get("teacher_direction", ""))
    if teacher_direction in {"call", "put"}:
        return teacher_direction

    call_pnl = row.get("call_pnl_live")
    put_pnl = row.get("put_pnl_live")

    if control == "fallback_put":
        return "put"
    if control == "fallback_call":
        return "call"
    if control == "fallback_flat":
        return ""
    if control == "fallback_random":
        if rng is None:
            raise ValueError("fallback_random requires rng")
        return "call" if rng.random() < 0.5 else "put"
    if control == "fallback_oracle":
        call_val = float(call_pnl) if call_pnl is not None and not pd.isna(call_pnl) else float("-inf")
        put_val = float(put_pnl) if put_pnl is not None and not pd.isna(put_pnl) else float("-inf")
        best = max(call_val, put_val, 0.0)
        if best <= 0.0:
            return ""
        return "call" if call_val >= put_val else "put"
    raise ValueError(f"Unknown control={control!r}")


def _apply_control(chosen: pd.DataFrame, control: str, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed) if seed is not None else None
    rows = []
    for _, row in chosen.iterrows():
        direction = _control_direction(row, control, rng=rng)
        if direction == "call":
            pnl = row["call_pnl_live"]
        elif direction == "put":
            pnl = row["put_pnl_live"]
        else:
            pnl = None
        if pnl is None or pd.isna(pnl):
            continue
        rows.append({
            "day": row["day"],
            "fold_idx": int(row["fold_idx"]),
            "bar_index": int(row["bar_index"]),
            "route_source": str(row["route_source"]),
            "direction": direction,
            "pnl": float(pnl),
        })
    return pd.DataFrame(rows)


def _metrics_for_trades(trades: pd.DataFrame, equity: float, n_days: int) -> dict[str, float]:
    if trades.empty:
        return {
            "pf": 0.0,
            "dd_pct": 0.0,
            "mean_pnl": 0.0,
            "trades": 0.0,
            "trades_per_day": 0.0,
            "call_pct": 0.0,
            "teacher_trades": 0.0,
            "fallback_trades": 0.0,
        }
    m = replay_metrics_from_pnls(trades["pnl"].astype(float).tolist(), equity)
    m["mean_pnl"] = float(trades["pnl"].mean())
    m["trades"] = float(len(trades))
    m["trades_per_day"] = float(len(trades) / max(n_days, 1))
    m["call_pct"] = float((trades["direction"] == "call").mean())
    m["teacher_trades"] = float((trades["route_source"] == "teacher").sum())
    m["fallback_trades"] = float((trades["route_source"] == "fallback").sum())
    m["dd_pct"] = float(m["max_dd_pct"])
    return m


def _print_control(name: str, overall: dict[str, float], per_fold: dict[int, dict[str, float]]) -> None:
    print()
    print(f"{name}")
    print(
        f"  overall: PF={overall['pf']:.3f} DD={overall['dd_pct']:.1f}% mean={overall['mean_pnl']:+.1f}$ "
        f"TPD={overall['trades_per_day']:.3f} teacher={int(overall['teacher_trades'])} fallback={int(overall['fallback_trades'])}"
    )
    for fold_idx in sorted(per_fold):
        m = per_fold[fold_idx]
        print(
            f"  fold {fold_idx}: PF={m['pf']:.3f} DD={m['dd_pct']:.1f}% mean={m['mean_pnl']:+.1f}$ "
            f"TPD={m['trades_per_day']:.3f} teacher={int(m['teacher_trades'])} fallback={int(m['fallback_trades'])}"
        )


def main() -> int:
    args = parse_args()
    trades, oof = _load_trade_context(args.run_dir)
    chosen = _chosen_with_context(trades, oof)

    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    chosen = _augment_contract_pnls(chosen, ds, cfg, args.equity)
    bundle = load_export_bundle(args.dataset)
    folds = list(bundle["meta"]["folds"])
    fold_days = {int(f["fold_idx"]): len(f["test_days"]) for f in folds}
    total_days = sum(fold_days.values())

    controls = {
        "teacher+put": ("fallback_put", None),
        "teacher+call": ("fallback_call", None),
        "teacher+flat": ("fallback_flat", None),
        "teacher+oracle_best": ("fallback_oracle", None),
    }

    print("=" * 100)
    print("Fallback-route diagnostic on exact chosen bars")
    print("=" * 100)
    print(f"Chosen bars: {len(chosen)}  teacher={int((chosen['route_source'] == 'teacher').sum())}  fallback={int((chosen['route_source'] == 'fallback').sum())}")

    results: dict[str, dict[str, Any]] = {}

    for label, (control, seed) in controls.items():
        sim = _apply_control(chosen, control, seed=seed)
        overall = _metrics_for_trades(sim, args.equity, total_days)
        per_fold = {
            fold_idx: _metrics_for_trades(sim[sim["fold_idx"] == fold_idx], args.equity, fold_days[fold_idx])
            for fold_idx in sorted(fold_days)
        }
        results[label] = {"overall": overall, "per_fold": per_fold}
        _print_control(label, overall, per_fold)

    random_overall = []
    random_per_fold: dict[int, list[dict[str, float]]] = {fold_idx: [] for fold_idx in sorted(fold_days)}
    for seed in range(args.random_seed, args.random_seed + args.random_n_seeds):
        sim = _apply_control(chosen, "fallback_random", seed=seed)
        random_overall.append(_metrics_for_trades(sim, args.equity, total_days))
        for fold_idx in sorted(fold_days):
            random_per_fold[fold_idx].append(
                _metrics_for_trades(sim[sim["fold_idx"] == fold_idx], args.equity, fold_days[fold_idx])
            )

    def _avg(metrics: list[dict[str, float]]) -> dict[str, float]:
        keys = metrics[0].keys() if metrics else []
        return {key: float(np.mean([m[key] for m in metrics])) for key in keys}

    random_summary = {
        "overall": _avg(random_overall),
        "per_fold": {fold_idx: _avg(ms) for fold_idx, ms in random_per_fold.items()},
    }
    results["teacher+random_mean"] = random_summary
    _print_control("teacher+random_mean", random_summary["overall"], random_summary["per_fold"])

    out_path = os.path.join(args.run_dir, "fallback_route_diagnostic.json")
    with open(out_path, "w") as f:
        import json
        json.dump(results, f, indent=2, sort_keys=True)
    print()
    print(f"Saved diagnostic: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
