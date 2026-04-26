"""Build the simulated-Layer-3 oracle per candidate contract.

For every (day, bar, candidate_contract) in the action-surface dataset
where tradeable_mask == 1, compute the PnL a rolling Layer-3 exit policy
would have realized on that contract. By default the Layer-3 models are
rebuilt from a champion chosen-trade artifact; optionally they can be
trained on a broader action-surface candidate sample. Saved as a sidecar
.npz file aligned to dataset row order.

See v3/reference/simulated_l3_oracle_design_2026_04_22.md.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.rolling_windows import generate_rolling_windows
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.action_surface_dataset import (
    DEFAULT_ACTION_SURFACE_DATASET_PATH,
    action_surface_dataset_fingerprint,
    validate_action_surface_bundle,
)
from v3.layer2.common import build_labeled_day
from v3.layer3.common import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_CANDIDATE_TRAIN_MAX_PER_DAY,
    DEFAULT_EQUITY,
    DEFAULT_MIN_TRAIN_TRADES,
    DEFAULT_SESSION_END_BAR,
    _build_trade_data,
    load_action_surface_candidate_trades,
    replay_trade_set,
    train_models_by_window,
)
from v3.layer3.common import load_unified_policy_trades, build_trade_dataset


TRIGGER_FLAT = 0
TRIGGER_MODEL = 1
TRIGGER_TIME_STOP_FALLBACK = 2
TRIGGER_FOLD0_TIME_STOP = 3
TRIGGER_NON_TRADEABLE = -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_ACTION_SURFACE_DATASET_PATH)
    p.add_argument(
        "--champion-chosen-trades",
        default="v3/artifacts/layer2_unified_policy_dev_3seed/seed_42/chosen_trades.pkl",
        help="Path to champion seed-42 chosen_trades.pkl to retrain the per-window L3 models on.",
    )
    p.add_argument(
        "--champion-calibration",
        default="v3/artifacts/layer3_unified_cpu_w000_seed42/rolling_layer3_calibrated_robust_90.json",
        help="Path to champion seed-42 robust-90 calibration JSON for per-window thresholds.",
    )
    p.add_argument(
        "--output",
        default="v3/artifacts/simulated_l3_oracle_seed42.npz",
    )
    p.add_argument(
        "--l3-training-source",
        default="champion",
        choices=("champion", "candidate_surface"),
        help=(
            "Training universe for the L3 models used to simulate candidate "
            "exits. champion preserves the historical chosen-trade source; "
            "candidate_surface trains on a deterministic broader sample from "
            "the action-surface dataset."
        ),
    )
    p.add_argument(
        "--candidate-train-max-per-day",
        type=int,
        default=DEFAULT_CANDIDATE_TRAIN_MAX_PER_DAY,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_CONTRACT)
    p.add_argument("--session-end-bar", type=int, default=DEFAULT_SESSION_END_BAR)
    p.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    p.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="If > 0, only process the first N dataset days. For smoke testing.",
    )
    p.add_argument(
        "--progress-every-days",
        type=int,
        default=25,
        help="Log progress every N days.",
    )
    p.add_argument(
        "--side-blind",
        action="store_true",
        help="Angle A: set direction_is_call=0 for all trades; forces a "
        "side-agnostic exit policy. Used to test whether removing per-side "
        "conditioning narrows cross-cell PF spread.",
    )
    return p.parse_args()


def _build_champion_models(
    chosen_trades_path: str,
    equity: float,
    session_end_bar: int,
    commission: float,
    seed: int,
    min_train_trades: int,
    *,
    side_blind: bool = False,
) -> tuple[dict[int, Any], list[dict]]:
    """Rebuild the champion per-window L3 models by retraining on chosen trades."""
    chosen, policy_meta = load_unified_policy_trades(chosen_trades_path)
    print(
        f"Champion source: {chosen_trades_path} -> {len(chosen)} chosen trades",
        flush=True,
    )
    trade_data, meta = build_trade_dataset(
        chosen,
        equity=equity,
        session_end_bar=session_end_bar,
        commission=commission,
        side_blind=side_blind,
    )
    print(
        f"Rebuilding champion L3 trade-datasets: usable={meta['n_trade_datasets']} skipped={meta['n_skipped']}",
        flush=True,
    )
    models, train_reports = train_models_by_window(
        trade_data, seed=seed, min_train_trades=min_train_trades
    )
    return models, train_reports


def _build_candidate_surface_models(
    dataset_path: str,
    max_per_day: int,
    equity: float,
    session_end_bar: int,
    commission: float,
    seed: int,
    min_train_trades: int,
    *,
    side_blind: bool = False,
) -> tuple[dict[int, Any], list[dict], dict[str, Any]]:
    """Train per-window L3 models on a broad action-surface candidate sample."""
    candidate_trades, candidate_meta = load_action_surface_candidate_trades(
        dataset_path,
        max_per_day=max_per_day,
    )
    print(
        f"Candidate L3 source: {dataset_path} -> {len(candidate_trades)} sampled trades "
        f"(max_per_day={max_per_day}, call_share={candidate_meta['call_share']:.3f})",
        flush=True,
    )
    trade_data, dataset_meta = build_trade_dataset(
        candidate_trades,
        equity=equity,
        session_end_bar=session_end_bar,
        commission=commission,
        side_blind=side_blind,
    )
    print(
        f"Rebuilding candidate-trained L3 trade-datasets: "
        f"usable={dataset_meta['n_trade_datasets']} skipped={dataset_meta['n_skipped']}",
        flush=True,
    )
    models, train_reports = train_models_by_window(
        trade_data, seed=seed, min_train_trades=min_train_trades
    )
    return models, train_reports, {
        "candidate_policy_meta": candidate_meta,
        "candidate_dataset_meta": dataset_meta,
    }


def _load_champion_thresholds(calibration_path: str) -> dict[int, float]:
    with open(calibration_path) as f:
        payload = json.load(f)
    out: dict[int, float] = {}
    for entry in payload["per_window_calibration"]:
        out[int(entry["window_idx"])] = float(entry["chosen_threshold"])
    return out


def _day_to_window(windows) -> dict[str, int]:
    return {d: w.window_idx for w in windows for d in w.oos_days}


def _infer_direction(right_is_call: float) -> tuple[str, str]:
    if right_is_call > 0.5:
        return "call", "C"
    return "put", "P"


def _simulate_candidate(
    *,
    day: str,
    bar_index: int,
    window_idx: int,
    strike: float,
    right_is_call: float,
    log: Any,
    sidecar: dict,
    paths_cache: dict,
    minute_map_cache: dict,
    ds: V2Dataset,
    session_end_bar: int,
    commission: float,
    models: dict[int, Any],
    thresholds: dict[int, float],
    side_blind: bool = False,
) -> tuple[float, int, int]:
    """Returns (l3_exit_pnl, exit_bar, trigger)."""
    direction, right = _infer_direction(right_is_call)
    entry_fill_bar = int(bar_index) + 1
    if entry_fill_bar >= session_end_bar:
        return (float("nan"), -1, TRIGGER_NON_TRADEABLE)
    fake_trade = pd.Series(
        {
            "bar_index": int(bar_index),
            "entry_fill_bar": int(entry_fill_bar),
            "direction": direction,
            "day": str(day),
            "selected_strike": float(strike),
            "selected_right": right,
            "time_stop_pnl": 0.0,
            "clean_entry_prob": 0.0,
            "window_idx": int(window_idx),
        }
    )
    td = _build_trade_data(
        fake_trade,
        log,
        sidecar,
        paths_cache,
        minute_map_cache,
        ds,
        session_end_bar,
        commission,
        side_blind=side_blind,
    )
    if td is None:
        return (float("nan"), -1, TRIGGER_NON_TRADEABLE)

    # Override the placeholder time_stop_pnl with the real last-bar PnL so
    # that fold-0 fallback and time_stop_fallback triggers return the correct
    # value (td was built from a fake trade with time_stop_pnl=0.0).
    if td["per_bar"]:
        td["time_stop_pnl"] = float(td["per_bar"][-1]["current_pnl"])
    td["window_idx"] = int(window_idx)
    model = models.get(int(window_idx))
    threshold = thresholds.get(int(window_idx), 0.20)

    rows = replay_trade_set(
        [td],
        model,
        float(threshold),
        fallback_policy="time_stop",
        fallback_bars=90,
    )
    row = rows[0]
    trigger_str = row["trigger"]
    trigger_int = {
        "model": TRIGGER_MODEL,
        "time_stop": TRIGGER_FOLD0_TIME_STOP if model is None else TRIGGER_TIME_STOP_FALLBACK,
        "time_stop_fallback": TRIGGER_TIME_STOP_FALLBACK,
    }.get(trigger_str, TRIGGER_TIME_STOP_FALLBACK)
    return (
        float(row["exit_pnl"]),
        int(row["exit_bar"]),
        int(trigger_int),
    )


def main() -> int:
    args = parse_args()
    t_start = time.time()

    # 1) Load L3 models + thresholds
    l3_training_meta: dict[str, Any]
    if args.l3_training_source == "candidate_surface":
        models, train_reports, candidate_meta = _build_candidate_surface_models(
            args.dataset,
            max_per_day=args.candidate_train_max_per_day,
            equity=args.equity,
            session_end_bar=args.session_end_bar,
            commission=args.commission,
            seed=args.seed,
            min_train_trades=args.min_train_trades,
            side_blind=args.side_blind,
        )
        l3_training_meta = {
            "source": "candidate_surface",
            "candidate_train_max_per_day": int(args.candidate_train_max_per_day),
            **candidate_meta,
        }
    else:
        models, train_reports = _build_champion_models(
            args.champion_chosen_trades,
            equity=args.equity,
            session_end_bar=args.session_end_bar,
            commission=args.commission,
            seed=args.seed,
            min_train_trades=args.min_train_trades,
            side_blind=args.side_blind,
        )
        l3_training_meta = {
            "source": "champion",
            "champion_chosen_trades": args.champion_chosen_trades,
        }
    thresholds = _load_champion_thresholds(args.champion_calibration)
    print(f"Loaded per-window thresholds: {thresholds}", flush=True)

    # 2) Load action-surface dataset
    print(f"Loading action-surface dataset from {args.dataset}", flush=True)
    with open(args.dataset, "rb") as f:
        bundle = pickle.load(f)
    validate_action_surface_bundle(bundle, required_labels={"tradeable_mask"})
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    contract_features = bundle["contract_features"]
    contract_strike = bundle["contract_strike"]
    tradeable_mask = bundle["action_labels"]["tradeable_mask"]
    n_rows = len(rows)
    n_actions = tradeable_mask.shape[1]
    top_k = int(bundle["meta"]["top_k_contracts_per_side"])

    # 3) Resolve day → window mapping using the same rolling windows the trainer uses
    unique_days = sorted(rows["day"].astype(str).unique().tolist())
    windows = generate_rolling_windows(unique_days)
    day_window_map = _day_to_window(windows)
    print(
        f"Rolling windows: {len(windows)} total; {len(day_window_map)} days covered OOS.",
        flush=True,
    )

    # Optional smoke: only first N dataset days
    if args.max_days and args.max_days > 0:
        keep_days = set(unique_days[: args.max_days])
        row_mask = rows["day"].astype(str).isin(keep_days).to_numpy()
        rows = rows.loc[row_mask].reset_index(drop=True)
        contract_features = contract_features[row_mask]
        contract_strike = contract_strike[row_mask]
        tradeable_mask = tradeable_mask[row_mask]
        n_rows = len(rows)
        print(f"[smoke] limiting to first {args.max_days} days -> {n_rows} rows", flush=True)
    dataset_fingerprint = meta.get("dataset_fingerprint")
    if not dataset_fingerprint or args.max_days > 0:
        dataset_fingerprint = action_surface_dataset_fingerprint(rows, meta)

    # 4) Allocate output arrays
    l3_exit_pnl = np.full((n_rows, n_actions), np.nan, dtype=np.float32)
    l3_exit_bar = np.full((n_rows, n_actions), -1, dtype=np.int32)
    l3_exit_trigger = np.full((n_rows, n_actions), TRIGGER_NON_TRADEABLE, dtype=np.int8)
    # flat action
    l3_exit_pnl[:, 0] = 0.0
    l3_exit_bar[:, 0] = -1
    l3_exit_trigger[:, 0] = TRIGGER_FLAT

    # 5) Walk days, simulate each tradeable (bar, candidate)
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    day_cache: dict[str, tuple[Any, Any]] = {}
    paths_cache: dict[int, Any] = {}
    minute_map_cache: dict[str, dict[int, int]] = {}

    n_sims = 0
    n_skipped = 0
    processed_days = 0
    row_by_day = rows.groupby("day").indices  # day -> np.ndarray of row indices

    for day_i, day in enumerate(sorted(row_by_day.keys())):
        window_idx = day_window_map.get(str(day))
        if window_idx is None:
            # Day is not in any OOS window (e.g., pre-first-window). Skip.
            continue
        if day not in day_cache:
            log_tmp, sc_tmp = build_labeled_day(ds, day, cfg, equity=args.equity)
            day_cache[day] = (log_tmp, sc_tmp)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            continue

        row_indices = row_by_day[day]
        for row_idx in row_indices:
            bar_index = int(rows.iloc[row_idx]["bar_index"])
            for j in range(top_k * 2):
                action_id = j + 1
                if tradeable_mask[row_idx, action_id] < 0.5:
                    continue
                strike = float(contract_strike[row_idx, j])
                right_is_call = float(contract_features[row_idx, j, 0])
                if not np.isfinite(strike):
                    n_skipped += 1
                    continue
                pnl, xbar, trig = _simulate_candidate(
                    day=day,
                    bar_index=bar_index,
                    window_idx=int(window_idx),
                    strike=strike,
                    right_is_call=right_is_call,
                    log=log,
                    sidecar=sidecar,
                    paths_cache=paths_cache,
                    minute_map_cache=minute_map_cache,
                    ds=ds,
                    session_end_bar=args.session_end_bar,
                    commission=args.commission,
                    models=models,
                    thresholds=thresholds,
                    side_blind=args.side_blind,
                )
                l3_exit_pnl[row_idx, action_id] = np.float32(pnl)
                l3_exit_bar[row_idx, action_id] = np.int32(xbar)
                l3_exit_trigger[row_idx, action_id] = np.int8(trig)
                n_sims += 1
        processed_days += 1
        # Drop caches to keep memory bounded
        paths_cache.pop(id(sidecar), None)
        minute_map_cache.pop(day, None)

        if (day_i + 1) % args.progress_every_days == 0:
            elapsed = time.time() - t_start
            pct = 100.0 * (day_i + 1) / max(len(row_by_day), 1)
            print(
                f"[{day_i + 1}/{len(row_by_day)} days, {pct:.1f}%] "
                f"sims={n_sims} skipped={n_skipped} elapsed={elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"Build complete: days={processed_days} sims={n_sims} skipped={n_skipped} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )

    # 6) Save .npz
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    meta_json = json.dumps(
        {
            "dataset_path": args.dataset,
            "champion_chosen_trades": args.champion_chosen_trades,
            "champion_calibration": args.champion_calibration,
            "l3_training_source": args.l3_training_source,
            "l3_training": l3_training_meta,
            "seed": int(args.seed),
            "equity": float(args.equity),
            "commission": float(args.commission),
            "session_end_bar": int(args.session_end_bar),
            "min_train_trades": int(args.min_train_trades),
            "n_rows": int(n_rows),
            "n_actions": int(n_actions),
            "dataset_fingerprint": str(dataset_fingerprint),
            "n_sims": int(n_sims),
            "n_skipped": int(n_skipped),
            "thresholds_by_window": {int(k): float(v) for k, v in thresholds.items()},
            "trade_reports": train_reports,
            "build_seconds": float(elapsed),
        },
        default=str,
    )
    np.savez_compressed(
        args.output,
        l3_exit_pnl=l3_exit_pnl,
        l3_exit_bar=l3_exit_bar,
        l3_exit_trigger=l3_exit_trigger,
        meta_json=np.array(meta_json, dtype=object),
    )
    print(f"Saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
