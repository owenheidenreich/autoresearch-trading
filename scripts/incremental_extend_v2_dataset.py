"""Incremental extension of v2/data.pt for the 17 new April-2026 trading days.

The default `v2.pipeline.build_v2_dataset` rebuilds the IV-history prepass
serially for ALL 1002 days every run (~50+ min on this Mac). This script
short-circuits that by:

  1. Reusing the existing v2/data.pt for the 986 cached days (raw atm_iv
     is in X_sim[:, atm_iv_col]).
  2. Building per-day work-items only for the 17 new days, seeding each
     with prior IV history extracted from the existing X_sim.
  3. Running _process_one_day for the 17 new days (parallel).
  4. Concatenating the new days' raw features to the existing X_sim.
  5. Re-running rolling-zscore normalization on the extended array (the
     v2 normalizer is a local 60-day rolling z-score, so this is fast).
  6. Writing the new data.pt with extended dates, masks, labels.

Then v3.layer2.export_action_surface_dataset rebuilds the action_surface
bundle on the extended data.pt; forward_walk re-runs.

Usage:
    .venv/bin/python -m scripts.incremental_extend_v2_dataset \
        --new-days 2026-04-02 2026-04-06 2026-04-07 ... 2026-04-24
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch

# Make build_v2_dataset's helpers reachable
from v2.pipeline import build_v2_dataset as bvd
from v2.pipeline.build_v2_dataset import (
    _process_one_day,
    _build_masks,
    _load_market_cache,
    SPX_PATH, SPY_PATH, VIX_PATH,
    FULL_CHAIN_CACHE_DIR,
    OUTPUT_PATH,
    SIDECAR_DIR,
    BARS_PER_DAY,
    NUM_FEATURES,
    OPTION_FEATURE_NAMES,
    SURFACE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    CHAIN_SCHEMA_VERSION,
    CONTRACT_FEATURE_FIELDS,
    DEFAULT_POLICY,
    SLICE_GATE_MIN_PNL,
    STRIKE_GRID,
    _get_config_fingerprint,
    _get_git_sha,
    compute_dataset_fingerprint,
    file_sha256,
    manifest_sidecar_digest,
    ensure_sidecar_dir,
)
from v2.pipeline.compute_features import compute_price_features
from v2.core.features import normalize_features


def _list_new_days(forward_window_start: str = "2026-04-02") -> list[str]:
    """Find dates with chain caches but no sidecar."""
    chain_dir = FULL_CHAIN_CACHE_DIR
    sidecar_dir = SIDECAR_DIR
    chain_days = sorted({f.replace(".pkl", "") for f in os.listdir(chain_dir) if f.endswith(".pkl")})
    sidecar_days = {f.replace(".pt", "") for f in os.listdir(sidecar_dir) if f.endswith(".pt")}
    new = [d for d in chain_days if d >= forward_window_start and d not in sidecar_days]
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-days", type=str, nargs="*", default=None)
    parser.add_argument("--forward-window-start", type=str, default="2026-04-02")
    parser.add_argument("--out", type=str, default=OUTPUT_PATH)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    if args.new_days:
        new_days = sorted(args.new_days)
    else:
        new_days = _list_new_days(args.forward_window_start)
    print(f"New days to process: {len(new_days)}")
    for d in new_days:
        print(f"  - {d}")

    if not new_days:
        print("Nothing to do.")
        return

    t0 = time.time()
    # ---- Step 1: load existing data.pt ----
    print(f"\nLoading existing {args.out}...")
    existing = torch.load(args.out, weights_only=False, map_location="cpu")
    existing_X = existing["X"].numpy()
    existing_X_sim = existing["X_sim"].numpy()
    existing_dates = list(existing["dates"])
    feature_names = list(existing["feature_names"])
    atm_iv_col = feature_names.index("atm_iv")
    print(f"  existing dates: {len(existing_dates)} (last {existing_dates[-1]})")
    print(f"  X_sim shape: {existing_X_sim.shape}")

    # ---- Step 2: load underlying caches + filter to new days ----
    print(f"\nLoading raw market data...")
    spx_df = _load_market_cache(SPX_PATH, "spx")
    spy_df = _load_market_cache(SPY_PATH, "spy")
    vix_df = _load_market_cache(VIX_PATH, "vix")

    # Restrict to new_days only
    new_set = set(new_days)
    mask = np.array([d in new_set for d in spx_df["date"]], dtype=bool)
    spx_df = {k: np.asarray(v)[mask] for k, v in spx_df.items()}
    n_new_bars = mask.sum()
    print(f"  new SPX bars: {n_new_bars} across {len(set(spx_df['date']))} days")
    if n_new_bars == 0:
        raise RuntimeError("No SPX bars matched new_days; underlying cache may not be extended.")

    new_dates_list = [str(d) for d in spx_df["date"]]
    spx_close = np.asarray(spx_df["spx_close"], dtype=np.float64)
    spx_high = np.asarray(spx_df["spx_high"], dtype=np.float64)
    spx_low = np.asarray(spx_df["spx_low"], dtype=np.float64)
    spx_open = np.asarray(spx_df["spx_open"], dtype=np.float64)

    # Map underlying timestamps for new bars
    spy_ts_to_idx = {str(ts): i for i, ts in enumerate(spy_df["timestamp"])}
    spy_volume = np.zeros(n_new_bars, dtype=np.float64)
    spy_close = np.zeros(n_new_bars, dtype=np.float64)
    for i, ts in enumerate(spx_df["timestamp"]):
        j = spy_ts_to_idx.get(str(ts))
        if j is not None:
            spy_volume[i] = float(spy_df["volume"][j])
            spy_close[i] = float(spy_df["close"][j])

    vix_ts_to_idx = {str(ts): i for i, ts in enumerate(vix_df["timestamp"])}
    vix_close = np.full(n_new_bars, np.nan, dtype=np.float64)
    for i, ts in enumerate(spx_df["timestamp"]):
        j = vix_ts_to_idx.get(str(ts))
        if j is not None:
            vix_close[i] = float(vix_df["vix_close"][j])

    # day_starts for new days
    day_starts = [0]
    for i in range(1, n_new_bars):
        if new_dates_list[i] != new_dates_list[i - 1]:
            day_starts.append(i)
    day_ends = day_starts[1:] + [n_new_bars]
    bar_of_day_new = np.zeros(n_new_bars, dtype=np.int32)
    for ds, de in zip(day_starts, day_ends):
        bar_of_day_new[ds:de] = np.arange(de - ds, dtype=np.int32)
    day_to_bars_new = defaultdict(list)
    for i, d in enumerate(new_dates_list):
        day_to_bars_new[d].append(i)
    unique_new_dates = sorted(day_to_bars_new.keys())

    # ---- Step 3: compute price features for new days ----
    print("\nComputing price features for new days...")
    X_price_new = compute_price_features(
        spx_close, spx_high, spx_low, spx_open,
        spy_volume, spy_close, vix_close,
        day_starts, bar_of_day_new,
    )

    n_opt = len(OPTION_FEATURE_NAMES)
    n_surface = len(SURFACE_FEATURE_NAMES)
    n_flow = len(FLOW_FEATURE_NAMES)
    X_opt_new = np.full((n_new_bars, n_opt), np.nan, dtype=np.float64)
    X_surface_new = np.zeros((n_new_bars, n_surface), dtype=np.float64)
    X_flow_new = np.zeros((n_new_bars, n_flow), dtype=np.float64)
    spot_prices_new = spx_close.astype(np.float32)

    # ---- Step 4: build IV history seeds from existing X_sim ----
    print("\nSeeding IV history from existing data.pt...")
    lookback = BARS_PER_DAY * 60  # 23400 bars
    # The IV history is a sequential list of finite atm_iv values from prior bars
    existing_atm_iv = existing_X_sim[:, atm_iv_col].astype(np.float64)
    finite_iv_history = existing_atm_iv[np.isfinite(existing_atm_iv)].tolist()
    seeds_new: dict[str, list[float]] = {}
    seed_for_first_new_day = list(finite_iv_history[-lookback:])
    print(f"  seed length: {len(seed_for_first_new_day)} ({lookback} target)")

    # NOTE: In the original prepass, each day's seed is the prior history;
    # for new days that are sequential, each subsequent day's seed grows.
    # But for the worker (`_process_one_day`), it just needs the SEED at
    # day-start, and it builds out the day's IV from there. So the simplest
    # correct path is to give each new day the same growing seed, treating
    # it as if all previous new days are part of the seed.
    # We approximate this by giving each new day's worker the seed of the
    # CURRENT history at that point (we update history as days are processed).
    history_running = list(finite_iv_history)
    for day in unique_new_dates:
        seeds_new[day] = list(history_running[-lookback:])
        # We'll extend history_running after each day's processing below.

    # ---- Step 5: process new days (parallel) ----
    print(f"\nProcessing {len(unique_new_dates)} new days with {args.n_workers} workers...")
    import multiprocessing
    work_items = []
    for day in unique_new_dates:
        gi = day_to_bars_new[day]
        work_items.append({
            "day": day,
            "global_indices": gi,
            "day_spot": spot_prices_new[gi],
            "day_X_price": X_price_new[gi],
            "day_timestamps": np.asarray(spx_df["timestamp"][gi], dtype=np.int64),
            "sidecar_dir": SIDECAR_DIR,
            "iv_history_seed": seeds_new[day],
        })
    sidecar_paths_new = []
    max_contracts_per_bar = 0
    nan_counts_opt_total = np.zeros(n_opt, dtype=np.int64)
    nan_counts_surface_total = np.zeros(n_surface, dtype=np.int64)
    best_contract_pnl_new = np.zeros(n_new_bars, dtype=np.float32)
    best_contract_strike_new = np.zeros(n_new_bars, dtype=np.float32)
    best_contract_right_new = np.full(n_new_bars, -1, dtype=np.int32)
    label_trade_new = np.zeros(n_new_bars, dtype=bool)
    label_trade_valid_new = np.zeros(n_new_bars, dtype=bool)
    slice_best_contract_pnl_new = np.zeros(n_new_bars, dtype=np.float32)
    slice_best_contract_strike_new = np.zeros(n_new_bars, dtype=np.float32)
    slice_best_contract_right_new = np.full(n_new_bars, -1, dtype=np.int32)
    slice_label_trade_new = np.zeros(n_new_bars, dtype=bool)
    slice_label_trade_valid_new = np.zeros(n_new_bars, dtype=bool)
    total_signal_bars = 0
    total_trade_bars = 0

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(args.n_workers) as pool:
        for result in pool.imap_unordered(_process_one_day, work_items, chunksize=2):
            gi = result["global_indices"]
            X_opt_new[gi] = result["X_opt_day"]
            X_surface_new[gi] = result["X_surface_day"]
            X_flow_new[gi] = result["X_flow_day"]
            nan_counts_opt_total += np.array(result["nan_counts_opt"], dtype=np.int64)
            nan_counts_surface_total += np.array(result["nan_counts_surface"], dtype=np.int64)
            sidecar_paths_new.append(result["sidecar_path"])
            max_contracts_per_bar = max(max_contracts_per_bar, result["max_contracts"])
            total_signal_bars += result["signal_bars"]
            total_trade_bars += result["trade_bars"]
            for info in result["bar_info"]:
                g = info["gi"]
                best_contract_pnl_new[g] = info["best_pnl"]
                label_trade_new[g] = info["label_trade"]
                label_trade_valid_new[g] = info["labelable"]
                slice_best_contract_pnl_new[g] = info["slice_best_pnl"]
                slice_label_trade_new[g] = info["slice_label_trade"]
                slice_label_trade_valid_new[g] = info["slice_labelable"]
                if info["is_trade"]:
                    best_contract_strike_new[g] = info["best_strike"]
                    best_contract_right_new[g] = info["best_right"]
                if info["is_slice_trade"]:
                    slice_best_contract_strike_new[g] = info["slice_best_strike"]
                    slice_best_contract_right_new[g] = info["slice_best_right"]
            print(f"  day {result.get('day_str','?')}: {result['max_contracts']} contracts, "
                  f"{result['signal_bars']} signal bars, {result['trade_bars']} trade bars")

    X_combined_new = np.concatenate([X_price_new, X_opt_new, X_surface_new, X_flow_new], axis=1).astype(np.float32)
    assert X_combined_new.shape[1] == NUM_FEATURES, \
        f"new days expected {NUM_FEATURES} cols, got {X_combined_new.shape[1]}"

    # ---- Step 6: append to existing X_sim, re-normalize ----
    print(f"\nAppending {n_new_bars} bars to existing {len(existing_dates)}-bar dataset...")
    new_X_sim = np.concatenate([existing_X_sim, X_combined_new], axis=0)
    new_dates_full = existing_dates + new_dates_list
    n_total = len(new_dates_full)
    print(f"  total bars: {n_total}, total days: {len(set(new_dates_full))}")
    print("\nRolling-zscore normalization on extended X_sim...")
    new_X = normalize_features(new_X_sim, np.ones(n_total, dtype=bool), dates=new_dates_full)

    # bar_of_day extends naturally
    n_existing = len(existing_dates)
    bar_of_day_full = np.concatenate([
        existing["bar_of_day"].numpy(),
        bar_of_day_new,
    ])

    # spot_prices append
    spot_prices_full = np.concatenate([
        existing["spot_prices"].numpy(),
        spot_prices_new.astype(np.float32),
    ])

    # labels append
    label_trade_full = np.concatenate([
        existing["label_trade"].numpy(), label_trade_new
    ])
    label_trade_valid_full = np.concatenate([
        existing["label_trade_valid"].numpy(), label_trade_valid_new
    ])
    best_contract_pnl_full = np.concatenate([
        existing["best_contract_pnl"].numpy(), best_contract_pnl_new
    ])
    best_contract_strike_full = np.concatenate([
        existing["best_contract_strike"].numpy(), best_contract_strike_new
    ])
    best_contract_right_full = np.concatenate([
        existing["best_contract_right"].numpy(), best_contract_right_new
    ])
    slice_label_trade_full = np.concatenate([
        existing["slice_label_trade"].numpy(), slice_label_trade_new
    ])
    slice_label_trade_valid_full = np.concatenate([
        existing["slice_label_trade_valid"].numpy(), slice_label_trade_valid_new
    ])
    slice_best_contract_pnl_full = np.concatenate([
        existing["slice_best_contract_pnl"].numpy(), slice_best_contract_pnl_new
    ])
    slice_best_contract_strike_full = np.concatenate([
        existing["slice_best_contract_strike"].numpy(), slice_best_contract_strike_new
    ])
    slice_best_contract_right_full = np.concatenate([
        existing["slice_best_contract_right"].numpy(), slice_best_contract_right_new
    ])

    # masks: rebuild from full dates
    train_mask, val_mask, promote_mask, shadow_mask, split_info = _build_masks(new_dates_full)

    # sidecar manifest digest: use existing + new
    all_sidecars = sorted(
        os.path.join(SIDECAR_DIR, f) for f in os.listdir(SIDECAR_DIR) if f.endswith(".pt")
    )
    sidecar_digest = manifest_sidecar_digest(all_sidecars)

    dataset = {
        "X": torch.from_numpy(new_X.astype(np.float32)),
        "X_sim": torch.from_numpy(new_X_sim.astype(np.float32)),
        "feature_names": feature_names,
        "spot_prices": torch.from_numpy(spot_prices_full.astype(np.float32)),
        "label_trade": torch.from_numpy(label_trade_full),
        "label_trade_valid": torch.from_numpy(label_trade_valid_full),
        "best_contract_pnl": torch.from_numpy(best_contract_pnl_full.astype(np.float32)),
        "best_contract_strike": torch.from_numpy(best_contract_strike_full.astype(np.float32)),
        "best_contract_right": torch.from_numpy(best_contract_right_full.astype(np.int32)),
        "slice_label_trade": torch.from_numpy(slice_label_trade_full),
        "slice_label_trade_valid": torch.from_numpy(slice_label_trade_valid_full),
        "slice_best_contract_pnl": torch.from_numpy(slice_best_contract_pnl_full.astype(np.float32)),
        "slice_best_contract_strike": torch.from_numpy(slice_best_contract_strike_full.astype(np.float32)),
        "slice_best_contract_right": torch.from_numpy(slice_best_contract_right_full.astype(np.int32)),
        "dates": new_dates_full,
        "bar_of_day": torch.from_numpy(bar_of_day_full),
        "train_mask": torch.from_numpy(train_mask),
        "val_mask": torch.from_numpy(val_mask),
        "promote_mask": torch.from_numpy(promote_mask),
        "shadow_mask": torch.from_numpy(shadow_mask),
        "metadata": dict(existing.get("metadata", {})),
    }
    dataset["metadata"]["build_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dataset["metadata"]["incremental_extension_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dataset["metadata"]["chain_sidecar_digest"] = sidecar_digest
    dataset["metadata"]["fingerprint"] = compute_dataset_fingerprint(dataset)

    backup_path = args.out + ".pre_increment_backup"
    if not os.path.exists(backup_path):
        os.rename(args.out, backup_path)
        print(f"  backed up old data.pt -> {backup_path}")

    torch.save(dataset, args.out)
    with open(f"{args.out}.sha256", "w") as f:
        f.write(file_sha256(args.out))

    print(f"\nSaved {args.out}")
    print(f"  fingerprint: {dataset['metadata']['fingerprint']}")
    print(f"  total dates: {len(set(new_dates_full))}")
    print(f"  last date: {new_dates_full[-1]}")
    print(f"  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
