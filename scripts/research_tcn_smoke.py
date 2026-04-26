"""H3c-1 smoke test: train TCN on a single rolling window, verify convergence.

Goal: prove the TCN architecture can learn the L3 task (per-bar peak
detection) on a small, well-characterized slice of data before committing
to the full 13-window × 5-seed build.

Methodology:
  - Use the existing trade-data builder from v3/layer3/common.py
  - Pick a single window (W12, the last/most-data window) and seed (42)
  - Train trades from windows < W12 → val on W12 OOS
  - Verify val BCE drops below the uniform-class baseline (i.e., the model
    is doing meaningfully better than always predicting the train pos rate)

Pass/fail:
  PASS — val BCE < uniform-class baseline by ≥ 0.01 (model has learned signal)
  FAIL — val BCE plateau at uniform baseline → architecture/training is wrong
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import DEFAULT_ACTION_SURFACE_DATASET_PATH
from v3.layer3.common import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_EQUITY,
    DEFAULT_SESSION_END_BAR,
    build_trade_dataset,
    load_action_surface_candidate_trades,
)
from v3.layer3.tcn_oracle import (
    TCNTrainConfig,
    train_tcn,
    trade_data_to_tensors,
    masked_bce_loss,
)


def main():
    print("=== H3c-1 smoke test: TCN on single window ===\n")

    # 1. Load candidate trades
    print("Loading candidate trades...")
    trades, candidate_meta = load_action_surface_candidate_trades(
        DEFAULT_ACTION_SURFACE_DATASET_PATH,
        max_per_day=4,
    )
    print(f"  total candidate trades: {len(trades)}, "
          f"call share: {candidate_meta['call_share']:.3f}\n")

    # 2. Build trade datasets (windows assigned via `window_idx`)
    print("Building trade datasets (this calls _build_trade_data per trade)...")
    t0 = time.time()
    trade_data, meta = build_trade_dataset(
        trades,
        equity=DEFAULT_EQUITY,
        session_end_bar=DEFAULT_SESSION_END_BAR,
        commission=DEFAULT_COMMISSION_PER_CONTRACT,
    )
    elapsed = time.time() - t0
    print(f"  usable trades: {meta['n_trade_datasets']}, skipped: {meta['n_skipped']}")
    print(f"  build time: {elapsed:.1f}s\n")

    # 3. Pick W12 as test, train on W0-W11
    target_window = 12
    train_data = [td for td in trade_data if td["window_idx"] < target_window]
    val_data = [td for td in trade_data if td["window_idx"] == target_window]
    print(f"Window split: train={len(train_data)} (W0-W{target_window-1}), val={len(val_data)} (W{target_window})")

    if len(train_data) < 100 or len(val_data) < 50:
        print(f"  WARNING: small sample sizes; results may be noisy")

    # 4. Compute baselines
    # Uniform-class baseline: always predict positive rate
    train_pos_rate = float(np.mean([
        np.mean([p["target"] for p in td["per_bar"]])
        for td in train_data if td["per_bar"]
    ]))
    print(f"  train positive rate: {train_pos_rate:.3f}")

    # Compute BCE if model always predicts train_pos_rate (constant prediction baseline)
    import torch
    val_x, val_y, val_m = trade_data_to_tensors(val_data, max_bars=200, n_features=99)
    constant_logit = float(np.log(train_pos_rate / (1 - train_pos_rate)))
    constant_logits = torch.full_like(val_y, constant_logit)
    uniform_bce = float(masked_bce_loss(constant_logits, val_y, val_m).item())
    print(f"  uniform-class val BCE baseline (always predict train pos rate): {uniform_bce:.4f}\n")

    # 5. Train TCN
    cfg = TCNTrainConfig(
        n_features=99,
        hidden=64,
        num_blocks=4,
        kernel_size=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=30,
        early_stop_patience=5,
        max_bars=200,
        device="cpu",
        seed=42,
    )
    print(f"TCN config: hidden={cfg.hidden}, num_blocks={cfg.num_blocks}, kernel={cfg.kernel_size}")
    print(f"Training (max_epochs={cfg.max_epochs}, early_stop_patience={cfg.early_stop_patience})...")
    t0 = time.time()
    model, std, history = train_tcn(train_data, cfg, val_data=val_data)
    train_elapsed = time.time() - t0

    # 6. Report
    print(f"\nTraining done in {train_elapsed:.1f}s ({len(history['train_loss'])} epochs)\n")
    print(f"{'epoch':>5} {'train_bce':>12} {'val_bce':>12}")
    for ep, (tr, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
        marker = " ←" if vl == min(history["val_loss"]) else ""
        print(f"{ep:>5} {tr:>12.4f} {vl:>12.4f}{marker}")

    best_val = min(history["val_loss"])
    print(f"\nBest val BCE: {best_val:.4f}")
    print(f"Uniform baseline:  {uniform_bce:.4f}")
    print(f"Improvement: {uniform_bce - best_val:+.4f} (positive = TCN beat baseline)")
    print()

    if best_val < uniform_bce - 0.01:
        print("✓ PASS: TCN learned meaningful signal (val BCE > 0.01 below baseline).")
        print("  Architecture works. Proceed to Phase H3c-2 (full seed-42 build).")
        return 0
    elif best_val < uniform_bce:
        print("≈ MARGINAL: TCN slightly better than baseline but not by 0.01.")
        print("  May indicate noise or slight signal. Inspect closely; consider tuning.")
        return 1
    else:
        print("✗ FAIL: TCN did not improve over uniform baseline.")
        print("  Architecture or training is wrong. Debug before proceeding.")
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
