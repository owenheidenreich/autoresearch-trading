"""Audit: live gate distributions and selectivity content.

Runs model inference on promote-mask bars and measures:
1. opportunity_logit distribution (the signal actually used at inference)
2. whether opportunity_logit carries selectivity signal
3. contract ranking quality by bucket (ATM/OTM, call/put, time-of-day, vol regime)
4. supervision hardness: how many bars are learnable after costs/margins

Usage:
    python3 -m v2.analysis.gate_audit
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.core.chain_data import padded_snapshot
from v2.train import TradingModel, LOOKBACK, NUM_FEATURES


def main():
    data_path = "v2/data.pt"
    model_path = "v2/models/model_candidate.pt"

    if not os.path.exists(model_path):
        model_path = "v2/models/model.pt"
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    meta = data.get("metadata", {})
    sidecar_dir = meta["chain_sidecar_dir"]
    max_contracts = int(meta["max_contracts_per_bar"])
    features = data["X"]
    dates = data["dates"]
    bar_of_day = data["bar_of_day"]
    promote_mask = data["promote_mask"].numpy()

    hp = checkpoint.get("hyperparams", {})
    model = TradingModel(
        d_model=hp.get("d_model", 96),
        depth=hp.get("depth", 3),
        n_heads=hp.get("n_heads", 4),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Collect eligible bars
    eligible = []
    for i in range(LOOKBACK, len(features)):
        if not promote_mask[i]:
            continue
        bod = int(bar_of_day[i])
        if bod < 30 or bod >= 270:
            continue
        eligible.append(i)

    print(f"Eligible promote bars: {len(eligible):,}")

    # Run inference in batches
    sidecar_cache = {}
    opp_logits = []
    max_contract_scores = []
    oracle_pnls = []
    bar_of_days = []
    best_contract_pnls = []
    n_valid_contracts = []
    oracle_ranks = []
    pred_best_pnls = []
    is_call_oracle = []
    vix_regimes = []

    BATCH = 512
    t0 = time.time()

    for batch_start in range(0, len(eligible), BATCH):
        batch_end = min(batch_start + BATCH, len(eligible))
        batch_indices = eligible[batch_start:batch_end]
        n = len(batch_indices)

        # Build windows
        windows = np.stack([features[i - LOOKBACK:i].numpy() for i in batch_indices])
        contracts_batch = np.zeros((n, max_contracts, 22), dtype=np.float32)
        labels_batch = np.full((n, max_contracts), float("nan"), dtype=np.float32)

        for j, i in enumerate(batch_indices):
            day = dates[i]
            local_bar = int(bar_of_day[i])
            if day not in sidecar_cache:
                sidecar_cache[day] = torch.load(
                    os.path.join(sidecar_dir, f"{day}.pt"),
                    map_location="cpu", weights_only=False,
                )
            sc = sidecar_cache[day]
            c, l, _ = padded_snapshot(sc, local_bar, max_contracts)
            contracts_batch[j] = c
            labels_batch[j] = l

        with torch.no_grad():
            outputs = model(
                torch.from_numpy(windows),
                torch.from_numpy(contracts_batch),
            )

        cs = outputs["contract_scores"]
        vm = outputs["valid_mask"]
        opp = outputs["opportunity_logit"]

        # Mask invalid contracts
        masked_cs = cs.clone()
        masked_cs[~vm] = -1e9

        max_cs, pred_idx = masked_cs.max(dim=-1)

        for j, i in enumerate(batch_indices):
            day = dates[i]
            local_bar = int(bar_of_day[i])
            sc = sidecar_cache[day]

            opp_logits.append(float(opp[j].item()))
            max_contract_scores.append(float(max_cs[j].item()))
            bar_of_days.append(local_bar)

            # Oracle PnL and rank
            labels_j = labels_batch[j]
            valid_j = contracts_batch[j, :, 0] > 0.5
            valid_labels = []
            for k in range(max_contracts):
                if valid_j[k] and np.isfinite(labels_j[k]):
                    valid_labels.append((k, float(labels_j[k])))

            n_valid_contracts.append(len(valid_labels))

            if valid_labels:
                sorted_by_pnl = sorted(valid_labels, key=lambda x: -x[1])
                oracle_idx = sorted_by_pnl[0][0]
                oracle_pnl = sorted_by_pnl[0][1]
                oracle_pnls.append(oracle_pnl)
                best_contract_pnls.append(oracle_pnl)

                # What did the model predict?
                pred_k = int(pred_idx[j].item())
                pred_pnl = float(labels_j[pred_k]) if pred_k < max_contracts and np.isfinite(labels_j[pred_k]) else float("nan")
                pred_best_pnls.append(pred_pnl)

                # Oracle rank of model's prediction
                rank = next((r for r, (k, _) in enumerate(sorted_by_pnl) if k == pred_k), len(sorted_by_pnl))
                oracle_ranks.append(rank)

                # Oracle side
                is_call = contracts_batch[j, oracle_idx, 2] < 0.5  # right_is_put
                is_call_oracle.append(is_call)
            else:
                oracle_pnls.append(float("nan"))
                best_contract_pnls.append(float("nan"))
                pred_best_pnls.append(float("nan"))
                oracle_ranks.append(-1)
                is_call_oracle.append(True)

            # VIX regime
            vix_idx = 14
            vix_regimes.append(float(features[i, vix_idx]))

        if batch_start % 2048 == 0:
            print(f"  {batch_start}/{len(eligible)}...")

    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s")

    # Convert to arrays
    opp = np.array(opp_logits)
    max_cs = np.array(max_contract_scores)
    oracle_pnl = np.array(oracle_pnls)
    bod = np.array(bar_of_days)
    pred_pnl = np.array(pred_best_pnls)
    ranks = np.array(oracle_ranks)
    n_valid = np.array(n_valid_contracts)
    vix = np.array(vix_regimes)
    n = len(opp)

    print(f"\n{'='*70}")
    print(f"  GATE AUDIT ({n:,} promote bars, model: {model_path})")
    print(f"{'='*70}")

    # --- 1. Gate signal distributions ---
    print(f"\n--- 1. Gate Signal Distributions ---")
    print(f"  opportunity_logit:  mean={opp.mean():.3f}  std={opp.std():.3f}  "
          f"min={opp.min():.3f}  max={opp.max():.3f}")
    print(f"  max_contract_score: mean={max_cs.mean():.3f}  std={max_cs.std():.3f}")

    # Pass rates at various thresholds
    print(f"\n  Pass rates (opportunity_logit > threshold):")
    for t in [-100, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]:
        rate = (opp > t).mean()
        print(f"    threshold {t:>6.1f}: {rate:.1%} pass ({int(rate * n):,} bars)")

    # --- 2. Does opportunity_logit carry selectivity signal? ---
    print(f"\n--- 2. Gate Selectivity Signal ---")
    valid_mask = np.isfinite(oracle_pnl)
    print(f"\n  opportunity_logit quartile analysis (oracle PnL):")
    sig_valid = opp[valid_mask]
    pnl_valid = oracle_pnl[valid_mask]
    pred_valid = pred_pnl[valid_mask]
    quartiles = np.percentile(sig_valid, [25, 50, 75])
    bins = [(-np.inf, quartiles[0]), (quartiles[0], quartiles[1]),
            (quartiles[1], quartiles[2]), (quartiles[2], np.inf)]
    for lo, hi in bins:
        mask = (sig_valid >= lo) & (sig_valid < hi)
        if mask.sum() == 0:
            continue
        q_oracle = pnl_valid[mask]
        q_pred = pred_valid[mask]
        q_pred_finite = q_pred[np.isfinite(q_pred)]
        print(f"    [{lo:>7.2f}, {hi:>7.2f}): n={mask.sum():5d}  "
              f"oracle_pnl={q_oracle.mean():.4f}  "
              f"pred_pnl={q_pred_finite.mean():.4f}" if len(q_pred_finite) else "")

    # --- 3. Contract ranking quality ---
    print(f"\n--- 3. Contract Ranking Quality ---")
    valid_ranks = ranks[ranks >= 0]
    print(f"  Model's predicted contract rank in oracle ordering:")
    print(f"    Mean rank: {valid_ranks.mean():.2f} (lower=better, 0=perfect)")
    print(f"    Median rank: {np.median(valid_ranks):.0f}")
    print(f"    Rank 0 (exact match): {(valid_ranks == 0).mean():.1%}")
    print(f"    Rank 0-2 (top 3): {(valid_ranks <= 2).mean():.1%}")
    print(f"    Mean n_valid_contracts: {n_valid[n_valid > 0].mean():.1f}")

    # Ranking by bucket
    print(f"\n  Ranking quality by time-of-day bucket:")
    time_buckets = [(30, 90, "open"), (90, 150, "mid-morning"),
                    (150, 210, "midday"), (210, 270, "afternoon")]
    for lo, hi, label in time_buckets:
        mask = (bod >= lo) & (bod < hi) & (ranks >= 0)
        if mask.sum() == 0:
            continue
        r = ranks[mask]
        op = oracle_pnl[mask & np.isfinite(oracle_pnl)]
        pp = pred_pnl[mask & np.isfinite(pred_pnl)]
        print(f"    {label:>12s} (bars {lo}-{hi}): n={mask.sum():5d}  "
              f"rank={r.mean():.2f}  oracle={op.mean():.3f}  pred={pp.mean():.3f}")

    print(f"\n  Ranking quality by VIX regime:")
    vix_buckets = [(-np.inf, -0.5, "low_vol"), (-0.5, 0.5, "normal"), (0.5, np.inf, "high_vol")]
    for lo, hi, label in vix_buckets:
        mask = (vix >= lo) & (vix < hi) & (ranks >= 0)
        if mask.sum() == 0:
            continue
        r = ranks[mask]
        op = oracle_pnl[mask & np.isfinite(oracle_pnl)]
        pp = pred_pnl[mask & np.isfinite(pred_pnl)]
        print(f"    {label:>12s}: n={mask.sum():5d}  "
              f"rank={r.mean():.2f}  oracle={op.mean():.3f}  pred={pp.mean():.3f}")

    # --- 4. Model's predicted PnL vs oracle PnL ---
    print(f"\n--- 4. Model Predicted vs Oracle P&L ---")
    both_valid = np.isfinite(oracle_pnl) & np.isfinite(pred_pnl)
    if both_valid.any():
        o = oracle_pnl[both_valid]
        p = pred_pnl[both_valid]
        gap = o - p
        print(f"  Oracle PnL mean:  {o.mean():.4f} ({o.mean()*100:.1f}%)")
        print(f"  Model PnL mean:   {p.mean():.4f} ({p.mean()*100:.1f}%)")
        print(f"  PnL gap mean:     {gap.mean():.4f} ({gap.mean()*100:.1f}%)")
        print(f"  Model PnL > 0:    {(p > 0).mean():.1%}")
        print(f"  Model PnL > 4%:   {(p > 0.04).mean():.1%}")
        print(f"  Correlation(oracle, model): {np.corrcoef(o, p)[0,1]:.4f}")

    # --- 5. Supervision hardness ---
    print(f"\n--- 5. Supervision Hardness ---")
    valid_pnl = oracle_pnl[np.isfinite(oracle_pnl)]
    # Estimate spread cost at ~0.8% round trip for typical trade
    spread_cost = 0.008
    net_pnl = valid_pnl - spread_cost
    print(f"  Bars with oracle PnL > 0 (raw):     {(valid_pnl > 0).mean():.1%}")
    print(f"  Bars with oracle PnL > 0 (net cost): {(net_pnl > 0).mean():.1%}")
    print(f"  Bars with oracle PnL > 4% (net):     {(net_pnl > 0.04).mean():.1%}")
    print(f"  Bars with oracle PnL > 10% (net):    {(net_pnl > 0.10).mean():.1%}")

    # Margin analysis
    both_labels = np.isfinite(oracle_pnl) & np.isfinite(pred_pnl)
    if both_labels.any():
        op_valid = oracle_pnl[both_labels]
        pp_valid = pred_pnl[both_labels]
        margin = op_valid - pp_valid
        print(f"\n  Oracle-vs-model PnL margin distribution:")
        print(f"    margin < 1%:  {(margin < 0.01).mean():.1%} (model ≈ oracle)")
        print(f"    margin 1-5%:  {((margin >= 0.01) & (margin < 0.05)).mean():.1%}")
        print(f"    margin 5-10%: {((margin >= 0.05) & (margin < 0.10)).mean():.1%}")
        print(f"    margin > 10%: {(margin >= 0.10).mean():.1%} (model far from oracle)")

    # --- 6. Would opportunity thresholding help? ---
    print(f"\n--- 6. Opportunity thresholding ---")
    print(f"  If we use opportunity_logit > T:")
    for t in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        pass_mask = (opp > t) & np.isfinite(pred_pnl)
        if pass_mask.sum() == 0:
            print(f"    T={t:.1f}: 0 trades")
            continue
        passing_pred = pred_pnl[pass_mask]
        passing_oracle = oracle_pnl[pass_mask]
        tpd = pass_mask.sum() / max(len(set(dates[eligible[j]] for j in range(n) if pass_mask[j])), 1)
        print(f"    T={t:.1f}: {pass_mask.sum():5d} trades ({pass_mask.mean():.1%})  "
              f"pred_pnl={passing_pred.mean():.4f}  oracle_pnl={passing_oracle[np.isfinite(passing_oracle)].mean():.4f}")

    print(f"\n{'='*70}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
