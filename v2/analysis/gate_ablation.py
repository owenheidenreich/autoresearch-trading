"""Gate ablation matrix: does the live gate carry selectivity?

Same checkpoint, same promote-mask data, live-gate threshold sweep.
Measures P&L of the model's predicted contract at bars that pass each gate.

Modes:
  A. opportunity_logit only (current inference path)
  B. no gate (trade every eligible bar)

For each mode × threshold, reports:
  - pass rate, trades/day estimate
  - mean predicted PnL (what the model actually picks)
  - mean oracle PnL (what the oracle would pick)
  - win rate of model's picks
  - ranking quality (mean oracle rank of model's choice)

Also: quantifies corrupted contract rows (solver-failed: valid=1 but Greeks=0)
and their impact on z-scoring and ranking.

Usage:
    python3 -m v2.analysis.gate_ablation
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.core.chain_data import padded_snapshot
from v2.train import LOOKBACK, TradingModel, _checkpoint_gate_arch, _checkpoint_uses_linear_heads


def main():
    data_path = "v2/data.pt"
    model_path = "v2/models/model_candidate.pt"
    if not os.path.exists(model_path):
        model_path = "v2/models/model.pt"

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
        linear_score_heads=_checkpoint_uses_linear_heads(checkpoint),
        num_features=int(hp.get("num_features", checkpoint["model_state_dict"]["input_proj.weight"].shape[1])),
        gate_arch=_checkpoint_gate_arch(checkpoint),
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

    # --- Run inference ---
    sidecar_cache = {}
    opp_logits = []
    pred_pnls = []
    oracle_pnls = []
    oracle_ranks = []
    bar_of_days = []
    n_valids = []
    n_corrupted_rows = []  # valid=1 but all Greeks=0
    vix_vals = []
    day_labels = []

    BATCH = 512
    t0 = time.time()

    for batch_start in range(0, len(eligible), BATCH):
        batch_end = min(batch_start + BATCH, len(eligible))
        batch_indices = eligible[batch_start:batch_end]
        n = len(batch_indices)

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

        masked_cs = cs.clone()
        masked_cs[~vm] = -1e9
        max_cs, pred_idx = masked_cs.max(dim=-1)

        for j, i in enumerate(batch_indices):
            day = dates[i]
            local_bar = int(bar_of_day[i])

            opp_logits.append(float(opp[j].item()))
            bar_of_days.append(local_bar)
            day_labels.append(day)
            vix_vals.append(float(features[i, 14]))

            # Contract quality audit
            cf = contracts_batch[j]
            valid = cf[:, 0] > 0.5
            n_valid = int(valid.sum())
            n_valids.append(n_valid)

            # Corrupted rows: valid=1 but iv(7)=0 AND delta(8)=0 AND gamma(9)=0
            n_corrupt = 0
            for k in range(max_contracts):
                if valid[k]:
                    iv = cf[k, 7]
                    delta = cf[k, 8]
                    gamma = cf[k, 9]
                    if abs(iv) < 1e-6 and abs(delta) < 1e-6 and abs(gamma) < 1e-6:
                        n_corrupt += 1
            n_corrupted_rows.append(n_corrupt)

            # Oracle and prediction
            labels_j = labels_batch[j]
            valid_labels = []
            for k in range(max_contracts):
                if valid[k] and np.isfinite(labels_j[k]):
                    valid_labels.append((k, float(labels_j[k])))

            if valid_labels:
                sorted_by_pnl = sorted(valid_labels, key=lambda x: -x[1])
                oracle_pnls.append(sorted_by_pnl[0][1])

                pred_k = int(pred_idx[j].item())
                pred_pnl = float(labels_j[pred_k]) if pred_k < max_contracts and np.isfinite(labels_j[pred_k]) else float("nan")
                pred_pnls.append(pred_pnl)

                rank = next((r for r, (k, _) in enumerate(sorted_by_pnl) if k == pred_k), len(sorted_by_pnl))
                oracle_ranks.append(rank)
            else:
                oracle_pnls.append(float("nan"))
                pred_pnls.append(float("nan"))
                oracle_ranks.append(-1)

    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s")

    # Convert
    opp = np.array(opp_logits)
    pred = np.array(pred_pnls)
    oracle = np.array(oracle_pnls)
    ranks = np.array(oracle_ranks)
    bod = np.array(bar_of_days)
    n_valid = np.array(n_valids)
    n_corrupt = np.array(n_corrupted_rows)
    vix = np.array(vix_vals)
    days = np.array(day_labels)
    n_total = len(opp)
    n_days = len(set(days))

    print(f"\n{'='*75}")
    print(f"  GATE ABLATION MATRIX ({n_total:,} bars, {n_days} days)")
    print(f"{'='*75}")

    # --- Corrupted contract rows ---
    print(f"\n--- Corrupted Contract Rows (valid=1, all Greeks=0) ---")
    print(f"  Bars with ≥1 corrupted row:  {(n_corrupt > 0).sum():,} ({(n_corrupt > 0).mean():.1%})")
    print(f"  Mean corrupted per bar:       {n_corrupt.mean():.2f} (of {n_valid.mean():.1f} valid)")
    print(f"  Max corrupted in one bar:     {n_corrupt.max()}")
    if (n_corrupt > 0).any():
        corrupt_bars = n_corrupt > 0
        clean_bars = n_corrupt == 0
        r_corrupt = ranks[corrupt_bars & (ranks >= 0)]
        r_clean = ranks[clean_bars & (ranks >= 0)]
        p_corrupt = pred[corrupt_bars & np.isfinite(pred)]
        p_clean = pred[clean_bars & np.isfinite(pred)]
        print(f"  Ranking on corrupt bars:  mean rank {r_corrupt.mean():.2f}, pred_pnl {p_corrupt.mean():.4f}")
        print(f"  Ranking on clean bars:    mean rank {r_clean.mean():.2f}, pred_pnl {p_clean.mean():.4f}")

    # --- Gate ablation matrix ---
    print(f"\n--- Gate Ablation Matrix ---")
    print(f"  {'Mode':<30s} {'Threshold':>9s} {'Pass%':>6s} {'TPD':>5s} "
          f"{'PredPnL':>8s} {'WinRate':>7s} {'Rank':>5s} {'OracPnL':>8s}")
    print(f"  {'-'*30} {'-'*9} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*5} {'-'*8}")

    valid_pred = np.isfinite(pred)
    valid_oracle = np.isfinite(oracle)
    valid_both = valid_pred & valid_oracle & (ranks >= 0)

    for mode_name, signal in [
        ("A: opportunity_logit", opp),
    ]:
        for threshold in [-100, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
            mask = (signal > threshold) & valid_both
            if mask.sum() < 10:
                continue
            pass_rate = mask.mean()
            tpd = mask.sum() / n_days
            mpnl = pred[mask].mean()
            wr = (pred[mask] > 0).mean()
            mr = ranks[mask].mean()
            opnl = oracle[mask].mean()
            print(f"  {mode_name:<30s} {threshold:>9.1f} {pass_rate:>5.1%} {tpd:>5.1f} "
                  f"{mpnl:>+8.4f} {wr:>6.1%} {mr:>5.1f} {opnl:>+8.4f}")

    # Mode B: no gate
    mask_all = valid_both
    tpd = mask_all.sum() / n_days
    mpnl = pred[mask_all].mean()
    wr = (pred[mask_all] > 0).mean()
    mr = ranks[mask_all].mean()
    opnl = oracle[mask_all].mean()
    print(f"\n  {'B: no gate':<30s} {'all':>9s} {mask_all.mean():>5.1%} {tpd:>5.1f} "
          f"{mpnl:>+8.4f} {wr:>6.1%} {mr:>5.1f} {opnl:>+8.4f}")

    # --- Ranking learnability by bucket ---
    print(f"\n--- Ranking Learnability by Bucket ---")

    def _bucket_stats(name, mask):
        m = mask & valid_both
        if m.sum() < 30:
            return
        r = ranks[m]
        p = pred[m]
        o = oracle[m]
        wr = (p > 0).mean()
        print(f"  {name:<35s} n={m.sum():>5d}  rank={r.mean():>5.2f}  "
              f"pred={p.mean():>+.4f}  wr={wr:.1%}  oracle={o.mean():>+.4f}")

    print(f"\n  By bar-of-day:")
    for lo, hi, label in [(30, 60, "30-59 (first 30m)"), (60, 105, "60-104 (old window)"),
                           (105, 150, "105-149"), (150, 210, "150-209"),
                           (210, 270, "210-269 (last hour)")]:
        _bucket_stats(f"  bars {label}", (bod >= lo) & (bod < hi))

    print(f"\n  By VIX regime:")
    for lo, hi, label in [(-5, -0.5, "low vol"), (-0.5, 0.5, "normal"),
                           (0.5, 5, "high vol")]:
        _bucket_stats(f"  vix {label}", (vix >= lo) & (vix < hi))

    print(f"\n  By contract count:")
    for lo, hi, label in [(1, 10, "sparse (1-9)"), (10, 20, "moderate (10-19)"),
                           (20, 30, "typical (20-29)"), (30, 999, "dense (30+)")]:
        _bucket_stats(f"  contracts {label}", (n_valid >= lo) & (n_valid < hi))

    print(f"\n  By corruption level:")
    _bucket_stats("  no corrupted rows", n_corrupt == 0)
    _bucket_stats("  1+ corrupted rows", n_corrupt > 0)
    _bucket_stats("  3+ corrupted rows", n_corrupt >= 3)
    _bucket_stats("  5+ corrupted rows", n_corrupt >= 5)

    print(f"\n  By oracle PnL bucket:")
    for lo, hi, label in [(0, 0.04, "<4% (marginal)"), (0.04, 0.10, "4-10%"),
                           (0.10, 0.20, "10-20%"), (0.20, 0.40, "20-40%"),
                           (0.40, 999, ">40% (strong)")]:
        _bucket_stats(f"  oracle {label}", (oracle >= lo) & (oracle < hi))

    print(f"\n{'='*75}")
    print(f"  ABLATION COMPLETE")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
