"""Phase 0b: Test whether opportunity_logit ordering is mis-scaled or mis-ordered.

For each confidence bucket, compute the oracle PnL of the model's chosen
contract (what would happen if we forced a trade on every bar in that bucket).
This distinguishes:
  - Mis-scaled: Top buckets have decent oracle-chosen-contract PnL but the
    gate lets through too many bad bars at high confidence
  - Mis-ordered: Top buckets genuinely pick worse contracts than lower buckets

Usage:
    python3 -m v2.analysis.bucket_ordering_test [--model path]
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import QUALITY_PARTIAL, load_sidecar_cached, padded_snapshot
from v2.core.policy import DecisionPolicy
from v2.replay import load_model_from_path
from v2.train import TradingModel


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v2/models/model.pt")
    args = parser.parse_args()

    print("Loading model and data...")
    model = load_model_from_path(args.model)
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    policy = DecisionPolicy()

    features = data["X"].numpy()
    mask = data["promote_mask"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    sidecar_dir = "v2/data_sidecars"
    max_contracts = data.get("metadata", {}).get("max_contracts_per_bar", 285)

    # Build eligible bars
    eligible = []
    for idx in range(len(mask)):
        if not mask[idx]:
            continue
        day = dates[idx]
        bod = int(bar_of_day[idx])
        if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
            continue
        eligible.append((day, idx, bod))

    # Batch inference
    lookback = 30
    gather_idx = []
    for _, g, _ in eligible:
        row = np.arange(max(0, g - lookback + 1), g + 1)
        if len(row) < lookback:
            pad = np.full(lookback - len(row), row[0])
            row = np.concatenate([pad, row])
        gather_idx.append(row)

    all_windows = features[np.stack(gather_idx)]

    snapshots = []
    for day, global_bar, local_bar in eligible:
        sc = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        snap = padded_snapshot(sc, local_bar, max_contracts)
        snapshots.append(snap)

    all_contracts = np.stack([s[0] for s in snapshots]).astype(np.float32)
    all_contract_labels = np.stack([s[1] for s in snapshots]).astype(np.float32)

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(all_windows).float()
        batch_c = torch.from_numpy(all_contracts).float()
        all_outputs = model(batch_x, batch_c)
        all_outputs = {k: v.cpu() for k, v in all_outputs.items()}

    # For each bar: extract opportunity_logit and model's chosen contract oracle PnL
    records = []
    for i, (day, global_bar, local_bar) in enumerate(eligible):
        opp_logit = float(all_outputs["opportunity_logit"][i].item())
        c_scores = all_outputs["contract_scores"][i].numpy()
        v_mask = all_outputs["valid_mask"][i].numpy().astype(bool)
        oracle_labels = all_contract_labels[i]

        # Model's best contract
        scores = c_scores.copy()
        scores[~v_mask] = -1e9
        best_row = int(np.argmax(scores)) if v_mask.any() else -1

        if best_row >= 0 and np.isfinite(oracle_labels[best_row]):
            chosen_oracle_pnl = float(oracle_labels[best_row])
        else:
            chosen_oracle_pnl = float("nan")

        # Oracle best
        ol = oracle_labels.copy()
        ol[~v_mask] = -1e9
        ol[~np.isfinite(ol)] = -1e9
        oracle_best_row = int(np.argmax(ol))
        oracle_best_pnl = float(oracle_labels[oracle_best_row]) if np.isfinite(oracle_labels[oracle_best_row]) else float("nan")

        n_valid = int(v_mask.sum())

        records.append({
            "opp_logit": opp_logit,
            "chosen_oracle_pnl": chosen_oracle_pnl,
            "oracle_best_pnl": oracle_best_pnl,
            "bar_of_day": local_bar,
            "n_valid": n_valid,
        })

    # Sort by opportunity_logit descending
    records.sort(key=lambda r: r["opp_logit"], reverse=True)
    n = len(records)

    # Define buckets
    buckets = [
        ("Top 1%", 0, max(1, n // 100)),
        ("Top 5%", 0, max(1, n // 20)),
        ("Top 10%", 0, max(1, n // 10)),
        ("10-25%", max(1, n // 10), max(1, n // 4)),
        ("25-50%", max(1, n // 4), n // 2),
        ("50-75%", n // 2, 3 * n // 4),
        ("Bottom 25%", 3 * n // 4, n),
    ]

    print(f"\nTotal eligible bars: {n}")
    print(f"\n{'='*100}")
    print("  BUCKET ORDERING TEST: Is the score mis-scaled or mis-ordered?")
    print(f"{'='*100}")
    print(f"  If model's chosen-contract oracle PnL is monotonically decreasing → mis-scaled (fixable)")
    print(f"  If top buckets have worse chosen-contract PnL than middle → mis-ordered (focal won't help)")
    print()

    hdr = (f"{'Bucket':<14} {'Count':>6} {'OppLogit':>10} "
           f"{'ChosenPnL':>10} {'ChosenWR':>9} {'ChosenPF':>8} "
           f"{'OraclePnL':>10} {'OracleWR':>9} "
           f"{'AvgBar':>7} {'nValid':>7}")
    print(hdr)
    print("-" * 100)

    for label, start, end in buckets:
        subset = records[start:end]
        logits = [r["opp_logit"] for r in subset]
        avg_logit = np.mean(logits)

        # Model's chosen contract performance (what would happen if forced to trade)
        chosen = [r["chosen_oracle_pnl"] for r in subset if np.isfinite(r["chosen_oracle_pnl"])]
        if chosen:
            chosen_avg = np.mean(chosen)
            chosen_wr = sum(1 for p in chosen if p > 0) / len(chosen) * 100
            chosen_gw = sum(p for p in chosen if p > 0)
            chosen_gl = abs(sum(p for p in chosen if p <= 0))
            chosen_pf = chosen_gw / chosen_gl if chosen_gl > 0 else float("inf") if chosen_gw > 0 else 0
        else:
            chosen_avg = chosen_wr = chosen_pf = 0

        # Oracle best (opportunity quality)
        oracle = [r["oracle_best_pnl"] for r in subset if np.isfinite(r["oracle_best_pnl"])]
        oracle_avg = np.mean(oracle) if oracle else 0
        oracle_wr = sum(1 for p in oracle if p > 0) / len(oracle) * 100 if oracle else 0

        bars = [r["bar_of_day"] for r in subset]
        avg_bar = np.mean(bars)
        n_valids = [r["n_valid"] for r in subset]
        avg_nvalid = np.mean(n_valids)

        print(f"  {label:<12} {len(subset):>6} {avg_logit:>9.4f} "
              f"{chosen_avg:>9.4f} {chosen_wr:>8.1f}% {chosen_pf:>8.3f} "
              f"{oracle_avg:>9.4f} {oracle_wr:>8.1f}% "
              f"{avg_bar:>7.1f} {avg_nvalid:>7.1f}")

    # Verdict
    print(f"\n{'='*100}")
    print("  VERDICT")
    print(f"{'='*100}")

    # Check: is top 10% chosen PnL worse than 25-50% chosen PnL?
    top10 = records[:max(1, n // 10)]
    mid = records[max(1, n // 4):n // 2]
    top10_chosen = [r["chosen_oracle_pnl"] for r in top10 if np.isfinite(r["chosen_oracle_pnl"])]
    mid_chosen = [r["chosen_oracle_pnl"] for r in mid if np.isfinite(r["chosen_oracle_pnl"])]
    top10_avg = np.mean(top10_chosen) if top10_chosen else 0
    mid_avg = np.mean(mid_chosen) if mid_chosen else 0

    if top10_avg < mid_avg:
        print(f"  ⚠ TOP 10% chosen PnL ({top10_avg:.4f}) < MIDDLE 25-50% ({mid_avg:.4f})")
        print(f"  → Score is MIS-ORDERED at the extreme. Focal loss alone may not fix this.")
        print(f"  → HARD STOP: Investigate what makes top-confidence bars toxic before proceeding.")
    else:
        print(f"  ✓ TOP 10% chosen PnL ({top10_avg:.4f}) >= MIDDLE 25-50% ({mid_avg:.4f})")
        print(f"  → Score ordering is directionally correct. Focal loss is a reasonable intervention.")
        print(f"  → Proceed to Phase 1.")

    # Also check early vs mid-session
    print(f"\n  --- Regime Split: Early (bar<60) vs Mid (60-150) vs Late (>150) ---")
    for regime_name, bar_lo, bar_hi in [("Early", 0, 60), ("Mid", 60, 150), ("Late", 150, 300)]:
        regime = [r for r in records if bar_lo <= r["bar_of_day"] < bar_hi]
        if not regime:
            continue
        rc = [r["chosen_oracle_pnl"] for r in regime if np.isfinite(r["chosen_oracle_pnl"])]
        if rc:
            wr = sum(1 for p in rc if p > 0) / len(rc) * 100
            avg = np.mean(rc)
        else:
            wr = avg = 0
        avg_logit = np.mean([r["opp_logit"] for r in regime])
        print(f"  {regime_name:<6} bars={len(regime):>5}  avg_opp_logit={avg_logit:>7.4f}  "
              f"chosen_WR={wr:>5.1f}%  chosen_avg_pnl={avg:>7.4f}")


if __name__ == "__main__":
    main()
