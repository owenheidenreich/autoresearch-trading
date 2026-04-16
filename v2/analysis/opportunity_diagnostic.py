"""Diagnostic: Is the opportunity label learnable from context features?

Analyzes the information gap between context features (what the opportunity
head sees) and the opportunity label (what it's asked to predict).

Usage:
    python3 -m v2.analysis.opportunity_diagnostic
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.train import _compute_strict_opportunity


def _compute_consensus(sc: dict, local_bar: int) -> bool:
    """True if any contract is profitable (>4%) under all 3 policies."""
    bar_ptrs = sc["bar_ptrs"]
    start = int(bar_ptrs[local_bar])
    end = int(bar_ptrs[local_bar + 1])
    if end <= start:
        return False
    labels_default = sc.get("row_labels")
    labels_short = sc.get("row_labels_short")
    labels_eod = sc.get("row_labels_eod")
    if labels_default is None or labels_short is None or labels_eod is None:
        return False
    for i in range(start, end):
        d = float(labels_default[i])
        s = float(labels_short[i])
        e = float(labels_eod[i])
        if (np.isfinite(d) and d > 0.04 and
            np.isfinite(s) and s > 0.04 and
            np.isfinite(e) and e > 0.04):
            return True
    return False


def _compute_frac_profitable(sc: dict, local_bar: int) -> float:
    """Fraction of executable contracts with PnL > 0."""
    bar_ptrs = sc["bar_ptrs"]
    start = int(bar_ptrs[local_bar])
    end = int(bar_ptrs[local_bar + 1])
    if end <= start:
        return 0.0
    labels = sc["row_labels"][start:end]
    valid = [float(l) for l in labels if np.isfinite(float(l))]
    if not valid:
        return 0.0
    return sum(1 for v in valid if v > 0) / len(valid)


def main():
    data_path = "v2/data.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    meta = data.get("metadata", {})
    sidecar_dir = meta["chain_sidecar_dir"]
    dates = data["dates"]
    bar_of_day = data["bar_of_day"]
    features = data["X"]
    train_mask = data["train_mask"].numpy()

    # Collect bar-level statistics from sidecars
    all_best_pnl = []
    all_strict = []
    all_consensus = []
    all_frac_prof = []
    all_old_label = []
    all_labelable = []
    all_context_feats = []

    sidecar_cache: dict[str, dict] = {}
    unique_dates = sorted(set(dates))
    print(f"Loading {len(unique_dates)} sidecars...")

    # Only analyze bars in the training window (bars 30-269) that are labelable
    n_bars = len(features)
    indices = np.arange(30, n_bars)  # skip first 30 for lookback
    # Use train mask to focus on training distribution
    train_indices = indices[train_mask[30:]]

    # Sample if too many bars (for speed)
    if len(train_indices) > 50000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(train_indices), 50000, replace=False)
        train_indices = train_indices[sample_idx]

    print(f"Analyzing {len(train_indices):,} training bars...")

    for count, i in enumerate(train_indices):
        if count % 10000 == 0 and count > 0:
            print(f"  {count:,}/{len(train_indices):,}...")
        day = dates[i]
        local_bar = int(bar_of_day[i])
        if day not in sidecar_cache:
            sc_path = os.path.join(sidecar_dir, f"{day}.pt")
            if not os.path.exists(sc_path):
                continue
            sidecar_cache[day] = torch.load(sc_path, map_location="cpu", weights_only=False)
        sc = sidecar_cache[day]

        if not bool(sc["bar_labelable"][local_bar]):
            continue

        best_pnl = float(sc["bar_best_pnl"][local_bar])
        strict = _compute_strict_opportunity(sc, local_bar)
        consensus = _compute_consensus(sc, local_bar)
        frac_prof = _compute_frac_profitable(sc, local_bar)
        old_label = bool(sc["bar_label_trade"][local_bar])

        all_best_pnl.append(best_pnl)
        all_strict.append(strict)
        all_consensus.append(consensus)
        all_frac_prof.append(frac_prof)
        all_old_label.append(old_label)
        all_labelable.append(True)
        all_context_feats.append(features[i].numpy())

    # Convert to arrays
    best_pnl = np.array(all_best_pnl)
    strict = np.array(all_strict, dtype=bool)
    consensus = np.array(all_consensus, dtype=bool)
    frac_prof = np.array(all_frac_prof)
    old_label = np.array(all_old_label, dtype=bool)
    context = np.stack(all_context_feats)
    n = len(best_pnl)

    print(f"\n{'='*70}")
    print(f"  OPPORTUNITY LABEL DIAGNOSTIC ({n:,} labelable training bars)")
    print(f"{'='*70}")

    # --- 1. Distribution of bar_best_pnl ---
    print(f"\n--- 1. Distribution of bar_best_pnl ---")
    pcts = [5, 10, 25, 50, 75, 90, 95]
    vals = np.percentile(best_pnl, pcts)
    for p, v in zip(pcts, vals):
        print(f"  P{p:02d}: {v:+.4f} ({v*100:+.1f}%)")
    print(f"  Mean:   {best_pnl.mean():+.4f}")
    print(f"  Std:    {best_pnl.std():.4f}")
    print(f"  >0:     {(best_pnl > 0).mean():.1%}")
    print(f"  >4%:    {(best_pnl > 0.04).mean():.1%}")
    print(f"  >10%:   {(best_pnl > 0.10).mean():.1%}")
    print(f"  >20%:   {(best_pnl > 0.20).mean():.1%}")
    print(f"  >30%:   {(best_pnl > 0.30).mean():.1%}")
    print(f"  <0:     {(best_pnl < 0).mean():.1%}")

    # --- 2. Quality breakdown of "trade" bars ---
    print(f"\n--- 2. Quality breakdown of strict-opportunity=True bars ---")
    strict_pnl = best_pnl[strict]
    if len(strict_pnl) > 0:
        print(f"  Total strict=True: {strict.sum():,} ({strict.mean():.1%} of labelable)")
        bins = [(0.04, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, 0.30), (0.30, 999)]
        for lo, hi in bins:
            frac = ((strict_pnl >= lo) & (strict_pnl < hi)).mean()
            count = ((strict_pnl >= lo) & (strict_pnl < hi)).sum()
            label = f"{lo*100:.0f}-{hi*100:.0f}%" if hi < 999 else f">{lo*100:.0f}%"
            print(f"  {label:>10s}: {frac:.1%} ({count:,} bars)")

    # --- 3. Class balance comparison ---
    print(f"\n--- 3. Class balance across label definitions ---")
    print(f"  Old (best_pnl > 4%):  {old_label.mean():.1%} trade / {1-old_label.mean():.1%} no-trade")
    print(f"  Strict opportunity:   {strict.mean():.1%} trade / {1-strict.mean():.1%} no-trade")
    print(f"  Consensus (3-policy): {consensus.mean():.1%} trade / {1-consensus.mean():.1%} no-trade")
    hi_thresh = best_pnl > 0.20
    print(f"  High threshold (>20%): {hi_thresh.mean():.1%} trade / {1-hi_thresh.mean():.1%} no-trade")

    # --- 4. Context-feature correlations ---
    print(f"\n--- 4. Context-feature correlations ---")
    print(f"  (Pearson r with each of 52 features, showing top-10 by |r|)")

    targets = {
        "strict_opp": strict.astype(float),
        "consensus": consensus.astype(float),
        "best_pnl": best_pnl,
        "frac_profitable": frac_prof,
        "high_threshold": hi_thresh.astype(float),
    }

    # Load feature names if available
    feature_names = meta.get("feature_names", [f"feat_{i}" for i in range(52)])
    if not feature_names:
        feature_names = [f"feat_{i}" for i in range(context.shape[1])]

    for target_name, target_vals in targets.items():
        print(f"\n  Target: {target_name}")
        correlations = []
        for fi in range(context.shape[1]):
            feat = context[:, fi]
            # Skip constant features
            if feat.std() < 1e-8 or np.std(target_vals) < 1e-8:
                correlations.append(0.0)
                continue
            r = np.corrcoef(feat, target_vals)[0, 1]
            correlations.append(r if np.isfinite(r) else 0.0)

        corrs = np.array(correlations)
        top_idx = np.argsort(np.abs(corrs))[::-1][:10]

        max_r = np.max(np.abs(corrs))
        mean_r = np.mean(np.abs(corrs))
        print(f"    Max |r|: {max_r:.4f}   Mean |r|: {mean_r:.4f}")
        for idx in top_idx:
            fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            print(f"    [{idx:2d}] {fname:30s}  r={corrs[idx]:+.4f}")

    # --- 5. Consensus vs strict overlap ---
    print(f"\n--- 5. Label overlap analysis ---")
    both = strict & consensus
    strict_only = strict & ~consensus
    consensus_only = ~strict & consensus
    neither = ~strict & ~consensus
    print(f"  Both strict & consensus:  {both.sum():,} ({both.mean():.1%})")
    print(f"  Strict only:              {strict_only.sum():,} ({strict_only.mean():.1%})")
    print(f"  Consensus only:           {consensus_only.sum():,} ({consensus_only.mean():.1%})")
    print(f"  Neither:                  {neither.sum():,} ({neither.mean():.1%})")

    # --- 6. Consensus bar quality ---
    if consensus.any():
        print(f"\n--- 6. Consensus=True bar quality ---")
        cons_pnl = best_pnl[consensus]
        print(f"  Count: {consensus.sum():,}")
        print(f"  Best PnL mean: {cons_pnl.mean():.4f} ({cons_pnl.mean()*100:.1f}%)")
        print(f"  Best PnL median: {np.median(cons_pnl):.4f}")
        print(f"  Frac profitable mean: {frac_prof[consensus].mean():.3f}")

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
