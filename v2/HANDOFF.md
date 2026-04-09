# Handoff: Sessions 6-7 (Walk-Forward + Domain-Informed)

Read this, then `v2/program.md`, then `v2/COMMANDS.md`.

## Current Best: exp_066, WF Score 5.523

```
Per-fold:  [5.17, 5.59, 5.66, 5.61, 5.59]  std=0.18
Trades:    431 across 175/300 test days
Fold 4:    Score 5.586, WR 77.8%, PF 10.53, 54 trades
Direction: Calls 78% WR, Puts 76% WR (gap closed)
Hold:      Calls avg 21 bars, Puts avg 16 bars
```

## Config (exp_066, in train.py now)

```
Lookback: 30, d_model: 64, depth: 3, dropout: 0.05
LR: 5e-4, batch: 2048, weight_decay: 0.03
Asymmetric loss: calls 4x, puts 6x (direction-asymmetric)
Sample weighting: 1 + |max_pnl|
RISK_W: 0.5, Huber delta: 0.5
Hold targets: calls 30 bars, puts 20 bars (0.67x multiplier)
Hold_frac normalization: / hold_hi (250), aligned with replay
Gate threshold: 0.50
```

## What Changed in Sessions 6-7 (16 experiments, exp_052-067)

**Session 6 (walk-forward baseline):** Single-split score 5.67 dropped to walk-forward 5.19. Revealed the model was fragile across market regimes. weight_decay 0.03 and asymmetric 4x improved to 5.31. Found and fixed hold_frac training/replay mismatch (was off by 37%).

**Session 7 (domain-informed):** Cross-referenced trade data with domain knowledge (Pickles, Sinclair, Douglas, Elder, 0DTE microstructure). Three changes worked:

1. **Direction-asymmetric loss** (puts 6x, calls 4x): Put predictions were noisier. Closed the WR gap from 16pp to 2pp. Sinclair: puts face steeper variance premium headwind.
2. **RISK_W 0.3 -> 0.5**: Risk head gets 33% of loss signal. Improved stop calibration, all folds broke above 5.0.
3. **Direction-dependent hold** (puts 20 bars, calls 30): Biggest single gain (+0.100). 0DTE theta decay punishes puts more for long holds.

## Infrastructure Changes

- **model_candidate.pt**: deploy.sh now downloads to `model_candidate.pt`, never overwrites `model.pt` directly. After reading the score:
  - KEEP: `python v2/ops/model_manage.py keep` (promotes candidate)
  - REVERT: `python v2/ops/model_manage.py revert` (discards candidate)
- **model_best.pt**: Canonical best model. Analysis tools default to this.
- **lease_check.py**: `python v2/ops/lease_check.py` between experiments. Auto-funds if < 1hr remaining.
- **CSV export**: `python -m v2.plot_trades` now produces trades.html, equity.html, AND trades.csv in v2/output/.
- **Output dir**: v2/output/ for all viewable artifacts.

## Trade Profile (what the model is doing)

54 trades on fold 4 promote (60 days). 12 losers:
- **8 BAD ENTRY** (0% MFE): immediately went wrong. Unfilterable noise (~10% of trades, matches Douglas's prediction).
- **4 HAD EDGE** (MFE 11-43%): direction was right but reversed. Exit timing issue.
- **STOP_LOSS**: 6 trades, 0% WR, avg -$851. Dominated by Feb 20 (-$3,338 single trade).
- **TAKE_PROFIT**: 20 trades, 100% WR, avg $1,110. The big winners.
- **MAX_HOLD**: 16 trades, 88% WR. Dramatically improved from 55% before direction-dependent holds.

## What to Try Next (ranked hypotheses from research phase)

**H3: Raw VIX level as feature (HIGHEST REMAINING PRIORITY)**
- VIX is z-scored (60-day rolling), so the model can't see absolute VIX level.
- Sinclair: VIX < 20 = 28% variance premium overpay for long options. VIX > 50 = premium inverts.
- The model is blind to the regime that determines how expensive it is to buy options.
- Requires: either adding raw VIX to feature pipeline (compute_features.py rebuild) or deriving it from existing z-scored data in train.py.

**New features: MACD-histogram slope, Force Index**
- Elder's #1 signal (MACD slope) and price+volume combo (Force Index) are not in the 47 features.
- Would require feature pipeline rebuild.

**What NOT to try (exhausted directions):**
- Hyperparameter tweaks (dropout, d_model, LR, batch): all failed or were no-ops
- Gate threshold changes: 0.55 caused gate failure on all folds
- Confidence-modulated gate: was a no-op (multiplicative boost doesn't change threshold crossing)
- Extended training time: best epoch is always 1-5, model converges immediately
- Stronger sample weighting: 2x was too aggressive

## Loop Discipline

```
1. Edit train.py / policy.py, commit
2. deploy.sh run_one exp_NNN
3. Read score from stdout
4. KEEP:   python v2/ops/model_manage.py keep
   REVERT: git checkout HEAD~1 -- v2/train.py v2/core/policy.py
           python v2/ops/model_manage.py revert
5. Log to results.tsv
6. python v2/ops/lease_check.py
7. python -m v2.plot_trades          (-> v2/output/trades.html, equity.html, trades.csv)
8. python v2/plot_progress.py        (-> v2/output/progress.png)
9. Read trades.csv, form hypothesis citing trade data + domain knowledge
10. Repeat
```
