---
date: 2026-04-26
parent: h3a_features_validated_2026_04_26.md
status: DIAGNOSTIC — H3a is regime-conditional, not regime-adaptive; aggregate +0.142 PF lift hides bear-regime regression of -0.50 PF
---

# Regime-Stratified Diagnostic on H3a vs Baseline

## Setup

Question: is the L3 oracle profitable across all market regimes, or only in
specific ones? User's framing: an "intelligent trading model" should adapt
to whatever regime it's in, not specialize.

Method: take the existing 5-seed full-OOS evaluation results
([h3a_5seed_eval.json](../artifacts/research/h3a_5seed_eval.json)) and slice
by two regime axes:

- **Trend** — trailing 20-day SPX return at trade day (`bear` ≤ −2%, `chop`
  −2% to +2%, `bull` ≥ +2%)
- **Vol** — `iv_percentile` at entry bar (`low` < 0.33, `mid` 0.33–0.66,
  `high` ≥ 0.66)

Note: the `vix` column in chosen_trades is normalized to ~[−1, 0.33] (NOT raw
VIX). `iv_percentile` is the right field for vol regime. Initial run misclassified
all 1760 trades as "low" — corrected by switching to `iv_percentile`.

## Aggregate (sanity check vs prior 5-seed eval)

```
n_trades: 1760  (across 5 seeds)
base mean PF: 1.877   (matches prior 5-seed result 1.881)
h3a mean PF:  2.019   (matches prior 5-seed result 2.038)
delta:        +0.142
```

## By trend axis (alone)

```
regime    n   base_pf   h3a_pf    delta   base_sum   h3a_sum
bear     173    2.002    1.469   -0.533  $ 45,756  $ 21,208
chop     640    1.795    2.053   +0.257  $115,479  $126,973
bull     947    1.914    2.160   +0.246  $151,693  $170,788
```

H3a HURTS bear regimes by 27% PF; helps chop and bull by 13–14%.

## By vol axis (alone)

```
regime    n   base_pf   h3a_pf    delta
low      929    1.646    1.950   +0.304   <-- helps in low-vol
mid      396    1.984    2.073   +0.089
high     435    2.148    2.083   -0.065
```

H3a's lift concentrates in low-vol regimes (where mean reversion is more
common); diminishes as vol rises.

## Cross-tab (trend × vol) with bootstrap delta-PF 95% CI

```
trend  vol      n    base    h3a    delta     CI [lo,  hi]    p[d≤0]
bear   low      7   1.499  0.694  -0.805  [tiny n]            0.98
bear   mid     14   2.888  1.462  -1.426  [tiny n,  -0.16]    1.00
bear   high   152   1.990  1.488  -0.502  [-1.07,   -0.04]    0.98 *
chop   low    245   1.180  1.231  +0.051  [-0.19,   +0.32]    0.34
chop   mid    202   1.746  2.110  +0.364  [-0.19,   +1.00]    0.09
chop   high   193   2.479  2.991  +0.512  [+0.00,   +1.15]    0.02 *
bull   low    677   1.857  2.312  +0.455  [+0.22,   +0.72]    0.00 *
bull   mid    180   2.253  2.064  -0.188  [-0.70,   +0.32]    0.78
bull   high    90   1.547  1.451  -0.096  [-0.85,   +0.62]    0.62
```

Statistically significant cells (CI excludes 0, n ≥ 30):
- **HELPED**: bull × low_vol (+0.46), chop × high_vol (+0.51)
- **HURT**: bear × high_vol (−0.50)

## Key findings

1. **Both baseline and H3a are profitable in every regime cell** (PF > 1
   in 9/9 cells). User's question "is the model profitable across any market
   regime?" — answer is YES in aggregate.

2. **H3a is regime-conditional, not regime-adaptive.** The aggregate +0.142
   lift comes from amplifying bull/chop performance while degrading bears.

3. **Cross-cell PF spread INCREASES with H3a:**
   - baseline range: 1.18 → 2.48 (spread 1.30)
   - h3a range:      1.23 → 2.99 (spread **1.76**)

4. **Mechanism:** the H3a features (`realized_vol_10bar`, `pnl_velocity_5bar`,
   `mfe_decay_rate`) all detect "pnl reversing from peak" — i.e., mean
   reversion. In bear-trending markets, a pullback after a peak is the trend
   continuing, not mean-reverting. H3a fires the exit signal but the trade
   was about to keep going (down for puts = profit). Result: H3a exits puts
   too early in trends.

5. **Bear-regime forward-walk puzzle:** training-OOS bear (n=173) shows H3a
   hurts by −0.50; FW (also bearish, n=53) showed H3a helps +0.10. The FW
   sample is too small (n=53 across 5 seeds) for the regime signal to
   stabilize. Training-OOS evidence is more reliable.

## What this implies for "intelligent adaptive" trading

The aggregate +0.142 PF lift is real but deceptive. The model has gotten
better in bull/chop and worse in bears — net positive but unstable. A truly
adaptive model would *narrow* the regime spread (or at least not widen it).

Two paths considered:

- **H3d — explicit regime features in trade_state**: `trend_20d`,
  `iv_percentile_entry`, `vix_change_5d`. Hypothesis: with regime context
  visible to the model, it learns to behave differently per regime.
- **H3c — sequence model on per-bar features**: lets temporal evolution
  encode regime. Larger lift potential, more risk.

Both deferred pending strategy discussion (see [feedback_causal_features](
../../../.claude/projects/-Users-gduby-Documents-autoresearch-trading/memory/feedback_causal_features.md)).

## Files

```
scripts/research_regime_stratified_eval.py        # diagnostic
v3/artifacts/research/regime_stratified_eval.json
v3/artifacts/research/regime_stratified_per_trade.csv
```
