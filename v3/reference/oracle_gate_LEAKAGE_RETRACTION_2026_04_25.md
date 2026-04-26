---
date: 2026-04-25
parent: oracle_gate_research_trail_2026_04_25.md
status: RETRACTION — the +24% to +180% PF lifts in iter 4-8 were label leakage; the gate does not work with clean features
---

# Retraction: Oracle Gate Was Reward-Hacking

## What I claimed

Iter 4 of the oracle-gate research trail reported:

```
Forward-walk (53 trades, 42 unseen days):
  Always oracle:        PF 1.890, sum +$12,189
  Trained gate:         PF 2.346, sum +$18,442  (+24% PF, +51% $)
  100% beat-baseline rate across 20 random seeds
  81.1% classification accuracy on hold-out
```

Iter 8 reported even larger lifts (PF 5.30+) when composing skip+gate.

**These results are FALSE — they were the classifier reading oracle labels.**

## What the leak was

The 16 features used by the classifier included:

```python
"side_margin_raw"        # = oracle_call - oracle_put  (best_forward_pnl_*)
"time_stop_margin_raw"   # = realized ts_call - realized ts_put
```

Both are computed from realized future PnL in `v3/layer2/common.py`:

```python
# line 330: side_margin_raw uses bar.labels.best_forward_pnl_call/put
side_margin_raw = float(oracle_call - oracle_put)

# line 339: time_stop_margin_raw uses _directional_time_stop_pnl which is realized
time_stop_margin_raw = float(ts_call - ts_put)
```

These values are LABELS — they encode the actual outcome of the trade. A
classifier using them effectively reads "the answer" before deciding what
to do.

## What's the honest result

After stripping `time_stop_margin_raw` and `side_margin_raw` from the
feature list (14 clean features remaining), retrained with same setup:

```
Threshold sweep on FW with clean features:
  thr=0.30: PF 1.890 (no change — keeps oracle on all)
  thr=0.40: PF 1.598 (worse than baseline)
  thr=0.45: PF 1.598 (worse)
  thr=0.50: PF 1.736 (worse)
  thr=0.55: PF 1.715 (worse)
  thr=0.60: PF 1.619 (worse)
  thr=0.70: PF 1.595 (worse)

Stability across 20 GB seeds at threshold 0.50:
  PF: 1.736 (constant)
  beat-baseline rate: 0%
```

**The gate does not beat always-oracle when label-leaking features are
removed.** The +24% lift was entirely reward hacking.

## Honest top-feature ranking (clean)

```
iv_percentile                  : 0.192
first15_range_pct              : 0.163
atm_iv                         : 0.143
vwap_slope                     : 0.116
sigma_pos                      : 0.081
volume_ratio                   : 0.069
bars_since_break_above_first15 : 0.047
vix                            : 0.040
pred_stopout_risk              : 0.035
pred_clean_entry_prob          : 0.035
pred_win_prob                  : 0.029
decision_margin                : 0.026
bars_since_break_below_first15 : 0.023
late_window_40_120_flag        : 0.001
```

These are real entry-time features but their combined signal is too weak
to beat the always-oracle baseline.

## What stays valid

The other findings from the session DO stand:

1. **Forward walk with oracle restored: cross-seed PF 1.70.** This is
   measured pnl, not classifier output. Still correct.
2. **Seed 46 alone is the strongest forward seed (PF 3.89).** Measured.
3. **K=2 consensus filter on offline OOS lifts PF.** Measured on real
   chosen_trades; unrelated to the gate.
4. **The L3 oracle has structural early-exit bias** — observed at trade
   level, not via classifier.

## What's invalidated

1. The trained gate at `v3/artifacts/oracle_gate/gate_classifier.pkl`
   (deleted).
2. The "+24% PF" claim on forward walk.
3. The "+180% PF" composition claim from Iter 8.
4. The `time_stop_margin_raw` as "the strongest predictor" finding —
   it was merely the leakiest.

## Lessons

1. **Always audit features for label provenance before training.** Any
   feature with a name like `*_margin_raw` or `*_value_raw` that is
   computed in the same place as labels is suspicious.
2. **Massive PF jumps deserve immediate skepticism.** +24% on a clean
   hold-out is unusually large; investigation found the cause.
3. **The user's instinct was correct.** When numbers look too good,
   audit the feature pipeline.

## Files updated

```
v3/live_shadow/oracle_gate.py    # FEATURES list updated; docstring retracts
v3/artifacts/oracle_gate/gate_classifier.pkl    # DELETED
v3/reference/oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md  ← this file
v3/reference/oracle_gate_research_trail_2026_04_25.md      # see notice below
```

The research trail document remains as a record of how the leak occurred
and how it was caught, but the headline numbers should be read with this
retraction in mind.

## Honest deployment recommendation (post-retraction)

For Monday's live shadow, the previous **non-classifier** findings still
hold:

1. **Run the L3 oracle** — restoration lifts forward-walk PF 1.14 → 1.70
   (this is measured PnL, not classifier output).
2. **Single-seed-46 deployment** — PF 3.89 / DD 10.2% / 19 trades. Still
   the strongest empirical seed.
3. **5-seed K=1 ensemble** — cross-seed mean PF 1.70 with oracle.
4. **DO NOT use the runtime gate.** Without leaky features it adds no
   value.

The actual exit-improvement work needs to come from a different angle:
* Train the L3 oracle with a different loss (e.g., maximize pnl rather
  than minimize drawdown) — but that's a substantive retrain, not a
  thin-gate hack.
* Examine MFE/MAE-based runtime extension that doesn't require a
  classifier — uses observed in-trade pnl to decide whether to extend.
* Accept that the oracle's current behavior is approximately correct
  (right 65% of the time on OOS) and don't expect a costless lift from
  pure entry-time gating.
