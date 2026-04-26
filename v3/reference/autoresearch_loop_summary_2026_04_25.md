---
date: 2026-04-25
parent: oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md
status: COMPLETE — Karpathy-style autoresearch loop on the L3 oracle exit-timing problem; multiple hypotheses tested and falsified; no deployable lift found over the existing oracle
---

# Autoresearch Loop on L3 Oracle Exit Timing — Final Summary

## Setup

User: "run an autonomous research loop based on autoresearch from karpathy"
plus: "ensure no look-ahead bias", "commit after each change and roll back if
it doesn't work rather than manually editing the code", agenda items 1-3 from
the prior plan.

Discipline applied:
- Branch `research/relaxed-oracle-target` for all hypothesis testing
- Look-ahead audit before any code change
- Commit code change → test → keep or `git revert`, never silent edits
- Falsifications documented as cleanly as breakthroughs

## Look-ahead audit (before any change)

L3 oracle features at each bar (state_features + trade_state):
  - `ds.X_sim[abs_idx]` — past+current scalars; computed in v2 pipeline from
    rolling-backward windows. ✓ no leak
  - `trade_state` — bars-since-entry, current_pnl, mfe-so-far, mae-so-far,
    mfe_bar_age, direction. All computed from current+past observations. ✓
  - `target = (current_pnl >= suffix_max[i])` uses future info but only as
    TRAINING LABEL. Model sees only features at inference. ✓ no leak

The oracle pipeline is structurally correct. New hypotheses preserved this
structure.

## Hypotheses tested

### H1: Relax exit target to `current_pnl >= 0.85 * suffix_max`

Hypothesis: too-aggressive exit target causes early-exit on winners; relaxing
should let winners run longer.

**FALSIFIED at upper bound (5-min CPU test, no oracle build):**

```
threshold       perfect-oracle PF   perfect-oracle sum
1.00 (orig)     92.5                $56,802 (ceiling)
0.85 (H1)       84.6                $51,908   ← LOWER ceiling
0.50            52.3                $31,857   ← much LOWER
```

The framing was wrong. Relaxing the threshold makes the perfect predictor
exit EARLIER (at first crossing of 85% peak), capturing LESS of the
available pnl. The strict target is structurally optimal. The actual
problem is the model can't predict the strict target reliably (real PF
1.89 vs ceiling 92.5 = ~2% capture).

Reverted via `git revert` (10f1822 → 3a9f270).

### H2: Regret-weighted training (cost-sensitive sample weights)

Hypothesis: weight each training bar by `max($50, $regret)` where regret
is the dollar cost of misclassifying that bar. Make the model focus on
high-cost decisions.

Code changes (committed `14e9f02`):
- Compute per-bar regret in `_build_trade_data` (forward info, label-only)
- Modify `_flatten_to_rows` to also return weights
- Pass `sample_weight` to `HistGradientBoostingClassifier.fit()`

Built H2 oracle for seed 42 (~60 min CPU).

**Initially looked promising on 5-trade FW subset** (PF 0.51 → 2.35), but
**FALSIFIED on full 359-trade OOS (seed 42)**:

```
                  orig          H2          delta
PF                2.195         2.032        -7%
sum               $89,757       $60,830     -$28,927  (-32%)
mean exit_bar     156.2         136.7       -19.5 bars

Per-side:
  calls (n=140):  PF 2.04 → 1.40   ← H2 hurts calls badly
  puts  (n=219):  PF 2.30 → 2.58   ← H2 helps puts modestly

Per-window losses on H2:
  W01: -$16,832    W06: -$10,618    W03: -$4,202
  W07: -$4,351     W08: -$1,946
```

Reverted (14e9f02 → 4ab7292).

**Lesson**: 5-trade hold-out is too small. Per-side asymmetry (puts respond
to weighting, calls don't) is interesting but didn't yield a net positive.

### Side experiment: trail-stop variants on FW trajectories

Built per-bar pnl trajectories using v3/oracles/opportunity._build_contract_paths.
Tested fixed-bar exits, simple trail-stop rules, and combined rules.

With **proper hybrid_live_utility scoring** (matches reported labels exactly):

```
rule                                    PF       sum       mean
oracle alone (baseline)                1.890     $12,189    $230
fixed bar+180 (best fixed)             1.276     $7,246     $137
trail 25% (best trail alone)           0.537     -$8,345    -$157
min(oracle, trail 50%, min+60)         1.959     $12,362    $233   ← marginal +0.07 PF
max(oracle, trail %) [HINDSIGHT]       2.91      $18,950    $358   ← undeployable
```

**Findings**:
- Trail-stop alone is much WORSE than oracle (theta + reversion on 0DTEs)
- The deployable `min(oracle, trail-stop)` rule gives at best a marginal
  +0.07 PF lift — within noise on 53 trades
- The hindsight `max(oracle, trail)` upper bound shows +54% PF lift IS
  available per-trade, but no entry-time signal cleanly picks the right
  rule per trade (gate retraction earlier in session)

## Audit-level findings (descriptive, no leakage)

From `oracle_exit_timing_audit.csv` on 1717 trades:
- Median exit_bar = 150 (later than time-stop's bar 120, NOT "early-exit"
  in absolute terms; the issue is it's the wrong bar)
- Capture-of-best by best_exit_pnl bucket:
  ```
  $0-200:    -1.43    (oracle BLOWS UP small winners)
  $200-500:  -0.49
  $500-1000: +0.17
  $1000-5000: +0.38
  $5000+:    +0.59
  ```
- 41% of winnable trades produce negative pnl or capture <50% of available

## Honest verdict

**The L3 oracle's exit timing is approximately Pareto-optimal among
deployable rules with the current feature set and HistGB model class.**

What I tested that failed:
- Relaxed exit target (H1) — falsified at upper bound
- Regret-weighted training (H2) — small-sample fluke; falsified on full OOS
- Pure trail-stop — much worse than oracle
- min(oracle, trail) deployable combo — marginal noise-level lift

What's **theoretically available** but undeployable from current features:
- max(oracle, trail) per trade ceiling: PF 2.91 (+54% over oracle)
- Per-trade peak ceiling: PF 92 (+4783%)

What would actually move the needle (not done in this session):
1. **Sequence model (LSTM/transformer)** on per-bar pnl trajectory.
   The current model uses point-in-time scalar features; a sequence
   model could learn pattern recognition (e.g., "the way pnl curved
   over the last 10 bars suggests reversal").
2. **Train per-side oracles** (separate calls/puts models). H2 showed
   per-side asymmetry; explicit per-side training might capture this.
   ~50 min × 2 sides × 5 seeds = 8 hours.
3. **Add features**: realized vol over last N bars, intraday spread
   evolution, MFE-since-running-peak. These could give the model
   stronger reversal signals.
4. **Different model class**: XGBoost with custom asymmetric loss (e.g.,
   focal_loss with class weights) would let us shape the precision-recall
   tradeoff explicitly.

For tonight, the right answer is: **stop tweaking; the oracle is right
enough**. Future sessions should pursue the sequence-model or per-side
training paths if exit-timing improvement remains a priority.

## Files

```
v3/reference/autoresearch_loop_summary_2026_04_25.md  ← this file
scripts/research_relaxed_target_upper_bound.py        # H1 falsification
scripts/research_eval_h2_oracle.py                    # H2 evaluation
scripts/research_trail_stop_hybrid_live.py            # trail-stop with proper scoring
v3/artifacts/research/h2_seed42_eval.json             # H2 evidence
v3/artifacts/simulated_l3_oracle_*_seed42_h2regret.npz  # H2 oracle (kept for record)
```

H1 and H2 code changes were reverted via `git revert`. Branch
`research/relaxed-oracle-target` contains the trail.

## Cost summary (this session)

- ~5 min CPU: H1 upper-bound falsification
- ~60 min CPU: H2 oracle build (seed 42 only)
- ~5 min CPU: trail-stop and combined-rule tests
- ~5 min CPU: trajectory builder + audit
- Total: ~75 min CPU. **$0 GPU. No money spent.**

## Branch state

```
research/relaxed-oracle-target:
  4ab7292 Revert "H2: regret-weighted L3 oracle training"
  14e9f02 H2: regret-weighted L3 oracle training
  533c10e H1 FALSIFIED: relaxed exit target hurts upper-bound PF
  3a9f270 Revert "H1: relax L3 oracle exit target to current >= 0.85 * suffix_max"
  10f1822 H1: relax L3 oracle exit target to current >= 0.85 * suffix_max

main / pre-experiment baseline at:
  a7d4584 v3 honest research: oracle exit-timing audit + trail-stop rule prototype
```

The branch can be merged into main (the reverts make it net-zero on code
changes, only adds research artifacts and documentation).
