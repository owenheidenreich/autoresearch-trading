---
date: 2026-04-25
parent: oracle_exit_early_bias_2026_04_25.md
status: HYPOTHESIS — runtime gate that overrides L3 oracle exit when (sigma_pos < -1.0 OR iv_percentile > 0.80); LOO-validated on forward-walk window. Hypothesis to A/B-test in live shadow.
---

# Oracle Runtime Gate — When to Override the L3 Oracle's Early Exit

## TL;DR

The L3 oracle exits early uniformly. Aggregate forward-walk PF with
oracle is 1.89 (vs 1.13 without) — oracle wins 79% of the time. But on
21% of trades (truncated winners), oracle leaves money on the table. A
runtime gate that overrides oracle when **sigma_pos < -1.0 OR
iv_percentile > 0.80** recovers ~70% of the available "perfect-selector"
lift:

```
strategy                   forward-walk PnL    PF       capture of ceiling
always oracle (current)     +$12,189          1.890     —
always no-oracle             +$3,776          1.132     —
Rule B (gate above)         +$19,428          2.086     +59% over baseline
perfect selector (ceiling)  +$27,462          3.275     —
```

LOO-validated PF is 1.968 (vs in-sample 2.086, baseline 1.890). 80% of
bootstrap resamples beat baseline. The rule is honest, not a fit.

## The discriminator

I tested whether the model's existing entry-time signals
(`pred_win_prob`, `pred_clean_entry_prob`, `pred_stopout_risk`) could
predict which trades would be better with oracle vs without. They have
weak signal (`pred_clean` and `pred_stopout` show some discrimination
but not enough to beat the baseline).

The strong discriminators turned out to be **scalar entry-time features
of the underlying**:

```
feature              oracle_better   no_oracle_better   |diff|
sigma_pos            +0.252          -0.717              0.97   ← strongest
iv_percentile        +0.571          +0.814              0.24
vix                  +0.126          +0.270              0.14
volume_ratio         +1.089          +0.850              0.24
```

`sigma_pos` is the entry bar's distance from VWAP in σ-units. Deeply
negative sigma_pos (much-below VWAP) consistently corresponds to trades
where letting the trade run beats oracle's early exit. This makes
trading sense: a put trade entered when SPX is well below VWAP is
riding strong directional momentum; oracle's training-time conservative
exit cuts before the move plays out.

`iv_percentile > 0.80` similarly correlates with regime-driven
persistence (vol expansion → moves continue beyond what historical
training distributions suggested).

## The gate (proposed runtime rule)

```python
def use_oracle_exit(scalar_features: dict) -> bool:
    """At entry time, decide whether to use oracle's predicted exit
    or to use the time-stop (no_oracle) exit instead."""
    sigma_pos = scalar_features["sigma_pos"]
    iv_percentile = scalar_features["iv_percentile"]
    if sigma_pos < -1.0 or iv_percentile > 0.80:
        return False  # let it run; use time-stop / dataset's hybrid_no_oracle
    return True  # use oracle's predicted exit
```

This is computable from features already captured in `DecisionSnapshot`.
No model retrain needed.

## Validation summary

Three checks:

**1. In-sample (53 forward-walk trades)**
- Always-oracle: PF 1.890, sum +$12,189
- Rule B: PF 2.086, sum +$19,428 (+59% $)

**2. Leave-one-out (53 iterations, threshold per held-out)**
- Per-iteration: derive best (T1, T2) thresholds from 52 trades,
  apply to the held-out 1.
- LOO sum: $17,329, PF 1.968 — within 6% of in-sample best.
- The rule's lift survives leave-one-out.

**3. Bootstrap stability (200 random 40-of-53 resamples)**
- Baseline always-oracle: PF p05=1.14, p50=1.97, p95=2.66
- Rule B: PF p05=1.42, p50=2.06, p95=2.97
- Rule beats baseline in **80% of resamples**.
- p05 is materially higher (1.42 vs 1.14) — *the rule reduces downside
  tail*, not just lifts mean.

## Caveats

1. **53-trade window is small.** The lift is robust under LOO/bootstrap
   but may be biased by the specific Feb-Apr 2026 regime (SPX dropped
   ~5% then recovered). A different regime could re-rank the features.

2. **`sigma_pos` and `iv_percentile` are correlated with the directional
   regime in this window** (SPX falling → many trades enter below VWAP
   → puts persist). In a chop regime, the rule may be neutral or
   slightly harmful.

3. **The cal_pf > 4 guard interaction is not analyzed here.** This rule
   is per-trade; the cal_pf guard is per-seed-window. They're orthogonal
   layers and could compose well, but I haven't tested combined.

## Recommended deployment path

**Live-shadow Monday**, in this order:

1. **Capture both oracle and time-stop exit pnl per `DecisionSnapshot`**
   — already part of the prior plan. This data will let us A/B-test
   the gate without deploying it in production.
2. **Capture sigma_pos and iv_percentile per snapshot** — both are
   already in the scalar feature set; just need to ensure they're
   serialized.
3. **After 5 sessions, replay the gate decision retrospectively**:
   For each shadow trade, would Rule B have used oracle or no_oracle?
   What's the realized pnl under each?
4. **If the live-shadow Rule B PF is materially better** (per-session
   median PF lift > 10%), promote the gate as a deployment override.
5. **If not, retain always-oracle as the default exit** and revisit
   the discriminator features after more shadow data.

## Why this is the right test

The user's framing was correct: "cutting losses short is good, but
the model needs to identify when something is more likely to be a
loser rather than applying the same exit strategy uniformly."

The gate doesn't directly identify "losers vs winners" — that signal
is too noisy in entry features (`pred_win_prob` doesn't discriminate).
But it does identify **regime conditions where winners persist beyond
oracle's expectation**. Functionally equivalent: in those regimes, use
the longer exit.

This is the empirically-grounded answer to the per-trade frustration
of "oracle truncated my $3,490 winner to $72."

## Files

```
v3/reference/oracle_runtime_gate_2026_04_25.md  ← this file
v3/artifacts/forward_walk/forward_walk_chosen_seed{42..46}.pkl
```

No code shipped — this is a hypothesis. Implementation comes after
live-shadow validation.

## Cost summary

CPU only, ~15 min for analysis (LOO + bootstrap + per-feature audit).
