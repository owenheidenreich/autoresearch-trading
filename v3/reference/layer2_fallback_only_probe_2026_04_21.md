# Layer-2 Fallback-Only Probe — 2026-04-21

## Summary

This was the next cheap gate after the route-aware fallback branch failed.

Question:

- if the route-aware branch failed because teacher bars dominated training,
  can a model trained **only on non-teacher rows** improve the no-teacher
  fallback actions on the detach-side baseline's exact chosen bars?

Answer:

- **no**
- full `call/put/flat` fallback modeling was worse than blunt `put`
- even the narrower `put-vs-flat` model was still worse than blunt `put`
- the problem is not primarily fallback direction choice
- the next unresolved problem is more likely **regime / payoff sufficiency**
  for fallback puts

## Setup

Baseline under study:

- [v3/artifacts/layer2_shared_enc_fixedq_detach](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_shared_enc_fixedq_detach)
- chosen bars: `275`
- teacher bars: `99`
- fallback bars: `176`

Probe script:

- [v3/analysis/layer2_fallback_only_probe.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_fallback_only_probe.py)

Method:

- keep the detach-side chosen bars fixed
- train only on **pre-test non-teacher rows** from the Layer-2 export bundle
- use two `HistGradientBoostingRegressor` heads:
  - `call`: `asinh(time_stop_pnl_call / 100)`
  - `put`: `asinh(time_stop_pnl_put / 100)`
- compare controls on the same chosen bars:
  - `teacher+put`
  - `teacher+call`
  - `teacher+flat`
  - `teacher+oracle_best`
  - `teacher+put_or_flat_model`
  - `teacher+fallback_model`

Artifacts:

- [v3/artifacts/layer2_fallback_only_probe/fallback_only_probe.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_fallback_only_probe/fallback_only_probe.json)

## Results

| control | PF | DD | trades/day | fallback trades | fallback mean PnL |
|---|---:|---:|---:|---:|---:|
| teacher + put | **1.455** | **36.9%** | **0.917** | **176** | **+$313.4** |
| teacher + call | 0.868 | 140.3% | 0.870 | 162 | -$147.4 |
| teacher + flat | 1.142 | 58.7% | 0.330 | 0 | $0.0 |
| teacher + oracle_best | 5.321 | 7.2% | 0.700 | 111 | +$1,808.9 |
| teacher + put_or_flat_model | 1.340 | 58.8% | 0.613 | 85 | +$282.3 |
| teacher + fallback_model | 1.159 | 96.6% | 0.697 | 110 | +$95.3 |

The important ordering is:

- blunt `put` is still the best real fallback
- `put_or_flat` is directionally closer, but still clearly worse
- full `call/put/flat` fallback modeling is much worse

## Why This Matters

The route-aware follow-up suggested the next problem might be
`call/put/flat` on fallback bars.

This probe says that is probably the wrong denominator.

Evidence:

- `teacher+call` is terrible
- `teacher+fallback_model` introduces calls and loses badly
- `teacher+put_or_flat_model` avoids the call problem, but still does not
  beat always-put
- fold 0 gets **worse**, not better, under both learned fallback controls:
  - always-put: PF `0.860`
  - put-or-flat: PF `0.651`
  - full fallback model: PF `0.504`

So the fold-0 problem is not simply “we should flat more” or “we should
route some bars to calls.”

## Chosen-Bar Quality Check

I also checked where the chosen fallback bars sit inside each day's
non-teacher universe.

Mean percentile of the chosen fallback bar among same-day non-teacher bars:

| fold | chosen fallback n | put-PnL percentile | entry-value percentile | chosen put mean |
|---|---:|---:|---:|---:|
| 0 | 29 | 0.535 | 0.512 | +$168.1 |
| 1 | 26 | 0.516 | 0.299 | +$248.8 |
| 2 | 50 | 0.473 | 0.555 | +$32.7 |
| 3 | 53 | 0.550 | 0.626 | +$741.2 |
| 4 | 18 | 0.638 | 0.652 | +$160.6 |
| overall | 176 | 0.530 | 0.541 | +$313.4 |

Interpretation:

- the detach-side branch is not choosing obviously elite fallback bars
  inside the non-teacher universe; they are roughly middle-of-the-pack
- fold 0 and fold 3 have similar same-day fallback percentiles
- the huge difference is **absolute payoff**, not ranking position:
  - fold 0 chosen-put mean: `+$168`
  - fold 3 chosen-put mean: `+$741`

So the binding difference is more likely:

- the underlying regime's downside payoff sufficiency
- option pricing / IV cost versus realized move
- not the fallback action choice itself

## Updated Hypothesis

The next grounded hypothesis is:

- the detach-side baseline's fallback edge is real
- blunt `put` is still the right default fallback action
- the unsolved problem is **when fallback puts are not worth taking**
- that is likely a regime / payoff sufficiency filter, not a direction router

In plain language:

- the model already knows enough to prefer `put` over `call` on fallback bars
- what it does **not** know yet is when the downside move is too small or
  too expensive to justify paying for the put

## What Not To Do Next

- Do not broaden the `call/put/flat` neural search.
- Do not launch GPU from the route-aware or fallback-only branch.
- Do not treat fallback action modeling as the main bottleneck.

## What To Test Next

If Layer-2 continues from here, the next cheap gate should target
fallback **regime gating**, not fallback routing.

Good candidate questions:

- can we predict when fallback puts should be suppressed because the
  payoff regime is too weak?
- do `atm_iv`, `iv_percentile`, `sigma_pos`, first-15 range, or related
  context explain the fold-0-vs-fold-3 payoff gap better than direction models?
- can a fallback gate trained specifically for “put vs flat on low-payoff
  regimes” beat always-put without harming the strong sell-off folds?
