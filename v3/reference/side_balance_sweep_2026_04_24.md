---
date: 2026-04-24
parent: w5_day_diagnostic_2026_04_24.md, side_aware_calibration_2026_04_24.md
exp_base: spx_side_balance_001
status: hypothesis confirmed in direction, magnitude insufficient for promotion
---

# Side-Balance Sample-Weight Sweep — Direction Right, Magnitude Insufficient

## Setup

Per the next-plan four-item sequence (git hygiene → W5 day diagnostic →
side-aware calibration → per-bar sample weights), this is the GPU half of
item 4. After:

- the W5 day diagnostic confirmed real put opportunities exist
  (`w5_day_diagnostic_2026_04_24.md`),
- side-aware calibration alone proved insufficient because the model's put
  scores have no profitable threshold on val
  (`side_aware_calibration_2026_04_24.md`),

the remaining intervention was to fix the gradient asymmetry at training
time. The `--side-balance-weight` flag (commit `682f1ec`) injects a per-row
inverse-frequency weight into `reg_weight_train`, broadcasting through the
regression / dollar / return loss heads.

Experiment: H100 single-seed (42) sweep at side_balance_weight ∈ {0.5, 1.0},
all other hyperparameters identical to `spx_live_hybrid_001`. Two
diagnostics post-train:
1. `side_prior_audit` — does the per-cohort `frac_call_above_put` gap shrink?
2. report.json aggregates — does W5 PF and aggregate PF improve, with
   stable trade count?

## Headline result

```
                                                                  W5
variant    agg_pf  agg_dd%  trades  call_share   pf      dd%   trades
baseline    1.486   17.644     396     0.947    0.703   21.6     14
w_side=1.0  1.449    0.000     410     0.902    3.155    2.8      6   (falsified, small-sample)
sb=0.5      1.496   12.484     370     0.835    0.804   14.8     10
sb=1.0      1.211   10.050     413     0.835    0.677   17.8     11
```

Per-cohort side discrimination on the full dataset (audit):

```
                  OOS truth=put            OOS truth=call
                  c-p mean   frac_c>p      c-p mean   frac_c>p
baseline           0.0928     0.899         0.0882     0.901
w_side=1.0         0.0990     0.915         0.0946     0.919  (worsening — falsified)
sb=0.5             0.0762     0.861         0.0736     0.866
sb=1.0             0.0663     0.828         0.0649     0.838
```

2024-04-01 spot:

```
variant    model_call  model_put
baseline           90          1
w_side=1.0         90          1
sb=0.5             88          3
sb=1.0             84          7
```

## What's confirmed

**The hypothesis direction is correct.** Unlike the falsified
`w_side_contrastive` sweep — which made discrimination *worse* on
truth=put bars (0.899 → 0.923) — the side-balance weight moves it the
right way (0.899 → 0.828 at sb=1.0, a 7 pp shift). On 2024-04-01 the
model's put count rises from 1 to 7. The chosen-trade share moves from
94.7% → 83.5% calls. The model is genuinely picking different bars.

## What's not enough

**Magnitude is insufficient to clear the promotion floor or rescue W5.**
Aggregate PF moves marginally (1.486 → 1.496 at sb=0.5; drops to 1.211 at
sb=1.0). W5 PF goes 0.703 → 0.804 → 0.677. None of these are anywhere
near the floor (PF ≥ 1.976 / 1.786, DD ≤ 12% mean). Trade counts are
stable (370–413 vs 396 baseline) — no small-sample artifact like the
falsified w_side run. So the changes are real but small.

Why partial:
1. The injection only re-weights `reg_weight_train`, which feeds the
   regression / dollar / return heads. The ranking, win, stopout, clean,
   and side_contrastive heads operate without per-bar weights. About half
   of the total loss budget remains gradient-asymmetric.
2. Re-weighting alone shifts argmax distribution but doesn't necessarily
   surface put *winners*. The new put picks may be picking arbitrary puts
   on call-best bars more often than picking the right puts on put-best
   bars.
3. The simulated-L3 oracle's exits were trained on a call-heavy chosen-
   trade set, so put-side simulated exit PnL has built-in extrapolation
   bias against puts. Some of the apparent put underperformance may be
   the oracle confound, not the entry stack.

## Pass-criteria scoring (vs original next-plan)

```
- call share falls from 94–99% without collapsing PF:        sb=0.5 PASS, sb=1.0 PARTIAL (PF dropped)
- W5 PF improves materially:                                 sb=0.5 marginal, sb=1.0 FAIL
- entry-only PF improves, not via L3 rescue:                 FAIL (agg PF flat or worse)
- trade count remains stable:                                PASS
```

2 of 4 at sb=0.5; 1 of 4 at sb=1.0. Not promotion-candidate quality on
either.

## Recommendation

The plan's four-item sequence is exhausted; we have a clean accounting of
what the model can and can't do under the current target/architecture:

- Side prior is real and dominant (audit + falsified loss-shape sweep).
- Real put opportunities exist in W5 labels (W5 day diagnostic).
- Calibration alone can't recover them — model's put scores are
  uninformative (side-aware calibration result).
- Sample-weight rebalance moves the prior in the right direction but not
  far enough (this experiment).

Three viable next directions, none of which I'm starting without your
green-light:

1. **Extend side-balance to all loss heads** (ranking, win, stopout,
   clean, side_contrastive). Strongest version of the same hypothesis.
   Roughly an hour of code + another single-seed retrain.
2. **Combined run: sb=0.5 + w_side_contrastive=0.5.** With balanced
   gradients, the same-bar contrastive term should now enforce
   discrimination instead of amplifying the prior — the failure mode
   from the original w_side falsification.
3. **Rebuild the simulated-L3 oracle from a side-balanced chosen-trade
   set.** The current oracle's call bias may itself poison put-side
   labels. Costs more (per-seed oracle rebuilds are ~40 min CPU each)
   and goes deeper into the target-mismatch concern that the next-plan
   originally flagged ("repair `hybrid_live_utility` before retraining"
   if labels are themselves biased).

If none of those are appealing, the alternative is to step back from the
spx_live_hybrid stack and either retire `hybrid_live` as the target or
accept the champion benchmark stays unchallenged for now.

## Outputs

- `v3/artifacts/layer2_unified_policy_spx_side_balance_001_sb{0p5,1p0}_seed42/seed_42/report.json` (gitignored except for the small JSONs already kept)
- `v3/artifacts/side_prior_audit/spx_side_balance_001_sb{0p5,1p0}_seed42_window05.json` — committed
- `v3/reference/side_balance_sweep_2026_04_24.md` — this file

H100 lease closed. Total GPU spend across the day: ~3 H100-hours
(spx_live_hybrid_001 + spx_w_side_sweep_001 + spx_side_balance_001).
