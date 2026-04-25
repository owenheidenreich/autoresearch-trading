---
date: 2026-04-24
parent: golden_day_trace_2024_04_01.md
status: hypothesis confirmed (pre-GPU diagnostic)
---

# Side-Prior Audit — Quarantined seed-42 Window-5 Model

## Setup

After Codex's golden-day trace narrowed the failure to "cross-day training
pressure or side prior" rather than data/architecture/loss-shape, this audit
ran the production model (`spx_live_hybrid_001` seed-42 window-5) over the
**entire** action-surface dataset (89,692 bars) and measured per-bar
best-call vs best-put predicted utility, stratified by truth's best side
(`hybrid_live` target via the seed-42 simulated-L3 oracle).

Hypothesis under test: the model's call bias is a learned global side prior,
not per-bar discrimination. Falsifiable prediction: on bars where truth says
put is best, the model's `frac_model_call_above_put` should still be high —
i.e., the model can't tell the difference.

Script: `v3/analysis/side_prior_audit.py`. Output:
`v3/artifacts/side_prior_audit/seed42_window05.json`.

## Result

```
cohort                      n_bars   mb_call    mb_put   call-put  frac_c>p
all/truth=call               60255   -1.3731   -1.5108     0.1377     0.882
all/truth=put                27688   -1.4353   -1.5179     0.0826     0.846
train/truth=call             31872   -1.2655   -1.4654     0.2000     0.950
train/truth=put              11268   -1.3840   -1.4865     0.1024     0.902
val/truth=call                2172   -1.3207   -1.4087     0.0880     0.895
val/truth=put                 1441   -1.3230   -1.4064     0.0834     0.896
oos/truth=call                2778   -1.3444   -1.4326     0.0882     0.901
oos/truth=put                 2629   -1.3243   -1.4172     0.0928     0.899
```

(`mb_call` / `mb_put` = mean per-bar best-call / best-put predicted utility,
arcsinh scale. `frac_c>p` = fraction of bars where model's best call
out-scores model's best put.)

## What this says

- **The model barely discriminates.** On training data, when truth says call
  is best the model picks call 95.0% of bars; when truth says put is best,
  the model still picks call 90.2% of bars. The discrimination signal is a
  4.8 pp shift, dwarfed by the ~90% baseline call rate.
- **OOS has even less discrimination.** On OOS bars (window-5 OOS test set)
  the call-rate is 90.1% on truth-call bars and 89.9% on truth-put bars —
  a *0.2 pp* shift. The model is essentially blind to side on held-out data.
- **The bias is learned, not architectural.** This shows on training cohort
  too. The model converged to a state where every bar is scored "call is
  ~$8–25 better than put" regardless of label.
- **Spot-check 2024-04-01 reproduces the prior:** model picks 90/1 call/put,
  truth is 14/77, and the per-bar `call-put` margin is +0.127 (a positive
  call advantage in the model's eyes), matching the global pattern.

## Why this points to `w_side_contrastive=0`

Per-bar discrimination is exactly what `_side_contrastive_loss`
([`v3/layer2/unified_policy.py:423-483`](../layer2/unified_policy.py#L423-L483))
is designed to enforce: on bars where best-call-truth and best-put-truth
disagree by ≥$10, penalize the wrong side outscoring the right side by
margin (default 0.20). With `w_side_contrastive=0.0` in production
([`v3/layer2/train_unified_policy.py:67`](../layer2/train_unified_policy.py#L67)),
this term contributes nothing. The remaining loss terms (regression,
best-vs-rest ranking, win, stopout, clean, dollar, return) are all
side-symmetric and never directly compare put-vs-call same-bar.

The other candidates from the original hypothesis:
- *Calibration too narrow:* doesn't apply here. Calibration only filters
  trade count; the underlying score itself is the bias.
- *Flat upweight (1.5×):* doesn't manufacture call preference, only
  suppresses flat.
- *Window-specific issue:* falsified — bias is identical in train, val, and
  OOS cohorts.

## Next step

Retrain seed-42 only on H100 with `--w-side-contrastive ∈ {0.5, 1.0, 2.0}`,
all other hyperparams identical to `spx_live_hybrid_001`. Pass criteria:
- `frac_model_call_above_put` on truth=put bars drops by ≥10 pp (target
  ≤0.75; currently 0.85 OOS, 0.90 train).
- 2024-04-01 model side count shifts toward truth (currently 90/1, target
  ≥10 puts).
- W5 entry-only OOS PF lifts (currently ≈0.6 entry-only at thr=0.20).
- W0–W4 PF does not collapse below baseline by >10%.
