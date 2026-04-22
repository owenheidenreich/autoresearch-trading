# Layer-2 Payoff-Sufficiency Gating Probe — 2026-04-21

## Verdict

**SOFT — atm_iv bottom-2-decile suppression is the only candidate with a
positive lift and the lift comes mainly from repairing fold 2 (PF 1.038
→ 1.444), not fold 0. With selection-bias risk (one of five features
tested) and fold 0 still sub-1.0, this is not strong enough to unblock
paper trading on its own.** The routing line (this probe + the prior two)
is exhausted; the regime-gating thesis is *weakly supported but not
clearly actionable*.

## Question

The fallback-routing branches (route-aware,
[fallback-only put-vs-flat](layer2_fallback_only_probe_2026_04_21.md))
both lost to blunt teacher+put. Their write-up argued the binding
constraint was *payoff-regime sufficiency*, not action choice. This probe
tests that thesis with the cheapest possible gate: do any single context
features carve the non-teacher universe into a low-payoff bucket whose
suppression on chosen bars improves PF without hurting fold 0?

## Setup

- Baseline under study:
  [v3/artifacts/layer2_shared_enc_fixedq_detach](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_shared_enc_fixedq_detach)
  — PF 1.455, DD 36.9%, 0.917 TPD, 275 trades (99 teacher / 176 fallback).
- Probe script:
  [v3/analysis/layer2_payoff_gating_probe.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_payoff_gating_probe.py).
- Artifact:
  [v3/artifacts/layer2_payoff_gating_probe/payoff_gating_probe.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_payoff_gating_probe/payoff_gating_probe.json).

## Method

1. Restrict the export bundle to non-teacher rows where a passing put
   exists and `time_stop_pnl_put` is finite (the universe the fallback
   actually faces). N = 58,543 rows pooled across train+val days.
2. For each candidate feature, bin the pooled non-teacher universe into
   10 deciles and report mean / median / hit-rate of `time_stop_pnl_put`
   per decile.
3. Compute Spearman rho between decile rank and decile-mean PnL. A flag
   fires when |top2_mean − bottom2_mean| ≥ $80 AND |rho| ≥ 0.50.
4. For the top 5 features by |rho|, simulate suppression on the 176
   chosen fallback bars: drop a bar to flat when its feature value lands
   in the *low-payoff* tail (bottom 2 deciles if rho > 0, top 2 deciles
   if rho < 0) of its fold's training universe. Compose the surviving
   trades chronologically with the unchanged 99 teacher trades and report
   PF / DD / TPD.

Candidate features: `atm_iv`, `iv_percentile`, `vix`, `first15_range_pct`,
`omar_range_pct`, `sigma_pos`, `abs_sigma_pos`, `last10_range_over_omar`,
`atm_iv_over_first15_range`.

## Universe-level separation

| feature | n | bot2 mean $ | top2 mean $ | rho | flag |
|---|---:|---:|---:|---:|:---:|
| sigma_pos | 58,543 | +42.6 | −85.5 | −0.806 | ✓ |
| abs_sigma_pos | 58,543 | −69.3 | −20.9 | +0.770 | |
| atm_iv | 58,543 | −99.6 | −20.3 | +0.673 | |
| last10_range_over_omar | 58,543 | −96.4 | −64.2 | +0.576 | |
| first15_range_pct | 58,543 | −67.0 | +9.8 | +0.552 | |
| vix | 58,543 | nan | −18.5 | +0.500 | |
| iv_percentile | 58,543 | −61.3 | −52.5 | +0.383 | |
| atm_iv_over_first15_range | 58,543 | +3.5 | −94.2 | −0.139 | |
| omar_range_pct | 58,543 | −22.9 | −64.5 | −0.067 | |

**Important:** every decile mean across every feature is structurally
near or below zero — the non-teacher fallback put universe is on average
a losing bet. The flag bar requires a meaningful spread (≥ $80) AND
monotonic ranking, not an absolute "good bin exists."

`sigma_pos` had the strongest ranking (the further above VWAP, the worse
fallback puts pay on average — consistent with the previously-validated
[VWAP direction rule](project_v3_vwap_findings.md)); `atm_iv`,
`abs_sigma_pos`, and the realized-range features show progressively
weaker but still monotonic patterns.

## Suppression on the 176 chosen fallback bars

| feature | rho | suppress | kept | dropped | PF (Δ) | DD (Δ) | TPD (Δ) | avoided $/trade |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| sigma_pos | −0.806 | top | 127 | 49 | 1.324 (−0.131) | 42.8% (+5.9) | 0.753 (−0.163) | +538 |
| abs_sigma_pos | +0.770 | bottom | 141 | 35 | 1.295 (−0.160) | 37.1% (+0.2) | 0.800 (−0.117) | +769 |
| **atm_iv** | **+0.673** | **bottom** | **110** | **66** | **1.530 (+0.075)** | **26.7% (−10.2)** | 0.697 (−0.220) | **+101** |
| last10_range_over_omar | +0.576 | bottom | 142 | 34 | 1.426 (−0.029) | 49.3% (+12.5) | 0.803 (−0.113) | +295 |
| first15_range_pct | +0.552 | bottom | 123 | 53 | 1.473 (+0.019) | 38.4% (+1.5) | 0.740 (−0.177) | +144 |

Reading:

- **`sigma_pos` and `abs_sigma_pos` HURT the chosen-bar PF**, even though
  they had the strongest universe-level signal. That is exactly what we
  expect when entry selection is already paying for the universe-level
  edge: the suppressed bars in those deciles average +$538 and +$769 of
  PnL — they are *atypical winners for their decile*, and the model has
  selected for them. Universe-level rho does not transfer to chosen bars
  for these features.
- **`atm_iv` is the one feature where suppression both lifts PF and
  shrinks DD.** Avoided $/trade is only +$101 (an order of magnitude
  smaller than `sigma_pos` or `abs_sigma_pos`), which is the signature
  of a low-quality bin: the suppressed bars are roughly break-even,
  losing them barely costs any expected PnL but tightens the equity
  curve. PF moves 1.455 → 1.530 (+0.075), DD shrinks 36.9% → 26.7%.
- The other two (`last10_range_over_omar`, `first15_range_pct`) are
  noise-band changes, with `last10_range_over_omar` markedly widening
  DD.

## atm_iv per-fold breakdown

| fold | trades | PF Δ | DD Δ |
|---:|---:|---:|---:|
| 0 | 55 → 45 | 0.860 → 0.953 (+0.093) | 28.2% → 24.9% |
| 1 | 40 → 40 | 1.439 → 1.439 (no change) | 22.5% → 22.5% |
| 2 | 60 → 29 | 1.038 → **1.444 (+0.405)** | 27.6% → **13.5%** |
| 3 | 60 → 42 | 2.095 → 1.846 (−0.249) | 21.4% → 17.4% |
| 4 | 60 → 53 | 1.801 → 1.969 (+0.169) | 32.2% → 29.1% |

Where the lift comes from:

- **Fold 2** does most of the work: it was the marginal "rounding-error
  above flat" fold, and gating fixes it both in PF (+0.405) and DD
  (−14.1pt) while halving trade count.
- **Fold 0** improves (+0.093) but stays sub-1.0. The losing-fold
  problem is real: gating helps but does not solve it.
- **Fold 3** loses about 12% of its PF (−0.249) but stays robustly
  profitable at 1.846.
- Folds 1 and 4 essentially unchanged or marginally better.

In aggregate this is consistent with the gate filtering out
near-break-even fallback puts taken in low-IV regimes — bars that
contributed little expected value but added equity-curve volatility
(notably in fold 2's chop).

## Why this is SOFT, not PASS

1. **Selection-bias risk.** Five features were tested; one improved.
   With small absolute lift (+0.075 PF), the headline effect is well
   inside the range you would expect from data snooping on a 176-bar
   sample. The fold-2 win is more compelling, but a single fold result
   is also small-sample.
2. **Fold 0 not solved.** atm_iv suppression buys 0.093 PF in fold 0;
   the loser fold remains sub-1.0. Whatever broke fold 0 is not just
   "low-IV fallback puts."
3. **Fold-3 give-back.** The strong fold loses a quarter PF. Not fatal
   (1.846 is still excellent), but it confirms the gate is trimming the
   distribution — not isolating bad bars cleanly.
4. **TPD drops 24%.** Equity curve smoother but capital deployment
   shrinks meaningfully. For a $25k account that may already be
   throughput-constrained.

## What this updates

- The fallback-routing line (route-aware → fallback-only put-vs-flat →
  full call/put/flat → payoff-gating) has now been four cheap probes
  deep with one weak SOFT signal at the end. The routing-and-gating
  workstream on this winning artifact is *exhausted at the cheap-probe
  level*.
- The earlier
  [fallback-only probe](layer2_fallback_only_probe_2026_04_21.md)
  hypothesised that "the model already knows enough to prefer put over
  call on fallback bars; what it doesn't know is when downside move is
  too small or option too expensive." This probe partially corroborates
  the *option-too-expensive* half — `atm_iv` (raw IV level) is the
  candidate that helped. The *downside-move-too-small* features
  (realized range, sigma_pos magnitude) did not transfer.
- The universe-level signals on `sigma_pos` and `abs_sigma_pos` are
  *real but already-priced-in by the entry selection*. That is mildly
  reassuring: it suggests the existing entry score is doing useful work
  on those dimensions even if the side head is cosmetic.

## What NOT to do next

- Do **not** stack atm_iv with other features and re-search for a
  bigger lift on this artifact. That is the multiple-comparisons trap
  and we will find a 1.65 PF "improvement" from noise.
- Do **not** launch GPU on a route-aware or payoff-gated branch.
- Do **not** treat this as a paper-trading unblock. Paper trading
  remains paused per the
  [reality checks verdict](layer2_reality_checks_2026_04_21.md).

## Defensible options going forward

The cheap-probe budget on this artifact is spent. Two reasonable
directions:

1. **Stop iterating on this artifact and pivot.** Treat the detach-side
   PF 1.455 baseline as "stable but architecturally limited" (random
   direction works at PF 1.116; gate-not-side is the load-bearing
   piece). Move attention to a different research question entirely.
2. **Out-of-sample atm_iv-gated retest as the *only* gate.** If the
   workflow allows pulling 2026-04-02 → 2026-04-21 fresh data, rebuild
   the dataset, run the detach-side baseline + atm_iv suppression on
   the new test window, and see whether the gate's lift survives an
   honest hold-out. This is more expensive than the cheap-probe budget
   we have been operating in but is the *only* way to settle the
   selection-bias concern. Gate of pass: lift survives at least
   directionally on fresh days.

These are presented as user-judgement calls, not a single recommended
path.

## Verification

- [x] `python -m py_compile v3/analysis/layer2_payoff_gating_probe.py` passes
- [x] Probe runs end-to-end on the existing detach-side artifacts in seconds
- [x] Baseline PF/DD recovered to canonical 1.455 / 36.9% (chronological
      ordering of teacher + kept fallback trades)
- [x] Five candidate features simulated; per-fold breakdown for the
      best feature recorded above
- [x] Verdict explicitly tagged SOFT, with selection-bias and fold-0
      caveats called out
- [ ] Commit
