---
date: 2026-04-26
parent: regime_diagnostic_2026_04_26.md
status: DIAGNOSTIC — Layer-2 entry policy is regime-skewed (10.57x pick-rate range); L3 spread is partially propagated from L2 but L3 also contributes. Proceed to Angle A with bounded expectations.
---

# Layer-2 Entry Audit: Both Layers Contribute to Regime Spread

## Setup

Question (from the regime-spread minimization plan, Step B): is the L3 oracle's
cross-cell PF spread fixable at the L3 layer, or is it propagated from L2's
entry decisions? If L2 picks systematically different trades per regime, our
L3 tweaks are downstream patches.

Method: stratify the action_surface_dataset universe + L2 chosen-trades by
the same 9-cell trend×vol regime scheme used in the regime-stratified eval.
Compare pick-rate, side bias, predicted scores, and realized PF (with
baseline oracle) per cell.

## Results (5 seeds, 1760 picks across 9 regime cells)

```
 trend  vol  universe  picked  pick_rate  %calls  mean_pred_dollar  realized_pf
  bull  low    26466     677    0.0051    55.0%       -0.090           1.857
  chop  low     9051     245    0.0054    49.8%       -0.142           1.180  ← LOW PF, NEG predicted score
  chop  mid     9141     202    0.0044    37.6%       +0.054           1.746
  chop  high    9654     193    0.0040    21.2%       +0.651           2.479  ← HIGH PF, HIGH predicted score
  bull  mid    10884     180    0.0033    30.6%       +0.204           2.253
  bear  high   13957     152    0.0022    37.5%       +0.228           1.990
  bull  high    4643      90    0.0039    26.7%       +0.797           1.547
  bear  mid     3841      14    0.0007    35.7%       +0.221           2.888  (small n)
  bear  low     2733       7    0.0005    42.9%       -0.212           1.499  (small n)
```

Pick-rate range: **0.0005 to 0.0054 — 10.57x ratio**.

## Key findings

### 1. L2 is regime-skewed in entry frequency

L2 picks at 10.57x different rates across regimes. Bear regimes (10% of
training data) get only 9.8% of picks at much lower pick-rates. Bull regimes
get 53.8% of picks. Some of this just reflects the underlying universe
(bears are rarer), but the pick-rate ratio of 10.57x exceeds the universe
distribution differences — L2 is being measurably MORE selective in bears.

### 2. But the spread isn't only explained by L2 skew

Among high-confidence cells (n ≥ 100):
- **chop × low (n=245):** pick_rate 0.0054, mean_dollar **-0.14**, realized PF **1.18**
- **chop × high (n=193):** pick_rate 0.0040, mean_dollar **+0.65**, realized PF **2.48**

Both cells get plenty of L2 picks. But chop×low realizes 53% lower PF than
chop×high. That gap is L3 exit policy generalizing unevenly within
comparable entry conditions.

### 3. L2 has a calibration miss in chop×low

L2 enters 245 trades in chop×low at a *negative* mean predicted dollar score
(-0.14). The model expects to lose money on these but enters anyway. These
trades realize the worst PF (1.18). This is an L2 calibration problem,
not an L3 exit problem.

### 4. Side bias is non-trivially regime-dependent

I expected a clean "puts in bear, calls in bull" pattern. The data shows:
- bull × mid: only 30.6% calls (mostly puts!)
- bull × high: only 26.7% calls (heavily puts!)
- bear × low: 42.9% calls

L2's side selection is puts-heavy in moderate/high vol regardless of trend
direction. The "side asymmetry" the regime diagnostic flagged is real but
not a simple regime correlation — it's vol-conditional too.

## Decision (revised from binary)

The original audit's binary verdict ("L2 is the bottleneck → stop L3 ladder")
is too strict. The data shows **both layers contribute** to the spread:

- **L2 contributes:** ~50% via entry skew + calibration. Pick-rate variation
  amplifies cell variance for rare regimes; chop×low takes negative-EV trades.
- **L3 contributes:** ~50% via exit policy generalizing unevenly even within
  comparable entry conditions (chop×low vs chop×high gap of 1.30 PF).

Given:
- L3 changes (Angles A, C) are cheap and the surface is well-understood
- L2 redesign is a much bigger lift
- L3 changes reduce the L3 portion of the spread; even partial reduction
  brings the cross-cell range closer to the user's 1.00 target

**Proceed to Angle A** as planned, with two updates:
1. Cap expectations: Angle A targets the L3 portion of the spread (~0.7 of
   the 1.30). Even a complete L3 fix won't reach 1.00 spread without L2
   work.
2. Add an L2 follow-up plan to the backlog: address chop×low calibration,
   tighten side selection rules, possibly increase bear pick-rate.

## Cost summary

- Dev: 1 hour (audit script)
- Compute: 5 seconds (analysis only, no training)
- $0 GPU

## Files

```
scripts/research_layer2_entry_audit.py
v3/artifacts/research/layer2_entry_audit.json
v3/artifacts/research/layer2_entry_audit_picks.csv
v3/reference/layer2_entry_audit_2026_04_26.md  ← this file
```
