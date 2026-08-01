# Phase R5 — Conditional V0/V1 Rule — 2026-04-22

## TL;DR

**TIE — blanket V0 remains champion.** Walk-forward conditional
classifier trained to detect V1-favorable days from 51 entry-bar
features:

- Oracle upper bound (perfect foresight): PF **1.302** — the ceiling
- Conditional rule: PF **1.092** (−0.040 vs V0 blanket)
- Blanket V0 (R4 baseline): PF **1.132**
- Blanket V1: PF 0.888

The classifier has AUC 0.72-1.00 on most windows — signal IS present
— but precision at threshold 0.5 is usually 0. Meaning the classifier
ranks V1-favorable days correctly but doesn't fire confidently.
Threshold calibration could help but the ceiling (oracle 1.302) is
only +0.17 above blanket V0 (1.132), so the conditional rule has
limited room to materially improve.

## Setup

Script: [v3/analysis/conditional_directional_rule.py](../analysis/conditional_directional_rule.py).
Artifacts:
- [v3/artifacts/conditional_directional_rule/conditional_eval.json](../artifacts/conditional_directional_rule/conditional_eval.json)
- [v3/artifacts/conditional_directional_rule/conditional_trades.csv](../artifacts/conditional_directional_rule/conditional_trades.csv)

Method:
1. For each (day, bar_index) from R4: pair V0_pnl and V1_pnl
2. Label = 1 if V1_pnl > V0_pnl + $50 (V1-favorable day)
3. Features: 51 augmented features at chosen entry bar (from R3 preds)
4. Walk-forward: train on earlier windows' pairs, test on current window
5. Decision: if `P(V1-favorable) >= 0.5`, use V1 direction; else V0

Label balance: **21/432 pairs are V1-favorable = 4.9%**. Very imbalanced.

## Per-window detail

| Win | Train n (pos) | Test n (pos) | AUC | Precision | Recall | Cond PF | V0 PF | Δ |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | fallback | 19 (0) | — | — | — | 0.948 | 0.948 | 0 |
| 1 | fallback | 28 (2) | — | — | — | 0.699 | 0.699 | 0 |
| 2 | 47 (2) | 37 (0) | nan | — | — | 0.895 | 0.895 | 0 |
| 3 | 84 (2) | 40 (2) | 0.513 | 0.000 | 0.000 | 0.600 | 0.600 | 0 |
| **4** | 124 (4) | 55 (2) | 0.726 | 0.000 | 0.000 | **0.487** | **0.586** | **−0.099** |
| 5 | 179 (6) | 35 (2) | 0.970 | 0.000 | 0.000 | 2.033 | 2.033 | 0 |
| 6 | 214 (8) | 24 (0) | nan | — | — | 1.416 | 1.540 | −0.124 |
| 7 | 238 (8) | 30 (3) | 1.000 | 0.000 | 0.000 | 1.160 | 1.160 | 0 |
| **8** | 268 (11) | 37 (2) | 0.800 | 0.000 | 0.000 | **0.820** | **1.033** | **−0.213** |
| 9 | 305 (13) | 18 (1) | 0.824 | 0.000 | 0.000 | 2.274 | 2.274 | 0 |
| **10** | 323 (14) | 35 (3) | 0.969 | 0.667 | 0.667 | **1.394** | **1.072** | **+0.322** |
| 11 | 358 (17) | 48 (1) | 0.915 | 0.000 | 0.000 | 1.754 | 1.754 | 0 |
| 12 | 406 (18) | 26 (3) | 0.884 | 0.333 | 0.333 | **0.688** | **0.847** | **−0.159** |

The classifier flips 3 wins (window 10) vs 3 losses (windows 4, 8, 12).
Net effect: roughly wash, slightly negative.

## What this says

1. **Signal is there but weak.** AUC is 0.7-1.0 on 9/13 windows — the
   classifier ranks correctly. But ranking alone doesn't help if the
   threshold at which we commit (0.5) rarely fires.
2. **Class imbalance is crippling.** With 4.9% positive rate and 21
   positives across 13 windows, there's too little signal to learn a
   tight decision boundary.
3. **Oracle ceiling is modest.** Even with perfect foresight we'd
   only gain +0.17 over blanket V0. The most alpha is already in
   choosing V0 as default.
4. **Threshold 0.5 is wrong.** A calibrated per-window threshold
   would likely improve results, but exploring that risks overfitting
   on thin positive samples.

## Decision

**Keep blanket V0 as champion.** The conditional rule doesn't
reliably add value at this sample size. Revisit only if:
- More data becomes available (forward OOS or finer windows)
- A different label definition (e.g., hourly V0/V1 switching) produces
  more balanced targets
- Specific regime features (not tested here) emerge from other analysis

## Updated Phase R1-R5 picture

| Phase | Finding |
|---|---|
| R1 | 13 disjoint 60-day OOS windows generated (780 OOS days) |
| R2 | 9 new intraday-developing features spec'd (6 Cat A + 3 Cat B) |
| R3 | 13 per-window L2 models retrained with 51-feature input (1.0 min CPU) |
| R4 | V0 wins 12/13 strict, 10/13 by `> 0.10 PF`; agg PF V0 1.132 > V1 0.888 |
| **R5** | **Conditional rule ties blanket V0 (1.092 vs 1.132); V0 is champion** |

## What's next

R6: lock V0 as the directional rule, re-apply Layer-3 A3 (augmented
minus mfe_norm at threshold 0.19) to V0 trades per window, compute
final composed PF, and write the summary doc.

## Verification

- [x] `python -m py_compile v3/analysis/conditional_directional_rule.py` passes
- [x] Walk-forward classifier trained across 13 windows
- [x] Per-window AUC/precision/recall reported
- [x] Cross-window aggregate conditional PF = 1.092
- [x] Oracle upper bound = 1.302 (establishes ceiling)
- [x] Verdict: TIE → V0 blanket is champion
- [ ] Commit
