# Stage-1 Experiment `exp001_payoff_gbm_baseline`

- Hypothesis: A payoff-trained gradient-boosted tree over the live-v1 contract features contains genuine individual-candidate information (G2) on the menu-v2 corpus; economic gates (G1/G4/G5/G7) establish the baseline profile and are expected to be the binding constraints on the first attempt.
- Config hash: `68484657f1fc314f653b79231916b144462bd946a1c323e0152767ca8de24709`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $2573, 3/5 folds profitable, mean sel z 3.29, win-rho 0.019, max DD $18327, 1.78 tr/day, ECE 0.012
- seed 43: pooled net $29516, 3/5 folds profitable, mean sel z 0.13, win-rho 0.009, max DD $22146, 1.78 tr/day, ECE 0.046
- seed 44: pooled net $-7255, 2/5 folds profitable, mean sel z 0.83, win-rho 0.013, max DD $19951, 1.57 tr/day, ECE 0.038

- Refit-null selection z envelope: [-2.32, 0.51, -0.96, 0.45, -1.64]

## Gates

- G1_profitability: **False**
- G2_beats_no_skill: **True**
- G3_beats_heuristics: **None**
- G4_drawdown: **False**
- G5_seed_robustness: **False**
- G6_era_guard: **True**
- G7_frequency_band: **True**
- G8_calibration: **True**
- G9_confirmation: **None**

## Guardrails

- Model-tier experiment; no promotion, no paper-submit, holdout untouched.
