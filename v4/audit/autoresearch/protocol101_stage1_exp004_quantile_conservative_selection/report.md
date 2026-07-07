# Stage-1 Experiment `exp004_quantile_conservative_selection`

- Hypothesis: exp001-003 triangulate winner's curse under heteroscedastic noise: extreme mean-predictions are noise-inflated in sparse regions. Training with quantile loss (alpha=0.25) and selecting where the PESSIMISTIC payoff estimate clears the fee suppresses curse picks: expect z recovery above the refit envelope, materially lower drawdown, and fewer trades/day than exp001 while staying above 0.3.
- Config hash: `360d5d89d473e98cc7c1dd6ff9e7d72f5ee78db793517b62d986d894fed6bf80`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $-337, 0/5 folds profitable, mean sel z 0.16, win-rho -0.158, max DD $298, 0.07 tr/day, ECE 0.006
- seed 43: pooled net $-4932, 0/5 folds profitable, mean sel z 0.35, win-rho -0.156, max DD $4116, 0.40 tr/day, ECE 0.010
- seed 44: pooled net $-8499, 0/5 folds profitable, mean sel z 1.06, win-rho -0.154, max DD $7783, 1.03 tr/day, ECE 0.009

- Refit-null selection z envelope: [0.0, 0.0, 0.0, 0.0, 0.0]

## Gates

- G1_profitability: **False**
- G2_beats_no_skill: **False**
- G3_beats_heuristics: **None**
- G4_drawdown: **False**
- G5_seed_robustness: **False**
- G6_era_guard: **regime_bound_requires_owner_review**
- G7_frequency_band: **False**
- G8_calibration: **True**
- G9_confirmation: **None**

## Guardrails

- Model-tier experiment; no promotion, no paper-submit, holdout untouched.
