# Stage-1 Experiment `exp003_rank_selectivity`

- Hypothesis: Information lives in prediction rank, not absolute values (exp002 falsification): selecting only the top-2 highest-scored minutes per session (premium uncapped, floor = fee) concentrates on the model's strongest relative convictions, restoring G2 above the refit envelope and reducing drawdown via ~4x fewer trades; frequency lands near 1.5-2.0/day within G7.
- Config hash: `f93f2128f7cb4aff4c0b4290805a35a30b0acfe8ffba3ec5b700c47792cf4c51`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $17410, 3/5 folds profitable, mean sel z 1.13, win-rho 0.019, max DD $17328, 1.10 tr/day, ECE 0.012
- seed 43: pooled net $2834, 3/5 folds profitable, mean sel z 0.37, win-rho 0.009, max DD $21261, 1.04 tr/day, ECE 0.046
- seed 44: pooled net $-22532, 2/5 folds profitable, mean sel z -0.43, win-rho 0.013, max DD $23428, 1.01 tr/day, ECE 0.038

- Refit-null selection z envelope: [-1.13, -1.58, -2.27, 0.17, -1.53]

## Gates

- G1_profitability: **False**
- G2_beats_no_skill: **False**
- G3_beats_heuristics: **None**
- G4_drawdown: **False**
- G5_seed_robustness: **False**
- G6_era_guard: **True**
- G7_frequency_band: **True**
- G8_calibration: **True**
- G9_confirmation: **None**

## Guardrails

- Model-tier experiment; no promotion, no paper-submit, holdout untouched.
