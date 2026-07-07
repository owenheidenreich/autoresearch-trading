# Stage-1 Experiment `exp005_rolling_regime_window`

- Hypothesis: exp001 fold analysis: signal sign flips by regime coherently across seeds (|z| 10-19 with alternating signs); expanding windows average opposing regimes. Training on only the trailing 60 sessions before each test window keeps the model in-regime: expect per-fold selection z to become predominantly positive (>=4 of 5 folds), reducing the seed spread; economics may remain unbound and are addressed separately if the sign stabilizes.
- Config hash: `8481aade1c78db2a424902cb7e55d526a3bc41b35b43f655f33cd902e6fab395`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $25030, 3/5 folds profitable, mean sel z 2.03, win-rho 0.058, max DD $23332, 1.87 tr/day, ECE 0.040
- seed 43: pooled net $24584, 4/5 folds profitable, mean sel z -0.21, win-rho 0.040, max DD $18277, 1.55 tr/day, ECE 0.022
- seed 44: pooled net $-3189, 3/5 folds profitable, mean sel z 3.41, win-rho 0.051, max DD $23522, 1.93 tr/day, ECE 0.028

- Refit-null selection z envelope: [5.46, 2.09, -2.74, -2.91, -1.33]

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
