# Stage-1 Experiment `exp006_return_on_premium_target`

- Hypothesis: The drawdown wall traces to the dollar-PnL training target, which rewards premium size (corr(premium,|pnl|)=0.37) though premium is uncorrelated with profit (corr=0) and carries 4x the per-trade variance at the top quartile. Training on return-on-premium (pnl/premium) removes this bias and steers selection toward cheaper, higher-return, lower-variance contracts: expect materially lower drawdown vs exp001 and preserved G2. Isolated on the same expanding window as exp001 for direct comparison.
- Config hash: `eec59099dd0e800ae3296416ac09fddc1e91ace79b5d4fed4dc87c2c8fe8b4d0`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $-18343, 1/5 folds profitable, mean sel z 1.97, win-rho 0.118, max DD $13826, 2.26 tr/day, ECE 0.068
- seed 43: pooled net $-8745, 2/5 folds profitable, mean sel z 5.54, win-rho 0.117, max DD $10927, 2.31 tr/day, ECE 0.070
- seed 44: pooled net $1582, 3/5 folds profitable, mean sel z 3.75, win-rho 0.113, max DD $10615, 2.77 tr/day, ECE 0.060

- Refit-null selection z envelope: [-1.61, -1.51, 1.07, -2.84, -0.96]

## Gates

- G1_profitability: **False**
- G2_beats_no_skill: **False**
- G3_beats_heuristics: **None**
- G4_drawdown: **False**
- G5_seed_robustness: **False**
- G6_era_guard: **regime_bound_requires_owner_review**
- G7_frequency_band: **True**
- G8_calibration: **True**
- G9_confirmation: **None**

## Guardrails

- Model-tier experiment; no promotion, no paper-submit, holdout untouched.
