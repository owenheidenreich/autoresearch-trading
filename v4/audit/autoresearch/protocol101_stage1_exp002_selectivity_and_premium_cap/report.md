# Stage-1 Experiment `exp002_selectivity_and_premium_cap`

- Hypothesis: exp001's G4/G5 failures are driven by unbounded premium-at-risk and marginal-EV trades: raising the selection floor to $30 predicted payoff (10x fee) and capping entry premium at $5.00 (=$500 at risk, 5% of account) restores G4 (DD <= $1,500) and G1 (4/5 folds) while keeping G2 above the refit envelope; trade frequency drops but stays above 0.3/day.
- Config hash: `aadb3ac6b0389eec8592c0a675bc1f7498f5d5ffa2b547c51f4fd448ca4c26ae`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $-12537, 1/5 folds profitable, mean sel z 0.84, win-rho 0.019, max DD $7227, 2.18 tr/day, ECE 0.012
- seed 43: pooled net $2766, 2/5 folds profitable, mean sel z 0.46, win-rho 0.009, max DD $7383, 1.85 tr/day, ECE 0.046
- seed 44: pooled net $-14516, 1/5 folds profitable, mean sel z -0.81, win-rho 0.013, max DD $6869, 2.06 tr/day, ECE 0.038

- Refit-null selection z envelope: [-1.22, -0.67, -2.2, -1.19, -1.41]

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
