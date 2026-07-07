# Stage-1 Experiment `exp008_abstention_synthesis`

- Hypothesis: Stage-1 culmination under the owner-revised relative G4 (<=25% peak equity): return-on-premium target + rolling 60-session in-regime window + conviction-quantile 0.85 abstention (trade only top-15% predicted-return candidates, enabling zero-trade low-edge sessions). Expect trades to concentrate into high-edge windows, lifting per-fold PnL toward G1 and pulling relative drawdown toward the 25% gate, while frequency stays above the G7 0.3/day floor and selection z holds above the refit envelope. Isolates whether disciplined abstention is the missing profitability lever before stage-2 learned exits.
- Config hash: `797dbdd976b788d32ccebbe8657c1c5aac82ec6eb2315e8c99c77b24117bbb2e`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $587, 2/5 folds profitable, mean sel z 0.94, win-rho 0.098, max DD $18947 (141% eq), 1.62 tr/day, ECE 0.032
- seed 43: pooled net $2226, 1/5 folds profitable, mean sel z 2.49, win-rho 0.109, max DD $15374 (134% eq), 1.82 tr/day, ECE 0.024
- seed 44: pooled net $20432, 3/5 folds profitable, mean sel z 1.63, win-rho 0.123, max DD $11037 (83% eq), 1.84 tr/day, ECE 0.031

- Refit-null selection z envelope: [-2.87, 0.21, -2.47, -1.62, -3.47]

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
