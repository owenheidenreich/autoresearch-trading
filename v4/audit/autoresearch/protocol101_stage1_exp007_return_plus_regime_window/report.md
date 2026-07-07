# Stage-1 Experiment `exp007_return_plus_regime_window`

- Hypothesis: Synthesis of the two independently-helpful levers: return-on-premium target (exp006: win-rho 0.115, DD -40%, consistent positive z) plus rolling 60-session in-regime training (exp005: regime-coherent, 2/3 seeds positive PnL). Expect the best profitability and most consistent selection z yet, isolating whether in-regime return-normalized ranking clears G1/G2 even if G4 remains structurally blocked by convexity's full-premium-loss drawdown.
- Config hash: `63c68e6d4e65f69b15faf725cdfc279e60aaa27fe1625fbeba48fe2345912ac9`
- Sessions: 274 | examples: 3450416

## Per-seed (primary fee $3.00)

- seed 42: pooled net $10029, 2/5 folds profitable, mean sel z 1.60, win-rho 0.098, max DD $14914, 1.81 tr/day, ECE 0.032
- seed 43: pooled net $-2268, 2/5 folds profitable, mean sel z 2.74, win-rho 0.109, max DD $19249, 2.13 tr/day, ECE 0.024
- seed 44: pooled net $13341, 3/5 folds profitable, mean sel z 1.90, win-rho 0.123, max DD $10998, 2.04 tr/day, ECE 0.031

- Refit-null selection z envelope: [-2.75, 0.24, -1.79, -1.44, -3.44]

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
