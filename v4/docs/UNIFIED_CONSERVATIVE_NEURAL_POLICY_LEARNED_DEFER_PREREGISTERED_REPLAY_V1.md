# REPLAY_UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_V1

What is this: strict replay / learned slot opportunity-cost defer overlay on the flat-calibrated conservative policy
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training in this runner: no
Decision: `learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training`

## Interpretation

The learned slot-cost overlay preserves nonnegative Q1/Q3 stress deltas while allowing challenger overrides; it is ready as a fixed guardrail for the next preregistered training run.

## Learned Defer Config

- Min net advantage margin: `0.0`
- Blocked-cost uncertainty weight: `0.25`
- Max blocked Protocol101 entries: `3`
- Estimator decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay`
- Estimator Q1 AUC: `0.810703363520855`

## Stress Results

| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | learned gate skips | PF |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | q1_2026 | 82990.00 | 80450.00 | 2540.00 | 190 | 33 | 157 | 127 | 3.38 |
| 0.00 | q3_2025 | 63895.00 | 60420.00 | 3475.00 | 235 | 6 | 229 | 69 | 5.42 |
| 0.00 | q4_2025 | 143830.00 | 91230.00 | 52600.00 | 250 | 17 | 233 | 126 | 10.69 |
| 0.00 | recent_2026 | 11805.00 | 3640.00 | 8165.00 | 64 | 4 | 60 | 7 | 1.58 |
| 0.00 | total | 302520.00 | 235740.00 | 66780.00 | 739 | 60 | 679 |  |  |
| 0.10 | q1_2026 | 79190.00 | 76110.00 | 3080.00 | 190 | 33 | 157 | 127 | 3.19 |
| 0.10 | q3_2025 | 59195.00 | 55660.00 | 3535.00 | 235 | 6 | 229 | 69 | 4.89 |
| 0.10 | q4_2025 | 138830.00 | 86190.00 | 52640.00 | 250 | 17 | 233 | 126 | 9.70 |
| 0.10 | recent_2026 | 10525.00 | 2200.00 | 8325.00 | 64 | 4 | 60 | 7 | 1.50 |
| 0.10 | total | 287740.00 | 220160.00 | 67580.00 | 739 | 60 | 679 |  |  |
| 0.25 | q1_2026 | 73490.00 | 69600.00 | 3890.00 | 190 | 33 | 157 | 127 | 2.92 |
| 0.25 | q3_2025 | 52145.00 | 48520.00 | 3625.00 | 235 | 6 | 229 | 69 | 4.16 |
| 0.25 | q4_2025 | 131330.00 | 78630.00 | 52700.00 | 250 | 17 | 233 | 126 | 8.39 |
| 0.25 | recent_2026 | 8605.00 | 40.00 | 8565.00 | 64 | 4 | 60 | 7 | 1.39 |
| 0.25 | total | 265570.00 | 196790.00 | 68780.00 | 739 | 60 | 679 |  |  |

## Next Required Evidence

1. Freeze this learned defer overlay configuration before the next preregistered neural policy run.
2. Keep Protocol101 as paper default until fill, untouched holdout data, live no-order parity, and formal validation controls pass.

## Remaining Challenge Blockers

- calibrated stochastic fill model unavailable
- untouched holdout data pending
- live no-order full-action parity pending
- formal validation controls pending

## Artifacts

- policy_model_artifacts: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts`
- policy_training_summary: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/summary.json`
- slot_estimator: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/slot_opportunity_cost_estimator.joblib`
- slot_estimator_summary: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/summary.json`

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/report.md`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_REPLAY_V1.md`
