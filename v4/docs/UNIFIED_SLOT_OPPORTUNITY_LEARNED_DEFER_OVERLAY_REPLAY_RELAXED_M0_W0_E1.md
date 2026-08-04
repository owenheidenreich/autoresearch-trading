# REPLAY_UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_V1

What is this: strict replay / learned slot opportunity-cost defer overlay on the flat-calibrated conservative policy
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training in this runner: no
Decision: `learned_slot_opportunity_defer_overlay_replay_blocks_training_q1_q3_stress_failed`

## Interpretation

The learned slot-cost overlay did not repair the protected Q1/Q3 stress deltas; more neural training remains blocked.

## Learned Defer Config

- Min net advantage margin: `0.0`
- Blocked-cost uncertainty weight: `0.0`
- Max blocked Protocol101 entries: `1`
- Estimator decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay`
- Estimator Q1 AUC: `0.810703363520855`

## Stress Results

| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | learned gate skips | PF |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | q1_2026 | 70620.00 | 80450.00 | -9830.00 | 191 | 42 | 149 | 303 | 2.86 |
| 0.00 | q3_2025 | 59680.00 | 60420.00 | -740.00 | 235 | 5 | 230 | 68 | 5.00 |
| 0.00 | q4_2025 | 140460.00 | 91230.00 | 49230.00 | 250 | 16 | 234 | 174 | 10.38 |
| 0.00 | recent_2026 | 1870.00 | 3640.00 | -1770.00 | 69 | 2 | 67 | 33 | 1.08 |
| 0.00 | total | 272630.00 | 235740.00 | 36890.00 | 745 | 65 | 680 |  |  |
| 0.10 | q1_2026 | 66800.00 | 76110.00 | -9310.00 | 191 | 42 | 149 | 303 | 2.70 |
| 0.10 | q3_2025 | 54980.00 | 55660.00 | -680.00 | 235 | 5 | 230 | 68 | 4.50 |
| 0.10 | q4_2025 | 135460.00 | 86190.00 | 49270.00 | 250 | 16 | 234 | 174 | 9.40 |
| 0.10 | recent_2026 | 490.00 | 2200.00 | -1710.00 | 69 | 2 | 67 | 33 | 1.02 |
| 0.10 | total | 257730.00 | 220160.00 | 37570.00 | 745 | 65 | 680 |  |  |
| 0.25 | q1_2026 | 61070.00 | 69600.00 | -8530.00 | 191 | 42 | 149 | 303 | 2.47 |
| 0.25 | q3_2025 | 47930.00 | 48520.00 | -590.00 | 235 | 5 | 230 | 68 | 3.80 |
| 0.25 | q4_2025 | 127960.00 | 78630.00 | 49330.00 | 250 | 16 | 234 | 174 | 8.11 |
| 0.25 | recent_2026 | -1580.00 | 40.00 | -1620.00 | 69 | 2 | 67 | 33 | 0.94 |
| 0.25 | total | 235380.00 | 196790.00 | 38590.00 | 745 | 65 | 680 |  |  |

## Next Required Evidence

1. Diagnose learned-estimator false negatives/underestimated blocked costs in Q1/Q3.
2. Do not run another neural policy until the learned overlay passes Q1/Q3 stress replay.

## Remaining Challenge Blockers

- calibrated stochastic fill model unavailable
- untouched holdout data pending
- live no-order full-action parity pending
- formal validation controls pending

## Artifacts

- policy_model_artifacts: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1/model_artifacts`
- policy_training_summary: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1/summary.json`
- slot_estimator: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/slot_opportunity_cost_estimator.joblib`
- slot_estimator_summary: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/summary.json`

## Outputs

- summary: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w0_e1/summary.json`
- report: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w0_e1/report.md`
- doc: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_REPLAY_RELAXED_M0_W0_E1.md`
