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
- Max blocked Protocol101 entries: `3`
- Estimator decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay`
- Estimator Q1 AUC: `0.810703363520855`

## Stress Results

| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | learned gate skips | PF |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | q1_2026 | 79390.00 | 80450.00 | -1060.00 | 167 | 41 | 126 | 160 | 3.33 |
| 0.00 | q3_2025 | 59680.00 | 60420.00 | -740.00 | 235 | 5 | 230 | 62 | 5.00 |
| 0.00 | q4_2025 | 140960.00 | 91230.00 | 49730.00 | 248 | 17 | 231 | 162 | 10.79 |
| 0.00 | recent_2026 | 6820.00 | 3640.00 | 3180.00 | 65 | 3 | 62 | 5 | 1.30 |
| 0.00 | total | 286850.00 | 235740.00 | 51110.00 | 715 | 66 | 649 |  |  |
| 0.10 | q1_2026 | 76050.00 | 76110.00 | -60.00 | 167 | 41 | 126 | 160 | 3.15 |
| 0.10 | q3_2025 | 54980.00 | 55660.00 | -680.00 | 235 | 5 | 230 | 62 | 4.50 |
| 0.10 | q4_2025 | 136000.00 | 86190.00 | 49810.00 | 248 | 17 | 231 | 162 | 9.76 |
| 0.10 | recent_2026 | 5520.00 | 2200.00 | 3320.00 | 65 | 3 | 62 | 5 | 1.24 |
| 0.10 | total | 272550.00 | 220160.00 | 52390.00 | 715 | 66 | 649 |  |  |
| 0.25 | q1_2026 | 71040.00 | 69600.00 | 1440.00 | 167 | 41 | 126 | 160 | 2.89 |
| 0.25 | q3_2025 | 47930.00 | 48520.00 | -590.00 | 235 | 5 | 230 | 62 | 3.80 |
| 0.25 | q4_2025 | 128560.00 | 78630.00 | 49930.00 | 248 | 17 | 231 | 162 | 8.40 |
| 0.25 | recent_2026 | 3570.00 | 40.00 | 3530.00 | 65 | 3 | 62 | 5 | 1.15 |
| 0.25 | total | 251100.00 | 196790.00 | 54310.00 | 715 | 66 | 649 |  |  |

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

- summary: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w0_e3/summary.json`
- report: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w0_e3/report.md`
- doc: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_REPLAY_RELAXED_M0_W0_E3.md`
