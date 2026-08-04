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
| 0.00 | q1_2026 | 80510.00 | 80450.00 | 60.00 | 169 | 40 | 129 | 214 | 3.37 |
| 0.00 | q3_2025 | 60510.00 | 60420.00 | 90.00 | 235 | 4 | 231 | 85 | 5.23 |
| 0.00 | q4_2025 | 137050.00 | 91230.00 | 45820.00 | 250 | 16 | 234 | 206 | 10.34 |
| 0.00 | recent_2026 | 1870.00 | 3640.00 | -1770.00 | 69 | 2 | 67 | 30 | 1.08 |
| 0.00 | total | 279940.00 | 235740.00 | 44200.00 | 723 | 62 | 661 |  |  |
| 0.10 | q1_2026 | 77130.00 | 76110.00 | 1020.00 | 169 | 40 | 129 | 214 | 3.19 |
| 0.10 | q3_2025 | 55810.00 | 55660.00 | 150.00 | 235 | 4 | 231 | 85 | 4.70 |
| 0.10 | q4_2025 | 132050.00 | 86190.00 | 45860.00 | 250 | 16 | 234 | 206 | 9.35 |
| 0.10 | recent_2026 | 490.00 | 2200.00 | -1710.00 | 69 | 2 | 67 | 30 | 1.02 |
| 0.10 | total | 265480.00 | 220160.00 | 45320.00 | 723 | 62 | 661 |  |  |
| 0.25 | q1_2026 | 72060.00 | 69600.00 | 2460.00 | 169 | 40 | 129 | 214 | 2.93 |
| 0.25 | q3_2025 | 48760.00 | 48520.00 | 240.00 | 235 | 4 | 231 | 85 | 3.97 |
| 0.25 | q4_2025 | 124550.00 | 78630.00 | 45920.00 | 250 | 16 | 234 | 206 | 8.04 |
| 0.25 | recent_2026 | -1580.00 | 40.00 | -1620.00 | 69 | 2 | 67 | 30 | 0.94 |
| 0.25 | total | 243790.00 | 196790.00 | 47000.00 | 723 | 62 | 661 |  |  |

## Next Required Evidence

1. Freeze this learned defer overlay configuration before the next preregistered neural policy run.
2. Keep Protocol101 as paper default until fill, untouched holdout data, live no-order parity, and formal validation controls pass.

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

- summary: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w025_e3/summary.json`
- report: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w025_e3/report.md`
- doc: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_REPLAY_RELAXED_M0_W025_E3.md`
