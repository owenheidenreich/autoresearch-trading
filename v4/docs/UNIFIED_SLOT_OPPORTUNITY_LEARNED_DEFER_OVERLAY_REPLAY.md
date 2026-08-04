# REPLAY_UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_V1

What is this: strict replay / learned slot opportunity-cost defer overlay on the flat-calibrated conservative policy
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training in this runner: no
Decision: `learned_slot_opportunity_defer_overlay_replay_safe_but_deferred_all_overconservative`

## Interpretation

The learned slot-cost overlay is safe but over-conservative: it defers all challenger overrides back to Protocol101.

## Learned Defer Config

- Min net advantage margin: `250.0`
- Blocked-cost uncertainty weight: `1.0`
- Max blocked Protocol101 entries: `1`
- Estimator decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay`
- Estimator Q1 AUC: `0.810703363520855`

## Stress Results

| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | learned gate skips | PF |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | q1_2026 | 80450.00 | 80450.00 | 0.00 | 217 | 0 | 217 | 1171 | 3.70 |
| 0.00 | q3_2025 | 60420.00 | 60420.00 | 0.00 | 238 | 0 | 238 | 199 | 5.18 |
| 0.00 | q4_2025 | 91230.00 | 91230.00 | 0.00 | 252 | 0 | 252 | 575 | 7.75 |
| 0.00 | recent_2026 | 3640.00 | 3640.00 | 0.00 | 72 | 0 | 72 | 141 | 1.16 |
| 0.00 | total | 235740.00 | 235740.00 | 0.00 | 779 | 0 | 779 |  |  |
| 0.10 | q1_2026 | 76110.00 | 76110.00 | 0.00 | 217 | 0 | 217 | 1171 | 3.45 |
| 0.10 | q3_2025 | 55660.00 | 55660.00 | 0.00 | 238 | 0 | 238 | 199 | 4.66 |
| 0.10 | q4_2025 | 86190.00 | 86190.00 | 0.00 | 252 | 0 | 252 | 575 | 6.89 |
| 0.10 | recent_2026 | 2200.00 | 2200.00 | 0.00 | 72 | 0 | 72 | 141 | 1.09 |
| 0.10 | total | 220160.00 | 220160.00 | 0.00 | 779 | 0 | 779 |  |  |
| 0.25 | q1_2026 | 69600.00 | 69600.00 | 0.00 | 217 | 0 | 217 | 1171 | 3.11 |
| 0.25 | q3_2025 | 48520.00 | 48520.00 | 0.00 | 238 | 0 | 238 | 199 | 3.94 |
| 0.25 | q4_2025 | 78630.00 | 78630.00 | 0.00 | 252 | 0 | 252 | 575 | 5.78 |
| 0.25 | recent_2026 | 40.00 | 40.00 | 0.00 | 72 | 0 | 72 | 141 | 1.00 |
| 0.25 | total | 196790.00 | 196790.00 | 0.00 | 779 | 0 | 779 |  |  |

## Next Required Evidence

1. Decide whether an all-defer overlay is acceptable as a safety guardrail or whether estimator calibration must be sharpened before retraining.
2. Do not treat all-defer behavior as evidence of Protocol101 improvement.

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

- summary: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay/summary.json`
- report: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay/report.md`
- doc: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_REPLAY.md`
