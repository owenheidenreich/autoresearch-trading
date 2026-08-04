# FOUNDATION_UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR_V1

What is this: foundation estimator / causal Protocol101 slot opportunity-cost model
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: yes
Decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay`

## Scope

- Rows loaded: `1919518`
- Train rows: `350000`
- Feature count: `78`
- Train splits: `['q3_2025', 'q4_2025']`
- Validation split: `q1_2026`

## Metrics

| split | rows | positive_cost_rate | cost_mae | cost_p90_abs_error | positive_auc | positive_brier | count_mae |
|---|---|---|---|---|---|---|---|
| train_sample | 350000 | 0.4000 | 115.3460 | 350.4608 | 0.9911 | 0.0477 | 0.2421 |
| q1_2026 | 120000 | 0.1631 | 148.9092 | 443.3381 | 0.8107 | 0.1645 | 0.4383 |
| q3_2025 | 120000 | 0.1880 | 53.3939 | 143.4709 | 0.9908 | 0.0640 | 0.2286 |
| q4_2025 | 120000 | 0.1981 | 71.5238 | 217.1037 | 0.9904 | 0.0603 | 0.2355 |
| recent_2026 | 120000 | 0.0813 | 80.8532 | 161.9030 | 0.7418 | 0.1246 | 0.4620 |

## Feature Contract

- Status: `pass`
- Runtime-forbidden label/exit/oracle columns are excluded from the feature set.

## Next Required Evidence

1. Replay the learned estimator as a strict defer overlay on flat-calibrated challenger overrides.
2. Reject challenger slot consumption unless predicted advantage clears estimated cost plus uncertainty.
3. Require Q1/Q3 nonnegative same-scope deltas under $0.00/$0.10/$0.25 stress before another neural policy run.

## Artifacts

- manifest: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/manifest.json`
- model: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/slot_opportunity_cost_estimator.joblib`

## Outputs

- summary: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/summary.json`
- report: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/report.md`
- doc: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR.md`
