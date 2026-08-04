# VALIDATION_LEARNED_DEFER_CHALLENGER_RESEARCH_PACKET_V1

What is this: research-only validation packet for the frozen learned-defer challenger
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Challenge allowed: `False`
Decision: `learned_defer_packet_complete_holdout_blocked_by_calibration`

## Reproduction

- Status: `pass`
- Slippage `0.00`: challenger `60`, trades `739`, delta `66780.0`
- Slippage `0.10`: challenger `60`, trades `739`, delta `67580.0`
- Slippage `0.25`: challenger `60`, trades `739`, delta `68780.0`

## Diagnostic Status

- Flat-entry anomaly: `resolved_policy_weighting_mismatch`
- Concentration: `pass_with_warnings`
- Slot-cost calibration: `blocked_undercoverage_or_split_instability`
- Formal validation readiness: `formal_overfit_control_blocked_missing_comparable_strategy_matrix`

## Concentration Summary

| slippage | total delta | top day share | top 5 trade share | top split share | warnings |
|---:|---:|---:|---:|---:|---|
| 0.00 | 66780.0000 | 0.1866 | 0.5566 | 0.7877 | top5_trade_positive_share_gt_0_50, single_split_positive_delta_share_gt_0_75 |
| 0.10 | 67580.0000 | 0.1857 | 0.5553 | 0.7789 | top5_trade_positive_share_gt_0_50, single_split_positive_delta_share_gt_0_75 |
| 0.25 | 68780.0000 | 0.1844 | 0.5534 | 0.7662 | top5_trade_positive_share_gt_0_50, single_split_positive_delta_share_gt_0_75 |

## Slot-Cost Calibration By Split

| split | rows | positive rate | AUC | Brier | MAE | p90 abs error | charge coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| q1_2026 | 551413 | 0.1642 | 0.8087 | 0.1663 | 150.1439 | 448.3850 | 0.8690 |
| q3_2025 | 529645 | 0.1897 | 0.9907 | 0.0641 | 53.7179 | 144.3369 | 0.9089 |
| q4_2025 | 534830 | 0.1992 | 0.9906 | 0.0605 | 72.3843 | 220.8249 | 0.8873 |
| recent_2026 | 303630 | 0.0798 | 0.7435 | 0.1236 | 79.0112 | 155.7816 | 0.9315 |

## Validation Readiness

- Status: `formal_overfit_control_blocked_missing_comparable_strategy_matrix`
- Comparable strategy summaries: `7`
- PBO/CSCV status: `not_computed_missing_comparable_strategy_matrix`

## Blockers

- blocked_undercoverage_or_split_instability
- calibrated stochastic fill model unavailable
- untouched holdout data pending
- live no-order full-action parity pending
- formal validation controls pending
- formal_overfit_control_blocked_missing_comparable_strategy_matrix

## Later Live Parity Schema

- `timestamp`
- `session`
- `full_candidate_surface_count`
- `feature_hashes`
- `greeks_freshness`
- `quote_freshness`
- `action_masks`
- `account_affordability_state`
- `protocol101_action`
- `challenger_scores`
- `slot_cost_estimate`
- `slot_cost_uncertainty_charge`
- `final_defer_or_override_decision`
- `latency_ms`
- `missing_or_invalid_candidate_reasons`
- `live_orders_enabled_false`
- `broker_endpoint_called_false`

## Later Fill Observation Schema

- `timestamp`
- `session`
- `intended_action`
- `side`
- `contract_id`
- `quote_age_ms`
- `bid`
- `ask`
- `spread`
- `limit_or_market_assumption`
- `submit_timestamp`
- `fill_timestamp`
- `fill_price`
- `cancel_or_reject_status`
- `realized_slippage`

## Next Required Evidence

1. Replace or recalibrate the global slot-cost uncertainty charge before promotion-grade scoring.
2. Keep Protocol101 as the paper default.
3. Do not run additional neural experiments while this packet is unresolved.
4. Build live no-order full-action parity for this exact frozen challenger.
5. Collect stratified paper/no-order fill evidence before stochastic fill replay.
6. Freeze and score an untouched holdout only after parity, fill, and formal validation gates pass.

## Outputs

- summary: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/summary.json`
- report: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/report.md`
- doc: `v4/docs/LEARNED_DEFER_CHALLENGER_RESEARCH_PACKET_V1.md`

## Artifacts

- freeze_manifest: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/freeze_manifest.json`
- flat_entry_anomaly: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/flat_entry_anomaly.csv`
- concentration_fragility: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/concentration_fragility.csv`
- slot_cost_calibration: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/slot_cost_calibration.csv`
- blocked_protocol101_attribution: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/blocked_protocol101_attribution.csv`
