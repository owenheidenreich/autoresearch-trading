# AUDIT_PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1

What is this: Protocol101 internal slot-cost archetype decomposition
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_slot_cost_archetype_decomposition_complete_model_design_ready_training_blocked`

## Bottom Line

Protocol101's internal slot-cost weakness is real enough to study, but not yet real enough to train against. The counterfactual-flat audit found many later Protocol101 signals blocked by current open positions; this packet shows where those blocked opportunities concentrate.

The strongest next research direction is not a broad neural model. It is a named, narrow hypothesis around same-side continuation/switching cost, long-duration/fallback holds, or small-winner/loser entries that block stronger later Protocol101 signals.

## Headline

- Open trades with blocked entries: `2291`
- Positive slot-cost open trades: `1109`
- Blocked entry events: `3723`
- Best-blocked-minus-open total: `$399,510`
- Positive slot-cost total: `$859,990`
- Actual-trade join failures: `0`

## Thesis Tests

| thesis | open trades | slot-cost total | positive rate | median slot cost | next action |
|---|---:|---:|---:|---:|---|
| losing_open_trade_blocks_positive_later_signal | 485 | $635,240 | 1.000 | $930 | Audit these rows first for pre-entry avoidability versus early exit/loss-cut evidence. |
| opposite_side_later_signal_blocked | 428 | $572,430 | 1.000 | $930 | Treat as a regime-reversal/exit question, not a runner-extension question. |
| sequence_residual_slot_cost | 741 | $472,930 | 1.000 | $380 | Segment by side/time/premium before proposing any gate. |
| long_duration_slot_cost | 396 | $409,130 | 1.000 | $640 | Build mutually exclusive replay for only long-duration positive-slot-cost rows. |
| same_side_later_signal_blocked | 681 | $287,560 | 1.000 | $280 | This is the cleanest candidate for a same-side runner/switching-cost lifecycle diagnostic. |
| fallback_exit_slot_cost | 324 | $261,450 | 1.000 | $550 | Prioritize lifecycle improvement here; this is more specific than generic hold-longer. |
| small_winner_blocks_larger_later_signal | 395 | $148,330 | 1.000 | $240 | Test a learned defer / earlier exit rule only in this declared low-open-PnL archetype. |
| high_mfe_giveback_blocks_later_signal | 29 | $18,750 | 1.000 | $260 | Pair with hold/exit action-advantage labels and require a giveback guard. |

## Top Open-Trade Archetypes By Slot Cost

| split | side | time | moneyness | premium | duration | exit | score margin | open trades | slot-cost total | positive rate | median MFE | median giveback |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| q4_2025 | P | post_open_morning | ITM | 3000_3500 | 6_10m | sequence_residual_override | gte_1.00 | 42 | $37,190 | 0.762 | $500 | $420 |
| q1_2026 | P | post_open_morning | ITM | 3000_3500 | 6_10m | sequence_residual_override | gte_1.00 | 24 | $18,910 | 0.667 | $440 | $610 |
| q1_2026 | P | post_open_morning | ITM | 2500_3000 | 6_10m | sequence_residual_override | gte_1.00 | 23 | $14,180 | 0.696 | $370 | $310 |
| q1_2026 | P | post_open_morning | ITM | 3000_3500 | 21_25m | protocol054_fallback | gte_1.00 | 16 | $12,840 | 0.688 | $330 | $710 |
| march_2026 | P | post_open_morning | ITM | 3000_3500 | 6_10m | sequence_residual_override | gte_1.00 | 11 | $12,450 | 0.818 | $330 | $610 |
| q3_2025 | C | post_open_morning | ITM | 2500_3000 | 21_25m | protocol054_fallback | gte_1.00 | 19 | $12,400 | 0.737 | $160 | $780 |
| q1_2026 | P | post_open_morning | ITM | 2500_3000 | 21_25m | protocol054_fallback | 0.50_1.00 | 6 | $10,560 | 1.000 | $455 | $1,060 |
| q1_2026 | C | late_afternoon | ITM | 3000_3500 | 21_25m | protocol054_fallback | gte_1.00 | 15 | $9,180 | 0.800 | $140 | $650 |
| march_2026 | P | post_open_morning | ITM | 2500_3000 | 6_10m | sequence_residual_override | gte_1.00 | 10 | $9,130 | 0.800 | $415 | $350 |
| q3_2025 | P | post_open_morning | ITM | 3000_3500 | 6_10m | hard_stop | gte_1.00 | 3 | $8,790 | 1.000 | $-40 | $1,570 |
| march_2026 | C | late_afternoon | ITM | 3000_3500 | 21_25m | protocol054_fallback | gte_1.00 | 10 | $8,710 | 1.000 | $140 | $880 |
| q1_2026 | C | post_open_morning | ITM | 3000_3500 | 21_25m | protocol054_fallback | gte_1.00 | 18 | $8,550 | 0.611 | $260 | $375 |
| q1_2026 | P | post_open_morning | ITM | 2000_2500 | 21_25m | hard_stop | gte_1.00 | 4 | $8,440 | 1.000 | $130 | $1,160 |
| q3_2025 | C | post_open_morning | ITM | 3000_3500 | 16_20m | hard_stop | gte_1.00 | 4 | $7,640 | 1.000 | $40 | $1,740 |
| march_2026 | C | post_open_morning | ITM | 2500_3000 | 16_20m | protocol054_fallback | 0.50_1.00 | 2 | $7,480 | 1.000 | $780 | $1,800 |
| q1_2026 | C | post_open_morning | ITM | 3000_3500 | 6_10m | sequence_residual_override | gte_1.00 | 7 | $7,410 | 0.857 | $630 | $90 |
| q1_2026 | C | post_open_morning | ITM | 2500_3000 | 16_20m | protocol054_fallback | 0.50_1.00 | 3 | $7,050 | 0.667 | $780 | $1,800 |
| march_2026 | P | post_open_morning | ITM | 2500_3000 | 21_25m | protocol054_fallback | 0.50_1.00 | 4 | $6,820 | 1.000 | $450 | $1,060 |
| q1_2026 | P | late_afternoon | ITM | 1000_1500 | 21_25m | protocol054_fallback | gte_1.00 | 3 | $6,740 | 1.000 | $640 | $280 |
| q4_2025 | C | post_open_morning | ITM | 3000_3500 | 6_10m | sequence_residual_override | 0.50_1.00 | 14 | $6,590 | 0.643 | $190 | $120 |

## Top Blocked-Event Archetypes

| split | open side | blocked side | relation | open exit | open time | blocked time | after open | before exit | events | blocked PnL | positive rate |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| q3_2025 | C | C | same_side_different_contract | sequence_residual_override | post_open_morning | post_open_morning | le_2m_after_open | 6_10m_before_exit | 55 | $23,110 | 0.855 |
| q1_2026 | P | C | opposite_side | sequence_residual_override | post_open_morning | post_open_morning | le_2m_after_open | 6_10m_before_exit | 23 | $15,730 | 0.739 |
| q1_2026 | C | C | same_side_different_contract | sequence_residual_override | post_open_morning | post_open_morning | 6_10m_after_open | 6_10m_before_exit | 25 | $15,400 | 0.840 |
| march_2026 | P | C | opposite_side | sequence_residual_override | post_open_morning | post_open_morning | le_2m_after_open | 6_10m_before_exit | 12 | $14,550 | 1.000 |
| q3_2025 | C | C | same_side_different_contract | sequence_residual_override | post_open_morning | post_open_morning | 6_10m_after_open | le_2m_before_exit | 41 | $13,880 | 0.902 |
| q1_2026 | P | P | same_side_different_contract | target | post_open_morning | post_open_morning | 3_5m_after_open | gt_10m_before_exit | 12 | $13,820 | 1.000 |
| q4_2025 | P | C | opposite_side | sequence_residual_override | post_open_morning | post_open_morning | le_2m_after_open | 6_10m_before_exit | 29 | $13,710 | 0.828 |
| q1_2026 | P | P | same_side_different_contract | sequence_residual_override | post_open_morning | post_open_morning | 6_10m_after_open | 3_5m_before_exit | 23 | $13,170 | 0.826 |
| q1_2026 | C | C | same_side_different_contract | sequence_residual_override | post_open_morning | post_open_morning | 6_10m_after_open | 3_5m_before_exit | 18 | $12,650 | 0.889 |
| q3_2025 | C | C | same_side_different_contract | sequence_residual_override | post_open_morning | post_open_morning | le_2m_after_open | 3_5m_before_exit | 37 | $12,130 | 0.946 |
| q1_2026 | C | C | same_side_different_contract | target | post_open_morning | post_open_morning | gt_10m_after_open | gt_10m_before_exit | 5 | $12,020 | 1.000 |
| q4_2025 | P | C | opposite_side | sequence_residual_override | post_open_morning | post_open_morning | 3_5m_after_open | 3_5m_before_exit | 21 | $11,560 | 1.000 |
| q1_2026 | P | C | opposite_side | hard_stop | post_open_morning | post_open_morning | le_2m_after_open | gt_10m_before_exit | 8 | $11,540 | 1.000 |
| q4_2025 | P | C | opposite_side | sequence_residual_override | post_open_morning | post_open_morning | 3_5m_after_open | 6_10m_before_exit | 22 | $11,280 | 0.682 |
| march_2026 | P | C | opposite_side | hard_stop | post_open_morning | post_open_morning | 6_10m_after_open | 6_10m_before_exit | 4 | $11,220 | 1.000 |
| q1_2026 | P | C | opposite_side | hard_stop | post_open_morning | post_open_morning | 6_10m_after_open | 6_10m_before_exit | 4 | $11,220 | 1.000 |
| q1_2026 | C | C | same_side_different_contract | protocol054_fallback | post_open_morning | post_open_morning | 6_10m_after_open | gt_10m_before_exit | 22 | $10,920 | 0.773 |
| q1_2026 | P | P | same_side_different_contract | protocol054_fallback | post_open_morning | post_open_morning | 3_5m_after_open | gt_10m_before_exit | 17 | $10,870 | 0.588 |
| q3_2025 | P | P | same_side_different_contract | sequence_residual_override | post_open_morning | post_open_morning | le_2m_after_open | le_2m_before_exit | 32 | $10,820 | 0.906 |
| q1_2026 | C | C | same_side_different_contract | target | post_open_morning | post_open_morning | 3_5m_after_open | gt_10m_before_exit | 3 | $10,350 | 1.000 |

## Stopping Rule

This Track A branch should stop here before model work. The diagnostics have isolated where Protocol101 may be weak, but the next step requires human trading judgment: review the top rows and choose exactly one strategy hypothesis to test. The current evidence is not a training label because it is counterfactual, non-additive, research-exposed, and not fill-calibrated.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/summary.json`
- Enriched open slots: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/enriched_open_trade_slot_summary.csv`
- Enriched blocked events: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/enriched_blocked_slot_events.csv`
- Open archetypes: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/slot_cost_open_trade_archetypes.csv`
- Blocked event archetypes: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/slot_cost_blocked_event_archetypes.csv`
- Thesis tests: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/slot_cost_thesis_tests.csv`
- Top open trades: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/slot_cost_top_open_trades.csv`
