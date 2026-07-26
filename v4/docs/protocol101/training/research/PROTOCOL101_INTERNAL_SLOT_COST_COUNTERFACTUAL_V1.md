# AUDIT_PROTOCOL101_INTERNAL_SLOT_COST_COUNTERFACTUAL_V1

What is this: Protocol101 internal slot-cost counterfactual
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_internal_slot_cost_counterfactual_complete_training_still_blocked`

## Bottom Line

This packet closes the previous Track A blocker by scoring the frozen Protocol101 event policy at every event as if the account were flat. It then attributes model-approved entries that actual strict-serial Protocol101 could not take because one contract was already open.

This is evidence about Protocol101's internal slot opportunity cost, not a replacement strategy. Blocked events inside a single open interval are mutually exclusive, and the audit has no calibrated live fill model.

## Headline

- Hypothetical flat Protocol101 entries: `8179`
- Blocked entry events while actual Protocol101 was holding: `3723`
- Actual open trades with at least one blocked entry: `2291`
- Open trades where best blocked entry beats actual open-trade PnL: `1109`
- Non-additive blocked candidate PnL: `$1,243,220`
- Best-blocked-minus-open total: `$399,510`

## Split Summary

| split | flat entries | blocked events | positive blocked | open trades w/ blocked | best-blocked beats open | best-blocked-minus-open |
|---|---:|---:|---:|---:|---:|---:|
| march_2026 | 1086 | 545 | 402 | 315 | 178 | $145,850 |
| q1_2026 | 2564 | 1310 | 933 | 748 | 397 | $208,980 |
| q3_2025 | 2124 | 832 | 648 | 556 | 204 | $-2,760 |
| q4_2025 | 2405 | 1036 | 725 | 672 | 330 | $47,440 |

## Interpretation

- If this audit shows large positive best-blocked-minus-open values, Protocol101 may be taking some early trades that consume the only slot before stronger later Protocol101 signals.
- If it is small or concentrated in one split, the internal slot-cost hypothesis is weaker than the runner/giveback and hard-stop questions.
- This must not be used as a training label until fill/latency realism and a mutually exclusive counterfactual replay are added.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/summary.json`
- Hypothetical flat entries: `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/hypothetical_flat_protocol101_entries.csv`
- Blocked events: `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/blocked_protocol101_internal_slot_events.csv`
- Open trade slot summary: `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/open_trade_slot_summary.csv`
- Split summary: `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/split_summary.csv`
