# AUDIT_PROTOCOL101_TRACK_A_FORENSICS_V1

What is this: Track A Protocol101 forensics packet
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_track_a_forensics_partial_complete_training_still_blocked`

## Bottom Line

Track A now has executable diagnostics for the Protocol101 weak points we can inspect from existing artifacts. The packet does not authorize a new model. It narrows the next work to manual hard-stop review, exact post-exit runner paths, counterfactual-flat Protocol101 slot-cost replay, and side/score interpretation.

## Headline

- Seed-1 Protocol101 trades: `1028`
- Seed-1 Protocol101 PnL: `$319,050`
- Win rate: `0.761`
- Profit factor: `4.421`
- Hard-stop rows: `11` for `$-14,470`
- Runner/giveback evidence rows from full-path oracle: `7496`
- Exact selected-trade runner path rows: `709`
- Exact selected-trade runner path coverage: `0.690`
- Exact selected-trade post-exit path match rate: `0.019`
- Hold/exit action-advantage state rows: `198261`
- Internal slot-cost blocked events: `3723`
- Slot-cost archetype rows: `622`

## Hard-Stop Mechanisms

| mechanism | trades | pnl | median MFE | median MAE | next test |
|---|---:|---:|---:|---:|---|
| unclassified_or_unavoidable_candidate | 8 | $-10,040 | $130 | $-1,290 | No model should act on this row without manual chart/path review and matched controls. |
| path_management_candidate | 2 | $-3,520 | $505 | $-1,760 | Test a post-entry early-MFE giveback/loss guard; this is lifecycle state evidence, not a new entry model. |
| fast_adverse_move_candidate | 1 | $-910 | $140 | $-910 | Inspect pre-entry exhaustion/reversal context; if not separable, treat as unavoidable cost of the playbook. |

## Losing-Day Mechanisms

| mechanism | days | pnl | trades | hard-stop pnl | early-MFE loss pnl |
|---|---:|---:|---:|---:|---:|
| path_management_dominated | 34 | $-16,870 | 102 | $0 | $-25,880 |
| hard_stop_dominated | 5 | $-4,790 | 17 | $-6,940 | $-6,830 |
| single_trade_concentration | 4 | $-4,330 | 8 | $0 | $-900 |

## Runner/Giveback Evidence

The runner audit now includes an exact selected-trade path packet when available. It remains a hindsight diagnostic, but it is much closer to the actual Protocol101 trade log than the broad Protocol199 oracle.
Exact selected-trade path coverage is `709` of `1028` selected trades.

### Exact Selected-Trade Paths

| runner state | rows | post-exit best delta | forced-flat delta | post-positive rate | forced-flat positive rate |
|---|---:|---:|---:|---:|---:|
| runner_extension_candidate | 314 | $849,940 | $530,570 | 1.000 | 0.997 |
| giveback_guard_required | 203 | $267,090 | $-318,060 | 1.000 | 0.000 |
| extension_destroys_value | 119 | $22,710 | $-225,030 | 0.807 | 0.000 |
| ambiguous_runner_state | 20 | $6,740 | $-270 | 1.000 | 0.450 |
| do_not_extend_or_already_right | 53 | $1,140 | $-110,470 | 0.396 | 0.000 |

### Broad Full-Path Oracle Context

Legacy broad-oracle exact selected-trade join coverage is `20` of `1028` selected trade keys.

| runner state | rows | post-exit best delta | forced-flat delta | post-positive rate |
|---|---:|---:|---:|---:|
| runner_extension_candidate | 4539 | $12,617,620 | $8,523,630 | 1.000 |
| giveback_guard_required | 2957 | $3,859,630 | $-4,464,420 | 1.000 |
| extension_destroys_value | 2510 | $461,800 | $-4,598,275 | 0.758 |
| ambiguous_runner_state | 1577 | $326,555 | $-159,055 | 0.846 |
| do_not_extend_or_already_right | 493 | $11,940 | $-552,770 | 0.513 |

## Hold/Exit Action-Advantage Foundation

The lifecycle direction from `engineer-response.md` is now represented as a foundation artifact: `A_hold = Q(hold) - Q(exit now at bid)`. This still does not authorize training. Its purpose is to stop framing runner work as `hold longer` and instead frame it as a causal continuation-advantage decision with explicit missing costs.

- Labeled holding-state rows: `198261`
- Covered Protocol101 selected trades: `709`
- Oracle-best hold fraction at Protocol101 exit: `0.917`
- One-step hold fraction at Protocol101 exit: `0.258`

Interpretation: the high future-best hold fraction says later upside often existed, but the low one-step hold fraction and negative forced-flat runner audit show why naive extension is dangerous. The next lifecycle question is whether continuation advantage survives slot opportunity cost, switching cost, fill/latency uncertainty, and distributional tail risk.

## Internal Slot-Cost Status

A counterfactual-flat Protocol101 event replay is now available. It scores the frozen Protocol101 event policy at every event as if the account were flat, then attributes model-approved entries that actual strict-serial Protocol101 blocked by already holding one contract.

- Hypothetical flat entries: `8179`
- Blocked entry events: `3723`
- Open trades with blocked entries: `2291`
- Open trades where best blocked entry beats open trade: `1109`
- Best-blocked-minus-open total: `$399,510`

This is still not a tradable replay: multiple blocked entries inside one open interval are mutually exclusive, and fill/latency realism is not calibrated.

| split | blocked events | open trades w/ blocked | best-blocked beats open | best-blocked-minus-open |
|---|---:|---:|---:|---:|
| march_2026 | 545 | 315 | 178 | $145,850 |
| q1_2026 | 1310 | 748 | 397 | $208,980 |
| q3_2025 | 832 | 556 | 204 | $-2,760 |
| q4_2025 | 1036 | 672 | 330 | $47,440 |

## Slot-Cost Archetype Decomposition

The slot-cost branch has reached its useful automated stopping point. The decomposition shows concrete weakness hypotheses, but the next step is manual strategy selection, not another model run.

- Open trades with blocked entries: `2291`
- Positive slot-cost open trades: `1109`
- Best-blocked-minus-open total: `$399,510`
- Positive slot-cost total: `$859,990`

| thesis | open trades | slot-cost total | median slot cost | next action |
|---|---:|---:|---:|---|
| losing_open_trade_blocks_positive_later_signal | 485 | $635,240 | $930 | Audit these rows first for pre-entry avoidability versus early exit/loss-cut evidence. |
| opposite_side_later_signal_blocked | 428 | $572,430 | $930 | Treat as a regime-reversal/exit question, not a runner-extension question. |
| sequence_residual_slot_cost | 741 | $472,930 | $380 | Segment by side/time/premium before proposing any gate. |
| long_duration_slot_cost | 396 | $409,130 | $640 | Build mutually exclusive replay for only long-duration positive-slot-cost rows. |
| same_side_later_signal_blocked | 681 | $287,560 | $280 | This is the cleanest candidate for a same-side runner/switching-cost lifecycle diagnostic. |

### Deployed-State Action Log

The older deployed-state action stream is still included as a sanity check. It records open intervals as `holding`, so by itself it cannot reveal counterfactual-flat entries.

| split | status | holding rows | enter rows | observable blocked enters |
|---|---|---:|---:|---:|
| q1_2026 | blocked_missing_counterfactual_flat_protocol101_actions | 16223 | 1254 | 0 |
| q3_2025 | blocked_missing_counterfactual_flat_protocol101_actions | 11815 | 1292 | 0 |
| q4_2025 | blocked_missing_counterfactual_flat_protocol101_actions | 10652 | 1369 | 0 |
| recent_2026 | blocked_missing_counterfactual_flat_protocol101_actions | 10405 | 450 | 0 |

## Next Actions

| priority | item | status | next action |
|---:|---|---|---|
| 1 | PROTOCOL101_HARD_STOP_AUTOPSY_V1 | diagnostic_ready_no_training | Manually review hard-stop rows, then build matched-control rejection/lifecycle tests before any model. |
| 2 | PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1 | exact_path_diagnostic_ready_no_training | Design a causal confirmed-MFE runner/giveback diagnostic using only in-trade state available before extension. |
| 3 | PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_V1 | foundation_complete_training_blocked | Add counterfactual flat-slot opportunity cost, switching cost, fill/latency penalties, and distributional risk targets before lifecycle training. |
| 4 | PROTOCOL101_INTERNAL_SLOT_COST_V1 | counterfactual_flat_diagnostic_ready_no_training | Decompose blocked entries by open-trade archetype and add mutually exclusive replay plus fill/latency realism before any learned defer gate. |
| 5 | PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1 | archetype_decomposition_complete_manual_review_required | Stop automated model work here; manually review top rows and choose exactly one named strategy hypothesis. |
| 6 | PROTOCOL101_SIDE_SPECIFIC_STRATEGY_AUDIT_V1 | diagnostic_ready_no_training | Use side audit to decide whether puts and calls need separate playbook notes. |
| 7 | PROTOCOL101_SCORE_RELIABILITY_V1 | selected_trade_proxy_only | Join rejected candidates for true calibration; do not treat score margin as utility yet. |

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/summary.json`
- Hard-stop classification: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/hard_stop_classification.csv`
- Hard-stop mechanism summary: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/hard_stop_mechanism_summary.csv`
- Losing-day mechanisms: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/losing_day_mechanisms.csv`
- Runner extension audit: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/runner_extension_audit.csv`
- Runner extension examples: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/runner_extension_examples.csv`
- Selected-trade runner join: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/selected_trade_runner_join.csv`
- Exact selected runner state summary: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/exact_selected_runner_state_summary.csv`
- Internal slot counterfactual summary: `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/summary.json`
- Slot-cost archetype summary: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/summary.json`
- Internal slot-cost proxy: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/internal_slot_cost_proxy.csv`
- Side strategy audit: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/side_strategy_audit.csv`
- Score reliability deep dive: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/score_reliability_deep.csv`
- Next actions: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/track_a_next_actions.csv`
