# AUDIT_PROTOCOL101_STRATEGY_SELECTION_PACKET_V1

What is this: Track A strategy-selection packet
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_strategy_selection_complete_first_diagnostic_recommended_training_blocked`

## Bottom Line

Track A has enough evidence to stop broad diagnostics and pick one narrow trader hypothesis for the next diagnostic. It does not have enough evidence to train a neural model.

Recommended first diagnostic: `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`.

Why this one: it is the largest raw slot-cost bucket, and most of the regime-flip damage appears inside it. The first test should stay narrow: a current Protocol101 trade is losing, and a later Protocol101-approved signal appears while the slot is still occupied. The opposite-side subcase is the first manual review bucket.

Loss-reversal / regime-flip overlap: `325` rows, `$517,980` positive slot-cost total, `0.815` of loss-reversal positive slot cost.

## Strategy Ranking

| rank | hypothesis | score | rows | slot-cost total | top-day share | top-5-trade share | causal | specificity | first diagnostic |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1 | 0.833 | 485 | $635,240 | 0.055 | 0.042 | 0.78 | 0.72 | Classify top loss rows into pre-entry avoidable, path-management avoidable, execution artifact, or unavoidable. |
| 2 | PROTOCOL101_REGIME_FLIP_EXIT_GATE_V1 | 0.817 | 428 | $572,430 | 0.062 | 0.046 | 0.90 | 0.88 | Build a mutually exclusive regime-flip replay with switching cost, no overlap, ask-entry/bid-exit, and fill stress. |
| 3 | PROTOCOL101_LONG_DURATION_FALLBACK_EXIT_V1 | 0.695 | 449 | $437,160 | 0.070 | 0.061 | 0.82 | 0.80 | Run mutually exclusive replay for long-duration/fallback rows before training any duration gate. |
| 4 | PROTOCOL101_SAME_SIDE_SWITCH_OR_RUNNER_V1 | 0.630 | 681 | $287,560 | 0.059 | 0.040 | 0.78 | 0.76 | Separate same-contract continuation from same-side contract replacement and charge explicit switching cost. |

## Interpretation

- `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` is first because it has the largest raw evidence and captures most of the opposite-side regime-flip damage.
- `PROTOCOL101_REGIME_FLIP_EXIT_GATE_V1` remains the most important subcase because a baseline-approved opposite-side setup while holding is live-observable.
- `PROTOCOL101_SAME_SIDE_SWITCH_OR_RUNNER_V1` is the disciplined version of the runner question, but it must distinguish hold-current from switch-contract economics.
- `PROTOCOL101_LONG_DURATION_FALLBACK_EXIT_V1` is a useful lifecycle patch, but it overlaps with the loss and regime-flip evidence and should not be first unless manual review rejects loss-reversal.

## Selected Hypothesis Contract

- Mechanism: The current Protocol101 position is failing while a later Protocol101-approved signal appears.
- Decision change: Exit or refuse to keep occupying the slot when causal loss/reversal evidence appears.
- First diagnostic: Classify top loss rows into pre-entry avoidable, path-management avoidable, execution artifact, or unavoidable.
- Label shape: A_exit_or_defer = Q(exit/release slot) - Q(hold current position), charged for missed Protocol101 opportunity.

The selected diagnostic should answer a trader question, not a generic ML question:

> When Protocol101 is losing while holding the only slot, and another Protocol101-approved event appears, is the current trade stale enough to release the slot, or is the later signal hindsight bait/noise?

## Required Next Diagnostic

Build `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` as a no-training diagnostic with:

- all losing-open-trade rows from the internal slot-cost counterfactual, with losing-plus-opposite-side rows reviewed first,
- causal in-trade state at the blocked signal time, including current bid, PnL, MFE, MAE, giveback, duration, spread, quote age if available, and score margin,
- mutually exclusive serial replay comparing keep-holding, exit-now, and exit/switch-to-later-signal where feasible,
- explicit switching cost: exit spread, new entry spread, fill uncertainty, latency risk, slot opportunity cost, and model uncertainty penalty,
- side/time/premium/moneyness buckets, day concentration, delay/slippage stress, and failure examples,
- a hard stop that no neural model is trained until this diagnostic shows a causal, live-observable rule survives costs.

## Evidence Examples

The full examples file contains `100` rows across hypotheses. The first rows are review targets, not training labels.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/summary.json`
- Strategy selection matrix: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/strategy_selection_matrix.csv`
- Selected strategy: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/selected_strategy.csv`
- Rejected alternatives: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/rejected_strategy_alternatives.csv`
- Hypothesis evidence examples: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/hypothesis_evidence_examples.csv`
- Hypothesis overlap matrix: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/hypothesis_overlap_matrix.csv`
- Selection gate checklist: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/selection_gate_checklist.csv`
- Input artifact manifest: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/input_artifact_manifest.json`
- Report: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/report.md`
