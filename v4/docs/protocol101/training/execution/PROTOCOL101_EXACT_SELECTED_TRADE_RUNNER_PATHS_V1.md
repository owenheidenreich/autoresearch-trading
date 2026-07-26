# AUDIT_PROTOCOL101_EXACT_SELECTED_TRADE_RUNNER_PATHS_V1

What is this: exact selected-trade Protocol101 post-exit runner/giveback path audit
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_exact_selected_runner_paths_complete_training_still_blocked`

## Bottom Line

This packet attaches actual normalized quote paths to Protocol101 selected trades and measures what happened after the frozen exit. It is a hindsight diagnostic for asking better runner/giveback questions, not a tradable runner policy.

## Coverage

- Selected trades: `1028`
- Exact path rows: `709`
- Path skips: `319`
- Exact path coverage: `0.690`

## Headline

- Frozen PnL with paths: `$224,750`
- Hindsight post-exit best delta: `$1,147,620`
- Naive forced-flat delta: `-$123,260`
- Material continuation rate: `87.0%`
- Forced-flat positive rate: `45.4%`

## Runner States

| runner state | rows | frozen pnl | post-exit best delta | forced-flat delta | material continuation | forced-flat positive |
|---|---:|---:|---:|---:|---:|---:|
| runner_extension_candidate | 314 | $105,110 | $849,940 | $530,570 | 100.0% | 99.7% |
| giveback_guard_required | 203 | $76,400 | $267,090 | -$318,060 | 100.0% | 0.0% |
| extension_destroys_value | 119 | $14,740 | $22,710 | -$225,030 | 66.4% | 0.0% |
| ambiguous_runner_state | 20 | $4,430 | $6,740 | -$270 | 95.0% | 45.0% |
| do_not_extend_or_already_right | 53 | $24,070 | $1,140 | -$110,470 | 3.8% | 0.0% |

## Path Skips

| reason | rows |
|---|---:|
| missing_session_or_contract_quotes | 319 |

## Interpretation

A positive post-exit best delta means there was later bid-path opportunity after Protocol101 exited. A negative forced-flat delta means naive hold-to-close would have destroyed value. Runner research should therefore focus on confirmed-MFE state transitions with giveback guards, not a blanket hold-longer rule.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/summary.json`
- Exact selected runner paths: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/exact_selected_runner_paths.csv`
- Runner state summary: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/runner_state_summary.csv`
- Runner split/state summary: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/runner_split_state_summary.csv`
- Runner examples: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/runner_candidate_examples.csv`
- Path skips: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/path_skips.csv`
- Path skip summary: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/path_skip_summary.csv`
