# Compact action-value policy fails before execution costs

**Date:** 2026-08-14  
**Status:** negative economic evidence; long-0DTE selector branch closed  
**Scope:** the single signed 48-parameter `compact_interaction_entry` fit with the frozen
`serial_action_advantage_120m` target and first-enter-above-$0-WAIT selector

## Result

The compact policy was finally evaluated under the exact signed law after a mechanical-only V3 reseal.
It selected 35 trades across 150 out-of-fold sessions and abstained on 115. At midpoint entry and midpoint
exit—removing the spread entirely—the selected trades produced:

- **-$15.93 mean per trade**;
- **-$3.72 mean per scored session**;
- **-$52.50 median per trade**;
- **14.29% winning trades**; and
- **-$282.50 worst trade**.

The decisive predeclared kill condition therefore failed. The evaluator stopped before reading bid
economics, the composition-matched control, the shuffled-policy economics, corrected confidence bounds,
chronological sign tests or the $10,000-account risk tests. Those fields are `null`, not failures and not
missing implementation.

## Integrity boundary

- The architecture count is the built canonical count:
  `computed_parameter_counts()['compact_interaction_entry'] == 48`.
- The fit receipt records `economics_read=false` and `operating_point_tuned=false`.
- Real and shuffled models used the identical fit path over the same 423,053 out-of-fold candidate rows.
- The V3 declarations reproduce the signed V2 research-law projection exactly at
  `3a7e26e78ad8c6f1c972ba779fd7a6ac1b640e95979dda49ce74891248499627`.
- The evaluation receipt self-hash reproduces and records `bid_economics_read=false`,
  `reserved_sessions_used=false` and `promotion_or_order=false`.

This is a measured negative, not Job 39's void ceiling-threshold defect. The selector had an attainable,
structural $0 WAIT floor and fired 35 times.

## Consequence

The result closes the declared long-0DTE action-value selector. There is no adjacent threshold, seed,
horizon, architecture or subgroup retry. Because the policy loses even with the spread removed, better
fills cannot rescue this exact trading game. The next research cycle must quantify a materially different
game and the evidence needed to test it; it may not relabel this branch.

## Evidence

- Fit declaration:
  `v5/work/entry-exit-attribution/ACTION_VALUE_FIT_DECLARATION_V3.json`
- Evaluation declaration:
  `v5/work/entry-exit-attribution/ACTION_VALUE_EVALUATION_DECLARATION_V3.json`
- Fit receipt:
  `v4/audit/autoresearch/causal_day_action_value_fit_2026_08_14_attempt003/receipt.json`
- Economics receipt:
  `v4/audit/autoresearch/causal_day_action_value_economics_2026_08_14_attempt003/receipt.json`
- Midpoint-only selected rows:
  `/Volumes/AR_TRADING_DATA/derived/causal_day_action_value_economics_v3/real_selected_mid_only.parquet`

