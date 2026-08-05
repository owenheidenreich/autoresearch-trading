# Protocol101 Round 12 - Acceptance Verifier V2

I implemented your Round 11 critique as `Protocol101OwnedRawAcceptanceRegistryV2` and reran acceptance on the currently completed October 2024 subset.

## Fixes implemented

### 1. Verifier version stamping

Every acceptance record now carries:

```text
schema_version = Protocol101OwnedRawAcceptanceRegistryV2
verifier_version = 2
```

`Protocol101FoldPlacementPredicateV1` now requires:

```text
era_permits_role
AND canonical_processed_rows_exist
AND acceptance_status_pass
AND verifier_version >= 2
AND not_early_close_fold_session
```

This prevents the earlier v1 ten-session smoke registry from being grandfathered into folds.

### 2. Ladder quality renamed and strengthened

The old `full_ladder_share` headline was removed from pass/fail semantics.

The verifier now reports:

```text
ladder_shape_ok_share
tradable_minute_share
mean_tradable_candidates
min_tradable_candidates
near_atm_tradable_share
```

The shape check remains only as `ladder_shape_ok_share`; the meaningful population checks are now candidate count and near-ATM tradability.

### 3. Label checks strengthened

The old `labels_present = label_finite_share > 0` check was replaced with:

```text
label_finite_share
label_nonzero_share
label_positive_share
label_negative_share
```

The verifier now rejects the all-zero label placeholder case that would have passed v1.

### 4. Raw CBBO label spot recompute

The verifier now samples label tuples per session, maps processed contract IDs back to Databento raw symbols, reads raw `cbbo-1m` Parquet, and recomputes labels from:

```text
entry = ask at decision time
exit path = bid after decision through min(entry + hold, forced_flat)
net = (exit_bid - entry_ask) * 100
fee_model = gross_no_fees if exact identity holds
```

For the current run:

```text
70 / 70 spot checks matched
fee_model = gross_no_fees
observed outcome reasons = stop_hit, target_hit, time_exit
```

Forced-flat-capped was not sampled in the 14-session subset, so this still needs targeted sampling before final fold admission.

### 5. Early-close handling

The verifier flags:

```text
early_close_session = true
```

and the placement predicate refuses early-close sessions for fold placement. This avoids making the current 16:00-close constants silently govern half days.

### 6. Mutation-style tests

Added tests proving verifier checks can fail for:

- old verifier version;
- all-zero labels;
- empty candidate population;
- zero-lag/future context;
- Databento raw-symbol mapping.

## V2 acceptance result

I completed/build-verified October 2024 through `2024-10-18` and ran v2 acceptance on the contiguous subset:

```text
2024-10-01 through 2024-10-18
14 trading sessions
```

Result:

```text
status: pass
schema_version: Protocol101OwnedRawAcceptanceRegistryV2
verifier_version: 2
session_count: 14
pass_count: 14
fail_count: 0
registry_hash: cf9107b369dc4673c60ef61d9f54cc430582b31ec23ead8e89a3a76e09c50aee
```

All 14 sessions show:

```text
rows = 360
expected = 360
ladder_shape_ok_share = 1.000
tradable_minute_share = 1.000
context_lag_exact_one_minute_share = 1.000
spot_checks = 5/5
fee_model = gross_no_fees
failed_checks = []
```

The weakest candidate-population day in this subset is `2024-10-18`:

```text
mean_tradable_candidates = 19.99
near_atm_tradable_share = 0.851
label_nonzero_share = 0.976
```

## Tests

The surrounding offline test slice passed:

```text
87 passed in 2.37s
```

## Build note

I attempted to finish the remainder of October 2024. It completed:

```text
2024-10-15
2024-10-16
2024-10-17
2024-10-18
```

The build then became idle on `2024-10-21` with no CPU for several minutes, so I stopped it deliberately rather than leave a silent process running. The next engineering fix is not a methodology change; it is a resumable month/batch runner with per-session timeout/logging so one stuck session cannot stall the whole raw-data acceptance pipeline.

## Remaining gaps before full fold admission

1. Full October 2024 must be built and accepted under v2.
2. The raw-label spot sampler should be upgraded from "5 deterministic tuples per session" to the stricter stratified coverage you requested:
   - policy 0/1/2;
   - stop-hit;
   - target-hit;
   - time-exit;
   - forced-flat-capped;
   - near/far offset.
3. Context reconstruction currently verifies processed-row causality. It still does not fully rebuild derived SPX/VIX context columns from raw index files.
4. Candidate-population thresholds are broad v2 floors. They are not yet calibrated from the larger accepted corpus.

## Question for Round 13

Do you agree that the next step should be:

```text
1. Build a resumable per-session/per-month batch runner with timeout and logs.
2. Finish full October 2024 live-v1 build.
3. Upgrade label spot sampling to targeted branch coverage, including forced-flat-capped.
4. Run full October v2 acceptance.
5. Then batch Oct 2024 -> Jun 2025 chronologically.
```

Or should the targeted label/context deep checks be completed before spending more wall-clock time on the remaining October build?

