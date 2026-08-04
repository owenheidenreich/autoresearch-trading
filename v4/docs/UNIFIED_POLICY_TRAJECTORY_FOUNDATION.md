# DATASET_UNIFIED_POLICY_TRAJECTORY_FOUNDATION_V1

What is this: dataset foundation / unified conservative policy trajectory manifest
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `unified_trajectory_foundation_ready_training_blocked_by_foundation_gates`

## Summary

The existing flat and holding artifacts can be read as `UnifiedPolicyTrajectoryDatasetV1` sources, but this remains a foundation manifest. Neural training is still controlled by the unified readiness packet.

## Coverage

- Flat candidate rows: `3509721` from `DATASET_FULL_SURFACE_ACTION_ADVANTAGE_V1 / Protocol270`
- Holding state rows: `4467517` from `DATASET_POSITION_STATE_ACTION_ADVANTAGE_V1 / Protocol274`
- Sample flat states built: `3`
- Sample holding states built: `3`

## Foundation Gates

- Fill model ready: `False`
- Untouched holdout ready: `True`
- Live no-order parity ready: `False`

## Protocol276 Evidence Kept

- Entry policy selected 165 negative-A_enter trades for -$35,480 PnL.
- Lifecycle timing left 159 overhold/late-exit trades with -$33,275 PnL.
- Actual exits left -$424,160 versus each trade's best observed path PnL.
- Replay skipped 16799 rows, dominated by {'unaffordable_current_equity': 13912, 'missing_contract_quotes': 2887}.

## Next Allowed Work

1. Materialize a versioned trajectory dataset only after choosing a storage layout and untouched block.
2. Expand the DP oracle so flat wait/enter and holding hold/exit values are computed on the same serial account trajectory.
3. Attach Protocol101 baseline actions to every trajectory event for conservative policy improvement.
4. Keep neural training blocked until the readiness packet clears the DP oracle, Protocol101 baseline, fill, and live no-order parity gates.

## Outputs

- Summary: `v4/audit/autoresearch/unified_policy_trajectory_foundation/summary.json`
- Report: `v4/audit/autoresearch/unified_policy_trajectory_foundation/report.md`
- Contract: `v4/audit/autoresearch/unified_policy_trajectory_foundation/trajectory_dataset_contract.json`
- Flat coverage: `v4/audit/autoresearch/unified_policy_trajectory_foundation/flat_coverage_by_split.csv`
- Holding coverage: `v4/audit/autoresearch/unified_policy_trajectory_foundation/holding_coverage_by_split.csv`
- Docs copy: `v4/docs/UNIFIED_POLICY_TRAJECTORY_FOUNDATION.md`
