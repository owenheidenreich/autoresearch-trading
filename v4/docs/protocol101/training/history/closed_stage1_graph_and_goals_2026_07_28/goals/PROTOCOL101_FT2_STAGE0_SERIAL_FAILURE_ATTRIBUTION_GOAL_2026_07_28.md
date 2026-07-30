# Goal: Protocol101 FT2 Stage-0 Serial Failure Attribution

## Objective

Explain the exact `-$4,190` Ledger-B disadvantage of the Stage-0 P5
HOLD/EXIT HGB without training, tuning, or proposing a repaired model from
outcome inspection.

Answer:

> Did the HGB lose because it exited common P5 trades badly, because its early
> exits reopened the account and admitted bad replacement P5 entries, or
> because both mechanisms failed?

This is accounting and causal routing only. It must stop before any new entry
or lifecycle model is designed or trained.

## Frozen Inputs

Hash and consume only:

- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/decision.json`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/pilot_results.json`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/serial_replays.parquet`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/serial_identity_receipts.json`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/episode_replays.parquet`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/predictions.parquet`
- `v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/hashes.sha256`
- `v4/model/protocol101_serial_simulator_v5.py`

Do not refit models or rebuild validation outcomes from other data.

## Output

Write only to:

`v4/audit/autoresearch/protocol101_ft2_stage0_serial_failure_attribution_attempt001/`

Evidence grade:

`analysis_only_non_promotable`

## Step 1: Verify The Frozen Result

Before attribution:

- validate every attempt002 checksum;
- require terminal decision `stop_no_preliminary_exit_signal`;
- reproduce the P5 serial PnL, HGB serial PnL, and `-$4,190` delta exactly;
- verify five validation sessions and unchanged ordered P5 intent streams;
- verify no duplicate trade or entry identities; and
- verify fee and two-clock fields are present.

If the frozen delta or identities cannot be reproduced, return
`attribution_blocked_artifact_mismatch`.

## Step 2: Identity Decomposition

For each validation session, classify admitted trades by the exact identity:

```text
(session, decision_time_ns, contract_id, canonical_strike_slot)
```

Create:

- common entries admitted by both P5 and HGB;
- P5-only admitted entries;
- HGB-only admitted entries; and
- ordered P5 intents skipped by each comparator.

For common entries, attribute:

- P5 exit timestamp, reason, holding time, fee-adjusted PnL;
- HGB exit timestamp, reason, holding time, fee-adjusted PnL; and
- HGB-minus-P5 exit PnL.

For side-only entries, report their full fee-adjusted PnL contribution.

The identity accounting must satisfy exactly:

```text
HGB total PnL - P5 total PnL
  = common-entry exit-PnL delta
  + HGB-only entry PnL
  - P5-only entry PnL
```

If it does not, report the exact residual and return
`attribution_blocked_accounting_residual`.

## Step 3: Re-Entry And Occupancy Attribution

Per session and pooled, report:

- admitted trade count for P5 and HGB;
- first admitted trade identity and PnL;
- PnL from first trades only;
- PnL from second-and-later trades;
- minutes held per trade;
- time from one exit to the next admitted entry;
- fees from common, P5-only, and HGB-only trades;
- skipped intent count and skipped intent timing;
- daily-stop activation and minimum equity;
- exit action counts and elapsed-minute distribution; and
- call/put, time-of-day, and premium breakdown of side-only entries.

Do not run cooldown, max-trade, threshold, or alternative-exit
counterfactuals. This Goal diagnoses the frozen result; it does not search for
a profitable patch.

## Step 4: Frozen Routing Logic

Return exactly one:

- `entry_timing_and_reentry_is_binding`
- `one_step_exit_target_is_binding`
- `both_entry_and_exit_are_binding`
- `attribution_inconclusive`
- `attribution_blocked_artifact_mismatch`
- `attribution_blocked_accounting_residual`

Define:

```text
common_exit_contribution =
  sum(HGB PnL - P5 PnL on common admitted entries)

entry_stream_contribution =
  sum(HGB-only PnL) - sum(P5-only PnL)
```

Route:

- `entry_timing_and_reentry_is_binding` when
  `common_exit_contribution >= 0` and `entry_stream_contribution < 0`.
- `one_step_exit_target_is_binding` when
  `common_exit_contribution < 0` and `entry_stream_contribution >= 0`.
- `both_entry_and_exit_are_binding` when both contributions are negative.
- `attribution_inconclusive` when neither contribution is negative despite
  the reproduced negative total, or evidence is too incomplete to apply the
  rules.

These signs are accounting routes, not statistical proof.

## Required Artifacts

- `input_verification.json`
- `trade_identity_decomposition.parquet`
- `session_attribution.csv`
- `occupancy_and_reentry.json`
- `accounting_reconciliation.json`
- `routing_decision.json`
- `report.md`
- `hashes.sha256`

## Hard Boundaries

Do not:

- fit, refit, rescore, or tune any model;
- construct a new target, feature, threshold, cooldown, or strategy;
- inspect training, final OOF, G9, holdout, recorder, sealed, shadow, or paper
  outcomes;
- run a new economic campaign;
- select or promote a model;
- contact a broker, submit paper orders, or download data; or
- change repository runtime, promotion, launchd, recorder, or real-money
  state.

## Highest Allowed Claim

`Frozen Stage-0 serial failure accounting attribution complete.`
