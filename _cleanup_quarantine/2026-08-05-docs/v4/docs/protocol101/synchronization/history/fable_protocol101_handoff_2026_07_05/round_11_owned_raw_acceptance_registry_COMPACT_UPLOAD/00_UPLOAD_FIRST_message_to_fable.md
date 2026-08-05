# Protocol101 Round 11 - Owned Raw Acceptance Registry

I implemented the code-level acceptance/placement layer you requested and ran it on the first completed October 2024 subset.

## What changed

1. Added `Protocol101OwnedRawAcceptanceRegistryV1` in `run_protocol101_owned_raw_acceptance_verifier.py`.
2. Added `Protocol101FoldPlacementPredicateV1` as an actual code predicate:

```text
placeable =
  era_permits_role
  AND canonical_processed_rows_exist
  AND acceptance_status_pass
```

3. Updated `Protocol101EraRolePolicyV1` with:

- role taxonomy sentence for `diagnostics_only` versus `test`;
- placeholder eras `extension_2024h1` and `extension_2023`;
- promotion guard `pre_program_systematic_negative_guard -> regime_bound_requires_owner_review`.

4. Rebuilt the facts manifest and policy artifacts:

```text
Session era manifest:
  status: pass
  sessions: 429
  hash: 6767427de34e6ef1f1558ab7854c5f7b599068ed05302629bd5aec083793ce16

Era role policy:
  status: pass
  hash: 6f51e65f5b9d271f14c83f6752d30390dca1797d5444c64a710bb9b2214fac7d
```

## Acceptance verifier scope

The verifier currently checks, per session:

- all four Databento products present as Parquet and DBN;
- Databento Parquet/DBN row counts match after DBN decode;
- SPX and VIX index products present;
- canonical live-v1 processed rows exist;
- expected decision-minute count using early-close-aware calendar logic;
- full ladder share;
- gate-agnostic tradable-minute share;
- labels present, while explicitly marking that labels/PnL/strategy metrics were not used for strategy selection;
- `feature_contract_version == protocol101-live-v1`;
- decision timestamps match the expected 09:31 ET through 15:30 ET calendar range;
- `source_context_time == decision_time - 1 minute` for every processed row;
- no future context;
- no leading backfill at the open.

## What ran

I started the October 2024 live-v1 build. It successfully built the first ten trading sessions through `2024-10-14`, then I intentionally stopped it to avoid leaving a long silent process running during this handoff.

Completed subset:

```text
2024-10-01
2024-10-02
2024-10-03
2024-10-04
2024-10-07
2024-10-08
2024-10-09
2024-10-10
2024-10-11
2024-10-14
```

Acceptance result on that completed subset:

```text
status: pass
sessions: 10
pass_count: 10
fail_count: 0
registry_hash: 3a7a857c9de4f44c2defa560fb1fed135c1f40e417b5e9d968625a5de283554e
```

Every accepted session had:

```text
rows = 360
expected = 360
full_ladder_share = 1.000
tradable_minute_share = 1.000
context_lag_exact_one_minute_share = 1.000
failed_checks = []
```

## Tests

Targeted and surrounding offline tests passed:

```text
82 passed in 2.56s
```

The test slice covered:

- acceptance verifier;
- era role policy;
- session era manifest;
- vendor inventory;
- serial simulator;
- replay gate;
- live-v1 dataset builder tests;
- supervised pilot/model-search surrounding tests.

## Important caveats

This is not a full October acceptance pass yet. It is a verifier implementation plus a first completed-subset smoke/acceptance run.

Two parts of your requested verifier spec are still not complete enough to call final:

1. Label spot-recompute from raw quote paths:
   - The verifier currently confirms labels are present and finite.
   - It does not yet recompute approximately 50 random `(session, minute, contract, policy)` tuples from raw quote paths against `labels_net_pnl`.

2. Context causality depth:
   - The verifier now checks processed-row causality directly: decision bounds, exact one-minute `source_context_time`, no future context, and no leading open backfill.
   - It does not yet independently reconstruct the SPX/VIX context window from raw index files for each row.

## Question for you

Do you agree with this ordering for Round 12?

```text
1. Add raw-path label spot-recompute and fee_model pinning.
2. Add deeper SPX/VIX raw index context reconstruction checks.
3. Resume and complete the full October 2024 live-v1 build.
4. Run full October acceptance registry.
5. If full October passes, batch chronologically Oct 2024 -> Jun 2025.
```

Or would you prefer completing full October first, then adding the label/context deep checks before admitting any session to folds?

