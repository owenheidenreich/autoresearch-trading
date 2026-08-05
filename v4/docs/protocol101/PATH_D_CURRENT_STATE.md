# Protocol101 Path-D current state

> **SUPERSEDED 2026-08-05 — see [`STATUS.md`](../../../STATUS.md) for current status.**
> Dated 2026-08-01 and overtaken by events. See STATUS.md.

- As of: **2026-08-01**
- Current sequencing: **architecture first; offline foundation steps 1–6 implemented locally and awaiting independent Claude verification**
- Active paper runtime: **unchanged and separate**

This is the shortest current-state pointer for the Path-D redesign. It does not
replace signed Protocol101 authority, promote a model, or change
`PAPER_DEFAULT_PROTOCOL101`.

## Architecture-first rule

Path D is being built as an additive boundary around the legacy system before
any new model work:

```text
historical Databento OPRA + historical ThetaData SPX
  -> canonical market events on received_timestamp_utc
  -> source-neutral features
  -> decision service
  -> ExecutionIntentV1
  -> deterministic governor (sole submit authorizer)
  -> simulated executor and reconciliation evidence
```

The production market-source target remains Databento Live OPRA plus ThetaData
SPX, with IBKR restricted to execution, positions, fills, account state, and
safety. That live bridge is deferred. The current offline overlay establishes
the seams needed to test it without contacting any vendor or broker.

## Current bounded implementation

The additive package is [v4/path_d/](../../path_d/README.md). Its current scope
contains:

- standard-library-only, strict v1 wire contracts and golden fixtures;
- a semantic `intent_id` law that excludes trace and measurement-only timing
  while binding all execution semantics;
- a source-neutral feature builder plus legacy compatibility shim and
  byte/hash parity tests;
- an offline canonical-event decision service using the single
  `received_timestamp_utc` clock;
- a preregistered deterministic exit fixture and deterministic governor with
  upward-only floor, fixed giveback/time stop, feed-loss forced flat, position,
  affordability, freshness, and account limits;
- a full simulated order lifecycle, including partial fill, no fill, cancel
  race, disconnect, `UNKNOWN_RECONCILE`, and reconciliation; and
- quote-supported 100/250/500/1000 ms latency bounds across the six owned paired
  sessions. Those bounds are not actual-fill evidence.

The detailed topology and import laws are in the
[transition architecture plan](training/execution/PROTOCOL101_PATH_D_TRANSITION_ARCHITECTURE_PLAN_2026_08_01.md),
[module map](PATH_D_MODULE_MAP.md), and
[migration manifest](PATH_D_MIGRATION_MANIFEST.csv).

## Preserved boundaries

- No live Databento or ThetaData request occurred.
- No IBKR or broker connection occurred.
- No model was trained or fitted.
- No registry, runtime flag, launchd job, paper default, Protocol158,
  Protocol160, or paper-runtime path was changed.
- Existing feature consumers were not rewired; the compatibility proof is
  additive and legacy behavior remains the control.
- The live IBKR adapter remains step 7 and is not implemented beyond a
  fake-gateway-only stub.

## Evidence and claim boundary

Offline evidence is written under
`v4/audit/autoresearch/path_d_offline_foundation/`:

- `offline_e2e_transcript.jsonl`
- `simulated_execution_transcript.jsonl`
- `latency_bounds.csv`
- `latency_bounds.json`
- `latency_bounds.md`
- `summary.json`

The highest allowed claim is: the offline Path-D contract, decision,
governor, and simulated-execution seams are implemented and locally tested.
This is not evidence of alpha, model quality, real fills, runtime parity, paper
readiness, promotion readiness, or real-money readiness.

## Next gate

Independent Claude verification of migration steps 1–6. Step 7—the real live
gateway/market adapters and any paper-runtime integration—remains deferred and
requires a separate owner-authorized goal.

## Reading order

1. This page.
2. [Transition architecture plan](training/execution/PROTOCOL101_PATH_D_TRANSITION_ARCHITECTURE_PLAN_2026_08_01.md).
3. [Path-D package README](../../path_d/README.md).
4. [Module map](PATH_D_MODULE_MAP.md).
5. [Migration manifest](PATH_D_MIGRATION_MANIFEST.csv).
6. [Walking-skeleton learnings ledger](training/execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md).
7. [Path-D transition plan](training/execution/PROTOCOL101_PATH_D_TRANSITION_PLAN_2026_08_01.md).
