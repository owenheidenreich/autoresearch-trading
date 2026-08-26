# Job 55 — exact-$2.00 lifetime-cap amendment and retry

## Authorized result

Resume the exact 21-session Job-51 Tier-0 acquisition after Job 52's transport failure and cap stop.
The only changed authority is the per-session lifetime committed-quote cap: it is exactly **USD 2.00**.
The total committed-quote cap remains **USD 32.00**. Dataset `OPRA.PILLAR`, schema `cmbp-1`,
`stype_in=raw_symbol`, `stype_out=instrument_id`, all 21 date/symbol requests, 2,373,877,845 expected
records, the shared bounded decoder, and `/Volumes/AR_TRADING_DATA` remain unchanged. No batch or split
request, wildcard, live feed, model, outcome, broker contact, or order is authorized.

## Preserved evidence and opening ledger

Every Job-51 and Job-52 artifact remains immutable. Job 55 must independently reconstruct the exact
Job-52 stopped state before credential access and before every vendor pair:

- Job-52 readiness semantic SHA-256 `6e4e4977d5ff74f06ff866409cbaacdc833a58dc85039d51d7c83ae7e9face6e`
  and raw SHA-256 `46ddc5631a69e26a9ec0284a1ec5d30723ebc737eb2762f1db2cab35ceed9c0c`;
- paid attempt 1 `0ebce8b7-cce2-4f9b-a6e5-94b6df3544e7`, whose journal raw SHA-256 is
  `8fcca63ee9b291eb36b131b0e384bcc3c9a5387ef9d1de65063e62580e6282c2` and whose exact
  `$0.950392448902` time-series start failed with `BentoError`;
- paid attempt 2 `2e8e5b32-d2f5-4a75-9f33-4c80414d184c`, whose journal raw SHA-256 is
  `3e13780e6b2aa8ad471a5498211dd632d729468d49e79c8e71b8b930e11524e8` and whose identical
  passing quote was stopped before a start by the former USD 1.50 cap;
- the exact Job-52 anchor, lock binding, marker, watermark, stop-receipt, and 89,745-byte partial-file
  identities bound by the Job-55 program contract.

The opening committed ledger is therefore exactly `$0.950392448902` for session `2025-08-22` and in
total. The Job-51 quote and Job-52's second quote consume zero because neither produced a Job-52
time-series start. Job 52's terminal cap result is superseded only by this exact current-conversation
amendment; every other prior malformed/cap/evidence state remains terminal.

## Call and commitment law

Before each missing session request, under the unchanged shared Job-51 flock:

1. revalidate the Job-55 readiness seal, mounted-volume identity/free space, exact Job-51/52 evidence,
   every Job-55 attempt, every existing final, and the reconstructed combined ledger;
2. durably journal an exact `metadata.get_cost` start and call it for the frozen request;
3. durably journal the finite, unsigned numeric SDK quote and projected same-session and total exposure;
4. stop before time series if same-session exposure would exceed `$2.00` or total exposure would exceed
   `$32.00`;
5. otherwise durably journal `TIMESERIES_CALL_START`, irrevocably committing the fresh quote, then call
   the identical `timeseries.get_range` request with `stype_out=instrument_id`, `limit=None`, and a
   private staging path;
6. fsync/hash the stream, iterate it with the shared decoder, reconcile the frozen record/mapping/QC
   laws, publish raw+QC as one same-volume directory rename, and only then consider another vendor pair.

Every failed or indeterminate Job-55 time-series start remains fully committed. A retry needs a fresh
quote in a new process attempt. A quote without a start is stale and contributes zero. A cap or malformed
quote result is terminal under Job 55. Actual invoice cost remains `UNKNOWN` because quote and fetch are
not atomic.

Each Job-55 process attempt may durably start at most one time-series request. After a successful stream,
QC, and atomic publication it exits; the next frozen session requires a fresh process attempt and complete
pre-call reconstruction. A cost-call or time-series error also ends that process attempt. This makes every
cross-session transition pass the same credential, evidence, mount, budget, and recovery gates.

## Namespace, crash, and recovery law

Job-55 attempts, controls, staging, final sessions, and aggregate receipt live only under
`/Volumes/AR_TRADING_DATA/cmbp-tier0/job55`. The stopped Job-51/52 external tree is read-only evidence;
Job 55 does not add a child, final, or receipt to it. Job 55 opens the existing Job-51 `RUN_LOCK_V1`
without rewriting it and binds its own
readiness beside a Job-55 attempt-set anchor. Job-55 journals are append-only, hash-chained, fsynced,
marker-mirrored, watermark-bound, and globally ordered by a contiguous attempt ordinal plus prior-header
lineage. Missing or unexpected files, links, foreign devices, attempt deletion, valid-prefix truncation,
or coordinated state inconsistency stop before credentials or calls.

A successful time-series result creates the same global local-recovery barrier: its exact staged bytes
or renamed final must be QC'd and journaled locally before credential construction or any later vendor
call. If result-bound bytes and final both disappear, stop rather than pay for a duplicate. Partial bytes
from the failed Job-52 result are immutable evidence and are never resumed, deleted, or treated as a
complete session.

The known failed session `2025-08-22` may receive at most one Job-55 time-series start regardless of
quote size. This is the singular retry authorized here and prevents a cheap quote from creating an
unbounded retry interpretation. It is also the first Job-55 vendor target and may receive at most one
Job-55 cost call: a quote transport error, malformed/cap quote, failed start, or indeterminate start ends
Job 55 before any later session is contacted. Other sessions remain governed by the exact USD 2.00
lifetime and USD 32.00 total commitment ceilings.

The Job-55 adoption receipt, anchor, and journals detect missing, truncated, rolled-back, or mutually
inconsistent destination components. Their unkeyed hashes do not claim protection against a coordinated
rewrite of every destination artifact; that requires an independently retained head or signature.

## QC and aggregate PASS

The unchanged decoder law requires exact census record counts, exact raw-symbol/instrument mappings,
strict `prior.ts_recv < trade.ts_recv`, receive-time tie exclusion, fatal global receive-time regression,
diagnostic event-time regression, and honest gap/reconnect/silence/flag evidence. Historical iteration
must remain bounded and may not load a full session into pandas.

PASS requires 21 exact final bundles, 2,373,877,845 reconciled records, 1,001 session-symbol mappings,
zero out-of-scope calls, combined Job-52+55 commitments at or below `$2.00` per session and `$32.00`
total, and one receipt binding every Job-51/52/55 journal, quote, call, source result, raw hash, decoder
summary, QC receipt, and publication. The receipt reports duplicate/failed attempts and actual invoice
`UNKNOWN`; it is not strategy or trading evidence.

## Pre-call seal

Before any Job-55 credential read or vendor call, a separate immutable Job-55 readiness receipt must:

- run only the frozen decoder, Job-51, Job-52, and Job-55 offline tests in a sterile subprocess;
- bind the exact clean test population, this plan/contract, all Job-51/52 immutable inputs, new code and
  tests, and the installed Databento/DBN/HTTP/compression/TLS dependency identities;
- record zero credential reads, external calls, downloads, and acquisition initiation;
- pass independent pre-call review and reconstruct exactly immediately before execution.

## Terminal outcomes

PASS means only exact acquisition and decoder QC under the amended quote caps. Any transport failure,
cap/malformed quote, evidence drift, path/device/free-space anomaly, causal/count/mapping mismatch, or
receipt inconsistency is a preserved STOP. No automatic action may widen the cap, switch transport, or
erase prior exposure.
