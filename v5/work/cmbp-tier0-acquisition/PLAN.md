# Job 51 — Tier-0 CMBP acquisition and shared streaming decoder

**Registered job:** 51 in [`v5/STATUS.md`](../../STATUS.md), 2026-08-24

**Current authority:** the owner's current-conversation instruction to acquire exactly the 21 frozen
zero-cost sessions and build the historical/live streaming decoder. This authority is limited to the
call law and destination below. It grants no paid request, widened scope, live subscription, model or
threshold work, broker contact, order, or unattended runtime mutation.

## Purpose

This job tests whether consolidated top-of-book (`cmbp-1`) data can be acquired, decoded, mapped, and
causally ordered across a fixed outcome-blind sample without loading a full session into pandas. It is
infrastructure evidence for historical/live parity. It does not test a strategy, return, fill, label,
P&L, or model.

## Frozen scope

The only acquisition scope is
[`TIER0_ACQUISITION_SCOPE_V1.json`](../cmbp-metadata-census/TIER0_ACQUISITION_SCOPE_V1.json):

- artifact type `JOB51_TIER0_ACQUISITION_SCOPE_V1`;
- semantic self-hash `87050c2b3ad248d4d96c5f6945b883241cdd89a53a2f9902d6e70baf5218a099`,
  recomputed from canonical JSON after removing only `scope_sha256`;
- raw file SHA-256 `1760ec585c23f5213a70a46c5eb6a76400f8cad27569fb1e4ef64abfe41b2463`;
- source Job-50 receipt raw SHA-256
  `319c05bd1ca06edd14f6c4a847debb4a64880dd2f3a5c80450af86fa41dce076`;
- source Job-50 response raw SHA-256
  `b759b48641ff529196356ce795eec8f53434452203c8b628cdd73bfa342cef71`;
- dataset `OPRA.PILLAR`, schema `cmbp-1`, input symbology `raw_symbol`;
- 21 chronological sessions, including the 2026-07-02 early close;
- 1,001 exact session-symbol memberships and 2,373,877,845 census records; and
- destination root exactly `/Volumes/AR_TRADING_DATA`.

The runner reconstructs the 20 evenly spaced zero-cost selections from the complete Job-50 response,
adds the early-close session if absent, and requires exact equality with the frozen scope. It also
requires every selected symbol list, request bound, cost, and record count to agree across the Job-49
declaration, Job-50 response, and Job-51 scope before credential read.

## External call law and money boundary

After all local gates pass, the runner constructs one authenticated `databento==0.77.0` historical
client. In chronological session order it may invoke only:

1. `metadata.get_cost` once for the exact frozen session request; then
2. if and only if the returned finite decimal is exactly `0`, `timeseries.get_range` once with the
   identical dataset, start, end, symbols, schema, and `stype_in`, plus `stype_out=instrument_id` and
   the attempt-local `.part` path.

There is no batch API, parent symbol, `ALL_SYMBOLS`, alternate schema, limit, date widening, automatic
application retry, or acquisition call without a new immediately preceding quote. A malformed,
nonfinite, negative, or positive quote stops before that session's time-series request. A transport or
decode failure stops the attempt; resuming a missing session requires another fresh cost quote before
another time-series request. The receipt reports exact observed quotes and that acquisition was
initiated, but leaves actual vendor invoice cost `UNKNOWN`; a zero quote is not an atomic invoice lock.
It never generalizes the zero-price observation beyond these calls.

Under the destination-wide run lock and before credential read or a new attempt, the runner validates
every prior Job-51 attempt for this frozen scope across readiness-seal identities. A corrupt, widened,
or unaccounted prior journal stops before vendor contact. Any prior nonzero or malformed quote remains
terminal and prevents later quotes or downloads unless the owner makes a new hash-bound decision.

The API key is read only from `DATABENTO_API_KEY` or the current repository-root `.env` after its owner,
mode, non-symlink, and assignment-shape checks pass. The file may contain other well-formed key
assignments, but it must contain exactly one nonempty `DATABENTO_API_KEY` assignment; values for all
other keys are ignored and the file is never evaluated as shell code. The older `v4/.env` is not read.
The key is never written, hashed, printed, or placed in an exception receipt, and is validated only
through the pinned SDK constructor. Local tests use fake clients and synthetic records only.

## Destination and crash law

All large artifacts live below:

`/Volumes/AR_TRADING_DATA/cmbp-tier0/job51/`

The immutable layout is:

- `attempts/<attempt_id>/ACQUISITION_JOURNAL_V1.jsonl` — append-only, hash-chained, fsynced state;
- `attempts/<attempt_id>/JOURNAL_WATERMARK_V1.json` and exclusive `call-markers/*.json` — independent
  terminal identity and per-call evidence that make valid-prefix journal truncation detectable;
- `attempts/<attempt_id>/sessions/<session>.bundle.part/` — in-flight DBN plus QC evidence;
- `sessions/<session>/data.cmbp-1.dbn.zst` — a completed, streamed, decoded, hash-verified session;
- `sessions/<session>/SESSION_QC_V1.json` — immutable per-session QC in the same atomically published
  directory bundle; and
- `receipts/JOB51_ACQUISITION_QC_RECEIPT_V1.json` — terminal aggregate evidence.

A final session directory is published only after its byte hash and streaming QC pass. Publication is
one same-volume atomic directory rename and a parent-directory fsync, so data and QC become visible
together. Existing final bundles are reusable only when their raw hash and per-session QC reconstruct
successfully against the same frozen inputs and pre-call readiness seal. Their original acquisition
attempt journal remains part of the aggregate provenance. A partial file is never treated as complete,
appended to, or silently overwritten. No protected repository or pre-existing external-drive data is
moved, deleted, or replaced.

The runner never creates `/Volumes/AR_TRADING_DATA`. Before credential read and before every vendor
pair it requires that path to remain the writable mounted APFS volume with UUID
`8CBA2FD2-1446-4439-866C-3BEA6C297E30`, the same recorded device ID, and adequate free bytes. The
planning guard reserves the greater of 100 GB and 32 bytes for every not-yet-published census record;
32 bytes/record is a conservative storage guard, not a measured compression claim. An unplug, remount,
device change, read-only transition, or free-space shortfall stops before the next quote/request.
All destination components are opened without following symlinks and must stay on the pinned volume.
A nonblocking exclusive run lock, bound in its contents to the scope and readiness seal, is held for the
entire process; a second Job-51 runner stops before credential read or vendor contact.

## Shared historical/live decoder

`v5/research/cmbp_stream.py` owns one state machine over an iterable of DBN records. Historical files
and future live clients pass records through the same `accept(record)` method; the live path additionally
may pass explicit connection control events to `note_disconnect`, `note_reconnect`, and `note_gap`.
This job exercises no live connection.

The decoder:

- consumes one record at a time and keeps only per-instrument prior state plus counters;
- accepts only CMBP-1 data records and declared symbol-mapping/control records;
- preloads historical mappings from DBN metadata and updates them from `SymbolMappingMsg` records;
- refuses an instrument ID that is unmapped, ambiguously mapped, or mapped outside the exact frozen
  raw-symbol set;
- proves nondecreasing receive time in stream order; receive-time regression is fatal, while event-time
  regression is reported as a diagnostic because `ts_recv`, not `ts_event`, is the causal clock;
- classifies a trade (`action == T`) only against the immediately prior same-instrument record;
- admits a causal prior touch only when `prior.ts_recv < trade.ts_recv`; equal receive times are counted
  and excluded, never ordered by file position or event time;
- excludes missing, undefined, locked, and crossed prior books from signed counts;
- counts at-bid, at-ask, inside, outside, and ambiguous prints without retaining a session frame; and
- records explicit disconnect/reconnect/gap controls separately from observed market-silence diagnostics.

The state machine exposes plain immutable event/QC values so historical acquisition and future live
capture can share semantics without importing the live client or contacting a service.

## QC pass law

Every per-session QC artifact binds the scope, source census response, exact request projection, fresh
cost quote, raw DBN SHA-256 and byte count, DBN metadata, decoder code hash, SDK identities, and journal
records. A session passes only when all of these hold:

- DBN metadata says `OPRA.PILLAR`, `cmbp-1`, `raw_symbol` to `instrument_id`, and the exact request
  bounds and symbols;
- the streaming decoded CMBP-1 record count equals that session's Job-50 census count exactly;
- the decoded instrument IDs and raw-symbol mapping are a one-to-one exact match to the frozen Job-50
  resolution for the session;
- no unmapped, extra, cross-expiry, out-of-window, wrong-rtype, truncated, or receive-clock-regressing
  record occurs; event-clock regression is reported but is not silently promoted to causal time;
- every causal prior used for trade classification satisfies the strict receive-clock inequality;
- tied receive clocks are reported and excluded;
- explicit system/gap/reconnect observations and maximum observed stream/per-instrument silence are
  reported, with `UNKNOWN` used where historical DBN has no explicit connection telemetry; and
- all totals reconcile from the per-session artifacts into the aggregate receipt.

The aggregate PASS requires 21/21 session PASS artifacts, exactly 2,373,877,845 decoded CMBP-1 records,
exactly 1,001 scoped session-symbol memberships, no scope widening, and no quoted paid request; actual
vendor invoice cost remains `UNKNOWN`. Across every
source attempt for the frozen scope, including earlier readiness-seal identities, it requires at least 21 cost calls and at least
21 time-series attempts, exactly 21 successfully published session bundles, and every time-series
attempt immediately preceded by a fresh exact-zero quote for the identical request. Duplicate attempts
for the same frozen session are disclosed; incomplete or failed attempts are never omitted from the
aggregate call totals. Any observed nonzero quote makes aggregate PASS impossible without a new owner
decision. Reuse of a completed session remains stricter: its bundle must bind the current readiness seal.

## Pre-call readiness seal

Before vendor call one, `LOCAL_READINESS_RECEIPT_V1.json` must be written and validated. It binds the
raw identity of this plan; the raw and semantic identities of the frozen scope and
`PROGRAM_CONTRACT_V1.json`; the Job-50 source receipt/response; the Job-49 declaration; both research modules; both operation scripts; both
focused test files and their passing JUnit report, plus the installed Databento version and exact SDK
source/binary identities used for cost, time-series streaming, HTTP streaming, and DBN iteration,
including `databento-dbn==0.56.0`. It states that no
credential was read and no external call ran while sealing. The acquisition runner reconstructs the
seal before credential read and again before each vendor pair. Every journal header, session QC, and
aggregate receipt binds the readiness receipt's semantic and raw-file hashes.

## Deliverables

- `v5/research/cmbp_stream.py` — shared streaming decoder and QC state machine;
- `v5/research/cmbp_tier0.py` — frozen-input verification, acquisition journal, session QC, and receipt;
- `v5/ops/acquire_cmbp_tier0.py` — authenticated chronological runner;
- `v5/ops/seal_cmbp_tier0_readiness.py`, `PROGRAM_CONTRACT_V1.json`, and
  `LOCAL_READINESS_RECEIPT_V1.json` — precredential execution/readiness seal;
- `v5/tests/test_cmbp_stream.py` and `v5/tests/test_cmbp_tier0.py` — synthetic/fake-client refusal and
  pass coverage;
- external raw DBN, session QC, journal, and aggregate receipt under the exact destination; and
- Job-51 closure in `STATUS.md` plus the external evidence index.

## Required checks before completion

1. Focused tests inspect and pass without network access.
2. The shared decoder round-trips synthetic historical and live-shaped iterables to identical results.
3. Fake-client tests prove positive, malformed, nonfinite, stale, or mismatched cost quotes stop before
   time-series call one for that session.
4. Crash/reuse tests prove `.part` files and unbound existing finals cannot become PASS evidence.
5. Every downloaded file is streamed through the decoder; no full-session pandas load exists.
6. The aggregate receipt reconstructs independently from immutable session artifacts.
7. `./.venv/bin/python v5/ops/check_project.py` passes before project-health is claimed.
