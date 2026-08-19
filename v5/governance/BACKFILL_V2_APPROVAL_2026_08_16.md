# Owner approval — corrected backfill declaration V2 and conditional acquisition

**Signed by the owner in conversation on 2026-08-16:** *"approve declaration V2 and let the
acquisition run if its total is at or below $75."*

This is the fresh owner decision that
[`BACKFILL_DECLARATION_V1.json`](../work/lifecycle-training/BACKFILL_DECLARATION_V1.json)'s
semantic-freeze rule requires after a request-shape change following a cost preflight. It does not
edit, reinterpret, or withdraw anything already on the record.

## What is approved

1. [`BACKFILL_DECLARATION_V2.json`](../work/lifecycle-training/BACKFILL_DECLARATION_V2.json),
   self-hash `4d8f01a71cbebc1e88fd621cca2d349f2e255cf3d0bf50e6a50e63084ac56990`, which requests the
   charter-authorized scope only: for each session, resolve the SPXW universe by symbology, keep
   only symbols whose own OSI expiry equals that session, and price and request exactly those.
2. Execution of the acquisition **conditional on** the corrected exact preflight reporting a total
   at or below **$75.00**. Above that figure the runner stops and the acquisition does not occur.

## What is unchanged

- The **$75 ceiling** of `DEVELOPMENT_CHARTER_2026_08.md` §2. This approval does not raise it.
- The 2026-08-15 STOP receipt and V1 declaration remain immutable records of what was priced and
  decided then. V2 supersedes V1; it does not overwrite it.
- Every other hard rail: no broker contact, no live subscription, no paper or live orders, no real
  money, no unattended jobs, no reserved post-2026-08-05 sessions, no other vendor request.
- Scope is exactly the previously authorized data — same dataset, schemas, date range and
  destination. Only the request shape narrows, from all listed SPXW expirations to the same-day
  expiry the charter names.

## Why the correction was needed

The 2026-08-15 preflight priced `stype_in="parent"` (`SPXW.OPT`), covering every listed SPXW
expiration — measured at 12,828-15,980 instruments per session — and returned $671.90. It stopped
correctly on the request it priced, but that request was broader than charter §2 authorizes.
Measured at the authorized scope: 362-988 contracts per session at $0.0215-$0.0483, with
`definition` at $0.00. Full evidence:
[`BACKFILL_REQUEST_SCOPE_2026_08_16.md`](../research/findings/BACKFILL_REQUEST_SCOPE_2026_08_16.md).

## Enforcement

`ops/acquire_lifecycle_backfill.py` verifies the declaration's self-hash and pinned runner hash,
refuses to acquire without a `PASS` preflight, and enforces the ceiling twice — once as the gate and
again as a running total inside the acquisition loop, so no per-session sequence can walk past it.
It also refuses to download a session whose resolved ladder changed size since its quote.

- Owner: **Owen Heidenreich** — approved in conversation, 2026-08-16.
