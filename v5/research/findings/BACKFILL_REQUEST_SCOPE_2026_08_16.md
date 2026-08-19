# The $671.90 STOP priced 40 expirations we never asked for

**2026-08-16. No money spent, no data downloaded.** The 2026-08-15 preflight was correct procedure
on a request that was wrong, and this finding separates those two facts so neither is lost.

## What changed for the bot

The quote backfill that job 46 needs is **not** unaffordable. Measured at the authorized scope with
the vendor's own cost endpoint, a session costs **$0.02–$0.05**, not the $0.85 the STOP receipt
recorded. The plan's data phase is alive.

## What happened

The exact preflight requested `cbbo-1m` with `stype_in="parent"`, symbol `SPXW.OPT`. That covers
**every SPXW expiration listed on the session** — measured by symbology resolution at
**12,828 / 15,836 / 15,980** instruments on three sampled dates. The declaration then applied the
same-day rule *after saving*, so the request paid for roughly forty expirations in order to keep
one.

The development charter §2 authorizes only "SPXW contracts whose own OSI expiry equals each session
date". **The priced request was therefore broader than the owner approved**, and its $671.90 was
never the price of the authorized data.

The mis-scoping was invisible in the receipt because the parent-scope price is nearly flat across
years ($0.77 in 2022, $0.94 in 2024) — it tracks the size of the whole SPXW universe, which barely
changes, rather than 0DTE activity, which grew fivefold over the same period. Recorded acquisition
costs for the owned corpus varied 5x ($0.0128–$0.0646) precisely because they *were* 0DTE-scoped.

## The measurement

An outcome-free probe priced both scopes over identical windows on every 100th session of the
frozen inventory ([receipt](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/request_scope_probe.json)):

| | parent scope | 0DTE symbol scope | ratio |
|---|---:|---:|---:|
| mean per session | $0.8483 | $0.0134 (traded subset) | **63.4x** |
| projected over 794 sessions | $673.56 | $10.62 | — |

The probe's symbol lists came from owned OHLCV, which records only contracts that *traded*, so its
totals are lower bounds. Priced on the **full quoted ladder** resolved from symbology:

| Session | Full 0DTE ladder | `cbbo-1m` cost |
|---|---:|---:|
| 2022-06-01 | 362 contracts | $0.0215 |
| 2024-03-15 | 812 contracts | $0.0483 |
| 2025-07-31 | 988 contracts | $0.0476 |

`definition` is **$0.00** at every one of the 794 sessions in the original receipt — the $23.03
definitions line in the pre-acquisition estimate was too high, not too low.

The probe reproduces the STOP receipt's parent-scope figure to within 0.2% ($673.56 projected
against $671.90 measured), which is what makes it a diagnosis of the request rather than a dispute
about the vendor's arithmetic.

## What was corrected

- [`BACKFILL_DECLARATION_V2.json`](../../work/lifecycle-training/BACKFILL_DECLARATION_V2.json)
  (self-hash `4d8f01a7…`) requests the same-day ladder explicitly: resolve the session's SPXW
  universe by symbology, keep only symbols whose own OSI expiry equals the session, then price and
  request exactly those. The same-day rule now binds at **request** time.
- [`acquire_lifecycle_backfill.py`](../../ops/acquire_lifecycle_backfill.py) implements it and
  enforces the $75 ceiling twice — as the preflight gate and as a running total inside the
  acquisition loop — so no per-session sequence can walk past it. It also refuses to proceed if a
  session's ladder has changed size since its quote. Nine focused tests, including one that fails
  if `cbbo-1m` is ever priced at parent scope again.

## What stands

The 2026-08-15 STOP stands as the correct decision on the request it actually priced, and its
receipt is untouched. V1's declaration and its cost receipt remain on the record; V2 supersedes
rather than edits. No money was spent under either.

## Evidence

- Probe: `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/request_scope_probe.json`
- Superseded STOP: `…/cost_preflight.json` ($671.90, parent scope)
- Corrected preflight: `…/cost_preflight_v2.json`
- Tools: [`probe_backfill_request_scope.py`](../../ops/probe_backfill_request_scope.py),
  [`acquire_lifecycle_backfill.py`](../../ops/acquire_lifecycle_backfill.py)
