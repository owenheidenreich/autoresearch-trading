# Job 52 — capped paid resume of the Job 51 Tier-0 acquisition

## Result this job is allowed to produce

Acquire and stream-QC only the 21 sessions already frozen by Job 51, preserving the Job-51 V1
zero-cost contract, readiness receipt, failed-zero-gate attempt, and stop receipt as immutable evidence.
This is a money-control overlay, not a new data selection or research result.

The target remains:

- dataset `OPRA.PILLAR`;
- schema `cmbp-1`;
- `stype_in=raw_symbol`, `stype_out=instrument_id`;
- the exact 21 request objects and 2,373,877,845 expected CMBP-1 records in
  `v5/work/cmbp-metadata-census/TIER0_ACQUISITION_SCOPE_V1.json`;
- raw DBN plus streaming decoder/QC evidence under `/Volumes/AR_TRADING_DATA/cmbp-tier0/job51`.

No P&L, return, strategy outcome, model fit, threshold search, live feed, broker action, or order is in
scope.

## Job-number resolution

Job 49's sealed `later_owner_gated_jobs` table used 50–54 as conditional sequencing labels, not as
reservations or authority; its own heading says those names are sequencing rather than authority and the
canonical status register controls. The subsequently owner-authorized and registered data jobs 50–52
therefore supersede those forecast labels. The forecast value/economic candidate formerly labelled 52
has not been opened and must receive a new non-colliding number if it is ever separately authorized.

## Owner authority and money boundary

The owner authorized this paid resume in the current conversation on 2026-08-24 with two hard caps:

- maximum lifetime quote commitment for any one session across all starts/retries: **USD 1.50**;
- maximum cumulative quote committed to time-series starts across the entire frozen acquisition:
  **USD 32.00**.

Both are exact decimal comparisons. Booleans, strings, nonfinite values, negative values, signed zero,
or otherwise malformed SDK results fail closed. A projected per-session or aggregate commitment equal to
its cap is allowed. A quote that would put either commitment above its cap is durably recorded and the
process stops before the corresponding `timeseries.get_range` call.

The budget ledger is deliberately conservative and reconstructible:

1. A cost quote alone consumes no acquisition budget.
2. Immediately before each `timeseries.get_range`, a durable `TIMESERIES_CALL_START` record commits the
   fresh quote. The committed total is the sum of all such starts.
3. A failed or indeterminate time-series start still consumes its full quote. A retry requires a fresh
   quote and creates another commitment against both the same session's lifetime USD 1.50 allowance and
   the all-session USD 32.00 allowance.
4. A crash after a passing quote but before the start record creates no commitment; the quote is stale
   and a restart must quote again.
5. The prior Job-51 quote of `0.950392448902` is permanently disclosed but contributes `0` to the paid
   budget because Job 51 made no time-series start.
6. The ledger reports observed SDK quotes and committed quote exposure. It never calls that amount an
   invoice or actual spend. Actual vendor invoice cost remains `UNKNOWN` because quote and fetch are not
   an atomic price lock.

Any projected per-session commitment above USD 1.50, any projected all-session commitment above USD
32.00, or any malformed quote is terminal under this authority. There is no automatic application retry
after such a stop.

## Immutable evidence chain

The paid-resume program contract and readiness receipt must bind, by raw SHA-256 and semantic identity
where applicable:

- the Job-51 V1 plan and program contract;
- the Job-51 V1 sterile JUnit report and local readiness receipt;
- `JOB51_ZERO_COST_GATE_STOP_RECEIPT_V1.json`;
- the exact external attempt, journal, watermark, stop receipt, and call markers named by that receipt;
- the frozen Tier-0 scope and its Job-49/Job-50 source evidence;
- the unchanged shared decoder and all unchanged Job-51 V1 implementation/tests;
- the Job-52 plan, contract, paid overlay implementation, operations scripts, and focused tests;
- installed Databento, DBN, HTTP, and compression source/binary identities used at execution.

Job-51 V1 artifacts are never overwritten or re-sealed. The only recognized pre-authority positive quote
is the exact attempt and stop receipt above; any different legacy positive-quote evidence is refused.

## External call law

For each missing session bundle, the only allowed vendor sequence is:

1. revalidate paid readiness, exact scope, external-volume identity/free space, all earlier attempt
   journals, all existing finals, and reconstructed committed budget;
2. journal and fsync an exact `metadata.get_cost` start marker;
3. call `metadata.get_cost` for the exact frozen request;
4. journal and fsync the observed result and both cap decisions;
5. if both caps pass, journal and fsync the identical `timeseries.get_range` start and updated committed
   total with nothing but required durable records between quote and start;
6. call `timeseries.get_range` with the identical market parameters, `stype_out=instrument_id`,
   `limit=None`, and the session's private staging path;
7. fsync, hash, journal, stream-decode, reconcile, and atomically publish the session directory bundle.

No batch API, parent symbol, wildcard, alternate dataset/schema/stype, out-of-scope date/symbol, manual
prequote, full-session pandas load, or hidden retry is permitted. Every quote and time-series attempt is
accounted across all Job-51/52 attempt directories, not only the winning process.

A successful `TIMESERIES_CALL_RESULT` creates a global local-recovery barrier until its exact staged DBN
or already-renamed final is stream-QC'd and journaled as `SESSION_PUBLISHED` or, after restart,
`SESSION_RECOVERED`. No credential construction or later vendor call for any session may cross that
barrier. If both the result-bound staging bytes and an exact final disappeared, the run stops rather than
paying for a duplicate fetch. Before credential access, the runner also refuses proxy, CA-bundle, netrc,
TLS-key-log, or related Requests/Python TLS environment overrides.

## Destination, crash, and resume law

`/Volumes/AR_TRADING_DATA` must already be the mounted, writable external APFS volume with UUID
`8CBA2FD2-1446-4439-866C-3BEA6C297E30`. The mountpoint is never created. Device identity and adequate
free space are rechecked before every vendor pair. All destination artifacts refuse symlinks, hard-link
aliases, cross-device entries, and unexpected tree children.

One nonblocking global flock is shared with Job 51 without rewriting the persistent Job-51 lock bytes.
An adjacent self-hashed Job-52 binding names the frozen scope, paid contract, and paid readiness; those
identities are independently revalidated under the lock before every pair. Journals are append-only,
hash-chained, fsynced record by record, mirrored by exclusive per-call markers, and anchored by a
self-hashed watermark so a valid-prefix truncation is refused. A root-level Job-52 attempt-set anchor is
updated before credential access and refuses a missing whole paid-attempt directory, preventing silent
rollback of the reconstructed quote ledger. Every paid attempt header and anchor entry also carries a
globally contiguous, 1-based attempt ordinal plus the exact prior paid attempt ID and prior header-record
hash. This lineage—not a random UUID or wall-clock timestamp—orders cost-only failures as well as
time-series commitments. Any paid attempt after a terminal over-cap or malformed quote is refused before
credential access; a cost or time-series transport error may be retried only in a new process attempt.

These are fail-closed accidental/single-component rollback controls, not cryptographic protection from an
attacker who can coherently rewrite every unkeyed destination artifact. Coordinated rewrite protection
would require an independently retained outside head or signature; that stronger threat remains outside
this acquisition's claim.

Raw DBN and QC are staged together in `<session>.bundle.part` and published with one same-volume directory
rename only after all checks pass. A restart validates every earlier attempt and every existing final
before credential access. Existing current-seal bundles may be reused only when their raw hash, DBN
header, source journal pair, cost commitment, request projection, decoder invariants, and QC self-hash all
reconstruct exactly. Otherwise it stops before a vendor call.

## Shared streaming decoder and QC pass law

Historical DBN iteration and live callbacks use the same bounded per-instrument state machine in
`v5/research/cmbp_stream.py`. Historical acquisition never loads a full session into pandas.

Each session must reconcile exactly to its frozen census record count and one-to-one raw-symbol mapping.
Trade signing uses only the immediately prior same-instrument book record with
`prior.ts_recv < trade.ts_recv`; receive-time ties are counted and excluded. Global `ts_recv` regression
is fatal. `ts_event` regressions are diagnostic. A live disconnect, reconnect, or gap clears all prior
book state. Historical transport-gap/reconnect telemetry is reported as unavailable rather than invented;
stream/instrument silence and system/heartbeat/flag diagnostics remain bound.

The aggregate PASS requires all 21 exact session bundles, 2,373,877,845 reconciled CMBP-1 records, all
1,001 frozen session-symbol mappings, zero out-of-scope calls, every time-series start paired to a fresh
passing quote, per-attempt and aggregate call/commitment reconciliation, all decoder causal/count
invariants, and precise disclosure of duplicates, failures, quotes, compressed bytes, gaps/reconnects,
ties, causal classifications, and actual-invoice `UNKNOWN`.

## Pre-call readiness

Before any paid-resume credential read or vendor call, a separate immutable Job-52 readiness receipt must
be built by a sterile subprocess that:

- runs only the frozen Job-51 decoder/Tier-0 tests plus the Job-52 paid-resume tests;
- ignores repository/environment pytest options and disables third-party plugin autoload;
- has no Databento credential in its environment and blocks network use inside the tests;
- binds the exact test-case population/identity, the program contract, every required file, and installed
  SDK source/binary manifests;
- records zero credential reads, zero external calls, zero downloads, and zero acquisition initiation.

The runner revalidates this receipt before credential access and before every vendor pair.

## Terminal outcomes

PASS means only that the exact frozen raw acquisition and decoder QC completed under the quoted caps. It
is not trading evidence and it does not claim the vendor invoice equals the quote ledger.

STOP is expected on cap failure, malformed quote, prior evidence drift, mount/device/free-space change,
concurrent runner, credential/auth failure, vendor transport failure, DBN/QC mismatch, record-count or
mapping mismatch, causal receive-time regression, path anomaly, or receipt/journal inconsistency. The
stop and all calls already made remain durable; retries follow the laws above and never erase exposure.
