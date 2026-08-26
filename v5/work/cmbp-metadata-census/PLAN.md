# Job 50 — exact CMBP metadata census runner and external-response gate

**Registered job:** 50 in [`v5/STATUS.md`](../../STATUS.md), 2026-08-24

**Current authority:** local build only

**Current execution contract:** [`PROGRAM_CONTRACT_V4.json`](PROGRAM_CONTRACT_V4.json), semantic
SHA-256 `01dbb71e7e58d08adf05114c2e89c74fa8870b653e527a66df083ae5b2408794`, raw SHA-256
`34bc793896c29e6675e539b7bec9cb3d03cf4e189fc5bc053e4f3f1b524829d1`. It is a sealed semantic
overlay on preserved [`PROGRAM_CONTRACT_V3.json`](PROGRAM_CONTRACT_V3.json),
[`PROGRAM_CONTRACT_V2.json`](PROGRAM_CONTRACT_V2.json), and
[`PROGRAM_CONTRACT_V1.json`](PROGRAM_CONTRACT_V1.json).

**Independent audit seal:** [`INDEPENDENT_TEST_AUDIT_V1.json`](INDEPENDENT_TEST_AUDIT_V1.json),
semantic SHA-256 `12bee42a434af787fcefb837e97f35562f2b4b430696552c2345ba170026a2ff`, raw SHA-256
`34adbc70cb5c57215a94b587eb18ca8d6e87abd8465f3b0d006ad4d53394b656`, closes all eleven frozen
coverage gaps on test-source SHA-256 `a9301f6a8aa6e6170e99be64017cfac93bbbb570f1b641fd369b9cda7ed2ca8a`.
V2 evidence remains forbidden until the mechanical V4/V2 path projection compiles, the unchanged
26-test source passes again, and the post-seal identity delta is independently rechecked.

### First-seal audit disposition and side-by-side repair

The first local seal passed the then-frozen 26 names, but an independent test audit found eleven
material coverage gaps. Its exact V1 quartet is therefore preserved in place as
`INTERIM_SUPERSEDED_AFTER_INDEPENDENT_AUDIT`, not accepted as current or terminal readiness:

- `TEST_RESULTS_V1.xml` raw SHA-256 `7ffac7b8205dd89ffb660489eb91c40c1c1f4d2bc69cb285bb4692dae052a993`;
- `SYNTHETIC_CALL_JOURNAL_V1.jsonl` raw SHA-256 `b5e0de9e03e52b6e448851cb05a236b9b6deec0e9c3ba8eacc0de2727ea5892d`;
- `SYNTHETIC_METADATA_RESPONSES_V1.json` raw SHA-256 `6b16fb0c1ed5386eb7e4bd4e5af87de56b9e4077162931d1537d8eb9c534760d`;
  and
- `LOCAL_READINESS_RECEIPT_V1.json` semantic SHA-256
  `cf9fcd8efcfec59dbba76e801b9e9bacc5eb02ae19739ff7c8181c477b3b01cc` and raw SHA-256
  `83715142fce65df685c4e18bba5495101e2b270361edefaae3e8979cb2783e75`.

Those files may not be moved, deleted, renamed, overwritten, or rebuilt. A later audit-cleared
generation uses only side-by-side `TEST_RESULTS_V2.xml`, `SYNTHETIC_CALL_JOURNAL_V2.jsonl`,
`SYNTHETIC_METADATA_RESPONSES_V2.json`, and `LOCAL_READINESS_RECEIPT_V2.json`. The V2 receipt must bind
the immutable V1 quartet and the strict self-hashed
`INDEPENDENT_TEST_AUDIT_V1.json`, whose eleven exact gap identifiers are frozen by V4. After V4 seals,
external precredential code refuses the V1 receipt and accepts only the V4-bound V2 receipt.

V4 also closes the external-provenance gap exposed during the strengthened audit. The private
executor defaults to synthetic provenance and accepts no caller-selected `source`, provenance, seal,
or external-context argument. External context is closure-captured only inside the exact future
owner-authorized public runner and becomes active only after its full on-disk revalidation; there is no
module-global token, setter, or separately callable external executor. Local tests may inspect that
production closure graph and prove counterfeit, raw-client, fake-capability, private-call, and keyword
forgery refusals. External wrapper construction likewise uses a closure-hidden receipt context created
only by full on-disk reconstruction, never a module-global receipt seal/type or caller mapping. Tests
may privately mint an exact-class capability solely to prove mutated on-disk authority refuses before
call one. A separate negative-test mint creates the exact public capability class with closure-hidden
`production_usable=false`; it must stop before call one and every receipt path when disk is otherwise
valid. The public core revalidates disk before that pre-call refusal so mutation tests exercise their
natural gates. Receipt build/finalize take the opposite order: exact type, then the production-usable
check immediately, before any binding read or receipt disk-context reconstruction. The closure-backed
production mint has no module-global capability seal, imports nothing, and
accepts only `type(client) is` the already-loaded exact pinned Databento Historical class after the
credential and SDK-identity gates; a fake or duck client cannot become production-usable. Neither kind
of local fixture may emit an `externally_supplied` response, nested structural parser receipt, or
top-level external PASS, even in a temporary directory. The audit and V2 readiness receipt therefore
state explicitly that the positive external path remains unexecuted pending a later owner-authorized
vendor attempt.

## What this job is for

Job 50 turns Job 49's sealed, outcome-blind Databento request declaration into a fail-closed metadata
runner. The future census can answer which declared SPXW sessions Databento reports as available in
`OPRA.PILLAR` `cmbp-1`, how many records each exact session-symbol request would contain, and the quoted
cost of those exact requests. That availability-and-price evidence is useful only for deciding whether
subminute data acquisition is worth a later owner authorization. It cannot show that a strategy exists,
that the owner or a model can select profitable trades, or that any quoted data may be downloaded.

The present authorization permits only the local runner, fake-client tests, synthetic response
artifacts, and a local readiness receipt. It does **not** permit constructing an authenticated client,
reading a Databento key, authenticating, contacting Databento, making an API request, downloading data,
spending money, opening an outcome, fitting a model, or placing an order.

## V2 external-receipt correction

V1 correctly reused Job 49's `CMBP_CATALOGUE_PREFLIGHT_RECEIPT_V1` parser, but incorrectly named that
artifact as Job 50's final external receipt. The inherited parser receipt is intentionally hard-coded as
`VALIDATOR_REHEARSAL_ONLY`, says Job 49 achieved no external preflight, and refuses to claim actual
vendor availability. Re-labeling it after a vendor run would make the evidence contradict itself.

[`PROGRAM_CONTRACT_V2.json`](PROGRAM_CONTRACT_V2.json) repairs only that classification. The original
Job 49 receipt remains an untouched nested structural check. A future completed vendor attempt instead
emits a top-level `JOB50_EXTERNAL_METADATA_RECEIPT_V1`, which binds the authorization, exact response,
raw and normalized result digests, call journal, cost, count, ceiling, and the nested parser receipt. Its
availability claim is limited to the exact sealed declaration and the authenticated entitlement at that
attempt; it is never a universal schema, date, parent-symbol, price-boundary, or population claim.

V2 also requires `external-run` to natively verify the sealed local-readiness receipt—including current
runner/CLI hashes and current SDK source identities—before it reads a credential or imports/constructs
the client. This repair grants no vendor authority. The local receipt binds both preserved V1 and current
V2 contract identities, and all new synthetic journals use the V2 semantic identity.

### V3 closure of the external evidence boundary

A final pre-seal audit of V2 found four additional interface details, without opening any external
action. [`PROGRAM_CONTRACT_V3.json`](PROGRAM_CONTRACT_V3.json) freezes an attempt-local raw file for the
unchanged nested parser receipt, the exact external integrity counters and authorization-effect text, an
opaque authorized-client capability that the external core alone accepts, and a total/idempotent stop
normalizer. It also states plainly that credential rotation and conversation authorship are trusted
owner attestations, not facts code can cryptographically establish, and extends the default no-retry
source identity through installed `urllib3`. The preserved first-generation synthetic journal and local
receipt bound V3 as their effective identity. V4 now supersedes only the audit-cleared execution and
readiness generation while preserving all three earlier contracts and that historical evidence.

## Sealed input and exact future call population

The runner accepts exactly the preserved Job 49 declaration
[`CMBP_CATALOGUE_DECLARATION_V1.json`](../human-policy-foundation/CMBP_CATALOGUE_DECLARATION_V1.json):

- declaration semantic SHA-256 `5677e2401677c7ce0be1c82ab009fb00547e8b2a8b4bdbe8895a64df5a59adad`;
- declaration raw-file SHA-256 `51066b09a0a6add8b2bed407c2a8b585f0689825a1065339abb986831b1b59ba`;
- request-manifest SHA-256 `17854671eb64a9576d933975749e3d6a301abbbd37b6ce496098a533e1cf288a`;
- 1,014 source sessions, of which 813 are `REQUEST_CANDIDATE` and 201 are
  `EXCLUDED_KNOWN_PRE_COVERAGE` before 2023-03-28; and
- 42,726 exact session-symbol memberships, never widened to a parent symbol.

A complete future attempt invokes exactly 2,440 SDK methods in this order:

1. `metadata.get_dataset_range` once; then
2. for each of the 813 request sessions in declaration order, exactly one `symbology.resolve`, one
   `metadata.get_cost`, and one `metadata.get_record_count`, in that order.

No other method is legal. In particular, `timeseries`, `download`, batch retrieval, live access, and
schema substitution are not available through this runner. The 201 known pre-coverage sessions remain
excluded and generate no calls. A manually supplied external-response JSON or a response's own
`method_attestation` is not evidence that this call law was followed.

## Frozen SDK adapter

The external adapter is bound to installed Python package `databento==0.77.0`. A version mismatch stops
before client construction or any call. Under that version:

| Frozen descriptor method | SDK keyword mapping |
|---|---|
| `metadata.get_dataset_range` | `dataset` unchanged |
| `symbology.resolve` | `dataset`, `symbols`, `stype_in`, and `stype_out` unchanged; descriptor `start` becomes SDK `start_date`; descriptor `end` becomes SDK `end_date` |
| `metadata.get_cost` | `dataset`, `start`, `end`, `symbols`, `schema`, and `stype_in` unchanged |
| `metadata.get_record_count` | `dataset`, `start`, `end`, `symbols`, `schema`, and `stype_in` unchanged |

The runner first recomputes every frozen descriptor's `request_sha256`. Only after that check may it
create a temporary SDK-argument mapping. The symbology keyword rename is therefore an adapter detail:
the declaration object and its hash remain byte- and semantics-preserved. The adapter may neither add a
`limit` nor omit, reorder, broaden, normalize, strip, split, or replace a symbol.

The bound SDK source makes one top-level `requests` invocation per declared SDK metadata-method
invocation and configures no automatic retry. Job 50 adds no retry. Low-level HTTP transaction count is
unobserved and may differ because redirects are transport behavior. The local receipt also binds the
exact `Historical` export, constructor, and endpoint-constructor sources so client construction cannot
silently drift at the same package version. Any later SDK, constructor, import-path, or retry behavior
change requires a new contract version and local review before vendor contact.

### Dataset-range caveat

`metadata.get_dataset_range` reports a dataset/entitlement range, not a schema-specific `cmbp-1` range.
It cannot by itself prove that `cmbp-1` exists on a session. A session counts as externally checked only
when its exact symbology, cost, and record-count calls all complete and the result passes the sealed Job
49 response laws. A dataset range spanning 2023-03-28 never permits silent coverage inference,
`cbbo-1m` fallback, or retention of a missing session as an available one.

## Future vendor-run gate

No vendor attempt may start under the current local-build authority. A future attempt additionally
requires all of the following at the exact, non-overridable canonical path
`v5/work/cmbp-metadata-census/authorizations/<attempt_id>/VENDOR_RUN_AUTHORIZATION_V2.json`:

- a fresh current-conversation owner authorization limited to the four metadata methods;
- a binding to this contract and the sealed Job 49 declaration;
- `credential_rotation_attested: true`, meaning the previously flagged credential was rotated or was
  affirmatively established not to be the exposed credential, without recording the secret itself;
- `metadata_only: true` and an exact copy of the four-method allowlist; and
- a finite, non-negative decimal string `quoted_acquisition_ceiling_usd` chosen by the owner before the
  first call.

The authorization also freezes distinct random canonical UUIDs `authorization_id` and `attempt_id`,
canonical issuance and expiry no more than 24 hours apart, `one_attempt_only: true`, direct V1/V2/V3
semantic and raw contract identities, the declaration bytes, and the sealed local-readiness receipt. It is strict
canonical JSON: duplicate, unknown, missing, nonfinite, or noncanonical fields stop. After every local
and authorization gate passes but before credential read, the runner exclusively writes and fsyncs
`authorization-consumptions/<authorization_sha256>.json`; an existing marker or attempt directory
refuses replay. Any failure after consumption spends the authorization. A retry requires new
authorization and attempt identities, not an in-place resume.

The ceiling is a decision boundary for the quote returned by the census. It is not acquisition authority
and the runner has no acquisition path. A result above the ceiling stops as `STOP_OVER_QUOTED_CEILING`.
The authorization artifact is intentionally absent from the local readiness build. Tests may use only
an explicitly in-memory fake authorization, fake `DATABENTO_API_KEY` mapping, and private fake-client
construction seam. Local `credentials_read=0` means no real or stored vendor secret, `.env` value,
keychain value, or process credential was read. The fake seam is identified as synthetic test input and
cannot emit externally supplied response, nested-parser, or external PASS evidence.

## Attempt identity, durable call accounting, and crashes

Every later vendor attempt uses its authorization-bound `attempt_id` and exclusively creates:

`v5/work/cmbp-metadata-census/external-attempts/<attempt_id>/`

The directory is never reused, resumed, overwritten, or repaired. It contains versioned artifacts:

- `CALL_JOURNAL_V1.jsonl` — append-only, hash-chained call evidence;
- `EXTERNAL_METADATA_RESPONSES_V1.json` — written only after complete success;
- `NESTED_STRUCTURAL_PARSER_RECEIPT_V1.json` — untouched, attempt-local Job 49 structural-parser evidence;
- `EXTERNAL_METADATA_RECEIPT_V1.json` — written only after response validation; or
- `ATTEMPT_STOP_V1.json` — a named, non-pass diagnostic when the runner can durably finalize a failure.

Before each SDK invocation, the runner appends and fsyncs a `CALL_START` record containing the attempt,
call ordinal, exact method, session or global scope, and frozen request hash. After the method returns it
appends and fsyncs either `CALL_RESULT`, containing a canonical digest of the raw returned value, or
`CALL_ERROR`, containing only a closed error class and no credential text. Every append advances the
journal hash chain. The completed response binds the terminal journal hash, journal file hash, attempt
ID, exact 2,440-call count, per-method counts, and every request/result pairing. The external receipt
reconstructs and verifies those facts; it never accepts self-attestation as a substitute.

Every journal, response, consumption marker, and receipt write must complete the exact byte count,
fsync the file and directory, then strictly reread and validate the canonical on-disk bytes and hashes.
A PASS uses the reread object, never only the pre-write in-memory object. A short, partial, malformed, or
mismatched file remains a named STOP and is never overwritten or resumed.

A process crash can leave `CALL_START` without a result. That directory is permanently partial and can
never emit or later gain a pass response. A caught SDK error gets a durable `CALL_ERROR` and named stop;
an uncatchable crash is diagnosed as `STOP_PARTIAL_OR_CRASHED_ATTEMPT` by offline inspection. No call is
silently replayed, and no partial result is converted into success. Any retry is a new attempt, requires
a new unique directory, and requires another fresh owner authorization; there is no in-place resume.

The public synthetic entrypoint internally constructs `SyntheticMetadataClient` and accepts no
arbitrary client. The public external entrypoint receives the complete effective V3 authorization at
the V2-named canonical artifact path and
path—not a bare hash—and re-reads and re-verifies it, the consumption marker, both contracts, the
declaration, the local-readiness receipt, and the current SDK identity immediately before the first SDK
invocation.

## Claim and outcome boundary

Even an externally completed Job 50 receipt may claim only:

- the normalized calendar-date projection of the dataset range returned for the authenticated
  entitlement, bound to the raw SDK-result digest but not presented as the exact discarded raw JSON;
- the exact declared sessions whose complete metadata responses passed;
- actual per-session record counts and cost quotes, summed rather than extrapolated;
- the total quoted acquisition cost and its relation to the predeclared ceiling; and
- exact call-accounting and artifact-integrity facts.

It may not claim a zero-price boundary beyond the observed requests, population event prevalence,
execution quality, expected P&L, a strategy edge, model readiness, or authority to acquire data. No
strategy outcome, return, fill, label, P&L, reserved-session economics, quote, trade, or time-series
record is read by this job.

## Local-build deliverables

- `v5/research/cmbp_metadata_census.py` — sealed-declaration adapter, fake/external attempt state
  machine, durable call-journal verifier, and response/receipt validation;
- `v5/ops/run_cmbp_metadata_census.py` — CLI whose local rehearsal path accepts only a fake client and
  whose external path fails without the separate authorization artifact;
- `v5/tests/test_cmbp_metadata_census.py` — exact refusal/pass suite;
- `v5/research/cmbp_metadata_census_receipt_v2.py` and
  `v5/ops/record_cmbp_metadata_census_readiness_v2.py` — audit-superseding native V2 receipt builder
  and exclusive writer;
- `SYNTHETIC_METADATA_RESPONSES_V2.json` — fake-client response bound to the V4 identity and V2
  synthetic call journal;
- `SYNTHETIC_CALL_JOURNAL_V2.jsonl` — local, no-network call-accounting rehearsal;
- `TEST_RESULTS_V2.xml`; and
- `LOCAL_READINESS_RECEIPT_V2.json` — a self-hashed receipt proving only current local readiness.

The V2 local receipt must bind this plan, preserved V1/V2/V3 contracts, current sealed V4, the immutable
V1 evidence quartet, the independent eleven-gap audit, the preserved Job 49 contract/declaration/receipt,
all Job 50 code and unchanged audited tests, the V2 synthetic response and call journal, the V2 JUnit
report, SDK version, endpoint method sources, `databento/common/http.py`, and the bound `requests` and
`urllib3` default transport sources plus Databento export/client-constructor sources. It also binds exact
expected call counts, `positive_external_pass_executed_locally=false`, zero external calls, no real/stored
credentials read, zero outcomes read, and zero purchase or acquisition spend initiated. Its terminal status is
`JOB50_LOCAL_RUNNER_READY_ONLY` and its authorization effect is `NONE`.

## Required local tests

1. `test_fake_census_calls_exact_four_methods_in_frozen_order_and_arguments`
2. `test_symbology_adapter_renames_start_end_only_after_descriptor_hash_validation`
3. `test_descriptor_hash_tamper_forbidden_method_and_scope_widening_fail_before_calls`
4. `test_sdk_version_mismatch_fails_before_calls`
5. `test_vendor_exception_leaves_durable_error_journal_and_no_final_artifact`
6. `test_partial_or_malformed_vendor_response_fails_without_silent_drop`
7. `test_zero_cost_with_positive_count_is_retained_and_zero_count_stops`
8. `test_attempt_paths_are_exclusive_and_partial_attempt_cannot_resume_or_overwrite`
9. `test_external_client_construction_requires_fresh_authorization_credential_and_numeric_cap`
10. `test_fake_suite_never_imports_sdk_authenticates_or_opens_socket`
11. `test_every_invocation_has_durable_start_and_result_pair_bound_to_response`
12. `test_external_receipt_wraps_structural_parser_receipt_and_scopes_claims`
13. `test_external_run_verifies_local_readiness_and_sdk_before_credential_access`
14. `test_external_receipt_binds_auth_raw_normalized_cost_count_and_ceiling`
15. `test_vendor_authorization_is_fresh_attempt_bound_and_consumed_once`
16. `test_core_external_execution_reverifies_full_authorization_not_bare_hash`
17. `test_real_client_cannot_use_synthetic_public_entrypoint`
18. `test_sdk_transport_source_and_requests_zero_retry_identity_are_bound`
19. `test_job49_stop_aliases_are_normalized_at_every_job50_boundary`
20. `test_external_receipt_rejects_counterfeit_external_pass`
21. `test_external_receipt_rejects_raw_or_normalized_digest_tamper`
22. `test_partial_attempt_classifier_rereads_on_disk_journal_and_never_passes`
23. `test_nested_parser_receipt_is_attempt_local_durable_and_bound`
24. `test_external_integrity_and_authorized_client_capability_are_exact`
25. `test_owner_attestations_are_explicit_manual_governance_inputs`
26. `test_urllib3_default_retry_identity_is_bound`

Tests may invoke injected Python fake methods to exercise accounting. They may not construct a
Databento client, import the SDK's authenticated client path, read `.env`, authenticate, resolve real
symbols, open a socket, or contact a vendor.

## Pass and stop law

The current job can pass only as `JOB50_LOCAL_RUNNER_READY_ONLY`. That status means a later
owner-authorized metadata attempt is mechanically prepared; it is not the external census.

Local or later external execution stops fail-closed under the most specific applicable status:

- `STOP_CONTRACT_OR_DECLARATION_DRIFT`
- `STOP_SDK_VERSION_OR_SIGNATURE_DRIFT`
- `STOP_AUTHORIZATION_MISSING_OR_INVALID`
- `STOP_CREDENTIAL_ROTATION_UNRESOLVED`
- `STOP_QUOTED_CEILING_MISSING_OR_INVALID`
- `STOP_FORBIDDEN_METHOD_OR_SCOPE`
- `STOP_ATTEMPT_PATH_EXISTS`
- `STOP_CALL_JOURNAL_INVALID`
- `STOP_PARTIAL_OR_CRASHED_ATTEMPT`
- `STOP_LOCAL_READINESS_RECEIPT_INVALID`
- `STOP_VENDOR_AUTH_OR_ENTITLEMENT`
- `STOP_SCHEMA_UNAVAILABLE`
- `STOP_SYMBOL_OR_EXPIRY_MISMATCH`
- `STOP_ZERO_RECORDS`
- `STOP_NONFINITE_COST`
- `STOP_MISSING_SESSION_RESPONSE`
- `STOP_OVER_QUOTED_CEILING`
- `STOP_RESPONSE_OR_RECEIPT_INVALID`
- `JOB50_AUTHORITY_VIOLATION`

No stop is silently retried, downgraded to a smaller session set, or converted into a partial pass.

## Execution order

1. Preserve sealed `PROGRAM_CONTRACT_V1.json` through `PROGRAM_CONTRACT_V3.json` and the complete V1
   evidence quartet; seal the independent eleven-gap audit and corrective V4 overlay under the same
   local-build-only authority.
2. Apply only the mechanical V4 identity and V2 readiness-path projection to the runner, CLI, and
   receipt integration; construct no external client.
3. Re-run the unchanged independently audited 26-test source, then produce only the side-by-side V2
   JUnit, synthetic journal, synthetic response, and local receipt with exclusive/durable writes.
4. Independently recheck the post-seal mechanical delta and stop at `JOB50_LOCAL_RUNNER_READY_ONLY`.
5. Only after a later explicit owner authorization, credential-rotation attestation, and numeric ceiling
   may a new external attempt directory be created.
6. After an external metadata receipt, stop again. Acquisition, download, and every outcome-bearing job
   remain separate owner decisions.
