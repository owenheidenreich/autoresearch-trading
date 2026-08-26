# Job 49 — human-policy and subminute-data foundation

**Registered job:** 49 in [`v5/STATUS.md`](../../STATUS.md), 2026-08-24

**Current state:** `LOCAL_OUTCOME_BLIND_FOUNDATION_ONLY`; the V2 tail-watermark repair must pass before
any real decision may be recorded

**Current program contract:** [`PROGRAM_CONTRACT_V2.json`](PROGRAM_CONTRACT_V2.json), a sealed semantic
overlay on preserved [`PROGRAM_CONTRACT_V1.json`](PROGRAM_CONTRACT_V1.json)

## Decision and claim boundary

Job 49 builds two local interfaces needed by a future SPXW 0DTE research program:

1. append-only instrumentation that can later record the owner's decisions, including genuine
   abstentions and unavailability; and
2. a deterministic, offline declaration and receipt path for an exact Databento `OPRA.PILLAR`
   `cmbp-1` catalogue preflight.

That is all this job may establish. A local pass means **the instruments are ready to be used after a
new owner decision**. It does not establish data availability, a free-data boundary, population event
prevalence, human predictability, a useful signal, executable fills, expected P&L, paper readiness, or
permission to trade. It spends no alpha because it reads no strategy outcome.

The current authority comes from the Job 49 paragraph and register row in
[`v5/STATUS.md`](../../STATUS.md). The forward reservation in
[`FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md`](../../governance/FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md)
continues to bind every session from 2026-08-06 onward.

## 2026-08-24 V2 tail-watermark repair

Independent valid-prefix attacks found one real limitation in the sealed V1 instrument: deleting any
number of trailing records leaves a syntactically valid hash chain. V1 disclosed that a bare chain could
not detect an unanchored tail rollback, and its receipt correctly pinned the frozen synthetic journal at
sequence 8 and head `c0326528…`; that final receipt did not provide a moving anchor while a future live
journal was still growing. Because the V1 contract is sealed and the repair changes gate logic and
deliverables, [`PROGRAM_CONTRACT_V2.json`](PROGRAM_CONTRACT_V2.json) supersedes V1 for human-journal
integrity only. Every authority, outcome firewall, risk term, CMBP term, and later-job gate is inherited
unchanged.

V2 requires a private `0600` sidecar at `<journal>.jsonl.watermark.json`. Initialization and every append
must persist the journal's `log_id`, terminal sequence, and terminal head there. The writer holds the
journal lock while it first proves journal/watermark equality, fsyncs the appended journal record, writes
and fsyncs a same-directory temporary watermark, atomically replaces the sidecar, fsyncs the directory,
and only then reports success. A crash between journal and watermark durability leaves a detectable
mismatch and is never repaired silently. Public append, status, verification, and training projection
require the sidecar by default. Optional caller expectations can also require an exact terminal head and
a minimum terminal sequence; they never weaken the sidecar check.

This is deliberately a bounded claim. A retained sidecar detects **journal-only** tail rollback and a
caller-supplied stale or regressed terminal expectation. An actor able to roll back or delete both the
journal and the same-machine mutable sidecar can still recreate an internally consistent older pair.
Every finalized real session therefore still needs an independently retained terminal receipt or anchor
outside that shared mutable fault domain. V2 is tail-rollback detection against a retained local
watermark, not absolute tamper-proofing, and it still authorizes no real capture.

## Authority for this job

### Permitted locally

- this work packet and an immutable, versioned program contract;
- native-v5 decision-instrumentation and metadata-preflight code;
- declarations made entirely from frozen local scope inputs;
- synthetic decision records and synthetic catalogue responses;
- already-owned parser fixtures, strictly without labels, returns, fills, or P&L;
- deterministic unit/integration tests and local immutable receipts; and
- read-only verification of the resulting files.

### Forbidden until a later owner gate

- any Databento, vendor, broker, network, live-feed, or account call;
- `get_cost`, `get_record_count`, symbology resolution, or any other metadata request;
- any time-series request, download, entitlement change, subscription, or spend;
- recording a real/live human decision or starting a prospective capture;
- opening or deriving a strategy outcome, fill, return, hit rate, P&L, or reserved-session economics;
- fitting a model, calibrating a score, choosing a threshold, ranking policies, or selecting a family;
- simulated/paper/live order submission, broker contact, or unattended execution; and
- claiming that the completed selected CMBP fixture represents an unbiased population.

An attempted forbidden action ends this job as `JOB49_AUTHORITY_VIOLATION`; it is recorded rather
than repaired or silently retried.

## Frozen v1 policy vocabulary

Job 49 freezes a recording vocabulary, not a strategy. It must not choose an entry time, signal,
threshold, contract ranker, target, or economic objective.

### Event and action vocabulary

- Event kind is exactly one of `MONITORING_ON`, `MONITORING_OFF`, `PROMPT`, `DECISION`, or
  `CORRECTION`.
- A decision action is exactly one of `WAIT`, `OPEN_CALL`, `OPEN_PUT`, `HOLD`, or `EXIT`.
- `MONITORING_OFF`, an unanswered `PROMPT`, or a missing event is never converted into `WAIT`.
- A later capture protocol freezes its prompt/risk-set schedule before the first real record. A
  `DECISION` may link to a prompt or be marked spontaneous; its origin is data, not inferred later.
- Short premium, spreads, rolls, scaling, simultaneous positions, and overnight positions are outside
  v1. Logging `EXIT` does not authorize an exit model. Learned exit timing remains parked under the
  signed Tier-A law.

Every later real event must carry a unique event ID, session, occurred-at UTC wall clock, writer-generated
append UTC wall clock, local monotonic clock, append sequence, previous-record hash, event kind, and fixed
event fields. A `DECISION` must be appended within 30 seconds of occurrence and also carries a unique
decision ID, action, explicit prompted/spontaneous origin, position state before and after, information-
source inventory, market-state hash, eligible-universe hash, program-contract hash, and risk-contract
hash. An open records the exact raw OSI, order intent, size, debit estimate, and declared stop; an exit
records the exact held OSI and owner intent. Legal transitions are only flat `WAIT`/`OPEN_*`, open
`HOLD`/`EXIT`, and no monitoring shutdown while a position is open.

The production library and CLI own both append clocks; caller-supplied append or monotonic clocks are
available only through private deterministic test hooks. Information sources, reason codes, owner intent,
and correction annotations use closed enumerations, and free text is refused. A verified training reader
projects only the declared structured `DECISION` fields; correction annotations, append/audit clocks, and
hash-chain fields are not exposed to a future learner. Caller-supplied event, decision, and prompt IDs
remain audit/linkage metadata in the immutable journal but are also excluded from that learner-facing
projection, so identifier strings cannot become an outcome-text feature channel.

A record is sealed on append. `CORRECTION` appends an annotation link to an earlier event and never
rewrites the contemporaneous action or reconstructed state. The chain detects mutation, reordering,
interior deletion, duplicate IDs, and clock regression. V2 also refuses a syntactically valid trailing
rollback when the journal is checked against its retained per-append watermark or caller-supplied
terminal expectation. Every finalized session still binds its terminal anchor in an independently
retained receipt because coordinated rollback of the journal and local sidecar is outside this repair's
claim. The schema contains no realized outcome, future path, fill result, return, hit, or P&L field.

### Binding v1 risk law

| Control | Frozen meaning |
|---|---|
| Product | Same-session-expiry SPXW, long single-leg premium only |
| Ticket | One contract; entry premium times 100 plus fees must be at most **$2,500** |
| Concurrency | At most one open position |
| Frequency | At most two executed entry tickets per session; rejects and unfilled intents do not create extra permission |
| Breaker | **20% of session-starting equity**; once blocked, no new entry is legal |
| Stop | Every eventual executable policy must declare and enforce a stop at **−40% or wider** (more negative), with next-bid gap/slippage retained |
| Account | Starts at **$10,000**, walks sessions serially, compounds, and never resets daily |
| Survival | **50% of initial capital**: $5,000 for the signed $10,000 account; the serial path blocks at or below it |
| Other signed controls | No martingale, no overnight hold, the independent `moneyness_band`/deep-ITM bar, and the Tier-0 SPY floor remain binding; win rate is diagnostic, never the objective |
| Review/reversion | Review at $25,000 equity. A breached declared maximum loss reverts the cap to $1,000; the first powered negative evaluation also reverts it, while `UNDERPOWERED` does not. Any change requires the signed amendment route |

These values are frozen risk metadata on recorded intent. The Job 49 logger does not simulate ticket
counts, serial equity, breaker firing, stops, survival, fills, or economics.

## Evidence gate H — local human-decision instrumentation

### V1 deliverables preserved as historical evidence

- `v5/research/human_decision_log.py`: immutable event schema, exact action/event enums, frozen risk
  metadata, canonical record hashing, append-only log verification, and correction-link semantics;
- `v5/ops/record_human_decision.py`: local-only initialize/append/status/verify CLI with no market,
  vendor, broker, or order dependency;
- `v5/tests/test_human_decision_log.py`: the frozen refusal/pass suite below; and
- human-gate section in
  `v5/work/human-policy-foundation/LOCAL_FOUNDATION_RECEIPT_V1.json`, produced only from synthetic
  records and bound to the program-contract, implementation, and terminal journal-anchor hashes.

### V2 repair deliverables

- sealed `v5/work/human-policy-foundation/PROGRAM_CONTRACT_V2.json`, which binds V1 rather than editing
  it;
- the same three implementation/CLI/test paths, updated to initialize new journals under
  `v5.human-decision-log.v2` and enforce the V2 watermark law while preserving the V1 journal bytes;
- `v5/work/human-policy-foundation/SYNTHETIC_HUMAN_JOURNAL_V2.jsonl` and its derived
  `.jsonl.watermark.json` sidecar;
- `v5/work/human-policy-foundation/TEST_RESULTS_V2.xml`; and
- `v5/work/human-policy-foundation/LOCAL_FOUNDATION_RECEIPT_V2.json`, with V1 artifacts preserved and
  the exact journal/watermark terminal equality bound.

### Required tests

1. `test_event_kind_vocabulary_is_exact`
2. `test_action_vocabulary_is_exact`
3. `test_monitoring_off_unanswered_prompt_and_missing_are_not_wait`
4. `test_decision_origin_and_prompt_link_are_explicit`
5. `test_open_requires_exact_same_day_spxw_osi_and_owner_intent`
6. `test_risk_metadata_matches_frozen_contract_without_simulation`
7. `test_append_chain_rejects_mutation_deletion_duplicate_and_backdated_sequence`
8. `test_correction_is_a_new_linked_event_not_an_overwrite`
9. `test_outcome_fill_return_and_pnl_fields_are_refused`
10. `test_synthetic_fixture_round_trips_and_verifies_offline`
11. `test_logger_has_no_network_broker_or_market_data_dependency`

V2 inherits every test above and additionally freezes:

12. `test_tail_truncation_is_refused_against_a_recorded_watermark`
13. `test_watermark_is_required_and_advances_on_every_successful_append`
14. `test_watermark_tamper_and_regression_fail_closed`
15. `test_writer_refuses_to_extend_or_launder_a_truncated_prefix`
16. `test_expected_terminal_assertions_are_additional_external_anchors`
17. `test_expected_terminal_anchor_rejects_coordinated_prefix_rollback`

### Pass and stops

The repaired gate passes only as `HUMAN_INSTRUMENTATION_READY_LOCAL_V2` when every inherited and added
test passes and the V2 synthetic journal, watermark, and receipt verify. Otherwise it stops as one of:

- `STOP_HUMAN_SCHEMA_INVALID`
- `STOP_HUMAN_EVENT_OR_RISK_METADATA_INVALID`
- `STOP_APPEND_ONLY_INTEGRITY_INVALID`
- `STOP_WATERMARK_MISSING_OR_INVALID`
- `STOP_WATERMARK_JOURNAL_DIVERGENCE`
- `STOP_OUTCOME_BOUNDARY_INVALID`
- `STOP_LOCAL_RECEIPT_INVALID`

No pass status authorizes collection of a real human decision.

## Evidence gate C — offline CMBP catalogue readiness

### Frozen local scope

The local builder freezes the 1,014 session files at
`/Volumes/AR_TRADING_DATA/lifecycle_corpus_spx_tape_2022-06-01_2026-07-31/ladder` and each file's
existing ±25-point same-day-expiry contract band. It reads only each filename and `raw_symbol` column,
converts every contract to a unique, sorted, 21-character raw SPXW OSI symbol, and validates root,
expiry, right, strike, session, the declared XNYS calendar rule, and half-open RTH bounds. The exact
scope is sealed as `CMBP_SCOPE_MANIFEST_V1.json`; its declaration is
`CMBP_CATALOGUE_DECLARATION_V1.json`. It must never widen to parent `SPXW.OPT` scope.

Four method-specific future request shapes are frozen: one dataset-level
`metadata.get_dataset_range`; per-session `symbology.resolve` over the exact raw symbols and RTH date;
and per-session `metadata.get_cost` and `metadata.get_record_count` for dataset `OPRA.PILLAR`, schema
`cmbp-1`, `stype_in="raw_symbol"`, the exact symbol set, and half-open RTH bounds. Dates before the
declared 2023-03-28 schema boundary carry local disposition `EXCLUDED_KNOWN_PRE_COVERAGE`; they are
never silently replaced by `cbbo-1m`. `STOP_SCHEMA_UNAVAILABLE` is reserved for a later authorized
vendor response. Time-series retrieval and download are never allowlisted by this contract.

The completed selected-symbol semantic work—64 sessions, 248 session-symbols—is reused as evidence that
the existing parser fixture can express strict prior-touch semantics. Job 49 does not rerun it, and its
selected composition is never used as population or economic evidence. Broad-band prevalence, the exact
zero-price boundary, exact eligible-session count, live entitlement, and total cost all remain
`UNKNOWN` until a separately authorized metadata receipt exists.

### Exact planned deliverables

- `v5/research/cmbp_catalogue_preflight.py`: offline scope/request/response-artifact schemas,
  deterministic validation, and disposition law, with no vendor-client construction or call path;
- `v5/ops/prepare_cmbp_catalogue_preflight.py`: offline-only declaration and request-manifest writer;
- `v5/tests/test_cmbp_catalogue_preflight.py`: the frozen refusal/pass suite below; and
- `v5/work/human-policy-foundation/CMBP_SCOPE_MANIFEST_V1.json` plus
  `CMBP_CATALOGUE_DECLARATION_V1.json`: materialized outcome-blind scope and request descriptions; and
- catalogue-gate section in
  `v5/work/human-policy-foundation/LOCAL_FOUNDATION_RECEIPT_V1.json`, proving only local readiness
  and binding contract/scope/code hashes.

### Required tests

1. `test_scope_uses_exact_raw_osi_band_and_never_parent_symbol`
2. `test_root_expiry_right_strike_uniqueness_and_sorted_order_are_required`
3. `test_request_is_one_session_rth_half_open_and_cmbp_only`
4. `test_only_dataset_range_symbology_cost_and_count_are_allowlisted`
5. `test_precoverage_date_stops_without_schema_fallback`
6. `test_zero_cost_requires_positive_record_count`
7. `test_nonfinite_cost_missing_count_and_session_drop_fail_closed`
8. `test_actual_per_session_values_are_summed_not_extrapolated`
9. `test_request_manifest_is_deterministic_and_hash_bound`
10. `test_job49_validator_accepts_offline_artifacts_and_has_no_vendor_client`
11. `test_selected_64_session_fixture_is_marked_parser_only`
12. `test_no_availability_cost_free_boundary_or_population_claim_is_emitted`

### Pass and stops

The gate passes only as `CMBP_CATALOGUE_READY_LOCAL`. It may stop as:

- `STOP_LOCAL_SCOPE_INVALID`
- `STOP_LOCAL_SYMBOL_OR_EXPIRY_INVALID`
- `STOP_LOCAL_REQUEST_LAW_INVALID`
- `STOP_LOCAL_OFFLINE_BOUNDARY_INVALID`
- `STOP_LOCAL_RECEIPT_INVALID`

The later external receipt, not Job 49, owns the vendor dispositions
`STOP_MISSING_KEY_OR_ENTITLEMENT`, `STOP_SCHEMA_UNAVAILABLE`, `STOP_ZERO_RECORDS`,
`STOP_NONFINITE_COST`, `STOP_OVER_HARD_CAP`, and `PREFLIGHT_PASS_ONLY`.

## Job 49 execution stages

1. **Contract freeze.** Add this plan and `PROGRAM_CONTRACT_V1.json`; validate JSON and link integrity.
   V1 may be repaired until its self-hash is sealed; any later semantic change creates V2.
1a. **Integrity repair.** Preserve every named V1 artifact, seal `PROGRAM_CONTRACT_V2.json` as a semantic
   overlay, and build only the local per-append watermark repair and its refusal tests.
2. **Human instrument build.** Implement gate H without reading a market or outcome file. Run only its
   synthetic tests.
3. **Catalogue readiness build.** Implement gate C as a validator of frozen offline response artifacts.
   Create no API client, fake or live, and make no external call.
4. **Local integration rehearsal.** Reuse the outcome-blind 1,014-session OSI scope without changing its
   V1 artifacts, then produce a new V2 receipt whose two gate sections bind both program contracts, the
   current implementations, V2 synthetic journal and watermark, completed parser-fixture identity, and
   V2 test results.
5. **Close or stop.** Both effective gates passing yields `JOB49_LOCAL_FOUNDATION_PASS_ONLY_V2`,
   immediately followed by `BLOCKED_AWAITING_JOB50_OWNER_GATE`. Any failure records its named STOP and
   ends the repair without widening scope.

## Later jobs — names are sequencing, not authority

Registering or describing these jobs does not authorize them. Each requires a fresh owner decision, and
internal subphases with a more expansive authority require another decision even if the job number exists.

| Job | Purpose | New authority required | Earliest valid completion claim |
|---:|---|---|---|
| **50** | Establish the exact historical/live CMBP twin: execute the metadata preflight, then only under an additional scope/spend gate acquire and certify the same-schema substrate | Vendor metadata authority first; any download/spend and live capture are separate owner gates. Separately authorized prospective human capture may begin in parallel only after its capture role and independent terminal-anchor retention protocol are frozen | `HISTORICAL_LIVE_TWIN_READY_ONLY`; no fit or economics |
| **51** | Fit and evaluate the behavior model against prospective owner intent | Fresh registration and fit authority **plus an explicit G1/training-precondition exception or reopening**, frozen capture cohort, contiguous train/calibration/sealed-test roles, model family, proper score, nulls, and power law | `BEHAVIOR_MODEL_PASS_ONLY`; imitation is not profitability |
| **52** | Build and evaluate one value/economic candidate, with abstention/veto the first admissible mode | Separate outcome-opening authority, explicit current options-STOP and applicable `DO_NOT_RETEST` reopening, pre-reservation development population, minimum retained-coverage law, absolute-positive-expectancy bar, fixed fill/exit/risk laws, baselines, multiplicity, session-clustered power, and a declared G5 validation route | `CANDIDATE_ONLY`; no shadow or paper authority |
| **53** | Reproduce the sealed candidate in no-order live shadow | Live-feed/no-order authority, a G5-validated candidate, completed G6 runtime parity, same-schema runtime, causal clocks, parity/fail-closed law, and 20 consecutive eligible sessions | `SHADOW_PASS_ONLY`; operational evidence, not economic confirmation |
| **54** | Preregister and run a hierarchical guarded paper evaluation | One frozen candidate, design-specific known-answer power receipt, fresh cohort and reserve-opening protocol, broker/paper authority, identical adapter/risk law, and safety/futility rules | `PAPER_RESEARCH_PASS_ONLY`; no real-money authority |

No prospective capture may start until the V2 repair receipt passes and the later capture protocol names
where finalized terminal anchors are retained outside the journal-sidecar mutable fault domain. The local
sidecar is still updated every append; the independent anchor closes the stronger coordinated-rollback
threat at the cadence that later protocol freezes.

For Job 51, human imitation uses contiguous train/calibration/sealed-test roles and session-clustered
known-answer recovery of at least 80% with clean family-corrected nulls. Historical broker fills, if a
complete export exists, may estimate only the historical **executed-ticket mixture**: aggregate tickets
to session, keep per-ticket results diagnostic, and use drop-best only as leave-one-session-out
sensitivity. They cannot reconstruct `WAIT`, monitoring-off, unanswered prompts, owner-unavailable
counterfactuals, or zero-trade eligible sessions. Any teacher-economics opening is a separately registered,
pre-reservation, owner-authorized job; positive retrospective expectancy does not prove the current policy,
and negative expectancy closes only exact imitation rather than every veto/overrule policy.

The outcome-blind 2026-08-24 inventory found no machine-readable history of the owner's discretionary
orders/fills locally. Four isolated IBKR paper round trips are infrastructure fixtures, not a teacher
sample; uninspected year-end tax PDFs are not an order/fill chronology. A teacher-history audit is
therefore `BLOCKED_EXPORT_ABSENT` unless the owner supplies a structured account export containing orders,
executions, commissions, timestamps/time zone, option identity, quantities, partial fills/cancels, and
account/source provenance. Supplying an export still does not authorize opening its economics.

Job 52 uses a fixed exit and separate pre-reservation outcomes; no learned exit is smuggled into entry
work, and `WAIT_ALWAYS` cannot win through zero coverage. Job 53 needs 20 consecutive eligible shadow
sessions. Before Job 54 opens a fresh paper cohort, its exact candidate, endpoints, pairing geometry,
session list, stopping law, and design-specific known-answer power campaign must be frozen. There is no
pre-approved 60-session cohort. The hierarchical paper estimands are: (1) absolute bot net expectancy
above zero over all eligible sessions, primary; (2) paired bot-minus-owner superiority on co-observed
owner-available sessions, secondary; and (3) availability contribution reported separately, never by
treating the missing owner counterfactual as zero. Non-inferiority is optional only with an independently
justified harm margin and its own power receipt. No early stop for success is allowed.

## Whole-job verification

The repaired Job 49 passes locally only when:

- V1 remains byte-identical to the hash bound by V2, and `PROGRAM_CONTRACT_V2.json` parses as strict JSON
  and verifies its normalized semantic self-hash;
- the V2 human journal and watermark, reused catalogue declaration and scope, and V2 integration receipt
  each verify through their native validators;
- the exact inherited and added test cases and the integration refusal suite pass from
  `TEST_RESULTS_V2.xml`;
- `LOCAL_FOUNDATION_RECEIPT_V2.json` binds this plan, both program contracts, the preserved V1 receipt,
  source scope, current code/tests, selected parser-fixture identity, V2 journal and watermark terminal
  equality, and zero-external-action attestations;
- `check_project.py` and `git diff --check` pass; and
- the terminal state remains `BLOCKED_AWAITING_JOB50_OWNER_GATE`, with no vendor/API call, download,
  spend, outcome access, real capture, fit, live feed, broker contact, or order.
