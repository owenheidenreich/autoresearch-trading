# Protocol101 Goal Prompt: FT1E Fresh Entry Independent Audit And Selection

Run this Goal in a fresh Codex task or subagent that did not implement or
execute FT1D.

GOAL ID:

`FT1E-FRESH-ENTRY-CAMPAIGN-INDEPENDENT-AUDIT-AND-SELECTION`

## Objective

Independently audit the completed owner-authorized FT1D fresh Protocol101
Stage-1 entry campaign, freeze an independently reconstructed 28-row G1-G8
result, and execute only the model-free selection-routing step.

The producer terminal route is:

```text
fresh_entry_campaign_g1_g8_complete_pending_independent_audit
```

The producer scientific result is a valid no-selection campaign outcome:

```text
D1: false
D5: true
D6: true
maxT_valid: true
campaign valid: false
hard-gate eligible rows: 0
multiplicity-adjusted signal rows: 5
G8 role: report_only
G9 executed: false
```

Independently reproduce or reject that result. If the audit accepts it, route
selection under the frozen model-free law. A failed global control must not be
overridden by adjusted-signal rows. The expected selection result, if all
inputs and independent calculations agree, is:

```text
routing_decision: campaign_invalid_no_selection
selected_candidate: null
eligible_row_count: 0
```

This Goal authorizes audit and selection routing only. It does not authorize
new fitting, reference recomputation, maxT recomputation, producer aggregation
changes, seed 45/G9, protected holdout access, HOLD/EXIT work, broker or paper
activity, promotion, or runtime changes.

## Workspace And Interpreter

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
campaign namespace: protocol101_full_trader_stage1_entry_fresh_attempt001
campaign execution id: P101-FT1D-20260726-001
```

## Owner Authorization

Verify these exact files before acting:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_owner_execution_authorization_2026_07_26/
owner_execution_authorization.json
sha256:
1bb31f15d6e07e88cbff84665c0795c6f65271c14a6d91018a43f5a6737f65af

v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_owner_execution_authorization_2026_07_26/
signed_contract_bundle.json
sha256:
dc7f1ff7dc414fa4baa1c29f8841faf08a8c7b616c701b4934ca15e5eee64e54
```

The signed authorization explicitly includes later independent audit and
entry selection. Do not create, broaden, or rewrite authorization.

## Frozen FT1D Inputs

Verify all file hashes before reading campaign result values:

```text
e93f045e35110a759d8ddd6241640778d4beb8b60aef18df62da4bdf5183962e
v4/docs/protocol101/training/goals/
PROTOCOL101_FT1D_FRESH_ENTRY_CAMPAIGN_RUN_TO_G1_G8_GOAL_2026_07_26.md

a7daaf3afda82fc0b009bb8ad00cf51a67615890808442e2a26e163675515fa4
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
real_campaign_packet.json
payload packet_sha256:
99d58039c2689bdf92ec40e2dd441604ed24e2b08e981b6a2c078fefd66caad7

1b4ec89c9305910e94091b6faece00469a76cfb65731dbd05b7fefe6e6c9d08d
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
g1_g8_aggregation.json
payload aggregation_sha256:
9f6fb1133b25d8f176e73d1f3f62821b9af8dbb9ea89acc81ca8f26cd804ff64

02937681124b75bee1891bffa2116bbfe7c025a7024e112e5f7a64e4b9e6ebc6
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
g1_g8_validation.json

9031247b0b424a9e59534fa124725bdf9294238706a1f11cf16eb051d5608769
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
terminal_validation.json

2ee220033d5d39c82072c40a68b3492bd7f8659e9902be40be6a23238a4ef554
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_execution_attempt001/
controller_journal.jsonl

4c90ef3e81bb23f59e32cbe5de8f1b013040227d2b7762c8f521b757a0ad07f3
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
execution_provenance_authority.json

5c2add721f926c632b32b0028fc1f3afc8af857ab751ad6bae5c25110dbe6823
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
control_authority.json

bfcc965f31eead2788c25a8026f575707c5887b691dc2694da166fa25c966602
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
real_v5_references_d1_d5_d6.json

b465b2c9ca73a0bc2b62b3cce35d02d2d92327afa8d79d7cf37926e52f06c44c
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
maxT/maxT_control.json

df2ed69539ed0c09ba17860c6d6d748f286c09b6d3051da09727864d3aac2c4c
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
all_wait_unit_source_certification.json
```

Also verify the producer reference/control detail receipts:

```text
7984d6bc40db802dad70195dfb87a18dd15d67ee91d59ffcfae097068d2334ad D1/D1_detail.json
882a2f6a21b787f03c5b1d23a3e6dad13ef9c79943dd3b465a65866a29d9411e D5/D5_detail.json
53f7254424027b7fa806fd2bcc9a048be42f7ea14a6d64cdcf6d6bb2c90c0955 D6/D6_detail.json
064327dcf404f237bf924212bff417b54f001364244944ed701db3a61fe54b97 real_v5_references_validation.json
ca603a6df42549487b0a6ba1f32b73a3892f330d065f507669b4430a07ccef71 execution_provenance_validation.json
```

Paths in the last block are relative to the FT1D G1-G8 output directory.

## Frozen Source Inventory

Verify these exact source hashes before executing the audit:

```text
7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548 v4/model/protocol101_serial_simulator_v5.py
fc65d9ab835dad8d702aac67fff5e972644ea36be9e877c6cad597ab608f2536 v4/model/protocol101_stage1_gate_contract.py
204eb06a8b1332a68ab82f01390f9e14e8dda72ed3a0066c12e1653092137937 v4/model/protocol101_stage1_controller_journal.py
bb773e0169fa5348157b6b1a30fd62fbdd4651be3c0e056e4b8428126d31b9fc v4/model/protocol101_fresh_model_artifact.py
5211cb5f981d5be80dd735d3d358a3d38dd93e7fe48f3836e58a690d5ac4a081 v4/model/protocol101_fresh_unit_summary.py
3181577670f26317c16b8405caa8f2d2676a201003621fa9cfcc9708e1c372a4 v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
e08a954ca021ec56dc14e00f1d2ac76f66446b435344b1cdd5f0bbc10506c5d5 v4/scripts/run_protocol101_scoped_stage1_independent_audit.py
233b8b21e6225bab4a00fd0ae5d6008e065c6de8b57def1a8a73f812fb05f232 v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
63e28901bcbf351f536f20effe5e693537b0034c5c91857481899ca9c369b7aa v4/scripts/run_protocol101_ft1d_real_campaign_g1_g8.py
da6ee50600ad806db0d7163232701de2eaede53384ddf203c51a9529ccee86ef v4/scripts/run_protocol101_full_trader_stage1_entry_campaign.py
```

Any mismatch is a blocking input route. Do not silently modernize or repair
the frozen campaign in an independent audit task.

## Writable Scope

Primary output directory:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_independent_audit_selection_attempt001/
```

Other allowed writes are limited to:

- atomic progress updates in that output directory;
- append-only `INDEPENDENT_AUDIT` and
  `MODEL_FREE_SELECTION_ROUTING` records in the existing controller journal,
  only after their validator receipts pass; and
- the existing campaign execution `progress.json`, only to record current
  node, exact blockers, repairs, and terminal route.

Do not duplicate the 4.2 GiB campaign packet. Use its immutable path and hash
as a reference. Do not rewrite any fitted model, unit summary, replay packet,
campaign packet, reference, D1/D5/D6 artifact, maxT artifact, producer
aggregation, authority, or prior journal record.

## Preregistration

Before invoking producer audit/selection code or inspecting row-level values,
write and hash:

```text
preregistration.json
source_inventory.json
input_bindings.json
progress.json
independent_oracle_manifest.json
```

Freeze:

- all 420 expected H/P/seed/fold axes;
- exact 28 H/P rows and seed order 42,43,44;
- simulator-v5 two-clock and fee arithmetic;
- G1-G7 formulas and boundaries;
- G8 report-only treatment;
- D1/D5/D6/maxT control conjunction;
- all-WAIT evidence acceptance and rejection laws;
- compact/full summary hash semantics;
- comparison tolerances;
- audit agreement law;
- selection law and tie break;
- expected no-selection handling for a failed global control;
- journal nodes and routes;
- forbidden actions; and
- terminal routes.

## Independence Contract

Create audit-local verifier code under the output directory. It must not
import producer aggregation helpers, producer expected-result helpers,
producer tests, or FT1D repair helpers.

It may call the public simulator-v5 implementation as a system under test,
but must independently verify candidate PnL arithmetic, contract multiplier,
fee timing, source bid, entry ask, decision clock, realized-exit occupancy
clock, and serial account schedule.

JSON mapping order may be rehydrated at the audit read boundary only when:

- key sets and values are unchanged;
- stable payload hashes remain exact; and
- the audit records original and rehydrated key order.

This is an adapter operation, not evidence mutation.

The nine zero-selected-candidate units may be accepted only if the audit
independently verifies:

- a nonzero scored decision count and nonzero underlying candidate-frame
  count;
- zero ENTER actions, entry intents, trades, and skipped events;
- immutable empty replay streams;
- correct `sha256([])` candidate/trade hashes;
- zero primary economics; and
- exact binding to the source certification and packet unit hash.

An empty unit missing any one of those facts must fail closed.

For D1, independently enforce the signed contract:

- permuted fit/calibration target overrides are `NON_CANDIDATE`;
- they are never replayed as economic trades;
- simulator-v5 quote/PnL preflight remains unchanged;
- validation labels, exits, and candidates remain ordinary and unpermuted;
- no synthetic exit bid is allowed; and
- D1 threshold work makes no executable-quote or PnL claim.

## Required Audit

1. Verify all frozen file, payload, authority, journal, and source hashes.
2. Verify the journal is an exact valid prefix ending at
   `G1_G8_AGGREGATION`, ordinal 6, with next node `INDEPENDENT_AUDIT`.
3. Verify all 420 model hashes, all 420 unit hashes, all 420 immutable replay
   packets, exact axes, feature lists, folds, sessions, identities, and zero
   old-model overlap.
4. Independently validate the nine all-WAIT folds and reject uncertified
   empty evidence.
5. Independently reconstruct every fold and every seed through simulator-v5
   using ordinary unpermuted validation candidates.
6. Independently recompute all 28 row metrics and G1-G8 values.
7. Independently validate the 28 real references, D1, D5, D6, and exact
   20,000-replicate maxT control.
8. Prove G8 does not affect eligibility or ranking and G9 is false/not run.
9. Compare the full independent result to the producer aggregation. Require
   exact identities, booleans, hashes, and accepted numeric tolerances.
10. Publish all 28 rows even though the campaign global control is false.

Audit acceptance means the independent result agrees with the frozen
producer. It does not mean the campaign passed its scientific controls.

On agreement, publish route:

```text
fresh_28_row_independent_audit_accepted
```

Then write an audit validator receipt and atomically append the
`INDEPENDENT_AUDIT` journal checkpoint.

On disagreement, preserve the exact reproducer and stop. Do not execute
selection.

## Required Selection Routing

Selection may run only from the journal-bound accepted independent audit
freeze.

Use exactly:

```text
primary rank:
descending median-seed fee-adjusted continuous strict-serial net PnL

tie break:
H0,H1,H2,H3 then P0,P1,P2,P3,P4,P5,P6

maximum selected candidates:
1

rank exclusions:
G8/ECE, win rate, drawdown, feature count, model complexity,
risk-adjusted utility, and post-result preference
```

Before ranking, independently enforce all global controls and hard
eligibility. Since D1 is false in the frozen producer result, the accepted
route must contain no selected candidate. The five adjusted-signal rows are
attribution evidence only and cannot become eligible.

Require:

```text
routing_decision: campaign_invalid_no_selection
selected_candidate: null
eligible_row_count: 0
G9_executed: false
```

Treat the selector's nonzero process exit for an invalid campaign as the
expected scientific no-selection route only if its complete output is
well-formed, hash-valid, audit-bound, and independently reproduced.

Write a selection-routing validator receipt, append
`MODEL_FREE_SELECTION_ROUTING`, and verify the journal now ends at `STOP`.

## Mechanical Repair Law

Do not terminate at a mechanical blocker without recording its exact
exception and inputs in both progress files.

Allowed repairs are limited to audit-local:

- streaming or reference-based reads;
- compact/full summary hash interpretation;
- JSON key-order rehydration with unchanged values/hashes;
- strict certified-abstention handling;
- deterministic receipt construction; and
- controller append plumbing.

Add focused positive and negative regressions for every repair. Do not repair
features, labels, folds, policies, model family, thresholds, fees, fills,
noise, simulator economics, gates, reference values, maxT values, or campaign
results.

Any such scientific change requires the blocking owner-decision route.

## Required Outputs

```text
preregistration.json
source_inventory.json
input_bindings.json
independent_oracle.py
test_independent_oracle.py
independent_oracle_manifest.json
campaign_integrity.json
unit_replay_integrity.json
all_wait_independent_validation.json
D1_non_candidate_contract_validation.json
reference_control_validation.json
independent_aggregation.json
producer_agreement.json
independent_audit.json
independent_audit_validation.json
selection.json
selection_validation.json
controller_journal_validation.json
side_effect_audit.json
terminal_validation.json
progress.json
summary.json
report.md
hashes.sha256
```

Write `hashes.sha256` last and include every top-level output except itself.

## Required Regressions

Run:

- independent oracle tests;
- simulator-v5 tests;
- gate contract, independent audit, selector, and controller journal tests;
- compact/full mixed-summary regression;
- positive certified-abstention regression;
- negative missing-evidence abstention regression;
- D1 non-candidate positive and executable-replay negative regressions;
- fresh-process sorted-JSON order regressions; and
- final frozen input/source hash recheck.

Do not run the full repository suite.

## Side-Effect Audit

Require all false:

```text
model_training_executed
reference_or_control_recomputed
producer_aggregation_recomputed
seed_45_or_G9_executed
protected_holdout_read
recorder_or_sealed_evidence_read
broker_endpoint_called
paper_submit_allowed
paid_data_downloaded
promotion_or_default_changed
runtime_or_launchd_changed
real_money_path_changed
HOLD_EXIT_started
```

## Terminal Routes

Success:

```text
fresh_entry_campaign_independently_audited_selection_routed_no_candidate
```

Blocking:

```text
fresh_entry_campaign_independent_audit_input_blocked
fresh_entry_campaign_independent_audit_rejected
fresh_entry_campaign_selection_routing_invalid
fresh_entry_campaign_independent_audit_scientific_owner_decision_required
```

## Final Response

Report:

- audit independence and exact frozen-input verification;
- 420 model/unit/replay integrity;
- compact/full and all-WAIT audit results;
- independently reconstructed D1/D5/D6/maxT results;
- grouped G1-G8 and 28-row agreement summary;
- selection route and selected candidate;
- final journal head and STOP state;
- exact side-effect audit;
- terminal route; and
- the next separately written Goal path, if any.

Do not claim a model was selected. Do not start G9, protected holdout,
HOLD/EXIT, paper, promotion, or real-money work.
