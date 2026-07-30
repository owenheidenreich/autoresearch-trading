# Protocol101 Goal Prompt: S1 Regimen Repair Machinery Independent Acceptance

Paste the text below into a fresh Codex Goal task.

---

GOAL ID:

`S1-REGIMEN-REPAIR-MACHINERY-INDEPENDENT-ACCEPTANCE`

OBJECTIVE:

Independently determine whether the completed Protocol101 Stage-1 regimen
repair machinery implements the owner-signed two-clock realized-exit,
identity-protection, simulator-v5, immutable-artifact, quote-age-reporting, and
17-feature-firewall contract.

Use a fresh audit-local reference implementation and independently designed
fixtures. Do not import the producer's production core into the reference
oracle. Compare independent expected outputs with the production machinery,
re-run focused production tests, verify the producer packet and source hashes,
and issue one explicit pass or repair-required decision.

This Goal accepts or rejects machinery only. Stop before economic replay,
model fitting or refitting, model reuse decisions, strategy scoring, gate
aggregation, ranking, selection, seed 45, holdout access, or Full Trader
training.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/docs/protocol101/training/goals/PROTOCOL101_S1_REGIMEN_REPAIR_MACHINERY_GOAL_2026_07_26.md`
- every file under
  `v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/`
- every design, schema, and test-matrix artifact under
  `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_correction_attempt002/`
- the production and test files listed in the producer packet's
  `changed_files.json`

OWNER AUTHORIZATION:

Require the signed owner packet:

```text
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_design_correction_attempt002/
owner_decision_packet.json
```

Its SHA-256 must equal:

```text
cd69707b34bf67af94b76c36443ba34cf0fa2c84e619ff03c48a8305b1703817
```

Require:

```text
amendment_signed: true
packet_status: OWNER_SIGNED_AND_BINDING
multiplicity choice: HARD_ELIGIBILITY_CONTROL
owner: Owen Heidenreich
signed date: 2026-07-26
```

Do not infer, rewrite, expand, or weaken owner intent.

FROZEN PRODUCER INPUTS:

Require these exact SHA-256 values before evaluating machinery:

```text
82c82864e2d35e0aa86adce4a37aa046fd31cab413a4b30821b0641bc5eeb069
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/summary.json

81aff57a45057931ea8b6564b5bc181058ec69171b1544ed842e1e5896f6f786
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/report.md

fc96d786a4b7ed53f08f3342e6a0ace40e1c3e165c0a1fa7dad656644e716dba
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/hashes.sha256

a1dc482ca8d4f42a7bb30d62c0a9e90e8f1fa9da0231791325d794ef5b368b36
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/implementation_manifest.json

2559a72553adeb6119fdb1ea03c9aec37389be76fd80a20544b34552eb24888a
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/changed_files.json

c959d207fd34e2680ac46ae430e14a8c9247db844b522a1395e2d794861eb10c
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_correction_attempt002/repair_test_matrix_v2.csv
```

Verify every internal checksum in the producer packet before using any
producer claim.

If a frozen producer hash differs, do not update the expected hash. Determine
whether the input was legitimately regenerated under a new owner-authorized
attempt. Without that evidence, terminate
`repair_machinery_independent_repair_required` with
`frozen_producer_input_changed`.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/
```

The output directory is the only location where this Goal may create verifier
code, fixtures, temporary data, receipts, or acceptance artifacts.

INDEPENDENCE CONTRACT:

This task is not a repair task and is not the machinery producer.

The fresh reference oracle:

- must live under the acceptance output directory;
- may use the Python standard library and general-purpose numerical/dataframe
  libraries;
- may read signed contracts, raw fixture inputs, governed non-holdout rows,
  and producer manifests;
- must not import or copy executable logic from:
  - `v4.model.protocol101_regimen_repair`;
  - `v4.model.protocol101_serial_simulator_v5`;
  - `v4.model.protocol101_repair_artifacts`;
  - `v4.scripts.run_protocol101_regimen_repair_machinery_validation`;
  - producer test helpers that encode expected outputs; or
  - another producer-generated oracle;
- must compute expected label, simulator, identity, firewall, quote-age, and
  artifact results from the signed prose and machine-readable contract;
- may invoke the production public interfaces separately to obtain actual
  results for comparison; and
- must record its source and SHA-256 in `independent_oracle_manifest.json`.

Run an import audit over the independent verifier. Any forbidden production
core import makes `INDEP-CROSS-001` fail.

Do not modify production code or producer tests. If production is wrong,
record a repair-required decision with a minimal reproducer.

PREREGISTRATION:

Before executing production machinery or reading producer result values beyond
hash/inventory checks, write and hash:

```text
preregistration.json
source_inventory.json
independent_oracle_manifest.json
progress.json
```

Preregister:

- every acceptance dimension and contract assertion;
- the fresh fixture inventory;
- full-corpus non-economic validation method;
- production entry points that will be invoked;
- expected output schemas;
- import-independence checks;
- exact pass/fail aggregation;
- bounded verifier-only repair budget;
- forbidden actions; and
- terminal routes.

The pass rule is all-or-nothing. Every required acceptance check must pass.

REQUIRED ACCEPTANCE ROWS:

Execute the two rows whose `acceptance_stage` is `acceptance` in:

```text
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_design_correction_attempt002/
repair_test_matrix_v2.csv
```

They are:

```text
INDEP-CROSS-001
  fresh cross-implementation oracle
  exact outputs/hashes without production-core import

BOUNDARY-NO-SEALED-001
  zero seed45/holdout/sealed/broker/paper access through acceptance
```

Both are hard requirements.

ACCEPTANCE BATTERY:

## A. Contract and source integrity

1. Verify the owner packet, signed amendment, producer packet, source hashes,
   test hashes, manifests, and checksum files.
2. Verify the production source hashes equal `changed_files.json`.
3. Confirm the producer output is immutable and contains no mixed simulator
   versions.
4. Confirm all producer side-effect flags are false.
5. Confirm no source or test changed during this acceptance run.

## B. Independent two-clock label oracle

Create new fixtures from the signed contract, not from producer expected
values. Cover at least:

- first stop-loss event;
- first take-profit event;
- first normalized no-bid event;
- max-hold using latest causal quote before deadline;
- forced-flat using latest causal quote before deadline;
- no exact-deadline quote;
- missing future quote;
- duplicate source-time quote identity;
- two events at the same timestamp with deterministic ordering;
- quote after deadline;
- cross-contract and cross-session quote contamination; and
- invalid policy-axis alignment.

For each valid cell independently compute:

```text
labels_net_pnl
labels_mid_pnl
label_realized_exit_time_ns
label_source_exit_quote_time_ns
label_exit_quote_age_ms
label_exit_reason_code
label_executable_exit_bid
label_policy_deadline_ns
label_policy_index
label_invalid_reason_code
```

Require exact equality for integers, identities, reasons, masks, and canonical
hashes. Require bit-identical or explicitly contract-toleranced equality for
floating values. Any tolerance must be preregistered before actual production
values are read.

## C. Independent simulator-v5 oracle

Implement a small event-driven reference simulator directly from the signed
contract. Cover at least:

- one account and one open contract;
- realized exit before a new decision at the same timestamp;
- occupancy release at threshold event time;
- deadline occupancy despite an earlier pricing quote;
- pending PnL realization before affordability;
- continuous cash across sessions;
- session-starting-equity daily stop;
- 5% daily stop behavior;
- $3 round-trip fee applied once;
- fee reserve and affordability;
- overlapping candidate rejection;
- end-of-stream realization;
- stress overlays that do not change occupancy or cash; and
- rejection of missing two-clock or simulator-v4 artifacts.

Compare the independent event ledger, trade identities, PnL, cash path,
drawdown path, frequency, skip reasons, and hashes with production outputs.

## D. Identity and firewall positive controls

Independently inject:

- duplicate governed session;
- session role/fold overlap;
- duplicate decision identity;
- duplicate contract inside a decision;
- duplicate canonical slot/right;
- duplicate contract/source-time path quote;
- unordered or misaligned policy axis;
- each new exit metadata field into model features;
- a future/path/label field under an alias; and
- automatic numeric-column discovery.

Require fail-closed typed blocker codes before fit, score, hash, or replay.
Use spies or subprocess sentinels independent of the producer tests to prove
that downstream work was not called.

Require the accepted model-facing matrix to contain exactly the signed 17
features.

## E. Quote-age and artifact behavior

Independently verify:

- quote age equals
  `(realized_exit_time_ns - source_exit_quote_time_ns) / 1_000_000`;
- quote age is nonnegative;
- reports contain the signed count and percentile fields by policy, session,
  and realized-exit time of day;
- no eligibility, row, candidate, trade, or feature code reads quote age;
- deterministic candidate/trade/config/source hashes;
- payload-first, manifest-last atomic behavior;
- matching complete packet verifies and skips;
- partial packet moves to `void_outputs/` with a receipt;
- hash mismatch fails without overwrite; and
- mixed simulator versions fail.

## F. Independent full-corpus non-economic validation

On the governed 271-session non-holdout corpus:

1. Independently stream or chunk the reference calculation so memory limits
   cannot change the result.
2. Recompute frozen net/mid label bytes and NaN masks without importing the
   producer label core.
3. Independently validate the two-clock invariants, policy axes, identity
   uniqueness, quote-age formula, and invalid-cell reasons.
4. Require 28,602,966 policy cells unless a signed governed-registry change
   predating this Goal is documented. Do not silently update the count.
5. Compare canonical per-session and aggregate hashes with production.

This remains non-economic machinery validation. Do not create trades, scores,
gates, rankings, or a model-reuse/refit decision.

## G. Production regression rerun

Using:

```text
~/.autoresearch-trading/runtime-venv/bin/python
```

Run:

- compilation for all producer-changed Python files;
- the focused regimen-repair label, identity, simulator-v5, artifact,
  governed-loader, alpha-firewall, and dataset tests;
- the preserved simulator-v4 historical regression tests; and
- the producer validation command in a disposable output directory only.

Producer tests support acceptance but cannot replace the independent oracle.

PASS/FAIL AGGREGATION:

Pass only if all of the following are true:

- owner and producer frozen hashes pass;
- every producer internal checksum passes;
- `INDEP-CROSS-001` passes;
- `BOUNDARY-NO-SEALED-001` passes;
- every independent label fixture passes;
- every independent simulator fixture passes;
- every identity and firewall positive control passes;
- quote-age and immutable-artifact controls pass;
- full-corpus non-economic validation passes;
- focused production tests and compilation pass;
- no production or producer-test file changed;
- no forbidden action or protected evidence access occurred; and
- acceptance artifacts are complete and internally hashed.

One failed required check means the machinery is not accepted. Do not average,
waive, reinterpret, or downgrade failures.

BLOCKER-RECOVERY LOOP:

Continue through mechanical verifier, fixture, path, memory, serialization, and
reporting blockers for up to three bounded attempts per acceptance component.

For a verifier-only failure:

1. Preserve the failed attempt under `void_outputs/`.
2. Record the root cause.
3. Repair only audit-local verifier code or fixtures.
4. Rerun the affected check and then the full acceptance battery.

Do not stop merely because the first independent verifier attempt fails.

If a production defect is reproduced, do not repair production in this Goal.
Write:

- the smallest failing fixture;
- expected versus actual values;
- violated signed clause and test ID;
- affected files and hashes;
- blast-radius assessment; and
- the exact bounded repair required.

Then terminate `repair_machinery_independent_repair_required`.

Stop for owner input only when the signed contract is genuinely contradictory
or required nonprotected input is unavailable after three documented attempts.

REQUIRED OUTPUT:

```text
preregistration.json
source_inventory.json
independent_oracle_manifest.json
acceptance_matrix_results.csv
contract_and_source_integrity.json
two_clock_cross_implementation.json
simulator_v5_cross_implementation.json
identity_positive_controls.json
alpha_firewall_independent_validation.json
quote_age_independent_validation.json
artifact_resume_independent_validation.json
full_corpus_non_economic_validation.json
production_regression_results.json
minimal_reproducers.json
independence_attestation.json
boundary_attestation.json
acceptance_decision.json
progress.json
summary.json
report.md
hashes.sha256
```

`minimal_reproducers.json` may be an empty list on pass.

FORBIDDEN WORK:

- No modification of production source or producer tests.
- No H0-H3 economic replay.
- No model fitting, refitting, scoring, prediction generation, calibration,
  threshold selection, or reuse/refit decision.
- No null, heuristic, D1, D5, G1-G9, maxT result, ranking, or selection work.
- No seed 45.
- No protected holdout, sealed recorder, confirmation, live-shadow, or paper
  evidence access.
- No broker/API contact, live market-data capture, paper-submit, paid download,
  promotion/default edit, runtime flag edit, launchd edit, or real-money path.
- No changes to signed feature, label-policy, fee, fill, account, fold, seed,
  gate, holdout, or multiplicity contracts.
- Do not modify or delete producer evidence or historical H0-H3 evidence.
- Do not start the fresh Full Trader entry campaign.

TERMINAL ROUTES:

- `repair_machinery_independently_accepted`
- `repair_machinery_independent_repair_required`
- `repair_machinery_independent_blocked_signed_contract_conflict`
- `repair_machinery_independent_blocked_missing_nonprotected_input`

HIGHEST ALLOWED PASS CLAIM:

```text
Protocol101 Stage-1 repair machinery is independently accepted for the next
owner-authorized preregistration phase.
```

This is not a profitable-model, training-ready campaign, entry-candidate,
Full-Trader, holdout, shadow, paper-ready, or live-trading claim.

On pass, stop and return the completed acceptance packet to the owner. Do not
name, invent, or start another Goal until its standalone prompt exists.

FINAL RESPONSE:

Report:

1. terminal route;
2. pass/fail for both acceptance-stage test IDs;
3. independent oracle scope and import audit;
4. full-corpus counts and hash result;
5. focused production-test result;
6. any reproduced defect and minimal reproducer;
7. boundary/side-effect result;
8. exact output paths; and
9. the highest allowed claim.

Do not continue into another phase.
