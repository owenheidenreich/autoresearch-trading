# Protocol101 Goal Prompt: S1 Regimen Repair Machinery Implementation

Paste the text below into a fresh Codex Goal task.

---

GOAL ID:

`S1-REGIMEN-REPAIR-MACHINERY-IMPLEMENTATION`

OBJECTIVE:

Implement the owner-signed Protocol101 Stage-1 two-clock realized-exit,
identity-protection, simulator-v5, immutable-artifact, quote-age-reporting, and
alpha-firewall machinery. Exercise it with focused, synthetic, regression, and
non-economic full-corpus checks. Produce an immutable implementation packet
that is ready for a separate independent machinery-acceptance Goal.

Stop before economic replay, model fitting or refitting, strategy scoring,
equivalence branching, gate aggregation, ranking, selection, seed 45, or
protected evidence.

The producing task must not accept or certify its own machinery.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

OWNER AUTHORIZATION:

The binding amendment is:

```text
v4/docs/protocol101/training/contracts/
PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md
```

The signed machine-readable owner packet is:

```text
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_design_correction_attempt002/
owner_decision_packet.json
```

Require its SHA-256 to equal:

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

If these checks fail, repair only a mechanical path/checksum issue when the
signed content remains unambiguous. Never infer or rewrite owner intent.

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/report.md`
- every JSON/Markdown/schema/test-matrix artifact under
  `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_correction_attempt002/`
- the seven preserved attempt001 artifacts referenced by
  `owner_decision_packet.json`
- current dataset, governed-loader, training, hashing, replay, and simulator
  source and tests that consume Stage-1 processed rows

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_machinery_attempt001/
```

PREIMPLEMENTATION FREEZE:

Before editing governed code:

1. Verify the signed owner-packet hash and selected hard multiplicity control.
2. Verify all attempt001 and attempt002 checksum manifests that predate owner
   signature. Record the expected owner-signature mutation separately.
3. Record the repository status without reverting unrelated user changes.
4. Freeze a source/input inventory with SHA-256 hashes.
5. Preregister the exact implementation files, test IDs, output schema,
   forbidden actions, repair budget, and terminal routes.
6. Write `progress.json` before implementation starts.

IMPLEMENTATION CONTRACT:

## A. Two-clock realized-exit labels

Implement an additive processed-row successor. Preserve the old schema for
historical audit and never silently reinterpret a legacy row.

For every canonical `[strike, right, policy]` cell, persist:

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

Use the frozen pricing-event law:

1. Consider same-contract, same-session source quotes strictly after entry and
   at or before the policy deadline.
2. Sort deterministically by source timestamp. Duplicate path quote identities
   fail closed.
3. The first normalized no-bid, stop-loss, or take-profit event wins.
4. Otherwise select the latest causal quote at or before the deadline.
5. No exact-deadline quote is required.
6. No future quote means an explicit invalid cell.

Clock law:

- stop-loss/take-profit/no-bid:
  `source_quote_time == realized_exit_time`
- max-hold/forced-flat:
  `realized_exit_time == policy_deadline` and
  `source_quote_time <= realized_exit_time`
- PnL is priced from the source quote.
- Serial occupancy is released at realized exit time.
- Quote age is `(realized-source)/1_000_000`, nonnegative, audit-only, and
  never a row, candidate, or trade gate.

Preserve existing `labels_net_pnl` and `labels_mid_pnl` bytes and NaN masks.
This Goal may verify equivalence but may not decide model reuse versus refit.

## B. Identity protections

Fail closed before fitting, scoring, thresholding, hashing, or replay on:

- duplicate governed session membership;
- one session appearing in multiple split/fold roles;
- duplicate decision identity;
- duplicate contract identity inside a decision;
- duplicate canonical slot/right identity;
- duplicate contract/source-time path quote identity; or
- unordered or misaligned policy axes.

Use typed errors and stable blocker codes. Positive-control tests must prove
that no downstream fit, score, hash, or replay function was called.

Do not silently deduplicate.

## C. Simulator v5

Implement:

```text
protocol101_serial_simulator_v5_label_realized_exit_
account_continuity_fee_reserve
```

Keep simulator v4 readable for historical audit but prohibit v4 artifacts from
entering repaired evidence.

Simulator v5 must:

- preflight the complete candidate stream before producing economic output;
- keep one account, one contract, and one open position;
- realize pending PnL only when
  `label_realized_exit_time_ns <= next decision_time_ns`;
- realize an exit before evaluating a new decision at the same timestamp;
- keep deadline exits occupied after their pricing quote until their deadline;
- update cash, daily-stop state, drawdown, and frequency at occupancy exit;
- preserve continuous cash across sessions, session-starting equity, the
  signed 5% daily stop, entry cutoff, forced-flat cap, affordability, and the
  $3 fee reserve;
- apply fees exactly once;
- prevent stress overlays from changing cash or occupancy; and
- never synthesize max-hold timing when repaired metadata is absent.

## D. Immutable artifacts and resume

Implement the signed v2 artifact schema and canonical hashes for:

- candidate stream identities;
- candidate payloads including both clocks, exit reason, bid, deadline, quote
  age, PnL, and feature/source hashes;
- trade identities;
- simulator source/configuration;
- quote-age reports; and
- manifests.

Write payloads atomically and the manifest last.

- Complete matching packet: verify and skip.
- Partial packet: move to `void_outputs/` with a receipt, then restart.
- Hash mismatch: fail closed and never overwrite.
- Mixed simulator versions: fail closed.

## E. Quote-age audit

Implement `exit_quote_age_report.json` by policy, session, and realized-exit
time-of-day with:

```text
valid_count
invalid_count
zero_age_share
minimum_ms
mean_ms
p50_ms
p90_ms
p95_ms
p99_ms
maximum_ms
```

No threshold may be introduced and no eligibility code may read quote age.

## F. Alpha firewall

The model-facing matrix remains exactly the signed 17 features.

Explicitly reject exit times, source exit quote time, quote age, exit reason,
executable exit bid, deadline, invalid reason, PnL labels, and future/path
fields from model inputs. Wildcard or automatic numeric-column discovery is
forbidden.

REQUIRED TESTING:

Implement and pass every row whose `acceptance_stage` is `machinery` in:

```text
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_design_correction_attempt002/
repair_test_matrix_v2.csv
```

There are 35 machinery-stage specifications. Preserve the five equivalence,
ten campaign, and two independent-acceptance rows for their later Goals.

Required checks include:

- reference versus vectorized two-clock label behavior;
- bit-identical frozen net/mid labels and NaN masks, reported only as
  non-economic machinery validation;
- stop, target, no-bid, max-hold, forced-flat, no-exact-deadline, missing-path,
  same-timestamp, and cross-session cases;
- quote-age formula/report/no-gate controls;
- simulator overlap, cash, daily-stop, continuity, drawdown, end-of-stream,
  fee, stress, and legacy-artifact cases;
- every duplicate/role-overlap positive control;
- exact 17-feature negative-control injection;
- deterministic artifact hash and resume behavior; and
- boundary attestations.

Run compilation and the focused affected test suites with:

```text
~/.autoresearch-trading/runtime-venv/bin/python
```

Disposable smoke outputs must live under the attempt output directory or a
temporary directory and may not become campaign evidence.

INDEPENDENCE BOUNDARY:

This Goal is the machinery producer.

It may state only:

```text
repair_machinery_complete_pending_independent_acceptance
```

It may not state that the machinery is accepted, valid evidence, training
ready, or eligible for economic rebuild. It may not create the independent
acceptance packet or import a future independent oracle.

BLOCKER-RECOVERY LOOP:

Continue through mechanical implementation, compilation, schema, hashing,
resume, and focused-test blockers for up to three bounded attempts per
component.

For each failed attempt:

1. Preserve the failure under `void_outputs/`.
2. Record root cause and affected contract/test IDs.
3. Repair only the same signed requirement.
4. Add or strengthen a focused regression test.
5. Rerun the affected checks and then the full machinery suite.

Do not stop merely because the first implementation attempt fails.

Stop for owner input only when proceeding would require changing a signed
feature, label, policy, fill, fee, account, fold, seed, gate, multiplicity,
holdout, or protected-data rule.

REQUIRED OUTPUT:

```text
preregistration.json
source_inventory.json
implementation_manifest.json
changed_files.json
test_matrix_results.csv
test_results.json
non_economic_label_validation.json
alpha_firewall_validation.json
identity_positive_controls.json
simulator_v5_validation.json
quote_age_report_validation.json
artifact_resume_validation.json
boundary_attestation.json
progress.json
summary.json
report.md
hashes.sha256
```

The manifest must identify every source/test hash and every implemented
contract/test-matrix ID.

FORBIDDEN WORK:

- No H0-H3 economic replay.
- No model fitting, refitting, scoring, prediction generation, recalibration,
  threshold selection, or equivalence reuse/refit decision.
- No null, heuristic, D1, D5, G1-G9, maxT result, ranking, or selection work.
- No seed 45.
- No protected holdout, sealed recorder, or confirmation evidence.
- No broker/API contact, live market-data capture, paper-submit, paid download,
  promotion/default edit, runtime flag edit, launchd edit, or real-money path.
- No changes to the signed feature, label-policy, fee, fill, account, fold,
  seed, gate, or multiplicity contracts.
- Do not modify or delete historical H0-H3 evidence.

TERMINAL ROUTES:

- `repair_machinery_complete_pending_independent_acceptance`
- `repair_machinery_blocked_signed_contract_conflict`
- `repair_machinery_blocked_missing_nonprotected_input`
- `repair_machinery_failed_after_bounded_repair_attempts`

PASS REQUIREMENTS:

- the signed owner packet and hard multiplicity choice verify;
- all two-clock fields and invariants are implemented;
- frozen PnL-label bytes remain unchanged in producer validation;
- simulator v5 uses occupancy exit time and source-priced PnL;
- identity failures occur before downstream work;
- exact 17-feature alpha firewall passes all injections;
- quote age is reported but cannot gate;
- artifact hashes and resume laws pass;
- all 35 machinery-stage test specifications pass;
- focused affected tests and compilation pass;
- output hashes verify;
- all forbidden side-effect flags are false; and
- no independent-acceptance or training-ready claim is made.

STOP BOUNDARY:

Stop after producing and mechanically validating the implementation packet.
Do not run the independent acceptance Goal, E01-E13 equivalence branch,
economic replay, training, gates, selection, G9, holdout, shadow, or paper
work.

HIGHEST ALLOWED CLAIM:

> Protocol101 Stage-1 repair machinery implementation is complete and pending
> separate independent acceptance.

FINAL RESPONSE:

Explain plainly:

1. Which production files changed.
2. How the two exit clocks now work.
3. How simulator v5 differs from v4.
4. Which identity and alpha-firewall protections were added.
5. Which of the 35 machinery tests passed or failed.
6. Whether frozen label bytes remained unchanged.
7. Whether any forbidden action occurred.
8. The exact terminal route.
9. The sole next Goal:
   `S1-REGIMEN-REPAIR-MACHINERY-INDEPENDENT-ACCEPTANCE`.
