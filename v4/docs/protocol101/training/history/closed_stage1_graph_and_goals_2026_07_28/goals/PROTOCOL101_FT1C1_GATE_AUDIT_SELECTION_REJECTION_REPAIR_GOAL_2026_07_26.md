# Protocol101 Goal Prompt: FT1C1 Gate, Audit, And Selection Rejection Repair

Run this Goal in a fresh Codex task or subagent.

---

GOAL ID:

`FT1C1-GATE-AUDIT-SELECTION-REJECTION-REPAIR`

OBJECTIVE:

Repair exactly the ten bounded defects reproduced by the first independent
FT1C acceptance attempt. Preserve all signed policy, gate thresholds, campaign
geometry, simulator-v5 semantics, multiplicity rules, selection objective, and
authorization boundaries.

This is a machinery repair only. Do not execute or inspect real campaign
economics, train models, select a real candidate, run seed 45/G9, open protected
or sealed evidence, or contact a broker.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- all signed Stage-1, G4, G8, and regimen-repair contracts
- `v4/docs/protocol101/training/goals/PROTOCOL101_FT1C_GATE_AUDIT_SELECTION_MACHINERY_REPAIR_GOAL_2026_07_26.md`
- `v4/docs/protocol101/training/goals/PROTOCOL101_FT1C_GATE_AUDIT_SELECTION_INDEPENDENT_ACCEPTANCE_GOAL_2026_07_26.md`
- all top-level artifacts and the `reproducers/` directory under:
  `v4/audit/autoresearch/protocol101_full_trader_stage1_gate_audit_selection_independent_acceptance_attempt001/`
- all FT1C producer-changed source and test files
- independently accepted runner-v5 and reference/multiplicity packets

REQUIRED INPUT ROUTE:

```text
gate_audit_selection_machinery_independent_rejection
```

FROZEN REJECTION EVIDENCE:

```text
7ddcb5bce0c9c2c947afaf6f3db509f68452060d9b76a2d5b56eef16ac8c9317 preregistration.json
202b3d705ccff065a2ebff19ffd168fc06a4d70d2fb1f0f1d8d773ea1adbf292 independent_oracle.py
cf7a179c259bead0b83c9fb05bbf8a4ecd3813e5211a0c9c8f3d420d347987da mutation_matrix.csv
c5f7240c9385ccb6266cb09a06787a68391fa122cb08b02335f284fee701b377 reproducers/reproducers.json
54abc30a33ac8d78b39295683c68551962105206436962d71840502a0f434f2c summary.json
b3a1d71afba30a042a8aebc5b7e14cddbecc8202f20e9e4199b605a3e151c874 report.md
71981bbfda2d3afb5d45964e035be2d95f3038c0fe1f2e21049a25b30bc29ffe hashes.sha256
```

Paths are relative to the independent rejection directory. Verify these before
editing and again before terminal reporting. The rejection packet is immutable.

FROZEN PRE-REPAIR SOURCE HASHES:

```text
9dc12adb949ccc1fdc12c05209945c655e525b04cc4caf09eb8f3221fa7253cf v4/model/protocol101_stage1_gate_contract.py
5479e22f3a728fe4d9c1f0ed68927672d610d7cf8ac20996d12d231426b3e561 v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
663ac627e936ee3625410b1b05990827eb39dd2b4258d3d6adbc49aff65af05e v4/scripts/run_protocol101_scoped_stage1_independent_audit.py
4c224d540f6079560762a2eb75db190892b1b7a84800c60485ebc159d05e611d v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
ea13ac4bb979ac46eec2975a3d10dc03162ad23cc8aea6c9266598da1b26369f v4/scripts/run_protocol101_stage1_autoresearch_graph.py
b60998516fc883226124cede5c929a9553ec0281b0c4286dd63ad42ff5a8cac0 v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
d74f415207d8efe1314cfd6fe6a4f2b13ba2d76e7d2ff11a6d127f9a5df76f1e v4/tests/test_protocol101_stage1_gate_contract.py
2c8f500b281f29c9cf2f2d208c84ff9dc46bb2a127595a1dec9e48bf97b2f4d1 v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py
d5fe20e8d8b47d263761a6a8edd125f5df0b0b7e9a38fd15c50df39151d3ea66 v4/tests/test_protocol101_scoped_stage1_independent_audit.py
0b3952fea20a51eb7fc669dde7691a12b8f537eae3e19fe03baea50feaf6c031 v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
89dd6c6e63262e346b2f5519771eec3232facbf5e8d3a0c9671d0555d85c7a1a v4/tests/test_protocol101_stage1_autoresearch_graph.py
aab9bd40c45a7dfd8bbdbd86f12de492433e876b8724434111ecf6badc64c487 v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

AUTHORIZED PRODUCTION FILES:

```text
v4/model/protocol101_stage1_gate_contract.py
v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py
v4/scripts/run_protocol101_scoped_stage1_independent_audit.py
v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py
v4/scripts/run_protocol101_stage1_autoresearch_graph.py
v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

AUTHORIZED TEST FILES:

```text
v4/tests/test_protocol101_stage1_gate_contract.py
v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py
v4/tests/test_protocol101_scoped_stage1_independent_audit.py
v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py
v4/tests/test_protocol101_stage1_autoresearch_graph.py
v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py
```

Do not modify any other production or test file. Audit artifacts may be written
only under the output directory below.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_gate_audit_selection_repair_attempt002/
```

PREREGISTRATION:

Before changing production code, write and hash under the output directory:

```text
preregistration.json
input_inventory.json
defect_contract.json
progress.json
```

Freeze:

- the ten exact case IDs;
- the old observed behavior;
- the required repaired behavior;
- all comparison tolerances;
- all evidence/provenance schemas;
- the allowed file list;
- the exact test commands;
- all-or-nothing repair acceptance; and
- the terminal routes.

REQUIRED REPAIRS:

## R1. Inclusive numerical boundaries

Repair `G4-CALMAR-AT` and `G6-AT` without weakening any strict gate.

- G4 Calmar `>= 1.0` must pass at the numerical boundary.
- G6 era median `>= 0.0` must pass at the numerical boundary.
- Use one explicit, documented inclusive-lower-bound comparison policy in the
  aggregator and an independently implemented equivalent in the production
  audit.
- Apply tolerance only to mathematically inclusive floating-point boundaries.
- Do not apply tolerance to strict `> 0` profitability, maxT integer
  exceedance/count rules, p-value rules, identity rules, hashes, or provenance.
- Add immediate-below, exact, immediate-above, signed-zero, infinity, NaN, and
  nonfinite tests.

## R2. Immutable unit provenance binding

Repair `FRESHNESS-changed_model_hash`.

Self-rehashing a mutated unit or packet is not proof of provenance. Add a
campaign execution-provenance authority/manifest that binds every exact
`H/P/seed/fold/unit` identity to its expected model hash, feature hash, corpus
hash, fold hash, contract hash, simulator hash, and immutable unit hash.

- The expected authority hash must enter gate aggregation from an external,
  preregistered graph/control receipt, not from the mutable campaign packet.
- Aggregator and production audit must independently verify complete 420-unit
  coverage, exact identities, exact per-unit bindings, and the authority hash.
- Missing, extra, reordered, duplicated, changed, self-resealed, or malformed
  provenance must fail closed.
- The graph must construct/freeze this authority only after the authorized RUN
  completes and before references/gates; no authorization means no authority
  and no RUN.

Synthetic validation may build synthetic authorities. Do not build one for or
inspect the real campaign.

## R3. Fresh-schema and old-evidence rejection

Repair:

```text
FRESHNESS-old_reference_schema
FRESHNESS-old_rank_payload
FRESHNESS-old_economics_marker
```

- Accept only explicit current reference/control/campaign schema allowlists.
- Reject unknown, missing, legacy, or malformed schema versions.
- Reject any prior ranking, selected-candidate, promotion, G9, holdout, old
  H0-H3 economics, benchmark-economics, or stale-campaign payload anywhere in
  gate input.
- Explicitly reject marker `OLD_H0_H3_ECONOMICS`.
- Perform recursive forbidden-field/marker checks before calculating metrics.
- Keep old 420 models and economics benchmark-only and outside the packet.

## R4. Immutable D5/D6 control receipt binding

Repair:

```text
CONTROL-D5-CHANGED-HASH
CONTROL-D6-CHANGED-HASH
```

Well-formed mutable control hashes are insufficient. Require a separate frozen
control-authority receipt, passed externally to aggregation and audit, that
binds exact D1, D5, D6, reference, and maxT artifact hashes and their accepted
schemas/routes.

- The graph freezes the control-authority receipt after the required controls
  finish and before GATES.
- Aggregator and production audit independently verify its hash and every
  embedded binding.
- D5 is dynamic campaign evidence: bind the exact independently accepted D5
  receipt produced for that campaign. Do not hard-code the current synthetic
  D5 artifact as future real authority.
- D6 is signed split-family authority: bind its exact independently accepted
  receipt.
- Any changed, missing, extra, malformed, self-resealed, or wrong-route
  receipt fails closed.

Synthetic validation may build synthetic control authorities. Do not run real
D5 or touch real evidence.

## R5. Selector freeze integrity

Repair:

```text
SELECTOR-CHANGED-FROZEN-PAYLOAD
SELECTOR-MALFORMED-FREEZE-HASH
```

- Require lowercase 64-hex SHA-256 fields.
- Recompute and verify the independent audit freeze hash.
- Bind the freeze to the exact independent-result payload hash, campaign
  authority hash, control-authority hash, aggregation hash, 28-row result hash,
  route, and source hashes.
- The selector must read/rank only data covered by the verified freeze.
- Any changed payload, stale freeze, malformed hash, missing binding, extra or
  partial family, producer-only summary, or mismatch routes exactly:
  `campaign_invalid_no_selection`.
- Preserve selection among eligible rows by plain median-seed strict-serial net
  PnL only, followed by frozen H0-H3 then P0-P6 tie order.

## R6. Graph fail-closed wiring

Wire both new authorities through:

```text
OWNER AUTH -> RUN -> EXECUTION PROVENANCE AUTHORITY
-> REFERENCES/D1/D5/D6 -> CONTROL AUTHORITY -> maxT -> GATES
-> INDEPENDENT AUDIT FREEZE -> SELECTION -> STOP
```

Without owner authorization, retain the exact route:

```text
full_trader_stage1_machinery_ready_owner_execution_authorization_required
```

The graph may not synthesize or infer owner authorization. It may not proceed
to G9, holdout, learned exits, transfer, paper, or promotion.

REPRODUCTION AND TEST CONTRACT:

1. Before repair, reproduce all ten rejection cases from the immutable
   independent packet and record exact observations.
2. After repair, rerun those ten exact deterministic cases against production.
3. Add local regression cases for every repair, including adversarial
   self-reseal/reorder/duplicate/malformed/extra-field variants.
4. Run the existing 98-test FT1C producer suite.
5. Run all six authorized focused test files.
6. Run compilation for all changed production files.
7. Run the audit-local validation runner in a fresh process.
8. Recheck rejection-packet hashes after all work.

Do not edit the independent oracle or its packet. If its reproduction command
requires a compatibility adapter, create that adapter only under the new output
directory and record it; do not alter expected results.

SELF-REPAIR LOOP:

Within this Goal, a failing test or reproduced case is not automatically a
terminal blocker. Diagnose and repair within the authorized file set, then rerun
the relevant battery. Stop only when:

- all ten cases pass and all required regressions pass; or
- a required repair would change signed policy, inspect real economics, touch a
  forbidden file, or require an owner decision.

REQUIRED OUTPUTS:

```text
preregistration.json
preregistration_freeze.sha256
input_inventory.json
defect_contract.json
before_reproduction.json
after_reproduction.json
repair_manifest.json
changed_files.json
authority_contracts.json
test_results.json
readiness_matrix.csv
summary.json
routing_decision.json
report.md
hashes.sha256
progress.json
```

Every output must state that no real campaign economics, training, seed 45/G9,
protected/sealed evidence, broker, paper-submit, promotion/default, runtime, or
launchd action occurred.

TERMINAL ROUTES:

Success:

```text
gate_audit_selection_rejection_repair_complete_pending_reacceptance
```

Bounded implementation blocker:

```text
gate_audit_selection_repair_blocked
```

Owner-policy blocker:

```text
gate_audit_selection_repair_owner_decision_required
```

HIGHEST ALLOWED CLAIM:

```text
FT1C rejection repairs implemented; fresh independent reacceptance required.
```

Do not claim machinery acceptance, training readiness, candidate eligibility,
paper readiness, or profitability.

NEXT PHASE:

Only after the success route, run a separately written, checksummed, fresh
independent FT1C reacceptance Goal. Do not reuse the first acceptance agent as
the accepting authority, and do not begin campaign execution.
