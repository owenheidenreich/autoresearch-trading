# Protocol101 Goal Prompt: S1 Regimen Repair Design

Paste the text below into a fresh Codex Goal task.

---

GOAL ID:

`S1-REGIMEN-REPAIR-DESIGN`

OBJECTIVE:

Produce one owner-signable Protocol101 Stage-1 regimen repair contract and one
immutable evidence-rebuild specification in response to the completed
adversarial audit.

This is a design-only Goal. It must define exactly how a later implementation
will:

1. preserve the actual fixed-policy exit timestamp, reason, and executable
   exit price that produced each payoff label;
2. replay one-account occupancy, cash, daily stops, affordability, drawdown,
   and frequency from that actual exit;
3. reject duplicate decision and contract identities;
4. control the 28-row campaign's multiple-comparison and repeated-CV risk;
5. resolve or explicitly route the blocked D1, D5, and D6 diagnostics;
6. decide mechanically whether the 420 frozen models may be replayed or the
   whole H0-H3 campaign must be refit; and
7. define a fresh independent acceptance gate for the repaired machinery.

Do not implement the repair, rebuild evidence, aggregate gates, rank rows,
select a candidate, or run seed 45 in this Goal.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

CURRENT AUTHORITY:

The completed audit has:

```text
status: complete
terminal_route: regimen_invalid_redesign_required
summary_hash: 770f2d1968270c383d70216e35ca83598988e1ebe2d5e88443e03daa8005a33d
experiment_validity: 28 INVALID_EVIDENCE rows
seed_45_spent: false
protected_holdout_read: false
```

The next bounded phase is design and specification only.

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_TRADE_SHAPE_MENU_V2_PROPOSAL.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_STAGE1_AUTORESEARCH_GRAPH_2026_07_25.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_SERIAL_SIMULATOR_ACCOUNTING_REPAIR_2026_07_25.md`
- `v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/report.md`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/summary.json`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/defects.json`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/trading_semantics_audit.json`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/ml_validity_audit.json`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/gate_validity_matrix.json`
- `v4/audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/reward_hacking_smokes.json`

Inspect these implementation paths read-only:

- `v4/dataset/spxw_0dte_neural.py`
- `v4/model/protocol101_scoped_stage1_hgb.py`
- `v4/model/protocol101_serial_simulator.py`
- `v4/scripts/run_protocol101_scoped_stage1_hgb_runner.py`
- `v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py`
- `v4/scripts/run_protocol101_scoped_stage1_independent_audit.py`

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/
```

ALLOWED WORK:

- Read current contracts, source, tests, and non-protected Stage-1 artifacts.
- Freeze input paths and SHA-256 hashes before analysis.
- Trace the label, candidate, simulator, gate, null, and heuristic schemas.
- Create design documents, JSON schemas, decision tables, and test
  specifications beneath the output directory.
- Create audit-only analysis scripts beneath the output directory when needed
  to inspect schemas or prove that a proposed field can be derived causally.
- Compare old and proposed schemas without computing repaired PnL, gates, or
  rankings.
- Continue through mechanical documentation or schema-inspection blockers.

FORBIDDEN WORK:

- Do not modify governed dataset, model, simulator, runner, gate, graph, or
  runtime code.
- Do not train, refit, recalibrate, tune thresholds, or execute feature uplift.
- Do not rebuild or replay H0-H3 economic evidence.
- Do not calculate repaired row PnL or repaired G1-G8 results.
- Do not rank H0-H3 or policies.
- Do not select or freeze a candidate.
- Do not run or inspect seed 45/G9.
- Do not read the protected holdout.
- Do not open sealed recorder evidence.
- Do not contact IBKR or any broker endpoint.
- Do not download paid data.
- Do not submit paper orders.
- Do not change promotion, defaults, runtime flags, launchd, or real-money
  paths.
- Do not amend a signed contract in place. Produce a draft owner amendment.

CHECKPOINT A - FREEZE AND DEFECT RECONCILIATION:

Before proposing a repair:

1. Write `source_inventory.json` with every consumed path, role, byte count,
   and SHA-256 hash.
2. Record the pre-existing repository status.
3. Verify the audit terminal route and summary hash above.
4. Reconcile all six unresolved defects:
   - `P1-SIM-001`
   - `P1-MULT-001`
   - `P1-DUP-001`
   - `P2-DIAG-001`
   - `P2-BENCH-001`
   - `P2-SOURCE-001`
5. Write `defect_resolution_matrix.json` showing for each defect:
   proposed resolution, later implementation owner, required tests, evidence
   invalidated, and whether owner signature is required.

If the audit packet, signed contracts, and current source disagree, preserve
the conflict and route it explicitly. Do not silently choose the most
convenient version.

CHECKPOINT B - REALIZED-EXIT DATA CONTRACT:

Define a successor processed-row and selected-candidate schema that carries,
for every decision slot and fixed policy:

- label net PnL;
- label mid PnL;
- realized exit timestamp;
- realized exit reason;
- executable exit bid;
- source exit quote timestamp;
- policy deadline;
- policy index; and
- an explicit missing/invalid reason when no causal executable path exists.

The specification must pin:

- timestamp timezone and precision;
- array shapes and alignment with strike/right/policy axes;
- allowed exit-reason vocabulary;
- stop-loss, take-profit, max-hold, and forced-flat precedence;
- no-bid behavior;
- missing-path behavior;
- same-minute ordering;
- entry `<` exit and same-session requirements;
- the exact relationship among entry ask, exit bid, fee, and stored net PnL;
- forced-flat cap behavior; and
- reference and vectorized implementation equivalence.

The exit timestamp used for occupancy must be the same event that generated
the label PnL. A later implementation may not synthesize occupancy from
maximum hold when an earlier stop or target generated the payoff.

Exit/path fields remain labels and audit data. They must be structurally
blocked from every model-facing feature matrix.

Write:

- `realized_exit_contract.json`
- `realized_exit_contract.md`
- `processed_row_schema_proposal.json`

CHECKPOINT C - SERIAL SIMULATOR SUCCESSOR CONTRACT:

Design a successor to:

```text
protocol101_serial_simulator_v4_account_continuity_fee_reserve
```

The new version must preserve the already-correct v4 rules for account
continuity, one contract, premium-plus-fee affordability, session-starting
equity, the 5% daily stop, entry cutoff, forced flat, and stress reporting.

It must change exit-time semantics to:

```text
label_realized_exit_time
```

Pin:

- realization before evaluating a new entry at the same timestamp;
- overlap and cash availability from actual realized exits;
- trade and skipped-event trace schemas;
- exit timestamp/reason in candidate payload hashes;
- candidate stream and payload ordering;
- invalid exit timestamp/reason behavior;
- exact end-of-session pending realization; and
- backward compatibility policy for v4 artifacts, which remain historical and
  may never masquerade as repaired evidence.

Write:

- `serial_simulator_successor_contract.json`
- `serial_simulator_successor_contract.md`
- `immutable_replay_artifact_schema.json`

CHECKPOINT D - FAIL-CLOSED IDENTITY CONTRACT:

Governed paths must fail closed before fitting, scoring, hashing, or replay
when any of these identities are duplicated:

- session membership;
- `(split, session, decision_time)` decision identity;
- `(split, session, decision_time, contract_id)` contract identity; or
- canonical strike/right slot identity inside one decision.

Do not silently keep first/last and do not allow row order to decide the
winner. Pin explicit exception types, blocker codes, audit counts, and positive
control tests.

Write `identity_uniqueness_contract.json`.

CHECKPOINT E - REPLAY-ONLY VERSUS FULL-REFIT BRANCH:

Freeze one mechanical branch before any repaired evidence is produced.

Replay-only reuse of the 420 frozen models is allowed only if an independent
equivalence certificate proves, across the full governed corpus:

- identical session and fold membership;
- identical decision and contract identities;
- identical model-facing feature names, order, values, and missingness;
- identical candidate masks and entry asks;
- identical `labels_net_pnl` and `labels_mid_pnl`;
- identical model target rows and weights;
- identical calibration and threshold-fitting inputs;
- identical frozen model, threshold, epsilon, and selection-contract hashes;
- newly added fields are limited to exit/audit metadata; and
- frozen-model predictions are exactly reproducible on a preregistered sample
  and hash-reproducible on the full rebuilt corpus.

If every item passes, the later rebuild may reuse frozen models and frozen
selection parameters but must regenerate all candidate, serial, null,
heuristic, gate, and audit evidence under the successor simulator.

If any item fails, replay-only reuse fails closed. The later campaign must
refit all 420 units under a new immutable attempt namespace using the same
H0-H3 hypotheses, seven policies, seeds 42-44, folds, model family, and signed
training contract. Partial model reuse or a mixture of old/new unit evidence
is forbidden.

Write:

- `replay_refit_branch_specification.json`
- `equivalence_certificate_schema.json`

CHECKPOINT F - MULTIPLICITY AND PROCESS-OVERFIT CONTROL:

Design one global, uniformly applied campaign control for the frozen family of:

```text
4 hypotheses x 7 policies = 28 rows
```

The design must:

- preserve dependence among rows sharing sessions and folds;
- use no protected, recorder, shadow, or seed-45 evidence;
- prevent a favorable row from being chosen merely because 28 rows were
  inspected;
- define the family, statistic, randomization unit, repetitions, seeds,
  adjusted measure, threshold, and handling of ties;
- apply identically to all 28 rows;
- state whether it is a hard eligibility control or a report-only diagnostic;
- distinguish statistical multiplicity from human repeated-campaign risk; and
- prohibit changing the method after repaired row results are visible.

Primary recommendation:

- a synchronized session-block max-statistic/family-wise null across all 28
  rows, preserving cross-row dependence;
- existing per-row G2 remains necessary;
- recommend an exact adjusted eligibility threshold; and
- if a valid joint null cannot be constructed, fail closed rather than
  substituting a post-result approximation.

Also provide one conservative alternative and explain its power cost. Do not
compute either method on H0-H3 results in this Goal. The owner packet must make
the final hard-versus-report-only choice explicit before the rebuild.

Write:

- `campaign_multiplicity_control_proposal.json`
- `campaign_multiplicity_control_proposal.md`

CHECKPOINT G - BLOCKED DIAGNOSTIC RESOLUTIONS:

Pin these before implementation:

1. **D1 label permutation:** use only complete 30-minute blocks. Exclude the
   trailing 29-minute partial block from this diagnostic only, record exactly
   29 excluded rows per 359-row full session, and preserve the existing 20
   seeds and pass criteria. Do not exclude those rows from training or normal
   replay.
2. **D5 heuristic:** rebuild the unchanged fixed heuristic under the successor
   exit/simulator contract before candidate aggregation. Persist ordered trade
   identities, identity hash, candidate stream/payload hashes, per-fold
   metrics, and pooled metrics. The old `$4,592` result remains historical and
   cannot be the repaired G3 benchmark.
3. **D6 source identity:** reconcile this diagnostic with the signed scoped
   synchronization decision. Recommend whether the signed split-family
   evidence is sufficient for offline repaired Stage-1 work while exact
   17-feature candidate transfer remains deferred to the mandatory no-order
   shadow gate. Do not access sealed evidence. If a new exact-vector
   diagnostic is truly required before offline rebuild, identify the smallest
   non-protected evidence and separate owner authorization needed.

Write `blocked_diagnostic_resolution.json`.

CHECKPOINT H - IMMUTABLE REBUILD AND INDEPENDENT ACCEPTANCE SPEC:

Define the later sequence without executing it:

```text
owner signs repair amendment
  -> implement dataset/simulator/identity machinery
  -> independent machinery acceptance
  -> equivalence certificate
  -> replay-only rebuild OR full 420-unit refit
  -> rebuild null and heuristic references
  -> rebuild all 28 row packets
  -> independent RUN/GATE/AUDIT acceptance
  -> model-free selection
  -> seed-45 G9
```

The implementation acceptance specification must include:

- reference-versus-vectorized label equality for PnL, exit time, reason, and
  exit price across all seven policies;
- synthetic stop, target, max-hold, forced-flat, no-bid, missing-path, and
  same-timestamp cases;
- actual-exit overlap, cash, daily-stop, drawdown, and frequency tests;
- duplicate positive controls at every governed boundary;
- proof that exit/path fields cannot enter model alpha;
- preservation of v4 account-continuity and fee-reserve tests;
- deterministic hashes and resume behavior;
- cross-implementation verification by a fresh task; and
- zero access to seed 45, holdout, sealed evidence, broker, or paper state.

The producing implementation task may not independently accept itself.

Write:

- `immutable_evidence_rebuild_specification.json`
- `independent_acceptance_specification.json`
- `repair_test_matrix.csv`

BLOCKER RECOVERY LOOP:

Do not stop at the first missing path, stale reference, schema mismatch, or
audit-tooling failure.

For every blocker:

1. Classify it as:
   - `transient`
   - `mechanical_non_substantive`
   - `contract_conflict`
   - `protected_action`
   - `scientific_owner_decision`
   - `missing_nonprotected_evidence`
2. Record it in `blockers.jsonl`.
3. Retry transient failures with bounded backoff.
4. Repair audit-only tooling and documentation mechanically when that does not
   change a scientific rule.
5. Preserve failed attempts beneath `void_outputs/`.
6. Continue every unaffected checkpoint.
7. Use at most three mechanical repair attempts per component.
8. Stop for owner input only when a genuine scientific or governance choice
   remains. Present the exact choice, recommendation, and consequence.

Never resolve a blocker by inspecting protected data, calculating repaired
winner economics, weakening a gate, changing H0-H3, or spending seed 45.

REQUIRED TERMINAL PACKET:

```text
source_inventory.json
preexisting_repository_status.txt
defect_resolution_matrix.json
realized_exit_contract.json
realized_exit_contract.md
processed_row_schema_proposal.json
serial_simulator_successor_contract.json
serial_simulator_successor_contract.md
immutable_replay_artifact_schema.json
identity_uniqueness_contract.json
replay_refit_branch_specification.json
equivalence_certificate_schema.json
campaign_multiplicity_control_proposal.json
campaign_multiplicity_control_proposal.md
blocked_diagnostic_resolution.json
immutable_evidence_rebuild_specification.json
independent_acceptance_specification.json
repair_test_matrix.csv
owner_decision_packet.json
owner_decision_packet.md
progress.json
summary.json
hashes.sha256
report.md
```

The owner packet must contain a concise signable amendment with every new
scientific/governance decision highlighted. It must not claim the amendment is
signed.

TERMINAL ROUTES:

- `repair_design_ready_for_owner_signature`
- `repair_design_owner_decisions_required`
- `repair_design_blocked_contract_conflict`
- `repair_design_blocked_missing_nonprotected_evidence`

`repair_design_ready_for_owner_signature` requires:

- all six defects have a proposed resolution;
- exact realized-exit and successor-simulator semantics are pinned;
- duplicate handling fails closed;
- replay-only/refit branching is mechanical and immutable;
- multiplicity has a primary recommendation and exact proposed threshold;
- D1 and D5 are fully specified;
- D6 has an owner-signable route consistent with synchronization authority;
- the rebuild and independent acceptance tests are complete; and
- all required files and hashes validate.

STOP BOUNDARY:

Stop after the design packet. Do not implement it. Do not start the machinery
repair, evidence rebuild, gate aggregation, selection, G9, holdout, shadow, or
paper phases.

HIGHEST ALLOWED CLAIM:

> Protocol101 Stage-1 regimen repair design complete and ready for owner
> decision.

FINAL RESPONSE:

Explain in plain English:

1. What was wrong with the old replay?
2. What exact information will now survive from labels into replay?
3. When may the 420 existing models be reused, and when is a full refit
   mandatory?
4. How will duplicates and 28-row model shopping be controlled?
5. How are D1, D5, and D6 resolved or routed?
6. What must the owner sign?
7. What is the one next Goal after signature?

