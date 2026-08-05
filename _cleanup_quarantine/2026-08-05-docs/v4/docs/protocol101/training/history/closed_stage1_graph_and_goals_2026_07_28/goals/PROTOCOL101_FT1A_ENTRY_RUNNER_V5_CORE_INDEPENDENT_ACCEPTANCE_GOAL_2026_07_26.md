# Protocol101 Goal Prompt: FT1A Entry Runner V5 Core Independent Acceptance

Run this Goal in a fresh Codex task or subagent that did not implement the
runner-core repair.

---

GOAL ID:

`FT1A-ENTRY-RUNNER-V5-CORE-INDEPENDENT-ACCEPTANCE`

OBJECTIVE:

Independently accept or reject the bounded fresh-entry runner-core repair.
Prove behaviorally, with a fresh audit-local oracle and fixtures, that the
executable fresh HGB path carries repaired two-clock decisions and uses
simulator v5 for calibration, validation, fee sensitivity, and noise
diagnostics.

Also verify fail-closed pre-fit identities, exact alpha boundaries,
fresh-campaign provenance, v5-only immutable artifacts, same-campaign resume,
truthful dry-run readiness, and protected boundaries.

This Goal is independent acceptance only. Do not repair production code, fit a
campaign model, replay campaign economics, or inspect old performance.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/protocol101/training/goals/PROTOCOL101_FT1A_ENTRY_RUNNER_V5_CORE_REPAIR_GOAL_2026_07_26.md`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/`
- every production and test file listed in the core-repair
  `changed_files.json`

FROZEN PRODUCER INPUTS:

Require these exact SHA-256 values:

```text
e8ef43c66e776f5168ee395e45effa04e534b1f6835506c434e280fb3536d1d3
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/preregistration.json

05525aa75c36b34712dc92d45192e6489940e3ca7d87811060147e68469703b5
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/implementation_manifest.json

27bcfd9f8a8c692386e9a730c6672ccaf9355bf391f68d8e81aa35ec5866797c
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/changed_files.json

83514bad1900b732500d640110db6f880b4c0de874d3765d9bf2d2a7e8f0851d
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/test_matrix.csv

8d583fa27b6bf4ee50e5b80f63ff42626c034ea254188a6fea57b388ad28ddbe
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/test_results.json

f10ac13af39171cac749d1885c0c8a42734ab97cf9362f26fa11b7454f6e6d57
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/fresh_runner_call_graph.json

1a63db5a8d9331c3f8a2075b50609a8a72d12b111aec0ff5e29db411d8dd18e1
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/v5_replay_rung_attestation.json

e2e58c15a6d7432ece0eb1793185a4eec661f5a13392972bb4eaa4fb166a367d
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/summary.json

d5ba06fa004d62e19f154b5f57b02e7473e039a2df2b1ce58c1ffc667aedf24a
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/routing_decision.json

6c7f1b30e26c62011045172607af666b3c01c9e136a610a8c8338f039cc78071
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_repair_attempt001/hashes.sha256
```

Require producer terminal route:

```text
entry_runner_v5_core_repair_complete_pending_independent_acceptance
```

Verify the producer checksum manifest and every after-hash in
`changed_files.json` before executing behavioral probes.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/
```

This is the only writable directory. Do not edit production code, tests,
documentation, signed contracts, campaign artifacts, or producer artifacts.

INDEPENDENCE CONTRACT:

Create the verifier and all fixtures under the output directory.

The fresh verifier:

- must not import the producer validation script;
- must not import producer test modules or copy their expected-output helpers;
- must not treat producer call-graph, test, or attestation files as proof;
- may independently parse production source with `ast`;
- may invoke production public interfaces as the system under test;
- may use the already independently accepted simulator-v5 and repair
  contracts as frozen dependencies;
- must construct new synthetic repaired decisions and expected invariants from
  the signed contract; and
- must record verifier source and SHA-256 in
  `independent_oracle_manifest.json`.

Run an import and source-similarity audit over the verifier. A forbidden
producer-validation or producer-test dependency is an automatic failure.

PREREGISTRATION:

Before invoking the repaired public runner or reading producer result values
beyond hashes and file inventory, write and hash:

```text
preregistration.json
source_inventory.json
independent_oracle_manifest.json
progress.json
```

Freeze:

- fixtures;
- independent expected invariants;
- behavioral probe order;
- float tolerances;
- source-integrity checks;
- exact pass aggregation;
- forbidden actions; and
- terminal routes.

The pass rule is all-or-nothing for every in-scope row.

INDEPENDENT ACCEPTANCE BATTERY:

## A. Producer and source integrity

1. Verify every frozen producer hash and checksum entry.
2. Verify every current production/test file equals the producer after-hash.
3. Verify no accepted simulator-v5, repair, artifact, loader, feature, signed
   contract, or campaign input changed during the producer run.
4. Record pre- and post-acceptance source hashes.

## B. Independent executable call graph

Parse source independently and prove:

- both public fresh runner implementations load repaired decisions;
- both invoke the explicit v5 unit entry;
- calibration reaches v5 replay;
- primary validation reaches v5 replay;
- alternate-fee replay reaches v5 replay;
- every registered noise diagnostic reaches v5 replay; and
- all those replay edges terminate in `simulate_serial_candidates_v5`.

Legacy v4 imports may remain solely for historical compatibility. Any fresh
edge to legacy selection or replay fails.

## C. Behavioral v5-rung probe

Build new synthetic `RepairedCanonicalDecision` fixtures with multiple
sessions, decisions, slots, policies, source quote times, realized exit times,
deadline exits, threshold exits, and one same-timestamp release/next-decision
case.

Use fake models or monkeypatched fit/score functions. Do not perform a real HGB
fit.

Behaviorally:

1. poison every legacy-v4 simulator and replay entry point so any call fails;
2. wrap or independently observe simulator-v5 calls;
3. invoke the public fresh v5 unit;
4. require v5 calls for every calibration threshold candidate;
5. require one primary validation replay;
6. require all three fee rungs;
7. require all three noise rungs;
8. verify selected candidate identities and both clocks;
9. verify realized exit releases occupancy before an eligible same-time
   decision;
10. verify fees change PnL exactly once and do not change either clock;
11. verify the 1.0x primary path remains the primary evidence rung; and
12. verify all emitted simulator versions are the exact accepted v5 string.

## D. Alpha and identity boundaries

Independently inspect model inputs and require:

- exact ordered feature names for H0-H3;
- no repaired exit metadata, path, future, label, or quarantined field in
  alpha; and
- payoff/return-on-premium target, not win probability.

Create duplicate-positive controls for manifest session, fold role, decision,
contract, slot, path quote, and policy axis. Monkeypatch the fit and score
boundaries to raise if entered. Every duplicate must fail before those
boundaries.

## E. Provenance, resume, and artifacts

Using temporary audit-local packet roots:

- verify exact fresh namespace and frozen campaign hashes;
- reject an old campaign namespace;
- reject a changed campaign, fold, feature, simulator, schema, source, model,
  threshold, epsilon, or manifest hash;
- accept only an exact same-campaign completed unit;
- verify the immutable root manifest is written last;
- verify every required replay payload and hash;
- reject mixed v4/v5 payloads; and
- verify source and realized clocks survive packet serialization.

Do not create a real campaign output directory.

## F. Truthful no-fit dry run and boundaries

Run the fresh dry-run into the acceptance output directory.

Require:

```text
status: v5_core_ready_pending_independent_acceptance
core blockers: []
model fit: false
model score: false
economic replay: false
seed 45: false
protected/sealed data: false
broker/paper: false
```

Independently inject a fresh v4 call edge into an audit-local source copy and
prove the readiness validator rejects it.

Verify null, heuristic, G1-G8, maxT, D1/D5/D6, selection, and G9 remain explicit
deferred blockers rather than silently passing.

## G. Regressions

Run:

- compilation of all producer-changed files;
- focused fresh-core tests;
- simulator-v5, identity, artifacts, loader, and feature-firewall tests; and
- preserved simulator-v4 regressions.

Record exact commands, counts, and exit statuses.

REQUIRED ACCEPTANCE ROWS:

Emit `acceptance_matrix.csv` with at least:

```text
IA01 producer/input integrity
IA02 verifier independence
IA03 fresh call graph v5-only
IA04 calibration behavior
IA05 validation behavior
IA06 fee behavior
IA07 noise behavior
IA08 two-clock and same-time occupancy
IA09 alpha firewall and target
IA10 pre-fit identity positive controls
IA11 fresh provenance
IA12 same-campaign resume
IA13 immutable v5 artifacts
IA14 truthful dry-run
IA15 protected boundaries
IA16 focused regressions
IA17 preserved v4 regressions
```

Every row is hard.

REQUIRED OUTPUTS:

```text
preregistration.json
source_inventory.json
independent_oracle.py
independent_oracle_manifest.json
independence_attestation.json
acceptance_matrix.csv
source_integrity.json
call_graph_independent.json
behavioral_v5_rung_validation.json
alpha_identity_validation.json
resume_artifact_validation.json
dry_run_validation.json
regression_results.json
minimal_reproducers.json
summary.json
acceptance_decision.json
report.md
progress.json
hashes.sha256
```

If any row fails, preserve a minimal reproducer and do not repair production.

TERMINAL ROUTES:

Use exactly one:

```text
entry_runner_v5_core_independently_accepted
entry_runner_v5_core_independent_repair_required
entry_runner_v5_core_independent_acceptance_blocked
```

ITERATION RULE:

Continue through verifier-only path, schema, fixture, and reporting repairs
until the packet is complete. Do not stop merely because an audit-local helper
or report validator needs correction.

Stop if:

- production behavior fails;
- a frozen input changed;
- independence cannot be established;
- campaign fitting or economic replay would be required; or
- an owner policy choice is genuinely required.

FORBIDDEN:

- production or test edits;
- campaign model fitting, scoring, or threshold selection;
- campaign economic replay;
- null, heuristic, gate, maxT, D1, D5, D6, audit, selection, or G9 execution;
- old H0-H3 performance inspection;
- seed 45;
- protected holdout or sealed recorder evidence;
- broker, paper-submit, paid-data, promotion, runtime, launchd, or real-money
  work;
- signed-document edits; and
- naming or starting another Goal before its standalone prompt exists.

HIGHEST ALLOWED CLAIM:

```text
Fresh entry-runner v5 core independently accepted; downstream evidence/gate
machinery and campaign training remain blocked.
```

Do not claim the complete campaign machinery is accepted, training is
authorized, an entry signal exists, or the Full Trader is paper-ready.

