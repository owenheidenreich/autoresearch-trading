# Protocol101 Goal Prompt: FT1B Reference And Multiplicity Independent Acceptance

Run this Goal in a fresh Codex task or subagent that did not implement FT1B.

---

GOAL ID:

`FT1B-REFERENCE-AND-MULTIPLICITY-INDEPENDENT-ACCEPTANCE`

OBJECTIVE:

Independently accept or reject the repaired Stage-1 reference and multiplicity
foundation. Use a fresh audit-local oracle and fixtures to verify:

- matched random null/canary semantics;
- fixed policy-5 heuristic replay under simulator v5;
- D1 complete-block label-permutation semantics;
- D6 authority boundaries; and
- the exact hard synchronized 20,000-replicate five-session moving-block maxT
  implementation for the 28-row H0-H3/P0-P6 family.

This is acceptance only. Do not repair production, fit or score campaign
models, execute campaign economics, aggregate G1-G8, rank or select rows, spend
seed 45, or inspect protected/sealed evidence.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/protocol101/training/goals/PROTOCOL101_FT1B_REFERENCE_AND_MULTIPLICITY_MACHINERY_REPAIR_GOAL_2026_07_26.md`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/produce_design_packet.py`
  sections `build_multiplicity` and `build_blocked_diagnostics`
- every top-level producer artifact under
  `v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_machinery_attempt001/`
- all six files in producer `changed_files.json`
- accepted runner-v5 core and simulator-v5 contracts

FROZEN PRODUCER ARTIFACTS:

Require these exact SHA-256 values:

```text
6491349a1eb975a92db956d8e3876186911e33f796cb907c2fc7c9e2af380ccf
  preregistration.json
c421b66d58d2009f275fd3b4003641f0075cebd85e31fba6011aeebd66b6bedd
  implementation_manifest.json
f00031130a74b54af2e0adfbda47fa98ed35f4fc00529f3e066d07b6f30c70c7
  changed_files.json
ffaf878bf5a7fd3b1f20bef77c176a3b1b8bbbcf2b05dbf02391cbb1d9af2af2
  reference_v5_call_graph.json
3b09633ffb1020872c100e6bfc72c9018490d3589e957d3a9cc78c8ae6d9edae
  synthetic_reference_validation.json
9a0f9f937bd3b6b5262353888068e4dbc8bc47f9b478cca2a0844c935195ccdf
  d1_contract_validation.json
c001194cd188dba303d82814121295a5bd98f9092db52a263a06f22036f148d1
  d5_contract_validation.json
1823d877dc61aad11d588c3322d75e07e1cdd9a9b28bf24f12a6674c869081c1
  d6_authority_receipt.json
079de972c585105db5f5e24bff948e61cf0a3d983e5137b28a9013f2d1f1ed4f
  maxT_contract.json
1355f026c9502e081890661c2ed9a618a9a0704c85b6d19e86882fa34d835d39
  maxT_schedule_manifest.json
b8f216bcfe745c6a836477f32dd9c50672047c13d3be755bc5e1ca478064fcfb
  maxT_synthetic_validation.json
6031ff9c067a925985facc71a03373a9d3078613caa5740933ffecf9c237e0a1
  readiness_matrix.csv
9d57dad5acac6d753cea17bf59c94df43e412f893079344a009eb52683903942
  test_results.json
7c782c0ee10bcf1271546c03bb758069142069d20cfd5b7c249064905aa290a0
  summary.json
3f85eef5b15987a7fde69cb5b138c5184ffb52a0dfabc8a3097b4c2a52ca97d3
  routing_decision.json
f00aff0197236c25db479c2d556ac87c11d33adaa093bfd490072b781c8cba52
  hashes.sha256
```

The paths above are relative to the producer output directory.

Require these exact current source hashes:

```text
f9ae712658f3590ace7995b1d744aea27cfc752522837018d445f7163fdad419
  v4/scripts/run_protocol101_scoped_stage1_reference_packets.py
079eea0464bb7f00b139018e0d23c0d31f7a649aab9b1b08a1a32a5d01b7277b
  v4/model/protocol101_stage1_reference_multiplicity.py
f699de1eb35a3cb71f91c124f09570527bf68b98d3b150b828a8029336b2c393
  v4/scripts/run_protocol101_full_trader_stage1_reference_multiplicity_validation.py
890ebca0adcc7c402185cb02d8680a8e9238c5ff21615e65b29794d5b31adf6c
  v4/tests/test_protocol101_scoped_stage1_reference_packets.py
3ef541cc022958b87533333ade79033f3d61cb568b70b21f02cf9f644fe10c10
  v4/tests/test_protocol101_stage1_reference_multiplicity.py
3d080157f3eabfd42e200880cecfb77f2fa1c5a289920eaddc3482a5ab76d6d7
  v4/tests/test_protocol101_full_trader_stage1_reference_multiplicity_validation.py
```

Require producer route:

```text
reference_multiplicity_machinery_repair_complete_pending_independent_acceptance
```

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/
```

This is the only writable directory. Do not edit production, tests,
documentation, contracts, campaign artifacts, or producer artifacts.

INDEPENDENCE CONTRACT:

Create all verifier code and fixtures under the output directory.

The verifier:

- must not import the producer validation script;
- must not import producer test modules;
- must not copy producer fixtures, expected-output helpers, or generated
  expected values;
- must not treat producer reports, call graphs, or validations as proof;
- may independently parse production source with `ast`;
- may invoke public production interfaces as the system under test;
- may use independently accepted simulator-v5 and repair contracts;
- must independently implement the signed D1 and maxT reference oracle; and
- must record verifier source and SHA-256 in
  `independent_oracle_manifest.json`.

Run an import and source-similarity audit. A forbidden producer validator/test
dependency is an automatic failure.

PREREGISTRATION:

Before invoking producer machinery or reading producer result values beyond
hashes/file inventory, write and hash:

```text
preregistration.json
source_inventory.json
independent_oracle_manifest.json
progress.json
```

Freeze independent fixtures, expected invariants, tolerances, probe order,
pass aggregation, forbidden actions, and terminal routes. Every acceptance row
is required; no partial pass.

INDEPENDENT ACCEPTANCE BATTERY:

## A. Integrity

1. Verify every frozen producer artifact and checksum.
2. Verify all six current source hashes.
3. Record pre/post hashes for all accepted runner/simulator/repair dependencies.
4. Verify only the six allowed producer files changed in FT1B.

## B. Fresh reference v5 behavior

Build new repaired two-clock decisions spanning folds, sessions, calls, puts,
several slots, threshold exits, deadline exits, fees, overlap, daily stop, and
same-time release/next-entry.

Poison every v4 replay entry point. Invoke fresh public null and D5 reference
interfaces and prove:

- all economic replay terminates in simulator v5;
- source quote time prices exit and realized time releases occupancy;
- same-time release permits the eligible next decision;
- fees apply once;
- identities survive selection and serialization;
- duplicates/nonfinite inputs fail before replay;
- P0-P6 random settings are exactly 200 draws and seed 101; and
- old namespaces and old `$4,592` economics are not imported.

Independently recompute sampled random identities for a small frozen fixture
and compare exactly. Run independent positive, no-edge/negative, and poisoned
identity canaries.

## C. D5 identity-complete heuristic

Independently encode the frozen VWAP-side nearest-ATM P5 rule on new fixtures.
Require exact producer-system agreement for:

- ordered candidate intents;
- candidate stream and payload hashes;
- ordered trade identities and trade hash;
- fold and pooled v5 metrics;
- both exit clocks; and
- immutable packet write order with manifest last.

Perturb candidate order, identity, payload, and clock fields separately and
prove the corresponding hash or fail-closed result changes.

## D. D1 independent oracle

Build fresh 359-row sessions containing multiple slots and distinctive feature,
identity, label, ask, and exit-clock values.

Using an independent implementation, require:

- first 330 rows only;
- 11 complete 30-row blocks;
- exactly 29 trailing rows excluded from D1 only;
- seeds 8600-8619;
- whole-ladder P5 net/mid labels move together;
- features, identities, asks, and exit metadata remain fixed;
- validation labels remain unpermuted and validation is truncated to 330;
- permuted fit rows are non-candidate/non-replayable; and
- normal training/replay remains 359 rows.

Compare independent source-row/block schedules to the production system for
multiple D1 seeds. Independently verify the frozen later aggregation law:
median PnL <= 0, median z < 1, and joint G1/G2 <=1 of 20.

## E. D6 authority

Independently reconstruct and hash the authority receipt from signed sources.
Require exact route, ordered 17 features, alpha prohibitions, no sealed access,
no broker/paper authority, and deferred exact candidate transfer. Mutated
contract, feature order, added exit-alpha field, or changed synchronization
hash must fail.

## F. Independent maxT oracle

Without using producer schedule or statistic helpers, independently implement:

- exact ordered 28 rows and seeds 42-44;
- one identical chronological five-fold session grid;
- fold-wise centering;
- synchronized circular moving blocks of length five;
- `ceil(n_fold/5)` uniform starts and exact truncation;
- NumPy `PCG64DXSM`;
- `SeedSequence(2026072601).spawn(20000)` in order;
- one shared schedule across all rows/seeds;
- the signed denominator, observed Z/T, null T, family max, `>=` ties,
  exceedance counts, and `(1+count)/20001` p-values.

Use at least two new synthetic grids with uneven fold lengths. For the full
20,000 replicates, require exact equality between independent and production:

- every sampled schedule index;
- schedule SHA-256;
- denominator by row/seed;
- observed Z and T;
- per-replicate per-row null T;
- family max vector;
- exceedance counts;
- p_FWER; and
- hard-pass booleans.

Persist independent arrays and hashes. The producer's published synthetic
values may be checked only after the independent oracle has frozen and run.

Independently test every fail-closed condition: missing/reordered row, missing
or reordered seed, grid mismatch, noncontiguous folds, duplicated session,
nonfinite PnL, zero variance, changed schedule, schedule not frozen, and
cross-row schedule divergence.

Independently test threshold arithmetic at counts 999 and 1000.

## G. Readiness and regressions

Independently verify:

- G8 remains report-only;
- real references, campaign economics, G1-G8 aggregation, ranking, selection,
  seed 45, G9, protected/sealed evidence, learned exits, and broker paths are
  still blocked;
- accepted runner-v5 core source hashes remain unchanged; and
- producer packet resume/freeze behavior is order-stable from a fresh process.

Run compilation, the exact producer-focused 76-test command, and independent
verifier tests. Do not run the entire repository suite.

REQUIRED OUTPUTS:

```text
preregistration.json
source_inventory.json
independent_oracle.py
independent_oracle_manifest.json
independence_audit.json
producer_integrity.json
reference_v5_acceptance.json
d1_independent_acceptance.json
d5_independent_acceptance.json
d6_independent_acceptance.json
maxT_independent_schedule.npy
maxT_independent_null_t.npy
maxT_independent_maxima.npy
maxT_independent_acceptance.json
fail_closed_matrix.csv
acceptance_matrix.csv
regression_results.json
progress.json
acceptance_decision.json
summary.json
report.md
hashes.sha256
```

Write `hashes.sha256` last and cover every top-level output except itself.

TERMINAL ROUTES:

All rows pass:

```text
reference_multiplicity_machinery_independently_accepted
```

Any reproducible defect:

```text
reference_multiplicity_machinery_independent_rejection
```

Missing/changed frozen input:

```text
reference_multiplicity_machinery_acceptance_input_blocked
```

Do not repair production in this Goal. On rejection, preserve the reproducer
and stop.

FORBIDDEN:

- production, test, documentation, contract, producer, or campaign edits;
- campaign fit, score, or economic replay;
- real null/D1/D5/maxT execution on campaign results;
- G1-G8 aggregation, ranking, selection, seed 45, or G9;
- protected holdout or sealed evidence;
- learned exits;
- broker/API, paper submit, paid download, promotion/default, runtime,
  launchd, or real-money changes; and
- acceptance based solely on producer-authored tests or reports.

HIGHEST ALLOWED CLAIM:

> Stage-1 reference and multiplicity machinery independently accepted.

On pass, the sole next phase is a separately written bounded Goal to repair
the G1-G8 aggregation, independent audit, ranking, and selection stack. Do not
start it here.

