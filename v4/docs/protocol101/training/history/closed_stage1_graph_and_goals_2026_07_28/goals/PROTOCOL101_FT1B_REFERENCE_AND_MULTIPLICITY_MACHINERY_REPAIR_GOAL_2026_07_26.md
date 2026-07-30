# Protocol101 Goal Prompt: FT1B Reference And Multiplicity Machinery Repair

Run this Goal in a fresh Codex task or subagent.

---

GOAL ID:

`FT1B-REFERENCE-AND-MULTIPLICITY-MACHINERY-REPAIR`

OBJECTIVE:

Implement the bounded, non-economic Stage-1 evidence foundation required by
the signed Full Trader entry campaign:

1. matched random-selection null and canary packets under simulator v5;
2. the unchanged policy-5 fixed heuristic under simulator v5;
3. the D1 complete-block label-permutation control;
4. the signed D6 offline-rebuild authority receipt; and
5. the hard synchronized 20,000-replicate five-session moving-block maxT
   multiplicity engine for the frozen 28-row H0-H3/P0-P6 family.

This Goal repairs and validates machinery only. Use synthetic or audit-local
fixtures to prove behavior. Do not fit a campaign model, execute references
against campaign economics, aggregate G1-G8, rank rows, select a model, spend
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
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_design_attempt001/produce_design_packet.py`
  sections `build_multiplicity` and `build_blocked_diagnostics`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/`
- `v4/scripts/run_protocol101_scoped_stage1_reference_packets.py`
- `v4/tests/test_protocol101_scoped_stage1_reference_packets.py`
- the accepted simulator-v5, repaired-decision, governed-loader, feature
  firewall, and immutable-artifact implementations and tests

FROZEN INPUT HASHES:

Require these exact SHA-256 values before editing:

```text
b02a99281b502434675c3440e9f214fc88b9ac974359e20bb7841e19a2a8065b
  v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md

d3fc327cdf82159f73388648c15643f42c2207a420c3cdab9c4fb08cc302cbc0
  v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md

7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/campaign_contract.json

40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/preregistration.json

72cbe1443cbddeb5af6dbb09bf384d9ea479658a4cefacbcda5b9c8463600bd2
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/acceptance_decision.json

b5a70e0dcc6df7c0bd95408a65d1561beaccb615c75ea978060eab15c60b41e0
  v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/hashes.sha256
```

Require the runner-core acceptance route:

```text
entry_runner_v5_core_independently_accepted
```

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_reference_multiplicity_machinery_attempt001/
```

PRODUCTION SCOPE:

Prefer the smallest coherent implementation. Authorized files are:

```text
v4/scripts/run_protocol101_scoped_stage1_reference_packets.py
v4/model/protocol101_stage1_reference_multiplicity.py
v4/scripts/run_protocol101_full_trader_stage1_reference_multiplicity_validation.py
v4/tests/test_protocol101_scoped_stage1_reference_packets.py
v4/tests/test_protocol101_stage1_reference_multiplicity.py
v4/tests/test_protocol101_full_trader_stage1_reference_multiplicity_validation.py
```

The two new production paths may be created. Do not edit the accepted runner
core, simulator v5, repair contracts, dataset builder, governed loader, feature
firewall, signed contracts, campaign preregistration, or prior evidence.

PREREGISTRATION:

Before executing comparisons or synthetic validation, write and hash:

```text
preregistration.json
source_inventory.json
progress.json
```

Freeze:

- exact algorithms below;
- fresh artifact namespaces;
- synthetic fixtures and expected invariants;
- float tolerances;
- all pass/fail aggregation rules;
- changed-file allowlist;
- forbidden actions; and
- terminal routes.

Do not alter any threshold or algorithm after viewing a result.

## A. Simulator-v5 reference contract

Both the matched random reference and fixed heuristic must consume repaired
two-clock decisions and terminate in `simulate_serial_candidates_v5`.

Require:

- source quote time prices the exit;
- realized exit time releases account occupancy;
- fees apply exactly once;
- v4 replay is impossible on every fresh reference path;
- exact campaign/fold/session/decision/contract/slot/policy identities;
- fail-closed duplicate and nonfinite-input preflight before reference replay;
- identical one-account serial, cash, daily-stop, frequency, and embargo
  semantics to the accepted fresh runner; and
- immutable reference artifacts with simulator/config/code/input hashes.

Preserve the historical reference CLI only as explicitly labeled legacy
compatibility. Fresh outputs must use new namespaces and may never overwrite or
reinterpret old reference packets.

## B. G2 matched random null and canaries

Preserve the existing preregistered matched-random settings:

```text
draws per policy: 200
PRNG seed: 101
policies: P0-P6
```

Do not confuse these 200 matched-random draws with the 20,000 maxT
replicates.

Each draw must:

- operate on the same governed session/fold opportunity grid;
- respect entry eligibility, affordability, serial occupancy, daily stop,
  fees, and all accepted v5 constraints;
- use simulator v5;
- preserve zero-trade days naturally; and
- persist the sampled identities or an exact reproducible schedule/hash.

Implement mechanical canaries that prove the null responds to:

- no-skill random selection;
- a deliberately impossible/poisoned identity;
- a known synthetic positive edge; and
- a known synthetic no-edge or negative edge.

This Goal may run those canaries only on synthetic fixtures.

## C. D5 fixed heuristic

Keep the existing frozen heuristic rule unchanged and policy fixed at P5.

The fresh D5 path must rebuild it under repaired rows and simulator v5 before
candidate aggregation or selection. It must persist:

```text
ordered candidate intents
candidate stream hash
candidate payload hash
ordered trade identities
trade identity hash
per-fold metrics
continuous pooled metrics
simulator/config/code/input hashes
```

The old `$4,592` value is historical and must not be imported, asserted,
compared as current evidence, or used as a test oracle.

Prove with synthetic fixtures that the candidate and trade identities are
stable, complete, order-sensitive, and reproducible.

## D. D1 label-permutation control

Implement the signed D1 control exactly:

```text
shape: H2-like bounded HGB, P5, NON_CANDIDATE
normal rows per full session: 359
D1 rows per full session: first 330 only
block size: 30 rows
complete blocks: 11
trailing rows excluded from D1 only: exactly 29
seeds/repetitions: 8600 through 8619, exactly 20
```

For each fit and calibration session:

- restrict to the first 330 decision rows;
- permute the eleven complete 30-row blocks;
- move the whole ladder's P5 net and mid label arrays together through a
  non-candidate target-override layer;
- do not move features, identities, entry asks, or canonical realized-exit
  metadata; and
- never replay permuted fit rows as economic trades.

Validation labels/exits remain unpermuted, but validation also excludes its
trailing 29 rows for D1 only. Normal campaign training and replay use all 359
rows.

Freeze D1 result aggregation for later execution:

```text
median pooled fee-adjusted PnL <= 0
median matched-null z < 1.0
joint G1/G2 passes <= 1 of 20
```

Two or more joint G1/G2 passes, or median z >= 1.0 without a preregistered
mathematical explanation, invalidates the control.

This Goal implements and synthetically validates block geometry and target
movement. It must not perform 20 real HGB fits.

## E. D6 authority receipt

Emit a deterministic D6 authority receipt that records:

```text
route:
SIGNED_SPLIT_FAMILY_SYNCHRONIZATION_SUFFICIENT_FOR_OFFLINE_LABEL_ONLY_REPAIR
offline 17-feature rebuild: allowed
repaired exit/label/audit metadata as alpha: forbidden
exact candidate transfer: deferred to mandatory no-order shadow
sealed evidence access: forbidden
broker/paper authority: none
```

The receipt must bind the signed synchronization decision, repair amendment,
campaign contract, exact ordered 17-feature firewall, and their hashes. It is
governance evidence, not a new synchronization claim.

## F. Hard 28-row maxT multiplicity engine

Implement exactly this signed family:

```text
rows: H0-H3 crossed with P0-P6
family size: 28
seeds: 42, 43, 44
input: repaired per-session OOF net PnL on one identical session/fold grid
purpose: one-sided FWER control of positive OOF net PnL
```

Observed statistic:

```text
Z[r,s] =
  pooled OOF net PnL[r,s] /
  SD_b(centered block-resampled pooled PnL[r,s])

T[r] = median over seeds 42,43,44 of Z[r,s]
```

Null statistic:

```text
T_null[b,r] =
  median over seeds of
  centered block-resampled pooled PnL[b,r,s] /
  the same frozen SD_b denominator[r,s]

M[b] = max over all 28 rows of T_null[b,r]
```

Within each of the five chronological outer-validation folds:

1. center each row/seed session-PnL series by that fold's mean;
2. use circular moving blocks of five consecutive sessions;
3. draw `ceil(n_fold / 5)` block starts uniformly from `0..n_fold-1`;
4. concatenate circular blocks and truncate to exactly `n_fold`; and
5. use the exact same sampled session-index schedule for all 28 rows and all
   three seeds in each replicate.

Freeze:

```text
replicates: 20,000
PRNG: NumPy PCG64DXSM
master seed: 2026072601
replicate seeds: SeedSequence(master_seed).spawn(20000), order 0..19999
tie rule: greater than or equal
p_FWER[r] = (1 + count(M[b] >= T_observed[r])) / 20001
hard pass: p_FWER <= 0.05, equivalent exceedance count <= 999
existing per-row G2: also required later
```

Persist every sampled index and an aggregate schedule SHA-256 before observed
adjusted p-values are evaluated.

Fail the whole joint null closed if:

- any row or seed is absent;
- session/fold grids differ;
- schedules differ across rows or seeds;
- nonfinite PnL is present;
- zero variance prevents any all-row statistic;
- the schedule was not frozen before observed evaluation; or
- schedule/max statistics cannot be exactly reproduced.

The signed maxT is a hard eligibility control. Do not implement Bonferroni as
an alternate selectable path. Do not permit post-result method switching.

Use small synthetic replicates for fast unit tests and one deterministic
20,000-replicate synthetic integration check. Record runtime and schedule
hash. Do not feed campaign economics to the engine in this Goal.

## G. Fresh readiness validator

Create a no-fit readiness command that proves:

- the independently accepted v5 runner core remains unchanged;
- fresh reference paths are v5-only;
- D1/D5/D6 contracts are executable and hashed;
- the 28-row maxT engine and frozen schedule are executable;
- G8 is report-only;
- campaign execution, G1-G8 aggregation, ranking, selection, and G9 remain
  blocked; and
- all protected boundaries remain false.

Its honest success status is:

```text
reference_multiplicity_machinery_ready_pending_independent_acceptance
```

## H. Tests and validation

Required tests include:

1. v4 poisoning for every fresh reference replay route;
2. exact two-clock preservation and same-time account release;
3. fee-once behavior;
4. null schedule reproducibility and identity completeness;
5. D5 identity/order/hash sensitivity;
6. exact D1 11-by-30 geometry and 29-row D1-only exclusion;
7. D1 moves label arrays together and does not move features/identities/clocks;
8. D6 exact authority and alpha prohibition;
9. maxT exact 28-by-3 grid acceptance;
10. maxT all fail-closed cases;
11. PCG64DXSM/SeedSequence schedule reproducibility;
12. exact p-value/tie/exceedance arithmetic;
13. deterministic 20,000-replicate synthetic integration;
14. fresh namespace and immutable artifact ordering;
15. no-fit readiness truthfulness; and
16. regression tests for accepted runner core, simulator v5, identities,
    artifacts, governed loader, and feature firewall.

Run compilation and focused pytest. Do not run the entire repository suite.

SELF-REPAIR LOOP:

Do not stop at the first ordinary implementation or test failure.

Within the authorized files:

1. reproduce the defect;
2. repair it;
3. rerun the narrow failing test;
4. rerun the focused suite; and
5. continue until success or a genuine owner blocker exists.

An owner blocker means only:

- a signed authority contradiction;
- a required frozen input is missing or hash-mismatched before this Goal
  begins;
- the requested behavior requires changing an unsigned policy choice; or
- progress is impossible without protected/sealed data, broker access, paid
  download, or campaign economic execution.

Code complexity, a failing test, long runtime, or stale legacy code is not an
owner blocker.

REQUIRED OUTPUTS:

```text
preregistration.json
source_inventory.json
implementation_manifest.json
changed_files.json
reference_v5_call_graph.json
synthetic_reference_validation.json
d1_contract_validation.json
d5_contract_validation.json
d6_authority_receipt.json
maxT_contract.json
maxT_schedule_manifest.json
maxT_synthetic_validation.json
readiness_matrix.csv
test_matrix.csv
test_results.json
progress.json
summary.json
routing_decision.json
report.md
hashes.sha256
```

Write `hashes.sha256` last and include every top-level output except itself.

TERMINAL ROUTES:

Success:

```text
reference_multiplicity_machinery_repair_complete_pending_independent_acceptance
```

Genuine owner blocker:

```text
reference_multiplicity_machinery_owner_blocked
```

Technical failure after exhausting the authorized self-repair loop:

```text
reference_multiplicity_machinery_repair_failed
```

On success, the sole next allowed phase is a separately written, fresh-agent
independent acceptance Goal. Do not start it here.

FORBIDDEN:

- campaign fitting or scoring;
- campaign economic replay;
- real null, heuristic, D1, or maxT execution on campaign results;
- G1-G8 aggregation;
- ranking or selection;
- seed 45 or G9;
- protected holdout or sealed recorder evidence;
- learned exits;
- broker/API calls, paper submit, paid downloads, promotion/default changes,
  runtime flags, launchd, or real-money paths;
- edits outside the authorized code/test files and this output directory; and
- reinterpretation of old H0-H3 economics.

HIGHEST ALLOWED CLAIM:

> Stage-1 reference and multiplicity machinery repair complete; independent
> acceptance is still required.

