# Protocol101 Goal Prompt: FT1A Entry Runner V5 Core Repair

Run this Goal in a fresh Codex task or subagent.

---

GOAL ID:

`FT1A-ENTRY-RUNNER-V5-CORE-REPAIR`

OBJECTIVE:

Repair the fresh Protocol101 Full Trader entry-campaign runner core so its
actual executable path uses repaired two-clock decisions and the independently
accepted simulator v5 for calibration, primary validation, alternate-fee
replay, and noise diagnostics.

Add fail-closed pre-fit identities, fresh-campaign provenance, v5-only
immutable unit artifacts, hash-verified same-campaign resume, and truthful
no-fit dry-run readiness.

This is a bounded production-machinery repair. It does not rebuild nulls or
heuristics, aggregate gates, implement campaign maxT, perform cross-hypothesis
selection, fit a campaign model, or replay campaign economics.

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
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/`
- every top-level artifact under
  `v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/`
- the current source and tests in the authorized write set below

FROZEN CAMPAIGN INPUTS:

Require these exact SHA-256 values:

```text
40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/preregistration.json

7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/campaign_contract.json

a54e01419e470de30b6fea6a80af5af103ee035040352536411d6bcd8ef2d9f6
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/runner_readiness_matrix.csv

d9373ba0bec3128e25420c70ef36c57adba833e08da357c22c1cde346429471c
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/runner_call_graph.json

eb62bcbf310842932976417de1f27c1882b3d3edfce04b9b6906861d2bcddf98
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/runner_gap_packet.json

056537be6312051420b51390070fe04800fb2046d0371dd05ce7a9384d661c6d
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/routing_decision.json

5a4f5d91489902b135dc39efd0bd50e26346e44dc84e3561291859a9355b3345
  v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/hashes.sha256
```

Require terminal route:

```text
fresh_entry_campaign_preregistered_runner_repair_required
```

Require campaign contract:

```text
namespace: protocol101_full_trader_stage1_entry_fresh_attempt001
contract: protocol101-scoped-canonical-stage1-v1
campaign sessions: 271
folds: 5
policies: 0-6
initial seeds: 42, 43, 44
fresh units: 420
old model reuse: forbidden
seed 45: protected
simulator: exact independently accepted simulator v5
```

Verify the campaign and independent-acceptance checksum manifests before
editing. If a frozen input changed without a new owner-authorized attempt,
stop with `entry_runner_v5_core_repair_blocked`.

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_entry_runner_v5_core_repair_attempt001/
```

AUTHORIZED WRITE SET:

Production:

```text
v4/model/protocol101_scoped_stage1_hgb.py
v4/scripts/run_protocol101_scoped_stage1_hgb_runner.py
v4/scripts/run_protocol101_scoped_stage1_hgb_runner_v2.py
v4/scripts/run_protocol101_full_trader_entry_runner_v5_validation.py
```

Tests:

```text
v4/tests/test_protocol101_scoped_stage1_hgb.py
v4/tests/test_protocol101_scoped_stage1_hgb_runner.py
v4/tests/test_protocol101_scoped_stage1_hgb_runner_v2.py
v4/tests/test_protocol101_full_trader_entry_runner_v5.py
```

Artifacts and audit-local helpers may be written only under the output
directory.

Do not revert unrelated user changes. Do not modify accepted simulator-v5,
label-builder, identity, artifact-writer, governed-loader, feature-contract, or
signed-contract files. Consume their public interfaces.

IMPLEMENTATION CONTRACT:

## 1. Separate the fresh campaign path from historical v4 behavior

Keep historical v4 helpers only where needed for old evidence compatibility,
but make the executable fresh-campaign runner call an explicitly named v5 unit
entry point.

The fresh path must not call:

```text
v4.model.protocol101_serial_simulator.simulate_serial_candidates
legacy selection_rows
legacy replay_candidates
legacy replay_candidates_at_fee
```

Do not rely on imports, comments, or dormant helpers. The real public runner
call graph must terminate in `simulate_serial_candidates_v5` for every replay
rung.

## 2. Carry repaired decisions without expanding model alpha

Fit, calibration, and validation containers must retain the accepted repaired
metadata:

```text
label_realized_exit_time_ns
label_source_exit_quote_time_ns
label_exit_quote_age_ms
label_exit_reason_code
label_executable_exit_bid
label_policy_deadline_ns
label_policy_index
label_invalid_reason_code
canonical_strike_slot
source_quote_time_ns
source_context_time_ns
```

Only the base hypothesis feature matrix may enter HGB. Repaired exit/path
metadata is replay and audit state, never alpha.

The model target remains fee-adjusted payoff / return-on-premium. Do not
replace it with win probability.

## 3. Use v5 for every selection and replay rung

Implement or complete v5 equivalents for:

- threshold sweep on training-tail calibration rows;
- validation candidate construction;
- primary 1.0x validation replay;
- $2.60, $3.00, and $4.00 fee replay;
- 0x, 0.5x, and 2.0x noise diagnostics; and
- per-unit metrics and serial semantics.

All rungs must preserve both clocks. Fee sensitivity must adjust the campaign
round-trip fee exactly once without changing label source values or either
clock.

Threshold, epsilon, confidence map, action dead-band, slot-margin gate,
fallback, and deterministic ordering remain exactly as preregistered.

## 4. Fail closed on identities before fitting or scoring

Use the accepted public identity checks before any model row is loaded,
fitted, scored, hashed, or replayed:

- manifest session uniqueness;
- fold-role uniqueness and no overlap;
- decision uniqueness;
- contract uniqueness;
- canonical slot uniqueness;
- path-quote uniqueness; and
- policy-axis validity.

The runner plan and preregistration must record the identity receipt. A
duplicate must fail before any fitting/scoring function can be entered.

## 5. Bind the runner to the fresh campaign

Pin in the runner plan, preregistration, unit summary, and root summary:

```text
campaign namespace
campaign-contract SHA-256
campaign-preregistration SHA-256
fold-governance SHA-256
acceptance-registry SHA-256
feature-contract ID and source SHA-256
simulator-v5 version and source SHA-256
two-clock schema version
identity-contract version
runner/core source SHA-256 values
```

The default fresh-campaign output path must be a new namespace. It must never
point to old H0-H3 artifact roots.

## 6. Same-campaign resume only

Resume may reuse a completed unit only when:

- it lives under the exact fresh campaign namespace;
- its campaign, preregistration, feature, fold, simulator, schema, source, and
  model hashes match;
- its manifest is complete and v5-only; and
- all referenced unit artifacts verify byte-for-byte.

Any mismatch fails closed. Old 420-model artifacts are not resumable units.
Mixed fresh/old reuse is forbidden.

## 7. Immutable v5-only unit artifacts

Use the accepted artifact writer or equivalent public interface to write a
unit packet with its root manifest last. At minimum preserve:

```text
candidate intents
realized trades
skipped events
fold/session/pooled metrics
exit quote-age report
simulator-v5/two-clock semantics
model artifact hash
source and contract hashes
```

No unit packet may mix v4 and v5 semantics.

## 8. Truthful dry-run readiness

The dry-run must inspect the executable fresh call graph and fail if:

- any fresh replay rung reaches simulator v4;
- repaired decisions are not the public fresh inputs;
- simulator-v5 or campaign hashes are absent;
- old output namespaces or model reuse are possible;
- pre-fit identities are absent;
- immutable v5 unit outputs are unavailable; or
- protected boundaries are not closed.

The dry-run may truthfully report the v5 runner core as repaired while still
listing the deferred evidence/gate-stack blockers. It must not call the entire
campaign ready.

DEFERRED BY DESIGN:

Do not repair these in this Goal:

- random/null/canary reference generation;
- fixed heuristic baseline generation;
- G1-G8 aggregation;
- hard campaign maxT;
- D1, D5, or D6 reporting;
- independent audit;
- cross-hypothesis selection; or
- G9.

Record these as explicit downstream blockers. Their continued existence does
not fail this bounded core-repair Goal.

REQUIRED ACCEPTANCE TESTS:

Add focused tests that prove at least:

1. the fresh public runner calls the v5 unit path;
2. poisoning every simulator-v4 entry point does not break the fresh path;
3. calibration threshold sweeps invoke v5 replay;
4. primary validation invokes v5 replay;
5. every fee sensitivity invokes v5 replay and applies fees once;
6. every noise diagnostic invokes v5 replay;
7. source and realized exit clocks survive selection, replay, and artifacts;
8. realized exits release occupancy before a later eligible entry, including a
   same-timestamp release-before-decision case;
9. model inputs contain only the exact authorized hypothesis features;
10. duplicate manifest/fold/decision/contract/slot/path/policy identities fail
    before fit or score;
11. dry-run detects a deliberately injected v4 edge;
12. dry-run passes the repaired core and reports deferred stack blockers;
13. old model roots cannot resume into the fresh namespace;
14. changed source or model hashes reject resume;
15. committed unit artifacts are v5-only and immutable;
16. seed 45, protected holdout, sealed recorder, and broker paths remain
    inaccessible; and
17. historical v4 regression tests that are outside the fresh path still pass.

Synthetic fixtures, monkeypatches, fake models, and mock score arrays are
allowed. Do not fit a real HGB model or load campaign economic rows.

VALIDATION:

Run:

- `py_compile` for every changed Python file;
- focused tests for all changed runner/core files;
- accepted simulator-v5, regimen-repair, governed-loader, feature-firewall,
  and immutable-artifact regressions;
- preserved simulator-v4 regressions;
- a no-fit fresh-campaign dry-run;
- checksum and output-schema validation; and
- a final process/side-effect check.

Capture exact commands and results.

REQUIRED OUTPUTS:

```text
preregistration.json
source_inventory_before.json
implementation_manifest.json
changed_files.json
test_matrix.csv
test_results.json
fresh_runner_call_graph.json
v5_replay_rung_attestation.json
identity_preflight_attestation.json
resume_and_artifact_attestation.json
dry_run/runner_plan.json
deferred_blockers.json
summary.json
routing_decision.json
report.md
progress.json
hashes.sha256
```

`changed_files.json` must record before/after SHA-256 values and distinguish
pre-existing dirty state from this Goal's changes.

TERMINAL ROUTES:

Use exactly one:

```text
entry_runner_v5_core_repair_complete_pending_independent_acceptance
entry_runner_v5_core_repair_blocked
entry_runner_v5_core_repair_owner_decision_required
```

Pass requires every in-scope acceptance test to pass and every changed source
to be represented in immutable output hashes. Do not claim pass merely because
the code compiles.

ITERATION RULE:

Continue through implementation, tests, and bounded fixes until an in-scope
terminal route is valid. Do not stop merely because an existing test encodes
the stale v4 expectation; update that focused test to the signed v5 contract.

Stop if:

- a required repair would change a signed feature, target, fold, fee, gate,
  policy, identity, two-clock, simulator, or protected-data rule;
- a frozen campaign input changed without authority;
- actual campaign fitting or economic replay would be required;
- a fix must escape the authorized write set; or
- a genuine owner policy choice is required.

FORBIDDEN:

- campaign model fitting, refitting, or reuse;
- campaign threshold selection;
- historical campaign economic replay;
- null, heuristic, gate, maxT, D1, D5, D6, audit, or selection execution;
- old H0-H3 performance inspection;
- seed 45;
- protected holdout or sealed recorder evidence;
- broker, paper-submit, paid-data, promotion, runtime, launchd, or real-money
  work;
- signed-contract edits;
- feature, target, policy, fold, fee, gate, or risk-rule changes; and
- naming or starting another Goal before its standalone prompt exists.

HIGHEST ALLOWED CLAIM:

```text
Fresh entry-runner v5 core repair complete, pending separate independent
acceptance; campaign training remains blocked.
```

Do not claim the evidence/gate stack is repaired, the full campaign runner is
accepted, training is authorized, an entry signal exists, or the Full Trader
is paper-ready.

