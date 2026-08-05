# Protocol101 Goal Prompt: FT1 Entry Campaign Preregistration And Runner Readiness

Run this Goal in a fresh Codex task or subagent.

---

GOAL ID:

`FT1-ENTRY-CAMPAIGN-PREREGISTRATION-AND-RUNNER-READINESS`

OBJECTIVE:

Freeze the independently accepted Protocol101 Stage-1 repair machinery into a
fresh Full Trader entry-campaign contract, preregister the complete H0-H3
campaign before any new economic result exists, and determine whether the
current exact-contract HGB runner and gate path are ready to execute that
campaign under simulator v5.

This Goal owns campaign specification and runner-readiness inspection only. It
must not fit or reuse a research model, select a threshold, replay campaign
economics, rebuild a null or heuristic, aggregate gates, rank a candidate,
spend seed 45, inspect the protected holdout, or start learned-exit work.

WORKSPACE AND INTERPRETER:

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
```

READ BEFORE ACTING:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md`
- `v4/docs/protocol101/training/goals/PROTOCOL101_S1_REGIMEN_REPAIR_MACHINERY_INDEPENDENT_ACCEPTANCE_GOAL_2026_07_26.md`
- every top-level acceptance artifact under
  `v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/`
- `v4/scripts/protocol101_training_scope.py`
- `v4/model/protocol101_canonical_stage1_contract.py`
- `v4/model/protocol101_scoped_stage1_hgb.py`
- `v4/model/protocol101_serial_simulator_v5.py`
- `v4/scripts/run_protocol101_scoped_stage1_hgb_runner.py`
- `v4/scripts/run_protocol101_scoped_stage1_hgb_runner_v2.py`
- `v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py`

AUTHORITY ORDER:

When older documents or code mention simulator v4, the owner-signed
2026-07-26 regimen-repair amendment supersedes that simulator clause.
Simulator v5 and the signed two-clock contract are mandatory for every new
campaign calculation.

The G4 and G8 signed revisions supersede their older rows. Campaign
multiplicity is a hard eligibility control.

Do not silently edit a signed document. Record any remaining conflict in the
readiness matrix and fail closed.

FROZEN INDEPENDENT ACCEPTANCE:

Require these exact SHA-256 values:

```text
8e73f3415547911804b53cff3d1de2afa2fee733af2b542f9a562155298d8e0d
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/acceptance_decision.json

8f226458c0ded481c7fcb9d99edfc234acd95bdd72030cb73b7c779e8ccd53a8
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/summary.json

bf3024a8397f82fa87c02cc0bc73d1a38bd8884c1446ae22554b4b4b8ab7ad1c
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/hashes.sha256

cf2bfab784a8b596203f9f1799aa99496894c854a77c23b9ef4fffe88fbcc783
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/independent_oracle_manifest.json

ead9c772dba55c2973a668cb5a3140c42526155517eca2c12acbe788ecba4e2b
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/full_corpus_non_economic_validation.json

dba92797a07223d7ddc51056b3254e9eff9231f7a2d65fb9c26ae5129ee5458a
  v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/production_regression_results.json
```

Require:

```text
terminal route: repair_machinery_independently_accepted
INDEP-CROSS-001: PASS
BOUNDARY-NO-SEALED-001: PASS
sessions: 271
policy cells: 28,602,966
label/clock/identity/policy mismatches: 0
protected holdout read: false
seed 45 read or execution: false
broker or paper action: false
```

Verify the acceptance checksum manifest and every accepted production-source
hash before freezing the machinery. If any frozen input changed without a new
owner-authorized acceptance attempt, stop with
`fresh_entry_campaign_preregistration_blocked_input_conflict`.

FROZEN SIGNED CONTRACT HASHES:

```text
3dbc1cf45e2200b7fd789c92be7e927c714b476b4b45aec95b5e0d9686c1ed66
  v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md

5c5a44e6bb4053276ef60b710a3788d1ee5ed9930a0aa8f10a4f45bbbe31417d
  v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md

d3fc327cdf82159f73388648c15643f42c2207a420c3cdab9c4fb08cc302cbc0
  v4/docs/protocol101/training/contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md

33405421b03c89b08ba5473259a8bce2f04a221c08bbb7230bc9ec442fe6c12a
  v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md

ff554ef5dd34086cd09954b906bbb5ce8455fa467776ae83906b01f80d565da5
  v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md

ac2cdbd9dabf8daa36aa53b8736d9691b8569b6f5b2b810d5a56707702465bf3
  v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md

b02a99281b502434675c3440e9f214fc88b9ac974359e20bb7841e19a2a8065b
  v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md
```

OUTPUT DIRECTORY:

```text
v4/audit/autoresearch/
protocol101_full_trader_entry_campaign_preregistration_attempt001/
```

This output directory is the only location where this Goal may write audit
code, temporary files, or result artifacts. Do not change production code,
tests, signed contracts, existing campaign artifacts, runtime state, or
documentation.

PREREGISTER BEFORE READINESS VERDICT:

Before tracing runner behavior or reading old campaign result values, write and
hash:

```text
preregistration.json
source_inventory.json
accepted_machinery_freeze.json
campaign_contract.json
progress.json
```

Reading source structure, signatures, schemas, paths, hashes, and row counts is
allowed before preregistration. Do not inspect old H0-H3 PnL, gate, ranking, or
selection values. The previous 420 units are benchmark provenance only.

FRESH CAMPAIGN CONTRACT:

Freeze all of the following in `campaign_contract.json`.

## Product and scope

```text
campaign namespace: protocol101_full_trader_stage1_entry_fresh_attempt001
entry contract: protocol101-scoped-canonical-stage1-v1
model family: bounded HistGradientBoostingRegressor
target: fee-adjusted payoff / return-on-premium
starting cash: $10,000
position size: one contract
account count: one
daily stop: 5% of session-starting equity
forced flat: 15:55 ET
primary round-trip fee: $3.00
fee sensitivities: $2.60 and $4.00
simulator: exact independently accepted simulator v5
exit semantics: owner-signed two-clock realized-exit contract
```

The campaign trains entry scorers. The seven fixed exits are measurement
scaffolds, not the final learned lifecycle. No entry-only candidate may be
called paper-ready.

## Hypotheses and model-facing features

Freeze the source-defined feature order from
`v4/model/protocol101_canonical_stage1_contract.py`:

```text
H0: exact 12 synchronized non-VIX context features
H1: H0 plus the exact three near-ATM D composites
H2: H0 plus internally recomputed E.bs.delta and E.bs.gamma
H3: all authorized 17 features
```

Require exact ordered feature names and their source hash. No generic market
window, option-ladder tensor, direct per-slot option-price path, VIX change,
internal IV, raw quote microstructure, volume/OI, vendor Greek, future path, or
label field may enter model alpha.

## Corpus and folds

```text
accepted registry sessions: 301
campaign sessions after protected holdout exclusion: 271
protected holdout: 30 sessions, 2025-05-16 through 2025-06-30
folds: five chronological expanding-window folds
fold geometry: 45/45, 90/45, 135/45, 180/45, 225/45
embargo: one session per fold
fold governance hash:
  c02c19feafbac7888a2317ddd7ef6753888b2b10a704d230355b622eb350b920
acceptance registry hash:
  6c656cfb2caeaab1d03e78eee39a164f7d169709abeaec5b0c490f1b1e29f0fe
```

Fail closed on any duplicate session, split-role overlap, decision identity,
contract identity, canonical slot identity, path quote identity, or policy-axis
identity before fitting or scoring.

## Registered search

```text
hypotheses: H0, H1, H2, H3
fixed-exit policies: 0 through 6
initial seeds: 42, 43, 44
folds per unit: 5
units per hypothesis: 105
total fresh fitted units: 420
G9 seed: 45, protected and not executable in this campaign
nearby conservative batch: none unless separately preregistered and owner-authorized
```

Every one of the 420 units must be freshly fit under simulator-v5-compatible
machinery. Do not resume, copy, relabel, or reuse any old fitted model. This is
a conservative fresh-fit choice within the signed all-or-nothing reuse rule.

## Training and selection mechanics

Freeze:

- training-only chronological tail for threshold and confidence fitting;
- measured divergence-noise injection at 1.0x for primary training and
  validation;
- 0x, 0.5x, and 2.0x noise as diagnostics only;
- fold-local score-drift epsilon from training/calibration data only;
- `k_action = 2`;
- `k_slot = 2`;
- deterministic score/strike-index/right-index ordering;
- score-independent nearest-ATM fallback when slot confidence is insufficient;
- exact boundary-stable intersection guards;
- pessimistic executable-fill labels for primary evidence;
- mid/fill alternatives as diagnostics only; and
- no validation value influencing fit, epsilon, threshold, confidence map, or
  feature choice.

## Evidence and gates

Freeze the requirement to recompute, under simulator v5 and the exact campaign
contract:

- matched random-selection nulls for all 28 hypothesis-policy rows;
- the fixed heuristic baseline;
- strict serial fold, seed, policy, and pooled metrics;
- fee, fill, noise, drawdown, frequency, concentration, side/time, churn,
  skipped-opportunity, outcome-bucket, harvest, underwater-duration, and
  worst-day diagnostics;
- D1 using eleven complete 30-row blocks, with the trailing 29 rows excluded
  from D1 only;
- D5 under simulator v5 with ordered identities and hashes;
- D6 under signed split-family authority for offline work, with exact
  candidate transfer deferred to no-order shadow;
- G1-G7 as signed hard gates;
- G8 as required report-only;
- synchronized 20,000-replicate five-session moving-block maxT with
  `p_FWER <= 0.05` as a hard campaign-level eligibility control; and
- no G9 until an independently selected candidate earns it.

No old null band, heuristic PnL, v4 serial replay, prior G1-G8 result, or prior
candidate ranking may become fresh campaign evidence.

RUNNER READINESS AUDIT:

Trace the actual executable call graph. Comments, function names, dormant v5
helpers, and imports are not proof that the runner uses simulator v5.

Inspect at least:

1. data load and two-clock field propagation;
2. duplicate checks before fit and score;
3. model fit target;
4. calibration replay used to choose thresholds;
5. validation candidate construction;
6. validation strict-serial replay;
7. alternate-fee replay;
8. noise-diagnostic replay;
9. per-unit simulator version and semantics;
10. durable resume and model-hash behavior;
11. artifact manifests and immutable hashes;
12. gate aggregation;
13. null/canary generation;
14. heuristic-baseline generation;
15. D1, D5, D6, and maxT support; and
16. protected seed, holdout, recorder, and broker boundaries.

Explicitly determine whether the public function called by the runner uses the
v5 selection/candidate/replay path for calibration, validation, fee
sensitivity, and noise diagnostics. A dormant `*_v5` helper does not pass.

Create `runner_readiness_matrix.csv` with one row per requirement and:

```text
requirement_id
component
entry_point
observed_call_path
expected_contract
status
blocking
evidence
required_repair
```

Create `runner_call_graph.json` with source paths, symbols, line references,
and source hashes. Create `runner_gap_packet.json` containing only concrete
missing or stale behavior and the narrowest repair scope.

READINESS PASS RULE:

The runner is ready only if every campaign calculation and every gate input is
provably simulator v5/two-clock/identity-safe, all code and artifact hashes
freeze the v5 path, all fresh outputs use a new namespace, old model reuse is
disabled, null and heuristic rebuilds are v5-ready, multiplicity machinery is
specified, and all protected boundaries fail closed.

Dry-run, import, compilation, and no-fit structural tests are allowed.
Disposable model fitting is forbidden in this Goal.

REQUIRED OUTPUTS:

```text
preregistration.json
source_inventory.json
accepted_machinery_freeze.json
campaign_contract.json
runner_readiness_matrix.csv
runner_call_graph.json
runner_gap_packet.json
owner_authorization_requirements.json
summary.json
routing_decision.json
report.md
progress.json
hashes.sha256
```

`owner_authorization_requirements.json` must distinguish:

- decisions already signed and binding;
- implementation work that does not alter owner intent;
- any genuine policy choice requiring a future owner decision; and
- economic campaign execution, which remains unauthorized by this Goal.

TERMINAL ROUTES:

Use exactly one:

```text
fresh_entry_campaign_preregistered_runner_ready_for_independent_acceptance
fresh_entry_campaign_preregistered_runner_repair_required
fresh_entry_campaign_preregistration_blocked_input_conflict
```

The expected honest route is determined by the call graph, not by this prompt.
Do not soften readiness because a partial v5 implementation exists.

ITERATION RULE:

Continue through audit-local schema, path, checksum, and reporting repairs
until the required packet is internally complete. Do not terminate merely
because an output directory is absent, a non-economic adapter needs a local
fix, or a report validator fails.

Stop immediately if:

- a frozen signed or accepted input changed without authority;
- determining readiness would require model fitting or economic replay;
- protected holdout, seed 45, sealed recorder data, or broker access would be
  required; or
- a new owner policy choice is genuinely necessary.

FORBIDDEN:

- model fitting, refitting, reuse, or copying;
- threshold selection or confidence fitting;
- campaign economic replay;
- null, heuristic, or maxT execution;
- old H0-H3 performance inspection;
- gate aggregation, ranking, or candidate selection;
- seed 45;
- protected holdout or sealed recorder evidence;
- learned-exit training;
- broker, paper-submit, paid-data, promotion, runtime, launchd, or real-money
  changes;
- production code or test edits;
- signed-contract edits; and
- naming or starting another Goal before its standalone prompt exists.

HIGHEST ALLOWED CLAIM:

One of:

```text
Fresh Full Trader entry campaign preregistered; runner repair is required
before independent runner acceptance.

Fresh Full Trader entry campaign preregistered; runner is ready for a separate
independent runner acceptance.
```

Do not claim training readiness, candidate eligibility, entry signal,
profitability, Full Trader readiness, paper readiness, or live readiness.

