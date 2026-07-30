# Protocol101 Goal Prompt: FT1D Fresh Entry Campaign Run To G1-G8

GOAL ID: `FT1D-FRESH-ENTRY-CAMPAIGN-RUN-TO-G1-G8`

## Objective

Execute the owner-authorized fresh Protocol101 Full Trader Stage-1 entry
campaign:

```text
H0, H1, H2, H3
x policies P0-P6
x seeds 42, 43, 44
x folds 1-5
= 420 fresh bounded-HGB fitted units
```

Then build the real immutable campaign packet, recompute the preregistered
simulator-v5 references and controls, run the 20,000-replicate synchronized
maxT control, and produce the model-free G1-G8 aggregation.

This Goal stops before independent audit and cross-hypothesis selection. A
fresh agent must perform those two steps under a separately written Goal.

This is real offline model training. Inspecting and reporting campaign
economics is authorized after the immutable 420-unit run is complete.

## Workspace And Interpreter

```text
workspace: /Users/gduby/Documents/autoresearch-trading
interpreter: ~/.autoresearch-trading/runtime-venv/bin/python
campaign namespace: protocol101_full_trader_stage1_entry_fresh_attempt001
```

## Owner Authorization

The owner, Owen Heidenreich, authorized:

- the owner-approved Option A local controller journal;
- fresh offline H0-H3 training;
- reference, D1, D5, D6, and maxT computation;
- G1-G8 aggregation with G8 report-only; and
- later independent audit and entry selection.

The authorization explicitly excludes seed 45/G9, protected holdout,
HOLD/EXIT training, broker or paper activity, paid downloads, promotion,
runtime/default/launchd changes, and real-money paths.

Verify these external owner records before acting:

```text
v4/audit/autoresearch/
protocol101_full_trader_stage1_trust_boundary_owner_decision_2026_07_26/
owner_decision_packet.md
sha256:
e122d996686e30275075c398dd856d5d6c4b1da9d419b52df1d4d966c45a88be

v4/audit/autoresearch/
protocol101_full_trader_stage1_offline_training_authorization_2026_07_26/
owner_authorization.md
sha256:
a12a21740a7d1113ab35aef24f0c4c747884fc21fa1cdc4061f6bcf707b083c4
```

Also verify the machine-readable owner execution authorization and signed
contract bundle named in the Option A journal genesis. The controller and
graph must not create, broaden, or rewrite owner authorization.

## Binding Accepted Foundation

The Goal may start only if all of these remain exact:

```text
campaign preregistration:
v4/audit/autoresearch/
protocol101_full_trader_entry_campaign_preregistration_attempt001/
preregistration.json
sha256:
40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5

campaign contract:
v4/audit/autoresearch/
protocol101_full_trader_entry_campaign_preregistration_attempt001/
campaign_contract.json
sha256:
7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0

simulator-v5 repair machinery acceptance:
v4/audit/autoresearch/
protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/
acceptance_decision.json
sha256:
8e73f3415547911804b53cff3d1de2afa2fee733af2b542f9a562155298d8e0d
route: repair_machinery_independently_accepted

fresh entry runner-v5 core acceptance:
v4/audit/autoresearch/
protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/
acceptance_decision.json
sha256:
72cbe1443cbddeb5af6dbb09bf384d9ea479658a4cefacbcda5b9c8463600bd2
route: entry_runner_v5_core_independently_accepted

reference and multiplicity acceptance:
v4/audit/autoresearch/
protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/
acceptance_decision.json
sha256:
d6211d8260fe43aadd30037da5d9df373bb53eec2d78ab49866f73dc52b4022f
route: reference_multiplicity_machinery_independently_accepted

Option A controller-journal acceptance:
v4/audit/autoresearch/
protocol101_full_trader_stage1_option_a_controller_journal_independent_acceptance_attempt001/
acceptance_decision.json
sha256:
6fe843f1094b6b21eb1ce4ac44cb5a7a1142f0e1a5db3f43bdf0c9f42ca96bb3
route: option_a_controller_journal_independently_accepted
```

If a binding hash or route has changed, fail closed. Do not silently
re-preregister, modernize, or repair a scientific contract.

## Frozen Trading And ML Contract

Use exactly:

```text
entry contract: protocol101-scoped-canonical-stage1-v1
model: bounded HistGradientBoostingRegressor
target: fee-adjusted payoff / return-on-premium, never win probability
sessions: 271 governed campaign sessions
protected holdout: 30 sessions from 2025-05-16 through 2025-06-30, excluded
folds: 5 chronological expanding-window folds
fold geometry: 45/45, 90/45, 135/45, 180/45, 225/45
embargo: 1 session
fold hash:
c02c19feafbac7888a2317ddd7ef6753888b2b10a704d230355b622eb350b920
registry hash:
6c656cfb2caeaab1d03e78eee39a164f7d169709abeaec5b0c490f1b1e29f0fe
policies: P0-P6, source indices 0-6
seeds: 42, 43, 44
G9 seed 45: forbidden
one account, one contract
starting cash: $10,000
daily stop: 5% of session-starting equity
forced flat: 15:55 ET
primary fee: $3.00 round trip
fee diagnostics: $2.60 and $4.00
primary divergence noise: 1.0x
diagnostic noise: 0x, 0.5x, 2.0x
fills: pessimistic executable
simulator: protocol101_serial_simulator_v5_realized_exit_occupancy
exit semantics: signed two-clock realized-exit contract
G8: required report-only
maxT: synchronized 20,000 replicate, five-session moving blocks,
      hard p_FWER <= 0.05
```

Feature hypotheses are source-defined and ordered by
`v4/model/protocol101_canonical_stage1_contract.py`:

```text
H0: exact 12 synchronized non-VIX context features
H1: H0 plus exact three near-ATM D composites
H2: H0 plus internally recomputed E.bs.delta and E.bs.gamma
H3: all authorized 17 features
```

No direct per-slot option-price paths, VIX changes, internal IV, raw quote
microstructure, volume/OI, vendor Greeks, future path, PnL, exit result, or
label field may enter model alpha.

## Required Output Roots

```text
fitted units:
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_fresh_attempt001/

controller and producer evidence:
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_execution_attempt001/

real campaign evidence and G1-G8:
v4/audit/autoresearch/
protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001/
```

Never overwrite an existing complete artifact. Resume only after verifying
its source, preregistration, model, replay-packet, and parent hashes. Mark
defective artifacts void and preserve them; do not count them as evidence.

Maintain an atomic `progress.json` that includes completed units by
hypothesis, current node, last verified checkpoint, blocker classification,
and all forbidden-action flags. Update it throughout long execution.

## Step 1: Bind The Owner Root And Option A Journal

Before any fit:

1. Verify the exact external machine-readable authorization schema
   `Protocol101Fresh420UnitOwnerExecutionAuthorizationV2`.
2. Require:
   `campaign_namespace=protocol101_full_trader_stage1_entry_fresh_attempt001`,
   `authorized=true`,
   route
   `owner_authorized_fresh_420_unit_campaign_execution`, and
   `seed_45_or_G9_authorized=false`.
3. Verify its `goal_sha256`, campaign preregistration hash, and signed
   contract-bundle hash.
4. Create exactly one owner-only (`0600`) append-only controller journal
   genesis using
   `v4/model/protocol101_stage1_controller_journal.py`.
5. Evaluate
   `v4/scripts/run_protocol101_stage1_autoresearch_graph.py` from that journal
   and require the next node to be `FRESH_420_UNIT_RUN`.

The graph may observe and validate the external authorization. It may never
manufacture authorization.

## Step 2: Narrow Execution Adapter

The accepted scientific runner intentionally remains locked without the
external Option A binding. If no accepted executable controller exists,
implement the smallest durable controller needed to:

- validate the owner authorization and journal genesis;
- call the accepted durable v5 runner for H0-H3;
- preserve the accepted scientific runner unchanged when possible;
- write atomic progress;
- resume only exact same-campaign completed units;
- reject old H0-H3 artifacts and foreign models;
- reject seed 45;
- reject a changed preregistration, contract, code, or parent hash; and
- append a journal checkpoint only after an independently verifiable
  immutable artifact and validator receipt exist.

Preferred production entrypoint:

```text
v4/scripts/run_protocol101_full_trader_stage1_entry_campaign.py
```

Add focused tests for authorization absence/tampering, journal tampering,
wrong campaign, wrong Goal/preregistration/contract hash, old-model reuse,
seed 45, interrupted resume, partial unit output, and duplicate execution.

This adapter is mechanical control plumbing. It may not change model
hyperparameters, features, labels, data membership, folds, policy semantics,
fees, fills, noise, threshold logic, confidence logic, simulator economics,
gates, or selection law.

Before fitting, run the focused tests and a no-fit dry-run. If this narrow
adapter cannot be built without a scientific-contract change, stop and report
the exact owner decision required.

## Step 3: Fit The Fresh 420 Units

Use the accepted durable runner
`v4/scripts/run_protocol101_scoped_stage1_hgb_runner_v2.py` scientific path.

Execute all four hypotheses. Each hypothesis must produce 105 unique units:

```text
7 policies x 3 seeds x 5 folds = 105
```

Each unit must:

- fit a new model;
- use only its hypothesis's ordered features;
- split fit/calibration/validation chronologically;
- derive threshold and confidence only from training/calibration data;
- validate at primary 1.0x measured divergence noise;
- replay through simulator v5 with actual realized occupancy exits;
- preserve one-account serial cash, overlap, daily-stop, and forced-flat law;
- write a hashed model and immutable replay packet; and
- bind the exact campaign, fold, registry, feature, simulator, schema,
  source, model, threshold, epsilon, and code hashes.

Never copy or resume a model from the old
`protocol101_scoped_canonical_stage1_h*_attempt001` directories.

Training completion requires:

```text
H0: 105/105
H1: 105/105
H2: 105/105
H3: 105/105
total: 420/420 fresh models and unit packets
```

After independent mechanical validation of the immutable run artifact, append
the `FRESH_420_UNIT_RUN` journal checkpoint.

## Step 4: Build The Real Campaign Packet

The accepted gate contract currently retains a compatibility schema name
containing `Synthetic`; do not populate the packet from synthetic fixtures.
Build the packet only from the 420 real immutable unit/replay artifacts.

If a production assembler is absent, implement the narrowest assembler plus
focused positive and mutation tests. It must:

- require the exact ordered 420-axis grid;
- independently reload and hash every unit, model, and replay packet;
- convert no diagnostic or future field into model alpha;
- preserve actual candidate identities and simulator-v5 replay economics;
- fail closed on duplicate/missing/reordered units or identities;
- create a sealed campaign packet accepted by
  `v4/model/protocol101_stage1_gate_contract.py`; and
- build/freeze the execution provenance authority from that exact packet.

After independent mechanical validation, append the
`EXECUTION_PROVENANCE_AUTHORITY` journal checkpoint.

## Step 5: Recompute Real References And Controls

Using the real campaign and accepted simulator-v5 machinery, recompute:

- matched random-selection references for all 28 H/P rows;
- the fixed VWAP-side nearest-ATM plus best-single-shape heuristic;
- D1 with eleven complete 30-row blocks and the trailing 29 rows excluded
  from D1 only;
- D5 under simulator v5 with ordered identities and hashes;
- D6 using the signed split-family offline authority;
- synchronized five-session moving-block maxT with exactly 20,000 replicates;
- fee, fill, noise, drawdown, frequency, concentration, side/time, churn,
  skipped-opportunity, worst-day, underwater-duration, outcome-bucket,
  harvest-ratio, daily-breaker, and shape-usage diagnostics.

Do not use stale v4 reference economics. Do not use synthetic references as
real evidence.

Freeze and validate each node before appending its journal checkpoint:

```text
REAL_V5_REFERENCES_D1_D5_D6
CONTROL_AUTHORITY
FROZEN_20000_REPLICATE_MAXT
```

## Step 6: Aggregate G1-G8

Run the accepted producer aggregator against the immutable real campaign,
execution authority, references, controls, and control authority.

Apply:

- G1-G7 exactly as signed;
- G8 report-only exactly as signed;
- maxT `p_FWER <= 0.05` as a hard campaign eligibility control.

Do not alter a threshold after seeing results. Do not choose a candidate.
Write the complete 28-row gate table with plain values for:

- pooled and per-fold fee-adjusted PnL;
- profitable folds;
- random-null z-scores by seed;
- heuristic comparison;
- drawdown;
- frequency;
- era behavior;
- calibration report;
- maxT adjusted p-value;
- eligibility and exact failure reasons.

After immutable validation, append the `G1_G8_AGGREGATION` journal checkpoint.

## Blocker Recovery Loop

Do not abandon this Goal at the first transient or mechanical blocker.

For every blocker, classify it:

```text
transient
mechanical_non_scientific
scientific_contract
protected_action
external_owner_required
```

- Retry transient failures with bounded backoff.
- For `mechanical_non_scientific`, make the narrowest repair, add focused
  tests, invalidate affected defective evidence, and resume from the last
  verified journal checkpoint.
- Do not change scientific/trading rules to make a model pass.
- Stop only when the needed change affects features, labels, data membership,
  folds, policies, model family/hyperparameters, fees, fills, noise,
  simulator economics, gates, protected data, paid data, broker/paper state,
  or owner authorization.

Disk guard: require at least 12 GiB free before fitting and 8 GiB while
running. Do not delete raw data or prior evidence automatically. If space is
insufficient, report the exact shortfall and largest removable non-authority
caches; do not destroy evidence.

## Required Validation

Before terminal success:

- all 420 model and unit hashes verify;
- all 420 immutable replay packets verify;
- no old model hash appears in the fresh campaign;
- exact H/P/seed/fold axes are complete once each;
- simulator v5 and two-clock fields are present everywhere;
- all feature lists equal the signed source-defined lists;
- all session/split/decision/contract/slot/path identities pass;
- campaign packet and authority hashes verify;
- reference/control routes and hashes verify;
- maxT replicate count is exactly 20,000;
- G1-G8 has exactly 28 rows;
- G8 is report-only;
- controller journal is a valid prefix ending at `G1_G8_AGGREGATION`;
- no `INDEPENDENT_AUDIT` or `MODEL_FREE_SELECTION_ROUTING` checkpoint exists;
- no seed 45/G9, holdout, learned exit, recorder/sealed evidence, broker,
  paper, paid download, promotion, runtime, launchd, or real-money action
  occurred.

## Terminal Routes

Success:

```text
fresh_entry_campaign_g1_g8_complete_pending_independent_audit
```

No-signal or gate failures are not execution failure. Complete the packet and
route to independent audit anyway.

Only these are blocking terminal routes:

```text
fresh_entry_campaign_blocked_scientific_contract_owner_decision_required
fresh_entry_campaign_blocked_protected_or_external_action
fresh_entry_campaign_invalid_mechanical_repair_exhausted
```

## Final Response

Report:

- whether real training ran;
- fresh model/unit count;
- completion by H0-H3;
- campaign and journal hashes;
- a compact 28-row or grouped economic summary;
- which G1-G7/maxT gates passed or failed and why;
- G8 diagnostics separately;
- exact side-effect audit;
- terminal route; and
- the separately written next Goal path for fresh independent audit and
  entry selection.

Do not claim an entry model has been selected. Do not start G9 or HOLD/EXIT.

