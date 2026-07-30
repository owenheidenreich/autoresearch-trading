# Goal: Protocol101 FT2 Stage-0 P5 HOLD/EXIT Feasibility

## Objective

Determine whether a simple causal lifecycle model shows enough preliminary
HOLD/EXIT skill on frozen deterministic P5 entries to justify a full governed
learned-lifecycle campaign.

This is an exploratory, non-promotable Stage-0 pilot. It is not the Full
Trader, cannot select or promote a model, and cannot authorize G9, protected
holdout access, live shadow, paper trading, or a full Stage-2 campaign.

The scientific question is:

> Given the exact trades that simulator-v5 deterministic P5 would open, can a
> causal HGB model improve how those same positions are exited, compared with
> P5's original fixed exit, the best fixed exit chosen on training data, and a
> feature-independent random-exit policy?

P5 remains responsible for entry timing, call/put direction, and nearest-ATM
contract selection. This Goal may change only the exit decision within each
frozen position episode.

## Owner Decision And Scope

The owner accepted the routing decision on 2026-07-28:

- Do not scale M0/M1 into the full contract-selector campaign.
- Keep deterministic P5 as the standardized entry baseline.
- Proceed to carefully isolated HOLD/EXIT feasibility research.

That decision authorizes only this offline exploratory pilot. The unsigned
Full Trader learned-lifecycle contract remains unsigned and a full lifecycle
campaign still requires an explicit owner decision.

## Why This Is The Next Step

Stage-0 contract-selector attempt002 found:

- M0 and M1 beat exact random contract selection;
- neither beat deterministic P5 contract selection; and
- every identity, feature-firewall, weighting, target, and risk-set check
  passed.

Therefore scaling those selector hypotheses is not justified. The useful
entry control already available is deterministic P5. The next separable
question is whether causal lifecycle state can improve P5's exits.

## Frozen Authorities

Hash and record these before constructing lifecycle rows:

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md`
- `v4/audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/acceptance_decision.json`
- `v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002/decision.json`
- `v4/audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002/preregistration.json`
- `v4/model/protocol101_stage1_reference_multiplicity.py`
- `v4/model/protocol101_serial_simulator_v5.py`
- The governed fold, corpus, feature, fee, and identity authorities hashed by
  selector Stage-0 attempt002.

The historical action-advantage foundation may be read as research context:

- `v4/docs/protocol101/training/research/PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md`

It is not current evidence, and its future-best labels and old simulator
assumptions must not be reused as the primary target or economic proof.

## Output

Write only to:

`v4/audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt001/`

Mark every result artifact:

`exploratory_non_promotable_stage0`

Repository code and focused tests may be added only where needed to build and
verify this pilot. Do not modify entry, promotion, runtime, launchd, or broker
behavior.

## Step 1: Preregister Before Results

Create and hash `preregistration.json` before fitting, scoring, or inspecting
validation economics.

Freeze:

- The same governed expanding fold 1 used by selector Stage-0 attempt002.
- The same 15 nested chronological training sessions.
- The same 5 later nested validation sessions.
- Zero overlap with final OOF validation, seed 45/G9, protected holdout,
  recorder evidence, sealed evidence, live shadow, or paper results.
- Entry policy: byte-equivalent simulator-v5 deterministic P5.
- Entry timing, side, and contract: unchanged.
- Lifecycle model: one bounded HGB regressor, seed 42.
- Primary target: one-minute executable-bid hold advantage.
- Decision rule: `predicted_hold_advantage > 0` means `HOLD`; otherwise
  `EXIT_NOW`.
- Strong-shuffle seeds: 8700 and 8701 only.
- No threshold optimization.
- No neural or sequence model.
- Ledger-A episode-local evaluation only.
- The terminal routes and pass criteria in this Goal.

Persist the preregistration and SHA-256 before validation results are
computed. If this order cannot be demonstrated, stop with
`stop_mechanical_blocker`.

## Step 2: Freeze Standardized P5 Entry Episodes

Build deterministic P5 references using the accepted simulator-v5 path and
`fixed_heuristic_candidates`.

For each nested role:

1. Replay P5 with simulator v5.
2. Freeze only P5 trades actually admitted by the serial simulator.
3. Preserve entry session, decision timestamp, contract ID, canonical slot,
   entry ask, entry quote timestamp, policy identity, and all source hashes.
4. Preserve the ordered entry list unchanged for every lifecycle comparator.
5. Do not admit an additional entry when an alternative exit releases capital
   earlier.
6. Do not remove a frozen entry when an alternative exit would still be
   holding.

These are independent standardized-entry episodes. Alternative exits may
change the PnL and duration of an episode, but not which episodes exist.
Overlapping wall-clock episodes created by alternative exits are permitted
only in this non-serial isolation ledger and must be reported. They are not
valid Full Trader economics.

Fail closed on:

- duplicate session/decision/contract/slot/episode identities;
- an entry not produced by the byte-equivalent P5 selector;
- an entry not admitted by simulator v5;
- missing entry ask or causal quote timestamp;
- missing or non-monotonic path identities;
- any final OOF, confirmation, holdout, recorder, shadow, or paper session;
  or
- any change in frozen entry identities across comparators.

## Step 3: Build The Causal Lifecycle View

For each frozen P5 episode, build one row per completed minute from entry
through the P5 policy deadline or forced-flat deadline, whichever is earlier.
Use the latest causal executable quote at or before each completed minute.

Allowed model-facing features are:

- elapsed minutes since entry;
- minutes to the frozen deadline;
- entry ask;
- current executable bid, ask, mid, and spread;
- current unrealized dollar PnL and return on entry premium;
- running MFE and MAE through the current minute only;
- giveback from MFE and time since MFE through the current minute only;
- one-, three-, and five-minute causal bid/PnL velocity;
- the signed 12 synchronized non-VIX context features at that minute;
- contract right, canonical slot, offset, moneyness, and premium geometry; and
- approved internally recomputed delta and gamma.

Do not include:

- future quotes, future PnL, future MFE/MAE, or future best/worst values;
- realized future exit timestamp or reason;
- oracle action;
- `q_exit`, `q_hold`, `a_hold`, action advantage, or target aliases;
- any entry-model score or learned selector output;
- vendor Greeks, internal IV, VIX-change features, volume/OI, sizes, quote
  update counts, or sub-minute features; or
- holdout, recorder, shadow, or paper outcomes.

The feature loader must fail closed on aliases and derived forms of forbidden
fields. Missing values remain native HGB missing values; no future-derived
imputation is allowed.

## Step 4: Construct The Frozen Target

The primary training target is deliberately myopic and executable:

```text
q_exit(t)    = fee-adjusted PnL from selling at the executable bid at t
q_hold_1m(t) = fee-adjusted PnL from selling at the executable bid at t+1m
a_hold_1m(t) = q_hold_1m(t) - q_exit(t)
```

Rules:

- Entry uses the frozen P5 entry ask.
- Exit uses executable bid.
- The same round-trip fee is present in both values and is applied exactly
  once to realized episode PnL.
- `t+1m` must be the next completed causal minute for the same contract.
- If the next minute has no valid executable bid, mark the target missing;
  do not bridge the gap or use a later quote.
- During replay, a minute with no valid executable bid cannot execute
  `EXIT_NOW`; record `no_executable_bid_forced_hold` and continue to the next
  completed minute. Missing targets may be excluded from fitting but the
  corresponding episode and replay minute may not be silently removed.
- At the deadline, the only action is `EXIT_NOW`.
- Deadline pricing and occupancy must follow the accepted two-clock contract:
  the latest causal executable bid at or before the deadline prices the exit,
  while occupancy ends at the deadline.
- Do not shorten or backfill paths.
- Future-best or dynamic-programming oracle values may be reported only as
  labeled upper-bound diagnostics. They may not train, tune, route, or gate
  this pilot.

Manually reproduce a deterministic sample of at least 50 targets spanning
calls, puts, early/late minutes, winners, losers, and missing-path cases.

## Step 5: Controls

Evaluate all policies on the identical frozen validation episodes:

1. Original P5 fixed exit.
2. `EXIT_NOW` at the first eligible lifecycle minute.
3. Hold to the frozen deadline.
4. The best preregistered fixed exit chosen using nested training sessions
   only, then frozen before validation.
5. Exact feature-independent random exit.
6. Constant-score HGB behavior.
7. Real HGB, seed 42.
8. Reversed real-HGB decisions.
9. HGB trained under strong shuffle seed 8700.
10. HGB trained under strong shuffle seed 8701.

For exact random exit, estimate the elapsed-minute exit hazard from the real
model's nested-training actions only. Freeze that hazard before validation and
compute exact expected validation PnL over every valid exit minute. Do not use
validation outcomes to fit the random policy.

For each strong shuffle, move complete target sequences across training
episodes using a seeded, group-preserving permutation that:

- never maps an episode to itself;
- keeps right and coarse path-length bucket fixed;
- maps across different sessions;
- preserves elapsed-minute order within the moved sequence; and
- reports every unmatched or truncated row.

Do not row-shuffle lifecycle targets independently.

## Step 6: Fit And Replay The Pilot

Fit one bounded HGB regressor on nested training rows with:

- seed 42;
- maximum depth 3;
- at most 200 boosting iterations;
- learning rate fixed before results;
- no hyperparameter search;
- no validation-based feature changes; and
- episode- and session-balanced training weights so long episodes do not
  dominate merely by producing more minute rows.

### Ledger A: Identical-Entry Exit Isolation

Apply the model causally to each validation episode:

1. Start at the first eligible lifecycle minute.
2. `HOLD` while predicted hold advantage is strictly positive.
3. Exit at the first non-positive prediction.
4. If no earlier exit occurs, force exit at the frozen deadline.
5. Price the exit with the current executable bid and apply the round-trip fee
   exactly once.

Do not admit replacement entries in Ledger A. This ledger attributes exit
quality on identical positions.

### Ledger B: Directional Serial Consequence

Run a second, explicitly labeled directional replay through simulator v5 over
the common ordered P5 entry-intent stream.

For each comparator:

- present the same ordered P5 intents;
- allow the comparator's actual exit time to determine occupancy;
- let simulator v5 skip later P5 intents while the account remains occupied;
- preserve continuous cash, affordability, fees, the session daily stop, one
  contract, and forced flat; and
- report the exact admitted and skipped entry identities.

Ledger B measures whether an apparently better exit gives back its gain by
blocking later P5 opportunities. Because admitted entries can differ after
the first exit-time divergence, it is integration evidence rather than pure
exit attribution.

Do not treat either ledger as promotable Full Trader economics.

## Step 7: Reporting

Report both ledgers per validation session and pooled:

- frozen episode count and lifecycle row count;
- missing-path and invalid-row counts;
- entry-identity equality across all comparators;
- exit count, exit-time distribution, and mean/median holding time;
- fee-adjusted dollar PnL and return on premium;
- improvement versus original P5, training-selected best fixed exit, and
  exact random exit;
- win rate as a diagnostic only;
- mean loss, p95 loss, tail-win capture, giveback, and harvest ratio;
- call/put, time-of-day, premium, and moneyness breakdowns;
- reversed and strong-shuffle results;
- prediction/target rank correlation on validation as a diagnostic;
- validation episode overlap that a real serial replay would need to resolve;
- Ledger-B admitted/skipped P5 entry identities and incremental opportunity
  cost caused by exit-time differences;
- runtime, peak memory, disk use, and full-campaign resource projection; and
- every repair, exclusion, and scientific limitation.

Use session-level paired summaries. Do not treat minute rows as independent
economic observations.

## Step 8: Terminal Decision

Return exactly one:

- `proceed_to_full_lifecycle_contract_signature_and_campaign`
- `stop_no_preliminary_exit_signal`
- `stop_mechanical_blocker`
- `stop_scientific_contract_defect`
- `stop_insufficient_data`

Return `proceed_to_full_lifecycle_contract_signature_and_campaign` only if:

1. Every identity, causality, feature-firewall, weighting, target, and role
   check passes.
2. At least 100 frozen validation episodes and all 5 nested validation
   sessions have usable evaluation evidence.
3. Real HGB has strictly positive pooled fee-adjusted PnL improvement versus
   original P5, the training-selected best fixed exit, and exact random exit
   in both Ledger A and Ledger B.
4. Real HGB improves on original P5 and the training-selected best fixed exit
   on at least 4 of 5 validation sessions in Ledger A, and does not lose to
   either baseline on more than 1 of 5 validation sessions in Ledger B.
5. Real HGB's minimum lift over those three baselines strictly exceeds the
   minimum lift of reversed and both strong-shuffle controls.
6. No result requires changing the frozen target, features, sessions, model
   configuration, controls, or routing criteria.

This is a directional feasibility rule, not statistical acceptance. Passing
means only that a full owner-signed lifecycle campaign is worth building.

If the data contain fewer than 100 usable validation episodes, return
`stop_insufficient_data` without interpreting performance.

## Bounded Repair Loop

Allow up to three repair iterations for mechanical defects only.

Permitted repairs:

- identity joins;
- causal timestamp alignment;
- missing-value plumbing;
- deterministic P5 reconstruction;
- fee application;
- artifact serialization;
- exact-random arithmetic; and
- test or reporting defects.

Forbidden repairs after results are visible:

- changing sessions, target horizon, feature families, HGB configuration,
  decision threshold, controls, pass criteria, or terminal routes;
- adding model capacity;
- removing losing sessions or episodes;
- redefining P5 entries;
- choosing a different fixed baseline from validation results; or
- tuning on validation economics.

Record every repair in `repair_log.json`. If the same blocker persists after
three iterations, stop.

## Required Artifacts

- `preregistration.json`
- `preregistration.sha256`
- `progress.json`
- `frozen_p5_entry_episodes.parquet`
- `entry_identity_receipt.json`
- `lifecycle_rows.parquet`
- `feature_contract.json`
- `feature_firewall_audit.json`
- `target_manual_reproduction.json`
- `machinery_checks.json`
- `model_receipts/`
- `predictions.parquet`
- `episode_replays.parquet`
- `serial_replays.parquet`
- `serial_identity_receipts.json`
- `fixed_exit_results.json`
- `random_exit_results.json`
- `control_results.json`
- `pilot_results.json`
- `resource_projection.json`
- `repair_log.json`
- `decision.json`
- `report.md`
- `hashes.sha256`

## Hard Boundaries

Do not:

- alter P5 entry timing, side, contract choice, or admitted entry list;
- train an entry model or contract selector;
- run more than one real HGB seed or more than two strong shuffles;
- train a neural or sequence model;
- optimize a HOLD/EXIT threshold;
- run the full five-fold lifecycle campaign;
- run any serial economic replay other than the bounded Ledger-B simulator-v5
  comparison over the common P5 intent stream;
- apply Stage-1 G1-G9 as if this were a candidate;
- perform campaign multiplicity or independent final acceptance;
- access seed 45/G9, final OOF results, protected holdout, recorder evidence,
  sealed evidence, live shadow, or paper outcomes;
- select, promote, or freeze a Full Trader candidate;
- contact a broker, submit a paper order, or download paid data; or
- modify promotion, runtime, launchd, recorder, or real-money state.

## Highest Allowed Claim

`Exploratory non-promotable P5 HOLD/EXIT Stage-0 feasibility decision complete.`
