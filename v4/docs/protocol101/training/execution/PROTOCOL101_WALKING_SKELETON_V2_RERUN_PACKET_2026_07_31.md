# Protocol101 Walking Skeleton V2 — Retrospective, Rerun Design, and Claude Handoff

**STATUS: PLANNING AND HANDOFF PACKET — NOT AUTHORIZATION.**

Prepared: 2026-07-31  
Audience: owner, Claude/Fable, Codex, independent reviewers  
Scope: a second quarantined end-to-end dry run of the entry-model and
lifecycle-model building process, followed by combined historical replay and
owner-facing visualization. This packet authorizes no fitting, paid download,
protected-data access, broker contact, recorder activation, paper order,
runtime/default change, or promotion.

Binding authority remains the signed Protocol101 contracts and Graph V2. If
this packet conflicts with them, the authority wins and the conflict must be
resolved before execution. This packet must not become a competing project
status page.

---

## 1. Owner request and intended outcome

The first Walking Skeleton found valuable defects and exercised a large part of
the offline pipe, but it drifted away from its original promise. The real entry
composer never bought a contract. A quarantined forced-BUY wrapper then allowed
the downstream code to run. That was legitimate branch-coverage evidence, but
it was not a learned end-to-end trader and cannot justify the originally
planned paper-trading stage.

The owner wants a second run that preserves the spirit of the exercise:

> Build a small, quarantined version of the trader through the same model,
> data, training, calibration, composer, lifecycle, replay, and evidence paths
> intended for the real campaign. Observe honestly whether it learns to WAIT,
> BUY an exact contract, HOLD, and EXIT. Record what happens and what it teaches
> us at every phase. Do not convert a model-path failure into a pass by forcing
> activity, weakening a gate, changing a target after seeing results, or
> silently substituting a simpler policy.

The desired deliverables are therefore two things:

1. A trustworthy second Walking Skeleton result.
2. A versioned set of evidence-backed changes and open questions for the real
   entry and lifecycle training specifications.

Success does **not** mean the throwaway model is profitable or promotable.
Success means the real intended construction process ran faithfully, every
model-path conclusion is supported by evidence, and any failure remains visible
instead of being routed around.

---

## 2. Executive recommendation

Run Walking Skeleton V2 as two deliberately separate lanes:

### Lane A — autonomous model-fidelity lane

This is the Walking Skeleton proper. It uses the intended learned flat-state
and open-state policies, with the real targets, action spaces, calibration,
composer, safety masks, simulator, and evidence rules. It contains no injected
BUY, threshold-zero EXIT, altered floor, fixed direction, fixed strike, or
other activity canary.

Lane A may end in `ABSTAINS`, `INSUFFICIENT_EVIDENCE`, `SCIENTIFIC_NEGATIVE`,
`MECHANICAL_FAILURE`, or `MODEL_PATH_PASS`. Only `MODEL_PATH_PASS` may satisfy
the original Stages 1–3 model-path intent or be considered for a later live
shadow/paper-readiness discussion.

### Lane B — deterministic plumbing-canary lane

Lane B exists only to exercise branches that Lane A did not naturally reach.
It may inject a deterministic intent or force a branch under an explicitly
named test configuration. Its artifacts, reports, and status are separate.

Lane B can prove that a seam is callable and that safety controls work. It
cannot:

- change Lane A's result;
- unlock the next scientific stage;
- support a model-quality or alpha claim;
- be presented as activity by the learned policy; or
- make the model eligible for live paper trading.

This two-lane structure removes the contradiction that damaged the first run:
the model no longer has to be both honest and artificially active in order for
the engineering branches to be testable.

### Recommended stopping point

Complete V2 through combined owned-data replay and visualization, then stop for
owner and independent review. Do not schedule paper orders as part of the same
goal. A subsequent no-order live shadow is the next honest exam for a Lane-A
`MODEL_PATH_PASS`. A one-contract paper canary, if separately desired, is a
different test with separate authorization and claims.

---

## 3. What Walking Skeleton V1 established

### 3.1 Disposition by stage

| Original stage | Intended evidence | Actual evidence | Honest disposition |
|---|---|---|---|
| Stage 0 | Governed small data slice and frozen miniature-real specification | 13 firewall-safe 1-second sessions after the original owned set proved mostly protected | **Passed after repair** |
| Stage 1 | Learned WAIT and natural BUY of exact contracts | Entry training/inference ran; all 1,432 headline decisions were WAIT | **Engineering path passed; policy acceptance did not** |
| Stage 2 | Frozen learned entry feeding learned lifecycle decisions | 552 real proposals were converted to 78 forced intents outside the real composer | **Quarantined seam coverage passed; autonomous model path untested** |
| Stage 3 | Combined learned trader replay and visual review | Nine forced-entry trades exercised lifecycle, floor, serial replay, and charts | **Plumbing/visualization passed; no alpha or combined-policy claim** |
| Stage 4 | Frozen learned trader causing guarded paper orders | Not started; no Walking Skeleton live adapter/bundle exists | **Not ready as originally defined** |

### 3.2 Data and cost facts

- Only 3 of the 30 originally owned one-second sessions were firewall-safe; 23
  intersected outer test and 4 were protected holdout.
- Eleven clean sessions were acquired, leaving 13 governed one-second sessions.
- The acquisition produced 69,830,302 rows and approximately 1.8 GB at an
  estimated successful-download cost of $10.35.
- It took approximately 66 minutes but only 37 CPU-seconds. Vendor job latency,
  retries, and serial orchestration dominated the elapsed time.
- The real acquisition must select sessions by governed role first and use
  bulk/batch submission rather than serial streaming.

### 3.3 Entry facts

The final V1 entry artifact stack used:

- 45 development sessions;
- 676,620 candidate rows;
- 120,229 eligible candidate rows;
- 16,110 decision rows;
- the authorized 17 causal features;
- 20 fit sessions, 1 embargo session, 20 calibration sessions, and 4 replay
  sessions;
- an HGB-only stack with 39 primary heads, 28 expected-upside support heads,
  three seeds, and 309 fitted component estimators.

The original strict-positive q10 favorable-outcome gate admitted 0 of 4,191
eligible replay contracts. Amendment A6 replaced the outcome-tail positivity
gate with a session-cluster-calibrated lower confidence bound on conditional
mean upside while retaining q10 for ranking. After the amendment:

- 3,545 replay rows passed raw conditional-mean support;
- 2,975 rows across 552 decisions passed the implemented calibrated mean-LCB
  gate;
- all 552 proposals were rejected by each downstream numeric confidence gate;
- the action-conditioned gate was unavailable because it had 19 realized WAIT
  outcomes against a minimum of 50; and
- the final result remained 1,432 WAIT / 0 BUY.

The strongest quantitative finding was the signal-to-error mismatch:

- cluster-gap median approximately 0.015;
- cluster-gap maximum approximately 0.078; and
- calibrated q90 absolute cluster-gap error approximately 0.704.

The current model's measured error was therefore roughly an order of magnitude
larger than its directional separation. That is a real feasibility question for
the future entry campaign, not permission to weaken the confidence gates.

### 3.4 Lifecycle and combined-replay facts

The quarantined V1 lifecycle baseline used:

- 119,769 fit rows across 8 sessions;
- 87,792 calibration rows across 2 sessions;
- 19 causal features;
- an HGB HOLD/EXIT classifier; and
- an operating threshold of approximately 0.5627.

Calibration ROC AUC was approximately 0.837, while average precision was only
approximately 0.066. EXIT is a rare event, so precision-recall, calibration,
session stability, and downstream economic behavior are more informative than
ROC AUC alone.

The full 13-session plumbing pass produced 36 learned exits, 41 floor triggers,
and 1 forced flat. The three-session serial validation accepted nine trades and
reported 31,454 HOLD decisions, 4 learned exits, 4 floor exits, 1 forced flat,
and descriptive after-fee PnL of -$552.

However, the first validation intents deliberately used branch canaries:

- a learned-exit canary set the model threshold to zero;
- a floor canary changed the initial floor fraction from 0.60 to 0.99; and
- a forced-flat canary disabled the learned exit and floor.

Those counts prove control-flow coverage, not natural combined-policy behavior.

### 3.5 Engineering findings worth preserving

- Exact 15:29-pass / 15:30-reject boundary tests are necessary.
- Simulator occupancy time and raw source-event/quote-age time must remain
  distinct, first-class clocks.
- A static HTML/schema check does not replace rendered browser QA; rendering
  found a real multi-session chart-coordinate defect.
- Failed attempts need unique immutable directories and an explicit
  `SUPERSEDED` or `VOID` marker. Ambiguous duplicate directories are an
  evidence-hygiene failure.
- A reviewed stage should be committed and frozen before the next stage starts.

### 3.6 Scientific-language correction

The V1 evidence showed that the former q10 gate was empirically unusable on the
dry-run distribution and mismatched the convex long-option objective. Marginal
q10 values below zero do not, alone, prove that every possible conditional
subpopulation has q10 below zero. Future reports should avoid “no model on any
data can pass” unless an oracle conditional-subset feasibility analysis actually
establishes it.

Likewise, universal rejection by downstream confidence gates proves fail-closed
behavior. It does not yet prove that every gate is well calibrated, jointly
satisfiable, or appropriately powered for the intended real model.

---

## 4. Definition of “miniature but real” for V2

The skeleton should shrink cost and iteration count, not change the trading
problem.

### 4.1 Invariants that may not be reduced or substituted

- Learned flat action space: `WAIT` or `BUY` one exact eligible contract from
  the full governed 42-slot ladder.
- Learned open action space: `HOLD` or `EXIT` for the exact held contract.
- Entry timing, direction, strike, moneyness, and contract choice remain learned.
- Entry uses the same causal history, feature admission, identity continuity,
  masks, labels, multi-horizon targets, composer ordering, calibration, and
  uncertainty rules intended for the real campaign.
- Lifecycle uses the real action-advantage objective and causal position state,
  not a convenient future-best binary label standing in for it.
- Entry fills use executable ask; exits use executable bid; fees, D48, D49,
  one-position serial accounting, daily breaker, cutoff, and forced flat remain
  identical.
- Session roles, embargoes, effective-sample calculations, session-block
  bootstraps, and protected-resource firewalls remain identical in meaning.
- Entry is frozen before lifecycle specialization. Lifecycle evaluation uses
  out-of-fold trajectories produced by that frozen entry policy.
- Natural model actions and canary actions are never combined in one headline
  count.
- The same production-intended tensor builders, trainer interfaces, artifact
  loaders, composer, and simulator code paths are used. Quarantine changes the
  namespace and permissions, not model semantics.

### 4.2 Dimensions that may be scaled down, but must be disclosed

- Number of registered search trials.
- Neural width/depth when the real model family explicitly treats them as
  tunable capacity parameters.
- Epoch count and early-stopping patience.
- Number of seeds, provided at least enough remain to prove seed handling and
  no stability claim is made from one seed.
- Training-session count only when every required role and minimum-evidence
  rule remains meaningful.
- Compute hardware and wall-clock budget.

Every difference belongs in a `fidelity_manifest.json` with:

```text
component
binding_real_spec_reference
real_campaign_value_or_rule
skeleton_value_or_rule
identical | scaled | unresolved
why_the_difference_is_permitted
claim_limited_by_the_difference
```

Any undeclared semantic difference fails the V2 fidelity review.

### 4.3 What “same model” should mean

Graph V2 names a joint temporal full-ladder neural primary with simpler
controls, while D54 requires a simple-model entry feasibility result before
neural/GPU spend. Therefore an HGB-only rerun is not enough to prove the real
trainer path.

Recommended interpretation:

1. Run the required HGB/simple-model feasibility control faithfully.
2. If D54 permits the neural path, train a skeleton-scale configuration through
   the same neural architecture and trainer intended for the real campaign.
3. If D54 does not permit it, a tiny local neural `machinery_smoke_only` may be
   considered only through separate owner authorization. It proves loading,
   forward/backward passes, checkpoint/resume, and artifact packaging; it is not
   a scientific candidate and does not turn D54 into a pass.
4. The lifecycle side must exercise both the transparent HGB baseline and the
   skeleton-scale temporal challenger if both remain in the real FT2-60 design.

If the exact neural entry or lifecycle architecture has not yet been frozen,
V2 Stage 0 must freeze it before anyone can truthfully say the skeleton follows
the real model-building path.

---

## 5. Blocking questions to settle before V2 execution

These are specification gaps, not implementation details.

### Q1 — What is the real lifecycle decision cadence?

The first skeleton used official one-second rows for one-second HOLD/EXIT and
floor decisions. The consolidated authority also requires sub-minute lifecycle
labels/calibration through D58/D59, but several binding sections describe
completed-minute EXIT fills and completed-minute floor checks.

Choose and govern one of these explicitly:

- minute-entry plus minute-lifecycle decisions, with one-second data used only
  for labels, slippage, and floor-crossing calibration; or
- minute-entry plus one-second lifecycle decisions, with matching historical,
  recorder, shadow, and live tensor semantics.

If the owner's intended real trader is one-second lifecycle, amend/reconcile the
authority and verify that the live data and runtime can reproduce it. V2 must
not silently choose.

### Q2 — What exact lifecycle learning target replaces the V1 proxy?

The existing action-advantage foundation says training is blocked until the
target handles:

- executable exit-now value;
- continuation value over multiple horizons;
- slot opportunity cost while the only position is occupied;
- switching/re-entry cost;
- fill, latency, and quote-age uncertainty;
- downside, recovery, giveback, and remaining-tail distributions; and
- causal replacement-opportunity forecasts.

V2 must either implement this governed target or explicitly record which
missing term makes the skeleton not structurally faithful. A simple binary
future EXIT label is not “the real lifecycle model.”

### Q3 — What is the frozen entry primary architecture and loss?

Freeze the actual temporal/full-ladder model class, shared representation,
prediction heads, losses, masking semantics, composer inputs, and skeleton-scale
configuration. Do not let the executor invent these while training.

### Q4 — What data volume makes the required calibration gates available?

The answer must come from a power/minimum-evidence calculation. The first run's
13 one-second sessions and 19 realized WAIT outcomes were inadequate for all
intended gates. Do not pick another “cheap N” and discover afterward that the
evaluation was impossible.

Use all safely available governed minute data when compute permits; reduce
model search rather than discarding statistical information. For one-second
data, first compute the minimum fit/calibration/outer/integrated session counts,
then inventory owned role-safe coverage, estimate acquisition cost, and request
owner authorization if more development data is needed.

### Q5 — What natural activity is required to continue into lifecycle?

Do not tune for a desired trade count. Preregister evidence sufficiency based
on the existing action-conditioned and lifecycle evaluation requirements.
Recommended rule:

- zero or insufficient natural BUYs is an honest Lane-A stop;
- the frozen minimums for realized ENTER and WAIT calibration outcomes must be
  satisfied;
- the frozen entry policy must yield enough out-of-fold selected trajectories,
  across enough independent sessions, to train and evaluate lifecycle without
  row-level pseudoreplication; and
- no canary trajectory may fill an evidence shortfall.

---

## 6. Walking Skeleton V2 stage plan

Every stage ends with a written receipt, learning delta, independent review,
and owner stop. No goal should span multiple review boundaries.

### WS2-0 — Authority reconciliation and fidelity freeze

**Objective:** Make “exactly like the real model-building process” testable.

Freeze:

- the answers to Q1–Q5;
- the exact entry and lifecycle architecture classes;
- target/head definitions and loss functions;
- production-intended code paths;
- the fidelity manifest and only permitted scale reductions;
- the two-lane separation;
- data roles and minimum session/effective-sample requirements;
- stage success, insufficiency, mechanical-failure, and scientific-negative
  outcomes;
- quarantine namespaces and promotion/broker prohibitions; and
- cost and wall-clock caps.

**Required evidence:** line-by-line map from the V2 skeleton configuration to
the consolidated authority, with every difference named.

**Pass:** no unresolved semantic difference.  
**Stop:** any ambiguity about action cadence, target, model, gate, data role, or
runtime feasibility.

### WS2-1 — Joint data-role, power, and acquisition preflight

**Objective:** Ensure every later question is answerable before fitting.

Build one manifest covering:

- entry fit/inner-validation/outer-evaluation roles;
- entry calibration and embargo roles;
- broad lifecycle pretraining roles;
- frozen-entry specialization roles;
- lifecycle calibration and outer-evaluation roles;
- the owned-data sessions with both minute and one-second coverage;
- the protected holdout, confirmation, outer-test, embargo, recorder, and
  sealed exclusions; and
- exact hashes/provenance for every input.

Compute:

- session-level variance and projected effective sample size;
- expected natural trajectory counts under the frozen composer;
- minimum sessions needed for each calibration gate;
- minimum detectable improvement relative to P5 and lifecycle comparators;
- expected storage, CPU/GPU time, vendor time, and download cost; and
- batch acquisition plan if owned one-second data is insufficient.

**Important design choice:** the end-to-end evaluation sessions must have both
entry-resolution and lifecycle-resolution data. Select the intersection at the
start rather than discovering the mismatch after entry training.

**Pass:** all required gates can in principle receive their minimum evidence,
or the packet returns an explicit `INSUFFICIENT_EVIDENCE_DESIGN` before spend.  
**Stop:** no paid acquisition without a separate owner-signed cost cap.

### WS2-2 — Tensor, label, and trainer machinery proof

**Objective:** Prove the real code paths before the substantive fit.

Run a tiny, disposable, non-scientific slice through:

- exact identity-keyed 90-minute entry tensor construction;
- all masks and history availability;
- all multi-horizon path-property labels;
- RLAC/action targets where applicable;
- real entry trainer forward/backward or HGB fit;
- checkpoint, kill, resume, deterministic refit, artifact serialization, and
  inference reload;
- lifecycle row construction at the governed cadence;
- action-advantage label construction; and
- real lifecycle trainer/artifact reload.

Validate no future/path/exit label enters a runtime feature. Validate exact
decision/fill clocks, missing-data behavior, quote age, 15:29/15:30, 15:54/15:55,
D48/D49 equality boundaries, and no-bid semantics.

**Pass:** deterministic mechanics and byte-addressed artifacts.  
**Stop:** repair only the demonstrated mechanical defect; mark the attempt
`VOID_MECHANICAL`, create a new attempt, and do not change scientific settings.

### WS2-3 — Entry feasibility control and scaling curve

**Objective:** Re-run the simple-model requirement faithfully and determine
whether more data plausibly closes the signal-to-error gap.

Use the production entry tensor, exact targets/heads, composer, and serial
simulator. Fit the HGB/simple-model control on preregistered nested session
sizes, for example small/medium/all-available governed development data. The
exact sizes come from WS2-1 and must be fixed before results.

For each size report:

- train and out-of-fold loss by head and horizon;
- calibration residuals and coverage by session;
- raw signal, error bar, and signal-to-error ratio;
- action-conditioned gate sample availability;
- full gate funnel, both marginal and sequential;
- natural WAIT and BUY counts by session;
- exact-contract regret and runner-up separation;
- stability across seeds, phase, side, moneyness, premium band, and era;
- P5 and matched-random comparisons under the same serial game; and
- learning-curve direction with uncertainty, without extrapolating beyond the
  observed points as fact.

**Pass to the primary-model stage:** the actual D54 rule and all applicable
minimum-evidence requirements, not merely “some trades happened.”  
**Stop:** if D54 fails, record whether the result is negative, insufficient, or
blocked by a particular gate. Do not soften the gate. A separately authorized
neural machinery smoke may still test code, but no neural scientific run is
unlocked by narrative.

### WS2-4 — Skeleton-scale real entry primary

**Objective:** Exercise the intended temporal full-ladder entry model and
produce a genuinely frozen entry policy.

Use:

- the same architecture class and shared/full-ladder semantics intended for the
  real campaign;
- the same causal features and admitted feature history;
- the same multi-head targets, losses, masks, composer, calibration, and
  uncertainty logic;
- nested chronological cross-fitting and session-block evidence; and
- a preregistered small search budget rather than an HGB substitute.

No thresholds, features, losses, head weights, horizons, calibration methods,
or trade-count preferences change after results are visible. If the intended
real model is not yet implemented, that is a machinery finding, not authority
to invent a skeleton-only approximation.

**Required entry review packet:** see Section 7.1.  
**Pass:** natural, evidence-sufficient learned entry behavior and the applicable
entry gates. Freeze weights, config, calibrators, composer, code commit, and all
hashes.  
**Stop:** `ABSTAINS`, `SCIENTIFIC_NEGATIVE`, `INSUFFICIENT_EVIDENCE`, or
`MECHANICAL_FAILURE` remains the Lane-A result. Do not enter WS2-5 using forced
trades.

### WS2-5 — Real lifecycle target and broad pretraining

**Objective:** Build the intended lifecycle model, not merely a classifier that
can make an EXIT branch fire.

Entry weights and composer are immutable here. Construct broad eligible causal
position paths and targets containing the governed components of hold advantage:
exit-now value at executable bid, multi-horizon continuation, downside,
recovery, giveback, remaining tail, uncertainty, slot opportunity cost, and
switching/fill costs.

Train the transparent HGB baseline and skeleton-scale temporal challenger on
identical evidence if that remains the real campaign design. Lead evaluation
with EXIT prevalence, precision-recall/AP, calibration, per-session stability,
and economic decision curves. ROC AUC is secondary.

**Pass:** deterministic training/reload and minimum calibration evidence; no
claim yet about the entry-selected distribution.  
**Stop:** any unresolved target component, cadence mismatch, leakage, or
unavailable live feature.

### WS2-6 — Frozen-entry lifecycle specialization and evaluation

**Objective:** Learn and evaluate HOLD/EXIT on trajectories the frozen entry
model naturally creates.

- Generate selected trajectories out of fold. Never train an exit model on
  in-sample entry selections and then evaluate that combination as unseen.
- Specialize the lifecycle model on training-role selected trajectories only.
- Evaluate on disjoint sessions with the entry model, lifecycle model,
  calibrators, floor, and thresholds all frozen.
- Run the entry model's no-action shadow while occupied so opportunity-cost
  inputs are causal.
- Keep the forecast-derived floor upward-only and evaluate floor-on and
  floor-off under identical learned exits.

Compare against exit-immediately, hold-to-flat, the preregistered transparent
time/stop/target set, legacy lifecycle where applicable, and matched-rate random
exits.

**Required lifecycle review packet:** see Section 7.2.  
**Pass:** the binding lifecycle criteria, including pooled and fold behavior,
not merely at least one EXIT.  
**Stop:** too few natural entry trajectories is `INSUFFICIENT_EVIDENCE` for the
combined model. Canary trajectories cannot repair it.

### WS2-7 — Combined serial replay, four-box attribution, and visualization

**Objective:** Determine what the learned pieces actually do together.

Run the frozen Lane-A trader under simulator v5 and the exact same-game rules.
Produce four-box attribution:

- A: control entry × control exit;
- B: learned entry × control exit;
- C: control entry × learned exit; and
- D: learned entry × learned exit.

Report the governed economics and safety evidence, including:

- equity and drawdown;
- fees and fill stress;
- four-bucket distribution;
- loss truncation and large-win capture;
- harvest ratio and floor-on/off ablation;
- MFE giveback, MAE, underwater duration, churn, and time in trade;
- skipped opportunities and slot opportunity cost;
- decisions/trades by side, phase, moneyness, premium, session, and regime;
- all block and forced-flat reasons; and
- natural WAIT/BUY/HOLD/EXIT/floor/forced-flat counts.

Render and visually inspect the equity curve, trades-on-SPX chart, and
four-bucket distribution across multiple sessions. Static checks and browser
rendering are both required.

After Lane A is final, Lane B may run its separate deterministic branch suite
against the same adapters. Its report must be titled “plumbing canary” and may
not be merged into the Lane-A report.

**Pass:** four-box combined evidence and all safety/parity checks satisfy the
preregistered V2 criteria.  
**Stop:** independent review and owner decision. No Stage 4 work in this goal.

---

## 7. Required model-building evidence packets

### 7.1 Entry model report

The entry report must tell a coherent story in this order:

1. **Question:** What entry behavior was the model asked to learn?
2. **Data:** Which independent sessions were used for fit, calibration, embargo,
   and evaluation? What was the effective sample size?
3. **Representation:** What exact live-reproducible information did it receive?
4. **Targets:** What did every head predict and why does it belong to the entry
   decision?
5. **Fit:** Did it learn out of sample, by head/horizon/session, or only reduce
   training loss?
6. **Calibration:** Are mean, q10/q90, regret, and action probabilities calibrated
   for the behavior they control?
7. **Composer funnel:** At each gate, how many decisions entered, passed that
   gate marginally, passed sequentially, and how far were failures from the
   threshold?
8. **Activity:** How many natural WAIT and BUY decisions occurred, across how
   many sessions? Which contracts, sides, phases, and premium bands?
9. **Economics:** How did selected entries compare with P5 and matched controls
   under identical exits and accounting?
10. **Stability:** Did the result survive seeds, folds, eras, and stress?
11. **Conclusion:** Mechanical pass, scientific pass, scientific negative,
    insufficient evidence, or abstention—and the highest allowed claim.
12. **Real-spec delta:** What does this teach the full campaign?

The gate funnel must contain, for every gate:

```text
population_entering
marginal_pass_count_and_rate
sequential_pass_count_and_rate
distinct_sessions
distance_to_threshold_distribution
oracle_or_realized_feasibility_diagnostic
calibration_sample_size_and_unit
primary_failure_reason
```

This would have made the V1 q10 problem and the later signal-to-error problem
clear without several rounds of changing narrative.

### 7.2 Lifecycle model report

The lifecycle report must answer:

1. What causal state and exact held-contract information did the model see?
2. What is the mathematical HOLD-versus-EXIT target, including opportunity and
   switching costs?
3. What is the EXIT prevalence by session and split?
4. What are PR-AUC/AP, precision and recall at the frozen operating point,
   Brier/calibration error, and session-cluster uncertainty?
5. How many natural HOLD and EXIT actions occurred? Keep floor and forced-flat
   actions separate.
6. Did the model truncate losses without destroying large-win capture?
7. How did it compare with every transparent exit baseline?
8. What changed with floor-on versus floor-off?
9. What were realized floor-crossing gaps and quote ages?
10. How much opportunity was skipped while the slot remained occupied?
11. Does behavior remain stable by session, phase, side, moneyness, premium,
    PnL state, and time held?
12. What is the highest allowed claim and the real-spec delta?

### 7.3 Combined report

The combined report must distinguish:

- model-produced behavior;
- hard safety behavior;
- floor-composer behavior;
- Lane-B canary behavior; and
- simulator rejection/block behavior.

No headline action or PnL table may mix these categories.

---

## 8. Learning capture and messaging protocol

### 8.1 Stage card

Every stage produces a one-page `stage_report.md` beginning with:

```text
Stage:
Question asked:
Expected before run:
Observed:
Engineering coverage: PASS | FAIL | PARTIAL | NOT_REACHED
Policy acceptance: PASS | FAIL | INSUFFICIENT | NOT_APPLICABLE
Natural action counts:
Canary action counts (separate):
What we learned:
Alternative explanations still open:
What changed during this attempt: NOTHING or exact mechanical repair reference
Effect on next skeleton stage:
Effect on real-model specification:
Highest allowed claim:
Stop boundary:
```

This two-axis status prevents “stage complete” from hiding that plumbing ran
while the learned policy did not.

### 8.2 Interpretation discipline

Every snag must be classified before a remedy is discussed:

- **Mechanical defect:** code does not implement the frozen contract. Abort,
  repair only that defect, create a new attempt, and rerun from the affected
  boundary.
- **Scientific negative:** machinery is valid and the model fails the frozen
  criterion. Record it; do not retune the criterion.
- **Insufficient evidence:** the question cannot be answered at the required
  confidence. Record the exact missing evidence and power implication.
- **Governance/specification conflict:** two intended rules disagree or the
  required behavior is undefined. Stop for owner/authority resolution.
- **Expected abstention:** WAIT is the correct output under the frozen policy,
  but it may still prevent downstream policy acceptance.
- **Canary-only coverage:** a forced branch worked; no scientific meaning.

An interpretation is `PROVISIONAL` until the diagnostic packet rules out the
nearest competing explanations. Words such as “structural,” “airtight,”
“definitive,” or “any model” require a named proof, not a persuasive story.

### 8.3 Attempt and handoff discipline

- One Codex goal per stage and one independent review before the next.
- One immutable attempt directory per run; never overwrite or create a
  space-suffixed duplicate.
- A failed attempt remains present with its status and superseding attempt.
- Freeze/commit reviewed code and manifests before the next stage.
- Handoffs contain only: authority hash, attempt ID, stage status, passed and
  failed criteria, unresolved questions, exact next authorized action, and stop
  boundary. Do not paste the entire history into every new task.
- The existing Walking Skeleton learnings ledger remains the methodology
  ledger. Conversation memory is a convenience, not authority.

### 8.4 Same-day learning disposition

Every finding receives one of:

- `DESIGN_CHANGE_PROPOSED`
- `PROCESS_CHANGE_PROPOSED`
- `VALIDATED_KEEP`
- `WATCH_AT_NAMED_GATE`
- `ACCEPTED_LIMITATION`
- `NO_CHANGE`

A design change is not “closed” until it has an authority path, checker,
independent review, and required owner signature. A process change must be
referenced by the real-campaign goal that will inherit it.

---

## 9. Seed register for the real-model specifications

The following V1 findings should enter the real-specification delta register.
They are not all adopted design changes.

| ID | Finding from V1 | Real entry/lifecycle implication | Required disposition |
|---|---|---|---|
| RSD-01 | Outcome-tail q10 positivity was empirically nonviable on the dry-run slice | Keep the A6 conditional-mean LCB gate and validate conditional feasibility without universal overclaim | Verify in V2 entry funnel |
| RSD-02 | Entry signal separation was far smaller than calibrated error | Measure session-count/model-capacity scaling before expensive search | Resolve at D54/MDE gate |
| RSD-03 | Action-conditioned gate had only 19 WAIT outcomes | Size data roles before fit so calibration can exist | WS2-1 power requirement |
| RSD-04 | Hundreds of thousands of rows came from few independent sessions | Session count and session bootstrap lead every evidence table | Permanent statistics rule |
| RSD-05 | Lifecycle AP was 0.066 despite ROC AUC 0.837 | Lead with PR/calibration and economic utility for rare EXIT actions | FT2-60/74 requirement |
| RSD-06 | V1 lifecycle used a convenient binary HOLD/EXIT baseline | Implement the governed multi-horizon action-advantage objective including slot/switching costs | Resolve before WS2-5 |
| RSD-07 | One-second V1 lifecycle behavior and completed-minute authority language differ | Freeze the real lifecycle decision cadence and same-game live path | Owner/authority decision before WS2-0 pass |
| RSD-08 | Simulator needed occupancy and source-event clocks | Make decision, occupancy/fill, source-event time, and quote age first-class | Tensor/runtime schema change proposal |
| RSD-09 | Floor action coverage was partly canary-shaped | Separate natural exit, floor, forced-flat, and canary results | Permanent reporting rule |
| RSD-10 | Static chart validation missed a real defect | Require rendered multi-session visual QA for D60 | Permanent process rule |
| RSD-11 | Role-blind “recent N” selected protected data | Data acquisition begins from governed roles and firewall proof | Permanent acquisition rule |
| RSD-12 | Vendor latency dominated one-second acquisition | Use batch-first acquisition with estimates and resumable manifests | Real backfill process |
| RSD-13 | Exact equality at the 15:30 cutoff was initially wrong | Test both sides and equality for every time/risk boundary | Permanent test rule |
| RSD-14 | Forced entry allowed later stages to be called complete | Track engineering coverage and policy acceptance separately | Permanent status rule |

V2 appends new rows rather than rewriting these. Each final row must cite an
artifact, state confidence, identify the affected authority section, and name
the owner gate where adoption will occur.

---

## 10. Criteria for feeling good about the second run

The second run should be considered trustworthy when all of these are true:

1. We can point from every skeleton component to the corresponding real-model
   component and explain every permitted scale reduction.
2. Entry and lifecycle use their actual intended objectives and action spaces.
3. The learned policy, not a wrapper, produces every action credited to Lane A.
4. A model that abstains or fails remains a visible result and stops the
   dependent scientific path.
5. Canary coverage is still available, but lives in a separate lane and cannot
   alter model status.
6. Every phase explains what was expected, what happened, why we think it
   happened, what remains uncertain, and what changes for the real campaign.
7. Gate feasibility, calibration evidence, and effective sample size are known
   before the expensive fit.
8. Entry and exit are evaluated separately and through four-box combined
   attribution.
9. No threshold, label, feature, cadence, or success definition changes after
   results are visible without a new governed attempt.
10. The reviewed implementation and artifacts are frozen before any live work.

If Lane A ends without natural BUYs, the run can still be a successful
experiment and a failed model-path acceptance. That result should feel honest,
not incomplete. It tells us not to spend on lifecycle specialization or paper
deployment until the entry feasibility problem is resolved.

If Lane A naturally buys but the lifecycle model fails, the entry artifact
remains frozen and the result identifies the lifecycle objective, evidence, or
model as the next research problem. Entry is not quietly retrained to make the
exit look better.

---

## 11. Paper-readiness boundary after V2

Even a V2 `MODEL_PATH_PASS` earns only eligibility for paper-readiness
validation. Before any guarded paper order, the project still needs:

- a frozen candidate bundle with weights, calibrators, composer, floor,
  schemas, source hashes, code commit, and independent receipt;
- a production runtime adapter using the same tensor and action semantics;
- historical-versus-runtime feature and action transfer evidence;
- measured artifact-load, feature-build, and inference latency/memory;
- no-order live shadow with reconstructable WAIT/BUY/HOLD/EXIT/floor behavior;
- stale-data, reconnect, incomplete-ladder, cutoff, breaker, and forced-flat
  tests;
- canonical README/Graph state reconciled to the actual approved exception;
  and
- explicit owner authorization for recorder/broker/paper activity.

A Lane-B canary can later test an order seam, but it cannot make a Lane-A
abstaining model paper-ready.

---

## 12. Requested Claude response

Claude should treat this packet as a review brief, not an execution prompt.
Please return:

1. A factual correction list, if any, tied to current artifacts or authority.
2. A decision proposal for Q1–Q5, highlighting every item that requires owner
   choice or an authority amendment.
3. A proposed WS2-0 fidelity contract with exact entry and lifecycle model
   definitions, real-versus-skeleton differences, role minimums, and pass/stop
   outcomes.
4. A proposed stage-by-stage Goal sequence that preserves maker/checker
   separation and stops after each stage.
5. A list of which V1 learnings should be folded into the binding real-model
   training specifications now, which should remain watch-items, and which need
   V2 evidence first.
6. A concise owner memo explaining what V2 will cost, what it can prove, what it
   cannot prove, and the decisions required before it begins.

Claude must not begin fitting, download data, contact IBKR, enable the recorder,
change runtime/default state, access protected resources, or alter a binding
threshold while responding to this packet.

---

## 13. Evidence references

- `v4/docs/protocol101/training/README.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_WALKING_SKELETON_DRYRUN_PLAN_2026_07_30.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_FT2_10_ENTRY_GATE_CONVEXITY_AMENDMENT_PROPOSAL_2026_07_30.md`
- `v4/docs/protocol101/training/research/PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md`
- `v4/audit/autoresearch/protocol101_walking_skeleton_stage1/receipt.json`
- `v4/audit/autoresearch/protocol101_walking_skeleton_stage1/diagnostic_packet.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_gate_convexity_amendment/codex_review.md`
- `v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3/receipt.json`
- `v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3/report.md`
- `v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3/delta_scoped_review.json`

