# Protocol101 Full Trader Program

> **STALE SYNC DUPLICATE — DO NOT READ (2026-08-05).** This is a Finder/iCloud duplicate of `README.md`, dated 2026-07-28 and superseded by the real `README.md` (2026-07-31) beside it. Despite calling itself the canonical training front page, it is NOT. Current status is [`STATUS.md`](../../../../STATUS.md).


Status: **CANONICAL TRAINING FRONT PAGE**

Last reconciled: **2026-07-28, after policy-neutral selector and P5
HOLD/EXIT Stage-0 routing**

This is the one starting point for Protocol101 model training. It defines the
complete trader, shows what has actually been earned, and names the one next
permitted gate.

This README does not amend a signed contract and does not replace experiment
evidence. Signed rules live in `contracts/`; detailed run artifacts live under
`v4/audit/autoresearch/`.

Do not create another general Protocol101 training index, control center,
status page, or handoff. Update this README after an independently accepted
gate.

## The Product

> One SPXW 0DTE trader that observes the synchronized market and option
> ladder, waits when appropriate, buys one call or put, holds the open
> position, and exits it for the best achievable fee-adjusted result.

The final trader must perform the whole job:

```text
observe market and eligible option ladder
  -> WAIT, BUY ONE CALL, or BUY ONE PUT
  -> while open: HOLD or EXIT-NOW
  -> finish every session flat
```

It uses one account and one contract at a time. It enters at conservative ask
accounting, exits at conservative executable-bid accounting, pays fees, obeys
affordability and daily-risk controls, and may abstain when no trustworthy edge
exists.

An entry-only model is not the finished product and cannot be declared ready
for IBKR paper trading.

## Current State

| Item | Status | Meaning |
|---|---|---|
| Scoped historical/live synchronization | **Passed** | Offline research may use the signed 17-feature entry contract. |
| Stage-1 repair design | **Signed** | Two-clock exits, simulator v5, identity controls, and hard campaign multiplicity are approved. |
| Repair machinery producer run | **Complete** | The implementation passed its producer tests and preserved frozen label PnL exactly. |
| Independent machinery acceptance | **Passed** | A fresh oracle accepted simulator v5, both exit clocks, identities, labels, artifacts, and the 17-feature firewall across all 271 governed sessions. |
| Fresh entry campaign preregistration | **Passed** | The new 420-unit H0-H3 campaign is frozen before results; old fitted models are excluded. |
| Fresh entry runner core producer | **Complete** | The bounded producer now routes repaired decisions through simulator v5 for calibration, validation, fee, and noise replay. |
| Fresh entry runner core acceptance | **Passed** | A separate verifier accepted the repaired call graph, two-clock v5 replay, identities, provenance, resume, and immutable artifacts. |
| Reference and multiplicity machinery | **Passed** | A separate verifier exactly reproduced the v5 references, D1/D5/D6 controls, and all 20,000 maxT schedules/statistics. |
| Fresh H0-H3 entry campaign | **Complete and frozen** | All 420 simulator-v5 HGB units were fitted and independently reproduced. |
| Original campaign audit | **Quarantined for selection** | Five rows showed adjusted signal, but the old D1 negative control asked the wrong absolute-PnL question. |
| Corrected D1 V2 law | **Binding** | Strong-shuffle HGB is compared with feature-independent random-time/random-contract P5 exposure. Absolute PnL is not a gate; `$3/trade` is diagnostic. |
| D1 V2 reaggregation | **Independently verified, insufficient controls** | All 460 consumed source chunks and all 420 models are hash-verified, but the frozen compact controls omit exact risk-set identity receipts; no outcome-aware draw search was allowed and no D1 seed can be certified from those artifacts. |
| Previous H0-H3 economics | **Invalid evidence** | The old simulator held capital until synthetic max hold instead of the realized exit. |
| Fresh 420 H0-H3 fitted models | **Frozen pending corrected D1** | All hashes are unchanged. They may be rescored/replayed without refitting and may become entry candidates only after corrected D1 and candidate-specific incremental-edge gates pass. |
| Policy-neutral contract selector Stage 0 | **Stopped: no preliminary signal** | M0/M1 learned more than random but did not beat deterministic P5 contract choice; the full selector campaign is not justified. |
| P5 HOLD/EXIT Stage 0 | **Stopped: no preliminary signal** | The one-step HGB looked favorable only in overlapping episodes and lost `$4,190` in strict one-account replay. |
| Stage-0 serial attribution | **Both entry and exit are binding** | HGB lost `$2,280` by exiting the five common trades worse and another `$1,910` through ten losing replacement entries. |
| Accepted entry model | **None** | H2/P5's prior unmatched total increment was positive but uncertain; no row has complete strict matched-random incremental evidence. |
| Accepted learned-exit model | **None** | No current lifecycle model has earned the Full Trader contract. |
| Seed 45 / Stage-1 G9 | **Unspent** | Fresh confirmation remains protected. |
| Protected holdout | **Unopened** | It may not be inspected during machinery, entry, or lifecycle development. |
| Paper readiness | **Not earned** | No complete entry-plus-exit candidate has passed transfer and shadow validation. |

The completed producer evidence is:

- [Machinery report](../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/report.md)
- [Machinery summary](../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/summary.json)
- [Full-corpus label validation](../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/non_economic_label_validation.json)
- [Machinery test matrix](../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_attempt001/test_matrix_results.csv)

Independent acceptance evidence:

- [Acceptance decision](../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/acceptance_decision.json)
- [Acceptance report](../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/report.md)
- [Full-corpus independent validation](../../../audit/autoresearch/protocol101_stage1_regimen_repair_machinery_independent_acceptance_attempt001/full_corpus_non_economic_validation.json)

The producer run did not train a model, replay campaign economics, select a
candidate, use the holdout, spend seed 45, or contact a broker.

## The Full Trader Program

### 1. Accept The Machinery

Independently verify simulator v5, the two exit clocks, realized exit reasons,
fee application, account continuity, duplicate rejection, immutable artifacts,
label equivalence, and the exact 17-feature firewall.

Pass freezes the accepted implementation and its hashes. Failure permits only
a bounded repair of the independently identified defect followed by another
independent acceptance run.

### 2. Train Fresh Entry Models

Freshly fit `S1-H0` through `S1-H3` under the accepted simulator-v5 contract.
The seven fixed exit shapes remain measurement scaffolds for entry quality;
they are not the final trader's lifecycle.

Every run uses five chronological expanding folds, one-session embargo, three
initial seeds, the governed non-holdout corpus, and the signed fees, stress,
frequency, and account rules.

### 3. Audit Entry Signal

An independent task applies G1-G7, reports G8, applies the signed campaign
multiplicity control, and reconstructs the selected trades under exact serial
semantics.

This phase answers one question: does the model find genuine entry
opportunities, or only attractive-looking historical noise?

### 4. Confirm Or Route The Entry

- If an entry candidate earns the signed Stage-1 gates, use G9 only through
  the spend-once confirmation path.
- If entry signal is genuine but fixed exits are proven to be the binding
  drawdown or loss-tail problem, route to learned lifecycle research without
  weakening the failed exit-dependent gate.
- If no genuine entry signal exists, stop and redesign features, labels, or
  strategy assumptions. Do not hide the result by adding model capacity.

No route in this phase is paper readiness.

### 5. Train Learned HOLD/EXIT Models

First test exit skill on a separately preregistered standardized-entry policy.
This isolates whether the lifecycle model can improve the same positions
relative to fixed exits.

Then train and test the lifecycle model on the selected learned-entry stream.
Final acceptance always measures the combined entry-plus-exit trader.

The lifecycle model sees only causal state available at that minute. Future
path and action-advantage fields are training answers, never runtime inputs.

### 6. Select The Full Trader

A transparent HGB lifecycle baseline and a neural/sequence challenger use the
same data roles, feature contract, folds, fees, and simulator. Neither model
family wins by preference. Only a hard-gate-eligible candidate may be compared,
and frozen unseen evidence selects the final architecture.

The selected object is the complete system:

```text
entry model
+ entry threshold and abstention
+ contract-selection rule
+ lifecycle model
+ HOLD/EXIT threshold
+ risk and forced-flat rules
+ simulator and feature-contract hashes
```

### 7. Confirm And Freeze The Complete System

Before final confirmation, freeze every model, feature, threshold, fallback,
fee, stress, fold, seed, simulator, and source hash.

The complete system must earn its lifecycle gates, fresh confirmation, and the
one-shot protected holdout under identical one-account semantics. The proposed
rule reserving the holdout for the complete trader becomes binding only when
the learned-lifecycle contract is owner-signed.

### 8. Prove Historical/Live Transfer

After the Full Trader is frozen:

1. Run candidate-specific historical-versus-IBKR decision shadow.
2. Verify entry features, selected contracts, scores, actions, lifecycle
   states, and HOLD/EXIT decisions.
3. Run no-order live shadow with complete reconstruction.
4. Review paper guards, rollback rules, and owner authorization.
5. Only then run guarded IBKR paper orders.

Real-money review is a separate later decision and is never implied by paper
readiness.

## Entry Hypotheses

The fresh campaign preserves the signed questions while fitting new models:

| Hypothesis | Model-facing inputs | Question |
|---|---|---|
| `S1-H0` | 12 synchronized non-VIX SPX context features | Can context alone select entries? |
| `S1-H1` | H0 plus three near-ATM option composites | Do implied-move and skew summaries add edge? |
| `S1-H2` | H0 plus internally calculated delta and gamma | Does deterministic contract geometry add edge? |
| `S1-H3` | H0 plus composites, delta, and gamma | Does the complete authorized 17-feature set improve the smaller subsets? |

The freshly fitted 420 units are frozen campaign evidence. They may become
entry candidates only through corrected D1, candidate-specific incremental
edge, independent selection, and the remaining confirmation gates. Earlier
pre-repair models remain benchmark evidence only.

## Entry Gates In Plain English

| Gate | Question | Role |
|---|---|---|
| G1 | Was the row profitable consistently across folds? | Hard |
| G2 | Did it beat matched random selection? | Hard, with signed campaign multiplicity control |
| G3 | Did learning beat the frozen heuristic? | Hard |
| G4 | Did profit justify drawdown while preserving the account floor? | Hard under the signed revision |
| G5 | Did the worst initial seed still work? | Hard |
| G6 | Did it avoid systematically losing governed eras? | Hard with sample-size reporting |
| G7 | Did it behave like the intended selective day trader? | Hard product gate |
| G8 | Is the confidence readout calibrated? | Required report-only diagnostic until confidence controls behavior |
| G9 | Does a fresh unused seed reproduce G1, G2, and G4? | Spend-once hard confirmation |

No gate may be softened after seeing a candidate fail. Any revision must apply
globally to comparable rows and requires owner approval.

## Model And Label Boundaries

### Entry alpha

Authorized:

- 12 synchronized non-VIX SPX context features;
- three synchronized near-ATM option composites; and
- internally recomputed delta and gamma.

Quarantined from entry alpha:

- direct per-slot option-price paths;
- internal IV and IV expansion/compression;
- VIX-change features;
- raw bid, ask, spread, sizes, and quote age;
- volume and open interest;
- vendor-provided Greeks; and
- future return, PnL, MFE, MAE, or post-entry path fields.

Quarantined quote fields remain available for candidate guards, fills, labels,
PnL, and audit.

### Lifecycle state

The unsigned learned-lifecycle draft proposes causal open-position state:
current executable quote state, elapsed time, unrealized return, MFE and MAE to
date, giveback, price velocity, synchronized context, contract geometry, and
approved internal Greeks.

Future path, future PnL, future best/worst values, oracle actions, and
action-advantage labels are forbidden as model inputs.

## Same-Game Integrity

Historical and IBKR feeds need not be byte-identical. The model must receive
the same causal trading game:

1. Same completed-minute clock.
2. Same ladder and contract identity rules.
3. Same model-facing feature calculations.
4. Same candidate guards and affordability.
5. Same entry and lifecycle action space.
6. Same thresholds, fallback rules, and forced-flat behavior.
7. Same one-account, one-contract risk rules.
8. Same simulator-v5 realized-exit occupancy and PnL semantics.
9. No future or label field entering model alpha.
10. Candidate-specific historical/IBKR shadow agreement after freezing.

## Reward-Hacking Controls

Every campaign must:

- preregister hypotheses, features, labels, folds, seeds, metrics, gates, and
  stopping rules before inspecting results;
- preserve fit, calibration, validation, confirmation, holdout, recorder, and
  shadow roles;
- fit every registered row and report every outcome;
- apply the signed campaign-level multiplicity control;
- use independent gate aggregation and acceptance;
- fail closed on duplicate session, decision, contract, slot, and path
  identities;
- block future/path/label fields from entry and lifecycle inputs;
- use exact simulator-v5 serial replay with one fee application;
- prohibit tuning on G9, holdout, recorder, or live-shadow outcomes; and
- freeze the complete trader before confirmation.

Positive PnL is never sufficient evidence by itself. A result must be
reproducible, unseen, executable, serially valid, and live-transferable.

## Authority

Read these in order:

1. This README.
2. [Trader charter](contracts/PROTOCOL101_TRADER_CHARTER.md).
3. [Scoped synchronization decision](../synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md).
4. [Stage-1 objective and G1-G9](contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md).
5. [G4 and holdout revision](contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md).
6. [G8 calibration revision](contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md).
7. [H0-H3 training design](contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md).
8. [Signed regimen repair amendment](contracts/PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md).
9. [D1 negative-control and incremental-edge amendment](contracts/PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md).
10. [Full Trader learned-lifecycle draft](contracts/PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md).
11. The current bounded work named below.

Historical research remains useful for experiment design, not current proof:

- [Prior campaign distillation](history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md)
- [Protocol farm lineage](history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md)
- [Hold/exit action-advantage foundation](research/PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md)
- [Strategy forensics](research/PROTOCOL101_STRATEGY_FORENSICS_PACKET_V1.md)

## Current Work

The two narrow Stage-0 branches are closed:

- [Contract-selector report](../../../audit/autoresearch/protocol101_policy_neutral_contract_selector_stage0_feasibility_attempt002/report.md)
- [P5 HOLD/EXIT report](../../../audit/autoresearch/protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002/report.md)
- [Serial failure attribution](../../../audit/autoresearch/protocol101_ft2_stage0_serial_failure_attribution_attempt001/report.md)

Do not scale M0/M1 contract selection or the one-step HOLD/EXIT HGB. The
strict-serial result shows that P5 contract choice alone is not a complete
entry policy: when an exit frees the account, P5 emits replacement entries
that can overtrade. The one-step exit target also made the common P5 trades
worse.

The next phase requires an owner architecture decision. The recommended route
is a clean Full Trader redesign that keeps P5 only as a deterministic
contract-choice baseline, learns selective flat-state `WAIT/ENTER` behavior,
and defines open-state `HOLD/EXIT` value with explicit serial opportunity cost.
That redesign must be preregistered before another economic model is fitted.

G9, seed 45, the protected holdout, full lifecycle training, live shadow, and
paper trading remain closed.

## Documentation Rule

Every future Goal names one goal ID, its binding inputs and hashes, the one
gate or decision it owns, its output directory, terminal routes, forbidden
actions, and the next phase it cannot start.

After each independent terminal result, update **Current State** and
**Current Work** here. Do not create another narrative training document.

The documentation migration record is
[MIGRATION_MANIFEST_2026_07_26.md](../MIGRATION_MANIFEST_2026_07_26.md).
