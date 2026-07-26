# Protocol101 Training

Status: **CANONICAL TRAINING FRONT PAGE**
Last reconciled: **2026-07-26, after the Stage-1 adversarial audit**

Start here for Protocol101 model-training work. This page explains the current
state, names the binding contracts, preserves prior alpha, and points to the
one next permitted phase.

This page does not amend a signed contract and does not replace experiment
artifacts. Detailed evidence remains under `v4/audit/autoresearch/`.

Do not create another general Protocol101 training index, status page, control
center, or handoff. Update this README after an independently accepted gate.

## Current Situation

Scoped historical/live synchronization has passed for the authorized
17-feature research contract. Stage 1 then trained four frozen entry-feature
hypotheses, `S1-H0` through `S1-H3`, across seven fixed exit-label policies,
three seeds, and five chronological folds:

```text
4 hypotheses x 7 policies x 3 seeds x 5 folds = 420 model units
```

Those runs produced several positive-looking rows, including `S1-H2/policy 5`.
The completed adversarial audit found that none of the 28 hypothesis-policy
rows is currently trustworthy evidence for selection.

The label builder exited positions at the first stop, target, or deadline, but
the processed artifact did not preserve that realized exit timestamp. The
serial simulator therefore kept capital occupied until entry plus maximum
hold. That changed overlap, buying power, daily stops, trade frequency,
drawdown, and serial PnL.

The terminal route is:

```text
regimen_invalid_redesign_required
```

Consequences:

- synchronization remains passed for the scoped feature boundary;
- the 420 fitted units remain historical evidence, not accepted candidates;
- all positive PnL claims must be rebuilt under repaired serial semantics;
- no candidate is selected;
- seed 45/G9 is unspent;
- the protected holdout is unopened; and
- paper readiness has not been earned.

The next bounded phase is an owner-authorized Stage-1 regimen repair design and
immutable evidence-rebuild specification. It must not select a row, spend seed
45, open the holdout, or contact IBKR.

Primary audit evidence:

- [Audit report](../../../audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/report.md)
- [Audit summary](../../../audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/summary.json)
- [Defect registry](../../../audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/defects.json)
- [Row failure waterfall](../../../audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/row_failure_waterfall.csv)

## Read First

Read these in order for any new training task:

1. This README.
2. [Trader charter](contracts/PROTOCOL101_TRADER_CHARTER.md).
3. [Scoped synchronization decision](../synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md).
4. [Stage-1 objective and G1-G9](contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md).
5. [G4 and holdout revision](contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md).
6. [G8 calibration revision](contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md).
7. [H0-H3 training design](contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md).
8. The current goal or audit specification named in **Current Work**.

Use [prior campaign distillation](history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md)
and [farm lineage](history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md)
when designing a new hypothesis or checking whether an idea was already
falsified. Historical PnL is not current proof.

## Namespaces

The project has reused short names. Always use the qualified form.

| Name | Meaning |
|---|---|
| `S1-H0` through `S1-H3` | Current 2026-07-25 Stage-1 entry-feature hypotheses |
| `APR-H1`, `APR-H2`, `APR-H3a`, etc. | Historical April experiments in the lineage documents |
| `policy 0` through `policy 6` | Current fixed Stage-1 exit-label shapes |
| `Protocol051`, `Protocol081`, etc. | Historical numbered systems, not current policy indices |
| `seed 42/43/44` | Initial training seeds |
| `seed 45` | Spend-once G9 confirmation seed |
| `fold 1-5` | Chronological expanding outer-validation folds |
| `G1-G9` | Current governance and confirmation gates |
| `Stage 1` | Entry scoring against fixed exit-label scaffolds |
| `Stage 2` | Separately governed learned-exit research |

Do not write bare `H2` when historical research is also in scope. Use
`S1-H2`. Do not write bare `P5`; use `policy 5`.

## What Stage 1 Is

At each completed decision minute, the model receives the eligible SPXW 0DTE
ladder and scores each contract's expected fee-adjusted return on premium. It
may:

- wait;
- buy one call; or
- buy one put.

The intended game uses one account, one contract, ask entry, bid exit,
conservative fees and fills, no overlapping positions, affordability, daily
risk controls, and forced flat by the close.

Stage 1 trains entry scorers. Its seven policies are seven fixed definitions
of what happens after entry, not seven dynamically selected strategies.
Learned exits belong to Stage 2 only after repaired entry evidence shows that
entry signal is genuine and fixed exits are the binding limitation.

## Frozen H0-H3 Questions

| Hypothesis | Model-facing inputs | Question |
|---|---|---|
| `S1-H0` | 12 synchronized non-VIX SPX context features | Can context alone select entries? |
| `S1-H1` | H0 plus three near-ATM option composites | Do implied-move and skew summaries add edge? |
| `S1-H2` | H0 plus internally calculated delta and gamma | Does deterministic contract geometry add edge? |
| `S1-H3` | H0 plus composites, delta, and gamma | Does the full authorized 17-feature set improve the smaller subsets? |

These hypotheses are frozen historical experiment definitions. The repaired
regimen must preserve them unless an owner-signed redesign explicitly defines
a new campaign namespace.

## G1-G9 In Plain English

| Gate | Question | Current role |
|---|---|---|
| G1 | Was the row profitable consistently across folds? | Hard |
| G2 | Did it beat matched random selection? | Hard; campaign multiplicity must also be reported |
| G3 | Did learning beat the frozen heuristic? | Hard |
| G4 | Was return sufficient for drawdown while preserving survival? | Hard under the signed revision |
| G5 | Did the worst initial seed still work? | Hard |
| G6 | Did it avoid systematically losing governed eras? | Hard with sample-size reporting |
| G7 | Did it behave like the intended selective day trader? | Hard product/charter gate |
| G8 | Is the confidence readout calibrated? | Required report-only diagnostic until confidence controls behavior |
| G9 | Does fresh seed 45 reproduce G1, G2, and G4? | Spend-once hard confirmation |

No gate may be softened for a row after seeing that row fail. A global owner
amendment applies to every comparable row.

## Same-Game Integrity

Historical and IBKR feeds need not be byte-identical. The fitted model must
receive the same causal decision problem:

1. Same completed-minute clock.
2. Same ladder geometry and contract identity rules.
3. Same model-facing feature calculations.
4. Same candidate guard and affordability semantics.
5. Same action space.
6. Same threshold, score-noise, slot-margin, and fallback rules.
7. Same one-account, one-contract risk game.
8. Future path information used only as the training answer, never as alpha.
9. Candidate-specific historical/IBKR shadow agreement after the candidate is
   frozen.

The scoped synchronization decision supports items 1-8 for offline research.
Item 9 remains downstream and candidate-specific.

## Alpha Boundary

Authorized model-facing families:

- 12 synchronized non-VIX SPX context features;
- three synchronized near-ATM option composites; and
- internally recomputed delta and gamma.

Quarantined from model alpha:

- direct per-slot option-price paths;
- internal IV and IV expansion/compression;
- VIX-change features;
- raw bid, ask, spread, sizes, and quote age;
- volume and open interest;
- vendor-provided Greeks; and
- future return, PnL, MFE, MAE, or post-entry path fields.

Quarantined quote fields remain available for guards, fills, labels, PnL, and
audit. They are excluded from alpha, not erased from the trading game.

## Alpha Preservation

The audit defect invalidates current economic selection. It does not erase
research memory.

| Clue | Current disposition |
|---|---|
| Return-on-premium target | Preserve; still better aligned than raw dollar PnL |
| Patient convex exit shapes | Preserve as a hypothesis; rebuild economics under corrected occupancy |
| Internal delta/gamma | Preserve as synchronized model inputs |
| Near-ATM composites | Preserve as synchronized model inputs |
| Learned exits | Preserve as an architectural prior; Stage 2 requires repaired entry/exit attribution |
| Causal trade-state features | Preserve for a future parity-reviewed Stage 2 |
| Same-side strike ranking | Preserve for repaired within-minute diagnostics |
| Opening context and ES/NQ/A-D context | Unresolved; requires separate parity-gated hypotheses |
| Direct price paths and internal IV expansion | Quarantined pending new evidence |
| Global always-put routing | Falsified as a global default |
| Simple global early-exit rules | Do not retest unchanged |
| Win-rate optimization | Forbidden as the primary objective |

Preserved alpha may motivate a preregistered experiment. It cannot bypass the
current feature contract, corrected serial simulator, CV, G9, holdout, or
candidate-specific transfer.

## Reward-Hacking Controls

Controls that remain mandatory:

- preregister hypotheses, metrics, gates, folds, seeds, and stopping rules;
- preserve fit, calibration, validation, confirmation, and holdout roles;
- block future/path/label fields from model inputs;
- execute all frozen rows rather than reporting only winners;
- use independent gate aggregation and acceptance;
- report campaign-level multiplicity;
- prohibit tuning on G9, holdout, recorder, or shadow results;
- fail closed on duplicate decision/contract identities;
- preserve realized exit timestamp and reason through serial replay;
- freeze selected model, feature, threshold, policy, source, and simulator
  hashes before confirmation.

Open risks:

- repeated use of the same five outer folds;
- 28-row multiple comparison;
- human or agent attention concentrating on positive rows;
- candidate-specific feed transfer;
- simulator or artifact semantics silently diverging from the signed game.

The repaired regimen must make these controls executable, not merely state
them in prose.

## Authority

### Binding contracts

- [Trader charter](contracts/PROTOCOL101_TRADER_CHARTER.md)
- [Stage-1 objective and gates](contracts/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md)
- [G4/holdout revision](contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md)
- [G8 revision](contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md)
- [H0-H3 training design](contracts/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md)
- [Trade-shape menu](contracts/PROTOCOL101_TRADE_SHAPE_MENU_V2_PROPOSAL.md)
- [Stage-2 proposal](contracts/PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md)

### Synchronization authority

- [Scoped synchronization decision](../synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md)
- [Live feature contract](../synchronization/contracts/PROTOCOL101_LIVE_FEATURE_CONTRACT_V1.md)
- [Parity reference](../synchronization/contracts/PROTOCOL101_LIVE_HISTORICAL_PARITY_REFERENCE_2026_06_11.md)

### Execution controls

- [Goal-sized gated system](execution/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md)
- [Stage-1 Autoresearch Graph](execution/PROTOCOL101_STAGE1_AUTORESEARCH_GRAPH_2026_07_25.md)
- [Stage-1 to live sequence](execution/PROTOCOL101_STAGE1_TO_LIVE_EXECUTION_PLAN.md)
- [Canonical serial simulator](execution/PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2.md)
- [Accounting repair record](execution/PROTOCOL101_SERIAL_SIMULATOR_ACCOUNTING_REPAIR_2026_07_25.md)

### Current audit record

- [Adversarial audit specification](audits/PROTOCOL101_STAGE1_ADVERSARIAL_AUDIT_SPEC.md)
- [Executed audit goal](goals/PROTOCOL101_S1_REGIMEN_AUDIT_GOAL_2026_07_26.md)
- [Audit terminal report](../../../audit/autoresearch/protocol101_stage1_training_regimen_adversarial_audit_attempt001/report.md)

### Historical research

- [Prior campaign distillation](history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md)
- [Protocol farm lineage](history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md)
- [Track A foundation](research/PROTOCOL101_TRACK_A_FOUNDATIONAL_TRUTH_V1.md)
- [Strategy forensics](research/PROTOCOL101_STRATEGY_FORENSICS_PACKET_V1.md)
- [Strategy selection packet](research/PROTOCOL101_STRATEGY_SELECTION_PACKET_V1.md)

The original pre-audit regimen narrative is preserved byte-for-byte at
[history/frozen/PROTOCOL101_TRAINING_REGIMEN_AND_INTEGRITY_CONTROL_2026_07_26.md](history/frozen/PROTOCOL101_TRAINING_REGIMEN_AND_INTEGRITY_CONTROL_2026_07_26.md).
It is historical input, not the current status.

## Current Work

No training goal is authorized by this README.

The next goal must own exactly one bounded deliverable:

```text
Stage-1 regimen repair design
and immutable evidence-rebuild specification
```

It must define, before any rebuild:

- preservation of realized exit timestamp and reason;
- exact one-account serial occupancy;
- duplicate identity rejection;
- immutable artifact schema and hashes;
- multiplicity and falsification controls;
- rebuild scope and stopping rules;
- independent acceptance criteria; and
- explicit prohibition on selection, seed 45, holdout, broker, and paper work.

Only after that design is owner-approved may a separate goal implement and
validate the machinery. Rebuilding H0-H3 is a later gate.

## Path From Here

```text
repair design and owner approval
  -> machinery implementation and independent acceptance
  -> immutable H0-H3 evidence rebuild
  -> independent RUN/GATE/AUDIT acceptance
  -> model-free cross-hypothesis selection
  -> one-time seed-45 G9
  -> final candidate fit and freeze
  -> one-shot protected holdout
  -> candidate-specific historical/IBKR shadow transfer
  -> no-order live shadow
  -> guarded IBKR paper review
  -> separate real-money review
```

Nothing after repair design has been earned yet.

## Documentation Rules

Every future Goal must name:

- one qualified goal ID;
- binding input paths and hashes;
- the one gate or decision it owns;
- its output directory;
- terminal routes;
- forbidden actions; and
- the next phase it cannot start.

After independent acceptance, update this README's **Current Situation**,
**Current Work**, and **Path From Here**. Do not create another narrative
index.

The path migration record is
[MIGRATION_MANIFEST_2026_07_26.md](../MIGRATION_MANIFEST_2026_07_26.md).
