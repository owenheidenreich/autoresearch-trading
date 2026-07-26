# Goal: Protocol101 Stage-1 Truth And Validity Audit

## Goal ID

`S1-REGIMEN-TRUTH-VALIDITY-AUDIT`

## Goal

Complete a read-only adversarial validity audit of the frozen Protocol101
Stage-1 H0-H3 training evidence under:

```text
v4/docs/PROTOCOL101_STAGE1_ADVERSARIAL_AUDIT_SPEC.md
```

Use:

```text
~/.autoresearch-trading/runtime-venv/bin/python
```

The audit is complete only when it can support one terminal route and
separately answer three questions for every hypothesis-policy row:

1. **Experiment validity:** Can the evidence be trusted given feature timing,
   leakage, CV, simulator, accounting, selection, and multiplicity controls?
2. **Economic evidence:** Are the frozen out-of-sample results negative,
   inconclusive, or consistent with signal before governance gates?
3. **Governance eligibility:** Does the row pass the currently signed global
   rules, independently of whether it appears profitable?

Do not collapse these into one pass/fail result.

## Frozen Boundaries

Use existing H0-H3 models and artifacts as immutable historical evidence.

Do not:

- retrain or retune H0-H3;
- reaggregate across hypotheses;
- rank or select a winner;
- spend seed 45;
- run G9;
- open the protected holdout;
- access sealed recorder evidence;
- contact IBKR;
- submit orders;
- modify governed feature, label, training, simulator, gate, or selection code;
- change promotion, runtime, launchd, paid-data, or real-money state.

G8 is required reporting but is not an eligibility gate under the signed
2026-07-26 revision. G9 is specification-only and all result fields must be
`NOT_RUN_GOVERNED`.

## Allowed Audit Work

Create audit-only code and artifacts beneath:

```text
v4/audit/autoresearch/
  protocol101_stage1_training_regimen_adversarial_audit_attempt001/
```

Disposable diagnostic fits are allowed only where the frozen specification
explicitly authorizes them. They must be preregistered, use governed
non-holdout data, remain under the audit directory, carry
`NON_CANDIDATE=true`, and never enter selection or promotion evidence.

Treat governed project code as read-only. Record any needed governed-code
change as a defect, not an allowed repair.

## Checkpoints

### A. Freeze And Reconcile

Freeze source/document/artifact hashes and the pre-existing repository state.
Create the 28-row truth map and `row_failure_waterfall`.

### B. Trading Semantics

Reproduce causal timing, ladder scoring, abstention, selection, fills, fees,
one-position serial accounting, affordability, daily stop, and forced flat.

### C. Machine-Learning Validity

Audit folds, embargo, role separation, threshold/epsilon fitting, effective
sample size, multiple comparisons, target/ranking/abstention distinctions, and
entry-versus-exit attribution.

### D. Falsification Smokes

Run only the diagnostics frozen in the specification. Do not modify a method
or pass criterion after seeing its result.

### E. Gate Audit And Terminal Packet

Recompute applicable per-row gates, audit G9's specification without running
it, complete cross-implementation verification, and issue one terminal route.

Update `progress.json` after every checkpoint with:

- current checkpoint;
- completed verifications;
- unresolved defects;
- blocked claims;
- commands run;
- files created;
- unauthorized repository changes, which must remain empty;
- next bounded action.

## Required Decision Artifact

`row_failure_waterfall.csv` and its JSON twin must show for all 28 rows:

- raw and fee-adjusted economics;
- null, heuristic, seed, era, stress, drawdown, and concentration evidence;
- first and all failing gates;
- whether it made money before gates;
- separate experiment-validity, economic-evidence, and governance verdicts;
- whether failure is primarily model economics, experiment invalidity,
  governance rules, or insufficient evidence;
- a plain-English explanation.

No model may be labeled confirmed genuine signal. The highest economic label is:

```text
EVIDENCE_CONSISTENT_WITH_SIGNAL
```

## Iteration

Continue through mechanical blockers in audit-only tooling. Preserve invalid
attempts under a `void_outputs` directory. Allow at most three mechanical
repairs per audit component.

Do not repair scientific failures by changing thresholds, features, gates,
models, labels, or simulator rules.

If evidence is missing, mark the affected claim `BLOCKED` or `NOT_TESTABLE`,
identify the exact non-protected evidence needed, and continue all unaffected
work.

## Terminal Routes

- `regimen_valid_ready_for_model_free_reaggregation`
- `regimen_repairable_owner_amendment_required`
- `regimen_invalid_redesign_required`
- `regimen_blocked_insufficient_evidence`

The first route requires zero unresolved P0/P1 defects, correct preregistered
falsification behavior, reproduced trading semantics, exact truth-map
reconciliation, reproducible applicable gates, and no required experiment
repair.

The producing run may perform cross-implementation verification but may not
call itself an independent auditor or independently accept itself. A fresh
Goal/thread will perform independent acceptance later.

## Final Response

Answer plainly:

1. Is the historical/live training game sufficiently realistic?
2. Was leakage, replay distortion, or reward hacking detected?
3. Which rows lose money before gates?
4. Which rows make money before gates but fail governance?
5. Which apparent profits cannot be trusted?
6. Are the models, gates, simulator, or evidentiary process the main problem?
7. What must happen before reaggregation, selection, and G9?
8. What is the one next bounded goal?

Highest allowed claim:

> Protocol101 Stage-1 training-regimen adversarial audit complete.
