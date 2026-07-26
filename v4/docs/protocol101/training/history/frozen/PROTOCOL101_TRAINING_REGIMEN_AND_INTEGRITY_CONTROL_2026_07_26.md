# Protocol101 Training Regimen And Integrity Control

Status: **EXPLANATORY CONTROL + COMPLETED AUDIT RECORD**  
Date: 2026-07-26  
Substantive gate amendment status: **G8 REVISION OWNER-SIGNED 2026-07-26**

This document explains the complete Protocol101 training program in plain
English, records the independent G1-G9 and anti-reward-hacking audit, and
defines what must still be proven before IBKR paper trading.

It does not silently amend the signed G1-G9 rules. The owner explicitly signed
the G8 revision in `PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`; that
revision is now binding.

## One-Page Summary

The project is training an SPXW 0DTE **entry model**. Once per completed market
minute, it receives a ladder of eligible call and put contracts, scores each
eligible contract's expected fee-adjusted return on premium, and either:

- abstains;
- buys one call; or
- buys one put.

The current research candidate is:

```text
hypothesis: H2
entry features: 14
exit label: policy 5 / convexity_run
initial seeds: 42, 43, 44
status: passes G1-G7; G8 reported under the signed v2 revision
fresh confirmation seed 45: not spent
```

H2 sees:

- 12 synchronized SPX market-context features; and
- internally recomputed delta and gamma for each eligible ladder contract.

H2 does **not** use raw bid, ask, spread, size, quote age, vendor Greeks, future
PnL, or exit-path information as model alpha. Raw quote data remains available
for the candidate guard, conservative fills, labels, PnL, and audit.

The independent integrity audit verified:

- all 420 H0-H3 model units and model hashes;
- the exact feature width of every fitted model;
- chronological and disjoint fit/calibration/validation roles;
- zero protected-holdout overlap;
- preregistration before result artifacts;
- no quarantined feature in a model-facing allowlist;
- no direct evidence of label/future leakage.

The audit did **not** prove that reward hacking is impossible. Repeated research
against the same five CV folds creates process-level overfitting risk even when
the code has no direct leakage. G9, the protected holdout, and candidate-specific
IBKR shadow transfer remain mandatory.

## What "The Same Game" Means

The historical and IBKR environments do not need byte-identical raw feeds. They
must present the fitted model with the same causal decision problem.

For this project, "same game" requires all of the following:

1. The same decision clock and completed-minute context.
2. The same static SPXW ladder geometry and contract identity rules.
3. The same model-facing feature definitions and calculation code.
4. The same candidate guard and affordability semantics.
5. The same action space: wait, buy one call, or buy one put.
6. The same threshold, score-noise epsilon, slot-margin, and fallback rules.
7. The same one-account, one-contract risk rules.
8. Historical labels use information unavailable at entry only as the training
   answer, never as an input.
9. A fitted candidate produces sufficiently matching actions and selected slots
   on historical-vendor and IBKR no-order shadow inputs.

The current evidence proves the first eight for the scoped research contract.
Item 9 remains candidate-specific and has not been completed because no
candidate has passed the pre-shadow gates yet.

Therefore the honest statement is:

> Family-level evidence supports training H2 on the scoped canonical game.
> We have not yet proven that the fitted H2 candidate is unable to distinguish
> historical and IBKR inputs in its final decisions.

That last proof is intentionally downstream. It must use the frozen candidate;
otherwise the model could be adapted to the shadow evidence.

## What The Model Actually Sees

### Candidate universe

The processed decision row carries a 21-strike by 2-right ladder:

```text
21 strikes across the static +/- $50 ladder
x calls and puts
= 42 possible slots before guards
```

The boundary-stable guard removes candidates that fail conservative price,
spread, freshness, size, or affordability rules. The model scores every
remaining eligible contract separately.

### H2 model-facing inputs

Every eligible contract receives the same 12 market-context fields:

1. `spx_vwap_gap_points`
2. `spx_vwap_gap_bps`
3. `spx_vwap_gap_over_session_range`
4. `session_range_bps`
5. `momentum_5m_bps`
6. `momentum_15m_bps`
7. `momentum_5m_over_session_range`
8. `momentum_15m_over_session_range`
9. `omar_clipped_neg3_pos3`
10. `vwap_side_alignment_flag`
11. `omar_side_alignment_flag`
12. `momentum15_side_alignment_flag`

It then receives two values specific to that contract:

13. `E.bs.delta`
14. `E.bs.gamma`

Delta and gamma are calculated internally from:

- tick-quantized option mid;
- SPX;
- strike;
- time to expiration;
- `r=0.05`; and
- `q=0`.

Historical and live-style paths use the same deterministic implementation.
Vendor-provided Greeks are not model inputs.

### Authorized features H2 is not using

The full synchronized contract contains three additional near-ATM composite
features:

- straddle mid over spot;
- put/call mid ratio; and
- side smile slope.

Those belong to H1 and H3. H2 does not use them. H3 tested the complete
17-feature contract but did not pass the economic/survival gates.

### Quarantined from model alpha

- Direct per-slot option-price paths.
- Internally computed IV and IV expansion/compression.
- VIX changes.
- Raw bid/ask/spread.
- Bid/ask size and quote age.
- Volume and open interest.
- Vendor-computed Greeks.
- Future return, future PnL, MFE, MAE, or post-entry path values.

Some quarantined fields remain essential outside alpha. For example, bid/ask
must be used for realistic fills and tradability; banning it from alpha does
not mean pretending it does not exist.

## What Stage 1 Trained

Stage 1 is supervised entry training, not a complete adaptive trader.

For each eligible minute/contract/policy:

1. The historical path generates a conservative ask-entry/bid-exit PnL label.
2. The label is converted to fee-adjusted return on premium.
3. A bounded HGB regressor learns that payoff target.
4. A chronological training-tail segment selects the abstention threshold.
5. Unseen validation sessions are scored once.
6. The serial simulator permits one account and one open contract at a time.

The model is never trained to maximize win rate. Its target is payoff.

### Seven policies are seven fixed training answers

The seven exit policies are not seven strategies dynamically selected by one
model. They are seven fixed definitions of "what would happen after entering
this contract now":

| Policy | Name | Stop | Target | Max hold |
|---|---|---:|---:|---:|
| 0 | quick_scalp | 35% | 60% | 10m |
| 1 | short_swing | 50% | 100% | 25m |
| 2 | extended_swing | 65% | 150% | 45m |
| 3 | patient_trend | 50% | 200% | 90m |
| 4 | thesis_hold | 100% | 300% | 120m |
| 5 | convexity_run | 100% | 999% | forced flat |
| 6 | tail_lottery | 100% | no practical target | forced flat |

Policy 5 means the entry model is rewarded for finding contracts worth letting
run. It does not contain a learned exit model. A learned lifecycle/exit model,
if justified, belongs to a separately signed Stage 2.

## Frozen H0-H3 Search

The search tested four preregistered feature hypotheses:

| Hypothesis | Features | Width |
|---|---|---:|
| H0 | Context only | 12 |
| H1 | Context + near-ATM composites | 15 |
| H2 | Context + internal delta/gamma | 14 |
| H3 | Context + composites + delta/gamma | 17 |

Each hypothesis ran:

```text
7 policies x 3 seeds x 5 folds = 105 model units
4 hypotheses x 105 = 420 total model units
```

The five folds are chronological expanding windows with one-session embargo.
Within each training fold, the final 20% of training sessions is reserved for
threshold, epsilon, and confidence calibration. It is not part of model fitting
and does not overlap outer validation.

## Current H0-H3 Result

The central economic result is:

- H0/H1 found signal in patient policies but failed drawdown/frequency and
  also failed the original hard-G8 benchmark.
- H3 found signal but failed G4 and also failed the original hard-G8
  benchmark.
- H2 policy 5 is the only row that passed G1-G7.
- H2 policy 5 failed only the original hard-G8 calibration rule.

The bounded isotonic-versus-Platt repair changed no trading behavior and still
failed G8:

| Seed | Original ECE | Repaired ECE | Limit |
|---:|---:|---:|---:|
| 42 | 0.114034 | 0.112032 | <= 0.10 |
| 43 | 0.105864 | 0.107318 | <= 0.10 |
| 44 | 0.107810 | 0.102025 | <= 0.10 |

The graph stopped correctly. Seed 45 was not spent.

## Independent Integrity Audit

Artifact:

```text
v4/audit/autoresearch/
  protocol101_stage1_gate_validity_and_integrity_audit_attempt001/
```

Results:

| Check | Result |
|---|---|
| Model units verified | 420 / 420 |
| Model files and hashes verified | 420 / 420 |
| Unique governed CV sessions | 271 |
| Protected-holdout overlap | 0 |
| Fit/calibration/validation role overlap | 0 |
| Preregistration before unit results | pass |
| Exact model feature widths | pass |
| Quarantined alpha in allowlists | 0 |
| Independent H0-H3 audits accepted | pass |
| Direct code/artifact leakage found | none |

This supports:

> No direct future-label leakage, protected-data use, feature-contract breach,
> or split-role violation was found in the audited Stage-1 implementation and
> artifacts.

It does not support:

> Reward hacking or research-process overfitting is impossible.

That stronger claim cannot be earned until the remaining independent evidence
is completed.

## G1-G9 In Plain English

| Gate | Question | Audit recommendation |
|---|---|---|
| G1 | Did it make money consistently across folds? | Keep hard |
| G2 | Is the result stronger than matched random selection? | Keep hard; add campaign-level multiplicity report |
| G3 | Did learning beat the fixed heuristic? | Keep hard |
| G4 | Is return sufficient relative to drawdown, without severe capital loss? | Keep hard |
| G5 | Does the result survive all three seeds? | Keep hard |
| G6 | Does it avoid systematically losing governed eras? | Keep with sample-size diagnostics |
| G7 | Does it behave like the intended day trader? | Keep as product/charter gate |
| G8 | Is the post-selection win-probability readout calibrated? | Signed v2: required report-only diagnostic until confidence controls behavior |
| G9 | Does one never-used training seed reproduce G1/G2/G4? | Keep spend-once hard |

### Why G8 was amended

The completed 28-row H0-H3 audit found:

```text
G1-profitable rows:                  7
G1-profitable rows passing G8:       0
non-G1 rows:                        21
non-G1 rows passing G8:             20
```

The calibrated confidence value is attached after entry action and slot
selection. It does not control entry, abstention, sizing, or exits.

The original hard G8 blocked a model because a non-operative diagnostic was
more difficult to calibrate for patient convex payoff policies. In this
campaign, higher ECE was positively associated with higher PnL
(`Spearman rho = 0.6108`, `p = 0.00056`).

This does not prove calibration is useless. It proves the current hard gate is
not aligned with the behavior being selected.

Signed amendment:

> G8 remains a required calibration diagnostic but is report-only while
> calibrated confidence does not control any trading action or risk. Before a
> future model uses confidence for abstention, sizing, routing, or exits, a new
> action-conditioned calibration gate must be preregistered and passed.

The amendment applies to all H0-H3 evidence. It cannot be used only to rescue
H2 after seeing its result.

## Reward-Hacking Risk Register

### Controls that passed

- All H0-H3 hypotheses were frozen before the search.
- Every hypothesis and all seven policies were executed.
- Model seeds and folds were frozen.
- Validation did not select model features.
- Threshold and epsilon came from training-tail calibration only.
- Protected holdout was excluded.
- Producers could not accept their own gate results.
- The G8 repair menu was preregistered before repaired ECE was computed.
- The repair changed no model, threshold, action, contract, trade, or PnL.
- G9 seed 45 remains unused.

### Material risks still open

#### 1. Repeated-CV research overfitting

The same five outer folds informed H0-H3 comparison and later H2 attention.
Even without code leakage, humans and agents can overfit decisions to repeated
validation evidence.

Control:

- freeze the gate amendment now;
- do not add another calibrator or H4;
- use seed 45 once;
- freeze the final candidate;
- open the protected holdout once.

#### 2. Multiple comparisons

The campaign compared 28 hypothesis-policy rows. G2 is a strong per-row matched
null, but it is not a campaign-level maximum-statistic correction.

Control:

- compute a report-only campaign max-statistic or conservative multiplicity
  sensitivity before final candidate freeze;
- do not tune the model from that report;
- require G9 and protected holdout regardless.

#### 3. Post-result calibration attention

The calibration repair was scientifically bounded and clean, but it was
designed after observing H2's G8 failure.

Control:

- no third calibration method;
- no additional H2-specific repair;
- apply any G8 amendment globally to all frozen rows.

#### 4. Candidate-specific live transfer

Family D and E source-discriminator AUCs were near chance:

```text
D composites:      0.502582
E delta/gamma:     0.500492
```

Direct option-price features scored `0.550524` and remain quarantined.

These results support the feature families, not the final fitted model.

Control:

- run the frozen candidate on paired historical and IBKR no-order shadow rows;
- gate action agreement and selected-slot agreement;
- report score drift and disagreement concentration;
- do not tune on confirmation evidence.

## Required Sequence From Here

### A. Gate amendment - COMPLETE

Owner signed the global G8 amendment on 2026-07-26.

No model training occurs.

### B. Global reaggregation

If signed, recompute H0-H3 eligibility from the existing frozen artifacts:

- no retraining;
- no threshold changes;
- no policy changes;
- no new features;
- no new seeds;
- add the campaign-level multiplicity diagnostic;
- independently audit selection.

### C. Fresh-seed G9

If H2 policy 5 remains selected, spend seed 45 exactly once across the five
frozen folds. G9 passes only on G1, G2, and G4 under the signed rule.

### D. Final fit

Fit one immutable candidate using the already-approved training scope. Freeze
model, feature, threshold, epsilon, policy, source, and simulator hashes.

### E. Protected holdout

Open the 30 protected sessions once. A failure burns the candidate; it cannot
be retuned against the holdout.

### F. Research-candidate freeze

Freeze the historical candidate and prohibit further model edits.

### G. Candidate-specific runtime transfer

Run paired no-order decision shadow:

- same canonical features;
- same candidate guard;
- same actions;
- same selected contracts;
- no broker orders.

### H. Live shadow

Observe the frozen system on IBKR with no orders through healthy and disrupted
windows. Verify the 15-minute clean-window reset.

### I. IBKR paper review

Only after all prior evidence passes may the owner separately authorize guarded
paper-submit.

## Non-Negotiable Stop Rules

- Never train on the protected holdout.
- Never use recorder/confirmation outcomes to choose a model.
- Never introduce a feature after viewing candidate results without a new
  signed foundation freeze.
- Never change a failed gate only for the candidate that failed it.
- Never replace a failed G9 seed.
- Never let raw bid/ask timing or vendor Greeks become alpha without new parity
  evidence.
- Never claim the historical and IBKR games are indistinguishable until the
  frozen candidate passes decision shadow.
- Never claim paper readiness from CV or holdout performance alone.

## Current Honest Status

```text
Scoped synchronization:       passed for offline Stage-1 research
H0-H3 training:               complete
Artifact/leakage audit:       passed with process-level risks noted
Gate-validity audit:          complete
G8 contract:                  signed v2; report-only diagnostic
G9 seed 45:                   unspent
Protected holdout:            unopened
Candidate-specific transfer:  not run
Paper readiness:              no
```

## Evidence Index

- `v4/docs/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md`
- `v4/model/protocol101_canonical_stage1_contract.py`
- `v4/model/protocol101_scoped_stage1_hgb.py`
- `v4/audit/autoresearch/protocol101_clean_window_hill_climb_readiness_2026_07_25/summary.json`
- `v4/audit/autoresearch/protocol101_scoped_canonical_stage1_readiness/summary.json`
- `v4/audit/autoresearch/protocol101_h2_policy5_calibration_repair_attempt001_audit/summary.json`
- `v4/audit/autoresearch/protocol101_stage1_gate_validity_and_integrity_audit_attempt001/summary.json`

## Owner Decision Recorded

The owner approved the global G8 amendment. Calibration remains required
reporting but does not determine eligibility until it controls behavior.

The next authorized work is model-free H0-H3 reaggregation and independent
selection. Seed 45 may be spent only if that frozen process selects H2 policy 5.

Signature: OWEN HEIDENREICH

Date: 07-26-2026
