## 1. PRE-MORTEM — FALSE POSITIVE

**Severity convention:** probability and damage are ordinal 1–5 judgments, not calibrated probabilities. Rank 1 is the highest combined risk; ties are ordered by prior occurrence and breadth of invalidation.

|   Rank | Probable mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Specific catching audit or control                                                                                                                                                                                                                                                                              |          P × D |
| -----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------: |
|  **1** | **Research-process contamination of the score blocks.** The 251-session quote corpus has already been used for economic experiments, artifact diagnosis, and design decisions; the proposed 794-session backfill is earlier history, while file 05 says to chronologically split **all** complete sessions. Unless those 251 reused sessions are excluded, later score blocks will be untouched by the fold’s model fit but not untouched by the research program. A positive result could therefore be human-overfit rather than confirmatory. `04_PLAN(20260815-150933).md` identifies the additional history as pre-2025-08-01 and says the successor architecture and protocol were designed before acquisition; `05_FROZEN_DESIGN_AND_PROTOCOL.md` protects score blocks only from entry/exit fitting.   | Before opening outcomes, create an append-only session-exposure ledger recording every prior use of each session’s economics, labels, predictions, or derived diagnostics. Inferential scoring must contain only sessions never used to design the architecture, objective, controls, action mask, or protocol. | **5 × 5 = 25** |
|  **2** | **The historical model learns a data-construction game that cannot exist live.** `01_STATUS.md` says four train/live divergence axes remain unresolved—revisions versus first print, definitions survivorship, emission lag, and partial fills/rejects—and that the enforcement assertion has no production caller. A profitable historical policy could therefore rely on finalized definitions, corrected quotes, or timing that the live system never receives.                                                                                                                                                                                                                                                                                                                                            | Run a machine-enforced divergence gate on the exact declared pipeline. Require signed evidence or an enforced repair for every axis affecting features, candidates, labels, or action prices; mutate first-print/revision timing and verify decisions remain legal under the live arrival law.                  | **5 × 5 = 25** |
|  **3** | **A feature or action price crosses the decision boundary by one minute or one vendor emission.** This program previously produced approximately **+$119 per trade** from a feature that read the minute after the decision, and its shuffled-label null did not detect the leak. File 05 says “completed” candles and a “contemporaneous” ladder but does not define the event-time, receive-time, emission-lag, and order-request boundary precisely enough to prevent the same defect in either a feature or the ask used for entry.                                                                                                                                                                                                                                                                       | For every feature, candidate, score, and action price, record `event_time`, `receive_time`, decision time, and knowable time. Mutate all records arriving or revised after the legal boundary and require the candidate universe, feature tensor, scores, action, and price to remain unchanged.                | **4 × 5 = 20** |
|  **4** | **Post-outcome structural QC or survivorship selects favorable sessions/contracts.** File 05 applies unspecified “structural QC,” then retains “complete sessions,” and refers to “eligible” contracts without freezing completeness, eligibility, or exclusion rules. The program has already suffered a future-dependent filter that dropped contracts which stopped printing and materially changed the measured break-even.                                                                                                                                                                                                                                                                                                                                                                               | Freeze all session, minute, and contract inclusion rules before labels are constructed. Produce a full acquired-population manifest with one outcome-blind exclusion reason per omitted item; explicitly prohibit exclusions based on later quotes, realized movement, exit availability, or labels.            | **4 × 5 = 20** |
|  **5** | **The quote corpus contains a new price artifact rather than executable prices.** Replacing prints with quotes closes the known tape artifact, but it does not by itself control stale quotes, locked/crossed markets, zero displayed size, revisions, duplicated interval records, or forward-filled one-sided quotes. The previous artifact survived chronology, shuffled labels, composition matching, and positive out-of-sample years because the contamination was in the price itself.                                                                                                                                                                                                                                                                                                                 | Freeze a quote admissibility law: no forward fill, bounded age, noncrossed bid/ask, positive displayed size, deterministic duplicate/revision handling, and first eligible quote strictly after the action request. Compare selected trades and controls on quote age, spread, size, and revision status.       | **4 × 5 = 20** |
|  **6** | **The inferential unit or denominator manufactures confidence.** File 05 requires a positive “mean” and a corrected lower bound but never states whether the mean is per selected trade, per active session, or per all scored sessions; it also does not freeze the resampling unit. The project’s states have 10–27-minute dependence and material session clustering, so treating selected trades or held-state rows as independent can create a false lower bound.                                                                                                                                                                                                                                                                                                                                        | Define the primary estimand as serial account P&L per scored session, including zero for no-trade sessions. Perform all absolute and paired inference by whole-session resampling, with the alpha, correction family, weighting, pairing, and zero-trade treatment frozen in advance.                           | **4 × 5 = 20** |
|  **7** | **The exit head is credited for holding less, not for state-dependent exit skill.** File 05 trains `HOLD` against the “best strictly later executable sale value,” an oracle-like suffix target, but contains no duration-matched clock control. The previous fitted exit appeared advantageous only because it held approximately eight minutes; a five-minute stopwatch achieved slightly more with no model.                                                                                                                                                                                                                                                                                                                                                                                               | On identical outer-fold entries, compare the learned exit against a complete fixed-clock ladder and a control matched to the learned policy’s out-of-sample holding-time distribution. Require paired superiority at both midpoint and touch before attributing value to the exit model.                        | **5 × 4 = 20** |
|  **8** | **The outcome-blind control remains compositionally non-equivalent.** File 05 matches session, regime, minute, side, delta, premium, and trade count, but omits moneyness, spread, quote age, displayed size, time to expiry, opportunity-set size, and duration. The program previously found that an unmatched random reference credited the model merely for selecting a different instrument.                                                                                                                                                                                                                                                                                                                                                                                                             | Freeze the matching algorithm and calipers, then publish standardized balance diagnostics for every economically relevant predecision variable. Reject the control if balance fails or if matching reuses a small number of controls excessively.                                                               | **4 × 4 = 16** |
|  **9** | **The shuffled-label null is structurally too easy or fails open.** File 05 permutes entry values within sessions and exit target pairs within trajectories, which destroys temporal and overlapping-label dependence without proving that the resulting null represents a no-signal version of the same learning problem. A prior permutation procedure counted invalid NaN draws as evidence for the hypothesis.                                                                                                                                                                                                                                                                                                                                                                                            | Freeze a dependence-preserving block or session-level null, its number of refits, seed, missingness behavior, and validity checks. Any NaN, empty group, failed match, or invalid draw must terminate the run rather than count as a null failure.                                                              | **4 × 4 = 16** |
| **10** | **The nominally single architecture contains unfrozen training multiplicity.** File 05 names one source path and a 120-parameter count, but does not freeze the exact feature definitions, target equations, preprocessing, missing-value semantics, initialization, seed, optimizer, regularization, loss weights, epochs, checkpoint rule, or nested-fit topology. A positive could therefore be a lucky initialization, checkpoint, or semantic code change while still being called the same architecture.                                                                                                                                                                                                                                                                                                | Before outcomes, hash the complete executable training specification and environment. Fix one seed and checkpoint law, record every attempted process in an append-only ledger, and require a deterministic reproduction to regenerate identical predictions and trades.                                        | **3 × 5 = 15** |

---

## 2. CONTROL CROSS-CHECK

|      # | File 05 status | Literal protocol language and defect                                                                                                                                                                                                                                                                                                                                                               |
| -----: | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **1** | **ABSENT**     | No language excludes sessions already used by the research program from inferential score blocks. The strongest wording is only that the outer block “stays untouched by both entry and exit fitting,” which does not address prior human or program-level exposure.                                                                                                                               |
|  **2** | **ABSENT**     | File 05 never requires clearance of the divergence register, revisions-versus-first-print, definitions survivorship, emission lag, or partial-fill/reject semantics. “Applicable owner-controlled gate” is not a literal implementation requirement for any of those axes.                                                                                                                         |
|  **3** | **PARTIAL**    | It requires that “Masked future candles and invisible ladder nodes may not change any score” and says to “Audit every feature timestamp, still-forming candles.” The gap is that it does not freeze the actual decision timestamp, receive-time guard, quote emission law, or action-price timestamp; a same-minute or post-boundary ask can still be used legally under the wording.              |
|  **4** | **PARTIAL**    | It says, “After acquisition and structural QC, sort all complete sessions,” and later requires an audit of “contract identity/survivorship.” Neither “structural QC,” “complete,” nor “eligible contract” is defined, hashed, or required to be outcome-blind.                                                                                                                                     |
|  **5** | **PARTIAL**    | It says to buy “at the ask,” sell “at the bid,” and audit “whole-ladder equality.” It does not define quote age, displayed-size sufficiency, crossed/locked treatment, revisions, duplicates, forward filling, or whether the action price is the first quote after the request.                                                                                                                   |
|  **6** | **PARTIAL**    | It requires a “multiplicity-corrected one-sided lower bound” and 4-of-5 signs. It does not state the primary estimand, alpha, correction family, resampling unit, cluster definition, trade/session weighting, paired statistic, or zero-trade treatment. Each of those choices could change the verdict.                                                                                          |
|  **7** | **ABSENT**     | No fixed-clock ladder or duration-matched comparator appears anywhere in file 05. The protocol defines the oracle-like `HOLD` target but provides no control that separates exit intelligence from merely exiting earlier.                                                                                                                                                                         |
|  **8** | **PARTIAL**    | Exact language: “Match an outcome-blind control on scored session, regime, opportunity minute, side, delta, premium and trade count.” The matching algorithm, calipers, replacement law, balance tolerances, quote-quality variables, moneyness, time to expiry, and duration are unspecified.                                                                                                     |
|  **9** | **PARTIAL**    | Exact language: “Entry executable values are permuted within training sessions” and exit pairs “within training trajectories.” The protocol does not preserve temporal blocks, define the number of permutations/refits, prohibit invalid draws, freeze the seed, or require a known-answer false-pass campaign.                                                                                   |
| **10** | **PARTIAL**    | It freezes one named architecture and says an architecture change needs a new declaration; it also requires a future self-hashed declaration. But it does not state what semantic inputs the hash must bind, and the declaration occurs after acquisition. A mutable source path plus an incomplete hash manifest permits a defective implementation without formally changing the architecture.   |

---

## 3. PRE-MORTEM — FALSE NEGATIVE

### 3.1 Five most probable mechanisms

|  Rank | Mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Specific catching audit or control                                                                                                                                                                                                                                                 |          P × D |
| ----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------: |
| **1** | **The complete gate is underpowered at the effect size whose absence would close the branch.** File 05 contains no end-to-end power or recovery requirement for the actual sparse selector, nested exit training, five folds, three comparisons, and session-clustered inference. The program has previously obtained clean null behavior while recovering the required planted effect only 8% of the time, demonstrating that a sound-looking gate can still be incapable of answering its declared question. `01_STATUS.md` reports that earlier failure explicitly.  | Before real labels are opened, run the exact frozen pipeline on null and planted-effect fixtures. Require a prespecified false-pass ceiling and at least 80% recovery at the minimum executable effect that the final gate is intended to accept.                                  | **5 × 5 = 25** |
| **2** | **The 120-parameter restriction forces representational underfitting on an unsupported budget.** The 122-parameter limit is a projection from effective-sample calculations on four binary ITM-depth labels, while file 05 trains continuous `ENTER`/`WAIT` dollar values and `SELL`/future-sale values. File 06 explicitly states that effective sample size is label-specific and must be remeasured for a different target.                                                                                                                                          | Recompute conservative ESS for the actual entry and exit targets on each training prefix and run outcome-blind representability/known-answer tests proving that the frozen architecture can recover economically relevant planted interactions.                                    | **4 × 5 = 20** |
| **3** | **The entry objective is inconsistent with the policy that will realize the trade.** Entry is trained on a fixed executable 120-minute `ENTER`-versus-`WAIT` law, but the deployed position is subsequently managed by a learned exit that can leave at any held state. A genuine opportunity whose value depends on exiting earlier than 120 minutes can therefore be labeled unattractive at entry and never generate a trajectory.                                                                                                                                   | On nested training data only, measure target-policy consistency: compare each entry label with the value actually produced by the frozen out-of-fold exit law. If the ranking or sign materially changes, the negative result may close only the fixed-120-minute entry objective. | **5 × 4 = 20** |
| **4** | **The staged freeze prevents entry–exit co-adaptation and starves the exit head of the states where exit skill exists.** File 05 freezes the shared representation and entry parameters before fitting the exit. The program’s own conditional-exit analysis found that exit value can disappear under random or unskilled entries while being valuable on the subset that becomes large movers.                                                                                                                                                                        | Before outcomes, use a known-answer sequential fixture whose positive value exists only through an entry–exit interaction. If the staged law cannot recover it, a negative result cannot close the complete joint-policy game.                                                     | **4 × 5 = 20** |
| **5** | **The comparator controls away the exact-contract skill the policy is supposed to learn.** The candidate is required to choose an exact contract, but its control is gifted the policy’s side, delta, premium, and trade count. A truly profitable policy whose edge comes from side or contract selection could be positive absolutely yet fail the paired control requirement because the control inherits those choices. The broader goal describes exact-contract selection as part of the policy itself.                                                           | Separate the full-policy comparator from the attribution comparator: use an ungifted outcome-blind action control for the trading-policy claim, while retaining composition matching only to attribute timing versus instrument selection.                                         | **4 × 4 = 16** |

### 3.2 File 05 cross-check for false-negative controls

|     # | File 05 status | Literal cross-check                                                                                                                                                                                                                                                |
| ----: | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1** | **PARTIAL**    | It requires an “attainable selector proof” and an applicable gate to return `PERMITTED`, but defines neither the planted effect, false-pass rate, recovery probability, nor exact power campaign. That language cannot distinguish `NO EDGE` from `UNDERPOWERED`.  |
| **2** | **PARTIAL**    | It requires the built count to remain below a “post-QC conservative evidence budget.” It does not require ESS for the actual entry and exit targets, each outer/nested training prefix, or each separately trained parameter block.                                |
| **3** | **ABSENT**     | The protocol mandates the mismatch: fixed 120-minute entry values first, learned exit afterward. It contains no target-policy consistency audit and no limitation on the scope of a negative conclusion.                                                           |
| **4** | **ABSENT**     | No joint entry–exit known-answer control exists. Freezing the shared representation and entry parameters before exit fitting is a design rule, not evidence that this factorization can recover a joint edge.                                                      |
| **5** | **ABSENT**     | File 05 provides only the composition-matched control. It contains no ungifted full-action comparator and no instruction to interpret failure against the composition control as “no incremental timing skill” rather than “no complete policy edge.”              |

---

## 4. INDEPENDENT DEFECTS

### 4.1 The 120-parameter model is not budgeted against the data available to any outer fit

The backfill projection is **1,011–1,037 complete sessions**, with a worst projected full-corpus budget of 122 parameters. File 05 then reserves the earliest 40% as the initial prefix and scores five expanding chronological blocks.  

At (n=1{,}011):

* Initial prefix: 404 sessions.
* Score blocks: 122, 122, 121, 121, 121 sessions.
* Outer training sizes: 404, 526, 648, 769, and 890 sessions.

Under the only projection described—scaling measured information with session count—the corresponding conservative parameter budgets are approximately:

[
122 \times
\left[
\frac{404}{1011},
\frac{526}{1011},
\frac{648}{1011},
\frac{769}{1011},
\frac{890}{1011}
\right]
=======

[49, 63, 78, 93, 107].
]

The built count of 120 exceeds every one. The (n=1{,}037) endpoint produces almost identical budgets of approximately 49, 63, 78, 93, and 107. Therefore, under file 05’s own whole-module parameter convention, no outer fit is authorized by the projected conservative evidence budget.

**UNKNOWN:** the exact number of trainable entry/shared parameters and exit-head parameters in each phase. The missing document is `v5/research/causal_day_compact_shared_lifecycle.py` together with its design receipt. If fewer than 120 parameters are trained in each phase, the exact violation must be recomputed phase by phase; file 05 does not do that. 

### 4.2 The exit-head evidence budget is not measured at all

The exit head is not trained on all session-minutes. It is trained only on trajectories generated by nested out-of-fold entries, with at most two entries per session. Its effective sample is therefore determined by selected trajectory count, trajectory length, and within-trajectory dependence—not by the full minute-state corpus. File 05 provides no minimum trajectory count, no exit-target ESS, and no exit-head parameter split.  

**UNKNOWN:** the projected number of nested out-of-fold entry trajectories and the exit head’s trainable parameter count. The missing documents are the model source, design receipt, and fresh fit declaration.

### 4.3 File 06 measures dependence in the wrong statistical object for this model

File 06 estimates autocorrelation and design effects for four binary ITM-depth labels. The new model fits continuous action-value targets and a suffix-best exit target. Even apart from the explicit target mismatch, label autocorrelation is not the same as the dependence of the model’s score equations, gradients, or residuals; and the design effect of session means is principally an estimator for an unconditional mean, not a demonstrated sample-complexity formula for a nonlinear conditional policy. File 06 itself says ESS is label-specific.   

The sevenfold disagreement between the two ESS routes is disclosed, but selecting a parameter budget from either route does not validate the mapping:

[
\text{model capacity} = \frac{\text{label ESS}}{20}.
]

A known-answer learning-power experiment is required to establish that mapping.

### 4.4 The 20-observations-per-parameter rule has no verifiable statistical justification in the attachments

File 06 applies a 20:1 ratio directly to raw parameter count. It does not justify why 20 is appropriate for this architecture, regularization, target noise, rare action rate, multi-output training, nested chronology, or clustered data. Raw neural parameter count is also not generally equal to effective degrees of freedom.  

**UNKNOWN:** the cited basis for the 20:1 rule. The missing document is `governance/CAUSAL_DAY_FIT_RERULING_2026_08_14.md`; the underlying receipt and implementation, `effective_sample_size_2026_08_14/receipt.json` and `ops/measure_effective_sample_size.py`, are also not attached.

### 4.5 Linear extrapolation of the design effect across the backfill is not preregistered or justified

`04_PLAN(20260815-150933).md` says the 122–216 budget comes from “scaling the measured dependence structure.” File 06’s measured structure comes from 243 sessions and explicitly captures between-session clustering. Older added years can change base rates, clustering, completeness, quote density, and regime heterogeneity, so the design effect need not scale linearly with session count. File 05 says “post-QC conservative evidence budget” but provides no frozen recalculation formula, confidence bound, or rule for choosing among target-specific budgets.   

### 4.6 The risk law is inconsistent with the current project breaker

File 05 permits two entries per session and allows each selected ticket and worst selected loss to reach $500. The current status records a binding conflict involving a 5% daily breaker on a $10,000 account. Without an explicit serial daily-loss rule, two legal $500 losses could produce a 10% day while each individual trade passes file 05.  

**UNKNOWN:** the controlling signed charter language after the identified owner conflict. The missing document is `governance/CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md` plus any later owner ruling.

### 4.7 The “detectability depends only on session count” claim in file 01 is overstated

File 01 states that move dispersion, horizon, trades per session, tradable minutes, and friction all cancel and that detectability depends on session count and the statistical standard “and on nothing else.” That can be a conditional algebraic identity for a frozen marginal-return test under its assumptions; it is not a universal power law for a learned, abstaining, heavy-tailed, rare-event policy with estimated parameters and a conjunction gate. 

The same status page later reports that the sparse-tail target was already provable on 27 sessions and that the constraint was an 11.1-fold precision lift rather than data volume. That is a direct counterexample to the unqualified “nothing else” language. 

The defensible statement is narrower: for a prespecified session-return process and fixed testing law, the minimum detectable standardized mean scales approximately as (1/\sqrt{n}). It does not by itself determine model capacity or the power of this policy experiment.

### 4.8 File 01 is internally inconsistent despite declaring itself authoritative

The beginning of `01_STATUS.md` says the prior “only calendar time” conclusion was falsified, that the current blocker is effect size rather than sample size, and that two branches are open. Later, the same file says no rung is startable and “the binding constraint is the number of owned sessions, which only calendar time changes.”  

Because file 01 says it wins over conflicting files, an internally contradictory status page cannot serve as an unambiguous authority for the fit gate. The frozen declaration must identify the exact controlling status sections and mark superseded sections nonbinding.

### 4.9 The total fit topology is not frozen

The experiment is described as one fit, but file 05 requires five outer entry fits, unspecified nested chronological entry fits, exit-head fits, and an identical shuffled path. It does not freeze the number or boundaries of inner folds, whether scaling and target construction are recomputed for each inner fit, the number of stochastic training runs, or how failed/zero-entry inner models are handled.  

**UNKNOWN:** the actual number of model fits and their topology. The missing document is the fresh self-hashed declaration referenced by file 05.

---

## 5. VERDICT

# **AMEND FIRST**

As written, the protocol cannot support either decisive interpretation:

* A **positive** can be produced by research-level score-set reuse, historical/live divergence, an undefined quote clock, outcome-dependent QC, clustered inference, or an exit-duration confound.
* A **negative** can be caused by an unpowered gate, invalid target-specific capacity sizing, objective mismatch, staged entry–exit underfitting, or a comparator that controls away part of the policy’s legitimate action.

Most decisively, the 120-parameter count is compared with a projected full-corpus budget even though no outer model trains on the full corpus. Under the program’s own linear scaling, the implied fold budgets are approximately 49–107 parameters, not 122.

### Ranked amendment language

#### 1. Per-fit, target-specific capacity gate

> **For every outer and nested entry or exit fit, compute conservative effective sample size using only that fit’s training sessions and its actual target. The trainable parameter count for that phase must not exceed `floor(ESS / 20)`. The minimum fold-and-phase budget binds; a full-corpus or projected budget may not authorize a smaller-prefix fit.**

#### 2. Research-process confirmation firewall

> **Before any acquired outcome is opened, hash an exposure ledger classifying every session as `DEVELOPMENT` or `CONFIRMATION`. Any session previously used for economic analysis, label inspection, prediction diagnosis, architecture or objective design, control design, action-mask design, or threshold choice is `DEVELOPMENT` and is excluded from inferential scoring. If no chronologically valid untouched score set remains, the result is exploratory and may neither authorize nor close the strategy class.**

#### 3. End-to-end known-answer power gate

> **Before real labels are opened, the complete frozen pipeline must demonstrate a null false-pass rate no greater than 5% and at least 80% recovery of the minimum executable effect the final gate is intended to accept, using the exact fold sizes, selector, entry and exit training law, controls, session clustering, and inference. Failure ends the experiment as `UNDERPOWERED` without reading real economics.**

#### 4. Complete immutable semantic freeze

> **The declaration must hash the acquired-session manifest, QC and exclusion rules, eligible-contract mask, feature and target code, quote, fee and settlement laws, model source, preprocessing, missing-value semantics, loss weights, optimizer, epochs, seed, checkpoint rule, all fold boundaries, control matcher, null generator, and inference code. The post-acquisition declaration may change paths and content hashes only; any semantic difference cancels the experiment.**

#### 5. Historical/live divergence gate

> **No fit may use a feature, universe rule, label, or price path whose train/live divergence status is `UNKNOWN` or `MEASURED_DIFFERENT` without an enforced declared repair. The signed divergence receipt for the exact pipeline is part of the declaration. Runtime-only execution axes may remain open only if the result is labeled historical and no executable or deployment claim is made.**

#### 6. Exact decision and action-price clock

> **A decision for interval ending `t` occurs only at `t + L`, where `L` is the signed arrival and emission guard. Only first-print records received by that boundary are admissible. Entry and exit use the first eligible quote strictly after the order request. Later revisions, still-forming bars, same-boundary fills, and forward-filled quotes are unavailable.**

#### 7. Outcome-blind population and survivorship law

> **Before labels are constructed, freeze and hash the complete-session rule, every exclusion, the eligible-contract mask, quote freshness, size and crossed-market rules, and missing-exit handling. No session, minute, or contract may be excluded because of a post-decision price, future quote, realized path, exit availability, or label. Report every acquired item and its inclusion or exclusion reason.**

#### 8. Primary estimand and clustered inference

> **The primary outcome is serial account P&L per scored session, with no-trade sessions recorded as zero. All absolute and paired intervals resample whole sessions. Freeze alpha, the multiplicity family and correction, weighting, pairing, zero-trade handling, and the 4-of-5 sign rule before outcomes. Per-trade means are diagnostic only.**

#### 9. Duration-matched exit control

> **On identical outer-fold entries, score the learned exit against a fixed-clock ladder and a control matched to its outer-fold holding-time distribution at both midpoint and touch. Exit skill requires paired improvement over the duration-matched control; otherwise any improvement is attributed to time in the position.**

#### 10. Valid dependence-preserving null

> **The null must preserve session and trajectory dependence, missingness, and opportunity structure. Freeze its block or permutation law, number of refits, seed, and invalid-draw policy. Any NaN, empty group, failed match, or invalid draw fails closed, and the null must pass the same full gate during the known-answer campaign.**

#### 11. Separate policy and attribution comparators

> **Freeze matching, calipers, replacement, and balance tolerances before outcomes. Use an ungifted outcome-blind full-action control for the complete-policy claim. Use the side, delta and premium composition-matched control only for attribution. Report balance on session, minute, moneyness, spread, quote age, displayed size, time to expiry, opportunity-set size, and holding duration.**

#### 12. Objective-consistency and closure scope

> **Because entry is trained on a 120-minute value and exit is trained afterward, failure closes only this staged fixed-entry-target specification. It may close the complete long-option branch only if a pre-run joint known-answer test demonstrates that this training law recovers a profitable entry–exit interaction under the frozen game.**

#### 13. Serial daily-risk enforcement

> **The controlling daily breaker applies to the serial account ledger. Once realized daily loss plus the worst-case remaining loss of any open position reaches $500, no new entry is legal. Permission for two entries does not override the daily breaker.**
