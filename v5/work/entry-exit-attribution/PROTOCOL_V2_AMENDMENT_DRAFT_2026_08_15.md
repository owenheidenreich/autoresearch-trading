# Draft amendments to the compact shared lifecycle protocol — revision 2, awaiting owner signature

**Status: DRAFT. Nothing here is adopted.** Revision 2 incorporates the second-round external
review verdict (**AMEND AGAIN — KEEP NO PURCHASE / NO FIT**): A1 and A2 of revision 1 overstated
what the synthetic campaigns measured and are replaced with the referee's narrower language; the
divergence-gate rejection is reversed; five new amendments are added. Reports under
`external-review/chatgpt-research/8-15-26/`; measurement status in
[`CAPACITY_KNOWN_ANSWER_2026_08_15`](../../research/findings/CAPACITY_KNOWN_ANSWER_2026_08_15.md).
Each amendment tightens the frozen protocol; none loosens a gate, adds a retry, or expands the
search space.

## A1 — Complete known-answer gate (replaced per referee)

> Before any real target is constructed, the exact executable experiment pipeline must pass a
> preregistered known-answer campaign at every actual outer and nested training size. "Complete
> frozen pipeline" means the identical production code paths for preprocessing, target
> construction, target scaling, loss, optimizer, regularization, checkpoint selection, chronology,
> serial account simulation, controls and inference. Recovery is the proportion of planted worlds
> in which the complete final gate passes; false pass is the proportion of null worlds in which
> that same gate passes. One-sided 95% confidence bounds, not point estimates, must establish
> recovery of at least 80% and false pass of at most 5%. A phase-specific or stateless harness may
> block only that phase and may not establish complete-policy power or close the lifecycle
> specification. **The existing V1–V3 campaigns do not satisfy this gate; they are development
> diagnostics.**

## A2 — Scope of capacity evidence (replaced per referee)

> No observations-per-parameter ratio may authorize a fit by itself; a ratio is at most a
> nonauthorizing screening bound. Authorization requires both a fit-specific conservative evidence
> bound and a faithful known-answer campaign at every binding training size; neither substitutes
> for the other. A failed known-answer campaign establishes inadequacy only for the exact
> architecture, preprocessing, target, optimizer, training law, effect family, noise law and
> training sizes tested; it does not establish that additional data cannot power a different
> predeclared training law, and it may not close the broader strategy class.

## A3 — Research-exposure firewall (unchanged; accepted by referee)

> Before any acquired outcome is opened, every session is classified `DEVELOPMENT` or
> `CONFIRMATION` in a hashed exposure ledger. Sessions whose economics, labels, predictions or
> diagnostics informed any design decision — including the 251-session quote corpus used throughout
> 08-13/08-14 — are `DEVELOPMENT` and are excluded from inferential scoring. If no chronologically
> valid untouched score set remains, the result is exploratory and can neither authorize nor close
> the strategy class.

## A4 — Serial daily breaker (unchanged; accepted by referee)

> The daily loss limit binds the serial account ledger. Once realized daily loss plus the
> worst-case remaining loss of any open position reaches the controlling breaker, no new entry is
> legal that session. Permission for two entries per session does not override the breaker.

## A5 — Duration-matched exit control (clarified per referee)

> On identical outer-fold entries, the learned exit must beat both a fixed-clock ladder and a
> control matched to its own out-of-fold holding-time distribution, paired, at midpoint and at the
> touch. The clock ladder's members, the duration-matching algorithm and the multiplicity treatment
> of the added control family are frozen before outcomes. Otherwise exit value is attributed to
> time in position, not to the model.

## A6 — Quote admissibility and decision clock (numeric laws bound to receipts)

> Freeze before outcomes: no forward-filled quotes; quote age at most the signed usable-latency
> envelope of the Track-A issuance (p99 319.521 ms plus the declared emission lag from the signed
> [emission rule](../../governance/EMISSION_LAG_LOWERING_RULE_2026_08_12.md) once signed, and until
> then the conservative 480 ms bound of capture declaration v9); non-crossed bid/ask; positive
> displayed size; deterministic duplicate/revision handling with corrections refused rather than
> resolved by delivery order; entry and exit at the first eligible quote strictly after the order
> request. A decision for the bar ending at time t may read only records received by that boundary;
> still-forming bars and later revisions are unavailable. Every numeric constant here must name its
> receipt; a constant without a receipt blocks the fit.

## A7 — Primary estimand and clustered inference (unchanged; accepted by referee)

> The primary outcome is serial account P&L per scored session, with no-trade sessions counted as
> zero. All absolute and paired intervals resample whole sessions. Alpha, the multiplicity family,
> weighting, pairing and zero-trade handling are frozen before outcomes. Per-trade means are
> diagnostic only.

## A8 — Semantic freeze, pre-acquisition binding, and closure scope (strengthened per referee)

> All experiment semantics are frozen **before vendor contact or acquisition**: the QC and
> exclusion rules, eligible-contract mask, feature and target code, quote/fee/settlement laws,
> model source and its complete imported dependency tree, preprocessing, optimizer, seed and
> checkpoint rule, fold boundaries, control matcher, null generator, inference code, and the
> runtime environment (interpreter, library versions, determinism settings). The post-acquisition
> declaration may populate only acquired file manifests, paths and content hashes; any other
> semantic difference cancels the experiment. A failed run closes only the declared staged
> specification; it may close the long-option branch only if a joint entry-exit known-answer test
> first shows the staged law can recover an interaction edge.

## A9 — Divergence execution gate (new; revision 1's rejection reversed)

> Before any fit, the declared pipeline must invoke `assert_no_unknown_on_path` on every feature,
> candidate-universe, target and execution dependency and attach a passing signed receipt to the
> declaration. Any required axis in `UNKNOWN` status, or in `MEASURED_DIFFERENT` status without its
> declared enforced repair, refuses the fit. Hashing the divergence implementation without
> executing its gate does not satisfy this amendment.

Basis: STATUS records that four axes still block a fit and that `assert_no_unknown_on_path` has no
production caller — hashed-but-unexecuted machinery binds nobody.

## A10 — Synthetic development firewall and deterministic seeds (new)

> Known-answer development worlds used to select or repair the generator, optimizer, stopping law
> or recovery statistic are permanently excluded from known-answer confirmation. Confirmation seeds
> must be listed in the declaration, derived deterministically by SHA-256 from literal trial
> identifiers, and never generated with language-runtime `hash()`. The runner must verify the
> declaration's law and transitive implementation hashes before execution, and the receipt must
> bind the declaration, runner, model, dependencies, seeds and per-trial outcomes. Outcome-blind
> early stopping must restore the checkpoint with the minimum declared training loss; a terminal
> in-memory state may not be scored unless it is that checkpoint.

Basis: V1/V2 violated all three clauses (process-salted seeds falsified V2's identical-seeds claim;
last-not-best checkpointing; unverified declarations), and each defect was found only by external
review.

## A11 — Calibration identity (new)

> The planted effect, sparsity, target variance, cross-contract covariance, within-session
> dependence and temporal persistence of any known-answer world must be calibrated to the declared
> quote-based target and serial-account estimand using development evidence. A known measurement
> artifact may not be reused as irreducible target noise. Every calibration choice and its
> sensitivity range is frozen before confirmation worlds are opened.

Basis: V1/V2 used the $102.60 print-artifact residual — the contamination the quote purchase exists
to remove — as independent per-minute noise.

## A12 — Null, comparator and QC semantics (new)

> Null transformations must preserve the declared session, trajectory, overlapping-label and
> missingness structure. Any NaN, empty group, invalid draw, failed match or unsupported action
> fails closed. The complete-policy claim uses an ungifted outcome-blind control; the
> composition-matched control supports attribution only. Every session, minute and contract
> exclusion is fixed before labels and may not depend on any post-decision quote, realized path,
> future availability, settlement result or target.

## A13 — Evidence publication (new)

> The declaration and the receipt required to reproduce every reported aggregate must live in a
> tracked evidence path. A finding may not depend solely on a machine-local ignored audit
> directory.

Basis: the V1/V2 receipts were cited by the pushed finding while sitting in a git-ignored
directory; the capacity receipts now carry an explicit `.gitignore` exception.

## What signing does and does not do

Signing adopts tighter law. It does not authorize a fit, a purchase, or vendor contact. The
120-parameter fit remains refused because no faithful end-to-end known-answer preflight exists,
the score blocks are exposure-contaminated, the exit-phase evidence is unmeasured, and the serial
risk conflict is unresolved — the branch is blocked, not closed.
