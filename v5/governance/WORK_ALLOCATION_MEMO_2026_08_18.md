# Memorandum — division of remaining job-46 work between Fable and Opus

**Status: SIGNED 2026-08-18 by the repository owner.** Instruction given in conversation: assign the
remaining job-46 tasks between Fable and Opus, require the executing session to **stop and flag the
model handoff at each phase boundary**, and sign. Binding on every session executing job 46.

**To:** the sessions executing job 46
**Subject:** which remaining tasks require design judgement (Fable) and which are execution against a
settled design (Opus)

## 1. Why this exists

The corpus build is running and the remaining path to a trained model is short. The tasks left are not
of one kind: some are engineering against decisions already made, and one or two are genuine design
judgements where a cold, adversarial reading is worth more than continuity. Assigning them by hand now
prevents two failure modes this project has already paid for — spending an expensive planning pass on
plumbing, and letting a design question get settled silently by whoever happens to write the code.

The governing principle: **Fable decides what is true or what should be built; Opus builds it and
proves it works.** Where a task is mostly mechanical but contains one buried judgement, it is assigned
to Opus with the judgement named and escalated.

## 2. Assigned to Opus — execution against settled design

These require no new decisions. The design, the risk law and the evaluation law are signed.

1. **Finish the corpus build.** Running now over 1,014 eligible sessions. Report the receipt with
   per-era base rates and the carried-close footnote (`settlement_close_carried`,
   `settlement_carry_minutes`, `cash_settled_share`).
2. **Re-measure the parameter budget** on the real chronology using
   `ops/measure_effective_sample_size.py`. The 122–216 projection was made against a corpus that did
   not exist; it must be replaced by measurement, not inherited.
3. **Write the `SessionEpisode` adapter.** The last structural gap between corpus and trainer: nothing
   in `v5/research` or `v5/ops` constructs a `SessionEpisode` today. **Named hazard, verified
   2026-08-18:** `sell_paths` must be keyed `(decision_minute, ladder_column)` — one path per
   *contract*, not per minute. The minute-only form was a real defect, fixed 2026-08-17; it throws no
   error and yields a plausible exit trained on the wrong instrument. Regression tests exist; the
   adapter must satisfy them rather than route around them.
4. **Build and register the declared architecture member.** Count it from the built module, never
   transcribe. It is a **new member beside the frozen baseline**, not an edit to
   `causal_day_compact_shared_lifecycle.py`, which the semantic freeze pins.
5. **Run Phase 4a**, the calibrated probe, and apply its pre-declared verdict rule exactly as written —
   including publishing the negative if it fires. The rule exists because an uncalibrated harness
   already nearly produced a false negative on 2026-08-15.
6. **Runtime parity and the dormant paper kit** (plan Phase 6), unchanged in scope.

## 3. Assigned to Fable — design judgement

Send these cold, with repository access, and only these.

1. **Pre-fit adversarial review of the assembled corpus and feature set.** The first moment the real
   thing exists is the last cheap moment to catch a defect in it. Specifically: do the two eras
   constitute one population or two? The measured base-rate difference (36.3% backfill against 31.8%
   owned) is real and per-era reporting is already required — but whether a **single model** should
   train across both eras, or whether era should be an input, or whether the older era should be
   weighted down, is a design judgement nobody has made. It bears directly on whether the first fit
   is meaningful.
2. **Adjudicate the Phase 4a outcome if it lands ambiguously.** If the probe clearly passes or clearly
   fails, Opus applies the rule and no judgement is needed. If it lands near the +4.0pp boundary, or
   passes on planted data while behaving oddly on real features, that is exactly the situation where
   this project has historically talked itself into a result. Fable reads it cold.
3. **Design the exit-phase evaluation split**, if the entry survives. The requirement is signed —
   loss-averted and capture-efficiency reported separately, each against a duration-matched control,
   with degenerate always-cut and always-hold rules reading as degenerate. **How to construct those
   two statistics** on real trajectories is unspecified and is a genuine design problem.

## 4. Explicitly assigned to neither, pending owner decision

- Anything reopening signed law: the $2,000 dollar-stated cap, the 20% breaker, the −40% backstop,
  the compounding account, the entry-then-frozen-exit ordering, the chronology, or the stated target
  (31.65% baseline, 45–50% survival, ≈+14 points).
- The **event calendar**, the highest-value input the design cannot currently use. It requires a data
  source, G3 admission and a live-parity proof. Standing instruction: flag event days in Phase 5
  reporting as a diagnostic only, so its value is measured before it is bought.
- Any purchase, vendor or broker contact, paper or live order, or unattended job.


## 5. Mandatory handoff protocol — stop and flag at every boundary

**A session may not cross a boundary in §2/§3 by continuing to work.** At each boundary below it must
halt, state which model should take the next task and why, and wait for the owner. This is a hard
stop, not a recommendation, and it holds even when the next task looks small.

The reason is specific: this project's most expensive errors were not wrong answers but decisions made
by whoever happened to be holding the keyboard — a design question settled silently inside an
implementation, a marginal result read favourably by the session that produced it. A handoff is cheap;
re-deriving a corrupted fit is not.

| Boundary | On reaching it | Hand to |
|---|---|---|
| Corpus build completes | Report the receipt, per-era base rates, carried-close footnote. **Stop.** | **Fable** — pre-fit adversarial review, incl. the one-population-or-two question |
| Fable's corpus review returns | Report its verdict and any design change it directs. **Stop.** | **Opus** — budget, adapter, architecture member |
| Adapter + member built, before Phase 4a runs | Report built parameter count and measured budget. **Stop.** | **Opus** may proceed to run 4a once the owner confirms |
| Phase 4a returns a **clear** pass or fail | Apply the pre-declared verdict rule exactly, publish either outcome. **Stop.** | **Opus** on a clear pass; on a clear fail the job stops and publishes |
| Phase 4a returns **ambiguous** — near the +4.0pp boundary, or plant recovered while real features behave oddly | Do **not** interpret it. Report the raw numbers only. **Stop.** | **Fable** — adjudication |
| Entry survives, before exit evaluation is designed | Report the surviving entry stream. **Stop.** | **Fable** — design the loss-averted / capture-efficiency split |
| Exit design returns | Report it. **Stop.** | **Opus** — implement and evaluate |
| Any point where a task not listed in §2/§3 appears | Classify it: execution → Opus, judgement → owner. **Stop and say which.** | Owner |

**Ambiguity rule.** If a session cannot tell whether the next task is execution or judgement, that
uncertainty is itself the answer: stop and ask. A task that feels like it needs a decision is one.

**What a handoff flag must contain**, so the owner can act on it in one read: what just completed and
its headline numbers; which model should take the next task; one sentence on why; and what that next
session needs to know that is not already in the work log.

## 6. Signature

Signing directs the sessions executing job 46 to follow this allocation, and confirms that a task not
listed here defaults to Opus if it is execution and to the owner if it is a decision.

- **Signed: repository owner, 2026-08-18**, as drafted, including the mandatory handoff protocol
  at §5.

## 7. Standing note on the reporting layer

Four separate defects in this job have been in the layer watching the work rather than the work: a
`| tail` that masked a crashed process as exit 0, a `status=$?` that killed a retry wrapper, a
preflight that could lose an hour to one vendor timeout, and a `Write` that replaced a live file as
silently as it would have created one. In every case the tool reported success while something was
lost. Before trusting any wrapper, monitor or guard, state what it does on the failure path.
