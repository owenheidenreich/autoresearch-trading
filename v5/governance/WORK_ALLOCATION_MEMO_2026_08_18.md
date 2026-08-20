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
- ~~The **event calendar**, the highest-value input the design cannot currently use.~~
  **RESOLVED — OWNER RULING, 2026-08-19. Closed; do not reopen.** The event calendar is **not an
  admitted feature and no calendar data will be acquired.** The question is closed by how the bot
  will be operated, not by a model change:
  - **Pre-open releases (CPI and similar, 08:30 ET) need no feature.** The print lands before the
    first decision minute at 09:31, so its effect is already in the tape the model reads. The model
    sees the aftermath, which is the only part it could trade.
  - **FOMC days are handled by not trading.** The owner will not run the bot on FOMC announcement
    days. This is an operating rule, not a model input.

  Two **diagnostic-only** reporting requirements follow, and neither gates anything. (1) Keep the
  event-day flag in Phase 5 reporting, so assumption one is *verified* rather than assumed: if losses
  cluster on pre-open-release mornings despite the print landing before 09:31, the aftermath is not
  fully absorbed by the open and that is worth knowing; if they do not cluster, the reasoning is
  confirmed with a number behind it. (2) Report results **with FOMC sessions excluded, alongside the
  all-sessions figures** — the corpus still contains FOMC days and the model will have learned from
  them, so there is a mild mismatch between what it trained on and how it will be run, and reporting
  both makes visible whether the days the owner plans to sit out were carrying the result.

  Recorded here so a future session does not rediscover "the model cannot see the calendar" and
  propose solving it again. Nothing in this ruling authorizes a purchase, a vendor contact, or a new
  feature admission.

  **Owner amplification, 2026-08-19 — the day-type factors that matter for 0DTE**, recorded because
  they are what Phase 5 must be able to slice by: FOMC days carry **elevated IV**; CPI days carry
  **large price swings**; **OPEX** and especially monthly OPEX (**MOPEX**, the third Friday) is its
  own regime; **end of month** matters and a last Friday sometimes coincides with MOPEX; and beyond
  any release, **time of year** and **time of day** are themselves factors.

  **A distinction worth keeping straight, because it changes what is and is not closed.** Only the
  *economic release* calendar — which days are CPI or FOMC — needs a source this project does not
  have. **OPEX, MOPEX, end of month, last-Friday, month of year and day of week are arithmetic on
  the session date**: zero acquisition, zero cost, no vendor, no G3 admission. Time of day is
  already carried by the five clock channels the member reads. So if any of these are ever wanted as
  *features* rather than report slices, the blocker is a design decision about the parameter
  contract, **not** data availability — and that should not be confused with the closed event-calendar
  question.

  **Phase 5 must therefore report by day type**, diagnostic only and gating nothing: event day,
  FOMC-excluded, OPEX/MOPEX, end of month, and month of year, alongside the all-sessions figures.
  **The FOMC prerequisite is RESOLVED, 2026-08-19.** The owner supplied and verified **34**
  announcement dates (statement days, not minutes-release days) from the Federal Reserve's published
  calendars, covering 2022-06 to 2026-07. They are recorded in
  [`research/session_calendar.py`](../research/session_calendar.py) with their source.

  **The list was verified rather than trusted, three ways.** Its shape is right — 8 meetings a year,
  with 5 in 2022 from June and 5 in 2026 through July. Every date is a weekday and exactly one is not
  a Wednesday: **2024-11-07**, which is correct rather than a slip, because the November 2024 meeting
  moved to the 6th–7th around the US general election on the 5th. And the dates carry a measurable
  signature in this project's own tape, independent of the source they came from: the 33 present in
  the corpus show **1.91x** the median one-minute maximum step (13.74 against 7.18 index points) and a
  median session range of 65.62 against 47.23. The owner's stated reason for sitting out FOMC days is
  therefore confirmed with a number behind it.

  **One FOMC day is missing from the corpus and it is not missing on purpose: 2025-07-30**, which the
  clock gate excluded for three absent interior minutes (11:20–11:22). So the all-sessions figures are
  already short one FOMC day before anything is excluded deliberately, and Phase 5 must say so rather
  than describe its excluded cut as complete.

  **Measured day-type signatures on the 1,014-session corpus**, for the record: FOMC 33 sessions
  (1.91x step, range 65.6); quarterly OPEX 17 sessions (1.07x step, **range 66.1** — the largest range
  effect after FOMC); month end 50 (1.14x, 56.3); last Friday 50 (1.06x, 53.7); monthly OPEX 50
  (**0.95x** step, range 52.1 — no step elevation at all).
- **Tape source — RESOLVED, OWNER RULING 2026-08-19: SPX-derived (parity spot), not ES.** The active
  policy is SPXW/SPX-only with no futures input. The basis genuinely cancels in the difference-based
  tape channels today, but **provenance is what a future session inherits**, and "the candles are ES"
  is exactly the kind of quiet inconsistency that gets discovered mid-fit and forces a rebuild.
  Implementation is execution (Opus): the corpus tape is rebuilt from parity spot.
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
