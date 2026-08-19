# Design brief — what the SPXW 0DTE lifecycle model should look at and learn

**For: Fable, with full repository access. From: job 46. Owner-approved 2026-08-16.**

You are designing the **learning content** of this model. Not its governance, not its risk law, not
its statistical gates — those are signed and settled. The machinery is in good shape; the
intellectual content is a single unwritten paragraph, and that paragraph is the whole experiment.

**Read the primary sources.** This brief deliberately does not transcribe evidence. Paraphrase decays;
the receipts do not. Every claim you need is behind a path below, and where this brief states a number
it is because the number exists nowhere on disk yet.

---

## 1. The question

> Given everything this project has measured, **what should this model actually look at, and what
> should it learn?**

Phases 1–3 of the approved plan (`~/.claude/plans/develop-a-plan-for-smooth-hare.md`) are well
specified. Phase 4 — the substance — is one line: *"train entry, then frozen-entry exit."* What the
model sees, what it learns, in what order, and how anyone tells a real signal from a restatement of
option geometry are all undecided.

## 2. The failure you are designing against

The last fitted model was attributed, and the finding was damning: **V5 became an additive
option-geometry sensor, not a trader.** Chart state added the same offset to every contract in a
minute, so the tape could never reorder the ladder; the WAIT and EXIT heads received no gradient at
all. Read it rather than my summary:

- `v4/audit/autoresearch/causal_day_v5_instability_attribution_2026_08_14_attempt002/`
- `v5/work/entry-exit-attribution/LOG.md` — the cycle-1 and cycle-4 entries

**The successor inherits that feature set.** `compact_shared_lifecycle` fixed the *architecture*
(state×side and state×moneyness interactions now exist, so state *can* reorder the ladder) but nobody
has asked whether the *features* carry information capable of doing so. A 120-parameter model fed the
same inputs is at risk of being a larger geometry sensor, and no amount of risk-law hardening
prevents that.

The owner's test for every feature you propose:

> **Could this feature change which strike the model prefers — or does it only describe the
> contract?**

## 3. Facts you cannot infer from the repository

Everything else, read. These four are true as of 2026-08-16 and contradict what the artifacts on disk
will tell you:

1. **The corpus is about to be ~1,045 sessions, not 243.** The backfill acquisition is in flight:
   2022-06-01 → 2025-07-31 joins the owned 2025-08 → 2026-07 quote corpus. **Two data roots and two
   eras.** Every existing artifact, receipt, dataset and parameter count was built against the
   243-session corpus alone. Treat all of them as scale-limited evidence, not as the corpus you are
   designing for.
2. **The parameter budget must be re-measured on the real chronology.** The 122–216 projection was an
   extrapolation from measured design effects, not a measurement. `v5/ops/measure_effective_sample_size.py`
   must be re-run on the built corpus and the budget taken from that. Do not inherit the projection.
3. **The entry ceiling changed today**, from an equity-derived $1,300 to a signed fixed $2,000
   (premium plus fees). **Every mask and candidate artifact built before 2026-08-16 is stale** and
   must not be mixed with rebuilt ones. The action space you are designing for is materially wider
   than the one any existing receipt describes.
4. **The pre-stated target**, which the declaration must carry before any fit: random-entry baseline
   **31.65%**, survival at a $10,000 account needs **45–50%** — about **+14 points**, roughly **twice
   the best entry effect this project has ever measured** (+7.1pp, and that one failed chronological
   stability). This is stated in advance precisely so a mediocre result cannot be talked into a good
   one afterwards.

## 4. Where to read

| Question | Path |
|---|---|
| How the last model failed | `v4/audit/autoresearch/causal_day_v5_instability_attribution_2026_08_14_attempt002/`, `v5/work/entry-exit-attribution/LOG.md` |
| What the model currently sees | `v5/research/causal_day_tensorizer.py`, `v5/research/causal_day_compact_shared_lifecycle.py`, `v5/research/causal_day_compact_interaction.py` |
| What is legally visible at decision time | `v5/research/feature_admission.py`, `v5/STATUS.md` §13 — G3 admits 51 of 73 scoped features; 32 are barred for named reasons; the certification **expires 2026-11-10** |
| The negative evidence that constrains any design | `v5/research/history/DO_NOT_RETEST.md` rows **326–345** — especially the 375-cell drift census, both selective-entry closures (print and quote), the stopwatch exit, and row 41's conditional decomposition |
| Sample-size reality | `v5/research/findings/EFFECTIVE_SAMPLE_SIZE_2026_08_14.md`, `v5/research/findings/SCALE_SENSITIVITY_2026_08_16.md` |
| Settled law — read, do not reopen | `v5/governance/DEVELOPMENT_CHARTER_2026_08.md`, `CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md`, `ADDENDUM_STOP_LEVEL_AND_UNDERPOWER_2026_08_16.md` |
| The trainer your design must feed | `v5/research/lifecycle_trainer.py`, `v5/tests/test_lifecycle_trainer.py` |
| Exit scoring already ruled | `v5/work/lifecycle-training/PLAN.md` phase 5 |

One pointer worth following deliberately: **the ledger names its own unexplored ground.** Row 338's
reopening conditions list what was never in any tested feature set — order-flow imbalance, the event
calendar, cross-asset context, chain-wide skew and term structure. The ladder tensor already carries
bid/ask sizes and a whole-chain surface that no fitted model has used as anything but geometry.

## 5. What to deliver

**A. The feature contract — field by field.** A reason for every inclusion *and* every bar. Each
field must answer the owner's test above. State explicitly which fields can reorder the ladder and
which merely describe a contract; a design where nothing can reorder is V5 again.

**B. The label at corpus scale.** The ENTER-versus-WAIT dollar value and the first-touch family
(+G before −L) across ~1,045 sessions and **two eras**. Address era differences in ladder composition
and settlement source directly — 2022 ladders are roughly a third the width of 2025 ones, and the
official-settlement source is verified for the owned year only.

**C. The curriculum.** What trains first, what freezes when, how the two declared members differ
concretely, and — the hard one — **what the model is asked to learn that is not already priced into
the premium.** Magnitude was measured predictable and *fully priced* (row 332): selecting the busiest
third raised the realised move 51% and the premium 42%, moving P&L by thirty cents. A design that
rediscovers magnitude is a design that loses.

**D. The diagnostic suite.** Tests that distinguish *learned geometry* from *learned timing*
**before economics are read**. V5's failure was visible only after the fit; make it visible during.
This is the deliverable most likely to save the pass.

**E. The wiring spec.** How the built corpus becomes the `SessionEpisode` records
`v5/research/lifecycle_trainer.py` consumes. **That adapter does not exist** and is the literal gap
between "built" and "runnable."

## 6. Boundaries

You may **not** reopen: risk limits, account law, chronology, the entry-then-frozen-exit ordering, or
the target. They are signed; reopening them spends the pass on settled ground. Do not write
governance — if your design needs a rule changed, say so in one line and stop.

## 7. The honest prior

**The measured prior on this strategy class is poor, and you should design knowing it.** Every
economic screen this project has run on long single-leg 0DTE has come back negative or underpowered:
the drift census found no positive cell in 375; the selective entry policy is zero at the mid with the
spread removed entirely; the exit is indistinguishable from a stopwatch; and the drawdown-ordered
experiment closed at its own preflight because the corpus could not certify an edge of any believable
size. The bar you are designing toward is roughly twice the largest entry effect ever measured here.

The reason the work continues is narrow and specific: **large winners demonstrably exist** — 36.7% of
random entries exceed +50% excursion, and perfect foresight keeps about +$623 per trade at the mid —
**and no design has yet been given features capable of separating them before entry.** That is the
gap you are being asked to close. If your honest conclusion is that the available features cannot
close it, say that; it is a more valuable answer than a design that reaches the fit and discovers V5's
failure a third time.
