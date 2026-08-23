# Amendment to the two-skill exit law

> **STATUS: SIGNED AND IN FORCE. Owner signature 2026-08-22, with TIER A selected.**
> Drafted the same day at the owner's direction following the cold review
> [`EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md`](../research/findings/EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md).
>
> **This amends `PLAN.md` phase 5. Where the two disagree, this file governs**, and the superseded
> clause is quoted in §2 so nothing has to be reconstructed from memory.
>
> *The filename carries no `DRAFT` on purpose.* This repository has already paid for a signed
> document whose own header still read "Nothing here is adopted" for a week, and whose filename could
> not be repaired because a semantic freeze pinned it. Nothing pins this file, so it is named for
> what it is.

## 1. Why an amendment is unavoidable

`PLAN.md` phase 5 requires both of the following, and they cannot both hold:

- each statistic measured **against a duration-matched control**; and
- **always-cut posting high loss-averted with near-zero capture, always-hold the reverse.**

Always-cut exits at minute 1 on every path, so a duration-matched control exits at minute 1 too and
is **identical** — its control-relative loss averted is **exactly zero, not high**. Always-hold gives
the same at minute 60. Permuting a constant changes nothing. **No sample size repairs this.**

The law is therefore not merely demanding; **it is not constructible.** Until it is amended there is
no exit-evaluation law to build against.

## 2. What this amendment changes — the degeneracy expectation only

**Struck:** *"an always-cut rule should post high loss-averted with near-zero capture, an always-hold
rule the reverse, and neither pattern counts as skill."*

**Replaced with:** *"Both boundary rules must post **exactly zero** duration-adjusted skill on
**both** statistics. Their raw, un-adjusted defensive and capture profiles are reported separately as
diagnostics. Those exact zeros are the implementation test: a design that does not return them is
wrong, and a rule that does return them has demonstrated no timing skill."*

**Everything else in the signed law is preserved unchanged** — two skills never averaged into one
number, duration-matched controls, attribution-never-selection, serial executable P&L as the decision
criterion, and "beat holding" barred as a timing-skill metric.

## 3. What this amendment adds, because the signed text was silent

**3.1 Populations, on a 60-minute horizon using valid two-sided midpoints with no forward filling.**
- **Developed** — some `h ≤ 60` with `M(h)/M(0) − 1 ≥ 0.50`, **regardless of whether −30% came first.**
- **Did not develop** — all 60 midpoints observable and none reaches +50%.
- **Unknown** — no +50% touch observed and the path is incomplete.

**This is not the existing first-touch label**, which stops scanning at −30% and so merges "lost
first, then recovered" with "never developed". Measured on the closed 644: **250 paths ever reached
+50% against 201 first-touch winners — reusing the label would misclassify 49 trades.** `UNKNOWN`
stays in serial P&L, is excluded from attribution, and is reported. **Population membership is
attribution only and may never become an entry filter.**

**3.2 Denominator.** Available executable gain is `A = max over h of max(X(h), 0)` in the existing
net-USD convention. **Developed paths with `A = 0` must be counted and reported**, never dropped from
the capture denominator.

**3.3 Four numbers, never averaged**: raw defensive profile; duration-adjusted defensive skill;
absolute capture efficiency; duration-adjusted capture skill. Plus **untruncated** net P&L on
developed paths — clipping at zero is right for "gain captured" but must not hide losses.

**3.4 Baselines: five, each with one role.** Always-hold is the **economic incumbent** (serial P&L
must beat it). The duration-matched randomized exit is the **attribution control**. The bracket is a
**target-coherence diagnostic only** and may never be the economic comparator. Always-cut is the
**degeneracy control**. The oracle is the **opportunity denominator** and is never tradable.

**3.5 Four chronological session roles**, both tickets from a session staying in one partition:
entry fitting → independent entry-survival validation → exit fitting on a frozen, *passed* entry
policy → an **outer exit holdout** untouched by any fitting, scaling, matching design, control seed,
threshold or stopping decision. **A post-hoc split of the existing 644 is invalid** — every row
trained the head.

**3.6 Settlement.** The 2026-08-22 twin discharges the **60-minute bracket only**. Any rule able to
exit later than the bracket must carry settled and zero-recovery versions through both attribution
statistics, serial P&L, **and the power calculation.**

## 4. The minimally worthwhile effect — OWNER DECISION, and it decides feasibility

No minimum effect was ever signed, so no power target was ever definable. Two tiers, and **choosing
between them is choosing whether this study can be run at all.**

| Tier | ΔLoss-averted | ΔCapture | Outer sessions | Outer holdout alone |
|---|---:|---:|---:|---:|
| **A — economically derived (recommended)** | **+$25/trade** | **+5 points** | ~2,560 | **~10.2 years** |
| B — detectability-driven | +$50/trade | +10 points | ~640 | ~2.5 years |

**Recommendation: Tier A, and accept what it implies.** Tier A is roughly the floor at which the exit
changes the account — about +$25/trade against an always-hold incumbent of −$1.74. Tier B is not a
better answer to the same question; it is a decision to only ever detect a very large effect, chosen
because it fits the sample. **Setting the bar to fit the available data is the pattern this project
forbids**, and the honest output of Tier A is a finding rather than a study:

> **On this program's data budget, exit-timing skill is not measurable.** At ~252 reserved sessions a
> year, Tier A needs about a decade of outer holdout *on top of* separate entry-validation and
> exit-training samples.

**And the binding constraint is one of our own laws.** The signed **two-tickets-a-day risk law** caps
trajectories at 2 per session; it is what turned 50,318 raw trajectories into 644. More tickets would
make the exit question answerable and would breach the risk law. **The risk law and the exit-skill
question are in direct tension, and the risk law should win** — but the owner should know the trade
is real rather than discover it later.

## 5. What signing this does and does not do

**Does:** make the two-skill split constructible, so a future exit study has a law to satisfy.

**Does not:** reopen either closed configuration; authorize any fit; create a holdout; or imply an
exit study is feasible. Sections 1–3 make the law coherent. Section 4 says that, once coherent, it
probably cannot be satisfied on this corpus — and that is the useful answer.

## 6. Signature

- [x] **Owner signature and date: repository owner, 2026-08-22.**
- [x] **Tier selected (§4): TIER A — +$25/trade loss-averted, +5 capture points.**

**What the owner signed, stated plainly so it is not softened later.** Tier A was chosen over the
reachable Tier B *knowing* its consequence, which is not a study but a finding:

> **On this program's data budget, exit-timing skill is not measurable.** Tier A needs roughly
> **2,560 outer-holdout sessions — about 10.2 trading years** — on top of separate entry-validation
> and exit-training samples, at ~252 reserved sessions a year.

This is a deliberate refusal to shrink the bar to fit the sample. **No future session may reopen the
tier because Tier A proved inconvenient.** Moving to Tier B, or to any weaker effect size, requires a
fresh owner signature that explicitly acknowledges it is choosing to detect only a large effect.

**The exit-skill question is therefore parked on evidence, not abandoned in confusion.** The law is
now constructible; the sample to satisfy it does not exist and will not for years.
