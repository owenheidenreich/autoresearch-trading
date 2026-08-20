# Phase 4a adjudication — outcome D, with the discriminating test specified

**Adjudicated 2026-08-19 by Fable, cold, under §3 item 2 and §5 of the signed
[work-allocation memo](../../governance/WORK_ALLOCATION_MEMO_2026_08_18.md).** The referral is
[`PHASE_4A_FEATURE_INFORMATION_2026_08_19.md`](PHASE_4A_FEATURE_INFORMATION_2026_08_19.md); its §8
posed two questions and both are answered here. No new measurement was run for this adjudication —
everything below comes from the receipts, the pinned code, and arithmetic.

**The ruling in two sentences. On §8.1: outcome D — the +8.65pp lift is a real measurement of
*something*, but this run cannot distinguish the design's ordering hypothesis from the label's own
mechanical sensitivity to volatility and leverage, and one cheap, label-side decomposition settles
it; that test is specified in §5 with its verdict rule pre-declared. On §8.7: option 1 — the
mechanical PROCEED stands as published under the bar as declared, nothing is rewritten, and it does
not, by itself, authorize the 118-parameter fit.**

---

## 1. What was verified cold

Every number in the referral reproduces from the receipts: the headline (+8.649074789891253pp, 648
selections, base 30.70%, verdict statistic +13.33pp), the plant (+19.98pp of +20.0pp), the null
(0/20 at the point estimate, 7/20 uppers over the +4.0pp bar, max upper +5.96pp), the ablation
table, and the fold-by-fold coefficients. The declaration was sealed before numbers existed, was
resealed once for a docstring with no threshold moved, and its implementation hashes match the
files on disk today. The data path was checked at the code level: `CANDIDATE_COLUMNS` and
`LADDER_COLUMNS` contain no dollar column, so the "no economics" claim is structural, not
procedural. The score blocks were untouched. The Boundary-2 directed changes were carried and
re-measured (`implied_spot_dispersion_ratio`, era probe 0.4314 against the ≈0.55 bar).

One apparent anomaly was chased and is benign: several null draws repeat identical lift values, and
two null values coincide exactly with ablation values. All lifts are quantised on a grid of 1/648
(one hit out of 648 selections ≈ 0.154pp), so collisions across independent runs are expected.
Verified: every quoted lift sits exactly on the k/648 grid.

Two structural facts the referral does not state were found and both matter downstream:

1. **The "alone" ablation rows for minute-common groups are tie-break composites, not group
   properties.** `_top_per_session` ranks with a stable `np.lexsort`, so when every contract in a
   minute shares one score — which is exactly what happens when a probe contains only minute-common
   fields — the top-2 selection falls to frame order. Verified on the corpus rather than reasoned:
   the candidates table is sorted `(entry_minute, contract_id)` and `contract_id` is
   `SPXW-<date>-<strike>-<C|P>`, so ascending order is **ascending strike**, and since eligible
   contracts are OTM, the lowest strikes are **the deepest-OTM puts — the cheapest and most
   leveraged contracts in the band**. On 2022-06-01 at 10:00 the tie-break takes puts at −22.2 and
   −17.2 points for $880 and $1,030, where the nearest-ATM contract is −2.2 points at $1,640.

   Exactly three groups are affected, because only they are wholly minute-common: **tape, clock and
   chain internals**. Their "alone" rows measure *that group's minute choice crossed with one fixed
   contract pick*, and that pick is a poor one — the tape-alone row implies a precision of **21.30%**
   against a 30.70% base, which is the deep-OTM lottery ticket's own hit rate, not a statement about
   the tape. The other three groups (`contract_price`, `contract_chain`, `ordering`) contain
   per-contract fields that break ties meaningfully and are clean, as is the entire marginal column.

   **This cuts against outcome B, which is why it is stated here rather than buried.** The referral's
   §8.5.2 rests partly on "the designed families contribute least, +1.86pp alone" — but that number
   is depressed by a contract pick the chain fields did not make, so the hypothesis families'
   standalone value is *unknown*, not *small*. The design's instruction that "the measured per-field
   information sets the shrink-ladder order" cannot be executed from these numbers at all.
2. **The three diagnostic receipts (`ablation`, `controls`, `null_distribution`) have no archived
   producer script.** The pinned library did the work, so the numbers are trustworthy, but the
   wrappers that called it are not on disk. Phase 4b (§5) must archive its wrappers with its
   receipts.

The referral brief itself was fair: both readings argued, the producing session's lean quarantined,
and every number it quotes accurate against the receipts.

## 2. Why not A, and why not C

**Not C.** The lift is not an artifact. The probe recovered +19.98pp of a +20.0pp plant, so it is
powered; zero of twenty within-session permutations approached +8.65pp, and the point estimate sits
+5.6 null standard deviations out; the declaration's hashes verified; the data path cannot see
dollars. Two *constructions* are defective — the verdict bar (§4) and the minute-common ablation
rows (§1) — but neither manufactures the headline number. Calling this "artifact" would be false
comfort: something real is being ranked.

**Not A.** Two independent reasons. First, the declared test cannot certify signal at this sample
size: a shuffled label clears the +4.0pp upper-bound bar in 7 of 20 draws, so PROCEED under that
bar carries almost no information about whether the fit is warranted — the point estimate is what
carries the evidence, and the point estimate was never given a declared bar. Second, and decisive
for A specifically: the entire attribution pattern is consistent with a mechanism that contains
**no ordering information at all**, laid out next.

## 3. The mechanical account, stated quantitatively — and what it would and would not mean

Member P's label counts a trade as a win only if the option's mid gains +50% before losing −30%
within 60 minutes. **A trade that never touches either line counts as a loss.** So the hit rate
factors exactly into two parts:

> P(win) = P(the path resolves — touches either line) × P(the gain line comes first, given it resolves)

Only the second factor is the design's hypothesis — cross-sectional *ordering*. The first factor is
pure magnitude: picking minutes and contracts that **move more in percentage terms** raises
resolution mechanically, with no opinion about direction, side, or which contract beats its
neighbours. And for a vol-dominated percentage path with barriers at +50/−30, the no-skill
gain-first share tends toward the barrier ratio **30/80 = 37.5%** — decay pushes it lower when the
path barely moves, which is exactly the case selection avoids.

Now the measured facts, read against that account:

- The observed selected precision is **39.35%** — within 1.85pp of the 37.5% no-skill benchmark.
- The one coefficient stable across all four folds is **`realised_vol_15m`** (+0.115 to +0.164),
  the vol channel.
- **Contract price alone scores +6.49pp** (per-contract fields, no tie-break contamination) — the
  cheap/high-IV leverage channel, three-quarters of the full lift by itself.
- The full set **without the tape group scores −0.76pp** — nothing survives without the vol channel.
- The hypothesis families score +2.62pp and +2.47pp at the margin — order of 1.5 null standard
  deviations. (Their *standalone* numbers are not evidence either way; see §1.)

This account also corrects the design's own text, and the error is mine by lineage: §5.3 of the
learning-content design claims the first-touch bracket "is a statement about *path order*, not
about magnitude." That was too strong. Because the label's zero pools loss-first with
never-resolved, **magnitude has a direct channel into this target**, and a vol-loaded ranker can
raise the hit rate substantially while learning nothing the premium has not priced. The design
promised a target that isolates ordering; it delivered a target that mixes ordering with
resolution. The probe may simply have found the mixture's larger ingredient.

**Why this does not establish outcome B either.** The account is quantitatively adequate but
unmeasured: the decisive quantity — P(gain-first | resolved) for the selected set against the
population — was never computed, and it is one groupby away from the columns the corpus already
stores. Three loose ends resist B: the hypothesis families' marginal contributions are small but
not zero; fold 0 (+1.70pp, scored on late-2022 sessions where the plant recovered at +25.9pp) is
unexplained under pure mechanics; and 39.35% exceeds the 37.5% benchmark by an amount the
decomposition would attribute cleanly one way or the other. Declaring B on a pattern-match to row
332 without running the decomposition would be the mirror image of the failure mode the memo exists
to stop — a session talking itself into a *negative* it finds congenial. The project's own rule
covers this exactly: a surprising result is a suspected bug **until its mechanism is proven**.
Proving the mechanism costs one label-side pass. That is outcome **D**.

## 4. The §8.7 ruling — option 1, with one prospective directive

**The PROCEED stands as the mechanical outcome of the rule as declared.** It was pre-registered,
applied without any threshold moving, and is published with the null behaviour beside it. Nothing
is rewritten. The declaration built the interval "generous to the features by declaration, so a
failure cannot be blamed on the interval" — a one-sided construction that makes a STOP unimpeachable
at the cost of making a PROCEED weak. The outcome landed on the weak side. Both halves of that
sentence go on the record together: **the verdict keeps its pre-registered meaning, and its
pre-registered meaning is now measured to be thin — a bar cleared by 35% of shuffled labels.**

**Consequently the PROCEED does not, by itself, authorize the fit.** That is not a retroactive
re-setting of the bar: the memo routed this exact situation to adjudication before any action, and
the adjudication (outcome D) is the action decision. The declaration's own rule that no secondary
probe may change the *verdict* is respected — the verdict is untouched; what waits is the spend.

**Option 2 — declare the test inconclusive and re-run under a null-calibrated bar — is rejected.**
The twenty permutations already establish what a recalibrated re-run would establish: the point
estimate sits +5.6 null standard deviations out and would clear any sanely calibrated bar. A re-run
would spend an experiment, and the alpha ledger's honesty, to re-learn a foregone conclusion while
leaving the actual open question — mechanism — untested. That is motion, not measurement.

**Prospective directive (Tier 2, binds future declarations, changes nothing retroactively).** Every
future preflight verdict statistic is calibrated against its own measured null **before** running:
at least 100 permutation draws of the same statistic through the same selection machinery, bar
stated relative to that distribution, and any deliberate one-sidedness in a rule must state which
error it protects against. The weak construction originated in the learning-content design's §5.2
— my lineage, not the producing session, which implemented it faithfully and then measured its
weakness unprompted.

## 5. Phase 4b — the discriminating test, specified for execution

Everything below is design judgement settled now, so what remains is execution (Opus, under memo
§2), pending the owner's confirmation. Scope: **training prefix only; label-side only** — the four
first-touch columns and no dollar column; a self-hashed declaration in the Phase-4a pattern;
alpha-charged; wrappers archived beside the receipts. First precondition: regenerate the declared
probe with seed 20260819 and require the headline to reproduce at exactly
+8.649074789891253pp — the run is deterministic, so anything else stops the phase.

**D1 — the verdict-bearing decomposition.** Define `resolved` = gain-minute finite OR loss-minute
finite (columns `first_touch_50pct_minute_60m`, `first_touch_loss_30pct_minute_50pct_60m`). For the
scored population and for the 648 selected actions compute P(resolved) and P(gain-first | resolved),
and split the lift into a **resolution component** — [P(resolved|sel) × P(gain-first|res, pop)] −
base — and an **ordering component** — precision(sel) − [P(resolved|sel) × P(gain-first|res, pop)].
Calibrate the ordering component's null with **at least 100** within-session label permutations
pushed through the identical fit-and-select path (count declared before running). Verdict rule,
pre-declared here and closed to reinterpretation:

| Ordering component | Outcome |
|---|---|
| ≤ its null 97.5th percentile | **B is established for this feature set on this label.** The lift is resolution mechanics — the magnitude family, already priced (row 332). Publish the negative under the design's own STOP language, with the mechanism named. The fit does not run on this feature set and label as designed. |
| > null 97.5th **and** ≥ +2.5pp | **A.** Ordering information exists and is material. The fit proceeds as designed, with the vol channel documented as conditioning and the D1–D8 suite unchanged. |
| > null 97.5th **and** < +2.5pp | Real but small. **Owner decision**, with both components and their nulls on the table. |

No other statistic may move the Phase 4b verdict.

**D2 — consistency control.** Within-session matching on entry-ask decile × `realised_vol_15m`
quintile: selected minus matched precision. Must agree in direction with D1; a contradiction stops
the phase and returns here — the producing session does not interpret it.

**D3 — diagnostics only, for the shrink order.** Re-run the group ablations with **randomized
tie-breaking** (pre-seeded, ≥20 tie-break draws averaged) so minute-common groups become
interpretable; probe `realised_vol_15m` alone and the full set minus it. These replace §1's
contaminated "alone" numbers wherever the shrink-ladder order is set. No verdict weight.

D1's components are also reported per fold and per calendar year: fold 0's near-zero lift is
unexplained under both readings, and the decomposition will show which factor vanishes there.

## 6. Out of scope, unchanged, and disclosure

Unchanged: the Phase 4a verdict and declaration; the signed risk law; the chronology; the
event-calendar and tape-source rulings; the 118-parameter member; every ledger row. Nothing here
opens economics.

Disclosure: the two design defects this adjudication corrects — the upper-bound bar construction
(§4) and the "path order, not magnitude" overclaim (§3) — both originate in the learning-content
design, which is Fable-lineage work. This ruling corrects its own line's errors, which should be
weighed when reading its refusal to accept either the congenial positive (A) or the congenial
negative (B) without the one measurement that separates them.

**One correction inside this document, recorded rather than silently edited.** §1's tie-break
finding first stated that the tie-break selects the two *nearest-ATM eligible calls*, "the most
expensive and least-leveraged contracts in the band". That was reasoned, not measured, and it is
wrong: `contract_id` sorts by strike, so the pick is the two *deepest-OTM puts* — the cheapest and
most leveraged. Checking it against the corpus reversed the direction of the bias and, with it, the
claim's consequence: the contaminated rows are depressed rather than inflated, which weakens the
referral's case for B rather than strengthening it. The ruling did not change; a supporting fact
did. It is left visible because an adjudication that hides its own corrections is worth less than
one that shows them.

---

## 7. Verification note, added by the executing session, 2026-08-19

Appended rather than edited into the text above, so the ruling stays as it was written. The ruling
itself is unaffected; one supporting fact inside §6's correction is.

**§6's correction is confirmed exactly.** `_top_per_session` sorts stably, so ties fall to stored
row order, and candidates are stored ascending by `contract_id`, which embeds the strike. On
2022-06-01 at 10:00 the pick is −22.23 and −17.23 points at **$880** and **$1,030**, against a
nearest-ATM contract at −2.23 for $1,640. Deepest-OTM puts, as §6 states.

**The mechanism is broader than one row, and this strengthens §6's conclusion.** Measured on the
probe frame: **tape, clock and chain-state are all minute-common — 16 of the 24 declared features
take a single value across every contract in a minute.** Those three groups cannot rank contracts at
all, so their "alone" numbers in the referral's §4 never measured contract selection; they measured
minute selection plus an arbitrary tie-break. That includes the chain-internal family, which is the
design's hypothesis. "Standalone value is unknown, not small" therefore holds for three groups, not
as a quirk of one.

**One supporting fact in §6 does not hold.** §6 reads the contaminated rows as *depressed* — the
tape-alone row implying 21.30% precision being "the lottery ticket's own hit rate". Measured over 81
prefix sessions (255,425 candidate actions), the rows a stored-order tie-break takes have a base rate
of **29.08% against 29.53% for all candidates — a difference of −0.45pp.** The deep-OTM tie-break
subset is not a depressed population, so the −9.41pp is not composition. It arises from *which
minutes* the tape-alone probe selects, conditioned on that subset.

**This cuts the same way as the ruling, by removing an explanation rather than adding one.** §6 used
"depressed" to argue that the contaminated rows weaken the case for B. They still do — but because
they measure nothing about contract ranking, not because their population is unfavourable. The
−9.41pp is now unexplained rather than explained away, and outcome D is unchanged.

**One execution constraint, recorded because the two D-tests pull against each other.** §5's D1
precondition requires the headline to reproduce at exactly +8.649074789891253pp, which requires the
current deterministic tie-break; D3 requires randomized tie-breaking. The seeded tie-break must
therefore be opt-in and must not touch the default path, or the precondition can never pass. It is
implemented in a **new module**: `feature_information_preflight.py` is pinned by the Phase-4a
declaration's implementation hashes, and editing it would invalidate a declaration whose result is
already published.
