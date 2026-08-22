# The signed two-skill exit law cannot be built as written

**Cold adversarial review, 2026-08-22 (Codex/Sol, substituted for Fable by owner decision).
Verdict: NO — not as signed. Verified independently by the executing session; every checkable
number reproduced exactly.**

## 1. The finding in one line

The two ideas behind the law are measurable — *did the exit protect bad entries?* and *did it keep
the gains on good ones?* — but the signed **combination** of duration matching with the required
always-cut / always-hold behaviour is **mathematically inconsistent. No sample size repairs it.**

This reopens nothing. Both configurations stay closed by the last two rows of
[`DO_NOT_RETEST.md`](../history/DO_NOT_RETEST.md), and the learned exit still loses to always-hold by
$16.41 a trade on the very trajectories it was trained on.

## 2. What breaks, exactly

`PLAN.md` phase 5 requires each statistic to be measured **against a duration-matched control**, and
separately requires that **always-cut post high loss-averted with near-zero capture, and always-hold
the reverse.** Those cannot both hold.

- **Always-cut exits at minute 1 on every path.** A control matched on duration must also exit at
  minute 1 — no duration is shorter. Its outcome is therefore *identical*, and control-relative loss
  averted is **exactly zero**, not "high".
- **Always-hold exits at minute 60 on every path.** Same argument: the matched control is identical
  and **both** its control-relative statistics are zero — not the required reverse pattern.
- **Permuting durations does not rescue it.** Permuting a constant changes nothing.
- **Comparing always-cut against always-hold** *would* give the intended raw profile, but it abandons
  duration matching and reinstates the exact "credit for holding time rather than deciding" that the
  law forbids, citing ledger row 341.
- **Always-hold does not definitionally capture an excursion** either: a path can touch +50% at
  minute 10 and finish worthless at minute 60.

The coherent reading is that **both boundary rules should report exactly zero incremental timing
skill**, with their raw defensive and capture profiles shown separately. That is a different
requirement from the one signed, and changing it is an owner amendment.

## 3. Why the existing artifacts cannot construct the statistics anyway

Independently verified against the archive:

| Claim | Verified |
|---|---|
| `exit_stream` holds only final rule P&Ls, timing and the oracle — no path, no entry midpoint, no "ever reached +50%" | **Confirmed**, 10 columns |
| The entry stream is a different scoring population | **Confirmed: 50 of 644 keys join** |
| The exit head has no holdout — all 644 trained it and all 644 were scored | **Confirmed** |

**The first-touch label is not the "did develop" population.** It stops scanning when −30% happens
first, so label 0 merges *"lost first, then recovered"* with *"never reached +50%"*. The review's
read-only reconstruction finds **250 of 644 paths ever reached +50% against 201 first-touch winners
— reusing the label would misclassify 49 trades.** One path became unobservable without proving a
+50% touch, which needs an explicit `UNKNOWN` population; it cannot honestly be called
"did not develop".

## 4. A correction to this project's own exit reporting

The 2026-08-21 receipt reported **"mean held minutes 60.0"**. That field is the *available path
length*, not the learned rule's realised holding time. Corrected and re-verified:

| Quantity | Reported | Correct |
|---|---:|---:|
| Mean realised holding time | 60.0 min | **54.44 min** |
| Trades running the full path | — | **498 of 644 (77.3%)** |
| Trades selling **strictly before** the final row | 23.4% | **146 of 644 (22.7%)** |
| Identical to always-hold | 79.5% | **512 of 644 (79.5%)** — unchanged |

The 23.4% figure counted 151 rows where a SELL fired *anywhere*, including five that fired on the
final row, which is forced liquidation rather than a decision. **The degeneracy conclusion is
unchanged and slightly strengthened.** Duration matching is therefore *nearly* vacuous on this
population rather than literally an all-60 distribution — which does not rescue the contradiction in
§2, because that argument is about constant-duration *rules*, not about this sample.

## 5. The baseline ruling: several, not one

The question "bracket or always-hold or duration-matched control?" was malformed. They answer
different questions and all are needed:

| Baseline | Proper role |
|---|---|
| **Always-hold** | **Economic incumbent** — the best measured fixed exit at −$1.74/trade. Serial P&L must beat it, even though "beat holding" is not an adequate *timing-skill* metric. |
| **Duration-matched randomized exit** | **Attribution control** — did the model pick useful *moments*, beyond holding a different length? |
| **Bracket** | **Target-coherence diagnostic only.** It cannot be the economic comparator: it loses to hold by $22.44/trade and was the target that fitted the failed entry. |
| **Always-cut** | Degeneracy control. |
| **Oracle** | Opportunity denominator and upper bound. Never a tradable comparator. |

## 6. The smallest coherent amendment

Implementable, but **not compliance with the current signed wording** until the owner amends the
degeneracy expectation. Summarised; the full specification is in the review.

- **Populations**, on a 60-minute horizon using valid two-sided midpoints with no forward filling:
  **developed** = some `h ≤ 60` with `M(h)/M(0) − 1 ≥ 0.50`, *regardless of whether −30% came first*;
  **did not develop** = all 60 midpoints observable and none reaches +50%; **unknown** = no touch
  observed and the path is incomplete. `UNKNOWN` stays in serial P&L, is excluded from attribution,
  and is reported. Population membership is **attribution only and may never become an entry filter.**
- **Executable series** in the existing net-USD convention; available gain
  `A = max_h max(X(h), 0)`. Developed paths with `A = 0` must be **counted and reported**, never
  silently dropped from the capture denominator.
- **Duration control**: freeze the rule, declare **pre-entry-only** strata (era, side, coarse
  entry-time bucket — never realised development status), permute realised durations *within*
  stratum, freeze ≥1,000 control seeds, and recompute controls inside every session bootstrap
  replicate. **A stratum with no duration variation has unmeasurable timing skill there, and that
  must not be manufactured by changing the comparator.**
- **Four numbers, never averaged**: raw defensive profile; duration-adjusted defensive skill;
  absolute capture efficiency; duration-adjusted capture skill. Plus untruncated net P&L on developed
  paths, because clipping at zero is right for "gain captured" but must not hide losses.
- **Four chronological session roles**: entry fitting → independent entry-survival validation → exit
  fitting on a frozen, passed entry policy → an outer exit holdout untouched by any fitting, scaling,
  matching design, control seed, threshold or stopping decision. **Both tickets from one session stay
  in the same partition.** A post-hoc split of the present 644 is invalid — every row trained the head.

**Worked sanity check, which is also the proof of §2.** A non-developing path `[−$10 … −$100]`: raw
loss averted for always-cut is +$90, its duration-matched control also cuts at minute 1, so
`ΔLA = 0` exactly. A developing path `[0, +$100, … 0]`: available gain $100, always-hold's realised
capture is 0%, its matched control is identical, so `ΔCE = 0`. **Those exact zeros are the
implementation test and the contradiction, in the same two lines.**

## 7. Power: the honest number is zero, then years

- **For the signed requirement the required `n` is not "large" — it is undefined.** No finite sample
  repairs a contradiction.
- **For the present run the exit-evaluation sample is zero**, because all 644 trajectories trained
  the head.
- The geometry is **323 session clusters, not 644 independent trades**; within-session dependence
  puts the learned-minus-hold effective sample near **385**.
- Non-authoritative scale checks for an outer test: all 323 sessions detect roughly **$42/trade** and
  **14 capture points**; half reserved detects **$59** and **20 points**. To recover $50 and 10
  points jointly needs about **640 outer sessions**; $25 and 10 points about **908**; $25 and 5
  points about **2,560** — roughly **2.5, 3.6 and 10.2 trading years** of outer holdout alone, on top
  of separate entry-validation and exit-training samples.
- **The 50,318 uncapped trajectories cannot inflate `n`**: they violate the executable two-ticket
  policy and remain heavily session-clustered. Reusing the opened 1,014-session corpus is development
  evidence, never confirmation.

## 8. Settlement

The 2026-08-22 twin discharges **the exact 60-minute bracket**, not a learned exit that can stay open
past the point the bracket would have sold. Any future learned policy, always-hold comparator or
randomized control able to exit later **must carry settled and zero-recovery versions through both
attribution statistics, serial P&L, and the power calculation.** At a measured **39.67%** settlement
share of matrix cells this is not ceremonial, even though the bracket's own exposure was 0.00%.

## 9. Document conflicts, verified and left visible

- **[`AGENTS.md`](../../AGENTS.md) §7 says "there is no confirmation firewall left"; `STATUS.md`
  says a forward confirmation reservation is SIGNED and in force, reserving every session from
  2026-08-06 onward.** **STATUS wins** by its own precedence rule, and the two are reconcilable —
  the *historical* holdout is spent, a *forward* reservation exists — but AGENTS' blanket sentence is
  wrong as written. **This is load-bearing, not cosmetic:** the forward reservation is the only
  mechanism by which the outer holdout sessions in §7 could ever exist.
- **[`SETTLEMENT_SOURCE_LAW_2026_08_22.md`](../../governance/SETTLEMENT_SOURCE_LAW_2026_08_22.md)
  §2.3 says backfill sessions end at 15:58.** That was true of the V4 acquisition and is **superseded
  by V5**, which restored the terminal bar on **788 of 794** sessions. Corrected in place; the
  operative law — *read the grid from the data, never assume* — is unchanged and was always the point.
- The entry finding's body calls the ordering effect settled; its §12 downgrades it to exploratory
  with causal significance `UNKNOWN`. The §12 reading governs.

## 10. What is wrong with the requirement itself, beyond the contradiction

"Did not develop" is not the existing label-0 population; missing paths create a third state the
binary wording omits; always-hold does not definitionally capture a transient excursion; "fraction of
gain available" is undefined without an exact executable denominator and a zero-denominator rule;
duration matching has little identifying variation here; realised-outcome groups support attribution
only and can never justify entry selection; per-trade attribution cannot replace serial account P&L,
ruin probability and time-to-ruin; and **no minimally worthwhile effect sizes were ever signed, so an
exact power target was never definable.**

## 11. Verification note

The executing session independently reproduced every checkable claim from the archive: the 50-of-644
join, 54.44-minute realised duration, 498 full-length trades, 146 strict early exits against the
23.4% previously reported, 512 always-hold-identical trades, the missing columns, and both document
conflicts. Nothing required correction. **Two of the corrections are to this project's own prior
reporting, not to the review.**

**Any change to the signed requirement belongs in a separate owner-signed governance amendment. This
finding records that the change is necessary; it does not make it.**
