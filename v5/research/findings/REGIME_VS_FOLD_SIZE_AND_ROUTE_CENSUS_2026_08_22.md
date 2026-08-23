# Job 46 regime versus fold size, and the route census — 2026-08-22

## Decision

**Verdict: `MIXED_NOT_SEPARABLE`; do not restart a fit on the current corpus.**

Two different gradients had been conflated:

1. The Phase-4b ordering gradient cannot distinguish chronology/regime from expanding-window fit
   size. Its four models train on 81, 162, 243 and 324 sessions while their holdouts move forward in
   time. Those two quantities have rank correlation **1.000** by construction.
2. The Phase-5 entry gradient is fit-size-free: one 118-parameter model, fitted once on the same
   405-session prefix, produced all five later score blocks. Its recent-minus-earlier dollar-ordering
   contrast is **+$7.2498/entry**, so expanding training-window size cannot cause that contrast.
   But the two “recent” blocks are exactly the two owned-data blocks. The recent indicator and the
   source-era indicator are identical, and the corpus has **zero same-session dual-source overlap**.
   Market regime and data-era/liquidity mechanics are therefore not identified.

The current Member-P entry remains economically negative in all five blocks; the learned exit is
degenerate; and the target used for entry is the bracket outcome even though always holding beats
that bracket by **$22.4457/trade**. The strongest defensible decision is to **terminate the current
fit cycle, preserve Member Q without running it, and preserve the forward confirmation reserve**.

If the owner declines the terminal branch, the only mechanism worth designing is a **conditional
within-minute contract ranker under an executable clock-hold target, with a separately specified
entry gate**. It is not authorized to run now, and the available power numbers make a modest effect
impractical to certify.

## Scope and governance

This is a continuation of registered job 46, not a new project. Before the receipt synthesis, the
non-fit precommit was frozen at
[`REGIME_FOLD_DIAGNOSTIC_PRECOMMIT_2026_08_22.json`](../../work/lifecycle-training/REGIME_FOLD_DIAGNOSTIC_PRECOMMIT_2026_08_22.json),
SHA-256 `0aa07718751b34f6b339df2ccfe178d0bff4ad8bc0ae1a4477130f596539c1d7`.
It is explicitly **not a fit declaration and has no permit authority**.

The synthesis:

- verified hashes and recomputed arithmetic only from already-published receipts;
- opened no corpus table, outcome column, reserved session, or model target;
- fitted no model and created no new outcome comparator;
- did not edit the protected alpha ledger, any declaration, the semantic freeze, the settlement law,
  the unsigned exit amendment, or `DO_NOT_RETEST.md`;
- does not upgrade the undeclared entry ordering result into independent confirmation.

The exact archived attempt-001 wrapper stopped on a Python boolean typo before writing a result. It
is preserved byte-for-byte with a failure receipt. Attempt 002 repaired only those boolean aliases,
used a new filename, and wrote the successful receipt.

## The precommitted discrimination

### 1. Fixed-model score-period contrast

The Phase-5 entry producer calls `train_entry_phase` once on the 405-session prefix ending
2024-01-29, then closes over the same model while streaming all five fit-forbidden score blocks.

| Score block | Dates | Source era | Dollar ordering edge | Binary ordering edge |
|---:|---|---|---:|---:|
| 0 | 2024-01-30–2024-07-29 | backfill | +$7.2486 | −0.1637pp |
| 1 | 2024-07-30–2025-01-29 | backfill | +$5.8767 | −0.8556pp |
| 2 | 2025-01-30–2025-07-31 | backfill | +$5.1576 | −0.5390pp |
| 3 | 2025-08-01–2026-02-02 | owned | +$11.8751 | +0.2704pp |
| 4 | 2026-02-03–2026-07-30 | owned | +$14.8132 | +0.8535pp |

Earlier blocks 0–2 average **+$6.0943**; recent blocks 3–4 average **+$13.3441**. The contrast is
**+$7.2498/entry**, and each recent block exceeds every earlier block. This clears the precommitted
+$4 materiality rule. **Expanding fit size is excluded for this fixed-fit score-period contrast.**

The binary contrast is much smaller: **−0.5194pp** earlier versus **+0.5619pp** recent, a
**+1.0814pp** change. At the exact source seam, block 2→3, the dollar edge jumps **+$6.7175** while
the binary edge moves only **+0.8094pp**. That is below both Phase 4b’s already-published 2.5pp
materiality floor and its 3.7874pp null p97.5. Those thresholds are not a p-value for this undeclared
entry diagnostic; they show scale. Under Phase-4b-like geometry, simple square-root scaling puts a
1.081pp contrast at roughly **4,000 sessions** and the 0.809pp seam contrast at roughly **7,100
sessions**. The existing binary evidence does not corroborate a large label-ordering regime shift.

### 2. Phase-4b chronology versus fit size

| Holdout fold | Expanding training sessions | Ordering component |
|---:|---:|---:|
| 0 | 81 | +0.7355pp |
| 1 | 162 | +9.5966pp |
| 2 | 243 | +9.3999pp |
| 3 | 324 | +13.7139pp |

Training size and holdout chronology move together perfectly. Phase 4b saved no row-level OOF score
artifact and no common-model/common-period comparison. Separating them now requires a new fit, which
is prohibited without a genuinely prospective declaration and `DeclaredFitPermit`; writing either
merely to unblock the comparison is also prohibited. **Verdict: `NOT_SEPARABLE_EXPANDING_WINDOW_VS_CHRONOLOGY`.**

Phase 4b’s by-year gradient (+2.44pp in 2022, +9.10pp in 2023, +18.74pp in the small 2024 tail)
occurs entirely within the backfill source and therefore argues against the 2025-08 source seam as
the sole explanation. It still cannot distinguish chronology from expanding fit size; the 2024
endpoint contains only 38 selected rows.

### 3. Recent market regime versus source era

The fixed-model score design is:

```text
recent indicator:  0 0 0 1 1
owned-data era:    0 0 0 1 1
```

There is no same-session observation under both quote roots. The design matrix is rank-confounded,
so no statistic on these five blocks can uniquely assign the jump to “market regime” or “source
era.” **Verdict: `NOT_IDENTIFIABLE_ON_CURRENT_DATA`.**

The pattern is also seam-shaped rather than a smooth recent trend: dollar ordering declines
+$7.25→+$5.88→+$5.16 through the three backfill score blocks, jumps at 2025-08-01, then rises to
+$14.81. The fixed model’s entries/session simultaneously fall 213.5→172.2 across the seam, so its
selected population also changes.

## Artifact census

The published controls narrow the ambiguity, but do not remove it:

- **Settlement valuation is excluded for Member P.** In the 24-session settled/zero-recovery twin,
  39.67% of exit-matrix cells use settlement, but 0 of 4,856 model-fired bracket exits do. Every
  stream and the +$9.86 ordering edge move by exactly **$0.00**. This does not discharge Member Q’s
  120-minute horizon.
- **Member-visible ladder width is stable.** The member sees a median 20 contracts and 45-point span
  in both eras. The wider full chain is not in its input.
- **Coverage is not the jump.** Valued coverage is 99.91%, 99.93%, 99.99%, 99.94%, and 100.00% across
  the five score blocks.
- **The corrected state is weakly era-classifiable.** The latest SPX-tape full-state AUC is about
  0.432, dispersion alone about 0.455, and the other chain channels about 0.481, below the declared
  ≈0.55 concern bar. This is a linear lower bound, not proof of invariance.
- **Label composition is comparatively stable.** Phase-4b foldwise population
  `P(gain first | resolved)` spans **2.4157pp**. The fixed-entry population hit rate spans 2.9570pp,
  and changes only +0.6294pp at the source seam.
- **Liquidity remains live.** Spread as a percentage of premium is published as
  **3.03/3.64/3.03/2.25/1.84%** across 2022→2026. The Phase-4 label is mid-defined; dollar economics
  are ask-in/bid-out and cost-sensitive. This is the strongest existing non-market explanation of
  the dollar jump.
- **The target is economically misaligned.** The 644 first-two trajectories realize bracket
  −$24.1825/trade versus always-hold −$1.7368, a **+$22.4457** advantage for holding. Yet
  `entry_value_usd` is the bracket outcome. Always-hold still loses, so changing the target is a
  reason to redesign—not evidence that a profitable bot exists.

## Ranked, costed, powered route census

The ranking is by decision value under the present evidence, not by novelty. “Alpha” means attempts
against the protected append-only ledger. It currently has six attempts; the next directional
accuracy bar is **0.66059463**, with true accuracy about **0.68659463** needed at the ledger’s
405-session geometry. That accuracy bar is not an economic-P&L bar and must never be substituted for
one.

| Rank | Route and present ruling | Power and route-finishing result | Alpha / compute / data / owner cost |
|---:|---|---|---|
| **1** | **Terminate current-corpus fits; preserve evidence. RECOMMENDED.** Member P and the exact exit are closed, all five entry blocks lose, the target is wrong, and attribution is not identified. | The precommitted current-route stop is met now: fixed-fit materiality exists but both causal attributions are rank-confounded and no published orthogonal comparison breaks either alias. Finish as a defensible negative, not a broad “options contain no information” claim. | **0 alpha, $0 data, 0 fit compute.** Owner cost: accept the terminal branch and keep protected files unchanged. |
| **2** | **Forward accumulation under the signed confirmation-only reservation. SURVIVES ONLY AS VALIDATION INFRASTRUCTURE.** It cannot be used to tune a policy. | The only published economic power anchor under two tickets/day is ~640 outer sessions (~2.5 years) for a very large +$50/trade and +10-capture-point effect, or ~2,560 (~10.2 years) for the economically derived +$25/trade and +5-point floor. The ledger’s 405-session geometry is ~1.6 trading years but does not by itself power economic P&L. Finish a frozen policy only when forward, session-unit serial P&L has a corrected lower bound above zero, clears its signed minimum effect, and passes ruin/time-to-ruin law; otherwise terminate it. | Collection itself costs **0 alpha**. Future confirmation costs the experiment declared for the frozen policy. Local compute is modest; future quote-data availability and monetary cost are **unknown and currently unauthorized**. Owner cost: preserve the 2026-08-06 reservation. |
| **3** | **Conditional contract ordering with a separate entry gate. Highest-mechanism candidate if the terminal recommendation is rejected.** The current model weakly ranks contracts (+$8.43 exploratory edge; Spearman +0.102) but its absolute level fires ~209 times/session and cannot gate against WAIT. | A prospective route must freeze the ranking target, gate, statistical unit, controls, and serial simulator before outcomes. Use the +$25/trade outer requirement as the honest floor: ~2,560 confirmation sessions under current risk geometry; +$50 still needs ~640. Finish only if a forward policy beats no-trade and its exact causal control with corrected session-level lower bounds, meets the minimum effect, and passes account-risk gates. | **At least 1 alpha attempt**; more if ranker and gate are separately tuned. One current-scale priced corpus build is ~21 minutes (1.24s × 1,014) plus a measured 0.57-minute fit; OOF/gate cost needs a dry cost probe. Data $0 on development corpus, but no independent historical holdout remains. Owner must authorize the new objective and a genuine declaration/permit. |
| **4** | **Replace the bracket target with an executable 60-minute clock-hold target. Mechanically justified, insufficient alone.** It addresses the $22.4457 target error, but hold itself is −$1.74/trade and a level estimator still needs a separate gate. | Same outer economic floor as rank 3. Using the published 323-cluster ≈$42/trade detectable scale and square-root scaling, the observed $22.4457 target delta would need roughly **1,132 sessions at 80% power or 1,568 at 90%**; this is a planning proxy, not a valid target-specific gate. A target-only fit does not finish the route. It finishes only when combined with a frozen gate and a forward policy whose session-level P&L lower bound exceeds zero and the signed +$25/trade minimum. | **At least 1 alpha attempt**, ~one current-scale entry pass, no new historical data cost. Owner must freeze horizon, delayed-bid/settlement law, zero-recovery treatment, gate, and minimum effect. Any same-corpus result remains development-grade. |
| **5** | **Same-date source twin, then recent-era training only if the twin clears. BLOCKED, NOT A CURRENT FIT ROUTE.** Recent-only retraining on the same bracket target would repeat the closed failure and has no independent recent holdout. | Current overlap is **0**, so paired power is undefined until a source-twin variance is available. The frozen identification floor requires a same-date source effect bounded to ±$2/entry, a within-source regime lower bound above +$4/entry, a same-direction label contrast ≥0.5pp, and ≤2.5pp label-composition drift. For spending, a stricter existing-evidence anchor is ≥$12.05/entry (half the $24.09 break-even gap) and ≥3.787pp binary contrast; the observed +$6.72/+0.809pp seam jump does not clear it. Only then could a recent-era fit be considered, and it would still need forward confirmation. | **At least 1 alpha attempt for a verdict-bearing source diagnostic and another for a fit** if it proceeds. A 243-session recent build is roughly five minutes before scoring, but a per-fit capacity statement is mandatory. Dual-source data price is **unknown; vendor contact/download is unauthorized**, so the effective current data budget is $0. Owner must authorize any preflight/acquisition separately. |
| **6** | **Member Q. PRESERVE, DO NOT RUN NOW.** It is the only fully specified member predating the outcomes, but only 0.72% of owned-year actions beat WAIT; median enter is −$173 versus wait +$1,437; P measured −$28 across ~1.3M actions; Q may not be promoted over P on point estimates. | Action count is not power: the unit is the session and Q’s session-clustered rare-positive count has not been published. Therefore it is **not yet powered**. Before any run, a non-outcome capacity/power statement must establish adequate independent positives. Finish only if settled and zero-recovery versions agree in sign, corrected session-level economics clear the declared bar, and serial risk gates pass. | **1 alpha attempt**, compute unmeasured for 120-minute targets, historical data $0. Owner must explicitly reopen the corpus and issue a genuine source-compiled declaration/permit. The zero-recovery twin is mandatory and un-discharged. |
| **7** | **Free date-arithmetic features. NO FEATURE FIT.** FOMC is already a sit-out rule. Quarterly OPEX, month end, last Friday, monthly OPEX, month and weekday cost no data, but the feature contract bars them. | Existing strata are only FOMC 33, quarterly OPEX 17, month end 50, last Friday 50, monthly OPEX 50. They cannot independently power a new bot; monthly OPEX has no step elevation. A single ex-ante gate could finish only on forward data under the rank-3 economic law. | Raw feature data **$0** and negligible compute; any outcome test still costs **1 alpha attempt** and any model admission needs an owner-approved parameter-contract change. No vendor spend. |
| **8** | **Higher occupancy than two tickets/day. TERMINATE UNDER CURRENT LAW.** The prior ticket-widening amendment was already refused and charged. More paths do not create independent sessions and increase account risk. | The signed policy yields 644 trades in 323 clusters; 50,318 uncapped trajectories cannot inflate `n`. Uncapped exit training was estimated at **~115 hours** versus 0.4 minutes for the capped head. Power still requires ~640 or ~2,560 independent outer sessions. No current occupancy level can finish the inference route. | At least **1 additional alpha attempt** after an owner-signed risk-law amendment; ~115h compute at the observed firing rate; $0 historical data. Owner cost is a substantive risk-policy change, not an implementation tweak. |

No row authorizes a fit, data purchase, download, vendor/broker contact, paper/live order, or edit to a
protected file.

## Owner decision summary

Recommended owner ruling:

1. **Accept `STOP_CURRENT_CORPUS_FITS`.** Do not run a recent-era fit, target refit, Member Q, calendar
   feature fit, or ticket-widening experiment now.
2. **Preserve Member Q** exactly as `PRESERVED, NOT RUN`; do not burn its uniquely prospective
   specification on an underpowered same-corpus attempt.
3. **Preserve the forward confirmation-only reservation.** It is the only remaining source of
   independent evidence.
4. **Leave the unsigned exit amendment unused for now.** If the owner later rules on its effect tier,
   Tier A (+$25/trade, +5 capture points) is the economically honest choice, and its consequence is
   “not measurable on this data budget,” not permission to lower the bar to Tier B.
5. If one future design is retained on paper, retain only **clock-hold-relative contract ranking plus
   a separately frozen gate**, and require the rank-5 source-identification precondition before
   restricting training to the recent era. Do not create a declaration merely to make it runnable.

This is the defensible evidence that no current route is ready. It is narrower than “SPX 0DTE longs
can never work”: chain state appears to contain some conditional ordering information, but this
corpus cannot identify its temporal cause, the current target/gate cannot monetize it, and the
independent sample needed for a modest effect is measured in years.

## Binding unknowns and inherited conflicts

- No same-date dual-source quotes exist, so source effects cannot be paired away.
- The entry stream does not preserve model scores, controls, spreads, or per-session control draws;
  the +$8.43 control and its inference law were undeclared.
- Phase 4b saved no row-level OOF scores, so the common-model/common-period question cannot be
  reconstructed without fitting.
- No clock-hold entry target, independent gate, entry-specific minimum worthwhile effect, or
  route-specific economic power calculation is signed.
- Member Q’s session-unit rare-event power and 120-minute compute cost are unmeasured; its settlement
  twin remains binding.
- Availability and price of dual-source overlap or future quote data are unknown; current authority
  permits neither contact nor acquisition.
- The Phase-4a historical archive has known missing producer wrappers. This work does not rewrite or
  retroactively repair that record.
- The work-allocation memo separates design, execution, and judgment. This task used independent
  read-only governance, regime-evidence, and route-census inventories; the only executed step was a
  frozen receipt synthesis, and the final ruling is presented as an owner decision rather than a
  hidden protocol change.

## Evidence and archive placement

- Precommit:
  [`v5/work/lifecycle-training/REGIME_FOLD_DIAGNOSTIC_PRECOMMIT_2026_08_22.json`](../../work/lifecycle-training/REGIME_FOLD_DIAGNOSTIC_PRECOMMIT_2026_08_22.json)
- Attempt-001 archived wrapper:
  [`regime_fold_receipt_synthesis_2026_08_22.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/regime_fold_receipt_synthesis_2026_08_22.py)
- Attempt-001 failure receipt:
  [`regime_fold_receipt_synthesis_2026_08_22_attempt001_failure.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/regime_fold_receipt_synthesis_2026_08_22_attempt001_failure.json)
- Attempt-002 archived wrapper:
  [`regime_fold_receipt_synthesis_2026_08_22_attempt002.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/regime_fold_receipt_synthesis_2026_08_22_attempt002.py)
- Successful receipt:
  [`regime_fold_receipt_synthesis_2026_08_22_attempt002.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/regime_fold_receipt_synthesis_2026_08_22_attempt002.json),
  self-hash `e67da9bd3e223f352eb9724882ab58963fd50c91515027d4ec2e0102ecaf29ab`
- Published entry evidence:
  [`ENTRY_FIT_ORDERING_WITHOUT_SURVIVAL_2026_08_21.md`](ENTRY_FIT_ORDERING_WITHOUT_SURVIVAL_2026_08_21.md)
- Published exit power:
  [`EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md`](EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md)
- Member-Q and settlement law:
  [`SETTLEMENT_SOURCE_LAW_2026_08_22.md`](../../governance/SETTLEMENT_SOURCE_LAW_2026_08_22.md)

## Verification

- `./.venv/bin/python v5/ops/check_project.py`: **PASS** — project and repository OK.
- `./.venv/bin/python -m pytest v5/tests/test_outcome_run_gate.py -q`: **14 passed**.
- `./.venv/bin/python -m pytest v5/tests -q`: **1,123 passed**, 105 existing numerical warnings,
  process exit 0 in 64.50 seconds. Full output is retained for this session at
  `/tmp/v5-pytest-regime-fold-20260822.out`.
