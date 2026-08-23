# Job 46 two-era SPXW corpus audit — 2026-08-22

**Part 1 verdict: `NOT-USABLE`.** The current 1,014-session corpus cannot support a new certifiable bot
test or the signed economic claim as one pooled sample. It still contains three backfill sessions
that fail a new interior whole-book liveness check; after their conservative exclusion, source era
remains perfectly confounded with the 2025-08-01 time cutoff; and no effective-size measurement
exists for either cleaned source stratum or for the proposed clock-dollar target. Even raw session
counts are below the signed economic certification requirements. This is a substrate and
identification failure, not evidence that every possible SPXW 0DTE edge is absent.

What would change the verdict is concrete: rebuild or mechanically exclude the three liveness
failures and re-receipt the corpus with a fail-closed interior detector; either obtain paired
same-date source observations or explicitly restrict inference to one source era; obtain owner
authorization for a predeclared, label-reading effective-size and target-power pass on those cleaned
bytes; and accumulate enough materially outcome-unseen sessions for the signed economic effect. No
fit, new label/P&L/forward-path statistic, reserved session, purchase, download, or broker contact
occurred. No alpha attempt was charged and no protected ledger was modified. Nothing in this finding
is adopted without the owner's ruling.

## Part 1 — the audit

The structural wrappers opened no outcome-bearing Parquet column and computed no new outcome
statistic. They did parse the already-published corpus-build receipt, whose session records include
published outcome summaries; rereading those existing values is free under the audit's alpha law,
and none was used to discover or score the liveness defect.

### (a) The standing one-population ruling does not survive

The evidence does **not** identify two causal populations. It also no longer supports pooling them
for target or dollar-economic inference. The correct finding is **source effect versus market regime
is not identified**, so the two eras must be treated operationally as separate strata.

The 2026-08-18 pre-fit review made its one-population ruling conditional on repairing the defective
dispersion field. A subsequent directed-change re-probe discharged the field defect, but did not
settle era separability. The current SPX receipt's direction-specific 11-field minute-state AUC is
0.4322; reversing score orientation gives 0.5678, slightly above the memo's approximate 0.55 concern
bar. Dispersion-only AUC is 0.4548 (0.5452 reversed). The holdout contained only 45 sessions (11
owned and 34 backfill), while AUC was scored on 16,648 correlated minute rows and no
session-clustered interval was reported. The result is unstable and inconclusive, not evidence of
either full-input era blindness or conditional-target exchangeability. The 11-field probe also omits
the policy's contract-level ask, spread, self-IV, theta, smile residual, depth imbalance, moneyness,
and side inputs, so it is not a probe of every actual policy input.

The corrected marginal label rates remain useful but do not settle that conditional question:
owned is 31.80% across 243 sessions (cross-session SD 6.12 points) and backfill is 30.59% across 769
label-bearing sessions (SD 6.66 points), a published +1.21-point gap with p=0.0088. The 2022-to-2023
gap inside backfill is larger at +1.47 points, and the era boundary is the quietest adjacent label
seam at +0.63 points versus changes as large as 1.80 points inside backfill. Those facts argue
against attributing the marginal gap uniquely to source or calling the boundary an exceptional
discontinuity; the published p-value is descriptive, not a causal seam test or a serially adjusted
calendar-family result. It does not establish marginal equality, and none of this certifies pooled
dollar economics.

The structural scan found both commonality and economically relevant **differences between
time/source-confounded era-date buckets**:

- Every current member ladder uses essentially 20 contracts and a 45-point strike span, and every
  bid/ask is on the five-cent grid. The receipt's “quote density” is member-ladder occupancy—ladder
  rows divided by minutes times the union of session contracts—not vendor CBBO completeness. Its
  mean is 0.458 in 2025 backfill (session bootstrap 95% interval 0.435–0.479) and 0.468 owned
  (0.454–0.482), so that occupancy measure alone does not split the eras.
- The mean session median spread is $20.68 in 2025 backfill (95% interval $19.57–$21.93) and $16.75
  owned ($16.13–$17.33). Median spread as a fraction of premium overlaps: 1.871% (1.796–1.946%) versus
  1.824% (1.770–1.877%). Dollar spreads therefore differ even though relative-spread intervals
  overlap; this audit cannot attribute that difference to source.
- A SHA-256-selected sample of 32 normalized full-chain sessions per year/source bucket shows wider
  books on the owned-era dates. Mean median live contracts per minute are 280.1 in 2025 backfill
  (95% interval 258.4–305.1) versus 359.2 owned (324.4–399.3); strike span is 5,164.8 points
  (4,786.7–5,496.9) versus 6,587.5 (6,368.8–6,850.0). The fixed member ladder hides this upstream
  composition difference; the sample does not identify its cause.
- Settlement source is exactly era-coded: all 771 backfill sessions use `parity_close`; all 243 owned
  sessions use `official_1600`. All final tapes are stamped `spx_parity_spot`, but the ingest paths
  did not originally construct underlying price the same way: backfill normalization computes
  call-put parity, while owned ingest supplied aligned SPX context. A common stamp therefore hides a
  policy-visible provenance difference: `underlying_price` feeds moneyness, recomputed self-Greeks,
  and the SPX tape. It can be harmonized by rebuilding under one law, but it was not a same-date
  source twin.
- Vendor Greek columns are present in the owned normalized quote schema and absent from the repaired
  backfill normalized schema, but the owned coverage receipt says IV/delta/gamma/theta/vega have zero
  usable coverage. This is a normalization/schema distinction, not evidence that Databento supplied
  valued Greeks. The member recomputes its own Greeks. Volume and open-interest coverage still splits the
  corpus/adapter-visible bytes: about 98.9%/100% owned and 0% backfill. Those fields appear in the
  tensorizer ladder contract but the declared policy architecture does not index them; they prove the
  normalized products are structurally distinguishable, not that the fitted member used them.

The trigger for this audit was the already-published fixed model's dollar ordering sequence, which
declines across its three backfill blocks and then jumps at the source seam:

```text
+$7.25  +$5.88  +$5.16  |  +$11.88  +$14.81
```

The recent-minus-early contrast is +$7.25 per selected entry, but it has no declared inferential law
or uncertainty interval and is exploratory only. It does not statistically reject population
equality. Its role is epistemic: it exposed that the prior probe never tested conditional target
invariance. Current outcomes could test an era/date difference under a properly declared law, but any
such difference remains unattributable to acquisition provenance versus market regime because no
paired source overlap exists. The structural differences keep source artifact live; none attributes
the jump to either data provenance or market regime.

Population scope matters: the current-corpus member-ladder bucket aggregates and all published
base-rate/ordering outcomes include the three newly failed backfill sessions. Their post-exclusion
values are not receipted, and outcome changes remain unknown by design. The saved 150-session era
probe and the SHA-selected 2023 full-chain sample contain none of those dates, so the AUC and sampled
full-chain width/span figures above are unaffected.

### (b) No construction on the current dates nonparametrically breaks the confound

Backfill ends 2025-07-31, owned begins 2025-08-01, and there are exactly zero same-date dual-source
sessions. Source is therefore a deterministic function of date.

Calendar matching, boundary buffers, volatility/liquidity matching, ranks, ratios, dropping
era-predictive fields, fixed effects, within-minute controls, and binary labels can reduce nuisance
sensitivity. None creates the missing source counterfactual. A one-era subset removes source
variation; it does not estimate a source effect. Difference-in-differences has neither an unaffected
control nor a justified parallel trend. A regression-discontinuity or smooth-date-trend model can
produce a coefficient, because source is not algebraically collinear with every chosen trend basis;
but its interpretation requires untestable continuity/no-coincident-regime-shift assumptions at
2025-08-01. It does not identify source from these bytes alone.

Several processing differences could be harmonized only by rebuilding. Recomputing owned underlying
by parity can move moneyness, self-Greeks, and even band/action membership, so it is not a metadata
transform. The published zero-recovery twin found exactly $0 effect for the exact Member-P
60-minute bracket rule because 0 of 4,856 fired exits used settlement; it does not remove era-coded
settlement bytes or discharge Member Q/120-minute and other targets. No same-date pair spans the
backfill-versus-owned acquisition laws. Identification requires such a historical pair or a
prospective simultaneous pipeline; neither is present or authorized.

### (c) Honest usable sample and effective size

Economic and serial-policy inference clusters at the session—not the quote, candidate, minute, or
trade. `measure_effective_sample_size.py` is a different, model-fitting capacity diagnostic: it
collapses the ladder to session-minute states and discounts their within-session dependence, so its
reported ESS can exceed the session count. It is not an economic trade count. Exact structural
accounting is:

| Population | Current serial sessions | After conservative liveness exclusion | Label-bearing | ESS-tool eligible |
|---|---:|---:|---:|---:|
| Combined | 1,014 | 1,011 | 1,009 | 1,008 |
| Backfill | 771 | **768** | 766 | 765 |
| Owned | 243 | 243 | 243 | 243 |

The two zero-candidate sessions are 2025-04-09 and 2025-04-10. They remain valid WAIT-only serial
sessions in the 768/1,011 account chronology; only label-bearing and ESS-tool counts exclude them. A
third backfill session, 2025-04-07, has only 11 distinct decision minutes and falls below the
effective-size tool's 30-state cutoff. The three liveness failures are separate and all otherwise
pass that state-count threshold.
Thus **768 backfill sessions** are the largest defensible single-source candidate after structural
repair. They are not a clean corpus yet; exclusion must be enforced in bytes or by a pinned reader,
not left as prose.

The required effective size for that 768-session stratum is **UNKNOWN**. The published combined
receipt measured 1,011 eligible sessions and 327,557 session-minute states, with autocorrelation ESS
12,687–33,790 and design-effect ESS 2,028–3,727 across four old binary labels. It includes all three
new liveness failures and excludes the sparse/zero sessions, so it is not a measurement of the
cleaned set. Effective size is nonlinear and cannot be obtained by subtraction. The older owned-era
receipt is tied to a different historical candidate input. An outcome-blind geometry receipt
measures 698,231 candidate rows and 77,254 unique session-minutes there versus 773,105 rows and
79,109 unique session-minutes in the current owned slice, despite both containing 243 sessions. Its
published ACF ESS 2,879–7,557 and design-effect ESS 590–1,016 are therefore only historical anchors,
not measurements of the current owned geometry.

Running `measure_effective_sample_size.py` on a cleaned era would necessarily open outcome labels and
write new base rates. The goal's alpha law requires the owner to authorize that exposure. Part 1 no
longer needs it to reach a decision, so this audit did not spend alpha merely to turn an unknown into
a number. The clean combined, backfill, owned, clock-target, and actual 40%-prefix effective sizes
remain unmeasured.

### (d) The known repairs mostly held; the defect class did not

The new structural receipts verify these exact properties on the current bytes:

- all five corpus tables contain the same 1,014 sessions and no `.partial` files;
- every candle table has 390 bars, 09:30–15:59, knowable 09:31–16:00;
- none of the 12 sessions rejected by the existing clock/liveness receipts leaked into the corpus;
- the vendor-padded 2022-11-25 early close is absent;
- all 214 carried-close sessions carry exactly one minute, not a longer fabricated tail;
- tape and settlement provenance stamps agree with the build receipt.

The first gate returns `FAIL_STRUCTURAL`; the corrected exhaustive raw-book gate returns
`FAIL_INTERIOR_FULL_BOOK_FREEZES_PRESENT`. It scanned all 1,014 included sessions, 167,097,488
in-clock raw rows, and 394,446 adjacent full-book comparisons—an exact census with no sampling—and
found three included backfill sessions whose entire quote book repeats exactly and then resumes:

| Session | Contracts per snapshot | Exact repeated interval(s) | Duplicate minute transitions |
|---|---:|---|---:|
| 2023-06-26 | 340 | 10:29–10:30 | 1 |
| 2023-10-19 | 308 | 12:16–12:18 | 2 |
| 2023-10-25 | 322 | 10:17–10:20 and 10:22–10:39 | 20 |

These are four interior runs and 23 duplicate transitions. The longest is 17 transitions. The
supplemental scan reads only raw quote identity, bid/ask, and sizes; it confirms the same runs in the
acquired raw snapshots and their downstream representations. That makes a builder-only accident
unlikely. It does **not** prove whether the upstream cause was vendor padding, a publication outage,
or a genuinely unchanged market, so mechanism remains unknown. Fail-closed exclusion is still the
correct disposition because the whole book freezes and later restarts.

The miss is explained. `verify_backfill_clock.py` checks a frozen *tail*, so an interior freeze can
pass. The pre-fit stale signature included recomputed option-volatility summaries, whose deterministic
time-decay drift can make a frozen quote book look live. The new detector fingerprints quote identity,
price, and size directly and inspects every adjacent minute. The build receipt also has no per-output
file hash manifest; its self-hash protects the report, not every corpus byte. A clean rebuild needs
both the corrected detector and content-addressed output evidence.

### (e) What effect this corpus can certify

It cannot certify the signed economic effect. The effect **could not be detected on the available
independent sample**; it was not **measured absent**.

| Published planning law | Required independent sessions | Available upper bound |
|---|---:|---:|
| Signed Tier A exit effect: +$25/trade and +5 capture points | about 2,560 outer sessions | at most 607 post-prefix sessions even if the barred pooling assumption were made |
| Refused Tier B: +$50/trade and +10 capture points | about 640 outer sessions | at most 607 post-prefix sessions even if the barred pooling assumption were made |
| $22.44 clock-target-delta proxy | about 1,132 at 80% / 1,568 at 90% | barred pooled ceiling 1,011; honest largest stratum 768 before its 307/461 role split |

The provisional post-exclusion combined chronology would allocate 404 sessions to the fixed 40%
prefix and leave at most 607 for every later role **if the eras were wrongly pooled**; backfill alone
would allocate 307 and leave 461. The signed design requires separate entry validation, exit training,
and outer evaluation, so 607 is deliberately a counterfactual impossible-best-case outer bound.
Effective size can only reduce it. Tier B was refused, and weakening signed Tier A now requires a
fresh owner signature that explicitly accepts testing only a larger effect.

The existing exit evidence reinforces the scale: 644 trajectories occupy 323 session clusters, with
within-session effective size near 385, but evaluation `n=0` because all 644 trained the exit head.
Treating all 323 as evaluation gives non-authoritative, **separate** scale checks of about $42/trade
and 14 capture points under the published approximation; it is not a joint certification statement.
The 50,318 uncapped trajectories cannot inflate `n`; the signed two-ticket law and session clustering
still bind.

The $22.44 calculation is a planning proxy, not target-specific power, and always-hold itself loses
$1.74/trade versus a +$254.64 oracle. No profitable replacement is inferred. What is measured here is
the inability of the present bytes and chronology to certify a worthwhile effect under the signed
law.

## Part 2 — replacement hypothesis: not reached

The Part 2 guard stops here because Part 1 failed. No target, gate, feature contract, model stack, or
falsification experiment is proposed or adopted. In particular, this audit does not promote the
route census's paper idea of clock-hold-relative contract ranking with a separate gate, does not size
a module against an inapplicable effective-size receipt, and does not create a fit declaration.

Member Q remains **PRESERVED, NOT RUN**. Its rare-positive session power is unpublished and its
120-minute zero-recovery twin remains undischarged. `DO_NOT_RETEST.md` row 187 closes the exact
Member-P bracket-dollar/$0-WAIT configuration and post-hoc seed, fold, threshold, architecture, and
gate bolt-ons; only a genuinely independent target/gate frozen before outcomes and tested on
materially unseen sessions could reopen that concept. Row 188 closes any new head, horizon, or
post-hoc statistic on the same 644 trajectories. No independent historical holdout exists, so
neither exception authorizes Part 2. Rejecting this audit would still not authorize Member Q, another
Member-P bolt-on, or an outcome exposure.

The alpha ledger remains at six attempts. Its next directional bar is 0.66059463 against 0.5799
break-even, but that directional threshold is inapplicable to this dollar-economic verdict and can
never substitute for signed Tier-A economics.

## What could not be determined

- **Source effect versus market regime:** not identified because source and date have zero overlap.
- **Exact cause of the four interior freeze runs:** quote identity is measured; vendor padding versus
  upstream outage versus genuine no-change is not.
- **Outcome impact of excluding the three sessions:** deliberately not read; it would be a new label,
  P&L, or path statistic and is unnecessary to this verdict.
- **Clean per-era and target-specific effective size:** no applicable receipt exists; the required
  tool opens outcomes and needs owner authorization after the clean population is pinned.
- **Minimum effect a cleaned replacement-target study could certify:** target-specific variance,
  fit/validation/outer role allocation, and known-answer recovery have not been defined or measured;
  the $42/14-point figures above are non-authoritative scale checks, not a threshold.
- **Current owned-only effective size:** the historical owned receipt is not byte-equivalent to the
  current job-46 candidate geometry.
- **Any economic edge for a replacement target or stack:** no fit or new outcome statistic was run.
- **Same-date source-twin variance, availability, and cost:** no paired sample exists, and vendor
  contact or download was prohibited.
- **Future confirmation power:** sessions from 2026-08-06 onward remain reserved and were not read.
- **Member Q power and settlement sensitivity:** unpublished and deliberately preserved.

## Documentation and runtime conflicts left visible

- `PREFIT_CORPUS_REVIEW_2026_08_18.md` and the current job-46 status narrative say one population;
  the later fixed-model seam, zero-overlap arithmetic, and this audit restrict that statement to a
  common structural pipeline. The owner has not adopted this replacement ruling.
- The pre-fit receipt names the removed ES-tape corpus. Carry-forward to the current SPX tape has
  exact base-rate reproduction and sampled structural checks, not a full byte-identity proof.
- “Owned has vendor Greeks” is schema-only; the coverage receipt reports zero valued coverage.
- All final tapes say parity spot even though the underlying ingest constructions differ by era.
- Coverage receipts report no clock/liveness exclusions while separate clock receipts do. Final
  membership is correct for the previously known exclusions, but the reporting layers disagree.
- The prior stale-book finding reported a maximum one-minute run because its signature drifted even
  when quote bytes did not. Runtime quote fingerprints supersede that prose.

## Evidence and immutable receipts

Every new producer refuses to overwrite an existing receipt. The corpus and owned-geometry wrappers
write a self-hashed `FAIL_WRAPPER_OR_STRUCTURAL_CHECK` receipt and exit 5 on caught exceptions; the
raw-book wrapper writes `FAIL_WRAPPER_OR_STRUCTURAL_INVARIANT` and exits 5. Structural defect gates
exit 4 rather than masquerading as execution success. The owned-geometry difference is the intended
structural result and exits 0.

- Outcome-blind corpus wrapper:
  [`two_era_structural_audit_2026_08_22.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/two_era_structural_audit_2026_08_22.py),
  SHA-256 `1ef2478136a8679db808cbecf4ca13df6f034d6f32f3a82ddaf22dabe910191b`.
- Its attempt-001 receipt:
  [`two_era_structural_audit_2026_08_22_attempt001.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/two_era_structural_audit_2026_08_22_attempt001.json),
  gate `FAIL_STRUCTURAL`, self-hash
  `d7468cba86ddcad39af507ca830f9b5166335c36937e12c6780215be9d4e4423`.
- Archived raw-book attempt-001 producer:
  [`interior_full_book_freeze_audit_2026_08_22_attempt001.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/interior_full_book_freeze_audit_2026_08_22_attempt001.py),
  SHA-256 `e94350ac3b7997bcd0809014f6121cb86d4010227405b390bdb12a5c0481de13`.
- Its preserved failure receipt:
  [`interior_full_book_freeze_audit_2026_08_22_attempt001.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/interior_full_book_freeze_audit_2026_08_22_attempt001.json),
  gate `FAIL_WRAPPER_OR_STRUCTURAL_INVARIANT`, self-hash
  `4b6f054a61ecd8ebd03e43f4cc128406e6760662edf83de13edc8118310c243e`.
  Pandas restored owned `ts_recv` as an index, so this attempt stopped with `KeyError` and supports no
  liveness claim.
- Corrected exhaustive raw-book producer:
  [`interior_full_book_freeze_audit_2026_08_22_attempt002.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/interior_full_book_freeze_audit_2026_08_22_attempt002.py),
  SHA-256 `d2d9cd111079fea320df70530aed04a7779b7fbd6c5c5d5a5c6b44ab41a93a23`.
- Its attempt-002 structural-failure receipt:
  [`interior_full_book_freeze_audit_2026_08_22_attempt002.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/interior_full_book_freeze_audit_2026_08_22_attempt002.json),
  gate `FAIL_INTERIOR_FULL_BOOK_FREEZES_PRESENT`, self-hash
  `a9d3acd5dd53570b3bd40a3129a77a19c5ad514e8a69ffa2b310df13b74b9211`.
- Current build receipt:
  [`corpus_build_spx_tape_receipt.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/corpus_build_spx_tape_receipt.json),
  self-hash `50b2ebae4f68ae07f856378ec6c7eb2f29281798156febbaa7f49a5eda5f6051`.
- Published effective-size receipt:
  [`effective_sample_size_2026_08_18.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/effective_sample_size_2026_08_18.json).
- Outcome-blind owned-geometry wrapper:
  [`owned_ess_geometry_audit_2026_08_22.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/owned_ess_geometry_audit_2026_08_22.py),
  SHA-256 `3c288b4019f55bc16a54e20aa8645a594880bd0032d30c2a01bdf4de91b107a9`.
- Its attempt-001 receipt:
  [`owned_ess_geometry_audit_2026_08_22_attempt001.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/owned_ess_geometry_audit_2026_08_22_attempt001.json),
  gate `GEOMETRY_DIFF_CURRENT_OWNED_ESS_UNKNOWN`, self-hash
  `25b350dcde3e2c8f727888ea6800cf4d9a328ff6705301dd7c64188b3592a9c3`.
- Corrected current-corpus era probe:
  [`era_probe_reprobe_spx_tape_2022-06-01_2026-07-31.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/era_probe_reprobe_spx_tape_2022-06-01_2026-07-31.json).
- Published fixed-model seam synthesis:
  [`regime_fold_receipt_synthesis_2026_08_22_attempt002.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/regime_fold_receipt_synthesis_2026_08_22_attempt002.json).
- Exact Member-P settlement twin:
  [`zero_recovery_twin_2026_08_22.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/zero_recovery_twin_2026_08_22.json).
- Standing pre-fit ruling and later regime synthesis:
  [`PREFIT_CORPUS_REVIEW_2026_08_18.md`](PREFIT_CORPUS_REVIEW_2026_08_18.md) and
  [`REGIME_VS_FOLD_SIZE_AND_ROUTE_CENSUS_2026_08_22.md`](REGIME_VS_FOLD_SIZE_AND_ROUTE_CENSUS_2026_08_22.md).
- Published economic-power review and signed law:
  [`EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md`](EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md) and
  [`EXIT_EVALUATION_LAW_AMENDMENT_2026_08_22.md`](../../governance/EXIT_EVALUATION_LAW_AMENDMENT_2026_08_22.md).
- Closed configurations and preserved Member Q:
  [`DO_NOT_RETEST.md`](../history/DO_NOT_RETEST.md).

## Verification

- `./.venv/bin/python v5/ops/check_project.py`: **PASS** — project and repository OK.
- `./.venv/bin/python -m pytest v5/tests -q`: **1,123 passed**, 105 existing numerical warnings,
  process exit 0 in 64.80 seconds. Unpiped output is retained at
  `/tmp/v5-pytest-two-era-audit-20260822.out`.
