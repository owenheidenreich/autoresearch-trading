# Pre-fit adversarial review of the two-era corpus and feature set

**Job 46, Boundary 1. Written 2026-08-18** under §3 item 1 of the signed
[work-allocation memo](../../governance/WORK_ALLOCATION_MEMO_2026_08_18.md). Controlling state is
[`STATUS.md`](../../STATUS.md); the design under review is
[`LEARNING_CONTENT_DESIGN_2026_08_16.md`](../../work/lifecycle-training/LEARNING_CONTENT_DESIGN_2026_08_16.md).

**The one-sentence answer: the two eras are one population and a single model should train across both
— but only after one feature is removed, because that feature, not the market, is what currently makes
the eras look different.**

This review changes what gets built in three concrete ways, listed in §6. Nothing else in the design
needs to change, and no signed law is reopened.

---

## 1. What was done, and what was deliberately not done

Every number below was re-derived from the corpus itself by
[`prefit_review_pass.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/prefit_review_pass.py)
and [`prefit_review_followups.py`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/prefit_review_followups.py),
receipted as `prefit_review_receipt.json` (`287e59bd…`), `prefit_review_followups.json`, and a
per-session metrics table `prefit_review_session_metrics.parquet`. Nothing was transcribed from an
earlier receipt; where this review agrees with the build receipt, it agrees by independent
measurement.

**Two things were deliberately not done.** No trade economics were read — no `net_*` column was ever
loaded — and **no statistic joining a feature to the label was computed**. Measuring whether the
features predict the label is Phase 4a's declared, alpha-ledger-charged job; doing it informally here
would be an uncharged experiment and would let the design's shrink ladder be tuned by peeking. Label
statistics and feature statistics were computed separately and never joined.

The one model fitted was the throwaway era probe the owner authorized in conversation on 2026-08-18:
state fields in, era out, no outcome or label data, coefficients receipted and kept nowhere else. It
exists to answer a question single-field statistics structurally cannot (§4).

---

## 2. The corpus is sound. Here is the evidence, including the checks that could have failed

Six independent checks, four of which no existing gate performs:

| Check | Result |
|---|---|
| Headline counts | 1,014 sessions, **3,226,673 candidates**, settlement **771 parity / 243 official** — reproduces the build receipt exactly |
| Per-era base rate | owned **31.80%** (n=243, sd 6.12), backfill **30.59%** (n=769, sd 6.66) — reproduces the handoff's corrected figures exactly |
| Carried close | **214 sessions, every one carried exactly 1 minute**; candidates that could depend on it: **0.709% at 60m**, 4.079% at 120m — reproduces the corrected footnote exactly |
| **Label law, re-implemented independently** | **0 mismatches in 32,377 candidates** over 10 sessions across both eras, comparing label, gain minute and loss minute |
| **Frozen-book scan on the built corpus** | Longest run of identical consecutive ladder summaries is **1 minute** in both eras. No stale session reached the corpus; the liveness gate held |
| **Exit resolution, per era** (the design's §4.3 pre-fit QC gate, never reported until now) | 60m: **97.14%** executable (backfill) vs **97.69%** (owned); cash-settled 2.64% vs 2.14%; **0.00% blocked in both eras at every horizon** |

Two further checks came back clean: **zero same-minute double-touch NaNs** in either era (the label's
unresolvable case never occurs at minute cadence), and **no infinite values** in any chain-internal
field.

The independent label recompute is the one worth dwelling on. It re-walks the +50%-before-−30%
first-touch rule from raw quote files under a separately written implementation of the same
executable-mid law. Agreement on 32,377 candidates in both eras means the corpus's central claim —
its labels — is not resting on a single implementation.

---

## 3. Two defects, both fixable before the fit, and one is blocking

### Defect 1 — two sessions are structurally untradeable and their files will crash the adapter

**2025-04-09 and 2025-04-10 contain zero candidates.** They are written as degenerate 40-column
parquet files with no label columns at all, while the other 1,012 sessions carry 139 columns.

The cause is not a bug, and it is worth stating plainly because it is also a genuine economic fact:
on those two April 2025 tariff-spike days, **the cheapest out-of-the-money contract within 25 points
of the money cost $2,000–$2,050**, so the signed $2,000 ticket cap admitted nothing. The risk law
worked exactly as designed and stood the strategy down on the two most violent days in the corpus.
(2025-04-08, the day before, had 206 affordable contract-minutes with a cheapest ask of $1,310.)

Three consequences, in ascending order of severity:

1. This is the **n=769 versus 771** discrepancy the handoff flagged. Two built sessions have no base
   rate because they have no candidates.
2. The build receipt records both as `BUILT` with `label_base_rate: null`. A session contributing
   nothing is indistinguishable from a session contributing normally without opening the file.
3. **A per-session read of the label column raises `ArrowInvalid` on these two files.** The
   `SessionEpisode` adapter is specified to read one session's parquet at a time, so it will crash on
   them. Directory-level reads unify the schema and fill nulls, which is exactly why nothing has
   noticed: every check run so far reads the directory.

This is the same pattern §7 of the memo records for the fourth and fifth time — the tool reports clean
because the shape it reads is not the shape the next tool will read.

### Defect 2 — one feature is a calendar detector, and it is the only thing making the eras separable

The state vector can identify a session's era. Measured on held-out sessions:

| Probe | Held-out AUC |
|---|---|
| All 11 state fields | **0.760** |
| **Without `implied_spot_dispersion_bps`** | **0.430** |
| **`implied_spot_dispersion_bps` alone** | **0.837** |
| The other six chain-internal fields together | 0.481 |
| The four tape fields together | 0.413 |

(An earlier, larger sample put the full-vector figure at 0.872; the ablation reproduces on both.)

**Every bit of era-detectability is one field.** Remove it and the state vector cannot tell 2022 from
2026 — 0.430 is no signal at all. The other six chain-internal fields, which are the design's actual
hypothesis, are at chance.

**The mechanism was measured, not guessed.** The field is dollars of cross-strike parity dispersion
divided by the index level, expressed in basis points. SPX rose from **3,960 to 6,845** across the
corpus — a factor of 1.73 — so the same dollar disagreement reads as steadily fewer basis points each
year. Predicted from the level effect alone, the 2022-to-owned ratio is **0.392**; measured, it is
**0.396**. The field is a proxy for the calendar wearing microstructure's clothes.

| Bucket | median SPX | dispersion (bps) | dispersion ($) | **dispersion ÷ spread** |
|---|---:|---:|---:|---:|
| backfill 2022 | 3,960 | 0.106 | 0.042 | **0.177** |
| backfill 2023 | 4,207 | 0.077 | 0.033 | **0.187** |
| backfill 2024 | 5,470 | 0.058 | 0.032 | **0.187** |
| backfill 2025 | 5,907 | 0.052 | 0.030 | **0.159** |
| owned | 6,845 | 0.042 | 0.028 | **0.177** |

The right-hand column is the fix, and it is already in the data: **dispersion as a fraction of the
quoted spread is flat across all five buckets.** That form keeps the microstructure-stress meaning the
design wanted and discards the index level.

This matters beyond one field. The design's §3.1 bar reads *"no field whose value is a monotone
function of ladder size may enter the state."* That bar is aimed at a hazard which, as §5 shows, does
not exist in the built corpus — while the hazard that does exist, a field monotone in the **index
level**, walks straight past it. **The bar should be generalised**: no field whose value is a monotone
function of any slowly-varying calendar quantity — ladder size, index level, tick regime — may enter
the state, and compliance is verified by measurement rather than asserted.

---

## 4. Why a joint probe was necessary, and what it would have missed

Judged one field at a time, `implied_spot_dispersion_bps` is conspicuous (single-field AUC 0.154, i.e.
0.846 the other way). But the general case is not so obliging: a set of fields each individually near
chance can be jointly separable, and no amount of per-field screening detects it. That is precisely
what the design's diagnostic D7 is for, and D7 runs *after* the fit.

Running it before costs one throwaway logistic regression and moves the finding earlier by the entire
length of a fit. The result also converts D7's pass bar from a guess into a measurement: **the clean
state vector's era-separability is 0.430–0.481, so D7's bar can be set at roughly 0.55** and a
violation will mean something specific.

The probe's honest limitation: it is linear, so it bounds era-detectability from below. A tree could
find more. It is enough to settle the question at hand because it found a large effect, not because
it proves the absence of a small one.

---

## 5. Three facts about the corpus that the design's own text gets wrong

None of these is a defect in the build. All three are places where a reader of the design would form a
false picture of what the model will see.

**5.1 The corpus candles are ES, not SPX.** The design (§7.2) specifies the SPX parity-spot series for
the four tape channels "in both eras, for era symmetry." The built candle table comes from the ES root
via the pinned builder's `prepare_es`. Measured signed gap between candle close and option-implied
spot: **+20.3 points (backfill), +29.8 points (owned)**, with a within-session standard deviation of
only **0.65–1.55 points** — the shape of a futures basis, not noise.

Numerically the harm is small: all four declared tape channels are differences or ratios, so a
near-constant basis largely cancels, and the tape-only era probe sits at chance (0.413). But the
provenance question is real, and it is **not an agent's to settle** — the owner's 2026-08-14 ruling is
SPXW/SPX-only for active policy inputs, and these are active policy inputs. The design already names
this as a one-line owner boundary ruling (§7.2). It is now a measured one.

**5.2 The ladder table is the ±25-point band, not the whole chain.** The builder writes
`ladder_state(whole_live_chain=False)`. Measured:

| | corpus ladder table | full live chain |
|---|---:|---:|
| contracts/minute (backfill) | **20.0** | 252.5 |
| contracts/minute (owned) | **20.0** | 280.0 |
| strike span (backfill) | **45 pts** | 4,750 pts |
| strike span (owned) | **45 pts** | 6,200 pts |

Two consequences. First, **the era-detector-by-ladder-width hazard does not exist in the built
corpus** — the band is 20 contracts and 45 points wide in 2022 and in 2026 alike. The design spends a
paragraph and a diagnostic on it; keeping the bar costs nothing, but it was aimed at the wrong target
while the real detector sat in a different field.

Second, and more consequential for interpreting the result: **the seven chain-internal features are
computed over the ±25-point band.** What the design calls "chain-wide skew" and treats as row 338's
named unexplored source is, in the built corpus, *band-local* skew; `chain_depth_imbalance` is band
depth, not chain depth. This is defensible — the band is where the tradeable action lives — but it
must be stated now, because otherwise the attribution pass will credit or blame chain-wide
information the model was never shown.

**5.3 `smile_curvature` pools the two sides; the design says per-side.** I expected this to matter —
a call and a put at the same signed moneyness sit at different strikes, so pooling could turn a
curvature measurement into a skew measurement. **Measured, it does not.** Pooled curvature tracks the
per-side average at **r = 0.80** and correlates with put-minus-call IV at **−0.03**. Reported as a
discrepancy to reconcile in wording, not as a directed change.

One related note: `smile_curvature` has a standard deviation of **4.5e-5**, five orders of magnitude
below `put_call_depth_ratio`. It carries genuine variation (15,400 distinct values in 15,600 minutes)
and the per-fold scaler the design already requires will handle it — but it is entirely dependent on
that scaler working.

---

## 6. The ruling: one population, single model, era not an input

The design brief left this open. It is now decided on measurement, and the answer is the simplest of
the four candidates.

### 6.1 The evidence

**The era gap is smaller than an ordinary year-to-year gap inside a single era.**

| Comparison | Difference | Welch p |
|---|---:|---:|
| owned vs backfill (the "era gap") | **+1.21pp** | 0.0088 |
| **2022 vs 2023 — both inside the backfill era** | **+1.47pp** | 0.0365 |

Yearly base rates run **31.71 / 30.24 / 30.16 / 30.99 / 32.18** (2022→2026). There is no trend and no
step: a shallow U whose two highest values are the two chronological *extremes* — the earliest
backfill year and the latest owned year. Whatever moves the base rate, it is not the era.

**Under the frozen chronology the fit will actually use, the era boundary is the quietest seam in the
data:**

| Block | Sessions | Span | Era | Base rate |
|---|---:|---|---|---:|
| training prefix | 405 | 2022-06-01 → 2024-01-29 | backfill | 30.87% |
| score block 1 | 122 | 2024-01-30 → 2024-07-29 | backfill | 29.07% |
| score block 2 | 122 | 2024-07-30 → 2025-01-29 | backfill | 30.87% |
| score block 3 | 122 | 2025-01-30 → 2025-07-31 | backfill | 30.89% |
| score block 4 | 122 | 2025-08-01 → 2026-02-02 | **owned** | 31.52% |
| score block 5 | 121 | 2026-02-03 → 2026-07-30 | **owned** | 32.08% |

The era boundary is block 3 → block 4: **+0.63pp, the smallest transition in the sequence.** The two
largest (−1.80pp and +1.80pp) are both interior to the backfill era.

**The remaining axes agree:**

- **Driver.** Session base rate correlates with realised volatility at **ρ = +0.355** — a market-regime
  variable that varies as much within an era as across the boundary.
- **Label mechanics.** Exit resolution differs by at most 0.55pp between eras at any horizon; zero
  blocked exits in either; zero double-touch NaNs in either; the label law reproduces bit-for-bit
  under an independent implementation in both.
- **Action space.** Identical: 20 ladder contracts, 45-point span, median eligible moneyness −12.5
  points, call share 0.500, ~3,180 candidates per session, in both eras.
- **Feature side.** With `implied_spot_dispersion_bps` removed, the state vector cannot separate the
  eras (AUC 0.430).

### 6.2 The ruling

**Pool the corpus. One model across both eras. Era is not an input, and the backfill is not
down-weighted.** Weighting down 76% of the corpus to correct a 1.21-point difference that is smaller
than the noise between two adjacent years would spend most of the data this job bought in order to
correct an effect the data says is not there.

**This ruling is conditional on Defect 2 being fixed.** As the corpus stands, "one population" is
false — but false because of a normalisation we introduced, not because of the market. Remove or
renormalise that one field and it becomes true on every axis measured.

### 6.3 What must change before the fit

1. **Renormalise `implied_spot_dispersion_bps` to dispersion ÷ quoted spread, or drop it.** Renormalising
   is preferred: it preserves the microstructure-stress channel the design wanted and the era-free form
   is already measured flat. Whichever is chosen, re-run the era probe afterwards and receipt the
   result. Field count is unchanged if renormalised, so the 118-parameter contract is unaffected.
2. **Generalise the §3.1 bar** from "monotone in ladder size" to "monotone in any slowly-varying
   calendar quantity," verified by measurement rather than asserted.
3. **Set D7's pass bar at ≈0.55 era-probe AUC**, from the measured clean-vector baseline of 0.430–0.481,
   and have D7 report the per-field ablation rather than a single number — a joint detector built from
   individually innocent fields is exactly what a single number hides.
4. **The adapter must handle zero-candidate sessions** (2025-04-09, 2025-04-10) rather than crash on
   them, and must not silently drop them either: they are legitimate no-trade days under the signed
   risk law, and a serial simulator needs to know the account sat flat. **Excluding them is not
   ledger-row-331 outcome-dropping** — the exclusion criterion is an entry-time affordability fact,
   knowable before any outcome.
5. **Report executable economics per block, never pooled across the corpus.** Not a population split —
   a cost-regime fact. The spread runs **3.03 / 3.64 / 3.03 / 2.25 / 1.84 percent of premium** across
   2022→2026, so the model trains in a ~3% toll regime and is scored in a ~1.9% one. The label is
   mid-defined and unaffected; the dollar target and all Phase-5 economics are not.
6. **State in the fit declaration that the chain-internal features are band-local**, so the attribution
   pass does not credit chain-wide information the model never saw.

### 6.4 What this review refuses to decide

**The ES-versus-SPX tape source (§5.1).** It is a one-line owner boundary ruling by the design's own
construction, and it touches the 2026-08-14 SPXW/SPX-only instruction. The measurement is supplied so
the ruling can be made on evidence: the basis is +20 to +30 points, nearly constant within a session,
and the four tape channels cancel it to the point of statistical invisibility. Recommendation, offered
not taken: proceed with ES for development under the charter and record it as a named divergence for
Phase 6, because switching the source now changes four features whose era-neutrality has just been
measured. The owner may reasonably rule the other way.

---

## 7. Verdict on the feature contract

**No change beyond §6.** The seven chain-internal state fields and two per-contract fields are
computable at scale, era-symmetric once Defect 2 is fixed, and none of the coverage failures the design
worried about materialised:

| Field | Coverage, worst bucket |
|---|---:|
| `chain_depth_imbalance`, `put_call_depth_ratio` | **100.0%** |
| `implied_spot_dispersion_bps` | 99.2% |
| `risk_reversal`, `smile_curvature` | 98.7% |
| `atm_iv_change_5m` | 97.4% |
| `risk_reversal_change_15m` | 94.9% |
| `smile_residual` (per-contract) | 95.8% |
| `contract_depth_imbalance` (per-contract) | 100.0%, zero degenerate books |

Coverage is **identical to three significant figures across all five buckets** for six of the seven —
the 2022 narrow-ladder degradation the design anticipated does not occur, because the ±25-point band
is the same width in every year (§5.2). The training prefix is not a feature-starved era.

---

## 8. What this review does not claim

It does not claim the features carry information about the label. That is Phase 4a's question, it was
deliberately left untouched here, and the honest prior stated in the design remains: the bar is roughly
twice the largest entry effect this project has ever measured. A clean corpus and an era-neutral
feature set are necessary for the fit to mean anything. They are not evidence that it will succeed.
