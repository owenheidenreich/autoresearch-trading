# No signal exists here — the selector-quality requirement curve

**2026-08-23. [UNKNOWN] No signal exists in this study, and genuine outcome-blind out-of-fold
performance cannot be computed from its score law.** Every synthetic score below contains the
realised P&L of the same session it scores. The later partition is therefore a chronological
**outcome-conditioned oracle** check, not a prediction, not evidence that a signal is obtainable,
and not permission to fit or trade one. The owner-facing action is **ADOPT NOTHING**.

This finding closes the unreceipted assertion that correlation `0.05` turns the measured 09:35 ATM
mean from +$2.01/ticket into +$43.47. The short answer is: **[VERIFIED] +$43.47 can be constructed,
but [VERIFIED] it is neither a unique consequence of correlation 0.05 nor a robust result.** Under
the preregistered Gaussian-oracle ensemble, nominal rho 0.05 fails the drop-best session lower bound
at every declared selection rate. **[UNKNOWN] Whether any causal feature can achieve even the
measured requirement remains completely unanswered.**

---

**Verdict convention:** receipt-derived measurements and table cells are **[VERIFIED]** unless a
sentence explicitly marks them **[INFERRED]** or **[UNKNOWN]**.

## 1. Plain-English answer

- **[VERIFIED] The headline is reproducible, not discovered.** On all 1,011 sessions, the literal
  formula `0.05*z(P&L) + sqrt(1-0.05^2)*noise`, PCG64 seed 25,474, and the top 20% (202 sessions)
  returns **+$43.4689/ticket**. Achieved Pearson is **+0.05031** and Spearman **+0.04830**.
- **[VERIFIED] That seed was not cherry-picked, and the refutation does not rest on claiming it was.**
  See section 3.1: +$43.47 is approximately the *median* of the rho-0.05 ensemble in this exact
  configuration. The number is typical, not lucky. What fails is the session-level uncertainty around
  it, not its construction.
- **[VERIFIED] It does not establish weak-signal sufficiency.** The selected-session bootstrap is
  **[-$47.82, +$139.02]**. Removing the one best selected session leaves **+$29.03**, but its interval
  is **[-$54.61, +$113.23]**. Removing the top five turns the point estimate to **-$15.51**; those
  five wins total **$11,835**, more than the selected subset's entire **$8,781** net P&L.
- **[UNKNOWN] The original assertion's provenance is unrecoverable.** It did not state a selection
  rate, RNG/seed, partition law, or whether `0.05` meant a formula coefficient, Pearson, or
  Spearman. Correlation alone does not determine which tail observations are selected.
- **[VERIFIED] Nominal rho 0.05 fails the preregistered robustness requirement.** In the later
  oracle partition, every declared rate has a negative drop-best 95% lower endpoint. The positive
  raw point estimates are exactly the tail-selection hazard the study was designed to expose.
- **[INFERRED] Under this one Gaussian oracle law, the least demanding robust crossing is the 50%
  rate at nominal rho in `(0.100, 0.105]`.** At the crossing, the achieved median correlations are
  Pearson **+0.1385** and Spearman **+0.1288**. This is a conditional valuation benchmark, not a
  universal signal requirement and not evidence such a score can be made causally.

## 2. What was measured

The immutable input is the already-computed 10,078-row P&L table; it was not rebuilt. Population QC
binds all 12 cells, 1,011 unique sessions, the exact `session,start,offset,pnl,why` schema, the
2022-06-01 through 2026-07-30 range, and the upstream table hash
`30d628892be455e4bd31e4932a35244daf2bdee194851b2737eba42fbfd1d2d2`.

The curve itself is deliberately limited to the disputed **09:35 ATM** cell. Its full-population
mean is **+$2.0123**, median **-$61.5362**, and drop-best mean **-$0.9018**. The natural chronology is
the data-source seam:

| role | sessions | dates | unconditional mean | drop-best mean |
|---|---:|---|---:|---:|
| calibration / backfill | 768 | 2022-06-01–2025-07-31 | -$11.71 | -$15.57 |
| later / owned | 243 | 2025-08-01–2026-07-30 | +$45.38 | +$34.99 |

**[VERIFIED] The calibration mean/scale and every numeric score cutoff are frozen before the later
partition.** The later score uses that frozen calibration scale. **[UNKNOWN] This does not make the
score causal:** each later score still directly consumes its own later realised P&L. Date and quote
source change together at the split, so regime stability and source stability are not identified.

For each nominal rho in `{0, .02, .05, .10, .20, .40}`, 2,000 unshopped standard-normal selector
worlds share the same noise across rho values. Rates are `{5%, 10%, 20%, 50%, 100%}`. The calibration
top-k score fixes a numeric cutoff; the later block applies that cutoff without later reranking.
Every economic interval uses 5,000 whole-session resamples. Drop-best removes the largest selected
P&L separately in each selector world and never refills it. The inverse scans rho `0..0.50` by
`0.005` and uses a one-sided 95% studentized lower band simultaneous over all 505 rho×rate cells.

## 3. The rho 0.05 curve

These are the fixed 2,000-world **later oracle** estimands. “Raw” is before the required deletion;
“drop-best” is after removing each world's largest selected win.

| target rate | realised rate | tickets/year | raw mean (95% session CI) | drop-best mean (95% session CI) |
|---:|---:|---:|---:|---:|
| 5% | 4.97% | 12.5 | +$146.17 `[+$43.97, +$250.51]` | +$14.50 `[-$59.37, +$90.36]` |
| 10% | 10.11% | 25.5 | +$129.84 `[+$30.18, +$232.05]` | +$52.98 `[-$27.41, +$135.11]` |
| 20% | 20.20% | 50.9 | +$113.43 `[+$16.56, +$213.42]` | +$69.83 `[-$14.91, +$156.51]` |
| 50% | 50.08% | 126.2 | +$83.07 `[-$9.23, +$178.99]` | +$63.40 `[-$25.37, +$152.24]` |
| 100% | 100.00% | 252.0 | +$45.38 `[-$42.32, +$136.78]` | +$34.99 `[-$52.07, +$123.61]` |

**[VERIFIED] The raw tail-selected means look spectacular at the sparse rates. [VERIFIED] None
survives the required drop-best lower bound.** This is not a contradiction: a long option's mean is
carried by rare wins, and a score constructed from the outcome preferentially selects those wins.

Nominal rho is a formula coefficient, not a forced sample correlation. At nominal 0.05, achieved
median correlation is:

| partition | Pearson median (fixed-session world 95% range) | Spearman median (fixed-session world 95% range) |
|---|---:|---:|
| calibration | +0.0513 `[-0.0185, +0.1232]` | +0.0480 `[-0.0231, +0.1195]` |
| later | +0.0640 `[-0.0597, +0.1914]` | +0.0593 `[-0.0616, +0.1839]` |

Those ranges describe construction-to-construction variation with sessions fixed; they are not
confidence intervals. Their width is another reason a bare statement such as “rho = 0.05” does not
identify selected-tail value.

### 3.1 The seed was not shopped

**[VERIFIED] Added 2026-08-23 by independent reproduction (Claude), after this finding was committed.**
Every economic figure in section 1 and the claim audit was recomputed from the input CSV with
independent code and matched to the cent; the receipt self-hash, the P&L input hash, both archived
source hashes, and all 12 semantic-freeze sources were re-verified.

The receipt's `why` field states that the headline “can be manufactured under many rates and seeds.”
That is literally true, but a reader can take “manufactured” to mean the seed was searched until a
flattering number appeared. **It was not, and the record should not imply it.** Drawing 20,000
independent selector worlds (PCG64 seed 20,260,823) at nominal rho 0.05 and the headline's own top-20%
rate over all 1,011 sessions:

| nominal rho | median selected mean | 5th pct | 95th pct | P(>= +$43.47) | P(<= $0) |
|---:|---:|---:|---:|---:|---:|
| 0.05 | **+$45.01** | -$18.22 | +$108.26 | **51.6%** | 12.0% |
| 0.10 | +$89.12 | +$25.17 | +$153.11 | 88.1% | 1.1% |
| 0.1385 | +$124.13 | +$60.15 | +$188.18 | 98.1% | 0.1% |

**[VERIFIED] Seed 25,474 sits near the median of its own ensemble**, which returns at least the
headline in 51.6% of constructions. A rho-0.05 oracle selector really does return about +$45/ticket on
this table, about half the time.

**[VERIFIED] This strengthens the finding rather than weakening it, by relocating the failure.** Two
independent uncertainties stack, and only the second is decisive:

1. *Which sessions a rho-0.05 score happens to pick* — wide (5th–95th percentile spans about $126) but
   mostly positive.
2. *Whether these 1,011 sessions represent the future* — the session bootstrap **[-$47.82, +$139.02]**,
   and the drop-top-five collapse to **-$15.51**.

The correct one-sentence refutation is therefore **not** “that number came from a lucky seed.” It is
**“that number is real for this table, and this table's mean is carried by five sessions.”** The
rho-0.1385 row shows the same hazard is not escaped by a better selector: it raises the ensemble median
to +$124 without touching the session-level tail concentration that the drop-best rule exposes.

**[UNKNOWN] Nothing here makes the score causal.** Every world above still scores each session using
that session's own realised P&L. This subsection changes the interpretation of the reconstruction, not
its status: the adoption verdict remains **ADOPT NOTHING**.

## 4. The inverse requirement

Two thresholds are reported so selector-world averaging cannot hide the answer:

1. **Ensemble mean:** the drop-best simultaneous session LCB is positive and remains positive at all
   higher grid points.
2. **World robust:** the ensemble-mean rule passes **and** the fixed-session fifth percentile across
   2,000 selector worlds is positive at all higher grid points.

| target rate | ensemble nominal-rho bracket | world-robust nominal-rho bracket | achieved Pearson / Spearman at robust crossing | drop-best point | world 5th pct | simultaneous session LCB |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | `(0.080, 0.085]` | `(0.205, 0.210]` | +0.2766 / +0.2568 | +$405.10 | +$7.23 | +$277.12 |
| 10% | `(0.070, 0.075]` | `(0.150, 0.155]` | +0.2049 / +0.1903 | +$255.64 | +$2.06 | +$141.41 |
| 20% | `(0.070, 0.075]` | `(0.110, 0.115]` | +0.1520 / +0.1408 | +$164.78 | +$4.77 | +$57.50 |
| 50% | `(0.100, 0.105]` | `(0.100, 0.105]` | **+0.1385 / +0.1288** | +$106.12 | +$30.61 | +$1.80 |
| 100% | `>0.50 / UNKNOWN` | `>0.50 / UNKNOWN` | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

**[INFERRED] The decision-relevant conditional benchmark is therefore achieved Pearson about
+0.138 and Spearman about +0.129 at a 50% rate**, because that is the smallest world-robust crossing
inside the declared grid. The weaker ensemble-average crossing at 10–20% is nominal 0.075 (achieved
median Pearson about +0.098), but it does not protect a future reader from the fact that correlation
alone leaves large selector-world tail variation.

**[UNKNOWN] This is not the rho a future model “will need” in general.** It is the requirement under
one Gaussian noise law, one payoff table, one source-confounded later period, and a fixed threshold
ensemble. Different copulas, ranks, tail dependence, score calibration, or causal feature sets can
produce different selected-tail value at the same Pearson rho.

## 5. The requested in-sample versus later gap

At nominal rho 0.05, calibration-minus-later drop-best gaps range from **-$12.71** at 5% to
**-$59.58** at 50%; every 95% gap interval spans zero. At 20%, for example, calibration is +$13.26,
later is +$69.83, and the gap is **-$56.57 `[-$150.02, +$36.74]`**.

**[VERIFIED] These are chronological oracle-era gaps. [UNKNOWN] They are not genuine OOF
generalisation gaps and cannot measure tail-fitting.** Both sides use their own outcomes inside
their scores, and the split is perfectly confounded with quote source. A true OOF gap requires a
causally available feature-to-score mapping learned without the later outcome; no such mapping
exists in this study, and the no-fit rail bars inventing one here.

## 6. Executability, alpha, and boundaries

- **[VERIFIED] Count cap:** one 09:35 ATM opportunity exists per session, so every rate is individually
  within the signed two-tickets/day count cap. The 50% benchmark is roughly 127 tickets/year, not
  daily trading. Combining cells was not evaluated.
- **[UNKNOWN] Dollar ticket eligibility:** the P&L CSV has no entry ask column, so per-ticket
  affordability under the signed dollar cap cannot be certified from this artifact.
- **[VERIFIED] Sanity floor:** all **10/10** rho-zero partition×rate checks passed before any nonzero
  rho was evaluated. At 100%, rho zero reproduces the unconditional partition exactly.
- **[VERIFIED] Alpha position:** this is a requirements study on an already-open outcome table, not a
  real feature test. No model was fit, no new market hypothesis was tested, and no alpha was spent.
  The ledger remains **6 experiments** with next bar **0.66059463**.
- **[VERIFIED] Hard rails:** no P&L rebuild, reserved session, purchase, download, vendor/broker
  contact, order, feature proposal, or strategy adoption occurred.

## 7. Evidence and failure behavior

The immutable PASS attempt is
[`selector_quality_requirement_curve_2026_08_23_attempt001`](../../../v4/audit/autoresearch/selector_quality_requirement_curve_2026_08_23_attempt001/receipt.json),
self-hash `d616a07e0f05aef609b70c396e371fb822d6d1a8923a76229a5324b5930517bf`.
Its [readable tables](../../../v4/audit/autoresearch/selector_quality_requirement_curve_2026_08_23_attempt001/readable_tables.md)
and [full requirement curve](../../../v4/audit/autoresearch/selector_quality_requirement_curve_2026_08_23_attempt001/requirement_curve.csv)
bind 60 curve rows, 12 correlation rows, five inverse summaries, and all 505 inverse trace cells.

**[VERIFIED] Receipt integrity:** every artifact byte count and SHA-256 reproduces; archived wrapper
and analysis hashes match their live sources at issuance; the P&L input, producer, economics module,
alpha ledger, and semantic freeze were identical before and after. The semantic anchor is
`71463cc0eeb4e242e43307e19a926ea4e258fd45a254c389e5a27110923893c4`: **12 sources, zero drift**.

**[VERIFIED] Failure behavior:** a pre-existing directory or dangling symlink is untouched and exits
nonzero. After attempt creation, any exception preserves both source archives when available,
partial artifacts, full traceback in `run.log`, and a canonical self-hashed `failure_receipt.json`;
the directory is never reused. Focused coverage is **14 tests**; the full V5 suite is **1,243
passed, 105 warnings**. `check_project.py` and `git diff --check` are green.

## 8. What remains unknown

- whether any causally available feature has positive relationship to future P&L;
- genuine outcome-blind OOF value, stability, and generalisation gap;
- whether the Gaussian-oracle requirement transfers across quote sources or regimes;
- the original +$43.47 claim's actual rate, seed, metric, and partition provenance;
- joint-cell and dollar-cap executability; and
- any route to meeting the measured requirement.

No feature is proposed, no strategy is proposed, and nothing is adopted.
