# Signed order flow moves us toward a valid experiment, not a profitable bot

**Verdict — NO under current constraints; retain `STOP_UNDER_CURRENT_CONSTRAINTS` and do not acquire
or open outcomes.** **VERIFIED:** the free 229-session broad-band proposal is computationally workable
on the external SSD, and the selected owned slice contains bursty, short-lived signed-flow structure.
**VERIFIED:** the current semantic result is weaker than reported: exactly **226,288 / 874,002
(25.89%)** pinned signs take their "prior" row from the same receive and event timestamp, while the
stored schema has no sequence field and DBN/Parquet row-order identity was never checked. **INFERRED:**
strict-touch flow plus causal non-replenishment is a genuinely new research primitive, but it moves
the project closer to a well-posed experiment, **not materially closer to a profitable bot**.
**UNKNOWN:** economic edge, direct-P&L power, broad-band prevalence, live parity and executable fills.
At 229 session clusters the optimistic analytic 80%-power floor is an annualised Sharpe of **3.453**;
plausible Sharpe 1–2 effects need roughly **2,731–683 sessions**. There is no credible, powered,
affordable bot path today.

This finding proposes one terminal experiment below. It **adopts nothing**. On present evidence that
experiment would be refused before outcomes because causal order, CMBP-to-live parity, a law-current
runtime and policy-specific power are not certified.

## 1. Owner decision in one page

| Question | State | Evidence | Consequence |
|---|---|---|---|
| Can 29.8 billion rows be built here? | **INFERRED YES, external SSD only** | A chunked single-worker probe measures **8.52–9.39 h** for DBN→Parquet plus one feature pass; conservative footprint **1.071 TB** | Engineering is serious but does not kill the route. The internal disk does. |
| Does the owned flow have structure? | **VERIFIED, selected slice only** | Event direction repeats, clusters and bursts; 1-minute persistence is small and 5/15-minute autocorrelation includes zero | A short-lived signal could exist. No relation to anything later was measured. |
| Is the sign stream causally certified? | **NO — UNKNOWN for tied clocks** | 226,288 signs use equal receive/event timestamps; raw Parquet and locally inspected DBN expose no `sequence`; prior identity covered counts and price sum only | Repair order identity or freeze the strictly-earlier receive-time law before any broad build or outcome. |
| Is 229 sessions powered? | **UNKNOWN for the real P&L gate; analytically weak** | Direct session variance and occupancy are unknown. Optimistic 80%-power sensitivity detects only Sharpe ≥3.453 | Do not call 229 decisive without a policy-specific known-answer campaign. Two tickets do not make 458 independent days. |
| Is there a bot runtime? | **VERIFIED NO** | Generic parity helpers and a historical minute simulator exist; no CMBP live twin, event fill path or current-law risk engine exists | Even a positive research number would not be deployment evidence. |
| Closest prior family | **VERIFIED: W2-H02 option microstructure continuation** | That family was `NO_SIGNAL`; generic magnitude/activity was fully priced | Strict prints plus non-replenishment are new only if they beat an activity-matched reversed-sign control. |

**Proposed owner disposition, not an adoption:** refuse acquisition and outcomes now. If the owner
wants the last admissible route tested, first authorize only the outcome-blind causal-order repair and
the external streaming build. Continue toward one outcome exposure only if the free known-answer,
historical/live parity and law-current runtime prerequisites pass.

## 2. What was and was not measured

- **VERIFIED:** all new market measurements read only the frozen 64-session CMBP manifest: **64
  sessions, 173,470,783 rows, 1,588,281 trades**, 2024-10-01 through 2024-12-31. Every session is
  before the confirmation reservation.
- **VERIFIED:** the characterization read an explicit market-field allowlist. It read no label,
  return, P&L, entry/exit value or forward path; it ran no fit and charged no alpha.
- **VERIFIED:** the compute probe read owned DBN/Parquet only, contacted no network/vendor/broker,
  and created only benchmark outputs on the external evidence disk.
- **VERIFIED:** the power receipt is arithmetic over the existing protected alpha ledger and
  statistics helpers. It read no market outcome and charged no alpha.
- **VERIFIED:** no download, purchase, subscription, live connection, order, reserved session,
  protected-file edit or Member-Q run occurred.
- **UNKNOWN:** every economic question. Nothing here says flow predicts a return, produces a fill,
  survives the spread, or earns money.

The characterization receipt is
[`cmbp_signed_flow_characterization_2026_08_23_attempt002.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/cmbp_signed_flow_characterization_2026_08_23_attempt002.json),
self-hash `7d715901…`. Its implementation and targeted tests are
[`analyze_cmbp_signed_flow.py`](../../ops/analyze_cmbp_signed_flow.py) and
[`test_cmbp_signed_flow.py`](../../tests/test_cmbp_signed_flow.py).

## 3. The semantic pass needs a correction

The earlier receipt remains an exact result under its implemented **stable prior-row law**, but
"touch standing strictly before the print" and "faithful decode" were stronger claims than its
evidence supported.

| Order fact | State | Quantity | Uncertainty |
|---|---|---:|---|
| Pinned prior-row signs | **VERIFIED exact census** | 874,002 | No sampling interval: census of frozen bytes |
| Prior row has strictly earlier `ts_recv` | **VERIFIED exact census** | 647,714 (**74.11%** of pinned signs; **40.78%** of all trades) | No sampling interval: census |
| Prior row ties both `ts_recv` and `ts_event` | **VERIFIED exact census** | 226,288 (**25.89%**) | No sampling interval: census |
| Tied prior is a cross-publisher non-trade row | **VERIFIED exact census** | 199,728 | No sampling interval: census |
| Tied prior is another trade | **VERIFIED exact census** | 26,560 | No sampling interval: census |
| Session-equal tied share | **VERIFIED selected-slice estimate** | 22.93% | 95% whole-session bootstrap **[21.42%, 24.39%]** |
| Sequence/order field in local Parquet and independently inspected DBN record layout | **VERIFIED absent** | no `sequence` field | Exact local schema inspection; this does not establish what a future feed could retain |
| Causal order among tied rows | **UNKNOWN** | — | Needs canonical stream ordinal plus raw DBN→Parquet order identity, or a new strictly-earlier law |

**VERIFIED:** the earlier decode-identity receipt checked only row count, trade count and price sum on
three sessions. It did not compare row order or every event field. **INFERRED:** stable file order may
preserve the historical stream order, but that has not been proven, and cross-publisher ties make an
unstated merge-order assumption load-bearing.

The new receipt therefore reports
`OUTCOME_BLIND_CHARACTERIZATION_ONLY_CAUSAL_ORDER_UNVERIFIED`. Its primary tables preserve the pinned
law solely for comparability, and every headline has a sensitivity requiring
`prior_ts_recv < trade_ts_recv`. The strictly-earlier sensitivity shows that clustering survives;
it does **not** retroactively certify the tied quarter of the old signs.

## 4. Outcome-blind characterization of the selected slice

All intervals below are deterministic 10,000-resample, whole-session percentile 95% intervals over
64 sessions. Exact row/event counts are censuses and carry no artificial sampling interval. The
inference unit is the session, never the event.

### 4.1 Balance and persistence

Directional sign means call buy or put sell = bullish; call sell or put buy = bearish. Imbalance is
`(bullish weight - bearish weight) / total weight`.

| Metric, session-equal mean | Pinned prior-row law | Strict-earlier `ts_recv` | Reading |
|---|---:|---:|---|
| Count imbalance | **+1.10% [−0.17%, +2.54%]** | **+0.90% [−0.53%, +2.52%]** | **VERIFIED:** no stable aggregate directional tilt |
| Contract imbalance | **+1.89% [+0.23%, +3.77%]** | **+1.88% [−0.03%, +3.99%]** | **VERIFIED:** the small pinned tilt no longer excludes zero after strict ordering |
| Premium-notional imbalance | **+1.81% [−0.06%, +3.94%]** | **+1.80% [−0.27%, +4.13%]** | **VERIFIED:** no stable notional tilt |
| Adjacent same-direction probability | **68.19% [67.09%, 69.22%]** | **65.31% [64.34%, 66.23%]** | **VERIFIED:** strong event-level repetition |
| Independence baseline | **50.16% [50.09%, 50.25%]** | **50.20% [50.11%, 50.32%]** | Mechanical sign-mix baseline |
| Same-direction excess | **+18.02 pp [+16.90, +19.07]** | **+15.10 pp [+14.09, +16.05]** | **VERIFIED:** tied timestamps do not explain the cluster |
| Positive-time-gap same-direction excess | **+16.98 pp [+15.87, +18.01]** | **+15.13 pp [+14.11, +16.08]** | **VERIFIED:** exact-clock bundles do not explain it |
| Same direction, same instrument | **73.08% [71.66%, 74.39%]** | **69.75% [68.44%, 71.01%]** | Contract stickiness is substantial |
| Same direction, cross instrument | **55.61% [54.59%, 56.59%]** | **55.37% [54.02%, 56.50%]** | **VERIFIED:** a smaller cross-contract component remains |
| Mean same-direction run | **3.18 [3.09, 3.27] events** | **2.90 [2.83, 2.97]** | Short event runs |
| 90th-percentile run | **6.82 [6.55, 7.07] events** | **6.01 [5.79, 6.23]** | Short event runs |
| Net-contract ACF, 1 minute | **0.0238 [0.0011, 0.0466]** | **0.0321 [0.0101, 0.0550]** | **VERIFIED:** small one-minute persistence |
| Net-contract ACF, 5 minutes | **0.0089 [−0.0097, 0.0275]** | **0.0095 [−0.0085, 0.0276]** | **VERIFIED:** interval includes zero |
| Net-contract ACF, 15 minutes | **0.0120 [−0.0071, 0.0299]** | **0.0125 [−0.0074, 0.0317]** | **VERIFIED:** interval includes zero |

**INFERRED:** the only plausible mechanism is very short-lived pressure conditioned on book
non-replenishment. A five- or fifteen-minute aggregate direction rule is not supported by the flow's
own persistence. **UNKNOWN:** whether even the short event run leads rather than merely accompanies
the option repricing.

### 4.2 Clustering, arrival intensity, spread and size

The Fano factor is count variance divided by count mean; a value near one is Poisson-like, while a
large value means events arrive in bursts.

| Metric, session-equal mean | Estimate and 95% interval | State and meaning |
|---|---:|---|
| Signed events / selected-symbol-minute | **8.063 [6.365, 9.861]** | **VERIFIED selected slice:** high but unstable intensity |
| Session distribution of that rate | p10 **0.516 [0.234, 0.927]**; median **7.165 [4.343, 8.939]**; p90 **17.378 [14.019, 21.304]** | **VERIFIED:** observation supply varies by over an order of magnitude |
| One-minute event-count Fano | **54.73 [44.53, 65.76]** | **VERIFIED:** extreme clustering |
| Fano after computing inside fixed 30-minute bins | **17.16 [13.73, 20.84]** | **VERIFIED:** the open/close curve is not the whole cluster |
| Strict-earlier within-bin Fano | **13.11 [10.50, 15.91]** | **VERIFIED:** tied clocks are not the whole cluster |
| Median prior relative spread | **181.73 bps [164.11, 200.17]** | **VERIFIED:** expensive aggressive entry environment |
| Median prior total displayed depth | **41.20 [34.87, 47.70] contracts** | **VERIFIED selected slice** |
| Median trade size | **1.01 [1.00, 1.02] contracts** | **VERIFIED:** the typical print is one contract |
| Median trade / same-side touch size | **0.368 [0.294, 0.446]** | **VERIFIED:** typical print consumes less than the displayed touch |
| Share trade size ≥ same-side touch | **34.05% [30.29%, 38.19%]** | **VERIFIED:** depletion-sized prints are a minority |

Contemporaneous, descriptive Spearman correlations:

- **VERIFIED:** event intensity versus prior spread **+0.109 [+0.025, +0.195]** and versus prior
  depth **+0.252 [+0.158, +0.341]**. Busy flow is not automatically cheap flow.
- **VERIFIED:** absolute imbalance versus spread **+0.030 [−0.016, +0.076]** and versus depth
  **−0.106 [−0.163, −0.049]**. Imbalance magnitude is not materially separated by spread here.
- **VERIFIED:** trade size versus same-side depth **+0.191 [+0.155, +0.227]**.
- **VERIFIED:** direction versus directional prior-depth imbalance **+0.408 [+0.377, +0.436]**.
  This is a same-instant book association, not a forecast.

### 4.3 Across the day and across sessions

| ET bin | Signed events / selected-symbol-minute | Median prior spread | Directional contract imbalance |
|---|---:|---:|---:|
| 09:30–10:00 | **11.65 [9.43, 14.01]** | **121.2 [110.8, 132.7] bps** | +1.50% [−1.46%, +4.57%] |
| 12:30–13:00 | **5.00 [3.70, 6.38]** | **159.0 [144.1, 175.2] bps** | +1.92% [−5.70%, +9.32%] |
| 14:30–15:00 | **10.17 [7.04, 13.62]** | **277.4 [217.2, 351.5] bps** | +1.29% [−5.64%, +8.08%] |
| 15:30–16:00 | **16.39 [11.36, 22.08]** | **1,077.7 [690.7, 1,529.2] bps** | +0.56% [−6.46%, +7.62%] |

- **VERIFIED:** intensity is U-shaped and the touch becomes dramatically wider late in the day.
- **VERIFIED descriptive only:** of thirteen 30-minute imbalance bins, only 10:00–10:30 excludes
  zero, +7.28% [+2.61%, +12.58%]. No multiplicity correction was applied, so this is not a signal.
- **VERIFIED:** every checked first-32 versus last-32 difference includes zero: contract imbalance
  −0.28 pp [−3.96, +3.11], events/symbol-minute −0.082 [−3.61, +3.29], 1-minute ACF +0.0084
  [−0.0376, +0.0540], positive-gap excess +0.0039 [−0.0178, +0.0245], and median spread +15.2 bps
  [−20.7, +51.2].
- **INFERRED:** the selected quarter does not show a chronological break in these summaries.
  **UNKNOWN:** stability outside this selected quarter and on the unbiased band.

### 4.4 The selection caveat is terminal for prevalence

**VERIFIED design fact:** the owned files contain up to seven symbols per session selected around
roughly 31 prior-route trades. That selection chose instruments before this analysis and is not a
random full-band sample. **UNKNOWN:** market-wide imbalance, event rate, clustering strength, spread
mix and trigger occupancy. Whole-session intervals quantify variation inside the frozen selected
slice; they cannot repair its sampling design.

## 5. Compute feasibility: possible, but only through a streaming external build

Successful probe evidence:
[`cmbp_compute_feasibility_probe_2026_08_23_attempt003.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/cmbp_compute_feasibility_probe_2026_08_23_attempt003.json),
self-hash `a97e4a99…`. Five disjoint input-size strata were used for each phase.

| Phase | Measured rate range | Weighted rate | Peak RSS | 29.8B-row sensitivity |
|---|---:|---:|---:|---:|
| DBN read/decode | **4.547–4.931M rows/s** | 4.760M/s | 321–334 MB | **1.679–1.821 h** |
| DBN→Snappy Parquet | **1.040–1.115M rows/s** | 1.099M/s | 186–190 MB; 200 MB repeat | **7.422–7.960 h** |
| Parquet→outcome-blind features | **5.797–7.562M rows/s** | 7.167M/s | 553–783 MB | **1.095–1.428 h** |

- **VERIFIED measurement:** CPU time was approximately wall time, so the workers are effectively
  single-core. Immediate repeats were 2.10%, 1.87% and 0.89% slower rather than faster; build and
  feature repeat outputs were byte-identical.
- **INFERRED:** sequential build plus one feature pass is **8.52–9.39 hours**. One additional feature
  extraction pass is **1.10–1.43 hours** under the measured rates.
- **UNKNOWN:** these ranges are min/max sensitivities, **not confidence intervals**. No safe cold-cache
  flush was used; thermal behavior, checksum/archive time, parallel scaling and contention are unknown.
- **VERIFIED:** one proposed broad session has 130,310,192 rows, **11.9×** the largest benchmark
  input. **INFERRED:** chunking bounds working memory, but full-scale rate and memory remain unproven.
- **VERIFIED code fact:** the new owned-slice characterization and the old semantic verifier load a
  complete Parquet session. **INFERRED:** they are not broad-session ready on 16 GB RAM; the chunked
  feasibility pattern must replace the full-frame path before acquisition.

Storage:

| Item | State | Size |
|---|---|---:|
| Owned DBN density | **VERIFIED** | 14.189 bytes/row |
| Owned Parquet density | **VERIFIED** | 17.958 bytes/row |
| Proposed full DBN | **INFERRED** | 422.84 GB |
| Proposed Parquet at supplied/owned estimate | **INFERRED** | 535–550 GB |
| Proposed Parquet at actual probe-writer range | **INFERRED** | 584–623 GB |
| Feature output sensitivity | **INFERRED** | 8.26–25.41 GB |
| Internal free space at end | **VERIFIED snapshot** | 11.72 GB |
| External free space at end | **VERIFIED snapshot** | 1.951 TB |
| Conservative DBN + 623GB Parquet + 25.41GB features | **INFERRED** | **1.071 TB**, leaving **~879.7 GB** |

**Verdict on task 1:** `COMPUTE_FEASIBLE_EXTERNAL_STREAMING_ONLY`. Storage and one serial pass fit.
Internal storage and full-session pandas processing do not.

### Preserved feasibility failures

- **VERIFIED attempt 001:** failed after 9.5 s because Databento 0.77 passed unsupported
  `row_group_size` into the installed PyArrow writer. Failure receipt self-hash `609e52a9…`; the
  zero-byte partial file remains.
- **VERIFIED attempt 002:** all DBN builds completed, then the feature reader failed because
  `ts_recv` was restored as the pandas index instead of a named column. Failure receipt self-hash
  `10498ede…`; six completed outputs totaling 623,031,939 bytes remain.
- **VERIFIED attempt 003 failure law:** worker errors exit 5, write a self-hashed failure receipt and
  preserve/inventory partial outputs; invariant failures exit 4; existing receipts and outputs are
  refused. **UNKNOWN:** a disk-full/import/receipt-I/O failure before finalization cannot guarantee a
  receipt.

## 6. Power at 229 sessions

The outcome-free arithmetic is archived in
[`signed_flow_power_scope_2026_08_23_attempt002.json`](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/signed_flow_power_scope_2026_08_23_attempt002.json),
self-hash `448b894b…`. It pins the protected ledger at six prior experiments and prices a proposed
seventh exposure at one-sided `alpha = .05 / 7 = .007142857`, `z = 2.449998`.

| Sensitivity | 229-session result | State |
|---|---:|---|
| Optimistic single-statistic annualised Sharpe detectable with 80% power | **3.45296** | **VERIFIED arithmetic**, assumes independent sessions and no gate penalty |
| Two-point directional accuracy detectable with 80% power | **68.7225%** | **VERIFIED arithmetic**, published payoff analogue only |
| Exact-binomial critical count | **152 / 229 = 66.3755%** | Null tail 0.00571 |
| Exact power if true accuracy is current 66.0595% bar | **49.05%** | **VERIFIED arithmetic** |
| Exact power if true accuracy is 68.6595% | **79.37%** | Existing +2.6 pp lower-bound-gap sensitivity |
| Sessions for 66.0595% true accuracy in the two-point analogue | **405** | Published rounded threshold |

The directional analogue is not the proposed economic score. The real primary statistic is session
P&L and its paired lift over a control.

- **UNKNOWN:** broad-band trigger occupancy, session-P&L standard deviation, paired-control
  covariance and the owner-signed minimum worthwhile entry effect. Therefore actual 229-session
  power is **not certifiable now**.
- **VERIFIED analytic sensitivity:** at attempt-seven alpha, 229 sessions can see only
  `0.217516 × session-P&L SD` at 80% power. For a **proposed, not signed** +$25 per executed ticket
  minimum and one ticket/session, power requires session SD ≤ **$114.93**. No such low dispersion has
  been established.
- **INFERRED optimistic analogue:** the already-published two-point option payoff has about $362
  per-trade SD at break-even; transplanting it would require roughly 2,273 one-ticket sessions to see
  +$25. This is a scale illustration, not a policy variance estimate.
- **VERIFIED session-count sensitivities:** Sharpe 1.0 / 1.5 / 2.0 / 2.5 / 3.0 need **2,731 / 1,214 /
  683 / 437 / 304 sessions**, respectively. The signed Tier-A exit requirement remains separately
  about **2,560 outer sessions**.
- **VERIFIED:** two tickets inside one day remain one cluster. They may change mean and variance; they
  do not turn 229 days into 458 independent observations.
- **INFERRED:** the real conjunction—two lower bounds, 4/5 fold signs, controls, latency and risk—will
  be no easier than the analytic one-statistic curve. A synthetic, session-clustered known-answer
  campaign must certify the exact gate before real outcomes are opened.

## 7. Proposed cheapest decisive experiment — not authorized, not run

### Hypothesis and mechanism

**INFERRED hypothesis:** option buyers/sellers who trade at the strict prior touch and leave the
consumed side unreplenished reveal directional pressure that persists beyond the complete live/order
latency long enough for one long SPXW option bought at the ask and sold at the bid to earn positive
net P&L. The control distinguishes signed direction from generic activity, volatility and expensive
options.

For signed event `i`:

- `a_i = +1` for a prior-ask print and `−1` for a prior-bid print;
- `r_i = +1` for a call and `−1` for a put;
- `d_i = a_i × r_i` is bullish/bearish direction;
- `w_i = 100 × print_price × contracts` is premium notional;
- `N_i = 1` only when, after a frozen causal response delay `R`, the consumed touch has moved away or
  remains at its price with smaller displayed size.

At decision time `tau`, the closed-form score is:

`F_tau = sum(d_i × w_i × N_i)` for `t_i in (tau − R − 5 seconds, tau − R]`.

**UNKNOWN today:** numeric `R`. It must be frozen before outcomes from same-schema local-receipt and
complete IBKR order-path latency, rounded upward to the next 100 ms. CBBO latency does not transfer.
If no such receipt exists, the experiment is `INVALID`, not approximately timed.

### Population and policy

- **PROPOSED:** all 229 unbiased full ±25-point sessions, including no-trigger days as zero P&L;
  five chronological folds; session is the inference unit.
- **PROPOSED:** before outcomes, freeze the 99.9th percentile of `|F|` separately in fixed 30-minute
  ET bins from the outcome-blind broad flow distribution. This is one threshold law, not a search.
- **PROPOSED:** take only the first threshold crossing while flat and at most the first two completed
  tickets in a session. Positive `F` buys the deterministic nearest eligible OTM call; negative `F`
  buys the put. Resolve ties by absolute moneyness, then lower ask, then raw OSI symbol.
- **PROPOSED current-law replay:** one contract; premium plus fees ≤$2,000; one open position; no
  entry after 15:00; ask entry after certified feature/decision/order latency; first causal bid exit
  at a −40% stop or at 60 minutes; 20% session-start-equity breaker; maximum two tickets/day; serial
  $10,000 compounding; 50% survival floor; no overnight position. Gap/slippage is retained.
- **PROPOSED:** settlement-source and zero-recovery twins must agree in verdict sign. This is a new
  −40% exit law, so the existing Member-P +50%/−30% discharge does not transfer. Member Q remains
  preserved and is not run.

### Exact statistic and control

For each session `s`, including zeros:

- `Y_s` = net P&L of the candidate serial policy;
- `C_s` = net P&L at the same trigger times and `|F|`, with call/put direction reversed under the
  identical fill and risk law;
- `D_s = Y_s − C_s`.

The primary statistic is

`T = min(LCB_alpha(mean(Y_s)), LCB_alpha(mean(D_s)))`,

using a five-session moving-block bootstrap at `alpha=.05/7`. The reversed-sign control isolates
direction from magnitude/activity. Constant-sign, causal-lag and whole-session-shuffle controls must
also fail the same gate.

### Predeclared verdict and falsification

**PROPOSED PASS**, all conjunctive:

1. `T > 0`;
2. absolute and paired mean signs are positive in at least 4/5 chronological folds;
3. point-estimate net P&L is at least **+$25 per executed ticket**—a proposed entry minimum, not a
   transfer of the signed exit tier; the owner must accept or replace it before outcomes, with power
   recomputed;
4. reversed, constant, lagged and shuffled controls do not pass;
5. causal-order, parity, latency, fill, settlement/zero-recovery, maximum-loss, breaker, serial-account
   and 50%-survival checks all pass.

**PROPOSED terminal rulings:**

- synthetic known-answer cannot recover the approved effect with ≥80% probability:
  `UNDERPOWERED`, before real outcomes;
- order/parity/law/control machinery fails: `INVALID`, no economic interpretation;
- powered real test fails any economic condition: `NO_TRADABLE_SIGNAL`, family closed with no retune;
- all conditions pass: `RESEARCH_PASS_ONLY`, still no paper/live authority.

**Falsification:** the mechanism is false if direction reversal performs as well, if a causal lag or
session shuffle passes, if the apparent effect disappears after matching activity/premium/spread, if
lead does not exceed full p99 latency, or if strict-earlier/order-verified signs disagree in verdict.

### Alpha cost

- **VERIFIED:** causal-order repair, broad-flow characterization, threshold freeze, compute benchmark,
  same-schema parity and a purely synthetic known-answer campaign are outcome-free and charge zero.
- **PROPOSED:** the decision to open any real label/path/P&L is one experiment, ledger **6→7**, even if
  the run is later `PASS`, `FAIL` or `REFUSED`. It requires a verified `DeclaredFitPermit`, declaration
  and journal through [`outcome_run_gate.py`](../outcome_run_gate.py); none exists now.
- **VERIFIED:** this session opened no economic outcome and did not change the ledger.

## 8. The system question: what exists and what does not

| Layer | Exists now | Missing before any outcome fit or bot claim |
|---|---|---|
| Historical CMBP ingestion | **VERIFIED:** selected DBN/Parquet files, strict prior-row parser, chunked feasibility pattern | **MISSING:** authorized broad downloader/build, streaming semantic/characterization path, canonical stream ordinal, full-field and row-order DBN↔Parquet identity, correction/package handling |
| Runtime symbology | **VERIFIED:** generic current-session definitions, raw OSI handling and `SymbolMappingMsg` concepts in [`training_twin.py`](../training_twin.py) | **MISSING:** CMBP live definitions and mapping handler; zero-partial raw OSI→IBKR `conId`/`localSymbol` receipt |
| Causal clock | **VERIFIED:** generic minute clock/selectors and historical latency/freshness receipt machinery | **MISSING:** one CMBP event decoder/feature builder shared by history and live; event/receive/vendor-output/local-monotonic/decision/submission/ack/fill clocks; tied-event total order; frozen `R` |
| Train/live parity | **VERIFIED:** generic paired-frame comparator | **MISSING HARD GATE:** same-schema no-order live-first capture then delayed historical raw value/order/correction/definition/feature equality; gap, drop, reconnect and slow-reader receipts; no skipped records |
| Fill law | **VERIFIED:** minute simulator has ask-in, bid-out, first later bid and settlement branches | **MISSING:** first attainable event quote after certified latency; marketable-limit behavior, partials, rejects, cancels, gaps and measured slippage; parallel IBKR shadow |
| Risk law | **VERIFIED PARTIAL:** $2,000 candidate cap; one-open-position and serial account foundations. Full signed tuple is represented only in hypothetical [`scale_sensitivity.py`](../scale_sensitivity.py) | **MISSING:** integrated runtime/replay enforcement. [`causal_day_simulator.py`](../../ops/causal_day_simulator.py) still has a 5% breaker, accepts 1/2/3 caps and has no mandatory percentage stop; signed law is 20%, max two and −40%-or-wider |
| Paper-before-live | **VERIFIED LEGACY ONLY:** frozen v4 paper guard/executor exist | **MISSING:** v5 CMBP no-order shadow, current-law paper adapter, state reconciliation, kill-switch evidence and paper survival. Neither paper nor live is authorized |

The architecture remains the standing finding's least-divergent route: Databento CMBP for both
historical and live features, IBKR for execution. It requires all eight prerequisites in
[`DATABENTO_SUBMINUTE_DIRECTION_DECISION_2026_08_22.md`](DATABENTO_SUBMINUTE_DIRECTION_DECISION_2026_08_22.md#hard-gate-architecture).
No current code certifies that architecture.

## 9. Prior-art comparison and honest novelty ruling

**VERIFIED closest dead family:** Path-D Wave-2 W2-H02 option `microstructure continuation` in
[`DO_NOT_RETEST.md`](../history/DO_NOT_RETEST.md) was `NO_SIGNAL` over 156,950 OOF candidates, 166
sessions and five folds. The generic long-premium magnitude/activity family also found that high
activity makes the option dearer without improving net P&L.

**VERIFIED novelty boundary:** strict individual prints signed against a causally prior touch plus
post-print non-replenishment cannot be reconstructed from CBBO-1m. It is a materially new observable,
explicitly permitted by the conditional-drift census's "new information source" clause. It is not a
new payoff law, position structure or execution cost.

**INFERRED fate:** if the proposed rule merely selects busy, wide or rich options, it shares W2-H02
and the magnitude family's fate. It escapes that family only if absolute ask/bid P&L and lift over
the activity-matched reversed-sign control both clear. Flow persistence by itself is not enough.

Chain internals are the second useful analogy: they produced genuine ordering worth roughly
**+$8.43/trade** yet did not close an approximately **$24** economic gap. Signed flow could likewise
contain real ordering and still be worthless to a long buyer. The proposed +$25 minimum makes that
distinction explicit.

## 10. What could not be determined

- **UNKNOWN — economic edge:** prohibited because no outcome was authorized or read.
- **UNKNOWN — direct-P&L power at n=229:** trigger occupancy, session variance and control covariance
  require a frozen policy and lawful power procedure.
- **UNKNOWN — unbiased prevalence and structure:** the 229 broad sessions were not downloaded.
- **UNKNOWN — causal order for tied historical events:** no sequence field or row-order identity.
- **UNKNOWN — package/complex-leg meaning:** consolidated L1 cannot prove package identity; the test
  must fail closed or neutralize it before interpretation.
- **UNKNOWN — response interval and lead margin:** same-schema live and broker receipts do not exist.
- **UNKNOWN — sub-minute SPX source/parity:** required for deterministic OTM call/put selection.
- **UNKNOWN — live entitlement, schema availability and account cost:** no vendor contact was allowed.
- **UNKNOWN — executable fills/slippage:** no current-law no-order shadow exists.
- **UNKNOWN — one-era stability:** 229 sessions remain about eleven months even if acquired.
- **UNKNOWN — Tier-A certification:** 229 is far below the signed ~2,560 outer sessions and cannot
  certify exit timing.

These are measurement limits, not negative evidence. They are also why the present verdict is a stop.

## 11. Evidence integrity and verification

New immutable evidence:

- Compute attempts 001/002/003: sibling `.py`, `.json` and `.log` files under
  `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/`; failed attempts and partial external
  outputs preserved.
- Flow attempts 001/002: sibling wrapper/log plus immutable failure/success receipts in the same
  archive. Attempt 001 failed before market access because the nested wrapper lacked repo-root import
  setup; attempt 002 changes only that bootstrap.
- Power attempts 001/002: attempt 001 invocation failure is preserved; its successful receipt was
  superseded because floating `ceil` reported 406 instead of the constructive rounded 405-session
  identity. Attempt 002 records that numerical correction without reading outcomes.

Wrapper failure behavior:

- **VERIFIED compute:** caught workers exit 5 with immutable self-hashed failure evidence and retain
  outputs; invariants exit 4; overwrite refused.
- **VERIFIED characterization:** no session is silently skipped; an input/invariant error writes one
  immutable failure receipt and exits 5; existing success/failure receipts cause exit 6 before market
  reads.
- **VERIFIED power:** changed pinned ledger/statistics hash, malformed ledger, existing output or
  self-hash mismatch aborts; no market data path exists in the wrapper.

Artifact checks, including a final rerun after this finding and its LOG entry:

- characterization receipt self-hash and implementation/manifest/wrapper hashes reproduce;
- all three compute receipt self-hashes reproduce; successful wrapper and all thirteen output hashes,
  Parquet footers and row/event partitions verify;
- both power receipt self-hashes reproduce; attempt 002 supersedes attempt 001 numerically;
- targeted signed-flow tests: **9 passed**;
- project checker: **PASS**;
- full v5 suite: **1,159 passed**, 105 pre-existing numerical warnings in 70.72 seconds;
- `git diff --check`: clean.

Final law: **no acquisition, no outcome and no bot claim.** Signed flow is a credible new question.
It is not yet a credible, powered, affordable answer.
