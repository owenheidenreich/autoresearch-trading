# Databento sub-minute direction decision — 2026-08-22

**Owner-facing verdict: `STOP_UNDER_CURRENT_CONSTRAINTS`.** There is no credible, certifiable path
from the presently authorized data and runtime to a profitable SPXW 0DTE long-option bot. The sole
scientifically coherent hypothesis left on paper is **strict trade-at-touch option flow conditioned
on the book's causal response**. It is not ready for acquisition or fitting: broad event history is
not owned, the exact all-session price is unknown and the current samples imply a cost above the
remaining charter, pre-2023 equivalent event data does not exist, a live train/twin receipt does not
exist, the v5 runtime does not enforce current risk law, and entry-specific power is unknown.

The single recommended owner decision is therefore **stop**. If the owner wants one more piece of
decision support despite that recommendation, the only admissible next action is an **outcome-blind
semantic gate on the 64 selected-symbol `cmbp-1` sessions already owned**. It may answer whether the
event primitive is reconstructible; it may not inspect outcomes, estimate an edge, authorize a
download, or turn the selected slices into a training corpus.

`ohlcv-1s` is **not** the cheap implementation of that hypothesis. It is a one-second aggregate of
last trades, without individual prints, a causally prior BBO, touch location, aggressor, sequencing,
or post-trade quote response. Its nominal 60-times-finer clock does not make it equivalent to event
microstructure. It is vetoed as an alpha direction even though sampled cost is zero.

Nothing in this finding is adopted. No fit, new outcome statistic, reserved session, market-data
download, purchase, subscription, vendor contact, broker connection, paper order, live order, or
protected-file edit occurred. Member Q remains **PRESERVED, NOT RUN**.

## Claim language and scope

- **VERIFIED** — reproduced from repository evidence, a retained metadata receipt, or a cited
  primary vendor document.
- **INFERRED** — arithmetic or a mechanism judgment from verified premises, with its assumption
  stated.
- **UNKNOWN** — not measured or not available under current authority. `UNKNOWN` is not a soft pass.

The trading game remains exactly one long, single-leg, same-day SPXW call or put at a time. Spreads,
short premium, other expirations, other instruments, cross-asset features, and a calendar-event
strategy remain outside scope. SPX may appear only as causally available context or a reaction
control. Every eventual policy must obey the signed one-contract, $2,000 premium-plus-fees cap,
two-tickets-per-session law, one-open-position law, 20% session-start-equity breaker, declared
simulator-enforced stop no tighter than -40%, serial compounding account, and 50% survival floor.

## 1. Ranked decision

| Rank | Direction | Ruling | Why |
|---:|---|---|---|
| **1** | **Strict trade-at-touch flow plus post-trade book resilience** | **SOLE PAPER SURVIVOR; STOPPED BEFORE ACQUISITION/FIT** | Actual prints joined to a causally prior touch and followed by replenishment/non-replenishment are materially different from minute microprice, imbalance, momentum, and last-trade bars. The exact historical, live, cost, runtime, and power gates are not cleared. |
| **2** | Same-right cross-strike or cross-venue sweep clusters | **VETO AS A STANDALONE FAMILY** | It is potentially a diagnostic inside rank 1, but OPRA consolidated L1 has no order/package identifier or full depth. Spread, overwrite, roll, hedge, and complex-leg ambiguity can reverse the apparent sign. |
| **3** | Option flow leading synchronized intra-minute SPX | **EXECUTABILITY CONTROL ONLY** | It is useful only to prove that rank-1 flow leads rather than reacts. The sub-minute SPX source and same-clock twin are unknown, and any lead shorter than live feature-plus-order latency is untradeable. Raw SPX momentum is closed. |
| **4** | Transient spread compression | **EXECUTION ADJUNCT ONLY** | Removing the typical entry half-spread can save roughly $10; direct round-trip friction is about $23. Compression cannot create alpha or rescue a zero-gross policy. |
| **5** | `ohlcv-1s` path, burst, volume, or call/put activity | **HARD VETO FOR ALPHA** | It is aggregated last-trade tape, cannot identify signed flow or book response, reopens the closed print/chart/magnitude family at a finer clock, and supplies no executable bid/ask label. |
| **6** | Quote revision/intensity without trades | **HARD VETO** | It directly resembles the dead W2-H02 microstructure-continuation composite and the fully priced magnitude/activity family. Conditioning it on actual trades collapses it back into rank 1. |

The distinction is deliberate: **rank 1 is the best scientific idea; stop is the best decision**.
Calling the idea “rank 1” does not make the route affordable, powered, or deployable.

## 2. Sole surviving hypothesis contract

### Mechanism

At each option print, reconstruct the last same-contract consolidated BBO whose **receive time is no
later than the print**. Retain only strict, unambiguous ask lifts or bid hits; refuse inside-spread,
outside-spread, stale, crossed, locked, corrected, or order-ambiguous events. OPRA does not publish a
usable aggressor side, so “aggressor” is an inference from the prior touch, never a vendor fact.

After a future response interval fixed from certified live latency—not chosen from outcomes—measure
whether same-side liquidity replenishes, the touch steps, the spread recovers, and adjacent strikes
revise coherently. The directional mapping is:

- call ask lift or put bid hit: bullish pressure;
- put ask lift or call bid hit: bearish pressure.

Candidate primitives are signed premium notional, contract count, distinct strikes and publishers,
strict-touch share, same-side replenishment, touch stepping, and quote-response breadth. A sweep
cluster may be reported only after conservative neutralization of same-time opposing/package-like
legs; it is not a second hypothesis family.

**INFERRED mechanism:** urgent option demand that consumes the touch and is not replenished may
precede a rare repricing large enough to overcome long-premium friction. The book-response condition
is load-bearing: without it, the feature is activity/magnitude, already measured as fully priced.

**Adversary's strongest objection:** an apparent ask lift may be a hedge or one leg of a package,
and the quote response may already be an automated reaction to SPX. A retail IBKR order may arrive
after the response. Consolidated L1 cannot recover the package, queue, or full-depth counterfactual.
Those are mechanism threats, not implementation details.

### Exact data required and what is owned

| Item | Required role | Ownership and adequacy |
|---|---|---|
| Same-day `definition` | Exact OSI, expiry, strike/right, current-session mapping | **VERIFIED owned** across the 1,014-session lifecycle corpus. Historical and live numeric instrument IDs cannot be carried across sessions. |
| `cmbp-1` | Every consolidated top-of-book update and trade; price/size, publisher, sequence, event and receive clocks; pre-trade reconstruction and post-trade response | **VERIFIED partly owned:** 64 selected-symbol sessions, 2024-10-01 through 2024-12-31, 173,470,783 rows across 248 session-symbols. DBN and Parquet are present. They were selected around a prior manifest/outcome route and are suitable only for parser/semantic checks, not unbiased economics. Broad same-day-band history is not owned. |
| `tcbbo` | Every print plus the CBBO immediately before it; a strict-touch cross-check | **VERIFIED not evidenced locally** for this branch. It is redundant if a complete `cmbp-1` decoder is trusted and cannot by itself observe every post-trade quote update. |
| `cbbo-1s` | Lossy one-second quote/sale cross-check | **VERIFIED partly owned:** 175 selected-symbol sessions, 2025-07-01 through 2026-03-30, 18,287,307 rows across 805 session-symbols. It cannot reconstruct exact update order or resilience. |
| `trades` | Print price, size, venue publisher and clocks | Broad history is **not owned**. Without a causally prior BBO, OPRA's `side=N` makes touch sign ambiguous. |
| Synchronized SPX event path | Reaction control and executable lead margin only | **UNKNOWN.** No homogeneous sub-minute SPX source, historical/live twin, cost, or admitted feature contract is established. ES/VIX substitution is barred. |
| Full depth / order IDs / queue position / complex package ID | Package and fill disambiguation | **VERIFIED unavailable from OPRA consolidated L1.** CMBP is consolidated top-of-book, not MBO or MBP-10. |

The owned high-resolution inventory is recorded in
[`databento_protocol101_highres_downloads.jsonl`](../../../v4/audit/databento_protocol101_highres_downloads.jsonl).
It does not repair the [`NOT-USABLE` two-era corpus](TWO_ERA_SPXW_CORPUS_AUDIT_2026_08_22.md): it is
selected-symbol rather than a broad ladder, supplies no clean independent holdout, and has no
event-schema live twin.

Official schema semantics are documented in Databento's
[OPRA dataset specification](https://databento.com/docs/knowledge-base/datasets/opra-pillar),
[CBBO/CMBP/TCBBO schema guide](https://databento.com/docs/schemas-and-data-formats/cbbo), and
[equity-options example](https://databento.com/docs/examples/options/equity-options-introduction/using-parent-symbology-to-fetch-an-option-chain).
Those sources verify that OPRA is consolidated L1, OPRA trade side is `N`, TCBBO attaches the
pre-trade CBBO, and CMBP carries consolidated top-of-book updates. They do not establish an edge.

### Effect size, sessions, calendar, and alpha

| Requirement | Current answer |
|---|---|
| Direct executable hurdle | **VERIFIED:** about **$23/trade** = $19.92 mean option spread + $3.08 fees. The $17.92/0.358-point ES hurdle is a different quantity. A predictive entry mechanism must create more than $23 gross option value merely to break even. |
| Minimum worthwhile entry effect | **UNKNOWN:** no entry-specific dollar/capture effect has been signed for this route. “Positive after cost” is necessary but is not a power target. |
| Required independent sessions | **UNKNOWN:** signal prevalence, serial policy, session variance, target and minimum effect are unfrozen. No within-decade certification claim is defensible. |
| Available event-era ceiling | **VERIFIED structural count:** 813 of the 1,014 corpus-session ladder files are on or after `cmbp-1`/`tcbbo` coverage start 2023-03-28. The three current quote-source liveness exclusions occur inside this era, but they cannot automatically be transferred to a new event source. The maximum is a ceiling, not a usable sample. |
| Recent free-era ceiling | **VERIFIED structural count:** 227 corpus sessions are on or after 2025-08-25. They are one recent regime, not an independent fit/validation/confirmation stack. |
| Exit-timing use | **VERIFIED veto:** signed Tier A requires about 2,560 independent outer-holdout sessions, about 10.2 years, in addition to entry validation and exit training. Rank 1 must be an entry hypothesis first; it cannot reopen exit work. |
| Historical data cost | **INFERRED planning:** about $148.60 for the CMBP covered/paid partition under the sampled-price/free-era assumptions in §3; exact total **UNKNOWN** and current authorization remaining is $36.51. |
| Compute and storage | **UNKNOWN:** the owned selected-symbol CMBP slice alone is 173,470,783 rows. No full-band decode/storage dry probe exists, and message volume rather than session count drives work. Metadata pricing itself is negligible local compute. |
| Waiting cost | **UNKNOWN for entry certification.** No route-specific power law exists. Waiting cannot recreate equivalent pre-2023 events; future dates remain confirmation-only. |
| Alpha cost of schema/semantic/live feasibility | **VERIFIED:** zero outcome-bearing attempts. |
| Alpha cost of one later declared outcome run | **VERIFIED:** one slot, ledger 6 to 7, whether PASS, FAIL, REFUSED, or abandoned. The current attempt-7 directional lower-bound bar is `0.66059463`, with planning true accuracy `0.68659463`, on the ledger's existing 405-session directional geometry only. **INFERRED:** after charging it, the next bar is about `0.66176472`. None is a P&L, capture, or spread-savings bar. |

Messages, prints, strikes, candidates, seconds, and two tickets do not multiply the economic sample.
The unit is the **session**. The existing 644 trajectories in 323 clusters had within-session
effective size near 385—about 1.19 effective observations per session, not two independent trades.

### Crisp falsification

The hypothesis is killed before outcomes if any of the following holds:

1. A print cannot be joined, under receive-time causality, to exactly one valid prior BBO; or DBN
   and Parquet decoders disagree on the retained event set.
2. Historical CMBP reconstruction and an independently decoded trade-attached BBO cannot agree on
   strict touch classification, corrections, order, and current-session symbols.
3. Package-like same-time opposing legs cannot be conservatively neutralized. This kills the sweep
   diagnostic; if ambiguity contaminates strict single-print classification, it kills the family.
4. A same-schema no-order live twin cannot reproduce symbols, event values, ordering, revisions,
   liveness and causal features; any slow-reader skip is a hard failure.
5. `q05(observed option-flow lead) - p99(feature + decision + IBKR submission latency) <= 0`.
   A historical lead with no positive live margin is not executable.
6. No signed entry effect and session-clustered known-answer power receipt exists before outcomes.
   This is `STOP_UNPOWERED`, not permission to look.
7. In a later, separately authorized outcome experiment, direct ask-in/bid-out serial economics fail
   to clear zero and the then-signed minimum effect with corrected session-level lower bounds and all
   account-risk gates.

Passing items 1-5 proves only that the primitive can exist live. It does not prove predictiveness.

## 3. Pricing: what the retained preflight does and does not establish

The concurrent owner-authorized preflight made metadata and symbology calls only; it downloaded no
time-series data and spent nothing. Attempt 1 is correctly retained as **VOID** because parent-scope
pricing failed the known-answer control by about 20 times. Attempt 2 used the existing ±25-point
corpus band—32 to 56 raw OSI contracts per sampled session—over RTH. Its one-sided `cbbo-1m`
control passed at $0.002497/session versus the broader receipted $0.024238/session.

Receipts:

- [attempt 1, VOID](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/depth_preflight_attempt001_VOID_2026_08_22.json), SHA-256 `e0064b27...`;
- [attempt 2](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/depth_preflight_attempt002_2026_08_22.json), SHA-256 `630c7e4a...`;
- [decision receipt](../../../v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/depth_preflight_receipt_2026_08_22.json), SHA-256 `88768997...`.

### Exact sampled quotes

| Schema | Coverage in retained metadata | Exact sampled `get_cost` result | Information limit |
|---|---|---:|---|
| `ohlcv-1s` | 2013-04-01 onward | **$0** on all five sampled sessions; nonempty counts verified on two sessions | One-second OHLCV trade aggregates; no BBO, individual event order, touch, aggressor, or response. |
| `trades` | 2013-04-01 onward | $2.240639, $4.405625, $4.398665 on the three paid-era samples; $0 on two recent samples | Individual prints, but no prior BBO and OPRA side is `N`. |
| `tcbbo` | 2023-03-28 onward | $5.507031 and $5.498332 on the two paid, in-coverage samples; $0 on two recent samples | Print plus pre-trade CBBO; not the complete post-trade quote path. |
| `cmbp-1` | 2023-03-28 onward | **$0.185384 and $0.321770** on the two paid, in-coverage samples; $0 on two recent samples | Correct rank-1 substrate, consolidated L1 only. |
| `cbbo-1s` | 2025-02-20 onward | $0 on the two recent samples; no paid in-coverage sample was retained | Time aggregate; no update ordering or revision intensity. |

**VERIFIED:** `$0` for `ohlcv-1s` did not mean “no data” on the two record-count probes: the band
returned 151,160 records on 2023-06-27 and 287,551 on 2025-11-04. **INFERRED, not exact:** all 1,014
sessions cost $0. Only five cost samples and two nonempty-count samples were taken; an all-session
quote/count census was not.

### Planning totals, corrected for coverage

The attempt-2 `usd_full_corpus_1014` fields multiply a paid-session mean by all 1,014 sessions. That
is a useful scale warning but not an exact acquisition quote, and it overstates schemas that start in
2023 and show zero price in the recent era.

| Candidate | Arithmetic | Planning total | Status |
|---|---:|---:|---|
| `trades` paid portion | 787 paid sessions times $4.630390 boundary-sample mean | **~$3,644.12** | **INFERRED**; exact all-session sum unknown. |
| `cmbp-1` covered paid portion | 586 covered, pre-recent sessions times $0.253577 two-sample mean | **~$148.60** | **INFERRED**; assumes the observed recent zero-price regime applies across the 227 recent sessions. Exact sum and exact free boundary are unknown. |
| `tcbbo` covered paid portion | 586 times $5.502681 two-sample mean | **~$3,224.57** | **INFERRED** under the same assumption. |
| `ohlcv-1s` full corpus | five sampled costs at zero | **$0** | **INFERRED** until every declared session is quoted and counted. It is scientifically vetoed regardless. |

The receipt's naive 1,014-session extrapolations—about $257 for CMBP and $5,580 for TCBBO—must not
be presented as exact totals. The coverage-partitioned estimates are more decision-relevant and
still exceed the remaining **$36.51** charter by about 4.1 times for CMBP and 88 times for TCBBO.
Because they remain estimates, the exact monetary total is **UNKNOWN**. Databento bills historical
data by uncompressed DBN bytes; message-volume schemas cannot be extrapolated from the old
`cbbo-1m` $/session receipt. See the official
[historical metering and metadata guide](https://databento.com/docs/api-reference-historical/metadata/metadata-get-dataset).

## 4. Exact next preflight—design only, not run

No further vendor call is recommended under the current stop. If the owner changes the budget or
requires an exact decision receipt, the preflight must have this exact shape.

### A. Freeze scope

1. Hash the 1,014 non-reserved session manifest and the per-session existing ±25-point band. Every
   current corpus date precedes the 2026-08-06 confirmation reservation.
2. Convert each frozen contract ID to the exact 21-character SPXW OSI symbol; require root `SPXW`,
   expiry equal to the session, right in `{C,P}`, a valid eight-digit strike, uniqueness, sorted
   order, and a retained count/hash.
3. Resolve that exact raw-symbol set with Databento symbology for `[D,D+1)` and require a complete
   mapping. Never price the `SPXW.OPT` parent.
4. Freeze the exchange calendar and the exact RTH half-open interval. No silent full-day or
   parent-scope widening is allowed.

### B. Quote each session separately

For each schema candidate, call only:

```python
client.metadata.get_cost(
    dataset="OPRA.PILLAR",
    schema=SCHEMA,
    symbols=raw_osi_band_for_D,
    stype_in="raw_symbol",
    start=rth_open_utc,
    end=rth_close_utc,
)

client.metadata.get_record_count(
    dataset="OPRA.PILLAR",
    schema=SCHEMA,
    symbols=raw_osi_band_for_D,
    stype_in="raw_symbol",
    start=rth_open_utc,
    end=rth_close_utc,
)
```

Quote `cmbp-1` first because it alone supplies both prints and every consolidated top-book response.
Do not add redundant TCBBO or derived OHLCV to that request. Mark every date before 2023-03-28 as
`STOP_SCHEMA_UNAVAILABLE`; do not fall back to `cbbo-1m`. Sum the actual per-session returns—never a
mean times a calendar count. A zero-cost session passes only with a positive record count.

### C. Receipt and failure law

The immutable receipt must carry session and symbol hashes, schema and coverage, exact UTC bounds,
every per-session cost/count, exact sums, code/dependency hashes, request hashes, the API method
allowlist, and proof that no time-series method ran. Required dispositions are:

- `STOP_MISSING_KEY_OR_ENTITLEMENT`
- `STOP_SYMBOL_OR_EXPIRY_MISMATCH`
- `STOP_SCHEMA_UNAVAILABLE`
- `STOP_ZERO_RECORDS`
- `STOP_NONFINITE_COST`
- `STOP_SCOPE_OR_CODE_DRIFT`
- `STOP_OVER_HARD_CAP`
- `PREFLIGHT_PASS_ONLY`

Any error fails the candidate; no session is silently dropped. `PREFLIGHT_PASS_ONLY` means only that
the request is available and within a newly owner-signed cap. It is not download authority.

### D. Outcome-blind semantic gate on already-owned CMBP

This is the only next action with nonzero decision value under the current stop, and it needs no new
data. Freeze the 64-session/248-session-symbol manifest, forbid every outcome/label/P&L table, and:

1. verify DBN/Parquet decode identity for symbol, publisher, action/side, price/size, sequence,
   `ts_event`, `ts_recv`, and top-book fields;
2. reconstruct only strict prior-touch prints and a deterministic post-trade BBO response stream;
3. publish retained/ambiguous/corrected/locked/crossed/stale counts by session, without a future
   return or entry/exit value;
4. refuse sweep semantics if package-like legs cannot be conservatively neutralized; and
5. write `SEMANTICS_PASS_ONLY` or one of the semantic STOP dispositions above.

Because these slices were selected, a pass establishes parser possibility only. Event prevalence on
an unbiased full band remains **UNKNOWN**.

## 5. Why `ohlcv-1s` is vetoed despite zero sampled price

The cheap schema fails the mechanism's outcome-blind known answer:

> Can the record distinguish a buyer-initiated trade from a seller-initiated trade using a quote
> already received at that instant, and then observe the book's causal response?

For `ohlcv-1s`, the answer is **no by construction**.

- **VERIFIED:** it contains no individual print ordering, causally prior BBO, touch location,
  aggressor, quote revision, size distribution, or executable bid/ask.
- **VERIFIED:** the prior selective `ohlcv-1m` policy learned a last-print parity residual with
  $102.60 standard deviation against roughly $23 friction; its apparent edge disappeared on fair or
  quote pricing. Finer last-trade bars do not remove bid/ask bounce.
- **VERIFIED:** W2-H02 already combined option volume, price momentum, displayed imbalance,
  microprice and spread compression and returned `NO_SIGNAL`.
- **VERIFIED:** the magnitude/activity family found substantially larger moves and dearer options in
  busy states but only about $0.30 P&L change: activity was fully priced.
- **INFERRED:** one-second OHLCV gives a learner more opportunities to select stale/low prints while
  destroying the within-second order needed to identify a sweep.
- **VERIFIED:** it carries no quote, so it cannot replace the stopped CBBO outcome/fill substrate.
  Any ask-in/bid-out label would still depend on the `NOT-USABLE` quote corpus or a new quote source.
- **VERIFIED:** pre-2023 OPRA history is subsampled rather than equivalent packet-capture event data;
  `ts_recv` is not a real arrival clock there. Pooling it with the later era would recreate a
  source-era confound. Restricting to the event-quality era gives roughly the same 813-session
  ceiling without recovering the missing semantics.

`ohlcv-1s` may be used later as a zero-cost pipeline/coverage control if separately authorized. It
may not be called actual signed order flow, a quote-corpus repair, or the next alpha family. This
ruling applies the exact reopening terms in
[`DO_NOT_RETEST.md`](../history/DO_NOT_RETEST.md), especially W2-H02, the minute last-print selective
policy, the quoted-price closure, and the magnitude family.

## 6. Train/live parity and the missing runtime

### What exists

- **VERIFIED:** [`training_twin.py`](../training_twin.py) has current-session definition handling,
  stable raw-OSI identity, a generic captured-frame comparator, latency receipts, and historical
  arrival injection. Its operative feature/entry clock is minute-specific (`cbbo-1m` feature state
  and `cbbo-1s` entry quote). It does not certify CMBP event features.
- **VERIFIED:** one prior 914-row same-session `cbbo-1m` comparison proved value identity only. The
  47.7-million-row historical corpus reports zero arrival lag, while the live capture's median local
  lag was about 227 ms. See
  [`HISTORICAL_ARRIVAL_PARITY_2026_08_05.md`](HISTORICAL_ARRIVAL_PARITY_2026_08_05.md).
- **VERIFIED:** the existing CBBO latency/freshness receipts and roughly 926 ms p99 observation do
  not transfer to CMBP, prints, TCBBO, an IBKR order path, or the feature's response interval.
- **VERIFIED:** no v5 production runtime is present. The legacy v4 paper machinery is frozen and is
  not a v5 dependency.
- **VERIFIED:** [`causal_day_simulator.py`](../../ops/causal_day_simulator.py) carries serial starting
  equity, but still encodes a 5% breaker, accepts trade caps 1/2/3, and has no mandatory percentage
  stop. Current signed law is 20%, at most two tickets, and a declared enforced stop no tighter than
  -40%. It is not law-ready.

### Hard-gate architecture

The least-divergent architecture is **Databento live CMBP for features plus IBKR for execution**.
Training and live features must use the same raw DBN schema and one causal builder. This is a
different live architecture from using IBKR market data alone and requires new owner authority,
Databento live entitlement/licensing, and exact account pricing; all are **UNKNOWN**.

Before any fit, it requires:

1. current-session Databento definitions, raw OSI identity, live `SymbolMappingMsg` handling, and an
   exact OSI-to-IBKR `conId`/`localSymbol` map with zero partial resolution;
2. raw retention of event, receive, vendor-output and local monotonic clocks, sequence, flags,
   publisher, corrections, mappings, decision time, IBKR submission, acknowledgement and fill;
3. one event decoder and feature builder for historical replay and live streaming, with decisions
   made only from locally received bytes;
4. multi-session same-schema latency, freshness, gap and liveness receipts; any Databento
   `SkippedRecordsAfterSlowReading` is fail-closed, not imputed;
5. no-order live-first capture followed by delayed Historical API comparison of raw values, order,
   corrections, definitions, added strikes and feature identity;
6. a parallel IBKR market-data shadow. If IBKR is to supply the feature stream, empirical event and
   feature equivalence must pass. Otherwise the architecture stops;
7. execution replay at the first attainable ask/bid after the certified arrival and order latency,
   including marketable-limit behavior, partial fills, rejects, cancels, gaps and slippage; and
8. a law-current risk engine: one contract, premium plus fees at most $2,000, OTM/deep-ITM bars, one
   open position, at most two tickets/session, 20% session-start-equity breaker, simulator-enforced
   -40%-or-wider declared stop with next-bid gap/slip, serial compounding, and 50% survival floor.

Only after all eight pass may a separately authorized no-order shadow be considered; paper comes
after a surviving declared policy, and live comes after paper. Neither is authorized here.

IBKR's current official educational material describes real-time market data as sampled or
intra-second aggregated rather than an exchange-identical tick stream, and says real-time bars need
not match historical bars: [general API FAQ](https://ibkrcampus.com/docs/third-party-integrations/general-third-party-frequently-asked-questions),
[market-data overview](https://ibkrcampus.com/campus/ibkr-quant-news/ibkr-market-data-from-real-time-bars-to-ticks/).
The legacy official API page says real-time tick-by-tick option data was unavailable; because that
page is legacy, present capability is **UNKNOWN**, not assumed impossible:
[tick-by-tick documentation](https://interactivebrokers.github.io/tws-api/tick_data.html). What is
decisive today is that event-level equivalence has not been measured.

Databento's official [live guide](https://databento.com/docs/getting-started/live) and
[live API documentation](https://databento.com/docs/api-reference-live) establish that a live key,
subscription/license entitlement, schema availability, mapping handling and stream operations are
separate from historical access. This account's exact live eligibility and every monetary price are
**UNKNOWN**.

## 7. Killed or subordinated directions

### Sweep clusters

**VETO standalone.** Same-right multi-strike urgent flow could in theory select rare tails large
enough to clear $23, but consolidated L1 has no package/order identity. A bullish-looking call sweep
can be a vertical, overwrite, roll, volatility trade, or hedge. Retain only a conservatively netted
diagnostic within rank 1; kill it if same-timestamp opposing-leg neutralization reverses sign.

### Quote revision/intensity

**VETO.** Update counts, publisher breadth, quote lifetime, price revisions and size changes without
prints are another W2-H02/activity transform. A genuinely new primitive must be impossible to
reconstruct from `cbbo-1m` and conditioned on a real strict-touch trade. Once conditioned, it is
rank 1, not a separate family.

### Spread compression

**ADJUNCT ONLY.** Removing the typical entry half-spread saves roughly $10 if exit cost is unchanged;
that is below the $23 direct round trip. Prior midpoint patience saved about $10 but suffered roughly
$60-$86 adverse selection. A live compression must persist beyond certified p99 order-path latency
with ask size at least one; otherwise it is not executable. It may route an independently validated
entry signal, never become the signal.

### Intra-minute SPX path

**BLOCKED AS A PREDICTOR; retained only as a reaction control.** Raw momentum, chart, clock and shared
price-level constructions are closed. The only new question is whether strict option flow leads SPX
by more than the complete live/order path. The exact data source, cost, causal clock and latency are
unknown. A nonpositive latency margin kills it without outcomes.

### Recent free event era

**PARSER/LIVENESS ONLY.** The 227-session recent era may make `trades`, TCBBO or CMBP cost zero under
the observed vendor regime, but those bytes are not broadly owned, require new acquisition authority,
are confined to one recent market/source regime, and provide no independent training/validation/
confirmation allocation. Event density does not turn 227 sessions into thousands.

### New exit timing, more tickets, or more days

**VETO.** The signed Tier-A exit bar needs about 2,560 outer sessions—over a decade before separate
entry and exit roles. Two tickets remain within one session cluster. Dates after 2026-08-06 remain
confirmation-only and were not touched. No historical vendor query can create future independent
outer holdout.

## 8. Binding unknowns

| Unknown | Why it matters | What would resolve it |
|---|---|---|
| Exact all-session CMBP/TCBBO/trades cost | Sample means and extrapolations are not acquisition quotes | The per-session raw-symbol metadata-only preflight in §4, after a new owner decision |
| Exact all-session `ohlcv-1s` nonempty/$0 status | Five cost and two count probes are not a census | Same preflight; it would not reverse the alpha veto |
| Strict-touch and response prevalence on an unbiased band | Selected owned slices cannot estimate population prevalence | Broad unbiased event data, not currently authorized |
| Package/complex-leg identifiability | Apparent directional flow may have the wrong sign | Outcome-blind schema semantics; if unavailable, sweep component stops |
| Entry target, minimum worthwhile effect and session power | Without them, “enough data” has no mathematical meaning | Owner-frozen policy/effect followed by a no-outcome known-answer campaign |
| Event response interval | It must exceed causal observation and live latency without being outcome-tuned | Same-schema live latency and broker-order receipts |
| Sub-minute SPX source and parity | Needed to separate option lead from reaction | Exact source contract and historical/live twin; no ES/VIX substitute |
| Databento live OPRA license, eligible schemas and price | Same-vendor live feed is the least-divergent route | Account-specific preflight under fresh authority |
| Current IBKR option event capability and exact feature equivalence | IBKR execution may see a different market-data clock/semantics | No-order paired capture; documentation alone cannot certify equality |
| Economic edge | No outcome experiment was run | Only a future declared, powered, session-clustered experiment after all prior gates |

## 9. Final owner choice

**Recommended:** accept `STOP_UNDER_CURRENT_CONSTRAINTS`; acquire and fit nothing. The available free
schema is the wrong observable, the correct observable is not available across the full era and is
planning-priced above the current charter, the unbiased sample and entry power are unresolved, and
the live/runtime chain is missing.

**If the owner explicitly declines the stop:** authorize only the outcome-blind owned-slice semantic
gate in §4D. Its terminal answer is binary:

- semantics fail: `STOP_MICROSTRUCTURE_FAMILY`;
- semantics pass: `SEMANTICS_PASS_ONLY`, while acquisition, fit and deployment remain stopped until
  exact cost, full coverage, live parity, a law-current runtime, a signed entry effect and
  session-level known-answer power all exist.

There is a credible **question**—whether strict touch flow plus non-replenishment is causally
measurable. There is no credible evidenced **bot path** under today's constraints. If the question
cannot be answered from the owned CMBP slices or the owner will not change the data/live constraints,
the honest endpoint is stop, not a finer last-trade bar.

## Verification scope

This synthesis used independent data-cartography, microstructure, measurability, runtime/parity and
adversarial-veto reviews. It read repository evidence and retained metadata receipts only. The only
new counts were outcome-blind filesystem/session-manifest arithmetic: 1,014 total ladder sessions,
813 on or after 2023-03-28, 586 between that coverage start and the observed recent-free era, and 227
from 2025-08-25 onward. No outcome-bearing table was opened.

No new wrapper or receipt was created because no preflight, acquisition, parser, fit or runtime action
was executed. The existing depth wrappers and receipts remain archived under
`v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/`.

Verification after the finding and log entry were present:

- `./.venv/bin/python v5/ops/check_project.py`: **PASS**;
- `./.venv/bin/python -m pytest v5/tests -q`: **1,138 passed**, 105 existing numerical warnings,
  exit 0 in 68.72 seconds.
