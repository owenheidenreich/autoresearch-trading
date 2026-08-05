# V5 workflow capability audit — 2026-08-05

**Meaning for the bot:** the data and much of the safe research machinery exist, but no causal market
edge has been shown. The independent measurability review remains the next job; model training would be
premature.

This is a dated capability finding, not a second status page. Current state and authorization always come
from [`v5/STATUS.md`](../../STATUS.md).

## 1. What is being built

The target is an automated day-trading bot that buys SPX zero-days-to-expiration calls or puts. Its entry
must be based on causal evidence about SPX/ES direction over the next 15–60 minutes, clear measured costs,
survive historical validation, reproduce its decisions in no-order live shadow, and only then enter
owner-authorized guarded paper testing. Entry selection and exit management are separate model stages.

## 2. Existing pieces that directly serve the v5 workflow

| Capability | What exists now | V5 treatment |
|---|---|---|
| Owned training corpus | 254 usable ES sessions and 251 option sessions within the 2025-08-01 to 2026-07-31 Path-D corpus | Protected external data, indexed rather than copied |
| Path-D boundary | Source-neutral contracts, offline replay, decision/risk separation, fake gateway, and broker-free internals under [`v4/path_d/`](../../../v4/path_d/) | Vetted legacy foundation; promote only the minimum reviewed behavior |
| Bounded research machinery | Typed hypotheses, chronological out-of-fold evaluation, prediction caches, paired component screens, multiplicity control, serial replay, and terminal result states under [`v4/research/autoresearch_v2/`](../../../v4/research/autoresearch_v2/) | Vetted legacy reference; it does not authorize a fit |
| Historical/live clock law | Exact completed-minute selection, current-session definitions, raw OSI identity, receipt-time cutoffs, and fresh entry quotes | Promoted to native [`training_twin.py`](../training_twin.py) with no `v4` import |
| Feature firewall | A content-addressed ledger containing 83 features; 8 are admitted and 75 barred. The only admitted family is the eight-field contract clock | Historical ledger remains evidence; native [`feature_admission.py`](../feature_admission.py) verifies it but refuses to authorize a new fit because its receipts have no current validity window |
| Track-A capture | Definition, OPRA arrival, missingness, warm-up, and latency capture tooling exists | Frozen in `v4`; prior dates are canceled, all jobs are unloaded, and a new run requires fresh owner authorization |
| Candidate validation views | The legacy exporter proved the trade-CSV, SPX-marker chart, and equity-curve presentation | Replaced for future candidates by the source-neutral [`candidate_packet.py`](../validation/candidate_packet.py); CSV and manifest are authoritative, charts are diagnostic |
| IBKR paper execution | A guarded paper round trip and account safety refusals prove the plumbing | Frozen legacy capability; not part of current research and not permission to contact IBKR |

The new native modules are deliberately inert. They accept caller-supplied local rows, ledgers, and trade
records. They do not download data, fit a model, tune a threshold, connect to a broker, submit an order,
or change runtime state.

## 3. What the evidence proves

- The acquisition manifest identifies Databento OPRA definition, CBBO-1m, CBBO-1s, OHLCV-1m and
  statistics; ES/VX minute context; and ThetaData SPX/VIX minutes. The quality report records 251 option
  sessions, 47,707,186 normalized base rows, and 90,073 aligned minute-entry rows.
- In one paired session, all 914 live CBBO-1m rows matched their Historical API counterparts byte-for-byte
  after decoding. Columns and data types also matched.
- The same session had 510 current definitions versus 492 prior-session known contracts. All 492 prior
  instrument IDs changed, so raw OSI identity plus a current-session mapping is binding.
- The short capture observed CBBO-1m gateway p99 of 54.493 ms and local-receipt p99 of 319.521 ms. These are
  one-session observations, not a safe multi-session bound.
- The first partial subscription interval is not a completed minute and must be discarded. Reconnects
  require the same warm-up behavior.
- Five ThetaData timing samples produced a frozen 2,336 ms shared emission allowance. The arithmetic is
  correct, but five dispersed samples are too thin to authorize fitting; a future Track-A capture must
  re-derive the allowance without lowering it unless a rule is preregistered first.
- Local-only tests establish the source-neutral clock selectors, definition updates, paired replay,
  fail-closed admission, deterministic validation outputs, and legacy Path-D contracts. Tests prove code
  behavior, not market edge.

## 4. What was disproven

- The prior `signed18` result was invalid: all 445,063 fitted rows consumed the ThetaData SPX close stamped
  at the current minute before that minute had completed. Only 18.44% of its decisions survived the lawful
  clock reconstruction.
- Unchanged long 0DTE premium at minute cadence lost about $13 per trade before costs in every fold.
- The range-position signal lost 0.324 ES points per trade gross; the option risk-reversal failed its fold
  gate; the prior 18-feature entry ranker was flat; and all six exit-repair arms lost to their comparators.
- The legacy Protocol101 historical signal rate was not reproduced in live sessions. It is not a
  historical/live-equivalent training path.

Binding results and reopening conditions remain in [`DO_NOT_RETEST.md`](../history/DO_NOT_RETEST.md).

## 5. What remains unknown or blocked

- No causal policy has shown an economic edge over 0.358 ES points per completed round trip.
- With 254 sessions, the optimistic 80%-power minimum detectable effect is 0.413 points even at the
  theoretical maximum of 26 independent trades per session. Real within-session dependence makes that
  floor optimistic.
- Multi-session OPRA arrival bounds, full proposed-feature parity, complete account-state parity, fill
  realism, and a clean confirmation path remain unproven.
- The existing G5 validation replay has four defects. The candidate packet makes results inspectable but
  does not repair those statistical defects.

Therefore the next authorized action is the independent measurement review. A favorable review releases
only the preregistered M1/M3 ES-direction screen. A 60-minute G1 pass permits one fixed option-dollar
replay; it does not permit hillclimbing or general model training.

## 6. Training order after the gates release it

> **Superseded as a sequence.** The route of record is
> [STATUS.md §2](../../STATUS.md#2-the-route-to-a-trading-bot). The list below remains the dated reasoning
> that produced it.

1. Resolve measurement feasibility.
2. Run the fixed G1 ES-direction screen once if released.
3. Run one locked option-dollar replay only after an exact 60-minute G1 pass.
4. With fresh owner authorization, collect multi-session Track-A evidence and certify every proposed
   feature through the shared historical/live adapter.
5. Freeze the entry contract, then run a bounded shallow-model search using admitted features and
   chronological out-of-fold predictions. Keep confirmation evidence sealed.
6. Freeze the entry trade stream before any exit-model search.
7. Validate with the authoritative trade CSV and manifest plus the SPX trade chart and equity curve.
8. Reproduce features, scores, and decisions in no-order live shadow before guarded paper testing.

This sequence corrects the earlier three-stage description by adding the measurement and parity work that
must precede training and the no-order shadow test that must precede paper orders.

## 7. New local interfaces

- `v5.research.training_twin.make_clock`, `compile_entry_source_receipt`,
  `compare_paired_frames`, and `write_parity_receipt`
- `v5.research.feature_admission.verify_ledger`, `assert_features_admitted`, and
  `admitted_feature_matrix`
- `v5.research.validation.export_candidate_packet`, plus
  `python -m v5.research.validation.candidate_packet` for explicit local files

These interfaces are foundations, not gate passes. In particular, `assert_features_admitted` requires an
unexpired validity window by default, so the historical Path-D ledger cannot silently become permission
for a new fit.

## 8. Evidence used

- [Acquisition manifest](../../../v4/audit/autoresearch/protocol101_pathd_data_acquisition/acquisition_manifest.json)
  and [data-quality report](../../../v4/audit/autoresearch/protocol101_pathd_data_acquisition/data_quality_report.json)
- [Same-session OPRA comparison](../../../v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/comparison_same_session_attempt002/comparison_result.json)
- [ThetaData emission allowance](../../../v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/shared_emission_lag.json)
  and [thin-sample correction](../../../v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/emission_lag_contradiction_resolution_2026_08_04.json)
- [Feature-admission ledger](../../../v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json)
- [Unattended Python receipt](../../../v4/audit/autoresearch/python_direct_unattended_execution_2026_08_05_attempt001/launchd_probe_receipt.json)
- [Gate-chain and measurement audit](GATE_CHAIN_AUDIT_2026_08_05.md)
