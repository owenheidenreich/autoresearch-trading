# DATA_CONTRACT.md

The data contract is the foundation of v4. Every row in every layer must be able to answer:

> *What was known, when it was known, where it came from, how it was transformed, and whether it could have been used live?*

This is not optional. It is the first artifact built, before any feature or model. This document is the source of truth for:
- Layer architecture (raw / normalized / feature / label / audit)
- Per-row required fields
- Mutability and recomputation rules
- Leak-detection invariants
- Vendor-specific notes (OptionsDX, Databento, IBKR, OptionsDepth)

Schema implementations in [../schema/](../schema/) must conform to this document. CI tests in [../tests/](../tests/) verify conformance.

---

## 1. Layer architecture

Five layers. **No feature crosses a layer without an explicit transformation.**

| Layer | Contents | Mutability | Stored as |
|---|---|---|---|
| **Raw** | Vendor-original messages, unchanged: Databento OPRA, OptionsDX CSVs, IBKR live captures | Immutable, versioned, hash-stamped | Parquet (compressed source-faithful copy) + raw original |
| **Normalized** | Unified contract symbology, timestamps, NBBO, trades, definitions, statistics/OI | Recomputable from Raw, byte-deterministic | Parquet, partitioned `(date, root, expiry_offset)` |
| **Feature** | Causal features available as of `decision_time` only | Recomputable; fails leak-detection CI on every commit | Parquet, partitioned `(date)` |
| **Label** | Future outcomes; never joined into features except as explicit training targets | Computed once per oracle definition; namespaced | Parquet, separate from features |
| **Audit** | Hashes, source versions, vendor licenses, build IDs, schema versions, leakage tests | Append-only | JSONL + markdown reports |

**Hard rules**:
1. Raw is never modified after first write. Hash-stamped on ingest; hash mismatch on re-read is a critical error.
2. Normalized must be a deterministic function of Raw. Re-running ingest produces byte-identical output.
3. Feature rows must NOT be joinable to Label rows except through the training-time pipeline that explicitly carries labels as targets.
4. Audit is append-only. Failed runs stay in the audit log forever — they are evidence that the discipline was followed.

---

## 2. Per-row required fields

### 2.1 Time fields (every layer)

```
timestamp_source         # NTP source, vendor wall-clock identifier
timestamp_precision      # 'ms' | 'us' | 'ns'
event_time               # when the market event occurred (vendor-reported)
receive_time             # when our system received the message
decision_time            # when the model would commit to a decision (typically bar_close + N ms)
```

**`decision_time` is the most important field in this contract.** Everything in the Feature layer must be computable from data with `event_time <= decision_time`. Any feature whose computation peeks at `event_time > decision_time` is a leak by definition.

### 2.2 Contract identity (Normalized, Feature, Label)

```
contract_id              # canonical (root, expiry, strike, right) — see ../parser/
expiry                   # ISO date
strike                   # decimal
right                    # 'C' | 'P'
root                     # 'SPX' | 'SPXW' | 'SPY' | ...
```

### 2.3 Market data (Normalized, Feature)

```
bid                      # best bid, decimal
ask                      # best ask, decimal
mid                      # (bid + ask) / 2; computed, not vendor-supplied
last_trade               # last trade price
last_trade_size
quote_age_ms             # how stale was the NBBO at decision_time
open_interest            # last EOD value
open_interest_asof_date  # when that EOD OI was published
volume                   # cumulative session volume at event_time
volume_asof_time         # the event_time this volume was computed at
```

### 2.4 Greeks / IV (Normalized only after computation; Feature consumes)

```
iv                       # implied volatility, decimal annualized
iv_source                # 'optionsdx' | 'black_scholes_v4' | 'ibkr_model'
delta, gamma, theta, vega, rho, charm, vanna, vomma
greek_source             # same enum as iv_source
greek_computation_ts     # when Greeks were computed (may differ from event_time)
risk_free_rate_used      # rate input to BS
dividend_yield_used      # SPX is index, but document the assumption
```

**Rule**: if `iv_source == 'optionsdx'`, the row was loaded from a vendor-supplied IV. If `iv_source == 'black_scholes_v4'`, it was computed by [../greeks/](../greeks/). Mixing in a single dataset is permitted only with `iv_source` per row.

### 2.5 Provenance (Normalized, Feature, Label)

```
vendor_source            # 'OPTIONSDX' | 'DATABENTO_OPRA' | 'IBKR_LIVE' | 'IBKR_PAPER' | 'OPTIONSDEPTH'
ingest_run_id            # links back to audit log entry
schema_version           # this contract's version, e.g. 'v1.0.0'
```

### 2.6 Causality (Feature layer — CRITICAL)

```
is_live_reproducible     # bool — could this row exist live at decision_time?
is_label                 # bool — is this a future outcome, not a feature?
feature_age_seconds      # how old is this feature snapshot at decision_time
feature_source           # 'computed_from_normalized_v4' | 'optionsdepth_10min' | etc.
feature_refresh_interval # nominal refresh cadence in seconds
feature_is_estimated     # bool — e.g., intraday OI estimates vs EOD truth
feature_is_revised       # bool — e.g., revised after-the-fact
```

**`is_live_reproducible` is the leak-prevention contract.** If a feature cannot exist live at decision_time, it does not belong in the Feature layer. CI test in [../leakage/](../leakage/) shuffles every column with `is_label=True` and retrains; performance must be statistically unchanged.

### 2.7 Audit fields (Audit layer)

```
ingest_run_id            # uuid
build_id                 # git sha + uncommitted-diff hash
schema_versions          # dict of layer -> version
vendor_license           # license under which this data was acquired
vendor_terms_hash        # hash of the written T&Cs we agreed to
ingest_started_at, ingest_finished_at
input_files              # list of (path, hash, size_bytes)
output_files             # list of (path, hash, row_count)
leakage_test_results     # list of (test_name, passed, metric_value)
deterministic_rebuild_hash  # hash of normalized output; CI compares to prior
```

---

## 3. Feature freshness is first-class

A 10-minute OptionsDepth dealer-flow snapshot carried into a 1-minute decision loop may still be useful, but **the model must know whether the snapshot is 30 seconds old or 9 minutes old**.

**Rule: do not silently forward-fill premium features.** Every forward-filled feature must carry `feature_age_seconds`, and the model must consume that field. CI test in [../tests/](../tests/) verifies that no feature is forward-filled without an age field within tolerance of its `feature_refresh_interval`.

---

## 4. Vendor-specific notes

### 4.1 OptionsDX (free, 2010–2023)

- Coverage: SPX/SPY option chains 2010–2023, minutely. **Does NOT cover 2024+, so insufficient for modern-regime validation.**
- Provides bid, ask, last, IV, delta/gamma/theta/vega, underlying. **Does not provide intraday OI** (vendor publishes EOD chains daily; intraday OI is unavailable).
- Used in v4 Phase 0 for: pipeline scaffolding, simulator skeleton, Greeks reconciliation against our Black-Scholes computation.
- Used in v4 NOT for: final edge proof, recent-regime validation, live deployment assumptions.
- Schema: CSV per day. Column names normalized in [../ingest/](../ingest/).

### 4.2 Databento OPRA (paid, Phase 1 only)

- Coverage: full OPRA history including SPXW, 2018+ available, 0DTE coverage clean.
- Provides: trades, NBBO/CBBO, definitions, statistics (including OI). **Does NOT provide pre-calculated IV or Greeks.**
- Greeks for Databento data are computed by [../greeks/](../greeks/) using Black-Scholes from underlying + risk-free + dividend assumptions.
- Cost: usage-based PAYG, billed by uncompressed GB. Use Databento's pricing estimator before committing.
- Schema: documented at https://databento.com/docs/venues-and-datasets/opra-pillar — capture the schema version per ingest.

### 4.3 IBKR (live, Phase 0.5 preflight + Phase 3+ deployment)

- Live OPRA NBBO + Greeks via TWS API per-contract subscriptions; **no chain-streaming mechanism**. Must subscribe per-contract via `tickOptionComputation`.
- Initial 100 concurrent market-data lines. Two-stage scanner architecture required for live deployment.
- Greeks come from IBKR's model when option + underlying are both subscribed.
- Live data subs cost ~$10–15/mo all-in for SPX 0DTE coverage.

### 4.4 OptionsDepth (Phase 1 one-cycle / Phase 3+ recurring)

- **Not usable in v4 unless written terms confirm** (see protocol Section 4.2):
  1. Historical intraday snapshots are exportable in machine-readable form
  2. Exported data may be retained locally after cancellation
  3. Exported data may be used to train private models
  4. Fields are point-in-time, not retrospectively revised
  5. Each snapshot has a precise timestamp and known refresh cadence
  6. Field definitions are stable and documented
  7. SPX and VIX coverage are both included at the subscribed tier
- Without those seven items, OptionsDepth is a manual dashboard input only — not a Feature-layer source.
- Refresh cadence: 10-minute intraday at Pro Max tier. Schema must be captured before any data is ingested.

---

## 5. Leak-detection invariants

These invariants are enforced by CI in [../leakage/](../leakage/) on every commit:

1. **Shuffled-label invariance**: shuffle every column with `is_label=True` along the time axis; retrain a small probe model; performance must be statistically unchanged (within bootstrap noise).
2. **`is_live_reproducible` audit**: every feature in a training set must have `is_live_reproducible=True` for every row. Failure halts CI.
3. **Time-shift sanity**: shifting features by +1 bar (i.e., using future) should produce a measurable performance lift; shifting by -1 bar (using stale features) should produce a measurable degradation. Both probes must fire as expected.
4. **Planted-leak test**: a deliberately-planted leak (a future-only column added at probe time) must produce suspiciously high performance, proving the detector works.
5. **Greek-source consistency**: every Feature row's `iv_source` and `greek_source` must be one of the documented enums; rows with ambiguous sources fail CI.

---

## 6. Schema versioning

Each row carries `schema_version` (semver). Schema changes:

- **Patch (x.y.Z)**: additive fields with safe defaults; no recompute required.
- **Minor (x.Y.0)**: additive required fields; backfill required; old data tagged with old version.
- **Major (X.0.0)**: breaking change; full deterministic rebuild required; old data archived.

CI compares the deterministic-rebuild hash of normalized output against the previous run's hash. Any mismatch without a documented schema bump is a critical error.

---

## 7. What this contract does NOT cover (yet)

- Live data wire formats (deferred to Phase 0.5 IBKR preflight).
- Streaming compaction strategy (deferred to Phase 1).
- Cross-vendor reconciliation tolerance (deferred to Phase 1, where two vendors first overlap).
- Promotion-packet schema (separate document in [../promotion/](../promotion/)).

These are deliberate Phase-0 exclusions — defer until needed, and document the deferment.

---

## 8. References

- Protocol Section 4.0 (source of this contract): `/Users/gduby/.claude/plans/ok-well-this-just-declarative-puppy.md`
- Anti-pattern #11 in protocol Section 7: forward-PnL labels as inputs (banned by leak-detection CI; this contract is how the ban is enforced)
- Anti-pattern #21 in protocol Section 7: reproducing v3 priors as a Phase-0 gate (replaced with the deterministic-pipeline gates encoded here)
