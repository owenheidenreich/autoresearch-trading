# Section 1: Market Data

## Scope
Raw data acquisition, feature engineering, normalization, label generation, dataset construction, and the final `data.pt` artifact that feeds training and replay.

## Critical Files

| File | Role | Lines | Mutable? |
|------|------|------:|----------|
| `v2/pipeline/download_wide_grid.py` | Downloads SPXW 0DTE option data from Polygon S3 flat files. Extracts 82-contract wide grid (ATM +/- 100pt) with full OHLCV per bar. | 320 | No (infra) |
| `v2/pipeline/extract_raw.py` | Extracts SPX/SPY/VIX minute-bar data from raw Polygon cache. Recovers OHLCV, bid-ask proxy, real-time SPX via call-put parity. | 323 | No (infra) |
| `v2/pipeline/compute_features.py` | Computes all 47 features in three groups: Price/Volume/Market (28), Option/Greeks (11), Volume/Flow (8). Includes Black-Scholes implementation. | 818 | No (infra) |
| `v2/pipeline/build_v2_dataset.py` | Ground-up dataset build from raw caches. Loads SPX/SPY/VIX pickles + spxw_wide option data. Computes triple-barrier labels, applies rolling z-score normalization. Produces `data.pt` with 4-way date-based split. | 663 | No (infra) |
| `v2/pipeline/build_dataset.py` | Legacy dataset builder (v1 reference). Loads v1 data.pt features and option prices, computes oracle labels. | 378 | No (legacy) |
| `v2/pipeline/relabel_tier3.py` | Relabels existing `data.pt` with Tier 3 grid search for variable risk parameters (stop/target/hold). | 283 | No (infra) |
| `v2/core/features.py` | Feature constants (47 names), indices, rolling z-score normalization (expanding window), exclusion lists for bounded/categorical features, adaptive spread computation. | 188 | No (harness) |
| `v2/core/labels.py` | Oracle labeler. Triple-barrier method over candidate trades x risk parameter grids. Three tiers: Tier 1 (fixed), Tier 2 (medium), Tier 3 (full grid search). | 350 | No (harness) |
| `v2/core/candidates.py` | Dynamic candidate generation from option chain data. ATM rounding to 5pt grid, filtering by min_bid, max_spread_bps, min_price. | 181 | No (harness) |
| `v2/data.pt` | Active dataset artifact. Contains features, labels, masks (train/val/promote/shadow), option prices. | binary | Output |

## Data Flow

```
Polygon S3 flat files
    |
    v
download_wide_grid.py --> spxw_wide/ (per-day option chain cache)
    |
extract_raw.py --> SPX/SPY/VIX pickle caches
    |
    v
compute_features.py
    |  reads: SPX/SPY/VIX pickles + spxw_wide
    |  produces: 47 features per bar (no normalization here)
    |
    v
build_v2_dataset.py
    |  reads: raw caches + compute_features output
    |  applies: rolling z-score normalization (expanding window)
    |  applies: triple-barrier labels via core/labels.py
    |  splits: train / val / promote / shadow (date-based)
    |  writes: data.pt
    |
    v
relabel_tier3.py (optional)
    |  reads: existing data.pt + raw option prices
    |  rewrites: labels with Tier 3 variable-risk grid search
    |  writes: updated data.pt
    |
    v
data.pt --> consumed by train.py, replay.py, analysis tools
```

## Key Interfaces

**Inputs to this section:**
- Polygon S3 bucket (raw SPXW option data, minute-bar market data)
- Date range configuration in build scripts

**Outputs from this section:**
- `v2/data.pt` -- the single artifact consumed by all downstream sections
- Contains: feature tensors, label tensors, option price arrays, date arrays, split masks

**Shared contracts:**
- `core/features.py` defines the 47-feature schema used by training, replay, and live
- `core/labels.py` defines oracle label format consumed by the loss function in `train.py`
- `core/candidates.py` defines candidate generation used by both labels and live decision

## Dependencies on Other Sections

| Section | Dependency |
|---------|------------|
| Training Runs | Consumes `data.pt` for model training |
| Validation | Consumes `data.pt` for replay simulation and baseline computation |
| IBKR Paper Trading | `core/features.py` defines feature schema used by live feature engine |

## Audit Surface Area

- Feature computation pipeline: are all 47 features computed correctly from raw data?
- Normalization: rolling z-score uses expanding window -- is there look-ahead bias?
- Label generation: oracle labeler searches future data by design (it's the training target) -- is the search grid appropriate?
- Data split: are train/val/promote/shadow splits date-based with no leakage?
- Spread cost model: adaptive BPS + min-tick floor -- does it match real-world execution?
- Raw data completeness: 999 trading days, any gaps or holidays mishandled?
- Feature exclusions: which features skip normalization and why?

---

## Audit Questions -- Direct Improvements

**1. Feature schema doc is stale and contradicts the actual code (39 vs 47 features).**
`v2/docs/feature_schema.md` describes 39 features with names like `iv_skew`, `prev_close_dist`, `overnight_gap` -- all removed in the rebuild. The actual feature list in `v2/core/features.py:43-60` and `v2/pipeline/compute_features.py:795-818` has 47 features with different names (`iv_skew_pct`, `current_moneyness_pct`, `theta_acceleration`, etc.). The normalization contract section also describes "per-day z-score with global standard deviation" while the code uses rolling z-score. 15-minute fix.

**2. `data_contract.md` describes dataset shape `(N_bars, 39)` and oracle label fields that do not exist in the actual `data.pt`.**
The data contract says `X` is `(N, 39)` and includes fields like `oracle_intents`, `oracle_pnl`, `oracle_trade`, `y_return_15`. The actual build pipeline (`build_v2_dataset.py:590-641`) produces `(N, 47)` with `label_call_pnl`, `label_put_pnl`, `label_trade`, `label_direction` -- completely different keys. This doc is actively misleading.

**3. `relabel_tier3.py` Tier 3 grid is missing `holds=[390]` that `core/labels.py` includes.**
`core/labels.py:42` defines `TIER3_MAX_HOLDS = [30, 60, 120, 240, 390]` (5 values). `relabel_tier3.py:39` defines `holds = [30, 60, 120, 240]` (4 values, missing 390). Since `relabel_tier3.py` is the script that produced the current `data.pt` labels, the labels never considered holding to EOD. For a 0DTE bot, holding to expiry is a valid risk parameter and its absence biases labels toward shorter holds.

**4. POC and Value Area features use full-day volume profile, not running profile -- look-ahead bias.**
`compute_features.py:234-274` computes POC and Value Area once per day using all bars `c[ds:de]` and `vol[ds:de]`, then assigns those values to every bar in the day. At bar 30, the model sees a POC computed from bars 0-389 (including 360 future bars). For intraday trading decisions, a running POC that updates bar-by-bar would be correct.

**5. `volume_ratio` lookback window disagrees between docs (20-bar) and code (60-bar).**
`feature_schema.md` says "bar volume / 20-bar SMA volume"; the code at `compute_features.py:417` uses `LOOKBACK=60`. Minor discrepancy but matters for live feature parity.

**6. Spread cost metadata records `cost_model: {spread_rt: 0.30}` but the code uses adaptive BPS model.**
`build_v2_dataset.py:635` metadata says flat $0.30 round-trip, but the code (`build_v2_dataset.py:483-491`) uses adaptive BPS varying by time-of-day, VIX regime, and OTM status. Anyone inspecting dataset metadata gets wrong cost assumptions.

## Audit Questions -- Deeper Planning

**7. Label generation uses only ATM strike, ignoring the full strike ladder.**
`build_v2_dataset.py:396-397` finds the single nearest-ATM strike, simulates only that for call and put. The labeling doc (`labeling.md:31-35`) says "ATM +/- 30 points in 5-point steps = up to 26 candidates." `core/labels.py` implements multi-strike search but is never called by `build_v2_dataset.py` or `relabel_tier3.py`. The oracle never discovers that an OTM10 put might have been the best trade at a given bar, capping label quality.

**8. Rolling z-score has a cold-start problem affecting the first ~23,400 bars (60 days).**
`core/features.py:97-126` uses expanding window for the first `window` bars (390*60 = 23,400), then switches to rolling. During the expanding phase, mean and std are dominated by early data. Also, normalization runs sequentially across all bars including validation/test, so rolling statistics on test bars incorporate training-period statistics. In live trading, the normalization must be bootstrapped from scratch each session -- how will the live feature engine replicate this 60-day history?

**9. The labeler and simulator diverge on exit-time spread cost calculation.**
In `build_v2_dataset.py:482-484`, exit spread uses the actual exit bar offset. In `relabel_tier3.py:83-84`, exit spread uses the max hold parameter, not the actual exit bar. A trade hitting stop loss at bar 5 of a 240-bar hold window gets charged spread as if it exited 240 bars later (wider spread). This systematically over-charges early exits, biasing labels toward shorter holds and tighter stops.

**10. `gamma_pressure` feature requires per-bar BS IV solve across 82 strikes -- live parity risk.**
`compute_features.py:669-685` iterates over every strike in the wide grid, calling `_bs_iv()` (Brent root-find) plus `_bs_greeks()` for each. At 82 strikes x 2 sides = 164 IV solves per bar. In live trading, if the live feature engine uses IBKR's streamed Greeks instead of running BS, the gamma_pressure values will differ, creating train/live distribution shift.

**11. Candidate generation synthetic pricing model is unrealistic for 0DTE.**
When `chain` is None, `candidates.py:92-100` uses `intrinsic + max(0.5, 3.0 - abs(offset) * 0.08)` as the option price. A 30pt OTM option gets time value of $0.60, which is near-zero in reality for 0DTE. If this fallback is hit during replay or live, candidate prices are nonsensical.

**12. Training labels use hardcoded `GATE_MIN_PNL = 0.04` (4%) with no connection to the replay gate threshold.**
`build_v2_dataset.py:509` and `relabel_tier3.py:209` hardcode `GATE_MIN_PNL = 0.04`. The model learns "trade = True when P&L > 4%." But the replay gate uses `core/policy.py` threshold. If these differ, gate predictions and label distributions are misaligned. Also means labels contain zero losing trades by construction -- the "100% winners" problem from `data_audit_findings.md` Finding 2.

**13. `iv_percentile` needs persistent IV history across live sessions.**
Training computes IV percentile with expanding history across 994 days (`build_v2_dataset.py:189-265`). Live system starts fresh each session. Without persisting IV history, the feature distribution shifts on day 1 of live trading.

## Related Documentation

- `v2/docs/data_contract.md` -- data specification and contracts
- `v2/docs/feature_schema.md` -- complete feature definitions
- `v2/docs/labeling.md` -- oracle labeling process specification
- `v2/docs/data_audit_findings.md` -- prior data audit findings
- `v2/docs/feature_rebuild_summary.md` -- feature pipeline rebuild notes
- `v2/docs/feature_audit_handoff.md` -- feature audit handoff
