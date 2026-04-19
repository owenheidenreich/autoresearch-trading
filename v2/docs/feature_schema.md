# v2 Feature Schema (live)

This doc is the authoritative audit of every feature that ships in `v2/data.pt` today.
It replaces a stale prior version that documented a 52-feature contract.

## Truth statement

- **Live count:** 79 context features.
- **Source of truth:** [ALL_FEATURE_NAMES](../pipeline/compute_features.py#L1142) in `v2/pipeline/compute_features.py`, re-exported as `FEATURE_NAMES` by [v2/core/features.py:52](../core/features.py#L52).
- **Integrity gate:** `NUM_FEATURES == len(FEATURE_NAMES) == 79`, enforced in [v2/core/data_integrity.py:141-142](../core/data_integrity.py#L141-L142) and `validate_feature_shape` ([features.py:160](../core/features.py#L160)).
- **Concat order (build-time):** `[X_price (46) , X_opt (12), X_surface (13), X_flow (8)]` — see [build_v2_dataset.py:933-935](../pipeline/build_v2_dataset.py#L933-L935). The first 46 columns are produced by `compute_price_features` and cover both the PRICE (32) and SESSION (14) named groups.
- **Separate contract surface (22 fields):** documented in `CONTRACT_FEATURE_FIELDS` ([v2/core/chain_data.py](../core/chain_data.py)). Not in scope for this audit.

## Group headline

| Group     | Indices | Count | Source function                                                                               |
|-----------|---------|------:|------------------------------------------------------------------------------------------------|
| PRICE     | 0-31    |    32 | `compute_price_features` ([compute_features.py:256](../pipeline/compute_features.py#L256))     |
| SESSION   | 32-45   |    14 | same function (session block starts at [compute_features.py:804](../pipeline/compute_features.py#L804)) |
| OPTION    | 46-57   |    12 | `compute_option_features` ([compute_features.py:895](../pipeline/compute_features.py#L895))    |
| SURFACE   | 58-70   |    13 | `_compute_surface_features_for_bar` ([build_v2_dataset.py:241](../pipeline/build_v2_dataset.py#L241)) |
| FLOW      | 71-78   |     8 | `compute_flow_features` ([compute_features.py:1042](../pipeline/compute_features.py#L1042))    |

Historical note: [features.py:54](../core/features.py#L54) still defines `LEGACY_52_FEATURE_NAMES = PRICE + OPTION + FLOW` (32 + 12 + 8 = 52). The 27 features added on top of that legacy contract are exactly the 14 SESSION + 13 SURFACE features. Several of these added features are central to the reversion thesis (see cross-reference at end), so the "52-feature" framing is not just stale — it omits thesis-load-bearing features.

## Per-feature table (79 rows)

Columns:
- **idx** — canonical order in `FEATURE_NAMES`
- **name** — canonical name
- **formula** — one-line summary of the computation
- **window** — computation scope (see legend below)
- **PIT** — point-in-time safety at bar `t` (see legend)
- **thesis** — alignment with the opening-reversion hypothesis (`rev` / `cont` / `vol` / `exec` / `neutral`)
- **collapse** — correlation-cluster letter (see "Collapse groups" below); `—` if standalone
- **source** — `compute_features.py:LINE` unless otherwise noted

**Window legend:** `bar-local` (uses only current bar), `session-accum` (accumulator from day start), `rolling-N` (needs N prior bars), `5min-k` (5-min aggregation, needs k 5-min bars = 5k 1-min bars), `first15-bcast` (max/min of bars 0-14 broadcast day-wide), `ib30-bcast` (max/min of bars 0-29 broadcast day-wide), `cross-day` (uses prior-day or multi-day state), `chain-snap` (bar-time option chain snapshot), `clock` (pure time-of-day).

**PIT legend:** `safe` (uses only information available at bar `t` close), `unsafe<X` (broadcast-fills with future bars; safe only when `bar_of_day >= X`), `cross-day-warmup` (cold start on day 0 or before N days of history), `warmup-N` (first N bars return 0 / default).

### PRICE (0-31)

| idx | name | formula | window | PIT | thesis | collapse | source |
|---:|---|---|---|---|---|---|---|
| 0 | ret_6 | `c[t]/c[t-30] - 1`, zero if `bod<30` | rolling-30 | warmup-30 | neutral | B | :576-583 |
| 1 | ret_12 | `c[t]/c[t-60] - 1`, zero if `bod<60` | rolling-60 | warmup-60 | neutral | B | :585-591 |
| 2 | volume_ratio | `vol[t] / rolling-60 mean(vol)`, per-day | rolling-60 | warmup-5 | neutral | — | :593-604 |
| 3 | bar_range | `(h - lo) / c` | bar-local | safe | neutral | — | :606-608 |
| 4 | realized_vol | 20-bar std of `log_ret`, per-day | rolling-20 | warmup-5 | vol | — | :610-618 |
| 5 | range_ratio | `bar_range / rolling-20 mean(bar_range)` | rolling-20 | warmup-5 | neutral | — | :620-629 |
| 6 | vwap_dist | `(c - VWAP) / c`, session VWAP | session-accum | safe | **rev** | — | :631-633 |
| 7 | session_range_pct | `(sess_high - sess_low) / c` | session-accum | safe | vol | — | :635-638 |
| 8 | prev_high_dist | `(c - prev_day_high) / c` | cross-day | cross-day-warmup (0 on day 0) | neutral | — | :640-642 |
| 9 | ema_cross | `(ema8 - ema21) / c`, per-day EMAs | rolling, per-day | warmup-~21 | cont | A | :644-647 |
| 10 | consec_direction | `sign * min(consec up/down bars, 10) / 10` | rolling-20 | warmup-2 | cont | — | :649-665 |
| 11 | speed_estimate | `|ret5| / realized_vol` | rolling-20 | warmup-5 | neutral | — | :667-674 |
| 12 | vix_roc | `(vix[t] - vix[t-10]) / vix[t-10]`, same-day | rolling-10 | warmup-10 | vol | — | :676-684 |
| 13 | minutes_to_close | `log1p(390 - bod) / log1p(390)` | clock | safe | neutral | J | :686-689 |
| 14 | vix_regime | VIX bucket: `{-1, -.33, .33, 1}` | regime | safe | vol | — | :691-697 |
| 15 | bollinger_position | 5-min Bollinger(20, 2) position, expanded | 5min-21 | warmup-~105 | neutral | K | :484-491 + :699-701 |
| 16 | rsi_7 | 5-min Wilder RSI(7), expanded | 5min-8 | warmup-~40 | neutral | K | :493-509 + :703-705 |
| 17 | session_range_position | `(c - sess_low) / (sess_high - sess_low)` | session-accum | safe | neutral | — | :707-710 |
| 18 | poc_dist | `(c - POC) / c`, POC from incremental volume profile | session-accum | **safe** (incremental, not full-day) | **rev** | — | :373-456 + :712-714 |
| 19 | va_position | `(c - VA_lo) / (VA_hi - VA_lo)`, incremental 70% value area | session-accum | **safe** (incremental) | **rev** | — | :440-456 + :716-719 |
| 20 | ib_break | `+1` if `c > IB_high`, `-1` if `c < IB_low`, else `0`; `bod >= 30` | ib30-bcast | unsafe<30 (returns 0 before bar 30; the IB values themselves are look-ahead) | cont | C | :721-727 |
| 21 | atr_14 | 14-bar ATR / `c`, per-day (min_periods=14) | rolling-14 | warmup-14 | vol | — | :729-744 |
| 22 | bar_delta | `(c - o) / max(h - lo, 1)`, clipped `[-1, 1]` | bar-local | safe | **rev** | — | :301-304 + :746-748 |
| 23 | session_cum_delta | cumsum of `bar_delta`, per-day | session-accum | safe | neutral | — | :306-309 + :750-752 |
| 24 | macdh_slope | `sign(diff(MACD_H))` on 5-min MACD(12,26,9) | 5min-27 | warmup-~135 | cont | A | :511-528 + :754-756 |
| 25 | force_index_2 | EMA(2) of `vol * diff(c)` / rolling std, 5-min | 5min-20 | warmup-~100 | cont | — | :530-542 + :758-760 |
| 26 | effort_vs_result | body-ratio / volume-ratio, 5-min | 5min-20 | warmup-~100 | neutral | — | :544-559 + :762-764 |
| 27 | trend_5min | `(ema13[k] - ema13[k-1]) / c`, 5-min EMA slope | 5min-13 | warmup-~65 | cont | A | :561-568 + :766-768 |
| 28 | vwap_slope | `(VWAP[t] - VWAP[t-5]) / VWAP[t-5]`, per-day | rolling-5 | warmup-5 | **rev** | — | :770-779 |
| 29 | intraday_sin | `sin(2π · bod / 390)` | clock | safe | neutral | J | :781-784 |
| 30 | intraday_cos | `cos(2π · bod / 390)` | clock | safe | neutral | J | :786-788 |
| 31 | intraday_phase | discrete 7-bucket phase / 6 | clock | safe | neutral | J | :790-802 |

### SESSION (32-45)

| idx | name | formula | window | PIT | thesis | collapse | source |
|---:|---|---|---|---|---|---|---|
| 32 | opening_gap_pct | `(session_open[t] - prev_close) / prev_close` | cross-day | cross-day-warmup (0 on day 0) | **rev** | — | :352-357 + :804-806 |
| 33 | session_open_dist | `(c - session_open) / c` | session-accum | safe | **rev** | — | :808-810 |
| 34 | first15_range_pct | `(first15_high - first15_low) / c`, bars 0-14 broadcast day-wide | first15-bcast | **unsafe<15** | vol | — | :332-349 + :812-815 |
| 35 | first15_close_position | `(first15_close - first15_low) / first15_range` | first15-bcast | **unsafe<15** | **rev** | D | :817-819 |
| 36 | first15_acceptance | clipped `(c - first15_mid) / (first15_range/2)`, `±1` if outside | first15-bcast | **unsafe<15** | **rev** | D | :821-828 |
| 37 | vwap_reclaim_state | `+1` if `c > VWAP` & any of prior 3 bars was `≤ VWAP`; `-1` symmetric | rolling-3 | warmup-1 | **rev** | — | :830-844 |
| 38 | ib_extension_pct | signed extension of `c` beyond IB high/low, normalized by `c` | ib30-bcast | **unsafe<30** | cont | C | :846-851 |
| 39 | marker_10am | tent: `clip(1 - |bod - 30|/10, 0, 1)` | clock | safe | neutral | J | :853-855 |
| 40 | marker_11am | tent around `bod = 90` | clock | safe | neutral | J | :857-859 |
| 41 | marker_1130am | tent around `bod = 120` | clock | safe | neutral | J | :861-863 |
| 42 | lunch_flag | `1` if `120 ≤ bod < 210` | clock | safe | neutral | J | :865-867 |
| 43 | power_hour_flag | `1` if `bod ≥ 300` | clock | safe | neutral | J | :869-871 |
| 44 | volume_climax_signal | `clip((volume_ratio - 1.25) * max(1 - |bar_delta|, 0), 0, 3)` — exhaustion proxy | rolling-60 | warmup-5 | **rev** | — | :873-876 |
| 45 | breakout_confirmation | sign-weighted magnitude when `c` breaks prev-session or IB levels with elevated volume | ib30-bcast + cross-day | **unsafe<30** + cross-day-warmup | cont | — | :878-885 |

### OPTION (46-57)

All OPTION features are computed bar-local from the wide-grid option data snapshot at bar `t`. None have look-ahead or session-broadcast issues beyond chain-data availability. One exception: `vrp` is filled after-the-fact in a merge step ([build_v2_dataset.py:918-925](../pipeline/build_v2_dataset.py#L918-L925)) using `realized_vol`, so it inherits the realized_vol warmup.

| idx | name | formula | window | PIT | thesis | collapse | source |
|---:|---|---|---|---|---|---|---|
| 46 | atm_iv | `(call_iv + put_iv) / 2` at nearest-ATM strike, Brent IV solver | chain-snap | safe (NaN if chain absent) | vol | — | :924-944 |
| 47 | vrp | `atm_iv² - realized_vol²` | rolling + snap | warmup-5 | vol | — | build_v2_dataset.py:918-925 |
| 48 | iv_percentile | rank of `atm_iv` in rolling 60-day IV history | cross-day | cross-day-warmup (`≥ 5` days) | vol | — | :950-955 |
| 49 | atm_gamma | BS gamma at ATM with solved IV | chain-snap | safe | **rev** (Q4) | — | :957-962 |
| 50 | atm_theta_per_bar | BS theta per bar at ATM | chain-snap | safe | **rev** (Q4) | — | :964-965 |
| 51 | gamma_pressure | `Σ γ · volume · side_sign` across all strikes | chain-snap | safe | vol | — | :967-990 |
| 52 | aggregate_charm | `Σ charm · volume · side_sign` across all strikes | chain-snap | safe | vol | — | :971-991 |
| 53 | option_spread_pct | `(call_high - call_low) / mid` at nearest ATM | chain-snap | safe | **exec** | — | :993-1000 |
| 54 | iv_skew_pct | `(put_iv - call_iv) / atm_iv` at ATM | chain-snap | safe | vol | — | :1002-1006 |
| 55 | current_moneyness_pct | `(atm_strike_open - spx) / spx * 100` — drift from session-open ATM | chain-snap | safe | neutral | L | :1008-1009 |
| 56 | near_atm_moneyness_pct | `(nearest_strike - spx) / spx * 100` | chain-snap | safe | neutral | L | :1011-1012 |
| 57 | theta_acceleration | `1 / sqrt(max(mtc, 1))` | clock | safe | **rev** (Q4) | — | :1014-1015 |

### SURFACE (58-70)

All SURFACE features are bar-local slice aggregates over strikes in a dynamic band around the current ATM (`dynamic_slice_bounds(spot)`). All source line numbers below are in [build_v2_dataset.py](../pipeline/build_v2_dataset.py).

| idx | name | formula | window | PIT | thesis | collapse | source |
|---:|---|---|---|---|---|---|---|
| 58 | slice_call_iv_mean | mean of solved IV across in-slice calls | chain-snap | safe (0 if empty slice) | vol | E | :284 |
| 59 | slice_put_iv_mean | mean of solved IV across in-slice puts | chain-snap | safe | vol | E | :285 |
| 60 | slice_iv_skew_slope | linear coeff of `IV ~ (strike - ATM)/STRIKE_GRID`, txn-weighted | chain-snap | safe (0 if `<2` unique strikes) | vol | — | :287-293 |
| 61 | slice_iv_curvature | quadratic coeff of same fit | chain-snap | safe (0 if `<3` unique strikes) | vol | — | :294-296 |
| 62 | slice_gamma_concentration | `max(|γ|) / sum(|γ|)` across in-slice | chain-snap | safe | vol | F | :298-300 |
| 63 | slice_gamma_dollar_concentration | same but with `γ · S²·0.01` (dollar gamma) | chain-snap | safe | vol | F | :302-304 |
| 64 | slice_theta_pressure | `sum(|θ|) / max(sum(mid), 1e-6)` | chain-snap | safe | vol | — | :306-307 |
| 65 | slice_dist_to_max_gamma | `(argmax_strike(|γ|) - ATM) / STRIKE_GRID` | chain-snap | safe | vol | G | :309-311 |
| 66 | slice_dist_to_max_gamma_dollar | same with dollar gamma | chain-snap | safe | vol | G | :313-315 |
| 67 | slice_call_put_gamma_imbalance | `(Σ|γ_call| - Σ|γ_put|) / total` | chain-snap | safe | vol | — | :317-321 |
| 68 | slice_mean_spread | mean spread-fraction across in-slice | chain-snap | safe | **exec** | — | :323-324 |
| 69 | slice_txn_center_share | fraction of slice transactions within ±2 strikes of ATM | chain-snap | safe | **exec** | — | :326-330 |
| 70 | slice_quality_share | fraction of in-slice contracts with `quality ≥ QUALITY_VALID` | chain-snap | safe | **exec** | — | :332 |

### FLOW (71-78)

All FLOW features are bar-local sums over the wide-grid snapshot. No warmup, no look-ahead.

| idx | name | formula | window | PIT | thesis | collapse | source |
|---:|---|---|---|---|---|---|---|
| 71 | log_near_call_volume | `log1p(call_volume at nearest-ATM)` | chain-snap | safe | neutral | I | :1062 |
| 72 | log_near_put_volume | `log1p(put_volume at nearest-ATM)` | chain-snap | safe | neutral | I | :1063 |
| 73 | call_put_flow_ratio | `call_vol / (call_vol + put_vol)` at nearest-ATM | chain-snap | safe | vol | H | :1064 |
| 74 | log_total_volume | `log1p(call + put at nearest-ATM)` | chain-snap | safe | neutral | I | :1065 |
| 75 | chain_call_put_ratio | `chain_call_vol / chain_total` across all strikes | chain-snap | safe | vol | H | :1067-1071 |
| 76 | log_chain_volume | `log1p(chain_total_volume)` | chain-snap | safe | neutral | I | :1072 |
| 77 | log_near_transactions | `log1p(call_txn + put_txn)` at nearest-ATM | chain-snap | safe | neutral | I | :1074-1078 |
| 78 | put_call_txn_ratio | `put_txn / (call_txn + put_txn)` at nearest-ATM | chain-snap | safe | vol | H | :1079 |

## Thesis-alignment summary

Distribution across 79 features:

| tag | count | share |
|-----|------:|------:|
| reversion (rev) | 14 | 18% |
| vol-expansion (vol) | 25 | 32% |
| continuation (cont) | 8 | 10% |
| execution-quality (exec) | 4 | 5% |
| neutral-structure | 28 | 35% |

**Reversion features (14):** `vwap_dist` [6], `poc_dist` [18], `va_position` [19], `bar_delta` [22], `vwap_slope` [28], `opening_gap_pct` [32], `session_open_dist` [33], `first15_close_position` [35], `first15_acceptance` [36], `vwap_reclaim_state` [37], `volume_climax_signal` [44], `atm_gamma` [49], `atm_theta_per_bar` [50], `theta_acceleration` [57].

**Continuation (8):** `ema_cross` [9], `consec_direction` [10], `ib_break` [20], `macdh_slope` [24], `force_index_2` [25], `trend_5min` [27], `ib_extension_pct` [38], `breakout_confirmation` [45].

**Vol-expansion (25):** `realized_vol` [4], `session_range_pct` [7], `vix_roc` [12], `vix_regime` [14], `atr_14` [21], `first15_range_pct` [34], `atm_iv` [46], `vrp` [47], `iv_percentile` [48], `gamma_pressure` [51], `aggregate_charm` [52], `iv_skew_pct` [54], `slice_call_iv_mean` [58], `slice_put_iv_mean` [59], `slice_iv_skew_slope` [60], `slice_iv_curvature` [61], `slice_gamma_concentration` [62], `slice_gamma_dollar_concentration` [63], `slice_theta_pressure` [64], `slice_dist_to_max_gamma` [65], `slice_dist_to_max_gamma_dollar` [66], `slice_call_put_gamma_imbalance` [67], `call_put_flow_ratio` [73], `chain_call_put_ratio` [75], `put_call_txn_ratio` [78].

**Execution-quality (4):** `option_spread_pct` [53], `slice_mean_spread` [68], `slice_txn_center_share` [69], `slice_quality_share` [70].

**Neutral-structure (28):** everything else (returns, generic TA oscillators, clock features, raw volume/transaction magnitudes).

**Ratio sanity check:** reversion at 14/79 (18%) is below the 30/79 inflation ceiling the plan requires, so the tagging is not over-fit to the winning thesis. The heavy `vol` bucket (25/79) is driven by 13 SURFACE + 4 SURFACE-volume-ratio + several Greeks — consistent with a 0DTE dataset that was expanded specifically to capture chain / dealer-hedging state.

## Unsafe-window reference

Features that are **not** safe to use blindly from bar 0. Every feature below needs either a `bar_of_day >= X` guard, a day-of-history guard, or a warmup bar index guard before it carries real information.

### Intra-session look-ahead (broadcast-fills with future bars)

These assign values to early-session bars using data that arrives **after** those bars. In a live system, the guard must be enforced explicitly.

| feature | idx | unsafe when | guard |
|---|---:|---|---|
| `first15_range_pct` | 34 | `bod < 15` | only use when `bar_of_day >= 15` |
| `first15_close_position` | 35 | `bod < 15` | same |
| `first15_acceptance` | 36 | `bod < 15` | same |
| `ib_break` | 20 | `bod < 30` | value is forced to 0 in code, but the IB high/low reference itself is look-ahead if consumed via join |
| `ib_extension_pct` | 38 | `bod < 30` | only use when `bar_of_day >= 30` |
| `breakout_confirmation` | 45 | `bod < 30` | same + prior-day warmup |

### Bar-rolling warmup (0-filled or wrong during warmup)

| feature | idx | warmup | notes |
|---|---:|---|---|
| `ret_6` | 0 | 30 bars | zero before bar 30 |
| `ret_12` | 1 | 60 bars | zero before bar 60 |
| `volume_ratio` | 2 | 5 bars | `min_periods=5` |
| `realized_vol` | 4 | 5 bars | `min_periods=5`; full at bar 20 |
| `range_ratio` | 5 | 5 bars | same |
| `vix_roc` | 12 | 10 bars | zero before bar 10 |
| `atr_14` | 21 | 14 bars | `min_periods=14` |
| `vwap_slope` | 28 | 5 bars | uses VWAP from 5 bars prior |
| `consec_direction` | 10 | 2 bars | needs prior bar |
| `vwap_reclaim_state` | 37 | 1 bar | needs at least one prior bar |
| `speed_estimate` | 11 | 5 + realized_vol warmup | |
| `bollinger_position` | 15 | ~105 bars | 5-min rolling needs n5 ≥ 21 |
| `rsi_7` | 16 | ~40 bars | 5-min needs n5 > 7 |
| `trend_5min` | 27 | ~65 bars | 5-min needs n5 > 13 |
| `force_index_2` | 25 | ~100 bars | 5-min needs n5 > 20 |
| `effort_vs_result` | 26 | ~100 bars | same |
| `macdh_slope` | 24 | ~135 bars | 5-min needs n5 > 26 |
| `volume_climax_signal` | 44 | 5 bars | inherits `volume_ratio` warmup |

### Cross-day warmup

| feature | idx | cold start | notes |
|---|---:|---|---|
| `prev_high_dist` | 8 | day 0 | zero on first day of the dataset |
| `opening_gap_pct` | 32 | day 0 | zero on first day |
| `iv_percentile` | 48 | `<5` days of IV history | NaN/0 until history builds |
| `breakout_confirmation` | 45 | day 0 | uses prev-session high/low |

### Safe from bar 0

Every other feature. Notable confirmations:

- **`poc_dist` [18] and `va_position` [19] are incremental.** The volume profile at bar `i` is built from `day_c[:i+1]` only ([compute_features.py:394-456](../pipeline/compute_features.py#L394-L456)). Older audit notes that claimed full-day look-ahead here are wrong about current code.
- **All SURFACE features (58-70)** are bar-time chain snapshots — no look-ahead.
- **All FLOW features (71-78)** are bar-time wide-grid aggregates — no look-ahead.
- **All clock features** (`minutes_to_close`, `intraday_sin/cos/phase`, `marker_*`, `lunch_flag`, `power_hour_flag`) are functions of `bar_of_day` alone — safe.

## Collapse-candidate groups

Clusters of features likely to be highly correlated. A first-pass model (mechanical or ML) that uses one representative per group gets nearly all the signal at a fraction of the surface. Proposed representative is the first name listed.

| group | members | proposed rep | rationale |
|---|---|---|---|
| **A — short-term trend** | `trend_5min` [27], `ema_cross` [9], `macdh_slope` [24] | `trend_5min` | all three measure short-term slope direction; `trend_5min` is continuous, easier to use than `sign()` of MACD-H |
| **B — intraday return scales** | `ret_12` [1], `ret_6` [0] | `ret_12` | same signal at two horizons; 60-bar is more stable |
| **C — IB breakout state** | `ib_extension_pct` [38], `ib_break` [20] | `ib_extension_pct` | continuous subsumes the `{-1, 0, 1}` ternary |
| **D — first-15 structure** | `first15_acceptance` [36], `first15_close_position` [35] | `first15_acceptance` | symmetric `[-1, 1]` with outside-the-range pinning; subsumes position |
| **E — slice IV level** | `slice_call_iv_mean` [58], `slice_put_iv_mean` [59] | prefer `atm_iv` [46] | the OPTION block already holds the ATM level; keep slice only if side-split matters |
| **F — gamma concentration** | `slice_gamma_dollar_concentration` [63], `slice_gamma_concentration` [62] | dollar version | dollar-weighting is the economically relevant measure |
| **G — gamma-wall distance** | `slice_dist_to_max_gamma_dollar` [66], `slice_dist_to_max_gamma` [65] | dollar version | same rationale as F |
| **H — call/put sentiment** | `chain_call_put_ratio` [75], `call_put_flow_ratio` [73], `put_call_txn_ratio` [78] | `chain_call_put_ratio` | broadest coverage; the other two are near-ATM-only or transaction-count variants |
| **I — flow magnitudes (logs)** | `log_total_volume` [74], `log_near_call_volume` [71], `log_near_put_volume` [72], `log_chain_volume` [76], `log_near_transactions` [77] | `log_total_volume` + `log_chain_volume` | keep one "near" and one "chain"; drop the side splits and transaction proxy |
| **J — clock encodings** | `intraday_phase` [31], `intraday_sin` [29], `intraday_cos` [30], `marker_10am/11am/1130am` [39-41], `lunch_flag` [42], `power_hour_flag` [43], `minutes_to_close` [13] | `minutes_to_close` | any mechanical strategy already fixes its trade window; the others leak policy decisions |
| **K — TA oscillators** | `rsi_7` [16], `bollinger_position` [15] | `rsi_7` | both generic, `rsi_7` is the simpler of the two |
| **L — moneyness** | `near_atm_moneyness_pct` [56], `current_moneyness_pct` [55] | `near_atm_moneyness_pct` | live nearest-ATM is bar-current; `current_moneyness_pct` is drift from session-open ATM |

Features not in any group are treated as standalone.

## Cross-reference — Codex's reversion shortlist

[feature_shortlist_opening_reversion.md](./feature_shortlist_opening_reversion.md) proposes a 12-core + 8-supporting set. Status after this audit:

### Codex core (12) — all confirmed present with the following caveats

| feature | idx | audit thesis tag | audit note |
|---|---:|---|---|
| `vwap_dist` | 6 | **rev** | confirmed; safe from bar 0 |
| `vwap_reclaim_state` | 37 | **rev** | confirmed; needs `bar_of_day >= 1` |
| `vwap_slope` | 28 | **rev** | confirmed; 5-bar warmup |
| `opening_gap_pct` | 32 | **rev** | confirmed; day-0 cold start |
| `session_open_dist` | 33 | **rev** | confirmed; safe from bar 0 |
| `first15_close_position` | 35 | **rev** | confirmed; **unsafe before `bar_of_day >= 15`** — strategy card's `bar_of_day ∈ [15, 120]` guard covers this, but any audit beyond that window must re-check |
| `first15_acceptance` | 36 | **rev** | same caveat |
| `bar_delta` | 22 | **rev** | confirmed; safe from bar 0 |
| `volume_climax_signal` | 44 | **rev** | confirmed; inherits 5-bar warmup from `volume_ratio` |
| `atm_gamma` | 49 | **rev** (Q4) | confirmed; bar-local chain snapshot |
| `atm_theta_per_bar` | 50 | **rev** (Q4) | confirmed; bar-local |
| `option_spread_pct` | 53 | **exec** (not rev) | confirmed. Codex groups this under core because expressivity requires execution feasibility; the tag distinction is a semantic note, not a disagreement |

### Codex supporting (8)

| feature | idx | audit tag | note |
|---|---:|---|---|
| `volume_ratio` | 2 | neutral | 5-bar warmup |
| `session_range_position` | 17 | neutral | safe from bar 0 |
| `atr_14` | 21 | vol | 14-bar warmup |
| `iv_percentile` | 48 | vol | cross-day warmup (`<5` days = unsafe) |
| `vrp` | 47 | vol | 5-bar warmup via realized_vol |
| `call_put_flow_ratio` | 73 | vol | safe; near-ATM only |
| `put_call_txn_ratio` | 78 | vol | safe |
| `slice_mean_spread` | 68 | exec | safe |

### Disagreements / call-outs — resolution (2026-04-19)

All five items below were raised by this audit and then adjudicated in a
joint pass with Codex. The outcomes are reflected in the current version of
[feature_shortlist_opening_reversion.md](./feature_shortlist_opening_reversion.md).

1. **`vrp` [47] and `iv_percentile` [48]** — *adopted as gate-first
   supporting.* They stay in the supporting bucket but are explicitly
   promoted to "natural first additions if the baseline needs a no-trade
   filter", because both answer Q4 ("is long premium a sane expression of
   this move?") at regime level.
2. **`theta_acceleration` [57]** — *deferred, not missing.* Confirmed
   thesis-aligned but redundant in v1 given (a) the morning-only trade
   window, (b) `atm_theta_per_bar` already in core, and (c) the 30-minute
   time stop. Explicitly listed as deferred supporting rather than as a
   missing omission.
3. **`va_position` [19]** — *promoted to supporting.* The audit's incremental
   confirmation cleared the historical look-ahead concern. `poc_dist` [18]
   remains in "useful but not first-order" as a reconsideration candidate if
   `va_position` alone proves insufficient.
4. **Clock features (group J)** — *no change.* Correctly excluded; the
   mechanical strategy card hard-codes the trade window, so clock features
   would double-encode policy.
5. **Broad SURFACE features** — *staged for v2.* Kept out of v1 on
   coherence grounds. The shortlist now names
   `slice_gamma_dollar_concentration` + `slice_dist_to_max_gamma_dollar` as
   the minimal vol-aware extension if the mechanical baseline shows
   promise — the cheapest way to add dealer-hedging state without
   broadening the surface.

**Open tag-note:** `option_spread_pct` [53] is tagged `execution-quality`,
not `reversion`. It stays in Codex's core shortlist on feasibility grounds
(no tradeable premium → no usable baseline); the tag distinction is a
semantic clarification, not a disagreement.

**This audit does not re-prune.** It provides the evidence the pruning
discussion needed.

## Normalization

Implemented in [features.py:105-150](../core/features.py#L105-L150).

- Rolling z-score with 60-day window (`60 × 390 = 23,400` bars), expanding during the first `window` bars.
- Clip to `[-5, 5]`, NaN → 0.
- Features in `_NO_NORMALIZE` ([features.py:61-98](../core/features.py#L61-L98)) are left untouched because they are already bounded or categorical: `minutes_to_close`, `theta_acceleration`, `iv_percentile`, `vix_regime`, `ib_break`, `macdh_slope`, `bollinger_position`, `rsi_7`, `session_range_position`, `va_position`, `bar_delta`, `consec_direction`, `effort_vs_result`, `call_put_flow_ratio`, `chain_call_put_ratio`, `put_call_txn_ratio`, `intraday_sin`, `intraday_cos`, `intraday_phase`, `first15_close_position`, `first15_acceptance`, `vwap_reclaim_state`, `marker_10am`, `marker_11am`, `marker_1130am`, `lunch_flag`, `power_hour_flag`, `volume_climax_signal`, `breakout_confirmation` (29 features total).

## What this doc does not cover

- **22 contract-row features** in `CONTRACT_FEATURE_FIELDS` — separate audit.
- **Labeling / sidecar fields** — see [labeling.md](labeling.md).
- **Normalization drift** — checked by `data_integrity.py` at dataset build time; separate concern from feature definition.
- **Feature-value distribution statistics** — not recomputed here; run `python3 -m v2.analysis.harness_eval --data v2/data.pt` for current shape checks.
