# Feature Shortlist — Opening-Structure Reversion

## Purpose

This file defines the **first-pass feature surface** for the opening-structure
reversion thesis in [strategy_card_opening_reversion.md](strategy_card_opening_reversion.md).

The goal is to keep only features that directly answer one question:

> Has the opening move stretched too far, failed, and begun reverting in a way
> that a short-horizon 0DTE long call or put can express?

## Feature truth

The live artifact has **79 base features** (32 price + 14 session + 12 option
+ 13 surface + 8 flow).

- Live source of truth: [`feature_schema.md`](feature_schema.md), regenerated
  against live code in the 2026-04-19 audit.
- Every decision in this shortlist is backed by that schema's per-feature
  table, point-in-time safety classification, and thesis tags. Whenever this
  file disagrees with the schema, the schema wins.

## Core features

Use these `12` as the first-pass core set.

| Feature | Purpose in the thesis | Source | Why it belongs |
|---|---|---|---|
| `vwap_dist` | Measures extension away from fair value | live base feature | Reversion starts with stretch away from VWAP |
| `vwap_reclaim_state` | Detects reclaim / reject event | live base feature | Captures the actual reversal moment |
| `vwap_slope` | Shows whether fair value is rising or falling | live base feature | Prevents taking reclaim signals against a sharply sloping VWAP blindly |
| `opening_gap_pct` | Encodes overnight displacement | live base feature | Gap context changes whether the move is a fade or continuation risk |
| `session_open_dist` | Measures snap-back versus the opening print | live base feature | Useful for gap-fade style reversions |
| `first15_close_position` | Where the opening auction settled | live base feature | Tells whether early structure accepted high or low |
| `first15_acceptance` | Whether current price is being accepted back into opening structure | live base feature | Distinguishes reclaim from ongoing expansion |
| `bar_delta` | Immediate directional conviction of the signal bar | live base feature | Filters weak reclaims / rejects |
| `volume_climax_signal` | Proxy for exhaustion and failed push | live base feature | Reversion often starts after an emotional/exhaustive move |
| `atm_gamma` | Measures responsiveness of long premium | live base feature | High gamma makes the short reversion move worth expressing |
| `atm_theta_per_bar` | Measures cost of being wrong or late | live base feature | Prevents ignoring the 0DTE decay tax |
| `option_spread_pct` | Execution feasibility | live base feature | A good spot pattern can still be untradeable through wide premium |

*Tag note:* the 2026-04-19 audit tags `option_spread_pct` as
`execution-quality`, not `reversion`. It stays in the core because a reversion
thesis that cannot be expressed through tradeable premium is not a usable
baseline. The semantic distinction is worth naming, but it does not change
the shortlist decision.

## Supporting features

These are allowed as secondary filters, but they are **not** part of the
minimum baseline trigger.

Two of them — `vrp` and `iv_percentile` — are promoted to **gate-first**
status: they are the natural first additions if the mechanical baseline needs
a no-trade filter, because both answer Q4 ("is long premium a sane expression
of this move?") with a clean, regime-level signal.

| Feature | Role |
|---|---|
| `va_position` | Where price sits within the incremental volume-profile value area — fair-value / acceptance proxy. Promoted from "useful but not first-order" after the audit confirmed it is genuinely incremental with no look-ahead. |
| `volume_ratio` | Confirms that the opening move happened on meaningful participation |
| `session_range_position` | Normalizes where price sits within the current day's range |
| `atr_14` | Helps normalize target and stop distances |
| `iv_percentile` | **Gate-first.** Natural no-trade filter when long premium is unusually rich relative to its own 60-day history |
| `vrp` | **Gate-first.** Natural no-trade filter in high-variance-risk-premium regimes where realized vol is unlikely to outrun paid premium |
| `call_put_flow_ratio` | Optional sentiment/participation hint |
| `put_call_txn_ratio` | Optional sentiment/participation hint |
| `slice_mean_spread` | Optional chain-quality filter if execution quality becomes a problem |

## Excluded for the first baseline

These may be useful later, but they do **not** belong in the first pass because
they point at different market stories or broaden the surface too early.

### Off-thesis continuation features

- `breakout_confirmation`
- `ib_break`
- `ib_extension_pct`
- `trend_5min`
- `ema_cross`
- `macdh_slope`
- `force_index_2`

These are better aligned with continuation or breakout follow-through than with
failed-extension reversion.

### Redundant clock encodings

- `intraday_sin`
- `intraday_cos`
- `intraday_phase`
- `marker_10am`
- `marker_11am`
- `marker_1130am`
- `lunch_flag`
- `power_hour_flag`

The thesis already fixes the trade window explicitly. Clock features would add
noise or leak policy decisions back into the model.

### Broad surface features excluded from v1

- `slice_call_iv_mean`
- `slice_put_iv_mean`
- `slice_iv_skew_slope`
- `slice_iv_curvature`
- `slice_gamma_concentration`
- `slice_gamma_dollar_concentration`
- `slice_theta_pressure`
- `slice_dist_to_max_gamma`
- `slice_dist_to_max_gamma_dollar`
- `slice_call_put_gamma_imbalance`
- `slice_txn_center_share`
- `slice_quality_share`

Reason: these describe the whole chain and can be useful, but they make the
feature story too broad for the first mechanical test. They are the **natural
next layer** when the simple reversion baseline proves it needs vol-aware
context. The audit's collapse groups F and G
([feature_schema.md](feature_schema.md)) identify the smallest sensible
addition: `slice_gamma_dollar_concentration` + `slice_dist_to_max_gamma_dollar`.

### Useful but not first-order to the thesis

- `poc_dist` — incremental and thesis-aligned per the audit; reconsider if
  `va_position` alone is insufficient
- `bollinger_position`
- `rsi_7`
- `session_cum_delta`
- `effort_vs_result`
- `current_moneyness_pct`
- `near_atm_moneyness_pct`
- `theta_acceleration` — **deferred supporting, not a missing omission.**
  Thesis-aligned (Q4), but redundant in v1 given the morning-only window,
  `atm_theta_per_bar` already in core, and the 30-minute time stop
- raw flow totals

These can become follow-up filters, but they are not load-bearing for v1.

## Safety rules to enforce

These rules apply to any consumer of this shortlist — mechanical backtest,
ML stage, or replay. They are the actionable output of the audit's unsafe-window
reference.

| Rule | Feature(s) | Guard |
|---|---|---|
| first-15 look-ahead | `first15_range_pct`, `first15_close_position`, `first15_acceptance` | only use when `bar_of_day >= 15` — the v1 trigger's `bod ∈ [15, 120]` already covers this |
| initial-balance look-ahead | `ib_break`, `ib_extension_pct`, `breakout_confirmation` | only use when `bar_of_day >= 30`. The v1 trigger excludes these continuation features, so the guard is redundant today; keep it in mind if a future continuation baseline reintroduces them |
| IV-percentile cross-day warmup | `iv_percentile` | needs ≥ 5 days of IV history before use; same for any rank-based filter built on top |
| old audit docs | 47/52-feature audit notes | untrusted — use [`feature_schema.md`](feature_schema.md) as the live contract |

**Confirmed (2026-04-19 audit):** `poc_dist` and `va_position` are computed
incrementally in live code — no full-day look-ahead. Older audit notes that
claimed otherwise are wrong about current code and should not be carried
forward.

## Needs-new-data list

These belong to the thesis family, but the current repo does not have them in a
clean first-class way.

- ES / NQ leader-laggard state
- event calendar / release-time flags
- breadth / internals (`A/D`, `A/D volume`)
- richer premarket inventory context

These should be added only if the simple baseline proves promising and the
missing context looks load-bearing.

## Pruning rule

For this thesis, a feature stays only if it can answer one of four questions:

1. Was price extended away from fair value?
2. Did that extension fail and reclaim / reject?
3. Is the opening structure accepting the reversal?
4. Is 0DTE long premium still a sane expression of the move?

If a feature cannot answer one of those questions directly, it does not belong
in the first baseline.
