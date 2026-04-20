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

## History note — the shortlist was revised by the V2 stress test

This file originally listed 12 core features built around the assumption
that the **reclaim event itself** (`vwap_reclaim_state`) was the central
signal. The V1A / V1B mechanical baselines (2026-04-19) built a hand-coded
reclaim/reject trigger on that premise and failed their falsification
bar. V2 (the learned RandomForest scorer over the same 12-core set)
passed falsification with a +2.28pp gap vs Control A.

The V2 stress-test ablations (2026-04-19, committed same day) then
revealed that the model's edge is concentrated in a *different* subset
of the shortlist:

- `first15_acceptance` is the single most load-bearing feature — removing
  it alone flips strategy mean and gap_vs_A negative.
- `opening_gap_pct` and `first15_close_position` are materially
  important.
- **`vwap_reclaim_state` is actively slightly harmful.** Ablating it
  *improves* the learned strategy by +0.36pp mean and +0.30pp gap. The
  original centerpiece of the shortlist is falsified for the learned
  baseline.
- `bar_delta` is useful (removing it drops gap by 0.79pp) but not the
  centerpiece.

The shortlist below has been reorganized to reflect what the model
actually learned. The reversion thesis FAMILY survives; its central
event has shifted from "reclaim back through VWAP" to "displacement +
first-15 settlement + first-15 acceptance."

## Core features (post-V2 stress test)

These are the features that survive as load-bearing for the V2 learned
scorer. Listed in decreasing order of measured importance / ablation
impact.

### Load-bearing (V2 collapses without these)

| Feature | Role in the learned model | Why it belongs |
|---|---|---|
| `first15_acceptance` | First-15 acceptance state (bounded `[-1, +1]`) | Single most important feature in RF importance across all 5 folds. Ablation drops mean to −0.73%, gap to −0.51%. |
| `opening_gap_pct` | Overnight displacement | #1 RF importance in every fold (24-32%). Ablation drops mean by 0.74pp, gap by 0.63pp. |
| `first15_close_position` | Where the opening auction settled | Consistent #2 RF importance. Ablation drops gap by 1.24pp. |

### Supporting (material but not catastrophic if removed)

| Feature | Role | Ablation Δ gap |
|---|---|---:|
| `bar_delta` | Directional conviction of the current bar | −0.79pp |
| `atm_theta_per_bar` | 0DTE decay cost at the candidate bar | (top-5 RF importance in 4 of 5 folds; not single-feature ablated) |
| `atm_gamma` | Responsiveness of long premium to the spot move | (top-5 RF importance in 2 of 5 folds) |
| `vwap_dist` | Distance from session VWAP | (top-5 RF importance in 3 of 5 folds) |

### Execution-quality (kept for feasibility, not as signal)

| Feature | Role |
|---|---|
| `option_spread_pct` | Context-level spread gate; rejected entries when the near-ATM call spread is > 20%. Not in the learned model's top features; kept because a good spot pattern is not tradeable through wide premium. |

### Falsified as a core premise

| Feature | Status | Evidence |
|---|---|---|
| **`vwap_reclaim_state`** | **Falsified as a core feature for this dataset.** Removing it *improves* V2 mean by +0.36pp and gap by +0.30pp. It is a minor drag on the learned scorer. | V2 stress test ablations, 2026-04-19 lab notebook entry. |

The V1A/V1B hand-coded trigger was built around `vwap_reclaim_state`.
The learned model finds the feature either uninformative or
conflicting with the other signals; the pattern it identifies as
"reclaim" does not coincide with the bars the scorer would choose to
trade.

### Note on `vwap_slope`, `session_open_dist`, `volume_climax_signal`

These three were in the original core set but did not rank in the V2
model's top-5 importance in any fold. They have not been stress-tested
individually (ablation plan focused on the highest- and lowest-impact
candidates). They stay in the model input set for now — they cost
nothing — but should not be described as load-bearing without further
evidence.

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
