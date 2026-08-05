# Protocol101 Round 2 — Exact Experiment Specifications

Generated: 2026-07-05
Author: external AI (Claude), responding to `01_round2_prompt.md`
Status: specification only — no thresholds tuned, no recorder days touched, no candidate promoted

This document is grounded in the uploaded code. Facts asserted about current mechanics were read from
`06_supervised_pilot.py`, `07_training_runner.py`, `05_protocol101_feature_contract.py`,
`09_feature_jitter_gate.py`, `10_failure_diagnostic.py`, and the attribution CSVs (`19`, `20`).
Where a spec depends on something I could not verify from the uploads, it is marked `VERIFY:`.

---

## 0. Ground Truth This Spec Is Built On

Extracted from the code, stated so your team can falsify my premises before implementing anything:

1. **Per-contract labels are already realizable policy values.** `labels_net_pnl[strike, right, policy]`
   is the net PnL of entering at ask at decision minute t and exiting at bid under the lifecycle policy.
   `simulate_model_policy` pays exactly `decision.labels[idx]` per trade. So "label surgery" targets
   *decision-level aggregation*, not the per-contract label pipeline.
2. **Three lifecycle policies are already precomputed** (`POLICY_META` in `07_training_runner.py`):
   - policy 0: `ask_to_bid_stop35_target60_hold10m`, cooldown 10 (used by attempts 107–137)
   - policy 1: `ask_to_bid_stop50_target100_hold25m`, cooldown 25
   - policy 2: `ask_to_bid_stop65_target150_hold45m`, cooldown 45
   This means a first lifecycle-attribution pass costs zero new data (Section 6, Phase 1).
3. **The base gate** `put_near_after_0940_vwap_m2_10` = ET time ≥ 09:40, `spx_close − spx_vwap ∈ [−2, +10)`,
   right = P, `10 ≤ |offset| ≤ 20`.
4. **The deterministic selector** `stable_abs_offset_20` = among eligible candidates, minimize the tuple
   `(| |offset| − 20 |, |offset|, offset, right, index)`. No quote data enters the ranking key.
5. **Tradability** (`candidate_is_tradable_values`): finite bid/ask/mid, `0.50 ≤ mid ≤ 35`, spread ≤ $0.50 abs
   and ≤ 25% of mid, bid/ask size ≥ 1, finite repaired greeks. Greeks come from the shared in-house pricer
   (`compute_repaired_greeks`) with vendor greeks audit-only under the live contract. (This mostly answers
   Round-1's in-house-IV concern; the July 1 IV delta therefore traces to quote-input differences at the
   decision instant, which debounce addresses, not to vendor-supplied IV.)
6. **Stress accounting:** reported "after stress" metrics subtract $20/trade (`_stress_trades`,
   `cash_pnl_adjustment`). `VERIFY:` whether exchange/broker fees are also inside `labels_net_pnl` or the
   $20 stress is the entire cost model beyond ask/bid crossing. Record the answer in the registry; the
   classification hurdle in Section 3 assumes the $20 stress is the cost model.
7. **Splits:** train = Jul–Dec 2025 (128 sessions). Validation ≈ first ~10 sessions of March 2026
   (Mar 3–16). Diagnostic ≈ last ~10 sessions of March 2026 (Mar 19–31). **The entire out-of-sample
   evidence base spans one calendar month.** ~137 attempts have been adjudicated on it; the validation
   split is burned as a selection instrument.
8. **Existing baseline machinery:** `simulate_baseline(kind="random_valid")` ignores the entry filter,
   session cap, daily loss, and affordability. It is *not* a valid null for gated strategies and must not
   be used as one.
9. **Attribution CSV evidence (new, computed from uploads 19/20):**
   - Side-consistency of the decision-level target: `decision_top_label_contract_id` has the same right as
     the traded put in 65% (att130) / 50% (att131) of validation trades but only **47% / 41%** of
     diagnostic trades. The presence/best targets are direction-blind volatility detectors; the policy
     hard-codes puts.
   - Oracle capture ratio (selected label sum ÷ decision-top-label sum): 36% / 16% validation →
     **3.8% / 0.9%** diagnostic. Median selected label in diagnostic: **−$60 / −$110**.
   These two numbers are the mechanism of the validation→diagnostic collapse and the primary motivation
   for label L1 below.

---

## 1. Experiment 1 — Gate-Only Strategy and Null Baselines

Answers to your numbered questions first, then the experiment cards.

1. **Which gate first:** the broad base gate `put_near_after_0940_vwap_m2_10`. It is the only gate with a
   split-stable average label (~$42 both splits) and sufficient trade count. Run the two sub-gates
   (`_near_vwap`, `_premium_gte_7_5`) as pre-registered variants in the same batch — they are one config
   line each — but the base gate is primary. Do not add new gates to this batch.
2. **Entry timing:** first eligible minute, then cooldown — this is what the live system can actually do
   and what `simulate_model_policy` already implements. Highest-label minute is forbidden (hindsight).
   A deterministic time-slot schedule is run separately as probe 1D, not as the primary.
3. **Contract selector:** `stable_abs_offset_20` exactly as implemented, gated by
   `candidate_is_tradable_values` and affordability. No VIX-conditioned offset in v1: context-conditioned
   selection is a new degree of freedom and needs Section 6 evidence (a materially negative selection
   delta) to justify its multiplicity cost. No premium-band-first selection in v1 for the same reason.
4. **Ties at equal offset distance:** the existing tuple `(| |offset|−20 |, |offset|, offset, right, index)`
   already resolves ties deterministically. Keep it. One requirement: ladder index order must be
   strike-sorted identically on both feeds (it is, by construction of the ±50/5-step ladder) — assert this
   in the paired-replay harness rather than assuming it.
5. **Both modes:** yes. 1A = strict serial (cap 3, daily loss 500, cooldown 10, affordability from $10k).
   1B = unconstrained parallel (every gated minute scored as an independent counterfactual; no cap, no
   cooldown, no affordability, no daily loss).
6. **Which counts as readiness evidence:** strict serial only. Parallel mode is diagnostic — it measures
   signal, not a tradable account path.
7. **The random-in-gate null, exactly:** two nulls, both under full serial rules (cooldown 10, cap 3,
   daily loss 500, affordability), both using the same `stable_abs_offset_20` selector so the null isolates
   *minute-choice skill only*:
   - **Null-U (uniform):** per session, walk minutes chronologically in a random order proposal: shuffle
     the session's gated eligible minutes with the run's seed, then accept minutes greedily in shuffled
     order subject to (≥ cooldown from all accepted minutes, cap not exceeded); then replay the accepted
     minutes chronologically through the serial account (daily-loss stop can truncate). This yields a
     uniformly random feasible entry set per session.
   - **Null-M (matched):** identical, except the number of accepted minutes per session is capped at the
     *candidate strategy's* realized trade count in that session (0 where the candidate skipped the day).
     Optional stratified variant additionally matches the candidate's `time_bucket` counts per session.
     Null-M is the null a learned model must beat, because per-day exposure is itself a choice the model
     made on burned data.
   Side/offset/premium distributions are **not** matched — the deterministic selector already fixes those
   given the minute; matching them would smuggle the candidate's information into the null.
8. **Runs:** 1,000 seeds per null per fold (seeds 0–999). These are label-replay passes over ~3.6k
   decisions; cost is trivial. Use 10,000 if you want smoother tails.
9. **Threshold:** exact percentile reported, not just pass/fail. Empirical p-value
   `p = (1 + #{null runs with stressed total PnL ≥ candidate}) / (1 + N)`. Pass requires the candidate
   ≥ p95 of Null-M pooled out-of-fold; with the 4-cell model menu of Section 5, the corrected requirement
   is ≥ p99 pooled **or** ≥ p95 in ≥ 70% of folds. A learned model must additionally beat gate-only 1A on
   winsorized mean PnL/trade — beating random while losing to a config-only baseline is operationally
   worthless.
10. **Interpretation matrix:** in the cards below.

### Experiment 1A: gate-only strict serial

```text
Inputs:     protocol101-live-v1 decision rows; entry_filter=put_near_after_0940_vwap_m2_10;
            predictions = np.ones(len(labels)) per decision; threshold = 0.0;
            min_score_margin = 0; max_score_ceiling = 0.
Selection:  selection_mode=stable_abs_offset_20; tradability + affordability enforced.
Lifecycle:  policy 0 (stop35/target60/hold10m); cooldown 10; cap 3/session; daily loss $500;
            starting cash $10,000; $20/trade stress applied.
Metrics:    metrics_for_trades (stressed): total_pnl, PF, trades, win_rate, median_pnl,
            max_drawdown (and % of start), sessions_traded, positive_day_fraction,
            mean/median daily PnL; per split and pooled over Section-5 folds.
Pass:       stressed PnL > 0 AND PF ≥ 1.25 AND ≥ 20 trades, in each evaluation window,
            AND ≥ p95 of Null-U (1C). Pooled fold criteria per Section 5 for candidacy.
Fail means: see interpretation matrix below — 1A failing is informative, not terminal.
```

### Experiment 1B: gate parallel per-minute counterfactual

```text
Inputs:     same rows and gate as 1A.
Selection:  stable_abs_offset_20 per gated minute, tradability enforced, affordability ignored,
            no serial constraints. One value V_sel(t) per gated minute; minutes with no
            tradable band contract are excluded and counted (no_label_minutes).
Lifecycle:  policy 0 labels (V_sel = labels_net_pnl of the selected contract).
Metrics:    n_minutes, raw mean, winsorized mean (±$600), trimmed mean 10%, median,
            share_positive, sum; stratified per Section 2.
Pass:       n/a (diagnostic). The number that matters: winsorized mean V_sel per split/fold.
Fail means: n/a.
```

### Experiment 1C: random-in-gate nulls

```text
Experiment 1C-U / 1C-M: random-in-gate null
Inputs:     same rows, gate, selector, lifecycle, serial rules as 1A.
Sampling:   Null-U and Null-M as defined in answer 7; 1,000 seeds each.
Metrics:    distribution (mean, sd, p5/p25/p50/p75/p95/p99) of stressed total PnL, PF,
            mean/trade, trades; plus the candidate's exact percentile and empirical p-value.
Pass:       n/a for the null itself; consumers pass/fail against it per answer 9.
Fail means: n/a.
```

### Experiment 1D: time-slot schedule probe (secondary)

Purpose: first-come + cap 3 + cooldown 10 structurally samples the earliest gated minutes (~09:40–10:10 on
active days). If gate value varies by time of day, 1A's result confounds "gate edge" with "morning edge."

```text
Inputs:     as 1A.
Selection:  enter at the FIRST eligible minute inside each of three pre-registered windows:
            [09:40–10:30), [10:30–12:00), [13:00–15:00) ET; max 1 entry per window;
            all other serial rules unchanged.
Metrics:    as 1A, plus per-window PnL.
Pass:       n/a (probe). Divergence between 1A and 1D localizes time-of-day dependence.
Fail means: n/a.
```

### Experiment 1E: label-semantics and cost audit (checklist, half a day)

- Recompute `labels_net_pnl` for ≥ 50 sampled (session, minute, contract) triples directly from raw minute
  quotes (entry ask at t; walk bid path; apply stop35/target60/hold10; exit at bid) and assert equality.
- Assert label units are dollars per 1 contract at multiplier 100.
- Record whether fees are inside labels or represented only by the $20 stress (`VERIFY:` item 6).
- Count `no_label_minutes` (gated minutes with no tradable band contract) per split.

### Interpretation matrix for Experiment 1

| 1B winsorized mean | 1A stressed PnL | Reading | Next action |
|---|---|---|---|
| > 0 | > 0, ≥ p95 Null-U | Gate carries realizable edge; simple candidate shape exists | 1A becomes the reference every model must beat; proceed to Section 3/5 |
| > 0 | ≤ 0 or ≈ Null-U median | Signal exists per-minute but serial first-come sampling destroys it | Inspect 1D; if slot variant positive, redesign serial sampling as a pre-registered policy change (e.g., per-window budget), not a tuned patch |
| ≈ 0 or < 0 | any | The pocket has no realizable fair-contract edge at this cost model in this era | Do not train on this pocket. Re-run the two-stage opportunity scan over broader gates/sides/times with V_sel-style realizable values instead of oracle labels |
| > 0 val, ≤ 0 diag (and same pattern across folds) | mixed | Regime-bound gate | Gate needs regime conditioning or rejection; see Section 7, challenge 6 |

---

## 2. Apples-to-Apples Label Uplift Table

Answers: unit = **decision-minute**. Computed **before** account constraints (affordability, cap, daily
loss, cooldown are account-path properties; the uplift table measures signal). Multiple contracts per
minute are collapsed by the deterministic selector (that *is* the point): `V_sel(t)` = `labels_net_pnl`
(policy 0) of the `stable_abs_offset_20` selection among tradable band candidates at t; additionally
compute `V_band(t)` = median across all tradable puts with `10 ≤ |offset| ≤ 20` at t. Minutes with no
tradable band contract are excluded everywhere and counted.

**Populations** (same statistics on each, always over minutes):

1. `all_gated` — every gated minute.
2. `gate_only_selected` — minutes 1A entered.
3. `model_selected` — minutes the candidate model entered in strict serial replay.
4. `model_topk_parallel` — the model's top-k scored gated minutes per session ignoring serial rules,
   k = the model's serial trade count that session (isolates score skill from serial path).
5. `null_selected` — Null-M minutes, statistics averaged over seeds (report the seed-spread too).

**Statistics:** report all of {raw mean, winsorized mean at ±$600, trimmed mean 10%, median,
share_positive, sum, n}. **Primary = winsorized mean** (labels are heavy-tailed: single trades of +$2,010
and −$620 appear in a 37-row sample). Raw mean is retained because the Section-6 decomposition identity
requires it.

**Stratification:** split/fold × each of {`time_bucket`, `vwap_side`, `momentum15_side`,
`premium_bucket`, `range_bucket`, VIX tercile (tercile edges fixed from train folds only), month}.
Report a stratum only when n ≥ 15 minutes; pool the rest into `other`. These columns already exist in the
failure-diagnostic row schema — reuse `context_regime`.

**Uplift definition:** `uplift(pop) = stat(pop) − stat(all_gated)`, primary stat = winsorized mean.

**Proof the model adds value:** model_selected uplift > 0 pooled with a 1,000-draw bootstrap CI (resample
sessions, not minutes — minutes within a session are dependent) excluding 0, AND positive in ≥ 70% of
folds, AND above the p95 of Null-M's uplift distribution, AND ≥ gate_only_selected's uplift.

**Proof the model is selecting noise:** uplift inside the Null-M band; or sign flips across folds; or
uplift concentrated in a single month/stratum that vanishes when that stratum is held out; or
model_topk_parallel uplift ≈ 0 while model_selected uplift > 0 (that pattern means the "skill" lives in
serial path accidents, not scores).

**Output schema.**

`label_uplift_rows.csv`:

```text
fold_id, split, population, stratum_key, stratum_value, n_minutes,
mean_raw, mean_winsor600, mean_trim10, median, share_positive, sum_raw,
no_label_minutes_in_stratum
```

`label_uplift_summary.json`:

```text
{
  "fold_id": ..., "split": ...,
  "populations": {name: {pooled stats as above}},
  "uplift_vs_all_gated": {name: {stat, bootstrap_ci_low, bootstrap_ci_high, n_boot: 1000}},
  "null_m_uplift_distribution": {p5,p25,p50,p75,p95,p99},
  "model_percentile_vs_null_m": ...,
  "bootstrap_unit": "session",
  "winsor_bounds": [-600, 600],
  "vix_tercile_edges_from_train": [...],
  "guardrails": {"recorder_days_used": false, "thresholds_tuned_here": false}
}
```

---

## 3. Label Surgery — Exact Definitions, Ranked

Diagnosis being repaired (from Section 0, item 9): `decision_best_profit_regression` regresses on the max
over ~30 correlated option paths — an order statistic dominated by two-sided realized volatility, not
direction. `decision_profit_presence_classifier` inherits the same via "any contract > $20", and its
positive contract is on the wrong side of the trade roughly half the time out-of-sample. The per-contract
labels themselves are fine.

Answers to the numbered questions: yes, the label should be based on the deterministic selected contract
only (L1); yes, deterministic selection happens **before** labeling — that is the definition; label is per
decision-minute; labels are unconstrained by account state (constraints live only in replay); dollars are
primary (sizing is 1 contract and all gates are in dollars — log return-on-premium as a secondary
diagnostic column only); lifecycle leakage is not created by using future outcomes as *targets* — leakage
would be exposing exit-path information as runtime *features*, which the contract already forbids;
rejected/stale/wide contracts are **absent** from the label universe (the selector cannot pick them, so
they don't exist in the counterfactual; a gated minute with an empty tradable band is a no-label minute —
excluded from training, and replay naturally waits).

### L1 (primary): selector-consistent decision value

```text
y_reg(t)  = clip( labels_net_pnl[ selected(t), policy0 ], −600, +600 ) / 100
selected(t) = stable_abs_offset_20 over tradable band candidates at t (affordability ignored)
y_cls(t)  = 1{ labels_net_pnl[selected(t), policy0] > 25.0 }
```

- Hurdle $25 = $20 stress + $5 buffer, pre-registered. Expected base rate ~40–50% given diagnostic median
  −$60 (verify and record; if base rate leaves the 25–75% band on train folds, record it — do not move the
  hurdle to fix it).
- Clip ±$600 keeps continuity with the existing `target_clip`; policy-0 stop/target bounds most outcomes
  inside it anyway. Report a p2/p98-train-winsorized sensitivity column; do not select on it.
- This is the exact counterfactual strict replay pays. Model prediction = calibrated tradability/EV of the
  *policy*, not of a hindsight-best contract.

### L2 (second): band-median decision value

```text
y_reg(t) = clip( median over tradable puts with 10 ≤ |offset| ≤ 20 of labels_net_pnl[·, policy0], −600, +600 ) / 100
y_cls(t) = 1{ that median > 25.0 }
```

Lower-variance, less selector-coupled. Promote L2 over L1 only if L1 and L2 models materially disagree on
selected minutes (>20% symmetric difference in top-k sets) *and* L2 shows better fold consistency — that
pattern means residual selection variance is still leaking into L1.

### L3 (robustness variant, one only): band lower-quartile

`y(t) = p25 of the band values` (or equivalently HGB quantile loss α≈0.35 on L1). Pessimistic EV; run as a
single pre-registered cell, not a family.

**Ranking: L1 > L2 > L3.** Menu discipline: Section 5 allows four learned cells total; spend them as
{L1-cls, L1-reg, L2-cls, L3} unless Experiment 1 changes the picture.

---

## 4. Allowed-Feature Matrix for Experiment 2

Model rows become **one row per gated decision-minute** (no per-contract rows). Concretely, replace
`candidate_feature_vector` with a `decision_feature_vector(row)` = concat of `market_last`, `market_mean`,
`market_std`, `market_delta` (the existing 30-minute window aggregates), the put-conditioned subset of
`environment_prior_features`, and `decision_time_features` — roughly 46 dims. Family: HGB
(`sklearn_hist_gradient_boosting`) as primary; keep the MLP ensemble only if HGB shows signal.

| Feature category | In score? | In gate? | In label? | Reason |
|---|---|---|---|---|
| SPX close/returns/momentum (5m, 15m), window mean/std/delta | yes | yes (gate uses VWAP gap) | no | Vendor-stable to sub-point precision; through t−1m per contract timestamp policy |
| SPX VWAP distance | yes | yes | no | Same |
| OMAR / opening range | yes | allowed | no | Causal per `opening_context_policy` (no leading backfill) |
| Realized session range | yes | allowed | no | Vendor-stable |
| VIX level / change / window stats | yes | allowed (tercile gates with hysteresis only) | no | Vendor-stable at index level |
| Time of day (progress, sin/cos, buckets) | yes | yes (≥09:40, ≤15:30) | no | Deterministic |
| Day of week | one variant only | no | no | 128 train sessions ≈ 25 per weekday — overfit surface; isolate in a single pre-registered cell |
| Distance from open/close | yes (covered by progress features) | yes | no | Deterministic |
| Prior-day context | yes, if built causally | allowed | no | Must respect no-backfill policy |
| Option premium (ask/mid) | **no** (v1) | yes — band gates (`min_mid`/`max_mid`, `premium_gte_7_5` variants) | yes (it *is* the entry price) | Premium correlates with vol/moneyness the index features already carry; if ever added to score: 3-bucket {<7.5, 7.5–15, ≥15} on the band median with a $0.25 dead-zone and 2-minute debounce, one pre-registered cell |
| Option spread / spread_frac | no | yes — existing tradability thresholds, plus proposed t and t−1 debounce | yes (inside ask/bid accounting) | Proven jitter-fragile in score space (attempt107 spread scenarios) |
| Option bid/ask sizes | no | yes (min size ≥ 1) | no | Proven jitter-fragile |
| Vendor IV / greeks | no | no | no | Audit-only per contract policy |
| In-house repaired IV/greeks (shared pricer) | no (v1) | allowed | allowed | Legitimately causal and vendor-symmetric, but the July 1 case shows quote-instant skew still moves them; keep out of score until a debounced variant is separately justified |
| Option volume / open interest | no | no | no | Zeroed by the live contract; never proven live-equivalent |
| Strike offset / moneyness | no (constant at decision level: selector targets 20) | yes (band 10–20) | implicit | Selection is deterministic; count of tradable band contracts may be a gate, not a score input |
| Quote age | no | yes (`max_quote_age_seconds = 90`) | no | Feed property, not signal |

**How deterministic selection uses option data without score brittleness (your Q4.7):** selection touches
quotes only through pass/fail tradability thresholds; the ranking key is pure strike arithmetic
(offset vs ATM from the index price). Proposed hardening, versioned as a contract amendment
(`protocol101-live-v1.1`, applied identically in the historical builder, live builder, and replay — never
as a replay-side patch): a contract is entry-eligible at t only if it was tradable at both t and t−1m.
This removes single-minute threshold flickers, which is exactly the class of cross-vendor mismatch left
after aggregate scoring.

---

## 5. Walk-Forward Evaluation

**Fold template (target):** rolling window, train = 128 sessions → embargo = 5 sessions → test = 21
sessions; step = 21. Backfill `protocol101-live-v1` rows from Databento/ThetaData to 2023-01 (SPXW daily
expiries are fully live from mid-2022): ~2023-01 → 2026-03 ≈ 800 sessions → ~14 folds and several hundred
pooled out-of-fold trades. Minimum viable backfill: 2024-07 → 2026-03 ≈ 440 sessions → ~8 folds.

**Fallback with current data only** (Jul 2025 – Mar 2026 ≈ 185 sessions): train = 64 → embargo 5 →
test 21, step 21 → ~5 folds. Every artifact from this layout carries
`"evidence_grade": "provisional_evidence_only"` and cannot satisfy paper-readiness by itself. The reduced
train window is itself informative — attempt107 already shows 64-vs-128 `split_sign_flip`.

**Rolling vs anchored:** rolling primary (constant train size → comparable folds; adapts to regime).
Run anchored-expanding once as a sensitivity row; do not select on it.

**Hyperparameters:** frozen globally, pre-registered (HGB: max_iter 300, learning_rate 0.06,
max_leaf_nodes 31, l2_regularization 1.0, early stopping on a 15% tail of the fit set). No per-fold HP
search. Changing HPs later = new registry family.

**Thresholds without leaking:** inside each fold via the existing `train_tail20_calibration` mechanism
(fit on earlier train sessions, choose threshold on the last 20 train sessions) with the
`jitter_stability_stressed` rule. Test sessions never touch threshold selection. The old plateau rule's
habit of maximizing validation PnL is retired along with the burned split.

**Multiple-testing control:** menu of at most 4 learned cells (Section 3) + gate-only + nulls, all
registered before results. Selection criterion pre-registered: pooled stressed out-of-fold PnL subject to
(≥ p99 Null-M pooled or ≥ p95 in ≥ 70% of folds) AND ≥ gate-only on winsorized mean/trade AND PF ≥ 1.25
pooled AND ≥ 150 pooled trades (target layout) with positive expectancy in ≥ 60% of folds. If you want a
formal omnibus test, run White's reality check / SPA on daily PnL series across the menu; the percentile
correction above is the lightweight equivalent.

**Burned data:** the March 3–16, 2026 validation sessions (and the whole 137-attempt Q1 apparatus) are
permanently reclassified as report-only. No selection, thresholding, or early stopping may read them.

**Recorder days (your Q5.8–5.9):** June/July 2026 IBKR recorder days remain fully excluded from training,
calibration, and model selection. Once a candidate is frozen (model bytes + thresholds + config hashes in
the registry), paired replay runs **once** against them. Any modification after seeing the results creates
a new candidate requiring *future* recorder days. Keep the recorder running daily now — you will want ≥ 15
paired days including at least one high-VIX and one threshold-crossing day, and the calendar is the
bottleneck. In parallel, run Databento-vs-ThetaData paired replay across all historical folds as the
large-sample synchronization gate, and fit the empirical per-field delta distribution between the two
vendors to replace the hand-picked jitter magnitudes in `VENDOR_MICROSTRUCTURE_JITTER_SCENARIOS`.

---

## 6. Lifecycle / Exit Attribution

**Phase 1 — zero new data (run now).** For strategy S with selected minute set M_S, over a decision set D
(all gated minutes), define per minute: `V_sel` (policy-0 value of the selected contract), `V_band`
(band median), and `V_pol(p)` (policy-p value of the *same* selected contract, p ∈ {0,1,2} — already in
`labels_net_pnl`). Exact additive decomposition of raw mean trade value:

```text
mean_{M_S}(V_sel) = mean_D(V_sel)                                     [gate value G]
                  + [mean_{M_S}(V_band) − mean_D(V_band)]             [entry-timing uplift E]
                  + [ (mean_{M_S}(V_sel) − mean_{M_S}(V_band))
                      − (mean_D(V_sel) − mean_D(V_band)) ]            [selection delta C]
```

plus two side quantities: **lifecycle delta** `L = mean_{M_S}( V_pol(p*) − V_pol(0) )` where p* ∈ {1, 2}
is chosen once on train folds only and evaluated out-of-fold (never per-trade best-of-3 — that is
hindsight); and **serial sampling delta** `A = mean(V_sel over model_topk_parallel) − mean(V_sel over
serial-selected)`. Use raw means for the identity; report winsorized versions alongside as robustness.

**Output table** `lifecycle_attribution.csv`:

```text
fold_id, split, strategy, n_trades,
gate_value_G, entry_uplift_E, selection_delta_C, identity_check_mean_V_sel,
lifecycle_delta_L, alt_policy_used, serial_sampling_delta_A,
winsor_variants..., bootstrap_ci_E_low/high, bootstrap_ci_C_low/high (session bootstrap, 1000)
```

Decision rule: the component explaining ≥ 60% of the validation-vs-diagnostic (or cross-fold) gap with a
CI excluding zero becomes the next repair target. Prior from current evidence: E dominates and C is small
under deterministic selection — but that is exactly what should be measured, not assumed.

**Phase 2 — path stats (requires dataset-builder extension; build only if Phase 1 implicates lifecycle or
E2 entries pass).** Emit per selected trade: `exit_reason ∈ {stop, target, time}`, `minutes_held`,
`mfe_frac` (max bid ÷ entry ask − 1 over the full 10-minute window regardless of exit), `mae_frac`,
`time_to_mfe`, `bid_at_timeout`, `whipsaw_flag` (stopped out, then bid ≥ target level within the window).
Report per split × exit_reason: count, mean PnL, mfe/mae medians, whipsaw rate, share of time-exits
positive. These directly answer "losers = bad entries or bad exits" and "winners exited too early."

**Lifecycle grid timing (your Q6.3):** the 3-policy comparison is free and safe now. A broader grid only
after entry edge is proven, and then: ≤ 6 pre-registered policies total (the existing 3 plus at most
{stop35_target60_hold20m, time-only hold15m, trail-25%-after-+30%}), alternative chosen on train folds,
evaluated once out-of-fold, promoted only if it improves pooled expectancy by > $20/trade (one stress
unit) with consistent sign in ≥ 70% of folds. Registry entry per grid run. No per-fold policy switching.

---

## 7. Challenges to the Revised 10-Step Plan

The plan's ordering is right. Corrections and additions, in priority order:

1. **Add step 0: backfill history.** Both March-2026 splits together span one month (Section 0.7). With
   per-trade σ ≈ $300–500 and ~20 trades per split, the standard error on mean PnL/trade is $65–110 —
   larger than every effect you are trying to measure. No experiment in this plan is decisive on the
   current windows alone; several are decisive on 8–14 folds. Backfill is the highest-leverage task in the
   program and can run concurrently with Experiments 1A–1E on existing data.
2. **Pre-register the go/no-go tree before running Experiment 1** (use Section 1's interpretation matrix
   verbatim in the registry). Your step 4 ("if gate-only is positive, treat it as a possible candidate")
   is outcome-flexible as written — bind it: gate-only becomes the reference iff it passes 1A's criteria
   including ≥ p95 Null-U on pooled folds; otherwise the branch is closed regardless of how the point
   estimate looks.
3. **Name the mechanism being repaired in step 6.** The label rebuild isn't generic hygiene: the current
   decision targets are direction-blind (47%/41% side-consistency out-of-sample) while the policy
   hard-codes puts. L1's selector-consistency is the fix. Write that sentence into the registry so the
   next hill-climb doesn't reintroduce a max-over-contracts target under a new name.
4. **Retire the burned splits from selection explicitly** (plan step 7 implies training continues to
   report validation PnL on March 3–16 sessions — fine as reporting, forbidden as selection or
   early-stopping input).
5. **Models must beat gate-only, not just null** (missing from step 7's implied acceptance): the
   operational alternative to a model is the config-only baseline, and it is maximally vendor-stable.
6. **Gate survivorship is untested.** The VWAP pocket descends from the Protocol101 heritage discovered in
   this data era. The 2023–24 folds test the *gate*, not just the model. Pre-commit to the reading: if the
   gate's V_sel is flat-to-negative pre-2025 but positive after, it is regime-bound — the honest options
   are a pre-registered regime gate (coarse, hysteretic, index-level) or rejection, not quiet window
   selection.
7. **Add the tradability debounce as a versioned contract amendment** (Section 4) rather than leaving all
   sync hardening to the model layer — it is the cheapest remaining reducer of threshold-adjacent paired
   mismatches and must ship simultaneously to both builders and replay to preserve the causal game.
8. **Add Experiment 1E** (label/cost audit). Every downstream number inherits its answer, and it is half a
   day.
9. **Keep the failed-attempt registry discipline exactly as is** — attempts 108–137 as recorded negative
   evidence is what makes this program auditable. Extend the registry schema with
   `evidence_grade ∈ {provisional, fold_backed, confirmation_backed}` so provisional-fold results can't be
   quietly cited as readiness evidence later.
10. **Do not spend on learned exits, side-switching, or context-conditioned selection until E (entry
    uplift) and C (selection delta) from Section 6 are measured.** Attempts 133–137 already demonstrated
    that exposure-shaping without discrimination fails; Section 6 tells you where discrimination is
    actually missing.

### Pre-registered execution order

```text
0. Backfill live-v1 rows (2024-07 → 2026-03 minimum; 2023-01 target). Concurrent with 1.
1. Experiments 1E, 1A, 1B, 1C (current splits + provisional folds).      [config + null sampler]
2. Experiment 2 uplift tables for gate-only and attempts 130/131.        [reporting only]
3. Section 6 Phase-1 attribution for gate-only and attempts 130/131.     [zero new data]
4. Decision per Section 1 matrix. If any positive reference exists:
5. L1/L2/L3 training on walk-forward folds per Sections 3–5.             [4 cells max]
6. Section 6 attribution + Experiment 2 uplift on the new cells.
7. Freeze best cell if Section-5 criteria pass → paired DB-vs-TD historical sync gate →
   one-shot IBKR recorder confirmation → owner review. Paper-submit stays disabled throughout.
```

---

## Appendix: Implementation Hooks

- **1A is a config:** `simulate_model_policy(decisions, predictions=[np.ones(len(d.labels)) for d in decisions],
  threshold=0.0, cooldown_minutes=10, entry_filter="put_near_after_0940_vwap_m2_10",
  selection_mode="stable_abs_offset_20", max_trades_per_session=3, max_daily_loss=500.0, ...)`.
- **New code needed:** the Null-U/Null-M sampler (a variant of `simulate_baseline` that accepts an
  `entry_filter`, serial rules, per-session count caps, and the deterministic selector); the uplift-table
  script (extend `10_failure_diagnostic.py` — `context_regime`, bucket helpers, and the row schema already
  exist); the Section-6 Phase-1 script (reads `labels_net_pnl` across policy indices 0–2); fold
  orchestration around `07_training_runner.py` (the `train_tail20_calibration` fit mode is the per-fold
  threshold mechanism).
- **New label targets:** add `decision_selected_value_regression` (L1-reg), `decision_selected_value_classifier`
  (L1-cls, hurdle $25), `decision_band_median_*` (L2), `decision_band_q25` (L3) alongside the existing
  target modes in `06_supervised_pilot.py`; each consumes the deterministic selector at label-build time.
- **Registry naming:** `exp1a_gateonly_basegate_policy0_cap3_dl500`, `exp1c_nullM_<candidate>_seeds1000`,
  `exp2_uplift_<candidate>_<fold>`, `attempt_138+_L1cls_indexonly_stableabs20_jittergate_wf<fold>`, etc.
- **Guardrail block for every artifact:** `broker_endpoint_called: false`, `paper_submit_allowed: false`,
  `recorder_days_used_for_selection: false`, `thresholds_tuned_on_test: false`, `evidence_grade: ...`.