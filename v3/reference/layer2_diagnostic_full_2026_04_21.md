# Layer-2 Shared-Encoder — Full Diagnostic

**Date:** 2026-04-21
**Run under review:** [`v3/artifacts/layer2_shared_enc_fixedq_detach/`](../artifacts/layer2_shared_enc_fixedq_detach/)
**Headline:** PF 1.455, DD 36.9%, 275 trades / 0.917 trades-per-day over 5 chronological walk-forward folds (2024-12-19 → 2026-03-04).
**Verdict:** paper trading **not** unblocked. Three findings below, one of them a hard stop.

---

## 1. Executive summary

After the shared encoder + `--detach-side` architecture cleared the
plan's §2 bar (PF ≥ 1.122, DD ≤ 56.2%, TPD ∈ [0.8, 1.1]) with aggregate
PF 1.455 and DD 36.9%, three reality checks were run to qualify the
model for paper trading:

| # | Check | Finding | Verdict |
|---|---|---|---|
| 1 | Per-fold regime diagnostic | Edge concentrates in folds 3 & 4 (PF 2.10 and 1.80); fold 0 is underwater (PF 0.86). No single feature shifts >1σ; largest is `atm_iv` at 0.71σ. Side_conf MAGNITUDE drifts ~10× across folds — model-side instability, not feature drift. 94% of trades on NON-oracle bars. | ⚠ concern (not hard stop) |
| 2 | Random-direction ablation | Random call/put direction at the same entry gate produces **PF 1.116** mean over 10 seeds (min 0.85, max 1.32). Model's PF of 1.455 adds +0.34 over random. | **🛑 HARD STOP** |
| 3 | Slippage / spread stress | At $50 additional per-round-trip slippage, Layer-2 PF stays 1.332 while teacher baseline drops to 0.876. Edge over teacher holds at ~+0.47 PF across the $0–$50 grid. | ✓ pass |

**What this means in plain terms:**

- The direction head is *cosmetic*. A coin flip at the same entry bars
  already produces PF > 1.0. The side head helps (+0.34 PF), but it
  isn't where the edge lives.
- The edge LIVES in: the entry_score gate (fixed-quantile 0.60), the
  product-score ranking (entry × side_conf) selecting one bar per day,
  and the `teacher_if_triggered_else_put` direction rule.
- Costs don't kill the edge.
- **One historical regime (late-2024 / early-2025) broke this model.**
  If that regime recurs, paper trading loses money. That's the binding
  uncertainty.

## 2. Artifact under review

```
run directory:         v3/artifacts/layer2_shared_enc_fixedq_detach/
architecture:          Layer2SharedEncoder (trunk depth=2, hidden=128, dropout=0.10)
                       + detach_side=True
                       + two linear heads (entry, side)
entry target:          entry_value_rank  (oracle best forward PnL, rank 0-1)
side target:           time_stop_margin_raw  (call − put time-stop PnL, arcsinh'd)
direction rule:        teacher_if_triggered_else_put
calibration:           fixed_quantiles  entry=0.60  side=0.10
scoring:               product (entry_score × side_conf)
training:              5 chronological folds, 40 max epochs with patience 6,
                       Adam lr=3e-4, CosineAnnealing, batch 4096, weight_decay 1e-4
```

Full training and replay commands in
[`layer2_shared_encoder_diagnostics_2026_04_21.md`](layer2_shared_encoder_diagnostics_2026_04_21.md) §"Shipped artifact".
Code referenced as committed in `bc85b37` (neural baseline) and `7ea7e69`
(GPU parity).

## 3. Check 1 — Per-fold regime diagnostic

Source: [`v3/analysis/layer2_per_fold_diagnostic.py`](../analysis/layer2_per_fold_diagnostic.py).

### 3a. Per-fold trade metrics

| Fold | Test window | N trades | PF | DD% | Win% | Mean $ | Call% | TPD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2024-12-19 .. 2025-03-19 | 55 | **0.860** | 28.2 | 30.9% | **−76** | 10.9% | 0.917 |
| 1 | 2025-03-20 .. 2025-06-13 | 40 | 1.439 | 22.5 | 32.5% | +223 | 22.5% | 0.667 |
| 2 | 2025-06-16 .. 2025-09-10 | 60 | 1.038 | 27.6 | 38.3% | +18 | 1.7% | 1.000 |
| 3 | 2025-09-11 .. 2025-12-04 | 60 | **2.095** | 21.4 | 35.0% | **+614** | 1.7% | 1.000 |
| 4 | 2025-12-05 .. 2026-03-04 | 60 | **1.801** | 32.2 | 43.3% | +322 | 41.7% | 1.000 |
| **Σ** | | **275** | **1.455** | 36.9 | 36% | +225 | 15.3% | 0.917 |

Fold 0 is negative. Fold 2 is marginal. Folds 3 & 4 carry the aggregate
PF. Folds 3 & 4 alone generate **~$56K in net PnL**; fold 0 generates
−$4.2K.

### 3b. Direction-side decomposition per fold

This is the most revealing table in the diagnostic:

| Fold | Call n | Call WR | Call net $ | Put n | Put WR | Put net $ | |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 6 | 50.0% | **+$1,929** | 49 | **28.6%** | **−$6,128** | puts bled |
| 1 | 9 | 44.4% | +$204 | 31 | 29.0% | +$8,724 | |
| 2 | 1 | 100.0% | +$733 | 59 | 37.3% | +$343 | nearly flat |
| 3 | 1 | 0.0% | −$541 | 59 | 35.6% | **+$37,374** | puts caught downtrend |
| 4 | 25 | 52.0% | +$7,300 | 35 | 37.1% | +$12,040 | both sides winning |

Calls are a tiny fraction of trades in folds 0–3. The model's PnL is
driven by PUTS via the `teacher_if_triggered_else_put` fallback.

- **Fold 0:** 49 puts, 28.6% win rate, **−$6,128 net**. The put-fallback
  bled theta in a regime where SPX didn't drop enough to make the
  premiums pay for themselves. 10 of these lost trades hit the
  `$1,200` premium cap (i.e. full-premium-loss on 10 days).
- **Fold 3:** 59 puts, 35.6% win rate (virtually the same as fold 0!),
  but **+$37,374 net**. Same win rate, 14× the net PnL — the 35.6% of
  winners were MUCH bigger because SPX made real down-moves.

**The fold-0-vs-fold-3 delta is not selection accuracy. It's put
payoff asymmetry.** Fold 0 was a chop regime. Fold 3 was a sell-off
regime. The model defaults to puts ⇒ it's implicitly short the market.

### 3c. Fold 0's worst 10 trades

| Date | Bar | Dir | $ PnL |
|---|---:|---|---:|
| 2025-03-05 | 120 | put | −$1,164 |
| 2025-03-07 | 71 | put | −$1,152 |
| 2025-02-27 | 33 | put | −$1,148 |
| 2024-12-19 | 36 | put | −$1,140 |
| 2025-03-14 | 53 | put | −$1,129 |
| 2025-01-28 | 51 | put | −$1,126 |
| 2024-12-24 | 44 | put | −$1,124 |
| 2025-03-04 | 30 | put | −$1,114 |
| 2025-03-06 | 38 | **call** | −$1,107 |
| 2025-01-10 | 33 | put | −$1,076 |

9 of 10 worst are puts. All clustered at the `$1,200` premium cap
(Layer 0's Moderate-tier rail) — these are **full-premium losers**,
meaning the underlying moved AWAY from the put by enough to take the
contract to ~0 by session end.

### 3d. Fold 0's best 5 trades

| Date | Bar | Dir | $ PnL |
|---|---:|---|---:|
| 2025-02-21 | 48 | put | +$4,903 |
| 2025-02-26 | 43 | put | +$4,150 |
| 2024-12-30 | 44 | call | +$2,776 |
| 2025-01-31 | 40 | put | +$2,154 |
| 2025-01-06 | 39 | put | +$1,870 |

The wins exist but there aren't enough of them to cover 10+
full-premium losers.

### 3e. Feature-distribution comparison (losing fold vs winning folds)

Median values by fold on the eligible-bar universe of each test window:

| Feature | f0 | f1 | f2 | f3 | f4 |
|---|---:|---:|---:|---:|---:|
| `vix` (regime code) | −0.33 | **+0.33** | −0.33 | −0.33 | −0.33 |
| `atm_iv` | **0.107** | 0.126 | 0.064 | 0.079 | 0.079 |
| `iv_percentile` | 0.115 | 0.079 | 0.139 | 0.167 | 0.125 |
| `first15_range_pct` | 0.003 | 0.004 | 0.002 | 0.002 | 0.003 |
| `omar_range_pct` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| `sigma_pos` | −0.008 | +0.594 | +0.317 | +0.232 | +0.267 |
| `omar_retest_dist_norm` | 1.760 | 1.329 | 1.522 | 1.620 | 1.492 |
| `last10_range_over_omar` | 1.508 | 1.352 | 1.096 | 1.335 | 1.144 |

Losing fold 0 vs winning folds 3 & 4 (pooled-σ shift):

| Feature | loss median | win median | pooled σ | |shift|σ | flag |
|---|---:|---:|---:|---:|---|
| `atm_iv` | 0.107 | 0.079 | 0.040 | **0.710** | closest to hard stop |
| `last10_range_over_omar` | 1.508 | 1.243 | 1.115 | 0.238 | |
| `first15_range_pct` | 0.003 | 0.003 | 0.002 | 0.190 | |
| `sigma_pos` | −0.008 | 0.254 | 1.394 | 0.188 | |
| `iv_percentile` | 0.115 | 0.146 | 0.306 | 0.099 | |
| `omar_range_pct` | 0.001 | 0.001 | 0.001 | 0.089 | |
| `omar_retest_dist_norm` | 1.760 | 1.549 | 2.492 | 0.085 | |
| `vix` | −0.33 | −0.33 | 0.352 | 0.000 | |

**No single feature crosses the >1σ hard-stop bar.** The largest drift
is `atm_iv` — fold 0's implied vol was ~36% higher than winning folds
(0.107 vs 0.079). That's consistent with the put-fallback losing money:
high IV means expensive puts, and you need bigger down-moves to make
them pay.

Secondary clue: `last10_range_over_omar` (0.24σ shift) and
`first15_range_pct` (0.19σ). Fold 0 had slightly wider intraday ranges
on average — more chop relative to the opening-minute volatility unit.

### 3f. Model-side drift — the subtler finding

| Fold | side_score median | side_conf median | side_conf std |
|---:|---:|---:|---:|
| 0 | **+0.103** | **+0.112** | 0.128 |
| 1 | −0.027 | +0.049 | 0.135 |
| 2 | +1.094 | +1.094 | 0.177 |
| 3 | **+1.183** | **+1.183** | 0.214 |
| 4 | +0.948 | +0.948 | 0.210 |

The model's confidence MAGNITUDE drifts 10× across folds. Fold 0 trains
on 666 train days, fold 4 on 906 days. Each fresh fold model learns a
different internal scale for `side_conf`. Since `fixed_quantiles`
calibrates thresholds per-fold, the gate ADAPTS — but the fact that
the model's internal representation isn't stable across training
configurations is a yellow flag.

### 3g. Oracle outcome mix

| Fold | empty | abstention | entered_right | side_error |
|---:|---:|---:|---:|---:|
| 0 | 52 (94.5%) | 1 (1.8%) | 0 | 2 (3.6%) |
| 1 | 40 (100.0%) | 0 | 0 | 0 |
| 2 | 59 (98.3%) | 1 (1.7%) | 0 | 0 |
| 3 | 58 (96.7%) | 2 (3.3%) | 0 | 0 |
| 4 | 56 (93.3%) | 2 (3.3%) | 1 (1.7%) | 1 (1.7%) |

**94%+ of chosen trades are NOT at an oracle bar.** The model isn't
finding what the opportunity oracle thinks is the best bar — it's
finding a different kind of bar. The PF is real, but the mechanism is
not "Layer-2 learns the oracle surface."

### 3h. Check 1 verdict

- No hard-stop on feature distribution (largest shift 0.71σ, below
  the 1σ threshold).
- Concern: model-side drift (side_conf scale factor 10× across folds).
- Concern: 94% of chosen trades on non-oracle bars. PnL is real but
  disconnected from the oracle label it was trained to predict.
- Mechanism identified: fold 0 vs fold 3 differ in put-payoff
  asymmetry, not selection accuracy. It's a sell-off-vs-chop regime
  difference carried by the put-fallback default.

## 4. Check 2 — Random-direction ablation (the hard stop)

Source: [`v3/analysis/layer2_random_direction_ablation.py`](../analysis/layer2_random_direction_ablation.py).

### 4a. Method

For each of 10 random seeds, take the exact entry bars the model chose
(275 trades), replace the model's direction with a 50/50 coin flip
per trade, and recompute PnL via the same
`compute_time_stop_pnl_for_direction` used in replay.py.

### 4b. Raw 10-seed results

| Seed | N | PF | DD% | Mean $ | Call | Put |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 269 | 1.105 | 59.3 | +50 | 147 | 128 |
| 102 | 267 | 1.053 | 115.1 | +28 | 138 | 137 |
| 103 | 267 | 1.222 | 54.5 | +112 | 141 | 134 |
| 104 | 268 | **0.851** | **100.0** | **−86** | 138 | 137 |
| 105 | 265 | 1.239 | 53.1 | +119 | 128 | 147 |
| 106 | 269 | 1.032 | 65.6 | +17 | 135 | 140 |
| 107 | 268 | 1.234 | 39.9 | +117 | 133 | 142 |
| 108 | 267 | 0.847 | 100.1 | −82 | 140 | 135 |
| 109 | 272 | 1.258 | 75.2 | +125 | 138 | 137 |
| 110 | 266 | 1.316 | 38.1 | +160 | 144 | 131 |

```
PF:        mean=1.116  std=0.168  min=0.847  max=1.316
DD%:       mean=70.1   std=26.8   min=38.1   max=115.1
mean_pnl:  mean=+56.0  std=86.8
```

### 4c. Interpretation

- **8 of 10 seeds** produce PF ≥ 1.0. The entry gate alone, with
  random direction, is profitable in most random slices.
- Mean PF 1.116 > 1.0 — plan §2b hard-stop criterion triggered.
- Model PF 1.455 adds +0.34 PF over random, meaningful but secondary.
- DD volatility is high (38–115% range across seeds) — coin-flip
  direction introduces high path-risk, which the model's direction
  head helps to smooth (model DD = 36.9%, near the best random seed).

**What this means for the claim:** "Layer-2 is a directional model"
is overstated. The honest description is:

> An entry-selection model on W2a + teacher context features,
> combined with a teacher-conditioned direction rule that falls back
> to puts. The side head adds modest lift (+0.34 PF) and tightens
> drawdown variance.

This is a real edge, just a narrower one than the multitask framing
suggested. The side head isn't broken, but it isn't the lever.

## 5. Check 3 — Slippage / spread stress (pass)

Source: [`v3/analysis/layer2_slippage_stress.py`](../analysis/layer2_slippage_stress.py).

### 5a. Grid results

Additional $ per round-trip applied uniformly to every trade:

| Slip $/RT | L2 PF | L2 DD% | L2 mean $ | L2 net $ | Teach PF | Teach mean $ | Edge PF | Edge net $ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  0 | **1.455** | 36.9 | +225 | +61,979 | 0.961 | −21 | **+0.494** | +68,275 |
| 10 | 1.429 | 38.9 | +215 | +59,229 | 0.943 | −31 | +0.486 | +68,505 |
| 25 | 1.392 | 43.4 | +200 | +55,104 | 0.917 | −46 | +0.474 | +68,850 |
| 50 | 1.332 | 52.0 | +175 | +48,229 | 0.876 | −71 | +0.456 | +69,425 |

### 5b. Interpretation

- Layer-2 stays profitable (PF > 1.0) at $50 additional slippage per
  round trip. That's aggressive — typical 0DTE mid-of-day slippage
  is $5–$15 per RT, $50 is worst-case vol-event/late-day.
- Teacher baseline goes NEGATIVE at $25+ slip. Its thin edge is fragile.
- **Edge over teacher is remarkably stable at ~+0.47 PF** across the
  entire stress grid. The net-dollar edge (Layer-2 net − Teacher net)
  is +$68–69K across all slip levels — because adding $slip to every
  trade affects both systems equally.
- DD grows (37% → 52%) as costs bite, but never blows up.

### 5c. Check 3 verdict

Pass. Realistic SPX 0DTE execution costs do not kill the edge. This
rules out "the model is an artifact of optimistic fill assumptions"
as a failure mode.

## 6. Cross-check synthesis

Where the three checks converge:

1. **There's a real edge, and it's mechanical (the gate), not learned
   (the side head).** Random-direction at the gate produces PF > 1
   (Check 2). The gate is a fixed-quantile `entry_score >= 0.60`
   filter plus a product-ranking top-1 selection per day — a
   deterministic rule, not a learned decision.
2. **The put-fallback is the dominant direction mechanism.** 85%+ of
   trades are puts in folds 0–3 (Check 1 §3b). The model is
   effectively short the market most of the time, via the
   `teacher_if_triggered_else_put` default.
3. **The regime blocker is sell-off-vs-chop.** Fold 0 lost on puts
   in a chop regime (same 28.6% put WR as fold 3's 35.6%, but
   $-6K vs $+37K net — win size differs). Fold 3 profited on puts
   in a sell-off regime. The distinction isn't in any single feature
   at >1σ — but `atm_iv` (0.71σ) and `last10_range_over_omar`
   (0.24σ) both move in the direction you'd expect for a chop regime.
4. **The side head's role is variance reduction, not direction
   accuracy.** Model DD = 36.9% is near the BEST random-seed DD
   (38.1%) from Check 2. The side_conf filter on top of the gate
   is keeping the model out of high-variance trades, even if its
   sign isn't where the edge comes from.
5. **Costs aren't the blocker.** Check 3 rules this out.

## 7. Mechanism hypotheses

What's happening under the hood (ordered by plausibility):

### 7a. The entry_score is a volatility / regime detector

The `entry_value_rank` target is "rank of the best forward PnL among
all oracle-eligible bars that day." Days with clear directional moves
produce high entry_value_rank scores; chop days produce low ones. The
model learns to identify bars where meaningful moves start.

Random direction at such bars still produces PF > 1 because: when the
market moves enough to make SOME direction profitable, you capture
50% of that move on average. When the market doesn't move, you lose
half the premium anyway. The EDGE is that you only enter when the
market is likely to move MORE than it normally would.

### 7b. The put-fallback is a net-short-premium policy in disguise

`direction_mode=teacher_if_triggered_else_put` defaults to put when
no teacher triggers. In folds 0–3, 85%+ of chosen trades have no
teacher trigger, so they default to put. The model is effectively
running "short SPX via long-premium puts, gated to high-move-potential
bars." This is a regime-dependent strategy — it wins when markets drop
more than implied, loses when they chop or rise more than implied.

Fold 0 vs fold 3 is exactly this asymmetry playing out.

### 7c. The 10× side_conf drift is a training instability, not a
feature

Each fold trains a fresh model on progressively more data. The
shared encoder + detach-side architecture has NO norm on the side
head's output (no sigmoid, no layer norm — raw linear projection).
As the training distribution shifts over folds, the side head's
output scale shifts with it. This is an MLP quirk, not a market
signal.

The `fixed_quantiles` calibration masks the problem at replay time
by taking the 10th percentile of the side_conf distribution FROM
THE VALIDATION SET — so the effective threshold adapts. But the fact
that the same architecture produces such different numerical scales
across folds is a warning about trusting the model's internal signal
beyond its rank-ordering.

## 8. What paper trading would actually bet on

If we paper-traded the `--detach-side` model as-shipped today, the
capital-at-risk is staking:

1. **The entry gate's ranking stays informative in live conditions.**
   The gate is deterministic given the features; it works as long as
   feature distributions stay comparable to training.
2. **The `teacher_if_triggered_else_put` fallback continues to match
   the market.** This is the dominant direction in the PnL and it's
   effectively "short SPX via 0DTE puts with a premium cap."
3. **The 2025-Q1-style regime does not recur.** If it does, we lose
   money on puts in chop. That's the biggest single risk.
4. **Costs stay under $50 per round trip.** Realistic for body-of-day
   SPX 0DTE.

Risk 3 is uncapped in current design. Risks 1, 2, 4 are tolerable.

## 9. Paths forward

Three options, ordered by effort. Not chosen here — user's call.

### Option A — simplify the architecture (lowest effort)

Drop the side head entirely. Use:

- `entry_score` from a single-head entry model
- Fixed-quantile gate at 0.60
- `teacher_if_triggered_else_put` direction rule
- Top-1 per day by entry_score alone (no product ranking)

Hypothesis: the simpler system hits PF ~1.15 (close to the random-
direction baseline of 1.116) with:

- Less model complexity
- No 10× side_conf drift to worry about
- Same entry mechanism, same risk exposure
- Clearer narrative for paper trading ("entry-selection + put fallback")

Cost: ~1 day of work. Worth doing regardless; establishes the
simpler baseline and quantifies what the side head actually adds.

### Option B — diagnose the fold-0 regime (medium effort)

Understand WHY 2024-12-2025-03 broke the model. Candidate
investigations:

1. Vol-regime dissection: higher `atm_iv` (0.107 vs 0.079 winning)
   means expensive puts. Does a simple "flatten when atm_iv > 0.10"
   rule turn fold 0 from −$4.2K into closer to break-even?
2. Trend-regime dissection: SPX 20-day return during fold 0 vs fold
   3. If fold 0 is a chop/uptrend and fold 3 is a sell-off, the
   fallback is essentially trend-following via short delta.
3. Event density: did fold 0 have more FOMC / CPI / NFP days?
   Those compress daily ranges around events and expand them after,
   which can hurt 0DTE time-stop exits.

Cost: 1–2 days. Output: a specific go/no-go rule for live deployment.

### Option C — regime-conditional gating (highest effort)

Extend Layer 0 with a live regime detector that flattens exposure
when a fold-0-like state is detected. Feature candidates from §3e:
high `atm_iv`, low sigma_pos magnitude (chop), elevated
`last10_range_over_omar`.

Cost: 3–5 days plus validation. Introduces new rail logic that
needs its own audit.

### Option D — live paper trading anyway

Not recommended given Check 2's hard stop and fold 0's $−4.2K loss.
But the alternative framing is: fold 0's $−4.2K is 17% of a $25K
account at max drawdown — uncomfortable but not catastrophic. A
paper account could absorb it, and a live fold-0 repeat would
generate information about the failure mode faster than in-sample
diagnostics.

Tradeoff: a month of paper trading tells you about this month's
regime only; it doesn't de-risk 2025-Q1 recurring.

## 10. What this diagnostic does NOT address

- **Tree baseline checks.** The tree fixed-quantile hybrid
  (`layer2_entry_side_fixedq_60_10`, PF 1.122 DD 56.2%) was not
  subjected to these three checks. Running them on the tree version
  would clarify whether:
  - The random-direction finding is a shared-encoder artifact or
    a Layer-2 architectural property (probably the latter).
  - The per-fold concentration pattern is the same.
  - The tree version is equally robust under slippage.
- **Fresh out-of-sample validation.** Data ends 2026-04-01; today is
  2026-04-21. 20 trading days of fresh data could be fetched and a
  6th pseudo-fold evaluated. Not free — requires rebuilding the v2
  sidecars and v3 Layer-2 export.
- **Transaction cost model realism.** Check 3 applies a fixed $/RT
  charge. A more realistic model would vary slippage by time of day,
  VIX, and contract premium.
- **Walk-forward adaptivity.** The fixed-quantile calibration
  responds to the side_conf drift across folds by design, but the
  entry threshold is also calibrated per-fold on the validation
  window. The `entry_quantile=0.60` is constant, but the
  entry_threshold in absolute terms drifts with the score
  distribution. Whether that's a bug or feature isn't addressed.

## 11. Artifacts

| File | Purpose |
|---|---|
| [v3/analysis/layer2_per_fold_diagnostic.py](../analysis/layer2_per_fold_diagnostic.py) | Check 1 script |
| [v3/analysis/layer2_random_direction_ablation.py](../analysis/layer2_random_direction_ablation.py) | Check 2 script |
| [v3/analysis/layer2_slippage_stress.py](../analysis/layer2_slippage_stress.py) | Check 3 script |
| [v3/reference/layer2_reality_checks_2026_04_21.md](layer2_reality_checks_2026_04_21.md) | Short-form verdict |
| This file | Full diagnostic |
| [v3/artifacts/layer2_shared_enc_fixedq_detach/layer2_trades.csv](../artifacts/layer2_shared_enc_fixedq_detach/layer2_trades.csv) | Per-trade detail |
| [v3/artifacts/layer2_shared_enc_fixedq_detach/replay_report.json](../artifacts/layer2_shared_enc_fixedq_detach/replay_report.json) | Aggregate replay result |

## 12. Commit history

```
8f21647  v3 Layer-2 reality checks: direction head is cosmetic; paper trading paused
7ea7e69  v3 Layer-2 GPU parity: detach-side trains cleanly on H100
bc85b37  v3 Layer-2 neural: detach-side is the new baseline (PF 1.455, DD 36.9%)
2029488  v3 Layer-2: package + shared-encoder multitask (partial pass)
```

## 13. One-sentence summary

**The Layer-2 shared-encoder model has a real edge — an entry-selection
gate + put-fallback that's profitable 4 of 5 folds and cost-robust —
but it's not a directional model, and the one fold it loses in
(2024-12 to 2025-03) reveals a put-fallback-in-chop regime risk that
paper trading would be unhedged against.**
