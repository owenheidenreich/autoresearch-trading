# Codex Adversarial Review — Path-D Phase-1 Close-Out, Fix Research, and Feasibility Study (2026-08-03)

## Bottom line

**Modify the recommendation.** Do not pivot to ES yet, and do not resume the tested one-minute aggressive
0DTE strategy. However, the proposed next step—test 30–60 minute options with passive entry on the owned
corpus—must be framed as an execution-aware feasibility study, not as the uniquely preferred strategy.
The passive headline is an optimistic touch model without queue data, while the feasibility study treats
its measured post-fill loss as if it were deterministic friction. ES remains a credible candidate, but its
spread is unmeasured. Both branches need a preregistered, distribution-aware comparison under executable
fill assumptions before either is trained.

The aggregate Phase-1 negative result is real. Several broader conclusions drawn from it are not.

## Independence and safety boundary

I recomputed from these source artifacts and raw partitions, not from Claude's result JSON/CSV or the
derived `reports/phase1_four_box/trajectory_outcomes.parquet`:

- `artifacts/entry_v2/oof_scores.parquet`
- `artifacts/entry_v2/oof_control_entries.parquet`
- `artifacts/entry_v2/evaluation_trajectory_index.parquet`
- the receipt paths named by that evaluation index
- `exit_features/session=*/<trajectory_id>.parquet`
- `exit_labels/session=*/<trajectory_id>.parquet`
- `artifacts/exit_v1/oof_predictions/fold=*/session=*/<trajectory_id>.parquet`
- sealed exit manifests/models under `artifacts/exit_v1/fold=*`
- raw OPRA `cbbo-1s`, GLBX ES `ohlcv-1m`, and XCBF VX `ohlcv-1m` partitions

No protected-holdout partition was opened, no model was fit, no market-data range was requested or
downloaded, and no broker/runtime/promotion state was touched. A read-only Databento `metadata.get_cost`
query was made for A1; it neither purchases nor retrieves data. The 36-session firewall remains closed for
this work (`holdout_open_count=0`).

## Verdict table

| Claim | Verdict | Independent result |
|---|---|---|
| C1 — negative before cost | **WEAKENED** | Aggregate bid-to-bid gross is **−$13.00**, median **−$90**, win **35.7%**, negative 5/5. Exact raw-quote mid-to-mid is still **−$12.12**, median **−$92.50**, win **36.0%**, negative 5/5. But the universal “no positive subpopulation” conclusion is false: 10:30–10:59 ET has **+$5.99** mean gross over 10,063 candidates/120 sessions, although net remains −$20.08 and fold 2 is negative. |
| C2 — friction dominates | **UPHELD** | Realized mean **−$39.48** decomposes exactly into bid-to-bid gross **−$13.00** plus **−$26.48** friction. Friction is **67.06%** of the realized loss and **4.683%** of mean $565.39 premium. |
| C3 — passive recovers 55% | **WEAKENED** | The optimistic offer-touch model reproduces exactly: **−$41.32 → −$18.56**, **+$22.76**, **83.26%** fill, adverse-selection contrast **−$0.72**, improvement positive 5/5. Requiring one-tick offer penetration gives **73.92%** fill and **−$28.76** conditional PnL, only **+$12.57** versus the all-entry baseline. Two-tick penetration gives **63.44%**, **−$39.01**, and **+$2.31**. Queue priority is unidentified. |
| C4 — zero ranking power | **UPHELD** | Replacing the uniform +$22.76 gift with each candidate's own quoted bid-entry saving leaves within-fold deciles non-monotonic: top **−$15.37**, bottom **−$16.32**, middle (decile 5) **−$10.79**. Gross Spearman by fold remains **+0.071/+0.031/+0.040/NaN/−0.009**. |
| C5 — learned exit is not a policy | **WEAKENED** | Counts reproduce exactly: index 0 in **980/1,031**, immediate-value identity in **982/1,031**. But this is not a serialization defect: sealed models reproduce stored predictions exactly, and utility recomputation has zero error. It is a degenerate, highly conservative learned policy, not an absent policy. |
| C6 — one-minute Path-D mathematically unclearable | **REFUTED** | **116.2%** is correct only for the median symmetric ±M screen ($26.48 / $20). A magnitude-weighted expected-value screen using mean \|move\|=$43.86 gives **80.2%**; one independent first window per trajectory gives median \|move\|=$50 and **76.48%**. One-minute aggressive entry remains unattractive, but “mathematically impossible” does not follow from win rate alone. |
| C7 — horizon dominates instrument | **UPHELD** | The within-option direction survives every attack: overlapping median screen is **116.2% at 1m vs 57.16% at 60m**; mean-move screen is **80.19% vs 53.88%**; first-window median screen is **76.48% vs 53.78%**. The separate 58.7% drift adjustment is not a valid general breakeven for an enter/wait option policy. |
| C8 — ES 15–60m is the only reachable cell | **REFUTED** | ES medians reproduce (**54.25/52.96/52.06%** at 15/30/60m with assumed $17 friction), but “only” is false. With the same expected-value/mean-move screen, aggressive options reach **55.51% at 30m and 53.88% at 60m**; using Claude's passive $18.56 constant gives **53.86% and 52.72%**. Even Claude's own median table rounds passive-option 60m to its declared reachable boundary, **55.0%**. ES friction remains unmeasured. |

## Detailed findings

### C1 and C2 — the decomposition is exact; the universal conclusion is not

For all 156,950 OOF candidate-label rows:

```text
gross_bid_to_bid = (exit_bid - entry_feature_bid) * 100
instant_friction = net_pnl_dollars - gross_bid_to_bid
```

The resulting means are −$13.00446, −$26.47942, and −$39.48388. Gross fold means are
−$18.59/−$14.00/−$11.63/−$9.26/−$11.33. This upholds the aggregate economics and C2's accounting.

For A7, I independently joined each candidate's `raw_symbol` and `exit_time_ns` back to the raw OPRA
CBBO-1s quote and used `(exit_bid + exit_ask)/2` at exit versus `entry_feature_mid`. The join covered
156,948/156,950 rows. Mid-to-mid gross is −$12.11594 overall and
−$17.90/−$13.85/−$10.98/−$7.57/−$9.93 by fold. Spread dynamics do not explain away C1.

The stronger close-out sentence—“no subpopulation is positive”—does not survive a local-time slice.
The 10:30 ET hour has +$5.99 gross, with fold means +$9.84/+7.56/−$17.29/+30.81/+6.41. Its median is
still −$60 and its after-cost mean is −$20.08. This is exploratory and multiplicity-exposed, not evidence
of a tradable edge, but it is enough to refute permanent closure of every conditional long-premium rule.

### C3 — exact reproduction of an upper bound, not an executable fill estimate

The headline reproduces only when `L` is the first post-entry `option_bid` in each of the 878 control exit
paths. At the first of 61 one-second states where `option_ask <= L`, an H=0 exit at that state's bid gives
−$18.56088 over 731 fills. The first-state aggressive mark is −$41.32005. The fill/no-fill aggressive
contrast is −$0.7194. These are exact reproductions.

The event is not a confirmed fill. CBBO has no queue position and no trade hitting the resting bid. I kept
the frozen fill law untouched and applied separate counterfactual stresses:

| Counterfactual event | Fill rate | Filled PnL | Improvement vs −$41.32 |
|---|---:|---:|---:|
| Offer touch: `ask <= L` | 83.26% | −$18.56 | +$22.76 |
| Touch persists for two seconds | 77.22% | −$19.78 | +$21.54 |
| Touch then one-tick continuation | 72.10% | −$21.21 | +$20.11 |
| Offer penetrates L by one tick | 73.92% | −$28.76 | +$12.57 |
| Offer penetrates L by two ticks | 63.44% | −$39.01 | +$2.31 |

The one-tick-penetration improvement remains positive in all five folds, so the mechanical spread saving
is real. Its magnitude is not identified. Real queue-aware PnL may lie outside this range because the
available files cannot identify whether or where the order was filled.

There is also a population issue: `DETERMINISTIC_CONTROL_OOF` is not an unbiased sample of all potential
orders. It is a deterministic, serially admissible selection from 900 control entries (878 evaluation
trajectories), while C4 covers 156,950 candidates. It is suitable for a controlled execution diagnostic,
not for estimating a universal passive cost constant.

### C4 — candidate-specific costing does not rescue the entry model

For every candidate I replaced the uniform correction with:

```text
candidate_bid_entry_net = net_pnl_dollars
                        + (fill_price - entry_feature_bid) * 100
```

That buys at the candidate's own displayed bid while retaining $3 round-trip fees. I formed deciles inside
each nonconstant fold, matching the original decile population (fold 3 is constant and therefore NaN).
The corrected bottom/middle/top values are −$16.32/−$10.79/−$15.37. The curve remains non-monotonic and
the top is not positive. C4 survives A5.

### C5 — the exit artifact is functioning exactly as serialized

The replay condition in `pathd_phase1_replay.py` reproduces 980 index-zero exits and 982 immediate-value
identities. Recomputing `mean_lcb90 + 0.25 * min(q10, 0)` gives zero numerical error. Reloading each of the
five sealed `model.joblib` artifacts through the verified manifest loader reproduces the first ten stored
predictions/actions of a trajectory per fold exactly (maximum absolute error 0).

The more useful diagnosis is calibration degeneracy:

- fold `mean_lcb90` offsets are −$813.64/−$895.13/−$454.53/−$646.37/−$638.01;
- mean utility at the first state is −$812.77;
- only 17.46% of first-state `a_ref_dollars` labels are positive;
- even among those positive-label rows, 171/180 exit at index zero.

Thus C5's behavior is upheld, but “not a policy” obscures the actual failure mode: a valid serialized
policy whose risk-lower-bound calibration overwhelms most point predictions. This does not invalidate the
four-box replay; it changes what should be debugged if the exit model is ever revisited.

### C6–C8 — cadence is important, but the published hurdle is not a feasibility theorem

For a sign strategy whose correctness probability is independent of move magnitude, the empirical
expected-value hurdle is:

```text
p_break_even = 0.5 + friction / (2 * mean(abs(move)))
```

Using the mean is magnitude-weighted and therefore uses the empirical payoff distribution rather than a
two-point median proxy. A genuinely distribution-aware hurdle for a real model requires the joint
distribution of its scores/actions and realized payoffs; no scalar win rate can supply it. Options add a
further mismatch: the proposed action is enter/wait, not a symmetric long/short sign bet.

| Instrument/entry | Horizon | Overlap median hurdle | Mean-payoff hurdle | First/nonoverlap median hurdle |
|---|---:|---:|---:|---:|
| SPXW aggressive | 1m | 116.20% | 80.19% | 76.48% first-window |
| SPXW aggressive | 15m | 66.55% | 57.85% | 64.71% nonoverlap |
| SPXW aggressive | 30m | 61.03% | 55.51% | 59.46% nonoverlap |
| SPXW aggressive | 60m | 57.16% | 53.88% | 55.95% nonoverlap; 53.78% first-window |
| ES, assumed $17 | 15m | 54.25% | 52.82% | 54.00% nonoverlap |
| ES, assumed $17 | 30m | 52.96% | 52.01% | 52.83% nonoverlap |
| ES, assumed $17 | 60m | 52.06% | 51.42% | 52.00% nonoverlap |

The option horizon conclusion is robust. The “mathematically unclearable” and “only reachable” language is
not.

### A1 — ES friction remains unmeasured and load-bearing

The owned corpus has GLBX ES OHLCV only; there are no local GLBX quote/MBP/TBBO partitions. The governed
acquisition plan explicitly lists ES/VX OHLCV as the purchased context and says other paid items stop for
authorization. An authenticated, read-only `metadata.get_cost` check for one regular session
(2026-07-30 13:30–20:00 UTC, `ES.FUT`) found the schemas available but priced, not included at zero cost:
`mbp-1` $0.896546, `tbbo` $0.822321, `bbo-1s` $0.072933, and `bbo-1m` $0.002260. No range request was made.
Thus a tiny quote sample is technically available, but acquiring even it still requires the owner's paid
data authorization under the project hard stop.

Assuming $4.50 commission plus 1–4 ES spread ticks ($12.50 each), the median hurdles are:

| ES spread | Friction | 15m | 30m | 60m |
|---:|---:|---:|---:|---:|
| 1 tick | $17.00 | 54.25% | 52.96% | 52.06% |
| 2 ticks | $29.50 | 57.38% | 55.13% | 53.58% |
| 3 ticks | $42.00 | 60.50% | 57.30% | 55.09% |
| 4 ticks | $54.50 | 63.63% | 59.48% | 56.61% |

This is why C8 cannot support a pivot before quote measurement at intended decision times.

### A2 — continuous-contract roll jumps do not contaminate the reported intraday windows

There are 254 nonempty ES session files and seven empty files (261 total), with four instrument-id changes:
2025-09-22, 2025-12-22, 2026-03-23, and 2026-06-19. The feasibility script differences only within each
daily file, so it never crosses an overnight/roll boundary. Excluding each change session plus its adjacent
sessions leaves the overlapping median absolute move unchanged at $200/$287.50/$412.50 for 15/30/60m.
Roll gaps therefore do not understate the hurdle. The absence of explicit handling is untidy, not a source
of the published C8 advantage.

### A6 and A8 — trajectory selection and overlap materially change option medians

At 60 minutes, only 852/1,031 trajectories have a complete first window. Within those, learned-entry
trajectories have median first-window |move| $380 versus $340 for controls (mean $429.67 versus $399.33),
so entry selection changes the option population. The corpus has no materialized exit path for every
unentered candidate, so the all-options-at-all-times counterfactual cannot be measured without rebuilding
source paths.

Overlapping windows are not merely a confidence-count problem here; they reweight long trajectories and
change the estimate:

| Horizon | Overlap n / median | Nonoverlap n / median | One first window n / median |
|---:|---:|---:|---:|
| 1m | 12,015,567 / $20 | 200,311 / $20 | 1,031 / $50 |
| 15m | 11,149,527 / $80 | 12,783 / $90 | 1,031 / $160 |
| 30m | 10,221,627 / $120 | 5,914 / $140 | 1,031 / $230 |
| 60m | 8,423,424 / $185 | 2,482 / $222.50 | 852 / $350 |

The published `n` is not an effective sample size, and its median is a row-weighted path statistic rather
than an entry-level opportunity statistic. C7's direction survives; C6's exact 116.2% does not generalize.

Bid versus mid changes little for these path moves: overlapping mid medians at 1/15/30/60m are
$20/$85/$125/$185 versus bid $20/$80/$120/$185.

## Additional findings not in A1–A8

1. **The feasibility script is not source-independent.** It reads the derived four-box
   `trajectory_outcomes.parquet` only to recover identities, and writes to a hard-coded Claude scratch
   directory. The report's reproduction section names `feasibility_study.py`, while the committed file is
   `v4/research/feasibility_gross_expectancy.py`. The numbers can be rebuilt directly, as done here, but the
   published reproduction contract is not portable.
2. **−$18.56 is not a measured passive “friction” constant.** It is conditional H=0 PnL after an optimistic
   touch fill: fees, spread at the fill state, and adverse price movement are mixed together. Applying it
   as a deterministic round-trip cost to arbitrary 30–60 minute option moves double-counts or misaligns
   path effects. Passive feasibility must simulate order placement, fill/no-fill, fill time, and then the
   hold from the actual fill.
3. **The VX cells are invalid as coded.** In 87 nonempty VX files, 59,930 of 62,814 rows belong to duplicated
   timestamps across publishers. Advancing `h` rows is therefore usually not advancing `h` minutes. This
   does not affect the ES claim but invalidates the reported VX comparison until publishers are resolved
   and timestamps deduplicated.
4. **UTC hour slicing hid a conditional gross-positive region.** Correcting to America/New_York exposes the
   10:30 ET gross-positive slice described under C1. It is unstable and after-cost negative, but the
   close-out's permanent “no subpopulation” statement was stronger than its evidence.

## What could not be checked

- **Actual ES spread:** the quote schemas are available on a positive-cost basis, but measuring the spread
  requires an owner-authorized small acquisition at the proposed trade times. No quote rows were bought in
  this review.
- **Passive queue fills:** requires MBP/order-book data with trades or broker paper/live no-order evidence
  capable of estimating queue position, cancels, and fills. CBBO offer touches cannot answer it.
- **Unselected-option move distribution:** requires source reconstruction for the full candidate surface
  under a preregistered sampling rule; entered trajectories alone are conditioned.
- **A real distribution-aware strategy breakeven:** requires a frozen action rule/model so payoff magnitude,
  misses, no-trades, fill probability, and prediction confidence can be evaluated jointly. This review did
  not train or tune one.

## Recommendation after review

Keep the halt on the tested one-minute aggressive Path-D formulation and do not pivot to ES on the $17
assumption. Replace Claude's sequence with a symmetric feasibility gate:

1. define one entry-level, nonoverlapping opportunity statistic and an expected-PnL metric before results;
2. test 30–60 minute SPXW only with a separate fill/no-fill path model and report touch, penetration, and
   queue-uncertainty bounds—not a fixed −$18.56 friction haircut;
3. after explicit owner authorization, confirm ES executable spreads with the smallest useful paid quote
   sample, then compare ES and SPXW under the same sampling and expected-value rules;
4. train neither branch until that comparison is preregistered and clears its data/fill gate.

This **supports the caution**, **modifies the ordering**, and **overturns the claim that the current study
has identified a unique reachable instrument/cell**.

STOP_FOR_CLAUDE_VERIFICATION
